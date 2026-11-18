#!/usr/bin/env python3
"""Where to score reachability: the workspace region, resolved from live ROS data.

What Q is scored over: the union of spheres around the MPC goal and one point
per configured `path_horizon_s` entry of the MPC predicted path — the pose the
end effector is expected to reach that many seconds from now — together
describing the approach corridor the arm actually traverses (by default: the
goal, plus the path points 2s and 1s out). The bare fiducial marker is *not*
that target: grasp.py offsets the tag by `approach_distance` (-0.2 m by
default) along the marker Z to build the pose it plans to, so scoring the tag
alone optimises around a point the arm never visits. The marker remains the
fallback when no MPC goal is available.

Before the MPC path exists (the arm's own planning takes 10-15 s after the
pregrasp order — run_20260914_110808/ep4), the corridor is stood in for by a
*synthetic path*: one sphere per configured horizon, placed where the real
horizon would be if the TCP left now along the straight line to the goal at
`synthetic_tcp_speed_mps` — i.e. `horizon_s * speed` from the end effector,
clamped at the goal. Same count and radii as the real horizons, so the region
has the same shape as the real one at t=0 of the plan, and the hand-over to
the real path is a small step rather than a rescale of Q.

Every center is transformed into the frame the voxel grid is published in, so
the region and the reachability map always live in the same space.
"""

from geometry_msgs.msg import Pose, PoseStamped
import math
from nav_msgs.msg import Path
import rclpy
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from ros2_markertracker_interfaces.msg import FiducialMarkerArray
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_pose_stamped
import tf2_ros
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

from capacitynet.control.scripts.grid_markers import build_region_markers
from capacitynet.control.scripts.marker_tracker import CameraPriority, select_marker
from capacitynet.control.components.node_component import NodeComponent


class WorkspaceRegionSource(NodeComponent):
    """Resolves the scored workspace region from MPC goal / path / fiducial marker.

    Owns everything about *where* to score: the topics, the TF lookups, the
    staleness policy, the radius, and the debug spheres. A node that only wants
    to look at the region — no inference, no motion — instantiates just this.
    """

    def __init__(self, node, prefix='', callback_group=None, tf_buffer=None):
        """Attach the region source to `node` and subscribe to its inputs.

        Args:
            node: host rclpy Node
            prefix: optional parameter/topic prefix ('' keeps flat names)
            callback_group: callback group for the subscriptions
            tf_buffer: reuse an existing tf2 Buffer instead of creating one
        """
        super().__init__(node, prefix, callback_group)

        # Kinect is the primary camera; the D405 only takes over once the
        # kinect hasn't reported a detection for camera_failover_timeout
        # seconds, and control returns to it as soon as it reports again.
        self.primary_marker_topic = self._declare(
            'primary_marker_topic', '/kinect/fiducial_markers')
        self.secondary_marker_topic = self._declare(
            'secondary_marker_topic', '/d405/fiducial_markers')
        self.camera_failover_timeout = float(
            self._declare('camera_failover_timeout', 1.0))
        self.target_marker_id = int(self._declare('target_marker_id', -1))
        # grasp.py republishes the offset grasp target on goal_topic at 10 Hz while
        # approaching, and the planner publishes the predicted path; together they
        # describe the corridor to score.
        self.goal_topic = self._declare(
            'goal_topic', '/curobo_trajectory_planner/mpc_goal')
        self.path_topic = self._declare('path_topic', '/mpc_predicted_path')
        # One path center per entry: the predicted end-effector position that
        # many seconds from now (not from the path's publish stamp), i.e.
        # where the arm will actually be when it reaches that part of the
        # corridor. Each pose in the path carries its own header.stamp; for
        # each horizon we pick the pose whose stamp is closest to
        # (now + horizon). If the horizon exceeds the path's time span, the
        # nearest-stamp search naturally clamps to the path's last pose.
        # Comma-separated list, e.g. "2.0,1.0" (the default: 2s and 1s
        # out) — first entry is the "primary" horizon used by
        # region_spheres_touching() for the arrival interlock, matching the
        # old single-horizon behavior. Empty string -> no path centers (goal
        # sphere only, matching the old path_tail_samples=0 behavior).
        self.path_horizons = self._parse_horizons(
            self._declare('path_horizon_s', '2.0,1.0'))
        # goal_topic carries a bare Pose with no header; assume this frame
        # (grasp.py always publishes it in the frame it passes to the action).
        self.planning_frame = self._declare('planning_frame', 'dsr01/base_link')
        self.marker_timeout = float(self._declare('marker_timeout', 2.0))
        self.mpc_timeout = float(self._declare('mpc_timeout', 2.0))
        self.workspace_radius = float(self._declare('workspace_radius', 0.10))
        # Radius of the path-tail spheres (region centers 1+), separate from
        # workspace_radius which sizes only the goal sphere (center 0). Defaults
        # to the same value so unconfigured setups score exactly as before this
        # param existed.
        self.path_radius = float(self._declare('path_radius', 0.05))
        # Synthetic path (see module docstring): assumed average TCP speed
        # along the straight end-effector -> goal line; each path_horizon_s
        # entry becomes a sphere horizon_s * speed from the end effector.
        # Only used while a goal is available and the real path is not.
        # 0 disables it (goal sphere alone until the path arrives).
        self.synthetic_tcp_speed = float(
            self._declare('synthetic_tcp_speed_mps', 0.06))
        self.ee_frame = self._declare('ee_frame', 'dsr01/link_6')
        # Arrival threshold for region_spheres_touching(): independent of
        # workspace_radius/path_radius (which size the Q-scoring spheres, not
        # "are we there yet") — the interlock fires once the path-horizon
        # point is within this distance of the goal, full stop.
        self.arrival_distance_threshold = float(
            self._declare('arrival_distance_threshold', 0.10))
        # Fallback region center, expressed in the voxel grid frame. Needed for bag
        # replay: no recorded bag contains /fiducial_markers. Kept as a string
        # ("x,y,z" or "[x, y, z]") so it passes identically through a launch
        # argument and a --ros-args -p override.
        self.static_workspace_center = self._parse_center(
            self._declare('static_workspace_center', ''))
        self.publish_debug = bool(self._declare('publish_debug_markers', True))

        # Latest fiducial marker pose (in its own camera frame), or None.
        self._marker_pose = None
        self._marker_stamp = None

        # Latest MPC goal (planning_frame) and predicted path (own frame): a
        # list of (stamp_seconds, (x, y, z)) samples, sorted by stamp, used to
        # look up the pose at path_horizon_s.
        self._mpc_goal = None
        self._mpc_goal_stamp = None
        self._path_samples = None
        self._path_frame = None
        self._path_stamp = None

        # Which source produced the last region; logged when it changes, because
        # switching source changes what Q measures and so rescales the gradient.
        self._source = None
        self._n_region_markers = 0
        # Whether the last resolved region's index 0 is a true anchor (mpc_goal
        # or marker — gets workspace_radius) or a path-tail point standing in
        # for it (gets path_radius like the rest) — see _resolve_centers and
        # _radii_for.
        self._region_has_goal = True
        # Whether the last resolved region's path centers are synthetic
        # (straight-line stand-ins, see _synthetic_path_centers). An arrival
        # cannot be measured on those — see _arrival().
        self._region_synthetic = False

        # Frame of the last resolved region, so region_spheres_touching() can
        # be polled by BaseCommander's own timer (10 Hz) without waiting on
        # the next inference cycle (~2 Hz) that resolve() is normally driven
        # by. Only the frame *name* is cached — region_spheres_touching()
        # re-resolves the centers fresh on every call (see its docstring).
        self._last_frame = None
        self._spheres_touching = False

        # Called when a region that *was* available goes stale. A controlling node
        # wires this to its stop command; a probe leaves it unset.
        self.on_lost = None
        # Called with (old, new) whenever the region source changes (see
        # _note_source). Q and |grad Q| are not comparable across sources, so
        # a controlling node uses it to drop a convergence verdict reached on
        # the previous source; a probe leaves it unset.
        self.on_source_change = None

        self.tf_buffer = tf_buffer if tf_buffer is not None else Buffer()
        self._tf_listener = (
            None if tf_buffer is not None
            else TransformListener(self.tf_buffer, node))

        self._setup_interfaces()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _setup_interfaces(self):
        node = self.node

        self._camera_priority = CameraPriority(
            self.camera_failover_timeout, node.get_clock())
        self.marker_sub_primary = node.create_subscription(
            FiducialMarkerArray, self.primary_marker_topic,
            lambda msg: self.on_markers(msg, True), 10,
            callback_group=self.callback_group)
        self.marker_sub_secondary = node.create_subscription(
            FiducialMarkerArray, self.secondary_marker_topic,
            lambda msg: self.on_markers(msg, False), 10,
            callback_group=self.callback_group)

        # Only the newest goal/path matter — an older one describes a corridor the
        # arm has already left, so keep depth 1 rather than queueing.
        if self.goal_topic:
            self.goal_sub = node.create_subscription(
                Pose, self.goal_topic, self.on_mpc_goal, 1,
                callback_group=self.callback_group)
        if self.path_topic and self.path_horizons:
            path_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.path_sub = node.create_subscription(
                Path, self.path_topic, self.on_path, path_qos,
                callback_group=self.callback_group)

        self.marker_pub = None
        self.source_pub = None
        if self.publish_debug:
            self.marker_pub = node.create_publisher(
                MarkerArray, self._private('region_markers'), 10)
            # Latched: a tool subscribing mid-run still learns what Q covers.
            source_qos = QoSProfile(
                depth=1,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.source_pub = node.create_publisher(
                String, self._private('region_source'), source_qos)

    @staticmethod
    def _parse_center(raw):
        """Parse a 'x,y,z' / '[x, y, z]' workspace center. Empty string -> None."""
        if not raw or not str(raw).strip():
            return None
        parts = [p for p in str(raw).replace('[', ' ').replace(']', ' ')
                 .replace(',', ' ').split() if p]
        if len(parts) != 3:
            raise ValueError(
                f'static_workspace_center must have 3 components, got {raw!r}')
        return tuple(float(p) for p in parts)

    @staticmethod
    def _parse_horizons(raw):
        """Parse a comma-separated list of path horizons, e.g. '2.0,1.0'.

        Order is preserved — the first entry is the "primary" horizon used
        by region_spheres_touching(). Empty string -> [] (no path centers).
        A negative entry is dropped rather than raising, so a single
        '-1.0' (the old disable-path convention) still works out to [].
        """
        if not raw or not str(raw).strip():
            return []
        return [h for h in (float(p) for p in str(raw).split(',') if p.strip())
                if h >= 0.0]

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def radius(self):
        """Region sphere radius in meters. Single source of truth for the package."""
        return self.workspace_radius

    @property
    def source(self):
        """Which input produced the last region.

        One of 'static' | 'mpc_goal+Npath' | 'path_only+N' | 'marker'.
        """
        return self._source

    @property
    def region_has_goal(self):
        """Whether the last resolved region's index 0 is a true mpc_goal center.

        False when the region was built from the path-horizon point alone (no
        fresh mpc_goal) — callers that assign workspace_radius to index 0 (see
        _radii_for, and GradientBasedController's own copy of that logic)
        need this to avoid treating a path point as the goal.
        """
        return self._region_has_goal

    def resolve(self, target_frame):
        """Workspace region in `target_frame` as a list of centers, or None.

        Priority: the explicit static override, then the MPC goal together with
        the path-horizon point of the predicted path, then the fiducial marker.
        Q is the mean over the *union* of the spheres, so both centers together
        describe the approach corridor rather than a single point.

        Also publishes the debug spheres and the region source, so a probe node
        writes no visualization code of its own.
        """
        centers = self._resolve_centers(target_frame)
        if centers is not None:
            self._last_frame = target_frame
            self._publish_debug(centers, target_frame)
        return centers

    def region_spheres_touching(self):
        """Whether the predicted end-effector position is within
        `arrival_distance_threshold` of the goal.

        Consulted by BaseCommander as a motion interlock, separate from the
        gradient-based convergence deadband. The arm's own *measured* TCP
        position was tried here first and rejected: it lags the MPC's plan by
        the controller's live tracking error, so it doesn't reflect "the plan
        is about to finish". The *primary* path-horizon point (the predicted
        pose `path_horizons[0]` seconds out — see WorkspaceRegionSource's
        module docstring) converges onto the goal as the plan nears
        completion instead, so its distance to the goal is read as that
        signal.

        This threshold is independent of workspace_radius/path_radius, which
        only size the Q-scoring spheres — conflating the two would make the
        arrival distance move whenever the scoring region is retuned.

        Only measurable with at least a goal and one path point (2+ centers);
        uses centers[0] (the goal) and centers[1] (the *primary*, i.e. first-
        configured, path horizon) and ignores any further path centers.
        Re-resolves the region fresh on every call (via `_resolve_centers`)
        instead of reusing `resolve()`'s cache, since this is polled at 10 Hz
        by BaseCommander's watchdog while `resolve()`'s own cache only
        refreshes once per ~1 Hz inference cycle.

        Returns:
            bool: False if the region has never been resolved yet (no
                `target_frame` known — otherwise gate() would never let the
                first resolve() cycle run at all). True (stop, by precaution)
                if no center at all is available — nothing to steer on.
                False with exactly one center (a goal or marker but no path
                point yet) or with a synthetic path (goal + straight-line
                stand-ins): the base servos on that region, so it starts as
                soon as the pregrasp order is out instead of waiting the
                10-15 s the arm's own planning can take
                (run_20260914_110808/ep4). Otherwise True iff the goal/path-
                horizon distance is within arrival_distance_threshold.

            This is the *gate* form. The precautionary True is not an
            arrival, so BaseCommander's arrival latch must read
            `region_arrived()` instead, which only reports a measured one.
        """
        touching, _ = self._arrival()
        return touching

    def region_arrived(self):
        """Measured form of `region_spheres_touching()` for the arrival latch.

        Returns:
            bool or None: True/False from the goal/path-horizon distance test
                when it can actually be made (2+ centers); None when it
                cannot (region never resolved, no center, goal only, or a
                synthetic path),
                so a caller that latches on True never latches on a
                precaution — which is exactly what pinned the base for a
                whole leg after one goal-only cycle in
                run_20260914_110808/ep2.
        """
        _, measured = self._arrival()
        return measured

    def _arrival(self):
        """Shared body of region_spheres_touching()/region_arrived().

        Returns:
            (gate, measured): `gate` is the bool described in
                region_spheres_touching(); `measured` is that same bool when
                it came from an actual distance test, else None.
        """
        if self._last_frame is None:
            return False, None

        centers = self._resolve_centers(self._last_frame)
        n_centers = 0 if centers is None else len(centers)
        distance = None
        if n_centers == 0:
            touching, measured = True, None
        elif n_centers == 1 or self._region_synthetic:
            # Goal alone, or goal + straight-line stand-ins: nothing here
            # says where the arm actually is in its plan, so keep steering
            # and never latch an arrival on it.
            touching, measured = False, None
        else:
            distance = math.dist(centers[0], centers[1])
            touching = distance <= self.arrival_distance_threshold
            measured = touching

        if touching != self._spheres_touching:
            self.log.info(
                f'{"Arrived" if touching else "Departed"}: goal/path-horizon '
                f'distance {"within" if touching else "beyond"} '
                f'arrival_distance_threshold'
                + (f' ({n_centers} center(s)'
                   f'{", synthetic" if self._region_synthetic else ""}, not measured)'
                   if distance is None else
                   f' (distance={distance:.3f}m, '
                   f'threshold={self.arrival_distance_threshold:.3f}m)'))
            self._spheres_touching = touching
        # info, not debug: a node-wide debug level also floods the log with
        # rcl/rclpy internals (e.g. "Subscription take succeeded"), so this
        # throttled trace has to be visible at the default level to be usable.
        if distance is not None:
            self.log.info(
                f'region_spheres_touching: distance={distance:.3f}m, '
                f'threshold={self.arrival_distance_threshold:.3f}m, touching={touching}',
                throttle_duration_sec=1.0)
        return touching, measured

    def _radii_for(self, centers):
        """Per-center sphere radii: workspace_radius for the goal (index 0),
        path_radius for every path-tail center. When the region has no true
        goal (path-tail-only — see region_has_goal), every center gets
        path_radius, including index 0.
        """
        if self._region_has_goal:
            return [self.workspace_radius] + [self.path_radius] * (len(centers) - 1)
        return [self.path_radius] * len(centers)

    def fits(self, centers, vg_info, extra_margin=0.0):
        """Check that every region sphere stays inside the grid, with `extra_margin` in x/y.

        VoxelMask.compute_mean returns 0.0 for an empty mask, so a sphere that
        leaves the grid scores as *zero reachability* rather than *unknown* and
        would fabricate a large gradient pointing away from the boundary. The
        scored centers are `center - offset`, so with an NxN candidate grid they
        range over +/- (grid_size//2)*delta in x and y — that is what
        `extra_margin` covers (the caller passes `delta * (grid_size // 2)`).
        Every center is checked: a partially observed corridor would bias Q just
        as badly as a partially observed goal.
        """
        origin = vg_info['origin']
        res = vg_info['resolution']
        lo = (origin.x, origin.y, origin.z)
        hi = (origin.x + vg_info['size_x'] * res,
              origin.y + vg_info['size_y'] * res,
              origin.z + vg_info['size_z'] * res)
        radii = self._radii_for(centers)

        for idx, (center, radius) in enumerate(zip(centers, radii)):
            # Z is never offset by the candidate grid, only X and Y.
            margins = (radius + extra_margin, radius + extra_margin, radius)
            for axis, (c, m) in enumerate(zip(center, margins)):
                if c - m < lo[axis] or c + m > hi[axis]:
                    which = 'goal' if idx == 0 else f'path point {idx}'
                    self.log.warn(
                        f'Workspace sphere ({which}) leaves the voxel grid on axis '
                        f'{"xyz"[axis]} (center={c:+.3f}, margin={m:.3f}, '
                        f'bounds=[{lo[axis]:+.3f}, {hi[axis]:+.3f}]) — skipping cycle',
                        throttle_duration_sec=5.0)
                    return False
        return True

    # ── Callbacks ────────────────────────────────────────────────────────────

    def on_markers(self, msg: FiducialMarkerArray, is_primary: bool):
        """Store the pose of the tracked marker (in its own camera frame),
        preferring the primary camera and falling back to the secondary once
        the primary has gone quiet — see CameraPriority."""
        selected = select_marker(msg, self.target_marker_id)
        if not self._camera_priority.accept(is_primary, selected is not None):
            return

        pose = PoseStamped()
        pose.header = selected.pose_cov_stamped.header
        pose.pose = selected.pose_cov_stamped.pose.pose
        self._marker_pose = pose
        self._marker_stamp = self.node.get_clock().now()

    def on_mpc_goal(self, msg: Pose):
        """Store the grasp target grasp.py is driving the arm to."""
        self._mpc_goal = (msg.position.x, msg.position.y, msg.position.z)
        self._mpc_goal_stamp = self.node.get_clock().now()

    def on_path(self, msg: Path):
        """Store the predicted path's timestamped poses, in the path's own frame."""
        if not msg.poses:
            return
        self._path_samples = [
            (rclpy.time.Time.from_msg(p.header.stamp).nanoseconds / 1e9,
             (p.pose.position.x, p.pose.position.y, p.pose.position.z))
            for p in msg.poses
        ]
        self._path_frame = msg.header.frame_id or self.planning_frame
        self._path_stamp = self.node.get_clock().now()

    # ── Resolution ───────────────────────────────────────────────────────────

    def _resolve_centers(self, target_frame):
        """Priority: static > mpc_goal(+path) > marker(+path) > path alone.

        The path tail joins whichever anchor is actually available (goal or
        marker) rather than being tied to the goal specifically — this lets
        a path-only publisher (e.g. mock_arm_trajectory, which never
        publishes mpc_goal) still combine with a live marker detection
        instead of being shadowed by it. Path-only (no anchor at all) stays
        the last resort, below the marker, so a continuously-republished
        mock path can never starve a real marker detection on its own.
        """
        if self.static_workspace_center is not None:
            self._region_has_goal = True
            self._note_source('static')
            return [self.static_workspace_center]

        path_centers = self._path_horizon_centers(target_frame)
        self._region_synthetic = False

        goal_centers = self._goal_center(target_frame)
        if goal_centers:
            self._region_has_goal = True
            if not path_centers:
                path_centers = self._synthetic_path_centers(goal_centers[0], target_frame)
                self._region_synthetic = bool(path_centers)
            kind = 'synth' if self._region_synthetic else 'path'
            self._note_source(f'mpc_goal+{len(path_centers)}{kind}')
            return goal_centers + path_centers

        marker_centers = self._marker_region(target_frame)
        if marker_centers:
            self._region_has_goal = True
            self._note_source(
                f'marker+{len(path_centers)}path' if path_centers else 'marker')
            return marker_centers + path_centers

        if path_centers:
            self._region_has_goal = False
            self._note_source(f'path_only+{len(path_centers)}')
            return path_centers

        return None

    def _note_source(self, source):
        """Log region-source changes: they rescale Q, and so |grad Q|."""
        if source != self._source:
            previous = self._source
            if previous is not None:
                self.log.info(
                    f'Workspace region source: {previous} -> {source} '
                    '(Q now covers a different region, |grad Q| rescales)')
            self._source = source
            if self.source_pub is not None:
                self.source_pub.publish(String(data=source))
            if previous is not None and self.on_source_change is not None:
                self.on_source_change(previous, source)

    def _goal_center(self, target_frame):
        """Single mpc_goal sphere, in target_frame. [] when absent/stale."""
        goal_fresh = (self._mpc_goal is not None
                      and self._age_s(self._mpc_goal_stamp) <= self.mpc_timeout)
        if not goal_fresh:
            return []
        return self._transform_positions(
            [self._mpc_goal], self.planning_frame, target_frame)

    def _synthetic_path_centers(self, goal, target_frame):
        """Straight-line stand-ins for the path horizons, in target_frame.

        One point per path_horizon_s entry, `horizon_s * synthetic_tcp_speed`
        from the end effector along the end-effector -> goal line (see
        module docstring), clamped at the goal so no point ever overshoots
        it. [] when disabled, when ee_frame is unavailable, or when the end
        effector already sits on the goal (no line to build).
        """
        if self.synthetic_tcp_speed <= 0.0 or not self.path_horizons:
            return []
        ee = self._transform_positions([(0.0, 0.0, 0.0)], self.ee_frame, target_frame)
        if not ee:
            return []
        gx, gy, gz = goal
        ex, ey, ez = ee[0]
        dx, dy, dz = gx - ex, gy - ey, gz - ez
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-3:
            return []
        ux, uy, uz = dx / dist, dy / dist, dz / dist
        centers = []
        for horizon_s in self.path_horizons:
            ahead = min(horizon_s * self.synthetic_tcp_speed, dist)
            centers.append((ex + ahead * ux, ey + ahead * uy, ez + ahead * uz))
        return centers

    def _path_horizon_centers(self, target_frame):
        """One sphere per configured path_horizon_s entry, in target_frame, in order.

        Independent of mpc_goal freshness: this is a standalone anchor check
        so the path can join whichever anchor (goal or marker) is actually
        available — see _resolve_centers.
        """
        centers = []
        for horizon_s in self.path_horizons:
            centers.extend(self._path_horizon_center_at(target_frame, horizon_s))
        return centers

    def _path_horizon_center_at(self, target_frame, horizon_s):
        """Single path-horizon sphere for one horizon, in target_frame. [] when absent/stale."""
        path_fresh = bool(self._path_samples) and self._age_s(self._path_stamp) <= self.mpc_timeout
        if not path_fresh:
            return []
        target_time = self.node.get_clock().now().nanoseconds / 1e9 + horizon_s
        _, position = min(self._path_samples, key=lambda s: abs(s[0] - target_time))
        return self._transform_positions(
            [position], self._path_frame, target_frame)

    def _marker_region(self, target_frame):
        """Single sphere on the fiducial marker. [] when unavailable."""
        if self._marker_pose is None:
            self.log.warn(
                'No workspace target yet (no MPC goal, no marker), not commanding base',
                throttle_duration_sec=5.0)
            return []

        age = self._age_s(self._marker_stamp)
        if age > self.marker_timeout:
            self.log.warn(f'Marker stale ({age:.1f}s), not commanding base',
                          throttle_duration_sec=5.0)
            if self.on_lost is not None:
                self.on_lost()
            return []

        center = self._transform_marker_to(target_frame)
        return [center] if center is not None else []

    def _transform_positions(self, positions, source_frame, target_frame):
        """Transform (x, y, z) tuples between frames with one TF lookup. [] on failure."""
        if source_frame == target_frame:
            return list(positions)

        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as ex:
            self.log.warn(
                f'TF {source_frame} -> {target_frame} unavailable: {ex}',
                throttle_duration_sec=5.0)
            return []

        transformed = []
        for x, y, z in positions:
            ps = PoseStamped()
            ps.header.frame_id = source_frame
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = z
            ps.pose.orientation.w = 1.0
            p = do_transform_pose_stamped(ps, tf).pose.position
            transformed.append((p.x, p.y, p.z))
        return transformed

    def _transform_marker_to(self, target_frame):
        """Transform the stored marker pose into target_frame. Returns (x,y,z) or None."""
        source_frame = self._marker_pose.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time()
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as ex:
            self.log.warn(
                f'TF {source_frame} -> {target_frame} unavailable: {ex}',
                throttle_duration_sec=5.0
            )
            return None

        transformed = do_transform_pose_stamped(self._marker_pose, tf)
        p = transformed.pose.position
        return (p.x, p.y, p.z)

    # ── Debug output ─────────────────────────────────────────────────────────

    def _publish_debug(self, centers, frame_id):
        if self.marker_pub is None:
            return
        array = build_region_markers(
            frame_id, self.node.get_clock().now().to_msg(),
            centers, self._radii_for(centers), n_stale=self._n_region_markers)
        self._n_region_markers = len(centers)
        self.marker_pub.publish(array)
