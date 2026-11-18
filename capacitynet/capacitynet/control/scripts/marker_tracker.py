#!/usr/bin/env python3
"""Track one fiducial marker's pose: smooth incoming detections, transform
the result into another frame, and build a gripper-facing target pose offset
along the marker's own Z axis. Same marker-tracking + TF logic needed by any
node that grasps or scores against a tag (grasp.py, release.py, ...) — pulled
out here so it isn't reimplemented per node.

Detections are averaged in a *fixed* frame (`fixed_frame`, e.g. `odom`) and
re-projected into the caller's `target_frame` (the arm's planning frame,
which rides on the mobile base) at the time of each `transform_to()` call.
The markers are static in the world, so a pose remembered in `odom` stays
valid however far the base drives after the tag was last seen — the
occlusion-means-stop rule this replaced came from averaging directly in the
base frame, where a cached pose went wrong by exactly the distance driven
(0.45 m in run_20260910_182418/ep3, then a 29 s stall in
run_20260914_153800 as the base was held for every occlusion).

Needs a tf2_ros.Buffer (already populated by a TransformListener in the
owning node) but no Node/topic access of its own: the owning node feeds it
messages via `on_markers()` and reads `pose`/`transform_to()`/`offset_along_z()`.
"""

from collections import deque

import rclpy
import tf2_ros
from geometry_msgs.msg import Pose, PoseStamped
from tf2_geometry_msgs import do_transform_pose_stamped

from .utils_geom import quat_to_axis, quat_mul


def select_marker(msg, target_marker_id: int):
    """Return the FiducialMarker matching target_marker_id from `msg`
    (or the first one if target_marker_id < 0), or None if it isn't
    present — e.g. an empty array, which a camera keeps publishing at its
    normal rate even while it sees no tag at all."""
    if not msg.marker:
        return None
    if target_marker_id < 0:
        return msg.marker[0]
    for m in msg.marker:
        if m.id == target_marker_id:
            return m
    return None


class CameraPriority:
    """Picks which of two marker-detecting cameras a *detection* should be
    accepted from: `primary` always wins whenever it actually sees the
    target marker; `secondary` only once `primary` hasn't seen it for
    longer than `timeout`. Keyed on detections, not messages, since a
    camera keeps publishing (empty arrays) at its normal rate even while
    the tag is out of view. Feed its verdict straight into whatever already
    consumes FiducialMarkerArray (e.g. MarkerTracker.on_markers) — the
    source frame_id travels with the message itself, so nothing downstream
    needs to know which camera won.
    """

    def __init__(self, timeout: float, clock):
        self._timeout = timeout
        self._clock = clock
        self._last_primary_seen = None

    def accept(self, is_primary: bool, has_detection: bool) -> bool:
        """Call once per incoming message, tagged with which camera it's
        from and whether it actually contains the target marker. Returns
        whether that message should be forwarded to the tracker."""
        now = self._clock.now()
        if is_primary:
            if has_detection:
                self._last_primary_seen = now
            return has_detection
        primary_stale = (self._last_primary_seen is None
                          or (now - self._last_primary_seen).nanoseconds / 1e9 > self._timeout)
        return primary_stale and has_detection


class MarkerTracker:

    def __init__(self, tf_buffer: tf2_ros.Buffer, target_frame: str, avg_window: int,
                 avg_max_jump: float, avg_reset_gap: float, logger=None,
                 fixed_frame: str = None):
        """
        Args:
            tf_buffer: populated tf2_ros.Buffer.
            target_frame: frame `transform_to()` re-projects the remembered
                pose into on every call, using the latest available TF
                (e.g. the arm's planning frame).
            fixed_frame: frame every detection is transformed into as it
                arrives (using the TF valid at that detection's own
                timestamp) before being folded into the running average, and
                in which `pose` is expressed. Must not move relative to the
                marker (an odometry/map frame, not a camera, end-effector or
                mobile-base frame) — otherwise samples from a moving camera
                would be averaged as if expressed in one frame, and a pose
                remembered through an occlusion would drift with the base.
                Defaults to target_frame (the pre-odom behaviour, where the
                remembered pose is only trustworthy while the base is still).
            avg_window: number of recent accepted detections averaged into `pose`.
            avg_max_jump: metres; a new sample farther than this from the
                current running average is dropped as a likely detection
                glitch rather than real marker motion.
            avg_reset_gap: seconds; if the marker hasn't been seen for longer
                than this, the buffer is stale and the new sample is accepted
                unconditionally as a fresh acquisition instead of being
                compared against it.
            logger: optional rclpy logger for info/warn messages.
        """
        self._tf_buffer = tf_buffer
        self._target_frame = target_frame
        self._fixed_frame = fixed_frame or target_frame
        self._avg_max_jump = avg_max_jump
        self._avg_reset_gap = avg_reset_gap
        self._logger = logger
        self._buffer = deque(maxlen=avg_window)
        # Latest (averaged) marker pose, in target_frame, or None.
        self._pose = None

    @property
    def pose(self) -> PoseStamped:
        """Latest smoothed marker pose, in fixed_frame, or None if no
        detection has been accepted yet.

        Deliberately *not* aged out: the marker is static in fixed_frame, so
        the last pre-occlusion detection stays valid (to odometry accuracy)
        for as long as the tag is out of view — and once the gripper closes
        over it, it is exactly what the masking/attach steps need. Use
        `transform_to()` for the pose in target_frame.
        """
        return self._pose

    def age_s(self, now_s: float):
        """Return seconds since the last *accepted* detection, or None if there is none.

        Measured on the detection's own header stamp, not on message
        arrival: a camera keeps publishing empty arrays at full rate while
        the tag is out of view, and `pose` keeps its last value, so neither
        of those says anything about freshness. `now_s` is the caller's
        clock in the same time base as the detection stamps.
        """
        if not self._buffer:
            return None
        return now_s - self._buffer[-1][2]

    def on_markers(self, msg, target_marker_id: int) -> None:
        """Feed one FiducialMarkerArray message. Selects `target_marker_id`
        (or the first marker in the array if target_marker_id < 0), transforms
        it into fixed_frame using the TF valid at the detection's own
        timestamp (not "now" — for a camera that moves, e.g. a wrist-mounted
        one, using the current TF would combine an old detection with the
        camera's current pose and make the result appear to move with the
        camera instead of staying fixed on the marker), and folds it into the
        running average described in __init__ if accepted. Updates `pose` in
        place; does nothing if the target marker isn't present, the frame
        transform at that timestamp isn't available, or the sample is
        rejected as a glitch.
        """
        selected = select_marker(msg, target_marker_id)
        if selected is None:
            return

        raw = PoseStamped()
        raw.header = selected.pose_cov_stamped.header
        raw.pose = selected.pose_cov_stamped.pose.pose
        stamp = raw.header.stamp.sec + raw.header.stamp.nanosec * 1e-9

        try:
            tf = self._tf_buffer.lookup_transform(
                self._fixed_frame, raw.header.frame_id,
                rclpy.time.Time.from_msg(raw.header.stamp)
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as ex:
            if self._logger is not None:
                self._logger.warn(
                    f"TF {raw.header.frame_id} -> {self._fixed_frame} at detection "
                    f"time unavailable, dropping detection: {ex}")
            return
        pose = do_transform_pose_stamped(raw, tf)
        pose.header.stamp = raw.header.stamp

        if self._buffer:
            last_stamp = self._buffer[-1][2]
            if stamp - last_stamp > self._avg_reset_gap:
                if self._logger is not None:
                    self._logger.info(
                        f"Marker reacquired after {stamp - last_stamp:.2f}s gap, "
                        f"resetting pose average")
                self._buffer.clear()

        if self._buffer:
            avg_x, avg_y, avg_z = self._average_position(self._buffer)
            p = pose.pose.position
            jump = ((p.x - avg_x) ** 2 + (p.y - avg_y) ** 2 + (p.z - avg_z) ** 2) ** 0.5
            if jump > self._avg_max_jump:
                if self._logger is not None:
                    self._logger.warn(
                        f"Marker detection jumped {jump:.3f}m from running average, "
                        f"discarding as a likely glitch")
                return

        self._buffer.append((
            (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z),
            (pose.pose.orientation.x, pose.pose.orientation.y,
             pose.pose.orientation.z, pose.pose.orientation.w),
            stamp,
        ))

        averaged = PoseStamped()
        averaged.header = pose.header  # latest frame_id + stamp
        ax, ay, az = self._average_position(self._buffer)
        aqx, aqy, aqz, aqw = self._average_quaternion(self._buffer)
        averaged.pose.position.x = ax
        averaged.pose.position.y = ay
        averaged.pose.position.z = az
        averaged.pose.orientation.x = aqx
        averaged.pose.orientation.y = aqy
        averaged.pose.orientation.z = aqz
        averaged.pose.orientation.w = aqw
        self._pose = averaged

    @staticmethod
    def _average_position(buffer):
        n = len(buffer)
        sx = sum(sample[0][0] for sample in buffer)
        sy = sum(sample[0][1] for sample in buffer)
        sz = sum(sample[0][2] for sample in buffer)
        return (sx / n, sy / n, sz / n)

    @staticmethod
    def _average_quaternion(buffer):
        """Chordal mean: q and -q represent the same rotation, so flip each
        sample into the same hemisphere as the first before averaging
        component-wise and renormalizing. Good enough for the small
        orientation variation expected between consecutive detections of the
        same marker (not a substitute for proper SLERP-based averaging across
        large rotations)."""
        ref = buffer[0][1]
        sx = sy = sz = sw = 0.0
        for _, q, _ in buffer:
            if q[0] * ref[0] + q[1] * ref[1] + q[2] * ref[2] + q[3] * ref[3] < 0.0:
                q = (-q[0], -q[1], -q[2], -q[3])
            sx += q[0]
            sy += q[1]
            sz += q[2]
            sw += q[3]
        n = len(buffer)
        x, y, z, w = sx / n, sy / n, sz / n, sw / n
        norm = (x * x + y * y + z * z + w * w) ** 0.5
        if norm == 0.0:
            norm = 1.0
        return (x / norm, y / norm, z / norm, w / norm)

    def transform_to(self, target_frame: str) -> PoseStamped:
        """Return the current smoothed `pose` re-projected into target_frame
        with the *latest* fixed_frame -> target_frame TF — i.e. where the
        static marker is relative to the base right now, however long ago
        it was last seen. Returns None if there's no pose yet, or if that TF
        is unavailable (logged, throttled by the caller's own cadence).
        The header stamp is the last detection's, so callers can still read
        "when was this measured" off it."""
        assert target_frame == self._target_frame, (
            f"MarkerTracker was constructed with target_frame={self._target_frame!r}, "
            f"cannot transform_to({target_frame!r})")
        if self._pose is None:
            return None
        if self._fixed_frame == self._target_frame:
            out = PoseStamped()
            out.header = self._pose.header
            out.pose = self._pose.pose
            return out
        try:
            tf = self._tf_buffer.lookup_transform(
                self._target_frame, self._fixed_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as ex:
            if self._logger is not None:
                self._logger.warn(
                    f"TF {self._fixed_frame} -> {self._target_frame} unavailable, "
                    f"cannot project marker pose: {ex}", throttle_duration_sec=2.0)
            return None
        out = do_transform_pose_stamped(self._pose, tf)
        out.header.stamp = self._pose.header.stamp
        return out

    @staticmethod
    def offset_along_z(marker_pose: PoseStamped, distance: float, axis_sign: float = 1.0,
                        rot_y = (0.0, 1.0, 0.0, 0.0), x_offset: float = 0.0) -> Pose:
        """Build a target Pose offset from marker_pose along its own Z axis.

        Gripper orientation = marker orientation rotated 180 degrees about its
        own Y axis (so gripper Z points into the tag, opposite the tag
        normal). Position = marker_pos + axis_sign * distance * marker_Z
        (+ x_offset * marker_X: unaffected by axis_sign, since rot_y already
        maps marker_X to the gripper/target frame's own Z axis — see
        ObjectReleaser._APPROACH_ROT_Y — so this is the axis that actually
        reads as "Z" against the published TF).
        """
        p = marker_pose.pose.position
        q = marker_pose.pose.orientation
        zx, zy, zz = quat_to_axis(q, 'z')
        xx, xy, xz = quat_to_axis(q, 'x')

        n = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) ** 0.5
        if n == 0.0:
            n = 1.0
        qx, qy, qz, qw = q.x / n, q.y / n, q.z / n, q.w / n

        d = axis_sign * distance
        target = Pose()
        target.position.x = p.x + d * zx + x_offset * xx
        target.position.y = p.y + d * zy + x_offset * xy
        target.position.z = p.z + d * zz + x_offset * xz

        gx, gy, gz, gw = quat_mul((qx, qy, qz, qw), rot_y)
        target.orientation.x = gx
        target.orientation.y = gy
        target.orientation.z = gz
        target.orientation.w = gw
        return target


    @staticmethod
    def rotation_pose(marker_pose: PoseStamped, rotation) -> Pose:
        """Build a Pose at marker_pose's position, with `rotation` applied
        to marker_pose's orientation in the marker's own local frame
        (result = marker_orientation * rotation).

        @param rotation: the rotation to apply, as a quaternion (x, y, z, w)
        """
        p = marker_pose.pose.position
        q = marker_pose.pose.orientation

        n = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) ** 0.5
        if n == 0.0:
            n = 1.0
        qx, qy, qz, qw = q.x / n, q.y / n, q.z / n, q.w / n

        gx, gy, gz, gw = quat_mul(
            (qx, qy, qz, qw), (rotation.x, rotation.y, rotation.z, rotation.w))

        result_pose = Pose()
        result_pose.position.x = p.x
        result_pose.position.y = p.y
        result_pose.position.z = p.z
        result_pose.orientation.x = gx
        result_pose.orientation.y = gy
        result_pose.orientation.z = gz
        result_pose.orientation.w = gw
        return result_pose
