#!/usr/bin/env python3
"""Grasp an object whose pose is given by a fiducial (ArTag) marker.

Same marker-tracking + TF logic as brain.py, but instead of computing a
reachability score it performs a simple pick, driven by an explicit
event-driven state machine:

  idle --user_trigger--> act_move_pregrasp
  act_move_pregrasp --arm_done--> act_open_gripper
  act_open_gripper --gripper_open--> act_move_grasp
  act_move_grasp --arm_done--> act_close_gripper
  act_close_gripper --gripper_closed--> srv_mask_object
  srv_mask_object (set_mask, then attach_object) --object_masked--> act_move_backoff
  act_move_backoff --arm_done--> act_retract_arm
  act_retract_arm --arm_done--> idle
  (any act_*/srv_* state) --failure--> idle

`self._fsm.state` doubles as the busy flag: only `idle` accepts
`user_trigger`, so a trigger received mid-sequence is ignored. Adding a new
state later = one constant, one `_on_enter_*` handler, one entry in the
`handlers` dict passed to `StateMachine`, one or two entries in `TRANSITIONS`.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy

from tf2_ros import Buffer, TransformListener, TransformBroadcaster, Duration

from geometry_msgs.msg import PoseStamped, Pose, TransformStamped, Vector3
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from control_msgs.action import GripperCommand
from curobo_msgs.action import SendTrajectory
from curobo_msgs.srv import SetMask, AttachObject
from ros2_markertracker_interfaces.msg import FiducialMarkerArray
from heron_interface.action import TriggerGrasp

from ..robot.execution_divergence_watchdog import ExecutionDivergenceWatchdog
from ..core.marker_tracker import CameraPriority, MarkerTracker, select_marker
from ..robot.state_machine import StateMachine
from ..core.utils_geom import quat_to_axis, unit_vector_between, lift_pose, quat_mul
from ..robot.doosan import DoosanArm, DR_BASE, DR_TOOL

from rclpy.action import ActionServer, CancelResponse

# States (plain strings, not Enum: a single place to touch when adding one).
IDLE = 'idle'
ACT_MOVE_PREGRASP = 'act_move_pregrasp'
ACT_OPEN_GRIPPER = 'act_open_gripper'
SRV_MASK_OBJECT = 'srv_mask_object'
ACT_MOVE_GRASP = 'act_move_grasp'
ACT_CLOSE_GRIPPER = 'act_close_gripper'
ACT_MOVE_BACKOFF = 'act_move_backoff'
ACT_RETRACT_ARM = 'act_retract_arm'

_BUSY_STATES = (
    ACT_MOVE_PREGRASP, ACT_OPEN_GRIPPER, SRV_MASK_OBJECT,
    ACT_MOVE_GRASP, ACT_CLOSE_GRIPPER, ACT_MOVE_BACKOFF, ACT_RETRACT_ARM,
)

TRANSITIONS = {
    (IDLE, 'user_trigger'): ACT_MOVE_PREGRASP,
    (ACT_MOVE_PREGRASP, 'arm_done'): ACT_OPEN_GRIPPER,
    (ACT_OPEN_GRIPPER, 'gripper_open'): ACT_MOVE_GRASP,
    (ACT_MOVE_GRASP, 'arm_done'): ACT_CLOSE_GRIPPER,
    (ACT_CLOSE_GRIPPER, 'gripper_closed'): SRV_MASK_OBJECT,
    (SRV_MASK_OBJECT, 'object_masked'): ACT_MOVE_BACKOFF,
    (ACT_MOVE_BACKOFF, 'arm_done'): ACT_RETRACT_ARM,
    (ACT_RETRACT_ARM, 'arm_done'): IDLE,
    # Generic abort: any busy state can bail out back to idle.
    **{(s, 'failure'): IDLE for s in _BUSY_STATES},
}


class ObjectGrasper(Node):
    """Pick an ArTag-defined object via an explicit state machine."""

    # States driven by Doosan services rather than a cancellable action goal
    # — a cancel there is delivered as a MoveStop (see _on_grasp_cancel_requested).
    _DOOSAN_STATES = (ACT_MOVE_GRASP, ACT_MOVE_BACKOFF, ACT_RETRACT_ARM)

    def __init__(self):
        super().__init__('object_grasper')

        self._load_parameters()

        self._setup_interfaces()
        self._setup_state_machine()

        self.get_logger().info("Object grasper initialized")
        self.get_logger().info(
            f"  - Marker topics: primary={self.primary_marker_topic}, "
            f"secondary={self.secondary_marker_topic} "
            f"(failover after {self.camera_failover_timeout:.1f}s, id={self.target_marker_id})")
        self.get_logger().info(
            f"  - Marker pose averaging: window={self.marker_avg_window}, "
            f"max_jump={self.marker_avg_max_jump:.3f}m, reset_gap={self.marker_avg_reset_gap:.1f}s")
        self.get_logger().info(f"  - Trigger topic: {self.trigger_topic}")
        self.get_logger().info(f"  - Planning frame: {self.planning_frame}")
        self.get_logger().info(f"  - Execute action: {self.execute_action}")
        self.get_logger().info(
            f"  - MPC live-goal topic: {self.mpc_goal_topic} (rate={self.mpc_goal_rate:.1f}Hz)")
        self.get_logger().info(
            f"  - Divergence watchdog: {'enabled' if self.divergence_watchdog_enabled else 'disabled'} "
            f"(threshold={self.divergence_threshold:.3f}m, debounce={self.divergence_debounce_count}, "
            f"grace={self.divergence_grace_period:.1f}s, max_resyncs={self.divergence_max_resyncs}, "
            f"min_interval={self.divergence_min_resync_interval:.1f}s)")
        self.get_logger().info(f"  - State topic: {self.state_topic} (std_msgs/String, latched)")
        self.get_logger().info(
            f"  - Compliant grasp approach: task_compliance={self.task_compliance_ctrl_service}, "
            f"move_line={self.move_line_service}, "
            f"release_compliance={self.release_compliance_ctrl_service}, "
            f"check_motion={self.check_motion_service}, "
            f"stiffness=[{self.compliant_translation_stiffness:.0f}N/m, "
            f"{self.compliant_rotation_stiffness:.0f}Nm/rad], ref={self.grasp_move_ref} "
            f"(base-frame approach vector), "
            f"settle={self.compliance_settle_time:.2f}s before MoveLine")
        self.get_logger().info(
            f"  - RT streaming toggle: {self.enable_streaming_service} "
            f"(paused during compliance calls)")
        self.get_logger().info(
            f"  - Doosan service call timeout: {self.doosan_service_timeout:.1f}s "
            f"(move_line: {self.move_line_timeout:.1f}s — blocking motion call)")
        self.get_logger().info(
            f"  - Lift: move_line ref={self.lift_move_ref}, +{self.lift_distance:.3f}m")
        self.get_logger().info(
            f"  - Gripper action: {self.gripper_action} (joint={self.gripper_joint_name})")
        self.get_logger().info(f"  - Mask service: {self.mask_service_name}")
        self.get_logger().info(f"  - Attach service: {self.attach_service_name}")
        self.get_logger().info(
            f"  - Cube offset from tag: {self.grasp_cube_distance:.3f}m "
            f"along local {self.grasp_cube_axis}-axis")


    def _setup_interfaces(self):
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._marker_tracker = MarkerTracker(
            self.tf_buffer, target_frame=self.planning_frame,
            avg_window=self.marker_avg_window,
            avg_max_jump=self.marker_avg_max_jump,
            avg_reset_gap=self.marker_avg_reset_gap, logger=self.get_logger(),
            fixed_frame=self.marker_fixed_frame)
        self._camera_priority = CameraPriority(
            self.camera_failover_timeout, self.get_clock())

        def _on_markers(msg, is_primary):
            has_detection = select_marker(msg, self.target_marker_id) is not None
            if self._camera_priority.accept(is_primary, has_detection):
                self._marker_tracker.on_markers(msg, self.target_marker_id)

        self.marker_sub_primary = self.create_subscription(
            FiducialMarkerArray, self.primary_marker_topic,
            lambda msg: _on_markers(msg, True), 10
        )
        self.marker_sub_secondary = self.create_subscription(
            FiducialMarkerArray, self.secondary_marker_topic,
            lambda msg: _on_markers(msg, False), 10
        )

        self.execute_client = ActionClient(self, SendTrajectory, self.execute_action)
        self.gripper_client = ActionClient(self, GripperCommand, self.gripper_action)
        self.mask_client = self.create_client(SetMask, self.mask_service_name)
        self.attach_client = self.create_client(AttachObject, self.attach_service_name)

        # Doosan native services for the compliant grasp approach and the
        # post-grasp lift (see DoosanArm for why it owns its own callback group).
        self._doosan = DoosanArm(
            self,
            move_line_service=self.move_line_service,
            task_compliance_ctrl_service=self.task_compliance_ctrl_service,
            release_compliance_ctrl_service=self.release_compliance_ctrl_service,
            check_motion_service=self.check_motion_service,
            enable_streaming_service=self.enable_streaming_service,
            set_tool_service=self.set_tool_service,
            service_timeout=self.doosan_service_timeout,
            move_line_timeout=self.move_line_timeout,
            motion_poll_period=self.motion_poll_period,
            motion_idle_timeout=self.motion_idle_timeout,
            move_stop_service=self.move_stop_service,
            cancel_check=lambda: self._cancel_requested,
        )

        # Live-tracking goal updates for the reactive controller while it servoes
        # (see mpc_goal_topic above).
        self.mpc_goal_pub = self.create_publisher(Pose, self.mpc_goal_topic, 10)

        # Transient-local so a late subscriber gets the current state
        # immediately without waiting for the next transition.
        state_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.state_pub = self.create_publisher(String, self.state_topic, state_qos)

        # Base-motion interlock on BaseCommander (gradient_base_controller):
        # grasp.py explicitly opens/closes it (SetBool) around the window
        # where the base should be free to move — act_move_pregrasp, up to
        # grasp_near_threshold from grasp_approach_target. See
        # _set_grasp_motion_enable and BaseCommander's module docstring.
        self.grasp_motion_enable_client = self.create_client(
            SetBool, self.grasp_motion_enable_service_name)
        # The one condition under which grasp.py lets the base move during
        # act_move_pregrasp, pushed by _sync_grasp_motion_enable: the arm is
        # still far from its approach target (_on_execute_feedback). An
        # edge-detector so the service is only called on an actual change,
        # not on every feedback tick. Losing sight of the marker no longer
        # closes the window: the pose is remembered in marker_fixed_frame
        # (odom) and re-projected into the planning frame on every
        # live-goal tick, so it stays valid while the base drives through
        # an occlusion.
        self._near_target_far = True
        # Last value actually sent to the service, so the edge-detector
        # above cannot double-send the same state.
        self._grasp_motion_sent = None

        # Grasp trigger as an action (allows cancel, returns success/message).
        # cancel_callback is required: rclpy's default rejects every cancel
        # request, which would make cancellation silently do nothing.
        self.grasp_action_server = ActionServer(
            self, TriggerGrasp, 'grasp_trigger', self.grasp_trigger_cb,
            cancel_callback=self._on_grasp_cancel_requested)

        # Track in-flight goal handles so we can cancel them.
        self._trajectory_goal_handle = None
        self._gripper_goal_handle = None
        self._cancel_requested = False
        # Set by _fire whenever a 'failure' event fires during a run — see
        # _fire's docstring for why grasp_trigger_cb needs this instead of
        # just checking self._fsm.state == IDLE.
        self._sequence_failed = False
        # The grasp_trigger goal currently in flight, if any — used to
        # publish feedback on every state-machine transition (see
        # _on_fsm_state_changed) in addition to the ~/state topic.
        self._trigger_goal_handle = None
        # Set by _on_execute_feedback once on_target=True is seen, so
        # execute_trajectory knows a cancel-on-success is in flight.
        self._on_target_reached = False
        # Set by _on_execute_feedback when the divergence watchdog decides a
        # resync is warranted, so execute_trajectory knows a cancel-and-resend
        # (as opposed to a real failure) is in flight.
        self._resync_requested = False
        self._divergence_watchdog = ExecutionDivergenceWatchdog(
            divergence_threshold=self.divergence_threshold,
            debounce_count=self.divergence_debounce_count,
            grace_period_s=self.divergence_grace_period,
            max_resyncs=self.divergence_max_resyncs,
            min_resync_interval_s=self.divergence_min_resync_interval,
        )

        # Broadcast the computed target poses as TF (approach/grasp/lift_target) so
        # the planned goal can be checked in RViz before triggering.
        self.tf_broadcaster = TransformBroadcaster(self)
        self.target_tf_timer = self.create_timer(0.1, self._publish_target_tf)

    def _setup_state_machine(self):
        # Run-scoped data (poses computed at trigger time), read by later
        # states so a moving marker mid-sequence can't shift the target.
        self._run = {}
        self._fsm = StateMachine(
            transitions=TRANSITIONS,
            initial_state=IDLE,
            handlers={
                ACT_MOVE_PREGRASP: self._on_enter_move_pregrasp,
                ACT_OPEN_GRIPPER: self._on_enter_open_gripper,
                SRV_MASK_OBJECT: self._on_enter_mask_object,
                ACT_MOVE_GRASP: self._on_enter_move_grasp,
                ACT_CLOSE_GRIPPER: self._on_enter_close_gripper,
                ACT_MOVE_BACKOFF: self._on_enter_move_backoff,
                ACT_RETRACT_ARM: self._on_enter_retract_arm,
            },
            on_state_changed=self._on_fsm_state_changed,
            logger=self.get_logger(),
        )

    def _on_fsm_state_changed(self, state: str) -> None:
        self.state_pub.publish(String(data=state))
        if self._trigger_goal_handle is not None:
            self._trigger_goal_handle.publish_feedback(
                TriggerGrasp.Feedback(state=state, message=''))

    async def _fire(self, event: str) -> None:
        """Wrapper around self._fsm.fire that also records whether this run
        ever fired 'failure' — every busy state's 'failure' transition lands
        on the same IDLE state as the normal success path (see TRANSITIONS),
        so grasp_trigger_cb can't tell the two apart from self._fsm.state
        alone and needs this flag instead.

        Also the single point where a cancel takes effect: once
        _cancel_requested is set, any non-failure event from a busy state is
        turned into 'failure', so a state whose work happened to complete
        after the cancel (a MoveLine that ran to its end, a gripper goal that
        wasn't in flight yet when the cancel callback ran) ends the sequence
        instead of stepping into the next state. Individual states only need
        to *finish* promptly on cancel, not to route it.
        """
        if event == 'failure':
            self._sequence_failed = True
        elif self._cancel_requested and self._fsm.state != IDLE:
            self.get_logger().info(
                f"Cancel requested: '{event}' from {self._fsm.state} overridden by 'failure'")
            self._sequence_failed = True
            event = 'failure'
        await self._fsm.fire(event)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_grasp_cancel_requested(self, goal_handle):
        """Accept the cancel request and cancel whatever sub-goal is in flight.

        Cancelling the sub-goal makes the awaiting execute_trajectory/
        call_gripper_action return False, which the state machine already
        turns into a 'failure' transition back to IDLE — no separate abort
        path needed.
        """
        self.get_logger().info("Cancel requested, cancelling in-flight sub-goal(s)...")
        self._cancel_requested = True
        if self._trajectory_goal_handle is not None:
            self._trajectory_goal_handle.cancel_goal_async()
        if self._gripper_goal_handle is not None:
            self._gripper_goal_handle.cancel_goal_async()
        # Doosan-driven states (MoveLine, force descent, motion-idle waits)
        # have no goal handle to cancel: stop the arm so their blocking
        # service call returns, then _fire's cancel guard ends the sequence.
        if self._fsm.state in self._DOOSAN_STATES:
            self._doosan.stop_now()
        return CancelResponse.ACCEPT

    async def grasp_trigger_cb(self, goal_handle):
        """Action callback for grasp trigger. Supports cancel to stop the sequence."""
        if self._fsm.state != IDLE:
            self.get_logger().warn(
                f"Grasp sequence already running (state={self._fsm.state}), rejecting")
            goal_handle.abort()
            result = TriggerGrasp.Result()
            result.success = False
            result.message = "Sequence already running"
            return result

        self.get_logger().info("Trigger received, starting grasp sequence")
        self._cancel_requested = False
        self._sequence_failed = False
        self._trigger_goal_handle = goal_handle

        try:
            try:
                await self._fire('user_trigger')
            except Exception as e:
                self.get_logger().error(f"Unexpected error in grasp sequence: {e}")
                self._fsm.reset(IDLE)
                goal_handle.abort()
                result = TriggerGrasp.Result()
                result.success = False
                result.message = str(e)
                return result

            result = TriggerGrasp.Result()
            if self._cancel_requested:
                self.get_logger().info("Grasp sequence cancelled")
                result.success = False
                result.message = "Cancelled"
                goal_handle.canceled()
                return result

            result.success = (self._fsm.state == IDLE) and not self._sequence_failed
            result.message = "Grasp sequence completed" if result.success else "Sequence failed"
            if result.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return result
        finally:
            self._trigger_goal_handle = None

    # ── State machine handlers ───────────────────────────────────────────────

    async def _set_grasp_motion_enable(self, enabled: bool) -> bool:
        """Open/close BaseCommander's grasp_motion_enable interlock (see its
        module docstring) -- the precise, distance-based counterpart to the
        old whole-state grasper_allowed_states gate."""
        if not self.grasp_motion_enable_client.service_is_ready():
            self.get_logger().warn("grasp_motion_enable service not available")
            return False
        req = SetBool.Request()
        req.data = enabled
        self._grasp_motion_sent = enabled
        resp = await self.grasp_motion_enable_client.call_async(req)
        return resp.success

    def _sync_grasp_motion_enable(self) -> None:
        """Push `_near_target_far` to the grasp_motion_enable service, but
        only when it actually changes. Fire-and-forget (call_async without
        await) because the caller is a plain feedback callback, not a
        coroutine; the explicit open/close at the leg boundaries stays
        awaited in _set_grasp_motion_enable."""
        allow = self._near_target_far
        if allow == self._grasp_motion_sent:
            return
        if not self.grasp_motion_enable_client.service_is_ready():
            self.get_logger().warn(
                "grasp_motion_enable service not available",
                throttle_duration_sec=5.0)
            return
        self._grasp_motion_sent = allow
        req = SetBool.Request()
        req.data = allow
        self.grasp_motion_enable_client.call_async(req)

    async def _on_enter_move_pregrasp(self) -> None:
        if self._marker_tracker.pose is None:
            self.get_logger().warn("No marker received yet, cannot grasp")
            await self._fire('failure')
            return
        marker_in_plan = self._marker_tracker.transform_to(self.planning_frame)
        if marker_in_plan is None:
            await self._fire('failure')
            return
        cube_in_plan = self._cube_pose_from_marker(marker_in_plan)

        approach_pose = self._marker_tracker.offset_along_z(
            cube_in_plan, self.approach_distance, self.approach_axis_sign, rot_y = (0.0, 0.0, 0.0, 1.0))
        grasp_pose = self._marker_tracker.offset_along_z(
            cube_in_plan, self.grasp_distance, self.approach_axis_sign, rot_y = (0.0, 0.0, 0.0, 1.0))
        self._run = {
            'marker_in_plan': marker_in_plan,
            'cube_in_plan': cube_in_plan,
            'approach_pose': approach_pose,
            # Unit vector approach->grasp, in the planning frame. act_move_grasp
            # moves along this via a base-frame relative MoveLine while
            # compliant (see grasp_move_ref), rather than assuming the tool's
            # +Z axis matches this direction.
            'approach_dir': unit_vector_between(approach_pose, grasp_pose),
        }

        self.get_logger().info("Moving to PREGRASP (approach) pose...")
        # Reset for this attempt: the edge-detector only fires the closing
        # service call on a transition, so a leftover False from a previous
        # attempt must not suppress the open call below.
        self._near_target_far = True
        await self._set_grasp_motion_enable(True)
        ok = await self.execute_trajectory(
            self._run['approach_pose'],
            live_pose_fn=lambda: self._recompute_live_pose(self.approach_distance))
        # Belt-and-suspenders close: normally _on_execute_feedback already
        # closed this once position_error crossed grasp_near_threshold, but a
        # failure/cancel before that ever happens must not leave the base free
        # to move for the rest of the (non-pregrasp) sequence.
        await self._set_grasp_motion_enable(False)
        await self._fire('arm_done' if ok else 'failure')

    async def _on_enter_open_gripper(self) -> None:
        self.get_logger().info("Opening gripper...")
        ok = await self.call_gripper_action(self.gripper_open_position, grasp_object = False)
        await self._fire('gripper_open' if ok else 'failure')

    async def _on_enter_mask_object(self) -> None:
        """Recompute the cube pose from the *current* marker detection rather
        than reusing the snapshot frozen in _run at act_move_pregrasp entry —
        several states (act_open_gripper, act_move_grasp, act_close_gripper)
        have elapsed since, and this now runs after the gripper has closed on
        the object. If the tag is occluded by the gripper by this point,
        MarkerTracker.pose simply holds its last-seen value rather than going
        stale/None, so this falls back to the last pre-occlusion detection
        instead of failing. Transformed into planning_frame (dsr01/base_link)
        rather than left in the detecting camera's own frame: set_mask and
        attach_object both need the pose in the robot base frame (set_mask
        explicitly via req.frame_id; attach_object via req.header.frame_id)."""
        self.get_logger().info(f"Masking object '{self.mask_object_name}'...")
        if self._marker_tracker.pose is None:
            self.get_logger().error("No marker pose available, cannot mask object")
            await self._fire('failure')
            return
        marker_in_plan = self._marker_tracker.transform_to(self.planning_frame)
        if marker_in_plan is None:
            await self._fire('failure')
            return
        cube = self._cube_pose_from_marker(marker_in_plan)
        ok = await self.call_set_mask_service(cube.pose, cube.header.frame_id)
        if ok:
            # AttachObject.srv now takes the object's shape/pose directly, so
            # this single call both creates the collision geometry and
            # attaches it to the gripper — no separate add_object step.
            ok = await self.call_attach_object_service(cube.pose, cube.header.frame_id)
        await self._fire('object_masked' if ok else 'failure')

    async def _on_enter_move_grasp(self) -> None:
        """Final pregrasp->grasp leg: TaskComplianceCtrl (soft), then a normal
        MoveLine toward the object along _run['approach_dir'] (a base-frame
        unit vector, computed in _on_enter_move_pregrasp) while compliant, so
        marker/pose error at contact is absorbed by compliance instead of
        fought by the free-space MPC. No SetDesiredForce: compliance alone
        plus a position move, nothing pushing with a target force.

        Ordering is load-bearing: the MoveLine must not be issued until the
        stiffness ramp has settled (compliance_settle_time), or it is accepted
        and silently dropped. Once it does run, sync_type=SYNC blocks normally
        under compliance — wait_motion_idle() after it is only a safety net,
        not the completion signal.

        dsr_controller2's RT command streaming is paused for the whole
        compliance window: left on, it keeps streaming commands that
        TASK_COMPLIANCE_CONTROL rejects (observed:
        "state[TASK_COMPLIANCE_CONTROL] rejected event[eSpeedJ]"), which
        faulted the controller into MOTION_HOLD. Note this is about the
        streaming loop's own commands, not movel: a plain movel runs fine with
        streaming left on (act_retract_arm does exactly that). Streaming is
        always re-enabled before leaving this state (success, failure, or
        exception) — act_retract_arm's MoveLine needs it back on.
        """
        self.get_logger().info("Moving to GRASP pose (compliant linear approach)...")
        approach_dir = self._run.get('approach_dir')
        if approach_dir is None:
            self.get_logger().error("No approach direction available, cannot move toward grasp")
            await self._fire('failure')
            return
        if not await self._doosan.enable_streaming(False):
            self.get_logger().error("Failed to disable RT streaming, aborting compliant approach")
            await self._fire('failure')
            return

        ok = False
        try:
            stx = (
                [self.compliant_translation_stiffness] * 3
                + [self.compliant_rotation_stiffness] * 3
            )
            ok = await self._doosan.task_compliance_ctrl(
                stx, ref=self.grasp_move_ref, ramp_time=self.compliance_ramp_time)

            if ok:
                # Do not remove: a MoveLine issued during the stiffness ramp is
                # accepted and then dropped — see compliance_settle_time.
                await self._doosan.sleep(self.compliance_settle_time)
                delta_mm = abs(
                    self.approach_axis_sign * (self.approach_distance - self.grasp_distance)) * 1000.0
                dx, dy, dz = (c * delta_mm for c in approach_dir)
                ok = await self._doosan.move_line_relative(
                    [dx, dy, dz, 0.0, 0.0, 0.0], ref=self.grasp_move_ref,
                    vel=self.compliant_move_vel, acc=self.compliant_move_acc)

            # Confirm the robot has actually stopped before releasing
            # compliance, on the failure path too: a motion may be running
            # regardless of what move_line reported.
            if not await self._doosan.wait_motion_idle():
                ok = False

            # Always try to release compliance — even on failure/cancel — so
            # the robot isn't left soft for the next (curobo) motion.
            if not await self._doosan.release_compliance_ctrl():
                self.get_logger().error("Failed to release compliance control!")
        finally:
            if not await self._doosan.enable_streaming(True):
                self.get_logger().error("Failed to re-enable RT streaming!")

        await self._fire('arm_done' if ok else 'failure')

    async def _on_enter_close_gripper(self) -> None:
        self.get_logger().info("Closing gripper...")
        ok = await self.call_gripper_action(self.gripper_closed_position, grasp_object = True)
        # if ok:
        #     # Payload/TCP now reflects the held object rather than the bare
        #     # gripper. Not load-bearing for the state machine: a failure here
        #     # is logged but doesn't abort an otherwise-successful grasp.
        #     if not await self._doosan.set_tool(self.grasp_tool_name):
        #         self.get_logger().error(f"Failed to set tool '{self.grasp_tool_name}'")
        await self._fire('gripper_closed' if ok else 'failure')

    async def _on_enter_move_backoff(self) -> None:
        """Post-grasp backoff leg (mirrors _on_enter_move_grasp in reverse):
        with the object now held, retreat back out along -_run['approach_dir']
        from grasp_distance to approach_distance. Free-space motion like the
        vertical act_retract_arm lift, so no compliance wrapper is needed —
        plain MoveLine, same as act_retract_arm. Uses lift_move_vel/acc (not
        compliant_move_vel/acc): this leg is grouped with the free-space
        post-grasp motions, not with the contact-sensitive plunge.
        """
        self.get_logger().info("Backing off to APPROACH pose (linear retreat)...")
        approach_dir = self._run.get('approach_dir')
        if approach_dir is None:
            self.get_logger().error("No approach direction available, cannot back off")
            await self._fire('failure')
            return

        # Tiny straight-up lift to break contact with the table before
        # retreating along approach_dir — otherwise the retreat can drag the
        # object/gripper along the surface.
        ok = await self._doosan.move_line_relative(
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0], ref=self.grasp_move_ref,
            vel=self.lift_move_vel, acc=self.lift_move_acc)

        if ok:
            delta_mm = abs(
                self.approach_axis_sign * (self.approach_distance - self.grasp_distance)) * 1000.0
            # Negated approach_dir: same axis as the plunge, opposite sense.
            dx, dy, dz = (-c * delta_mm for c in approach_dir)
            ok = await self._doosan.move_line_relative(
                [dx-100, dy, dz, 0.0, 0.0, 0.0], ref=self.grasp_move_ref,
                vel=self.lift_move_vel, acc=self.lift_move_acc)

        ok = await self._doosan.wait_motion_idle() and ok

        await self._fire('arm_done' if ok else 'failure')

    async def _on_enter_retract_arm(self) -> None:
        """Post-grasp lift: a relative linear move (Doosan MoveLine) straight up
        along ref=lift_move_ref's Z axis, not curobo — this is free-space motion
        away from contact, so no compliance wrapper is needed."""
        self.get_logger().info(f"Retracting arm (+{self.lift_distance:.3f}m Z, linear move)...")
        ok = await self._doosan.move_line_relative(
            [0.0, 0.0, self.lift_distance * 1000.0, 0.0, 0.0, 0.0], ref=self.lift_move_ref,
            vel=self.lift_move_vel, acc=self.lift_move_acc)
        # Outside compliance move_line does appear to block, but don't rely on
        # it — report the grasp complete only once the arm has actually stopped.
        ok = await self._doosan.wait_motion_idle() and ok
        if ok:
            self.get_logger().info("Grasp sequence complete")
        await self._fire('arm_done' if ok else 'failure')

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _cube_pose_from_marker(self, marker_in_plan: PoseStamped) -> PoseStamped:
        """The cube isn't exactly at the tag: it sits `grasp_cube_distance`
        away from the tag along the tag's own local `grasp_cube_axis` axis.
        Same orientation as the marker (only position shifts)."""
        ax, ay, az = quat_to_axis(marker_in_plan.pose.orientation, self.grasp_cube_axis)
        p = marker_in_plan.pose.position
        cube = PoseStamped()
        cube.header = marker_in_plan.header
        cube.pose.position.x = p.x + self.grasp_cube_distance * ax
        cube.pose.position.y = p.y + self.grasp_cube_distance * ay
        cube.pose.position.z = p.z + self.grasp_cube_distance * az
        # cube.pose.orientation = marker_in_plan.pose.orientation
        gx, gy, gz, gw = quat_mul((marker_in_plan.pose.orientation.x, 
                                   marker_in_plan.pose.orientation.y, 
                                   marker_in_plan.pose.orientation.z, 
                                   marker_in_plan.pose.orientation.w), 
                                   (0.0, 1.0, 0.0, 0.0))
        cube.pose.orientation.x = gx
        cube.pose.orientation.y = gy
        cube.pose.orientation.z = gz
        cube.pose.orientation.w = gw
        return cube

    def _recompute_live_pose(self, distance: float) -> Pose:
        """Re-derive the `distance`-offset target from the *current* marker
        pose (as opposed to the frozen snapshot in self._run), so a moving
        marker can keep steering the reactive controller mid-motion. Returns
        None if the marker/TF isn't available right now."""
        if self._marker_tracker.pose is None:
            return None
        marker_in_plan = self._marker_tracker.transform_to(self.planning_frame)
        if marker_in_plan is None:
            return None
        cube_in_plan = self._cube_pose_from_marker(marker_in_plan)
        return self._marker_tracker.offset_along_z(cube_in_plan, distance, self.approach_axis_sign, rot_y = (0.0, 0.0, 0.0, 1.0))

    def _publish_target_tf(self):
        """Broadcast approach/grasp/lift_target as TF for RViz verification."""
        if self._marker_tracker.pose is None:
            return
        marker_in_plan = self._marker_tracker.transform_to(self.planning_frame)
        if marker_in_plan is None:
            return
        cube_in_plan = self._cube_pose_from_marker(marker_in_plan)
        approach_pose = self._marker_tracker.offset_along_z(
            cube_in_plan, self.approach_distance, self.approach_axis_sign, rot_y = (0.0, 0.0, 0.0, 1.0))
        grasp_pose = self._marker_tracker.offset_along_z(
            cube_in_plan, self.grasp_distance, self.approach_axis_sign, rot_y = (0.0, 0.0, 0.0, 1.0))
        lifted_pose = lift_pose(grasp_pose, self.lift_distance)
        self._broadcast_pose(cube_in_plan.pose, 'grasp_cube_target')
        self._broadcast_pose(approach_pose, 'grasp_approach_target')
        self._broadcast_pose(grasp_pose, 'grasp_target')
        self._broadcast_pose(lifted_pose, 'grasp_lift_target')

    def _broadcast_pose(self, pose: Pose, child_frame: str):
        """Broadcast a Pose (in the planning frame) as a TF frame."""
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.planning_frame
        tf.child_frame_id = child_frame
        tf.transform.translation.x = pose.position.x
        tf.transform.translation.y = pose.position.y
        tf.transform.translation.z = pose.position.z
        tf.transform.rotation = pose.orientation
        self.tf_broadcaster.sendTransform(tf)

    async def execute_trajectory(self, target_pose: Pose, live_pose_fn=None) -> bool:
        """Plan and execute a motion to target_pose (moves the arm).

        Stores the goal_handle so grasp_trigger_cb can cancel it if requested.

        If `live_pose_fn` is given, it's polled at mpc_goal_rate for the
        duration of the goal and its result republished on mpc_goal_topic —
        this lets the in-flight reactive controller keep following a marker
        that moves after the goal was sent, instead of only ever chasing the
        pose captured at trigger time.

        The divergence watchdog (see ExecutionDivergenceWatchdog) covers the
        whole attempt, including any resyncs within it: if it fires mid-goal,
        the goal is cancelled and resent — from the live pose if
        `live_pose_fn` is given, since the stale target_pose is exactly what
        the resync is meant to correct for — instead of failing outright, up
        to `divergence_max_resyncs` times.
        """
        self._divergence_watchdog.reset()
        current_target = target_pose
        while True:
            outcome = await self._send_and_await_goal(current_target, live_pose_fn)
            if outcome is not None:
                return outcome
            # outcome is None: the watchdog requested a resync. Recompute the
            # target from the live pose if we have one, else resend as-is.
            if live_pose_fn is not None:
                fresh = live_pose_fn()
                if fresh is not None:
                    current_target = fresh
            self.get_logger().warn(
                f"Divergence watchdog: resending goal "
                f"(resync {self._divergence_watchdog.resync_count}/{self.divergence_max_resyncs})")

    async def _send_and_await_goal(self, target_pose: Pose, live_pose_fn) -> bool:
        """One send-goal/await-result cycle.

        Returns True/False for a final outcome, or None if execute_trajectory
        should resend the goal (the divergence watchdog fired and the attempt
        wasn't otherwise externally cancelled).
        """
        if not self.execute_client.server_is_ready():
            self.get_logger().error("Execute trajectory action not available")
            return False

        goal = SendTrajectory.Goal()
        goal.target_pose = target_pose
        goal.allow_cached = False  # always plan+execute fresh for this target
        self.get_logger().info(
            f"Executing trajectory: target=({target_pose}"
        )
        self._on_target_reached = False
        self._resync_requested = False
        goal_handle = await self.execute_client.send_goal_async(
            goal, feedback_callback=self._on_execute_feedback
        )
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Execute goal rejected")
            return False

        # Store goal_handle so grasp_trigger_cb (and the feedback callback,
        # once on_target=True or a resync is warranted) can cancel it.
        self._trajectory_goal_handle = goal_handle
        if self._cancel_requested:
            # Cancel arrived before the handle existed: the cancel callback
            # couldn't reach this goal, so do it here.
            goal_handle.cancel_goal_async()
        live_goal_timer = None
        if live_pose_fn is not None:
            live_goal_timer = self.create_timer(
                1.0 / self.mpc_goal_rate, lambda: self._publish_live_goal(live_pose_fn))
        try:
            result_response = await goal_handle.get_result_async()
            if self._on_target_reached:
                # Server is in continuous servoing (TRACKING) and never ends
                # the goal on its own — we cancel it ourselves as soon as
                # feedback reports on_target, and treat that as success.
                self.get_logger().info("On target: trajectory cancelled as success")
                return True
            if self._resync_requested and not self._cancel_requested:
                # A real external cancel (grasp_trigger_cb) takes priority
                # over a watchdog-driven resync: fall through to the normal
                # result handling below instead of looping again.
                return None
            result = result_response.result
            if result.success:
                self.get_logger().info(f"Trajectory succeeded: {result.message}")
                return True
            self.get_logger().error(f"Trajectory failed: {result.message}")
            return False
        finally:
            if live_goal_timer is not None:
                live_goal_timer.cancel()
                self.destroy_timer(live_goal_timer)
            self._trajectory_goal_handle = None

    def _publish_live_goal(self, live_pose_fn) -> None:
        """Timer callback: recompute the live target and publish it on
        mpc_goal_topic. The marker pose is remembered in marker_fixed_frame
        and re-projected into the planning frame here through the latest
        TF, so the goal keeps tracking the (static) tag while the base
        drives, seen or not; every fresh detection folds into the average
        and corrects it. Only an unavailable TF skips the tick."""
        pose = live_pose_fn()
        if pose is not None:
            self.mpc_goal_pub.publish(pose)

    def _on_execute_feedback(self, feedback_msg):
        fb = feedback_msg.feedback
        # Edge-triggered base-motion interlock: only call
        # grasp_motion_enable on an actual far<->near transition, not on
        # every feedback tick, and re-evaluated both ways (not latched to
        # near) so the base correctly regains permission if a
        # resync/divergence recovery moves the arm back away from the target.
        far = fb.position_error >= self.grasp_near_threshold
        if far != self._near_target_far:
            self._near_target_far = far
            self._sync_grasp_motion_enable()
        self.get_logger().info(
            f"  [{fb.state}] progress={fb.step_progression:.2f} "
            f"err={fb.position_error:.4f}m on_target={fb.on_target} "
            f"controller_err={fb.controller_position_error:.4f}m",
            throttle_duration_sec=1.0,
        )
        # self.get_logger().info(f"is on Target from curobo: {fb.on_target}")

        # Fallback: if the solver's own error estimate agrees with the
        # FK-measured error within threshold, AND is itself extremely low,
        # consider the arm arrived even if on_target never latches (e.g. a
        # planner whose hold_count/orientation gating never converges). The
        # tight bypass threshold on controller_position_error alone guards
        # against two errors that are merely similar but both still large.
        # controller_position_error == -1.0 means the active planner doesn't
        # expose it (only LBFGS does) — skip the check in that case.
        controller_error_converged = (
            fb.controller_position_error != -1.0
            and fb.controller_position_error < self.controller_error_bypass_threshold
            and abs(fb.controller_position_error - fb.position_error) < self.controller_error_match_threshold
        )
        if controller_error_converged and not fb.on_target:
            self.get_logger().info(
                f"controller_position_error below {self.controller_error_bypass_threshold}m "
                "and matches position_error: treating as on target"
            )

        on_target = fb.on_target or controller_error_converged
        if on_target and not self._on_target_reached and self._trajectory_goal_handle is not None:
            self._on_target_reached = True
            self.get_logger().info("on_target=True, cancelling trajectory goal")
            self._trajectory_goal_handle.cancel_goal_async()
            return

        if (self.divergence_watchdog_enabled and not self._resync_requested
                and not self._on_target_reached and self._trajectory_goal_handle is not None):
            now_s = self.get_clock().now().nanoseconds * 1e-9
            outcome = self._divergence_watchdog.on_feedback(
                fb.position_error, fb.controller_position_error, now_s)
            if outcome == ExecutionDivergenceWatchdog.RESYNC:
                self.get_logger().warn(
                    f"Divergence watchdog: position_error={fb.position_error:.4f}m vs "
                    f"controller_position_error={fb.controller_position_error:.4f}m "
                    "— cancelling and resending goal")
                self._resync_requested = True
                self._trajectory_goal_handle.cancel_goal_async()
            elif outcome == ExecutionDivergenceWatchdog.BUDGET_EXHAUSTED:
                self.get_logger().error(
                    f"Divergence watchdog: still diverging after "
                    f"{self.divergence_max_resyncs} resyncs — failing this attempt")
                # _resync_requested stays False: _send_and_await_goal falls
                # through to the normal (cancelled -> not success) result
                # handling below, so this is reported as a genuine failure
                # rather than mistaken for a user-requested cancel or a resync.
                self._trajectory_goal_handle.cancel_goal_async()

    async def call_gripper_action(self, position: float, grasp_object = False) -> bool:
        """Send a ParallelGripperCommand goal for gripper_joint_name to position.

        Note: the real Robotiq controller only exposes a `position` command
        interface (effort/velocity command interfaces were intentionally not
        claimed, since the hardware doesn't support them) — so this goal only
        ever sets position.
        """
        if self.simulate_gripper:
            action = "OPEN" if position <= self.gripper_open_position else "CLOSE"
            self.get_logger().info(f"[SIM] Gripper {action} (position={position:.3f})")
            return True

        if not self.gripper_client.server_is_ready():
            self.get_logger().error("Gripper action server not available")
            return False

        goal_msg = GripperCommand.Goal()
        # goal_msg.command.name = [self.gripper_joint_name]
        goal_msg.command.position = float(position)
        self.get_logger().info(f"Gripper command: position={position:.3f}")
        goal_handle = await self.gripper_client.send_goal_async(goal_msg)
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected")
            return False

        # Store goal_handle so grasp_trigger_cb can cancel if needed.
        self._gripper_goal_handle = goal_handle
        if self._cancel_requested:
            goal_handle.cancel_goal_async()
        try:
            result = (await goal_handle.get_result_async()).result
            if(grasp_object):
                self.get_logger().info(f"Gripper done: Grasp object={result.stalled}")
                return result.stalled
            else:
                self.get_logger().info(f"Gripper done: reached_goal={result.reached_goal}")
                return result.reached_goal
        finally:
            self._gripper_goal_handle = None

    async def call_set_mask_service(self, pose: Pose, frame_id: str) -> bool:
        """Exclude the target object (cuboid, mask_object_dimensions) from
        world collision at `pose` (expressed in `frame_id`), via curobo_ros's
        set_mask service — curobo resolves the TF transform itself."""
        if not self.mask_client.service_is_ready():
            self.get_logger().error("set_mask service not available")
            return False

        req = SetMask.Request()
        req.type = SetMask.Request.CUBOID
        req.name = self.mask_object_name
        req.frame_id = frame_id
        req.pose = pose
        req.dimensions = Vector3(
            x=self.mask_object_dimensions[0],
            y=self.mask_object_dimensions[1],
            z=self.mask_object_dimensions[2],
        )
        self.get_logger().info(f"request mask: {req}")
        resp = await self.mask_client.call_async(req)
        self.get_logger().info(
            f"set_mask '{self.mask_object_name}': success={resp.success} msg=\"{resp.message}\"")
        return resp.success

    async def call_attach_object_service(self, pose: Pose, frame_id: str) -> bool:
        """Create the target object's collision geometry (cuboid,
        mask_object_dimensions) at `pose` and attach it to the gripper in one
        call, via curobo_ros's attach_object service, so it moves with the
        tool for the rest of the sequence (act_move_grasp onward) instead of
        staying fixed at the pre-grasp pose. `frame_id` is the TF frame
        `pose` is expressed in (req.header.frame_id); the service transforms
        it to the gripper at call time, matching the current joint config."""
        if not self.attach_client.service_is_ready():
            self.get_logger().error("attach_object service not available")
            return False

        req = AttachObject.Request()
        req.type = AttachObject.Request.CUBOID
        req.header.frame_id = frame_id
        req.pose = pose
        req.dimensions = Vector3(
            x=self.mask_object_dimensions[0],
            y=self.mask_object_dimensions[1],
            z=self.mask_object_dimensions[2],
        )
        self.get_logger().info(f"request attach object: {req}")

        resp = await self.attach_client.call_async(req)
        self.get_logger().info(
            f"attach_object '{self.mask_object_name}': success={resp.success} "
            f"msg=\"{resp.message}\"")
        return resp.success

    def _load_parameters(self):
        # Kinect is the primary camera; the D405 only takes over once the
        # kinect hasn't reported a detection for camera_failover_timeout
        # seconds, and control returns to it as soon as it reports again.
        self.declare_parameter('primary_marker_topic', '/kinect/fiducial_markers')
        self.declare_parameter('secondary_marker_topic', '/d405/fiducial_markers')
        self.declare_parameter('camera_failover_timeout', 1.0)
        self.declare_parameter('target_marker_id', -1)
        self.declare_parameter('trigger_topic', '/brain/trigger')

        # Marker pose smoothing: average the last N detections instead of
        # trusting each one raw, so a single bad detection frame doesn't
        # instantly become the grasp target.
        self.declare_parameter('marker_avg_window', 5)
        # A new sample farther than this from the current running average is
        # dropped as a likely detection glitch rather than real marker motion.
        self.declare_parameter('marker_avg_max_jump', 0.05)  # m
        # ...unless the marker hasn't been seen for longer than this: then the
        # buffer is stale and the new sample is accepted unconditionally as a
        # fresh acquisition instead of being compared against it.
        self.declare_parameter('marker_avg_reset_gap', 1.0)  # s
        # Fixed frame the marker pose is remembered in (see MarkerTracker):
        # the tag is static in the world, so a pose kept in the lidar/map
        # odometry frame stays valid (to odom accuracy, ~5 cm) however far
        # the base drives while the tag is occluded — the phantom chase of
        # run_20260910_182418/ep3 came from caching it in planning_frame,
        # which rides on the base. Must be TF-connected to planning_frame
        # at all times.
        self.declare_parameter('marker_fixed_frame', 'odom')

        # Planning frame = curobo base_link (target_pose has no frame_id).
        self.declare_parameter('planning_frame', 'dsr01/base_link')
        # Unified "plan + execute" action: moves the arm to target_pose.
        self.declare_parameter('execute_action', '/curobo_trajectory_planner/execute_trajectory')
        # While execute_trajectory's reactive controller (MPC) is servoing toward
        # pregrasp/grasp, republish the offset target recomputed from the *live*
        # marker pose here so a moving tag keeps pulling the goal (mpc_goal_callback
        # forwards it into planner.set_live_goal(...)).
        self.declare_parameter('mpc_goal_topic', '/curobo_trajectory_planner/mpc_goal')
        self.declare_parameter('mpc_goal_rate', 10.0)
        # Fallback arrival check for planners whose on_target never fires
        # (e.g. hold_count/orientation gating never converges): if the
        # solver's own controller_position_error agrees with the FK-measured
        # position_error within this threshold AND position_error itself is
        # below it, treat the arm as on target. controller_position_error is
        # -1.0 when the active planner doesn't expose it (only LBFGS does) —
        # that sentinel is excluded from the comparison.
        self.declare_parameter('controller_error_match_threshold', 0.05)  # m
        # controller_position_error must itself be below this (very tight)
        # threshold to bypass on_target — a loose match against position_error
        # alone isn't enough, since two errors that are merely similar but
        # both still large don't mean the arm has actually arrived.
        self.declare_parameter('controller_error_bypass_threshold', 1e-3)  # m

        # Watchdog for a curobo execution that has internally "solved" (its
        # own controller_position_error keeps shrinking) while the real arm
        # has stalled (position_error plateaus) — observed after a rejected
        # SpeedJ command (acceleration limit / missed control deadline),
        # often triggered by simultaneous base motion. See
        # execution_divergence_watchdog.py for the mechanism. Cancelling and
        # resending the goal re-seeds the solver from the real robot state
        # and reliably recovers it; retargeting via mpc_goal alone does not.
        self.declare_parameter('divergence_watchdog_enabled', True)
        self.declare_parameter('divergence_threshold', 0.02)      # m
        self.declare_parameter('divergence_debounce_count', 3)    # feedback samples
        self.declare_parameter('divergence_grace_period', 3.0)    # s after a resync
        self.declare_parameter('divergence_max_resyncs', 3)       # per attempt
        self.declare_parameter('divergence_min_resync_interval', 5.0)  # s

        # Current state-machine state (e.g. "idle", "act_move_grasp"), as
        # std_msgs/String. Transient-local + published on every transition, so
        # a subscriber started mid-sequence still gets the current value
        # immediately instead of waiting for the next change.
        self.declare_parameter('state_topic', '~/state')

        # SetBool service on BaseCommander (gradient_base_controller) opened
        # while in act_move_pregrasp and farther than grasp_near_threshold
        # from grasp_approach_target, closed otherwise -- lets the mobile base
        # keep moving during the grasp approach. Mirrors release.py's
        # release_motion_enable; see base_commander.py's module docstring for
        # why it's OR'd with, not another entry alongside, grasper_allowed_states
        # (which no longer includes act_move_pregrasp by default, since this
        # service now supersedes that whole-state gate with a precise cutoff).
        self.declare_parameter(
            'grasp_motion_enable_service_name',
            '/gradient_base_controller/grasp_motion_enable')
        # FK position_error (m) below which the arm is considered close enough
        # to grasp_approach_target that the mobile base must stop moving, even
        # though execute_trajectory's own on_target/arrival tolerance
        # (controller_error_*_threshold above) may not have fired yet.
        self.declare_parameter('grasp_near_threshold', 0.10)  # m

        # act_move_grasp (final pregrasp->grasp leg, right before contact) uses
        # the Doosan driver's native compliance instead of curobo, so
        # marker/pose error at contact is absorbed by compliance rather than
        # fought by the free-space MPC: TaskComplianceCtrl (soft), then a
        # normal MoveLine toward the object while compliant, then
        # ReleaseComplianceCtrl. An earlier attempt saw movel rejected while
        # TASK_COMPLIANCE_CONTROL was active ("state[TASK_COMPLIANCE_CONTROL]
        # rejected event[eSpeedJ]"), but that was before RT streaming was
        # disabled first (see enable_streaming_service below) — the rejected
        # event may have been the streaming loop's own commands rather than
        # movel itself. Verify on the robot; if movel is still rejected under
        # compliance even with streaming off, this needs a different approach
        # (e.g. SetDesiredForce) after all.
        self.declare_parameter('move_line_service', '/dsr01/dsr_controller2/motion/move_line')
        self.declare_parameter('task_compliance_ctrl_service', '/dsr01/dsr_controller2/force/task_compliance_ctrl')
        self.declare_parameter('release_compliance_ctrl_service', '/dsr01/dsr_controller2/force/release_compliance_ctrl')
        # Polled before ReleaseComplianceCtrl and before reporting the grasp
        # complete, so neither happens while the arm is still moving.
        self.declare_parameter('check_motion_service', '/dsr01/dsr_controller2/motion/check_motion')
        # dsr_controller2's RT command streaming must be paused before any
        # compliance call: left on, it keeps streaming commands (observed
        # rejection: internal event 'eSpeedJ') that TASK_COMPLIANCE_CONTROL
        # rejects, which faulted the controller into MOTION_HOLD.
        self.declare_parameter('enable_streaming_service', '/ExecuteTrajectory/enable_streaming')
        # Called once the gripper has closed on the object, so the driver's
        # payload/TCP reflects the object now held rather than the bare gripper.
        self.declare_parameter('set_tool_service', '/dsr01/dsr_controller2/system/set_current_tool')
        # MoveStop, sent on cancel while a Doosan-driven state (MoveLine under
        # compliance, backoff, retract) is running: a sync MoveLine only
        # replies once the motion ends, so without it a cancel waits for the
        # move to finish (see DoosanArm.stop_now).
        self.declare_parameter('move_stop_service', '/dsr01/dsr_controller2/motion/move_stop')
        self.declare_parameter('grasp_tool_name', 'gripper-objet')
        # Hard timeout on every individual Doosan service call: the driver was
        # observed to never reply at all to a rejected/invalid command, hanging
        # the await forever otherwise — this bounds every single call so the
        # state machine can always fail back to idle instead of freezing until
        # an external cancel.
        self.declare_parameter('doosan_service_timeout', 2.0)  # s
        # MoveLine needs its own, much longer bound: with sync_type=SYNC the
        # service can only reply once the motion physically finishes (confirmed
        # outside compliance: a 100mm@50mm/s lift takes >2s and tripped
        # doosan_service_timeout even though the move itself was running fine).
        # Must exceed the longest commanded move's duration, not the driver's
        # round-trip latency.
        self.declare_parameter('move_line_timeout', 20.0)  # s
        # check_motion poll period + overall bound on waiting for the robot to
        # come back to DR_STATE_IDLE. Belt-and-braces after MoveLine, which is
        # expected to have blocked already.
        self.declare_parameter('motion_poll_period', 0.1)   # s
        self.declare_parameter('motion_idle_timeout', 20.0)  # s
        # Soft on all three translations, default-stiff on rotation (matches
        # TaskComplianceCtrl's own rotational default of 200 Nm/rad).
        self.declare_parameter('compliant_translation_stiffness', 200.0)  # N/m, range 0-20000
        self.declare_parameter('compliant_rotation_stiffness', 20.0)     # Nm/rad, range 0-400
        self.declare_parameter('compliance_ramp_time', 0.4)               # s, range 0-1.0
        # MANDATORY dead time between task_compliance_ctrl and the MoveLine:
        # a movel issued while the stiffness is still ramping is ACCEPTED
        # (success=True) and then silently dropped — check_motion never leaves
        # IDLE and the robot doesn't move at all. Reproduced standalone: same
        # command, gap=0 -> movel replies in 0.20s and nothing moves; gap=1.0s
        # -> movel blocks 3.96s and the move runs, at both 500 and 3000 N/m.
        # Must be > compliance_ramp_time; 1.0s is the verified value (0.4s
        # alone was never tested and has no margin).
        self.declare_parameter('compliance_settle_time', 1.0)             # s
        # Reference frame for both TaskComplianceCtrl and the grasp MoveLine:
        # base-frame, matching the explicit approach->grasp direction vector
        # computed in _on_enter_move_pregrasp (_run['approach_dir']). Assumes
        # planning_frame is the controller's DR_BASE frame; switch to
        # DR_WORLD(2) if your controller (M2.40+) defines a separate world
        # frame that planning_frame actually matches.
        self.declare_parameter('grasp_move_ref', DR_BASE)
        self.declare_parameter('compliant_move_vel', [30.0, 20.0])  # [mm/s, deg/s]
        self.declare_parameter('compliant_move_acc', [60.0, 40.0])  # [mm/s^2, deg/s^2]

        # act_retract_arm (post-grasp lift) also uses MoveLine directly instead of
        # curobo. Not compliance-wrapped: this leaves contact and moves through
        # free space, so no need to absorb force error. ref=DR_BASE assumes
        # planning_frame ('dsr01/base_link') is the robot's own base frame with Z up;
        # switch to DR_WORLD if your controller (M2.40+) defines a separate world
        # frame that this TF frame actually corresponds to.
        self.declare_parameter('lift_move_ref', DR_BASE)
        self.declare_parameter('lift_move_vel', [50.0, 30.0])  # [mm/s, deg/s]
        self.declare_parameter('lift_move_acc', [80.0, 50.0])  # [mm/s^2, deg/s^2]

        # Gripper (real controller serves control_msgs/action/GripperCommand,
        # a sensor_msgs/JointState-based goal — NOT the older GripperCommand type).
        self.declare_parameter('gripper_action', '/robotiq_gripper_controller/gripper_cmd')
        self.declare_parameter('gripper_joint_name', 'robotiq_85_left_knuckle_joint')
        self.declare_parameter('gripper_open_position', 0.01)
        # URDF joint limit is [0.0, 0.8]; 0.7929 matches the driver's own
        # fully-closed default (2f_85.ros2_control.xacro).
        self.declare_parameter('gripper_closed_position', 0.79)
        # Gripper not wired yet: just log open/close instead of calling the action.
        self.declare_parameter('simulate_gripper', False)

        # Object masking: excludes the grasped object's known geometry from
        # world collision (curobo_ros robot_segmentation's set_mask service)
        # so the planner can enter its volume between mask and grasp.
        self.declare_parameter('mask_service_name', '/set_mask')
        self.declare_parameter('mask_object_name', 'grasp_target')
        self.declare_parameter('mask_object_dimensions', [0.1, 0.1, 0.3])
        # Creates the masked object's collision geometry and attaches it to
        # the gripper in one call, so it travels with the tool from
        # act_move_grasp on.
        self.declare_parameter('attach_service_name', '/curobo_trajectory_planner/attach_object')

        # Approach geometry: offsets along the marker Z axis (metres).
        # position = marker_pos + approach_axis_sign * distance * marker_Z
        self.declare_parameter('approach_distance', -0.2)
        self.declare_parameter('grasp_distance', -0.15)
        # -1.0: approach from the -Z side so the gripper Z (aligned with tag Z)
        # points toward the object. Flip to +1.0 if the tag Z points the other way.
        self.declare_parameter('approach_axis_sign', -1.0)
        # Vertical (world/planning-frame Z) lift distance for act_retract_arm.
        self.declare_parameter('lift_distance', 0.10)
        # The cube is not exactly at the tag: it sits `grasp_cube_distance`
        # away from the tag along the tag's own local `grasp_cube_axis` axis
        # ('x', 'y', or 'z'). Tune both empirically by watching the
        # 'cube_target' TF frame (published alongside approach/grasp/lift)
        # against the real cube position in RViz.
        self.declare_parameter('grasp_cube_distance', 0.05)
        self.declare_parameter('grasp_cube_axis', 'x')

        self.primary_marker_topic = self.get_parameter('primary_marker_topic').value
        self.secondary_marker_topic = self.get_parameter('secondary_marker_topic').value
        self.camera_failover_timeout = self.get_parameter('camera_failover_timeout').value
        self.target_marker_id = self.get_parameter('target_marker_id').value
        self.trigger_topic = self.get_parameter('trigger_topic').value
        self.marker_avg_window = self.get_parameter('marker_avg_window').value
        self.marker_avg_max_jump = self.get_parameter('marker_avg_max_jump').value
        self.marker_avg_reset_gap = self.get_parameter('marker_avg_reset_gap').value
        self.marker_fixed_frame = self.get_parameter('marker_fixed_frame').value
        self.planning_frame = self.get_parameter('planning_frame').value
        self.execute_action = self.get_parameter('execute_action').value
        self.mpc_goal_topic = self.get_parameter('mpc_goal_topic').value
        self.state_topic = self.get_parameter('state_topic').value
        self.grasp_motion_enable_service_name = self.get_parameter('grasp_motion_enable_service_name').value
        self.grasp_near_threshold = self.get_parameter('grasp_near_threshold').value
        self.mpc_goal_rate = self.get_parameter('mpc_goal_rate').value
        self.controller_error_match_threshold = self.get_parameter('controller_error_match_threshold').value
        self.controller_error_bypass_threshold = self.get_parameter('controller_error_bypass_threshold').value
        self.divergence_watchdog_enabled = self.get_parameter('divergence_watchdog_enabled').value
        self.divergence_threshold = self.get_parameter('divergence_threshold').value
        self.divergence_debounce_count = self.get_parameter('divergence_debounce_count').value
        self.divergence_grace_period = self.get_parameter('divergence_grace_period').value
        self.divergence_max_resyncs = self.get_parameter('divergence_max_resyncs').value
        self.divergence_min_resync_interval = self.get_parameter('divergence_min_resync_interval').value
        self.move_line_service = self.get_parameter('move_line_service').value
        self.task_compliance_ctrl_service = self.get_parameter('task_compliance_ctrl_service').value
        self.release_compliance_ctrl_service = self.get_parameter('release_compliance_ctrl_service').value
        self.check_motion_service = self.get_parameter('check_motion_service').value
        self.enable_streaming_service = self.get_parameter('enable_streaming_service').value
        self.set_tool_service = self.get_parameter('set_tool_service').value
        self.move_stop_service = self.get_parameter('move_stop_service').value
        self.grasp_tool_name = self.get_parameter('grasp_tool_name').value
        self.doosan_service_timeout = self.get_parameter('doosan_service_timeout').value
        self.move_line_timeout = self.get_parameter('move_line_timeout').value
        self.motion_poll_period = self.get_parameter('motion_poll_period').value
        self.motion_idle_timeout = self.get_parameter('motion_idle_timeout').value
        self.compliant_translation_stiffness = self.get_parameter('compliant_translation_stiffness').value
        self.compliant_rotation_stiffness = self.get_parameter('compliant_rotation_stiffness').value
        self.compliance_ramp_time = self.get_parameter('compliance_ramp_time').value
        self.compliance_settle_time = self.get_parameter('compliance_settle_time').value
        self.grasp_move_ref = self.get_parameter('grasp_move_ref').value
        self.compliant_move_vel = self.get_parameter('compliant_move_vel').value
        self.compliant_move_acc = self.get_parameter('compliant_move_acc').value
        self.lift_move_ref = self.get_parameter('lift_move_ref').value
        self.lift_move_vel = self.get_parameter('lift_move_vel').value
        self.lift_move_acc = self.get_parameter('lift_move_acc').value
        self.gripper_action = self.get_parameter('gripper_action').value
        self.gripper_joint_name = self.get_parameter('gripper_joint_name').value
        self.gripper_open_position = self.get_parameter('gripper_open_position').value
        self.gripper_closed_position = self.get_parameter('gripper_closed_position').value
        self.simulate_gripper = self.get_parameter('simulate_gripper').value
        self.mask_service_name = self.get_parameter('mask_service_name').value
        self.mask_object_name = self.get_parameter('mask_object_name').value
        self.mask_object_dimensions = self.get_parameter('mask_object_dimensions').value
        self.attach_service_name = self.get_parameter('attach_service_name').value
        self.approach_distance = self.get_parameter('approach_distance').value
        self.grasp_distance = self.get_parameter('grasp_distance').value
        self.approach_axis_sign = self.get_parameter('approach_axis_sign').value
        self.lift_distance = self.get_parameter('lift_distance').value
        self.grasp_cube_distance = self.get_parameter('grasp_cube_distance').value
        self.grasp_cube_axis = self.get_parameter('grasp_cube_axis').value


def main(args=None):
    rclpy.init(args=args)
    node = ObjectGrasper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
