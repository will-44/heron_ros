#!/usr/bin/env python3
"""Place a previously grasped object at a pose given by a flat fiducial
(ArTag) marker lying on the table (its own Z axis already points up, unlike
grasp.py's marker which is vertical on the object's side).

Symmetric to grasp.py's pick sequence, driven by the same explicit
event-driven state machine primitives:

  idle --user_trigger--> act_move_preplace
  act_move_preplace --arm_done--> act_move_place
  act_move_place --arm_done--> act_open_gripper
  act_open_gripper (opens gripper, then detach_object) --gripper_open--> srv_mask_placed
  srv_mask_placed --object_masked--> act_retract_arm
  act_retract_arm (lift, then best-effort remove_mask) --arm_done--> idle
  (any act_*/srv_* state) --failure--> idle

`self._fsm.state` doubles as the busy flag: only `idle` accepts
`user_trigger`, so a trigger received mid-sequence is ignored.

Unlike grasp.py there is no "cube offset from tag" step: the flat marker
already *is* the target point (no lateral offset to a cube face), so
approach/place poses are derived directly from the marker pose via
MarkerTracker.offset_along_z(). approach_axis_sign defaults to +1.0 (marker Z
already points up), producing a gripper pointing straight down at the table.
This flat marker does NOT use offset_along_z's own default orientation
convention (grasp.py's vertical marker does) - the correct rot_y for this
marker was verified empirically against RViz and is centralized in
ObjectReleaser._offset_target_pose(), the single place every consumer here
(execution, live tracking, TF display) must go through.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy

from tf2_ros import Buffer, TransformListener, TransformBroadcaster, Duration

from geometry_msgs.msg import PoseStamped, Pose, TransformStamped, Vector3
from std_msgs.msg import String
from control_msgs.action import GripperCommand
from curobo_msgs.action import SendTrajectory
from curobo_msgs.srv import SetMask, RemoveObject
from std_srvs.srv import Trigger, SetBool
from ros2_markertracker_interfaces.msg import FiducialMarkerArray
from heron_interface.action import TriggerRelease

from ..scripts.execution_divergence_watchdog import ExecutionDivergenceWatchdog
from ..scripts.marker_tracker import CameraPriority, MarkerTracker, select_marker
from ..scripts.state_machine import StateMachine
from ..scripts.utils_geom import lift_pose
from ..scripts.doosan import DoosanArm, DR_BASE, DR_FC_MOD_ABS

from rclpy.action import ActionServer, CancelResponse

# States (plain strings, not Enum: a single place to touch when adding one).
IDLE = 'idle'
ACT_MOVE_PREPLACE = 'act_move_preplace'
ACT_MOVE_PLACE = 'act_move_place'
ACT_OPEN_GRIPPER = 'act_open_gripper'
SRV_MASK_PLACED = 'srv_mask_placed'
ACT_RETRACT_ARM = 'act_retract_arm'

_BUSY_STATES = (
    ACT_MOVE_PREPLACE, ACT_MOVE_PLACE, ACT_OPEN_GRIPPER,
    SRV_MASK_PLACED, ACT_RETRACT_ARM,
)

TRANSITIONS = {
    (IDLE, 'user_trigger'): ACT_MOVE_PREPLACE,
    (ACT_MOVE_PREPLACE, 'arm_done'): ACT_MOVE_PLACE,
    (ACT_MOVE_PLACE, 'arm_done'): ACT_OPEN_GRIPPER,
    (ACT_OPEN_GRIPPER, 'gripper_open'): SRV_MASK_PLACED,
    (SRV_MASK_PLACED, 'object_masked'): ACT_RETRACT_ARM,
    (ACT_RETRACT_ARM, 'arm_done'): IDLE,
    # Generic abort: any busy state can bail out back to idle.
    **{(s, 'failure'): IDLE for s in _BUSY_STATES},
}


class ObjectReleaser(Node):
    """Place a held object at an ArTag-defined flat target, via an explicit state machine."""

    # States driven by Doosan services rather than a cancellable action goal
    # — a cancel there is delivered as a MoveStop (see _on_release_cancel_requested).
    _DOOSAN_STATES = (ACT_MOVE_PLACE, ACT_RETRACT_ARM)

    def __init__(self):
        super().__init__('object_releaser')

        self._load_parameters()

        self._setup_interfaces()
        self._setup_state_machine()

        self.get_logger().info("Object releaser initialized")
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
            f"  - Compliant place approach: task_compliance={self.task_compliance_ctrl_service}, "
            f"release_compliance={self.release_compliance_ctrl_service}, "
            f"check_motion={self.check_motion_service}, "
            f"stiffness=[{self.compliant_translation_stiffness:.0f}N/m, "
            f"{self.compliant_rotation_stiffness:.0f}Nm/rad], ref={self.place_move_ref}, "
            f"settle={self.compliance_settle_time:.2f}s before force descent")
        self.get_logger().info(
            f"  - Force-controlled descent: set_desired_force={self.set_desired_force_service}, "
            f"release_force={self.release_force_service}, "
            f"push={self.place_push_force_n:.1f}N, contact={self.place_contact_force_n:.1f}N "
            f"(max {self.place_contact_force_max_n:.1f}N) as |dF| vs a "
            f"{self.contact_baseline_samples}-sample baseline, ref {self.contact_check_ref} "
            f"(tool_force from {self.get_tool_force_service}), "
            f"ramp={self.force_ramp_time:.2f}s, poll={self.force_poll_period:.2f}s, "
            f"timeout={self.place_contact_timeout:.1f}s, release_ramp={self.release_force_time:.2f}s")
        self.get_logger().info(
            f"  - RT streaming toggle: {self.enable_streaming_service} "
            f"(paused during compliance calls); handover: "
            f"{'move_stop(SSTO) + ' if self.handover_stop_motion else 'NO move_stop, '}"
            f"verify STANDBY/IDLE via {self.get_robot_state_service}, alarms from {self.robot_error_topic}")
        self.get_logger().info(
            f"  - Doosan service call timeout: {self.doosan_service_timeout:.1f}s "
            f"(move_line: {self.move_line_timeout:.1f}s — blocking motion call)")
        self.get_logger().info(
            f"  - Lift: move_line ref={self.lift_move_ref}, +{self.lift_distance:.3f}m")
        self.get_logger().info(
            f"  - Gripper action: {self.gripper_action} (joint={self.gripper_joint_name})")
        self.get_logger().info(f"  - Mask service: {self.mask_service_name}")
        self.get_logger().info(
            f"  - Detach service: {self.detach_service_name}, "
            f"remove mask service: {self.remove_mask_service_name}")
        self.get_logger().info(
            f"  - Approach height: {self.approach_height:.3f}m, "
            f"place height: {self.place_height:.3f}m (object height) along marker Z, "
            f"x offset: {self.approach_x_offset:+.3f}m (marker Y, applied to both)")

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
        self.detach_client = self.create_client(Trigger, self.detach_service_name)
        self.remove_mask_client = self.create_client(RemoveObject, self.remove_mask_service_name)

        # Doosan native services for the compliant place approach and the
        # post-place lift (see DoosanArm for why it owns its own callback group).
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
            set_desired_force_service=self.set_desired_force_service,
            release_force_service=self.release_force_service,
            move_stop_service=self.move_stop_service,
            get_robot_state_service=self.get_robot_state_service,
            get_last_alarm_service=self.get_last_alarm_service,
            robot_error_topic=self.robot_error_topic,
            get_tool_force_service=self.get_tool_force_service,
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
        # release.py explicitly opens/closes it (SetBool) around the window
        # where the base should be free to move — act_move_preplace, up to
        # release_near_threshold from release_approach_target. See
        # _set_release_motion_enable and BaseCommander's module docstring for
        # why this is OR'd with, not a replacement for, its grasper interlock.
        self.release_motion_enable_client = self.create_client(
            SetBool, self.release_motion_enable_service_name)
        # The one condition under which release.py lets the base move during
        # act_move_preplace, pushed by _sync_release_motion_enable: the arm is
        # still far from its approach target (_on_execute_feedback). An
        # edge-detector so the service is only called on an actual change.
        # Losing sight of the marker no longer closes the window: the pose
        # is remembered in marker_fixed_frame (odom) and re-projected into
        # the planning frame on every live-goal tick, so it stays valid
        # while the base drives through an occlusion.
        self._near_target_far = True
        # Last value actually sent to the service, so the edge-detector
        # above cannot double-send the same state.
        self._release_motion_sent = None

        # Release trigger as an action (allows cancel, returns success/message).
        # cancel_callback is required: rclpy's default rejects every cancel
        # request, which would make cancellation silently do nothing.
        self.release_action_server = ActionServer(
            self, TriggerRelease, 'release_trigger', self.release_trigger_cb,
            cancel_callback=self._on_release_cancel_requested)

        # Track in-flight goal handles so we can cancel them.
        self._trajectory_goal_handle = None
        self._gripper_goal_handle = None
        self._cancel_requested = False
        # Set by _fire whenever a 'failure' event fires during a run — see
        # _fire's docstring for why release_trigger_cb needs this instead of
        # just checking self._fsm.state == IDLE.
        self._sequence_failed = False
        # The release_trigger goal currently in flight, if any — used to
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

        # Broadcast the computed target poses as TF (approach/place) so the
        # planned goal can be checked in RViz before triggering.
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
                ACT_MOVE_PREPLACE: self._on_enter_move_preplace,
                ACT_MOVE_PLACE: self._on_enter_move_place,
                ACT_OPEN_GRIPPER: self._on_enter_open_gripper,
                SRV_MASK_PLACED: self._on_enter_mask_placed,
                ACT_RETRACT_ARM: self._on_enter_retract_arm,
            },
            on_state_changed=self._on_fsm_state_changed,
            logger=self.get_logger(),
        )

    def _on_fsm_state_changed(self, state: str) -> None:
        self.state_pub.publish(String(data=state))
        if self._trigger_goal_handle is not None:
            self._trigger_goal_handle.publish_feedback(
                TriggerRelease.Feedback(state=state, message=''))

    async def _fire(self, event: str) -> None:
        """Wrapper around self._fsm.fire that also records whether this run
        ever fired 'failure' — every busy state's 'failure' transition lands
        on the same IDLE state as the normal success path (see TRANSITIONS),
        so release_trigger_cb can't tell the two apart from self._fsm.state
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

    def _on_release_cancel_requested(self, goal_handle):
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

    async def release_trigger_cb(self, goal_handle):
        """Action callback for release trigger. Supports cancel to stop the sequence."""
        if self._fsm.state != IDLE:
            self.get_logger().warn(
                f"Release sequence already running (state={self._fsm.state}), rejecting")
            goal_handle.abort()
            result = TriggerRelease.Result()
            result.success = False
            result.message = "Sequence already running"
            return result

        self.get_logger().info("Trigger received, starting release sequence")
        self._cancel_requested = False
        self._sequence_failed = False
        self._trigger_goal_handle = goal_handle

        try:
            try:
                await self._fire('user_trigger')
            except Exception as e:
                self.get_logger().error(f"Unexpected error in release sequence: {e}")
                self._fsm.reset(IDLE)
                goal_handle.abort()
                result = TriggerRelease.Result()
                result.success = False
                result.message = str(e)
                return result

            result = TriggerRelease.Result()
            if self._cancel_requested:
                self.get_logger().info("Release sequence cancelled")
                result.success = False
                result.message = "Cancelled"
                goal_handle.canceled()
                return result

            result.success = (self._fsm.state == IDLE) and not self._sequence_failed
            result.message = "Release sequence completed" if result.success else "Sequence failed"
            if result.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return result
        finally:
            self._trigger_goal_handle = None

    # ── State machine handlers ───────────────────────────────────────────────

    async def _on_enter_move_preplace(self) -> None:
        if self._marker_tracker.pose is None:
            self.get_logger().warn("No marker received yet, cannot place")
            await self._fire('failure')
            return
        marker_in_plan = self._marker_tracker.transform_to(self.planning_frame)
        if marker_in_plan is None:
            await self._fire('failure')
            return

        approach_pose = self._offset_target_pose(marker_in_plan, self.approach_height)
        place_pose = self._offset_target_pose(marker_in_plan, self.place_height)
        self._run = {
            'marker_in_plan': marker_in_plan,
            'approach_pose': approach_pose,
            'place_pose': place_pose,
        }

        self.get_logger().info("Moving to PREPLACE (approach) pose...")
        # Reset for this attempt: the edge-detector only fires the closing
        # service call on a transition, so a leftover False from a previous
        # attempt must not suppress the open call below.
        self._near_target_far = True
        await self._set_release_motion_enable(True)
        ok = await self.execute_trajectory(
            self._run['approach_pose'],
            live_pose_fn=lambda: self._recompute_live_pose(self.approach_height))
        # Belt-and-suspenders close: normally _on_execute_feedback already
        # closed this once position_error crossed release_near_threshold, but
        # a failure/cancel before that ever happens must not leave the base
        # free to move for the rest of the (non-preplace) sequence.
        await self._set_release_motion_enable(False)
        await self._fire('arm_done' if ok else 'failure')

    async def _on_enter_move_place(self) -> None:
        """Final preplace->place leg: TaskComplianceCtrl (soft), then
        SetDesiredForce pushes down along place_move_ref's Z axis until
        CheckForceCondition reports contact (object/gripper touching the
        table), instead of a fixed-distance MoveLine — a MoveLine toward a
        table position estimated from the marker has no way to stop early if
        that estimate is off, and previously drove the arm into the table
        hard enough to fault the controller into MOTION_HOLD. Force control
        stops exactly at contact regardless of marker error.

        approach_axis_sign flips which way is "down" (see its own
        declare_parameter comment) — the commanded force is signed to match,
        not hardcoded negative. Detection is on the Z axis of place_move_ref
        specifically (not a projection onto approach_dir): simpler, and valid
        as long as place_move_ref is a frame with Z up per its own comment.

        Ordering is load-bearing: force must not be commanded until the
        stiffness ramp has settled (compliance_settle_time), or it is
        accepted and silently dropped (same MoveLine pitfall as before).

        dsr_controller2's RT command streaming is handed over to direct
        service control for the whole compliance window
        (DoosanArm.stop_streaming_for_direct_control: streaming off, soft
        stop, verified STANDBY/IDLE) — see that method for why merely
        switching streaming off leaves the controller in MOTION_HOLD, where
        the force commands were rejected on one run and hung the whole
        driver on the next. Every direct command is then gated on
        assert_ready(), because a rejected command is the *good* outcome:
        the driver was also seen never answering again after one.

        Streaming is re-enabled before leaving this state only once
        release_compliance_ctrl has confirmed the arm is out of compliance
        mode: streamed speedj into a compliant arm is itself a fault
        ("state[TASK_COMPLIANCE_CONTROL] rejected event[eSpeedJ]",
        grasp.py). If compliance could not be released, streaming stays off
        and the log says what the operator has to do.
        """
        self.get_logger().info("Moving to PLACE pose (force-controlled descent)...")
        if not await self._doosan.stop_streaming_for_direct_control(
                stop_motion=self.handover_stop_motion):
            self.get_logger().error("Streaming->direct control handover failed, aborting place descent")
            await self._fire('failure')
            return

        ok = False
        compliance_on = False
        compliance_released = False
        try:
            stx = (
                [self.compliant_translation_stiffness] * 3
                + [self.compliant_rotation_stiffness] * 3
            )
            compliance_on = await self._doosan.task_compliance_ctrl(
                stx, ref=self.place_move_ref, ramp_time=self.compliance_ramp_time)
            if not compliance_on and not self._doosan.responsive:
                # Timed out, not rejected: the driver may well have applied
                # it. Treat as on so the cleanup below still tries to release.
                compliance_on = True

            if compliance_on:
                # Do not remove: force commanded during the stiffness ramp is
                # accepted and then dropped — see compliance_settle_time.
                await self._doosan.sleep(self.compliance_settle_time)
                # The ramp is also when a speedj time-out / MOTION_HOLD would
                # have surfaced as an alarm: check before sending force.
                if self._doosan.error_alarms_since_clear():
                    self.get_logger().error(
                        "Controller alarm during compliance ramp, not sending force")
                else:
                    ok = await self._descend_until_contact()

            # If the driver stopped answering (unacknowledged force command),
            # a MoveStop was seen to get it talking again — try that once
            # before the cleanup calls, or they all just time out in turn.
            if not self._doosan.responsive:
                ok = False
                await self._doosan.try_unblock()

            # Confirm the robot has actually stopped before releasing
            # compliance, on the failure path too: a motion may be running
            # regardless of what the force descent reported.
            if not await self._doosan.wait_motion_idle():
                ok = False

            # Always try to release compliance — even on failure/cancel — so
            # the robot isn't left soft for the next (curobo) motion.
            if compliance_on:
                compliance_released = await self._doosan.release_compliance_ctrl()
                if not compliance_released:
                    self.get_logger().error("Failed to release compliance control!")
            else:
                compliance_released = True
        finally:
            if compliance_released:
                if not await self._doosan.enable_streaming(True):
                    self.get_logger().error("Failed to re-enable RT streaming!")
            else:
                self.get_logger().error(
                    "Compliance NOT confirmed released: leaving RT streaming OFF (streamed speedj "
                    "into a compliant arm faults the controller). "
                    + ("Driver is not answering services — restart dsr_controller2, then "
                       if not self._doosan.responsive else "Then ")
                    + "re-enable streaming: ros2 service call "
                    f"{self.enable_streaming_service} std_srvs/srv/SetBool \"{{data: true}}\"")

        await self._fire('arm_done' if ok else 'failure')

    async def _descend_until_contact(self) -> bool:
        """Push down (SetDesiredForce, place_push_force_n along
        place_move_ref's Z) and poll GetToolForce until the measured force
        has moved away from its pre-push baseline by place_contact_force_n
        (norm of F - F0, translational part), or place_contact_timeout
        elapses.

        Why a baseline delta on the norm, not CheckForceCondition on an
        axis: measured on the robot, the sensor reads several N with nothing
        touching (tool weight / offset: F=[+1.4, +2.7, +1.5] N in DR_TOOL),
        and which axis the contact shows up on depends on the tool frame's
        convention (it landed on tool X, not Z, for a push along base Z).
        Subtracting a baseline taken right before the push removes the
        offset, and the norm removes the axis guess — a 10 N push then read
        as a ~5.5 N delta, from a rest noise well under 1 N.

        The detection threshold must sit well below the commanded push: the
        measured force converges to the setpoint from below (10 N commanded
        read ~7 N) and hovers, so "delta >= setpoint" never latches.
        A delta above place_contact_force_max_n aborts: that's a collision,
        not the table.

        Always tries release_force before returning, success or not, so a
        timeout doesn't leave the arm commanding a steady downward force
        into release_compliance_ctrl. Returns True only on detected contact.
        """
        fd = [0.0] * 6
        fd[2] = -self.approach_axis_sign * self.place_push_force_n
        dir = [0, 0, 1, 0, 0, 0]
        # Note the gate is on STANDBY, not "not MOTION_HOLD": TaskComplianceCtrl
        # is expected to leave robot_state at STANDBY (confirmed on the robot).
        if not await self._doosan.assert_ready("set_desired_force"):
            return False

        baseline = await self._measure_force_baseline()
        if baseline is None:
            self.get_logger().error("Cannot read tool force, aborting place descent")
            return False

        if not await self._doosan.set_desired_force(
                fd, dir, ref=self.place_move_ref, time=self.force_ramp_time, mod=DR_FC_MOD_ABS):
            self.get_logger().error("set_desired_force failed, aborting place descent")
            if not self._doosan.responsive:
                # Timed out, not rejected: the force may be active. Best-effort
                # zero it rather than leave a steady push into the cleanup.
                await self._doosan.release_force(time=self.release_force_time)
            return False

        contact = False
        deadline = self.get_clock().now() + Duration(seconds=self.place_contact_timeout)
        last_force_log = None
        while self.get_clock().now() < deadline:
            if self._cancel_requested:
                self.get_logger().warning("Cancel requested during force descent, stopping push")
                break
            wrench = await self._doosan.get_tool_force(ref=self.contact_check_ref)
            if wrench is None:
                self.get_logger().error("get_tool_force failed during descent, aborting")
                break
            delta = [wrench[i] - baseline[i] for i in range(3)]
            delta_norm = math.sqrt(sum(d * d for d in delta))
            now = self.get_clock().now()
            if last_force_log is None or (now - last_force_log) >= Duration(seconds=self.force_log_period):
                last_force_log = now
                self.get_logger().info(
                    f"tool_force(ref={self.contact_check_ref}): "
                    f"F=[{wrench[0]:+.1f}, {wrench[1]:+.1f}, {wrench[2]:+.1f}]N "
                    f"|dF|={delta_norm:.2f}N (threshold {self.place_contact_force_n:.1f}N)")
            if delta_norm > self.place_contact_force_max_n:
                self.get_logger().error(
                    f"|dF|={delta_norm:.1f}N exceeds place_contact_force_max_n "
                    f"({self.place_contact_force_max_n:.1f}N): collision, aborting place descent")
                break
            if delta_norm >= self.place_contact_force_n:
                contact = True
                self.get_logger().info(
                    f"Contact detected: |dF|={delta_norm:.2f}N >= {self.place_contact_force_n:.1f}N "
                    f"(dF=[{delta[0]:+.1f}, {delta[1]:+.1f}, {delta[2]:+.1f}]N, "
                    f"baseline=[{baseline[0]:+.1f}, {baseline[1]:+.1f}, {baseline[2]:+.1f}]N)")
                break
            await self._doosan.sleep(self.force_poll_period)
        else:
            self.get_logger().error(
                f"No contact detected after {self.place_contact_timeout:.1f}s, aborting place descent")

        if not await self._doosan.release_force(time=self.release_force_time):
            self.get_logger().error("release_force failed")
            contact = False
        return contact

    async def _measure_force_baseline(self):
        """Average contact_baseline_samples GetToolForce readings (arm at
        rest, compliance settled, before the push) — the sensor offset the
        descent's contact delta is measured against. None if unreadable."""
        samples = []
        for _ in range(self.contact_baseline_samples):
            if self._cancel_requested:
                return None
            wrench = await self._doosan.get_tool_force(ref=self.contact_check_ref)
            if wrench is None:
                return None
            samples.append(wrench[:3])
            await self._doosan.sleep(self.force_poll_period)
        baseline = [sum(s[i] for s in samples) / len(samples) for i in range(3)]
        self.get_logger().info(
            f"Force baseline (ref={self.contact_check_ref}, {len(samples)} samples): "
            f"[{baseline[0]:+.2f}, {baseline[1]:+.2f}, {baseline[2]:+.2f}]N")
        return baseline

    async def _on_enter_open_gripper(self) -> None:
        self.get_logger().info("Opening gripper (releasing object)...")
        ok = await self.call_gripper_action(self.gripper_open_position, grasp_object=False)
        if ok:
            ok = await self.call_detach_object_service()
        await self._fire('gripper_open' if ok else 'failure')

    async def _on_enter_mask_placed(self) -> None:
        self.get_logger().info(f"Masking placed object '{self.mask_object_name}' at drop pose...")
        pose = self._run['place_pose']
        ok = await self.call_set_mask_service(pose)
        await self._fire('object_masked' if ok else 'failure')

    async def _on_enter_retract_arm(self) -> None:
        """Post-place lift: a relative linear move (Doosan MoveLine) straight up
        along ref=lift_move_ref's Z axis, not curobo — this is free-space motion
        away from contact, so no compliance wrapper is needed."""
        self.get_logger().info(f"Retracting arm (+{self.lift_distance:.3f}m Z, linear move)...")
        ok = await self._doosan.move_line_relative(
            [-self.lift_distance * 1000.0, 0.0, 0.0,  0.0, 0.0, 0.0], ref=self.lift_move_ref,
            vel=self.lift_move_vel, acc=self.lift_move_acc)
        # Outside compliance move_line does appear to block, but don't rely on
        # it — report the release complete only once the arm has actually stopped.
        ok = await self._doosan.wait_motion_idle() and ok
        if ok:
            # Best-effort: the object is now static at its drop pose, so this
            # is cleanup rather than something the release itself depends on
            # — don't fail the whole sequence over it.
            if not await self.call_remove_mask_service():
                self.get_logger().warn("remove_mask cleanup failed, continuing")
            self.get_logger().info("Release sequence complete")
        await self._fire('arm_done' if ok else 'failure')

    # ── Helpers ──────────────────────────────────────────────────────────────

    # Single source of truth for marker -> target-pose orientation: verified
    # against RViz for this node's flat, upward-facing marker (see module
    # docstring). grasp.py's vertical marker uses offset_along_z's own
    # default instead - the two conventions are not interchangeable, which is
    # exactly the bug this constant fixes: every consumer here (execution,
    # live tracking, TF display) must go through _offset_target_pose so they
    # can never diverge from each other again.
    _APPROACH_ROT_Y = (0.0, -1.0, 0.0, 1.0)

    def _offset_target_pose(self, marker_in_plan: PoseStamped, distance: float) -> Pose:
        return self._marker_tracker.offset_along_z(
            marker_in_plan, distance, self.approach_axis_sign,
            rot_y=self._APPROACH_ROT_Y, x_offset=self.approach_x_offset)

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
        return self._offset_target_pose(marker_in_plan, distance)

    async def _set_release_motion_enable(self, enabled: bool) -> bool:
        """Open/close BaseCommander's release_motion_enable interlock (see
        its module docstring) — the only thing that lets the base move once
        grasp.py's own window (act_move_pregrasp) has long since closed."""
        if not self.release_motion_enable_client.service_is_ready():
            self.get_logger().warn("release_motion_enable service not available")
            return False
        req = SetBool.Request()
        req.data = enabled
        self._release_motion_sent = enabled
        resp = await self.release_motion_enable_client.call_async(req)
        return resp.success

    def _sync_release_motion_enable(self) -> None:
        """Push `_near_target_far` to the release_motion_enable service, but
        only when it actually changes. Fire-and-forget (call_async without
        await) because the caller is a plain feedback callback, not a
        coroutine; the explicit open/close at the leg boundaries stays
        awaited in _set_release_motion_enable."""
        allow = self._near_target_far
        if allow == self._release_motion_sent:
            return
        if not self.release_motion_enable_client.service_is_ready():
            self.get_logger().warn(
                "release_motion_enable service not available",
                throttle_duration_sec=5.0)
            return
        self._release_motion_sent = allow
        req = SetBool.Request()
        req.data = allow
        self.release_motion_enable_client.call_async(req)

    def _publish_target_tf(self):
        """Broadcast approach/place targets as TF for RViz verification."""
        if self._marker_tracker.pose is None:
            return
        marker_in_plan = self._marker_tracker.transform_to(self.planning_frame)
        if marker_in_plan is None:
            return
        approach_pose = self._offset_target_pose(marker_in_plan, self.approach_height)
        place_pose = self._offset_target_pose(marker_in_plan, self.place_height)
        lifted_pose = lift_pose(place_pose, self.lift_distance)
        self._broadcast_pose(approach_pose, 'release_approach_target')
        self._broadcast_pose(place_pose, 'release_place_target')
        self._broadcast_pose(lifted_pose, 'release_lift_target')

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

        Stores the goal_handle so release_trigger_cb can cancel it if requested.

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

        # Store goal_handle so release_trigger_cb (and the feedback callback,
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
                # A real external cancel (release_trigger_cb) takes priority
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
        # release_motion_enable on an actual far<->near transition, not on
        # every feedback tick, and re-evaluated both ways (not latched to
        # near) so the base correctly regains permission if a
        # resync/divergence recovery moves the arm back away from the target.
        far = fb.position_error >= self.release_near_threshold
        if far != self._near_target_far:
            self._near_target_far = far
            self._sync_release_motion_enable()
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

    async def call_gripper_action(self, position: float, grasp_object=False) -> bool:
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
        goal_msg.command.position = float(position)
        self.get_logger().info(f"Gripper command: position={position:.3f}")
        goal_handle = await self.gripper_client.send_goal_async(goal_msg)
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected")
            return False

        # Store goal_handle so release_trigger_cb can cancel if needed.
        self._gripper_goal_handle = goal_handle
        if self._cancel_requested:
            goal_handle.cancel_goal_async()
        try:
            result = (await goal_handle.get_result_async()).result
            if grasp_object:
                self.get_logger().info(f"Gripper done: Grasp object={result.stalled}")
                return result.stalled
            else:
                self.get_logger().info(f"Gripper done: reached_goal={result.reached_goal}")
                return result.reached_goal
        finally:
            self._gripper_goal_handle = None

    async def call_set_mask_service(self, pose: Pose) -> bool:
        """Register the placed object (cuboid, mask_object_dimensions) as
        world collision at `pose`, via curobo_ros's set_mask service, so
        future planning treats it as an obstacle at its new resting pose."""
        if not self.mask_client.service_is_ready():
            self.get_logger().error("set_mask service not available")
            return False

        req = SetMask.Request()
        req.type = SetMask.Request.CUBOID
        req.name = self.mask_object_name
        req.pose = pose
        req.dimensions = Vector3(
            x=self.mask_object_dimensions[0],
            y=self.mask_object_dimensions[1],
            z=self.mask_object_dimensions[2],
        )
        resp = await self.mask_client.call_async(req)
        self.get_logger().info(
            f"set_mask '{self.mask_object_name}': success={resp.success} msg=\"{resp.message}\"")
        return resp.success

    async def call_detach_object_service(self) -> bool:
        """Detach whatever object grasp.py attached to the gripper, via
        curobo_ros's detach_object service (std_srvs/Trigger — only one
        object can be attached at a time, so there's nothing to name)."""
        if not self.detach_client.service_is_ready():
            self.get_logger().error("detach_object service not available")
            return False

        resp = await self.detach_client.call_async(Trigger.Request())
        self.get_logger().info(
            f"detach_object: success={resp.success} msg=\"{resp.message}\"")
        return resp.success

    async def call_remove_mask_service(self) -> bool:
        """Remove the mask_object_name set_mask entry via curobo_ros's
        remove_mask service, now that the object is done being placed."""
        if not self.remove_mask_client.service_is_ready():
            self.get_logger().error("remove_mask service not available")
            return False

        req = RemoveObject.Request()
        req.name = self.mask_object_name
        resp = await self.remove_mask_client.call_async(req)
        self.get_logger().info(
            f"remove_mask '{self.mask_object_name}': success={resp.success} "
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
        self.declare_parameter('trigger_topic', '/brain/release_trigger')

        # Marker pose smoothing: average the last N detections instead of
        # trusting each one raw, so a single bad detection frame doesn't
        # instantly become the place target.
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
        # the base drives while the tag is occluded. Must be TF-connected to
        # planning_frame at all times.
        self.declare_parameter('marker_fixed_frame', 'odom')

        # Planning frame = curobo base_link (target_pose has no frame_id).
        self.declare_parameter('planning_frame', 'dsr01/base_link')
        # Unified "plan + execute" action: moves the arm to target_pose.
        self.declare_parameter('execute_action', '/curobo_trajectory_planner/execute_trajectory')
        # While execute_trajectory's reactive controller (MPC) is servoing toward
        # preplace/place, republish the offset target recomputed from the *live*
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
        # has stalled (position_error plateaus) — see grasp.py's
        # execution_divergence_watchdog.py for the mechanism. Cancelling and
        # resending the goal re-seeds the solver from the real robot state
        # and reliably recovers it; retargeting via mpc_goal alone does not.
        self.declare_parameter('divergence_watchdog_enabled', True)
        self.declare_parameter('divergence_threshold', 0.02)      # m
        self.declare_parameter('divergence_debounce_count', 3)    # feedback samples
        self.declare_parameter('divergence_grace_period', 3.0)    # s after a resync
        self.declare_parameter('divergence_max_resyncs', 3)       # per attempt
        self.declare_parameter('divergence_min_resync_interval', 5.0)  # s

        # Current state-machine state (e.g. "idle", "act_move_place"), as
        # std_msgs/String. Transient-local + published on every transition, so
        # a subscriber started mid-sequence still gets the current value
        # immediately instead of waiting for the next change.
        self.declare_parameter('state_topic', '~/state')

        # SetBool service on BaseCommander (gradient_base_controller) opened
        # while in act_move_preplace and farther than release_near_threshold
        # from release_approach_target, closed otherwise — see
        # _set_release_motion_enable and BaseCommander's module docstring.
        self.declare_parameter(
            'release_motion_enable_service_name',
            '/gradient_base_controller/release_motion_enable')
        # FK position_error (m) below which the arm is considered close enough
        # to release_approach_target that the mobile base must stop moving,
        # even though execute_trajectory's own on_target/arrival tolerance
        # (controller_error_*_threshold above) may not have fired yet — this
        # is a separate, deliberately looser safety margin for the base alone.
        self.declare_parameter('release_near_threshold', 0.10)  # m

        # act_move_place (final preplace->place leg, right before contact) uses
        # the Doosan driver's native compliance instead of curobo, so
        # marker/pose error at contact is absorbed by compliance rather than
        # fought by the free-space MPC: TaskComplianceCtrl (soft), then
        # SetDesiredForce/CheckForceCondition push down until contact is
        # detected (see _descend_until_contact), then ReleaseComplianceCtrl.
        # move_line_service itself is only used by act_retract_arm's lift now.
        self.declare_parameter('move_line_service', '/dsr01/dsr_controller2/motion/move_line')
        self.declare_parameter('task_compliance_ctrl_service', '/dsr01/dsr_controller2/force/task_compliance_ctrl')
        self.declare_parameter('release_compliance_ctrl_service', '/dsr01/dsr_controller2/force/release_compliance_ctrl')
        # Polled before ReleaseComplianceCtrl and before reporting the release
        # complete, so neither happens while the arm is still moving.
        self.declare_parameter('check_motion_service', '/dsr01/dsr_controller2/motion/check_motion')
        # dsr_controller2's RT command streaming must be paused before any
        # compliance call: left on, it keeps streaming commands (observed
        # rejection: internal event 'eSpeedJ') that TASK_COMPLIANCE_CONTROL
        # rejects, which faulted the controller into MOTION_HOLD.
        self.declare_parameter('enable_streaming_service', '/ExecuteTrajectory/enable_streaming')
        self.declare_parameter('set_tool_service', '/dsr01/dsr_controller2/system/set_current_tool')
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
        self.declare_parameter('compliant_translation_stiffness', 500.0)  # N/m, range 0-20000
        self.declare_parameter('compliant_rotation_stiffness', 200.0)     # Nm/rad, range 0-400
        self.declare_parameter('compliance_ramp_time', 0.4)               # s, range 0-1.0
        # MANDATORY dead time between task_compliance_ctrl and the MoveLine:
        # a movel issued while the stiffness is still ramping is ACCEPTED
        # (success=True) and then silently dropped — check_motion never leaves
        # IDLE and the robot doesn't move at all (see grasp.py for the
        # reproduced timings). Must be > compliance_ramp_time.
        self.declare_parameter('compliance_settle_time', 1.0)             # s
        # Reference frame for TaskComplianceCtrl and the SetDesiredForce/
        # CheckForceCondition descent (_descend_until_contact): base-frame,
        # with Z assumed up (see place_contact_force_n below and
        # approach_axis_sign). Assumes planning_frame is the controller's
        # DR_BASE frame; switch to DR_WORLD(2) if your controller (M2.40+)
        # defines a separate world frame that planning_frame actually matches.
        self.declare_parameter('place_move_ref', DR_BASE)
        # Force-controlled descent (_descend_until_contact): pushes down
        # along place_move_ref's Z axis via SetDesiredForce until
        # CheckForceCondition reports the measured force on Z has reached
        # this magnitude (contact with the table) — replaces a fixed-distance
        # MoveLine, which had no way to stop early if the marker-estimated
        # place_height was off and previously drove the arm into the table
        # hard enough to fault the controller into MOTION_HOLD.
        # Commanded push (SetDesiredForce along place_move_ref's Z). Must be
        # clearly above place_contact_force_n: the measured force converges to
        # this setpoint from below and hovers around it, so a detection
        # threshold equal to it never latches (observed on the robot).
        self.declare_parameter('place_push_force_n', 8.0)   # N
        # Frame GetToolForce is read in for the contact delta. Any frame works
        # (the delta is a norm), DR_TOOL keeps the logged numbers comparable
        # to the values measured on the robot.
        self.declare_parameter('contact_check_ref', 1)     # DR_TOOL
        # GetToolForce readings averaged for the pre-push baseline.
        self.declare_parameter('contact_baseline_samples', 5)
        # GetToolForce is logged at this period during the descent so the
        # real numbers are available to tune the thresholds.
        self.declare_parameter('force_log_period', 0.25)   # s
        self.declare_parameter('get_tool_force_service', '/dsr01/dsr_controller2/aux_control/get_tool_force')
        self.declare_parameter('place_contact_force_n', 4.0)     # N
        # Upper bound passed to CheckForceCondition alongside
        # place_contact_force_n — guards against reading a spurious huge
        # force as "in range" (CheckForceCondition takes a [min, max] band).
        self.declare_parameter('place_contact_force_max_n', 50.0)  # N
        # SetDesiredForce's own ramp time for the commanded force to take
        # effect (0~1.0s per the driver's own range).
        self.declare_parameter('force_ramp_time', 0.4)            # s
        self.declare_parameter('force_poll_period', 0.05)         # s
        # Upper bound on _descend_until_contact: if no contact is detected by
        # then (table absent, marker/table height badly off, force sensor
        # fault...), the descent fails cleanly (force released, compliance
        # released) rather than pushing indefinitely.
        self.declare_parameter('place_contact_timeout', 5.0)      # s
        # ReleaseForce's own ramp time back to zero commanded force, called
        # once contact is detected (or the descent times out).
        self.declare_parameter('release_force_time', 0.2)         # s
        self.declare_parameter('set_desired_force_service', '/dsr01/dsr_controller2/force/set_desired_force')
        self.declare_parameter('release_force_service', '/dsr01/dsr_controller2/force/release_force')
        # Streaming->direct-control handover (DoosanArm.
        # stop_streaming_for_direct_control): once ExecuteTrajectory stops
        # publishing speedj_rt_stream, the controller's last speedj (0.1 s
        # validity) expires with alarm 1215 and leaves it in MOTION_HOLD,
        # where SetDesiredForce was rejected (1903) on one run and hung the
        # driver outright on the next. With handover_stop_motion the speedj
        # session is instead ended by an explicit soft stop (MoveStop
        # DR_SSTO) sent within that window, and the controller is verified
        # STANDBY/IDLE (GetRobotState + CheckMotion) with no ERROR alarm on
        # robot_error_topic before any compliance/force call is sent. Set
        # handover_stop_motion false only to reproduce the alarm on purpose.
        self.declare_parameter('handover_stop_motion', True)
        self.declare_parameter('move_stop_service', '/dsr01/dsr_controller2/motion/move_stop')
        self.declare_parameter('get_robot_state_service', '/dsr01/dsr_controller2/system/get_robot_state')
        self.declare_parameter('get_last_alarm_service', '/dsr01/dsr_controller2/system/get_last_alarm')
        self.declare_parameter('robot_error_topic', '/dsr01/error')

        # act_retract_arm (post-place lift) also uses MoveLine directly instead of
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
        # fully-closed default (2f_85.ros2_control.xacro). Kept for parity
        # with grasp.py even though this node only ever opens the gripper.
        self.declare_parameter('gripper_closed_position', 0.79)
        # Gripper not wired yet: just log open/close instead of calling the action.
        self.declare_parameter('simulate_gripper', False)

        # Object masking: registers the placed object's known geometry as
        # world collision (curobo_ros robot_segmentation's set_mask service)
        # at its new resting pose, once released.
        self.declare_parameter('mask_service_name', '/set_mask')
        self.declare_parameter('mask_object_name', 'grasp_target')
        self.declare_parameter('mask_object_dimensions', [0.1, 0.1, 0.05])
        # Detaches the object grasp.py attached to the gripper, right after
        # the gripper opens (std_srvs/Trigger: only one object can be
        # attached at a time, so no name is needed).
        self.declare_parameter('detach_service_name', '/curobo_trajectory_planner/detach_object')
        # Once the arm has retracted, the object no longer needs the
        # set_mask exclusion that let the gripper approach/release it —
        # best-effort cleanup so that region goes back to normal collision
        # checking against its real (now static) geometry.
        self.declare_parameter('remove_mask_service_name', '/remove_mask')

        # Approach geometry: offsets along the (flat, upward-facing) marker Z axis (metres).
        # position = marker_pos + approach_axis_sign * distance * marker_Z
        self.declare_parameter('approach_height', 0.20)
        # Object height: the gripper releases the object here so its base
        # lands on the table. Tune empirically by watching the 'place_target'
        # TF frame against the real table surface in RViz.
        self.declare_parameter('place_height', 0.05)
        # +1.0: the marker lies flat with its Z axis already pointing up, so
        # approach/place poses sit above it. Flip to -1.0 if the tag is
        # mounted facing down.
        self.declare_parameter('approach_axis_sign', 1.0)
        # Extra distance (metres) added, along the marker's raw X axis, to
        # every _offset_target_pose call (approach_height and place_height
        # alike) — see offset_along_z's docstring for why marker_X, not
        # marker_Z, is the axis that reads as "Z" against the published
        # TF/RViz orientation once _APPROACH_ROT_Y is applied. A uniform
        # calibration correction that also propagates to the post-place lift
        # (it moves relative to place_pose). 0.0 = no change.
        self.declare_parameter('approach_x_offset', 0.0)
        # Vertical (world/planning-frame Z) lift distance for act_retract_arm.
        self.declare_parameter('lift_distance', 0.10)

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
        self.release_motion_enable_service_name = self.get_parameter(
            'release_motion_enable_service_name').value
        self.release_near_threshold = self.get_parameter('release_near_threshold').value
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
        self.doosan_service_timeout = self.get_parameter('doosan_service_timeout').value
        self.move_line_timeout = self.get_parameter('move_line_timeout').value
        self.motion_poll_period = self.get_parameter('motion_poll_period').value
        self.motion_idle_timeout = self.get_parameter('motion_idle_timeout').value
        self.compliant_translation_stiffness = self.get_parameter('compliant_translation_stiffness').value
        self.compliant_rotation_stiffness = self.get_parameter('compliant_rotation_stiffness').value
        self.compliance_ramp_time = self.get_parameter('compliance_ramp_time').value
        self.compliance_settle_time = self.get_parameter('compliance_settle_time').value
        self.place_move_ref = self.get_parameter('place_move_ref').value
        self.place_contact_force_n = self.get_parameter('place_contact_force_n').value
        self.place_push_force_n = self.get_parameter('place_push_force_n').value
        self.contact_check_ref = self.get_parameter('contact_check_ref').value
        self.contact_baseline_samples = self.get_parameter('contact_baseline_samples').value
        self.force_log_period = self.get_parameter('force_log_period').value
        self.get_tool_force_service = self.get_parameter('get_tool_force_service').value
        self.place_contact_force_max_n = self.get_parameter('place_contact_force_max_n').value
        self.force_ramp_time = self.get_parameter('force_ramp_time').value
        self.force_poll_period = self.get_parameter('force_poll_period').value
        self.place_contact_timeout = self.get_parameter('place_contact_timeout').value
        self.release_force_time = self.get_parameter('release_force_time').value
        self.set_desired_force_service = self.get_parameter('set_desired_force_service').value
        self.release_force_service = self.get_parameter('release_force_service').value
        self.handover_stop_motion = self.get_parameter('handover_stop_motion').value
        self.move_stop_service = self.get_parameter('move_stop_service').value
        self.get_robot_state_service = self.get_parameter('get_robot_state_service').value
        self.get_last_alarm_service = self.get_parameter('get_last_alarm_service').value
        self.robot_error_topic = self.get_parameter('robot_error_topic').value
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
        self.detach_service_name = self.get_parameter('detach_service_name').value
        self.remove_mask_service_name = self.get_parameter('remove_mask_service_name').value
        self.approach_height = self.get_parameter('approach_height').value
        self.place_height = self.get_parameter('place_height').value
        self.approach_axis_sign = self.get_parameter('approach_axis_sign').value
        self.lift_distance = self.get_parameter('lift_distance').value
        self.approach_x_offset = self.get_parameter('approach_x_offset').value


def main(args=None):
    rclpy.init(args=args)
    node = ObjectReleaser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
