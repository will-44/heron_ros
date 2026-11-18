#!/usr/bin/env python3
"""Intermediary between grasp.py and release.py: a small event-driven state
machine that drives a full pick-and-place cycle by calling each node's own
action server in sequence, rather than knowing anything about markers, TF, or
the arm itself. It also brackets the cycle with the arm's init configuration
and the mobile base's gradient controller.

  idle --user_trigger--> act_home_enter
  act_home_enter (planner->joint, send init config, planner->lbfgs, enable
    base) --success--> act_grasp
  act_grasp (calls grasp.py's grasp_trigger action) --success--> act_release
  act_release (calls release.py's release_trigger action) --success--> act_home_exit
  act_home_exit (planner->lbfgs, send final_pose as a Cartesian target_pose,
    disable base) --success--> idle
  (any busy state) --failure--> idle

act_home_exit's move to final_pose is Cartesian (target_pose, lbfgs planner)
rather than the open-loop joint-space move act_home_enter uses to reach
init_joint_positions — the two ends of the cycle are not symmetric.

grasp.py and release.py are each already self-contained state machines
(explicit TRANSITIONS table, own action server, own ~/state topic) — brain.py
does not duplicate any of that. It only sequences them around the home
positioning: if a busy state fails, the cycle aborts straight to idle without
attempting the remaining steps (including the exit homing) — a failure is
expected to need operator attention before the arm/base are touched again.

`self._fsm.state` doubles as the busy flag: only `idle` accepts
`user_trigger`, so a trigger received mid-cycle is ignored.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import String
from std_srvs.srv import SetBool
from geometry_msgs.msg import Pose
from curobo_msgs.srv import SetPlanner
from curobo_msgs.action import SendTrajectory
from heron_interface.action import TriggerGrasp, TriggerRelease, TriggerPickPlace

from ..scripts.execution_divergence_watchdog import ExecutionDivergenceWatchdog
from ..scripts.state_machine import StateMachine

# States (plain strings, not Enum: a single place to touch when adding one).
IDLE = 'idle'
ACT_HOME_ENTER = 'act_home_enter'
ACT_GRASP = 'act_grasp'
ACT_RELEASE = 'act_release'
ACT_HOME_EXIT = 'act_home_exit'

_BUSY_STATES = (ACT_HOME_ENTER, ACT_GRASP, ACT_RELEASE, ACT_HOME_EXIT)

TRANSITIONS = {
    (IDLE, 'user_trigger'): ACT_HOME_ENTER,
    (ACT_HOME_ENTER, 'success'): ACT_GRASP,
    (ACT_GRASP, 'success'): ACT_RELEASE,
    (ACT_RELEASE, 'success'): ACT_HOME_EXIT,
    (ACT_HOME_EXIT, 'success'): IDLE,
    # Generic abort: any busy state can bail out back to idle (e.g. grasp
    # failing means there is nothing to release, so this also short-circuits
    # straight past act_release and act_home_exit).
    **{(s, 'failure'): IDLE for s in _BUSY_STATES},
}


class Brain(Node):
    """Sequence grasp.py then release.py via their own action servers."""

    def __init__(self):
        super().__init__('brain')

        self._load_parameters()

        self._setup_interfaces()
        self._setup_state_machine()

        self.get_logger().info("Brain initialized")
        self.get_logger().info(
            f"  - Grasp action: {self.grasp_action_name}, "
            f"release action: {self.release_action_name}")
        self.get_logger().info(
            f"  - set_planner: {self.set_planner_service_name} "
            f"(joint_space={self.planner_type_joint_space}, lbfgs={self.planner_type_lbfgs}), "
            f"home execute action: {self.curobo_execute_action}")
        self.get_logger().info(
            f"  - final_pose (act_home_exit, Cartesian, lbfgs): "
            f"position=({self.final_pose.position.x:.3f}, {self.final_pose.position.y:.3f}, "
            f"{self.final_pose.position.z:.3f}), orientation=("
            f"{self.final_pose.orientation.x:.3f}, {self.final_pose.orientation.y:.3f}, "
            f"{self.final_pose.orientation.z:.3f}, {self.final_pose.orientation.w:.3f})")
        self.get_logger().info(
            f"  - Base enable service: {self.base_enable_service_name} "
            f"(base_control={self.base_control}"
            f"{'' if self.base_control else ', service never called'})")
        self.get_logger().info(f"  - Pick-and-place action server: pick_place_trigger")
        self.get_logger().info(f"  - State topic: {self.state_topic} (std_msgs/String, latched)")
        self.get_logger().info(
            f"  - Outcome topic: {self.outcome_topic} (std_msgs/String, latched)")

    def _setup_interfaces(self):
        self.grasp_client = ActionClient(self, TriggerGrasp, self.grasp_action_name)
        self.release_client = ActionClient(self, TriggerRelease, self.release_action_name)

        # Home positioning: switch cuRobo's planner, move the arm via the
        # same unified plan+execute action grasp.py uses (target_joint_positions
        # instead of target_pose), then switch back, bracketing the cycle.
        self.set_planner_client = self.create_client(SetPlanner, self.set_planner_service_name)
        self.home_execute_client = ActionClient(self, SendTrajectory, self.curobo_execute_action)
        # Mobile base's gradient controller: enabled only while the base
        # should be free to move (act_home_enter..act_home_exit).
        self.base_enable_client = self.create_client(SetBool, self.base_enable_service_name)

        # Transient-local so a late subscriber gets the current state
        # immediately without waiting for the next transition.
        state_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.state_pub = self.create_publisher(String, self.state_topic, state_qos)

        # Outcome of the cycle that just ended, as "<success|failure|cancelled>|<message>".
        # The state topic alone cannot carry it: every terminal transition — success,
        # abort and cancel alike — lands on the same `idle` string, so a subscriber
        # watching only the state cannot tell a completed cycle from an aborted one.
        # Latched like the state topic, and published *before* the idle state so a
        # recorder keying episodes on the state topic has it in hand at the boundary.
        self.outcome_pub = self.create_publisher(String, self.outcome_topic, state_qos)

        # Pick-and-place trigger as an action (allows cancel, returns
        # success/message) — same shape as grasp.py/release.py's own trigger
        # actions. cancel_callback is required: rclpy's default rejects every
        # cancel request, which would make cancellation silently do nothing.
        self.pick_place_action_server = ActionServer(
            self, TriggerPickPlace, 'pick_place_trigger', self.pick_place_trigger_cb,
            cancel_callback=self._on_pick_place_cancel_requested)

        # Track whichever sub-goal is currently in flight so a cancel request
        # can be forwarded to it.
        self._grasp_goal_handle = None
        self._release_goal_handle = None
        self._home_goal_handle = None
        self._cancel_requested = False
        # The pick_place_trigger goal currently in flight, if any — used to
        # publish feedback on every state-machine transition of brain's own
        # cycle, and to forward every transition reported by whichever
        # sub-sequence (grasp/release) is currently running.
        self._pick_place_goal_handle = None

        # Set by _on_execute_feedback once on_target=True is seen, so
        # execute_trajectory knows a cancel-on-success is in flight. curobo's
        # execute_trajectory action, under LBFGS, is continuous servoing
        # (TRACKING) and never ends the goal on its own — see
        # _on_execute_feedback's docstring. Without this, _call_move_to_final_pose
        # (the Cartesian, lbfgs, act_home_exit move) hangs on get_result_async()
        # forever, exactly like the un-patched release.py/grasp.py did.
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

    def _setup_state_machine(self):
        # Run-scoped data (the outcome of the current cycle), reset at
        # user_trigger and read once the machine is back at idle.
        self._run = {}
        self._fsm = StateMachine(
            transitions=TRANSITIONS,
            initial_state=IDLE,
            handlers={
                ACT_HOME_ENTER: self._on_enter_act_home_enter,
                ACT_GRASP: self._on_enter_act_grasp,
                ACT_RELEASE: self._on_enter_act_release,
                ACT_HOME_EXIT: self._on_enter_act_home_exit,
            },
            on_state_changed=self._on_fsm_state_changed,
            logger=self.get_logger(),
        )

    def _on_fsm_state_changed(self, state: str) -> None:
        # Outcome first, state second: the two are separate topics with no
        # inter-topic ordering guarantee, but publishing in this order makes the
        # common case (recorder sees the outcome before, or with, the idle state)
        # the likely one rather than the racy one.
        if state == IDLE and self._run:
            if self._cancel_requested:
                # The cancel flag is set before the sub-goals are cancelled, so it
                # is already true by the time the resulting 'failure' transition
                # brings the machine back here.
                self._publish_outcome('cancelled', 'Cancelled')
            else:
                self._publish_outcome(
                    'success' if self._run.get('final_success', False) else 'failure',
                    self._run.get('final_message', ''))
        self.state_pub.publish(String(data=state))
        self._publish_pick_place_feedback(state)

    def _publish_outcome(self, outcome: str, message: str) -> None:
        """Announce how the cycle that just ended finished. See `outcome_pub`."""
        self.outcome_pub.publish(String(data=f'{outcome}|{message}'))

    def _publish_pick_place_feedback(self, brain_state: str, sub_state: str = '') -> None:
        """Publish combined feedback on the in-flight pick_place_trigger goal,
        if any: "<brain_state>" alone, or "<brain_state>:<sub_state>" while
        act_grasp/act_release is forwarding grasp.py's/release.py's own
        state-machine feedback."""
        if self._pick_place_goal_handle is None:
            return
        state = f"{brain_state}:{sub_state}" if sub_state else brain_state
        self._pick_place_goal_handle.publish_feedback(TriggerPickPlace.Feedback(state=state))

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_pick_place_cancel_requested(self, goal_handle):
        """Accept the cancel request and cancel whatever sub-goal is in flight.

        Cancelling the sub-goal makes the awaiting _call_grasp/_call_release
        return success=False, which the state machine already turns into a
        'failure' transition back to idle — no separate abort path needed.
        """
        self.get_logger().info("Cancel requested, cancelling in-flight sub-goal(s)...")
        self._cancel_requested = True
        if self._grasp_goal_handle is not None:
            self._grasp_goal_handle.cancel_goal_async()
        if self._release_goal_handle is not None:
            self._release_goal_handle.cancel_goal_async()
        if self._home_goal_handle is not None:
            self._home_goal_handle.cancel_goal_async()
        return CancelResponse.ACCEPT

    async def pick_place_trigger_cb(self, goal_handle):
        """Action callback for the pick-and-place trigger. Supports cancel to
        stop the cycle."""
        if self._fsm.state != IDLE:
            self.get_logger().warn(
                f"Pick-and-place cycle already running (state={self._fsm.state}), rejecting")
            goal_handle.abort()
            result = TriggerPickPlace.Result()
            result.success = False
            result.message = "Cycle already running"
            return result

        self.get_logger().info("Trigger received, starting pick-and-place cycle")
        self._cancel_requested = False
        self._run = {'final_success': False, 'final_message': ''}
        self._pick_place_goal_handle = goal_handle

        try:
            try:
                await self._fsm.fire('user_trigger')
            except Exception as e:
                self.get_logger().error(f"Unexpected error in pick-and-place cycle: {e}")
                self._fsm.reset(IDLE)
                goal_handle.abort()
                result = TriggerPickPlace.Result()
                result.success = False
                result.message = str(e)
                return result

            result = TriggerPickPlace.Result()
            if self._cancel_requested:
                self.get_logger().info("Pick-and-place cycle cancelled")
                result.success = False
                result.message = "Cancelled"
                goal_handle.canceled()
                return result

            result.success = self._run.get('final_success', False)
            result.message = self._run.get('final_message', '')
            if result.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return result
        finally:
            self._pick_place_goal_handle = None

    # ── State machine handlers ───────────────────────────────────────────────

    async def _on_enter_act_home_enter(self) -> None:
        self.get_logger().info("Positioning arm at init configuration (start of cycle)...")
        ok, message = await self._go_to_init_configuration()
        if ok:
            ok, base_message = await self._call_set_base_enabled(True)
            if not ok:
                message = f"Base enable failed: {base_message}"
        if not ok:
            self._run['final_message'] = f"Home (start) failed: {message}"
        self.get_logger().info(f"Home (start) {'succeeded' if ok else 'failed'}: {message}")
        await self._fire('success' if ok else 'failure')

    async def _on_enter_act_grasp(self) -> None:
        self.get_logger().info(f"Calling grasp sequence ({self.grasp_action_name})...")
        ok, message = await self._call_grasp()
        if not ok:
            self._run['final_message'] = f"Grasp failed: {message}"
        self.get_logger().info(f"Grasp sequence {'succeeded' if ok else 'failed'}: {message}")
        await self._fire('success' if ok else 'failure')

    async def _on_enter_act_release(self) -> None:
        self.get_logger().info(f"Calling release sequence ({self.release_action_name})...")
        ok, message = await self._call_release()
        if not ok:
            self._run['final_message'] = f"Release failed: {message}"
        self.get_logger().info(f"Release sequence {'succeeded' if ok else 'failed'}: {message}")
        await self._fire('success' if ok else 'failure')

    async def _on_enter_act_home_exit(self) -> None:
        self.get_logger().info("Positioning arm at final pose (end of cycle)...")
        ok, message = await self._go_to_final_pose()
        if ok:
            ok, base_message = await self._call_set_base_enabled(False)
            if not ok:
                message = f"Base disable failed: {base_message}"
        self._run['final_success'] = ok
        self._run['final_message'] = (
            "Pick-and-place cycle completed" if ok else f"Home (end) failed: {message}")
        self.get_logger().info(f"Home (end) {'succeeded' if ok else 'failed'}: {message}")
        await self._fire('success' if ok else 'failure')

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _on_grasp_feedback(self, feedback_msg) -> None:
        self._publish_pick_place_feedback(self._fsm.state, feedback_msg.feedback.state)

    def _on_release_feedback(self, feedback_msg) -> None:
        self._publish_pick_place_feedback(self._fsm.state, feedback_msg.feedback.state)

    async def _fire(self, event: str) -> None:
        """Single point where a cancel takes effect: once _cancel_requested
        is set, a 'success' from a busy state is turned into 'failure', so a
        step that happened to complete after the cancel (home motion already
        on target, grasp/release sub-goal that finished before the cancel
        reached it) ends the cycle instead of starting the next step —
        without this, a cancel during act_home_enter would let a whole grasp
        begin. Mirrors grasp.py/release.py's _fire."""
        if event != 'failure' and self._cancel_requested and self._fsm.state != IDLE:
            self.get_logger().info(
                f"Cancel requested: '{event}' from {self._fsm.state} overridden by 'failure'")
            event = 'failure'
        await self._fsm.fire(event)

    async def _call_grasp(self) -> tuple:
        """Send an empty TriggerGrasp goal to grasp.py and await its result.
        Returns (success, message)."""
        if not self.grasp_client.server_is_ready():
            return False, "grasp_trigger action not available"

        goal_handle = await self.grasp_client.send_goal_async(
            TriggerGrasp.Goal(), feedback_callback=self._on_grasp_feedback)
        if goal_handle is None or not goal_handle.accepted:
            return False, "Grasp goal rejected"

        self._grasp_goal_handle = goal_handle
        if self._cancel_requested:
            # Cancel arrived before the handle existed: the cancel callback
            # couldn't reach this goal, so do it here.
            goal_handle.cancel_goal_async()
        try:
            result = (await goal_handle.get_result_async()).result
            return result.success, result.message
        finally:
            self._grasp_goal_handle = None

    async def _call_release(self) -> tuple:
        """Send an empty TriggerRelease goal to release.py and await its
        result. Returns (success, message)."""
        if not self.release_client.server_is_ready():
            return False, "release_trigger action not available"

        goal_handle = await self.release_client.send_goal_async(
            TriggerRelease.Goal(), feedback_callback=self._on_release_feedback)
        if goal_handle is None or not goal_handle.accepted:
            return False, "Release goal rejected"

        self._release_goal_handle = goal_handle
        if self._cancel_requested:
            # Cancel arrived before the handle existed: the cancel callback
            # couldn't reach this goal, so do it here.
            goal_handle.cancel_goal_async()
        try:
            result = (await goal_handle.get_result_async()).result
            return result.success, result.message
        finally:
            self._release_goal_handle = None

    async def _go_to_init_configuration(self) -> tuple:
        """Switch cuRobo to the joint-space planner, move the arm to
        init_joint_positions, then switch to the lbfgs planner. Returns
        (success, message); stops at the first failing step."""
        ok, message = await self._call_set_planner(self.planner_type_joint_space)
        if not ok:
            return False, f"set_planner(joint_space) failed: {message}"

        ok, message = await self._call_move_to_init_config()
        if not ok:
            return False, f"move to init config failed: {message}"

        ok, message = await self._call_set_planner(self.planner_type_lbfgs)
        if not ok:
            return False, f"set_planner(lbfgs) failed: {message}"

        return True, "Arm at init configuration"

    async def _call_set_planner(self, planner_type: int) -> tuple:
        """Switch cuRobo's active trajectory planner. Returns (success, message)."""
        if not self.set_planner_client.service_is_ready():
            return False, "set_planner service not available"

        req = SetPlanner.Request()
        req.planner_type = planner_type
        resp = await self.set_planner_client.call_async(req)
        return resp.success, resp.message

    async def _call_move_to_init_config(self) -> tuple:
        """Send an open-loop joint-space goal to init_joint_positions via
        cuRobo's unified plan+execute action. Returns (success, message)."""
        if not self.home_execute_client.server_is_ready():
            return False, "execute_trajectory action not available"

        goal = SendTrajectory.Goal()
        goal.target_joint_positions = self.init_joint_positions
        goal.allow_cached = False
        goal_handle = await self.home_execute_client.send_goal_async(goal)
        if goal_handle is None or not goal_handle.accepted:
            return False, "Home goal rejected"

        self._home_goal_handle = goal_handle
        if self._cancel_requested:
            # Cancel arrived before the handle existed: the cancel callback
            # couldn't reach this goal, so do it here.
            goal_handle.cancel_goal_async()
        try:
            result = (await goal_handle.get_result_async()).result
            return result.success, result.message
        finally:
            self._home_goal_handle = None

    async def _go_to_final_pose(self) -> tuple:
        """Switch cuRobo to the lbfgs planner and move the arm to final_pose
        via a Cartesian target_pose goal. Unlike _go_to_init_configuration,
        this never goes through the joint-space planner — the end-of-cycle
        move is a reactive Cartesian move, not an open-loop joint move.
        Returns (success, message); stops at the first failing step."""
        ok, message = await self._call_set_planner(self.planner_type_lbfgs)
        if not ok:
            return False, f"set_planner(lbfgs) failed: {message}"

        ok, message = await self._call_move_to_final_pose()
        if not ok:
            return False, f"move to final pose failed: {message}"

        return True, "Arm at final pose"

    async def _call_move_to_final_pose(self) -> tuple:
        """Send a Cartesian target_pose goal to final_pose via cuRobo's
        unified plan+execute action (lbfgs planner). Returns (success, message).

        Goes through execute_trajectory (not a plain send-goal/await-result,
        unlike _call_move_to_init_config) because under lbfgs curobo's
        execute_trajectory server is continuous servoing (TRACKING) and never
        ends the goal on its own — see execute_trajectory's docstring.
        """
        goal = SendTrajectory.Goal()
        goal.target_pose = self.final_pose
        goal.allow_cached = False  # always plan+execute fresh for this target
        return await self.execute_trajectory(goal)

    async def execute_trajectory(self, goal: SendTrajectory.Goal) -> tuple:
        """Send goal to home_execute_client and await a final outcome.

        Duplicates release.py's/grasp.py's execute_trajectory pattern: under
        lbfgs, curobo's execute_trajectory server is continuous servoing
        (TRACKING) and never ends the goal on its own, so a plain
        get_result_async() await (as _call_move_to_init_config still does for
        its open-loop joint-space move) hangs forever even once the arm has
        physically arrived — we must watch feedback ourselves and cancel the
        goal as soon as on_target is reported, treating that as success.

        The divergence watchdog (see ExecutionDivergenceWatchdog) covers the
        whole attempt, including any resyncs within it: if it fires mid-goal,
        the goal is cancelled and resent unchanged (brain has no live pose to
        retarget from, unlike release.py/grasp.py's marker-tracking case) up
        to divergence_max_resyncs times, instead of failing outright.

        Returns (success, message).
        """
        if not self.home_execute_client.server_is_ready():
            return False, "execute_trajectory action not available"

        self._divergence_watchdog.reset()
        while True:
            outcome = await self._send_and_await_goal(goal)
            if outcome is not None:
                return outcome
            # outcome is None: the watchdog requested a resync. Nothing to
            # retarget (static goal), so just resend as-is.
            self.get_logger().warn(
                f"Divergence watchdog: resending goal "
                f"(resync {self._divergence_watchdog.resync_count}/{self.divergence_max_resyncs})")

    async def _send_and_await_goal(self, goal: SendTrajectory.Goal):
        """One send-goal/await-result cycle.

        Returns a (success, message) final outcome, or None if
        execute_trajectory should resend the goal (the divergence watchdog
        fired and the attempt wasn't otherwise externally cancelled).
        """
        self.get_logger().info(f"Executing trajectory: target=({goal}")
        self._on_target_reached = False
        self._resync_requested = False
        goal_handle = await self.home_execute_client.send_goal_async(
            goal, feedback_callback=self._on_execute_feedback
        )
        if goal_handle is None or not goal_handle.accepted:
            return False, "Home goal rejected"

        self._home_goal_handle = goal_handle
        if self._cancel_requested:
            # Cancel arrived before the handle existed: the cancel callback
            # couldn't reach this goal, so do it here.
            goal_handle.cancel_goal_async()
        try:
            result_response = await goal_handle.get_result_async()
            if self._on_target_reached:
                # Server is in continuous servoing (TRACKING) and never ends
                # the goal on its own — we cancel it ourselves as soon as
                # feedback reports on_target, and treat that as success.
                self.get_logger().info("On target: trajectory cancelled as success")
                return True, "Arm at target"
            if self._resync_requested and not self._cancel_requested:
                # A real external cancel (pick_place_trigger_cb) takes priority
                # over a watchdog-driven resync: fall through to the normal
                # result handling below instead of looping again.
                return None
            result = result_response.result
            if result.success:
                self.get_logger().info(f"Trajectory succeeded: {result.message}")
                return True, result.message
            self.get_logger().error(f"Trajectory failed: {result.message}")
            return False, result.message
        finally:
            self._home_goal_handle = None

    def _on_execute_feedback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f"  [{fb.state}] progress={fb.step_progression:.2f} "
            f"err={fb.position_error:.4f}m on_target={fb.on_target} "
            f"controller_err={fb.controller_position_error:.4f}m",
            throttle_duration_sec=1.0,
        )

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
        if on_target and not self._on_target_reached and self._home_goal_handle is not None:
            self._on_target_reached = True
            self.get_logger().info("on_target=True, cancelling trajectory goal")
            self._home_goal_handle.cancel_goal_async()
            return

        if (self.divergence_watchdog_enabled and not self._resync_requested
                and not self._on_target_reached and self._home_goal_handle is not None):
            now_s = self.get_clock().now().nanoseconds * 1e-9
            outcome = self._divergence_watchdog.on_feedback(
                fb.position_error, fb.controller_position_error, now_s)
            if outcome == ExecutionDivergenceWatchdog.RESYNC:
                self.get_logger().warn(
                    f"Divergence watchdog: position_error={fb.position_error:.4f}m vs "
                    f"controller_position_error={fb.controller_position_error:.4f}m "
                    "— cancelling and resending goal")
                self._resync_requested = True
                self._home_goal_handle.cancel_goal_async()
            elif outcome == ExecutionDivergenceWatchdog.BUDGET_EXHAUSTED:
                self.get_logger().error(
                    f"Divergence watchdog: still diverging after "
                    f"{self.divergence_max_resyncs} resyncs — failing this attempt")
                self._home_goal_handle.cancel_goal_async()

    async def _call_set_base_enabled(self, enabled: bool) -> tuple:
        """Enable/disable the mobile base's gradient controller. Returns
        (success, message)."""
        # `base_control:=false` is the control arm of the with/without-base
        # experiment: the controller node stays launched (same GPU load, same
        # debug topics, same logging) but is never enabled, so it publishes only
        # zeros. Skipping the call rather than letting it fail matters — a failed
        # enable aborts act_home_enter and with it the whole cycle, which would
        # make the "without base control" condition impossible to run.
        if not self.base_control:
            return True, 'base control disabled by parameter'

        if not self.base_enable_client.service_is_ready():
            return False, "base enable service not available"

        req = SetBool.Request()
        req.data = enabled
        resp = await self.base_enable_client.call_async(req)
        return resp.success, resp.message

    def _load_parameters(self):
        # Action names exposed by grasp.py/release.py — relative, so they
        # resolve against brain's own namespace (matching the pattern of
        # grasp.py/release.py themselves being launched unnamespaced).
        self.declare_parameter('grasp_action_name', 'grasp_trigger')
        self.declare_parameter('release_action_name', 'release_trigger')

        # Home positioning (act_home_enter/act_home_exit): cuRobo's
        # switch-planner service and unified plan+execute action (same one
        # grasp.py uses, via target_joint_positions instead of target_pose).
        self.declare_parameter('set_planner_service_name', '/curobo_trajectory_planner/set_planner')
        self.declare_parameter('curobo_execute_action', '/curobo_trajectory_planner/execute_trajectory')
        # curobo_msgs/srv/SetPlanner planner_type values for this robot's
        # deployment (not the enum names in SetPlanner.srv/its docstring).
        self.declare_parameter('planner_type_joint_space', 5)
        self.declare_parameter('planner_type_lbfgs', 2)
        # Arm's init/home joint configuration (radians, joint_1..joint_6),
        # sent open-loop via the joint-space planner at the start and end of
        # every pick-and-place cycle.
        self.declare_parameter('init_joint_positions', [
            -0.02840041182935238,
            -1.0044194459915161,
            1.897898554801941,
            3.1413369178771973,
            -1.2819790840148926,
            0.028416428714990616,
        ])

        # Arm's end-of-cycle pose (act_home_exit), sent as a Cartesian
        # target_pose via the lbfgs planner rather than a joint-space move —
        # see _go_to_final_pose. target_pose carries no header; assumed frame
        # is dsr01/base_link, matching grasp.py/release.py's planning_frame.
        self.declare_parameter('final_pose_position', [0.307, 0.027, 0.682])
        self.declare_parameter('final_pose_orientation', [0.707, 0.0, 0.707, 0.0])

        # Fallback arrival check for planners whose on_target never fires
        # (e.g. hold_count/orientation gating never converges): if the
        # solver's own controller_position_error agrees with the FK-measured
        # position_error within this threshold AND position_error itself is
        # below it, treat the arm as on target. controller_position_error is
        # -1.0 when the active planner doesn't expose it (only LBFGS does) —
        # that sentinel is excluded from the comparison. Same defaults as
        # release.py/grasp.py.
        self.declare_parameter('controller_error_match_threshold', 0.05)  # m
        # controller_position_error must itself be below this (very tight)
        # threshold to bypass on_target — a loose match against position_error
        # alone isn't enough, since two errors that are merely similar but
        # both still large don't mean the arm has actually arrived.
        self.declare_parameter('controller_error_bypass_threshold', 1e-3)  # m

        # Watchdog for a curobo execution that has internally "solved" (its
        # own controller_position_error keeps shrinking) while the real arm
        # has stalled (position_error plateaus) — see
        # execution_divergence_watchdog.py. Cancelling and resending the goal
        # re-seeds the solver from the real robot state and reliably recovers
        # it. Same defaults as release.py/grasp.py.
        self.declare_parameter('divergence_watchdog_enabled', True)
        self.declare_parameter('divergence_threshold', 0.02)      # m
        self.declare_parameter('divergence_debounce_count', 3)    # feedback samples
        self.declare_parameter('divergence_grace_period', 3.0)    # s after a resync
        self.declare_parameter('divergence_max_resyncs', 3)       # per attempt
        self.declare_parameter('divergence_min_resync_interval', 5.0)  # s

        # Mobile base's gradient controller enable service (std_srvs/SetBool)
        # — enabled once the arm reaches init config at cycle start, disabled
        # once it's back at init config at cycle end.
        self.declare_parameter('base_enable_service_name', '/gradient_base_controller/enable')

        # Current state-machine state (e.g. "idle", "act_grasp"), as
        # std_msgs/String. Transient-local + published on every transition, so
        # a subscriber started mid-cycle still gets the current value
        # immediately instead of waiting for the next change.
        self.declare_parameter('state_topic', '~/state')

        # How the cycle that just ended finished, as "<outcome>|<message>";
        # see outcome_pub. Same transient-local QoS as state_topic.
        self.declare_parameter('outcome_topic', '~/outcome')

        # Whether this cycle drives the mobile base at all. False is the
        # control arm of the with/without-base experiment: the enable service
        # is never called, so the controller stays disabled and publishes only
        # zeros while remaining launched (same GPU load, same logging).
        self.declare_parameter('base_control', True)

        self.grasp_action_name = self.get_parameter('grasp_action_name').value
        self.release_action_name = self.get_parameter('release_action_name').value
        self.state_topic = self.get_parameter('state_topic').value
        self.outcome_topic = self.get_parameter('outcome_topic').value
        self.base_control = bool(self.get_parameter('base_control').value)

        self.set_planner_service_name = self.get_parameter('set_planner_service_name').value
        self.curobo_execute_action = self.get_parameter('curobo_execute_action').value
        self.planner_type_joint_space = self.get_parameter('planner_type_joint_space').value
        self.planner_type_lbfgs = self.get_parameter('planner_type_lbfgs').value
        self.init_joint_positions = list(self.get_parameter('init_joint_positions').value)

        final_pose_position = self.get_parameter('final_pose_position').value
        final_pose_orientation = self.get_parameter('final_pose_orientation').value
        self.final_pose = Pose()
        self.final_pose.position.x = float(final_pose_position[0])
        self.final_pose.position.y = float(final_pose_position[1])
        self.final_pose.position.z = float(final_pose_position[2])
        self.final_pose.orientation.x = float(final_pose_orientation[0])
        self.final_pose.orientation.y = float(final_pose_orientation[1])
        self.final_pose.orientation.z = float(final_pose_orientation[2])
        self.final_pose.orientation.w = float(final_pose_orientation[3])

        self.base_enable_service_name = self.get_parameter('base_enable_service_name').value

        self.controller_error_match_threshold = self.get_parameter('controller_error_match_threshold').value
        self.controller_error_bypass_threshold = self.get_parameter('controller_error_bypass_threshold').value
        self.divergence_watchdog_enabled = self.get_parameter('divergence_watchdog_enabled').value
        self.divergence_threshold = self.get_parameter('divergence_threshold').value
        self.divergence_debounce_count = self.get_parameter('divergence_debounce_count').value
        self.divergence_grace_period = self.get_parameter('divergence_grace_period').value
        self.divergence_max_resyncs = self.get_parameter('divergence_max_resyncs').value
        self.divergence_min_resync_interval = self.get_parameter('divergence_min_resync_interval').value


def main(args=None):
    rclpy.init(args=args)
    node = Brain()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
