#!/usr/bin/env python3
"""Doosan-native motion/compliance services used for the final grasp
approach and the post-grasp lift, instead of curobo: TaskComplianceCtrl
(soft), a normal MoveLine while compliant, then ReleaseComplianceCtrl for the
approach; a plain MoveLine for the lift. `DoosanArm` owns the service
clients, the timeout/poll machinery around them, and the driver's own enum
values — the caller (grasp.py's state machine) only decides when to call
what and along which direction/distance.
"""

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.task import Future
from rclpy.duration import Duration

from std_srvs.srv import SetBool
from dsr_msgs2.srv import (
    MoveLine, TaskComplianceCtrl, ReleaseComplianceCtrl, CheckMotion, SetCurrentTool,
    SetDesiredForce, CheckForceCondition, ReleaseForce, GetToolForce,
    MoveStop, GetRobotState, GetLastAlarm,
)
from dsr_msgs2.msg import RobotError

# Doosan dsr_msgs2 enums (see motion_services.html#moveline and
# force_services.html#taskcompliancectrl) — hardcoded here rather than pulled
# from generated message constants, since it's unconfirmed whether this
# driver version exposes them as such; adjust if it does.
DR_BASE = 0
DR_TOOL = 1
DR_WORLD = 2
MOVE_MODE_RELATIVE = 1
SYNC = 0
# CheckMotion.status
DR_STATE_IDLE = 0
DR_STATE_INIT = 1
DR_STATE_BUSY = 2
# CheckForceCondition.axis (see force_services.html#checkforcecondition)
DR_AXIS_X = 0
DR_AXIS_Y = 1
DR_AXIS_Z = 2
DR_AXIS_A = 10
DR_AXIS_B = 11
DR_AXIS_C = 12
# SetDesiredForce.mod (see force_services.html#setdesiredforce)
DR_FC_MOD_ABS = 0
DR_FC_MOD_REL = 1
# MoveStop.stop_mode
DR_QSTOP_STO = 0
DR_QSTOP = 1
DR_SSTO = 2
DR_HOLD = 3
# GetRobotState.robot_state (subset; see GetRobotState.srv for the rest)
STATE_INITIALIZING = 0
STATE_STANDBY = 1
STATE_MOVING = 2
STATE_SAFE_OFF = 3
STATE_SAFE_STOP = 5
STATE_EMERGENCY_STOP = 6
STATE_RECOVERY = 8
STATE_NAMES = {
    0: 'INITIALIZING', 1: 'STANDBY', 2: 'MOVING', 3: 'SAFE_OFF', 4: 'TEACHING',
    5: 'SAFE_STOP', 6: 'EMERGENCY_STOP', 7: 'HOMMING', 8: 'RECOVERY', 9: 'SAFE_STOP2',
    10: 'SAFE_OFF2', 15: 'NOT_READY',
}
# RobotError.level
ERROR_LEVEL_INFO = 1
ERROR_LEVEL_WARN = 2
ERROR_LEVEL_ERROR = 3


class DoosanArm:
    """Wraps the Doosan driver's native motion/compliance/streaming services
    behind one bounded-timeout call each, plus a poll-until-idle helper.

    Owns a dedicated ReentrantCallbackGroup for every client plus the timers
    used to bound/poll them: their done-callbacks (or timeout timers) resolve
    Futures that an awaiting caller's coroutine is suspended on. If these
    shared the caller's default MutuallyExclusiveCallbackGroup (e.g. an
    action server's), that await would deadlock — the group won't start a
    new callback while the (suspended-but-not-finished) awaiting callback
    still occupies it.
    """

    def __init__(self, node, move_line_service: str, task_compliance_ctrl_service: str,
                 release_compliance_ctrl_service: str, check_motion_service: str,
                 enable_streaming_service: str, set_tool_service: str, service_timeout: float = 2.0,
                 move_line_timeout: float = 20.0, motion_poll_period: float = 0.1,
                 motion_idle_timeout: float = 20.0,
                 set_desired_force_service: str = None, check_force_condition_service: str = None,
                 release_force_service: str = None, move_stop_service: str = None,
                 get_robot_state_service: str = None, get_last_alarm_service: str = None,
                 robot_error_topic: str = None, get_tool_force_service: str = None,
                 cancel_check=None):
        """
        Args:
            node: host rclpy Node the service clients and timers are created on.
            move_line_service, task_compliance_ctrl_service,
                release_compliance_ctrl_service, check_motion_service,
                enable_streaming_service, set_tool_service: Doosan/driver service names.
            service_timeout: hard bound on every service call except MoveLine
                (default for `_call_service_with_timeout`) — the driver was
                observed to never reply at all to a rejected/invalid command,
                hanging the await forever otherwise.
            move_line_timeout: MoveLine's own, much longer bound: with
                sync_type=SYNC the service can only reply once the motion
                physically finishes (confirmed outside compliance: a
                100mm@50mm/s lift takes >2s and tripped service_timeout even
                though the move itself was running fine). Must exceed the
                longest commanded move's duration, not the driver's
                round-trip latency.
            motion_poll_period: check_motion poll period in wait_motion_idle.
            motion_idle_timeout: overall bound on waiting for the robot to
                come back to DR_STATE_IDLE in wait_motion_idle.
            set_desired_force_service, check_force_condition_service,
                release_force_service: Doosan force-control services (see
                force_services.html) — optional, only needed by callers that
                use set_desired_force/check_force_condition/release_force
                (release.py's force-controlled place descent). None disables
                the clients (calling those methods then errors immediately).
            move_stop_service, get_robot_state_service,
                get_last_alarm_service: optional, needed by
                stop_streaming_for_direct_control() / assert_ready() — the
                streaming→direct-API handover and the "is the controller
                actually in STANDBY" gate in front of compliance/force calls.
            get_tool_force_service: optional GetToolForce service (lives under
                `aux_control/`, not `force/`, on this driver) — the measured
                external wrench on the tool, used for client-side contact
                detection (see get_tool_force).
            cancel_check: optional zero-arg callable returning True once the
                owning state machine has been asked to cancel. Polled by
                wait_motion_idle so a cancel doesn't sit behind a 20 s
                motion-idle wait; see also stop_now() for the blocking
                MoveLine case.
            robot_error_topic: optional dsr_msgs2/RobotError topic the driver
                publishes controller alarms on (`/dsr01/error`). Subscribed so
                alarms (e.g. 1215 speedj time-out, 1903 "state[MOTION_HOLD]
                rejected event[...]") land in *this* node's log next to the
                call that triggered them, and so callers can check
                `alarms_since_clear` before sending a command the controller
                would reject — or, worse, that the driver would never answer.
        """
        self._node = node
        self._logger = node.get_logger()
        self._service_timeout = service_timeout
        self._move_line_timeout = move_line_timeout
        self._motion_poll_period = motion_poll_period
        self._motion_idle_timeout = motion_idle_timeout
        # False once any driver service timed out and no service has replied
        # since — the driver was observed to stop answering *every* service
        # after one unanswered SetDesiredForce, until restarted. Callers use
        # it to report "driver hung, state unknown" rather than pretend the
        # cleanup calls that followed did anything.
        self.responsive = True
        self._alarms = []
        self._cancel_check = cancel_check

        self._callback_group = ReentrantCallbackGroup()

        self._move_line_client = node.create_client(
            MoveLine, move_line_service, callback_group=self._callback_group)
        self._task_compliance_client = node.create_client(
            TaskComplianceCtrl, task_compliance_ctrl_service, callback_group=self._callback_group)
        self._release_compliance_client = node.create_client(
            ReleaseComplianceCtrl, release_compliance_ctrl_service,
            callback_group=self._callback_group)
        self._check_motion_client = node.create_client(
            CheckMotion, check_motion_service, callback_group=self._callback_group)
        # Must be disabled before, and re-enabled after, every compliance
        # window — see enable_streaming().
        self._enable_streaming_client = node.create_client(
            SetBool, enable_streaming_service, callback_group=self._callback_group)
        self._set_tool_client = node.create_client(
            SetCurrentTool, set_tool_service, callback_group=self._callback_group)

        self._set_desired_force_client = (
            node.create_client(
                SetDesiredForce, set_desired_force_service, callback_group=self._callback_group)
            if set_desired_force_service else None)
        self._check_force_condition_client = (
            node.create_client(
                CheckForceCondition, check_force_condition_service, callback_group=self._callback_group)
            if check_force_condition_service else None)
        self._release_force_client = (
            node.create_client(
                ReleaseForce, release_force_service, callback_group=self._callback_group)
            if release_force_service else None)
        self._move_stop_client = (
            node.create_client(MoveStop, move_stop_service, callback_group=self._callback_group)
            if move_stop_service else None)
        self._get_robot_state_client = (
            node.create_client(
                GetRobotState, get_robot_state_service, callback_group=self._callback_group)
            if get_robot_state_service else None)
        self._get_last_alarm_client = (
            node.create_client(
                GetLastAlarm, get_last_alarm_service, callback_group=self._callback_group)
            if get_last_alarm_service else None)
        self._get_tool_force_client = (
            node.create_client(
                GetToolForce, get_tool_force_service, callback_group=self._callback_group)
            if get_tool_force_service else None)
        self._error_sub = (
            node.create_subscription(
                RobotError, robot_error_topic, self._on_robot_error, 10,
                callback_group=self._callback_group)
            if robot_error_topic else None)

    def _on_robot_error(self, msg: RobotError) -> None:
        text = (f"Doosan alarm level={msg.level} group={msg.group} code={msg.code} "
                f"msg1=\"{msg.msg1}\" msg2=\"{msg.msg2}\" msg3=\"{msg.msg3}\"")
        if msg.level >= ERROR_LEVEL_ERROR:
            self._logger.error(text)
        else:
            self._logger.warning(text)
        self._alarms.append(msg)

    def clear_alarms(self) -> None:
        """Forget alarms received so far; pairs with alarms_since_clear."""
        self._alarms = []

    @property
    def alarms_since_clear(self):
        """RobotError messages received since clear_alarms() (or construction)."""
        return list(self._alarms)

    def error_alarms_since_clear(self):
        """Subset of alarms_since_clear at ERROR level."""
        return [a for a in self._alarms if a.level >= ERROR_LEVEL_ERROR]

    async def _call_service_with_timeout(self, client, request, label: str, timeout: float = None,
                                         driver: bool = True):
        """Call `request` on `client`, bounded by `timeout` (default
        service_timeout — pass move_line_timeout for blocking motion calls,
        which only reply once the robot physically stops). `driver=False`
        marks a service that isn't the Doosan driver's (enable_streaming
        lives on the ExecuteTrajectory node) so its reply doesn't count as
        evidence the driver is responsive.

        Uses a ROS timer + rclpy.task.Future (not asyncio.wait_for) so a
        Doosan service that never replies to a rejected/invalid command
        (observed: SetDesiredForce after a MOTION_HOLD fault) can't hang the
        caller forever. Returns the response, or None on
        unavailable/timeout (already logged).
        """
        if timeout is None:
            timeout = self._service_timeout
        if not client.service_is_ready():
            self._logger.error(f"{label} service not available")
            return None
        call_future = client.call_async(request)
        outcome = Future()

        def _on_call_done(fut):
            if not outcome.done():
                outcome.set_result(fut)

        def _on_timeout():
            if not outcome.done():
                outcome.set_result(None)

        call_future.add_done_callback(_on_call_done)
        timer = self._node.create_timer(
            timeout, _on_timeout, callback_group=self._callback_group)
        try:
            resolved = await outcome
        finally:
            timer.cancel()
            self._node.destroy_timer(timer)

        if resolved is None:
            self._logger.error(f"{label}: timed out after {timeout:.1f}s waiting for response")
            if driver:
                self.responsive = False
            return None
        if driver:
            self.responsive = True
        return resolved.result()

    async def enable_streaming(self, enable: bool) -> bool:
        """Toggle dsr_controller2's RT command streaming. Must be disabled
        before, and re-enabled after, any compliance/force call: left on, it
        keeps streaming commands (observed rejection: internal event
        'eSpeedJ') that TASK_COMPLIANCE_CONTROL rejects, faulting the
        controller into MOTION_HOLD."""
        resp = await self._call_service_with_timeout(
            self._enable_streaming_client, SetBool.Request(data=enable), "enable_streaming",
            driver=False)
        if resp is None:
            return False
        self._logger.info(f"enable_streaming({enable}): success={resp.success} message=\"{resp.message}\"")
        return resp.success

    async def task_compliance_ctrl(self, stx, ref: int, ramp_time: float) -> bool:
        """Enter Doosan task compliance control with target stiffness `stx`
        (6 values: 3 translational N/m + 3 rotational Nm/rad), ramped in over
        `ramp_time` seconds in reference frame `ref`."""
        req = TaskComplianceCtrl.Request()
        req.stx = stx
        req.ref = ref
        req.time = ramp_time
        resp = await self._call_service_with_timeout(
            self._task_compliance_client, req, "task_compliance_ctrl")
        if resp is None:
            return False
        self._logger.info(f"task_compliance_ctrl: success={resp.success} stx={stx}")
        return resp.success

    async def release_compliance_ctrl(self) -> bool:
        """Exit compliance control and restore normal operation mode."""
        resp = await self._call_service_with_timeout(
            self._release_compliance_client, ReleaseComplianceCtrl.Request(), "release_compliance_ctrl")
        if resp is None:
            return False
        self._logger.info(f"release_compliance_ctrl: success={resp.success}")
        return resp.success

    async def set_desired_force(self, fd, dir, ref: int, time: float = 0.0, mod: int = DR_FC_MOD_ABS) -> bool:
        """Command a target force/moment (DR_FC_MOD_ABS: absolute; _REL:
        relative to the state when this is called) along the axes flagged in
        `dir` (1=force-controlled, 0=compliance in TaskComplianceCtrl's
        stiffness for that axis) — must be called after task_compliance_ctrl,
        which is what actually puts the arm in compliance mode. `fd` is
        6 values (3 translational N + 3 rotational Nm), in frame `ref`."""
        if self._set_desired_force_client is None:
            self._logger.error("set_desired_force: no service configured")
            return False
        req = SetDesiredForce.Request()
        req.fd = list(fd)
        req.dir = list(dir)
        req.ref = ref
        req.time = time
        req.mod = mod
        resp = await self._call_service_with_timeout(
            self._set_desired_force_client, req, "set_desired_force")
        if resp is None:
            return False
        self._logger.info(f"set_desired_force: success={resp.success} fd={fd} dir={dir}")
        return resp.success

    async def check_force_condition(self, axis: int, min: float, max: float, ref: int) -> bool:
        """True once the *magnitude* of the measured force/moment along
        `axis` (direction disregarded) falls within [min, max] in frame
        `ref` — a single poll, not a blocking wait; call repeatedly from a
        caller-owned loop (see release.py's force-controlled place descent).
        Returns False on failure/unavailable, not just "condition not met" —
        callers must not conflate the two (contact truly not yet detected)
        without checking this method's own failure logging."""
        if self._check_force_condition_client is None:
            self._logger.error("check_force_condition: no service configured")
            return False
        req = CheckForceCondition.Request()
        req.axis = axis
        req.min = min
        req.max = max
        req.ref = ref
        resp = await self._call_service_with_timeout(
            self._check_force_condition_client, req, "check_force_condition")
        if resp is None:
            return False
        return resp.success

    async def get_tool_force(self, ref: int):
        """Measured external wrench on the tool, 6 values ([Fx, Fy, Fz] N in
        frame `ref` for the force; moments are tool-frame per the srv doc),
        or None on unavailable/timeout/failure.

        Preferred over check_force_condition for contact detection: it
        returns the number, so the caller can subtract a baseline. Measured
        on this robot at rest: Fz = +5.5 N in DR_BASE with nothing touching
        (tool weight / sensor offset) — larger than the contact threshold,
        so any magnitude-only test against the raw value is meaningless.
        """
        if self._get_tool_force_client is None:
            self._logger.error("get_tool_force: no service configured")
            return None
        resp = await self._call_service_with_timeout(
            self._get_tool_force_client, GetToolForce.Request(ref=ref), "get_tool_force")
        if resp is None or not resp.success:
            return None
        return list(resp.tool_force)

    async def release_force(self, time: float = 0.0) -> bool:
        """Smoothly zero out a force commanded by set_desired_force, over
        `time` seconds (0~1.0s). Does not exit compliance mode — that's
        still release_compliance_ctrl's job."""
        if self._release_force_client is None:
            self._logger.error("release_force: no service configured")
            return False
        resp = await self._call_service_with_timeout(
            self._release_force_client, ReleaseForce.Request(time=time), "release_force")
        if resp is None:
            return False
        self._logger.info(f"release_force: success={resp.success}")
        return resp.success

    async def move_line_relative(self, pos, ref: int, vel, acc) -> bool:
        """Relative MoveLine by `pos` ([dx,dy,dz] mm + [drx,dry,drz] deg) in
        reference frame `ref`.

        sync_type=SYNC: the reply normally arrives only once the robot has
        physically stopped. An early success=True reply means the command was
        accepted and dropped, not that the move was fast — the known cause is
        commanding during a compliance stiffness ramp (see
        grasp.py's compliance_settle_time).
        """
        req = MoveLine.Request()
        req.pos = list(pos)
        req.vel = list(vel)
        req.acc = list(acc)
        req.time = 0.0
        req.radius = 0.0
        req.ref = ref
        req.mode = MOVE_MODE_RELATIVE
        req.blend_type = 0
        req.sync_type = SYNC
        self._logger.info(f"move_line: relative pos={req} ref={ref}")
        resp = await self._call_service_with_timeout(
            self._move_line_client, req, "move_line", timeout=self._move_line_timeout)
        if resp is None:
            return False
        self._logger.info(f"move_line: success={resp.success}")
        return resp.success

    def cancel_requested(self) -> bool:
        """True once the owner's cancel_check says so (False without one)."""
        return bool(self._cancel_check and self._cancel_check())

    def stop_now(self, stop_mode: int = DR_SSTO) -> None:
        """Fire-and-forget MoveStop, for use from a synchronous cancel
        callback. A sync_type=SYNC MoveLine only replies once the motion
        ends, so a cancel that arrives mid-move would otherwise wait for
        the move to run to completion (bounded only by move_line_timeout);
        stopping the motion makes the pending MoveLine reply now, and the
        state's normal cleanup (compliance release, streaming) then runs.
        No-op with a log if no move_stop service is configured."""
        if self._move_stop_client is None:
            self._logger.error("stop_now: no move_stop service configured, motion continues")
            return
        if not self._move_stop_client.service_is_ready():
            self._logger.error("stop_now: move_stop service not available, motion continues")
            return
        self._logger.warning(f"Cancel: sending move_stop(mode={stop_mode})")
        self._move_stop_client.call_async(MoveStop.Request(stop_mode=stop_mode))

    async def move_stop(self, stop_mode: int = DR_SSTO) -> bool:
        """Stop the current motion (DR_SSTO: soft stop, no fault)."""
        if self._move_stop_client is None:
            self._logger.error("move_stop: no service configured")
            return False
        resp = await self._call_service_with_timeout(
            self._move_stop_client, MoveStop.Request(stop_mode=stop_mode), "move_stop")
        if resp is None:
            return False
        self._logger.info(f"move_stop(mode={stop_mode}): success={resp.success}")
        return resp.success

    async def get_robot_state(self):
        """Controller state (STATE_STANDBY/MOVING/SAFE_STOP/...), or None on
        unavailable/timeout/failure."""
        if self._get_robot_state_client is None:
            self._logger.error("get_robot_state: no service configured")
            return None
        resp = await self._call_service_with_timeout(
            self._get_robot_state_client, GetRobotState.Request(), "get_robot_state")
        if resp is None or not resp.success:
            return None
        return resp.robot_state

    async def get_last_alarm(self):
        """Last controller alarm as a dsr_msgs2/LogAlarm (index 0 = none),
        or None on unavailable/timeout/failure."""
        if self._get_last_alarm_client is None:
            self._logger.error("get_last_alarm: no service configured")
            return None
        resp = await self._call_service_with_timeout(
            self._get_last_alarm_client, GetLastAlarm.Request(), "get_last_alarm")
        if resp is None or not resp.success:
            return None
        return resp.log_alarm

    async def assert_ready(self, label: str) -> bool:
        """Gate in front of every direct-API command sent while streaming is
        off: True only if the controller reports STATE_STANDBY, check_motion
        reports IDLE, and no ERROR-level alarm arrived since clear_alarms().

        Exists because the failure modes on the other side of this gate are
        not just "command rejected": the driver was observed to stop
        answering every service — until restarted — after a SetDesiredForce
        sent while the controller sat in MOTION_HOLD. Refusing to send is the
        only client-side protection there is.
        """
        errors = self.error_alarms_since_clear()
        if errors:
            codes = ", ".join(str(a.code) for a in errors)
            self._logger.error(
                f"{label}: refusing, {len(errors)} controller alarm(s) since handover (codes {codes})")
            return False
        state = await self.get_robot_state()
        if state is None:
            self._logger.error(f"{label}: refusing, robot state unknown")
            return False
        if state != STATE_STANDBY:
            self._logger.error(
                f"{label}: refusing, robot_state={state} ({STATE_NAMES.get(state, '?')}), "
                f"expected STANDBY")
            return False
        status = await self.check_motion()
        if status != DR_STATE_IDLE:
            self._logger.error(f"{label}: refusing, check_motion={status}, expected IDLE")
            return False
        return True

    async def try_unblock(self) -> bool:
        """Recovery attempt once a driver service has timed out: MoveStop
        (DR_SSTO) then a GetRobotState probe. Observed on the robot: after
        a run where SetDesiredForce, CheckMotion and ReleaseComplianceCtrl
        all went unanswered, a MoveStop was answered and every service
        replied again afterwards — consistent with the driver's command
        queue to the controller being stuck behind the never-acknowledged
        force command, and stop() being what purges it. Not proven to be
        the cause (the driver may also recover on its own eventually), so
        callers treat True as "worth retrying the cleanup", nothing more.
        """
        self._logger.warning("Driver not answering: sending move_stop(SSTO) to try to unblock it")
        await self.move_stop(DR_SSTO)
        state = await self.get_robot_state()
        if state is None:
            self._logger.error("Driver still not answering after move_stop")
            return False
        self._logger.info(
            f"Driver answering again after move_stop: robot_state={state} "
            f"({STATE_NAMES.get(state, '?')})")
        return True

    async def stop_streaming_for_direct_control(self, stop_motion: bool = True) -> bool:
        """Hand the controller from ExecuteTrajectory's speedj_rt streaming to
        direct service control, verified.

        enable_streaming(False) only makes the ExecuteTrajectory node stop
        publishing speedj_rt_stream. Each streamed speedj is valid for
        0.1 s, so the controller's last one then expires on its own: observed
        ~80 ms later as alarm 1215 "[SpeedJ] speedj() generates time-out
        error", after which the controller sits in MOTION_HOLD and rejects
        (1903 "state[MOTION_HOLD] rejected event[eConfigDesiredForce]") — or
        never answers — what comes next. So the speedj session is ended by
        an explicit soft stop (move_stop DR_SSTO), sent right after streaming
        goes off and within that 0.1 s validity window, then the controller
        is required to be back in STANDBY/IDLE before returning True.

        `stop_motion=False` skips the move_stop (A/B on the robot: with it
        off, expect alarm 1215 in this log ~0.1 s after enable_streaming).
        On any failure streaming is re-enabled before returning False.
        """
        self.clear_alarms()
        if not await self.enable_streaming(False):
            return False
        if stop_motion and not await self.move_stop(DR_SSTO):
            self._logger.error("Handover: move_stop failed, re-enabling streaming")
            await self.enable_streaming(True)
            return False
        if not await self.wait_motion_idle():
            self._logger.error("Handover: robot did not come to IDLE, re-enabling streaming")
            await self.enable_streaming(True)
            return False
        if not await self.assert_ready("Handover"):
            alarm = await self.get_last_alarm()
            if alarm is not None and alarm.index != 0:
                self._logger.error(
                    f"Handover: last controller alarm level={alarm.level} group={alarm.group} "
                    f"index={alarm.index} param={list(alarm.param)}")
            await self.enable_streaming(True)
            return False
        self._logger.info("Handover: streaming off, controller STANDBY/IDLE, direct control OK")
        return True

    async def set_tool(self, name: str) -> bool:
        """Set the currently installed tool by name (e.g. after the gripper
        has closed on an object, so payload/TCP reflects the held object)."""
        resp = await self._call_service_with_timeout(
            self._set_tool_client, SetCurrentTool.Request(name=name), "set_tool")
        if resp is None:
            return False
        self._logger.info(f"set_tool({name}): success={resp.success}")
        return resp.success

    async def check_motion(self):
        """Current motion status (DR_STATE_IDLE/INIT/BUSY), or None on
        unavailable/timeout/failure."""
        resp = await self._call_service_with_timeout(
            self._check_motion_client, CheckMotion.Request(), "check_motion")
        if resp is None or not resp.success:
            return None
        return resp.status

    async def wait_motion_idle(self, timeout: float = None) -> bool:
        """Poll check_motion until the robot reports DR_STATE_IDLE.

        Safety net rather than the primary completion signal: MoveLine's
        sync_type=SYNC already blocks until the robot stops, so this normally
        returns on the first poll. It exists so the caller never releases
        compliance or reports success while the arm is still moving. Returns
        False on timeout (motion still running) or if check_motion is
        unreachable.
        """
        if timeout is None:
            timeout = self._motion_idle_timeout
        deadline = self._node.get_clock().now() + Duration(seconds=timeout)
        # Give the driver one poll period to leave IDLE before believing an
        # IDLE reading: check_motion right after MoveLine returns can still
        # report the pre-motion state.
        await self.sleep(self._motion_poll_period)
        stop_sent = False
        while self._node.get_clock().now() < deadline:
            status = await self.check_motion()
            if status is None:
                self._logger.error("check_motion unavailable, cannot confirm motion end")
                return False
            if status == DR_STATE_IDLE:
                # A cancel that arrived while waiting still counts as failure
                # for the caller, even though the arm did come to rest.
                return not self.cancel_requested()
            if self.cancel_requested() and not stop_sent:
                # Don't just bail: the arm is still moving. Stop it, then
                # keep polling so the caller's cleanup runs on a still arm.
                stop_sent = True
                await self.move_stop(DR_SSTO)
            await self.sleep(self._motion_poll_period)
        self._logger.error(f"Motion still running after {timeout:.1f}s (check_motion != IDLE)")
        return False

    async def sleep(self, seconds: float) -> None:
        """Await `seconds` without blocking the executor — a one-shot ROS
        timer resolving a Future, on the same dedicated callback group as
        every other wait here, for the same deadlock-avoidance reason."""
        done = Future()
        timer = self._node.create_timer(
            seconds, lambda: done.done() or done.set_result(True),
            callback_group=self._callback_group)
        try:
            await done
        finally:
            timer.cancel()
            self._node.destroy_timer(timer)
