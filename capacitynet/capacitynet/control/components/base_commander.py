#!/usr/bin/env python3
"""Turn grad Q into a mobile-base velocity command, safely.

Everything that decides *motion* lives here: gain, saturation, the adaptive
speed cap, the taper, the convergence deadband, the enable interlock, the
task interlocks, and the watchdog that republishes (or zeroes) /cmd_vel.

Safety model: the commander starts DISABLED unless `start_enabled` is set, and
it disables motion unless some task interlock is currently open. A node that
never instantiates this component cannot command the base at all — that is
the point of keeping it separate.

Both `grasp.py` and `release.py` each get their own dedicated, explicit
interlock: a `~/grasp_motion_enable` / `~/release_motion_enable` SetBool
service (mirroring `~/enable`'s idiom, but a separate flag) that the node
calls explicitly around its own approach leg (`act_move_pregrasp` /
`act_move_preplace`) — open until the arm is within `grasp_near_threshold` /
`release_near_threshold` (a node-local computation from FK position_error) of
its approach target, closed otherwise. This supersedes the older, coarser
`grasper_state_topic` + `grasper_allowed_states` whole-state gate for base
motion, whose default is now empty (retired) — that mechanism is kept only
for a node that wants a whole-state gate instead of a distance-based one.

The three interlocks are OR'd, not AND'd — grasp.py reporting a "blocking"
state (e.g. its terminal `idle`, which persists for the rest of the run,
including all of release.py's phases) must not veto the window release.py
explicitly opens, and vice versa. An interlock nobody has ever called
contributes nothing to that OR (see `_grasper_allows_motion`/
`_release_allows_motion`/`_grasp_allows_motion`), so a node that only wires
one of the three behaves exactly as if the others did not exist.

Two stop conditions latch rather than being re-evaluated each cycle: gradient
convergence (`_converged`, set in `submit()`) and arrival (`_arrived`, set the
first time `region_arrived()` reports a *measured* touch of the goal and
path-horizon spheres — `region_converged()`'s precautionary True with no
region at all must not latch, or one goal-only cycle pins the base for the
whole leg). Opening either explicit interlock clears both (see `_clear_latches`):
each rising edge starts a new approach leg with a new region to score, so the
previous leg's convergence or arrival must not carry over.
"""

import math

from geometry_msgs.msg import Twist, Vector3
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

from .node_component import NodeComponent

ZERO_TWIST = Twist()


def gradient_to_velocity(grad_x, grad_y, max_vel):
    """Normalized-direction control: v = max_vel · ∇Q / ‖∇Q‖.

    Every non-zero gradient is commanded at the same speed, `max_vel`; only
    the direction reflects ∇Q, never its magnitude. The single implementation
    of that law in the package — the mock harness and brain_orchestrator use
    it too, so tuning cannot drift between the nodes that drive the base.

    Args:
        grad_x, grad_y: float, gradient components
        max_vel: float, commanded speed in m/s

    Returns:
        tuple: (vx, vy) velocity command in m/s. (0.0, 0.0) if the gradient is
            exactly zero (e.g. an empty region mask), instead of dividing by it.
    """
    magnitude = math.hypot(grad_x, grad_y)
    if magnitude == 0.0:
        return 0.0, 0.0

    scale = max_vel / magnitude
    return grad_x * scale, grad_y * scale


class BaseCommander(NodeComponent):
    """Publishes /cmd_vel from a gradient, with interlocks and a watchdog."""

    def __init__(self, node, delta, prefix='', callback_group=None):
        """Attach the commander to `node` and start its watchdog.

        Args:
            node: host rclpy Node
            delta: float, candidate grid spacing in meters — bounds how far the
                base may travel between two gradient updates
            prefix: optional parameter/topic prefix ('' keeps flat names)
            callback_group: callback group for the interlocks and the watchdog.
                Keep it separate from the inference group, or /cmd_vel collapses
                to the inference rate (~2 Hz) and most base drivers time out
                below ~4 Hz.
        """
        super().__init__(node, prefix, callback_group)
        self._delta = delta

        self.cmd_vel_topic = self._declare('cmd_vel_topic', '/cmd_vel')
        # base_speed is the hardware-validated ceiling; the adaptive cap below can
        # only lower it, never raise it.
        self.base_speed = float(self._declare('base_speed', 0.05))
        self.step_fraction = float(self._declare('step_fraction', 0.5))
        # Below this |grad Q| the node declares convergence and latches to
        # zero. Tested on the gradient directly, not on the commanded speed:
        # with the normalized-direction law in gradient_to_velocity() the
        # commanded speed is always base_speed (or zero), so a speed-based
        # threshold would no longer distinguish "near the optimum" from "far
        # from it" — see submit().
        self.min_gradient = float(self._declare('min_gradient', 0.025))
        # Taper reference; 0 disables it. Above it the base runs at base_speed,
        # below it the speed falls off in proportion to |grad Q| — see submit().
        # It bounds the slowest command ever issued, at base_speed *
        # min_gradient / gradient_taper_ref, so the two thresholds have to be set
        # together or the base gets commands too slow for its drives to execute.
        self.gradient_taper_ref = float(self._declare('gradient_taper_ref', 0.0))
        self.publish_rate = float(self._declare('publish_rate', 20.0))
        self.command_timeout = float(self._declare('command_timeout', 1.0))
        self.start_enabled = bool(self._declare('start_enabled', False))
        self.grasper_state_topic = self._declare(
            'grasper_state_topic', '/object_grasper/state')
        # Comma-separated states that permit base motion. Empty by default —
        # see the module docstring for why this whole-state gate is retired
        # in favor of grasp_motion_enable/release_motion_enable.
        self.grasper_allowed_states = frozenset(
            s.strip() for s in
            str(self._declare('grasper_allowed_states', '')).split(',')
            if s.strip())
        # Whether the MPC plan is about to finish: the goal sphere and the
        # path-horizon sphere touching. A controlling node wires this to
        # WorkspaceRegionSource.region_spheres_touching; left alone, the
        # interlock never engages (belt-and-suspenders for a probe/mock node
        # that never wires it at all).
        self.region_converged = lambda: False
        # Measured form of the same signal, for the arrival latch: True/False
        # only when the goal/path-horizon distance was actually tested, None
        # otherwise. Wired to WorkspaceRegionSource.region_arrived.
        self.region_arrived = lambda: None

        self._enabled = self.start_enabled
        self._converged = False
        # Latched form of region_converged(): the raw signal is re-evaluated
        # every cycle against a distance threshold that sits right in the
        # arrival noise band, so on its own it only ever closes the gate for
        # one cycle at a time (3 of 32 in run_20260910_182418/ep3, while the
        # base drove 0.45 m past a target the arm had already reached). Once
        # set it holds, like _converged, until the next task window opens.
        self._arrived = False
        # None = never heard from the grasper. Treated as permissive so the
        # controller still runs standalone when grasp.py is not up.
        self._grasper_state = None
        # None/False = release.py either isn't up or isn't currently opening
        # its window. Unlike _grasper_state, unheard is NOT permissive here —
        # see _release_allows_motion and the module docstring.
        self._release_motion_allowed = None
        # Same semantics as _release_motion_allowed, for grasp.py's own
        # grasp_motion_enable interlock — see _grasp_allows_motion.
        self._grasp_motion_allowed = None
        # EWMA of the measured cycle time; None until the first cycle completes.
        self._cycle_ewma = None

        # Last computed command, republished by the watchdog until it goes stale.
        self._last_twist = Twist()
        self._last_cmd_time = None

        # Last value returned by allows_motion(), so a change logs once instead
        # of on every publish_command() tick (10 Hz).
        self._last_allows_motion = None

        self._setup_interfaces()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _setup_interfaces(self):
        node = self.node

        self.cmd_vel_pub = node.create_publisher(Twist, self.cmd_vel_topic, 10)
        # Debug: x=grad_x, y=grad_y, z=|grad Q| — the raw input to submit(),
        # before gain/cap/taper/deadband are applied.
        self.gradient_pub = node.create_publisher(
            Vector3, self._private('gradient'), 10)
        # Debug: x=vx, y=vy, z=v_cap (the adaptive cap in force this cycle) — the
        # output of submit(), so gradient_pub vs velocity_debug_pub lets you see
        # exactly how much of the gradient's magnitude survives gain, saturation,
        # the taper, and the convergence deadband.
        self.velocity_debug_pub = node.create_publisher(
            Vector3, self._private('velocity_debug'), 10)

        if self.grasper_state_topic:
            # Transient-local to match grasp.py's publisher, otherwise the latched
            # current state is missed and only the next transition is seen.
            state_qos = QoSProfile(
                depth=1,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.grasper_state_sub = node.create_subscription(
                String, self.grasper_state_topic, self.on_grasper_state, state_qos,
                callback_group=self.callback_group)

        self.enable_sub = node.create_subscription(
            Bool, self._private('enable'), self.on_enable_msg, 10,
            callback_group=self.callback_group)
        self.enable_srv = node.create_service(
            SetBool, self._private('enable'), self.on_enable_srv,
            callback_group=self.callback_group)

        # release.py's interlock (see module docstring) — a separate flag
        # from ~/enable above, explicitly opened/closed by release.py itself.
        self.release_motion_enable_srv = node.create_service(
            SetBool, self._private('release_motion_enable'),
            self.on_release_motion_enable_srv, callback_group=self.callback_group)

        # grasp.py's own interlock, mirroring release_motion_enable above.
        self.grasp_motion_enable_srv = node.create_service(
            SetBool, self._private('grasp_motion_enable'),
            self.on_grasp_motion_enable_srv, callback_group=self.callback_group)

        self.timer = node.create_timer(
            1.0 / self.publish_rate, self.publish_command,
            callback_group=self.callback_group)

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def enabled(self):
        """Whether the enable topic/service currently permits commanding."""
        return self._enabled

    @property
    def converged(self):
        """Whether the last submitted gradient was below min_gradient (holding position)."""
        return self._converged

    @property
    def arrived(self):
        """Whether the arrival latch is set (region spheres touched at least once this leg)."""
        return self._arrived

    @property
    def grasper_state(self):
        """Last state reported by grasp.py on grasper_state_topic, or None if never heard."""
        return self._grasper_state

    @property
    def grasper_allows_motion(self):
        """Whether the last known grasper state is in grasper_allowed_states."""
        return self._grasper_allows_motion()

    def allows_motion(self):
        """Enabled, not converged, some task interlock open, and the region
        not yet converged."""
        enabled = self._enabled
        not_converged = not self._converged
        grasper_ok = self._grasper_allows_motion()
        release_ok = self._release_allows_motion()
        grasp_ok = self._grasp_allows_motion()
        task_ok = grasper_ok or release_ok or grasp_ok
        # Gate form first (cheap short-circuit: the measured form re-resolves
        # the region again, and only matters when the gate already says stop).
        touching = self.region_converged()
        if touching and not self._arrived and self.region_arrived() is True:
            self._arrived = True
            self.log.info(
                'Arrival latch set: region spheres touching — base holds until '
                'the next task window opens')
        spheres_separated = not self._arrived and not touching
        result = enabled and not_converged and task_ok and spheres_separated

        if result != self._last_allows_motion:
            self.log.info(
                f'allows_motion: {self._last_allows_motion} -> {result} '
                f'(enabled={enabled}, not_converged={not_converged}, '
                f'grasper_ok={grasper_ok}, release_ok={release_ok}, '
                f'grasp_ok={grasp_ok}, spheres_separated={spheres_separated})')
            self._last_allows_motion = result
        return result

    def submit(self, gradient, cycle_s, score_center=None):
        """Shape a gradient into the commanded velocity and cache it.

        Convergence deadband on |grad Q|, then the normalized-direction law
        (v = base_speed · ∇Q / ‖∇Q‖), then the optional taper.

        Args:
            gradient: (grad_x, grad_y) of Q, per meter
            cycle_s: how long the cycle that produced it took, in seconds
            score_center: Q at the current position, only used in the convergence log

        Returns:
            tuple: ((vx, vy) commanded in m/s, v_cap: the adaptive per-cycle
                travel cap in force this cycle, for the debug topics/CSV only
                — the command itself is always bounded by base_speed, not
                v_cap; see _velocity_cap())
        """
        # No longer bounds the command (see docstring) — kept only so the
        # debug topics/CSV still carry it.
        v_cap = self._velocity_cap()
        self._cycle_ewma = cycle_s if self._cycle_ewma is None else (
            0.3 * cycle_s + 0.7 * self._cycle_ewma)

        grad_x, grad_y = gradient
        grad_mag = math.hypot(grad_x, grad_y)
        self.gradient_pub.publish(
            Vector3(x=float(grad_x), y=float(grad_y), z=float(grad_mag)))

        if grad_mag < self.min_gradient:
            self._converged = True
            self.log.info(
                f'Converged: |grad Q|={grad_mag:.5f} < min_gradient={self.min_gradient} 1/m. '
                + (f'Q_center={score_center:.4f}. ' if score_center is not None else '')
                + 'Re-enable, or open a task interlock, to resume.')
            self._last_twist = Twist()
            self._last_cmd_time = self.node.get_clock().now()
            self.velocity_debug_pub.publish(Vector3(x=0.0, y=0.0, z=float(v_cap)))
            return (0.0, 0.0), v_cap

        vx, vy = gradient_to_velocity(grad_x, grad_y, self.base_speed)

        # Optional linear taper: restores the deceleration into the optimum that
        # the normalized-direction law above throws away. Only active below
        # g_ref, where the deadband has not yet latched, so it does not slow the
        # approach from far away.
        #
        # Enabling it makes the law proportional-with-saturation, since
        #     base_speed * min(1, |grad Q| / g_ref) == min(base_speed, k * |grad Q|)
        # for k = base_speed / g_ref. Off (g_ref = 0) the commanded speed is
        # base_speed or zero, with nothing in between.
        if self.gradient_taper_ref > 0.0:
            scale = min(1.0, grad_mag / self.gradient_taper_ref)
            vx *= scale
            vy *= scale

        twist = Twist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)
        self._last_twist = twist
        self._last_cmd_time = self.node.get_clock().now()
        self.velocity_debug_pub.publish(Vector3(x=float(vx), y=float(vy), z=float(v_cap)))
        return (vx, vy), v_cap

    def stop(self):
        """Command an immediate stop and forget the cached velocity."""
        self._last_twist = Twist()
        self._last_cmd_time = None
        self.cmd_vel_pub.publish(ZERO_TWIST)

    def shutdown(self):
        """Best-effort stop, safe to call while the context is being torn down."""
        try:
            self._enabled = False
            self.stop()
        except Exception as ex:  # noqa: BLE001 - shutdown must never raise
            self.log.warn(f'Could not publish stop command: {ex}')

    # ── Watchdog ─────────────────────────────────────────────────────────────

    def publish_command(self):
        """Republish the last command, or zero when it cannot be trusted."""
        twist = self._last_twist

        if not self.allows_motion():
            twist = ZERO_TWIST
        elif self._last_cmd_time is None:
            twist = ZERO_TWIST
        else:
            age = self._age_s(self._last_cmd_time)
            if age > self.command_timeout:
                self.log.warn(
                    f'No reachability update for {age:.1f}s (> {self.command_timeout}s), '
                    'stopping base',
                    throttle_duration_sec=5.0)
                twist = ZERO_TWIST

        self.cmd_vel_pub.publish(twist)

    # ── Interlocks ───────────────────────────────────────────────────────────

    def _velocity_cap(self):
        """Speed ceiling keeping travel per cycle within step_fraction * delta."""
        if self._cycle_ewma is None or self._cycle_ewma <= 0.0:
            return self.base_speed
        cap = self.step_fraction * self._delta / self._cycle_ewma
        return min(self.base_speed, cap)

    def _grasper_allows_motion(self):
        # Unknown state (grasper not running) is permissive by design.
        return self._grasper_state is None or self._grasper_state in self.grasper_allowed_states

    def _release_allows_motion(self):
        # Unlike _grasper_allows_motion, unheard/False contributes nothing —
        # this interlock only ever widens the OR while release.py is
        # actively publishing True (see module docstring). Never having heard
        # from release.py must behave exactly as if it weren't wired in.
        return self._release_motion_allowed is True

    def _grasp_allows_motion(self):
        # Same semantics as _release_allows_motion: unheard/False contributes
        # nothing to the OR.
        return self._grasp_motion_allowed is True

    def _set_enabled(self, value, source):
        value = bool(value)
        if value != self._enabled:
            self.log.info(f'{"Enabled" if value else "Disabled"} via {source}')
        self._enabled = value
        self._converged = False
        self._arrived = False
        if not value:
            self.stop()

    def on_enable_msg(self, msg: Bool):
        self._set_enabled(msg.data, 'topic')

    def on_enable_srv(self, request, response):
        self._set_enabled(request.data, 'service')
        response.success = True
        response.message = 'enabled' if self._enabled else 'disabled'
        return response

    def on_grasper_state(self, msg: String):
        previous = self._grasper_state
        self._grasper_state = msg.data
        if previous == msg.data:
            return
        if msg.data not in self.grasper_allowed_states:
            self.log.info(f"Grasper busy ('{msg.data}') — holding base still")
            self.stop()
        elif previous is not None and previous not in self.grasper_allowed_states:
            self.log.info(f"Grasper state '{msg.data}' — base motion allowed again")

    def clear_convergence(self, reason):
        """Lift the convergence latch alone (the arrival latch stays).

        For a region-source change: `submit()` latched `_converged` on the
        |grad Q| of the *previous* region, which says nothing about the new
        one. Seen in run_20260914_133256/ep3+4: the base converged on the
        synthetic path (|grad Q| 0.02 < min_gradient) 6 s before the real
        path arrived, and then never moved again for the leg because only a
        task interlock's rising edge cleared the latch. Arrival is different:
        it was measured on the goal/path-horizon distance and keeps its
        meaning across the swap, so it is left alone.
        """
        if not self._converged:
            return
        self._converged = False
        self.log.info(f'convergence latch cleared by {reason} — base may move again')

    def _clear_latches(self, source):
        """Lift both stop latches because a new task window just opened.

        `submit()` latches `_converged` when |grad Q| drops below
        min_gradient, `allows_motion()` latches `_arrived` when the region
        spheres first touch, and both are AND'ed in, so either stops the base
        for good. That is the right behaviour *within* one approach leg, but
        each task interlock's rising edge starts a new leg with a new region
        to score (grasp.py's pregrasp, then release.py's preplace), and the
        previous leg's convergence or arrival says nothing about the new one.
        Without this, converging or arriving during act_move_pregrasp
        permanently pins the base: opening the release interlock only widens
        the `task_ok` OR and can never clear a latch on its own.
        """
        cleared = []
        if self._converged:
            self._converged = False
            cleared.append('convergence')
        if self._arrived:
            self._arrived = False
            cleared.append('arrival')
        if cleared:
            self.log.info(
                f"{' and '.join(cleared)} latch cleared by {source} — base may move again")

    def on_release_motion_enable_srv(self, request, response):
        if request.data != self._release_motion_allowed:
            self.log.info(
                f"Release motion {'enabled' if request.data else 'disabled'} via service")
            if request.data:
                self._clear_latches('release_motion_enable')
        self._release_motion_allowed = request.data
        response.success = True
        response.message = 'enabled' if self._release_motion_allowed else 'disabled'
        return response

    def on_grasp_motion_enable_srv(self, request, response):
        if request.data != self._grasp_motion_allowed:
            self.log.info(
                f"Grasp motion {'enabled' if request.data else 'disabled'} via service")
            if request.data:
                self._clear_latches('grasp_motion_enable')
        self._grasp_motion_allowed = request.data
        response.success = True
        response.message = 'enabled' if self._grasp_motion_allowed else 'disabled'
        return response
