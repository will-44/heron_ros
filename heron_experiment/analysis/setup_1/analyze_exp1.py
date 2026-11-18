#!/usr/bin/env python3
"""Offline analysis of experiment 1 (gradient base controller ON, 35 episodes).

Reads the raw/ directory of a run written by `workspace_reachability_node` +
`gradient_base_controller` and regenerates every figure and table used in
`REPORT_base_mobile.md`. ROS-free: only pandas / numpy / matplotlib.

    python3 analysis/setup_1/analyze_exp1.py                 # setup 1, base mobile (runs.SETUP_1_MOVING)
    python3 analysis/setup_1/analyze_exp1.py --run-dir setup_1/<run_id> --out /path/to/out

The loading / per-leg / figure functions are run-agnostic and are reused by the
fixed-base run (`analyze_exp2.py`) through `run(...)`, with that run's own operator
notes and outcome classes.

Outputs (relative to --out, default = the run directory, next to its raw/):
    figures/fig_*.png        every figure of the report
    tables/*.csv|md          per-episode / per-leg tables and the headline numbers
    tables/summary.json      the numbers quoted in the report, machine-readable

Conventions worth knowing before reading the code:

* A *leg* is one (episode, approach phase) pair during which the base controller
  is gated open: `act_grasp:act_move_pregrasp` (pick approach) or
  `act_release:act_move_preplace` (place approach). Those are the only phases in
  which the controller ever produced a cycle in this run.
* `control_debug.csv` has no episode/phase columns; cycles are joined to the
  recorder's episodes/phases by wall-clock timestamp.
* The recorder never received the base-side Twist topic in this run, so
  `x_est/y_est/dist_est` are zero everywhere. The base displacement is
  re-integrated here from the controller's own /cmd_vel echo (10 Hz,
  piecewise-constant), and validated against the shift of the MPC goal in the
  arm frame (the goal is fixed in the world, so its motion in `dsr01/base_link`
  is minus the base motion).
* The first controller cycle of a leg sometimes scores a stale reachability map
  (rm age > 10 s) with a goal-only synthetic region; Q on that cycle is an
  artefact (0.14-0.19 where the next cycle reads 0.86). Leg-level Q statistics
  therefore start at the first cycle whose region has all three spheres.
"""

import argparse
import warnings
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ── Palette (reference data-viz palette, light surface) ─────────────────────
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
SURFACE = '#fcfcfb'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948')
GOOD, WARNING, SERIOUS, CRITICAL = '#0ca30c', '#fab219', '#ec835a', '#d03b3b'
SEQ_BLUE = matplotlib.colors.LinearSegmentedColormap.from_list(
    'seq_blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])

PHASE_COLOR = {
    'act_home_enter': AQUA,
    'act_grasp': BLUE,
    'act_release': ORANGE,
    'act_home_exit': YELLOW,
}
CLASS_COLOR = {
    'success': GOOD,
    'bad-grasp cancel': WARNING,
    'grasp/release failed': SERIOUS,
    'collision': CRITICAL,
    'excluded': MUTED,
}
CLASS_ORDER = ['success', 'bad-grasp cancel', 'grasp/release failed', 'collision', 'excluded']

# A grasp or release failure is not an outcome of the base controller: at that
# moment the base is static in both conditions, and what fails is the AprilTag
# placement plus the arm's 5 mm positioning accuracy. Those cycles are excluded
# from the success rate (operator rule, 2026-09-23) but kept in every
# trajectory statistic, where the approach they recorded is valid data.
GRASP_CLASSES = ('bad-grasp cancel', 'grasp/release failed')
OUTCOME_EXCLUDED = ('excluded',) + GRASP_CLASSES


def outcome_valid(idx):
    """The episodes every statistic is computed over (an excluded episode
    appears nowhere but in the per-episode table that records why)."""
    return idx[~idx.cls.isin(OUTCOME_EXCLUDED)]


PREGRASP = 'act_grasp:act_move_pregrasp'
PREPLACE = 'act_release:act_move_preplace'
LEG_LABEL = {PREGRASP: 'pick approach (pregrasp)', PREPLACE: 'place approach (preplace)'}
LEG_COLOR = {PREGRASP: BLUE, PREPLACE: ORANGE}

# Operator notes on the run (source: experimenter, 2026-09-16). These override
# the raw `outcome` column of episodes_index.csv, which only knows how brain.py
# ended the cycle, not why.
EPISODE_NOTES = {
    4: ('excluded', 'infrastructure: set_planner service unavailable at cycle start (6 ms episode)'),
    3: ('excluded', 'operator intervention during pregrasp (setup error)'),
    29: ('excluded', 'operator intervention during pregrasp (setup error)'),
    10: ('collision', 'arm collision during pregrasp; cancelled after 135 s'),
    33: ('collision', 'arm collision at grasp (Grasp failed: Sequence failed)'),
    24: ('success', 'pick-and-place completed; collision only during the final homing'),
    27: ('success', 'pick-and-place completed; cancelled during the final homing'),
}
BAD_GRASP_NOTE = 'object badly seized -> perturbed obstacle map -> cancelled during place approach'


DEFAULT_CANCEL = ('bad-grasp cancel', BAD_GRASP_NOTE)


def classify(row, notes=EPISODE_NOTES, default_cancel=DEFAULT_CANCEL, default_failure=None):
    """(class, note) of an episode: operator notes first, then the raw outcome."""
    ep = int(row.episode_id)
    if ep in notes:
        return notes[ep]
    if row.outcome == 'success':
        return 'success', 'pick-and-place completed'
    if row.outcome == 'cancelled':
        return default_cancel
    if default_failure is not None:
        return default_failure[0], f'{default_failure[1]} ({row.outcome_message})'
    return 'excluded', row.outcome_message


# ── Loading ─────────────────────────────────────────────────────────────────

def read_csv(path):
    try:
        if os.path.getsize(path) == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def concat_episode_files(run_dir, name):
    frames = []
    for d in sorted(glob.glob(os.path.join(run_dir, 'episode_*'))):
        df = read_csv(os.path.join(d, name))
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load(run_dir, notes=EPISODE_NOTES, default_cancel=DEFAULT_CANCEL, default_failure=None):
    idx = read_csv(os.path.join(run_dir, 'episodes_index.csv'))
    cls = idx.apply(classify, axis=1, result_type='expand', notes=notes,
                    default_cancel=default_cancel, default_failure=default_failure)
    idx['cls'], idx['note'] = cls[0], cls[1]
    phases = read_csv(os.path.join(run_dir, 'phase_summary.csv'))
    ctrl = read_csv(os.path.join(run_dir, 'control_debug.csv'))

    # Join controller cycles to episodes and phases by wall-clock time.
    ctrl['episode_id'] = -1
    ctrl['phase'] = ''
    for _, e in idx.iterrows():
        in_ep = (ctrl.timestamp >= e.t_start) & (ctrl.timestamp <= e.t_end)
        ctrl.loc[in_ep, 'episode_id'] = int(e.episode_id)
        for _, p in phases[phases.episode_id == e.episode_id].iterrows():
            in_ph = in_ep & (ctrl.timestamp >= e.t_start + p.t_start_rel) \
                & (ctrl.timestamp < e.t_start + p.t_end_rel)
            ctrl.loc[in_ph, 'phase'] = p.phase
    ctrl['active'] = ctrl.skip_reason.isna()
    ctrl['moving'] = ctrl.active & (ctrl.allows_motion == True) \
        & ((ctrl.vx_cmd.abs() > 1e-6) | (ctrl.vy_cmd.abs() > 1e-6))  # noqa: E712

    samples = concat_episode_files(run_dir, 'samples.csv')
    cmd_vel = concat_episode_files(run_dir, 'cmd_vel.csv')
    mpc = concat_episode_files(run_dir, 'mpc_diagnostics.csv')
    joints = concat_episode_files(run_dir, 'joint_states.csv')

    # An excluded episode contributes to nothing: drop its rows from every
    # measurement frame here, once, so no statistic downstream can see it.
    # `idx` and `phases` keep every episode, for the per-episode table and the
    # timeline that document why it was excluded.
    keep = set(outcome_valid(idx).episode_id)
    frames = dict(ctrl=ctrl, samples=samples, cmd_vel=cmd_vel, mpc=mpc, joints=joints)
    for k, df in frames.items():
        if not df.empty and 'episode_id' in df.columns:
            frames[k] = df[df.episode_id.isin(keep)].reset_index(drop=True)
    return dict(idx=idx, phases=phases, **frames)


# ── Base displacement (re-integrated from the controller's /cmd_vel echo) ───

def integrate_cmd_vel(cmd_vel):
    """Add x/y/dist columns: piecewise-constant integration per episode."""
    out = []
    for ep, g in cmd_vel[cmd_vel.source == 'controller'].groupby('episode_id'):
        g = g.sort_values('timestamp').copy()
        t = g.timestamp.to_numpy()
        dt = np.diff(t, append=t[-1])
        g['dt'] = dt
        g['x'] = np.cumsum(g.vx.to_numpy() * dt)
        g['y'] = np.cumsum(g.vy.to_numpy() * dt)
        g['dist'] = np.cumsum(np.hypot(g.vx, g.vy).to_numpy() * dt)
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ── Per-leg table ──────────────────────────────────────────────────────────

def stop_reason(g):
    stop = g[g.allows_motion == False]  # noqa: E712
    if stop.empty:
        return 'interlock closed (arm near target)'
    s = stop.iloc[0]
    if s.arrived:
        return 'arrival latch (spheres touching)'
    if s.converged:
        return 'gradient convergence'
    return 'interlock closed (arm near target)'


def build_legs(data, cv):
    ctrl, samples, idx = data['ctrl'], data['samples'], data['idx']
    rows = []
    for (ep, phase), g in ctrl[ctrl.active & ctrl.phase.isin([PREGRASP, PREPLACE])].groupby(
            ['episode_id', 'phase']):
        g = g.sort_values('timestamp')
        mv = g[g.moving]
        full = g[g.region_size >= 3]           # skip the goal-only first cycle
        q_series = full.q_center if not full.empty else g.q_center
        qg_series = full.ws_mean_0 if not full.empty else g.ws_mean_0
        qp_series = ((full.ws_mean_1 + full.ws_mean_2) / 2) if not full.empty else pd.Series(dtype=float)
        e = idx[idx.episode_id == ep].iloc[0]
        t0 = e.t_start

        # Commanded displacement over the phase, from the 10 Hz echo.
        c = cv[(cv.episode_id == ep) & (cv.phase == phase)]
        dist = float(np.sum(np.hypot(c.vx, c.vy) * c.dt)) if not c.empty else 0.0
        dx = float(np.sum(c.vx * c.dt)) if not c.empty else 0.0
        dy = float(np.sum(c.vy * c.dt)) if not c.empty else 0.0

        # Observed displacement: goal shift in the arm frame over the motion window.
        s = samples[(samples.episode_id == ep) & (samples.phase == phase)].sort_values('t_rel')
        dg = (np.nan, np.nan)
        goal0 = (np.nan, np.nan, np.nan)
        goal1 = (np.nan, np.nan, np.nan)
        dc_win = (np.nan, np.nan)
        if not s.empty and not mv.empty and not c.empty:
            t_a = max(s.t_rel.min(), c.t_rel.min())
            t_b = (mv.timestamp.max() - t0) + 1.0
            s0, s1 = s[s.t_rel >= t_a].iloc[0], s[s.t_rel <= t_b].iloc[-1]
            cw = c[(c.t_rel >= t_a) & (c.t_rel <= t_b)]
            goal0 = (s0.goal_x, s0.goal_y, s0.goal_z)
            goal1 = (s1.goal_x, s1.goal_y, s1.goal_z)
            dg = (s1.goal_x - s0.goal_x, s1.goal_y - s0.goal_y)
            if not cw.empty:
                dc_win = (cw.x.iloc[-1] - cw.x.iloc[0], cw.y.iloc[-1] - cw.y.iloc[0])

        # Direction reversals: consecutive moving cycles whose gradient turns > 90 deg.
        ang = np.arctan2(mv.grad_y, mv.grad_x).to_numpy()
        dang = np.degrees(np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)) if len(ang) > 1 else np.array([])

        rows.append(dict(
            episode_id=ep, phase=phase, leg=LEG_LABEL[phase], cls=e.cls,
            t_leg_start=float(g.timestamp.min() - t0),
            n_cycles=len(g), n_moving=len(mv),
            t_moving_s=float(mv.timestamp.max() - mv.timestamp.min()) if len(mv) > 1 else 0.0,
            phase_duration_s=float(c.t_rel.max() - c.t_rel.min()) if not c.empty else np.nan,
            dist_m=dist, dx_m=dx, dy_m=dy, net_m=float(np.hypot(dx, dy)),
            q_start=float(q_series.iloc[0]), q_end=float(mv.q_center.iloc[-1]) if len(mv) else np.nan,
            q_max=float(q_series.max()), q_min=float(q_series.min()),
            qgoal_start=float(qg_series.iloc[0]), qgoal_end=float(mv.ws_mean_0.iloc[-1]) if len(mv) else np.nan,
            qpath_start=float(qp_series.iloc[0]) if len(qp_series) else np.nan,
            qpath_end=float((mv.ws_mean_1.iloc[-1] + mv.ws_mean_2.iloc[-1]) / 2) if len(mv) else np.nan,
            grad_med=float(mv.grad_mag.median()) if len(mv) else np.nan,
            n_reversals=int((dang > 90).sum()),
            stop=stop_reason(g),
            goal0_x=goal0[0], goal0_y=goal0[1], goal0_z=goal0[2],
            goal1_x=goal1[0], goal1_y=goal1[1],
            dgoal_x=dg[0], dgoal_y=dg[1], dcmd_win_x=dc_win[0], dcmd_win_y=dc_win[1],
        ))
    legs = pd.DataFrame(rows)
    legs['dq'] = legs.q_end - legs.q_start
    legs['dq_goal'] = legs.qgoal_end - legs.qgoal_start
    legs['dq_path'] = legs.qpath_end - legs.qpath_start
    legs['path_efficiency'] = legs.net_m / legs.dist_m.replace(0, np.nan)
    return legs


# ── Per-cycle gradient validity ─────────────────────────────────────────────

def build_cycle_pairs(ctrl):
    rows = []
    for (ep, phase), g in ctrl[ctrl.active & ctrl.phase.isin([PREGRASP, PREPLACE])
                               & (ctrl.allows_motion == True)].groupby(['episode_id', 'phase']):  # noqa: E712
        g = g.sort_values('timestamp')
        t, gx, gy = g.timestamp.to_numpy(), g.grad_x.to_numpy(), g.grad_y.to_numpy()
        q, vx, vy = g.q_center.to_numpy(), g.vx_cmd.to_numpy(), g.vy_cmd.to_numpy()
        for i in range(len(g) - 1):
            dt = t[i + 1] - t[i]
            if dt > 2.5:
                continue
            a1, a2 = np.arctan2(gy[i], gx[i]), np.arctan2(gy[i + 1], gx[i + 1])
            rows.append(dict(
                episode_id=ep, phase=phase, dt=dt,
                pred=(gx[i] * vx[i] + gy[i] * vy[i]) * dt, meas=q[i + 1] - q[i],
                dang=np.degrees(np.abs((a2 - a1 + np.pi) % (2 * np.pi) - np.pi)),
                grad_mag=np.hypot(gx[i], gy[i])))
    return pd.DataFrame(rows)


# ── Figure helpers ──────────────────────────────────────────────────────────

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)



# --- median across legs, without hiding the legs that outlive the quorum -------------
# The old rule was `ok = n >= 5`: the median was drawn only while at least 5 legs were
# still running. That hid the story at both ends -- the plotted average was taken over a
# shrinking, self-selected set of legs, and the curve stopped several seconds before the
# longest legs actually ended (setup 2, pick approach: curve cut at 18 s, legs running to
# 27 s). Now every leg is shown to its own end: the median is drawn over the whole span,
# solid while at least half the legs are still running and dashed once the survivors are a
# minority, and the number of legs still running is drawn as a step curve along the bottom
# of the panel (the "number at risk" strip of a survival plot).
SOLID_FRAC = 0.5
T_GRID_MAX = 600


def _n_twin(ax):
    tw = getattr(ax, '_n_twin_ax', None)
    if tw is None:
        tw = ax.twinx()
        tw.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
        tw.set_yticks([])
        for s in tw.spines.values():
            s.set_visible(False)
        ax._n_twin_ax = tw
        ax._n_max = 0
    return tw


def draw_n_at_risk(ax, grid_t, n, color):
    """Step curve of how many legs are still running, in the bottom fifth of the panel."""
    tw = _n_twin(ax)
    ax._n_max = max(ax._n_max, int(n.max()) if n.size else 0)
    tw.set_ylim(0, max(1, ax._n_max) * 5.0)
    alive = n >= 1
    if alive.any():
        end = int(np.flatnonzero(alive)[-1])
        tw.step(grid_t[:end + 1], n[:end + 1], where='post', color=color,
                linewidth=1.0, alpha=0.45, zorder=1)
    return tw


def median_over_legs(ax, grid_t, arr, color, label=None, band=True, linewidth=2.0,
                     n_at_risk=True):
    """Draw the median (and IQR) across legs over the full span, with the survival split.

    Returns (n, t_end, t_solid_end): leg count per time step, last time any leg is still
    running, and the last time at least `SOLID_FRAC` of them are.
    """
    if not arr.size:
        return np.zeros(len(grid_t), int), 0, 0
    n = np.sum(~np.isnan(arr), axis=0)
    if not (n >= 1).any():
        return n, 0, 0
    alive = n >= 1
    solid = n >= max(2, int(np.ceil(SOLID_FRAC * n.max())))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(arr, axis=0)
        lo = np.nanpercentile(arr, 25, axis=0)
        hi = np.nanpercentile(arr, 75, axis=0)
    # dashed over the whole span first, the solid part is drawn on top of it
    ax.plot(grid_t[alive], med[alive], '--', color=color, linewidth=linewidth,
            dashes=(3, 2.2), alpha=0.9, zorder=3)
    if band:
        ax.fill_between(grid_t[solid], lo[solid], hi[solid], color=color, alpha=0.18,
                        linewidth=0, zorder=2)
    ax.plot(grid_t[solid], med[solid], '-', color=color, linewidth=linewidth, label=label,
            zorder=4)
    if n_at_risk:
        draw_n_at_risk(ax, grid_t, n, color)
    t_end = int(grid_t[np.flatnonzero(alive)[-1]])
    t_solid = int(grid_t[np.flatnonzero(solid)[-1]]) if solid.any() else 0
    return n, t_end, t_solid


def legend_survival_handles(colors_labels, ink, strip=True):
    """Legend entries: one per run, plus what solid / dashed / the bottom strip mean.

    `strip=False` for a panel drawn without the number-at-risk strip.
    """
    h = [Line2D([], [], color=c, linewidth=2, label=l) for l, c in colors_labels]
    h += [Line2D([], [], color=ink, linewidth=2,
                 label=f'median, {int(SOLID_FRAC * 100)}%+ of legs still running'),
          Line2D([], [], color=ink, linewidth=2, linestyle='--', dashes=(3, 2.2),
                 label='median, only a minority of legs left')]
    if strip:
        h += [Line2D([], [], color=ink, linewidth=1, alpha=0.45,
                     label='bottom strip: number of legs still running')]
    return h

def new_fig(nrows=1, ncols=1, figsize=(8, 4.5), **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE, **kw)
    for ax in np.atleast_1d(axes).ravel():
        style_axes(ax)
    return fig, axes


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', os.path.relpath(path))


# ── Figures ─────────────────────────────────────────────────────────────────

def fig_timeline(data, legs, out_dir, t_max=300, moving_label='base moving (controller cycle)'):
    idx, phases, ctrl = data['idx'], data['phases'], data['ctrl']
    eps = idx.sort_values('episode_id')
    fig, ax = new_fig(figsize=(10, 8.5))
    for k, (_, e) in enumerate(eps.iterrows()):
        y = k
        for _, p in phases[phases.episode_id == e.episode_id].iterrows():
            ax.barh(y, p.duration_s, left=p.t_start_rel, height=0.62,
                    color=PHASE_COLOR.get(p.brain_state, MUTED), linewidth=0)
        mv = ctrl[(ctrl.episode_id == e.episode_id) & ctrl.moving]
        if not mv.empty:
            ax.plot(mv.timestamp - e.t_start, np.full(len(mv), y - 0.42), '|',
                    color=INK, markersize=4, markeredgewidth=0.8)
        ax.text(e.duration_s + 2, y, f'{e.cls}', va='center', fontsize=7.5,
                color=CLASS_COLOR[e.cls], fontweight='bold')
    ax.set_yticks(range(len(eps)))
    ax.set_yticklabels([f'ep {int(e)}' for e in eps.episode_id], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel('time since episode start (s)')
    ax.set_xlim(0, t_max)
    handles = [Patch(color=PHASE_COLOR[k], label=k) for k in PHASE_COLOR]
    handles.append(Line2D([], [], marker='|', color=INK, linestyle='none', label=moving_label))
    ax.legend(handles=handles, loc='lower right', fontsize=7.5, frameon=False)
    ax.set_title('Episode timeline by phase, with the cycles in which the base was commanded to move',
                 fontsize=10, color=INK, loc='left')
    save(fig, out_dir, 'fig_timeline.png')


def fig_base_paths(data, legs, cv, out_dir):
    """Base path in the object frame: goal at the origin, base starts at -goal0."""
    fig, axes = new_fig(1, 2, figsize=(11, 5.2))
    for ax, phase in zip(axes, (PREGRASP, PREPLACE)):
        col = LEG_COLOR[phase]
        sub = legs[(legs.phase == phase) & legs.goal0_x.notna()]
        for _, L in sub.iterrows():
            c = cv[(cv.episode_id == L.episode_id) & (cv.phase == phase)]
            if c.empty:
                continue
            x = -L.goal0_x + (c.x - c.x.iloc[0]).to_numpy()
            y = -L.goal0_y + (c.y - c.y.iloc[0]).to_numpy()
            ax.plot(x, y, '-', color=col, alpha=0.35, linewidth=1)
            ax.plot(x[0], y[0], 'o', color=MUTED, markersize=3.5, alpha=0.8)
            ax.plot(x[-1], y[-1], 'o', color=col, markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.plot(0, 0, 'X', color=INK, markersize=10, zorder=5)
        sx, sy = (-sub.goal0_x).median(), (-sub.goal0_y).median()
        mx, my = (-sub.goal1_x).median(), (-sub.goal1_y).median()
        ax.plot(mx, my, '+', color=INK, markersize=12, markeredgewidth=1.5, zorder=6)
        ax.annotate(f'median start ({sx:.2f}, {sy:.2f})\nmedian stop ({mx:.2f}, {my:.2f})', (mx, my),
                    xytext=(10, -22), textcoords='offset points', fontsize=8, color=INK)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlim(-1.05, 0.12)
        ax.set_ylim((-0.95, 0.12) if phase == PREPLACE else (-0.12, 0.85))
        ax.set_xlabel('x, arm base relative to goal (m, arm-frame axes)')
        ax.set_ylabel('y, arm base relative to goal (m, arm-frame axes)')
        ax.set_title(LEG_LABEL[phase], fontsize=10, color=INK, loc='left')
        ax.legend(handles=[Line2D([], [], marker='X', color=INK, linestyle='none', label='goal (object / place pose)'),
                           Line2D([], [], marker='o', color=MUTED, linestyle='none', label='base at leg start'),
                           Line2D([], [], marker='o', color=col, linestyle='none', label='base when it stopped'),
                           Line2D([], [], color=col, alpha=0.5, label='commanded path (re-integrated)')],
                  fontsize=7.5, frameon=False, loc='lower left' if phase == PREPLACE else 'upper right')
    fig.suptitle('Where the controller takes the base, relative to the target it is scoring',
                 fontsize=11, color=INK, x=0.01, ha='left')
    save(fig, out_dir, 'fig_base_paths.png')


def fig_odometry_check(legs, out_dir,
                       title='The base executed the commands: goal shift vs integrated command, per leg'):
    fig, ax = new_fig(figsize=(5.2, 5.2))
    sub = legs[legs.dgoal_x.notna() & legs.dcmd_win_x.notna()]
    for phase in (PREGRASP, PREPLACE):
        s = sub[sub.phase == phase]
        ax.plot(s.dcmd_win_x, -s.dgoal_x, 'o', color=LEG_COLOR[phase], markersize=5, alpha=0.85,
                markeredgecolor=SURFACE, markeredgewidth=0.6, label=f'{LEG_LABEL[phase]}, x')
        ax.plot(s.dcmd_win_y, -s.dgoal_y, 's', color=LEG_COLOR[phase], markersize=5, alpha=0.85,
                markeredgecolor=SURFACE, markeredgewidth=0.6, label=f'{LEG_LABEL[phase]}, y')
    lim = (-0.35, 0.5)
    ax.plot(lim, lim, '-', color=AXIS, linewidth=1, zorder=0)
    X = np.concatenate([sub.dcmd_win_x, sub.dcmd_win_y])
    Y = -np.concatenate([sub.dgoal_x, sub.dgoal_y])
    k = np.sum(X * Y) / np.sum(X * X)
    r = np.corrcoef(X, Y)[0, 1]
    ax.text(0.03, 0.97, f'slope {k:.3f}, r = {r:.3f}, n = {len(X)} components',
            transform=ax.transAxes, va='top', fontsize=8.5, color=INK)
    ax.set_xlabel('commanded displacement, integrated /cmd_vel (m)')
    ax.set_ylabel('observed displacement, −Δgoal in arm frame (m)')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(fontsize=7.5, frameon=False, loc='lower right')
    ax.set_title(title, fontsize=9.5, color=INK, loc='left')
    save(fig, out_dir, 'fig_odometry_check.png')


def fig_q_legs(data, legs, out_dir, t_max=None, stop_label='base stops'):
    # `t_max` is accepted for call compatibility and no longer used: the x axis ends
    # with the longest leg instead of being clipped to a fixed horizon.
    ctrl = data['ctrl']
    fig, axes = new_fig(2, 3, figsize=(12, 6.8), sharey=True)
    series = [('q_center', 'Q (3-sphere region mean, controller objective)'),
              ('ws_mean_0', 'Q goal sphere'),
              ('qpath', 'Q path spheres (mean of the two)')]
    for r, phase in enumerate((PREGRASP, PREPLACE)):
        col = LEG_COLOR[phase]
        grid_t = np.arange(0, T_GRID_MAX + 1)
        stacks = {k: [] for k, _ in series}
        for _, L in legs[legs.phase == phase].iterrows():
            g = ctrl[(ctrl.episode_id == L.episode_id) & (ctrl.phase == phase) & ctrl.active
                     & (ctrl.region_size >= 3)].sort_values('timestamp')
            if len(g) < 2:
                continue
            g = g.assign(qpath=(g.ws_mean_1 + g.ws_mean_2) / 2)
            t = (g.timestamp - g.timestamp.iloc[0]).to_numpy()
            mv_end = g[g.moving].timestamp.max() - g.timestamp.iloc[0] if g.moving.any() else 0
            for c, (key, _) in enumerate(series):
                ax = axes[r, c]
                ax.plot(t, g[key], '-', color=col, alpha=0.25, linewidth=0.9)
                ax.plot(mv_end, np.interp(mv_end, t, g[key]), 'o', color=col, markersize=3, alpha=0.6)
                stacks[key].append(np.interp(grid_t, t, g[key], right=np.nan))
        t_end = 0
        v_min = min((np.nanmin(np.array(v)) for v in stacks.values() if v),
                    default=0.1)
        for c, (key, _) in enumerate(series):
            _, te, _ = median_over_legs(axes[r, c], grid_t, np.array(stacks[key]), INK,
                                        label='median across legs')
            t_end = max(t_end, te)
        for c, (key, label) in enumerate(series):
            ax = axes[r, c]
            ax.set_xlim(0, t_end + 1)   # the axis ends with the longest leg, not with a quorum
            ax.set_ylim(max(0.0, float(v_min) - 0.03), 1.02)   # no leg clipped off the bottom
            ax.set_xlabel('time since first full-region cycle (s)')
            if c == 0:
                ax.set_ylabel(f'{LEG_LABEL[phase]}\nQ')
            if r == 0:
                ax.set_title(label, fontsize=9.5, color=INK, loc='left')
    axes[0, 0].legend(handles=legend_survival_handles([], INK)
                      + [Line2D([], [], marker='o', color=BLUE, linestyle='none', label=stop_label)],
                      fontsize=7.5, frameon=False, loc='lower left')
    fig.suptitle('Reachability score over each approach leg (one thin line per leg, every leg shown '
                 'to its own end)', fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    save(fig, out_dir, 'fig_q_legs.png')


def fig_dq_legs(legs, out_dir):
    fig, axes = new_fig(1, 3, figsize=(11, 4.2), sharey=True)
    metrics = [('dq', 'ΔQ region mean (controller objective)'),
               ('dq_goal', 'ΔQ goal sphere'),
               ('dq_path', 'ΔQ path spheres')]
    rng = np.random.default_rng(0)
    for ax, (key, label) in zip(axes, metrics):
        for i, phase in enumerate((PREGRASP, PREPLACE)):
            s = legs[legs.phase == phase][key].dropna()
            x = i + rng.uniform(-0.18, 0.18, len(s))
            ax.plot(x, s, 'o', color=LEG_COLOR[phase], alpha=0.75, markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=0.6)
            med = s.median()
            ax.plot([i - 0.3, i + 0.3], [med, med], '-', color=INK, linewidth=2)
            ax.text(i + 0.32, med, f'median {med:+.3f}\n{(s > 0).mean() * 100:.0f}% > 0',
                    fontsize=7.5, va='center', color=INK)
        ax.axhline(0, color=AXIS, linewidth=1, zorder=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['pick approach', 'place approach'])
        ax.set_xlim(-0.6, 1.9)
        ax.set_title(label, fontsize=9.5, color=INK, loc='left')
    axes[0].set_ylabel('Q at base stop − Q at first full-region cycle')
    fig.suptitle('Change of the reachability score over each leg, first full cycle → base stop',
                 fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    save(fig, out_dir, 'fig_dq_legs.png')


def fig_q_landscape(data, legs, out_dir):
    """Q of the goal sphere vs where the goal sits in the arm frame (binned means)."""
    s = data['samples']
    s = s[s.phase.isin([PREGRASP, PREPLACE]) & (s.goal_age_s < 1.0)]
    fig, axes = new_fig(1, 3, figsize=(15, 4.8), gridspec_kw=dict(wspace=0.35))
    for ax, phase in zip(axes[:2], (PREGRASP, PREPLACE)):
        g = s[s.phase == phase]
        hb = ax.hexbin(g.goal_x, g.goal_y, C=g.q_goal, reduce_C_function=np.mean, gridsize=22,
                       cmap=SEQ_BLUE, vmin=0.3, vmax=0.95, mincnt=3, linewidths=0.2, edgecolors=SURFACE)
        L = legs[(legs.phase == phase) & legs.goal1_x.notna()]
        ax.plot(L.goal0_x, L.goal0_y, '+', color=INK, markersize=7, markeredgewidth=1.0,
                label='goal at leg start')
        ax.plot(L.goal1_x, L.goal1_y, 'o', markerfacecolor='none', markeredgecolor=CRITICAL,
                markersize=6, markeredgewidth=1.1, label='goal when the base stopped')
        z = g.goal_z.median()
        short = 'pick approach' if phase == PREGRASP else 'place approach'
        ax.set_title(f'{short}, goal z ≈ {z:.2f} m', fontsize=9.5, color=INK, loc='left')
        ax.set_xlabel('goal x in arm base frame (m)')
        ax.set_ylabel('goal y in arm base frame (m)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(fontsize=7.5, frameon=False, loc='upper right' if phase == PREGRASP else 'lower left')
    cb = fig.colorbar(hb, ax=axes[:2], shrink=0.85, pad=0.02)
    cb.set_label('mean Q goal sphere per cell (recorder, r = 0.10 m)', color=INK2, fontsize=8)
    cb.ax.tick_params(labelsize=8, colors=INK2)

    # Hysteresis panel: same goal x, outbound vs return, pick approach.
    ax = axes[2]
    rows = []
    for ep, g in s[s.phase == PREGRASP].groupby('episode_id'):
        g = g.sort_values('t_rel').copy()
        g['tl'] = g.t_rel - g.t_rel.min()
        g = g[g.tl < 30]
        t_xmax = g.tl.iloc[int(g.goal_x.argmax())]
        g['part'] = np.where(g.tl <= t_xmax, 'outbound (base backing away)', 'return (base coming back)')
        rows.append(g)
    d = pd.concat(rows)
    d['xb'] = (d.goal_x / 0.025).round() * 0.025
    for part, col in (('outbound (base backing away)', MUTED), ('return (base coming back)', BLUE)):
        b = d[d.part == part].groupby('xb').q_goal.agg(['mean', 'size'])
        b = b[b['size'] >= 8]
        ax.plot(b.index, b['mean'], '-o', color=col, markersize=3.5, linewidth=1.6, label=part)
    ax.set_xlabel('goal x in arm base frame (m), 2.5 cm bins')
    ax.set_ylabel('mean Q goal sphere')
    ax.set_ylim(0.3, 0.95)
    ax.legend(fontsize=7.5, frameon=False, loc='lower left')
    ax.set_title('Pick approach: same goal x, outbound vs return', fontsize=9.5, color=INK, loc='left')
    fig.suptitle('Measured reachability of the goal as a function of where it sits relative to the arm',
                 fontsize=11, color=INK, x=0.01, ha='left')
    save(fig, out_dir, 'fig_q_landscape.png')


def fig_gradient_validity(pairs, out_dir):
    fig, axes = new_fig(1, 3, figsize=(12, 4))
    ax = axes[0]
    for phase in (PREGRASP, PREPLACE):
        p = pairs[pairs.phase == phase]
        ax.plot(p.pred, p.meas, 'o', color=LEG_COLOR[phase], markersize=3, alpha=0.5,
                markeredgewidth=0, label=LEG_LABEL[phase])
    ax.axhline(0, color=AXIS, linewidth=1, zorder=0)
    ax.axvline(0, color=AXIS, linewidth=1, zorder=0)
    lim = 0.05
    ax.plot([-lim, lim], [-lim, lim], '-', color=AXIS, linewidth=1, zorder=0)
    ax.set_xlim(-0.002, 0.04)
    ax.set_ylim(-0.25, 0.25)
    ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04])
    agree = (np.sign(pairs.pred) == np.sign(pairs.meas)).mean()
    r = np.corrcoef(pairs.pred, pairs.meas)[0, 1]
    ax.text(0.03, 0.97, f'sign agreement {agree * 100:.0f}%\nPearson r = {r:.2f}\nn = {len(pairs)} cycle pairs',
            transform=ax.transAxes, va='top', fontsize=8, color=INK)
    ax.set_xlabel('predicted ΔQ over one cycle, ∇Q·v·Δt')
    ax.set_ylabel('measured ΔQ, Q(t+1) − Q(t)')
    ax.set_title('Per-cycle prediction vs measurement', fontsize=9.5, color=INK, loc='left')
    ax.legend(fontsize=7.5, frameon=False, loc='lower right', markerscale=2.5)

    ax = axes[1]
    bins = np.arange(0, 181, 10)
    for phase in (PREGRASP, PREPLACE):
        p = pairs[pairs.phase == phase]
        ax.hist(p.dang, bins=bins, color=LEG_COLOR[phase], alpha=0.6, label=LEG_LABEL[phase],
                edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(90, color=INK, linewidth=1, linestyle=':')
    ax.set_xlabel('turn of the commanded direction between consecutive cycles (deg)')
    ax.set_ylabel('cycles')
    med = pairs.dang.median()
    ax.text(0.97, 0.97, f'median {med:.0f}°\n{(pairs.dang > 90).mean() * 100:.1f}% > 90° (reversal)',
            transform=ax.transAxes, va='top', ha='right', fontsize=8, color=INK)
    ax.set_title('Direction stability of ∇Q', fontsize=9.5, color=INK, loc='left')
    ax.legend(fontsize=7.5, frameon=False, loc='center right')

    ax = axes[2]
    bins = np.arange(0, 1.05, 0.05)
    for phase in (PREGRASP, PREPLACE):
        p = pairs[pairs.phase == phase]
        ax.hist(p.grad_mag.clip(upper=1.0), bins=bins, color=LEG_COLOR[phase], alpha=0.6,
                label=LEG_LABEL[phase], edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(0.025, color=INK, linewidth=1, linestyle=':')
    ax.text(0.03, 0.97, 'min_gradient = 0.025 1/m', transform=ax.transAxes, va='top', fontsize=8, color=INK)
    ax.set_xlabel('|∇Q| (1/m), clipped at 1.0')
    ax.set_ylabel('cycles')
    ax.set_title('Gradient magnitude while moving', fontsize=9.5, color=INK, loc='left')
    fig.tight_layout()
    save(fig, out_dir, 'fig_gradient_validity.png')


def fig_timing(data, out_dir):
    ctrl, mpc, samples = data['ctrl'], data['mpc'], data['samples']
    fig, axes = new_fig(1, 3, figsize=(12, 3.8))
    ax = axes[0]
    c = ctrl[ctrl.active].cycle_s.dropna() * 1000
    ax.hist(c, bins=np.arange(400, 1000, 20), color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(1000, color=INK, linewidth=1, linestyle=':')
    ax.text(0.97, 0.97, f'mean {c.mean():.0f} ms, max {c.max():.0f} ms\nperiod 1000 ms',
            transform=ax.transAxes, va='top', ha='right', fontsize=8, color=INK)
    ax.set_xlabel('gradient cycle duration (ms): 9 candidate maps + region scoring')
    ax.set_ylabel('cycles')
    ax.set_title('Base controller cycle time', fontsize=9.5, color=INK, loc='left')

    ax = axes[1]
    m = mpc.solve_ms.dropna()
    ax.hist(m, bins=np.arange(0, 660, 15), color=ORANGE, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(400, color=INK, linewidth=1, linestyle=':')
    over = mpc.budget_exceeded.astype(bool).mean()
    ax.text(0.97, 0.97, f'mean {m.mean():.0f} ms, p95 {m.quantile(0.95):.0f} ms\n{over * 100:.1f}% over the 400 ms budget',
            transform=ax.transAxes, va='top', ha='right', fontsize=8, color=INK)
    ax.set_xlabel('MPC solve time (ms)')
    ax.set_title('Arm MPC solve time during episodes', fontsize=9.5, color=INK, loc='left')

    ax = axes[2]
    s = samples.compute_ms.dropna()
    ax.hist(s, bins=np.arange(0, 220, 5), color=AQUA, edgecolor=SURFACE, linewidth=0.5)
    ax.text(0.97, 0.97, f'median {s.median():.0f} ms, p95 {s.quantile(0.95):.0f} ms',
            transform=ax.transAxes, va='top', ha='right', fontsize=8, color=INK)
    ax.set_xlabel('recorder scoring time per path message (ms)')
    ax.set_title('Recorder Q_goal + Q_path scoring', fontsize=9.5, color=INK, loc='left')
    fig.tight_layout()
    save(fig, out_dir, 'fig_timing.png')


def fig_outcome_diag(data, out_dir, classes=('success', 'collision'),
                     title='Place-approach diagnostics: completed cycles vs cycles ending in a collision'):
    mpc, idx = data['mpc'], data['idx']
    idx = outcome_valid(idx)
    cls_map = idx.set_index('episode_id').cls
    m = mpc[mpc.episode_id.isin(idx.episode_id)].copy()
    m['cls'] = m.episode_id.map(cls_map)
    per = m[m.phase == PREPLACE].groupby(['episode_id', 'cls']).agg(
        fk_err_last=('arrival_fk_err_m', lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan),
        scene_col_frac=('con_scene_collision', lambda s: (s > 0).mean()),
        n=('solve_ms', 'size')).reset_index()
    fig, axes = new_fig(1, 2, figsize=(9 + 1.5 * (len(classes) - 2), 4))
    rng = np.random.default_rng(1)
    for ax, (key, label) in zip(axes, [('fk_err_last', 'arm-to-target FK error at the end of the place approach (m)'),
                                       ('scene_col_frac', 'fraction of MPC steps with an active scene-collision constraint')]):
        for i, c in enumerate(classes):
            s = per[per.cls == c][key].dropna()
            if s.empty:
                continue
            x = i + rng.uniform(-0.15, 0.15, len(s))
            ax.plot(x, s, 'o', color=CLASS_COLOR[c], markersize=5.5, alpha=0.85,
                    markeredgecolor=SURFACE, markeredgewidth=0.6)
            ax.plot([i - 0.28, i + 0.28], [s.median()] * 2, '-', color=INK, linewidth=2)
            ax.text(i + 0.3, s.median(), f'median {s.median():.3f}', fontsize=7.5, va='center', color=INK)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([f'{c}\n(n={int((per.cls == c).sum())})' for c in classes], fontsize=8)
        ax.set_xlim(-0.6, len(classes) - 1 + 0.9)
        ax.set_ylabel(label, fontsize=8)
    axes[0].set_title('Did the arm reach the pre-place pose?', fontsize=9.5, color=INK, loc='left')
    axes[1].set_title('Was the obstacle map fighting the MPC?', fontsize=9.5, color=INK, loc='left')
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    save(fig, out_dir, 'fig_outcome_diag.png')
    return per


def fig_arm_metrics(data, out_dir):
    phases, idx = data['phases'], data['idx']
    valid = outcome_valid(idx).episode_id
    p = phases[phases.episode_id.isin(valid)]
    fig, axes = new_fig(1, 2, figsize=(11, 4))
    ax = axes[0]
    order = [PREGRASP, PREPLACE, 'act_home_exit']
    names = ['pick approach', 'place approach', 'final homing']
    sub = p[p.phase.isin(order) & (p.n_samples > 0)]
    pos = np.arange(len(order))
    for off, key, col, lab in ((-0.18, 'q_goal_mean', BLUE, 'Q goal sphere (phase mean)'),
                               (0.18, 'q_path_mean', ORANGE, 'Q path spheres (phase mean)')):
        vals = [sub[sub.phase == ph][key].dropna().to_numpy() for ph in order]
        bp = ax.boxplot(vals, positions=pos + off, widths=0.3, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, linewidth=1.5),
                        boxprops=dict(facecolor=col, alpha=0.6, linewidth=0.8, edgecolor=col),
                        whiskerprops=dict(color=col, linewidth=1), capprops=dict(color=col, linewidth=1),
                        flierprops=dict(marker='o', markersize=3, markerfacecolor=col, markeredgewidth=0))
    ax.set_xticks(pos)
    ax.set_xticklabels(names)
    ax.set_ylabel('Q')
    ax.set_ylim(0.5, 1.02)
    ax.legend(handles=[Patch(color=BLUE, alpha=0.6, label='Q goal sphere (phase mean)'),
                       Patch(color=ORANGE, alpha=0.6, label='Q path spheres (phase mean)')],
              fontsize=7.5, frameon=False, loc='lower left')
    ax.set_title('Recorder reachability per phase (valid episodes)', fontsize=9.5, color=INK, loc='left')
    ax.text(0.99, 0.02, 'final homing: goal is stale (last place pose)', transform=ax.transAxes,
            fontsize=7, ha='right', color=MUTED)

    ax = axes[1]
    order2 = ['act_home_enter', PREGRASP, 'act_grasp:act_move_grasp', PREPLACE, 'act_release:act_move_place', 'act_home_exit']
    names2 = ['home\nenter', 'pick\napproach', 'grasp\nmove', 'place\napproach', 'place\nmove', 'home\nexit']
    vals = [p[p.phase == ph].manipulability_mean.dropna().to_numpy() for ph in order2]
    ax.boxplot(vals, positions=np.arange(len(order2)), widths=0.5, patch_artist=True,
               medianprops=dict(color=INK, linewidth=1.5),
               boxprops=dict(facecolor=AQUA, alpha=0.6, linewidth=0.8, edgecolor=AQUA),
               whiskerprops=dict(color=AQUA, linewidth=1), capprops=dict(color=AQUA, linewidth=1),
               flierprops=dict(marker='o', markersize=3, markerfacecolor=AQUA, markeredgewidth=0))
    ax.set_xticks(np.arange(len(order2)))
    ax.set_xticklabels(names2, fontsize=8)
    ax.set_ylabel('Yoshikawa manipulability (phase mean)')
    ax.set_title('Arm manipulability per phase', fontsize=9.5, color=INK, loc='left')
    fig.tight_layout()
    save(fig, out_dir, 'fig_arm_metrics.png')


def _gap_to_nan(t, y, max_gap):
    """Insert NaN where consecutive samples are further apart than max_gap (no line across gaps)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t) < 2:
        return t, y
    gaps = np.where(np.diff(t) > max_gap)[0]
    t_out, y_out = list(t), list(y)
    for i in sorted(gaps, reverse=True):
        t_out.insert(i + 1, t[i] + 1e-3)
        y_out.insert(i + 1, np.nan)
    return np.array(t_out), np.array(y_out)


def fig_episode_example(data, cv, out_dir, ep=0):
    """One episode end-to-end: Q from the controller, base speed, goal position."""
    ctrl, idx, phases, samples = data['ctrl'], data['idx'], data['phases'], data['samples']
    e = idx[idx.episode_id == ep].iloc[0]
    g = ctrl[(ctrl.episode_id == ep) & ctrl.active & (ctrl.region_size >= 3)].sort_values('timestamp')
    s = samples[(samples.episode_id == ep) & (samples.goal_age_s < 1.0)].sort_values('t_rel')
    c = cv[cv.episode_id == ep]
    fig, axes = new_fig(3, 1, figsize=(10, 7.5), sharex=True)
    for ax in axes:
        for _, p in phases[phases.episode_id == ep].iterrows():
            if p.brain_state in PHASE_COLOR and p.duration_s > 1:
                ax.axvspan(p.t_start_rel, p.t_end_rel, color=PHASE_COLOR[p.brain_state], alpha=0.12, linewidth=0)
    t = (g.timestamp - e.t_start).to_numpy()
    for key, col, lw, label in (('q_center', INK, 1.8, 'Q region mean (controller objective)'),
                                ('ws_mean_0', BLUE, 1.2, 'Q goal sphere'),
                                ('qpath', ORANGE, 1.2, 'Q path spheres')):
        y = g[key] if key != 'qpath' else (g.ws_mean_1 + g.ws_mean_2) / 2
        tt, yy = _gap_to_nan(t, y, 3.0)
        axes[0].plot(tt, yy, '-', color=col, linewidth=lw, label=label)
    axes[0].set_ylabel('Q')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7.5, frameon=False, loc='lower left', ncol=3)
    axes[1].plot(c.t_rel, np.hypot(c.vx, c.vy) * 100, '-', color=INK, linewidth=1)
    axes[1].set_ylabel('|v| commanded (cm/s)')
    axes[1].set_ylim(-0.1, 2.5)
    for key, col, label in (('goal_x', BLUE, 'goal x (arm frame)'), ('goal_y', ORANGE, 'goal y (arm frame)')):
        tt, yy = _gap_to_nan(s.t_rel, s[key], 3.0)
        axes[2].plot(tt, yy, '-', color=col, linewidth=1.2, label=label)
    axes[2].set_ylabel('goal position (m)')
    axes[2].set_xlabel('time since episode start (s)')
    axes[2].legend(fontsize=7.5, frameon=False, loc='center right')
    axes[0].set_title(f'Episode {ep} ({e.cls}): controller score, commanded speed and goal position over the cycle',
                      fontsize=10, color=INK, loc='left')
    handles = [Patch(color=PHASE_COLOR[k], alpha=0.3, label=k) for k in PHASE_COLOR]
    axes[1].legend(handles=handles, fontsize=7.5, frameon=False, loc='upper right', ncol=4)
    fig.tight_layout()
    save(fig, out_dir, f'fig_episode_{ep:03d}.png')


# ── Tables ──────────────────────────────────────────────────────────────────

def md_table(df, floatfmt='{:.3f}'):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append('' if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def write_tables(data, legs, pairs, per_preplace, out_dir):
    idx, phases = data['idx'], data['phases']
    last = phases.sort_values('t_start_rel').groupby('episode_id').last().phase
    ep = idx[['episode_id', 'duration_s', 'n_phases', 'outcome', 'cls', 'note']].copy()
    ep['last_phase'] = ep.episode_id.map(last)
    for phase, tag in ((PREGRASP, 'pick'), (PREPLACE, 'place')):
        L = legs[legs.phase == phase].set_index('episode_id')
        ep[f'{tag}_move_s'] = ep.episode_id.map(L.t_moving_s)
        ep[f'{tag}_dist_m'] = ep.episode_id.map(L.dist_m)
        ep[f'{tag}_net_m'] = ep.episode_id.map(L.net_m)
        ep[f'{tag}_dQ'] = ep.episode_id.map(L.dq)
        ep[f'{tag}_stop'] = ep.episode_id.map(L.stop)
    ep.to_csv(os.path.join(out_dir, 'episodes.csv'), index=False)
    with open(os.path.join(out_dir, 'episodes.md'), 'w') as f:
        f.write(md_table(ep.drop(columns=['note']).round(3)))
        f.write('\n')
    legs.round(4).to_csv(os.path.join(out_dir, 'legs.csv'), index=False)
    pairs.round(5).to_csv(os.path.join(out_dir, 'cycle_pairs.csv'), index=False)
    per_preplace.round(4).to_csv(os.path.join(out_dir, 'preplace_mpc_diag.csv'), index=False)

    # Headline numbers
    valid = outcome_valid(idx)
    succ = idx[idx.cls == 'success']
    summary = {
        'episodes_recorded': int(len(idx)),
        'episodes_excluded': int(idx.cls.isin(OUTCOME_EXCLUDED).sum()),
        'episodes_valid': int(len(valid)),
        'success': int((idx.cls == 'success').sum()),
        'bad_grasp_cancel': int((idx.cls == 'bad-grasp cancel').sum()),
        'grasp_release_failed': int((idx.cls == 'grasp/release failed').sum()),
        'collision': int((idx.cls == 'collision').sum()),
        'class_counts': {k: int(v) for k, v in idx.cls.value_counts().items()},
        'episodes_excluded_grasp': int(idx.cls.isin(GRASP_CLASSES).sum()),
        'success_rate': float((idx.cls == 'success').sum() / len(valid)),
        'cycle_duration_success_s': {k: float(v) for k, v in succ.duration_s.describe().items()},
        'controller_cycles_active': int(data['ctrl'].active.sum()),
        'controller_cycles_moving': int(data['ctrl'].moving.sum()),
        'controller_cycle_ms_mean': float(data['ctrl'][data['ctrl'].active].cycle_s.mean() * 1000),
        'controller_cycle_ms_max': float(data['ctrl'][data['ctrl'].active].cycle_s.max() * 1000),
        'mpc_solve_ms_mean': float(data['mpc'].solve_ms.mean()),
        'mpc_solve_ms_p95': float(data['mpc'].solve_ms.quantile(0.95)),
        'mpc_budget_exceeded_frac': float(data['mpc'].budget_exceeded.astype(bool).mean()),
        'mpc_steps': int(len(data['mpc'])),
        'total_commanded_distance_m': float(legs.dist_m.sum()),
        'cycle_pairs': int(len(pairs)),
        'cycle_sign_agreement': float((np.sign(pairs.pred) == np.sign(pairs.meas)).mean()),
        'cycle_pearson_r': float(np.corrcoef(pairs.pred, pairs.meas)[0, 1]),
        'cycle_pred_dq_median': float(pairs.pred.median()),
        'cycle_meas_dq_std': float(pairs.meas.std()),
        'direction_turn_median_deg': float(pairs.dang.median()),
        'direction_reversal_frac': float((pairs.dang > 90).mean()),
    }
    sub = legs[legs.dgoal_x.notna() & legs.dcmd_win_x.notna()]
    X = np.concatenate([sub.dcmd_win_x, sub.dcmd_win_y])
    Y = -np.concatenate([sub.dgoal_x, sub.dgoal_y])
    summary['odometry_check_slope'] = float(np.sum(X * Y) / np.sum(X * X)) if len(X) else np.nan
    summary['odometry_check_r'] = float(np.corrcoef(X, Y)[0, 1]) if len(X) > 2 else np.nan
    for phase, tag in ((PREGRASP, 'pick'), (PREPLACE, 'place')):
        L = legs[legs.phase == phase]
        summary[f'legs_{tag}'] = {
            'n': int(len(L)),
            'n_moving_cycles_median': float(L.n_moving.median()),
            't_moving_s_median': float(L.t_moving_s.median()),
            'phase_duration_s_median': float(L.phase_duration_s.median()),
            'dist_m_median': float(L.dist_m.median()),
            'net_m_median': float(L.net_m.median()),
            'path_efficiency_median': float(L.path_efficiency.median()),
            'q_start_median': float(L.q_start.median()),
            'q_end_median': float(L.q_end.median()),
            'dq_median': float(L.dq.median()),
            'dq_pos_frac': float((L.dq > 0).mean()),
            'dq_goal_median': float(L.dq_goal.median()),
            'dq_goal_pos_frac': float((L.dq_goal > 0).mean()),
            'dq_path_median': float(L.dq_path.median()),
            'qgoal_start_median': float(L.qgoal_start.median()),
            'qgoal_end_median': float(L.qgoal_end.median()),
            'grad_med_median': float(L.grad_med.median()),
            'stop_reasons': {k: int(v) for k, v in L.stop.value_counts().items()},
            'goal0_median_xy': [float(L.goal0_x.median()), float(L.goal0_y.median())],
            'goal1_median_xy': [float(L.goal1_x.median()), float(L.goal1_y.median())],
            'goal0_std_xy': [float(L.goal0_x.std()), float(L.goal0_y.std())],
            'goal1_std_xy': [float(L.goal1_x.std()), float(L.goal1_y.std())],
            'reversals_median': float(L.n_reversals.median()),
        }
    pp = per_preplace
    summary['preplace_diag'] = {
        c: {'fk_err_last_median': float(pp[pp.cls == c].fk_err_last.median()),
            'scene_col_frac_median': float(pp[pp.cls == c].scene_col_frac.median()),
            'n': int((pp.cls == c).sum())}
        for c in CLASS_ORDER if (pp.cls == c).any()}
    # Phase durations of successful cycles
    ps = phases[phases.episode_id.isin(succ.episode_id)]
    summary['phase_duration_success_median_s'] = {
        k: float(v) for k, v in ps.groupby('phase').duration_s.median().items()}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, 'legs.md'), 'w') as f:
        cols = ['episode_id', 'leg', 'cls', 'n_cycles', 'n_moving', 't_moving_s', 'phase_duration_s',
                'dist_m', 'net_m', 'path_efficiency', 'q_start', 'q_end', 'dq', 'qgoal_start', 'qgoal_end',
                'dq_goal', 'grad_med', 'n_reversals', 'stop']
        f.write(md_table(legs[cols].round(3)))
        f.write('\n')
    print(f'  wrote {os.path.basename(out_dir)}/*.csv, *.md, summary.json')
    return summary


# ── Main ────────────────────────────────────────────────────────────────────

def run(run_dir, out, notes=EPISODE_NOTES, default_cancel=DEFAULT_CANCEL, default_failure=None,
        base_moves=True, diag_classes=('success', 'collision'),
        diag_title='Place-approach diagnostics: completed cycles vs cycles ending in a collision',
        example_eps=(0, 34), timeline_t_max=300, leg_t_max=40, loader=None,
        fig_dirname='figures', tab_dirname='tables'):
    """Whole pipeline for one run; returns (data, cv, legs, pairs, summary).

    `base_moves=False` (fixed-base run) skips the figures that only make sense
    when the base actually followed the command (paths, Q landscape). `loader`
    replaces `load` when a run needs its own post-processing of the index.
    `fig_dirname`/`tab_dirname` name the output directories inside `out`; each
    run writes into its own run directory, so `figures`/`tables` already say which.
    """
    fig_dir, tab_dir = os.path.join(out, fig_dirname), os.path.join(out, tab_dirname)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    print('loading', run_dir)
    if loader is not None:
        data = loader(run_dir)
    else:
        data = load(run_dir, notes=notes, default_cancel=default_cancel, default_failure=default_failure)
    cv = integrate_cmd_vel(data['cmd_vel'])
    legs = build_legs(data, cv)
    pairs = build_cycle_pairs(data['ctrl'])

    print('figures ->', fig_dir)
    fig_timeline(data, legs, fig_dir, t_max=timeline_t_max, moving_label=(
        'base moving (controller cycle)' if base_moves else 'motion commanded (controller cycle)'))
    fig_odometry_check(legs, fig_dir, title=(
        'The base executed the commands: goal shift vs integrated command, per leg' if base_moves else
        'The base did not follow the commands: goal shift vs integrated command, per leg'))
    if base_moves:
        fig_base_paths(data, legs, cv, fig_dir)
        fig_q_landscape(data, legs, fig_dir)
    fig_q_legs(data, legs, fig_dir, t_max=leg_t_max,
               stop_label='base stops' if base_moves else 'command stops')
    fig_dq_legs(legs, fig_dir)
    fig_gradient_validity(pairs, fig_dir)
    fig_timing(data, fig_dir)
    per_preplace = fig_outcome_diag(data, fig_dir, classes=diag_classes, title=diag_title)
    fig_arm_metrics(data, fig_dir)
    for ep in example_eps:
        fig_episode_example(data, cv, fig_dir, ep=ep)

    print('tables ->', tab_dir)
    summary = write_tables(data, legs, pairs, per_preplace, tab_dir)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=2))
    return data, cv, legs, pairs, summary


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import runs

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', default=runs.SETUP_1_MOVING,
                    help='<experience>/<run_id> under results/, or a path (default: %(default)s)')
    ap.add_argument('--out', default=None, help='output root (default: the run directory)')
    args = ap.parse_args()
    run(runs.raw(args.run_dir), args.out or runs.run_dir(args.run_dir))


if __name__ == '__main__':
    main()
