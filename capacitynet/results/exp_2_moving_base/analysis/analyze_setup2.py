#!/usr/bin/env python3
"""Setup 2 (cluttered scene): fixed base vs gradient-controlled base, and their comparison.

Two runs of the same pick-and-place task in the same cluttered setup, with the
same controller configuration; the only difference is whether the base's motor
relay was enabled:

    exp2_base_fixe     2026-09-18, base immobilised (Q, dQ and /cmd_vel still computed)
    exp_2_moving_base  2026-09-21, base free to follow the gradient

    python3 analysis/analyze_setup2.py                     # both runs + comparison
    python3 analysis/analyze_setup2.py --fixed-dir X --moving-dir Y --out Z

Writes, for each run, `exp_2_<condition>_figures/fig_*.png` and
`exp_2_<condition>_tables/*` inside that run's own directory, and the comparison
(`exp_2_comparison_figures/cmp_*.png`, `exp_2_comparison_tables/cmp_*`) into --out
(default: the moving-base run directory, next to REPORT.md).

Outcome classes (source: experimenter, 2026-09-21; `episodes_index.csv` only
records how `brain` ended the cycle, not why):

* success  - pick and place both completed. A cycle cancelled (or whose homing
  failed) *after* the place counts here, as in the setup-1 runs.
* collision - the arm was blocked by / collided with the clutter and never
  reached the target pose; the cycle was cancelled.
* excluded - not a valid attempt: infrastructure failures (`set_planner`
  unavailable, home planning failed), a gripper failure with the arm correctly
  posed, the cycle the run was stopped in, and the episodes whose per-episode
  recording the operator deleted (20-23 of the moving-base run).

The `collision` class is what makes setup 2 different from setup 1: there the
failures were grasp-quality problems, here the arm cannot get through the
clutter to a target the base has not repositioned for.
"""

import argparse
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import reach_analysis as R  # noqa: E402

DEFAULT_MOVING = os.path.dirname(HERE)
DEFAULT_FIXED = os.path.join(os.path.dirname(DEFAULT_MOVING), 'exp2_base_fixe')

HOMING_CANCEL = ('success', 'pick-and-place completed; cancelled during the final homing')
BLOCKED = 'arm blocked by the clutter during the place approach; pre-place pose never reached'

FIXED_NOTES = {
    15: ('excluded', 'infrastructure: set_planner service unavailable at cycle start (4 ms episode)'),
    0: ('collision', BLOCKED),
    6: ('collision', BLOCKED),
    7: ('collision', BLOCKED),
    11: ('collision', BLOCKED),
    12: ('collision', BLOCKED),
    17: ('collision', BLOCKED + '; release sequence failed'),
    1: ('collision', 'cancelled 10 s into the pick approach, arm still 0.62 m from the target'),
    14: ('success', 'pick-and-place completed; the final homing failed (execution canceled)'),
}
FIXED_NOTES.update({ep: HOMING_CANCEL for ep in (2, 3, 4, 5, 8, 9, 10, 13, 16)})

MOVING_NOTES = {
    0: ('excluded', 'infrastructure: set_planner service unavailable at cycle start (9 ms episode)'),
    6: ('excluded', 'infrastructure: home planning failed at cycle start (14 s episode)'),
    7: ('excluded', 'infrastructure: home planning failed at cycle start (14 s episode)'),
    8: ('excluded', 'infrastructure: home planning failed at cycle start (14 s episode)'),
    18: ('excluded', 'gripper failure: arm on target (FK error 1 mm), grasp sequence failed'),
    29: ('excluded', 'run stopped mid-cycle (outcome unknown, 489 s)'),
    1: ('collision', 'cancelled 28 s into the pick approach, arm still 0.62 m from the target'),
    5: ('collision', BLOCKED + ' (scene-collision constraint active in 32 % of MPC steps)'),
    19: HOMING_CANCEL,
}
# Episodes whose per-episode recording the operator deleted: index and phase rows
# survive, the CSVs do not. Excluded so that leg statistics and episode counts
# describe the same set of cycles.
MOVING_NOTES.update({ep: ('excluded', 'per-episode recording deleted by the operator (invalid attempt)')
                     for ep in (20, 21, 22, 23)})

RUNS = [('base fixed', R.ORANGE), ('base moving (gradient)', R.BLUE)]
PHASES = [('act_home_enter', 'home\nenter'), (R.PREGRASP, 'pick\napproach'),
          ('act_grasp:act_move_grasp', 'grasp\nmove'), (R.PREPLACE, 'place\napproach'),
          ('act_release:act_move_place', 'place\nmove'), ('act_home_exit', 'home\nexit')]
LEG_T_MAX = {R.PREGRASP: 25, R.PREPLACE: 120}
DIR_BIN_S = {R.PREGRASP: 5.0, R.PREPLACE: 10.0}
CMP_CLASSES = ['success', 'collision']


# ── Loading ─────────────────────────────────────────────────────────────────

def load_fixed(run_dir):
    return R.load(run_dir, notes=FIXED_NOTES, default_cancel=('collision', BLOCKED),
                  default_failure=('collision', BLOCKED))


def load_moving(run_dir):
    return R.load(run_dir, notes=MOVING_NOTES, default_cancel=('collision', BLOCKED),
                  default_failure=('collision', BLOCKED))


def load_both(fixed_dir, moving_dir):
    """Both runs with their legs, cycle pairs and per-episode MPC diagnostics."""
    out = []
    for (name, _), loader, d in zip(RUNS, (load_fixed, load_moving), (fixed_dir, moving_dir)):
        data = loader(d)
        cv = R.integrate_cmd_vel(data['cmd_vel'])
        legs = R.build_legs(data, cv)
        valid = R.outcome_valid(data['idx']).episode_id
        legs = legs[legs.episode_id.isin(valid)].copy()
        pairs = R.build_cycle_pairs(data['ctrl'][data['ctrl'].episode_id.isin(valid)])
        data.update(name=name, run_dir=d, cv=cv, legs=legs, pairs=pairs)
        out.append(data)
    return out


# ── Small helpers ───────────────────────────────────────────────────────────

def strip(ax, x, values, color, width=0.32, rng=None, fmt='{:.2f}'):
    """Jittered dot column with a median bar; returns the median."""
    rng = rng or np.random.default_rng(0)
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return np.nan
    ax.plot(x + rng.uniform(-width * 0.6, width * 0.6, len(v)), v, 'o', color=color, alpha=0.7,
            markersize=4.5, markeredgecolor=R.SURFACE, markeredgewidth=0.5)
    med = float(np.median(v))
    ax.plot([x - width, x + width], [med, med], '-', color=R.INK, linewidth=2)
    ax.text(x, med, ' ' + fmt.format(med), fontsize=7, va='bottom', ha='left', color=R.INK)
    return med


def leg_curves(data, phase, key, t_max):
    """Per-leg series resampled on a 1 s grid since the first full-region cycle."""
    ctrl, legs = data['ctrl'], data['legs']
    grid_t = np.arange(0, t_max + 1)
    stack = []
    for _, L in legs[legs.phase == phase].iterrows():
        g = ctrl[(ctrl.episode_id == L.episode_id) & (ctrl.phase == phase) & ctrl.active
                 & (ctrl.region_size >= 3)].sort_values('timestamp')
        if len(g) < 2:
            continue
        g = g.assign(qpath=(g.ws_mean_1 + g.ws_mean_2) / 2)
        t = (g.timestamp - g.timestamp.iloc[0]).to_numpy()
        stack.append(np.interp(grid_t, t, g[key], right=np.nan))
    return grid_t, np.array(stack)


def leg_curves_progress(data, phase, key, n_pts=101):
    """Per-leg series resampled on a 0-100 % grid of the leg's own duration.

    0 % is the first full-region cycle, 100 % the last active full-region cycle of the leg,
    so every leg spans the whole axis and the median is always taken over all of them.
    """
    ctrl, legs = data['ctrl'], data['legs']
    grid_p = np.linspace(0, 100, n_pts)
    stack = []
    for _, L in legs[legs.phase == phase].iterrows():
        g = ctrl[(ctrl.episode_id == L.episode_id) & (ctrl.phase == phase) & ctrl.active
                 & (ctrl.region_size >= 3)].sort_values('timestamp')
        if len(g) < 2:
            continue
        g = g.assign(qpath=(g.ws_mean_1 + g.ws_mean_2) / 2)
        t = (g.timestamp - g.timestamp.iloc[0]).to_numpy()
        stack.append(np.interp(grid_p, 100 * t / t[-1], g[key]))
    return grid_p, np.array(stack)


def heading_over_leg(data, phase, bin_s=10.0, t_max=120.0):
    """Per-leg circular-mean heading of ∇Q in bins of `bin_s` seconds since the leg's first full cycle."""
    ctrl, legs = data['ctrl'], data['legs']
    bins = np.arange(0, t_max, bin_s)
    rows = []
    for ep in legs[legs.phase == phase].episode_id:
        g = ctrl[(ctrl.episode_id == ep) & (ctrl.phase == phase) & ctrl.active
                 & (ctrl.region_size >= 3)].sort_values('timestamp')
        if g.empty:
            continue
        t = (g.timestamp - g.timestamp.iloc[0]).to_numpy()
        ang = np.arctan2(g.grad_y.to_numpy(), g.grad_x.to_numpy())
        rows.append([np.degrees(np.arctan2(np.sin(ang[(t >= b) & (t < b + bin_s)]).mean(),
                                           np.cos(ang[(t >= b) & (t < b + bin_s)]).mean()))
                     if ((t >= b) & (t < b + bin_s)).any() else np.nan for b in bins])
    H = np.array(rows)
    s, c = np.nanmean(np.sin(np.radians(H)), axis=0), np.nanmean(np.cos(np.radians(H)), axis=0)
    return bins, np.degrees(np.arctan2(s, c)), np.hypot(s, c), np.sum(~np.isnan(H), axis=0)


def mpc_per_leg(data, phase):
    """One row per (episode, leg): arrival error, constraint activity, solve time, duration."""
    mpc, idx = data['mpc'], data['idx']
    valid = R.outcome_valid(idx).episode_id
    m = mpc[(mpc.phase == phase) & mpc.episode_id.isin(valid)].copy()
    m['cls'] = m.episode_id.map(idx.set_index('episode_id').cls)
    return m.groupby(['episode_id', 'cls']).agg(
        fk_err_last=('arrival_fk_err_m', lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan),
        scene_col_frac=('con_scene_collision', lambda s: (s > 0).mean()),
        solve_ms=('solve_ms', 'mean'), n=('solve_ms', 'size'),
        dur_s=('t_rel', lambda s: s.max() - s.min())).reset_index()


# ── Comparison figures ──────────────────────────────────────────────────────

def fig_outcomes(runs, out_dir):
    fig, ax = R.new_fig(figsize=(10, 3.0))
    ax.grid(False)
    rows = []
    for y, d in enumerate(runs):
        idx = d['idx']
        valid = R.outcome_valid(idx)
        left = 0.0
        for c in CMP_CLASSES:
            n = int((valid.cls == c).sum())
            if not n:
                continue
            frac = n / len(valid)
            ax.barh(y, frac, left=left, height=0.55, color=R.CLASS_COLOR[c], linewidth=0)
            ax.text(left + frac / 2, y, f'{n}\n{frac * 100:.0f}%', ha='center', va='center',
                    fontsize=8, color=R.SURFACE, fontweight='bold')
            left += frac
            rows.append(dict(run=d['name'], cls=c, n=n, frac=frac))
        ax.text(1.01, y, f'n = {len(valid)} valid\n(+{int((idx.cls == "excluded").sum())} excluded)',
                va='center', fontsize=7.5, color=R.INK2)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([d['name'] for d in runs], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['0', '25 %', '50 %', '75 %', '100 %'])
    ax.legend(handles=[Patch(color=R.CLASS_COLOR[c], label=c) for c in CMP_CLASSES],
              ncol=2, fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    ax.set_title('Setup 2 (cluttered scene): outcome of the valid pick-and-place cycles',
                 fontsize=10, color=R.INK, loc='left')
    R.save(fig, out_dir, 'cmp_outcomes.png')
    return pd.DataFrame(rows)


def fig_phase_durations(runs, out_dir):
    fig, axes = R.new_fig(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[4, 1]))
    rng = np.random.default_rng(1)
    rows = []
    for k, d in enumerate(runs):
        succ = d['idx'][d['idx'].cls == 'success'].episode_id
        p = d['phases'][d['phases'].episode_id.isin(succ)]
        for i, (ph, _) in enumerate(PHASES):
            v = p[p.phase == ph].groupby('episode_id').duration_s.sum()
            med = strip(axes[0], i + (k - 0.5) * 0.36, v, RUNS[k][1], width=0.16, rng=rng, fmt='{:.0f}')
            rows.append(dict(run=d['name'], phase=ph, n=len(v), median_s=med, mean_s=v.mean(),
                             q25_s=v.quantile(0.25), q75_s=v.quantile(0.75)))
        v = d['idx'][d['idx'].cls == 'success'].duration_s
        med = strip(axes[1], k, v, RUNS[k][1], width=0.3, rng=rng, fmt='{:.0f}')
        rows.append(dict(run=d['name'], phase='whole cycle', n=len(v), median_s=med, mean_s=v.mean(),
                         q25_s=v.quantile(0.25), q75_s=v.quantile(0.75)))
    axes[0].set_xticks(range(len(PHASES)))
    axes[0].set_xticklabels([n for _, n in PHASES], fontsize=8)
    axes[0].set_ylabel('phase duration (s), successful cycles')
    axes[0].set_title('Phase durations of the successful cycles (one dot per cycle, bar = median)',
                      fontsize=9.5, color=R.INK, loc='left')
    axes[0].legend(handles=[Line2D([], [], marker='o', linestyle='none', color=c, label=n) for n, c in RUNS],
                   fontsize=7.5, frameon=False, loc='upper left')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['base\nfixed', 'base\nmoving'], fontsize=8)
    axes[1].set_xlim(-0.7, 1.7)
    axes[1].set_ylabel('cycle duration (s)')
    axes[1].set_title('Whole cycle', fontsize=9.5, color=R.INK, loc='left')
    fig.tight_layout()
    R.save(fig, out_dir, 'cmp_phase_durations.png')
    return pd.DataFrame(rows)


def fig_q_legs(runs, out_dir):
    fig, axes = R.new_fig(2, 3, figsize=(12.5, 6.8), sharey=True)
    series = [('q_center', 'Q region mean (controller objective)'), ('ws_mean_0', 'Q goal sphere'),
              ('qpath', 'Q path spheres (mean of the two)')]
    for r, phase in enumerate((R.PREGRASP, R.PREPLACE)):
        t_end, v_min = 0, 1.0
        for c, (key, label) in enumerate(series):
            ax = axes[r, c]
            for k, d in enumerate(runs):
                grid_t, arr = leg_curves(d, phase, key, R.T_GRID_MAX)
                if not arr.size:
                    continue
                # the median only: no per-leg lines, no inter-quartile band and no
                # number-at-risk strip. The solid/dashed split still says where the
                # median stops being representative, so nothing is hidden by the
                # simplification -- it is only quieter.
                _, te, _ = R.median_over_legs(ax, grid_t, arr, RUNS[k][1],
                                              label=f'{RUNS[k][0]} ({arr.shape[0]} legs)',
                                              band=False, n_at_risk=False)
                t_end = max(t_end, te)
                with warnings.catch_warnings():   # the grid outlives the longest leg
                    warnings.simplefilter('ignore', RuntimeWarning)
                    med = np.nanmedian(arr, axis=0)
                v_min = min(v_min, float(np.nanmin(med[~np.isnan(med)])))
        for c in range(len(series)):
            ax = axes[r, c]
            ax.set_xlim(0, t_end + 1)   # the axis ends with the longest leg, not with a quorum
            ax.set_ylim(max(0.0, v_min - 0.03), 1.02)
            ax.set_xlabel('time since first full-region cycle (s)')
            if c == 0:
                ax.set_ylabel(f'{R.LEG_LABEL[phase]}\nQ')
            if r == 0:
                ax.set_title(series[c][1], fontsize=9.5, color=R.INK, loc='left')
    axes[0, 0].legend(handles=R.legend_survival_handles(list(RUNS), R.INK, strip=False),
                      fontsize=7, frameon=False, loc='lower left')
    fig.suptitle('Controller-side scores over the approach legs: median across legs, solid while '
                 'half the legs are still running', fontsize=11, color=R.INK, x=0.01, ha='left')
    fig.tight_layout()
    R.save(fig, out_dir, 'cmp_q_legs.png')


def fig_q_legs_progress(runs, out_dir, classes=('success',)):
    """cmp_q_legs with the leg's own progress (%) on x instead of elapsed time: mean and
    ±1 std across the legs of the cycles whose outcome is in `classes` (default: successes only)."""
    fig, axes = R.new_fig(2, 3, figsize=(12.5, 6.8), sharey=True)
    series = [('q_center', 'Q region mean (controller objective)'), ('ws_mean_0', 'Q goal sphere'),
              ('qpath', 'Q path spheres (mean of the two)')]
    v_min = 1.0   # the y axis is shared by both rows
    for r, phase in enumerate((R.PREGRASP, R.PREPLACE)):
        n_legs = []
        for c, (key, _) in enumerate(series):
            ax = axes[r, c]
            for k, d in enumerate(runs):
                keep = d['idx'][d['idx'].cls.isin(classes)].episode_id
                grid_p, arr = leg_curves_progress(dict(d, legs=d['legs'][d['legs'].episode_id.isin(keep)]),
                                                phase, key)
                if c == 0:
                    n_legs.append(len(arr))
                if not arr.size:
                    continue
                mean, std = arr.mean(axis=0), arr.std(axis=0)
                ax.fill_between(grid_p, mean - std, mean + std, color=RUNS[k][1], alpha=0.18,
                                linewidth=0, zorder=2)
                ax.plot(grid_p, mean, '-', color=RUNS[k][1], linewidth=2.0, zorder=4)
                v_min = min(v_min, float((mean - std).min()))
        for c in range(len(series)):
            ax = axes[r, c]
            ax.set_xlim(0, 100)
            ax.set_xlabel('leg progress (% of the leg duration)')
            if c == 0:
                ax.set_ylabel(f'{R.LEG_LABEL[phase]}\nQ')
            if r == 0:
                ax.set_title(series[c][1], fontsize=9.5, color=R.INK, loc='left')
        axes[r, 0].legend(handles=[Line2D([], [], color=col, linewidth=2, label=f'{name} ({n} legs)')
                                   for (name, col), n in zip(RUNS, n_legs)]
                          + [Patch(color=R.INK, alpha=0.18, label='±1 std across legs')],
                          fontsize=7, frameon=False, loc='lower left')
    axes[0, 0].set_ylim(max(0.0, v_min - 0.03), 1.02)
    fig.suptitle(f'Controller-side scores over the approach legs against leg progress: mean and '
                 f'±1 std across legs ({", ".join(classes)} cycles only)',
                 fontsize=11, color=R.INK, x=0.01, ha='left')
    fig.tight_layout()
    R.save(fig, out_dir, 'cmp_q_legs_progress.png')


def fig_q_arrival(runs, out_dir):
    """Goal-sphere score at leg start, when the command stops, and when the arm arrives."""
    fig, axes = R.new_fig(1, 2, figsize=(11, 4.2), sharey=True)
    rng = np.random.default_rng(2)
    rows = []
    for ax, phase in zip(axes, (R.PREGRASP, R.PREPLACE)):
        for k, d in enumerate(runs):
            L = d['legs'][d['legs'].phase == phase]
            s = d['samples']
            valid = R.outcome_valid(d['idx']).episode_id
            s = s[(s.phase == phase) & s.episode_id.isin(valid) & (s.goal_age_s < 1)]
            end = s.groupby('episode_id').apply(
                lambda g: g[g.t_rel >= g.t_rel.max() - 3.0].q_goal.mean(), include_groups=False)
            for i, (vals, lab) in enumerate(((L.qgoal_start, 'start of leg'),
                                             (L.qgoal_end, 'end of command'),
                                             (end, 'end of approach (recorder, last 3 s)'))):
                med = strip(ax, i + (k - 0.5) * 0.36, vals, RUNS[k][1], width=0.16, rng=rng)
                rows.append(dict(run=d['name'], leg=R.LEG_LABEL[phase], when=lab,
                                 n=int(np.sum(~np.isnan(np.asarray(vals, float)))), median=med))
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Q goal at leg start\n(controller)', 'Q goal at end of command\n(controller)',
                            'Q goal at end of approach\n(recorder, last 3 s)'], fontsize=8)
        ax.set_ylim(0.25, 1.0)
        ax.set_title(R.LEG_LABEL[phase], fontsize=9.5, color=R.INK, loc='left')
    axes[0].set_ylabel('Q goal sphere')
    axes[0].legend(handles=[Line2D([], [], marker='o', linestyle='none', color=c, label=n) for n, c in RUNS],
                   fontsize=7.5, frameon=False, loc='lower left')
    fig.suptitle('Goal-sphere reachability at the start of the approach, when the command stops, and when the arm arrives',
                 fontsize=11, color=R.INK, x=0.01, ha='left')
    fig.tight_layout()
    R.save(fig, out_dir, 'cmp_q_arrival.png')
    return pd.DataFrame(rows)


def fig_geometry(runs, out_dir):
    """Where the target sits in the arm frame at the start of the leg and when the arm arrives."""
    fig, axes = R.new_fig(1, 2, figsize=(10, 6.2))
    rows = []
    for ax, phase in zip(axes, (R.PREGRASP, R.PREPLACE)):
        xs, ys = [0.0], [0.0]
        for k, d in reversed(list(enumerate(runs))):   # fixed run drawn last, on top
            L = d['legs'][d['legs'].phase == phase]
            col = RUNS[k][1]
            ax.plot(L.goal0_x, L.goal0_y, 'o', color=col, markersize=4, alpha=0.35,
                    markeredgecolor=R.SURFACE, markeredgewidth=0.5, zorder=3 + k)
            ax.plot(L.goal1_x, L.goal1_y, 's', color=col, markersize=5, alpha=0.85,
                    markeredgecolor=R.SURFACE, markeredgewidth=0.5, zorder=3 + k)
            x0, y0, x1, y1 = L.goal0_x.median(), L.goal0_y.median(), L.goal1_x.median(), L.goal1_y.median()
            xs += [x0, x1]
            ys += [y0, y1]
            if np.hypot(x1 - x0, y1 - y0) > 0.02:
                ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=6,
                            arrowprops=dict(arrowstyle='-|>', color=R.INK, linewidth=1.6,
                                            shrinkA=0, shrinkB=0))
            ax.text(x1 + 0.05, y1, f'{np.hypot(x1, y1):.2f} m', fontsize=8, color=col,
                    ha='left', va='center', fontweight='bold', zorder=7,
                    bbox=dict(facecolor=R.SURFACE, edgecolor='none', pad=1.5))
            rows.append(dict(run=d['name'], leg=R.LEG_LABEL[phase], n=len(L),
                             goal0_x=x0, goal0_y=y0, goal1_x=x1, goal1_y=y1,
                             range0_m=np.hypot(x0, y0), range1_m=np.hypot(x1, y1),
                             bearing0_deg=np.degrees(np.arctan2(y0, x0)),
                             bearing1_deg=np.degrees(np.arctan2(y1, x1))))
        ax.plot(0, 0, '^', color=R.INK, markersize=9)
        ax.text(0.02, -0.03, 'arm base', fontsize=7.5, color=R.INK2)
        for r_m in (0.6, 0.8, 1.0):
            th = np.linspace(-np.pi / 2, np.pi / 2, 100)
            ax.plot(r_m * np.cos(th), r_m * np.sin(th), '-', color=R.GRID, linewidth=0.8, zorder=0)
            ax.text(r_m * np.cos(np.radians(75)), r_m * np.sin(np.radians(75)), f'{r_m:.1f} m',
                    fontsize=6.5, color=R.MUTED, ha='center')
        ax.set_aspect('equal')
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.62, 1.12)
        ax.set_xlabel('goal x in the arm frame (m)')
        ax.set_ylabel('goal y in the arm frame (m)')
        ax.set_title(R.LEG_LABEL[phase], fontsize=9.5, color=R.INK, loc='left')
    axes[1].legend(handles=[Line2D([], [], marker='o', linestyle='none', color=c, alpha=0.5,
                                   label=f'{n}: leg start') for n, c in RUNS]
                   + [Line2D([], [], marker='s', linestyle='none', color=c, label=f'{n}: arm arrives')
                      for n, c in RUNS],
                   fontsize=7, frameon=False, loc='lower right')
    fig.suptitle('Where the target sits relative to the arm, before and after the approach',
                 fontsize=11, color=R.INK, x=0.01, ha='left')
    fig.tight_layout()
    R.save(fig, out_dir, 'cmp_geometry.png')
    return pd.DataFrame(rows)


def fig_direction(runs, out_dir):
    """Wanted direction (circular mean of ∇Q over legs) against time in the leg."""
    fig, axes = R.new_fig(1, 2, figsize=(11, 3.9))
    window = {R.PREGRASP: (-90, 270), R.PREPLACE: (0, 360)}
    rows = []
    for ax, phase in zip(axes, (R.PREGRASP, R.PREPLACE)):
        lo, hi = window[phase]
        t_max = LEG_T_MAX[phase]
        for k, d in enumerate(runs):
            bin_s = DIR_BIN_S[phase]
            bins, mean_deg, conc, n = heading_over_leg(d, phase, bin_s=bin_s, t_max=t_max + bin_s)
            ok = (n >= 5) & (conc >= 0.3)
            ax.plot(bins[ok] + bin_s / 2, (mean_deg[ok] - lo) % 360 + lo, '-o', color=RUNS[k][1],
                    linewidth=2, markersize=4, label=RUNS[k][0])
            for b, m, c_, ok_ in zip(bins, mean_deg, conc, ok):
                if ok_:
                    rows.append(dict(run=d['name'], leg=R.LEG_LABEL[phase], t_bin_start_s=b,
                                     heading_deg=m, concentration_R=c_))
        labels = {0: '+x forward', 90: '+y left', 180: '−x back', 270: '−y right',
                  -90: '−y right', -180: '−x back', -270: '+y left', 360: '+x forward'}
        ticks = list(range(lo, hi + 1, 90))
        for y in ticks:
            ax.axhline(y, color=R.GRID, linewidth=0.8, zorder=0)
            ax.text(t_max * 1.01, y, labels[y], fontsize=7, color=R.MUTED, va='center')
        ax.set_xlim(0, t_max * 1.22)
        ax.set_ylim(lo - 20, hi + 20)
        ax.set_yticks(ticks)
        ax.set_xlabel('time since first full-region cycle (s)')
        ax.set_title(f'{R.LEG_LABEL[phase]}, {DIR_BIN_S[phase]:.0f} s bins',
                     fontsize=9.5, color=R.INK, loc='left')
    axes[0].set_ylabel('heading of ∇Q in the arm frame (deg)')
    axes[0].legend(fontsize=7.5, frameon=False, loc='lower left')
    fig.suptitle('Direction the controller wants the base to go: circular mean of ∇Q over the legs',
                 fontsize=11, color=R.INK, x=0.01, ha='left')
    axes[1].text(0.02, 0.03, 'bins where legs disagree (concentration R < 0.3) are not drawn',
                 transform=axes[1].transAxes, fontsize=7, color=R.MUTED, ha='left')
    fig.tight_layout()
    R.save(fig, out_dir, 'cmp_direction.png')
    return pd.DataFrame(rows)


def fig_mpc(runs, out_dir):
    fig, axes = R.new_fig(2, 3, figsize=(12, 7))
    rng = np.random.default_rng(3)
    rows = []
    metrics = [('dur_s', 'leg duration (s)', '{:.0f}'),
               ('scene_col_frac', 'MPC steps with active scene-collision constraint', '{:.3f}'),
               ('fk_err_last', 'FK error to target at the end of the leg (m)', '{:.3f}')]
    for r, phase in enumerate((R.PREGRASP, R.PREPLACE)):
        for c, (key, label, fmt) in enumerate(metrics):
            ax = axes[r, c]
            for k, d in enumerate(runs):
                per = mpc_per_leg(d, phase)
                for i, cls in enumerate(CMP_CLASSES):
                    x = i + (k - 0.5) * 0.36
                    v = per[per.cls == cls][key]
                    med = strip(ax, x, v, RUNS[k][1], width=0.16, rng=rng, fmt=fmt)
                    ax.text(x, 0.01, f'n={int(v.notna().sum())}', transform=ax.get_xaxis_transform(),
                            ha='center', va='bottom', fontsize=6.5, color=R.MUTED)
                    rows.append(dict(run=d['name'], leg=R.LEG_LABEL[phase], outcome=cls, metric=key,
                                     n=int(v.notna().sum()), median=med))
            ax.set_xticks(range(len(CMP_CLASSES)))
            ax.set_xticklabels([f'{c_} cycles' for c_ in CMP_CLASSES], fontsize=8)
            ax.set_xlim(-0.6, len(CMP_CLASSES) - 0.4)
            ax.set_ylabel(label, fontsize=8)
            if key == 'fk_err_last':
                ax.set_yscale('log')
            if c == 0:
                ax.text(-0.02, 1.02, R.LEG_LABEL[phase], transform=ax.transAxes, fontsize=9.5,
                        color=R.INK, fontweight='bold', va='bottom')
    axes[0, 0].legend(handles=[Line2D([], [], marker='o', linestyle='none', color=c, label=n) for n, c in RUNS],
                      fontsize=7.5, frameon=False, loc='upper left')
    fig.suptitle('Arm MPC over the approach legs: duration, obstacle-constraint activity, arrival error',
                 fontsize=11, color=R.INK, x=0.01, ha='left')
    fig.tight_layout()
    R.save(fig, out_dir, 'cmp_mpc.png')
    return pd.DataFrame(rows)


def fig_timing(runs, out_dir):
    fig, axes = R.new_fig(1, 3, figsize=(12, 3.8))
    rows = []
    for k, d in enumerate(runs):
        col = RUNS[k][1]
        c = d['ctrl'][d['ctrl'].active].cycle_s.dropna() * 1000
        axes[0].hist(c, bins=np.arange(400, 1000, 20), color=col, alpha=0.55, edgecolor=R.SURFACE,
                     linewidth=0.4, label=f'{RUNS[k][0]}: mean {c.mean():.0f} ms')
        m = d['mpc'].solve_ms.dropna()
        over = d['mpc'].budget_exceeded.astype(bool).mean()
        axes[1].hist(m, bins=np.arange(0, 660, 15), color=col, alpha=0.55, edgecolor=R.SURFACE, linewidth=0.4,
                     density=True, label=f'{RUNS[k][0]}: mean {m.mean():.0f} ms, {over * 100:.1f}% > 400 ms')
        p = d['pairs']
        axes[2].hist(p.meas, bins=np.arange(-0.3, 0.31, 0.01), color=col, alpha=0.55, edgecolor=R.SURFACE,
                     linewidth=0.4, density=True, label=f'{RUNS[k][0]}: std {p.meas.std():.3f}')
        rows.append(dict(run=d['name'], ctrl_cycle_ms_mean=c.mean(), ctrl_cycle_ms_max=c.max(),
                         mpc_solve_ms_mean=m.mean(), mpc_solve_ms_p95=m.quantile(0.95),
                         mpc_over_budget_frac=over, dq_cycle_std=p.meas.std(),
                         direction_turn_median_deg=p.dang.median(),
                         direction_reversal_frac=(p.dang > 90).mean()))
    axes[0].axvline(1000, color=R.INK, linewidth=1, linestyle=':')
    axes[0].set_xlabel('gradient cycle duration (ms)')
    axes[0].set_ylabel('cycles')
    axes[0].set_title('Base controller cycle time', fontsize=9.5, color=R.INK, loc='left')
    axes[1].axvline(400, color=R.INK, linewidth=1, linestyle=':')
    axes[1].set_xlabel('MPC solve time (ms)')
    axes[1].set_ylabel('density')
    axes[1].set_title('Arm MPC solve time', fontsize=9.5, color=R.INK, loc='left')
    axes[2].set_xlabel('ΔQ between consecutive commanded cycles')
    axes[2].set_ylabel('density')
    axes[2].set_title('Cycle-to-cycle change of the objective', fontsize=9.5, color=R.INK, loc='left')
    for ax in axes:
        ax.set_ylim(top=ax.get_ylim()[1] * 1.28)
        ax.legend(fontsize=7, frameon=False, loc='upper right')
    fig.tight_layout()
    R.save(fig, out_dir, 'cmp_timing.png')
    return pd.DataFrame(rows)


def fig_manip(runs, out_dir):
    fig, ax = R.new_fig(figsize=(9, 3.9))
    rng = np.random.default_rng(4)
    rows = []
    for k, d in enumerate(runs):
        valid = R.outcome_valid(d['idx']).episode_id
        p = d['phases'][d['phases'].episode_id.isin(valid)]
        for i, (ph, _) in enumerate(PHASES):
            v = p[p.phase == ph].manipulability_mean
            med = strip(ax, i + (k - 0.5) * 0.36, v, RUNS[k][1], width=0.16, rng=rng, fmt='{:.3f}')
            rows.append(dict(run=d['name'], phase=ph, n=int(v.notna().sum()), median=med))
    ax.set_xticks(range(len(PHASES)))
    ax.set_xticklabels([n for _, n in PHASES], fontsize=8)
    ax.set_ylabel('Yoshikawa manipulability (phase mean)')
    ax.legend(handles=[Line2D([], [], marker='o', linestyle='none', color=c, label=n) for n, c in RUNS],
              fontsize=7.5, frameon=False, loc='upper left')
    ax.set_title('Arm manipulability per phase, valid episodes', fontsize=9.5, color=R.INK, loc='left')
    fig.tight_layout()
    R.save(fig, out_dir, 'cmp_manip.png')
    return pd.DataFrame(rows)


# ── Comparison table ────────────────────────────────────────────────────────

def write_tables(runs, out_dir, extra):
    rows = []
    for d in runs:
        idx, legs, pairs, mpc = d['idx'], d['legs'], d['pairs'], d['mpc']
        valid = R.outcome_valid(idx)
        succ = idx[idx.cls == 'success']
        row = {'run': d['name'], 'episodes recorded': len(idx),
               'excluded: infrastructure/operator': int((idx.cls == 'excluded').sum()),
               'excluded: grasp/release failure': int(idx.cls.isin(R.GRASP_CLASSES).sum()),
               'valid': len(valid),
               'success': int((idx.cls == 'success').sum()),
               'success rate': (idx.cls == 'success').sum() / len(valid),
               'collision / blocked': int((idx.cls == 'collision').sum()),
               'cycle duration, success, median (s)': succ.duration_s.median(),
               'cycle duration, success, mean (s)': succ.duration_s.mean()}
        ps = d['phases'][d['phases'].episode_id.isin(succ.episode_id)]
        for ph, lab in PHASES:
            row[f'{lab.replace(chr(10), " ")} median (s)'] = \
                ps[ps.phase == ph].groupby('episode_id').duration_s.sum().median()
        for phase, tag in ((R.PREGRASP, 'pick'), (R.PREPLACE, 'place')):
            L = legs[legs.phase == phase]
            per = mpc_per_leg(d, phase)
            ok = per[per.cls == 'success']
            row.update({
                f'{tag}: legs': len(L),
                f'{tag}: commanded cycles, median': L.n_moving.median(),
                f'{tag}: commanded time, median (s)': L.t_moving_s.median(),
                f'{tag}: commanded path, median (m)': L.dist_m.median(),
                f'{tag}: net displacement, median (m)': L.net_m.median(),
                f'{tag}: goal range at leg start, median (m)': np.hypot(L.goal0_x, L.goal0_y).median(),
                f'{tag}: goal range when the arm arrives, median (m)': np.hypot(L.goal1_x, L.goal1_y).median(),
                f'{tag}: Q goal at leg start, median': L.qgoal_start.median(),
                f'{tag}: Q goal at end of command, median': L.qgoal_end.median(),
                f'{tag}: ΔQ goal over the command, median': L.dq_goal.median(),
                f'{tag}: ΔQ goal > 0 (fraction of legs)': (L.dq_goal > 0).mean(),
                f'{tag}: Q region mean at leg start, median': L.q_start.median(),
                f'{tag}: Q region mean at end of command, median': L.q_end.median(),
                f'{tag}: |∇Q| median (1/m)': L.grad_med.median(),
                f'{tag}: stop = arrival latch': int(L.stop.str.startswith('arrival').sum()),
                f'{tag}: stop = interlock': int(L.stop.str.startswith('interlock').sum()),
                f'{tag}: stop = convergence': int(L.stop.str.startswith('gradient').sum()),
                f'{tag}: MPC scene-collision active, success legs, median': ok.scene_col_frac.median(),
                f'{tag}: MPC FK error at end, success legs, median (m)': ok.fk_err_last.median(),
            })
        sub = legs[legs.dgoal_x.notna() & legs.dcmd_win_x.notna()]
        X = np.concatenate([sub.dcmd_win_x, sub.dcmd_win_y])
        Y = -np.concatenate([sub.dgoal_x, sub.dgoal_y])
        row.update({
            'executed vs commanded displacement: slope': np.sum(X * Y) / np.sum(X * X),
            'executed vs commanded displacement: r': np.corrcoef(X, Y)[0, 1],
            'controller cycle time, mean (ms)': d['ctrl'][d['ctrl'].active].cycle_s.mean() * 1000,
            'MPC solve time, mean (ms)': mpc.solve_ms.mean(),
            'MPC solve time, p95 (ms)': mpc.solve_ms.quantile(0.95),
            'MPC over budget (fraction of steps)': mpc.budget_exceeded.astype(bool).mean(),
            'ΔQ per cycle, std': pairs.meas.std(),
            'gradient direction turn per cycle, median (deg)': pairs.dang.median(),
            'gradient direction reversals (> 90°), fraction': (pairs.dang > 90).mean(),
        })
        rows.append(row)
    summary = pd.DataFrame(rows).set_index('run').T
    summary.to_csv(os.path.join(out_dir, 'cmp_summary.csv'))
    with open(os.path.join(out_dir, 'cmp_summary.md'), 'w') as f:
        f.write('| metric | ' + ' | '.join(summary.columns) + ' |\n|---|---|---|\n')
        for k, r in summary.iterrows():
            cells = []
            for v in r:
                if isinstance(v, (float, np.floating)):
                    cells.append('' if np.isnan(v) else (f'{v:.3f}' if abs(v) < 10 else f'{v:.1f}'))
                else:
                    cells.append(str(v))
            f.write(f'| {k} | ' + ' | '.join(cells) + ' |\n')
    for name, df in extra.items():
        df.round(4).to_csv(os.path.join(out_dir, f'cmp_{name}.csv'), index=False)
    pd.concat([d['legs'].assign(run=d['name']) for d in runs], ignore_index=True) \
        .round(4).to_csv(os.path.join(out_dir, 'cmp_legs.csv'), index=False)
    print('  wrote exp_2_comparison_tables/cmp_summary.csv|md, cmp_*.csv')
    return summary


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fixed-dir', default=DEFAULT_FIXED)
    ap.add_argument('--moving-dir', default=DEFAULT_MOVING)
    ap.add_argument('--out', default=None, help='output root for the comparison (default: moving-base run dir)')
    ap.add_argument('--skip-per-run', action='store_true', help='only rebuild the comparison')
    args = ap.parse_args()
    out = args.out or args.moving_dir

    if not args.skip_per_run:
        # Per-run figures and tables, each inside its own run directory.
        R.run(args.fixed_dir, args.fixed_dir, loader=load_fixed, base_moves=False,
              fig_dirname='exp_2_fixed_figures', tab_dirname='exp_2_fixed_tables',
              diag_classes=CMP_CLASSES,
              diag_title='Place-approach diagnostics by outcome (setup 2, base fixed)',
              example_eps=(2, 0), timeline_t_max=420, leg_t_max=LEG_T_MAX[R.PREPLACE])
        R.run(args.moving_dir, args.moving_dir, loader=load_moving, base_moves=True,
              fig_dirname='exp_2_moving_figures', tab_dirname='exp_2_moving_tables',
              diag_classes=CMP_CLASSES,
              diag_title='Place-approach diagnostics by outcome (setup 2, base moving)',
              example_eps=(9, 5), timeline_t_max=300, leg_t_max=80)

    runs = load_both(args.fixed_dir, args.moving_dir)
    for d in runs:
        print(d['name'], d['idx'].cls.value_counts().to_dict())
    fig_dir = os.path.join(out, 'exp_2_comparison_figures')
    tab_dir = os.path.join(out, 'exp_2_comparison_tables')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    print('comparison figures ->', fig_dir)
    extra = {'outcomes': fig_outcomes(runs, fig_dir),
             'phase_durations': fig_phase_durations(runs, fig_dir),
             'q_arrival': fig_q_arrival(runs, fig_dir),
             'geometry': fig_geometry(runs, fig_dir),
             'direction': fig_direction(runs, fig_dir),
             'mpc': fig_mpc(runs, fig_dir),
             'timing': fig_timing(runs, fig_dir),
             'manip': fig_manip(runs, fig_dir)}
    fig_q_legs(runs, fig_dir)
    fig_q_legs_progress(runs, fig_dir)
    print('comparison tables ->', tab_dir)
    summary = write_tables(runs, tab_dir, extra)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 200)
    print(summary.round(3).to_string())


if __name__ == '__main__':
    main()
