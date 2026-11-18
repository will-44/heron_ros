#!/usr/bin/env python3
"""Experiment 1 (base moving) vs experiment 2 (base fixed): side-by-side figures and tables.

    python3 analysis/setup_1/compare_runs.py                 # runs.SETUP_1_MOVING vs runs.SETUP_1_FIXED
    python3 analysis/setup_1/compare_runs.py --exp1 setup_1/<run_id> --exp2 setup_1/<run_id> --out DIR

Writes figures/cmp_*.png and tables/cmp_*.csv|md into --out (default:
results/setup_1/analysis/comparison). Both runs are loaded with their own classification
(`analyze_exp1.load` / `analyze_exp2.load`); the per-leg and per-cycle tables
are built by the shared code in analyze_exp1.py.

Colour convention in every comparison figure: blue = experiment 1 (base moving),
orange = experiment 2 (base fixed). Legs are separated by panel, never by colour.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import analyze_exp2 as B  # noqa: E402
import analyze_exp1 as A  # noqa: E402

RUNS = [('base moving (exp 1)', A.BLUE), ('base fixed (exp 2)', A.ORANGE)]
PHASES = [('act_home_enter', 'home\nenter'), (A.PREGRASP, 'pick\napproach'),
          ('act_grasp:act_move_grasp', 'grasp\nmove'), (A.PREPLACE, 'place\napproach'),
          ('act_release:act_move_place', 'place\nmove'), ('act_home_exit', 'home\nexit')]
LEG_T_MAX = {A.PREGRASP: 40, A.PREPLACE: 70}


def load_both(exp1_dir, exp2_dir):
    out = []
    for name, d in zip([r[0] for r in RUNS], (A.load(exp1_dir), B.load(exp2_dir))):
        cv = A.integrate_cmd_vel(d['cmd_vel'])
        legs = A.build_legs(d, cv)
        legs = A.outcome_valid(legs).copy()
        pairs = A.build_cycle_pairs(d['ctrl'][d['ctrl'].episode_id.isin(A.outcome_valid(d['idx']).episode_id)])
        d.update(name=name, cv=cv, legs=legs, pairs=pairs)
        out.append(d)
    return out


def strip(ax, x, values, color, width=0.32, rng=None, label_median=True, fmt='{:.2f}'):
    rng = rng or np.random.default_rng(0)
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return np.nan
    ax.plot(x + rng.uniform(-width * 0.6, width * 0.6, len(v)), v, 'o', color=color, alpha=0.7,
            markersize=4.5, markeredgecolor=A.SURFACE, markeredgewidth=0.5)
    med = np.median(v)
    ax.plot([x - width, x + width], [med, med], '-', color=A.INK, linewidth=2)
    if label_median:
        ax.text(x, med, ' ' + fmt.format(med), fontsize=7, va='bottom', ha='left', color=A.INK,
                transform=ax.transData)
    return med


# ── Figures ─────────────────────────────────────────────────────────────────

def fig_outcomes(runs, out_dir):
    fig, ax = A.new_fig(figsize=(10, 3.2))
    ax.grid(False)
    order = ['success', 'collision']
    rows = []
    for y, d in enumerate(runs):
        idx = d['idx']
        valid = A.outcome_valid(idx)
        left = 0
        for c in order:
            n = int((valid.cls == c).sum())
            if n == 0:
                continue
            frac = n / len(valid)
            ax.barh(y, frac, left=left, height=0.55, color=A.CLASS_COLOR[c], linewidth=0)
            ax.text(left + frac / 2, y, f'{n}\n{frac * 100:.0f}%', ha='center', va='center', fontsize=8,
                    color=A.SURFACE, fontweight='bold')
            left += frac
            rows.append(dict(run=d['name'], cls=c, n=n, frac=frac))
        ax.text(1.01, y, f'n = {len(valid)} valid\n(+{int((idx.cls == "excluded").sum())} excluded,'
                f' +{int(idx.cls.isin(A.GRASP_CLASSES).sum())} grasp/release failures)',
                va='center', fontsize=7.5, color=A.INK2)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([d['name'] for d in runs], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['0', '25 %', '50 %', '75 %', '100 %'])
    ax.legend(handles=[Patch(color=A.CLASS_COLOR[c], label=c) for c in order],
              ncol=4, fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax.set_title('Outcome of the valid pick-and-place cycles (grasp/release failures excluded)', fontsize=10, color=A.INK, loc='left')
    A.save(fig, out_dir, 'cmp_outcomes.png')
    return pd.DataFrame(rows)


def fig_phase_durations(runs, out_dir):
    fig, axes = A.new_fig(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[4, 1]))
    rng = np.random.default_rng(1)
    rows = []
    ax = axes[0]
    for k, d in enumerate(runs):
        idx, ps = d['idx'], d['phases']
        succ = idx[idx.cls == 'success'].episode_id
        p = ps[ps.episode_id.isin(succ)]
        for i, (ph, _) in enumerate(PHASES):
            v = p[p.phase == ph].groupby('episode_id').duration_s.sum()
            med = strip(ax, i + (k - 0.5) * 0.36, v, RUNS[k][1], width=0.16, rng=rng, fmt='{:.0f}')
            rows.append(dict(run=d['name'], phase=ph, n=len(v), median_s=med, mean_s=v.mean(),
                             q25_s=v.quantile(0.25), q75_s=v.quantile(0.75)))
    ax.set_xticks(range(len(PHASES)))
    ax.set_xticklabels([n for _, n in PHASES], fontsize=8)
    ax.set_ylabel('phase duration (s), successful cycles')
    ax.set_title('Phase durations of successful cycles (one dot per cycle, bar = median)',
                 fontsize=9.5, color=A.INK, loc='left')
    ax.legend(handles=[Line2D([], [], marker='o', linestyle='none', color=c, label=n) for n, c in RUNS],
              fontsize=7.5, frameon=False, loc='upper left')
    ax = axes[1]
    for k, d in enumerate(runs):
        idx = d['idx']
        v = idx[idx.cls == 'success'].duration_s
        med = strip(ax, k, v, RUNS[k][1], width=0.3, rng=rng, fmt='{:.0f}')
        rows.append(dict(run=d['name'], phase='whole cycle', n=len(v), median_s=med, mean_s=v.mean(),
                         q25_s=v.quantile(0.25), q75_s=v.quantile(0.75)))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['base\nmoving', 'base\nfixed'], fontsize=8)
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylabel('cycle duration (s)')
    ax.set_title('Whole cycle', fontsize=9.5, color=A.INK, loc='left')
    fig.tight_layout()
    A.save(fig, out_dir, 'cmp_phase_durations.png')
    return pd.DataFrame(rows)


def _leg_curves(d, phase, key, t_max):
    ctrl, legs = d['ctrl'], d['legs']
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


def fig_q_legs(runs, out_dir):
    fig, axes = A.new_fig(2, 3, figsize=(12.5, 6.8), sharey=True)
    series = [('q_center', 'Q region mean (controller objective)'), ('ws_mean_0', 'Q goal sphere'),
              ('qpath', 'Q path spheres (mean of the two)')]
    for r, phase in enumerate((A.PREGRASP, A.PREPLACE)):
        t_end, v_min = 0, 1.0
        for c, (key, label) in enumerate(series):
            ax = axes[r, c]
            for k, d in enumerate(runs):
                grid_t, arr = _leg_curves(d, phase, key, A.T_GRID_MAX)
                if not arr.size:
                    continue
                # every leg, thin, drawn to its own end, underneath the summary
                for row in arr:
                    alive = ~np.isnan(row)
                    ax.plot(grid_t[alive], row[alive], '-', color=RUNS[k][1], alpha=0.16,
                            linewidth=0.8, zorder=1)
                _, te, _ = A.median_over_legs(ax, grid_t, arr, RUNS[k][1],
                                                  label=f'{RUNS[k][0]} ({arr.shape[0]} legs)')
                t_end = max(t_end, te)
                v_min = min(v_min, float(np.nanmin(arr)))
        for c in range(len(series)):
            ax = axes[r, c]
            ax.set_xlim(0, t_end + 1)   # the axis ends with the longest leg, not with a quorum
            ax.set_ylim(max(0.0, v_min - 0.03), 1.02)   # no leg clipped off the bottom
            ax.set_xlabel('time since first full-region cycle (s)')
            if c == 0:
                ax.set_ylabel(f'{A.LEG_LABEL[phase]}\nQ')
            if r == 0:
                ax.set_title(series[c][1], fontsize=9.5, color=A.INK, loc='left')
    axes[0, 0].legend(handles=A.legend_survival_handles(list(RUNS), A.INK),
                      fontsize=7, frameon=False, loc='lower left')
    fig.suptitle('Controller-side scores over the approach legs: every leg (thin), median and '
                 'inter-quartile band across legs', fontsize=11, color=A.INK, x=0.01, ha='left')
    fig.tight_layout()
    A.save(fig, out_dir, 'cmp_q_legs.png')


def _leg_curves_progress(d, phase, key, n_pts=101):
    """Per-leg series resampled on a 0-100 % grid of the leg's own duration (first to last
    active full-region cycle), so every leg spans the whole axis."""
    ctrl, legs = d['ctrl'], d['legs']
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


def fig_q_legs_progress(runs, out_dir, classes=('success',)):
    """cmp_q_legs with the leg's own progress (%) on x instead of elapsed time: mean and
    ±1 std across the legs of the cycles whose outcome is in `classes` (default: successes only)."""
    fig, axes = A.new_fig(2, 3, figsize=(12.5, 6.8), sharey=True)
    series = [('q_center', 'Q region mean (controller objective)'), ('ws_mean_0', 'Q goal sphere'),
              ('qpath', 'Q path spheres (mean of the two)')]
    v_min = 1.0   # the y axis is shared by both rows
    for r, phase in enumerate((A.PREGRASP, A.PREPLACE)):
        n_legs = []
        for c, (key, _) in enumerate(series):
            ax = axes[r, c]
            for k, d in enumerate(runs):
                keep = d['idx'][d['idx'].cls.isin(classes)].episode_id
                grid_p, arr = _leg_curves_progress(dict(d, legs=d['legs'][d['legs'].episode_id.isin(keep)]),
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
                ax.set_ylabel(f'{A.LEG_LABEL[phase]}\nQ')
            if r == 0:
                ax.set_title(series[c][1], fontsize=9.5, color=A.INK, loc='left')
        axes[r, 0].legend(handles=[Line2D([], [], color=col, linewidth=2, label=f'{name} ({n} legs)')
                                   for (name, col), n in zip(RUNS, n_legs)]
                          + [Patch(color=A.INK, alpha=0.18, label='±1 std across legs')],
                          fontsize=7, frameon=False, loc='lower left')
    axes[0, 0].set_ylim(max(0.0, v_min - 0.03), 1.02)
    fig.suptitle(f'Controller-side scores over the approach legs against leg progress: mean and '
                 f'±1 std across legs ({", ".join(classes)} cycles only)',
                 fontsize=11, color=A.INK, x=0.01, ha='left')
    fig.tight_layout()
    A.save(fig, out_dir, 'cmp_q_legs_progress.png')


def fig_q_arrival(runs, out_dir):
    """Score when the arm arrives: last commanded cycle of the leg (controller) and the
    recorder's mean over the grasp/place move that follows."""
    fig, axes = A.new_fig(1, 2, figsize=(11, 4.2), sharey=True)
    rng = np.random.default_rng(2)
    rows = []
    for ax, (phase, move) in zip(axes, ((A.PREGRASP, 'act_grasp:act_move_grasp'),
                                        (A.PREPLACE, 'act_release:act_move_place'))):
        for k, d in enumerate(runs):
            L = d['legs'][d['legs'].phase == phase]
            s = d['samples']
            valid = A.outcome_valid(d['idx']).episode_id
            # Recorder score over the last 3 s of the approach: the arm is at the pre-grasp /
            # pre-place pose and the base has been still for ~20 s in both runs.
            s = s[(s.phase == phase) & s.episode_id.isin(valid) & (s.goal_age_s < 1)]
            end = s.groupby('episode_id').apply(
                lambda g: g[g.t_rel >= g.t_rel.max() - 3.0].q_goal.mean(), include_groups=False)
            for i, (vals, lab) in enumerate(((L.qgoal_start, 'start of leg'), (L.qgoal_end, 'end of command'),
                                             (end, 'end of approach (recorder, last 3 s)'))):
                med = strip(ax, i + (k - 0.5) * 0.36, vals, RUNS[k][1], width=0.16, rng=rng)
                rows.append(dict(run=d['name'], leg=A.LEG_LABEL[phase], when=lab, n=int(vals.notna().sum()), median=med))
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Q goal at leg start\n(controller)', 'Q goal at end of command\n(controller)',
                            'Q goal at end of approach\n(recorder, last 3 s)'], fontsize=8)
        ax.set_ylim(0.3, 1.0)
        ax.set_title(A.LEG_LABEL[phase], fontsize=9.5, color=A.INK, loc='left')
    axes[0].set_ylabel('Q goal sphere')
    axes[0].legend(handles=[Line2D([], [], marker='o', linestyle='none', color=c, label=n) for n, c in RUNS],
                   fontsize=7.5, frameon=False, loc='lower left')
    fig.suptitle('Goal-sphere reachability at the start of the approach, when the command stops, and when the arm arrives',
                 fontsize=11, color=A.INK, x=0.01, ha='left')
    fig.tight_layout()
    A.save(fig, out_dir, 'cmp_q_arrival.png')
    return pd.DataFrame(rows)


def fig_direction(runs, out_dir):
    """Circular-mean wanted direction per 5 s bin, both runs, both legs."""
    fig, axes = A.new_fig(1, 2, figsize=(11, 3.9))
    rows = []
    # Angle window per leg, chosen so that neither run's curve crosses the wrap:
    # pick headings live around -150..-40 deg, place headings around 60..200 deg.
    window = {A.PREGRASP: (-270, 90), A.PREPLACE: (0, 360)}
    t_show = {A.PREGRASP: 30, A.PREPLACE: 65}
    for ax, phase in zip(axes, (A.PREGRASP, A.PREPLACE)):
        lo, hi = window[phase]
        for k, d in enumerate(runs):
            bins, H = B.heading_over_leg(d['ctrl'], d['idx'], phase, t_max=t_show[phase] + 5)
            mean_deg, R = B.circ_mean_deg(H)
            n = np.sum(~np.isnan(H), axis=0)
            ok = (n >= 5) & (R >= 0.3)    # skip bins where the legs disagree (near-uniform directions)
            shown = (mean_deg[ok] - lo) % 360 + lo
            ax.plot(bins[ok] + 2.5, shown, '-o', color=RUNS[k][1], linewidth=2, markersize=4, label=RUNS[k][0])
            for b, m, r, ok_ in zip(bins, mean_deg, R, ok):
                if ok_:
                    rows.append(dict(run=d['name'], leg=A.LEG_LABEL[phase], t_bin_start_s=b, heading_deg=m, concentration_R=r))
        labels = {0: '+x forward', 90: '+y left', 180: '−x back', 270: '−y right', -90: '−y right',
                  -180: '−x back', -270: '+y left', 360: '+x forward'}
        ticks = [y for y in range(lo, hi + 1, 90)]
        for y in ticks:
            ax.axhline(y, color=A.GRID, linewidth=0.8, zorder=0)
            ax.text(t_show[phase] + 0.5, y, labels[y], fontsize=7, color=A.MUTED, va='center')
        ax.set_xlim(0, t_show[phase] + 9)
        ax.set_ylim(lo - 20, hi + 20)
        ax.set_yticks(ticks)
        ax.set_xlabel('time since first full-region cycle (s)')
        ax.set_title(f'{A.LEG_LABEL[phase]}: wanted direction (circular mean over legs, 5 s bins)',
                     fontsize=9.5, color=A.INK, loc='left')
    axes[0].set_ylabel('heading of ∇Q in the arm frame (deg)')
    axes[0].legend(fontsize=7.5, frameon=False, loc='upper right')
    axes[1].text(0.99, 0.02, 'bins where legs disagree (concentration R < 0.3) are not drawn',
                 transform=axes[1].transAxes, fontsize=7, color=A.MUTED, ha='right')
    fig.tight_layout()
    A.save(fig, out_dir, 'cmp_direction.png')
    return pd.DataFrame(rows)


def _mpc_per_leg(d, phase):
    mpc, idx = d['mpc'], d['idx']
    m = mpc[(mpc.phase == phase) & mpc.episode_id.isin(A.outcome_valid(idx).episode_id)].copy()
    m['cls'] = m.episode_id.map(idx.set_index('episode_id').cls)
    per = m.groupby(['episode_id', 'cls']).agg(
        fk_err_last=('arrival_fk_err_m', lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan),
        scene_col_frac=('con_scene_collision', lambda s: (s > 0).mean()),
        self_col_frac=('con_self_collision', lambda s: (s > 0).mean()),
        solve_ms=('solve_ms', 'mean'), n=('solve_ms', 'size'),
        dur_s=('t_rel', lambda s: s.max() - s.min())).reset_index()
    return per


def fig_mpc(runs, out_dir):
    fig, axes = A.new_fig(2, 3, figsize=(12, 7))
    rng = np.random.default_rng(3)
    rows = []
    metrics = [('dur_s', 'leg duration (s)', '{:.0f}'),
               ('scene_col_frac', 'MPC steps with active scene-collision constraint', '{:.3f}'),
               ('fk_err_last', 'FK error to target at the end of the leg (m)', '{:.3f}')]
    for r, phase in enumerate((A.PREGRASP, A.PREPLACE)):
        for c, (key, label, fmt) in enumerate(metrics):
            ax = axes[r, c]
            for k, d in enumerate(runs):
                per = _mpc_per_leg(d, phase)
                for i, sel in enumerate(('success', 'other')):
                    s = per[per.cls == 'success'] if sel == 'success' else per[per.cls != 'success']
                    med = strip(ax, i + (k - 0.5) * 0.36, s[key], RUNS[k][1], width=0.16, rng=rng, fmt=fmt)
                    rows.append(dict(run=d['name'], leg=A.LEG_LABEL[phase], outcome=sel, metric=key,
                                     n=int(s[key].notna().sum()), median=med))
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['successful cycles', 'cycles ending in a collision'], fontsize=8)
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylabel(label, fontsize=8)
            if key == 'fk_err_last':
                ax.set_yscale('log')
            if c == 0:
                ax.text(-0.02, 1.02, A.LEG_LABEL[phase], transform=ax.transAxes, fontsize=9.5,
                        color=A.INK, fontweight='bold', va='bottom')
    axes[0, 0].legend(handles=[Line2D([], [], marker='o', linestyle='none', color=c, label=n) for n, c in RUNS],
                      fontsize=7.5, frameon=False, loc='upper left')
    fig.suptitle('Arm MPC over the approach legs: duration, obstacle-constraint activity, arrival error',
                 fontsize=11, color=A.INK, x=0.01, ha='left')
    fig.tight_layout()
    A.save(fig, out_dir, 'cmp_mpc.png')
    return pd.DataFrame(rows)


def fig_timing(runs, out_dir):
    fig, axes = A.new_fig(1, 3, figsize=(12, 3.8))
    rows = []
    for k, d in enumerate(runs):
        col = RUNS[k][1]
        c = d['ctrl'][d['ctrl'].active].cycle_s.dropna() * 1000
        axes[0].hist(c, bins=np.arange(400, 1000, 20), color=col, alpha=0.55, edgecolor=A.SURFACE, linewidth=0.4,
                     label=f'{RUNS[k][0]}: mean {c.mean():.0f} ms')
        m = d['mpc'].solve_ms.dropna()
        over = d['mpc'].budget_exceeded.astype(bool).mean()
        axes[1].hist(m, bins=np.arange(0, 660, 15), color=col, alpha=0.55, edgecolor=A.SURFACE, linewidth=0.4,
                     density=True, label=f'{RUNS[k][0]}: mean {m.mean():.0f} ms, {over * 100:.1f}% > 400 ms')
        p = d['pairs']
        axes[2].hist(p.meas, bins=np.arange(-0.3, 0.31, 0.01), color=col, alpha=0.55, edgecolor=A.SURFACE,
                     linewidth=0.4, density=True, label=f'{RUNS[k][0]}: std {p.meas.std():.3f}')
        rows.append(dict(run=d['name'], ctrl_cycle_ms_mean=c.mean(), ctrl_cycle_ms_max=c.max(),
                         mpc_solve_ms_mean=m.mean(), mpc_solve_ms_p95=m.quantile(0.95), mpc_over_budget_frac=over,
                         dq_cycle_std=p.meas.std(), dq_cycle_mad=(p.meas - p.meas.median()).abs().median(),
                         direction_turn_median_deg=p.dang.median(), direction_reversal_frac=(p.dang > 90).mean()))
    axes[0].axvline(1000, color=A.INK, linewidth=1, linestyle=':')
    axes[0].set_xlabel('gradient cycle duration (ms)')
    axes[0].set_ylabel('cycles')
    axes[0].set_title('Base controller cycle time', fontsize=9.5, color=A.INK, loc='left')
    axes[1].axvline(400, color=A.INK, linewidth=1, linestyle=':')
    axes[1].set_xlabel('MPC solve time (ms)')
    axes[1].set_ylabel('density')
    axes[1].set_title('Arm MPC solve time', fontsize=9.5, color=A.INK, loc='left')
    axes[2].set_xlabel('ΔQ between consecutive commanded cycles')
    axes[2].set_ylabel('density')
    axes[2].set_title('Cycle-to-cycle change of the objective', fontsize=9.5, color=A.INK, loc='left')
    for ax in axes:
        ax.legend(fontsize=7, frameon=False, loc='upper right')
    fig.tight_layout()
    A.save(fig, out_dir, 'cmp_timing.png')
    return pd.DataFrame(rows)


def fig_manip(runs, out_dir):
    fig, ax = A.new_fig(figsize=(9, 3.9))
    rng = np.random.default_rng(4)
    rows = []
    for k, d in enumerate(runs):
        idx, ps = d['idx'], d['phases']
        p = ps[ps.episode_id.isin(A.outcome_valid(idx).episode_id)]
        for i, (ph, _) in enumerate(PHASES):
            v = p[p.phase == ph].manipulability_mean
            med = strip(ax, i + (k - 0.5) * 0.36, v, RUNS[k][1], width=0.16, rng=rng, fmt='{:.3f}')
            rows.append(dict(run=d['name'], phase=ph, n=int(v.notna().sum()), median=med))
    ax.set_xticks(range(len(PHASES)))
    ax.set_xticklabels([n for _, n in PHASES], fontsize=8)
    ax.set_ylabel('Yoshikawa manipulability (phase mean)')
    ax.legend(handles=[Line2D([], [], marker='o', linestyle='none', color=c, label=n) for n, c in RUNS],
              fontsize=7.5, frameon=False, loc='upper left')
    ax.set_title('Arm manipulability per phase, valid episodes', fontsize=9.5, color=A.INK, loc='left')
    fig.tight_layout()
    A.save(fig, out_dir, 'cmp_manip.png')
    return pd.DataFrame(rows)


# ── Tables ──────────────────────────────────────────────────────────────────

def write_tables(runs, out_dir, extra):
    rows = []
    for d in runs:
        idx, legs, pairs, mpc = d['idx'], d['legs'], d['pairs'], d['mpc']
        valid = A.outcome_valid(idx)
        succ = idx[idx.cls == 'success']
        row = {
            'run': d['name'],
            'episodes recorded': len(idx),
            'excluded: infrastructure/operator': int((idx.cls == 'excluded').sum()),
            'excluded: grasp/release failure': int(idx.cls.isin(A.GRASP_CLASSES).sum()),
            'valid': len(valid),
            'success': int((idx.cls == 'success').sum()),
            'success rate': (idx.cls == 'success').sum() / len(valid),
            'bad-grasp cancel': int((idx.cls == 'bad-grasp cancel').sum()),
            'grasp/release failed': int((idx.cls == 'grasp/release failed').sum()),
            'collision': int((idx.cls == 'collision').sum()),
            'cycle duration, success, median (s)': succ.duration_s.median(),
            'cycle duration, success, mean (s)': succ.duration_s.mean(),
        }
        ps = d['phases'][d['phases'].episode_id.isin(succ.episode_id)]
        for ph, lab in PHASES:
            row[f'{lab.replace(chr(10), " ")} median (s)'] = ps[ps.phase == ph].groupby('episode_id').duration_s.sum().median()
        for phase, tag in ((A.PREGRASP, 'pick'), (A.PREPLACE, 'place')):
            L = legs[legs.phase == phase]
            row.update({
                f'{tag}: legs': len(L),
                f'{tag}: commanded cycles, median': L.n_moving.median(),
                f'{tag}: commanded time, median (s)': L.t_moving_s.median(),
                f'{tag}: commanded path, median (m)': L.dist_m.median(),
                f'{tag}: Q goal at leg start, median': L.qgoal_start.median(),
                f'{tag}: Q goal at end of command, median': L.qgoal_end.median(),
                f'{tag}: ΔQ goal over the command, median': L.dq_goal.median(),
                f'{tag}: ΔQ goal > 0 (fraction of legs)': (L.dq_goal > 0).mean(),
                f'{tag}: Q region mean at leg start, median': L.q_start.median(),
                f'{tag}: Q region mean at end of command, median': L.q_end.median(),
                f'{tag}: |∇Q| median (1/m)': L.grad_med.median(),
                f'{tag}: stop = arrival latch': int((L.stop.str.startswith('arrival')).sum()),
                f'{tag}: stop = interlock': int((L.stop.str.startswith('interlock')).sum()),
                f'{tag}: stop = convergence': int((L.stop.str.startswith('gradient')).sum()),
            })
            per = _mpc_per_leg(d, phase)
            ok = per[per.cls == 'success']
            row[f'{tag}: MPC scene-collision active, success legs, median'] = ok.scene_col_frac.median()
            row[f'{tag}: MPC FK error at end, success legs, median (m)'] = ok.fk_err_last.median()
        row.update({
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
    legs_all = pd.concat([d['legs'].assign(run=d['name']) for d in runs], ignore_index=True)
    legs_all.round(4).to_csv(os.path.join(out_dir, 'cmp_legs.csv'), index=False)
    print('  wrote tables/cmp_summary.csv|md, cmp_*.csv')
    return summary


def main():
    sys.path.insert(0, os.path.dirname(HERE))
    from runs import SETUP_1_FIXED, SETUP_1_MOVING, raw, study_dir

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp1', default=SETUP_1_MOVING, help='base-moving run (default: %(default)s)')
    ap.add_argument('--exp2', default=SETUP_1_FIXED, help='fixed-base run (default: %(default)s)')
    ap.add_argument('--out', default=None, help='output root (default: results/setup_1/analysis/comparison)')
    args = ap.parse_args()
    out = args.out or study_dir('comparison', 'setup_1')
    fig_dir = os.path.join(out, 'figures')
    tab_dir = os.path.join(out, 'tables')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    runs = load_both(raw(args.exp1), raw(args.exp2))
    for d in runs:
        print(d['name'], d['idx'].cls.value_counts().to_dict())
    print('figures ->', fig_dir)
    extra = {}
    extra['outcomes'] = fig_outcomes(runs, fig_dir)
    extra['phase_durations'] = fig_phase_durations(runs, fig_dir)
    fig_q_legs(runs, fig_dir)
    fig_q_legs_progress(runs, fig_dir)
    extra['q_arrival'] = fig_q_arrival(runs, fig_dir)
    extra['direction'] = fig_direction(runs, fig_dir)
    extra['mpc'] = fig_mpc(runs, fig_dir)
    extra['timing'] = fig_timing(runs, fig_dir)
    extra['manip'] = fig_manip(runs, fig_dir)
    print('tables ->', tab_dir)
    summary = write_tables(runs, tab_dir, extra)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 200)
    print(summary.round(3).to_string())


if __name__ == '__main__':
    main()
