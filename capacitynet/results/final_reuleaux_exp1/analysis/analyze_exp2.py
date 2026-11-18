#!/usr/bin/env python3
"""Offline analysis of experiment 2 (fixed base: controller ON, base physically stopped).

Same pipeline as experiment 1 (`../../exp_1_resultat_finaux_heron/analysis/analyze_exp1.py`,
imported here), with this run's operator notes and outcome classes, plus one
figure specific to the fixed-base condition: how the *wanted* direction
(gradient / commanded velocity) evolves while the arm moves and the base stays put.

    python3 analysis/analyze_exp2.py            # from the run directory
    python3 analysis/analyze_exp2.py --run-dir /path/to/run --out /path/to/out

Outputs (relative to --out, default = run directory): exp_1_fixed_figures/fig_*.png,
exp_1_fixed_tables/*.

Facts about this run that the code relies on (source: experimenter, 2026-09-16):

* `base_control` was true: the controller evaluated Q, its gradient and published
  /cmd_vel exactly as in experiment 1, but the base's motor relay was disabled,
  so the base did not move. The recorder again never received the base-side
  Twist topic; `x_est/y_est/dist_est` are zero. The "commanded displacement" is
  therefore a virtual quantity here (integral of a command that was not
  executed) and is only used to show what the controller wanted.
* Episodes 16 and 18: the base did move (relay re-enabled by mistake) -> excluded.
* Episodes 1, 3, 4: place completed, cancelled during the final homing -> success
  (same rule as episodes 24/27 of experiment 1).
* Episodes 5, 8, 10, 13 (`Grasp failed: Sequence failed`) and 2 (`Release failed`):
  the arm was very close to the object but the grasp/release sequence failed;
  neither a collision nor a success -> class 'grasp/release failed'.
* Every other cancel (9, 19, 21) is an arm collision with the environment.
* The base was re-placed by hand after episode 18: episodes 0-17 start from pose
  A (pick goal at x ~ 0.43 m in the arm frame), episodes 19-24 from pose B
  (x ~ 0.48 m, place goal x ~ 0.65 instead of 0.55 m).
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, '..', '..'))


def find_exp1_run(results=RESULTS):
    """The experiment-1 run directory, located by its content rather than its name.

    The run directories get renamed by hand between campaigns, so look for the
    one that holds the shared analysis library instead of hard-coding a name.
    """
    for name in sorted(os.listdir(results)):
        cand = os.path.join(results, name)
        if os.path.isfile(os.path.join(cand, 'analysis', 'analyze_exp1.py')):
            return cand
    raise FileNotFoundError(f'no experiment-1 run (analysis/analyze_exp1.py) under {results}')


EXP1_RUN_DIR = find_exp1_run()
EXP1_ANALYSIS = os.path.join(EXP1_RUN_DIR, 'analysis')
sys.path.insert(0, EXP1_ANALYSIS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import analyze_exp1 as A  # noqa: E402

BASE_MOVED = ('excluded', 'base moved during the cycle (fixed-base condition violated)')
HOMING_CANCEL = ('success', 'pick-and-place completed; cancelled during the final homing')
NEAR_MISS = ('grasp/release failed', 'grasp sequence failed: arm at the object but grasp missed (no collision)')
EPISODE_NOTES = {
    16: BASE_MOVED,
    18: BASE_MOVED,
    1: HOMING_CANCEL,
    3: HOMING_CANCEL,
    4: HOMING_CANCEL,
    2: ('grasp/release failed', 'release sequence failed after a 134 s place approach (no collision)'),
    5: NEAR_MISS, 8: NEAR_MISS, 10: NEAR_MISS, 13: NEAR_MISS,
    9: ('collision', 'arm collision with the environment during the place approach; cancelled'),
    21: ('collision', 'arm collision with the environment during the place approach; cancelled'),
    19: ('collision', 'arm collision with the environment during the place move; cancelled'),
}
DEFAULT_CANCEL = ('collision', 'arm collision with the environment; cancelled')
DEFAULT_FAILURE = ('grasp/release failed', 'grasp/release sequence failed (no collision)')
POSE_B_FROM = 19  # episodes >= this start from the re-placed base pose


def add_pose(idx):
    idx['pose'] = np.where(idx.episode_id >= POSE_B_FROM, 'B', 'A')
    return idx


def reclassify_homing_cancels(idx, phases):
    """A cancel whose last phase is the final homing completed the pick-and-place.

    Episodes 1, 3, 4 are listed explicitly in EPISODE_NOTES; this catches the same
    case for episodes recorded after the notes were written (the run was still
    recording when this analysis was made).
    """
    last = phases.sort_values('t_start_rel').groupby('episode_id').last().phase
    for i, e in idx.iterrows():
        if e.outcome == 'cancelled' and int(e.episode_id) not in EPISODE_NOTES \
                and last.get(e.episode_id) == 'act_home_exit':
            idx.loc[i, ['cls', 'note']] = HOMING_CANCEL
    return idx


# The run kept recording after this analysis was written; the report describes the
# 30-episode snapshot 0-29, so the loader stops there. Raise it (or set it to None)
# and re-run both scripts to fold the later episodes in — the report's numbers then
# have to be refreshed with them.
SNAPSHOT_MAX_EP = 29


def load(run_dir, max_ep=None):
    """Experiment-2 data with this run's classification (used by compare_runs.py too)."""
    data = A.load(run_dir, notes=EPISODE_NOTES, default_cancel=DEFAULT_CANCEL,
                  default_failure=DEFAULT_FAILURE)
    max_ep = SNAPSHOT_MAX_EP if max_ep is None else max_ep
    if max_ep is not None:
        for k in ('idx', 'phases', 'samples', 'ctrl', 'cmd_vel', 'mpc'):
            d = data.get(k)
            if d is not None and 'episode_id' in getattr(d, 'columns', []):
                data[k] = d[d.episode_id <= max_ep]
    reclassify_homing_cancels(data['idx'], data['phases'])
    add_pose(data['idx'])
    return data


def heading_over_leg(ctrl, idx, phase, bin_s=5.0, t_max=70.0):
    """Per-leg commanded heading (deg) in bins of `bin_s` seconds since the first full-region cycle.

    Returns (bins, H) with H[leg, bin] the circular mean of the gradient direction in
    that bin (NaN where the leg has no cycle), restricted to valid episodes.
    """
    valid = A.outcome_valid(idx).episode_id
    c = ctrl[ctrl.active & (ctrl.phase == phase) & (ctrl.region_size >= 3) & ctrl.episode_id.isin(valid)]
    bins = np.arange(0, t_max, bin_s)
    H = []
    for _, g in c.groupby('episode_id'):
        g = g.sort_values('timestamp')
        t = (g.timestamp - g.timestamp.iloc[0]).to_numpy()
        ang = np.arctan2(g.grad_y.to_numpy(), g.grad_x.to_numpy())
        row = []
        for b in bins:
            m = (t >= b) & (t < b + bin_s)
            row.append(np.degrees(np.arctan2(np.sin(ang[m]).mean(), np.cos(ang[m]).mean())) if m.any() else np.nan)
        H.append(row)
    return bins, np.array(H)


def circ_mean_deg(H):
    """Circular mean across legs (axis 0) and the resultant length R in [0, 1]."""
    s, c = np.nanmean(np.sin(np.radians(H)), axis=0), np.nanmean(np.cos(np.radians(H)), axis=0)
    return np.degrees(np.arctan2(s, c)), np.hypot(s, c)


def fig_wanted_direction(data, out_dir):
    """Fixed base: the direction the controller wanted, as a function of time in the leg.

    With the base stopped, any change of the commanded direction over the leg is
    caused by the arm's motion (region spheres following the MPC path, voxel-grid
    changes), not by the base's position.
    """
    ctrl, idx = data['ctrl'], data['idx']
    fig, axes = A.plt.subplots(1, 2, figsize=(12, 5.2), facecolor=A.SURFACE, subplot_kw=dict(projection='polar'))
    for ax, phase in zip(axes, (A.PREGRASP, A.PREPLACE)):
        ax.set_facecolor(A.SURFACE)
        ax.spines['polar'].set_color(A.AXIS)
        ax.grid(True, color=A.GRID, linewidth=0.6)
        ax.tick_params(colors=A.INK2, labelsize=8)
        bins, H = heading_over_leg(ctrl, idx, phase)
        mean_deg, R = circ_mean_deg(H)
        n = np.sum(~np.isnan(H), axis=0)
        ok = n >= 5
        # One point per (leg, bin): angle = wanted direction, radius = time in the leg.
        rng = np.random.default_rng(0)
        for row in H:
            jitter = rng.uniform(-1.2, 1.2, len(row))
            ax.plot(np.radians(row), bins + 2.5 + jitter, 'o', color=A.LEG_COLOR[phase], alpha=0.35,
                    markersize=3, markeredgewidth=0)
        ax.plot(np.radians(mean_deg[ok]), bins[ok] + 2.5, '-', color=A.INK, linewidth=2.2)
        ax.plot(np.radians(mean_deg[ok]), bins[ok] + 2.5, 'o', color=A.INK, markersize=4)
        for b, m in zip(bins[ok], mean_deg[ok]):
            if int(b) % 10 == 0:
                ax.annotate(f'{int(b)}–{int(b) + 5} s', (np.radians(m), b + 2.5), xytext=(6, 4),
                            textcoords='offset points', fontsize=7, color=A.INK)
        # Arm frame: +x forward (up), +y left (left of the plot) -> counter-clockwise angles.
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)
        ax.set_xticks(np.radians([0, 90, 180, 270]))
        ax.set_xticklabels(['+x (forward)', '+y (left)', '−x (back)', '−y (right)'], fontsize=8, color=A.INK2)
        ax.set_rlim(0, bins[ok].max() + 6)
        ax.set_rlabel_position(250)
        ax.set_yticks(bins[ok][::2] + 2.5)
        ax.set_yticklabels([f'{int(b)} s' for b in bins[ok][::2]], fontsize=7, color=A.MUTED)
        ax.set_title(f'{A.LEG_LABEL[phase]}: wanted direction (angle) vs time in leg (radius)',
                     fontsize=9.5, color=A.INK, loc='left', pad=14)
        ax.text(0.0, -0.13, 'black: circular mean over legs per 5 s bin; concentration R = '
                + ', '.join(f'{r:.2f}' for r in R[ok][:6]) + ' …',
                transform=ax.transAxes, fontsize=7.5, color=A.MUTED)
    axes[0].legend(handles=[Line2D([], [], marker='o', color=A.BLUE, linestyle='none', alpha=0.5,
                                   label='one leg, one 5 s bin'),
                            Line2D([], [], marker='o', color=A.INK, linewidth=2, label='circular mean over legs')],
                   fontsize=7.5, frameon=False, loc='upper left', bbox_to_anchor=(-0.25, 1.12))
    fig.suptitle('Fixed base: the direction the controller wanted rotates as the arm moves',
                 fontsize=11, color=A.INK, x=0.01, ha='left')
    fig.tight_layout()
    A.save(fig, out_dir, 'fig_wanted_direction.png')
    return bins, {phase: circ_mean_deg(heading_over_leg(ctrl, idx, phase)[1]) for phase in (A.PREGRASP, A.PREPLACE)}


FIG_DIR, TAB_DIR = 'exp_1_fixed_figures', 'exp_1_fixed_tables'


def run(run_dir, out):
    data, cv, legs, pairs, summary = A.run(
        run_dir, out, loader=load, base_moves=False,
        diag_classes=('success', 'collision'),
        diag_title='Place-approach diagnostics by outcome (fixed base)',
        example_eps=(0, 22), timeline_t_max=370, leg_t_max=70,
        fig_dirname=FIG_DIR, tab_dirname=TAB_DIR)
    fig_dir, tab_dir = os.path.join(out, FIG_DIR), os.path.join(out, TAB_DIR)
    bins, headings = fig_wanted_direction(data, fig_dir)
    rows = []
    for phase, (mean_deg, R) in headings.items():
        for b, m, r in zip(bins, mean_deg, R):
            if not np.isnan(m):
                rows.append(dict(leg=A.LEG_LABEL[phase], t_bin_start_s=b, heading_deg=m, concentration_R=r))
    pd.DataFrame(rows).round(3).to_csv(os.path.join(tab_dir, 'wanted_direction.csv'), index=False)
    # Pose A/B on the episode table
    ep = pd.read_csv(os.path.join(tab_dir, 'episodes.csv'))
    ep['pose'] = np.where(ep.episode_id >= POSE_B_FROM, 'B', 'A')
    ep.to_csv(os.path.join(tab_dir, 'episodes.csv'), index=False)
    print(f'  wrote {TAB_DIR}/wanted_direction.csv')
    return data, cv, legs, pairs, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', default=os.path.dirname(HERE))
    ap.add_argument('--out', default=None, help='output root (default: run dir)')
    args = ap.parse_args()
    run(args.run_dir, args.out or args.run_dir)


if __name__ == '__main__':
    main()
