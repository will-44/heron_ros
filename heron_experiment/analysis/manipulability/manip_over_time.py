# -*- coding: utf-8 -*-
"""Time-resolved analysis of the arm's Yoshikawa manipulability over the approach legs.

The per-phase means of `cmp_manip.png` mix legs of very different durations (the
fixed-base place approach lasts twice as long as the moving-base one), so a higher
mean can simply mean "more time spent parked in the final posture". This script
works on the raw `manipulability` column of `joint_states.csv` (~106 Hz) instead,
aligned on the start of each approach leg, and reports *when* the arm reaches a
well-conditioned posture rather than what it averages.

Run from heron_experiment/:  python3 analysis/manipulability/manip_over_time.py
Writes results/analysis/manipulability/{figures,tables}/.
Outputs: manipulability/figures/*.png, manipulability/tables/*.csv|md
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import runs  # noqa: E402
RESULTS = str(runs.paths.results_dir())
import leg_survival as LS  # noqa: E402
OUT = runs.study_dir('manipulability')
FIG, TAB = os.path.join(OUT, 'figures'), os.path.join(OUT, 'tables')

FIX, MOB, INK, MUT, RULE = '#E8734A', '#4C8BC9', '#141413', '#6E6C64', '#E2DFD5'
COND_COLOR = {'base fixe': FIX, 'base mobile': MOB}

# Two reference levels on w, deliberately NOT called "singular".
# W_LOW is a low-manipulability band, not a singularity: across the four runs 34 % of all
# logged samples sit below it, and the arm's own parked posture sits at w = 0.012-0.020,
# i.e. below it. It is a comparison band, nothing more.
W_LOW = 0.05
# W_NEAR_SING is where the posture really is degenerate. Measured against it, the place
# approach spends 0.0 s there in BOTH conditions (its w never drops under 0.032); only
# the pick approach, which sweeps the wrist through q5 = 0, actually goes there.
W_NEAR_SING = 0.01
# A leg is "settled" once w stays above 80 % of the value it ends the leg on.
SETTLE_FRAC = 0.80
ARRIVAL_S = 3.0        # window at the end of a leg used for "w on arrival"


def load_runs():
    """The four runs, each with its own classification (excluded episodes dropped)."""
    import analyze_exp1 as A
    loaded = [(c.setup_label, c.base_label, c.load(c.raw)) for c in runs.conditions()]
    out = []
    for setup, cond, d in loaded:
        # idx keeps the excluded episodes (to document them); the measurement frames do
        # not. Intersecting also drops episodes recorded after a run's snapshot.
        keep = set(d['idx'].episode_id) & set(d['joints'].episode_id)
        d['joints'] = d['joints'][d['joints'].episode_id.isin(keep)].reset_index(drop=True)
        out.append((setup, cond, d, A))
    return out


LEGS = [('act_grasp:act_move_pregrasp', 'approche de saisie', 45),
        ('act_release:act_move_preplace', 'approche de dépose', 120)]


def leg_samples(data, phase):
    """One (episode_id, t_leg, w) frame per approach leg, time zeroed on the leg start."""
    j = data['joints']
    g = j[j.phase == phase]
    for ep, e in g.groupby('episode_id'):
        e = e.sort_values('t_rel')
        if len(e) < 10:
            continue
        yield ep, (e.t_rel - e.t_rel.iloc[0]).to_numpy(), e.manipulability.to_numpy()


def leg_table(runs):
    rows = []
    for setup, cond, d, _ in runs:
        cls = dict(zip(d['idx'].episode_id, d['idx'].cls))
        for phase, leg, _ in LEGS:
            for ep, t, w in leg_samples(d, phase):
                dur = t[-1]
                w_arr = float(np.median(w[t >= max(dur - ARRIVAL_S, 0)]))
                target = SETTLE_FRAC * w_arr
                # Settling time: the instant after which w never drops back below
                # 80 % of its final value — a leg that dips mid-way (the arm passing
                # through a stretched posture) is not "settled" by its first crossing.
                below = np.nonzero(w < target)[0]
                rows.append(dict(
                    setup=setup, condition=cond, leg=leg, episode_id=ep,
                    cls=cls.get(ep, '?'), duration_s=dur,
                    w_start=float(w[0]), w_median=float(np.median(w)), w_min=float(w.min()),
                    w_max=float(w.max()), w_arrival=w_arr,
                    t_settle_s=float(t[below[-1]]) if len(below) else 0.0,
                    frac_low=float(np.mean(w < W_LOW)),
                    t_low_s=float(np.mean(w < W_LOW) * dur),
                    frac_near_sing=float(np.mean(w < W_NEAR_SING)),
                    t_near_sing_s=float(np.mean(w < W_NEAR_SING) * dur)))
    return pd.DataFrame(rows)


def band(ax, runs_for_panel, phase, t_max=None, min_legs=None):
    # `t_max` / `min_legs` kept for call compatibility: the axis now ends with the longest
    # leg and the median is split solid / dashed instead of being cut at a quorum.
    grid = np.arange(0, LS.T_GRID_MAX + 1, 1.0)
    t_end = 0.0
    for cond, d in runs_for_panel:
        stack = [np.interp(grid, t, w, right=np.nan) for _, t, w in leg_samples(d, phase)]
        _, te, _ = LS.median_over_legs(ax, grid, np.array(stack), COND_COLOR[cond], label=cond)
        t_end = max(t_end, te)
    ax.axhline(W_LOW, color=MUT, linestyle=':', linewidth=1)
    ax.set_xlim(0, t_end + 1)
    ax.set_ylim(0, 0.30)
    ax.grid(True, color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUT, labelsize=9)


def fig_curves(runs):
    setups = sorted({s for s, _, _, _ in runs}, key=lambda x: x[:7])
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharey=True)
    fig.patch.set_facecolor('white')
    for r, (phase, leg, t_max) in enumerate(LEGS):
        for c, setup in enumerate(setups):
            ax = axes[r, c]
            band(ax, [(cond, d) for s, cond, d, _ in runs if s == setup], phase, t_max)
            if r == 0:
                ax.set_title(setup, fontsize=11, color=INK, loc='left')
            ax.set_xlabel('temps depuis le début de la phase (s)', fontsize=9.5, color=MUT)
            if c == 0:
                ax.set_ylabel('%s\nmanipulabilité w' % leg, fontsize=10, color=INK)
    axes[0, 0].legend(handles=LS.legend_handles([(c, COND_COLOR[c]) for c in COND_COLOR], INK),
                      frameon=False, fontsize=8, loc='upper left')
    axes[0, 0].text(1, W_LOW + 0.006, 'bande basse w = %.2f (34 %% du temps total)' % W_LOW,
                    fontsize=8, color=MUT)
    fig.suptitle('Manipulabilité du bras au fil de l’approche — médiane et intervalle interquartile sur les approches',
                 fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'manip_over_time.png'), dpi=130)
    plt.close(fig)


def fig_metrics(legs):
    metrics = [('t_settle_s', 'stabilisation (s)'),
               ('w_arrival', 'w à l’arrivée'),
               ('t_low_s', 'temps sous w = %.2f (s)' % W_LOW)]
    panels = [(s, leg) for s in sorted(legs.setup.unique(), key=lambda x: x[:7])
              for _, leg, _ in LEGS]
    fig, axes = plt.subplots(len(metrics), len(panels), figsize=(13, 8), sharey='row')
    fig.patch.set_facecolor('white')
    rng = np.random.default_rng(0)
    for r, (key, label) in enumerate(metrics):
        for c, (setup, leg) in enumerate(panels):
            ax = axes[r, c]
            sub = legs[(legs.setup == setup) & (legs.leg == leg)]
            for x, cond in enumerate(('base fixe', 'base mobile')):
                v = sub[sub.condition == cond][key].dropna()
                ax.scatter(x + rng.uniform(-0.09, 0.09, len(v)), v, s=22,
                           color=COND_COLOR[cond], alpha=0.75, linewidths=0)
                if len(v):
                    ax.plot([x - 0.25, x + 0.25], [v.median()] * 2, color=INK, linewidth=2)
                    ax.annotate('%.3g' % v.median(), (x + 0.28, v.median()), fontsize=8.5,
                                color=INK, va='center')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['fixe', 'mobile'], fontsize=9, color=MUT)
            ax.set_xlim(-0.5, 1.6)
            ax.grid(True, axis='y', color=RULE, linewidth=0.8)
            ax.set_axisbelow(True)
            for s_ in ('top', 'right'):
                ax.spines[s_].set_visible(False)
            for s_ in ('left', 'bottom'):
                ax.spines[s_].set_color(RULE)
            ax.tick_params(colors=MUT, labelsize=9)
            if r == 0:
                ax.set_title('%s\n%s' % (setup, leg), fontsize=9.5, color=INK)
            if c == 0:
                ax.set_ylabel(label, fontsize=9.5, color=INK)
    fig.suptitle('Ce que la moyenne par phase cache : dynamique de la manipulabilité, une approche par point',
                 fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(os.path.join(FIG, 'manip_dynamics.png'), dpi=130)
    plt.close(fig)


def fig_outcome(runs):
    """Place approach only: successful cycles against cycles ending in a collision."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    fig.patch.set_facecolor('white')
    setups = sorted({s for s, _, _, _ in runs}, key=lambda x: x[:7])
    grid = np.arange(0, 121, 1.0)
    styles = {'success': ('-', 'cycles réussis'), 'collision': ((0, (4, 2)), 'cycles finissant en collision')}
    for ax, setup in zip(axes, setups):
        for s, cond, d, _ in runs:
            if s != setup:
                continue
            cls = dict(zip(d['idx'].episode_id, d['idx'].cls))
            for k, (ls, _lab) in styles.items():
                stack = [np.interp(grid, t, w, right=np.nan)
                         for ep, t, w in leg_samples(d, LEGS[1][0]) if cls.get(ep) == k]
                if len(stack) < 3:
                    continue
                arr = np.array(stack)
                n = np.sum(~np.isnan(arr), axis=0)
                alive = n >= 1
                solid = n >= max(2, int(np.ceil(LS.SOLID_FRAC * n.max())))
                with np.errstate(all='ignore'):
                    med = np.nanmedian(arr, axis=0)
                ax.plot(grid[alive], med[alive], linestyle=':', color=COND_COLOR[cond],
                        linewidth=1.6, alpha=0.9)
                ax.plot(grid[solid], med[solid], linestyle=ls, color=COND_COLOR[cond],
                        linewidth=2)
        ax.axhline(W_LOW, color=MUT, linestyle=':', linewidth=1)
        ax.set_title(setup, fontsize=11, color=INK, loc='left')
        ax.set_xlabel('temps depuis le début de l’approche de dépose (s)', fontsize=9.5, color=MUT)
        ax.grid(True, color=RULE, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, 0.30)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        for s_ in ('left', 'bottom'):
            ax.spines[s_].set_color(RULE)
        ax.tick_params(colors=MUT, labelsize=9)
    axes[0].set_ylabel('manipulabilité w', fontsize=10, color=INK)
    axes[0].legend(handles=[Line2D([], [], color=FIX, lw=2, label='base fixe'),
                            Line2D([], [], color=MOB, lw=2, label='base mobile'),
                            Line2D([], [], color=INK, lw=2, label='cycles réussis'),
                            Line2D([], [], color=INK, lw=2, ls=(0, (4, 2)),
                                   label='cycles finissant en collision')],
                   frameon=False, fontsize=8.5, loc='lower right', ncol=2)
    fig.suptitle('Approche de dépose : la manipulabilité distingue-t-elle les cycles qui échouent ?',
                 fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'manip_by_outcome.png'), dpi=130)
    plt.close(fig)


def summary(legs):
    out = []
    keys = [('t_settle_s', 'temps avant stabilisation à 80 % de w final (s)'),
            ('w_arrival', 'w à l’arrivée'),
            ('w_median', 'w médian sur la phase'),
            ('w_min', 'w minimal'),
            ('t_low_s', 'temps dans la bande basse w < %.2f (s)' % W_LOW),
            ('t_near_sing_s', 'temps réellement quasi-singulier w < %.2f (s)' % W_NEAR_SING),
            ('duration_s', 'durée de la phase (s)')]
    for setup in sorted(legs.setup.unique(), key=lambda x: x[:7]):
        for _, leg, _ in LEGS:
            sub = legs[(legs.setup == setup) & (legs.leg == leg)]
            f = sub[sub.condition == 'base fixe']
            m = sub[sub.condition == 'base mobile']
            for key, label in keys:
                a, b = f[key].dropna(), m[key].dropna()
                p = mannwhitneyu(a, b).pvalue if len(a) > 2 and len(b) > 2 else np.nan
                out.append(dict(setup=setup, leg=leg, metric=label, n_fixe=len(a), n_mobile=len(b),
                                fixe=a.median(), mobile=b.median(), p_mannwhitney=p))
    return pd.DataFrame(out)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    runs = load_runs()
    legs = leg_table(runs)
    legs.to_csv(os.path.join(TAB, 'manip_legs.csv'), index=False)
    fig_curves(runs)
    fig_metrics(legs)
    fig_outcome(runs)
    s = summary(legs)
    s.to_csv(os.path.join(TAB, 'manip_summary.csv'), index=False)
    with open(os.path.join(TAB, 'manip_summary.md'), 'w') as fh:
        cols = list(s.columns)
        fh.write('| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n')
        for _, r in s.iterrows():
            cells = [('%.3f' % r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols]
            fh.write('| ' + ' | '.join(cells) + ' |\n')
    print(s.to_string(index=False, float_format=lambda v: '%.3f' % v))


if __name__ == '__main__':
    main()
