# -*- coding: utf-8 -*-
"""Arm joint velocities (and the base's commanded speed) over the approach legs.

Companion to manipulability_analysis/: same runs, same exclusion rule, same alignment
on the start of each approach leg. `joint_states.csv` carries vel_joint_1..6 at ~106 Hz.

Two views of the same data:
  - pooled |q̇| over the six joints (comparable to the deck's acceleration metric),
  - the norm ‖q̇‖ of the joint-velocity vector (what "how fast the arm moves" means
    when the joints are considered together).

Run from the results directory:  python3 velocity_analysis/vel_over_time.py
"""
import importlib.util
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
sys.path.insert(0, RESULTS)
import leg_survival as LS  # noqa: E402
FIG, TAB = os.path.join(HERE, 'figures'), os.path.join(HERE, 'tables')

FIX, MOB, INK, MUT, RULE = '#E8734A', '#4C8BC9', '#141413', '#6E6C64', '#E2DFD5'
COND_COLOR = {'base fixe': FIX, 'base mobile': MOB}

LEGS = [('act_grasp:act_move_pregrasp', 'approche de saisie', 45),
        ('act_release:act_move_preplace', 'approche de dépose', 120)]
MOVE_PHASES = [('act_home_enter', 'sortie de repos'),
               ('act_grasp:act_move_pregrasp', 'approche saisie'),
               ('act_grasp:act_move_grasp', 'saisie'),
               ('act_release:act_move_preplace', 'approche dépose'),
               ('act_release:act_move_place', 'dépose'),
               ('act_home_exit', 'retour repos')]


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_runs():
    A = _mod('A', os.path.join(RESULTS, 'exp_1_resultat_finaux_heron/analysis/analyze_exp1.py'))
    E2 = _mod('E2', os.path.join(RESULTS, 'final_reuleaux_exp1/analysis/analyze_exp2.py'))
    _mod('R', os.path.join(RESULTS, 'exp_2_moving_base/analysis/reach_analysis.py'))
    S2 = _mod('S2', os.path.join(RESULTS, 'exp_2_moving_base/analysis/analyze_setup2.py'))
    runs = [('setup 1 — table vide', 'base fixe', E2.load(os.path.join(RESULTS, 'final_reuleaux_exp1'))),
            ('setup 1 — table vide', 'base mobile', A.load(os.path.join(RESULTS, 'exp_1_resultat_finaux_heron'))),
            ('setup 2 — table encombrée', 'base fixe', S2.load_fixed(os.path.join(RESULTS, 'exp2_base_fixe'))),
            ('setup 2 — table encombrée', 'base mobile', S2.load_moving(os.path.join(RESULTS, 'exp_2_moving_base')))]
    out = []
    for setup, cond, d in runs:
        keep = set(d['idx'].episode_id) & set(d['joints'].episode_id)
        j = d['joints'][d['joints'].episode_id.isin(keep)].copy()
        vel = [c for c in j.columns if c.startswith('vel_joint_')]
        v = j[vel].to_numpy(dtype=float)
        j['v_norm'] = np.linalg.norm(v, axis=1)
        j['v_mean_abs'] = np.abs(v).mean(axis=1)
        j['v_max_abs'] = np.abs(v).max(axis=1)
        d['joints'] = j
        out.append((setup, cond, d, vel))
    return out


def per_phase_table(runs):
    rows = []
    for setup, cond, d, vel in runs:
        j = d['joints']
        for phase, label in MOVE_PHASES:
            g = j[j.phase == phase]
            if g.empty:
                continue
            flat = np.abs(g[vel].to_numpy(dtype=float)).ravel()
            flat = flat[np.isfinite(flat)]
            rows.append(dict(setup=setup, condition=cond, phase=label, n_samples=len(g),
                             v_abs_mean=flat.mean(), v_abs_p95=np.percentile(flat, 95),
                             v_abs_max=flat.max(),
                             v_norm_mean=g.v_norm.mean(), v_norm_p95=np.percentile(g.v_norm, 95),
                             v_norm_max=g.v_norm.max(),
                             frac_still=float((g.v_norm < 0.01).mean())))
    return pd.DataFrame(rows)


def leg_table(runs):
    rows = []
    for setup, cond, d, vel in runs:
        cls = dict(zip(d['idx'].episode_id, d['idx'].cls))
        for phase, leg, _ in LEGS:
            for ep, e in d['joints'][d['joints'].phase == phase].groupby('episode_id'):
                e = e.sort_values('t_rel')
                if len(e) < 10:
                    continue
                t = (e.t_rel - e.t_rel.iloc[0]).to_numpy()
                vn = e.v_norm.to_numpy()
                rows.append(dict(setup=setup, condition=cond, leg=leg, episode_id=ep,
                                 cls=cls.get(ep, '?'), duration_s=float(t[-1]),
                                 v_norm_mean=float(vn.mean()), v_norm_median=float(np.median(vn)),
                                 v_norm_p95=float(np.percentile(vn, 95)), v_norm_max=float(vn.max()),
                                 frac_still=float(np.mean(vn < 0.01)),
                                 t_still_s=float(np.mean(vn < 0.01) * t[-1]),
                                 joint_travel_rad=float(np.trapz(np.abs(e[vel].to_numpy(dtype=float)).sum(axis=1), t))))
    return pd.DataFrame(rows)


def _style(ax):
    ax.grid(True, color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUT, labelsize=9)


def fig_curves(runs, smooth_s=1.0):
    """‖q̇‖ over the legs, smoothed to 1 s so the 106 Hz ripple does not hide the shape."""
    setups = sorted({s for s, _, _, _ in runs}, key=lambda x: x[:7])
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharey='row')
    fig.patch.set_facecolor('white')
    for r, (phase, leg, _t_max) in enumerate(LEGS):
        grid = np.arange(0, LS.T_GRID_MAX + 1, 1.0)
        for c, setup in enumerate(setups):
            ax = axes[r, c]
            t_end = 0.0
            for s, cond, d, _ in runs:
                if s != setup:
                    continue
                stack = []
                for ep, e in d['joints'][d['joints'].phase == phase].groupby('episode_id'):
                    e = e.sort_values('t_rel')
                    t = (e.t_rel - e.t_rel.iloc[0]).to_numpy()
                    # mean ‖q̇‖ inside each 1 s bin, then interpolate on the common grid
                    binned = [e.v_norm.to_numpy()[(t >= g0) & (t < g0 + smooth_s)] for g0 in grid]
                    y = np.array([b.mean() if len(b) else np.nan for b in binned])
                    stack.append(y)
                _, te, _ = LS.median_over_legs(ax, grid, np.array(stack), COND_COLOR[cond],
                                               label=cond)
                t_end = max(t_end, te)
            ax.set_xlim(0, t_end + 1)   # l'axe s'arrête avec la jambe la plus longue
            _style(ax)
            if r == 0:
                ax.set_title(setup, fontsize=11, color=INK, loc='left')
            ax.set_xlabel('temps depuis le début de la phase (s)', fontsize=9.5, color=MUT)
            if c == 0:
                ax.set_ylabel('%s\n‖q̇‖ (rad/s)' % leg, fontsize=10, color=INK)
    axes[0, 0].legend(handles=LS.legend_handles([(c, COND_COLOR[c]) for c in COND_COLOR],
                                                INK), frameon=False, fontsize=8,
                      loc='upper right')
    fig.suptitle('Vitesse articulaire du bras au fil de l’approche — médiane et intervalle interquartile (moyenne par seconde)',
                 fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'vel_over_time.png'), dpi=130)
    plt.close(fig)


def fig_phases(phases):
    setups = sorted(phases.setup.unique(), key=lambda x: x[:7])
    fig, axes = plt.subplots(2, len(setups), figsize=(12, 7), sharex='col')
    fig.patch.set_facecolor('white')
    labels = [l for _, l in MOVE_PHASES]
    x = np.arange(len(labels))
    for c, setup in enumerate(setups):
        for r, (key, ylab) in enumerate([('v_norm_mean', 'moyenne de ‖q̇‖ (rad/s)'),
                                         ('v_norm_p95', 'p95 de ‖q̇‖ (rad/s)')]):
            ax = axes[r, c]
            for i, cond in enumerate(('base fixe', 'base mobile')):
                sub = phases[(phases.setup == setup) & (phases.condition == cond)].set_index('phase')
                vals = [sub[key].get(l, np.nan) for l in labels]
                ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=COND_COLOR[cond],
                       label=cond if r == 0 and c == 0 else None)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8.5, color=MUT, rotation=20, ha='right')
            _style(ax)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=9.5, color=INK)
            if r == 0:
                ax.set_title(setup, fontsize=11, color=INK, loc='left')
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('Vitesse articulaire par phase du cycle', fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'vel_by_phase.png'), dpi=130)
    plt.close(fig)


def summary(legs):
    keys = [('v_norm_mean', 'moyenne de ‖q̇‖ (rad/s)'), ('v_norm_p95', 'p95 de ‖q̇‖ (rad/s)'),
            ('v_norm_max', 'max de ‖q̇‖ (rad/s)'), ('t_still_s', 'temps quasi immobile (s)'),
            ('frac_still', 'fraction de la phase quasi immobile'),
            ('joint_travel_rad', 'course articulaire cumulée (rad)'),
            ('duration_s', 'durée de la phase (s)')]
    out = []
    for setup in sorted(legs.setup.unique(), key=lambda x: x[:7]):
        for _, leg, _ in LEGS:
            sub = legs[(legs.setup == setup) & (legs.leg == leg)]
            f, m = sub[sub.condition == 'base fixe'], sub[sub.condition == 'base mobile']
            for key, label in keys:
                a, b = f[key].dropna(), m[key].dropna()
                p = mannwhitneyu(a, b).pvalue if len(a) > 2 and len(b) > 2 else np.nan
                out.append(dict(setup=setup, leg=leg, metric=label, n_fixe=len(a), n_mobile=len(b),
                                fixe=a.median(), mobile=b.median(), p_mannwhitney=p))
    return pd.DataFrame(out)


def write_md(df, path):
    cols = list(df.columns)
    with open(path, 'w') as fh:
        fh.write('| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n')
        for _, r in df.iterrows():
            fh.write('| ' + ' | '.join(('%.3f' % r[c]) if isinstance(r[c], float) else str(r[c])
                                       for c in cols) + ' |\n')


def main():
    runs = load_runs()
    phases = per_phase_table(runs)
    legs = leg_table(runs)
    phases.to_csv(os.path.join(TAB, 'vel_by_phase.csv'), index=False)
    legs.to_csv(os.path.join(TAB, 'vel_legs.csv'), index=False)
    fig_curves(runs)
    fig_phases(phases)
    s = summary(legs)
    s.to_csv(os.path.join(TAB, 'vel_summary.csv'), index=False)
    write_md(s, os.path.join(TAB, 'vel_summary.md'))
    print(phases.to_string(index=False, float_format=lambda v: '%.3f' % v))
    print()
    print(s.to_string(index=False, float_format=lambda v: '%.3f' % v))


if __name__ == '__main__':
    main()
