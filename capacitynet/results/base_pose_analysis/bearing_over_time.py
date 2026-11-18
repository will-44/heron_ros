# -*- coding: utf-8 -*-
"""Where the mobile base ends up relative to the goal, and why it is not in front of it.

`samples.csv` logs the MPC goal in `dsr01/world`, the arm's base-attached frame, so
    bearing  = atan2(goal_y, goal_x)   (0° = goal straight ahead of the base)
    distance = hypot(goal_x, goal_y)
are the goal's polar coordinates as the arm sees them. A fixed-base run must show a flat
bearing — that is the sanity check for the frame.

`control_debug.csv` gives the model's own gradient (grad_x, grad_y) each cycle. Projected
on the goal direction and on its perpendicular it says what the controller *wants*:
    radial > 0  -> moving along ∇Q brings the base closer to the goal
    facing > 0  -> moving along ∇Q reduces |bearing|, i.e. turns the base toward the goal

Run from the results directory:  python3 base_pose_analysis/bearing_over_time.py
"""
import importlib.util
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
FIG, TAB = os.path.join(HERE, 'figures'), os.path.join(HERE, 'tables')

FIX, MOB, INK, MUT = '#E8734A', '#4C8BC9', '#141413', '#6E6C64'
COND_COLOR = {'base fixe': FIX, 'base mobile': MOB}
LEGS = [('act_move_pregrasp', 'approche de saisie', 45),
        ('act_move_preplace', 'approche de dépose', 90)]


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_runs():
    A = _mod('A', os.path.join(RESULTS, 'exp_1_resultat_finaux_heron/analysis/analyze_exp1.py'))
    E2 = _mod('E2', os.path.join(RESULTS, 'final_reuleaux_exp1/analysis/analyze_exp2.py'))
    RA = _mod('RA', os.path.join(RESULTS, 'exp_2_moving_base/analysis/reach_analysis.py'))
    S2 = _mod('S2', os.path.join(RESULTS, 'exp_2_moving_base/analysis/analyze_setup2.py'))
    spec = [('setup 1 — table vide', 'base fixe', E2.load, 'final_reuleaux_exp1'),
            ('setup 1 — table vide', 'base mobile', A.load, 'exp_1_resultat_finaux_heron'),
            ('setup 2 — table encombrée', 'base fixe', S2.load_fixed, 'exp2_base_fixe'),
            ('setup 2 — table encombrée', 'base mobile', S2.load_moving, 'exp_2_moving_base')]
    out = []
    for setup, cond, loader, folder in spec:
        d = loader(os.path.join(RESULTS, folder))
        keep = set(d['idx'].episode_id)
        s = d['samples'][d['samples'].episode_id.isin(keep)].copy()
        s['bearing'] = np.degrees(np.arctan2(s.goal_y, s.goal_x)).abs()
        s['distance'] = np.hypot(s.goal_x, s.goal_y)
        c = d['ctrl'][d['ctrl'].episode_id.isin(keep)].copy()
        out.append((setup, cond, s, c, RA))
    return out


def polar_table(runs):
    rows = []
    for setup, cond, s, _, _ in runs:
        for sub, label, _ in LEGS:
            g = s[s.sub_state.astype(str) == sub]
            for ep, e in g.groupby('episode_id'):
                e = e.sort_values('t_rel')
                if len(e) < 5:
                    continue
                rows.append(dict(setup=setup, condition=cond, leg=label, episode_id=ep,
                                 bearing_start=e.bearing.iloc[0], bearing_end=e.bearing.iloc[-1],
                                 distance_start=e.distance.iloc[0], distance_end=e.distance.iloc[-1],
                                 q_start=e.q_goal.iloc[0], q_end=e.q_goal.iloc[-1]))
    df = pd.DataFrame(rows)
    df['d_bearing'] = df.bearing_end - df.bearing_start
    df['d_distance'] = df.distance_end - df.distance_start
    return df


def gradient_table(runs):
    """What the gradient asks for, over the leg and at the last three cycles."""
    rows = []
    for setup, cond, s, c, _ in runs:
        if cond == 'base fixe':
            continue
        c = c.sort_values('timestamp')
        s2 = s.sort_values('timestamp')[['timestamp', 'goal_x', 'goal_y']]
        m = pd.merge_asof(c, s2, on='timestamp', direction='nearest', tolerance=2.0).dropna(subset=['goal_x'])
        g = m[['goal_x', 'goal_y']].to_numpy(float)
        n = np.linalg.norm(g, axis=1)
        gh = g / n[:, None]
        G = m[['grad_x', 'grad_y']].to_numpy(float)
        gm = np.linalg.norm(G, axis=1)
        ok = gm > 1e-9
        Gh = np.zeros_like(G)
        Gh[ok] = G[ok] / gm[ok, None]
        th = np.arctan2(g[:, 1], g[:, 0])
        perp = np.column_stack([-gh[:, 1], gh[:, 0]])
        m = m.assign(radial=(gh * Gh).sum(1), facing=np.sign(th) * (perp * Gh).sum(1))
        m = m[(m.allows_motion == True) & (m.enabled == True)]      # noqa: E712
        for sub, label, _ in LEGS:
            gl = m[m.phase.astype(str).str.contains(sub, na=False)]
            if len(gl) < 20:
                continue
            last = gl.groupby('episode_id').tail(3)
            rows.append(dict(setup=setup, leg=label, n_cycles=len(gl),
                             grad_mag=gl.grad_mag.median(),
                             radial=gl.radial.median(), facing=gl.facing.median(),
                             facing_frac_pos=float((gl.facing > 0).mean()),
                             grad_mag_end=last.grad_mag.median(),
                             radial_end=last.radial.median(), facing_end=last.facing.median()))
    return pd.DataFrame(rows)


def stop_table(runs):
    rows = []
    for setup, cond, _, c, RA in runs:
        if cond == 'base fixe':
            continue
        for sub, label, _ in LEGS:
            g = c[c.phase.astype(str).str.contains(sub, na=False)]
            if g.empty:
                continue
            vc = g.groupby('episode_id').apply(RA.stop_reason, include_groups=False).value_counts()
            rows.append(dict(setup=setup, leg=label, **{k: int(v) for k, v in vc.items()}))
    return pd.DataFrame(rows).fillna(0)


def fig_polar(runs, bin_s=2.0):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6), sharex='col')
    setups = ['setup 1 — table vide', 'setup 2 — table encombrée']
    for col, setup in enumerate(setups):
        for row, (sub, label, xmax) in enumerate(LEGS):
            ax = axes[row][col]
            for s, cond, sm, _, _ in runs:
                if s != setup:
                    continue
                series = []
                for ep, e in sm[sm.sub_state.astype(str) == sub].groupby('episode_id'):
                    e = e.sort_values('t_rel')
                    t = e.t_rel.to_numpy(float)
                    series.append((t - t[0], e.bearing.to_numpy(float)))
                edges = np.arange(0, xmax + bin_s, bin_s)
                mid, med, lo, hi = [], [], [], []
                for k in range(len(edges) - 1):
                    vals = [v[(tt >= edges[k]) & (tt < edges[k + 1])].mean()
                            for tt, v in series if ((tt >= edges[k]) & (tt < edges[k + 1])).any()]
                    if len(vals) < 5:
                        continue
                    mid.append(0.5 * (edges[k] + edges[k + 1]))
                    med.append(np.median(vals))
                    lo.append(np.percentile(vals, 25))
                    hi.append(np.percentile(vals, 75))
                c = COND_COLOR[cond]
                ax.fill_between(mid, lo, hi, color=c, alpha=0.16, linewidth=0)
                ax.plot(mid, med, color=c, lw=2, label=cond)
            ax.axhline(0, color=INK, lw=1)
            ax.set_ylim(0, 70)
            ax.set_xlim(0, xmax)
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.set_title(f'{setup.split("—")[1].strip()} — {label}', fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel('|azimut de la cible| (°)', fontsize=9)
            if row == 1:
                ax.set_xlabel('temps depuis le début de la phase (s)', fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle('Azimut de la cible vu par la base — 0° = cible droit devant',
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIG, 'bearing_over_time.png'), dpi=150)
    plt.close(fig)


def write_md(df, path):
    cols = list(df.columns)
    with open(path, 'w') as fh:
        fh.write('| ' + ' | '.join(cols) + ' |\n')
        fh.write('|' + '---|' * len(cols) + '\n')
        for _, r in df.iterrows():
            fh.write('| ' + ' | '.join(f'{r[c]:.3f}' if isinstance(r[c], float) else str(r[c])
                                       for c in cols) + ' |\n')


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    runs = load_runs()
    pd.set_option('display.width', 220)

    pt = polar_table(runs)
    pt.to_csv(os.path.join(TAB, 'polar_legs.csv'), index=False)
    agg = pt.groupby(['setup', 'condition', 'leg']).agg(
        n=('episode_id', 'size'),
        bearing_start=('bearing_start', 'median'), bearing_end=('bearing_end', 'median'),
        d_bearing=('d_bearing', 'median'),
        distance_start=('distance_start', 'median'), distance_end=('distance_end', 'median'),
        q_start=('q_start', 'median'), q_end=('q_end', 'median')).round(3).reset_index()
    agg.to_csv(os.path.join(TAB, 'polar_summary.csv'), index=False)
    write_md(agg, os.path.join(TAB, 'polar_summary.md'))
    print('\n== goal in the base frame, start vs end of leg')
    print(agg.to_string(index=False))

    gt = gradient_table(runs)
    gt.to_csv(os.path.join(TAB, 'gradient_direction.csv'), index=False)
    print('\n== what the gradient asks for (mobile runs only)')
    print(gt.round(3).to_string(index=False))

    st = stop_table(runs)
    st.to_csv(os.path.join(TAB, 'stop_reasons.csv'), index=False)
    print('\n== why the base stopped')
    print(st.to_string(index=False))

    fig_polar(runs)
    print('\nfigures ->', FIG)


if __name__ == '__main__':
    main()
