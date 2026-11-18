# -*- coding: utf-8 -*-
"""Arm joint accelerations: what the logged column really contains, and what is left
once the differentiation noise is removed.

`workspace_reachability_node._on_joint_states` builds vel = Δpos/Δt and acc = Δvel/Δt
from consecutive /joint_states messages whose Δt is strongly irregular (~24 % of the
samples arrive < 2 ms after the previous one, ~54 % near 10 ms). The velocity survives
that — lag-1 autocorrelation ≈ +0.98 — but the second difference does not: the logged
`acc_joint_*` has a lag-1 autocorrelation of about −0.5, the signature of a sign-
alternating noise sequence. Any max/mean/std computed on that column measures the
logging jitter, not the arm.

This script therefore re-derives the acceleration: positions resampled on a uniform
100 Hz grid, gaps longer than GAP_S masked out, Savitzky-Golay differentiation
(polyorder 3, window WIN_S). Absolute values depend on the window and are only
comparable at a fixed window — the fixed/moving comparison is what the numbers are for.

Companion to velocity_analysis/ and manipulability_analysis/: same runs, same
exclusion rule, same alignment on the start of each approach leg.

Run from the results directory:  python3 acceleration_analysis/accel_over_time.py
"""
import importlib.util
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
sys.path.insert(0, RESULTS)
import leg_survival as LS  # noqa: E402
FIG, TAB = os.path.join(HERE, 'figures'), os.path.join(HERE, 'tables')

FIX, MOB, INK, MUT, RULE = '#E8734A', '#4C8BC9', '#141413', '#6E6C64', '#E2DFD5'
COND_COLOR = {'base fixe': FIX, 'base mobile': MOB}

FS = 100.0          # Hz, uniform resampling grid
DT = 1.0 / FS
WIN_S = 0.41        # Savitzky-Golay window (41 samples)
POLY = 3
GAP_S = 0.05        # original gaps longer than this are masked out of the statistics

LEGS = [('act_grasp:act_move_pregrasp', 'approche de saisie', 45),
        ('act_release:act_move_preplace', 'approche de dépose', 120)]


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
    spec = [('setup 1 — table vide', 'base fixe', E2.load, 'final_reuleaux_exp1'),
            ('setup 1 — table vide', 'base mobile', A.load, 'exp_1_resultat_finaux_heron'),
            ('setup 2 — table encombrée', 'base fixe', S2.load_fixed, 'exp2_base_fixe'),
            ('setup 2 — table encombrée', 'base mobile', S2.load_moving, 'exp_2_moving_base')]
    out = []
    for setup, cond, loader, folder in spec:
        d = loader(os.path.join(RESULTS, folder))
        keep = set(d['idx'].episode_id) & set(d['joints'].episode_id)
        out.append((setup, cond, d['joints'][d['joints'].episode_id.isin(keep)].copy()))
    return out


def resample_leg(e):
    """Uniform 100 Hz positions + a validity mask. Returns (t, Q, valid) or None."""
    e = e.sort_values('t_rel')
    t = e.t_rel.to_numpy(dtype=float)
    cols = [f'pos_joint_{i}' for i in range(1, 7)]
    P = e[cols].to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(P).all(axis=1)
    t, P = t[ok], P[ok]
    if len(t) < 200 or t[-1] - t[0] < 5.0:
        return None
    tu = np.arange(t[0], t[-1], DT)
    if len(tu) < int(WIN_S * FS) + 5:
        return None
    Q = np.column_stack([np.interp(tu, t, P[:, j]) for j in range(P.shape[1])])
    # mask the interpolated stretches that cover a real hole in the log
    valid = np.ones(len(tu), dtype=bool)
    gaps = np.nonzero(np.diff(t) > GAP_S)[0]
    for g in gaps:
        valid[(tu >= t[g]) & (tu <= t[g + 1])] = False
    return tu - tu[0], Q, valid


def derivatives(Q, win_s=WIN_S):
    win = int(round(win_s * FS)) | 1
    V = savgol_filter(Q, win, POLY, deriv=1, delta=DT, axis=0)
    Acc = savgol_filter(Q, win, POLY, deriv=2, delta=DT, axis=0)
    Jrk = savgol_filter(Q, win, POLY, deriv=3, delta=DT, axis=0)
    return V, Acc, Jrk


def sparc(speed, fs=FS, padlevel=4, fc=10.0, amp_th=0.05):
    """Spectral arc length of a speed profile (Balasubramanian et al., 2015).

    Dimensionless, amplitude- and duration-invariant; less negative = smoother.
    Unlike the acceleration magnitudes it does not depend on the Savitzky-Golay
    window, which makes it the robust half of the smoothness comparison.
    """
    if len(speed) < 16 or not np.any(speed > 0):
        return np.nan
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    mag = np.abs(np.fft.fft(speed, nfft))
    mag = mag / mag.max()
    cut = int(np.ceil(fc / (fs / nfft)))
    f, mag = f[:cut], mag[:cut]
    keep = np.nonzero(mag >= amp_th)[0]
    if len(keep) < 2:
        return np.nan
    f, mag = f[keep[0]:keep[-1] + 1], mag[keep[0]:keep[-1] + 1]
    df = np.diff(f) / (f[-1] - f[0])
    return float(-np.sum(np.sqrt(df ** 2 + np.diff(mag) ** 2)))


def leg_table(runs, win_s=WIN_S):
    rows = []
    for setup, cond, j in runs:
        for phase, label, _ in LEGS:
            for ep, e in j[j.phase == phase].groupby('episode_id'):
                r = resample_leg(e)
                if r is None:
                    continue
                t, Q, valid = r
                V, Acc, Jrk = derivatives(Q, win_s)
                edge = int(round(win_s * FS))          # drop the filter's edge effects
                m = valid.copy()
                m[:edge] = m[-edge:] = False
                if m.sum() < 100:
                    continue
                a_norm = np.linalg.norm(Acc[m], axis=1)
                v_norm = np.linalg.norm(V[m], axis=1)
                j_norm = np.linalg.norm(Jrk[m], axis=1)
                dur = float(t[m][-1] - t[m][0])
                # acceleration reversals: sign changes per joint per second, pooled
                sgn = np.sign(Acc[m])
                flips = int((np.diff(sgn, axis=0) != 0).sum())
                row = dict(setup=setup, condition=cond, leg=label, episode_id=ep,
                           duration_s=dur,
                           a_mean=float(a_norm.mean()),
                           a_p95=float(np.percentile(a_norm, 95)),
                           a_max=float(a_norm.max()),
                           jerk_mean=float(j_norm.mean()),
                           reversals_per_s=flips / dur / 6.0,
                           sparc=sparc(v_norm),
                           a_per_v=float(a_norm.mean() / max(v_norm.mean(), 1e-9)))
                for i in range(6):
                    row[f'a_mean_j{i + 1}'] = float(np.abs(Acc[m][:, i]).mean())
                rows.append(row)
    return pd.DataFrame(rows)


METRICS = [('a_mean', 'moyenne de ‖q̈‖ (rad/s²)'),
           ('a_p95', 'p95 de ‖q̈‖ (rad/s²)'),
           ('a_max', 'max de ‖q̈‖ (rad/s²)'),
           ('jerk_mean', 'moyenne de ‖q⃛‖ (rad/s³)'),
           ('reversals_per_s', 'inversions d\'accélération (1/s/axe)'),
           ('sparc', 'SPARC (douceur, moins négatif = plus doux)'),
           ('a_per_v', '‖q̈‖ / ‖q̇‖ moyen (1/s)'),
           ('duration_s', 'durée de la phase (s)')]


def summary(df):
    rows = []
    for (setup, leg), g in df.groupby(['setup', 'leg'], sort=False):
        f = g[g.condition == 'base fixe']
        m = g[g.condition == 'base mobile']
        for key, label in METRICS:
            a, b = f[key].dropna(), m[key].dropna()
            if len(a) < 3 or len(b) < 3:
                continue
            p = mannwhitneyu(a, b).pvalue
            rows.append(dict(setup=setup, leg=leg, metric=label, n_fixe=len(a), n_mobile=len(b),
                             fixe=a.median(), mobile=b.median(), p_mannwhitney=p))
    return pd.DataFrame(rows)


def noise_check(runs):
    """Lag-1 autocorrelation of the logged column vs the re-derived one."""
    rows = []
    for setup, cond, j in runs:
        raw, reb, n = [], [], 0
        for phase, label, _ in LEGS:
            for ep, e in j[j.phase == phase].groupby('episode_id'):
                r = resample_leg(e)
                if r is None:
                    continue
                a = e.sort_values('t_rel')[[f'acc_joint_{i}' for i in range(1, 7)]].to_numpy(float)
                a = a[np.isfinite(a).all(axis=1)]
                _, Acc, _ = derivatives(r[1])
                for X, acc in ((a, raw), (Acc, reb)):
                    x = X - X.mean(axis=0)
                    den = (x * x).sum(axis=0)
                    ok = den > 0
                    if ok.any():
                        acc.append(float(np.mean((x[1:] * x[:-1]).sum(axis=0)[ok] / den[ok])))
                n += 1
        rows.append(dict(setup=setup, condition=cond, n_legs=n,
                         ac1_logged=float(np.mean(raw)), ac1_rederived=float(np.mean(reb))))
    return pd.DataFrame(rows)


def fig_curves(runs, bin_s=1.0):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6), sharex='col')
    setups = ['setup 1 — table vide', 'setup 2 — table encombrée']
    for col, setup in enumerate(setups):
        for row, (phase, label, xmax) in enumerate(LEGS):
            ax = axes[row][col]
            t_end = 0.0
            for s, cond, j in runs:
                if s != setup:
                    continue
                series = []
                for ep, e in j[j.phase == phase].groupby('episode_id'):
                    r = resample_leg(e)
                    if r is None:
                        continue
                    t, Q, valid = r
                    _, Acc, _ = derivatives(Q)
                    edge = int(round(WIN_S * FS))
                    m = valid.copy()
                    m[:edge] = m[-edge:] = False
                    series.append((t[m], np.linalg.norm(Acc[m], axis=1)))
                if not series:
                    continue
                # per-leg mean in each bin, NaN once the leg has ended: no bin is dropped,
                # the summary is split solid / dashed on how many legs are left instead
                edges = np.arange(0, LS.T_GRID_MAX + bin_s, bin_s)
                mid = 0.5 * (edges[:-1] + edges[1:])
                arr = np.full((len(series), len(mid)), np.nan)
                for i, (tt, v) in enumerate(series):
                    for k in range(len(mid)):
                        m = (tt >= edges[k]) & (tt < edges[k + 1])
                        if m.any():
                            arr[i, k] = v[m].mean()
                _, te, _ = LS.median_over_legs(ax, mid, arr, COND_COLOR[cond], label=cond,
                                               linewidth=2)
                t_end = max(t_end, te)
            ax.set_xlim(0, t_end + 1)   # l'axe s'arrête avec la jambe la plus longue
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.set_title(f'{setup.split("—")[1].strip()} — {label}', fontsize=10, color=INK)
            if col == 0:
                ax.set_ylabel('‖q̈‖ (rad/s²)', fontsize=9)
            if row == 1:
                ax.set_xlabel('temps depuis le début de la phase (s)', fontsize=9)
    axes[0][0].legend(handles=LS.legend_handles([(c, COND_COLOR[c]) for c in COND_COLOR], INK),
                      frameon=False, fontsize=8)
    fig.suptitle('Accélération articulaire au fil des approches — médiane et intervalle interquartile',
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(os.path.join(FIG, 'accel_over_time.png'), dpi=150)
    plt.close(fig)


def fig_noise(runs):
    """One leg, logged column vs re-derived, to justify discarding the former."""
    setup, cond, j = runs[0]
    phase = LEGS[1][0]
    for ep, e in j[j.phase == phase].groupby('episode_id'):
        r = resample_leg(e)
        if r is None:
            continue
        t, Q, valid = r
        _, Acc, _ = derivatives(Q)
        e = e.sort_values('t_rel')
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
        w = (e.t_rel - e.t_rel.iloc[0] < 20)
        axes[0].plot(e.t_rel[w] - e.t_rel.iloc[0], e.acc_joint_1[w], color=MUT, lw=0.5)
        axes[0].set_title('colonne journalisée `acc_joint_1` (Δv/Δt)', fontsize=10)
        axes[1].plot(t[t < 20], Acc[t < 20, 0], color=INK, lw=1.4)
        axes[1].set_title(f'redérivée (Savitzky-Golay {WIN_S:.2f} s sur 100 Hz uniforme)', fontsize=10)
        for ax in axes:
            ax.set_xlabel('temps depuis le début de la phase (s)', fontsize=9)
            ax.set_ylabel('rad/s²', fontsize=9)
            ax.grid(alpha=0.25, linewidth=0.6)
        fig.suptitle('Même signal, 20 premières secondes d\'une approche de dépose',
                     fontsize=12, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(os.path.join(FIG, 'accel_noise.png'), dpi=150)
        plt.close(fig)
        return


def fig_joints(df):
    cols = [f'a_mean_j{i}' for i in range(1, 7)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), sharey=True)
    for ax, (leg, g) in zip(axes, df.groupby('leg', sort=False)):
        x = np.arange(6)
        for k, (cond, gg) in enumerate(g.groupby('condition', sort=False)):
            ax.bar(x + (k - 0.5) * 0.36, [gg[c].median() for c in cols], 0.34,
                   color=COND_COLOR[cond], label=cond)
        ax.set_xticks(x, [f'J{i}' for i in range(1, 7)])
        ax.set_title(leg, fontsize=10, color=INK)
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel('|q̈| moyen (rad/s²)', fontsize=9)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Contribution de chaque axe à l\'accélération (les deux setups confondus)',
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(FIG, 'accel_by_joint.png'), dpi=150)
    plt.close(fig)


def fig_wrist(runs, bin_s=1.0, xmax=45):
    """|q5| over the pick approach: the wrist crosses its singularity every time."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), sharey=True)
    setups = ['setup 1 — table vide', 'setup 2 — table encombrée']
    phase = LEGS[0][0]
    for ax, setup in zip(axes, setups):
        for s, cond, j in runs:
            if s != setup:
                continue
            series = []
            for ep, e in j[j.phase == phase].groupby('episode_id'):
                e = e.sort_values('t_rel')
                t = e.t_rel.to_numpy(float)
                series.append((t - t[0], np.degrees(np.abs(e.pos_joint_5.to_numpy(float)))))
            edges = np.arange(0, xmax + bin_s, bin_s)
            mid, med, lo, hi = [], [], [], []
            for k in range(len(edges) - 1):
                vals = [v[(tt >= edges[k]) & (tt < edges[k + 1])].mean()
                        for tt, v in series if ((tt >= edges[k]) & (tt < edges[k + 1])).any()]
                if len(vals) < 5:
                    continue
                mid.append(0.5 * (edges[k] + edges[k + 1]))
                med.append(np.median(vals)); lo.append(np.percentile(vals, 25)); hi.append(np.percentile(vals, 75))
            c = COND_COLOR[cond]
            ax.fill_between(mid, lo, hi, color=c, alpha=0.16, linewidth=0)
            ax.plot(mid, med, color=c, lw=2, label=cond)
        ax.axhline(0, color=INK, lw=1)
        ax.axhspan(0, 10, color=MUT, alpha=0.12, linewidth=0)
        ax.set_xlim(0, xmax)
        ax.set_xlabel('temps depuis le début de l\'approche de saisie (s)', fontsize=9)
        ax.set_title(setup.split('—')[1].strip(), fontsize=10, color=INK)
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel('|q₅| (degrés)', fontsize=9)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Le poignet passe par sa singularité (q₅ ≈ 0) à chaque approche de saisie',
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(FIG, 'wrist_singularity.png'), dpi=150)
    plt.close(fig)


def write_md(df, path):
    cols = list(df.columns)
    with open(path, 'w') as fh:
        fh.write('| ' + ' | '.join(cols) + ' |\n')
        fh.write('|' + '---|' * len(cols) + '\n')
        for _, r in df.iterrows():
            cells = [f'{r[c]:.3f}' if isinstance(r[c], float) else str(r[c]) for c in cols]
            fh.write('| ' + ' | '.join(cells) + ' |\n')


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    runs = load_runs()

    nc = noise_check(runs)
    nc.to_csv(os.path.join(TAB, 'accel_noise_check.csv'), index=False)
    print('\n== lag-1 autocorrelation (logged vs re-derived)')
    print(nc.to_string(index=False))

    df = leg_table(runs)
    df.to_csv(os.path.join(TAB, 'accel_legs.csv'), index=False)
    s = summary(df)
    s.to_csv(os.path.join(TAB, 'accel_summary.csv'), index=False)
    write_md(s, os.path.join(TAB, 'accel_summary.md'))
    print('\n== per-leg summary')
    print(s.to_string(index=False))

    # window sensitivity: does the fixed/moving ordering survive?
    rows = []
    for w in (0.21, 0.41, 0.61):
        d = leg_table(runs, w)
        for (setup, leg), g in d.groupby(['setup', 'leg'], sort=False):
            f, m = g[g.condition == 'base fixe'], g[g.condition == 'base mobile']
            rows.append(dict(window_s=w, setup=setup, leg=leg,
                             a_mean_fixe=f.a_mean.median(), a_mean_mobile=m.a_mean.median(),
                             ratio=f.a_mean.median() / m.a_mean.median(),
                             p=mannwhitneyu(f.a_mean, m.a_mean).pvalue))
    ws = pd.DataFrame(rows)
    ws.to_csv(os.path.join(TAB, 'accel_window_sensitivity.csv'), index=False)
    print('\n== window sensitivity')
    print(ws.to_string(index=False))

    fig_curves(runs)
    fig_noise(runs)
    fig_joints(df)
    fig_wrist(runs)
    print('\nfigures ->', FIG)


if __name__ == '__main__':
    main()
