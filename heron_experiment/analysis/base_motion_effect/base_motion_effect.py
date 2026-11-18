#!/usr/bin/env python3
"""Est-ce que le bras est gêné quand la base bouge ?

Deux figures demandées : la vitesse de déplacement de la base contre (a) la qualité Q et
(b) l'accélération maximale du bras.

Point de méthode à lire avant les figures : la commande de base est en tout-ou-rien. Sur
les deux runs à base mobile, `hypot(vx, vy)` ne prend que **deux** valeurs, 0 et
0,020 m/s ; agrégée par seconde, elle vaut exactement 0 dans 58-64 % des secondes et
exactement 0,020 dans 32-38 %, moins de 5 % des secondes tombant entre les deux. Un nuage
« Q en fonction de la vitesse » est donc deux traits verticaux, pas une courbe. Les
figures montrent le nuage réel (demandé) *et* la comparaison qui, elle, a un sens :
seconde-à-seconde, base à l'arrêt contre base en mouvement.

Le piège à éviter : pendant une approche, le bras accélère surtout au début et la base est
commandée surtout au début. Une corrélation brute confond donc l'effet de la base avec le
moment dans la phase. On la contrôle de deux façons : un appariement à seconde-de-phase
égale (chaque seconde « base en mouvement » comparée aux secondes « base à l'arrêt » du
même instant de la même phase du même run) et une corrélation partielle de Spearman.

    python3 base_motion_effect/base_motion_effect.py     # depuis results/

Écrit figures/base_speed_vs_q.png, figures/base_speed_vs_accel.png,
tables/bins.csv, tables/summary.csv|md.
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: I100,I201,E402
from scipy.signal import savgol_filter  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import runs  # noqa: E402
RESULTS = str(runs.paths.results_dir())
OUT = runs.study_dir('base_motion_effect')
FIG, TAB = os.path.join(OUT, 'figures'), os.path.join(OUT, 'tables')

FIX, MOB, INK, INK2, MUT, RULE = '#E8734A', '#4C8BC9', '#141413', '#45443F', '#6E6C64', '#E2DFD5'
PHASE_COLOR = {'approche de saisie': '#2C5B8C', 'approche de dépose': '#8C5A2C'}

# dérivation identique à acceleration/ : la colonne acc_joint_* journalisée est
# du bruit de différentiation (autocorrélation à lag 1 de -0,58), voir ce README §0.
FS, POLY, WIN_S = 100.0, 3, 0.41
DT = 1.0 / FS
BIN_S = 1.0
V_CMD = 0.020          # la seule vitesse non nulle commandée, m/s
MOVING_FRAC = 0.5      # une seconde est « en mouvement » si la base bouge plus de la moitié

LEGS = [('act_grasp:act_move_pregrasp', 'approche de saisie'),
        ('act_release:act_move_preplace', 'approche de dépose')]


def load_runs():
    """Les quatre runs, chacun avec sa propre classification des épisodes."""
    by = {(c.setup, c.base): c for c in runs.conditions()}
    order = [('setup_1', 'moving'), ('setup_2', 'moving'), ('setup_1', 'fixed'), ('setup_2', 'fixed')]
    spec = [(by[k].setup_label, by[k].base_label, by[k].load, by[k].run) for k in order]
    out = []
    for setup, cond, loader, folder in spec:
        d = loader(runs.raw(folder))
        keep = set(d['idx'].episode_id)          # les épisodes écartés n'entrent nulle part
        for k in ('joints', 'cmd_vel'):
            d[k] = d[k][d[k].episode_id.isin(keep)]
        out.append((setup, cond, d))
    return out


def accel_series(e):
    """‖q̇‖ et ‖q̈‖ rééchantillonnés à 100 Hz sur l'horloge absolue. (ts, a, v) ou None."""
    e = e.sort_values('timestamp')
    t = e.timestamp.to_numpy(dtype=float)
    P = e[[f'pos_joint_{i}' for i in range(1, 7)]].to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(P).all(axis=1)
    t, P = t[ok], P[ok]
    if len(t) < 200 or t[-1] - t[0] < 5.0:
        return None
    tu = np.arange(t[0], t[-1], DT)
    win = int(round(WIN_S * FS)) | 1
    if len(tu) < win + 5:
        return None
    Q = np.column_stack([np.interp(tu, t, P[:, j]) for j in range(P.shape[1])])
    Vel = savgol_filter(Q, win, POLY, deriv=1, delta=DT, axis=0)
    Acc = savgol_filter(Q, win, POLY, deriv=2, delta=DT, axis=0)
    a = np.linalg.norm(Acc, axis=1)
    vq = np.linalg.norm(Vel, axis=1)
    edge = win                       # bords du filtre écartés
    return tu[edge:-edge], a[edge:-edge], vq[edge:-edge]


def bin_table(runs):
    """Une ligne par seconde d'approche : vitesse de la base, Q, accélération du bras."""
    rows = []
    for setup, cond, d in runs:
        ctrl, cv, jt = d['ctrl'], d['cmd_vel'], d['joints']
        for phase, leg in LEGS:
            for ep, e in jt[jt.phase == phase].groupby('episode_id'):
                r = accel_series(e)
                if r is None:
                    continue
                ts, a, vq = r
                c = ctrl[(ctrl.episode_id == ep) & (ctrl.phase == phase)
                         & (ctrl.region_size >= 3)].sort_values('timestamp')
                v = cv[(cv.episode_id == ep) & (cv.phase == phase)].sort_values('timestamp')
                if len(c) < 2 or len(v) < 2:
                    continue
                t0 = ts[0]
                speed = np.hypot(v.vx.to_numpy(float), v.vy.to_numpy(float))
                edges = np.arange(0, ts[-1] - t0 + BIN_S, BIN_S)
                for k in range(len(edges) - 1):
                    lo, hi = t0 + edges[k], t0 + edges[k + 1]
                    ma = (ts >= lo) & (ts < hi)
                    mv = (v.timestamp.to_numpy() >= lo) & (v.timestamp.to_numpy() < hi)
                    if not ma.any() or not mv.any():
                        continue
                    vb = float(speed[mv].mean())
                    rows.append(dict(
                        setup=setup, cond=cond, leg=leg, episode_id=ep, t_bin=float(edges[k]),
                        v_base=vb, duty=min(1.0, vb / V_CMD), moving=vb >= MOVING_FRAC * V_CMD,
                        a_max=float(a[ma].max()), a_mean=float(a[ma].mean()),
                        v_arm=float(vq[ma].mean()),
                        q_goal=float(np.interp(0.5 * (lo + hi), c.timestamp, c.ws_mean_0)),
                        q_center=float(np.interp(0.5 * (lo + hi), c.timestamp, c.q_center))))
    return pd.DataFrame(rows)


def partial_spearman(x, y, z):
    """Spearman de x et y une fois le rang de z retiré des deux (corrélation partielle)."""
    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    rx, ry, rz = (v - v.mean() for v in (rx, ry, rz))
    bx = rx - rz * (rx @ rz) / (rz @ rz)
    by = ry - rz * (ry @ rz) / (rz @ rz)
    return spearmanr(bx, by)


def matched_pairs(df, key):
    """Apparié à seconde-de-phase égale : médiane bougeant / médiane à l'arrêt, même instant.

    Dans un run et une phase donnés, chaque seconde t est un couple (médiane de `key` sur
    les épisodes dont la base bougeait à t, médiane sur ceux où elle était à l'arrêt). Le
    moment dans la phase est ainsi tenu constant, ce qui est le seul confondant sérieux.
    """
    out = []
    for (setup, leg), g in df[df.cond == 'base mobile'].groupby(['setup', 'leg']):
        for t, h in g.groupby('t_bin'):
            mv, st = h[h.moving][key], h[~h.moving][key]
            if len(mv) >= 3 and len(st) >= 3:
                out.append(dict(setup=setup, leg=leg, t_bin=t, n_moving=len(mv), n_still=len(st),
                                moving=mv.median(), still=st.median()))
    return pd.DataFrame(out)


V_ARM_MOVING = 0.02      # rad/s : seuil « le bras bouge vraiment » (cf. velocity_analysis)


def summary(df, arm_moving_only=False):
    rows = []
    if arm_moving_only:
        df = df[df.v_arm >= V_ARM_MOVING]
    mob = df[df.cond == 'base mobile']
    for (setup, leg), g in mob.groupby(['setup', 'leg']):
        mv, st = g[g.moving], g[~g.moving]
        for key, lab in (('q_goal', 'Q sphère du but'), ('a_max', 'accélération max ‖q̈‖ (rad/s²)')):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p = mannwhitneyu(mv[key], st[key]).pvalue if len(mv) and len(st) else np.nan
                rho = spearmanr(g.v_base, g[key])
                prho = partial_spearman(g.v_base.to_numpy(), g[key].to_numpy(), g.t_bin.to_numpy())
            pr = matched_pairs(g, key)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pw = wilcoxon(pr.moving, pr.still).pvalue if len(pr) >= 6 else np.nan
            fx = df[(df.cond == 'base fixe') & (df.setup == setup) & (df.leg == leg)][key]
            rows.append(dict(
                setup=setup, leg=leg, mesure=lab, n_sec_arret=len(st), n_sec_mouvement=len(mv),
                mediane_arret=st[key].median(), mediane_mouvement=mv[key].median(),
                p_brut=p, rho_brut=rho.statistic, p_rho=rho.pvalue,
                rho_partiel=prho.statistic, p_rho_partiel=prho.pvalue,
                n_sec_appariees=len(pr),
                appariee_arret=pr.still.median() if len(pr) else np.nan,
                appariee_mouvement=pr.moving.median() if len(pr) else np.nan,
                p_appariee=pw, reference_base_fixe=fx.median() if len(fx) else np.nan))
    return pd.DataFrame(rows)


def write_md(df, path):
    cols = list(df.columns)
    with open(path, 'w') as fh:
        fh.write('| ' + ' | '.join(cols) + ' |\n')
        fh.write('|' + '---|' * len(cols) + '\n')
        for _, r in df.iterrows():
            cells = [f'{r[c]:.3f}' if isinstance(r[c], float) else str(r[c]) for c in cols]
            fh.write('| ' + ' | '.join(cells) + ' |\n')


TRANS_W = 3     # secondes de part et d'autre d'un démarrage / arrêt de la base


def transitions(df, key='a_max', w=TRANS_W):
    """Événementiel : `key` dans les w s avant / après chaque démarrage et arrêt de base.

    C'est le test causal le plus propre disponible ici. La comparaison globale « base qui
    bouge » contre « base à l'arrêt » confond l'effet de la base avec le moment dans la
    phase (la base est commandée pendant le grand transit du bras et s'arrête quand le
    bras se pose). Autour d'une transition, en revanche, le mouvement du bras est presque
    le même de part et d'autre : seule la base change d'état. On exige que les deux
    fenêtres soient entières et qu'aucune autre transition ne s'y trouve.
    """
    out = []
    for (setup, leg, ep), g in df[df.cond == 'base mobile'].groupby(['setup', 'leg', 'episode_id']):
        g = g.sort_values('t_bin')
        t = g.t_bin.to_numpy()
        if len(t) < 2 * w + 2 or np.any(np.diff(t) != BIN_S):
            continue
        mv = g.moving.to_numpy().astype(int)
        v = g[key].to_numpy()
        for i in range(w, len(t) - w):
            if mv[i] == mv[i - 1]:
                continue
            before, after = mv[i - w:i], mv[i:i + w]
            if not (before == before[0]).all() or not (after == after[0]).all():
                continue    # une autre transition dans la fenêtre
            out.append(dict(setup=setup, leg=leg, episode_id=ep, t_bin=t[i],
                            evenement='démarrage' if mv[i] else 'arrêt',
                            avant=float(np.max(v[i - w:i])), apres=float(np.max(v[i:i + w]))))
    return pd.DataFrame(out)


def fig_transitions(tr, fname='base_motion_transitions.png'):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), facecolor='white')
    for ax, ev in zip(axes, ('démarrage', 'arrêt')):
        h = tr[tr.evenement == ev]
        for leg, mk in zip([lab for _, lab in LEGS], ('o', '^')):
            k = h[h.leg == leg]
            ax.plot(k.avant, k.apres, mk, color=PHASE_COLOR[leg], alpha=0.6, markersize=5,
                    markeredgewidth=0, label=leg)
        lim = (0.01, 6)
        ax.plot(lim, lim, '-', color=MUT, linewidth=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel('max ‖q̈‖ dans les %d s avant (rad/s²)' % TRANS_W, fontsize=9.5, color=MUT)
        ax.set_ylabel('dans les %d s après (rad/s²)' % TRANS_W, fontsize=9.5, color=MUT)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pw = wilcoxon(h.apres, h.avant).pvalue if len(h) >= 6 else np.nan
        ax.set_title('%s de la base — n = %d, médiane %.2f → %.2f, p = %.4f'
                     % (ev, len(h), h.avant.median(), h.apres.median(), pw),
                     fontsize=10, color=INK, loc='left')
        ax.legend(frameon=False, fontsize=8, loc='lower right')
        _style(ax)
    fig.suptitle('Accélération maximale du bras autour des démarrages et arrêts de la base',
                 fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, fname), dpi=150, facecolor='white')
    plt.close(fig)
    print('  wrote', os.path.relpath(os.path.join(FIG, fname), RESULTS))


N_SPEED_BINS = 6


def matched_arm_speed(df, key='a_max', nbins=N_SPEED_BINS):
    """`key` à vitesse de bras égale : déciles de ‖q̇‖, base à l'arrêt contre en mouvement.

    Contrôle bien plus solide que l'appariement temporel : si la base ne faisait que
    coïncider avec les grands mouvements du bras, alors à ‖q̇‖ égal les deux états
    devraient donner la même accélération.
    """
    rows = []
    g0 = df[(df.cond == 'base mobile') & (df.v_arm >= V_ARM_MOVING)]
    for leg, g in g0.groupby('leg'):
        q = pd.qcut(g.v_arm, nbins, labels=False)
        for k in range(nbins):
            h = g[q == k]
            a, b = h[~h.moving][key], h[h.moving][key]
            if len(a) < 10 or len(b) < 10:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p = mannwhitneyu(b, a).pvalue
            rows.append(dict(leg=leg, bin=k, v_arm_median=h.v_arm.median(),
                             n_arret=len(a), n_mouvement=len(b),
                             mediane_arret=a.median(), mediane_mouvement=b.median(),
                             ratio=b.median() / a.median(), p=p))
    return pd.DataFrame(rows)


def _style(ax):
    ax.grid(True, color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=MUT, labelsize=9)


def fig_scatter(df, key, ylabel, title, fname, ylim=None, logy=False, third='matched'):
    """Nuage demandé (gauche) + les deux lectures qui contrôlent le moment dans la phase."""
    mob = df[df.cond == 'base mobile']
    setups = sorted(mob.setup.unique())
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), facecolor='white')
    rng = np.random.default_rng(0)
    for r, (phase, leg) in enumerate(LEGS):
        g0 = mob[mob.leg == leg]
        # --- colonne 1 : le nuage brut, vitesse de la base en abscisse ---------------
        ax = axes[r][0]
        for setup, mk in zip(setups, ('o', '^')):
            h = g0[g0.setup == setup]
            x = h.v_base + rng.uniform(-8e-4, 8e-4, len(h))      # gigue pour voir la densité
            ax.plot(x, h[key], mk, color=PHASE_COLOR[leg], alpha=0.16, markersize=3,
                    markeredgewidth=0, label=setup.split('—')[1].strip())
        for xv in (0.0, V_CMD):
            h = g0[np.isclose(g0.v_base, xv, atol=0.25 * V_CMD)]
            if len(h):
                ax.plot([xv - 0.0018, xv + 0.0018], [h[key].median()] * 2, '-', color=INK, lw=2.4,
                        zorder=5)
        ax.set_xlim(-0.003, 0.023)
        ax.set_xticks([0.0, 0.005, 0.010, 0.015, 0.020])
        ax.set_xlabel('vitesse de la base sur la seconde (m/s)', fontsize=9.5, color=MUT)
        ax.set_ylabel(f'{leg}\n{ylabel}', fontsize=10, color=INK)
        if r == 0:
            ax.set_title('nuage brut (une seconde d’approche = un point)', fontsize=10,
                         color=INK, loc='left')
        # --- colonne 2 : distributions à l'arrêt / en mouvement -----------------------
        ax = axes[r][1]
        data, labs, cols = [], [], []
        for setup in setups:
            h = g0[g0.setup == setup]
            for mo, lab, col in ((False, 'arrêt', MUT), (True, 'mouvement', MOB)):
                data.append(h[h.moving == mo][key].to_numpy())
                labs.append('%s\n%s' % (setup.split('—')[1].strip().replace('table ', ''), lab))
                cols.append(col)
        bp = ax.boxplot(data, labels=labs, widths=0.55, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, linewidth=2))
        for patch, col in zip(bp['boxes'], cols):
            patch.set_facecolor(col)
            patch.set_alpha(0.28)
            patch.set_edgecolor(col)
        for i, (vals, col) in enumerate(zip(data, cols), start=1):
            if len(vals):
                ax.plot(i + rng.uniform(-0.16, 0.16, len(vals)), vals, 'o', color=col, alpha=0.10,
                        markersize=2.4, markeredgewidth=0, zorder=1)
        ax.tick_params(axis='x', labelsize=8)
        if r == 0:
            ax.set_title('les deux états réels de la base', fontsize=10, color=INK, loc='left')
        # --- colonne 3 : le contrôle ---------------------------------------------------
        ax = axes[r][2]
        if third == 'vs_arm':
            # à vitesse de bras égale : si la base ne faisait que coïncider avec les grands
            # mouvements du bras, les deux nuages se superposeraient
            h0 = g0[g0.v_arm >= V_ARM_MOVING]
            for mo, lab, col in ((False, 'base à l’arrêt', MUT), (True, 'base en mouvement', MOB)):
                h = h0[h0.moving == mo]
                ax.plot(h.v_arm, h[key], 'o', color=col, alpha=0.13, markersize=2.6,
                        markeredgewidth=0)
                if len(h) > 30:
                    q = pd.qcut(h0.v_arm, N_SPEED_BINS, labels=False)
                    xs, ys = [], []
                    for k in range(N_SPEED_BINS):
                        kk = h0[(q == k) & (h0.moving == mo)]
                        if len(kk) >= 10:
                            xs.append(kk.v_arm.median())
                            ys.append(kk[key].median())
                    ax.plot(xs, ys, '-o', color=col, linewidth=2.2, markersize=5, label=lab,
                            zorder=5)
            ax.set_xscale('log')
            ax.set_xlabel('vitesse du bras ‖q̇‖ sur la seconde (rad/s)', fontsize=9.5, color=MUT)
            ax.set_ylabel(ylabel, fontsize=9.5, color=MUT)
            ax.legend(frameon=False, fontsize=8, loc='upper left')
            if r == 0:
                ax.set_title('à vitesse de bras égale', fontsize=10, color=INK, loc='left')
        else:
            for setup, mk in zip(setups, ('o', '^')):
                pr = matched_pairs(mob[mob.setup == setup], key)
                pr = pr[pr.leg == leg]
                if not len(pr):
                    continue
                ax.plot(pr.still, pr.moving, mk, color=PHASE_COLOR[leg], alpha=0.65, markersize=5,
                        markeredgewidth=0, label=setup.split('—')[1].strip())
            lim = ylim if ylim else axes[r][0].get_ylim()
            ax.plot(lim, lim, '-', color=MUT, linewidth=1)
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            ax.set_xlabel('même seconde, base à l’arrêt', fontsize=9.5, color=MUT)
            ax.set_ylabel('base en mouvement', fontsize=9.5, color=MUT)
            ax.legend(frameon=False, fontsize=8, loc='lower right')
            if r == 0:
                ax.set_title('apparié à seconde-de-phase égale', fontsize=10, color=INK, loc='left')
        for c_, ax in enumerate(axes[r]):
            _style(ax)
            if ylim and not (third == 'matched' and c_ == 2):
                ax.set_ylim(*ylim)
            if logy:
                ax.set_yscale('log')
    axes[0][0].legend(handles=[Line2D([], [], marker=mk, linestyle='none', color=INK2,
                                      label=s.split('—')[1].strip())
                               for s, mk in zip(setups, ('o', '^'))]
                      + [Line2D([], [], color=INK, lw=2.4, label='médiane de l’état')],
                      frameon=False, fontsize=8, loc='best')
    fig.suptitle(title, fontsize=12.5, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, fname), dpi=150, facecolor='white')
    plt.close(fig)
    print('  wrote', os.path.relpath(os.path.join(FIG, fname), RESULTS))


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    runs = load_runs()
    df = bin_table(runs)
    df.to_csv(os.path.join(TAB, 'bins.csv'), index=False)
    mob = df[df.cond == 'base mobile']
    print('secondes d’approche : %d à base mobile (%d à l’arrêt, %d en mouvement, %d entre deux),'
          ' %d à base fixe'
          % (len(mob), (mob.v_base < 0.02 * V_CMD).sum(), (mob.duty > 0.98).sum(),
             ((mob.duty > 0.02) & (mob.duty < 0.98)).sum(), (df.cond == 'base fixe').sum()))
    for tag, only in (('summary', False), ('summary_arm_moving', True)):
        s = summary(df, arm_moving_only=only)
        s.to_csv(os.path.join(TAB, f'{tag}.csv'), index=False)
        write_md(s.round(4), os.path.join(TAB, f'{tag}.md'))
        print('\n== %s' % ('toutes les secondes' if not only
                            else 'seulement les secondes où le bras bouge (‖q̇‖ ≥ %.2f rad/s)'
                            % V_ARM_MOVING))
        print(s.round(4).to_string(index=False))
    n_tot = len(df[df.cond == 'base mobile'])
    n_arm = len(df[(df.cond == 'base mobile') & (df.v_arm >= V_ARM_MOVING)])
    print('\nsecondes à base mobile où le bras bouge : %d / %d (%.0f %%)'
          % (n_arm, n_tot, 100 * n_arm / n_tot))
    tr = transitions(df)
    tr.to_csv(os.path.join(TAB, 'transitions.csv'), index=False)
    print('\n== autour des transitions de la base (%d s de part et d’autre)' % TRANS_W)
    for ev, h in tr.groupby('evenement'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pw = wilcoxon(h.apres, h.avant).pvalue
        print('  %-10s n = %3d   max ‖q̈‖ %.3f -> %.3f   p = %.4f   hausse dans %.0f %% des cas'
              % (ev, len(h), h.avant.median(), h.apres.median(), pw, 100 * (h.apres > h.avant).mean()))
    fig_transitions(tr)
    fig_scatter(df, 'q_goal', 'Q sphère du but',
                'Vitesse de la base et qualité Q — la base ne bouge qu’à 0 ou 0,020 m/s',
                'base_speed_vs_q.png', ylim=(0.2, 1.02))
    fig_scatter(df, 'a_max', 'max ‖q̈‖ sur la seconde (rad/s²)',
                'Vitesse de la base et accélération maximale du bras — le bras est-il gêné ?',
                'base_speed_vs_accel.png', ylim=(0.008, 6.0), logy=True, third='vs_arm')
    ms = matched_arm_speed(df)
    ms.to_csv(os.path.join(TAB, 'matched_arm_speed.csv'), index=False)
    write_md(ms.round(4), os.path.join(TAB, 'matched_arm_speed.md'))
    print('\n== max ‖q̈‖ à vitesse de bras égale')
    print(ms.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
