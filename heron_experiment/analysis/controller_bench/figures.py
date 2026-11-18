#!/usr/bin/env python3
"""Phase 2 — figures et matrice de corrélation des résidus.

Consomme out/dense_q.npz produit par measure.py. Aucun modèle chargé, aucun
node ROS, aucun /cmd_vel : pur post-traitement numpy/matplotlib.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rm_harness import (FIGURES, TABLES, BASE_SPEED, CONTROL_GAIN, DELTA,  # noqa: E402
                        MIN_SPEED, Harness)

STEP = 0.02
K = int(round(DELTA / STEP))
STENCIL = [(1, -1), (1, 0), (1, 1), (0, -1), (0, 0), (0, 1), (-1, -1), (-1, 0), (-1, 1)]


def stencil_scores(q, i, j):
    return np.array([q[j - dj * K, i + di * K] for dj, di in STENCIL])


def main():
    os.makedirs(FIGURES, exist_ok=True)
    d = np.load(f'{TABLES}/dense_q.npz')
    q, axis = d['q'], d['axis']
    n = len(axis)
    c = n // 2

    # ── 2.2  Matrice de correlation des residus entre les 9 poses ────────────
    R = []
    for j in range(K, n - K):
        for i in range(K, n - K):
            s = stencil_scores(q, i, j)
            gx, gy, cc, _ = Harness.fit_plane(s)
            offs = Harness.grid_offsets()
            pred = cc + gx * np.array([o[0] for o in offs]) + gy * np.array(
                [o[1] for o in offs])
            R.append(s - pred)
    R = np.array(R)                      # (n_voisinages, 9)
    C = np.corrcoef(R.T)
    rms = R.std(axis=0, ddof=1)

    print('=== 2.2  Residus par pose, sur %d voisinages ===' % len(R))
    lab = ['(-d,+d)', '( 0,+d)', '(+d,+d)', '(-d, 0)', '( 0, 0)',
           '(+d, 0)', '(-d,-d)', '( 0,-d)', '(+d,-d)']
    print('  pose      biais moyen   ecart-type')
    for k in range(9):
        print(f'  {lab[k]}  {R[:, k].mean():+11.5f}  {rms[k]:11.5f}')
    print(f'\n  biais moyen global (doit etre ~0 par construction) : {R.mean():+.2e}')

    print('\n  Matrice de correlation des residus (9x9) :')
    print('        ' + ' '.join(f'{i:>6d}' for i in range(9)))
    for i in range(9):
        print(f'  {i} ' + ' '.join(f'{C[i, j]:>6.2f}' for j in range(9)))

    off = C[~np.eye(9, dtype=bool)]
    print(f'\n  correlation hors-diagonale : moy={off.mean():+.3f} '
          f'min={off.min():+.3f} max={off.max():+.3f}')
    # Part de la variance des residus qui se propage dans le gradient :
    # gx = sum(x_i r_i)/sum(x_i^2), var = x^T C_cov x / (sum x^2)^2
    offs = Harness.grid_offsets()
    x = np.array([o[0] for o in offs])
    y = np.array([o[1] for o in offs])
    Cov = np.cov(R.T)
    var_gx = float(x @ Cov @ x) / (x @ x) ** 2
    var_gy = float(y @ Cov @ y) / (y @ y) ** 2
    var_indep = float(np.trace(Cov) / 9 * (x @ x)) / (x @ x) ** 2
    print(f'\n  ecart-type induit sur gx : {np.sqrt(var_gx):.4f} 1/m')
    print(f'  ecart-type induit sur gy : {np.sqrt(var_gy):.4f} 1/m')
    print(f'  (si les residus etaient independants : {np.sqrt(var_indep):.4f} 1/m)')
    print(f'  facteur de reduction par correlation : '
          f'{np.sqrt(var_indep / var_gx):.2f}x sur gx')

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xticklabels(lab, rotation=90, fontsize=7)
    ax.set_yticklabels(lab, fontsize=7)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center', fontsize=5.5)
    ax.set_title("Corrélation des résidus d'ajustement plan\nentre les 9 poses "
                 f'({len(R)} voisinages)', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(f'{FIGURES}/fig_2_2_correlation.png', dpi=160)

    # ── 2.4  Carte dense + stencil + gradient + pas commande ─────────────────
    s = stencil_scores(q, c, c)
    gx, gy, _, res_lin = Harness.fit_plane(s)
    coef, res_q, H = Harness.fit_quadratic(s)
    vx, vy, regime = Harness.command(gx, gy)
    ev, evec = np.linalg.eigh(H)
    newton = -np.linalg.solve(H, np.array([gx, gy]))
    ang = np.degrees(np.arccos(np.clip(
        np.array([gx, gy]) @ newton / (np.hypot(gx, gy) * np.linalg.norm(newton)),
        -1, 1)))

    print('\n=== 2.4 / 2.5  Au centre de la carte dense ===')
    print(f'  Q = {q[c, c]:.4f}')
    print(f'  grad (plan)      = ({gx:+.4f}, {gy:+.4f})  |g|={np.hypot(gx,gy):.4f}  '
          f'residu={res_lin:.4f}')
    print(f'  grad (quadrat.)  = ({coef[1]:+.4f}, {coef[2]:+.4f})  residu={res_q:.4f}')
    print(f'  H = [[{H[0,0]:+.2f} {H[0,1]:+.2f}], [{H[1,0]:+.2f} {H[1,1]:+.2f}]]  '
          f'lambda = {ev[0]:+.2f}, {ev[1]:+.2f}')
    print(f'  ||H^-1 g|| = {np.linalg.norm(newton):.4f} m   '
          f'angle(g, Newton) = {ang:.1f} deg')
    print(f'  commande = ({vx:+.4f}, {vy:+.4f}) m/s  regime={regime}  '
          f'pas/cycle = {np.hypot(vx,vy)*1.0*100:.1f} cm')

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ext = [axis[0] - STEP/2, axis[-1] + STEP/2, axis[0] - STEP/2, axis[-1] + STEP/2]
    im = ax.imshow(q, origin='lower', extent=ext, cmap='viridis', aspect='equal')
    ax.contour(axis, axis, q, levels=14, colors='w', linewidths=0.5, alpha=0.65)
    ox = np.array([o[0] for o in Harness.grid_offsets()])
    oy = np.array([o[1] for o in Harness.grid_offsets()])
    ax.plot(ox, oy, 'o', ms=7, mfc='none', mec='r', mew=1.6,
            label='9 poses échantillonnées (δ=0.1 m)')
    g = np.array([gx, gy]); gn = g / np.hypot(gx, gy)
    ax.arrow(0, 0, gn[0]*0.18, gn[1]*0.18, width=0.006, color='r',
             length_includes_head=True, label=f'ĝ  (|g|={np.hypot(gx,gy):.2f})')
    ax.arrow(0, 0, vx*1.0, vy*1.0, width=0.006, color='k',
             length_includes_head=True,
             label=f'pas commandé ({np.hypot(vx,vy)*100:.0f} cm, {regime})')
    nn = newton / max(np.linalg.norm(newton), 1e-9)
    ax.arrow(0, 0, nn[0]*0.18, nn[1]*0.18, width=0.004, color='orange',
             length_includes_head=True, label=f'Newton −H⁻¹g ({ang:.0f}° de ĝ)')
    jy, jx = np.unravel_index(np.argmax(q), q.shape)
    ax.plot(axis[jx], axis[jy], 'w*', ms=14, mec='k', label='max de la fenêtre')
    ax.set_xlabel('déplacement x de la base [m]')
    ax.set_ylabel('déplacement y de la base [m]')
    ax.set_title('Q(dx, dy) — carte dense 41×41 à 2 cm, scène statique', fontsize=10)
    ax.legend(fontsize=7, loc='lower left', framealpha=0.85)
    fig.colorbar(im, ax=ax, shrink=0.85, label='Q')
    fig.tight_layout()
    fig.savefig(f'{FIGURES}/fig_2_4_carte_dense.png', dpi=160)

    # ── Distributions 2.2 / 2.5 / 2.7 ────────────────────────────────────────
    nb = np.load(f'{TABLES}/neighbourhoods.npz')
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.6))
    a = axs[0, 0]
    a.hist(nb['res_lin'], bins=50, alpha=0.75, label='plan (3 coeff.)')
    a.hist(nb['res_quad'], bins=50, alpha=0.75, label='quadratique (6 coeff.)')
    a.axvline(0, color='k', lw=1, ls='--', label='σ temporel mesuré = 0')
    a.set_xlabel("résidu d'ajustement"); a.set_ylabel('voisinages')
    a.set_title('2.2  Résidu : plan vs quadratique', fontsize=9); a.legend(fontsize=7)

    a = axs[0, 1]
    a.hist(nb['angle'][np.isfinite(nb['angle'])], bins=60)
    a.axvline(5, color='r', ls='--', lw=1.2, label='seuil 5°')
    a.set_xlabel('angle(g, −H⁻¹g) [°]'); a.set_ylabel('voisinages')
    a.set_title('2.5  Écart gradient / Newton', fontsize=9); a.legend(fontsize=7)

    a = axs[1, 0]
    a.hist(np.clip(nb['newton_norm'], 0, 2), bins=60)
    a.axvline(BASE_SPEED, color='r', ls='--', lw=1.2,
              label=f'r = v_max·T = {BASE_SPEED} m')
    a.set_xlabel('‖H⁻¹g‖ [m] (tronqué à 2 m)'); a.set_ylabel('voisinages')
    a.set_title('2.6  Longueur du pas de Newton', fontsize=9); a.legend(fontsize=7)

    a = axs[1, 1]
    a.hist(nb['gmag'], bins=60)
    a.axvline(MIN_SPEED / CONTROL_GAIN, color='g', ls='--', lw=1.2,
              label=f'zone morte |g|={MIN_SPEED/CONTROL_GAIN:.3f}')
    a.axvline(BASE_SPEED / CONTROL_GAIN, color='r', ls='--', lw=1.2,
              label=f'saturation |g|={BASE_SPEED/CONTROL_GAIN:.2f}')
    a.set_xlabel('‖∇Q‖ [1/m]'); a.set_ylabel('voisinages')
    a.set_title('2.7  Amplitude du gradient', fontsize=9); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{FIGURES}/fig_2_distributions.png', dpi=160)

    print(f'\nFigures : {FIGURES}/fig_2_2_correlation.png, fig_2_4_carte_dense.png, '
          f'fig_2_distributions.png')


if __name__ == '__main__':
    main()
