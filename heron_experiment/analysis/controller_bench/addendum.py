#!/usr/bin/env python3
"""Phase 2 — complément : ce que le résidu mesure vraiment, et l'erreur
réelle de direction du gradient estimé.

Deux constats sont apparus en calculant la matrice de corrélation :
  (a) le résidu des moindres carrés est, par construction, orthogonal aux
      colonnes x et y du design — il ne se propage donc PAS dans (gx, gy).
      Le résidu n'est pas une erreur de gradient.
  (b) l'ajustement quadratique redonne exactement le même (gx, gy) que le plan
      sur un réseau 3x3 symétrique.
On quantifie ici (b) sur tous les voisinages, et on mesure la vraie erreur de
direction : angle entre le gradient estimé à δ=0.1 m et le gradient local réel
lu sur la carte dense à 2 cm.

Post-traitement pur, aucun modèle, aucun node ROS.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rm_harness import TABLES, DELTA, Harness  # noqa: E402

STEP = 0.02
K = int(round(DELTA / STEP))
STENCIL = [(1, -1), (1, 0), (1, 1), (0, -1), (0, 0), (0, 1), (-1, -1), (-1, 0), (-1, 1)]


def describe(name, v, unit=''):
    v = np.asarray([x for x in v if np.isfinite(x)])
    print(f'  {name}: n={len(v)} moy={v.mean():.4f} med={np.median(v):.4f} '
          f'p05={np.percentile(v,5):.4f} p95={np.percentile(v,95):.4f} '
          f'max={v.max():.4f} {unit}')
    return v


def main():
    d = np.load(f'{TABLES}/dense_q.npz')
    q, axis = d['q'], d['axis']
    n = len(axis)

    dgx = np.full((n, n), np.nan)
    dgy = np.full((n, n), np.nan)
    dgx[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / (2 * STEP)
    dgy[1:-1, :] = (q[2:, :] - q[:-2, :]) / (2 * STEP)

    dgrad, dplan, dquad, ang_true, ratio = [], [], [], [], []
    for j in range(K, n - K):
        for i in range(K, n - K):
            s = np.array([q[j - dj * K, i + di * K] for dj, di in STENCIL])
            gx, gy, _, _ = Harness.fit_plane(s)
            coef, _, _ = Harness.fit_quadratic(s)
            dgrad.append(max(abs(gx - coef[1]), abs(gy - coef[2])))
            tx, ty = dgx[j, i], dgy[j, i]
            tn, en = np.hypot(tx, ty), np.hypot(gx, gy)
            if tn > 1e-9 and en > 1e-9:
                ca = (tx * gx + ty * gy) / (tn * en)
                ang_true.append(np.degrees(np.arccos(np.clip(ca, -1, 1))))
                ratio.append(en / tn)
            dplan.append(np.hypot(gx - tx, gy - ty))
            dquad.append(np.hypot(coef[1] - tx, coef[2] - ty))

    print('=== (b) Le terme quadratique change-t-il le gradient estime ? ===')
    dg = np.array(dgrad)
    print(f'  max |g_plan - g_quadratique| sur {len(dg)} voisinages : {dg.max():.3e}')
    print('  -> sur un reseau 3x3 symetrique, les colonnes x,y sont orthogonales')
    print('     aux colonnes 1, x^2, y^2, xy : ajouter la courbure au modele ne')
    print('     modifie PAS (gx, gy). Le plan et la quadratique donnent le meme')
    print('     gradient ; seule la direction de Newton differe.')

    print('\n=== (a) Le residu se propage-t-il dans le gradient ? ===')
    offs = Harness.grid_offsets()
    x = np.array([o[0] for o in offs])
    print('  Sum(x_i * r_i) est nul a la precision machine par construction des')
    print('  moindres carres (residu orthogonal au design). Verification sur un')
    print('  voisinage :')
    s = np.array([q[n//2 - dj * K, n//2 + di * K] for dj, di in STENCIL])
    gx, gy, c, _ = Harness.fit_plane(s)
    y = np.array([o[1] for o in offs])
    r = s - (c + gx * x + gy * y)
    print(f'    Sum(x_i r_i) = {float(x @ r):+.3e}   Sum(y_i r_i) = {float(y @ r):+.3e}')
    print('  -> le residu mesure la part du signal que le plan ne represente pas,')
    print('     pas une erreur sur (gx, gy).')

    print('\n=== Vraie erreur de direction du gradient estime ===')
    print('  (gradient plan a delta=0.1 m  vs  gradient local reel a 2 cm)')
    a = describe('angle [deg]', ang_true)
    for thr in (5, 10, 20, 45, 90):
        print(f'    > {thr:2d} deg : {100*(a > thr).mean():.1f} %')
    describe('|g_estime| / |g_reel|', ratio)
    describe('||g_plan - g_reel|| [1/m]', dplan)
    describe('||g_quad - g_reel|| [1/m]', dquad)
    print('\n  |g| reel a 2 cm :')
    describe('  ||g_reel||', np.hypot(dgx[K:n-K, K:n-K], dgy[K:n-K, K:n-K]).ravel())


if __name__ == '__main__':
    main()
