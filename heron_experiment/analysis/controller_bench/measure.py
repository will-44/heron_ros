#!/usr/bin/env python3
"""Phase 2 — mesures passives, robot immobile.

Aucun node ROS n'est créé, rclpy n'est jamais importé : il n'existe aucun
publisher /cmd_vel dans ce processus. Toutes les poses candidates sont
virtuelles (translation de la carte d'obstacles).

Sections : 2.1 bruit temporel, 2.2 bruit spatial intra-cycle / résidu,
2.3 voxels entrants, 2.4 carte dense de Q, 2.5 validité du modèle local,
2.6 régime de saturation, 2.7 amplitude du gradient.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rm_harness import (TABLES, BASE_SPEED, CONTROL_GAIN, DELTA, MIN_SPEED,  # noqa: E402
                        Harness)

DENSE_HALF = 0.40     # demi-étendue de la carte dense, en m
DENSE_STEP = 0.02     # pas de la carte dense (= 1 voxel)
N_REPEAT = 120        # répétitions pour sigma temporel
report = {}


def sect(title):
    print(f'\n{"=" * 72}\n{title}\n{"=" * 72}')


# ── 2.1 Bruit de prédiction (temporel) ───────────────────────────────────────

def s21(h, base_mask):
    sect('2.1  Bruit de prediction sigma — N repetitions, meme pose, scene fixe')
    res = {}
    for dist in (0.0, 0.5, 1.0):
        offs = [(dist, 0.0)] * N_REPEAT
        t0 = time.time()
        qs = h.q_at(offs, base_mask=base_mask)
        dt = time.time() - t0
        # Paquet de 1 vs paquet de 9. Avec l'ancien TRTModel ce n'étaient pas les
        # mêmes noyaux (CUDA Graph vs execute_async_v3) ; heron.inference passe
        # chaque Scene seule, donc ils doivent coïncider exactement.
        q1 = h.q_at([(dist, 0.0)], base_mask=base_mask, chunk=1)[0]
        res[dist] = {
            'n': int(len(qs)), 'mean': float(qs.mean()), 'std': float(qs.std(ddof=1)),
            'min': float(qs.min()), 'max': float(qs.max()),
            'ptp': float(np.ptp(qs)), 'n_unique': int(len(np.unique(qs))),
            'q_batch1_cudagraph': float(q1),
            'delta_batch1_vs_batchN': float(q1 - qs[0]),
            'ms_per_inference': dt / len(qs) * 1000,
        }
        r = res[dist]
        print(f'  d={dist:.1f} m : Q={r["mean"]:.6f}  sigma={r["std"]:.3e}  '
              f'etendue={r["ptp"]:.3e}  valeurs distinctes={r["n_unique"]}/{r["n"]}')
        print(f'            batch=1 (CUDA Graph) Q={q1:.6f}  '
              f'ecart vs batch=N : {r["delta_batch1_vs_batchN"]:+.3e}')
        print(f'            {r["ms_per_inference"]:.1f} ms / inference')
    report['2.1'] = res
    return res


# ── 2.3 Voxels entrants ──────────────────────────────────────────────────────

def s23(h, base_mask):
    sect('2.3  Voxels entrants (non observes) lors de la translation')
    scene = h.scene[0, 0]
    total = scene.numel()
    rows = []
    for dx, dy in h.grid_offsets():
        di = -int(round(dx / h.res))
        dj = -int(round(dy / h.res))
        if di == 0 and dj == 0:
            continue
        ent = h.entering_mask(di, dj)[0, 0]
        mask = h.mask_for(base_mask, di, dj).mask
        n_ent = int(ent.sum())
        n_mask = int(mask.sum())
        n_ent_in_mask = int((ent & mask).sum())
        # Contenu observe perdu/gagne : obstacles avant vs apres translation
        occ = scene > 0
        kept = h.shifted_scene(dx, dy)[0][0, 0] > 0
        rows.append({
            'dx': dx, 'dy': dy,
            'frac_volume': n_ent / total,
            'n_mask': n_mask,
            'frac_mask': (n_ent_in_mask / n_mask) if n_mask else float('nan'),
            'n_occ_before': int(occ.sum()), 'n_occ_after': int(kept.sum()),
            'd_occ': int(kept.sum()) - int(occ.sum()),
        })
    print(f'  {"offset":>14} {"% volume entrant":>18} {"% masque entrant":>18} '
          f'{"d_obstacles":>13}')
    for r in rows:
        print(f'  ({r["dx"]:+.2f},{r["dy"]:+.2f}) {100*r["frac_volume"]:>17.3f}% '
              f'{100*r["frac_mask"]:>17.3f}% {r["d_occ"]:>13d}')

    # Asymetrie +x / -x et +y / -y sur le contenu observe
    def get(dx, dy):
        return next(r for r in rows if abs(r['dx'] - dx) < 1e-9 and abs(r['dy'] - dy) < 1e-9)
    asym = {
        'x': get(DELTA, 0.0)['d_occ'] - get(-DELTA, 0.0)['d_occ'],
        'y': get(0.0, DELTA)['d_occ'] - get(0.0, -DELTA)['d_occ'],
    }
    print(f'\n  Asymetrie du contenu observe  +x vs -x : {asym["x"]:+d} voxels occupes')
    print(f'  Asymetrie du contenu observe  +y vs -y : {asym["y"]:+d} voxels occupes')
    print(f'  Fraction du masque non observee : '
          f'{100*max(r["frac_mask"] for r in rows):.3f} % (max sur les 8 poses)')
    report['2.3'] = {'rows': rows, 'asym': asym}
    return rows


# ── 2.4 Carte dense de Q ─────────────────────────────────────────────────────

def s24(h, base_mask):
    sect('2.4  Carte dense de Q')
    axis = np.round(np.arange(-DENSE_HALF, DENSE_HALF + 1e-9, DENSE_STEP), 6)
    n = len(axis)
    offs = [(float(x), float(y)) for y in axis for x in axis]
    print(f'  grille {n}x{n} pas {DENSE_STEP} m, etendue +/-{DENSE_HALF} m '
          f'-> {len(offs)} inferences...')
    t0 = time.time()
    q = h.q_at(offs, base_mask=base_mask).reshape(n, n)   # q[iy, ix]
    dt = time.time() - t0
    print(f'  fait en {dt:.1f} s ({dt/len(offs)*1000:.1f} ms/inference)')
    print(f'  Q : min={q.min():.4f} max={q.max():.4f} moy={q.mean():.4f}')
    iy, ix = np.unravel_index(np.argmax(q), q.shape)
    print(f'  maximum en (dx={axis[ix]:+.2f}, dy={axis[iy]:+.2f}) m, Q={q.max():.4f}')
    print(f'  Q au centre (0,0) = {q[n//2, n//2]:.4f}')

    # Unimodalite : nombre de maxima locaux stricts (voisinage 8)
    loc = 0
    for j in range(1, n - 1):
        for i in range(1, n - 1):
            w = q[j-1:j+2, i-1:i+2]
            if q[j, i] >= w.max() and q[j, i] > np.median(w):
                loc += 1
    # Discontinuites : plus grand saut entre voxels adjacents
    jmp = max(np.abs(np.diff(q, axis=0)).max(), np.abs(np.diff(q, axis=1)).max())
    print(f'  maxima locaux (voisinage 8) : {loc}')
    print(f'  plus grand saut entre cases adjacentes (2 cm) : {jmp:.4f}')
    report['2.4'] = {
        'n': n, 'step': DENSE_STEP, 'half': DENSE_HALF,
        'q_min': float(q.min()), 'q_max': float(q.max()),
        'q_center': float(q[n//2, n//2]),
        'argmax_dx': float(axis[ix]), 'argmax_dy': float(axis[iy]),
        'n_local_maxima': int(loc), 'max_jump_2cm': float(jmp),
        'seconds': dt,
    }
    np.savez_compressed(f'{TABLES}/dense_q.npz', q=q, axis=axis)
    return axis, q


# ── 2.2 / 2.5 / 2.6 / 2.7 sur la carte dense ─────────────────────────────────

def s22_25_26_27(axis, q):
    sect('2.2 / 2.5 / 2.6 / 2.7  Residu, modele local, hessienne, saturation')
    n = len(axis)
    step = DENSE_STEP
    k = int(round(DELTA / step))     # 5 voxels = delta

    offs = Harness.grid_offsets()
    ox = np.array([o[0] for o in offs])
    oy = np.array([o[1] for o in offs])
    A_lin = np.column_stack([np.ones(9), ox, oy])
    A_quad = np.column_stack([np.ones(9), ox, oy, ox**2, oy**2, ox*oy])

    recs = []
    for j in range(k, n - k):
        for i in range(k, n - k):
            # les 9 Q du stencil 3x3 a delta=0.1 m centre sur (i, j)
            s = np.array([q[j - dj_ * k, i + di_ * k]
                          for dj_, di_ in [(1, -1), (1, 0), (1, 1),
                                           (0, -1), (0, 0), (0, 1),
                                           (-1, -1), (-1, 0), (-1, 1)]])
            gx, gy, c, res_lin = Harness.fit_plane(s)
            coef, res_quad, H = Harness.fit_quadratic(s)
            g = np.array([gx, gy])
            gmag = float(np.hypot(gx, gy))

            ev = np.linalg.eigvalsh(H)
            aniso = (abs(ev).max() / abs(ev).min()) if abs(ev).min() > 1e-12 else np.inf
            try:
                newton = -np.linalg.solve(H, g)
                nn = float(np.linalg.norm(newton))
                cosang = (float(g @ newton) / (gmag * nn)) if gmag * nn > 1e-15 else np.nan
                ang = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
            except np.linalg.LinAlgError:
                nn, ang = np.inf, np.nan

            v = CONTROL_GAIN * gmag
            regime = ('deadband' if min(v, BASE_SPEED) < MIN_SPEED
                      else ('saturated' if v > BASE_SPEED else 'proportional'))
            recs.append(dict(
                dx=float(axis[i]), dy=float(axis[j]), q=float(q[j, i]),
                gmag=gmag, res_lin=res_lin, res_quad=res_quad,
                lam1=float(ev[0]), lam2=float(ev[1]), aniso=float(aniso),
                neg_def=bool(ev[0] < 0 and ev[1] < 0),
                newton_norm=nn, angle=ang, regime=regime))

    print(f'  {len(recs)} voisinages 3x3 (delta={DELTA} m) evalues sur la carte dense')

    def dist(name, vals, fmt='.4f'):
        v = np.array([x for x in vals if np.isfinite(x)])
        if not len(v):
            print(f'  {name}: aucune valeur finie')
            return {}
        d = dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                 p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                 min=float(v.min()), max=float(v.max()))
        print(f'  {name}: n={d["n"]} moy={d["mean"]:{fmt}} med={d["median"]:{fmt}} '
              f'p05={d["p05"]:{fmt}} p95={d["p95"]:{fmt}} max={d["max"]:{fmt}}')
        return d

    out = {'n_neighbourhoods': len(recs)}
    print('\n  --- 2.2  Residu de l\'ajustement (sigma temporel mesure = 0, cf. 2.1) ---')
    out['res_lin'] = dist('residu PLAN      (9-3 ddl)', [r['res_lin'] for r in recs])
    out['res_quad'] = dist('residu QUADRATIQUE (9-6 ddl)', [r['res_quad'] for r in recs])
    rl = np.array([r['res_lin'] for r in recs])
    rq = np.array([r['res_quad'] for r in recs])
    ratio = float(np.median(rl[rq > 1e-12] / rq[rq > 1e-12])) if (rq > 1e-12).any() else np.nan
    print(f'  rapport median residu_plan / residu_quad = {ratio:.2f}')
    out['res_ratio_median'] = ratio

    print('\n  --- 2.5  Hessienne et direction de Newton ---')
    out['lam1'] = dist('lambda_1 (le plus negatif)', [r['lam1'] for r in recs])
    out['lam2'] = dist('lambda_2', [r['lam2'] for r in recs])
    out['aniso'] = dist('anisotropie |lmax|/|lmin|', [r['aniso'] for r in recs], '.2f')
    nd = sum(r['neg_def'] for r in recs)
    print(f'  H definie negative (les deux lambda < 0) : {nd}/{len(recs)} = '
          f'{100*nd/len(recs):.1f} %')
    out['frac_neg_def'] = nd / len(recs)
    out['angle'] = dist('angle(gradient, Newton) [deg]', [r['angle'] for r in recs], '.2f')
    ang = np.array([r['angle'] for r in recs if np.isfinite(r['angle'])])
    for thr in (5, 15, 30, 60):
        print(f'    angle > {thr:2d} deg : {100*(ang > thr).mean():.1f} %')
    out['angle_gt'] = {str(t): float((ang > t).mean()) for t in (5, 15, 30, 60)}

    print('\n  --- 2.6  Regime de saturation ---')
    nn = np.array([r['newton_norm'] for r in recs if np.isfinite(r['newton_norm'])])
    r_step = BASE_SPEED * 1.0   # v_max * cycle_period = 0.02 m
    out['newton_norm'] = dist('||H^-1 g|| [m]', nn)
    print(f'  fraction avec ||H^-1 g|| > r={r_step} m : {100*(nn > r_step).mean():.1f} %')
    out['frac_newton_gt_r'] = float((nn > r_step).mean())
    reg = {}
    for r in recs:
        reg[r['regime']] = reg.get(r['regime'], 0) + 1
    for kk, vv in sorted(reg.items()):
        print(f'  regime {kk:14s} : {vv:5d}/{len(recs)} = {100*vv/len(recs):.1f} %')
    out['regimes'] = reg

    print('\n  --- 2.7  Amplitude du gradient ---')
    out['gmag'] = dist('||grad Q|| [1/m]', [r['gmag'] for r in recs])
    g = np.array([r['gmag'] for r in recs])
    print(f'  ||g|| sous le seuil de zone morte ({MIN_SPEED/CONTROL_GAIN:.3f}) : '
          f'{100*(g < MIN_SPEED/CONTROL_GAIN).mean():.1f} %')
    print(f'  ||g|| au-dela du seuil de saturation ({BASE_SPEED/CONTROL_GAIN:.3f}) : '
          f'{100*(g > BASE_SPEED/CONTROL_GAIN).mean():.1f} %')

    # Erreur angulaire attendue : sigma_g = sigma/(h*sqrt(6)). sigma mesure = 0
    # pour le modele ; on tabule donc l'effet d'un sigma HYPOTHETIQUE d'origine
    # perceptive, ce qui donne le seuil de zone morte a viser.
    print('\n  --- 2.7b  Erreur angulaire attendue pour un sigma hypothetique ---')
    print('  (sigma_modele mesure = 0 ; ce tableau dimensionne la zone morte face '
          'a un bruit de scene)')
    tab = {}
    print(f'  {"sigma":>8} {"sigma_g":>10} | ' +
          ' '.join(f'{f"|g|={gg}":>12}' for gg in (0.05, 0.1, 0.25, 0.5, 1.0)))
    for sig in (1e-4, 1e-3, 5e-3, 1e-2, 3e-2):
        sg = sig / (DELTA * np.sqrt(6))
        cells = []
        for gg in (0.05, 0.1, 0.25, 0.5, 1.0):
            cells.append(f'{np.degrees(np.arctan(sg/gg)):>11.2f}d')
        tab[str(sig)] = sg
        print(f'  {sig:>8.0e} {sg:>10.4f} | ' + ' '.join(cells))
    out['sigma_g_table'] = tab

    report['2.2_2.5_2.6_2.7'] = out
    np.savez_compressed(f'{TABLES}/neighbourhoods.npz',
                        **{k: np.array([r[k] for r in recs])
                           for k in ('dx', 'dy', 'q', 'gmag', 'res_lin', 'res_quad',
                                     'lam1', 'lam2', 'aniso', 'newton_norm', 'angle')})
    return recs


def main():
    import os
    os.makedirs(TABLES, exist_ok=True)
    h = Harness(group='0')
    base_mask = h.base_mask()
    print(f'[harness] masque region : {base_mask.count()} voxels '
          f'(r={0.10} m attendu ~ {4/3*np.pi*(0.10/h.res)**3:.0f})')

    s21(h, base_mask)
    s23(h, base_mask)
    axis, q = s24(h, base_mask)
    s22_25_26_27(axis, q)

    with open(f'{TABLES}/report.json', 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f'\nEcrit : {TABLES}/report.json, dense_q.npz, neighbourhoods.npz')


if __name__ == '__main__':
    main()
