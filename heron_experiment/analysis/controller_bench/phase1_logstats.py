#!/usr/bin/env python3
"""Phase 1 — statistiques passives sur les control_debug.csv de runs passés.

Lecture seule. Aucun node ROS, aucun /cmd_vel. Répond aux questions 1.8
(chronologie réelle) et 2.7 (distribution de |grad Q|) du cahier des charges.

    python3 analysis/controller_bench/phase1_logstats.py              # tous les runs de results/
    python3 analysis/controller_bench/phase1_logstats.py 'setup_2/*'  # <expérience>/<run_id>, globs acceptés
"""

import csv
import glob
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import runs  # noqa: E402

ROOT = str(runs.paths.results_dir())


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(rows, key):
    out = []
    for r in rows:
        v = r.get(key, '')
        if v not in ('', None):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def q(vals, p):
    if not vals:
        return float('nan')
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[k]


def describe(name, vals, unit=''):
    if not vals:
        print(f'  {name}: (aucune valeur)')
        return
    print(f'  {name}: n={len(vals)} moy={st.mean(vals):.4f} med={q(vals,0.5):.4f} '
          f'p05={q(vals,0.05):.4f} p95={q(vals,0.95):.4f} '
          f'min={min(vals):.4f} max={max(vals):.4f} {unit}')


def main():
    all_cycle, all_dt, all_gmag, all_qc, all_v = [], [], [], [], []
    patterns = sys.argv[1:] or ['*/*']
    for path in sorted(p for pattern in patterns
                       for p in glob.glob(os.path.join(ROOT, pattern, 'raw', 'control_debug.csv'))):
        rows = load(path)
        inferred = [r for r in rows if not r.get('skip_reason')]
        skipped = [r for r in rows if r.get('skip_reason')]
        reasons = {}
        for r in skipped:
            reasons[r['skip_reason']] = reasons.get(r['skip_reason'], 0) + 1

        cycle = num(inferred, 'cycle_s')
        dt = [d for d in num(inferred, 'dt_since_last_s') if 0 < d < 30]
        gmag = num(inferred, 'grad_mag')
        qc = num(inferred, 'q_center')
        vx = num(inferred, 'vx_cmd')
        vy = num(inferred, 'vy_cmd')
        v = [math.hypot(a, b) for a, b in zip(vx, vy)]
        vcap = num(inferred, 'v_cap')
        enabled = sum(1 for r in inferred if r.get('enabled') in ('True', 'true', '1'))
        moving = sum(1 for s in v if s > 1e-9)

        print(f'\n=== {os.path.relpath(os.path.dirname(os.path.dirname(path)), ROOT)} ===')
        print(f'  lignes={len(rows)} inférées={len(inferred)} skips={len(skipped)} {reasons}')
        describe('cycle_s (inférence+scoring)', cycle, 's')
        describe('dt_since_last_s (période réelle boucle)', dt, 's')
        describe('|grad Q|', gmag)
        describe('Q_center', qc)
        describe('|v| commandé', v, 'm/s')
        describe('v_cap', vcap, 'm/s')
        print(f'  enabled={enabled}/{len(inferred)}  cycles avec |v|>0 : {moving}')
        if v:
            sat = sum(1 for s, c in zip(v, vcap) if c > 0 and s >= c - 1e-9)
            print(f'  cycles en saturation (|v| == v_cap) : {sat}/{len(v)}')

        all_cycle += cycle
        all_dt += dt
        all_gmag += gmag
        all_qc += qc
        all_v += v

    print('\n=== TOUS RUNS CONFONDUS ===')
    describe('cycle_s', all_cycle, 's')
    describe('dt_since_last_s', all_dt, 's')
    describe('|grad Q|', all_gmag)
    describe('Q_center', all_qc)
    describe('|v|', all_v, 'm/s')

    # Seuils dérivés de la config déployée
    k, vmax, vmin = 0.2, 0.02, 0.005
    if all_gmag:
        below = sum(1 for g in all_gmag if k * g < vmin)
        satg = sum(1 for g in all_gmag if k * g > vmax)
        print(f'\n  Avec k={k}, base_speed={vmax}, min_speed={vmin} :')
        print(f'    |g| < {vmin/k:.4f} (zone morte / convergence latchée) : '
              f'{below}/{len(all_gmag)} = {100*below/len(all_gmag):.1f} %')
        print(f'    |g| > {vmax/k:.4f} (saturation) : '
              f'{satg}/{len(all_gmag)} = {100*satg/len(all_gmag):.1f} %')
        print(f'    régime proportionnel strict : '
              f'{len(all_gmag)-below-satg}/{len(all_gmag)} = '
              f'{100*(len(all_gmag)-below-satg)/len(all_gmag):.1f} %')


if __name__ == '__main__':
    main()
