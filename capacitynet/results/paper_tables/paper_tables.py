# -*- coding: utf-8 -*-
"""The two paper tables (task performance, controller cost) for the four conditions.

One script so that every number in the paper comes from one place, with one exclusion
rule and one explicit definition per row. Run from the results directory:

    python3 paper_tables/paper_tables.py

Writes `tables/perf.csv`, `tables/cost.csv`, `tables/accel_pool.csv` and
`tables/tables.tex` (the two LaTeX tables, filled).

Definitions, all of them deliberate:

* **Conditions.** setup 1 = empty table (`final_reuleaux_exp1` fixed,
  `exp_1_resultat_finaux_heron` moving), setup 2 = cluttered table
  (`exp2_base_fixe` fixed, `exp_2_moving_base` moving). Each run is loaded by its own
  loader, which carries its own episode classification.
* **Exclusion.** `outcome_valid()`: an excluded episode enters no statistic at all.
  N is therefore the number of *valid* runs, and it differs per condition.
* **Success rate** = successes / valid runs.
* **Durations** are `mean ± std` over **successful** runs: the whole cycle from
  `episodes_index.csv`, the two approach phases from `phase_summary.csv`.
* **R_i at the end of the approach** is the goal-sphere reachability as the recorder
  measured it, averaged over the **last 3 s** of the phase (`samples.q_goal`, fresh
  goals only), then `mean ± std` over successful runs. This is the value the arm
  actually ends on, not the controller's own last cycle.
* **Base-target distance** is `hypot(goal_x, goal_y)` in the arm frame, at the first
  sample of the phase and averaged over its last 3 s, meaned over successful runs.
  Note this is the mean of the per-run ranges; the comparison figures instead plot the
  range of the *median goal position*, which is not the same number.
* **MPC / gradient cycle times** are pooled over every logged cycle of every valid
  run, successful or not: a cost is paid whatever the outcome.
* **Joint acceleration** is re-derived from the logged positions (uniform 100 Hz grid,
  Savitzky-Golay, polyorder 3, window 0.41 s) because the logged `acc_joint_*` column
  is differentiation noise -- see `../acceleration_analysis/README.md`. Pooled over
  every sample of every motion phase (approaches, transfers and both homings) of every
  valid run; idle stretches are excluded. Absolute values depend on the window, the
  fixed/moving comparison does not.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(RESULTS, 'acceleration_analysis'))
TAB = os.path.join(HERE, 'tables')

PREGRASP = 'act_grasp:act_move_pregrasp'
PREPLACE = 'act_release:act_move_preplace'
MOVE_PHASES = ['act_home_enter', PREGRASP, 'act_grasp:act_move_grasp',
               PREPLACE, 'act_release:act_move_place', 'act_home_exit']
TAIL_S = 3.0        # the "end of the approach" window
GOAL_AGE_S = 1.0    # a sample whose goal is older than this is a stale echo
BUDGET_MS = 400.0


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def load_conditions():
    """The four (setup, condition, data) triples, each with its own loader."""
    A = _mod('A', os.path.join(RESULTS, 'exp_1_resultat_finaux_heron/analysis/analyze_exp1.py'))
    E2 = _mod('E2', os.path.join(RESULTS, 'final_reuleaux_exp1/analysis/analyze_exp2.py'))
    _mod('R', os.path.join(RESULTS, 'exp_2_moving_base/analysis/reach_analysis.py'))
    S2 = _mod('S2', os.path.join(RESULTS, 'exp_2_moving_base/analysis/analyze_setup2.py'))
    R = sys.modules['R']
    spec = [('setup 1', 'fixed', E2.load, A, 'final_reuleaux_exp1'),
            ('setup 1', 'moving', A.load, A, 'exp_1_resultat_finaux_heron'),
            ('setup 2', 'fixed', S2.load_fixed, R, 'exp2_base_fixe'),
            ('setup 2', 'moving', S2.load_moving, R, 'exp_2_moving_base')]
    out = []
    for setup, cond, loader, mod, folder in spec:
        d = loader(os.path.join(RESULTS, folder))
        # The run's own exclusion rule, not a re-implementation of it: an excluded
        # episode (infrastructure, operator, or a grasp/release failure that the base
        # controller had no part in) enters no statistic anywhere.
        d['valid'] = mod.outcome_valid(d['idx'])
        d['success'] = set(d['idx'][d['idx'].cls == 'success'].episode_id) \
            & set(d['valid'].episode_id)
        # The controller's own per-leg record, for the objective it actually optimises.
        d['legs'] = mod.build_legs(d, mod.integrate_cmd_vel(d['cmd_vel']))
        out.append((setup, cond, d))
    return out


def ms(v):
    """mean, std, median, n of a series, NaNs dropped."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(mean=np.nan, std=np.nan, median=np.nan, n=0)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                median=float(np.median(v)), n=int(len(v)))


def phase_durations(d, phase):
    """Per-successful-run duration of one phase (s)."""
    p = d['phases']
    p = p[(p.phase == phase) & p.episode_id.isin(d['success'])]
    return p.groupby('episode_id').duration_s.sum()


def approach_end(d, phase):
    """Per-successful-run (R_i, range at leg start, range at leg end) of one approach."""
    s = d['samples']
    s = s[(s.phase == phase) & s.episode_id.isin(d['success']) & (s.goal_age_s < GOAL_AGE_S)]
    rows = []
    for ep, g in s.groupby('episode_id'):
        g = g.sort_values('t_rel')
        tail = g[g.t_rel >= g.t_rel.max() - TAIL_S]
        rows.append(dict(episode_id=ep,
                         q_end=float(tail.q_goal.mean()),
                         range_start=float(np.hypot(g.goal_x.iloc[0], g.goal_y.iloc[0])),
                         range_end=float(np.hypot(tail.goal_x.mean(), tail.goal_y.mean()))))
    return pd.DataFrame(rows)


def accel_pool(d, accel_mod):
    """Every re-derived |q̈| sample of every motion phase of every valid run."""
    J = d['joints']
    J = J[J.episode_id.isin(set(d['valid'].episode_id)) & J.phase.isin(MOVE_PHASES)]
    chunks, n_legs = [], 0
    for _, g in J.groupby(['episode_id', 'phase']):
        r = accel_mod.resample_leg(g)
        if r is None:
            continue
        _, Q, valid = r
        _, Acc, _ = accel_mod.derivatives(Q)
        chunks.append(np.linalg.norm(Acc, axis=1)[valid])
        n_legs += 1
    a = np.concatenate(chunks) if chunks else np.array([])
    return a, n_legs


def build(conditions, accel_mod):
    perf, cost, pool = [], [], []
    for setup, cond, d in conditions:
        key = dict(setup=setup, condition=cond)
        n_valid = len(d['valid'])
        n_succ = len(d['success'])
        perf.append(dict(**key, metric='runs recorded', value=len(d['idx'])))
        perf.append(dict(**key, metric='runs excluded', value=len(d['idx']) - n_valid))
        perf.append(dict(**key, metric='N (valid runs)', value=n_valid))
        perf.append(dict(**key, metric='successes', value=n_succ))
        perf.append(dict(**key, metric='success rate [%]', value=100.0 * n_succ / max(n_valid, 1)))

        tot = d['idx'][d['idx'].episode_id.isin(d['success'])].duration_s
        for name, v in (('total run time [s]', tot),
                        ('pick approach [s]', phase_durations(d, PREGRASP)),
                        ('place approach [s]', phase_durations(d, PREPLACE))):
            st = ms(v)
            perf.append(dict(**key, metric=name, **st))

        for name, phase in (('pick', PREGRASP), ('place', PREPLACE)):
            e = approach_end(d, phase)
            perf.append(dict(**key, metric=f'{name} R_i at end of approach', **ms(e.q_end)))
            perf.append(dict(**key, metric=f'{name} base-target dist. start [m]',
                             **ms(e.range_start)))
            perf.append(dict(**key, metric=f'{name} base-target dist. end [m]',
                             **ms(e.range_end)))

        # The controller objective Q (region mean over the three spheres) at the last
        # commanded cycle of the leg -- a different instant and a different quantity
        # from R_i above: end of the *base command*, and all three spheres, not the
        # goal sphere alone. Over every valid leg, which is the controller's own unit.
        for name, phase in (('pick', PREGRASP), ('place', PREPLACE)):
            L = d['legs'][d['legs'].phase == phase]
            perf.append(dict(**key, metric=f'{name} Q region at leg start', **ms(L.q_start)))
            perf.append(dict(**key, metric=f'{name} Q region at end of command',
                             **ms(L.q_end)))
            perf.append(dict(**key, metric=f'{name} dQ region over the command',
                             **ms(L.q_end - L.q_start)))

        mpc = d['mpc'].solve_ms.dropna() if not d['mpc'].empty else pd.Series(dtype=float)
        cost.append(dict(**key, metric='MPC cycle time [ms]', **ms(mpc),
                         p95=float(np.percentile(mpc, 95)) if len(mpc) else np.nan,
                         over_budget_pct=(100.0 * float((mpc > BUDGET_MS).mean())
                                          if len(mpc) else np.nan)))
        ctrl = d['ctrl']
        cyc = ctrl[ctrl.episode_id.isin(set(d['valid'].episode_id))].cycle_s.dropna() * 1e3
        cost.append(dict(**key, metric='gradient cycle time [ms]', **ms(cyc),
                         p95=float(np.percentile(cyc, 95)) if len(cyc) else np.nan,
                         over_budget_pct=np.nan))

        a, n_legs = accel_pool(d, accel_mod)
        pool.append(dict(**key, n_samples=len(a), n_phases=n_legs,
                         mean=float(a.mean()), std=float(a.std(ddof=1)),
                         median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                         max=float(a.max())))
    return pd.DataFrame(perf), pd.DataFrame(cost), pd.DataFrame(pool)


# ── LaTeX ───────────────────────────────────────────────────────────────────

ORDER = [('setup 1', 'fixed'), ('setup 1', 'moving'), ('setup 2', 'fixed'), ('setup 2', 'moving')]


def _pick(df, metric):
    return [df[(df.setup == s) & (df.condition == c) & (df.metric == metric)].iloc[0]
            for s, c in ORDER]


def _row(label, cells, bold=(), skip=0):
    """One tabular row; `skip` adds the [Npt] gap LaTeX wants on the row itself."""
    out = []
    for i, c in enumerate(cells):
        out.append(f'\\textbf{{{c}}}' if i in bold else c)
    tail = '\\\\' + (f'[{skip}pt]' if skip else '')
    return f'{label} & ' + ' & '.join(out) + tail


def _group(title):
    return f'\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\'


def latex(perf, cost, pool):
    def pm(metric, fmt='{:.0f}'):
        return [f'{fmt.format(r["mean"])}\\,$\\pm$\\,{fmt.format(r["std"])}'
                for r in _pick(perf, metric)]

    n = [f'{int(r["value"])}' for r in _pick(perf, 'N (valid runs)')]
    L = ['\\begin{table}[t]', '\\centering',
         '\\caption{Task performance. Values are mean\\,$\\pm$\\,std over the valid runs of',
         f'each condition ($N={n[0]}$, {n[1]}, {n[2]}, {n[3]} for setup~1 fixed/moving and setup~2',
         'fixed/moving). Timing, $R_i$ and geometry statistics are computed over successful',
         'runs only.}',
         '\\label{tab:results_perf}', '\\footnotesize',
         '\\setlength{\\tabcolsep}{3.5pt}', '\\begin{tabular}{@{}lcccc@{}}', '\\toprule',
         ' & \\multicolumn{2}{c}{Setup 1} & \\multicolumn{2}{c}{Setup 2}\\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
         ' & Fixed & Moving & Fixed & Moving\\\\', '\\midrule',
         _group('Task outcome'),
         _row('Valid / recorded runs',
              [f'{int(a["value"])}\\,/\\,{int(b["value"])}'
               for a, b in zip(_pick(perf, 'N (valid runs)'), _pick(perf, 'runs recorded'))]),
         _row('Success rate [\\%]',
              [f'{r["value"]:.0f}' for r in _pick(perf, 'success rate [%]')], bold=(1, 3), skip=2),
         _group('Duration [s]'),
         _row('Total run time', pm('total run time [s]'), bold=(1, 3)),
         _row('\\quad Pick approach', pm('pick approach [s]')),
         _row('\\quad Place approach', pm('place approach [s]'), bold=(1, 3), skip=2),
         _group('Reachability index at end of approach'),
         _row('Pick, $R_i$', pm('pick R_i at end of approach', '{:.2f}')),
         _row('Place, $R_i$', pm('place R_i at end of approach', '{:.2f}'), bold=(1, 3), skip=2),
         _group('Controller objective $Q$ (3 spheres) at end of base command'),
         _row('\\quad Pick approach', pm('pick Q region at end of command', '{:.2f}'),
              bold=(1, 3)),
         _row('\\quad Place approach', pm('place Q region at end of command', '{:.2f}'),
              bold=(1, 3), skip=2),
         _group('Geometry: base--target dist. [m], leg start $\\rightarrow$ arm arrival')]
    for lab, leg in (('\\quad Pick approach', 'pick'), ('\\quad Place approach', 'place')):
        a = _pick(perf, f'{leg} base-target dist. start [m]')
        z = _pick(perf, f'{leg} base-target dist. end [m]')
        L.append(_row(lab, [f'{x["mean"]:.2f}\\,$\\rightarrow$\\,{y["mean"]:.2f}'
                            for x, y in zip(a, z)]))
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']

    def cm(metric, field, fmt='{:.0f}'):
        sel = cost[cost.metric == metric]
        return [fmt.format(sel[(sel.setup == s) & (sel.condition == c)].iloc[0][field])
                for s, c in ORDER]

    def cpm(metric):
        return [f'{m}\\,$\\pm$\\,{d}' for m, d in zip(cm(metric, 'mean'), cm(metric, 'std'))]

    def po(field, fmt):
        return [fmt.format(pool[(pool.setup == s) & (pool.condition == c)].iloc[0][field])
                for s, c in ORDER]

    L += ['\\begin{table}[t]', '\\centering',
          '\\caption{Controller cost and arm behaviour. Statistics are pooled over all logged',
          'cycles of all valid runs in each condition, successful or not. The MPC budget is',
          '400\\,ms; if no solution is returned within the budget, the previous trajectory is',
          'kept. Joint accelerations are re-derived from the logged positions (Savitzky--Golay,',
          '0.41\\,s window, 100\\,Hz grid) over the motion phases only.}',
          '\\label{tab:results_cost}', '\\footnotesize',
          '\\setlength{\\tabcolsep}{3.5pt}', '\\begin{tabular}{@{}lcccc@{}}', '\\toprule',
          ' & \\multicolumn{2}{c}{Setup 1} & \\multicolumn{2}{c}{Setup 2}\\\\',
          '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
          ' & Fixed & Moving & Fixed & Moving\\\\', '\\midrule',
          _group('MPC cycle time [ms]'),
          _row('\\quad mean\\,$\\pm$\\,std', cpm('MPC cycle time [ms]')),
          _row('\\quad median', cm('MPC cycle time [ms]', 'median')),
          _row('\\quad p95', cm('MPC cycle time [ms]', 'p95')),
          _row('\\quad cycles $>$ 400\\,ms [\\%]',
               cm('MPC cycle time [ms]', 'over_budget_pct', '{:.1f}')),
          _row('\\quad logged cycles', cm('MPC cycle time [ms]', 'n'), skip=2),
          _group('Gradient cycle time [ms]'),
          _row('\\quad mean\\,$\\pm$\\,std', cpm('gradient cycle time [ms]')),
          _row('\\quad median', cm('gradient cycle time [ms]', 'median')),
          _row('\\quad p95', cm('gradient cycle time [ms]', 'p95')),
          _row('\\quad logged cycles', cm('gradient cycle time [ms]', 'n'), skip=2),
          _group('Joint acceleration $\\lVert\\ddot{q}\\rVert$ [rad/s$^2$]'),
          _row('\\quad mean\\,$\\pm$\\,std',
               [f'{m}\\,$\\pm$\\,{d}' for m, d in zip(po('mean', '{:.3f}'), po('std', '{:.3f}'))]),
          _row('\\quad median', po('median', '{:.3f}')),
          _row('\\quad p95', po('p95', '{:.2f}')),
          _row('\\quad max', po('max', '{:.2f}')),
          '\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(L)


def main():
    os.makedirs(TAB, exist_ok=True)
    accel_mod = __import__('accel_over_time')
    conditions = load_conditions()
    perf, cost, pool = build(conditions, accel_mod)
    perf.to_csv(os.path.join(TAB, 'perf.csv'), index=False)
    cost.to_csv(os.path.join(TAB, 'cost.csv'), index=False)
    pool.to_csv(os.path.join(TAB, 'accel_pool.csv'), index=False)
    with open(os.path.join(TAB, 'tables.tex'), 'w') as f:
        f.write(latex(perf, cost, pool) + '\n')
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 300)
    print(perf.to_string(index=False))
    print()
    print(cost.to_string(index=False))
    print()
    print(pool.to_string(index=False))
    print('\nwrote', TAB)


if __name__ == '__main__':
    main()
