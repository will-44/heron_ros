# analysis

Offline analysis of the runs in `../results/`. Plain Python (pandas, numpy,
matplotlib, scipy), no ROS except `tools/plot_reachability_surface.py`. Not installed
by colcon: run the scripts from `heron_experiment/`, for example

```bash
cd heron_ros/heron_experiment
python3 analysis/setup_1/analyze_exp1.py
```

`HERON_EXPERIMENT_DIR` defaults to this source tree. Set it to read the runs from
somewhere else, for example a copy on a laptop.

## Where things are read and written

| | |
|---|---|
| `results/<experience>/<run_id>/raw/` | what the nodes recorded; read only |
| `results/<experience>/<run_id>/{tables,figures}/` | the analysis of that one run |
| `results/<experience>/analysis/<study>/` | a study over several runs of one experience |
| `results/analysis/<study>/` | a study over several experiences |

[`runs.py`](runs.py) is the only place that names runs. It holds the four
conditions of the paper (`runs.conditions()`) and the functions that turn a run into
its paths. To analyse a new run, pass `--run-dir <experience>/<run_id>`, or add it
to `runs.py` if a study needs it.

## Studies

| Directory | Reads | Writes |
|---|---|---|
| `setup_1/` | setup 1 (empty table): `analyze_exp1.py` base mobile, `analyze_exp2.py` base fixe, `compare_runs.py` both | each run's `tables/`, `figures/`; `results/setup_1/analysis/comparison/` |
| `setup_2/` | setup 2 (cluttered table): `analyze_setup2.py` both runs and their comparison; `export_q_legs_progress_eps.py` the paper's EPS (`--font-size 16`) | each run's `tables/`, `figures/`; `results/setup_2/analysis/comparison/` |
| `acceleration/`, `velocity/`, `manipulability/`, `base_pose/`, `base_motion_effect/` | the four conditions | `results/analysis/<study>/` |
| `paper_tables/` | the four conditions | `results/analysis/paper_tables/tables/` (`tables.tex`) |
| `controller_bench/` | one control cycle dumped by `~/save_reachability_maps` (`results/controller_bench/20260811_152423_debug_maps/raw/`) | that run's `tables/`, `figures/` |
| `tools/` | `aggregate_episodes.py` (pools runs into long tables), `plot_metrics.py` (a `metrics_recorder` CSV), `plot_reachability_surface.py` (Q(x, y) on a bag or a live robot) | given by `--out` |

Each study keeps its README or REPORT next to its code. `leg_survival.py` is the
median-over-legs plot shared by `acceleration/`, `velocity/` and `manipulability/`.

`setup_2/reach_analysis.py` started as a copy of `setup_1/analyze_exp1.py` and
has since diverged: setup 2 drops invalid episodes before drawing its figures.
Merging the two would change the published setup-1 numbers, so they stay separate.

## Checked at the move

This directory was moved here from `capacitynet/results/` and `capacitynet/analysis/`
on 2026-11-18. The old runs were copied into `results/setup_1/`, `setup_2/` and
`tests/`. Every table and figure regenerated from the new layout is byte-identical
to the one in `src/capacitynet/results/`, with one exception: the paper's EPS, whose
header carries its creation date.

The controller bench now loads its network through the Heron Package. With PyTorch it
gives the same Q as the old `ReachabilityEngine` path to within 1.2e-7. See
`controller_bench/validate.py` for why the dump's shifted candidates can no longer be
compared voxel by voxel.
