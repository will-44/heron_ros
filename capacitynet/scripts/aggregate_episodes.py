#!/usr/bin/env python3
"""Pool several experiment runs into analysis-ready tables.

Offline tool, same status as `evaluation.py` and `plot_metrics.py`: not a ROS
node, no entry in `setup.py`, run it directly.

    python3 scripts/aggregate_episodes.py results/run_A results/run_B --out /tmp/analyse
    python3 scripts/aggregate_episodes.py 'results/run_*' --out /tmp/analyse --plots

Writes into `--out`:

  episodes_all.csv     one row per episode, every run pooled, with its outcome
  phase_summary_all.csv one row per (run, episode, phase) -- the A/B table
  samples_long.csv     tidy/long format: one row per (episode, t_rel, metric)

`samples_long.csv` is the point of the exercise. Wide per-signal files cannot be
compared across runs without a join per file; in long format the whole experiment
answers a question in one line:

    df.groupby(['base_control', 'phase', 'metric'])['value'].describe()

Runs written before the rename (`tests_index.csv`, `test_<id>_*` directories,
a `test_id` column) are read too and mapped onto the current schema, with
`phase='unknown'` where the phase was never recorded.
"""

import argparse
import glob
import os
import sys

import pandas as pd

# Import experiment_plots from the package next door without needing the workspace
# to be sourced -- this script is meant to run on a laptop over copied results too.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Signals lifted into samples_long.csv, per source file. Anything not listed stays
# in the per-episode CSVs; this is the set that is actually compared across runs.
LONG_METRICS = {
    'samples.csv': [
        'q_goal', 'q_path_mean', 'q_path_min', 'q_path_max',
        'rm_age_ms', 'compute_ms', 'goal_age_s',
    ],
    'mpc_diagnostics.csv': [
        'solve_ms', 'budget_ms', 'arrival_fk_err_m', 'arrival_fk_rot_err_deg',
        'cost_tool_pose_pos', 'cost_cspace_vel', 'cost_cspace_acc',
    ],
    'joint_states.csv': ['manipulability'],
    'cmd_vel.csv': ['vx', 'vy', 'x_est', 'y_est', 'dist_est'],
}

_KEYS = ['run', 'run_label', 'base_control', 'episode_id', 'phase', 'brain_state',
         'sub_state', 't_rel']


def _read_csv(path):
    try:
        if os.path.getsize(path) == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def _normalize(df):
    """Map pre-rename columns onto the current schema and fill missing episode keys."""
    if df.empty:
        return df
    if 'episode_id' not in df.columns and 'test_id' in df.columns:
        df = df.rename(columns={'test_id': 'episode_id'})
    for column, default in (('phase', 'unknown'), ('brain_state', 'unknown'),
                            ('sub_state', ''), ('t_rel', float('nan'))):
        if column not in df.columns:
            df[column] = default
    return df


def _episode_dirs(run_dir):
    found = sorted(glob.glob(os.path.join(run_dir, 'episode_*')) +
                   glob.glob(os.path.join(run_dir, 'test_*')))
    return [d for d in found if os.path.isdir(d)]


def _run_identity(run_dir, index):
    """(run, run_label, base_control) for a run, falling back to the directory name."""
    run = os.path.basename(os.path.normpath(run_dir))
    label, base_control = '', ''
    if not index.empty:
        if 'run_label' in index.columns and index['run_label'].notna().any():
            label = str(index['run_label'].dropna().iloc[0])
        if 'base_control' in index.columns and index['base_control'].notna().any():
            base_control = index['base_control'].dropna().iloc[0]
    return run, label, base_control


def load_index(run_dir):
    """Episode index of one run, stamped with its identity and missing columns filled."""
    index = _normalize(_read_csv(os.path.join(run_dir, 'episodes_index.csv')))
    if index.empty:
        index = _normalize(_read_csv(os.path.join(run_dir, 'tests_index.csv')))
    if index.empty:
        return index
    for column, default in (('outcome', 'unknown'), ('outcome_message', ''),
                            ('run_label', ''), ('base_control', ''),
                            ('base_dist_est', 0.0), ('n_phases', 0)):
        if column not in index.columns:
            index[column] = default
    run, label, base_control = _run_identity(run_dir, index)
    index.insert(0, 'run', run)
    index['run_label'] = index['run_label'].fillna(label).replace('', label)
    return index


def load_phase_summary(run_dir, run, label, base_control):
    """Per-(episode, phase) rows of one run, rebuilt from phases.csv if absent."""
    summary = _normalize(_read_csv(os.path.join(run_dir, 'phase_summary.csv')))
    if summary.empty:
        # Older runs, or a run killed before an episode closed: rebuild from the
        # per-episode phases.csv files rather than reporting nothing.
        frames = []
        for directory in _episode_dirs(run_dir):
            phases = _normalize(_read_csv(os.path.join(directory, 'phases.csv')))
            if not phases.empty:
                frames.append(phases)
        if not frames:
            return pd.DataFrame()
        summary = pd.concat(frames, ignore_index=True)
    summary.insert(0, 'run', run)
    if 'run_label' not in summary.columns:
        summary['run_label'] = label
    if 'base_control' not in summary.columns:
        summary['base_control'] = base_control
    return summary


def load_samples_long(run_dir, run, label, base_control):
    """Every per-sample signal of a run, melted into (keys..., metric, value)."""
    frames = []
    for directory in _episode_dirs(run_dir):
        for filename, metrics in LONG_METRICS.items():
            df = _normalize(_read_csv(os.path.join(directory, filename)))
            if df.empty:
                continue
            present = [m for m in metrics if m in df.columns]
            if not present:
                continue
            df = df.assign(run=run, run_label=label, base_control=base_control,
                           source_file=filename)
            if 'source' in df.columns:
                # cmd_vel.csv distinguishes the controller topic from the base
                # topic; keep that in the metric name or the two would be pooled.
                df['metric_prefix'] = df['source'].astype(str) + '_'
            else:
                df['metric_prefix'] = ''
            long = df.melt(
                id_vars=_KEYS + ['source_file', 'metric_prefix'],
                value_vars=present, var_name='metric', value_name='value')
            long['metric'] = long['metric_prefix'] + long['metric']
            frames.append(long.drop(columns=['metric_prefix']))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=['value'])


def expand_run_dirs(patterns):
    """Expand shell globs the caller may have quoted, keeping only directories."""
    run_dirs = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or [pattern]
        run_dirs += [m for m in matches if os.path.isdir(m)]
    return run_dirs


def main():
    """Pool the given run directories into the three analysis tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('run_dirs', nargs='+', help='run directories (globs accepted)')
    parser.add_argument('--out', required=True, help='directory receiving the pooled CSVs')
    parser.add_argument('--plots', action='store_true',
                        help="also regenerate each run's figures from its CSVs")
    args = parser.parse_args()

    run_dirs = expand_run_dirs(args.run_dirs)
    if not run_dirs:
        parser.error(f'no run directory matched {args.run_dirs}')
    os.makedirs(args.out, exist_ok=True)

    indexes, summaries, longs = [], [], []
    for run_dir in run_dirs:
        index = load_index(run_dir)
        run, label, base_control = _run_identity(run_dir, index)
        n_episodes = len(index) if not index.empty else len(_episode_dirs(run_dir))
        print(f'{run_dir}: {n_episodes} episodes, label={label!r} '
              f'base_control={base_control!r}')
        if not index.empty:
            indexes.append(index)
        summary = load_phase_summary(run_dir, run, label, base_control)
        if not summary.empty:
            summaries.append(summary)
        long = load_samples_long(run_dir, run, label, base_control)
        if not long.empty:
            longs.append(long)

        if args.plots:
            from capacitynet.control.scripts import experiment_plots
            for path in experiment_plots.plot_run(run_dir):
                print(f'  figure: {path}')

    written = []
    for frames, name in ((indexes, 'episodes_all.csv'),
                         (summaries, 'phase_summary_all.csv'),
                         (longs, 'samples_long.csv')):
        if not frames:
            print(f'  (nothing to write for {name})')
            continue
        out_path = os.path.join(args.out, name)
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
        written.append(out_path)
        print(f'wrote {out_path}')

    if written:
        print('\nExample analysis:\n'
              "  df = pd.read_csv('samples_long.csv')\n"
              "  df.groupby(['base_control', 'phase', 'metric'])['value'].describe()")


if __name__ == '__main__':
    main()
