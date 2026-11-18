#!/usr/bin/env python3
"""Figures for a pick-and-place experiment run. ROS-free on purpose.

Same status as `grid_markers.py`: pure functions over plain data, importable both
by the recording node (which calls them at the end of a run) and by the offline
aggregation script (which regenerates the same figures without re-running the
robot). Keeping them here is what stops the plotting code from being written
twice and drifting apart.

Everything except `plot_episode` reads a *run directory* as written by
`workspace_reachability_node`:

    run_dir/episodes_index.csv, phase_summary.csv, control_debug.csv
    run_dir/episode_<id>_<ts>/{samples,cmd_vel,mpc_diagnostics,phases}.csv

Older runs written before the rename (`tests_index.csv`, `test_<id>_*`, `test_id`)
are read too, so the figures can be regenerated over the existing `results/`.
"""

import glob
import os

import matplotlib

import numpy as np

import pandas as pd

# Headless-safe backend switch (the node has no display); done after the import
# block so it stays a normal statement rather than splitting the imports.
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402,I100 - must follow the backend switch

# matplotlib renamed boxplot's `labels` argument to `tick_labels` in 3.9.
_MPL_GE_3_9 = tuple(int(p) for p in matplotlib.__version__.split('.')[:2]) >= (3, 9)

# One color per brain state, so the same phase reads the same in every figure.
PHASE_COLORS = {
    'idle': '#bab0ac',
    'act_home_enter': '#76b7b2',
    'act_grasp': '#4e79a7',
    'act_release': '#f28e2b',
    'act_home_exit': '#59a14f',
}
_UNKNOWN_PHASE_COLOR = '#d3d3d3'

_OUTCOME_MARKERS = {'success': ('✓', '#59a14f'), 'failure': ('✗', '#e15759'),
                    'cancelled': ('⊘', '#f28e2b'), 'unknown': ('?', '#999999')}


def phase_color(brain_state):
    """Color for a brain state, shared by every figure so phases stay comparable."""
    return PHASE_COLORS.get(str(brain_state), _UNKNOWN_PHASE_COLOR)


# ── Run loading ─────────────────────────────────────────────────────────────

def _read_csv(path):
    """Read a CSV, returning an empty DataFrame rather than raising."""
    try:
        if os.path.getsize(path) == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def _normalize(df):
    """Map the pre-rename column names onto the current ones."""
    if df.empty:
        return df
    if 'episode_id' not in df.columns and 'test_id' in df.columns:
        df = df.rename(columns={'test_id': 'episode_id'})
    for column, default in (('phase', 'unknown'), ('brain_state', 'unknown'),
                            ('sub_state', ''), ('t_rel', np.nan)):
        if column not in df.columns:
            df[column] = default
    return df


def episode_dirs(run_dir):
    """Per-episode subdirectories, current and pre-rename, in id order."""
    found = sorted(glob.glob(os.path.join(run_dir, 'episode_*')) +
                   glob.glob(os.path.join(run_dir, 'test_*')))
    return [d for d in found if os.path.isdir(d)]


def _concat_per_episode(run_dir, filename):
    frames = []
    for directory in episode_dirs(run_dir):
        df = _normalize(_read_csv(os.path.join(directory, filename)))
        if df.empty:
            continue
        df['episode_dir'] = os.path.basename(directory)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_run(run_dir):
    """All of a run's tables, each possibly empty. Never raises on missing files."""
    index = _normalize(_read_csv(os.path.join(run_dir, 'episodes_index.csv')))
    if index.empty:
        index = _normalize(_read_csv(os.path.join(run_dir, 'tests_index.csv')))
    return {
        'run_dir': run_dir,
        'index': index,
        'phase_summary': _normalize(_read_csv(os.path.join(run_dir, 'phase_summary.csv'))),
        'control_debug': _read_csv(os.path.join(run_dir, 'control_debug.csv')),
        'samples': _concat_per_episode(run_dir, 'samples.csv'),
        'cmd_vel': _concat_per_episode(run_dir, 'cmd_vel.csv'),
        'mpc': _concat_per_episode(run_dir, 'mpc_diagnostics.csv'),
        'phases': _concat_per_episode(run_dir, 'phases.csv'),
    }


def _empty_panel(ax, message):
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=9,
            color='#999999', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _episode_q_series(samples, episode_id):
    """(t_rel, q_goal) for one episode, sorted, or two empty arrays."""
    if samples.empty:
        return np.array([]), np.array([])
    rows = samples[samples['episode_id'] == episode_id].sort_values('t_rel')
    if rows.empty or 'q_goal' not in rows.columns:
        return np.array([]), np.array([])
    return rows['t_rel'].to_numpy(float), rows['q_goal'].to_numpy(float)


def _draw_phase_bands(ax, phases, episode_id=None):
    """Shade the axes background by brain_state over t_rel."""
    if phases.empty:
        return
    rows = phases if episode_id is None else phases[phases['episode_id'] == episode_id]
    for _, row in rows.iterrows():
        try:
            start = float(row['t_start_rel'])
            end = float(row['t_end_rel'])
        except (KeyError, TypeError, ValueError):
            continue
        ax.axvspan(start, end, color=phase_color(row.get('brain_state')), alpha=0.12, lw=0)


# ── Per-episode figure ──────────────────────────────────────────────────────

def plot_episode(out_path, *, episode_id, rel_times, q_goal, q_path_mean,
                 q_path_min, q_path_max, phase_events, summary):
    """Q_goal / Q_path over one episode, with a vertical rule at each phase change.

    Takes plain lists so the recording node can call it without first writing and
    re-reading its own CSVs.
    """
    # Reserve a band at the top of the axes (in axes-fraction, independent of the
    # data range) for the rotated phase labels, so they can never collide with the
    # title above the axes.
    label_band = 0.34 if phase_events else 0.0
    figsize = (9, 5.6) if phase_events else (9, 4.5)

    fig, ax = plt.subplots(figsize=figsize)
    t = list(rel_times)

    ax.fill_between(t, q_path_min, q_path_max, color='#e15759', alpha=0.15,
                    label='Q_path range [min, max]')
    ax.plot(t, q_path_mean, marker='.', color='#e15759', linewidth=1.2,
            label='Q_path mean')
    ax.plot(t, q_goal, marker='.', color='#4e79a7', linewidth=1.2, label='Q_goal')

    ax.axhline(summary['q_goal_mean'], color='#4e79a7', linestyle='--', linewidth=1,
               alpha=0.6, label=f"Q_goal episode mean ({summary['q_goal_mean']:.3f})")
    ax.axhline(summary['q_path_mean'], color='#e15759', linestyle='--', linewidth=1,
               alpha=0.6, label=f"Q_path episode mean ({summary['q_path_mean']:.3f})")

    # Background band per phase: from each phase change to the next, or to the end.
    t_end = max(t) if t else 0.0
    for i, (rel_t, phase) in enumerate(phase_events):
        next_t = phase_events[i + 1][0] if i + 1 < len(phase_events) else t_end
        ax.axvspan(rel_t, next_t, color=phase_color(str(phase).split(':')[0]),
                   alpha=0.12, lw=0)

    # Two de-cluttering rules, needed as soon as a cycle has short phases (a real
    # grasp packs srv_mask_object / act_close_gripper into a few hundred ms):
    #   - drop the brain_state prefix, already carried by the band color, and keep
    #     only the sub-state, which is what actually differs between neighbours;
    #   - flip tightly spaced labels to the other side of their rule, so a run of
    #     short phases alternates instead of overprinting.
    data_top = 1.0 - label_band
    min_sep = 0.04 * (t_end - min(t)) if t else 0.0

    # Every phase boundary gets its rule, but bursts share one label: a real grasp
    # fires act_open_gripper / srv_mask_object / act_move_grasp inside ~60 ms, and
    # at that spacing three rotated labels overprint whatever slot they are given.
    # Rotating them to alternating sides only buys two positions, so instead the
    # burst is named once, at its midpoint. The individual names stay in phases.csv.
    groups = []
    for rel_t, phase in phase_events:
        ax.axvline(rel_t, ymax=data_top, color='#666666', linestyle=':', linewidth=1,
                   alpha=0.8)
        # Drop the brain_state prefix: the band color already carries it, and the
        # sub-state is what differs between neighbours.
        text = str(phase).split(':', 1)[-1] or str(phase)
        if groups and (rel_t - groups[-1][-1][0]) < min_sep:
            groups[-1].append((rel_t, text))
        else:
            groups.append([(rel_t, text)])

    for index, group in enumerate(groups):
        names = [name for _, name in group]
        label = ' + '.join(names)
        if len(label) > 28:
            label = f'{names[0]} +{len(names) - 1} more'
        at_t = 0.5 * (group[0][0] + group[-1][0])
        side = index % 2
        ax.text(at_t, data_top + 0.01, f' {label}' if side == 0 else f'{label} ',
                transform=ax.get_xaxis_transform(), rotation=90, va='bottom',
                ha='left' if side == 0 else 'right', fontsize=7, color='#666666')

    ax.set_xlabel('Time since episode start (s)')
    ax.set_ylabel('Mean reachability Q')
    ax.set_title(
        f"Episode {episode_id} [{summary.get('outcome', 'unknown')}] — workspace "
        f"reachability (n={summary['n_samples']} samples, "
        f"{summary['duration_s']:.1f}s, "
        f"manipulability_mean={summary['manipulability_mean']:.3f})")
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ── Base trajectory ─────────────────────────────────────────────────────────

def _plot_trajectory_on(ax, run, annotate=True):
    """Draw every episode's integrated base path, colored by Q_goal."""
    cmd_vel = run['cmd_vel']
    if cmd_vel.empty or 'x_est' not in cmd_vel.columns:
        _empty_panel(ax, 'no cmd_vel.csv\n(base position not recorded)')
        return None

    # Only the integrated source carries a position; the other is a raw command echo.
    if 'source' in cmd_vel.columns:
        sources = set(cmd_vel['source'].unique())
        integrated = 'base' if 'base' in sources else 'controller'
        cmd_vel = cmd_vel[cmd_vel['source'] == integrated]
    if cmd_vel.empty:
        _empty_panel(ax, 'no integrated base position')
        return None

    samples = run['samples']
    scatter = None
    for episode_id, rows in cmd_vel.groupby('episode_id'):
        rows = rows.sort_values('t_rel')
        x = rows['x_est'].to_numpy(float)
        y = rows['y_est'].to_numpy(float)
        if len(x) < 2:
            continue

        ax.plot(x, y, '-', color='#cccccc', linewidth=1, zorder=1)

        t_q, q = _episode_q_series(samples, episode_id)
        if len(t_q) >= 1:
            q_at = np.interp(rows['t_rel'].to_numpy(float), t_q, q)
            scatter = ax.scatter(x, y, c=q_at, cmap='viridis', s=9, zorder=2)
        else:
            ax.scatter(x, y, color='#4e79a7', s=9, zorder=2)

        ax.plot(x[0], y[0], 'x', color='#333333', markersize=8, zorder=3)
        ax.plot(x[-1], y[-1], 'o', color='#333333', markersize=6,
                markerfacecolor='none', zorder=3)
        if annotate:
            ax.annotate(f'ep {episode_id}', (x[-1], y[-1]), fontsize=7,
                        xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel('x_est (m, base frame)')
    ax.set_ylabel('y_est (m, base frame)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    return scatter


def plot_base_trajectory(run_dir, out_path=None):
    """Standalone base-trajectory figure: where the base went, and how Q followed.

    Returns the path written, or None when the run holds no base position at all
    (which is itself the answer for a `base_control:=false` run).
    """
    run = load_run(run_dir)
    if run['cmd_vel'].empty:
        return None

    out_path = out_path or os.path.join(run_dir, 'base_trajectory.png')
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    scatter = _plot_trajectory_on(ax, run, annotate=True)
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, label='Q_goal')

    total = run['index']['base_dist_est'].sum() if 'base_dist_est' in run['index'] else 0.0
    ax.set_title(
        f'Base trajectory (open-loop integration of the commanded velocity)\n'
        f'{len(episode_dirs(run_dir))} episodes, {total:.2f} m travelled — '
        f'× start, ○ end')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ── Run dashboard ───────────────────────────────────────────────────────────

def _panel_q_over_time(ax, run):
    samples = run['samples']
    if samples.empty:
        _empty_panel(ax, 'no samples.csv')
        return
    _draw_phase_bands(ax, run['phases'])
    for episode_id, rows in samples.groupby('episode_id'):
        rows = rows.sort_values('t_rel')
        ax.plot(rows['t_rel'], rows['q_goal'], linewidth=1, alpha=0.85,
                label=f'ep {episode_id} Q_goal')
        if 'q_path_mean' in rows.columns:
            ax.plot(rows['t_rel'], rows['q_path_mean'], linewidth=0.9, alpha=0.5,
                    linestyle='--')
    ax.set_xlabel('t_rel (s)')
    ax.set_ylabel('Q')
    ax.set_title('Q over the cycle (solid: Q_goal, dashed: Q_path)')
    ax.grid(True, alpha=0.3)
    if samples['episode_id'].nunique() <= 6:
        ax.legend(fontsize=6, loc='best')


def _panel_timeline(ax, run):
    phases, index = run['phases'], run['index']
    if phases.empty:
        _empty_panel(ax, 'no phases.csv')
        return

    outcomes = {}
    if not index.empty and 'outcome' in index.columns:
        outcomes = dict(zip(index['episode_id'], index['outcome']))

    seen = {}
    episode_ids = sorted(phases['episode_id'].unique())
    for row_y, episode_id in enumerate(episode_ids):
        rows = phases[phases['episode_id'] == episode_id]
        end = 0.0
        for _, phase_row in rows.iterrows():
            start = float(phase_row['t_start_rel'])
            duration = float(phase_row['duration_s'])
            brain_state = str(phase_row.get('brain_state', 'unknown'))
            ax.barh(row_y, duration, left=start, height=0.6,
                    color=phase_color(brain_state), edgecolor='white', linewidth=0.4)
            seen[brain_state] = phase_color(brain_state)
            end = max(end, start + duration)
        marker, color = _OUTCOME_MARKERS.get(
            str(outcomes.get(episode_id, 'unknown')), _OUTCOME_MARKERS['unknown'])
        ax.text(end, row_y, f' {marker}', va='center', ha='left', fontsize=11, color=color)

    ax.set_yticks(range(len(episode_ids)))
    ax.set_yticklabels([f'ep {i}' for i in episode_ids], fontsize=7)
    ax.set_xlabel('t_rel (s)')
    ax.set_title('Episode timeline by phase (✓ success, ✗ failure, ? unknown)')
    ax.grid(True, axis='x', alpha=0.3)
    if seen:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in seen.values()]
        ax.legend(handles, list(seen.keys()), fontsize=6, loc='lower right')


def _panel_q_by_phase(ax, run):
    samples = run['samples']
    if samples.empty or 'phase' not in samples.columns:
        _empty_panel(ax, 'no per-phase samples')
        return
    phases = list(samples['phase'].dropna().unique())
    data = [samples.loc[samples['phase'] == p, 'q_goal'].dropna().to_numpy()
            for p in phases]
    keep = [(p, d) for p, d in zip(phases, data) if len(d)]
    if not keep:
        _empty_panel(ax, 'no Q per phase')
        return
    labels, arrays = zip(*keep)
    # `tick_labels` is matplotlib >= 3.9; the Jetson image ships 3.6, where the
    # same argument is spelled `labels` (deprecated, then removed, in 3.9+).
    label_kw = 'tick_labels' if _MPL_GE_3_9 else 'labels'
    box = ax.boxplot(arrays, patch_artist=True, showfliers=False,
                     **{label_kw: labels})
    for patch, label in zip(box['boxes'], labels):
        patch.set_facecolor(phase_color(str(label).split(':')[0]))
        patch.set_alpha(0.6)
    ax.set_ylabel('Q_goal')
    ax.set_title('Q_goal distribution by phase')
    ax.tick_params(axis='x', labelrotation=60, labelsize=6)
    ax.grid(True, axis='y', alpha=0.3)


def _panel_performance(ax, run):
    mpc, control_debug = run['mpc'], run['control_debug']
    drew = False
    if not mpc.empty and 'solve_ms' in mpc.columns:
        solve = mpc['solve_ms'].dropna().to_numpy(float)
        if len(solve):
            ax.hist(solve, bins=40, color='#4e79a7', alpha=0.75, label='MPC solve_ms')
            drew = True
        if 'budget_ms' in mpc.columns and len(mpc['budget_ms'].dropna()):
            budget = float(mpc['budget_ms'].dropna().median())
            ax.axvline(budget, color='#e15759', linestyle='--', linewidth=1.2,
                       label=f'budget {budget:.0f} ms')
        if 'budget_exceeded' in mpc.columns and len(mpc):
            exceeded = mpc['budget_exceeded'].astype(str).str.lower().isin(
                ['true', '1', '1.0']).sum()
            ax.set_title(f'MPC solve time — {exceeded}/{len(mpc)} over budget')
    if not control_debug.empty and 'cycle_s' in control_debug.columns:
        cycle_ms = control_debug['cycle_s'].dropna().to_numpy(float) * 1000.0
        if len(cycle_ms):
            ax.hist(cycle_ms, bins=40, color='#59a14f', alpha=0.55,
                    label='gradient cycle (ms)')
            drew = True
    if not drew:
        _empty_panel(ax, 'no mpc_diagnostics.csv / control_debug.csv')
        return
    ax.set_xlabel('milliseconds')
    ax.set_ylabel('count')
    if not ax.get_title():
        ax.set_title('Real-time performance')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)


def _panel_cmd_vel(ax, run):
    """|v| commanded by the controller vs |v| actually reaching the base.

    The controller publishes on /cmd_vel and an external relay forwards it to the
    base's own topic. A run where the blue trace moves and the orange one stays
    flat at zero is a dead relay, not a controller that decided to stay put --
    the one failure this panel exists to make visible.
    """
    cmd_vel = run['cmd_vel']
    if cmd_vel.empty or 'source' not in cmd_vel.columns:
        _empty_panel(ax, 'no cmd_vel.csv')
        return
    colors = {'controller': '#4e79a7', 'base': '#f28e2b'}
    drew = False
    for source, rows in cmd_vel.groupby('source'):
        rows = rows.sort_values(['episode_id', 't_rel'])
        speed = np.hypot(rows['vx'].to_numpy(float), rows['vy'].to_numpy(float))
        ax.plot(rows['t_rel'].to_numpy(float), speed, '.', markersize=2,
                color=colors.get(str(source), '#999999'), alpha=0.7,
                label=f'{source} (max {speed.max():.3f} m/s)' if len(speed) else str(source))
        drew = True
    if not drew:
        _empty_panel(ax, 'no cmd_vel samples')
        return
    ax.set_xlabel('t_rel (s)')
    ax.set_ylabel('|v| (m/s)')
    ax.set_title('Commanded speed: controller vs base topic')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)


def plot_run_summary(run_dir, out_path=None):
    """Six-panel dashboard giving the whole run at a glance.

    Returns the path written, or None if the run directory holds no episode at all.
    """
    run = load_run(run_dir)
    if run['index'].empty and not episode_dirs(run_dir):
        return None

    out_path = out_path or os.path.join(run_dir, 'run_summary.png')
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))

    scatter = _plot_trajectory_on(axes[0][0], run, annotate=False)
    axes[0][0].set_title('Base trajectory, colored by Q_goal')
    if scatter is not None:
        fig.colorbar(scatter, ax=axes[0][0], label='Q_goal', fraction=0.046)

    _panel_q_over_time(axes[0][1], run)
    _panel_timeline(axes[0][2], run)
    _panel_q_by_phase(axes[1][0], run)
    _panel_performance(axes[1][1], run)
    _panel_cmd_vel(axes[1][2], run)

    fig.suptitle(_run_title(run_dir, run), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _run_title(run_dir, run):
    index = run['index']
    n_episodes = len(index) if not index.empty else len(episode_dirs(run_dir))
    label = ''
    base_control = ''
    if not index.empty:
        if 'run_label' in index.columns and index['run_label'].notna().any():
            label = str(index['run_label'].dropna().iloc[0])
        if 'base_control' in index.columns and len(index):
            base_control = f", base_control={index['base_control'].iloc[0]}"
    outcomes = ''
    if not index.empty and 'outcome' in index.columns:
        counts = index['outcome'].value_counts().to_dict()
        outcomes = ' — ' + ', '.join(f'{k}: {v}' for k, v in counts.items())
    header = os.path.basename(os.path.normpath(run_dir))
    if label:
        header = f'{header} [{label}]'
    return f'{header}{base_control} — {n_episodes} episodes{outcomes}'


def plot_run(run_dir):
    """Write both run-level figures; returns the list of paths actually written."""
    written = []
    for fn in (plot_base_trajectory, plot_run_summary):
        path = fn(run_dir)
        if path:
            written.append(path)
    return written


def main():
    """Regenerate the figures of one or more run directories."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('run_dirs', nargs='+', help='run directories to (re)plot')
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        for path in plot_run(run_dir):
            print(path)


if __name__ == '__main__':
    main()
