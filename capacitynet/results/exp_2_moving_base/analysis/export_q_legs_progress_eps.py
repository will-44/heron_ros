#!/usr/bin/env python3
"""Publication variant of cmp_q_legs_progress.png, as EPS.

Same data and layout as analyze_setup2.fig_q_legs_progress, but: no suptitle,
'leg' renamed 'run', and the '(controller objective)', '(gradient)',
'(pregrasp)', '(preplace)' qualifiers dropped (the caption carries them).

EPS has no transparency, and rasterized bands land misplaced once
bbox_inches='tight' crops the figure (matplotlib 3.6). So the ±1 std bands are
drawn fully vector with opaque colours that reproduce alpha=0.18 over the
background, plus their overlap in the colour the two stacked bands would give.

Usage:
    python3 export_q_legs_progress_eps.py
"""

import argparse
import os

import matplotlib as mpl
import numpy as np
from matplotlib.colors import to_rgb

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import analyze_setup2 as S
import reach_analysis as R

RUN_NAMES = ['base fixed', 'base moving']
LEG_LABEL = {R.PREGRASP: 'pick approach', R.PREPLACE: 'place approach'}
# Column symbols as in the caption: workspace index, goal-sphere index, and the
# mean of the two path-sphere indices.
SERIES = [('q_center', r'$\bar{R}$'), ('ws_mean_0', r'$\bar{R}_{\mathrm{goal}}$'),
          ('qpath', r'$\bar{R}_{\mathrm{path}}$')]


BAND_ALPHA = 0.18


def over(color, below, alpha=BAND_ALPHA):
    """Opaque colour of `color` at `alpha` composited over `below`."""
    return alpha * np.array(to_rgb(color)) + (1 - alpha) * np.array(to_rgb(below))


def draw_bands(ax, bands):
    """±1 std bands as opaque vector fills that look like stacked alpha fills."""
    for x, lo, hi, col in bands:
        ax.fill_between(x, lo, hi, color=over(col, R.SURFACE), linewidth=0, zorder=2)
    if len(bands) == 2:
        (x, lo0, hi0, c0), (_, lo1, hi1, c1) = bands
        lo, hi = np.maximum(lo0, lo1), np.minimum(hi0, hi1)
        ax.fill_between(x, lo, hi, where=lo < hi, interpolate=True,
                        color=over(c1, over(c0, R.SURFACE)), linewidth=0, zorder=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fixed-dir', default=S.DEFAULT_FIXED)
    ap.add_argument('--moving-dir', default=S.DEFAULT_MOVING)
    ap.add_argument('--out', default=os.path.join(S.DEFAULT_MOVING, 'exp_2_comparison_figures',
                                                  'cmp_q_legs_progress.eps'))
    ap.add_argument('--preview', default=None, help='optional PNG preview path')
    ap.add_argument('--font-size', type=float, default=14, help='base font size (pt)')
    args = ap.parse_args()
    mpl.rcParams.update({'font.size': args.font_size, 'axes.labelsize': args.font_size,
                         'xtick.labelsize': args.font_size - 1,
                         'ytick.labelsize': args.font_size - 1})

    runs = S.load_both(args.fixed_dir, args.moving_dir)
    classes = ('success',)
    fig, axes = R.new_fig(2, 3, figsize=(14, 8), sharey=True)
    v_min = 1.0
    for r, phase in enumerate((R.PREGRASP, R.PREPLACE)):
        n_runs = []
        for c, (key, _) in enumerate(SERIES):
            ax = axes[r, c]
            bands = []
            for k, d in enumerate(runs):
                keep = d['idx'][d['idx'].cls.isin(classes)].episode_id
                grid_p, arr = S.leg_curves_progress(
                    dict(d, legs=d['legs'][d['legs'].episode_id.isin(keep)]), phase, key)
                if c == 0:
                    n_runs.append(len(arr))
                if not arr.size:
                    continue
                mean, std = arr.mean(axis=0), arr.std(axis=0)
                bands.append((grid_p, mean - std, mean + std, S.RUNS[k][1]))
                ax.plot(grid_p, mean, '-', color=S.RUNS[k][1], linewidth=2.0, zorder=4)
                v_min = min(v_min, float((mean - std).min()))
            draw_bands(ax, bands)
        for c in range(len(SERIES)):
            ax = axes[r, c]
            ax.set_xlim(0, 100)
            ax.tick_params(labelsize=args.font_size - 1)
            ax.set_xlabel('run progress (% of the run duration)')
            if c == 0:
                ax.set_ylabel(LEG_LABEL[phase])
            if r == 0:
                ax.set_title(SERIES[c][1], fontsize=args.font_size + 3, color=R.INK, loc='left')
        axes[r, 0].legend(handles=[Line2D([], [], color=col, linewidth=2, label=f'{name} ({n} runs)')
                                   for name, (_, col), n in zip(RUN_NAMES, S.RUNS, n_runs)]
                          # Opaque swatch pre-blended over the background: a
                          # transparent vector patch would turn solid in EPS.
                          + [Patch(color=over(R.INK, R.SURFACE),
                                   label='±1 std across runs')],
                          fontsize=args.font_size - 2, frameon=False, loc='lower left')
    axes[0, 0].set_ylim(max(0.0, v_min - 0.03), 1.02)
    fig.tight_layout()
    fig.savefig(args.out, format='eps', facecolor=R.SURFACE, bbox_inches='tight')
    print('wrote', args.out)
    if args.preview:
        fig.savefig(args.preview, dpi=80, facecolor=R.SURFACE, bbox_inches='tight')


if __name__ == '__main__':
    main()
