"""Draw a median across legs without hiding the legs that outlive the quorum.

Every per-leg curve in these analyses used to be masked with `ok = n >= 5`: the summary
was drawn only while at least five legs were still running. That hid the story at both
ends. The plotted average was taken over a shrinking, *self-selected* set of legs — the
slow ones, which are also the ones whose score was still bad — and the curve stopped
several seconds before the longest legs actually ended (setup 2, pick approach: the curve
was cut at 18 s while legs ran to 27 s, so a quarter of the sample was never drawn).

The rule here instead:

* every leg is shown to its own end, and the x axis ends with the longest leg;
* the median is **solid** while at least `SOLID_FRAC` of the legs are still running and
  **dashed** once the survivors are a minority — the dashed part is still the truth about
  those legs, it is just no longer representative of the group;
* the inter-quartile band is drawn only over the solid part;
* the number of legs still running is drawn as a step curve along the bottom of the panel
  (the "number at risk" strip of a survival plot), so the reader can see the group empty.

Used by velocity/, acceleration/ and manipulability/; the
per-run analysis scripts carry their own copy of the same functions.
"""

import warnings

import numpy as np
from matplotlib.lines import Line2D  # noqa: I100,I201

SOLID_FRAC = 0.5
T_GRID_MAX = 600      # generous grid; the axis is then trimmed to the longest leg


def _n_twin(ax):
    tw = getattr(ax, '_n_twin_ax', None)
    if tw is None:
        tw = ax.twinx()
        tw.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
        tw.set_yticks([])
        for s in tw.spines.values():
            s.set_visible(False)
        ax._n_twin_ax = tw
        ax._n_max = 0
    return tw


def draw_n_at_risk(ax, grid, n, color):
    """Step curve of how many legs are still running, in the bottom fifth of the panel."""
    tw = _n_twin(ax)
    ax._n_max = max(ax._n_max, int(n.max()) if n.size else 0)
    tw.set_ylim(0, max(1, ax._n_max) * 5.0)
    alive = n >= 1
    if alive.any():
        end = int(np.flatnonzero(alive)[-1])
        tw.step(grid[:end + 1], n[:end + 1], where='post', color=color, linewidth=1.0,
                alpha=0.45, zorder=1)
    return tw


def median_over_legs(ax, grid, arr, color, label=None, band=True, linewidth=2.2,
                     n_at_risk=True, legs=False):
    """Median (and IQR) across legs over the full span, with the survival split.

    `arr` is (n_legs, len(grid)) with NaN once a leg has ended. With `legs=True` each leg
    is also drawn as a thin line underneath. Returns (n, t_end, t_solid_end).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return np.zeros(len(grid), int), 0.0, 0.0
    n = np.sum(~np.isnan(arr), axis=0)
    if not (n >= 1).any():
        return n, 0.0, 0.0
    alive = n >= 1
    solid = n >= max(2, int(np.ceil(SOLID_FRAC * n.max())))
    if legs:
        for row in arr:
            m = ~np.isnan(row)
            ax.plot(grid[m], row[m], '-', color=color, alpha=0.16, linewidth=0.8, zorder=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(arr, axis=0)
        lo = np.nanpercentile(arr, 25, axis=0)
        hi = np.nanpercentile(arr, 75, axis=0)
    ax.plot(grid[alive], med[alive], '--', color=color, linewidth=linewidth,
            dashes=(3, 2.2), alpha=0.9, zorder=3)
    if band:
        ax.fill_between(grid[solid], lo[solid], hi[solid], color=color, alpha=0.18,
                        linewidth=0, zorder=2)
    ax.plot(grid[solid], med[solid], '-', color=color, linewidth=linewidth, label=label,
            zorder=4)
    if n_at_risk:
        draw_n_at_risk(ax, grid, n, color)
    t_end = float(grid[np.flatnonzero(alive)[-1]])
    t_solid = float(grid[np.flatnonzero(solid)[-1]]) if solid.any() else 0.0
    return n, t_end, t_solid


def legend_handles(colors_labels, ink, french=True):
    """Run entries plus what solid / dashed / the bottom strip mean."""
    h = [Line2D([], [], color=c, linewidth=2.2, label=lab) for lab, c in colors_labels]
    if french:
        a = 'médiane, %d %% ou plus des jambes encore en cours' % int(SOLID_FRAC * 100)
        b = 'médiane, seule une minorité de jambes subsiste'
        c = 'bande du bas : nombre de jambes encore en cours'
    else:
        a = 'median, %d%%+ of legs still running' % int(SOLID_FRAC * 100)
        b = 'median, only a minority of legs left'
        c = 'bottom strip: number of legs still running'
    h += [Line2D([], [], color=ink, linewidth=2.2, label=a),
          Line2D([], [], color=ink, linewidth=2.2, linestyle='--', dashes=(3, 2.2), label=b),
          Line2D([], [], color=ink, linewidth=1, alpha=0.45, label=c)]
    return h
