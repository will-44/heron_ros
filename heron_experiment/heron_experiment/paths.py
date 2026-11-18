#!/usr/bin/env python3
"""Where heron_experiment keeps what it reads and writes outside the installed package.

    <experiment dir>/heron_packages/<experiment>/      Heron Packages (heron export), engines built here
    <experiment dir>/results/<experience>/<run_id>/    one directory per run:
        raw/       what the nodes record (CSV, H5, episode directories)
        tables/    what analysis derives from raw/
        figures/   plots, the nodes' own and analysis's
        config/    the launch parameters and the Heron Package manifest the run used

The experiment dir is $HERON_EXPERIMENT_DIR, the heron_experiment source directory
(colcon copies the package on install, so it cannot be found from here). Both
folders are kept in git as a README only.
"""

import os
from datetime import datetime
from pathlib import Path

RUN_SUBDIRS = ('raw', 'tables', 'figures', 'config')


def experiment_dir() -> Path:
    """The heron_experiment source directory, from $HERON_EXPERIMENT_DIR."""
    root = os.environ.get('HERON_EXPERIMENT_DIR')
    if not root:
        raise RuntimeError(
            'HERON_EXPERIMENT_DIR is not set: export it as the heron_experiment source directory, '
            'e.g. /home/ros2_ws/src/heron_ros/heron_experiment')
    return Path(root).expanduser()


def results_dir() -> Path:
    return experiment_dir() / 'results'


def packages_dir() -> Path:
    return experiment_dir() / 'heron_packages'


def run_id(tag: str = '', now: datetime | None = None) -> str:
    """AAAAMMJJ_HHMMSS, then _<tag> if given: sorts by date, never reuses an old run's name."""
    stamp = (now or datetime.now()).strftime('%Y%m%d_%H%M%S')
    return f'{stamp}_{tag}' if tag else stamp


def new_run_dir(experience: str, tag: str = '', now: datetime | None = None) -> Path:
    """Create results/<experience>/<run_id>/ with its subdirectories, and return it."""
    run = results_dir() / experience / run_id(tag, now)
    for sub in RUN_SUBDIRS:
        (run / sub).mkdir(parents=True, exist_ok=True)
    return run
