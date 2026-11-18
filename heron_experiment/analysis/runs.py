#!/usr/bin/env python3
"""The runs the analyses read, and where each analysis writes.

Every analysis script starts with

    sys.path.insert(0, <this directory>)
    import runs

which also puts heron_experiment (this source tree, no colcon build needed), setup_1/
and setup_2/ on the path, so the run loaders import by name.

Where things are (see heron_experiment.paths):

    results/<experience>/<run_id>/raw/            what the nodes recorded: read here
    results/<experience>/<run_id>/{tables,figures}/   what the analysis of that one run derives
    results/<experience>/analysis/<study>/        a study over several runs of one experience
    results/analysis/<study>/                     a study over several experiences

The analyses are not installed (setup.py does not see this directory), so unlike the
nodes they know where the source tree is: $HERON_EXPERIMENT_DIR defaults to it here.
"""

import os
import sys
from pathlib import Path
from typing import Callable, NamedTuple

ANALYSIS = Path(__file__).resolve().parent
PACKAGE_ROOT = ANALYSIS.parent
os.environ.setdefault('HERON_EXPERIMENT_DIR', str(PACKAGE_ROOT))
for _p in (ANALYSIS / 'setup_2', ANALYSIS / 'setup_1', PACKAGE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heron_experiment import paths  # noqa: E402

# The four conditions of the paper, as <experience>/<run_id>.
SETUP_1_FIXED = 'setup_1/20260916_141632_base_fixe'
SETUP_1_MOVING = 'setup_1/20260916_110958_base_mobile'
SETUP_2_FIXED = 'setup_2/20260918_174739_base_fixe'
SETUP_2_MOVING = 'setup_2/20260921_100545_base_mobile'

SETUP_LABEL = {'setup_1': 'setup 1 — table vide', 'setup_2': 'setup 2 — table encombrée'}
BASE_LABEL = {'fixed': 'base fixe', 'moving': 'base mobile'}


def run_dir(run) -> Path:
    """results/<experience>/<run_id>, from 'setup_1/20260916_...' or an absolute path."""
    run = Path(run)
    return run if run.is_absolute() else paths.results_dir() / run


def raw(run) -> Path:
    """What the nodes recorded for this run: control_debug.csv, episodes_index.csv, episode_*/."""
    return run_dir(run) / 'raw'


def study_dir(study, experience=None) -> Path:
    """results/[<experience>/]analysis/<study>: where a study over several runs writes its
    tables/ and figures/ (the study makes them)."""
    base = paths.results_dir() / experience if experience else paths.results_dir()
    return base / 'analysis' / study


class Condition(NamedTuple):
    setup: str          # 'setup_1' | 'setup_2'
    base: str           # 'fixed' | 'moving'
    run: str            # <experience>/<run_id>
    load: Callable      # raw dir -> the run's data dict (episode index, phases, control log, ...)
    module: object      # the module that analyses this run (analyze_exp1 or reach_analysis)

    @property
    def setup_label(self):
        return SETUP_LABEL[self.setup]

    @property
    def base_label(self):
        return BASE_LABEL[self.base]

    @property
    def raw(self):
        return raw(self.run)


def conditions():
    """The four conditions, in the paper's order: setup 1 fixed, setup 1 moving, setup 2 fixed, moving.

    Each run is loaded by its own loader, which carries that run's operator notes and
    outcome classes (see setup_1/ and setup_2/).
    """
    import analyze_exp1 as A
    import analyze_exp2 as E2
    import analyze_setup2 as S2
    import reach_analysis as R

    return [Condition('setup_1', 'fixed', SETUP_1_FIXED, E2.load, A),
            Condition('setup_1', 'moving', SETUP_1_MOVING, A.load, A),
            Condition('setup_2', 'fixed', SETUP_2_FIXED, S2.load_fixed, R),
            Condition('setup_2', 'moving', SETUP_2_MOVING, S2.load_moving, R)]
