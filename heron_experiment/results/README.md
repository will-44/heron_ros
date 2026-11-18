# results

One directory per run, created by the launch file (see `heron_experiment/paths.py`):

```
results/<experience>/<run_id>/      run_id = AAAAMMJJ_HHMMSS[_<run_tag>]
    raw/       what the nodes record: control_debug.csv, episodes_index.csv,
               phase_summary.csv, episode_<id>_<ts>/, maps/ (H5 debug dumps)
    tables/    what analysis/ derives from raw/
    figures/   run-level figures (the nodes' and analysis's)
    config/    the params files of the launch, and heron_package_manifest.yaml:
               which Heron Package (Experiment, checkpoint, grid) the run used
```

A new run never overwrites an old one: the run id starts with its date.

Studies over several runs write next to them (see `../analysis/README.md`):

```
results/<experience>/analysis/<study>/{tables,figures}/   several runs of one experience
results/analysis/<study>/{tables,figures}/                several experiences
```

The runs recorded before heron_ros (September 2026, the paper's four conditions) were
copied in as `setup_1/20260916_141632_base_fixe`, `setup_1/20260916_110958_base_mobile`,
`setup_2/20260918_174739_base_fixe`, `setup_2/20260921_100545_base_mobile` and
`tests/20260916_095446_final_test_5`, each named after the date of its first episode.

Only this README is in git; the runs themselves are not (they are large, and
specific to the robot). Copy them elsewhere to keep them.
