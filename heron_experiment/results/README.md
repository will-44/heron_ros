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

Only this README is in git; the runs themselves are not (they are large, and
specific to the robot). Copy them elsewhere to keep them.
