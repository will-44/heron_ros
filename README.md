# heron_ros

Heron on the robot: reachability inference with a Heron Package, gradient-based
mobile base positioning, and the pick-and-place experiments of the thesis.

The network, its data and its training live in [Heron](https://github.com/Lab-CORO/Heron).
This repository only runs a network exported from there (see docs/adr/0001).

| Package | |
|---|---|
| `heron_experiment` | Nodes, controller, launch files, analysis; runs in `results/`, Heron Packages in `heron_packages/` |
| `heron_interface` | Actions: `TriggerGrasp`, `TriggerPickPlace`, `TriggerRelease` |

## Setup

In the robot's container (ROS 2 Jazzy, Jetson Orin):

```bash
pip install --no-deps -e /home/heron            # Heron, mounted next to the workspace
export HERON_EXPERIMENT_DIR=/home/ros2_ws/src/heron_ros/heron_experiment
cd /home/ros2_ws && colcon build --symlink-install --packages-select heron_interface heron_experiment
```

Then put a Heron Package in `heron_experiment/heron_packages/` and build its engines
(heron_experiment/heron_packages/README.md).

## Run an experience

```bash
ros2 launch heron_experiment gradient_base_controller.launch.py \
    experience:=exp_1 run_tag:=with_base base_control:=true
```

Everything the run records lands in `heron_experiment/results/exp_1/<date>_with_base/`.
`workspace_probe.launch.py` runs the same inference and scoring without any base
command, for bag replay.

## Analyse runs

```bash
cd heron_experiment
python3 analysis/setup_2/analyze_setup2.py          # per-run tables and figures, then the comparison
python3 analysis/paper_tables/paper_tables.py       # the paper's two tables
```

See [heron_experiment/analysis/README.md](heron_experiment/analysis/README.md).

Vocabulary: [GLOSSARY.md](GLOSSARY.md); decisions: [docs/adr/](docs/adr/).
