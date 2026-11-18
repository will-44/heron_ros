#!/usr/bin/env python3
r"""Full pick-and-place experiment: base controller + brain + grasp + release + recorder.

Every node of one run writes into the same run directory,
results/<experience>/<run_id>/ (see heron_experiment.paths): the nodes record into
raw/, the run-level figures go to figures/, and config/ gets a copy of the params
files and of the Heron Package manifest the run used. Name the experience and tag
the run; earlier runs are never overwritten:

    ros2 launch heron_experiment gradient_base_controller.launch.py \\
        experience:=exp_1 run_tag:=with_base \\
        run_label:=with_base base_control:=true

`run_dir:=<path>` replaces the whole results/<experience>/<run_id> path.

`base_control:=false` is the control arm of the A/B: the controller node is still
launched (same GPU load, same instrumentation) but brain never enables it, so it
publishes only zeros. The flag is passed to the recorder too, which stamps it on
every index and phase-summary row -- that is what keeps the two arms separable
once the runs are pooled for analysis.
"""

import shutil
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os
from ament_index_python.packages import get_package_share_directory

CONFIG_DIR = os.path.join(get_package_share_directory('heron_experiment'), 'config')

# Topic remappings, not node parameters: WorkspaceRegionSource's TransformListener
# subscribes to the plain 'tf'/'tf_static' topics. This robot (leeloo) publishes
# its TF tree under the /leeloo namespace rather than the default /tf, /tf_static.
TF_REMAP_ARGUMENTS = [
    ('tf_topic', '/leeloo/tf', 'Topic carrying tf2_msgs/TFMessage for dynamic transforms'),
    ('tf_static_topic', '/leeloo/tf_static',
     'Topic carrying tf2_msgs/TFMessage for static transforms'),
]


def generate_launch_description():
    """Launch the reachability-gradient mobile base controller."""
    declared = [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in TF_REMAP_ARGUMENTS
    ]

    # gradient_base_controller and workspace_reachability_node's params (both the
    # ones each node uses on its own and the ones they share, e.g. goal_topic) live
    # in one yaml -- see its header for how to override a single value.
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(CONFIG_DIR, 'gradient_base_controller_params.yaml'),
        description='Params file for gradient_base_controller and workspace_reachability_node',
    )
    declared.append(params_file_arg)

    # grasp.py's own params, passed through to the grasp node started alongside
    # the controller so it actually drives mpc_goal/mpc_predicted_path and
    # grasper_state_topic (both consumed above) -- see its header for how to
    # override a single value.
    grasp_params_file_arg = DeclareLaunchArgument(
        'grasp_params_file',
        default_value=os.path.join(CONFIG_DIR, 'grasp_params.yaml'),
        description='Params file for the grasp node',
    )
    declared.append(grasp_params_file_arg)

    release_params_file_arg = DeclareLaunchArgument(
        'release_params_file',
        default_value=os.path.join(CONFIG_DIR, 'release_params.yaml'),
        description='Params file for the release node',
    )
    declared.append(release_params_file_arg)

    brain_params_file_arg = DeclareLaunchArgument(
        'brain_params_file',
        default_value=os.path.join(CONFIG_DIR, 'brain_params.yaml'),
        description='Params file for the brain node',
    )
    declared.append(brain_params_file_arg)

    declared.append(DeclareLaunchArgument(
        'experience', default_value='unnamed',
        description='Experience name: the run goes to results/<experience>/<run_id>/'))
    declared.append(DeclareLaunchArgument(
        'run_tag', default_value='',
        description='Suffix of the run id, after its date (e.g. with_base)'))
    declared.append(DeclareLaunchArgument(
        'run_dir', default_value='',
        description='Directory of this run, replacing results/<experience>/<run_id>'))
    declared.append(DeclareLaunchArgument(
        'run_label', default_value='',
        description='Free-text label stamped on every recorded row (e.g. with_base)'))
    declared.append(DeclareLaunchArgument(
        'base_control', default_value='true',
        description='Whether brain enables the base controller (A/B condition)'))

    return LaunchDescription(declared + [OpaqueFunction(function=_launch_run)])


def _prepare_run(context):
    """Create this run's directory once, and keep what configured it in its config/.

    Computed once here rather than by each node, so the recorder's episode
    directories and the controller's control_debug.csv land in the *same* run
    directory and can be read together afterwards.
    """
    from heron_experiment import paths
    from heron_experiment.core.reachability_model import package_path

    given = LaunchConfiguration('run_dir').perform(context)
    if given:
        run = Path(given).expanduser()
        for sub in paths.RUN_SUBDIRS:
            (run / sub).mkdir(parents=True, exist_ok=True)
    else:
        run = paths.new_run_dir(LaunchConfiguration('experience').perform(context),
                                LaunchConfiguration('run_tag').perform(context))
    params = {}
    for arg in ('params_file', 'grasp_params_file', 'release_params_file', 'brain_params_file'):
        path = Path(LaunchConfiguration(arg).perform(context))
        shutil.copy2(path, run / 'config' / path.name)
        if arg == 'params_file':
            params = yaml.safe_load(path.read_text()) or {}
    package = next((node.get('ros__parameters', {}).get('heron_package')
                    for node in params.values() if isinstance(node, dict)
                    and node.get('ros__parameters', {}).get('heron_package')), None)
    if package:
        manifest = package_path(package) / 'manifest.yaml'
        if manifest.is_file():
            shutil.copy2(manifest, run / 'config' / 'heron_package_manifest.yaml')
    return run


def _launch_run(context):
    run = _prepare_run(context)
    raw = str(run / 'raw')
    # Both nodes declare base_control as a bool; a bare LaunchConfiguration
    # substitutes to the string 'true' and would fail the type check.
    base_control = ParameterValue(LaunchConfiguration('base_control'), value_type=bool)
    tf_remappings = [
        ('tf', LaunchConfiguration('tf_topic')),
        ('tf_static', LaunchConfiguration('tf_static_topic')),
    ]

    # Later entries win, so these overrides take precedence over the yaml.
    gradient_base_controller_node = Node(
        package='heron_experiment',
        executable='gradient_base_controller',
        name='gradient_base_controller',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'debug_csv_path': raw + '/control_debug.csv'},
        ],
        remappings=tf_remappings,
    )

    # Drives mpc_goal/mpc_predicted_path/grasper_state_topic, which the
    # controller above consumes (region anchor + grasper-state interlock).
    grasp_node = Node(
        package='heron_experiment',
        executable='grasp',
        name='object_grasper',
        output='screen',
        parameters=[LaunchConfiguration('grasp_params_file')],
        remappings=tf_remappings,
    )

    # Named object_releaser to match the node's own default name (release.py) and
    # the release_state_topic the recorder subscribes to.
    release_node = Node(
        package='heron_experiment',
        executable='release',
        name='object_releaser',
        output='screen',
        parameters=[LaunchConfiguration('release_params_file')],
        remappings=tf_remappings,
    )

    # Sequences the whole cycle and owns the episode boundary the recorder keys on.
    brain_node = Node(
        package='heron_experiment',
        executable='brain',
        name='brain',
        output='screen',
        parameters=[
            LaunchConfiguration('brain_params_file'),
            {'base_control': base_control},
        ],
        remappings=tf_remappings,
    )

    # Per-episode reachability/manipulability/MPC-timing recorder, driven by the same
    # voxel grid / goal / path as the controller above and by brain_node's state topic.
    workspace_reachability_node = Node(
        package='heron_experiment',
        executable='workspace_reachability_node',
        name='workspace_reachability_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'output_dir': raw,
             'figures_dir': str(run / 'figures'),
             'run_label': LaunchConfiguration('run_label'),
             'base_control': base_control},
        ],
        remappings=tf_remappings,
    )

    return [
        gradient_base_controller_node,
        brain_node,
        grasp_node,
        release_node,
        workspace_reachability_node,
    ]
