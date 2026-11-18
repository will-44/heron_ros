#!/usr/bin/env python3
r"""Full pick-and-place experiment: base controller + brain + grasp + release + recorder.

Every node of one run writes into the same timestamped `run_dir`, so a run is a
single self-contained directory rather than files scattered between `results/` and
whatever `debug_csv_path` happened to point at. Override it to name a run:

    ros2 launch heron_experiment gradient_base_controller.launch.py \\
        run_dir:=/home/ros2_ws/src/capacitynet/results/with_base_01 \\
        run_label:=with_base base_control:=true

`base_control:=false` is the control arm of the A/B: the controller node is still
launched (same GPU load, same instrumentation) but brain never enables it, so it
publishes only zeros. The flag is passed to the recorder too, which stamps it on
every index and phase-summary row -- that is what keeps the two arms separable
once the runs are pooled for analysis.
"""

from datetime import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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

RESULTS_ROOT = '/home/ros2_ws/src/capacitynet/results'


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

    # Computed once here rather than by each node, so the recorder's episode
    # directories and the controller's control_debug.csv land in the *same* run
    # directory and can be read together afterwards.
    default_run_dir = f"{RESULTS_ROOT}/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    declared.append(DeclareLaunchArgument(
        'run_dir', default_value=default_run_dir,
        description='Directory receiving every output of this run (one per experiment)'))
    declared.append(DeclareLaunchArgument(
        'run_label', default_value='',
        description='Free-text label stamped on every recorded row (e.g. with_base)'))
    declared.append(DeclareLaunchArgument(
        'base_control', default_value='true',
        description='Whether brain enables the base controller (A/B condition)'))

    run_dir = LaunchConfiguration('run_dir')
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
            {'debug_csv_path': [run_dir, '/control_debug.csv']},
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
            {'output_dir': run_dir,
             'run_label': LaunchConfiguration('run_label'),
             'base_control': base_control},
        ],
        remappings=tf_remappings,
    )

    return LaunchDescription(declared + [
        gradient_base_controller_node,
        brain_node,
        grasp_node,
        release_node,
        workspace_reachability_node,
    ])
