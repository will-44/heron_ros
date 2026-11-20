#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

CONFIG_DIR = os.path.join(get_package_share_directory('heron_experiment'), 'config')

# Only the region and inference knobs — the probe has no motion parameters
# because it has no BaseCommander, and so no way to command the base.
ARGUMENTS = [
    ('voxel_grid_topic', '/curobo_trajectory_planner/voxel_grid_sparse',
     'curobo_msgs/SparseVoxelGrid topic driving the evaluation loop'),
    ('cycle_period', '2.0',
     'Seconds between inference cycles; the voxel grid subscription only caches '
     'the latest message, a timer runs the cycle on it at this fixed cadence'),
    ('primary_marker_topic', '/kinect/fiducial_markers',
     'FiducialMarkerArray topic providing the fallback workspace center (primary camera)'),
    ('secondary_marker_topic', '/d405/fiducial_markers',
     'FiducialMarkerArray topic used once the primary camera has not seen the '
     'target marker for camera_failover_timeout seconds'),
    ('camera_failover_timeout', '1.0',
     'Seconds without a primary-camera detection before falling back to the secondary'),
    ('goal_topic', '/curobo_trajectory_planner/mpc_goal',
     'geometry_msgs/Pose grasp target published by grasp.py; primary region center'),
    ('path_topic', '/mpc_predicted_path',
     'nav_msgs/Path whose horizon point is unioned with the goal sphere'),
    ('path_horizon_s', '2.0,1.0',
     'Comma-separated seconds-from-now points to look up in the path (nearest '
     'header.stamp each), one sphere per entry; empty = goal only'),
    ('planning_frame', 'dsr01/base_link',
     'Frame assumed for goal_topic, which carries no header'),
    ('target_marker_id', '-1',
     'Marker id used as workspace center; -1 = first marker in the array'),
    ('heron_package', 'full_setup_20260805_1144',
     'Heron Package: a name under $HERON_EXPERIMENT_DIR/heron_packages, or a path'),
    ('precision', 'fp16',
     'Engine precision: fp32, fp16 or int8 (built beforehand by build_engine)'),

    ('workspace_radius', '0.10',
     'Goal sphere radius in meters (region center 0)'),
    ('path_radius', '0.10',
     'Path-tail sphere radius in meters (region centers 1+). Defaults to the '
     'same value as workspace_radius'),
    ('grid_spacing', '0.10',
     'Grid spacing delta in meters between candidate base positions'),
    ('grid_size', '3',
     'Number of candidate base positions per grid side (must be odd, >= 3); '
     'total candidates = grid_size**2 (3 -> 9, 5 -> 25, 7 -> 49). Cost scales '
     'linearly with candidate count — see OPTIMIZATIONS.md'),
    ('gradient_method', 'least_squares',
     "Gradient stencil: 'least_squares' (all grid_size**2 maps) or 'central' "
     '(4 nearest neighbors of center, regardless of grid_size)'),

    ('use_static_obstacles', 'false',
     'Load world-fixed obstacles from YAML so they are not translated with the base'),
    ('static_obstacles_yaml', os.path.join(CONFIG_DIR, 'floor_world.yml'),
     'Path to the static obstacles YAML'),

    ('marker_timeout', '2.0',
     'Seconds without a marker detection before the marker region is dropped'),
    ('mpc_timeout', '2.0',
     'Seconds without an MPC goal/path before falling back to the marker'),
    ('static_workspace_center', '',
     'Fixed workspace center "x,y,z" in the voxel grid frame, overriding both the '
     'MPC goal and the marker. Required for bag replay, since no recorded bag '
     'contains /fiducial_markers or /mpc_predicted_path'),

    ('log_timing', 'true',
     'Log gradient, Q and cycle time each evaluation cycle'),
    ('log_quality_scores', 'false',
     'Log the grid_size x grid_size block of quality scores each cycle'),
    ('publish_debug_markers', 'true',
     'Publish the region spheres, the grid_size**2 candidate scores and the '
     'gradient arrow'),
]

# Topic remappings, not node parameters: WorkspaceRegionSource's TransformListener
# subscribes to the plain 'tf'/'tf_static' topics. This robot (leeloo) publishes
# its TF tree under the /leeloo namespace rather than the default /tf, /tf_static.
TF_REMAP_ARGUMENTS = [
    ('tf_topic', '/leeloo/tf', 'Topic carrying tf2_msgs/TFMessage for dynamic transforms'),
    ('tf_static_topic', '/leeloo/tf_static', 'Topic carrying tf2_msgs/TFMessage for static transforms'),
]


def generate_launch_description():
    """Launch the reachability pipeline without any base actuation."""

    declared = [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in ARGUMENTS + TF_REMAP_ARGUMENTS
    ]

    # grasp.py's own params, passed through to the grasp node started alongside
    # the probe so it actually drives mpc_goal/mpc_predicted_path -- see its
    # header for how to override a single value.
    declared.append(DeclareLaunchArgument(
        'grasp_params_file',
        default_value=os.path.join(CONFIG_DIR, 'grasp_params.yaml'),
        description='Params file for the grasp node',
    ))

    parameters = {name: LaunchConfiguration(name) for name, _, _ in ARGUMENTS}

    tf_remappings = [
        ('tf', LaunchConfiguration('tf_topic')),
        ('tf_static', LaunchConfiguration('tf_static_topic')),
    ]

    workspace_probe_node = Node(
        package='heron_experiment',
        executable='workspace_probe',
        name='workspace_probe',
        output='screen',
        parameters=[parameters],
        remappings=tf_remappings,
    )

    # Drives mpc_goal/mpc_predicted_path so the probe has something live to
    # score, instead of only ever working off a bag or static_workspace_center.
    grasp_node = Node(
        package='heron_experiment',
        executable='grasp',
        name='object_grasper',
        output='screen',
        parameters=[LaunchConfiguration('grasp_params_file')],
        remappings=tf_remappings,
    )

    return LaunchDescription(declared + [workspace_probe_node, grasp_node])
