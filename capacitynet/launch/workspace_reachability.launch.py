#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Topic remappings, not node parameters: WorkspaceRegionSource's TransformListener
# subscribes to the plain 'tf'/'tf_static' topics. This robot (leeloo) publishes
# its TF tree under the /leeloo namespace rather than the default /tf, /tf_static.
TF_REMAP_ARGUMENTS = [
    ('tf_topic', '/leeloo/tf', 'Topic carrying tf2_msgs/TFMessage for dynamic transforms'),
    ('tf_static_topic', '/leeloo/tf_static', 'Topic carrying tf2_msgs/TFMessage for static transforms'),
]


def generate_launch_description():
    """Launch the workspace reachability evaluation node."""

    declared = [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in TF_REMAP_ARGUMENTS
    ]

    # grasp.py's own params, passed through to the grasp node started alongside
    # the probe so it actually drives mpc_goal/mpc_predicted_path -- see its
    # header for how to override a single value.
    grasp_params_file_arg = DeclareLaunchArgument(
        'grasp_params_file',
        default_value='/home/ros2_ws/src/capacitynet/config/grasp_params.yaml',
        description='Params file for the grasp node',
    )
    declared.append(grasp_params_file_arg)

    tf_remappings = [
        ('tf', LaunchConfiguration('tf_topic')),
        ('tf_static', LaunchConfiguration('tf_static_topic')),
    ]


    model_config_path_arg = DeclareLaunchArgument(
        'model_config_path',
        default_value='/home/ros2_ws/src/capacitynet/config/test_reach.yaml',
        description='Path to the reachability model YAML config'
    )

    workspace_radius_arg = DeclareLaunchArgument(
        'workspace_radius',
        default_value='0.30',
        description='Workspace sphere radius in meters'
    )

    path_max_samples_arg = DeclareLaunchArgument(
        'path_max_samples',
        default_value='16',
        description='Max number of path poses evaluated per trajectory message (0 = all)'
    )

    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='',
        description='Directory for tests_index.csv and per-test subdirectories '
                    '(empty = timestamped default under capacitynet/results)'
    )

    log_each_sample_arg = DeclareLaunchArgument(
        'log_each_sample',
        default_value='true',
        description='Log every per-trajectory-message sample to the console'
    )

    generate_episode_plot_arg = DeclareLaunchArgument(
        'generate_episode_plot',
        default_value='true',
        description='Save a matplotlib PNG (Q_goal / Q_path over time) at the end of each test'
    )

    state_topic_arg = DeclareLaunchArgument(
        'state_topic',
        default_value='/object_grasper/state',
        description='std_msgs/String topic carrying the grasp state machine state; a test runs '
                    'from the moment this leaves idle_state_value until it returns to it'
    )

    idle_state_value_arg = DeclareLaunchArgument(
        'idle_state_value',
        default_value='idle',
        description='grasp.py state value meaning "no test in progress"'
    )

    joint_states_topic_arg = DeclareLaunchArgument(
        'joint_states_topic',
        default_value='/dsr01/joint_states',
        description='sensor_msgs/JointState topic logged for velocity/acceleration/manipulability'
    )

    mpc_diagnostics_topic_arg = DeclareLaunchArgument(
        'mpc_diagnostics_topic',
        default_value='/mpc_step_diagnostics',
        description='curobo_msgs/MpcStepDiagnostics topic logged for MPC solve timing/costs, '
                    'and the trigger for the per-cycle IK/FK-batch reachability check'
    )

    fk_batch_service_arg = DeclareLaunchArgument(
        'fk_batch_service',
        default_value='/curobo_trajectory_planner/fk_batch',
        description='curobo_msgs/FkBatch service used to validate IKDH-generated candidate configs'
    )

    ikdh_robot_yaml_arg = DeclareLaunchArgument(
        'ikdh_robot_yaml',
        default_value='doosan_robotics_m1013.yaml',
        description='IKDH bundled robot preset (DH + jacobian) used for IK and manipulability'
    )

    fk_batch_size_arg = DeclareLaunchArgument(
        'fk_batch_size',
        default_value='8',
        description='Fixed size of each FkBatch request; IKDH solutions are zero-padded to it'
    )

    arm_joint_names_arg = DeclareLaunchArgument(
        'arm_joint_names',
        default_value='[joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]',
        description='Doosan arm joint names, in DH/IKDH order, matched against joint_states_topic'
    )

    workspace_reachability_node = Node(
        package='capacitynet',
        executable='workspace_reachability_node',
        name='workspace_reachability_node',
        output='screen',
        parameters=[{
            'model_config_path': LaunchConfiguration('model_config_path'),
            'workspace_radius': LaunchConfiguration('workspace_radius'),
            'path_max_samples': LaunchConfiguration('path_max_samples'),
            'output_dir': LaunchConfiguration('output_dir'),
            'log_each_sample': LaunchConfiguration('log_each_sample'),
            'generate_episode_plot': LaunchConfiguration('generate_episode_plot'),
            'state_topic': LaunchConfiguration('state_topic'),
            'idle_state_value': LaunchConfiguration('idle_state_value'),
            'joint_states_topic': LaunchConfiguration('joint_states_topic'),
            'mpc_diagnostics_topic': LaunchConfiguration('mpc_diagnostics_topic'),
            'fk_batch_service': LaunchConfiguration('fk_batch_service'),
            'ikdh_robot_yaml': LaunchConfiguration('ikdh_robot_yaml'),
            'fk_batch_size': LaunchConfiguration('fk_batch_size'),
            'arm_joint_names': LaunchConfiguration('arm_joint_names'),
        }]
    )
    # Drives mpc_goal/mpc_predicted_path so the probe has something live to
    # score, instead of only ever working off a bag or static_workspace_center.
    grasp_node = Node(
        package='capacitynet',
        executable='grasp',
        name='object_grasper',
        output='screen',
        parameters=[LaunchConfiguration('grasp_params_file')],
        remappings=tf_remappings,
    )


    return LaunchDescription(declared +[
        model_config_path_arg,
        workspace_radius_arg,
        path_max_samples_arg,
        output_dir_arg,
        log_each_sample_arg,
        generate_episode_plot_arg,
        state_topic_arg,
        idle_state_value_arg,
        joint_states_topic_arg,
        mpc_diagnostics_topic_arg,
        fk_batch_service_arg,
        ikdh_robot_yaml_arg,
        fk_batch_size_arg,
        arm_joint_names_arg,
        workspace_reachability_node,
        grasp_node
    ])
