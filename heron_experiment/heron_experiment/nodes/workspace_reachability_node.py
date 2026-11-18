#!/usr/bin/env python3
"""Record one pick-and-place *episode* end-to-end: reachability, arm dynamics, MPC timing.

An "episode" is not a time window or a trajectory stream -- it is exactly the
interval during which `brain.py`'s state machine (via its latched ~/state topic)
is away from IDLE. That covers the whole cycle the article is about:

    act_home_enter -> act_grasp -> act_release -> act_home_exit -> idle

Keying on brain rather than on grasp alone is the difference between recording a
third of the cycle and recording all of it: `/object_grasper/state` returns to
idle the moment the object is in the gripper, so the place leg and both homings
used to fall outside every recorded interval.

Every row of every per-sample file carries the same five episode columns --
`episode_id`, `t_rel`, `phase`, `brain_state`, `sub_state` -- so the files are
joinable on `(episode_id, t_rel)` and groupable on `phase` without any further
processing. `phase` is the composite `brain_state` or `brain_state:sub_state`,
where `sub_state` is grasp.py's own state during `act_grasp` and release.py's
during `act_release` (the convention already used by
`heron_interface/action/TriggerPickPlace.action`'s feedback).

Layout, one directory per run (`output_dir`, normally set by the launch file so
that the controller's own `control_debug.csv` lands in the same place):

  episodes_index.csv     -- one summary row per episode, with its outcome
  phase_summary.csv      -- one row per (episode, phase): the A/B-ready table
  run_summary.png        -- 6-panel dashboard for the whole run
  base_trajectory.png    -- integrated base path, colored by Q
  episode_<id>_<ts>/
    samples.csv          -- Q_goal / Q_path per /mpc_predicted_path message
    joint_states.csv     -- /dsr01/joint_states: position, finite-diff velocity/
                            acceleration, and IKDH manipulability, per message
    mpc_diagnostics.csv  -- /mpc_step_diagnostics: solve time vs. its real-time
                            budget, arrival FK error, cost/constraint terms
    ik_fk_validity.csv   -- one row per IKDH-generated IK candidate for the current
                            mpc_goal, with the FkBatch-reported pose validity
    phases.csv           -- one row per phase traversed, with its duration
    cmd_vel.csv          -- base velocity commands and the integrated base position
    episode_<id>.png     -- Q_goal/Q_path over the episode, with phase bands

Builds a reachability map (RM) from /curobo_trajectory_planner/voxel_grid_sparse
messages (own ReachabilityEngine instance), scored as Q_goal (sphere on mpc_goal)
and Q_path (spheres over an /mpc_predicted_path subsample).

IK generation: on every /mpc_step_diagnostics message (i.e. every MPC control cycle)
during an episode, IKDH (https://github.com/Lab-CORO/IKDH, `pip install ikdh`)
computes *all* analytic IK solutions reaching the current mpc_goal -- up to 8 for
this 6R spherical-wrist arm. Those are zero-padded to exactly `fk_batch_size` joint
configs and sent to curobo_ros's FkBatch service, which reports per-candidate
validity (collision + joint-limit checked); only the real candidates get logged.

Base position (`x_est`, `y_est`, `dist_est` in cmd_vel.csv) is a piecewise-constant
integration of the commanded Twist, under two assumptions worth stating:
  (i)  the controller never commands a rotation (`base_commander.py` only ever sets
       `linear.x/y`), so integrating in the base frame gives the displacement up to
       one fixed unknown rotation;
  (ii) it is an open-loop estimate of the *command*, not odometry. If the robot
       publishes an `/odom`, adding it as a third `source` here makes the
       command-vs-actual comparison free.
"""

import csv
from datetime import datetime
import math
import os
import shutil
import time

import numpy as np

from curobo_msgs.msg import MpcStepDiagnostics, SparseVoxelGrid
from curobo_msgs.srv import FkBatch
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, String
from tf2_geometry_msgs import do_transform_pose_stamped
import tf2_ros
from tf2_ros import Buffer, TransformListener

from heron_experiment.core import experiment_plots
from heron_experiment.robot.ikdh_manipulability import IkdhModel
from heron_experiment.core.workspace_evaluation import WorkspaceEvaluation
from heron_experiment.legacy_engine.reachability_engine import ReachabilityEngine

# Prefix shared by every per-sample file. This is the uniformisation: whatever the
# signal, a row says which episode it belongs to, how far into it, and under which
# phase -- so the files join on (episode_id, t_rel) and group on phase.
_EPISODE_FIELDS = ['episode_id', 't_rel', 'phase', 'brain_state', 'sub_state']

_SAMPLE_CSV_FIELDS = ['timestamp'] + _EPISODE_FIELDS + [
    'seq', 'goal_x', 'goal_y', 'goal_z', 'goal_age_s', 'path_frame',
    'n_poses', 'n_eval', 'q_goal', 'q_path_mean', 'q_path_min', 'q_path_max',
    'rm_age_ms', 'compute_ms',
]

_MPC_DIAG_CSV_FIELDS = ['timestamp'] + _EPISODE_FIELDS + [
    'solve_ms', 'budget_ms', 'budget_exceeded',
    'arrival_fk_err_m', 'arrival_fk_rot_err_deg',
    'cost_tool_pose_pos', 'cost_tool_pose_orient', 'cost_cspace',
    'cost_cspace_vel', 'cost_cspace_acc', 'cost_cspace_jerk',
    'con_self_collision', 'con_scene_collision', 'con_cspace_bound',
]

_IK_FK_CSV_FIELDS = ['timestamp'] + _EPISODE_FIELDS + [
    'mpc_stamp', 'candidate_idx', 'manipulability', 'valid',
    'pose_x', 'pose_y', 'pose_z',
]

_CMD_VEL_CSV_FIELDS = ['timestamp'] + _EPISODE_FIELDS + [
    'source', 'vx', 'vy', 'wz', 'x_est', 'y_est', 'dist_est',
]

_PHASES_CSV_FIELDS = [
    'episode_id', 'phase', 'brain_state', 'sub_state',
    't_start_rel', 't_end_rel', 'duration_s', 'n_samples',
    'q_goal_mean', 'q_path_mean', 'q_path_min', 'q_path_max',
    'manipulability_mean', 'n_mpc_diagnostics', 'solve_ms_mean',
    'budget_exceeded_count', 'base_dist_est',
]

# Long-format, one row per (episode, phase), fixed schema whatever phases the run
# actually traversed -- this is the file the A/B analysis groups on. Identical to
# phases.csv plus the two run-identifying columns, so it is a plain concatenation
# of the per-episode files with no recomputation.
_PHASE_SUMMARY_CSV_FIELDS = ['run_label', 'base_control'] + _PHASES_CSV_FIELDS

_EPISODES_INDEX_CSV_FIELDS = [
    'episode_id', 't_start', 't_end', 'duration_s', 'dir',
    'outcome', 'outcome_message', 'run_label', 'base_control',
    'n_samples', 'rate_hz', 'q_goal_mean', 'q_path_mean', 'q_path_min', 'q_path_max',
    'goal_x', 'goal_y', 'goal_z',
    'n_joint_samples', 'manipulability_mean', 'manipulability_min', 'manipulability_max',
    'n_mpc_diagnostics', 'solve_ms_mean', 'solve_ms_max', 'budget_exceeded_count',
    'base_dist_est', 'n_phases',
]


def _mean(values):
    return float(np.mean(values)) if len(values) else 0.0


class _PhaseAccumulator:
    """Per-(episode, phase) statistics, closed when the phase changes."""

    def __init__(self, phase, brain_state, sub_state, t_start_rel, dist_at_start):
        self.phase = phase
        self.brain_state = brain_state
        self.sub_state = sub_state
        self.t_start_rel = t_start_rel
        self.dist_at_start = dist_at_start

        self.q_goal_values = []
        self.q_path_means = []
        self.q_path_min = float('inf')
        self.q_path_max = float('-inf')
        self.manipulability_values = []
        self.solve_ms_values = []
        self.budget_exceeded_count = 0

    def summarize(self, t_end_rel, dist_at_end):
        return {
            'phase': self.phase,
            'brain_state': self.brain_state,
            'sub_state': self.sub_state,
            't_start_rel': self.t_start_rel,
            't_end_rel': t_end_rel,
            'duration_s': max(t_end_rel - self.t_start_rel, 0.0),
            'n_samples': len(self.q_goal_values),
            'q_goal_mean': _mean(self.q_goal_values),
            'q_path_mean': _mean(self.q_path_means),
            'q_path_min': self.q_path_min if self.q_path_means else 0.0,
            'q_path_max': self.q_path_max if self.q_path_means else 0.0,
            'manipulability_mean': _mean(self.manipulability_values),
            'n_mpc_diagnostics': len(self.solve_ms_values),
            'solve_ms_mean': _mean(self.solve_ms_values),
            'budget_exceeded_count': self.budget_exceeded_count,
            'base_dist_est': dist_at_end - self.dist_at_start,
        }


class _EpisodeAccumulator:
    """State + open output files for one brain cycle (one IDLE -> ... -> IDLE run)."""

    def __init__(self, episode_id, t_start, episode_dir):
        self.episode_id = episode_id
        self.t_start = t_start
        self.dir = episode_dir

        self.n_samples = 0
        self.q_goal_values = []
        self.q_path_means = []
        self.q_path_min = float('inf')
        self.q_path_max = float('-inf')
        self.last_goal = None

        # Per-sample series (relative to t_start) kept for the end-of-episode plot.
        self.rel_times = []
        self.q_path_min_series = []
        self.q_path_max_series = []

        # (rel_time, phase) markers, replacing the old raw-state events.
        self.phase_events = []
        self.phase = None            # active _PhaseAccumulator
        self.n_phases = 0

        # Finite-difference basis for the joint_states callback: (t, position, velocity).
        self.last_joint_sample = None
        self.manipulability_values = []

        self.solve_ms_values = []
        self.budget_exceeded_count = 0

        # Open-loop integration of the commanded base velocity; see module docstring.
        self.x_est = 0.0
        self.y_est = 0.0
        self.dist_est = 0.0
        self._odom_v = (0.0, 0.0)
        self._odom_t = t_start

        # Set once the brain state returns to idle; the flush may still be waiting
        # for the outcome topic at that point (see `outcome_grace_s`).
        self.t_end = None
        self.closing = False

        self.samples_file = self.samples_writer = None
        self.joint_file = self.joint_writer = None
        self.mpc_file = self.mpc_writer = None
        self.ikfk_file = self.ikfk_writer = None
        self.phases_file = self.phases_writer = None
        self.cmd_vel_file = self.cmd_vel_writer = None

    # ── Phases ───────────────────────────────────────────────────────────────

    def open_phase(self, t_now, phase, brain_state, sub_state):
        self.phase = _PhaseAccumulator(
            phase, brain_state, sub_state, t_now - self.t_start, self.dist_est)
        self.phase_events.append((t_now - self.t_start, phase))

    def close_phase(self, t_now):
        """Write the active phase's row to phases.csv; returns its summary."""
        if self.phase is None:
            return None
        summary = self.phase.summarize(t_now - self.t_start, self.dist_est)
        self.phase = None
        self.n_phases += 1
        if self.phases_writer is not None:
            self.phases_writer.writerow(
                {'episode_id': self.episode_id,
                 **{k: summary[k] for k in _PHASES_CSV_FIELDS[1:]}})
            self.phases_file.flush()
        return summary

    # ── Sample accumulation ──────────────────────────────────────────────────

    def add_sample(self, t_now, q_goal, q_path_mean, q_path_min, q_path_max, goal):
        self.n_samples += 1
        self.q_goal_values.append(q_goal)
        self.q_path_means.append(q_path_mean)
        self.q_path_min = min(self.q_path_min, q_path_min)
        self.q_path_max = max(self.q_path_max, q_path_max)
        self.last_goal = goal

        self.rel_times.append(t_now - self.t_start)
        self.q_path_min_series.append(q_path_min)
        self.q_path_max_series.append(q_path_max)

        if self.phase is not None:
            self.phase.q_goal_values.append(q_goal)
            self.phase.q_path_means.append(q_path_mean)
            self.phase.q_path_min = min(self.phase.q_path_min, q_path_min)
            self.phase.q_path_max = max(self.phase.q_path_max, q_path_max)

    def add_manipulability(self, value):
        self.manipulability_values.append(value)
        if self.phase is not None:
            self.phase.manipulability_values.append(value)

    def add_mpc_diagnostics(self, solve_ms, budget_exceeded):
        self.solve_ms_values.append(solve_ms)
        if budget_exceeded:
            self.budget_exceeded_count += 1
        if self.phase is not None:
            self.phase.solve_ms_values.append(solve_ms)
            if budget_exceeded:
                self.phase.budget_exceeded_count += 1

    def advance_odometry(self, t_now):
        """Integrate the velocity held since the last command over the elapsed time.

        Piecewise-constant on purpose: a Twist stays in effect until the next one
        arrives, so the segment [t_prev, t_now) is travelled at the *previous*
        command, not the one just received.
        """
        dt = t_now - self._odom_t
        self._odom_t = t_now
        if dt <= 0.0:
            return
        vx, vy = self._odom_v
        self.x_est += vx * dt
        self.y_est += vy * dt
        self.dist_est += math.hypot(vx, vy) * dt

    def set_odometry_velocity(self, vx, vy):
        self._odom_v = (vx, vy)

    def summarize(self, t_end, outcome, outcome_message, run_label, base_control):
        duration = max(t_end - self.t_start, 1e-9)
        goal = self.last_goal or (0.0, 0.0, 0.0)
        return {
            'episode_id': self.episode_id,
            't_start': self.t_start,
            't_end': t_end,
            'duration_s': duration,
            'dir': self.dir,
            'outcome': outcome,
            'outcome_message': outcome_message,
            'run_label': run_label,
            'base_control': base_control,
            'n_samples': self.n_samples,
            'rate_hz': self.n_samples / duration,
            'q_goal_mean': _mean(self.q_goal_values),
            'q_path_mean': _mean(self.q_path_means),
            'q_path_min': self.q_path_min if self.q_path_means else 0.0,
            'q_path_max': self.q_path_max if self.q_path_means else 0.0,
            'goal_x': goal[0], 'goal_y': goal[1], 'goal_z': goal[2],
            'n_joint_samples': len(self.manipulability_values),
            'manipulability_mean': _mean(self.manipulability_values),
            'manipulability_min': (
                float(np.min(self.manipulability_values))
                if self.manipulability_values else 0.0),
            'manipulability_max': (
                float(np.max(self.manipulability_values))
                if self.manipulability_values else 0.0),
            'n_mpc_diagnostics': len(self.solve_ms_values),
            'solve_ms_mean': _mean(self.solve_ms_values),
            'solve_ms_max': float(np.max(self.solve_ms_values)) if self.solve_ms_values else 0.0,
            'budget_exceeded_count': self.budget_exceeded_count,
            'base_dist_est': self.dist_est,
            'n_phases': self.n_phases,
        }


class WorkspaceReachabilityNode(Node):
    def __init__(self):
        super().__init__('workspace_reachability_node')

        self._load_parameters()
        self._setup_engine()
        self._setup_ikdh()
        self._setup_csv()
        self._setup_interfaces()

        # Latest reachability map + the voxel-grid metadata it was built from.
        self._rm = None
        self._vg_info = None
        self._vg_frame = None
        self._rm_stamp = None
        self._last_rm_wall_t = 0.0

        self._goal = None            # (x, y, z) in self._vg_frame, or None
        self._goal_pose = None       # full geometry_msgs/Pose (for IKDH), or None
        self._goal_stamp = None      # wall clock of the last mpc_goal, for goal_age_s
        self._goal_eval = None       # cached WorkspaceEvaluation for the current goal
        self._goal_eval_center = None

        self._brain_state = None
        self._grasp_state = self.idle_state_value
        self._release_state = self.idle_state_value

        # Latest ("<outcome>|<message>", reception time) from the brain outcome topic.
        self._pending_outcome = None
        self._outcome_timer = None

        self._episode = None         # active _EpisodeAccumulator, or None while idle
        self._next_episode_id = 0
        self._seq = 0

        self._log_configuration()

    def _log_configuration(self):
        log = self.get_logger()
        log.info('Workspace reachability / episode recorder initialized')
        log.info(f'  - Voxel grid topic: {self.voxel_grid_topic}')
        log.info(f'  - Goal topic: {self.goal_topic}')
        log.info(f'  - Path topic: {self.path_topic}')
        log.info(
            f'  - Episode boundary: {self.episode_state_topic} '
            f'(idle value: {self.idle_state_value!r})')
        log.info(
            f'  - Sub-states: grasp={self.grasp_state_topic} '
            f'release={self.release_state_topic}')
        log.info(f'  - Outcome topic: {self.outcome_topic} (grace {self.outcome_grace_s:.2f}s)')
        log.info(
            f'  - cmd_vel: controller={self.cmd_vel_topic} '
            f'base={self.base_cmd_vel_topic or "(none)"} '
            f'(position integrated from {self._integrate_source!r})')
        log.info(f'  - Joint states topic: {self.joint_states_topic}')
        log.info(f'  - MPC diagnostics topic: {self.mpc_diagnostics_topic}')
        log.info(
            f'  - FK batch service: {self.fk_batch_service} (batch size {self.fk_batch_size})')
        log.info(f'  - Workspace radius: {self.workspace_radius} m')
        if self.rm_min_period_s > 0.0:
            log.info(f'  - RM inference throttled to one every {self.rm_min_period_s:.2f}s')
        log.info(
            f'  - Run: label={self.run_label!r} base_control={self.base_control} '
            f'-> {self.output_dir}')

    # ── Setup ────────────────────────────────────────────────────────────────

    def _load_parameters(self):
        self.declare_parameter(
            'model_config_path', '/home/ros2_ws/src/capacitynet/config/test_reach.yaml')
        self.declare_parameter('fp16', False)

        self.declare_parameter('voxel_grid_topic', '/curobo_trajectory_planner/voxel_grid_sparse')
        self.declare_parameter('goal_topic', '/curobo_trajectory_planner/mpc_goal')
        self.declare_parameter('path_topic', '/mpc_predicted_path')

        # Episode boundary: brain's own state, not the grasper's. See module docstring.
        self.declare_parameter('episode_state_topic', '/brain/state')
        self.declare_parameter('grasp_state_topic', '/object_grasper/state')
        self.declare_parameter('release_state_topic', '/object_releaser/state')
        self.declare_parameter('outcome_topic', '/brain/outcome')
        self.declare_parameter('idle_state_value', 'idle')

        # Twist topics. The controller publishes on `cmd_vel_topic`; an external
        # relay (outside this workspace) forwards it to `base_cmd_vel_topic`,
        # which is what the base actually listens to. Both are logged: a run where
        # the controller source is non-zero while the base source stays at zero is
        # a dead relay -- a failure that is invisible after the fact otherwise.
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('base_cmd_vel_topic', '/r100_0597/joy_teleop/cmd_vel')

        self.declare_parameter('joint_states_topic', '/dsr01/joint_states')
        self.declare_parameter('mpc_diagnostics_topic', '/mpc_step_diagnostics')
        self.declare_parameter('planning_frame', 'dsr01/base_link')

        self.declare_parameter('fk_batch_service', '/curobo_trajectory_planner/fk_batch')
        self.declare_parameter('ikdh_robot_yaml', 'doosan_robotics_m1013.yaml')
        self.declare_parameter('fk_batch_size', 8)
        self.declare_parameter(
            'arm_joint_names',
            ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])

        self.declare_parameter('workspace_radius', 0.30)
        self.declare_parameter('path_max_samples', 16)

        # Seconds to wait for the outcome topic after the brain returns to idle:
        # state and outcome are two topics with no relative ordering guarantee.
        self.declare_parameter('outcome_grace_s', 0.5)
        # Episodes shorter than this are discarded, directory and all. Opt-in
        # (0.0 keeps everything); it is what removes the few-millisecond spurious
        # episodes a state topic can produce when it flickers through idle.
        self.declare_parameter('min_episode_duration_s', 0.0)
        # Minimum wall-clock seconds between two reachability inferences. This node
        # runs a *second* UNet3D on the same Orin as the controller; when the run is
        # measuring real-time performance, throttle it here. 0.0 = infer on every
        # voxel-grid message (previous behaviour).
        self.declare_parameter('rm_min_period_s', 0.0)

        # Written verbatim into every index/summary row so a run stays identifiable
        # and the A/B arms stay separable after the fact.
        self.declare_parameter('run_label', '')
        self.declare_parameter('base_control', True)

        self.declare_parameter('output_dir', '')
        self.declare_parameter('log_each_sample', True)
        self.declare_parameter('generate_episode_plot', True)
        self.declare_parameter('generate_run_plots', True)

        self.model_config_path = self.get_parameter('model_config_path').value
        self.fp16 = bool(self.get_parameter('fp16').value)

        self.voxel_grid_topic = self.get_parameter('voxel_grid_topic').value
        self.goal_topic = self.get_parameter('goal_topic').value
        self.path_topic = self.get_parameter('path_topic').value

        self.episode_state_topic = self.get_parameter('episode_state_topic').value
        self.grasp_state_topic = self.get_parameter('grasp_state_topic').value
        self.release_state_topic = self.get_parameter('release_state_topic').value
        self.outcome_topic = self.get_parameter('outcome_topic').value
        self.idle_state_value = self.get_parameter('idle_state_value').value

        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.base_cmd_vel_topic = self.get_parameter('base_cmd_vel_topic').value
        # With no distinct downstream topic there is only one source to integrate.
        self._integrate_source = (
            'base' if self.base_cmd_vel_topic
            and self.base_cmd_vel_topic != self.cmd_vel_topic else 'controller')

        self.joint_states_topic = self.get_parameter('joint_states_topic').value
        self.mpc_diagnostics_topic = self.get_parameter('mpc_diagnostics_topic').value
        self.planning_frame = self.get_parameter('planning_frame').value

        self.fk_batch_service = self.get_parameter('fk_batch_service').value
        self.ikdh_robot_yaml = self.get_parameter('ikdh_robot_yaml').value
        self.fk_batch_size = int(self.get_parameter('fk_batch_size').value)
        self.arm_joint_names = list(self.get_parameter('arm_joint_names').value)

        self.workspace_radius = float(self.get_parameter('workspace_radius').value)
        self.path_max_samples = int(self.get_parameter('path_max_samples').value)

        self.outcome_grace_s = float(self.get_parameter('outcome_grace_s').value)
        self.min_episode_duration_s = float(self.get_parameter('min_episode_duration_s').value)
        self.rm_min_period_s = float(self.get_parameter('rm_min_period_s').value)

        self.run_label = self.get_parameter('run_label').value
        self.base_control = bool(self.get_parameter('base_control').value)

        self.log_each_sample = bool(self.get_parameter('log_each_sample').value)
        self.generate_episode_plot = bool(self.get_parameter('generate_episode_plot').value)
        self.generate_run_plots = bool(self.get_parameter('generate_run_plots').value)

        output_dir = self.get_parameter('output_dir').value
        if not output_dir:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f'/home/ros2_ws/src/capacitynet/results/run_{ts}'
        self.output_dir = os.path.expanduser(output_dir)

    def _setup_engine(self):
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Loading reachability model on {self.device}...')
        self.engine = ReachabilityEngine(self.model_config_path, fp16=self.fp16)
        self.get_logger().info('Reachability model loaded')

    def _setup_ikdh(self):
        self.get_logger().info(f'Loading IKDH robot model {self.ikdh_robot_yaml!r}...')
        self._ikdh = IkdhModel(self.ikdh_robot_yaml)
        self.get_logger().info('IKDH robot model loaded')

    def _setup_csv(self):
        os.makedirs(self.output_dir, exist_ok=True)

        self._index_path = os.path.join(self.output_dir, 'episodes_index.csv')
        self._index_file = open(self._index_path, 'w', newline='')
        self._index_writer = csv.DictWriter(
            self._index_file, fieldnames=_EPISODES_INDEX_CSV_FIELDS)
        self._index_writer.writeheader()
        self._index_file.flush()

        self._phase_summary_path = os.path.join(self.output_dir, 'phase_summary.csv')
        self._phase_summary_file = open(self._phase_summary_path, 'w', newline='')
        self._phase_summary_writer = csv.DictWriter(
            self._phase_summary_file, fieldnames=_PHASE_SUMMARY_CSV_FIELDS)
        self._phase_summary_writer.writeheader()
        self._phase_summary_file.flush()

    def _setup_interfaces(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            SparseVoxelGrid, self.voxel_grid_topic, self.on_voxel_grid, 10)
        self.create_subscription(
            Pose, self.goal_topic, self.on_goal, 10)

        path_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Path, self.path_topic, self.on_path, path_qos)

        # Matches the state_pub QoS of brain.py/grasp.py/release.py (TRANSIENT_LOCAL)
        # so we pick up the current state immediately on startup instead of waiting
        # for the next transition.
        state_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, self.episode_state_topic, self.on_brain_state, state_qos)
        self.create_subscription(
            String, self.grasp_state_topic, self.on_grasp_state, state_qos)
        self.create_subscription(
            String, self.release_state_topic, self.on_release_state, state_qos)
        self.create_subscription(
            String, self.outcome_topic, self.on_outcome, state_qos)

        self.create_subscription(
            Twist, self.cmd_vel_topic,
            lambda msg: self.on_cmd_vel(msg, 'controller'), 10)
        if self._integrate_source == 'base':
            self.create_subscription(
                Twist, self.base_cmd_vel_topic,
                lambda msg: self.on_cmd_vel(msg, 'base'), 10)

        self.create_subscription(
            JointState, self.joint_states_topic, self.on_joint_states, 10)
        self.create_subscription(
            MpcStepDiagnostics, self.mpc_diagnostics_topic, self.on_mpc_diagnostics, 10)

        self.fk_batch_client = self.create_client(FkBatch, self.fk_batch_service)

        self.q_goal_pub = self.create_publisher(Float32, '/workspace_reachability/q_goal', 10)
        self.q_path_mean_pub = self.create_publisher(
            Float32, '/workspace_reachability/q_path_mean', 10)
        self.episode_q_goal_pub = self.create_publisher(
            Float32, '/workspace_reachability/test_q_goal_mean', 10)
        self.episode_q_path_pub = self.create_publisher(
            Float32, '/workspace_reachability/test_q_path_mean', 10)

    # ── Phase tracking ───────────────────────────────────────────────────────

    def _current_phase(self):
        """(phase, brain_state, sub_state) for the states seen so far.

        `sub_state` is the state of whichever sub-machine brain has delegated to;
        outside act_grasp/act_release there is none and the phase is the brain
        state alone.
        """
        brain_state = self._brain_state or self.idle_state_value
        if brain_state == 'act_grasp':
            sub_state = self._grasp_state or ''
        elif brain_state == 'act_release':
            sub_state = self._release_state or ''
        else:
            sub_state = ''
        phase = f'{brain_state}:{sub_state}' if sub_state else brain_state
        return phase, brain_state, sub_state

    def _refresh_phase(self, t_now=None):
        """Close the active phase and open the current one, if they differ."""
        episode = self._episode
        if episode is None or episode.closing:
            return
        phase, brain_state, sub_state = self._current_phase()
        if episode.phase is not None and episode.phase.phase == phase:
            return
        t_now = time.time() if t_now is None else t_now
        episode.close_phase(t_now)
        episode.open_phase(t_now, phase, brain_state, sub_state)

    def _row_prefix(self, t_now=None):
        """Build the `timestamp` + episode columns every per-sample row starts with."""
        t_now = time.time() if t_now is None else t_now
        episode = self._episode
        phase, brain_state, sub_state = self._current_phase()
        return {
            'timestamp': t_now,
            'episode_id': episode.episode_id if episode is not None else -1,
            't_rel': (t_now - episode.t_start) if episode is not None else 0.0,
            'phase': phase,
            'brain_state': brain_state,
            'sub_state': sub_state,
        }

    # ── Episode lifecycle (gated on the brain state != idle) ────────────────

    def on_brain_state(self, msg: String):
        """Open, close or re-phase the current episode on a brain state change."""
        new_state = msg.data
        if new_state == self._brain_state:
            return
        prev_state = self._brain_state
        self._brain_state = new_state

        prev_was_idle = prev_state is None or prev_state == self.idle_state_value
        now_is_idle = new_state == self.idle_state_value

        if prev_was_idle and not now_is_idle:
            self._start_episode()
            self._refresh_phase()
        elif not prev_was_idle and now_is_idle:
            self._end_episode()
        else:
            self._refresh_phase()

    def on_grasp_state(self, msg: String):
        """Track the grasp sub-state feeding the composite phase."""
        if msg.data == self._grasp_state:
            return
        self._grasp_state = msg.data
        self._refresh_phase()

    def on_release_state(self, msg: String):
        """Track the release sub-state feeding the composite phase."""
        if msg.data == self._release_state:
            return
        self._release_state = msg.data
        self._refresh_phase()

    def on_outcome(self, msg: String):
        """Record how the cycle finished; see brain.py's outcome_pub.

        Format is "<outcome>|<message>". The outcome normally arrives just before
        the idle state, in which case `_end_episode` finds it here and flushes
        without waiting.
        """
        outcome, _, message = msg.data.partition('|')
        self._pending_outcome = (outcome.strip() or 'unknown', message, time.time())

        # Arriving during the grace window: flush now instead of at the timeout.
        if self._episode is not None and self._episode.closing:
            self._cancel_outcome_timer()
            self._flush_episode()

    def _start_episode(self):
        if self._episode is not None:
            self._cancel_outcome_timer()
            self._flush_episode()

        now = time.time()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        episode_dir = os.path.join(
            self.output_dir, f'episode_{self._next_episode_id:03d}_{ts}')
        os.makedirs(episode_dir, exist_ok=True)

        episode = _EpisodeAccumulator(self._next_episode_id, now, episode_dir)
        self._next_episode_id += 1
        self._open_episode_csvs(episode)
        self._episode = episode
        self._seq = 0

        self.get_logger().info(f'Episode {episode.episode_id} started -> {episode_dir}')

    def _open_episode_csvs(self, episode):
        def _open(name, fields):
            handle = open(os.path.join(episode.dir, name), 'w', newline='')
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            handle.flush()
            return handle, writer

        episode.samples_file, episode.samples_writer = _open('samples.csv', _SAMPLE_CSV_FIELDS)
        episode.joint_file, episode.joint_writer = _open(
            'joint_states.csv', self._joint_csv_fields())
        episode.mpc_file, episode.mpc_writer = _open(
            'mpc_diagnostics.csv', _MPC_DIAG_CSV_FIELDS)
        episode.ikfk_file, episode.ikfk_writer = _open(
            'ik_fk_validity.csv', self._ikfk_csv_fields())
        episode.phases_file, episode.phases_writer = _open('phases.csv', _PHASES_CSV_FIELDS)
        episode.cmd_vel_file, episode.cmd_vel_writer = _open('cmd_vel.csv', _CMD_VEL_CSV_FIELDS)

    def _episode_files(self, episode):
        return (episode.samples_file, episode.joint_file, episode.mpc_file,
                episode.ikfk_file, episode.phases_file, episode.cmd_vel_file)

    def _joint_csv_fields(self):
        fields = ['timestamp'] + list(_EPISODE_FIELDS)
        for jn in self.arm_joint_names:
            fields += [f'pos_{jn}', f'vel_{jn}', f'acc_{jn}']
        fields.append('manipulability')
        return fields

    def _ikfk_csv_fields(self):
        fields = list(_IK_FK_CSV_FIELDS)
        for jn in self.arm_joint_names:
            fields.append(f'q_{jn}')
        return fields

    def _end_episode(self):
        """Brain is back at idle: stop recording, then flush once the outcome is known."""
        episode = self._episode
        if episode is None:
            return

        now = time.time()
        episode.advance_odometry(now)
        episode.close_phase(now)
        episode.t_end = now
        episode.closing = True

        if self._outcome_is_fresh(episode):
            self._flush_episode()
            return

        # /brain/state and /brain/outcome are separate topics with no relative
        # ordering guarantee. Give the outcome a moment before giving up on it.
        self._cancel_outcome_timer()
        self._outcome_timer = self.create_timer(self.outcome_grace_s, self._on_outcome_timeout)

    def _outcome_is_fresh(self, episode):
        """Report whether an outcome was received after this episode began.

        The topic is latched, so a subscriber may hold the *previous* cycle's
        outcome from before the episode even started -- reporting that one would
        be worse than reporting nothing.
        """
        return self._pending_outcome is not None and self._pending_outcome[2] >= episode.t_start

    def _on_outcome_timeout(self):
        self._cancel_outcome_timer()
        self._flush_episode()

    def _cancel_outcome_timer(self):
        if self._outcome_timer is not None:
            self._outcome_timer.cancel()
            self.destroy_timer(self._outcome_timer)
            self._outcome_timer = None

    def _flush_episode(self, t_end=None):
        episode = self._episode
        self._episode = None
        if episode is None:
            return

        if t_end is None:
            t_end = episode.t_end if episode.t_end is not None else time.time()
        if not episode.closing:
            episode.advance_odometry(t_end)
            episode.close_phase(t_end)

        outcome, outcome_message = 'unknown', ''
        if self._outcome_is_fresh(episode):
            outcome, outcome_message, _ = self._pending_outcome
        self._pending_outcome = None

        for f in self._episode_files(episode):
            if f is not None:
                f.close()

        summary = episode.summarize(
            t_end, outcome, outcome_message, self.run_label, self.base_control)

        if summary['duration_s'] < self.min_episode_duration_s:
            self.get_logger().warn(
                f"Episode {episode.episode_id} lasted {summary['duration_s']:.3f}s "
                f'(< min_episode_duration_s={self.min_episode_duration_s}); discarding '
                f'{episode.dir}')
            shutil.rmtree(episode.dir, ignore_errors=True)
            return

        self._index_writer.writerow(summary)
        self._index_file.flush()
        self._append_phase_summary(episode)

        if self.generate_episode_plot and episode.n_samples > 0:
            self._plot_episode(episode, summary)

        msg_goal = Float32()
        msg_goal.data = summary['q_goal_mean']
        self.episode_q_goal_pub.publish(msg_goal)

        msg_path = Float32()
        msg_path.data = summary['q_path_mean']
        self.episode_q_path_pub.publish(msg_path)

        self.get_logger().info(
            f"Episode {summary['episode_id']} done [{summary['outcome']}]: "
            f"n={summary['n_samples']} "
            f"({summary['rate_hz']:.1f} Hz over {summary['duration_s']:.1f}s, "
            f"{summary['n_phases']} phases) | "
            f"Q_goal_mean={summary['q_goal_mean']:.4f} "
            f"Q_path_mean={summary['q_path_mean']:.4f} "
            f"[{summary['q_path_min']:.4f}, {summary['q_path_max']:.4f}] | "
            f"manipulability_mean={summary['manipulability_mean']:.4f} | "
            f"solve_ms_mean={summary['solve_ms_mean']:.2f} "
            f"budget_exceeded={summary['budget_exceeded_count']}/{summary['n_mpc_diagnostics']} | "
            f"base_dist_est={summary['base_dist_est']:.3f} m"
        )

    def _append_phase_summary(self, episode):
        """Copy this episode's phases.csv into the run-level phase_summary.csv.

        phases.csv is already written and closed at this point; re-reading it is
        what keeps a single source of truth for the per-phase statistics.
        """
        phases_path = os.path.join(episode.dir, 'phases.csv')
        try:
            with open(phases_path, newline='') as handle:
                rows = list(csv.DictReader(handle))
        except OSError as ex:
            self.get_logger().warn(f'Cannot read {phases_path}: {ex}')
            return
        for row in rows:
            out = {k: row.get(k, '') for k in _PHASE_SUMMARY_CSV_FIELDS}
            out['episode_id'] = episode.episode_id
            out['run_label'] = self.run_label
            out['base_control'] = self.base_control
            self._phase_summary_writer.writerow(out)
        self._phase_summary_file.flush()

    # ── Reachability sample callbacks (gated on being in an episode) ────────

    def on_voxel_grid(self, msg: SparseVoxelGrid):
        now = time.time()
        if self.rm_min_period_s > 0.0 and (now - self._last_rm_wall_t) < self.rm_min_period_s:
            return
        self._last_rm_wall_t = now

        voxel_map = self.engine.preprocess(
            msg.occupied_indices, msg.size_x, msg.size_y, msg.size_z)
        self._rm = self.engine.predict(voxel_map)
        self._vg_info = {
            'origin': msg.origin,
            'resolution': float(msg.resolution),
            'size_x': msg.size_x, 'size_y': msg.size_y, 'size_z': msg.size_z,
        }
        self._vg_frame = msg.header.frame_id
        self._rm_stamp = self.get_clock().now()

    def on_goal(self, msg: Pose):
        # mpc_goal carries no header/frame; assume planning_frame (see grasp.py, where
        # this Pose is always expressed in the same frame passed to the trajectory action).
        self._goal = (msg.position.x, msg.position.y, msg.position.z)
        self._goal_pose = msg
        self._goal_stamp = time.time()

    def on_path(self, msg: Path):
        t_start = time.time()

        if self._episode is None or self._episode.closing:
            return
        if self._rm is None:
            self.get_logger().warn(
                'No reachability map yet, skipping trajectory evaluation',
                throttle_duration_sec=5.0)
            return
        if self._goal is None:
            self.get_logger().warn(
                'No mpc_goal received yet, skipping trajectory evaluation',
                throttle_duration_sec=5.0)
            return
        if not msg.poses:
            return

        now = time.time()
        centers = self._extract_path_centers(msg)
        if not centers:
            return

        goal_eval = self._get_goal_evaluation(self._goal)
        q_goal = goal_eval.compute_quality_score(self._rm, **self._vg_info)

        q_path_values = WorkspaceEvaluation.compute_quality_scores_batched(
            self._rm, centers, radius=self.workspace_radius, device=self.device,
            **self._vg_info)
        q_path_mean = float(np.mean(q_path_values))
        q_path_min = float(np.min(q_path_values))
        q_path_max = float(np.max(q_path_values))

        compute_ms = (time.time() - t_start) * 1000.0
        rm_age_ms = (self.get_clock().now() - self._rm_stamp).nanoseconds / 1e6

        self._publish_and_log_sample(
            msg, centers, q_goal, q_path_mean, q_path_min, q_path_max,
            rm_age_ms, compute_ms)

        self._episode.add_sample(now, q_goal, q_path_mean, q_path_min, q_path_max, self._goal)
        self._seq += 1

    # ── Base velocity + open-loop position (only while in an episode) ───────

    def on_cmd_vel(self, msg: Twist, source: str):
        episode = self._episode
        if episode is None or episode.closing:
            return

        now = time.time()
        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z

        if source == self._integrate_source:
            episode.advance_odometry(now)
            episode.set_odometry_velocity(vx, vy)

        row = self._row_prefix(now)
        row.update({
            'source': source, 'vx': vx, 'vy': vy, 'wz': wz,
            'x_est': episode.x_est, 'y_est': episode.y_est, 'dist_est': episode.dist_est,
        })
        episode.cmd_vel_writer.writerow(row)
        episode.cmd_vel_file.flush()

    # ── Joint-state dynamics + manipulability (only while in an episode) ────

    def on_joint_states(self, msg: JointState):
        episode = self._episode
        if episode is None or episode.closing:
            return

        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        try:
            idx = [name_to_idx[name] for name in self.arm_joint_names]
        except KeyError as missing:
            self.get_logger().warn(
                f'{self.joint_states_topic} is missing joint {missing}',
                throttle_duration_sec=5.0)
            return

        positions = np.array([msg.position[i] for i in idx])  # radians

        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        if t == 0.0:
            t = time.time()

        velocities = np.zeros(len(idx))
        accelerations = np.zeros(len(idx))
        if episode.last_joint_sample is not None:
            prev_t, prev_positions, prev_velocities = episode.last_joint_sample
            dt = t - prev_t
            if dt > 1e-6:
                velocities = (positions - prev_positions) / dt
                accelerations = (velocities - prev_velocities) / dt
        episode.last_joint_sample = (t, positions, velocities)

        manipulability = self._ikdh.manipulability(np.degrees(positions))
        episode.add_manipulability(manipulability)

        row = self._row_prefix()
        for i, jn in enumerate(self.arm_joint_names):
            row[f'pos_{jn}'] = positions[i]
            row[f'vel_{jn}'] = velocities[i]
            row[f'acc_{jn}'] = accelerations[i]
        row['manipulability'] = manipulability
        episode.joint_writer.writerow(row)
        episode.joint_file.flush()

    # ── MPC diagnostics + IK/FK-batch reachable-config check ───────────────

    def on_mpc_diagnostics(self, msg: MpcStepDiagnostics):
        episode = self._episode
        if episode is None or episode.closing:
            return

        episode.add_mpc_diagnostics(msg.solve_ms, msg.budget_exceeded)

        row = self._row_prefix()
        row.update({
            'solve_ms': msg.solve_ms, 'budget_ms': msg.budget_ms,
            'budget_exceeded': msg.budget_exceeded,
            'arrival_fk_err_m': msg.arrival_fk_err_m,
            'arrival_fk_rot_err_deg': msg.arrival_fk_rot_err_deg,
            'cost_tool_pose_pos': msg.cost_tool_pose_pos,
            'cost_tool_pose_orient': msg.cost_tool_pose_orient,
            'cost_cspace': msg.cost_cspace,
            'cost_cspace_vel': msg.cost_cspace_vel,
            'cost_cspace_acc': msg.cost_cspace_acc,
            'cost_cspace_jerk': msg.cost_cspace_jerk,
            'con_self_collision': msg.con_self_collision,
            'con_scene_collision': msg.con_scene_collision,
            'con_cspace_bound': msg.con_cspace_bound,
        })
        episode.mpc_writer.writerow(row)
        episode.mpc_file.flush()

        self._request_fk_batch(msg)

    def _request_fk_batch(self, mpc_msg):
        if self._goal_pose is None:
            return
        if not self.fk_batch_client.service_is_ready():
            self.get_logger().warn(
                f'{self.fk_batch_service} not available, skipping IK/FK-batch check',
                throttle_duration_sec=5.0)
            return

        solutions = self._ikdh.inverse_kinematics(
            self._goal_pose.position, self._goal_pose.orientation)
        solutions = solutions[:self.fk_batch_size]

        request = FkBatch.Request()
        for i in range(self.fk_batch_size):
            js = JointState()
            js.name = list(self.arm_joint_names)
            if i < len(solutions):
                js.position = [float(v) for v in np.radians(solutions[i])]
            else:
                js.position = [0.0] * len(self.arm_joint_names)
            request.joint_states.append(js)

        episode = self._episode
        stamp = mpc_msg.header.stamp
        mpc_stamp = stamp.sec + stamp.nanosec * 1e-9
        # Snapshot the episode columns now: the response is asynchronous and the
        # phase may well have moved on by the time it comes back.
        prefix = self._row_prefix()

        future = self.fk_batch_client.call_async(request)
        future.add_done_callback(
            lambda f: self._on_fk_batch_response(f, episode, prefix, mpc_stamp, solutions))

    def _on_fk_batch_response(self, future, episode, prefix, mpc_stamp, solutions):
        try:
            response = future.result()
        except Exception as ex:  # noqa: BLE001 - rclpy futures surface arbitrary service errors
            self.get_logger().warn(
                f'{self.fk_batch_service} call failed: {ex}', throttle_duration_sec=5.0)
            return
        if not response.success:
            self.get_logger().warn(
                f'{self.fk_batch_service} rejected: {response.error_msg.data}',
                throttle_duration_sec=5.0)
            return
        if episode.ikfk_file is None or episode.ikfk_file.closed:
            return  # the episode ended before this async response came back

        for i, q_deg in enumerate(solutions):
            pose = response.poses[i]
            row = dict(prefix)
            row.update({
                'mpc_stamp': mpc_stamp,
                'candidate_idx': i,
                'manipulability': self._ikdh.manipulability(q_deg),
                'valid': response.poses_valid[i].data,
                'pose_x': pose.position.x, 'pose_y': pose.position.y, 'pose_z': pose.position.z,
            })
            for j, jn in enumerate(self.arm_joint_names):
                row[f'q_{jn}'] = q_deg[j]
            episode.ikfk_writer.writerow(row)
        episode.ikfk_file.flush()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _plot_episode(self, episode, summary):
        """Save a PNG of Q_goal / Q_path over the episode next to the CSV outputs."""
        path = os.path.join(episode.dir, f'episode_{episode.episode_id:03d}.png')
        try:
            experiment_plots.plot_episode(
                path,
                episode_id=episode.episode_id,
                rel_times=episode.rel_times,
                q_goal=episode.q_goal_values,
                q_path_mean=episode.q_path_means,
                q_path_min=episode.q_path_min_series,
                q_path_max=episode.q_path_max_series,
                phase_events=episode.phase_events,
                summary=summary)
        except Exception as ex:  # noqa: BLE001 - a plotting failure must never cost data
            self.get_logger().warn(f'Episode plot failed: {ex}')
            return
        self.get_logger().info(f'Episode {episode.episode_id} plot saved to {path}')

    def _plot_run(self):
        """Write the run-level dashboard and base-trajectory figures."""
        for name, fn in (('base_trajectory.png', experiment_plots.plot_base_trajectory),
                         ('run_summary.png', experiment_plots.plot_run_summary)):
            try:
                out = fn(self.output_dir)
            except Exception as ex:  # noqa: BLE001 - never let a figure cost the run
                self.get_logger().warn(f'{name} failed: {ex}')
                continue
            if out:
                self.get_logger().info(f'Run figure saved to {out}')

    def _get_goal_evaluation(self, goal):
        """Reuse the cached WorkspaceEvaluation while the goal hasn't moved.

        WorkspaceEvaluation caches its sphere mask internally keyed on the voxel
        grid parameters, but rebuilding the object also invalidates that cache, so
        we additionally avoid recreating it every message when the goal is static.
        """
        if self._goal_eval is None or self._goal_eval_center != goal:
            self._goal_eval = WorkspaceEvaluation(
                centers_xyz=goal, radius=self.workspace_radius, device=self.device)
            self._goal_eval_center = goal
        return self._goal_eval

    def _extract_path_centers(self, msg: Path):
        """Subsample path poses (always keeping first/last) and transform to the RM frame."""
        n = len(msg.poses)
        if self.path_max_samples > 0 and n > self.path_max_samples:
            idx = np.linspace(0, n - 1, self.path_max_samples).round().astype(int)
            idx = sorted({int(i) for i in idx})
        else:
            idx = range(n)

        path_frame = msg.header.frame_id or self.planning_frame
        need_transform = path_frame != self._vg_frame

        tf = None
        if need_transform:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self._vg_frame, path_frame, rclpy.time.Time())
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException) as ex:
                self.get_logger().warn(
                    f'TF {path_frame} -> {self._vg_frame} unavailable: {ex}',
                    throttle_duration_sec=5.0)
                return []

        centers = []
        for i in idx:
            pose_stamped = msg.poses[i]
            if need_transform:
                ps = PoseStamped()
                ps.header.frame_id = path_frame
                ps.pose = pose_stamped.pose
                transformed = do_transform_pose_stamped(ps, tf)
                p = transformed.pose.position
            else:
                p = pose_stamped.pose.position
            centers.append((p.x, p.y, p.z))
        return centers

    def _publish_and_log_sample(
            self, msg, centers, q_goal, q_path_mean, q_path_min, q_path_max,
            rm_age_ms, compute_ms):
        m_goal = Float32()
        m_goal.data = q_goal
        self.q_goal_pub.publish(m_goal)

        m_path = Float32()
        m_path.data = q_path_mean
        self.q_path_mean_pub.publish(m_path)

        gx, gy, gz = self._goal
        # mpc_goal is only published during the approach legs (act_move_pregrasp /
        # act_move_preplace); everywhere else this Q_goal is scored against a stale
        # target. The age says by how much, so those rows can be filtered out.
        goal_age_s = (time.time() - self._goal_stamp) if self._goal_stamp else float('nan')

        row = self._row_prefix()
        row.update({
            'seq': self._seq,
            'goal_x': gx, 'goal_y': gy, 'goal_z': gz,
            'goal_age_s': goal_age_s,
            'path_frame': msg.header.frame_id,
            'n_poses': len(msg.poses),
            'n_eval': len(centers),
            'q_goal': q_goal,
            'q_path_mean': q_path_mean,
            'q_path_min': q_path_min,
            'q_path_max': q_path_max,
            'rm_age_ms': rm_age_ms,
            'compute_ms': compute_ms,
        })
        self._episode.samples_writer.writerow(row)
        self._episode.samples_file.flush()

        if self.log_each_sample:
            self.get_logger().info(
                f"[ep {row['episode_id']} {row['phase']}] Q_goal={q_goal:.4f} "
                f'Q_path_mean={q_path_mean:.4f} [{q_path_min:.4f}, {q_path_max:.4f}] '
                f'n={len(centers)}/{len(msg.poses)} rm_age={rm_age_ms:.0f}ms '
                f'compute={compute_ms:.1f}ms'
            )

    def destroy_node(self):
        self._cancel_outcome_timer()
        if self._episode is not None:
            self._flush_episode()
        for attr in ('_index_file', '_phase_summary_file'):
            handle = getattr(self, attr, None)
            if handle is not None and not handle.closed:
                handle.close()
        if getattr(self, 'generate_run_plots', False):
            self._plot_run()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceReachabilityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        # rclpy's own SIGINT handler has usually shut the context down already;
        # calling shutdown() again raises and would mask the exit code.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
