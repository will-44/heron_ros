#!/usr/bin/env python3
"""Estimate sigma, the noise on Q, by watching Q_center over repeated cycles.

Reuses the same WorkspaceRegionSource + ReachabilityPipeline as
`workspace_probe`, with no actuation (no BaseCommander). At a physically
static base pose and static scene, each cycle re-runs the full pipeline
(voxel grid -> preprocess -> predict -> Q) on the latest live voxel grid,
same as production. `score_center` is Q at the untranslated (0,0) candidate,
i.e. the base's actual current pose, so consecutive cycles are repeated
measurements of the same quantity. Their rolling std is reported every
cycle once the window fills, so sigma can be watched converge live.

This is total pipeline noise, not model-only noise: it mixes sensor/occupancy
jitter frame to frame with any model or GPU non-determinism. That is
deliberate -- it is what actually reaches the controller and the deadband in
`base_commander.py`, not an idealized lower bound.

    ros2 run heron_experiment rm_noise_probe --ros-args \
        -p static_workspace_center:="0.6,0.3,0.4" \
        -p sample_window:=30

Every region/inference parameter of `gradient_base_controller` works here
unchanged (see `workspace_probe.py`); the motion parameters simply do not
exist.
"""

import collections

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from heron_experiment.components.reachability_pipeline import ReachabilityPipeline
from heron_experiment.components.workspace_region import WorkspaceRegionSource


class RMNoiseProbe(Node):
    """Rolling mean/std of Q_center at a fixed pose -- no actuation."""

    def __init__(self):
        super().__init__('rm_noise_probe')

        self.cuda_group = MutuallyExclusiveCallbackGroup()
        self.control_group = MutuallyExclusiveCallbackGroup()

        self.region = WorkspaceRegionSource(self, callback_group=self.control_group)
        self.pipeline = ReachabilityPipeline(
            self, self.region, callback_group=self.cuda_group)
        self.pipeline.on_result = self._on_result

        self.sample_window = int(
            self.declare_parameter('sample_window', 30).value)
        self._window = collections.deque(maxlen=self.sample_window)

        log = self.get_logger()
        log.info('RM noise probe initialized (no /cmd_vel publisher)')
        log.info(
            f'  - Voxel grid topic: {self.pipeline.voxel_grid_topic} '
            f'(cycle every {self.pipeline.cycle_period}s)')
        log.info(
            f'  - Region: goal={self.region.goal_topic} path={self.region.path_topic} '
            f'(horizons={self.region.path_horizons}s), fallback=marker')
        if self.region.static_workspace_center:
            log.info(
                f'  - Static workspace center: {self.region.static_workspace_center}')
        log.info(f'  - Rolling window: {self.sample_window} samples')
        log.info(
            'Keep the base and the scene physically static; sigma converges '
            f'once {self.sample_window} cycles have run '
            f'(~{self.sample_window * self.pipeline.cycle_period:.0f}s).')

    def _on_result(self, result):
        self._window.append(result.score_center)
        n = len(self._window)
        if n < 2:
            self.get_logger().info(f'Q_c={result.score_center:.5f} (sample {n})')
            return

        arr = np.array(self._window)
        mean = arr.mean()
        std = arr.std(ddof=1)  # sample std -- n is always small here
        status = 'FULL' if n == self.sample_window else f'{n}/{self.sample_window}'
        self.get_logger().info(
            f'Q_c={result.score_center:.5f} | window[{status}] '
            f'mean={mean:.5f} sigma={std:.5f} '
            f'min={arr.min():.5f} max={arr.max():.5f}')


def main(args=None):
    rclpy.init(args=args)
    node = RMNoiseProbe()
    # Two threads for the same reason as workspace_probe: region callbacks must
    # keep running while a ~440 ms inference is in flight.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
