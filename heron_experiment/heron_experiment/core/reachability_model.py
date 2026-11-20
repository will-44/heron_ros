#!/usr/bin/env python3
"""A Heron Package as the robot uses it: curobo voxel grid in, Reachability maps out.

    model = ReachabilityModel(package, precision='fp16', log=print)
    model.check_grid(msg)                 # None, or why this voxel grid is not the network's input
    scene = model.scene(msg)              # (sx, sy, sz) bool occupancy on the GPU
    maps = model.predict(scenes)          # (N, sx, sy, sz) bool -> (N, sx, sy, sz) float32, on the GPU

The network runs as the package's TensorRT engine, built beforehand on this machine
(build_engine), fed through the encoding the package's manifest spells out: no heron,
no pytorch-3dunet, and the Scenes never leave the GPU. Before serving, the model runs
the package's reference Scenes and refuses an engine whose maps differ from heron's
(see core/heron_package.py and docs/adr/0004).

ROS-free: a message is anything with occupied_indices, size_x/y/z, resolution and
origin.x/y/z, like curobo_msgs/SparseVoxelGrid.
"""

import numpy as np
import torch

from heron_experiment.core.heron_package import HeronPackage, PackageError, package_path  # noqa: F401

# Voxel size and origin are float32 in the message and in the training files.
_GRID_TOLERANCE_M = 1e-4


class ReachabilityModel:
    """The network of one Heron Package, as one of its TensorRT engines on this machine's GPU."""

    def __init__(self, package, precision='fp16', log=print):
        """Load the package's engine for `precision` and check it on the reference Scenes.

        Args:
            package: Heron Package directory (made by `heron export`), or its name (see package_path)
            precision: 'fp32', 'fp16' or 'int8'; its engine must have been built (build_engine)
            log: callable taking one message string

        Raises:
            PackageError: not a format-1 package, no engine for this machine, or an engine that
                fails the reference check. The message says what to do.
        """
        if not torch.cuda.is_available():
            raise PackageError('no GPU visible: the Heron Package runs as a TensorRT engine')
        self.package = HeronPackage(package)
        self.precision = precision
        self.device = torch.device('cuda')
        self.engine = self.package.engine(precision)
        self.verification = self.package.verify(self.engine, precision)
        check = self.verification
        summary = (f'{self.package.experiment} {precision}: reference Scenes mean |diff| '
                   f'{check["mean_abs_diff"]:.1e}, max {check["max_abs_diff"]:.1e} over {check["scenes"]} Scenes')
        if not check['ok']:
            raise PackageError(f'{summary}, beyond the tolerance {check["tolerance"]}: this engine does not '
                               f'compute the network heron trained. Rebuild it (build_engine --rebuild) and '
                               f'check the package was copied whole.')
        log(summary)
        self.input_shape = self.package.shape
        self.grid = self.package.grid

    @property
    def backend(self):
        """What runs the network: 'tensorrt_fp16', ..."""
        return f'tensorrt_{self.precision}'

    @property
    def experiment(self):
        return self.package.experiment

    def check_grid(self, msg):
        """None if `msg` is laid out like the network's input, else a sentence saying how it differs."""
        size = (int(msg.size_x), int(msg.size_y), int(msg.size_z))
        if size != self.input_shape:
            return f'voxel grid is {size}, the network takes {self.input_shape}'
        if abs(float(msg.resolution) - self.grid['voxel_size']) > _GRID_TOLERANCE_M:
            return (f'voxel size is {float(msg.resolution):.4f} m, '
                    f'the network was trained at {self.grid["voxel_size"]} m')
        origin = (float(msg.origin.x), float(msg.origin.y), float(msg.origin.z))
        if any(abs(a - b) > _GRID_TOLERANCE_M for a, b in zip(origin, self.grid['origin'])):
            return (f'voxel grid origin is {[round(v, 4) for v in origin]} m, '
                    f'the network was trained with {self.grid["origin"]} m')
        return None

    def scene(self, msg):
        """The voxel grid as a dense occupancy tensor (sx, sy, sz), bool, on the GPU."""
        sx, sy, sz = int(msg.size_x), int(msg.size_y), int(msg.size_z)
        occupied = torch.as_tensor(np.asarray(msg.occupied_indices, dtype=np.int64), device=self.device)
        flat = torch.zeros(sx * sy * sz, dtype=torch.bool, device=self.device)
        if occupied.numel() > 0:
            flat[occupied] = True
        # curobo's linear index is x-major: index = (x * sy + y) * sz + z, the package's x, y, z axes.
        return flat.reshape(sx, sy, sz)

    def predict(self, scenes):
        """Reachability maps of a batch of Scenes, one engine call each, all on the GPU.

        Args:
            scenes: (N, sx, sy, sz) or (sx, sy, sz) occupancy, bool or {0, 1}, on the GPU

        Returns:
            torch.Tensor float32 of the same shape, on the GPU, values in [0, 1]
        """
        single = scenes.dim() == 3
        batch = scenes.unsqueeze(0) if single else scenes
        # Batching gains nothing on the Orin (39.8 ms per map alone, 41.1 ms in a batch
        # of 9, see docs/OPTIMIZATIONS.md), so the engines take one Scene at a time.
        inputs = self.package.encode(batch.to(self.device))
        out = torch.stack([self.engine(x[None, None]).reshape(x.shape) for x in inputs])
        return out[0] if single else out
