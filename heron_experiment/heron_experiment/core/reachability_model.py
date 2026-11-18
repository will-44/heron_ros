#!/usr/bin/env python3
"""A Heron Package as the robot uses it: curobo voxel grid in, Reachability maps out.

    model = ReachabilityModel(package, precision='int8', fallback=True, log=print)
    model.check_grid(msg)                 # None, or why this voxel grid is not the network's input
    scene = model.scene(msg)              # (sx, sy, sz) bool occupancy on the device
    maps = model.predict(scenes)          # (N, sx, sy, sz) bool -> (N, sx, sy, sz) float32, same device

Everything about the network (its input encoding, PyTorch or TensorRT, which engine)
lives in heron.inference: callers here only ever handle occupancy, True where a voxel
holds an obstacle. What this module adds is the curobo side: the linear index order
of SparseVoxelGrid, and the check that a voxel grid has the size, voxel size and
origin the network was trained on (the package manifest's grid).

ROS-free: a message is anything with occupied_indices, size_x/y/z, resolution and
origin.x/y/z, like curobo_msgs/SparseVoxelGrid.
"""

from pathlib import Path

import numpy as np
import torch

# Voxel size and origin are float32 in the message and in the training files.
_GRID_TOLERANCE_M = 1e-4


def package_path(name_or_path):
    """A Heron Package directory: a path, or a name under heron_packages/ (see heron_experiment.paths)."""
    from heron_experiment.paths import packages_dir

    candidate = Path(str(name_or_path)).expanduser()
    if candidate.is_absolute() or candidate.is_dir():
        return candidate
    return packages_dir() / str(name_or_path)


class ReachabilityModel:
    """The network of one Heron Package, at one precision, on this machine's GPU (or CPU)."""

    def __init__(self, package, precision='pytorch', fallback=False, log=print):
        """Load the package's network. Never builds a TensorRT engine (see heron.inference.load).

        Args:
            package: Heron Package directory (made by `heron export`), or its name (see package_path)
            precision: 'pytorch', 'fp32', 'fp16' or 'int8'
            fallback: run PyTorch, and say so through `log`, if the engine is missing
            log: callable taking one message string
        """
        from heron.inference import load

        self.package = package_path(package)
        self.predictor = load(self.package, precision, fallback=fallback, log=log)
        self.device = self.predictor.device
        manifest = self.predictor.manifest
        self.input_shape = tuple(manifest['input_shape'])
        self.grid = manifest.get('grid')
        if self.grid is None:
            raise ValueError(
                f'{self.package}: the manifest has no grid, so voxel grids cannot be checked '
                'against what the network was trained on. Export it again with '
                '`heron export <experiment> --force --voxel-size M --origin X Y Z`.')

    @property
    def backend(self):
        """What runs the network: 'pytorch', 'tensorrt_int8', ..."""
        return self.predictor.backend

    @property
    def experiment(self):
        return self.predictor.manifest['experiment']

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
        """The voxel grid as a dense occupancy tensor (sx, sy, sz), bool, on the device."""
        sx, sy, sz = int(msg.size_x), int(msg.size_y), int(msg.size_z)
        occupied = torch.as_tensor(np.asarray(msg.occupied_indices, dtype=np.int64), device=self.device)
        flat = torch.zeros(sx * sy * sz, dtype=torch.bool, device=self.device)
        if occupied.numel() > 0:
            flat[occupied] = True
        # curobo's linear index is x-major: index = (x * sy + y) * sz + z.
        return flat.reshape(sx, sy, sz)

    def predict(self, scenes):
        """Reachability maps of a batch of Scenes, one network call each.

        Args:
            scenes: (N, sx, sy, sz) or (sx, sy, sz) occupancy, bool or {0, 1}

        Returns:
            torch.Tensor float32 of the same shape, on the device, values in [0, 1]
        """
        single = scenes.dim() == 3
        batch = scenes.unsqueeze(0) if single else scenes
        # Batching gains nothing on the Orin (39.8 ms per map alone, 41.1 ms in a batch
        # of 9, see docs/OPTIMIZATIONS.md), so the engines take one Scene at a time.
        maps = [torch.from_numpy(self.predictor.predict(s.to(torch.float32).cpu().numpy()))
                for s in batch]
        out = torch.stack(maps).to(self.device)
        return out[0] if single else out
