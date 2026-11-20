#!/usr/bin/env python3
"""A Heron Package, read and run on this machine with TensorRT alone: no heron, no pytorch-3dunet.

    package = HeronPackage('full_setup_20260805_1144')    # a name under heron_packages/, or a path
    package.build_engine('fp16')                          # once per machine: minutes on the Orin
    engine = package.engine('fp16')                       # (1, 1, D, H, W) float32 CUDA tensor -> map
    report = package.verify(engine)                       # reference.h5 through our encoding and engine

Everything the robot must agree on with the network's training is in manifest.yaml, as
values (format 1, written by `heron export`): the input shape and axes, the value of a free
and of an occupied voxel, the grid, the output range, and the files. The network itself
travels as ONNX, layers and activations included, so nothing here reads a model config.

What could still drift from heron (the encoding as applied here, the engine build) is
checked by verify(): the package's reference Scenes, encoded here and run through the
engine, must give the maps heron's PyTorch gave them. See docs/adr/0004.
"""

import re
import time
from pathlib import Path

import numpy as np
import yaml

SUPPORTED_FORMATS = (1,)
PRECISIONS = ('fp32', 'fp16', 'int8')
MANIFEST = 'manifest.yaml'
ENGINES = 'engines'

# Worst reference Scene, |engine - heron PyTorch| over the voxels (mean, max). Measured on
# full_setup_20260805_1144, Orin, TensorRT 10.16 (2026-11-20): fp32 2.1e-7 / 4.2e-6,
# fp16 3.9e-4 / 1.2e-2, int8 1.1e-2 / 0.39. A wrong encoding gives a mean of 0.21 (0/1 instead
# of -1/+1) or 0.27 (inverted); occupied at 0.9 instead of 1 gives 3.3e-4 / 6.1e-2, which only
# the fp32 tolerance catches. Set with margin for another build's tactics, far below a wrong input.
TOLERANCE = {
    'fp32': {'mean_abs_diff': 1e-5, 'max_abs_diff': 1e-4},
    'fp16': {'mean_abs_diff': 2e-3, 'max_abs_diff': 0.1},
    'int8': {'mean_abs_diff': 3e-2, 'max_abs_diff': 0.6},
}


class PackageError(RuntimeError):
    """A Heron Package this node cannot run, and what to do about it."""


def package_path(name_or_path):
    """A Heron Package directory: a path, or a name under heron_packages/ (see heron_experiment.paths)."""
    from heron_experiment.paths import packages_dir

    candidate = Path(str(name_or_path)).expanduser()
    if candidate.is_absolute() or candidate.is_dir():
        return candidate
    return packages_dir() / str(name_or_path)


def slug(name: str) -> str:
    """The GPU name as heron writes it into engine file names ('Orin' -> 'orin')."""
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def _get(manifest, dotted, path):
    value = manifest
    for key in dotted.split('.'):
        if not isinstance(value, dict) or value.get(key) is None:
            raise PackageError(f'{path}: manifest.yaml has no {dotted}. Export it again with a heron that '
                               f'writes format 1: heron export <experiment> --force')
        value = value[key]
    return value


class HeronPackage:
    """One package directory: its manifest, its engines on this machine, its reference check."""

    def __init__(self, name_or_path):
        self.path = package_path(name_or_path)
        manifest = self.path / MANIFEST
        if not manifest.is_file():
            raise PackageError(f'{self.path}: not a Heron Package (no {MANIFEST}). Copy the directory '
                               f'`heron export` made into heron_packages/')
        self.manifest = yaml.safe_load(manifest.read_text())
        fmt = self.manifest.get('format')
        if fmt is None:
            raise PackageError(f'{self.path}: exported before package format 1 (no input encoding, no '
                               f'reference Scenes). Export it again: heron export <experiment> --force')
        if fmt not in SUPPORTED_FORMATS:
            raise PackageError(f'{self.path}: package format {fmt}, this heron_experiment reads '
                               f'{list(SUPPORTED_FORMATS)}: update heron_ros')

        m, p = self.manifest, self.path
        self.experiment = _get(m, 'experiment', p)
        self.shape = tuple(int(s) for s in _get(m, 'input.shape', p))
        if list(_get(m, 'input.axes', p)) != ['x', 'y', 'z']:
            raise PackageError(f'{p}: input axes {m["input"]["axes"]}, this node builds Scenes as x, y, z')
        encoding = _get(m, 'input.encoding', p)
        self.free = float(_get(encoding, 'free', p))
        self.occupied = float(_get(encoding, 'occupied', p))
        self.grid = {'voxel_size': float(_get(m, 'grid.voxel_size', p)),
                     'origin': [float(v) for v in _get(m, 'grid.origin', p)]}
        self.onnx = p / _get(m, 'onnx.file', p)
        self.reference = p / _get(m, 'reference.file', p)
        self.calibration = p / _get(m, 'calibration.file', p)
        self.calibration_dataset = _get(m, 'calibration.dataset', p)
        for f in (self.onnx, self.reference, self.calibration):
            if not f.is_file():
                raise PackageError(f'{p}: {f.name} is missing: copy the whole package directory')

    # ── Input ────────────────────────────────────────────────────────────────

    def encode(self, scenes):
        """Occupancy (bool or {0, 1}, any leading dims) -> the network's input values, float32, same device."""
        import torch

        occupied = scenes.bool() if scenes.dtype != torch.bool else scenes
        return torch.where(occupied, torch.tensor(self.occupied, device=scenes.device),
                           torch.tensor(self.free, device=scenes.device)).to(torch.float32)

    # ── Engines ──────────────────────────────────────────────────────────────

    def engine_path(self, precision, gpu=None, trt_version=None):
        """engines/<gpu>_trt<version>_<precision>.engine, the name heron benchmark also uses."""
        if gpu is None or trt_version is None:
            import tensorrt
            import torch
            gpu = gpu or torch.cuda.get_device_name(0)
            trt_version = trt_version or tensorrt.__version__
        return self.path / ENGINES / f'{slug(gpu)}_trt{trt_version}_{precision}.engine'

    def engine(self, precision):
        """The engine built for this GPU and TensorRT version. Never builds one (see build_engine)."""
        _check_precision(precision)
        path = self.engine_path(precision)
        if not path.is_file():
            raise PackageError(
                f'{self.path.name}: no {precision} engine for this GPU and TensorRT ({path.name}). '
                f'Build it once on this machine: ros2 run heron_experiment build_engine '
                f'--package {self.path.name} --precision {precision}')
        return TrtEngine(path, self.shape)

    def build_engine(self, precision, rebuild=False, log=print):
        """Build this machine's engine from the package's ONNX unless cached. Returns (path, seconds or None)."""
        import tensorrt as trt

        _check_precision(precision)
        major = int(trt.__version__.split('.')[0])
        if major < 10:
            raise PackageError(f'TensorRT {trt.__version__}: needs TensorRT 10')
        if precision == 'int8' and major >= 11:
            raise PackageError(f'TensorRT {trt.__version__} has no implicit int8 calibration: use fp16')
        path = self.engine_path(precision)
        if path.is_file() and not rebuild:
            return path, None
        path.parent.mkdir(exist_ok=True)

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(self.onnx)):
            errors = '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise PackageError(f'{self.onnx}: TensorRT {trt.__version__} cannot parse it: {errors}')
        config = builder.create_builder_config()
        config.clear_flag(trt.BuilderFlag.TF32)             # fp32 means fp32, as heron's reference
        if precision in ('fp16', 'int8'):
            config.set_flag(trt.BuilderFlag.FP16)           # int8: fp16 for what int8 cannot run
        if precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            cache = self.path / ENGINES / f'calibration_trt{trt.__version__}.cache'
            if rebuild and cache.is_file():
                cache.unlink()
            config.int8_calibrator = _calibrator(self.calibration, self.calibration_dataset, cache)
        log(f'{precision}: building the TensorRT engine from {self.onnx.name} (minutes on a Jetson)')
        start = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise PackageError(f'TensorRT could not build the {precision} engine (see its messages above)')
        path.write_bytes(bytes(serialized))
        return path, time.perf_counter() - start

    # ── Check ────────────────────────────────────────────────────────────────

    def verify(self, engine, precision=None):
        """Run the reference Scenes through encode() and `engine`; compare with heron's PyTorch maps.

        Returns {'scenes', 'mean_abs_diff', 'max_abs_diff', 'tolerance', 'ok'}, the diffs being the
        worst Scene's. ok is None when no tolerance is set for the precision.
        """
        import h5py
        import torch

        means, maxes = [], []
        with h5py.File(self.reference, 'r') as f:
            for i in range(len(f['scene'])):
                scene = torch.from_numpy(f['scene'][i]).cuda()
                expected = torch.from_numpy(f['reachability'][i]).cuda()
                diff = (engine(self.encode(scene)[None, None]).reshape(self.shape) - expected).abs()
                means.append(float(diff.mean()))
                maxes.append(float(diff.max()))
        tolerance = TOLERANCE.get(precision)
        report = {'scenes': len(means), 'mean_abs_diff': max(means), 'max_abs_diff': max(maxes),
                  'tolerance': tolerance}
        report['ok'] = None if tolerance is None else (
            report['mean_abs_diff'] <= tolerance['mean_abs_diff']
            and report['max_abs_diff'] <= tolerance['max_abs_diff'])
        return report


class TrtEngine:
    """An engine called on a (1, 1, D, H, W) float32 CUDA tensor; returns a new (1, 1, D, H, W) tensor."""

    def __init__(self, path, shape):
        import tensorrt as trt
        import torch

        self.torch = torch
        self.path = Path(path)
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise PackageError(f'{self.path}: TensorRT cannot load it (another TensorRT or GPU?): rebuild it '
                               f'with build_engine --rebuild')
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        output = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        expected = (1, 1, *shape)
        if tuple(self.engine.get_tensor_shape(self.input)) != expected:
            raise PackageError(f'{self.path}: engine input {tuple(self.engine.get_tensor_shape(self.input))}, '
                               f'the manifest says {expected}: rebuild it with build_engine --rebuild')
        self.out = torch.empty(tuple(self.engine.get_tensor_shape(output)), dtype=torch.float32, device='cuda')
        self.context.set_tensor_address(output, self.out.data_ptr())
        # Its own stream: on the default one, TensorRT adds synchronizations of its own.
        self.stream = torch.cuda.Stream()

    def __call__(self, x):
        x = x.contiguous()
        current = self.torch.cuda.current_stream()
        self.stream.wait_stream(current)                    # x is ready
        x.record_stream(self.stream)
        self.context.set_tensor_address(self.input, x.data_ptr())
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('TensorRT inference failed')
        current.wait_stream(self.stream)                    # out is ready for the next reader
        return self.out.clone()                             # out is overwritten by the next call


def _check_precision(precision):
    if precision not in PRECISIONS:
        hint = (' PyTorch runs in Heron, not on the robot; fp32 is the full-precision engine.'
                if precision == 'pytorch' else '')
        raise PackageError(f'precision must be one of {", ".join(PRECISIONS)}, got {precision!r}.{hint}')


def _calibrator(path, dataset, cache):
    """Entropy calibrator over the package's calibration volumes, already encoded by heron."""
    import h5py
    import tensorrt as trt
    import torch

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.file = h5py.File(path, 'r')
            self.volumes = self.file[dataset]
            self.i = 0
            self.batch = None           # kept alive until TensorRT asks for the next one

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.i >= len(self.volumes):
                return None
            self.batch = torch.from_numpy(np.ascontiguousarray(self.volumes[self.i]))[None, None].cuda()
            self.i += 1
            return [self.batch.data_ptr()]

        def read_calibration_cache(self):
            return cache.read_bytes() if cache.is_file() else None

        def write_calibration_cache(self, data):
            cache.write_bytes(bytes(data))

    return Calibrator()
