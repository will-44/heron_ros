"""The Heron Package as the robot reads it: manifest checks, encoding, engines, reference check.

The tests that run an engine need a GPU, TensorRT and the package of the parity test
built on this machine (heron_packages/full_setup_20260805_1144, `build_engine`); they
skip otherwise.
"""

import os
import shutil
from pathlib import Path

import pytest
import yaml

from heron_experiment.core import heron_package
from heron_experiment.core.heron_package import HeronPackage, PackageError

REAL = Path(__file__).resolve().parents[1] / 'heron_packages' / 'full_setup_20260805_1144'

MANIFEST = {
    'format': 1,
    'experiment': 'tiny',
    'input': {'shape': [4, 4, 4], 'axes': ['x', 'y', 'z'], 'dtype': 'float32', 'batch': 1, 'channels': 1,
              'encoding': {'free': -1.0, 'occupied': 1.0}},
    'grid': {'voxel_size': 0.02, 'origin': [-0.04, -0.04, -0.04]},
    'output': {'shape': [4, 4, 4], 'dtype': 'float32', 'activation': 'sigmoid', 'range': [0.0, 1.0]},
    'onnx': {'file': 'model.onnx'},
    'calibration': {'file': 'calibration.h5', 'dataset': 'input'},
    'reference': {'file': 'reference.h5'},
}


def make_package(tmp_path, **changes):
    manifest = yaml.safe_load(yaml.safe_dump(MANIFEST))
    for dotted, value in changes.items():
        *parents, key = dotted.split('__')
        target = manifest
        for p in parents:
            target = target[p]
        if value is None:
            target.pop(key, None)
        else:
            target[key] = value
    for name in ('model.onnx', 'calibration.h5', 'reference.h5'):
        (tmp_path / name).write_bytes(b'')
    (tmp_path / 'manifest.yaml').write_text(yaml.safe_dump(manifest))
    return tmp_path


def test_the_manifest_gives_everything_the_robot_needs(tmp_path):
    package = HeronPackage(make_package(tmp_path))
    assert package.shape == (4, 4, 4) and (package.free, package.occupied) == (-1.0, 1.0)
    assert package.grid == {'voxel_size': 0.02, 'origin': [-0.04, -0.04, -0.04]}
    assert package.experiment == 'tiny' and package.calibration_dataset == 'input'


def test_a_package_exported_before_format_1_asks_to_export_it_again(tmp_path):
    with pytest.raises(PackageError, match='before package format 1.*heron export <experiment> --force'):
        HeronPackage(make_package(tmp_path, format=None))


def test_a_newer_format_asks_to_update_heron_ros(tmp_path):
    with pytest.raises(PackageError, match='package format 2.*update heron_ros'):
        HeronPackage(make_package(tmp_path, format=2))


def test_a_missing_value_is_named(tmp_path):
    with pytest.raises(PackageError, match='no input.encoding'):
        HeronPackage(make_package(tmp_path, input__encoding=None))
    with pytest.raises(PackageError, match='no grid.origin'):
        HeronPackage(make_package(tmp_path, grid__origin=None))


def test_scenes_must_be_laid_out_x_y_z(tmp_path):
    with pytest.raises(PackageError, match='input axes'):
        HeronPackage(make_package(tmp_path, input__axes=['z', 'y', 'x']))


def test_a_missing_file_asks_for_the_whole_package(tmp_path):
    path = make_package(tmp_path)
    (path / 'reference.h5').unlink()
    with pytest.raises(PackageError, match='reference.h5 is missing'):
        HeronPackage(path)


def test_occupancy_is_encoded_with_the_manifest_values(tmp_path):
    torch = pytest.importorskip('torch')
    package = HeronPackage(make_package(tmp_path, input__encoding={'free': 0.0, 'occupied': 0.5}))
    scene = torch.tensor([[True, False], [False, True]])
    out = package.encode(scene)
    assert out.dtype == torch.float32 and out.tolist() == [[0.5, 0.0], [0.0, 0.5]]
    assert package.encode(scene.to(torch.uint8)).tolist() == out.tolist()


def test_engines_are_named_as_heron_benchmark_names_them(tmp_path):
    package = HeronPackage(make_package(tmp_path))
    assert package.engine_path('fp16', gpu='Orin', trt_version='10.16.1.11') == \
        tmp_path / 'engines' / 'orin_trt10.16.1.11_fp16.engine'


def test_pytorch_is_not_a_precision_on_the_robot(tmp_path):
    with pytest.raises(PackageError, match='PyTorch runs in Heron, not on the robot'):
        HeronPackage(make_package(tmp_path)).engine('pytorch')


# --- with a GPU and the real package ----------------------------------------------------------

def real_package_with(precision):
    torch = pytest.importorskip('torch')
    pytest.importorskip('tensorrt')
    if not torch.cuda.is_available():
        pytest.skip('needs a GPU')
    if not (REAL / 'manifest.yaml').is_file():
        pytest.skip(f'needs {REAL}')
    package = HeronPackage(REAL)
    if not package.engine_path(precision).is_file():
        pytest.skip(f'needs its {precision} engine: build_engine --package {REAL.name} --precision {precision}')
    return package


@pytest.mark.parametrize('precision', heron_package.PRECISIONS)
def test_each_engine_gives_the_reference_scenes_the_maps_heron_gave_them(precision):
    package = real_package_with(precision)
    report = package.verify(package.engine(precision), precision)
    assert report['scenes'] == 5 and report['ok'], report


def test_a_wrong_encoding_fails_the_reference_check(tmp_path):
    """The check is what catches an encoding that drifted from training: 0/1 instead of -1/+1."""
    package = real_package_with('fp16')
    copy = tmp_path / REAL.name
    shutil.copytree(REAL, copy, ignore=shutil.ignore_patterns('engines'), copy_function=os.symlink)
    (copy / 'engines').mkdir()
    engine = package.engine_path('fp16')
    os.symlink(engine, copy / 'engines' / engine.name)
    manifest = yaml.safe_load((copy / 'manifest.yaml').read_text())
    manifest['input']['encoding'] = {'free': 0.0, 'occupied': 1.0}
    (copy / 'manifest.yaml').unlink()                   # the symlink, not the real manifest
    (copy / 'manifest.yaml').write_text(yaml.safe_dump(manifest))
    wrong = HeronPackage(copy)
    report = wrong.verify(wrong.engine('fp16'), 'fp16')
    assert not report['ok'] and report['mean_abs_diff'] > 0.1


def test_the_model_refuses_an_engine_that_fails_the_check(tmp_path, monkeypatch):
    package = real_package_with('fp16')
    from heron_experiment.core.reachability_model import ReachabilityModel
    monkeypatch.setitem(heron_package.TOLERANCE, 'fp16', {'mean_abs_diff': 0.0, 'max_abs_diff': 0.0})
    with pytest.raises(PackageError, match='does not compute the network heron trained'):
        ReachabilityModel(package.path, 'fp16', log=lambda *_: None)
