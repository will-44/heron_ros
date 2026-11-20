#!/usr/bin/env python3
"""Build, check and time a Heron Package's TensorRT engines on this machine.

    ros2 run heron_experiment build_engine --package full_setup_20260805_1144 --precision fp16 int8

For each precision: build the engine from the package's ONNX (cached in its engines/
directory; --rebuild forces it), run the package's reference Scenes through it and compare
with heron's PyTorch maps, then time it. Writes results/benchmarks/<run_id>_<experiment>/
engines.yaml. Exits with 2 if an engine fails its reference check: the nodes refuse it too.
"""

import argparse
import statistics
import sys
import time
from datetime import datetime, timezone

import yaml

from heron_experiment import paths
from heron_experiment.core.heron_package import PRECISIONS, HeronPackage, PackageError


def time_engine(package, engine, samples, warmup):
    """Median and p95 (ms) of one Scene, encode included, on a reference Scene."""
    import h5py
    import torch

    with h5py.File(package.reference, 'r') as f:
        scene = torch.from_numpy(f['scene'][0]).cuda().bool()
    times = []
    for i in range(warmup + samples):
        torch.cuda.synchronize()
        start = time.perf_counter()
        engine(package.encode(scene)[None, None])
        torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1e3)
    times.sort()
    return {'median_ms': round(statistics.median(times), 2), 'p95_ms': round(times[int(0.95 * (len(times) - 1))], 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', required=True, help='a name under heron_packages/, or a path')
    ap.add_argument('--precision', nargs='+', default=['fp16'], choices=PRECISIONS)
    ap.add_argument('--rebuild', action='store_true', help='build again even if the engine is cached')
    ap.add_argument('--samples', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=10)
    args = ap.parse_args(argv)

    import tensorrt
    import torch

    try:
        package = HeronPackage(args.package)
        gpu = torch.cuda.get_device_name(0)
        report = {'package': str(package.path), 'experiment': package.experiment,
                  'gpu': gpu, 'tensorrt': tensorrt.__version__, 'measured':
                  datetime.now(timezone.utc).isoformat(timespec='seconds'), 'engines': {}}
        failed = []
        for precision in args.precision:
            path, seconds = package.build_engine(precision, rebuild=args.rebuild)
            engine = package.engine(precision)
            check = package.verify(engine, precision)
            timing = time_engine(package, engine, args.samples, args.warmup)
            report['engines'][precision] = {
                'engine': path.name, 'build_seconds': round(seconds, 1) if seconds else 'cached',
                'reference_check': check, 'per_scene': timing}
            verdict = 'ok' if check['ok'] else 'FAILED'
            print(f'{precision}: {path.name}  reference {verdict} (worst Scene: mean |diff| '
                  f'{check["mean_abs_diff"]:.2e}, max {check["max_abs_diff"]:.2e})  '
                  f'{timing["median_ms"]:.1f} ms per Scene')
            if not check['ok']:
                failed.append(precision)
            del engine
            torch.cuda.empty_cache()
    except PackageError as exc:
        print(f'build_engine: {exc}', file=sys.stderr)
        return 2

    out = paths.results_dir() / 'benchmarks' / paths.run_id(package.experiment)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'engines.yaml').write_text(yaml.safe_dump(report, sort_keys=False))
    print(f'report: {out / "engines.yaml"}')
    if failed:
        print(f'build_engine: {", ".join(failed)} gave the reference Scenes other maps than heron did: '
              f'the nodes will refuse {"it" if len(failed) == 1 else "them"}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
