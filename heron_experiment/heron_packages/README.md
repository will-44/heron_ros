# heron_packages

Heron Packages the nodes can run, one directory per Experiment, named after it:

```
heron_packages/<experiment>/        made by `heron export` (Heron repository)
    manifest.yaml  config.yaml  checkpoint.pytorch  model.onnx  calibration.h5
    engines/                        TensorRT engines, built on this machine
```

1. Where the Experiment was trained: `heron export <experiment>` (add
   `--voxel-size 0.02 --origin X Y Z` if its volumes do not carry their grid).
2. Copy the package directory here, without `engines/`.
3. In the robot's container, build and time its engines for this GPU:
   `python -m heron.inference.benchmark --package heron_packages/<experiment> --precision fp16 int8 --results-dir <dir>`
4. Launch with `heron_package:=<experiment> precision:=fp16`.

A node refuses a package whose manifest has no `grid`, and a voxel grid that
does not match it. Only this README is in git.
