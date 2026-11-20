# heron_packages

Heron Packages the nodes can run, one directory per Experiment, named after it:

```
heron_packages/<experiment>/        made by `heron export` (Heron repository), package format 1
    manifest.yaml       the input shape and axes, the encoding, the grid, the output: as values
    model.onnx          the network, layers and activations included
    reference.h5        occupancy Scenes and the maps Heron's PyTorch gives them
    calibration.h5      volumes already encoded, for int8
    config.yaml  checkpoint.pytorch     for Heron only; the robot never reads them
    engines/            TensorRT engines, built on this machine
```

The robot needs neither Heron nor pytorch-3dunet: TensorRT and torch from JetPack are
enough (docs/adr/0004).

1. Where the Experiment was trained, with a Heron that writes format 1:
   `heron export <experiment>` (add `--voxel-size 0.02 --origin X Y Z` if its volumes
   do not carry their grid).
2. Copy the whole package directory here, without `engines/`.
3. On the robot, build its engines for this GPU, check them against the reference
   Scenes and time them (minutes per precision):
   `ros2 run heron_experiment build_engine --package <experiment> --precision fp16 int8`
4. Launch with `heron_package:=<experiment> precision:=fp16`.

A node refuses to start on a package of another format, without the engine for its
precision, or with an engine whose maps of the reference Scenes differ from Heron's.
It refuses a voxel grid that does not match the manifest's grid. Only this README is in git.
