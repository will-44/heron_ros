# The Heron Package is self-contained: the robot runs it without Heron

Status: accepted (supersedes 0001 and 0002; amends 0003)

ADR 0001 removed capacitynet's copy of the inference code by importing Heron. It worked,
but it put the training repository, pytorch-3dunet and their versions on the robot, as
a dependency nothing declared and an update could break.

The robot now reads a Heron Package of format 1 (written by `heron export`, Heron
9e28290) and needs only TensorRT and torch:

- **The network travels as ONNX**, layers and activations included: nothing on the
  robot reads a model config, so a checkpoint cannot be loaded into the wrong
  architecture (capacitynet loaded an ELU network as ReLU, without error).
- **Everything else is a value in manifest.yaml**: input shape and axes, the value of a
  free and of an occupied voxel, the grid, the output range. Heron finds the encoding by
  running the Experiment's own val transform on probe Scenes, and refuses to export a
  transform that one value per kind of voxel cannot stand for.
- **reference.h5 checks what could still drift.** Five occupancy Scenes and the maps
  Heron's PyTorch gives them. Every node, at startup, encodes them with the manifest's
  values, runs them through its engine, and refuses to start if the maps differ beyond
  the precision's tolerance (core/heron_package.py, TOLERANCE).
- **Engines are built on the robot from the ONNX**, by `build_engine`, which also checks
  and times them. int8 calibrates on calibration.h5, whose volumes Heron stores already
  encoded.

## Consequences

- One directory to copy, one command to run on a new machine; no Heron, no
  pytorch-3dunet, no `pip install` in the container.
- No PyTorch on the robot, so no PyTorch fallback: a missing engine or one that fails
  its reference check stops the node with what to do (0003 without its fallback).
- The Scenes never leave the GPU: the encoding is a `torch.where`, not pytorch-3dunet's
  NumPy transform (0002's cost is gone).
- The engine build exists twice, in Heron (benchmark) and here. They agree on the file
  name, so either can build an engine the other loads; the reference check covers both.
- Packages exported before format 1 are refused, with the command to export them again.

## Measured (2026-11-20, Orin, TensorRT 10.16, full_setup_20260805_1144)

`build_engine`, worst of the 5 reference Scenes against Heron's PyTorch:

| Engine | mean \|diff\| | max \|diff\| | per Scene | build |
|---|---|---|---|---|
| fp32 | 2.1e-7 | 4.2e-6 | 147 ms | 143 s |
| fp16 | 3.9e-4 | 1.2e-2 | 40.2 ms | 213 s |
| int8 | 1.1e-2 | 0.39 | 35.4 ms | 387 s |

A wrong encoding gives a mean of 0.21 (0/1 instead of -1/+1) or 0.27 (inverted), far
beyond every tolerance. Occupied at 0.9 instead of 1 gives 3.3e-4: only the fp32
tolerance catches an error that small. int8 is 12 % faster than fp16 for 28 times its
error.

Bag replay against capacitynet, same inputs bit for bit (results/parity/
20261120_124500_self_contained_package/):

- capacitynet PyTorch vs this fp32 engine, 24 cycles: Q within 6.6e-7, grad Q within 3.7e-7.
- capacitynet PyTorch vs this fp16 engine, 25 cycles: Q within 1.8e-3, grad Q direction within 1.0°.
- Cycle (inference and scoring), fp16: 428 ms against 441 ms for capacitynet's engine in the
  same process, and 518 ms for the heron.inference path of 0001. The real
  `workspace_probe` node: 23 cycles, median 447 ms.
