# A missing engine stops the node at startup; it is never built there

Status: accepted, amended by 0004 (no PyTorch fallback; engines built by build_engine)

A node asks for one precision of a Heron Package. If the engine for this GPU and
TensorRT version is not in the package's engines/, `heron.inference.load` raises
with the command that builds it (`python -m heron.inference.benchmark ...`). It
never builds one itself: that takes minutes, longer on a Jetson, and would look like
a hung node.

`allow_pytorch_fallback` (on by default on the robot) runs the PyTorch model
instead, and says so in the log and in the backend the nodes report. Heron's own
evaluation and benchmark never fall back: a silent fallback once made every
"TensorRT" measurement a PyTorch one.

## Consequences

- Before a first launch on a machine, build the engines with heron benchmark (see
  heron_experiment/heron_packages/README.md).
- A run's config/heron_package_manifest.yaml and the logged backend say what
  actually ran.
