# Hand Scenes to Heron as NumPy, one at a time, until measured otherwise

Status: accepted

The Candidates' Scenes are built on the GPU, then each goes to `Predictor.predict`
as a NumPy array: Heron applies the Experiment's val transform with pytorch-3dunet's
own Transformer, which works on NumPy. Keeping everything on the GPU would mean
writing that transform again in torch, a second copy of the encoding (see ADR 0001).

One Scene per call, because batching gains nothing on the Orin (39.8 ms per map
alone, 41.1 ms per map in a batch of 9; docs/OPTIMIZATIONS.md) and Heron's engines
are built for one input.

## Consequences

- Each cycle copies the Scenes to the CPU and the maps back. On the Orin, CPU and
  GPU share memory, and the min-max transform of 128³ voxels costs a few ms against
  ~40 ms of network per Candidate.
- Revisit only with a measurement showing that this cost matters.
