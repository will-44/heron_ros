# Hand Scenes to Heron as NumPy, one at a time, until measured otherwise

Status: superseded by 0004 (the encoding is a value in the manifest; the Scenes stay on the GPU)

The Candidates' Scenes are built on the GPU, then each goes to `Predictor.predict`
as a NumPy array: Heron applies the Experiment's val transform with pytorch-3dunet's
own Transformer, which works on NumPy. Keeping everything on the GPU would mean
writing that transform again in torch, a second copy of the encoding (see ADR 0001).

One Scene per call, because batching gains nothing on the Orin (39.8 ms per map
alone, 41.1 ms per map in a batch of 9; docs/OPTIMIZATIONS.md) and Heron's engines
are built for one input.

## Consequences

- Each cycle copies the Scenes to the CPU and the maps back, and applies the val
  transform on the CPU.
- Revisit only with a measurement showing that this cost matters.

## Measured (2026-11-18, parity test on the Orin)

The estimate above ("a few ms") was wrong.
- Heron benchmark, fp16 engine: 45.2 ms end to end against 33.8 ms of network, so
  about 11 ms per Scene in copies and transform, about 100 ms per 9-Candidate cycle.
- On a bag replay, inference plus scoring takes about 520 ms, against 380 ms for
  capacitynet's batched engine.
- It still fits the deployed `cycle_period` of 1.0 s. Moving the val transform
  onto the GPU is the lever if the cycle period goes down or `grid_size` goes up.
  The numbers are in `results/parity/20261118_151100_capacitynet_vs_heron/`.
