# Run the network through heron.inference, never a copy of it

Status: accepted

heron_experiment calls `heron.inference.load(package, precision)` and keeps no
inference code of its own: no model loading, no TensorRT runtime, no input encoding.
capacitynet had its own (ReachabilityEngine, TRTModel), and the copies drifted from
training and from each other: the encoding was a hand-set `normalize_input` flag,
the obstacle transformer hard-coded the encoded value of free space (-1), and a config
loaded a checkpoint trained with ELU into a ReLU network without any error.

## Consequences

- Heron must be installed where the nodes run (`pip install --no-deps -e /home/heron`
  in the container). heron_experiment cannot run without the training repository.
- What `heron evaluate` and `heron benchmark` measure is what the robot runs.
- heron_experiment handles occupancy only (True = obstacle); Heron encodes it with
  the Experiment's own val transform.
- The curobo side stays here (core/reachability_model.py): the SparseVoxelGrid index
  order and the check of a voxel grid against the Heron Package's grid.
