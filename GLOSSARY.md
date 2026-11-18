# heron_ros

heron_ros runs Heron's reachability network on a mobile manipulator and uses it to
place the base before grasping: it scores how reachable a region of the workspace
is from the current base pose and from candidate poses around it, and moves the
base up that gradient.

Terms of the network and its data — **Scene**, **Reachability map**, **Voxel size**,
**Experiment**, **Package**, **Engine** — are Heron's, defined in Heron's
GLOSSARY.md and not redefined here. In this repository, write **Heron Package**
rather than "package" alone, which is ambiguous next to ROS packages.

## Language

### Positioning

**Workspace region**:
Where the arm is about to work, as a set of spheres: the MPC goal, plus points of the
MPC predicted path ahead of the arm. A fiducial marker stands in when there is no goal.
_Avoid_: workspace (alone), target zone

**Q**:
How reachable the Workspace region is from one base pose: the mean of the Reachability
map over each sphere, every sphere weighing the same (1/N) whatever its size or overlap.
_Avoid_: quality score (in prose), reachability score

**Candidate**:
One virtual base pose around the current one, on a grid_size × grid_size lattice of
step δ. The robot does not move to it: its Scene is the current Scene translated by
the opposite of its offset.
_Avoid_: neighbour, sample

**Gradient**:
∇Q in the floor plane, from the Q of every Candidate (least-squares plane fit, or
central differences). The base command follows it.

### Running

**Experience**:
One experimental condition of the thesis (a setup, base fixed or moving). Groups runs
under results/<experience>/.
_Avoid_: experiment (that is Heron's training run)

**Run**:
One launch of an Experience, recorded in results/<experience>/<run_id>/: raw/,
tables/, figures/, config/.
_Avoid_: test, session

**Episode**:
One pick-and-place cycle within a Run, from leaving IDLE to returning to it.
