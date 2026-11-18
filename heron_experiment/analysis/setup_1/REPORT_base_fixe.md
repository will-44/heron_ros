# Experiment 2 — Fixed base with the reachability controller running, and comparison with experiment 1

**Run:** `setup_1/20260916_141632_base_fixe` (recorded as `run_20260916_141542`), 2026-09-16 14:15. The run was still recording when this analysis was made; everything below is the **30-episode snapshot, 0–29, taken at 16:45**, and the loader is pinned to it (`SNAPSHOT_MAX_EP = 29` in `analyze_exp2.py`) so that re-running the scripts reproduces these numbers. The run reached 32 episodes; raise the pin and refresh the numbers here to fold in episodes 30–31.
**Condition:** identical stack and parameters to experiment 1 (`base_control:=true`: the gradient controller scored Q, computed ∇Q and published `/cmd_vel` every second during both approach legs), but the base's motor relay was disabled, so **the base did not move**. Experiment 1 (`setup_1/20260916_110958_base_mobile`, 35 episodes, base moving) is the reference; its report is [`REPORT_base_mobile.md`](REPORT_base_mobile.md).
**Reproduce:** from `heron_experiment/`, `python3 analysis/setup_1/analyze_exp2.py` (this run's own figures `../../results/setup_1/20260916_141632_base_fixe/figures/fig_*.png` and `../../results/setup_1/20260916_141632_base_fixe/tables/*`), then `python3 analysis/setup_1/compare_runs.py` (side-by-side figures `../../results/setup_1/analysis/comparison/figures/cmp_*.png` and `../../results/setup_1/analysis/comparison/tables/cmp_*`). Both scripts reuse `analyze_exp1.py`, imported from the same directory.

---

## Abstract

With the base fixed, 5 cycles ended with a missed grasp or release; those are not scored (the base is static in both conditions when the grasp is attempted, so the failure belongs to the AprilTag placement and the arm's 5 mm accuracy, not to the base), leaving 23 scored cycles of which 18 completed (**78 %, vs 92 % with the base moving**) and 5 ended with an arm collision — all five collisions in the place phase, against 2 of 25 (both in the pick phase) when the base moved. The pick approach is unaffected by the base (41–42 s in both runs), but **the place approach takes 71 s at a fixed base against 42 s when the base moves** (+29 s, +69 %), which lengthens the whole cycle from 178 s to 214 s. The arm's MPC has the scene-collision constraint active twice as often during a fixed-base place approach (5.1 % vs 2.7 % of steps in successful cycles), and the goal-sphere reachability score when the arm arrives at the pre-place pose is 0.72 against 0.83. The fixed-base run also settles two questions raised by experiment 1. (i) The goal-sphere score is **constant** during the pick approach when the base does not move (0.79, per-leg σ = 0.009), so the systematic dip to 0.62 observed in experiment 1 was produced by the base's own initial backward motion, not by the arm; and the direction the controller wants to go follows the same time course whether or not the base moves (back-right for 10 s, then forward-right), i.e. it is set by the arm's motion along the approach, not by where the base is. (ii) During the place approach the goal-sphere score at a fixed base falls steadily from 0.77 to 0.69 as the arm carries the object toward the goal; the base motion of experiment 1 held it at 0.76–0.79. The cycle-to-cycle fluctuation of Q is the same in both runs (σ = 0.053 vs 0.060), so it is not caused by the base's motion, and the compute load is identical (MPC 204 vs 208 ms, controller cycle 697 vs 701 ms), which makes the two runs a clean A/B.

---

## 1. Setup

Same hardware, software and parameters as experiment 1 (§1 of its report): Heron (omnidirectional base + Doosan M1013, Jetson Orin AGX), `gradient_base_controller` with the 3-sphere region (goal + 2 MPC-path spheres, r = 0.10 m), 3×3 candidates at δ = 0.10 m, least-squares gradient, `base_speed = 0.02 m/s`, `cycle_period = 1.0 s`, cuRobo MPC with a 400 ms budget, `workspace_reachability_node` as recorder. The only difference is the base's motor relay, disabled for this run.

**Base pose.** Every cycle starts from the same home pose as experiment 1: the pick goal is seen by the arm at (0.43, −0.66) m in both runs. The base was re-placed by hand after episode 18, ≈ 5 cm further from the table: episodes 0–17 start from *pose A* (pick goal x ≈ 0.43 m, place goal (0.55, 0.60) m), episodes 19–29 from *pose B* (pick goal x ≈ 0.48 m, place goal (0.65, 0.60) m). Both poses are kept and tagged in `../../results/setup_1/20260916_141632_base_fixe/tables/episodes.csv` (`pose` column).

**Was the base really still?** Over the 23 scored episodes the controller commanded 20.1 m of motion in total (1 031 cycles at 0.02 m/s). The shift of the fixed-in-world MPC goal in the arm frame over each leg — the independent observation that validated the base motion in experiment 1 (slope 1.001, r = 0.991) — is uncorrelated with the commanded displacement (slope 0.019, r = 0.142, Figure 2). Episodes 16 and 18, in which the base did move, are excluded and appear in no figure or statistic of this report.

### 1.1 Outcome classification

`episodes_index.csv` only records how `brain` ended the cycle; the experimenter's notes give the cause:

| Class | n | Episodes | Rule |
|---|---|---|---|
| **success** | 18 | 0, 1, 3, 4, 6, 7, 11, 12, 14, 15, 17, 20, 22–27 | pick *and* place completed. Episodes 1, 3, 4 were cancelled during the final homing after the place, and count as successes like episodes 24/27 of experiment 1. |
| **grasp/release failed** *(not scored)* | 5 | 5, 8, 10, 13 (`Grasp failed: Sequence failed`), 2 (`Release failed`) | the arm reached the object but the grasp (or, for ep 2, the release after a 134 s approach) failed; no collision — neither a success nor a collision |
| **collision** | 5 | 9, 21, 28 (during the place approach), 19 (during the place move), 29 | arm collision with the environment → cancelled |
| **excluded** | 2 | 16, 18 | the base moved (relay re-enabled) |

Data-valid attempts: **28** (every trajectory, Q and timing statistic below). **Scored attempts: 23** — the 5 grasp/release failures are excluded from the success rate by the rule agreed with the experimenter (2026-09-23), for the same reason they are in experiment 1: the base is static at that moment. Success rate over scored attempts: 18/23 = **78 %** (18/28 = 64 % with them counted as failures). Per pose: A (n = 17) 11 success / 5 grasp failed / 1 collision; B (n = 11) 7 success / 0 / 4 collisions.

---

## 2. Results — experiment 2 on its own

![timeline](../../results/setup_1/20260916_141632_base_fixe/figures/fig_timeline.png)
*Figure 1 — Episodes by phase; black ticks are the controller cycles in which a non-zero velocity was published (not executed).*

![odometry](../../results/setup_1/20260916_141632_base_fixe/figures/fig_odometry_check.png)
*Figure 2 — Goal shift in the arm frame vs integrated command, per leg: the base did not follow.*

### 2.1 Cycle timing

A successful cycle lasts a median **214 s** (IQR 189–236 s, min 171, max 358 s). Median phase durations (successful cycles): home-enter 37.8 s, pick approach 41.8 s, grasp move 5.1 s, **place approach 71.4 s**, place move 13.0 s, home-exit 21.8 s. The place approach is the only phase that differs from experiment 1; it is long in both base poses (successful cycles: pose A 69.5 s, pose B 73.2 s) and highly variable (40–196 s).

### 2.2 What the controller did (and would have done)

The controller was gated open only during the two approach legs (1 084 active cycles, 1 031 with a non-zero command, always |v| = 0.020 m/s) and its cycle took 697 ms (max 923 ms) — the same activity profile as in experiment 1. Its stop reasons are also the same: the arrival latch in 36 of the 46 scored legs, the interlock in 8, gradient convergence in 2. Because the base never moved, the "commanded displacement" is a virtual path (0.36 m per pick leg, 0.45 m per place leg) and the useful quantity is the **direction** the controller asked for (Figure 3):

![wanted direction](../../results/setup_1/20260916_141632_base_fixe/figures/fig_wanted_direction.png)
*Figure 3 — Direction of ∇Q in the arm frame as a function of time in the leg (radius), one dot per leg and 5 s bin; black: circular mean over legs.*

- **Pick approach:** for the first ~10 s the controller asks to move backward-right (circular mean −141° / −149°, concentration R = 0.96 / 0.85 over the legs), then swings within one bin to forward-right (−72° at 10–15 s, −44° to −49° afterwards, R ≥ 0.8). The base pose did not change during this swing: the rotation of the wanted direction is caused by the arm's motion (the two path spheres follow the MPC prediction as the arm leaves home and extends toward the table).
- **Place approach:** after a first bin in which the legs disagree (R = 0.16), the wanted direction is backward-left and remarkably steady for the whole leg (128° → 150–170°, R ≥ 0.9): from either fixed pose the controller consistently wanted to back the base away from the place goal.

### 2.3 Reachability scores at a fixed base

![Q over legs](../../results/setup_1/20260916_141632_base_fixe/figures/fig_q_legs.png)
*Figure 4 — Controller-side scores over every leg at the fixed base; dots mark the last non-zero command. Every leg is drawn to its own end (thin); the median is solid while at least half the legs are still running and dashed once only a minority are left, and the step curve along the bottom of each panel is the number of legs still running. The earlier version cut every curve as soon as fewer than 5 legs remained, which both hid the longest legs and turned the tail of the median into an average over a self-selected minority.*

- **Pick approach: the goal-sphere score does not move.** Q_goal = 0.79 at every cycle of every leg (commanded-cycle IQR 0.779–0.795; median per-leg standard deviation 0.009). The path spheres, which sit 1–2 s ahead of the TCP, drop from 0.90 to 0.82 as the arm extends toward the table; the 3-sphere objective therefore falls from 0.86 to 0.81 over the leg (ΔQ < 0 in 96 % of legs) although nothing changed for the goal.
- **Place approach: the goal-sphere score decays.** Q_goal starts at 0.77 and falls to 0.69 at the last command (ΔQ_goal median −0.068, positive in only 17 % of legs; per-leg σ 0.060) and to ~0.60 by 60 s in the legs that last that long; the recorder's Q_goal over the last 3 s of the approach is 0.72. The path spheres rise slightly (0.80 → 0.83). Since the base is still, this decay is produced by the arm carrying the object toward the goal.

### 2.4 Failure diagnostics

![outcome diag](../../results/setup_1/20260916_141632_base_fixe/figures/fig_outcome_diag.png)
*Figure 5 — Place-approach MPC diagnostics by outcome.*

In every successful cycle the arm reached the pre-place pose (final FK error 1 mm); in the 5 collision cycles it did not (median 0.58 m) and the scene-collision constraint was active in 8.9 % of MPC steps (5.1 % in successful cycles, already twice the 2.7 % of experiment 1). The pick approach is clean in all classes (FK error 1 mm, scene-collision fraction 0.000–0.006). The five grasp/release failures are excluded from this diagnostic as from every other statistic; in the recordings the arm was correctly positioned at the object when they occurred (pick-approach FK error 1 mm), i.e. they are grasp-execution issues, not positioning issues.

### 2.5 Compute load

Controller cycle 697 ms mean, MPC 204 ms mean / 316 ms p95 with 1.1 % of steps over the 400 ms budget, 5 637 MPC steps: within 2 % of experiment 1 on every count (`../../results/setup_1/20260916_141632_base_fixe/figures/fig_timing.png`). The two runs put the same load on the Orin, as intended by the `base_control` A/B design.

---

## 3. Comparison: base moving (experiment 1) vs base fixed (experiment 2)

Blue = experiment 1 (base moving, 25 scored cycles), orange = experiment 2 (base fixed, 23 scored cycles). Excluded episodes — infrastructure/operator and grasp/release failures — are dropped before any figure or statistic below is computed. All numbers are in `../../results/setup_1/analysis/comparison/tables/cmp_summary.md`.

### 3.1 Outcomes

![outcomes](../../results/setup_1/analysis/comparison/figures/cmp_outcomes.png)
*Figure 6 — Outcome of the valid cycles.*

| | base moving | base fixed |
|---|---|---|
| data-valid cycles | 32 | 28 |
| grasp-quality problems (not scored) | 7 bad-grasp cancels (22 %) | 5 missed grasps/releases (18 %) |
| scored cycles | 25 | 23 |
| success | 23 (**92 %**) | 18 (**78 %**) |
| arm collisions | 2 (8 %), both in the pick phase | 5 (22 %), all in the place phase |
| successful cycle duration, median | 178 s | 214 s |

Grasp-quality problems occur at the same rate in both runs and manifest differently (badly seized object that perturbs the map vs missed grasp); they do not involve the base. The difference that does involve the base is the **place phase**: five collisions at a fixed base versus none when the base moved, and a 29 s longer approach when it does not collide. With 23–25 scored cycles per arm these collision counts are indicative rather than conclusive (5/23 vs 0/25 in the place phase; Fisher exact on the scored outcomes, 18/23 vs 23/25, p = 0.24), but they go in the same direction as the timing and MPC evidence below.

### 3.2 Timing

![phase durations](../../results/setup_1/analysis/comparison/figures/cmp_phase_durations.png)
*Figure 7 — Phase durations of successful cycles.*

Home-enter (38 s), pick approach (41 vs 42 s), grasp move (5 s), place move (12 vs 13 s) and home-exit (22 s) are identical. The place approach is **42 s with the base moving and 71 s at a fixed base** (means 48 vs 84 s); its spread also grows (IQR 39–49 s vs 57–93 s). Since the two base poses of experiment 2 give the same long approach (69.5 and 73.2 s), this is not an artefact of the re-placement.

### 3.3 The scores during the approach

![Q legs](../../results/setup_1/analysis/comparison/figures/cmp_q_legs.png)
*Figure 8 — Median and inter-quartile band of the controller's scores over the legs, both runs. Every leg is drawn to its own end (thin); the median is solid while at least half the legs are still running and dashed once only a minority are left, and the step curve along the bottom of each panel is the number of legs still running. The earlier version cut every curve as soon as fewer than 5 legs remained, which both hid the longest legs and turned the tail of the median into an average over a self-selected minority.*

![Q arrival](../../results/setup_1/analysis/comparison/figures/cmp_q_arrival.png)
*Figure 9 — Goal-sphere score at the start of the leg, at the last command, and over the last 3 s of the approach (recorder).*

| Q goal sphere (median) | pick, base moving | pick, base fixed | place, base moving | place, base fixed |
|---|---|---|---|---|
| leg start | 0.79 | 0.79 | 0.72 | 0.77 |
| last command | 0.84 | 0.79 | 0.76 | 0.69 |
| end of approach (recorder, last 3 s) | 0.80 | 0.78 | **0.83** | **0.72** |
| ΔQ_goal over the command (median, % of legs > 0) | +0.04 (72 %) | −0.00 (43 %) | +0.02 (65 %) | −0.07 (17 %) |

**Pick approach.** At a fixed base Q_goal is flat at 0.79 for the whole leg (Figure 8, centre-top, orange band of width 0.01). The moving base of experiment 1 follows the same flat score until ~3 s, dips to 0.62 at 10 s and recovers to 0.84 by 20 s. Together with Figure 10 (below), the reading is unambiguous: **the dip was caused by the base's own motion** — the base followed a gradient that pointed backward for the first 10 s, which moved the goal to x ≈ 0.6 m in the arm frame where the goal sphere scores lower, then followed the gradient forward and recovered. The net effect at the grasp is small (recorder Q_goal 0.80 vs 0.78) and the pick approach is not faster.

The path spheres tell the other half of the story: at a fixed base they fall from 0.90 to 0.82 as the arm extends (the corridor ahead of the TCP approaches the table), and the 3-sphere objective falls with them (0.86 → 0.81). With the base moving they stay at 0.86–0.93 and the objective stays flat (0.86 → 0.85). The controller *did* optimise what it was asked to optimise; the objective simply weighs the corridor twice as much as the goal.

**Place approach.** At a fixed base Q_goal decays from 0.77 to 0.69 (0.60 by 60 s) as the arm carries the object toward the goal; with the base moving it rises from 0.72 to 0.76 and the recorder reads 0.83 over the last 3 s of the approach, against 0.72. The 3-sphere objective rises from 0.79 to 0.82–0.86 with the base moving and stays at 0.79 at the fixed base. Here the base motion produced both a better score at the place and a shorter approach.

### 3.4 The wanted direction is set by the arm, not by the base position

![direction](../../results/setup_1/analysis/comparison/figures/cmp_direction.png)
*Figure 10 — Circular mean of the wanted direction per 5 s bin, both runs.*

In the pick approach the two curves coincide (−141°/−156° in the first 5 s, −149°/−191° at 5–10 s, then −72°/−96°, −44°/−39°, −49°/−45°): the direction the controller asks for at time *t* in the leg is the same whether the base has been moving for *t* seconds or not. The gradient is therefore a function of the arm's progress along its approach (through the path spheres) far more than of the base's position, which is why a quasi-static gradient descent moves the base back and forth in this leg (path efficiency 0.42 in experiment 1). In the place approach the fixed-base controller asks to back away to the left (150–170°) throughout; the moving base, which is actually doing it, sees the wanted direction rotate from back-left (114–132°) to left (96°) as it moves — here the field does depend on position, and the base converges to where the goal-sphere score is higher.

### 3.5 MPC behaviour, arm metrics, noise and compute

![mpc](../../results/setup_1/analysis/comparison/figures/cmp_mpc.png)
*Figure 11 — Arm MPC over the approach legs.*

- **Scene-collision constraint** (fraction of MPC steps in which it is active), successful cycles: pick 0.013 vs 0.000, place **0.027 vs 0.051**. The fixed-base place approach spends twice as many steps with the obstacle constraint active; in the cycles that ended in a collision at the fixed base it is 0.089.
- **Arrival**: FK error at the end of a successful approach 1–2 mm in both runs.
- **Manipulability** (`../../results/setup_1/analysis/comparison/figures/cmp_manip.png`): no benefit of the base motion — grasp move 0.217 (moving) vs 0.240 (fixed), place move 0.210 vs 0.217, approaches 0.12/0.19 vs 0.13/0.19. Those are phase means over legs of different lengths; the time-resolved traces ([`../manipulability/`](../manipulability/README.md)) say it more precisely: on the empty table the two conditions follow the same manipulability curve and end at the same value (pick 0.233 fixed vs 0.214 moving, place 0.217 vs 0.210), but the moving base settles into its final posture after 18.4 s against 30.3 s (p = 0.018) because its leg is 42 s long instead of 67 s. Neither condition goes anywhere near a singularity during the place approach here — w never drops below 0.05 at all, in either condition. The *pick* approach does: 1.4 s below w = 0.01 in both conditions (p = 0.43), at the same instant (12–18 s into the leg), which makes it a property of the arm leaving the home posture rather than of the base. (An earlier wording counted time below w = 0.05 as "near-singular"; it is not — 34 % of all samples in these runs sit below that level and the parked arm itself sits at w = 0.012–0.020.)
- **Joint velocities** ([`../velocity/`](../velocity/README.md)): the instantaneous speeds do not separate the two conditions (mean ‖q̇‖ 0.080 fixed vs 0.095 moving on the place approach, p = 0.15; 0.102 vs 0.094 on the pick, p = 0.22) — the MPC bounds them the same way whatever the base does. The **cumulated joint travel** does: 11.3 rad fixed against 8.3 rad moving over the place approach (p = 0.010), for equivalent travel on the pick (8.81 vs 8.42 rad). Same speed, a third more distance. The commanded base speed is a single value, exactly 0.020 m/s whenever non-zero (18–20 % of the `/cmd_vel` messages): the law is bang-bang, the taper and deadband never engage.
- **Joint accelerations** ([`../acceleration/`](../acceleration/README.md)): the logged `acc_joint_*` column is a second finite difference over irregularly-timed messages and is noise (lag-1 autocorrelation −0.59 here, against +0.99 once re-derived on a uniform 100 Hz grid) — **no acceleration figure taken from it is usable**. Re-derived, the amplitudes do not separate the conditions (mean ‖q̈‖ 0.157 fixed vs 0.152 moving on the pick, 0.169 vs 0.173 on the place, p ≥ 0.41). The SPARC smoothness gap on the place approach (−3.71 vs −2.81, p = 0.039) tracks the leg duration (ρ = −0.79 within the fixed-base group alone) and should not be quoted as a smoothness result. As in setup 2, **every pick approach sweeps q₅ through 0** (per-leg minimum |q₅| = 0.0° in median, 12–18 s into the leg, identical in both conditions), which is the wrist singularity behind the near-singular passage noted above, and J5 is correspondingly the most accelerated axis on that leg.
- **Noise of Q** (`../../results/setup_1/analysis/comparison/figures/cmp_timing.png`, right): the cycle-to-cycle change of the objective has σ = 0.060 with the base moving and 0.053 with it fixed. The fluctuation that hides the gradient's per-cycle prediction (≈ 0.004) in experiment 1 is therefore intrinsic to the score under a moving arm, not induced by the base motion; the sign agreement between predicted and measured ΔQ is 46 % (moving) and 43 % (fixed — where the prediction has no physical meaning and 50 % is expected).
- **Compute**: controller cycle 701 vs 697 ms, MPC 208 vs 204 ms mean, 1.1 % vs 1.1 % over budget. Same load.

---

## 4. Discussion

**What the base motion buys.** With the numbers of these two runs, the benefit of the reachability-gradient base motion is concentrated in the place phase: a 29 s shorter approach (−41 % of the phase, −17 % of the cycle), a higher goal-sphere score at the place (0.83 vs 0.72), half as many MPC steps fighting the obstacle constraint, and no collision in 25 scored cycles against 5 in 23. In the pick phase the base motion changes neither the duration nor the collision count nor the manipulability at the grasp; it raises the goal-sphere score slightly at the end of the command (0.84 vs 0.79) but not at the grasp itself (0.80 vs 0.78).

**Why the place phase and not the pick phase.** From the home pose the pick goal is already in the region the model scores best (Q_goal 0.79 at a fixed base, flat during the whole approach): there is nothing for the base to gain, and the initial backward excursion of experiment 1 is a detour driven by the path spheres. The place goal, on the other hand, is at (0.55–0.65, 0.60) m in the arm frame with the object in the gripper, and its score decays as the arm approaches; the MPC is visibly constrained there (longer, more variable approach, obstacle constraint active). Backing the base away to the left — which is what the controller consistently asks for at the fixed pose, and what it did in experiment 1 (goal brought to (0.78, 0.45) m) — removes the constraint.

**What the fixed-base run says about the gradient.** In the pick leg the wanted direction has the same time course with and without base motion: the field the controller descends is dominated by the arm's progress (the path spheres following the MPC prediction), so its gradient at a given instant says more about where the arm is on its approach than about where the base should be. This is consistent with experiment 1's finding that the per-cycle prediction ∇Q·v·Δt does not correlate with the measured ΔQ, and with the identical Q fluctuation in both runs. The controller nevertheless converges to a repeatable and useful pose in the place leg because there the direction is stable over the leg (R ≥ 0.9) and position-dependent.

**Caveats.** (1) Small samples: 23 vs 25 scored cycles, 5 vs 2 collisions; the timing and score differences are large and consistent, the collision difference is suggestive. (2) The fixed-base run was performed after the moving-base run on the same day; the scene, lighting and object placement were nominally identical but not controlled. (3) The re-placement after episode 18 introduces two fixed poses; both give the same long place approach, but 4 of the 5 collisions occurred in pose B. (4) The run was still recording at the time of this snapshot.

---

## 5. Recommendations

1. **Keep the base motion for the place leg; question it for the pick leg.** A variant that enables the controller only for the place approach (or gates it on a low initial Q_goal) would keep the measured benefit and remove the pick-leg detour.
2. **Re-weight the objective toward the goal sphere** (the corridor currently carries 2/3 of the weight and drives the backward excursion at pick); with the per-candidate sphere means logged, this can be evaluated offline on the bags of both runs.
3. **Start the place approach from a deliberately poor pose** in a further run to measure the controller's capacity to recover, now that the home pose is known to be already good for the pick.
4. **Log the base-side Twist topic and the `fk_batch` service** (still missing in this run), and keep the relay state in the recorder so that a violated fixed-base condition (episodes 16, 18) is detected automatically rather than by hand.

---

## Appendix A — Files

| Path | Content |
|---|---|
| `analyze_exp2.py` | this run's figures and tables (`../../results/setup_1/20260916_141632_base_fixe/figures/fig_*.png`, `../../results/setup_1/20260916_141632_base_fixe/tables/episodes|legs|cycle_pairs|preplace_mpc_diag|wanted_direction.csv`, `summary.json`) |
| `compare_runs.py` | side-by-side figures and tables (`../../results/setup_1/analysis/comparison/figures/cmp_*.png`, `../../results/setup_1/analysis/comparison/tables/cmp_*.csv|md`) |
| `analyze_exp1.py` | shared loading, per-leg and per-cycle code (run-agnostic; also regenerates experiment 1) |
| `../../results/setup_1/analysis/comparison/tables/cmp_summary.md` | every number of §3 side by side |
| `../../results/setup_1/analysis/comparison/tables/cmp_legs.csv` | one row per (run, episode, leg) |

## Appendix B — Per-episode table (snapshot of 30 episodes)

Pose A/B: base pose before/after the manual re-placement. Net = virtual displacement integrated from the (unexecuted) command; ΔQ = change of the 3-sphere objective between the first full-region cycle and the last command.

| ep | pose | class | dur. (s) | last phase | pick net (m) | pick ΔQ | place net (m) | place ΔQ | note |
|---|---|---|---|---|---|---|---|---|---|
| 0 | A | success | 173 | home_exit | 0.07 | -0.18 | 0.36 | -0.07 | pick-and-place completed |
| 1 | A | success | 300 | home_exit | 0.22 | -0.06 | 0.72 | +0.05 | pick-and-place completed; cancelled during the final homing |
| 2 | A | excluded | 228 | release: idle |  |  |  |  | release sequence failed after a 134 s place approach (no collision) |
| 3 | A | success | 227 | home_exit | 0.18 | -0.05 | 0.21 | +0.08 | pick-and-place completed; cancelled during the final homing |
| 4 | A | success | 358 | home_exit | 0.25 | -0.12 | 0.82 | -0.28 | pick-and-place completed; cancelled during the final homing |
| 5 | A | excluded | 84 | grasp: idle |  |  |  |  | grasp sequence failed: arm at the object but grasp missed (no collision) |
| 6 | A | success | 213 | home_exit | 0.23 | -0.07 | 0.47 | -0.09 | pick-and-place completed |
| 7 | A | success | 204 | home_exit | 0.18 | -0.05 | 0.35 | +0.00 | pick-and-place completed |
| 8 | A | excluded | 93 | grasp: close_gripper |  |  |  |  | grasp sequence failed: arm at the object but grasp missed (no collision) |
| 9 | A | collision | 124 | release: idle | 0.20 | -0.03 | 0.35 | +0.03 | arm collision with the environment during the place approach; cancelled |
| 10 | A | excluded | 87 | grasp: idle |  |  |  |  | grasp sequence failed: arm at the object but grasp missed (no collision) |
| 11 | A | success | 239 | home_exit | 0.21 | -0.04 | 0.56 | -0.07 | pick-and-place completed |
| 12 | A | success | 203 | home_exit | 0.18 | -0.06 | 0.05 | -0.00 | pick-and-place completed |
| 13 | A | excluded | 79 | grasp: idle |  |  |  |  | grasp sequence failed: arm at the object but grasp missed (no collision) |
| 14 | A | success | 187 | home_exit | 0.30 | -0.08 | 0.36 | +0.08 | pick-and-place completed |
| 15 | A | success | 197 | home_exit | 0.29 | -0.04 | 0.25 | +0.01 | pick-and-place completed |
| 16 | A | excluded | 226 | home_exit |  |  |  |  | base moved during the cycle (fixed-base condition violated) |
| 17 | A | success | 268 | home_exit | 0.14 | -0.04 | 0.56 | -0.00 | pick-and-place completed |
| 18 | A | excluded | 149 | release: idle |  |  |  |  | base moved during the cycle (fixed-base condition violated) |
| 19 | B | collision | 117 | release: idle | 0.13 | -0.03 | 0.13 | +0.07 | arm collision with the environment during the place move; cancelled |
| 20 | B | success | 171 | home_exit | 0.27 | -0.03 | 0.38 | +0.11 | pick-and-place completed |
| 21 | B | collision | 124 | release: idle | 0.20 | -0.03 | 0.44 | -0.12 | arm collision with the environment during the place approach; cancelled |
| 22 | B | success | 282 | home_exit | 0.34 | -0.04 | 0.19 | +0.03 | pick-and-place completed |
| 23 | B | success | 173 | home_exit | 0.25 | -0.04 | 0.25 | +0.05 | pick-and-place completed |
| 24 | B | success | 225 | home_exit | 0.10 | -0.04 | 0.33 | +0.04 | pick-and-place completed |
| 25 | B | success | 183 | home_exit | 0.20 | -0.05 | 0.50 | -0.02 | pick-and-place completed |
| 26 | B | success | 215 | home_exit | 0.28 | -0.05 | 0.36 | -0.02 | pick-and-place completed |
| 27 | B | success | 216 | home_exit | 0.37 | -0.08 | 0.31 | -0.09 | pick-and-place completed |
| 28 | B | collision | 128 | release: move_preplace | 0.32 | +0.03 | 0.21 | +0.01 | arm collision with the environment; cancelled |
| 29 | B | collision | 99 | release: idle | 0.22 | -0.06 | 0.05 | +0.01 | arm collision with the environment; cancelled |
