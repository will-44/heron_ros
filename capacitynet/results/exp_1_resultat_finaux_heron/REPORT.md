# Experiment 1 — Reachability-gradient base positioning during live pick-and-place

**Run:** `run_20260916_110908` (35 episodes, 2026-09-16 11:09 → 13:27), gradient base controller **enabled** (`base_control:=true`).
**Data:** this directory (`episodes_index.csv`, `phase_summary.csv`, `control_debug.csv`, one `episode_*/` folder per cycle).
**Reproduce:** `python3 analysis/analyze_exp1.py` regenerates every figure in `exp_1_moving_figures/` and every table in `exp_1_moving_tables/` (headline numbers in `exp_1_moving_tables/summary.json`).

---

## Abstract

Thirty-five pick-and-place cycles were run on the Heron mobile manipulator with the reachability-gradient base controller active during both arm approach legs (pre-grasp and pre-place). Three episodes were invalidated by setup/infrastructure problems and seven more were cancelled after a bad grasp; a grasp fails with the base standing still, for reasons outside the base controller (AprilTag placement and the arm's 5 mm positioning accuracy), so those seven are excluded outright: they contribute to no number in this report. **23 of the 25 scored cycles (92 %) completed the pick-and-place**; the other 2 ended in an arm collision. The controller was gated open only during the two approach legs, moved the base at its 0.02 m/s ceiling for a median of 20–21 s per leg (≈ 0.40–0.41 m of path, 0.13–0.26 m net), and was stopped in three cases out of four by the arm's arrival rather than by gradient convergence. The commanded motion was executed faithfully by the base (slope 1.001, r = 0.991 between the integrated command and the displacement of the fixed goal seen from the arm frame). The controller drives the base to a repeatable position relative to the target (goal brought from 57° to 48° off the arm's forward axis in pick, 45° → 30° in place, at nearly constant range), and the goal-sphere reachability score improves modestly over a leg (median +0.041 in pick, +0.016 in place; positive in 72 % / 65 % of legs). Two findings qualify the method: (i) the per-cycle change of Q is not predicted by the gradient (sign agreement 46 %, r = 0.06 — measurement noise is ten times the predicted signal), and (ii) in nearly every pick approach the base first backs away for ≈ 10 s while the goal-sphere score drops, then reverses; the same goal position scores +0.14 higher on the way back than on the way out, i.e. the score is **non-stationary during the arm motion**, which a quasi-static gradient descent cannot exploit. Real-time budgets held: the 9-candidate gradient cycle took 0.70 s (period 1.0 s), and the arm MPC exceeded its 400 ms budget on 1.1 % of steps (1.9 % while the controller's inference was running).

---

## 1. Setup and protocol

### 1.1 System

| Item | Value |
|---|---|
| Platform | Heron: Clearpath-type omnidirectional base + Doosan M1013 arm, Jetson Orin AGX |
| Task | pick an ArUco-tagged object from a table on the right of the robot, place it on the left, return home |
| Task sequencing | `brain` → `grasp` / `release` state machines, cuRobo MPC for the arm (`/mpc_predicted_path`, 400 ms budget) |
| Base controller | `gradient_base_controller` (`WorkspaceRegionSource` + `ReachabilityPipeline` + `BaseCommander`) |
| Reachability model | UNet3D (`/home/ros2_ws/src/test.yaml`), PyTorch, same weights in the controller and the recorder |
| Recorder | `workspace_reachability_node`, second UNet3D instance on the same GPU, un-throttled (`rm_min_period_s: 0`) |

### 1.2 Controller configuration (`config/gradient_base_controller_params.yaml`)

| Parameter | Value | Effect in this run |
|---|---|---|
| `cycle_period` | 1.0 s | one gradient evaluation per second |
| `grid_size` / `grid_spacing` | 3 / 0.10 m | 9 translated obstacle maps per cycle, ±10 cm |
| `gradient_method` | least_squares | plane fit over the 9 candidates |
| region | goal sphere r = 0.10 m + 2 path spheres r = 0.10 m at `path_horizon_s = 2.0, 1.0` | Q = mean of 3 spheres, each weighing 1/3 |
| `base_speed` | 0.02 m/s | commanded speed is exactly 0.02 m/s whenever moving (normalized-direction law) |
| `min_gradient` | 0.025 1/m | convergence deadband |
| `gradient_taper_ref` | 0.02 1/m | **inactive**: below `min_gradient`, so the taper never engages and the command is bang-bang (0 or 0.02 m/s) — confirmed in the data |
| `use_static_obstacles` | false | the whole voxel grid, including any self-geometry, is translated for the candidates |
| interlocks | `grasp_motion_enable` / `release_motion_enable` (distance-based), arrival latch on region spheres touching | see §2.2 |

### 1.3 Episode definition and recorded signals

An episode is one `brain` cycle: `act_home_enter → act_grasp:* → act_release:* → act_home_exit → idle`. Every recorded row carries `(episode_id, t_rel, phase)`. Per episode: `samples.csv` (recorder Q_goal / Q_path per MPC path message, ≈ 1 Hz), `cmd_vel.csv` (controller Twist echo, 10 Hz), `mpc_diagnostics.csv` (solve time, budget, FK arrival error, constraint terms), `joint_states.csv` (+ Yoshikawa manipulability), `phases.csv`. The controller writes its own per-cycle log (`control_debug.csv`: Q at the centre, the three sphere means `ws_mean_0..2`, ∇Q, command, latches, gate state).

### 1.4 Outcome classification

`episodes_index.csv` only records how `brain` ended the cycle (success / cancelled / failure). The experimenter's notes were used to assign the class actually analysed:

| Class | n | Episodes | Rule |
|---|---|---|---|
| **success** | 23 | 0, 1, 2, 6, 7, 8, 11–15, 17, 18, 20, 21, 23, 24, 26, 27, 30, 31, 32, 34 | pick *and* place completed. Ep 24 (collision only during the final homing) and ep 27 (cancelled during the final homing) are counted here. |
| **bad-grasp cancel** *(not scored)* | 7 | 5, 9, 16, 19, 22, 25, 28 | object badly seized → the object in the gripper perturbs the obstacle map → operator cancels during the place approach |
| **collision** | 2 | 10 (during pre-grasp, cancelled after 135 s), 33 (at grasp) | arm collision |
| **excluded** | 3 | 3, 29 (operator intervention, setup error), 4 (`set_planner` service unavailable, 6 ms episode) | not a valid attempt |

Data-valid attempts: **32** (used for every trajectory, Q and timing statistic below). **Scored attempts: 25** — the 7 bad-grasp cancels are excluded from the success rate as well, by the rule agreed with the experimenter (2026-09-23): the base is static in both conditions when the grasp is attempted, and what fails is the AprilTag placement plus the arm's 5 mm accuracy, not the base controller. Success rate over scored attempts: **23 / 25 = 92.0 %** (23 / 32 = 71.9 % if the grasp cancels are counted as failures).

---

## 2. Results

### 2.1 Task outcomes and cycle timing

![timeline](exp_1_moving_figures/fig_timeline.png)
*Figure 1 — Every episode by phase; black ticks mark the controller cycles in which the base was commanded to move. The base is only ever active in the first half of the two approach legs.*

A successful cycle lasts a median **178 s** (IQR 171–194 s, min 159, max 253 s). Median phase durations over successful cycles: home-enter 37.9 s, **pick approach 41.1 s**, grasp (open→retract) 17.7 s, **place approach 42.3 s**, place (move→retract) 15.7 s, home-exit 22.3 s. The two approach legs — the only phases in which the base moves — account for 47 % of the cycle.

Both collisions happened in the pick leg. (The 7 bad-grasp cancels of §1.4 are excluded and appear in no figure or statistic; for the record they all occurred during the place approach, 20–60 s after the grasp.)

### 2.2 What the base controller did

The controller logged 1 041 active cycles (532 in pick approaches, 509 in place approaches); the gate was closed in every other phase (3 440 skipped ticks). Table 1 summarises the 48 legs (25 pick, 23 place).

**Table 1 — Per-leg controller activity (median [IQR])**

| | pick approach (n = 25) | place approach (n = 23) |
|---|---|---|
| leg (phase) duration | 41.2 s [38.8–45.4] | 42.2 s [38.5–48.8] |
| cycles with motion | 21 [17–23] | 21 [18–24] |
| time moving | 20.1 s [16.0–22.1] | 20.9 s [17.0–27.4] |
| fraction of the leg spent moving | 47 % | 47 % |
| path length commanded | 0.41 m [0.34–0.45] | 0.40 m [0.34–0.47] |
| net displacement | **0.13 m** [0.11–0.15] | **0.26 m** [0.21–0.32] |
| path efficiency (net / path) | 0.35 | 0.64 |
| direction reversals (> 90° between cycles) | 1 [0–1] | 0 [0–2] |
| median \|∇Q\| while moving | 0.20 1/m | 0.19 1/m |
| stop reason: arrival latch / interlock closed / gradient convergence | 18 / 5 / 2 | 16 / 7 / 0 |

Three observations:

1. **The base stops because the arm arrives, not because Q converged.** In 34 of 48 legs the arrival latch fired (goal sphere and path-horizon sphere touching), in 12 the grasp/release node closed its interlock (arm within its near-threshold), and only in **2 of 48** did |∇Q| fall below `min_gradient`. With `base_speed = 0.02 m/s` and a 40 s approach, the base is still at full speed when the arm reaches the target: the leg is truncated by the arm, not finished by the controller.
2. **The command is bang-bang.** Every non-zero command has |v| = 0.020 m/s (taper inactive, §1.2). Total path commanded over the run: 19.5 m.
3. **Pick legs hunt more than place legs.** Path efficiency 0.35 vs 0.64 and a median of one reversal per pick leg — §2.4 shows why.

**The base executed what was commanded.** The recorder never received the base-side Twist topic (`/r100_0597/joy_teleop/cmd_vel`; the relay used another topic), so its `x_est/y_est/dist_est` columns are zero and the shipped `base_trajectory.png` is empty. The displacement was therefore re-integrated from the controller's own 10 Hz `/cmd_vel` echo and checked against an independent observation: the MPC goal is fixed in the world, so its position in `dsr01/base_link` moves by exactly minus the base displacement. Over the 48 legs the two agree with slope **1.001** and **r = 0.991** (96 x/y components, Figure 2). The base therefore followed the commanded velocity within a few percent, and the integrated command can be used as odometry for this run.

![odometry](exp_1_moving_figures/fig_odometry_check.png)
*Figure 2 — Per-leg displacement: goal shift seen from the arm (observed) vs integrated /cmd_vel (commanded).*

### 2.3 Where the controller takes the base

![base paths](exp_1_moving_figures/fig_base_paths.png)
*Figure 3 — Base paths expressed in a frame centred on the goal (axes aligned with the arm base frame). Grey: base at leg start; coloured: base when it stopped; ×: goal.*

The start positions are tightly repeatable (σ ≈ 4 cm) because every cycle begins from the same home pose and the same table layout. The controller moves the base to a consistent end region (σ ≈ 6 cm), and the move is the same in every leg of a given type:

| | goal in arm frame at leg start (median) | goal when the base stopped | range | bearing from arm +x |
|---|---|---|---|---|
| pick approach | (0.42, −0.64) m | (0.51, −0.57) m | 0.77 → 0.77 m | −57° → −48° |
| place approach | (0.70, 0.69) m | (0.78, 0.45) m | 0.98 → 0.89 m | +45° → +30° |

In other words the controller **re-centres the target toward the arm's forward axis at nearly constant range** rather than closing the distance: it moves sideways toward the target (−y in pick, +y in place) while backing away slightly in x. This is coherent with the measured Q landscape (Figure 6): for the low pick goal (z ≈ 0.13 m) Q_goal is highest when the object sits at x ≈ 0.45–0.55 m and falls off beyond 0.6 m; for the higher place goal (z ≈ 0.25 m) the high-Q band lies at x ≈ 0.6–0.8 m, y ≈ 0.4–0.6 m, which is where the place legs end.

### 2.4 Effect on the reachability score

![Q over legs](exp_1_moving_figures/fig_q_legs.png)
*Figure 4 — Controller-side scores over every leg (thin lines), from the first cycle with the full 3-sphere region; dots mark where the base stopped; black: median. Every leg is drawn to its own end (thin); the median is solid while at least half the legs are still running and dashed once only a minority are left, and the step curve along the bottom of each panel is the number of legs still running. The earlier version cut every curve as soon as fewer than 5 legs remained, which both hid the longest legs and turned the tail of the median into an average over a self-selected minority.*

![dQ](exp_1_moving_figures/fig_dq_legs.png)
*Figure 5 — Change of each score between the first full-region cycle and the cycle at which the base stopped.*

**Table 2 — Scores at leg start and at base stop (medians)**

| | Q region mean (controller objective) | Q goal sphere | Q path spheres |
|---|---|---|---|
| pick: start → stop | 0.858 → 0.852 (Δ −0.005, 44 % > 0) | 0.791 → **0.836 (Δ +0.041, 72 % > 0)** | 0.89 → 0.86 (Δ −0.032) |
| place: start → stop | 0.794 → 0.832 (Δ +0.038, 61 % > 0) | 0.723 → **0.773 (Δ +0.016, 65 % > 0)** | 0.82 → 0.86 (Δ +0.042) |

The quantity the controller optimises — the 3-sphere mean — barely changes over a leg (it starts at 0.79–0.86, i.e. already close to the plateau of the model's output). The goal sphere, the score that matters for the grasp, improves by a small but consistent amount in both legs. The path spheres, which sit 1–2 s ahead of the TCP along the MPC prediction, are near 0.89 at the start of a pick leg (the arm is still near home) and can only fall as the arm extends toward the table; they carry two thirds of the objective's weight.

**The systematic dip in every pick approach.** The middle-top panel of Figure 4 shows the same shape in almost every pick leg: Q_goal starts at 0.79, falls to 0.59 on the median curve at ≈ 8 s (per-leg minimum: median 0.52 at 10.8 s, below 0.4 in 32 % of legs, lowest 0.19), then recovers to 0.84 by 20 s. During those first 10 s the base backs away (vx < 0 in 96 % of legs; median heading −172°, i.e. almost straight back, slightly to the right), the goal's x in the arm frame grows from 0.42 to ≈ 0.62 m, and the moment of minimum Q_goal coincides with the moment of maximum goal x in 19 of 25 legs. Between 10 and 20 s the heading swings to −57° (forward-right, vy < 0 in 83 % of legs) and Q_goal recovers as the base comes back. This detour is the reason for the 0.35 path efficiency of pick legs: the 0.41 m of commanded path buy 0.13 m of net displacement.

Figure 6 (right) shows that this is not simply the base crossing a poor region of a fixed landscape: **for the same goal x, Q_goal is on average 0.14 higher on the way back than on the way out** (11 cells with ≥ 8 samples on each side). A quadratic in the goal position explains 38 % of the variance of Q_goal in pick legs; adding time-in-leg raises it to 46 %. The reachability map — hence the score and the gradient computed from it — changes while the arm moves, independently of where the base is.

![landscape](exp_1_moving_figures/fig_q_landscape.png)
*Figure 6 — Left, centre: mean recorder Q_goal per hexagonal cell of the goal position in the arm frame; + leg start, ○ base stop. Right: pick approach, mean Q_goal vs goal x, split into the outbound (goal x increasing) and return part of each leg.*

### 2.5 Does the gradient predict what happens next?

![gradient validity](exp_1_moving_figures/fig_gradient_validity.png)
*Figure 7 — Left: predicted one-cycle change ∇Q·v·Δt vs measured Q(t+1) − Q(t), 949 consecutive moving cycles. Centre: turn of the commanded direction between consecutive cycles. Right: |∇Q| while moving.*

Over 949 pairs of consecutive moving cycles the gradient's prediction of the next cycle's change in Q has **46 % sign agreement and r = 0.06** with what was measured; over 5-cycle windows the numbers are the same (46 %, r = 0.09, n = 747). The predicted change is small by construction — median 0.004 per cycle (|∇Q| ≈ 0.2 1/m × 0.02 m) — whereas the measured cycle-to-cycle change has a standard deviation of 0.060, fifteen times larger: the score fluctuates (region spheres following the MPC path, voxel grid updates, the non-stationarity of §2.4) far more than the base's 2 cm per cycle can move it. The commanded *direction*, on the other hand, is stable from cycle to cycle (median turn 10°, reversals > 90° in 5.9 % of cycles), so the hunting of §2.2 comes from a few large reversals per leg rather than from jitter.

### 2.6 Real-time performance

![timing](exp_1_moving_figures/fig_timing.png)
*Figure 8 — Controller cycle time, arm MPC solve time, recorder scoring time.*

| | value |
|---|---|
| gradient cycle (9 candidate maps + 3-sphere scoring) | mean **701 ms**, max 898 ms, period 1 000 ms — never overran |
| MPC solve time, all 4 972 logged steps | mean 208 ms, p95 323 ms, **1.1 % over the 400 ms budget** |
| MPC solve time within 1 s of a controller cycle start (inference running) | mean 231 ms vs 197 ms otherwise (+17 %); budget exceeded 1.9 % vs 0.7 % |
| MPC solve time in `act_home_exit` (controller and recorder idle) | mean 189 ms |
| recorder Q_goal + Q_path scoring | median 38 ms, p95 164 ms; 52 vs 36 ms median with / without the controller's inference running |

Three GPU consumers shared the Orin during the approach legs (arm MPC, controller UNet3D ×9, recorder UNet3D). The measured contention is moderate — the MPC's budget violations rose 2.6× but stayed below 2 % — and the 0.70 s cycle leaves 30 % headroom at `cycle_period = 1.0`. The 440 ms per cycle quoted in `OPTIMIZATIONS.md` for the controller alone was not reached here, consistent with the un-throttled second model.

### 2.7 What distinguishes the cancelled cycles

![outcome diag](exp_1_moving_figures/fig_outcome_diag.png)
*Figure 9 — Place approach diagnostics, by outcome.*

Among the scored cycles the place approach is uneventful: the arm reached the pre-place pose in all 23 of them (final FK error median 2 mm, scene-collision constraint active in a median 0 % of MPC steps), and both collisions of the run happened in the pick leg instead. Ep 10 is the clear case — scene-collision constraint active in 10.5 % of its pick-leg steps, final FK error 0.16 m after 133 s before the operator cancelled.

Arm-side metrics are stable across the run (Figure 10): recorder Q_goal per phase 0.77 ± 0.11 (pick) and 0.77 ± 0.10 (place); Yoshikawa manipulability (medians) 0.012 at home-enter, 0.14 during the pick approach, 0.20 during the place approach, 0.21–0.22 at the grasp/place moves.

![arm metrics](exp_1_moving_figures/fig_arm_metrics.png)
*Figure 10 — Recorder scores and manipulability per phase, valid episodes.*

---

## 3. Discussion

**The integration works end to end.** The controller is correctly gated to the approach legs, the interlocks and latches fire as designed, the base follows the commands to within a few percent, and the whole stack runs within its real-time budgets on the Orin with a second model instance alongside. Task-level success is 23/25 = 92 % once the grasp-quality cancels are set aside (23/32 = 72 % with them counted); what is left is dominated by obstacle-map issues that do not involve the base.

**The controller converges to a sensible, repeatable base pose** (Figure 3): for both legs it brings the target toward the arm's forward axis, into the band where the model scores the goal highest, and the goal-sphere score at the moment the base stops is higher than at the start in ~70 % of legs. The improvement is modest (+0.04 / +0.02 median) because the start pose of this protocol is already good (Q_goal ≈ 0.79 / 0.72) — the experiment was not designed with a deliberately poor initial pose, so it measures the controller's behaviour, not its rescue capability.

**Two properties of the score limit what a gradient descent can do here.**
1. *Non-stationarity.* The goal-sphere score changes with the arm's motion, not only with the base's position (hysteresis of 0.12 at equal position, Figure 6 right). Candidate causes, testable offline: the arm entering the depth camera's field of view (occlusion of the table/object region → the voxel grid, hence the map, changes), or the arm's own occupancy leaking into the grid near the goal. A run with `base_control:=false` would show directly whether the dip exists without base motion.
2. *Signal-to-noise per cycle.* At 2 cm per cycle the predicted change in Q is ~0.004 while the measured fluctuation is ~0.06; the controller cannot verify its own gradient from one cycle to the next, which is why the per-cycle prediction is uncorrelated with the measurement (§2.5). The gradient *direction* is nonetheless consistent and the end poses are repeatable, so the least-squares plane fit is doing its job on the model's output; the issue is the model's output moving under the controller.

**The objective weights the corridor twice as much as the goal.** With `path_horizon_s = 2.0, 1.0` the two path spheres sit 1–2 s ahead of the TCP, i.e. close to the arm's current position, and they score ≈ 0.95 at the start of a pick leg; the leg's initial backward motion improves nothing measurable and lowers Q_goal. Whether the backward motion is what the model actually predicts for a translated map (e.g. relieving elbow/table interference for a goal 13 cm above the table) or an artefact of translating self-geometry along with the scene (`use_static_obstacles: false`) cannot be decided from this run, since only the centre candidate's per-sphere scores are logged.

**The base is stopped by the arm.** The arm's approach takes ~40 s; the base moves for ~19 s at 0.02 m/s and is then latched by arrival. At this speed the controller never gets to convergence (5 of 64 legs); the net displacement (13–27 cm) is bounded by the arm, not by the optimum.

---

## 4. Known issues in this dataset

- `base_dist_est`, `x_est`, `y_est` are zero in every file and `base_trajectory.png` / the base panel of `run_summary.png` are empty: the recorder subscribed to `/r100_0597/joy_teleop/cmd_vel`, which was not the topic the relay used. All base-motion results above use the re-integrated controller command, validated in §2.2. Set `base_cmd_vel_topic` to the actual relay topic (or to `/cmd_vel`) for the next run.
- `ik_fk_validity.csv` is empty in every episode (the `fk_batch` service was not available), so the IK-candidate validity analysis is not possible on this run.
- The first controller cycle of a leg sometimes scores a stale map (recorder `rm_age_ms` up to 14 s at leg start) with a goal-only synthetic region, giving Q ≈ 0.14–0.19 where the next cycle reads 0.86 (ep 11, 32, 34). These cycles are excluded from every leg statistic here; they would inflate ΔQ by +0.7 if taken at face value.
- `gradient_taper_ref` (0.02) < `min_gradient` (0.025): the taper is dead code in this configuration; the comment in the yaml assumes 0.25.
- Per-candidate, per-sphere scores are not logged (only the 9 region means feed the gradient, and only the centre's three sphere means are written), so the gradient cannot be decomposed by sphere after the fact.

---

## 5. Recommendations for the next runs

1. **Run the control arm** (`base_control:=false`, same protocol) — it is the only way to attribute the systematic Q_goal dip to the arm's motion vs the base's, and to compare cycle time, grasp success and MPC health with and without base motion.
2. **Log the per-candidate sphere means** (`ws_mean_k` for all 9 candidates) so the gradient can be attributed to the goal vs the corridor spheres; then test `path_horizon_s` weighting (e.g. goal-only, or a heavier goal weight) on a bag with `workspace_probe`.
3. **Test `use_static_obstacles:=true`** on the same bags: if the initial backward motion disappears, the translated self-geometry was the cause.
4. **Give the base time or speed**: either start the base motion before the arm is released (the base reaches its end pose in ~15 s at 0.02 m/s) or raise `base_speed`; and fix the taper (`gradient_taper_ref` ≈ 0.25, `min_gradient` ≈ 0.08 per the yaml's own note) so the base decelerates into the optimum instead of being cut off at full speed.
5. **Start some cycles from a deliberately poor base pose** (goal at > 0.6 m in x, or far to the side) to measure the controller's rescue capability, which this protocol (Q_goal ≈ 0.79 at start) does not exercise.
6. Fix the two recording gaps (§4): `base_cmd_vel_topic`, and the `fk_batch` service.

---

## Appendix A — Files

| Path | Content |
|---|---|
| `analysis/analyze_exp1.py` | regenerates everything below (`python3 analysis/analyze_exp1.py`, ~1 min) |
| `exp_1_moving_figures/fig_timeline.png` … `fig_episode_034.png` | Figures 1–10 and two annotated example episodes (0 and 34) |
| `exp_1_moving_tables/episodes.csv` / `.md` | one row per episode: class, note, last phase, per-leg motion and ΔQ |
| `exp_1_moving_tables/legs.csv` / `.md` | one row per (episode, approach leg): cycles, motion, Q start/stop, stop reason, commanded vs observed displacement |
| `exp_1_moving_tables/cycle_pairs.csv` | the 1 243 consecutive-cycle pairs of §2.5 |
| `exp_1_moving_tables/preplace_mpc_diag.csv` | per-episode place-approach MPC diagnostics of §2.7 |
| `exp_1_moving_tables/summary.json` | every number quoted in this report |
| `episode_*/episode_*.png`, `run_summary.png` | the recorder's own end-of-episode / end-of-run figures (base panel empty, see §4) |

## Appendix B — Per-episode table

Net = net base displacement over the leg (re-integrated command); ΔQ_goal = goal-sphere score at base stop minus at the first full-region cycle.

| ep | class | dur. (s) | last phase | pick net (m) | pick ΔQ_goal | place net (m) | place ΔQ_goal | note |
|---|---|---|---|---|---|---|---|---|
| 0 | success | 159 | home_exit | 0.13 | −0.15 | 0.15 | −0.04 | pick-and-place completed |
| 1 | success | 176 | home_exit | 0.07 | +0.03 | 0.40 | +0.07 | pick-and-place completed |
| 2 | success | 171 | home_exit | 0.07 | +0.04 | 0.24 | +0.00 | pick-and-place completed |
| 3 | excluded | 90 | grasp:idle |  |  |  |  | operator intervention during pregrasp (setup error) |
| 4 | excluded | 0 | home_enter |  |  |  |  | infrastructure: `set_planner` service unavailable (6 ms episode) |
| 5 | excluded | 142 | release:idle |  |  |  |  | bad grasp → perturbed obstacle map → cancelled during place approach |
| 6 | success | 178 | home_exit | 0.13 | +0.06 | 0.16 | −0.09 | pick-and-place completed |
| 7 | success | 201 | home_exit | 0.15 | +0.06 | 0.32 | −0.22 | pick-and-place completed |
| 8 | success | 192 | home_exit | 0.18 | +0.08 | 0.32 | +0.18 | pick-and-place completed |
| 9 | excluded | 132 | release:idle |  |  |  |  | bad grasp → cancelled during place approach |
| 10 | collision | 172 | grasp:move_pregrasp | 0.12 | +0.04 |  |  | arm collision during pregrasp; cancelled after 135 s |
| 11 | success | 162 | home_exit | 0.05 | −0.01 | 0.19 | +0.11 | pick-and-place completed |
| 12 | success | 173 | home_exit | 0.14 | −0.05 | 0.19 | +0.01 | pick-and-place completed |
| 13 | success | 184 | home_exit | 0.15 | −0.03 | 0.29 | −0.01 | pick-and-place completed |
| 14 | success | 186 | home_exit | 0.18 | +0.04 | 0.25 | +0.03 | pick-and-place completed |
| 15 | success | 218 | home_exit | 0.12 | +0.06 | 0.21 | +0.12 | pick-and-place completed |
| 16 | excluded | 158 | release:idle |  |  |  |  | bad grasp → cancelled during place approach |
| 17 | success | 198 | home_exit | 0.13 | +0.05 | 0.34 | +0.02 | pick-and-place completed |
| 18 | success | 181 | home_exit | 0.09 | +0.06 | 0.31 | −0.14 | pick-and-place completed |
| 19 | excluded | 163 | release:idle |  |  |  |  | bad grasp → cancelled during place approach |
| 20 | success | 183 | home_exit | 0.11 | −0.07 | 0.23 | −0.03 | pick-and-place completed |
| 21 | success | 253 | home_exit | 0.13 | +0.05 | 0.22 | +0.09 | pick-and-place completed |
| 22 | excluded | 168 | release:idle |  |  |  |  | bad grasp → cancelled during place approach |
| 23 | success | 163 | home_exit | 0.14 | +0.10 | 0.21 | −0.16 | pick-and-place completed |
| 24 | success | 195 | home_exit | 0.19 | +0.07 | 0.44 | +0.07 | pick-and-place completed; collision only during the final homing |
| 25 | excluded | 100 | release:move_preplace |  |  |  |  | bad grasp → cancelled during place approach |
| 26 | success | 165 | home_exit | 0.25 | +0.00 | 0.26 | +0.09 | pick-and-place completed |
| 27 | success | 201 | home_exit | 0.28 | +0.06 | 0.34 | +0.21 | pick-and-place completed; cancelled during the final homing |
| 28 | excluded | 148 | release:idle |  |  |  |  | bad grasp → cancelled during place approach |
| 29 | excluded | 109 | grasp:idle |  |  |  |  | operator intervention during pregrasp (setup error) |
| 30 | success | 173 | home_exit | 0.11 | +0.01 | 0.18 | +0.01 | pick-and-place completed |
| 31 | success | 170 | home_exit | 0.12 | −0.07 | 0.30 | +0.03 | pick-and-place completed |
| 32 | success | 174 | home_exit | 0.10 | +0.03 | 0.38 | +0.02 | pick-and-place completed |
| 33 | collision | 90 | grasp:idle | 0.22 | +0.08 |  |  | arm collision at grasp (`Grasp failed: Sequence failed`) |
| 34 | success | 172 | home_exit | 0.08 | −0.01 | 0.28 | −0.41 | pick-and-place completed |
