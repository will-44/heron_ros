# Setup 2 (cluttered scene) — fixed base vs reachability-gradient base positioning

**Runs.** `setup_2/20260918_174739_base_fixe` (2026-09-18, 17:47 → 19:07, 18 cycles, base immobilised) and `setup_2/20260921_100545_base_mobile` (2026-09-21, 10:05 → 12:00, 30 cycles, base free). Same stack, same parameters, same scene; the only difference is whether the base's motor relay was enabled. Both runs logged `base_control=true`, i.e. the controller scored Q, computed ∇Q and published `/cmd_vel` every second in both approach legs — in the fixed run those commands simply went nowhere.

**Setup 1 for reference.** The same A/B was run in a lightly cluttered scene: [`../setup_1/REPORT_base_mobile.md`](../setup_1/REPORT_base_mobile.md) (base moving) and [`../setup_1/REPORT_base_fixe.md`](../setup_1/REPORT_base_fixe.md) (base fixed + comparison). Setup 2 adds obstacles around the table and the place station.

**Reproduce.** `python3 analysis/setup_2/analyze_setup2.py` (from `heron_experiment/`) regenerates everything: each run's own figures and tables inside its own run directory (`../../results/setup_2/20260918_174739_base_fixe/{figures,tables}/` and `../../results/setup_2/20260921_100545_base_mobile/{figures,tables}/`), and the side-by-side `cmp_*.png` + `cmp_*` tables in `../../results/setup_2/analysis/comparison/{figures,tables}/`. `--skip-per-run` rebuilds only the comparison. The shared library is `reach_analysis.py` (a snapshot of the setup-1 library that has since diverged; see its docstring).

---

## Abstract

In a cluttered scene, letting the reachability gradient position the base raised the pick-and-place success rate from **10/17 (59 %)** to **18/20 (90 %)** (Fisher exact p = 0.052) and cut the median successful cycle from **250 s to 189 s** (−24 %, p = 0.0004). Every failure in both runs is the same failure: the arm cannot reach the pre-place pose through the clutter and the cycle is cancelled — 7 of 17 cycles at a fixed base, 2 of 20 with the base moving. The whole difference is concentrated in one phase: the **place approach, 105 s fixed against 50 s moving** (p = 0.0007), the only phase whose duration differs. The mechanism is visible in the score the controller optimises: over a fixed-base place approach the goal-sphere reachability *falls* (0.591 → 0.510, improving in 2 of 16 legs), while with the base moving it *rises* (0.542 → 0.632, improving in 15 of 19 legs, p = 0.0004); when the arm actually arrives at the pre-place pose the score is **0.53 fixed vs 0.73 moving**. The base achieved this by closing 0.15 m of range and 13° of bearing on the place target (1.07 m → 0.92 m) using 0.38 m of commanded path in 19 s. The pick approach, by contrast, needs no base motion at all: the goal-sphere score on arrival is **0.858 in both runs** — the controller spends 0.31 m of path and 4 s of extra phase time on the pick leg for no measurable gain. The runs also answer a question left open by experiment 1: the dip of the goal score during the pick approach (0.82 → 0.54 → 0.86 here) is **caused by the base's own excursion**, not by the arm entering the camera's field of view — with the base still and the same arm motion, the score does not dip (0.825 → 0.786 → 0.861, goal displacement 8 mm). Compute is identical in both runs (MPC 208 vs 211 ms mean, controller cycle 697 vs 691 ms), so the two runs are a clean A/B on everything but the motor relay.

---

## 1. Setup and classification

### 1.1 Condition

Heron (omnidirectional base + Doosan M1013, Jetson Orin AGX); `gradient_base_controller` with the 3-sphere region (MPC goal + 2 spheres on the MPC predicted path, r = 0.10 m, equal weights), 3×3 translated obstacle maps at δ = 0.10 m, least-squares gradient, `base_speed = 0.02 m/s`, `cycle_period = 1.0 s`, `use_static_obstacles = false`; cuRobo MPC with a 400 ms budget; `workspace_reachability_node` as recorder. Setup 2 differs from setup 1 only in the scene: more obstacles around the table and the place station.

The two runs are well matched on the pick geometry — the object sits at (0.497, −0.348) m fixed vs (0.494, −0.337) m moving in the arm frame at the start of the leg, i.e. 0.61 m at −35.0° vs 0.60 m at −34.3° — and on the place *range* (1.062 m vs 1.072 m), but **not on the place bearing**: 55.8° (fixed) vs 48.2° (moving). The base's home pose relative to the place station was therefore ≈ 8° different between the two days; see §7 for what that does and does not explain.

### 1.2 Was the base still / did it move?

The base's own odometry is still not recorded (§8), so base motion is reconstructed by integrating the `/cmd_vel` the controller published, and validated against an independent observation: the shift of the fixed-in-world MPC goal expressed in the arm frame over each leg.

| Run | slope (observed vs commanded) | r | n components |
|---|---|---|---|
| base fixed | **0.001** | 0.017 | 66 |
| base moving | **1.014** | 0.980 | 78 |

The fixed run commanded 15.0 m of motion (786 cycles at 0.020 m/s) that produced no displacement at all; the moving run commanded 14.4 m and executed it one-to-one. Figures: [`../../results/setup_2/20260918_174739_base_fixe/figures/fig_odometry_check.png`](../../results/setup_2/20260918_174739_base_fixe/figures/fig_odometry_check.png) and [`../../results/setup_2/20260921_100545_base_mobile/figures/fig_odometry_check.png`](../../results/setup_2/20260921_100545_base_mobile/figures/fig_odometry_check.png).

### 1.3 Outcome classification

`episodes_index.csv` only records how `brain` ended the cycle; the classes below come from the experimenter's notes (2026-09-21).

**`setup_2/20260918_174739_base_fixe` — 18 recorded, 17 valid**

| Class | n | Episodes | Rule |
|---|---|---|---|
| **success** | 10 | 2, 3, 4, 5, 8, 9, 10, 13, 16 + 14 | pick *and* place completed. All were cancelled during the final homing (ep 14: homing reported `Home (end) failed`), as in the setup-1 runs. |
| **collision / arm blocked** | 7 | 0, 6, 7, 11, 12, 17 (place approach), 1 (pick approach) | the arm was blocked by the clutter and never reached the target pose; cycle cancelled. Ep 17 also failed its release sequence. |
| **excluded** | 1 | 15 | `set_planner` service unavailable at cycle start (4 ms episode) |

**`setup_2/20260921_100545_base_mobile` — 30 recorded, 20 valid**

| Class | n | Episodes | Rule |
|---|---|---|---|
| **success** | 18 | 2, 3, 4, 9–17, 24–28 + 19 | pick *and* place completed (ep 19 cancelled during the final homing) |
| **collision / arm blocked** | 2 | 1 (pick approach, arm still 0.62 m from the target), 5 (place approach, 0.39 m short, scene-collision constraint active in 32 % of MPC steps) | |
| **excluded** | 10 | 0 (`set_planner` unavailable), 6, 7, 8 (`Planning failed` at home start, ~14 s), 18 (arm on target — FK error 1 mm — gripper sequence failed), 29 (run stopped mid-cycle), 20, 21, 22, 23 (per-episode recording deleted by the operator: invalid attempts) | not valid attempts |

Neither run has a grasp/release-failure class, so the scored set equals the valid set here; the one setup-2 episode of that kind (moving ep 18, arm on target, gripper sequence failed) is already excluded above, consistent with the rule applied to setup 1. Success over scored attempts: **10/17 = 58.8 %** fixed, **18/20 = 90.0 %** moving (Fisher exact p = 0.052). Note the asymmetry in what was excluded: the moving run carries 10 exclusions, 4 of them (20–23) removed by the operator as invalid attempts and therefore unverifiable from the data. Counting those four as failures instead would still give 18/24 = 75 %.

![outcomes](../../results/setup_2/analysis/comparison/figures/cmp_outcomes.png)
*Figure 1 — Outcome of the valid cycles. In setup 2 there is only one failure mode: the arm blocked by the clutter.*

---

## 2. What each run did

![timeline fixed](../../results/setup_2/20260918_174739_base_fixe/figures/fig_timeline.png)
*Figure 2 — `setup_2/20260918_174739_base_fixe`: cycles by phase; black ticks mark the controller cycles in which a non-zero velocity was published (not executed).*

![timeline moving](../../results/setup_2/20260921_100545_base_mobile/figures/fig_timeline.png)
*Figure 3 — `setup_2/20260921_100545_base_mobile`: same plot, ticks are cycles in which the base actually moved.*

The controller was gated open only during the two approach legs: 846 active cycles (786 commanded) in the fixed run, 773 (740) in the moving one, always at |v| = 0.020 m/s — the taper never engages, so the law is bang-bang (see the middle panel of [`../../results/setup_2/20260921_100545_base_mobile/figures/fig_episode_009.png`](../../results/setup_2/20260921_100545_base_mobile/figures/fig_episode_009.png)).

**Where the legs end.** The pick leg almost always ends on the arrival latch (fixed 9/17, moving 16/20) or on the grasper interlock (7/17 vs 3/20); gradient convergence ends one leg in each run. In the fixed run **all 16 place legs ended on the arrival latch** — the controller never converged and never stopped itself; it kept commanding for a median 78 s and up to 289 s / 1.43 m of virtual path (ep 13). With the base moving, the place leg ends after a median 19 s, 14 times on the latch and 5 times on the interlock.

---

## 3. The place approach is the whole story

![phase durations](../../results/setup_2/analysis/comparison/figures/cmp_phase_durations.png)
*Figure 4 — Phase durations of the successful cycles (one dot per cycle, bar = median).*

| Phase (median, successful cycles) | base fixed | base moving |
|---|---|---|
| home enter | 30.9 s | 37.8 s |
| pick approach | 35.8 s | 39.3 s |
| grasp move | 5.1 s | 5.1 s |
| **place approach** | **105.1 s** | **49.9 s** |
| place move | 9.9 s | 12.2 s |
| home exit | 39.2 s | 23.6 s |
| **whole cycle** | **249.7 s** | **189.4 s** |

Every phase except the place approach is within ±7 s between the two runs, and the pick approach is *longer* with the base moving. The place approach is halved (p = 0.0007, Mann-Whitney) and carries the whole 60 s cycle-time difference (p = 0.0004). Six of the seven fixed-base failures happen in this phase.

---

## 4. The scores during the approach

![q legs](../../results/setup_2/analysis/comparison/figures/cmp_q_legs.png)
*Figure 5 — Median across legs of the controller's own scores, resampled on a 1 s grid from the first cycle with a complete 3-sphere region. The median is solid while at least half the legs are still running and dashed once only a minority are left, and every curve runs to the end of its longest leg. The earlier version cut every curve as soon as fewer than 5 legs remained, which both hid the longest legs and turned the tail of the median into an average over a self-selected minority; the solid/dashed split replaces that quorum without dropping anything. (The per-leg lines, the inter-quartile band and the number-at-risk strip were removed for legibility — the same panels with all three are still produced by the other runs' `fig_q_legs`.)*

![q arrival](../../results/setup_2/analysis/comparison/figures/cmp_q_arrival.png)
*Figure 6 — Goal-sphere score at three instants: start of the leg and end of the command (controller), and when the arm actually arrives at the target pose (recorder, mean over the last 3 s of the phase).*

| Leg | quantity | base fixed | base moving |
|---|---|---|---|
| pick | Q goal, leg start → end of command | 0.825 → 0.813 (**−0.012**, improves in 3/17 legs) | 0.823 → 0.864 (**+0.041**, 12/20) |
| pick | Q goal when the arm arrives | **0.858** | **0.859** |
| pick | Q region mean (the objective) | 0.872 → 0.886 | 0.870 → 0.930 |
| place | Q goal, leg start → end of command | 0.591 → 0.510 (**−0.091**, 2/16) | 0.542 → 0.632 (**+0.124**, 15/19) |
| place | Q goal when the arm arrives | **0.532** | **0.726** |
| place | Q region mean (the objective) | 0.848 → 0.671 | 0.835 → 0.804 |

Two things are worth separating here.

**The place leg is where base motion pays.** The difference in ΔQ_goal over the place leg is large and consistent (Mann-Whitney p = 0.0004); the difference over the pick leg is not (p = 0.27). At a fixed base the goal score *decays* during the place approach even though nothing about the base changed — the arm is extended and carrying the object across the camera's view, so the occupancy grid, and with it the predicted reachability, degrades as the phase drags on. Base motion more than compensates: +0.12 instead of −0.09, and +0.19 on the score measured when the arm arrives.

**The pick leg buys nothing.** Both runs deliver the arm to the object with the *same* goal-sphere score, 0.858 — but the moving run spends 0.31 m of commanded path, 15 s of motion and ~4 s of extra phase time to get there, and ends with the object 0.76 m away instead of 0.60 m. In this setup the home pose is already good enough for the pick.

![geometry](../../results/setup_2/analysis/comparison/figures/cmp_geometry.png)
*Figure 7 — Where the target sits in the arm frame at the start of the leg (circles) and when the arm arrives (squares); the arrow joins the two medians. Fixed-base markers coincide, by construction.*

The place leg closes **0.15 m of range and 13° of bearing** (1.07 m → 0.92 m, 48° → 36°) with 0.38 m of commanded path — path efficiency 0.72. The pick leg *opens* the range (0.60 → 0.76 m) with an efficiency of only 0.52 and a median of one direction reversal per leg, which is the excursion described in §5.

---

## 5. The pick-leg excursion, and an answer to a question from experiment 1

![direction](../../results/setup_2/analysis/comparison/figures/cmp_direction.png)
*Figure 8 — Circular mean over legs of the direction of ∇Q in the arm frame, per time bin. Bins where the legs disagree (concentration R < 0.3) are not drawn.*

During the pick approach the two runs start by asking for the same thing — back away, ≈ 171–175° — with near-perfect agreement across legs (R ≥ 0.99). The fixed run keeps asking for it for the whole leg (157–162°). The moving run, which is actually executing it, **reverses**: 138° at 5–10 s, +37° at 10–15 s, −2° afterwards. The base backs away ~0.18 m, the objective (region mean) climbs 0.870 → 0.930 the whole time, but the goal sphere alone collapses — the median curve bottoms at 0.64 around 7 s and the median per-leg minimum is 0.54 (0.20 in ep 5) — and then recovers to 0.86 as the arm extends, the MPC path shortens and the two path spheres fold back onto the goal.

So the controller's objective and the goal-sphere score genuinely disagree in the first half of the pick leg: with equal weights, two path spheres out of three describe where the arm *is going*, and early in the leg that is still near the home pose. The disagreement is transient and self-correcting here, but it is the reason the pick leg wanders.

**This settles the open question of experiment 1** (§3, point 1 of [`../setup_1/REPORT_base_mobile.md`](../setup_1/REPORT_base_mobile.md)), which proposed two candidate causes for the dip — the base's own motion, or the arm entering the depth camera's field of view and corrupting the voxel grid. With the base fixed, the arm performs exactly the same approach and the dip does not happen:

| Pick approach, recorder Q_goal (medians over legs) | base fixed | base moving |
|---|---|---|
| at the start of the leg | 0.825 | 0.821 |
| minimum during the leg | **0.786** | **0.536** |
| at the end of the leg | 0.861 | 0.860 |
| displacement of the goal in the arm frame | 0.008 m | 0.175 m |

The dip is produced by the base's excursion, not by the arm.

![q landscape](../../results/setup_2/20260921_100545_base_mobile/figures/fig_q_landscape.png)
*Figure 9 — Moving run: measured goal-sphere score against where the goal sits in the arm frame (left, centre), and the hysteresis of the pick leg (right).*

The score is nevertheless **not a function of the planar offset alone**. Comparing, within each pick leg, the score at the same goal-x bin on the way out and on the way back gives a median gap of **+0.177** in favour of the return (IQR 0.153–0.265), positive in **18 of 18** legs with enough overlap. Part of this is geometric — the return does not retrace the outbound y — but the size of the gap means the reachability map is also changing under the controller as the arm moves.

---

## 6. Failure diagnostics, arm metrics, compute

![mpc](../../results/setup_2/analysis/comparison/figures/cmp_mpc.png)
*Figure 10 — Per leg: duration, fraction of MPC steps with an active scene-collision constraint, and forward-kinematics error to the target at the end of the leg (log scale). `n` under each column is the number of legs.*

The failures are unambiguous. In successful cycles the arm ends the place approach a median 3 mm (fixed) / 1 mm (moving) from the target pose; in the failed ones, **0.201 m** (fixed, n = 6) and 0.393 m (the single moving-run place failure). The obstacle map is measurably in the way: the scene-collision constraint is active in 3.5 % of MPC steps in a successful fixed-base place approach against 5.3 % in a failed one, and in 32 % of the steps of the moving run's one place failure (ep 5). The same split is in [`../../results/setup_2/20260918_174739_base_fixe/figures/fig_outcome_diag.png`](../../results/setup_2/20260918_174739_base_fixe/figures/fig_outcome_diag.png).

![manip](../../results/setup_2/analysis/comparison/figures/cmp_manip.png)
*Figure 11 — Yoshikawa manipulability, phase means, valid episodes.*

Manipulability is *higher* with the base moving in the phases that precede the place (pick approach 0.069 vs 0.052, grasp move 0.129 vs 0.095) and *lower* during the place approach and place move (0.152 vs 0.208, 0.224 vs 0.275). The fixed-base place figures are computed over a phase that lasts twice as long and that failed 6 times out of 16, so they describe a different, longer arm trajectory rather than a better-posed one; this metric should not be read as "the fixed base gave a better arm configuration".

**Time-resolved manipulability.** That caveat is now quantified: `manipulability` is logged at ~106 Hz in `joint_states.csv`, so the phase mean can be replaced by the whole trace, aligned on the start of each approach leg ([`../manipulability/`](../manipulability/README.md), Figure 11b).

![manip over time](../manipulability/figures/manip_over_time.png)
*Figure 11b — Manipulability over the approach legs, median and inter-quartile band across legs, both setups (`manipulability/figures/manip_over_time.png`).*

- **The fixed-base advantage in the place approach is a duration artefact.** The fixed base does end the leg in a better-conditioned posture (w 0.273 vs 0.223 on arrival, p < 0.001) — but it takes 90 s to get there against 49 s, and it settles (stops dropping below 80 % of its final w) after 31.8 s against 22.0 s (p = 0.001). It also spends a median **8.7 s below w = 0.05** where the moving base spends none (p = 0.001, Cliff's δ = 0.66), which is 7.7 % of its phase.
  - *Correction to an earlier wording:* that dip was described as "near-singular". It is not. Across the four runs 34 % of all logged samples sit below w = 0.05, and the arm's parked posture itself sits at w = 0.012–0.020. On the place approach the **minimum** w over the leg is 0.032 (fixed) and 0.053 (moving), so the time actually spent below w = 0.01 is **0.0 s in both conditions** (p = 0.68). The difference is real but it is a dip into a band the arm routinely occupies, not a singularity approach.
- **In the cluttered scene the base motion clearly improves the *pick* posture**, which the phase means understated: w on arrival 0.121 vs 0.085 (+42 %, p < 0.001), median over the phase 0.073 vs 0.053 (p = 0.037). This is the one metric on which the pick leg does benefit from the base motion, contrary to the duration and Q evidence of §5.
- **The pick approach is the one that genuinely goes near-singular**, in both conditions: 1.4–2.3 s below w = 0.01 (3–7 % of the leg, against 0.0 s on the place approach), with a minimum of 0.000 in most legs, at 12–18 s. The two curves coincide there (p = 0.43 and 0.12), so it belongs to the arm leaving its home posture, not to the base — and the mechanism is now identified: q₅ sweeps through 0, the M1013 wrist singularity (Figure 13c).
- **Manipulability does not separate the failures** (`manip_by_outcome.png`): the place approaches that ended in a collision have the same, or a slightly higher, w than the successful ones (0.257 vs 0.273 on arrival at the fixed base). The failure is the goal being unreachable through the clutter — the FK error and the scene-collision fraction above — not an ill-conditioned arm.

**Joint velocities.** The same traces give the joint speeds ([`../velocity/`](../velocity/README.md), Figure 11c). The instantaneous speeds are the same in both conditions — mean ‖q̇‖ 0.079 vs 0.081 rad/s on the place approach, 0.080 vs 0.073 on the pick, no significant difference (p ≥ 0.15) — which is expected, since the MPC bounds them identically whatever the base does. What differs is the integral: the **cumulated joint travel** ∫Σ|q̇|dt over the place approach is **15.7 rad at the fixed base against 8.5 rad with the base moving** (p < 0.001; 11.3 vs 8.3 rad in setup 1, p = 0.010). The arm covers nearly twice the joint-space distance for the same task, and the curves show why: after the initial peak the fixed-base place approach keeps creeping at 0.02–0.05 rad/s out to 90–120 s instead of stopping, with a higher peak (max ‖q̇‖ 0.527 vs 0.442 rad/s, p = 0.001) and more absolute near-still time (20.9 vs 14.2 s, p = 0.003). The extra time is spent moving, not waiting.

![velocity over time](../velocity/figures/vel_over_time.png)
*Figure 11c — ‖q̇‖ over the approach legs, median and inter-quartile band across legs, both setups (`velocity/figures/vel_over_time.png`).*

The commanded base speed carries no information: in all four runs `/cmd_vel` takes exactly one non-zero value, 0.020 m/s, in 18–20 % of the published messages — the law is bang-bang and neither the adaptive cap nor the taper nor the deadband ever engaged.

**Joint accelerations — and a warning about the logged column.** `acc_joint_*` is a second finite difference (`Δ(Δpos/Δt)/Δt`) over `/joint_states` messages whose Δt is strongly irregular (~23 % of samples arrive < 2 ms after the previous one). It does not survive: its lag-1 autocorrelation is **−0.57 to −0.60** in all four runs, the signature of sign-alternating noise, against **+0.99** for the same signal re-derived on a uniform 100 Hz grid with a Savitzky-Golay filter. **Every acceleration statistic taken from that column — max above all, but also mean and std — measures the logger's timestamp jitter, not the arm** ([`../acceleration/`](../acceleration/README.md) §0, Figure 13a). Re-derived:

![accel over time](../acceleration/figures/accel_over_time.png)
*Figure 13b — ‖q̈‖ over the approach legs, re-derived (Savitzky-Golay, 0.41 s window), median and inter-quartile band (`acceleration/figures/accel_over_time.png`).*

- The **amplitude** does not separate the conditions (mean ‖q̈‖ 0.12–0.17 rad/s², p ≥ 0.12 everywhere, stable across 0.21/0.41/0.61 s windows) — same story as the velocities.
- What separates them is again the **persistence**: on the cluttered place approach the moving base falls below 0.05 rad/s² by 40 s while the fixed base stays at 0.10–0.30 rad/s² past 60 s. Those are repeated re-accelerations, not a slow drift. Matched on leg duration (9 vs 12 legs, 63 vs 54 s, p = 0.41), the fixed base has a **higher mean (0.182 vs 0.153, p = 0.002) and a higher peak (2.12 vs 1.62 rad/s², p = 0.036)**, while the moving base makes **more but smaller corrections** (4.13 vs 3.83 acceleration reversals/s/axis, p = 0.017). Small n — a trend, not a settled result.
- A **SPARC smoothness difference should not be quoted**: it is large (−5.65 vs −3.15, p = 0.001) but correlates with leg duration at ρ = −0.84 and vanishes once durations are matched (p = 0.37). It restates the duration.

**The pick approach crosses the wrist singularity, every time.** J5 is by far the most accelerated axis on that leg (0.087 rad/s² median vs 0.030–0.063 for the others), and the reason is that q₅ sweeps −73° → 0 → +38°, with a per-leg minimum |q₅| of **0.0° in median, in all four runs**, between 12 and 18 s — exactly the near-singular passage (w = 0.02–0.03) reported above from the manipulability traces. The two conditions' curves coincide, so this belongs to the home → pregrasp trajectory, not to the base. On the place approach |q₅| never drops below ~20°, and there is no near-singular passage.

![wrist singularity](../acceleration/figures/wrist_singularity.png)
*Figure 13c — |q₅| over the pick approach; the shaded band is |q₅| < 10° (`acceleration/figures/wrist_singularity.png`).*

This is the most actionable finding of the three post-hoc analyses: the q₅ = 0 crossing buys nothing, costs manipulability and wrist acceleration, and is fixed by changing the home posture or constraining q₅ on the way out — independently of the base controller.

![timing](../../results/setup_2/analysis/comparison/figures/cmp_timing.png)
*Figure 12 — Controller cycle time, MPC solve time, and the cycle-to-cycle change of the objective.*

| | base fixed | base moving |
|---|---|---|
| controller cycle, mean (max) | 697 ms (868) | 691 ms (875) |
| MPC solve, mean / p95 | 208 / 321 ms | 211 / 326 ms |
| MPC steps over the 400 ms budget | 0.9 % | 1.2 % |
| ΔQ between consecutive commanded cycles, σ | 0.041 | 0.048 |
| direction turn per cycle, median | 4.2° | 10.6° |
| direction reversals (> 90°) | 0 % | 3.4 % |

Compute is identical, so nothing in §3–§5 can be attributed to a difference in load. The gradient's one-step prediction is as weak here as in setup 1: over 699 commanded-cycle pairs the sign of the predicted ΔQ matches the measured one 44 % of the time (r = 0.02) — at 0.02 m/s the per-cycle displacement (2 cm) moves Q by less than the measurement noise (σ ≈ 0.048), which is why the direction, not the step, is what matters.

---

## 7. Comparison with setup 1, and limitations

| | setup 1 fixed | setup 1 moving | setup 2 fixed | setup 2 moving |
|---|---|---|---|---|
| data-valid cycles | 28 | 32 | 17 | 20 |
| scored cycles (grasp/release failures removed) | 23 | 25 | 17 | 20 |
| success | 78 % | 92 % | **59 %** | **90 %** |
| dominant failure | grasp/release (5), collision (5) | bad grasp (7), collision (2) | collision (7) | collision (2) |
| cycle duration (success) | 214 s | 178 s | 250 s | 189 s |
| place approach | 71 s | 42 s | 105 s | 50 s |
| place ΔQ_goal over the leg | −0.075 | +0.017 | −0.091 | +0.124 |

Adding obstacles does exactly what one would hope for an experiment about base positioning: it **penalises the fixed base** (78 % → 59 %, place approach 71 s → 105 s) while the moving base holds up (92 % → 90 %, place approach 42 s → 50 s). The failure taxonomy moves with it — in setup 1 most cancelled cycles were grasp-quality problems (excluded from every number here), leaving few scored failures, whereas in setup 2 every scored failure is the arm being unable to reach through the clutter, which is precisely the failure that repositioning the base can remove.

**What this comparison does not control.**

1. **Different days, different sessions.** The two runs are 3 days apart (2026-09-18 evening, 2026-09-21 morning). Lighting, camera calibration and the exact obstacle placement were not re-verified between them.
2. **The place bearing differs by ≈ 8°** (55.8° fixed vs 48.2° moving) at the same range. The moving run therefore started its place leg from a slightly different pose. The effect is bounded, though: the fixed run's *own* scores decay over the leg from a starting point (Q_goal 0.591) that is *higher* than the moving run's (0.542), and it is the moving run that ends higher.
3. **Asymmetric exclusions** (1 vs 10), four of them unverifiable (deleted). The worst case for the moving run — counting all four as failures — is 75 % against 59 %.
4. **Sample size.** 17 and 20 valid cycles; the outcome difference is at the edge of significance (Fisher exact p = 0.052). The timing and score differences are well clear of it (p ≤ 0.0007).
5. One operator, one object, one place station: the numbers describe this task in this scene, not a general success rate.

---

## 8. Known gaps in these datasets

- **No base odometry topic** is recorded in either run; base motion is re-integrated from the controller's own `/cmd_vel` echo and validated only indirectly (§1.2). Logging the base's Twist would make the check direct.
- `ik_fk_validity.csv` exists in every episode directory but contains **headers only** (0 rows) — same as in setup 1.
- `control_debug.csv` has **no episode column**; controller cycles are joined to episodes by timestamp.
- Moving run: episode directories 006–008 exist but contain no samples (the cycle died at home start); **020–023 and 029 have no directory at all** (deleted / stopped mid-cycle). Index and phase rows survive for them, which is why they still appear, greyed out, in Figure 3.
- The moving run has no `run_summary.png` (the recorder's own overview), the fixed run does.
- The first controller cycle of a leg can carry a stale region (goal only, `region_size < 3`); all leg statistics here start at the first cycle with a complete 3-sphere region.

---

## 9. Recommendations

1. **Run the place leg, skip the pick leg.** The pick approach delivers the same arrival score (0.858) with and without base motion, while costing 0.31 m of path, one direction reversal per leg and ~4 s. Gating the controller to the place approach in this setup would keep all of the measured benefit.
2. **Weight the goal sphere above the path spheres**, or weight the path spheres by how far along the path they sit. The equal-weight region is what drives the base away from a goal that is already reachable in the first half of the pick leg (§5); a goal-weighted region would remove the excursion without changing the place-leg behaviour, where all three spheres agree.
3. **Give the place leg a convergence criterion it can actually reach.** All 16 fixed-base place legs ran to the arrival latch, one of them for 289 s; a leg that has commanded > N cycles without improving Q is a leg that should stop and report, not keep pushing.
4. **Log the base Twist** and fix `ik_fk_validity` logging before the next campaign.
5. **Repeat the A/B in one session**, alternating the relay state cycle by cycle, to remove the day-to-day confound of §7. With ~20 valid cycles per arm, an interleaved run would turn the p = 0.052 into something conclusive.

---

## Appendix A — Files

| Path | Content |
|---|---|
| `analyze_setup2.py` | regenerates everything below (`--skip-per-run` for the comparison only) |
| `reach_analysis.py` | shared library (snapshot of the experiment-1 analysis library) |
| `../../results/setup_2/analysis/comparison/figures/cmp_*.png` | the 9 comparison figures of this report |
| `../../results/setup_2/analysis/comparison/tables/cmp_summary.csv` \| `.md` | the full side-by-side metric table (≈ 60 rows) |
| `../../results/setup_2/analysis/comparison/tables/cmp_*.csv` | the data behind each comparison figure; `cmp_legs.csv` = one row per approach leg, both runs |
| `../../results/setup_2/20260921_100545_base_mobile/figures/fig_*.png`, `../../results/setup_2/20260921_100545_base_mobile/tables/*` (here) | the moving run on its own, including `fig_base_paths.png` and `fig_q_landscape.png` |
| `../../results/setup_2/20260918_174739_base_fixe/figures/`, `../../results/setup_2/20260918_174739_base_fixe/tables/` | the fixed run on its own |
| `../../results/setup_2/20260921_100545_base_mobile/tables/episodes.csv` \| `.md` | per-episode table with the class, the operator's note and the per-leg summary |
| `../../results/setup_2/20260921_100545_base_mobile/tables/summary.json` | every headline number of this report, per run |

## Appendix B — Per-episode tables

`setup_2/20260918_174739_base_fixe`: [`../../results/setup_2/20260918_174739_base_fixe/tables/episodes.md`](../../results/setup_2/20260918_174739_base_fixe/tables/episodes.md) — `setup_2/20260921_100545_base_mobile`: [`../../results/setup_2/20260921_100545_base_mobile/tables/episodes.md`](../../results/setup_2/20260921_100545_base_mobile/tables/episodes.md).

Columns: cycle duration, number of phases, `brain`'s own outcome, the class used here, the last phase reached, and for each approach leg the commanded time, commanded path, net displacement, ΔQ and the reason the leg ended.
