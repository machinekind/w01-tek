# Locomotion reward redesign (2026-07-12)

Status: design. Nothing here is implemented yet.

The current joystick env carries 17 reward terms plus a walk/trot gait blend and
a commandable body height. Nine tuning iterations (v1–v9) moved along a
speed/precision frontier without leaving it, and the real robot showed a failure
the reward set never addresses: the hips splay outward and the motors are too
weak to pull the legs back under the body. This design restarts from the
smallest reward set the literature supports and attacks the splay directly.

## Goals

One policy that trots at commanded variable speed, stands, and turns. The
actuation signal must be smooth. Torque use must be low. The legs must stay
under the body.

## What gets removed

**The height command.** It brings the HEIGHT_TABLE anchor, the
`height_tracking` reward, height-posed resets, and a fourth command dimension.
None of it serves the goals above. No height term replaces it. The policy picks
its own operating height, because the owner wants the legs used as springs, and
any reward that pins body height or punishes vertical velocity suppresses
exactly that bounce. Standing height still comes out fixed, because `stand_still`
anchors the joints to the home pose (~0.125 m) whenever the command is zero.

**The gait clock, entirely.** The 4-beat walk offsets, the duty blend, the
`trot_band`, the frequency schedule, and the trot clock itself all go, and the
8 phase observations go with them. The clock hard-codes cadence and duty. The
argument that unpins height applies to gait timing too: the optimizer should
find the cadence that minimizes torque. Clock-free gait shaping is the
canonical legged_gym approach and runs on real hardware (ANYmal, Go1, Barkour).
It also already works in this repo. The 2026-07-09/10 free-gait campaign
trained ~20 clock-free variants, and its keeper, spin_posture
([PR #19](https://github.com/machinekind/wojtek/pull/19)), steps
cleanly, stands best of that fleet, and is the only variant that plants all
feet after a walk→stop. The older counterexample, fbb_v2's skating, ran on
the underdamped kd=0.5 actuator model, and the kd=1.0 fix removed the
resonance it exploited. Deployment also gets simpler, because the robot no
longer needs a phase generator.

The known risk: with no contact schedule, the gait can degenerate into skating,
pronking, or pacing. spin_posture already crossed this gap, and its recipe
(air time + high_step + slip) is adopted below. The battery's duty-factor and
diagonal-correlation checks still guard it. The fallback is reintroducing the
trot clock with the current env's `contact_match` and `feet_phase` terms.

The command becomes (vx, vy, wz), at spin_posture's trained ranges since the
inherited weights were calibrated under them: vx ∈ [−0.8, 1.2],
vy ∈ [−0.5, 0.5], wz ∈ [−1.0, 1.0], zeroed with probability 0.25 for stand
training. Adopt spin_posture's exposure fixes on top: with probability 0.25
the command keeps only wz (pure spin) and with probability 0.2 only vy (pure
strafe). Uniform box sampling almost never draws those, so turning and
side-stepping stay undertrained without the fix, and both are goals here.

## Core reward: 14 terms

Weights are in the current env frame (total reward is multiplied by dt).
"moving" means commanded speed > 0.05 m/s. "standing" is its complement.

| Term | Formula | Weight | Gate |
|---|---|---|---|
| tracking_lin_vel | exp(−‖cmd_xy − v_xy‖² / 0.25) | +4.0 | — |
| tracking_ang_vel | exp(−(cmd_wz − ω_z)² / 0.25) | +1.6 | — |
| orientation | ‖g_xy‖² | −5.0 | — |
| pose | Σ wⱼ (qⱼ − q_home,ⱼ)², w = [1.0, 0.1, 0.1] per leg | −1.0 | — |
| feet_air_time | Σ (min(t_air, 0.35) − 0.1) · first_contact | +4.0 | moving |
| high_step | mean over feet of clip(clearance / 0.08, 0, 1) · swing | +4.0 | moving |
| feet_slip | Σ v²_xy,foot · contact | −0.35 | moving |
| torques | Σ τ² | −6e-4 | — |
| torque_rate | Σ (τₜ − τₜ₋₁)² | −0.02 | — |
| torque_limit | Σ max(\|τ\| − 0.85 · τ_max, 0) | −0.1 | — |
| action_rate | Σ (aₜ − aₜ₋₁)² | −0.25 | — |
| stand_still | Σ \|q − q_home\| + 0.2 · Σ \|q̇\| | −2.5 | standing |
| stand_feet_down | Σ clip(foot clearance, 0, ∞) | −30.0 | standing |
| termination | 1[fall] | −1.0 | — |

Terms dropped from the current set: `height_tracking` (command removed),
`contact_match` and `feet_phase` (they need the clock, and they return only as
the degeneration fallback), and `lin_vel_z`, `ang_vel_xy`, `energy`,
`action_accel` (moved to tier 2). The drops are where this design deliberately
diverges from spin_posture, which keeps `lin_vel_z` at −5.0, `ang_vel_xy` at
−0.5, and EMA height tracking at +2.5. Those trunk-rigidity terms are the
likely source of its planted stops and quiet stance, and they also fight the
springy vertical motion this design wants. Dropping them is the experiment,
and they are the first tier-2 pulls if stance quality disappoints.

Why springiness is the bet (owner decision, 2026-07-12). A rigid trunk must
kill the body's vertical momentum in a short leg travel at every touchdown,
so force rises sharply. A compliant stance spreads the same momentum change
over a longer stroke, so peaks drop. Peaks are the scarce resource under the
6 Nm cap, and touchdown saturation is how weak-motor robots fall. Rigidity
also has a speed ceiling under a torque cap, because faster gaits bring more
momentum per step. `torque_rate` guards the failure mode springiness opens:
torque waves at the 2–3 Hz stride rhythm change little between 20 ms control
steps and pay almost nothing, while resonance buzzing (the old fbb_v2 7 Hz
exploit) pays heavily. One caveat stays on record: the motors store no energy
the way tendons do, so springiness buys lower peaks and smoother profiles,
not necessarily less battery. The verdict is cheap to get: compare torque
p90/p99 and the vibration index at matched speed between this core and the
tier-2 trunk fines.

Twelve of the fourteen weights match spin_posture's trained config (including
the env defaults it kept active), so they start calibrated instead of guessed.
The two others are this design's anti-splay additions: `pose`, whose weights
copy the Playground Go1 pattern of anchoring the hip an order of magnitude
harder than the joints that swing the leg, and `torque_limit`. The literature
agrees where it overlaps: the tracking form and σ² = 0.25 appear verbatim in
legged_gym, MuJoCo Playground, Walk These Ways, and CaT. Two calibration notes
from the campaign: `torque_rate` −0.02 has a measured freeze cliff at −0.06,
and the air-time cap (0.35 s) bounds the swing reward rather than targeting a
cadence. The torques coefficient −6e-4 triples the −2e-4 literature consensus,
affordable because spin_posture trained under a 6 Nm cap.

## Anti-splay package

Sim PD reaches any commanded hip angle, so the policy learns stances that need
torque the real motor does not have. Rewards alone cannot close that gap. The
package is mostly structural:

1. **Clamp the abduction ctrlrange to ±0.44 rad (25°).** Today it is ±π.
   The mechanical range exceeds 45° (owner, 2026-07-12), so the software clamp
   is the binding limit.
2. **Halve the abduction action scale**: 0.25 versus 0.5 on the leg joints.
   Walk These Ways ships this knob as `hip_scale_reduction`, and PR #19's
   vector `action_scale` implements it here: `[0.25, 0.5, 0.5]` per leg.
3. **Train under a ±6 Nm cap** using PR #19's `max_torque` clamp. That
   matches the caps the deployed runner already uses, and the `torque_limit`
   hinge (core table) fires at 85% of it, so the policy never learns postures
   that live near saturation. The getup env already uses the hinge pattern.
4. **Weak-motor domain randomization.**
   [PR #15](https://github.com/machinekind/wojtek/pull/15) rebuilds the
   DR module into per-field toggles, but forcerange still rides the same
   random sample as the kp gain scale, so a weak-motor world also feels soft,
   which reads as compliance instead of saturation. Add a separate forcerange
   field to that taxonomy, scaled 0.5–1.1. The weak-side skew is our inference
   from the failure mode. Published ranges are symmetric ±10–20%.
5. **Joint-zero offset randomization.** The motors have relative encoders, so
   every boot can be slightly mis-zeroed. PR #15 already implements this as
   `encoder.enable` (a per-joint constant offset, added to the observed angle
   and subtracted from the written target, off by default, ±0.02 rad knob).
   Enable it. Walk These Ways ships ±0.05 rad, so widening is on the table.
6. **Clip knee ctrl at 3.15 rad**, under the 3.2 four-bar singularity, as the
   jump env already does. The joystick env currently has no guard.

If splay persists in sim after all this, PR #19 carries two dormant knobs
aimed near the problem: `abduction_swing` (penalize abduction deviation while
a leg swings) and `swing_lateral`. spin_posture trained with both off. The
final escalation is a Raibert-style feet-under-hips reward. Walk These Ways
weights it −10.0, their largest single weight. It stays out of the core
because it needs foot-frame bookkeeping the simpler measures may make
unnecessary.

The sim torque cap itself (9 Nm) is a placeholder. The motor model is unknown.
A rough stall measurement of one motor would let us replace the guess and
tighten the randomization range.

## Smoothness and curriculum

Two groups independently report the same trap: full-strength smoothness and
torque penalties from step 0 collapse training into standing still
([quiet walking, aibo](https://arxiv.org/abs/2502.10983);
[15-minute humanoid](https://arxiv.org/abs/2512.01996)). Our v7 "got cautious"
result is the mild version. So training runs in two phases: phase A with
`action_rate`, `torques`, and `torque_limit` at 0.3× weight, phase B at full
weight via checkpoint restore. Runs cost ~15 minutes, so two phases are cheap.

PR #15 also adds randomized control latency (0–5 substeps, sampled per env,
off by default). Enable it in place of the fixed one-step action delay. A
policy trained against variable latency cannot rely on exact actuation timing,
and that robustness is a standard sim-to-real tool. PR #19 carries a second
latency mechanism, per-step loop-timing jitter. Owner decision (2026-07-12):
#15's randomization is the project's approach, so the jitter was not ported.
Revisit only if hardware evidence asks for it after #19 merges.

Tier-2 terms, each added only when the battery shows its symptom:

| Term | Weight | Add when |
|---|---|---|
| action_accel | −0.1 | vibration index stays high after phase B |
| lin_vel_z | −1.0 → −5.0 | bounce turns destructive or stance quality lags spin_posture's; −5.0 is its trained value, and this knowingly trades away springiness |
| ang_vel_xy | −0.05 → −0.5 | base rocks; −0.5 is spin_posture's value |
| energy (Σ \|q̇\| · \|τ\|) | −2e-3 | torque percentiles fine but power draw high |
| EMA height tracking to a fixed 0.125 m (PR #19 `height_avg_tau` 0.5) | +2.5 | the free height drifts somewhere bad; pins the average, leaves the bounce |

A Lipschitz gradient penalty on the policy (λ ≈ 0.002,
[arXiv:2410.11825](https://arxiv.org/abs/2410.11825)) is the strongest
hardware-validated smoothing tool we found. It requires patching the Brax PPO
loss, so it is a later experiment. The v5 lesson stands: no EMA action filter.

## Fix the yardstick first

The battery cannot yet see the new goals:

- No turning or side-stepping scenario exists. Add a turn scenario (vx = 0.4
  with a ±0.8 wz sweep, plus a pure-spin hold at wz = 0.8) scored on
  angular-velocity error and falls, and a strafe scenario (pure vy at ±0.4)
  scored on lateral-velocity error and falls.
- No splay metric exists. Add p95 |q_abduction| and the fraction of steps with
  |τ| above 85% of the cap, per joint group.
- Diagonal correlation alone cannot catch skating, and it cannot tell a pace
  or a pronk from noise. Add per-foot duty factor (the fbb_v2 failure read as
  duty ~1.0) and lateral-pair contact correlation next to the diagonal one
  (trot: diagonal high; pace: lateral high; pronk: both high).
- No walk→stop scenario exists. Add one (command 0.5 m/s, then zero, hold
  300 steps), scored on time until all four feet plant and max foot clearance
  during the hold. This is spin_posture's headline win (−0.6 mm foot-up vs
  60–108 mm fleet-wide), and it is the direct test of whether dropping the
  trunk fines cost us stance quality.
- Nothing verifies that springiness actually emerges once it is allowed. Add
  base-height peak-to-peak amplitude and its spectral power in the 2–4 Hz
  stride band during steady trot. "Bounce allowed but unused" and "bounce
  used" should be distinguishable in the report, or the experiment tests
  nothing.
- Torque comparisons must be speed-fair. A policy that tracks velocity worse
  is effectively slower and gets cheap torque numbers for free (spin_posture
  vel err 0.142 vs v8 0.080 makes this a live hazard). Score torque p90/p99
  binned by achieved speed, not commanded speed, and run spin_posture through
  this same extended battery as the baseline row.
- The vibration index, hold-window qvel, and torque percentiles already cover
  the jitter goals and stay as they are. Run the full battery after curriculum
  phase A as well, so a stand-still collapse or a degenerate gait is caught
  before phase B spends the second run.

Videos, one MP4 per scenario under `videos/<run_name>/`, per trained variant:
`stand_to_trot_ramp` (the fast section shows whether stride-rhythm bounce
emerges), `turn` (the pure-spin hold shows whether feet reposition fast
enough to rotate in place), `strafe`, and `walk_to_stop` (the direct visual
check on planted feet, the thing the trunk-fine drop puts at risk). The v8-era
set shrinks accordingly: the three per-height ramps collapse into the one
ramp, and `stand_heights` and `walk_to_trot` go away with the height command
and the walk gait.

## Implementation order

1. Battery additions (turn scenario, splay and saturation metrics).
2. Done on this branch (2026-07-12). PR #15 merged into main, the branch is
   rebased onto it, and the necessary subset of PR #19 is ported here as a
   credited commit: torque_rate, high_step, stand_feet_down, the air-time
   cap, the max_torque clamp, vector action_scale, and the pure-spin/strafe
   command sampling, all default-off, 69/69 tests green. The loop-jitter,
   obs-delay, displacement, and height-EMA machinery stayed in #19, because
   #15's randomization is the project's approach. Remaining code: the
   abduction ctrlrange clamp, the knee ctrl clip, the `torque_limit` hinge,
   and the decoupled forcerange DR field, plus the experiment preset. The
   height command stays in the code and gets pinned by config
   (`height: [0.125, 0.125]`, `height_tracking: 0`), which is cheaper than
   deleting the plumbing.
3. Two-phase training run, scored against the extended battery.

## Sources

- spin_posture, keeper of the 2026-07-09/10 free-gait campaign; in-repo
  trained evidence for clock-free locomotion and the source of twelve core
  weights: <https://github.com/machinekind/wojtek/pull/19>
- Rudin et al. 2022, legged_gym base config:
  <https://github.com/leggedrobotics/legged_gym>
- MuJoCo Playground Go1 joystick (pose weights, torque/energy forms):
  <https://github.com/google-deepmind/mujoco_playground>
- Margolis & Agrawal, Walk These Ways (hip scale reduction, Raibert term):
  <https://github.com/Improbable-AI/walk-these-ways>
- Barkour: <https://arxiv.org/abs/2305.14654>
- CaT, Constraints as Terminations (why weight balancing is the real cost of
  large reward sets): <https://arxiv.org/abs/2403.18765>
- Learning Quiet Walking for a Small Home Robot (weak-actuator analog,
  penalty curriculum): <https://arxiv.org/abs/2502.10983>
- Sim-to-Real Humanoid Locomotion in 15 Minutes (minimal-reward philosophy):
  <https://arxiv.org/abs/2512.01996>
- Lipschitz-Constrained Policies: <https://arxiv.org/abs/2410.11825>
