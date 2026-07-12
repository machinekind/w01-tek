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
The one clock-free failure in this project's history is fbb_v2, which skated
instead of stepping. That run used the underdamped kd=0.5 actuator model, and
the kd=1.0 fix removed the resonance it exploited. Deployment also gets
simpler, because the robot no longer needs a phase generator.

The known risk: with no contact schedule, the gait can degenerate into skating,
pronking, or pacing. The battery's duty-factor and diagonal-correlation checks
detect all three. The fallback is reintroducing the trot clock with the current
env's `contact_match` and `feet_phase` terms.

The command becomes (vx, vy, wz): vx ∈ [−0.6, 1.0], vy ∈ [−0.3, 0.3],
wz ∈ [−0.7, 0.7], zeroed with probability 0.25 for stand training.

## Core reward: 11 terms

Weights are in the current env frame (total reward is multiplied by dt).
"moving" means commanded speed > 0.05 m/s. "standing" is its complement.

| Term | Formula | Weight | Gate |
|---|---|---|---|
| tracking_lin_vel | exp(−‖cmd_xy − v_xy‖² / 0.25) | +2.0 | — |
| tracking_ang_vel | exp(−(cmd_wz − ω_z)² / 0.25) | +1.0 | — |
| orientation | ‖g_xy‖² | −5.0 | — |
| pose | Σ wⱼ (qⱼ − q_home,ⱼ)², w = [1.0, 0.1, 0.1] per leg | −1.0 | — |
| feet_air_time | Σ (t_air − 0.2) · first_contact | +2.0 | moving |
| feet_slip | Σ v²_xy,foot · contact | −0.3 | moving |
| torques | Σ τ² | −2e-4 | — |
| torque_limit | Σ max(\|τ\| − 0.85 · τ_max, 0) | −0.1 | — |
| action_rate | Σ (aₜ − aₜ₋₁)² | −0.25 | — |
| stand_still | Σ \|q − q_home\| + 0.2 · Σ \|q̇\| | −1.0 | standing |
| termination | 1[fall] | −1.0 | — |

Terms dropped from the current set: `height_tracking` (command removed),
`contact_match` and `feet_phase` (they need the clock, and they return only as
the degeneration fallback), and `lin_vel_z`, `ang_vel_xy`, `energy`,
`action_accel` (moved to tier 2).

The tracking form and σ² = 0.25 appear verbatim in legged_gym, MuJoCo
Playground, Walk These Ways, and CaT. The torque coefficient −2e-4 is where
Playground, Barkour, and Unitree's shipped Go2 config independently landed. The
academic default −1e-5 proved ~20× too weak on hardware. The pose weights copy
the Playground Go1 pattern of anchoring the hip an order of magnitude harder
than the joints that swing the leg. The air-time threshold sets the cadence
floor and is the main gait knob. legged_gym uses 0.5 s on ANYmal, Playground
uses 0.1 s on Go1, and the closest small-robot precedent (quiet-walking aibo)
uses 0.2 s. We start at 0.2 s.

## Anti-splay package

Sim PD reaches any commanded hip angle, so the policy learns stances that need
torque the real motor does not have. Rewards alone cannot close that gap. The
package is mostly structural:

1. **Clamp the abduction ctrlrange to ±0.44 rad (25°).** Today it is ±π.
   The mechanical range exceeds 45° (owner, 2026-07-12), so the software clamp
   is the binding limit.
2. **Halve the abduction action scale**: 0.25 versus 0.5 on the leg joints.
   Walk These Ways ships this exact knob (`hip_scale_reduction`).
3. **The `torque_limit` hinge** (core table) fires at 85% of the sim torque
   cap, so the policy never learns postures that live near saturation. The
   getup env already uses this pattern.
4. **Weak-motor domain randomization.** Today one sample scales kp, kd, and
   forcerange together, so a weak-motor world also feels soft, which reads as
   compliance instead of saturation. Sample forcerange separately, scaled
   0.5–1.1. The weak-side skew is our inference from the failure mode. Published
   ranges are symmetric ±10–20%.
5. **Joint-zero offset randomization, ±0.05 rad.** The motors have relative
   encoders, so every boot can be slightly mis-zeroed.
6. **Clip knee ctrl at 3.15 rad**, under the 3.2 four-bar singularity, as the
   jump env already does. The joystick env currently has no guard.

If splay persists in sim after all this, the escalation is a Raibert-style
feet-under-hips reward. Walk These Ways weights it −10.0, their largest single
weight. It stays out of the core because it needs foot-frame bookkeeping the
simpler measures may make unnecessary.

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

Tier-2 terms, each added only when the battery shows its symptom:

| Term | Weight | Add when |
|---|---|---|
| action_accel | −0.1 | vibration index stays high after phase B |
| lin_vel_z | −1.0 | trot bounces destructively (knowingly trades away springiness) |
| ang_vel_xy | −0.05 | base rocks |
| energy (Σ \|q̇\| · \|τ\|) | −2e-3 | torque percentiles fine but power draw high |
| feet_clearance (Playground form) | −2.0 | feet scuff during swing |

A Lipschitz gradient penalty on the policy (λ ≈ 0.002,
[arXiv:2410.11825](https://arxiv.org/abs/2410.11825)) is the strongest
hardware-validated smoothing tool we found. It requires patching the Brax PPO
loss, so it is a later experiment. The v5 lesson stands: no EMA action filter.

## Fix the yardstick first

The battery cannot yet see the new goals:

- No turning scenario exists. Add one (e.g. vx = 0.4 with a ±0.7 wz sweep),
  scored on angular-velocity error and falls.
- No splay metric exists. Add p95 |q_abduction| and the fraction of steps with
  |τ| above 85% of the cap, per joint group.
- Diagonal correlation alone cannot catch skating. Add per-foot duty factor
  (the fbb_v2 failure read as duty ~1.0).
- The vibration index, hold-window qvel, and torque percentiles already cover
  the jitter goals and stay as they are.

## Implementation order

1. Battery additions (turn scenario, splay and saturation metrics).
2. Model changes (ctrlrange clamp, action scales, knee clip, randomization).
3. Reward rewrite (clock-free, 11-term core).
4. Two-phase training run, scored against the extended battery.

## Sources

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
