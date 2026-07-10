---
name: mjx-robot-model-prep
description: Prepare MuJoCo MJCF or URDF-derived robot models for stable, efficient MJX RL simulation. Use for auditing physical parameters and deployable sensors, generating an MJX variant with MjSpec, primitive contact design, closed-loop constraints and joint conventions, standing keyframes, and pre-training validation.---

# Prepare robot models for MJX

The deployed robot interface is the contract. Generate the training model from the source model,
validate it mechanically, and hand it to `brax-locomotion-training` only after the static gates
pass. Worked example: `training/wojtek_rl/build_model.py`, sources under
`ros/src/wojtek_description/mujoco/`. Read `references/wojtek-model-lessons.md` only when
adapting Wojtek or when a concrete closed-loop example helps.

## Workflow

| Stage | Required result |
| --- | --- |
| Audit | Mass, inertia, torque, sensors, joint conventions match hardware evidence |
| Generate | A repeatable builder produces the MJX variant; source model untouched |
| Contacts | Every task-relevant contact represented without an unbounded pair graph |
| Home pose | Robot settles under deployment-like control with closed loops intact |
| Validate | Compiled-model tests, standing test, and throughput baseline pass |

## Audit the physical contract

- Inspect total and per-body mass. Missing inertials make MuJoCo infer mass from mesh volume;
  round values with textbook inertias are usually placeholders.
- Derive deployable actuator force from torque constant, current limit, gearing, and driver
  limits; check the motors can statically support the audited mass.
- Actor observations are only signals the hardware measures; world-frame velocity, unavailable
  IMU signals, and ideal contact state go to the privileged critic.
- Scale domain-randomization ranges to uncertainty: broad for placeholders, narrower nonzero for
  measured values. Resolve contradictory evidence before rewards or solver tuning.

## Generate the training model

Use a `mujoco.MjSpec` builder that reads the source XML and writes the generated MJX XML, all
edits parameterized and tested.

- Replace task-relevant mesh contacts with measured primitives. Disabling mesh collision also
  disables its self-collision — decide explicitly which floor, body, and inter-link pairs the
  task needs.
- Give the floating base a plausible explicit inertial when the source value is a placeholder.
- Match the actuator interface to deployment: position targets with a PD loop become equivalent
  `gainprm`/`biasprm` with force clamped to the deployable limit. Don't impose PD when deployment
  uses another interface.
- Choose physics and control timesteps from contact stability and controller bandwidth, then
  measure throughput. Repository defaults are evidence, not an MJX convention.
- Preserve closed-loop equality constraints. Change a joint zero convention via the joint's
  `ref`; rotating a child body can silently move compiled loop-closure anchors.

## Contacts

Start with the smallest set that represents every contact the policy can exploit: measured foot
primitives and a base collider for locomotion; leg-to-floor primitives for fall recovery;
self-collision pairs where reachable intersections would let the policy cheat. Exclude pairs that
overlap by construction, and assert the home pose has no unintended contacts. Benchmark the
potential-pair count and MJX step rate — a faster invalid model is not an optimization.

## Home pose and validation

Render a grid of candidate poses, drop each under deployment-like PD hold, and inspect the
settled images — numbers alone cannot distinguish standing from belly contact. Record the settled
`qpos` and controls as a generated `home` keyframe. Before training, verify: expected
bodies/joints/actuators/sensors/mass; force limits matching the audited interface; loop-closure
error within millimetre tolerance at reference and settled poses; sustained standing without
unintended contacts; a reproducible throughput baseline.

Static validation does not prove exploration stability — probe any solver relaxation with a real
training run via `brax-locomotion-training`.

## Red flags

- Editing generated XML by hand
- Placing a collider from a link name instead of measured geometry
- Disabling mesh collision without auditing lost self-collision
- Training an actor on signals the deployed robot cannot measure
