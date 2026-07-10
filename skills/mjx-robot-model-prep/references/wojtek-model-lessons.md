# Wojtek MJX model lessons

Load only for Wojtek work or when a concrete closed-loop quadruped example helps.

## Repository map

- Source model `ros/src/wojtek_description/mujoco/wojtek.xml`; generated `wojtek_mjx.xml` and
  scene `scene_mjx.xml` in the same directory
- Deployment-packaged copy: `ros/src/wojtek_bringup/config/wojtek_mjx.xml`
- Builder `training/wojtek_rl/build_model.py`; tests `training/tests/test_build_model.py`,
  `training/tests/test_keyframe.py`; static/MJX checks `training/wojtek_rl/check_model_mjx.py`

Read current constants from the builder and deployment config, not legacy paths or design plans.

## Model-copy provenance

The differing MJX model copies are per-generation, not accidental drift:

- description + builder (±9 Nm force limits, ~16 kg total): the current training line's baseline;
- bringup copy (±6 Nm, ~10 kg, own home keyframe): the exact physics the deployed `fbb_loco_v8`
  policy was trained on, kept intentionally.

Still open: the true hardware torque limit is unverified, and the canonical model for a future
consolidation pass is undecided. Until then, treat neither file as "the" physics model, and do
not re-flag the ±6 copy as a bug.

## Lessons that generalized

- The source model shipped with placeholder physical values; comparing total mass against
  actuator torque exposed it before reward work began.
- An early policy used IMU actor observations the deployed robot could not provide; actor and
  critic observation catalogs now require an explicit hardware audit.
- Fix encoder or joint-zero conventions with joint `ref`. Rotating a child frame can move the
  second anchor of a compiled `connect` equality and break a four-bar loop while tests stay green.
- Measure the foot contact point from settled geometry; a link named "foot" need not contain the
  lowest point of a closed linkage.
- Contacts follow the task: locomotion needs feet and base; get-up adds leg-to-floor; reachable
  inter-link intersections may need self-collision.
- Growing the contact graph from 14 to 78 potential pairs cut throughput from ~127k to ~48k
  steps/s. Benchmark pair changes; the numbers are not targets.
- One Newton iteration passed static tests but diverged ~23M steps into exploration; two were
  stable for this model. Probe solver relaxations with the real training distribution.

The home keyframe was chosen by rendering settled candidates under PD hold; tests guard standing
duration and loop-closure error. Re-run that process whenever geometry, gains, mass, or joint
references change — do not assume the stored keyframe stays valid.
