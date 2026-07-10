# Wojtek Brax training lessons

Load only for Wojtek work or when a concrete locomotion-training example helps.

## Repository map

- Package `training/wojtek_rl/`, Hydra configs `training/wojtek_rl/conf/`, launcher `training/run.sh`
- Iteration history: `training/docs/2026-07-06-locomotion-overnight-iterations.md`
- Some configs and run names still carry pre-rename (`fbb_*`) identifiers; don't propagate them
  into new artifacts.

```bash
cd training
./run.sh smoke +experiment=locomotion_v8 run_name=wojtek_smoke ppo.num_timesteps=100000
./run.sh train +experiment=locomotion_v8 run_name=wojtek_loco_v8
./run.sh battery --run runs/wojtek_loco_v8
```

The explicit smoke budget is required: experiment presets set `ppo.num_timesteps` (300M for
locomotion) and `cfg.ppo` overrides apply after the trainer's smoke defaults.

## Lessons that generalized

- Build the fixed eval battery before reward tuning; pushes and command-transition motion
  initially contaminated comparisons.
- Track average episode length alongside reward: a policy can raise reward by dying and
  resetting early.
- Measure standing only in hold windows; report settling separately, or transitions look like
  persistent vibration.
- One intent per iteration. Longer training improved the proxy reward but did not repair a
  mis-balanced tracking-versus-slip objective.
- A policy can track velocity while oscillating rapidly; the joint-velocity spectral metric
  exposed what mean velocity and reward missed.
- Mechanical-power, joint-velocity, action-rate, and action-acceleration penalties reduced
  buzzing, at an expected cost in peak tracking.
- A probe once showed exploding value loss with finite, rising evals; the matching full run
  trained normally. Probe value loss is evidence, not a stop condition.
- Locomotion presets use a no-IMU actor (`obs: no_imu`) with privileged critic signals, matching
  deployable sensors.

Record the exact Hydra config, generated model, seed, checkpoint path, and battery result per
run; override legacy `fbb_*` preset run names with Wojtek names.
