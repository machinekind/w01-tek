---
name: brax-locomotion-training
description: Design, run, evaluate, and debug MuJoCo MJX locomotion training with Brax PPO. Use for deployable observations and rewards, smoke/probe/full-run gates, NaN divergence, pytree scan-carry errors, gait quality and vibration metrics, and judging run health.---

# Train locomotion policies with Brax

Prepare the model with the `mjx-robot-model-prep` skill first; for GPU-box operations use
`vastai-gpu-training-ops`. Worked example: `training/wojtek_rl/`. Read
`references/wojtek-training-lessons.md` only when training Wojtek.

## Training ladder

| Stage | Gate |
| --- | --- |
| Unit/static tests | Wiring tests pass |
| CPU smoke | Reset, step, PPO, checkpoint, and `run.json` paths all execute |
| GPU probe | Real model and config: eval values finite, behavior improves |
| Full run | Selected budget spent once, no divergence |
| Policy evaluation | Fixed battery and videos meet success criteria |

Never promote a solver, timestep, randomization, or reward change on static tests alone; only a
probe with the real training distribution exposes exploration instability.

## Judgment calls

- Actor observations must exist on hardware; sim-only signals go to the privileged critic. Keep
  observation names and shapes explicit so deployment can reproduce the actor input.
- Agree on task goals, gait style, and the energy-versus-robustness trade-off before porting a
  reward package. Tuned recipes (e.g. Playground Go1 PPO) are starting values, not invariants.
- Estimate the maximum undiscounted episode reward from weights, control period, and episode
  length; a large mismatch usually means missing time scaling or a wrong sign.
- Keep every pytree carried through `jax.lax.scan` structurally identical across steps: merge new
  metrics over `state.metrics` (wrappers inject keys). A "symmetric difference of key sets" error
  is a structural mismatch, not a reason to disable evals.
- Judge probes by finite eval rewards, episode length, and rendered motion. Probe value loss is
  noisy; sustained explosive growth in a full run still warrants investigation.
- Stop a run whose metrics or parameters go non-finite. Check checkpoints with
  `<project-python> skills/brax-locomotion-training/scripts/check_nan.py RUN_DIR`. Fix
  the cause — solver settings, action scale, reward outliers, randomization extremes — then
  re-probe.

## Evaluate behavior, not reward

Build a fixed deterministic battery before tuning: commanded-vs-achieved velocity, stance height,
falls, slip, episode length; rendered walk/turn/stand/transition rollouts; steady-state windows
separated from transitions; no undocumented disturbances. Quantify vibration as the fraction of
joint-velocity FFT power above a chosen frequency; buzzing responds to mechanical-power,
joint-velocity, action-rate, and action-acceleration penalties at a measurable tracking cost.
Change one intent per experiment and compare against the battery, not reward alone.

## Red flags

- Disabling evals to hide a scan-carry error
- Treating higher reward as proof of a better policy
- Running the full budget before a real GPU probe
- Ignoring declining episode length or non-finite eval values
