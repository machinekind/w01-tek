# Wojtek agent guide

This repository contains Wojtek's ROS stack (`ros/`) and its MuJoCo MJX/Brax
training project (`training/`). On supported CUDA GPUs, MJX uses the **MJWarp**
backend; its JAX backend is the CPU/non-MJWarp fallback. Brax provides PPO and
training wrappers, not the physics engine. Treat training configuration as an
explicit, reviewable experiment input: resolve it before launching a run and
keep the command, seed, and resulting run directory together.

## Repository map and boundaries

- `training/` is the Hydra-configured MuJoCo MJX/Brax PPO project, with
  **MJWarp** (the Warp implementation of MJX) as its primary CUDA backend.
  `task.env.sim.backend=auto` selects `warp` on a supported GPU and `jax`
  otherwise. Its generated outputs belong in ignored `training/runs/`,
  `training/videos/`, caches, and WandB directories—not in commits.
- `ros/` is the deployment and robot-control workspace.  Treat real-robot
  launch and policy deployment as human-authorized, safety-sensitive actions.
- `ros/src/wojtek_description/mujoco/wojtek.xml` is the model source;
  `wojtek_mjx.xml` and `scene_mjx.xml` are generated via `./training/run.sh
  build`.  Do not hand-edit generated XML.
- `skills/` contains opt-in local guides.  Claude users must explicitly
  symlink a skill or ask to read its `SKILL.md`; see [skills/README.md](skills/README.md).
- Use **Wojtek** for new prose and artifact names. Keep `wojtek` only for
  repository paths and `fbb-*` only where it is a literal historical/current
  configuration or artifact identifier.

## Start with the training reference

- [Training configuration reference](training/docs/configuration.md) — every
  Hydra group, root setting, task setting, domain-randomization switch, PPO
  setting, experiment preset, and command-mode usage.  Its "Course benchmark"
  section defines the path-following scores and the frozen follower constants
  that must not be retuned.
- [Training lessons](skills/brax-locomotion-training/references/wojtek-training-lessons.md)
  — evidence from previous locomotion iterations; consult it before changing
  rewards, observations, or gait behavior.
- [MJWarp backend report](docs/plans/mjwarp-phase0-report.md) — backend buffer
  sizing and validation context.
- [Demo guide](training/demo/README.md) — interactive navigation demo usage.

## Safe training workflow

Run commands from the repository root.  `training/run.sh` changes into the
training project before invoking Python.

```bash
# Inspect the resolved Hydra configuration/overrides; this does not train.
./training/run.sh train --cfg job --resolve

# Resolve an intended change before spending GPU time.
./training/run.sh train +experiment=locomotion \
  ++ppo.num_timesteps=300000000 --cfg job --resolve

# Then use a CPU smoke run to check the full train/checkpoint path.  A preset
# can override smoke's short PPO budget, so pin it again when selecting one.
./training/run.sh smoke +experiment=locomotion \
  ++ppo.num_timesteps=100000 \
  run_name=wojtek_smoke_locomotion_$(date +%Y%m%d_%H%M%S) wandb.enable=false
```

Rules for agents changing or launching training:

- Use the detailed reference rather than guessing Hydra paths.  Code-defaulted
  environment fields require `+` or `++` on the command line; `++` is the
  safest form when a key may or may not already be present in YAML.
- Run `--cfg job --resolve` first. Use a smoke run, then a GPU model check
  (`./training/run.sh check --gpu --backend warp`) and a bounded train before
  a full budget. Do not claim a reward change is successful without the fixed
  evaluation battery or report.
- Actor observations must be available on the physical robot.  Observation
  layout changes invalidate checkpoints and deployment artifacts.
- Keep task-specific options task-specific: latency and encoder offsets exist
  only for `joystick`; do not add them to `getup` or `jump` runs.
- `task.env.sim.num_envs` is set by the trainer from PPO batch settings.  Tune
  `++ppo.num_envs`, not the environment field, for a training run.
- A policy trained with `task.env.action_filter > 0` needs the equivalent
  filter in its deployment control loop.
- Do not submit HPC/cloud jobs, deploy a policy, or launch/arm the physical
  robot without explicit user authorization.  Resolve/configure locally first.

## Validation by change type

```bash
# Documentation-only change
git diff --check

# Training config or Python change
./training/run.sh train --cfg job --resolve
./training/run.sh test        # tests/unit: model-free, ~3 s

# Anything touching the env, the model, DR, or the latency path
./training/run.sh test-slow   # tests/integration: real MJX, minutes

# Model-generation change (inspect generated XML before committing it)
./training/run.sh build
./training/run.sh check
```

`test` and `test-slow` are a hard split: `tests/unit` never instantiates an
env or puts a model on device (a guard test enforces it), which is why it
stays fast enough to run on every edit.  `tests/integration` pays real MJX
compile time (6m23s measured cold); it sets a persistent JAX compilation
cache so repeat runs skip recompilation.  Run it before claiming an env/model
change is validated.

For wider repository checks, use `make verify-static` first and
`make verify-quick` when its prerequisites are available.  A full
`make verify` includes the slower Docker/ROS gate.

## cluster access

Personal cluster values (`cluster_USER` etc.) live in the gitignored repo-root
`.env`, which agent sessions are permission-denied from reading — never
`cat`/`grep` it, and don't conclude the cluster is unreachable when that
denial hits.  `./cluster.sh <cmd>` runs a command on the login node and the
`make hpc-*` targets handle rsync/submit/status; both load `.env`
themselves.

## Common commands

```bash
# Full training (choose a unique run name for a durable run)
./training/run.sh train +experiment=locomotion run_name=wojtek_locomotion seed=1

# Evaluate, compare, and prepare a completed joystick policy for ROS
./training/run.sh report --run runs/wojtek_locomotion
./training/run.sh courses --run runs/wojtek_locomotion   # path-following score per scenario
./training/run.sh eval --run runs/wojtek_locomotion --x-vel 0.3 --height 0.125
./training/run.sh export --run runs/wojtek_locomotion
```

Generated training outputs live below `training/runs/` and are intentionally
ignored by Git.  Do not commit checkpoints, videos, or WandB artifacts.
Policy artifacts are never vendored into `ros/` either: `export` writes a
self-contained `policy.npz` + `policy_meta.json` (the schema-2 deployment
contract), keepers publish that pair to their Hugging Face repo, and the
ROS stack loads a policy by reference -- `policy:=<org/name[@rev] | dir>`
on the launch files.  Changing the deployed policy is a config change, not
a code change.
