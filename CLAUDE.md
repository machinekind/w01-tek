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
- `experiments/` holds work that is **not production and not on the robot**.
  Each subdirectory is one self-contained experiment with its own `README.md`
  stating its status, and carries whatever it needs (ROS packages, Python
  packages, tests, docs).  Nothing in `experiments/` may become a dependency
  of `wojtek_bringup` or reach the robot through `ros/deploy.sh`, which
  rsyncs `ros/src/` only.  Interfaces there are unstable by definition:
  promote code into `ros/` or `training/` when it stops being an experiment,
  and do not treat an experiment's layout as precedent for the rest of the
  tree.  See
  [experiments/autonomous_architecture_ros2_v1/README.md](experiments/autonomous_architecture_ros2_v1/README.md).
- `skills/` contains opt-in local guides.  Claude users must explicitly
  symlink a skill or ask to read its `SKILL.md`; see [skills/README.md](skills/README.md).
- Use **Wojtek** everywhere: prose, artifact names, run names, and new paths.
  `fbb-*` survives only where it is a literal identifier of an existing
  configuration or published artifact.
- Remote execution and machine provisioning live in a separate private
  operations repository; this repository holds only cluster-agnostic job
  payloads under `training/jobs/`.
- This repository is public.  Never commit credentials or the identity of
  private infrastructure: passwords, tokens, keys; hostnames, IPs, logins,
  or port maps of private machines; cluster site, account, grant, or
  partition names; personal emails or aliases.  Such values enter at run
  time through an environment variable declared at the top of the script
  and filled from the gitignored `.env` (see `.env.example`) — the
  `UBUNTU_PRO_TOKEN` and `WOJTEK_AP_PSK` handling in `ros/deploy.sh` is
  the pattern to copy.  A script that targets one concrete private machine
  does not belong here at all; it goes to the operations repository.
  `training/tests/unit/test_job_scripts.py` enforces this for job
  payloads; the rule covers the whole tree, comments and docs included.
- New code is Apache-2.0 by default; do not pick another license unless the
  author explicitly asks for one.  Declare it as `<license>Apache License
  2.0</license>` in `package.xml` and `license="Apache-2.0"` in
  `setup.py`/`pyproject.toml`.  The only exception is the GPL-licensed IMU
  firmware directories; see `NOTICE`.

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
- [SCAN-Planner](training/docs/scan-planner.md) — the collision-aware local
  planner that executes the VLM's mid-level commands; read it before touching
  `wojtek_rl/scan/`, the room demo's executor, or the nav-eval collision
  metrics.

## Safe training workflow

Run commands from the repository root.  `training/run.sh` changes into the
training project before invoking Python.

```bash
# Inspect the resolved Hydra configuration/overrides; this does not train.
./training/run.sh train --cfg job --resolve

# Resolve an intended change before spending GPU time.
./training/run.sh train +experiment=locomotion \
  ++ppo.num_timesteps=300000000 --cfg job --resolve

# The train/checkpoint path is checked with a short bounded run on a GPU
# box before the full budget.  CPU smoke runs are not viable (MJX compile
# on CPU); do not run them.  A preset can override a short run's PPO
# budget, so pin ++ppo.num_timesteps when selecting one.
```

Rules for agents changing or launching training:

- Use the detailed reference rather than guessing Hydra paths.  Code-defaulted
  environment fields require `+` or `++` on the command line; `++` is the
  safest form when a key may or may not already be present in YAML.
- Run `--cfg job --resolve` first, then a GPU model check
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
- Remote training jobs and policy deployment stay human-authorized: do not
  start a remote run, deploy a policy, or launch/arm the physical robot
  without explicit user authorization.  Resolve/configure locally first.

## Validation by change type

```bash
# Documentation-only change
git diff --check

# Training config or Python change
./training/run.sh train --cfg job --resolve
./training/run.sh test        # tests/unit: model-free, ~3 s

# Anything touching the env, the model, DR, or the latency path
./training/run.sh test-slow   # tests/integration: real MJX, minutes

# Anything under experiments/ (that experiment's own model-free suite)
./experiments/autonomous_architecture_ros2_v1/run.sh test

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
on the launch files.  A Hugging Face reference is resolved from the
gitignored policy store (`ros/policies/`).  `ros/deploy.sh` resolves the
pinned default (see `wojtek_policy/policy_source.py`) into the store and
syncs the store to the RPi, which has no internet.  `ros/deploy.sh --policy
<ref>` fetches a one-off reference, syncs it, and makes the robot run it, and
a plain `ros/deploy.sh` returns the robot to the pin.
Changing the deployed policy is a config change, not a code change.
