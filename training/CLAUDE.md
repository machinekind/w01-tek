# Training project guide

Read the root `CLAUDE.md` first for the resolve/bounded-run/full-run workflow and
the validation rules. This file adds the payload conventions for
`training/jobs/`. As everywhere in the repository, new code is Apache-2.0
(`license = "Apache-2.0"` in `pyproject.toml`) unless the author explicitly
asks for a different license.

## Payloads: parameterized tools, not experiment records

Every script in `training/jobs/` is a reusable, parameterized tool. Experiment-
specific values — checkpoints, rung lists, gate baselines, run names — enter
through environment variables, and every such variable is declared at the top
of the script. A committed script never hard-codes an experiment value.

The declarations are plain shell, so they are both the documentation and the
code that applies them:

```bash
: "${CKPTS_LIST:?space-separated run names}"    # required
: "${BACKEND:=auto}"                            # optional
```

Read the declaration block to see what a script takes. A missing required
value stops the script before it does any work.

`training/jobs/README.md` covers the conventions in full.

The as-run instance of a completed experiment (with its baked baselines and
exact parameter values) is archived in the resulting keeper's Hugging Face
model repo under `hpc/`, next to the checkpoint it produced:

- [wojtek-stiff-locomotion](https://huggingface.co/<HF_ORGANIZATION>/wojtek-stiff-locomotion) — kp40 keeper: probes + two-phase scripts
- [wojtek-stiff-kp80-locomotion](https://huggingface.co/<HF_ORGANIZATION>/wojtek-stiff-kp80-locomotion) — ladder + consolidation scripts
- [wojtek-stiff-kp90-locomotion](https://huggingface.co/<HF_ORGANIZATION>/wojtek-stiff-kp90-locomotion) — ladder + consolidation scripts
- [wojtek-springy-locomotion](https://huggingface.co/<HF_ORGANIZATION>/wojtek-springy-locomotion) / [-v2](https://huggingface.co/<HF_ORGANIZATION>/wojtek-springy-locomotion-v2) — springy two-phase script

When an experiment finishes and its keeper is published, upload the as-run
scripts to the keeper's HF repo with `hf upload <repo> <dir> hpc`, then delete
any one-shot script from Git and fold reusable logic into the parameterized
payloads below.

## The payloads

| Script | Purpose |
|---|---|
| `train.sh` | One training run of any `+experiment=` preset, with optional terrain build and wandb setup. |
| `stiff_ladder.sh` | Gated PD-stiffness ladder: fine-tunes successively stiffer rungs from a start checkpoint and stops on gate rejection or diminishing returns. |
| `stiff_grid.sh` | Eval-only sim2real robustness grid over existing runs, sweeping Kt miscalibration, actuator lag and torque envelope, then one aggregated report. |
| `imu_grid.sh` | Eval-only IMU robustness grid over existing runs. Sweeps a pinned gyro bias, and optionally white gyro noise, the gyro-vib feedback gain, pinned control latency, and actuator-torque lag, alone or combined. Scores standing and walking vibration, the 20-25 Hz limit-cycle band, falls, and walk tracking. `BISECT_VIB` adds the critical-gain search, which finds the vibration gain where a policy's stand stops being motionless. |
| `terrain_scan.sh` | Eval-only terrain measurement suite: builds the fixed measurement arena and scores checkpoints on it. |
| `terrain_sizing.sh` | Bounded terrain training slices at several env counts, reporting peak GPU memory and steps/s per size. |

Per the root `CLAUDE.md`, do not start a remote training job without explicit
user authorization — resolve locally first, and keep wandb enabled on every
real run (offline mode needs no key and can be synced later).
