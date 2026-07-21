# Training project guide

Read the root `CLAUDE.md` first for the resolve/smoke/full-run workflow and
the validation rules. This file adds the HPC-script conventions for
`training/hpc/`.

## HPC scripts: parameterized tools, not experiment records

Every `training/hpc/*.slurm` script in this repository is a reusable,
parameterized tool. Experiment-specific values — checkpoints, rung lists,
gate baselines, run names — enter via `sbatch --export=ALL,VAR=value` and
must never be hard-coded into a committed script. Each script's header
documents its required variables, tunables, and a worked submit line.

The as-run instance of a completed experiment (with its baked baselines and
exact submit values) is archived in the resulting keeper's Hugging Face
model repo under `hpc/`, next to the checkpoint it produced:

- [wojtek-stiff-locomotion](https://huggingface.co/<HF_ORGANIZATION>/wojtek-stiff-locomotion) — kp40 keeper: probes + two-phase scripts
- [wojtek-stiff-kp80-locomotion](https://huggingface.co/<HF_ORGANIZATION>/wojtek-stiff-kp80-locomotion) — ladder + consolidation scripts
- [wojtek-stiff-kp90-locomotion](https://huggingface.co/<HF_ORGANIZATION>/wojtek-stiff-kp90-locomotion) — ladder + consolidation scripts

When an experiment finishes and its keeper is published, upload the as-run
scripts (plus the `_common.sh` they sourced) to the keeper's HF repo with
`hf upload <repo> <dir> hpc`, then delete any one-shot script from Git and
fold reusable logic into the parameterized tools below.

## The scripts

All jobs are submitted from the repo root on `ui.cluster.example`; `mkdir -p logs`
first (SLURM writes `logs/%x-%j.{out,err}`). `_common.sh` provides the
shared venv/cache/GPU-assert plumbing and the quota-fallback logic for the
JAX compile cache; every script sources it via `WORKDIR`.

| Script | Partition | Purpose |
|---|---|---|
| `train.slurm` | `PARTITION | Single training run of any `+experiment=` preset. |
| `stiff_ladder.slurm` | `PARTITION | Gated PD-stiffness ladder: fine-tunes successively stiffer rungs from a keeper checkpoint, stops on gate rejection or diminishing returns. Requires `START_CHECKPOINT`, `RUNGS_LIST`, `BASELINE_MEAN_TRACK_ERR`, `BASELINE_VIBRATION_JSON`. |
| `stiff_grid.slurm` | `PARTITION | Eval-only sim2real robustness grid (`--alpha`/`--lag-tau`/`--torque-envelope`) over existing runs; aggregates with `wojtek_rl.grid_report`. Requires `CKPTS_LIST`. |
| `springy_two_phase.slurm` | `PARTITION | Two-phase springy-gait pipeline (predates this convention; its as-run values are still baked in). |

cluster specifics (partitions, quotas, debugging) live in the opt-in
`cluster-hpc` skill; see `skills/README.md`. Per the root `CLAUDE.md`, do not
submit HPC jobs without explicit user authorization — resolve and smoke
locally first, and keep wandb enabled on every real run (offline mode needs
no key; sync from the login node).
