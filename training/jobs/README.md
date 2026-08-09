# Job scripts

Each script here is a reusable, parameterized tool. It holds domain logic only:
build a model, train a policy, run an evaluation battery, or chain those steps.
No script knows how it is dispatched, and none is an experiment record.

They are plain bash. Nothing outside this repository is needed to run one.

## Running one

Set up the training project the usual way, so `training/.venv` exists, then
run a script from anywhere:

```bash
EXPERIMENT=locomotion bash training/jobs/train.sh
```

It finds the repository from its own path, puts `training/.venv/bin` on `PATH`
when no environment is already active, sets a headless-safe `MUJOCO_GL`, and
uses whatever GPUs the machine has.

## Parameters

Every input is an environment variable, declared at the top of the script in
one of three ordinary shell forms:

```bash
: "${CKPTS_LIST:?space-separated run names}"    # required
: "${BACKEND:=auto}"                            # optional
export MUJOCO_GL="${MUJOCO_GL:-disabled}"       # optional, exported
```

Each form carries its own behaviour. A missing required input prints its name
and stops the script before it does anything. An optional one takes its
default. Empty counts as missing in all three, so a blank value cannot slip
past a required input.

The declaration is the code that applies it, so there is no second copy that
can drift. To see what a script takes, read its declaration block. To change a
value, pass the variable.

Experiment-specific values (checkpoints, rung lists, gate baselines, run names)
always enter this way. A committed script never hard-codes one.

Two of the inputs describe the script rather than configure it. They are
declared the same way as the rest, so there is only ever one syntax:

```bash
: "${GPUS:=4}"
: "${ON_FAILURE:=aborts, the first failing step ends the run}"
```

## Conventions

- Outputs go under `training/runs`, plus a `logs` directory at the repository
  root and an offline tracking directory when those are used.
- A script exits non zero when it fails.
- A script contains no scheduler directive, no hostname, and no file transfer.
  `tests/unit/test_job_scripts.py` enforces this, along with the rule that
  every input is declared before any work starts.
- `_lib.sh` holds the shared logging and environment setup. It is local to
  this repository and is not part of any dispatch interface.

## Running these remotely

Remote execution and machine provisioning live in a separate private
operations repository. Nothing here dispatches, schedules, or copies anything.

That dispatcher reads its per-project values (payload directory, remote
checkout name, sync excludes) from a `.gpu-ops` file at this repository's
root. The file is personal and gitignored; `.gpu-ops.example` at the root is
the committed template, and no script here reads either file.
