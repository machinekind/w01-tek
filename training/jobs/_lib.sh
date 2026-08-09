#!/bin/bash
# Shared logging and environment setup for the payloads in this directory.
#
# This is local to this repository. It is not the dispatch contract: payloads
# declare their parameters in plain shell, and nothing here is required to run
# one. Source it after changing to the repository root.

PROJECT="$PWD/training"

# Defaults that let a payload run on a plain checkout with nothing prepared.
# A caller that prepares its own environment sets these first and wins.

# The project venv, the one training/run.sh uses. Skipped when an environment
# is already active.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -x "$PROJECT/.venv/bin/python3" ]; then
    PATH="$PROJECT/.venv/bin:$PATH"
    export PATH
fi

# These jobs never render, and a compute node has no display. Without this,
# mujoco picks a GL backend at import and fails on a headless machine.
export MUJOCO_GL="${MUJOCO_GL:-disabled}"

# Progress lines appear while the job runs instead of when the buffer fills.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Runs a command from the training project directory and returns its exit
# code. Capture the code before the final echo: a caller writing
# "run_main X || handler" suspends set -e inside the function, so without the
# capture a crashed X would return the echo's 0 instead.
run_main() {
    cd "$PROJECT"
    # No hostname: this log gets read somewhere else, and a machine name is
    # the caller's business to record, not the job's. JOB_ID identifies the run.
    echo "== $(date -Iseconds) job=${JOB_ID:-none} =="
    echo "== running: $*"
    "$@"
    local rc=$?
    echo "== $(date -Iseconds) done rc=$rc =="
    return "$rc"
}
