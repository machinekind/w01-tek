#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source training/jobs/_lib.sh

# One training run of a Hydra experiment preset.
# The default trainer is Brax PPO, which data-parallelizes over every
# local GPU, so NUM_ENVS and BATCH are whole-machine totals, not per-GPU
# numbers. MODULE selects another trainer with the same Hydra surface;
# wojtek_rl.distill is single-GPU and keeps the preset's own env count.

# The floor for the machine's GPU count, which the submit decides. PPO
# sizes itself to whatever the box has, so a single-GPU rental is as valid
# as a 4-GPU node.
: "${GPUS:=1}"
: "${ON_FAILURE:=aborts, the first failing step ends the run}"
# Persistent dir for tracking output, if the machine has one.
: "${STORE_DIR:=}"
# Python module to run: wojtek_rl.train (PPO) or wojtek_rl.distill.
: "${MODULE:=wojtek_rl.train}"
: "${EXPERIMENT:=locomotion}"
# Empty: the preset picks the run dir.
: "${RUN_NAME:=}"
: "${NUM_ENVS:=32768}"
: "${BATCH:=1024}"
# 1: build the train arena first.
: "${TERRAIN:=0}"
# The preset must declare the same flat row.
: "${FLAT_ROW:=0}"
# Extra build_terrain flags, word split.
: "${TERRAIN_FLAGS:=}"
: "${WANDB:=1}"
# Empty: decided by the probe below.
: "${WANDB_MODE:=}"
: "${WANDB_API_KEY:=}"
# Empty: STORE_DIR/wandb, or training/wandb.
: "${WANDB_DIR:=}"
# Extra hydra overrides, word split.
: "${EXTRA:=}"

# Online needs credentials and a reachable API host. Offline needs neither and
# can be synced later.
if [ "$WANDB" = "1" ]; then
    if [ -z "$WANDB_MODE" ]; then
        # The API root answers plain GETs with 404, so any HTTP status counts
        # as reachable. Only 000 means no connection.
        wandb_http=$(curl -sm 5 -o /dev/null -w '%{http_code}' https://api.wandb.ai 2>/dev/null || true)
        if { [ -n "$WANDB_API_KEY" ] || grep -qs api.wandb.ai "$HOME/.netrc"; } \
            && [ "${wandb_http:-000}" != "000" ]; then
            WANDB_MODE=online
        else
            echo "wandb: no credentials or api.wandb.ai unreachable (http=${wandb_http:-000}); offline mode"
            WANDB_MODE=offline
        fi
    fi
    export WANDB_MODE
    export WANDB_DIR="${WANDB_DIR:-${STORE_DIR:-$PROJECT}/wandb}"
    # Keep logging alive when the configured directory is not writable.
    # Offline run directories are small, so the checkout can hold them.
    if ! { mkdir -p "$WANDB_DIR" && touch "$WANDB_DIR/.write_probe"; } 2>/dev/null; then
        echo "WARN: $WANDB_DIR not writable; using $PROJECT/wandb-offline"
        export WANDB_DIR="$PROJECT/wandb-offline"
        mkdir -p "$WANDB_DIR"
    else
        rm -f "$WANDB_DIR/.write_probe"
    fi
    WANDB_FLAG="wandb.enable=true"
else
    WANDB_FLAG="wandb.enable=false"
fi

if [ "$TERRAIN" = "1" ]; then
    # Mandatory on terrain, not a tuning preference: warp allocates its EPA
    # scratch outside the XLA pool.
    export XLA_PYTHON_CLIENT_PREALLOC=false
    TERRAIN_ARGS=()
    if [ "$FLAT_ROW" = "1" ]; then
        TERRAIN_ARGS+=("--flat-row")
    fi
    if [ -n "$TERRAIN_FLAGS" ]; then
        # Word splitting is the point: TERRAIN_FLAGS is a flag string.
        # shellcheck disable=SC2206
        TERRAIN_ARGS+=($TERRAIN_FLAGS)
    fi
    # Bash 3.2 with set -u errors on an empty array expansion.
    run_main python3 -m wojtek_rl.build_terrain --arena train \
        ${TERRAIN_ARGS[@]+"${TERRAIN_ARGS[@]}"}
fi

RUN_ARGS=()
if [ -n "$RUN_NAME" ]; then
    RUN_ARGS+=("run_name=$RUN_NAME")
fi

# ++ = add-or-override: the root `ppo:` block is an empty dict, so plain
# `ppo.foo=` fails hydra's struct check for keys the preset did not set.
# shellcheck disable=SC2086
run_main python3 -m "$MODULE" \
    "+experiment=$EXPERIMENT" \
    "++ppo.num_envs=$NUM_ENVS" \
    "++ppo.batch_size=$BATCH" \
    "$WANDB_FLAG" \
    ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} \
    $EXTRA
