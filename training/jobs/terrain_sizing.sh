#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source training/jobs/_lib.sh

# Bounded terrain training slices at several global env counts.
# Reports peak GPU memory and steps/s per size; both are hardware specific, so
# a new machine class needs its own measurement before a real run.
# Slices are throwaway trainings of the exact preset the real run will use.

: "${GPUS:=4}"
: "${ON_FAILURE:=a failed slice is recorded as FAILED and the next size still runs; a failed terrain build ends the job}"
: "${EXPERIMENT:=terrain_blind_v2}"
# Ascending: an OOM on a big size cannot block the smaller ones.
: "${SIZES_LIST:=8192 16384 32768}"
# Timestep budget per slice.
: "${STEPS:=30000000}"
# Extra hydra overrides, word split.
: "${EXTRA:=}"
# Tags slice run names and memory logs. A dispatcher supplies its job id.
: "${JOB_ID:=$(date +%Y%m%d_%H%M%S)}"

# Mandatory on terrain, not a tuning preference: warp allocates its EPA
# scratch outside the XLA pool. It also makes nvidia-smi's memory.used track
# real usage instead of the pool.
export XLA_PYTHON_CLIENT_PREALLOC=false

LOG_DIR="$PWD/logs"
mkdir -p "$LOG_DIR"

run_main python3 -m wojtek_rl.build_terrain --arena train

# One data point for the launch decision, not a gate.
if curl -sI --max-time 10 https://api.wandb.ai >/dev/null 2>&1; then
    echo "NET: this machine CAN reach api.wandb.ai"
else
    echo "NET: this machine CANNOT reach api.wandb.ai (use offline wandb)"
fi

for envs in $SIZES_LIST; do
    # Batch is num_envs/32: at the default 32 minibatches brax wants
    # batch_size*num_minibatches divisible by num_envs.
    batch=$((envs / 32))
    run_name="sizing_${EXPERIMENT}_e${envs}_${JOB_ID}"
    mem_log="$LOG_DIR/sizing-mem-${envs}-${JOB_ID}.csv"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 5 \
        > "$mem_log" &
    poller=$!
    echo "== SIZING slice: num_envs=$envs batch=$batch steps=$STEPS =="
    # shellcheck disable=SC2086
    if run_main python3 -m wojtek_rl.train \
        "+experiment=$EXPERIMENT" \
        "++ppo.num_envs=$envs" \
        "++ppo.batch_size=$batch" \
        "++ppo.num_timesteps=$STEPS" \
        "run_name=$run_name" \
        wandb.enable=false \
        $EXTRA; then
        verdict="OK"
    else
        verdict="FAILED"
    fi
    kill "$poller" 2>/dev/null || true
    wait "$poller" 2>/dev/null || true
    peak=$(awk -F', ' '{if ($2+0 > m[$1]) m[$1]=$2} END {for (g in m) printf "gpu%s:%sMiB ", g, m[g]}' "$mem_log")
    echo "== SIZING RESULT envs=$envs batch=$batch verdict=$verdict peak_mem: $peak"
done
echo "== SIZING DONE =="
