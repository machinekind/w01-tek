#!/bin/bash
# IMU robustness grid: eval-only gyro-bias (and optional gyro white-noise)
# sweep over existing runs, scored on standing and straight-walk rollouts --
# vibration, the near-Nyquist 20-25 Hz band, falls, and walk tracking. See
# wojtek_rl/imu_grid.py for what a cell means and why the bias is pinned.
# CPU-mode for determinism and parity with the battery and the courses.
set -euo pipefail
cd "$(dirname "$0")/../.."
source training/jobs/_lib.sh

: "${ON_FAILURE:=a missing run directory is skipped with a WARN; a crash inside the grid aborts the remaining cells (per-run JSONs written so far survive)}"
# Space-separated run names under the training runs dir.
: "${CKPTS_LIST:?space-separated run names under training/runs}"
# Gyro bias magnitudes in rad/s; 0 is the baseline cell.
: "${BIAS_LEVELS:=0 0.05 0.1 0.2}"
# Body axes biased, one cell each.
: "${AXES:=x y z}"
# Optional absolute white gyro-noise scales to sweep (each rebuilds the
# measurement env); empty = the run's own trained value only.
: "${NOISE_GYRO:=}"
# Rollouts per cell and scenario.
: "${SEEDS:=3}"
# Measured standing / walking window per rollout, seconds.
: "${STAND_SEC:=10}"
: "${WALK_SEC:=10}"
# Walk scenario's commanded speed, m/s.
: "${WALK_VX:=0.4}"
# Combined report path, relative to the training project directory.
: "${IMU_GRID_OUT:=runs/imu_grid_report.md}"

export JAX_PLATFORMS=cpu

CKPTS=($CKPTS_LIST)
RUNS=()
for run_name in "${CKPTS[@]}"; do
    if [ ! -d "$PROJECT/runs/$run_name" ]; then
        echo "WARN: runs/$run_name does not exist; skipping"
        continue
    fi
    RUNS+=("runs/$run_name")
done
if [ "${#RUNS[@]}" -eq 0 ]; then
    echo "ERROR: none of the requested runs exist under $PROJECT/runs"
    exit 1
fi

NOISE_FLAG=()
if [ -n "$NOISE_GYRO" ]; then
    NOISE_FLAG=(--noise-gyro $NOISE_GYRO)
fi

# Guard the array expansion: under set -u, bash before 4.4 treats an empty
# array expanded the plain way as unbound.
run_main python3 -m wojtek_rl.imu_grid \
    --runs "${RUNS[@]}" \
    --bias-levels $BIAS_LEVELS \
    --axes $AXES \
    --seeds "$SEEDS" \
    --stand-sec "$STAND_SEC" --walk-sec "$WALK_SEC" --walk-vx "$WALK_VX" \
    ${NOISE_FLAG[@]+"${NOISE_FLAG[@]}"} \
    --out "$IMU_GRID_OUT"
