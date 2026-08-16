#!/bin/bash
# IMU robustness grid: an eval-only gyro-bias sweep over existing runs,
# plus optional white gyro noise, gyro-vib feedback, pinned control
# latency, and actuator-torque lag. Those axes can be combined, because
# the real robot carries all of them at once. Standing and straight-walk
# rollouts are scored on vibration, the near-Nyquist 20-25 Hz band, falls,
# and walk tracking. See wojtek_rl/imu_grid.py for what a cell means and
# why the bias is pinned. The grid runs on CPU for determinism and parity
# with the battery and the courses.
set -euo pipefail
cd "$(dirname "$0")/../.."
source training/jobs/_lib.sh

: "${ON_FAILURE:=a missing run directory is skipped with a WARN. A crash inside the grid aborts the remaining cells, and the per-run JSONs written so far survive}"
# Space-separated run names under the training runs dir.
: "${CKPTS_LIST:?space-separated run names under training/runs}"
# Gyro bias magnitudes in rad/s. 0 is the baseline cell.
: "${BIAS_LEVELS:=0 0.05 0.1 0.2}"
# Body axes biased, one cell each.
: "${AXES:=x y z}"
# Optional absolute white gyro-noise scales to sweep. Each value rebuilds
# the measurement env. Empty keeps the run's own trained value only.
: "${NOISE_GYRO:=}"
# Optional gains for the vibration feedback on the actor's gyro
# (obs_noise.gyro_vib). Each value rebuilds the measurement env. Empty
# leaves the loop off.
: "${VIB_GAIN:=}"
# Actuator-torque lag time constants in seconds; 0 keeps the env's own
# ideal actuators. Every value runs inside every env build and costs a
# JIT compile there.
: "${LAG_TAU:=0}"
# Optional control-latency substep counts to pin, one grid axis each.
# Empty leaves the env's own per-episode draw.
: "${LATENCY_SUBSTEPS:=}"
# 1 also runs every NOISE_GYRO level paired with every VIB_GAIN, on top
# of the one-at-a-time cells. Each pair is another env build.
: "${CROSS_NOISE_VIB:=0}"
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
VIB_FLAG=()
if [ -n "$VIB_GAIN" ]; then
    VIB_FLAG=(--vib-gain $VIB_GAIN)
fi
LAG_FLAG=()
if [ -n "$LAG_TAU" ]; then
    LAG_FLAG=(--lag-tau $LAG_TAU)
fi
LATENCY_FLAG=()
if [ -n "$LATENCY_SUBSTEPS" ]; then
    LATENCY_FLAG=(--latency-substeps $LATENCY_SUBSTEPS)
fi
CROSS_FLAG=()
if [ "$CROSS_NOISE_VIB" != "0" ]; then
    CROSS_FLAG=(--cross-noise-vib)
fi

# Guard the array expansion. Under set -u, bash before 4.4 treats an empty
# array expanded the plain way as unbound.
run_main python3 -m wojtek_rl.imu_grid \
    --runs "${RUNS[@]}" \
    --bias-levels $BIAS_LEVELS \
    --axes $AXES \
    --seeds "$SEEDS" \
    --stand-sec "$STAND_SEC" --walk-sec "$WALK_SEC" --walk-vx "$WALK_VX" \
    ${NOISE_FLAG[@]+"${NOISE_FLAG[@]}"} \
    ${VIB_FLAG[@]+"${VIB_FLAG[@]}"} \
    ${LAG_FLAG[@]+"${LAG_FLAG[@]}"} \
    ${LATENCY_FLAG[@]+"${LATENCY_FLAG[@]}"} \
    ${CROSS_FLAG[@]+"${CROSS_FLAG[@]}"} \
    --out "$IMU_GRID_OUT"
