#!/bin/bash
# Sim2real robustness grid: eval-only plant perturbations over existing runs,
# sweeping Kt miscalibration, actuator lag and torque envelope, then one report.
# The whole grid runs CPU-mode for determinism and parity with the battery.
set -euo pipefail
cd "$(dirname "$0")/../.."
source training/jobs/_lib.sh

: "${GPUS:=1}"
: "${ON_FAILURE:=a crashed cell logs a WARN and increments a counter, a missing run directory is skipped, and the aggregation always runs}"
# Space-separated run names under the training runs dir.
: "${CKPTS_LIST:?space-separated run names under training/runs}"
# Kt-miscalibration factors.
: "${ALPHAS:=1.0 1.58}"
# Actuator-lag time constants in seconds; 0 keeps the native pipeline.
: "${LAGS:=0 0.005 0.010}"
# Each entry is "none" or "OMEGA_B,OMEGA_0" in rad/s.
: "${ENVELOPES:=none}"
# Path relative to the training project directory.
: "${GRID_REPORT_OUT:=runs/grid_report.md}"

export JAX_PLATFORMS=cpu

CKPTS=($CKPTS_LIST)
ALPHA_LIST=($ALPHAS)
LAG_LIST=($LAGS)
ENV_LIST=($ENVELOPES)

FAILED_CELLS=0
TOTAL_CELLS=0

for run_name in "${CKPTS[@]}"; do
    run_dir="$PROJECT/runs/$run_name"
    if [ ! -d "$run_dir" ]; then
        echo "WARN: runs/$run_name does not exist; skipping"
        continue
    fi
    mkdir -p "$run_dir/grid"

    for alpha in "${ALPHA_LIST[@]}"; do
        for lag in "${LAG_LIST[@]}"; do
            for env in "${ENV_LIST[@]}"; do
                TOTAL_CELLS=$((TOTAL_CELLS + 1))
                lag_ms=$(python3 -c "print(int(round(float('$lag') * 1000)))")
                if [ "$env" = "none" ]; then
                    # No --torque-envelope flag at all. The flag forces the
                    # explicit-PD path even at lag_tau=0, so passing it would
                    # change the cell that is meant to be the native one.
                    env_tag="none"
                    env_flag=()
                else
                    env_tag="${env//,/-}"
                    env_flag=(--torque-envelope "$env")
                fi
                out="$run_dir/grid/battery_a${alpha}_lag${lag_ms}ms_env${env_tag}.json"
                echo "== cell run=$run_name alpha=$alpha lag_tau=${lag}s env=$env_tag -> $out =="
                # Guard the array expansion. Under set -u, bash before 4.4
                # treats an empty array expanded the plain way as unbound.
                if ! run_main python3 -m wojtek_rl.battery \
                    --run "runs/$run_name" --alpha "$alpha" --lag-tau "$lag" \
                    ${env_flag[@]+"${env_flag[@]}"} --out "$out"; then
                    echo "WARN: cell run=$run_name alpha=$alpha lag_tau=$lag env=$env_tag crashed; continuing"
                    FAILED_CELLS=$((FAILED_CELLS + 1))
                fi
            done
        done
    done
done

echo ""
echo "== GRID SUMMARY: $((TOTAL_CELLS - FAILED_CELLS))/$TOTAL_CELLS cells completed, $FAILED_CELLS crashed =="

# shellcheck disable=SC2086
run_main python3 -m wojtek_rl.grid_report --runs $CKPTS_LIST --out "$GRID_REPORT_OUT"

exit 0
