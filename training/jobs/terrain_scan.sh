#!/bin/bash
# Terrain measurement suite: build the fixed measurement arena, then score
# checkpoints on it. Eval only, no training.
# Warp produces the reported numbers; BACKEND=jax with a cell subset is the
# cheaper cross-check.
set -euo pipefail
cd "$(dirname "$0")/../.."
source training/jobs/_lib.sh

: "${GPUS:=1}"
: "${ON_FAILURE:=a missing run dir is skipped, a crashed scan seed or report logs a WARN and the loop moves on; only a failed arena build aborts}"
: "${CKPTS_LIST:?space-separated run names under training/runs}"
# Empty means absolute bars only. Accepts a path, a directory holding
# terrain_scan.json, or a model-hub reference org/name[@rev].
: "${BASELINE:=}"
: "${BACKEND:=auto}"
# Warp contact pool per env, allocated up front for every env at once. The
# scan refuses to call an overflowed run a measurement; on overflow use 256.
: "${NACONMAX_PER_ENV:=88}"
# Empty keeps the env's own 320. Constraint rows past the limit apply no force
# and warn nowhere, so the scan records the peak instead.
: "${NJMAX:=}"
# Empty means the suite's full set.
: "${CELLS:=}"
: "${SPEEDS:=}"
# Seed 0 writes terrain_scan.json, other seeds write terrain_scan_seed<k>.json.
: "${EVAL_SEEDS:=0}"
: "${CHECK_CONTACTS:=1}"
# clean | dark. dark feeds a zero height scan and adds a _dark infix.
: "${SCAN:=clean}"
# Applies to the report only, never to the scan. The scan is always warp on the
# GPU, because warp produces the reported numbers. Empty means the report uses
# whatever the machine offers; cpu pins it. A report's flat battery only
# compares with another run's when both used the same one.
: "${EVAL_PLATFORM:=}"

# run_main changes the working directory, so record the repository root now.
REPO_ROOT="$PWD"

EVAL_ENV=()
if [ -n "$EVAL_PLATFORM" ]; then
    EVAL_ENV=(env "JAX_PLATFORMS=$EVAL_PLATFORM")
fi

# The measurement arena is generated and not tracked in Git, so build it.
# Its seed, rows and pad radius come from the suite and cannot be overridden.
run_main python3 -m wojtek_rl.build_terrain --arena eval

if [ "$CHECK_CONTACTS" = "1" ]; then
    # This payload must never build the train arena. Its rows, seed and tile
    # size are per-experiment, so a default rebuild would replace the arena a
    # policy trained on. Measure it only when it already exists.
    ARENAS=(eval)
    TRAIN_SCENE="$REPO_ROOT/ros/src/wojtek_description/mujoco/scene_terrain.xml"
    if [ -f "$TRAIN_SCENE" ]; then
        ARENAS+=(train)
    else
        echo "NOTE: no training arena built; skipping its contact measurement."
        echo "      Build it yourself with the run's own parameters:"
        echo "      ./training/run.sh build-terrain --arena train [--rows N --seed N]"
    fi
    for arena in "${ARENAS[@]}"; do
        run_main python3 -m wojtek_rl.check_terrain \
            --backend "$BACKEND" --arena "$arena" \
            --num-envs 256 --steps 200 \
            || echo "WARN: check-terrain --arena $arena failed; continuing"
    done
fi

SCAN_FLAGS=(--backend "$BACKEND" --naconmax-per-env "$NACONMAX_PER_ENV" --scan "$SCAN")
if [ -n "$NJMAX" ]; then SCAN_FLAGS+=(--njmax "$NJMAX"); fi
if [ -n "$BASELINE" ]; then SCAN_FLAGS+=(--baseline "$BASELINE"); fi
if [ -n "$CELLS" ]; then SCAN_FLAGS+=(--cells "$CELLS"); fi
if [ -n "$SPEEDS" ]; then SCAN_FLAGS+=(--speeds "$SPEEDS"); fi

CKPTS=($CKPTS_LIST)
FAILED=0
TOTAL=0
for run_name in "${CKPTS[@]}"; do
    run_dir="$PROJECT/runs/$run_name"
    if [ ! -d "$run_dir" ]; then
        echo "WARN: runs/$run_name does not exist; skipping"
        continue
    fi
    TOTAL=$((TOTAL + 1))
    CKPT_FAILED=0
    for seed in $EVAL_SEEDS; do
        mode=""
        if [ "$SCAN" = "dark" ]; then mode="_dark"; fi
        out="runs/$run_name/terrain_scan${mode}.json"
        if [ "$seed" != "0" ]; then out="runs/$run_name/terrain_scan${mode}_seed${seed}.json"; fi
        echo "== terrain scan run=$run_name seed=$seed backend=$BACKEND baseline=${BASELINE:-none} =="
        if ! run_main python3 -m wojtek_rl.terrain_scan \
            --run "runs/$run_name" --eval-seed "$seed" --out "$out" \
            "${SCAN_FLAGS[@]}"; then
            echo "WARN: terrain scan for $run_name seed $seed crashed; continuing"
            CKPT_FAILED=1
            continue
        fi
    done
    if [ "$CKPT_FAILED" = "1" ]; then
        FAILED=$((FAILED + 1))
        continue
    fi
    # The report picks the scan JSON up from the run directory, so the
    # markdown carries the terrain table next to the flat battery.
    run_main ${EVAL_ENV[@]+"${EVAL_ENV[@]}"} python3 -m wojtek_rl.report \
        --run "runs/$run_name" \
        || echo "WARN: report for $run_name failed; the scan JSON is still there"
done

echo ""
echo "== TERRAIN SCAN SUMMARY: $((TOTAL - FAILED))/$TOTAL checkpoints scanned, $FAILED crashed =="
echo "== publish each keeper's runs/<name>/terrain_scan.json to its HF repo: it IS the next run's BASELINE =="

exit 0
