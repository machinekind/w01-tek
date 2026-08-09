#!/bin/bash
# PD-stiffness ladder. Each rung fine-tunes from the previous accepted rung,
# rung 1 from START_CHECKPOINT. Gates run after each rung; the first rejection
# or diminishing-returns verdict stops the ladder and the job still exits 0.
set -euo pipefail
cd "$(dirname "$0")/../.."
source training/jobs/_lib.sh

: "${GPUS:=4}"
: "${ON_FAILURE:=a bad rung stops the ladder but the summary still prints and the job exits 0; a training crash aborts hard; rung 1 not finishing training and battery is a job failure}"
: "${START_CHECKPOINT:?checkpoint the first rung fine-tunes from}"
# Space-separated kp values, run in order.
: "${RUNGS_LIST:?space-separated kp values}"
# Start checkpoint's 4-scenario mean track_err_rms.
: "${BASELINE_MEAN_TRACK_ERR:?start checkpoint mean track_err_rms}"
# JSON object: per-scenario vibration of the start checkpoint.
: "${BASELINE_VIBRATION_JSON:?per-scenario vibration JSON of the start checkpoint}"
# Persistent dir for tracking output, if the machine has one.
: "${STORE_DIR:=}"
# Rung kp K trains with +experiment=<prefix>K.
: "${PRESET_PREFIX:=stiff_ladder_kp}"
: "${RUN_PREFIX:=wojtek_stiff_kp}"
# Run-name suffix shared by all rungs.
: "${STAMP:=$(date +%Y%m%d_%H%M%S)}"
: "${NUM_ENVS:=32768}"
: "${BATCH:=1024}"
: "${SEED:=1}"
# 1 enables wandb, anything else disables it.
: "${WANDB:=1}"
: "${WANDB_MODE:=offline}"
# Empty: STORE_DIR/wandb, or training/wandb.
: "${WANDB_DIR:=}"
# Empty: whatever the machine offers. cpu pins the battery and report.
: "${EVAL_PLATFORM:=}"
# Extra hydra overrides, word-split, applied to every rung.
: "${EXTRA:=}"

# Gate 3 compares vibration against BASELINE_VIBRATION_JSON and rung 1 compares
# tracking against BASELINE_MEAN_TRACK_ERR. Both baselines must have been
# measured on the platform selected here: warp is float32 with a fixed contact
# pool, CPU MuJoCo is float64 with none, so the two do not compare.
EVAL_ENV=()
if [ -n "$EVAL_PLATFORM" ]; then
    EVAL_ENV=(env "JAX_PLATFORMS=$EVAL_PLATFORM")
fi

# train.py resolves a relative restore= against the training project dir.
# Resolve START_CHECKPOINT the same way so the check matches the load.
if [[ "$START_CHECKPOINT" != /* ]]; then
    START_CHECKPOINT="$PROJECT/$START_CHECKPOINT"
fi
if [ ! -d "$START_CHECKPOINT" ]; then
    echo "ERROR: START_CHECKPOINT=$START_CHECKPOINT does not exist" >&2
    exit 1
fi

if [ "$WANDB" = "1" ]; then
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

# ------------------------------------------------------------------ gates

# Evaluates one rung's runs/<name>/eval_report.json against the 5 gates and
# writes a "VERDICT|mean_track_err_rms|max_saturation|worst_vib_ratio" line
# to $2 (the gate summary file). Exit code: 0 ACCEPTED, 1 REJECTED (gates
# 1-4 failed), 2 DIMINISHING (only gate 5 failed). $3 is the previous
# accepted rung's mean_track_err_rms (or the baseline's, for rung 1).
run_gates() {
    local report_json="$1" summary_out="$2" prev_mean="$3"
    python3 - "$report_json" "$summary_out" "$prev_mean" "$BASELINE_VIBRATION_JSON" <<'PYEOF'
import json
import sys

report_path, summary_out, prev_mean_s, baseline_vib_json = sys.argv[1:5]
prev_mean = float(prev_mean_s)
baseline_vibration = json.loads(baseline_vib_json)

SCENARIOS = ["stand_to_trot_ramp", "turn", "strafe", "walk_to_stop"]
VEL_ERR_SCENARIOS = ["stand_to_trot_ramp", "turn", "walk_to_stop"]

report = json.load(open(report_path))
battery = report["battery"]

reasons = []
ok_1_4 = True

# gate 1: no falls in any of the 4 scenarios
for sc in SCENARIOS:
    fell_at = battery[sc].get("fell_at")
    status = "PASS" if fell_at is None else "FAIL"
    if fell_at is not None:
        ok_1_4 = False
        reasons.append(f"gate1 fell: {sc} fell_at={fell_at}")
    print(f"gate1 fell_at[{sc}]={fell_at} [{status}]")

# gate 2: anti-stander-collapse velocity tracking
for sc in VEL_ERR_SCENARIOS:
    v = battery[sc].get("vel_err_overall")
    status = "PASS" if (v is not None and v < 0.20) else "FAIL"
    if status == "FAIL":
        ok_1_4 = False
        reasons.append(f"gate2 vel_err_overall: {sc}={v}")
    print(f"gate2 vel_err_overall[{sc}]={v} [{status}]")
vy_err = battery["strafe"].get("vy_err")
status = "PASS" if (vy_err is not None and vy_err < 0.20) else "FAIL"
if status == "FAIL":
    ok_1_4 = False
    reasons.append(f"gate2 vy_err: strafe={vy_err}")
print(f"gate2 vy_err[strafe]={vy_err} [{status}]")

# gate 3: vibration <= 1.3x the baseline reference, per scenario
worst_vib_ratio = 0.0
for sc in SCENARIOS:
    v = battery[sc].get("vibration")
    ref = baseline_vibration[sc]
    limit = 1.3 * ref
    ratio = (v / ref) if (v is not None and ref) else None
    if ratio is not None:
        worst_vib_ratio = max(worst_vib_ratio, ratio)
    status = "PASS" if (v is not None and v <= limit) else "FAIL"
    if status == "FAIL":
        ok_1_4 = False
        reasons.append(f"gate3 vibration: {sc}={v} > {limit:.4f} (1.3x{ref})")
    print(f"gate3 vibration[{sc}]={v} limit={limit:.4f} ratio={ratio} [{status}]")

# gate 4: torque saturation, max over all joint groups and scenarios
max_sat = 0.0
for sc in SCENARIOS:
    sat = battery[sc].get("saturation") or {}
    for joint, frac in sat.items():
        if frac is not None:
            max_sat = max(max_sat, frac)
status = "PASS" if max_sat < 0.05 else "FAIL"
if status == "FAIL":
    ok_1_4 = False
    reasons.append(f"gate4 saturation: max={max_sat} >= 0.05")
print(f"gate4 saturation max={max_sat} [{status}]")

# gate 5: diminishing returns -- mean track_err_rms must beat the previous
# accepted rung's mean by >=5%. Only evaluated if gates 1-4 already passed
# (a rejected rung's tracking numbers aren't a meaningful comparison).
errs = [battery[sc].get("track_err_rms") for sc in SCENARIOS]
mean_err = None
gate5_pass = False
if any(e is None for e in errs):
    print("gate5 mean_track_err_rms: missing data [FAIL]")
else:
    mean_err = sum(errs) / len(errs)
    threshold = prev_mean * 0.95
    gate5_pass = mean_err <= threshold
    status = "PASS" if gate5_pass else "FAIL"
    print(
        f"gate5 mean_track_err_rms={mean_err:.4f} prev={prev_mean:.4f} "
        f"threshold(-5%)={threshold:.4f} [{status}]"
    )

if not ok_1_4:
    verdict, rc = "REJECTED", 1
elif not gate5_pass:
    verdict, rc = "DIMINISHING", 2
else:
    verdict, rc = "ACCEPTED", 0

print(f"VERDICT={verdict}")
for r in reasons:
    print(f"  reason: {r}")

with open(summary_out, "w") as f:
    f.write(f"{verdict}|{mean_err if mean_err is not None else 'nan'}|{max_sat}|{worst_vib_ratio}\n")

sys.exit(rc)
PYEOF
}

# ------------------------------------------------------------------- loop

RUNGS=($RUNGS_LIST)
RESTORE="$START_CHECKPOINT"
PREV_MEAN="$BASELINE_MEAN_TRACK_ERR"

R_KP=() R_RUNDIR=() R_STATUS=() R_MEAN=() R_MAXSAT=() R_VIB=()
WINNER_KP=""
WINNER_RUNDIR=""
STOP_REASON=""
RUNG1_DONE=0
IDX=0

for kp in "${RUNGS[@]}"; do
    RUN_NAME="${RUN_PREFIX}${kp}_${STAMP}"
    RUN_DIR="$PROJECT/runs/$RUN_NAME"

    echo "== rung kp$kp: run_name=$RUN_NAME restore=$RESTORE =="

    # A training crash aborts the job here via set -e. That is the only
    # hard failure mode.
    # shellcheck disable=SC2086
    run_main python3 -m wojtek_rl.train \
        "+experiment=${PRESET_PREFIX}$kp" \
        "++ppo.num_envs=$NUM_ENVS" "++ppo.batch_size=$BATCH" \
        "seed=$SEED" "$WANDB_FLAG" "run_name=$RUN_NAME" "restore=$RESTORE" $EXTRA

    if ! run_main ${EVAL_ENV[@]+"${EVAL_ENV[@]}"} python3 -m wojtek_rl.battery --run "runs/$RUN_NAME"; then
        echo "WARN: battery crashed for $RUN_NAME; ladder stops here"
        R_KP+=("$kp"); R_RUNDIR+=("$RUN_DIR"); R_STATUS+=("BATTERY-CRASHED")
        R_MEAN+=("-"); R_MAXSAT+=("-"); R_VIB+=("-")
        STOP_REASON="battery crashed on rung kp$kp"
        break
    fi
    [ "$IDX" -eq 0 ] && RUNG1_DONE=1

    if ! run_main ${EVAL_ENV[@]+"${EVAL_ENV[@]}"} python3 -m wojtek_rl.report --run "runs/$RUN_NAME"; then
        echo "WARN: report crashed for $RUN_NAME; ladder stops here (no eval_report.json to gate on)"
        R_KP+=("$kp"); R_RUNDIR+=("$RUN_DIR"); R_STATUS+=("REPORT-CRASHED")
        R_MEAN+=("-"); R_MAXSAT+=("-"); R_VIB+=("-")
        STOP_REASON="report crashed on rung kp$kp"
        break
    fi

    GATE_SUMMARY="$RUN_DIR/gate_summary.txt"
    set +e
    run_gates "$RUN_DIR/eval_report.json" "$GATE_SUMMARY" "$PREV_MEAN"
    GATE_RC=$?
    set -e

    if [ ! -s "$GATE_SUMMARY" ]; then
        echo "WARN: gate evaluation crashed for $RUN_NAME (rc=$GATE_RC); ladder stops here"
        R_KP+=("$kp"); R_RUNDIR+=("$RUN_DIR"); R_STATUS+=("GATE-CRASHED")
        R_MEAN+=("-"); R_MAXSAT+=("-"); R_VIB+=("-")
        STOP_REASON="gate evaluation crashed on rung kp$kp"
        break
    fi

    IFS='|' read -r VERDICT MEAN_ERR MAX_SAT WORST_VIB < "$GATE_SUMMARY"
    R_KP+=("$kp"); R_RUNDIR+=("$RUN_DIR")
    R_MEAN+=("$MEAN_ERR"); R_MAXSAT+=("$MAX_SAT"); R_VIB+=("$WORST_VIB")

    if [ "$GATE_RC" -eq 0 ]; then
        R_STATUS+=("PASS")
        WINNER_KP="$kp"
        WINNER_RUNDIR="$RUN_DIR"
        RESTORE=$(ls -d "$RUN_DIR/checkpoints/"[0-9]* | sort | tail -1)
        PREV_MEAN="$MEAN_ERR"
    elif [ "$GATE_RC" -eq 2 ]; then
        R_STATUS+=("DIMINISHING")
        STOP_REASON="diminishing returns at rung kp$kp (mean_track_err_rms=$MEAN_ERR did not beat $PREV_MEAN by >=5%); winner stays the previous accepted rung"
        break
    else
        R_STATUS+=("REJECTED")
        STOP_REASON="rung kp$kp rejected on gates 1-4; winner stays the previous accepted rung"
        break
    fi

    IDX=$((IDX + 1))
done

# ---------------------------------------------------------------- summary

echo ""
echo "== LADDER SUMMARY =="
for kp in "${RUNGS[@]}"; do
    found=0
    for j in "${!R_KP[@]}"; do
        if [ "${R_KP[$j]}" = "$kp" ]; then
            printf "kp%-3s %-12s rundir=%-70s mean_track_err_rms=%-8s max_saturation=%-8s worst_vib_ratio=%s\n" \
                "$kp" "${R_STATUS[$j]}" "${R_RUNDIR[$j]}" "${R_MEAN[$j]}" "${R_MAXSAT[$j]}" "${R_VIB[$j]}"
            found=1
            break
        fi
    done
    [ "$found" -eq 0 ] && printf "kp%-3s %-12s (not started)\n" "$kp" "NOT-RUN"
done
if [ -n "$WINNER_RUNDIR" ]; then
    echo "WINNER: kp$WINNER_KP  $WINNER_RUNDIR"
else
    echo "WINNER: none (rung kp${RUNGS[0]} did not pass gates)"
fi
[ -n "$STOP_REASON" ] && echo "stopped: $STOP_REASON"
echo "== END LADDER SUMMARY =="

# A rejected or diminishing rung is a valid result, not a job failure.
if [ "$RUNG1_DONE" -ne 1 ]; then
    echo "FATAL: rung kp${RUNGS[0]} did not complete training+battery" >&2
    exit 1
fi
exit 0
