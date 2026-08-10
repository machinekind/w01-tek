#!/bin/bash
# Multi-run course benchmark + flat battery in one job. Eval only, no
# training. Both tools default to the CPU path: courses are
# CPU-deterministic and the course noise bands are defined there.
#
# Output isolation: several of these jobs may run in parallel with the same
# run as a member of each, and the tools default their outputs INTO the
# evaluated run dir, so concurrent jobs would clobber courses.json and
# battery.json. Every invocation here passes an explicit --out under
# runs/<run>/<OUT_SUB>/ and nothing writes to a default path. No --video or
# --paths either: those are the only flags that write the courses artifacts
# dir.
#
# Skip-missing: before any eval, each member is tested for a completed
# export marker, deploy/.export_ok or deploy/policy_meta.json as the
# fallback for runs exported by hand. A member without one is logged
# SKIPPED and the job proceeds with the rest. The KEEPER member is exempt:
# it trained earlier and may have no deploy/ on this checkout, so it is
# checked for run.json + a checkpoint instead.
#
# Provenance: runs/<run>/<OUT_SUB>/provenance.json per present member --
# checkpoint dir name, checkpoint sha256, the dispatcher's job id, the
# run.json seed, and the SKIPPED list -- written BEFORE the evals so a
# timeout cannot lose it.
set -euo pipefail
cd "$(dirname "$0")/../.."
source training/jobs/_lib.sh

: "${GPUS:=1}"
: "${ON_FAILURE:=a member without an export marker is skipped, a crashed member eval logs a WARN and the loop continues, and the summary always prints}"
# Space-separated run names under training/runs; a runs/ prefix on an entry
# is accepted and stripped.
: "${RUNS:?space-separated run names under training/runs}"
# Eval tag; outputs land in runs/<run>/<OUT_SUB>/.
: "${TAG:?eval tag, names the output subdirectory}"
# Member exempt from the export marker.
: "${KEEPER:=wojtek_terrain_blind_v4_2}"
# Course rollouts per scenario.
: "${SEEDS:=8}"
# courses --seed-base offset; a replicate pass uses a disjoint band.
: "${SEED_BASE:=0}"
# Per-run output subdir, relative to the run dir.
: "${OUT_SUB:=eval/$TAG}"
# JAX platform for the evals. Empty: whatever devices the machine has.
: "${EVAL_PLATFORM:=cpu}"
# Run name whose policy acts on pure-spin and stand command windows:
# every member is then measured as the seam-test composite (--spin-run,
# see wojtek_rl.seam). Empty: plain single-policy evals.
: "${SPIN_RUN:=}"

PLATFORM_ENV=()
[ -n "$EVAL_PLATFORM" ] && PLATFORM_ENV=(JAX_PLATFORMS="$EVAL_PLATFORM")

# Accept "runs/name" and bare "name" entries alike.
MEMBERS=""
# shellcheck disable=SC2086
for entry in $RUNS; do
    MEMBERS="$MEMBERS ${entry#runs/}"
done
MEMBERS="${MEMBERS# }"

member_present() {
    local name="$1"
    local dir="$PROJECT/runs/$name"
    if [ "$name" = "$KEEPER" ]; then
        # The keeper predates the export marker; run.json + a checkpoint
        # is its presence test.
        [ -f "$dir/run.json" ] || return 1
        local d
        for d in "$dir/checkpoints"/[0-9]*; do
            [ -d "$d" ] && return 0
        done
        return 1
    fi
    [ -f "$dir/deploy/.export_ok" ] && return 0
    [ -f "$dir/deploy/policy_meta.json" ] && return 0
    return 1
}

# Which checkpoint this job actually reads (the tools pick the
# highest-numbered dir -- mirror that), hashed so a report can assert on it.
write_provenance() {
    local name="$1" out_dir="$2"
    local dir="$PROJECT/runs/$name"
    local ckpt="" sha="" d b
    for d in "$dir/checkpoints"/[0-9]*; do
        [ -d "$d" ] || continue
        b="${d##*/}"
        case "$b" in *[!0-9]*) continue ;; esac
        if [ -z "$ckpt" ] || [ "$b" -gt "$ckpt" ]; then ckpt="$b"; fi
    done
    if [ -n "$ckpt" ]; then
        sha=$(cd "$dir/checkpoints/$ckpt" \
              && find . -type f -print0 | LC_ALL=C sort -z \
                 | xargs -0 sha256sum | sha256sum | awk '{print $1}')
    fi
    P_RUN="$name" P_CKPT="$ckpt" P_SHA="$sha" P_OUT="$out_dir" P_TAG="$TAG" \
    P_MEMBERS="$MEMBERS" P_SKIPPED="$SKIPPED" P_JOB="${JOB_ID:-}" \
    P_DIR="$dir" python3 - <<'PY'
import datetime
import json
import os

env = os.environ
seed = None
try:
    with open(os.path.join(env["P_DIR"], "run.json")) as f:
        seed = json.load(f).get("hydra_config", {}).get("seed")
except Exception as e:  # unreadable run.json is itself a finding
    print("WARN: run.json seed unreadable:", e)
rec = {
    "tag": env["P_TAG"],
    "run": env["P_RUN"],
    "checkpoint": env["P_CKPT"] or None,
    "checkpoint_sha256": env["P_SHA"] or None,
    "job_id": env["P_JOB"] or None,
    "run_json_seed": seed,
    "members": env["P_MEMBERS"].split(),
    "skipped": env["P_SKIPPED"].split(),
    "written": datetime.datetime.now().isoformat(timespec="seconds"),
}
path = os.path.join(env["P_OUT"], "provenance.json")
with open(path, "w") as f:
    json.dump(rec, f, indent=2)
    f.write("\n")
print("provenance:", path)
PY
}

# Markers first: SKIPPED is final before anything runs.
SKIPPED=""
PRESENT=""
for name in $MEMBERS; do
    if member_present "$name"; then
        PRESENT="$PRESENT $name"
    else
        echo "SKIPPED: runs/$name -- no completed export marker; proceeding without it"
        SKIPPED="$SKIPPED $name"
    fi
done
SKIPPED="${SKIPPED# }"
PRESENT="${PRESENT# }"
if [ -z "$PRESENT" ]; then
    echo "ERROR: no member present -- nothing to evaluate" >&2
    exit 1
fi

# The spin donor is load-bearing for every member: without it the whole
# job would silently measure something else, so a missing donor is a hard
# error, not a skip. Checked keeper-style: it may predate the export
# marker.
SPIN_FLAGS=()
if [ -n "$SPIN_RUN" ]; then
    if [ ! -f "$PROJECT/runs/$SPIN_RUN/run.json" ]; then
        echo "ERROR: SPIN_RUN runs/$SPIN_RUN has no run.json" >&2
        exit 1
    fi
    found=""
    for d in "$PROJECT/runs/$SPIN_RUN/checkpoints"/[0-9]*; do
        [ -d "$d" ] && found=1 && break
    done
    if [ -z "$found" ]; then
        echo "ERROR: SPIN_RUN runs/$SPIN_RUN has no checkpoint" >&2
        exit 1
    fi
    SPIN_FLAGS=(--spin-run "runs/$SPIN_RUN")
fi

# Provenance before the evals: a timeout mid-battery must not lose it.
for name in $PRESENT; do
    mkdir -p "$PROJECT/runs/$name/$OUT_SUB"
    write_provenance "$name" "$PROJECT/runs/$name/$OUT_SUB"
done

FAILED=""
for name in $PRESENT; do
    echo "== courses tag=$TAG run=$name seeds=$SEEDS seed-base=$SEED_BASE =="
    if ! run_main env ${PLATFORM_ENV[@]+"${PLATFORM_ENV[@]}"} python3 -m wojtek_rl.courses \
        --run "runs/$name" --seeds "$SEEDS" --seed-base "$SEED_BASE" \
        --out "runs/$name/$OUT_SUB/courses.json" \
        ${SPIN_FLAGS[@]+"${SPIN_FLAGS[@]}"}; then
        echo "WARN: courses for $name crashed; continuing"
        FAILED="$FAILED $name:courses"
    fi
    echo "== battery tag=$TAG run=$name =="
    if ! run_main env ${PLATFORM_ENV[@]+"${PLATFORM_ENV[@]}"} python3 -m wojtek_rl.battery \
        --run "runs/$name" --out "runs/$name/$OUT_SUB/battery.json" \
        ${SPIN_FLAGS[@]+"${SPIN_FLAGS[@]}"}; then
        echo "WARN: battery for $name crashed; continuing"
        FAILED="$FAILED $name:battery"
    fi
done
FAILED="${FAILED# }"

echo ""
echo "== COURSES+BATTERY $TAG SUMMARY: members [$MEMBERS]," \
     "skipped [${SKIPPED:-none}], crashed [${FAILED:-none}] =="
exit 0
