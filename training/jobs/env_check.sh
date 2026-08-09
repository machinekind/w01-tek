#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source training/jobs/_lib.sh

# Pipeline smoke check. Proves that a dispatched job reaches a working
# environment: the venv answers, jax imports and sees the accelerators the
# job was sized for. It trains nothing and finishes in seconds, so it is
# the cheapest way to validate a dispatch path end to end.

: "${GPUS:=1}"
: "${ON_FAILURE:=any failed check ends the job non zero}"
# Seconds to stay alive after the checks pass, for exercising watchers.
: "${LINGER:=0}"
# 1: also prove the physics stack, a warp-backed MJX step on the device.
: "${CHECK_MJX:=0}"

visible="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
echo "driver sees $visible gpu(s), job is sized for $GPUS"
if [ "${visible:-0}" -lt "$GPUS" ]; then
    echo "ERROR: fewer GPUs visible than the job is sized for" >&2
    exit 1
fi

run_main python3 - "$GPUS" <<'PY'
import sys

print("python", sys.version.split()[0])
import jax

gpus = [d for d in jax.devices() if d.platform == "gpu"]
print("jax", jax.__version__, "sees", len(gpus), "gpu device(s)")
assert len(gpus) >= int(sys.argv[1]), "jax sees fewer GPUs than the job is sized for"
PY

if [ "$CHECK_MJX" = "1" ]; then
    run_main python3 - <<'PY'
import warp as wp

wp.init()
import jax
import mujoco
from mujoco import mjx

xml = "<mujoco><worldbody><body pos='0 0 1'><freejoint/><geom type='sphere' size='0.05'/></body></worldbody></mujoco>"
m = mujoco.MjModel.from_xml_string(xml)
mx = mjx.put_model(m)
dx = mjx.make_data(m)
dx = jax.jit(mjx.step)(mx, dx)
print("mjx step ok, t =", float(dx.time))
PY
fi

if [ "$LINGER" -gt 0 ]; then
    echo "lingering for ${LINGER}s"
    sleep "$LINGER"
fi
echo "env check passed"
