#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

case "${1:-}" in
  build) shift; "$PY" -m fbb_rl.build_model "$@" ;;
  pose)  shift; "$PY" -m fbb_rl.pose_explorer "$@" ;;
  check) shift; "$PY" -m fbb_rl.check_model_mjx "$@" ;;
  train) shift; "$PY" -m fbb_rl.train "$@" ;;
  smoke) shift; JAX_PLATFORMS=cpu "$PY" -m fbb_rl.train smoke=true wandb.enable=false "$@" ;;
  eval)  shift; MUJOCO_GL="${MUJOCO_GL:-egl}" "$PY" -m fbb_rl.eval "$@" ;;
  battery) shift; JAX_PLATFORMS=cpu "$PY" -m fbb_rl.battery "$@" ;;
  export) shift; JAX_PLATFORMS=cpu "$PY" -m fbb_rl.export_policy "$@" ;;
  app)   shift; MUJOCO_GL="${MUJOCO_GL:-egl}" "$PY" -m demo.app "$@" ;;
  test)  shift; "$PY" -m pytest tests -q "$@" ;;
  *) echo "usage: run.sh {build|pose|check|train|smoke|eval|app|test} [args]"; exit 1 ;;
esac
