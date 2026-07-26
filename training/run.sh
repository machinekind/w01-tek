#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
# Headless-render default for eval/app: egl exists only on Linux; macOS
# must stay unset so mujoco picks its own default (cgl).
if [[ "$(uname)" == "Linux" ]]; then export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi

case "${1:-}" in
  build) shift; "$PY" -m wojtek_rl.build_model "$@" ;;
  build-terrain) shift; "$PY" -m wojtek_rl.build_terrain "$@" ;;
  pose)  shift; "$PY" -m wojtek_rl.pose_explorer "$@" ;;
  check) shift; "$PY" -m wojtek_rl.check_model_mjx "$@" ;;
  check-terrain) shift; "$PY" -m wojtek_rl.check_terrain "$@" ;;
  train) shift; "$PY" -m wojtek_rl.train "$@" ;;
  smoke) shift; JAX_PLATFORMS=cpu "$PY" -m wojtek_rl.train smoke=true wandb.enable=false "$@" ;;
  eval)  shift; "$PY" -m wojtek_rl.eval "$@" ;;
  battery) shift; JAX_PLATFORMS=cpu "$PY" -m wojtek_rl.battery "$@" ;;
  courses) shift; JAX_PLATFORMS=cpu "$PY" -m wojtek_rl.courses "$@" ;;  # path-following benchmark
  report) shift; JAX_PLATFORMS=cpu "$PY" -m wojtek_rl.report "$@" ;;
  export) shift; JAX_PLATFORMS=cpu "$PY" -m wojtek_rl.export_policy "$@" ;;
  sysid) shift; "$PY" -m wojtek_rl.sysid "$@" ;;  # engine params from rosbags, docs/sysid.md
  app)   shift; MUJOCO_GL="${MUJOCO_GL:-egl}" "$PY" -m demo.app "$@" ;;
  room-assets) shift; "$PY" -m wojtek_rl.room_assets "$@" ;;
  build-room)  shift; "$PY" -m wojtek_rl.build_room "$@" ;;
  room)  shift; "$PY" -m wojtek_rl.room_app "$@" ;;  # GL backend picked per-OS in room_app
  grid)  shift; "$PY" -m wojtek_eval.gridmap "$@" ;;
  nav-eval) shift; MUJOCO_GL="${MUJOCO_GL:-$([ "$(uname)" = Linux ] && echo egl || echo cgl)}" "$PY" -m wojtek_eval.runner "$@" ;;
  nav-episode) shift; MUJOCO_GL="${MUJOCO_GL:-$([ "$(uname)" = Linux ] && echo egl || echo cgl)}" "$PY" -m wojtek_rl.nav_episode "$@" ;;  # headless VLM goal runner
  # tests/unit is model-free by construction: no env instantiation, no
  # mjx.put_model, no MjModel.from_xml_* -- so the whole set is seconds, and
  # `test` stays usable in an edit loop. tests/integration builds and steps
  # real MJX models; those cost tens of seconds each even warm.
  test)  shift; "$PY" -m pytest tests/unit -q "$@" ;;
  test-slow) shift; JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax_cache}" \
             JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
             "$PY" -m pytest tests/integration -q "$@" ;;
  test-all)  shift; JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax_cache}" \
             JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
             "$PY" -m pytest tests -q "$@" ;;
  *) echo "usage: run.sh {build|pose|check|train|smoke|eval|courses|app|test|test-slow|test-all} [args]"; exit 1 ;;
esac
