#!/usr/bin/env bash
# Test entry point for the experimental agentic stack.  Everything here is
# model-free: no GPU, no ROS runtime, no env instantiation, so it runs on any
# machine in seconds.  See README.md for what this experiment is.
set -euo pipefail
cd "$(dirname "$0")"

# The training venv is the interpreter that has numpy and pytest.  A worktree
# has no venv, so allow an explicit interpreter and fall back to python3:
#   EXP_PY=/path/to/python ./run.sh test
PY="${EXP_PY:-../../training/.venv/bin/python}"
[ -x "$PY" ] || command -v "$PY" >/dev/null 2>&1 || PY="python3"

case "${1:-}" in
  test)
    shift
    echo ">> wojtek_agent (python layer)"
    PYTHONPATH=".:../../training${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY" -m pytest tests -q "$@"
    echo ">> ROS node logic (rclpy-free modules)"
    PYTHONPATH="ros/src/wojtek_brain:ros/src/wojtek_voice${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY" -m pytest ros/src/wojtek_brain/test ros/src/wojtek_voice/test -q "$@"
    ;;
  *)
    echo "usage: run.sh test [pytest args]"
    exit 1
    ;;
esac
