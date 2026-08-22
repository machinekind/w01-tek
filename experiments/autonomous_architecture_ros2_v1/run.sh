#!/usr/bin/env bash
# Entry point for the experimental agentic stack.  Self-contained on purpose:
# nothing outside this directory is configured to know about the experiment, so
# every path it needs is set up here.  See README.md.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
TRAINING="$(cd ../../training && pwd)"

# The training venv is the interpreter that has numpy/pytest/mujoco.  A
# worktree has no venv, so allow an explicit interpreter:
#   EXP_PY=/path/to/python ./run.sh test
PY="${EXP_PY:-$TRAINING/.venv/bin/python}"
[ -x "$PY" ] || command -v "$PY" >/dev/null 2>&1 || PY="python3"

# The experiment imports the training project (sim, policy runtime, planner);
# never the other way round.
export PYTHONPATH="$HERE:$TRAINING${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-}" in
  test)
    shift
    echo ">> wojtek_agent (python layer)"
    "$PY" -m pytest tests -q "$@"
    echo ">> ROS node logic (rclpy-free modules)"
    PYTHONPATH="$HERE/ros/src/wojtek_brain:$HERE/ros/src/wojtek_voice:$HERE/ros/src/wojtek_futurenav_bridge:$PYTHONPATH" \
      "$PY" -m pytest ros/src/wojtek_brain/test ros/src/wojtek_voice/test \
        ros/src/wojtek_futurenav_bridge/test -q "$@"
    ;;
  room)
    # The agent demo: MuJoCo room sim + walking policy + chat agent + voice.
    # This is the experiment's fork of wojtek_rl.room_app; the training
    # project's own room demo is unchanged and still runs via
    # ./training/run.sh room.  Scene via SCENE=castle|flat|room|apartment.
    shift
    cd "$TRAINING"   # the sim resolves scenes/policies relative to the project
    exec "$PY" -m wojtek_demo.room_app "$@"
    ;;
  build)
    # colcon build for this experiment's ROS packages only.  They live outside
    # ros/src so that ros/deploy.sh (which rsyncs ros/src to the robot) can
    # never ship them; that also means ros/sim.sh does not build them, hence
    # this target.  Run it in an environment that has ROS 2 sourced.
    shift
    command -v colcon >/dev/null 2>&1 || { echo "colcon not found: source a ROS 2 setup.bash first" >&2; exit 1; }
    exec colcon build --base-paths ros/src "$@"
    ;;
  *)
    echo "usage: run.sh {test|room|build} [args]"
    echo "  test   model-free unit tests (python layer + ROS node logic)"
    echo "  room   the interactive agent demo (this experiment's fork)"
    echo "  build  colcon build of this experiment's ROS packages"
    exit 1
    ;;
esac
