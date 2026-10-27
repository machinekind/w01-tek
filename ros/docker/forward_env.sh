# Sourced by ../sim.sh and ../dev.sh, from this directory (ros/docker), with
# DOCKER_ENV already declared as an array. Not a script: no shebang, no
# set -e of its own.
#
# Personal values the launches read, carried into the container from the
# host environment or else the gitignored repo-root .env (see .env.example).
# Each one that is set is exported here and gets a bare `-e VAR` in
# DOCKER_ENV, with no value after it. docker exec then reads the value from
# its own environment, so a token never shows up in the process list (ps)
# while the session runs. `docker exec` passes no host environment on its
# own, which is why this exists. Read by name, not sourced, so nothing else
# in .env leaks into the container.
#
#   HF_ORGANIZATION, HF_TOKEN  the pinned default policy's org and the token
#                              that downloads it
#   WOJTEK_POLICY              a policy over the pin for the sim launch (the
#                              robot's launch never reads it): a Hugging Face
#                              reference only (org/name[@rev]); the training
#                              tools put paths in the same variable, and the
#                              container cannot see host paths, so those are
#                              dropped
#   VLM_URL, VLM_MODEL         the VLM brain's model server (Ollama)
#   ROS_LOCALHOST_ONLY         1 on a host whose Docker multicast loopback is
#                              broken (a headless GPU box: `ros2 node list`
#                              hangs, nodes in one container never find each
#                              other); never for a session that must reach
#                              the robot over the network
#   MUJOCO_GL                  the sim camera's GL backend inside the Linux
#                              container, when egl is not the one to use.
#                              Only the names that work on Linux go through
#                              (egl, osmesa, glx, glfw, disabled); a Mac's
#                              cgl, or any other name, is dropped with a
#                              message and the container keeps egl

WOJTEK_FORWARD_VARS="${WOJTEK_FORWARD_VARS:-HF_ORGANIZATION HF_TOKEN WOJTEK_POLICY VLM_URL VLM_MODEL ROS_LOCALHOST_ONLY MUJOCO_GL}"

forward_env() {
  local var val envfile
  for var in $WOJTEK_FORWARD_VARS; do
    if [ -z "${!var:-}" ]; then
      for envfile in ../../.env ../.env; do
        [ -f "$envfile" ] || continue
        # `|| true`: a key absent from the file is the normal case, not an
        # error for set -e/pipefail to kill the script on (it did, silently).
        # A .env saved on Windows ends every line with a carriage return, and
        # an editor can leave spaces after a value. Both are cut off here.
        # Otherwise ROS_LOCALHOST_ONLY=1 would arrive as "1\r" and not count,
        # and a key left empty would arrive as a lone "\r" and look set.
        val=$({ grep -E "^${var}=" "$envfile" || true; } | tail -1 | cut -d= -f2- | tr -d '\r"'"'" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
        if [ -n "$val" ]; then export "$var=$val"; break; fi
      done
    fi
    if [ "$var" = WOJTEK_POLICY ] && [ -n "${WOJTEK_POLICY:-}" ]; then
      # org/name with an optional @revision and nothing else: a relative
      # path such as runs/x/export has two slashes, an absolute one starts
      # with /, a bare directory name has none.
      if ! [[ "$WOJTEK_POLICY" =~ ^[A-Za-z0-9_-][A-Za-z0-9._-]*/[A-Za-z0-9_-][A-Za-z0-9._-]*(@[^/[:space:]]+)?$ ]]; then
        echo ">> WOJTEK_POLICY=${WOJTEK_POLICY} is a path, not org/name[@rev] -- not forwarded; the launch runs the pin (or pass policy:=)"
        unset WOJTEK_POLICY
      fi
    fi
    if [ "$var" = MUJOCO_GL ] && [ -n "${MUJOCO_GL:-}" ]; then
      # The container is Linux whatever the host runs. A Mac shell may export
      # cgl for its own renders, and MuJoCo on Linux refuses that name at
      # import, which takes the sim camera down. Only the names MuJoCo takes
      # on Linux go through; anything else stays behind and the container
      # keeps its egl default.
      case "$(printf '%s' "$MUJOCO_GL" | tr '[:upper:]' '[:lower:]')" in
        egl|osmesa|glx|glfw|disabled) ;;
        *)
          echo ">> MUJOCO_GL=${MUJOCO_GL} does not work in the Linux container -- not forwarded; the sim camera uses egl"
          unset MUJOCO_GL
          ;;
      esac
    fi
    # A bare `-e VAR`: docker exec takes the value from its own environment,
    # so it never sits in the docker command line where ps can show it. The
    # export makes sure that environment has it.
    if [ -n "${!var:-}" ]; then export "$var"; DOCKER_ENV+=(-e "$var"); fi
  done
  return 0
}

forward_env
