# Sourced by ../sim.sh and ../dev.sh, from this directory (ros/docker), with
# DOCKER_ENV already declared as an array. Not a script: no shebang, no
# set -e of its own.
#
# Personal values the launches read, carried into the container from the
# host environment or else the gitignored repo-root .env (see .env.example):
# one `-e VAR=value` appended to DOCKER_ENV for each that is set. `docker
# exec` passes no host environment on its own, which is why this exists.
# Read by name, not sourced, so nothing else in .env leaks into the
# container.
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
#   MUJOCO_GL                  the sim camera's GL backend when the default
#                              (egl on Linux) is not the one to use

WOJTEK_FORWARD_VARS="${WOJTEK_FORWARD_VARS:-HF_ORGANIZATION HF_TOKEN WOJTEK_POLICY VLM_URL VLM_MODEL ROS_LOCALHOST_ONLY MUJOCO_GL}"

forward_env() {
  local var val envfile
  for var in $WOJTEK_FORWARD_VARS; do
    if [ -z "${!var:-}" ]; then
      for envfile in ../../.env ../.env; do
        [ -f "$envfile" ] || continue
        # `|| true`: a key absent from the file is the normal case, not an
        # error for set -e/pipefail to kill the script on (it did, silently).
        val=$({ grep -E "^${var}=" "$envfile" || true; } | tail -1 | cut -d= -f2- | tr -d '"'"'")
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
    [ -n "${!var:-}" ] && DOCKER_ENV+=(-e "$var=${!var}")
  done
  return 0
}

forward_env
