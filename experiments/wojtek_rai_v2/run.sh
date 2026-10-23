#!/usr/bin/env bash
# Entry point for the RAI-on-Wojtek experiment. Self-contained on purpose:
# nothing outside this directory is configured to know about it. See README.md.
#
# Everything runs in the wojtek_rai container (docker/compose.yaml), a sibling
# of wojtek_robot on the same host network and DDS domain. Start the simulation
# first, from the repository root:  ./ros/sim.sh
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
CONTAINER=wojtek_rai

# Credentials enter only through the gitignored repo-root .env (.env.example is
# the committed template). Sourced here, never echoed; forwarded into the
# container BY NAME (`docker exec -e NAME`, no value), and only when set.
# Never enable a shell trace in this script.
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env"
  set +a
fi
CRED_VARS=(
  OPENAI_API_KEY
  AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
  GOOGLE_API_KEY
  LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY
)
# Every WOJTEK_RAI_* switch (READONLY=1 strips the movement tools, ODOMETRY=0
# drops the closed-loop turn, COLOR_TRANSPORT=compressed reads the JPEG colour
# stream over WiFi, PLACES / WORKSPACE_M bound navigation) is forwarded like
# the credentials, by name. WOJTEK_RAI_INFERENCE_SSH is host-side only.
CRED_ENV_ARGS=()
for name in "${CRED_VARS[@]}" "${!WOJTEK_RAI_@}"; do
  [ "$name" = WOJTEK_RAI_INFERENCE_SSH ] && continue
  if [ -n "${!name:-}" ]; then
    CRED_ENV_ARGS+=(-e "$name")
  fi
done

usage() {
  cat >&2 <<'USAGE'
usage: run.sh {build|up|down|shell|topics|agent|chat|bench|tunnel|inference|pull|test|nav ...|perception ...} [args]
  build    build the wojtek_rai image (once per machine; slow the first time)
  up       start the wojtek_rai container (idempotent)
  down     stop and remove it
  shell    interactive shell inside it (ROS 2 + rai_interfaces sourced)
  topics   print the live ROS 2 graph the simulation exposes (step 1 smoke)
  agent    the chat agent: streamlit on http://localhost:8501
  chat     one-shot terminal chat: run.sh chat [--debug] "walk forward for 2 seconds"
  bench    time one ReAct step of the LLM (prompt tokens, TTFT, tok/s):
           run.sh bench [--model M] [--images N] [--no-tools] [--history-steps N] [--runs N]
  tunnel   SSH tunnel localhost:11435 -> Ollama on the inference box
           (WOJTEK_RAI_INFERENCE_SSH=<ssh host alias> in the root .env)
  inference  start Ollama on the inference box (user-space install in ~/ollama)
  pull <model>  pull a model on the inference box (e.g. qwen3-vl:30b-a3b-instruct)
  test     model-free unit tests (EXP_PY=<host python> to run outside docker)
  nav build|up|down|shell|launch [sim]|exec <cmd>
           the navigation container: Nav2 + slam_toolbox + depth->scan
           (launch = start the whole nav stack in the foreground)
  perception build|up|down|logs|shell
           GroundingDINO (/detection) + SAM2 (/segmentation) on the laptop GPU
USAGE
}

compose() {
  (cd "$HERE/docker" && docker compose "$@")
}

up() {
  compose up -d wojtek_rai
}

# Runs a command inside the container through the image entrypoint, which
# sources ROS 2 and the rai_interfaces overlay and cds into /exp. Credentials
# forwarded by name only (see CRED_ENV_ARGS above).
cexec() {
  docker exec -i ${CRED_ENV_ARGS[@]+"${CRED_ENV_ARGS[@]}"} "$CONTAINER" /entrypoint.sh "$@"
}

case "${1:-}" in
  build)
    shift
    # Only the agent image, as the usage text says: nav and perception have
    # their own `build` subcommands (perception alone is ~8 GB).
    compose build wojtek_rai "$@"
    ;;
  up)
    up
    ;;
  down)
    compose down
    ;;
  shell)
    up
    exec docker exec -it ${CRED_ENV_ARGS[@]+"${CRED_ENV_ARGS[@]}"} "$CONTAINER" /entrypoint.sh bash
    ;;
  topics)
    shift
    up
    cexec python3 -m wojtek_rai.topics "$@"
    ;;
  agent)
    shift
    up
    # -i only (no -t): works from a plain terminal and from a background job.
    exec docker exec -i ${CRED_ENV_ARGS[@]+"${CRED_ENV_ARGS[@]}"} "$CONTAINER" /entrypoint.sh \
      streamlit run --server.headless=true --server.showEmailPrompt=false \
      --server.port=8501 wojtek_rai/app.py "$@"
    ;;
  chat)
    shift
    up
    cexec python3 -m wojtek_rai.chat "$@"
    ;;
  bench)
    shift
    up
    cexec python3 -m wojtek_rai.bench_llm "$@"
    ;;
  inference)
    # Ollama as a user-space tarball on the box (no sudo): ~/ollama/bin/ollama
    # serve, bound to loopback, reached only through `tunnel`. Idempotent.
    # Measured on the box with Ollama 0.34 (bench: `run.sh bench`), the env line worth
    # running is (kill the old `ollama serve` first; this check skips the relaunch):
    #   OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=2h OLLAMA_CONTEXT_LENGTH=32768 \
    #     OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=4 OLLAMA_FLASH_ATTENTION=1 \
    #     nohup ~/ollama/bin/ollama serve >> ~/ollama-logs/serve.log 2>&1 &
    # CONTEXT_LENGTH=32768 matches the agent's num_ctx so a client that sends no
    # num_ctx no longer forces a ~17 s runner reload (+3 s full prompt re-eval);
    # FA is auto-on already (documented), NUM_PARALLEL=2 and KV_CACHE_TYPE=q8_0 gave
    # nothing (q8_0 was ~2% slower). The step time is thinking tokens: use the
    # instruct tag (qwen3-vl:30b-a3b-instruct), not the bare qwen3-vl:30b (a thinking build).
    : "${WOJTEK_RAI_INFERENCE_SSH:?set WOJTEK_RAI_INFERENCE_SSH=<ssh alias> in the root .env}"
    exec ssh "$WOJTEK_RAI_INFERENCE_SSH" 'set -e
      mkdir -p ~/ollama-logs
      if curl -fs -m 3 http://127.0.0.1:11434/api/version >/dev/null; then
        echo ">> ollama already serving"; exit 0; fi
      OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=2h \
        nohup ~/ollama/bin/ollama serve >> ~/ollama-logs/serve.log 2>&1 &
      for i in $(seq 1 20); do
        curl -fs -m 2 http://127.0.0.1:11434/api/version && echo && exit 0; sleep 1; done
      echo "!! ollama did not come up; see ~/ollama-logs/serve.log" >&2; exit 1'
    ;;
  pull)
    shift
    : "${WOJTEK_RAI_INFERENCE_SSH:?set WOJTEK_RAI_INFERENCE_SSH=<ssh alias> in the root .env}"
    [ $# -ge 1 ] || { echo "usage: run.sh pull <model>" >&2; exit 1; }
    exec ssh "$WOJTEK_RAI_INFERENCE_SSH" "OLLAMA_HOST=127.0.0.1:11434 ~/ollama/bin/ollama pull $(printf '%q' "$1")"
    ;;
  tunnel)
    # The inference box is private infrastructure: its ssh alias comes from
    # the gitignored .env, never from this script. Ollama listens on 11434
    # there; 11435 here leaves room for a local Ollama on the default port.
    : "${WOJTEK_RAI_INFERENCE_SSH:?set WOJTEK_RAI_INFERENCE_SSH=<ssh alias> in the root .env}"
    echo ">> tunnel localhost:11435 -> ${WOJTEK_RAI_INFERENCE_SSH}:11434 (Ctrl-C to close)"
    # The link to the box flaps; reconnect until Ctrl-C.
    while true; do
      ssh -N -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes -o ConnectTimeout=20 \
        -L 11435:127.0.0.1:11434 "$WOJTEK_RAI_INFERENCE_SSH" || true
      echo ">> tunnel dropped; reconnecting in 5 s" >&2
      sleep 5
    done
    ;;
  nav)
    shift
    NAV=wojtek_nav
    case "${1:-}" in
      build) shift; compose build wojtek_nav "$@" ;;
      up) compose up -d wojtek_nav ;;
      down) compose stop wojtek_nav && compose rm -f wojtek_nav ;;
      shell) compose up -d wojtek_nav; exec docker exec -it "$NAV" /entrypoint.sh bash ;;
      launch)
        shift
        compose up -d wojtek_nav
        exec docker exec -i "$NAV" /entrypoint.sh \
          ros2 launch wojtek_rai/nav/launch/nav.launch.py "$@"
        ;;
      exec) shift; compose up -d wojtek_nav; exec docker exec -i "$NAV" /entrypoint.sh "$@" ;;
      *) echo "usage: run.sh nav {build|up|down|shell|launch [args]|exec <cmd>}" >&2; exit 1 ;;
    esac
    ;;
  perception)
    shift
    case "${1:-}" in
      build) shift; compose build wojtek_perception "$@" ;;
      up) compose up -d wojtek_perception ;;
      down) compose stop wojtek_perception && compose rm -f wojtek_perception ;;
      logs) shift; compose logs -f --tail=100 wojtek_perception ;;
      shell) compose up -d wojtek_perception; exec docker exec -it wojtek_perception /entrypoint.sh bash ;;
      *) echo "usage: run.sh perception {build|up|down|logs|shell}" >&2; exit 1 ;;
    esac
    ;;
  test)
    shift
    if [ -n "${EXP_PY:-}" ]; then
      exec "$EXP_PY" -m pytest tests -q "$@"
    fi
    up
    # The venv sees the system site-packages, where ros-dev-tools' apt pytest
    # plugins (launch_testing) were built against an older pytest and break
    # entry-point autoload; load only the plugin this suite needs.
    cexec env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p pytest_timeout tests -q "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac
