#!/usr/bin/env bash
# The model server behind the VLM brain: vLLM serving Qwen3-VL-8B-Instruct
# on a GPU box, OpenAI-compatible, with the JSON-schema structured output
# the brain holds the model to.
#
#   ./serve_vlm.sh                  # docker, foreground, port 8000 (Ctrl-C stops it)
#   ./serve_vlm.sh --detach         # docker, in the background, survives logout
#   ./serve_vlm.sh --logs           # follow the detached server's log
#   ./serve_vlm.sh --stop           # stop and remove the detached server
#   ./serve_vlm.sh --bare           # a vllm already installed in this Python
#   ./serve_vlm.sh --check          # is a server up? (curl /v1/models)
#
# Nothing here names a machine. Run it ON the GPU box; the PC then points
# the brain at it through VLM_URL in the repo-root .env (see .env.example):
#   VLM_URL=http://<that box>:8000
# and `./sim.sh ... vlm:=true` or `ros2 run wojtek_bringup robot --vlm`.
# A box reachable only over ssh gets a tunnel from the PC instead:
#   ssh -f -N -L 8000:localhost:8000 <box>     and  VLM_URL=http://host.docker.internal:8000
# (the dev container sees the PC's own ports under that name).
#
# Knobs, all environment variables with defaults:
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"   # or ...-8B-Instruct-FP8 for the A/B
VLM_PORT="${VLM_PORT:-8000}"
VLM_GPUS="${VLM_GPUS:-all}"           # docker --gpus value: all, '"device=0"', ...
VLM_MAX_LEN="${VLM_MAX_LEN:-8192}"    # one 1280x720 picture is ~1.5k tokens; 8k is plenty
VLM_GPU_MEM="${VLM_GPU_MEM:-0.6}"     # fraction of the GPU memory vLLM may take: the 8B in
                                      # bf16 is ~17 GB, so 0.6 of anything from 32 GB up is
                                      # plenty, and on a unified-memory box (DGX Spark) the
                                      # rest stays with the OS
# The image: vllm/vllm-openai on x86; on an arm64 box (DGX Spark / GB10)
# NVIDIA's build, e.g. nvcr.io/nvidia/vllm:26.08-py3. Either way the
# container runs `vllm serve` as its entrypoint below, so the two images'
# different default entrypoints do not matter.
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"   # Qwen3-VL needs vLLM >= 0.11
VLM_NAME="${VLM_NAME:-wojtek-vlm}"    # the detached container's name
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"        # weights cache, shared with docker
VLLM_API_KEY="${VLLM_API_KEY:-}"      # empty = no key (the brain then sends "EMPTY")
# HF_TOKEN, if set, is passed through for a gated checkout.
set -euo pipefail

MODE=docker
case "${1:-}" in
  --bare) MODE=bare ;;
  --check) MODE=check ;;
  --detach) MODE=detach ;;
  --logs) exec docker logs -f "$VLM_NAME" ;;
  --stop) docker rm -f "$VLM_NAME" && echo ">> $VLM_NAME stopped"; exit 0 ;;
  "") ;;
  *) echo "usage: $0 [--detach|--logs|--stop|--bare|--check]" >&2; exit 2 ;;
esac

if [ "$MODE" = check ]; then
  url="${VLM_URL:-http://127.0.0.1:${VLM_PORT}}"
  url="${url%/}"; case "$url" in */v1) ;; *) url="$url/v1" ;; esac
  echo ">> $url/models"
  curl -sf -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" "$url/models" \
    && echo && echo ">> up" || { echo ">> no server at $url" >&2; exit 1; }
  exit 0
fi

# The serve arguments, one list for both modes. Structured output: vLLM's
# default guided-decoding backend handles response_format json_schema
# (what the brain sends); nothing to enable. One image per request is the
# brain's shape, and the cap keeps the profiler from reserving room for
# more. Reasoning is off: the instruct model answers a short JSON, the
# latency the brain measured (1.1-1.5 s a call) depends on it staying short.
ARGS=(
  --port "$VLM_PORT"
  --host 0.0.0.0
  --max-model-len "$VLM_MAX_LEN"
  --gpu-memory-utilization "$VLM_GPU_MEM"
  --limit-mm-per-prompt '{"image":1}'
  --max-num-seqs 4
)
[ -n "$VLLM_API_KEY" ] && ARGS+=(--api-key "$VLLM_API_KEY")

echo ">> serving $VLM_MODEL on port $VLM_PORT ($MODE)"
echo ">> then, on the PC:  VLM_URL=http://<this box>:$VLM_PORT  in ros/.env"

if [ "$MODE" = bare ]; then
  exec vllm serve "$VLM_MODEL" "${ARGS[@]}"
fi

mkdir -p "$HF_HOME"
DOCKER_ENV=(-e "HF_HOME=/root/.cache/huggingface")
[ -n "${HF_TOKEN:-}" ] && DOCKER_ENV+=(-e "HF_TOKEN=$HF_TOKEN")
if [ "$MODE" = detach ]; then
  RUN=(docker run -d --name "$VLM_NAME" --restart unless-stopped)
else
  RUN=(docker run --rm -it --name "$VLM_NAME")
fi
exec "${RUN[@]}" \
  --gpus "$VLM_GPUS" \
  --ipc=host \
  -p "${VLM_PORT}:${VLM_PORT}" \
  -v "$HF_HOME:/root/.cache/huggingface" \
  "${DOCKER_ENV[@]}" \
  --entrypoint vllm \
  "$VLLM_IMAGE" \
  serve "$VLM_MODEL" "${ARGS[@]}"
