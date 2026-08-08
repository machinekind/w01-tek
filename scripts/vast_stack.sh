#!/usr/bin/env bash
# Provision BOTH of Wojtek's remote models on one vast.ai box:
#
#   FutureNav-4B  (port 8100)  the navigation specialist -- follows a VLN-CE
#                              instruction, emits MOVE_FORWARD/TURN/STOP
#   Qwen3-VL-4B   (port 8090)  the chat agent + search observer, served by
#                              vLLM behind an OpenAI-compatible API
#
# The room demo then runs with navigation on one and everything else on the
# other:
#   ./training/run.sh room --vlm-backend futurenav \
#       --vlm-url http://127.0.0.1:8100 --agent-url http://127.0.0.1:8090
#
# This script NEVER creates or destroys an instance. `plan` prints the vast
# commands for that; you run them, then hand the ssh target to `deploy`.
#
#   scripts/vast_stack.sh plan                      # sizing + offer search
#   VAST_SSH='ssh -p 12345 root@1.2.3.4' scripts/vast_stack.sh check
#   VAST_SSH=... scripts/vast_stack.sh deploy       # both models, idempotent
#   VAST_SSH=... scripts/vast_stack.sh serve        # start both, detached
#   VAST_SSH=... scripts/vast_stack.sh tunnel       # 8090+8100 -> localhost
#   scripts/vast_stack.sh status                    # probe both (via tunnel)
#   VAST_SSH=... scripts/vast_stack.sh logs futurenav|vllm
#   VAST_SSH=... scripts/vast_stack.sh stop
#
# VAST_SSH is the full ssh command vast gives you (`vastai ssh-url <id>` ->
# ssh://root@host:port). Everything else has a default:
#   VLLM_PORT=8090 FUTURENAV_PORT=8100
#   VLLM_GPU / FUTURENAV_GPU   pin each model to a GPU index on a multi-GPU box
#   AGENT_MODEL=Qwen/Qwen3-VL-4B-Instruct-FP8
set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8090}"
ASR_PORT="${ASR_PORT:-8110}"
ASR_MODEL="${ASR_MODEL:-large-v3}"
ASR_LANGUAGE="${ASR_LANGUAGE:-pl}"
FUTURENAV_PORT="${FUTURENAV_PORT:-8100}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3-VL-4B-Instruct-FP8}"
VLLM_DIR="${VLLM_DIR:-~/wojtek_vllm}"
FUTURENAV_DIR="${FUTURENAV_DIR:-~/futurenav}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

need_ssh() {
  if [[ -z "${VAST_SSH:-}" ]]; then
    echo "VAST_SSH is not set. Get it with: vastai ssh-url <instance-id>" >&2
    echo "  e.g. VAST_SSH='ssh -p 12345 root@84.1.2.3' $0 $*" >&2
    exit 1
  fi
}

# vast hands you `ssh -p 12345 root@host`; scp wants the port as -P and the
# host on its own, so split the command once and reuse the pieces.
# POSIX character class, not \+ : BSD sed (macOS) does not understand \+ and
# silently produces nothing, which sends scp to port 22 and hangs.
ssh_port() { sed -n 's/.*-p  *\([0-9][0-9]*\).*/\1/p' <<<"${VAST_SSH}"; }
ssh_host() { awk '{print $NF}' <<<"${VAST_SSH}"; }

rsh() { need_ssh; ${VAST_SSH} "$@"; }

rcp() {  # rcp <local...> <remote-dir>
  need_ssh
  local port dest
  port="$(ssh_port)"
  dest="${*: -1}"
  set -- "${@:1:$(($# - 1))}"
  scp ${port:+-P "$port"} "$@" "$(ssh_host):${dest}"
}

cmd="${1:-plan}"
case "$cmd" in

plan)
  cat <<PLAN
WHERE THE VOICE STACK RUNS: not here. Polish speech recognition
(faster-whisper) and Polish speech synthesis (Piper) run in the demo process
on YOUR machine, next to the browser -- they are small, and keeping them
local avoids shipping microphone audio to a rented box and the reply audio
back. Only the brain (and FutureNav) live on vast. Knobs: ASR_MODEL,
ASR_DEVICE, TTS_ENGINE, TTS_VOICE.

Polish note on the brain: Qwen3-Omni's SPEECH support does not include
Polish in either direction, so there is no reason to rent an 80 GB card for
it. Any capable text+vision model works; Polish TEXT quality is what matters
and it improves sharply with size. Set AGENT_MODEL to pick.

Two 4B models share one box. Measured/vendor footprints:

  FutureNav-4B (bf16 + VGGT)   ~10 GB, its deploy header asks for 24 GB
                               dedicated to be comfortable
  Qwen3-VL-4B-FP8 under vLLM   ~8 GB at 16k ctx with
                               --gpu-memory-utilization 0.45 (it grabs 80 %
                               of the card if you let it -- always pin it
                               when sharing)

  Bigger brain for better Polish  Qwen3-VL-30B-A3B-Instruct-FP8 ~35 GB,
                                  Qwen3-Omni-30B-A3B (text mode) ~70 GB bf16
                                  -- set AGENT_MODEL and raise the card size

MEASURED, not estimated -- a 24 GB card ran OUT during a recording:
  FutureNav idle 6.9 GB, but its VGGT KV cache GROWS through an episode and
  was at 8.2 GB by the end; vLLM at 0.30 reserves ~7.4 GB and keeps it;
  whisper large-v3 is 4.7 GB fp16 / 2.5 GB int8. Total on a 24 GB 3090
  reached 24.3/24.6 GB and the ASR died mid-session with
  "cudaErrorInvalidDevice".

So: ONE GPU with >= 40 GB (A6000 48, L40S 48, RTX 6000/5880 Ada 48, A100)
is the only comfortable option once ASR joins the box -- FutureNav's growth
needs headroom nothing else can predict. On 24 GB it fits ONLY with vLLM at
0.30 AND ASR_COMPUTE=int8, with nothing to spare. TWO 24 GB GPUs also works
-- pin one model per GPU:
    FUTURENAV_GPU=0 VLLM_GPU=1 $0 serve

BANDWIDTH IS THE REAL BILL, not the GPU. A cold deploy pulls ~30 GB
(FutureNav weights 17, Qwen 6, vLLM wheels 7); across four boxes that was
\$1.97 of download against \$0.78 of GPU time. Keep ONE box alive across a
session rather than destroying and re-deploying, filter offers on
inet_down_cost, and consider a vast volume (median \$0.20/GB/month, so ~\$6/mo
for this stack) if the weights need to survive overnight.

Find an offer and create it YOURSELF (this script never spawns anything):

  vastai search offers 'gpu_ram>=40 num_gpus=1 disk_space>=80 \\
      cuda_vers>=12.4 inet_down>=200 rentable=true' -o 'dph+' | head

  # devel image: FutureNav's deps build CUDA extensions
  vastai create instance <OFFER_ID> \\
      --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \\
      --disk 80 --ssh --direct

  vastai ssh-url <INSTANCE_ID>        # -> VAST_SSH for this script

Budget guard: skills/futurenav-nav-demo/scripts/vast_autodestroy.sh caps spend.
Destroy with: vastai destroy instance <INSTANCE_ID>

Then:
  VAST_SSH='ssh -p <port> root@<host>' $0 deploy
  VAST_SSH=... $0 serve
  VAST_SSH=... $0 tunnel      # keep this shell open
  $0 status
  eval "\$($0 env)" && ./training/run.sh room --vlm-backend futurenav \\
      --vlm-url http://127.0.0.1:${FUTURENAV_PORT} --agent-url http://127.0.0.1:${VLLM_PORT}
PLAN
  ;;

check)
  rsh 'bash -s' <<'EOF'
set -u
echo "== gpus:"; nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>&1 | head -4
echo "== driver/cuda:"; nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1
echo "== disk:"; df -h / | tail -1
echo "== python:"; python3 --version
echo "== nvcc (devel image?):"; command -v nvcc >/dev/null && nvcc --version | tail -1 || echo "  MISSING -- use a -devel image, FutureNav builds CUDA extensions"
EOF
  ;;

deploy)
  need_ssh
  # `deploy vllm` skips FutureNav (repo clone + 4B weights, several minutes)
  # when you only need the chat/search brain. `deploy futurenav` is the
  # mirror. Default does both.
  what="${2:-both}"
  FN_SRC="$REPO/training/wojtek_rl/futurenav_server"
  if [[ "$what" == "vllm" ]]; then
    echo ">> skipping FutureNav (deploy futurenav|both to include it)"
  else
  echo ">> [1/2] FutureNav -> ${FUTURENAV_DIR}"
  # Same steps as futurenav_server/deploy.sh (which assumes a plain ssh host
  # and so cannot reach vast's non-22 port). Every line there that mattered is
  # kept: the cu124 torch pin (an unpinned wheel resolves to cu130+, fails CUDA
  # init SILENTLY on vast's older drivers and serves from CPU at ~30 s/act) and
  # the assert that refuses to leave a CPU process squatting the port.
  rsh "mkdir -p $FUTURENAV_DIR"
  rcp "$FN_SRC/server.py" "$FN_SRC/smoke_test.py" "$FN_SRC/requirements.txt" "$FUTURENAV_DIR/"
  rsh "bash -s" <<EOF
set -euo pipefail
cd $FUTURENAV_DIR
[ -d FutureNav ] || git clone --depth 1 https://github.com/linglingxiansen/FutureNav.git
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cu124 torch torchvision
./venv/bin/pip install -q -r requirements.txt
[ -f weights/FutureNav-4B-Base/config.json ] || \
  ./venv/bin/hf download llxs/FutureNav --include "FutureNav-4B-Base/*" --local-dir weights
./venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU -- do NOT start the server'; print('cuda ok:', torch.cuda.get_device_name(0))"
echo FUTURENAV_OK
EOF
  fi

  if [[ "$what" == "futurenav" ]]; then
    echo ">> skipping vLLM"; exit 0
  fi
  echo ">> [2/2] vLLM + ${AGENT_MODEL} -> ${VLLM_DIR}"
  rsh "bash -s" <<EOF
set -euo pipefail
mkdir -p $VLLM_DIR && cd $VLLM_DIR
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
[ -d .venv ] || uv venv --python 3.12 .venv
# ninja is NOT a vllm dependency but its compile path shells out to it; without
# it the server dies with FileNotFoundError right after loading weights.
uv pip install --python .venv/bin/python -q -U vllm ninja huggingface_hub
.venv/bin/python -c "import vllm; print('vllm', vllm.__version__)"
.venv/bin/hf download ${AGENT_MODEL}
# Speech recognition shares the vLLM venv: it is one more CUDA consumer and
# needs no isolation. large-v3 on a GPU is ~0.13x realtime vs ~2x on a laptop.
uv pip install --python .venv/bin/python -q faster-whisper fastapi "uvicorn[standard]"
echo VLLM_OK
EOF
  rcp "$REPO/training/wojtek_rl/agent/asr_server.py" "${VLLM_DIR}/"
  echo ">> deployed both. next: $0 serve"
  ;;

serve)
  need_ssh
  rsh "bash -s" <<EOF
set -euo pipefail
cd $FUTURENAV_DIR
if ! curl -sf localhost:${FUTURENAV_PORT}/health >/dev/null 2>&1; then
  # uvicorn directly rather than a generated start.sh: one less layer of
  # heredoc quoting to get wrong, and the env is visible right here.
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export FUTURENAV_WEIGHTS="\$PWD/weights/FutureNav-4B-Base"
  export FUTURENAV_SRC="\$PWD/FutureNav/src"
  ${FUTURENAV_GPU:+export CUDA_VISIBLE_DEVICES=${FUTURENAV_GPU}}
  setsid nohup ./venv/bin/python -m uvicorn server:app \
    --host 0.0.0.0 --port ${FUTURENAV_PORT} >> futurenav.log 2>&1 < /dev/null &
  echo "futurenav starting on ${FUTURENAV_PORT}"
else
  echo "futurenav already up"
fi

cd $VLLM_DIR
if ! curl -sf localhost:${VLLM_PORT}/v1/models >/dev/null 2>&1; then
  export PATH="\$PWD/.venv/bin:\$HOME/.local/bin:\$PATH"
  # FlashInfer JIT-builds its sampler against the system nvcc; when their CUB
  # versions disagree the build fails and engine init dies. Native sampler is
  # fine at our request rate.
  export VLLM_USE_FLASHINFER_SAMPLER=0
  ${VLLM_GPU:+export CUDA_VISIBLE_DEVICES=${VLLM_GPU}}
  # Pinned low on purpose: FutureNav shares this card unless you split GPUs.
  setsid nohup .venv/bin/vllm serve '${AGENT_MODEL}' \
    --host 0.0.0.0 --port ${VLLM_PORT} \
    --max-model-len 16384 --gpu-memory-utilization \${VLLM_GPU_FRACTION:-0.45} \
    >> vllm.log 2>&1 < /dev/null &
  echo "vllm starting on ${VLLM_PORT}"
else
  echo "vllm already up"
fi

if ! curl -sf localhost:${ASR_PORT}/health >/dev/null 2>&1; then
  ASR_MODEL=${ASR_MODEL} ASR_LANGUAGE=${ASR_LANGUAGE} ASR_DEVICE=cuda ASR_COMPUTE=float16 \
    setsid nohup .venv/bin/python asr_server.py --port ${ASR_PORT} >> asr.log 2>&1 < /dev/null &
  echo "asr starting on ${ASR_PORT}"
else
  echo "asr already up"
fi
EOF
  echo ">> both starting (first run loads weights). watch: $0 logs vllm | $0 logs futurenav"
  ;;

tunnel)
  need_ssh
  echo ">> forwarding ${VLLM_PORT} and ${FUTURENAV_PORT} to localhost (Ctrl-C to drop)"
  ${VAST_SSH} -N \
    -L "${VLLM_PORT}:localhost:${VLLM_PORT}" \
    -L "${FUTURENAV_PORT}:localhost:${FUTURENAV_PORT}" \
    -L "${ASR_PORT}:localhost:${ASR_PORT}"
  ;;

status)
  ok=0
  echo -n "vllm      :${VLLM_PORT}  "
  curl -sf -m 8 "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null && { echo "UP"; } || { echo "down"; ok=1; }
  echo -n "asr       :${ASR_PORT}  "
  curl -sf -m 8 "http://127.0.0.1:${ASR_PORT}/health" && echo || { echo "down"; ok=1; }
  echo -n "futurenav :${FUTURENAV_PORT}  "
  curl -sf -m 8 "http://127.0.0.1:${FUTURENAV_PORT}/health" && echo || { echo "down"; ok=1; }
  exit $ok
  ;;

logs)
  need_ssh
  case "${2:-vllm}" in
    vllm) rsh "tail -n 60 $VLLM_DIR/vllm.log" ;;
    futurenav) rsh "tail -n 60 $FUTURENAV_DIR/futurenav.log" ;;
    asr) rsh "tail -n 60 $VLLM_DIR/asr.log" ;;
    *) echo "usage: $0 logs vllm|futurenav|asr" >&2; exit 1 ;;
  esac
  ;;

stop)
  need_ssh
  # `pkill -f 'server.py'` misses it: the server runs as `python -m uvicorn`,
  # and the pattern then matches your own ssh command line instead.
  rsh "pkill -f 'vllm serve' || true; pkill -f 'uvicorn server:app' || true; pkill -f asr_server.py || true; echo stopped"
  ;;

env)
  echo "export AGENT_URL=http://127.0.0.1:${VLLM_PORT}"
  echo "export AGENT_MODEL=${AGENT_MODEL}"
  echo "export VLM_BACKEND=futurenav"
  echo "export VLM_URL=http://127.0.0.1:${FUTURENAV_PORT}"
  # Voice runs locally; these are the defaults the demo already uses, printed
  # so one `eval` sets up the whole session.
  echo "export ASR_LANGUAGE=${ASR_LANGUAGE:-pl}"
  echo "export ASR_MODEL=${ASR_MODEL:-large-v3}"
  echo "export ASR_URL=http://127.0.0.1:${ASR_PORT}"
  echo "export TTS_ENGINE=${TTS_ENGINE:-piper}"
  echo "export TTS_VOICE=${TTS_VOICE:-pl_PL-mc_speech-medium}"
  ;;

*)
  echo "usage: $0 {plan|check|deploy|serve|tunnel|status|logs|stop|env}" >&2
  exit 1
  ;;
esac
