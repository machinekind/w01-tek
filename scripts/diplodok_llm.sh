#!/usr/bin/env bash
# Host the agent VLM (and optionally FutureNav) on diplodok over tailscale.
#
# diplodok is a shared GPU box (RTX 4090 Mobile 16 GB, Ada -> native FP8)
# reachable at diplodok.tail45a5e.ts.net (100.103.127.123), login `greg`.
# Ports 8000, 8001, 8080 (owner's FastAPI apps + Weaviate) and 8110 (Greg's
# NAVIDA server, the current GPU tenant) are TAKEN -- do not bind them. This
# script uses 8090 (vLLM) by default and leaves 8100 for a FutureNav server.
# Mind the GPU tenant: 16 GB shared; check nvidia-smi before serving.
#
# Usage (SSH access to diplodok required; set DIPLODOK_USER):
#   DIPLODOK_USER=<user> scripts/diplodok_llm.sh check     # GPU, disk, ports
#   DIPLODOK_USER=<user> scripts/diplodok_llm.sh install   # uv + vllm venv
#   DIPLODOK_USER=<user> scripts/diplodok_llm.sh serve     # start vLLM (detached)
#   scripts/diplodok_llm.sh status                          # probe /v1/models (no ssh)
#   DIPLODOK_USER=<user> scripts/diplodok_llm.sh logs      # tail server log
#   DIPLODOK_USER=<user> scripts/diplodok_llm.sh stop
#
# Then point the demo at it:
#   AGENT_URL=http://diplodok.tail45a5e.ts.net:8090 ./training/run.sh room ...
#
# Notes:
# - MODEL is FP8 and the 4090 is Ada (compute 8.9), so FP8 runs natively.
# - DRIVER SHIM, mandatory today: diplodok's installed nvidia userspace
#   (580.173.02) is ahead of the loaded kernel module (580.159.03), so a plain
#   CUDA process dies with `cuInit 804 / Driver/library version mismatch`.
#   Greg keeps matching 580.159 userspace libs in ~/nvidia-580.159, and
#   `source ~/nvidia-580.159/env.sh` (LD_LIBRARY_PATH + PATH) makes the GPU
#   work again -- per shell, and inherited by anything launched from it, so
#   EVERY remote command here sources it. It becomes a harmless no-op after a
#   reboot (which is the real fix, but needs root and would drop the other
#   tenant's services on 8000/8001/8080).
# - Everything lives under ~/wojtek_vllm on diplodok; nothing touches the
#   owner's services. Uninstall = `rm -rf ~/wojtek_vllm`.

set -euo pipefail

HOST="${DIPLODOK_HOST:-diplodok.tail45a5e.ts.net}"
USER_="${DIPLODOK_USER:-greg}"
PORT="${DIPLODOK_PORT:-8090}"
MODEL="${AGENT_MODEL:-Qwen/Qwen3-VL-4B-Instruct-FP8}"
REMOTE_DIR="wojtek_vllm"
BASE_URL="http://${HOST}:${PORT}"

need_user() {
  if [[ -z "$USER_" ]]; then
    echo "DIPLODOK_USER is not set (ssh login for ${HOST})" >&2
    exit 1
  fi
}

# Sourced first by every remote command: see DRIVER SHIM in the header.
SHIM='[ -f ~/nvidia-580.159/env.sh ] && source ~/nvidia-580.159/env.sh; '

rssh() { ssh -o ConnectTimeout=10 "${USER_}@${HOST}" "${SHIM}$*"; }

cmd="${1:-status}"
case "$cmd" in
  check)
    need_user
    # No `set -e` here: a broken driver makes nvidia-smi fail, and that is
    # exactly the case this probe exists to report -- keep going and print it.
    rssh "
      echo '== gpu:'; nvidia-smi --query-gpu=name,memory.total,memory.used,compute_cap,driver_version --format=csv,noheader 2>&1 | head -2
      echo '== kernel module vs userspace:'
      echo \"  loaded:   \$(cat /proc/driver/nvidia/version 2>/dev/null | sed -n 's/.*Module *\\([0-9.]*\\).*/\\1/p')\"
      echo \"  on-disk:  \$(modinfo nvidia 2>/dev/null | sed -n 's/^version: *//p')\"
      echo '== cuda usable (with shim):'
      python3 -c 'import ctypes; rc = ctypes.CDLL(\"libcuda.so.1\").cuInit(0); print(\"  OK\" if rc == 0 else f\"  BROKEN (cuInit {rc}) -- see DRIVER SHIM in script header\")' 2>&1 | tail -1
      echo '== gpu tenants:'; fuser -v /dev/nvidia0 2>&1 | tail -n +2 | head -4 || echo '  none'
      echo '== disk (home):'; df -h ~ | tail -1
      echo '== port ${PORT}:'; (ss -ltn 2>/dev/null || netstat -ltn) | grep -q ':${PORT} ' && echo '  TAKEN' || echo '  free'
      echo '== weights cached:'; du -sh ~/.cache/huggingface/hub/models--\$(echo '${MODEL}' | sed 's|/|--|g') 2>/dev/null || echo '  not downloaded'
    "
    ;;
  install)
    need_user
    rssh "
      set -e
      mkdir -p ~/${REMOTE_DIR}
      cd ~/${REMOTE_DIR}
      command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH=\"\$HOME/.local/bin:\$PATH\"
      [ -d .venv ] || uv venv --python 3.12 .venv
      # ninja is not a vllm dependency but its FP8/torch.compile path shells
      # out to it -- without it the server dies with FileNotFoundError: 'ninja'
      # after loading weights.
      uv pip install --python .venv/bin/python -U vllm ninja
      .venv/bin/python -c 'import vllm; print(\"vllm\", vllm.__version__)'
    "
    ;;
  serve)
    need_user
    rssh "
      set -e
      cd ~/${REMOTE_DIR}
      if curl -sf localhost:${PORT}/v1/models >/dev/null 2>&1; then
        echo 'already serving on ${PORT}'; exit 0
      fi
      # The venv's bin dir must be ON PATH, not just used by absolute path:
      # vllm's compile path shells out to \`ninja\`, which lives there.
      export PATH=\"\$PWD/.venv/bin:\$HOME/.local/bin:\$PATH\"
      # FlashInfer JIT-builds its sampling kernels against the SYSTEM nvcc,
      # whose CUB no longer has BlockAdjacentDifference::FlagHeads -> nvcc
      # errors, ninja fails, engine init dies. vLLM's native sampler is fine
      # for our request rate; drop this once flashinfer matches the toolkit.
      export VLLM_USE_FLASHINFER_SAMPLER=0
      # setsid: survive this ssh session closing. The shim env is already in
      # this shell (see rssh) and is inherited by the server process.
      setsid nohup .venv/bin/vllm serve '${MODEL}' \
        --host 0.0.0.0 --port ${PORT} \
        --max-model-len 16384 --gpu-memory-utilization 0.80 \
        > vllm.log 2>&1 < /dev/null &
      echo \$! > vllm.pid
      echo 'started (pid '\$(cat vllm.pid)'); first run downloads weights -- watch: '
      echo '  DIPLODOK_USER=${USER_} $0 logs'
    "
    ;;
  status)
    echo "probing ${BASE_URL}/v1/models ..."
    curl -sf -m 10 "${BASE_URL}/v1/models" && echo || { echo "not serving"; exit 1; }
    ;;
  env)
    # Paste-able hookup for the room demo (also usable for nav-eval's --base-url).
    echo "export AGENT_URL=${BASE_URL}"
    echo "export AGENT_MODEL=${MODEL}"
    ;;
  logs)
    need_user
    rssh "tail -n 60 ~/${REMOTE_DIR}/vllm.log"
    ;;
  stop)
    need_user
    rssh "
      [ -f ~/${REMOTE_DIR}/vllm.pid ] && kill \$(cat ~/${REMOTE_DIR}/vllm.pid) 2>/dev/null || true
      pkill -u \$(whoami) -f 'vllm serve' || true
      echo stopped
    "
    ;;
  *)
    echo "usage: $0 {check|install|serve|status|env|logs|stop}" >&2
    exit 1
    ;;
esac
