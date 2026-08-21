#!/usr/bin/env bash
# The whole demo on ONE rented DGX Spark: sim + brain + ears + voice + takes.
#
# Recordings happen on GB10 because that is the hardware the robot carries;
# a take timed on a discrete x86 card is a fiction (user call, 2026-08-21).
#
#   SPARK_SSH='ssh -p 599XX root@IP' scripts/spark_demo_rig.sh deploy
#   SPARK_SSH=... scripts/spark_demo_rig.sh serve       # vllm + asr + tts, warmed
#   SPARK_SSH=... scripts/spark_demo_rig.sh room flat   # (re)start the sim
#   SPARK_SSH=... scripts/spark_demo_rig.sh record flat_rapidfire --require-speech 6
#   SPARK_SSH=... scripts/spark_demo_rig.sh fetch ~/Desktop/takes
#   SPARK_SSH=... scripts/spark_demo_rig.sh status | logs vllm|asr|tts|room | stop
#
# Secrets: HF_TOKEN is read from THIS shell's environment and shipped over
# stdin to a 600-mode file (the policy checkpoint repo is private). Nothing
# secret goes on a command line or into this file.
#
# GB10 traps this script bakes in (docs/plans/spark-port.md):
#   - pip must never touch the vendor-matched torch     (--no-deps + pins)
#   - vLLM: VLLM_USE_DEEP_GEMM=0 (sm_121 FP8), ABSOLUTE memory sizing, and a
#     warmup that includes an IMAGE (first vision turn costs 82 s cold)
#   - EGL: pin the NVIDIA ICD or mujoco dies in mesa's dri2 path
#   - unified memory: drop caches before the servers claim their arenas
set -euo pipefail

REMOTE_DIR="${REMOTE_DIR:-/root/wojtek}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3-VL-4B-Instruct-FP8}"
SCENE_LIST="${SCENE_LIST:-flat castle}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

need_ssh() { [[ -n "${SPARK_SSH:-}" ]] || { echo "set SPARK_SSH='ssh -p PORT root@IP'" >&2; exit 1; }; }
rsh() { need_ssh; ${SPARK_SSH} "$@"; }
ssh_port() { sed -n 's/.*-p  *\([0-9][0-9]*\).*/\1/p' <<<"${SPARK_SSH}"; }
ssh_host() { awk '{print $NF}' <<<"${SPARK_SSH}"; }

cmd="${1:-status}"; shift || true
case "$cmd" in

deploy)
  need_ssh
  echo ">> code + scenes + robot model (~320 MB, follows asset symlinks)"
  rsh "mkdir -p ${REMOTE_DIR}/ros/src"
  rsync -rlptzL -e "ssh -p $(ssh_port)" \
    --exclude runs --exclude .venv --exclude __pycache__ --exclude '.jax_cache' \
    --exclude 'assets/room' --exclude tests --exclude jobs \
    "$REPO/training" "$(ssh_host):${REMOTE_DIR}/"
  rsync -rlptz -e "ssh -p $(ssh_port)" \
    "$REPO/ros/src/wojtek_description" "$(ssh_host):${REMOTE_DIR}/ros/src/"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s' "$HF_TOKEN" | rsh "cat > ~/.hf_token && chmod 600 ~/.hf_token"
    echo ">> HF token shipped (private policy repo)"
  fi
  rsh "bash -s" <<'EOF'
set -euo pipefail
apt-get update -qq >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-pip libegl1 libgles2 libglvnd0 ffmpeg >/dev/null
PIP="pip3 install -q --break-system-packages"
# torch FIRST, from the cu130 index; everything after must not move it.
$PIP torch --index-url https://download.pytorch.org/whl/cu130
$PIP 'numpy<2' mujoco 'transformers==5.2.0' soundfile accelerate \
  fastapi uvicorn websockets httpx loguru pillow scipy \
  imageio imageio-ffmpeg huggingface_hub
$PIP --ignore-installed PyJWT vllm 2>&1 | tail -1
# Chatterbox lives in its OWN venv: its transformers/tokenizers pins are
# incompatible with vllm's, and a shared env ends up half-upgraded.
python3 -m venv /root/venv_tts
/root/venv_tts/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu130
/root/venv_tts/bin/pip install -q --no-deps chatterbox-tts
/root/venv_tts/bin/pip install -q 'numpy<2' 'transformers==4.46.3' 'librosa==0.11.0' \
  s3tokenizer 'diffusers==0.29.0' resemble-perth 'conformer==0.3.2' \
  safetensors omegaconf pyloudnorm fastapi uvicorn
python3 - <<'PY'
import torch, platform
assert "cu13" in torch.__version__, f"pip moved torch to {torch.__version__}!"
print("deploy ok:", platform.machine(), "torch", torch.__version__,
      "cap", torch.cuda.get_device_capability(0))
PY
EOF
  ;;

serve)
  need_ssh
  rsh "bash -s" <<EOF
set -euo pipefail
cd ${REMOTE_DIR}/training
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
[ -f ~/.hf_token ] && export HF_TOKEN="\$(cat ~/.hf_token)"
mkdir -p /root/logs /root/takes
if ! curl -sf localhost:8090/v1/models >/dev/null 2>&1; then
  setsid nohup env VLLM_USE_DEEP_GEMM=0 HF_TOKEN="\${HF_TOKEN:-}" \
    vllm serve ${AGENT_MODEL} --port 8090 --max-model-len 8192 \
    --gpu-memory-utilization 0.22 --enforce-eager \
    > /root/logs/vllm.log 2>&1 &
  echo "vllm starting"
fi
if ! curl -sf localhost:8110/health >/dev/null 2>&1; then
  setsid nohup env ASR_BACKEND=transformers HF_TOKEN="\${HF_TOKEN:-}" \
    python3 -m wojtek_rl.agent.asr_server --port 8110 --model large-v3 \
    > /root/logs/asr.log 2>&1 &
  echo "asr starting"
fi
if ! curl -sf localhost:8120/health >/dev/null 2>&1; then
  # tts_server runs from ITS OWN venv: chatterbox pins transformers 4.46 /
  # tokenizers 0.20 while vllm needs 5.x / 0.23, and one shared env cannot
  # hold both (cost a serve cycle to learn). deploy builds /root/venv_tts.
  setsid nohup env TTS_LANGUAGE=pl TQDM_DISABLE=1 \
    /root/venv_tts/bin/python -m wojtek_rl.agent.tts_server --port 8120 \
    > /root/logs/tts.log 2>&1 &
  echo "tts starting (warmup inside)"
fi
for i in \$(seq 1 120); do
  curl -sf localhost:8090/v1/models >/dev/null 2>&1 && \
  curl -sf localhost:8110/health >/dev/null 2>&1 && \
  curl -sf localhost:8120/health >/dev/null 2>&1 && break
  sleep 10
done
echo "--- health:"
curl -sf localhost:8090/v1/models >/dev/null && echo "vllm  UP" || echo "vllm  DOWN"
curl -sf localhost:8110/health && echo || echo "asr   DOWN"
curl -sf localhost:8120/health && echo || echo "tts   DOWN"
# Warm the VISION path: the first image turn costs ~82 s (tower autotune)
# and must never land inside a take.
python3 - <<'PY'
import base64, io, time
import httpx
from PIL import Image
import numpy as np
img = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype("uint8"))
buf = io.BytesIO(); img.save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()
t0 = time.monotonic()
r = httpx.post("http://127.0.0.1:8090/v1/chat/completions", timeout=300, json={
    "model": "${AGENT_MODEL}",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": "One word: what colour dominates?"}]}],
    "max_tokens": 8})
print(f"vision warm in {time.monotonic()-t0:.0f}s status {r.status_code}")
PY
EOF
  ;;

room)
  need_ssh
  scene="${1:?usage: room flat|castle}"
  spawn=""
  [[ "$scene" == castle ]] && spawn='WOJTEK_SPAWN=2.5,-3.0'
  rsh "bash -s" <<EOF
set -euo pipefail
cd ${REMOTE_DIR}/training
pkill -f 'wojtek_rl.room_app' 2>/dev/null || true
sleep 2
[ -f ~/.hf_token ] && export HF_TOKEN="\$(cat ~/.hf_token)"
setsid nohup env SCENE=${scene} ${spawn} MUJOCO_GL=egl \
  HF_TOKEN="\${HF_TOKEN:-}" TQDM_DISABLE=1 \
  VLM_BACKEND=openai AGENT_URL=http://127.0.0.1:8090 \
  ASR_URL=http://127.0.0.1:8110 TTS_ENGINE=remote TTS_URL=http://127.0.0.1:8120 \
  AGENT_TRACE=/root/takes/${scene}_trace.jsonl \
  python3 -m wojtek_rl.room_app --port 8010 --agent-model "${AGENT_MODEL}" \
  > /root/logs/room_${scene}.log 2>&1 &
for i in \$(seq 1 60); do curl -sf localhost:8010/api/info >/dev/null 2>&1 && break; sleep 5; done
curl -sf localhost:8010/api/info >/dev/null && echo "room(${scene}) UP" \
  || { echo "room(${scene}) DOWN -- log tail:"; tail -20 /root/logs/room_${scene}.log; exit 1; }
EOF
  ;;

record)
  need_ssh
  name="${1:?usage: record <scenario-name> [scenario.py gates...]}"; shift || true
  rsh "bash -s" <<EOF
set -euo pipefail
cd ${REMOTE_DIR}/training
TQDM_DISABLE=1 python3 -m wojtek_rl.agent.scenario \
  --script scenarios/${name}.json --video /root/takes/${name}.mp4 $* \
  2>&1 | tail -15
ls -la /root/takes/${name}.mp4
EOF
  ;;

fetch)
  need_ssh
  out="${1:-$HOME/Desktop/wojtek_takes}"
  mkdir -p "$out"
  rsync -rtz -e "ssh -p $(ssh_port)" "$(ssh_host):/root/takes/" "$out/"
  ls -la "$out"
  ;;

logs) rsh "tail -30 /root/logs/${1:-vllm}.log" ;;
status) rsh "nvidia-smi --query-gpu=memory.used --format=csv,noheader; for p in 8090 8110 8120 8010; do curl -sf -m 3 localhost:\$p/health >/dev/null 2>&1 || curl -sf -m 3 localhost:\$p/v1/models >/dev/null 2>&1 || curl -sf -m 3 localhost:\$p/api/info >/dev/null 2>&1 && echo \"\$p UP\" || echo \"\$p down\"; done" ;;
stop) rsh "pkill -f 'vllm serve|asr_server|tts_server|room_app' || true; echo stopped" ;;
*) echo "usage: $0 {deploy|serve|room <scene>|record <name> [gates]|fetch [dir]|logs <svc>|status|stop}" >&2; exit 1 ;;
esac
