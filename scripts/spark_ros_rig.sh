#!/usr/bin/env bash
# W3 on ONE rented DGX Spark: the ROS agent stack walks against the ROS sim
# bridge.  Union of spark_demo_rig.sh (model serving, GB10 traps) and
# vast_voice_stack.sh (ROS 2 build); the demo's room_app is replaced by
# ros2 launch: agent_stack (brain) + world (sim bridge on :8010).
#
#   SPARK_SSH='ssh -p 599XX root@IP' scripts/spark_ros_rig.sh deploy
#   SPARK_SSH=... scripts/spark_ros_rig.sh serve        # tts + 2x vllm, warmed
#   SPARK_SSH=... scripts/spark_ros_rig.sh stack flat   # both launch trees
#   SPARK_SSH=... scripts/spark_ros_rig.sh record flat_rapidfire --require-speech 6
#   SPARK_SSH=... scripts/spark_ros_rig.sh fetch ~/Desktop/wojtek_takes_ros
#   SPARK_SSH=... scripts/spark_ros_rig.sh status | logs <svc> | stop
#
# Differences from the demo rig, all deliberate:
#   - ASR runs INSIDE asr_node (transformers backend on aarch64), no asr_server.
#   - TTS stays the optimized tts_server in its own venv; the ROS tts node
#     uses engine:=remote, inheriting conditionals-once/token-cap/tail-trim.
#   - The latency probe writes perf JSONL with the agent stages
#     (agent.turn, brain.translate) -- read with ./training/run.sh perf.
#
# Secrets: HF_TOKEN rides stdin to a 600-mode file, never a command line.
set -euo pipefail

REMOTE_DIR="${REMOTE_DIR:-/root/wojtek}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3-VL-4B-Instruct-FP8}"
BIELIK_MODEL="${BIELIK_MODEL:-speakleash/Bielik-4.5B-v3.0-Instruct-FP8-Dynamic}"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
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
  echo ">> code + scenes + robot model + ROS packages"
  rsh "mkdir -p ${REMOTE_DIR}/ros/src ${REMOTE_DIR}/ws/src"
  rsync -rlptzL -e "ssh -p $(ssh_port)" \
    --exclude runs --exclude .venv --exclude __pycache__ --exclude '.jax_cache' \
    --exclude 'assets/room' --exclude tests --exclude jobs \
    "$REPO/training" "$(ssh_host):${REMOTE_DIR}/"
  rsync -rlptz -e "ssh -p $(ssh_port)" \
    "$REPO/ros/src/wojtek_description" "$REPO/ros/src/wojtek_policy" \
    "$(ssh_host):${REMOTE_DIR}/ros/src/"
  rsync -rlptz -e "ssh -p $(ssh_port)" --exclude __pycache__ \
    "$REPO/ros/src/wojtek_agent_msgs" "$REPO/ros/src/wojtek_voice" \
    "$REPO/ros/src/wojtek_brain" "$REPO/ros/src/wojtek_agent_bringup" \
    "$REPO/ros/src/wojtek_agent_perf" "$REPO/ros/src/wojtek_sim_bridge" \
    "$(ssh_host):${REMOTE_DIR}/ws/src/"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s' "$HF_TOKEN" | rsh "cat > ~/.hf_token && chmod 600 ~/.hf_token"
    echo ">> HF token shipped (private policy repo)"
  fi
  rsh "bash -s" <<EOF
set -euo pipefail
. /etc/os-release
[ "\${VERSION_ID}" = "24.04" ] || { echo "need ubuntu 24.04 for ROS ${ROS_DISTRO}, got \${VERSION_ID}"; exit 1; }
apt-get update -qq >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  curl gnupg lsb-release software-properties-common \
  python3-pip python3.12-venv libegl1 libgles2 libglvnd0 ffmpeg >/dev/null
if [ ! -d /opt/ros/${ROS_DISTRO} ]; then
  add-apt-repository -y universe >/dev/null
  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu \$UBUNTU_CODENAME main" \
    > /etc/apt/sources.list.d/ros2.list
  apt-get update -qq >/dev/null
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    ros-${ROS_DISTRO}-ros-base ros-dev-tools >/dev/null
fi
PIP="pip3 install -q --break-system-packages"
# torch FIRST, from the cu130 index; everything after must not move it
# (docs/plans/spark-port.md).
\$PIP --ignore-installed typing_extensions
\$PIP torch --index-url https://download.pytorch.org/whl/cu130
\$PIP 'numpy<2' mujoco 'transformers==5.16.0' 'tokenizers>=0.23.1,<0.24' soundfile accelerate \
  fastapi uvicorn websockets httpx loguru pillow scipy \
  imageio imageio-ffmpeg huggingface_hub wsproto
# typing_extensions: this image ships it via apt (no RECORD file);
# pip must overlay it, not uninstall it.
\$PIP --ignore-installed PyJWT typing_extensions vllm 2>&1 | tail -1
# Chatterbox in its OWN venv: its transformers pin fights vllm's (spark rig).
python3 -m venv /root/venv_tts
/root/venv_tts/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu130
/root/venv_tts/bin/pip install -q --no-deps chatterbox-tts
/root/venv_tts/bin/pip install -q 'numpy<2' 'transformers==4.46.3' 'librosa==0.11.0' \
  s3tokenizer 'diffusers==0.29.0' resemble-perth 'conformer==0.3.2' \
  safetensors omegaconf pyloudnorm fastapi uvicorn loguru httpx pillow
pip3 install -q --break-system-packages ${REMOTE_DIR}/ros/src/wojtek_policy 2>&1 | tail -1
# Build the ROS workspace with the SYSTEM python (the same one that got the
# model deps above), so node imports just work.
set +u; source /opt/ros/${ROS_DISTRO}/setup.bash; set -u
cd ${REMOTE_DIR}/ws && colcon build --symlink-install \
  --packages-select wojtek_agent_msgs wojtek_voice wojtek_brain \
                    wojtek_agent_bringup wojtek_agent_perf wojtek_sim_bridge \
  2>&1 | tail -3
python3 - <<'PY'
import torch, platform
assert "cu13" in torch.__version__, f"pip moved torch to {torch.__version__}!"
print("deploy ok:", platform.machine(), "torch", torch.__version__,
      "cap", torch.cuda.get_device_capability(0))
# Clock sanity: some vast GB10 machines are platform power-capped to
# ~940 MHz of 3003 and never boost (machine 51319, 2026-08-26) -- every
# latency number from such a box is a fiction. Load the GPU and look.
import subprocess, threading
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
stop = False
def burn():
    global a
    while not stop: a = a @ a * 0.001
t = threading.Thread(target=burn); t.start()
import time; time.sleep(5)
out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout
stop = True; t.join(); torch.cuda.synchronize()
cur, mx = [int(x.split()[0]) for x in out.strip().split(",")]
print(f"clock under load: {cur} / {mx} MHz")
assert cur > mx * 0.6, (
    f"GPU stuck at {cur} MHz of {mx} -- power-capped host, DESTROY this "
    "instance and rent a different machine; latency here is meaningless")
PY
EOF
  ;;

serve)
  need_ssh
  rsh "bash -s" <<EOF
set -euo pipefail
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
# CUDA MPS or everything time-slices (18 s vs 5.4 s medians, 2026-08-22).
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log
  mkdir -p /tmp/mps /tmp/mps-log
  pgrep -f nvidia-cuda-mps-control >/dev/null || nvidia-cuda-mps-control -d \
    && echo "MPS control daemon up" || echo "MPS unavailable; running time-sliced"
fi
[ -f ~/.hf_token ] && export HF_TOKEN="\$(cat ~/.hf_token)"
mkdir -p /root/logs /root/takes
cd ${REMOTE_DIR}/training
if ! curl -sf localhost:8120/health >/dev/null 2>&1; then
  setsid nohup env TTS_LANGUAGE=pl TQDM_DISABLE=1 TTS_STREAM_SPLIT=on \
    /root/venv_tts/bin/python -m wojtek_rl.agent.tts_server --port 8120 \
    --temperature 0.6 --cache 256 \
    > /root/logs/tts.log 2>&1 &
  echo "tts starting (warmup inside)"
fi
for i in \$(seq 1 60); do
  curl -sf localhost:8120/health >/dev/null 2>&1 && break; sleep 10
done
# vllm LAST; two loads staggered (unified-memory profiler trap).
if [ -n "${BIELIK_MODEL}" ] && ! curl -sf localhost:8091/v1/models >/dev/null 2>&1; then
  setsid nohup env VLLM_USE_DEEP_GEMM=0 HF_TOKEN="\${HF_TOKEN:-}" \
    vllm serve ${BIELIK_MODEL} --port 8091 --max-model-len 4096 \
    --gpu-memory-utilization 0.18 --enforce-eager \
    > /root/logs/bielik.log 2>&1 &
  echo "bielik starting"; sleep 20
fi
if ! curl -sf localhost:8090/v1/models >/dev/null 2>&1; then
  setsid nohup env VLLM_USE_DEEP_GEMM=0 HF_TOKEN="\${HF_TOKEN:-}" \
    vllm serve ${AGENT_MODEL} --port 8090 --max-model-len 8192 \
    --gpu-memory-utilization 0.22 --enforce-eager \
    > /root/logs/vllm.log 2>&1 &
  echo "vllm starting"
fi
for i in \$(seq 1 120); do
  curl -sf localhost:8090/v1/models >/dev/null 2>&1 && \
  curl -sf localhost:8120/health >/dev/null 2>&1 && break
  sleep 10
done
echo "--- health:"
curl -sf localhost:8090/v1/models >/dev/null && echo "vllm   UP" || echo "vllm   DOWN"
curl -sf localhost:8091/v1/models >/dev/null && echo "bielik UP" || echo "bielik DOWN"
curl -sf localhost:8120/health >/dev/null && echo "tts    UP" || echo "tts    DOWN"
# Warm the VISION path (first image turn ~82 s cold; never inside a take).
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

stack)
  need_ssh
  scene="${1:?usage: stack flat|castle}"
  spawn=""
  [[ "$scene" == castle ]] && spawn='WOJTEK_SPAWN=2.5,-3.0'
  rsh "bash -s" <<EOF
set -euo pipefail
pkill -f 'agent_stack.launch|world.launch|install/wojtek' 2>/dev/null || true
sleep 2
[ -f ~/.hf_token ] && export HF_TOKEN="\$(cat ~/.hf_token)"
set +u; source /opt/ros/${ROS_DISTRO}/setup.bash; set -u
(cd ${REMOTE_DIR}/ws && colcon build --symlink-install \
  --packages-select wojtek_agent_msgs wojtek_voice wojtek_brain \
                    wojtek_agent_bringup wojtek_agent_perf wojtek_sim_bridge \
  2>&1 | tail -1)
set +u; source ${REMOTE_DIR}/ws/install/setup.bash; set -u
export PYTHONPATH="${REMOTE_DIR}/training:\${PYTHONPATH:-}"
export HF_ORGANIZATION=hvsr-robotics TQDM_DISABLE=1
# Brain half: voice pipeline + VLM agent. openai nav backend = Qwen drives
# navigation from the same served model, as in the recorded demo takes.
setsid nohup ros2 launch wojtek_agent_bringup agent_stack.launch.py \
  agent_url:=http://127.0.0.1:8090 agent_model:=${AGENT_MODEL} \
  vlm_backend:=openai forward_scale:=2.0 \
  bielik_url:=http://127.0.0.1:8091 bielik_model:=${BIELIK_MODEL} \
  asr_backend:=transformers asr_model:=large-v3 \
  tts_engine:=remote tts_url:=http://127.0.0.1:8120 \
  trace_path:=/root/takes/ros_${scene}_trace.jsonl \
  perf:=true perf_out:=/root/takes/ros_${scene}_perf.jsonl \
  >> /root/logs/agent_stack.log 2>&1 &
# World half: the sim bridge, scene via env like the demo rig.
setsid nohup env SCENE=${scene} ${spawn} MUJOCO_GL=egl \
  HF_TOKEN="\${HF_TOKEN:-}" HF_ORGANIZATION=hvsr-robotics TQDM_DISABLE=1 \
  PYTHONPATH="${REMOTE_DIR}/training:\${PYTHONPATH:-}" \
  ros2 launch wojtek_sim_bridge world.launch.py \
  >> /root/logs/world.log 2>&1 &
for i in \$(seq 1 60); do
  python3 -c "import socket; socket.create_connection(('127.0.0.1', 8010), 2).close()" 2>/dev/null && break
  sleep 5
done
python3 -c "import socket; socket.create_connection(('127.0.0.1', 8010), 2).close()" \
  && echo "world(${scene}) UP on :8010" \
  || { echo "world(${scene}) DOWN -- log tails:"; tail -15 /root/logs/world.log; tail -15 /root/logs/agent_stack.log; exit 1; }
# A take must not start while the ASR engine is still warming: its first
# question would queue behind the warmup decode and answer a turn late
# ("Siad!" executed after "Daj łapę", 2026-08-26).
for i in \$(seq 1 30); do
  grep -q "ASR warmed" /root/logs/agent_stack.log 2>/dev/null && { echo "ASR warmed"; break; }
  sleep 5
done
# Prerecord the canned phrase bank into the TTS cache: a canned ack then
# costs a cache hit (~0.1 s) instead of a first-time synthesis (2-4 s).
PYTHONPATH="${REMOTE_DIR}/ws/src/wojtek_brain" python3 - <<'PREWARM'
import subprocess, sys
sys.path.insert(0, "${REMOTE_DIR}/ws/src/wojtek_brain")
import httpx
from wojtek_brain.phrases import all_phrases
lines = all_phrases()
for line in lines:
    httpx.post("http://127.0.0.1:8120/synthesize", json={"text": line}, timeout=120)
print(f"prewarmed {len(lines)} canned phrases into the TTS cache")
PREWARM
EOF
  ;;

record)
  need_ssh
  name="${1:?usage: record <scenario-name> [scenario.py gates...]}"; shift || true
  rsh "bash -s" <<EOF
set -euo pipefail
cd ${REMOTE_DIR}/training
TQDM_DISABLE=1 python3 -m wojtek_rl.agent.scenario \
  --script scenarios/${name}.json --video /root/takes/ros_${name}.mp4 $* \
  2>&1 | tail -15
ls -la /root/takes/ros_${name}.mp4
EOF
  ;;

fetch)
  need_ssh
  out="${1:-$HOME/Desktop/wojtek_takes_ros}"
  mkdir -p "$out"
  rsync -rtz -e "ssh -p $(ssh_port)" "$(ssh_host):/root/takes/" "$out/"
  rsync -rtz -e "ssh -p $(ssh_port)" "$(ssh_host):/root/logs/" "$out/logs/" 2>/dev/null || true
  ls -la "$out"
  ;;

logs) rsh "tail -40 /root/logs/${1:-agent_stack}.log" ;;
status) rsh "nvidia-smi --query-gpu=memory.used --format=csv,noheader; for p in 8090 8091 8120 8010; do curl -sf -m 3 localhost:\$p/health >/dev/null 2>&1 || curl -sf -m 3 localhost:\$p/v1/models >/dev/null 2>&1 || python3 -c \"import socket; socket.create_connection(('127.0.0.1', \$p), 2).close()\" 2>/dev/null && echo \"\$p UP\" || echo \"\$p down\"; done; pgrep -fc 'vllm serve' | sed 's/^/vllm engines: /'" ;;
stop) rsh "pkill -f 'vllm serve|tts_server|agent_stack.launch|world.launch|install/wojtek' || true; echo stopped" ;;
*) echo "usage: $0 {deploy|serve|stack <scene>|record <name> [gates]|fetch [dir]|logs <svc>|status|stop}" >&2; exit 1 ;;
esac
