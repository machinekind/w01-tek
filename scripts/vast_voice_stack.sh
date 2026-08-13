#!/usr/bin/env bash
# ROS 2 voice stack (talk-only milestone) on one rented vast.ai box:
# audio bridge + VAD + whisper ASR + router + Bielik (vLLM) + TTS, per
# docs/plans/agentic-ros2.md.  Companion to vast_stack.sh (FutureNav +
# Qwen); both can share a box — ports do not collide.
#
# This script never creates or destroys instances; `plan` prints the
# commands for that.  VAST_SSH is the full ssh command vast gives you
# (`vastai ssh-url <id>` -> ssh://root@host:port):
#
#   VAST_SSH='ssh -p 12345 root@1.2.3.4' scripts/vast_voice_stack.sh check
#   VAST_SSH=... scripts/vast_voice_stack.sh deploy
#   VAST_SSH=... TTS_REF_WAV=~/refs/voice.wav TTS_REF_TEXT_FILE=~/refs/voice.txt \
#     scripts/vast_voice_stack.sh refs        # upload a clone reference
#   VAST_SSH=... scripts/vast_voice_stack.sh serve
#   VAST_SSH=... scripts/vast_voice_stack.sh tunnel   # keep open; browser -> ws://localhost:8765
#
# Then open ros/src/wojtek_voice/web/mic.html and connect.
set -euo pipefail

WS_PORT="${WS_PORT:-8765}"
VLLM_PORT="${VLLM_PORT:-8091}"                 # 8090 belongs to vast_stack.sh's Qwen
BIELIK_MODEL="${BIELIK_MODEL:-speakleash/Bielik-4.5B-v3.0-Instruct-FP8-Dynamic}"
ASR_MODEL="${ASR_MODEL:-large-v3}"             # benchmark large-v2 vs large-v3 (design doc)
ASR_LANGUAGE="${ASR_LANGUAGE:-pl}"
VAD_BACKEND="${VAD_BACKEND:-silero}"
TTS_ENGINE="${TTS_ENGINE:-chatterbox}"
TTS_REF_WAV="${TTS_REF_WAV:-}"                 # local wav to upload with `refs`
TTS_REF_TEXT_FILE="${TTS_REF_TEXT_FILE:-}"     # its exact transcript (F5 needs it)
STACK_DIR="${STACK_DIR:-~/wojtek_voice}"
VLLM_DIR="${VLLM_DIR:-~/wojtek_vllm}"          # shared with vast_stack.sh on a shared box
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
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
  scp -r ${port:+-P "$port"} "$@" "$(ssh_host):${dest}"
}

cmd="${1:-plan}"
case "$cmd" in

plan)
  cat <<PLAN
TALK-ONLY VOICE STACK, everything on the rented box (design Decision 4):
the browser is a dumb mic/speaker over one websocket.  Footprints:

  Bielik-4.5B FP8-Dynamic under vLLM   ~5 GB weights; pin
                                       --gpu-memory-utilization 0.35
  whisper ${ASR_MODEL} (fp16)             ~4.7 GB (int8: 2.5 GB)
  Chatterbox multilingual              ~3 GB
  silero VAD                           ~0 (2 MB, CPU)

A 24 GB card is comfortable; 16 GB works with ASR_COMPUTE=int8.  FP8 wants
Ada (compute 8.9); on Ampere either accept the Marlin fallback or set
BIELIK_MODEL=speakleash/Bielik-4.5B-v3.0-Instruct (bf16, ~9 GB).

IMAGE MUST BE ubuntu24.04 — ROS 2 ${ROS_DISTRO} exists only for Noble.  The
cudnn variant is REQUIRED: faster-whisper (ctranslate2) needs cuDNN 9 at
runtime and the plain -devel image does not ship it.

  vastai search offers 'gpu_ram>=20 num_gpus=1 disk_space>=60 \\
      compute_cap>=890 cuda_vers>=12.6 inet_down>=500 rentable=true' -o 'dph+' | head
  vastai create instance <OFFER_ID> \\
      --image nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 \\
      --disk 60 --ssh --direct
  vastai attach ssh <INSTANCE_ID> "\$(cat ~/.ssh/id_rsa.pub)"
  vastai ssh-url <INSTANCE_ID>        # -> VAST_SSH for this script

Filter on inet_down_cost — the ~15 GB cold download is the real bill
(Bielik 5, whisper 3, torch+chatterbox wheels ~6).  Arm teardown early:
scripts/vast_destroy_at.sh install <INSTANCE_ID> <HH:MM>.

Then:
  VAST_SSH='ssh -p <port> root@<host>' $0 deploy
  VAST_SSH=... $0 refs      # optional voice-clone reference
  VAST_SSH=... $0 serve
  VAST_SSH=... $0 tunnel    # keep open
  open ros/src/wojtek_voice/web/mic.html, connect to ws://localhost:${WS_PORT}
PLAN
  ;;

check)
  rsh 'bash -s' <<'EOF'
set -u
echo "== os:"; . /etc/os-release && echo "$PRETTY_NAME"
echo "== gpus:"; nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap --format=csv,noheader 2>&1 | head -4
echo "== disk:"; df -h / | tail -1
echo "== cudnn (faster-whisper needs 9):"; ldconfig -p | grep -m1 libcudnn || echo "  MISSING -- use a -cudnn- image"
echo "== ros:"; ls /opt/ros 2>/dev/null || echo "  none yet (deploy installs it)"
EOF
  ;;

deploy)
  need_ssh
  echo ">> [1/3] ROS 2 ${ROS_DISTRO} + system deps"
  rsh "bash -s" <<EOF
set -euo pipefail
. /etc/os-release
[ "\${VERSION_ID}" = "24.04" ] || { echo "need ubuntu 24.04 for ROS ${ROS_DISTRO}, got \${VERSION_ID} -- recreate with the image from '$0 plan'"; exit 1; }
if [ ! -d /opt/ros/${ROS_DISTRO} ]; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl gnupg lsb-release software-properties-common
  add-apt-repository -y universe
  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=\$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu \$(. /etc/os-release && echo \$UBUNTU_CODENAME) main" \
    > /etc/apt/sources.list.d/ros2.list
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    ros-${ROS_DISTRO}-ros-base ros-dev-tools python3-venv espeak-ng
fi
echo ROS_OK
EOF

  echo ">> [2/3] workspace + python deps -> ${STACK_DIR}"
  rsh "mkdir -p ${STACK_DIR}/ws/src"
  rcp "$REPO/ros/src/wojtek_agent_msgs" "$REPO/ros/src/wojtek_voice" \
      "$REPO/ros/src/wojtek_brain" "$REPO/ros/src/wojtek_agent_bringup" \
      "${STACK_DIR}/ws/src/"
  rsh "bash -s" <<EOF
set -euo pipefail
cd ${STACK_DIR}
# --system-site-packages so the venv sees rclpy (apt python3.12); torch 2.6+
# resolves to cu124 wheels from PyPI by default, which matches vast's older
# drivers (the unpinned-cu130 trap from the FutureNav deploy does not apply,
# but assert CUDA below anyway).
[ -d venv ] || python3 -m venv --system-site-packages venv
./venv/bin/pip install -q --upgrade pip
# chatterbox-tts, NOT the chatterbox-streaming fork: the fork is English-only
# (no mtl_tts module — verified live 2026-08-13); Polish needs multilingual.
./venv/bin/pip install -q faster-whisper websockets httpx chatterbox-tts fastapi 'uvicorn[standard]'
./venv/bin/python - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU -- do NOT serve"
print("cuda ok:", torch.cuda.get_device_name(0))
PYEOF
# Build with the venv python active so the installed node scripts get venv
# shebangs -- that is how the heavy deps become importable inside rclpy nodes.
set +u; source /opt/ros/${ROS_DISTRO}/setup.bash; set -u
source venv/bin/activate
cd ws && colcon build --symlink-install \
  --packages-select wojtek_agent_msgs wojtek_voice wojtek_brain wojtek_agent_bringup
echo WS_OK
EOF

  echo ">> [3/3] vLLM + ${BIELIK_MODEL} -> ${VLLM_DIR}"
  rsh "bash -s" <<EOF
set -euo pipefail
mkdir -p ${VLLM_DIR} && cd ${VLLM_DIR}
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
[ -d .venv ] || uv venv --python 3.12 .venv
# ninja is NOT a vllm dependency but its compile path shells out to it; without
# it the server dies with FileNotFoundError right after loading weights.
uv pip install --python .venv/bin/python -q -U vllm ninja huggingface_hub
.venv/bin/hf download ${BIELIK_MODEL}
echo VLLM_OK
EOF
  echo ">> deployed. next: $0 refs (optional voice clone), then $0 serve"
  ;;

refs)
  # Voice references never live in the repository (cloned character voices
  # are private-tinkering assets); they ride in from your machine at deploy
  # time.  Chatterbox uses the wav alone; F5 also needs the exact transcript.
  [[ -n "$TTS_REF_WAV" ]] || { echo "set TTS_REF_WAV=/path/to/ref.wav" >&2; exit 1; }
  rsh "mkdir -p ${STACK_DIR}/refs"
  rcp "$TTS_REF_WAV" "${STACK_DIR}/refs/ref.wav"
  if [[ -n "$TTS_REF_TEXT_FILE" ]]; then
    rcp "$TTS_REF_TEXT_FILE" "${STACK_DIR}/refs/ref.txt"
  fi
  echo ">> refs uploaded; serve picks them up automatically"
  ;;

serve)
  need_ssh
  rsh "bash -s" <<EOF
set -euo pipefail
cd ${VLLM_DIR}
if ! curl -sf localhost:${VLLM_PORT}/v1/models >/dev/null 2>&1; then
  export PATH="\$PWD/.venv/bin:\$HOME/.local/bin:\$PATH"
  # FlashInfer JIT-builds its sampler against the system nvcc; when their CUB
  # versions disagree the build fails and engine init dies.
  export VLLM_USE_FLASHINFER_SAMPLER=0
  setsid nohup .venv/bin/vllm serve ${BIELIK_MODEL} \
    --host 0.0.0.0 --port ${VLLM_PORT} --gpu-memory-utilization 0.35 \
    >> bielik.log 2>&1 < /dev/null &
  echo "bielik vllm starting on ${VLLM_PORT} (first load takes ~2 min)"
else
  echo "bielik vllm already up"
fi

cd ${STACK_DIR}
if pgrep -f "voice_stack.launch" >/dev/null 2>&1; then
  echo "voice stack already up"
else
  set +u
  source /opt/ros/${ROS_DISTRO}/setup.bash
  source ws/install/setup.bash
  set -u
  # colcon normalizes installed-script shebangs to the system python, which
  # cannot see the venv's model deps -- put the venv's site-packages on
  # PYTHONPATH instead (same CPython 3.12, so wheels are compatible).
  export PYTHONPATH="\$PWD/venv/lib/python3.12/site-packages:\${PYTHONPATH:-}"
  REF_WAV=""; REF_TEXT=""
  [ -f refs/ref.wav ] && REF_WAV="\$PWD/refs/ref.wav"
  [ -f refs/ref.txt ] && REF_TEXT="\$(cat refs/ref.txt)"
  setsid nohup ros2 launch wojtek_agent_bringup voice_stack.launch.py \
    ws_port:=${WS_PORT} vad_backend:=${VAD_BACKEND} \
    asr_model:=${ASR_MODEL} asr_language:=${ASR_LANGUAGE} \
    bielik_url:=http://127.0.0.1:${VLLM_PORT} bielik_model:=${BIELIK_MODEL} \
    tts_engine:=${TTS_ENGINE} tts_ref_wav:="\$REF_WAV" tts_ref_text:="\$REF_TEXT" \
    >> voice.log 2>&1 < /dev/null &
  sleep 8
  pgrep -f "voice_stack.launch" >/dev/null || { echo "LAUNCH DIED -- tail voice.log:"; tail -20 voice.log; exit 1; }
  echo "voice stack up (ws :${WS_PORT})"
fi
EOF
  ;;

tts-http)
  # Chatterbox+clone as HTTP for the ROOM DEMO (TTS_ENGINE=remote on the
  # laptop).  Reuses this stack's venv and refs; independent of the ROS nodes.
  need_ssh
  rcp "$REPO/training/wojtek_rl/agent/tts_server.py" "${STACK_DIR}/"
  rsh "bash -s" <<EOF
set -euo pipefail
cd ${STACK_DIR}
if ! curl -sf localhost:8120/health >/dev/null 2>&1; then
  setsid nohup env TTS_REF_WAV="\$PWD/refs/ref.wav" TTS_LANGUAGE=pl \\
    ./venv/bin/python tts_server.py --port 8120 >> tts_http.log 2>&1 < /dev/null &
  sleep 8
  pgrep -f tts_server.py >/dev/null && echo "tts http up on :8120" || { tail -10 tts_http.log; exit 1; }
else
  echo "tts http already up"
fi
EOF
  ;;

tunnel)
  need_ssh
  echo ">> ws://localhost:${WS_PORT} for mic.html; :${VLLM_PORT} for vLLM debugging. Keep open."
  ${VAST_SSH} -N \
    -L "${WS_PORT}:localhost:${WS_PORT}" \
    -L "${VLLM_PORT}:localhost:${VLLM_PORT}" \
    -L "8120:localhost:8120"
  ;;

status)
  rsh "bash -s" <<EOF
set -u
echo "== vllm:"; curl -sf localhost:${VLLM_PORT}/v1/models 2>/dev/null | head -c 300 || echo "down"; echo
echo "== nodes:"; pgrep -af "wojtek_(voice|brain)|voice_stack" | grep -v pgrep || echo "none"
echo "== gpu:"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1
EOF
  ;;

logs)
  rsh "tail -n 40 ${STACK_DIR}/voice.log ${VLLM_DIR}/bielik.log"
  ;;

stop)
  rsh "pkill -f 'voice_stack.launch' 2>/dev/null; pkill -f 'vllm serve ${BIELIK_MODEL}' 2>/dev/null; echo stopped"
  ;;

*)
  echo "usage: $0 plan|check|deploy|refs|serve|tunnel|status|logs|stop" >&2
  exit 1
  ;;
esac
