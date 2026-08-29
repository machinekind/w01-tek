#!/usr/bin/env bash
# Measure the cloned voice on a rented DGX Spark (GB10), in one paste.
#
# Why this script exists: every RTF number we have was taken on rented
# discrete GPUs -- 1.33-1.61 on an RTX A6000 (Ada), 0.86-0.90 on an RTX 5080
# (Blackwell).  The deployment target is a GB10 with roughly HALF the 5080's
# compute and a THIRD of its memory bandwidth, so neither number predicts it,
# and the difference decides whether streaming helps or stutters (see
# training/docs/tts-optimization.md).  Spark hours are rentable by the hour,
# so measure instead of extrapolating.
#
#   # on the rented Spark, as root:
#   curl -fsSL <this file> | bash          # or scp it over and run it
#   # or from your laptop:
#   ssh <spark> 'bash -s' < scripts/spark_tts_bench.sh
#
# Environment:
#   REF_WAV=/path/ref.wav   optional voice-clone reference (pace matters: a
#                           brisk reference produces less audio, and cost
#                           tracks generated DURATION -- lever L12)
#   OUT=/root/spark_bench.json  where the machine-readable result lands
#   REPO=https://github.com/<org>/<repo>  clone source; defaults to the copy
#                           already present if run from a checkout
set -euo pipefail

REF_WAV="${REF_WAV:-}"
OUT="${OUT:-/root/spark_bench.json}"
WORKDIR="${WORKDIR:-/root/wojtek_bench}"

say() { printf '\n=== %s\n' "$*"; }

say "what are we on?"
uname -m
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || \
  echo "nvidia-smi unavailable -- is this really a GB10 box?"
python3 - <<'PY'
import platform
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda,
          "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
          "capability", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "-")
except Exception as e:                       # noqa: BLE001 - report, do not fail
    print("torch missing or broken:", e)
print("python", platform.python_version(), platform.machine())
PY

# The Spark ships the NVIDIA AI stack (torch built for aarch64 + its CUDA).
# NEVER let pip replace it: chatterbox pins torch==2.6.0, which on this box
# would pull an incompatible wheel and lose the vendor build. --no-deps, then
# the dependencies by hand. Same trap cost a rental on sm_120 already.
say "installing chatterbox WITHOUT touching torch"
pip install -q --no-deps chatterbox-tts
pip install -q "librosa==0.11.0" s3tokenizer "transformers==4.46.3" \
  "diffusers==0.29.0" resemble-perth "conformer==0.3.2" safetensors omegaconf \
  pyloudnorm "numpy<2"
python3 -c 'import torch; print("torch after install:", torch.__version__)'

say "fetching the benchmark"
mkdir -p "$WORKDIR"
if [[ -f "$(dirname "$0")/../training/wojtek_rl/agent/tts_bench.py" ]]; then
  cp "$(dirname "$0")/../training/wojtek_rl/agent/tts_bench.py" "$WORKDIR/"
elif [[ -n "${REPO:-}" ]]; then
  git clone --depth 1 "$REPO" "$WORKDIR/repo"
  cp "$WORKDIR/repo/training/wojtek_rl/agent/tts_bench.py" "$WORKDIR/"
else
  echo "no tts_bench.py nearby and no REPO set; scp training/wojtek_rl/agent/tts_bench.py here" >&2
  exit 2
fi

say "measuring (first run downloads the voice model, a few GB)"
cd "$WORKDIR"
TQDM_DISABLE=1 python3 tts_bench.py \
  ${REF_WAV:+--ref "$REF_WAV"} \
  --json "$OUT" 2>&1 | grep -vE 'Sampling|it/s'

say "clock sweep: is this loop clock-bound or overhead-bound?"
# Needs privileges; the bench degrades to a warning when it cannot set them.
TQDM_DISABLE=1 python3 tts_bench.py --reps 1 \
  --clock-sweep 800,1200,1600,2000 2>&1 | grep -E 'MHz|cannot lock' || true

say "done -- copy $OUT back and paste the RTF line into docs/tts-optimization.md"
