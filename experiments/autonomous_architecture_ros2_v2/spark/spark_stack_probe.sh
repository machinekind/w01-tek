#!/usr/bin/env bash
# Does the REST of the agent stack run on a DGX Spark?
#
# The voice was measured at RTF 0.44 on GB10 (docs/tts-optimization.md); the
# open question is everything else, because the Spark is aarch64 with
# capability sm_121 and most ML wheels are built for x86 and sm_120 or below.
# A wheel without sm_121 cubins may still run through CUDA family
# compatibility -- or may not run at all -- and the only way to know is here.
#
#   ssh <spark> 'bash -s' < scripts/spark_stack_probe.sh
#   ONLY=vllm ssh <spark> 'bash -s' < scripts/spark_stack_probe.sh
#
# Each section is time-boxed and independent: a component that cannot install
# reports FAIL and the probe moves on, because "which parts are broken" is
# the whole output. Nothing here is fast; budget ~30-45 minutes.
set -uo pipefail                      # NOT -e: a failing check is a result

ONLY="${ONLY:-all}"
PIP="pip3 install -q --break-system-packages"
want() { [[ "$ONLY" == all || "$ONLY" == "$1" ]]; }
say() { printf '\n========== %s\n' "$*"; }
ok()  { printf 'RESULT %-14s OK    %s\n' "$1" "${2:-}"; }
bad() { printf 'RESULT %-14s FAIL  %s\n' "$1" "${2:-}"; }

say "inventory"
uname -m
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
python3 -V

if want torch; then
  say "torch: does it exist for this arch, and how fast is a plain matmul?"
  $PIP torch --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -2
  $PIP 'numpy<2' 2>&1 | tail -1
  python3 - <<'PY' && ok torch || bad torch
import time, torch
print("torch", torch.__version__, "| cap", torch.cuda.get_device_capability(0))
print("arch_list", torch.cuda.get_arch_list())
print("sm_121 cubins:", "sm_121" in torch.cuda.get_arch_list())
for dt in (torch.float16, torch.bfloat16):
    x = torch.randn(4096, 4096, device="cuda", dtype=dt)
    y = torch.randn(4096, 4096, device="cuda", dtype=dt)
    for _ in range(3): x @ y                      # warm
    torch.cuda.synchronize(); t0 = time.monotonic()
    for _ in range(20): z = x @ y
    torch.cuda.synchronize()
    tf = 20 * 2 * 4096**3 / (time.monotonic() - t0) / 1e12
    print(f"{str(dt).split('.')[-1]:>9} matmul: {tf:6.1f} TFLOPS")
PY
fi

if want whisper; then
  say "faster-whisper (ctranslate2): ASR is on the critical path"
  # ctranslate2 is the risk: a C++ extension that needs an aarch64 CUDA wheel.
  timeout 600 $PIP faster-whisper 2>&1 | tail -2
  python3 - <<'PY' && ok whisper || bad whisper
import time, numpy as np, wave, tempfile, os
from faster_whisper import WhisperModel
sr = 16000
pcm = (np.sin(np.linspace(0, 400, sr * 3)) * 6000).astype(np.int16)   # 3 s tone
path = os.path.join(tempfile.gettempdir(), "probe.wav")
with wave.open(path, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
for device in ("cuda", "cpu"):
    try:
        t0 = time.monotonic()
        m = WhisperModel("tiny", device=device, compute_type="int8" if device == "cpu" else "float16")
        load = time.monotonic() - t0
        t0 = time.monotonic()
        list(m.transcribe(path)[0])
        print(f"{device}: load {load:.1f}s, transcribe 3s audio in {time.monotonic()-t0:.2f}s")
    except Exception as e:
        print(f"{device}: FAILED {type(e).__name__}: {str(e)[:120]}")
PY
fi

if want vllm; then
  say "vLLM: the brain server (Qwen3-VL + Bielik) -- the biggest unknown"
  timeout 1200 $PIP vllm 2>&1 | tail -3
  python3 - <<'PY' && ok vllm-import || bad vllm-import
import vllm
print("vllm", vllm.__version__)
from vllm import LLM  # noqa: F401
print("import OK")
PY
  # Serving is the real test; a tiny model keeps the download honest.
  say "vLLM: can it actually serve?"
  timeout 900 python3 - <<'PY' && ok vllm-serve || bad vllm-serve
import time
from vllm import LLM, SamplingParams
t0 = time.monotonic()
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=2048,
          gpu_memory_utilization=0.25, enforce_eager=True)
print(f"loaded in {time.monotonic()-t0:.0f}s")
t0 = time.monotonic()
out = llm.generate(["Czym jest robot?"], SamplingParams(max_tokens=64, temperature=0))
dt = time.monotonic() - t0
n = len(out[0].outputs[0].token_ids)
print(f"generated {n} tokens in {dt:.2f}s = {n/dt:.1f} tok/s")
PY
fi

if want mujoco; then
  say "mujoco headless render (the demo's sim, EGL on ARM)"
  $PIP mujoco 2>&1 | tail -1
  MUJOCO_GL=egl python3 - <<'PY' && ok mujoco || bad mujoco
import time, mujoco, numpy as np
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><geom size='.1'/></body></worldbody></mujoco>")
d = mujoco.MjData(m)
r = mujoco.Renderer(m, height=480, width=640)
t0 = time.monotonic()
for _ in range(20):
    mujoco.mj_step(m, d); r.update_scene(d); frame = r.render()
print(f"render 640x480: {(time.monotonic()-t0)/20*1000:.1f} ms/frame, shape {np.shape(frame)}")
PY
fi

say "summary"
echo "grep the RESULT lines above; FAIL means that component needs work on aarch64/sm_121"
