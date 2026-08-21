# Porting the agent stack to the DGX Spark (GB10)

What runs, what does not, and what it costs — measured on rented GB10 boxes
(vast, ~$0.34–0.39/hr against a prepaid balance), 2026-08-21.

The box: `aarch64`, 20× Cortex-X925, 121 GB unified, CUDA 13.0, driver
595.71.05, **compute capability sm_121** — not sm_120 like consumer
Blackwell. Reproduce any of this with `scripts/spark_stack_probe.sh`.

## Component status

| component | status | measured |
|---|---|---|
| **torch** (PyPI `2.13.0+cu130`) | ✅ works | 84.1 TFLOPS fp16 / 82.3 bf16 (4096³ matmul) |
| **Chatterbox TTS** | ✅ works | **RTF 0.44**, first piece 0.62–1.39 s |
| **vLLM** 0.27.1 | ✅ works | 0.5B model: load 242 s, **95.7 tok/s**, coherent Polish |
| **Qwen3-VL-4B-Instruct-FP8** (the demo brain) | ✅ with two flags | load 303 s; text turn **34.2 tok/s** (64 tok ≈ 1.9 s); vision turn **1.0 s warm** — but **82 s cold** |
| **whisper via transformers+CUDA** | ✅ works | large-v3: **RTF 0.084** (5 s audio in 0.42 s) |
| **faster-whisper / ctranslate2** | ❌ **CPU only** | wheel "not compiled with CUDA support"; tiny on CPU ≈ RTF 0.53 |
| **MuJoCo EGL rendering** | ❌ fails | `EGLError` even with `libegl1`+NVIDIA vendor json; osmesa also broken |
| ARM wheels generally | ✅ | librosa/llvmlite, s3tokenizer, conformer, diffusers — nothing compiled |

## The two problems, and what to do about them

**ASR must move off faster-whisper on this hardware.** `ctranslate2`'s
aarch64 wheel has no CUDA, so `wojtek_voice`'s ASR node would run whisper
large-v3 on 20 ARM cores — far slower than real time, on the critical path of
every spoken turn. The fallback is already measured and is *better* than the
old GPU path: `transformers` + CUDA runs large-v3 at **RTF 0.084**. Options,
cheapest first:

1. Add a transformers-backed engine to `wojtek_voice/asr_engine.py` and
   select it on aarch64 (keep faster-whisper for x86 dev boxes).
2. Build ctranslate2 from source with CUDA for aarch64 (slow, fragile).
3. NVIDIA Riva/Parakeet — different model, different Polish quality; needs
   its own evaluation.

Option 1 is the recommendation: one engine class behind the existing
protocol, and the number says it costs nothing in speed.

**MuJoCo renders on the Spark — 0.4 ms/frame — once EGL is pointed at the
right driver.** (Updated 2026-08-21, second session.) Two layers had to
align: the container must carry the NVIDIA EGL libraries
(`libEGL_nvidia.so`, present on hosts exposing graphics capability), and
glvnd must be pinned to the NVIDIA ICD —
`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
— because with mesa's ICD also installed, initialisation dies inside mesa's
dri2 path before the NVIDIA driver is tried, and mujoco surfaces that as an
opaque `EGLError`. room_app now sets the pin automatically when the NVIDIA
json exists. A leftover `EGLError` **at interpreter exit** is destructor
noise, not a render failure — check for frames, not for a clean shutdown.
This unblocks recording the demo takes on GB10, i.e. on the same hardware
the robot will carry.

## GB10 quirks that cost a session each (second visit, 2026-08-21)

- **Unified memory: the page cache eats "GPU" memory.** After the 8 GB model
  download, vLLM saw 35 GiB free of 121 and refused a 0.45 utilization ask.
  On GB10 size vLLM by *absolute* need, not fraction (0.22 fits the 4B with
  8k ctx comfortably), and `sync; echo 3 > /proc/sys/vm/drop_caches` before
  launch when the host allows it.
- **FP8 needs `VLLM_USE_DEEP_GEMM=0`.** DeepGEMM asserts "Unknown SF
  transformation" on sm_121; with it disabled the FP8 checkpoint loads and
  runs. Re-test on vLLM upgrades — this is a backend maturity gap, not a
  hardware one.
- **The FIRST vision turn costs 82 s** (vision tower warm-up/autotune); warm
  turns are ~1 s. Warmup must therefore include one image request, not just
  text — same policy as tts_server's spoken warmup, extended to the eyes.

Projected spoken turn on GB10, all measured parts: endpoint 0.7 s + ASR
~0.3 s + one warm vision turn ~1–2 s (+1 s per extra tool round trip) + TTS
first piece 0.6–1.4 s → **~3–4.5 s mic-to-voice** for a tool-using turn,
~2.5 s for chat-only, against ~6+ s on the August A6000 stack.

## Operational notes

- vast GB10 containers ship `/root/.ssh/authorized_keys` with wrong
  permissions and sshd refuses every key; `vastai execute` only works on
  STOPPED instances. Create with
  `--onstart-cmd 'chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys'`.
- `pip install vllm` collides with Debian's PyJWT: add
  `--ignore-installed PyJWT` (or use a venv, as the deploy scripts do).
- Never let pip replace torch. Install it from the cu130 index first, then
  everything else with `--no-deps` where a package pins a torch version.
- vLLM's 242 s cold load is worth designing around: warm it before a demo,
  exactly like `tts_server`'s warmup.

## Measurement hygiene

An earlier run reported 7.9 TFLOPS for the same matmul and prompted a "maybe
we need a vendor torch build" investigation. That number was an artifact of
timing the first kernels without warm-up; with three warm-up iterations the
same box does 84 TFLOPS. Warm before timing, or the conclusion is about
kernel compilation rather than throughput.
