# Making the cloned voice fast enough to stream

Working notes for anyone who picks up the TTS latency work.  The measured
numbers are in [latency.md](latency.md); this file is the option space, split
by whether an option changes what comes out of the speaker, and ordered by
what is worth trying first.

## Where the time is (measured, RTX A6000, 2026-08-17)

Chatterbox multilingual runs at **RTF 1.33–1.61** — it generates slower than
it plays.  The split, from the sampler rate in the server log (~23.5 it/s
against a 25 Hz speech-token rate, 92 tokens ≈ 3.9 s of the ~4.7–5 s call):

| stage | share of the budget |
|---|---|
| T3 — the autoregressive LM, text → speech tokens (Llama ~0.5B, batch 1) | **~80 %** |
| s3gen — flow matching → mel, then HiFiGAN → waveform | ~20 % |

**Optimise the LM.**  Anything that speeds up only the vocoder is capped at a
fifth of the total, and most of that fifth is cheap already.

The target is RTF < 1: below it, chunked streaming keeps the speaker fed
forever; above it, playback underruns after `B / (RTF − 1)` seconds of speech
(with a 1 s prebuffer at RTF 1.35, that is ~2.9 s — shorter than a typical
reply).  Streaming is a latency tool for engines that are already faster than
real time; it does not rescue a slow one.

## Measured on Blackwell (RTX 5080, torch 2.8+cu128, 2026-08-17)

The deployment target is a DGX Spark (GB10, Grace Blackwell), so benchmarks
moved to a Blackwell rental. This changed the conclusion completely:

| variant | median call | RTF | verdict |
|---|---|---|---|
| base (as shipped) | 3.44–3.60 s | **0.86–0.90** | already under 1.0 |
| S15 no hidden states per step | 3.61 s | 0.89 | **no gain** |
| S15+S16 analyzer off, SDPA restored (verified applied) | 3.58 s | 0.89 | **no gain** |
| S2 `torch.compile(reduce-overhead)` | 3.60 s | 0.92 | **slightly worse** |

## MEASURED ON THE TARGET (DGX Spark, GB10, 2026-08-21)

Settled by renting an actual Spark for $0.04. **RTF 0.44** — the cloned voice
generates at more than twice real time on the deployment hardware, so
clause-level streaming is comfortable and cannot underrun.

| | A6000 (Ada) | RTX 5080 (Blackwell) | **GB10 (target)** |
|---|---|---|---|
| RTF | 1.33-1.61 | 0.86-0.90 | **0.44** |
| torch | 2.6 / cu124 | 2.8 / cu128 | 2.13 / cu130 |

First-piece latency, which is what a listener actually waits for:

| first piece | GPU time | audio |
|---|---|---|
| `Szukam kanapy,` | **0.62 s** | 1.04 s |
| `Widzę kanapę.` | 1.32 s | 2.76 s |
| `Już się zatrzymuję!` | 1.39 s | 3.00 s |

**Caveat on the comparison**: the torch version moved with the hardware
(2.8/cu128 on the 5080, 2.13/cu130 on the Spark), so the three columns are
not a clean hardware A/B. The absolute number on the target is what matters,
and it is 0.44.

### What the box actually is

- `aarch64`, **20x Cortex-X925**, 121 GB unified (torch reports 130.6 GB
  total), driver 595.71.05, CUDA 13.0, Python 3.12.
- **Compute capability 12.1 (sm_121)** -- NOT sm_120 like consumer Blackwell.
- PyPI `torch==2.13.0+cu130` installs and works, but its `arch_list` is
  `[sm_80, sm_90, sm_100, sm_110, sm_120]` -- **no sm_121 cubins**. Kernels
  still run (CUDA family compatibility covers sm_120 -> sm_121) and they are
  fast: **84.1 TFLOPS fp16 / 82.3 bf16** on a 4096^3 matmul. (An earlier run
  reported 7.9 TFLOPS and sent us looking for a vendor torch build; that was
  a no-warm-up artifact -- time kernels after warming them, or you measure
  compilation.)
- Every dependency has an aarch64 wheel: librosa/llvmlite, s3tokenizer,
  conformer, transformers, diffusers. Nothing compiled from source.
- Install recipe that works: `pip install torch --index-url
  https://download.pytorch.org/whl/cu130`, then chatterbox `--no-deps` plus
  its dependencies with `transformers==4.46.3` pinned. Never let pip touch
  torch.
- vast.ai host quirk: the container's `/root/.ssh/authorized_keys` had wrong
  permissions and sshd refused every key. `vastai execute` only works on
  STOPPED instances, so the fix is to recreate with
  `--onstart-cmd 'chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys'`.

### Superseded: the extrapolation that worried us (kept, because it was wrong)

Before the Spark was measured, the reasoning below predicted RTF 1.3-1.8 on
GB10 by scaling the 5080 result down by core count and bandwidth. The real
answer came out **three to four times better** than that scaling suggested,
which is the case for measuring rather than extrapolating -- in both
directions.

The 5080 got the *architecture and toolchain* right (sm_120, torch 2.8+cu128,
the package pins below), which is what makes the packaging finding
trustworthy. It is NOT a speed proxy for GB10:

| | RTX 5080 (measured) | GB10 (target, approx. public specs) |
|---|---|---|
| CUDA cores | ~10 750 | ~6 100 |
| memory bandwidth | ~960 GB/s GDDR7 | **~273 GB/s LPDDR5X** |
| host CPU | x86 | 20× ARM (Grace) |

Roughly half the compute and a third of the bandwidth, so a
compute-proportional scaling would put GB10 near **RTF 1.3–1.8 — back above
1.0**. Measured: 0.44. The scaling argument was simply wrong.

What is known: decode ran at ~30 tokens/s reading ~1 GB of weights per token,
i.e. ~30 GB/s effective — **3 % of the 5080's bandwidth**, so this workload is
not bandwidth-bound, which is the axis where the Spark is weakest. Against
that, GB10 has fewer SMs at lower clocks, and the decode loop is a Python
loop with a GPU sync per token, which Grace's weaker single-thread
performance will punish harder than an x86 host does.

Settled on 2026-08-21 with `./training/run.sh tts-bench` on a rented GB10 —
see the target section above.

**A Spark does not have to be owned to be measured.** GB10 boxes rent by the
hour with full root SSH (~$0.60–0.80/hr as of 2026-08: Enverge Spark private
beta, SPARK Hosting, and individuals on the NVIDIA developer forums), which
is cheaper than the discrete-GPU rentals this file already quotes.
Rentals bill until the instance is DELETED (Enverge documents no spend cap,
no idle auto-stop, and no prepaid credits, and a "restart" keeps the meter
running), so arm `scripts/rental_watchdog.sh` before starting one and put a
hard limit on the card that pays for it. The watchdog is a second line, not
enforcement: it runs on a laptop, so a sleeping Mac protects nothing.

`scripts/spark_tts_bench.sh` is the whole session in one paste: it reports
what the box actually is, installs Chatterbox **without replacing the
vendor's aarch64 torch** (chatterbox pins `torch==2.6.0`; letting pip act on
that would destroy the NVIDIA build), runs the benchmark, and finishes with
the clock sweep. Account signup and payment are the user's to do; the script
assumes an SSH prompt already exists.

**The same engine is RTF 0.90 on a $0.08/hr RTX 5080 and RTF 1.35 on an
A6000.** Hardware generation, not our patches, is what crosses the
streaming threshold — and the micro-optimisations that looked compelling in
the source are worth nothing measurable at these sequence lengths. Decode is
dominated by the per-layer GEMMs of a 30-layer 0.5B model, not by attention
kernels, hidden-state allocation, or launch overhead.

**Do not implement the vendored patched decode (S15/S16).** It was measured
and it does not pay. Same for `--compile`: it exists behind a flag, it did
not help here, leave it off unless a future card says otherwise.

First-piece latency, which is what clause-level streaming actually exposes:

| first piece | GPU time | audio produced |
|---|---|---|
| `Szukam kanapy,` | 1.90 s | 1.96 s |
| `Widzę kanapę.` | 3.02 s | 2.20 s |
| `Już się zatrzymuję!` | 3.83 s | 4.32 s |

Note the third row: 19 characters produced **4.32 s of audio**. Cost tracks
*generated audio duration*, and the duration is set by how the model paces
speech — which Chatterbox copies from the reference clip. A brisk reference
should cut both the audio length and the GPU time proportionally (L12); this
is untested and is the cheapest experiment left.

## What the source actually does (chatterbox-tts 0.1.7)

Read before trusting any estimate below — three of them changed once the
package source was read rather than recalled.

- `generate(audio_prompt_path=...)` calls `prepare_conditionals` **on every
  call**: librosa load + resample of the reference, `s3gen.embed_ref`, the S3
  tokenizer and the voice encoder. Fixed here (S1) — the server prepares the
  voice once and then omits the argument.
- **`cfg_weight=0` does NOT halve the compute.** `text_tokens` and the
  per-step embedding are duplicated unconditionally
  (`torch.cat([text_tokens, text_tokens])`); the weight only changes how the
  two logit streams are combined. Halving the batch needs a patched decode,
  not a flag.
- The decode loop passes **`output_attentions=True, output_hidden_states=True`
  on every step**. Attention outputs force the slow attention path (no
  SDPA/flash) and materialise B×H×L×L per layer per token; hidden states are
  never read. The attentions are only needed when the alignment analyzer is
  active. This is probably the largest single safe win available, and it
  needs a patched `T3.inference` (S15).
- The loop syncs the GPU every token (`generated_ids[0, -1].item()`, plus the
  EOS comparison), which stalls the pipeline once per 40 ms of audio (S16).
- `max_new_tokens=1000` is hardcoded, not a parameter — capping it (L4) means
  patching, and it is what the `Sampling: n/1000` bar counts.

## Safe — same output distribution, no perceptual change

| # | Lever | Est. gain | Effort | Notes |
|---|---|---|---|---|
| S1 | Cache the reference conditionals (speaker embedding + prompt tokens/mel per ref wav) | per-request re-encode removed | low | **DONE.** Prepared once at warmup; `x-cond-ms` is non-zero exactly once. If it is ever non-zero twice, this regressed. |
| S2 | `torch.compile(mode="reduce-overhead")` + static KV cache on T3 decode | 1.5–2.5× on AR | low–med | **Flag added** (`--compile`, guarded, falls back to eager). Unmeasured — needs a GPU A/B. Static KV cache would need a patched decode. |
| S15 | Stop requesting attentions/hidden states per decode step | likely large | med | Needs a patched `T3.inference`. Attentions are only required when the alignment analyzer is on; hidden states never. Restores the SDPA/flash path. |
| S16 | Remove the per-token GPU syncs (`.item()`, EOS compare) from the decode loop | moderate | med | Same patch as S15. Check EOS every N steps, keep the last token on device. |
| S3 | Explicit CUDA Graphs | folded into S2 | med | Only if launch overhead survives S2. |
| S4 | TensorRT-LLM for T3 (+ TRT engines for flow/vocoder) | 2–3× on AR | high | Highest safe ceiling; you own the export pipeline. |
| S5 | FlashAttention-2 / SDPA kernels | small | low | Sequences are short; attention is not the bottleneck at 25 Hz. |
| S6 | bf16/fp16 end to end, no stray fp32 | 1.2–2× if currently fp32 | low | Check what `from_pretrained` actually loads on cuda. |
| S7 | Kill per-step Python overhead: disable the tqdm sampler bar | 2–8 % | trivial | **DONE** — `TQDM_DISABLE=1` unless `--progress`. The bar printed ~24×/s. |
| S8 | Skip redundant resample / WAV encode | ~1 % | trivial | s3gen is already 24 kHz, so `resample_linear` should be a no-op; the stream endpoint already returns raw PCM. |
| S9 | Pipeline the stages — vocode chunk *k* while the LM generates *k+1* | up to 20 % wall | med | Only meaningful once streaming is on. |
| S10 | Pre-render + cache fixed lines | ∞ on repeats | **done** | `tts_server` LRU + `x-cache` header. Pre-warm with the stage script's fixed phrases. |
| S11 | Faster GPU (4090 / H100) | 1.3–1.8× | zero eng., costs money | Batch-1 AR is clock/latency bound, not FLOP bound. |
| S12 | Batch concurrent utterances | throughput only | low | Does nothing for a single reply's first audio. |
| S13 | Speculative decoding (draft model + rejection sampling) | 1.5–2× theoretical | high | Lossless by construction, but needs a draft model for *speech* tokens. Unproven here. |
| S14 | Drop the Perth watermarker from the hot path | ~1 % | done-ish | CPU-side and negligible. It is a provenance/policy decision, not a speed lever — treat it as one. |

## Lossy — changes what comes out

| # | Lever | Est. gain | Effort | Quality risk |
|---|---|---|---|---|
| L1 | `cfg_weight=0` (drop classifier-free guidance) | **~0 as a flag**; ~2× only with a patched decode | one line / med | The batch is duplicated unconditionally in 0.1.7, so the flag alone changes quality without buying speed. Exposed as `--cfg-weight` for the quality A/B; the speed needs the patch. |
| L2 | Fewer flow-matching steps (NFE) | near-linear on the 20 % | one line | Artifacts, muddier high end. |
| L3 | FP8 / int8 weights on T3 (AWQ/GPTQ/SmoothQuant; FP8 native on Ada) | 1.5–2× | med | Usually small at 0.5B, but TTS is sensitive to tail behaviour. |
| L4 | Cap max tokens / tighten EOS | cuts long tails | low | Truncation mid-word; the alignment guard already forces EOS on long tails. |
| L5 | Shorter reference clip | prefill + conditioning | trivial | Slightly worse clone fidelity. |
| L6 | Chunked streaming with limited lookahead | latency, not compute | med | Boundary seams, prosody discontinuity. |
| L7 | Swap vocoder (Vocos, smaller HiFiGAN) | ≤10 % total | med | Poor return: only 20 % of the budget is downstream. |
| L8 | Distil the flow decoder to few-step / one-step (consistency, rectified flow) | most of the 20 % | high (training) | Research effort for a fifth of the budget. |
| L9 | Prune / distil / shrink T3 | large | high (training) | Best theoretical payoff, worst effort, the voice may shift. |
| L10 | Lower sample rate / mel resolution | small | low | Audible on a PA system. |
| L11 | Different engine entirely (Piper, Kokoro, XTTS-v2, F5) | **10× measured** | low | Loses the cloned voice — that is the whole trade. Much less pressing now that Blackwell puts Chatterbox under RTF 1.0. |
| L12 | **Brisker reference clip** so the model paces speech faster | proportional to audio length | trivial | Untested and the cheapest experiment left: 19 chars generated 4.32 s of audio, and cost tracks generated duration. Chatterbox copies the reference's pace, so a quick-speaking reference should cut GPU time by the same factor. Changes how the dog sounds. |

## The plan

Status of every lever. Tick a box only when a number backs it; write the
number in Result, and say where it was measured — a win on one card is not a
win on another, which is the lesson of this whole file.

### Shipped

- [x] **S1 · reference voice encoded once** — Result: `prepare_conditionals`
  no longer runs per request; `x-cond-ms` non-zero exactly once. Not yet
  timed on GPU with a real reference (our benches ran ref-less), so the size
  of the win is unquantified — but it is strictly less work.
- [x] **S7 · tqdm bar off** (`--progress` restores it) — Result: not
  separately measurable above noise; free, kept.
- [x] **S8 · skip the no-op resample** — Result: ~1 %, free.
- [x] **S10 · line cache + startup warmup** — Result: repeated line costs the
  GPU **zero** (`x-cache: hit`); warmup absorbs a **57 s** model load (A6000)
  that used to land on the first spoken line.
- [x] **first-chunk-early splitting + clause streaming** — Result: long reply
  **6.42 s → 3.32 s** to first sound (A6000); short replies unchanged, their
  floor is the engine.

### Measured and rejected — do not redo these

- [x] **S15 · no hidden states per decode step** — Result: RTF 0.89 vs 0.90
  base (RTX 5080). **No gain.** Looked compelling in the source; is not.
- [x] **S16 · analyzer off, SDPA restored** (patch verified applied:
  `attn: sdpa, analyzer: False`) — Result: RTF 0.89 vs 0.90. **No gain**, and
  it costs the long-tail EOS guard. Rejected.
- [x] **S2 · `torch.compile(reduce-overhead)`** — Result: RTF **0.92**,
  slightly *worse* than eager, 3 s compile warm. Flag kept (`--compile`),
  default off. Re-test only if a future card says otherwise.
- [x] **L1 · `cfg_weight=0` as a flag** — Result: **buys no speed**; the
  batch is duplicated unconditionally in 0.1.7, so the flag only changes
  quality. Exposed as `--cfg-weight` for voice A/B, not for latency.

### Open, in priority order

- [x] **Verify RTF on the actual target (GB10)** — Result: **0.44**, first
  piece 0.62–1.39 s, measured on a rented Spark for $0.04. Streaming is
  comfortable; nothing below is needed to make the voice viable.
- [ ] **L12 · brisker reference clip.** Cheapest untried lever: 19 characters
  produced **4.32 s of audio**, cost tracks generated duration, and
  Chatterbox copies the reference's pace. Needs one (ideally two) reference
  wavs — they are the user's private assets, so this is blocked on a file.
- [ ] **Live read→heard on Blackwell**, server + client end to end with the
  cache warm, rather than model-only timings. Expect ~1.9–3.0 s first piece.
- [x] **Check whether torch is leaving performance on the table** — Result:
  no. 84.1 TFLOPS fp16 with warm-up; the 7.9 figure was a measurement
  artifact. PyPI cu130 is fine, no vendor build needed.
- [ ] **L2 · fewer flow-matching steps** — cheap, but no longer needed for
  viability; only if we want headroom.
- [ ] **L3 · FP8 on T3** — Blackwell has native FP8. Demoted: at RTF 0.44 the
  voice is no longer the constraint.
- [ ] **L1 with a patched decode** — halving the duplicated CFG batch is the
  only structural change still worth the risk. Changes the voice; A/B it.
- [ ] **S4 · TensorRT-LLM** — only if this becomes a permanent product path.
- [ ] **L8 / L9 / S13** — flow distillation, pruning, speculative decoding.
  Research-grade; parked.
- [ ] **L11 · swap engine** — pressure depends entirely on the GB10 number.
  If the Spark lands above ~1.2 RTF, this returns to the top of the list.

Rough stacking: S1 + S2 + L1 plausibly gives ~2.5–3× on the AR stage, i.e.
total RTF ≈ 0.5–0.6 — under 1.0, where streaming finally becomes a win
instead of a stutter.

## Answering "is it fast enough HERE" on a new machine

```bash
./training/run.sh tts-bench                          # RTF over the fixed lines
./training/run.sh tts-bench --ref ~/refs/voice.wav   # with the cloned voice
./training/run.sh tts-bench --clock-sweep 1000,1500,2000,2500
```

It prints the RTF, the first-piece latencies, and what the number means
("streaming is comfortable" / "underruns after N s of speech"). `--json`
keeps the run for comparison. The clock sweep answers whether the loop is
clock-bound (RTF scales with MHz) or overhead-bound (flat) — which is how to
predict a slower card without owning one.

## How to A/B any of them

Everything above is measurable with what is already in the repo, so change
ONE thing at a time and compare against the same lines:

```bash
# per-request server-side truth: x-synth-ms / x-queue-ms / x-audio-ms / x-cache
curl -s -D- -o /dev/null -X POST localhost:8120/synthesize \
  -H 'content-type: application/json' -d '{"text": "Widzę kanapę."}'
```

```bash
# what a person waits: reply.text_to_sound, first-chunk size, RTF
./training/run.sh perf runs/agent_traces/<trace>.jsonl
```

The fixed comparison set used for the 2026-08-17 numbers (short, medium and
long replies, because cost tracks generated tokens rather than input
characters):

- `Widzę kanapę. Hau hau!`
- `Już się zatrzymuję!`
- `Szukam kanapy, ale nie widzę jej teraz – idę w stronę, gdzie są półki i meble.`
- `Jestem teraz przy stole w kuchni, patrzę w stronę okna. Zaraz sprawdzę co jest dalej.`

Estimates in the tables are exactly that — estimates, apart from the measured
rows (S10, L11) and the budget split above. Replace them with numbers as they
are measured.
