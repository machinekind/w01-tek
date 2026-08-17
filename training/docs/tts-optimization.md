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
| L11 | Different engine entirely (Piper, Kokoro, XTTS-v2, F5) | **10× measured** | low | Loses the cloned voice — that is the whole trade. |

## Order to try them

Done already (shipped, unmeasured on GPU except where noted):

- **S1** reference conditionals encoded once — verify with `x-cond-ms`.
- **S7** tqdm bar off by default.
- **S10** line cache (`x-cache`), **warmup** at startup (57 s on an A6000,
  measured), **S8** no-op resample skip.
- **S2** available behind `--compile`; needs a GPU A/B before trusting.

Next, in order:

1. **S15 + S16** — one patched `T3.inference`: no attentions/hidden states
   per step, no per-token GPU sync. Biggest remaining safe win, hits the 80 %
   stage, and it is a contained vendored copy of one method.
2. **S2 measurement** — run `--compile` against the fixed line set; keep it
   only if the A/B says so.
3. **L2** — fewer flow-matching steps (the 20 % stage, one knob).
4. **L3** — FP8/int8 on T3, once the decode patch and compile are in.
5. **S11** — rent a faster card: 1.3–1.8× for money instead of engineering.
6. **L1 with a patched decode** — dropping the duplicated CFG batch is the
   other ~2×, but it changes the voice; A/B on the real stage lines.
7. **S9 + L6** — streaming and pipelining, deliberately *after* the above:
   they pay only once RTF < 1.
8. **S4** — TensorRT-LLM, if this becomes a permanent product path.
9. **L8 / L9 / S13** — distillation, pruning, speculative decoding.
   Research-grade; park them.
10. **L11** — swap engine. Last technically, first practically if the stage
    date is close and a generic voice is acceptable.

Rough stacking: S1 + S2 + L1 plausibly gives ~2.5–3× on the AR stage, i.e.
total RTF ≈ 0.5–0.6 — under 1.0, where streaming finally becomes a win
instead of a stutter.

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
