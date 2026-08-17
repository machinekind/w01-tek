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

## Safe — same output distribution, no perceptual change

| # | Lever | Est. gain | Effort | Notes |
|---|---|---|---|---|
| S1 | Cache the reference conditionals (speaker embedding + prompt tokens/mel per ref wav) | possibly large | low | Chatterbox recomputes `prepare_conditionals` whenever `audio_prompt_path` is passed, and `tts_server.py` passes it on every request. **Verify first** — may be free. |
| S2 | `torch.compile(mode="reduce-overhead")` + static KV cache on T3 decode | 1.5–2.5× on AR | low–med | Batch-1 decode is kernel-launch bound; the canonical fix. Needs static shapes. |
| S3 | Explicit CUDA Graphs | folded into S2 | med | Only if launch overhead survives S2. |
| S4 | TensorRT-LLM for T3 (+ TRT engines for flow/vocoder) | 2–3× on AR | high | Highest safe ceiling; you own the export pipeline. |
| S5 | FlashAttention-2 / SDPA kernels | small | low | Sequences are short; attention is not the bottleneck at 25 Hz. |
| S6 | bf16/fp16 end to end, no stray fp32 | 1.2–2× if currently fp32 | low | Check what `from_pretrained` actually loads on cuda. |
| S7 | Kill per-step Python overhead: disable the tqdm sampler bar, avoid per-call retokenisation, no `.cpu().numpy()` syncs inside the loop | 2–8 % | trivial | The `Sampling: n/1000` bar prints ~24×/s in the server log. |
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
| L1 | `cfg_weight=0` (drop classifier-free guidance) | **~2× on AR** | one line | Real: weaker prompt adherence and expressiveness. A/B on the actual stage lines. |
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

1. **S1** — reference-conditional caching. Check first; possibly free.
2. **S2** — `torch.compile` + static KV cache. Biggest reliable safe win, hits the 80 %.
3. **L1** — `cfg_weight=0`. One line, ~2× on the dominant stage; cheapest experiment here.
4. **S6/S7** — precision audit, kill the progress bar and the CPU syncs.
5. **L2** — fewer flow steps.
6. **L3** — FP8/int8 on T3, once compile is in place.
7. **S11** — rent a faster card: buy 1.3–1.8× with money instead of engineering.
8. **S9 + L6** — streaming and pipelining, deliberately *after* the above: they pay only once RTF < 1.
9. **S4** — TensorRT-LLM, if this becomes a permanent product path.
10. **L8 / L9 / S13** — distillation, pruning, speculative decoding. Research-grade; park them.
11. **L11** — swap engine. Last technically, first practically if the stage date is close and a generic voice is acceptable.

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
