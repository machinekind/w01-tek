# Polish voice stack — research report and decision record

Research round of 2026-08-07: what we picked, and everything we rejected.
Every claim was verified against model cards, technical reports and the papers
themselves rather than blog posts. This report is the decision record for the
realtime Polish voice stack; the implementation it produced is described in
[agent.md](agent.md).

**The one-line conclusion:** no self-hostable model hears *and* speaks Polish,
so the stack is a cascade — Whisper hears, the brain thinks in English, a
Polish TTS speaks. Everything below is why.

---

## The shipped stack

| stage | choice | licence | cost | why |
|---|---|---|---|---|
| **ASR** | `faster-whisper large-v3`, int8 | MIT | ~2.5 GB, CPU or GPU | best-measured open Polish recogniser |
| **Brain** | `Qwen/Qwen3-VL-4B-Instruct-FP8` | Apache-2.0 | ~8 GB | vision + tools; text-only Polish exposure |
| **TTS** | Piper `pl_PL-gosia-medium` | MIT voice / CC0 data | 0 GB, CPU | measured RTF 0.18 on a laptop |
| **Language policy** | English everywhere; Polish only in the spoken sentence | — | free | see [agent.md](agent.md) |

Measured end to end, laptop CPU + remote 4B brain: **2.2 s** from end of
speech to recognised text, **2.7 s** to first reply audio, barge-in cuts
playback in **90 ms**.

---

## 1. Omni models — why none of them work

An "omni" model that hears and speaks directly would remove two stages. None
can do it in Polish.

### Qwen family

| model | speech in | speech out | Polish | weights |
|---|---|---|---|---|
| **Qwen3.5-Omni** Plus/Flash | 74 langs | **29 langs + 7 dialects** | ✅ **both** (FLEURS pl **1.9 WER**, speech-gen 1.427) | ❌ **API only** |
| Qwen3-Omni-30B-A3B | 19 langs | 10 langs | ❌ neither | ✅ Apache-2.0 |
| Qwen2.5-Omni-7B / 3B | limited | **en + zh only** | ❌ | ✅ / ⚠️ 3B is research-only |

Qwen3-Omni's Talker speaks `de en es fr it ja ko pt ru zh`. Russian is there,
Polish is not. Its *text* side covers 119 languages, so Polish text is fine —
which is precisely why it is usable as a brain but not as a mouth.

### Everything else with open weights

Verified card by card. **Not one supports Polish speech output.**

| model | licence | speech out | languages | duplex | first audio |
|---|---|---|---|---|---|
| **MiniCPM-o 4.5** (9B) | ✅ Apache-2.0 | native interleaved | zh/en | ✅ real | TTFT 0.6 s |
| **Moshi / Moshiko** (7–8B) | ✅ CC-BY-4.0 | native, 24 GB | **en only** | ✅ | **200 ms** in practice |
| Freeze-Omni | ⚠️ Apache + Tencent AUP | AR single-codebook | zh + en | ✅ | 745 ms avg |
| LLaMA-Omni2 (0.5–32B) | ⚠️ academic only | CosyVoice2 tokens | en, or en+zh | ❌ | 457–663 ms |
| LLaMA-Omni | ⚠️ academic only | HuBERT units | en (LJSpeech) | ❌ | 236 ms |
| Step-Audio-2-mini (8.3B) | ✅ Apache-2.0 | native | zh/en | — | unpublished |
| Mini-Omni / Mini-Omni2 | ✅ MIT | native SNAC | **en only, explicitly** | claimed, not delivered | unpublished |
| SLAM-Omni | ✅ MIT (en) / ⚠️ GPL zh | native codec | zh + en | ❌ | none |
| LFM2.5-Audio (1.5B) | ⚠️ $10M revenue cap | native Mimi | **en only** | — | <100 ms |
| GLM-4-Voice-9B | ⚠️ registration required | native interleaved | zh/en | ❌ | — |
| Baichuan-Omni-1.5 | ⚠️ DAU<1M, revocable | native RVQ | zh/en | ❌ | — |
| Ming-Lite-Omni | ✅ MIT | native | zh/en | ❌ | — |
| Kimi-Audio-7B | ✅ MIT | native + BigVGAN | zh/en | ❌ | — |

Mini-Omni's FAQ states the pattern plainly: *"the model is only trained on
English… the output is only in English."* The architectural reason is
consistent — every one of these learns discrete audio tokens from English or
English+Chinese corpora. Whisper encoders mean several of them *hear* Polish
somewhat; none can *speak* it.

**The one exception, on a technicality:** `facebook/seamless-m4t-v2-large`
(2.3B) does take Polish speech in and emit Polish speech out — `pol | Sp, Tx
| Sp, Tx` in its own table, `pol` present in `vocoder_lang_code_to_id`. But it
is **CC-BY-NC** and it is a *translation* model: no instruction following, no
dialogue, no tools. Not a brain.

**Reality check on latency claims:** at the ICASSP 2026 HumDial challenge,
measured conversational turn-taking delay was **2.5–2.9 s** for Freeze-Omni
and Moshi — an order of magnitude off the published "200 ms first audio"
figures. Vendor latency is first-packet on vendor hardware, not time-to-reply.

---

## 2. ASR — hearing Polish

### Picked: `faster-whisper large-v3`, int8

Best measured open Polish recogniser. Same weights as OpenAI Whisper, ~4×
faster via CTranslate2, and int8 drops VRAM from ~4.5 GB to **~2.5 GB** at
roughly fp16 speed.

| benchmark | whisper-large-v3 | Canary-1B-v2 | Parakeet-TDT-0.6B-v3 | seamless-m4t-v2 |
|---|---|---|---|---|
| FLEURS pl | **4.74** | 6.64 | 7.31 | 8.76 |
| MLS pl | 8.88 | 8.77 | **7.28** | 5.74 |

On real-world Polish (PolEval 2024 / BIGOS, spontaneous speech) vanilla
large-v3 scores **14.51 % WER** — the best of 24 evaluated off-the-shelf
systems. The PolEval winner, a Whisper-large-v3 fine-tune, reached 11.07 %;
its weights were never published.

### The upgrade worth benchmarking

**`yuriyvnv/parakeet-tdt-0.6b-polish`** (627M, CC-BY-4.0) reports **6.07 % WER
val / 11.81 % test** on BIGOS v2 + CV17 — better than large-v3 on Polish, at
a tenth the size. Caveats: published by an individual rather than an
institution, and it needs the NeMo runtime rather than faster-whisper, so
swapping it in is real integration work. Benchmark before believing it.

### Rejected

| option | why not |
|---|---|
| Qwen3-ASR-1.7B | MLS-pl **15.26** vs Whisper's 8.88; Polish is its weakest European language and appears in none of its per-language tables |
| Qwen Omni built-in encoder | Polish is not among its 19 speech languages — no supported path at all |
| `nemotron-3.5-asr-streaming` | genuinely streaming (80 ms chunks) but **15–20 % WER** on Polish |
| Canary-1B / -flash | en/de/es/fr only — **no Polish** |
| Voxtral (Mistral) | no Polish in any variant |
| WhisperX | batch alignment/diarisation tool, wrong shape for a realtime loop |
| Polish Whisper fine-tunes on HF | none has traction (≤64 downloads) or a published WER |

Worth a look if latency ever dominates: `CohereLabs/cohere-transcribe-03-2026`
(2B, Apache-2.0) lists Polish and beats large-v3 on the English leaderboard;
its Polish figure exists only in a chart image, so it is unverified.

**Realtime note:** Polish forces chunked-Whisper latency (~500–800 ms with
sliding-window VAD segmentation). The only natively streaming Polish-capable
model is Nemotron, and it is far too inaccurate. `fixie-ai/ultraVAD` is worth
adopting for turn-end detection — **65–110 ms**, F1 81.3, 26 languages
including `pl` — though its licence is undeclared.

---

## 3. TTS — speaking Polish

### Picked: Piper `pl_PL-gosia-medium`

MIT-licensed voice trained on CC0 data, ONNX/VITS, **RTF 0.18 measured on
this laptop's CPU** — zero GPU cost, which matters when the card is busy with
the brain. It is the always-works floor, not the quality ceiling.

Note the project moved: `rhasspy/piper` was archived 2025-10-06, and the
maintained runtime is `OHF-Voice/piper1-gpl` (**GPL-3.0 runtime**, MIT/CC0
voices). Pin the archived MIT runtime if copyleft matters for distribution.
Polish voices available: `gosia`, `darkman`, `bass-high` (Apache-2.0), plus
community `zenski`/`meski` whose repo declares no licence.

### Upgrades, in the order I would try them

1. **`OpenMOSS-Team/MOSS-TTS-Realtime`** (1.7B, **Apache-2.0**) — Polish in
   its official 20-language list, **180 ms TTFB after warmup**, RTF 0.51 on a
   single L20, native multi-turn streaming with voice consistency across
   turns. The only candidate clearing permissive licence + genuine Polish +
   streaming simultaneously.
2. **`OpenMOSS-Team/MOSS-TTS-Nano-100M`** (Apache-2.0) — streams on **4 CPU
   cores, no GPU**. A direct Piper replacement with a modern architecture.
3. **`openbmb/VoxCPM2`** (2B, Apache-2.0) — 30 languages including Polish,
   ~8 GB, RTF 0.13–0.30.
4. **`ResembleAI/chatterbox`** Multilingual (0.5B, **MIT**) — 23 languages
   incl. Polish, voice cloning. ⚠️ issue #311 ("strong English accent, almost
   impossible to understand") is **open and unanswered**; much of it was an
   ONNX export bug (truncated embedding table breaking ą ę ć ś ź ż ń ó) fixed
   in `Folx/chatterbox-ONNX-polish`. Use the PyTorch path or that export.

### Rejected

| option | why not |
|---|---|
| **Kokoro-82M** | **no Polish** — blocked by G2P in `misaki`; no Polish fine-tune exists |
| Qwen3-TTS | 10 languages, no Polish (despite 97 ms latency) |
| CosyVoice 2/3, Orpheus, Spark-TTS, IndexTTS, Dia, Zonos, CSM | no Polish |
| **XTTS-v2** | best *verified* Polish data (198.8 h, CER 0.759) but **CPML — non-commercial**, and Coqui shut down; the `idiap` fork relicenses only the code, not the weights |
| MMS-TTS-pol | CC-BY-NC, 16 kHz, single speaker, trained on **religious readings** — liturgical prosody, wrong register for a dog |
| F5-TTS + Polish fine-tunes | weights are CC-BY-NC (Emilia data); one fine-tune claims MIT, which cannot be true of a CC-BY-NC derivative |
| Fish Audio S2-Pro | ~100 ms TTFA and good Polish, but a **paid commercial licence** |
| Higgs Audio v3 | Polish in its "production-quality" tier, but research licence |
| VibeVoice-Realtime | Polish outputs *"unsupported and may be unintelligible"* per its own card, and it bakes an audible AI disclaimer into output |
| Supertonic 3 | 99M, CPU-only, 31 langs incl. `pl` — genuinely interesting, but OpenRAIL-M carries behavioural-use restrictions |

---

## 4. Polish-language institutions: nothing to use

Enumerated via the HuggingFace API rather than trusting announcements.

| org | models | speech models |
|---|---|---|
| `speakleash` (Bielik) | 85 | **0** |
| `CYFRAGOVPL` (PLLuM) | 37 | **0** |
| `clarin-pl` | 184+ | **0** |
| `amu-cai` | 6 | **0** |
| `Voicelab` | 7 | **0** (despite the name) |
| NASK | no HF org at all | **0** |

There is no Bielik Audio, no PLLuM Voice — not shipped, not announced. NASK's
HIVE AI names **vision** as its next modality. EU projects are no better: no
EuroLLM speech model; UTTER's Spire is CC-BY-NC with no Polish; Meetween's
SpeechLMM covers 10 languages, Polish absent.

**Poland's real contribution here is data**, from AMU Poznań:
`amu-cai/pl-asr-bigos-v2` (NeurIPS 2024 D&B), `amu-cai/nEMO`, and the Polish
ASR Leaderboard. Also `czyzi0/the-mc-speech-dataset` — 22 h single speaker,
44.1 kHz, **CC0** — the best Polish TTS training set, already underpinning
Piper's `mc_speech` voice. The Polish-capable ASR that exists came from
NVIDIA.

---

## 5. The gap that should bother us

**No public benchmark ranks TTS on Polish.** TTSDS2 is the only one that ever
did; its Polish table is a May 2025 snapshot of four systems, where Fish
Speech 1.5 failed **44 %** of Polish prompts outright. TTS Arena is
English-only, the HF Open ASR multilingual track excludes Polish.

So every TTS quality claim in this document — including my choice of Piper —
is provenance-based reasoning, not measurement. Before trusting any of them:
~50 Polish prompts × 3–4 candidates, scored by WER through a Polish ASR and a
native-speaker preference test, with `czyzi0/the-mc-speech-dataset` as the
reference distribution. That is an afternoon, and it would replace the
weakest link in this decision record.

Two more things nobody has published, worth measuring on our own audio:
**int8 vs fp16 Whisper WER on Polish** (the "<0.2 % regression" figure is
blog folklore) and **large-v3-turbo vs large-v3 on Polish** — if turbo holds
up, it is a significant latency win.

One live datapoint on brain quality: at 4B the Polish is good but not
flawless — asked what it saw, it produced *"idealne na snop lub kolację"*
where *snop* (a sheaf of corn) should have been *drzemkę* (a nap). Fluent,
occasionally wrong in a way a native speaker notices immediately. A 30B would
likely fix this; see [agent.md](agent.md) for the brain-size options.

---

# Appendix A — rejected on licence

Everything here is technically capable and legally unusable for anything
beyond private research. Listed so nobody re-evaluates them in six months.

| model | licence | what it actually forbids |
|---|---|---|
| **Qwen3.5-Omni** Plus/Flash/Light | proprietary, API-only | no weights at any price; the only Qwen that speaks Polish |
| **SeamlessM4T v2 / seamless-streaming** | CC-BY-NC-4.0 | non-commercial; the *only* open model doing Polish speech in **and** out |
| **XTTS-v2** | CPML | non-commercial **including its outputs**, and forbids training other models on it; upstream company dissolved |
| **MMS-TTS-pol** | CC-BY-NC-4.0 | non-commercial |
| **F5-TTS weights** (+ all Polish fine-tunes) | CC-BY-NC (Emilia data) | non-commercial; one fine-tune claims MIT, which cannot survive a CC-BY-NC parent |
| **Fish Audio S2-Pro / OpenAudio S1-mini** | Fish Research / CC-BY-NC-SA | paid licence for commercial use |
| **Higgs Audio v3** | Boson Research License | research only |
| **LLaMA-Omni / LLaMA-Omni2** | "other" — academic only; **LLaMA-Omni2 declares no licence field at all** | explicit no-commercial-use clause in the card body; contact required |
| **Spirit LM** (Meta) | FAIR Noncommercial + form gate | noncommercial, *"use in languages other than English"* is out-of-scope |
| **GLM-4-Voice-9B** | custom | free for academic use; commercial requires registration |
| **Baichuan-Omni-1.5** | Apache tag **overridden** by community licence | DAU < 1M, no cloud providers, emailed application, **revocable** |
| **LFM2.5-Audio** | LFM Open License | commercial use dies above **$10M group revenue** |
| **Covo-Audio-Chat** (Tencent) | academic only | *"refrain from any commercial or production purposes under any circumstances"* |
| **Freeze-Omni** | Apache-2.0 **+ Tencent AUP** | usable, but 19 extra clauses incl. no high-stakes automated decisions, no military, Tencent may amend unilaterally |
| **IndexTTS** | bilibili licence | §3.4(c) forbids using outputs to improve other models; §4.2 restricts **automated decision-making and critical-infrastructure control** — read that twice before putting it on a robot |
| **Supertonic 3** | OpenRAIL-M | commercial OK but with behavioural-use restrictions + attribution |
| **VibeVoice-Realtime** | MIT tag, research-restricted card | also bakes an audible "AI-generated" disclaimer into every output |
| **SLAM-Omni Chinese model** | GPL-3.0 (Belle data) | research only; the English checkpoints are MIT |
| **Qwen2.5-Omni-3B** | Qwen Research License | non-commercial (the 7B is Apache-2.0) |
| **ultraVAD** (fixie-ai) | **undeclared** | no licence file at all — legally unusable until clarified, which is a shame because it is the best turn-detector we found |
| **MOSS-Speech, SpeechGPT-2.0, BayLing-Duplex, Step-Audio-EditX** | **undeclared weights** | no licence metadata; treat as unusable |

**Licence-clean and usable**, for contrast: Whisper (MIT), faster-whisper,
Piper voices (MIT/CC0, GPL runtime), MOSS-TTS family (Apache-2.0), VoxCPM2
(Apache-2.0), Chatterbox (MIT), Parakeet/Canary (CC-BY-4.0), Qwen3-VL and
Qwen3-Omni (Apache-2.0), MiniCPM-o (Apache-2.0), Moshi (CC-BY-4.0),
Mini-Omni (MIT), Kimi-Audio (MIT), Ming-Omni (MIT).

---

# Appendix B — the English-only stack we are not building

If the Polish requirement were dropped, the whole cascade collapses into one
model that hears and speaks natively. Recorded because it is the design we
would want, and because an English demo is a plausible future ask.

### Ranked, for English speech-to-speech

| # | model | VRAM | licence | latency | duplex | serving |
|---|---|---|---|---|---|---|
| **1** | **MiniCPM-o 4.5** (9B) | **19 GB bf16 / 11 GB int4** | ✅ Apache-2.0 | TTFT 0.6 s | ✅ genuine — 1 Hz speak/don't-speak decision | **vLLM + SGLang + llama.cpp + Ollama** |
| 2 | **Moshi / Moshiko** (7–8B) | 24 GB | ✅ CC-BY-4.0 | *"160 ms theoretical, 200 ms in practice"* on an L4 | ✅ | own Rust/Candle server only |
| 3 | Qwen3-Omni-30B-A3B | ~70 GB bf16 | ✅ Apache-2.0 | 234 ms first packet | experimental | **vllm-omni / sglang-omni** |
| 4 | Step-Audio-2-mini (8.3B) | — | ✅ Apache-2.0 | unpublished | — | vLLM fork + Docker |
| 5 | Mini-Omni2 (0.5B backbone) | — | ✅ MIT | unpublished | claimed, not delivered | litGPT + Flask |
| 6 | LFM2.5-Audio (1.5B) | **1.5 GB Q4** | ⚠️ revenue cap | <100 ms | — | llama.cpp / ONNX / WebGPU |

**MiniCPM-o 4.5 is the pick**: the only one that is simultaneously
Apache-2.0, genuinely full-duplex, documented for VRAM *and* latency, and
upstream in both vLLM and SGLang. Moshi holds the latency crown but is
English-only, 24 GB, and locked to Kyutai's own server. LFM2.5-Audio is the
only sub-2 GB native speech-to-speech — relevant if this ever has to run on
the Jetson rather than a rented card.

**What switching would buy:** one model instead of three, no ASR stage, no
TTS stage, no segmenter, and true barge-in from the model rather than our
energy heuristic. **What it would cost:** the robot could only be talked to in
English.

**What stays either way:** the tool layer, the goal state machine, the search
FSM, tracing, and the mic/playback transport. Only `voice.py`'s recogniser
and `tts.py` would be replaced — which is exactly why both sit behind
one-method interfaces.

### English-capable pieces if you keep the cascade

Worth knowing these exist even in the Polish build, as fallbacks or for an
English demo mode: `whisper-large-v3-turbo` (MIT, fastest Whisper),
`nvidia/parakeet-tdt-0.6b-v3` (CC-BY-4.0, 25 languages), Kyutai STT/TTS
(CC-BY-4.0, en/fr), and `Kokoro-82M` (Apache-2.0) — the last of which is a
strong English TTS and, as noted above, has **no Polish** and no route to it.
