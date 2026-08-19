---
name: wojtek-voice-agent-demo
description: Bring up the full talking-dog demo — chat agent, object search, FutureNav navigation and Polish voice — with the three models on one rented GPU and the sim + UI on the Mac. Use for demoing or recording the agent layer end to end, including GPU sizing, provisioning, tunnels, scripted recordings and guaranteed teardown.
---

# Bring up the Wojtek voice-agent demo

Four processes. On the Mac: the room app (MuJoCo scene + walking policy + UI + Piper TTS).
On one rented CUDA box: **FutureNav-4B** (`:8100`, navigation), **vLLM/Qwen3-VL-4B-FP8**
(`:8090`, chat + tools + search scoring), **faster-whisper large-v3** (`:8110`, Polish ASR).
They meet over one SSH tunnel.

Budget **~25 min** from nothing to a talking dog, then ~$0.60/hr. Design notes live in
`experiments/autonomous_architecture_ros2_v1/docs/agent-layer.md`; the model survey and
every rejected option in `experiments/autonomous_architecture_ros2_v1/docs/polish-voice-research-report.md`.

Layers, so you know which one broke:

```
mic → VoiceSegmenter → whisper(:8110) → WojtekAgent → tool
                                                       ├ navigate → FutureNav(:8100) → mid-level cmd → SCAN → policy
                                                       └ search   → Qwen(:8090) scores views → SCAN goto
reply → Piper(local) → PCM frames → browser
```

## 1. Run from the main checkout, not a worktree

Same rule as the FutureNav skill: a worktree lacks `training/.venv`, the scanned scenes and
the built `scene_*.xml`. If you must work in a worktree, symlink the assets and run Python
with the main checkout's interpreter plus `PYTHONPATH=.` — and know that `./training/run.sh`
will not work there (no `.venv`).

## 2. Pick the GPU by bandwidth, not just $/hr

**Download is the bill, not compute.** A cold deploy pulls ~30 GB (FutureNav weights 17,
Qwen 6, vLLM wheels 7). Across four boxes in one session that was **~2.5× more spent on
download than on GPU time**. Whatever the provider, select offers by egress/ingress price and
link speed as well as $/hr: at $0.004/GB the deploy costs ~$0.12, at $0.04/GB it costs ~$1.20 —
an order of magnitude, on the same GPU.

**Size: 48 GB.** Measured with all three serving: **37.5 / 49.1 GB**, split
vLLM 23.2 (8 GB weights + 15 GB KV reservation) / FutureNav 10.0 / whisper 4.2.
FutureNav **grows** — 6.9 GB at startup, 8.5 GB after 43 steps, as its VGGT cache fills. On a
24 GB card the same stack hit 24.3/24.6 GB and the ASR died mid-recording with
`cudaErrorInvalidDevice`. 24 GB works only with `--gpu-memory-utilization 0.30` **and**
`ASR_COMPUTE=int8`, with nothing spare.

**Compute capability ≥ 8.6** matters: FP8 needs Ampere or newer (a Turing card is cheap and
useless here). Ask for **≥100 GB disk** and a **`-devel` image**
(e.g. `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`) — FutureNav builds CUDA extensions, so a
runtime image cannot serve it.

If the provider registers your SSH key on the instance, do it before the first login and give
it ~15 s: a missing key fails as `Permission denied (publickey)`, which reads like a broken box.

## 3. Deploy, serve, tunnel

Provisioning and teardown scripts for concrete machines live in the **private operations
repository** — this repository never names private hosts or a provider account (root
`CLAUDE.md`). What the deploy has to end up with:

- FutureNav on `:8100`, vLLM/Qwen3-VL on `:8090`, faster-whisper on `:8110`, each started
  detached and verified with `pgrep` in the same ssh call.
- A cold deploy of all three takes ~15 min; skipping FutureNav when you only need chat and
  search saves ~10 min and 17 GB of download.
- One SSH tunnel forwarding 8090 + 8100 + 8110 to localhost, kept open for the session.

Then the demo, from the main checkout, pointed at the tunnel:

```bash
./training/run.sh room --vlm-backend futurenav \
  --vlm-url http://127.0.0.1:8100 --agent-url http://127.0.0.1:8090
```

Open `http://127.0.0.1:8010`, click the mic, talk Polish. The **AGENT DECISIONS** panel streams
every model output, tool call, FSM transition and FutureNav action; **debug** adds raw model
I/O; `GET /api/trace` and `runs/agent_traces/*.jsonl` have the same events after the fact.

## 4. Recording a session

```bash
python -m wojtek_agent.record --out demo.mp4 --seconds 180
```

Films chase cam + ego view + map + captions straight off the websocket, with the dog's speech
on the audio track. No screen capture, so it works headless.

**Reset before every take.** A goal left running by a previous test keeps walking into the
recording and announces its outcome over the opening seconds — one take opened with
*"Utknąłem, nie mogę tam dojść."* Send `goal_cancel`, `chat_reset`, `reset`, wait ~4 s, then
start. Version the output with a timestamp so takes are comparable.

## 5. Teardown — arm it before you need it

Arm an absolute-deadline destroy **before** deploying, using the operations repository's
teardown script. Two things that do not work on a Mac: `cron` (silently skips a job whose time
passed while the machine slept) and a backgrounded `sleep` (dies with the shell). Use launchd
with an absolute deadline poll.

Also check whether your provider offers a *server-side* scheduled destroy — some expose
`show`/`delete` for scheduled jobs but no `create`, and no settable end date, in which case
nothing fires if your laptop is off at the appointed time and the credit balance is the only
real backstop.

**Keep one box alive across a session** rather than destroying between runs: GPU idle costs
under a dollar an hour, while a re-deploy is 30 GB of download plus 15 minutes.

## Traps that cost real time

- **Detached starts.** `nohup cmd &` inside an `ssh "..."` string routinely dies with no log.
  What works: `setsid env VARS cmd >> log 2>&1 < /dev/null & disown; sleep 12; pgrep -f cmd`.
  Always verify with `pgrep` in the same ssh call.
- **`from __future__ import annotations` + a function-local `from fastapi import Request`**
  makes the annotation an unresolvable string; FastAPI decides the body param is a *query*
  param and every POST returns **422 Field required**. Import fastapi at module level.
- **vLLM needs `ninja` on `PATH`** (not merely installed) or it dies with
  `FileNotFoundError: 'ninja'` right after loading weights, and
  **`VLLM_USE_FLASHINFER_SAMPLER=0`** — FlashInfer JIT-builds against the system nvcc whose CUB
  dropped `BlockAdjacentDifference::FlagHeads`, and engine init fails.
- **FutureNav torch must be cu124-pinned.** An unpinned wheel resolves to cu130+, fails CUDA
  init *silently* on older drivers and serves from CPU at ~30 s/action. `deploy` asserts CUDA
  before leaving the server running.
- **Whisper hallucinates on silence** — a phantom `"6V"` reached the agent as a command and
  preempted a running goal. The ASR service drops segments with `no_speech_prob > 0.6` or
  `avg_logprob < -1.0`.
- **The sim only steps while a browser is attached.** A search started via `/api/chat` with no
  viewer open times out every command and looks broken.
- **BSD sed** has no `\+`. A port parser written with it produced nothing, so `scp` silently
  went to port 22 and hung — the trap applies to any new deploy script written on macOS.

## Related

This stack is **experimental** and lives in one directory,
`experiments/autonomous_architecture_ros2_v1/`. Its documents, relative to that root:

- `README.md` — what this experiment is, its layout, and how to run its tests
- `docs/agent-layer.md` — architecture, prompting contracts, the measured latencies
- `docs/polish-voice-research-report.md` — why the voice is a cascade, every model considered, licences
- `docs/architecture.md` and `docs/w1-w2-implementation-report.md` — the ROS 2 port of this
  stack: target architecture, and what is built and verified so far

Elsewhere:

- `skills/futurenav-nav-demo/SKILL.md` — the narrower FutureNav-only loop
