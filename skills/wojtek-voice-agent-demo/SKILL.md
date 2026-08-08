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
`training/docs/agent.md`; the model survey and every rejected option in
`training/docs/polish-voice.md`.

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
Qwen 6, vLLM wheels 7). Across four boxes in one session that was **$1.97 of download against
$0.78 of GPU**. Always filter on `inet_down_cost`:

```bash
vastai search offers 'gpu_ram>=44 num_gpus=1 disk_space>=100 cuda_vers>=12.4 \
  rentable=true compute_cap>=860 inet_down>=1000' -o 'dph+' --raw \
  | python3 -c 'import json,sys; [print(f"{x[\"id\"]} {x[\"gpu_name\"]:16} {x[\"gpu_ram\"]/1024:.0f}GB \
${x[\"dph_total\"]:.3f}/hr ${x.get(\"inet_down_cost\",0):.4f}/GB") for x in json.load(sys.stdin)[:6]]'
```

A host at $0.004/GB makes the deploy cost $0.12; one at $0.04/GB makes it $1.20.

**Size: 48 GB.** Measured with all three serving: **37.5 / 49.1 GB**, split
vLLM 23.2 (8 GB weights + 15 GB KV reservation) / FutureNav 10.0 / whisper 4.2.
FutureNav **grows** — 6.9 GB at startup, 8.5 GB after 43 steps, as its VGGT cache fills. On a
24 GB card the same stack hit 24.3/24.6 GB and the ASR died mid-recording with
`cudaErrorInvalidDevice`. 24 GB works only with `--gpu-memory-utilization 0.30` **and**
`ASR_COMPUTE=int8`, with nothing spare.

`compute_cap>=860` matters: FP8 needs Ampere or newer (Turing RTX 8000 is cheap and useless
here). Use a **`-devel` image** — FutureNav builds CUDA extensions:

```bash
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  --disk 110 --ssh --direct
vastai attach ssh <INSTANCE_ID> "$(cat ~/.ssh/id_rsa.pub)"   # no key = permission denied
vastai ssh-url <INSTANCE_ID>
```

The key attach is **required and easy to forget** — a fresh account has no SSH key registered
and every login fails with `Permission denied (publickey)`. Wait ~15 s after attaching.

## 3. Deploy, serve, tunnel

```bash
export VAST_SSH='ssh -o BatchMode=yes -p <port> root@<host>'
scripts/vast_stack.sh deploy both     # ~15 min: FutureNav + vLLM + ASR
scripts/vast_stack.sh serve           # all three, detached
scripts/vast_stack.sh tunnel          # 8090 + 8100 + 8110 → localhost (keep open)
scripts/vast_stack.sh status
```

`deploy vllm` skips FutureNav when you only need chat/search — saves ~10 min and 17 GB.

Then the demo, from the main checkout:

```bash
eval "$(scripts/vast_stack.sh env)"
./training/run.sh room --vlm-backend futurenav \
  --vlm-url http://127.0.0.1:8100 --agent-url http://127.0.0.1:8090
```

Open `http://127.0.0.1:8010`, click the mic, talk Polish. The **AGENT DECISIONS** panel streams
every model output, tool call, FSM transition and FutureNav action; **debug** adds raw model
I/O; `GET /api/trace` and `runs/agent_traces/*.jsonl` have the same events after the fact.

## 4. Recording a session

```bash
python -m wojtek_rl.agent.record --out demo.mp4 --seconds 180
```

Films chase cam + ego view + map + captions straight off the websocket, with the dog's speech
on the audio track. No screen capture, so it works headless.

**Reset before every take.** A goal left running by a previous test keeps walking into the
recording and announces its outcome over the opening seconds — one take opened with
*"Utknąłem, nie mogę tam dojść."* Send `goal_cancel`, `chat_reset`, `reset`, wait ~4 s, then
start. Version the output with a timestamp so takes are comparable.

## 5. Teardown — arm it before you need it

```bash
scripts/vast_destroy_at.sh install <INSTANCE_ID> 23:55   # launchd, survives this shell
vastai destroy instance <INSTANCE_ID>                    # or now
```

Not cron (silently skips a job whose time passed while the Mac slept) and not a backgrounded
`sleep` (dies with the shell). vast has **no server-side scheduled destroy** — `vastai` exposes
`show`/`delete scheduled-job` but no `create`, and `end_date` is not settable. If the Mac is
off at the appointed time, nothing fires; the credit balance is the only real backstop.

**Keep one box alive across a session** rather than destroying between runs. GPU idle is
$0.60/hr; a re-deploy is 30 GB of download plus 15 minutes.

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
- **BSD sed** has no `\+`; the old port parser produced nothing and scp went to port 22 and
  hung. Fixed in `vast_stack.sh`, but the same trap applies to any new script on macOS.

## Related

- `training/docs/agent.md` — architecture, prompting contracts, the measured latencies
- `training/docs/polish-voice.md` — why the voice is a cascade, every model considered, licences
- `skills/futurenav-nav-demo/SKILL.md` — the narrower FutureNav-only loop
- `skills/vastai-gpu-training-ops/SKILL.md` — offer selection and spend caps in general
