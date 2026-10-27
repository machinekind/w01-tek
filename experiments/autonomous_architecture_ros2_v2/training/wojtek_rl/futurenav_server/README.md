# FutureNav-4B action server

Serves [FutureNav](https://github.com/linglingxiansen/FutureNav) (Qwen3-VL-4B
VLA for Vision-Language Navigation, weights `llxs/FutureNav`) behind a small
HTTP API so the room simulator's `futurenav` backend
(`wojtek_brain/futurenav_nav.py`) can drive the robot from another machine.

Host-agnostic: deploy to any Linux box with a CUDA GPU and SSH access.

## Deploy

```bash
./deploy.sh <ssh-host> [remote-dir]     # default remote-dir: ~/futurenav
ssh <ssh-host> 'cd ~/futurenav && ./start.sh'
```

`deploy.sh` is idempotent: syncs `server.py`, clones the FutureNav repo,
builds the venv from `requirements.txt`, downloads the 11.7 GB weights, and
writes a `start.sh`. Re-run it to push server updates.

Point the simulator at it:

```bash
./run.sh room --vlm-url http://<host>:8100
./run.sh nav-episode --goal "go to the bed" --vlm-url http://<host>:8100
```

## API

```
POST /reset {"instruction": "walk to the bed"}   -> {"ok": true}
POST /act   {"frame_b64": "<jpeg/png base64>"}   -> {"action": "MOVE_FORWARD", "raw": "...", "step": 3}
GET  /health                                     -> {"status": "ok", "vram_gib": 6.9, ...}
```

One episode at a time; the engine keeps the frame history server-side and the
client streams only the current ego frame (mirrors FutureNav's `eval/agent.py`
protocol, minus habitat).

## GPU sizing and env knobs

| var | default | meaning |
|---|---|---|
| `FUTURENAV_PORT` | `8100` | listen port (start.sh) |
| `FUTURENAV_WEIGHTS` | `<dir>/weights/FutureNav-4B-Base` | checkpoint dir |
| `FUTURENAV_SRC` | `<dir>/FutureNav/src` | FutureNav repo `src/` (custom model code) |
| `FUTURENAV_MAX_HISTORY` | `8` | history frames in the prompt (+1 current) |
| `FUTURENAV_ATTN` | `sdpa` | attention impl (`eager` = upstream, more memory) |
| `FUTURENAV_QUANT` | `8bit` | `8bit` (bitsandbytes, LLM blocks only) or `none` (bf16) |
| `FUTURENAV_VGGT_CACHE` | `6` | geometry cache: `0` = drop per step, `full` = upstream unbounded, `N` = anchor frame + N−1 recent frames (bounded, ~151 MiB/frame) |
| `FUTURENAV_MAX_NEW_TOKENS` | `8` | decode budget; actions are ≤5 tokens (upstream used 24) |
| `FUTURENAV_PREFIX_CACHE` | `1` | reuse the episode's prompt KV across steps (~1.8× faster inference; rare near-tie argmax flips from chunked-prefill numerics — set `0` for benchmark-exact decoding) |

Defaults target a **shared 16 GB card** (6.9 GiB load, ~8 GiB peak,
0.5–1.6 s per action). For **paper-faithful inference** you need a dedicated
24 GB+ card and:

```bash
FUTURENAV_QUANT=none FUTURENAV_ATTN=eager FUTURENAV_VGGT_CACHE=full \
FUTURENAV_MAX_NEW_TOKENS=24 ./start.sh
```

bf16 on a 16 GB card is possible if it is (mostly) dedicated: measured
11.5 GiB steady with the window'd VGGT cache and prefix cache on, model time
0.32–0.41 s/step (vs 0.45–0.50 s in 8-bit) — and bf16 removes the 8-bit
near-tie decode flips. On the current host that means stopping the dev
embedding containers first (robot mode):

```bash
docker stop unite_transformers_dev research-podcast-text2vec-transformers-1   # free 2.9 GiB
docker start unite_transformers_dev research-podcast-text2vec-transformers-1  # restore
```

Note on `full`: upstream's own eviction (`StartRecentKVCache(8, 48)` on the
frame axis) only binds past 56 frames, i.e. effectively never within an
R2R episode — the cache grows ~151 MiB/frame all episode. The paper evaluated
on a 96 GB H20, where that is invisible; it OOMs a 16 GB card around frame
15. The bounded `N` mode reuses their eviction class with sizes that
actually trigger (first frame = geometry anchor, per `reference_frame:
"first"`), keeping recent-window cross-step geometry at flat memory.

## Hard-won constraints

- `transformers==4.57.x` required; 5.x breaks the custom class (tied-weights
  API change). Pinned in `requirements.txt`.
- The released checkpoint (`Qwen3VLForJanusVLN_ObsHeadV4`) has auxiliary-head
  shapes that differ from the repo's current class; loaded with
  `ignore_mismatched_sizes=True` — those heads are unused at evaluation.
- bf16 weights need ~11.3 GiB before any activations — on a 16 GB card that
  leaves nothing, hence the 8-bit default.
- The VGGT KV cache costs >1 GiB per accumulated frame (observed OOM at step
  ~19 on 16 GB). Default drops it every step: the model keeps its 9-image
  prompt history, only cross-step geometry accumulation is lost.
- `einops` is required but missing from upstream `requirements.txt`.

## Smoke test

```bash
FUTURENAV_WEIGHTS=... ./venv/bin/python smoke_test.py   # loads model, 3 synthetic steps
```
