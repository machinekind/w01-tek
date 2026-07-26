---
name: futurenav-nav-demo
description: Bring up the closed-loop FutureNav navigation demo — room app UI on a local port, FutureNav-4B action server on a rented GPU, SSH tunnel between them. Use for demoing or debugging the discrete-action VLN loop end to end, including policy resolution, GPU provisioning, server start, and spend-capped teardown.
---

# Bring up the FutureNav navigation demo

Two processes: the room app (MuJoCo scan scene + walking policy + UI) on the Mac, the
FutureNav-4B action server on a rented CUDA box. They meet over an SSH tunnel on `:8100`.
Budget ~20 min from nothing to a walking dog; ~$0.15/hr for the GPU.

Layers, so you know which one broke: FutureNav action → `futurenav_nav.decision_from_action`
→ mid-level command string → `MidLevelExecutor` latched pose target → body-frame `[vx, vy, wz]`
→ exported NumPy policy at 50 Hz. See `training/wojtek_rl/midlevel.py`.

## 1. Run from the main checkout, not a worktree

Worktrees lack every generated/ignored input the demo needs: `training/.venv`,
`training/assets/scenes/*`, `assets/room/{manifest,occupancy}`, and the built
`ros/src/wojtek_description/mujoco/scene_*.xml`. Do not symlink them together.

Point the preview launcher at the main checkout's `run.sh` (absolute path) and keep working
in the worktree. Confirm the scene exists before starting:

```bash
ls "$MAIN/ros/src/wojtek_description/mujoco/" | grep scene_
```

## 2. Policy: HF token from `.env`, not the CLI login

`paths.DEFAULT_POLICY` lives in the private `<HF_ORGANIZATION>` HF org. The machine-level
`hf auth login` identity may be a different account with no access — a 404 on
`policy.npz` means wrong identity, not a missing repo. The usable token is in the repo-root
`.env` (gitignored) as `HUGGINGFACE_API_KEY`.

`.claude/launch.json` has no `env` field, so pass it through a wrapper (see
`scripts/room_launch.sh`) and make that the `runtimeExecutable`. Verify access first:

```bash
HF_TOKEN=$(grep '^HUGGINGFACE_API_KEY=' "$MAIN/.env" | cut -d= -f2) \
  "$MAIN/training/.venv/bin/python" -c "
import os
from huggingface_hub import HfApi
a = HfApi(token=os.environ['HF_TOKEN'])
print(a.whoami()['name'], a.list_repo_files('<HF_ORGANIZATION>/wojtek-springy-locomotion')[:3])"
```

Do not try to revive an old local `policy.npz`. Pre-schema-2 keepers fail twice: no
`schema_version`, and layouts like `phase:8` that the schema-2 runtime's `KNOWN_COMPONENTS`
no longer interprets. `migrate_keeper_meta.py` needs `run.json` from the same private repo,
so it does not rescue an offline copy either.

The room app loads the policy in a FastAPI startup hook — a policy failure is a startup
failure, no UI at all.

Start the UI before the GPU exists. Click-to-walk on the minimap proves the whole
executor → policy path with no VLM in the loop.

## 3. Rent the GPU

Filter offers on: `gpu_ram ≥ 22000`, `disk_space ≥ 60`, `reliability2 ≥ 0.98`, high
`inet_down` (11.7 GB of weights). Pick Ampere or newer — Turing (RTX 2080 Ti) has no real
bf16 and the server loads with `dtype=bfloat16`.

```bash
vastai create instance OFFER_ID \
  --image nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 \
  --disk 60 --ssh --direct --label futurenav-demo \
  --env '-p 8100:8100' --cancel-unavail --raw
```

Use a **devel** image. Triton JIT-compiles a kernel on the first `/act`, so a `runtime`
image serves `/health` fine and then returns HTTP 500 with
`RuntimeError: Failed to find C compiler`. On a runtime image, `apt-get install -y
build-essential python3-dev` and restart the server.

Also install `python3.10-venv` before deploying: on Ubuntu 22.04 `python3-venv` alone leaves
`ensurepip` missing and `deploy.sh` dies at venv creation with a half-built `venv/` that must
be `rm -rf`'d before retrying.

Poll until `actual_status` is `running` and read `status_msg` while waiting. A host-side
failure (`failed to inject CDI devices: ... gpu=1: unknown`) never recovers — destroy and
pick a different `machine_id`.

SSH access needs a key on the instance, not on the account:

```bash
vastai attach ssh INSTANCE_ID "$(cat ~/.ssh/id_rsa.pub)"
```

Connect to the direct address from `vastai ssh-url ID`, not the `sshN.vast.ai` proxy
(see `vastai-gpu-training-ops`).

## 4. Arm the spend cap before deploying

`vastai create` has no auto-destroy and the CLI has no scheduled teardown. Launch
`scripts/vast_autodestroy.sh` (absolute-deadline poll — survives laptop sleep, unlike a
single `sleep 3600`) as soon as the instance id exists.

Size the deadline for the whole session, not the demo: any destroy costs a fresh 9-minute
weight download. Extend by writing a later epoch into the deadline file; the watchdog fires
exactly on time and does not warn first.

## 5. Deploy and start

```bash
cd "$MAIN/training/wojtek_rl/futurenav_server" && ./deploy.sh futurenav-gpu
```

~9 min: clone, venv, torch wheels, then 11.7 GB of `llxs/FutureNav`. Ends with `DEPLOY_OK`.

Start it fully detached — a backgrounded command chain inside `ssh host '...'` dies with the
session and leaves no log:

```bash
ssh -n futurenav-gpu 'cd /root/futurenav && setsid nohup ./start.sh > server.log 2>&1 < /dev/null &'
```

Model load is ~2 min (3 shards, 8-bit). Then tunnel and poll:

```bash
ssh -N -L 8100:localhost:8100 futurenav-gpu &
until curl -sf -m 5 http://127.0.0.1:8100/health; do sleep 10; done
```

Tunnel rather than the public `8100/tcp` port mapping: the server is unauthenticated.

Healthy idle: `{"status":"ok","device":"cuda","step":0,"vram_gib":6.92}`. Under load it sits
at 8.5–9.2 GiB and `step` climbs by ~1 per 3 s on a 3090.

## 6. Read the failure from the client error

| Symptom in ROBOT THINKING | Meaning |
|---|---|
| `failed ([Errno 61] Connection refused)` | no tunnel |
| `failed ([Errno 54] Connection reset by peer)` | tunnel up, nothing listening on the box |
| `failed (HTTP Error 500 ...)` | server-side traceback — read `~/futurenav/server.log` |
| three failures then `error` state | `MAX_CONSECUTIVE_FAILURES`; fix the cause, resubmit the goal |

Working demo, per decision: `forward 0.25 · futurenav: 'MOVE_FORWARD' · completed` in the
log, `midlevel · forward 0.25m (0.063 left)` in the header, badge `WALKING`. Long
`MOVE_FORWARD` runs with no turns are normal in a corridor.

## 7. Teardown

The CLI `destroy instance` prompts y/N and will hang a non-interactive shell. Use the API:

```bash
curl -s -X DELETE -H "Authorization: Bearer $VAST_API_KEY" \
  "https://console.vast.ai/api/v0/instances/INSTANCE_ID/" -d '{}'
```

Then confirm `instances` and `volumes` are both empty, stop the watchdog and the tunnel, and
revert `.claude/launch.json` if the demo edited it.
