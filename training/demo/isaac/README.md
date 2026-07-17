# Wojtek Isaac Sim demo

Isaac Sim port of the MuJoCo click-to-walk demo (`training/demo/app.py`).
Runs the exported joystick locomotion policy on a photoreal RTX warehouse
scene and serves the same browser protocol: click-to-walk minimap, discrete
VLM commands (`turn_left 15 | forward 0.5 | stop`), chase + ego camera
streams over a websocket.

The policy is loaded from `POLICY_DIR` (below), not vendored here, so the
demo tracks whatever `ros/src/wojtek_policy/config` ships. Current policy is
`wojtek_springy_b_20260713_1827` (IMU-blind 40-dim obs, no gait clock, 4-dim
command with a pinned 0.125 m height). The numpy `WojtekPolicy` runtime also
loads the older `fbb_loco_v8` (48-dim, gait clock) unchanged — the server
reads the standing height via `getattr(pol, "command_height",
getattr(pol, "default_height", ...))`, covering both runtimes.

## Files

| File | Purpose |
|------|---------|
| `isaac_room_server.py` | Headless Isaac Sim server: physics + policy loop on the main thread, uvicorn (`/`, `/api/info`, `/ws`) in a daemon thread. Port via `WOJTEK_PORT` (default 8200). |
| `isaac_room.html` | Browser GUI (industrial-editorial design matching the Wojtek site): ego (VLM) hero + command console on the left, orbit camera + dark tactical map on the right. Served per request — UI edits need no server restart. |
| `wojtek_mjcf2usd.py` | One-off MJCF → USD conversion of `wojtek_mjx.xml` via `isaacsim.asset.importer.mjcf`. |
| `body_masses.json` | Per-body masses exported from MuJoCo; overwrites Isaac's importer masses (importer drift: 22.2 kg vs 16.04 kg). |
| `home_qpos.json` | Loop-consistent home configuration settled in MuJoCo; required because teleporting closed-chain legs to arbitrary q makes PhysX NaN. |

## Requirements

- GPU host with Isaac Sim 5.0 (`pip install "isaacsim[all,extscache]==5.0.0"
  --extra-index-url https://pypi.nvidia.com`), Python 3.11.
- Exported policy artifacts (`policy.npz`, `policy_meta.json`) plus
  `policy.py` / `navigation.py` from `training/wojtek_rl/` on the
  server's `POLICY_DIR` path.
- Robot USD produced by `wojtek_mjcf2usd.py`.

## Run

```bash
OMNI_KIT_ACCEPT_EULA=YES WOJTEK_PORT=8200 python -u isaac_room_server.py
```

First boot compiles shaders (~100 s) and streams warehouse assets from
NVIDIA's asset server (needs internet). Open `http://<host>:8200/`.

## Running on a remote GPU host

The server needs an RTX GPU and a full Isaac Sim install, so it usually runs
on a separate machine while you drive it from a laptop. Nothing below is tied
to a specific host — set `GPU_HOST` to your `user@host` (or an SSH config
alias) and reach the UI over whatever network gives you the box (LAN, VPN,
Tailscale, or an SSH tunnel).

```bash
GPU_HOST=user@gpu-box          # your SSH target
# copy the demo + assets over (first time / after edits)
scp isaac_room_server.py isaac_room.html "$GPU_HOST":~/

# launch detached so it survives the SSH session; redirect stdin from
# /dev/null or the SSH channel blocks until the process exits
ssh "$GPU_HOST" 'setsid nohup env OMNI_KIT_ACCEPT_EULA=YES WOJTEK_PORT=8200 \
    python -u ~/isaac_room_server.py </dev/null >~/isaac_room.log 2>&1 &'

# watch it come up (first boot ~100 s); "SERVER_START :8200" = ready
ssh "$GPU_HOST" 'tail -f ~/isaac_room.log'
```

Then open `http://<gpu-host-address>:8200/` in a browser, or forward it:
`ssh -L 8200:localhost:8200 "$GPU_HOST"` and use `http://localhost:8200/`.

Operational notes:

- **`isaac_room.html` is read per request** — re-`scp` it and refresh the
  browser (hard-refresh to beat the cache); no server restart needed. Only
  `isaac_room_server.py` changes require a restart.
- **Restart by PID, not `pkill -f`** — a `pkill -f isaac_room_server` run over
  SSH also matches the shell running it and kills its own session (exit 255),
  so the relaunch never fires. Instead: `pgrep -f 'isaac_room_server[.]py'`,
  `kill -9 <pid>`, confirm `pgrep` is empty, then relaunch. Stale orphaned
  launchers can keep the old log file open, so relaunch to a fresh log path
  when a restart looks stuck.
- If the host is password-only SSH, keep the password in an env var
  (`SSHPASS`) and use `sshpass -e ssh …`; never hard-code it in scripts.

## Physics/rendering notes (hard-won)

- MJCF importer drops **all** collision geoms — the server recreates the
  training contact model (foot spheres r=0.046 + friction material, base box).
- Importer applies a spurious `ArticulationRootAPI` on the massless
  `worldBody`; it must be removed or PhysX init fails.
- Orbit camera: drag to rotate around the robot, scroll to zoom, double-click
  to reset. The server keeps an orbit state (yaw offset from "behind" the
  smoothed heading, pitch, distance) and rebuilds the eye each render; the
  heading is EMA-smoothed on the unit vector (alpha 0.08) and position on
  0.88/0.12 — a hard-locked camera turns trot oscillation into shake. Browser
  sends `{"type":"cam", dyaw|dpitch|zoom|reset}`.
- Sync rendering + DLAA + even frame pacing (render every 2nd tick of the
  50 Hz loop) are what make the stream smooth.
- Render resolution ceiling on this GPU (RTX 4090 Laptop 16 GB): internal
  1280×720 renders at rtf 1.0 but the process dies deterministically ~5 min in
  (tick 15000) — not OOM (15 GB free), an RTX-pipeline limit. 1152×648 internal
  / chase 896×504 / ego 800×600 is stable and still sharper than the old
  960×540 / 640×360. Raise cautiously and watch for the timed crash.
