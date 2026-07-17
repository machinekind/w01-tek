# Wojtek Isaac Sim demo

Isaac Sim port of the MuJoCo click-to-walk demo (`training/demo/app.py`).
Runs the exported `fbb_loco_v8` joystick policy on a photoreal RTX warehouse
scene and serves the same browser protocol: click-to-walk minimap, discrete
VLM commands (`turn_left 15 | forward 0.5 | stop`), chase + ego camera
streams over a websocket.

## Files

| File | Purpose |
|------|---------|
| `isaac_room_server.py` | Headless Isaac Sim server: physics + policy loop on the main thread, uvicorn (`/`, `/api/info`, `/ws`) in a daemon thread. Port via `WOJTEK_PORT` (default 8200). |
| `isaac_room.html` | Browser GUI (industrial-editorial design matching the Wojtek site): chase-cam hero, command console, ego cam, dark tactical map. Served per request — UI edits need no server restart. |
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

## Physics/rendering notes (hard-won)

- MJCF importer drops **all** collision geoms — the server recreates the
  training contact model (foot spheres r=0.046 + friction material, base box).
- Importer applies a spurious `ArticulationRootAPI` on the massless
  `worldBody`; it must be removed or PhysX init fails.
- Chase camera follows the robot's smoothed heading (EMA on the unit heading
  vector, alpha 0.08) 2.4 m behind, 0.6 m up; position EMA 0.88/0.12 —
  a hard-locked camera turns trot oscillation into shake.
- Sync rendering + DLAA + even frame pacing (render every 2nd tick of the
  50 Hz loop) are what make the stream smooth; resolution and mesh detail
  were never the bottleneck.
