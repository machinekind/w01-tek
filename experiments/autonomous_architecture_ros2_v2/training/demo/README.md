# Click-to-walk demo

Interactive web app: **click a point on the map and Wojtek walks
there.** A FastAPI server holds the MJX env + trained policy and runs a
closed-loop sim over a websocket, streaming a live chase-cam view + the robot
pose to the browser.

This is a thin presentation layer on top of the training package — it imports
`wojtek_rl.env` (the robot's MJX env), `wojtek_rl.policy_io` (checkpoint loader) and
`wojtek_rl.train` (PPO config), so it runs inside the training project's venv.
Ported from the Go1 click-to-walk app and retargeted to the dog (chase-cam +
nav constants scaled for its ~0.10 m standing height).

## Layout

| File | What |
| ---- | ---- |
| `app.py`          | FastAPI server: env + policy + renderer, websocket rollout loop |
| `navigation.py`   | go-to-point controller — turns a clicked world point into the `[vx, vy, vyaw]` joystick command (pure NumPy/math, unit-testable) |
| `static/index.html` | top-down minimap browser UI |

## Run

From the training project root:

```bash
./run.sh app                          # serves http://127.0.0.1:8010
WOJTEK_RUN_DIR=policies/fbb_v3 ./run.sh app
```

Needs the GPU (EGL rendering + JAX). Open the URL and click anywhere on the map;
the robot navigates to it and stops within the nav `stop_radius`. If it falls it
auto-recovers so the demo keeps going.
