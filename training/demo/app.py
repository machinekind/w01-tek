"""Click-to-walk web app: click a point, the trained four_bar_bot walks to it.

Ported from 3_jaxpot_robotics/jaxpot_robotics/app.py, trimmed to the single
WojtekJoystick robot (wojtek_rl isn't in the mujoco_playground registry, so
there's no ``envs.load_env``/``envs.load_policy`` indirection -- we build the
env and load the checkpoint directly, mirroring wojtek_rl/eval.py's jit/reset/
step/inference idiom).

A FastAPI server holds the MJX env + trained policy and runs a closed-loop sim
over a websocket. Each tick it asks `navigation.command_to_target` for the
joystick command that drives the robot toward the current target, steps the
policy, streams a chase-cam JPEG + the robot pose to the browser. The browser
draws a top-down minimap; clicking it drops a new target.

Run:
    ./run.sh app                          # serves http://127.0.0.1:8010
    WOJTEK_RUN_DIR=runs/<run> ./run.sh app
"""

from __future__ import annotations

# Headless GPU rendering backend; must precede any mujoco import.
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

# Persistent XLA compilation cache: the first run populates it (~30-40s of
# JIT); every later start reuses the compiled executables, so launching the app
# live is fast (no waiting on compile). Cache lives in the repo's .jax_cache/.
# Override the dir with JAX_COMPILATION_CACHE_DIR.
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".jax_cache")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", _CACHE_DIR)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.0")

import argparse
import asyncio
import base64
import io
import json
from pathlib import Path

import numpy as np
from loguru import logger

ROBOT_KEY = "fbb"
ROBOT_LABEL = "four_bar_bot"
DEFAULT_RUN_DIR = os.environ.get("WOJTEK_RUN_DIR", "policies/fbb_v3")

# Chase-cam + nav profile tuned for this robot's scale (standing height ~0.10 m,
# body ~0.35 m long -- roughly 1/3 the size of jaxpot's Go1). The jaxpot Go1
# profile (distance 2.6, lookat_z 0.25) is scaled down and brought lower.
VIEW = dict(lookat_z=0.14, distance=0.85, elevation=-15.0, azimuth=135.0)
NAV = dict(vx_max=0.4, vy_max=0.25, yaw_max=0.7, stop_radius=0.12)


def _latest_checkpoint(path: Path) -> Path:
    """Deepest numeric checkpoint step dir under ``path``, or ``path`` itself."""
    path = path.resolve()
    if (path / "_METADATA").exists() or (path / "ppo_network_config.json").exists():
        return path
    steps = [d for d in path.glob("*") if d.is_dir() and d.name.isdigit()]
    if steps:
        return max(steps, key=lambda d: int(d.name))
    return path


def _resolve_checkpoint(run_dir: str) -> Path:
    """Same lookup as wojtek_rl.eval: run.json's checkpoint_dir, latest step."""
    run_path = Path(run_dir)
    run_json = run_path / "run.json"
    if run_json.exists():
        meta = json.loads(run_json.read_text())
        ckpt_root = Path(meta["checkpoint_dir"])
    else:
        ckpt_root = run_path
    return _latest_checkpoint(ckpt_root)


class Sim:
    """Owns the env, policy, persistent renderer, and the live rollout state."""

    def __init__(self, run_dir: str):
        import jax
        import mujoco
        from mujoco import mjx

        from wojtek_rl import env as wojtek_env
        from demo.navigation import NavConfig
        from wojtek_rl.policy_io import load_policy
        from wojtek_rl.train import build_ppo_params

        self.env_name = "WojtekJoystick"
        ckpt_dir = _resolve_checkpoint(run_dir)
        logger.info(f"[{ROBOT_KEY}] loading {self.env_name} policy from {ckpt_dir}")

        self._jax = jax
        self._mjx = mjx
        self.env = wojtek_env.WojtekJoystick()
        ppo_params = build_ppo_params([], smoke=False)
        policy = load_policy(ckpt_dir, self.env, ppo_params, deterministic=True)

        self._reset = jax.jit(self.env.reset)
        self._step = jax.jit(self.env.step)
        self._inference = jax.jit(policy)

        self.mjm = self.env.mj_model
        self.renderer = mujoco.Renderer(self.mjm, height=360, width=480)
        self._lookat_z = VIEW["lookat_z"]
        self.cam = mujoco.MjvCamera()
        self.cam.distance = VIEW["distance"]
        self.cam.elevation = VIEW["elevation"]
        self.cam.azimuth = VIEW["azimuth"]

        self.cfg = NavConfig(**NAV)
        self._reset_counter = 0
        self.target: tuple[float, float] | None = None
        self.state = None
        self.reset()
        logger.success("sim ready")

    def reset(self):
        key = self._jax.random.PRNGKey(self._reset_counter)
        self._reset_counter += 1
        self.state = self._reset(key)
        self.target = None
        self._cmd = (0.0, 0.0, 0.0)
        self._key = self._jax.random.PRNGKey(1234)

    def set_target(self, x: float, y: float):
        self.target = (float(x), float(y))

    def pose(self) -> tuple[float, float, float]:
        from demo.navigation import quat_to_yaw

        q = np.asarray(self.state.data.qpos[:7])
        return float(q[0]), float(q[1]), quat_to_yaw(float(q[3]), float(q[4]), float(q[5]), float(q[6]))

    def step(self) -> dict:
        """Advance one control step toward the target; return a state dict."""
        import jax.numpy as jp

        from demo.navigation import command_to_target

        x, y, yaw = self.pose()
        reached = False
        dist = 0.0
        if self.target is not None:
            vx, vy, vyaw, dist, reached = command_to_target(
                x, y, yaw, self.target[0], self.target[1], self.cfg
            )
            self._cmd = (vx, vy, vyaw)
        else:
            self._cmd = (0.0, 0.0, 0.0)

        # Pin the command every step (mirrors wojtek_rl.eval): the env
        # auto-resamples internally on a timer, but since we overwrite
        # info["command"] before every step() call, that resample never wins.
        self.state.info["command"] = jp.array(self._cmd)
        self._key, act_key = self._jax.random.split(self._key)
        action, _ = self._inference(self.state.obs, act_key)
        self.state = self._step(self.state, action)

        # robot may fall (e.g. odd target, random push) -> auto-recover so the
        # demo keeps going without manual intervention
        if bool(self.state.done):
            self.reset()

        nx, ny, nyaw = self.pose()
        return {
            "type": "state",
            "x": nx, "y": ny, "yaw": nyaw,
            "tx": self.target[0] if self.target else None,
            "ty": self.target[1] if self.target else None,
            "reached": reached,
            "dist": dist,
            "cmd": [round(c, 3) for c in self._cmd],
        }

    def render_jpeg(self) -> str:
        from PIL import Image

        x, y, _ = self.pose()
        self.cam.lookat[:] = [x, y, self._lookat_z]
        d = self._mjx.get_data(self.mjm, self.state.data)
        self.renderer.update_scene(d, camera=self.cam)
        frame = self.renderer.render()
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("ascii")


# --- FastAPI app ------------------------------------------------------------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="four_bar_bot click-to-walk")
_STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

CONTROL_HZ = 50.0          # real-time pacing (matches env ctrl_dt = 0.02s)
RENDER_EVERY = 2           # stream a frame every N control steps (~25 fps)
WORLD_HALF = 2.5           # map spans [-2.5, 2.5] m -- scaled to the robot's size/speed

_sim: Sim | None = None


def get_sim() -> Sim:
    global _sim
    if _sim is None:
        _sim = Sim(DEFAULT_RUN_DIR)
        # Pre-compile step + render now so the first client's websocket
        # handshake isn't a ~20s JIT stall.
        _sim.step()
        _sim.render_jpeg()
        _sim.reset()
        logger.success(f"sim ready: {ROBOT_KEY} ({_sim.env_name})")
    return _sim


@app.on_event("startup")
def _warmup():
    # Build the env + policy + renderer before any client connects.
    get_sim()


@app.get("/")
def index():
    return FileResponse(str(_STATIC / "index.html"))


@app.get("/api/info")
def info():
    sim = get_sim()
    return {
        "world_half": WORLD_HALF,
        "current": ROBOT_KEY,
        "robots": [{"key": ROBOT_KEY, "label": ROBOT_LABEL, "env": sim.env_name}],
    }


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    sim = get_sim()

    async def reader():
        try:
            while True:
                msg = json.loads(await sock.receive_text())
                t = msg.get("type")
                if t == "target":
                    sim.set_target(msg["x"], msg["y"])
                elif t == "reset":
                    sim.reset()
        except WebSocketDisconnect:
            pass

    reader_task = asyncio.create_task(reader())
    dt = 1.0 / CONTROL_HZ
    i = 0
    try:
        loop = asyncio.get_event_loop()
        while True:
            t0 = loop.time()
            payload = sim.step()
            payload["robot"] = ROBOT_KEY
            if i % RENDER_EVERY == 0:
                payload["frame"] = sim.render_jpeg()
            await sock.send_text(json.dumps(payload))
            i += 1
            # pace to real time
            await asyncio.sleep(max(0.0, dt - (loop.time() - t0)))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        reader_task.cancel()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8010")))
    args = p.parse_args(argv)

    import uvicorn

    logger.info(f"serving four_bar_bot click-to-walk on http://{args.host}:{args.port}")
    # wsproto backend: uvicorn's default "websockets" impl uses the legacy API
    # removed in websockets>=14, which breaks the WS handshake.
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", ws="wsproto")


if __name__ == "__main__":
    main()
