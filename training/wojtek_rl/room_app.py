"""Room demo: the trained robot walks a photo-textured scanned room.

Plain-MuJoCo counterpart of wojtek_rl.app (which is MJX-only and cannot do the
room's mesh collisions): mujoco.mj_step drives the exported NumPy policy
(piesek_ws fbb_policy, the same runtime the real robot runs) at 50 Hz inside
scene_room.xml. The browser gets a chase cam, the onboard ego cam (the VLM's
view), click-to-walk on a minimap, a mid-level command box ("turn_left 30",
"forward 1.5", "stop"), and a VLM goal box ("go to the bed") that lets a
VLM drive that same command interface closed-loop (wojtek_rl.vlm_nav).

Two VLM backends (--vlm-backend / VLM_BACKEND):
  local (default)  Qwen3-VL via mlx-vlm, fully on-device (Apple Silicon);
                   needs `uv sync --extra vlm-local`, weights download from
                   HuggingFace on first use (~18 GB)
  anthropic        Claude via API; needs `--extra vlm` + ANTHROPIC_API_KEY

Run (after ./run.sh room-assets && ./run.sh build && ./run.sh build-room):
    ./run.sh room                                  # local Qwen3-VL goal box
    ANTHROPIC_API_KEY=... ./run.sh room --vlm-backend anthropic
"""

from __future__ import annotations

# Headless rendering backend; must precede any mujoco import.
# egl is Linux-only; macOS offscreen rendering uses cgl.
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "cgl")

import argparse
import asyncio
import base64
import io
import json
import math
import threading
from pathlib import Path

import numpy as np
from loguru import logger

from wojtek_rl import paths
from wojtek_rl.midlevel import Forward, MidLevelExecutor, Stop, parse_command
from wojtek_rl.navigation import NavConfig, command_to_target, quat_to_yaw
from wojtek_rl.np_policy import actuator_addresses, gravity_from_quat, load_fbb_policy
from wojtek_rl.vlm_nav import DEFAULT_MODEL, AnthropicVlmClient, VlmNavigator

ROBOT_KEY = "wojtek"
ROBOT_LABEL = "Wojtek (room)"

# Same nav/view profile as the flat-scene app.
VIEW = dict(lookat_z=0.14, distance=0.85, elevation=-15.0, azimuth=135.0)
NAV = dict(vx_max=0.4, vy_max=0.25, yaw_max=0.7, stop_radius=0.12)

CONTROL_HZ = 50.0
RENDER_EVERY = 2           # ~25 fps per view
FALLEN_HEIGHT = 0.05       # base z below this -> fell into a weird pose
FALLEN_GRAVITY_Z = -0.5    # body-frame gravity z above this -> tipped over
MIN_FORWARD_CLEARANCE_M = 0.45  # refuse a forward step when an object is closer
# FutureNav's training camera (Habitat R2R RGB sensor): square, HFOV 90.
VLM_FRAME_PX = 224
VLNCE_HFOV_DEG = 90.0


_scene_name = os.environ.get("SCENE", "room")


def _world_half() -> float:
    """Minimap half-extent from the scene manifest (fallback: flat-app value)."""
    manifest = paths.scene_manifest(_scene_name)
    if manifest.exists():
        aabb = json.loads(manifest.read_text())["aabb"]
        return float(max(abs(v) for corner in aabb for v in corner[:2]) + 0.5)
    return 2.5


class RoomSim:
    """Plain-MuJoCo sim + NumPy policy + both cameras + command sources."""

    def __init__(self, scene_xml: Path, policy_npz: Path, vlm_cam: str = "ego"):
        import mujoco

        self._mujoco = mujoco
        # "ego" = onboard ~10 cm cam; "bench" = VLN-CE-style 1.25 m mast cam.
        self.vlm_cam = vlm_cam
        logger.info(f"[{ROBOT_KEY}] scene {scene_xml}, policy {policy_npz}, vlm cam {vlm_cam}")
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, vlm_cam) < 0:
            raise ValueError(f"camera {vlm_cam!r} not in scene (rebuild with run.sh build?)")
        self.data = mujoco.MjData(self.model)
        self.policy = load_fbb_policy(policy_npz)

        names = [self.model.actuator(i).name for i in range(self.model.nu)]
        if names != self.policy.joint_names:
            raise ValueError(
                f"actuator order mismatch: model {names} vs policy {self.policy.joint_names}"
            )
        self._qadr, self._vadr = actuator_addresses(self.model)
        self._substeps = round(self.policy.ctrl_dt / self.model.opt.timestep)

        # Enlarge the offscreen framebuffer so the high-res chase renderer fits
        # (the model default is 640x480, which would cap every renderer there).
        self.model.vis.global_.offwidth = 1024
        self.model.vis.global_.offheight = 768
        # 480x640 (VGA, VLN-CE agent-camera geometry) -- the VLM/bench frame the
        # FutureNav server sees; higher than the old 360x480 for more detail.
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        # Separate, higher-res renderer for the browser chase view only: it fills
        # a large panel, so keep it crisp without inflating the VLM frame (and its
        # FutureNav token cost / inference time).
        self.chase_renderer = mujoco.Renderer(self.model, height=768, width=1024)
        self.cam = mujoco.MjvCamera()
        self.cam.distance = VIEW["distance"]
        self.cam.elevation = VIEW["elevation"]
        self.cam.azimuth = VIEW["azimuth"]

        self.cfg = NavConfig(**NAV)
        self.executor = MidLevelExecutor(self.cfg)
        self._init_online_map()
        self.target: tuple[float, float] | None = None
        self._cmd = (0.0, 0.0, 0.0)
        self.resets = 0  # lets the VLM navigator tell a fall-reset from completion
        self.reset()
        logger.success("room sim ready")

    def _init_online_map(self):
        """Agent-built occupancy map (same one the eval tier uses): fused
        from the ego DEPTH camera at each VLM decision, composited into the
        VLM frame as the minimap HUD and streamed to the demo UI. Grid
        geometry (bounds/resolution) comes from the oracle grid when built,
        else from the manifest AABB -- no cell contents leak from either."""
        from wojtek_eval.gridmap import RESOLUTION
        from wojtek_eval.mapping import OnlineMap

        mujoco = self._mujoco
        self._ego_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.vlm_cam)
        self._ego_fovy = float(self.model.cam_fovy[self._ego_id])

        # Model (FutureNav) frame: a clean square, HFOV-90, VLN-CE-height view
        # matching upstream's Habitat R2R RGB sensor (224x224, HFOV 90). Uses the
        # mast 'bench' camera; its fovy is forced to 90 so a square render yields
        # HFOV 90 (the XML's 73.74 assumed a 4:3 render). Left untouched when it
        # doubles as the browser cam (vlm_cam == "bench"), so the ablation keeps
        # its geometry; falls back to the browser cam if no bench camera exists.
        self._vlm_frame_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "bench")
        if self._vlm_frame_id >= 0:
            self._vlm_frame_cam = "bench"
            if self.vlm_cam != "bench":
                self.model.cam_fovy[self._vlm_frame_id] = VLNCE_HFOV_DEG
        else:
            self._vlm_frame_cam = self.vlm_cam
        self.vlm_renderer = mujoco.Renderer(self.model, height=VLM_FRAME_PX, width=VLM_FRAME_PX)
        occ = paths.scene_dir(_scene_name) / "occupancy.npz"
        if occ.exists():
            from wojtek_eval.gridmap import GridMap

            g = GridMap.load(occ)
            res, origin, shape = g.res, g.origin, g.occ.shape
        else:
            half = _world_half()
            res = RESOLUTION
            origin = (-half, -half)
            shape = (int(2 * half / res), int(2 * half / res))
        self._omap_args = (res, origin, shape)
        self.omap = OnlineMap(res=res, origin=origin, shape=shape)

    def reset(self):
        self.resets += 1
        mujoco = self._mujoco
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        # The home keyframe was settled on the flat training plane; the room's
        # scanned floor hulls sit a couple of cm higher. Settle onto them
        # (ctrl holds the home targets) so the demo doesn't start with a hop.
        for _ in range(int(0.5 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.policy.reset()
        self.executor.clear()
        self.target = None
        self._cmd = (0.0, 0.0, 0.0)
        if hasattr(self, "omap"):
            from wojtek_eval.mapping import OnlineMap

            res, origin, shape = self._omap_args
            self.omap = OnlineMap(res=res, origin=origin, shape=shape)

    def pose(self) -> tuple[float, float, float]:
        q = self.data.qpos
        return float(q[0]), float(q[1]), quat_to_yaw(
            float(q[3]), float(q[4]), float(q[5]), float(q[6])
        )

    # -- command sources (mutually exclusive) ------------------------------

    def submit_command(self, text: str) -> dict:
        try:
            cmd = parse_command(text)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        # Proactive stop: refuse a forward step that would drive into an object
        # right ahead, so the robot halts near the goal instead of ramming it.
        if isinstance(cmd, Forward):
            clr = self._forward_clearance_m()
            if clr < MIN_FORWARD_CLEARANCE_M:
                return {"ok": False, "error": f"obstacle {clr:.2f} m ahead (too close to move forward)"}
        self.target = None
        self.executor.submit(cmd)
        return {"ok": True, "command": text.strip()}

    def _forward_clearance_m(self) -> float:
        """Nearest depth (m) in a central forward cone from the onboard ego cam."""
        r = self.renderer
        r.enable_depth_rendering()
        r.update_scene(self.data, camera="ego")
        depth = r.render().copy()
        r.disable_depth_rendering()
        # Upper-central band: looks ahead at body height, above the near floor
        # (lower rows would read floor distance as a false obstacle).
        h, w = depth.shape[:2]
        cone = depth[int(h * 0.15):int(h * 0.50), int(w * 0.4):int(w * 0.6)]
        return float(cone.min()) if cone.size else float("inf")

    def set_target(self, x: float, y: float):
        self.executor.submit(Stop())
        self.target = (float(x), float(y))

    # -- sim loop -----------------------------------------------------------

    def _fallen(self) -> bool:
        q = self.data.qpos
        gz = gravity_from_quat(float(q[3]), float(q[4]), float(q[5]), float(q[6]))[2]
        return float(q[2]) < FALLEN_HEIGHT or gz > FALLEN_GRAVITY_Z

    def step(self) -> dict:
        x, y, yaw = self.pose()
        reached = False
        dist = 0.0
        if self.executor.active:
            mode = "midlevel"
            self._cmd = self.executor.update(x, y, yaw)
        elif self.target is not None:
            mode = "target"
            vx, vy, vyaw, dist, reached = command_to_target(
                x, y, yaw, self.target[0], self.target[1], self.cfg
            )
            self._cmd = (vx, vy, vyaw)
        else:
            mode = "idle"
            self._cmd = (0.0, 0.0, 0.0)

        targets = self.policy.step(
            None,
            None,
            self.data.qpos[self._qadr],
            self.data.qvel[self._vadr],
            np.array(self._cmd, np.float32),
        )
        self.data.ctrl[:] = targets
        for _ in range(self._substeps):
            self._mujoco.mj_step(self.model, self.data)

        if self._fallen():
            logger.warning("robot fell -- auto reset")
            self.reset()

        nx, ny, nyaw = self.pose()
        self.omap.mark_pose(nx, ny)
        return {
            "type": "state",
            "x": nx, "y": ny, "yaw": nyaw,
            "tx": self.target[0] if self.target else None,
            "ty": self.target[1] if self.target else None,
            "reached": reached,
            "dist": dist,
            "cmd": [round(float(c), 3) for c in self._cmd],
            "mode": mode,
            "exec": self.executor.status(),
        }

    def _jpeg(self, **scene_kw) -> str:
        """Chase view for the browser: rendered on the high-res chase renderer."""
        from PIL import Image

        self.chase_renderer.update_scene(self.data, **scene_kw)
        frame = self.chase_renderer.render()
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def vlm_frame_jpeg(self) -> str:
        """Clean square VLN-CE-style RGB frame for the FutureNav model: no HUD,
        no minimap -- exactly what upstream feeds its policy (224x224, HFOV 90)."""
        from PIL import Image

        self.vlm_renderer.update_scene(self.data, camera=self._vlm_frame_cam)
        frame = self.vlm_renderer.render()
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def render_pair(self) -> tuple[str, str]:
        """(chase, ego) JPEGs from the one shared renderer.

        The chase cam follows BEHIND the robot's heading (smoothed so the
        gait wobble doesn't shake the view): screen-left is always the
        robot's left, so commanded turns read correctly. A world-fixed
        azimuth mirrors the apparent turn direction whenever the robot
        faces the camera.
        """
        x, y, yaw = self.pose()
        target_az = math.degrees(yaw)  # MjvCamera azimuth == yaw -> viewed from behind
        delta = (target_az - self.cam.azimuth + 180.0) % 360.0 - 180.0
        self.cam.azimuth += 0.12 * delta
        self.cam.lookat[:] = [x, y, VIEW["lookat_z"]]
        return self._jpeg(camera=self.cam), self.ego_jpeg()

    def ego_jpeg(self) -> str:
        """VLM frame: ego RGB + self-built minimap HUD. Also the map-update
        tick -- the ego depth image is fused into the online map here, i.e.
        exactly when the VLM looks."""

        from wojtek_eval.mapping import compose_hud, unproject_depth

        self.renderer.update_scene(self.data, camera=self.vlm_cam)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.vlm_cam)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        cam_pos = self.data.cam_xpos[self._ego_id].copy()
        cam_mat = self.data.cam_xmat[self._ego_id].reshape(3, 3).copy()
        pts = unproject_depth(depth, self._ego_fovy, cam_pos, cam_mat, stride=4)
        self.omap.integrate_points(pts, cam_xy=cam_pos[:2])
        frame = compose_hud(rgb, self.omap, self.pose(), px=170)
        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def map_jpeg(self, px: int = 300) -> str:
        """Standalone agent-map panel for the demo UI."""
        from PIL import Image

        img = Image.fromarray(self.omap.map_image(self.pose(), px=px))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("ascii")


# --- FastAPI app ------------------------------------------------------------
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Wojtek room demo")
_STATIC = Path(__file__).resolve().parent.parent / "demo" / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

_sim: RoomSim | None = None
_scene_xml = paths.scene_xml(_scene_name)
_policy_npz = paths.POLICY_NPZ
_navigator: VlmNavigator | None = None
_vlm_backend = os.environ.get("VLM_BACKEND", "local")
_vlm_model = os.environ.get("VLM_MODEL")  # None -> backend default
_vlm_url = os.environ.get("VLM_URL")  # futurenav backend only


def get_sim() -> RoomSim:
    global _sim
    if _sim is None:
        _sim = RoomSim(_scene_xml, _policy_npz)
    return _sim


def get_navigator() -> VlmNavigator:
    """Lazy: the VLM stack (mlx-vlm or anthropic SDK) is only imported on
    first use, so the server runs fine without either extra until someone
    submits a goal."""
    global _navigator
    if _navigator is None:
        if _vlm_backend == "futurenav":
            from wojtek_rl.futurenav_nav import (
                DEFAULT_FUTURENAV_URL,
                FUTURENAV_MAX_ROTATION,
                FUTURENAV_MAX_STEPS,
                FUTURENAV_TIMEOUT_S,
                FutureNavVlmClient,
            )

            client = FutureNavVlmClient(_vlm_url or DEFAULT_FUTURENAV_URL)
            # vlnce_frame: feed the clean square HFOV-90 VLN-CE frame upstream
            # trained on. max_rotation: upstream EARLY_STOP_ROTATION anti-spin.
            # overlap: pipeline the next decision while the current 0.25 m step
            # runs, so the robot keeps moving instead of idling during inference.
            _navigator = VlmNavigator(
                get_sim(), client, max_steps=FUTURENAV_MAX_STEPS,
                vlm_timeout_s=FUTURENAV_TIMEOUT_S, overlap=True,
                vlnce_frame=True, max_rotation=FUTURENAV_MAX_ROTATION,
            )
        elif _vlm_backend == "anthropic":
            client = AnthropicVlmClient(_vlm_model or DEFAULT_MODEL)
            _navigator = VlmNavigator(get_sim(), client)
        else:
            from wojtek_rl.vlm_local import (
                DEFAULT_LOCAL_MODEL,
                LOCAL_VLM_TIMEOUT_S,
                LocalVlmClient,
            )

            client = LocalVlmClient(_vlm_model or DEFAULT_LOCAL_MODEL)
            _navigator = VlmNavigator(get_sim(), client, vlm_timeout_s=LOCAL_VLM_TIMEOUT_S)
    return _navigator


def _preload_local_vlm():
    """Warm the local weights so the first goal doesn't wait for the load."""
    try:
        get_navigator().client.preload()
    except Exception as e:
        # Not fatal: the first goal retries the load and surfaces the real
        # error (missing extra, no disk space, ...) in the UI.
        logger.warning(f"local VLM preload failed: {e}")


@app.on_event("startup")
def _warmup():
    get_sim()
    if _vlm_backend == "local":
        threading.Thread(target=_preload_local_vlm, daemon=True).start()


@app.get("/")
def index():
    page = _STATIC / "room.html"
    return FileResponse(str(page if page.exists() else _STATIC / "index.html"))


_transcriber = None
_transcriber_lock = threading.Lock()


def _get_transcriber():
    global _transcriber
    with _transcriber_lock:
        if _transcriber is None:
            from wojtek_eval.hearing import Transcriber

            _transcriber = Transcriber("small")
            _transcriber._ensure_loaded()
    return _transcriber


@app.on_event("startup")
def _warm_ears():
    # Whisper loads in ~5 s; do it before the first mic press.
    threading.Thread(target=_get_transcriber, daemon=True).start()


@app.post("/api/hear")
async def hear(request: Request):
    """Browser mic audio (webm/opus blob) -> transcript. The transcript is
    NOT auto-submitted: the UI shows what the dog heard and sends it as a
    goal, so a mis-hearing is visible before the robot acts on it."""
    import tempfile

    body = await request.body()
    if len(body) < 200:
        return {"ok": False, "error": "no audio received"}
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(body)
        tmp = Path(f.name)
    try:
        text = await asyncio.to_thread(_get_transcriber().transcribe, tmp)
    except Exception as e:
        logger.warning(f"transcription failed: {e}")
        return {"ok": False, "error": f"transcription failed: {e}"}
    finally:
        tmp.unlink(missing_ok=True)
    logger.info(f"heard: {text!r}")
    return {"ok": True, "transcript": text}


@app.get("/api/info")
def info():
    sim = get_sim()
    res, origin, shape = sim._omap_args
    return {
        "world_half": _world_half(),
        "current": ROBOT_KEY,
        "robots": [{"key": ROBOT_KEY, "label": ROBOT_LABEL, "env": "RoomSim"}],
        "vlm_backend": _vlm_backend,
        "vlm_model": _vlm_model,
        "vlm_url": _vlm_url,
        "map_geom": {"res": res, "origin": list(origin), "shape": list(shape)},
    }


# The control loop below steps the ONE shared sim, so only one websocket
# client may drive it at a time -- a second concurrent loop would step
# physics twice per tick and corrupt the rollout.
_ws_active = False


@app.websocket("/ws")
async def ws(sock: WebSocket):
    global _ws_active
    await sock.accept()
    if _ws_active:
        await sock.send_text(json.dumps(
            {"type": "error", "error": "another client is connected (one viewer at a time)"}
        ))
        await sock.close()
        return
    _ws_active = True
    sim = get_sim()
    acks: asyncio.Queue[dict] = asyncio.Queue()

    async def reader():
        try:
            while True:
                raw = await sock.receive_text()
                try:
                    msg = json.loads(raw)
                    t = msg.get("type")
                    # Manual input always wins over an active VLM goal.
                    if t in ("target", "command", "reset") and _navigator and _navigator.running:
                        _navigator.cancel(t)
                    if t == "target":
                        sim.set_target(msg["x"], msg["y"])
                    elif t == "command":
                        ack = sim.submit_command(str(msg.get("text", "")))
                        await acks.put({"type": "command_ack", **ack})
                    elif t == "reset":
                        sim.reset()
                    elif t == "goal":
                        try:
                            ack = get_navigator().start(str(msg.get("text", "")))
                        except Exception as e:  # missing anthropic extra / API key
                            ack = {"ok": False, "error": str(e)}
                        await acks.put({"type": "goal_ack", **ack})
                    elif t == "goal_cancel":
                        if _navigator is not None:
                            _navigator.cancel("user")
                except (ValueError, KeyError, TypeError) as e:
                    # Malformed client message must not kill the reader.
                    logger.warning(f"bad ws message {raw!r}: {e}")
        except WebSocketDisconnect:
            pass

    reader_task = asyncio.create_task(reader())
    dt = 1.0 / CONTROL_HZ
    i = 0
    vlm_rev = -1  # -1 so every (re)connected client gets the current status
    try:
        loop = asyncio.get_event_loop()
        while True:
            t0 = loop.time()
            payload = sim.step()
            payload["robot"] = ROBOT_KEY
            if i % RENDER_EVERY == 0:
                payload["frame"], payload["ego"] = sim.render_pair()
            if i % 50 == 0:  # agent-map panel, 1 Hz is plenty
                payload["map"] = sim.map_jpeg()
            while not acks.empty():
                await sock.send_text(json.dumps(acks.get_nowait()))
            await sock.send_text(json.dumps(payload))
            if _navigator is not None and _navigator.rev != vlm_rev:
                vlm_rev = _navigator.rev
                await sock.send_text(json.dumps({"type": "vlm_status", **_navigator.status()}))
            i += 1
            await asyncio.sleep(max(0.0, dt - (loop.time() - t0)))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        _ws_active = False
        reader_task.cancel()


def main(argv=None):
    global _scene_xml, _policy_npz, _vlm_model, _vlm_backend
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8010")))
    p.add_argument(
        "--scene-name",
        default=os.environ.get("SCENE", "room"),
        help="named scene (room, apartment, ...): resolves the scene XML, "
        "manifest and occupancy grid together (see paths.scene_dir)",
    )
    p.add_argument("--scene", type=Path, default=None,
                   help="explicit scene XML path (overrides --scene-name's XML only)")
    p.add_argument("--policy", type=Path, default=paths.POLICY_NPZ)
    p.add_argument(
        "--vlm-backend",
        choices=("local", "anthropic", "futurenav"),
        default=os.environ.get("VLM_BACKEND", "local"),
        help="local = Qwen3-VL via mlx-vlm on-device; anthropic = Claude API; "
        "futurenav = FutureNav-4B action server (--vlm-url)",
    )
    p.add_argument(
        "--vlm-model",
        default=os.environ.get("VLM_MODEL"),
        help="HuggingFace repo (local) or Anthropic model id; default per backend",
    )
    p.add_argument(
        "--vlm-url",
        default=os.environ.get("VLM_URL"),
        help="FutureNav action server base URL (futurenav backend only)",
    )
    args = p.parse_args(argv)
    global _scene_name, _vlm_url
    _scene_name = args.scene_name
    _scene_xml = args.scene or paths.scene_xml(_scene_name)
    _policy_npz = args.policy
    _vlm_backend, _vlm_model = args.vlm_backend, args.vlm_model
    _vlm_url = args.vlm_url

    import uvicorn

    logger.info(f"serving room demo on http://{args.host}:{args.port}")
    # wsproto backend: uvicorn's default "websockets" impl uses the legacy API
    # removed in websockets>=14, which breaks the WS handshake.
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", ws="wsproto")


if __name__ == "__main__":
    main()
