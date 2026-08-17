"""Room demo: the trained robot walks a photo-textured scanned room.

Plain-MuJoCo counterpart of wojtek_rl.app (which is MJX-only and cannot do the
room's mesh collisions): mujoco.mj_step drives the exported NumPy policy
(wojtek_policy, the same runtime the real robot runs) at 50 Hz inside
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

from wojtek_rl import paths, perf
from wojtek_rl.agent.goals import TERMINAL_STATES, outcome_phrase
from wojtek_rl.agent.llm import DEFAULT_AGENT_MODEL, DEFAULT_AGENT_URL
from wojtek_rl.agent.nav import TracedVlmNavigator
from wojtek_rl.agent.spatial import PoseHistory
from wojtek_rl.agent.voice import SAMPLE_RATE as VOICE_SAMPLE_RATE
from wojtek_rl.midlevel import Forward, MidLevelExecutor, Stop, parse_command
from wojtek_rl.navigation import NavConfig, command_to_target, quat_to_yaw
from wojtek_rl.np_policy import actuator_addresses, gravity_from_quat, load_policy_runtime
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
SENSE_EVERY = 5            # depth into the SCAN map at 10 Hz while walking
# Map panel cadence. 1 Hz was fine for a static occupancy grid; the planner
# panel animates (guidance, spline, the twin cylinders turning with the body),
# so it is worth 5 Hz -- plan_image costs ~1 ms.
MAP_EVERY = 10
# FutureNav's training camera (Habitat R2R RGB sensor): square, HFOV 90.
VLM_FRAME_PX = 224
VLNCE_HFOV_DEG = 90.0
# Whole chat turn (may include several model calls plus tool work). Past this
# the UI gets a failure it can show instead of an endless "thinking…".
CHAT_TURN_TIMEOUT_S = 90.0


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

    def __init__(self, scene_xml: Path, policy_npz: Path, vlm_cam: str = "ego",
                 local_planner: bool = True):
        import mujoco

        self._mujoco = mujoco
        # "ego" = onboard ~10 cm cam; "bench" = VLN-CE-style 1.25 m mast cam.
        self.vlm_cam = vlm_cam
        logger.info(f"[{ROBOT_KEY}] scene {scene_xml}, policy {policy_npz}, vlm cam {vlm_cam}")
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, vlm_cam) < 0:
            raise ValueError(f"camera {vlm_cam!r} not in scene (rebuild with run.sh build?)")
        self.data = mujoco.MjData(self.model)
        self.policy = load_policy_runtime(policy_npz)

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
        self._init_online_map()
        # Mid-level commands go through the SCAN local planner unless it is
        # switched off for an A/B (--no-local-planner): same submit/active/
        # blocked contract either way, so VlmNavigator cannot tell.
        self.scan = self._make_scan() if local_planner else None
        self.executor = self.scan.executor if self.scan else MidLevelExecutor(self.cfg)
        self.target: tuple[float, float] | None = None
        self._cmd = (0.0, 0.0, 0.0)
        self._tick = 0
        self.resets = 0  # lets the VLM navigator tell a fall-reset from completion
        # Chat-agent spatial memory: sim clock + timestamped pose ring buffer
        # ("describe your last 3 seconds"). Both survive fall-resets, like the
        # online map: the robot's history didn't un-happen because it fell.
        self.sim_time = 0.0
        self.pose_history = PoseHistory()
        # Planner footprint cylinders on the chase cam: debug aid, off by
        # default (a UI toggle turns them on).
        self.show_overlay = False
        self.reset()
        logger.success("room sim ready")

    def _make_scan(self):
        """SCAN local planner over its own depth-built map (nothing is read
        from the oracle grid; on the real robot there is no oracle grid)."""
        from wojtek_rl.scan.stack import ScanStack

        return ScanStack(self.model, self.data, dt=1.0 / CONTROL_HZ, cfg=self.cfg)

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
        self._spawn_xy = None
        spawn_env = os.environ.get("WOJTEK_SPAWN")
        if spawn_env:
            sx, sy = (float(v) for v in spawn_env.split(","))
            self._spawn_xy = (sx, sy)
        if occ.exists():
            from wojtek_eval.gridmap import GridMap

            g = GridMap.load(occ)
            res, origin, shape = g.res, g.origin, g.occ.shape
            # The home keyframe spawns at the world origin, but a scanned
            # scene's origin can sit INSIDE furniture (the castle hall has its
            # table cluster there); a spawn inside inflated occupancy makes
            # the planner reject every command -- the robot answers but never
            # moves (live finding 2026-08-14).  Move to the nearest free cell
            # unless WOJTEK_SPAWN chose a spot.
            if self._spawn_xy is None and not g.is_free(0.0, 0.0):
                snapped = g._snap_free(0.0, 0.0, max_r=4.0)
                if snapped is not None:
                    self._spawn_xy = g.cell_to_world(*snapped)
                    logger.info(
                        f"scene origin is occupied; spawning at "
                        f"({self._spawn_xy[0]:.2f}, {self._spawn_xy[1]:.2f})"
                    )
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
        if getattr(self, "_spawn_xy", None) is not None:
            self.data.qpos[0] = self._spawn_xy[0]
            self.data.qpos[1] = self._spawn_xy[1]
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
        if getattr(self, "scan", None) is not None:
            self.scan.reset()
            self.executor = self.scan.executor
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
        # Without the planner, a proactive veto is the only thing standing
        # between "forward 2" and the coffee table. With it, the move is
        # planned around the table instead of refused.
        if isinstance(cmd, Forward) and self.scan is None:
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
        if self.scan is not None:
            self.target = None
            self.scan.executor.goto(float(x), float(y), (self.pose()[0], self.pose()[1]))
            return
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
        if self.scan is not None:
            self._tick += 1
            if self._tick % SENSE_EVERY == 0:
                self.scan.sense(x, y)
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
        self.sim_time += 1.0 / CONTROL_HZ
        self.pose_history.add(self.sim_time, nx, ny, nyaw)
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
        if self.show_overlay:
            self._overlay_footprint(self.chase_renderer)
        frame = self.chase_renderer.render()
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _overlay_footprint(self, renderer) -> None:
        """Draw the planner's twin collision cylinders into a render scene.

        Overlay geoms only: appended to the MjvScene after update_scene, so
        they exist in the browser's chase view and nowhere else -- not in
        physics, not in the ego/VLM frames, not in the depth channel the map
        is built from. This is the visual answer to "is the body model
        actually where I think it is": two translucent yellow columns
        spanning exactly the obstacle band the map stores.
        """
        if self.scan is None:
            return
        import mujoco

        scene = getattr(renderer, "scene", None) or renderer._scene
        fp = self.scan.footprint
        z0, z1 = self.scan.map.cfg.z_step, self.scan.map.cfg.z_up
        x, y, yaw = self.pose()
        front, rear = fp.centers(x, y, yaw)
        for cx, cy in (front[0], rear[0]):
            if scene.ngeom >= scene.maxgeom:
                return
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=np.array([fp.radius, fp.radius, (z1 - z0) / 2.0]),
                pos=np.array([cx, cy, (z0 + z1) / 2.0]),
                mat=np.eye(3).reshape(-1),
                rgba=np.array([1.0, 0.85, 0.2, 0.28], np.float32),
            )
            scene.ngeom += 1

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

    def ego_jpeg(self, hud: bool = True) -> str:
        """VLM frame: ego RGB, optionally with the self-built minimap HUD.
        Also the map-update tick -- the ego depth image is fused into the
        online map here, i.e. exactly when the VLM looks.

        hud=False returns the bare camera view. The HUD exists for the
        navigator, whose prompt explains the minimap legend; a model that was
        NOT told about the inset reads it as part of the room. Measured: the
        search observer scored the inset itself as the target ("a map showing
        a toy's location", bbox exactly over the paste rectangle), approached
        it, lost it at close range, blacklisted the spot and repeated. Any
        consumer whose prompt does not describe the minimap wants hud=False.
        """

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
        if hud:
            frame = compose_hud(rgb, self.omap, self.pose(), px=170)
        else:
            from PIL import Image

            frame = Image.fromarray(rgb)
        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def map_jpeg(self, px: int = 300) -> str:
        """Standalone agent-map panel for the demo UI.

        With the planner on this is the planner's own view -- log-odds map,
        A* guidance, optimised trajectory, twin-cylinder footprint -- which is
        the panel that shows *why* the robot went around something."""
        from PIL import Image

        if self.scan is not None:
            from wojtek_rl.scan.viz import plan_image

            img = Image.fromarray(plan_image(
                self.scan.map, self.pose(), self.scan.planner,
                px=px, footprint=self.scan.footprint,
            ))
        else:
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
# WOJTEK_POLICY overrides the keeper for this sim process (np_policy honours
# it too, but only when the ref is None -- and this module always passes one).
_policy_npz = os.environ.get("WOJTEK_POLICY") or paths.DEFAULT_POLICY
_navigator: VlmNavigator | None = None
_vlm_backend = os.environ.get("VLM_BACKEND", "local")
_vlm_model = os.environ.get("VLM_MODEL")  # None -> backend default
_vlm_url = os.environ.get("VLM_URL")  # futurenav backend only
_local_planner = os.environ.get("LOCAL_PLANNER", "1") != "0"
# Chat agent (wojtek_rl.agent): its own OpenAI-compatible endpoint/model,
# independent of the navigation backend -- one vLLM server can serve both.
_agent_url = os.environ.get("AGENT_URL")
_agent_model = os.environ.get("AGENT_MODEL")
_vlm_max_steps = int(os.environ.get("VLM_MAX_STEPS") or 0) or None
# Voice mode: Polish by default, because that is what this demo is for. The
# open Qwen Omni weights do not cover Polish speech in EITHER direction, so
# hearing is Whisper and speaking is a separate TTS -- see docs/agent.md.
_asr_language = os.environ.get("ASR_LANGUAGE", "pl")
# ASR_URL points recognition at a GPU box (wojtek_rl.agent.asr_server).
# Unset = run whisper in-process on the CPU, which is ~2x realtime for
# large-v3 and adds seconds of dead air to every exchange.
_asr_url = os.environ.get("ASR_URL")
_asr_model = os.environ.get("ASR_MODEL", "large-v3")
_asr_device = os.environ.get("ASR_DEVICE", "cpu")
_asr_compute = os.environ.get("ASR_COMPUTE", "int8")
_tts_engine_kind = os.environ.get("TTS_ENGINE", "piper")
_tts_voice = os.environ.get("TTS_VOICE", "pl_PL-mc_speech-medium")
_tts_engine = None
# Everything in this system runs in English -- reasoning, tools, traces --
# except the sentence the dog speaks aloud, which is ASR_LANGUAGE. `direct`
# has the model write that sentence itself; `translate` adds a rendering call
# (slower and worse on a 4B, see docs/polish-voice.md).
_lang_mode = os.environ.get("AGENT_LANG_MODE", "direct")
_agent = None
_agent_llm = None
_goals = None
_trace = None
_trace_path = os.environ.get("AGENT_TRACE")  # None -> runs/agent_traces/<scene>_<stamp>.jsonl


def get_trace():
    """Session trace, created on first use and appended to for the life of
    the process (a scene switch keeps the same file: it is one session)."""
    global _trace
    if _trace is None:
        from wojtek_rl.agent.trace import Trace, default_trace_path

        path = (
            Path(_trace_path)
            if _trace_path
            else default_trace_path(paths.PROJECT_DIR / "runs", _scene_name)
        )
        _trace = Trace(path)
        # Stage timings ride the same file: a session trace IS the latency
        # recording, readable with `./training/run.sh perf <file>`.
        perf.bind(_trace)
        logger.info(f"agent trace -> {path}")
        _trace.add("session.start", scene=_scene_name, agent_model=_agent_model,
                   agent_url=_agent_url, nav_backend=_vlm_backend)
    return _trace


def get_sim() -> RoomSim:
    global _sim
    if _sim is None:
        _sim = RoomSim(_scene_xml, _policy_npz, local_planner=_local_planner)
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
            _navigator = TracedVlmNavigator(
                get_sim(), client, max_steps=FUTURENAV_MAX_STEPS,
                vlm_timeout_s=FUTURENAV_TIMEOUT_S, overlap=True,
                vlnce_frame=True, max_rotation=FUTURENAV_MAX_ROTATION,
                trace=get_trace(),
            )
        elif _vlm_backend == "anthropic":
            client = AnthropicVlmClient(_vlm_model or DEFAULT_MODEL)
            _navigator = TracedVlmNavigator(get_sim(), client, trace=get_trace())
        elif _vlm_backend == "openai":
            # Any OpenAI-compatible server (vLLM). Defaults to the SAME
            # endpoint/model as the chat agent, so one served model drives the
            # goal box, the agent's `navigate` tool and the search observer --
            # no second GPU process, no 18 GB on-device download.
            from wojtek_eval.vlm_openai import OpenAIVlmClient
            from wojtek_rl.agent.nav import (
                INSTRUCTION_PROMPT,
                NAV_MAX_ROTATION,
                NAV_MAX_STEPS,
            )
            from wojtek_rl.vlm_nav import ACTIONS

            client = OpenAIVlmClient(
                base_url=_vlm_url or _agent_url or DEFAULT_AGENT_URL,
                model=_vlm_model or _agent_model or DEFAULT_AGENT_MODEL,
                # The client's own defaults are wojtek_eval's, whose action set
                # includes frontier `explore` -- Tier-A-only, so RoomSim
                # rejects it and every step fails. Pin the base nav contract,
                # extended for the multi-step routes the chat agent forwards.
                system_prompt=INSTRUCTION_PROMPT,
                actions=ACTIONS,
            )
            _navigator = TracedVlmNavigator(
                get_sim(), client, max_steps=_vlm_max_steps or NAV_MAX_STEPS,
                max_rotation=NAV_MAX_ROTATION, trace=get_trace(),
            )
        else:
            from wojtek_rl.vlm_local import (
                DEFAULT_LOCAL_MODEL,
                LOCAL_VLM_TIMEOUT_S,
                LocalVlmClient,
            )

            client = LocalVlmClient(_vlm_model or DEFAULT_LOCAL_MODEL)
            _navigator = TracedVlmNavigator(
                get_sim(), client, vlm_timeout_s=LOCAL_VLM_TIMEOUT_S, trace=get_trace()
            )
    return _navigator


def get_agent_llm():
    """One shared chat client for the agent brain AND the search observer."""
    global _agent_llm
    if _agent_llm is None:
        from wojtek_rl.agent.llm import AgentLLM

        _agent_llm = AgentLLM(
            base_url=_agent_url or _vlm_url or DEFAULT_AGENT_URL,
            model=_agent_model or DEFAULT_AGENT_MODEL,
        )
    return _agent_llm


def get_goals():
    """The goal state machine: navigate delegates to the shared VlmNavigator,
    search builds its controller lazily over the shared agent LLM."""
    global _goals
    if _goals is None:
        from wojtek_rl.agent.goals import GoalManager
        from wojtek_rl.agent.search import SearchController, make_score_view

        def search_factory():
            sim = get_sim()
            return SearchController(
                sim,
                make_score_view(get_agent_llm()),
                hfov_deg=sim._ego_fovy,
                # Bare camera: the observer is never told about the minimap
                # HUD and otherwise detects the target inside its own inset.
                frame_fn=lambda: sim.ego_jpeg(hud=False),
                trace=get_trace(),
            )

        _goals = GoalManager(
            navigator_factory=get_navigator,
            search_factory=search_factory,
            trace=get_trace(),
            language=_asr_language,
        )
    return _goals


def get_agent():
    global _agent
    if _agent is None:
        from wojtek_rl.agent.chat import WojtekAgent
        from wojtek_rl.agent.tools import build_tools

        sim = get_sim()
        # One dict shared by the loop and its tools: the agent writes the raw
        # user text each turn, so `navigate` can forward the instruction as
        # spoken instead of whatever noun phrase the model distilled.
        turn_context: dict = {}
        from wojtek_rl.agent.search import make_score_view

        score_view = make_score_view(get_agent_llm())

        async def visibility_check(target: str):
            """Is `target` in the CURRENT camera view?  Drives the
            navigate-vs-search decision: walking toward something not in
            view is a guess, so navigate redirects to search."""
            try:
                frame = sim.ego_jpeg(hud=False)
            except TypeError:
                frame = sim.ego_jpeg()
            vs = await score_view(target, frame)
            return vs.visible

        tools = build_tools(
            sim, get_goals(), sim.pose_history, turn_context=turn_context,
            visibility_check=visibility_check,
        )
        _agent = WojtekAgent(
            get_agent_llm(), tools, trace=get_trace(), turn_context=turn_context,
            lang_mode=_lang_mode, reply_language=_asr_language,
        )
        logger.info(f"agent language: mode={_lang_mode} reply={_asr_language}")
    return _agent


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
        if _transcriber is None and _asr_url:
            from wojtek_rl.agent.voice import RemoteTranscriber

            _transcriber = RemoteTranscriber(_asr_url)
            try:
                logger.info(f"ASR: remote {_asr_url} -> {_transcriber.health()}")
            except Exception as e:
                logger.warning(f"remote ASR at {_asr_url} unreachable: {e}")
        if _transcriber is None:
            from wojtek_eval.hearing import Transcriber

            # Polish needs a bigger model than the eval battery's "small":
            # large-v3 is the best-measured open Polish recogniser, and int8
            # keeps it around 2.5 GB so it can share a card with the brain.
            logger.info(f"ASR: faster-whisper {_asr_model} ({_asr_compute}) lang={_asr_language}")
            _transcriber = Transcriber(
                _asr_model,
                device=_asr_device,
                compute_type=_asr_compute,
                language=_asr_language,
            )
            _transcriber._ensure_loaded()
    return _transcriber


def get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        from wojtek_rl.agent.tts import build_engine

        _tts_engine = build_engine(_tts_engine_kind, _tts_voice)
        logger.info(f"TTS: {type(_tts_engine).__name__} voice={_tts_voice}")
    return _tts_engine


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


def _available_scenes() -> list[str]:
    """Scenes whose XML and manifest both exist (built + assets present)."""
    names = {"room"}
    if paths.SCENES_DIR.exists():
        names |= {d.name for d in paths.SCENES_DIR.iterdir() if d.is_dir()}
    return sorted(
        n for n in names
        if paths.scene_xml(n).exists() and paths.scene_manifest(n).exists()
    )


@app.post("/api/scene")
async def set_scene(request: Request):
    """Switch the scanned scene: rebuild the ONE shared sim in place.

    Async on purpose -- it runs on the same event loop as the websocket
    control loop, so the loop is parked between ticks while the new model
    loads: no concurrent stepping, no lock. The client reloads the page
    afterwards, which refreshes /api/info (minimap extent, map geometry).
    """
    global _sim, _navigator, _scene_name, _scene_xml, _agent, _goals
    body = await request.json()
    name = str(body.get("name", ""))
    have = _available_scenes()
    if name not in have:
        return {"ok": False, "error": f"unknown scene {name!r} (have {have})"}
    if name == _scene_name and _sim is not None:
        return {"ok": True, "scene": name}
    if _goals is not None:
        _goals.cancel("scene change")
        _goals = None
    _agent = None  # its tools close over the old sim's map/pose history
    if _navigator is not None:
        _navigator.cancel("scene change")
        _navigator = None
    old_sim, old_name, old_xml = _sim, _scene_name, _scene_xml
    _scene_name, _scene_xml, _sim = name, paths.scene_xml(name), None
    try:
        get_sim()
    except Exception as e:  # keep the working scene rather than a dead server
        logger.exception(f"scene switch to {name} failed")
        _sim, _scene_name, _scene_xml = old_sim, old_name, old_xml
        return {"ok": False, "error": str(e)}
    if old_sim is not None:  # free the old GL renderers
        for r in ("renderer", "chase_renderer", "vlm_renderer"):
            try:
                getattr(old_sim, r).close()
            except Exception:
                pass
        if old_sim.scan is not None:
            old_sim.scan.close()
    logger.success(f"scene switched to {name}")
    return {"ok": True, "scene": name}


@app.get("/api/footprint")
def get_footprint():
    sim = get_sim()
    if sim.scan is None:
        return {"ok": False, "error": "planner off"}
    fp = sim.scan.footprint
    return {"ok": True, "d_off": fp.d_off, "radius": fp.radius, "length": round(fp.length, 3)}


@app.post("/api/footprint")
async def set_footprint(request: Request):
    """Live-tune the twin-cylinder footprint (the /tune helper page).

    Visual/geometry tuning only: the overlay, the planner's collision probes
    and the turn-sweep check follow immediately, but the MAP's inflation
    kernel keeps the radius it was built with -- rebuilding the dilation per
    slider tick is not worth it for eyeballing body coverage. Once a size
    looks right, it belongs in Footprint's defaults and MapConfig.inflate_r.
    """
    from dataclasses import replace

    sim = get_sim()
    if sim.scan is None:
        return {"ok": False, "error": "planner off"}
    body = await request.json()
    fp = sim.scan.footprint
    d_off = min(max(float(body.get("d_off", fp.d_off)), 0.0), 0.40)
    radius = min(max(float(body.get("radius", fp.radius)), 0.05), 0.40)
    fp = replace(fp, d_off=d_off, radius=radius)
    sim.scan.footprint = fp
    sim.scan.executor.planner.fp = fp
    return {"ok": True, "d_off": fp.d_off, "radius": fp.radius, "length": round(fp.length, 3)}


@app.get("/api/chase.jpg")
async def chase_jpg():
    """One chase frame on demand, cylinders included.

    Lets the /tune page watch the robot without taking the single websocket
    viewer slot. Async on purpose: it renders on the event loop, serialized
    with the control loop, so there is no concurrent use of the renderer.
    """
    from fastapi.responses import Response

    sim = get_sim()
    x, y, _ = sim.pose()
    sim.cam.lookat[:] = [x, y, VIEW["lookat_z"]]
    return Response(content=base64.b64decode(sim._jpeg(camera=sim.cam)),
                    media_type="image/jpeg")


@app.get("/api/top.jpg")
async def top_jpg():
    """Straight-down snapshot, body-aligned (nose up), cylinders overlaid.

    The tuner's second panel: radius reads best from above, d_off from the
    side, so it shows both.
    """
    from fastapi.responses import Response

    import mujoco

    sim = get_sim()
    x, y, yaw = sim.pose()
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [x, y, 0.10]
    cam.distance = 1.5
    cam.elevation = -90.0
    cam.azimuth = math.degrees(yaw)
    return Response(content=base64.b64decode(sim._jpeg(camera=cam)),
                    media_type="image/jpeg")


@app.get("/tune")
def tune():
    return FileResponse(str(_STATIC / "tune.html"))


@app.get("/api/trace")
def api_trace(limit: int = 200, kind: str | None = None):
    """Recent session trace: what fired, when, with what result.

    The live debug panels answer "what is it doing now"; this answers "what
    did it do ten minutes ago", and survives a browser reload. `kind` is a
    prefix filter (`chat`, `search`, `nav`, `goal`). The same events are on
    disk as JSONL at `path`.
    """
    if _trace is None:
        return {"ok": True, "path": None, "events": [], "note": "nothing traced yet"}
    return {
        "ok": True,
        "path": str(_trace.path) if _trace.path else None,
        "total": _trace.seq,
        "events": _trace.recent(limit=min(max(limit, 1), 1000), kind_prefix=kind),
    }


@app.get("/api/perf")
def api_perf():
    """Where this session's time is going, ranked, live.

    Built from the in-memory ring, so it covers the recent past rather than
    the whole session -- the trace FILE is complete, and
    `./training/run.sh perf <path>` reports on all of it.
    """
    from wojtek_rl import perf_report

    if _trace is None:
        return {"ok": True, "path": None, "stages": [], "note": "nothing timed yet"}
    events = _trace.recent(limit=_trace.ring.maxlen or 2000)
    return {
        "ok": True,
        "path": str(_trace.path) if _trace.path else None,
        "partial": _trace.seq > len(events),
        **perf_report.report(events),
    }


@app.get("/api/agent_health")
async def agent_health():
    """Is the chat agent's endpoint actually serving? The UI header turns red
    when it is not, so a dead vLLM box is visible before someone types."""
    llm = get_agent_llm()
    try:
        client = llm._ensure_client()
        resp = await client.get(f"{llm.base_url}/models", timeout=5.0)
        models = [m.get("id") for m in (resp.json().get("data") or [])]
        return {"ok": resp.status_code == 200, "url": llm.base_url, "models": models}
    except Exception as e:
        return {"ok": False, "url": llm.base_url, "error": str(e)}


@app.post("/api/chat")
async def api_chat(request: Request):
    """One chat turn with the dog, for curl/scripts (the demo UI uses /ws).

    Async on purpose: tool calls render on the event loop, serialized with
    the control loop. Long tool turns park the sim between ticks, same
    trade the scene switch makes.
    """
    body = await request.json()
    text = str(body.get("text", ""))
    try:
        return await asyncio.wait_for(get_agent().ask(text), CHAT_TURN_TIMEOUT_S)
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"no answer within {CHAT_TURN_TIMEOUT_S:g}s"}
    except Exception as e:
        logger.exception("chat turn failed")
        return {"ok": False, "error": str(e)}


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
        # Chat agent endpoint, shown in the UI header: the demo is useless to
        # debug if you cannot see which brain the dog is actually talking to.
        "agent_url": _agent_url or _vlm_url or DEFAULT_AGENT_URL,
        "agent_model": _agent_model or DEFAULT_AGENT_MODEL,
        # Voice stack, so the UI can say what it is actually running (and
        # whether it will be able to speak at all).
        "voice": {
            "asr_model": _asr_model,
            "asr_language": _asr_language,
            "asr_url": _asr_url,
            "tts_engine": type(get_tts_engine()).__name__,
            "tts_voice": _tts_voice,
            "sample_rate": VOICE_SAMPLE_RATE,
            "lang_mode": _lang_mode,
        },
        "local_planner": sim.scan is not None,
        "scene": _scene_name,
        "scenes": _available_scenes(),
        "map_geom": {"res": res, "origin": list(origin), "shape": list(shape)},
    }


# The control loop below steps the ONE shared sim, so only one websocket
# client may drive it at a time -- a second concurrent loop would step
# physics twice per tick and corrupt the rollout. Rather than lock the sim to
# whoever connected first (a stale tab then wedges every later one into a
# reconnect loop that only says "disconnected"), the NEWEST viewer takes over:
# each connection claims a generation number and older loops exit on their
# next tick.
_ws_gen = 0


@app.websocket("/ws")
async def ws(sock: WebSocket):
    global _ws_gen
    await sock.accept()
    _ws_gen += 1
    mine = _ws_gen
    # Let the previous loop notice it has been superseded and stop stepping.
    await asyncio.sleep(2.5 / CONTROL_HZ)
    sim = get_sim()
    acks: asyncio.Queue[dict] = asyncio.Queue()
    chat_task: asyncio.Task | None = None

    # -- voice: mic frames in, spoken reply out (both binary ws frames) ----
    from wojtek_rl.agent.tts import Speaker
    from wojtek_rl.agent.voice import VoiceListener

    async def send_audio(pcm_bytes: bytes):
        await sock.send_bytes(pcm_bytes)

    speaker = Speaker(get_tts_engine(), send_audio)

    async def on_heard(text: str, utt):
        """One recognised utterance -> a chat turn -> a spoken reply."""
        await acks.put({"type": "heard", "text": text, "seconds": round(utt.seconds, 1)})
        if chat_task is not None and not chat_task.done():
            # Preempt rather than drop: if you said something new, the answer
            # being composed for the previous sentence is already stale.
            logger.info("new utterance preempts the turn in flight")
            chat_task.cancel()
            speaker.cancel()
        start_chat(text, spoken=True)

    listener = VoiceListener(_get_transcriber(), on_heard)

    # Live decision feed: every trace event (model output, tool call, FSM
    # transition, FutureNav action) reaches the browser as it happens.
    def on_trace(event: dict):
        try:
            acks.put_nowait({"type": "trace", **event})
        except asyncio.QueueFull:
            pass
        # Say the goal switch immediately, before the model's own reply: the
        # robot is already turning away from the old target by now.
        if event.get("kind") == "goal.switch" and event.get("text"):
            speaker.say(event["text"])
            acks.put_nowait({"type": "chat_reply", "ok": True, "say": event["text"],
                             "steps": [], "spoken": True})

    get_trace().subscribe(on_trace)

    async def chat_turn(text: str, spoken: bool = False):
        # A spoken turn's clock already started when the mic closed the
        # utterance; a typed one starts here, so both kinds get a critical
        # path in the report.
        if perf.current_turn() is None:
            perf.start_turn("text", chars=len(text))
        # A turn that never returns leaves the UI on "thinking…" forever, which
        # is what a dead endpoint used to look like. Bound it and report why.
        try:
            result = await asyncio.wait_for(
                get_agent().ask(text, voice=spoken), CHAT_TURN_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            logger.warning(f"chat turn timed out after {CHAT_TURN_TIMEOUT_S:g}s")
            result = {
                "ok": False,
                "error": f"no answer within {CHAT_TURN_TIMEOUT_S:g}s "
                f"(is the agent model at {_agent_url or DEFAULT_AGENT_URL} reachable?)",
            }
        except Exception as e:  # endpoint down / misconfigured
            logger.exception("chat turn failed")
            result = {"ok": False, "error": str(e)}
        await acks.put({"type": "chat_reply", **result, "spoken": spoken})
        if spoken and result.get("ok"):
            # The answer is on screen from here; everything after this is the
            # gap between reading it and hearing it, which is the complaint
            # people actually voice about the demo.
            perf.mark("reply.text")
            speaker.say(result.get("say", ""))

    def start_chat(text: str, spoken: bool = False):
        nonlocal chat_task
        chat_task = asyncio.create_task(chat_turn(text, spoken=spoken))

    async def reader():
        nonlocal chat_task
        try:
            while True:
                # Two kinds of client frame: JSON control messages, and raw
                # PCM16 from the mic worklet. receive() demuxes both.
                packet = await sock.receive()
                if packet.get("type") == "websocket.disconnect":
                    break
                if packet.get("bytes") is not None:
                    # Barge-in: the dog stops talking the moment the human
                    # starts. Checked before the frame is consumed, so the
                    # first voiced frame already silences playback.
                    was_speaking = listener.seg.speaking
                    await listener.feed_frame(packet["bytes"])
                    if listener.seg.speaking and not was_speaking and speaker.speaking:
                        speaker.cancel()
                        await acks.put({"type": "barge_in"})
                    continue
                raw = packet.get("text")
                if raw is None:
                    continue
                try:
                    msg = json.loads(raw)
                    t = msg.get("type")
                    # Manual input always wins over an active VLM goal.
                    if t in ("target", "command", "reset"):
                        if _navigator and _navigator.running:
                            _navigator.cancel(t)
                        if _goals is not None:
                            _goals.cancel(t)
                    if t == "target":
                        get_sim().set_target(msg["x"], msg["y"])
                    elif t == "command":
                        ack = get_sim().submit_command(str(msg.get("text", "")))
                        await acks.put({"type": "command_ack", **ack})
                    elif t == "reset":
                        get_sim().reset()
                    elif t == "goal":
                        try:
                            ack = get_navigator().start(str(msg.get("text", "")))
                        except Exception as e:  # missing anthropic extra / API key
                            ack = {"ok": False, "error": str(e)}
                        await acks.put({"type": "goal_ack", **ack})
                    elif t == "goal_cancel":
                        if _navigator is not None:
                            _navigator.cancel("user")
                        if _goals is not None:
                            _goals.cancel("user")
                    elif t == "chat":
                        if chat_task is not None and not chat_task.done():
                            # The newest instruction wins. Dropping it made the
                            # robot look like it ignored you; the old turn's
                            # answer is stale the moment you say something else.
                            logger.info("new message preempts the turn in flight")
                            chat_task.cancel()
                            speaker.cancel()
                            start_chat(str(msg.get("text", "")), spoken=listener.enabled)
                        else:
                            # Speak the answer whenever the mic is live, even
                            # if this particular question was typed: once you
                            # are in a conversation, replies should be heard.
                            start_chat(str(msg.get("text", "")), spoken=listener.enabled)
                    elif t == "chat_reset":
                        if _agent is not None:
                            _agent.reset()
                    elif t == "voice":
                        on = bool(msg.get("on"))
                        listener.set_enabled(on)
                        if not on:
                            speaker.cancel()
                        await acks.put({"type": "voice_state", "on": on})
                    elif t == "shush":  # user hit stop while the dog talks
                        speaker.cancel()
                    elif t == "overlay":
                        get_sim().show_overlay = bool(msg.get("on"))
                        await acks.put({"type": "overlay_state",
                                        "on": get_sim().show_overlay})
                except (ValueError, KeyError, TypeError) as e:
                    # Malformed client message must not kill the reader.
                    logger.warning(f"bad ws message {raw!r}: {e}")
        except WebSocketDisconnect:
            pass

    reader_task = asyncio.create_task(reader())
    dt = 1.0 / CONTROL_HZ
    # The loop's stages fire at CONTROL_HZ; one trace event per tick per stage
    # would bury the decision events, so they are rolled up per window.
    meter = perf.Meter(trace=get_trace())
    i = 0
    vlm_rev = -1  # -1 so every (re)connected client gets the current status
    agent_rev = -1
    announced: tuple | None = None  # last outcome spoken, so it is said once
    try:
        loop = asyncio.get_event_loop()
        while True:
            if mine != _ws_gen:  # a newer viewer took over
                await sock.send_text(json.dumps(
                    {"type": "error", "error": "another viewer took over this tab"}
                ))
                break
            t0 = loop.time()
            sim = get_sim()  # /api/scene may have swapped it since last tick
            with meter.time("sim.step"):
                payload = sim.step()
            payload["robot"] = ROBOT_KEY
            if i % RENDER_EVERY == 0:
                with meter.time("sim.render_pair"):
                    payload["frame"], payload["ego"] = sim.render_pair()
            if i % MAP_EVERY == 0:
                with meter.time("sim.map_jpeg"):
                    payload["map"] = sim.map_jpeg()
            while not acks.empty():
                await sock.send_text(json.dumps(acks.get_nowait()))
            with meter.time("ws.send"):
                await sock.send_text(json.dumps(payload))
            if _navigator is not None and _navigator.rev != vlm_rev:
                vlm_rev = _navigator.rev
                await sock.send_text(json.dumps({"type": "vlm_status", **_navigator.status()}))
            if _goals is not None and _goals.rev != agent_rev:
                agent_rev = _goals.rev
                status = _goals.status()
                await sock.send_text(json.dumps({"type": "agent_status", **status}))
                # Say out loud how the goal ended. A behaviour that finishes
                # silently is indistinguishable from one still running.
                key = (status.get("kind"), status.get("state"), status.get("goal"))
                if status.get("state") in TERMINAL_STATES and key != announced:
                    announced = key
                    phrase = outcome_phrase(
                        status.get("kind") or "",
                        status.get("state") or "",
                        status.get("goal") or "",
                        reason=(status.get("detail") or {}).get("reason"),
                        language=_asr_language,
                    )
                    if phrase:
                        logger.info(f"announcing outcome: {phrase!r}")
                        speaker.say(phrase)
                        get_trace().add("goal.announce", text=phrase, goal_state=status.get("state"))
                        await acks.put({"type": "chat_reply", "ok": True, "say": phrase,
                                        "steps": [], "spoken": True})
            i += 1
            # The whole tick, including the parts not metered separately: when
            # this exceeds dt the robot's control loop is running slow, which
            # looks like sluggish walking rather than a slow answer.
            meter.add("loop.tick", (loop.time() - t0) * 1000.0)
            meter.tick()
            await asyncio.sleep(max(0.0, dt - (loop.time() - t0)))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        reader_task.cancel()
        if chat_task is not None and not chat_task.done():
            chat_task.cancel()
        speaker.cancel()
        meter.flush(force=True)  # the last window still counts
        get_trace().unsubscribe(on_trace)


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
    p.add_argument("--policy", default=os.environ.get("WOJTEK_POLICY") or paths.DEFAULT_POLICY)
    p.add_argument(
        "--no-local-planner",
        action="store_true",
        help="execute mid-level commands as straight marches (pre-SCAN "
        "behaviour); the A/B baseline for obstacle avoidance",
    )
    p.add_argument(
        "--vlm-backend",
        choices=("local", "anthropic", "futurenav", "openai"),
        default=os.environ.get("VLM_BACKEND", "local"),
        help="local = Qwen3-VL via mlx-vlm on-device; anthropic = Claude API; "
        "futurenav = FutureNav-4B action server (--vlm-url); openai = any "
        "OpenAI-compatible server, by default the same one as --agent-url",
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
    p.add_argument(
        "--agent-url",
        default=os.environ.get("AGENT_URL"),
        help="chat agent's OpenAI-compatible base URL (default: --vlm-url, "
        "else http://127.0.0.1:8000)",
    )
    p.add_argument(
        "--agent-model",
        default=os.environ.get("AGENT_MODEL"),
        help="chat agent model id served at --agent-url "
        "(default: Qwen/Qwen3-VL-4B-Instruct-FP8)",
    )
    p.add_argument(
        "--vlm-max-steps",
        type=int,
        default=int(os.environ.get("VLM_MAX_STEPS") or 0) or None,
        help="cap the decisions in one navigation goal. Unset (the default on "
        "the openai backend) means no cap: the goal ends when the model says "
        "done/stop, you cancel, it spins in place, or it is wedged. Set this "
        "only for benchmark-shaped, comparable episodes",
    )
    args = p.parse_args(argv)
    global _scene_name, _vlm_url, _local_planner, _agent_url, _agent_model, _vlm_max_steps
    _vlm_max_steps = args.vlm_max_steps
    _local_planner = not args.no_local_planner
    _scene_name = args.scene_name
    _scene_xml = args.scene or paths.scene_xml(_scene_name)
    _policy_npz = args.policy
    _vlm_backend, _vlm_model = args.vlm_backend, args.vlm_model
    _vlm_url = args.vlm_url
    _agent_url, _agent_model = args.agent_url, args.agent_model

    import uvicorn

    logger.info(f"serving room demo on http://{args.host}:{args.port}")
    # wsproto backend: uvicorn's default "websockets" impl uses the legacy API
    # removed in websockets>=14, which breaks the WS handshake.
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", ws="wsproto")


if __name__ == "__main__":
    main()
