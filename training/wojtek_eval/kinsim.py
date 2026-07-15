"""Tier-A kinematic sim: the VLM navigation loop without legged physics.

Exposes the same surface VlmNavigator drives in the physics room app
(ego_jpeg / pose / submit_command / executor.active / executor.blocked /
resets), but commands move the base kinematically on the oracle occupancy
grid: forward/backward march in 2.5 cm substeps and stop at the first
non-free cell (reported as blocked, mirroring MidLevelExecutor's
stall-abort). Rendering is real MuJoCo (photorealistic scan), and every VLM
frame fuses the ego DEPTH image into the agent's OnlineMap -- the minimap the
VLM sees is only what the camera has seen.

Execution is deferred to the first executor.active poll so VlmNavigator's
blocked-detection (which snapshots executor.blocked after submit, before
polling) keeps working unchanged.

Two eval-only extensions over the physics app:
  - the VLM input frame is a composite: ego RGB + top-down minimap HUD;
  - an `explore` action (wojtek_eval.navigator.EvalNavigator) that walks
    toward the nearest map frontier.
"""

from __future__ import annotations

import base64
import io
import math

import numpy as np

from wojtek_rl import paths
from wojtek_rl.midlevel import Backward, Forward, Stop, Turn, parse_command
from wojtek_eval.gridmap import GridMap
from wojtek_eval.mapping import FrontierPlanner, OnlineMap, unproject_depth

SUBSTEP_M = 0.025
RENDER_W, RENDER_H = 640, 480
MINIMAP_PX = 210
JPEG_QUALITY = 82
DEPTH_STRIDE = 4


class DeferredExecutor:
    """Executes the queued command on the first `active` poll (see module
    docstring for why), then reports idle. `blocked` counts collision-
    truncated moves, same contract as MidLevelExecutor."""

    def __init__(self, sim: "KinematicSim"):
        self._sim = sim
        self._pending: Turn | Forward | Backward | None = None
        self.blocked = 0

    def submit(self, cmd) -> None:
        if isinstance(cmd, Stop):
            self._pending = None
            return
        self._pending = cmd

    @property
    def active(self) -> bool:
        if self._pending is not None:
            cmd, self._pending = self._pending, None
            self._sim._execute(cmd)
        return False

    def status(self) -> dict:
        return {"active": None, "remaining": 0.0, "queued": 0}


class KinematicSim:
    """Photorealistic render + grid-kinematic motion for one scene."""

    def __init__(
        self,
        scene_name: str,
        start: tuple[float, float, float] = (0, 0, 0),
        vlm_cam: str = "ego",
        hud: bool = True,
    ):
        import mujoco

        self._mujoco = mujoco
        self.scene_name = scene_name
        # What the VLM sees: which camera renders its RGB frame ("ego" =
        # onboard ~10 cm cam, "bench" = VLN-CE-style 1.25 m mast cam) and
        # whether the minimap HUD is composited in. Depth for the online map
        # always comes from the ego camera -- that is the robot's actual
        # depth sensor. VLN-trained backends (futurenav) never saw HUDs:
        # run them with hud=False.
        self.vlm_cam = vlm_cam
        self.hud = hud
        self.model = mujoco.MjModel.from_xml_path(str(paths.scene_xml(scene_name)))
        self.data = mujoco.MjData(self.model)
        self.grid = GridMap.load(paths.scene_dir(scene_name) / "occupancy.npz")

        free_jnt = [
            j for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(free_jnt) != 1:
            raise ValueError(f"expected one free joint, found {len(free_jnt)}")
        self._qadr = int(self.model.jnt_qposadr[free_jnt[0]])
        key = self.model.key("home")
        self._home_qpos = np.array(key.qpos)
        self._z0 = float(self._home_qpos[self._qadr + 2])

        self._ego = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
        if self._ego < 0:
            raise ValueError("scene has no 'ego' camera")
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, vlm_cam) < 0:
            raise ValueError(f"scene has no {vlm_cam!r} camera (rebuild with run.sh build?)")
        self._fovy = float(self.model.cam_fovy[self._ego])
        self._renderer = mujoco.Renderer(self.model, height=RENDER_H, width=RENDER_W)

        self.omap = OnlineMap(res=self.grid.res, origin=self.grid.origin,
                              shape=self.grid.shape)
        self._planner = FrontierPlanner(self.grid.res)

        self.resets = 0  # kinematic robots do not fall
        self.executor = DeferredExecutor(self)
        self.path_length = 0.0
        self.commands: list[dict] = []
        self.frame_dir = None  # set to a Path to save each VLM frame (media)
        self._frame_n = 0
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.reset(start)

    # -- state -----------------------------------------------------------------

    def reset(self, start: tuple[float, float, float]) -> None:
        x, y, yaw = start
        if not self.grid.is_free(x, y):
            raise ValueError(f"start ({x:.2f}, {y:.2f}) is not free in {self.scene_name}")
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self._sync_model()
        self.omap.mark_pose(self.x, self.y)

    def pose(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.yaw)

    def _sync_model(self) -> None:
        q = self._home_qpos.copy()
        a = self._qadr
        q[a : a + 3] = (self.x, self.y, self._z0)
        q[a + 3 : a + 7] = (math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2))
        self.data.qpos[:] = q
        self.data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    # -- commands ----------------------------------------------------------------

    def submit_command(self, text: str) -> dict:
        text = text.strip()
        if text == "explore":
            step = self._planner.next_step(self.omap, self.pose())
            if step is None:
                return {"ok": False, "error": "no unexplored frontier left on the map"}
            turn_deg, fwd_m = step
            side = Turn(math.radians(turn_deg))
            self.executor.submit(side)
            _ = self.executor.active  # execute the turn now
            self.executor.submit(Forward(max(fwd_m, 0.1)))
            return {"ok": True}
        try:
            cmd = parse_command(text)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        self.executor.submit(cmd)
        return {"ok": True}

    def _execute(self, cmd) -> None:
        rec = {"cmd": type(cmd).__name__.lower(), "pose0": self.pose(), "blocked": False}
        if isinstance(cmd, Turn):
            self.yaw = (self.yaw + cmd.angle_rad + math.pi) % (2 * math.pi) - math.pi
        else:
            sign = -1.0 if isinstance(cmd, Backward) else 1.0
            remaining = cmd.meters
            dx = sign * math.cos(self.yaw) * SUBSTEP_M
            dy = sign * math.sin(self.yaw) * SUBSTEP_M
            while remaining > 0:
                step = min(SUBSTEP_M, remaining)
                nx = self.x + dx * (step / SUBSTEP_M)
                ny = self.y + dy * (step / SUBSTEP_M)
                if not self.grid.is_free(nx, ny):
                    self.executor.blocked += 1
                    rec["blocked"] = True
                    break
                self.x, self.y = nx, ny
                self.path_length += step
                remaining -= step
                self.omap.mark_pose(self.x, self.y)
        self._sync_model()
        rec["pose1"] = self.pose()
        self.commands.append(rec)

    # -- sensing -----------------------------------------------------------------

    def _render_ego(self) -> tuple[np.ndarray, np.ndarray]:
        """(rgb from the VLM camera, depth from the ego camera)."""
        self._renderer.update_scene(self.data, camera=self.vlm_cam)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera="ego")
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return rgb, depth

    def _integrate_depth(self, depth: np.ndarray) -> None:
        cam_pos = self.data.cam_xpos[self._ego].copy()
        cam_mat = self.data.cam_xmat[self._ego].reshape(3, 3).copy()
        pts = unproject_depth(depth, self._fovy, cam_pos, cam_mat, stride=DEPTH_STRIDE)
        self.omap.integrate_points(pts, cam_xy=cam_pos[:2])

    def ego_jpeg(self) -> str:
        """Composite VLM frame: ego RGB with the agent-map HUD bottom-right.
        Also the map-update tick -- the VLM only 'sees' when this is called."""
        from PIL import Image

        rgb, depth = self._render_ego()
        self._integrate_depth(depth)
        self.omap.mark_pose(self.x, self.y)

        frame = Image.fromarray(rgb)
        if self.hud:
            hud = Image.fromarray(self.omap.map_image(self.pose(), px=MINIMAP_PX))
            # 2px frame so the HUD reads as an inset, not scene content.
            from PIL import ImageOps

            hud = ImageOps.expand(hud, border=2, fill=(250, 220, 60))
            frame.paste(hud, (frame.width - hud.width - 8, frame.height - hud.height - 8))
        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=JPEG_QUALITY)
        if self.frame_dir is not None:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            (self.frame_dir / f"step_{self._frame_n:03d}.jpg").write_bytes(buf.getvalue())
            self._frame_n += 1
        return base64.b64encode(buf.getvalue()).decode()

    def frame_png(self, path=None) -> np.ndarray:
        """Third-person-ish debug/media frame (track camera)."""
        self._renderer.update_scene(self.data, camera="track")
        img = self._renderer.render().copy()
        if path is not None:
            from PIL import Image

            Image.fromarray(img).save(path)
        return img

    def close(self) -> None:
        self._renderer.close()
