"""Offscreen rendering: frame size, the scene model, the onboard depth view.

`SceneView` is the one renderer the video tools use. It owns the chase camera,
the offscreen buffer sizing, and -- when a tool asks for them -- the depth
camera at the height-scan mask's mount and the overlays drawn on each frame.
"""

import argparse
import math

import numpy as np

from wojtek_rl import height_scan
from wojtek_rl.video import overlays

DEFAULT_SIZE = (960, 720)
SCAN_CAM = "scan_probe"
ROOT_BODY = "root"


def frame_size(text):
    """Frame size written as WxH."""
    parts = str(text).lower().split("x")
    try:
        w, h = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"frame size must be WxH, got {text!r}") from None
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"frame size must be positive, got {text!r}")
    return w, h


def mask_camera_frame(mask):
    """(mount, quat, fovy) of a camera sharing the mask's frustum."""
    from wojtek_rl.build_model import _xyaxes_to_quat

    pitch = math.radians(float(mask.pitch_deg))
    mount = [float(v) for v in mask.mount]
    # MuJoCo cameras look down -z with +y up: x right = -y body, up pitched
    # by the mask's angle, which is the frame visible_mask assumes.
    quat = _xyaxes_to_quat([0.0, -1.0, 0.0, math.sin(pitch), 0.0, math.cos(pitch)])
    return mount, quat, float(mask.vfov_deg)


def _named_camera(model, mount, quat, fovy):
    """Name of a root-mounted camera already at that pose and fovy, if any."""
    import mujoco

    root = model.body(ROOT_BODY).id
    return next(
        (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)
            if model.cam_bodyid[i] == root
            and np.allclose(model.cam_pos[i], mount, atol=1e-4)
            and np.allclose(model.cam_quat[i], quat, atol=1e-4)
            and abs(float(model.cam_fovy[i]) - fovy) < 1e-4
        ),
        None,
    )


def scene_model(env, size, mask=None):
    """(model, depth camera name) for rendering the env's scene at `size`.

    Recompiles the scene only when it has to: to widen an offscreen buffer
    smaller than the frame, or to add a camera at `mask`'s mount and pitch
    that the scene does not already carry.
    """
    import mujoco

    model = env.mj_model
    mount, quat, fovy = (None, None, None) if mask is None else mask_camera_frame(mask)
    camera = None if mask is None else _named_camera(model, mount, quat, fovy)
    fits = model.vis.global_.offwidth >= size[0] and model.vis.global_.offheight >= size[1]
    if fits and (mask is None or camera is not None):
        return model, camera
    spec = mujoco.MjSpec.from_file(env.xml_path)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, size[0])
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, size[1])
    if mask is not None and camera is None:
        spec.body(ROOT_BODY).add_camera(
            name=SCAN_CAM, pos=mount, quat=quat.tolist(), fovy=fovy
        )
        camera = SCAN_CAM
    return spec.compile(), camera


def depth_size(mask, width):
    """(w, h) whose aspect gives the render mask.hfov_deg horizontally while
    the camera's fovy carries mask.vfov_deg."""
    ratio = math.tan(math.radians(float(mask.hfov_deg)) / 2) / math.tan(
        math.radians(float(mask.vfov_deg)) / 2
    )
    return width, max(1, round(width / ratio))


def project(points, cam_pos, cam_mat, fovy, size):
    """(u, v, depth) of world `points` in a pinhole camera image."""
    w, h = size
    local = (np.asarray(points) - cam_pos) @ cam_mat
    depth = -local[:, 2]
    f = (h / 2) / math.tan(math.radians(fovy) / 2)
    ahead = np.maximum(depth, 1e-6)
    return w / 2 + f * local[:, 0] / ahead, h / 2 - f * local[:, 1] / ahead, depth


def display_grid(scan):
    """The flat body-frame scan as an image grid: far row first, +y on left."""
    # body_grid indexes ix*ny + iy with x forward and y left.
    return np.asarray(scan).reshape(height_scan.NX, height_scan.NY)[::-1, ::-1]


class SceneView:
    """Chase-camera renderer with the overlays a tool switches on.

    `torque` draws the actuator bar strip against the env's effective limit;
    `onboard` renders the depth camera at the height-scan mask's mount, pastes
    it in as an inset and projects the scan points onto it, colored by
    `height_scan.visible_mask`. With neither, `frame` returns the plain render.
    """

    def __init__(
        self, env, *, size=DEFAULT_SIZE, camera="track", torque=False, onboard=False,
    ):
        import mujoco

        from wojtek_rl.battery import torque_cap_of

        hs = env._config.get("height_scan")
        if onboard and hs is None:
            raise ValueError(
                "no height_scan config, so there is no mask geometry to place "
                "the depth camera at"
            )
        self._env = env
        self.size = tuple(size)
        self.camera = camera
        self.model, depth_cam = scene_model(
            env, self.size, hs.mask if onboard else None
        )
        self.data = mujoco.MjData(self.model)
        self._view = mujoco.Renderer(self.model, height=self.size[1], width=self.size[0])
        self._cap = torque_cap_of(env) if torque else None
        self._clip = float(hs.clip) if hs is not None else 0.0
        self._depth_view = None
        if onboard:
            self._mask = hs.mask
            self._mount = np.asarray(hs.mask.mount, float)
            self._grid = height_scan.body_grid(hs.x_range, hs.y_range, hs.nx, hs.ny)
            self._depth_cam = depth_cam
            self._cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, depth_cam
            )
            self._fovy = float(self.model.cam_fovy[self._cam_id])
            self._depth_size = depth_size(hs.mask, overlays.inset_width(self.size))
            self._depth_view = mujoco.Renderer(
                self.model, height=self._depth_size[1], width=self._depth_size[0]
            )
            self._depth_view.enable_depth_rendering()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self._view.close()
        if self._depth_view is not None:
            self._depth_view.close()

    def frame(self, qpos, *, torque=None, scan_hold=None, header=None):
        """The scene at `qpos`, with the overlays this view was built for."""
        import mujoco

        qpos = np.asarray(qpos)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self._view.update_scene(self.data, camera=self.camera)
        frame = self._view.render()
        layers = {}
        if self._depth_view is not None:
            self._depth_view.update_scene(self.data, camera=self._depth_cam)
            layers["depth"] = overlays.depth_rgb(self._depth_view.render())
            layers["scan_uv"], layers["visible"] = self._scan_dots(qpos)
        if scan_hold is not None:
            layers["scan_hold"] = display_grid(scan_hold)
            layers["clip"] = self._clip
        if self._cap is not None and torque is not None:
            layers["torque"], layers["cap"] = np.asarray(torque), self._cap
        if header:
            layers["header"] = header
        return overlays.compose(frame, **layers) if layers else frame

    def _scan_dots(self, qpos):
        """(uv, visible) of the scan grid points in the depth image."""
        env, mask = self._env, self._mask
        xy = height_scan.world_xy(
            self._grid, qpos[0:2], height_scan.yaw_from_quat(qpos[3:7], xp=np), xp=np
        )
        z = (
            np.asarray(env._terrain.height(xy)) if env._terrain_enabled
            else np.zeros(len(xy))
        )
        points = np.concatenate([xy, z[:, None]], axis=-1)
        visible = np.asarray(
            height_scan.visible_mask(
                points, qpos[0:3], qpos[3:7], self._mount, mask.pitch_deg,
                mask.hfov_deg, mask.vfov_deg, mask.min_depth, mask.max_depth,
                xp=np,
            )
        )
        if mask.occlusion and env._terrain_enabled:
            cam = height_scan.camera_pos(qpos[0:3], qpos[3:7], self._mount, xp=np)
            occ = np.asarray(
                height_scan.occluded_mask(
                    points, cam, lambda p: np.asarray(env._terrain.height(p)),
                    mask.occlusion_samples, mask.occlusion_margin, xp=np,
                )
            )
            visible = visible & ~occ
        u, v, ahead = project(
            points, self.data.cam_xpos[self._cam_id],
            self.data.cam_xmat[self._cam_id].reshape(3, 3), self._fovy,
            self._depth_size,
        )
        keep = ahead > 0.0
        return np.stack([u[keep], v[keep]], axis=-1), visible[keep]
