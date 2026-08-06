"""Offscreen MuJoCo renderer emulating the RealSense D435 for the sim node.

Two responsibilities, both driven by wojtek_pc.camera_spec:

* ``load_model_with_camera`` -- inject the ``d435_depth`` camera into the sim
  model at load time via ``mujoco.MjSpec``. The sim still runs the model copy
  shipped in wojtek_pc/config (see machinekind/wojtek#93 for retiring it);
  injecting here keeps this feature physics-neutral: a camera adds no mass,
  no geoms, no dynamics.
* ``SimDepthCamera`` -- owns the offscreen renderers and a PRIVATE MjData.
  GL contexts are thread-affine, so the contract is: construct the object
  wherever, but call start()/render_*()/close() from ONE thread (the sim
  node's render thread). The physics MjData is never touched; callers pass a
  (qpos, qvel) snapshot and the pose is re-derived here with mj_forward.

mujoco is imported lazily so this module can be imported (e.g. by tests that
only want depth_to_mm re-exports) before the sim node has settled MUJOCO_GL.
"""

import numpy as np

from wojtek_pc import camera_spec


def _free_root_body(spec, mujoco):
    """The body owning the free joint -- where a head camera belongs."""
    for body in spec.bodies:
        for jnt in body.joints:
            if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
                return body
    raise ValueError("model has no free-jointed body to mount the camera on")


def inject_camera(spec):
    """Add the d435_depth camera (camera_spec pose/fov) to a loaded MjSpec."""
    import mujoco

    body = _free_root_body(spec, mujoco)
    cam = body.add_camera()
    cam.name = camera_spec.CAMERA_NAME
    cam.pos = list(camera_spec.MOUNT_XYZ)
    cam.quat = camera_spec.xyaxes_to_quat()
    cam.fovy = camera_spec.FOVY_DEG
    return cam


def load_model_with_camera(xml_path):
    """Compile xml_path with the d435_depth camera injected on the root body.

    Raises on any failure (missing MjSpec, no free body, compile error);
    the caller decides whether that is fatal or degrades to camera-off.
    """
    import mujoco

    spec = mujoco.MjSpec.from_file(xml_path)
    inject_camera(spec)
    return spec.compile()


class SimDepthCamera:
    """D435-shaped depth + colour renders from a (qpos, qvel) snapshot."""

    def __init__(self, model,
                 depth_size=(camera_spec.DEPTH_WIDTH, camera_spec.DEPTH_HEIGHT),
                 color_size=(camera_spec.COLOR_WIDTH, camera_spec.COLOR_HEIGHT),
                 min_m=camera_spec.DEPTH_MIN_M, max_m=camera_spec.DEPTH_MAX_M):
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.depth_size = depth_size
        self.color_size = color_size
        self.min_m, self.max_m = min_m, max_m
        self.cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_spec.CAMERA_NAME
        )
        if self.cam_id < 0:
            raise ValueError(
                f"model has no {camera_spec.CAMERA_NAME!r} camera -- "
                "was it loaded through load_model_with_camera()?"
            )
        # The offscreen framebuffer must cover the largest render BEFORE any
        # Renderer exists; the XML default (640x480) is not guaranteed.
        model.vis.global_.offwidth = max(
            model.vis.global_.offwidth, depth_size[0], color_size[0]
        )
        model.vis.global_.offheight = max(
            model.vis.global_.offheight, depth_size[1], color_size[1]
        )
        self._data = None
        self._depth_r = None
        self._color_r = None

    # -- lifecycle: all three below must run on the same (render) thread ----
    def start(self):
        """Allocate the GL contexts and the private MjData."""
        mujoco = self._mujoco
        self._data = mujoco.MjData(self.model)
        self._depth_r = mujoco.Renderer(
            self.model, height=self.depth_size[1], width=self.depth_size[0]
        )
        self._color_r = mujoco.Renderer(
            self.model, height=self.color_size[1], width=self.color_size[0]
        )

    def close(self):
        for r in (self._depth_r, self._color_r):
            if r is not None:
                r.close()
        self._depth_r = self._color_r = self._data = None

    # -- rendering -----------------------------------------------------------
    def _sync(self, qpos, qvel):
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        self._mujoco.mj_forward(self.model, self._data)

    def render_depth(self, qpos, qvel):
        """uint16 (h, w) depth in mm; 0 = no return, D435-style."""
        self._sync(qpos, qvel)
        r = self._depth_r
        r.enable_depth_rendering()
        try:
            r.update_scene(self._data, camera=camera_spec.CAMERA_NAME)
            depth_m = r.render()
        finally:
            r.disable_depth_rendering()
        return camera_spec.depth_to_mm(depth_m, self.min_m, self.max_m)

    def render_color(self, qpos, qvel):
        """uint8 (h, w, 3) rgb8 image."""
        self._sync(qpos, qvel)
        self._color_r.update_scene(self._data, camera=camera_spec.CAMERA_NAME)
        return self._color_r.render()
