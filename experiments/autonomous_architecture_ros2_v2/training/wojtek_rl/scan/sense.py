"""Simulated RealSense depth channel feeding the SCAN local map.

One small offscreen renderer on the onboard ego camera, sampled at ~10 Hz
while the robot walks (not only at VLM decision points -- a planner that only
sees when the VLM looks is blind for the two seconds it is executing a move).

Two things this does that a plain depth render does not:

* **Self-masking.** The ego camera sits ~15 cm up on the body and the robot's
  own legs swing through the near field. Unmasked, they fuse into the map as
  a phantom obstacle trail that follows the robot around and walls it in. A
  segmentation pass drops every pixel belonging to the robot's own geoms --
  the sim equivalent of the extrinsic mask a real RealSense install gets from
  knowing where the body is.
* **Standing-proof free space.** The camera cannot see the floor closer than
  ~0.4 m, so the ring around the robot would stay unknown forever. The robot
  is standing there, which is proof it is free: stamp it.

The render is deliberately small (160x120 by default). Depth quality at that
size is ample for a 5 cm grid and keeps a sensing tick at ~1 ms.
"""

from __future__ import annotations

import numpy as np

from wojtek_eval.mapping import unproject_depth
from wojtek_rl.scan.localmap import SlidingOccupancyMap

SENSE_W, SENSE_H = 160, 120
FOOTPRINT_FREE_R = 0.35  # radius of the standing-proof free disc


class EgoDepthSensor:
    """Ego-camera depth -> world points -> SlidingOccupancyMap."""

    def __init__(self, model, data, camera: str = "ego",
                 width: int = SENSE_W, height: int = SENSE_H, stride: int = 2):
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = data
        self.stride = stride
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if self.cam_id < 0:
            raise ValueError(f"scene has no {camera!r} camera")
        self.camera = camera
        self.fovy = float(model.cam_fovy[self.cam_id])
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._self_geoms = _robot_geom_ids(mujoco, model)

    def close(self) -> None:
        self._renderer.close()

    def depth(self) -> np.ndarray:
        """Self-masked depth image (masked pixels set to 0 = dropped)."""
        r = self._renderer
        r.enable_depth_rendering()
        r.update_scene(self.data, camera=self.camera)
        depth = r.render().copy()
        r.disable_depth_rendering()
        if self._self_geoms.size:
            r.enable_segmentation_rendering()
            r.update_scene(self.data, camera=self.camera)
            seg = r.render()[:, :, 0]
            r.disable_segmentation_rendering()
            depth[np.isin(seg, self._self_geoms)] = 0.0
        return depth

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        """(world points, camera xy) for the current sim state."""
        depth = self.depth()
        cam_pos = self.data.cam_xpos[self.cam_id].copy()
        cam_mat = self.data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        pts = unproject_depth(depth, self.fovy, cam_pos, cam_mat, stride=self.stride)
        return pts, cam_pos[:2]

    def update(self, omap: SlidingOccupancyMap, robot_xy) -> None:
        """One sensing tick: recentre the window, fuse, stamp the footprint."""
        omap.recenter(float(robot_xy[0]), float(robot_xy[1]))
        pts, cam_xy = self.points()
        omap.integrate_points(pts, cam_xy)
        omap.mark_free_disc(float(robot_xy[0]), float(robot_xy[1]), FOOTPRINT_FREE_R)


def _robot_geom_ids(mujoco, model) -> np.ndarray:
    """Geom ids belonging to the floating-base robot (everything under the
    body that owns the free joint) -- the pixels to mask out."""
    free = [
        j for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if not free:
        return np.empty(0, np.int32)
    root = int(model.body_rootid[model.jnt_bodyid[free[0]]])
    bodies = np.flatnonzero(np.asarray(model.body_rootid) == root)
    return np.flatnonzero(np.isin(np.asarray(model.geom_bodyid), bodies)).astype(np.int32)
