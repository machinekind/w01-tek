"""Top-down view of what the planner is thinking.

The point of the whole exercise is to be able to watch the robot refuse to
walk into things, so the local map, the A* guidance, the optimised spline and
the twin-cylinder footprint all get drawn into one small image that the demo
UI, the A/B runner and the episode media all reuse.

Colour key: dark = never observed, grey = observed free, red = occupied,
dusty red = inflated (where a cylinder centre may not go), cyan = A* guide,
green = optimised trajectory, yellow = robot with its two cylinders.
"""

from __future__ import annotations

import math

import numpy as np

BG_UNKNOWN = (38, 38, 44)
BG_FREE = (208, 208, 208)
INFLATED = (120, 66, 62)
OCCUPIED = (198, 60, 48)
GUIDE = (60, 190, 205)
TRAJ = (90, 220, 110)
ROBOT = (250, 210, 60)
GOAL = (250, 120, 200)


def plan_image(omap, pose, planner=None, px: int = 260, footprint=None) -> np.ndarray:
    """(px, px) RGB view of the sliding window, north up."""
    inflated, known = omap.window_arrays()
    occ = omap.occupied_window()
    img = np.full((omap.n, omap.n, 3), BG_UNKNOWN, np.uint8)
    img[known] = BG_FREE
    img[inflated] = INFLATED
    img[occ] = OCCUPIED

    def put(x, y, color, r=0):
        i = int(math.floor(y / omap.cfg.res)) - omap.i0
        j = int(math.floor(x / omap.cfg.res)) - omap.j0
        if not (0 <= i < omap.n and 0 <= j < omap.n):
            return
        img[max(0, i - r) : i + r + 1, max(0, j - r) : j + r + 1] = color

    if planner is not None:
        for gx, gy in planner.debug.guide:
            put(gx, gy, GUIDE)
        for tx, ty in planner.debug.traj:
            put(tx, ty, TRAJ)
        if planner.goal is not None:
            put(planner.goal[0], planner.goal[1], GOAL, r=1)
        if planner.debug.repaired_goal is not None:
            put(*planner.debug.repaired_goal, GOAL, r=0)

    x, y, yaw = pose
    if footprint is not None:
        front, rear = footprint.centers(x, y, yaw)
        rad = int(round(footprint.radius / omap.cfg.res))
        for cx, cy in (front[0], rear[0]):
            _ring(img, omap, cx, cy, rad, ROBOT)
    for t in np.linspace(0, 0.35, 8):  # heading whisker
        put(x + t * math.cos(yaw), y + t * math.sin(yaw), ROBOT)
    put(x, y, ROBOT, r=1)

    img = img[::-1]  # +y up
    return _resize(img, px)


def _ring(img, omap, cx, cy, rad, color):
    ci = int(math.floor(cy / omap.cfg.res)) - omap.i0
    cj = int(math.floor(cx / omap.cfg.res)) - omap.j0
    ang = np.linspace(0, 2 * math.pi, max(12, 4 * rad), endpoint=False)
    ii = ci + np.round(rad * np.sin(ang)).astype(int)
    jj = cj + np.round(rad * np.cos(ang)).astype(int)
    ok = (ii >= 0) & (ii < omap.n) & (jj >= 0) & (jj < omap.n)
    img[ii[ok], jj[ok]] = color


def _resize(img: np.ndarray, px: int) -> np.ndarray:
    from PIL import Image

    scale = px / max(img.shape[:2])
    out = Image.fromarray(img).resize(
        (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))),
        Image.NEAREST,
    )
    return np.asarray(out)
