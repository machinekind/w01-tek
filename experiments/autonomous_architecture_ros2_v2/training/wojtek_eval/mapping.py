"""Online occupancy mapping from the ego depth camera + frontier exploration.

What the real dog will do with its RealSense, done honestly in sim: every VLM
step the mapper unprojects the ego depth image, classifies points into
obstacle band vs floor, and fuses them into a persistent known/free/occupied
grid. Nothing is read from the oracle grid -- the agent map only knows what
the camera has seen.

The map serves two consumers:
  - map_image(): annotated top-down view (robot, heading, trail, frontiers)
    composited into the VLM's input frame as a HUD minimap;
  - FrontierPlanner: turns "explore" into a concrete step toward the nearest
    boundary between seen-free and unknown space.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from wojtek_eval.gridmap import OBSTACLE_MAX_Z, OBSTACLE_MIN_Z

UNKNOWN, FREE, OCC = 0, 1, 2
MAX_DEPTH_M = 6.0  # RealSense-ish usable range; beyond it the scan is noise
TRAIL_MAX = 4000


@dataclass
class OnlineMap:
    """Agent-built occupancy grid, same frame/indexing as gridmap.GridMap."""

    res: float
    origin: tuple[float, float]
    shape: tuple[int, int]
    state: np.ndarray = field(init=False)  # (H, W) uint8 in {UNKNOWN, FREE, OCC}
    trail: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        self.state = np.zeros(self.shape, np.uint8)

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(math.floor((y - self.origin[1]) / self.res)),
            int(math.floor((x - self.origin[0]) / self.res)),
        )

    def cell_to_world(self, i: int, j: int) -> tuple[float, float]:
        return (self.origin[0] + (j + 0.5) * self.res, self.origin[1] + (i + 0.5) * self.res)

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.shape[0] and 0 <= j < self.shape[1]

    # -- fusion ---------------------------------------------------------------

    def integrate_points(self, points_w: np.ndarray, cam_xy: np.ndarray) -> None:
        """Fuse one frame of world-frame points from the depth camera.

        Obstacle-band points mark OCC. A ray that reached the FLOOR passed
        through unobstructed space, so free cells are carved along its whole
        horizontal footprint (fixed 48 samples per ray, vectorized) -- point
        marking alone leaves the map a speckle of frontier noise. OCC is never
        downgraded: thin obstacles beat floor seen through gaps.
        """
        if not len(points_w):
            return
        z = points_w[:, 2]
        H, W = self.shape

        def mark_free(pts_xy: np.ndarray) -> None:
            jj = np.floor((pts_xy[:, 0] - self.origin[0]) / self.res).astype(int)
            ii = np.floor((pts_xy[:, 1] - self.origin[1]) / self.res).astype(int)
            ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
            ii, jj = ii[ok], jj[ok]
            keep = self.state[ii, jj] != OCC
            self.state[ii[keep], jj[keep]] = FREE

        floor_xy = points_w[(z < OBSTACLE_MIN_Z) & (z > -0.3)][:, :2]
        if len(floor_xy):
            t = np.linspace(0.05, 1.0, 48)[None, :, None]  # (1, K, 1)
            rays = cam_xy[None, None, :] + t * (floor_xy[:, None, :] - cam_xy[None, None, :])
            mark_free(rays.reshape(-1, 2))

        obst = points_w[(z >= OBSTACLE_MIN_Z) & (z <= OBSTACLE_MAX_Z)]
        if len(obst):
            jj = np.floor((obst[:, 0] - self.origin[0]) / self.res).astype(int)
            ii = np.floor((obst[:, 1] - self.origin[1]) / self.res).astype(int)
            ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
            self.state[ii[ok], jj[ok]] = OCC

    def mark_pose(self, x: float, y: float) -> None:
        """The robot stands here, so here is free -- and remember the trail."""
        i, j = self.world_to_cell(x, y)
        if self.in_bounds(i, j):
            self.state[i, j] = FREE
        if len(self.trail) < TRAIL_MAX:
            self.trail.append((x, y))

    # -- frontiers --------------------------------------------------------------

    def frontier_cells(self) -> np.ndarray:
        """(N, 2) array of FREE cells bordering UNKNOWN (4-neighborhood)."""
        free = self.state == FREE
        unknown = self.state == UNKNOWN
        border = np.zeros_like(free)
        border[1:, :] |= free[1:, :] & unknown[:-1, :]
        border[:-1, :] |= free[:-1, :] & unknown[1:, :]
        border[:, 1:] |= free[:, 1:] & unknown[:, :-1]
        border[:, :-1] |= free[:, :-1] & unknown[:, 1:]
        return np.argwhere(border)

    # -- rendering ---------------------------------------------------------------

    def map_image(
        self, pose: tuple[float, float, float], px: int = 220
    ) -> np.ndarray:
        """Top-down HUD map (uint8 RGB, north up): dark=unseen, light=seen
        floor, red=obstacle, cyan=frontier, green trail, yellow robot arrow."""
        H, W = self.shape
        img = np.full((H, W, 3), 40, np.uint8)
        img[self.state == FREE] = (210, 210, 210)
        img[self.state == OCC] = (185, 60, 50)
        fc = self.frontier_cells()
        if len(fc):
            img[fc[:, 0], fc[:, 1]] = (60, 190, 200)
        for tx, ty in self.trail[-600:]:
            i, j = self.world_to_cell(tx, ty)
            if self.in_bounds(i, j):
                img[i, j] = (70, 160, 70)
        x, y, yaw = pose
        i, j = self.world_to_cell(x, y)
        for t in np.linspace(0, 6, 13):  # heading arrow, ~30 cm
            ai = i + int(round(t * math.sin(yaw)))
            aj = j + int(round(t * math.cos(yaw)))
            if self.in_bounds(ai, aj):
                img[ai, aj] = (250, 220, 60)
        if self.in_bounds(i, j):
            img[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2] = (250, 220, 60)

        from PIL import Image

        img = img[::-1]  # +y up
        scale = px / max(img.shape[:2])
        out = Image.fromarray(img).resize(
            (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))),
            Image.NEAREST,
        )
        return np.asarray(out)


def compose_hud(frame_rgb: np.ndarray, omap: "OnlineMap",
                pose: tuple[float, float, float], px: int = 210):
    """Ego frame + minimap inset, the composite the VLM (and the demo UI)
    sees. Returns a PIL Image."""
    from PIL import Image, ImageOps

    frame = Image.fromarray(frame_rgb)
    hud = Image.fromarray(omap.map_image(pose, px=px))
    hud = ImageOps.expand(hud, border=2, fill=(250, 220, 60))
    frame.paste(hud, (frame.width - hud.width - 8, frame.height - hud.height - 8))
    return frame


def unproject_depth(
    depth: np.ndarray, fovy_deg: float, cam_pos: np.ndarray, cam_mat: np.ndarray,
    stride: int = 4, max_depth: float = MAX_DEPTH_M,
) -> np.ndarray:
    """Depth image -> world points. cam_mat is MuJoCo's 3x3 camera rotation
    (columns = camera axes in world; camera looks along -z)."""
    h, w = depth.shape
    f = 0.5 * h / math.tan(math.radians(fovy_deg) / 2)
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[::stride, ::stride].reshape(-1)
    ok = np.isfinite(d) & (d > 0.05) & (d < max_depth)
    u = uu.reshape(-1)[ok].astype(np.float64)
    v = vv.reshape(-1)[ok].astype(np.float64)
    d = d[ok]
    x = (u - w / 2) / f * d
    y = -(v - h / 2) / f * d
    pts_cam = np.stack([x, y, -d], axis=1)
    return pts_cam @ cam_mat.T + cam_pos


class FrontierPlanner:
    """Turn 'explore' into one concrete (turn, forward) step.

    BFS from the robot over traversable cells (FREE or UNKNOWN -- optimistic
    planning through unseen space) to the nearest frontier; walk ~1.2 m along
    that path. Returns None when no frontier remains (map closed)."""

    STEP_M = 1.2

    def __init__(self, grid_res: float):
        self.res = grid_res

    def next_step(
        self, omap: OnlineMap, pose: tuple[float, float, float]
    ) -> tuple[float, float] | None:
        """-> (turn_deg_left, forward_m) toward the nearest frontier."""
        cells_arr = omap.frontier_cells()
        if not len(cells_arr):
            return None
        # Drop lone frontier speckle: keep cells with another frontier cell
        # within 2 cells (cheap cluster-size>=2 proxy).
        pts = cells_arr[:, None, :] - cells_arr[None, :, :]
        near = (np.abs(pts).max(axis=2) <= 2).sum(axis=1) >= 3  # self + 2 others
        keep = cells_arr[near] if near.any() else cells_arr
        frontier = {(int(i), int(j)) for i, j in keep}
        start = omap.world_to_cell(pose[0], pose[1])
        if not omap.in_bounds(*start):
            return None
        traversable = omap.state != OCC
        came: dict[tuple[int, int], tuple[int, int]] = {}
        seen = {start}
        q = deque([start])
        hit = None
        while q:
            cur = q.popleft()
            if cur in frontier:
                hit = cur
                break
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cur[0] + di, cur[1] + dj)
                if (
                    nxt not in seen
                    and omap.in_bounds(*nxt)
                    and traversable[nxt[0], nxt[1]]
                ):
                    seen.add(nxt)
                    came[nxt] = cur
                    q.append(nxt)
        if hit is None:
            return None
        cells = [hit]
        while cells[-1] in came:
            cells.append(came[cells[-1]])
        cells.reverse()  # start -> frontier
        k = min(len(cells) - 1, max(1, int(self.STEP_M / self.res)))
        tx, ty = omap.cell_to_world(*cells[k])
        x, y, yaw = pose
        bearing = math.atan2(ty - y, tx - x)
        turn = math.degrees((bearing - yaw + math.pi) % (2 * math.pi) - math.pi)
        # Floor at 0.3 m: a nearby frontier is usually just map fuzz behind
        # the robot; stepping a bit past it keeps exploration moving.
        dist = max(0.3, min(math.hypot(tx - x, ty - y), self.STEP_M))
        return (turn, dist)
