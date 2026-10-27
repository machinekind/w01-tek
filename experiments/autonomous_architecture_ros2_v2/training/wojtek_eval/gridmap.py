"""2D occupancy grids for navigation eval: oracle (from scene meshes) and
online (fused from ego depth by wojtek_eval.mapping).

The oracle grid is ground truth: episode start sampling, reachability,
geodesic shortest paths (the SPL denominator), and collision for the
kinematic sim. Obstacles are mesh faces in the robot's height band
[OBSTACLE_MIN_Z, OBSTACLE_MAX_Z]; free space is scan floor not covered by an
inflated obstacle. Everything is pure numpy on a ~5 cm grid, so Dijkstra over
the whole map is milliseconds.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RESOLUTION = 0.05  # m per cell
ROBOT_RADIUS = 0.16  # inflation radius: half-width 0.09 + foot overhang margin
OBSTACLE_MIN_Z = 0.06  # below: scan floor waviness, not an obstacle
OBSTACLE_MAX_Z = 0.35  # above: the ~0.2 m walking robot passes underneath
FLOOR_MAX_Z = 0.05  # upward faces below this are walkable floor

# 8-connected neighborhood with step costs (unit = one cell).
_NEIGHBORS = [
    (-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)),
    (0, -1, 1.0), (0, 1, 1.0),
    (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2)),
]


@dataclass
class GridMap:
    """Occupancy grid in world coordinates.

    occ:  True where an (inflated) obstacle lives.
    free: True where the robot may stand (floor present, not occ).
    Cells outside both are unknown/void (holes in the scan, outside walls).
    """

    res: float
    origin: tuple[float, float]  # world (x, y) of cell [0, 0]'s corner
    occ: np.ndarray  # (H, W) bool
    free: np.ndarray  # (H, W) bool

    # -- coordinates ---------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return self.occ.shape

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(math.floor((y - self.origin[1]) / self.res)),
            int(math.floor((x - self.origin[0]) / self.res)),
        )

    def cell_to_world(self, i: int, j: int) -> tuple[float, float]:
        return (
            self.origin[0] + (j + 0.5) * self.res,
            self.origin[1] + (i + 0.5) * self.res,
        )

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.occ.shape[0] and 0 <= j < self.occ.shape[1]

    def is_free(self, x: float, y: float) -> bool:
        i, j = self.world_to_cell(x, y)
        return self.in_bounds(i, j) and bool(self.free[i, j])

    # -- algorithms ----------------------------------------------------------

    def geodesic(self, sources_xy: list[tuple[float, float]]) -> np.ndarray:
        """Distance-to-nearest-source (meters) over free cells; inf elsewhere.

        Multi-source Dijkstra. Sources landing on non-free cells are snapped
        to the nearest free cell within 0.5 m (goal objects sit ON obstacles).
        """
        dist = np.full(self.occ.shape, np.inf)
        heap: list[tuple[float, int, int]] = []
        for x, y in sources_xy:
            ij = self._snap_free(x, y)
            if ij is None:
                continue
            i, j = ij
            if dist[i, j] > 0.0:
                dist[i, j] = 0.0
                heapq.heappush(heap, (0.0, i, j))
        step = self.res
        while heap:
            d, i, j = heapq.heappop(heap)
            if d > dist[i, j]:
                continue
            for di, dj, w in _NEIGHBORS:
                ni, nj = i + di, j + dj
                if not self.in_bounds(ni, nj) or not self.free[ni, nj]:
                    continue
                nd = d + w * step
                if nd < dist[ni, nj]:
                    dist[ni, nj] = nd
                    heapq.heappush(heap, (nd, ni, nj))
        return dist

    def _snap_free(self, x: float, y: float, max_r: float = 0.5) -> tuple[int, int] | None:
        i0, j0 = self.world_to_cell(x, y)
        if self.in_bounds(i0, j0) and self.free[i0, j0]:
            return (i0, j0)
        r_cells = int(max_r / self.res)
        best, best_d = None, np.inf
        for i in range(max(0, i0 - r_cells), min(self.occ.shape[0], i0 + r_cells + 1)):
            for j in range(max(0, j0 - r_cells), min(self.occ.shape[1], j0 + r_cells + 1)):
                if self.free[i, j]:
                    d = (i - i0) ** 2 + (j - j0) ** 2
                    if d < best_d:
                        best, best_d = (i, j), d
        return best

    def distance_at(self, dist: np.ndarray, x: float, y: float) -> float:
        """Read a geodesic field at a world point (snapped to free)."""
        ij = self._snap_free(x, y)
        if ij is None:
            return float("inf")
        return float(dist[ij])

    def shortest_path(
        self, start_xy: tuple[float, float], goal_xy: tuple[float, float]
    ) -> list[tuple[float, float]] | None:
        """A* path (world waypoints, cell-resolution) or None if unreachable."""
        start = self._snap_free(*start_xy)
        goal = self._snap_free(*goal_xy)
        if start is None or goal is None:
            return None
        h = lambda i, j: math.hypot(i - goal[0], j - goal[1])  # noqa: E731
        g = {start: 0.0}
        came: dict[tuple[int, int], tuple[int, int]] = {}
        heap = [(h(*start), start)]
        while heap:
            _, cur = heapq.heappop(heap)
            if cur == goal:
                cells = [cur]
                while cur in came:
                    cur = came[cur]
                    cells.append(cur)
                return [self.cell_to_world(i, j) for i, j in reversed(cells)]
            for di, dj, w in _NEIGHBORS:
                ni, nj = cur[0] + di, cur[1] + dj
                if not self.in_bounds(ni, nj) or not self.free[ni, nj]:
                    continue
                ng = g[cur] + w
                if ng < g.get((ni, nj), np.inf):
                    g[(ni, nj)] = ng
                    came[(ni, nj)] = cur
                    heapq.heappush(heap, (ng + h(ni, nj), (ni, nj)))
        return None

    def sample_free(self, rng: np.random.Generator, clearance: float = 0.10) -> tuple[float, float]:
        """Random free world point with extra clearance from obstacles."""
        extra = int(clearance / self.res)
        free = self.free
        if extra > 0:
            free = free & ~_dilate(self.occ, extra)
        idx = np.flatnonzero(free)
        if not len(idx):
            raise ValueError("no free cells at requested clearance")
        k = rng.choice(idx)
        i, j = divmod(int(k), free.shape[1])
        return self.cell_to_world(i, j)

    # -- io --------------------------------------------------------------------

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, res=self.res, origin=np.array(self.origin), occ=self.occ, free=self.free
        )

    @classmethod
    def load(cls, path: Path) -> "GridMap":
        z = np.load(path)
        return cls(
            res=float(z["res"]),
            origin=(float(z["origin"][0]), float(z["origin"][1])),
            occ=z["occ"].astype(bool),
            free=z["free"].astype(bool),
        )


def _dilate(mask: np.ndarray, cells: int) -> np.ndarray:
    """Binary dilation by a disk of radius `cells` (scipy-backed)."""
    from scipy import ndimage

    if cells <= 0:
        return mask
    yy, xx = np.mgrid[-cells : cells + 1, -cells : cells + 1]
    disk = (xx * xx + yy * yy) <= cells * cells
    return ndimage.binary_dilation(mask, structure=disk)


# -- oracle grid from scene meshes ---------------------------------------------


def _sample_faces(vertices: np.ndarray, faces: np.ndarray, res: float) -> np.ndarray:
    """Points densely covering the faces (~res/2 spacing), vectorized."""
    tri = vertices[faces]  # (F, 3, 3)
    # Enough samples along each edge that no cell inside a triangle is missed.
    # The cap only guards memory; a flattened floor can be two room-sized
    # triangles, which need hundreds of subdivisions to cover every cell.
    edge = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max(axis=1)
    n = np.clip(np.ceil(edge / (res * 0.5)).astype(int), 1, 512)
    points = []
    for level in np.unique(n):
        sel = tri[n == level]
        # Barycentric lattice with `level`+1 points per edge.
        bary = []
        for a in range(level + 1):
            for b in range(level + 1 - a):
                c = level - a - b
                bary.append((a / level if level else 1.0, b / level if level else 0.0,
                             c / level if level else 0.0))
        bary = np.array(bary)  # (B, 3)
        pts = np.einsum("bk,fkd->fbd", bary, sel).reshape(-1, 3)
        points.append(pts)
    return np.concatenate(points, axis=0)


def build_oracle_grid(scene_name: str, res: float = RESOLUTION,
                      robot_radius: float = ROBOT_RADIUS) -> GridMap:
    """Rasterize a converted scene's visual meshes into a GridMap."""
    import trimesh

    from wojtek_rl import paths

    scene_dir = paths.scene_dir(scene_name)
    manifest = json.loads(paths.scene_manifest(scene_name).read_text())
    visual_dir = scene_dir / manifest["visual_dir"]

    (x0, y0, _), (x1, y1, _) = manifest["aabb"]
    pad = 0.3
    origin = (x0 - pad, y0 - pad)
    W = int(math.ceil((x1 - x0 + 2 * pad) / res))
    H = int(math.ceil((y1 - y0 + 2 * pad) / res))
    occ = np.zeros((H, W), bool)
    floor = np.zeros((H, W), bool)

    def mark(points: np.ndarray, grid: np.ndarray) -> None:
        jj = np.floor((points[:, 0] - origin[0]) / res).astype(int)
        ii = np.floor((points[:, 1] - origin[1]) / res).astype(int)
        ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
        grid[ii[ok], jj[ok]] = True

    for entry in manifest["visual"]:
        mesh = trimesh.load(str(visual_dir / entry["mesh"]), process=False)
        meshes = mesh.geometry.values() if hasattr(mesh, "geometry") else [mesh]
        for m in meshes:
            v, f = np.asarray(m.vertices), np.asarray(m.faces)
            z = v[f][:, :, 2]
            nrm = m.face_normals
            band = (z.max(axis=1) > OBSTACLE_MIN_Z) & (z.min(axis=1) < OBSTACLE_MAX_Z)
            if band.any():
                mark(_sample_faces(v, f[band], res), occ)
            # |nz|: scan/export winding is unreliable, a floor face may point
            # "down". Lower bound keeps below-floor scan artifacts out.
            zc = v[f][:, :, 2].mean(axis=1)
            fl = (np.abs(nrm[:, 2]) > 0.7) & (zc < FLOOR_MAX_Z) & (zc > -0.10)
            if fl.any():
                mark(_sample_faces(v, f[fl], res), floor)

    inflated = _dilate(occ, int(round(robot_radius / res)))
    free = floor & ~inflated
    # Keep only the biggest connected free region: disconnected slivers
    # (scan noise outside windows, under-furniture pockets) are unreachable.
    from scipy import ndimage

    labels, n = ndimage.label(free)
    if n > 1:
        sizes = ndimage.sum(free, labels, index=range(1, n + 1))
        free = labels == (1 + int(np.argmax(sizes)))
    return GridMap(res=res, origin=origin, occ=inflated, free=free)


def main(argv=None) -> None:
    import argparse

    from wojtek_rl import paths

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="room")
    p.add_argument("--res", type=float, default=RESOLUTION)
    args = p.parse_args(argv)
    grid = build_oracle_grid(args.name, res=args.res)
    out = paths.scene_dir(args.name) / "occupancy.npz"
    grid.save(out)
    print(
        f"wrote {out}: {grid.shape[0]}x{grid.shape[1]} cells @ {grid.res} m, "
        f"free {int(grid.free.sum())} occ {int(grid.occ.sum())}"
    )


if __name__ == "__main__":
    main()
