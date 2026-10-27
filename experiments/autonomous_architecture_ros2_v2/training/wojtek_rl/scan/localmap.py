"""Robot-centric sliding log-odds occupancy map (SCAN-Planner, Sec. V-A).

The agent's only obstacle memory for local planning. Depth points are fused
probabilistically -- hits raise a cell's log-odds, the rays that reached them
lower every cell they passed through -- with symmetric clamps so persistent
structure stays occupied while a moved chair (or a person) is forgotten in a
second or two. That transient behaviour is exactly why the paper uses
log-odds instead of the sticky uint8 states in wojtek_eval.mapping (which is
a *global* map for the HUD/frontier explorer and has no reason to forget).

The window is a fixed-size grid that slides with the robot, addressed
modulo N: the absolute cell index maps to a ring address, so recentring only
clears the rows/columns that just entered the window -- no copies, bounded
memory, and (unlike a world-sized array) the same cost in a 4 m room and a
40 m corridor.

Height handling is the paper's twin-cylinder vertical clearance collapsed to
the one band that matters for a flat-floor indoor robot: points below
``z_step`` are steppable floor, points above ``z_up`` are overhead clearance
the robot walks under, and only the band between them is an obstacle. Both
bounds are shared with the oracle grid (wojtek_eval.gridmap) so the online
and ground-truth maps agree on what counts as an obstacle.

The inflated layer is the map dilated by the twin-cylinder radius, which is
what turns whole-body collision checking into two point queries
(see footprint.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wojtek_eval.gridmap import OBSTACLE_MAX_Z, OBSTACLE_MIN_Z

MAX_RANGE_M = 6.0  # RealSense D435i usable depth; beyond it the scan is noise


@dataclass(frozen=True)
class MapConfig:
    """Sliding-map geometry and log-odds parameters."""

    res: float = 0.05            # m per cell
    extent_m: float = 8.0        # window side length (robot-centred)
    recenter_m: float = 0.6      # slide once the robot drifts this far off centre
    z_step: float = OBSTACLE_MIN_Z   # below: steppable / floor waviness
    z_up: float = OBSTACLE_MAX_Z     # above: overhead clearance, robot passes under
    inflate_r: float = 0.20      # twin-cylinder radius (footprint.py)
    l_hit: float = 0.85          # log-odds added by an obstacle return
    l_miss: float = -0.45        # log-odds added by a ray passing through
    l_min: float = -2.0          # lower clamp: free cells re-occupy quickly
    l_max: float = 3.5           # upper clamp: no over-confident obstacles
    l_occupied: float = 0.80     # decision threshold (one hit is enough)
    max_range: float = MAX_RANGE_M


class SlidingOccupancyMap:
    """Log-odds occupancy in a circular buffer that slides with the robot."""

    def __init__(self, cfg: MapConfig | None = None):
        self.cfg = cfg or MapConfig()
        self.n = int(round(self.cfg.extent_m / self.cfg.res))
        self.logodds = np.zeros((self.n, self.n), np.float32)
        self.known = np.zeros((self.n, self.n), bool)
        # Absolute cell index of the window's lower-left corner (row=y, col=x).
        self.i0 = -self.n // 2
        self.j0 = -self.n // 2
        self._inflated = np.zeros((self.n, self.n), bool)
        self._inflate_dirty = True
        self._disc = _disc_kernel(self.cfg.inflate_r / self.cfg.res)

    # -- addressing --------------------------------------------------------

    def cell_of(self, x, y):
        """World (x, y) -> absolute (row, col) cell index (vectorized)."""
        j = np.floor(np.asarray(x, np.float64) / self.cfg.res).astype(np.int64)
        i = np.floor(np.asarray(y, np.float64) / self.cfg.res).astype(np.int64)
        return i, j

    def world_of(self, i, j):
        """Absolute cell index -> world (x, y) of the cell centre."""
        return ((np.asarray(j) + 0.5) * self.cfg.res, (np.asarray(i) + 0.5) * self.cfg.res)

    def inside(self, i, j):
        """True where the absolute cell index falls inside the window."""
        i, j = np.asarray(i), np.asarray(j)
        return (i >= self.i0) & (i < self.i0 + self.n) & (j >= self.j0) & (j < self.j0 + self.n)

    def _addr(self, i, j):
        """Ring address of an absolute cell index (only valid when inside)."""
        return np.asarray(i) % self.n, np.asarray(j) % self.n

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of the current window."""
        r = self.cfg.res
        return (self.j0 * r, self.i0 * r, (self.j0 + self.n) * r, (self.i0 + self.n) * r)

    def recenter(self, x: float, y: float) -> None:
        """Slide the window so the robot is (re)centred; clears incoming cells."""
        i, j = self.cell_of(x, y)
        want_i, want_j = int(i) - self.n // 2, int(j) - self.n // 2
        off = self.cfg.recenter_m / self.cfg.res
        if abs(want_i - self.i0) < off and abs(want_j - self.j0) < off:
            return
        for axis, old, new in ((0, self.i0, want_i), (1, self.j0, want_j)):
            shift = new - old
            if shift == 0:
                continue
            if abs(shift) >= self.n:
                self.logodds[:] = 0.0
                self.known[:] = False
                break
            # Absolute lines that leave the window; their ring addresses are
            # reused by the lines entering on the other side.
            gone = np.arange(old, new) if shift > 0 else np.arange(new + self.n, old + self.n)
            addr = gone % self.n
            if axis == 0:
                self.logodds[addr, :] = 0.0
                self.known[addr, :] = False
            else:
                self.logodds[:, addr] = 0.0
                self.known[:, addr] = False
        self.i0, self.j0 = want_i, want_j
        self._inflate_dirty = True

    # -- fusion ------------------------------------------------------------

    def integrate_points(self, points_w: np.ndarray, sensor_xy) -> None:
        """Fuse one frame of world-frame depth points.

        Band points are hits; every ray (band or floor) frees the cells it
        crossed on the way. Cells hit in this frame are never freed by another
        ray of the same frame -- a thin chair leg seen past a floor return
        must survive the floor ray that grazed it.
        """
        pts = np.asarray(points_w, np.float64)
        if pts.size == 0:
            return
        cfg = self.cfg
        sx, sy = float(sensor_xy[0]), float(sensor_xy[1])
        z = pts[:, 2]
        rng = np.hypot(pts[:, 0] - sx, pts[:, 1] - sy)
        usable = (rng > 1e-3) & (rng <= cfg.max_range)
        band = usable & (z >= cfg.z_step) & (z <= cfg.z_up)
        floor = usable & (z < cfg.z_step) & (z > -0.3)

        occ_flat = self._flat(pts[band, 0], pts[band, 1])
        ray_pts = pts[band | floor, :2]
        free_flat = self._free_along_rays(sx, sy, ray_pts)

        occ_flat = np.unique(occ_flat)
        free_flat = np.setdiff1d(np.unique(free_flat), occ_flat, assume_unique=True)
        flat = self.logodds.reshape(-1)
        known = self.known.reshape(-1)
        if free_flat.size:
            flat[free_flat] = np.maximum(flat[free_flat] + cfg.l_miss, cfg.l_min)
            known[free_flat] = True
        if occ_flat.size:
            flat[occ_flat] = np.minimum(flat[occ_flat] + cfg.l_hit, cfg.l_max)
            known[occ_flat] = True
        self._inflate_dirty = True

    def _free_along_rays(self, sx: float, sy: float, ends_xy: np.ndarray) -> np.ndarray:
        """Flat addresses of the cells each ray passes through (endpoint excluded).

        Fixed-count sampling rather than exact Bresenham: with a sample step
        below half a cell the miss set is identical in practice, and it stays
        one vectorized expression over ~10^4 rays.
        """
        if not len(ends_xy):
            return np.empty(0, np.int64)
        d = np.hypot(ends_xy[:, 0] - sx, ends_xy[:, 1] - sy)
        k = int(min(160, max(4, math.ceil(float(d.max()) / (0.4 * self.cfg.res)))))
        # Stop half a cell short of the endpoint so a hit is not freed by its own ray.
        t = np.linspace(0.0, 1.0, k)[None, :]
        t = t * np.maximum(0.0, 1.0 - 0.5 * self.cfg.res / np.maximum(d, 1e-6))[:, None]
        xs = sx + t * (ends_xy[:, 0] - sx)[:, None]
        ys = sy + t * (ends_xy[:, 1] - sy)[:, None]
        return self._flat(xs.reshape(-1), ys.reshape(-1))

    def _flat(self, x, y) -> np.ndarray:
        """World points -> flat ring addresses, dropping anything off-window."""
        i, j = self.cell_of(x, y)
        ok = self.inside(i, j)
        ai, aj = self._addr(i[ok], j[ok])
        return (ai * self.n + aj).astype(np.int64)

    def mark_free_disc(self, x: float, y: float, radius: float) -> None:
        """Stamp a disc as observed-free: the robot is standing there.

        The ego camera cannot see the floor closer than ~0.4 m, so without
        this the ring around the robot stays UNKNOWN forever and the planner
        refuses to start.
        """
        r = int(math.ceil(radius / self.cfg.res))
        i, j = self.cell_of(x, y)
        di, dj = np.mgrid[-r : r + 1, -r : r + 1]
        keep = (di**2 + dj**2) <= r * r
        ii, jj = int(i) + di[keep], int(j) + dj[keep]
        ok = self.inside(ii, jj)
        ai, aj = self._addr(ii[ok], jj[ok])
        self.logodds[ai, aj] = np.minimum(self.logodds[ai, aj], self.cfg.l_min)
        self.known[ai, aj] = True
        self._inflate_dirty = True

    # -- queries -----------------------------------------------------------

    @property
    def occupied(self) -> np.ndarray:
        """(N, N) bool ring image of occupied cells."""
        return self.logodds > self.cfg.l_occupied

    @property
    def inflated(self) -> np.ndarray:
        """Occupied dilated by the twin-cylinder radius (lazily rebuilt)."""
        if self._inflate_dirty:
            self._inflated = _dilate(self.occupied, self._disc)
            self._inflate_dirty = False
        return self._inflated

    def is_inflated_occupied(self, x, y, outside: bool = True):
        """Inflated-occupancy lookup at world points. Cells outside the window
        return ``outside`` (True = the planner treats the unseen exterior as
        blocked, which is what keeps trajectories inside the map)."""
        return self._lookup(x, y, self.inflated, outside)

    def is_unknown(self, x, y):
        """True where the cell has never been observed (or is off-window)."""
        return self._lookup(x, y, ~self.known, True)

    def _lookup(self, x, y, ring: np.ndarray, outside: bool):
        scalar = np.isscalar(x) and np.isscalar(y)
        i, j = self.cell_of(np.atleast_1d(x), np.atleast_1d(y))
        ok = self.inside(i, j)
        out = np.full(i.shape, outside, bool)
        if ok.any():
            ai, aj = self._addr(i[ok], j[ok])
            out[ok] = ring[ai, aj]
        return bool(out[0]) if scalar else out

    # -- views for planning / drawing --------------------------------------

    def window_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(inflated, known) as window-ordered arrays, row 0 = self.i0.

        The ring image is contiguous in address space, not in world space;
        A* wants world-ordered rows, so unroll once per replan.
        """
        shift_i = (-self.i0) % self.n
        shift_j = (-self.j0) % self.n
        roll = (shift_i, shift_j)
        return (
            np.roll(self.inflated, roll, axis=(0, 1)),
            np.roll(self.known, roll, axis=(0, 1)),
        )

    def occupied_window(self) -> np.ndarray:
        """Raw (un-inflated) occupancy in window order."""
        return np.roll(self.occupied, ((-self.i0) % self.n, (-self.j0) % self.n), axis=(0, 1))


def _disc_kernel(radius_cells: float) -> np.ndarray:
    r = int(math.ceil(radius_cells))
    di, dj = np.mgrid[-r : r + 1, -r : r + 1]
    return (di**2 + dj**2) <= radius_cells**2


def _dilate(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Binary dilation, wrapping at the ring boundary (scipy when available).

    Wrap is the correct mode here: the array is a circular buffer, so the
    modular neighbour genuinely is the world neighbour one cell away -- except
    across the window seam, which is the far edge of the map and never carries
    a trajectory (is_inflated_occupied treats off-window as blocked).
    """
    if not mask.any():
        return np.zeros_like(mask)
    r = kernel.shape[0] // 2
    try:
        from scipy.ndimage import binary_dilation

        padded = np.pad(mask, r, mode="wrap")
        return binary_dilation(padded, structure=kernel)[r:-r, r:-r]
    except ImportError:  # pragma: no cover - scipy ships with the sim extras
        out = np.zeros_like(mask)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if kernel[di + r, dj + r]:
                    out |= np.roll(mask, (di, dj), axis=(0, 1))
        return out
