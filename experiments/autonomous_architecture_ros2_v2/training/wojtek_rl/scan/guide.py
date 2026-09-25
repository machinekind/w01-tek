"""Projected A* guidance with boundary fallback (SCAN-Planner, Sec. V-B/V-C).

Two jobs. When the B-spline initialisation clips an obstacle, this search
produces the collision-free path the rebound penalty is built from -- it is
*guidance*, never the executed trajectory. And when the local goal is
unreachable inside the sliding window, the same search escapes through a
virtual free layer outside the boundary and hands back the boundary cell it
left through, converting a dead end into a bounded recovery move instead of a
planning failure.

Two adaptations to the paper, both because our scenes are flat scanned rooms
rather than multi-floor buildings:

* The paper searches an inclined surface whose height is interpolated between
  the segment endpoints (z-gradient suppression). On a flat floor that
  surface is the floor, so the search is planar and the suppression is free.
* Node acceptance still uses the yaw-aware footprint of Eq. (5), with the
  checking yaw set to the expansion direction. Since the 8-connected
  neighbourhood has only eight directions, the eight traversability layers
  are precomputed with two array shifts each rather than a per-node query --
  the same test, a hundred times cheaper.

Unknown space is traversable but priced above known-free. The ego camera sees
a 70-degree wedge; treating everything it has not looked at as a wall would
pin the robot in place, and treating it as free makes it dive into unseen
clutter. Pricing it makes the planner prefer the corridor it has already seen
and still commit to unseen space when that is the only way forward.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from wojtek_rl.scan.footprint import Footprint

# 8-connected neighbourhood: (di, dj, step cost in cells).
_NEIGHBORS = tuple(
    (di, dj, math.hypot(di, dj))
    for di in (-1, 0, 1)
    for dj in (-1, 0, 1)
    if (di, dj) != (0, 0)
)
UNKNOWN_COST = 1.6      # multiplier on a step into never-observed space
CLEARANCE_PREF_M = 0.30  # below this distance to the inflated set, steps cost more
# How much more, at zero clearance. Tuned on the physics tier, where 95 % of
# furniture contacts happen with the base exactly on plan (median 4 mm off)
# inside the inflation halo of an obstacle the map already knows -- the legs
# reach 0.248 m laterally, the cylinders model 0.19 m. Widening the map
# instead costs doorways (apartment SR 0.88 -> 0.75 at r 0.26); buying
# clearance where it is free does not: 1.8 -> 6.0 took the room from 1.07 to
# 0.61 contacts/m and the apartment from 3.34 to 2.51 with SR 0.88 -> 1.00.
CLEARANCE_WEIGHT = 6.0
_BOUNDARY_MARGIN = 5     # cells from the window edge that count as "the boundary"


class GuideSearch:
    """A* over one snapshot of the sliding map.

    Rebuilt per replan: it caches the eight yaw-layers, so reusing it across
    ticks would plan against a stale map.
    """

    def __init__(self, omap, footprint: Footprint | None = None,
                 unknown_cost: float = UNKNOWN_COST):
        self.omap = omap
        self.fp = footprint or Footprint(radius=omap.cfg.inflate_r)
        self.unknown_cost = unknown_cost
        self.res = omap.cfg.res
        self.n = omap.n
        self.i0, self.j0 = omap.i0, omap.j0
        inflated, known = omap.window_arrays()
        self.inflated = inflated
        self.unknown = ~known
        self._layers = {
            (di, dj): self._traversable_layer(math.atan2(di, dj))
            for di, dj, _ in _NEIGHBORS
        }
        self.cell_cost = self._clearance_cost()

    def _clearance_cost(self) -> np.ndarray:
        """Per-cell cost multiplier that prefers room to spare.

        Not in the paper, and needed here: a shortest path is happy to graze
        the inflated boundary, where the inflation radius *is* the whole
        margin. Tracking error then puts a cylinder centre inside an obstacle
        and the next A* has nowhere to start from. Pricing the last 30 cm of
        clearance keeps the guide off the wall while leaving genuinely narrow
        passages usable -- it is a cost, not a constraint.
        """
        cost = np.ones((self.n, self.n), np.float32)
        if not self.inflated.any():
            return cost
        try:
            from scipy.ndimage import distance_transform_edt

            d = distance_transform_edt(~self.inflated) * self.res
        except ImportError:  # pragma: no cover
            return cost
        near = np.clip(1.0 - d / CLEARANCE_PREF_M, 0.0, 1.0)
        return cost + CLEARANCE_WEIGHT * near

    # -- geometry ----------------------------------------------------------

    def cell(self, x: float, y: float) -> tuple[int, int]:
        """World -> window (row, col), unclamped."""
        return (
            int(math.floor(y / self.res)) - self.i0,
            int(math.floor(x / self.res)) - self.j0,
        )

    def world(self, i: int, j: int) -> tuple[float, float]:
        return ((self.j0 + j + 0.5) * self.res, (self.i0 + i + 0.5) * self.res)

    def in_window(self, i: int, j: int) -> bool:
        return 0 <= i < self.n and 0 <= j < self.n

    def _traversable_layer(self, yaw: float) -> np.ndarray:
        """Cells where the twin-cylinder footprint at ``yaw`` is collision-free."""
        (fi, fj), (ri, rj) = self.fp.cell_offsets(yaw, self.res)
        blocked = _shift(self.inflated, -fi, -fj) | _shift(self.inflated, -ri, -rj)
        return ~blocked

    def free_at(self, i: int, j: int, direction: tuple[int, int]) -> bool:
        layer = self._layers[direction]
        return bool(layer[i, j])

    def nearest_free(self, x: float, y: float, yaw: float,
                     max_r_m: float = 1.2) -> tuple[float, float] | None:
        """Nearest cell to (x, y) whose footprint at ``yaw`` is free.

        Used to repair a local goal the VLM aimed straight into furniture:
        the intent (that direction, that far) is kept, the last feasible
        point along it is what gets planned to.
        """
        layer = self._traversable_layer(yaw)
        ci, cj = self.cell(x, y)
        max_r = int(math.ceil(max_r_m / self.res))
        for r in range(max_r + 1):
            best = None
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    i, j = ci + di, cj + dj
                    if not self.in_window(i, j) or not layer[i, j]:
                        continue
                    d = di * di + dj * dj
                    if best is None or d < best[0]:
                        best = (d, i, j)
            if best is not None:
                return self.world(best[1], best[2])
        return None

    # -- search ------------------------------------------------------------

    def search(self, start_xy, goal_xy, allow_boundary_fallback: bool = True):
        """A* from start to goal.

        Returns ``(path, status)`` where path is a list of world (x, y) from
        start to the reached cell and status is one of ``reached``,
        ``fallback`` (dead end; path ends at the window boundary), or
        ``blocked`` (no motion possible at all).
        """
        si, sj = self.cell(*start_xy)
        gi, gj = self.cell(*goal_xy)
        if not self.in_window(si, sj):
            return [], "blocked"
        gi = min(max(gi, 0), self.n - 1)
        gj = min(max(gj, 0), self.n - 1)

        came: dict[int, tuple[int, tuple[int, int]]] = {}
        gscore = np.full(self.n * self.n, np.inf, np.float32)
        start = si * self.n + sj
        gscore[start] = 0.0
        goal = gi * self.n + gj
        # Boundary cells are the fallback targets: reaching one means the
        # hypothesis path left the window (the paper's virtual free layer).
        heap = [(0.0, start)]
        closed = np.zeros(self.n * self.n, bool)
        seen_boundary: int | None = None
        found = False
        while heap:
            _, cur = heapq.heappop(heap)
            if closed[cur]:
                continue
            closed[cur] = True
            if cur == goal:
                found = True
                break
            ci, cj = divmod(cur, self.n)
            # First boundary cell expanded: the heuristic orders the frontier
            # toward the goal, so this is where a hypothesis path would leave
            # the window on its way there. "Boundary" is a margin, not the
            # outermost row: a cylinder centre cannot sit in the last few
            # cells (their footprint query falls outside the window), so an
            # exact-edge test would never fire.
            edge = _BOUNDARY_MARGIN
            if seen_boundary is None and (
                ci < edge or ci >= self.n - edge or cj < edge or cj >= self.n - edge
            ):
                seen_boundary = cur
            g = gscore[cur]
            for di, dj, step in _NEIGHBORS:
                ni, nj = ci + di, cj + dj
                if not self.in_window(ni, nj):
                    continue
                if not self._layers[(di, dj)][ni, nj]:
                    continue
                cost = step * self.cell_cost[ni, nj]
                if self.unknown[ni, nj]:
                    cost *= self.unknown_cost
                nxt = ni * self.n + nj
                ng = g + cost
                if ng < gscore[nxt]:
                    gscore[nxt] = ng
                    came[nxt] = (cur, (di, dj))
                    h = math.hypot(ni - gi, nj - gj)
                    heapq.heappush(heap, (ng + h, nxt))

        if found:
            return self._unwind(came, start, goal), "reached"
        if not allow_boundary_fallback or seen_boundary is None:
            return [], "blocked"
        # Dead end: aim at the boundary cell the search escaped through. The
        # path stays inside the real map; only the target came from outside.
        return self._unwind(came, start, seen_boundary), "fallback"

    def _unwind(self, came, start: int, end: int) -> list[tuple[float, float]]:
        cells = [end]
        while cells[-1] != start:
            prev = came.get(cells[-1])
            if prev is None:
                return []
            cells.append(prev[0])
        cells.reverse()
        return [self.world(*divmod(c, self.n)) for c in cells]


def _shift(mask: np.ndarray, di: int, dj: int, fill: bool = True) -> np.ndarray:
    """Shift a window array by (di, dj) cells, filling exposed edges with
    ``fill`` (True = outside the window is blocked)."""
    out = np.full_like(mask, fill)
    n_i, n_j = mask.shape
    si_dst = slice(max(di, 0), n_i + min(di, 0))
    si_src = slice(max(-di, 0), n_i + min(-di, 0))
    sj_dst = slice(max(dj, 0), n_j + min(dj, 0))
    sj_src = slice(max(-dj, 0), n_j + min(-dj, 0))
    out[si_dst, sj_dst] = mask[si_src, sj_src]
    return out
