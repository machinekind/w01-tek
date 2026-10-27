"""Spatial memory the agent can talk about: pose history + map summaries.

Pure numpy/text -- no mujoco, no model. Everything here turns the robot's
self-built OnlineMap and its recent trajectory into short factual text the
small VLM can quote when asked "what do we know about the room?" or
"describe your last 3 seconds of route". Text beats handing the model a raw
grid; the map image goes alongside as a second content part.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from wojtek_eval.mapping import FREE, OCC, OnlineMap


@dataclass(frozen=True)
class PoseSample:
    t: float
    x: float
    y: float
    yaw: float


class PoseHistory:
    """Timestamped pose ring buffer, appended by the sim loop at 50 Hz.

    OnlineMap.trail keeps positions but no timestamps and no yaw, so "the
    past 3 seconds" is unanswerable from it; this buffer exists exactly for
    time-windowed route questions. 6000 samples = 120 s at 50 Hz.
    """

    def __init__(self, maxlen: int = 6000):
        self._buf: deque[PoseSample] = deque(maxlen=maxlen)

    def add(self, t: float, x: float, y: float, yaw: float) -> None:
        self._buf.append(PoseSample(t, x, y, yaw))

    def __len__(self) -> int:
        return len(self._buf)

    def window(self, seconds: float) -> list[PoseSample]:
        if not self._buf:
            return []
        cutoff = self._buf[-1].t - seconds
        return [s for s in self._buf if s.t >= cutoff]

    def describe(self, seconds: float = 3.0) -> str:
        """Human-readable route summary over the trailing window."""
        w = self.window(seconds)
        if len(w) < 2:
            return "no route recorded yet"
        first, last = w[0], w[-1]
        span = last.t - first.t
        path = sum(
            math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(w, w[1:])
        )
        net = math.hypot(last.x - first.x, last.y - first.y)
        # Net displacement expressed in the starting body frame, so the text
        # reads as the robot experienced it ("forward and a bit left").
        dx, dy = last.x - first.x, last.y - first.y
        fwd = dx * math.cos(first.yaw) + dy * math.sin(first.yaw)
        lat = -dx * math.sin(first.yaw) + dy * math.cos(first.yaw)
        turn = math.degrees(_wrap(last.yaw - first.yaw))
        side = "left" if turn >= 0 else "right"
        parts = [f"over the last {span:.1f} s: walked {path:.2f} m of path"]
        if net < 0.05 and abs(turn) >= 10:
            parts.append(f"turned {abs(turn):.0f} deg {side} roughly in place")
        else:
            parts.append(
                f"net move {fwd:+.2f} m forward, {lat:+.2f} m to the left"
                if abs(lat) >= 0.05
                else f"net move {fwd:+.2f} m forward"
            )
            if abs(turn) >= 10:
                parts.append(f"heading changed {abs(turn):.0f} deg {side}")
        return "; ".join(parts)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class FrontierCluster:
    """One connected frontier segment, reduced to its midpoint."""

    x: float
    y: float
    cells: int


def frontier_clusters(omap: OnlineMap, min_cells: int = 4) -> list[FrontierCluster]:
    """Connected components (8-neighborhood) of the frontier cell set.

    min_cells drops depth-noise speckle -- single stray frontier cells that
    would otherwise each look like an unexplored doorway.
    """
    cells = omap.frontier_cells()
    if not len(cells):
        return []
    remaining = {(int(i), int(j)) for i, j in cells}
    out: list[FrontierCluster] = []
    while remaining:
        seed = remaining.pop()
        comp = [seed]
        stack = [seed]
        while stack:
            ci, cj = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nxt = (ci + di, cj + dj)
                    if nxt in remaining:
                        remaining.discard(nxt)
                        comp.append(nxt)
                        stack.append(nxt)
        if len(comp) < min_cells:
            continue
        arr = np.asarray(comp, float)
        mi, mj = arr.mean(axis=0)
        wx, wy = omap.cell_to_world(int(round(mi)), int(round(mj)))
        out.append(FrontierCluster(x=wx, y=wy, cells=len(comp)))
    out.sort(key=lambda c: -c.cells)
    return out


def bearing_text(x: float, y: float, pose: tuple[float, float, float]) -> str:
    """'2.1 m away, 40 deg to the left' relative to the robot's heading."""
    px, py, yaw = pose
    dist = math.hypot(x - px, y - py)
    rel = math.degrees(_wrap(math.atan2(y - py, x - px) - yaw))
    if abs(rel) < 15:
        where = "ahead"
    elif abs(rel) > 150:
        where = "behind"
    else:
        where = f"{abs(rel):.0f} deg to the {'left' if rel > 0 else 'right'}"
    return f"{dist:.1f} m away, {where}"


def map_summary(
    omap: OnlineMap,
    pose: tuple[float, float, float],
    clusters: list[FrontierCluster] | None = None,
    max_frontiers: int = 5,
) -> str:
    """What the self-built map knows, as a few factual lines."""
    cell_m2 = omap.res * omap.res
    free = int((omap.state == FREE).sum())
    occ = int((omap.state == OCC).sum())
    total = omap.state.size
    seen_frac = (free + occ) / total if total else 0.0
    if clusters is None:
        clusters = frontier_clusters(omap)
    lines = [
        f"explored {free * cell_m2:.1f} m2 of floor, {occ} obstacle cells mapped, "
        f"{seen_frac:.0%} of the map area seen",
        f"travelled {len(omap.trail)} recorded steps since the map started",
    ]
    if clusters:
        lines.append(f"{len(clusters)} unexplored openings (frontiers):")
        for c in clusters[:max_frontiers]:
            lines.append(f"  - frontier ({c.cells} cells): {bearing_text(c.x, c.y, pose)}")
    else:
        lines.append("no frontiers left -- the reachable area looks fully explored")
    return "\n".join(lines)
