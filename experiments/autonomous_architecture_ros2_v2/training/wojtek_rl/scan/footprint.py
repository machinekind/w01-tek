"""Yaw-aware twin-cylinder footprint (SCAN-Planner, Sec. IV-B).

A quadruped is elongated: a circle big enough to contain it (r ~ 0.30 m for
Wojtek, nose to tail) refuses passages the robot fits through, and a circle
sized to its width lets the nose clip a chair leg while turning. The paper's
answer is two vertical cylinders on the body's longitudinal axis. Inflate the
map by the cylinder radius once, and whole-body collision checking collapses
to two point queries at the transformed cylinder centres -- cheap enough to
run inside an A* expansion and inside every optimizer iteration.

Wojtek geometry (ros/src/wojtek_description/mujoco/wojtek.xml): hips at
x = +-0.131 m, stance half-width 0.174 m, front/rear feet at x = +-0.257 m.
Defaults d_off 0.25 / radius 0.20 were chosen visually against the walking
robot in the /tune helper (2026-07-27): each disc sits on a leg pair, which
is where the measured envelope actually bulges. Note the discs are disjoint
(d_off > radius), so the waist between them is uncovered -- a deliberate
trade accepting mid-body side-swipes as unlikely in exchange for footprints
that hug the legs.

The vertical clearances d_up / d_down of the paper live in the map instead
(MapConfig.z_step / z_up): a point below z_step is a steppable lip, a point
above z_up is overhead clearance the robot walks under, and only the band
between them ever enters the grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Footprint:
    """Twin-cylinder body model. ``radius`` must match MapConfig.inflate_r.

    ``sweep_margin`` widens the turn-sweep check beyond the map's inflation.
    Measured from the collision meshes over a walking rollout the body really
    reaches |y| = 0.248 m and x = +-0.38 m, so the 0.19 m cylinders do not
    cover the legs -- but raising this measured *nothing* on either scene
    (room 8 vs 8 contacts, apartment 177 vs 181) while costing turns the
    robot could have made (apartment SR 0.88 -> 0.75 at 0.07). Default off;
    the clearance-cost layer in guide.py is what actually buys margin.
    """

    d_off: float = 0.25
    radius: float = 0.20
    sweep_margin: float = 0.0

    @property
    def length(self) -> float:
        return 2 * (self.d_off + self.radius)

    def centers(self, x, y, yaw):
        """(front, rear) cylinder centres in world frame; vectorized over poses."""
        x, y, yaw = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(yaw)
        cx, sy_ = np.cos(yaw), np.sin(yaw)
        front = np.stack([x + self.d_off * cx, y + self.d_off * sy_], axis=-1)
        rear = np.stack([x - self.d_off * cx, y - self.d_off * sy_], axis=-1)
        return front, rear

    def collides(self, omap, x, y, yaw, margin: float = 0.0):
        """Whole-body collision indicator H_k of Eq. (5), on the inflated map.

        ``margin`` widens the effective cylinder beyond the map's inflation
        radius by sampling a ring of that radius around each centre -- the map
        answers "is anything within inflate_r of this point", so a ring of
        radius m answers it for inflate_r + m without a second dilation layer.
        Used where extra caution is cheap (see sweep_free), while planning
        keeps the tighter radius that fits through doorways.
        """
        scalar = np.ndim(x) == 0
        front, rear = self.centers(x, y, yaw)
        probes = [front, rear]
        if margin > 0.0:
            ring = [(math.cos(a) * margin, math.sin(a) * margin)
                    for a in np.linspace(0.0, 2 * math.pi, 8, endpoint=False)]
            probes += [c + np.array(o) for c in (front, rear) for o in ring]
        hit = np.zeros(len(np.atleast_1d(x)), bool)
        for p in probes:
            hit |= omap.is_inflated_occupied(p[:, 0], p[:, 1])
        return bool(hit[0]) if scalar else hit

    def sweep_free(self, omap, x: float, y: float, yaw0: float, yaw1: float,
                   step_rad: float = 0.17) -> float:
        """Largest rotation from yaw0 toward yaw1 that stays collision-free.

        Rotating in place sweeps an elongated body through orientations the
        planner never checked -- the yaw-aware footprint says the body fits
        *at this yaw*, not at every yaw in between. Returns the signed angle
        that can be turned (0.0 if even the first step collides, the full
        delta if the whole sweep is clear).
        """
        delta = (yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi
        n = max(1, int(math.ceil(abs(delta) / step_rad)))
        yaws = yaw0 + np.linspace(delta / n, delta, n)
        hit = self.collides(omap, np.full(n, x), np.full(n, y), yaws,
                            margin=self.sweep_margin)
        bad = np.flatnonzero(hit)
        if not len(bad):
            return float(delta)
        return float(delta * (bad[0] / n))

    def cell_offsets(self, yaw: float, res: float) -> tuple[tuple[int, int], tuple[int, int]]:
        """Cylinder offsets in (row, col) cells for a fixed yaw.

        Lets a whole traversability layer be built with two array shifts
        instead of a per-cell query (guide.py). Rounding to the grid adds at
        most half a cell (2.5 cm) of conservatism.
        """
        di = int(round(self.d_off * math.sin(yaw) / res))
        dj = int(round(self.d_off * math.cos(yaw) / res))
        return (di, dj), (-di, -dj)
