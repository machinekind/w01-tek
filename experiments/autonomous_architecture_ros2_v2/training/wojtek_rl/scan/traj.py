"""Uniform cubic B-spline trajectory + rebound optimisation (SCAN-Planner IV).

Position-only optimisation: the control points carry (x, y) and the yaw used
for collision checking is *induced* from the local tangent, Eq. (2). That is
the trick that keeps a quadruped's yaw-dependent footprint in the cost
without adding yaw decision variables -- and it matches how the robot is
actually driven, since the locomotion policy turns to face where it is going.

The collision term is EGO-Planner's rebound formulation, which the paper
adopts: no ESDF is built. When the initialisation clips an obstacle, the
projected A* guide path (guide.py) supplies, for each offending control
point, an anchor on the obstacle surface and a unit direction pointing out of
it; the penalty is a smooth one-sided function of the signed distance along
that direction. Obstacles enter the optimiser only through those lazily
constructed {a, v} pairs.

Everything is small: ~8-16 control points, so the difference operators are
dense matrices and the analytic gradient is a couple of matrix products.
L-BFGS converges in well under a millisecond, which is what makes a 10 Hz
replan affordable next to a 50 Hz control loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Cubic uniform B-spline basis matrix (de Boor), rows = u^3, u^2, u, 1.
_M = np.array(
    [[-1.0, 3.0, -3.0, 1.0],
     [3.0, -6.0, 3.0, 0.0],
     [-3.0, 0.0, 3.0, 0.0],
     [1.0, 4.0, 1.0, 0.0]]
) / 6.0

RHO_EPS = 1e-2  # knee of the C^2 one-sided penalty


@dataclass
class BSpline:
    """Uniform cubic B-spline over control points Q (n, 2) with knot span dt."""

    Q: np.ndarray
    dt: float

    @property
    def duration(self) -> float:
        return (len(self.Q) - 3) * self.dt

    def sample(self, t):
        """Position at time(s) t, clamped to the trajectory's domain."""
        t = np.atleast_1d(np.asarray(t, np.float64))
        s = np.clip(np.floor(t / self.dt).astype(int), 0, len(self.Q) - 4)
        u = np.clip(t / self.dt - s, 0.0, 1.0)
        basis = np.stack([u**3, u**2, u, np.ones_like(u)], axis=1) @ _M  # (k, 4)
        idx = s[:, None] + np.arange(4)[None, :]
        return np.einsum("kb,kbc->kc", basis, self.Q[idx])

    def velocity(self, t):
        """dp/dt at time(s) t."""
        t = np.atleast_1d(np.asarray(t, np.float64))
        s = np.clip(np.floor(t / self.dt).astype(int), 0, len(self.Q) - 4)
        u = np.clip(t / self.dt - s, 0.0, 1.0)
        dbasis = np.stack(
            [3 * u**2, 2 * u, np.ones_like(u), np.zeros_like(u)], axis=1
        ) @ _M / self.dt
        idx = s[:, None] + np.arange(4)[None, :]
        return np.einsum("kb,kbc->kc", dbasis, self.Q[idx])

    def induced_yaw(self) -> np.ndarray:
        """Eq. (2): yaw at each control point from the central difference."""
        Q = self.Q
        d = np.empty_like(Q)
        d[1:-1] = Q[2:] - Q[:-2]
        d[0] = Q[1] - Q[0]
        d[-1] = Q[-1] - Q[-2]
        yaw = np.arctan2(d[:, 1], d[:, 0])
        flat = np.hypot(d[:, 0], d[:, 1]) < 1e-9
        if flat.any():  # a stationary stretch keeps the last meaningful heading
            good = np.flatnonzero(~flat)
            yaw[flat] = yaw[good[0]] if len(good) else 0.0
        return yaw


@dataclass
class OptConfig:
    """Objective weights and dynamic limits (Eq. 6, 12, 13)."""

    lambda_c: float = 200.0   # rebound collision
    lambda_s: float = 1.0     # smoothness (raw third differences, see cost())
    lambda_f: float = 20.0    # velocity/acceleration feasibility
    v_max: float = 0.40       # m/s, the policy's commanded forward cap
    a_max: float = 0.80       # m/s^2
    safe_clearance: float = 0.08  # s_f in Eq. (10), on top of the inflation
    max_iter: int = 40
    outer_iter: int = 3       # re-detect collisions and re-solve this many times


@dataclass
class ReboundPairs:
    """Lazily built {anchor, direction} pairs, keyed by control-point index."""

    idx: list[int] = field(default_factory=list)
    anchor: list[np.ndarray] = field(default_factory=list)
    direction: list[np.ndarray] = field(default_factory=list)

    def add(self, i: int, a: np.ndarray, v: np.ndarray) -> None:
        self.idx.append(i)
        self.anchor.append(np.asarray(a, np.float64))
        self.direction.append(np.asarray(v, np.float64))

    def __len__(self) -> int:
        return len(self.idx)

    def arrays(self):
        if not self.idx:
            return (np.empty(0, int), np.empty((0, 2)), np.empty((0, 2)))
        return (np.asarray(self.idx), np.asarray(self.anchor), np.asarray(self.direction))


def diff_operator(n: int, dt: float, order: int) -> np.ndarray:
    """(n - order, n) matrix mapping control points to their k-th differences."""
    D = np.eye(n)
    for k in range(order):
        rows = n - k - 1
        step = np.zeros((rows, D.shape[0]))
        step[np.arange(rows), np.arange(rows)] = -1.0 / dt
        step[np.arange(rows), np.arange(rows) + 1] = 1.0 / dt
        D = step @ D
    return D


def rho(c: np.ndarray) -> np.ndarray:
    """C^2 one-sided penalty; zero for c <= 0 (clearance satisfied)."""
    c = np.asarray(c, np.float64)
    out = np.zeros_like(c)
    mid = (c > 0) & (c <= RHO_EPS)
    hi = c > RHO_EPS
    out[mid] = c[mid] ** 3
    out[hi] = 3 * RHO_EPS * c[hi] ** 2 - 3 * RHO_EPS**2 * c[hi] + RHO_EPS**3
    return out


def rho_grad(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.float64)
    out = np.zeros_like(c)
    mid = (c > 0) & (c <= RHO_EPS)
    hi = c > RHO_EPS
    out[mid] = 3 * c[mid] ** 2
    out[hi] = 6 * RHO_EPS * c[hi] - 3 * RHO_EPS**2
    return out


class TrajectoryOptimizer:
    """Minimise J = lc*Jc + ls*Js + lf*Jf over the interior control points."""

    FIXED = 3  # cubic: the first/last 3 control points pin the boundary state

    def __init__(self, cfg: OptConfig | None = None):
        self.cfg = cfg or OptConfig()

    def cost(self, Q: np.ndarray, dt: float, pairs: ReboundPairs):
        """(J, dJ/dQ) for the full control-point set."""
        cfg = self.cfg
        n = len(Q)
        grad = np.zeros_like(Q)
        total = 0.0

        # Smoothness on the raw third differences of the control points, not
        # on jerk in m/s^3 (EGO-Planner's convention). Dividing by dt^3 would
        # scale this term by ~500 at dt = 0.35 s and drown every other term:
        # the optimiser would rather clip a chair than bend the curve.
        D3 = diff_operator(n, 1.0, 3)
        J3 = D3 @ Q
        total += cfg.lambda_s * float(np.sum(J3**2))
        grad += cfg.lambda_s * 2.0 * (D3.T @ J3)

        for order, limit in ((1, cfg.v_max), (2, cfg.a_max)):
            D = diff_operator(n, dt, order)
            X = D @ Q
            norm = np.linalg.norm(X, axis=1)
            over = np.maximum(norm - limit, 0.0)
            if over.any():
                total += cfg.lambda_f * float(np.sum(over**2))
                safe = np.maximum(norm, 1e-9)[:, None]
                grad += cfg.lambda_f * (D.T @ (2.0 * over[:, None] * X / safe))

        idx, anchor, direction = pairs.arrays()
        if len(idx):
            d = np.sum((Q[idx] - anchor) * direction, axis=1)
            c = cfg.safe_clearance - d
            total += cfg.lambda_c * float(np.sum(rho(c)))
            g = -cfg.lambda_c * rho_grad(c)[:, None] * direction
            np.add.at(grad, idx, g)
        return total, grad

    def solve(self, spline: BSpline, pairs: ReboundPairs) -> BSpline:
        """One L-BFGS solve with the pairs held fixed (EGO's lazy scheme)."""
        from scipy.optimize import minimize

        Q0 = spline.Q.copy()
        n = len(Q0)
        free = slice(self.FIXED, n - self.FIXED)
        if free.stop <= free.start:
            return spline

        def fun(z):
            Q = Q0.copy()
            Q[free] = z.reshape(-1, 2)
            J, g = self.cost(Q, spline.dt, pairs)
            return J, g[free].reshape(-1)

        res = minimize(
            fun, Q0[free].reshape(-1), jac=True, method="L-BFGS-B",
            options={"maxiter": self.cfg.max_iter},
        )
        Q = Q0.copy()
        Q[free] = res.x.reshape(-1, 2)
        return BSpline(Q, spline.dt)


def optimize_with_rebound(spline: BSpline, guide_xy, omap, footprint,
                          optimizer: "TrajectoryOptimizer") -> tuple[BSpline, int]:
    """EGO's lazy loop: detect collisions, build {a, v} pairs, solve, repeat.

    One solve is rarely enough -- the pairs are built from the *current*
    control points, and pushing them out moves which points are in collision.
    Returns the optimised spline and how many pairs the last round needed
    (zero means the initialisation was already clear).
    """
    rebounds = 0
    for _ in range(optimizer.cfg.outer_iter):
        pairs = build_rebound_pairs(spline, guide_xy, omap, footprint)
        if not len(pairs):
            break
        rebounds = len(pairs)
        spline = optimizer.solve(spline, pairs)
    return spline, rebounds


def init_from_path(path_xy, dt: float, state, v_nom: float,
                   horizon_s: float = 3.0, min_points: int = 8,
                   max_points: int = 20) -> BSpline:
    """Control points from a guide path, with the start pinned to the state.

    ``state`` is (position, velocity, acceleration) in world frame. The path
    is resampled by arc length at the distance one knot span covers at the
    nominal speed, so the initialisation already has roughly feasible
    velocities before the optimiser sees it, and the trailing three control
    points coincide -- the trajectory comes to rest at its horizon end, which
    is what a mid-level "forward 1.5" means when the horizon is the goal.
    """
    pts = np.asarray(path_xy, np.float64).reshape(-1, 2)
    p0, v0, a0 = (np.asarray(s, np.float64) for s in state)
    if len(pts) < 2:
        pts = np.stack([p0, p0])

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    length = min(float(arc[-1]), v_nom * horizon_s)
    span = max(v_nom * dt, 1e-3)
    n = int(np.clip(round(length / span) + 3, min_points, max_points))
    s = np.linspace(0.0, length, n)
    samples = np.stack([np.interp(s, arc, pts[:, 0]), np.interp(s, arc, pts[:, 1])], axis=1)

    Q = samples
    # Boundary state -> first three control points (p(0), v(0), a(0) of a
    # uniform cubic B-spline are (Q0+4Q1+Q2)/6, (Q2-Q0)/2dt, (Q0-2Q1+Q2)/dt^2).
    Q[0] = p0 - v0 * dt + a0 * dt * dt / 3.0
    Q[1] = p0 - a0 * dt * dt / 6.0
    Q[2] = p0 + v0 * dt + a0 * dt * dt / 3.0
    Q[-3:] = samples[-1]
    return BSpline(Q, dt)


def build_rebound_pairs(spline: BSpline, guide_xy, omap, footprint,
                        step_m: float | None = None) -> ReboundPairs:
    """Detect colliding control points and build their {a, v} pairs.

    For a colliding Q_i the guide path supplies a collision-free counterpart
    p_i; marching from p_i back toward Q_i finds where the inflated obstacle
    starts, and that transition point is the anchor. The direction points out
    of the obstacle, so the penalty gradient pushes the control point into
    free space -- laterally, since the guide path detoured laterally.
    """
    pairs = ReboundPairs()
    guide = np.asarray(guide_xy, np.float64).reshape(-1, 2)
    if len(guide) < 2:
        return pairs
    Q = spline.Q
    yaw = spline.induced_yaw()
    res = omap.cfg.res
    step = step_m or 0.5 * res

    hit = footprint.collides(omap, Q[:, 0], Q[:, 1], yaw)
    g_arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(guide, axis=0), axis=1))]
    )
    for i in np.flatnonzero(hit):
        if i < TrajectoryOptimizer.FIXED or i >= len(Q) - TrajectoryOptimizer.FIXED:
            continue
        # Counterpart on the guide path at the same fraction of the route.
        frac = i / max(len(Q) - 1, 1)
        target = np.array([
            np.interp(frac * g_arc[-1], g_arc, guide[:, 0]),
            np.interp(frac * g_arc[-1], g_arc, guide[:, 1]),
        ])
        delta = target - Q[i]
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            continue
        v = delta / dist
        anchor = _surface_anchor(Q[i], v, dist, yaw[i], omap, footprint, step)
        if anchor is not None:
            pairs.add(int(i), anchor, v)
    return pairs


def _surface_anchor(q, v, dist, yaw, omap, footprint, step):
    """First collision-free point marching from q along v; None if none is."""
    k = max(2, int(math.ceil(dist / step)))
    ts = np.linspace(0.0, dist, k)
    pts = q[None, :] + ts[:, None] * v[None, :]
    free = ~footprint.collides(omap, pts[:, 0], pts[:, 1], np.full(k, yaw))
    idx = np.flatnonzero(free)
    if not len(idx):
        return None
    return pts[idx[0]]
