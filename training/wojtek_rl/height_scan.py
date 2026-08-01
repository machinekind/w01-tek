"""Body-frame height scan: grid geometry, camera visibility, sensor corruption.

Pure functions on numpy/jax arrays. The terrain height lookup and the
reference z are caller-supplied, so nothing here needs a model or an env.
"""

import jax
import jax.numpy as jp
import numpy as np

# Corruption regimes.
NOISE, DRIFT, BLACKOUT = 0, 1, 2

# Grid shape the observation components are sized for. symmetry.py pins the
# component width to it, so another grid needs both changed.
NX, NY = 5, 5
SIZE = NX * NY


def body_grid(x_range, y_range, nx=NX, ny=NY, xp=np):
    """(nx*ny, 2) body-frame grid points, index ix*ny + iy."""
    xs = xp.linspace(x_range[0], x_range[1], nx)
    ys = xp.linspace(y_range[0], y_range[1], ny)
    return xp.stack([xp.repeat(xs, ny), xp.tile(ys, nx)], axis=-1)


def mirror_map(nx=NX, ny=NY):
    """(perm, sign) mirroring the flattened grid about the body xz-plane."""
    perm = np.arange(nx * ny).reshape(nx, ny)[:, ::-1].reshape(-1)
    return perm, np.ones(nx * ny)


def yaw_from_quat(quat, xp=jp):
    """Heading of a (w, x, y, z) quaternion."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return xp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotation_matrix(quat, xp=jp):
    """(w, x, y, z) quaternion as a body-to-world rotation matrix."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return xp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def world_xy(grid, base_xy, yaw, xp=jp):
    """Body-frame xy points placed at a base pose, in world coordinates."""
    c, s = xp.cos(yaw), xp.sin(yaw)
    gx, gy = grid[..., 0], grid[..., 1]
    return xp.stack([gx * c - gy * s, gx * s + gy * c], axis=-1) + base_xy


def scan_values(heights, ref_z, clip, xp=jp):
    """Terrain height relative to a reference z, clipped to +-clip."""
    return xp.clip(heights - ref_z, -clip, clip)


def visible_mask(
    points, base_pos, quat, mount, pitch_deg, hfov_deg, vfov_deg,
    min_depth, max_depth, xp=jp,
):
    """Which world `points` a body-mounted depth camera can see.

    The camera sits at `mount` in the body frame with its optical axis
    pitched `pitch_deg` below the body x-axis. Its frame follows the full
    base orientation, so base pitch and roll swing the frustum off the
    ground the yaw-placed grid sits on.
    """
    rot = rotation_matrix(quat, xp=xp)
    pitch = xp.radians(pitch_deg)
    fwd_b = xp.array([xp.cos(pitch), 0.0, -xp.sin(pitch)])
    up_b = xp.array([xp.sin(pitch), 0.0, xp.cos(pitch)])
    body = (points - (base_pos + rot @ mount)) @ rot
    fwd = body @ fwd_b
    lat = body[..., 1]
    up = body @ up_b
    dist = xp.linalg.norm(body, axis=-1)
    return (
        (fwd > 0.0)
        & (xp.abs(xp.arctan2(lat, fwd)) <= xp.radians(hfov_deg) / 2.0)
        & (xp.abs(xp.arctan2(up, fwd)) <= xp.radians(vfov_deg) / 2.0)
        & (dist >= min_depth)
        & (dist <= max_depth)
    )


def sample_corruption(
    rng, noise_prob, drift_prob, blackout_prob, drift_z, drift_tilt
):
    """(regime, (drift_z, drift_tilt)) for one episode."""
    r_regime, r_drift = jax.random.split(rng)
    probs = jp.array([noise_prob, drift_prob, blackout_prob])
    regime = jax.random.choice(r_regime, 3, p=probs / jp.sum(probs))
    drift = jax.random.uniform(
        r_drift,
        (2,),
        minval=jp.array([-drift_z, -drift_tilt]),
        maxval=jp.array([drift_z, drift_tilt]),
    )
    return regime.astype(jp.int32), drift


def apply_corruption(
    rng, scan, regime, drift, grid_x, noise_std, dropout_prob
):
    """Actor-side sensor corruption of one masked scan.

    NOISE adds iid gaussian, DRIFT a per-episode offset growing with
    body-frame x, BLACKOUT returns zeros. NOISE and DRIFT also drop
    individual points to zero.
    """
    r_noise, r_drop = jax.random.split(rng)
    noisy = scan + noise_std * jax.random.normal(r_noise, scan.shape)
    drifted = scan + drift[0] + drift[1] * (grid_x - jp.min(grid_x))
    out = jp.where(regime == DRIFT, drifted, noisy)
    keep = jax.random.bernoulli(r_drop, 1.0 - dropout_prob, scan.shape)
    out = jp.where(keep, out, 0.0)
    return jp.where(regime == BLACKOUT, jp.zeros_like(scan), out)
