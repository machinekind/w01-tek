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

# Occlusion rays: where along the camera-to-point segment the terrain samples
# start, and how far short of the point they stop. A sample inside the point's
# own terrain cell (0.04 m) reads that point's surface and would occlude it
# with itself.
OCC_T_MIN = 0.1
OCC_END_GAP = 0.05


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


def camera_pos(base_pos, quat, mount, xp=jp):
    """World position of a body-frame `mount` at a base pose."""
    return base_pos + rotation_matrix(quat, xp=xp) @ mount


def visible_mask(
    points, base_pos, quat, mount, pitch_deg, hfov_deg, vfov_deg,
    min_depth, max_depth, xp=jp,
):
    """Which world `points` a body-mounted depth camera can see.

    The camera sits at `mount` in the body frame with its optical axis
    pitched `pitch_deg` below the body x-axis. Its frame follows the full
    base orientation, so base pitch and roll swing the frustum off the
    ground the yaw-placed grid sits on. Frustum only: `occluded_mask` is
    the line-of-sight test.
    """
    rot = rotation_matrix(quat, xp=xp)
    pitch = xp.radians(pitch_deg)
    fwd_b = xp.array([xp.cos(pitch), 0.0, -xp.sin(pitch)])
    up_b = xp.array([xp.sin(pitch), 0.0, xp.cos(pitch)])
    body = (points - camera_pos(base_pos, quat, mount, xp=xp)) @ rot
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


def occluded_mask(points, cam_pos, height_fn, num_samples, margin, xp=jp):
    """Which world `points` nearer terrain hides from a camera at `cam_pos`.

    Each camera-to-point segment is sampled at `num_samples` interior
    fractions; a sample whose ground stands more than `margin` above the ray
    blocks it. `height_fn` maps world xy `(..., 2)` to ground height, so the
    whole batch is one lookup. `num_samples` is static: the sampling never
    depends on the data, only on the config.
    """
    delta = points - cam_pos
    dist = xp.linalg.norm(delta, axis=-1)
    t_end = xp.clip(1.0 - OCC_END_GAP / xp.maximum(dist, 1e-6), OCC_T_MIN, 1.0)
    span = xp.linspace(0.0, 1.0, num_samples)
    t = OCC_T_MIN + (t_end - OCC_T_MIN)[..., None] * span
    ray = cam_pos + t[..., None] * delta[..., None, :]
    ground = height_fn(ray[..., 0:2])
    return xp.any(ground > ray[..., 2] + margin, axis=-1)


def sample_corruption(
    rng, noise_prob, drift_prob, blackout_prob, drift_z, drift_tilt,
    pitch_jitter_deg=0.0, mount_jitter=0.0,
):
    """(regime, (drift_z, drift_tilt), (dpitch_deg, dx, dy, dz)) per episode.

    The last vector is the camera pose the actor's mask is computed from,
    as an offset from the nominal mount. It moves what the mask keeps, not
    the values themselves; zero bounds draw exact zeros.
    """
    r_regime, r_drift, r_cam = jax.random.split(rng, 3)
    probs = jp.array([noise_prob, drift_prob, blackout_prob])
    regime = jax.random.choice(r_regime, 3, p=probs / jp.sum(probs))
    drift = jax.random.uniform(
        r_drift,
        (2,),
        minval=jp.array([-drift_z, -drift_tilt]),
        maxval=jp.array([drift_z, drift_tilt]),
    )
    bound = jp.array(
        [pitch_jitter_deg, mount_jitter, mount_jitter, mount_jitter]
    )
    cam_jit = jax.random.uniform(r_cam, (4,), minval=-bound, maxval=bound)
    return regime.astype(jp.int32), drift, cam_jit


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
