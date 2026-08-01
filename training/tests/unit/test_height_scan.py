"""Height-scan geometry, camera mask and corruption, model-free."""

import jax
import numpy as np
import pytest

from wojtek_rl import height_scan

X_RANGE = (0.65, 1.45)
Y_RANGE = (-0.3, 0.3)
MOUNT = np.array([0.32, 0.0, 0.07])
CAM = dict(
    mount=MOUNT, pitch_deg=15.0, hfov_deg=90.7, vfov_deg=61.2,
    min_depth=0.3, max_depth=3.0,
)
UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])
ORIGIN = np.zeros(3)


def _pitch_quat(deg):
    """Rotation about the body y-axis; positive pitches the nose down."""
    a = np.radians(deg) / 2.0
    return np.array([np.cos(a), 0.0, np.sin(a), 0.0])


def _sees(point, quat=UPRIGHT, base_pos=ORIGIN):
    mask = height_scan.visible_mask(
        np.asarray(point)[None, :], base_pos, quat, **CAM
    )
    return bool(mask[0])


def test_grid_shape_ordering_and_endpoints():
    grid = height_scan.body_grid(X_RANGE, Y_RANGE)
    assert grid.shape == (height_scan.SIZE, 2)
    xs = grid.reshape(5, 5, 2)[:, 0, 0]
    ys = grid.reshape(5, 5, 2)[0, :, 1]
    np.testing.assert_allclose(xs, np.linspace(0.65, 1.45, 5))
    np.testing.assert_allclose(ys, np.linspace(-0.3, 0.3, 5))
    # index ix*ny + iy: x is constant within a row of five
    np.testing.assert_allclose(grid[0:5, 0], 0.65)
    np.testing.assert_allclose(grid[20:25, 0], 1.45)
    np.testing.assert_allclose(grid[0:5, 1], np.linspace(-0.3, 0.3, 5))


def test_world_placement_rotates_by_yaw_then_translates():
    grid = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = height_scan.world_xy(grid, np.array([2.0, -1.0]), np.pi / 2)
    np.testing.assert_allclose(out, [[2.0, 0.0], [1.0, -1.0]], atol=1e-6)


def test_scan_values_are_relative_and_clipped():
    h = np.array([0.5, -0.5, 0.02])
    np.testing.assert_allclose(
        height_scan.scan_values(h, 0.01, 0.3), [0.3, -0.3, 0.01], atol=1e-7
    )


def test_point_ahead_at_mid_range_is_visible():
    assert _sees([1.5, 0.0, 0.0])


@pytest.mark.parametrize(
    "point,reason",
    [
        ([-1.0, 0.0, 0.0], "behind the camera"),
        ([0.4, 0.0, 0.0], "closer than min_depth"),
        ([4.0, 0.0, 0.0], "further than max_depth"),
        ([1.0, 2.0, 0.0], "outside the horizontal fov"),
        ([1.32, 0.0, 0.87], "outside the vertical fov"),
    ],
)
def test_points_outside_the_frustum_are_invisible(point, reason):
    assert not _sees(point), reason


def test_nose_up_pitch_hides_the_near_row():
    grid = height_scan.body_grid(X_RANGE, Y_RANGE)
    near = np.concatenate([grid[0:5], np.zeros((5, 1))], axis=-1)
    level = height_scan.visible_mask(near, ORIGIN, UPRIGHT, **CAM)
    nose_up = height_scan.visible_mask(near, ORIGIN, _pitch_quat(-40.0), **CAM)
    assert bool(np.all(level))
    assert not bool(np.any(nose_up))


# -- occlusion ---------------------------------------------------------------

CAM_POS = np.array([0.0, 0.0, 0.4])


def _flat(xy):
    return np.zeros(np.shape(xy)[:-1])


def _ridge(x0, x1, top):
    """Ground at `top` for x in [x0, x1], zero elsewhere."""
    def height_fn(xy):
        return np.where((xy[..., 0] >= x0) & (xy[..., 0] <= x1), top, 0.0)

    return height_fn


def _hidden(points, height_fn, num_samples=8, margin=0.02):
    return np.asarray(
        height_scan.occluded_mask(
            np.asarray(points), CAM_POS, height_fn, num_samples, margin, xp=np
        )
    )


def test_flat_ground_hides_nothing():
    grid = height_scan.body_grid(X_RANGE, Y_RANGE)
    points = np.concatenate([grid, np.zeros((height_scan.SIZE, 1))], axis=-1)
    assert not _hidden(points, _flat).any()


def test_a_ridge_hides_what_is_behind_it_only():
    points = np.array([[0.5, 0.0, 0.0], [1.4, 0.0, 0.0]])
    near, far = _hidden(points, _ridge(0.6, 0.9, 0.25))
    assert not near
    assert far


@pytest.mark.parametrize("rise,blocked", [(0.01, False), (0.05, True)])
def test_margin_is_the_clearance_the_ground_needs(rise, blocked):
    # Camera and point at the same height: the ray runs level at z, so a
    # ridge blocks it exactly when it stands more than the margin above.
    point = np.array([[1.4, 0.0, CAM_POS[2]]])
    ridge = _ridge(0.6, 0.9, CAM_POS[2] + rise)
    assert bool(_hidden(point, ridge, margin=0.02)[0]) is blocked


def test_sample_count_is_fixed_and_the_batch_is_one_lookup():
    seen = []

    def height_fn(xy):
        seen.append(np.asarray(xy))
        return np.zeros(np.shape(xy)[:-1])

    points = np.stack([[0.7, 0.0, 0.0], [1.1, 0.2, 0.1], [1.4, -0.3, 0.0]])
    _hidden(points, height_fn, num_samples=6)
    assert [a.shape for a in seen] == [(3, 6, 2)]


def test_samples_stay_off_both_ends_of_the_ray():
    seen = []

    def height_fn(xy):
        seen.append(np.asarray(xy))
        return np.zeros(np.shape(xy)[:-1])

    _hidden(np.array([[1.4, 0.0, 0.0]]), height_fn)
    x = seen[0][0, :, 0]
    assert x.min() > 0.0
    assert x.max() < 1.4 - 0.04  # the point's own terrain cell is excluded


# -- corruption -------------------------------------------------------------

GRID_X = height_scan.body_grid(X_RANGE, Y_RANGE)[:, 0]
SCAN = np.linspace(-0.1, 0.1, height_scan.SIZE)
DRIFT = np.array([0.04, 0.03])


def _corrupt(regime, key=0, drift=DRIFT, noise_std=0.02, dropout_prob=0.0):
    return np.asarray(
        height_scan.apply_corruption(
            jax.random.PRNGKey(key), SCAN, regime, drift, GRID_X,
            noise_std, dropout_prob,
        )
    )


def test_blackout_zeroes_everything():
    assert not np.any(_corrupt(height_scan.BLACKOUT, dropout_prob=0.5))


def test_drift_is_a_constant_ramp_in_grid_x():
    out = _corrupt(height_scan.DRIFT)
    np.testing.assert_allclose(
        out, SCAN + DRIFT[0] + DRIFT[1] * (GRID_X - 0.65), atol=1e-6
    )
    # same params, different key: no per-step randomness in this regime
    np.testing.assert_allclose(out, _corrupt(height_scan.DRIFT, key=7), atol=1e-7)


def test_noise_differs_across_keys_and_stays_near_the_clean_scan():
    a = _corrupt(height_scan.NOISE)
    b = _corrupt(height_scan.NOISE, key=7)
    assert not np.allclose(a, b)
    assert np.abs(a - SCAN).max() < 0.15


def test_dropout_zeroes_some_points():
    out = _corrupt(height_scan.NOISE, noise_std=0.0, dropout_prob=0.5)
    dropped = out == 0.0
    assert dropped.any() and not dropped.all()


def test_regime_sampling_follows_the_probabilities():
    keys = jax.random.split(jax.random.PRNGKey(0), 4000)
    regimes, drift = jax.vmap(
        lambda k: height_scan.sample_corruption(k, 0.6, 0.3, 0.1, 0.05, 0.05)
    )(keys)
    regimes = np.asarray(regimes)
    for regime, want in ((0, 0.6), (1, 0.3), (2, 0.1)):
        assert abs((regimes == regime).mean() - want) < 0.04
    drift = np.asarray(drift)
    assert np.abs(drift).max() <= 0.05
    assert drift.min() < -0.03 and drift.max() > 0.03


def test_regime_sampling_is_deterministic_per_key():
    a = height_scan.sample_corruption(jax.random.PRNGKey(3), 0.6, 0.3, 0.1, 0.05, 0.05)
    b = height_scan.sample_corruption(jax.random.PRNGKey(3), 0.6, 0.3, 0.1, 0.05, 0.05)
    assert int(a[0]) == int(b[0])
    np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))


# -- mirror map -------------------------------------------------------------


def test_mirror_map_is_an_involution_that_reverses_y_only():
    perm, sign = height_scan.mirror_map()
    assert perm.shape == sign.shape == (height_scan.SIZE,)
    assert set(sign.tolist()) == {1.0}
    x = np.arange(float(height_scan.SIZE))
    np.testing.assert_array_equal((sign * x[perm])[perm] * sign, x)
    # within each x-row the y index reverses; the rows themselves stay put
    np.testing.assert_array_equal(
        x[perm].reshape(5, 5), x.reshape(5, 5)[:, ::-1]
    )
