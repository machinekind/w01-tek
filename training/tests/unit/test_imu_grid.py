"""IMU robustness grid, model-free: the spectral detector actually detects
the failure mode it was built for (a 25 Hz standing limit cycle at the
50 Hz control rate), the grid's cell layout, and the bias vectors. Rollout
application shares the courses' pin mechanism and the env's info["gyro_bias"]
path, covered by test_gyro_bias.py.
"""

import numpy as np
import pytest

from wojtek_rl.imu_grid import (
    NYQUIST_BAND_HZ,
    bias_vector,
    grid_cells,
    scenario_metrics,
)

DT = 0.02  # the 50 Hz control rate the real limit cycle halved


def _qvel(signal):
    """A (T, 3) joint-velocity history: `signal` on one joint, quiet others."""
    return np.stack([signal, 0.01 * signal, np.zeros_like(signal)], axis=1)


def test_limit_cycle_lands_in_the_band():
    # The real failure: alternation at half the control rate = 25 Hz, the
    # last rfft bin at dt=0.02. The detector must call this one loudly.
    t = np.arange(500)
    cycle = 0.5 * np.cos(np.pi * t)  # +A, -A, +A, ... = 25 Hz exactly
    m = scenario_metrics(_qvel(cycle), DT)
    assert m["band_20_25"] > 0.95
    assert m["vibration"] > 0.95
    # The absolute-scale guard: a 0.5 rad/s cycle is real joint motion.
    assert 0.2 < m["qvel_rms"] < 0.5


def test_healthy_stand_stays_out_of_the_band():
    # Slow postural sway (0.5 Hz) plus mild broadband noise: some >5 Hz
    # power exists, but the near-Nyquist band stays a small fraction.
    rng = np.random.default_rng(0)
    t = np.arange(500) * DT
    sway = 0.2 * np.sin(2 * np.pi * 0.5 * t) + 0.01 * rng.standard_normal(500)
    m = scenario_metrics(_qvel(sway), DT)
    assert m["band_20_25"] < 0.2
    assert m["vibration"] < 0.5


def test_band_is_the_nyquist_neighbourhood():
    lo, hi = NYQUIST_BAND_HZ
    assert lo < 25.0 < hi, "the 25 Hz bin itself must be inside the band"
    assert lo >= 20.0, "the band is a limit-cycle detector, not a vibration dup"


def test_grid_cells_collapse_the_zero_level():
    cells = grid_cells([0.0, 0.05, 0.1], ["x", "y", "z"])
    assert cells[0] == ("-", 0.0), "baseline first"
    assert cells.count(("-", 0.0)) == 1, "one baseline, not one per axis"
    assert len(cells) == 1 + 2 * 3
    assert ("y", 0.1) in cells


def test_grid_cells_without_baseline():
    assert grid_cells([0.1], ["z"]) == [("z", 0.1)]


@pytest.mark.parametrize(
    "axis,idx", [("x", 0), ("y", 1), ("z", 2)]
)
def test_bias_vector_is_one_axis(axis, idx):
    v = bias_vector(axis, 0.1)
    assert v[idx] == 0.1 and np.count_nonzero(v) == 1
