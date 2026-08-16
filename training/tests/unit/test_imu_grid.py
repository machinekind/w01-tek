"""Model-free tests of the IMU robustness grid. They check that the
spectral detector detects the failure mode it was built for, a 25 Hz
standing limit cycle at the 50 Hz control rate, and they check the grid's
cell layout, its env-build and lag enumeration, the bias vectors, and the
critical-gain bisection against a fake probe. Applying the bias in a
rollout uses the courses' pin mechanism and the env's info["gyro_bias"]
path, which test_gyro_bias.py covers.
"""

import numpy as np
import pytest

from wojtek_rl.imu_grid import (
    NYQUIST_BAND_HZ,
    bias_vector,
    bisect_threshold,
    critical_gain_str,
    env_builds,
    grid_cells,
    lag_levels,
    noise_vib_levels,
    scenario_metrics,
)

DT = 0.02  # the 50 Hz control rate the real limit cycle halved


def _qvel(signal):
    """Build a (T, 3) joint-velocity history with `signal` on one joint and
    the others quiet."""
    return np.stack([signal, 0.01 * signal, np.zeros_like(signal)], axis=1)


def test_limit_cycle_lands_in_the_band():
    # The real failure alternates at half the control rate, 25 Hz, which
    # is the last rfft bin at dt=0.02. The detector must call this one
    # loudly.
    t = np.arange(500)
    cycle = 0.5 * np.cos(np.pi * t)  # +A, -A, +A, ... = 25 Hz exactly
    m = scenario_metrics(_qvel(cycle), DT)
    assert m["band_20_25"] > 0.95
    assert m["vibration"] > 0.95
    # The absolute-scale guard reads a 0.5 rad/s cycle as real joint motion.
    assert 0.2 < m["qvel_rms"] < 0.5


def test_healthy_stand_stays_out_of_the_band():
    # Slow postural sway (0.5 Hz) plus mild broadband noise. Some power
    # above 5 Hz exists, but the near-Nyquist band stays a small fraction.
    rng = np.random.default_rng(0)
    t = np.arange(500) * DT
    sway = 0.2 * np.sin(2 * np.pi * 0.5 * t) + 0.01 * rng.standard_normal(500)
    m = scenario_metrics(_qvel(sway), DT)
    assert m["band_20_25"] < 0.2
    assert m["vibration"] < 0.5


def test_band_is_the_nyquist_neighbourhood():
    lo, hi = NYQUIST_BAND_HZ
    assert lo < 25.0 < hi, "the 25 Hz bin itself must be inside the band"
    assert lo >= 20.0, "the band must exclude ordinary vibration below 20 Hz"


def test_grid_cells_collapse_the_zero_level():
    cells = grid_cells([0.0, 0.05, 0.1], ["x", "y", "z"])
    assert cells[0] == ("-", 0.0), "baseline first"
    assert cells.count(("-", 0.0)) == 1, "exactly one baseline cell"
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


def test_env_builds_default_to_one_mechanism_each():
    levels = noise_vib_levels([0.4, 0.8], [0.1, 0.3])
    assert levels[0] == (None, None), "the untouched baseline runs first"
    assert levels == [
        (None, None), (0.4, None), (0.8, None), (None, 0.1), (None, 0.3)
    ]
    assert all(n is None or v is None for n, v in levels), (
        "without --cross-noise-vib no cell carries both faults"
    )


def test_crossing_adds_every_pair():
    plain = noise_vib_levels([0.4, 0.8], [0.1, 0.3])
    crossed = noise_vib_levels([0.4, 0.8], [0.1, 0.3], cross=True)
    assert crossed[0] == (None, None), "the baseline still runs first"
    assert crossed[:len(plain)] == plain, "the single-fault cells stay"
    assert set(crossed[len(plain):]) == {
        (0.4, 0.1), (0.4, 0.3), (0.8, 0.1), (0.8, 0.3)
    }


def test_baseline_is_the_whole_grid_when_nothing_is_swept():
    assert noise_vib_levels() == [(None, None)]
    assert noise_vib_levels(None, None, cross=True) == [(None, None)]
    # Crossing needs both lists. One list alone has nothing to pair with.
    assert noise_vib_levels([0.4], None, cross=True) == [(None, None),
                                                        (0.4, None)]


def test_one_env_build_per_noise_level():
    # The vibration gain is pinned per step, so every gain of a noise
    # level shares that level's build. Only the noise scale is baked in.
    builds = env_builds(noise_vib_levels([0.4], [0.1, 0.3], cross=True))
    assert builds == [(None, [None, 0.1, 0.3]), (0.4, [None, 0.1, 0.3])]


def test_env_builds_keep_the_baseline_first():
    assert env_builds([(None, None)]) == [(None, [None])]
    builds = env_builds(noise_vib_levels([0.8, 0.4], [0.3]))
    assert [n for n, _ in builds] == [None, 0.8, 0.4], "first-asked order"
    assert builds[0] == (None, [None, 0.3])


# A fake policy that goes unstable above a gain of 0.2 and shakes harder
# the higher the gain goes. bisect_threshold only ever reads the bool.
def _fake_probe(flip=0.2, calls=None):
    def probe(gain):
        if calls is not None:
            calls.append(gain)
        return gain > flip, round(0.01 + 10 * max(gain - flip, 0.0), 4)

    return probe


def test_bisection_brackets_the_flip():
    res = bisect_threshold(_fake_probe(0.2), 0.0, 1.0, tol=0.02)
    assert res["lo"] <= 0.2 <= res["hi"]
    assert res["hi"] - res["lo"] <= 0.02
    assert abs(res["critical_gain"] - 0.2) <= 0.02
    assert res["reason"] is None


def test_bisection_records_every_probe_in_order():
    calls = []
    res = bisect_threshold(_fake_probe(0.2, calls), 0.0, 1.0, tol=0.02)
    assert [p[0] for p in res["probes"]] == calls
    assert calls[:3] == [0.0, 1.0, 0.5], "the ends first, then the middle"
    assert res["probes"][0] == (0.0, False, 0.01)
    assert res["probes"][1][1] is True
    # Every probe carries the number that decided it.
    assert all(p[2] is not None for p in res["probes"])


def test_a_plain_bool_probe_is_enough():
    res = bisect_threshold(lambda g: g > 0.2, 0.0, 1.0, tol=0.05)
    assert abs(res["critical_gain"] - 0.2) <= 0.05
    assert res["probes"][0] == (0.0, False, None)


def test_bisection_reports_a_bracket_that_never_flips():
    # Already unstable at the low end, so the margin is below the bracket.
    low = bisect_threshold(_fake_probe(0.2), 0.5, 1.0, tol=0.02)
    assert low["critical_gain"] is None
    assert low["outcome"] == "below the bracket"
    assert critical_gain_str(low) == "below 0.5"
    assert len(low["probes"]) == 1, "no point probing on"
    # Stable all the way up, so the margin is above it. A gentle policy
    # that never snaps inside the bracket lands here, and "above 0.5" is
    # a real answer, so the number it measured up there comes with it.
    high = bisect_threshold(_fake_probe(2.0), 0.0, 0.5, tol=0.02)
    assert high["critical_gain"] is None
    assert high["outcome"] == "above the bracket"
    assert critical_gain_str(high) == "above 0.5"
    assert len(high["probes"]) == 2
    assert high["probes"][-1] == (0.5, False, 0.01)
    assert "qvel_rms 0.01" in high["reason"]


def test_bisection_rejects_an_empty_bracket():
    res = bisect_threshold(_fake_probe(), 1.0, 1.0, tol=0.02)
    assert res["critical_gain"] is None and res["probes"] == []
    assert critical_gain_str(res) == "none"


def test_a_tighter_tolerance_costs_more_probes():
    coarse = bisect_threshold(_fake_probe(0.2), 0.0, 1.0, tol=0.2)
    fine = bisect_threshold(_fake_probe(0.2), 0.0, 1.0, tol=0.001)
    assert coarse["hi"] - coarse["lo"] <= 0.2
    assert fine["hi"] - fine["lo"] <= 0.001
    assert len(fine["probes"]) > len(coarse["probes"])


def test_max_iters_stops_an_impossible_tolerance():
    res = bisect_threshold(_fake_probe(0.2), 0.0, 1.0, tol=0.0, max_iters=5)
    assert len(res["probes"]) == 2 + 5
    assert res["critical_gain"] is not None, "the bracket still stands"


def test_lag_levels_default_to_the_ideal_actuators():
    assert lag_levels() == [0.0]
    assert lag_levels([]) == [0.0]


def test_lag_levels_lead_with_zero_and_drop_repeats():
    assert lag_levels([0.005, 0.0, 0.01]) == [0.0, 0.005, 0.01]
    assert lag_levels([0.005, 0.005, 0.01]) == [0.005, 0.01]
    assert lag_levels(["0.005", 0.01]) == [0.005, 0.01], "strings parse"
