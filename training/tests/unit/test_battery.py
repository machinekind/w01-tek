import numpy as np
import pytest

from wojtek_rl.battery import (
    abduction_p95,
    band_power_fraction,
    battery_scenarios,
    diag_corr,
    duty_factor,
    lateral_corr,
    plant_step,
    saturation_fractions,
    tracking_error,
)


# -- battery_scenarios: hand-verify the exact windows the redesign asked for


def test_battery_scenarios_step_counts():
    scenarios = battery_scenarios()
    assert set(scenarios.keys()) == {
        "stand_to_trot_ramp", "turn", "strafe", "walk_to_stop",
        # added by the phase-C rotation/arc and height work;
        # absent from earlier baselines
        "arc", "height_step",
    }
    assert scenarios["stand_to_trot_ramp"][1] == 750
    assert scenarios["turn"][1] == 750
    assert scenarios["strafe"][1] == 600
    assert scenarios["walk_to_stop"][1] == 550
    assert scenarios["arc"][1] == 700
    assert scenarios["height_step"][1] == 800


def test_stand_to_trot_ramp_windows():
    cmd, _ = battery_scenarios()["stand_to_trot_ramp"]
    assert float(cmd(0)[0]) == 0.0
    assert float(cmd(99)[0]) == 0.0
    assert float(cmd(350)[0]) == (350 - 100) / 500  # mid-ramp
    assert float(cmd(600)[0]) == 1.0  # ramp saturates at the hold boundary
    assert float(cmd(749)[0]) == 1.0
    # height pinned throughout
    assert float(cmd(0)[3]) == 0.125
    assert float(cmd(749)[3]) == 0.125


# cmd_at returns jp.array (float32 by default in this project -- no x64
# config anywhere), so 0.4/0.8-scale values carry ~1e-7 rounding error
# against a plain Python float; compare with a tolerance loose enough to
# absorb that but tight enough to catch a real logic error.
def _close(a, b, tol=1e-5):
    return abs(a - b) < tol


def test_turn_windows():
    cmd, _ = battery_scenarios()["turn"]
    assert float(cmd(0)[2]) == 0.0  # stand: no wz
    assert float(cmd(99)[2]) == 0.0
    assert _close(float(cmd(100)[0]), 0.4)
    assert _close(float(cmd(100)[2]), -0.8)  # sweep starts at -0.8
    assert _close(float(cmd(499)[2]), 0.8)  # sweep ends at +0.8
    assert float(cmd(500)[0]) == 0.0  # pure-spin hold: vx drops to 0
    assert _close(float(cmd(500)[2]), 0.8)
    assert _close(float(cmd(749)[2]), 0.8)


def test_strafe_windows():
    cmd, _ = battery_scenarios()["strafe"]
    assert float(cmd(0)[1]) == 0.0
    assert _close(float(cmd(100)[1]), 0.4)
    assert _close(float(cmd(349)[1]), 0.4)
    assert _close(float(cmd(350)[1]), -0.4)
    assert _close(float(cmd(599)[1]), -0.4)


def test_walk_to_stop_windows():
    cmd, _ = battery_scenarios()["walk_to_stop"]
    assert float(cmd(0)[0]) == 0.5
    assert float(cmd(249)[0]) == 0.5
    assert float(cmd(250)[0]) == 0.0
    assert float(cmd(549)[0]) == 0.0


# -- lateral_corr / diag_corr -----------------------------------------------


def test_lateral_corr_pace_gait_high_diag_low():
    # Pace: same-side (left/right) pairs move together, LEGS order
    # RL, RR, FR, FL -- left = (0, 3), right = (1, 2).
    contacts = np.array([[1, 0, 0, 1], [0, 1, 1, 0]] * 4, dtype=bool)
    assert lateral_corr(contacts) > 0.99
    assert diag_corr(contacts) < -0.99


def test_lateral_corr_trot_gait_low_diag_high():
    # Trot: diagonal pairs move together, LEGS order RL, RR, FR, FL --
    # diagonal = (0, 2) and (1, 3).
    contacts = np.array([[1, 0, 1, 0], [0, 1, 0, 1]] * 4, dtype=bool)
    assert diag_corr(contacts) > 0.99
    assert lateral_corr(contacts) < -0.99


# -- duty_factor -------------------------------------------------------------


def test_duty_factor_known_values():
    contacts = np.array(
        [[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=bool
    )
    assert duty_factor(contacts) == [0.75, 0.5, 0.5, 0.25]


def test_duty_factor_skating_reads_as_one():
    contacts = np.ones((20, 4), dtype=bool)
    assert duty_factor(contacts) == [1.0, 1.0, 1.0, 1.0]


# -- abduction_p95 ------------------------------------------------------------


def test_abduction_p95_known_value():
    n = 20
    q = np.zeros((n, 12))
    vals = np.arange(n) * 0.01 - 0.05  # spans negative and positive
    for idx in (0, 3, 6, 9):
        q[:, idx] = vals
    # non-abduction columns carry a huge outlier; if the function pooled
    # the wrong columns this would blow the result up to ~1000.
    for idx in (1, 2, 4, 5, 7, 8, 10, 11):
        q[:, idx] = 1000.0
    expected = float(np.percentile(np.abs(np.tile(vals, 4)), 95))
    assert abduction_p95(q) == expected


# -- saturation_fractions -----------------------------------------------------


def test_saturation_fractions_known_values():
    n = 4
    f = np.zeros((n, 12))
    # abduction: half the (step, joint) samples over threshold
    for idx in (0, 3, 6, 9):
        f[:, idx] = [9.0, 9.0, 5.0, 5.0]
    # hip: always under
    for idx in (1, 4, 7, 10):
        f[:, idx] = 1.0
    # knee: always over
    for idx in (2, 5, 8, 11):
        f[:, idx] = 20.0
    r = saturation_fractions(f, torque_cap=10.0)  # threshold = 8.5
    assert r == {"abduction": 0.5, "hip": 0.0, "knee": 1.0}


def test_saturation_fractions_unknown_cap():
    f = np.ones((3, 12))
    r = saturation_fractions(f, torque_cap=0.0)
    assert r == {"abduction": None, "hip": None, "knee": None}


# -- tracking_error ------------------------------------------------------------


def test_tracking_error_excludes_settle_window():
    n, j = 60, 12
    ctrl_hist = np.zeros((n, j))
    qpos_hist = np.zeros((n, j))
    # first 50 steps: garbage error, must be excluded by the settle cut.
    ctrl_hist[:50] = 7.0
    # last 10 steps: constant 0.1 rad error everywhere.
    ctrl_hist[50:] = 0.1
    r = tracking_error(ctrl_hist, qpos_hist)
    assert r["rms"] == pytest.approx(0.1)
    assert r["p95"] == pytest.approx(0.1)


def test_tracking_error_short_rollout_returns_none():
    ctrl_hist = np.zeros((30, 12))
    qpos_hist = np.zeros((30, 12))
    assert tracking_error(ctrl_hist, qpos_hist) == {"rms": None, "p95": None}


def test_tracking_error_mixed_errors():
    n, j = 60, 12
    ctrl_hist = np.zeros((n, j))
    qpos_hist = np.zeros((n, j))
    # last window: half the joints (even columns) carry 0.2 rad error, the
    # other half (odd columns) carry none.
    ctrl_hist[50:, 0::2] = 0.2
    r = tracking_error(ctrl_hist, qpos_hist)
    assert r["rms"] == pytest.approx(np.sqrt(0.02))
    assert r["p95"] == pytest.approx(0.2)


# -- band_power_fraction ------------------------------------------------------


def test_band_power_fraction_in_band():
    dt = 0.01  # 100 Hz sampling
    n = 100  # 1 s window, 1 Hz bin resolution
    t = np.arange(n) * dt
    # 3 Hz sinusoid, exactly 3 cycles across the window -- no spectral
    # leakage, so essentially all AC power lands in the 2-4 Hz band.
    signal = np.sin(2 * np.pi * 3.0 * t)
    frac = band_power_fraction(signal, dt, 2.0, 4.0)
    assert abs(frac - 1.0) < 1e-6


def test_band_power_fraction_out_of_band():
    dt = 0.01
    n = 100
    t = np.arange(n) * dt
    signal = np.sin(2 * np.pi * 6.0 * t)  # 6 Hz, outside [2, 4)
    frac = band_power_fraction(signal, dt, 2.0, 4.0)
    assert frac < 1e-6


def test_band_power_fraction_guards_zero_variance():
    # A perfectly still signal has zero total power; the tiny-denominator
    # floor must keep this finite instead of a 0/0 blow-up.
    signal = np.full(50, 5.0)
    assert band_power_fraction(signal, 0.02, 2.0, 4.0) == 0.0


# -- plant_step ----------------------------------------------------------------


def test_plant_step_never_plants():
    # One foot (front_left, column 3) never touches down.
    contacts = np.zeros((30, 4), dtype=bool)
    contacts[:, :3] = True
    assert plant_step(contacts, switch_idx=2, hold=5) is None


def test_plant_step_brief_touch_then_lift_not_counted():
    contacts = np.array(
        [
            [1, 1, 1, 1],  # all down (run len 1)
            [1, 1, 1, 1],  # all down (run len 2 -- short of hold=3)
            [1, 1, 1, 0],  # a foot lifts, breaking the run
            [1, 1, 1, 1],
            [1, 1, 1, 1],  # run len 2 again
            [0, 1, 1, 1],  # breaks again
            [1, 1, 1, 1],
            [1, 1, 1, 1],  # ends on a run of 2 -- never reaches 3
        ],
        dtype=bool,
    )
    assert plant_step(contacts, switch_idx=0, hold=3) is None


def test_plant_step_skips_false_start_and_finds_real_plant():
    contacts = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],  # brief run of 2, insufficient for hold=3
            [1, 0, 1, 1],  # breaks
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],  # genuine run of 3 starting here (relative i=3)
            [1, 1, 1, 1],
        ],
        dtype=bool,
    )
    assert plant_step(contacts, switch_idx=0, hold=3) == 3


def test_plant_step_ignores_contact_before_switch_idx():
    contacts = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],  # a full plant BEFORE the switch -- must be ignored
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],  # genuine post-switch plant at relative i=3
            [1, 1, 1, 1],
        ],
        dtype=bool,
    )
    assert plant_step(contacts, switch_idx=4, hold=2) == 3
