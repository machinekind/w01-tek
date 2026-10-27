"""gait_metrics: swing apex + touchdown softness KPIs, model-free."""

import math

import numpy as np

from wojtek_rl.courses.scoring import gait_metrics


def _rec_from_swing(apex, td_speed, steps=10, feet=4):
    """One triangular swing per foot: rise to `apex`, descend arriving at
    the ground with `td_speed` on the last airborne step."""
    up = np.linspace(0.0, apex, steps)
    down = np.linspace(apex, 0.001, steps)
    z = np.concatenate([[0.0], up, down, [0.0, 0.0]])
    vz = np.gradient(z)
    last_airborne = np.nonzero(z > 0.005)[0][-1]
    vz[last_airborne] = -td_speed  # pre-contact downward speed
    clear = np.tile(z[:, None], (1, feet))
    foot_vz = np.tile(vz[:, None], (1, feet))
    return {"foot_clear": clear, "foot_vz": foot_vz}


def test_apex_measured_per_swing():
    rec = _rec_from_swing(apex=0.05, td_speed=0.3)
    g = gait_metrics(rec)
    assert abs(g["swing_apex_med_m"] - 0.05) < 1e-6
    assert g["swings"] == 4  # one per foot


def test_touchdown_softness_uses_freefall_reference():
    apex, td = 0.05, 0.3
    g = gait_metrics(_rec_from_swing(apex, td))
    v_ff = math.sqrt(2 * 9.81 * apex)
    assert abs(g["touchdown_v_med"] - td) < 1e-6
    assert abs(g["touchdown_softness_med"] - td / v_ff) < 1e-3
    # a brick (free-fall touchdown) scores ~1.0
    brick = gait_metrics(_rec_from_swing(apex, v_ff))
    assert abs(brick["touchdown_softness_med"] - 1.0) < 1e-3


def test_blips_and_ground_noise_ignored():
    clear = np.zeros((20, 4))
    clear[5, 0] = 0.02  # 1-step blip
    clear[10:12, 1] = 0.003  # below the 5 mm band
    rec = {"foot_clear": clear, "foot_vz": np.zeros((20, 4))}
    assert gait_metrics(rec) == {}


def test_truncated_final_swing_not_counted():
    clear = np.zeros((10, 4))
    clear[6:, :] = 0.04  # airborne at the end of the record (no touchdown)
    rec = {"foot_clear": clear, "foot_vz": np.zeros((10, 4))}
    assert gait_metrics(rec) == {}
