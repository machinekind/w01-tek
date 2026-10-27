"""Trick clips: envelope and contract checks per set of legs (no sim, pure numpy)."""

import numpy as np
import pytest

from wojtek_policy import poses, robots, tricks
from wojtek_policy.poses import HOME_CTRL

LEGS = sorted({p.training_robot for p in robots.PROFILES.values()})
CASES = [(legs, name) for legs in LEGS for name in tricks.for_robot(legs).names]


def test_every_set_of_legs_has_the_four_pad_tricks():
    # The gamepad's D-pad and the consoles call these four by name.
    for legs in LEGS:
        assert set(tricks.for_robot(legs).names) == {"bow", "sit", "paw_wave", "shake"}


@pytest.mark.parametrize("legs,name", CASES)
def test_clip_starts_and_ends_home(legs, name):
    home = poses.for_robot(legs).home_ctrl
    clips = tricks.for_robot(legs)
    assert np.allclose(clips.sample(name, 0.0), home, atol=1e-9)
    assert np.allclose(clips.sample(name, clips.duration(name)), home, atol=1e-9)


@pytest.mark.parametrize("legs,name", CASES)
def test_clip_targets_within_envelope(legs, name):
    p = poses.for_robot(legs)
    clips = tricks.for_robot(legs)
    ts = np.arange(0.0, clips.duration(name) + 0.02, 0.01)
    for t in ts:
        c = clips.sample(name, t)
        assert np.all(np.isfinite(c))
        assert np.all(c >= p.ctrl_low - 1e-9) and np.all(c <= p.ctrl_high + 1e-9)
        if legs == "wojtek":
            knees = c[2::3]
            assert knees.min() >= tricks.KNEE_LO - 1e-9
            assert knees.max() <= tricks.KNEE_HI + 1e-9
            # Abduction/hip targets stay well inside the +-pi ctrlrange.
            assert np.abs(c[0::3]).max() <= 1.5
            assert np.abs(c[1::3]).max() <= 2.5


@pytest.mark.parametrize("legs,name", CASES)
def test_clip_rate_is_bounded(legs, name):
    """No keyframe/oscillation demands a target jump PD would slam through."""
    clips = tricks.for_robot(legs)
    dt = 0.02  # the 50 Hz playback tick
    ts = np.arange(0.0, clips.duration(name), dt)
    prev = clips.sample(name, 0.0)
    worst = 0.0
    for t in ts[1:]:
        c = clips.sample(name, t)
        worst = max(worst, float(np.abs(c - prev).max()))
        prev = c
    assert worst < 0.12, f"max per-tick target step {worst:.3f} rad"


def test_sample_clamps_time():
    c = tricks.sample("bow", -1.0)
    assert np.allclose(c, HOME_CTRL, atol=1e-9)
    c = tricks.sample("bow", 999.0)
    assert np.allclose(c, HOME_CTRL, atol=1e-9)
    v2 = tricks.for_robot("legs_v627")
    home = poses.for_robot("legs_v627").home_ctrl
    assert np.allclose(v2.sample("sit", -1.0), home, atol=1e-9)
    assert np.allclose(v2.sample("sit", 999.0), home, atol=1e-9)


def test_v627_clips_keep_the_abduction_bounded():
    # bow/sit/shake are the M3 variant (abduction scaled to the stock foot's
    # lateral travel on legs twice as long); paw_wave is M2, the stock 0.35
    # rad lean, because the M3 lean tips the robot over at a 0.035 rad
    # zeroing error. Nothing goes beyond the stock clips' 0.35 rad.
    v2 = tricks.for_robot("legs_v627")
    for name in v2.names:
        ts = np.arange(0.0, v2.duration(name), 0.02)
        abd = np.abs(np.array([v2.sample(name, t)[0::3] for t in ts])).max()
        assert abd <= (0.36 if name == "paw_wave" else 0.18), (name, abd)


def test_v627_paw_wave_is_the_slow_variant():
    assert tricks.for_robot("legs_v627").duration("paw_wave") == pytest.approx(13.0)
