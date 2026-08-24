"""Trick clips: envelope and contract checks (no sim, pure numpy)."""

import numpy as np
import pytest

from wojtek_policy import tricks
from wojtek_policy.poses import HOME_CTRL


@pytest.mark.parametrize("name", sorted(tricks.TRICKS))
def test_clip_starts_and_ends_home(name):
    assert np.allclose(tricks.sample(name, 0.0), HOME_CTRL, atol=1e-9)
    assert np.allclose(
        tricks.sample(name, tricks.duration(name)), HOME_CTRL, atol=1e-9
    )


@pytest.mark.parametrize("name", sorted(tricks.TRICKS))
def test_clip_targets_within_envelope(name):
    ts = np.arange(0.0, tricks.duration(name) + 0.02, 0.01)
    for t in ts:
        c = tricks.sample(name, t)
        assert np.all(np.isfinite(c))
        knees = c[2::3]
        assert knees.min() >= tricks.KNEE_LO - 1e-9
        assert knees.max() <= tricks.KNEE_HI + 1e-9
        # Abduction/hip targets stay well inside the +-pi ctrlrange.
        assert np.abs(c[0::3]).max() <= 1.5
        assert np.abs(c[1::3]).max() <= 2.5


@pytest.mark.parametrize("name", sorted(tricks.TRICKS))
def test_clip_rate_is_bounded(name):
    """No keyframe/oscillation demands a target jump PD would slam through."""
    dt = 0.02  # the 50 Hz playback tick
    ts = np.arange(0.0, tricks.duration(name), dt)
    prev = tricks.sample(name, 0.0)
    worst = 0.0
    for t in ts[1:]:
        c = tricks.sample(name, t)
        worst = max(worst, float(np.abs(c - prev).max()))
        prev = c
    assert worst < 0.12, f"max per-tick target step {worst:.3f} rad"


def test_sample_clamps_time():
    c = tricks.sample("bow", -1.0)
    assert np.allclose(c, HOME_CTRL, atol=1e-9)
    c = tricks.sample("bow", 999.0)
    assert np.allclose(c, HOME_CTRL, atol=1e-9)
