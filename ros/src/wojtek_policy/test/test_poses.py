"""Named poses per set of legs: the stock set is what the stack had before
poses followed the robot profile, and the v6.27 set stays inside its model's
joint ranges (no sim here, pure numpy)."""

import numpy as np
import pytest

from wojtek_policy import poses, robots

LEGS = ("rear_left", "rear_right", "front_right", "front_left")


def test_every_profile_has_poses_and_they_are_well_formed():
    for profile in robots.PROFILES.values():
        p = poses.for_robot(profile.training_robot)
        for v in (p.home_ctrl, p.folded_ctrl, p.ctrl_low, p.ctrl_high):
            assert v.shape == (12,)
            assert np.all(np.isfinite(v))
        assert np.all(p.ctrl_low <= p.home_ctrl) and np.all(p.home_ctrl <= p.ctrl_high)
        assert np.all(p.ctrl_low <= p.folded_ctrl) and np.all(p.folded_ctrl <= p.ctrl_high)
        # Every passive polynomial names a leg that has a driving joint.
        for name in p.passive_from_driver:
            leg = name.rsplit("_", 2)[0]
            assert leg in LEGS, name
            assert f"{leg}_{p.passive_driver}" in poses.ACTUATOR_NAMES


def test_stock_poses_are_what_the_stack_had_before():
    p = poses.for_robot("wojtek")
    assert np.array_equal(p.home_ctrl, poses.HOME_CTRL)
    assert np.array_equal(
        p.folded_ctrl, poses.folded_ctrl(poses.ACTUATOR_NAMES, poses.FOLDED_KNEE_RAD)
    )
    assert p.passive_from_driver == poses.PASSIVE_FROM_KNEE
    assert p.passive_driver == "third_joint"
    assert p.passive_constant == {}
    # The knob real_io_node exposes still reaches the folded pose.
    assert poses.for_robot("wojtek", folded_knee_rad=0.5).folded_ctrl[2] == 0.5


def test_v627_poses_follow_the_mirrored_mounting():
    p = poses.for_robot("legs_v627")
    for v in (p.home_ctrl, p.folded_ctrl):
        rear, front = v[:6].reshape(2, 3), v[6:].reshape(2, 3)
        # Both rear legs alike, both front legs alike, abduction zero.
        assert np.array_equal(rear[0], rear[1]) and np.array_equal(front[0], front[1])
        assert rear[0, 0] == 0.0 and front[0, 0] == 0.0
        # Rear hip and lever run negative, front positive.
        assert np.all(rear[0, 1:] < 0) and np.all(front[0, 1:] > 0)
    # The lying pose: shins flat on the floor, body on 35 mm standoffs ->
    # hips at +-2.903 (inside the envelope, past the exported model's
    # 2.695 range), levers on their folding limit, abduction zero.
    f = p.folded_ctrl
    assert np.allclose(np.abs(f[1::3]), 2.903, atol=1e-9)
    assert np.all(f[1::3] >= p.ctrl_low[1::3]) and np.all(f[1::3] <= p.ctrl_high[1::3])
    assert np.allclose(np.abs(f[2::3]), 0.244346, atol=1e-9)
    assert np.all(f[0::3] == 0.0)
    # The free foot joints are published at zero.
    assert set(p.passive_constant) == {f"{leg}_foot_joint" for leg in LEGS}


def test_v627_passive_table_matches_the_models_home_keyframe():
    # fourth/fifth at the home keyframe of the built legs_v627 model
    # (wojtek_description/mujoco/legs_v627/scene_mjx.xml), read on 2026-10-22.
    # The table is fitted on the exact closure; the model's keyframe and its
    # settled stance leave the soft closure 0.2 mm open, which the short
    # rod turns into up to 8 mrad at the fifth joint -- hence the tolerance.
    expected = {
        "rear_left": (-1.9316, -2.17111, -1.98844),
        "rear_right": (-1.9316, 2.17111, 1.98844),
        "front_right": (1.90961, 2.15082, 1.97585),
        "front_left": (1.90961, -2.15082, -1.97585),
    }
    p = poses.for_robot("legs_v627")
    for leg, (third, fourth, fifth) in expected.items():
        assert np.polyval(p.passive_from_driver[f"{leg}_fourth_joint"], third) == pytest.approx(fourth, abs=1e-2)
        assert np.polyval(p.passive_from_driver[f"{leg}_fifth_joint"], third) == pytest.approx(fifth, abs=1e-2)


def test_v627_passive_table_is_smooth_over_the_lever_range():
    p = poses.for_robot("legs_v627")
    for leg in LEGS:
        i = poses.ACTUATOR_NAMES.index(f"{leg}_third_joint")
        grid = np.linspace(p.ctrl_low[i], p.ctrl_high[i], 500)
        for joint in ("fourth", "fifth"):
            q = np.polyval(p.passive_from_driver[f"{leg}_{joint}_joint"], grid)
            assert np.all(np.isfinite(q))
            # No wrap or kink: the passive angle changes less than 20 mrad
            # per 5 mrad step of the lever over the whole range.
            assert np.abs(np.diff(q)).max() < 0.02


def test_unknown_legs_are_refused():
    with pytest.raises(ValueError, match="no poses"):
        poses.for_robot("no_such_legs")
