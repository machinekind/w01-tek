"""Unit tests for the numpy policy runtime and joint map (no ROS required).

Run: ros/build.sh test (or pytest with numpy+yaml on the host).

The shipped config is the springy policy (IMU-blind, 4-D command, vector
action scale); the v3-style legacy layout (gyro+gravity+gait clock, scalar
scale) is covered with a synthetic single-layer network whose output is
known in closed form.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from wojtek_policy.joint_map import JointMap  # noqa: E402
from wojtek_policy.policy import (  # noqa: E402
    WojtekPolicy,
    gravity_from_quat,
    height_anchor,
)

CONFIG = PKG / "config"
META = json.loads((CONFIG / "policy_meta.json").read_text())


@pytest.fixture
def policy():
    return WojtekPolicy(CONFIG / "policy.npz")


def _v3_style_policy(tmp_path, bias12=None, deploy=None, action_scale=0.5):
    """Synthetic single-layer policy in the wojtek_v3 obs layout.

    With a zero kernel the MLP output is tanh(bias) for any obs, so the
    action -- and therefore the target pipeline -- is known exactly.
    """
    obs_size = 53
    bias = np.zeros(24, np.float32)
    if bias12 is not None:
        bias[:12] = bias12
    np.savez(
        tmp_path / "policy.npz",
        norm_mean=np.zeros(obs_size, np.float32),
        norm_std=np.ones(obs_size, np.float32),
        hidden_0_kernel=np.zeros((obs_size, 24), np.float32),
        hidden_0_bias=bias,
    )
    meta = dict(META)
    meta["obs_size"] = obs_size
    meta["obs_layout"] = [
        "gyro:3", "gravity:3", "qpos-home:12", "qvel:12", "last_act:12",
        "command:3", "phase_cos_sin:8",
    ]
    meta["action_scale"] = action_scale
    meta.pop("deploy", None)
    if deploy is not None:
        meta["deploy"] = deploy
    (tmp_path / "policy_meta.json").write_text(json.dumps(meta))
    return WojtekPolicy(tmp_path / "policy.npz")


# -- springy (shipped config) -------------------------------------------------

def test_obs_assembly_matches_layout(policy):
    assert policy.uses_imu is False
    q = policy.home_ctrl + np.linspace(0.01, 0.12, 12)
    dq = np.linspace(-1.0, 1.0, 12)
    cmd = [0.3, -0.1, 0.2]
    policy.step(np.zeros(3), [0, 0, -1.0], q, dq, cmd)
    obs = policy.last_obs
    assert obs.shape == (META["obs_size"],)
    assert np.allclose(obs[0:12], q - policy.home_ctrl, atol=1e-6)
    assert np.allclose(obs[12:24], dq, atol=1e-6)
    assert np.allclose(obs[24:36], 0.0)  # last_act starts at zero
    assert np.allclose(obs[36:40], [0.3, -0.1, 0.2, 0.125], atol=1e-6)


def test_ignores_imu(policy):
    q, dq, cmd = policy.home_ctrl, np.zeros(12), [0.4, 0.0, 0.1]
    t1 = policy.step(np.zeros(3), [0, 0, -1.0], q, dq, cmd)
    policy.reset()
    t2 = policy.step([9.0, -9.0, 9.0], [1.0, 0.0, 0.0], q, dq, cmd)
    assert np.allclose(t1, t2)


def test_targets_respect_training_clamps(policy):
    rng = np.random.default_rng(0)
    for _ in range(100):
        q = policy.home_ctrl + rng.uniform(-0.5, 0.5, 12)
        dq = rng.uniform(-8, 8, 12)
        cmd = rng.uniform(policy.command_low, policy.command_high)
        t = policy.step(np.zeros(3), [0, 0, -1.0], q, dq, cmd)
        assert np.all(np.abs(t[0::3]) <= 0.44 + 1e-6)  # abduction clamp
        assert np.all(t[2::3] <= 3.15 + 1e-6)  # knee target cap
        assert np.all(t >= policy.target_low - 1e-6)


def test_anchor_is_height_shifted_home(policy):
    # dsecond(0.125) = 0.15 * (0.125-0.121)/(0.139-0.121) = 1/30
    d = 1.0 / 30.0
    expect = policy.home_ctrl + np.tile([0.0, 1.0, 2.0], 4) * d
    assert np.allclose(policy.anchor_ctrl, expect, atol=1e-6)
    # the table's 0.121 row is the home pose itself
    at_home = height_anchor(
        policy.home_ctrl, 0.121, policy.ctrl_low, policy.ctrl_high
    )
    assert np.allclose(at_home, policy.home_ctrl, atol=1e-6)


def test_command_box_from_meta(policy):
    assert np.allclose(policy.command_low, [-0.8, -0.5, -1.0])
    assert np.allclose(policy.command_high, [1.2, 0.5, 1.0])
    assert policy.command_width == 4
    assert policy.command_height_low < policy.command_height
    assert policy.command_height < policy.command_height_high


def test_height_command_moves_anchor_and_obs(policy):
    # A 4-D command re-anchors the stance to the commanded height (as the
    # training env does every step) and shows up verbatim in the obs.
    policy.step(np.zeros(3), [0, 0, -1.0], policy.home_ctrl, np.zeros(12),
                [0.1, 0.0, 0.0, 0.17])
    assert np.allclose(policy.last_obs[36:40], [0.1, 0.0, 0.0, 0.17], atol=1e-6)
    expect = height_anchor(
        policy.home_ctrl, 0.17, policy.ctrl_low, policy.ctrl_high
    )
    assert np.allclose(policy.anchor_ctrl, expect, atol=1e-6)
    # dropping back to a 3-D command falls back to the meta's fixed height
    # in both the obs padding and the anchor
    policy.step(np.zeros(3), [0, 0, -1.0], policy.home_ctrl, np.zeros(12),
                [0.1, 0.0, 0.0])
    assert np.allclose(policy.last_obs[36:40], [0.1, 0.0, 0.0, 0.125], atol=1e-6)
    default = height_anchor(
        policy.home_ctrl, 0.125, policy.ctrl_low, policy.ctrl_high
    )
    assert np.allclose(policy.anchor_ctrl, default, atol=1e-6)


def test_determinism(policy):
    seq_a = []
    policy.reset()
    for _ in range(10):
        seq_a.append(
            policy.step(np.zeros(3), [0, 0, -1.0], policy.home_ctrl,
                        np.zeros(12), [0.2, 0, 0])
        )
    policy.reset()
    for i in range(10):
        t = policy.step(np.zeros(3), [0, 0, -1.0], policy.home_ctrl,
                        np.zeros(12), [0.2, 0, 0])
        assert np.allclose(t, seq_a[i])


def test_last_action_feeds_back(policy):
    t1 = policy.step(np.zeros(3), [0, 0, -1.0], policy.home_ctrl,
                     np.zeros(12), [0.3, 0, 0])
    assert not np.allclose(policy.last_action, 0.0)
    t2 = policy.step(np.zeros(3), [0, 0, -1.0], policy.home_ctrl,
                     np.zeros(12), [0.3, 0, 0])
    assert not np.allclose(t1, t2)  # last_act changed -> different action
    policy.reset()
    assert np.allclose(policy.last_action, 0.0)


# -- legacy v3-style layout ---------------------------------------------------

def test_v3_layout_obs_and_targets(tmp_path):
    b = np.linspace(-0.4, 0.4, 12).astype(np.float32)
    pol = _v3_style_policy(tmp_path, bias12=b)
    assert pol.uses_imu is True
    assert np.allclose(pol.anchor_ctrl, pol.home_ctrl)

    gyro = [0.1, -0.2, 0.3]
    grav = [0.0, 0.1, -0.99]
    q = pol.home_ctrl + 0.05
    dq = np.full(12, 0.2)
    p0 = pol.phase.copy()
    t = pol.step(gyro, grav, q, dq, [0.3, 0.1, -0.2])

    obs = pol.last_obs
    assert obs.shape == (53,)
    assert np.allclose(obs[0:3], gyro, atol=1e-6)
    assert np.allclose(obs[3:6], grav, atol=1e-6)
    assert np.allclose(obs[6:18], 0.05, atol=1e-6)
    assert np.allclose(obs[30:42], 0.0)  # last_act
    assert np.allclose(obs[42:45], [0.3, 0.1, -0.2], atol=1e-6)
    assert np.allclose(obs[45:49], np.cos(p0), atol=1e-6)
    assert np.allclose(obs[49:53], np.sin(p0), atol=1e-6)
    assert not np.allclose(pol.phase, p0)  # gait clock advanced

    # zero kernel -> action = tanh(bias), targets known in closed form
    expect = np.clip(
        pol.home_ctrl + np.tanh(b) * 0.5, pol.ctrl_low, pol.ctrl_high
    )
    assert np.allclose(t, expect, atol=1e-6)


def test_no_deploy_block_falls_back_to_ctrlrange(tmp_path):
    # bias saturating the knees high: without a deploy block the only guard
    # below the model ctrlrange (5.8) is the clamp_knee singularity clip
    b = np.full(12, 5.0, np.float32)
    pol = _v3_style_policy(tmp_path, bias12=b)
    assert np.allclose(pol.target_high, pol.ctrl_high)
    assert np.allclose(pol.command_high, [0.6, 0.4, 0.7])
    pol.clamp_knee = True
    t = pol.step(np.zeros(3), [0, 0, -1.0], pol.home_ctrl, np.zeros(12),
                 [0.5, 0, 0])
    assert np.all(t[2::3] <= pol.knee_singularity + 1e-9)


# -- shared helpers -----------------------------------------------------------

def test_gravity_from_quat():
    # identity: upright -> gravity along -z in body frame
    assert np.allclose(gravity_from_quat(1, 0, 0, 0), [0, 0, -1])
    # 90 deg pitch about +y: gravity ends up along a body x axis
    g = gravity_from_quat(np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0)
    assert np.allclose(g, [1, 0, 0], atol=1e-9) or np.allclose(g, [-1, 0, 0], atol=1e-9)
    assert np.isclose(np.linalg.norm(g), 1.0)


def test_joint_map_roundtrip():
    jm = JointMap(CONFIG / "joint_map.yaml")
    names = jm.names()
    q = np.random.default_rng(0).uniform(-2, 2, len(names))
    back = jm.to_mjc(names, jm.to_urdf(names, q))
    assert np.allclose(back, q)
    dq = np.random.default_rng(1).uniform(-5, 5, len(names))
    assert np.allclose(jm.vel_to_mjc(names, jm.vel_to_urdf(names, dq)), dq)


def test_joint_map_home_within_urdf_limits_for_second_joint():
    """The trained home pose maps to finite URDF angles (sanity of offsets)."""
    jm = JointMap(CONFIG / "joint_map.yaml")
    policy = WojtekPolicy(CONFIG / "policy.npz")
    urdf_home = jm.to_urdf(policy.joint_names, policy.home_ctrl)
    assert np.all(np.isfinite(urdf_home))
