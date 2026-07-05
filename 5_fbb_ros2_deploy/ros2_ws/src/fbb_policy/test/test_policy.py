"""Unit tests for the numpy policy runtime and joint map (no ROS required).

Run: 5_fbb_ros2_deploy/run.sh test
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from fbb_policy.joint_map import JointMap  # noqa: E402
from fbb_policy.policy import FbbPolicy, gravity_from_quat  # noqa: E402

CONFIG = PKG / "config"


@pytest.fixture
def policy():
    return FbbPolicy(CONFIG / "policy.npz")


def test_home_obs_gives_bounded_targets(policy):
    targets = policy.step(
        gyro=np.zeros(3),
        gravity_body=[0.0, 0.0, -1.0],
        joint_pos=policy.home_ctrl,
        joint_vel=np.zeros(12),
        command=np.zeros(3),
    )
    assert targets.shape == (12,)
    assert np.all(targets >= policy.ctrl_low) and np.all(targets <= policy.ctrl_high)
    # standing near home with zero command should not command a huge jump
    assert np.max(np.abs(targets - policy.home_ctrl)) <= policy.action_scale


def test_phase_and_last_action_evolve(policy):
    p0 = policy.phase.copy()
    t1 = policy.step(np.zeros(3), [0, 0, -1.0], policy.home_ctrl, np.zeros(12), [0.3, 0, 0])
    assert not np.allclose(policy.phase, p0)
    assert not np.allclose(policy.last_action, 0.0)
    t2 = policy.step(np.zeros(3), [0, 0, -1.0], policy.home_ctrl, np.zeros(12), [0.3, 0, 0])
    assert not np.allclose(t1, t2)  # gait clock advanced -> different action
    policy.reset()
    assert np.allclose(policy.last_action, 0.0)


def test_determinism(policy):
    seq_a = []
    policy.reset()
    for _ in range(10):
        seq_a.append(
            policy.step(np.zeros(3), [0, 0, -1.0], policy.home_ctrl, np.zeros(12), [0.2, 0, 0])
        )
    policy.reset()
    for i in range(10):
        t = policy.step(np.zeros(3), [0, 0, -1.0], policy.home_ctrl, np.zeros(12), [0.2, 0, 0])
        assert np.allclose(t, seq_a[i])


def test_knee_clamp():
    clamped = FbbPolicy(CONFIG / "policy.npz", clamp_knee=True)
    for _ in range(50):
        t = clamped.step(np.zeros(3), [0, 0, -1.0], clamped.home_ctrl, np.zeros(12), [0.6, 0.4, 0.7])
        assert np.all(t[2::3] <= clamped.knee_singularity + 1e-9)


def test_gravity_from_quat():
    # identity: upright -> gravity along -z in body frame
    assert np.allclose(gravity_from_quat(1, 0, 0, 0), [0, 0, -1])
    # 90 deg pitch about +y: body x axis points down -> gravity along -x...
    # R = Ry(90): world -z maps to body frame as R^T [0,0,-1] = [1*? ...]
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
    policy = FbbPolicy(CONFIG / "policy.npz")
    urdf_home = jm.to_urdf(policy.joint_names, policy.home_ctrl)
    assert np.all(np.isfinite(urdf_home))
