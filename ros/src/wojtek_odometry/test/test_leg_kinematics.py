"""Kinematics sanity: geometry against the URDF the robot actually uses.

The URDF comes from xacro at test time (wojtek_description must be in the
environment, which colcon test guarantees); no fixture copy that could go
stale. The reference numbers are physical invariants, not regression
snapshots: symmetric legs mirror each other, the standing foot sits below
the base at roughly the measured stand height, and the Jacobian matches
finite differences of the FK it was derived from.
"""

import subprocess

import numpy as np
import pytest

from wojtek_odometry.leg_kinematics import LEGS, LegKinematics

from wojtek_policy import poses
from wojtek_policy.joint_map import JointMap

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover
    get_package_share_directory = None


@pytest.fixture(scope="module")
def urdf():
    if get_package_share_directory is None:
        pytest.skip("no ament index (not in a ROS environment)")
    share = get_package_share_directory("wojtek_bringup")
    xacro = f"{share}/urdf/wojtek_real.urdf.xacro"
    result = subprocess.run(
        ["xacro", xacro], capture_output=True, text=True, check=True)
    return result.stdout


@pytest.fixture(scope="module")
def home_urdf():
    share = get_package_share_directory("wojtek_policy")
    jmap = JointMap(f"{share}/config/joint_map.yaml")
    q = jmap.to_urdf(list(poses.ACTUATOR_NAMES), poses.HOME_CTRL)
    return dict(zip(poses.ACTUATOR_NAMES, q))


def _home_q(home_urdf, leg):
    return np.array([home_urdf[f"{leg}_{j}_joint"]
                     for j in ("first", "second", "third")])


def test_standing_feet_below_base(urdf, home_urdf):
    # Standing height measured in sim: base at z ~= 0.118 m with a 0.046 m
    # foot sphere -- so the sphere centre sits ~72 mm under the base.
    for leg in LEGS:
        kin = LegKinematics(urdf, leg)
        foot = kin.foot_position(_home_q(home_urdf, leg))
        assert -0.12 < foot[2] < -0.04, (leg, foot)


def test_leg_symmetry(urdf, home_urdf):
    feet = {leg: LegKinematics(urdf, leg).foot_position(_home_q(home_urdf, leg))
            for leg in LEGS}
    # Left/right mirror in y, front/rear mirror in x.
    np.testing.assert_allclose(
        feet["front_left"] * [1, -1, 1], feet["front_right"], atol=1e-6)
    np.testing.assert_allclose(
        feet["rear_left"] * [1, -1, 1], feet["rear_right"], atol=1e-6)
    np.testing.assert_allclose(
        feet["front_left"] * [-1, 1, 1], feet["rear_left"], atol=1e-6)


def test_jacobian_matches_fk(urdf, home_urdf):
    rng = np.random.default_rng(7)
    kin = LegKinematics(urdf, "front_left")
    q0 = _home_q(home_urdf, "front_left")
    for _ in range(5):
        q = q0 + rng.uniform(-0.3, 0.3, 3)
        jac = kin.jacobian(q)
        dq = rng.uniform(-1e-4, 1e-4, 3)
        predicted = jac @ dq
        actual = kin.foot_position(q + dq) - kin.foot_position(q)
        np.testing.assert_allclose(predicted, actual, atol=1e-8)


def test_knee_column_includes_fourbar(urdf, home_urdf):
    # Freezing the passive polynomials would change the knee column; a pure
    # serial-chain Jacobian is the bug this guards against.
    kin = LegKinematics(urdf, "front_left")
    q = _home_q(home_urdf, "front_left")
    with_fourbar = kin.jacobian(q)[:, 2]
    frozen = dict(kin._passive)
    kin._passive = {k: [0.0] * (len(v) - 1) + [float(np.polyval(v, q[2]))]
                    for k, v in frozen.items()}
    without = kin.jacobian(q)[:, 2]
    kin._passive = frozen
    assert np.linalg.norm(with_fourbar - without) > 1e-3
