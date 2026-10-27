"""The IMU mount and the sums that take it back out of a raw reading.

Run: pytest ros/src/wojtek_policy/test/test_imu_mount.py (no numpy, no ROS).

The IMU sits on the body turned a quarter turn, so the raw topic is in the
chip's axes. The operator displays turn it into the robot's roll, pitch,
heading and turn rates with into_base / attitude_in_base. These tests build
a reading the way the simulator does (the body's attitude with the mount
added on top, rates turned into the chip's axes) and check that the robot's
own numbers come back out.
"""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from wojtek_policy.imu_mount import (  # noqa: E402
    IMU_MOUNT_RPY,
    attitude_in_base,
    quat_conj,
    quat_from_rpy,
    quat_mul,
    rotate,
    rpy_from_quat,
)

DEG = math.pi / 180


def close(a, b, tol):
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


def test_identity_mount_passes_through():
    q = quat_from_rpy(0.2, -0.3, 1.1)
    w = (0.4, -0.5, 0.6)
    rpy, gyro = attitude_in_base(q, w, (0.0, 0.0, 0.0))
    assert close(rpy, rpy_from_quat(q), 1e-12)
    assert close(gyro, w, 1e-12)


def test_yawed_mount_recovers_the_robot_attitude():
    robot = (3 * DEG, 10 * DEG, 30 * DEG)
    q_wb = quat_from_rpy(*robot)
    q_bs = quat_from_rpy(0.0, 0.0, -1.5707963)
    # The way the simulator builds the topic's quaternion: the body (the
    # site) first, the mount on top of it (mju_mulQuat(site, mount)).
    q_ws = quat_mul(q_wb, q_bs)
    rpy, _ = attitude_in_base(q_ws, (0.0, 0.0, 0.0), IMU_MOUNT_RPY)
    assert close(rpy, robot, 1e-6)
    # Read raw, the robot's pitch shows up as a roll: the bug this fixes.
    raw = rpy_from_quat(q_ws)
    assert not close(raw, robot, 1 * DEG)
    assert abs(raw[0] - (-robot[1])) < 0.5 * DEG


def test_yawed_mount_recovers_the_robot_rates():
    q_bs = quat_from_rpy(0.0, 0.0, -1.5707963)
    robot = (0.1, 0.2, 0.3)
    # The simulator turns the body's rates into the chip's axes.
    w_s = rotate(quat_conj(q_bs), robot)
    assert close(w_s, (-0.2, 0.1, 0.3), 1e-6)
    _, gyro = attitude_in_base((0.0, 0.0, 0.0, 1.0), w_s, IMU_MOUNT_RPY)
    assert close(gyro, robot, 1e-9)


def test_unfused_zero_quaternion_reads_level():
    # The real driver sends an all-zero quaternion until its filter has
    # settled. It must stay a level reading, not turn into NaN or a tilt.
    rpy, gyro = attitude_in_base((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), IMU_MOUNT_RPY)
    assert rpy == (0.0, 0.0, 0.0)
    assert all(math.isfinite(v) for v in gyro)


def _floats(text):
    return [float(v) for v in text.split()]


def test_mount_matches_the_urdf_and_the_sim():
    src = Path(__file__).resolve().parents[2]
    body = src / "wojtek_description" / "urdf" / "body.urdf.xacro"
    sim = src / "wojtek_pc" / "urdf" / "wojtek_sim.urdf.xacro"
    if not body.is_file() or not sim.is_file():
        pytest.skip("the xacro sources are not next to this package (installed layout)")

    joint = next((j for j in ET.parse(body).getroot().iter("joint")
                  if j.get("name") == "imu_joint"), None)
    assert joint is not None, "no imu_joint in body.urdf.xacro"
    urdf_rpy = _floats(joint.find("origin").get("rpy"))
    assert close(urdf_rpy, IMU_MOUNT_RPY, 1e-9)

    param = next((p for p in ET.parse(sim).getroot().iter("param")
                  if p.get("name") == "imu_mount_rpy"), None)
    assert param is not None, "no imu_mount_rpy param in wojtek_sim.urdf.xacro"
    assert close(_floats(param.text), IMU_MOUNT_RPY, 1e-9)
