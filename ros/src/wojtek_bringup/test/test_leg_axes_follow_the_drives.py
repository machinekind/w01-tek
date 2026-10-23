"""Every leg variant's URDF turns its hips and knees the way the drives do.

The MD80 interface writes URDF joint angles to the drives and the drives did
not change with the legs, so the sense of rotation of the first, second and
third joint of every leg is fixed by the stock URDF, whatever legs are bolted
on. A variant whose URDF takes its axes from its own model instead (the
v6.27 legs did, until 2026-10-23) turns some drives the wrong way: with the
identity map, four of them, at the first stand-up on the robot. The sign
between such a model and the drives belongs in the variant's joint map
(wojtek_policy/config, tested in wojtek_policy's test_robots.py).

Checked here: the axis of each actuated joint, expressed in base_link at
the zero configuration. The abduction and hip drives are the body's, so
every variant turns them like the stock URDF (the mounts are the same, so
the frames are comparable). The knee drives sit on the legs themselves, so
their sense is the variant's own, measured on the robot and pinned here:
the v6.37a knees all turn about +y of the body (the stock ones are mirrored
left/right; with that mirroring the right levers ran backwards, 2026-10-23).
"""

import math
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from ament_index_python.packages import get_package_share_directory

XACRO = f"{get_package_share_directory('wojtek_bringup')}/urdf/wojtek_real.urdf.xacro"
ACTUATED = ("first", "second", "third")
LEGS = ("rear_left", "rear_right", "front_right", "front_left")


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def _axes_in_base(legs):
    cmd = ["xacro", XACRO, f"legs:={legs}", "kp:=1", "kd:=1", "max_torque:=1"]
    root = ET.fromstring(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    joints = {}
    for j in root.iter("joint"):
        if j.find("parent") is None:
            continue  # ros2_control <joint> entries
        origin = j.find("origin")
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        axis = j.find("axis")
        joints[j.get("name")] = dict(
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            R=_rpy(*rpy), axis=None if axis is None else np.array([float(v) for v in axis.get("xyz").split()]),
        )
    by_child = {j["child"]: name for name, j in joints.items()}

    def rotation_to_base(link):
        R = np.eye(3)
        while link in by_child:
            j = joints[by_child[link]]
            R = j["R"] @ R
            link = j["parent"]
        return R

    return {
        name: np.round(rotation_to_base(j["parent"]) @ j["R"] @ j["axis"], 6)
        for name, j in joints.items() if j["axis"] is not None
    }


@pytest.fixture(scope="module")
def stock():
    return _axes_in_base("stock")


@pytest.mark.parametrize("legs", ["legs_v627"])
def test_body_drives_turn_like_the_stock_urdf(stock, legs):
    other = _axes_in_base(legs)
    for leg in LEGS:
        for joint in ("first", "second"):
            name = f"{leg}_{joint}_joint"
            assert np.allclose(other[name], stock[name]), (
                f"{name}: {legs} turns about {other[name]}, the drives about {stock[name]}"
            )


# Measured on the robot: the sense of each variant's knee drives in base_link.
KNEE_AXIS = {"legs_v627": [0, 1, 0]}


@pytest.mark.parametrize("legs", list(KNEE_AXIS))
def test_knee_drives_turn_as_measured_on_the_robot(legs):
    other = _axes_in_base(legs)
    for leg in LEGS:
        name = f"{leg}_third_joint"
        assert np.allclose(other[name], KNEE_AXIS[legs]), (
            f"{name}: {legs} turns about {other[name]}, the drive about {KNEE_AXIS[legs]}"
        )


def test_stock_axes_are_what_the_map_was_fitted_against(stock):
    # The stock URDF turns every left leg's hip and knee about +y and every
    # right leg's about -y (its joint_map.yaml's signs undo exactly that
    # towards the model's single convention). A change here would silently
    # move the drives' reference, so pin it.
    for leg in LEGS:
        y = 1.0 if leg.endswith("left") else -1.0
        assert np.allclose(stock[f"{leg}_second_joint"], [0, y, 0])
        assert np.allclose(stock[f"{leg}_third_joint"], [0, y, 0])
