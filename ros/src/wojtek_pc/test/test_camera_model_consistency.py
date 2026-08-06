"""Cross-check: URDF camera frames == camera_spec == MJCF injection.

The classic failure this prevents: someone moves the camera in one place
(the render pose, the TF chain, the spec constants) and the depth cloud
lands in the wrong spot in RViz with no error anywhere.

The URDF is parsed textually (ElementTree on the xacro file) rather than
expanded through xacro, so this runs with no ROS and no ament index; the
xacro constructs used in the camera block are plain attributes plus
``${-pi/2}``-style literals handled below.
"""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from wojtek_pc import camera_spec  # noqa: E402

BODY_XACRO = (
    PKG.parent / "wojtek_description" / "urdf" / "body.urdf.xacro"
)


def _xacro_float(token):
    """Evaluate the tiny subset of xacro expressions the camera block uses."""
    token = token.strip()
    if token.startswith("${") and token.endswith("}"):
        return float(eval(token[2:-1], {"pi": math.pi}))  # noqa: S307
    return float(token)


def _joint(root, name):
    for joint in root.iter("joint"):
        if joint.get("name") == name:
            return joint
    raise AssertionError(f"joint {name!r} not found in body.urdf.xacro")


def _origin(joint):
    origin = joint.find("origin")
    xyz = [_xacro_float(t) for t in origin.get("xyz", "0 0 0").split()]
    rpy = [_xacro_float(t) for t in origin.get("rpy", "0 0 0").split()]
    return np.array(xyz), np.array(rpy)


def _rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (
        np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y),
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


@pytest.fixture(scope="module")
def urdf_root():
    return ET.parse(BODY_XACRO).getroot()


class TestMountAgreement:
    def test_camera_link_translation_matches_spec(self, urdf_root):
        xyz, _ = _origin(_joint(urdf_root, "camera_joint"))
        assert xyz == pytest.approx(np.array(camera_spec.MOUNT_XYZ), abs=1e-9)

    def test_camera_link_pitch_matches_spec(self, urdf_root):
        _, rpy = _origin(_joint(urdf_root, "camera_joint"))
        assert rpy[0] == 0.0 and rpy[2] == 0.0
        assert rpy[1] == pytest.approx(camera_spec.MOUNT_PITCH_RAD, abs=1e-6)

    def test_depth_frames_are_identity_until_optical(self, urdf_root):
        xyz, rpy = _origin(_joint(urdf_root, "camera_depth_joint"))
        assert xyz == pytest.approx(np.zeros(3))
        assert rpy == pytest.approx(np.zeros(3))

    def test_optical_rotation_is_the_standard_one(self, urdf_root):
        _, rpy = _origin(_joint(urdf_root, "camera_depth_optical_joint"))
        assert rpy == pytest.approx(
            np.array([-math.pi / 2, 0.0, -math.pi / 2]), abs=1e-9
        )

    def test_color_chain_matches_depth_chain(self, urdf_root):
        for depth, color in (
            ("camera_depth_joint", "camera_color_joint"),
            ("camera_depth_optical_joint", "camera_color_optical_joint"),
        ):
            dx, dr = _origin(_joint(urdf_root, depth))
            cx, cr = _origin(_joint(urdf_root, color))
            assert cx == pytest.approx(dx)
            assert cr == pytest.approx(dr)


class TestOpticalAxisAgreement:
    def test_urdf_optical_z_equals_mjcf_view_direction(self, urdf_root):
        """The URDF optical frame's z axis (ROS: points out of the lens)
        must equal the MJCF camera's view direction (-z of the camera
        frame built from XYAXES), both expressed in base_link."""
        _, mount_rpy = _origin(_joint(urdf_root, "camera_joint"))
        _, opt_rpy = _origin(_joint(urdf_root, "camera_depth_optical_joint"))
        r_base_optical = _rpy_matrix(mount_rpy) @ _rpy_matrix(opt_rpy)
        urdf_forward = r_base_optical[:, 2]

        x = np.array(camera_spec.XYAXES[:3])
        y = np.array(camera_spec.XYAXES[3:])
        mjcf_forward = -np.cross(x, y)

        # 1e-6: the URDF carries the pitch as a 7-digit literal.
        assert urdf_forward == pytest.approx(mjcf_forward, abs=1e-6)

    def test_urdf_optical_x_is_camera_right(self, urdf_root):
        _, mount_rpy = _origin(_joint(urdf_root, "camera_joint"))
        _, opt_rpy = _origin(_joint(urdf_root, "camera_depth_optical_joint"))
        r_base_optical = _rpy_matrix(mount_rpy) @ _rpy_matrix(opt_rpy)
        assert r_base_optical[:, 0] == pytest.approx(
            np.array(camera_spec.XYAXES[:3]), abs=1e-9
        )
