"""The real robot's URDF carries the D435's mount and nothing below it.

The costmap, the pixel resolver and the VLM brain's bearing all transform
camera points into odom, which needs base_link -> camera_link. The
realsense2_camera driver publishes camera_link -> *_optical itself but
cannot know where the camera sits on the body; that edge is the URDF's
(body.urdf.xacro with_camera_mount), so the robot_state_publisher gives it
in every mode that runs a camera. The optical chain must NOT be in the real
URDF: two publishers of one TF edge fight (the sim URDF, with_camera, has
the whole chain because it renders its own camera).

The numbers are the design mount, the same as wojtek_pc/camera_spec.py and
wojtek_perception_bringup/config/extrinsics.yaml (checked here against the
yaml; wojtek_pc's test_camera_model_consistency checks the spec).
"""

import math
import subprocess
import xml.etree.ElementTree as ET

import yaml
from ament_index_python.packages import get_package_share_directory

XACRO = f"{get_package_share_directory('wojtek_bringup')}/urdf/wojtek_real.urdf.xacro"
EXTRINSICS = (
    f"{get_package_share_directory('wojtek_perception_bringup')}/config/extrinsics.yaml"
)


def _real_urdf(**args):
    cmd = ["xacro", XACRO] + [f"{k}:={v}" for k, v in args.items()]
    return ET.fromstring(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)


def _joint(root, name):
    for joint in root.iter("joint"):
        if joint.get("name") == name:
            return joint
    return None


def test_the_real_urdf_has_the_camera_mount_at_the_design_pose():
    root = _real_urdf()
    joint = _joint(root, "camera_joint")
    assert joint is not None, "no camera_joint: the robot has no base_link -> camera_link"
    assert joint.find("parent").get("link") == "base_link"
    assert joint.find("child").get("link") == "camera_link"
    origin = joint.find("origin")
    xyz = [float(t) for t in origin.get("xyz").split()]
    rpy = [float(t) for t in origin.get("rpy").split()]
    assert xyz == [0.32, 0.0, 0.07]
    assert rpy[0] == 0.0 and rpy[2] == 0.0
    assert rpy[1] == math.radians(15.0) or abs(rpy[1] - math.radians(15.0)) < 1e-6


def test_the_real_urdf_leaves_the_optical_chain_to_the_driver():
    root = _real_urdf()
    links = {link.get("name") for link in root.iter("link")}
    assert "camera_link" in links
    for driver_owned in (
        "camera_depth_frame", "camera_depth_optical_frame",
        "camera_color_frame", "camera_color_optical_frame",
    ):
        assert driver_owned not in links, f"{driver_owned} would be published twice"


def test_the_mount_survives_the_mock_hardware_swap():
    # ros/hw_tests/perf runs the real graph over mock drivers; the camera
    # frames must be the same there, or a perf run measures a different TF.
    assert _joint(_real_urdf(mock_hw="true"), "camera_joint") is not None


def test_extrinsics_yaml_says_the_same_as_the_urdf():
    with open(EXTRINSICS) as fh:
        tf = yaml.safe_load(fh)["static_transform"]
    assert (tf["parent_frame"], tf["child_frame"]) == ("base_link", "camera_link")
    origin = _joint(_real_urdf(), "camera_joint").find("origin")
    xyz = [float(t) for t in origin.get("xyz").split()]
    rpy = [float(t) for t in origin.get("rpy").split()]
    assert [tf["x"], tf["y"], tf["z"]] == xyz
    assert [tf["roll"], tf["yaw"]] == [rpy[0], rpy[2]]
    assert abs(tf["pitch"] - rpy[1]) < 1e-6
