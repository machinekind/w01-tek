"""What may cross the link. Pure."""

import pytest

from wojtek_link import topics


def test_out_is_the_brain_and_the_page_and_nothing_that_moves_a_joint():
    assert set(topics.OUT) == {"/wojtek/nav/goal", "/wojtek/nav/cancel", "/cmd_vel"}
    for forbidden in ("/wojtek/joint_targets", "/wojtek/arm", "/tf", "/joint_states"):
        assert forbidden in topics.FORBIDDEN_OUT and forbidden not in topics.OUT


def test_in_is_small_and_the_camera_is_a_separate_switch():
    assert set(topics.IN) == {"/wojtek/odom", "/tf_static", "/wojtek/nav/status"}
    assert not set(topics.IN) & set(topics.IN_CAMERA)
    assert all(t.startswith("/camera/") for t in topics.IN_CAMERA)
    assert "/camera/camera/color/image_raw" not in topics.IN_CAMERA   # the JPEG, never the raw image


def test_the_module_s_lists_pass_their_own_check():
    topics.check()


def test_a_topic_both_in_and_out_is_refused():
    with pytest.raises(AssertionError, match="loop"):
        topics.check(inn={"/cmd_vel": "geometry_msgs/msg/Twist"})


def test_a_forbidden_topic_on_out_is_refused():
    with pytest.raises(AssertionError, match="forbidden"):
        topics.check(out={"/wojtek/joint_targets": "sensor_msgs/msg/JointState"})


def test_no_private_address_is_baked_in():
    # The default is the robot's own AP anchor (as in ros/deploy.sh); a
    # real robot address comes from WOJTEK_BRIDGE_URL at run time.
    assert topics.DEFAULT_URL == "ws://10.42.0.2:8765"


# -- the camera Pi's link -------------------------------------------------------

def test_the_camera_pi_carries_the_camera_and_its_static_frames_only():
    assert set(topics.CAMERA_PI) == set(topics.IN_CAMERA) | {"/tf_static"}
    assert not set(topics.CAMERA_PI) & set(topics.OUT)


def test_a_camera_pi_list_with_more_than_the_camera_is_refused():
    with pytest.raises(AssertionError, match="more than the camera"):
        topics.check(camera_pi={**topics.CAMERA_PI, "/wojtek/odom": "nav_msgs/msg/Odometry"})


class _T:
    """TransformStamped-shaped: what merge_static reads."""

    def __init__(self, parent, child):
        self.header = type("H", (), {"frame_id": parent})()
        self.child_frame_id = child


def test_the_camera_pi_adds_its_optical_chain_below_the_robot_s_mount():
    held = {}
    assert topics.merge_static(held, [_T("base_link", "camera_link"), _T("base_link", "imu_link")], "robot") == []
    refused = topics.merge_static(held, [
        _T("camera_link", "camera_color_frame"),
        _T("camera_color_frame", "camera_color_optical_frame"),
        _T("camera_link", "camera_depth_frame"),
        _T("camera_depth_frame", "camera_depth_optical_frame"),
    ], "camera")
    assert refused == []
    assert held["camera_link"][0] == "robot"
    assert held["camera_color_optical_frame"][0] == "camera"


def test_the_camera_pi_never_sets_the_mount_or_a_body_frame():
    held = {}
    topics.merge_static(held, [_T("base_link", "camera_link")], "robot")
    mount = held["camera_link"][1]
    refused = topics.merge_static(held, [
        _T("base_link", "camera_link"),           # a second mount: the robot's URDF owns it
        _T("camera_link", "base_link"),           # re-parenting the body
        _T("odom", "camera_link"),
        _T("base_link", "camera_extra"),          # a camera frame hung off the body
    ], "camera")
    assert sorted(refused) == ["base_link", "camera_extra", "camera_link", "camera_link"]
    assert held["camera_link"][1] is mount and "base_link" not in held


def test_the_robot_s_urdf_wins_a_frame_the_camera_pi_set_first():
    held = {}
    topics.merge_static(held, [_T("camera_link", "camera_color_frame")], "camera")
    assert topics.merge_static(held, [_T("camera_link", "camera_color_frame")], "robot") == []
    assert held["camera_color_frame"][0] == "robot"
    assert topics.merge_static(held, [_T("camera_link", "camera_color_frame")], "camera") == ["camera_color_frame"]


def test_no_camera_pi_address_is_baked_in():
    assert topics.CAMERA_URL_ENV == "WOJTEK_CAMERA_URL"
    assert not hasattr(topics, "DEFAULT_CAMERA_URL")
