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
