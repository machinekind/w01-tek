"""One camera, one driver: the robot launch refuses two RealSense drivers.

perception:=true (the full d435.yaml pipeline) and deck_camera:=true (the
panel's camera, which also carries the depth and the JPEG that nav:=true
and the VLM brain read) both start realsense2_camera_node as /camera/camera
on the robot's one D435. Together the second one finds the device taken, or
resets it under the first, and the camera works by luck. The launch refuses
the pair instead, before it fetches a policy. One of them alone (the robot's
service runs deck_camera:=true; a SLAM session runs perception:=true) and
any pair in the simulation, which has no deck camera node, still come up.

Needs the ROS launch packages, so it runs in the ROS container (colcon
test), not on a bare Mac.
"""

import pytest
from launch import LaunchContext

from wojtek_bringup import launch_common


def _context(**args):
    ctx = LaunchContext()
    ctx.launch_configurations.update(args)
    return ctx


def test_two_camera_drivers_are_refused_on_the_robot():
    with pytest.raises(ValueError, match="deck_camera"):
        launch_common._refuse_two_camera_drivers(
            _context(perception="true", deck_camera="true"), "real",
        )


@pytest.mark.parametrize(
    "args",
    [
        {"deck_camera": "true"},  # the robot's service today
        {"perception": "true", "deck_camera": "false"},  # a SLAM session
        {"perception": "true"},  # deck_camera never declared
        {},  # neither declared
    ],
)
def test_one_driver_comes_up_on_the_robot(args):
    launch_common._refuse_two_camera_drivers(_context(**args), "real")


def test_the_simulation_has_no_deck_camera_so_the_pair_is_fine():
    launch_common._refuse_two_camera_drivers(
        _context(perception="true", deck_camera="true"), "sim",
    )


def test_the_launch_refuses_before_it_loads_a_policy(monkeypatch):
    def no_policy(*args, **kwargs):
        raise AssertionError("the policy was loaded before the camera check")

    monkeypatch.setattr(launch_common, "load_policy", no_policy)
    with pytest.raises(ValueError, match="deck_camera"):
        launch_common._launch_setup(
            _context(perception="True", deck_camera="1", policy="org/policy"),
            with_rviz=False,
            hardware="real",
        )
