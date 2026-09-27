"""What crosses the link, as plain constants. No ROS import; tested model-free.

The robot runs foxglove_bridge on port 8765 (the service's foxglove:=true).
The relay on the DGX is a client of that bridge and the only door between
the robot's ROS graph and the DGX's: the DGX's own graph stays on its own
domain, localhost only, and nothing on it reaches the robot unless it is on
OUT below. A sim, a test node or a second brain on the DGX cannot touch the
robot by accident.

IN  (robot -> DGX): the robot's pose and goto's word, nothing heavy.
OUT (DGX -> robot): what the brain and the operator page produce, nothing
    that sets a joint or arms a motor.
"""

from __future__ import annotations

# topic -> ROS type (the bridge's schemaName). Odometry also becomes the
# DGX's odom -> base_link TF: relaying the robot's whole /tf (every joint at
# the joint-state rate) would cost the robot's bridge far more than one
# 25 Hz Odometry.
IN = {
    "/wojtek/odom": "nav_msgs/msg/Odometry",
    "/tf_static": "tf2_msgs/msg/TFMessage",          # base_link -> camera_link and the rest of the URDF's fixed joints
    "/wojtek/nav/status": "std_msgs/msg/String",     # goto's word: idle / turning / driving / blocked / reached
}

# The camera, IN as well, ONLY for the sim test (--in-camera): on the robot
# the pictures come from the camera Pi straight to the DGX, never through
# the robot's bridge (that would put the frames on the robot's cores 0,1).
IN_CAMERA = {
    "/camera/camera/color/image_raw/compressed": "sensor_msgs/msg/CompressedImage",
    "/camera/camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
    "/camera/camera/depth/image_rect_raw": "sensor_msgs/msg/Image",
    "/camera/camera/depth/camera_info": "sensor_msgs/msg/CameraInfo",
}

OUT = {
    "/wojtek/nav/goal": "geometry_msgs/msg/PoseStamped",   # the resolver's setpoints, the brain's approach steps
    "/wojtek/nav/cancel": "std_msgs/msg/Empty",            # STOP, a replaced task
    "/cmd_vel": "geometry_msgs/msg/Twist",                 # the brain's search turns, the page's STOP zeros
}

# Never relayed, whatever a command line says: what moves a joint, arms,
# or re-parents the robot's pose.
FORBIDDEN_OUT = (
    "/wojtek/joint_targets", "/wojtek/arm", "/wojtek/enable", "/tf", "/tf_static",
    "/joint_states", "/wojtek/odom", "/robot_description",
)

STATUS_TOPIC = "/wojtek/link/status"   # std_msgs/String JSON, latched, once a second

DEFAULT_URL = "ws://10.42.0.2:8765"     # the robot's static anchor on its own AP (ros/deploy.sh's RPI_HOST)


def check(inn=None, camera=None, out=None, forbidden=None) -> None:
    """The invariants the relay relies on. IN and OUT disjoint, so nothing
    the relay republishes on the DGX can be picked up and sent back; OUT
    free of anything forbidden. Arguments default to the module's lists."""
    inn = IN if inn is None else inn
    camera = IN_CAMERA if camera is None else camera
    out = OUT if out is None else out
    forbidden = FORBIDDEN_OUT if forbidden is None else forbidden
    loop = (set(inn) | set(camera)) & set(out)
    if loop:
        raise AssertionError(f"topics both IN and OUT would loop: {sorted(loop)}")
    bad = set(out) & set(forbidden)
    if bad:
        raise AssertionError(f"forbidden topics on OUT: {sorted(bad)}")


check()
