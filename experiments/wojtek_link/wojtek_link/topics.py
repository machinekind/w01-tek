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

# The camera. On the robot it comes from the camera Pi's own read-only
# bridge straight to the DGX (--camera-url, CAMERA_PI below), never through
# the robot's bridge: that would put the frames on the robot's cores 0,1.
# Through the robot's bridge ONLY for the sim test (--in-camera).
IN_CAMERA = {
    "/camera/camera/color/image_raw/compressed": "sensor_msgs/msg/CompressedImage",
    "/camera/camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
    "/camera/camera/depth/image_rect_raw": "sensor_msgs/msg/Image",
    "/camera/camera/depth/camera_info": "sensor_msgs/msg/CameraInfo",
}

# What the relay reads from the camera Pi: the pictures and the driver's
# own static frames (camera_link -> the optical chain). Nothing goes back:
# the camera Pi's bridge advertises no clientPublish (camera_pi/bridge.yaml).
CAMERA_PI = {**IN_CAMERA, "/tf_static": "tf2_msgs/msg/TFMessage"}

# base_link -> camera_link is the robot's URDF (the mount, fork PR #9): the
# camera Pi's driver adds only what hangs below it.
MOUNT_FRAME = "camera_link"
CAMERA_FRAME_PREFIX = "camera_"

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
BRIDGE_URL_ENV = "WOJTEK_BRIDGE_URL"    # the robot's bridge, at run time
CAMERA_URL_ENV = "WOJTEK_CAMERA_URL"    # the camera Pi's bridge, at run time (no default: off unless given)


def check(inn=None, camera=None, out=None, forbidden=None, camera_pi=None) -> None:
    """The invariants the relay relies on. IN and OUT disjoint, so nothing
    the relay republishes on the DGX can be picked up and sent back; OUT
    free of anything forbidden; the camera Pi's list only pictures and its
    static frames. Arguments default to the module's lists."""
    inn = IN if inn is None else inn
    camera = IN_CAMERA if camera is None else camera
    out = OUT if out is None else out
    forbidden = FORBIDDEN_OUT if forbidden is None else forbidden
    camera_pi = CAMERA_PI if camera_pi is None else camera_pi
    loop = (set(inn) | set(camera) | set(camera_pi)) & set(out)
    if loop:
        raise AssertionError(f"topics both IN and OUT would loop: {sorted(loop)}")
    bad = set(out) & set(forbidden)
    if bad:
        raise AssertionError(f"forbidden topics on OUT: {sorted(bad)}")
    extra = set(camera_pi) - set(camera) - {"/tf_static"}
    if extra:
        raise AssertionError(f"the camera Pi's list carries more than the camera: {sorted(extra)}")


def camera_frame_ok(parent: str, child: str) -> bool:
    """May the camera Pi's /tf_static set this edge? Only inside its own
    tree below the mount: never the mount itself, never a body frame."""
    return (parent.startswith(CAMERA_FRAME_PREFIX) and child.startswith(CAMERA_FRAME_PREFIX)
            and child != MOUNT_FRAME)


def merge_static(held: dict, transforms, source: str) -> list:
    """Merge one /tf_static message into `held` (child frame -> (source,
    transform)); returns the child frames refused. The robot's URDF owns
    every frame it names and may replace anything; the camera Pi may only
    add its optical chain (camera_frame_ok) and never replaces a frame the
    robot holds. `transforms` are TransformStamped-like (header.frame_id,
    child_frame_id)."""
    refused = []
    for t in transforms:
        child = t.child_frame_id
        if source != "robot":
            owner = held.get(child, (None, None))[0]
            if owner == "robot" or not camera_frame_ok(t.header.frame_id, child):
                refused.append(child)
                continue
        held[child] = (source, t)
    return refused


check()
