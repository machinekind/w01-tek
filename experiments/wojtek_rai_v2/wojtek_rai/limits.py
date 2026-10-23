"""The safety envelope of the agent, as plain constants and pure functions.

Everything the LLM can touch on the robot is enumerated here, and nothing
else. Tested model-free in tests/test_limits.py; no ROS import.
"""

from __future__ import annotations

import os

# --- ROS 2 names the experiment relies on (checked by `run.sh topics`) -------

NAV_COMMAND_TOPIC = "/wojtek/nav_command"          # std_msgs/String, text_commander
CMD_VEL_TOPIC = "/cmd_vel"                          # geometry_msgs/Twist, policy input
COLOR_IMAGE_RAW_TOPIC = "/camera/camera/color/image_raw"
COLOR_COMPRESSED_SUFFIX = "/compressed"
# The colour transport the agent reads. Raw is right next to the simulator;
# on the physical robot the raw stream (640x480 rgb8 at 30 Hz, ~27 MB/s)
# starves over WiFi (measured: 4 frames in 10 s) while the driver's JPEG
# transport (~1 MB/s) arrives at full rate. WOJTEK_RAI_COLOR_TRANSPORT=compressed
# switches every colour reader (camera tool, sidebar feed, find_objects).
COLOR_TRANSPORT = os.environ.get("WOJTEK_RAI_COLOR_TRANSPORT", "raw").strip().lower()
if COLOR_TRANSPORT not in ("raw", "compressed"):
    raise ValueError(f"WOJTEK_RAI_COLOR_TRANSPORT must be raw or compressed; got {COLOR_TRANSPORT!r}")
COLOR_IMAGE_TOPIC = COLOR_IMAGE_RAW_TOPIC + (COLOR_COMPRESSED_SUFFIX if COLOR_TRANSPORT == "compressed" else "")
DEPTH_IMAGE_TOPIC = "/camera/camera/depth/image_rect_raw"


def is_compressed_topic(topic: str) -> bool:
    """True for an image_transport JPEG/PNG topic (sensor_msgs/CompressedImage)."""
    return topic.endswith(COLOR_COMPRESSED_SUFFIX)

ODOM_FRAME = "odom"
BASE_FRAME = "base_link"

STAND_UP_SERVICE = "/wojtek/stand_up"               # std_srvs/Trigger
LIE_DOWN_SERVICE = "/wojtek/lie_down"               # std_srvs/Trigger

REQUIRED_TOPICS = (NAV_COMMAND_TOPIC, CMD_VEL_TOPIC, COLOR_IMAGE_TOPIC)
REQUIRED_SERVICES = (STAND_UP_SERVICE, LIE_DOWN_SERVICE)

# --- Nav2 (the wojtek_nav container) ----------------------------------------

MAP_FRAME = "map"
NAV_ACTION = "navigate_to_pose"                     # nav2_msgs/action/NavigateToPose
MAP_TOPIC = "/map"
# The place registry and workspace box default to the MuJoCo arena. A real-robot
# session overrides them: WOJTEK_RAI_PLACES=none (no registry, so go_to_place is
# not offered) and WOJTEK_RAI_WORKSPACE_M=3 (one room) until a real map and
# registry exist. Read at import time so the constants stay plain values.
PLACES_FILE = os.environ.get("WOJTEK_RAI_PLACES", "wojtek_rai/nav/config/places_sim.yaml")
WORKSPACE_HALF_M = float(os.environ.get("WOJTEK_RAI_WORKSPACE_M", "8"))
# Goals outside this box (map frame, metres) are refused before Nav2 sees them.
WORKSPACE_MIN = (-WORKSPACE_HALF_M, -WORKSPACE_HALF_M, -1.0)
WORKSPACE_MAX = (WORKSPACE_HALF_M, WORKSPACE_HALF_M, 1.0)
NAV_GOAL_TIMEOUT_S = 120.0

# --- perception (the wojtek_perception container) ---------------------------

DETECTION_SERVICE = "/detection"                    # rai_interfaces/srv/RAIGroundingDino
DEPTH_INFO_TOPIC = "/camera/camera/depth/camera_info"
COLOR_INFO_TOPIC = "/camera/camera/color/camera_info"
CAMERA_FORWARD_OFFSET_M = 0.32                      # camera_link x in base_link (URDF)
OBJECT_STANDOFF_M = 0.6

# What the agent may write: only the text-command topic (behind text_commander's
# dead-man) and the two posture services. Everything else that can move or arm
# the robot is forbidden outright, so even a generic RAI tool cannot reach it.
WRITABLE_TOPICS = (NAV_COMMAND_TOPIC,)
WRITABLE_SERVICES = (STAND_UP_SERVICE, LIE_DOWN_SERVICE)
WRITABLE_ACTIONS = ("navigate_to_pose", "/navigate_to_pose")
READABLE_TOPICS = (
    COLOR_IMAGE_TOPIC, DEPTH_IMAGE_TOPIC,
    "/camera/camera/depth/camera_info", "/camera/camera/color/camera_info",
)
FORBIDDEN = (
    CMD_VEL_TOPIC,
    "/wojtek/joint_targets",
    "/wojtek/arm",
    "/wojtek/enable",
    "/wojtek/zero",
    "/wojtek/reset",
    "/sim/reset",
)

# --- walking ----------------------------------------------------------------

# text_commander's vocabulary. Its speeds (v_forward=0.3 m/s, w_turn=0.5 rad/s)
# and dead-man (2 s) are its parameters, not ours.
WALK_DIRECTIONS = ("forward", "left", "right")
STOP_COMMAND = "stop"

# One walk tool call moves for at most this long; the LLM must look again
# before walking further. Re-publish period must stay well under the
# text_commander dead-man (2.0 s).
MOVE_MAX_SECONDS = 5.0
MOVE_MIN_SECONDS = 0.2
REPUBLISH_PERIOD_S = 0.5
TEXT_COMMANDER_DEADMAN_S = 2.0
TURN_MAX_SECONDS = 15.0
# text_commander's subscription on the command topic appears ~2 s after a
# publisher is created when it runs on the robot (WiFi AP, measured); the
# tools wait this long for it before deciding nobody is listening.
SUBSCRIBER_WAIT_S = 5.0
SUBSCRIBER_POLL_S = 0.1
TURN_STOP_LEAD_RAD = 0.25   # measured overshoot after the stop command in the sim

assert REPUBLISH_PERIOD_S < TEXT_COMMANDER_DEADMAN_S / 2


def validate_walk(direction: str, seconds: float) -> tuple[str, float]:
    """Normalise a walk request or raise ValueError with a message the LLM can act on."""
    d = str(direction).strip().lower()
    if d not in WALK_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {', '.join(WALK_DIRECTIONS)}; got {direction!r}"
        )
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        raise ValueError(f"seconds must be a number; got {seconds!r}") from None
    if not (MOVE_MIN_SECONDS <= s <= MOVE_MAX_SECONDS):
        raise ValueError(
            f"seconds must be between {MOVE_MIN_SECONDS} and {MOVE_MAX_SECONDS}; got {s}"
        )
    return d, s
