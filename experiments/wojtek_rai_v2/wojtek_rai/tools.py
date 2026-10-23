"""The tools the agent gets. Every actuator path goes through `limits`.

Walking is deliberately indirect: the tool publishes text commands to
`/wojtek/nav_command`, and the PC-side `text_commander` node (ros/src/
wojtek_teleop) turns them into `/cmd_vel` with a 2 s dead-man. If this
process dies mid-walk the robot stops on its own; the LLM never writes
`/cmd_vel`. That dead-man exists only while text_commander runs: the sim
launch starts it, the real-robot bringup does not, so on the physical robot
it must be started on the PC (`ros2 run wojtek_teleop text_commander`).
`_send` refuses to publish while nothing subscribes, so a missing
text_commander is an error, never a silent no-op.
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rai.communication.ros2 import ROS2Connector
from rai.communication.ros2.messages import ROS2Message
from rai.tools.ros2 import GetROS2ImageConfiguredTool, GetROS2TransformConfiguredTool
from rai.tools.ros2.base import BaseROS2Tool
from rai.tools.time import WaitForSecondsTool

from wojtek_rai import limits

STRING_MSG = "std_msgs/msg/String"
TRIGGER_SRV = "std_srvs/srv/Trigger"
SERVICE_TIMEOUT_S = 10.0

# Closed-loop turning needs an odometry yaw that moves. On the physical robot
# TF odom->base_link is a static identity (no odometry source), and over a
# flapping WiFi link the transform can simply stop updating; in both cases
# the yaw never changes and the loop would otherwise spin until
# TURN_MAX_SECONDS. Abort once no measurable yaw change has been seen for
# this long -- longer than the ~1 s the gait needs to start turning.
# Defined in limits.py when present; the fallback here keeps tools.py
# self-contained.
TURN_STALL_SECONDS = getattr(limits, "TURN_STALL_SECONDS", 3.0)
TURN_STALL_RAD = 0.05

# WOJTEK_RAI_ODOMETRY=0: the target has no odometry (the physical robot, where
# odom->base_link is a static identity). The closed-loop turn tool is not
# offered at all and the position tool says so, instead of promising the LLM
# a heading that never changes. Default on: the simulation publishes odometry.
ODOMETRY = os.environ.get("WOJTEK_RAI_ODOMETRY", "1").strip().lower() not in ("0", "false", "no")


def _wrap_angle(rad: float) -> float:
    """Wrap an angle difference into (-pi, pi]."""
    import math

    return math.atan2(math.sin(rad), math.cos(rad))


def _permissions() -> dict[str, list[str]]:
    return {
        "readable": list(limits.READABLE_TOPICS) + [limits.MAP_TOPIC],
        "writable": list(limits.WRITABLE_TOPICS)
        + list(limits.WRITABLE_SERVICES)
        + list(limits.WRITABLE_ACTIONS),
        "forbidden": list(limits.FORBIDDEN),
    }


class _NavCommandMixin:
    """Publish one text command to text_commander.

    Uses one persistent rclpy publisher on the connector's node. RAI's
    send_message() sets a publisher up per call, and a freshly created DDS
    publisher drops the first messages until the subscriber discovers it --
    with 0.5 s pulses that meant whole walks were lost.
    """

    connector: ROS2Connector

    def _publisher(self):
        pub = getattr(self, "_nav_pub", None)
        if pub is None:
            from std_msgs.msg import String

            pub = self.connector.node.create_publisher(String, limits.NAV_COMMAND_TOPIC, 10)
            object.__setattr__(self, "_nav_pub", pub)
        return pub

    def _send(self, command: str) -> None:
        if not self.is_writable(limits.NAV_COMMAND_TOPIC):  # type: ignore[attr-defined]
            raise ValueError(f"{limits.NAV_COMMAND_TOPIC} is not writable")
        from std_msgs.msg import String

        pub = self._publisher()
        # text_commander is the only consumer: the sim launch starts it, the
        # real bringup does not (it is started by hand, on the robot so that
        # its dead-man survives a WiFi drop). Without a subscriber the publish
        # goes nowhere; fail loudly instead. Over the AP its subscription is
        # matched ~2 s after the publisher exists, so look for it for a
        # bounded time rather than once.
        deadline = time.monotonic() + limits.SUBSCRIBER_WAIT_S
        while pub.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise ValueError(
                    f"nothing subscribes to {limits.NAV_COMMAND_TOPIC} after "
                    f"{limits.SUBSCRIBER_WAIT_S:.0f} s: text_commander is not running "
                    "(start it: ros2 run wojtek_teleop text_commander, on the robot)"
                )
            time.sleep(limits.SUBSCRIBER_POLL_S)
        pub.publish(String(data=command))


class WalkInput(BaseModel):
    direction: str = Field(
        ..., description="One of: forward, left (turn in place), right (turn in place)."
    )
    seconds: float = Field(
        ...,
        description=(
            f"How long to move, {limits.MOVE_MIN_SECONDS} to {limits.MOVE_MAX_SECONDS}"
            " seconds. Forward is about 0.3 m/s, turning about 0.5 rad/s."
        ),
    )


class WalkTool(_NavCommandMixin, BaseROS2Tool):
    name: str = "walk"
    description: str = (
        "Walk forward or turn in place for a bounded number of seconds, then stop. "
        "Blocks until the movement is over."
    )
    args_schema: Type[WalkInput] = WalkInput

    def _run(self, direction: str, seconds: float) -> str:
        d, s = limits.validate_walk(direction, seconds)
        deadline = time.monotonic() + s
        while True:
            self._send(d)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(limits.REPUBLISH_PERIOD_S, remaining))
        self._send(limits.STOP_COMMAND)
        return f"Walked {d} for {s:.1f} s; the robot is now stopped."


class TurnInput(BaseModel):
    degrees: float = Field(
        ..., description="Angle to turn in place, degrees; positive = left, negative = right; |value| <= 180."
    )


class TurnTool(_NavCommandMixin, BaseROS2Tool):
    """Closed-loop turn on odometry yaw. The gait needs ~1 s to start turning
    and stops slowly, so open-loop 'turn for N seconds' is unreliable; this
    pulses left/right until the measured yaw change reaches the target."""

    name: str = "turn"
    description: str = (
        "Turn in place by a given angle in degrees (positive left, negative right), "
        "measured on odometry, up to 180 degrees. Blocks until done (max 15 s). "
        "Aborts if the odometry yaw does not move (no odometry on the real robot): "
        "use walk left/right there."
    )
    args_schema: Type[TurnInput] = TurnInput

    def _yaw(self) -> float:
        import math

        tf = self.connector.get_transform(limits.ODOM_FRAME, limits.BASE_FRAME, timeout_sec=2.0)
        q = tf.transform.rotation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _run(self, degrees: float) -> str:
        import math

        target = max(-180.0, min(180.0, float(degrees)))
        if abs(target) < 2.0:
            return "Turn of less than 2 degrees skipped."
        direction = "left" if target > 0 else "right"
        y0 = self._yaw()
        turned = 0.0
        stalled = False
        t0 = time.monotonic()
        # The stop command is sent whatever ends the loop -- target reached,
        # timeout, stall, or _yaw() raising (TF lost over WiFi); otherwise the
        # robot would keep turning until text_commander's dead-man fires.
        try:
            # The gait keeps turning ~0.25 rad after the stop command; stop early.
            while abs(turned) < abs(math.radians(target)) - limits.TURN_STOP_LEAD_RAD:
                elapsed = time.monotonic() - t0
                if elapsed > limits.TURN_MAX_SECONDS:
                    break
                if elapsed > TURN_STALL_SECONDS and abs(turned) < TURN_STALL_RAD:
                    # Stall guard: see TURN_STALL_SECONDS. Stop rather than spin
                    # blind for the rest of TURN_MAX_SECONDS.
                    stalled = True
                    break
                self._send(direction)
                time.sleep(limits.REPUBLISH_PERIOD_S / 2)
                # One TF lookup per pulse: each carries a 2 s timeout, and two
                # of them could push the pulse period past the dead-man.
                turned = _wrap_angle(self._yaw() - y0)
        finally:
            self._send(limits.STOP_COMMAND)
        if stalled:
            return (
                f"Turn aborted: odometry yaw did not change in {TURN_STALL_SECONDS:.0f} s "
                "(no odometry on this robot, or TF lost). "
                "Use walk left/right with a short duration instead."
            )
        time.sleep(0.5)
        final = math.degrees(_wrap_angle(self._yaw() - y0))
        return f"Turned {final:+.0f} degrees (target {target:+.0f}); the robot is now stopped."


class StopTool(_NavCommandMixin, BaseROS2Tool):
    name: str = "stop"
    description: str = "Stop the robot immediately."

    def _run(self) -> str:
        self._send(limits.STOP_COMMAND)
        return "Stop command sent."


class _TriggerServiceTool(BaseROS2Tool):
    service_name: str = ""

    def _run(self) -> str:
        if not self.is_writable(self.service_name):
            raise ValueError(f"{self.service_name} is not writable")
        response = self.connector.service_call(
            ROS2Message(payload={}),
            self.service_name,
            msg_type=TRIGGER_SRV,
            timeout_sec=SERVICE_TIMEOUT_S,
        )
        payload: Any = response.payload
        success = getattr(payload, "success", None)
        message = getattr(payload, "message", "")
        if isinstance(payload, dict):
            success = payload.get("success")
            message = payload.get("message", "")
        return f"{self.service_name}: success={success} {message}".strip()


class StandUpTool(_TriggerServiceTool):
    name: str = "stand_up"
    description: str = (
        "Bring the robot to the standing posture. Only on explicit user request; "
        "refused while the robot is armed (it is normally already standing)."
    )
    service_name: str = limits.STAND_UP_SERVICE


class LieDownTool(_TriggerServiceTool):
    name: str = "lie_down"
    description: str = (
        "Lower the robot to the resting posture. Only on explicit user request; "
        "refused while the robot is armed."
    )
    service_name: str = limits.LIE_DOWN_SERVICE


class GetPoseTool(GetROS2TransformConfiguredTool):
    name: str = "get_robot_position"
    description: str = (
        "Current position and heading of the robot body in the odometry frame: "
        "translation x, y (metres) and rotation as a quaternion. The start pose is the origin."
    )


# What the position tool says of itself when the target has no odometry
# (build_tools(odometry=False)): the transform is a static identity, so the
# LLM must not read distance travelled or heading from it.
NO_ODOMETRY_POSE_DESCRIPTION = (
    "The odom -> base_link transform. This robot has NO odometry: the transform "
    "is static, so it never reflects movement and cannot be used to measure "
    "distance travelled or heading. Judge movement from the camera instead."
)


class GetCameraImageTool(GetROS2ImageConfiguredTool):
    name: str = "get_camera_image"
    description: str = "Take one colour image from the forward-facing camera."


def build_tools(
    connector: ROS2Connector,
    read_only: bool = False,
    nav: bool = True,
    odometry: bool = ODOMETRY,
) -> List[BaseTool]:
    """All tools, or -- read_only -- only the ones that cannot move the robot
    (camera, position, wait). read_only is the first-contact mode for the
    physical robot: the agent can look and report, nothing else. nav adds the
    Nav2 tools (needs the wojtek_nav container). odometry=False (the physical
    robot, WOJTEK_RAI_ODOMETRY=0) drops the closed-loop turn tool, every Nav2
    tool (navigate_to_pose, go_to_place, go_to_object, map tools) and marks
    the position tool as static; find_objects stays (bearing and distance
    need no map)."""
    perms = _permissions()
    actuators: List[BaseTool] = [] if read_only else [
        WalkTool(connector=connector, **perms),
        *([TurnTool(connector=connector, **perms)] if odometry else []),
        StopTool(connector=connector, **perms),
        StandUpTool(connector=connector, **perms),
        LieDownTool(connector=connector, **perms),
    ]
    if not read_only:
        from wojtek_rai.perception_tools import build_perception_tools

        # Nav2 needs odometry to track the robot: without it (the physical
        # robot before N4) no tool may send a goal, whatever `nav` says.
        navigation = nav and odometry
        if navigation:
            from wojtek_rai.nav_tools import build_nav_tools

            actuators += build_nav_tools(connector, perms)
        actuators += build_perception_tools(connector, perms, navigation=navigation)
    pose_description = {} if odometry else {"description": NO_ODOMETRY_POSE_DESCRIPTION}
    return actuators + [
        GetPoseTool(
            connector=connector,
            target_frame=limits.ODOM_FRAME,
            source_frame=limits.BASE_FRAME,
            timeout_sec=5.0,
            **pose_description,
            **perms,
        ),
        GetCameraImageTool(connector=connector, topic=limits.COLOR_IMAGE_TOPIC, **perms),
        WaitForSecondsTool(),
    ]
