"""Opt-in ROS command arbiter. The standalone output is isolated by default."""

import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty, String
from std_srvs.srv import SetBool

from wojtek_teleop.command_mux import CommandMux, SOURCES


CANCEL_QUIET_S = 0.5


class CommandMuxNode(Node):
    def __init__(self):
        super().__init__("wojtek_command_mux")
        timeout = float(self.declare_parameter("source_timeout_s", 0.5).value)
        hz = float(self.declare_parameter("publish_hz", 20.0).value)
        if not math.isfinite(hz) or hz <= 0:
            raise ValueError("publish_hz must be finite and positive")
        self._mux = CommandMux(timeout)
        self._last_status = None
        self._last_cancel_at = float("-inf")
        commands = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub = self.create_publisher(Twist, "/wojtek/cmd_vel/selected", commands)
        self._cancel = self.create_publisher(Empty, "/wojtek/nav/cancel", commands)
        self._status = self.create_publisher(String, "/wojtek/command_mux/status", status_qos)
        self._subscriptions = [
            self.create_subscription(
                Twist, f"/wojtek/cmd_vel/{source}",
                lambda message, source=source: self._on_command(source, message),
                commands,
            )
            for source in SOURCES
        ]
        self.create_subscription(Empty, "/wojtek/nav/cancel", self._on_cancel, commands)
        self.create_service(SetBool, "/wojtek/autonomy_enable", self._on_enable)
        self.create_timer(1.0 / hz, self._tick)
        self._publish_status()
        self.get_logger().info(
            "command mux: autonomy inhibited; remap every writer and the consumer "
            "before connecting this output to robot control"
        )

    def _on_command(self, source, message):
        self._mux.receive(
            source,
            (message.linear.x, message.linear.y, message.linear.z,
             message.angular.x, message.angular.y, message.angular.z),
            time.monotonic(),
        )

    def _on_cancel(self, _message):
        # Our own cancel returns here too. It inhibits autonomy idempotently
        # and leaves a manual owner alone; it never publishes another cancel.
        now = time.monotonic()
        self._last_cancel_at = now
        self._mux.cancel_autonomy(now, notify=False)

    def _on_enable(self, request, response):
        now = time.monotonic()
        if request.data and now - self._last_cancel_at < CANCEL_QUIET_S:
            response.success = False
            response.message = "cancel_settling"
            return response
        response.success = self._mux.set_autonomy_enabled(request.data, now)
        response.message = self._mux.reason
        self._publish_status()
        return response

    def _publish_status(self):
        payload = json.dumps(self._mux.status(), sort_keys=True)
        if payload != self._last_status:
            self._status.publish(String(data=payload))
            self._last_status = payload

    def _tick(self):
        now = time.monotonic()
        decision = self._mux.step(now)
        if decision.cancel_autonomy:
            self._last_cancel_at = now
            self._cancel.publish(Empty())
        if decision.command is not None:
            message = Twist()
            (message.linear.x, message.linear.y, message.linear.z,
             message.angular.x, message.angular.y, message.angular.z) = decision.command
            self._pub.publish(message)
        self._publish_status()


def main(args=None):
    rclpy.init(args=args)
    node = CommandMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
