"""Xbox-controller teleop for the Wojtek policy.

  subscribes  /joy            (sensor_msgs/Joy, from the standard `joy` driver)
  publishes   /cmd_vel        (geometry_msgs/Twist)   left stick = vx/vy,
                                                      right stick X = wz
              /wojtek/cmd_height (std_msgs/Float32)      right stick Y slowly
                                                      raises/lowers the body

The Bluetooth link itself is handled by the OS + the `joy` package
(sudo apt install ros-$ROS_DISTRO-joy). Pair once on the robot:

  bluetoothctl
    scan on            # hold the pad's pair button until it blinks fast
    pair <MAC>; trust <MAC>; connect <MAC>

then `ros2 launch wojtek_bringup teleop.launch.py` starts joy_node + this.

Command scaling comes from policy_meta.json (cmd_low/cmd_high,
cmd_height_range), so the pad can never command more than the policy was
trained to track. If /joy goes stale (controller off / out of range) the
node publishes a zero Twist so the dog stops; the height command holds.

Axis indices default to the ROS 2 `joy` (SDL) layout for an Xbox pad:
0 = left X, 1 = left Y, 2 = right X, 3 = right Y. Override the axis_*
parameters if your driver maps differently (check: ros2 topic echo /joy).
"""

import json
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32

from wojtek_policy.xbox_mapping import JoyCommandMapper

# Fallbacks for metas without the fields (same as policy_node).
CMD_LOW = (-0.6, -0.4, -0.7)
CMD_HIGH = (0.6, 0.4, 0.7)
HEIGHT_RANGE = (0.09, 0.17)


class XboxTeleopNode(Node):
    def __init__(self):
        super().__init__("xbox_teleop")
        share = get_package_share_directory("wojtek_policy")
        self.declare_parameter("policy_dir", f"{share}/config")
        self.declare_parameter("axis_vx", 1)  # left stick Y
        self.declare_parameter("axis_vy", 0)  # left stick X
        self.declare_parameter("axis_wz", 2)  # right stick X
        self.declare_parameter("axis_height", 3)  # right stick Y
        self.declare_parameter("deadzone", 0.15)
        # m/s of body-height change at full stick -- "slowly".
        self.declare_parameter("height_rate", 0.02)
        self.declare_parameter("initial_height", 0.13)
        self.declare_parameter("rate_hz", 50.0)
        # Zero the velocity command when /joy is older than this.
        self.declare_parameter("joy_timeout_s", 0.5)

        meta_path = Path(self.get_parameter("policy_dir").value) / "policy_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.mapper = JoyCommandMapper(
            cmd_low=meta.get("cmd_low", CMD_LOW),
            cmd_high=meta.get("cmd_high", CMD_HIGH),
            height_range=meta.get("cmd_height_range", HEIGHT_RANGE),
            height_rate=self.get_parameter("height_rate").value,
            deadzone=self.get_parameter("deadzone").value,
            initial_height=self.get_parameter("initial_height").value,
        )

        self._axes = None  # latest /joy axes
        self._joy_stamp = None
        self._was_stale = True

        self.create_subscription(Joy, "joy", self._on_joy, 10)
        self._pub_vel = self.create_publisher(Twist, "cmd_vel", 10)
        self._pub_height = self.create_publisher(Float32, "wojtek/cmd_height", 10)
        self._dt = 1.0 / self.get_parameter("rate_hz").value
        self.create_timer(self._dt, self._tick)
        self.get_logger().info(
            f"cmd box {np.round(self.mapper.cmd_low, 2).tolist()}"
            f"..{np.round(self.mapper.cmd_high, 2).tolist()}, "
            f"height {self.mapper.height_range} m "
            f"@ {self.mapper.height_rate} m/s, waiting for /joy"
        )

    def _on_joy(self, msg):
        self._axes = msg.axes
        self._joy_stamp = self.get_clock().now()

    def _axis(self, name):
        i = self.get_parameter(name).value
        return self._axes[i] if i < len(self._axes) else 0.0

    def _tick(self):
        timeout = self.get_parameter("joy_timeout_s").value
        stale = (
            self._joy_stamp is None
            or (self.get_clock().now() - self._joy_stamp).nanoseconds > timeout * 1e9
        )
        if stale != self._was_stale:
            self.get_logger().warning(
                "controller lost -- commanding zero velocity" if stale
                else "controller connected"
            )
            self._was_stale = stale

        msg = Twist()
        if not stale:
            vx, vy, wz = self.mapper.velocity(
                self._axis("axis_vx"), self._axis("axis_vy"), self._axis("axis_wz")
            )
            msg.linear.x, msg.linear.y, msg.angular.z = vx, vy, wz
            h = Float32()
            h.data = self.mapper.step_height(self._axis("axis_height"), self._dt)
            self._pub_height.publish(h)
        self._pub_vel.publish(msg)


def main():
    rclpy.init()
    node = XboxTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
