#!/usr/bin/env python3
"""Xbox-gamepad teleop -- drive Wojtek with a bluetooth pad.

    ros2 launch wojtek_viz gamepad.launch.py      # joy driver + this node

Sits behind the standard ROS `joy` driver (sensor_msgs/Joy in) and speaks the
SAME surface the operator consoles do -- /cmd_vel out (height on linear.z),
/wojtek/arm for arming -- so nothing in wojtek_bringup / wojtek_policy
changes and it works identically against the sim and the real robot.

Mapping (Linux xpad/xpadneo enumeration; every index is a parameter in case
another driver enumerates differently):

  left stick   up/down     vx (forward/back)
               left/right  yaw (left = positive wz, i.e. turn left)
  right stick  left/right  vy (strafe; left = positive vy)
  A                        toggle /wojtek/arm
  D-pad up/down            standing height +- height_step (held set-point)

Driving is ALWAYS live -- there is no drive on/off gate like the consoles
have; the sticks command velocity whenever the stack accepts it. Two rails
stay in place anyway:
  * sticks scale into the policy's trained command box (policy_meta.json
    deploy.command_low/high, the same source policy_node clips against), and
  * a dead-man zeroes the motion if Joy messages stop (pad powered off / out
    of bluetooth range / joy driver gone) for more than cmd_timeout_s. The
    height set-point is held through stops, same rule as the consoles.
"""
import json
import os

import rclpy
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_srvs.srv import SetBool

# Fallbacks for a policy_meta.json without a deploy box -- same values and
# same role as in web_console.py / operator_console.py.
DEFAULT_CMD_LOW = (-0.6, -0.4, -0.7)
DEFAULT_CMD_HIGH = (0.6, 0.4, 0.7)
DEFAULT_HEIGHT_RANGE = (0.09, 0.17)
DEFAULT_HEIGHT = 0.125

DRIVE_TICK_HZ = 20.0     # /cmd_vel publish rate (same as the consoles)


class GamepadTeleop(Node):
    def __init__(self):
        super().__init__("gamepad_teleop")
        # Axis/button indices follow the joy driver's Xbox enumeration under
        # Linux xpad/xpadneo: 0/1 left stick (left/up = +1), 3 right stick
        # horizontal, D-pad hat as axes 6/7, button 0 = A.
        self._ax_vx = self.declare_parameter("vx_axis", 1).value
        self._ax_yaw = self.declare_parameter("yaw_axis", 0).value
        self._ax_vy = self.declare_parameter("vy_axis", 3).value
        self._ax_dpad_y = self.declare_parameter("dpad_y_axis", 7).value
        self._btn_arm = self.declare_parameter("arm_button", 0).value
        # Extra deadzone on top of the joy driver's own, rescaled so full
        # deflection still reaches the box edge (bluetooth pads idle off-center).
        self._deadzone = self.declare_parameter("deadzone", 0.1).value
        self._height_step = self.declare_parameter("height_step", 0.005).value
        self._cmd_timeout = self.declare_parameter("cmd_timeout_s", 0.5).value

        self._load_meta()

        self._cmd = (0.0, 0.0, 0.0)   # normalized [-1, 1] (vx, vy, yaw)
        self._height = self.height_default
        self._joy_stamp = None
        self._deadman_hit = False
        self._arm_btn_prev = 0
        self._dpad_y_prev = 0.0
        self._armed = False           # last state we successfully commanded

        self._arm_cli = self.create_client(SetBool, "wojtek/arm")
        self._pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Joy, "joy", self._on_joy, 10)
        self.create_timer(1.0 / DRIVE_TICK_HZ, self._tick)
        self.get_logger().info(
            "gamepad teleop up -- left stick vx/yaw, right stick strafe, "
            f"A toggles arm, D-pad height +-{self._height_step * 1000:.0f} mm "
            f"({self.height_range[0]:.3f}..{self.height_range[1]:.3f} m)"
        )

    def _load_meta(self):
        """Trained command box + height envelope, same source as the consoles."""
        self.cmd_low = list(DEFAULT_CMD_LOW)
        self.cmd_high = list(DEFAULT_CMD_HIGH)
        self.height_range = list(DEFAULT_HEIGHT_RANGE)
        self.height_default = DEFAULT_HEIGHT
        try:
            share = get_package_share_directory("wojtek_policy")
            meta = json.loads(
                open(os.path.join(share, "config", "policy_meta.json")).read())
        except (PackageNotFoundError, OSError, ValueError) as e:
            self.get_logger().warning(f"no policy meta ({e}); driving with "
                                      "default command limits")
            return
        deploy = meta.get("deploy", {})
        self.cmd_low = list(deploy.get("command_low", DEFAULT_CMD_LOW))[:3]
        self.cmd_high = list(deploy.get("command_high", DEFAULT_CMD_HIGH))[:3]
        self.height_range = [
            deploy.get("command_height_low", DEFAULT_HEIGHT_RANGE[0]),
            deploy.get("command_height_high", DEFAULT_HEIGHT_RANGE[1]),
        ]
        self.height_default = deploy.get("command_height", 0.0) or DEFAULT_HEIGHT

    # ---- pad input -----------------------------------------------------------
    def _shape(self, v):
        """Deadzone with rescale: dead near center, still 1.0 at full throw."""
        if abs(v) < self._deadzone:
            return 0.0
        s = (abs(v) - self._deadzone) / (1.0 - self._deadzone)
        return min(1.0, s) if v > 0 else -min(1.0, s)

    def _axis(self, msg, idx):
        return msg.axes[idx] if 0 <= idx < len(msg.axes) else 0.0

    def _on_joy(self, msg):
        self._cmd = (
            self._shape(self._axis(msg, self._ax_vx)),
            self._shape(self._axis(msg, self._ax_vy)),
            self._shape(self._axis(msg, self._ax_yaw)),
        )
        self._joy_stamp = self.get_clock().now()

        # A: toggle arm on the press edge.
        btn = msg.buttons[self._btn_arm] if self._btn_arm < len(msg.buttons) else 0
        if btn and not self._arm_btn_prev:
            self._toggle_arm()
        self._arm_btn_prev = btn

        # D-pad up/down: step the held height set-point on the press edge
        # (the hat reports -1/0/+1, +1 = up).
        dpad = self._axis(msg, self._ax_dpad_y)
        if abs(self._dpad_y_prev) < 0.5 and abs(dpad) >= 0.5:
            lo, hi = self.height_range
            step = self._height_step if dpad > 0 else -self._height_step
            self._height = min(max(self._height + step, lo), hi)
            self.get_logger().info(f"height set-point {self._height:.3f} m")
        self._dpad_y_prev = dpad

    def _toggle_arm(self):
        if not self._arm_cli.service_is_ready():
            self.get_logger().warning("wojtek/arm service unavailable "
                                      "(sim has no arming; drive just works)")
            return
        target = not self._armed
        fut = self._arm_cli.call_async(SetBool.Request(data=target))

        def done(f, target=target):
            try:
                resp = f.result()
            except Exception as e:  # noqa: BLE001 -- surface any RPC failure
                self.get_logger().error(f"arm call failed: {e}")
                return
            if resp.success:
                self._armed = target
            level = self.get_logger().info if resp.success else self.get_logger().error
            level(f"arm({target}): {resp.message}")
        fut.add_done_callback(done)

    # ---- drive tick ------------------------------------------------------------
    def _tick(self):
        stale = (
            self._joy_stamp is None
            or (self.get_clock().now() - self._joy_stamp).nanoseconds
            > self._cmd_timeout * 1e9
        )
        if stale:
            # Dead-man: zero the motion, hold the stance height. Publishing
            # (not going silent) keeps the last non-zero stick from sticking
            # in policy_node, which latches the last received command.
            vx = vy = yaw = 0.0
            if not self._deadman_hit and self._joy_stamp is not None:
                self.get_logger().warning("joy input stale -- zeroing /cmd_vel")
        else:
            # Normalized [-1,1] -> the trained (asymmetric) command box:
            # positive stick scales by high, negative by low.
            lo, hi = self.cmd_low, self.cmd_high
            vx, vy, yaw = (v * hi[i] if v >= 0 else v * -lo[i]
                           for i, v in enumerate(self._cmd))
        self._deadman_hit = stale

        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(yaw)
        # Standing-height command; policy_node treats 0 as "use the default".
        t.linear.z = float(self._height)
        self._pub_cmd.publish(t)


def main():
    rclpy.init()
    node = GamepadTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
