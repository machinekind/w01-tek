#!/usr/bin/env python3
"""Xbox-gamepad teleop -- drive Wojtek with a bluetooth pad.

    ros2 launch wojtek_teleop gamepad.launch.py   # joy driver + this node

Lives in wojtek_teleop (no GUI deps) so it runs on the RPi as well as the
PC container: pair the pad with whichever machine runs this launch.

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
  Y                        /wojtek/stand_up (ramp to standing pose)
  B                        /wojtek/lie_down (ramp down to folded)
  LB / RB                  standing height -/+ height_step (held set-point)
  D-pad up                 /wojtek/trick_paw_wave (offer + shake a paw)
  D-pad left               /wojtek/trick_bow
  D-pad right              /wojtek/trick_sit
  D-pad down               /wojtek/trick_shake (shake-off)

Tricks are real_io_node's scripted clips: like the Y/B ramps they are
refused while ARMED and (additionally) unless the robot stands near the
home pose, so a mid-walk D-pad press is rejected server-side.

The Y/B ramps mirror the operator consoles' Stand up / Lie down buttons --
same services, same rule that real_io_node refuses them while ARMED, so a
mid-walk Y/B press is rejected server-side rather than fought over.

Driving is ALWAYS live -- there is no drive on/off gate like the consoles
have; the sticks command velocity whenever the stack accepts it. Two rails
stay in place anyway:
  * sticks scale into the policy's trained command box, read from the
    contract of the `policy` parameter's reference (the same reference
    policy_node gets; empty = conservative defaults), shrunk by
    speed_scale (default 0.4; the launches' gamepad_speed), and
  * a dead-man zeroes the motion if Joy messages stop (pad powered off / out
    of bluetooth range / joy driver gone) for more than cmd_timeout_s. The
    height set-point is held through stops, same rule as the consoles.

/cmd_vel is shared with the other drive sources (deck gateway, consoles,
text commander), and policy_node keeps whichever message came last. So this
node publishes only while somebody drives: the joy driver repeats an idle
pad's state forever, but centred sticks are not input. When the sticks go
back to centre (or the joy stream stops) the motion is zeroed for
silence_after_s and the node then goes quiet, so a pad lying next to the
robot does not overwrite another source's commands twenty times a second.
The rule lives in pad_drive.py, tested without ROS.
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_srvs.srv import SetBool, Trigger

from wojtek_policy.policy_source import load_meta
from wojtek_teleop.pad_drive import IDLE, LIVE, ZEROING, PadDrive

# Fallbacks when no policy reference is set (or it fails to load) -- same
# values and same role as in web_console.py / operator_console.py.
DEFAULT_CMD_LOW = (-0.6, -0.4, -0.7)
DEFAULT_CMD_HIGH = (0.6, 0.4, 0.7)
DEFAULT_HEIGHT_RANGE = (0.09, 0.17)
DEFAULT_HEIGHT = 0.125

DRIVE_TICK_HZ = 20.0     # /cmd_vel publish rate (same as the consoles)
SILENCE_AFTER_S = 2.0    # zeroing burst length before going quiet


class GamepadTeleop(Node):
    def __init__(self):
        super().__init__("gamepad_teleop")
        # Axis/button indices follow the joy driver's Xbox enumeration under
        # Linux xpad/xpadneo: 0/1 left stick (left/up = +1), 3 right stick
        # horizontal, D-pad hat as axes 6/7, button 0 = A.
        self._ax_vx = self.declare_parameter("vx_axis", 1).value
        self._ax_yaw = self.declare_parameter("yaw_axis", 0).value
        self._ax_vy = self.declare_parameter("vy_axis", 3).value
        self._ax_dpad_x = self.declare_parameter("dpad_x_axis", 6).value
        self._ax_dpad_y = self.declare_parameter("dpad_y_axis", 7).value
        self._btn_arm = self.declare_parameter("arm_button", 0).value
        self._btn_stand = self.declare_parameter("stand_button", 3).value  # Y
        self._btn_lie = self.declare_parameter("lie_button", 1).value      # B
        self._btn_height_dn = self.declare_parameter("height_down_button", 4).value  # LB
        self._btn_height_up = self.declare_parameter("height_up_button", 5).value    # RB
        # Extra deadzone on top of the joy driver's own, rescaled so full
        # deflection still reaches the box edge (bluetooth pads idle off-center).
        self._deadzone = self.declare_parameter("deadzone", 0.1).value
        self._height_step = self.declare_parameter("height_step", 0.005).value
        self._cmd_timeout = self.declare_parameter("cmd_timeout_s", 0.5).value
        # Fraction of the command box full stick deflection reaches (vx, vy
        # and yaw alike). The box is what the policy CAN track, not what is
        # comfortable to drive indoors: at 1.0 the pad walks the robot at
        # its trained top speed. Raise it for open floor.
        self._speed_scale = min(1.0, max(
            0.0, float(self.declare_parameter("speed_scale", 0.4).value)))
        # Same reference policy_node gets (HF repo id or local directory);
        # empty = drive with the conservative default limits below.
        self.declare_parameter("policy", "")

        self._load_meta()
        # What full stick reaches: the command box shrunk by speed_scale.
        self.drive_low = [v * self._speed_scale for v in self.cmd_low]
        self.drive_high = [v * self._speed_scale for v in self.cmd_high]

        self._gate = PadDrive(self.drive_low, self.drive_high, self.height_range,
                              self.height_default, timeout_s=self._cmd_timeout,
                              silence_after_s=SILENCE_AFTER_S)
        self._last_state = IDLE
        self._arm_btn_prev = 0
        self._stand_btn_prev = 0
        self._lie_btn_prev = 0
        self._height_btn_prev = (0, 0)
        self._dpad_prev = (0.0, 0.0)
        self._armed = False           # last state we successfully commanded

        self._arm_cli = self.create_client(SetBool, "wojtek/arm")
        self._stand_cli = self.create_client(Trigger, "wojtek/stand_up")
        self._lie_cli = self.create_client(Trigger, "wojtek/lie_down")
        # D-pad direction -> real_io_node's scripted show clips.
        self._trick_clis = {
            "up": (self.create_client(Trigger, "wojtek/trick_paw_wave"),
                   "trick_paw_wave"),
            "left": (self.create_client(Trigger, "wojtek/trick_bow"),
                     "trick_bow"),
            "right": (self.create_client(Trigger, "wojtek/trick_sit"),
                      "trick_sit"),
            "down": (self.create_client(Trigger, "wojtek/trick_shake"),
                     "trick_shake"),
        }
        self._pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Joy, "joy", self._on_joy, 10)
        self.create_timer(1.0 / DRIVE_TICK_HZ, self._tick)
        self.get_logger().info(
            "gamepad teleop up -- left stick vx/yaw, right stick strafe, "
            "A toggles arm, Y stand up, B lie down, "
            f"LB/RB height +-{self._height_step * 1000:.0f} mm "
            f"({self.height_range[0]:.3f}..{self.height_range[1]:.3f} m), "
            "D-pad tricks: up=paw_wave left=bow right=sit down=shake; "
            f"speed_scale {self._speed_scale:g} -> full stick "
            f"vx {self.drive_high[0]:.2f} m/s, vy {self.drive_high[1]:.2f} m/s, "
            f"yaw {self.drive_high[2]:.2f} rad/s"
        )

    def _load_meta(self):
        """Trained command box + height envelope from the policy contract."""
        self.cmd_low = list(DEFAULT_CMD_LOW)
        self.cmd_high = list(DEFAULT_CMD_HIGH)
        self.height_range = list(DEFAULT_HEIGHT_RANGE)
        self.height_default = DEFAULT_HEIGHT
        ref = self.get_parameter("policy").value
        if not ref:
            self.get_logger().warning(
                "no policy reference set; driving with default command limits")
            return
        try:
            meta, source = load_meta(ref)
        except Exception as e:  # resolver/network/file -- stay drivable
            self.get_logger().warning(
                f"could not load policy contract {ref!r} ({e}); driving "
                "with default command limits")
            return
        self.cmd_low = [float(v) for v in meta["command_low"][:3]]
        self.cmd_high = [float(v) for v in meta["command_high"][:3]]
        if len(meta["command_low"]) >= 4:
            self.height_range = [
                float(meta["command_low"][3]), float(meta["command_high"][3])
            ]
        if meta.get("command_fill"):
            self.height_default = float(meta["command_fill"][0])
        self.get_logger().info(
            f"command box from {meta['run_name']} ({source})")

    # ---- pad input -----------------------------------------------------------
    def _shape(self, v):
        """Deadzone with rescale: dead near center, still 1.0 at full throw."""
        if abs(v) < self._deadzone:
            return 0.0
        s = (abs(v) - self._deadzone) / (1.0 - self._deadzone)
        return min(1.0, s) if v > 0 else -min(1.0, s)

    def _axis(self, msg, idx):
        return msg.axes[idx] if 0 <= idx < len(msg.axes) else 0.0

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _on_joy(self, msg):
        self._gate.joy(
            self._now(),
            self._shape(self._axis(msg, self._ax_vx)),
            self._shape(self._axis(msg, self._ax_vy)),
            self._shape(self._axis(msg, self._ax_yaw)),
        )

        # A: toggle arm on the press edge.
        btn = msg.buttons[self._btn_arm] if self._btn_arm < len(msg.buttons) else 0
        if btn and not self._arm_btn_prev:
            self._toggle_arm()
        self._arm_btn_prev = btn

        # Y / B: the consoles' stand-up / lie-down ramps, on the press edge.
        btn = msg.buttons[self._btn_stand] if self._btn_stand < len(msg.buttons) else 0
        if btn and not self._stand_btn_prev:
            self._call_trigger(self._stand_cli, "stand_up")
        self._stand_btn_prev = btn

        btn = msg.buttons[self._btn_lie] if self._btn_lie < len(msg.buttons) else 0
        if btn and not self._lie_btn_prev:
            self._call_trigger(self._lie_cli, "lie_down")
        self._lie_btn_prev = btn

        # LB/RB: step the held height set-point on the press edge.
        h_dn = msg.buttons[self._btn_height_dn] if self._btn_height_dn < len(msg.buttons) else 0
        h_up = msg.buttons[self._btn_height_up] if self._btn_height_up < len(msg.buttons) else 0
        if (h_dn and not self._height_btn_prev[0]) or (
                h_up and not self._height_btn_prev[1]):
            step = self._height_step if h_up else -self._height_step
            height = self._gate.step_height(self._now(), step)
            self.get_logger().info(f"height set-point {height:.3f} m")
        self._height_btn_prev = (h_dn, h_up)

        # D-pad: one trick per direction, on the press edge (the hat
        # reports -1/0/+1 per axis; +1 = up on y, +1 = left on x).
        dx = self._axis(msg, self._ax_dpad_x)
        dy = self._axis(msg, self._ax_dpad_y)
        px, py = self._dpad_prev
        direction = None
        if abs(py) < 0.5 and abs(dy) >= 0.5:
            direction = "up" if dy > 0 else "down"
        elif abs(px) < 0.5 and abs(dx) >= 0.5:
            direction = "left" if dx > 0 else "right"
        if direction is not None:
            cli, name = self._trick_clis[direction]
            self._call_trigger(cli, name)
        self._dpad_prev = (dx, dy)

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
            # Two distinct call sites on purpose: rclpy caches the severity
            # per call site, so routing success and failure through ONE
            # logger call raises "Logger severity cannot be changed between
            # calls" on the first refusal after a success -- which killed
            # this node the first time real_io refused to re-arm
            # (2026-08-10, robot out of the home-pose arming envelope).
            if resp.success:
                self.get_logger().info(f"arm({target}): {resp.message}")
            else:
                self.get_logger().error(f"arm({target}): {resp.message}")
        fut.add_done_callback(done)

    def _call_trigger(self, client, name):
        """Fire one of real_io_node's Trigger ramps and log its verdict."""
        if not client.service_is_ready():
            self.get_logger().warning(f"wojtek/{name} service unavailable "
                                      "(sim runs without real_io_node)")
            return
        fut = client.call_async(Trigger.Request())

        def done(f):
            try:
                resp = f.result()
            except Exception as e:  # noqa: BLE001 -- surface any RPC failure
                self.get_logger().error(f"{name} call failed: {e}")
                return
            # Distinct call sites; see the arm callback above.
            if resp.success:
                self.get_logger().info(f"{name}: {resp.message}")
            else:
                self.get_logger().error(f"{name}: {resp.message}")
        fut.add_done_callback(done)

    # ---- drive tick ------------------------------------------------------------
    def _tick(self):
        # The gate decides whether this node is driving at all (see
        # pad_drive.py): the pad speaks through the joy driver even when
        # nobody holds it, and an idle pad must not overwrite whatever else
        # (deck, console) drives /cmd_vel.
        now = self._now()
        out = self._gate.tick(now)
        state = self._gate.state
        if state != self._last_state:
            if state == ZEROING and self._gate.pad_lost(now):
                self.get_logger().warning("joy input stale -- zeroing /cmd_vel")
            elif state == ZEROING:
                self.get_logger().info("sticks released -- zeroing /cmd_vel")
            elif state == LIVE:
                self.get_logger().info("pad drives /cmd_vel")
            elif state == IDLE:
                self.get_logger().info("pad idle -- /cmd_vel released")
            self._last_state = state
        if out is None:
            return
        vx, vy, yaw, height = out
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(yaw)
        # Standing-height command; policy_node treats 0 as "use the default".
        t.linear.z = float(height)
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
