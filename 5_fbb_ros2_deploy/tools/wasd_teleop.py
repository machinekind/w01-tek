#!/usr/bin/env python3
"""Hold-to-move WASD teleop for fbb_policy (publishes geometry_msgs/Twist on /cmd_vel).

Unlike teleop_twist_keyboard (which latches the last command forever), this
node treats terminal key autorepeat as "key still held": while a movement key
keeps repeating the command stays active, and once no key arrives within
RELEASE_TIMEOUT the command drops to zero — so the robot stops when you let go.

Keys:
  w / s   forward / back
  a / d   turn left / right
  q / e   strafe left / right
  + / -   scale speed up / down
  space   immediate stop
  x       quit

Tip: lower your desktop keyboard-repeat delay for snappier response, e.g.
GNOME: gsettings set org.gnome.desktop.peripherals.keyboard delay 200
"""
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

PUBLISH_RATE_HZ = 20.0
# Must exceed the desktop's initial autorepeat delay (~500 ms default),
# otherwise the command drops out between the first press and the repeats.
RELEASE_TIMEOUT_S = 0.6
DEFAULT_SPEED = 0.4      # m/s, policy trained to vx <= 0.6
DEFAULT_TURN = 0.8       # rad/s
SPEED_STEP = 1.2         # multiplicative +/- step

# key -> (vx, vy, wz) direction multipliers
BINDINGS = {
    "w": (1.0, 0.0, 0.0),
    "s": (-1.0, 0.0, 0.0),
    "a": (0.0, 0.0, 1.0),
    "d": (0.0, 0.0, -1.0),
    "q": (0.0, 1.0, 0.0),
    "e": (0.0, -1.0, 0.0),
}


class WasdTeleop(Node):
    def __init__(self):
        super().__init__("wasd_teleop")
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.speed = DEFAULT_SPEED
        self.turn = DEFAULT_TURN
        self.active = (0.0, 0.0, 0.0)
        self.last_key_time = 0.0
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self.tick)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def handle_key(self, key):
        if key in BINDINGS:
            self.active = BINDINGS[key]
            self.last_key_time = self.now_s()
        elif key == " ":
            self.active = (0.0, 0.0, 0.0)
        elif key == "+":
            self.speed *= SPEED_STEP
            self.turn *= SPEED_STEP
            print(f"speed {self.speed:.2f} m/s  turn {self.turn:.2f} rad/s\r")
        elif key == "-":
            self.speed /= SPEED_STEP
            self.turn /= SPEED_STEP
            print(f"speed {self.speed:.2f} m/s  turn {self.turn:.2f} rad/s\r")
        elif key in ("x", "\x03"):  # x or ctrl-c
            raise KeyboardInterrupt

    def tick(self):
        if self.active != (0.0, 0.0, 0.0) and \
                self.now_s() - self.last_key_time > RELEASE_TIMEOUT_S:
            self.active = (0.0, 0.0, 0.0)
        msg = Twist()
        msg.linear.x = self.active[0] * self.speed
        msg.linear.y = self.active[1] * self.speed
        msg.angular.z = self.active[2] * self.turn
        self.pub.publish(msg)


def main():
    if not sys.stdin.isatty():
        sys.exit("wasd_teleop needs an interactive terminal")
    print(__doc__)
    rclpy.init()
    node = WasdTeleop()
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    tty.setraw(fd)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            ready, _, _ = select.select([sys.stdin], [], [], 0.02)
            if ready:
                node.handle_key(sys.stdin.read(1))
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        node.pub.publish(Twist())  # zero command on exit
        node.destroy_node()
        rclpy.shutdown()
        print("\nstopped")


if __name__ == "__main__":
    main()
