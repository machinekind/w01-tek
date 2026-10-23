"""Dead-man between Nav2 and the locomotion policy.

Republishes `/cmd_vel_nav` (Twist) as `/cmd_vel`; when no message arrives for
`timeout_s`, publishes one zero Twist and goes quiet until Nav2 speaks again.
`policy_node` holds its last command forever, so without this a crashed or
paused Nav2 would leave the robot walking.

What this node can and cannot cover:

* Nav2 crash / pause / Ctrl-C on the nav stack: covered. The dead-man sends
  a stop, and the process takes SIGINT/SIGTERM itself so the stop goes out
  while the rclpy context is still valid (rclpy's own handler tears the
  context down first and the stop would never leave the laptop).
* A WiFi drop between the laptop and the RPi: NOT covered by a PC-side
  node. The last non-zero Twist is already latched in policy_node and no
  message from here can reach it while the link is down. The mitigation
  here is a heartbeat: the node watches a robot-originated stream
  (`heartbeat_topic`, `/joint_states` on both the RPi and the sim) and,
  when it resumes after being stale for `heartbeat_timeout_s`, bursts zero
  Twists for `STOP_BURST_S` at the tick rate so the re-matched reader gets a
  stop. The burst is gated by `stop_burst_on_reconnect` (default off;
  nothing enables it yet) because it fights the pad -- twist_mux with pad
  priority must land first, after which the real-target launch can pass
  `-p stop_burst_on_reconnect:=true`. A cmd_vel age timeout inside
  policy_node (a ros/ change) is the real fix.

    ros2 run --prefix '' python3 -m wojtek_rai.nav.cmd_vel_watchdog
"""

from __future__ import annotations

import signal
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState

TICK_S = 0.1
# Stop on shutdown: a few repeats back to back so one lost sample is not the
# difference between a stopped and a walking robot.
STOP_REPEATS, STOP_PERIOD_S = 5, 0.02
# Stop burst after the heartbeat resumes: one zero per tick for this long.
STOP_BURST_S = 1.0
STOP_BURST_TICKS = round(STOP_BURST_S / TICK_S)


class CmdVelWatchdog(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_watchdog")
        self.declare_parameter("input_topic", "/cmd_vel_nav")
        self.declare_parameter("output_topic", "/cmd_vel")
        self.declare_parameter("timeout_s", 0.5)
        self.declare_parameter("heartbeat_topic", "/joint_states")
        self.declare_parameter("heartbeat_timeout_s", 1.0)
        self.declare_parameter("stop_burst_on_reconnect", False)
        self._timeout = float(self.get_parameter("timeout_s").value)
        self._hb_timeout = float(self.get_parameter("heartbeat_timeout_s").value)
        self._burst_enabled = bool(self.get_parameter("stop_burst_on_reconnect").value)
        self._pub = self.create_publisher(Twist, self.get_parameter("output_topic").value, 10)
        self.create_subscription(Twist, self.get_parameter("input_topic").value, self._on_cmd, 10)
        self.create_subscription(
            JointState, self.get_parameter("heartbeat_topic").value, self._on_heartbeat, 10
        )
        self._last = None
        self._stopped = True
        self._hb_last = None
        self._hb_lost = False
        self._burst_left = 0
        self.create_timer(TICK_S, self._tick)
        self.get_logger().info(
            f"cmd_vel watchdog: {self.get_parameter('input_topic').value} -> "
            f"{self.get_parameter('output_topic').value}, dead-man {self._timeout:.2f} s, "
            f"heartbeat {self.get_parameter('heartbeat_topic').value} "
            f"({self._hb_timeout:.1f} s, stop burst on reconnect "
            f"{'on' if self._burst_enabled else 'off'})"
        )

    def send_stop(self) -> None:
        """Publish STOP_REPEATS zero Twists back to back (used on shutdown)."""
        for _ in range(STOP_REPEATS):
            self._pub.publish(Twist())
            time.sleep(STOP_PERIOD_S)

    def _on_cmd(self, msg: Twist) -> None:
        self._last = self.get_clock().now()
        self._stopped = False
        self._pub.publish(msg)

    def _on_heartbeat(self, _msg: JointState) -> None:
        now = self.get_clock().now()
        if self._hb_lost:
            # The robot is talking again after a gap: the reader on the RPi
            # may have been re-matched and is still holding the last command
            # that got through. Only a stop published now can reach it.
            self._hb_lost = False
            if self._burst_enabled:
                self._burst_left = STOP_BURST_TICKS
                self._stopped = True
                self.get_logger().warn(
                    f"heartbeat back after {(now - self._hb_last).nanoseconds * 1e-9:.1f} s "
                    f"-- bursting stop for {STOP_BURST_S:.1f} s"
                )
            else:
                self.get_logger().warn("heartbeat back -- stop burst disabled, not sending")
        self._hb_last = now

    def _tick(self) -> None:
        if self._burst_left > 0:
            self._burst_left -= 1
            self._pub.publish(Twist())
        self._check_heartbeat()
        if self._stopped or self._last is None:
            return
        age = (self.get_clock().now() - self._last).nanoseconds * 1e-9
        if age > self._timeout:
            self._pub.publish(Twist())
            self._stopped = True
            self.get_logger().warn(f"no cmd_vel_nav for {age:.2f} s -- sent stop")

    def _check_heartbeat(self) -> None:
        if self._hb_last is None or self._hb_lost:
            return
        age = (self.get_clock().now() - self._hb_last).nanoseconds * 1e-9
        if age > self._hb_timeout:
            # Link (or robot) gone: enter the stopped state. The zero sent
            # here most likely cannot reach the robot; the burst on reconnect
            # is what covers it.
            self._hb_lost = True
            self._stopped = True
            self._pub.publish(Twist())
            self.get_logger().warn(f"no heartbeat for {age:.2f} s -- link down? sent stop")


def main() -> None:
    # rclpy's own SIGINT handler tears the context down before we can publish;
    # take the signals ourselves and send the stop while the context is valid.
    # The 0.1 s tick guarantees the Python-level handler runs within ~TICK_S.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = CmdVelWatchdog()

    def _stop(signum, _frame):
        node.send_stop()
        node.get_logger().warn(f"signal {signum} -- sent stop")
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _stop)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
