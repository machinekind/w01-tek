"""ROS shell for GotoController: setpoint in, /cmd_vel out, costmap veto.

Inputs
  wojtek/nav/goal      geometry_msgs/PoseStamped. The next setpoint. Only
                       the position is used. Any frame TF can resolve to
                       odom AT THE MESSAGE'S STAMP: a VLM answering a
                       picture sends "1 m ahead, 0.5 m left" in base_link
                       with the picture's stamp, and the robot that walked
                       on during the inference still gets the point the
                       VLM meant. An unstamped base_link goal means "now".
  wojtek/nav/costmap   nav_msgs/OccupancyGrid (latched), the rolling window
                       from nav2_costmap_2d; probed at a few points ahead.
  wojtek/nav/cancel    std_msgs/Empty. Drop the setpoint NOW (one zero
                       Twist, then idle) instead of at the dead-man: the
                       brain's stop, the console's stop button.
  TF odom->base_link   the robot's pose (leg_odometry).

Outputs
  cmd_vel              geometry_msgs/Twist at DRIVE_TICK_HZ while moving;
                       exactly ONE zero Twist when it stops (reached,
                       blocked, expired), then silence -- the zero is
                       mandatory (policy_node latches the last command),
                       the silence lets another drive source take over
                       without a shouting match. Same protocol as
                       text_commander.
  wojtek/nav/status    std_msgs/String, the controller's status word on
                       every change: idle / turning / driving / blocked /
                       reached. What a VLM loop reads before it decides.
"""

import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401 -- registers PoseStamped with tf2

from wojtek_nav.goto import GotoController

DRIVE_TICK_HZ = 20.0


class GotoNode(Node):
    def __init__(self):
        super().__init__("wojtek_goto")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("v_max", 0.3)
        self.declare_parameter("w_max", 0.5)
        self.declare_parameter("goal_timeout", 3.0)
        self.declare_parameter("reach_tolerance", 0.15)
        p = self.get_parameter
        self._odom_frame = p("odom_frame").value
        self._base_frame = p("base_frame").value
        self._ctrl = GotoController(
            v_max=p("v_max").value, w_max=p("w_max").value,
            goal_timeout=p("goal_timeout").value,
            reach_tolerance=p("reach_tolerance").value,
        )
        self._tf = Buffer()
        self._listener = TransformListener(self._tf, self)
        self._grid = None
        self._last_status = None
        self._moving = False
        self._cancelled_at = None   # node time of the last cancel

        latched = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(PoseStamped, "wojtek/nav/goal", self._on_goal, 10)
        self.create_subscription(OccupancyGrid, "wojtek/nav/costmap", self._on_grid, latched)
        self.create_subscription(Empty, "wojtek/nav/cancel", self._on_cancel, 10)
        self._pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)
        self._pub_status = self.create_publisher(String, "wojtek/nav/status", latched)
        self.create_timer(1.0 / DRIVE_TICK_HZ, self._tick)
        self._set_status("idle")
        self.get_logger().info(
            "goto up: PoseStamped on wojtek/nav/goal -> cmd_vel, veto from "
            "wojtek/nav/costmap, dead-man %.1f s" % p("goal_timeout").value
        )

    # -- inputs ----------------------------------------------------------

    def _on_goal(self, msg):
        # The resolver re-sends its setpoint on its own clock; one stamped
        # before a cancel can land here after it (two subscribers, no
        # ordering). Such a goal is the cancelled one and must not restart
        # the robot. An unstamped goal means "now" and is taken.
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._cancelled_at is not None and 0.0 < stamp <= self._cancelled_at:
            return
        try:
            if msg.header.frame_id and msg.header.frame_id != self._odom_frame:
                # At the message's stamp (zero stamp = latest), so a goal
                # relative to a picture is anchored where the picture was.
                msg = self._tf.transform(msg, self._odom_frame, timeout=rclpy.duration.Duration(seconds=0.2))
        except Exception as e:  # noqa: BLE001 -- TF, not our bug to crash on
            self.get_logger().warning(f"goal in '{msg.header.frame_id}' dropped: {e}")
            return
        self._ctrl.set_goal(msg.pose.position.x, msg.pose.position.y, self._now())

    def _on_grid(self, msg):
        self._grid = msg

    def _on_cancel(self, _msg):
        # The tick sees an idle controller next and, if the robot was
        # moving, sends the one zero Twist on that edge.
        self._cancelled_at = self._now()
        self._ctrl.cancel()

    # -- the loop --------------------------------------------------------

    def _tick(self):
        pose = self._pose()
        if pose is None:
            return
        cmd = self._ctrl.step(pose, self._now(), self._probe)
        moving = cmd.status in ("turning", "driving")
        if moving or self._moving:
            # While moving: the command. On the edge moving->stopped: the
            # one mandatory zero. Afterwards: nothing.
            twist = Twist()
            twist.linear.x, twist.angular.z = float(cmd.vx), float(cmd.wz)
            self._pub_cmd.publish(twist)
        self._moving = moving
        self._set_status(cmd.status)

    def _pose(self):
        try:
            t = self._tf.lookup_transform(self._odom_frame, self._base_frame, rclpy.time.Time())
        except Exception:  # noqa: BLE001 -- odometry not up yet
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return (t.transform.translation.x, t.transform.translation.y, yaw)

    def _probe(self, x, y):
        g = self._grid
        if g is None:
            return None
        i = int((x - g.info.origin.position.x) / g.info.resolution)
        j = int((y - g.info.origin.position.y) / g.info.resolution)
        if not (0 <= i < g.info.width and 0 <= j < g.info.height):
            return None
        return g.data[j * g.info.width + i]

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _set_status(self, status):
        if status != self._last_status:
            self._last_status = status
            self._pub_status.publish(String(data=status))
            self.get_logger().info(status)


def main():
    rclpy.init()
    node = GotoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
