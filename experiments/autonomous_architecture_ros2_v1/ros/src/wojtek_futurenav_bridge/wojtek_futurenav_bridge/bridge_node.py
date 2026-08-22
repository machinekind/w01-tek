"""FutureNav bridge node: VLM decisions in, /cmd_vel out.

The missing edge of the sim E2E loop (issue #13): it lets the deployment
ROS graph (`./ros/sim.sh`, hw:=mujoco) be driven by the FutureNav action
server with nothing else from the agent stack in between.

    /wojtek/nav_instruction (String)         "walk to the sofa" | "stop"
    /camera/camera/color/image_raw (Image)   the simulated D435 color feed
    TF odom -> base_link                     pose for the mid-level executor
        |
        v
    POST /reset, POST /act  ->  discrete action  ->  FutureNavEpisode
        |
        v
    /cmd_vel (Twist) at cmd_rate_hz while an episode runs

Command-stream etiquette matches text_commander: while an episode is active
the node publishes continuously (zeros while a decision is in flight, so a
non-zero command is never left latched downstream); when the episode ends --
done, aborted, cancelled, HTTP failure -- it publishes exactly one zero
Twist and goes silent, so the console or a pad can take /cmd_vel without
being shouted over.

HTTP runs in a single worker thread (one request in flight at a time);
results come back through a queue drained by the command timer, so every
state change happens on the rclpy executor.
"""

from __future__ import annotations

import queue
import threading

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from wojtek_rl.navigation import NavConfig, quat_to_yaw

from .episode import (
    DEFAULT_MAX_BLOCKED,
    DEFAULT_MAX_ROTATION,
    DEFAULT_MAX_STEPS,
    DEFAULT_NAV,
    FutureNavEpisode,
)
from .frames import image_to_jpeg_b64
from .futurenav_http import DEFAULT_FUTURENAV_URL, FutureNavHttpClient, FutureNavHttpError

CANCEL_WORDS = ("stop", "cancel", "stój", "stop.")


class FutureNavBridgeNode(Node):
    def __init__(self):
        super().__init__("wojtek_futurenav_bridge")
        self.declare_parameter("vlm_url", DEFAULT_FUTURENAV_URL)
        self.declare_parameter("cmd_rate_hz", 20.0)
        self.declare_parameter("frame_px", 224)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("vx_max", DEFAULT_NAV.vx_max)
        self.declare_parameter("vy_max", DEFAULT_NAV.vy_max)
        self.declare_parameter("yaw_max", DEFAULT_NAV.yaw_max)
        self.declare_parameter("max_steps", DEFAULT_MAX_STEPS)
        self.declare_parameter("max_rotation", DEFAULT_MAX_ROTATION)
        self.declare_parameter("max_blocked", DEFAULT_MAX_BLOCKED)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.frame_px = int(p("frame_px"))
        self.odom_frame = p("odom_frame")
        self.base_frame = p("base_frame")
        nav = NavConfig(
            vx_max=float(p("vx_max")),
            vy_max=float(p("vy_max")),
            yaw_max=float(p("yaw_max")),
            stop_radius=DEFAULT_NAV.stop_radius,
        )
        self.episode = FutureNavEpisode(
            nav_cfg=nav,
            max_steps=int(p("max_steps")),
            max_rotation=int(p("max_rotation")),
            max_blocked=int(p("max_blocked")),
        )
        self.client = FutureNavHttpClient(p("vlm_url"))
        self.get_logger().info(f"FutureNav server: {self.client.url}")

        # TF: the sim plant publishes ground-truth odom->base_link at 100 Hz.
        from tf2_ros import Buffer, TransformListener

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_frame: Image | None = None
        self._results: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._needs_reset = False
        self._was_publishing = False

        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(
            Image, "/camera/camera/color/image_raw", self._on_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String, "/wojtek/nav_instruction", self._on_instruction, 10,
        )
        self.create_timer(1.0 / float(p("cmd_rate_hz")), self._on_timer)

    # -- inputs -------------------------------------------------------------

    def _on_image(self, msg: Image) -> None:
        self.latest_frame = msg

    def _on_instruction(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return
        if text.lower() in CANCEL_WORDS:
            self.episode.cancel("stop requested")
            return
        self.get_logger().info(f"instruction: {text!r}")
        self.episode.start(text)
        self._needs_reset = True

    # -- decision worker ----------------------------------------------------

    def _request_decision(
        self, frame_b64: str, do_reset: bool, instruction: str, epoch: int
    ) -> None:
        def work():
            try:
                if do_reset:
                    self.client.reset(instruction)
                action = self.client.act(frame_b64)
                self._results.put(("action", action, epoch))
            except FutureNavHttpError as exc:
                self._results.put(("error", str(exc), epoch))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _drain_results(self) -> None:
        while True:
            try:
                kind, value, epoch = self._results.get_nowait()
            except queue.Empty:
                return
            self._worker = None
            if kind == "action":
                self.get_logger().info(f"step {self.episode.steps + 1}: {value}")
                self.episode.apply_action(value, epoch)
            else:
                self.get_logger().error(f"action server: {value}")
                self.episode.fail_decision(value, epoch)

    # -- main loop ----------------------------------------------------------

    def _lookup_pose(self) -> tuple[float, float, float] | None:
        from rclpy.time import Time

        try:
            tf = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, Time()
            )
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return t.x, t.y, quat_to_yaw(q.w, q.x, q.y, q.z)

    def _on_timer(self) -> None:
        self._drain_results()

        end = self.episode.take_end()
        if end is not None:
            self.get_logger().info(
                f"episode {end.outcome} after {end.steps} steps: {end.reason}"
            )

        if not self.episode.active:
            if self._was_publishing or end is not None:
                self.pub_cmd.publish(Twist())  # the single mandatory zero
                self._was_publishing = False
            return

        if self.episode.needs_decision and self._worker is None:
            if self.latest_frame is None:
                self.get_logger().warning(
                    "no camera frame yet; waiting", throttle_duration_sec=5.0
                )
            else:
                try:
                    frame_b64 = image_to_jpeg_b64(self.latest_frame, self.frame_px)
                except ValueError as exc:
                    self.get_logger().error(str(exc))
                    self.episode.fail_decision(str(exc))
                    return
                epoch = self.episode.mark_decision_pending()
                do_reset, self._needs_reset = self._needs_reset, False
                self._request_decision(
                    frame_b64, do_reset, self.episode.instruction, epoch
                )

        pose = self._lookup_pose()
        if pose is None:
            # No pose, no motion: publish zero so nothing stays latched.
            self.get_logger().warning(
                f"TF {self.odom_frame}->{self.base_frame} unavailable",
                throttle_duration_sec=5.0,
            )
            self.pub_cmd.publish(Twist())
            self._was_publishing = True
            return

        cmd = self.episode.tick(*pose)
        if cmd is None:  # the tick itself may have ended the episode
            return
        vx, vy, wyaw = cmd
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wyaw)
        self.pub_cmd.publish(msg)
        self._was_publishing = True


def main(args=None):
    rclpy.init(args=args)
    node = FutureNavBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
