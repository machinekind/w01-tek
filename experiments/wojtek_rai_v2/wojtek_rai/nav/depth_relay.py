"""QoS bridge for the depth camera.

The camera (sim and RealSense alike) publishes best-effort; depthimage_to_
laserscan subscribes reliable, so they never match and no scan appears.
This node subscribes with sensor-data QoS and republishes image + camera_info
reliable on /nav/depth/{image,camera_info}, unchanged.

    python3 -m wojtek_rai.nav.depth_relay
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class DepthRelay(Node):
    def __init__(self) -> None:
        super().__init__("depth_relay")
        self.declare_parameter("image_in", "/camera/camera/depth/image_rect_raw")
        self.declare_parameter("info_in", "/camera/camera/depth/camera_info")
        self.declare_parameter("image_out", "/nav/depth/image")
        self.declare_parameter("info_out", "/nav/depth/camera_info")
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._img_pub = self.create_publisher(Image, self.get_parameter("image_out").value, reliable)
        self._info_pub = self.create_publisher(CameraInfo, self.get_parameter("info_out").value, reliable)
        self.create_subscription(
            Image, self.get_parameter("image_in").value, self._img_pub.publish, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, self.get_parameter("info_in").value, self._info_pub.publish, qos_profile_sensor_data
        )
        self.get_logger().info("depth relay up: best-effort camera -> reliable /nav/depth/*")


def main() -> None:
    rclpy.init()
    node = DepthRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
