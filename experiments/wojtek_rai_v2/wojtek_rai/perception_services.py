"""Detection-only perception services (GroundingDINO on /detection).

RAI's own launcher also loads SAM2-large for /segmentation; together they do
not fit a 4 GB laptop GPU next to the simulator's renderer. The navigation
tools need detection + depth only, so this starts just the DetectionService.

    python3 -m wojtek_rai.perception_services
"""

from __future__ import annotations

import rclpy
from rai.agents import wait_for_shutdown
from rai.communication.ros2 import ROS2Connector
from rai_perception.services import DetectionService


def main() -> None:
    rclpy.init()
    connector = ROS2Connector("detection_service", executor_type="single_threaded")
    connector.node.declare_parameter("enable_legacy_service_names", False)
    service = DetectionService(ros2_connector=connector)
    service.run()
    wait_for_shutdown([service])
    rclpy.shutdown()


if __name__ == "__main__":
    main()
