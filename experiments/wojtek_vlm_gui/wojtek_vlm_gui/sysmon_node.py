"""The robot's own load, published for the page's Computer panel.

    cd ~/wojtek_vlm_gui_bench && python3 -m wojtek_vlm_gui.sysmon_node   # on the RPi; README, "The camera bench"

One std_msgs/String a second on wojtek/sys/stat: the JSON of
`sysmon.Snapshot` (per-core percent busy, the RT cores, load, memory,
SoC temperature). The only piece of this experiment that runs on the
robot, because /proc is where the numbers are; everything else (the
page, the brain, the model) runs off the robot. Best effort and not
latched, like the camera: behind the cable bench's zenoh bridge a
latched topic replayed old readings out of order, and a reliable one
stalled for every reader once one reader died without saying goodbye (a
restarted page server). A new reading comes every second anyway.

Costs the RPi one /proc read a second and an idle rclpy executor.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from wojtek_vlm_gui import limits
from wojtek_vlm_gui.sysmon import SysMon, snapshot_to_dict


class SysMonNode(Node):
    def __init__(self) -> None:
        super().__init__("wojtek_sysmon")
        self.mon = SysMon()
        self.pub = self.create_publisher(String, limits.SYSMON_TOPIC, qos_profile_sensor_data)
        self.create_timer(limits.SYSMON_PERIOD_S, self.tick)
        self.get_logger().info(f"sysmon up: {self.mon.host} on {limits.SYSMON_TOPIC} "
                               f"every {limits.SYSMON_PERIOD_S} s")

    def tick(self) -> None:
        self.pub.publish(String(data=json.dumps(snapshot_to_dict(self.mon.read()))))


def main() -> None:
    rclpy.init()
    node = SysMonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
