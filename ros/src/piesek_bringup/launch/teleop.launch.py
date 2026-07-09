"""Xbox-controller teleop: joy driver + fbb teleop mapper.

Run alongside real.launch.py or sim.launch.py:

    ros2 launch piesek_bringup teleop.launch.py

Left stick = velocity (vx/vy), right stick X = yaw, right stick Y slowly
raises/lowers the commanded body height. Pair the pad over Bluetooth first
(bluetoothctl: scan on / pair / trust / connect) -- joy_node then picks it
up as the first game controller.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="joy",
                executable="joy_node",
                output="screen",
                parameters=[
                    {
                        # Keep /joy flowing when sticks are idle so the
                        # teleop watchdog sees a live controller.
                        "autorepeat_rate": 20.0,
                        # Shaping happens in xbox_teleop_node.
                        "deadzone": 0.0,
                    }
                ],
            ),
            Node(
                package="wojtek_policy",
                executable="xbox_teleop_node",
                output="screen",
            ),
        ]
    )
