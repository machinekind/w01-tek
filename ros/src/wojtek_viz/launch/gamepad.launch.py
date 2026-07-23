"""Bluetooth Xbox pad teleop: the standard `joy` driver + gamepad_teleop.

    ros2 launch wojtek_viz gamepad.launch.py [device_id:=0]

Pair the pad over bluetooth on the machine this runs on first (bluetoothctl:
scan on -> pair -> connect; the pad must show up under /dev/input). Runs
unchanged against the sim or the real stack -- it publishes the same
/cmd_vel the consoles do. `./sim.sh --gamepad` brings this up in place of
the web console; see wojtek_viz/gamepad_teleop.py for the stick mapping.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            # Which /dev/input device the joy driver opens (0 = first pad).
            DeclareLaunchArgument("device_id", default_value="0"),
            Node(
                package="joy",
                executable="joy_node",
                output="screen",
                parameters=[
                    {
                        "device_id": ParameterValue(
                            LaunchConfiguration("device_id"), value_type=int
                        ),
                        # Keep publishing while sticks are held steady --
                        # gamepad_teleop's dead-man counts on a steady stream
                        # and must only trip when the pad actually drops off.
                        "autorepeat_rate": 20.0,
                        "deadzone": 0.05,
                    }
                ],
            ),
            Node(
                package="wojtek_viz",
                executable="gamepad_teleop",
                output="screen",
            ),
        ]
    )
