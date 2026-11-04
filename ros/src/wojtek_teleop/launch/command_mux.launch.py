"""Start the opt-in arbiter without connecting it to /cmd_vel by default."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("output", default_value="/wojtek/cmd_vel/selected"),
        DeclareLaunchArgument("source_timeout_s", default_value="0.5"),
        DeclareLaunchArgument("publish_hz", default_value="20.0"),
        Node(
            package="wojtek_teleop", executable="command_mux", output="screen",
            parameters=[{
                "source_timeout_s": ParameterValue(
                    LaunchConfiguration("source_timeout_s"), value_type=float,
                ),
                "publish_hz": ParameterValue(LaunchConfiguration("publish_hz"), value_type=float),
            }],
            remappings=[("/wojtek/cmd_vel/selected", LaunchConfiguration("output"))],
        ),
    ])
