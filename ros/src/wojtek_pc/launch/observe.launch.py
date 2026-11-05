"""Loopback-only Foxglove observer, without command or parameter capabilities."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        Node(package="foxglove_bridge", executable="foxglove_bridge",
             name="wojtek_observer", output="screen",
             parameters=[PathJoinSubstitution([
                 FindPackageShare("wojtek_pc"), "config", "observer.yaml",
             ])]),
    ])
