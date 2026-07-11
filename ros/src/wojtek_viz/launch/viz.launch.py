"""PC-side visualization/debug for the LIVE robot (running on the RPi).

    ros2 launch wojtek_viz viz.launch.py [rviz:=true] [plotjuggler:=false]

This starts no hardware and no robot_state_publisher -- those run on the
robot (wojtek_bringup robot.launch.py on the RPi). Over the shared DDS link
this machine only consumes what the robot publishes:
  - RViz reads /robot_description (latched by the robot's RSP) and /tf.
  - PlotJuggler subscribes to the robot's topics for live plotting.

Teleop needs its own terminal (it reads the keyboard on stdin), so run it
separately rather than from here:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard

Make sure ROS_DOMAIN_ID and RMW match the robot (see docker/compose.yaml).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    policy_share = get_package_share_directory("wojtek_policy")

    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("plotjuggler", default_value="false"),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", os.path.join(policy_share, "rviz", "wojtek.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
            Node(
                package="plotjuggler",
                executable="plotjuggler",
                condition=IfCondition(LaunchConfiguration("plotjuggler")),
            ),
        ]
    )
