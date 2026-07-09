#!/usr/bin/env python3

# Copyright 2026 Jakub Delicat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    urdf_file = os.path.join(
        get_package_share_directory("four_bar_bot_description"),
        "urdf",
        "four_bar_bot.urdf.xacro",
    )
    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare("quadruped_controller"),
            "config",
            "hardware.yaml",
        ]
    )

    use_hardware = LaunchConfiguration("use_hardware")
    delcare_use_hardware = DeclareLaunchArgument(
        name="use_hardware",
        default_value="false",
        description="Use hardware or not",
    )

    use_sim = LaunchConfiguration("use_sim")
    declare_use_sim = DeclareLaunchArgument(
        name="use_sim",
        default_value="true",
        description="Use simulation or not",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_controllers],
        emulate_tty=True,
        remappings=[
            ("/diff_drive_controller/cmd_vel_unstamped", "/cmd_vel"),
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        condition=UnlessCondition(use_sim),
    )

    twist_to_trajectory = Node(
        package="quadruped_controller",
        executable="twist_to_trajectory.py",
    )

    return LaunchDescription(
        [
            delcare_use_hardware,
            declare_use_sim,
            DeclareLaunchArgument(
                name="use_verbose",
                default_value="false",
                description='Set to "true" to run verbose logging.',
            ),
            DeclareLaunchArgument(
                name="robot_description",
                default_value=Command(
                    ["xacro ", urdf_file, " use_hardware:=", use_hardware, " use_sim:=", use_sim]
                ),
                description="Absolute path to robot urdf file",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim,
                        "robot_description": LaunchConfiguration("robot_description"),
                    }
                ],
            ),
            joint_state_broadcaster_spawner,
            control_node,
            twist_to_trajectory,
        ]
    )
