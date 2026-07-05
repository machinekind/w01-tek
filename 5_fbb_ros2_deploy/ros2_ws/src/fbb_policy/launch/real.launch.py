"""Policy on the real robot: MD80 (IMPEDANCE) + BMX160 IMU via ros2_control.

    ros2 launch fbb_policy real.launch.py [max_torque:=2.0] [dry_run:=true]

Startup procedure (see README for the full checklist):
  1. Place the robot in the home standing pose, THEN power the motors.
  2. Launch this file. The policy runs immediately but real_io_node starts
     DISARMED -- no commands reach the motors.
  3. Verify /fbb/joint_states_abs and the RViz pose look sane.
  4. ros2 service call /fbb/arm std_srvs/srv/SetBool '{data: true}'
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("fbb_policy")
    xacro_file = os.path.join(share, "urdf", "fbb_real.urdf.xacro")
    max_torque = LaunchConfiguration("max_torque")
    robot_description = ParameterValue(
        Command(["xacro ", xacro_file, " max_torque:=", max_torque]),
        value_type=str,
    )

    return LaunchDescription(
        [
            # Start with a low torque cap for the first tests; the policy was
            # trained with 6 Nm available.
            DeclareLaunchArgument("max_torque", default_value="6.0"),
            DeclareLaunchArgument("dry_run", default_value="false"),
            DeclareLaunchArgument("rviz", default_value="false"),
            Node(
                package="controller_manager",
                executable="ros2_control_node",
                parameters=[
                    {"robot_description": robot_description},
                    os.path.join(share, "config", "real_controllers.yaml"),
                ],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_state_broadcaster", "imu_sensor_broadcaster",
                           "forward_position_controller"],
            ),
            # RViz/robot_state_publisher use ABSOLUTE joint angles.
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
                remappings=[("joint_states", "fbb/joint_states_abs")],
            ),
            Node(
                package="fbb_policy",
                executable="real_io_node",
                output="screen",
                parameters=[{"dry_run": LaunchConfiguration("dry_run")}],
            ),
            Node(
                package="fbb_policy",
                executable="policy_node",
                output="screen",
                parameters=[
                    {
                        # URDF imu_joint: rpy 0 pi 0 relative to base_link.
                        "imu_mount_rpy": [0.0, 3.141592653589793, 0.0],
                        "auto_enable": True,  # real_io arming is the gate
                        "soft_start_s": 2.0,
                        "clamp_knee": True,
                        "watchdog_timeout_s": 0.2,
                    }
                ],
                remappings=[
                    ("joint_states", "fbb/joint_states_abs"),
                    ("imu/data", "imu_sensor_broadcaster/imu"),
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", os.path.join(share, "rviz", "fbb.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
