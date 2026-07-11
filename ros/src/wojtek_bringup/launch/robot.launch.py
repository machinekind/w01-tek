"""Robot-side launch: MD80 (IMPEDANCE) + IMU via ros2_control + policy.

This is the canonical launch for the RPi. It starts only hardware/control
nodes -- no RViz, no GUI. Run visualization/debug on the PC separately:
    ros2 launch wojtek_viz viz.launch.py

    ros2 launch wojtek_bringup robot.launch.py [max_torque:=2.0] [dry_run:=true]
                                               [boot_pose:=home|folded]

Startup/arming procedure is unchanged from real.launch.py:
  1. Power the motors and launch this file (any robot pose is fine). The
     policy runs immediately but real_io_node starts DISARMED -- no commands
     reach the motors.
  2. Physically pose the robot in the boot pose (default boot_pose:=home)
     and compare against RViz on the PC.
  3. ros2 service call /wojtek/zero std_srvs/srv/Trigger
  4. ros2 service call /wojtek/stand_up std_srvs/srv/Trigger
  5. ros2 service call /wojtek/arm std_srvs/srv/SetBool '{data: true}'
  6. When done: disarm, then /wojtek/lie_down to ramp gently down to folded.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("wojtek_bringup")
    xacro_file = os.path.join(share, "urdf", "wojtek_real.urdf.xacro")
    max_torque = LaunchConfiguration("max_torque")
    imu_port = LaunchConfiguration("imu_port")
    robot_description = ParameterValue(
        Command(
            ["xacro ", xacro_file, " max_torque:=", max_torque,
             " imu_port:=", imu_port]
        ),
        value_type=str,
    )

    return LaunchDescription(
        [
            # Start with a low torque cap for the first tests; the policy was
            # trained with 6 Nm available.
            DeclareLaunchArgument("max_torque", default_value="6.0"),
            # Default (empty) keeps the xacro's by-id path for the CP2102;
            # override with e.g. imu_port:=/dev/ttyUSB0 if the adapter is
            # ever swapped for one with a different by-id name.
            DeclareLaunchArgument(
                "imu_port",
                default_value=(
                    "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_"
                    "Bridge_Controller_0001-if00-port0"
                ),
            ),
            DeclareLaunchArgument("dry_run", default_value="false"),
            # Pose the robot is in when the motors activate / zero. "home"
            # (standing, position 0) by default; "folded" only if the drives'
            # raw zero matches the folded pose -- see real_io_node.
            DeclareLaunchArgument("boot_pose", default_value="home"),
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
                remappings=[("joint_states", "wojtek/joint_states_abs")],
            ),
            # wojtek.rviz uses odom as the fixed frame (the sim publishes ground
            # truth odom -> base_link); there is no odometry on the real robot
            # yet, so pin base_link at the origin for visualization.
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                arguments=["--frame-id", "odom", "--child-frame-id", "base_link"],
            ),
            Node(
                package="wojtek_bringup",
                executable="real_io_node",
                output="screen",
                parameters=[
                    {
                        "dry_run": LaunchConfiguration("dry_run"),
                        "boot_pose": LaunchConfiguration("boot_pose"),
                    }
                ],
            ),
            Node(
                package="wojtek_policy",
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
                    ("joint_states", "wojtek/joint_states_abs"),
                    ("imu/data", "imu_sensor_broadcaster/imu"),
                ],
            ),
        ]
    )
