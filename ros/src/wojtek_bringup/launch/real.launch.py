"""Policy on the real robot: MD80 (IMPEDANCE) + IMU via ros2_control.

    IMU is currently a loaner BMI160 (no magnetometer) -- see
    bmi160_serial_hardware_interface and wojtek_real.urdf.xacro. Swap back to
    wojtek_imu_ros2_control there once the real BMX160 is sourced.

    ros2 launch wojtek_bringup real.launch.py [max_torque:=2.0] [dry_run:=true]
                                              [boot_pose:=home|folded] [bag:=false]

    Every run records a full rosbag (all topics) to bag_dir/run_<timestamp>
    (bag_dir defaults to ~/wojtek_bags). Disable with bag:=false.

Startup procedure:
  1. Power the motors and launch this file (any robot pose is fine). The
     policy runs immediately but real_io_node starts DISARMED -- no commands
     reach the motors.
  2. Physically pose the robot in the boot pose (default boot_pose:=home,
     the standing pose RViz shows; folded also available -- see real_io_node)
     and compare the real robot against RViz.
  3. ros2 service call /wojtek/zero std_srvs/srv/Trigger  -- declares "the robot
     is in the boot pose NOW" and re-zeros the offsets. (Skippable only if
     the robot was already exactly in the boot pose at activation.)
  4. ros2 service call /wojtek/stand_up std_srvs/srv/Trigger  (slow ramp to the
     home standing pose; skip if already standing in home)
  5. ros2 service call /wojtek/arm std_srvs/srv/SetBool '{data: true}'
  6. When done: disarm, then /wojtek/lie_down to ramp gently down to folded.
"""

import datetime
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    TextSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("wojtek_bringup")
    policy_share = get_package_share_directory("wojtek_policy")
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

    # One rosbag per run: a fresh timestamped subdirectory under bag_dir, named
    # when the launch is generated (each `ros2 launch` gets its own bag). We
    # collect little data, so recording everything (-a) as a per-run log is
    # cheap and worth having.
    default_bag_dir = os.path.join(os.path.expanduser("~"), "wojtek_bags")
    bag_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_output = PathJoinSubstitution(
        [LaunchConfiguration("bag_dir"), TextSubstitution(text=f"run_{bag_stamp}")]
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
            DeclareLaunchArgument("rviz", default_value="false"),
            # Pose the robot is in when the motors activate / zero. "home"
            # (standing, position 0) by default; "folded" only if the drives'
            # raw zero matches the folded pose -- see real_io_node.
            DeclareLaunchArgument("boot_pose", default_value="home"),
            # Record the whole run to a rosbag by default. Turn off with
            # bag:=false; change where they land with bag_dir:=/some/path.
            DeclareLaunchArgument("bag", default_value="true"),
            DeclareLaunchArgument("bag_dir", default_value=default_bag_dir),
            # Optional CPU affinity for the recorder (comma list, e.g. "0,1");
            # empty = inherit. Mainly for the RPi service (see robot.launch.py).
            DeclareLaunchArgument("bag_cpus", default_value=""),
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
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", os.path.join(policy_share, "rviz", "wojtek.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
            # bash -c: mkdir the parent (rosbag2 creates the bag dir itself but
            # not missing parents), then exec the recorder -- optionally under
            # taskset when bag_cpus is set. Paths/cpus are passed as argv ($1..$3),
            # not spliced into the script, so values with spaces are safe.
            ExecuteProcess(
                condition=IfCondition(LaunchConfiguration("bag")),
                cmd=[
                    "bash", "-c",
                    'mkdir -p "$1"\n'
                    'if [ -n "$3" ]; then\n'
                    '  exec taskset -c "$3" ros2 bag record -a -o "$2"\n'
                    'fi\n'
                    'exec ros2 bag record -a -o "$2"\n',
                    "wojtek_bag_record",  # $0 (shell name in messages)
                    LaunchConfiguration("bag_dir"),  # $1
                    bag_output,  # $2
                    LaunchConfiguration("bag_cpus"),  # $3
                ],
                output="screen",
            ),
        ]
    )
