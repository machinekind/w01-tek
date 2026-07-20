"""Robot-side launch: MD80 (IMPEDANCE) + IMU via ros2_control + policy.

This is the canonical launch for the RPi. It starts only hardware/control
nodes -- no RViz, no GUI. Run visualization/debug on the PC separately:
    ros2 launch wojtek_viz viz.launch.py

    ros2 launch wojtek_bringup robot.launch.py [max_torque:=2.0] [dry_run:=true]
                                               [boot_pose:=home|folded] [bag:=true]

Recording note: the PC's viz.launch.py already records the whole run over
DDS (all RPi topics are visible there), so on-robot recording is OFF by
default here. Enable it -- bag:=true bag_cpus:=0,1 -- for a guaranteed
lossless capture (localhost, no wifi loss), e.g. untethered runs where the
PC's bag would drop samples. Bags go to bag_dir/run_<timestamp>
(bag_dir defaults to ~/wojtek_bags).

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

import datetime
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
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
    xacro_file = os.path.join(share, "urdf", "wojtek_real.urdf.xacro")
    max_torque = LaunchConfiguration("max_torque")
    imu_bus = LaunchConfiguration("imu_bus")
    imu_addr_ag = LaunchConfiguration("imu_addr_ag")
    imu_addr_mag = LaunchConfiguration("imu_addr_mag")
    bus = LaunchConfiguration("bus")
    can_baud = LaunchConfiguration("can_baud")
    use_imu = LaunchConfiguration("use_imu")
    robot_description = ParameterValue(
        Command(
            ["xacro ", xacro_file, " max_torque:=", max_torque,
             " use_imu:=", use_imu, " imu_bus:=", imu_bus,
             " imu_addr_ag:=", imu_addr_ag, " imu_addr_mag:=", imu_addr_mag,
             " bus:=", bus, " can_baud:=", can_baud]
        ),
        value_type=str,
    )

    # One rosbag per run: a fresh timestamped subdirectory under bag_dir, named
    # when the launch is generated (each `ros2 launch` / service (re)start gets
    # its own bag). We collect little data, so recording everything (-a) as a
    # per-run log is cheap and worth having.
    default_bag_dir = os.path.join(os.path.expanduser("~"), "wojtek_bags")
    bag_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_output = PathJoinSubstitution(
        [LaunchConfiguration("bag_dir"), TextSubstitution(text=f"run_{bag_stamp}")]
    )

    return LaunchDescription(
        [
            # The policy was trained with 6 Nm available. This launch always
            # passes max_torque on to xacro, so THIS default is the effective
            # one -- the arg in wojtek_real.urdf.xacro never gets a say.
            DeclareLaunchArgument("max_torque", default_value="6.0"),
            # CAN link to the drives: CANdle HAT over SPI at 8M by default
            # (the drives' flashed baudrate since 2026-07-17). bus:=usb
            # can_baud:=1 = the legacy USB dongle, only after flashing the
            # drives back to 1M (ros/hw_tests: candle_bus_test baud).
            DeclareLaunchArgument("bus", default_value="spi"),
            DeclareLaunchArgument("can_baud", default_value="8"),
            # IMU is the Adafruit 5543 (LSM6DS3TR-C + LIS3MDL) straight on
            # I2C1 -- imu_i2c_hardware_interface, bench-tested but not yet
            # exercised through ros2_control on the robot, hence off by
            # default until that first hardware test passes.
            DeclareLaunchArgument("use_imu", default_value="false"),
            DeclareLaunchArgument("imu_bus", default_value="/dev/i2c-1"),
            # 0x6B/0x1E if the board's SDO pins are pulled high.
            DeclareLaunchArgument("imu_addr_ag", default_value="0x6A"),
            DeclareLaunchArgument("imu_addr_mag", default_value="0x1C"),
            DeclareLaunchArgument("dry_run", default_value="false"),
            # Pose the robot is in when the motors activate / zero. "home"
            # (standing, position 0) by default; "folded" only if the drives'
            # raw zero matches the folded pose -- see real_io_node.
            DeclareLaunchArgument("boot_pose", default_value="home"),
            # On-robot recording is OFF by default: the PC's viz.launch.py
            # records the run over DDS. Enable (bag:=true, ideally with
            # bag_cpus:=0,1) for a lossless localhost capture -- see the module
            # docstring. bag_dir:=/some/path relocates the output.
            DeclareLaunchArgument("bag", default_value="false"),
            DeclareLaunchArgument("bag_dir", default_value=default_bag_dir),
            # Optional CPU affinity for the recorder (comma list, e.g. "0,1").
            # Empty = inherit. The RPi service wraps the whole launch in
            # `taskset -c 2,3` (the isolated RT cores); it sets bag_cpus:=0,1 to
            # keep the recorder's disk I/O off the control loop's cores.
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
                condition=IfCondition(use_imu),
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_state_broadcaster",
                           "forward_position_controller"],
                condition=UnlessCondition(use_imu),
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
                # IMU needs no remap: policy_node subscribes the broadcaster's
                # topic name directly (the sim publishes the same name).
                remappings=[
                    ("joint_states", "wojtek/joint_states_abs"),
                ],
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
                    'echo ">> rosbag: recording to $2"\n'
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
