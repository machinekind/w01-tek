"""Shared launch body for the real-robot bringups (real/robot launch files).

Both start the same MD80 + IMU + policy node set from a policy reference and
differ only in two things: whether RViz runs (PC vs the headless RPi service)
and the rosbag default. Those are the `with_rviz`/`bag_default` parameters
here; the per-file docstrings document the two workflows. The robot-side
launch additionally offers the pad teleop (`with_gamepad`).

Imported by the launch files at launch time. The same environment that lets
them import wojtek_policy.policy_source makes this sibling module importable.
"""

import datetime
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    TextSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from wojtek_policy.policy_source import load_policy


def _launch_setup(context, with_rviz):
    share = get_package_share_directory("wojtek_bringup")
    policy_share = get_package_share_directory("wojtek_policy")
    xacro_file = os.path.join(share, "urdf", "wojtek_real.urdf.xacro")

    loaded = load_policy(
        LaunchConfiguration("policy").perform(context),
        overrides={k: LaunchConfiguration(k).perform(context)
                   for k in ("kp", "kd", "max_torque")},
    )
    pd = loaded.pd
    print(f">> policy {loaded.run_name} from {loaded.source}; servo settings "
          f"kp={pd['kp']:g} kd={pd['kd']:g} max_torque={pd['max_torque']:g}")

    use_imu = LaunchConfiguration("use_imu")
    robot_description = ParameterValue(
        Command(
            ["xacro ", xacro_file,
             f" kp:={pd['kp']} kd:={pd['kd']} max_torque:={pd['max_torque']}",
             " use_imu:=", use_imu,
             " imu_bus:=", LaunchConfiguration("imu_bus"),
             " imu_addr_ag:=", LaunchConfiguration("imu_addr_ag"),
             " imu_addr_mag:=", LaunchConfiguration("imu_addr_mag"),
             " bus:=", LaunchConfiguration("bus"),
             " can_baud:=", LaunchConfiguration("can_baud"),
             " dry_run:=", LaunchConfiguration("dry_run")]
        ),
        value_type=str,
    )

    # One rosbag per run: a fresh timestamped subdirectory under bag_dir, named
    # when the launch is generated (each `ros2 launch` / service (re)start gets
    # its own bag). We collect little data, so recording everything (-a) as a
    # per-run log is cheap and worth having.
    bag_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_output = PathJoinSubstitution(
        [LaunchConfiguration("bag_dir"), TextSubstitution(text=f"run_{bag_stamp}")]
    )

    nodes = [
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
                       "magnetometer_broadcaster",
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
                    # Already-resolved local dir + its provenance, so the node
                    # loads the same files without resolving the ref again.
                    "policy": str(loaded.directory),
                    "policy_source": loaded.source,
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
    ]

    if with_rviz:
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", os.path.join(policy_share, "rviz", "wojtek.rviz")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            )
        )

    nodes.append(
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
        )
    )
    return nodes


def common_launch_description(with_rviz, bag_default, with_gamepad=False):
    """LaunchDescription shared by real.launch.py and robot.launch.py.

    with_rviz adds the RViz node (and its `rviz` toggle arg); bag_default is
    the rosbag `bag` default ("true"/"false") for this launch's workflow;
    with_gamepad offers the robot-side pad teleop (its `gamepad` arg).
    """
    default_bag_dir = os.path.join(os.path.expanduser("~"), "wojtek_bags")
    args = [
        # Which policy runs: a Hugging Face repo id (org/name[@revision]) or a
        # local directory with policy.npz + policy_meta.json. For a durable
        # real-robot run pin a commit: policy:=<repo>@<sha>. First use of a new
        # revision needs network + an HF token on the RPi (prefetch:
        # python3 -m wojtek_policy.policy_source <ref>); afterwards it loads
        # from the local HF cache.
        DeclareLaunchArgument(
            "policy",
            default_value="<HF_ORGANIZATION>/wojtek-stiff-locomotion-v2"
            "@4dda27e12101a68dbf52bb134721b18dc166a7d3",
        ),
        # Explicit overrides of the policy contract's servo settings (empty =
        # from the contract). E.g. max_torque:=2 for cautious first tests.
        DeclareLaunchArgument("kp", default_value=""),
        DeclareLaunchArgument("kd", default_value=""),
        DeclareLaunchArgument("max_torque", default_value=""),
        # CAN link to the drives: CANdle HAT over SPI at 8M by default (the
        # drives' flashed baudrate since 2026-07-17). bus:=usb can_baud:=1 =
        # the legacy USB dongle, only after flashing the drives back to 1M
        # (ros/hw_tests: candle_bus_test baud).
        DeclareLaunchArgument("bus", default_value="spi"),
        DeclareLaunchArgument("can_baud", default_value="8"),
        # IMU is the Adafruit 5543 (LSM6DS3TR-C + LIS3MDL) straight on I2C1 --
        # imu_i2c_hardware_interface. On by default; use_imu:=false brings the
        # stack up with the sensor absent/unwired.
        DeclareLaunchArgument("use_imu", default_value="true"),
        DeclareLaunchArgument("imu_bus", default_value="/dev/i2c-1"),
        # 0x6B/0x1E if the board's SDO pins are pulled high.
        DeclareLaunchArgument("imu_addr_ag", default_value="0x6A"),
        DeclareLaunchArgument("imu_addr_mag", default_value="0x1C"),
        DeclareLaunchArgument("dry_run", default_value="false"),
        # Pose the robot is in when the motors activate / zero. "home"
        # (standing, position 0) by default; "folded" only if the drives' raw
        # zero matches the folded pose -- see real_io_node.
        DeclareLaunchArgument("boot_pose", default_value="home"),
        # Record the whole run to a rosbag (bag_dir/run_<timestamp>). Default
        # is per launch: on for the PC (real.launch.py), off for manual
        # robot.launch.py runs; the RPi service opts in explicitly with
        # bag:=true bag_cpus:=0,1 (wojtek-robot.service) -- see each launch's
        # docstring. bag_dir:=/some/path relocates output.
        DeclareLaunchArgument("bag", default_value=bag_default),
        DeclareLaunchArgument("bag_dir", default_value=default_bag_dir),
        # Optional CPU affinity for the recorder (comma list, e.g. "0,1");
        # empty = inherit. The RPi service pins it to 0,1 to keep the
        # recorder's disk I/O off the control loop's isolated RT cores.
        DeclareLaunchArgument("bag_cpus", default_value=""),
    ]
    if with_rviz:
        args.append(DeclareLaunchArgument("rviz", default_value="false"))
    if with_gamepad:
        args += [
            # Bluetooth Xbox pad paired with the RPi itself: joy driver +
            # wojtek_teleop's /cmd_vel mapping, no PC in the loop. Off by
            # default -- enable once the pad is paired (bluetoothctl; the
            # bluez/ERTM groundwork comes from deploy/rpi/install.sh). Only
            # one drive source at a time: with the pad on, leave the web
            # console's pad/drive alone, both publish the same /cmd_vel.
            DeclareLaunchArgument("gamepad", default_value="false"),
            IncludeLaunchDescription(
                PathJoinSubstitution(
                    [
                        FindPackageShare("wojtek_teleop"),
                        "launch",
                        "gamepad.launch.py",
                    ]
                ),
                condition=IfCondition(LaunchConfiguration("gamepad")),
            ),
        ]
    args.append(
        OpaqueFunction(function=_launch_setup, kwargs={"with_rviz": with_rviz})
    )
    return LaunchDescription(args)
