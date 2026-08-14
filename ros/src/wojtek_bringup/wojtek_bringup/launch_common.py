"""Shared launch body for every bringup: the real robot and the simulation.

One node set (ros2_control + real_io_node + policy_node + rsp + bag), one set
of parameters, one arming procedure. The launch files differ only in:

  hardware      "real" = MD80 over CAN + the I2C IMU; "sim" = the same graph
                with the hardware plugin swapped for a simulated one, from
                wojtek_pc's xacro. This is the whole point: the simulation is
                not a second stack, so soft_start_s/clamp_knee/watchdog and
                the IMU mount can not drift apart between the two.
  with_rviz     RViz on (PC) or off (the headless RPi service).
  bag_default   rosbag on/off for this workflow.
  with_gamepad  offer the robot-side pad teleop.

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
    PythonExpression,
    TextSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from wojtek_policy.policy_source import load_policy


def _launch_setup(context, with_rviz, hardware):
    share = get_package_share_directory("wojtek_bringup")
    policy_share = get_package_share_directory("wojtek_policy")

    loaded = load_policy(
        LaunchConfiguration("policy").perform(context),
        overrides={k: LaunchConfiguration(k).perform(context)
                   for k in ("kp", "kd", "max_torque")},
    )
    pd = loaded.pd
    print(f">> policy {loaded.run_name} from {loaded.source}; servo settings "
          f"kp={pd['kp']:g} kd={pd['kd']:g} max_torque={pd['max_torque']:g}")

    use_imu = LaunchConfiguration("use_imu")
    # The servo contract (gains, torque cap), the IMU switch and the bench flag
    # are the same question on both sides, so they go to both xacros. What
    # differs is what the plugin needs to reach its hardware: a CAN link and an
    # I2C address for the real drives, a physics backend for the simulated one.
    xacro_args = [
        f" kp:={pd['kp']} kd:={pd['kd']} max_torque:={pd['max_torque']}",
        " use_imu:=", use_imu,
        " dry_run:=", LaunchConfiguration("dry_run"),
    ]
    if hardware == "real":
        xacro_file = os.path.join(share, "urdf", "wojtek_real.urdf.xacro")
        xacro_args += [
            " imu_bus:=", LaunchConfiguration("imu_bus"),
            " imu_addr_ag:=", LaunchConfiguration("imu_addr_ag"),
            " imu_addr_mag:=", LaunchConfiguration("imu_addr_mag"),
            " bus:=", LaunchConfiguration("bus"),
            " can_baud:=", LaunchConfiguration("can_baud"),
        ]
    else:
        # wojtek_pc is PC-side and never deployed, so this import-by-name is
        # resolved only when a simulation is actually launched -- wojtek_bringup
        # must not depend on it.
        pc_share = get_package_share_directory("wojtek_pc")
        xacro_file = os.path.join(pc_share, "urdf", "wojtek_sim.urdf.xacro")
        # The plant starts in the pose real_io_node is told the robot is in;
        # anything else and zeroing would start from a lie.
        xacro_args += [
            " hw:=", LaunchConfiguration("hw"),
            " boot_pose:=", LaunchConfiguration("boot_pose"),
            " model_xml:=" + (
                LaunchConfiguration("model_xml").perform(context)
                or os.path.join(pc_share, "config", "scene_mjx.xml")
            ),
        ]
    robot_description = ParameterValue(
        Command(["xacro ", xacro_file] + xacro_args), value_type=str,
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
        # The RViz views use odom as the fixed frame; there is no odometry on
        # the real robot yet, so pin base_link at the origin for visualization.
        # A physics-backed simulation knows the true base pose and publishes
        # that transform itself, so there the static one would fight it.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["--frame-id", "odom", "--child-frame-id", "base_link"],
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration("hw"), "' != 'mujoco'"])
            ) if hardware == "sim" else None,
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
                    # URDF imu_joint: rpy 0 0 0 relative to base_link -- the
                    # sensor sits upright and its driver publishes the chip
                    # axes unmodified, so nothing needs rotating here. Keep
                    # this equal to body.urdf.xacro's imu_joint: the value is
                    # a copy, not derived, and the two drifting apart is what
                    # fed v41 an upside-down gravity vector.
                    "imu_mount_rpy": [0.0, 0.0, 0.0],
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
                arguments=["-d", LaunchConfiguration("rviz_config")],
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


# Which policy every bringup -- robot and simulation -- comes up with. Pinned
# to a revision on purpose: an unpinned repo id follows whatever main is, so
# what walks today would silently be a different policy tomorrow. Override for
# one run with policy:=<repo>@<sha> or a local artifact directory. The org
# comes from HF_ORGANIZATION in the environment (see .env.example). The repo
# is PRIVATE, but only the operator PC ever talks to Hugging Face: it fetches
# the revision into the policy store (python3 -m wojtek_policy.policy_source
# <ref>) and ./deploy.sh syncs that store to the RPi, which resolves from it.
# Without HF_ORGANIZATION the default is empty and every launch needs an
# explicit policy:=.
_HF_ORGANIZATION = os.environ.get("HF_ORGANIZATION", "")
DEFAULT_POLICY = (_HF_ORGANIZATION + "/wojtek-quiet-locomotion"
                  "@553795b13001cc1f519a4abc0235f275095129f8"
                  if _HF_ORGANIZATION else "")


def common_launch_description(
    with_rviz, bag_default, with_gamepad=False, hardware="real",
    policy_default=DEFAULT_POLICY,
):
    """LaunchDescription shared by the real-robot and simulation launches.

    with_rviz adds the RViz node (and its `rviz`/`rviz_config` args);
    bag_default is the rosbag `bag` default ("true"/"false") for this launch's
    workflow; with_gamepad offers the robot-side pad teleop (its `gamepad`
    arg); hardware is "real" (MD80 + I2C IMU) or "sim" (simulated plugin from
    wojtek_pc's xacro), which selects the URDF and the hardware-specific args;
    policy_default is the policy reference this workflow comes up with when
    none is given. Every launch takes DEFAULT_POLICY today -- the hook stays
    because a simulation and the robot may reasonably differ on which policy
    is the one to look at by default (an experimental one at the desk, a
    vetted one on the robot); a one-off divergence is `policy:=` instead.
    """
    if hardware not in ("real", "sim"):
        raise ValueError(f"hardware must be 'real' or 'sim', got {hardware!r}")
    default_bag_dir = os.path.join(os.path.expanduser("~"), "wojtek_bags")
    args = [
        # Which policy runs: a Hugging Face repo id (org/name[@revision]) or a
        # local directory with policy.npz + policy_meta.json. For a durable
        # real-robot run pin a commit: policy:=<repo>@<sha>. An HF reference is
        # answered from the policy store, so fetch the revision on the operator
        # PC first (python3 -m wojtek_policy.policy_source <ref>) and let
        # ./deploy.sh sync the store over; the robot needs no network.
        DeclareLaunchArgument("policy", default_value=policy_default),
        # Explicit overrides of the policy contract's servo settings (empty =
        # from the contract). E.g. max_torque:=2 for cautious first tests.
        DeclareLaunchArgument("kp", default_value=""),
        DeclareLaunchArgument("kd", default_value=""),
        DeclareLaunchArgument("max_torque", default_value=""),
        # The IMU switch is a question on both sides: use_imu:=false brings the
        # stack up with the sensor absent (unwired on the robot, left out of
        # the simulated component). It also drops the two sensor broadcasters,
        # so the policy runs without gravity/gyro -- bench use only.
        DeclareLaunchArgument("use_imu", default_value="true"),
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
    if hardware == "real":
        args += [
            # CAN link to the drives: CANdle HAT over SPI at 8M by default (the
            # drives' flashed baudrate since 2026-07-17). bus:=usb can_baud:=1 =
            # the legacy USB dongle, only after flashing the drives back to 1M
            # (ros/hw_tests: candle_bus_test baud).
            DeclareLaunchArgument("bus", default_value="spi"),
            DeclareLaunchArgument("can_baud", default_value="8"),
            # IMU is the Adafruit 5543 (LSM6DS3TR-C + LIS3MDL) straight on I2C1
            # -- imu_i2c_hardware_interface.
            DeclareLaunchArgument("imu_bus", default_value="/dev/i2c-1"),
            # 0x6B/0x1E if the board's SDO pins are pulled high.
            DeclareLaunchArgument("imu_addr_ag", default_value="0x6A"),
            DeclareLaunchArgument("imu_addr_mag", default_value="0x1C"),
        ]
    else:
        args += [
            # Which simulated hardware plugin runs. "mujoco" is the physics
            # backend; "mock" is ros2_control's own GenericSystem, which just
            # echoes commands back as states -- no dynamics, but the full node
            # graph, which makes it the fast way to test the stack's own logic
            # (arming, zeroing, ramps, watchdog) and the CI-friendly one.
            DeclareLaunchArgument(
                "hw", default_value="mujoco",
                choices=["mock", "mujoco"],
            ),
            # Physics scene for hw:=mujoco; empty = the plugin's default
            # (scene_mjx.xml shipped by wojtek_pc).
            DeclareLaunchArgument("model_xml", default_value=""),
        ]
    if with_rviz:
        # On the desk the simulation is watched, so RViz comes up by itself;
        # against the real robot viz is opt-in (viz.launch.py / robot.py own
        # that decision, and a stray RViz on the control machine is noise).
        args.append(
            DeclareLaunchArgument(
                "rviz", default_value="true" if hardware == "sim" else "false",
            )
        )
        args.append(
            DeclareLaunchArgument(
                "rviz_config",
                default_value=os.path.join(
                    get_package_share_directory("wojtek_pc"), "config", "sim.rviz",
                ) if hardware == "sim" else os.path.join(
                    get_package_share_directory("wojtek_policy"), "rviz",
                    "wojtek.rviz",
                ),
            )
        )
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
    args += [
        # RealSense D435 depth (+ colour for the VLM) and the grid reduction.
        # Off by default: the depth path is not on the robot's critical path,
        # and a missing driver package would take the whole launch -- and with
        # it the control stack -- down with it. Flip perception:=true in
        # wojtek-robot.service once the pipeline is proven on hardware.
        DeclareLaunchArgument("perception", default_value="false"),
        # The camera pipeline must not run on the isolated RT cores: the
        # service starts this whole tree under `taskset -c 2,3`, and children
        # inherit that mask, so without re-affinitizing them the driver and
        # the reduction (~26% of a core together, measured) compete with the
        # 400 Hz control loop. Same treatment the bag recorder gets.
        DeclareLaunchArgument("perception_cpus", default_value="0,1"),
        IncludeLaunchDescription(
            PathJoinSubstitution(
                [
                    FindPackageShare("wojtek_perception_bringup"),
                    "launch",
                    "perception.launch.py",
                ]
            ),
            launch_arguments={
                "cpus": LaunchConfiguration("perception_cpus"),
            }.items(),
            condition=IfCondition(LaunchConfiguration("perception")),
        ),
        OpaqueFunction(
            function=_launch_setup,
            kwargs={"with_rviz": with_rviz, "hardware": hardware},
        ),
    ]
    return LaunchDescription(args)
