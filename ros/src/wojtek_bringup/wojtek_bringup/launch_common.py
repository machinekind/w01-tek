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

from wojtek_policy.policy_source import active_policy, load_policy


def _launch_setup(context, with_rviz, hardware):
    share = get_package_share_directory("wojtek_bringup")
    policy_share = get_package_share_directory("wojtek_policy")

    loaded = load_policy(
        LaunchConfiguration("policy").perform(context),
        overrides={k: LaunchConfiguration(k).perform(context)
                   for k in ("kp", "kd", "max_torque")},
    )
    pd = loaded.pd
    # Feed-forward torque head: read straight from the contract, so a
    # tau_ff policy is (like everything else) a config change, not a launch
    # flag. The DRIVE torque limit must cover the trained envelope's sum --
    # the sim clamps the PD servo (max_torque) and the head (scale)
    # separately, so their peaks can coincide.
    tff = loaded.meta.get("tau_ff") or {}
    tau_ff_on = bool(tff.get("enable"))
    tau_ff_scale = float(tff.get("scale", 0.0))
    drive_torque = pd["max_torque"] + (tau_ff_scale if tau_ff_on else 0.0)
    print(f">> policy {loaded.run_name} from {loaded.source}; servo settings "
          f"kp={pd['kp']:g} kd={pd['kd']:g} max_torque={pd['max_torque']:g}"
          + (f"; tau_ff head +-{tau_ff_scale:g} N*m -> drive torque limit "
             f"{drive_torque:g}" if tau_ff_on else ""))

    use_imu = LaunchConfiguration("use_imu")
    # The servo contract (gains, torque cap), the IMU switch and the bench flag
    # are the same question on both sides, so they go to both xacros. What
    # differs is what the plugin needs to reach its hardware: a CAN link and an
    # I2C address for the real drives, a physics backend for the simulated one.
    xacro_args = [
        f" kp:={pd['kp']} kd:={pd['kd']} max_torque:={drive_torque}",
        f" tau_ff:={'true' if tau_ff_on else 'false'}",
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
            # The sim plant clamps the servo and the head separately (as the
            # training sim does), so it needs the scale next to the summed
            # drive cap; the real drive takes only the sum.
            f" tau_ff_scale:={tau_ff_scale if tau_ff_on else 0.0}",
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

    # Deterministic placement on the robot. isolcpus turns OFF load
    # balancing between the isolated cores, so under the service's plain
    # {2,3} mask every child stays wherever fork put it -- observed on the
    # Pi 4 as the whole Python stack piling onto CPU2 (99% busy, policy
    # down to ~44 Hz) while the RT loop's CPU3 idled. Measured budget that
    # fits (2026-08-24, with perception + odometry + the on-robot map):
    #   core 3: ros2_control (RT loop) + real_io        ~65%
    #   core 2: policy + leg_odometry                   ~65%
    #   cores 0,1 (with the OS): camera driver, the accumulated map,
    #     pad/joy/robot_state_publisher -- UI-rate, none of it
    #     control-critical, all fine sharing with the system.
    # SCHED_FIFO keeps the control loop preemptive over its core-mate
    # either way. The sim runs unpinned -- no isolcpus on a PC.
    ctrl_prefix = "taskset -c 3" if hardware == "real" else None
    aux_prefix = "taskset -c 2" if hardware == "real" else None
    ui_prefix = "taskset -c 0,1" if hardware == "real" else None

    nodes = [
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[
                {"robot_description": robot_description},
                os.path.join(share, "config", "real_controllers.yaml"),
            ],
            prefix=ctrl_prefix,
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster", "imu_sensor_broadcaster",
                       "magnetometer_broadcaster",
                       "forward_position_controller"]
            # The effort channel exists only when the URDF exported the
            # interface (tau_ff contract) -- spawning it otherwise would
            # fail claiming a missing command interface.
            + (["forward_effort_controller"] if tau_ff_on else []),
            condition=IfCondition(use_imu),
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster",
                       "forward_position_controller"]
            + (["forward_effort_controller"] if tau_ff_on else []),
            condition=UnlessCondition(use_imu),
        ),
        # RViz/robot_state_publisher use ABSOLUTE joint angles.
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            remappings=[("joint_states", "wojtek/joint_states_abs")],
            prefix=ui_prefix,
        ),
        # The odom->base_link edge. On the real robot leg_odometry owns it
        # (below); the static identity stays only as the no-IMU fallback so
        # bench runs still render in RViz. A physics-backed simulation knows
        # the true base pose and publishes the transform itself, so there
        # the static one would fight it.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["--frame-id", "odom", "--child-frame-id", "base_link"],
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration("hw"), "' != 'mujoco'"])
            ) if hardware == "sim" else UnlessCondition(use_imu),
        ),
        # Leg-kinematics + IMU odometry, on the robot itself: autonomy keeps
        # no PC in the loop, and cloud_accumulate/nav consume wojtek/odom
        # locally. Publishes the odom->base_link TF (deliberately replacing
        # the static identity above). Needs the IMU, hence the gate.
        Node(
            package="wojtek_odometry",
            executable="leg_odometry_node",
            output="screen",
            prefix=aux_prefix,
            # input_stride 2: the abs joint stream arrives at ~50 Hz on the
            # robot (joint_state_broadcaster's rate), and the per-message
            # kinematics costs ~6 ms on the A72 -- 25 Hz processing fits
            # the core budget; full rate does not (see the node).
            parameters=[{"publish_tf": True, "input_stride": 2}],
            condition=IfCondition(use_imu),
        ) if hardware == "real" else None,
        Node(
            package="wojtek_bringup",
            executable="real_io_node",
            output="screen",
            prefix=ctrl_prefix,
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
            prefix=aux_prefix,
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
    # Sim-only entries resolve to None on the other hardware (and vice
    # versa); drop them instead of handing launch a None action.
    nodes = [n for n in nodes if n is not None]

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


DEFAULT_POLICY = active_policy()


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
        # Which policy runs: a Hugging Face repo id (org/name[@revision]) or
        # a local directory with policy.npz + policy_meta.json. Pin a commit
        # for a durable real-robot run: policy:=<repo>@<sha>. A Hugging Face
        # reference is answered from the policy store. deploy.sh keeps the
        # default there, and deploy.sh --policy <ref> ships and activates
        # any other reference. The robot needs no network.
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
                # UI-rate nodes: on the robot they go to the system cores
                # (see the placement note in _launch_setup); in the sim they
                # inherit (empty = no taskset).
                launch_arguments={
                    "cpus": "0,1" if hardware == "real" else "",
                }.items(),
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
        # inherit that mask, so without re-affinitizing it the driver
        # (~0.7 of a Pi 4 core measured with colour+RGBD on) competes with
        # the 400 Hz control loop. Same treatment the bag recorder gets.
        # Even off the RT cores the camera is not free for the loop: its USB
        # traffic and the IMU's i2c completion share CPU0's interrupt path
        # (see the I2C_TIMEOUT note in imu_i2c.cpp).
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
                # 6 fps on both streams: the only depth consumer left is
                # the accumulated map (~2 processed frames/s), and rclpy
                # pays a fixed per-message deserialization cost for every
                # frame it receives -- at 15 fps that alone saturated a
                # Pi 4 core together with the driver. Colour rides along:
                # the RGBD product pairs depth with colour, so their rates
                # must match, and the VLM decides at ~0.3-0.5 Hz anyway.
                "depth_profile": "848x480x6",
                "color_profile": "1280x720x6",
            }.items(),
            condition=IfCondition(LaunchConfiguration("perception")),
        ),
        OpaqueFunction(
            function=_launch_setup,
            kwargs={"with_rviz": with_rviz, "hardware": hardware},
        ),
    ]
    return LaunchDescription(args)
