"""Shared launch body for every bringup: the real robot and the simulation.

One node set (ros2_control + real_io_node + policy_node + rsp, plus the bag,
the telemetry and a Foxglove bridge when a run asks for them), one set of
parameters, one arming procedure. The launch files differ only in:

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

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
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

# The IMU's rotation in base_link, a copy of body.urdf.xacro's imu_joint rpy:
# upright, yawed -90 deg (chip +y forward, +x right). Every node that reads
# the sensor's axes gets it from here, and the simulator emulates the same
# mount (wojtek_sim.urdf.xacro), so the value is exercised in the sim too.
# The copy drifting from the URDF is what fed v41 an upside-down gravity.
IMU_MOUNT_RPY = [0.0, 0.0, -1.5707963]


def resolve_scene(model_xml):
    """The MuJoCo scene file a simulation loads, from the model_xml argument.

    Empty picks wojtek_pc's furnished scene_sim.xml; a bare file name
    (`model_xml:=scene_nav.xml`) is one of wojtek_pc's config/ scenes; a
    path is taken as given. One function for the two loaders of the scene
    (the plant inside ros2_control and the camera renderer), so they cannot
    resolve the same argument two ways and simulate different worlds.
    """
    config = os.path.join(get_package_share_directory("wojtek_pc"), "config")
    if not model_xml:
        return os.path.join(config, "scene_sim.xml")
    if os.sep not in model_xml:
        return os.path.join(config, model_xml)
    return model_xml


def _cpu_prefix(context, arg):
    """A taskset prefix from a comma list of cores, or nothing when empty.

    The RPi service starts this whole tree under `taskset -c 2,3` and every
    child inherits that mask. This moves anything that is not the control
    loop back to the other cores. The bag recorder and the perception
    pipeline get the same treatment.
    """
    cpus = LaunchConfiguration(arg).perform(context)
    return [f"taskset -c {cpus}"] if cpus else None


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
    mock_hw = hardware == "real" and (
        LaunchConfiguration("mock_hw").perform(context).lower() in ("true", "1")
    )
    # Who owns odom->base_link: the leg-kinematics + IMU odometry (the robot,
    # and the sim when a map is to inherit the odometry's honest drift) or
    # the platform's placeholder (static identity / the sim's ground truth).
    leg_odom = LaunchConfiguration("leg_odom").perform(context).lower() in ("true", "1")
    # The servo contract (gains, torque cap), the IMU switch and the bench flag
    # are the same question on both sides, so they go to both xacros. What
    # differs is what the plugin needs to reach its hardware: a CAN link and an
    # I2C address for the real drives, a physics backend for the simulated one.
    xacro_args = [
        f" kp:={pd['kp']} kd:={pd['kd']} max_torque:={drive_torque}",
        f" tau_ff:={'true' if tau_ff_on else 'false'}",
        # The head's own clamp. The real drives take the summed cap above;
        # the simulated plant clamps servo and head separately, like the
        # training sim, and needs the head's share to do it.
        f" tau_ff_scale:={tau_ff_scale if tau_ff_on else 0.0}",
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
            " mock_hw:=", LaunchConfiguration("mock_hw"),
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
            " model_xml:=" + resolve_scene(
                LaunchConfiguration("model_xml").perform(context)
            ),
            # Where the plant broadcasts the TRUE base pose. With the leg
            # odometry owning odom->base_link the truth steps aside to
            # base_link_gt, still in TF for RViz and the drift meters.
            " ground_truth_frame:=" + ("base_link_gt" if leg_odom else "base_link"),
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
    # down to ~44 Hz) while the RT loop's CPU3 idled. The cores come in as
    # launch arguments (control_cpus, policy_cpus, ui_cpus; empty = no
    # taskset, which is what the sim runs with), the service names them.
    # Measured budget that fits (2026-08-24, with perception + odometry +
    # the on-robot map):
    #   control_cpus (3): ros2_control (RT loop) + real_io      ~65%
    #   policy_cpus  (2): policy + leg_odometry                 ~65%
    #   ui_cpus    (0,1): robot_state_publisher, pad/joy, camera driver,
    #     nav -- UI-rate, none of it control-critical, sharing with the OS.
    # SCHED_FIFO keeps the control loop preemptive over its core-mate
    # either way.

    nodes = [
        # The control loop; the service says control_cpus:=3.
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            prefix=_cpu_prefix(context, "control_cpus"),
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
            prefix=_cpu_prefix(context, "ui_cpus"),
        ),
        # The odom->base_link edge has exactly one owner per run:
        #   leg_odom:=true   leg_odometry (the robot's default; needs the IMU)
        #   otherwise        the sim's physics ground truth (hw:=mujoco), or
        #                    a static identity so a bench / mock run still
        #                    renders in RViz.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["--frame-id", "odom", "--child-frame-id", "base_link"],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration("hw"), "' != 'mujoco' and not ", str(leg_odom),
            ])) if hardware == "sim" else UnlessCondition(PythonExpression([
                "'", use_imu, "' == 'true' and ", str(leg_odom),
            ])),
        ),
        # Leg-kinematics + IMU odometry. On the robot itself: autonomy keeps
        # no PC in the loop, and the SLAM/nav consume wojtek/odom locally.
        # In the sim on request (leg_odom:=true), so a map built on top of
        # it inherits the odometry's real drift instead of the ground truth
        # -- the same node, the same parameters, the same TF edge.
        Node(
            package="wojtek_odometry",
            executable="leg_odometry_node",
            output="screen",
            # Shares the policy's core: the budget above was measured with
            # the two together there.
            prefix=_cpu_prefix(context, "policy_cpus"),
            # input_stride 2: the abs joint stream arrives at ~50 Hz
            # (joint_state_broadcaster's rate), and the per-message
            # kinematics costs ~6 ms on the robot's A72 -- 25 Hz processing
            # fits the core budget; full rate does not (see the node).
            parameters=[{
                "publish_tf": True,
                "input_stride": 2,
                "imu_mount_rpy": IMU_MOUNT_RPY,
            }],
            condition=IfCondition(use_imu),
        ) if leg_odom else None,
        Node(
            package="wojtek_bringup",
            executable="real_io_node",
            # With the control loop, not the policy: the policy's core also
            # carries the leg odometry now (see the budget above). Before
            # the odometry landed this sat on policy_cpus; re-measure if the
            # loop's core shows overruns.
            prefix=_cpu_prefix(context, "control_cpus"),
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
            prefix=_cpu_prefix(context, "policy_cpus"),
            output="screen",
            parameters=[
                {
                    # Already-resolved local dir + its provenance, so the node
                    # loads the same files without resolving the ref again.
                    "policy": str(loaded.directory),
                    "policy_source": loaded.source,
                    # The driver publishes the chip axes unmodified; this
                    # rotates them into base_link (see IMU_MOUNT_RPY).
                    "imu_mount_rpy": IMU_MOUNT_RPY,
                    "auto_enable": True,  # real_io arming is the gate
                    "soft_start_s": 2.0,
                    "clamp_knee": True,
                    "watchdog_timeout_s": 0.2,
                    # Same switch as the sysinfo node above, so one argument
                    # turns both topics on together.
                    "publish_timing": ParameterValue(
                        LaunchConfiguration("telemetry"), value_type=bool
                    ),
                }
            ],
            # IMU needs no remap: policy_node subscribes the broadcaster's
            # topic name directly (the sim publishes the same name).
            remappings=[
                ("joint_states", "wojtek/joint_states_abs"),
            ],
        ),
        # How the computer itself is doing: CPU, memory, SoC temperature,
        # throttling, free space and wifi traffic on /wojtek/sysinfo. It
        # reports free space on the disk the bag goes to, so it takes
        # bag_dir.
        Node(
            package="wojtek_telemetry",
            executable="sysinfo_node",
            output="screen",
            parameters=[{"disk_path": LaunchConfiguration("bag_dir")}],
            prefix=_cpu_prefix(context, "sysinfo_cpus"),
            condition=IfCondition(LaunchConfiguration("telemetry")),
        ),
    ]
    # Sim-only entries resolve to None on the other hardware (and vice
    # versa); drop them instead of handing launch a None action.
    nodes = [n for n in nodes if n is not None]

    # A websocket bridge on the robot itself, so watching a run in Foxglove
    # needs nothing running on the PC. The bridge is a separate apt package.
    # When it is missing, log a line and bring the rest up anyway.
    if LaunchConfiguration("foxglove").perform(context).lower() in ("true", "1"):
        try:
            get_package_share_directory("foxglove_bridge")
        except PackageNotFoundError:
            print(">> foxglove_bridge is not installed -- no live Foxglove "
                  "link (apt install ros-$ROS_DISTRO-foxglove-bridge)")
        else:
            nodes.append(
                Node(
                    package="foxglove_bridge",
                    executable="foxglove_bridge",
                    parameters=[{
                        "port": 8765,
                        # Never the raw colour image: one viewer panel on it
                        # pulled ~19 MB/s through DDS and eth0, saturated
                        # cores 0,1 and stretched the policy's output gaps
                        # from 23 to 111 ms (ros/hw_tests/perf). The JPEG
                        # (<topic>/compressed) and depth stay available;
                        # the depth's own /compressed (a lossy 8-bit JPEG of
                        # 16-bit millimetres) is useless and costs the Pi an
                        # encode per frame, so it goes too.
                        "topic_whitelist": [
                            r"^(?!/camera/camera/(color/image_raw"
                            r"|depth/image_rect_raw/compressed)$).*$"
                        ],
                    }],
                    prefix=_cpu_prefix(context, "foxglove_cpus"),
                    output="screen",
                )
            )

    if with_rviz:
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", LaunchConfiguration("rviz_config")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            )
        )

    # The deck panel's robot-side gateway (wojtek_deck): the page, the
    # command websocket with the dead-man, the MJPEG camera stream. Gets
    # the resolved policy directory so its command box matches policy_node.
    # deck_cpus keeps the JPEG encoder off the isolated RT cores on the RPi.
    deck_cpus = LaunchConfiguration("deck_cpus").perform(context)
    # The gateway receives the raw camera image. At 640x480 a frame is
    # 0.9 MB, hundreds of UDP fragments through the loopback, and with
    # Cyclone's default socket buffer one lost fragment discards the whole
    # frame: under load the gateway then sees no frames at all. So it gets
    # its own, bigger receive buffer, appended to whatever Cyclone config
    # the process already has (the robot's pins its interfaces there).
    # `max` asks for 8 MB and keeps what the kernel grants; `min` would
    # make it a requirement, and Cyclone refuses to start when the kernel
    # cannot meet one. The robot's install.sh raises rmem_max so it gets
    # the full 8 MB; the sim and an unprovisioned box get the kernel's cap
    # and a log line instead of a gateway that never comes up.
    cyclone_base = os.environ.get("CYCLONEDDS_URI", "")
    cyclone_uri = (cyclone_base + "," if cyclone_base else "") + (
        "<CycloneDDS><Domain><Internal>"
        '<SocketReceiveBufferSize max="8MB"/>'
        "</Internal></Domain></CycloneDDS>"
    )
    nodes.append(
        Node(
            package="wojtek_deck",
            executable="deck_gateway",
            output="screen",
            condition=IfCondition(LaunchConfiguration("deck")),
            prefix=f"taskset -c {deck_cpus}" if deck_cpus else None,
            additional_env={"CYCLONEDDS_URI": cyclone_uri},
            parameters=[
                {
                    "policy": str(loaded.directory),
                    "port": ParameterValue(
                        LaunchConfiguration("deck_port"), value_type=int
                    ),
                    # The panel's restart button restarts this unit; the
                    # simulation has none, so there the button stays off.
                    # So does a mock_hw stack (ros/hw_tests/perf): it does
                    # not run in the boot service, and restarting that one
                    # from here would start the real drivers next to it.
                    "stack_unit": (
                        "wojtek-robot.service"
                        if hardware == "real" and not mock_hw else ""
                    ),
                    # Frames a second the gateway passes on to the panel.
                    "stream_hz": ParameterValue(
                        LaunchConfiguration("deck_stream_hz"), value_type=float
                    ),
                }
            ],
        )
    )

    # The colour camera for the panel, on the robot only (the simulation
    # renders its own). Colour plus depth at the sensor's own 424x240 is
    # the set measured safe on the Pi; no point cloud, no sync, no align:
    # the perception stack's d435.yaml turns those on together and the
    # RealSense node dies with SIGSEGV the moment the RGB sensor starts
    # (see ros/deploy/deck/README.md). The depth is the raw stream, without
    # d435.yaml's preset and temporal filter. 640x480 rather than the sensor's
    # 1280x720 is the Pi's budget: at full size the camera node and the
    # gateway starved the control loop until the drives dropped to idle.
    # initial_reset: a D435 that comes up publishing nothing (seen after a
    # power cycle) is cured by resetting it before the streams start.
    if hardware == "real":
        nodes.append(
            Node(
                package="realsense2_camera",
                executable="realsense2_camera_node",
                namespace="camera",
                name="camera",
                output="screen",
                condition=IfCondition(LaunchConfiguration("deck_camera")),
                prefix=f"taskset -c {deck_cpus}" if deck_cpus else None,
                parameters=[
                    {
                        "initial_reset": True,
                        # Depth rides in the same node when asked for (the
                        # off-board VLM's map): the sensor's own low
                        # resolution, so the Pi does no decimation, and
                        # no sync/align/cloud -- those stay off-board.
                        "enable_depth": ParameterValue(
                            LaunchConfiguration("deck_camera_depth"), value_type=bool
                        ),
                        "depth_module.depth_profile": LaunchConfiguration(
                            "deck_camera_depth_profile"
                        ),
                        # Beyond 3 m the D435's depth is noise for the map
                        # (docs/perception/d435-noise.md) and a far or
                        # invalid return can come out as 65535: zero it in
                        # the driver, which also keeps a viewer's depth scale
                        # sane.
                        "clip_distance": 3.0,
                        "enable_infra1": False,
                        "enable_infra2": False,
                        "enable_color": True,
                        "rgb_camera.color_profile": LaunchConfiguration(
                            "deck_camera_profile"
                        ),
                        "pointcloud.enable": False,
                        "align_depth.enable": False,
                        "enable_rgbd": False,
                        "enable_sync": False,
                        # The compressed transport plugin encodes the JPEG
                        # the gateway streams, in C++ and only while the
                        # gateway subscribes. This is the plugin's quality
                        # parameter as the node declares it (the leading
                        # dot is image_transport's naming); 95, its
                        # default, made 120 KB frames, 80 makes 40 KB.
                        ".camera.color.image_raw.compressed.jpeg_quality": 80,
                    }
                ],
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
        # /wojtek/sysinfo and /wojtek/policy_timing: the state of the
        # computer and the cost of each control tick. One switch for both, so
        # a run either has the whole picture or none of it. Off for manual
        # runs, same as the recorder; the RPi service opts in with
        # telemetry:=true (wojtek-robot.service).
        DeclareLaunchArgument("telemetry", default_value="false"),
        # Cores for the system-info node. It reads a handful of counters a
        # few times a second. That is small, but it still has no business on
        # the isolated RT cores, so this one defaults to 0,1. Empty = inherit.
        DeclareLaunchArgument("sysinfo_cpus", default_value="0,1"),
        # foxglove_bridge on port 8765, so the native Foxglove app connects
        # to the robot directly. Off for manual runs, like the recorder and
        # the telemetry above. The RPi service opts in with foxglove:=true.
        # A simulation session gets its bridge from viz.launch.py instead,
        # and two of them would fight over the port.
        DeclareLaunchArgument("foxglove", default_value="false"),
        DeclareLaunchArgument("foxglove_cpus", default_value="0,1"),
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
            # Motors unpowered, computer under test: the drives and the IMU
            # become ros2_control's GenericSystem while every node, rate and
            # core pin stays as it is on the robot. For profiling the RPi
            # (ros/hw_tests/perf), not for driving.
            DeclareLaunchArgument("mock_hw", default_value="false"),
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
            # (scene_sim.xml shipped by wojtek_pc: the training scene plus
            # the props the camera and its detector have something to see in).
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
            # Cores for the joy driver and the teleop node (the include takes
            # them as `cpus`, the same contract as perception and nav).
            # Empty = wherever the tree runs; the service says 0,1, because
            # on the isolated RT cores with no load balancing they shared
            # one core with policy_node and real_io and took a fifth of it.
            DeclareLaunchArgument("gamepad_cpus", default_value=""),
            IncludeLaunchDescription(
                PathJoinSubstitution(
                    [
                        FindPackageShare("wojtek_teleop"),
                        "launch",
                        "gamepad.launch.py",
                    ]
                ),
                launch_arguments={
                    "cpus": LaunchConfiguration("gamepad_cpus"),
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
                # 6 fps on both streams: the depth consumer is the SLAM,
                # which keys at ~2 Hz, and the driver's post-processing at
                # 15 fps alone saturated a Pi 4 core. Colour rides along:
                # the RGBD product pairs depth with colour, so their rates
                # must match, and the VLM decides at ~0.3-0.5 Hz anyway.
                "depth_profile": "848x480x6",
                "color_profile": "1280x720x6",
            }.items(),
            condition=IfCondition(LaunchConfiguration("perception")),
        ),
        # Owner of odom->base_link, see _launch_setup. The robot's default is
        # its own odometry; the sim's is the ground truth, until a run wants
        # the odometry's drift in the picture (the SLAM sessions do).
        DeclareLaunchArgument(
            "leg_odom", default_value="true" if hardware == "real" else "false",
        ),
        # Local perception for navigation (wojtek_nav): the rolling costmap
        # around the robot in odom, from the depth camera. Off by default,
        # like the camera it needs (perception:=true on the robot, the
        # virtual camera in the sim) and the odometry under it
        # (leg_odom:=true in the sim).
        DeclareLaunchArgument("nav", default_value="false"),
        DeclareLaunchArgument(
            "nav_cpus", default_value="0,1" if hardware == "real" else "",
        ),
        IncludeLaunchDescription(
            PathJoinSubstitution(
                [FindPackageShare("wojtek_nav"), "launch", "costmap.launch.py"]
            ),
            launch_arguments={"cpus": LaunchConfiguration("nav_cpus")}.items(),
            condition=IfCondition(LaunchConfiguration("nav")),
        ),
        # The deck panel (wojtek_deck): a browser cockpit for a handheld on
        # the robot's wifi. On in the simulation (open http://localhost:8090),
        # opt-in on the robot (deck:=true deck_cpus:=0,1 in the service).
        DeclareLaunchArgument(
            "deck", default_value="true" if hardware == "sim" else "false",
        ),
        DeclareLaunchArgument("deck_port", default_value="8090"),
        DeclareLaunchArgument("deck_cpus", default_value=""),
        # The panel's colour camera, robot only (deck_camera:=true in the
        # service). Profile WxHxFPS. 640x480 is what the Pi affords next to
        # the control loop; with the camera node doing the JPEG itself
        # (compressed transport) 30 fps fits, and deck_stream_hz is how
        # many of those the gateway passes on to the panel.
        # Where the control loop and the policy side run, as taskset
        # lists; empty = wherever the tree runs. The RPi service pins the
        # controller to 3 and policy_node + real_io to 2 (see the nodes).
        DeclareLaunchArgument("control_cpus", default_value=""),
        DeclareLaunchArgument("policy_cpus", default_value=""),
        # UI-rate nodes of the tree itself (robot_state_publisher). The
        # robot sends them to the system cores; the sim inherits.
        DeclareLaunchArgument(
            "ui_cpus", default_value="0,1" if hardware == "real" else "",
        ),
        DeclareLaunchArgument("deck_camera", default_value="false"),
        DeclareLaunchArgument("deck_camera_profile", default_value="640x480x30"),
        # Depth from the same camera node, for an off-board consumer (the
        # VLM's map). On with the camera; profile WxHxFPS at the sensor.
        # 424x240 is the camera contract the simulation renders to
        # (wojtek_pc.camera_spec), native on the D435 so the Pi does no
        # decimation.
        DeclareLaunchArgument("deck_camera_depth", default_value="true"),
        DeclareLaunchArgument("deck_camera_depth_profile", default_value="424x240x15"),
        DeclareLaunchArgument("deck_stream_hz", default_value="30.0"),
        OpaqueFunction(
            function=_launch_setup,
            kwargs={"with_rviz": with_rviz, "hardware": hardware},
        ),
    ]
    return LaunchDescription(args)
