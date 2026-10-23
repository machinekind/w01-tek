"""The navigation stack around Wojtek: depth -> scan, odom relay, slam_toolbox,
Nav2 (planner, controller, behaviors, bt_navigator, velocity smoother) and the
cmd_vel watchdog. Runs in the wojtek_nav container:

    ./experiments/wojtek_rai_v2/run.sh nav launch                 # sim (wall clock)
    ./experiments/wojtek_rai_v2/run.sh nav launch target:=real    # physical robot
    ./experiments/wojtek_rai_v2/run.sh nav launch slam:=false map:=/exp/runs/map.yaml

Topic contract (the robot/sim side, unchanged):
  in   /camera/camera/depth/image_rect_raw + camera_info, TF odom->base_link, /odom_vel
  out  /cmd_vel (through the watchdog only)

`target` gates what only the sim can provide. The MuJoCo plant broadcasts a
ground-truth odom->base_link and /odom_vel; the relay turns those into /odom
and the launch pins map->odom to identity. The physical robot has neither:
its odom->base_link is a static identity (wojtek_bringup) and /odom_vel does
not exist, so relaying it would fabricate a frozen /odom and Nav2 would drive
blind. With target:=real the relay and the static map->odom are not started,
slam_toolbox publishes map->odom itself from scan matching, and /odom plus a
dynamic odom->base_link must come from an external odometry node
(rgbd_odometry / EKF, plan item N4). A preflight process checks that odometry
is live before Nav2 is brought up; if it is not, the launch shuts down.
"""

from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    RegisterEventHandler,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import AndSubstitution, EqualsSubstitution, LaunchConfiguration
from launch_ros.actions import Node

CONFIG = "/exp/wojtek_rai/nav/config"

# Real robot: slam_toolbox must own map->odom (no static identity to lean on)
# and correct the drifting external odometry from the scan.
SLAM_REAL_OVERRIDES = {"transform_publish_period": 0.02, "use_scan_matching": True}

# The bounded odometry check (see preflight.py) must finish before Nav2 is
# brought up in real mode.
PREFLIGHT_CMD = ["python3", "-m", "wojtek_rai.nav.preflight"]


def _lifecycle_manager(name: str, node_names: list[str], common: dict, condition, **extra) -> Node:
    return Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name=name,
        parameters=[common, {"autostart": True, "node_names": node_names, **extra}],
        output="screen",
        condition=condition,
    )


def generate_launch_description() -> LaunchDescription:
    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = LaunchConfiguration("params_file")
    slam = LaunchConfiguration("slam")
    map_yaml = LaunchConfiguration("map")
    target = LaunchConfiguration("target")

    is_sim = EqualsSubstitution(target, "sim")
    is_real = EqualsSubstitution(target, "real")
    common = {"use_sim_time": use_sim_time}
    lifecycle_nodes = [
        "controller_server",
        "smoother_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "velocity_smoother",
    ]

    # The camera publishes best-effort; depthimage_to_laserscan subscribes
    # reliable. The relay bridges the QoS on /nav/depth/*.
    depth_relay = ExecuteProcess(
        cmd=["python3", "-m", "wojtek_rai.nav.depth_relay"], cwd="/exp", output="screen"
    )
    # depth -> point cloud -> planar scan sliced by height in base_link. The
    # camera looks 15 deg down, so a row-based depth->scan would report the
    # floor as a wall 0.8 m ahead; the height slice drops it.
    depth_to_cloud = Node(
        package="depth_image_proc",
        executable="point_cloud_xyz_node",
        name="depth_to_cloud",
        parameters=[{"queue_size": 4}, common],
        remappings=[
            ("image_rect", "/nav/depth/image"),
            ("camera_info", "/nav/depth/camera_info"),
            ("points", "/nav/points"),
        ],
        output="screen",
    )
    depth_to_scan = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        parameters=[os.path.join(CONFIG, "cloud_to_scan.yaml"), common],
        remappings=[("cloud_in", "/nav/points"), ("scan", "/scan")],
        output="screen",
    )

    # Sim only: /odom from the ground-truth TF and /odom_vel. On the real
    # robot odom->base_link is static and /odom_vel absent, so the relay
    # would publish a frozen pose at the origin; it is not started there.
    odom_relay = ExecuteProcess(
        cmd=["python3", "-m", "wojtek_rai.nav.odom_relay"],
        cwd="/exp",
        output="screen",
        condition=IfCondition(is_sim),
    )
    watchdog = ExecuteProcess(
        cmd=["python3", "-m", "wojtek_rai.nav.cmd_vel_watchdog"], cwd="/exp", output="screen"
    )

    # slam_toolbox is a lifecycle node in Jazzy: it does nothing until a
    # lifecycle manager configures and activates it.
    slam_node_sim = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        parameters=[os.path.join(CONFIG, "slam_toolbox.yaml"), common],
        output="screen",
        condition=IfCondition(AndSubstitution(slam, is_sim)),
    )
    # Real: the same config with map->odom publishing and scan matching on
    # (slam_toolbox.yaml turns both off for the sim's ground-truth odometry).
    slam_node_real = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        parameters=[os.path.join(CONFIG, "slam_toolbox.yaml"), common, SLAM_REAL_OVERRIDES],
        output="screen",
        condition=IfCondition(AndSubstitution(slam, is_real)),
    )
    # Sim: map == odom by construction (ground-truth odometry, no scan
    # matching). A static identity keeps the map frame alive regardless of
    # slam_toolbox's scan processing. Never on the real robot: there
    # slam_toolbox owns map->odom (transform_publish_period > 0 above).
    static_map_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom_static",
        arguments=["--frame-id", "map", "--child-frame-id", "odom"],
        output="screen",
        condition=IfCondition(is_sim),
    )
    # slam_toolbox has no bond heartbeat; bond_timeout 0 disables the check.
    slam_manager = _lifecycle_manager(
        "lifecycle_manager_slam", ["slam_toolbox"], common, IfCondition(slam), bond_timeout=0.0
    )
    # Static map instead of SLAM: map_server + AMCL.
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        parameters=[{"yaml_filename": map_yaml}, common],
        output="screen",
        condition=UnlessCondition(slam),
    )
    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        parameters=[params_file, common],
        output="screen",
        condition=UnlessCondition(slam),
    )
    localization_manager = _lifecycle_manager(
        "lifecycle_manager_localization", ["map_server", "amcl"], common, UnlessCondition(slam)
    )

    # Nav2 core. The controller's raw output goes through the velocity smoother
    # to /cmd_vel_nav; only the watchdog writes /cmd_vel.
    controller = Node(
        package="nav2_controller",
        executable="controller_server",
        parameters=[params_file, common],
        remappings=[("cmd_vel", "/cmd_vel_raw")],
        output="screen",
    )
    smoother = Node(
        package="nav2_smoother",
        executable="smoother_server",
        parameters=[params_file, common],
        output="screen",
    )
    planner = Node(
        package="nav2_planner",
        executable="planner_server",
        parameters=[params_file, common],
        output="screen",
    )
    behaviors = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        parameters=[params_file, common],
        remappings=[("cmd_vel", "/cmd_vel_nav")],
        output="screen",
    )
    bt_navigator = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        parameters=[params_file, common],
        output="screen",
    )
    velocity_smoother = Node(
        package="nav2_velocity_smoother",
        executable="velocity_smoother",
        parameters=[params_file, common],
        remappings=[("cmd_vel", "/cmd_vel_raw"), ("cmd_vel_smoothed", "/cmd_vel_nav")],
        output="screen",
    )
    # Sim: Nav2 is activated right away. Real: only after the preflight has
    # seen live odometry (/odom publisher, advancing stamp, odom->base_link
    # not on /tf_static); a failed preflight shuts the launch down so Nav2
    # never drives a robot it cannot track.
    navigation_manager_sim = _lifecycle_manager(
        "lifecycle_manager_navigation", lifecycle_nodes, common, IfCondition(is_sim)
    )
    navigation_manager_real = _lifecycle_manager(
        "lifecycle_manager_navigation", lifecycle_nodes, common, None
    )
    preflight = ExecuteProcess(
        cmd=PREFLIGHT_CMD, cwd="/exp", output="screen", condition=IfCondition(is_real)
    )

    def _after_preflight(event, context):  # noqa: ARG001 -- launch passes the context
        if event.returncode == 0:
            return [navigation_manager_real]
        reason = f"nav preflight failed (exit {event.returncode}): not bringing up Nav2"
        return [LogInfo(msg=reason), EmitEvent(event=Shutdown(reason=reason))]

    gate_navigation_on_preflight = RegisterEventHandler(
        OnProcessExit(target_action=preflight, on_exit=_after_preflight)
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("params_file", default_value=os.path.join(CONFIG, "nav2.yaml")),
            DeclareLaunchArgument("slam", default_value="true"),
            DeclareLaunchArgument("map", default_value="/exp/runs/map.yaml"),
            DeclareLaunchArgument(
                "target",
                default_value="sim",
                choices=["sim", "real"],
                description="sim: relay ground-truth odometry and pin map->odom; "
                "real: external odometry required, checked by the preflight.",
            ),
            static_map_tf,
            depth_relay,
            depth_to_cloud,
            depth_to_scan,
            odom_relay,
            watchdog,
            slam_node_sim,
            slam_node_real,
            slam_manager,
            map_server,
            amcl,
            localization_manager,
            controller,
            smoother,
            planner,
            behaviors,
            bt_navigator,
            velocity_smoother,
            gate_navigation_on_preflight,
            preflight,
            navigation_manager_sim,
        ]
    )
