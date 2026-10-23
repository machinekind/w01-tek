"""Local perception: the rolling costmap around the robot, from the depth camera.

    ros2 launch wojtek_nav costmap.launch.py
    ros2 launch wojtek_nav costmap.launch.py decimation:=2
    ros2 launch wojtek_nav costmap.launch.py --show-args

Runs standalone (against a running camera + odometry) and is meant to be
included by the robot/sim bringup (nav:=true), which passes the CPU pin.

    depth image ──► crop_decimate ──► point_cloud_xyz ──► nav2_costmap_2d ──► /wojtek/nav/costmap ──┐
    (424x240)       (every Nth pixel)  /wojtek/nav/points   (rolling, odom)     /wojtek/nav/voxel_grid │
    TF odom->base_link (leg_odometry), base_link->camera (URDF / driver) ──┘                          ▼
    /wojtek/nav/goal (PoseStamped, the VLM's next setpoint) ─────────────────────────────► goto_node ──► /cmd_vel
                                                                                            /wojtek/nav/status

goto_node (goto:=true, the default) is the costmap's consumer: it walks
straight at the setpoint and stops when the line ahead is blocked -- the
strategy is the VLM's, the veto is the robot's. See wojtek_nav/goto.py.

Why three nodes and not one: each is a stock C++ node doing one thing, and
the expensive one -- deprojecting depth into points -- is C++ for that
reason (the rclpy version of this, the old cloud accumulator, cost most of
an RPi core). Decimation comes first because the costmap ray-traces a line
through its 3D grid for EVERY point it is given: 424x240 is 100k rays a
frame, 106x60 is 6k, and at a 5 cm cell the 6k already over-sample it.

The costmap node is nav2's own, unchanged, configured from config/costmap.yaml
(what the map is for, and what every number follows from, is argued there).
It autostarts through its lifecycle; there is no lifecycle manager, because
there is one node.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "wojtek_nav"
NAMESPACE = "wojtek/nav"


def _setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    # Optional CPU affinity, the same treatment the camera driver gets from
    # the robot bringup: off the isolated RT cores.
    cpus = arg("cpus")
    prefix = [f"taskset -c {cpus}"] if cpus else None

    decimation = int(arg("decimation"))
    if decimation < 1:
        raise ValueError(f"decimation must be >= 1, got {decimation}")

    depth_topic, info_topic = arg("depth_topic"), arg("depth_info_topic")
    actions = []

    if decimation > 1:
        # Every Nth pixel in both axes, no interpolation: a depth image is
        # not a picture, averaging two depths across an edge invents a
        # point in mid-air.
        actions.append(
            Node(
                package="image_proc",
                executable="crop_decimate_node",
                name="depth_decimate",
                namespace=NAMESPACE,
                parameters=[{
                    "decimation_x": decimation,
                    "decimation_y": decimation,
                    "interpolation": 0,
                }],
                # image_transport derives a camera's info topic from its
                # image topic's directory (<dir>/camera_info), so the pair
                # must be published as <dir>/image + <dir>/camera_info or
                # the deprojector waits forever for a sibling that never
                # comes.
                remappings=[
                    ("in/image_raw", depth_topic),
                    ("in/camera_info", info_topic),
                    ("out/image_raw", f"/{NAMESPACE}/depth/image"),
                    ("out/camera_info", f"/{NAMESPACE}/depth/camera_info"),
                ],
                prefix=prefix,
                output="screen",
            )
        )
        depth_topic, info_topic = f"/{NAMESPACE}/depth/image", f"/{NAMESPACE}/depth/camera_info"

    actions += [
        Node(
            package="depth_image_proc",
            executable="point_cloud_xyz_node",
            name="depth_cloud",
            namespace=NAMESPACE,
            remappings=[
                ("image_rect", depth_topic),
                ("camera_info", info_topic),
                ("points", arg("points_topic")),
            ],
            prefix=prefix,
            output="screen",
        ),
        Node(
            package="nav2_costmap_2d",
            executable="nav2_costmap_2d",
            # The executable names its node "costmap" itself; the namespace
            # is ours, so its topics land under /wojtek/nav/.
            namespace=NAMESPACE,
            parameters=[arg("params_file")],
            prefix=prefix,
            output="screen",
        ),
    ]
    if arg("goto").lower() in ("true", "1"):
        actions.append(
            Node(
                package=PKG,
                executable="goto_node",
                output="screen",
                parameters=[{"goal_timeout": float(arg("goal_timeout"))}],
                prefix=prefix,
            )
        )
        # The VLM's pixel -> goto's setpoint (wojtek_nav/pixel_goal.py). It
        # reads the same raw depth stream the costmap does and hands goto a
        # setpoint in odom, so it belongs to goto's session, not the map's.
        actions.append(
            Node(
                package=PKG,
                executable="pixel_goal_node",
                output="screen",
                parameters=[{
                    "depth_topic": arg("depth_topic"),
                    "depth_info_topic": arg("depth_info_topic"),
                }],
                prefix=prefix,
            )
        )
    return actions


def generate_launch_description():
    share = get_package_share_directory(PKG)
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file", default_value=f"{share}/config/costmap.yaml",
                description="Costmap settings; see the file's commentary.",
            ),
            DeclareLaunchArgument(
                "depth_topic", default_value="/camera/camera/depth/image_rect_raw",
                description="The RAW depth (full 90 deg field of view), not "
                            "the colour-aligned one.",
            ),
            DeclareLaunchArgument(
                "depth_info_topic", default_value="/camera/camera/depth/camera_info",
            ),
            DeclareLaunchArgument(
                "points_topic", default_value=f"/{NAMESPACE}/points",
                description="The point cloud the costmap consumes (the "
                            "observation sources in the params file name "
                            "it too).",
            ),
            DeclareLaunchArgument(
                "decimation", default_value="4",
                description="Keep every Nth depth pixel per axis before "
                            "deprojection; 1 = none. 4 turns 424x240 into "
                            "106x60, ~6k points -- the costmap ray-traces "
                            "each one.",
            ),
            DeclareLaunchArgument(
                "goto", default_value="true",
                description="Run goto_node, the setpoint driver on the "
                            "costmap (wojtek/nav/goal -> cmd_vel).",
            ),
            DeclareLaunchArgument(
                "goal_timeout", default_value="3.0",
                description="Dead-man (s): a setpoint older than this "
                            "stops the robot; the VLM must keep talking.",
            ),
            DeclareLaunchArgument(
                "cpus", default_value="",
                description="CPU affinity for all three nodes (comma list); "
                            "empty = inherit.",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
