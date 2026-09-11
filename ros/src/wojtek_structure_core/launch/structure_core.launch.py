"""Occipital Structure Core (STO2D-C) as the targeting camera.

    ros2 launch wojtek_structure_core structure_core.launch.py
    ros2 launch wojtek_structure_core structure_core.launch.py cpus:=0,1
    ros2 launch wojtek_structure_core structure_core.launch.py \
        serial_number:=<exact serial> depth_range_mode:=Short

Runs standalone and is meant to be included by a targeting bringup. It owns
the sensor and the two optical-frame TF edges, and nothing else: the mount
transform (base_link -> targeting_camera_link) is a singleton that belongs to
whoever owns the robot, exactly as wojtek_perception_bringup keeps its
extrinsics opt-out.

ON CPU AFFINITY -- read this before copying a number out of another launch
file. Cores 2,3 are the isolcpus RT cores on this robot: the service starts
the robot tree with `taskset -c 2,3`, and wojtek-affinity.sh pins
controller_manager to core 3 and the other RT threads to core 2. Non-RT work
goes to 0,1 -- that is what bag_cpus and deck_cpus use. So the value that
keeps this camera OFF the control loop is `cpus:=0,1`, and verifying it is
`taskset -p <pid>`, not `ros2 param list` (which shows parameters, and
affinity is not one).

That still leaves interrupts. isolcpus does not move them: the xHCI
interrupts this camera generates land on the isolated cores unless
`irqaffinity=` is set on the kernel cmdline, and that reads downstream as
control-loop jitter correlated with what the camera is looking at. A second
USB3 camera doubles the exposure. Check the cmdline before blaming anything
else.

NOTE ON COMPOSITION: a plain node, deliberately, for the same reason
wojtek_perception_bringup gives -- zero-copy needs two C++ nodes in one
process, the RPi hosts no container, and nothing else in this graph shares
the driver's process. The detector is a separate effort on a separate node.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    cpus = arg("cpus")
    # Empty means "inherit whatever affinity the parent has", which is what a
    # desk run wants. On the robot, pass 0,1.
    prefix = [f"taskset -c {cpus}"] if cpus else None

    return [
        Node(
            package="wojtek_structure_core",
            executable="structure_core_node",
            name="structure_core",
            prefix=prefix,
            output="screen",
            parameters=[
                {
                    "camera_name": arg("camera_name"),
                    "camera_namespace": arg("camera_namespace"),
                    "serial_number": arg("serial_number"),
                    "depth_resolution": arg("depth_resolution"),
                    "depth_range_mode": arg("depth_range_mode"),
                    "depth_framerate": float(arg("depth_framerate")),
                    "color_framerate": float(arg("color_framerate")),
                    "enable_depth": arg("enable_depth").lower() == "true",
                    "enable_color": arg("enable_color").lower() == "true",
                    "enable_infrared": arg("enable_infrared").lower() == "true",
                    "depth_min_m": float(arg("depth_min_m")),
                    "depth_max_m": float(arg("depth_max_m")),
                    "latency_reducer": arg("latency_reducer").lower() == "true",
                    "expensive_correction": (
                        arg("expensive_correction").lower() == "true"
                    ),
                }
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            # Not `camera`/`camera` -- that is the terrain D435, already live
            # on /camera/camera/... . The doubled name mirrors the RealSense
            # driver's own layout so both cameras read the same way.
            DeclareLaunchArgument("camera_name", default_value="targeting_camera"),
            DeclareLaunchArgument("camera_namespace", default_value="targeting_camera"),
            # Empty = first sensor found. Pin it once there is more than one
            # depth camera on the bus: enumeration order is a coin flip.
            DeclareLaunchArgument("serial_number", default_value=""),
            DeclareLaunchArgument(
                "depth_resolution",
                default_value="VGA",
                description="QVGA | VGA | SXGA",
            ),
            DeclareLaunchArgument(
                "depth_range_mode",
                default_value="Medium",
                description="VeryShort | Short | Medium | Long | VeryLong | Hybrid",
            ),
            DeclareLaunchArgument("depth_framerate", default_value="30.0"),
            DeclareLaunchArgument("color_framerate", default_value="30.0"),
            DeclareLaunchArgument("enable_depth", default_value="true"),
            DeclareLaunchArgument("enable_color", default_value="true"),
            # Off by default: nothing in the targeting chain consumes IR, and
            # it is another stream across DDS on a Pi that is already tight.
            DeclareLaunchArgument("enable_infrared", default_value="false"),
            DeclareLaunchArgument("depth_min_m", default_value="0.3"),
            DeclareLaunchArgument("depth_max_m", default_value="5.0"),
            # A stale frame is worse than a dropped one for an aiming loop.
            DeclareLaunchArgument("latency_reducer", default_value="true"),
            # Costs CPU the Pi may not have spare with a detector running.
            DeclareLaunchArgument("expensive_correction", default_value="false"),
            DeclareLaunchArgument(
                "cpus",
                default_value="",
                description="taskset core list. On the robot: 0,1 (2,3 are RT).",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
