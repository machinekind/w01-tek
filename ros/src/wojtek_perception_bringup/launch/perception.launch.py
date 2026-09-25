"""Perception pipeline: RealSense D435 depth (+ colour/RGBD for the VLM).

    ros2 launch wojtek_perception_bringup perception.launch.py
    ros2 launch wojtek_perception_bringup perception.launch.py depth_profile:=848x480x30

Runs standalone (the line above brings up the camera and its settings,
nothing else) and is meant to be included by the robot bringup:

    IncludeLaunchDescription(
        PythonLaunchDescriptionSource(".../perception.launch.py"),
        launch_arguments={"extrinsics": "false"}.items(),
    )

The only singleton in here is the camera->body static transform, which is
why it is opt-out (`extrinsics:=false`) for the case where the robot bringup
already owns that TF edge.

This launch owns the SENSOR only. What is built on the streams -- the
obstacle perception, the costmap -- belongs to the navigation packages,
included next to this one by the robot bringup.
(Two earlier in-package consumers are gone: the depth->grid reduction for the
SCAN-planner, 2026-08, and the odom-frame accumulated cloud, 2026-09, both
superseded by the SLAM's own map.)

NOTE ON COMPOSITION: the driver runs as a plain node, deliberately. Loading
it into a component container would buy intra-process zero-copy only if it
shared that process with another C++ node -- and it would not: the RPi hosts
no container (nothing else in this workspace creates one, and ros2_control
runs as a standalone process). The depth therefore crosses DDS on loopback:
decimation halves each dimension before publishing, so that is 424x240x16
bit at 15 Hz, ~3 MB/s, which is affordable.

(The colour stream is the heavier one -- 848x480 rgb8 at 6 fps is ~7 MB/s
uncompressed; the SLAM is its consumer when it runs.)

Both config files are ordinary ROS parameter files, loaded by the nodes
themselves. Launch arguments override single values on top of the file: each
entry of `parameters=` becomes its own --params-file in list order, so the
override that comes last wins.
"""

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = "wojtek_perception_bringup"


def _setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    camera_name, camera_ns = arg("camera_name"), arg("camera_namespace")

    # Optional CPU affinity. On the robot this matters: the bringup runs the
    # whole tree under `taskset -c 2,3` (the isolcpus RT cores), and without
    # this the camera driver lands on the cores the 400 Hz control loop owns
    # exclusively.
    cpus = arg("cpus")
    prefix = [f"taskset -c {cpus}"] if cpus else None

    # Single-value overrides layered on top of the camera parameter file, so
    # a one-off experiment does not need an edited config. An empty argument
    # means "whatever the file says".
    overrides = {
        "enable_color": arg("enable_color").lower() in ("true", "1"),
        # The colour image's JPEG sibling (image_transport's compressed
        # plugin, encoded in C++ only while somebody subscribes) is what
        # leaves the robot for the VLM brain and the web console on the PC:
        # ~100 KB a frame at 1280x720 instead of the raw 2.7 MB. 80 is the
        # quality the deck's stream uses; the plugin's default 95 triples
        # the size for nothing a VLM can see.
        f".{camera_name}.color.image_raw.compressed.jpeg_quality": 80,
    }
    for name, param in (
        ("depth_profile", "depth_module.depth_profile"),
        ("color_profile", "rgb_camera.color_profile"),
    ):
        if arg(name):
            overrides[param] = arg(name)

    actions = [
        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            name=camera_name,
            namespace=camera_ns,
            parameters=[arg("camera_params_file"), overrides],
            prefix=prefix,
            output="screen",
        )
    ]

    if arg("extrinsics").lower() in ("true", "1"):
        # Not a parameter file: static_transform_publisher takes the pose on
        # the command line, and roll/pitch/yaw is the form a mount is
        # measured in (its parameter interface is quaternion-only).
        with open(arg("extrinsics_file")) as fh:
            tf = yaml.safe_load(fh)["static_transform"]
        actions.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="camera_extrinsics",
                arguments=[
                    "--x", str(tf["x"]), "--y", str(tf["y"]), "--z", str(tf["z"]),
                    "--roll", str(tf["roll"]), "--pitch", str(tf["pitch"]),
                    "--yaw", str(tf["yaw"]),
                    "--frame-id", tf["parent_frame"],
                    "--child-frame-id", tf["child_frame"],
                ],
                output="screen",
            )
        )

    return actions


def generate_launch_description():
    share = get_package_share_directory(PKG)
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_params_file", default_value=f"{share}/config/d435.yaml",
                description="Camera settings. The defaults are measured; see "
                            "the file's own commentary.",
            ),
            DeclareLaunchArgument(
                "extrinsics_file", default_value=f"{share}/config/extrinsics.yaml",
                description="Camera->body static transform. PLACEHOLDER "
                            "values until the mount is measured.",
            ),
            DeclareLaunchArgument(
                "extrinsics", default_value="true",
                description="Publish the camera->body static transform. Off "
                            "when the robot bringup already owns that edge.",
            ),
            DeclareLaunchArgument(
                "depth_profile", default_value="",
                description="Override the depth stream, WxHxFPS (e.g. "
                            "848x480x30). Empty = the config file's 15 fps, "
                            "the sensor's nearest rate at or above the "
                            "planner's 10 Hz replan; 30 does not fit the "
                            "RPi's post-processing budget.",
            ),
            DeclareLaunchArgument(
                "color_profile", default_value="",
                description="Override the colour stream, WxHxFPS. Empty = "
                            "the config file.",
            ),
            DeclareLaunchArgument(
                "enable_color", default_value="true",
                description="Colour stream (for the VLM; the depth pipeline "
                            "does not read it). On at 6 fps, which is the "
                            "sensor's slowest. enable_color:=false drops it "
                            "when only the depth path is being worked on.",
            ),
            DeclareLaunchArgument(
                "cpus", default_value="",
                description="CPU affinity for the camera driver (comma "
                            "list, e.g. \"0,1\"); empty = inherit. The "
                            "robot bringup pins it to the non-isolated "
                            "cores so it cannot steal time from the 400 Hz "
                            "control loop.",
            ),
            DeclareLaunchArgument(
                "camera_name", default_value="camera",
                description="Driver node name.",
            ),
            DeclareLaunchArgument(
                "camera_namespace", default_value="camera",
                description="Driver namespace; topics land under "
                            "/<namespace>/<name>/.",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
