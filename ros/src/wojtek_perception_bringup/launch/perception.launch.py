"""Perception pipeline: RealSense D435 depth -> coarse terrain reference cloud.

    ros2 launch wojtek_perception_bringup perception.launch.py
    ros2 launch wojtek_perception_bringup perception.launch.py depth_profile:=848x480x30
    ros2 launch wojtek_perception_bringup perception.launch.py reduce:=false

Runs standalone (the line above brings up the camera, its settings and the
reduction, nothing else) and is meant to be included by the robot bringup:

    IncludeLaunchDescription(
        PythonLaunchDescriptionSource(".../perception.launch.py"),
        launch_arguments={"extrinsics": "false"}.items(),
    )

The only singleton in here is the camera->body static transform, which is
why it is opt-out (`extrinsics:=false`) for the case where the robot bringup
already owns that TF edge. The reduction works in the camera's own frame and
needs nothing from the rest of the graph.

NOTE ON COMPOSITION: the driver runs as a plain node, deliberately. Loading
it into a component container would buy intra-process zero-copy only if it
shared that process with another C++ node -- and it would not: the RPi hosts
no container (nothing else in this workspace creates one, and ros2_control
runs as a standalone process), and the only consumer of the depth image is
this package's rclpy reduction, which cannot be composed at all. The depth
therefore crosses DDS on loopback: decimation halves each dimension before
publishing, so that is 424x240x16 bit at 15 Hz, ~3 MB/s, which is
affordable. If the reduction is ever rewritten in C++ for a higher frame
rate, that is the moment to introduce a container -- not before.

(The colour stream is the heavier one -- 848x480 rgb8 at 6 fps is ~7 MB/s
uncompressed -- but nothing on the robot subscribes to it yet, and DDS does
not serialise a topic with no subscribers.)

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
    # this the camera driver and the reduction land on the cores the 400 Hz
    # control loop owns exclusively.
    cpus = arg("cpus")
    prefix = [f"taskset -c {cpus}"] if cpus else None

    # Single-value overrides layered on top of the camera parameter file, so
    # a one-off experiment does not need an edited config. An empty argument
    # means "whatever the file says".
    overrides = {"enable_color": arg("enable_color").lower() in ("true", "1")}
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

    if arg("reduce").lower() in ("true", "1"):
        depth_topic = f"/{camera_ns}/{camera_name}/depth/image_rect_raw"
        info_topic = f"/{camera_ns}/{camera_name}/depth/camera_info"
        actions.append(
            Node(
                package=PKG,
                executable="cloud_reduce_node",
                name="cloud_reduce",
                parameters=[arg("reduce_params_file")],
                remappings=[
                    ("depth/image", depth_topic),
                    ("depth/camera_info", info_topic),
                ],
                prefix=prefix,
                output="screen",
            )
        )

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
                "reduce_params_file",
                default_value=f"{share}/config/cloud_reduce.yaml",
                description="Grid reduction settings.",
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
                "reduce", default_value="true",
                description="Run the depth->grid reduction. Off to look at "
                            "raw depth in RViz.",
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
                description="CPU affinity for the camera driver and the "
                            "reduction (comma list, e.g. \"0,1\"); empty = "
                            "inherit. The robot bringup pins them to the "
                            "non-isolated cores so they cannot steal time "
                            "from the 400 Hz control loop.",
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
