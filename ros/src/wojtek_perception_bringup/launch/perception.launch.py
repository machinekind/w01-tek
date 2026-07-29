"""Perception pipeline: RealSense D435 depth -> coarse terrain reference cloud.

    ros2 launch wojtek_perception_bringup perception.launch.py
    ros2 launch wojtek_perception_bringup perception.launch.py fps:=30 enable_color:=true
    ros2 launch wojtek_perception_bringup perception.launch.py reduce:=false

Runs standalone (the line above brings up the camera, its settings and the
reduction, nothing else) and is meant to be included by the robot bringup:

    IncludeLaunchDescription(
        PythonLaunchDescriptionSource(".../perception.launch.py"),
        launch_arguments={"container": "wojtek_container"}.items(),
    )

The only thing in here that must exist exactly once is the **component
container**. Pass `container:=<name>` and this launch loads the driver into
that existing container instead of creating its own -- which is also how you
get the driver into the same process as other C++ nodes, i.e. intra-process
zero-copy for the depth frames. With `container:=''` (the default) it owns
one. Nothing else here is a singleton: the reduction works in the camera's
own frame and the extrinsics publisher is opt-out (`extrinsics:=false`) for
the case where the robot bringup already owns the TF tree.

The camera settings are in config/d435.yaml and are measured, not guessed;
that file carries the reasoning per parameter.

NOTE ON COMPOSITION: the reduction node is rclpy, and Python nodes cannot be
loaded into a component container -- only the C++ driver is composable here.
The depth image therefore still crosses DDS to reach the reduction. That is
affordable at 424x240x10 (~2 MB/s), but if the reduction moves to the full
frame rate or grows, it needs rewriting as a C++ composable node to stay on
the zero-copy path.
"""

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode

PKG = "wojtek_perception_bringup"


def _flatten(prefix, node, out):
    """YAML nesting -> flat ROS parameter names ('a': {'b': 1} -> 'a.b')."""
    for key, value in node.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            _flatten(f"{name}.", value, out)
        else:
            out[name] = value
    return out


def _setup(context, *args, **kwargs):
    def arg(name):
        return LaunchConfiguration(name).perform(context)

    with open(arg("params_file")) as fh:
        cfg = yaml.safe_load(fh)

    rs_params = _flatten("", cfg.get("realsense", {}), {})
    reduce_params = dict(cfg.get("reduce", {}))

    # Launch arguments win over the file, so a one-off experiment does not
    # need an edited config.
    fps = int(arg("fps"))
    for key in ("depth_module.depth_profile", "rgb_camera.color_profile"):
        if key in rs_params:
            width, height, _ = str(rs_params[key]).split("x")
            rs_params[key] = f"{width}x{height}x{fps}"
    rs_params["enable_color"] = arg("enable_color").lower() in ("true", "1")

    camera_name, camera_ns = arg("camera_name"), arg("camera_namespace")
    driver = ComposableNode(
        package="realsense2_camera",
        plugin="realsense2_camera::RealSenseNodeFactory",
        name=camera_name,
        namespace=camera_ns,
        parameters=[rs_params],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    actions = []
    container_name = arg("container")
    if container_name:
        # Someone else owns the container: just load into it.
        actions.append(
            LoadComposableNodes(
                target_container=container_name,
                composable_node_descriptions=[driver],
            )
        )
    else:
        actions.append(
            ComposableNodeContainer(
                name="perception_container",
                namespace="",
                package="rclcpp_components",
                executable="component_container_mt",
                composable_node_descriptions=[driver],
                output="screen",
            )
        )

    if arg("reduce").lower() in ("true", "1"):
        depth_topic = f"/{camera_ns}/{camera_name}/depth/image_rect_raw"
        info_topic = f"/{camera_ns}/{camera_name}/depth/camera_info"
        actions.append(
            Node(
                package=PKG,
                executable="cloud_reduce_node",
                name="cloud_reduce",
                parameters=[reduce_params],
                remappings=[
                    ("depth/image", depth_topic),
                    ("depth/camera_info", info_topic),
                ],
                output="screen",
            )
        )

    if arg("extrinsics").lower() in ("true", "1"):
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
                "container", default_value="",
                description="Empty: this launch creates its own component "
                            "container. A name: load the driver into that "
                            "existing container instead (shared process, "
                            "intra-process depth frames).",
            ),
            DeclareLaunchArgument(
                "params_file", default_value=f"{share}/config/d435.yaml",
                description="Camera + reduction settings. The defaults are "
                            "measured; see the file's own commentary.",
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
                "fps", default_value="10",
                description="Depth frame rate. 10 matches the planner's "
                            "replan rate; 30 costs about a whole RPi core in "
                            "post-processing for frames nobody reads.",
            ),
            DeclareLaunchArgument(
                "enable_color", default_value="false",
                description="Colour stream. Off by default: the depth "
                            "pipeline does not use it and it costs USB "
                            "bandwidth. Turn on for VLM work.",
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
