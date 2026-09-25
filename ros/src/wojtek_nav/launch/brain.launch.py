"""The VLM brain, as a launch: one node, its endpoint and its camera named once.

    ros2 launch wojtek_nav brain.launch.py url:=http://<vlm host>:8000
    ros2 launch wojtek_nav brain.launch.py url:=... instruction:="podejdź do fioletowego słupa"
    ros2 launch wojtek_nav brain.launch.py --show-args

Included by the simulation (sim.launch.py vlm:=true) and started by
`ros2 run wojtek_bringup robot --vlm` against the robot; runs standalone
against any session that has the costmap, goto and the pixel resolver up
(costmap.launch.py). The brain runs where the pictures can reach the
model -- the PC, not the RPi -- and takes the camera's own JPEG
(<image_topic>/compressed) so what crosses the robot's wifi is tens of
kilobytes a frame, not the raw image's megabytes.

`url` is the model server's base URL, with or without /v1. The default
model is Qwen3-VL-8B-Instruct on vLLM (scripts/serve_vlm.sh starts one);
the environment's VLM_URL (see .env.example) is the usual source of `url`.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from wojtek_nav.vlm_brain import DEFAULT_MODEL, DEFAULT_URL


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "url",
                default_value=EnvironmentVariable("VLM_URL", default_value=DEFAULT_URL),
                description="Base URL of the OpenAI-compatible model server "
                            "(vLLM/Ollama); VLM_URL from the environment "
                            "when set.",
            ),
            DeclareLaunchArgument(
                "model",
                default_value=EnvironmentVariable("VLM_MODEL", default_value=DEFAULT_MODEL),
                description="Model name as the server knows it (a vLLM "
                            "--served-model-name, an Ollama tag); VLM_MODEL "
                            "from the environment when set.",
            ),
            DeclareLaunchArgument(
                "api_key",
                default_value=EnvironmentVariable("VLLM_API_KEY", default_value="EMPTY"),
                description="Bearer token for the server, if it wants one.",
            ),
            DeclareLaunchArgument(
                "instruction", default_value="",
                description="A task to run at start; later ones arrive on "
                            "wojtek/vlm/instruction (the web console's brain "
                            "panel).",
            ),
            DeclareLaunchArgument(
                "image_topic", default_value="/camera/camera/color/image_raw",
                description="The colour image; with compressed:=true its "
                            "/compressed sibling is what is read.",
            ),
            DeclareLaunchArgument(
                "compressed", default_value="true",
                description="Read the camera node's own JPEG "
                            "(<image_topic>/compressed) instead of the raw "
                            "image.",
            ),
            Node(
                package="wojtek_nav",
                executable="vlm_brain_node",
                output="screen",
                # Every string pinned as str: launch_ros YAML-parses bare
                # values, so an all-digit api key or model tag would reach
                # the node as an int and its declare_parameter would abort.
                parameters=[{
                    "url": ParameterValue(LaunchConfiguration("url"), value_type=str),
                    "model": ParameterValue(LaunchConfiguration("model"), value_type=str),
                    "api_key": ParameterValue(LaunchConfiguration("api_key"), value_type=str),
                    "instruction": ParameterValue(
                        LaunchConfiguration("instruction"), value_type=str
                    ),
                    "image_topic": LaunchConfiguration("image_topic"),
                    "compressed": ParameterValue(
                        LaunchConfiguration("compressed"), value_type=bool
                    ),
                }],
            ),
        ]
    )
