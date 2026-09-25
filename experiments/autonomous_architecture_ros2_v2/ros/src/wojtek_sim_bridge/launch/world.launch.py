"""The world box: MuJoCo sim + websocket bridge, nothing robot-bound.

Scene and policy come through the same environment variables the demo rig
uses (SCENE, WOJTEK_POLICY, WOJTEK_SPAWN, LOCAL_PLANNER), set before launch;
the node reads them when it builds the sim. Only transport knobs are ROS
parameters.

    SCENE=flat ros2 launch wojtek_sim_bridge world.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = [
    ("ws_port", "8010", "browser/scenario websocket port"),
    ("ws_host", "0.0.0.0", "websocket bind address"),
]


def generate_launch_description():
    args = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in ARGS]
    cfg = {n: LaunchConfiguration(n) for n, _d, _h in ARGS}

    return LaunchDescription(args + [
        Node(package="wojtek_sim_bridge", executable="sim_bridge",
             parameters=[{
                 "ws_port": cfg["ws_port"],
                 "ws_host": cfg["ws_host"],
             }]),
    ])
