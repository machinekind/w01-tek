"""PC-side visualization/debug for the LIVE robot (running on the RPi).

    ros2 launch wojtek_pc viz.launch.py [rviz:=true] [plotjuggler:=false]
                                         [foxglove:=false] [bag:=true]

foxglove:=true starts foxglove_bridge on ws://localhost:8765 for the native
Foxglove app -- the way to get a fast UI on macOS, where RViz would have to
go through X11. Its 3D panel reads /robot_description like RViz (meshes are
served over the bridge's asset channel), and its Teleop panel publishes
/cmd_vel, replacing the drive pad of the Qt operator console.

This starts no hardware and no robot_state_publisher -- those run on the
robot (wojtek_bringup robot.launch.py on the RPi). Over the shared DDS link
this machine only consumes what the robot publishes:
  - RViz reads /robot_description (latched by the robot's RSP) and /tf.
  - PlotJuggler subscribes to the robot's topics for live plotting.

Because DDS makes every RPi-published topic visible here, `bag:=true`
records the WHOLE run from the PC -- one bag, all topics. Off by
default: bags are ~GB per session and pile up in the wojtek_bags
volume, and most viz sessions are watching, not experiments worth
keeping. Caveat when recording: the bag is only as complete as the
link. Over ethernet (docked) it is effectively lossless; over wifi --
especially once the robot is on its own AP and moving -- best-effort
topics can drop samples and a dropped link leaves a gap. The RPi service
already records every run on-robot losslessly (wojtek-robot.service
passes bag:=true bag_cpus:=0,1); a manual robot.launch.py run records
with the same flags.
Bags land in bag_dir/run_<timestamp> (bag_dir defaults to ~/wojtek_bags).

Teleop needs its own terminal (it reads the keyboard on stdin), so run it
separately rather than from here:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard

Make sure ROS_DOMAIN_ID and RMW match the robot (see docker/compose.yaml).
"""

import datetime
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    TextSubstitution,
)
from launch_ros.actions import Node


def generate_launch_description():
    policy_share = get_package_share_directory("wojtek_policy")

    # One rosbag per run, recorded from the PC: DDS makes every RPi-published
    # topic visible here, so `ros2 bag record -a` captures the whole run. Fresh
    # timestamped subdir under bag_dir per `ros2 launch` invocation.
    default_bag_dir = os.path.join(os.path.expanduser("~"), "wojtek_bags")
    bag_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bag_output = PathJoinSubstitution(
        [LaunchConfiguration("bag_dir"), TextSubstitution(text=f"run_{bag_stamp}")]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="true"),
            # Layout to load. Default = the plain live-robot view; the sim
            # bringup (robot.py --sim) passes wojtek_pc's sim.rviz, which
            # adds the virtual camera's image panels.
            DeclareLaunchArgument(
                "rviz_config",
                default_value=os.path.join(policy_share, "rviz", "wojtek.rviz"),
            ),
            DeclareLaunchArgument("plotjuggler", default_value="false"),
            DeclareLaunchArgument("foxglove", default_value="false"),
            # bag:=true records the whole run (all topics, over DDS);
            # bag_dir:=/some/path to relocate.
            DeclareLaunchArgument("bag", default_value="false"),
            DeclareLaunchArgument("bag_dir", default_value=default_bag_dir),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", LaunchConfiguration("rviz_config")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
            Node(
                package="plotjuggler",
                executable="plotjuggler",
                condition=IfCondition(LaunchConfiguration("plotjuggler")),
            ),
            # Headless websocket bridge for the native Foxglove app. On macOS
            # docker/compose.mac.yaml publishes the port to the host (Linux
            # runs network_mode:host, so it is reachable either way).
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                parameters=[{"port": 8765}],
                condition=IfCondition(LaunchConfiguration("foxglove")),
            ),
            # bash -c: mkdir the parent (rosbag2 creates the bag dir itself but
            # not missing parents), then exec the recorder. bag_dir/bag_output
            # are passed as argv ($1/$2), not spliced into the script.
            ExecuteProcess(
                condition=IfCondition(LaunchConfiguration("bag")),
                cmd=[
                    "bash", "-c",
                    'mkdir -p "$1"\n'
                    'echo ">> rosbag: recording to $2"\n'
                    'exec ros2 bag record -a -o "$2"\n',
                    "wojtek_bag_record",  # $0 (shell name in messages)
                    LaunchConfiguration("bag_dir"),  # $1
                    bag_output,  # $2
                ],
                output="screen",
            ),
        ]
    )
