"""Benchmark rig over a running simulation.

Companion to `ros2 launch wojtek_pc sim.launch.py` (hw:=mujoco, the
default), started in a second shell -- the rig is external instrumentation
and stays out of the robot's own bringup on purpose:

    ros2 launch wojtek_benchmark sim_rig.launch.py

Starts the rig camera (renders the injected tags + tripod view from
/sim/qpos), the tag tracker (calibrates the bench world frame from the
floor tags, then tracks the robot tag), and the error monitor (tracked
pose vs /sim/qpos ground truth on /benchmark/pose_error_mm and
/benchmark/yaw_error_deg).

The tracker's expected leg lengths -- reality's tape-measure numbers --
are DERIVED here from sim_rig.yaml placements, so the sim exercises the
same refusal gate a real course must pass, with legs that are true by
construction.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from wojtek_benchmark.sim_rig import leg_lengths, load_rig_config


def generate_launch_description():
    share = get_package_share_directory("wojtek_benchmark")
    rig_config = os.path.join(share, "config", "sim_rig.yaml")
    tags_config = os.path.join(share, "config", "apriltags.yaml")
    leg_x, leg_y = leg_lengths(load_rig_config(rig_config))

    return LaunchDescription([
        DeclareLaunchArgument("monitor", default_value="true"),
        Node(
            package="wojtek_benchmark",
            executable="benchmark_camera_node",
            output="screen",
            # Same GL contract as the other sim renderer: name the backend
            # up front or MuJoCo may abort on a missing library.
            additional_env={"MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")},
            parameters=[{
                "rig_config": rig_config,
                "tags_config": tags_config,
            }],
        ),
        Node(
            package="wojtek_benchmark",
            executable="tag_tracker_node",
            output="screen",
            parameters=[{
                "tags_config": tags_config,
                "expected_leg_x_m": leg_x,
                "expected_leg_y_m": leg_y,
            }],
        ),
        Node(
            package="wojtek_benchmark",
            executable="rig_error_monitor",
            output="screen",
            condition=IfCondition(LaunchConfiguration("monitor")),
            parameters=[{"rig_config": rig_config}],
        ),
    ])
