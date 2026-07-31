"""MuJoCo sim + policy + RViz.

    ros2 launch wojtek_viz sim.launch.py [rviz:=false] [initial_pose:=folded]
                                         [camera:=false]

initial_pose:=folded spawns the robot in the real robot's boot/zeroing pose
(lying flat, knees folded -- see real_io_node) instead of standing at home.
Note the policy starts immediately, so from folded it will try to stand on
its own; use this to inspect/verify the boot pose, not for clean walking.

camera:=false turns off the D435-compatible virtual camera (on by default;
the off-switch for weak machines). camera_depth_hz/camera_color_hz tune the
render rates; see wojtek_viz/mujoco_sim_node.py for the camera contract.

Drive with any Twist teleop, e.g.:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard
    ros2 launch wojtek_teleop gamepad.launch.py   # bluetooth Xbox pad

text_commander (wojtek#92) is always up: text commands on /wojtek/nav_command
(the web console's VLM panel, or `ros2 topic pub`) drive /cmd_vel with a 2 s
dead-man. Resident by design -- it publishes NOTHING until commanded and goes
silent after its single stop Twist, so it never fights the other teleops.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("wojtek_viz")
    xacro_file = os.path.join(share, "urdf", "wojtek_sim.urdf.xacro")
    robot_description = ParameterValue(Command(f"xacro {xacro_file}"), value_type=str)

    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("initial_pose", default_value="home"),
            # D435-compatible virtual camera (default on; off-switch for
            # weak machines). Rendering runs on its own thread in the sim
            # node, so the physics real-time factor is unaffected.
            DeclareLaunchArgument("camera", default_value="true"),
            DeclareLaunchArgument("camera_depth_hz", default_value="15.0"),
            DeclareLaunchArgument("camera_color_hz", default_value="5.0"),
            # Default view includes the virtual camera's image panels;
            # override for a different layout (e.g. the plain wojtek.rviz).
            DeclareLaunchArgument(
                "rviz_config",
                default_value=os.path.join(share, "config", "sim.rviz"),
            ),
            # HF repo id (org/name[@revision]) or a local directory with
            # policy.npz + policy_meta.json -- see wojtek_policy/policy_source.py.
            DeclareLaunchArgument(
                "policy",
                default_value="<HF_ORGANIZATION>/wojtek-stiff-height-locomotion",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="wojtek_viz",
                executable="mujoco_sim_node",
                output="screen",
                parameters=[
                    {
                        "initial_pose": LaunchConfiguration("initial_pose"),
                        # Match the simulated plant (servo gains, torque cap)
                        # to the same policy contract policy_node loads.
                        "policy": LaunchConfiguration("policy"),
                        "camera": ParameterValue(
                            LaunchConfiguration("camera"), value_type=bool
                        ),
                        "camera_depth_hz": ParameterValue(
                            LaunchConfiguration("camera_depth_hz"),
                            value_type=float,
                        ),
                        "camera_color_hz": ParameterValue(
                            LaunchConfiguration("camera_color_hz"),
                            value_type=float,
                        ),
                    }
                ],
            ),
            Node(
                package="wojtek_policy",
                executable="policy_node",
                output="screen",
                parameters=[
                    {
                        "policy": LaunchConfiguration("policy"),
                        "imu_mount_rpy": [0.0, 0.0, 0.0],  # sim IMU is in base_link
                        "auto_enable": True,
                        "soft_start_s": 0.5,
                        "clamp_knee": False,
                    }
                ],
            ),
            # Text-command bridge (wojtek#92), resident by design -- see
            # the module docstring for why that is safe.
            Node(
                package="wojtek_teleop",
                executable="text_commander",
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", LaunchConfiguration("rviz_config")],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
        ]
    )
