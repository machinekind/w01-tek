"""Bluetooth Xbox pad teleop: the standard `joy` driver + gamepad_teleop.

    ros2 launch wojtek_teleop gamepad.launch.py [device_id:=0]

Pair the pad over bluetooth on the machine this runs on first (bluetoothctl:
scan on -> pair -> trust -> connect; the pad must show up under /dev/input).
On the RPi the bluez/ERTM groundwork comes from deploy/rpi/install.sh. Runs
unchanged against the sim or the real stack -- it publishes the same
/cmd_vel the consoles do. `sim.launch.py gamepad:=true` brings this up next to
the web console; see wojtek_teleop/gamepad_teleop.py for the stick mapping.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _setup(context, *args, **kwargs):
    # Optional CPU affinity, same contract as perception.launch.py's `cpus`:
    # empty = inherit the parent's mask. The robot bringup passes a concrete
    # core so these UI-rate nodes get a deterministic spot inside the
    # isolated pair (isolcpus does not balance between the isolated cores).
    cpus = LaunchConfiguration("cpus").perform(context)
    prefix = [f"taskset -c {cpus}"] if cpus else None
    return [
        Node(
            package="joy",
            executable="joy_node",
            output="screen",
            prefix=prefix,
            parameters=[
                {
                    "device_id": ParameterValue(
                        LaunchConfiguration("device_id"), value_type=int
                    ),
                    # Keep publishing while sticks are held steady --
                    # gamepad_teleop's dead-man counts on a steady stream
                    # and must only trip when the pad actually drops off.
                    "autorepeat_rate": 20.0,
                    "deadzone": 0.05,
                }
            ],
        ),
        Node(
            package="wojtek_teleop",
            executable="gamepad_teleop",
            output="screen",
            prefix=prefix,
            parameters=[{"policy": LaunchConfiguration("policy")}],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            # Which /dev/input device the joy driver opens (0 = first pad).
            DeclareLaunchArgument("device_id", default_value="0"),
            # Policy reference whose contract sets the stick command box
            # (same reference policy_node gets); empty = the teleop node's
            # conservative default limits.
            DeclareLaunchArgument("policy", default_value=""),
            DeclareLaunchArgument("cpus", default_value=""),
            OpaqueFunction(function=_setup),
        ]
    )
