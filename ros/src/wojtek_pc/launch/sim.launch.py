"""Simulation bringup: the robot's own node graph over a simulated plant.

    ros2 launch wojtek_pc sim.launch.py [hw:=mock|mujoco] [rviz:=false]
                                       [boot_pose:=folded] [camera:=false]

This is `robot.launch.py` with the hardware plugin swapped -- same
controller_manager at 400 Hz, same broadcasters, same real_io_node, same
policy_node parameters (see wojtek_bringup/launch_common.py, which both
launches share). So the startup procedure is the robot's procedure:

  1. Launch. The policy runs immediately but real_io_node starts DISARMED.
  2. ros2 service call /wojtek/zero std_srvs/srv/Trigger
  3. ros2 service call /wojtek/stand_up std_srvs/srv/Trigger
  4. ros2 service call /wojtek/arm std_srvs/srv/SetBool '{data: true}'
  5. When done: disarm, then /wojtek/lie_down.

That is the point of this launch: a run here exercises the arming path, the
zero offset, the ramps and the watchdog, so a failure shows up on the desk
instead of on the robot. What it cannot show is in docs/sim-test-contract.md.

hw:=mock (the default) runs ros2_control's GenericSystem: no dynamics, the
commands come straight back as states. Everything above still works, the
robot just cannot fall over or walk. hw:=mujoco puts the trained plant under
it (see wojtek_mujoco_hardware_interface).

boot_pose:=folded starts from the real robot's boot/zeroing pose instead of
standing, which is how you rehearse the real startup.

Drive with any Twist teleop, e.g.:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard
    ros2 launch wojtek_teleop gamepad.launch.py   # bluetooth Xbox pad

text_commander (wojtek#92) is always up: text commands on /wojtek/nav_command
(the web console's VLM panel, or `ros2 topic pub`) drive /cmd_vel with a 2 s
dead-man. Resident by design -- it publishes NOTHING until commanded and goes
silent after its single stop Twist, so it never fights the other teleops.

camera:=false turns off the D435-compatible virtual camera (on by default;
the off-switch for weak machines). It needs a physics-backed plant, so it is
inert with hw:=mock. camera_depth_hz/camera_color_hz tune the render rates.
"""

from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

from wojtek_bringup.launch_common import common_launch_description


def generate_launch_description():
    # RViz on by default (this is the desk workflow), recording opt-in, and
    # everything else -- nodes, parameters, the arming procedure -- shared
    # verbatim with the robot bringup.
    ld = common_launch_description(
        with_rviz=True, bag_default="false", hardware="sim",
    )
    for action in (
        # D435-compatible virtual camera. Declared here so the callers that
        # pass these through (sim.sh, `ros2 run wojtek_bringup robot --sim`)
        # keep working, but the renderer needs a physics state to draw from:
        # it comes back with hw:=mujoco and is inert until then.
        DeclareLaunchArgument("camera", default_value="true"),
        DeclareLaunchArgument("camera_depth_hz", default_value="15.0"),
        DeclareLaunchArgument("camera_color_hz", default_value="5.0"),
        # Text-command bridge (wojtek#92), resident by design -- see the
        # module docstring for why that is safe.
        Node(
            package="wojtek_teleop",
            executable="text_commander",
            output="screen",
        ),
    ):
        ld.add_action(action)
    return ld
