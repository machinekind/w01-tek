"""Simulation bringup: the robot's own node graph over a simulated plant.

Everything a simulation session needs, from one command inside the container
(./dev.sh gets you the shell): the plant, the control stack, the policy, the
virtual camera, RViz, the operator console and optionally a gamepad.

    ros2 launch wojtek_pc sim.launch.py [hw:=mock|mujoco] [rviz:=false]
                                       [boot_pose:=folded] [camera:=false]
                                       [console:=web|qt|none] [gamepad:=true]
                                       [telemetry:=true] [deck:=false]
                                       [leg_odom:=true nav:=true]
                                       [vlm:=true vlm_url:=http://host:8000 vlm_model:=...]

This is `robot.launch.py` with the hardware plugin swapped -- same
controller_manager at 200 Hz, same broadcasters, same real_io_node, same
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

hw:=mujoco (the default) is the trained plant, physics included, from
wojtek_mujoco_hardware_interface. hw:=mock swaps in ros2_control's
GenericSystem: no dynamics, commands come straight back as states -- the
arming path above still works, the robot just cannot fall over or walk, which
makes it the fast option for testing the stack's own logic.

boot_pose:=folded starts from the real robot's boot/zeroing pose instead of
standing, which is how you rehearse the real startup.

Drive from the operator console (console:=web, the default, on
http://localhost:8080), from a pad with gamepad:=true, or from any Twist
teleop in a second container shell:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard

text_commander (wojtek#92) is always up: text commands on /wojtek/nav_command
(the web console's VLM panel, or `ros2 topic pub`) drive /cmd_vel with a 2 s
dead-man. Resident by design -- it publishes NOTHING until commanded and goes
silent after its single stop Twist, so it never fights the other teleops.

deck:=true (the default here) also serves the deck panel on
http://localhost:8090 -- the handheld cockpit from wojtek_deck: camera,
pad, and charts read from the Foxglove bridge (viz.launch.py foxglove:=true).

camera:=false turns off the D435-compatible virtual camera (on by default;
the off-switch for weak machines). It needs a physics-backed plant, so it is
inert with hw:=mock. camera_depth_hz/camera_color_hz tune the render rates.

leg_odom:=true hands odom->base_link to the leg odometry, the same node
with the same parameters the robot runs, and moves the plant's ground truth
to base_link_gt (still in TF, for the drift meters). Anything built on the
robot's own pose -- the local costmap, a map -- then inherits the odometry's
honest drift instead of reading the simulator's mind.

A navigation session: model_xml:=scene_nav.xml leg_odom:=true nav:=true --
the corridor scene, the legs' odometry, and wojtek_nav's rolling costmap on
top (/wojtek/nav/costmap). See wojtek_nav/README.md.

vlm:=true adds the VLM brain (wojtek_nav/brain.launch.py) on top of that:
instructions typed into the web console's brain panel go to a model server
at vlm_url (VLM_URL from the environment, else localhost:8000 -- the vLLM
that scripts/serve_vlm.sh starts on the GPU box) and come back as pixel
goals for the resolver. The brain reads the camera's JPEG
(/camera/camera/color/image_raw/compressed), which the sim camera publishes
too, so the sim and the robot feed the same input.

The world the camera draws is config/scene_sim.xml: the training scene plus
a ball, a fire hydrant, a traffic light, a stop sign, a clock and a person
standing around the spawn, so the deck panel's detector has something to
name. The plant loads the same file, so they are solid. model_xml:= takes
you back to the empty floor (model_xml:=scene_mjx.xml), into the walled
navigation room (model_xml:=scene_nav.xml), or anywhere else by path.

telemetry:=true adds /wojtek/sysinfo and /wojtek/policy_timing, the same
opt-in the robot service uses. It is off by default here too. The Foxglove bridge
stays with viz.launch.py in a simulation, so leave foxglove:= alone unless
nothing else holds port 8765.
"""

import os

from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    EqualsSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from wojtek_bringup.launch_common import common_launch_description, resolve_scene

# The brain's defaults live with the brain; the sim only forwards them.
VLM_DEFAULT_URL = "http://127.0.0.1:8000/v1"
VLM_DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def _camera_node(context):
    return [
        Node(
            package="wojtek_pc",
            executable="sim_camera_node",
            output="screen",
            # Rendering needs a GL backend named up front: left to guess inside
            # the container MuJoCo picks one whose library is missing and
            # ABORTS the process instead of raising. An explicit MUJOCO_GL in
            # the environment still wins.
            additional_env={"MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")},
            condition=IfCondition(
                PythonExpression([
                    "'", LaunchConfiguration("camera"),
                    "'.lower() in ('true', '1') and '",
                    LaunchConfiguration("hw"), "' == 'mujoco'",
                ])
            ),
            parameters=[
                {
                    # Resolved by the same function the plant's launch uses:
                    # one scene, one physics state.
                    "model_xml": resolve_scene(
                        LaunchConfiguration("model_xml").perform(context)
                    ),
                    "depth_hz": ParameterValue(
                        LaunchConfiguration("camera_depth_hz"), value_type=float
                    ),
                    "color_hz": ParameterValue(
                        LaunchConfiguration("camera_color_hz"), value_type=float
                    ),
                }
            ],
        ),
    ]


def generate_launch_description():
    # RViz on by default (this is the desk workflow), recording opt-in, and
    # everything else -- nodes, parameters, the arming procedure, and the
    # default policy (launch_common.DEFAULT_POLICY) -- shared verbatim with the
    # robot bringup, so what you watch here is what the robot comes up with.
    ld = common_launch_description(
        with_rviz=True, bag_default="false", hardware="sim", with_gamepad=True,
    )
    for action in (
        # D435-compatible virtual camera: a separate node now, because the
        # physics lives inside ros2_control_node and the renderer is Python.
        # It mirrors the plant's /sim/qpos into its own copy of the model, so
        # there is still one physics state. Needs something to mirror, hence
        # the hw:=mujoco condition -- with the mock there is no pose to draw.
        DeclareLaunchArgument("camera", default_value="true"),
        DeclareLaunchArgument("camera_depth_hz", default_value="15.0"),
        DeclareLaunchArgument("camera_color_hz", default_value="5.0"),
        OpaqueFunction(function=_camera_node),
        # Text-command bridge (wojtek#92), resident by design -- see the
        # module docstring for why that is safe.
        Node(
            package="wojtek_teleop",
            executable="text_commander",
            output="screen",
        ),
        # The manual-control surface, so a session never needs raw service
        # calls: "web" is the browser console (no X11, works from a phone),
        # "qt" the X11 one, "none" for a headless run.
        DeclareLaunchArgument(
            "console", default_value="web", choices=["web", "qt", "none"],
        ),
        Node(
            package="wojtek_pc",
            executable="web_console",
            output="screen",
            condition=IfCondition(
                EqualsSubstitution(LaunchConfiguration("console"), "web")
            ),
            # The console reads its command box from the same policy contract
            # the plant and the policy node do.
            parameters=[{"policy": LaunchConfiguration("policy")}],
        ),
        Node(
            package="wojtek_pc",
            executable="console",
            output="screen",
            condition=IfCondition(
                EqualsSubstitution(LaunchConfiguration("console"), "qt")
            ),
        ),
        # The VLM brain (wojtek_nav), opt-in: it needs nav:=true underneath
        # (goto + the pixel resolver) and a model server to talk to.
        DeclareLaunchArgument("vlm", default_value="false"),
        DeclareLaunchArgument(
            "vlm_url",
            default_value=EnvironmentVariable("VLM_URL", default_value=VLM_DEFAULT_URL),
        ),
        DeclareLaunchArgument(
            "vlm_model",
            default_value=EnvironmentVariable("VLM_MODEL", default_value=VLM_DEFAULT_MODEL),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution(
                [FindPackageShare("wojtek_nav"), "launch", "brain.launch.py"]
            )),
            launch_arguments={
                "url": LaunchConfiguration("vlm_url"),
                "model": LaunchConfiguration("vlm_model"),
            }.items(),
            condition=IfCondition(LaunchConfiguration("vlm")),
        ),
    ):
        ld.add_action(action)
    return ld
