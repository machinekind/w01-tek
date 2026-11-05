"""Robot-side launch: MD80 (IMPEDANCE) + IMU via ros2_control + policy.

This is the canonical launch for the RPi. It starts only hardware/control
nodes -- no RViz, no GUI. Run visualization/debug on the PC separately:
    ros2 launch wojtek_pc viz.launch.py

    ros2 launch wojtek_bringup robot.launch.py [policy:=org/name@sha]
                                               [robot:=wojtek|wojtek_v2]
                                               [max_torque:=2.0] [dry_run:=true]
                                               [boot_pose:=home|folded] [bag:=true]
                                               [gamepad:=true] [command_mux:=true]
                                               [deck_camera:=true nav:=true | perception:=true]
                                               [telemetry:=true] [foxglove:=true]
                                               [deck:=true deck_cpus:=0,1]
                                               [control_cpus:=3 policy_cpus:=2]
                                               [mock_hw:=true]

One camera driver at a time: deck_camera:=true (the panel's camera, which
also carries the depth and the JPEG that nav:=true and the VLM brain read)
or perception:=true (the full d435.yaml pipeline). The launch refuses both.

mock_hw:=true swaps the MD80 drives and the I2C IMU for ros2_control's
GenericSystem with the same joints, interfaces and sensor, every other node
untouched: the way to run the whole stack with the motors unpowered
(ros/hw_tests/perf). control_cpus/policy_cpus pin the control loop and the
policy to cores, as the service does.

The servo settings the MD80s run with (impedance kp/kd, torque cap) come
from the loaded policy's contract: policy_meta.json carries the pd block
the policy trained against, and this launch feeds it into the xacro.
Explicit kp:=/kd:=/max_torque:= override the contract verbatim -- e.g. a
low max_torque for cautious first tests.

robot:= names the robot profile (wojtek_policy/robots.py), "wojtek" by
default. It picks the URDF legs, the joint map, the knee clamp and the
default policy, and a policy trained for other legs is refused. The systemd
service passes the robot:= that `deploy.sh --provision` wrote into its
drop-in from WOJTEK_ROBOT (deploy/rpi/wojtek-robot-local.conf).

Recording, telemetry and Foxglove are opt-in: bag:=true, telemetry:=true and
foxglove:=true. The installed robot-local drop-in disables all three, overriding
the base service. Inspect `systemctl cat wojtek-robot.service` for the effective
configuration. Bags go to bag_dir/run_<timestamp> (default: ~/wojtek_bags).

For a commissioned robot, follow docs/operations.md:
  1. With motor power removed, position the robot in its configured boot pose.
     A manual launch defaults to home; the installed service uses folded.
  2. Apply power using the commissioned procedure, then launch. The hardware
     enables the drives unless dry_run:=true. Disarmed blocks policy targets;
     it does not remove torque, and pose ramps can still move the robot.
  3. Check joint feedback and the declared pose, then call /wojtek/zero.
  4. Call /wojtek/stand_up and wait for the ramp-complete log.
  5. Call /wojtek/arm with '{data: true}' only after checking the home stance.
  6. To finish, cancel autonomy, disarm, lie down and wait for ramp completion.
     Stop the service and remove motor power using the commissioned procedure.
"""

from wojtek_bringup.launch_common import common_launch_description


def generate_launch_description():
    # Headless (no RViz), with optional recording and robot-side pad teleop.
    return common_launch_description(
        with_rviz=False, bag_default="false", with_gamepad=True
    )
