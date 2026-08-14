"""Robot-side launch: MD80 (IMPEDANCE) + IMU via ros2_control + policy.

This is the canonical launch for the RPi. It starts only hardware/control
nodes -- no RViz, no GUI. Run visualization/debug on the PC separately:
    ros2 launch wojtek_pc viz.launch.py

    ros2 launch wojtek_bringup robot.launch.py [policy:=org/name@sha]
                                               [max_torque:=2.0] [dry_run:=true]
                                               [boot_pose:=home|folded] [bag:=true]
                                               [gamepad:=true] [perception:=true]
                                               [telemetry:=true] [foxglove:=true]

The servo settings the MD80s run with (impedance kp/kd, torque cap) come
from the loaded policy's contract: policy_meta.json carries the pd block
the policy trained against, and this launch feeds it into the xacro.
Explicit kp:=/kd:=/max_torque:= override the contract verbatim -- e.g. a
low max_torque for cautious first tests.

Recording note: manual runs of this file do not record by default (record
on demand: bag:=true bag_cpus:=0,1). The systemd service
(ros/deploy/wojtek-robot.service) passes exactly those flags, so every
service-driven run leaves a lossless on-robot bag -- the black box of a
real run, now that PC viz also records on demand only. Bags go to
bag_dir/run_<timestamp> (bag_dir defaults to ~/wojtek_bags).

Telemetry is opt-in, like recording. A manual run publishes no
/wojtek/sysinfo and no /wojtek/policy_timing, and starts no Foxglove
bridge. Ask for them with telemetry:=true foxglove:=true. The systemd
service passes both, so a service-driven run can be watched live on port
8765 and read back from its own bag afterwards.

Startup/arming procedure is unchanged from real.launch.py:
  1. Power the motors and launch this file (any robot pose is fine). The
     policy runs immediately but real_io_node starts DISARMED -- no commands
     reach the motors.
  2. Physically pose the robot in the boot pose (default boot_pose:=home)
     and compare against RViz on the PC.
  3. ros2 service call /wojtek/zero std_srvs/srv/Trigger
  4. ros2 service call /wojtek/stand_up std_srvs/srv/Trigger
  5. ros2 service call /wojtek/arm std_srvs/srv/SetBool '{data: true}'
  6. When done: disarm, then /wojtek/lie_down to ramp gently down to folded.
"""

from wojtek_bringup.launch_common import common_launch_description


def generate_launch_description():
    # Headless (no RViz). Recording is opt-in here; the systemd service
    # opts in (see the recording note above). The pad teleop is robot-side
    # only, so this is the launch that can bring it up (gamepad:=true).
    return common_launch_description(
        with_rviz=False, bag_default="false", with_gamepad=True
    )
