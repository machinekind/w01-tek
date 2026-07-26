"""Policy on the real robot: MD80 (IMPEDANCE) + IMU via ros2_control.

    IMU is the Adafruit 5543 (LSM6DS3TR-C + LIS3MDL) on the Pi's I2C1 -- see
    imu_i2c_hardware_interface and wojtek_real.urdf.xacro. On by default;
    use_imu:=false brings the stack up with the sensor absent/unwired.

    ros2 launch wojtek_bringup real.launch.py [policy:=org/name@sha]
                                              [max_torque:=2.0] [dry_run:=true]
                                              [boot_pose:=home|folded] [bag:=false]

    The MD80 servo settings (impedance kp/kd, torque cap) come from the
    loaded policy's contract. Explicit kp:=/kd:=/max_torque:= override it
    verbatim -- e.g. max_torque:=2 for first tests.

    Every run records a full rosbag (all topics) to bag_dir/run_<timestamp>
    (bag_dir defaults to ~/wojtek_bags). Disable with bag:=false.

Startup procedure:
  1. Power the motors and launch this file (any robot pose is fine). The
     policy runs immediately but real_io_node starts DISARMED -- no commands
     reach the motors.
  2. Physically pose the robot in the boot pose (default boot_pose:=home,
     the standing pose RViz shows; folded also available -- see real_io_node)
     and compare the real robot against RViz.
  3. ros2 service call /wojtek/zero std_srvs/srv/Trigger  -- declares "the robot
     is in the boot pose NOW" and re-zeros the offsets. (Skippable only if
     the robot was already exactly in the boot pose at activation.)
  4. ros2 service call /wojtek/stand_up std_srvs/srv/Trigger  (slow ramp to the
     home standing pose; skip if already standing in home)
  5. ros2 service call /wojtek/arm std_srvs/srv/SetBool '{data: true}'
  6. When done: disarm, then /wojtek/lie_down to ramp gently down to folded.
"""

from wojtek_bringup.launch_common import common_launch_description


def generate_launch_description():
    # PC workflow: RViz available, full rosbag on by default.
    return common_launch_description(with_rviz=True, bag_default="true")
