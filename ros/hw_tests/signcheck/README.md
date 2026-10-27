# Joint sign check by hand (no torque)

Why: the simulator cannot tell which way a real drive turns. The plant, the
policy and the joint map all live in the same model, so any sign convention
round-trips and looks right, and the drives' sense of rotation exists only in
the motors. On 2026-10-23 the first stand-up of the v6.37a legs turned four
drives the wrong way (identity map), and the second looked like two more
(a zero taken with one lever off its stop). This check measures every sign
on the robot before any torque is applied. Run it for every new set of legs,
every re-mounted drive, and whenever a joint "goes the wrong way".

## Procedure

1. Service stopped, motor power ON, robot lying in the profile's folded pose
   (or on a stand with the legs free to move).

       ssh rpi@10.42.0.2 'sudo systemctl stop wojtek-cm-watchdog wojtek-robot'

2. On the robot, the stack in dry run: drives stay DISABLED, encoders stream.

       source /opt/ros/jazzy/setup.bash && source ~/wojtek_ws/install/setup.bash
       HF_ORGANIZATION=<org> ros2 launch wojtek_bringup robot.launch.py \
           robot:=wojtek_v2 can_baud:=1 boot_pose:=folded dry_run:=true \
           gamepad:=false nav:=false telemetry:=false foxglove:=false perception:=false

3. In a second shell on the robot, the readout (change of every joint since
   its start, URDF = the drives' sense, one line a second). hw_tests is not
   part of the deploy rsync, so copy the script over first:

       scp ros/hw_tests/signcheck/signread.py rpi@10.42.0.2:/tmp/
       ssh rpi@10.42.0.2   # then: source the two setup.bash files and
       python3 -u /tmp/signread.py

4. Move one joint at a time by hand towards the standing pose and note the
   sign. Compare with the expected signs of the profile:

       PYTHONPATH=ros/src/wojtek_policy python3 ros/hw_tests/signcheck/expected_signs.py wojtek_v2

   A joint whose measured sign differs from the expected one needs its sign
   flipped in the profile's joint map (and the URDF axis of that joint, so
   TF stays honest), never a change of the pose tables.

## Record, 2026-10-23, legs v6.37a (profile wojtek_v2)

Lying -> standing by hand, drives disabled, rad:

| leg | hip | lever |
|---|---|---|
| rear_left (4x) | -0.23 | -1.67 |
| rear_right (3x) | +0.53 | +2.11 |
| front_right (2x) | +0.58 | +1.37 |
| front_left (1x) | -0.34 | -2.09 |

Left legs negative, right legs positive on both joints: joint_map_v627.yaml
(hip sign `dir`, knee sign `-dir`). The expected-signs script agrees.
