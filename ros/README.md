# piesek_ws

ROS 2 workspace for the four-bar-bot dog: `wojtek_policy` (RL policy runtime +
node), `piesek_bringup` (real robot / MuJoCo sim launch), hardware
interfaces (MD80 motors, BMI160/BMX160 IMU), robot description.

Real-robot startup procedure (zero → stand_up → arm) is documented in
`src/piesek_bringup/launch/real.launch.py`.

## Xbox controller teleop

Left stick = velocity (vx/vy), right stick X = yaw, right stick Y = slowly
raise/lower body height. Command ranges are read from `policy_meta.json`,
so the pad can never command more than the policy was trained to track.
If the controller disconnects, the dog gets a zero velocity command.

### One-time setup (on the robot)

```bash
sudo apt install ros-$ROS_DISTRO-joy
```

Pair the controller — hold the pad's small pair button (top edge) until
the Xbox logo blinks fast, then:

```bash
bluetoothctl
  scan on            # wait for "Xbox Wireless Controller" + its MAC
  pair <MAC>
  trust <MAC>        # trust = autoconnect on every future power-on
  connect <MAC>
```

After `trust`, pressing the Xbox button reconnects automatically — no
bluetoothctl ever again. If pairing connects then instantly drops (older
kernels, <5.12): `echo 1 | sudo tee /sys/module/bluetooth/parameters/disable_ertm`
and retry.

### Run

```bash
ros2 launch piesek_bringup real.launch.py    # or sim.launch.py
ros2 launch piesek_bringup teleop.launch.py  # second terminal
```

Launch order vs. controller power-on doesn't matter (`joy_node` hot-plugs).

### First-run check

`ros2 topic echo /joy` and wiggle the sticks — Bluetooth Xbox pads
sometimes enumerate axes differently than the default mapping
(0=LX, 1=LY, 2=RX, 3=RY). Remap via the `axis_*` parameters of
`xbox_teleop_node` if needed. Other useful parameters: `height_rate`
(m/s of height change at full stick, default 0.02), `deadzone` (0.15),
`initial_height` (0.13 m).
