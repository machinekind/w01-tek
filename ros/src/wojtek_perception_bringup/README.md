# wojtek_perception_bringup

Prototype. Brings up the RealSense D435 depth stream and its
post-processing settings (plus colour/RGBD for the VLM). Also home of
`cloud_accumulate_node`, the PC-side odom-frame accumulated cloud.

```bash
ros2 launch wojtek_perception_bringup perception.launch.py
ros2 launch wojtek_perception_bringup perception.launch.py --show-args
```

```
realsense2_camera ──depth 424x240@15──► (cloud_accumulate on the PC / viz)
   (plain node)   │     ~3 MB/s          /camera/camera/depth/image_rect_raw
                  ├──── colour 1280x720@15 ───────────────────────────────► (VLM)
                  │     ~41 MB/s          /camera/camera/color/image_raw
                  └──── RGBD 1280x720@15 ─────────────────────────────────► (VLM)
                        ~69 MB/s          /camera/camera/rgbd
```

(The depth->8x8 grid reduction that used to sit in this diagram fed only the
SCAN-planner and cost ~0.8 of a Pi 4 core; removed 2026-08 when that path
was dropped. Its bench results stay recorded below.)

848x480 comes off the depth sensor; `decimation_filter.filter_magnitude: 2`
halves each dimension before anything is published, so what crosses DDS is
424x240x16 bit at 15 Hz. Both rates are constrained by what the D435
actually offers (depth 6/15/30/60/90, colour 6/15/30/60), and 6 is the slowest
colour rate the sensor has, which is still ~12x what the VLM's ~0.3-0.5 Hz
decision loop needs.

Colour and RGBD are on by default (`enable_color:=false` drops both) and are
**deliberately more than the machine can deliver**: at 848x480x15 the RGBD
message is 2.04 MB and a Python subscriber received ~70% of them, with the
kernel dropping nothing (0 UDP buffer errors) -- it is deserialisation that
cannot keep up. At 1280x720 expect proportionally worse. That is an accepted
trade for now; the fix, when one is needed, is a C++ consumer in the driver's
own process (see the composition note in the launch file), not more bandwidth.

**Depth consumers keep using the RAW depth topic, not the aligned one.**
Alignment reprojects depth into the colour intrinsics, and measured on this
unit that narrows the horizontal field of view from 90.8 deg to 70.0 deg
(fx 418.0 -> 605.6) -- a quarter of the view, lost at the sides where
obstacles are.

## Why this is a `*_bringup`

It composes the third-party `realsense2_camera` driver with this
package's measured `config/`, and carries the accumulator node the PC viz
launches. A single-node subsystem does not need one -- `wojtek_teleop`
carries its own `gamepad.launch.py` and that is the right shape for it.

The launch runs standalone and is includable. The one singleton in it is the
camera->body static transform, so that is opt-out (`extrinsics:=false`) for
the case where the robot bringup already owns that TF edge.

The config files are ordinary ROS parameter files (`/**: ros__parameters:`)
loaded by the nodes themselves, so `--params-file` and `ros2 param dump`
work on them like on any other node's config. Launch arguments layer single
overrides on top (`depth_profile:=`, `color_profile:=`, `enable_color:=`):
each entry of `parameters=` becomes its own `--params-file` in list order,
and the last one wins.

## Status

| piece | state |
|---|---|
| camera settings | measured on hardware, see `config/d435.yaml` |
| camera -> body extrinsics | **placeholder numbers**, must be measured |
| odom-frame accumulated cloud | `cloud_accumulate_node`, run by the PC viz |
| depth -> 8x8 grid reduction | REMOVED 2026-08 (fed only the dropped SCAN-planner path) |

## What has actually been verified

Bench numbers behind `config/d435.yaml`, measured on serial `030522071460`
(fx 418.0, baseline 50.13 mm), scene at 2-4 m:

| setting | effect |
|---|---|
| `visual_preset: 3` (HIGH_ACCURACY) vs default | temporal noise at 3-4 m: 97 -> 33 mm |
| `temporal_filter.filter_smooth_delta: 100` vs the default 20 | 2-3 m: 39 -> 10 mm; at 20 the filter is measurably *worse* than none at range |
| `laser_power` 150 -> 360 | ~15% on the default preset, nothing on HIGH_ACCURACY |
| reduction to an 8x8 grid (removed) | noise 1.8x better (not 80x -- stereo noise is spatially correlated), fill 74% -> 100% |

Depth error grows with the square of range (`sigma ~= 0.002*z^2` m on this
unit at HIGH_ACCURACY), which is why `clip_distance` is 3.0: past that the
radial error exceeds a 0.05 m map cell.

## Traps

- **Parameter names.** `config/d435.yaml` targets `realsense2_camera` 4.54+.
  The driver has renamed parameters across releases and silently ignores
  names it does not know. Check with `ros2 param list /camera/camera` before
  believing a setting took effect.
- **No component container, on purpose.** Zero-copy needs two C++ nodes in
  one process. The RPi hosts no container (nothing else in this workspace
  creates one; `ros2_control_node` is a standalone process), and the only
  consumer of the depth image is this package's rclpy reduction, which
  cannot be composed at all. So the depth crosses DDS on loopback:
  424x240x2 B at 15 Hz is ~3 MB/s, which is affordable. Rewriting the
  reduction in C++ for a higher frame rate is the change that would make a
  container worth introducing.
- **Pitch beats sensor noise.** 1 deg of camera pitch error puts a floor
  point 52 mm off at 3 m; the sensor's own noise there is ~33 mm. A constant
  mounting error reads downstream as a permanent phantom obstacle. Measure
  the extrinsics before trusting anything.
- **Angular resolution, not noise, is the limit at range.** At a 0.25 m
  mount height one cell of an 8x8 grid covers ~2 m of floor at 2 m range.
  This grid says "flat / rising / obstacle"; it does not resolve stair
  treads.
- **On the RPi, IRQ affinity.** `isolcpus` does not move interrupts. The
  xHCI interrupts from the camera will land on the isolated RT cores unless
  `irqaffinity=` is set, and that shows up as control-loop jitter
  correlated with what the camera is looking at.
- **Power.** The D435 draws ~700 mA on USB3. An underpowered RPi supply
  makes the camera drop out under load, which looks exactly like a bad
  cable.

## Tests

```bash
cd ros/src/wojtek_perception_bringup
source /opt/ros/jazzy/setup.bash
PYTHONPATH=$PWD:$PYTHONPATH python3 -m pytest test/ -q   # no camera needed
```

(`PYTHONPATH=$PWD` alone replaces the ROS paths instead of extending them,
and then `rclpy`/`launch` do not import.)

Covers the launch file's composition logic (which nodes get created, the
parameter files they load, the argument overrides on top of them, and that
the measured settings are still in the config).
