# wojtek_perception_bringup

Prototype. Brings up the RealSense D435 depth stream, its post-processing
settings, and the reduction of that depth into a coarse terrain-reference
cloud.

```bash
ros2 launch wojtek_perception_bringup perception.launch.py
ros2 launch wojtek_perception_bringup perception.launch.py --show-args
```

```
realsense2_camera ──depth 424x240@10──► cloud_reduce ──8x8 ordered cloud──► (planner / policy)
   (composable,                          (rclpy)          /cloud_reduce/terrain_points
    in a container)
```

## Why this is a `*_bringup`

It composes more than one package: the third-party `realsense2_camera`
driver plus this package's own reduction node, with shared `config/`. A
single-node subsystem does not need one -- `wojtek_teleop` carries its own
`gamepad.launch.py` and that is the right shape for it.

The launch runs standalone and is includable. The one singleton in it is the
**component container**: pass `container:=<name>` to load the driver into a
container someone else owns instead of creating one. Everything else is
opt-out (`extrinsics:=false`, `reduce:=false`).

## Status

| piece | state |
|---|---|
| camera settings | measured on hardware, see `config/d435.yaml` |
| depth -> 8x8 grid reduction | implemented, unit-tested |
| camera -> body extrinsics | **placeholder numbers**, must be measured |
| occupancy map / planner hook-up | not here yet |
| RPi validation | not run |

## What has actually been verified

Bench numbers behind `config/d435.yaml`, measured on serial `030522071460`
(fx 418.0, baseline 50.13 mm), scene at 2-4 m:

| setting | effect |
|---|---|
| `visual_preset: 3` (HIGH_ACCURACY) vs default | temporal noise at 3-4 m: 97 -> 33 mm |
| `temporal_filter.filter_smooth_delta: 100` vs the default 20 | 2-3 m: 39 -> 10 mm; at 20 the filter is measurably *worse* than none at range |
| `laser_power` 150 -> 360 | ~15% on the default preset, nothing on HIGH_ACCURACY |
| reduction to an 8x8 grid | noise 1.8x better (not 80x -- stereo noise is spatially correlated), fill 74% -> 100% |

Depth error grows with the square of range (`sigma ~= 0.002*z^2` m on this
unit at HIGH_ACCURACY), which is why `clip_distance` is 3.0: past that the
radial error exceeds the planner's 0.05 m map cell.

## Traps

- **Parameter names.** `config/d435.yaml` targets `realsense2_camera` 4.54+.
  The driver has renamed parameters across releases and silently ignores
  names it does not know. Check with `ros2 param list /camera/camera` before
  believing a setting took effect.
- **The reduction is rclpy, so it is not composable.** Python nodes cannot
  load into a component container; the depth image crosses DDS to reach it.
  Fine at 424x240x10 (~2 MB/s), not fine if it moves to full frame rate --
  that needs a C++ rewrite.
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
PYTHONPATH=$PWD python3 -m pytest test/ -q     # 14 tests, no camera needed
```

Covers the reduction maths (median vs flying pixels, range gating, sparse
patches, encoding and camera_info mismatches) and the launch file's
composition logic (container ownership, argument overrides, and that the
measured settings are still in the config).
