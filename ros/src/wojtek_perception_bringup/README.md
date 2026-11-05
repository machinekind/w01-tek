# RealSense perception bringup

This package starts a RealSense D435 driver with the full bench processing
configuration. The robot's normal Deck camera is a lighter path in
`wojtek_bringup`; both can feed navigation, but only **one driver may open the
camera**. See [architecture](../../../docs/ros-architecture.md) and
[navigation](../wojtek_nav/README.md).

## Choose a camera path

| Path | Use | Streams/settings |
|---|---|---|
| `deck_camera:=true` in robot bringup | Operator panel and current navigation setup | Colour 640×480×30 JPEG; raw depth 424×240×15 by default |
| Standalone `perception.launch.py` | Bench experiments with full post-processing | Depth 848×480×15, decimated to 424×240; colour 1280×720×15, aligned depth and RGBD |
| `perception:=true` in robot bringup | Full pipeline with lower sensor rates | Depth 848×480×6 and colour 1280×720×6; camera transform supplied by robot URDF |
| `sim_camera_node` | ROS MuJoCo | D435-compatible topics and intrinsics; no physical device |

The launch refuses `deck_camera:=true perception:=true`. The full colour/RGBD
path has substantially higher CPU and DDS cost; its existence does not establish
that it fits a given RPi alongside control.

## Bench launch

Inside the ROS container on a Linux host with the camera available:

```bash
ros2 launch wojtek_perception_bringup perception.launch.py
ros2 launch wojtek_perception_bringup perception.launch.py --show-args
```

Parameters come from `config/d435.yaml`; launch overrides such as
`depth_profile:=`, `color_profile:=` and `enable_color:=` take precedence.
`extrinsics:=false` omits this launch's static body-to-camera transform when
another node already owns it.

| Topic | Meaning |
|---|---|
| `/camera/camera/depth/image_rect_raw` | Raw optical-frame depth; use for geometry |
| `/camera/camera/depth/camera_info` | Intrinsics for that depth image |
| `/camera/camera/color/image_raw/compressed` | Colour JPEG; preferred for off-board viewing/VLM |
| `/camera/camera/color/camera_info` | Colour intrinsics |
| `/camera/camera/aligned_depth_to_color/image_raw` | Depth reprojected into colour geometry, when alignment is enabled |
| `/camera/camera/rgbd` | Combined colour/depth product, when enabled |

Use sensor-data QoS for image/CameraInfo subscribers. Inspect actual endpoints
and encodings with `ros2 topic info -v` and `ros2 topic echo --once` before
connecting a new consumer.

## Geometry and calibration

Geometry consumers use raw depth. On the measured camera, colour alignment
reduced horizontal field of view from 90.8° to 70.0°. A pixel selected in the
colour image therefore needs proper reprojection into raw depth; array indices
cannot be copied between the images. `pixel_goal_node` handles this conversion
using both CameraInfo streams and TF.

The design camera mount is 0.32 m forward, 0.07 m up and pitched 15° down relative
to the base. The URDF, sim camera and `config/extrinsics.yaml` carry matching
values. **Physical mount measurement remains open.** Check the selected robot's
geometry; a shared design value is not proof of the assembled mount.

A 1° pitch error moves a floor projection roughly 52 mm vertically at 3 m,
which can create a false obstacle. Commissioning should record the mount,
intrinsics, serial/firmware, stream profiles, calibration method and repeatable
floor/known-target residuals. Define acceptance tolerances for the intended
navigation range before calling the calibration complete.

## Recorded measurements and limits

The existing bench results, at 2–4 m, support these configuration choices:

| Change | Recorded effect |
|---|---|
| HIGH_ACCURACY preset | Temporal noise at 3–4 m fell from 97 to 33 mm |
| Temporal filter delta 20 → 100 on the default preset | At 2–3 m noise fell from 41 to 28 mm |
| HIGH_ACCURACY + temporal delta 100, versus default without a temporal filter | At 2–3 m noise fell from 39 to 10 mm |
| Laser power 150 → 360 | About 15% improvement on default preset; none on HIGH_ACCURACY |

See [D435 noise measurements](../../../docs/perception/d435-noise.md) for the
conditions. The original measurement scripts are not committed, so reproducing
the measurements requires completing that tooling. These are historical bench
results, not a fresh validation of the current robot.

The old 8×8 reduction and accumulated-cloud nodes were removed. SLAM was also
archived; no persistent map replaces them in the current launch. Navigation now
uses a local rolling costmap. Negative obstacles and step-height decisions are
not established capabilities.

## Diagnose the camera

```bash
ros2 node list
ros2 param list /camera/camera
ros2 topic info -v /camera/camera/depth/image_rect_raw
ros2 topic hz /camera/camera/depth/image_rect_raw
ros2 run tf2_ros tf2_echo base_link camera_link
```

Check for a second driver, missing JPEG plugin, incorrect parameter names,
USB/power dropouts, and camera work on the control cores. Driver parameter names
change across versions; inspect what the installed node actually declares.
The [troubleshooting guide](../../../docs/troubleshooting.md) covers system load
and recovery.

## Tests

In a ROS-enabled environment, with the workspace sourced:

```bash
cd ros/src/wojtek_perception_bringup
PYTHONPATH=$PWD:$PYTHONPATH python3 -m pytest test/ -q
```

These tests check launch composition and configuration, not physical depth
accuracy or mounting calibration.
