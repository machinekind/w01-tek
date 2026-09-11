# wojtek_structure_core

ROS 2 Jazzy driver for the Occipital **Structure Core STO2D-C**, written to be
the targeting camera from `wojtek_targeting`'s brief. Colour and depth on
RealSense-shaped topic names, so a consumer written against the terrain D435
moves over by changing a namespace.

```
/targeting_camera/targeting_camera/color/image_raw        rgb8
/targeting_camera/targeting_camera/color/camera_info
/targeting_camera/targeting_camera/depth/image_rect_raw   16UC1, mm, 0 = no return
/targeting_camera/targeting_camera/depth/camera_info
/targeting_camera/targeting_camera/infra/image_raw        mono16  (off by default)
/targeting_camera/targeting_camera/status/healthy         Bool, latched
```

Not `camera`/`camera`: that is `wojtek_perception_bringup`'s terrain D435,
already live on `/camera/camera/...`.

## Will this run on the Pi at all — check this first

**The official Linux distribution of the Structure SDK ships x86_64 objects.
If there is no `aarch64` build of `libStructure.so` in the archive you have,
this driver cannot run on the Raspberry Pi at any quality of code**, and no
amount of work on this package changes that. It is a 30-second check:

```bash
find "$STRUCTURE_SDK_DIR" -name 'libStructure*' -exec file {} \;
```

Want to see `ARM aarch64`. If you see only `x86-64`, stop and pick one of the
three ways out, in the order the brief's own timebox implies:

1. **Second RealSense D435** on a distinct namespace. `realsense2_camera` is
   already an `exec_depend` of this workspace — no new SDK, no arch question,
   and the brief already calls it the lowest-risk path.
2. **Run the Structure Core off a tethered x86 machine** (the dev laptop) and
   let DDS carry the images to the Pi. Works, costs the wifi link, and the
   laptop has to be on the robot or cabled to it.
3. **Android/NDK arm64 build**, if the archive has one. That is an Android
   build against bionic, not glibc — treat as a research project, not a
   weekend path.

Everything below is correct regardless of which way that goes; only item 1
makes this package moot.

## The split, and why the SDK is quarantined

| file | depends on | tested by |
|---|---|---|
| `frame_convert.{hpp,cpp}` | nothing but libstdc++ | `test/host/test_frame_convert.cpp` |
| `structure_core_node.cpp` | Structure SDK, rclcpp | bench + a live sensor |

The same split `ms5611_math.c` / `ms5611.c` uses in the flight firmware, for
the same reason: everything that can be wrong about a pixel format or a
calibration matrix is reachable from a host test, leaving the SDK layer with
USB, delegate threads and reconnects.

```bash
./test/host/run_tests.sh     # 13 checks, no SDK, no ROS, no camera
```

**The package builds without the SDK.** CMake finds it or warns and skips the
node, still building and testing the conversion layer. That is deliberate:
`ros/deploy.sh` builds the robot workspace with `--packages-up-to
wojtek_bringup`, and a package that hard-failed on a missing closed-source SDK
would take the robot's build down with it.

The SDK itself is **not vendored** — this repository is public and the SDK
(XRPro LLC) is not ours to redistribute. Its path enters through
`STRUCTURE_SDK_DIR`, declared at the top of `CMakeLists.txt`, the same way
`UBUNTU_PRO_TOKEN` and `WOJTEK_AP_PSK` enter `ros/deploy.sh`.

```bash
export STRUCTURE_SDK_DIR=/opt/StructureSDK-CrossPlatform
colcon build --packages-select wojtek_structure_core
```

## Depth encoding: the one real conversion

The SDK gives **float millimetres with NaN for "no return"**. This workspace
speaks **16UC1 millimetres with 0 for "no return"** (RealSense's convention,
which `cloud_reduce` and `d435.yaml` already assume). `frame_convert` bridges
that, and the test file pins the cases worth arguing about:

- NaN and ±inf become 0.
- Out-of-range is **dropped, not clamped**. A clamped pixel is
  indistinguishable from a real surface at the range limit, and a gimbal would
  aim at it.
- `min_mm` is raised to 1 when 0 is asked for: a valid pixel may not collide
  with the encoding's own "no return".
- Negative floats and values past 65535 become 0 rather than wrapping into a
  plausible short range.
- The conversion returns a **valid-pixel count**, and the node warns when it
  is zero — that is the failure that looks exactly like a broken driver
  (frames arriving, every pixel out of range) and is otherwise invisible.

## Traps

- **Timestamps.** Frames are stamped with the ROS clock **on arrival**, not
  with the frame's SDK timestamp. The two clocks have different epochs and
  this workspace has already paid for mixing them once: ground truth stamped
  from `controller_manager`'s monotonic clock produced a TF tree where every
  frame was in the buffer and none could be looked up — it looks like missing
  transforms and is an epoch mismatch. The SDK stamp is still tracked, and
  drift against ROS time is logged, because that delta is the only thing that
  separates a slow sensor from a slow node.
- **Delegate callbacks arrive on the SDK's thread**, not the executor's.
  Publishers are thread-safe; the scratch buffer is mutex-guarded; nothing in
  the callback blocks. A `Disconnected` event deliberately does **not** tear
  the session down from inside its own callback — that is how a USB hiccup
  becomes a deadlock. Monitoring continues and the watchdog reports the gap.
- **CPU affinity: `cpus:=0,1`, not 2,3.** Cores 2,3 are the isolcpus RT cores
  here (the service starts the robot tree with `taskset -c 2,3`;
  `wojtek-affinity.sh` pins `controller_manager` to 3 and other RT threads to
  2). Non-RT work goes to 0,1 — what `bag_cpus` and `deck_cpus` use. Verify
  with `taskset -p <pid>`; `ros2 param list` shows parameters, and affinity is
  not one.
- **Interrupts are not moved by isolcpus.** The xHCI interrupts from this
  camera land on the isolated RT cores unless `irqaffinity=` is set on the
  kernel cmdline, and that reads as control-loop jitter correlated with what
  the camera is looking at. A second USB3 camera doubles the exposure. This is
  the one way this package perturbs the walking stack while touching none of
  its code.
- **udev.** Without the SDK's udev rules installed, only root sees the sensor
  and `startMonitoring` fails with nothing useful. Install the rules the SDK
  archive ships (its `Scripts/` directory) rather than writing a rule against
  a guessed vendor id.
- **Power and USB.** A D435 alone draws ~700 mA on USB3 and an underpowered
  supply makes a camera drop out in a way that looks exactly like a bad cable.
  Two depth cameras plus servos on one Pi is worth measuring at the connector.
- **The colour↔depth baseline is a placeholder (`0.0`).** A bbox centre in the
  colour image does **not** index the depth image at the same pixel. Until
  that is measured (or read from the SDK's depth-to-visible extrinsic), treat
  any range number from this driver as unvalidated. This is exactly the open
  decision the brief already lists: whether this slice uses depth at all.
- **The mount transform is not published here.** `base_link ->
  targeting_camera_link` is a singleton that belongs to the robot's bringup,
  matching how `wojtek_perception_bringup` keeps its extrinsics opt-out.
  Measure it: 1° of pitch error is 52 mm at 3 m, larger than this class of
  sensor's own noise there.
- **IMU is off by decision, not by omission.** The brief cuts it: it only buys
  aim compensation while walking, and the fallback plan is
  stop-then-lock-then-track.

## Reaching the robot

`ros/deploy.sh` rsyncs `ros/src/` and then builds `--packages-up-to
wojtek_bringup`. A package that is not in that dependency graph is copied to
the Pi and **never built**. The workspace's established answer is one
`exec_depend` line in `wojtek_bringup/package.xml` — the mechanism that
already ships `wojtek_deck`, `wojtek_teleop` and
`wojtek_perception_bringup`, with no build coupling and nothing added to the
running graph.

That line is **not** added yet, because the targeting brief says not to touch
`wojtek_bringup` and that call is the author's to make.

## Status

| piece | state |
|---|---|
| conversion layer + host tests | done, 13 checks pass |
| driver node (params, topics, TF, watchdog, health) | written, **not compiled against real SDK headers** |
| SDK arch check on the Pi (`aarch64` libStructure) | **open — blocks everything** |
| exact sensor serial | open (`3400` looks partial) |
| udev rules installed on the Pi | not done |
| `exec_depend` in `wojtek_bringup` so deploy builds it | not done, needs author's call |
| colour↔depth baseline measured | not done |
| two cameras + detector on one Pi, measured | not done |
