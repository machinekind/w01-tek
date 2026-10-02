# Wojtek follow v1: lock onto an object on the Deck, walk after it

Status as of 2026-09-12. A plan, nothing built yet.

## The goal

YOLOX on the Steam Deck shows a can in the camera view. The operator taps
it. Wojtek turns toward it, walks after it, holds about one metre, and does
not walk into furniture on the way.

## The decision in short

Version 1 is reactive. It builds no map and estimates no robot pose. The
bearing comes from the tracked box on the image. The distance comes from
the depth camera. Obstacle avoidance is a three-band stop-and-sidestep gate
on the same depth image. SCAN-Planner is deferred to version 2, because it
needs the robot's pose in a world frame and the robot has no odometry.

The Deck page does the lock-in and the tracking. The gateway forwards the
track to a topic and multiplexes the drive. A new follow node on the robot
does the geometry and the control.

## Facts from the checkout that shape the plan

- YOLOX-nano runs in a Web Worker inside the Deck page, on frames it grabs
  from the gateway's MJPEG stream, capped at 15 Hz. Boxes are
  `{x, y, w, h, label, p}` in pixels of the camera frame, 80 COCO classes.
  Nothing reaches ROS. `ros/src/wojtek_deck/web/det_worker.js`,
  `yolox.js`, `yolox.json`.
- The page has no click handling on the video. The overlay canvas has
  `pointer-events: none`.
- `deck_gateway` accepts only `cmd`, `stop`, `height` and `call`. It is the
  single `/cmd_vel` publisher in the Deck setup and holds the dead-man in
  `drive.py`, which is plain Python with unit tests.
- The D435 publishes raw depth at 424×240 and 15 Hz on the robot. Depth
  never crosses the wifi. The Deck runbook launches the camera colour-only
  at 1280×720. Perception is opt-in in the bringup. The camera-to-body
  extrinsics in `wojtek_perception_bringup` are placeholders.
- The Deck to robot round trip is 120 to 250 ms.
- The robot is a Raspberry Pi 3. Cores 2 and 3 are isolated for the control
  loop. Cores 0 and 1 carry the system, USB, hostapd, foxglove_bridge and
  the gateway. Nobody has measured the headroom.
- The walking policy takes `(vx, vy, wz)` and a height on `/cmd_vel`. The
  trained box is 1.2 m/s forward, 0.8 reverse, 0.5 sideways and 1.0 to 1.5
  rad/s yaw. The policy freezes its gait clock below about 0.05 m/s, so the
  sim's mid-level layer floors commands at 0.12 m/s.
- There is no odometry, no pose estimator and no follow or track logic
  anywhere in `ros/` or `experiments/`.
- COCO has no "can" class. The nearest labels are `bottle` and `cup`. A
  0.5 l can is 6.6 cm wide. At 3 m it is about 6 px in the network input.
  Expect detection only inside about 1.5 m.

## Split between machines

1. The Deck page designates and tracks. A tap on the video selects the
   YOLOX box under the finger. If no box is there, a default box around the
   tap becomes the target. The page holds the lock by matching the next
   detections of the same label by nearest centre, and keeps the lock for
   0.7 s after the detection disappears. It sends
   `{"t":"track", cx, cy, w, h, fw, fh, label, age}` at 10 Hz and
   `{"t":"unlock"}`. No new dependency. The overlay gets pointer events.
2. The gateway publishes the track as `vision_msgs/Detection2D` on
   `/wojtek/follow/target`. The turret plan proposes the same message under
   the name `aim`. Both projects use one message. `DriveGate` gets a second
   frame source. In follow mode the gate takes velocities from the follow
   node. The dead-man applies the same way. A stick movement or `stop`
   ends follow at once. The robot keeps one `/cmd_vel` publisher.
3. The follow node is new, in Python. It subscribes to the target, the raw
   depth image, the depth and colour camera_info and the IMU. It runs at
   10 Hz. It publishes `/wojtek/follow/cmd_vel` and a status with the
   state, the range, the bearing and the nearest obstacle. The maths lives
   in a module without ROS, tested on synthetic depth images.

## What the follow node computes

- Bearing. The pixel from the page goes through the colour intrinsics and
  gives azimuth and elevation. The ray is projected straight onto the depth
  image. The offset between the colour and depth sensors of the D435 gives
  under one degree of error at one metre. Aligned depth is out. It narrows
  the field of view from 91° to 70° and a Python subscriber cannot keep up
  with its size.
- Range. The low quantile of the depth values in a window around the
  projected box centre, zeros dropped. The low quantile picks the target
  over the background behind it.
- Floor and own legs. Every frame fits a floor plane to the lower rows of
  the depth image. That gives the camera height and pitch without measured
  extrinsics, and it holds while the body rocks during walking. A fixed
  exclusion zone, measured on the standing robot, removes the legs.
- Obstacles. Points between 0.06 and 0.35 m above the floor are split into
  three vertical bands, left, centre and right. Each band keeps its nearest
  range. SCAN uses the same height band. The target's cone is cut out to
  the target's range, so a person carrying the can is not an obstacle.
- Controller. Yaw rate is a gain times the azimuth, with a 2° dead band
  and a 0.7 rad/s limit. Forward speed is a gain times the range error
  from 1.0 m, with ±0.15 m of hysteresis. The limits are 0.4 m/s forward
  and 0.2 m/s reverse. Commands under 0.12 m/s become zero. Above 30° of
  bearing error the robot turns in place first. A centre-band obstacle
  under 0.5 m zeroes the forward speed. Between 0.5 and 1.0 m the forward
  speed scales down and the robot sidesteps at 0.25 m/s toward the emptier
  side band.
- Target loss. The robot holds its heading for 1 s, then stops. After 3 s
  it drops the lock and the page shows "lost". No track or no depth for
  0.5 s gives zeros.

The bearing gain is bounded by latency. A first-order loop is marginal
when gain times delay reaches π/2. At 250 ms a gain of 1.5 has margin.

## Where it lives

`experiments/wojtek_follow_v1/` with its own README, ROS package and
model-free tests, rsynced to the robot by hand as in the turret plan.
Three things change in `ros/`. The page gets the tap, the lock and the HUD.
The gateway gets the two messages and the drive mux. The `Detection2D`
dependency arrives. The gateway imports nothing from the experiment. It
subscribes to a topic. After a working demo the code moves to
`ros/src/wojtek_follow`.

## Order of work

1. Lock-in on the Deck page: tap, lock, detection matching, the track
   message, the HUD. Tests in `web/test` on frame sequences. Needs no
   change on the robot and can be built against the stack as it runs
   today. One day.
2. Alongside step 1, a half-hour range check. Put the can at one, two and
   three metres and record the label and the distance at which YOLOX sees
   it. This confirms that both lock modes are needed.
3. Gateway: the messages, the topic, the mux in `DriveGate` with unit
   tests. One day. This is also the moment to measure whether the D435 on
   the Pi 3 over USB2 streams colour and depth at the same time. Colour
   drops to 640×480. YOLOX loses nothing, because the network input is
   416 px after the letterbox, and the JPEG encoding on cores 0 and 1 gets
   cheaper.
4. Follow node: the core with tests on synthetic depth first, then the thin
   node. The first run is in simulation through `ros/sim.sh`, because
   `sim_camera_node` publishes the same colour and depth topics. A ball in
   the scene gives YOLOX something to detect. The whole chain runs without
   the robot. Two to three days.
5. Robot on a leash. Bearing while standing. Range to a standing target. A
   box in the path. A person with the can walking through the room. One
   day.
6. Version 2 only if the reactive gate is not enough.

## Version 2 options, in order of effort

- Polar histogram. Bin the obstacle points into 5° sectors, score sectors
  by distance from the target bearing and free range, steer toward the
  best sector with hysteresis. Goes around obstacles rather than stepping
  aside. About 150 lines.
- Polar histogram with one to two seconds of memory, rotated by the IMU
  yaw rate and shifted by the commanded velocity. Covers the obstacle that
  leaves the view during a turn toward the target.
- Dynamic window. Roll a handful of candidate commands one second forward
  and check the twin-cylinder footprint against the points. Handles the
  legs sweeping wide.
- SCAN-Planner port. Pure numpy already, needs world points and a pose.
  The pose would be dead-reckoned from `/cmd_vel` through a measured slip
  factor and the IMU yaw. The map is robot-centric and forgets in about a
  second, so drift over one to two seconds is acceptable. The target enters
  as `set_goal` every tick. See `training/docs/scan-planner.md`.

All options keep the same controller interface. Swapping is a change
inside one module.

## Risks

- Pi 3 headroom. The follow node on depth decimated four times takes a few
  milliseconds per tick. It has to be measured with the walking loop live.
- Legs swing 25 cm from the axis and the body rocks. In the sim SCAN still
  collected 2.5 contacts per metre in the apartment from walking alone.
  The 0.5 m stop threshold accounts for it. The robot will stop in tight
  passages.
- The D435 sees nothing under 0.3 m and nothing to the sides. Reversing is
  blind, which is why it is limited to 0.2 m/s and only when the target
  comes too close.
- A can in a hand means the robot follows the person. A can on the floor
  means the robot walks to one metre and stops.
