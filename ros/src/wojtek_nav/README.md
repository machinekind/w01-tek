# wojtek_nav

Navigation for Wojtek: **local perception** and the **setpoint driver** on
top of it. A rolling costmap in the `odom` frame, built from the depth
camera, that remembers what the fixed ~90-degree camera (the raw depth's
field of view; colour-aligned depth sees 70) can no longer see;
and `goto_node`, which walks straight at the setpoint a VLM hands over and
stops when the costmap says the line ahead is blocked. The perception is
stock image_pipeline + nav2 composed by a launch; the driver is ours.

```bash
ros2 launch wojtek_nav costmap.launch.py                                   # standalone, against a running camera + odometry
ros2 launch wojtek_pc sim.launch.py model_xml:=scene_nav.xml leg_odom:=true nav:=true   # a sim session
ros2 launch wojtek_bringup robot.launch.py perception:=true nav:=true      # the robot (not run yet)
```

```
depth image ──► crop_decimate ──► point_cloud_xyz ──► nav2_costmap_2d ──► /wojtek/nav/costmap ──┐
(424x240)       (every 4th px)    /wojtek/nav/points   (rolling 6x6 m)     /wojtek/nav/voxel_grid │
TF odom->base_link (leg_odometry), base_link->camera (URDF / driver) ──┘                          ▼
/wojtek/nav/goal (PoseStamped: the VLM's next setpoint) ─────────────────────────► goto_node ──► /cmd_vel
                                                                                    /wojtek/nav/status
```

| piece | where |
|---|---|
| odometry (`odom->base_link`) | `wojtek_odometry`; on the robot by default, in the sim with `leg_odom:=true` |
| depth stream | `wojtek_perception_bringup` (robot), `sim_camera_node` (sim); the RAW depth, 90 deg of view |
| camera extrinsics | placeholder on the robot (see the perception README), exact in the sim |
| costmap settings | `config/costmap.yaml` -- what the map is for and what every number follows from |
| test world | `wojtek_pc/config/scene_nav.xml`: a corridor with two branches, a crate, a pillar, a 0.15 m box |

## What the costmap is and is not

It is not a map of the world. It is a 6 x 6 m window that moves with the
robot and answers, per 5 cm cell: obstacle, free, or never seen. The
frame is `odom` because over a 6 m window the leg odometry's drift is
centimetres; there is no `map` frame and nothing needs one.

The property that matters is **memory**. The camera is a cone to the
front on a fixed head, blind to the sides and behind. A cell marked while
the crate was in view stays marked after the body has turned away from it
-- until the robot either sees through that cell again (a ray to a floor
point behind it clears it) or drives far enough that it leaves the window.
Free cells are kept the same way. That is what stops the robot turning
into the thing it walked past a moment ago.

Three decisions behind the configuration (the file argues each number):

- **Two observation sources on one cloud.** `depth_mark` marks from 6 cm
  above the floor up; `depth_clear` clears with rays down to the floor.
  One source cannot do both: either the floor becomes a wall or nothing
  ever clears.
- **Heights are in odom, levelled by the IMU.** The body pitches a few
  degrees with every step, 15 cm at 3 m; the odometry carries roll/pitch
  from the IMU, so the floor stays flat in this frame and the 6 cm margin
  holds. This is also why the camera's pitch extrinsic will matter on the
  robot: 1 degree of it is 5 cm at 3 m, a permanent phantom step.
- **Decimate before deprojecting.** The costmap ray-traces every point it
  is given through its 3D voxel grid. 424x240 is 100k rays a frame;
  every 4th pixel is 6k, and at a 5 cm cell that still over-samples.

## Verified (sim, 2026-09-25)

`scene_nav.xml`, `wojtek-stiff-height-locomotion`, depth 15 fps: the
crate 2 m ahead-right is marked from the spawn (23 lethal cells in its
0.6 x 0.6 m box), stays marked after walking 1.1 m towards it (34), and
**stays marked after a 92-degree turn in place that takes it out of the
view** (44). The window follows the robot; free cells behind persist.
Costmap published at ~1.7 Hz, updated at 5 Hz.

What the sim cannot show: sensor noise on the floor (the 6 cm margin is
against a noiseless floor here), the extrinsics error, and the CPU cost on
the RPi -- three C++ nodes at 15 fps and 6k points is a fraction of a core
on the PC, unmeasured on the robot.

## The setpoint driver (`goto_node`)

The contract, decided 2026-09-25: the VLM does the strategy, the robot
keeps the reflexes. A setpoint is a `geometry_msgs/PoseStamped` on
`/wojtek/nav/goal`, ~1 m ahead, about once a second; the robot walks
straight at it (turning in place first when it is far off the nose) and
stops when it gets there. No local planner: which way round the crate is
the VLM's call, from the picture. What the robot vetoes on its own is a
collision: a few cells along the line ahead are probed in the costmap,
and an inscribed/lethal one stops the robot with status `blocked` --
including against the thing it walked past a moment ago and can no
longer see. Turning towards the goal stays allowed while blocked; only a
robot pointed at its goal with an obstacle ahead has nothing left to try.

- **Frame and time.** Any frame TF resolves: a goal in `base_link` with
  the *picture's* stamp is transformed to `odom` at that stamp, so the
  point the VLM meant survives the seconds the robot walked on during
  inference. A goal in `odom` is taken as is.
- **Dead-man.** A setpoint expires `goal_timeout` (3 s) after arrival;
  the VLM must keep talking. The stop is one zero `/cmd_vel` and then
  silence, the same protocol as `text_commander`, so a pad or console can
  take over without a shouting match.
- **Status** on `/wojtek/nav/status` (latched): `idle` / `turning` /
  `driving` / `blocked` / `reached` -- what a VLM loop reads before it
  decides the next point.
- **Non-holonomic on purpose.** The policy can strafe; the camera cannot.
  A robot that walks sideways walks blind.

Verified (sim, 2026-09-25): a setpoint straight through the crate stops
the robot `blocked` 1 m short of it and `idle` after the dead-man; a
setpoint past the crate on its free side is `reached` within 6 cm of the
truth. The pure controller (`goto.py`) has its own desk tests.

## The pixel resolver (`pixel_goal_node`)

Decided 2026-09-25: the VLM hands over a **pixel**, not metres. It sees a
JPEG and answers with where on it to walk; everything metric happens on
the robot, from the depth image taken with that picture. The node is the
one piece between a VLM client and `goto`:

```
colour pixel (u, v) + picture stamp ──► depth pixel on the same viewing ray
   ──► patch median of the depth image nearest that stamp ──► pinhole point in
   camera_depth_optical_frame ──► odom at the picture's stamp (leg odometry)
   ──► standoff 0.7 m in front of the object, facing it ──► /wojtek/nav/goal
```

- **Input** `/wojtek/nav/pixel_goal` (`geometry_msgs/PointStamped`):
  `header.stamp` is the colour picture's stamp, `point.x`/`point.y` the
  pixel normalised to the colour image (a VLM adapter divides Qwen's
  0-1000 by 1000; a value above 1 is taken as an absolute pixel),
  `point.z` a standoff in metres (0 = the `standoff_m` parameter). No new
  message package: the contract is documented, not typed, until it grows
  a second field.
- **Depth, not the colour image.** The node keeps the raw depth stream in
  a ring keyed by stamp and picks the frame nearest the picture's (within
  `max_skew_s`, 0.1 s). It needs the colour *intrinsics* to map the pixel
  through the viewing ray, never the colour bytes.
- **Resolved once, re-sent in odom.** goto expires a setpoint 3 s after
  it arrived, and its TF buffer keeps 10 s: a camera-frame goal re-sent for
  longer than that is dropped (seen in the sim: the robot stopped 0.4 m
  short). So the object point is transformed once, at the picture's
  stamp, and the standoff setpoint goes out in `odom` every `repeat_s`
  until goto says `reached`, has said `blocked` for `blocked_hold_s`
  (goto turns while blocked and may go on; a moment of it is not a
  failure), or `max_goal_s` passed.
- **Status** on `/wojtek/nav/pixel_status` (latched): `resolving` /
  `no_frame` / `no_depth` / `no_tf` / `sent` / `reached` / `blocked` /
  `timeout` / `replaced`; the object point on `/wojtek/nav/pixel_target`
  (odom) for the map view and the eval. A new pixel replaces the goal in
  flight.

Verified (sim, 2026-09-25, `scene_nav.xml`, the crate's near face
projected into the colour image as the "VLM's" pixel): the resolved point
is within 2 mm of the truth in the robot's own frame, three runs, and the
robot stops 0.85 m from the crate (standoff plus goto's tolerance). In
`odom` the same point is off by the odometry's drift at that moment
(0.3-0.4 m late in a session of turns): the robot and the goal share that
frame, so the approach does not care; a persistent map would.

What the picture cannot fix: a point the VLM puts on the object (not the
floor) is what the standoff is for; a target closer than ~0.8 m falls out
of the 15-degree-down camera's view, so ask for `done`/`turn` there, not a
pixel; a setpoint whose straight line clips an obstacle is goto's known
limit -- the VLM must hand over the way round as the next pixel.

## The brain (`vlm_brain_node`)

The loop on top: a user's instruction in, an exploration until the target
is seen, never a guessed goal. The policy is `wojtek_nav/vlm_brain.py`
(pure, desk-tested); the node talks to any OpenAI-compatible endpoint with
JSON-schema structured output (vLLM, Ollama) and to the two nodes above.

```
ros2 run wojtek_nav vlm_brain_node --ros-args \
    -p instruction:="podejdź do fioletowego słupa" \
    -p url:=http://127.0.0.1:11434/v1 -p model:=qwen3-vl:30b-a3b-instruct
```

Every step: a fresh colour frame → the model answers `goal` (a pixel) /
`turn` / `not_visible` / `done` under the schema → a `goal` is **verified**
with a second yes/no question that repeats the task (kind, colour, size)
→ only then the pixel goes to `pixel_goal_node`. `not_visible` turns 45°
and looks again; after a full circle it steps 1 m forward. A target
beyond the depth window (`no_depth`) is approached 1 m and looked at
again; a `blocked` approach or goal turns instead of pushing the same
answer. **Arrival is the executive's call**: `done` when goto reached the
setpoint and the resolved object point is within `done_within_m` (1.1 m);
the model's own `done` is not trusted (a thin pillar never "fills the
view"). Status JSON on `/wojtek/vlm/status`, the picture with the model's
point on `/wojtek/vlm/annotated`; a new task on `/wojtek/vlm/instruction`
replaces the running one.

Measured (sim, 2026-09-25, `scene_nav.xml`, qwen3-vl:30b-a3b-instruct on
Ollama on the DGX): "podejdź do fioletowego słupa" from a pose facing
away: 4 turns, then goal → `reached` in 25 s. "podejdź do niskiej
pomarańczowej skrzynki" from behind the pillar: 6 steps, 55 s, the robot
stopped 0.8 m from the box; on the way the model twice took the big
orange crate for the low box (both orange) and the first approach ran
into the pillar's costmap halo -- the verification and the turn-after-
block are what got it out. Per call: pointing 1.1-1.5 s, verify 0.2 s.

`scripts/point_bench.py` is the offline pointing benchmark behind the
model choice: nine sim frames with the objects' true pixels and depth,
per-object queries and absent-object queries, scored in metres through
the same maths as the resolver. On it qwen3-vl 8B (vLLM, bf16) and
30B-A3B (Ollama, Q4) point equally well (median 0.15-0.18 m); the 30B-A3B
invented a "chair" on the only visible box in 4 of 9 absent cases, the 8B
in 0-1 -- the reason the loop verifies before it moves.

## Next

The verification prompt against look-alikes (the crate/low-box case), a
larger pointing set with masks, and an A/B of the 8B on vLLM (FP8) as the
brain's model. Then: negative obstacles (a hole or a step down is *missing* floor, which
this costmap reads as unknown, not as danger) and the step-height decision
for a legged robot (the 0.15 m box is a wall here; whether it should be is
the policy's business).

## Tests

```bash
cd ros/src/wojtek_nav && PYTHONPATH=$PWD:$PYTHONPATH python3 -m pytest test/ -q
```

Launch composition (the nodes in order, the decimated pair published
where image_transport looks for it, the cloud landing on the topic both
observation sources read, CPU pins), the costmap file's invariants
(rolling window in odom, footprint covers the measured robot,
marking/clearing split by floor height, ranges match the camera), and
the go-to controller on a desk (reaches, turns first, obeys the limits,
blocks and resumes, dead-man, turns while blocked).
