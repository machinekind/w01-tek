# wojtek_nav

Navigation for Wojtek: **local perception** and the **setpoint driver** on
top of it. A rolling costmap in the `odom` frame, built from the depth
camera, that remembers what the fixed 70-degree camera can no longer see;
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

## Nav2 without a map (`nav2.launch.py`)

The planner and the follower on top of the same perception, when a goal a
few metres out has to be reached AROUND what the camera has seen rather
than stopped in front of it. Stock nav2 (NavFn planner, regulated pure
pursuit follower, the default replanning-and-recovery tree), configured
from `config/nav2.yaml`; the file argues every number that is not already
argued in `costmap.yaml`.

```bash
ros2 launch wojtek_pc sim.launch.py model_xml:=scene_nav.xml leg_odom:=true nav2:=true
ros2 launch wojtek_nav nav2.launch.py                    # standalone, against /wojtek/nav/points + odometry
ros2 topic pub -1 /goal_pose geometry_msgs/PoseStamped "{header: {frame_id: odom}, pose: {position: {x: 3.0}}}"
```

```
/wojtek/nav/points ──► local_costmap  (6x6 m, odom)  ──► controller_server (RPP) ──► /cmd_vel
                   └─► global_costmap (10x10 m, odom) ──► planner_server (NavFn) ──► /plan
/goal_pose (RViz "2D Goal Pose"), /navigate_to_pose action ──► bt_navigator ──► behavior_server (spin, back up, wait)
```

Why no map. Both costmaps are rolling windows in `odom`, the "global"
one merely larger (10 m, where the leg odometry's drift is still inside a
cell). No `map_server`, no static layer, no AMCL, no `map` frame: nothing
here answers "where is the robot in the building", only "how do I get to
that point without hitting what I have seen". The VLM keeps the strategy.

Why not both drivers. `nav2:=true` turns the standalone costmap and
`goto_node` OFF (the bringup passes `costmap:=false goto:=false` to
`costmap.launch.py`, which keeps only the point-cloud pipeline). Nav2 has
its own costmaps, and `goto_node`'s dead-man publishes a zero `/cmd_vel`
that would fight the follower. Without `nav2:=true` nothing changes:
`nav:=true` is the costmap and `goto_node`, the fallback.

The goal is a point. `yaw_goal_tolerance` is over pi: the VLM gives no
heading, so any final orientation counts, and the follower never turns
in place at the end. `xy_goal_tolerance` is `goto_node`'s 0.15 m.

Origin: J. Delicat's first Nav2 configuration (`archive/deli/nav2`,
2026-08-07), with the map, the lidar sources, DWB, the collision monitor
and the waypoint follower taken out.

The local costmap has no inflation layer. Nav2's collision checker scores
the footprint's edges and calls the inscribed cost a hit, and inflation
paints that cost 0.24 m (the inscribed radius) around every obstacle: the
0.76 x 0.48 polygon becomes a 0.69 m circle, and in the 1.8 m corridor a
turn in place passes only within 0.2 m of the centreline. With inflation
in place (first sim session): 21 "detected collision ahead" over four
goals, three "Failed to make progress", one goal lost with Spin and BackUp
both refusing beside a wall 0.66 m away. Without it: the exact polygon, 4
warnings over two goals, no recoveries. The planner keeps its gradient in
the global costmap; the follower's cost-regulated speed scaling is off.

### Verified (sim, 2026-09-25)

`scene_nav.xml`, `wojtek-stiff-height-locomotion`, depth 15 fps, goals
over `/navigate_to_pose` from the spawn (0, 0) facing +x:

| goal (odom) | line to it | result | time | final error |
|---|---|---|---|---|
| (3.0, 0.4) | free | succeeded | 17 s | 0.17 m |
| (0.5, 0.4) back | free, 180 deg turn | succeeded | 49 s (25 s of recoveries, with inflation) | 0.16 m |
| (3.0, -0.4) | crosses the crate | succeeded | 16 s | 0.14 m |
| (0.5, 0.4) back | crosses the crate, 180 deg turn | succeeded | 24 s | 0.16 m |

For the two crate goals the ground truth (`odom -> base_link_gt` at 5 Hz)
never put the footprint polygon over the crate box; the body centre stayed
>= 0.51 m from it, and `/plan` passed the crate at y = +0.38 where the
straight line runs at -0.09. Leg odometry drifted ~0.25 m against the
truth over ~12 m of walking and turning; inside a 5 cm cell per window,
as costmap.yaml assumes.

Seen and left open (perception, not Nav2): after several goals the
rolling costmap held lethal cells on open floor around (0.1, 0.7), a
stale memory of something that was no longer in view -- the robot's own
legs while turning, or the floor through a moment of pitch, are the two
suspects. `publish_voxel_map` in costmap.yaml is how to look.

Not yet run on the robot. Open before that: the CPU cost of two ray-tracing
costmaps on the RPi (`nav_cpus`, or the planner on the DGX), the phantom
marks above, and the recovery behaviours -- backing up blind and spinning
in a corridor are the stock tree's answers, not necessarily a quadruped's.

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

## Next

The VLM loop itself (picture in, setpoint out, on the DGX), and whether it
should hand over metres or a pixel the robot projects through the depth.
Then: negative obstacles (a hole or a step down is *missing* floor, which
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
