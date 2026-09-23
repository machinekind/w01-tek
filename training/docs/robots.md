# Robot variants

A robot variant is one set of legs on the Wojtek body. Every variant uses the
same body and the same joint, sensor and actuator names, so a policy's
observation layout does not depend on the variant. Link lengths, masses, the
contact pad and the motor limits differ between variants. What a policy
observes is set by the experiment preset, and only the four-part list in
`legs_v627_locomotion` (joint positions, joint velocities, last action,
command) can be exported for the robot.

| Variant | Legs | Standing height | Status |
|---|---|---|---|
| `wojtek` | The robot as built. | 0.12 m | Default. Every task and every existing preset. |
| `legs_v627` | The v6.27 four-bar legs, about 2.3 times longer. | 0.345 m | Simulation only. Joystick task on flat ground. One policy trained, see "First trained policy". |

## Selecting a variant

```bash
# Build the variant's model once. The stock model is built by plain `build`.
./training/run.sh build --robot legs_v627
./training/run.sh check --robot legs_v627

# Resolve, then train. The experiment preset selects the robot itself.
./training/run.sh train +experiment=legs_v627_locomotion --cfg job --resolve
./training/run.sh train +experiment=legs_v627_locomotion run_name=<name> seed=1
```

`robot=legs_v627` selects the variant without an experiment preset. The group
sets `task.env.robot`, `task.env.sim_dt`, the commanded height range, the fall
height and the swing height. The stock experiment presets are tuned to the
stock legs. The env refuses a run that combines `robot=legs_v627` with a stock
`command.height` range.

`run.json` records `task.env.robot`. `eval`, `report`, `courses` and `export`
rebuild the env from `run.json`, so they load the right model without a flag.
A run recorded before the key existed loads the stock robot.

## Looking at a variant

```bash
open -n -a ~/Applications/MuJoCo.app --args \
  "$PWD/ros/src/wojtek_description/mujoco/legs_v627/scene_view.xml"
```

`build` writes `scene_view.xml` next to the training scene. A viewer opens a
model at its rest pose with every control at zero. In `scene_view.xml` a
control is an offset from the home target, so the robot holds its stand and the
sliders bend the joints around it. Nothing trains on this file.

The training scene `scene_mjx.xml` takes absolute joint targets. A zero target
is outside these legs' joint ranges, so in a viewer the robot leaves its stand
until the `home` keyframe is loaded.

The built `legs_v627` model rests in the CAD keyframe pose. The CAD model
itself rests with every joint at zero, which folds each shin back along its
thigh. `build` moves the rest pose and keeps the meaning of every joint value.

The export shows a 40 mm gap between each hip housing and the thigh's cage. The
thigh frame sits 66.8 mm out along the hip axis, against 23 mm on the stock
legs. The motor block that fills the space on the real part has no mesh. The
mechanical team confirmed the offset and left the mesh out on purpose. The
block's mass is included in the thigh's inertial.

## Where a variant lives

| What | Where |
|---|---|
| Per-variant numbers, each with the reason for its value | `training/wojtek_rl/robots.py` |
| Source MJCF | `ros/src/wojtek_description/mujoco/<robot>/wojtek.xml` |
| Generated model and flat scene | `wojtek_mjx.xml` and `scene_mjx.xml` in the same directory |
| Generated viewing copy | `wojtek_view.xml` and `scene_view.xml` in the same directory |
| Visual meshes | `ros/src/wojtek_description/meshes/<robot>/` |
| Config group | `training/wojtek_rl/conf/robot/<robot>.yaml` |
| URDF leg macro | `ros/src/wojtek_description/urdf/leg_v627.urdf.xacro` |
| ROS robot profile: legs, joint map, pinned policy, knee clamp | `ros/src/wojtek_policy/wojtek_policy/robots.py` |

The stock robot keeps its original paths in `wojtek_description/mujoco/`. A
variant never writes to them.

## Importing a CAD export

The mechanical team ships a leg design as a `sim_robot` archive with an MJCF, a
URDF and meshes. That MJCF is a viewer scene. It has its own floor, light and
`<option>`, and its collision is about 330 convex pieces per leg.

```bash
./training/run.sh import-robot --robot legs_v627 --archive <sim_robot.tgz>
./training/run.sh build --robot legs_v627
```

`import-robot` writes the source MJCF and copies the variant's visual meshes. It
drops the floor, the light, `<option>` and the convex collision pieces. It
keeps names, joints, inertials, actuators, sensors, the loop-closure
constraints and the CAD keyframe as exported. `build` then adds the primitive
colliders, the PD servos and the `home` keyframe, as it does for the stock
robot.

A new variant needs an entry in `robots.py` before it can be imported. A new
export of an existing variant needs its `robots.py` numbers checked again. The
height table and the leg colliders depend on the link lengths.

## What is different about `legs_v627`

**The foot.** On the stock legs the loop closes at the foot, and `foot_link` is
the ground contact. On these legs the loop closes 42 mm below the knee.
`foot_link` is that closure point and touches nothing. The pad is a 20 mm disc
at the tip of the shin, so the contact sphere sits on `sixth_link`. `build`
raises an error if the `{leg}_foot` site is not on the body the variant names
as its foot body.

**Mounting.** Front and rear legs are mounted mirrored, with the knees pointing
at each other. Rear second and third joints therefore run negative.
`Robot.leg_sign` carries that sign. The left-right mirror used by
`symmetry.enable` was checked on this model and holds unchanged.

**Height command.** The stock env extends a leg by moving the third joint twice
as far as the second. On these legs that rule swings the foot 17 cm fore and
aft. Each variant now has a `dthird_table` next to its `dsecond_table`. The
offsets for `legs_v627` were solved on the exact loop kinematics, with the foot
held under the hip. The stock table gives the same bits as the old rule.

**Timestep and solver.** The crank that drives the knee is 33 mm long. A loop
closure that opens by a few millimetres changes the leg. Measured on the built
model with `kp=60`, `kd=2`, standing for 2 s and then taking random targets
within 0.3 rad at 50 Hz:

| Timestep | Iterations | Closure `solimp` | Stand height | Loop error standing | Loop error under random targets |
|---|---|---|---|---|---|
| 4 ms | 2 | default | 0.287 m | 23.6 mm | 26.3 mm |
| 4 ms | 4 | default | 0.282 m | 11.0 mm | 13.5 mm |
| 2 ms | 4 | default | 0.338 m | 1.8 mm | 4.6 mm |
| 4 ms | 2 | 0.99 / 0.999 | 0.340 m | 0.6 mm | 152.9 mm, falls |
| 4 ms | 4 | 0.99 / 0.999 | 0.342 m | 0.6 mm | 0.8 mm |
| 2 ms | 2 | 0.99 / 0.999 | 0.345 m | 0.3 mm | 61.1 mm |
| **2 ms** | **4** | **0.99 / 0.999** | **0.345 m** | **0.3 mm** | **0.5 mm** |
| 1 ms | 4 | 0.99 / 0.999 | 0.347 m | 0.1 mm | 0.2 mm |

The bold row is the variant's setting. A control step is 10 physics steps with
4 solver iterations each, against 5 steps with 2 iterations on the stock robot.
The env refuses a `sim_dt` coarser than the variant's.

The 4 ms row with 4 iterations costs half as many physics steps. With random
targets of 1.0 rad, which throw the robot to the floor, its loop error reaches
3.8 mm against 0.7 mm at 2 ms. It became usable only after the passive-joint
friction was corrected on 2026-09-20. With the export's original friction it
reached 13.6 mm at 0.3 rad. Moving to it means changing `timestep` in
`robots.py` and `sim_dt` in the config group. A GPU step-rate measurement
should come first.

These measurements ran on CPU MuJoCo. The step rate on a GPU under MJWarp has
not been measured. `check --robot legs_v627 --gpu --backend warp` measures it
and scales the result by the timestep before comparing with the Go1 gate.

## First trained policy

`wojtek_legs_v627_loco_v1` is `+experiment=legs_v627_locomotion` from scratch,
seed 1, 32768 envs, 498M steps on one RTX 5090 under MJWarp, 2026-09-20. It
trained at about 335k steps/s and took 25 minutes. The stock robot trains at
a similar rate, so the 2 ms step is affordable.

| Measure | Value |
|---|---|
| Final eval reward | 72.0, flat at 67-72 from 130M steps on |
| Falls in the six battery scenarios | 0 |
| Achieved speed at a 0.5 m/s command | 0.43 m/s |
| Achieved speed at a 1.0 m/s command | 0.90 m/s |
| Velocity error, ramp / turn / arc | 0.09 / 0.15 / 0.36 m/s |
| Height error | 2-3 mm, 16 mm in walk-to-stop |
| Torque p50 / p99 / max | 1.9 / 14.4 / 22.0 N*m |
| Gait on the ramp | diagonal pairs in phase (0.50), sides in antiphase (-0.55), duty 0.62 |
| Body attitude, p95 | pitch 2.6 deg, roll 1.9 deg |

The arc scenario is the weak one. The gait numbers in the preset were never
tuned, and nothing in this run pointed at one of them as wrong.

## Level stance and steadier body, 2026-09-21

The first policy carried its body 2.6 to 3.8 degrees nose-up. The stance was
the cause. With the CAD keyframe's equal targets the model stands 1.1 degrees
nose-up, because the centre of mass is 18 mm behind the middle and the rear
legs sag more. The `pose` reward pulls a policy toward that stance.
`stand_pose` in `robots.py` now levels it, and the battery reports the mean
and the standard deviation of pitch and roll, so a lean and rocking read as
two numbers.

Two 367M-step runs from scratch on the level model, seed 1, 32768 envs. Both
use `tracking_sigma=0.15`, `command.arc_prob=0.2` and
`command.pure_wz_prob=0.1`. `wojtek_legs_v627_loco_v2b` also has
`ang_vel_xy=-0.15` and `orientation=-10`. The first policy is scored again on
the level model for the comparison, which is not the model it trained on.

| Measure | First policy | v2a | v2b |
|---|---|---|---|
| Falls in the six scenarios | 0 | 0 | 0 |
| Ramp: pitch / roll standard deviation, deg | 1.22 / 1.05 | 1.36 / 0.93 | 0.98 / 0.62 |
| Ramp: tilt rate rms, deg/s | 25.1 | 23.4 | 16.5 |
| Ramp: velocity error, m/s | 0.067 | 0.068 | 0.055 |
| Turn: roll p95, deg | 3.4 | 3.0 | 2.0 |
| Arc: roll p95, deg | 4.4 | 3.8 | 1.8 |
| Arc: velocity error, m/s | 0.33 | 0.35 | 0.28 |
| Walk to stop: roll p95, deg | 10.1 | 10.0 | 5.1 |
| Front stride span, m | 0.196 | 0.200 | 0.208 |
| Torque p99, N*m | 14.6 | 14.5 | 14.1 |

The attitude terms did the work. The command and tracking changes alone
(v2a) moved nothing by more than noise. v2b rocks a third less, tracks
better, and takes a longer stride at the same duty factor, so it did not buy
its steadiness with a shuffle. Two weak points remain in all three policies.
Yaw rate in the arc is about 0.2 rad/s short of the 0.6 rad/s command. The
stop still rolls the body 5 degrees.

`v2b` is published as the keeper `<HF_ORGANIZATION>/wojtek_v2-locomotion`:
checkpoint, `run.json`, the battery and three videos. It carries no
`policy.npz` pair. The policy observes the gait clock, which the ROS runtime
does not implement, so `export` refuses it.

`report`, `battery` and `eval` take their stance heights from the run's robot.
The stock robot keeps 0.125 m with steps to 0.105 m and 0.155 m. Another
variant uses the middle of its trained `command.height` range, and a quarter
of the way in from each end for the height steps.

## Open questions for `legs_v627`

- `kp=60` and `kd=2` are a starting point. Standing takes about 8 N·m at the
  knee crank, so the legs sag 0.13 rad there. `task.env.pd_kp` and `pd_kd`
  override the gains per run.
- The gait clock, swing height and trot band in `legs_v627_locomotion` are
  scaled from the stock values by leg length. The first policy walks with
  them; none of them has been compared against an alternative.

## Answers from the mechanical team, 2026-09-20

- All twelve motors, abduction included, have 22 N·m. The 2 N·m and 9 N·m
  abduction limits in the export are superseded.
- The dry friction of the passive joints is 0.48 N·m on the rod's joint
  (`fourth_joint`) and 0 on the knee (`fifth_joint`). The export has 1.26 and
  1.48 N·m. `Robot.joint_frictionloss` applies the corrected values at build
  time, and the imported source file stays as exported.
- The 66.8 mm hip offset is correct. The motor block between the hip housing
  and the thigh is left out of the meshes on purpose.

## What does not support a variant yet

- `getup`, `jump` and `biped` hold literal poses and knee angles of the stock
  legs. They refuse another robot.
- Terrain arenas include the stock model and are sized to its footprint.
  `terrain.enable` refuses another robot.
- `fall.max_toggle_deg` and the `toggle_flat` reward measure an angle at the
  stock closure point. They refuse `legs_v627`.
- The course benchmark normalizes scores by the stock stance width and height.
  The SCAN planner footprint and the demo camera are sized to the stock robot.
  Scores from another variant are not comparable with archived ones.
- The bench camera sits 1.11 m above the base, which gives 1.25 m on the stock
  robot and 1.45 m on `legs_v627`.

## Deployment contract

`export` writes the variant's name as `robot` in `policy_meta.json`. For a
variant other than the stock one, `height_table` also carries `dthird` and
`leg_sign`, and `knee_singularity` is `null`. The ROS policy runtime reads
both. It refuses its `clamp_knee` option for a policy without a knee
singularity.

The ROS side picks a robot with a robot profile, in
`ros/src/wojtek_policy/wojtek_policy/robots.py`. A profile names the training
variant it runs. A bringup refuses a policy whose `robot` field names another
variant. A policy exported before the field existed counts as `wojtek`.

| Profile | Variant | Knee clamp | Pinned policy |
|---|---|---|---|
| `wojtek` | `wojtek` | on | yes |
| `wojtek_v2` | `legs_v627` | off | none yet |

A `legs_v627` policy still cannot run on the real robot, because none is
exportable yet. The preset observes the gait clock, which the ROS runtime does
not implement, so `export` refuses it. The same holds in a ROS simulation.

## ROS description

`wojtek_body` in `wojtek_description` takes `legs:=stock` or `legs:=legs_v627`.
`wojtek_sim.urdf.xacro` and `wojtek_real.urdf.xacro` pass the same argument
through. Both leg macros use the same link and joint names, so the
`ros2_control` joint list and TF frame names do not change.

The `legs_v627` URDF uses the MuJoCo model's joint angles. The stock URDF has
gear-correction offsets, which `joint_map.yaml` removes. The v6.27 legs use
`joint_map_identity.yaml`, which maps every joint to itself.

`real.launch.py`, `robot.launch.py` and `sim.launch.py` take `robot:=wojtek`,
the default, or `robot:=wojtek_v2`. The profile sets the xacro `legs`, the joint
map, `clamp_knee`, and the default policy when no `policy:=` is given. With
`robot:=wojtek_v2` a MuJoCo plant loads
`wojtek_description/mujoco/legs_v627/scene_mjx.xml`, which has no props.
`model_xml:=` still overrides it.

```bash
ros2 launch wojtek_pc sim.launch.py robot:=wojtek_v2 policy:=<a legs_v627 policy>
```

`real_io_node` still holds the stock legs' home and folded poses and their
passive-joint table. Its zero, stand-up and lie-down services are not ready
for the v6.27 legs.
