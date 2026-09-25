# SCAN-Planner: collision-aware local planning under VLM guidance

Wojtek's navigation VLMs — FutureNav, Qwen3-VL, Claude — all fail the same
way: the robot walks into furniture. The VLM is not the problem. A mid-level
`forward 1.5` was executed as a *straight line*, nothing looked at the depth
channel while the robot was moving, and by the time the next frame reached
the model the nose was already in a chair leg.

This adds a local planner between the VLM and the locomotion policy. The VLM
keeps deciding **where** to go; the planner decides **how to get there
without hitting anything**, from the simulated RealSense depth channel only.
It is a port of [SCAN-Planner: Spatial Collision-Aware Local Planning for
Route-Guided Long-Range Quadruped Navigation](https://arxiv.org/abs/2606.19555)
(Zheng et al., 2026), sim-only, no ROS.

```
VLM  ──"forward 1.5"──►  ScanExecutor  ──local goal──►  ScanPlanner  ──(vx,vy,wz)──►  policy
                          (mid-level                     ▲   │                         50 Hz
                           drop-in)          10 Hz replan │   │ 10 Hz sensing
                                                   SlidingOccupancyMap ◄── ego depth (self-masked)
```

## Where it lives

| module | what it is |
|---|---|
| `wojtek_rl/scan/localmap.py` | robot-centric sliding log-odds occupancy (circular buffer, bounded memory) |
| `wojtek_rl/scan/sense.py` | ego depth render with segmentation self-mask; the simulated RealSense |
| `wojtek_rl/scan/footprint.py` | yaw-aware twin-cylinder body model (Eq. 3–5) |
| `wojtek_rl/scan/guide.py` | projected A* guidance, clearance cost, boundary-fallback dead-end recovery |
| `wojtek_rl/scan/traj.py` | uniform cubic B-spline + EGO-style rebound optimisation (Eq. 6–13) |
| `wojtek_rl/scan/planner.py` | 10 Hz replan, trajectory tracking, safety check |
| `wojtek_rl/scan/executor.py` | `MidLevelExecutor` drop-in: same submit/active/blocked contract |
| `wojtek_rl/scan/stack.py` | the four pieces assembled, shared by both simulators |
| `wojtek_rl/scan/viz.py` | the top-down panel: map, guidance, trajectory, footprint |
| `wojtek_rl/scan_bench.py` | A/B harness (`./run.sh scan-bench`) |

## Measured effect

Oracle-VLM guidance (a scripted policy with perfect goal knowledge, emitting
the same mid-level commands a VLM would) on episodes generated so the
straight line start→goal is blocked but a route exists.

**Tier A — kinematic base.** Collisions are steps into an oracle-grid
obstacle; the march stops dead at the first one.

| scene | mode | success | collisions | per metre | episodes that collided | SPL |
|---|---|---|---|---|---|---|
| room (n=6) | straight march | 0.00 | 83 | 11.9 | 6/6 | 0.00 |
| room (n=6) | **SCAN** | **1.00** | **0** | **0.0** | **0/6** | 0.97 |
| apartment (n=8) | straight march | 0.00 | 110 | 12.9 | 8/8 | 0.00 |
| apartment (n=8) | **SCAN** | **1.00** | **0** | **0.0** | **0/8** | 0.90 |

**Tier B — MuJoCo with the locomotion policy walking.** Collisions here are
*real contact events*: a robot geom touching scene geometry above the floor
band, counted on rising edges. The two tiers are not interchangeable — the
oracle grid is inflated by 0.16 m, so a physical body passes through cells
the grid calls occupied, and the straight march often bulldozes to the goal
instead of stopping.

On `fbb_loco_v8`:

| scene | mode | success | contacts | per metre | falls | SPL |
|---|---|---|---|---|---|---|
| room (n=6) | straight march | 0.67 | 101 | 9.0 | 0 | 0.61 |
| room (n=6) | **SCAN** | **1.00** | **4** | **0.29** | 0 | 0.96 |
| apartment (n=8) | straight march | 0.00 | 145 | 19.3 | 0 | 0.00 |
| apartment (n=8) | **SCAN** | **1.00** | 142 | **3.0** | 0 | 0.81 |

On the intended keeper with the corrected `action_scale`, after the
clearance tuning below:

| scene | mode | success | contacts | per metre | episodes with contact | falls | SPL |
|---|---|---|---|---|---|---|---|
| room (n=6) | straight march | 0.33 | 131 | 16.9 | 5/6 | 0 | 0.33 |
| room (n=6) | **SCAN** | **1.00** | **9** | **0.61** | **2/6** | 0 | 0.93 |
| apartment (n=8) | straight march | 0.00 | 132 | 17.4 | 5/8 | 0 | 0.00 |
| apartment (n=8) | **SCAN** | **1.00** | 128 | **2.51** | 7/8 | 0 | 0.78 |

Contacts per metre is the fair column: the straight march collects its hits
inside its first metre and never arrives, while the planner walks 6 m per
episode. (An earlier revision of this table divided by the per-episode *mean*
path instead of the suite total, inflating every per-metre figure by n; the
scoreboard now computes `collisions_per_m` itself.)

**Footprint defaults changed 2026-07-27.** All tables above were measured at
the original `d_off 0.13 / r 0.19`. The shipped defaults are now the visually
tuned `d_off 0.25 / r 0.20` (chosen against the walking robot in the `/tune`
helper: one disc per leg pair, where the measured envelope actually bulges;
the waist between the discs is deliberately uncovered). Re-measured at the
new defaults:

| suite | old defaults | new defaults |
|---|---|---|
| room kinematic | SR 1.00, 0 coll | SR 0.50, 0 coll (blocked 6, paths 2×) |
| apartment kinematic | SR 1.00, 0 coll | SR 0.88, 1 coll |
| room physics | SR 1.00, 0.61 contacts/m | SR 0.83, **0.22 contacts/m**, blocked 2→16 |

The trade: the 0.90 m body refuses passages the 0.64 m one squeezed through —
3× fewer real contacts, more refusals, and lost success in the tight room.
Revert is one line each in `footprint.py` / `localmap.py`; the `/tune` page
(`http://<demo>/tune`) is the tool for picking a different point.

Reproduce (Tier A, no GPU, ~2 s per episode):

```bash
./training/run.sh scan-bench --scene apartment --episodes 8 --seed 3
```

Tier B (needs a policy that walks — see below):

```bash
WOJTEK_POLICY=runs/legacy_policy ./training/run.sh scan-bench --scene room --episodes 6 --seed 5 --tier physics
```

`--video` writes an MP4 per episode: chase camera beside the planner's own
map, showing the guidance and the optimised trajectory bending around
furniture as the map fills in.

## What is faithful to the paper, and what is not

Kept, because it is the point:

- **Yaw-aware twin-cylinder footprint.** Two cylinders at ±0.13 m on the body
  axis, radius 0.19 m; the map is inflated by that radius, so whole-body
  collision checking is two point queries. A single circle big enough for
  Wojtek (~0.30 m) refuses doorways it fits through.
- **Position-only B-spline with induced yaw.** Yaw for collision checking
  comes from the trajectory tangent (Eq. 2) instead of being optimised.
- **Rebound collision penalty.** No ESDF: colliding control points get
  `{anchor, direction}` pairs from the guide path, and the penalty is the
  smooth one-sided `ρ(s_f − d)` of Eq. (10).
- **Log-odds map with clamps.** A moved chair is forgotten in about a second
  (`test_logodds_forgets_a_moved_obstacle`), which the sticky `OnlineMap`
  used by the HUD/frontier explorer cannot do.
- **Robot-centric sliding window with circular-buffer addressing.** Memory is
  bounded and recentring copies nothing; only incoming rows are cleared.
- **Boundary fallback.** A goal unreachable inside the window becomes a
  bounded recovery move toward the boundary, not a planning failure.

Adapted, with reasons:

- **2D band instead of a 3D voxel grid.** The paper's vertical clearances
  `d_up`/`d_down` collapse to the obstacle band `[z_step, z_up] = [0.06,
  0.35] m` that `wojtek_eval.gridmap` already defines: below it is a lip to
  step over, above it is clearance to walk under. Our scenes are flat scanned
  rooms; there are no stairs and no multi-floor structure, so the projected
  A*'s interpolated ground surface is the floor and z-gradient suppression is
  free.
- **Clearance cost in A\*.** Not in the paper. A shortest path grazes the
  inflated boundary, where the inflation radius is the entire margin;
  tracking error then puts a cylinder centre inside an obstacle and the next
  replan has nowhere to start. Pricing the last 30 cm of clearance fixed the
  behaviour that first showed up as the robot wedging itself against a box.
- **Unknown space is traversable but priced** (×1.6), and the tracker creeps
  (0.22 m/s) when the lookahead point is unobserved. The paper's Livox sees
  360°; a forward-facing RealSense does not, and treating unseen space as
  either free or blocked fails in opposite directions.
- **Omnidirectional tracking.** The locomotion policy takes `(vx, vy, wyaw)`,
  so the tracker sidesteps instead of turning in place, while the heading
  still chases the tangent (which is what the footprint check assumed).
- **Smoothness on raw third differences**, EGO's convention. Dividing by
  `dt³` scales that term ~500× at `dt = 0.35 s` and the optimiser would
  rather clip a chair than bend the curve.

## Using it

Both simulators default to the planner; `--no-local-planner` is the A/B
baseline (and the pre-SCAN behaviour).

```bash
./training/run.sh room                              # physics demo, planner on
./training/run.sh room --no-local-planner           # straight-march baseline
./training/run.sh nav-eval ... --no-local-planner   # eval suite baseline
```

The eval scoreboard now carries `collisions` per episode (steps into an
oracle obstacle) next to `blocked`. `off_floor` counts steps off the scanned
floor — the room mesh ends before its walls do, and no depth camera can tell
that from carpet, so it is booked separately and is not a planner failure.

In the room demo the map panel becomes the planner's own view: log-odds map,
A* guidance (cyan), optimised trajectory (green), twin cylinders (yellow).
Click-to-walk is planned too, so clicking a point behind the sofa routes
around it.

## Walking policy (temporary)

### The published keeper's contract is wrong, not its weights

`<HF_ORGANIZATION>/wojtek-springy-locomotion` does not walk through the numpy
runtime (5 mm in 5 s at a commanded 0.4 m/s) but **does walk inside the
training env** — 1.06 m in 5 s with the same weights. The difference is one
field in `policy_meta.json`:

| `action_scale` | result at the contract's own height (0.125 m) |
|---|---|
| as published, `[0.25, 0.5, 0.5] × 4` | 0.005 m / 5 s, body crouches to z 0.09 |
| scalar `0.5` (the env's value) | **1.30 m / 5 s**, healthy z 0.133–0.166 |

The published per-joint value halves the abduction joints' authority, and the
runtime faithfully applies it. (`target_low/high` also carry clamps the env
default does not have — abduction ±0.44, knee ≤3.15 — which cost another
~50 %, but `action_scale` alone accounts for the failure.) Ruled out along the
way: scene, solver iterations and timestep, friction cone, PD gains (model
kp 20 / kd 1 matches the meta exactly), actuator ordering, and the command
channel.

**This is not a sim-only problem.** The same meta is the deployment contract
the robot reads, so a hardware run would get the same crouching shuffle. The
fix belongs upstream: re-export from the run's `run.json` with
`deploy_contract.build_contract`, or correct the meta and re-upload.

Until then `runs/springy_fixed_scale/` holds a local copy with
`action_scale: 0.5` and a `_note` recording why. It is a *behavioural*
inference, not a config the run confirmed — the run directory is not on this
machine — so it is a sim crutch, never a deployment artifact.

### fbb_loco_v8 bridge

The alternative crutch is `fbb_loco_v8`, the last export known to walk in the
demo. It is schema-1 and its observation *ends* with the gait clock:

```
joint_pos:12  joint_vel:12  last_act:12  command:4  phase:8   = 48
```

The deployment runtime (`wojtek_policy.policy.WojtekPolicy`) rejects both the
schema and the `phase` component, so
[`wojtek_rl/legacy_policy.py`](../wojtek_rl/legacy_policy.py) reimplements the
numpy runtime for it, mirroring `env.py`'s clock (`_phase_dt`), walk/trot
blend (`_leg_phases`) and standing-height anchor (`_height_ctrl`) from
constants in the export's own meta. It lives in `training/` deliberately:
nothing in `ros/` changes, so the robot's deployment path cannot pick it up.
`load_policy_runtime` routes to it automatically for schema-1 phase exports
and logs a warning every time.

Measured on the flat scene: 0.15 m/s commanded 0.2, 0.34 commanded 0.4,
0.70 commanded 0.8, and it stands still at zero command.

Point at it with either:

```bash
WOJTEK_POLICY=runs/legacy_policy ./training/run.sh room
```

or `--policy runs/legacy_policy`. `WOJTEK_POLICY` only affects sim/demo
processes; `paths.DEFAULT_POLICY`, which the export contract and deploy
tooling read, is untouched. `runs/` is gitignored, so the artifact is not
vendored. **Delete `legacy_policy.py`, the `WOJTEK_POLICY` hook and this
section once a walking schema-2 keeper lands.**

## Cost

Replan (guide + optimise + safety check) is **1.5 ms mean, 5.4 ms worst** at
10 Hz on one CPU core; tracking ticks are ~0.05 ms. Sensing is a 160×120
depth render plus a segmentation pass for the self-mask. Nothing here needs a
GPU, and the whole loop is comfortably real-time next to the 50 Hz policy.

## Known limits

- **No global route.** The paper's local planner is fed a coarse global route;
  ours is fed the VLM's bearing to the goal, which is not a route. Both
  suites pass now, but that is the 8 m window being large relative to these
  scenes, not the problem being solved: a goal several rooms away behind a
  concave dead end will still trap a greedy bearing. Frontier exploration
  (`wojtek_eval.mapping.FrontierPlanner`, M4 in the
  [roadmap](../../docs/vlm-nav-roadmap.md)) is the natural route source.
- **Residual contact rate with legs.** 2.5 contacts/m in the apartment,
  against 0 in the kinematic tier. Attributed, not guessed: 95 % of them
  happen in cells the map already knows (inside the inflation halo of a known
  obstacle) with the base a median 4 mm off its planned trajectory, at
  0.075 m/s. It is the model, not the sensing or the tracking -- the walking
  body reaches |y| = 0.248 m and the cylinders model 0.19 m, so the legs
  brush what the base cleared. Three fixes were measured; only one paid:
  raising the A* clearance weight 1.8 -> 6.0 (room 1.07 -> 0.61 contacts/m,
  apartment 3.34 -> 2.51 with SR 0.88 -> 1.00). Inflating the map to the true
  envelope (r 0.26) cut apartment contacts to 1.3/m but cost doorways
  (SR 0.88 -> 0.75, blocked 14 -> 37), and widening the turn-sweep check
  changed nothing measurable (8 vs 8, 177 vs 181) while costing turns. The
  honest next lever is the locomotion side -- a policy that tracks lateral
  commands with less body sway -- not more planner margin.
- **Give-up conditions are timing, not reasoning.** The planner declares
  blocked on "did not move for 1.6 s" or "moved for 8 s without getting
  nearer". The second threshold has to be generous, because a legitimate
  detour increases the distance to the goal for seconds; judging both on the
  same clock aborted every avoidance manoeuvre in the multi-room scene
  (apartment SR 0.5 → 1.0 in the physics tier when they were split).
- **The physics tier runs on a temporary policy bridge.** The published
  keeper `<HF_ORGANIZATION>/wojtek-springy-locomotion` does not walk: a constant
  `vx = 0.4` moves the robot 5 mm in 5 s (flat scene and room alike, and
  `tests/test_room_app.py::test_forward_command_moves_robot` fails identically
  on `main`). Its exported observation is
  `joint_pos + joint_vel + last_act + command` = 40 — no IMU and, decisively,
  **no gait phase clock** — and a memoryless MLP without a clock settles on a
  fixed point instead of a gait. See "Walking policy" below.
- **Yaw-aware, not yaw-optimising.** Like the paper, we do not search over
  yaw; a passage that needs a deliberate pirouette will read as blocked.
- **Sim only.** No ROS node, no odometry model. On hardware the map would be
  driven by real RealSense frames and a real pose estimate; the pose here is
  ground truth, which is the biggest sim-to-real gap in this layer.
