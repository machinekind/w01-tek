# Actuator-parameter identification from rosbags

`./training/run.sh sysid` fits MuJoCo physics parameters — actuator kp/kd,
joint damping/armature/frictionloss, command latency and (opt-in) the
torque-limit scale — so that batched fixed-base MJX replays of a recorded
command stream reproduce the recorded joint trajectories. The optimizer is
CMA-ES (`cmaes` package); every generation evaluates the whole population
in one vmapped MJX rollout, reusing the batched-model mechanics of
`wojtek_rl/randomize.py`.

Identification is **air-rig only**: the robot is held off the ground (a
raised support under its belly) with the legs hanging freely, and the
fit runs against a fixed-base model. On-ground fitting was tried and
removed —
each scoring window has to guess the floating base and contact state,
neither is observable, and the optimizer launders that error into
whatever parameters resist motion (validated twice on ground-truth sim
bags: converged fits produced confident, wrong values). Ground bags are
still useful as a held-out validation set: replay them under identified
vs default parameters and compare.

Install the extra once:

```bash
cd training && uv sync --extra sysid
```

## 1. Record a bag

Rest the robot's belly on a raised support so all four legs hang and
swing freely, and keep it still for the whole recording. A level rig is
the model's default upright orientation — no IMU or `--base-quat`
needed. The tool needs two topics, plus one optional:

| topic | msg | role |
|---|---|---|
| `/wojtek/joint_targets` | `sensor_msgs/JointState` | commanded positions (input) |
| `/joint_states` | `sensor_msgs/JointState` | measured positions (fit target) |
| `/imu/data` | `sensor_msgs/Imu` | rig orientation (skippable: pass `--base-quat` instead) |

The excitation program is designed so each parameter has a phase that
pins it:

- **ramps** — slow constant-velocity triangles per joint type: Coulomb
  friction (frictionloss) appears as a clean constant torque offset.
- **chirps** — per-joint-type sweeps in paired (amplitude, top-frequency)
  passes: big-and-slow (2.0x, to 1.5 Hz), medium (1.0x, to 4 Hz),
  small-and-fast (0.5x, to 8 Hz). The amplitude spread separates Coulomb
  from viscous kd/damping, the fast pass crosses the closed-loop servo
  resonance (~2-4 Hz at kp=20) where armature becomes visible, and the
  pairing keeps peak joint velocity roughly constant.
- **steps** — per-joint-type +/- steps with holds: pins latency and rise
  time.
- **multisine** — all joints at once: held-out coupling data.

The full program is ~2 min; any phase is skippable via its `*_sec`
parameter.

### From the ROS MuJoCo sim (ground-truth validation)

The sim's parameters are known (kp=20, kd=1, damping=0.05, armature=0.01,
frictionloss=0.01), so a bag recorded against it validates the whole
pipeline: the tool should recover those values. Note the stock sim node
spawns the robot standing on the floor — for a clean air-style
validation bag the robot model should be held off the ground, so treat
sim bags from the stock node as plumbing checks, not recovery tests.

```bash
# terminal 1: the simulator
ros2 run wojtek_viz mujoco_sim_node

# terminal 2: the recorder
ros2 bag record -o bags/sysid_air /wojtek/joint_targets /joint_states /imu/data

# terminal 3: the excitation program (anchors at the current pose, then
# ramps -> paired-amplitude chirps -> steps -> multisine; ~2 min)
ros2 run wojtek_viz sysid_excitation
```

Stop the recording when the node logs `program done`.

### From the real robot

Same recorder and excitation node against the real bringup, robot on its
belly on the raised support. Start with `--ros-args -p amplitude_scale:=0.5` on
the first run and be ready on the kill switch; check that no leg can hit
the frame, the floor, or another leg at full amplitude before scaling up.

Do not fit on policy-driven (walking/teleop) bags: the policy feeds
measured state back into the commands, which biases closed-loop
identification, and contact-rich open-loop replays diverge from any
initial-state error regardless of parameters. Use walking bags as a
held-out validation set — replay them under identified vs default
parameters and compare.

## 2. Fit

```bash
./training/run.sh sysid --bag bags/sysid_air
```

The fit always runs against the fixed-base model: the free joint is
removed and the base is welded clear of the floor, oriented by
`--base-quat w,x,y,z` if given, else the bag's median IMU sample, else
the model's upright default — which is already correct for a level
belly-on-support rig. The orientation only sets which way gravity pulls
on the legs; pass `--base-quat` for a tilted rig.

Useful options (defaults in parentheses):

- `--params` (`kp,kd,damping,armature,frictionloss,latency`): comma list
  from `kp,kd,damping,armature,frictionloss,torque_scale,latency`.
  `torque_scale` is opt-in: without torque saturation in the bag it only
  soaks up window-init error.
- `--grouping` (`per_type`): `shared` | `per_type` (first/second/third) | `per_joint`
- `--generations` (60), `--popsize` (64), `--sigma` (0.25), `--seed` (0)
- `--window-sec` (2.0), `--stride-sec` (1.0), `--max-windows` (8),
  `--warmup-sec` (0.3): the bag is cut into overlapping windows, each
  re-initialized from the measured joint state (encoders fully determine
  a fixed-base window); the warmup part of each window is simulated but
  not scored while the four-bar closure settles.
- `--backend` (`auto`): `jax` | `warp`. On a CUDA GPU warp evaluates
  popsize x windows rollouts in parallel; CPU/jax works for short bags.
  **Prefer `--backend jax` for now, even on GPU**: in a 2026-07-22 A/B on
  the same bag and config (RTX 4090), CMA-ES failed to converge on warp
  (generation stats oscillated for all 60 generations, junk parameters at
  their bounds) while jax converged smoothly in ~3 min. Warp's losses are
  deterministic, so the suspect is a rugged/deceptive landscape — e.g.
  silent contact-buffer overflow for stiff candidates (see
  `data_budget_kwargs`); undiagnosed. GPU-jax is fast enough for sysid.
- `--ctrl-dt` (physics step, 4 ms): command grid; latency is expressed in
  these steps and interpolated fractionally, so it stays continuous.

Outputs land in `training/runs/sysid_<timestamp>/`:

- `best.json` — identified physical values (plus baseline), genome, config
- `history.csv` — per-generation best/mean RMS error
- `fit.png` — measured vs sim-default vs sim-identified traces, window 0

## 3. Apply the result

- **kp/kd** → `./training/run.sh build --kp <v> --kd <v>` regenerates the
  model, or override per run with `++task.env.pd_kp=` / `++task.env.pd_kd=`.
- **damping/armature/frictionloss** → edit the joint defaults in
  `ros/src/wojtek_description/mujoco/wojtek.xml` and rebuild (`run.sh
  build`); do not hand-edit the generated `wojtek_mjx.xml`.
- **latency** → steps are on the ctrl grid (default 4 ms each); mirror it
  in training with `++task.env.latency.enable=true` and min/max substeps
  around the identified value.
- **torque_scale** (when identified from a deliberately saturating bag) →
  center the DR range in `conf/config.yaml`'s `dr:` block on it instead
  of treating it as a free robustness knob. Floor friction is outside
  this tool's scope entirely — it needs a dedicated on-ground slip
  experiment.

Identifiability caveats:

- **kd vs damping are nearly degenerate** — both resist joint velocity, so
  the data pins their *sum* much harder than the split (validated on a
  synthetic bag: kd+damping recovered within 10%, the split not). If you
  need them separately, fix one (`--params` without `damping`) or accept
  the sum; for training it rarely matters which one carries it.
- Parameters excite only through the data: a bag without torque
  saturation cannot identify `torque_scale`. Check `fit.png` and treat
  parameters whose bounds the optimizer wandered to as unidentified.
- **Repeat the fit with 2-3 seeds** (`--seed`) and window subsets;
  parameters that agree within ~10% are identified, parameters that
  scatter are not — the spread is the error bar.

Interpretation caveat: the real MD80 motors run cascaded firmware PID in
impedance mode, so the identified sim kp/kd is an *effective* PD
equivalent of that closed loop (with latency and frictionloss soaking up
what PD cannot express), not the firmware's literal gains. That is exactly
what training needs — the sim actuator that best mimics the real
closed-loop response — but do not push the numbers back into the motor
firmware.

Actor observations and deployment artifacts are unaffected: sysid changes
physics parameters only, never the observation layout.
