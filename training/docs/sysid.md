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
still useful as a held-out validation set, but this fixed-base fitter is not
a walking-bag replay tool. See [digital-twin validation](digital-twin.md).

Install the extra once:

```bash
uv sync --project training --locked --extra dev --extra sysid
```

## 1. Record a bag

Rest the robot's belly on a raised support so all four legs hang and
swing freely, and keep it still for the whole recording. A level rig is
the model's default upright orientation. Use `--base-quat 1,0,0,0` for that
rig so a sensor-frame orientation cannot silently change the base orientation.
The tool needs two topics, plus one optional:

| topic | msg | role |
|---|---|---|
| `/wojtek/joint_targets` | `sensor_msgs/JointState` | commanded positions (input) |
| `/wojtek/joint_states_abs` | `sensor_msgs/JointState` | measured absolute URDF positions (fit target) |
| `/imu/data` | `sensor_msgs/Imu` | optional orientation of the **base**, not the mounted sensor |

Current ROS `/joint_states` is boot-relative and cannot be converted with the
URDF joint map alone. Older simulation bags used absolute `/joint_states`;
read those with `--states-topic /joint_states` only after confirming their
coordinate convention. `--targets-topic` and `--imu-topic` select alternatives.
The current IMU broadcaster uses the mounted sensor frame; use an explicit
measured base quaternion instead of passing that stream as base orientation.

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

### Recording and excitation

Prepare a fixed-base simulated rig or a physically supported air rig first.
The regular ROS simulation starts on the ground; it is not a system-identification
fixture. Simulated recovery tests must use the model's actual gains/limits,
including the loaded policy contract, rather than assume historical defaults.

Before excitation, an operator must select the matching profile, zero the robot,
prepare command ownership and stop policy/teleop output. The excitation node
publishes joint targets; it does not prepare the rig or open the arm gate. The
repository does not provide a complete automatic air-rig bringup.

```bash
# In a sourced ROS container, against the prepared rig:
ros2 bag record -o bags/sysid_air \
  /wojtek/joint_targets /wojtek/joint_states_abs

# A second ROS shell, stock legs. Anchor on absolute joint states.
ros2 run wojtek_pc sysid_excitation --ros-args \
  -r joint_states:=/wojtek/joint_states_abs -p amplitude_scale:=0.5
```

For new legs, also set `joint_map_yaml` to the installed
`wojtek_policy/config/joint_map_v627.yaml`; locate that package with
`ros2 pkg prefix --share wojtek_policy`. The excitation map and fitting map must
agree. Stop the recording when the node logs `program done`; it continues
holding the anchor until stopped.

### From the real robot

Use the recorder and excitation node above with the robot's belly on the raised
support. Confirm the physical motor-power isolation procedure first; disarm is
not torque-off. Start with reduced amplitude and check that no leg can hit
the frame, the floor, or another leg at full amplitude before scaling up.

Do not fit on policy-driven (walking/teleop) bags: the policy feeds
measured state back into the commands, which biases closed-loop
identification, and contact-rich open-loop replays diverge from any
initial-state error regardless of parameters. Use walking bags as a
held-out validation set for a separate ground-contact comparison workflow.

## 2. Fit

```bash
# Repository root. Bag paths passed to run.sh are relative to training/.
./training/run.sh sysid --bag /absolute/path/to/sysid_air \
  --robot wojtek --base-quat 1,0,0,0 --backend jax

# New-leg model and coordinate map:
./training/run.sh sysid --bag /absolute/path/to/sysid_v2_air \
  --robot legs_v627 --base-quat 1,0,0,0 --backend jax
```

The fit always runs against the fixed-base model: the free joint is
removed and the base is welded clear of the floor, oriented by
`--base-quat w,x,y,z` if given, else the bag's median IMU sample, else
the model's upright default — which is already correct for a level
belly-on-support rig. The orientation only sets which way gravity pulls
on the legs; pass `--base-quat` for a tilted rig.

`--robot` selects the generated model and its URDF-to-MuJoCo map together.
`--xml` and `--joint-map` permit measured custom inputs. Known cross-variant
model/map combinations are refused. A bag does not identify its physical robot
automatically: record its unit, revision, zeroing and calibration alongside it.
Selection and coordinate conversion are covered by model-free tests; this does
not establish physical fitting accuracy for the new legs.

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
- `--ctrl-dt` (selected model's physics step): command grid; latency is expressed in
  these steps and interpolated fractionally, so it stays continuous.

Outputs land in `training/runs/sysid_<timestamp>/`:

- `best.json` — identified physical values (plus baseline), genome, resolved
  model/map paths, selected robot/topics and SHA-256 hashes of both input files
- `history.csv` — per-generation best/mean RMS error
- `fit.png` — measured vs sim-default vs sim-identified traces, window 0

## 3. Apply the result

- **kp/kd** → `./training/run.sh build --robot <variant> --kp <v> --kd <v>` regenerates the
  model, or override per run with `++task.env.pd_kp=` / `++task.env.pd_kd=`.
- **damping/armature/frictionloss** → edit the joint defaults in
  variant's source `wojtek.xml` and rebuild (`run.sh build --robot <variant>`);
  check variant-specific overrides in `robots.py`. Do not hand-edit generated XML.
- **latency** → steps are on the selected control grid; mirror it
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
