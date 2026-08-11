# Wojtek training configuration reference

This is the agent-facing reference for the project-owned configuration surface
in `training/wojtek_rl`. Commands below are written from the repository root;
`./training/run.sh` changes into `training/` first. The code defaults in the
environment modules remain the final source of truth, while Hydra YAML supplies
overrides on top of them.

## Backend terminology

Wojtek uses **MJWarp**—the Warp implementation of MuJoCo's MJX API—for normal
CUDA training. The implementation is selected by `task.env.sim.backend`:
`auto` resolves to `warp` when MJWarp is installed and JAX has a GPU, and to
MJX's JAX (`jax`) backend otherwise. Brax supplies PPO and training wrappers,
not the physics engine.

Backend selection is local to the host that constructs the environment:
training creates both train and evaluation environments with it, while
`eval`, `battery`, `courses`, and `report` rebuild the saved environment
configuration and resolve `auto` again. Pin `+task.env.sim.backend=warp` or
`jax` when a specific backend matters. `smoke`, `battery`, `courses`,
`report`, and `export` force CPU execution, so `auto` resolves to JAX for
those commands.

## Naming

Use **Wojtek** in new prose and new `run_name` values. The repository lives
at `github.com/machinekind/w01-tek`. `fbb-locomotion`, `fbb_*` preset run names, and `policies/fbb_v3`
are literal current historical configuration/artifact names; do not rename
them in documentation without changing the corresponding source
configuration or artifact.

## First: resolve, then run

Hydra accepts configuration groups and `key=value` overrides.  Resolve a job
before launching it:

```bash
# Print the resolved Hydra job and exit without training.
./training/run.sh train --cfg job --resolve

# Resolve a preset plus overrides.
./training/run.sh train +experiment=locomotion \
  ++ppo.num_timesteps=300000000 seed=1 --cfg job --resolve
```

Use these forms deliberately:

| Form | Use it for | Example |
|---|---|---|
| `group=value` | Select one of an existing config group | `task=jump`, `network=small`, `obs=no_imu` |
| `key=value` | Override a key already present in YAML | `seed=1`, `task.env.latency.enable=true` |
| `+key=value` | Add a code-defaulted key absent from YAML | `+task.env.sim.backend=jax` |
| `++key=value` | Add or override; safe for uncertain/code-defaulted keys | `++ppo.num_envs=8192` |

Quote list overrides in shells that treat brackets specially:

```bash
./training/run.sh train \
  '++task.env.gait.freq=[1.4,3.6]' \
  'dr.dof.damping=[0.9,1.1]'
```

The merge order matters:

1. `conf/config.yaml` selects the base task, network, and observation groups.
2. An optional `+experiment=<name>` changes groups and/or values.
3. Task YAML becomes `task.env` and overlays the selected task's Python
   `default_config()`.
4. `task.ppo`, then root `ppo`, overlay the upstream Go1 PPO baseline.
5. Command-line overrides win.

Therefore, a field missing from YAML may still be valid if it exists in the
task's Python default. Use `++task.env.<field>=...` for it.  A field missing
from both places is not supported. Avoid pairing a task-specific experiment
with a contradictory `task=` override (for example, `+experiment=locomotion
task=jump`); resolve an intentional combination before running it.

## Top-level training settings

Defined in [`conf/config.yaml`](../wojtek_rl/conf/config.yaml).

| Key | Default | Meaning / usage |
|---|---:|---|
| `task` | `joystick` | Task group: `joystick`, `getup`, or `jump`. Example: `task=jump`. |
| `network` | `default` | Network group: `default`, `small`, or `large`. |
| `obs` | `full` | Actor sensor-suite group: `full` or `no_imu`. It writes `task.env.obs.include`. |
| `run_name` | timestamped task name | Output is `training/runs/<run_name>`. The automatic name is `<task>_<timestamp>`, prefixed with `smoke_` for a smoke run. Set it for reproducible, named runs. |
| `seed` | `0` | PPO random seed. |
| `smoke` | `false` | Trainer-level tiny CPU-friendly setting; `./training/run.sh smoke` also forces it and disables WandB. |
| `restore` | `null` | Checkpoint directory to warm start from. Relative paths are relative to `training/`, e.g. `restore=runs/base/checkpoints/000100000000`. |
| `domain_rand` | `true` | Enables the legacy model randomization and permits explicitly enabled `dr.*` additions. The four expanded `dr.*` switches still default false. Set false only to remove all model DR. |
| `wandb.enable` | `true` | Enable WandB if import/login succeeds; use `wandb.enable=false` for local checks. |
| `wandb.project` | `fbb-locomotion` | WandB project name. |
| `ppo` | `{}` | Global PPO overrides. Use `++ppo.<key>=...` because this mapping is intentionally empty in YAML. |
| `hydra.run.dir` | `.` | Keeps Hydra from creating a per-job working directory. Normally leave unchanged. |
| `hydra.output_subdir` | `null` | Suppresses Hydra's `.hydra` output directory. |
| `hydra.job.chdir` | `false` | Keeps the working directory stable while training. |

`domain_rand=false` cannot be combined with any enabled `dr.*` switch; training
raises an error rather than silently ignoring the requested randomization.

## Tasks and environment settings

Task group selection chooses the Python environment listed here.  The YAML
files only contain task-specific overlays; all other keys below are valid
code defaults.  Use `++task.env.<key>=...` for a code-defaulted field that is
not already represented in task YAML.

| Task | Select | Default episode | Python source |
|---|---|---:|---|
| Joystick locomotion | `task=joystick` | 1,000 steps (20 s) | [`env.py`](../wojtek_rl/env.py) |
| Fall recovery | `task=getup` | 300 steps (6 s) | [`env_getup.py`](../wojtek_rl/env_getup.py) |
| Commanded jump | `task=jump` | 250 steps (5 s) | [`env_jump.py`](../wojtek_rl/env_jump.py) |

`joystick` has no task-level PPO override. `getup` and `jump` set
`task.ppo.num_timesteps=150000000`; root `++ppo.num_timesteps=...` still wins.

### Shared task options

All three tasks support these paths through their `default_config()`:

| Path | Default | Notes |
|---|---:|---|
| `task.env.ctrl_dt` | `0.02` s | Policy/control interval. |
| `task.env.sim_dt` | `0.004` s | Physics timestep. Keep control and simulation timing compatible; the joystick latency limit is the resulting substep count. |
| `task.env.sim.backend` | `auto` | `auto`, `jax`, or `warp`. `auto` selects the primary MJWarp CUDA path when available and MJX/JAX otherwise; forcing Warp on an unsupported host fails loudly. Example: `+task.env.sim.backend=warp`. |
| `task.env.sim.naconmax_per_env` | `32` | Warp contact budget per environment. |
| `task.env.sim.njmax` | `320` | Warp constraint-row budget per world. Rows past it apply no force with no warning anywhere; a condim-4 contact costs ~6 rows, so this ceiling is nearer than it looks. `check-terrain` and `terrain-scan` record the peak (`nefc_max`), which is what sizes it. |
| `task.env.sim.num_envs` | `1` | Do not set for training: `train.py` replaces it with `max(ppo.num_envs, ppo.num_eval_envs)`. Set `++ppo.num_envs` instead. |
| `task.env.episode_length` | task-specific | Explicitly override only when PPO's `episode_length` is also understood; absent PPO override, the trainer mirrors this value. |
| `task.env.action_scale` | `0.5` rad | Scales policy action to motor-target displacement. `joystick` additionally accepts a 3-vector `[abduction, hip, knee]` tiled over the 4 legs, giving joints different command authority (e.g. `'task.env.action_scale=[0.25,0.5,0.5]'`); `getup` and `jump` take the scalar only. |
| `task.env.obs_noise.gyro` | `0.2` | Uniform actor gyro noise scale. |
| `task.env.obs_noise.gravity` | `0.05` | Uniform actor gravity-vector noise scale. |
| `task.env.obs_noise.joint_pos` | `0.01` rad | Uniform actor joint-position noise scale. |
| `task.env.obs_noise.joint_vel` | `1.5` rad/s | Uniform actor joint-velocity noise scale. |
| `task.env.obs_noise.gyro_bias` | `0.0` rad/s (off) | Half-width of the per-episode gyro zero-rate offset: drawn U(±bound) at reset, constant through the episode, added to the actor's gyro only. The white `gyro` noise cannot represent it. |
| `task.env.obs_noise.gyro_vib` | `0.0` (off) | Gain of the gyro-vib feedback corruption. Each control step, the change in joint torque drives a resonator tuned to half the control rate, and the resonator's state is added to the gyro reading the actor sees (`base.gyro_vib_step`, mixing `base.VIB_MIX`). This stands in for the frame vibration a real IMU picks up from the policy's own action jitter. Also the `imu-grid --vib-gain` axis. |
| `task.env.obs_noise.gyro_vib_decay` | `0.9` | How sharply the resonator rings. A drive at the resonant frequency is amplified by 1/(1−decay), about tenfold at 0.9. |
| `task.env.obs.state` | task-specific | Ordered actor observation names. |
| `task.env.obs.privileged` | task-specific | Ordered critic-only observation names. |
| `task.env.obs.include` | `[]` | Whitelist applied to actor observations. The `obs` group sets this; an empty list means use every name in `obs.state`. |
| `task.env.reward.scales.<term>` | task-specific | Multiplies each named reward/cost. Available terms are listed with each task below. |

Available shared observation catalog entries are `gyro`, `gravity`,
`joint_pos`, `joint_vel`, `last_act`, `linvel`, `base_height`,
`actuator_force`, and `foot_contact`.  Joystick additionally supplies
`command` and `phase`; jump supplies `jump_signal`.

Changing the actor list or its order changes checkpoint and deployment
compatibility. Actor inputs must be signals available on hardware; put
simulation-only signals in `privileged` only.

### Joystick task (`task=joystick`)

Default actor observations are `gyro`, `gravity`, `joint_pos`, `joint_vel`,
`last_act`, `command`, and `phase`. Its default critic adds `linvel` and
`foot_contact`.

| Path | Default | Meaning |
|---|---:|---|
| `task.env.command.vx` | `[-0.6, 0.6]` m/s | Sampled forward-speed range. |
| `task.env.command.vy` | `[-0.4, 0.4]` m/s | Sampled lateral-speed range. |
| `task.env.command.wz` | `[-0.7, 0.7]` rad/s | Sampled yaw-rate range. |
| `task.env.command.height` | `[0.09, 0.17]` m | Sampled stance-height range. |
| `task.env.command.resample_steps` | `250` | Command refresh period (5 s at default control timing). |
| `task.env.command.zero_prob` | `0.15` | Probability of zeroing velocity commands while keeping the height command live. |
| `task.env.command.pure_wz_prob` | `0.0` | Probability of keeping only the yaw command (vx/vy zeroed): dedicated spin-in-place training. Uniform box sampling almost never draws pure rotation, so turning stays undertrained without it. |
| `task.env.command.pure_vy_prob` | `0.0` | Probability of keeping only the lateral command (vx/wz zeroed): dedicated pure-strafe training, the same exposure fix as `pure_wz_prob`. |
| `task.env.command.arc_prob` | `0.0` | Probability of a forward arc: vx redrawn from `arc_vx`, vy zeroed, the current wz kept. The uniform box rarely draws a clean sustained curve, so arc following stays undertrained without it. |
| `task.env.command.arc_vx` | `[0.3, 0.8]` m/s | Forward-speed range the arc draw redraws vx from. |
| `task.env.command.pure_slow_prob` | `0.0` | Probability of a clean slow straight walk: vx redrawn from `slow_vx`, vy and wz zeroed. Without it the box serves slow forward commands contaminated by lateral/yaw and the gait never learns to scale down. |
| `task.env.command.slow_vx` | `[0.1, 0.35]` m/s | Forward-speed range the slow draw redraws vx from. |
| `task.env.command.pure_fast_prob` | `0.0` | Probability of a clean fast straight walk: vx redrawn from `fast_vx`, vy and wz zeroed. |
| `task.env.command.fast_vx` | `[0.8, 1.2]` m/s | Forward-speed range the fast draw redraws vx from. |
| `task.env.command.pure_back_prob` | `0.0` | Probability of a clean backward walk: vx redrawn from `back_vx`, vy and wz zeroed. These are draw probabilities, not realized shares: the samplers are independent draws applied in order — wz, vy, arc, slow, fast, back, then zero — and each overwrites the last, so the zero draw runs last and wins. |
| `task.env.command.back_vx` | `[-0.8, -0.2]` m/s | Backward-speed range the backward draw redraws vx from. |
| `task.env.push.enable` | `true` | Enable random horizontal pushes. |
| `task.env.push.interval_steps` | `200` | Push interval. |
| `task.env.push.vel` | `0.4` m/s | Velocity impulse magnitude. |
| `task.env.max_torque` | `0.0` Nm (off) | Clamps every actuator's forcerange to `±max_torque` (the model's default is ±9). A binding cap forces compliant leg use over torque spikes; deployment must apply the same cap (`real.launch.py` max torque). |
| `task.env.pd_kp` | `0.0` (off) | `0` keeps the model XML's `kp=20`; a nonzero value patches `actuator_gainprm[:,0]`/`actuator_biasprm[:,1]` in `_customize_model`, overriding the PD servo gains at model load. `dr.joint_gains`'s per-joint gain scale multiplies on top. Deployment gains must match a policy trained with these. |
| `task.env.pd_kd` | `0.0` (off) | `0` keeps the model XML's `kd=1`; a nonzero value patches `actuator_biasprm[:,2]` in `_customize_model`, overriding the PD servo gains at model load. `dr.joint_gains`'s `kd_pct` scale multiplies on top. Deployment gains must match a policy trained with these. |
| `task.env.abduction_ctrl_limit` | `0.0` rad (off) | Intersects the 4 abduction actuators' ctrlrange with `±limit` in `_customize_model`; the mechanical hip travel is looser, so a nonzero value becomes the binding limit. Deploy must match. |
| `task.env.knee_target_max` | `0.0` rad (off) | Upper bound on the knee (third-joint) motor target in `step()`. The knee ctrlrange upper bound is 5.8 rad, past `KNEE_SINGULARITY` (3.2), so a nonzero value is a real clamp, not a no-op. |
| `task.env.action_delay` | `1` | Legacy fixed control-period delay. Set `0` for no fixed delay when latency randomization is off. |
| `task.env.latency.enable` | `false` | Sample a delay in substeps at reset; see [Latency and encoder randomization](#latency-and-encoder-randomization). |
| `task.env.latency.min_substeps` | `0` | Inclusive randomized-delay lower bound. |
| `task.env.latency.max_substeps` | `5` | Inclusive randomized-delay upper bound; must be at most `n_substeps`. |
| `task.env.encoder.enable` | `false` | Enable per-joint encoder-zero offsets. |
| `task.env.encoder.range` | `0.02` rad | Uniform encoder offset is in `[-range, range]`. |
| `task.env.action_filter` | `0.0` | EMA filter strength on actions (`0` disables it). Mirror this filter on the robot if enabled during training. |
| `task.env.symmetry.enable` | `false` | Per-env left-right mirrored worlds: observations mirrored on the way out, actions un-mirrored on the way in, physics and rewards in the real frame. A skill learned turning one way then exists turning the other by construction. Training-only; the deployed policy always sees real observations. |
| `task.env.symmetry.mirror_prob` | `0.5` | Probability that an env is a mirrored one. Sampled at reset, so under the Brax auto-reset wrapper it stays fixed per env for the run, like the latency and encoder draws. |
| `task.env.terrain.enable` | `false` | Train on the terrain arena. Needs `build-terrain` first. See [Terrain curriculum](#terrain-curriculum). |
| `task.env.terrain.arena` | `train` | Which generated arena to load: `train`, `eval` (the fixed measurement course) or `test`. Build it with `build-terrain --arena <kind>`. |
| `task.env.terrain.flat_row` | `false` | Train on the arena variant built with `build-terrain --arena train --flat-row`: level 0 is flat ground and every terrain row sits one level higher. The build and this key must agree or `terrain_env` refuses the load; eval-arena loads ignore the flag. |
| `task.env.terrain.spawn_yaw` | `true` | Random heading at every spawn. |
| `task.env.terrain.pad_jitter` | `0.15` m | Spawn scatter around the pad centre. Keep it under 0.2. |
| `task.env.terrain.init_level_frac` | `0.5` | First spawns come from the easiest half of the rows. |
| `task.env.terrain.demote_fraction` | `0.5` | Drop a level when the episode walked less than this fraction of the distance its commands asked for over a full episode. |
| `task.env.height_scan.enable` | `false` | Add the body-frame terrain-height grid to the observation catalog: `height_scan_clean` (fresh, unmasked) for the critic and `height_scan` (masked, held, optionally corrupted) for the actor. Needs the terrain height lookup: an env built without one (the exporter rebuild, flat-scene tasks) reads all zeros, which deploys as the trained camera-dark case. The realistic flat-floor reading — values near zero plus sensor noise, what the robot's node reports on flat ground — comes from the terrain arena's flat level-0 row, where the lookup is live. Naming either component in an obs list with this off is an error. |
| `task.env.height_scan.x_range` | `[0.65, 1.45]` m | Body-frame forward extent of the grid. |
| `task.env.height_scan.y_range` | `[-0.3, 0.3]` m | Body-frame lateral extent of the grid. |
| `task.env.height_scan.nx` / `.ny` | `5` / `5` | Grid points per axis, index `ix*ny + iy`. The observation width and its mirror map are sized for 5×5, so another grid needs `wojtek_rl/height_scan.py` changed too. |
| `task.env.height_scan.clip` | `0.3` m | Each value is the terrain height minus the lowest foot's world z, clipped to `±clip`. |
| `task.env.height_scan.hold_steps` | `3` | Control steps between camera captures (a 15 Hz depth stream under the 50 Hz policy). Each env starts on a random phase. |
| `task.env.height_scan.delay_steps` | `1` | Control steps between a capture and its arrival in the actor's buffer. Must be under `hold_steps`. |
| `task.env.height_scan.dark` | `false` | Feed the actor zeros while the critic still sees the grid. |
| `task.env.height_scan.mask.enable` | `true` | Zero the points an analytic camera frustum cannot see. The frustum follows the full base orientation, so pitching the nose up blinds the near rows. |
| `task.env.height_scan.mask.mount` | `[0.32, 0.0, 0.07]` m | Body-frame camera position. |
| `task.env.height_scan.mask.pitch_deg` | `15.0` | Downward tilt of the optical axis from the body x-axis. |
| `task.env.height_scan.mask.hfov_deg` / `.vfov_deg` | `90.7` / `61.2` | Field of view: a D435 depth stream at 424×240 (`fx=fy=209`). |
| `task.env.height_scan.mask.min_depth` / `.max_depth` | `0.3` / `3.0` m | Usable depth band; points outside it are invisible. |
| `task.env.height_scan.mask.occlusion` | `true` | Also drop points a nearer rise stands in front of: the frustum reaches them, the line of sight does not. Off leaves frustum-only visibility. |
| `task.env.height_scan.mask.occlusion_samples` | `8` | Terrain samples per camera-to-point ray, fixed at trace time. A rise narrower than the ray's sample spacing can slip between two samples. |
| `task.env.height_scan.mask.occlusion_margin` | `0.02` m | How far the ground has to stand above the ray before it counts as blocking it. |
| `task.env.height_scan.corrupt.enable` | `false` | Apply the sensor-failure model to the actor's copy. Off, the actor still gets the masked and stale grid. |
| `task.env.height_scan.corrupt.noise_prob` | `0.6` | Episode draws iid gaussian noise of `noise_std`. |
| `task.env.height_scan.corrupt.drift_prob` | `0.3` | Episode draws a fixed offset `drift_z + drift_tilt * (x - x_min)` over the grid's body-frame x. |
| `task.env.height_scan.corrupt.blackout_prob` | `0.1` | Episode reads all zeros. |
| `task.env.height_scan.corrupt.noise_std` | `0.02` m | Noise-regime standard deviation. |
| `task.env.height_scan.corrupt.drift_z` / `.drift_tilt` | `0.05` / `0.05` | Drift offset and ramp are drawn uniformly from `±` these, once per episode. |
| `task.env.height_scan.corrupt.dropout_prob` | `0.05` | Per-point chance of reading zero, in the noise and drift regimes. |
| `task.env.height_scan.corrupt.pitch_jitter_deg` | `0.0` | Optical-axis pitch offset drawn uniformly from `±` this, once per episode. The actor's mask (frustum and line of sight) is computed from the jittered pose; the critic's clean grid and the height values are untouched. |
| `task.env.height_scan.corrupt.mount_jitter` | `0.0` m | Same draw for the mount position, independently per body axis. Together with the pitch these say how well the camera's pose on the robot is known. |
| `task.env.fall.min_height` | `0.06` m | Fall threshold. |
| `task.env.fall.max_tilt_gz` | `-0.4` | Fall tilt threshold in body-frame gravity z. |
| `task.env.gait.freq` | `[1.4, 3.0]` Hz | Clock frequency range from slow walking to maximum command speed. |
| `task.env.gait.swing_height` | `0.03` m | Target swing-foot clearance. |
| `task.env.gait.trot_band` | `[0.35, 0.55]` m/s | Planar-speed range that blends walk into trot. |
| `task.env.gait.duty` | `[0.7, 0.5]` | Walk/trot stance fractions. |
| `task.env.gait.air_time_cap` | `0.0` s (uncapped) | Cap on the swing time `feet_air_time` pays for. An anti-runaway bound on the swing reward, not a cadence target. |
| `task.env.reward.tracking_sigma` | `0.25` | Velocity-tracking reward width. |
| `task.env.reward.phase_sigma` | `0.002` | Gait-phase reward width. |
| `task.env.reward.height_sigma` | `0.001` m² | Height-tracking reward width. |
| `task.env.reward.pose_leg_weight` | `0.25` | Relative leg-joint weight in the pose anchor. |
| `task.env.reward.torque_limit_frac` | `0.85` | Fraction of each actuator's forcerange cap (post `max_torque`, when set) above which the `torque_limit` hinge starts paying. |
| `task.env.reward.tracking_product` | `false` | Gate the linear and angular tracking kernels by each other, so ignoring half a command collects roughly nothing. Off is the legacy additive form, where a policy that deadlocks under a pure-spin command still earns the full `tracking_lin_vel`. |
| `task.env.reward.tracking_relative` | `false` | Divide the tracking error by the commanded magnitude (floored, so zero commands stay well-defined), so the same fraction of a command pays the same at every speed. On, the kernels use `tracking_rel_sigma` and `tracking_sigma` is inert. |
| `task.env.reward.tracking_rel_sigma` | `0.25` | Dimensionless width of the relative tracking kernels. Read only when `tracking_relative` is on. |
| `task.env.reward.apex_target` | `0.05` m | Per-swing peak-clearance target that `feet_apex` pays at touchdown. |
| `task.env.reward.glide_height` | `0.03` m | Pre-contact clearance band in which `feet_landing` prices downward foot speed; a foot above the band contributes nothing. |
| `task.env.reward.orientation_tol_deg` | `0.0` | Tolerance cone around upright for the `orientation` penalty, half-angle in degrees. The penalty becomes `max(sin^2(tilt) - sin^2(tol), 0)`: zero inside the cone, rising continuously from its edge. `0` is the bit-exact legacy penalty. |
| `task.env.reward.terrain_gate.enable` | `false` | Scale the gait shaping with the terrain under and just ahead of the robot: `feet_apex` and `high_step` are multiplied by the gate, `feet_landing` is discounted by `landing_soften`. Terrain only; off, every factor is the literal `1.0` and the terms are unchanged. |
| `task.env.reward.terrain_gate.floor` | `0.15` | Fraction of the lift shaping that pays on flat ground. |
| `task.env.reward.terrain_gate.rough_ref` | `0.05` m | Local height spread at which the gate reaches 1. Roughness is `max - min` of the surface over the four foot positions plus a 3×3 body-frame strip at `x ∈ {0.17, 0.33, 0.5}`, `y ∈ {-0.2, 0, 0.2}`. |
| `task.env.reward.terrain_gate.landing_soften` | `0.5` | Fraction of the `feet_landing` charge that roughness removes, so a foot may be planted hard on a riser. |

Joystick reward-scale defaults:

| Term | Value | Term | Value |
|---|---:|---|---:|
| `tracking_lin_vel` | `1.5` | `tracking_ang_vel` | `0.8` |
| `height_tracking` | `1.0` | `lin_vel_z` | `-2.0` |
| `ang_vel_xy` | `-0.05` | `orientation` | `-5.0` |
| `torques` | `-0.0002` | `torque_rate` | `0.0` |
| `action_rate` | `-0.25` | `action_accel` | `-0.1` |
| `energy` | `-0.002` | `pose` | `-0.5` |
| `feet_air_time` | `2.0` | `feet_slip` | `-0.25` |
| `feet_phase` | `1.0` | `contact_match` | `1.0` |
| `high_step` | `0.0` | `stand_still` | `-0.5` |
| `stand_feet_down` | `0.0` | `termination` | `-1.0` |
| `torque_limit` | `0.0` | `feet_apex` | `0.0` |
| `feet_landing` | `0.0` |  |  |

The zero-default terms are dormant until a preset or override enables them:
`torque_rate` penalizes step-to-step change in actuator torque (bang-bang
motor commands the policy-side `action_rate` cannot see), `high_step`
rewards swing-foot clearance up to `gait.swing_height` while moving,
`stand_feet_down` penalizes total foot clearance at zero command,
`feet_apex` pays each swing's peak clearance against `apex_target` once at
touchdown (`high_step` pays duration-averaged clearance, which a skimming
gait collects too), `feet_landing` penalizes downward foot speed inside
`glide_height` of the ground, and `torque_limit` is the saturation hinge
described above.

Example custom joystick distribution:

```bash
./training/run.sh train task=joystick \
  '++task.env.command.vx=[-0.8,1.2]' \
  '++task.env.gait.freq=[1.4,3.6]' \
  ++task.env.push.vel=0.5
```

### Getup task (`task=getup`)

Default actor observations are `gyro`, `gravity`, `joint_pos`, `joint_vel`,
and `last_act`; the critic additionally receives `linvel`, `base_height`,
`actuator_force`, and `foot_contact`.

| Path | Default | Meaning |
|---|---:|---|
| `task.env.drop_from_height_prob` | `0.8` | Probability that reset begins from a dropped state. |
| `task.env.drop_height` | `0.35` m | Starting base height for a dropped reset. |
| `task.env.settle_time` | `0.7` s | Random-target settling period used to create a valid crumpled pose. |
| `task.env.knee_target_max` | `3.15` rad | Hard safe target cap below the four-bar snap-through singularity. |
| `task.env.stand_height` | `0.0` | `0.0` resolves to the home keyframe's settled height at initialization. |

Getup reward-scale defaults: `orientation=1.0`, `torso_height=1.0`,
`posture=1.0`, `stand_still=1.0`, `action_rate=-0.001`,
`torques=-0.0001`, `dof_acc=-2.5e-7`, `dof_vel=-0.1`,
`dof_pos_limits=-0.1`, `lateral_vel=-0.5`, and
`knee_singularity=-1.0`.

The task YAML also sets `task.ppo.num_timesteps=150000000`.

### Jump task (`task=jump`)

Default actor observations are `gyro`, `gravity`, `joint_pos`, `joint_vel`,
`last_act`, and `jump_signal`; the critic additionally receives `linvel`,
`base_height`, `actuator_force`, and `foot_contact`.

| Path | Default | Meaning |
|---|---:|---|
| `task.env.jump_at_steps` | `[50, 120]` | Inclusive/exclusive range that samples when the command arrives. |
| `task.env.countdown_steps` | `25` | Visible pre-load/countdown duration. |
| `task.env.knee_target_max` | `3.15` rad | Hard safe target cap below the singularity. |
| `task.env.crouch_depth` | `0.06` m | Target base-height reduction during wind-up. |
| `task.env.crouch_sigma` | `0.02` m | Crouch shaping width. |
| `task.env.active_cost_scale` | `0.25` | Smoothness/energy cost multiplier during wind-up and flight. |
| `task.env.forcerange` | `6.0` Nm | Per-actuator torque cap for this task. |
| `task.env.stand_height` | `0.0` | `0.0` resolves to the home keyframe's settled height. |
| `task.env.fall.min_height` | `0.05` m | Fall threshold. |
| `task.env.fall.max_tilt_gz` | `-0.4` | Fall tilt threshold. |

Jump reward-scale defaults: `jump_peak=40.0`, `flight=10.0`,
`launch_vel=5.0`, `crouch=2.0`, `stand_height=1.0`, `posture=1.0`,
`orientation=-5.0`, `lateral_vel=-1.0`, `torques=-0.0002`,
`action_rate=-0.25`, `action_accel=-0.1`, `energy=-0.002`,
`dof_acc=-2.5e-7`, `dof_vel=-0.1`, `knee_singularity=-1.0`, and
`termination=-1.0`.

The task YAML also sets `task.ppo.num_timesteps=150000000`.

## Observation and network groups

| Group | Values | What it changes |
|---|---|---|
| `obs` | `full`, `no_imu` | `full` preserves all hardware-facing task signals. `no_imu` retains joint encoders, action history, and task signals but removes gyro/gravity from the actor. The critic list is unchanged. |
| `network` | `default`, `small`, `large` | Actor/critic MLP layer sizes. `default` inherits Playground's tuned Go1 layout; `small` uses actor `[128,128]`, critic `[256,256]`; `large` uses `[512,256,128]` for both. |

For an ad-hoc architecture, use the network factory paths:

```bash
./training/run.sh train \
  '++network.policy_hidden_layer_sizes=[256,128]' \
  '++network.value_hidden_layer_sizes=[256,128]'
```

Network/observation changes require a fresh run.  A checkpoint can only be
restored or exported if its actor and critic network shapes match.

## Domain randomization

`domain_rand=true` is the default.  It always includes the original,
hard-coded distribution below; these legacy ranges do not currently have
Hydra keys:

| Quantity | Draw per environment |
|---|---|
| Floor friction | Uniform `[0.6, 1.2]` |
| Root/base mass scale | Uniform `[0.7, 1.3]` |
| Per-link mass scale | Uniform `[0.9, 1.1]` |
| Shared actuator gain scale | Uniform `[0.8, 1.2]` |
| Shared actuator KD scale | Uniform `[0.8, 1.2]` |

The expanded fields below are independently optional and default off. They
all require `domain_rand=true`.

| Feature | Enable | Tunables | Behavior |
|---|---|---|---|
| Root CoM offset | `dr.com_offset.enable=true` | `dr.com_offset.xy=0.02`, `dr.com_offset.z=0.01` m | Uniform root `body_ipos` offset of ±xy in x/y and ±z in z. |
| Per-joint gains/KD | `dr.joint_gains.enable=true` | `dr.joint_gains.gain_pct=0.2`, `dr.joint_gains.kd_pct=0.2` | Replaces the legacy shared scalar with independent 12-joint ±fraction draws. |
| DOF properties | `dr.dof.enable=true` | `dr.dof.damping`, `dr.dof.armature`, `dr.dof.frictionloss` ranges | Jointly randomizes all three multiplicative DOF properties. |
| Per-foot friction | `dr.foot_friction.enable=true` | `dr.foot_friction.range=[0.8,1.2]` | Independent multiplicative draw for each foot; foot contact priority makes the foot draw take effect. |
| Motor strength | `dr.motor_strength.enable=true` | `dr.motor_strength.range=[0.5,1.1]` | Independent per-actuator forcerange scale, decoupled from `joint_gains`' gain draw. Weak-skewed by default: disabled, forcerange keeps riding the gain scale (a weak-motor world also reads as soft); enabled, it draws its own sample. |

Example enabling every expanded model randomization:

```bash
./training/run.sh train domain_rand=true \
  dr.com_offset.enable=true dr.joint_gains.enable=true \
  dr.dof.enable=true 'dr.dof.damping=[0.9,1.1]' \
  'dr.dof.armature=[0.9,1.1]' 'dr.dof.frictionloss=[0.9,1.1]' \
  dr.foot_friction.enable=true 'dr.foot_friction.range=[0.8,1.2]' \
  dr.motor_strength.enable=true 'dr.motor_strength.range=[0.5,1.1]'
```

`friction-probe` checks the foot-friction draw end to end on the backend the
box has: it samples per-foot multipliers, stands the robot on the flat floor
and compares each foot-floor contact's friction with the draw. Equal-priority
geoms combine friction with an element-wise max, so without the priority fix
a draw below the floor's 0.9 would never reach the contact.

```bash
./training/run.sh friction-probe --backend warp --num-envs 8 --range 0.4 1.35
```

### Latency and encoder randomization

These switches exist only on `joystick`; they do **not** depend on
`domain_rand`.

| Feature | Enable | Default tunables | Semantics |
|---|---|---|---|
| Substep control latency | `task.env.latency.enable=true` | `min_substeps=0`, `max_substeps=5` | Uniform integer delay sampled at reset, inclusive. `0` applies the new action immediately; `n_substeps` applies the previous motor targets for the whole control period, delaying the new action until the next period. Bounds must satisfy `0 <= min <= max <= n_substeps`. |
| Encoder-zero offset | `task.env.encoder.enable=true` | `range=0.02` rad | Each joint gets uniform ±range offset, added to observed joint angle and subtracted from written motor target. |

With latency randomization off, the legacy `action_delay=1` keeps a fixed
one-control-period delay. It is not a zero-latency setting. Under Brax
auto-reset, both sampled values remain fixed per parallel environment for a
training run rather than being re-sampled for every episode.

```bash
./training/run.sh train task=joystick \
  task.env.latency.enable=true \
  task.env.latency.min_substeps=0 task.env.latency.max_substeps=5 \
  task.env.encoder.enable=true task.env.encoder.range=0.02
```

## Terrain curriculum

Joystick only, off by default. When `task.env.terrain.enable=true`:

- the env loads the terrain arena instead of the flat floor,
- every env spawns on a tile, easy rows first,
- heights and foot contact are measured from the terrain surface, so a step
  up is not a height error and a box top is not a fall,
- after each episode the env moves to a harder or easier tile.

Build the arena first. The files are generated and gitignored:

```bash
./training/run.sh build-terrain            # default 10-row arena, seed 0
./training/run.sh train task=joystick ++task.env.terrain.enable=true
```

If any of the four generated files is missing, the env tells you to run
`build-terrain`.

There are three arena kinds, each with its own file set next to the robot XML:
`train` (the shuffled curriculum arena), `eval` (the fixed measurement course,
see [Terrain measurement suite](#terrain-measurement-suite)) and `test` (the
test suite's scratch arena). Separate sets are what keeps a measurement or a
test run from overwriting the arena a policy trained on.

The train arena can carry a flat row:

```bash
./training/run.sh build-terrain --arena train --flat-row
./training/run.sh train task=joystick ++task.env.terrain.enable=true \
  ++task.env.terrain.flat_row=true
```

Level 0 is then genuinely flat ground, which a difficulty-0 row is not — its
stair tiles still have a 2 cm riser — and every terrain row sits one level
higher, so subtract one before comparing `terrain_lvl_train` against a run
without the row. The build and `task.env.terrain.flat_row` have to agree:
the env refuses a mismatched pair rather than train a level short of its
configuration or log levels that name a different terrain. `spawn_level=0`
pins every spawn to the flat row on such an arena. The eval arena never
carries the row (`build-terrain --arena eval --flat-row` is an error), so an
eval load ignores the flag and a flat-row policy still scores on the standard
course. The `train.sh` payload takes `FLAT_ROW=1` next to `TERRAIN=1` for its
in-job build.

Keys under `task.env.terrain`:

| Key | Default | Meaning |
|---|---:|---|
| `enable` | `false` | Terrain scene and curriculum on. |
| `arena` | `train` | Which generated file set to load: `train`, `eval` or `test`. |
| `flat_row` | `false` | Train on the arena variant built with `build-terrain --arena train --flat-row`: level 0 is flat ground, every terrain row one level higher. The build and the key must agree or the load is refused; an eval load ignores the flag, since the eval arena never carries the row. |
| `spawn_yaw` | `true` | Random heading at every spawn. |
| `pad_jitter` | `0.15` m | Spawn scatter around the pad centre. Keep it under 0.2: the pad is 0.6 m in radius and a standing robot needs 0.36 m of it. |
| `init_level_frac` | `0.5` | First spawns come from the easiest half of the rows. |
| `spawn_level` | `-1` | Pin every spawn to one row (0-based, clamped to the arena). `-1` keeps the `init_level_frac` sampling. For videos and debugging; the curriculum still moves the level after the first episode. |
| `demote_fraction` | `0.5` | See the rule below. |

The rule, from legged_gym, applied when an episode ends by fall or by
timeout:

- Walked further than half a tile: one row harder.
- Covered less than `demote_fraction` of the distance the commands asked for
  over a **full episode**: one row easier. The commanded distance actually
  accumulated is scaled by `episode_length / steps_lived`, because this env
  resamples the command mid-episode and legged_gym's literal rule (half the
  reset command's speed times the whole episode) is not available. A timeout
  lived the whole episode, so its threshold is unchanged; a fall at step 50 of
  1000 gets twenty times the distance commanded so far, so almost any fall
  demotes. That is the escape valve legged_gym has, and falling is the dominant
  termination on terrain.
- Standing episodes stay where they are.
- Promotion wins when both fire, matching legged_gym's `move_down * ~move_up`.
- Promoted while already at the top row: random row, so easy terrain stays
  in training.

Two things worth knowing about the rule:

- The promote threshold is hardcoded at half a tile while demote is
  configurable. Half a tile knows nothing about how long the obstacle is, which
  is what bounds the stair flight at six steps: treads end 1.25 m from the tile
  centre and promotion needs 1.5 m, so an env crosses the flight and then
  reaches the threshold on flat ground. A longer flight needs promotion defined
  against the feature band first.
- Timeouts produce some spurious demotions. A perfect tracker demotes on 14 to
  17 percent of timeout episodes because resampled command directions cancel,
  so the commanded distance exceeds the net displacement. Harmless for drift,
  worth remembering when reading `terrain_lvl_train`.
- The walked distance is Euclidean while every feature is a concentric square,
  so on a diagonal heading 1.5 m is only 1.06 m out in Chebyshev terms — tread
  4 of 6. About a quarter of headings promote without crossing the whole
  flight. Kept as-is because it is legged_gym's rule; the measurement scan
  uses Chebyshev radii for exactly this reason.

Each env keeps one terrain type for the whole run. Only the row changes. The
robot is teleported between tiles. That is safe because no observation
contains world position or heading. Observations did not change at all, so
checkpoints stay compatible with flat runs. Letting the policy see the
terrain ahead of it (a height scan in the observations) is a later change;
today it walks blind and feels the ground only through its own joints.

Watching it work: `terrain_lvl` in the logs comes from the eval env, which
starts fresh at every evaluation, so it will not climb. To watch the real
curriculum, run with `++ppo.log_training_metrics=true` and read
`terrain_lvl_train`. Any terrain training preset should set this flag;
`terrain_blind_v1` is the one that does.

For the curriculum to ratchet at all, the envs must survive between epochs:
the Go1 baseline's `ppo.num_resets_per_eval=10` rebuilds every env ten times
per evaluation, and each rebuild re-randomises the level the env had climbed
to. `train.py` therefore pins `ppo.num_resets_per_eval=0` for terrain runs
(the evaluations still happen; epochs just run longer between resets). An
explicit `++ppo.num_resets_per_eval` override wins over the pin.

The legs collide with terrain, and there is no option to turn that off. The
shins are what hit a riser face: without them a leg swings through the step
wall and the foot lands on the tread, which is a way up the stairs the robot
does not have. It costs contacts and step rate, both measured rather than
assumed.

Foot contact is a height lookup, not a contact force, and that has a blind
band on terrain. A foot pressed against a riser face reads as airborne, and so
does a foot on the ground within about one heightfield cell (0.04 m) of a box
edge, where the lookup returns the box top. Both errors are one-sided and both
land on the stair and step tiles the curriculum aims at.

Warp needs a bigger contact budget. The flat default
(`sim.naconmax_per_env=32`) is too small for terrain, and warp drops extra
contacts silently. Warp allows four contacts per geom-heightfield pair and 22
of the robot's geoms touch the field, so 88 heightfield contacts before any box
contact. Measure it, on the GPU and with `--backend warp` -- the jax backend
sizes its own buffers and never applies the budget, so a jax run cannot tell
you what warp needs:

```bash
./training/run.sh check-terrain --backend warp --arena train
./training/run.sh check-terrain --backend warp --arena eval   # measurement course
./training/run.sh train task=joystick ++task.env.terrain.enable=true \
  ++task.env.sim.naconmax_per_env=128
```

A warp terrain run warns when its budget is below that 88-contact floor, which
the env computes from the robot's own collision set rather than from a rule of
thumb. The floor is not a budget: boxes add to it, and the real number has to be
measured.

There is a second, quieter budget: `sim.njmax`, the constraint rows per world.
Rows past it apply no force with no warning anywhere — no printf, no counter in
the step — and one contact costs about 6 rows at `condim=4` with the default
pyramidal cone, so the flat-scene 320 covers only ~50 simultaneous contacts.
`check-terrain` and `terrain-scan` both record the peak (`nefc_max`) next to
the contact peak; size `sim.njmax` for a training run from that measurement,
the same way `naconmax_per_env` is sized from `nacon_max`.

## Terrain measurement suite

`./run.sh terrain-scan` scores a checkpoint on a fixed course and writes
`runs/<name>/terrain_scan.json`. `report` reads that file rather than
recomputing it, so a scan can run on a GPU machine and a laptop can render the
report from it. `training/jobs/terrain_scan.sh` is the parameterized payload.

```bash
./training/run.sh build-terrain --arena eval          # 12 rows, 0.40 m pads
./training/run.sh terrain-scan --run runs/wojtek_terrain_v2 \
  --baseline <HF_ORGANIZATION>/wojtek-terrain-v1
./training/run.sh terrain-scan --list-cells           # the 43 cells and bars
./training/run.sh terrain-scan --run runs/wojtek_terrain_scan_v5 --scan dark
```

`--scan dark` feeds the actor a zero height scan on the same course, which is
the blind fallback score of a scan-observing policy; the critic's grid is
untouched. The mode is recorded in the document as `scan_mode`, so a dark scan
is not mistaken for a clean one. Either way the measurement env turns the
sensor's corruption model off (`height_scan.corrupt.enable=false`) and keeps
the camera mask and the sample-and-hold, so the score measures the sensor as
built rather than one draw from its failure distribution.

The course, per cell: spawn at the tile centre on a fixed heading, walk out
until the base is 1.45 m from the centre (one crossing), then the commanded
forward speed flips sign and the robot walks back to within 0.30 m (the
second). Four crossings. A run passes when all four finish inside its step
budget with no fall. 8 headings x 4 start offsets x 2 noise draws = 64 runs per
cell and commanded speed, at 0.4 / 0.7 m/s: 5504 runs, about 6.3M environment
steps. Nothing is sampled, so two scans of one checkpoint agree. A third speed,
0.2 m/s, was measured and dropped on 2026-07-27: finishing inside the step
budget needs 62% speed tracking, the measured policies track about 60% at 0.2,
and every one of them scored 0 there.

The two noise draws are the same 32 physical starts, run twice. That doubling
is what suite version `v3` is: the measured test-retest wobble moved one
cell/speed pair by up to 6 of 32 runs across `--eval-seed` draws, and that
spread is noise-draw variance, so a second draw of the same starts attacks
exactly that axis without changing what is being tested. Draw is the outermost
index in the course, so runs 0-31 are the old 32-run layout and 32-63 repeat it
on the keys the per-run split hands them. Every number recorded out of 32 is a
`v2` number and is not comparable; the relative gate refuses a baseline from a
different suite version.

Those radii are **Chebyshev** distances from the tile centre — `max(|dx|, |dy|)`,
not the Euclidean radius — because that is how the terrain is built. Every
feature is a concentric square: the slope frustum and the stair pit are carved
against `terrain._cheby`, and the scattered boxes are held inside a square
reach. Read as a Euclidean radius, 1.45 m on a 45-degree heading leaves the base
only 1.03 m out in Chebyshev — still on tread 4 of 6 — so half of the eight
headings would complete a "crossing" without climbing the last two risers, and
one bar would be covering two different tests.

A diagonal heading therefore walks √2 as far as an axis heading to reach the
same radius, and each run gets its own step deadline sized on its own distance.
One shared budget would hand the axis headings that extra slack, which would
make the speed a run has to sustain — and so the effective difficulty —
heading-dependent. Every run has to sustain the same 62% of its commanded speed.

One known limitation, accepted: at 1.45 m the base is 0.05 m from the tile
border, so the leading feet reach 0.21 m into a neighbouring tile at the
turnaround (0.255 m on a diagonal, where the foot corner leads). That is
unavoidable with a six-step flight on a 3 m tile — clearing the last riser
needs the base at 1.25 + 0.257 = 1.51 m. Which neighbour depends on the
heading: along x it shares the row (same difficulty, different type); along y
it is the same type one row over — a harder or easier difficulty, the worst
case; on a diagonal it is the corner tile, which differs in both. Tiles are
flat to within 2 cm at the border itself, but 0.21 m in the neighbour's own
features have begun — worst case a steep inverted slope, 10 cm below grade at the
hardest gated row and 15 cm at the frontier rows. The crossing is already scored
when the base reaches the radius, so this is the state at the end of a leg, not
terrain the robot had to cross.

Crossings rather than distance walked, because a stair tile is 3 m across and
its treads only occupy the band from 0.60 m to 1.25 m from the centre -- "half
the commanded distance" can be walked on the flat pad without meeting the
obstacle. Counting crossings also gives a failure a reason: the robot fell, or
it never reached 1.45 m, and on a stair those are different problems.

The reversal is deliberate. A 180 degree turn on a 13 cm tread might be the
hardest thing in the suite, and the number would then measure turning instead
of climbing. Walking backwards is inside every real preset's command box.

43 cells, 18 of them gated:

| Family | Cells | Bar |
|---|---|---|
| Rough | 1 / 2.5 / 4 cm | 95 / 80 / 60 % |
| Slope up and down | 8 / 15 / 22 deg | 95 / 80 / 60 % |
| Stairs up and down | 3 / 5 / 7 cm riser | 95 / 80 / 60 % |
| Stairs up and down | 9 cm riser | tracked |
| Steps | 2.5 / 5 / 6.5 cm | 95 / 80 / 60 % |
| Steps | 8 cm | tracked |
| Rubble and wave | 3 rows each | tracked |
| Every type at difficulty 1.2 and 1.4 | 16 cells | tracked |

As counts out of 64 runs the bars are 61, 52 and 39. Every threshold is
printed with where its number came from: `plan` at 0.4 m/s, which is the only
speed the terrain plan sets bars for, and `provisional` at 0.7, where the same
numbers were carried across rather than invented. A provisional failure is a
prompt to check the bar. After the first terrain keeper, measured numbers
replace them.

The 8 cm step is tracked, not gated at 60 % as the plan has it: 8 cm is 0.64 of
this robot's 12.5 cm hip height, above the 0.5-0.6 the same document calls the
blind limit. A 6.5 cm cell takes the 60 % bar instead. The 8 cm number is still
measured and reported.

Cell names (`pyramid_stairs_5cm`) are stable identifiers: a gate compares them
against a baseline, so renaming one retires its history.

Two gates. The absolute bars above, and the relative rule from the terrain
plan, resized to the scan's measured noise: no cell drops more than 25 points
against the previous keeper. The terrain plan wrote 10, but three scans of one
checkpoint on different `--eval-seed` draws swung a single cell/speed pair by
up to 18.75 points, so 10 fails on noise alone; the whole-course pass total
moved only ~2% in the same test, which is where a tighter gate should live
(2026-07-27 validation report). That measurement was taken on the 32-run
course, so at 64 runs the same per-run wobble is worth half the points and 25
is conservative — tighten it from a fresh test-retest measurement at 64, not by
halving the old number. The
baseline is an input, published with the keeper it came from, not a file this
repository keeps -- a best-ever number held in the repo hides which run set the
bar, so a rejected policy could leave a bar behind nobody can trace.
`--baseline` takes a path, a directory holding `terrain_scan.json`, or an HF
reference `org/name[@rev]`. Three rules keep the comparison honest: a cell the
baseline never measured counts as nothing to compare against rather than a
failure; a baseline from a different arena is refused, because scores from two
terrains are not comparable; and a baseline from a different engine is refused
until the cross-engine spread is measured, since warp is float32 with a contact
budget and CPU MuJoCo is float64 with none.

Warp on the GPU produces the reported numbers. MJX-jax is a cross-check on a
few cells (`--backend jax --cells ...`): it has no contact budget, so a
disagreement points at the budget. The scan records the peak `nacon` it saw and
refuses to call an overflowed run a measurement.

What the suite does not measure: it is the nominal plant only, with no pushes
and no randomized dynamics. Robustness on terrain needs its own protocol.

## PPO configuration

The baseline is `mujoco_playground.config.locomotion_params`' tuned Go1 PPO
configuration. At the pinned dependency versions, `build_ppo_params()` exposes
these baseline values (this table is not every keyword accepted by the upstream
Brax trainer):

| Key | Default |
|---|---:|
| `ppo.action_repeat` | `1` |
| `ppo.batch_size` | `256` |
| `ppo.discounting` | `0.97` |
| `ppo.entropy_cost` | `0.01` |
| `ppo.episode_length` | `1000` (normally replaced by task episode length) |
| `ppo.learning_rate` | `0.0003` |
| `ppo.max_grad_norm` | `1.0` |
| `ppo.normalize_observations` | `true` |
| `ppo.num_envs` | `8192` |
| `ppo.num_evals` | `10` |
| `ppo.num_eval_envs` | `128` upstream default | Optional: not present in the baseline map, but supported by the current Brax trainer. The training env allocates for `max(num_envs, num_eval_envs)`. |
| `ppo.num_minibatches` | `32` |
| `ppo.num_resets_per_eval` | `10` | Terrain runs pin this to 0 (`train.py`): each periodic reset rebuilds every env and re-randomises its curriculum level, so at 10 the curriculum never ratchets. An explicit override still wins. |
| `ppo.num_timesteps` | `200000000` |
| `ppo.num_updates_per_batch` | `4` |
| `ppo.reward_scaling` | `1.0` |
| `ppo.unroll_length` | `20` |
| `ppo.network_factory.policy_obs_key` | `state` |
| `ppo.network_factory.value_obs_key` | `privileged_state` |
| `ppo.network_factory.{policy,value}_hidden_layer_sizes` | `[512,256,128]` |

Use `++ppo.*` on the CLI. `task.ppo` establishes task/preset values first;
root `ppo` overrides it. The project forwards safe PPO keyword overrides to
the installed Brax trainer; inspect its current signature after dependency
updates rather than assuming that every upstream keyword is stable. Avoid
implementation-owned inputs such as `environment`, `eval_env`,
`wrap_env_fn`, `randomization_fn`, checkpoint paths, `seed`, and
`network_factory`: `train.py` supplies those itself. Examples:

```bash
./training/run.sh train +experiment=locomotion \
  ++ppo.num_timesteps=300000000 ++ppo.num_envs=8192 \
  ++ppo.batch_size=256 ++ppo.learning_rate=0.0003
```

To inspect the installed baseline after a dependency update:

```bash
(cd training && ./.venv/bin/python -c \
  'from wojtek_rl.train import build_ppo_params; print(build_ppo_params({}, False).to_dict())')
(cd training && ./.venv/bin/python -c \
  'import inspect; from brax.training.agents.ppo import train; print(inspect.signature(train.train))')
```

## Experiment presets

Select a preset with `+experiment=<name>`.  Each is a normal Hydra overlay,
so command-line values can still override it.

| Preset | Base task | Purpose |
|---|---|---|
| `getup` | getup | Safe fall recovery baseline. |
| `jump` | jump | Commanded jump baseline. |
| `jump_v3` | jump | Higher torque and deliberate wind-up jump recipe. |
| `locomotion` | joystick | Deployable no-IMU locomotion, stand/walk/trot schedule. |
| `locomotion_3090` | locomotion | Single-GPU locomotion variant. |
| `locomotion_v2` … `locomotion_v8` | locomotion | Recorded locomotion iteration recipes. |
| `run` | joystick | Extended command range / faster gait for running. |
| `run_v2` | joystick | Running iteration that includes a warm-start path. |
| `springy_phase_b` | joystick | 2026-07-12 reward redesign: clock-free, height-command-free springy locomotion with the anti-splay package, full penalty weight. Self-contained; does not inherit `locomotion`. |
| `springy_phase_a` | springy_phase_b | Same recipe at a 0.3x smoothness/torque penalty curriculum; phase B restores from its checkpoint. |
| `stiff_probe_kp40` | springy_phase_a | Stiffness-arm ranking probe: `pd_kp=40`/`pd_kd=1.4`. |
| `stiff_probe_kp60` | springy_phase_a | Stiffness-arm ranking probe: `pd_kp=60`/`pd_kd=1.73`. |
| `stiff_phase_a` | springy_phase_a | Phase A of the kp40 probe winner's full two-phase run: `pd_kp=40`/`pd_kd=1.4`. |
| `stiff_phase_b` | springy_phase_b | Phase B of the kp40 probe winner: `pd_kp=40`/`pd_kd=1.6` (bumped for damping margin), restored from phase A. |
| `locomotion_stiff_v1` | joystick | Frozen keeper of the 2026-07-17 stiffness experiment: kp40/kd1.6 stiff PD operating point on the springy clock-free design. Self-contained; does not inherit `springy_phase_*`/`stiff_phase_*`. |
| `stiff_ladder_kp50` | springy_phase_b | Rung 1 of the stiffness ladder off the kp40/kd1.6 keeper: `pd_kp=50`/`pd_kd=1.79`. |
| `stiff_ladder_kp60` | springy_phase_b | Rung 2 of the stiffness ladder: `pd_kp=60`/`pd_kd=1.96`, restored from the accepted kp50 rung. |
| `stiff_ladder_kp70` | springy_phase_b | Rung 3 of the stiffness ladder: `pd_kp=70`/`pd_kd=2.12`, restored from the accepted kp60 rung. |
| `stiff_ladder_kp80` | springy_phase_b | Rung 4 of the stiffness ladder: `pd_kp=80`/`pd_kd=2.26`, restored from the accepted kp70 rung. |
| `stiff_ladder_kp90` | springy_phase_b | Rung 5, an extension of the kp50-80 ladder: `pd_kp=90`/`pd_kd=2.40`, restored from the kp50-80 ladder's winner (kp80). |
| `stiff_ladder_kp100` | springy_phase_b | Rung 6, extending the ladder further: `pd_kp=100`/`pd_kd=2.53`, restored from the accepted kp90 rung. |
| `locomotion_stiff_kp80_v1` | joystick | Keeper v2, safe-without-compensation: consolidated stiffness-ladder rung kp80/kd2.26 (+800M steps beyond the ladder). Self-contained; does not inherit `springy_phase_*`/`stiff_ladder_*`. |
| `locomotion_stiff_kp90_v1` | joystick | Keeper v2, maximum stiffness (sim ceiling at the 9 Nm cap): consolidated stiffness-ladder rung kp90/kd2.40 (+800M steps beyond the ladder). Self-contained; does not inherit `springy_phase_*`/`stiff_ladder_*`. Deploy only after matching the real actuators' stand-sag alpha — see the preset file's deployment contract. |
| `terrain_blind_v1` | joystick | Verification only — the preset the 2026-07-27 GPU validation session ran; do not launch a real terrain run from it. Terrain policies never start from a flat keeper: the family starts from scratch with the IMU in the observations, and terrain policies fine-tune from terrain policies only (family rule, 2026-07-27). This preset keeps the kp80c obs layout (no IMU) so that session could restore the kp80c checkpoint. Its measured warp budgets carry over to the real terrain preset. |
| `terrain_blind_v2` | joystick | The terrain family's founding preset, and the one to launch a real terrain run from; `terrain_blind_v1` is its verification-only predecessor. Blind locomotion on the tiled train arena, trained from scratch with the IMU in the observations, on the flattened phase-C kp40 operating point (`stiff_phase_c`, published as `<HF_ORGANIZATION>/wojtek-stiff-locomotion-v2`). Self-contained; does not inherit `stiff_phase_c`/`locomotion_stiff_v1`. |
| `terrain_blind_v3` | joystick | Run two of the terrain family, from scratch: `terrain_blind_v2`'s configuration plus clean backward commands, product/relative tracking, an orientation tolerance cone, apex and landing shaping, mirrored worlds, and a flat level-0 row, at a 32 Nm torque cap. Deployable via the exported contract, which carries the 32 Nm cap; the launch clamp follows it. |
| `terrain_blind_v4` | joystick | Run three of the terrain family, from scratch: `terrain_blind_v3`'s operating point plus wider tracking kernels, the tracking-gated gait shaping, the no-progress cut, and the far-field tracking blend. |
| `terrain_blind_v4_1` | joystick | `terrain_blind_v4` plus foot-friction randomization, the roughness gate on the gait shaping, a slow clean-walk command draw, and the clean height scan on the critic. The actor stays on the family's 46-dim blind layout, so checkpoints interchange with `terrain_blind_v4`. |
| `terrain_scan_v5` | joystick | `terrain_blind_v4_1` with the height scan in the actor observation as well (71-dim actor). "Corrupted" refers to the training-time sensor model on the actor's copy only — per-episode noise / drift / blackout regimes plus point dropout (`height_scan.corrupt`) — while the critic keeps the clean grid; it says nothing about which runs are launched. Its checkpoints share nothing with the blind family. |

Read the matching file in [`conf/experiment`](../wojtek_rl/conf/experiment)
before choosing a historical version: the comments explain its intended
comparison and any inherited or restore assumptions. For example:

```bash
./training/run.sh train +experiment=jump_v3 run_name=wojtek_jump_v3_seed1 seed=1
./training/run.sh train +experiment=locomotion_v8 \
  run_name=wojtek_locomotion_v8_seed1 wandb.enable=false
```

The exact preset deltas are below so an agent can choose intentionally rather
than infer them from a historical run name:

| Preset | `run_name` | Effective overrides beyond its base |
|---|---|---|
| `getup` | `fbb_getup_v1` | Selects `task=getup`. |
| `jump` | `fbb_jump_v1` | Selects `task=jump`. |
| `jump_v3` | `fbb_jump_v3` | `task.env.forcerange=12`, `task.env.action_scale=0.8`, `task.env.countdown_steps=40`, `task.env.reward.scales.{flight,launch_vel,action_rate,action_accel}={15,5,-0.4,-0.2}`. |
| `locomotion` | `fbb_loco_v1` | Joystick + `obs=no_imu`; `task.env.command.{vx,vy,wz,height}={[-0.8,1.2],[-0.5,0.5],[-1,1],[0.09,0.17]}`, `task.env.gait.{freq,swing_height,trot_band}={[1.4,3.2],0.035,[0.35,0.55]}`, `task.env.push.vel=0.5`, `task.ppo.num_timesteps=300000000`. |
| `locomotion_3090` | `fbb_loco_3090_v1` | Inherits `locomotion`, changes only the run name. |
| `locomotion_v2` | `fbb_loco_v2` | `locomotion` + `task.env.reward.height_sigma=5e-4`, `task.env.reward.scales.height_tracking=2`, `task.env.reward.scales.stand_still=-1`. |
| `locomotion_v3` | `fbb_loco_v3` | v2 fields + `task.env.gait.freq=[1.4,3.6]`, `task.env.reward.scales.tracking_lin_vel=2`. |
| `locomotion_v4` | `fbb_loco_v4` | v3 fields + `task.env.command.zero_prob=0.25`, `task.env.reward.scales.stand_still=-2.5`. |
| `locomotion_v5` | `fbb_loco_v5` | v3 fields + `task.env.action_filter=0.5`, `task.env.command.zero_prob=0.25`. |
| `locomotion_v6` | `fbb_loco_v6` | v3 fields + `task.env.command.zero_prob=0.25`. |
| `locomotion_v7` | `fbb_loco_v7` | v6 fields + `task.env.reward.scales.height_tracking=2.5`, `task.env.reward.scales.feet_slip=-0.5`. |
| `locomotion_v8` | `fbb_loco_v8` | v6 fields + `task.env.reward.scales.height_tracking=2.5`, `task.env.reward.scales.tracking_lin_vel=2.5`, `task.env.reward.scales.feet_slip=-0.35`. |
| `run` | `fbb_run_v1` | `task.env.command.{vx,vy,wz}={[-0.8,1.8],[-0.5,0.5],[-1,1]}`, `task.env.gait.{freq,swing_height}={[2,3.5],0.04}`, `task.env.push.vel=0.5`. |
| `run_v2` | `fbb_run_v2` | `task.env.command.{vx,vy,wz}={[-0.8,1.8],[-0.5,0.5],[-1,1]}`, `task.env.gait.{freq,swing_height}={[1.8,4.5],0.04}`, `task.env.push.vel=0.5`, historical `restore=runs/fbb_run_v1/checkpoints/000206438400`. |
| `springy_phase_b` | `wojtek_springy_b` | Self-contained joystick preset (no `locomotion` inheritance). `task.env.{action_scale,max_torque,abduction_ctrl_limit,knee_target_max}={[0.25,0.5,0.5],6.0,0.44,3.15}`; `task.env.{latency,encoder}.enable=true`; `task.env.obs.include=[joint_pos,joint_vel,last_act,command]` (drops gyro/gravity/phase from the actor, critic list unchanged); `task.env.command.{vx,vy,wz,height,zero_prob,pure_wz_prob,pure_vy_prob}={[-0.8,1.2],[-0.5,0.5],[-1,1],[0.125,0.125],0.25,0.25,0.2}` (height pinned, not removed); `task.env.push.vel=0.8`; `task.env.gait.{swing_height,air_time_cap}={0.08,0.35}`; `task.env.reward.pose_leg_weight=0.1`; `task.env.reward.scales.{tracking_lin_vel,tracking_ang_vel,height_tracking,lin_vel_z,ang_vel_xy,orientation,torques,torque_rate,torque_limit,action_rate,action_accel,energy,pose,feet_air_time,feet_slip,feet_phase,contact_match,high_step,stand_still,stand_feet_down,termination}={4.0,1.6,0.0,0.0,0.0,-5.0,-6e-4,-0.02,-0.1,-0.25,0.0,0.0,-1.0,4.0,-0.35,0.0,0.0,4.0,-2.5,-30.0,-1.0}`; `dr.motor_strength.enable=true` (`range=[0.5,1.1]`); `ppo.num_timesteps=200000000`. |
| `springy_phase_a` | `wojtek_springy_a` | Inherits `springy_phase_b`, overrides only the penalty curriculum: `task.env.reward.scales.{action_rate,torques,torque_limit}={-0.075,-1.8e-4,-0.03}` (0.3x of phase B; `torque_rate` stays `-0.02`, already below its measured `-0.06` freeze cliff); `ppo.num_timesteps=100000000`. Launch phase B afterward with `restore=<phase-A checkpoint>`; the preset itself does not set `restore`. |
| `stiff_probe_kp40` | `wojtek_stiff40_probe` | Inherits `springy_phase_a`. Stiffness-arm ranking probe: `task.env.{pd_kp,pd_kd,max_torque}={40.0,1.4,9.0}` (gains patched onto the model at load; `max_torque` widened from phase A's `6.0` to the XML's own `±9` forcerange for transient headroom); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=300000000`. |
| `stiff_probe_kp60` | `wojtek_stiff60_probe` | Inherits `springy_phase_a`. Stiffness-arm ranking probe: `task.env.{pd_kp,pd_kd,max_torque}={60.0,1.73,9.0}` (gains patched onto the model at load; `max_torque` widened from phase A's `6.0` to the XML's own `±9` forcerange for transient headroom); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=300000000`. |
| `stiff_phase_a` | `wojtek_stiff_a` | Inherits `springy_phase_a`. Phase A of the kp40 probe winner's full two-phase run: `task.env.{pd_kp,pd_kd,max_torque}={40.0,1.4,9.0}` (kd matches the probe for curriculum continuity; `max_torque` widened to the XML's `±9` forcerange as in the probes); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=1200000000`. |
| `stiff_phase_b` | `wojtek_stiff_b` | Inherits `springy_phase_b`. Phase B of the kp40 probe winner: `task.env.{pd_kp,pd_kd,max_torque}={40.0,1.6,9.0}` (`pd_kd` bumped from the probe's `1.4` for damping margin — its >5 Hz vibration metric exceeded the kp20 baseline on 3 of 4 scenarios — putting the damping ratio at ~1.13x the kp20/kd=1.0 reference, inside the ±20% kd DR band phase A already trained under; `max_torque` overrides phase B's `6.0` back to `9.0`); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=800000000`. Launched with `restore=<stiff_phase_a checkpoint>`; the preset itself does not set `restore`. **Deployment must match phase B** (`pd_kp=40`/`pd_kd=1.6`/`max_torque=9.0` in `wojtek_real.urdf.xacro`) if this becomes a keeper. |
| `locomotion_stiff_v1` | `wojtek_loco_stiff_v1` | Frozen keeper, self-contained (no `springy_phase_*`/`stiff_phase_*` inheritance): reproduces `stiff_phase_b`'s effective config verbatim — `task.env.{action_scale,pd_kp,pd_kd,max_torque,abduction_ctrl_limit,knee_target_max}={[0.25,0.5,0.5],40.0,1.6,9.0,0.44,3.15}`; `task.env.{latency,encoder}.enable=true`; `task.env.obs.include=[joint_pos,joint_vel,last_act,command]`; command/push/gait/reward blocks identical to `springy_phase_b`; `dr.motor_strength.enable=true` (`range=[0.5,1.1]`) and `dr.joint_gains.enable=true` (`gain_pct=0.2`, `kd_pct=0.2`); `ppo.num_timesteps=800000000`. FROZEN from the trained run `wojtek_stiff_b_20260717_235321` (commit `eb8bad2`) so it stays reproducible while `stiff_phase_*` evolves. Deployment must match: `pd_kp=40`/`pd_kd=1.6`/`max_torque=9.0` on the robot. |
| `stiff_ladder_kp50` | `wojtek_stiff_kp50` | Inherits `springy_phase_b`. Ladder rung 1 off the kp40/kd1.6 keeper (`wojtek_stiff_b_20260717_235321`): `task.env.{pd_kp,pd_kd,max_torque}={50.0,1.79,9.0}` (`pd_kd` holds the keeper's damping ratio: `kd=1.6*sqrt(kp/40)`); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=400000000`. Launched with `restore=<keeper checkpoint>`; the preset itself does not set `restore`. |
| `stiff_ladder_kp60` | `wojtek_stiff_kp60` | Inherits `springy_phase_b`. Ladder rung 2: `task.env.{pd_kp,pd_kd,max_torque}={60.0,1.96,9.0}` (`pd_kd=1.6*sqrt(kp/40)`); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=400000000`. Launched with `restore=<stiff_ladder_kp50 accepted checkpoint>`; the preset itself does not set `restore`. |
| `stiff_ladder_kp70` | `wojtek_stiff_kp70` | Inherits `springy_phase_b`. Ladder rung 3: `task.env.{pd_kp,pd_kd,max_torque}={70.0,2.12,9.0}` (`pd_kd=1.6*sqrt(kp/40)`); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=400000000`. Launched with `restore=<stiff_ladder_kp60 accepted checkpoint>`; the preset itself does not set `restore`. |
| `stiff_ladder_kp80` | `wojtek_stiff_kp80` | Inherits `springy_phase_b`. Ladder rung 4: `task.env.{pd_kp,pd_kd,max_torque}={80.0,2.26,9.0}` (`pd_kd=1.6*sqrt(kp/40)`); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=400000000`. Launched with `restore=<stiff_ladder_kp70 accepted checkpoint>`; the preset itself does not set `restore`. |
| `stiff_ladder_kp90` | `wojtek_stiff_kp90` | Inherits `springy_phase_b`. Ladder rung 5, an extension rung beyond the kp50-80 ladder: `task.env.{pd_kp,pd_kd,max_torque}={90.0,2.40,9.0}` (`pd_kd=1.6*sqrt(kp/40)`); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=400000000`. Launched with `restore=<kp50-80 ladder's winner, stiff_ladder_kp80's accepted checkpoint, wojtek_stiff_kp80_20260718_100225>`; the preset itself does not set `restore`. |
| `stiff_ladder_kp100` | `wojtek_stiff_kp100` | Inherits `springy_phase_b`. Ladder rung 6: `task.env.{pd_kp,pd_kd,max_torque}={100.0,2.53,9.0}` (`pd_kd=1.6*sqrt(kp/40)`); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=400000000`. Launched with `restore=<stiff_ladder_kp90 accepted checkpoint>`; the preset itself does not set `restore`. |
| `locomotion_stiff_kp80_v1` | `wojtek_loco_stiff_kp80_v1` | Frozen keeper, self-contained (no `springy_phase_*`/`stiff_ladder_*` inheritance): reproduces `wojtek_stiff_kp80c_20260719_105503`'s effective config verbatim — `task.env.{action_scale,pd_kp,pd_kd,max_torque,abduction_ctrl_limit,knee_target_max}={[0.25,0.5,0.5],80.0,2.26,9.0,0.44,3.15}`; `task.env.{latency,encoder}.enable=true`; `task.env.obs.include=[joint_pos,joint_vel,last_act,command]`; command/push/gait/reward blocks identical to `locomotion_stiff_v1`/`springy_phase_b`; `dr.motor_strength.enable=true` (`range=[0.5,1.1]`) and `dr.joint_gains.enable=true` (`gain_pct=0.2`, `kd_pct=0.2`); `ppo.num_timesteps=800000000` (the consolidation budget only — lineage is the `locomotion_stiff_v1` keeper (2.0B steps) → ladder rungs kp50→60→70→80 (+400M each) → this +800M consolidation, 4.4B cumulative). FROZEN from the trained run `wojtek_stiff_kp80c_20260719_105503` (seed 1), consolidation restored from `wojtek_stiff_kp80_20260718_100225/checkpoints/000412876800`. Battery mean `track_err_rms` 0.05788 (pre-consolidation 0.06305, −8.2%); zero falls in 4/4 scenarios; max torque saturation 3.2%. Robustness grid gates 1-4 all PASS: `alpha=1.58` (uncompensated Kt miscalibration) → mean 0.0527; torque envelope `5,12` → mean 0.0579; actuator lag ≤10 ms harmless (proven on the pre-consolidation rung). Role: the safe-without-compensation keeper. Deployment must match: `pd_kp=80`/`pd_kd=2.26`/`max_torque=9.0` on the robot. |
| `locomotion_stiff_kp90_v1` | `wojtek_loco_stiff_kp90_v1` | Frozen keeper, self-contained (no `springy_phase_*`/`stiff_ladder_*` inheritance): reproduces `wojtek_stiff_kp90c_20260719_105503`'s effective config verbatim — `task.env.{action_scale,pd_kp,pd_kd,max_torque,abduction_ctrl_limit,knee_target_max}={[0.25,0.5,0.5],90.0,2.40,9.0,0.44,3.15}`; `task.env.{latency,encoder}.enable=true`; `task.env.obs.include=[joint_pos,joint_vel,last_act,command]`; command/push/gait/reward blocks identical to `locomotion_stiff_v1`/`springy_phase_b`; `dr.motor_strength.enable=true` (`range=[0.5,1.1]`) and `dr.joint_gains.enable=true` (`gain_pct=0.2`, `kd_pct=0.2`); `ppo.num_timesteps=800000000` (the consolidation budget only — lineage is the `locomotion_stiff_v1` keeper (2.0B steps) → ladder rungs kp50→60→70→80→90 (+400M each; kp90 restored from the kp80 ladder winner, `wojtek_stiff_kp80_20260718_100225`) → this +800M consolidation, 4.8B cumulative). FROZEN from the trained run `wojtek_stiff_kp90c_20260719_105503` (seed 1), consolidation restored from `wojtek_stiff_kp90_20260718_141500/checkpoints/000412876800`. Battery mean `track_err_rms` 0.05468 (pre-consolidation 0.059125, −7.5%); zero falls in 4/4 scenarios; max torque saturation 4.3%. Robustness grid gates 1-4 all PASS: `alpha=1.58` → mean 0.0498, turn `vel_err_overall` 0.16 (consolidation fixed the pre-consolidation rung's marginal `alpha=1.58` result — turn `vel_err_overall` 0.21-0.23 across lag 0/5/10 ms, over gate 2's 0.2 threshold); torque envelope `5,12` → mean 0.0546. kp100 (one rung further) was rejected at 5.7% saturation, over the ladder's 5% gate — kp90 is the stiffest accepted operating point at the 9 Nm cap. **DEPLOYMENT CONTRACT:** deploy ONLY after measuring the real actuators' stand-sag `alpha`; command `pd_kp/alpha`, `pd_kd/alpha`, and torque cap `9/alpha` so the physical plant matches training. `wojtek_real.urdf.xacro` and the launch file's `max_torque` MUST carry the matched values before any robot use; `HEIGHT_TABLE` must be re-measured at the matched operating point. |
| `terrain_blind_v1` | `wojtek_terrain_blind_v1` | Self-contained (no `locomotion_stiff_kp80_v1` inheritance): repeats that keeper's operating point verbatim — `task.env.{action_scale,pd_kp,pd_kd,max_torque,abduction_ctrl_limit,knee_target_max}={[0.25,0.5,0.5],80.0,2.26,9.0,0.44,3.15}`; `task.env.{latency,encoder}.enable=true`; `task.env.obs.include=[joint_pos,joint_vel,last_act,command]`; command/push/gait/reward blocks identical to `locomotion_stiff_kp80_v1`; `dr.motor_strength.enable=true` (`range=[0.5,1.1]`) and `dr.joint_gains.enable=true` (`gain_pct=0.2`, `kd_pct=0.2`). Terrain on top: `task.env.terrain={enable:true,arena:train}`, `task.env.sim.naconmax_per_env=88` (the env's heightfield-only floor, 22 colliding geoms × 4 contacts), `task.env.sim.njmax=512` (measured: worst `nefc_max` anywhere is 228, in forced-fall states on the eval arena — 512 is 2.2× that), `ppo.log_training_metrics=true` (for `terrain_lvl_train`), `ppo.num_timesteps=800000000` (provisional — the H100 sizing session owns the real budget). `train.py` pins `ppo.num_resets_per_eval=0` for terrain runs; the preset does not set it. The session launched it with `restore=<kp80c keeper checkpoint>` and `XLA_PYTHON_CLIENT_PREALLOC=false`; the preset itself sets neither. Verification only: terrain policies never start from a flat keeper — the family starts from scratch with the IMU in the observations, and fine-tunes within itself (family rule, 2026-07-27). This preset predates the rule. Deployment must match: `pd_kp=80`/`pd_kd=2.26`/`max_torque=9.0` on the robot. |
| `terrain_blind_v2` | `wojtek_terrain_blind_v2` | The terrain family's founding preset. Self-contained (no `stiff_phase_c`/`locomotion_stiff_v1` inheritance): flattens phase C's effective config, the current head of the kp40 line and the deployed flat policy (`<HF_ORGANIZATION>/wojtek-stiff-locomotion-v2`) — `task.env.{action_scale,pd_kp,pd_kd,max_torque,abduction_ctrl_limit,knee_target_max}={[0.25,0.5,0.5],40.0,1.6,9.0,0.44,3.15}`; `task.env.{latency,encoder}.enable=true`; `task.env.command.{vx,vy,wz,height,zero_prob,pure_wz_prob,pure_vy_prob,arc_prob,arc_vx}={[-0.8,1.2],[-0.5,0.5],[-1.5,1.5],[0.10,0.16],0.25,0.25,0.2,0.2,[0.3,0.8]}`; `task.env.push.vel=0.8`; `task.env.gait.{swing_height,air_time_cap}={0.08,0.35}`; `task.env.reward.{tracking_sigma,pose_leg_weight}={0.1,0.1}` and phase C's scales (`tracking_ang_vel=2.4`, `height_tracking=1.0`, the rest as in `locomotion_stiff_v1`); `dr.motor_strength.enable=true` (`range=[0.5,1.1]`) and `dr.joint_gains.enable=true` (`gain_pct=0.2`, `kd_pct=0.2`). Family obs layout: `task.env.obs.include=[gyro,gravity,joint_pos,joint_vel,last_act,command]` — the IMU is in, which makes the layout incompatible with every flat checkpoint, so training from scratch is the family default; starting from a keeper or adding a flat warm-up is a per-run decision, made explicitly (softened 2026-07-28). Terrain on top: `task.env.terrain={enable:true,arena:train}`, `task.env.sim.naconmax_per_env=116` (the 2026-07-27 chessboard base collider: 29 geoms that can pair with the heightfield — 20 leg geoms plus the 9-cell base chessboard — at 4 contacts each; the 2026-07-27 validation session measured a walking-scan peak of 12 contacts/env against the previous 88 pool with nothing tripped), `task.env.sim.njmax=640` (that session's worst measured `nefc_max` was 228, in forced-fall states on the eval arena with the 2-box base; the 9-cell chessboard can put more, smaller contacts down in fall states, so the margin is widened until `check-terrain` measures the new plant), `ppo.log_training_metrics=true` (for `terrain_lvl_train`), `ppo.num_timesteps=2000000000` (from-scratch budget at the kp40 keeper's 2.0B scale; the launch session owns the final number). No `restore` (from scratch, the family default). Launch it with `./training/run.sh build-terrain --arena train` first and `XLA_PYTHON_CLIENT_PREALLOC=false`; `train.py` pins `ppo.num_resets_per_eval=0` for terrain runs. Scan baseline: score the v2 keeper once on the eval arena before gating any terrain policy. Deployment must match: `pd_kp=40`/`pd_kd=1.6`/`max_torque=9.0` on the robot, plus the IMU wired into the policy node (CANdle-hat BMI, PR #75) and phase C's per-command height anchor. |
| `terrain_blind_v3` | `wojtek_terrain_blind_v3` | v2's operating point (obs layout, kp40/kd1.6, DR, latency/encoder, base-contact termination, command and gait blocks) + the run-two changes, after run one (`wojtek_terrain_blind_v2c_20260727`) trained clean and scored 0 of 2752 on the measurement suite by refusing every backward command: `task.env.max_torque=32.0` (the X8-32 peak) with `task.env.reward.torque_limit_frac=0.25` and `task.env.reward.scales.torque_limit=-0.2`, so the hinge starts charging at the 8 Nm rated torque — a standing price on sustained torque rather than a rarely-touched saturation guard; `task.env.reward.orientation_tol_deg=20.0` (a tolerance cone, because the flat-referenced penalty prices the body pitch that climbing a riser requires; a nosedive is far outside it); `task.env.command.{pure_back_prob,back_vx}={0.2,[-0.8,-0.2]}` (clean backward commands, drawn at 0.2 and landing at ~0.15 of commands after the zero overwrite); `task.env.reward.{tracking_product,tracking_relative}=true` with `task.env.reward.tracking_rel_sigma=0.25` replacing v2's `tracking_sigma`, which is inert in relative mode; `task.env.reward.apex_target=0.05` with `task.env.reward.scales.{feet_apex,feet_landing}={5.0,-0.5}` (`high_step` stays `4.0`; both terms run); `task.env.symmetry={enable:true,mirror_prob:0.5}`; `task.env.terrain.flat_row=true` (level 0 is flat ground and every terrain row sits one higher, so subtract one before comparing `terrain_lvl_train` with run one); `task.env.sim.{naconmax_per_env,njmax}={116,640}`; `ppo.num_timesteps=2000000000` (from-scratch budget; continuing from the checkpoint later is expected). No `restore`. Launch it with `./training/run.sh build-terrain --arena train --flat-row` first (`FLAT_ROW=1` for `train.sh`'s in-job build) and `XLA_PYTHON_CLIENT_PREALLOC=false`; `train.py` pins `ppo.num_resets_per_eval=0` for terrain runs. Deployment: the exported schema-2 contract carries `max_torque=32` and the launch clamp (`wojtek_real.urdf.xacro` + launch `max_torque`) follows the policy's contract — the usual config change. A clamp below the trained cap truncates the learned spikes and changes the gait; 32 Nm peaks draw ~30 A phase current per motor, a PSU/bus budget item for the deploy session. |

`run_v2`'s restore path is an artifact dependency, not a guaranteed portable
starting point. Its config comments note that the historical checkpoint no
longer loads after observation-size changes; confirm path and shapes before
using the preset.

## `run.sh` command modes

| Command | Usage / output |
|---|---|
| `build` | `./training/run.sh build [--total-mass M --kp K --kd D]`; regenerates the training model and home keyframe. Defaults: `16` kg, `20`, `1`. |
| `pose` | `./training/run.sh pose [--second RAD --third RAD --kp K]`; renders linkage pose samples; `--kp` defaults to `20`. |
| `check` | `./training/run.sh check [--gpu --backend {jax,warp,auto} --nenv N --nsteps N]`; static checks plus a small MJX/JAX fallback compile by default. Use `--gpu --backend warp` to exercise the primary MJWarp CUDA path. GPU CLI defaults: backend `jax`, `nenv=4096`, `nsteps=200`. |
| `train` | `./training/run.sh train [Hydra overrides]`; writes checkpoints and `run.json` to `training/runs/<run_name>`. |
| `smoke` | `./training/run.sh smoke [Hydra overrides]`; tiny CPU pipeline check with WandB disabled. A selected preset can override smoke's short PPO budget, so use `++ppo.num_timesteps=100000` when combining a preset with smoke. |
| `eval` | `./training/run.sh eval --run runs/<name> [--x-vel V --y-vel V --yaw-vel W --height H --steps N --out FILE --video-size WxH --overlay-torque --overlay-camera]`; renders a rollout. See "Rendering videos" below for the three render switches. |
| `video-probe` | `./training/run.sh video-probe --run runs/<name> [--arena {flat,train,eval} --cell NAME --seconds S --fps F --vx V --camera NAME --out FILE --list-cells]`; renders the showcase clip: 960x720 with every overlay on. `--cell` films one measurement cell of the eval arena from the spawn `terrain-scan` uses; `--arena train`/`eval` needs a terrain run. Output lands in `training/videos/<run_name>/` unless `--out` says otherwise. |
| `battery` | `./training/run.sh battery --run runs/<name> [--out FILE --alpha A --lag-tau T]`; writes the fixed comparison battery. `--alpha`/`--lag-tau` are eval-only plant perturbations, see "Robustness grid (eval-only)" below. |
| `courses` | `./training/run.sh courses --run runs/<name> [--seeds N --only NAME... --video --video-size WxH --overlay-torque --overlay-camera --paths --out FILE --list]`; writes the path-following course benchmark. `--list` prints the catalogue without loading a run. See "Course benchmark" below. |
| `imu-grid` | `./training/run.sh imu-grid --runs runs/<name>... [--bias-levels B... --axes x y z --noise-gyro S... --vib-gain G... --latency-substeps D... --lag-tau T --seeds N --stand-sec S --walk-sec S --walk-vx V --out FILE]`; the gyro bias/noise robustness grid, with the action-correlated gyro-vib loop, pinned control latency and actuator-torque lag. See "IMU robustness grid" below. |
| `report` | `./training/run.sh report --run runs/<name> [--out-json FILE --out-md FILE]`; writes battery, torque, power, impact proxy, and termination summary. |
| `export` | `./training/run.sh export --run runs/<name> [--out DIR]`; writes `policy.npz` plus `policy_meta.json`, the schema-2 deployment contract built from the run's env (`wojtek_rl/deploy_contract.py`), and validates the deploy runtime end-to-end against the env before writing. |
| `app` | `./training/run.sh app [--host HOST --port PORT]`; runs the interactive navigation demo. `WOJTEK_RUN_DIR`, `HOST`, and `PORT` environment variables supply defaults; see [demo README](../demo/README.md). |
| `test` | `./training/run.sh test [pytest args]`; runs `training/tests/unit` — model-free, ~3 s, safe in an edit loop. |
| `test-slow` | `./training/run.sh test-slow [pytest args]`; runs `training/tests/integration` — builds and steps real MJX models, 6m23s cold. Sets `JAX_COMPILATION_CACHE_DIR=training/.jax_cache` so repeat runs reuse compiled executables (measured on one file: 45s cold, 16s warm; the whole suite's warm time was not measured). |
| `test-all` | `./training/run.sh test-all [pytest args]`; both suites, with the same compilation cache. |

The live dashboard is a direct Python module, not a `run.sh` mode. Supply a
log path explicitly because its source default is developer-local:

```bash
(cd training && ./.venv/bin/python -m wojtek_rl.dashboard \
  --log /path/to/training.log --port 8765 --budget 200000000)
```

For a completed locomotion policy, run the report before treating it as a
candidate for deployment:

```bash
./training/run.sh report --run runs/my_locomotion
./training/run.sh eval --run runs/my_locomotion --x-vel 0.3 --height 0.125
./training/run.sh export --run runs/my_locomotion --out runs/my_locomotion/deploy
```

For a keeper, upload the exported pair to its Hugging Face model repo next
to the checkpoint (`hf upload <HF_ORGANIZATION>/<keeper> <deploy-dir> . --include
"policy*"`). The ROS stack loads policies by reference -- an HF repo id or a
local directory -- via the `policy` launch argument (see
`ros/src/wojtek_policy/wojtek_policy/policy_source.py`); keepers exported
before the schema-2 contract are regenerated with
`wojtek_rl/migrate_keeper_meta.py`.

## Rendering videos

Every tool that writes an mp4 of a rollout goes through `wojtek_rl/video/`:
`writer.py` (the one mediapy writer, falling back to the bundled ffmpeg when
the system has none), `render.py` (`SceneView`, the offscreen chase-camera
renderer), `overlays.py` (the in-frame panels, pure functions over the values
they draw) and `probe.py` (the `video-probe` CLI). `pose` and the scan bench
render their own interactive and benchmark output and do not use it.

`eval` and `courses --video` render 640x480 with no overlays, so their clips
stay comparable with the ones earlier reports were built from. Both take the
same three switches:

| Flag | Default | Effect |
|---|---|---|
| `--video-size WxH` | `640x480` | Rendered frame size. Panel geometry scales with it, and a frame larger than the scene's offscreen buffer recompiles the scene to fit. `eval`'s stacked `--plots` panels follow the frame width. |
| `--overlay-torque` | off | One signed bar per actuator against the env's effective torque limit, grouped by leg, along the bottom of the frame. Not the `--plots torques` strip, which is a full-episode matplotlib trace stacked under the video. |
| `--overlay-camera` | off | The onboard depth view as a top-right inset, with the height-scan points projected onto it and colored by `height_scan.visible_mask`. |

`--overlay-camera` needs a run whose task config carries a `height_scan`
block: the depth camera is placed at that block's `mask` mount, pitch and
vertical field of view, so the render and the analytic mask share a frustum.
Without it there is no geometry to place the camera at and the flag is
refused.

`video-probe` is the showcase preset of the same stack: 960x720, torque bars
and depth inset always on, plus the held 5x5 scan as a heatmap when the actor
reads `height_scan`. It is a rendering tool, not a measurement: it reads no
training config and writes none.

## Course benchmark (path following)

The `wojtek_rl/courses/` package answers a different question from
`battery.py`. The
battery drives the policy with *open-loop* velocity commands and measures gait
quality. The course benchmark closes the loop: a frozen pure-pursuit follower
turns the base pose into the `[vx, vy, wz, height]` command, and each scenario
scores how faithfully the robot walked a geometric path.

```bash
./training/run.sh courses --list                       # the catalogue, no run needed
./training/run.sh courses --run runs/my_locomotion     # 20 scenarios x 8 seeds
./training/run.sh courses --run runs/my_locomotion \
  --only circle_r075 u_turn --seeds 4 --paths          # iterate on two rows
./training/run.sh courses --run runs/my_locomotion --video --paths
```

Twenty scenarios in five families, each varying exactly one thing off the
nominal (flat floor, model friction, 0.5 m/s, no disturbance) so a bad row has
a single interpretation: eight path geometries (`straight_10m`,
`arc_r3_90deg`, `circle_r2`, `circle_r075`, `figure_eight_r15`, `square_3m`,
`slalom_05m`, `u_turn`), four speed rows (`straight_slow`, `straight_fast`,
`circle_r2_fast`, `speed_steps_straight`), two friction rows
(`straight_slippery`, `circle_r1_slippery` at `mu = 0.4`), two impulse rows
(`straight_push`, `straight_push_fast`), and four rotate-in-place rows
(`spin_left`/`spin_right` isolating chirality at 0.8 rad/s — a policy can be
asymmetric; the stiff_b keeper shipped unable to spin right because nothing
tested it — and `spin_slow`/`spin_fast` isolating rate at 0.4/1.2 rad/s CCW).

Each scenario's score is the **weakest** of five sub-scores, every one a
measured error divided into a *physical* reference -- so there is no
calibration file, no hand-tuned cutoff, and nothing to re-baseline:

| sub-score | formula | 1.0 means |
|---|---|---|
| `tracking` | `0.174 / RMS cross-track error` | off the path by a full stance half-width |
| `speed` | `mean commanded speed / RMS along-path speed error` | speed error as large as the command |
| `height` | `0.129 / RMS height error` | height error equal to the whole standing height |
| `grip` | `base distance / foot-slip distance` | feet slide as far as the body moved |
| `smoothness` | `1 / vibration_index` | all joint-velocity power above 5 Hz |

The spin rows have no path and no forward travel, so tracking/speed/grip are
replaced by two rotate-specific sub-scores with the same construction:
`rotation` = `|commanded wz| / RMS yaw-rate error` and `drift` =
`0.174 / max planar drift from the start`; height and smoothness are shared.
Their completion gate is one full rotation, accumulated in the commanded
direction, within the time budget.

Higher is always better and there is no ceiling (`SUBSCORE_CAP` clips only so
the JSON stays finite). `min` rather than a weighted average, so one bad axis
cannot be diluted by four good ones -- the binding sub-score is reported next
to every score. Completion is a **gate**, not a sub-score: a fall or an
unfinished course scores 0 and the rollout is abandoned there, because any
sub-score capped at 1.0 would cap the whole `min`. Raw metres-and-m/s metrics
are kept alongside the sub-scores so a zero is still diagnosable.

Each scenario runs `--seeds` rollouts (default 8) and reports median and worst,
so a policy that only sometimes falls cannot pass by luck. Results go to
`<run>/courses.json`; `--video` writes one mp4 per scenario (seed 0) and
`--paths` one overhead commanded-vs-actual PNG per scenario, both into
`<run>/courses/`. The clips take the render switches from "Rendering videos"
above; with none of them they are the plain 640x480 chase-camera view. A
machine with no usable GL drops the video with a warning and keeps the
numbers.

Cost: a single-env Python rollout loop, so measured ~30 s per 2600-step course
per seed on CPU — roughly half an hour for the full 20 x 8 matrix, up to an
hour if most scenarios time out rather than finish. Use `--only NAME...
--seeds 1` while iterating.

The follower's constants (`LOOKAHEAD_M`, `YAW_MAX`, `SPIN_ENTER_RAD`/
`SPIN_EXIT_RAD`, `GOAL_RADIUS_M`, `GOAL_MIN_PROGRESS_M`) are deliberately
**not** `navigation.py`'s `NavConfig`:
tuning the demo's go-to-point gains must never move benchmark numbers. It is
non-holonomic on purpose (`vy` is always 0) -- letting it strafe would let the
robot crab through the turning courses without ever testing turning. Changing
any of these invalidates every score previously recorded;
`tests/unit/test_courses.py` asserts them so the change is loud.

The package is modular so scenarios stay cheap to add: waypoint builders in
`courses/geometry.py`, the `Course` dataclass and shared nominals in
`courses/spec.py`, the frozen follower in `courses/follower.py`, rollout and
scoring in `courses/rollout.py` / `courses/scoring.py`, and the catalogue in
`courses/families/` — one module per family, one entry per scenario (`Course` for paths, `SpinCourse` for rotate-in-place).
Adding a scenario is appending a `Course` to the right family's `COURSES`
list (or adding a new family module and listing it in
`families/__init__.py`'s `FAMILY_MODULES`).

## Robustness grid (eval-only)

`wojtek_rl/battery.py` accepts three eval-only plant perturbations, applied
to the run's already-built, already-customized model -- none of them is a
training-path or `env.py` change:

| Flag | Default | Effect |
|---|---|---|
| `--alpha FLOAT` | `1.0` (no-op) | Kt (torque-constant) miscalibration: scales the model's effective `actuator_gainprm[:,0]`/`actuator_biasprm[:,1:3]` (kp/kd) and `actuator_forcerange` (the torque cap) by `alpha`, in place, via `apply_kt_miscalibration`. Works even when the run's config has `pd_kp=0.0` (XML defaults), since it reads whatever the model's effective values are post-`_customize_model`, not the config. |
| `--lag-tau SECONDS` | `0.0` (native pipeline) | Actuator-bandwidth first-order lag on the JOINT TORQUE (not the setpoint): a value `>0` switches the battery rollout to an explicit-PD substep loop (`make_lagged_rollout_fns`) that computes `kp*(ctrl-qpos)-kd*qvel`, clips to the effective torque cap, then applies `tau_applied += (1-exp(-dt_sub/lag_tau))*(tau_pd-tau_applied)` every physics substep, so the feedback path itself lags -- the mechanism that destabilizes a high-kp policy in practice. The filter state persists across control steps in `info["tau_applied"]`, zeroed at reset. |
| `--torque-envelope "OMEGA_B,OMEGA_0"` | none (flat cap, unchanged) | Speed-dependent DRIVING-torque cap (`apply_torque_envelope`): real motors lose available driving torque as joint speed rises (back-EMF eats bus voltage), which the flat `actuator_forcerange` cap does not model. In the DRIVING quadrant (`tau*qvel >= 0`) the allowed `\|tau\|` is the static cap for `\|qvel\| <= OMEGA_B`, ramps linearly to 0 at `\|qvel\| == OMEGA_0`, and is 0 beyond; the BRAKING quadrant (`tau*qvel < 0`, regenerative) keeps the full static cap. `cap` is the model's current `actuator_forcerange` upper bound, so the envelope composes with `--alpha` (an alpha-scaled cap raises the envelope's plateau too). Applied last in the substep loop, after the lag filter -- the physically produced torque can never exceed what the envelope allows. Passing this flag forces the explicit-PD path even at `--lag-tau 0` (the envelope needs per-substep qvel, which only that path has); the lag filter is then an exact passthrough. |
| `--out PATH` | `<run>/battery.json` | Where the perturbed battery result is written, so a grid cell never clobbers the run's canonical `battery.json`/`eval_report.json`. |

Every cell -- including `--alpha 1.0 --lag-tau 0`, no `--torque-envelope` --
runs the same code path, so results are directly comparable. A tiny nonzero
`--lag-tau` (e.g. `1e-9`) reproduces the native (unperturbed) pipeline's
battery numbers to within a few percent (chaotic contact dynamics amplify
float32 rounding across a ~700-step rollout; see `tests/test_robustness_grid.py`
for the short-rollout tolerance and its rationale) -- that equivalence is the
gate for trusting the explicit-PD substep loop at all.

```bash
# One perturbed cell against an existing run, without touching its battery.json
JAX_PLATFORMS=cpu ./.venv/bin/python -m wojtek_rl.battery \
  --run runs/wojtek_stiff_b_20260717_235321 \
  --alpha 1.58 --lag-tau 0.005 --torque-envelope "15,28" \
  --out runs/wojtek_stiff_b_20260717_235321/grid/battery_a1.58_lag5ms_env15-28.json
```

[`training/jobs/stiff_grid.sh`](../jobs/stiff_grid.sh) sweeps this over a
set of checkpoints and an alpha/lag/envelope grid (CPU-mode eval for
determinism and parity with the battery), writing
`runs/<run>/grid/battery_a<alpha>_lag<ms>ms_env<tag>.json` per cell -- `<tag>`
is `none` (no `--torque-envelope` passed for that cell, preserving the native
path) or `<OMEGA_B>-<OMEGA_0>`. A crashed cell logs a WARN and the sweep
continues, since a fallen-over policy under a harsh perturbation is a data
point, not a job failure. `wojtek_rl/grid_report.py` then aggregates every
listed run's cells into one markdown table (row = run x alpha x envelope,
columns = lags; cell = mean `track_err_rms` over the 4 battery scenarios,
gated PASS/FAIL against the stiffness ladder's gates 1-4 -- see
`training/jobs/stiff_ladder.sh`'s `run_gates` -- keeper reference
`wojtek_stiff_b_20260717_235321`), ending with the stiffest run
that stays PASS across every lag and envelope, per alpha-world.
Filenames without the `_env<tag>` segment (grid runs predating this axis)
are still read, treated as envelope `none`.

## IMU robustness grid (eval-only)

`wojtek_rl/imu_grid.py` is the sensor-side counterpart of the plant grid
above, built after the terrain v4.1 policy oscillated while standing on the
real robot (a ~25 Hz limit cycle -- half the 50 Hz control rate -- driven by
gyro noise and loop latency) and nothing in the battery or the courses
perturbed the IMU to detect it. It sweeps a **pinned gyro bias** (a fixed
per-cell vector written into `info["gyro_bias"]` every step, the same key
the env's per-episode bias DR draws at reset; the actor's gyro carries it,
the critic and the physics never see it) and, optionally, absolute white
gyro-noise scales (each rebuilds the measurement env via `env_overrides`).
Because the key exists in every env (zeros when the run never trained bias
DR), the grid runs against any checkpoint of this project unchanged --
that is the point: measure a policy trained *without* bias DR under the
bias it will meet on hardware.

The `--vib-gain` axis closes the loop that made the robot oscillate. On
the machine, the policy's action jitter shakes the frame, and the IMU
sits on that frame, so the policy sees its own jitter and reacts to it.
The sim carries none of that, which is why white noise and bias probes
leave the policy standing still. `task.env.obs_noise.gyro_vib` (default
0, off for every preset) models the loop as a sensor effect: each
control step, the change in joint torque drives a resonator tuned to
half the control rate, and the resonator's state is added to the gyro
reading the actor sees (`base.gyro_vib_step`, mixing `base.VIB_MIX`).
The critic and the physics see nothing. Because it lives in the env
config, a future run can also train against it. The grid sweeps the
gain to find where a policy's stand goes unstable; that critical gain
is the policy's stability margin, and the margin is what to compare
across policies.

Two more loop-delay axes complete the picture. `--latency-substeps` pins
the per-episode control-latency draw (`info["ctrl_delay"]`, same pin
mechanism) to chosen substep counts -- the random draw only lands a seed
on the worst case ~1/6 of the time, so an unpinned grid mostly measures
mild latency. `--lag-tau` swaps in the battery's explicit-PD substep loop
(`make_lagged_rollout_fns`): a first-order lag on the joint torque, the
plant-bandwidth mechanism the env's ideal actuators cannot show.

Per cell it scores a 10 s stand and a 10 s straight walk over a few seeds:
`vibration` (the battery's >5 Hz joint-velocity power index), `band_20_25`
(spectral power fraction in the 20-25 Hz near-Nyquist band where the
real-robot limit cycle lived), `qvel_rms` (absolute joint-velocity scale
-- the spectral scores are power FRACTIONS, so a near-motionless stand
can score high on microscopic buzz; a high ratio only means a real
oscillation when qvel_rms says the joints actually move), falls, and the
walk's `vx_err_rms`. No gates -- compare cells against the
always-included bias=0 baseline row and across runs. Results land in
`runs/<run>/imu_grid/imu_grid.json`, plus one combined markdown table via
`--out`.

Measured on `wojtek_terrain_blind_v4_1` (2026-08-11): the EXOGENOUS axes
do not reproduce the real robot's 25 Hz standing limit cycle -- stand
`qvel_rms` stays at 0.013 rad/s (motionless; the ~0.69 vibration ratio
there is quiet-signal noise, which is what `qvel_rms` is in the table to
catch) across bias to 0.2 rad/s, white noise to 4x trained, pinned
full-period latency, and 10 ms torque lag, with zero falls. The
CORRELATED axis does reproduce it: sweeping `--vib-gain` over
{0.03, 0.1, 0.3, 1.0}, the stand snaps from motionless (`qvel_rms`
0.012-0.014 through gain 0.1) to a violent Nyquist-band oscillation at
gain 0.3 -- `qvel_rms` 3.04 rad/s with 68% of power in 20-25 Hz, 96% at
gain 1.0 -- the hardware failure's exact signature. The critical gain
between 0.1 and 0.3 is the policy's stability margin against the
action->vibration->IMU loop; compare that margin across policies (e.g.
`action_filter` on vs off) rather than reading any single cell as
pass/fail.

```bash
# Does the un-filtered terrain policy show the standing limit cycle in sim?
./training/run.sh imu-grid --runs runs/wojtek_terrain_blind_v4_1 \
  runs/wojtek_terrain_blind_v5 --out runs/imu_grid_report.md

# Probe the stability margin: crank white gyro noise past the trained value
./training/run.sh imu-grid --runs runs/<name> --noise-gyro 0.4 0.8
```

The course benchmark can run under the same pinned bias:
`./training/run.sh courses --run runs/<name> --gyro-bias 0,0.1,0` writes
`courses_bias.json` (never the frozen `courses.json` -- biased scores are a
different measurement and must not overwrite the benchmark's). The follower
constants stay frozen; the bias pin changes only what the actor's gyro
reads. [`training/jobs/imu_grid.sh`](../jobs/imu_grid.sh) is the payload
form: `CKPTS_LIST` (required), `BIAS_LEVELS`, `AXES`, `NOISE_GYRO`,
`SEEDS`, `STAND_SEC`/`WALK_SEC`/`WALK_VX`, `IMU_GRID_OUT`.

## Job payload configuration

Every script in `training/jobs/` is a parameterized, reusable tool. Experiment-
specific values enter through environment variables declared with a `param`
line, and the as-run instance of a finished experiment is archived in its
keeper's Hugging Face repo under `hpc/` — see
[`training/CLAUDE.md`](../CLAUDE.md) for the convention.

A payload's own declaration is the authoritative list of its variables,
their kinds and their defaults. Print it with `--describe`:

```bash
bash training/jobs/train.sh --describe
```

[`training/jobs/train.sh`](../jobs/train.sh) takes `EXPERIMENT` (the name passed
as `+experiment`), `RUN_NAME`, `NUM_ENVS` (`++ppo.num_envs`, a whole-machine
total), `BATCH` (`++ppo.batch_size`), the terrain and wandb switches, and
`EXTRA` for additional space-separated Hydra overrides.

Use the same resolve/smoke/probe/full-run ladder on an appropriate local or
single-GPU environment before spending a large multi-GPU budget.

## Process environment defaults

These variables are not Hydra configuration, but they materially affect the
way a command runs:

| Scope | Default behavior |
|---|---|
| `train` | Sets `XLA_FLAGS=--xla_gpu_triton_gemm_any=true` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` only when those variables are not already set. |
| `smoke`, `battery`, `courses`, `report`, `export` | `run.sh` sets `JAX_PLATFORMS=cpu`; `sim.backend=auto` therefore resolves to the MJX/JAX backend. |
| `eval`, `app` | `run.sh` defaults `MUJOCO_GL=egl` on Linux only; macOS leaves it unset so mujoco picks its native GL. Set it explicitly to override. |
| `app` | Defaults `WOJTEK_RUN_DIR=policies/fbb_v3`, `HOST=127.0.0.1`, and `PORT=8010`. The demo is a joystick-specific presentation layer, not a generic evaluator for arbitrary task/configuration shapes. |
