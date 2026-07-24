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
`eval`, `battery`, and `report` rebuild the saved environment configuration
and resolve `auto` again. Pin `+task.env.sim.backend=warp` or `jax` when a
specific backend matters. `smoke`, `battery`, `report`, and `export` force CPU
execution, so `auto` resolves to JAX for those commands.

## Naming

Use **Wojtek** in new prose and new `run_name` values. The repository lives
at `github.com/machinekind/wojtek` (older checkouts may still sit in
directories named `wojtek`; cluster paths are parameterized, see the HPC
section). `fbb-locomotion`, `fbb_*` preset run names, and `policies/fbb_v3`
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
| `task.env.sim.naconmax_per_env` | `32` | Warp contact budget per environment. See [the MJWarp report](../../docs/plans/mjwarp-phase0-report.md). |
| `task.env.sim.njmax` | `320` | Warp constraint-row budget per world. |
| `task.env.sim.num_envs` | `1` | Do not set for training: `train.py` replaces it with `max(ppo.num_envs, ppo.num_eval_envs)`. Set `++ppo.num_envs` instead. |
| `task.env.episode_length` | task-specific | Explicitly override only when PPO's `episode_length` is also understood; absent PPO override, the trainer mirrors this value. |
| `task.env.action_scale` | `0.5` rad | Scales policy action to motor-target displacement. `joystick` additionally accepts a 3-vector `[abduction, hip, knee]` tiled over the 4 legs, giving joints different command authority (e.g. `'task.env.action_scale=[0.25,0.5,0.5]'`); `getup` and `jump` take the scalar only. |
| `task.env.obs_noise.gyro` | `0.2` | Uniform actor gyro noise scale. |
| `task.env.obs_noise.gravity` | `0.05` | Uniform actor gravity-vector noise scale. |
| `task.env.obs_noise.joint_pos` | `0.01` rad | Uniform actor joint-position noise scale. |
| `task.env.obs_noise.joint_vel` | `1.5` rad/s | Uniform actor joint-velocity noise scale. |
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
| `task.env.terrain.enable` | `false` | Train on the generated terrain arena with a spawn curriculum; see [Terrain curriculum](#terrain-curriculum). Requires `build-terrain` first. Flat path is byte-identical when off. |
| `task.env.terrain.feet_only` | `true` | Collide only the four foot spheres (and the base box) with terrain, dropping the leg-link collision geoms; terrain contact sizing. |
| `task.env.terrain.spawn_yaw` | `true` | Random base yaw at every spawn/respawn. Free, because the actor obs is yaw-invariant. |
| `task.env.terrain.pad_jitter` | `0.15` m | xy jitter half-width at the spawn pad centre. Keep `<= 0.2` so the settle transient stays on the 0.6 m pad. |
| `task.env.terrain.init_level_frac` | `0.5` | Initial difficulty level drawn uniformly over the lower this-fraction of rows (start easy). |
| `task.env.terrain.demote_fraction` | `0.5` | Demote a level when an episode walked less than this fraction of its commanded planar distance. |
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
| `torque_limit` | `0.0` |  |  |

The zero-default terms are dormant until a preset or override enables them:
`torque_rate` penalizes step-to-step change in actuator torque (bang-bang
motor commands the policy-side `action_rate` cannot see), `high_step`
rewards swing-foot clearance up to `gait.swing_height` while moving,
`stand_feet_down` penalizes total foot clearance at zero command, and
`torque_limit` is the saturation hinge described above.

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

Terrain training exists only on `joystick`, off by default, and does not depend
on `domain_rand`. When `task.env.terrain.enable=true` the env loads the shared
terrain arena instead of the flat scene, spawns each parallel env on a
curriculum tile, measures base height, foot clearance, and foot contact against
the local terrain surface (so stepping up is not punished as a height error and
a box top does not read as a fall), and trains under a terrain-aware auto-reset
wrapper that promotes and demotes tiles between episodes.

**Prerequisite: build the arena first.** The scene, heightfield, spec, and
lookup grid are generated on demand and gitignored:

```bash
./training/run.sh build-terrain            # default 10-row arena, seed 0
./training/run.sh train task=joystick ++task.env.terrain.enable=true
```

`terrain.enable=true` with the assets missing fails at construction with a
message pointing back at `build-terrain`.

Keys (all under `task.env.terrain`):

| Key | Default | Meaning |
|---|---:|---|
| `enable` | `false` | Load the terrain scene and the spawn curriculum. |
| `feet_only` | `true` | Collide only the four foot spheres (plus the base box) with terrain; the leg-link collision geoms are dropped on the loaded model (never in the XML), following mujoco_playground's rough-terrain contact sizing. |
| `spawn_yaw` | `true` | Draw a random base yaw at each spawn and respawn. The actor obs carries no heading (gravity is yaw-invariant), so this adds variety at no observation cost. |
| `pad_jitter` | `0.15` m | xy jitter half-width around the spawn pad centre. Keep `<= 0.2` so the reset settle transient never leaves the 0.6 m flat pad. |
| `init_level_frac` | `0.5` | Each env's initial level is uniform over the lower this-fraction of rows (legged_gym starts easy). |
| `demote_fraction` | `0.5` | The legged_gym demote threshold (see below). |

**Curriculum rule (legged_gym), applied at every episode end — fall or
timeout.** Per env: `walked = ‖last_xy − spawn_xy‖`.

- **Promote** one level when `walked > tile_size / 2` (the robot crossed its
  tile).
- **Demote** one level when `walked < demote_fraction × commanded_distance`,
  where `commanded_distance` accumulates `‖command_xy‖ × ctrl_dt` over the
  episode. A standing episode (near-zero commanded distance) clears neither
  threshold, so it is neutral.
- Levels clip to `[0, n_rows − 1]`. An env **promoting from the top level
  respawns on a uniformly random row** instead of clipping, so easy terrain is
  not forgotten.

The env's terrain **type is fixed for the whole run** (one of the eight arena
types, drawn at first reset); only the difficulty level moves. Respawn is a
teleport onto the new tile's pad: this is valid precisely because the actor
observation contains no absolute position or heading. **The observation layout
is unchanged (still 54-dim actor / 61-dim critic), so a terrain run's
checkpoint stays layout-compatible** — terrain awareness here is in the
rewards, terminations, and spawn logic, not the obs (scandots are step 6).

Watching the curriculum: the eval metric `eval/episode_terrain_level_per_step`
(printed as `terrain_lvl`) is the **eval env's** level. That env is a separate
instance reset to the initial spawn distribution at every evaluation, so it
reflects the eval spawn spread — it hovers near the init mean and does **not**
show the training curriculum climbing. To watch the actual training curriculum,
run with `++ppo.log_training_metrics=true` (optionally
`++ppo.training_metrics_steps=<env-steps>`): brax's `EpisodeMetricsLogger` then
reads the training env's per-episode metrics, divides the `_per_step` value by
episode length, and reports `episode/terrain_level_per_step` through the same
progress/WandB path (printed as `terrain_lvl_train`). The step-5 blind-terrain
preset should set `ppo.log_training_metrics=true` so the mean training level is
recorded for the run.

**Warp contact budget.** The flat `sim.naconmax_per_env=32` is undersized for
feet-on-terrain contacts, and warp drops the overflow silently. A warp terrain
run must set an explicit larger budget — start at roughly 2x the flat default
and measure the real number with `check-terrain` on the GPU (step 5):

```bash
# CPU sanity check that the terrain scene still steps (jax backend).
./training/run.sh check-terrain --backend jax --num-envs 4 --steps 50
# Warp run: raise the contact budget (measure, do not guess).
./training/run.sh train task=joystick ++task.env.terrain.enable=true \
  ++task.env.sim.naconmax_per_env=64
```

Constructing a terrain env on the warp backend with the untouched flat default
prints a loud warning to that effect.

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
| `ppo.num_resets_per_eval` | `10` |
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
| `stiff_phase_a` | `wojtek_stiff_a` | Inherits `springy_phase_a`. Phase A of the kp40 probe winner's (job NNNNNNN) full two-phase run: `task.env.{pd_kp,pd_kd,max_torque}={40.0,1.4,9.0}` (kd matches the probe for curriculum continuity; `max_torque` widened to the XML's `±9` forcerange as in the probes); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=1200000000`. |
| `stiff_phase_b` | `wojtek_stiff_b` | Inherits `springy_phase_b`. Phase B of the kp40 probe winner: `task.env.{pd_kp,pd_kd,max_torque}={40.0,1.6,9.0}` (`pd_kd` bumped from the probe's `1.4` for damping margin — its >5 Hz vibration metric exceeded the kp20 baseline on 3 of 4 scenarios — putting the damping ratio at ~1.13x the kp20/kd=1.0 reference, inside the ±20% kd DR band phase A already trained under; `max_torque` overrides phase B's `6.0` back to `9.0`); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=800000000`. Launched with `restore=<stiff_phase_a checkpoint>`; the preset itself does not set `restore`. **Deployment must match phase B** (`pd_kp=40`/`pd_kd=1.6`/`max_torque=9.0` in `wojtek_real.urdf.xacro`) if this becomes a keeper. |
| `locomotion_stiff_v1` | `wojtek_loco_stiff_v1` | Frozen keeper, self-contained (no `springy_phase_*`/`stiff_phase_*` inheritance): reproduces `stiff_phase_b`'s effective config verbatim — `task.env.{action_scale,pd_kp,pd_kd,max_torque,abduction_ctrl_limit,knee_target_max}={[0.25,0.5,0.5],40.0,1.6,9.0,0.44,3.15}`; `task.env.{latency,encoder}.enable=true`; `task.env.obs.include=[joint_pos,joint_vel,last_act,command]`; command/push/gait/reward blocks identical to `springy_phase_b`; `dr.motor_strength.enable=true` (`range=[0.5,1.1]`) and `dr.joint_gains.enable=true` (`gain_pct=0.2`, `kd_pct=0.2`); `ppo.num_timesteps=800000000`. FROZEN from the trained run `wojtek_stiff_b_20260717_235321` (job NNNNNNN, commit `eb8bad2`) so it stays reproducible while `stiff_phase_*` evolves. Deployment must match: `pd_kp=40`/`pd_kd=1.6`/`max_torque=9.0` on the robot. |
| `stiff_ladder_kp50` | `wojtek_stiff_kp50` | Inherits `springy_phase_b`. Ladder rung 1 off the kp40/kd1.6 keeper (`wojtek_stiff_b_20260717_235321`, job NNNNNNN): `task.env.{pd_kp,pd_kd,max_torque}={50.0,1.79,9.0}` (`pd_kd` holds the keeper's damping ratio: `kd=1.6*sqrt(kp/40)`); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=400000000`. Launched with `restore=<keeper checkpoint>`; the preset itself does not set `restore`. |
| `stiff_ladder_kp60` | `wojtek_stiff_kp60` | Inherits `springy_phase_b`. Ladder rung 2: `task.env.{pd_kp,pd_kd,max_torque}={60.0,1.96,9.0}` (`pd_kd=1.6*sqrt(kp/40)`); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=400000000`. Launched with `restore=<stiff_ladder_kp50 accepted checkpoint>`; the preset itself does not set `restore`. |
| `stiff_ladder_kp70` | `wojtek_stiff_kp70` | Inherits `springy_phase_b`. Ladder rung 3: `task.env.{pd_kp,pd_kd,max_torque}={70.0,2.12,9.0}` (`pd_kd=1.6*sqrt(kp/40)`); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=400000000`. Launched with `restore=<stiff_ladder_kp60 accepted checkpoint>`; the preset itself does not set `restore`. |
| `stiff_ladder_kp80` | `wojtek_stiff_kp80` | Inherits `springy_phase_b`. Ladder rung 4: `task.env.{pd_kp,pd_kd,max_torque}={80.0,2.26,9.0}` (`pd_kd=1.6*sqrt(kp/40)`); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=400000000`. Launched with `restore=<stiff_ladder_kp70 accepted checkpoint>`; the preset itself does not set `restore`. |
| `stiff_ladder_kp90` | `wojtek_stiff_kp90` | Inherits `springy_phase_b`. Ladder rung 5, an extension rung beyond the kp50-80 ladder: `task.env.{pd_kp,pd_kd,max_torque}={90.0,2.40,9.0}` (`pd_kd=1.6*sqrt(kp/40)`); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=400000000`. Launched with `restore=<kp50-80 ladder's winner, stiff_ladder_kp80's accepted checkpoint, wojtek_stiff_kp80_20260718_100225>`; the preset itself does not set `restore`. |
| `stiff_ladder_kp100` | `wojtek_stiff_kp100` | Inherits `springy_phase_b`. Ladder rung 6: `task.env.{pd_kp,pd_kd,max_torque}={100.0,2.53,9.0}` (`pd_kd=1.6*sqrt(kp/40)`); enables `dr.joint_gains` (`gain_pct=0.2`, `kd_pct=0.2`, per-joint); `ppo.num_timesteps=400000000`. Launched with `restore=<stiff_ladder_kp90 accepted checkpoint>`; the preset itself does not set `restore`. |
| `locomotion_stiff_kp80_v1` | `wojtek_loco_stiff_kp80_v1` | Frozen keeper, self-contained (no `springy_phase_*`/`stiff_ladder_*` inheritance): reproduces `wojtek_stiff_kp80c_20260719_105503`'s effective config verbatim — `task.env.{action_scale,pd_kp,pd_kd,max_torque,abduction_ctrl_limit,knee_target_max}={[0.25,0.5,0.5],80.0,2.26,9.0,0.44,3.15}`; `task.env.{latency,encoder}.enable=true`; `task.env.obs.include=[joint_pos,joint_vel,last_act,command]`; command/push/gait/reward blocks identical to `locomotion_stiff_v1`/`springy_phase_b`; `dr.motor_strength.enable=true` (`range=[0.5,1.1]`) and `dr.joint_gains.enable=true` (`gain_pct=0.2`, `kd_pct=0.2`); `ppo.num_timesteps=800000000` (the consolidation budget only — lineage is the `locomotion_stiff_v1` keeper (2.0B steps) → ladder rungs kp50→60→70→80 (+400M each) → this +800M consolidation, 4.4B cumulative). FROZEN from the trained run `wojtek_stiff_kp80c_20260719_105503` (job NNNNNNN, seed 1), consolidation restored from `wojtek_stiff_kp80_20260718_100225/checkpoints/000412876800`. Battery mean `track_err_rms` 0.05788 (pre-consolidation 0.06305, −8.2%); zero falls in 4/4 scenarios; max torque saturation 3.2%. Robustness grid gates 1-4 all PASS: `alpha=1.58` (uncompensated Kt miscalibration) → mean 0.0527; torque envelope `5,12` → mean 0.0579; actuator lag ≤10 ms harmless (proven on the pre-consolidation rung). Role: the safe-without-compensation keeper. Deployment must match: `pd_kp=80`/`pd_kd=2.26`/`max_torque=9.0` on the robot. |
| `locomotion_stiff_kp90_v1` | `wojtek_loco_stiff_kp90_v1` | Frozen keeper, self-contained (no `springy_phase_*`/`stiff_ladder_*` inheritance): reproduces `wojtek_stiff_kp90c_20260719_105503`'s effective config verbatim — `task.env.{action_scale,pd_kp,pd_kd,max_torque,abduction_ctrl_limit,knee_target_max}={[0.25,0.5,0.5],90.0,2.40,9.0,0.44,3.15}`; `task.env.{latency,encoder}.enable=true`; `task.env.obs.include=[joint_pos,joint_vel,last_act,command]`; command/push/gait/reward blocks identical to `locomotion_stiff_v1`/`springy_phase_b`; `dr.motor_strength.enable=true` (`range=[0.5,1.1]`) and `dr.joint_gains.enable=true` (`gain_pct=0.2`, `kd_pct=0.2`); `ppo.num_timesteps=800000000` (the consolidation budget only — lineage is the `locomotion_stiff_v1` keeper (2.0B steps) → ladder rungs kp50→60→70→80→90 (+400M each; kp90 restored from the kp80 ladder winner, `wojtek_stiff_kp80_20260718_100225`) → this +800M consolidation, 4.8B cumulative). FROZEN from the trained run `wojtek_stiff_kp90c_20260719_105503` (job NNNNNNN, seed 1), consolidation restored from `wojtek_stiff_kp90_20260718_141500/checkpoints/000412876800`. Battery mean `track_err_rms` 0.05468 (pre-consolidation 0.059125, −7.5%); zero falls in 4/4 scenarios; max torque saturation 4.3%. Robustness grid gates 1-4 all PASS: `alpha=1.58` → mean 0.0498, turn `vel_err_overall` 0.16 (consolidation fixed the pre-consolidation rung's marginal `alpha=1.58` result — turn `vel_err_overall` 0.21-0.23 across lag 0/5/10 ms, over gate 2's 0.2 threshold); torque envelope `5,12` → mean 0.0546. kp100 (one rung further) was rejected at 5.7% saturation, over the ladder's 5% gate — kp90 is the stiffest accepted operating point at the 9 Nm cap. **DEPLOYMENT CONTRACT:** deploy ONLY after measuring the real actuators' stand-sag `alpha`; command `pd_kp/alpha`, `pd_kd/alpha`, and torque cap `9/alpha` so the physical plant matches training. `wojtek_real.urdf.xacro` and the launch file's `max_torque` MUST carry the matched values before any robot use; `HEIGHT_TABLE` must be re-measured at the matched operating point. |

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
| `eval` | `./training/run.sh eval --run runs/<name> [--x-vel V --y-vel V --yaw-vel W --height H --steps N --out FILE]`; renders a rollout. |
| `battery` | `./training/run.sh battery --run runs/<name> [--out FILE --alpha A --lag-tau T]`; writes the fixed comparison battery. `--alpha`/`--lag-tau` are eval-only plant perturbations, see "Robustness grid (eval-only)" below. |
| `report` | `./training/run.sh report --run runs/<name> [--out-json FILE --out-md FILE]`; writes battery, torque, power, impact proxy, and termination summary. |
| `export` | `./training/run.sh export --run runs/<name> [--out DIR]`; writes `policy.npz` plus `policy_meta.json`, the schema-2 deployment contract built from the run's env (`wojtek_rl/deploy_contract.py`), and validates the deploy runtime end-to-end against the env before writing. |
| `app` | `./training/run.sh app [--host HOST --port PORT]`; runs the interactive navigation demo. `WOJTEK_RUN_DIR`, `HOST`, and `PORT` environment variables supply defaults; see [demo README](../demo/README.md). |
| `test` | `./training/run.sh test [pytest args]`; runs `training/tests`. |

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

[`training/hpc/stiff_grid.slurm`](../hpc/stiff_grid.slurm) sweeps this over a
set of checkpoints and an alpha/lag/envelope grid (CPU-mode eval on a
`PARTITION node — the shared venv only runs there; its single requested GPU
sits idle), writing
`runs/<run>/grid/battery_a<alpha>_lag<ms>ms_env<tag>.json` per cell -- `<tag>`
is `none` (no `--torque-envelope` passed for that cell, preserving the native
path) or `<OMEGA_B>-<OMEGA_0>`. A crashed cell logs a WARN and the sweep
continues, since a fallen-over policy under a harsh perturbation is a data
point, not a job failure. `wojtek_rl/grid_report.py` then aggregates every
listed run's cells into one markdown table (row = run x alpha x envelope,
columns = lags; cell = mean `track_err_rms` over the 4 battery scenarios,
gated PASS/FAIL against the stiffness ladder's gates 1-4 -- see
`training/hpc/stiff_ladder.slurm`'s `run_gates` -- keeper reference
`wojtek_stiff_b_20260717_235321`, job NNNNNNN), ending with the stiffest run
that stays PASS across every lag and envelope, per alpha-world.
Filenames without the `_env<tag>` segment (grid runs predating this axis)
are still read, treated as envelope `none`.

## HPC launch configuration

Every `training/hpc/*.slurm` script is a parameterized, reusable tool;
experiment-specific values enter via `sbatch --export`, and the as-run
instance of a finished experiment is archived in its keeper's Hugging Face
repo under `hpc/` — see [`training/CLAUDE.md`](../CLAUDE.md) for the
convention and the per-script requirements.

[`training/hpc/train.slurm`](../hpc/train.slurm) accepts these environment
variables through `sbatch --export` or `make hpc-train`:

| Variable | Default | Meaning |
|---|---|---|
| `WORKDIR` | submit directory | Repository checkout on the cluster; jobs are submitted from the repo root, so this rarely needs setting. |
| `STORE_DIR` | from `.env` | Persistent per-person storage for the venv, caches, and offline wandb; set it in the repo-root `.env` (template in `.env.example`) or export explicitly. |
| `EXPERIMENT` | `locomotion` | Name passed as `+experiment`. |
| `RUN_NAME` | unset | Optional explicit run name. |
| `NUM_ENVS` | `32768` | `++ppo.num_envs` across the 4-GPU default job. |
| `BATCH` | `1024` | `++ppo.batch_size`. |
| `GPUS` | `4` | Required visible GPU count check. |
| `EXTRA` | unset | Additional space-separated Hydra overrides. |
| `VENV_DIR` | sibling project venv | Optional Python environment override in `_common.sh`. |

The Make wrapper additionally accepts these transport/scheduler variables:

| Variable | Default | Meaning |
|---|---|---|
| `cluster_USER` | **required**, via `.env` | Full SSH destination, `<user>@ui.cluster.example` (template in `.env.example`). |
| `HPC_NS` | **required**, via `.env` | Your namespace directory under `$HOME` on the shared account. |
| `HPC_REPO` | `wojtek` | Checkout directory name inside the namespace. |
| `REMOTE` | `/home/<user>/$(HPC_NS)/$(HPC_REPO)` | Remote checkout destination, composed from the above. |
| `TIME` | unset | Optional Slurm time override passed by `make hpc-train`. |

Example:

```bash
make hpc-train EXPERIMENT=locomotion RUN_NAME=loco_seed1 \
  EXTRA='seed=1 dr.com_offset.enable=true task.env.encoder.enable=true'
```

Use the same resolve/smoke/probe/full-run ladder on an appropriate local or
single-GPU environment before submitting a large multi-GPU budget.

## Process environment defaults

These variables are not Hydra configuration, but they materially affect the
way a command runs:

| Scope | Default behavior |
|---|---|
| `train` | Sets `XLA_FLAGS=--xla_gpu_triton_gemm_any=true` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` only when those variables are not already set. |
| `smoke`, `battery`, `report`, `export` | `run.sh` sets `JAX_PLATFORMS=cpu`; `sim.backend=auto` therefore resolves to the MJX/JAX backend. |
| `eval`, `app` | `run.sh` defaults `MUJOCO_GL=egl` on Linux only; macOS leaves it unset so mujoco picks its native GL. Set it explicitly to override. |
| `app` | Defaults `WOJTEK_RUN_DIR=policies/fbb_v3`, `HOST=127.0.0.1`, and `PORT=8010`. The demo is a joystick-specific presentation layer, not a generic evaluator for arbitrary task/configuration shapes. |
