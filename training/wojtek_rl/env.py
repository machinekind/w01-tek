"""Joystick velocity-tracking environment for four_bar_bot.

Skeleton copied from 3_jaxpot_robotics/jaxpot_robotics/race_env.py; rewards
ported from the mujoco_playground Go1 joystick task. Actor observations use
only signals the real robot has (IMU + joint encoders + own history).
"""

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from wojtek_rl import height_scan, paths
from wojtek_rl import terrain_env
from wojtek_rl.base import ABDUCTION_ACTUATORS, KNEE_ACTUATORS, WojtekEnv

# 3 gyro + 3 gravity + 12 qpos + 12 qvel + 12 last_act + 4 cmd + 8 phase
OBS_SIZE = 54
PRIVILEGED_SIZE = 61  # OBS_SIZE + 3 linvel + 4 contacts

# Gait clock offsets per leg, paths.LEGS order (rear_left, rear_right,
# front_right, front_left): trot = diagonal pairs in antiphase; walk =
# 4-beat lateral sequence (RL, FL, RR, FR a quarter cycle apart). The
# command speed blends walk->trot offsets across gait.trot_band.
TROT_PHASE = (0.0, np.pi, 0.0, np.pi)
WALK_PHASE = (0.0, np.pi, 1.5 * np.pi, 0.5 * np.pi)

# Settled standing height vs uniform leg-extension offset, measured with a
# 2 s PD-hold settle on the current model (kp=20/kd=1): dsecond in rad on
# every second joint, dthird = 2*dsecond on every third joint.
HEIGHT_TABLE = (0.084, 0.094, 0.106, 0.121, 0.139, 0.160, 0.182)
DSECOND_TABLE = (-0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45)

# Body-frame xy of the strip the terrain gate measures roughness over,
# alongside the four foot positions.
GATE_STRIP_X = (0.17, 0.33, 0.5)
GATE_STRIP_Y = (-0.2, 0.0, 0.2)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        # Physics backend. auto picks warp on a CUDA host and jax elsewhere.
        # naconmax_per_env scales with the training batch; njmax is per
        # world. See docs/plans/mjwarp-phase0-report.md section 4.
        sim=config_dict.create(
            backend="auto", naconmax_per_env=32, njmax=320, num_envs=1
        ),
        episode_length=1000,
        # Scalar, or a 3-tuple (per joint within a leg: first/abduction,
        # second, third — tiled over the 4 legs) to give joints different
        # command authority around the anchor. Ported from PR #19.
        action_scale=0.5,
        # Clamp actuator forcerange to +-this many N*m (0 = keep the model's
        # +-9). A binding cap forces compliant leg use instead of torque
        # spikes; deploy MUST match (real.launch.py max_torque). Ported from
        # PR #19.
        max_torque=0.0,
        # Override the PD servo gains on the loaded model (0 = keep the
        # XML's kp=20/kd=1 from build_model). Patched at construction like
        # max_torque, so stiffness experiments need no XML regeneration.
        # DR gain/kd scales multiply on top; deploy MUST match
        # (wojtek_real.urdf.xacro kp/kd).
        pd_kp=0.0,
        pd_kd=0.0,
        # Judge the REAL pose, not the set one (off = legacy tables,
        # bit-identical). Legacy pose/stand_still/reset compare measured
        # joints against COMMANDED ctrl targets, so any PD sag becomes a
        # penalty floor the policy cannot remove (0.343 rad summed on the
        # kp40 plant, ~97% of its standing residual) — and HEIGHT_TABLE
        # itself was settled at kp20/kd1, so the height mapping drifts on
        # every other plant. With this flag the env settles a QUASI-RIGID
        # copy of the model (kp 2000, torque unclamped) once per rung at
        # construction: the result is the KINEMATIC standing family — a
        # pure function of the robot's geometry, identical for every
        # pd_kp/pd_kd/max_torque — used as the height mapping and as the
        # pose/stand_still/reset reference. Compensating the actual
        # plant's sag to reach that pose is the policy's job. Motor-target
        # centering stays in ctrl space.
        real_pose_ref=False,
        # Clamp the 4 abduction actuators' ctrlrange to +-this many rad
        # (0 = keep the model's +-pi). The mechanical hip travel exceeds this,
        # so once set the software clamp is the binding limit, not the joint
        # stop; deploy MUST match.
        abduction_ctrl_limit=0.0,
        # Upper bound on the knee (third-joint) motor target, rad (0 = keep
        # the model's ctrlrange). A guard under KNEE_SINGULARITY: the model's
        # ctrlrange upper bound reaches past the singularity on its own, so
        # this is a real clamp, not a no-op, whenever it is below that bound.
        knee_target_max=0.0,
        obs_noise=config_dict.create(
            gyro=0.2, gravity=0.05, joint_pos=0.01, joint_vel=1.5
        ),
        # Declarative observation spec: ordered lists of catalog names (see
        # WojtekEnv._obs_catalog + this env's command/phase additions).
        # `state` is the actor (sensors the real robot has), `privileged`
        # the critic. Changing either changes obs sizes, so checkpoints
        # don't transfer across obs configs.
        obs=config_dict.create(
            include=(),  # obs presets whitelist actor sensors; () = all
            state=(
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_act",
                "command",
                "phase",
            ),
            privileged=(
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_act",
                "command",
                "phase",
                "linvel",
                "foot_contact",
            ),
        ),
        command=config_dict.create(
            vx=(-0.6, 0.6),
            vy=(-0.4, 0.4),
            wz=(-0.7, 0.7),
            # Commanded standing height; the measured stable envelope is
            # 0.084-0.182 m, kept inside with margin for gait dynamics.
            height=(0.09, 0.17),
            resample_steps=250,  # 5 s
            # Velocity part zeroes with this prob (stand training); the
            # height command stays live so the robot stands AT the height.
            zero_prob=0.15,
            # With this prob the linear part zeroes but wz stays: dedicated
            # spin-in-place training (uniform box sampling almost never
            # produces pure rotation, so policies can't turn on the spot).
            # Ported from PR #19.
            pure_wz_prob=0.0,
            # With this prob keep vy, zero vx and wz: dedicated pure-strafe
            # training. Same fix as pure_wz_prob — the uniform box almost
            # never draws pure lateral, so strafing stays undertrained.
            pure_vy_prob=0.0,
            # With this prob command a forward arc: vx redrawn from arc_vx,
            # vy zeroed, wz kept. Same fix family as the two above — the
            # uniform box rarely holds a clean sustained curve (vy crabs it
            # sideways, vx is often near zero), so arc following stays
            # undertrained. Applied after pure_wz/pure_vy (an arc draw
            # overrides them) and before zeroing (standing still wins).
            arc_prob=0.0,
            arc_vx=(0.3, 0.8),
            # With this prob command a clean slow straight walk: vx redrawn
            # from slow_vx, vy and wz zeroed. Same fix family as pure_wz/
            # pure_vy/arc — the uniform box rarely draws a slow forward
            # command uncontaminated by lateral/yaw, so the gait never
            # learns to scale down (courses: straight_slow 0.69 vs 1.24
            # fast). Applied after arc and before zeroing.
            pure_slow_prob=0.0,
            slow_vx=(0.1, 0.35),
            # With this prob command a clean fast straight walk: vx redrawn
            # from fast_vx, vy and wz zeroed. Same fix family as slow —
            # the uniform box serves vx > arc_vx contaminated with random
            # vy/wz, which product-gated tracking prices at ~0, so clean
            # fast walking is never profitable and the policy deadlocks at
            # commands past the arc range (H4: 0.70 m/s at cmd 0.8, 0.00
            # at cmd 1.0). Applied after slow and before zeroing.
            pure_fast_prob=0.0,
            fast_vx=(0.8, 1.2),
            # With this prob command a clean backward walk: vx redrawn from
            # back_vx, vy and wz zeroed. Same fix family as slow/fast — the
            # uniform box only ever serves backward vx contaminated with
            # random vy/wz, and that coverage did not teach the skill:
            # terrain_blind_v2c refused every backward command outright
            # (0.000 m/s). Applied after fast and before zeroing.
            pure_back_prob=0.0,
            back_vx=(-0.8, -0.2),
        ),
        push=config_dict.create(enable=True, interval_steps=200, vel=0.4),
        action_delay=1,  # control steps of latency between policy and motors
        # Control latency at substep granularity, off by default (reproduces
        # the action_delay path above). See _step_with_latency. Sampled at
        # reset. Under the brax training auto-reset wrapper, auto-reset
        # restores a cached state without re-running reset, so this value is
        # fixed per env for the whole training run, not resampled per episode.
        latency=config_dict.create(enable=False, min_substeps=0, max_substeps=5),
        # Encoder miscalibration: a per-joint constant epsilon added to the
        # observed joint angle and subtracted from the written target, so
        # true qpos and the four-bar anchor stay untouched. Off by default
        # (epsilon = 0). Sampled at reset; under the brax training auto-reset
        # wrapper it is fixed per env for the whole run, for the same reason
        # as latency above.
        encoder=config_dict.create(enable=False, range=0.02),
        # Terrain training, off by default. The env then loads the terrain
        # scene, spawns every env on a tile, measures heights from the
        # terrain surface, and the wrapper in terrain_wrapper.py moves envs
        # between tiles after each episode. With enable=False nothing
        # changes, down to the byte. Observations do not change either;
        # that is what makes the tile teleports safe.
        terrain=config_dict.create(
            enable=False,
            # Which generated arena to load: `train` is the shuffled
            # curriculum arena, `eval` the fixed measurement course, `test`
            # the test suite's scratch arena. Build it with
            # `build-terrain --arena <kind>`.
            arena="train",
            # Train on the arena built with `build-terrain --arena train
            # --flat-row`: level 0 is flat ground, every terrain row sits
            # one level higher. The build and this key must agree;
            # terrain_env refuses a mismatched pair.
            flat_row=False,
            # Random heading at every spawn. Costs nothing: the obs carry
            # no heading.
            spawn_yaw=True,
            # Spawn scatter around the pad centre, m. Keep under 0.2 so
            # spawns stay on the 0.6 m pad.
            pad_jitter=0.15,
            # First spawns come from the easiest half of the rows.
            init_level_frac=0.5,
            # Pin every spawn to one row (0-based, clamped to the arena).
            # -1 keeps the init_level_frac sampling. Videos and debugging
            # use this; the curriculum still moves the level afterwards.
            spawn_level=-1,
            # Drop a level when the episode walked less than this fraction
            # of its commanded distance.
            demote_fraction=0.5,
        ),
        # Body-frame grid of terrain heights ahead of the robot, relative to
        # the lowest foot. Terrain only: on the flat scene there is no
        # surface to sample and both components read zero. The critic gets
        # the clean grid, the actor a masked, stale, optionally corrupted
        # copy; height_scan.py has the geometry and the corruption model.
        height_scan=config_dict.create(
            enable=False,
            x_range=(0.65, 1.45),
            y_range=(-0.3, 0.3),
            nx=5,
            ny=5,
            clip=0.3,
            # Sample-and-hold: a capture every hold_steps control steps
            # reaches the actor delay_steps later, so a 15 Hz camera under
            # the 50 Hz policy holds each frame for three steps.
            hold_steps=3,
            delay_steps=1,
            # Feed the actor zeros while the critic still sees the grid.
            dark=False,
            mask=config_dict.create(
                enable=True,
                # Body-frame camera mount and its downward optical-axis
                # pitch. The angles are a D435 depth stream at 424x240
                # (fx=fy=209); the depth range is the usable band.
                mount=(0.32, 0.0, 0.07),
                pitch_deg=15.0,
                hfov_deg=90.7,
                vfov_deg=61.2,
                min_depth=0.3,
                max_depth=3.0,
                # Line of sight against nearer terrain, on top of the
                # frustum: a real camera cannot see a pit floor through the
                # lip in front of it. Rays are sampled, so a rise thinner
                # than the segment/occlusion_samples spacing can slip
                # between two samples.
                occlusion=True,
                occlusion_samples=8,
                occlusion_margin=0.02,
            ),
            corrupt=config_dict.create(
                enable=False,
                noise_prob=0.6,
                drift_prob=0.3,
                blackout_prob=0.1,
                noise_std=0.02,
                drift_z=0.05,
                drift_tilt=0.05,
                dropout_prob=0.05,
                # Per-episode camera pose error, drawn U(+-bound). It moves
                # the actor's mask (frustum and line of sight); the values
                # inside the mask are the drift regime's business.
                pitch_jitter_deg=0.0,
                mount_jitter=0.0,
            ),
        ),
        # EMA low-pass on actions before the PD targets (0 = off). Kills the
        # noise-driven standing limit cycle structurally; mirror the same
        # one-liner on the robot when deploying a policy trained with it.
        action_filter=0.0,
        # Left/right mirror augmentation: with mirror_prob, an env presents
        # the policy a laterally mirrored world (obs mirrored on the way
        # out, actions un-mirrored on the way in; physics, rewards and
        # termination stay in the real frame). Everything learned about one
        # side then transfers to the other by construction — the fix for
        # one-sided emergent collapses (stiff_b's dead right spin). Sampled
        # at reset; under the brax auto-reset wrapper the flag is fixed per
        # env for the whole run, like latency above. Training-only: the
        # deployed policy always sees real observations.
        symmetry=config_dict.create(enable=False, mirror_prob=0.5),
        # NOTE: design standing height is ~0.10 m (Task 3 correction); the
        # plan's original 0.10 min_height would terminate almost every step.
        # max_toggle_deg (0 = off): terminate when any leg's four-bar
        # toggle angle — the interior angle between sixth_link and
        # foot_link, 180 deg = links collinear = branch crossing — exceeds
        # this. A STATE-based guard replacing the knee_target_max command
        # clamp (owner design, 2026-07-27): the same knee target can sit at
        # toggle 14 deg (coordinated extension, measured safe to the full
        # 0.212 m stance) or 177 deg (hip left behind), so the command
        # never was the danger variable. Measured envelope: every healthy
        # behavior <= ~14 deg; crossing at 180. Termination teaches
        # avoidance like fall avoidance — no deployment clamp parity
        # needed (the driver may keep an independent last-resort limit).
        fall=config_dict.create(
            min_height=0.06,
            max_tilt_gz=-0.4,
            max_toggle_deg=0.0,
            # Terrain only: also end the episode when any base chessboard
            # cell is analytically down on the terrain (height lookup, not
            # contact forces). A lying robot is the expensive simulation
            # state and carries no learning signal for locomotion.
            on_base_contact=False,
        ),
        # No-progress termination, CaT-style (arXiv 2403.18765): an env
        # whose measured progress keeps falling short of its command is
        # cut probabilistically. Ending the episode forfeits all remaining
        # income, which is the whole penalty — no reward term is attached
        # and rewards["termination"] stays fall-only. The hazard ramps
        # linearly from 0 where the smoothed progress reaches risk_below
        # of the commanded speed to p_max per control step at zero
        # progress, so expected survival at a dead stop is 1/p_max steps
        # (~1 s at the defaults). Zero commands never arm the cut, and a
        # command change re-arms it only after grace_sec. Respawns keep the
        # dying episode's info, so TerrainAutoResetWrapper reseeds the
        # meter on done; a flat run turning this on needs the same reseed
        # in its auto-reset path.
        no_progress=config_dict.create(
            enable=False,
            grace_sec=2.0,   # no hazard this long after reset/resample
            ema_sec=1.0,     # smoothing horizon of the progress measure
            risk_below=0.5,  # hazard starts below this fraction of demand
            p_max=0.02,      # per-step hazard at zero progress
        ),
        # Trot clock: fbb_v2 skated at 7 Hz instead of stepping (duty factor
        # ~1.0); the phase reward makes periodic swings the only way to score.
        gait=config_dict.create(
            # Clock frequency band: freq[0] at the slowest walk, freq[1] at
            # max commanded speed (speed-scaled; the fbb_run_v1 lesson —
            # stride length is capped by the 0.21 m legs, fast commands
            # need a faster clock, not a frozen one).
            freq=(1.4, 3.0),  # Hz
            swing_height=0.03,  # target peak foot clearance, m
            # Walk->trot blend band on planar command speed (m/s): below the
            # band the offsets are a 4-beat walk, above it a diagonal trot,
            # inside they interpolate (an amble, like a real dog's).
            trot_band=(0.35, 0.55),
            # Stance fraction of the cycle, blended walk->trot with the same
            # band. 50% duty at walking speeds reads as a soft trot (v1:
            # diag-corr 0.82 in the walk band); a real walk is ~70% stance.
            duty=(0.7, 0.5),
            # Cap on the swing time the feet_air_time reward pays for, in
            # seconds (0 = uncapped). Anti-runaway bound on the swing reward;
            # NOT a cadence target. Ported from PR #19.
            air_time_cap=0.0,
        ),
        reward=config_dict.create(
            tracking_sigma=0.25,
            # Far-field mix-in for the two velocity tracking kernels
            # (weight 0 = off, exact legacy kernel). exp(-err^2/sigma) is
            # gradient-free once the error is a few sigma out, so a
            # capability the policy never explored (stiff_b's right spin)
            # gets no pull toward the command at all; blending in a wider
            # second exponential keeps a usable gradient at range while
            # the optimum and the [0, 1] bound stay unchanged.
            tracking_far_weight=0.0,
            tracking_far_sigma=2.5,
            # Height band of the feet_landing penalty: downward foot speed
            # is penalized proportionally to how far INSIDE this clearance
            # band the foot is (1 at the floor, 0 at glide_height and
            # above), so the gradient reads "decelerate as you approach".
            # Measured pre-contact, unlike an at-contact penalty which
            # under-reads hard hits at ctrl_dt granularity (the solver has
            # already absorbed the impact by the step contact is seen).
            glide_height=0.03,
            # Per-swing apex target for the feet_apex reward, m. high_step
            # pays AVERAGE clearance per airborne step, so a long 2 cm skim
            # collects nearly as much as a crisp arc and the optimizer
            # skims (H5b: 1.5-2 cm median apex). feet_apex instead pays,
            # once per swing at touchdown, how close the swing's PEAK got
            # to this target — pricing the apex itself.
            apex_target=0.05,
            # Relative velocity-tracking error (off = legacy absolute).
            # The absolute kernel pays only within ~sqrt(sigma) of the
            # target regardless of the target's size, so fast commands put
            # the whole reward cliff out of exploration's reach (H2:
            # completes wz 0.8 pivots but deadlocks at wz 1.2 — spinning
            # at its proven 0.8 under a 1.2 command would earn only 0.20).
            # Relative mode divides the error by the commanded magnitude
            # (floored below, so zero commands stay well-defined): 80% of
            # target pays the same at every speed. tracking_rel_sigma is
            # dimensionless, calibrated so the keeper's measured operating
            # points (vel_err 0.071 on ramp, wz err 0.13 on turn) score
            # within a few percent of the absolute kernel.
            tracking_relative=False,
            tracking_rel_sigma=0.25,
            tracking_rel_floor_lin=0.3,  # m/s
            tracking_rel_floor_ang=0.4,  # rad/s
            # Multiplicative velocity tracking (off = legacy additive).
            # Additive tracking pays the easy half of a command: a robot
            # that deadlocks under a pure-spin command still earns the
            # FULL tracking_lin_vel (its commanded linear velocity is 0
            # and standing still "tracks" it perfectly) — 4.0/step in the
            # stiff presets, which makes ignoring rotation profitable.
            # With tracking_product both terms are gated by the product of
            # the two kernels: full pay only when the WHOLE command is
            # tracked; a deadlocked spin earns ~0.
            tracking_product=False,
            # Gate the positive gait terms (feet_air_time, feet_apex,
            # high_step) by the linear tracking kernel. Those terms pay on
            # a commanded env whether or not it translates, which made
            # stand-and-lift the top income under a command on terrain
            # (terrain_blind_v3 post-mortem, 2026-07-29). Gated, a stride
            # pays in proportion to how well the command is being served;
            # standing pays nothing for lifting legs.
            shaping_tracking_gate=False,
            phase_sigma=0.002,
            height_sigma=1e-3,  # m^2: 1 cm err -> 0.90, 3 cm -> 0.41
            # Pose anchors to the commanded-height stance; leg joints get a
            # lighter weight than abduction so the gait may flex around it.
            pose_leg_weight=0.25,
            # torque_limit hinge fires above this fraction of each
            # actuator's forcerange cap (the max_torque-customized one, when
            # set), so the policy never learns postures that live near
            # saturation.
            torque_limit_frac=0.85,
            # Tolerance cone around upright for the orientation penalty,
            # half-angle in degrees (0 = legacy penalty, bit-exact). The
            # penalty is sin^2 of the base's tilt from vertical; with a
            # cone it becomes max(sin^2(tilt) - sin^2(tol), 0), zero inside
            # and rising continuously from the edge. A flat-referenced
            # penalty prices the body pitch that climbing requires; a
            # nosedive stays far outside any cone worth setting.
            orientation_tol_deg=0.0,
            # Scale the gait shaping with the roughness under and just ahead
            # of the robot: `floor` of it on flat ground, all of it once the
            # local height spread reaches `rough_ref`. landing_soften
            # discounts the feet_landing charge by the same measure, so a
            # foot may be planted hard on a riser. Terrain only.
            terrain_gate=config_dict.create(
                enable=False, floor=0.15, rough_ref=0.05, landing_soften=0.5
            ),
            scales=config_dict.create(
                tracking_lin_vel=1.5,
                tracking_ang_vel=0.8,
                height_tracking=1.0,
                lin_vel_z=-2.0,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                torques=-2e-4,
                # Step-to-step torque change: penalizes bang-bang motor
                # commands directly at the actuator (action_rate only sees
                # the policy output, not what the PD loop does with it).
                # Ported from PR #19.
                torque_rate=0.0,
                action_rate=-0.25,
                action_accel=-0.1,
                energy=-2e-3,
                pose=-0.5,
                feet_air_time=2.0,
                feet_slip=-0.25,
                # Soft-landing shaping: downward foot speed^2 within
                # glide_height of the floor (see glide_height above). 0 =
                # off; a policy trained with it glides feet into stance
                # instead of striking the floor at swing free-fall speed.
                feet_landing=0.0,
                # Four-bar flat-proximity shaping (0 = off): quadratic ramp
                # from 0 at toggle 90 deg to 1 at 180, summed over legs.
                # The gradient that teaches the policy to stay off the
                # toggle BEFORE fall.max_toggle_deg terminates: H8/H8b
                # showed a bare termination cliff next to the tall-stance
                # operating region is unlearnable (11/12 episodes died on
                # it at 472M; ep_len pinned ~180).
                toggle_flat=0.0,
                # Per-swing apex shaping (see apex_target above). 0 = off.
                feet_apex=0.0,
                feet_phase=1.0,
                contact_match=1.0,
                # High-step shaping (clock-free presets): reward swing-foot
                # clearance up to gait.swing_height whenever moving. Ported
                # from PR #19.
                high_step=0.0,
                stand_still=-0.5,
                # Feet planted while standing: penalize total foot CLEARANCE
                # (continuous height above the floor) at zero command, so a
                # tucked leg is pushed down from any height. Ported from
                # PR #19.
                stand_feet_down=0.0,
                termination=-1.0,
                # Actuator-saturation hinge: 0 well inside the cap, positive
                # above torque_limit_frac of it. Complements the max_torque
                # clamp (the model's absolute ceiling) with a soft margin.
                torque_limit=0.0,
            ),
        ),
    )


class WojtekJoystick(WojtekEnv):
    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        # latency.min/max_substeps and n_substeps are static Python ints at
        # this point (config, not traced arrays), so plain Python checks are
        # safe here and run once outside jit.
        lat = self._config.latency
        if lat.min_substeps > lat.max_substeps:
            raise ValueError(
                f"latency.min_substeps ({lat.min_substeps}) must be <= "
                f"latency.max_substeps ({lat.max_substeps})"
            )
        if lat.max_substeps > self.n_substeps:
            raise ValueError(
                f"latency.max_substeps ({lat.max_substeps}) must be <= "
                f"n_substeps ({self.n_substeps})"
            )
        # Per-actuator motor-target bounds for the step() clip. knee_target_max
        # lowers the knee actuators' upper bound below the model's ctrlrange;
        # 0 keeps self._ctrlrange as-is (mirrors env_jump.py's _target_lo/
        # _target_hi precompute).
        knee_idx = jp.array(KNEE_ACTUATORS)
        self._knee_idx = knee_idx
        self._target_lo = self._ctrlrange[:, 0]
        ktm = self._config.get("knee_target_max", 0.0)
        if ktm:
            self._target_hi = self._ctrlrange[:, 1].at[knee_idx].set(
                jp.minimum(self._ctrlrange[knee_idx, 1], ktm)
            )
        else:
            self._target_hi = self._ctrlrange[:, 1]
        # Four-bar toggle instrumentation (see fall.max_toggle_deg): per
        # leg, the ids needed to compute the interior angle between
        # sixth_link and foot_link — 180 deg = links collinear = the
        # branch-crossing configuration.
        m5 = self._mj_model
        self._toggle_j5 = jp.array(
            [m5.joint(f"{leg}_fifth_joint").id for leg in paths.LEGS]
        )
        self._toggle_jf = jp.array(
            [m5.joint(f"{leg}_foot_joint").id for leg in paths.LEGS]
        )
        self._toggle_site = jp.array(
            [m5.site(f"{leg}_chain_close_b").id for leg in paths.LEGS]
        )
        # Per-actuator torque cap for the torque_limit hinge, read after
        # _customize_model ran so max_torque (when set) is the cap.
        self._torque_cap = jp.array(self._mj_model.actuator_forcerange[:, 1])
        # Orientation tolerance cone as sin^2 of its half-angle, so it can be
        # subtracted directly from sum(square(gravity[:2])). Static config,
        # like the other reward constants resolved here.
        self._orientation_tol = float(
            np.square(
                np.sin(
                    np.radians(
                        self._config.reward.get("orientation_tol_deg", 0.0)
                    )
                )
            )
        )
        # Kinematic standing family (see real_pose_ref in default_config):
        # gain-invariant by construction — settled on a quasi-rigid copy,
        # so it depends on the geometry only, never on this run's gains.
        if self._config.get("real_pose_ref", False):
            heights, poses = self._settle_height_grid()
            # The runtime target clamps (knee_target_max) cap the reachable
            # extension: past the peak the settle heights flatten and then
            # DROP (knees pinned while second joints over-extend). The
            # honest envelope is the strictly-increasing prefix up to the
            # peak; commands above its top saturate there by interp-clamp.
            cut = int(np.argmax(heights)) + 1
            heights, poses = heights[:cut], poses[:cut]
            dsecond = np.array(DSECOND_TABLE)[:cut]
            if cut < 3 or not np.all(np.diff(heights) > 0):
                raise ValueError(
                    f"real_pose_ref: unusable settled-height envelope "
                    f"(increasing prefix of {cut} rungs): {heights}"
                )
            self._anchor_heights = jp.array(heights)
            self._anchor_poses = jp.array(poses)
            self._anchor_dsecond = jp.array(dsecond)
        hs = self._config.height_scan
        self._scan_enabled = bool(hs.enable)
        # The exporter rebuilds this env on the flat scene, so an enabled
        # scan still has to assemble without a height lookup.
        self._scan_live = self._scan_enabled and self._terrain_enabled
        named = {"height_scan", "height_scan_clean"} & (
            set(self._config.obs.state)
            | set(self._config.obs.get("include", ()))
            | set(self._config.obs.privileged)
        )
        if named and not self._scan_enabled:
            raise ValueError(
                f"obs lists name {sorted(named)} but "
                "task.env.height_scan.enable is false"
            )
        if self._scan_enabled:
            if (hs.nx, hs.ny) != (height_scan.NX, height_scan.NY):
                raise ValueError(
                    f"height_scan grid {hs.nx}x{hs.ny}: the observation "
                    f"component and its mirror map are sized for "
                    f"{height_scan.NX}x{height_scan.NY}"
                )
            # Capture and promotion share a tick when the delay reaches the
            # hold, which would make the buffer neither stale nor fresh.
            if not 0 <= hs.delay_steps < hs.hold_steps:
                raise ValueError(
                    f"height_scan.delay_steps ({hs.delay_steps}) must be in "
                    f"[0, hold_steps={hs.hold_steps})"
                )
            self._scan_grid = jp.array(
                height_scan.body_grid(hs.x_range, hs.y_range, hs.nx, hs.ny)
            )
        if self._config.reward.terrain_gate.enable:
            self._gate_strip = jp.array(
                [[x, y] for x in GATE_STRIP_X for y in GATE_STRIP_Y]
            )
        # Mirror maps for the symmetry augmentation, assembled once from
        # the resolved obs lists (so the include filter is respected).
        if self._config.symmetry.enable:
            from wojtek_rl import symmetry

            act_perm, act_sign = symmetry.joint_mirror()
            self._act_perm = jp.array(act_perm)
            self._act_sign = jp.array(act_sign)
            state_perm, state_sign = symmetry.obs_mirror(self.actor_obs_names)
            priv_perm, priv_sign = symmetry.obs_mirror(self._config.obs.privileged)
            self._state_perm = jp.array(state_perm)
            self._state_sign = jp.array(state_sign)
            self._priv_perm = jp.array(priv_perm)
            self._priv_sign = jp.array(priv_sign)

    def _customize_model(self, m):
        kp = self._config.get("pd_kp", 0.0)
        if kp:
            m.actuator_gainprm[:, 0] = kp
            m.actuator_biasprm[:, 1] = -kp
        kd = self._config.get("pd_kd", 0.0)
        if kd:
            m.actuator_biasprm[:, 2] = -kd
        # Ported from PR #19.
        mt = self._config.get("max_torque", 0.0)
        if mt:
            m.actuator_forcerange[:, 0] = -mt
            m.actuator_forcerange[:, 1] = mt
        limit = self._config.get("abduction_ctrl_limit", 0.0)
        if limit:
            idx = np.array(ABDUCTION_ACTUATORS)
            m.actuator_ctrlrange[idx, 0] = np.maximum(
                m.actuator_ctrlrange[idx, 0], -limit
            )
            m.actuator_ctrlrange[idx, 1] = np.minimum(
                m.actuator_ctrlrange[idx, 1], limit
            )

    def _sample_command(self, rng):
        """(vx, vy, wz, height). Zeroing (stand training) keeps the height."""
        r1, r2, r3, r4, r5, r6, r7 = jax.random.split(rng, 7)
        c = self._config.command
        vel = jp.array(
            [
                jax.random.uniform(r1, minval=c.vx[0], maxval=c.vx[1]),
                jax.random.uniform(r2, minval=c.vy[0], maxval=c.vy[1]),
                jax.random.uniform(r3, minval=c.wz[0], maxval=c.wz[1]),
            ]
        )
        # spin-in-place training: keep wz, zero the linear part
        pure_wz = jax.random.bernoulli(r6, c.get("pure_wz_prob", 0.0))
        vel = jp.where(pure_wz, vel.at[:2].set(0.0), vel)
        # pure-strafe training: keep vy, zero vx and wz
        pure_vy = jax.random.bernoulli(r7, c.get("pure_vy_prob", 0.0))
        vel = jp.where(pure_vy, jp.array([0.0, vel[1], 0.0]), vel)
        # forward-arc training: redraw vx from arc_vx, zero vy, keep wz.
        # Gated on the static config value and keyed off a fold_in of the
        # incoming rng, so presets with arc_prob=0 draw nothing extra and
        # keep their sampling stream (and trajectories) bit-identical.
        arc_p = c.get("arc_prob", 0.0)
        if arc_p:
            r8, r9 = jax.random.split(jax.random.fold_in(rng, 1))
            arc = jax.random.bernoulli(r8, arc_p)
            vx_arc = jax.random.uniform(
                r9, minval=c.arc_vx[0], maxval=c.arc_vx[1]
            )
            vel = jp.where(arc, jp.array([vx_arc, 0.0, vel[2]]), vel)
        # slow-walk training: redraw vx from slow_vx, zero vy and wz.
        # Gated and fold_in-keyed like arc, so presets with the prob at 0
        # keep their sampling stream bit-identical.
        slow_p = c.get("pure_slow_prob", 0.0)
        if slow_p:
            r10, r11 = jax.random.split(jax.random.fold_in(rng, 2))
            slow = jax.random.bernoulli(r10, slow_p)
            vx_slow = jax.random.uniform(
                r11, minval=c.slow_vx[0], maxval=c.slow_vx[1]
            )
            vel = jp.where(slow, jp.array([vx_slow, 0.0, 0.0]), vel)
        # fast-walk training: redraw vx from fast_vx, zero vy and wz.
        # Gated and fold_in-keyed like arc/slow (index 3), stream-safe.
        fast_p = c.get("pure_fast_prob", 0.0)
        if fast_p:
            r12, r13 = jax.random.split(jax.random.fold_in(rng, 3))
            fast = jax.random.bernoulli(r12, fast_p)
            vx_fast = jax.random.uniform(
                r13, minval=c.fast_vx[0], maxval=c.fast_vx[1]
            )
            vel = jp.where(fast, jp.array([vx_fast, 0.0, 0.0]), vel)
        # backward training: redraw vx from back_vx, zero vy and wz.
        # Gated and fold_in-keyed like arc/slow/fast (index 4), stream-safe.
        back_p = c.get("pure_back_prob", 0.0)
        if back_p:
            r14, r15 = jax.random.split(jax.random.fold_in(rng, 4))
            back = jax.random.bernoulli(r14, back_p)
            vx_back = jax.random.uniform(
                r15, minval=c.back_vx[0], maxval=c.back_vx[1]
            )
            vel = jp.where(back, jp.array([vx_back, 0.0, 0.0]), vel)
        zero = jax.random.bernoulli(r4, c.zero_prob)
        vel = jp.where(zero, jp.zeros(3), vel)
        height = jax.random.uniform(r5, minval=c.height[0], maxval=c.height[1])
        return jp.concatenate([vel, height[None]])

    def _settle_height_grid(self):
        """Kinematic standing family: settle each DSECOND_TABLE rung on a
        QUASI-RIGID copy of the model (kp 2000 / kd 100, torque
        unclamped), so sag is ~1e-3 rad and the result is a function of
        the geometry alone — identical for every runtime gains config.
        Plain MuJoCo, ~1 s one-off at construction.
        """
        import copy

        import mujoco

        m = copy.deepcopy(self._mj_model)
        # kp 400 leaves ~5e-3 rad of sag (negligible) while staying
        # integrable; the implicitfast integrator + a fine timestep keep
        # the stiff servo stable regardless of the training sim_dt.
        m.actuator_gainprm[:, 0] = 400.0
        m.actuator_biasprm[:, 1] = -400.0
        m.actuator_biasprm[:, 2] = -20.0
        m.actuator_forcerange[:, 0] = -1e6
        m.actuator_forcerange[:, 1] = 1e6
        m.opt.timestep = 5e-4
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        d = mujoco.MjData(m)
        qadr = np.asarray(self._qadr)
        # Clamp with the RUNTIME target bounds (knee_target_max included),
        # not the raw model ctrlrange: step() clips motor targets to
        # _target_lo/_target_hi, so an envelope settled past those bounds
        # would promise heights the policy can never command (measured:
        # the kinematic table reaches 0.21 m but knee anchors pass the
        # 3.15 singularity guard above ~0.155 m).
        lo = np.asarray(self._target_lo)
        hi = np.asarray(self._target_hi)
        home_ctrl = np.asarray(self._home_ctrl)
        n_steps = int(round(2.0 / m.opt.timestep))
        heights, poses = [], []
        for ds in DSECOND_TABLE:
            ctrl = np.clip(
                home_ctrl + np.tile([0.0, 1.0, 2.0], 4) * ds, lo, hi
            )
            mujoco.mj_resetData(m, d)
            d.qpos[:] = np.asarray(self._home_qpos)
            d.ctrl[:] = ctrl
            for _ in range(n_steps):
                mujoco.mj_step(m, d)
            heights.append(float(d.qpos[2]))
            poses.append(d.qpos[qadr].copy())
        return np.array(heights), np.array(poses)

    def _pose_ref(self, height):
        """Joint-pose reference for a commanded height: the kinematic real
        pose when real_pose_ref is on, else the legacy ctrl anchor (which
        conflates set targets with expected readings — kept for old
        presets)."""
        if self._config.get("real_pose_ref", False):
            return jp.stack(
                [
                    jp.interp(height, self._anchor_heights, self._anchor_poses[:, j])
                    for j in range(12)
                ]
            )
        return self._height_ctrl(height)

    def _cmd_speed(self, command):
        """Planar speed the gait clock should serve; turning counts too."""
        return jp.linalg.norm(command[:2]) + 0.3 * jp.abs(command[2])

    def _height_ctrl(self, height):
        """Ctrl anchor for a commanded standing height (measured table)."""
        if self._config.get("real_pose_ref", False):
            dsecond = jp.interp(height, self._anchor_heights, self._anchor_dsecond)
        else:
            dsecond = jp.interp(
                height, jp.array(HEIGHT_TABLE), jp.array(DSECOND_TABLE)
            )
        offset = jp.tile(jp.array([0.0, 1.0, 2.0]), 4) * dsecond
        return jp.clip(
            self._home_ctrl + offset, self._ctrlrange[:, 0], self._ctrlrange[:, 1]
        )

    def _gait_frac(self, command):
        """0 = walk, 1 = trot, blended across gait.trot_band."""
        band = self._config.gait.trot_band
        return jp.clip(
            (self._cmd_speed(command) - band[0]) / (band[1] - band[0]), 0.0, 1.0
        )

    def _leg_phases(self, info):
        """Per-leg clock: master phase + speed-blended walk/trot offsets."""
        f = self._gait_frac(info["command"])
        offsets = (1.0 - f) * jp.array(WALK_PHASE) + f * jp.array(TROT_PHASE)
        phase = info["phase"] + offsets
        return jp.fmod(phase + jp.pi, 2 * jp.pi) - jp.pi

    def _gait_targets(self, info):
        """(target foot clearance, stance mask) from the duty-aware clock.

        theta in [0,1) is the normalized leg phase; the swing window is the
        first (1-duty) of the cycle with a sinusoidal clearance bump, the
        rest is stance on the ground.
        """
        g = self._config.gait
        f = self._gait_frac(info["command"])
        duty = (1.0 - f) * g.duty[0] + f * g.duty[1]
        theta = jp.fmod(self._leg_phases(info) + 2 * jp.pi, 2 * jp.pi) / (
            2 * jp.pi
        )
        swing_frac = 1.0 - duty
        in_swing = theta < swing_frac
        rz = g.swing_height * jp.sin(jp.pi * theta / swing_frac) * in_swing
        return rz, ~in_swing

    # -- height scan --------------------------------------------------------
    def _scan_raw(self, data, cam_jit=None):
        """(clean scan, camera visibility mask) at the current pose.

        `cam_jit` offsets the camera pose the mask is computed from,
        (dpitch_deg, dx, dy, dz); None is the nominal mount. The clean scan
        does not depend on it.
        """
        hs = self._config.height_scan
        quat = self._quat(data)
        xy = height_scan.world_xy(
            self._scan_grid, data.qpos[0:2], height_scan.yaw_from_quat(quat)
        )
        h = self._terrain.height(xy)
        ref_z = jp.min(data.geom_xpos[self._foot_geom_ids][:, 2])
        clean = height_scan.scan_values(h, ref_z, hs.clip)
        if not hs.mask.enable:
            return clean, jp.ones_like(clean, dtype=bool)
        m = hs.mask
        points = jp.concatenate([xy, h[:, None]], axis=-1)
        mount = jp.array(m.mount)
        pitch_deg = m.pitch_deg
        if cam_jit is not None:
            mount = mount + cam_jit[1:4]
            pitch_deg = pitch_deg + cam_jit[0]
        mask = height_scan.visible_mask(
            points,
            data.qpos[0:3],
            quat,
            mount,
            pitch_deg,
            m.hfov_deg,
            m.vfov_deg,
            m.min_depth,
            m.max_depth,
        )
        if m.occlusion:
            mask = mask & ~height_scan.occluded_mask(
                points,
                height_scan.camera_pos(data.qpos[0:3], quat, mount),
                self._terrain.height,
                m.occlusion_samples,
                m.occlusion_margin,
            )
        return clean, mask

    def _scan_actor(self, data, rng, regime, drift, cam_jit=None):
        """Masked scan with the actor-side sensor corruption applied."""
        clean, mask = self._scan_raw(data, cam_jit)
        scan = clean * mask
        c = self._config.height_scan.corrupt
        if not c.enable:
            return scan
        return height_scan.apply_corruption(
            rng, scan, regime, drift, self._scan_grid[:, 0], c.noise_std,
            c.dropout_prob,
        )

    def _step_scan(self, data, info):
        """Advance the sample-and-hold camera one control step: capture into
        scan_pending every hold_steps, promote it to scan_hold delay_steps
        later."""
        hs = self._config.height_scan
        rng, r_scan = jax.random.split(info["scan_rng"])
        info["scan_rng"] = rng
        fresh = self._scan_actor(
            data, r_scan, info["scan_regime"], info["scan_drift"],
            info["scan_cam_jit"],
        )
        phase = info["scan_step"] % hs.hold_steps
        pending = jp.where(phase == 0, fresh, info["scan_pending"])
        info["scan_pending"] = pending
        info["scan_hold"] = jp.where(
            phase == hs.delay_steps, pending, info["scan_hold"]
        )
        info["scan_step"] = info["scan_step"] + 1

    def _phase_dt(self, command):
        """Clock increment: speed-scaled; frozen when told to stand."""
        g = self._config.gait
        c = self._config.command
        vmax = self._cmd_speed(
            jp.array([max(abs(c.vx[0]), abs(c.vx[1])), 0.0, 0.0, 0.0])
        )
        speed = self._cmd_speed(command)
        frac = jp.clip(speed / vmax, 0.0, 1.0)
        freq = g.freq[0] + (g.freq[1] - g.freq[0]) * frac
        return jp.where(speed > 0.05, 2 * jp.pi * self.dt * freq, 0.0)

    # -- reset / step -------------------------------------------------------
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, r_cmd, r_pos = jax.random.split(rng, 3)
        command = self._sample_command(r_cmd)
        # Start posed for the commanded height: legs on the pose reference
        # (the measured settled pose when calibrated, else the ctrl
        # anchor), plus joint noise; ctrl holds the ctrl-space anchor.
        anchor = self._height_ctrl(command[3])
        qpos = self._home_qpos.at[self._qadr].set(
            self._pose_ref(command[3])
            + jax.random.uniform(r_pos, (12,), minval=-0.05, maxval=0.05)
        )
        qpos = qpos.at[2].set(command[3])
        # Terrain spawn: random type, easy starting row, base lifted by the
        # pad height. The disabled branch draws no rng and adds no info
        # keys, so flat runs are untouched.
        if self._terrain_enabled:
            rng, r_type, r_level, r_spawn, r_trng = jax.random.split(rng, 5)
            terrain_type = jax.random.randint(r_type, (), 0, self._terrain.n_types)
            if self._terrain.spawn_level >= 0:
                level = jp.minimum(
                    self._terrain.spawn_level, self._terrain.n_rows - 1
                )
            else:
                init_rows = max(
                    1, round(self._terrain.n_rows * self._terrain.init_level_frac)
                )
                level = jax.random.randint(r_level, (), 0, init_rows)
            spawn_xy, pad_height, quat = terrain_env.sample_tile_spawn(
                r_spawn, terrain_type, level,
                self._terrain.origin_xy, self._terrain.pad_h,
                self._terrain.pad_jitter, self._terrain.spawn_yaw,
            )
            qpos = qpos.at[0:2].set(spawn_xy)
            qpos = qpos.at[2].set(pad_height + command[3])
            qpos = qpos.at[3:7].set(quat)
        data = self._make_data()
        data = data.replace(qpos=qpos, qvel=jp.zeros(self._mj_model.nv), ctrl=anchor)
        data = mjx.forward(self._mjx_model, data)
        # Control delay, sampled at reset (fixed per env for the whole
        # training run under auto-reset; see default_config). The disabled
        # branch draws no rng, so the default trajectory is unchanged.
        lat = self._config.latency
        if lat.enable:
            rng, r_delay = jax.random.split(rng)  # only consumed when enabled
            d = jax.random.randint(r_delay, (), lat.min_substeps, lat.max_substeps + 1)
            d = jp.clip(d, 0, self.n_substeps)
        else:
            d = self.n_substeps if self._config.action_delay > 0 else 0
        # Per-joint encoder offset, sampled at reset (fixed per env for the
        # whole training run under auto-reset; see default_config). The
        # disabled branch draws no rng, so the default trajectory is
        # unchanged.
        enc = self._config.encoder
        if enc.enable:
            rng, r_enc = jax.random.split(rng)  # only consumed when enabled
            epsilon = jax.random.uniform(
                r_enc, (12,), minval=-enc.range, maxval=enc.range
            )
        else:
            epsilon = jp.zeros(12)
        # Mirror flag, sampled at reset (fixed per env for the whole run
        # under auto-reset, like latency/encoder above). The disabled
        # branch draws no rng, so the default trajectory is unchanged.
        sym = self._config.symmetry
        if sym.enable:
            rng, r_mirror = jax.random.split(rng)  # only consumed when enabled
            mirror = jax.random.bernoulli(r_mirror, sym.mirror_prob)
        else:
            mirror = jp.array(False)
        # Height scan: the episode's corruption draw, a random capture phase
        # so the envs are not all on the same camera tick, and both buffers
        # seeded with the current frame (a zero seed would read as flat
        # ground for the first hold).
        if self._scan_live:
            rng, r_regime, r_scan, r_phase, r_hold = jax.random.split(rng, 5)
            hc = self._config.height_scan.corrupt
            scan_regime, scan_drift, scan_cam_jit = height_scan.sample_corruption(
                r_regime, hc.noise_prob, hc.drift_prob, hc.blackout_prob,
                hc.drift_z, hc.drift_tilt, hc.pitch_jitter_deg,
                hc.mount_jitter,
            )
            scan0 = self._scan_actor(
                data, r_scan, scan_regime, scan_drift, scan_cam_jit
            )
            scan_step = jax.random.randint(
                r_phase, (), 0, self._config.height_scan.hold_steps
            )
        info = {
            "mirror": mirror,
            "rng": rng,
            "command": command,
            "last_act": jp.zeros(12),
            "last_last_act": jp.zeros(12),
            "filtered_act": jp.zeros(12),
            "steps_since_cmd": jp.array(0),
            "feet_air_time": jp.zeros(4),
            "swing_apex": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "last_torque": jp.zeros(12),
            "motor_targets": anchor,
            "step_count": jp.array(0),
            "phase": jp.array(0.0),  # master clock; per-leg via _leg_phases
            "ctrl_delay": jp.int32(d),
            "encoder_offset": epsilon,
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward.scales}
        if self._config.no_progress.enable:
            # Optimistic seed: a fresh episode starts at progress ratio 1,
            # so the hazard can only come from measured shortfall, never
            # from the seed (see no_progress in default_config).
            info["progress_ema"] = self._cmd_speed(command)
            metrics["no_progress_cut"] = jp.zeros(())
            metrics["progress_ratio_per_step"] = jp.zeros(())
        # State for the curriculum wrapper. terrain_type never changes; the
        # wrapper rewrites the rest on done. The env refreshes last_xy and
        # commanded_dist every step.
        if self._terrain_enabled:
            info.update(
                terrain_type=terrain_type,
                terrain_level=level,
                spawn_xy=spawn_xy,
                spawn_height=command[3],
                last_xy=spawn_xy,
                commanded_dist=jp.zeros(()),
                terrain_rng=r_trng,
            )
            metrics["terrain_level_per_step"] = level.astype(jp.float32)
            # Diagnostic, written every step below. Zero here: the spawn stands
            # on a flat pad and no episode has run yet.
            metrics["base_contact_alive_per_step"] = jp.zeros(())
            metrics["base_contact_at_done"] = jp.zeros(())
        if self._scan_live:
            info.update(
                scan_regime=scan_regime,
                scan_drift=scan_drift,
                scan_cam_jit=scan_cam_jit,
                scan_rng=r_hold,
                scan_step=scan_step,
                scan_hold=scan0,
                scan_pending=scan0,
            )
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        info = dict(state.info)
        if self._config.no_progress.enable:
            # The extra key is only split off when the cut is on, so the
            # default trajectory is unchanged (the latency/encoder pattern).
            rng, r_noise, r_cmd, r_push, r_term = jax.random.split(info["rng"], 5)
        else:
            rng, r_noise, r_cmd, r_push = jax.random.split(info["rng"], 4)
            r_term = None
        info["rng"] = rng

        # Symmetry augmentation: the policy acted on mirrored observations,
        # so its action is in the mirrored frame; map it back before the
        # physics. Everything below (dynamics, rewards, info) stays in the
        # real frame; _get_obs mirrors the outgoing observations (which
        # maps the real-frame last_act back to the action the policy
        # actually emitted, since the mirror is an involution).
        if self._config.symmetry.enable:
            action = jp.where(
                info["mirror"], self._act_sign * action[self._act_perm], action
            )

        # Actions center on the commanded-height stance, so "zero action"
        # means "stand at the commanded height" at any height.
        af = self._config.action_filter
        filt = af * info["filtered_act"] + (1.0 - af) * action
        info["filtered_act"] = filt
        scale = jp.asarray(self._config.action_scale)
        scale = jp.tile(scale, 4) if scale.ndim > 0 and scale.size == 3 else scale
        motor_targets = jp.clip(
            self._height_ctrl(info["command"][3]) + filt * scale,
            self._target_lo,
            self._target_hi,
        )
        data = state.data
        # random horizontal push, every push.interval_steps
        if self._config.push.enable:
            push_now = (info["step_count"] % self._config.push.interval_steps) == (
                self._config.push.interval_steps - 1
            )
            push = jax.random.uniform(r_push, (2,), minval=-1.0, maxval=1.0)
            push = push / (jp.linalg.norm(push) + 1e-6) * self._config.push.vel
            qvel = data.qvel.at[:2].add(jp.where(push_now, push, jp.zeros(2)))
            data = data.replace(qvel=qvel)

        # Subtract the encoder offset from the written targets, matching the
        # real driver's error against q~ = q + epsilon. Zero when disabled.
        eps = info["encoder_offset"]
        if self._config.latency.enable:
            # Substeps before ctrl_delay use the previous targets, the rest
            # the new ones.
            data = self._step_with_latency(
                data, info["motor_targets"] - eps, motor_targets - eps, info["ctrl_delay"]
            )
        else:
            # Default path: one ctrl for the whole period. Kept on the stock
            # mjx_env.step so the trajectory is unchanged; the where-scan
            # above shifts the float output by a few ULPs.
            applied_targets = (
                info["motor_targets"] if self._config.action_delay > 0
                else motor_targets
            )
            data = mjx_env.step(
                self._mjx_model, data, applied_targets - eps, self.n_substeps
            )

        contact = self._foot_contact(data)
        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0) & contact_filt
        # Track each swing's peak clearance while the foot is airborne;
        # at first_contact the reward reads the completed swing's apex.
        step_clearance = self._foot_clearance(data)
        info["swing_apex"] = jp.where(
            ~contact_filt,
            jp.maximum(info["swing_apex"], step_clearance),
            info["swing_apex"],
        )
        rewards, done = self._get_reward(
            data, info, action, motor_targets, first_contact, contact
        )
        info["swing_apex"] = jp.where(contact_filt, 0.0, info["swing_apex"])
        info["feet_air_time"] = jp.where(
            contact_filt, 0.0, info["feet_air_time"] + self.dt
        )
        info["last_contact"] = contact
        info["last_last_act"] = info["last_act"]
        info["last_act"] = action
        info["last_torque"] = data.actuator_force
        info["motor_targets"] = motor_targets
        # For the curriculum wrapper: where the base is, and how far the
        # commands asked to go so far. The resample below has not run yet,
        # so this is the command that drove this step.
        if self._terrain_enabled:
            info["last_xy"] = data.qpos[0:2]
            info["commanded_dist"] = info["commanded_dist"] + (
                jp.linalg.norm(info["command"][:2]) * self.dt
            )
        # Master clock advances at the speed the CURRENT command asks for
        # (frozen when standing); per-leg phases come from _leg_phases.
        phase = info["phase"] + self._phase_dt(info["command"])
        info["phase"] = jp.fmod(phase + jp.pi, 2 * jp.pi) - jp.pi
        info["step_count"] = info["step_count"] + 1
        info["steps_since_cmd"] = info["steps_since_cmd"] + 1
        # No-progress cut (see no_progress in default_config): an env that
        # keeps ignoring its command dies with a probability that grows
        # with the shortfall. Evaluated against the command that drove
        # this step, before the resample below.
        if self._config.no_progress.enable:
            npg = self._config.no_progress
            cmd = info["command"]
            demand = self._cmd_speed(cmd)
            # Achieved speed in service of the command: planar velocity
            # projected onto the commanded direction, plus yaw rate toward
            # the commanded turn, blended like _cmd_speed blends demand.
            # Moving against the command reads negative, i.e. worse than
            # standing.
            served = (
                jp.dot(self._local_linvel(data)[:2], cmd[:2])
                / jp.maximum(jp.linalg.norm(cmd[:2]), 1e-6)
                + 0.3 * self._gyro(data)[2] * jp.sign(cmd[2])
            )
            alpha = self.dt / npg.ema_sec
            info["progress_ema"] = (
                (1.0 - alpha) * info["progress_ema"] + alpha * served
            )
            progress_ratio = info["progress_ema"] / jp.maximum(demand, 1e-6)
            hazard = npg.p_max * jp.clip(
                (npg.risk_below - progress_ratio) / npg.risk_below, 0.0, 1.0
            )
            armed = (demand > 0.05) & (
                info["steps_since_cmd"] * self.dt >= npg.grace_sec
            )
            no_progress_cut = jax.random.bernoulli(
                r_term, jp.where(armed, hazard, 0.0)
            )
            done = done | no_progress_cut
        resample = info["steps_since_cmd"] >= self._config.command.resample_steps
        info["command"] = jp.where(
            resample, self._sample_command(r_cmd), info["command"]
        )
        info["steps_since_cmd"] = jp.where(resample, 0, info["steps_since_cmd"])
        if self._config.no_progress.enable:
            # A fresh command restarts the meter at ratio 1: the EMA is
            # reseeded to the new demand, and the grace window keeps the
            # hazard at zero while the robot turns the gait around.
            info["progress_ema"] = jp.where(
                resample, self._cmd_speed(info["command"]), info["progress_ema"]
            )

        reward = sum(
            rewards[k] * self._config.reward.scales[k] for k in rewards
        )
        reward = jp.clip(reward * self.dt, -100.0, 100.0)
        # Merge over the incoming metrics: brax's EvalWrapper injects keys
        # (e.g. "reward") that must survive the step for scan carry parity.
        metrics = {
            **state.metrics,
            **{f"reward/{k}": v for k, v in rewards.items()},
        }
        if self._config.no_progress.enable:
            # Episode sum is 1 exactly when the episode ended on the cut;
            # the ratio is a per-step mean, clipped so over-delivery on a
            # slow command cannot hide shortfall elsewhere in the average.
            metrics["no_progress_cut"] = no_progress_cut.astype(jp.float32)
            metrics["progress_ratio_per_step"] = jp.clip(progress_ratio, 0.0, 2.0)
        # The `_per_step` suffix makes brax report the mean, not the sum.
        if self._terrain_enabled:
            metrics["terrain_level_per_step"] = info["terrain_level"].astype(jp.float32)
            # Which states put the base on the terrain, which is where the
            # MJWarp heightfield contact cap gets hit. Belly-dragging while the
            # episode runs is the one that grows with the curriculum; a base
            # down on the terminating step is an ordinary fall. Split so the
            # two are not read as one number. The second has no `_per_step`
            # suffix on purpose: brax reports its episode sum, which for this
            # indicator is 1 exactly when the episode ended with the base down.
            base_down = self._base_terrain_contact(data).astype(jp.float32)
            terminated = done.astype(jp.float32)
            metrics["base_contact_alive_per_step"] = base_down * (1.0 - terminated)
            metrics["base_contact_at_done"] = base_down * terminated
        if self._scan_live:
            self._step_scan(data, info)
        obs = self._get_obs(data, info, r_noise)
        return mjx_env.State(data, obs, reward, done.astype(jp.float32), metrics, info)

    # -- observations -------------------------------------------------------
    def _obs_catalog(self, data, info):
        catalog = super()._obs_catalog(data, info)
        # Add the encoder offset to the observed joint angle (q~ = q +
        # epsilon). Zero when disabled.
        catalog["joint_pos"] = catalog["joint_pos"] + info["encoder_offset"]
        catalog["command"] = info["command"]
        leg_phase = self._leg_phases(info)
        catalog["phase"] = jp.concatenate([jp.cos(leg_phase), jp.sin(leg_phase)])
        if self._scan_enabled:
            if not self._scan_live:
                zeros = jp.zeros(height_scan.SIZE)
                catalog["height_scan_clean"] = zeros
                catalog["height_scan"] = zeros
            else:
                catalog["height_scan_clean"] = self._scan_raw(data)[0]
                catalog["height_scan"] = (
                    jp.zeros(height_scan.SIZE)
                    if self._config.height_scan.dark
                    else info["scan_hold"]
                )
        return catalog

    def _get_obs(self, data, info, rng=None):
        obs = self._build_obs(data, info, rng)
        if self._config.symmetry.enable:
            m = info["mirror"]
            obs = {
                "state": jp.where(
                    m, self._state_sign * obs["state"][self._state_perm],
                    obs["state"],
                ),
                "privileged_state": jp.where(
                    m,
                    self._priv_sign * obs["privileged_state"][self._priv_perm],
                    obs["privileged_state"],
                ),
            }
        return obs

    # -- rewards ------------------------------------------------------------
    def _get_reward(self, data, info, action, motor_targets, first_contact, contact):
        cmd = info["command"]
        linvel = self._local_linvel(data)
        gyro = self._gyro(data)
        gravity = self._gravity_body(data)
        sigma = self._config.reward.tracking_sigma
        moving = self._cmd_speed(cmd) > 0.05
        # Pose/stand references judge the REAL expected pose (measured
        # settled pose when calibrated), not the commanded targets.
        anchor = self._pose_ref(cmd[3])
        # Abduction full weight; leg joints lighter (gait flexes around it).
        pose_w = jp.tile(
            jp.array([1.0, self._config.reward.pose_leg_weight,
                      self._config.reward.pose_leg_weight]), 4
        )

        # Terrain: measured from the surface. Flat: the old expressions,
        # unchanged.
        base_height = self._base_height(data)
        fall = (base_height < self._config.fall.min_height) | (
            gravity[2] > self._config.fall.max_tilt_gz
        )
        max_toggle = self._config.fall.get("max_toggle_deg", 0.0)
        toggle_on = max_toggle or self._config.reward.scales.get("toggle_flat", 0.0)
        toggle_ramp = jp.zeros(())
        if toggle_on:
            # Four-bar toggle angle per leg (see fall.max_toggle_deg):
            # interior angle between sixth_link (fifth->foot joint anchors)
            # and foot_link (foot joint anchor -> chain_close_b site).
            a5 = data.xanchor[self._toggle_j5]
            af = data.xanchor[self._toggle_jf]
            sb = data.site_xpos[self._toggle_site]
            u = af - a5
            v = sb - af
            cosang = jp.sum(u * v, axis=-1) / (
                jp.linalg.norm(u, axis=-1) * jp.linalg.norm(v, axis=-1)
                + 1e-9
            )
            toggle_deg = jp.degrees(jp.arccos(jp.clip(cosang, -1.0, 1.0)))
            # smooth proximity ramp: 0 at 90 deg, 1 at 180 (see toggle_flat)
            toggle_ramp = jp.sum(
                jp.square(jp.clip((toggle_deg - 90.0) / 90.0, 0.0, 1.0))
            )
            if max_toggle:
                # angle > max_toggle_deg <=> cos(angle) < cos(max_toggle_deg)
                fall = fall | jp.any(
                    cosang < jp.cos(jp.deg2rad(max_toggle))
                )
        if self._terrain_enabled and self._config.fall.on_base_contact:
            fall = fall | self._base_terrain_contact(data)

        # Foot clearance above the local ground vs the duty-aware swing profile,
        # plus explicit contact/stance schedule matching.
        foot_clearance = self._foot_clearance(data)
        target_clearance, stance_mask = self._gait_targets(info)
        phase_err = jp.sum(jp.square(foot_clearance - target_clearance))
        contact_match = jp.mean(
            (contact == stance_mask).astype(jp.float32)
        )

        # Horizontal foot speed while in contact = skating.
        feet_vel = data.sensordata[self._foot_linvel_adr]
        slip = jp.sum(jp.sum(jp.square(feet_vel[:, :2]), axis=-1) * contact)

        # High-step shaping (ported from PR #19): swing feet earn clearance
        # up to the swing_height target.
        swing = ~contact
        high_step = jp.mean(
            jp.clip(foot_clearance / self._config.gait.swing_height, 0.0, 1.0)
            * swing
        )

        qvel = data.qvel[self._vadr]

        # Velocity tracking kernels (absolute or relative), optionally
        # blended with a wider far-field exponential so far-off-command
        # states still see a gradient toward the command (see
        # tracking_far_weight in default_config). far_w is static config,
        # so 0 reproduces either bare kernel exactly.
        far_w = self._config.reward.get("tracking_far_weight", 0.0)

        def _far_blend(kernel, err_sq):
            if not far_w:
                return kernel
            far = jp.exp(-err_sq / self._config.reward.tracking_far_sigma)
            return (1.0 - far_w) * kernel + far_w * far

        def _track(err_sq):
            return _far_blend(jp.exp(-err_sq / sigma), err_sq)

        if self._config.reward.get("tracking_relative", False):
            r = self._config.reward
            err_lin = jp.sum(jp.square(cmd[:2] - linvel[:2]))
            err_ang = jp.square(cmd[2] - gyro[2])
            denom_lin = jp.maximum(
                jp.linalg.norm(cmd[:2]), r.tracking_rel_floor_lin
            )
            denom_ang = jp.maximum(jp.abs(cmd[2]), r.tracking_rel_floor_ang)
            k_lin = jp.exp(
                -err_lin / (r.tracking_rel_sigma * jp.square(denom_lin))
            )
            k_ang = jp.exp(
                -err_ang / (r.tracking_rel_sigma * jp.square(denom_ang))
            )
            # The far mix-in applies to the relative kernels too. The far
            # kernel stays absolute: a state far off the command sees the
            # same pull at any commanded speed, and the kernel's optimum
            # is unchanged.
            k_lin = _far_blend(k_lin, err_lin)
            k_ang = _far_blend(k_ang, err_ang)
        else:
            k_lin = _track(jp.sum(jp.square(cmd[:2] - linvel[:2])))
            k_ang = _track(jp.square(cmd[2] - gyro[2]))
        if self._config.reward.get("tracking_product", False):
            # Gate each term by the other's kernel: tracking pays only for
            # tracking the whole command (see default_config note).
            k_lin, k_ang = k_lin * k_ang, k_ang * k_lin
        # Gait-shaping gate (see shaping_tracking_gate in default_config):
        # the positive gait terms follow the linear tracking kernel, after
        # the product gate when that is on.
        shape_gate = (
            k_lin
            if self._config.reward.get("shaping_tracking_gate", False)
            else 1.0
        )
        # Roughness gate (see reward.terrain_gate): the height spread under
        # the feet and over the strip just ahead of them. Flat ground pays
        # `floor` of the lift shaping and the full landing charge; rough
        # ground the reverse. Both factors stay Python 1.0 when the gate is
        # off, so the terms are the expressions they always were.
        gate = 1.0
        landing_soften = 1.0
        if self._terrain_enabled and self._config.reward.terrain_gate.enable:
            tg = self._config.reward.terrain_gate
            strip = height_scan.world_xy(
                self._gate_strip,
                data.qpos[0:2],
                height_scan.yaw_from_quat(self._quat(data)),
            )
            local = self._terrain.height(
                jp.concatenate(
                    [data.geom_xpos[self._foot_geom_ids][:, :2], strip]
                )
            )
            rough = jp.clip(
                (jp.max(local) - jp.min(local)) / tg.rough_ref, 0.0, 1.0
            )
            gate = tg.floor + (1.0 - tg.floor) * rough
            landing_soften = 1.0 - tg.landing_soften * rough

        # sin^2 of the tilt from vertical, less the tolerance cone (see
        # reward.orientation_tol_deg). Static config, so 0 leaves the legacy
        # expression untouched.
        orientation = jp.sum(jp.square(gravity[:2]))
        if self._orientation_tol:
            orientation = jp.maximum(orientation - self._orientation_tol, 0.0)

        rewards = {
            "tracking_lin_vel": k_lin,
            "tracking_ang_vel": k_ang,
            "height_tracking": jp.exp(
                -jp.square(base_height - cmd[3])
                / self._config.reward.height_sigma
            ),
            "lin_vel_z": jp.square(linvel[2]),
            "ang_vel_xy": jp.sum(jp.square(gyro[:2])),
            "orientation": orientation,
            "torques": jp.sum(jp.square(data.actuator_force)),
            "torque_rate": jp.sum(
                jp.square(data.actuator_force - info["last_torque"])
            ),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "action_accel": jp.sum(
                jp.square(action - 2 * info["last_act"] + info["last_last_act"])
            ),
            "energy": jp.sum(jp.abs(qvel) * jp.abs(data.actuator_force)),
            "pose": jp.sum(pose_w * jp.square(data.qpos[self._qadr] - anchor)),
            "feet_air_time": jp.sum(
                (
                    (
                        jp.minimum(
                            info["feet_air_time"],
                            self._config.gait.air_time_cap,
                        )
                        if self._config.gait.air_time_cap > 0
                        else info["feet_air_time"]
                    )
                    - 0.1
                )
                * first_contact
            )
            * moving
            * shape_gate,
            "feet_slip": slip * moving,
            # Flat-linkage proximity (see toggle_flat in default_config).
            "toggle_flat": toggle_ramp,
            # Per-swing apex: paid once per swing at touchdown, scaled by
            # how close the swing's peak clearance came to apex_target —
            # the term that prices lift height itself (high_step pays
            # duration-averaged clearance and tolerates skimming).
            "feet_apex": jp.sum(
                jp.clip(
                    info["swing_apex"] / self._config.reward.apex_target,
                    0.0,
                    1.0,
                )
                * first_contact
            )
            * moving
            * shape_gate
            * gate,
            # Downward foot speed^2, gated by proximity to the floor
            # (pre-contact, so hard strikes are read before the solver
            # absorbs them). Stance feet contribute ~0 (vz ~ 0); swing
            # above glide_height contributes 0 (gate).
            "feet_landing": jp.sum(
                jp.square(jp.clip(feet_vel[:, 2], None, 0.0))
                * jp.clip(
                    1.0 - foot_clearance / self._config.reward.glide_height,
                    0.0,
                    1.0,
                )
            )
            * moving
            * landing_soften,
            "feet_phase": jp.exp(-phase_err / self._config.reward.phase_sigma)
            * moving,
            "high_step": high_step * moving * shape_gate * gate,
            "contact_match": contact_match * moving,
            # Position pull to the anchor plus velocity damping: L1 position
            # alone caused bang-bang fidgeting around the anchor (v2).
            "stand_still": (
                jp.sum(jp.abs(data.qpos[self._qadr] - anchor))
                + 0.2 * jp.sum(jp.abs(qvel))
            )
            * (~moving),
            "stand_feet_down": jp.sum(jp.clip(foot_clearance, 0.0, None))
            * (~moving),
            "termination": fall.astype(jp.float32),
            "torque_limit": jp.sum(
                jp.maximum(
                    jp.abs(data.actuator_force)
                    - self._config.reward.torque_limit_frac * self._torque_cap,
                    0.0,
                )
            ),
        }
        return rewards, fall
