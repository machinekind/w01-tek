"""Build the deployment contract (policy_meta.json, schema 2) from a live env.

The contract is everything the ROS runtime needs to run a policy, as
RESOLVED numbers -- final 12-vectors computed by the same env instance that
defined training, never recipes for the deploy side to re-derive. The
runtime (ros/src/wojtek_policy/wojtek_policy/policy.py) is a plain
interpreter of this file:

  obs = concat(obs_layout components)
  filt = action_filter * filt + (1 - action_filter) * tanh_mlp(obs)
  motor_targets = clip(anchor_ctrl + filt * action_scale,
                       target_low, target_high)

The one exception is the live standing height: when the command box trains
a real height range (4th dim, low < high), the stance anchor is a function
of the commanded height and cannot be a single resolved vector. The
contract then ships ctrl_low/ctrl_high and the runtime re-anchors with its
copy of the env's measured height table (policy.py height_anchor).

Every env-config key must be classified below as either CONSUMED (it shapes
a contract field) or TRAINING_ONLY (provably irrelevant to deployment).
build_contract() refuses to export when it meets a key it does not know --
that error is the point: a new env option with deploy implications must be
added here (and, if needed, to the runtime) before a policy trained with it
can ship.
"""

import numpy as np

SCHEMA_VERSION = 2

# Past the knee (third-joint) singularity the four-bar can snap through and
# break the linkage; the runtime's clamp_knee safety option clips there.
KNEE_SINGULARITY = 3.2

# Keys whose values become contract fields (directly or via the env's
# resolved state: model customization, target bounds, anchor).
CONSUMED_KEYS = {
    "ctrl_dt",
    "action_scale",
    "action_filter",
    "abduction_ctrl_limit",
    "knee_target_max",
    "max_torque",
    "pd_kp",
    "pd_kd",
    "obs",
    "command",
    # Changes the env's height->anchor mapping (kinematic calibrated table
    # instead of the legacy static one); consumed via the exported
    # "height_table" field, which the runtime's height_anchor() prefers.
    "real_pose_ref",
}

# Keys that shape training only: physics stepping, randomization, episode
# structure, rewards. Nothing here changes what the robot must do with the
# exported network at 50 Hz. Latency/encoder/action_delay model real-world
# imperfections during training; the real robot has the real thing.
TRAINING_ONLY_KEYS = {
    "sim",
    "sim_dt",
    "episode_length",
    "obs_noise",
    "push",
    "latency",
    "encoder",
    "action_delay",
    "fall",
    "gait",
    "reward",
    # Mirror augmentation: env-internal frame flip during training only;
    # the deployed policy always receives real observations.
    "symmetry",
    # Terrain shapes the training scene, spawn, and curriculum only; it leaves
    # the observation layout unchanged (no world position/heading reaches the
    # actor), so nothing about it changes what the robot does with the policy.
    "terrain",
    # CaT-style early termination for episodes that ignore their command: a
    # PPO training signal (future value zero), no reward term, nothing in the
    # control loop. The robot has no episodes to cut.
    "no_progress",
}

# Sub-keys of `command` that define the trained command box (contract) vs
# the sampling curriculum (training only).
COMMAND_BOX_KEYS = ("vx", "vy", "wz", "height")
COMMAND_TRAINING_KEYS = {
    "resample_steps",
    "zero_prob",
    "pure_wz_prob",
    "pure_vy_prob",
    # Arc curriculum: redraws vx within the trained box for a fraction of
    # samples; changes what is practiced, not the command box itself.
    "arc_prob",
    "arc_vx",
    # Slow/fast clean-walk curricula: same family as arc — practice
    # emphasis within the trained box (see product-tracking coverage
    # lesson, 2026-07-27), no contract impact.
    "pure_slow_prob",
    "slow_vx",
    "pure_fast_prob",
    "fast_vx",
    # Backward curriculum (terrain family, run three): clean backward
    # commands drawn within the trained box's negative vx range.
    "pure_back_prob",
    "back_vx",
}


def check_config_covered(env_config: dict) -> None:
    """Fail export when env_config has a key this module has not classified."""
    unknown = set(env_config) - CONSUMED_KEYS - TRAINING_ONLY_KEYS
    if unknown:
        raise ValueError(
            f"env config key(s) {sorted(unknown)} are not classified in "
            "wojtek_rl/deploy_contract.py -- decide whether they affect "
            "deployment (add a contract field + runtime support) or not "
            "(add to TRAINING_ONLY_KEYS) before exporting"
        )
    cmd = env_config.get("command", {})
    unknown_cmd = set(cmd) - set(COMMAND_BOX_KEYS) - COMMAND_TRAINING_KEYS
    if unknown_cmd:
        raise ValueError(
            f"command config key(s) {sorted(unknown_cmd)} are not classified "
            "in wojtek_rl/deploy_contract.py (COMMAND_BOX_KEYS / "
            "COMMAND_TRAINING_KEYS)"
        )


def build_contract(env, run: dict, checkpoint: str = "") -> dict:
    """Schema-2 policy_meta dict for a WojtekJoystick env instance.

    `env` must be built with the run's env_config so its resolved state
    (customized model, target bounds, anchor) IS the training-time state.
    `run` is the parsed run.json (run_name, task, env_config).
    """
    import jax

    env_config = run.get("env_config", {})
    check_config_covered(env_config)

    task = run.get("task", "joystick")
    if task != "joystick":
        raise NotImplementedError(
            f"deploy contract is defined for the joystick task only, got "
            f"'{task}' -- the ROS runtime has no target pipeline for it"
        )

    # One reset gives a real catalog to read component widths from; no
    # physics stepping beyond mjx.forward.
    state = env.reset(jax.random.PRNGKey(0))
    catalog = env._obs_catalog(state.data, state.info)
    names = env.actor_obs_names
    layout = [(n, int(np.asarray(catalog[n]).shape[0])) for n in names]

    unsupported = [n for n, _ in layout if n == "phase"]
    if unsupported:
        raise NotImplementedError(
            "this policy observes the gait clock ('phase'); the deploy "
            "runtime has no faithful implementation of the speed-blended "
            "walk/trot clock -- extend the contract and the runtime before "
            "exporting a phase-observing policy"
        )

    c = env._config.command
    box = [tuple(getattr(c, k)) for k in COMMAND_BOX_KEYS]
    command_low = [float(lo) for lo, _ in box]
    command_high = [float(hi) for _, hi in box]
    cmd_width = dict(layout).get("command", 0)
    if cmd_width != len(box):
        raise ValueError(
            f"command obs width {cmd_width} != command box dims {len(box)} "
            "-- COMMAND_BOX_KEYS is out of sync with _sample_command"
        )
    # Dims beyond (vx, vy, wz) have no /cmd_vel source on the robot; the
    # runtime fills them with the training box midpoint (for a pinned
    # height range, that IS the trained height).
    command_fill = [
        (lo + hi) / 2.0 for lo, hi in box[3:]
    ]

    fill_height = command_fill[0]
    anchor = np.asarray(env._height_ctrl(fill_height), np.float32)

    # The height->dsecond mapping the env actually anchored with, so the
    # runtime re-anchors live-height commands with the SAME table. With
    # real_pose_ref the env measures a kinematic table at construction
    # (gains-invariant, truncated at the reachable peak) that differs from
    # the legacy static HEIGHT_TABLE by the calibration sag — a runtime
    # holding only the legacy copy would compute wrong anchors for such
    # policies (policy.py height_anchor prefers this field when present).
    if env._config.get("real_pose_ref", False):
        height_table = {
            "heights": np.asarray(env._anchor_heights, np.float32).tolist(),
            "dsecond": np.asarray(env._anchor_dsecond, np.float32).tolist(),
        }
    else:
        from wojtek_rl.env import DSECOND_TABLE, HEIGHT_TABLE

        height_table = {
            "heights": list(HEIGHT_TABLE),
            "dsecond": list(DSECOND_TABLE),
        }

    scale = np.asarray(env._config.action_scale, np.float32)
    if scale.ndim == 0:
        scale = np.full(12, float(scale), np.float32)
    elif scale.size == 3:
        scale = np.tile(scale, 4)

    m = env.mj_model
    kp = float(m.actuator_gainprm[0, 0])
    kd = float(-m.actuator_biasprm[0, 2])

    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": run["run_name"],
        "checkpoint": str(checkpoint),
        "task": task,
        "obs_size": int(sum(w for _, w in layout)),
        "action_size": int(env.action_size),
        "obs_layout": [f"{n}:{w}" for n, w in layout],
        "actuator_names": [m.actuator(i).name for i in range(m.nu)],
        "home_ctrl": np.asarray(env._home_ctrl, np.float32).tolist(),
        "anchor_ctrl": anchor.tolist(),
        "action_scale": scale.tolist(),
        "target_low": np.asarray(env._target_lo, np.float32).tolist(),
        "target_high": np.asarray(env._target_hi, np.float32).tolist(),
        # The customized model's ctrlrange (what _height_ctrl clips to).
        # Live-height contracts need it: the runtime re-derives the stance
        # anchor from the commanded height and must clip exactly as the
        # training env did -- to this range, not the narrower target bounds.
        "ctrl_low": np.asarray(env._ctrlrange[:, 0], np.float32).tolist(),
        "ctrl_high": np.asarray(env._ctrlrange[:, 1], np.float32).tolist(),
        "command_low": command_low,
        "command_high": command_high,
        "command_fill": [float(v) for v in command_fill],
        "height_table": height_table,
        "action_filter": float(env._config.action_filter),
        "ctrl_dt": float(env._config.ctrl_dt),
        "knee_singularity": KNEE_SINGULARITY,
        # Informational: the PD servo config the policy trained against.
        # The real driver (wojtek_real.urdf.xacro kp/kd, launch max_torque)
        # must match; the policy node logs these at load.
        "pd": {
            "kp": kp,
            "kd": kd,
            "max_torque": float(m.actuator_forcerange[0, 1]),
        },
    }
