"""One (course, seed) rollout in the measurement env, and its step budget."""

import math

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from wojtek_rl.courses.follower import Pursuit
from wojtek_rl.courses.spec import HEIGHT_CMD, PUSH_VEL, SpinCourse
from wojtek_rl.navigation import _wrap, quat_to_yaw

STAND_STEPS = 50  # 1 s standing at HEIGHT_CMD before the course starts, so
# the reset transient (reset poses to a randomly sampled
# height command) is not charged to the course
TIME_FACTOR = 2.5  # step budget = TIME_FACTOR x ideal time + 2 s
MAX_COURSE_STEPS = 6000  # hard ceiling (120 s at ctrl_dt=0.02)


def friction_geom_ids(floor_id: int, foot_geom_ids) -> np.ndarray:
    """The geoms whose sliding friction sets a course's contact friction.

    Both the floor AND the feet. MuJoCo resolves an equal-priority contact by
    taking the ELEMENT-WISE MAX of the two geoms' friction, and the feet only
    get priority 1 under foot-friction domain randomization (randomize.py) --
    not here. Both start at 0.9 in this model, so lowering the floor alone is
    a silent no-op: measured, mu_floor 0.9 -> 0.4 left the foot-slip distance
    bit-identical at 5.314 m, because max(0.4, 0.9) is still 0.9. Setting
    every one of these geoms instead takes slip 5.314 -> 20.590 m on the same
    rollout, which is what makes the slippery scenarios mean anything.
    """
    return np.unique(np.concatenate([[int(floor_id)], np.asarray(foot_geom_ids).ravel()]))


def step_budget(course, dt: float) -> int:
    """Steps allowed for a scenario: TIME_FACTOR x ideal time, plus 2 s.

    Ideal time is what a perfect tracker would need: path length over
    commanded speed for a Course (summed per segment), commanded rotation
    over commanded rate for a SpinCourse. The factor covers the yaw-in-place
    time that corners and reversals genuinely cost.
    """
    if isinstance(course, SpinCourse):
        ideal = course.turns * 2.0 * math.pi / abs(course.wz)
    else:
        wp = np.asarray(course.waypoints, dtype=float)
        seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
        ideal = float((seg / course.segment_speeds).sum())
    return min(MAX_COURSE_STEPS, int(round((TIME_FACTOR * ideal + 2.0) / dt)))


def _hold_command(state, cmd):
    """Pin the follower's command into the env state for the next step.

    Also zeroes steps_since_cmd so the env's own command resampling never
    fires: step() replaces info["command"] with a fresh random sample once
    the counter reaches command.resample_steps (250 = 5 s) and then builds
    the observation from it -- the policy would be handed a random command
    instead of ours, and lurch off the task.
    """
    state.info["command"] = cmd
    state.info["steps_since_cmd"] = jp.zeros_like(state.info["steps_since_cmd"])


def _settle(env, reset, step, inf, seed):
    """Reset, then stand at HEIGHT_CMD for STAND_STEPS.

    Reset poses the robot for a randomly sampled height command, and that
    transient is not the scenario's fault. Returns (rng, state), with state
    None if the policy fell over standing still -- nothing to measure then.
    """
    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    stand_cmd = jp.array([0.0, 0.0, 0.0, HEIGHT_CMD])
    for _ in range(STAND_STEPS):
        _hold_command(state, stand_cmd)
        rng, k = jax.random.split(rng)
        act, _ = inf(state.obs, k)
        state = step(state, act)
        if float(state.done):
            return rng, None
    return rng, state


_SETTLE_FALL_INFO = {"fell_at": 0, "completed": False, "steps": 0, "frames": []}


def course_rollout(env, reset, step, inf, course, foot_radius, seed=0, renderer=None):
    """Walk `course` once under `inf`; record what the score needs.

    `reset`/`step` are the jitted env fns (passed in so the caller compiles
    them once per friction setting and reuses them). Structure mirrors
    battery.rollout, with the command coming from `Pursuit` instead of a
    step-index function.

    Returns (rec, info) where `rec` maps signal -> np.ndarray over the
    course steps (the standing settle excluded) and `info` carries
    `fell_at`, `completed`, `steps` and, when `renderer` is given, the
    rendered `frames`.
    """
    rng, state = _settle(env, reset, step, inf, seed)
    if state is None:
        return {}, dict(_SETTLE_FALL_INFO)

    q = np.asarray(state.data.qpos)
    pursuit = Pursuit.from_course(course, float(q[0]), float(q[1]), quat_to_yaw(*q[3:7]))
    n_steps = step_budget(course, env.dt)

    mj_model = env.mj_model
    render_data = mujoco.MjData(mj_model) if renderer is not None else None
    render_every = max(1, round(1 / (30 * env.dt)))

    rec = {
        "xy": [], "s": [], "xte": [], "cmd_v": [], "v_fwd": [], "v_planar": [],
        "h": [], "qvel": [], "slip_speed": [],
    }
    fell_at, completed, pushed = None, False, course.push_at_m is None
    frames = []
    for i in range(n_steps):
        q = np.asarray(state.data.qpos)
        x, y, yaw = float(q[0]), float(q[1]), quat_to_yaw(*q[3:7])
        cmd, xte, s, reached = pursuit.command(x, y, yaw)
        if reached:
            completed = True
            break

        # One deterministic lateral impulse, the moment the course passes
        # push_at_m. env's own random push is disabled by
        # load_checkpoint_policy, so this is the only disturbance.
        if not pushed and s >= course.push_at_m:
            pushed = True
            # Left of the current heading: a lateral shove is what actually
            # threatens a path-follower, and its sign is fixed so the two
            # push rows differ from each other only in commanded speed.
            kick = jp.array([-math.sin(yaw), math.cos(yaw)]) * PUSH_VEL
            state = state.replace(
                data=state.data.replace(qvel=state.data.qvel.at[:2].add(kick))
            )

        _hold_command(state, cmd)
        rng, k = jax.random.split(rng)
        act, _ = inf(state.obs, k)
        state = step(state, act)
        d = state.data
        if float(state.done):
            fell_at = i
            break

        fv = np.asarray(d.sensordata)[np.asarray(env._foot_linvel_adr)]
        gx = np.asarray(d.geom_xpos)[env._foot_geom_ids]
        contact = gx[:, 2] < foot_radius + 0.005
        planar = np.asarray(d.qvel[:2])
        rec["xy"].append([float(d.qpos[0]), float(d.qpos[1])])
        rec["s"].append(s)
        rec["xte"].append(xte)
        rec["cmd_v"].append(float(cmd[0]))
        rec["v_fwd"].append(float(np.asarray(env._local_linvel(d))[0]))
        rec["v_planar"].append(float(np.linalg.norm(planar)))
        rec["h"].append(float(d.qpos[2]))
        rec["qvel"].append(np.asarray(d.qvel[env._vadr]))
        # Slip speed: how fast the feet that are ON THE GROUND are sliding.
        rec["slip_speed"].append(
            float((np.linalg.norm(fv[:, :2], axis=-1) * contact).sum())
        )
        if renderer is not None and i % render_every == 0:
            render_data.qpos[:] = np.asarray(d.qpos)
            mujoco.mj_forward(mj_model, render_data)
            renderer.update_scene(render_data, camera="track")
            frames.append(renderer.render())

    return {k: np.asarray(v) for k, v in rec.items()}, {
        "fell_at": fell_at,
        "completed": completed,
        "steps": len(rec["s"]),
        "frames": frames,
        "path": pursuit.pts,
        "total_length": pursuit.total_length,
    }


def spin_rollout(env, reset, step, inf, spin: SpinCourse, seed=0, renderer=None):
    """Hold `spin`'s pure-spin command until the rotation completes.

    No follower: the command is [0, 0, wz, HEIGHT_CMD] every step. Progress
    is the yaw accumulated in the COMMANDED direction (per-step wrapped
    deltas summed, so multi-turn spins don't alias); completion is
    `spin.turns` full rotations inside the step budget.

    Returns (rec, info) shaped like course_rollout's, with `rec` carrying
    the spin signals (yaw_progress, wz measured vs commanded, planar drift
    from the start point) and `total_length` the required rotation in rad.
    """
    rng, state = _settle(env, reset, step, inf, seed)
    if state is None:
        return {}, dict(_SETTLE_FALL_INFO)

    q = np.asarray(state.data.qpos)
    start_xy = q[:2].copy()
    prev_yaw = quat_to_yaw(*q[3:7])
    required = spin.turns * 2.0 * math.pi
    direction = math.copysign(1.0, spin.wz)
    cmd = jp.array([0.0, 0.0, spin.wz, HEIGHT_CMD])
    n_steps = step_budget(spin, env.dt)

    mj_model = env.mj_model
    render_data = mujoco.MjData(mj_model) if renderer is not None else None
    render_every = max(1, round(1 / (30 * env.dt)))

    rec = {"yaw_progress": [], "wz": [], "drift": [], "h": [], "qvel": []}
    fell_at, completed, progress = None, False, 0.0
    frames = []
    for i in range(n_steps):
        _hold_command(state, cmd)
        rng, k = jax.random.split(rng)
        act, _ = inf(state.obs, k)
        state = step(state, act)
        d = state.data
        if float(state.done):
            fell_at = i
            break

        q = np.asarray(d.qpos)
        yaw = quat_to_yaw(*q[3:7])
        # Signed progress toward the commanded direction; backwards wobble
        # subtracts, so a robot oscillating in place never "completes".
        progress += direction * _wrap(yaw - prev_yaw)
        prev_yaw = yaw
        rec["yaw_progress"].append(progress)
        rec["wz"].append(float(np.asarray(env._gyro(d))[2]))
        rec["drift"].append(float(np.hypot(*(q[:2] - start_xy))))
        rec["h"].append(float(q[2]))
        rec["qvel"].append(np.asarray(d.qvel[env._vadr]))
        if renderer is not None and i % render_every == 0:
            render_data.qpos[:] = np.asarray(q)
            mujoco.mj_forward(mj_model, render_data)
            renderer.update_scene(render_data, camera="track")
            frames.append(renderer.render())
        if progress >= required:
            completed = True
            break

    return {k: np.asarray(v) for k, v in rec.items()}, {
        "fell_at": fell_at,
        "completed": completed,
        "steps": len(rec["h"]),
        "frames": frames,
        "total_length": required,
    }
