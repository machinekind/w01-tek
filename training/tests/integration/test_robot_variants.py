"""The built model and the env of every robot variant other than the stock
one. The stock robot has test_build_model.py, test_keyframe.py, test_env.py.
"""

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from wojtek_rl import build_model, paths, robots
from wojtek_rl.registry import make_env

VARIANTS = [n for n in robots.NAMES if n != paths.DEFAULT_ROBOT]


def _scene(robot):
    return mujoco.MjModel.from_xml_path(str(paths.robot_files(robot)["scene"]))


def _env_overrides(robot):
    rb = robots.get(robot)
    mid = rb.height_table[len(rb.height_table) // 2]
    return {
        "robot": robot,
        "sim_dt": rb.timestep,
        "command": {"height": [mid, rb.height_table[-2]]},
        "sim": {"backend": "jax"},
    }


@pytest.mark.parametrize("robot", VARIANTS)
def test_same_skeleton_as_the_stock_robot(robot):
    m = build_model.build_spec(robot=robot).compile()
    stock = build_model.build_spec().compile()
    assert (m.nq, m.nv, m.nu, m.neq) == (stock.nq, stock.nv, stock.nu, stock.neq)
    for kind, count in (("actuator", m.nu), ("joint", m.njnt), ("sensor", m.nsensor)):
        names = [getattr(m, kind)(i).name for i in range(count)]
        assert names == [getattr(stock, kind)(i).name for i in range(count)], kind


@pytest.mark.parametrize("robot", VARIANTS)
def test_built_files_are_current(robot):
    # The generated XML is committed; it must be what the builder produces
    # from today's source model and robots.py.
    fresh = build_model.build_spec(robot=robot).compile()
    built = mujoco.MjModel.from_xml_path(str(paths.robot_files(robot)["robot"]))
    for field in (
        "body_mass", "body_inertia", "body_pos", "geom_size", "geom_pos",
        "geom_contype", "geom_conaffinity", "jnt_range", "dof_frictionloss",
        "actuator_gainprm", "actuator_biasprm", "actuator_forcerange",
        "actuator_ctrlrange", "eq_solimp",
    ):
        assert np.allclose(getattr(fresh, field), getattr(built, field), atol=1e-6), field
    assert fresh.opt.timestep == built.opt.timestep == robots.get(robot).timestep
    assert fresh.opt.iterations == built.opt.iterations


@pytest.mark.parametrize("robot", VARIANTS)
def test_only_primitives_collide_and_feet_sit_on_the_pads(robot):
    rb = robots.get(robot)
    m = _scene(robot)
    d = mujoco.MjData(m)
    d.qpos[:] = m.key("home").qpos
    mujoco.mj_forward(m, d)
    for g in range(m.ngeom):
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            assert m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
    for leg in paths.LEGS:
        foot = m.geom(f"{leg}_foot_sphere")
        assert m.body(foot.bodyid[0]).name == f"{leg}_{rb.foot_body}"
        assert foot.size[0] == rb.foot_radius
        # Standing: the sphere rests on the floor, under its own hip. The
        # settled legs sag, which carries the foot about 3 cm off the hip's
        # plumb line; a sphere placed in the wrong body's frame is 20 cm off.
        assert abs(d.geom_xpos[foot.id][2] - rb.foot_radius) < 5e-3
        hip = d.xanchor[m.joint(f"{leg}_second_joint").id]
        assert abs(d.geom_xpos[foot.id][0] - hip[0]) < 0.06


@pytest.mark.parametrize("robot", VARIANTS)
def test_leg_primitives_are_mirrored_left_to_right(robot):
    # A right leg built with a left leg's capsule pokes 17 cm sideways into
    # the floor and rolls the robot over within a second.
    m = _scene(robot)
    d = mujoco.MjData(m)
    d.qpos[:] = m.key("home").qpos
    mujoco.mj_forward(m, d)
    for left, right in (("rear_left", "rear_right"), ("front_left", "front_right")):
        for link in robots.get(robot).leg_collision:
            a = d.geom_xpos[m.geom(f"{left}_{link}_floor").id]
            b = d.geom_xpos[m.geom(f"{right}_{link}_floor").id]
            assert np.allclose(a * [1, -1, 1], b, atol=2e-3), link


@pytest.mark.parametrize("robot", VARIANTS)
def test_stands_and_keeps_its_loops_closed(robot):
    m = _scene(robot)
    d = mujoco.MjData(m)
    key = m.key("home")
    d.qpos[:] = key.qpos
    d.ctrl[:] = key.ctrl
    z_home = key.qpos[2]
    rng = np.random.default_rng(0)
    worst = 0.0
    a = [m.site(f"{leg}_chain_close_a").id for leg in paths.LEGS]
    b = [m.site(f"{leg}_chain_close_b").id for leg in paths.LEGS]
    steps_per_ctrl = int(round(0.02 / m.opt.timestep))
    for i in range(int(4.0 / m.opt.timestep)):
        if d.time > 2.0 and i % steps_per_ctrl == 0:
            # what an untrained policy does: new random targets at 50 Hz
            d.ctrl[:] = np.clip(
                key.ctrl + rng.uniform(-0.3, 0.3, m.nu),
                m.actuator_ctrlrange[:, 0],
                m.actuator_ctrlrange[:, 1],
            )
        mujoco.mj_step(m, d)
        if abs(d.time - 2.0) < m.opt.timestep / 2:
            assert abs(d.qpos[2] - z_home) < 0.01, "home keyframe is not at rest"
            # The pose reward pulls a policy toward this stance, so a stance
            # that leans teaches every gait to lean.
            down = d.xmat[m.body("root").id].reshape(3, 3).T @ [0.0, 0.0, -1.0]
            assert abs(np.degrees(np.arcsin(down[0]))) < 0.3, "stands pitched"
        worst = max(worst, np.linalg.norm(d.site_xpos[a] - d.site_xpos[b], axis=1).max())
    assert d.qpos[2] > 0.8 * z_home, "fell under random targets"
    assert worst < 2e-3, f"four-bar loop opened by {worst * 1e3:.1f} mm"


@pytest.mark.parametrize("robot", VARIANTS)
def test_a_viewer_opens_it_standing(robot):
    # What a viewer does: reset the model, leave the controls at zero, run.
    if not robots.get(robot).rest_pose_from_source_key:
        pytest.skip("rest pose is the source model's")
    m = mujoco.MjModel.from_xml_path(str(paths.robot_files(robot)["view_scene"]))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    a = [m.site(f"{leg}_chain_close_a").id for leg in paths.LEGS]
    b = [m.site(f"{leg}_chain_close_b").id for leg in paths.LEGS]
    assert np.linalg.norm(d.site_xpos[a] - d.site_xpos[b], axis=1).max() < 1e-4
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert abs(d.qpos[2] - m.key("home").qpos[2]) < 0.01
    # and the training model means the same thing by every joint value
    train = _scene(robot)
    assert np.allclose(train.qpos0, m.qpos0)
    assert np.allclose(train.jnt_range, m.jnt_range)


@pytest.mark.parametrize("robot", VARIANTS)
def test_height_command_moves_the_base_straight_up(robot):
    env = make_env("joystick", {**_env_overrides(robot), "real_pose_ref": True})
    heights = np.asarray(env._anchor_heights)
    assert np.all(np.diff(heights) > 0)
    assert len(heights) == len(robots.get(robot).height_table)
    # Every leg extends by the same amount, whatever its mounting sign. The
    # comparison is on the move away from the stand pose, because front and
    # rear stand poses may differ by the offset that levels the body.
    ctrl = np.asarray(env._height_ctrl(float(heights[1])))
    sign = np.repeat(robots.get(robot).leg_sign, 3)
    home = np.asarray(robots.get(robot).stand_pose, dtype=float).ravel()
    per_leg = ((ctrl - home) * sign).reshape(4, 3)
    assert np.allclose(per_leg, per_leg[0], atol=1e-5)  # float32 targets


@pytest.mark.parametrize("robot", VARIANTS)
def test_joystick_env_steps(robot):
    env = make_env("joystick", _env_overrides(robot))
    assert env.n_substeps == int(round(0.02 / robots.get(robot).timestep))
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert state.obs["state"].shape == (54,)  # same layout as the stock robot
    step = jax.jit(env.step)
    for _ in range(25):
        state = step(state, jp.zeros(env.action_size))
    assert bool(jp.isfinite(state.data.qpos).all())
    assert not bool(state.done)
    assert bool(jp.all(env._foot_contact(state.data)))


@pytest.mark.parametrize("robot", VARIANTS)
def test_what_does_not_fit_the_variant_is_refused(robot):
    ov = _env_overrides(robot)
    with pytest.raises(ValueError, match="does not run on robot"):
        make_env("getup", {"robot": robot})
    with pytest.raises(ValueError, match="coarser"):
        make_env("joystick", {**ov, "sim_dt": 0.004})
    with pytest.raises(ValueError, match="outside the standing heights"):
        make_env("joystick", {**ov, "command": {"height": [0.09, 0.17]}})
    with pytest.raises(ValueError, match="toggle"):
        make_env("joystick", {**ov, "fall": {"max_toggle_deg": 150.0}})
    with pytest.raises(ValueError, match="no arena"):
        make_env("joystick", {**ov, "terrain": {"enable": True}})
