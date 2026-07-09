import mujoco
import numpy as np

from fbb_rl import paths


def _load():
    return mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))


def test_home_keyframe_exists():
    m = _load()
    assert m.nkey >= 1
    assert m.key("home").qpos.shape == (31,)


def test_stands_for_five_seconds_under_pd_hold():
    m = _load()
    d = mujoco.MjData(m)
    key = m.key("home")
    d.qpos[:] = key.qpos
    d.ctrl[:] = key.ctrl
    mujoco.mj_forward(m, d)
    # Design standing height is ~0.10 m: the controller's gait code puts feet
    # 0.15 below the hip frame and hips mount 0.06 above base center.
    z_home = d.qpos[2]
    assert z_home > 0.08, "home pose too low to be standing"
    for _ in range(int(5.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert d.qpos[2] > 0.8 * z_home, "robot sank or fell under PD hold"
    # body z axis still points up
    zz = 1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2)  # R[2,2] from quat wxyz
    assert zz > 0.9, "robot tipped over"


def test_loop_closure_error_small_after_settle():
    m = _load()
    d = mujoco.MjData(m)
    key = m.key("home")
    d.qpos[:] = key.qpos
    d.ctrl[:] = key.ctrl
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    for leg in paths.LEGS:
        b1 = d.body(f"{leg}_foot_link").xpos
        b2 = d.body(f"{leg}_chain_close_a_link").xpos
        assert np.linalg.norm(b1 - b2) < 2e-3, f"{leg} four-bar loop drifted open"
