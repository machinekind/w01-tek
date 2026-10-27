"""real_pose_ref: rewards judge the kinematic REAL pose, not the set one."""

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from wojtek_rl import env as wojtek_env


def _cfg(real_pose_ref, kp=40.0, kd=1.6):
    cfg = wojtek_env.default_config()
    cfg.pd_kp = kp
    cfg.pd_kd = kd
    cfg.max_torque = 9.0
    cfg.real_pose_ref = real_pose_ref
    cfg.push.enable = False
    return cfg


@pytest.fixture(scope="module")
def env_real():
    return wojtek_env.WojtekJoystick(_cfg(True))


def test_legacy_flag_keeps_height_table(env_real):
    """real_pose_ref=False reproduces the static-table anchor exactly."""
    legacy = wojtek_env.WojtekJoystick(_cfg(False))
    h = jp.array(0.125)
    dsecond = jp.interp(
        h, jp.array(wojtek_env.HEIGHT_TABLE), jp.array(wojtek_env.DSECOND_TABLE)
    )
    expect = jp.clip(
        legacy._home_ctrl + jp.tile(jp.array([0.0, 1.0, 2.0]), 4) * dsecond,
        legacy._ctrlrange[:, 0],
        legacy._ctrlrange[:, 1],
    )
    np.testing.assert_allclose(np.array(legacy._height_ctrl(h)), np.array(expect))
    np.testing.assert_allclose(np.array(legacy._pose_ref(h)), np.array(expect))


def test_reference_is_gain_invariant(env_real):
    """The kinematic family must be a function of geometry alone: a soft
    kp20/kd1 env computes the identical reference."""
    soft = wojtek_env.WojtekJoystick(_cfg(True, kp=20.0, kd=1.0))
    np.testing.assert_allclose(
        np.array(soft._anchor_heights), np.array(env_real._anchor_heights),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(soft._anchor_poses), np.array(env_real._anchor_poses),
        atol=1e-5,
    )


def test_measured_heights_are_monotonic_and_plausible(env_real):
    h = np.array(env_real._anchor_heights)
    assert np.all(np.diff(h) > 0)
    assert h[0] < 0.11 < h[-1] and h[0] < 0.16 < h[-1]


def test_legacy_anchor_really_sags_on_stiff_plant(env_real):
    """Settle the ACTUAL kp40 plant on the ctrl anchor: the joints land
    away from the commanded targets (the sag this feature stops billing
    the policy for) yet close to the kinematic reference direction."""
    height = jp.array(0.125)
    ctrl = np.asarray(env_real._height_ctrl(height))
    ref = np.asarray(env_real._pose_ref(height))
    m = env_real.mj_model  # the real kp40 plant
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(env_real._home_qpos)
    d.ctrl[:] = ctrl
    for _ in range(int(round(2.0 / m.opt.timestep))):
        mujoco.mj_step(m, d)
    settled = d.qpos[np.asarray(env_real._qadr)]
    sag_vs_ctrl = np.abs(settled - ctrl).sum()
    err_vs_ref = np.abs(settled - ref).sum()
    assert sag_vs_ctrl > 0.15, (
        f"kp40 plant barely sags ({sag_vs_ctrl:.3f} rad); feature premise wrong?"
    )
    # the passive plant sags BELOW the kinematic pose too — the point is
    # the reward now asks the POLICY to close this gap actively, so the
    # reference only needs to be the true geometric target, not the
    # passive settle point. Sanity: same order of magnitude.
    assert err_vs_ref < 2 * sag_vs_ctrl


def test_reset_poses_on_reference(env_real):
    state = jax.jit(env_real.reset)(jax.random.PRNGKey(0))
    ref = np.asarray(env_real._pose_ref(state.info["command"][3]))
    q = np.asarray(state.data.qpos[np.asarray(env_real._qadr)])
    assert np.abs(q - ref).max() <= 0.05 + 1e-6
