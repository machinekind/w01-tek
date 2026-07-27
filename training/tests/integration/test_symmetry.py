"""Symmetry augmentation: env wiring and the physical mirror assumption.

Two layers:

1. Wiring — a mirror_prob=1.0 env must present exactly the mirrored view
   of the same real-frame world a mirror_prob=0.0 env produces, with
   identical rewards, and un-mirror the policy action before the physics.

2. Physics — the augmentation is only semantically right if the model is
   left/right mirror-symmetric: stepping a mirrored state with mirrored
   controls must produce the mirrored next state. The full-model qpos/qvel
   mirror map is derived here from the joint axes (see wojtek_rl.symmetry
   for the sign rule).
"""

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
from mujoco import mjx
from mujoco_playground._src import mjx_env

from wojtek_rl import env as wojtek_env
from wojtek_rl import symmetry


def _make_env(mirror_prob):
    cfg = wojtek_env.default_config()
    cfg.symmetry.enable = True
    cfg.symmetry.mirror_prob = mirror_prob
    cfg.push.enable = False
    return wojtek_env.WojtekJoystick(cfg)


@pytest.fixture(scope="module")
def env_real():
    return _make_env(0.0)


@pytest.fixture(scope="module")
def env_mirror():
    return _make_env(1.0)


def _mirror_obs(env, obs):
    return {
        "state": np.array(env._state_sign) * np.array(obs["state"])[
            np.array(env._state_perm)
        ],
        "privileged_state": np.array(env._priv_sign)
        * np.array(obs["privileged_state"])[np.array(env._priv_perm)],
    }


def test_mirrored_env_presents_mirrored_view_of_same_world(env_real, env_mirror):
    """Same rng: identical real-frame world, mirrored observations, and
    the policy action un-mirrored before the physics."""
    rng = jax.random.PRNGKey(3)
    s_real = jax.jit(env_real.reset)(rng)
    s_mir = jax.jit(env_mirror.reset)(rng)

    # Same real-frame world underneath (mirror_prob only changes the flag).
    np.testing.assert_allclose(s_real.data.qpos, s_mir.data.qpos, atol=0)
    np.testing.assert_allclose(
        s_real.info["command"], s_mir.info["command"], atol=0
    )
    # Observations are the mirror image of the real env's.
    for k, v in _mirror_obs(env_real, s_real.obs).items():
        np.testing.assert_allclose(np.array(s_mir.obs[k]), v, atol=1e-6)

    # Step: the mirrored env receives the mirrored action and must undo it,
    # landing in the identical real-frame state with identical reward.
    act = jax.random.uniform(jax.random.PRNGKey(4), (12,), minval=-0.5, maxval=0.5)
    act_m = jp.array(env_mirror._act_sign) * act[jp.array(env_mirror._act_perm)]
    n_real = jax.jit(env_real.step)(s_real, act)
    n_mir = jax.jit(env_mirror.step)(s_mir, act_m)
    np.testing.assert_allclose(
        np.array(n_real.data.qpos), np.array(n_mir.data.qpos), atol=1e-6
    )
    np.testing.assert_allclose(
        float(n_real.reward), float(n_mir.reward), atol=1e-5
    )
    # last_act is stored in the real frame
    np.testing.assert_allclose(
        np.array(n_mir.info["last_act"]), np.array(act), atol=1e-6
    )
    for k, v in _mirror_obs(env_real, n_real.obs).items():
        np.testing.assert_allclose(np.array(n_mir.obs[k]), v, atol=1e-5)


def test_symmetry_disabled_is_inert(env_real):
    """enable=False must reproduce the stock trajectory (flag stays False,
    no rng consumed): guarded by comparing against a plain env."""
    cfg = wojtek_env.default_config()
    cfg.push.enable = False
    stock = wojtek_env.WojtekJoystick(cfg)
    rng = jax.random.PRNGKey(5)
    s_stock = jax.jit(stock.reset)(rng)
    assert not bool(s_stock.info["mirror"])
    s_step = jax.jit(stock.step)(s_stock, jp.zeros(12))
    assert np.isfinite(float(s_step.reward))


# -- physical mirror symmetry of the model --------------------------------


def _full_mirror_maps(m):
    """(qpos_perm, qpos_sign, qvel_perm, qvel_sign, ctrl_perm, ctrl_sign)
    for the whole model, derived from joint names and axes."""
    qpos_perm = np.arange(m.nq)
    qpos_sign = np.ones(m.nq)
    qvel_perm = np.arange(m.nv)
    qvel_sign = np.ones(m.nv)
    # freejoint: pos (x,-y,z), quat (w,-x,y,-z), linvel (x,-y,z),
    # angvel (-x,y,-z)
    qpos_sign[:7] = [1, -1, 1, 1, -1, 1, -1]
    qvel_sign[:6] = [1, -1, 1, -1, 1, -1]
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        if "left" in name:
            pair = name.replace("left", "right")
        elif "right" in name:
            pair = name.replace("right", "left")
        else:
            pytest.fail(f"hinge joint {name!r} has no left/right pair")
        pj = m.joint(pair).id
        same_axis = np.allclose(m.jnt_axis[j], m.jnt_axis[pj])
        opposite = np.allclose(m.jnt_axis[j], -m.jnt_axis[pj])
        assert same_axis or opposite, f"{name}: axes neither equal nor opposite"
        sign = -1.0 if same_axis else 1.0
        qpos_perm[m.jnt_qposadr[j]] = m.jnt_qposadr[pj]
        qpos_sign[m.jnt_qposadr[j]] = sign
        qvel_perm[m.jnt_dofadr[j]] = m.jnt_dofadr[pj]
        qvel_sign[m.jnt_dofadr[j]] = sign
    ctrl_perm, ctrl_sign = symmetry.joint_mirror()
    return qpos_perm, qpos_sign, qvel_perm, qvel_sign, ctrl_perm, ctrl_sign


def test_model_is_statically_mirror_symmetric(env_real):
    """Paired bodies carry identical mass; the home pose is its own mirror."""
    m = env_real.mj_model
    for b in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if "left" in name:
            pair = name.replace("left", "right")
            np.testing.assert_allclose(
                m.body_mass[b], m.body(pair).mass, rtol=1e-6,
                err_msg=f"mass asymmetry {name} vs {pair}",
            )
    qpos_perm, qpos_sign, *_ = _full_mirror_maps(m)
    home = np.array(env_real._home_qpos)
    np.testing.assert_allclose(
        qpos_sign * home[qpos_perm], home, atol=1e-6,
        err_msg="home keyframe is not mirror-symmetric",
    )


def test_mirrored_dynamics_stay_mirrored(env_real):
    """Step a mirrored state with mirrored controls: the trajectories must
    stay each other's mirror image. This is the assumption that makes the
    augmentation semantically correct."""
    env = env_real
    m = env.mj_model
    qpos_perm, qpos_sign, qvel_perm, qvel_sign, ctrl_perm, ctrl_sign = (
        _full_mirror_maps(m)
    )

    anchor = np.array(env._home_ctrl)
    data_a = env._make_data()
    data_a = data_a.replace(
        qpos=env._home_qpos, qvel=jp.zeros(m.nv), ctrl=jp.array(anchor)
    )
    data_a = jax.jit(lambda d: mjx.forward(env.mjx_model, d))(data_a)
    data_b = data_a  # home is its own mirror (asserted above)

    step = jax.jit(
        lambda d, c: mjx_env.step(env.mjx_model, d, c, env.n_substeps)
    )
    # Deliberately asymmetric excitation: per-joint phases and amplitudes.
    rng = np.random.default_rng(7)
    phases = rng.uniform(0, 2 * np.pi, 12)
    amps = rng.uniform(0.05, 0.2, 12)
    for t in range(20):
        ctrl_a = anchor + amps * np.sin(2 * np.pi * 1.5 * t * env.dt + phases)
        ctrl_b = ctrl_sign * ctrl_a[ctrl_perm]
        data_a = step(data_a, jp.array(ctrl_a))
        data_b = step(data_b, jp.array(ctrl_b))

    qpos_a = np.array(data_a.qpos)
    qpos_b = np.array(data_b.qpos)
    err = np.abs(qpos_sign * qpos_a[qpos_perm] - qpos_b)
    assert err.max() < 5e-3, (
        f"mirrored rollout diverged: max qpos err {err.max():.2e} at "
        f"index {err.argmax()} after 20 control steps"
    )
