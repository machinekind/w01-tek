"""RMA plumbing: the proprio_history buffer really rolls, and
env_factors is the env's own DR draw, never a silent constant.

Same silent-failure surface as test_mu_obs: both components read state
that only becomes meaningful through specific wiring (the info-carried
ring buffer; the DR wrapper's per-env model rebinding), and both would
degrade to constants without it.
"""

import functools

import jax
import jax.numpy as jp
import numpy as np

from wojtek_rl import env as wojtek_env
from wojtek_rl.randomize import make_domain_randomize

H = 5
FRAME = 36  # joint_pos + joint_vel + last_act


def _rma_env():
    cfg = wojtek_env.default_config()
    cfg.obs.history_len = H
    cfg.obs.state = tuple(cfg.obs.state) + ("proprio_history",)
    cfg.obs.privileged = tuple(cfg.obs.privileged) + ("env_factors",)
    return wojtek_env.WojtekJoystick(config=cfg)


def test_history_buffer_rolls():
    env = _rma_env()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    hist = np.asarray(state.info["proprio_hist"])
    assert hist.shape == (H * FRAME,)
    # Reset pushed exactly one frame; older slots are the zero seed.
    assert np.allclose(hist[: (H - 1) * FRAME], 0.0)
    step = jax.jit(env.step)
    for _ in range(2):
        act = jp.zeros(env.action_size)
        state = step(state, act)
    hist = np.asarray(state.info["proprio_hist"])
    # Newest frame is the CURRENT step's actor-visible proprio.
    catalog = env._obs_catalog(state.data, state.info)
    newest = hist[-FRAME:]
    np.testing.assert_allclose(
        newest[:12], np.asarray(catalog["joint_pos"]), rtol=1e-6
    )
    np.testing.assert_allclose(
        newest[12:24], np.asarray(catalog["joint_vel"]), rtol=1e-6
    )
    # And the buffer content reaches the actor obs tail (appended last).
    np.testing.assert_allclose(
        np.asarray(state.obs["state"])[-H * FRAME:][-FRAME:][:12],
        newest[:12],
        rtol=1e-6,
    )


def test_env_factors_nominal_and_randomized():
    env = _rma_env()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    ef = np.asarray(env._obs_catalog(state.data, state.info)["env_factors"])
    assert ef.shape == (44,)
    # Unrandomized: mu at the model constant, every scale exactly 1,
    # CoM offset exactly 0.
    np.testing.assert_allclose(ef[:4], 0.9, rtol=1e-6)
    np.testing.assert_allclose(ef[4:40], 1.0, rtol=1e-6)
    np.testing.assert_allclose(ef[40:41], 1.0, rtol=1e-6)
    np.testing.assert_allclose(ef[41:44], 0.0, atol=1e-8)

    from mujoco_playground._src.wrapper import (
        BraxDomainRandomizationVmapWrapper,
    )

    n = 8
    dr = {
        "com_offset": {"enable": True, "xy": 0.02, "z": 0.01},
        "joint_gains": {"enable": True, "gain_pct": 0.2, "kd_pct": 0.2},
        "dof": {
            "enable": False,
            "damping": [0.9, 1.1],
            "armature": [0.9, 1.1],
            "frictionloss": [0.9, 1.1],
        },
        "foot_friction": {"enable": True, "range": [0.25, 1.35]},
        "motor_strength": {"enable": True, "range": [0.5, 1.1]},
    }
    rand_fn = make_domain_randomize(env.mj_model, dr)
    wrapped = BraxDomainRandomizationVmapWrapper(
        env,
        functools.partial(rand_fn, rng=jax.random.split(jax.random.PRNGKey(1), n)),
    )
    state = wrapped.reset(jax.random.split(jax.random.PRNGKey(2), n))
    ef = np.asarray(state.obs["privileged_state"])[:, -44:]
    # Per-env variance in every factor family the DR draws.
    for sl, label in [
        (slice(0, 4), "mu"),
        (slice(4, 16), "gain"),
        (slice(16, 28), "kd"),
        (slice(28, 40), "force"),
        (slice(41, 44), "com"),
    ]:
        assert np.unique(ef[:, sl].round(5), axis=0).shape[0] > 1, (
            f"env_factors[{label}] identical across envs"
        )
