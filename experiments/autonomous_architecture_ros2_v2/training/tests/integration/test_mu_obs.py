"""The foot_friction observation: really the env's own mu, never a
silent constant.

The component reads self._mjx_model.geom_friction, which only becomes
per-env under the brax DR wrapper's model rebinding — a plumbing detail
that would fail SILENTLY (constant 0.9 for every env) if the wrapper
stopped rebinding. These tests pin both halves: the catalog wiring, and
the end-to-end variance under the real randomization path.
"""

import jax
import jax.numpy as jp
import numpy as np

from wojtek_rl import env as wojtek_env
from wojtek_rl.randomize import make_domain_randomize

MU_RANGE = (0.4, 1.35)


def _teacher_env():
    cfg = wojtek_env.default_config()
    cfg.obs.state = tuple(cfg.obs.state) + ("foot_friction",)
    cfg.obs.privileged = tuple(cfg.obs.privileged) + ("foot_friction",)
    return wojtek_env.WojtekJoystick(config=cfg)


def test_catalog_exposes_model_friction():
    env = _teacher_env()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    catalog = env._obs_catalog(state.data, state.info)
    mu = np.asarray(catalog["foot_friction"])
    assert mu.shape == (4,)
    # Unrandomized env: the model constant.
    np.testing.assert_allclose(mu, 0.9, rtol=1e-6)
    # And it is the tail of the actor obs (appended last in the lists).
    np.testing.assert_allclose(np.asarray(state.obs["state"])[-4:], 0.9, rtol=1e-6)


def test_mu_varies_per_env_under_dr_wrapper():
    from mujoco_playground._src.wrapper import (
        BraxDomainRandomizationVmapWrapper,
    )

    env = _teacher_env()
    n = 8
    dr = {
        "com_offset": {"enable": False, "xy": 0.02, "z": 0.01},
        "joint_gains": {"enable": False, "gain_pct": 0.2, "kd_pct": 0.2},
        "dof": {
            "enable": False,
            "damping": [0.9, 1.1],
            "armature": [0.9, 1.1],
            "frictionloss": [0.9, 1.1],
        },
        "foot_friction": {"enable": True, "range": list(MU_RANGE)},
        "motor_strength": {"enable": False, "range": [0.5, 1.1]},
    }
    rand_fn = make_domain_randomize(env.mj_model, dr)
    import functools

    wrapped = BraxDomainRandomizationVmapWrapper(
        env,
        functools.partial(rand_fn, rng=jax.random.split(jax.random.PRNGKey(1), n)),
    )
    state = wrapped.reset(jax.random.split(jax.random.PRNGKey(2), n))
    mu_obs = np.asarray(state.obs["state"])[:, -4:]
    # The foot_friction DR is MULTIPLICATIVE on the model's base mu (0.9),
    # so the effective range is [0.9*lo, 0.9*hi]. Every env inside it, and
    # the envs genuinely differ — the silent-constant failure mode reads
    # 0.9 everywhere with zero spread.
    assert mu_obs.min() >= 0.9 * MU_RANGE[0] - 1e-6
    assert mu_obs.max() <= 0.9 * MU_RANGE[1] + 1e-6
    assert np.unique(mu_obs.round(4)).size > 1, "mu identical across envs"
    # And the obs matches the wrapper's own randomized model, env by env.
    model_mu = np.asarray(
        wrapped._mjx_model_v.geom_friction[:, env._foot_geom_ids, 0]
    )
    np.testing.assert_allclose(mu_obs, model_mu, rtol=1e-6)
