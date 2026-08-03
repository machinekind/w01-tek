"""The terrain command bias: terrain rows lean forward, the flat row keeps
the base mix, and a disabled bias is draw-for-draw the legacy sampler.

`_sample_command` reads only `self._config.command`, so a bare namespace
stands in for the env and nothing touches a device model.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np

from wojtek_rl import env as wojtek_env


def _sampler(**command_overrides):
    cfg = wojtek_env.default_config()
    cfg.command.update(command_overrides)
    fake = SimpleNamespace(_config=cfg)
    return lambda rng, on_flat=None: wojtek_env.WojtekJoystick._sample_command(
        fake, rng, on_flat
    )


BASE = dict(zero_prob=0.25, pure_wz_prob=0.25, pure_vy_prob=0.2,
            pure_slow_prob=0.1, pure_back_prob=0.2)


def test_disabled_bias_matches_the_legacy_draws_exactly():
    plain = _sampler(**BASE)
    biased_off = _sampler(**BASE)  # terrain_bias.enable defaults to False
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    a = jax.vmap(lambda k: plain(k))(keys)
    b = jax.vmap(lambda k: biased_off(k, jp.array(False)))(keys)
    np.testing.assert_array_equal(np.array(a), np.array(b))


def test_terrain_rows_stand_and_spin_less():
    cfg = dict(BASE)
    sample = _sampler(
        **cfg,
        terrain_bias=dict(enable=True, zero_prob=0.0, pure_wz_prob=0.0,
                          pure_vy_prob=0.0, pure_slow_prob=0.5,
                          pure_back_prob=0.0),
    )
    keys = jax.random.split(jax.random.PRNGKey(1), 512)
    on_terrain = jax.vmap(lambda k: sample(k, jp.array(False)))(keys)
    on_flat = jax.vmap(lambda k: sample(k, jp.array(True)))(keys)
    vt = np.array(on_terrain)[:, :3]
    vf = np.array(on_flat)[:, :3]
    # Terrain rows: zero_prob 0 means no standing draws at all.
    standing_t = (np.abs(vt) < 1e-9).all(axis=1).mean()
    standing_f = (np.abs(vf) < 1e-9).all(axis=1).mean()
    assert standing_t == 0.0
    assert standing_f > 0.1  # the flat row keeps its stand draws
    # Terrain rows walk forward far more often (slow draws at 0.5).
    forward_t = ((vt[:, 0] > 0.05) & (np.abs(vt[:, 1]) < 1e-9)
                 & (np.abs(vt[:, 2]) < 1e-9)).mean()
    forward_f = ((vf[:, 0] > 0.05) & (np.abs(vf[:, 1]) < 1e-9)
                 & (np.abs(vf[:, 2]) < 1e-9)).mean()
    assert forward_t > forward_f + 0.15
