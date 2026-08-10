"""Per-episode gyro-bias DR, model-free: the _build_obs application path
(actor's gyro biased, critic clean) via a stub catalog, plus the config
default and the v4_1_2 preset wiring. The reset-time draw itself needs a
live env and is covered by tests/integration.
"""

import jax
import jax.numpy as jp
import numpy as np
import yaml
from ml_collections import config_dict

from wojtek_rl import base
from wojtek_rl import env as wojtek_env
from wojtek_rl import paths

EXPERIMENT_DIR = paths.PROJECT_DIR / "wojtek_rl" / "conf" / "experiment"

GYRO = jp.array([0.1, -0.2, 0.3])
FOO = jp.array([1.0, 2.0])
BIAS = jp.array([0.05, -0.02, 0.01])


def _make_stub(obs_noise=None):
    """An object with the real _build_obs / actor_obs_names over a fixed
    two-component catalog; no model, no env construction."""

    class Stub:
        _build_obs = base.WojtekEnv._build_obs
        _noisy = base.WojtekEnv._noisy
        actor_obs_names = base.WojtekEnv.actor_obs_names

        def __init__(self):
            self._config = config_dict.create(
                obs=config_dict.create(
                    include=(),
                    state=("gyro", "foo"),
                    privileged=("gyro", "foo"),
                ),
                obs_noise=config_dict.create(**(obs_noise or {})),
            )

        def _obs_catalog(self, data, info):
            return {"gyro": GYRO, "foo": FOO}

    return Stub()


def test_bias_hits_actor_only():
    obs = _make_stub()._build_obs(None, {"gyro_bias": BIAS})
    np.testing.assert_allclose(obs["state"][:3], GYRO + BIAS, rtol=1e-6)
    np.testing.assert_allclose(obs["state"][3:], FOO, rtol=1e-6)
    # The critic's copy stays clean.
    np.testing.assert_allclose(
        obs["privileged_state"], jp.concatenate([GYRO, FOO]), rtol=1e-6
    )


def test_no_bias_key_is_passthrough():
    # Envs that never draw a bias (getup/jump) put no key in info.
    obs = _make_stub()._build_obs(None, {})
    np.testing.assert_allclose(obs["state"], jp.concatenate([GYRO, FOO]))


def test_bias_survives_the_noise_path():
    # With every white-noise scale at zero, the rng branch must still
    # deliver clean + bias (the bias is not lost when noise is applied).
    stub = _make_stub(obs_noise={"gyro": 0.0, "foo": 0.0, "gyro_bias": 0.05})
    obs = stub._build_obs(None, {"gyro_bias": BIAS}, jax.random.PRNGKey(0))
    np.testing.assert_allclose(
        obs["state"], jp.concatenate([GYRO + BIAS, FOO]), rtol=1e-6
    )


def test_default_config_disables_gyro_bias():
    assert wojtek_env.default_config().obs_noise.gyro_bias == 0.0


def test_v412_preset_is_v411_plus_gyro_bias():
    v411 = yaml.safe_load(
        (EXPERIMENT_DIR / "terrain_blind_v4_1_1.yaml").read_text()
    )
    v412 = yaml.safe_load(
        (EXPERIMENT_DIR / "terrain_blind_v4_1_2.yaml").read_text()
    )
    assert v412["task"]["env"]["obs_noise"] == {"gyro_bias": 0.05}
    assert v412["run_name"] == "wojtek_terrain_blind_v4_1_2"
    # Everything else is v4_1_1's operating point, unchanged.
    del v412["task"]["env"]["obs_noise"]
    v411.pop("run_name"), v412.pop("run_name")
    assert v412 == v411
