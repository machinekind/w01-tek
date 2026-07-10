"""Backend flag plumbing (Workstream A of the MJWarp migration).

The suite runs with JAX_PLATFORMS=cpu, so "auto" must resolve to jax here.
Warp itself needs CUDA and is exercised on the GPU box, not in this suite.
"""

import jax
import pytest

from wojtek_rl import env as wojtek_env
from wojtek_rl import env_getup, env_jump
from wojtek_rl.base import data_budget_kwargs, resolve_backend


def test_resolve_backend_passes_explicit_values_through():
    assert resolve_backend("jax") == "jax"
    assert resolve_backend("warp") == "warp"


def test_resolve_backend_auto_is_jax_on_a_cpu_host():
    assert jax.default_backend() == "cpu"
    assert resolve_backend("auto") == "jax"


def test_resolve_backend_rejects_unknown_values():
    with pytest.raises(ValueError):
        resolve_backend("cuda")


def test_budget_kwargs_empty_for_jax():
    assert data_budget_kwargs("jax", 32, 320, 4096) == {}


def test_budget_kwargs_scale_naconmax_only():
    kw = data_budget_kwargs("warp", 32, 320, 4096)
    assert kw == {"naconmax": 32 * 4096, "njmax": 320}


def test_every_task_config_has_the_sim_block():
    for mod in (wojtek_env, env_getup, env_jump):
        sim = mod.default_config().sim
        assert sim.backend == "auto"
        assert sim.naconmax_per_env == 32
        assert sim.njmax == 320
        assert sim.num_envs == 1


def test_env_resolves_and_stores_the_backend():
    env = wojtek_env.WojtekJoystick()
    assert env._backend == "jax"
