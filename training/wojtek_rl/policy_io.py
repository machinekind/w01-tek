"""Rebuild the PPO network and load checkpoint params.

Copied from 3_jaxpot_robotics/jaxpot_robotics/envs.py::load_policy, which
exists because brax's own checkpoint.load_policy crashes on this brax
version (mean_kernel_init_fn serializes as null). Keep the two in sync.
"""

from __future__ import annotations

import functools
from typing import Callable

from brax.training import types as brax_types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks


def _build_network_factory(ppo_params) -> Callable:
    """Reconstruct the PPO network factory from a PPO ConfigDict.

    Mirrors mujoco_playground's train_jax_ppo.py: the tuned config carries a
    `network_factory` sub-config (hidden layer sizes, obs keys, etc.) that
    must be threaded into `make_ppo_networks`.
    """
    if hasattr(ppo_params, "network_factory"):
        return functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        )
    return ppo_networks.make_ppo_networks


def load_policy(ckpt_dir, env, ppo_params, deterministic: bool = True):
    """Rebuild a deterministic inference fn from a Brax PPO checkpoint.

    We deliberately do NOT use ``brax...ppo.checkpoint.load_policy``: its
    ``load_config`` crashes on this brax version when an optional
    kernel-init field (e.g. ``mean_kernel_init_fn``) is serialized as
    ``null`` (``KERNEL_INITIALIZER[None]`` -> KeyError). Instead we
    reconstruct the network from the same tuned ``ppo_params`` used at
    train time and the env's observation/action spec, then load just the
    params pytree -- which is robust across config-serialization quirks.
    """
    network_factory = _build_network_factory(ppo_params)
    normalize = (
        running_statistics.normalize
        if ppo_params.get("normalize_observations", True)
        else brax_types.identity_observation_preprocessor
    )
    ppo_network = network_factory(
        env.observation_size, env.action_size, preprocess_observations_fn=normalize
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    params = ppo_checkpoint.load(ckpt_dir)
    return make_inference_fn(params, deterministic=deterministic)
