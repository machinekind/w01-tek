"""Brax PPO trainer with a Random Network Distillation intrinsic bonus.

Vendored fork of ``brax.training.agents.ppo.train`` (brax 0.14.2), taken
because the upstream trainer has no hook for an auxiliary trained network
or for editing the reward stream between rollout and GAE. Every RND change
sits inside ``# RND >>> ... # <<< RND`` fences; with those blocks removed
the function is upstream ``train()`` line for line (helpers are imported
from the upstream module, not copied). Re-diff the fences when bumping brax.

What the fences add (math in ``wojtek_rl.rnd``):

- A frozen random target MLP and a trained predictor over one observation
  component (``rnd_config['obs_key']``, normally ``privileged_state``),
  fed the trainer's running-statistics-normalized obs clipped to [-5, 5].
- Per-transition intrinsic reward = predictor MSE on the unroll's
  ``next_observation``, divided by a running RMS of itself, scaled by
  ``rnd_config['coef']`` and ADDED to the env reward before GAE — so both
  the advantage and the value target see it (single reward stream, no
  separate intrinsic value head; the paper's dual-head variant is not
  implemented).
- One Adam step on the predictor per training_step, on the same batch.

RND state rides next to TrainingState in the scan carries; checkpoint
save/restore is untouched (a restored run re-initializes the predictor).

rnd_config keys: obs_key, hidden, out_dim, learning_rate, coef, seed
(optional; defaults to the trainer seed).
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import optimizer as ppo_optimizer
from brax.training.agents.ppo.train import (
    _PMAP_AXIS_NAME,
    TrainingState,
    _maybe_wrap_env,
    _random_translate_pixels,
    _remove_pixels,
    _strip_weak_type,
    _unpmap,
)
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp
import numpy as np
import optax

from wojtek_rl import rnd as rnd_lib

Metrics = types.Metrics


def train(
    environment: envs.Env,
    num_timesteps: int,
    # RND >>>
    rnd_config: Mapping[str, Any],
    # <<< RND
    # LCP >>> observation-gradient penalty coefficient (0 = off)
    lcp_coefficient: float = 0.0,
    # <<< LCP
    max_devices_per_host: Optional[int] = None,
    # high-level control flow
    wrap_env: bool = True,
    vision: bool = False,
    augment_pixels: bool = False,
    # environment wrapper
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # ppo params
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    normalize_observations_std_eps: float = 0.0,
    normalize_observations_mode: str = "welford",
    normalize_until_count: Optional[int] = None,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    clipping_epsilon_value: float | None = None,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    vf_loss_coefficient: float = 0.5,
    bootstrap_on_timeout: bool = False,
    use_distributional_critic: bool = False,
    desired_kl: float = 0.01,
    learning_rate_schedule: Optional[
        Union[str, ppo_optimizer.LRSchedule]
    ] = None,
    learning_rate_schedule_min_lr: float = 1e-5,
    learning_rate_schedule_max_lr: float = 1e-2,
    network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = True,
    run_evals: bool = True,
):
  """PPO training with an RND intrinsic bonus. See module docstring."""
  assert batch_size * num_minibatches % num_envs == 0

  if vision and action_repeat != 1:
    raise ValueError(
        "Implement action_repeat using PipelineEnv's _n_frames to avoid"
        ' unnecessary rendering!'
    )

  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d',
      jax.device_count(),
      process_count,
      process_id,
      local_device_count,
      local_devices_to_use,
  )
  device_count = local_devices_to_use * process_count

  # The number of environment steps executed for every training step.
  env_step_per_training_step = (
      batch_size * unroll_length * num_minibatches * action_repeat
  )
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of training_step calls per training_epoch call.
  # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
  #                                 num_resets_per_eval))
  num_training_steps_per_epoch = np.ceil(
      num_timesteps
      / (
          num_evals_after_init
          * env_step_per_training_step
          * max(num_resets_per_eval, 1)
      )
  ).astype(int)

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, key_env, eval_key = jax.random.split(local_key, 3)
  # key_networks should be global, so that networks are initialized the same
  # way for different processes.
  key_policy, key_value = jax.random.split(global_key)
  del global_key

  assert num_envs % device_count == 0

  env = _maybe_wrap_env(
      environment,
      wrap_env,
      num_envs,
      episode_length,
      action_repeat,
      device_count,
      key_env,
      wrap_env_fn,
      randomization_fn,
  )

  def reset_fn_donated_env_state(env_state_donated, key_envs):
    return env.reset(key_envs)

  key_envs = jax.random.split(key_env, num_envs // process_count)
  key_envs = jnp.reshape(
      key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
  )
  if local_devices_to_use > 1 or use_pmap_on_reset:
    reset_fn_ = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
    env_state = reset_fn_(key_envs)
    reset_fn = jax.pmap(
        reset_fn_donated_env_state,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(0,),
    )
  else:
    reset_fn_ = jax.jit(jax.vmap(env.reset))
    env_state = reset_fn_(key_envs)
    reset_fn = jax.jit(
        reset_fn_donated_env_state, donate_argnums=(0,), keep_unused=True
    )

  # Discard the batch axes over devices and envs.
  obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize
  if use_distributional_critic and clipping_epsilon_value is None:
    raise AssertionError(
        'clipping_epsilon_value must not be None when '
        'use_distributional_critic=True (it serves as kappa for quantile '
        'Huber loss)'
    )

  ppo_network = network_factory(
      obs_shape, env.action_size, preprocess_observations_fn=normalize
  )
  make_policy = ppo_networks.make_inference_fn(
      ppo_network,
      compute_value=bootstrap_on_timeout or clipping_epsilon_value is not None,
      use_distributional_critic=use_distributional_critic,
  )

  # Optimizer.
  base_optimizer = optax.adam(learning_rate=learning_rate)
  lr_schedule = learning_rate_schedule or ppo_optimizer.LRSchedule.NONE
  lr_schedule = ppo_optimizer.LRSchedule(lr_schedule)
  lr_is_adaptive_kl = lr_schedule == ppo_optimizer.LRSchedule.ADAPTIVE_KL
  if lr_is_adaptive_kl:
    base_optimizer = optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    )
  if max_grad_norm is not None:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        base_optimizer,
    )
  else:
    optimizer = base_optimizer

  loss_fn = functools.partial(
      ppo_losses.compute_ppo_loss,
      ppo_network=ppo_network,
      entropy_cost=entropy_cost,
      discounting=discounting,
      reward_scaling=reward_scaling,
      gae_lambda=gae_lambda,
      clipping_epsilon=clipping_epsilon,
      normalize_advantage=normalize_advantage,
      vf_coefficient=vf_loss_coefficient,
      clipping_epsilon_value=clipping_epsilon_value,
      use_distributional_critic=use_distributional_critic,
  )

  # LCP >>> Lipschitz-constrained policy via observation-gradient penalty
  # (arXiv 2410.11825): total = ppo_loss + lcp_coefficient *
  # E[||grad_obs log pi(a|s)||^2]. Replaces smoothness reward tuning with
  # a differentiable constraint on the policy itself; the paper's
  # recommended coefficient is ~0.002. Zero keeps the stock loss
  # (bit-exact: the wrapper is not even installed).
  if lcp_coefficient > 0.0:
    _base_loss_fn = loss_fn
    _policy_apply = ppo_network.policy_network.apply
    _action_dist = ppo_network.parametric_action_distribution

    def loss_fn(params, normalizer_params, data, rng):  # noqa: F811
      loss, metrics = _base_loss_fn(params, normalizer_params, data, rng)
      raw_action = data.extras['policy_extras']['raw_action']

      def _logp(obs, raw):
        logits = _policy_apply(normalizer_params, params.policy, obs)
        return jnp.sum(_action_dist.log_prob(logits, raw))

      _grad = jax.grad(_logp)

      def _sq_norm(obs, raw):
        g = _grad(obs, raw)
        return sum(
            jnp.sum(jnp.square(leaf))
            for leaf in jax.tree_util.tree_leaves(g)
        )

      def _flat2(x):
        return x.reshape((-1,) + x.shape[2:])

      obs_flat = jax.tree_util.tree_map(_flat2, data.observation)
      raw_flat = _flat2(raw_action)
      gp = jnp.mean(jax.vmap(_sq_norm)(obs_flat, raw_flat))
      metrics['lcp_gp'] = gp
      return loss + lcp_coefficient * gp, metrics
  # <<< LCP

  loss_and_pgrad_fn = gradients.loss_and_pgrad(
      loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
  )

  # RND >>> networks, predictor gradient, and the reward coefficient
  rnd_obs_key = rnd_config['obs_key']
  if not isinstance(obs_shape, Mapping) or rnd_obs_key not in obs_shape:
    raise KeyError(
        f'rnd.obs_key {rnd_obs_key!r} not found in the observation dict '
        f'(available: {sorted(obs_shape) if isinstance(obs_shape, Mapping) else "non-dict obs"})'
    )
  rnd_coef = float(rnd_config['coef'])
  # Self-ident mode (rnd.mode == 'selfident'): the predictor regresses the
  # last `target_dim` entries of `target_key` (the env_factors component,
  # appended last in the privileged list) from the actor obs, and its error
  # IS the intrinsic reward — end-to-end active identification, no proxies.
  _si_mode = str(rnd_config.get('mode', 'rnd')) == 'selfident'
  _si_target_key = str(rnd_config.get('target_key', 'privileged_state'))
  _si_target_dim = int(rnd_config.get('target_dim', 44))
  if _si_mode:
    rnd = rnd_lib.make_selfident(
        obs_size=int(obs_shape[rnd_obs_key][0]),
        target_dim=_si_target_dim,
        hidden=tuple(int(h) for h in rnd_config['hidden']),
        learning_rate=float(rnd_config['learning_rate']),
    )
  else:
    rnd = rnd_lib.make_rnd(
        obs_size=int(obs_shape[rnd_obs_key][0]),
        hidden=tuple(int(h) for h in rnd_config['hidden']),
        out_dim=int(rnd_config['out_dim']),
        learning_rate=float(rnd_config['learning_rate']),
    )
  rnd_loss_and_pgrad_fn = gradients.loss_and_pgrad(
      rnd.loss, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=False
  )

  def rnd_inputs(obs, normalizer_params):
    # Always normalized by the trainer's running statistics (maintained even
    # when normalize_observations is off) and clipped, per the paper.
    norm = running_statistics.normalize(_remove_pixels(obs), normalizer_params)
    x = rnd_lib.clip_inputs(norm[rnd_obs_key])
    if _si_mode:
      y = rnd_lib.clip_inputs(norm[_si_target_key][..., -_si_target_dim:])
      return jnp.concatenate([x, y], axis=-1)
    return x
  # <<< RND

  steps_between_logging = training_metrics_steps or env_step_per_training_step
  metrics_aggregator = metric_logger.EpisodeMetricsLogger(
      steps_between_logging=steps_between_logging,
      progress_fn=progress_fn,
  )

  def minibatch_step(
      carry,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key = carry
    key, key_loss = jax.random.split(key)
    (_, metrics), grads = loss_and_pgrad_fn(
        params, normalizer_params, data, key_loss
    )

    if lr_is_adaptive_kl:
      kl_mean = metrics['kl_mean']
      kl_mean = jax.lax.pmean(kl_mean, axis_name=_PMAP_AXIS_NAME)
      optimizer_state, lr = ppo_optimizer.adaptive_kl_learning_rate(
          optimizer_state, kl_mean, desired_kl,
          min_learning_rate=learning_rate_schedule_min_lr,
          max_learning_rate=learning_rate_schedule_max_lr,
      )
    else:
      lr = jnp.array(learning_rate)
    metrics['learning_rate'] = lr

    # apply gradients
    params_update, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, params_update)

    return (optimizer_state, params, key), metrics

  def sgd_step(
      carry,
      unused_t,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)

    if augment_pixels:
      key, key_rt = jax.random.split(key)
      r_translate = functools.partial(_random_translate_pixels, key=key_rt)
      data = types.Transition(
          observation=r_translate(data.observation),  # pytype: disable=wrong-arg-types
          action=data.action,
          reward=data.reward,
          discount=data.discount,
          next_observation=r_translate(data.next_observation),  # pytype: disable=wrong-arg-types
          extras=data.extras,
      )

    def convert_data(x: jnp.ndarray):
      x = jax.random.permutation(key_perm, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    shuffled_data = jax.tree_util.tree_map(convert_data, data)
    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(minibatch_step, normalizer_params=normalizer_params),
        (optimizer_state, params, key_grad),
        shuffled_data,
        length=num_minibatches,
    )

    return (optimizer_state, params, key), metrics

  def training_step(
      # RND >>> rnd_state joins the carry
      carry: Tuple[TrainingState, rnd_lib.RNDState, envs.State, PRNGKey],
      unused_t,
  ) -> Tuple[
      Tuple[TrainingState, rnd_lib.RNDState, envs.State, PRNGKey], Metrics
  ]:
    training_state, rnd_state, state, key = carry
    # <<< RND
    key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

    policy = make_policy((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))

    def f(carry, unused_t):
      current_state, current_key = carry
      current_key, next_key = jax.random.split(current_key)
      extra_fields = ['truncation', 'episode_metrics', 'episode_done']
      if bootstrap_on_timeout:
        extra_fields.append('time_out')
      next_state, data = acting.generate_unroll(
          env,
          current_state,
          policy,
          current_key,
          unroll_length,
          extra_fields=tuple(extra_fields),
      )
      return (next_state, next_key), data

    (state, _), data = jax.lax.scan(
        f,
        (state, key_generate_unroll),
        (),
        length=batch_size * num_minibatches // num_envs,
    )
    # Have leading dimensions (batch_size * num_minibatches, unroll_length)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
    )
    assert data.discount.shape[1:] == (unroll_length,)

    if bootstrap_on_timeout:  # bootstrap reward on timeout
      time_out = data.extras['state_extras']['time_out']
      value = data.extras['policy_extras']['value']
      data = types.Transition(
          observation=data.observation,
          action=data.action,
          reward=data.reward + discounting * time_out * value,
          discount=data.discount,
          next_observation=data.next_observation,
          extras=data.extras,
      )

    # RND >>> intrinsic bonus onto the reward stream + one predictor step.
    # Novelty is measured on the REACHED state (next_observation), with the
    # normalizer params from before this batch's stats update.
    x = rnd_inputs(data.next_observation, training_state.normalizer_params)
    raw = rnd.intrinsic(rnd_state, x)
    batch_ms = jax.lax.pmean(
        jnp.mean(jnp.square(raw)), axis_name=_PMAP_AXIS_NAME
    )
    scaled, rnd_state = rnd_lib.normalize_intrinsic(rnd_state, raw, batch_ms)
    data = data._replace(reward=data.reward + rnd_coef * scaled)
    rnd_pred_loss, rnd_grads = rnd_loss_and_pgrad_fn(
        rnd_state.predictor_params, rnd_state.target_params, x
    )
    rnd_update, rnd_opt_state = rnd.optimizer.update(
        rnd_grads, rnd_state.opt_state
    )
    rnd_state = rnd_state.replace(
        predictor_params=optax.apply_updates(
            rnd_state.predictor_params, rnd_update
        ),
        opt_state=rnd_opt_state,
    )
    # <<< RND

    normalizer_params = training_state.normalizer_params
    if not lr_is_adaptive_kl:
      # Update normalization params before SGD for backwards compatibility.
      normalizer_params = running_statistics.update(
          normalizer_params,
          _remove_pixels(data.observation),
          pmap_axis_name=_PMAP_AXIS_NAME,
          until_count=normalize_until_count,
      )

    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(
            sgd_step, data=data, normalizer_params=normalizer_params
        ),
        (training_state.optimizer_state, training_state.params, key_sgd),
        (),
        length=num_updates_per_batch,
    )

    if lr_is_adaptive_kl:
      # For adaptive KL, normalization params should be updated after SGD s.t.
      # old distribution outputs are valid for KL computation.
      normalizer_params = running_statistics.update(
          normalizer_params,
          _remove_pixels(data.observation),
          pmap_axis_name=_PMAP_AXIS_NAME,
          until_count=normalize_until_count,
      )

    new_training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=params,
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + env_step_per_training_step,
    )

    # RND >>> surface the bonus scale next to the PPO losses
    metrics['rnd_intrinsic_raw'] = jnp.mean(raw)
    metrics['rnd_intrinsic_scaled'] = rnd_coef * jnp.mean(scaled)
    metrics['rnd_predictor_loss'] = rnd_pred_loss
    metrics['rnd_reward_rms'] = jnp.sqrt(rnd_state.reward_ms)
    # <<< RND

    if log_training_metrics:  # log unroll metrics
      jax.debug.callback(
          metrics_aggregator.update_episode_metrics,
          data.extras['state_extras']['episode_metrics'],
          data.extras['state_extras']['episode_done'],
          metrics,
      )

    # RND >>>
    return (new_training_state, rnd_state, state, new_key), metrics
    # <<< RND

  def training_epoch(
      # RND >>> rnd_state threads through the epoch scan
      training_state: TrainingState,
      rnd_state: rnd_lib.RNDState,
      state: envs.State,
      key: PRNGKey,
  ) -> Tuple[TrainingState, rnd_lib.RNDState, envs.State, Metrics]:
    (training_state, rnd_state, state, _), loss_metrics = jax.lax.scan(
        training_step,
        (training_state, rnd_state, state, key),
        (),
        length=num_training_steps_per_epoch,
    )
    loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
    return training_state, rnd_state, state, loss_metrics
    # <<< RND

  training_epoch = jax.pmap(
      training_epoch,
      axis_name=_PMAP_AXIS_NAME,
      # RND >>> rnd_state is donated like the other carried state
      donate_argnums=(0, 1, 2),
      # <<< RND
  )

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      # RND >>>
      training_state: TrainingState,
      rnd_state: rnd_lib.RNDState,
      env_state: envs.State,
      key: PRNGKey,
  ) -> Tuple[TrainingState, rnd_lib.RNDState, envs.State, Metrics]:
    # <<< RND
    nonlocal training_walltime
    t = time.time()
    # RND >>>
    training_state, rnd_state, env_state = _strip_weak_type(
        (training_state, rnd_state, env_state)
    )
    result = training_epoch(training_state, rnd_state, env_state, key)
    training_state, rnd_state, env_state, metrics = _strip_weak_type(result)
    # <<< RND

    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
        num_training_steps_per_epoch
        * env_step_per_training_step
        * max(num_resets_per_eval, 1)
    ) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()},
    }
    # RND >>>
    return training_state, rnd_state, env_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade
    # <<< RND

  # Initialize model params and training state.
  init_params = ppo_losses.PPONetworkParams(
      policy=ppo_network.policy_network.init(key_policy),
      value=ppo_network.value_network.init(key_value),
  )

  obs_shape = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
  )
  training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
      optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
      params=init_params,
      normalizer_params=running_statistics.init_state(
          _remove_pixels(obs_shape),
          std_eps=normalize_observations_std_eps,
          mode=normalize_observations_mode,
      ),
      env_steps=types.UInt64(hi=0, lo=0),
  )

  # RND >>> fresh key stream, deliberately not spliced into the upstream
  # ones so an rnd run keeps upstream's policy/env draws bit-identical.
  # Never restored from checkpoints: a restored run re-learns the predictor
  # (the intrinsic scale re-normalizes within a few batches).
  rnd_state = rnd.init(
      jax.random.PRNGKey(int(rnd_config.get('seed', seed)) + 7919)
  )
  # <<< RND

  if restore_checkpoint_path is not None:
    params = checkpoint.load(restore_checkpoint_path)
    value_params = params[2] if restore_value_fn else init_params.value
    training_state = training_state.replace(
        normalizer_params=params[0],
        params=training_state.params.replace(
            policy=params[1], value=value_params
        ),
    )

  if restore_params is not None:
    logging.info('Restoring TrainingState from `restore_params`.')
    value_params = restore_params[2] if restore_value_fn else init_params.value
    training_state = training_state.replace(
        normalizer_params=restore_params[0],
        params=training_state.params.replace(
            policy=restore_params[1], value=value_params
        ),
    )

  if num_timesteps == 0:
    return (
        make_policy,
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        ),
        {},
    )

  training_state = jax.device_put_replicated(
      training_state, jax.local_devices()[:local_devices_to_use]
  )
  # RND >>>
  rnd_state = jax.device_put_replicated(
      rnd_state, jax.local_devices()[:local_devices_to_use]
  )
  # <<< RND

  eval_env = _maybe_wrap_env(
      eval_env or environment,
      wrap_env,
      num_eval_envs,
      episode_length,
      action_repeat,
      device_count=1,  # eval on the host only
      key_env=eval_key,
      wrap_env_fn=wrap_env_fn,
      randomization_fn=randomization_fn,
  )
  evaluator = acting.Evaluator(
      eval_env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key,
  )

  training_metrics = {}
  training_walltime = 0
  current_step = 0

  # Run initial eval
  metrics = {}
  if process_id == 0 and num_evals > 1 and run_evals:
    metrics = evaluator.run_evaluation(
        _unpmap((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )),
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)

  # Run initial policy_params_fn.
  params = _unpmap((
      training_state.normalizer_params,
      training_state.params.policy,
      training_state.params.value,
  ))
  policy_params_fn(current_step, make_policy, params)

  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    for _ in range(max(num_resets_per_eval, 1)):
      # optimization
      epoch_key, local_key = jax.random.split(local_key)
      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
      # RND >>>
      (training_state, rnd_state, env_state, training_metrics) = (
          training_epoch_with_timing(
              training_state, rnd_state, env_state, epoch_keys
          )
      )
      # <<< RND
      current_step = int(_unpmap(training_state.env_steps))

      key_envs = jax.vmap(
          lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
      )(key_envs, key_envs.shape[1])
      # TODO(brax-team): move extra reset logic to the AutoResetWrapper.
      if num_resets_per_eval > 0:
        env_state = reset_fn(env_state, key_envs)

    if process_id != 0:
      continue

    # Process id == 0.
    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))

    policy_params_fn(current_step, make_policy, params)

    if save_checkpoint_path is not None:
      ckpt_config = checkpoint.network_config(
          observation_size=obs_shape,
          action_size=env.action_size,
          normalize_observations=normalize_observations,
          network_factory=network_factory,
      )
      checkpoint.save(
          save_checkpoint_path, current_step, params, ckpt_config
      )

    if num_evals > 0:
      metrics = training_metrics
      if run_evals:
        metrics = evaluator.run_evaluation(
            params,
            training_metrics,
        )
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  if not total_steps >= num_timesteps:
    raise AssertionError(
        f'Total steps {total_steps} is less than `num_timesteps`='
        f' {num_timesteps}.'
    )

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  params = _unpmap((
      training_state.normalizer_params,
      training_state.params.policy,
      training_state.params.value,
  ))
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)
