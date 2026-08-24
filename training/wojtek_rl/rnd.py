"""Random Network Distillation primitives (arXiv 1810.12894).

A frozen, randomly initialized target MLP maps observations to an embedding;
a trained predictor chases it. The predictor's per-sample MSE is the
intrinsic reward: large where the state distribution has not been visited
(the predictor has never regressed there), shrinking as visits accumulate.

This module holds the math only — networks, state, intrinsic reward, the
predictor loss, and the running normalization of the intrinsic scale — as
pure functions over an explicit ``RNDState``, so ``ppo_rnd.py`` (the brax
PPO fork that wires them into the training loop) stays a minimal diff and
everything here is unit-testable without an environment.

Inputs are expected to be normalized observations (the trainer's running
statistics) clipped to [-5, 5], per the paper; ``clip_inputs`` is that clip.
The intrinsic reward is divided by a running RMS of itself (a simplification
of the paper's returns-std normalization) so ``coef`` in the trainer stays
meaningful across tasks and training time.
"""

import dataclasses
from typing import Any, Callable, Sequence

import flax
import jax
import jax.numpy as jp
import optax
from brax.training import networks
from brax.training.types import Params, PRNGKey

INPUT_CLIP = 5.0
_RMS_EPS = 1e-8


@flax.struct.dataclass
class RNDState:
    target_params: Params
    predictor_params: Params
    opt_state: optax.OptState
    # Running mean square of the RAW intrinsic reward, and the sample count
    # behind it. The normalized intrinsic is raw * rsqrt(reward_ms + eps).
    reward_ms: jax.Array
    count: jax.Array


@dataclasses.dataclass(frozen=True)
class RND:
    """Bundle of pure functions over RNDState (closes over the networks)."""

    init: Callable[[PRNGKey], RNDState]
    intrinsic: Callable[[RNDState, jax.Array], jax.Array]
    loss: Callable[[Params, Params, jax.Array], jax.Array]
    optimizer: Any


def clip_inputs(x: jax.Array) -> jax.Array:
    return jp.clip(x, -INPUT_CLIP, INPUT_CLIP)


def make_rnd(
    obs_size: int,
    hidden: Sequence[int] = (256, 256),
    out_dim: int = 128,
    learning_rate: float = 1e-4,
) -> RND:
    target = networks.MLP(layer_sizes=list(hidden) + [out_dim])
    predictor = networks.MLP(layer_sizes=list(hidden) + [out_dim])
    optimizer = optax.adam(learning_rate=learning_rate)

    def init(key: PRNGKey) -> RNDState:
        k_target, k_pred = jax.random.split(key)
        dummy = jp.zeros((1, obs_size))
        predictor_params = predictor.init(k_pred, dummy)
        return RNDState(
            target_params=target.init(k_target, dummy),
            predictor_params=predictor_params,
            opt_state=optimizer.init(predictor_params),
            # Seeded at 1 so the very first normalized batch is order-1; the
            # count-weighted update below mostly overwrites it immediately.
            reward_ms=jp.ones(()),
            count=jp.zeros(()),
        )

    def intrinsic(state: RNDState, x: jax.Array) -> jax.Array:
        """Per-sample raw intrinsic reward, shape x.shape[:-1]."""
        t = target.apply(state.target_params, x)
        p = predictor.apply(state.predictor_params, x)
        return jp.mean(jp.square(p - jax.lax.stop_gradient(t)), axis=-1)

    def loss(
        predictor_params: Params, target_params: Params, x: jax.Array
    ) -> jax.Array:
        t = jax.lax.stop_gradient(target.apply(target_params, x))
        p = predictor.apply(predictor_params, x)
        return jp.mean(jp.square(p - t))

    return RND(init=init, intrinsic=intrinsic, loss=loss, optimizer=optimizer)


def make_selfident(
    obs_size: int,
    target_dim: int,
    hidden: Sequence[int] = (256, 256),
    learning_rate: float = 1e-4,
) -> RND:
    """End-to-end active identification (EPI-style, arXiv 1907.11740):
    the predictor regresses the TRUE privileged environment factors from
    the actor-visible observation, and its per-sample error is the
    intrinsic reward — the policy is paid for visiting states where its
    own parameters are still ambiguous, with no hand-chosen excitation
    proxies. Unlike RND's novelty, the target is a per-env CONSTANT, so
    there is no noisy-TV failure mode; the irreducible floor is the
    genuinely unidentifiable part.

    Plumbing trick: callers pass x_aug = concat([inputs, target]) through
    the same RND bundle signatures; functions split internally. The
    RNDState.target_params slot is unused (kept for dataclass parity).
    """
    predictor = networks.MLP(layer_sizes=list(hidden) + [target_dim])
    optimizer = optax.adam(learning_rate=learning_rate)

    def _split(xy: jax.Array):
        return xy[..., :obs_size], xy[..., obs_size:]

    def init(key: PRNGKey) -> RNDState:
        dummy = jp.zeros((1, obs_size))
        predictor_params = predictor.init(key, dummy)
        return RNDState(
            target_params={},
            predictor_params=predictor_params,
            opt_state=optimizer.init(predictor_params),
            reward_ms=jp.ones(()),
            count=jp.zeros(()),
        )

    def intrinsic(state: RNDState, xy: jax.Array) -> jax.Array:
        x, y = _split(xy)
        p = predictor.apply(state.predictor_params, x)
        return jp.mean(jp.square(p - jax.lax.stop_gradient(y)), axis=-1)

    def loss(
        predictor_params: Params, target_params: Params, xy: jax.Array
    ) -> jax.Array:
        del target_params
        x, y = _split(xy)
        p = predictor.apply(predictor_params, x)
        return jp.mean(jp.square(p - jax.lax.stop_gradient(y)))

    return RND(init=init, intrinsic=intrinsic, loss=loss, optimizer=optimizer)


def normalize_intrinsic(
    state: RNDState, raw: jax.Array, batch_ms: jax.Array
) -> tuple[jax.Array, RNDState]:
    """Scale `raw` by the running RMS and fold this batch into the stats.

    `batch_ms` is mean(raw**2) over the batch — passed in (rather than
    computed here) so a pmapped caller can pmean it across devices first
    and every device folds the same number in.
    """
    n = raw.size
    new_count = state.count + n
    w = n / new_count
    reward_ms = state.reward_ms + (batch_ms - state.reward_ms) * w
    scaled = raw * jax.lax.rsqrt(reward_ms + _RMS_EPS)
    return scaled, state.replace(reward_ms=reward_ms, count=new_count)
