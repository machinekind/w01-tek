"""wojtek_rl.rnd: the RND math, model-free.

The trainer-side plumbing (reward mixing, pmean, carry threading) lives in
ppo_rnd.py and is exercised by a training smoke run; these tests pin the
pure functions: the predictor actually learns the target, novelty reads
higher off-distribution than on it, and the running normalization tracks
the intrinsic scale.
"""

import jax
import jax.numpy as jp
import numpy as np
import optax
import pytest

from wojtek_rl import rnd as rnd_lib

OBS = 16


@pytest.fixture(scope="module")
def rnd():
    return rnd_lib.make_rnd(OBS, hidden=(32, 32), out_dim=8, learning_rate=1e-3)


@pytest.fixture(scope="module")
def state(rnd):
    return rnd.init(jax.random.PRNGKey(0))


def _update(rnd, state, x, steps):
    loss_grad = jax.jit(jax.value_and_grad(rnd.loss), static_argnums=())
    for _ in range(steps):
        _, grads = loss_grad(state.predictor_params, state.target_params, x)
        update, opt_state = rnd.optimizer.update(grads, state.opt_state)
        state = state.replace(
            predictor_params=optax.apply_updates(
                state.predictor_params, update
            ),
            opt_state=opt_state,
        )
    return state


def test_init_is_deterministic_and_asymmetric(rnd):
    a = rnd.init(jax.random.PRNGKey(3))
    b = rnd.init(jax.random.PRNGKey(3))
    chex_equal = jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda x, y: bool(jp.all(x == y)), a.predictor_params,
            b.predictor_params,
        )
    )
    assert chex_equal
    # Target and predictor must start different, or the intrinsic reward is
    # born zero everywhere.
    leaves_t = jax.tree_util.tree_leaves(a.target_params)
    leaves_p = jax.tree_util.tree_leaves(a.predictor_params)
    assert any(
        t.shape != p.shape or not bool(jp.all(t == p))
        for t, p in zip(leaves_t, leaves_p)
    )


def test_predictor_learns_visited_states(rnd, state):
    x = jax.random.normal(jax.random.PRNGKey(1), (256, OBS))
    before = float(jp.mean(rnd.intrinsic(state, x)))
    trained = _update(rnd, state, x, steps=200)
    after = float(jp.mean(rnd.intrinsic(trained, x)))
    assert after < 0.5 * before, (before, after)


def test_novelty_is_higher_off_distribution(rnd, state):
    seen = jax.random.normal(jax.random.PRNGKey(1), (256, OBS))
    trained = _update(rnd, state, seen, steps=300)
    novel = 3.0 + jax.random.normal(jax.random.PRNGKey(2), (256, OBS))
    on_dist = float(jp.mean(rnd.intrinsic(trained, seen)))
    off_dist = float(jp.mean(rnd.intrinsic(trained, novel)))
    assert off_dist > 2.0 * on_dist, (on_dist, off_dist)


def test_target_params_never_change(rnd, state):
    x = jax.random.normal(jax.random.PRNGKey(1), (64, OBS))
    trained = _update(rnd, state, x, steps=10)
    same = jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda a, b: bool(jp.all(a == b)),
            state.target_params,
            trained.target_params,
        )
    )
    assert same


def test_normalize_intrinsic_tracks_rms(rnd, state):
    raw = jp.full((100,), 4.0)
    batch_ms = jp.mean(jp.square(raw))  # 16
    scaled, new_state = rnd_lib.normalize_intrinsic(state, raw, batch_ms)
    # count was 0, so the batch fully overwrites the seed reward_ms.
    np.testing.assert_allclose(float(new_state.reward_ms), 16.0, rtol=1e-6)
    np.testing.assert_allclose(float(new_state.count), 100.0)
    # RMS-normalized constant signal reads ~1.
    np.testing.assert_allclose(np.asarray(scaled), 1.0, rtol=1e-3)
    # A second identical batch: stats converge, scale stays ~1.
    scaled2, s2 = rnd_lib.normalize_intrinsic(new_state, raw, batch_ms)
    np.testing.assert_allclose(float(s2.reward_ms), 16.0, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(scaled2), 1.0, rtol=1e-3)


def test_clip_inputs_bounds():
    x = jp.array([-100.0, -5.0, 0.0, 5.0, 100.0])
    np.testing.assert_array_equal(
        np.asarray(rnd_lib.clip_inputs(x)), [-5.0, -5.0, 0.0, 5.0, 5.0]
    )
