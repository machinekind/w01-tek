"""The v4.2 curriculum rules: band promotion, strikes, grace, pinning.

`curriculum_step` is a pure function, so every rule is checked here without
an env or a device model. The legacy defaults are pinned first: with them the
function must behave exactly as it did before the v4.2 arguments existed.
"""

import jax
import jax.numpy as jp
import numpy as np

from wojtek_rl import terrain_env

N_ROWS = 10
TILE = 3.0
DEMOTE = 0.5
EPISODE = 1000


def _step(level, walked, commanded, steps_lived=EPISODE, key=0, **kw):
    lvl, strikes, _ = terrain_env.curriculum_step(
        jp.int32(level), jp.float32(walked), jp.float32(commanded),
        jp.int32(steps_lived), EPISODE,
        jax.random.PRNGKey(key), N_ROWS, TILE, DEMOTE, **kw,
    )
    return int(lvl), int(strikes)


def test_legacy_defaults_promote_and_demote_in_one_episode():
    assert _step(2, walked=2.0, commanded=3.0) == (3, 0)
    assert _step(5, walked=0.2, commanded=3.0) == (4, 0)
    assert _step(5, walked=0.0, commanded=0.0) == (5, 0)


def test_crossed_replaces_the_walk_rule():
    # A long diagonal stroll that never crossed the band: no promotion.
    lvl, _ = _step(5, walked=2.0, commanded=3.0, crossed=jp.array(False))
    assert lvl == 5
    # A crossing promotes even when the walked distance is short.
    lvl, _ = _step(5, walked=1.0, commanded=1.0, crossed=jp.array(True))
    assert lvl == 6


def test_three_strikes_demote_and_a_clean_walk_clears():
    fail = dict(walked=0.2, commanded=3.0, demote_strikes=3)
    lvl, s = _step(5, strikes=jp.int32(0), **fail)
    assert (lvl, s) == (5, 1)
    lvl, s = _step(5, strikes=jp.int32(1), **fail)
    assert (lvl, s) == (5, 2)
    lvl, s = _step(5, strikes=jp.int32(2), **fail)
    assert (lvl, s) == (4, 0)  # third strike drops and clears
    # A clean MOVEMENT episode clears the count without moving the level
    # (walked most of a modest commanded distance, no band crossing).
    lvl, s = _step(5, walked=1.0, commanded=1.2, strikes=jp.int32(2),
                   demote_strikes=3)
    assert (lvl, s) == (5, 0)


def test_stand_and_spin_episodes_are_strike_transparent():
    # A stand/spin episode (no commanded distance) neither clears nor
    # strikes: fall, spin, fall, spin, fall still demotes on the third fail.
    fail = dict(walked=0.2, commanded=3.0, demote_strikes=3)
    spin = dict(walked=0.02, commanded=0.0, demote_strikes=3)
    _, s = _step(5, strikes=jp.int32(0), **fail)
    assert s == 1
    lvl, s = _step(5, strikes=jp.int32(s), **spin)
    assert (lvl, s) == (5, 1)  # transparent, count held
    _, s = _step(5, strikes=jp.int32(s), **fail)
    assert s == 2
    lvl, s = _step(5, strikes=jp.int32(s), **spin)
    assert (lvl, s) == (5, 2)
    lvl, s = _step(5, strikes=jp.int32(s), **fail)
    assert (lvl, s) == (4, 0)


def test_grace_falls_are_neutral_and_keep_strikes():
    # A fall at step 30 of a 50-step grace: no strike, no move, count held.
    lvl, s = _step(5, walked=0.05, commanded=0.02, steps_lived=30,
                   strikes=jp.int32(2), demote_strikes=3, grace_steps=50)
    assert (lvl, s) == (5, 2)
    # One step past the grace the same episode strikes.
    lvl, s = _step(5, walked=0.05, commanded=0.02, steps_lived=51,
                   strikes=jp.int32(2), demote_strikes=3, grace_steps=50)
    assert (lvl, s) == (4, 0)


def test_pinned_envs_never_move():
    lvl, s = _step(7, walked=2.5, commanded=3.0, pinned=jp.array(True),
                   crossed=jp.array(True))
    assert (lvl, s) == (7, 0)
    lvl, s = _step(7, walked=0.1, commanded=3.0, pinned=jp.array(True),
                   strikes=jp.int32(2), demote_strikes=3)
    assert (lvl, s) == (7, 2)


def test_feature_spawn_stays_inside_the_tile():
    origin = jp.array([[[4.5, -1.5]]], dtype=jp.float32)
    pad_h = jp.array([[0.2]], dtype=jp.float32)
    keys = jax.random.split(jax.random.PRNGKey(1), 256)
    xy, ph, quat = jax.vmap(
        lambda k: terrain_env.sample_feature_spawn(
            k, jp.int32(0), jp.int32(0), jp.array(False),
            origin, pad_h, 0.15, TILE, True,
        )
    )(keys)
    r = np.max(np.abs(np.array(xy) - np.array([4.5, -1.5])), axis=1)
    assert r.max() <= TILE / 2 - terrain_env.SPAWN_EDGE_MARGIN + 1e-6
    # The draw covers the tile, not just the pad.
    assert r.max() > 0.8
    assert np.allclose(np.linalg.norm(np.array(quat), axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.array(ph), 0.2)


def test_feature_spawn_on_flat_keeps_the_pad_draw():
    origin = jp.array([[[0.0, 0.0]]], dtype=jp.float32)
    pad_h = jp.array([[0.0]], dtype=jp.float32)
    keys = jax.random.split(jax.random.PRNGKey(2), 128)
    xy, _, _ = jax.vmap(
        lambda k: terrain_env.sample_feature_spawn(
            k, jp.int32(0), jp.int32(0), jp.array(True),
            origin, pad_h, 0.15, TILE, True,
        )
    )(keys)
    assert np.abs(np.array(xy)).max() <= 0.15 + 1e-6
