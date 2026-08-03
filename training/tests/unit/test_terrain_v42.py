"""The v4.2 generator parameters: treads, step counts, caps, feature radii.

`terrain.generate` is numpy only, so all of this runs without a scene or a
device. The one invariant that protects every published scan is pinned first:
the default arguments still build the arena the code always built, down to the
spec dict and the box list.
"""

import numpy as np

from wojtek_rl import terrain

STAIRS = ("pyramid_stairs", "inverted_pyramid_stairs")


def _stairs_tiles(arena):
    return [t for t in arena.spec.tiles if t.terrain_type in STAIRS]


def test_defaults_build_the_legacy_arena():
    a = terrain.generate(seed=3, n_rows=4)
    d = terrain.spec_to_dict(a.spec)
    # The extended keys stay off the wire, so a legacy spec is byte-identical.
    assert "stair_tread" not in d
    assert "type_caps" not in d
    assert all("feature_radius" not in t for t in d["tiles"])
    assert all(t.n_steps == terrain.N_STEPS for t in _stairs_tiles(a))
    assert all(t.stair_tread == terrain.TREAD for t in _stairs_tiles(a))


def test_stair_steps_round_trips_the_legacy_geometry():
    assert terrain.stair_steps(terrain.TREAD) == terrain.N_STEPS
    # Real treads on the legacy platform: three risers.
    assert terrain.stair_steps(0.30) == 3
    # A summit platform buys one more.
    assert terrain.stair_steps(0.30, stair_platform_half=0.3) == 4
    # The clamp floors at a two-tread flight.
    assert terrain.stair_steps(1.0) == 3


def test_tread_range_draws_per_tile_and_flight_fits_the_tile():
    a = terrain.generate(seed=0, n_rows=6, stair_tread=(0.25, 0.45),
                         pad_radius=0.3, stair_platform_half=0.3)
    tiles = _stairs_tiles(a)
    treads = np.array([t.stair_tread for t in tiles])
    assert (treads >= 0.25).all() and (treads <= 0.45).all()
    assert len(set(np.round(treads, 6))) > 1  # a draw, not one value
    for t in tiles:
        assert t.n_steps == terrain.stair_steps(t.stair_tread, 0.3)
        flight_edge = 0.3 + (t.n_steps - 1) * t.stair_tread
        assert flight_edge <= terrain.TILE_SIZE / 2 - terrain.RIM_MARGIN + 1e-9
        assert t.feature_radius == flight_edge


def test_type_caps_compress_the_column():
    caps = {"pyramid_stairs": 0.7, "random_grid": 0.55}
    a = terrain.generate(seed=0, n_rows=5, type_caps=caps)
    b = terrain.generate(seed=0, n_rows=5)
    for ta, tb in zip(a.spec.tiles, b.spec.tiles):
        cap = caps.get(ta.terrain_type, 1.0)
        assert np.isclose(ta.difficulty, tb.difficulty * cap)
    # The capped top row realizes the cap exactly (row difficulties end at 1).
    top = max(t.difficulty for t in a.spec.tiles
              if t.terrain_type == "pyramid_stairs")
    assert np.isclose(top, 0.7)


def test_feature_radius_per_type():
    a = terrain.generate(seed=1, n_rows=3, flat_row=True)
    ring = terrain.TILE_SIZE / 2 - terrain.DISCRETE_EDGE_MARGIN
    for t in a.spec.tiles:
        if t.row == 0:
            assert t.feature_radius == 0.0
        elif t.terrain_type in STAIRS:
            assert np.isclose(t.feature_radius, terrain.stair_pit_half())
        else:
            assert np.isclose(t.feature_radius, ring)


def test_extended_spec_round_trips_through_the_dict():
    a = terrain.generate(seed=0, n_rows=3, stair_tread=(0.25, 0.45),
                         type_caps={"pyramid_stairs": 0.7})
    d = terrain.spec_to_dict(a.spec)
    assert d["stair_tread"] == [0.25, 0.45]
    assert d["type_caps"] == {"pyramid_stairs": 0.7}
    for tile, td in zip(a.spec.tiles, d["tiles"]):
        assert td["feature_radius"] == tile.feature_radius
        assert td["stair_tread"] == tile.stair_tread
        assert td["n_steps"] == tile.n_steps
