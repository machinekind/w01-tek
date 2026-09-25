"""The prepended flat row: its geometry, and that it is off unless asked for.

`terrain.generate` is numpy only -- no scene, no model, no device -- so the row
geometry is checked here. tests/integration/test_terrain.py owns everything
that needs a compiled scene (ray casts against the lookup grid, contacts).

The flat row is the one thing the difficulty ramp cannot give: at difficulty 0
the stairs still have a 2 cm riser.
"""

import numpy as np
import pytest

from wojtek_rl import terrain, terrain_env, terrain_suite

# Grid rows one arena row covers, and the y shift a prepended row puts on every
# row above it: the arena stays centred on the origin, so it grows half a tile
# at each end.
ROW_CELLS = int(round(terrain.TILE_SIZE / terrain.CELL_SIZE))
ROW_SHIFT = terrain.TILE_SIZE / 2


@pytest.fixture(scope="module")
def plain():
    return terrain.generate()


@pytest.fixture(scope="module")
def flat():
    return terrain.generate(flat_row=True)


def _rows(arena):
    """{row: {type: (col, difficulty, pad_height)}} -- a row's whole content."""
    out: dict[int, dict] = {}
    for t in arena.spec.tiles:
        out.setdefault(t.row, {})[t.terrain_type] = (t.col, t.difficulty, t.pad_height)
    return out


def _box_params(arena):
    return np.array([b.pos + b.half + (b.yaw,) for b in arena.boxes])


def test_the_flat_row_is_flat_ground(flat):
    for tile in (t for t in flat.spec.tiles if t.row == 0):
        cx, cy, _ = tile.origin
        half = flat.spec.tile_size / 2
        gx, gy = np.meshgrid(
            np.linspace(cx - half, cx + half, 40),
            np.linspace(cy - half, cy + half, 40),
        )
        h = terrain.lookup_height(flat, gx.ravel(), gy.ravel())
        assert np.all(h == 0.0), f"{tile.terrain_type} tile is not flat"
        assert tile.pad_height == 0.0
        assert tile.difficulty == 0.0
    # Every type once, like any row: the spawn table is indexed by (row, type)
    # and the env draws a type uniformly.
    assert sorted(t.terrain_type for t in flat.spec.tiles if t.row == 0) == sorted(
        terrain.TYPES
    )


def test_no_box_reaches_over_the_flat_row(flat):
    top = flat.spec.y_min + flat.spec.border + flat.spec.tile_size
    for b in flat.boxes:
        hx, hy, _ = b.half
        reach = hx * abs(np.sin(b.yaw)) + hy * abs(np.cos(b.yaw))
        assert b.pos[1] - reach > top


def test_the_flat_row_adds_one_level_and_shifts_the_rest(plain, flat):
    assert flat.spec.n_rows == plain.spec.n_rows + 1
    assert flat.spec.hfield.nrow == plain.spec.hfield.nrow + ROW_CELLS
    assert flat.spec.hfield.ncol == plain.spec.hfield.ncol
    assert len(flat.spec.tiles) == len(plain.spec.tiles) + len(terrain.TYPES)

    # Nothing is lost: every terrain row keeps its column order, difficulty and
    # pads, one level higher.
    before, after = _rows(plain), _rows(flat)
    for row in range(plain.spec.n_rows):
        assert after[row + 1] == before[row]
    # ... and the same surface, one tile higher on the grid.
    assert np.array_equal(flat.hfield_data[ROW_CELLS:], plain.hfield_data)
    shifted = _box_params(plain)
    shifted[:, 1] += ROW_SHIFT
    assert np.allclose(_box_params(flat), shifted, rtol=0, atol=1e-12)


def test_the_flag_off_builds_the_arena_it_always_built(plain):
    off = terrain.generate(flat_row=False)
    assert np.array_equal(off.lookup, plain.lookup)
    assert np.array_equal(off.hfield_data, plain.hfield_data)
    assert terrain.spec_to_dict(off.spec) == terrain.spec_to_dict(plain.spec)
    # The key is absent, not false, so the written spec stays byte-identical to
    # one built before the flag existed.
    assert "flat_row" not in terrain.spec_to_dict(plain.spec)
    assert terrain.spec_to_dict(terrain.generate(flat_row=True).spec)["flat_row"]


def test_the_measurement_arena_has_no_flat_row():
    eval_arena = terrain.generate(**terrain_suite.eval_arena_kwargs())
    assert eval_arena.spec.n_rows == len(terrain_suite.DIFFICULTIES)
    assert "flat_row" not in terrain.spec_to_dict(eval_arena.spec)


@pytest.mark.parametrize("spec", [{}, {"flat_row": False}])
def test_require_flat_row_accepts_a_matching_pair(spec):
    terrain_env.require_flat_row(spec, "train", False)
    terrain_env.require_flat_row({"flat_row": True}, "train", True)


def test_require_flat_row_refuses_a_mismatched_pair():
    with pytest.raises(ValueError, match="--flat-row"):
        terrain_env.require_flat_row({}, "train", True)
    with pytest.raises(ValueError, match="built with the flat row"):
        terrain_env.require_flat_row({"flat_row": True}, "train", False)


def test_require_flat_row_ignores_the_flag_on_eval():
    # The flag describes the training arena. Scoring a flat-row run on the
    # eval course keeps the run's terrain.flat_row=true in its config, and
    # the guard must not refuse the load.
    terrain_env.require_flat_row({}, "eval", True)
    terrain_env.require_flat_row({}, "eval", False)
    with pytest.raises(ValueError, match="built with the flat row"):
        terrain_env.require_flat_row({"flat_row": True}, "eval", True)


def test_flat_row_key_overrides_onto_default_config():
    # The preset path: task.env.terrain.flat_row must exist in default_config
    # or registry._apply_overrides dies on getattr at env construction.
    from wojtek_rl import env as wojtek_env
    from wojtek_rl import registry

    cfg = wojtek_env.default_config()
    assert cfg.terrain.flat_row is False
    registry._apply_overrides(cfg, {"terrain": {"flat_row": True}})
    assert cfg.terrain.flat_row is True
