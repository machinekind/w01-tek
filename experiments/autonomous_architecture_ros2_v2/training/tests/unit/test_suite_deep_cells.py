"""Suite version 2: the deep-tread stair cells and their arena.

The legacy suite is pinned untouched first -- its cells, version tag and
fingerprint are what every published scan compares against.
"""

from wojtek_rl import terrain, terrain_suite


def test_legacy_cells_and_fingerprint_are_untouched():
    assert terrain_suite.CELLS_VERSION == "v3"
    assert all(c.arena == "eval" for c in terrain_suite.CELLS)
    fp = terrain_suite.arena_fingerprint()
    assert "stair_tread" not in fp
    assert fp["n_steps"] == terrain.N_STEPS
    kwargs = terrain_suite.eval_arena_kwargs()
    assert "stair_tread" not in kwargs  # legacy arena stays byte-identical


def test_deep_cells_are_tracked_stairs_on_their_own_arena():
    assert len(terrain_suite.DEEP_CELLS) == 8
    for c in terrain_suite.DEEP_CELLS:
        assert c.arena == "eval_deep"
        assert c.tracked  # no plan bars for the new geometry yet
        assert c.terrain_type in ("pyramid_stairs", "inverted_pyramid_stairs")
        assert "deep" in c.name
        # The row indexes the shared difficulty table.
        assert abs(terrain_suite.DIFFICULTIES[c.row] - c.difficulty) < 1e-9
    names = {c.name for c in terrain_suite.DEEP_CELLS}
    assert "pyramid_stairs_deep_5cm" in names
    assert "inverted_pyramid_stairs_deep_9cm" in names


def test_deep_arena_kwargs_and_fingerprint():
    kwargs = terrain_suite.deep_arena_kwargs()
    assert kwargs["stair_tread"] == terrain_suite.DEEP_TREAD
    assert kwargs["difficulties"] == terrain_suite.DIFFICULTIES
    fp = terrain_suite.deep_arena_fingerprint()
    assert fp["stair_tread"] == terrain_suite.DEEP_TREAD
    assert fp["n_steps"] == terrain.stair_steps(terrain_suite.DEEP_TREAD)
    assert terrain_suite.SUITE_VERSION == 2


def test_all_cells_merge_without_name_collisions():
    assert len(terrain_suite.ALL_CELLS) == (
        len(terrain_suite.CELLS) + len(terrain_suite.DEEP_CELLS)
    )
    assert len(terrain_suite.ALL_CELLS_BY_NAME) == len(terrain_suite.ALL_CELLS)


def test_deep_arena_generates_and_check_arena_accepts_it():
    from wojtek_rl import terrain_scan

    arena = terrain.generate(**terrain_suite.deep_arena_kwargs())
    spec = terrain.spec_to_dict(arena.spec)
    terrain_scan.check_arena(spec, "eval_deep")
    # The legacy check must REFUSE the deep arena, and vice versa.
    legacy = terrain.generate(**terrain_suite.eval_arena_kwargs())
    legacy_spec = terrain.spec_to_dict(legacy.spec)
    terrain_scan.check_arena(legacy_spec, "eval")
    for spec_, kind in ((spec, "eval"), (legacy_spec, "eval_deep")):
        try:
            terrain_scan.check_arena(spec_, kind)
        except ValueError:
            continue
        raise AssertionError(f"check_arena({kind}) accepted the wrong arena")
