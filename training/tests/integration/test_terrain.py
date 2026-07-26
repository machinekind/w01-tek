import json

import mujoco
import numpy as np
import pytest

from wojtek_rl import build_terrain, paths, terrain


TEST_ARENA = "test"


@pytest.fixture(scope="module")
def arena():
    """A generated arena written to the `test` file set, never the one a policy
    trains on, and removed afterwards. Writing over the training arena is how a
    later training run silently collapsed its curriculum to three rungs with no
    record in run.json.

    The `test` file set is one shared location, so two pytest processes running
    at once (a second suite in another shell, or -p xdist) will fight over it.
    `run.sh test` is a single process."""
    a = terrain.generate(seed=0)
    build_terrain.write_arena(a, TEST_ARENA)
    yield a
    for p in paths.terrain_paths(TEST_ARENA).values():
        p.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def model(arena):
    return mujoco.MjModel.from_xml_path(
        str(paths.terrain_paths(TEST_ARENA)["scene"])
    )


def _tile(arena, ttype, row):
    return next(t for t in arena.spec.tiles if t.terrain_type == ttype and t.row == row)


def _ray_down(model, data, x, y):
    pnt = np.array([x, y, 5.0])
    vec = np.array([0.0, 0.0, -1.0])
    gid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, pnt, vec, None, 1, -1, gid)
    return 5.0 - dist if dist >= 0 else None


def _tile_relief(arena, tile):
    """Peak-to-peak surface height over a tile interior, from the lookup grid.

    Sampled inside a margin so the footprint never bleeds into a neighbour tile
    across the shared edge."""
    s = arena.spec
    cx, cy, _ = tile.origin
    reach = s.tile_size / 2 - 0.2
    xs = np.linspace(cx - reach, cx + reach, 40)
    ys = np.linspace(cy - reach, cy + reach, 40)
    gx, gy = np.meshgrid(xs, ys)
    h = terrain.lookup_height(arena, gx.ravel(), gy.ravel())
    return float(np.ptp(h))


def _layout(a):
    return [(t.row, t.col, t.terrain_type) for t in a.spec.tiles]


def test_determinism(arena):
    a2 = terrain.generate(seed=0)
    assert np.array_equal(arena.lookup, a2.lookup)
    assert np.array_equal(arena.hfield_data, a2.hfield_data)
    assert terrain.spec_to_dict(arena.spec) == terrain.spec_to_dict(a2.spec)
    # The default layout shuffles each row, but every row is still a full
    # permutation of the types, so (type, row) selection stays unique.
    for r in range(arena.spec.n_rows):
        row = sorted(t.terrain_type for t in arena.spec.tiles if t.row == r)
        assert row == sorted(terrain.TYPES)
    # A different seed reshuffles the layout and redraws the tiles.
    a3 = terrain.generate(seed=1)
    assert not np.array_equal(arena.lookup, a3.lookup)
    assert _layout(arena) != _layout(a3)


def test_ordered_layout_is_legacy(arena):
    """ordered=True restores the sorted-column, exact-nominal-difficulty arena."""
    a = terrain.generate(seed=0, ordered=True)
    s = a.spec
    assert s.ordered is True
    assert terrain.spec_to_dict(s)["ordered"] is True
    for r in range(s.n_rows):
        cols = sorted((t for t in s.tiles if t.row == r), key=lambda t: t.col)
        assert [t.terrain_type for t in cols] == list(terrain.TYPES)
        assert all(t.difficulty == r / (s.n_rows - 1) for t in cols)
    # The shuffled default differs from the legacy layout.
    assert _layout(a) != _layout(arena)


def test_row_difficulty_jittered_but_monotone(arena):
    """Interior rows are jittered off nominal; row 0 and the last stay exact and
    realized difficulty is still strictly increasing across rows."""
    s = arena.spec
    ds = [next(t.difficulty for t in s.tiles if t.row == r) for r in range(s.n_rows)]
    assert ds[0] == 0.0 and ds[-1] == 1.0
    assert all(np.diff(ds) > 0)
    assert any(d != r / (s.n_rows - 1) for r, d in enumerate(ds))


def test_ramp_inverses_round_trip():
    """The measurement suite asks for physical dimensions and inverts these
    ramps to get the difficulty that realizes them, so each pair has to be an
    exact round trip or the suite's row table means something else than the
    terrain it describes."""
    pairs = [
        (terrain.rough_amplitude, terrain.rough_difficulty, (0.01, 0.025, 0.04)),
        (terrain.slope_angle, terrain.slope_difficulty, (0.14, 0.26, 0.38)),
        (terrain.stair_riser, terrain.stair_difficulty, (0.03, 0.05, 0.07, 0.09)),
        (terrain.discrete_max_height, terrain.discrete_difficulty,
         (0.025, 0.05, 0.065, 0.08)),
        (terrain.wave_amplitude, terrain.wave_difficulty, (0.02, 0.04, 0.06)),
    ]
    for ramp, inverse, targets in pairs:
        for target in targets:
            assert ramp(inverse(target)) == pytest.approx(target, abs=1e-12)
            # and the other way round, over the frontier rows too
            for d in (0.0, 0.5, 1.0, 1.4):
                assert inverse(ramp(d)) == pytest.approx(d, abs=1e-12)


def test_six_step_stair_flight(arena):
    """Six treads, and they end inside the tile with room for the crossing
    radius the measurement course walks out to."""
    assert terrain.N_STEPS == 6
    assert terrain.stair_pit_half() == pytest.approx(1.25)
    assert terrain.stair_pit_half() < arena.spec.tile_size / 2
    # 0.78 m of run, against a 0.514 m front-to-rear foot spacing: there is a
    # window with all four feet on treads. Four treads would be one wheelbase.
    assert terrain.N_STEPS * terrain.TREAD > 0.514
    # The base box has to sit below the deepest pit the hardest row digs.
    deepest = terrain.N_STEPS * terrain.stair_riser(max(1.4, 1.0))
    assert terrain.HFIELD_BASE_Z > deepest, (terrain.HFIELD_BASE_Z, deepest)


def test_grid_layout(arena):
    s = arena.spec
    assert s.n_cols == len(terrain.TYPES)
    assert len(s.tiles) == s.n_rows * s.n_cols
    # Every tile has a spawn pad of at least 0.5 m radius.
    assert all(t.pad_radius >= 0.5 for t in s.tiles)


def test_difficulty_monotonic_and_row0_flat(arena):
    rows = list(range(arena.spec.n_rows))
    ds = [r / (arena.spec.n_rows - 1) for r in rows]
    # Analytic difficulty schedules: strictly increasing, near-flat at row 0.
    assert all(np.diff([terrain.rough_amplitude(d) for d in ds]) > 0)
    assert all(np.diff([terrain.slope_angle(d) for d in ds]) > 0)
    assert all(np.diff([terrain.stair_riser(d) for d in ds]) > 0)
    assert all(np.diff([terrain.discrete_max_height(d) for d in ds]) > 0)
    assert all(np.diff([terrain.wave_amplitude(d) for d in ds]) > 0)
    assert terrain.rough_amplitude(0) <= 0.01
    assert terrain.slope_angle(0) == 0.0
    assert terrain.stair_riser(0) <= 0.02
    assert terrain.discrete_max_height(0) <= 0.01
    assert terrain.wave_amplitude(0) <= 0.01

    # Realized relief from the lookup grid grows with difficulty (the realized,
    # per-row-jittered d, which is strictly increasing). Slope and stair relief
    # is dominated by the analytic ramp/step, so it stays strictly monotone even
    # though slopes now carry the small rough overlay; rough, discrete,
    # random_grid and wave draw more randomness, so only require r0 << r9. Rows 0
    # and 9 keep exact nominal d (0 and 1), so their bounds are unaffected.
    for ttype in ("pyramid_slope", "inverted_pyramid_slope",
                  "pyramid_stairs", "inverted_pyramid_stairs"):
        relief = [_tile_relief(arena, _tile(arena, ttype, r)) for r in rows]
        assert all(np.diff(relief) > -1e-6), (ttype, relief)
    for ttype in ("rough_uniform", "discrete_obstacles", "random_grid", "wave"):
        r0 = _tile_relief(arena, _tile(arena, ttype, 0))
        r9 = _tile_relief(arena, _tile(arena, ttype, 9))
        assert r0 < r9
        assert r0 <= 0.03, (ttype, r0)


def test_scene_compiles(model, arena):
    assert model.key("home").id >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track") >= 0
    hf = model.geom("terrain_hfield")
    assert model.geom_type[hf.id] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert hf.conaffinity[0] == 15 and model.geom_condim[hf.id] == 3
    assert model.hfield_nrow[0] == arena.spec.hfield.nrow
    assert model.hfield_ncol[0] == arena.spec.hfield.ncol
    assert (model.hfield_size[0] > 0).all()
    n_boxes = sum(
        1 for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("terrain_box_")
    )
    assert n_boxes == len(arena.boxes)


def test_terrain_pairs_like_the_floor(model, arena):
    # Every terrain geom carries the floor's collision semantics so it pairs
    # with feet (contype 1/conaff 1), legs (2/0) and base (1/15) as the floor does.
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name == "terrain_hfield" or name.startswith("terrain_box_"):
            assert model.geom_contype[i] == 1
            assert model.geom_conaffinity[i] == 15
            assert model.geom_condim[i] == 3


def test_lookup_matches_geometry(model, arena):
    """Lookup grid vs mj_ray straight down on box tiles and hfield tiles."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    s = arena.spec

    # Pad centres of every tile (flat, node-aligned): tight tolerance.
    for t in s.tiles:
        cx, cy, _ = t.origin
        if abs(cx) < 0.6 and abs(cy) < 0.6:
            continue  # skip the tile corner under the home-keyframe robot
        zr = _ray_down(model, data, cx, cy)
        zl = float(terrain.lookup_height(arena, cx, cy))
        assert zr is not None and abs(zr - zl) < 0.01, (t.terrain_type, zr, zl)

    # A tread centre on every stair tile (interior of a box top face).
    for ttype in ("pyramid_stairs", "inverted_pyramid_stairs"):
        for r in range(s.n_rows):
            t = _tile(arena, ttype, r)
            cx, cy, _ = t.origin
            y = cy + terrain.STAIR_PLATFORM_HALF + 0.5 * terrain.TREAD
            zr = _ray_down(model, data, cx, y)
            zl = float(terrain.lookup_height(arena, cx, y))
            assert zr is not None and abs(zr - zl) < 0.012, (ttype, r, zr, zl)

    # A ramp point on every slope tile (interior of the heightfield).
    for ttype in ("pyramid_slope", "inverted_pyramid_slope"):
        for r in range(s.n_rows):
            t = _tile(arena, ttype, r)
            cx, cy, _ = t.origin
            x = cx + terrain.SLOPE_PLATFORM_HALF + 0.5
            zr = _ray_down(model, data, x, cy)
            zl = float(terrain.lookup_height(arena, x, cy))
            assert zr is not None and abs(zr - zl) < 0.02, (ttype, r, zr, zl)

    # Box-top centres sampled across the arena (covers discrete + yawed rubble).
    rng = np.random.default_rng(0)
    sampled = [arena.boxes[i] for i in rng.integers(0, len(arena.boxes), 40)]
    assert any(b.yaw != 0.0 for b in sampled)  # rotated boxes are in the mix
    for b in sampled:
        px, py, _ = b.pos
        top = b.pos[2] + b.half[2]
        zr = _ray_down(model, data, px, py)
        zl = float(terrain.lookup_height(arena, px, py))
        assert zr is not None and abs(zr - zl) < 0.012, (b, zr, zl)
        assert abs(zr - top) < 0.012


def pad_flatness(arena, spawn_jitter):
    """(worst height spread, worst offset from the declared pad height) over the
    ground a spawned robot actually stands on: its footprint reach, displaced by
    the worst spawn jitter, capped by the pad.

    Not the pad rim. A stair tile's platform half-size equals its pad radius, so
    the rim node IS the first tread -- correct geometry, and not something a
    spawn ever stands on (0.15 m jitter plus a 0.36 m footprint reaches 0.51 m
    of the 0.60 m platform)."""
    worst_std, worst_off = 0.0, 0.0
    for t in arena.spec.tiles:
        cx, cy, _ = t.origin
        reach = min(t.pad_radius, terrain.FOOTPRINT_REACH + spawn_jitter)
        r = np.linspace(0.0, reach, 8)
        ang = np.linspace(0.0, 2 * np.pi, 32)
        xs = cx + np.outer(r, np.cos(ang)).ravel()
        ys = cy + np.outer(r, np.sin(ang)).ravel()
        h = terrain.lookup_height(arena, xs, ys)
        worst_std = max(worst_std, float(h.std()))
        worst_off = max(worst_off, abs(float(h.mean()) - t.pad_height))
    return worst_std, worst_off


def test_spawn_pads_flat(arena):
    """Every spawn stands on flat ground at the tile's declared pad height,
    at the training arena's 0.15 m spawn jitter."""
    spread, offset = pad_flatness(arena, spawn_jitter=0.15)
    assert spread < 1e-3, spread
    assert offset < 1e-3, offset


def _perimeter_max_abs(arena, tile, inset=0.02, n=25):
    """Max |surface height| sampled along the four tile edges, small inset in."""
    s = arena.spec
    cx, cy, _ = tile.origin
    h = s.tile_size / 2 - inset
    lin = np.linspace(-h, h, n)
    xs = np.concatenate([cx + lin, cx + lin, cx - h * np.ones(n), cx + h * np.ones(n)])
    ys = np.concatenate([cy - h * np.ones(n), cy + h * np.ones(n), cy + lin, cy + lin])
    return float(np.max(np.abs(terrain.lookup_height(arena, xs, ys))))


def test_tile_borders_seamless(arena):
    """Every tile reaches ~flat ground at its border, so a robot promoted across
    the edge does not hit a wall or cliff. Rough and the modality overlay now
    taper to 0 at the rim, so rough holds the same tight bound as slopes and
    stairs. Discrete boxes are held off the edge (loose bound kept as a net)."""
    for t in arena.spec.tiles:
        m = _perimeter_max_abs(arena, t)
        if t.terrain_type in ("pyramid_slope", "inverted_pyramid_slope",
                               "pyramid_stairs", "inverted_pyramid_stairs",
                               "wave", "rough_uniform"):
            bound = 0.02
        else:  # discrete_obstacles, random_grid: boxes are held off the edge
            bound = terrain.discrete_max_height(t.difficulty) + 1e-3
        assert m <= bound, (t.terrain_type, t.row, m, bound)


def test_arena_seams_no_step(arena):
    """No one-cell height step above 2.5 cm across any internal tile border.
    Scans lookup nodes within 2 cells of each seam line, so it never crosses a
    stair-pit interior or a box top (both held well inside their tiles)."""
    s = arena.spec
    L = arena.lookup
    xs = np.linspace(s.x_min, s.x_max, s.hfield.ncol)
    ys = np.linspace(s.y_min, s.y_max, s.hfield.nrow)
    grid_x0 = -s.n_cols * s.tile_size / 2
    grid_y0 = -s.n_rows * s.tile_size / 2
    band = 2 * s.cell_size + 1e-9
    worst = 0.0
    # Vertical seams (between columns): horizontal one-cell steps in the band.
    for j in range(1, s.n_cols):
        cols = np.nonzero(np.abs(xs - (grid_x0 + j * s.tile_size)) <= band)[0]
        block = L[:, cols.min():cols.max() + 1]
        worst = max(worst, float(np.abs(np.diff(block, axis=1)).max()))
    # Horizontal seams (between rows): vertical one-cell steps in the band.
    for i in range(1, s.n_rows):
        rows = np.nonzero(np.abs(ys - (grid_y0 + i * s.tile_size)) <= band)[0]
        block = L[rows.min():rows.max() + 1, :]
        worst = max(worst, float(np.abs(np.diff(block, axis=0)).max()))
    assert worst <= 0.025, worst


def test_rotated_box_top_matches_ray(model, arena):
    """At least one box is yawed. Its rasterised lookup and mj_ray straight down
    agree at the box centre, so the rotated-rect rasterisation matches physics.
    Where a yawed box is the local surface, that shared height is its own top
    (discrete boxes may overlap, so a taller neighbour can cap the centre)."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    yawed = [b for b in arena.boxes if b.yaw != 0.0]
    assert yawed
    checked_top = False
    for b in yawed[:: max(1, len(yawed) // 20)]:
        px, py, _ = b.pos
        top = b.pos[2] + b.half[2]
        zr = _ray_down(model, data, px, py)
        zl = float(terrain.lookup_height(arena, px, py))
        assert zr is not None and abs(zr - zl) < 0.012, (b, zr, zl)
        if abs(zr - top) < 0.012:
            checked_top = True
    assert checked_top


def test_physics_smoke_on_rough_pad(model, arena):
    """Hold the home pose at a rough spawn pad; no NaNs, no fall-through."""
    tile = _tile(arena, "rough_uniform", 5)
    key = model.key("home")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key.id)
    data.qpos[0] = tile.origin[0]
    data.qpos[1] = tile.origin[1]
    data.qpos[2] = key.qpos[2] + tile.pad_height
    data.ctrl[:] = key.ctrl
    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert not np.isnan(data.qpos).any()
    local = float(terrain.lookup_height(arena, data.qpos[0], data.qpos[1]))
    assert data.qpos[2] - local > 0.05, (data.qpos[2], local)


def test_sidecars_written(arena):
    files = paths.terrain_paths(TEST_ARENA)
    assert files["hfield"].exists()
    assert files["scene"].exists()
    meta = json.loads(files["spec"].read_text())
    assert meta["seed"] == 0 and len(meta["tiles"]) == arena.spec.n_rows * arena.spec.n_cols
    npz = np.load(files["lookup"])
    assert npz["lookup"].shape == (arena.spec.hfield.nrow, arena.spec.hfield.ncol)
    assert np.array_equal(npz["lookup"], arena.lookup)
