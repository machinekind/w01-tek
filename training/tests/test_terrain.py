import json

import mujoco
import numpy as np
import pytest

from wojtek_rl import build_terrain, paths, terrain


@pytest.fixture(scope="module")
def arena():
    a = terrain.generate(seed=0)
    build_terrain.write_arena(a)  # deterministic; refreshes the scene artifacts
    return a


@pytest.fixture(scope="module")
def model(arena):
    return mujoco.MjModel.from_xml_path(str(paths.TERRAIN_SCENE_XML))


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


def test_determinism(arena):
    a2 = terrain.generate(seed=0)
    assert np.array_equal(arena.lookup, a2.lookup)
    assert np.array_equal(arena.hfield_data, a2.hfield_data)
    assert terrain.spec_to_dict(arena.spec) == terrain.spec_to_dict(a2.spec)
    # A different seed changes the random tiles (rough/discrete) but not the grid.
    a3 = terrain.generate(seed=1)
    assert not np.array_equal(arena.lookup, a3.lookup)
    assert terrain.spec_to_dict(arena.spec)["tiles"] == terrain.spec_to_dict(a3.spec)["tiles"]


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
    assert terrain.rough_amplitude(0) <= 0.01
    assert terrain.slope_angle(0) == 0.0
    assert terrain.stair_riser(0) <= 0.02
    assert terrain.discrete_max_height(0) <= 0.01

    # Realized relief from the lookup grid grows with difficulty. Slopes and
    # stairs are deterministic in d, so require strict monotonicity; rough and
    # discrete draw random heights, so only require row 0 << row 9.
    for ttype in ("pyramid_slope", "inverted_pyramid_slope",
                  "pyramid_stairs", "inverted_pyramid_stairs"):
        relief = [_tile_relief(arena, _tile(arena, ttype, r)) for r in rows]
        assert all(np.diff(relief) > -1e-6), (ttype, relief)
    for ttype in ("rough_uniform", "discrete_obstacles"):
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

    # Box-top centres sampled across the arena (covers the discrete obstacles).
    rng = np.random.default_rng(0)
    for b in [arena.boxes[i] for i in rng.integers(0, len(arena.boxes), 40)]:
        px, py, _ = b.pos
        top = b.pos[2] + b.half[2]
        zr = _ray_down(model, data, px, py)
        zl = float(terrain.lookup_height(arena, px, py))
        assert zr is not None and abs(zr - zl) < 0.012, (b, zr, zl)
        assert abs(zr - top) < 0.012


def test_spawn_pads_flat(arena):
    """Lookup variance within each spawn pad is below a tight epsilon."""
    for t in arena.spec.tiles:
        cx, cy, _ = t.origin
        r = np.linspace(0.0, t.pad_radius, 8)
        ang = np.linspace(0.0, 2 * np.pi, 32)
        xs = cx + np.outer(r, np.cos(ang)).ravel()
        ys = cy + np.outer(r, np.sin(ang)).ravel()
        h = terrain.lookup_height(arena, xs, ys)
        assert h.std() < 1e-3, (t.terrain_type, t.row, float(h.std()))
        assert abs(float(h.mean()) - t.pad_height) < 1e-3


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
    the edge does not hit a wall or cliff. Discrete boxes are held off the edge,
    so those borders are flat too (loose bound kept as a documented safety net)."""
    for t in arena.spec.tiles:
        m = _perimeter_max_abs(arena, t)
        if t.terrain_type in ("pyramid_slope", "inverted_pyramid_slope",
                               "pyramid_stairs", "inverted_pyramid_stairs"):
            bound = 0.02
        elif t.terrain_type == "rough_uniform":
            bound = terrain.rough_amplitude(t.difficulty) + 0.01
        else:
            bound = terrain.discrete_max_height(t.difficulty) + 1e-3
        assert m <= bound, (t.terrain_type, t.row, m, bound)


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
    assert paths.TERRAIN_HFIELD.exists()
    assert paths.TERRAIN_SCENE_XML.exists()
    meta = json.loads(paths.TERRAIN_SPEC_JSON.read_text())
    assert meta["seed"] == 0 and len(meta["tiles"]) == arena.spec.n_rows * arena.spec.n_cols
    npz = np.load(paths.TERRAIN_LOOKUP_NPZ)
    assert npz["lookup"].shape == (arena.spec.hfield.nrow, arena.spec.hfield.ncol)
    assert np.array_equal(npz["lookup"], arena.lookup)
