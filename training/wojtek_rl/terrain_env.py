"""The terrain arena at runtime: loaded files, height lookup, spawns, curriculum.

Everything reads the files written by ``./run.sh build-terrain``. No brax
imports here, so base.py can use it without the training stack.

``Arena.load`` owns all of it. An env holds one ``Arena`` (``env._terrain``) and
asks it for heights and spawn tables; nothing about reading the generated files,
validating them, or interpreting the config lives in the env classes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from wojtek_rl import paths, terrain


def bilinear_sample(grid, x_min, cell_x, y_min, cell_y, x, y):
    """Sample ``grid`` at world ``(x, y)`` on device: ``terrain._bilinear``
    bound to jax.numpy, so the sampler the physics-side lookups run through is
    the same body the generator and its ground-truth tests use."""
    return terrain._bilinear(grid, x_min, cell_x, y_min, cell_y, x, y, xp=jp)


@dataclass(frozen=True)
class Arena:
    """One loaded terrain arena, plus the curriculum settings from the config.

    Built by `load`. Everything here comes from the four generated files or from
    `task.env.terrain`, so an env can hold one of these and never touch the file
    layout itself.
    """

    kind: str  # train | eval | test
    files: dict
    lookup: jp.ndarray  # ground-truth surface height, on device
    x_min: float
    y_min: float
    cell_x: float
    cell_y: float
    origin_xy: jp.ndarray  # (n_rows, n_types, 2) tile centres
    pad_h: jp.ndarray  # (n_rows, n_types) spawn pad heights
    feature_r: jp.ndarray  # (n_rows, n_types) outer Chebyshev feature radii
    n_rows: int  # curriculum levels, the flat row included when there is one
    flat_row: bool  # true: level 0 is flat, every terrain row sits one higher
    n_types: int
    tile_size: float
    pad_radius: float  # the arena's flat pad radius, from its spec
    pad_jitter: float
    spawn_yaw: bool
    demote_fraction: float
    init_level_frac: float
    spawn_level: int
    spawn_mode: str  # pad | feature
    spawn_grace_sec: float
    pinned_frac: float
    demote_strikes: int

    def height(self, xy):
        """Terrain surface height under world ``xy`` (``(..., 2)``)."""
        return bilinear_sample(
            self.lookup, self.x_min, self.cell_x, self.y_min, self.cell_y,
            xy[..., 0], xy[..., 1],
        )


def require_assets(files: dict, kind: str) -> None:
    """All four generated files, heightfield included. The scene XML points at
    the heightfield binary by relative path, so a missing .bin is a raw MuJoCo
    compile error rather than the message below."""
    missing = [p for p in files.values() if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"terrain.enable=true but the generated terrain assets are missing "
            f"({names}). Run `./training/run.sh build-terrain --arena {kind}` "
            f"first; the sidecars are gitignored and built on demand."
        )


def require_current_geometry(spec: dict, kind: str) -> None:
    """Refuse an arena built by a different version of the generator.

    The generated files are self-consistent -- their lookup grid matches their
    own boxes -- so a stale arena raises nothing and trains fine. It just trains
    on terrain nobody asked for: an arena built before the stair flight went from
    four steps to six has four-step stairs, and the run's own run.json would say
    six. Rows, seed and tile size are per-experiment and not checked; the stair
    geometry is a code constant, so a mismatch is always staleness.
    """
    expected = {
        "n_steps": terrain.N_STEPS,
        "stair_platform_half": terrain.STAIR_PLATFORM_HALF,
    }
    stale = {
        key: (spec.get(key), value)
        for key, value in expected.items()
        if spec.get(key) != value
    }
    if stale:
        detail = "; ".join(
            f"{k}: arena has {f!r}, this code builds {w!r}" for k, (f, w) in stale.items()
        )
        extra = " [--rows N --seed N]` with this run's own parameters" if kind == "train" else "`"
        raise ValueError(
            f"the {kind} terrain arena was built by a different generator "
            f"({detail}). Rebuild it: `./training/run.sh build-terrain "
            f"--arena {kind}{extra}"
        )


def require_stair_geometry(spec: dict, kind: str, terrain_cfg) -> None:
    """Refuse an arena whose stair treads, caps or pad differ from the run's.

    These are baked into the generated files the way the flat row is, so the
    build and the preset have to agree. The config's None means the legacy
    defaults, which an arena spec expresses by omitting the keys (the legacy
    measurement arena stays byte-identical that way).

    The measurement arenas are their own definition (terrain_suite), so like
    the flat row this check applies to the arena a policy trains on, not the
    one it is measured on."""
    if kind in ("eval", "eval_deep"):
        return
    want_tread = terrain_cfg.get("stair_tread_range", None)
    want_caps = terrain_cfg.get("type_caps", None)
    want_pad = terrain_cfg.get("pad_radius", None)
    has_tread = spec.get("stair_tread", None)
    has_caps = spec.get("type_caps", None)
    has_pad = float(spec["pad_radius"])
    problems = []
    want_tread_l = list(want_tread) if want_tread is not None else None
    if want_tread_l != (list(has_tread) if isinstance(has_tread, list) else has_tread):
        problems.append(f"stair_tread: arena has {has_tread!r}, run wants {want_tread_l!r}")
    want_caps_d = dict(want_caps) if want_caps is not None else None
    if want_caps_d != has_caps:
        problems.append(f"type_caps: arena has {has_caps!r}, run wants {want_caps_d!r}")
    if want_pad is not None and abs(float(want_pad) - has_pad) > 1e-9:
        problems.append(f"pad_radius: arena has {has_pad}, run wants {want_pad}")
    if problems:
        raise ValueError(
            f"the {kind} terrain arena disagrees with the run's stair "
            f"geometry ({'; '.join(problems)}). Rebuild it: "
            f"`./training/run.sh build-terrain --arena {kind}` with this "
            f"run's geometry flags, or fix the preset to match the arena."
        )


def require_flat_row(spec: dict, kind: str, want: bool) -> None:
    """Refuse an arena whose flat row does not match what the run asked for.

    The flat row is baked into the generated files, so the arena build and the
    experiment preset have to agree. Neither mismatch shows up on its own:
    without the row the run trains a level short of its configuration, and with
    a row nobody asked for every logged level names a different terrain.

    The flag describes the training arena. The measurement arenas never
    carry the row (build-terrain refuses to add it), so eval loads ignore
    the run's flag: a flat-row policy still scores on the standard course.
    """
    if kind in ("eval", "eval_deep"):
        want = False
    has = bool(spec.get("flat_row", False))
    if has == want:
        return
    flag = " --flat-row" if want else ""
    built = "with" if has else "without"
    raise ValueError(
        f"terrain.flat_row={want} but the {kind} arena was built {built} the "
        f"flat row. Rebuild it: `./training/run.sh build-terrain "
        f"--arena {kind}{flag}`, or set terrain.flat_row={has} to train on the "
        f"arena as built."
    )


def load(terrain_cfg) -> Arena:
    """Read one arena's generated files and the curriculum settings around them.

    Config values are read by attribute on purpose: the defaults live in
    `default_config` only, so a missing key should fail loud rather than fall
    back on a second, quieter default here.
    """
    kind = str(terrain_cfg.arena)
    files = paths.terrain_paths(kind)
    require_assets(files, kind)

    npz = np.load(files["lookup"])
    x_min, x_max = float(npz["x_min"]), float(npz["x_max"])
    y_min, y_max = float(npz["y_min"]), float(npz["y_max"])
    ncol, nrow = int(npz["ncol"]), int(npz["nrow"])

    spec = json.loads(Path(files["spec"]).read_text())
    require_current_geometry(spec, kind)
    # Read with .get, unlike the fields below: an arena and a config from
    # before the flat row both mean "no flat row", so absence is an answer
    # here rather than a missing default.
    want_flat = bool(terrain_cfg.get("flat_row", False))
    require_flat_row(spec, kind, want_flat)
    require_stair_geometry(spec, kind, terrain_cfg)
    # The arena's own truth, not the run's flag: an eval load under a
    # flat-row run carries no row.
    flat_row = bool(spec.get("flat_row", False))
    origin_xy, pad_h, feature_r = tables_from_spec(spec, terrain.TYPES)

    # A coarse fit check for pad-confined spawns: the spawn scatter plus the
    # standing footprint has to fit the arena's flat pad, or spawns start
    # with feet on the features. The linear sum ignores corner draws, but the
    # pad's taper band absorbs that fringe; what this catches is a category
    # error -- training with the default 0.15 m jitter on the eval arena's
    # deliberately small 0.40 m pads. Feature spawns read the ground height
    # under the spawn point instead, so the pad owes them nothing.
    spawn_mode = str(terrain_cfg.get("spawn_mode", "pad"))
    if spawn_mode not in ("pad", "feature"):
        raise ValueError(f"terrain.spawn_mode={spawn_mode!r}: use 'pad' or 'feature'")
    pad_jitter = float(terrain_cfg.pad_jitter)
    pad_radius = float(spec["pad_radius"])
    if spawn_mode == "pad" and pad_jitter + terrain.FOOTPRINT_REACH > pad_radius:
        raise ValueError(
            f"terrain.pad_jitter={pad_jitter} scatters spawns off the {kind} "
            f"arena's flat pad: jitter + the {terrain.FOOTPRINT_REACH} m "
            f"standing footprint exceeds the {pad_radius} m pad radius. "
            f"Lower pad_jitter or rebuild with a larger --pad-radius."
        )

    return Arena(
        kind=kind,
        files=files,
        lookup=jp.asarray(npz["lookup"], dtype=jp.float32),
        x_min=x_min,
        y_min=y_min,
        cell_x=(x_max - x_min) / (ncol - 1),
        cell_y=(y_max - y_min) / (nrow - 1),
        origin_xy=jp.asarray(origin_xy),
        pad_h=jp.asarray(pad_h),
        feature_r=jp.asarray(feature_r),
        n_rows=int(spec["n_rows"]),
        flat_row=flat_row,
        n_types=len(terrain.TYPES),
        tile_size=float(spec["tile_size"]),
        pad_radius=pad_radius,
        pad_jitter=pad_jitter,
        spawn_yaw=bool(terrain_cfg.spawn_yaw),
        demote_fraction=float(terrain_cfg.demote_fraction),
        init_level_frac=float(terrain_cfg.init_level_frac),
        spawn_level=int(terrain_cfg.get("spawn_level", -1)),
        spawn_mode=spawn_mode,
        spawn_grace_sec=float(terrain_cfg.get("spawn_grace_sec", 0.0)),
        pinned_frac=float(terrain_cfg.get("pinned_frac", 0.0)),
        demote_strikes=int(terrain_cfg.get("demote_strikes", 1)),
    )


def tables_from_spec(spec: dict, types) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile-centre xy, pad height and feature radius for every (row, type) pair.

    Indexed by type, not column, because each row shuffles its column order.
    Shapes: ``(n_rows, n_types, 2)`` and ``(n_rows, n_types)`` twice.

    Legacy specs carry no per-tile ``feature_radius``; the fallback rebuilds
    it from the spec's own stair constants, matching what the generator would
    have written (the flat row, when present, has no band and reads 0)."""
    n_rows = int(spec["n_rows"])
    type_idx = {t: i for i, t in enumerate(types)}
    origin_xy = np.zeros((n_rows, len(types), 2), dtype=np.float32)
    pad_h = np.zeros((n_rows, len(types)), dtype=np.float32)
    feature_r = np.zeros((n_rows, len(types)), dtype=np.float32)
    stair_fallback = terrain.stair_pit_half(
        float(spec["stair_platform_half"]), terrain.TREAD, int(spec["n_steps"])
    )
    ring_fallback = float(spec["tile_size"]) / 2 - terrain.DISCRETE_EDGE_MARGIN
    flat_row = bool(spec.get("flat_row", False))
    for t in spec["tiles"]:
        r = int(t["row"])
        j = type_idx[t["type"]]
        origin_xy[r, j] = (t["origin"][0], t["origin"][1])
        pad_h[r, j] = t["pad_height"]
        if "feature_radius" in t:
            feature_r[r, j] = t["feature_radius"]
        elif flat_row and r == 0:
            feature_r[r, j] = 0.0
        elif t["type"] in ("pyramid_stairs", "inverted_pyramid_stairs"):
            feature_r[r, j] = stair_fallback
        else:
            feature_r[r, j] = ring_fallback
    return origin_xy, pad_h, feature_r


# A feature spawn keeps this much Chebyshev clearance to the tile border, so
# the standing footprint (reach 0.31 m + foot radius) starts on the tile it
# is scored on rather than astride a border.
SPAWN_EDGE_MARGIN = 0.35


def sample_tile_spawn(rng, terrain_type, level, origin_xy, pad_h, pad_jitter, yaw_enable):
    """Pick a spawn pose on one tile.

    Returns the pad centre plus a small xy jitter, the pad's height, and a
    yaw quaternion. ``pad_jitter`` and ``yaw_enable`` are plain Python
    values, so the function can be vmapped over (rng, terrain_type, level)."""
    xy0 = origin_xy[level, terrain_type]
    ph = pad_h[level, terrain_type]
    rj, ry = jax.random.split(rng)
    jitter = jax.random.uniform(rj, (2,), minval=-pad_jitter, maxval=pad_jitter)
    spawn_xy = xy0 + jitter
    if yaw_enable:
        yaw = jax.random.uniform(ry, minval=-jp.pi, maxval=jp.pi)
        quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
    else:
        quat = jp.array([1.0, 0.0, 0.0, 0.0])
    return spawn_xy, ph, quat


def sample_feature_spawn(
    rng, terrain_type, level, on_flat, origin_xy, pad_h, pad_jitter,
    tile_size, yaw_enable,
):
    """Pick a spawn pose anywhere on one tile's interior.

    xy lands uniformly inside the tile's Chebyshev square minus
    SPAWN_EDGE_MARGIN -- on the stairs, the rubble, the slope, wherever.
    The flat row keeps the pad draw (there is no feature to land on, and
    the pad jitter is its identity). The caller sets z from the terrain
    height under the returned xy; the returned height is only the flat
    row's pad height, used where ``on_flat`` is true.

    Both candidate draws always run, so the rng cost is fixed; ``on_flat``
    is traced and selects between them."""
    xy0 = origin_xy[level, terrain_type]
    ph = pad_h[level, terrain_type]
    r_feat, r_pad, ry = jax.random.split(rng, 3)
    half = tile_size / 2 - SPAWN_EDGE_MARGIN
    feat_xy = xy0 + jax.random.uniform(r_feat, (2,), minval=-half, maxval=half)
    pad_xy = xy0 + jax.random.uniform(
        r_pad, (2,), minval=-pad_jitter, maxval=pad_jitter
    )
    spawn_xy = jp.where(on_flat, pad_xy, feat_xy)
    if yaw_enable:
        yaw = jax.random.uniform(ry, minval=-jp.pi, maxval=jp.pi)
        quat = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
    else:
        quat = jp.array([1.0, 0.0, 0.0, 0.0])
    return spawn_xy, ph, quat


def curriculum_step(
    level, walked, commanded_dist, steps_lived, episode_length,
    rng, n_rows, tile_size, demote_fraction,
    strikes=0, demote_strikes=1, crossed=None, grace_steps=0, pinned=False,
):
    """Move one env's level after an episode, legged_gym style with three
    v4.2 amendments: band promotion, strike demotion, and spawn grace.

    Promotion: with ``crossed=None``, the legacy rule -- walked more than
    half a tile, a Euclidean distance whose diagonal leak the v4.1 ledger
    documents. The caller that has the geometry passes ``crossed`` instead:
    a traced bool saying the episode actually crossed the tile's feature
    band (Chebyshev, terrain_wrapper computes it). Going up from the top
    level lands on a random row, so easy terrain stays in training.
    Promotion wins when promote and fail both fire, matching legged_gym's
    ``move_down * ~move_up``.

    Demotion: an episode fails when it covered less than
    ``demote_fraction`` of the distance its commands asked for, projected
    onto a full episode -- a fall at step 50 of 1000 gets twenty times the
    distance commanded so far, so almost any fall fails. The projection is
    legged_gym's escape valve, kept as is. What changed is that one failure
    no longer demotes: failures increment ``strikes``, a clean or promoted
    episode clears them, and only ``demote_strikes`` consecutive failures
    drop a level (and clear the count). One is the legacy behavior.

    Grace: an episode that ended within ``grace_steps`` is neutral -- no
    strike, no promotion, strikes carried unchanged. Feature spawns can put
    the robot on a stair edge mid four-bar-closure relaxation; those falls
    say nothing about the level.

    ``pinned`` freezes the env: level and strikes pass through untouched.
    The 20% coverage slice rides on this.

    The rng always advances, so the caller can gate everything on ``done``
    with one where. Returns (new_level, new_strikes, rng).
    """
    promote = (walked > 0.5 * tile_size) if crossed is None else crossed
    # steps_lived is 0 only before the first step has run, where commanded_dist
    # is 0 too and the threshold is 0 either way; the maximum just keeps the
    # division finite.
    full_episode = commanded_dist * episode_length / jp.maximum(steps_lived, 1)
    fail = walked < demote_fraction * full_episode
    graced = steps_lived <= grace_steps
    promote = promote & ~graced
    fail = fail & ~graced & ~promote
    clean = ~fail & ~graced
    new_strikes = jp.where(clean, 0, jp.where(fail, strikes + 1, strikes))
    demote = new_strikes >= demote_strikes
    new_strikes = jp.where(demote & ~promote, 0, new_strikes)
    delta = jp.where(promote, 1, jp.where(demote, -1, 0))
    stepped = jp.clip(level + delta, 0, n_rows - 1)
    at_max = level >= (n_rows - 1)
    rng, sub = jax.random.split(rng)
    rand_row = jax.random.randint(sub, (), 0, n_rows)
    new_level = jp.where(promote & at_max, rand_row, stepped)
    new_level = jp.where(pinned, level, new_level)
    new_strikes = jp.where(pinned, strikes, new_strikes)
    return new_level, new_strikes, rng
