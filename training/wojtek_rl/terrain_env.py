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
    n_rows: int  # curriculum levels, the flat row included when there is one
    flat_row: bool  # true: level 0 is flat, every terrain row sits one higher
    n_types: int
    tile_size: float
    pad_jitter: float
    spawn_yaw: bool
    demote_fraction: float
    init_level_frac: float
    spawn_level: int

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


def require_flat_row(spec: dict, kind: str, want: bool) -> None:
    """Refuse an arena whose flat row does not match what the run asked for.

    The flat row is baked into the generated files, so the arena build and the
    experiment preset have to agree. Neither mismatch shows up on its own:
    without the row the run trains a level short of its configuration, and with
    a row nobody asked for every logged level names a different terrain.

    The flag describes the training arena. The eval arena never carries the
    row (build-terrain refuses to add it), so eval loads ignore the run's
    flag: a flat-row policy still scores on the standard course.
    """
    if kind == "eval":
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
    # The arena's own truth, not the run's flag: an eval load under a
    # flat-row run carries no row.
    flat_row = bool(spec.get("flat_row", False))
    origin_xy, pad_h = tables_from_spec(spec, terrain.TYPES)

    # A coarse fit check: the spawn scatter plus the standing footprint has to
    # fit the arena's flat pad, or spawns start with feet on the features. The
    # linear sum ignores corner draws, but the pad's taper band absorbs that
    # fringe; what this catches is a category error -- training with the
    # default 0.15 m jitter on the eval arena's deliberately small 0.40 m pads.
    pad_jitter = float(terrain_cfg.pad_jitter)
    pad_radius = float(spec["pad_radius"])
    if pad_jitter + terrain.FOOTPRINT_REACH > pad_radius:
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
        n_rows=int(spec["n_rows"]),
        flat_row=flat_row,
        n_types=len(terrain.TYPES),
        tile_size=float(spec["tile_size"]),
        pad_jitter=pad_jitter,
        spawn_yaw=bool(terrain_cfg.spawn_yaw),
        demote_fraction=float(terrain_cfg.demote_fraction),
        init_level_frac=float(terrain_cfg.init_level_frac),
        spawn_level=int(terrain_cfg.get("spawn_level", -1)),
    )


def tables_from_spec(spec: dict, types) -> tuple[np.ndarray, np.ndarray]:
    """Tile-centre xy and pad height for every (row, type) pair.

    Indexed by type, not column, because each row shuffles its column order.
    Shapes: ``(n_rows, n_types, 2)`` and ``(n_rows, n_types)``."""
    n_rows = int(spec["n_rows"])
    type_idx = {t: i for i, t in enumerate(types)}
    origin_xy = np.zeros((n_rows, len(types), 2), dtype=np.float32)
    pad_h = np.zeros((n_rows, len(types)), dtype=np.float32)
    for t in spec["tiles"]:
        r = int(t["row"])
        j = type_idx[t["type"]]
        origin_xy[r, j] = (t["origin"][0], t["origin"][1])
        pad_h[r, j] = t["pad_height"]
    return origin_xy, pad_h


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


def curriculum_step(
    level, walked, commanded_dist, steps_lived, episode_length,
    rng, n_rows, tile_size, demote_fraction,
):
    """Move one env's level after an episode, legged_gym style.

    Walked more than half a tile: one level up. Covered less than
    ``demote_fraction`` of the distance the commands asked for over a FULL
    episode: one level down. Otherwise stay, which covers standing episodes
    too. Levels stay inside ``[0, n_rows-1]``, except going up from the top
    level lands on a random row, so easy terrain stays in training. Promotion
    wins when both fire, matching legged_gym's ``move_down * ~move_up``. The
    rng always advances, so the caller can gate everything on ``done`` with
    one where.

    The demote threshold is projected onto the full episode. legged_gym
    compares walked distance against half the commanded speed times the whole
    episode length, using the command held at reset; this env resamples the
    command mid-episode, so the equivalent is to scale the commanded distance
    actually accumulated by ``episode_length / steps_lived``. A timeout has
    ``steps_lived == episode_length``, so the factor is 1 and the threshold is
    what it always was. A fall at step 50 of 1000 gets twenty times the
    distance commanded so far, so almost any fall demotes -- the escape valve
    legged_gym has, and falling is the dominant termination on terrain.

    Note the asymmetry: the promote threshold is hardcoded at half a tile,
    while demote is configurable. Half a tile knows nothing about how long the
    obstacle is, which is what bounds the stair flight at six steps (treads end
    at 1.25 m of a 1.5 m half-tile). A longer flight needs promotion defined
    against the feature band instead.

    The threshold is heading-dependent for the same reason: ``walked`` is a
    Euclidean distance while every feature is a concentric square, so 1.5 m on
    a diagonal is only 1.06 m out in Chebyshev terms -- tread 4 of 6. About a
    quarter of headings promote without crossing the whole flight. Kept as-is
    because it is legged_gym's rule; the measurement scan uses Chebyshev radii
    for exactly this reason.
    """
    promote = walked > 0.5 * tile_size
    # steps_lived is 0 only before the first step has run, where commanded_dist
    # is 0 too and the threshold is 0 either way; the maximum just keeps the
    # division finite.
    full_episode = commanded_dist * episode_length / jp.maximum(steps_lived, 1)
    demote = walked < demote_fraction * full_episode
    delta = jp.where(promote, 1, jp.where(demote, -1, 0))
    stepped = jp.clip(level + delta, 0, n_rows - 1)
    at_max = level >= (n_rows - 1)
    rng, sub = jax.random.split(rng)
    rand_row = jax.random.randint(sub, (), 0, n_rows)
    new_level = jp.where(promote & at_max, rand_row, stepped)
    return new_level, rng
