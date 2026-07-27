"""Build a terrain arena and its scene from a seed.

Generates the arena with wojtek_rl.terrain and writes, next to the robot XML
(names shown for the default `train` arena; see paths.terrain_paths):
  - scene_terrain.xml   -- mirrors build_model.SCENE_XML_TEXT, but the flat
    floor plane is replaced by the heightfield geom plus the terrain boxes,
    all carrying the floor's collision semantics (default contype,
    conaffinity=15, condim=3) so the robot pairs with terrain as with the
    floor. Four apron strips continue the flat border outward, so walking
    off the arena lands on ground, not in a void
  - terrain_hfield.bin  -- MuJoCo raw heightfield elevation data
  - terrain_spec.json   -- grid meta, per-tile type/difficulty/origin, pads
  - terrain_lookup.npz  -- ground-truth height grid + extents/resolution

`--arena eval` writes a separate file set holding the fixed measurement course
(wojtek_rl.terrain_suite): its 12 rows, sorted columns and 0.40 m pads. Its
row/pad settings come from the suite, so `--rows`, `--ordered` and
`--pad-radius` do not apply there. `--arena test` is a scratch set for the test
suite. Separate sets are what keeps a measurement from overwriting the arena a
policy trained on.

The heightfield file path is emitted relative to the robot's meshdir, because
MuJoCo resolves <hfield file=> against meshdir (the included robot XML sets it).

Run:
    ./run.sh build-terrain [--arena {train,eval,test}] [--seed N] [--rows N]
                           [--tile-size M] [--border M] [--cell-size M]
                           [--pad-radius M] [--ordered] [--no-check]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import mujoco
import numpy as np

from wojtek_rl import paths, terrain, terrain_suite

# Box tint, alternating by parity so steps read in the viewer; collision
# attributes match the floor line in build_model.SCENE_XML_TEXT.
_BOX_RGBA = ("0.55 0.50 0.45 1", "0.45 0.42 0.40 1")

# Flat ground outside the arena, flush with the border (z = 0). The
# heightfield ends exactly at the arena extent, so without it an env that
# walks past the border free-falls into a void and is scored -- and demoted --
# for a fall no observation could warn it about. Four static strips frame the
# arena; a robot on the tiles never overlaps their AABBs, so they cost nothing
# until one is actually near an edge. The height lookup clamps to the border
# height (0) out there, so terrain-relative measurements stay right too.
APRON_EXTENT = 25.0
APRON_HALF_Z = 0.5


def _apron_geoms(radius_x: float, radius_y: float) -> list[str]:
    """Four box strips around the arena, tops at z = 0, corners covered."""
    ox = radius_x + APRON_EXTENT
    strips = (
        ("apron_north", 0.0, radius_y + APRON_EXTENT / 2, ox, APRON_EXTENT / 2),
        ("apron_south", 0.0, -radius_y - APRON_EXTENT / 2, ox, APRON_EXTENT / 2),
        ("apron_west", -radius_x - APRON_EXTENT / 2, 0.0, APRON_EXTENT / 2, radius_y),
        ("apron_east", radius_x + APRON_EXTENT / 2, 0.0, APRON_EXTENT / 2, radius_y),
    )
    return [
        f'    <geom name="{name}" type="box" '
        f'size="{hx:.4f} {hy:.4f} {APRON_HALF_Z}" '
        f'pos="{cx:.4f} {cy:.4f} {-APRON_HALF_Z}" material="groundplane" '
        f'condim="3" conaffinity="15"/>'
        for name, cx, cy, hx, hy in strips
    ]


def _robot_meshdir() -> Path:
    match = re.search(r'meshdir="([^"]+)"', paths.ROBOT_XML.read_text())
    rel = match.group(1) if match else "."
    return (paths.MUJOCO_DIR / rel).resolve()


def build_scene_xml(arena: terrain.Arena, hfield_file: str) -> str:
    hf = arena.spec.hfield
    size = f"{hf.radius_x:.4f} {hf.radius_y:.4f} {hf.elevation_z:.6f} {hf.base_z:.4f}"
    box_lines = []
    for k, b in enumerate(arena.boxes):
        pos = " ".join(f"{v:.4f}" for v in b.pos)
        half = " ".join(f"{v:.4f}" for v in b.half)
        # quat (not euler) sidesteps MJCF's angle-unit ambiguity; only emitted
        # when the box is turned, so axis-aligned boxes read as before.
        quat = ""
        if b.yaw != 0.0:
            quat = f' quat="{np.cos(b.yaw / 2):.6f} 0 0 {np.sin(b.yaw / 2):.6f}"'
        box_lines.append(
            f'    <geom name="terrain_box_{k}" type="box" size="{half}" '
            f'pos="{pos}"{quat} condim="3" conaffinity="15" rgba="{_BOX_RGBA[k % 2]}"/>'
        )
    boxes = "\n".join(_apron_geoms(hf.radius_x, hf.radius_y) + box_lines)
    return f"""<mujoco model="wojtek_terrain_scene">
  <include file="wojtek_mjx.xml"/>
  <statistic center="0 0 0.15" extent="0.8"/>
  <visual>
    <headlight diffuse="0.3 0.3 0.3" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="120" elevation="-20"/>
    <quality shadowsize="8192"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
    <hfield name="terrain" file="{hfield_file}" size="{size}"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" castshadow="true" pos="0 0 20" dir="0.25 0.35 -0.9" diffuse="0.7 0.7 0.7" specular="0 0 0"/>
    <geom name="terrain_hfield" type="hfield" hfield="terrain" material="groundplane" pos="0 0 {hf.pos_z:.6f}" condim="3" conaffinity="15"/>
{boxes}
    <camera name="track" mode="trackcom" pos="0.9 -1.3 0.5" xyaxes="0.83 0.55 0 -0.15 0.23 0.96"/>
  </worldbody>
</mujoco>
"""


# Highest mtime this process has stamped on each heightfield. Deleting a file
# does not clear MuJoCo's cache entry for its name, so the memory has to outlive
# the file -- see _force_new_mtime.
_STAMPED_MTIME: dict[Path, float] = {}


def _force_new_mtime(path: Path, replaced_mtime: float | None) -> None:
    """Make `path` look strictly newer than any version this process has seen.

    MuJoCo (3.10) caches heightfield assets by filename at one-second mtime
    resolution. Two arenas written to one filename inside the same second, and
    the second compile silently serves the FIRST one's elevation data -- a scene
    whose physics does not match the lookup grid the env reads heights from, with
    no error anywhere. File size is not part of the cache key, so a different
    grid shape does not save you. Reproduced on 2 of 3 tight-loop trials.

    Deleting the file in between does not help: the cache is keyed on the name,
    so a fresh file with a fresh (same-second) mtime collides with the entry the
    deleted one left behind. That is exactly what the test fixtures do at
    teardown, which is why the bump cannot be conditional on finding a file to
    replace -- it has to be against the highest mtime this process ever wrote.

    Only the heightfield needs this: the scene XML is re-parsed every load, and
    the spec/lookup sidecars go through json/numpy, which cache nothing. The
    mtime can land up to one second per rebuild in the future, bounded by how
    many times a single process rebuilds one arena.
    """
    previous = max(_STAMPED_MTIME.get(path, 0.0), replaced_mtime or 0.0)
    if previous == 0.0:
        # First time this process has written this name: nothing is cached under
        # it, so the natural mtime is already distinct.
        _STAMPED_MTIME[path] = path.stat().st_mtime
        return
    target = max(time.time(), previous + 1.0)
    os.utime(path, (target, target))
    _STAMPED_MTIME[path] = target


def write_arena(arena: terrain.Arena, kind: str = "train") -> None:
    """Write one arena's four files. `kind` picks the file set to overwrite."""
    out = paths.terrain_paths(kind)
    replaced = out["hfield"].stat().st_mtime if out["hfield"].exists() else None
    terrain.write_hfield_bin(out["hfield"], arena.hfield_data)
    _force_new_mtime(out["hfield"], replaced)
    hfield_file = os.path.relpath(out["hfield"], _robot_meshdir()).replace(os.sep, "/")
    out["scene"].write_text(build_scene_xml(arena, hfield_file))
    out["spec"].write_text(json.dumps(terrain.spec_to_dict(arena.spec), indent=2))
    s = arena.spec
    np.savez_compressed(
        out["lookup"],
        lookup=arena.lookup,
        x_min=s.x_min, x_max=s.x_max, y_min=s.y_min, y_max=s.y_max,
        cell_size=s.cell_size, nrow=s.hfield.nrow, ncol=s.hfield.ncol,
    )
    for p in (out["hfield"], out["scene"], out["spec"], out["lookup"]):
        print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arena", choices=paths.TERRAIN_KINDS, default="train",
                   help="which file set to write; eval is the fixed "
                        "measurement course (terrain_suite)")
    # Default None on the four flags the eval arena owns, so "not given" is
    # distinguishable from "given the same value the default happens to be".
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--pad-radius", type=float, default=None)
    p.add_argument("--ordered", action="store_true", default=None,
                   help="sorted-column, exact-difficulty layout")
    p.add_argument("--tile-size", type=float, default=terrain.TILE_SIZE)
    p.add_argument("--border", type=float, default=terrain.BORDER)
    p.add_argument("--cell-size", type=float, default=terrain.CELL_SIZE)
    p.add_argument("--no-check", action="store_true", help="skip the compile check")
    args = p.parse_args()

    owned = {
        "--seed": args.seed, "--rows": args.rows,
        "--pad-radius": args.pad_radius, "--ordered": args.ordered,
    }
    kwargs = dict(
        seed=terrain.DEFAULT_SEED if args.seed is None else args.seed,
        n_rows=terrain.DEFAULT_N_ROWS if args.rows is None else args.rows,
        pad_radius=terrain.PAD_RADIUS if args.pad_radius is None else args.pad_radius,
        ordered=bool(args.ordered),
        tile_size=args.tile_size, border=args.border, cell_size=args.cell_size,
    )
    if args.arena == "eval":
        # The measurement course is a definition, not a CLI choice: its seed,
        # rows, column order and pad radius come from the suite. Overriding one
        # of them silently would produce an arena the scan then stamps with the
        # suite's fingerprint -- numbers filed under a description of a
        # different terrain -- so a conflicting flag is an error, not a
        # preference. (terrain_scan.check_arena refuses such an arena too.)
        suite = terrain_suite.eval_arena_kwargs()
        conflicts = sorted(flag for flag, given in owned.items() if given is not None)
        if conflicts:
            p.error(
                f"--arena eval defines {', '.join(conflicts)} itself "
                "(wojtek_rl.terrain_suite); drop them or build a `train` arena"
            )
        kwargs.update(suite)
        print(
            f"eval arena: seed {terrain_suite.EVAL_SEED}, "
            f"{len(terrain_suite.DIFFICULTIES)} rows "
            f"{terrain_suite.DIFFICULTIES}, pad {terrain_suite.EVAL_PAD_RADIUS} m"
        )
    arena = terrain.generate(**kwargs)
    write_arena(arena, args.arena)
    if not args.no_check:
        scene = paths.terrain_paths(args.arena)["scene"]
        m = mujoco.MjModel.from_xml_path(str(scene))
        print(
            f"compiled: {m.ngeom} geoms, {len(arena.boxes)} terrain boxes, "
            f"hfield {m.hfield_nrow[0]}x{m.hfield_ncol[0]}"
        )


if __name__ == "__main__":
    main()
