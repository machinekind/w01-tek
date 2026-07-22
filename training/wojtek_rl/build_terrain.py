"""Build the shared terrain arena and its scene from a seed.

Generates the arena with wojtek_rl.terrain and writes, next to the robot XML:
  - scene_terrain.xml   -- mirrors build_model.SCENE_XML_TEXT, but the flat
    floor plane is replaced by the heightfield geom plus the terrain boxes,
    all carrying the floor's collision semantics (default contype,
    conaffinity=15, condim=3) so the robot pairs with terrain as with the floor
  - terrain_hfield.bin  -- MuJoCo raw heightfield elevation data
  - terrain_spec.json   -- grid meta, per-tile type/difficulty/origin, pads
  - terrain_lookup.npz  -- ground-truth height grid + extents/resolution

The heightfield file path is emitted relative to the robot's meshdir, because
MuJoCo resolves <hfield file=> against meshdir (the included robot XML sets it).

Run:
    ./run.sh build-terrain [--seed N] [--rows N] [--tile-size M] [--border M]
                           [--cell-size M] [--no-check]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import mujoco
import numpy as np

from wojtek_rl import paths, terrain

# Box tint, alternating by parity so steps read in the viewer; collision
# attributes match the floor line in build_model.SCENE_XML_TEXT.
_BOX_RGBA = ("0.55 0.50 0.45 1", "0.45 0.42 0.40 1")


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
        box_lines.append(
            f'    <geom name="terrain_box_{k}" type="box" size="{half}" '
            f'pos="{pos}" condim="3" conaffinity="15" rgba="{_BOX_RGBA[k % 2]}"/>'
        )
    boxes = "\n".join(box_lines)
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


def write_arena(arena: terrain.Arena) -> None:
    terrain.write_hfield_bin(paths.TERRAIN_HFIELD, arena.hfield_data)
    hfield_file = os.path.relpath(paths.TERRAIN_HFIELD, _robot_meshdir()).replace(os.sep, "/")
    paths.TERRAIN_SCENE_XML.write_text(build_scene_xml(arena, hfield_file))
    paths.TERRAIN_SPEC_JSON.write_text(json.dumps(terrain.spec_to_dict(arena.spec), indent=2))
    s = arena.spec
    np.savez_compressed(
        paths.TERRAIN_LOOKUP_NPZ,
        lookup=arena.lookup,
        x_min=s.x_min, x_max=s.x_max, y_min=s.y_min, y_max=s.y_max,
        cell_size=s.cell_size, nrow=s.hfield.nrow, ncol=s.hfield.ncol,
    )
    for p in (
        paths.TERRAIN_HFIELD, paths.TERRAIN_SCENE_XML,
        paths.TERRAIN_SPEC_JSON, paths.TERRAIN_LOOKUP_NPZ,
    ):
        print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=terrain.DEFAULT_SEED)
    p.add_argument("--rows", type=int, default=terrain.DEFAULT_N_ROWS)
    p.add_argument("--tile-size", type=float, default=terrain.TILE_SIZE)
    p.add_argument("--border", type=float, default=terrain.BORDER)
    p.add_argument("--cell-size", type=float, default=terrain.CELL_SIZE)
    p.add_argument("--no-check", action="store_true", help="skip the compile check")
    args = p.parse_args()

    arena = terrain.generate(
        seed=args.seed, n_rows=args.rows, tile_size=args.tile_size,
        border=args.border, cell_size=args.cell_size,
    )
    write_arena(arena)
    if not args.no_check:
        m = mujoco.MjModel.from_xml_path(str(paths.TERRAIN_SCENE_XML))
        print(
            f"compiled: {m.ngeom} geoms, {len(arena.boxes)} terrain boxes, "
            f"hfield {m.hfield_nrow[0]}x{m.hfield_ncol[0]}"
        )


if __name__ == "__main__":
    main()
