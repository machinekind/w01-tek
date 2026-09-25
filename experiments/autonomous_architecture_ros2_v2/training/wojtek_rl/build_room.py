"""Generate scene_room.xml: Wojtek inside the scanned room.

Reads assets/room/manifest.json (produced by wojtek_rl.room_assets) and writes
scene_room.xml next to scene_mjx.xml -- it must live there so
``<include file="wojtek_mjx.xml"/>`` and the robot's relative meshdir
keep resolving. Mesh/texture paths are emitted relative to the robot's
meshdir/texturedir, so the file stays portable within the repo.

Collision layout mirrors the flat training scene: an invisible plane at z=0
(group 3) is the walking surface; the room's CoACD hulls / fallback boxes
collide like walls and furniture; the photo-textured meshes are visual-only.

Run:
    ./run.sh build-room [--room-offset X Y YAW_DEG] [--no-check]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from wojtek_rl import paths

# Copied from the flat scene (build_model.SCENE_XML_TEXT) so the chase cam
# and lighting feel identical between apps.
TRACK_CAMERA = '<camera name="track" mode="trackcom" pos="0.9 -1.3 0.5" xyaxes="0.83 0.55 0 -0.15 0.23 0.96"/>'

# Same pairing as the flat floor; group 3 keeps collision geometry invisible
# (the default renderer shows geom groups 0-2 only).
VISUAL_GEOM = 'contype="0" conaffinity="0" group="2"'
COLLISION_GEOM = 'contype="1" conaffinity="15" group="3"'


def _robot_meshdir() -> Path:
    """The meshdir the compiled robot XML declares, resolved from MUJOCO_DIR."""
    match = re.search(r'meshdir="([^"]+)"', paths.ROBOT_XML.read_text())
    rel = match.group(1) if match else "."
    return (paths.MUJOCO_DIR / rel).resolve()


def _rel(target: Path, base: Path) -> str:
    return os.path.relpath(target, base).replace(os.sep, "/")


def _obj_zbounds(path: Path) -> tuple[float, float] | None:
    """(zmin, zmax) of an OBJ's vertices; None when unreadable."""
    zmin, zmax = float("inf"), float("-inf")
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("v "):
                    z = float(line.split()[3])
                    zmin, zmax = min(zmin, z), max(zmax, z)
    except (OSError, ValueError, IndexError):
        return None
    return (zmin, zmax) if zmin <= zmax else None


def _decal_lifts(manifest: dict, visual_dir: Path) -> dict[str, float]:
    """Per-mesh z-lift for rugs/decals coplanar with the floor slab.

    ReplicaCAD ships rugs as separate flat meshes whose top face is EXACTLY
    coplanar with the floor's -- the renderer then z-fights per pixel and the
    floor flickers black/wood on camera (user report, 2026-08-27). The mesh
    with the largest footprint is the floor; every thin mesh whose bottom
    sits within 3 mm of the floor's top gets lifted 3 mm.
    """
    bounds, foot = {}, {}
    for entry in manifest["visual"]:
        name = Path(entry["mesh"]).stem
        zb = _obj_zbounds(visual_dir / entry["mesh"])
        if zb is None:
            continue
        bounds[name] = zb
        xs, ys = [], []
        # footprint from the same pass would need x/y too; reread cheaply
        with open(visual_dir / entry["mesh"]) as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    xs.append(float(parts[1])); ys.append(float(parts[2]))
        foot[name] = (max(xs) - min(xs)) * (max(ys) - min(ys)) if xs else 0.0
    if not bounds:
        return {}
    # The floor is the largest-FOOTPRINT flat mesh that actually lies at
    # ground height -- without the flatness guard the wall shell wins, and
    # without the height guard the CEILING does (same footprint, z=3 m).
    flat_meshes = {n: a for n, a in foot.items()
                   if bounds[n][1] - bounds[n][0] < 0.3 and bounds[n][1] < 0.5}
    if not flat_meshes:
        return {}
    floor = max(flat_meshes, key=flat_meshes.get)
    floor_top = bounds[floor][1]
    lifts = {}
    for name, (zmin, zmax) in bounds.items():
        if name == floor:
            continue
        if abs(zmin - floor_top) < 0.003 and (zmax - zmin) < 0.05:
            lifts[name] = 0.003
    return lifts


def build_scene_xml(
    manifest: dict, offset: tuple[float, float, float], scene_name: str = "room"
) -> str:
    """Assemble the room scene MJCF from the asset manifest."""
    scene_dir = paths.scene_dir(scene_name)
    visual_dir = scene_dir / manifest["visual_dir"]
    collision_dir = scene_dir / manifest["collision_dir"]
    meshdir = _robot_meshdir()
    lifts = _decal_lifts(manifest, visual_dir)

    assets, geoms = [], []
    for entry in manifest["visual"]:
        name = Path(entry["mesh"]).stem
        # inertia="shell": scan meshes are open surfaces with ~zero volume,
        # which the default (legacy) inertia computation rejects. The room is
        # a static body, so the inertia value itself never matters.
        assets.append(
            f'    <mesh name="{name}" inertia="shell" '
            f'file="{_rel(visual_dir / entry["mesh"], meshdir)}"/>'
        )
        if entry["texture"]:
            assets.append(
                f'    <texture type="2d" name="{name}_tex" file="{entry["texture"]}"/>'
            )
            assets.append(
                f'    <material name="{name}_mat" texture="{name}_tex"/>'
            )
        else:
            assets.append(
                f'    <material name="{name}_mat" rgba="0.6 0.6 0.6 1"/>'
            )
        lift = f'pos="0 0 {lifts[name]:.3f}" ' if name in lifts else ""
        geoms.append(
            f'      <geom name="{name}_visual" type="mesh" mesh="{name}" '
            f'{lift}material="{name}_mat" {VISUAL_GEOM}/>'
        )

    for hull in manifest["collision_meshes"]:
        name = Path(hull).stem
        assets.append(
            f'    <mesh name="{name}" inertia="shell" '
            f'file="{_rel(collision_dir / hull, meshdir)}"/>'
        )
        geoms.append(
            f'      <geom name="{name}_col" type="mesh" mesh="{name}" {COLLISION_GEOM}/>'
        )

    for i, box in enumerate(manifest["collision_boxes"]):
        pos = " ".join(f"{v:.4f}" for v in box["pos"])
        size = " ".join(f"{v:.4f}" for v in box["halfsize"])
        geoms.append(
            f'      <geom name="box_{i}_col" type="box" pos="{pos}" size="{size}" '
            f"{COLLISION_GEOM}/>"
        )

    x, y, yaw_deg = offset
    nl = "\n"
    return f"""<mujoco model="wojtek_{scene_name}_scene">
  <include file="wojtek_mjx.xml"/>
  <statistic center="0 0 0.15" extent="0.8"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.45 0.45 0.45" specular="0 0 0"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <compiler texturedir="{_rel(visual_dir, paths.MUJOCO_DIR)}"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.4 0.5" rgb2="0.1 0.1 0.15" width="512" height="3072"/>
{nl.join(assets)}
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.01" condim="3" conaffinity="15" group="3"/>
    <light directional="true" pos="0 0 3" dir="0 0 -1" diffuse="0.5 0.5 0.5"/>
    {TRACK_CAMERA}
    <body name="room" pos="{x:.3f} {y:.3f} 0" euler="0 0 {yaw_deg:.1f}">
{nl.join(geoms)}
    </body>
  </worldbody>
</mujoco>
"""


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--room-offset",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "YAW_DEG"),
        help="pose of the room so that (0,0) -- the robot spawn -- is open floor",
    )
    p.add_argument("--name", default="room", help="scene name (see room_assets --name)")
    p.add_argument("--no-check", action="store_true", help="skip the compile check")
    args = p.parse_args(argv)

    manifest = json.loads(paths.scene_manifest(args.name).read_text())
    xml = build_scene_xml(manifest, tuple(args.room_offset), scene_name=args.name)
    scene_xml = paths.scene_xml(args.name)
    scene_xml.write_text(xml)
    print(f"wrote {scene_xml}")

    if not args.no_check:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(scene_xml))
        print(
            f"compile check ok: {model.nmesh} meshes, {model.ngeom} geoms, "
            f"ego cam id {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'ego')}"
        )


if __name__ == "__main__":
    main()
