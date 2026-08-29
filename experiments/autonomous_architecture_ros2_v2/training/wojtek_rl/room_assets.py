"""Download and convert a photo-textured room scan into MuJoCo-ready assets.

Pipeline (idempotent; each stage skipped when its output exists):
  1. download the Habitat test-scenes zip and extract van-gogh-room.glb
  2. rotate glTF Y-up -> MuJoCo Z-up, recenter so the floor sits at z=0 and
     the room's footprint is centered on the origin
  3. export per-material visual OBJs (+ texture PNGs) for rendering
  4. CoACD convex decomposition for collision (or --boxes AABB fallback);
     hulls that live at floor level are dropped -- the scene keeps a flat
     MuJoCo plane as the walking surface, matching how the policy trained
  5. write manifest.json, consumed by wojtek_rl.build_room

Run:
    ./run.sh room-assets [--glb path.glb] [--boxes] [--threshold 0.05]
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from wojtek_rl import paths

DEFAULT_URL = "https://dl.fbaipublicfiles.com/habitat/habitat-test-scenes_v1.0.zip"
GLB_NAME = "van-gogh-room.glb"

RAW_DIR = paths.ROOM_DIR / "raw"  # shared zip/glb cache for every scene

# Leftover slivers whose top stays below this are scan noise, not obstacles.
FLOOR_HULL_MAX_Z = 0.04
# Faces below this belong to the scan floor. The scanned floor undulates by
# several cm, which a ~0.1 m-tall robot on a flat plane visibly hovers over
# (or sinks into), so the floor VERTICES are snapped flat to z=0 -- textures
# are untouched and the deformation is invisible at room scale. Physics then
# walks on the invisible z=0 plane, which now coincides with the visual
# floor everywhere; floor faces are stripped before convex decomposition.
FLOOR_CUT_Z = 0.05
# Hulls that start above this can never touch the ~0.35 m robot (ceiling,
# lamp shades); dropping them keeps the contact broadphase lean.
CEILING_HULL_MIN_Z = 0.4
# Sanity band for a real room vs a ~0.35 m robot.
MIN_PLANAR_EXTENT = 3.0
MAX_PLANAR_EXTENT = 15.0

# MuJoCo is +Z-up. glTF is nominally +Y-up, but many scans (including
# van-gogh-room.glb -- verified by rendering) already store Z-up geometry;
# rotating those lays the room on its side. Pick with --up.
YUP_TO_ZUP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
UP_TRANSFORMS = {"z": np.eye(4), "y": YUP_TO_ZUP}

LICENSE_TEXT = """# Room scan asset

"Van Gogh Room" scan, distributed with the Habitat test scenes
(https://dl.fbaipublicfiles.com/habitat/habitat-test-scenes_v1.0.zip),
original model by ruslans3d on Sketchfab, licensed CC Attribution.

Only used locally for simulation; the raw/processed asset files are
gitignored and re-created by `./run.sh room-assets`.
"""


def download_glb(url: str, dest: Path, member: str = GLB_NAME) -> None:
    """Fetch the test-scenes zip and extract one scene glb."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / Path(url).name
    if not zip_path.exists():
        print(f"downloading {url} ...")
        tmp = zip_path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.endswith(member)]
        if not members:
            raise FileNotFoundError(f"{member} not found in {zip_path}")
        with zf.open(members[0]) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
    print(f"extracted {dest}")


def load_meshes(glb: Path, up: str = "z", max_extent: float = MAX_PLANAR_EXTENT) -> list:
    """Load the glb, apply node transforms, rotate to Z-up, floor to z=0."""
    import trimesh

    scene = trimesh.load(str(glb), force="scene")
    up_transform = UP_TRANSFORMS[up]
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, gname = scene.graph[node]
        mesh = scene.geometry[gname].copy()
        mesh.apply_transform(transform)
        mesh.apply_transform(up_transform)
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"no geometry in {glb}")

    bounds = np.array([m.bounds for m in meshes])  # (n, 2, 3)
    lo, hi = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    extent = hi - lo
    if not (MIN_PLANAR_EXTENT <= max(extent[0], extent[1]) <= max_extent):
        raise ValueError(
            f"room planar extent {extent[:2]} m outside "
            f"[{MIN_PLANAR_EXTENT}, {max_extent}] -- wrong scale or units?"
        )
    center = (lo + hi) / 2.0
    shift = np.array([-center[0], -center[1], -_floor_surface_z(meshes)])
    for m in meshes:
        m.apply_translation(shift)
    _flatten_floor(meshes)
    return meshes


def _flatten_floor(meshes) -> None:
    """Snap the walkable floor surface to exactly z=0 (in place).

    Vertices of upward-facing faces near floor level move vertically to 0;
    everything else (furniture bases, wall bottoms sharing vertices) deforms
    by at most the scan's waviness, invisible at room scale.
    """
    for m in meshes:
        up = m.face_normals[:, 2] > 0.7
        low = m.triangles_center[:, 2] < FLOOR_CUT_Z
        vids = np.unique(m.faces[up & low])
        if not len(vids):
            continue
        vertices = m.vertices.copy()
        snap = vids[np.abs(vertices[vids, 2]) < 2 * FLOOR_CUT_Z]
        vertices[snap, 2] = 0.0
        m.vertices = vertices


def _floor_surface_z(meshes) -> float:
    """z of the walkable floor surface: median height of the large
    upward-facing faces in the scan's bottom half-meter. The raw AABB min is
    unreliable -- scans carry below-floor artifacts."""
    min_z = min(m.bounds[0][2] for m in meshes)
    zs, weights = [], []
    for m in meshes:
        centers = m.triangles_center
        sel = (m.face_normals[:, 2] > 0.7) & (centers[:, 2] < min_z + 0.5)
        zs.append(centers[sel, 2])
        weights.append(m.area_faces[sel])
    zs = np.concatenate(zs)
    weights = np.concatenate(weights)
    if not len(zs):
        return min_z
    order = np.argsort(zs)
    cdf = np.cumsum(weights[order])
    return float(zs[order][np.searchsorted(cdf, 0.5 * cdf[-1])])


def _texture_image(mesh):
    """PIL image of the mesh's base color texture, or None."""
    material = getattr(mesh.visual, "material", None)
    if material is None:
        return None
    image = getattr(material, "image", None)
    if image is None:
        image = getattr(material, "baseColorTexture", None)
    return image


def export_visual(meshes, visual_dir: Path, prefix: str = "room") -> list[dict]:
    """Write <prefix>_<i>.obj (+ texture png); return manifest entries."""
    import trimesh

    if visual_dir.exists():
        shutil.rmtree(visual_dir)
    visual_dir.mkdir(parents=True)
    entries = []
    for i, mesh in enumerate(meshes):
        name = f"{prefix}_{i}"
        obj_text = trimesh.exchange.obj.export_obj(
            mesh, include_texture=True, write_texture=False
        )
        (visual_dir / f"{name}.obj").write_text(obj_text)
        texture = None
        image = _texture_image(mesh)
        if image is not None:
            texture = f"{name}.png"
            image.convert("RGB").save(visual_dir / texture)
        entries.append({"mesh": f"{name}.obj", "texture": texture})
    return entries


def _export_hull(vertices, faces, index: int, collision_dir: Path, prefix: str) -> str:
    import trimesh

    name = f"{prefix}_hull_{index}.obj"
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(
        collision_dir / name
    )
    return name


def _strip_floor(merged):
    """Remove faces entirely below FLOOR_CUT_Z: after _flatten_floor the
    walking surface is the z=0 plane, so the floor needs no hulls."""
    face_top = merged.vertices[merged.faces][:, :, 2].max(axis=1)
    merged.update_faces(face_top >= FLOOR_CUT_Z)
    merged.remove_unreferenced_vertices()
    return merged


def decompose_coacd(
    meshes, threshold: float, collision_dir: Path, prefix: str = "room"
) -> tuple[list[str], int]:
    """CoACD hulls over the merged, floor-stripped room mesh."""
    import coacd
    import trimesh

    merged = _strip_floor(
        trimesh.util.concatenate(
            [
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
                for m in meshes
            ]
        )
    )
    parts = coacd.run_coacd(
        coacd.Mesh(merged.vertices, merged.faces), threshold=threshold
    )
    files, dropped = [], 0
    for verts, faces in parts:
        z = np.asarray(verts)[:, 2]
        if z.max() < FLOOR_HULL_MAX_Z or z.min() > CEILING_HULL_MIN_Z:
            dropped += 1
            continue
        files.append(_export_hull(verts, faces, len(files), collision_dir, prefix))
    return files, dropped


def decompose_boxes(meshes) -> tuple[list[dict], int]:
    """Fallback: AABB box per connected component. Crude -- diagonal walls
    become fat boxes -- but needs no CoACD."""
    boxes, dropped = [], 0
    for mesh in meshes:
        for part in mesh.split(only_watertight=False):
            lo, hi = part.bounds
            if hi[2] < FLOOR_HULL_MAX_Z or lo[2] > CEILING_HULL_MIN_Z:
                dropped += 1
                continue
            size = np.maximum((hi - lo) / 2.0, 0.005)
            boxes.append(
                {"pos": ((lo + hi) / 2.0).tolist(), "halfsize": size.tolist()}
            )
    return boxes, dropped


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--glb", type=Path, default=None, help="use a local glb instead of downloading")
    p.add_argument(
        "--name",
        default="room",
        help="scene name: 'room' keeps the legacy assets/room location, anything "
        "else lands in assets/scenes/<name>/ (see paths.scene_dir)",
    )
    p.add_argument(
        "--zip-member",
        default=GLB_NAME,
        help="glb filename inside the test-scenes zip (e.g. apartment_1.glb)",
    )
    p.add_argument("--boxes", action="store_true", help="AABB-box collision fallback (no CoACD)")
    p.add_argument(
        "--skip-collision",
        action="store_true",
        help="visual assets only -- for scenes used kinematically (occupancy-"
        "grid collision) where CoACD on a monolithic scan is not worth it",
    )
    p.add_argument("--threshold", type=float, default=0.05, help="CoACD concavity threshold")
    p.add_argument(
        "--max-extent",
        type=float,
        default=MAX_PLANAR_EXTENT,
        help="planar sanity cap in meters (apartments run larger than rooms)",
    )
    p.add_argument(
        "--up",
        choices=sorted(UP_TRANSFORMS),
        default="z",
        help="which axis of the source file points up (van-gogh-room.glb is z)",
    )
    args = p.parse_args(argv)

    scene_dir = paths.scene_dir(args.name)
    visual_dir = scene_dir / "processed" / "visual"
    collision_dir = scene_dir / "processed" / "collision"

    glb = args.glb or RAW_DIR / args.zip_member
    if not glb.exists():
        if args.glb is not None:
            raise FileNotFoundError(glb)
        download_glb(args.url, glb, member=args.zip_member)

    scene_dir.mkdir(parents=True, exist_ok=True)
    license_md = scene_dir / "LICENSE.md"
    if not license_md.exists():
        license_md.write_text(LICENSE_TEXT)

    print(f"loading {glb} (up={args.up}) ...")
    meshes = load_meshes(glb, up=args.up, max_extent=args.max_extent)
    visual = export_visual(meshes, visual_dir, prefix=args.name)
    print(f"visual: {len(visual)} mesh(es) -> {visual_dir}")

    if collision_dir.exists():
        shutil.rmtree(collision_dir)
    collision_dir.mkdir(parents=True)
    collision_meshes: list[str] = []
    collision_boxes: list[dict] = []
    dropped = 0
    if args.skip_collision:
        print("collision: skipped (kinematic scene)")
    elif args.boxes:
        collision_boxes, dropped = decompose_boxes(meshes)
        print(f"collision: {len(collision_boxes)} boxes ({dropped} floor parts dropped)")
    else:
        collision_meshes, dropped = decompose_coacd(
            meshes, args.threshold, collision_dir, prefix=args.name
        )
        print(f"collision: {len(collision_meshes)} hulls ({dropped} floor hulls dropped)")

    bounds = np.array([m.bounds for m in meshes])
    lo, hi = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    manifest = {
        "source": args.url if args.glb is None else str(args.glb),
        "name": args.name,
        "aabb": [lo.tolist(), hi.tolist()],
        "visual_dir": str(visual_dir.relative_to(scene_dir)),
        "collision_dir": str(collision_dir.relative_to(scene_dir)),
        "visual": visual,
        "collision_meshes": collision_meshes,
        "collision_boxes": collision_boxes,
        "dropped_floor_parts": dropped,
    }
    paths.scene_manifest(args.name).write_text(json.dumps(manifest, indent=2))
    print(f"wrote {paths.scene_manifest(args.name)}")


if __name__ == "__main__":
    main()
