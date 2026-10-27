"""Import a CAD-exported robot archive as the source model of a robot variant.

The mechanical team ships a leg design as a `sim_robot` archive: robot.xml
(MJCF), robot.urdf, loop_closure.yaml and a meshes/ directory. That MJCF is
a viewer scene, not a source model: it carries its own floor, light and
<option>, and its collision is several hundred convex pieces per leg, which
no batched backend can afford. This script turns it into the same kind of
file as wojtek.xml, robot only, and leaves the rest to build_model:

  - the floor, the light, <option> and the convex collision pieces
    (`col_*` meshes and the geoms that use them) are dropped
  - meshes the variant shares with the stock robot keep pointing at
    wojtek_description/meshes; the variant's own visual meshes are copied
    to wojtek_description/meshes/<robot>/
  - names, joints, inertials, actuators, sensors, the loop-closure
    equalities and the CAD keyframe are kept exactly as exported

    ./run.sh import-robot --robot legs_v627 --archive ../sim_robot_v627.tgz
"""

import argparse
import shutil
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from wojtek_rl import paths, robots

# Prefix of the convex-decomposition collision meshes in the export.
COLLISION_MESH_PREFIX = "col_"


def _find_export(root: Path) -> Path:
    hits = sorted(root.rglob("robot.xml"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one robot.xml under {root}, found {hits}")
    return hits[0].parent


def convert(export_dir: Path, robot: str) -> tuple[ET.ElementTree, list[str]]:
    """The source-model tree for `robot`, and the mesh files it must copy."""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(export_dir / "robot.xml", parser=parser)
    top = tree.getroot()
    top.set("model", f"wojtek_{robot}")

    for option in top.findall("option"):
        top.remove(option)

    shared = {p.name for p in paths.MESH_DIR.glob("*.stl")}
    own_meshes = []
    asset = top.find("asset")
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name").startswith(COLLISION_MESH_PREFIX):
            asset.remove(mesh)
        elif mesh.get("file") not in shared:
            own_meshes.append(mesh.get("file"))
            mesh.set("file", f"{robot}/{mesh.get('file')}")
    top.find("compiler").set("meshdir", "../../meshes")

    world = top.find("worldbody")
    for child in list(world):
        if child.tag != "body":
            world.remove(child)
    for parent in world.iter():
        for geom in list(parent.findall("geom")):
            if geom.get("mesh", "").startswith(COLLISION_MESH_PREFIX):
                parent.remove(geom)
    return tree, own_meshes


def import_robot(archive: Path, robot: str) -> None:
    robots.get(robot)  # a variant is declared in robots.py before it is imported
    files = paths.robot_files(robot)
    with tempfile.TemporaryDirectory() as tmp:
        if archive.is_dir():
            export_dir = _find_export(archive)
        else:
            with tarfile.open(archive) as tar:
                tar.extractall(tmp, filter="data")
            export_dir = _find_export(Path(tmp))
        tree, own_meshes = convert(export_dir, robot)
        mesh_dir = paths.MESH_DIR / robot
        mesh_dir.mkdir(parents=True, exist_ok=True)
        for name in own_meshes:
            shutil.copyfile(export_dir / "meshes" / name, mesh_dir / name)
    files["source"].parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(files["source"], encoding="unicode")
    print(f"wrote {files['source']}")
    print(f"copied {len(own_meshes)} meshes to {mesh_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", required=True, choices=robots.NAMES)
    p.add_argument("--archive", required=True, type=Path,
                   help="sim_robot .tgz, or a directory it was unpacked into")
    args = p.parse_args()
    import_robot(args.archive, args.robot)


if __name__ == "__main__":
    main()
