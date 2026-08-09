"""Repo-relative paths and robot constants shared by every module."""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
MUJOCO_DIR = REPO_ROOT / "ros/src/wojtek_description/mujoco"
SOURCE_XML = MUJOCO_DIR / "wojtek.xml"
ROBOT_XML = MUJOCO_DIR / "wojtek_mjx.xml"
SCENE_XML = MUJOCO_DIR / "scene_mjx.xml"

# Room-scan scene (room_assets.py -> build_room.py -> room_app.py).
ROOM_DIR = PROJECT_DIR / "assets/room"
ROOM_MANIFEST = ROOM_DIR / "manifest.json"
ROOM_SCENE_XML = MUJOCO_DIR / "scene_room.xml"

# Named scan scenes (same pipeline, one directory per scene). "room" stays at
# its legacy location so existing tooling keeps working.
SCENES_DIR = PROJECT_DIR / "assets/scenes"


def scene_dir(name: str) -> Path:
    return ROOM_DIR if name == "room" else SCENES_DIR / name


def scene_manifest(name: str) -> Path:
    return scene_dir(name) / "manifest.json"


def scene_xml(name: str) -> Path:
    return ROOM_SCENE_XML if name == "room" else MUJOCO_DIR / f"scene_{name}.xml"

# Procedural terrain arenas (terrain.py -> build_terrain.py). One file set per
# arena kind: `train` is what policies train on, `eval` is the fixed
# measurement course, `test` belongs to the test suite. Every set sits next to
# the robot XML, because the scene includes the robot by relative path and
# MuJoCo resolves <hfield file=> against the robot's mesh directory -- an arena
# written to a temporary directory does not compile. Separate sets are what
# keeps a measurement or a test run from overwriting the arena a policy
# trained on.
TERRAIN_KINDS = ("train", "eval", "eval_deep", "test")


def terrain_paths(kind: str = "train") -> dict[str, Path]:
    """The four generated files for one arena kind: scene XML, heightfield
    binary, spec JSON, lookup NPZ. `train` keeps the original unsuffixed
    names, so an arena built before the kinds existed still loads."""
    if kind not in TERRAIN_KINDS:
        raise ValueError(
            f"terrain arena kind must be one of {TERRAIN_KINDS}, got {kind!r}"
        )
    tag = "" if kind == "train" else f"_{kind}"
    return {
        "scene": MUJOCO_DIR / f"scene_terrain{tag}.xml",
        "hfield": MUJOCO_DIR / f"terrain{tag}_hfield.bin",
        "spec": MUJOCO_DIR / f"terrain{tag}_spec.json",
        "lookup": MUJOCO_DIR / f"terrain{tag}_lookup.npz",
    }

# Exported NumPy policy runtime (ROS-free), shared with the real robot.
WOJTEK_POLICY_PKG = REPO_ROOT / "ros/src/wojtek_policy"
def _hf_organization() -> str:
    """HF org hosting the keeper policy repos: HF_ORGANIZATION from the
    environment, else from the repo-root .env (see .env.example)."""
    org = os.environ.get("HF_ORGANIZATION", "")
    if org:
        return org
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("HF_ORGANIZATION="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


HF_ORGANIZATION = _hf_organization()

# Default policy reference for sim/demo apps: any form accepted by
# wojtek_policy.policy_source (HF repo id, local dir, path to policy.npz).
# None until HF_ORGANIZATION is configured; tools then need an explicit
# policy reference.
DEFAULT_POLICY = (f"{HF_ORGANIZATION}/wojtek-stiff-height-locomotion"
                  if HF_ORGANIZATION else None)

# URDF <-> MuJoCo affine joint map, shared with the ROS nodes (sysid bag
# reader converts recorded signals with the exact same table).
JOINT_MAP_YAML = WOJTEK_POLICY_PKG / "config/joint_map.yaml"

# XML declaration order. Actuators and joints follow this order.
LEGS = ("rear_left", "rear_right", "front_right", "front_left")
