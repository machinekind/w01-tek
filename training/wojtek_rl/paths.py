"""Repo-relative paths and robot constants shared by every module."""

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

# Exported NumPy policy runtime (ROS-free), shared with the real robot.
# (policy_meta.json is found by WojtekPolicy next to the npz.)
WOJTEK_POLICY_PKG = REPO_ROOT / "ros/src/wojtek_policy"
POLICY_NPZ = WOJTEK_POLICY_PKG / "config/policy.npz"

# XML declaration order. Actuators and joints follow this order.
LEGS = ("rear_left", "rear_right", "front_right", "front_left")
