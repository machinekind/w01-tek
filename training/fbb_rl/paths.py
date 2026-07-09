"""Repo-relative paths and robot constants shared by every module."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
MUJOCO_DIR = REPO_ROOT / "ros/src/four_bar_bot_description/mujoco"
SOURCE_XML = MUJOCO_DIR / "four_bar_bot.xml"
ROBOT_XML = MUJOCO_DIR / "four_bar_bot_mjx.xml"
SCENE_XML = MUJOCO_DIR / "scene_mjx.xml"

# XML declaration order. Actuators and joints follow this order.
LEGS = ("rear_left", "rear_right", "front_right", "front_left")
