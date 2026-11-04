"""Select a model and coordinate map together, before reading a bag."""

from dataclasses import dataclass
from pathlib import Path

from wojtek_rl import paths, robots


JOINT_MAPS = {
    "wojtek": paths.WOJTEK_POLICY_PKG / "config/joint_map.yaml",
    "legs_v627": paths.WOJTEK_POLICY_PKG / "config/joint_map_v627.yaml",
}


@dataclass(frozen=True)
class IdentificationInputs:
    robot: str
    xml: Path
    joint_map: Path


def resolve_inputs(robot="wojtek", xml=None, joint_map=None):
    """Resolve one variant; reject known files belonging to another variant.

    Custom model/map files are allowed. Their physical identity cannot be
    inferred from a filename or a JointState bag: the operator selects it.
    """
    robots.get(robot)
    xml = Path(xml or paths.robot_files(robot)["scene"]).resolve()
    joint_map = Path(joint_map or JOINT_MAPS[robot]).resolve()
    for other in robots.NAMES:
        if other == robot:
            continue
        other_files = paths.robot_files(other)
        if xml in {path.resolve() for path in other_files.values()}:
            raise ValueError(f"model belongs to {other}, but --robot is {robot}")
        if joint_map == JOINT_MAPS[other].resolve():
            raise ValueError(f"joint map belongs to {other}, but --robot is {robot}")
    for label, path in (("model", xml), ("joint map", joint_map)):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")
    return IdentificationInputs(robot, xml, joint_map)
