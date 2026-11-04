"""Profile and coordinate selection without compiling or stepping a model."""

from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from wojtek_rl import paths, robots
from wojtek_rl.sysid.bag import read_bag
from wojtek_rl.sysid.inputs import JOINT_MAPS, resolve_inputs


@pytest.mark.parametrize("robot", robots.NAMES)
def test_profile_selects_model_and_map(robot):
    inputs = resolve_inputs(robot)
    assert inputs.xml == paths.robot_files(robot)["scene"].resolve()
    assert inputs.joint_map == JOINT_MAPS[robot].resolve()


@pytest.mark.parametrize("robot,other", [("wojtek", "legs_v627"), ("legs_v627", "wojtek")])
def test_known_cross_variant_inputs_are_refused(robot, other):
    with pytest.raises(ValueError, match="model belongs"):
        resolve_inputs(robot, xml=paths.robot_files(other)["scene"])
    with pytest.raises(ValueError, match="joint map belongs"):
        resolve_inputs(robot, joint_map=JOINT_MAPS[other])


def test_custom_files_require_existing_inputs(tmp_path):
    with pytest.raises(ValueError, match="model does not exist"):
        resolve_inputs(xml=tmp_path / "missing.xml")
    xml = tmp_path / "measured.xml"
    xml.write_text("<mujoco/>")
    selected = resolve_inputs("legs_v627", xml=xml)
    assert selected.joint_map == JOINT_MAPS["legs_v627"].resolve()


def test_cli_uses_absolute_states_and_records_selected_variant():
    from wojtek_rl.sysid.__main__ import _parse_args

    args = _parse_args(["--bag", "example", "--robot", "legs_v627"])
    assert args.states_topic == "/wojtek/joint_states_abs"
    assert args.xml == str(paths.robot_files("legs_v627")["scene"].resolve())
    assert args.joint_map == str(JOINT_MAPS["legs_v627"].resolve())
    legacy = _parse_args(["--bag", "example", "--states-topic", "/joint_states"])
    assert legacy.states_topic == "/joint_states"


def test_v627_bag_conversion_uses_drive_signs_and_absolute_topic(monkeypatch):
    # These two hip drives have opposite signs in the v627 map. Include
    # a misleading boot-relative stream to verify the selected topic wins.
    names = ["rear_left_second_joint", "front_left_second_joint"]
    command = SimpleNamespace(name=names, position=[0.4, -0.6], velocity=[])
    measured = SimpleNamespace(name=names, position=[0.2, -0.3], velocity=[1.0, 2.0])
    boot = SimpleNamespace(name=names, position=[9.0, 9.0], velocity=[])
    rows = [("/wojtek/joint_targets", command),
            ("/wojtek/joint_states_abs", measured), ("/joint_states", boot)]

    class Reader:
        def __init__(self, _paths):
            self.connections = [SimpleNamespace(topic=t, msgtype="joint") for t, _ in rows]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def messages(self, connections):
            for conn, (_, msg) in zip(self.connections, rows):
                if conn in connections:
                    yield conn, 1000000000, msg

        def deserialize(self, raw, _type):
            return raw

    module = ModuleType("rosbags.highlevel")
    module.AnyReader = Reader
    monkeypatch.setitem(sys.modules, "rosbags.highlevel", module)
    inputs = resolve_inputs("legs_v627")
    signals = read_bag("example", names, joint_map_yaml=inputs.joint_map,
                       states_topic="/wojtek/joint_states_abs")
    np.testing.assert_allclose(signals.cmd, [[-0.4, -0.6]])
    np.testing.assert_allclose(signals.qpos, [[-0.2, -0.3]])
    np.testing.assert_allclose(signals.qvel, [[-1.0, 2.0]])
