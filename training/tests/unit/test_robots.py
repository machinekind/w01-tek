"""Robot variants: their data, their files, the config group, the importer.

Model-free, like everything in tests/unit. What the built models do is in
tests/integration/test_robot_variants.py.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import wojtek_rl
from wojtek_rl import import_robot, paths, robots

CONF_DIR = str(Path(wojtek_rl.__file__).parent / "conf")


def _env_cfg(*overrides):
    with initialize_config_dir(CONF_DIR, version_base=None):
        cfg = compose("config", overrides=list(overrides))
    return OmegaConf.to_container(cfg.task.env, resolve=True)


def test_default_robot_keeps_the_original_paths():
    files = paths.robot_files()
    assert files == {
        "source": paths.SOURCE_XML,
        "robot": paths.ROBOT_XML,
        "scene": paths.SCENE_XML,
    }


def test_a_variant_never_shares_a_generated_file_with_the_stock_robot():
    stock = set(paths.robot_files().values())
    for name in robots.NAMES:
        if name != paths.DEFAULT_ROBOT:
            assert not stock & set(paths.robot_files(name).values())


def test_unknown_robot_is_refused():
    with pytest.raises(ValueError, match="robot must be one of"):
        robots.get("no_such_legs")


@pytest.mark.parametrize("name", robots.NAMES)
def test_height_tables_are_usable(name):
    rb = robots.get(name)
    assert len(rb.height_table) == len(rb.dsecond_table) == len(rb.dthird_table)
    assert np.all(np.diff(rb.height_table) > 0)  # jp.interp needs it
    assert 0.0 in rb.dsecond_table  # one rung is the home pose itself
    assert rb.dthird_table[rb.dsecond_table.index(0.0)] == 0.0
    assert len(rb.leg_sign) == len(rb.stand_pose) == len(paths.LEGS)


def test_stock_third_joint_moves_exactly_twice_the_second():
    # The env used to compute 2 * dsecond. The table has to give the same
    # bits, or every stock policy's anchors move.
    rb = robots.WOJTEK
    for ds, dt in zip(rb.dsecond_table, rb.dthird_table):
        assert np.float32(dt) == np.float32(2.0) * np.float32(ds)


def test_leg_sign_matches_the_stand_pose():
    # A leg mounted the other way round has its joint values negated. Front
    # and rear may differ by the few hundredths of a radian that level the
    # body (see stand_pose in robots.py), never by more.
    for rb in robots.ROBOTS.values():
        front = rb.stand_pose[paths.LEGS.index("front_left")]
        for pose, sign in zip(rb.stand_pose, rb.leg_sign):
            ref = rb.leg_sign[paths.LEGS.index("front_left")]
            assert np.allclose(
                np.array(pose) * sign, np.array(front) * ref, atol=0.06
            )


def test_left_and_right_legs_stand_alike():
    # symmetry.enable mirrors left and right, so their targets must match.
    for rb in robots.ROBOTS.values():
        pose = dict(zip(paths.LEGS, rb.stand_pose))
        assert pose["front_left"] == pose["front_right"]
        assert pose["rear_left"] == pose["rear_right"]


def test_default_config_selects_the_stock_robot():
    env = _env_cfg()
    assert env["robot"] == "wojtek"
    assert "sim_dt" not in env  # the stock robot rides the env default


def test_robot_group_sets_what_follows_from_the_geometry():
    env = _env_cfg("robot=legs_v627")
    rb = robots.LEGS_V627
    assert env["robot"] == rb.name
    assert env["sim_dt"] == rb.timestep
    lo, hi = env["command"]["height"]
    assert rb.height_table[0] < lo < hi < rb.height_table[-1]


def test_the_variants_experiment_selects_its_robot():
    env = _env_cfg("+experiment=legs_v627_locomotion")
    assert env["robot"] == "legs_v627"
    assert env["sim_dt"] == robots.LEGS_V627.timestep


EXPORT = """<mujoco model="cad">
  <compiler angle="radian" meshdir="meshes"/>
  <option timestep="0.0005"/>
  <asset>
    <mesh name="base_link" file="base_link.stl"/>
    <mesh name="vis_shin" file="vis_shin.stl"/>
    <mesh name="col_shin_1" file="col_shin_1.stl"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 1"/>
    <light name="top"/>
    <body name="root">
      <freejoint/>
      <geom name="base_visual" mesh="base_link"/>
      <body name="shin">
        <!-- a CAD note -->
        <geom name="shin_visual" mesh="vis_shin"/>
        <geom name="shin_c1" mesh="col_shin_1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_import_keeps_the_robot_and_drops_the_viewer_scene(tmp_path):
    (tmp_path / "robot.xml").write_text(EXPORT)
    tree, own = import_robot.convert(tmp_path, "legs_v627")
    top = tree.getroot()
    assert top.find("option") is None
    assert [c.tag for c in top.find("worldbody")] == ["body"]
    geoms = {g.get("name") for g in top.iter("geom")}
    assert geoms == {"base_visual", "shin_visual"}
    files = {m.get("name"): m.get("file") for m in top.iter("mesh")}
    # base_link.stl is one of the stock robot's meshes and is not copied.
    assert files == {
        "base_link": "base_link.stl",
        "vis_shin": "legs_v627/vis_shin.stl",
    }
    assert own == ["vis_shin.stl"]
    assert top.find("compiler").get("meshdir") == "../../meshes"
    assert "a CAD note" in ET.tostring(top, encoding="unicode")
