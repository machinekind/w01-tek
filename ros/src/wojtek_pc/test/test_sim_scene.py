"""The simulation scene with its props (config/scene_sim.xml).

Two things have to keep holding, and neither is obvious from reading the
XML:

* the props must not add a single number to the generalized position. The
  physics plugin publishes /sim/qpos and the camera node copies it into its
  own model; the two agree only because every prop is a static body.
* the pictures the two signs wear have to survive the trip through the
  staging directory, which is what the camera node and the plugin both hand
  to MuJoCo. Staging used to take the XMLs and leave everything else behind.

Run inside the dev container, where mujoco and the ament index exist:

    docker exec wojtek_robot python3 -m pytest /ros2_ws/src/wojtek_pc/test -q
"""

import re
import shutil
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

mujoco = pytest.importorskip("mujoco")

CONFIG = PKG / "config"
MESHES = PKG.parent / "wojtek_description" / "meshes"
PROPS = (
    "prop_ball",
    "prop_hydrant",
    "prop_traffic_light",
    "prop_stop_sign",
    "prop_clock",
    "prop_person",
)


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """config/ copied with meshdir pointed at the source-tree meshes.

    The same shape of copy the camera node and the plugin make at run time,
    props directory included.
    """
    out = tmp_path_factory.mktemp("wojtek_scene")
    for entry in CONFIG.iterdir():
        if entry.is_dir():
            shutil.copytree(entry, out / entry.name)
        elif entry.suffix == ".xml":
            (out / entry.name).write_text(
                re.sub(r'meshdir="[^"]*"', f'meshdir="{MESHES}"', entry.read_text())
            )
    return out


def _load(staged, name):
    return mujoco.MjModel.from_xml_path(str(staged / name))


def test_props_leave_the_generalized_position_alone(staged):
    empty = _load(staged, "scene_mjx.xml")
    furnished = _load(staged, "scene_sim.xml")
    assert (furnished.nq, furnished.nv, furnished.nu) == (
        empty.nq, empty.nv, empty.nu
    )


def test_every_prop_is_there(staged):
    model = _load(staged, "scene_sim.xml")
    for name in PROPS:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0, name


def test_the_signs_wear_their_pictures(staged):
    """A file texture that fails to load is a compile error, so getting this
    far is most of the test; the names say both pictures arrived."""
    model = _load(staged, "scene_sim.xml")
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE, i)
        for i in range(model.ntex)
    }
    assert {"stop_face", "clock_face"} <= names


def test_the_robot_can_walk_into_the_props(staged):
    """Each prop needs a geom that collides, or the robot walks through it."""
    model = _load(staged, "scene_sim.xml")
    for name in PROPS:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        first = model.body_geomadr[body]
        geoms = range(first, first + model.body_geomnum[body])
        assert any(model.geom_contype[g] for g in geoms), name


def test_staging_brings_the_props_along():
    """The camera node's own staging has to carry props/, not just XMLs."""
    pytest.importorskip("rclpy")
    from wojtek_pc.sim_camera_node import _staged_scene

    scene = Path(_staged_scene(str(CONFIG / "scene_sim.xml")))
    assert (scene.parent / "props" / "stop_sign.png").is_file()
    # And the staged copy still compiles, which is the whole point of it.
    assert mujoco.MjModel.from_xml_path(str(scene)).ngeom > 0
