"""Room scene checks. Skipped until `./run.sh room-assets && ./run.sh
build-room` has produced the (gitignored) assets and scene_room.xml."""

import mujoco
import pytest

from wojtek_rl import paths

pytestmark = pytest.mark.skipif(
    not (paths.ROOM_MANIFEST.exists() and paths.ROOM_SCENE_XML.exists()),
    reason="room assets not built (run.sh room-assets + build-room)",
)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(paths.ROOM_SCENE_XML))


def test_scene_compiles_with_home_key_and_ego_cam(model):
    assert model.key("home").id >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track") >= 0


def test_floor_plane_pairing(model):
    floor = model.geom("floor")
    assert floor.conaffinity[0] == 15
    assert model.geom_group[floor.id] == 3  # invisible in default rendering


def test_room_collision_geoms(model):
    room = model.body("room")
    col = [
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == room.id and model.geom_contype[i] != 0
    ]
    assert col, "room has no collision geometry"
    for i in col:
        assert model.geom_conaffinity[i] == 15
        assert model.geom_group[i] == 3


def test_spawn_settles_standing(model):
    """After settling, the robot stands on the room floor near the spawn:
    no deep penetration, base at a standing height."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert 0.07 < data.qpos[2] < 0.20, (
        f"base settled at z={data.qpos[2]:.3f} -- robot is not standing; move "
        "the room (./run.sh build-room --room-offset X Y YAW)"
    )
    if data.ncon:
        assert data.contact.dist[: data.ncon].min() > -0.03
