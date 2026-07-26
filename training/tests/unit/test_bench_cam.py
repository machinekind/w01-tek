"""The benchmark camera: exists in the built model, selectable in RoomSim.

Skipped until the room assets exist (./run.sh room-assets + build-room).
"""

import base64
import io

import pytest

from wojtek_rl import paths

pytestmark = pytest.mark.skipif(
    not (
        paths.ROOM_MANIFEST.exists()
        and paths.ROOM_SCENE_XML.exists()
        and paths.POLICY_NPZ.exists()
    ),
    reason="room assets not built (run.sh room-assets + build-room)",
)


@pytest.fixture(scope="module")
def bench_sim():
    from wojtek_rl.room_app import RoomSim

    return RoomSim(paths.ROOM_SCENE_XML, paths.POLICY_NPZ, vlm_cam="bench")


def test_bench_camera_defined_with_vlnce_geometry(bench_sim):
    import mujoco

    model = bench_sim.model
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "bench")
    assert cam_id >= 0
    # 90 deg HFOV at the 4:3 render -> 73.74 deg vertical FOV.
    assert model.cam_fovy[cam_id] == pytest.approx(73.74)
    # Mast height: ~1.25 m above the floor with the base at ~0.14 m.
    assert model.cam_pos[cam_id][2] == pytest.approx(1.11)


def test_vlm_view_renders_from_selected_camera(bench_sim):
    from PIL import Image

    jpg = base64.b64decode(bench_sim.ego_jpeg())
    img = Image.open(io.BytesIO(jpg))
    # VLM frame is the shared 640x480 renderer output (ego RGB + minimap HUD).
    assert img.size == (640, 480)


def test_unknown_camera_rejected():
    from wojtek_rl.room_app import RoomSim

    with pytest.raises(ValueError, match="camera"):
        RoomSim(paths.ROOM_SCENE_XML, paths.POLICY_NPZ, vlm_cam="selfie")
