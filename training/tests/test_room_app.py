"""RoomSim integration tests. Skipped until the room assets are built;
render tests additionally need a working GL backend (EGL/OSMesa)."""

import base64
import io
import math

import numpy as np
import pytest

from wojtek_rl import paths

pytestmark = pytest.mark.skipif(
    not (
        paths.ROOM_MANIFEST.exists()
        and paths.ROOM_SCENE_XML.exists()
        and paths.POLICY_NPZ.exists()
    ),
    reason="room assets or exported policy missing",
)


@pytest.fixture(scope="module")
def sim():
    from wojtek_rl.room_app import RoomSim

    return RoomSim(paths.ROOM_SCENE_XML, paths.POLICY_NPZ)


def test_forward_command_moves_robot(sim):
    sim.reset()
    x0, y0, yaw0 = sim.pose()
    ack = sim.submit_command("forward 0.5")
    assert ack["ok"], ack
    for _ in range(200):  # 4 s at 50 Hz
        sim.step()
    x1, y1, _ = sim.pose()
    along = (x1 - x0) * math.cos(yaw0) + (y1 - y0) * math.sin(yaw0)
    assert along > 0.2, f"robot only moved {along:.3f} m along heading"


def test_bad_command_rejected(sim):
    ack = sim.submit_command("fly 3")
    assert not ack["ok"]
    assert "expected" in ack["error"]


def test_render_pair_decodes(sim):
    from PIL import Image

    sim.reset()
    sim.step()
    chase_b64, ego_b64 = sim.render_pair()
    # Chase is rendered high-res for the browser; ego stays VGA for the VLM.
    for b64, shape in ((chase_b64, (768, 1024, 3)), (ego_b64, (480, 640, 3))):
        img = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))
        assert img.shape == shape
        assert img.std() > 5.0, "frame is (nearly) constant -- camera sees nothing"
