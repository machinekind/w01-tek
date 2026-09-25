"""Integration: a scripted fake VLM drives the real RoomSim end to end.

Skipped until the room assets exist (./run.sh room-assets + build-room).
Mimics the websocket loop with a 50 Hz pump task; no API key needed.
"""

import asyncio
import math

import pytest

from wojtek_rl import paths

pytestmark = pytest.mark.skipif(
    not (
        paths.ROOM_MANIFEST.exists()
        and paths.ROOM_SCENE_XML.exists()
    ),
    reason="room assets not built (run.sh room-assets + build-room)",
)


@pytest.fixture(scope="module")
def sim():
    from wojtek_rl.room_app import RoomSim

    try:
        return RoomSim(paths.ROOM_SCENE_XML, paths.DEFAULT_POLICY)
    except Exception as e:  # policy unresolvable (offline, no HF cache/token)
        pytest.skip(f"policy {paths.DEFAULT_POLICY} unavailable: {e}")


def test_fake_vlm_drives_room_sim(sim):
    from wojtek_rl.vlm_nav import VlmDecision, VlmNavigator

    class FakeClient:
        def __init__(self):
            self.script = [
                VlmDecision("turn_left", 30.0, "scan"),
                VlmDecision("forward", 0.5, "walk"),
                VlmDecision("done", None, "arrived"),
            ]
            self.egos = []

        async def decide(self, goal, ego_b64, history, step, max_steps, pose):
            self.egos.append(ego_b64)
            return self.script.pop(0)

    sim.reset()
    x0, y0, _ = sim.pose()
    client = FakeClient()
    nav = VlmNavigator(sim, client, cmd_timeout_s=30.0, poll_s=0.02)

    async def main():
        async def pump():  # stands in for the ws control loop
            while True:
                sim.step()
                await asyncio.sleep(0.02)

        pump_task = asyncio.create_task(pump())
        try:
            await asyncio.wait_for(nav._run("go to the bed"), timeout=120)
        finally:
            pump_task.cancel()

    asyncio.run(main())

    assert nav.status()["state"] == "done"
    assert nav.status()["reason"] == "vlm_done"
    x1, y1, _ = sim.pose()
    assert math.hypot(x1 - x0, y1 - y0) > 0.2, "robot did not move"
    assert len(client.egos) == 3
    assert all(len(e) > 100 for e in client.egos), "ego frames look empty"
