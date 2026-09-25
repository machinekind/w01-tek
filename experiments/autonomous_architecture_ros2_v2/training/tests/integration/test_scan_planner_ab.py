"""SCAN planner vs straight march on the real scanned scene.

Integration: builds the MuJoCo scene and renders depth, so it lives outside
tests/unit (see tests/unit/test_suite_split.py). Skipped when the room assets
are not built.
"""

import math  # noqa: F401  (kept for parity with the unit module helpers)

import pytest


def _scene_available(name="room") -> bool:
    from wojtek_rl import paths

    return (paths.scene_dir(name) / "occupancy.npz").exists() and paths.scene_xml(
        name
    ).exists()


@pytest.mark.skipif(not _scene_available(), reason="room assets not built")
def test_kinsim_planner_beats_the_straight_march():
    """The headline claim, on the real scanned room: same oracle-VLM guidance,
    straight march collides, planner arrives."""
    import os

    os.environ.setdefault("MUJOCO_GL", "cgl")
    from wojtek_eval.gridmap import GridMap
    from wojtek_rl import paths
    from wojtek_rl.scan_bench import generate_episodes, run_episode

    grid = GridMap.load(paths.scene_dir("room") / "occupancy.npz")
    episodes = generate_episodes(grid, 2, seed=1)
    if not episodes:
        pytest.skip("no blocked-line episodes in this scene")
    for ep in episodes:
        straight, _ = run_episode("room", ep, planner=False)
        scan, _ = run_episode("room", ep, planner=True)
        assert straight.collisions > 0, "episode was supposed to be blocked"
        assert scan.collisions == 0, f"planner collided on ep{ep.idx}"
        assert scan.final_dist_m < straight.final_dist_m
