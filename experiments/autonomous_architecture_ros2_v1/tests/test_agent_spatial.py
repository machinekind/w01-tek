"""Pose-history route descriptions and occupancy-map text summaries."""

import math

from wojtek_eval.mapping import FREE, OCC, OnlineMap
from wojtek_agent.spatial import (
    PoseHistory,
    bearing_text,
    frontier_clusters,
    map_summary,
)


def make_map(n=40, res=0.05):
    return OnlineMap(res=res, origin=(-1.0, -1.0), shape=(n, n))


# -- PoseHistory ---------------------------------------------------------------


def test_describe_empty():
    assert "no route" in PoseHistory().describe(3.0)


def test_describe_straight_walk():
    h = PoseHistory()
    for k in range(51):  # 1 s at 50 Hz, 0.5 m straight +x
        h.add(k * 0.02, k * 0.01, 0.0, 0.0)
    text = h.describe(3.0)
    assert "0.50 m of path" in text
    assert "+0.50 m forward" in text


def test_describe_turn_in_place():
    h = PoseHistory()
    for k in range(51):
        h.add(k * 0.02, 0.0, 0.0, math.radians(k))  # 50 deg left, no motion
    text = h.describe(3.0)
    assert "in place" in text
    assert "left" in text


def test_describe_windowing():
    h = PoseHistory()
    h.add(0.0, 0.0, 0.0, 0.0)
    h.add(10.0, 5.0, 0.0, 0.0)
    h.add(10.5, 5.1, 0.0, 0.0)
    # 1 s window sees only the last two samples: 0.1 m, not 5.1 m.
    assert "0.10 m of path" in h.describe(1.0)


def test_lateral_move_reported():
    h = PoseHistory()
    h.add(0.0, 0.0, 0.0, 0.0)
    h.add(1.0, 0.3, 0.4, 0.0)  # forward-right in the start frame? +y = left
    text = h.describe(3.0)
    assert "+0.40 m to the left" in text


# -- frontiers -------------------------------------------------------------------


def test_frontier_clusters_none_on_blank_map():
    assert frontier_clusters(make_map()) == []


def test_frontier_clusters_finds_one_edge():
    m = make_map()
    m.state[10:20, 10:20] = FREE  # free block inside unknown -> ring frontier
    clusters = frontier_clusters(m, min_cells=4)
    assert len(clusters) == 1
    c = clusters[0]
    # Midpoint lands near the block's center in world coordinates.
    wx, wy = m.cell_to_world(14, 14)
    assert abs(c.x - wx) < 0.15 and abs(c.y - wy) < 0.15


def test_frontier_min_cells_drops_speckle():
    m = make_map()
    m.state[5, 5] = FREE  # lone free cell: 4ish frontier cells but tiny
    assert frontier_clusters(m, min_cells=6) == []


def test_two_separate_clusters():
    m = make_map()
    m.state[5:10, 5:10] = FREE
    m.state[25:30, 25:30] = FREE
    assert len(frontier_clusters(m, min_cells=4)) == 2


# -- summaries -------------------------------------------------------------------


def test_map_summary_counts_area():
    m = make_map()
    m.state[10:20, 10:20] = FREE  # 100 cells * 0.0025 m2 = 0.25 m2
    m.state[15, 15] = OCC
    text = map_summary(m, (0.0, 0.0, 0.0))
    assert "0.2 m2" in text
    assert "1 obstacle cells" in text
    assert "frontier" in text


def test_map_summary_closed_map():
    m = make_map()
    m.state[:] = FREE
    assert "fully explored" in map_summary(m, (0.0, 0.0, 0.0))


def test_bearing_text_ahead_and_behind():
    assert "ahead" in bearing_text(1.0, 0.0, (0.0, 0.0, 0.0))
    assert "behind" in bearing_text(-1.0, 0.0, (0.0, 0.0, 0.0))
    assert "left" in bearing_text(0.0, 1.0, (0.0, 0.0, 0.0))
