"""The video overlays draw from explicit inputs, with no model behind them.

`wojtek_rl/video/overlays.py` takes arrays and numbers and returns a frame, so
every panel is checkable here on synthetic inputs. The renderer that feeds it
(`video/render.py`) needs a scene and belongs in tests/integration.
"""

import argparse

import numpy as np
import pytest

from wojtek_rl import height_scan
from wojtek_rl.video import overlays
from wojtek_rl.video.render import display_grid, frame_size

SIZE = (640, 480)


def blank(size=SIZE):
    return np.zeros((size[1], size[0], 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "text,expected", [("640x480", (640, 480)), ("1280X720", (1280, 720))]
)
def test_frame_size_parses_wxh(text, expected):
    assert frame_size(text) == expected


@pytest.mark.parametrize("text", ["640", "640x", "640x0", "-640x480", "wide"])
def test_frame_size_rejects_junk(text):
    with pytest.raises(argparse.ArgumentTypeError):
        frame_size(text)


def test_compose_without_overlays_returns_the_frame_unchanged():
    frame = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    assert np.array_equal(overlays.compose(frame), frame)


def test_torque_strip_stays_in_the_bottom_band():
    frame = blank()
    out = overlays.compose(frame, torque=np.zeros(12), cap=9.0)
    assert out.shape == frame.shape
    half = frame.shape[0] // 2
    assert not out[:half].any()
    assert out[half:].any()


def test_torque_bars_grow_with_the_torque():
    def strip(value):
        return overlays.compose(blank(), torque=np.full(12, value), cap=9.0)

    idle = strip(0.0)

    def bar_pixels(value):
        return int(np.any(strip(value) != idle, axis=-1).sum())

    assert bar_pixels(8.0) > bar_pixels(2.0) > 0


def test_depth_inset_lands_top_right_with_its_scan_dots():
    depth = np.full((90, 120), 1.5, dtype=np.float32)
    out = overlays.compose(
        blank(), depth=overlays.depth_rgb(depth),
        scan_uv=np.array([[60.0, 45.0]]), visible=np.array([True]),
    )
    x0 = SIZE[0] - overlays.MARGIN - 120
    assert out[overlays.MARGIN:overlays.MARGIN + 90, x0:x0 + 120].any()
    assert not out[:, :x0 - 1].any()
    dot = out[overlays.MARGIN + 45, x0 + 60]
    assert tuple(int(c) for c in dot) == overlays.VISIBLE_RGB


def test_depth_rgb_flattens_pixels_outside_the_window():
    lo, hi = overlays.DEPTH_WINDOW
    depth = np.array([[lo - 0.1, (lo + hi) / 2, hi + 0.1]], dtype=np.float32)
    rgb = overlays.depth_rgb(depth)
    assert tuple(rgb[0, 0]) == overlays.OUT_OF_BAND_RGB
    assert tuple(rgb[0, 2]) == overlays.OUT_OF_BAND_RGB
    assert tuple(rgb[0, 1]) != overlays.OUT_OF_BAND_RGB


def test_scan_hold_heatmap_is_drawn_right_aligned():
    grid = np.linspace(-0.1, 0.1, height_scan.NX * height_scan.NY)
    out = overlays.compose(blank(), scan_hold=display_grid(grid), clip=0.15)
    cell = max(10, round(overlays.HEAT_CELL * overlays.scale_of(SIZE)))
    left = SIZE[0] - overlays.MARGIN - height_scan.NY * cell
    assert out[:, left:].any()
    assert not out[:, :left - 1].any()


def test_panels_scale_with_the_frame():
    assert overlays.scale_of((480, 360)) < 1.0 < overlays.scale_of((1920, 1440))
    assert overlays.scale_of((overlays.REF_W, overlays.REF_H)) == 1.0
    assert overlays.inset_width((480, 360)) < overlays.INSET_W


def test_header_lines_land_in_the_top_left():
    out = overlays.compose(blank(), header=["wojtek", "t 1.00 s"])
    assert out[:80, :200].any()
    assert not out[200:].any()


def test_display_grid_puts_the_far_row_first_and_plus_y_left():
    # body_grid indexes ix*ny + iy, x forward and y left; here value = index.
    grid = display_grid(np.arange(height_scan.NX * height_scan.NY))
    assert grid.shape == (height_scan.NX, height_scan.NY)
    assert grid[0, -1] == (height_scan.NX - 1) * height_scan.NY
    assert grid[-1, 0] == height_scan.NY - 1
