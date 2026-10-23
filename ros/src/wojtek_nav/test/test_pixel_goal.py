"""Desk tests for the pixel-goal resolver: the VLM's pixel becomes a setpoint.

Run: cd ros/src/wojtek_nav && PYTHONPATH=$PWD python3 -m pytest test/ -q

The camera numbers are the simulator's (camera_spec.py): colour 848x480
with fx = fy = 418, depth 424x240 with fx = fy = 209, both cx, cy at
(size - 1) / 2, both rendered from the same MJCF camera. The resolver
never sees a VLM; it sees a pixel, a depth image and intrinsics.
"""

import math

import numpy as np
import pytest

from wojtek_nav.pixel_goal import (
    GoalTracker,
    colour_to_depth_pixel,
    deproject,
    depth_at,
    nearest_frame,
    standoff,
)

K_COLOUR = (418.0, 0.0, 423.5, 0.0, 418.0, 239.5, 0.0, 0.0, 1.0)
K_DEPTH = (209.0, 0.0, 211.5, 0.0, 209.0, 119.5, 0.0, 0.0, 1.0)


# -- depth sampling --------------------------------------------------------

def _depth_image(value_m, holes=()):
    img = np.full((240, 424), int(value_m * 1000), dtype=np.uint16)
    for (v, u) in holes:
        img[v, u] = 0
    return img


def test_depth_is_the_patch_median_in_metres():
    img = _depth_image(1.5)
    z, frac = depth_at(img, u=200.4, v=100.6, radius=4)
    assert z == pytest.approx(1.5)
    assert frac == 1.0


def test_holes_are_ignored_and_reported():
    img = _depth_image(2.0, holes=[(100, 200), (101, 200), (100, 201)])
    z, frac = depth_at(img, u=200, v=100, radius=1)  # 3x3 patch, 3 holes
    assert z == pytest.approx(2.0)
    assert frac == pytest.approx(6 / 9)


def test_too_many_holes_gives_no_depth():
    img = _depth_image(2.0)
    img[95:106, 195:206] = 0
    z, frac = depth_at(img, u=200, v=100, radius=4)
    assert z is None
    assert frac == 0.0


def test_out_of_range_depth_counts_as_a_hole():
    img = _depth_image(5.0)  # beyond the D435's useful 3 m window
    z, frac = depth_at(img, u=200, v=100, radius=2, z_max=3.0)
    assert z is None


def test_edge_pixels_clip_the_patch_instead_of_failing():
    img = _depth_image(1.0)
    z, frac = depth_at(img, u=0, v=0, radius=4)
    assert z == pytest.approx(1.0)
    assert frac == 1.0


def test_float32_depth_is_already_metres():
    img = np.full((240, 424), 1.25, dtype=np.float32)
    z, _ = depth_at(img, u=10, v=10, radius=2)
    assert z == pytest.approx(1.25)


# -- pixel geometry --------------------------------------------------------

def test_colour_pixel_maps_through_the_viewing_ray():
    # The simulator renders colour and depth from one camera at 2:1, so the
    # ray through a colour pixel lands at half the offset from the centre.
    u_d, v_d = colour_to_depth_pixel(423.5, 239.5, K_COLOUR, K_DEPTH)
    assert (u_d, v_d) == pytest.approx((211.5, 119.5))
    u_d, v_d = colour_to_depth_pixel(823.5, 439.5, K_COLOUR, K_DEPTH)
    assert (u_d, v_d) == pytest.approx((411.5, 219.5))


def test_normalised_pixel_scales_with_the_colour_image():
    # u, v in [0, 1] of the colour image (what the VLM adapter emits).
    u_d, v_d = colour_to_depth_pixel(0.5 * 848, 0.5 * 480, K_COLOUR, K_DEPTH)
    assert u_d == pytest.approx(211.75)
    assert v_d == pytest.approx(119.75)


def test_deproject_is_the_pinhole_model_in_the_optical_frame():
    # Optical frame: x right, y down, z forward; the centre pixel is the axis.
    x, y, z = deproject(211.5, 119.5, 2.0, K_DEPTH)
    assert (x, y, z) == pytest.approx((0.0, 0.0, 2.0))
    x, y, z = deproject(211.5 + 209.0, 119.5, 2.0, K_DEPTH)
    assert (x, y, z) == pytest.approx((2.0, 0.0, 2.0))  # 45 deg right


# -- the goal in front of the object ----------------------------------------

def test_standoff_backs_off_along_the_line_from_the_robot():
    gx, gy, yaw = standoff(target=(3.0, 0.0), robot=(1.0, 0.0), distance=0.7)
    assert (gx, gy) == pytest.approx((2.3, 0.0))
    assert yaw == pytest.approx(0.0)


def test_standoff_faces_the_object_from_any_side():
    gx, gy, yaw = standoff(target=(1.0, 1.0), robot=(1.0, -1.0), distance=0.5)
    assert (gx, gy) == pytest.approx((1.0, 0.5))
    assert yaw == pytest.approx(math.pi / 2)


def test_standoff_closer_than_the_distance_stays_put_and_turns():
    gx, gy, yaw = standoff(target=(1.3, 0.0), robot=(1.0, 0.0), distance=0.7)
    assert (gx, gy) == pytest.approx((1.0, 0.0))
    assert yaw == pytest.approx(0.0)


# -- picking the depth frame by the picture's stamp ------------------------

def test_nearest_frame_by_stamp_within_the_skew():
    frames = [(10.00, "a"), (10.07, "b"), (10.13, "c")]
    assert nearest_frame(frames, stamp=10.06, max_skew=0.05) == "b"
    assert nearest_frame(frames, stamp=10.30, max_skew=0.05) is None
    assert nearest_frame([], stamp=10.0, max_skew=0.05) is None


# -- tracking the setpoint until goto has an answer ------------------------

def test_tracker_resends_once_a_second_until_reached():
    t = GoalTracker(repeat_s=1.0, blocked_hold_s=5.0, max_s=60.0)
    t.start(now=0.0)
    assert t.step(status="driving", now=0.0) == "send"
    assert t.step(status="driving", now=0.5) is None
    assert t.step(status="driving", now=1.0) == "send"
    assert t.step(status="reached", now=1.5) == "reached"
    assert t.done


def test_tracker_waits_out_a_transient_block():
    # goto turns while blocked and may go on: blocked is final only when
    # it stays for blocked_hold_s.
    t = GoalTracker(repeat_s=1.0, blocked_hold_s=5.0, max_s=60.0)
    t.start(now=0.0)
    t.step(status="driving", now=0.0)
    assert t.step(status="blocked", now=2.0) == "send"    # keeps talking while blocked
    assert t.step(status="turning", now=4.0) == "send"    # the block cleared
    assert t.step(status="blocked", now=5.0) == "send"    # blocked again: the clock restarts
    assert t.step(status="blocked", now=9.9) == "send"    # 4.9 s, not yet final
    assert t.step(status="blocked", now=10.1) == "blocked"
    assert t.done


def test_tracker_gives_up_after_max_s():
    t = GoalTracker(repeat_s=1.0, blocked_hold_s=5.0, max_s=10.0)
    t.start(now=100.0)
    t.step(status="driving", now=100.0)
    assert t.step(status="driving", now=110.5) == "timeout"
    assert t.done


def test_tracker_ignores_the_stale_status_before_goto_saw_the_goal():
    # goto's latched status may still say "reached" from the previous goal
    # when the new one goes out; a terminal word counts only after goto has
    # reported movement (or blocked) for THIS goal.
    t = GoalTracker(repeat_s=1.0, blocked_hold_s=5.0, max_s=60.0)
    t.start(now=0.0)
    assert t.step(status="reached", now=0.0) == "send"
    assert t.step(status="reached", now=0.3) is None
    assert not t.done
    assert t.step(status="driving", now=0.5) is None
    assert t.step(status="reached", now=0.8) == "reached"
