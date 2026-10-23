"""The pure helpers of perception_tools (depth ROI/median, depth decoding,
sighting text) and the projection maths of `_detect` / `go_to_object`
(bearing, distance, map point, goal in front of the object). No ROS runtime,
no GPU, no LLM.

Needs the rai package (runs inside the wojtek_rai container, or any interpreter
with rai-core); skipped elsewhere.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

rai = pytest.importorskip("rai")

from test_nav_timeout import _ActionClient, _Future, _GoalHandle  # noqa: E402
from test_perception_tools import (  # noqa: E402
    PITCH,
    FakeClock,
    _detection,
    _quat,
    _service,
    _transform,
    make_connector,
)

from wojtek_rai import limits  # noqa: E402
from wojtek_rai import nav_tools  # noqa: E402
from wojtek_rai import perception_tools as pt  # noqa: E402
from wojtek_rai.perception_tools import (  # noqa: E402
    FindObjectsTool,
    GoToObjectTool,
    Sighting,
    _depth_array,
    _describe,
    _median_depth,
    _roi_depths,
)
from wojtek_rai.tools import _permissions  # noqa: E402

# make_connector's colour intrinsics: fx 420, cx 320, cy 180 on a 640x360 image.
FX_C, CX_C, CY_C = 420.0, 320.0, 180.0
CAMERA_X, CAMERA_Z = 0.32, 0.07  # base_link -> camera_link translation in map_to_optical()


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(pt, "time", fake)
    monkeypatch.setattr(pt, "RAIGroundingDino", SimpleNamespace(Request=SimpleNamespace))
    return fake


# --- _roi_depths / _median_depth -----------------------------------------------------


def test_roi_depths_keeps_only_returns_inside_the_valid_range():
    depth = np.full((240, 424), 2.0)
    depth[120, 212] = 0.0                       # no return
    depth[121, 212] = pt.MIN_VALID_DEPTH_M      # boundary is exclusive
    depth[122, 212] = pt.MAX_VALID_DEPTH_M
    depth[123, 212] = 9.0                       # beyond the D435's useful range
    depth[124, 212] = 0.5

    valid = _roi_depths(depth, 212, 122, 2, 10, fraction=1.0)

    assert set(np.unique(valid)) == {0.5, 2.0}
    assert (valid == 0.5).sum() == 1


def test_roi_depths_clips_the_box_at_the_image_border():
    depth = np.arange(240 * 424, dtype=float).reshape(240, 424) * 0 + 1.0
    # A 40x40 box centred on the top-left corner: only the in-image quarter counts.
    corner = _roi_depths(depth, 0, 0, 40, 40, fraction=1.0)
    assert corner.size == 21 * 21
    # A box straddling the bottom-right corner is clipped the same way.
    far = _roi_depths(depth, 423, 239, 40, 40, fraction=1.0)
    assert far.size == 21 * 21
    # A 1x1 box still reads at least one pixel (half-width clamped to 1).
    assert _roi_depths(depth, 212, 120, 1, 1, fraction=1.0).size == 3 * 3


def test_median_depth_uses_the_central_half_of_the_box_first():
    depth = np.full((240, 424), 3.0)            # a wall behind the object
    depth[110:131, 202:223] = 1.0               # the object fills the central 50%

    # Central patch (half-width 10 -> 21 px) is pure object.
    assert _median_depth(depth, 212, 120, 40, 40) == pytest.approx(1.0)
    # The whole box mixes object and wall; the central patch must win.
    assert np.median(depth[100:141, 192:233]) == pytest.approx(3.0)


def test_median_depth_is_none_when_nothing_in_the_box_is_valid():
    depth = np.zeros((240, 424))
    assert _median_depth(depth, 212, 120, 40, 40) is None


# --- _depth_array ----------------------------------------------------------------------


def test_depth_array_decodes_16uc1_millimetres_to_metres():
    mm = np.array([[1500, 0], [250, 6000]], dtype=np.uint16)
    msg = SimpleNamespace(encoding="16UC1", height=2, width=2, data=mm.tobytes())
    assert _depth_array(msg).tolist() == [[1.5, 0.0], [0.25, 6.0]]


def test_depth_array_passes_32fc1_metres_through():
    m = np.array([[1.5, 0.0], [0.25, 6.0]], dtype=np.float32)
    msg = SimpleNamespace(encoding="32FC1", height=2, width=2, data=m.tobytes())
    assert np.array_equal(_depth_array(msg), m)


def test_depth_array_rejects_an_unknown_encoding():
    msg = SimpleNamespace(encoding="rgb8", height=1, width=1, data=b"\x00\x00\x00")
    with pytest.raises(RuntimeError, match="unsupported depth encoding"):
        _depth_array(msg)


# --- _describe -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bearing, side",
    [(0.0, "ahead"), (0.04, "ahead"), (-0.04, "ahead"), (0.3, "left"), (-0.3, "right")],
)
def test_describe_names_the_side_from_the_bearing(bearing, side):
    text = _describe(Sighting("ball", 0.9, bearing, 2.0, (1.0, 0.5)))
    assert text.startswith("ball (score 0.90): 2.00 m")
    assert f"{abs(math.degrees(bearing)):.0f} deg {side}" in text
    assert text.endswith("map (1.00, 0.50)")


def test_describe_without_depth_or_map_says_so():
    text = _describe(Sighting("ball", 0.5, 0.0, None, None))
    assert "distance unknown" in text
    assert "map" not in text


# --- _detect: bearing, distance, map point -----------------------------------------------


def _find(connector):
    return FindObjectsTool(connector=connector, **_permissions())


def test_detect_projects_the_image_centre_straight_ahead(clock):
    connector = make_connector(clock)  # depth 3.0 m everywhere
    _service(connector, [_detection("ball", CX_C, CY_C, 40, 40)])

    sightings, err = _find(connector)._detect(["ball"])

    assert err is None
    (s,) = sightings
    assert s.name == "ball"
    assert s.bearing_rad == pytest.approx(0.0)
    assert s.distance_m == pytest.approx(3.0)
    # 3 m along the optical axis, pitched 15 deg down from the camera mount.
    assert s.map_xy == pytest.approx((CAMERA_X + 3.0 * math.cos(PITCH), 0.0), abs=1e-6)


def test_detect_bearing_and_distance_follow_the_viewing_ray(clock):
    connector = make_connector(clock)
    # One focal length to the right of centre: tx = 1, i.e. 45 deg right.
    _service(connector, [_detection("ball", CX_C + FX_C, CY_C, 40, 40)])

    (s,), _ = _find(connector)._detect(["ball"])

    assert s.bearing_rad == pytest.approx(-math.pi / 4)
    assert s.distance_m == pytest.approx(3.0 * math.sqrt(2.0))
    # Optical x (right) is -y in base_link; z stays on the pitched axis.
    assert s.map_xy == pytest.approx((CAMERA_X + 3.0 * math.cos(PITCH), -3.0), abs=1e-6)


def test_detect_sorts_by_score_and_skips_detections_without_results(clock):
    connector = make_connector(clock)
    _service(connector, [
        _detection("ball", CX_C, CY_C, 40, 40, score=0.4),
        SimpleNamespace(results=[], bbox=None),
        _detection("hydrant", CX_C, CY_C, 40, 40, score=0.8),
    ])

    sightings, _ = _find(connector)._detect(["ball", "hydrant"])

    assert [s.name for s in sightings] == ["hydrant", "ball"]


def test_detect_without_depth_returns_no_distance_and_no_map_point(clock):
    connector = make_connector(clock, depth_arr=np.zeros((240, 424)))
    _service(connector, [_detection("ball", CX_C, CY_C, 40, 40)])

    (s,), err = _find(connector)._detect(["ball"])

    assert err is None
    assert s.distance_m is None and s.map_xy is None
    assert "distance unknown" in _find(connector)._run(["ball"])


# --- go_to_object: the goal in front of the object ------------------------------------------


def _goto(connector):
    return GoToObjectTool(
        connector=connector, frame_id=limits.MAP_FRAME, action_name=limits.NAV_ACTION, **_permissions()
    )


def _robot_at_origin(connector):
    """map->base_link identity; map->optical is the URDF chain as before."""
    from test_perception_tools import map_to_optical

    def get_transform(target, source, timeout_sec=2.0):
        assert target == limits.MAP_FRAME
        return _transform(0.0, 0.0, 0.0, _quat()) if source == limits.BASE_FRAME else map_to_optical()

    connector.get_transform.side_effect = get_transform


def _capture(goals):
    """Stub for `_TimedNavMixin._navigate` (the goal path of go_to_object)
    that records the (x, y, yaw) it was asked for instead of sending it."""
    return lambda self, x, y, yaw: goals.append({"x": x, "y": y, "yaw": yaw}) or "sent"


def test_go_to_object_stops_the_standoff_short_of_the_object_facing_it(clock, monkeypatch):
    goals = []
    monkeypatch.setattr(pt.GoToObjectTool, "_navigate", _capture(goals))
    connector = make_connector(clock)
    _robot_at_origin(connector)
    _service(connector, [_detection("ball", CX_C, CY_C, 40, 40)])

    out = _goto(connector)._run("ball")

    (goal,) = goals
    object_x = CAMERA_X + 3.0 * math.cos(PITCH)
    assert goal["x"] == pytest.approx(object_x - limits.OBJECT_STANDOFF_M, abs=1e-6)
    assert goal["y"] == pytest.approx(0.0, abs=1e-6)
    assert goal["yaw"] == pytest.approx(0.0, abs=1e-6)
    assert out.endswith("sent")


def test_go_to_object_never_backs_past_the_robot_when_the_object_is_close(clock, monkeypatch):
    goals = []
    monkeypatch.setattr(pt.GoToObjectTool, "_navigate", _capture(goals))
    # 0.25 m ahead of the camera: the object is inside the 0.6 m standoff.
    connector = make_connector(clock, depth_arr=np.full((240, 424), 0.25))
    _robot_at_origin(connector)
    _service(connector, [_detection("ball", CX_C, CY_C, 40, 40)])

    _goto(connector)._run("ball")

    (goal,) = goals
    assert (goal["x"], goal["y"]) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_go_to_object_refuses_without_a_usable_distance(clock, monkeypatch):
    goals = []
    monkeypatch.setattr(pt.GoToObjectTool, "_navigate", _capture(goals))
    connector = make_connector(clock, depth_arr=np.zeros((240, 424)))
    _service(connector, [_detection("ball", CX_C, CY_C, 40, 40)])

    out = _goto(connector)._run("ball")

    assert "not detected with a usable distance" in out
    assert goals == []


# --- go_to_object: the goal goes through _TimedNavMixin, with its guards -------------------


def _centred_ball(clock, monkeypatch, **connector_kw):
    """A ball 3 m ahead of a robot at the map origin, with `_navigate` left
    real: the fake ActionClient from test_nav_timeout is what it reaches."""
    from builtin_interfaces.msg import Time

    connector = make_connector(clock, **connector_kw)
    _robot_at_origin(connector)
    # _TimedNavMixin._goal stamps a real PoseStamped, which rejects a MagicMock.
    connector.node.get_clock.return_value.now.return_value.to_msg.return_value = Time()
    _service(connector, [_detection("ball", CX_C, CY_C, 40, 40)])
    monkeypatch.setattr(limits, "NAV_GOAL_TIMEOUT_S", 0.05)
    monkeypatch.setattr(nav_tools, "NAV_SERVER_TIMEOUT_S", 0.05)
    return connector


def test_go_to_object_refuses_a_goal_outside_the_workspace_before_any_client_is_made(clock, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ActionClient must not be created")

    monkeypatch.setattr(nav_tools, "ActionClient", boom)
    # The default 8 m box cannot be left with a valid (< 6 m) depth return
    # from the origin, so shrink it under the ~2.3 m goal.
    monkeypatch.setattr(limits, "WORKSPACE_MAX", (1.0, 1.0, 1.0))
    connector = _centred_ball(clock, monkeypatch)

    out = _goto(connector)._run("ball")

    assert "outside the workspace" in out and "refused" in out
    assert nav_tools._ACTIVE["handle"] is None


def test_go_to_object_refuses_when_the_nav_action_is_not_writable(clock, monkeypatch):
    monkeypatch.setattr(nav_tools, "ActionClient", lambda *a, **k: pytest.fail("no client"))
    connector = _centred_ball(clock, monkeypatch)
    perms = {**_permissions(), "writable": list(limits.WRITABLE_TOPICS)}
    tool = GoToObjectTool(connector=connector, frame_id=limits.MAP_FRAME, action_name=limits.NAV_ACTION, **perms)

    assert "not writable" in tool._run("ball")


def test_go_to_object_goal_is_bounded_by_the_nav_timeout_and_cancellable(clock, monkeypatch):
    seen = []

    class _Handle(_GoalHandle):
        def get_result_async(self):
            seen.append(nav_tools._ACTIVE["handle"])  # in flight: cancel_navigation can reach it
            return super().get_result_async()

    handle = _Handle(_Future(done=False))
    monkeypatch.setattr(nav_tools, "ActionClient", lambda *a, **k: _ActionClient(handle))
    connector = _centred_ball(clock, monkeypatch)

    out = _goto(connector)._run("ball")

    assert "timed out" in out and "cancelled" in out
    assert handle.cancelled == 1
    assert seen == [handle]
    assert nav_tools._ACTIVE["handle"] is None
