"""Named-place resolution and the Nav2 permission wiring. Model-free."""

import math

import pytest

from wojtek_rai import limits

rai = pytest.importorskip("rai")

from wojtek_rai.nav_tools import load_places  # noqa: E402


def test_places_are_pulled_back_towards_the_origin_and_face_the_object():
    places = load_places()
    ball = places["ball"]
    # scene_sim.xml: ball at (1.60, 0.58); standoff 0.6 along the ray from the origin
    dist_goal = math.hypot(ball["x"], ball["y"])
    assert math.isclose(dist_goal, math.hypot(1.60, 0.58) - 0.6, abs_tol=0.01)
    assert math.isclose(ball["yaw"], math.atan2(0.58, 1.60), abs_tol=0.01)


def test_home_has_no_standoff_and_explicit_yaw():
    home = load_places()["home"]
    assert home == {"x": 0.0, "y": 0.0, "yaw": 0.0}


def test_every_place_is_inside_the_workspace_bounds():
    for name, p in load_places().items():
        assert limits.WORKSPACE_MIN[0] <= p["x"] <= limits.WORKSPACE_MAX[0], name
        assert limits.WORKSPACE_MIN[1] <= p["y"] <= limits.WORKSPACE_MAX[1], name


def test_nav_action_is_writable_but_cmd_vel_stays_forbidden():
    assert limits.NAV_ACTION in limits.WRITABLE_ACTIONS
    assert limits.CMD_VEL_TOPIC in limits.FORBIDDEN


# --- load_places on a hand-written registry ---------------------------------------


def _registry(tmp_path, text):
    path = tmp_path / "places.yaml"
    path.write_text(text)
    return path


def test_load_places_applies_the_default_and_per_place_standoff(tmp_path):
    places = load_places(_registry(tmp_path, """
standoff: 0.6
places:
  far:   {x: 3.0, y: 4.0}
  near:  {x: 3.0, y: 4.0, standoff: 1.0}
  exact: {x: 3.0, y: 4.0, standoff: 0.0}
"""))
    # (3, 4) is 5 m out; pulled back along the ray from the origin.
    assert places["far"] == {"x": pytest.approx(3.0 * 4.4 / 5), "y": pytest.approx(4.0 * 4.4 / 5), "yaw": 0.927}
    assert places["near"] == {"x": pytest.approx(3.0 * 4.0 / 5), "y": pytest.approx(4.0 * 4.0 / 5), "yaw": 0.927}
    assert places["exact"] == {"x": 3.0, "y": 4.0, "yaw": 0.0}
    # The goal always faces the original point.
    assert places["far"]["yaw"] == pytest.approx(math.atan2(4.0, 3.0), abs=1e-3)


def test_load_places_keeps_a_place_inside_the_standoff_radius_where_it_is(tmp_path):
    places = load_places(_registry(tmp_path, "standoff: 0.6\nplaces:\n  close: {x: 0.3, y: 0.0}\n"))
    assert places["close"] == {"x": 0.3, "y": 0.0, "yaw": 0.0}


def test_load_places_keeps_an_explicit_yaw_after_the_pull_back(tmp_path):
    places = load_places(_registry(tmp_path, "standoff: 0.6\nplaces:\n  dock: {x: 2.0, y: 0.0, yaw: 3.1}\n"))
    assert places["dock"] == {"x": 1.4, "y": 0.0, "yaw": 3.1}


def test_load_places_without_a_standoff_or_places_section(tmp_path):
    assert load_places(_registry(tmp_path, "places:\n  a: {x: 1.0, y: 2.0}\n")) == {"a": {"x": 1.0, "y": 2.0, "yaw": 0.0}}
    assert load_places(_registry(tmp_path, "frame: map\n")) == {}
    assert load_places(_registry(tmp_path, "places:\n")) == {}


def test_load_places_rounds_to_millimetres(tmp_path):
    places = load_places(_registry(tmp_path, "places:\n  a: {x: 1.23456, y: -0.00049}\n"))
    assert places["a"] == {"x": 1.235, "y": -0.0, "yaw": 0.0}


def test_no_registry_means_no_places():
    """WOJTEK_RAI_PLACES=none (the physical robot has no map yet) is not a file."""
    assert load_places("none") == {}
    assert load_places("") == {}


def test_go_to_place_is_not_offered_without_a_registry(monkeypatch):
    from unittest.mock import MagicMock

    from rai.communication.ros2 import ROS2Connector

    from wojtek_rai.nav_tools import build_nav_tools
    from wojtek_rai.tools import _permissions

    monkeypatch.setattr(limits, "PLACES_FILE", "none")
    names = {t.name for t in build_nav_tools(MagicMock(spec=ROS2Connector), _permissions())}
    assert "go_to_place" not in names
    assert "navigate_to_pose" in names
