"""The closed-loop `turn` tool against a fake odometry yaw. No ROS runtime, no LLM.

The fake yaw follows the commands the tool publishes: every left/right pulse
advances it by a fixed step (a stall is a step of zero). Time is a fake clock
that advances by whatever the tool sleeps, so a stall or a timeout is reached
in a few loop iterations rather than seconds.

Needs the rai package (runs inside the wojtek_rai container, or any interpreter
with rai-core); skipped elsewhere.
"""

import math
from types import SimpleNamespace

import pytest

rai = pytest.importorskip("rai")

from test_tools import make_fake_connector  # noqa: E402

from wojtek_rai import limits  # noqa: E402
from wojtek_rai import tools  # noqa: E402
from wojtek_rai.tools import (  # noqa: E402
    NO_ODOMETRY_POSE_DESCRIPTION,
    TurnTool,
    _permissions,
    _wrap_angle,
    build_tools,
)

PULSE_S = limits.REPUBLISH_PERIOD_S / 2  # the turn loop's sleep per pulse


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """Fake time inside wojtek_rai.tools: sleeping advances monotonic()."""
    state = {"t": 0.0, "slept": []}

    def sleep(s):
        state["t"] += s
        state["slept"].append(s)

    monkeypatch.setattr(tools.time, "sleep", sleep)
    monkeypatch.setattr(tools.time, "monotonic", lambda: state["t"])
    return state


def _yaw_transform(yaw):
    q = SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
    return SimpleNamespace(transform=SimpleNamespace(rotation=q))


def make_odometry(connector, step_rad, start=0.0):
    """odom->base_link yaw = start + step * (#left - #right pulses published)."""

    def get_transform(target, source, timeout_sec=2.0):
        assert (target, source) == (limits.ODOM_FRAME, limits.BASE_FRAME)
        commands = [p["data"] for _, _, p in connector.published]
        n = commands.count("left") - commands.count("right")
        return _yaw_transform(start + n * step_rad)

    connector.get_transform.side_effect = get_transform


def _commands(connector):
    return [p["data"] for _, _, p in connector.published]


def _turn(connector):
    return TurnTool(connector=connector, **_permissions())


@pytest.fixture
def connector():
    return make_fake_connector()


# --- _wrap_angle -------------------------------------------------------------


@pytest.mark.parametrize(
    "angle, expected",
    [(0.0, 0.0), (1.0, 1.0), (-1.0, -1.0), (math.pi + 0.5, -math.pi + 0.5), (-math.pi - 0.5, math.pi - 0.5),
     (2 * math.pi, 0.0)],
)
def test_wrap_angle_maps_into_minus_pi_pi(angle, expected):
    assert _wrap_angle(angle) == pytest.approx(expected, abs=1e-9)


# --- closed loop ---------------------------------------------------------------


def test_turn_left_pulses_until_the_yaw_reaches_the_target_minus_the_stop_lead(connector):
    make_odometry(connector, step_rad=0.2)

    out = _turn(connector)._run(degrees=90.0)

    commands = _commands(connector)
    assert commands[-1] == limits.STOP_COMMAND
    assert set(commands[:-1]) == {"left"}
    # Stop once turned >= 90 deg - TURN_STOP_LEAD_RAD: 1.571 - 0.25 = 1.321 rad, 7 pulses of 0.2.
    expected_pulses = math.ceil((math.radians(90.0) - limits.TURN_STOP_LEAD_RAD) / 0.2)
    assert len(commands) - 1 == expected_pulses
    assert out.startswith(f"Turned +{math.degrees(expected_pulses * 0.2):.0f} degrees (target +90)")
    assert "stopped" in out


def test_turn_right_is_a_negative_angle(connector):
    make_odometry(connector, step_rad=0.2)

    out = _turn(connector)._run(degrees=-90.0)

    commands = _commands(connector)
    assert commands[-1] == limits.STOP_COMMAND
    assert set(commands[:-1]) == {"right"}
    assert "Turned -" in out and "(target -90)" in out


def test_turned_angle_is_measured_across_the_pi_wrap(connector):
    # Start at +170 deg: a 30 deg left turn crosses +180 and the raw yaw jumps
    # to about -160; the wrapped difference must stay a small positive angle.
    make_odometry(connector, step_rad=0.1, start=math.radians(170.0))

    out = _turn(connector)._run(degrees=30.0)

    commands = _commands(connector)
    assert commands[-1] == limits.STOP_COMMAND
    pulses = len(commands) - 1
    assert pulses == math.ceil((math.radians(30.0) - limits.TURN_STOP_LEAD_RAD) / 0.1)
    assert out.startswith(f"Turned +{math.degrees(pulses * 0.1):.0f} degrees")


def test_tiny_turn_is_skipped_without_publishing(connector):
    make_odometry(connector, step_rad=0.2)

    out = _turn(connector)._run(degrees=1.0)

    assert "skipped" in out
    assert connector.published == []
    connector.get_transform.assert_not_called()


def test_target_is_clamped_to_180_degrees(connector):
    make_odometry(connector, step_rad=0.5)

    out = _turn(connector)._run(degrees=720.0)

    assert "(target +180)" in out
    assert _commands(connector)[-1] == limits.STOP_COMMAND


# --- guards ----------------------------------------------------------------------


def test_static_yaw_aborts_after_the_stall_window_and_stops(connector):
    make_odometry(connector, step_rad=0.0)  # the physical robot: odom->base_link is static

    out = _turn(connector)._run(degrees=90.0)

    commands = _commands(connector)
    assert "aborted" in out and "walk left/right" in out
    assert commands[-1] == limits.STOP_COMMAND
    pulses = len(commands) - 1
    # Stops just after TURN_STALL_SECONDS, long before TURN_MAX_SECONDS.
    assert pulses <= math.ceil(tools.TURN_STALL_SECONDS / PULSE_S) + 1
    assert pulses < limits.TURN_MAX_SECONDS / PULSE_S / 2


def test_slow_yaw_is_not_a_stall_but_times_out_at_turn_max_seconds(connector, clock):
    # 0.01 rad per pulse: past TURN_STALL_RAD within the stall window, yet
    # nowhere near 180 deg by TURN_MAX_SECONDS.
    make_odometry(connector, step_rad=0.01)

    out = _turn(connector)._run(degrees=180.0)

    commands = _commands(connector)
    assert "aborted" not in out
    assert out.startswith("Turned +")
    assert commands[-1] == limits.STOP_COMMAND
    assert clock["t"] == pytest.approx(limits.TURN_MAX_SECONDS + PULSE_S + 0.5, abs=PULSE_S)


def test_stop_is_sent_when_the_yaw_lookup_fails_mid_turn(connector):
    connector.get_transform.side_effect = [
        _yaw_transform(0.0),
        _yaw_transform(0.2),
        RuntimeError("odom->base_link lookup timed out"),
    ]

    with pytest.raises(RuntimeError, match="lookup timed out"):
        _turn(connector)._run(degrees=90.0)

    commands = _commands(connector)
    assert commands[-1] == limits.STOP_COMMAND
    assert commands[:-1] == ["left", "left"]


def test_turn_refuses_when_nav_command_is_not_writable(connector):
    make_odometry(connector, step_rad=0.2)
    tool = TurnTool(connector=connector, writable=[], forbidden=list(limits.FORBIDDEN))

    with pytest.raises(ValueError, match="not writable"):
        tool._run(degrees=90.0)
    assert connector.published == []


# --- build_tools: no odometry, no turn tool ---------------------------------------


def test_build_tools_offers_turn_only_with_odometry(connector):
    with_odom = {t.name: t for t in build_tools(connector, nav=False, odometry=True)}
    without = {t.name: t for t in build_tools(connector, nav=False, odometry=False)}

    assert "turn" in with_odom
    assert "turn" not in without
    assert without["get_robot_position"].description == NO_ODOMETRY_POSE_DESCRIPTION
    assert with_odom["get_robot_position"].description != NO_ODOMETRY_POSE_DESCRIPTION
    assert set(without) == set(with_odom) - {"turn"}


def test_read_only_build_has_no_actuators_at_all(connector):
    names = {t.name for t in build_tools(connector, read_only=True, nav=True, odometry=True)}
    # RAI's wait tool keeps its class name as its tool name.
    assert names == {"get_robot_position", "get_camera_image", "WaitForSecondsTool"}
