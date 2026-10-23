"""The safety envelope is exactly what the README promises. Model-free."""

import pytest

from wojtek_rai import limits


def test_validate_walk_accepts_every_direction_in_range():
    for d in limits.WALK_DIRECTIONS:
        assert limits.validate_walk(d, 1.0) == (d, 1.0)


def test_validate_walk_normalises_case_and_whitespace():
    assert limits.validate_walk("  Forward ", "2") == ("forward", 2.0)


@pytest.mark.parametrize("direction", ["backward", "stop", "", "up", None])
def test_validate_walk_rejects_unknown_direction(direction):
    with pytest.raises(ValueError, match="direction must be one of"):
        limits.validate_walk(direction, 1.0)


@pytest.mark.parametrize("seconds", [0.0, -1.0, limits.MOVE_MAX_SECONDS + 0.01, 60])
def test_validate_walk_rejects_out_of_range_seconds(seconds):
    with pytest.raises(ValueError, match="seconds must be between"):
        limits.validate_walk("forward", seconds)


def test_validate_walk_rejects_non_numeric_seconds():
    with pytest.raises(ValueError, match="must be a number"):
        limits.validate_walk("forward", "two")


def test_republish_is_well_inside_text_commander_deadman():
    assert limits.REPUBLISH_PERIOD_S * 2 < limits.TEXT_COMMANDER_DEADMAN_S


def test_only_nav_command_is_writable_and_cmd_vel_is_forbidden():
    assert limits.WRITABLE_TOPICS == (limits.NAV_COMMAND_TOPIC,)
    assert limits.CMD_VEL_TOPIC in limits.FORBIDDEN
    assert not set(limits.WRITABLE_TOPICS) & set(limits.FORBIDDEN)
    assert not set(limits.WRITABLE_SERVICES) & set(limits.FORBIDDEN)


def test_every_actuator_path_is_forbidden():
    for name in ("/wojtek/arm", "/wojtek/enable", "/wojtek/zero", "/wojtek/reset",
                 "/wojtek/joint_targets", "/sim/reset"):
        assert name in limits.FORBIDDEN


# --- colour transport (WiFi to the physical robot) --------------------------


def _limits_with_env(monkeypatch, **env):
    """The limits module re-imported under `env` (its constants are read at import)."""
    import importlib

    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    from types import SimpleNamespace

    from wojtek_rai import limits as mod

    # reload() re-executes the module in place, so snapshot its namespace
    # before restoring the environment (and the module) for the other tests.
    snapshot = SimpleNamespace(**vars(importlib.reload(mod)))
    monkeypatch.undo()
    importlib.reload(mod)
    return snapshot


def test_colour_topic_is_the_raw_stream_by_default(monkeypatch):
    mod = _limits_with_env(monkeypatch, WOJTEK_RAI_COLOR_TRANSPORT=None)
    assert mod.COLOR_IMAGE_TOPIC == "/camera/camera/color/image_raw"
    assert not mod.is_compressed_topic(mod.COLOR_IMAGE_TOPIC)


def test_compressed_transport_switches_every_colour_reference(monkeypatch):
    mod = _limits_with_env(monkeypatch, WOJTEK_RAI_COLOR_TRANSPORT="compressed")
    assert mod.COLOR_IMAGE_TOPIC == "/camera/camera/color/image_raw/compressed"
    assert mod.is_compressed_topic(mod.COLOR_IMAGE_TOPIC)
    assert mod.COLOR_IMAGE_TOPIC in mod.REQUIRED_TOPICS
    assert mod.COLOR_IMAGE_TOPIC in mod.READABLE_TOPICS
    assert "/camera/camera/color/image_raw" not in mod.READABLE_TOPICS


def test_unknown_colour_transport_is_refused(monkeypatch):
    import importlib

    import pytest

    from wojtek_rai import limits as mod

    monkeypatch.setenv("WOJTEK_RAI_COLOR_TRANSPORT", "theora")
    with pytest.raises(ValueError, match="WOJTEK_RAI_COLOR_TRANSPORT"):
        importlib.reload(mod)
    monkeypatch.undo()
    importlib.reload(mod)
