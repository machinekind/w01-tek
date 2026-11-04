from types import SimpleNamespace

import pytest

from launch import LaunchContext
from wojtek_bringup.command_routing import command_topic
from wojtek_bringup.real_io_node import RealIoNode


@pytest.mark.parametrize("enabled", ["true", "1", "false"])
@pytest.mark.parametrize("source", ["pad", "deck", "console", "external", "nav", "brain", "text"])
def test_opt_in_routes_sources_without_changing_legacy_default(enabled, source):
    context = LaunchContext()
    context.launch_configurations["command_mux"] = enabled
    expected = f"/wojtek/cmd_vel/{source}" if enabled != "false" else "/cmd_vel"
    assert command_topic(source).perform(context) == expected


def test_undeclared_mux_option_preserves_the_direct_path():
    assert command_topic("external").perform(LaunchContext()) == "/cmd_vel"


def test_unknown_source_is_refused():
    with pytest.raises(ValueError, match="unknown command source"):
        command_topic("unregistered")


@pytest.mark.parametrize("enabled", [False, True])
def test_disarm_cancels_navigation_only_in_opt_in_mode(enabled):
    published = []
    io = SimpleNamespace(
        _armed=True,
        get_parameter=lambda key: SimpleNamespace(value=enabled),
        _pub_nav_cancel=SimpleNamespace(publish=published.append),
    )
    response = RealIoNode._srv_arm(io, SimpleNamespace(data=False), SimpleNamespace())
    assert response.success and not io._armed
    assert len(published) == int(enabled)
