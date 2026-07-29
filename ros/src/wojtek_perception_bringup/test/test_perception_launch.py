"""Tests for the launch file's OpaqueFunction.

`ros2 launch --show-args` only evaluates the argument declarations, so the
part that actually composes the pipeline -- yaml loading, the fps override,
and the container-ownership branch -- would otherwise stay untested until
someone plugs a camera in.
"""

import importlib.util
from pathlib import Path

import pytest
from launch import LaunchContext
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.utilities import evaluate_parameters

PKG_DIR = Path(__file__).resolve().parents[1]
CONFIG = PKG_DIR / "config"


def _load_launch_module():
    path = PKG_DIR / "launch" / "perception.launch.py"
    spec = importlib.util.spec_from_file_location("perception_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def launch_mod():
    return _load_launch_module()


def _context(**overrides):
    ctx = LaunchContext()
    defaults = {
        "container": "",
        "params_file": str(CONFIG / "d435.yaml"),
        "extrinsics_file": str(CONFIG / "extrinsics.yaml"),
        "extrinsics": "true",
        "reduce": "true",
        "fps": "10",
        "enable_color": "false",
        "camera_name": "camera",
        "camera_namespace": "camera",
    }
    defaults.update(overrides)
    ctx.launch_configurations.update(defaults)
    return ctx


def test_default_setup_owns_a_container(launch_mod):
    actions = launch_mod._setup(_context())
    kinds = [type(a) for a in actions]
    assert ComposableNodeContainer in kinds
    assert LoadComposableNodes not in kinds
    # container + reduction + extrinsics
    assert len(actions) == 3


def test_named_container_loads_into_it_instead(launch_mod):
    actions = launch_mod._setup(_context(container="wojtek_container"))
    kinds = [type(a) for a in actions]
    assert LoadComposableNodes in kinds
    assert ComposableNodeContainer not in kinds


def test_optional_pieces_can_be_turned_off(launch_mod):
    actions = launch_mod._setup(_context(reduce="false", extrinsics="false"))
    assert len(actions) == 1
    assert isinstance(actions[0], ComposableNodeContainer)


def _driver_params(actions, ctx):
    """The realsense driver's parameters, substitutions resolved."""
    # ComposableNodeContainer exposes no public accessor for what it holds.
    container = actions[0]
    driver = container._ComposableNodeContainer__composable_node_descriptions[0]
    return evaluate_parameters(ctx, driver.parameters)[0]


def test_fps_argument_rewrites_the_stream_profiles(launch_mod):
    ctx = _context(fps="30")
    params = _driver_params(launch_mod._setup(ctx), ctx)
    assert params["depth_module.depth_profile"].endswith("x30")
    assert params["rgb_camera.color_profile"].endswith("x30")


def test_enable_color_argument_overrides_the_config_file(launch_mod):
    off_ctx, on_ctx = _context(), _context(enable_color="true")
    assert _driver_params(launch_mod._setup(off_ctx), off_ctx)["enable_color"] is False
    assert _driver_params(launch_mod._setup(on_ctx), on_ctx)["enable_color"] is True


def test_config_keeps_the_measured_depth_settings(launch_mod):
    """Guards the bench result: the preset and the temporal delta are the two
    settings that carried a 3x and a 4x noise improvement respectively."""
    ctx = _context()
    params = _driver_params(launch_mod._setup(ctx), ctx)
    assert params["depth_module.visual_preset"] == 3          # HIGH_ACCURACY
    assert params["temporal_filter.filter_smooth_delta"] == 100
    assert params["pointcloud.enable"] is False               # we reduce from the image


def test_extrinsics_publisher_gets_the_configured_frames(launch_mod):
    actions = launch_mod._setup(_context())
    tf_node = [a for a in actions if isinstance(a, Node)][-1]
    args = [str(a) for a in tf_node._Node__arguments]
    assert "--frame-id" in args and "base_link" in args
    assert "--child-frame-id" in args and "camera_link" in args
