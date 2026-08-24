"""Tests for the launch file's OpaqueFunction.

`ros2 launch --show-args` only evaluates the argument declarations, so the
part that actually composes the pipeline -- which nodes get created, the
parameter files they load, and the launch-argument overrides layered on top
-- would otherwise stay untested until someone plugs a camera in.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml
from launch import LaunchContext
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
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
        "camera_params_file": str(CONFIG / "d435.yaml"),
        "extrinsics_file": str(CONFIG / "extrinsics.yaml"),
        "extrinsics": "true",
        "accumulate": "true",
        "depth_profile": "",
        "color_profile": "",
        "enable_color": "true",
        "cpus": "",
        # launch's own global prefix mechanism; empty unless someone sets it.
        "launch-prefix": "",
        "camera_name": "camera",
        "camera_namespace": "camera",
    }
    defaults.update(overrides)
    ctx.launch_configurations.update(defaults)
    return ctx


def _prefix(node):
    """The launch prefix (taskset ...) a node will be started under."""
    executable = node._ExecuteLocal__process_description
    return executable._Executable__prefix


def _params(node, ctx):
    """A node's parameters, substitutions resolved."""
    return evaluate_parameters(ctx, node._Node__parameters)


def test_default_setup_is_driver_accumulate_extrinsics(launch_mod):
    actions = launch_mod._setup(_context())
    assert [type(a) for a in actions] == [Node, Node, Node]
    assert len(actions) == 3


def test_driver_is_a_plain_node_not_a_composable_one(launch_mod):
    """The RPi hosts no component container, and the only consumer of the
    depth image is rclpy, which cannot be composed anyway."""
    driver = launch_mod._setup(_context())[0]
    assert driver._Node__package == "realsense2_camera"
    # An executable, not a plugin: composition would need a container to
    # exist somewhere, and none does.
    assert "realsense2_camera_node" in str(driver._Node__node_executable)


def test_optional_pieces_can_be_turned_off(launch_mod):
    actions = launch_mod._setup(
        _context(accumulate="false", extrinsics="false"))
    assert len(actions) == 1
    assert actions[0]._Node__package == "realsense2_camera"


def test_accumulate_gets_the_depth_topics(launch_mod):
    ctx = _context()
    acc = launch_mod._setup(ctx)[1]
    assert "cloud_accumulate" in str(acc._Node__node_executable)
    remaps = {
        perform_substitutions(ctx, src): perform_substitutions(ctx, dst)
        for src, dst in acc._Node__remappings
    }
    assert remaps["depth/image"] == "/camera/camera/depth/image_rect_raw"
    assert remaps["depth/camera_info"] == "/camera/camera/depth/camera_info"


def test_driver_loads_the_camera_parameter_file(launch_mod):
    ctx = _context()
    params = _params(launch_mod._setup(ctx)[0], ctx)
    # File first, overrides after it -- launch_ros passes each entry as its
    # own --params-file in order, so the last one wins.
    assert str(params[0]).endswith("d435.yaml")
    assert isinstance(params[1], dict)


def test_depth_profile_argument_overrides_the_file(launch_mod):
    ctx = _context(depth_profile="848x480x30")
    overrides = _params(launch_mod._setup(ctx)[0], ctx)[1]
    assert overrides["depth_module.depth_profile"] == "848x480x30"


def test_empty_profile_argument_leaves_the_file_alone(launch_mod):
    ctx = _context()
    overrides = _params(launch_mod._setup(ctx)[0], ctx)[1]
    assert "depth_module.depth_profile" not in overrides
    assert "rgb_camera.color_profile" not in overrides


def test_enable_color_argument_overrides_the_file(launch_mod):
    on_ctx, off_ctx = _context(), _context(enable_color="false")
    assert _params(launch_mod._setup(on_ctx)[0], on_ctx)[1]["enable_color"] is True
    assert _params(launch_mod._setup(off_ctx)[0], off_ctx)[1]["enable_color"] is False


def _ros_params(path):
    """The parameters out of a ROS parameter file, wildcard node key."""
    loaded = yaml.safe_load(path.read_text())
    assert list(loaded) == ["/**"], f"{path.name} is not a ROS parameter file"
    return loaded["/**"]["ros__parameters"]


def test_config_keeps_the_measured_depth_settings():
    """Guards the bench result: the preset and the temporal delta are the two
    settings that carried a 3x and a 4x noise improvement respectively."""
    params = _ros_params(CONFIG / "d435.yaml")
    assert params["depth_module"]["visual_preset"] == 3          # HIGH_ACCURACY
    assert params["temporal_filter"]["filter_smooth_delta"] == 100


def test_extrinsics_publisher_gets_the_configured_frames(launch_mod):
    actions = launch_mod._setup(_context())
    tf_node = actions[-1]
    args = [str(a) for a in tf_node._Node__arguments]
    assert "--frame-id" in args and "base_link" in args
    assert "--child-frame-id" in args and "camera_link" in args


def test_stream_rates_are_ones_the_sensor_offers():
    """The D435 exposes a fixed set of frame rates and librealsense rejects
    the whole request if a profile names one it does not have -- the pipeline
    then fails to start. 10 fps, which this config used to ask for, is not on
    either list."""
    params = _ros_params(CONFIG / "d435.yaml")
    depth_fps = int(params["depth_module"]["depth_profile"].rsplit("x", 1)[1])
    color_fps = int(params["rgb_camera"]["color_profile"].rsplit("x", 1)[1])
    assert depth_fps in (6, 15, 30, 60, 90)
    assert color_fps in (6, 15, 30, 60)
    # Depth must keep up with the planner's 10 Hz replan; colour serves the
    # VLM at ~0.3-0.5 Hz and should stay as slow as the sensor allows.
    assert depth_fps >= 10
    assert color_fps <= 15


def test_cpus_argument_pins_the_camera_driver(launch_mod):
    """On the robot the bringup starts everything under `taskset -c 2,3` (the
    isolcpus RT cores) and children inherit that mask. The driver must be
    re-affinitized off those cores or it competes with the 400 Hz control
    loop; measured on the Pi 4 it wants ~0.7 of a core with colour+RGBD on."""
    ctx = _context(cpus="0,1")
    driver, acc, _tf = launch_mod._setup(ctx)
    for node in (driver, acc):
        assert perform_substitutions(ctx, _prefix(node)) == "taskset -c 0,1"


def test_no_cpus_argument_inherits_the_affinity(launch_mod):
    """No cpus argument leaves launch's own (empty) prefix in place, so the
    node inherits whatever mask its parent was started with."""
    ctx = _context()
    driver = launch_mod._setup(ctx)[0]
    assert perform_substitutions(ctx, _prefix(driver)) == ""


def test_rgbd_has_its_two_prerequisites():
    """enable_rgbd needs align_depth AND enable_sync: without either the
    driver logs nothing and ~/rgbd simply stays empty. Guarding the trio
    together because turning one off is a silent break, not an error."""
    params = _ros_params(CONFIG / "d435.yaml")
    if params.get("enable_rgbd"):
        assert params["align_depth"]["enable"] is True
        assert params["enable_sync"] is True
        # RGBD is aligned INTO the colour frame, so colour is what sets its
        # resolution and rate -- they cannot be chosen independently.
        assert params["rgb_camera"]["color_profile"].rsplit("x", 1)[1] == \
            params["depth_module"]["depth_profile"].rsplit("x", 1)[1]


def test_accumulate_does_not_feed_off_the_drivers_cloud():
    """The driver's XYZRGB cloud is on for viewing, but the accumulator must
    keep reading the depth IMAGE: consuming a 100k-point cloud that was
    serialised first is the expensive way round, and that is the whole reason
    cloud_accumulate subscribes to an Image."""
    from wojtek_perception_bringup.cloud_accumulate_node import CloudAccumulateNode
    import inspect
    src = inspect.getsource(CloudAccumulateNode)
    assert "PointCloud2, \"depth" not in src
    assert 'Image, "depth/image"' in src
