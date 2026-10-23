"""Tests for the launch file's OpaqueFunction and the costmap parameter file.

`ros2 launch --show-args` only evaluates the argument declarations; the part
that composes the perception -- which nodes, wired to which topics, in what
order, and the invariants of the costmap configuration -- would otherwise
stay untested until a camera and an odometry are up.
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

# The robot's half-extents at the home pose, legs included, measured on
# the MuJoCo model (2026-09-25): the footprint must cover this box.
ROBOT_HALF_X, ROBOT_HALF_Y = 0.373, 0.230


@pytest.fixture(scope="module")
def launch_mod():
    path = PKG_DIR / "launch" / "costmap.launch.py"
    spec = importlib.util.spec_from_file_location("costmap_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context(**overrides):
    ctx = LaunchContext()
    defaults = {
        "params_file": str(CONFIG / "costmap.yaml"),
        "depth_topic": "/camera/camera/depth/image_rect_raw",
        "depth_info_topic": "/camera/camera/depth/camera_info",
        "points_topic": "/wojtek/nav/points",
        "decimation": "4",
        "goto": "true",
        "goal_timeout": "3.0",
        "cpus": "",
        "launch-prefix": "",
    }
    defaults.update(overrides)
    ctx.launch_configurations.update(defaults)
    return ctx


def _params(node, ctx):
    return evaluate_parameters(ctx, node._Node__parameters)


def _remaps(node, ctx):
    return {
        perform_substitutions(ctx, src): perform_substitutions(ctx, dst)
        for src, dst in node._Node__remappings
    }


def _prefix(node):
    return node._ExecuteLocal__process_description._Executable__prefix


def _costmap_params():
    with open(CONFIG / "costmap.yaml") as fh:
        return yaml.safe_load(fh)["/**/costmap"]["ros__parameters"]


def _executables(nodes):
    return [str(n._Node__node_executable).rsplit("/", 1)[-1] for n in nodes]


def test_setup_is_decimate_deproject_costmap_then_goto(launch_mod):
    nodes = launch_mod._setup(_context())
    assert all(isinstance(n, Node) for n in nodes)
    assert _executables(nodes) == [
        "crop_decimate_node", "point_cloud_xyz_node", "nav2_costmap_2d", "goto_node",
        "pixel_goal_node",
    ]


def test_goto_can_be_left_out(launch_mod):
    nodes = launch_mod._setup(_context(goto="false"))
    assert "goto_node" not in _executables(nodes)
    # The pixel resolver only makes sense with goto to drive its setpoints.
    assert "pixel_goal_node" not in _executables(nodes)


def _by_executable(nodes, name):
    return next(n for n in nodes if _executables([n]) == [name])


def test_goto_gets_the_dead_man(launch_mod):
    ctx = _context(goal_timeout="2.5")
    goto = _by_executable(launch_mod._setup(ctx), "goto_node")
    assert _params(goto, ctx)[0]["goal_timeout"] == pytest.approx(2.5)


def test_pixel_goal_reads_the_costmaps_depth_stream(launch_mod):
    ctx = _context(depth_topic="/d/image", depth_info_topic="/d/info")
    node = _by_executable(launch_mod._setup(ctx), "pixel_goal_node")
    params = _params(node, ctx)[0]
    assert params["depth_topic"] == "/d/image"
    assert params["depth_info_topic"] == "/d/info"


def test_decimation_one_skips_the_decimator(launch_mod):
    ctx = _context(decimation="1")
    nodes = launch_mod._setup(ctx)
    assert _executables(nodes) == [
        "point_cloud_xyz_node", "nav2_costmap_2d", "goto_node", "pixel_goal_node",
    ]
    # The deprojector then reads the camera directly.
    remaps = _remaps(nodes[0], ctx)
    assert remaps["image_rect"] == "/camera/camera/depth/image_rect_raw"
    assert remaps["camera_info"] == "/camera/camera/depth/camera_info"


def test_bad_decimation_is_refused(launch_mod):
    with pytest.raises(ValueError):
        launch_mod._setup(_context(decimation="0"))


def test_decimated_pair_shares_a_directory(launch_mod):
    """image_transport finds a camera's info as <dir of image>/camera_info;
    a pair published anywhere else is never joined and the deprojector
    waits forever (seen: '/wojtek/nav/depth' + '/wojtek/nav/camera_info')."""
    ctx = _context()
    decim, cloud = launch_mod._setup(ctx)[:2]
    out = _remaps(decim, ctx)
    image, info = out["out/image_raw"], out["out/camera_info"]
    assert image.rsplit("/", 1)[0] == info.rsplit("/", 1)[0]
    assert info.endswith("/camera_info")
    read = _remaps(cloud, ctx)
    assert (read["image_rect"], read["camera_info"]) == (image, info)


def test_decimator_takes_every_nth_pixel_without_interpolating(launch_mod):
    ctx = _context(decimation="4")
    params = _params(launch_mod._setup(ctx)[0], ctx)[0]
    assert params["decimation_x"] == params["decimation_y"] == 4
    assert params["interpolation"] == 0, "averaging depths across an edge invents mid-air points"


def test_cloud_lands_on_the_topic_the_costmap_reads(launch_mod):
    ctx = _context()
    _, cloud, costmap = launch_mod._setup(ctx)[:3]
    points = _remaps(cloud, ctx)["points"]
    sources = _costmap_params()["voxel_layer"]
    for name in sources["observation_sources"].split():
        assert sources[name]["topic"] == points, name
    assert str(_params(costmap, ctx)[0]).endswith("costmap.yaml")


def test_cpus_argument_pins_every_node(launch_mod):
    ctx = _context(cpus="0,1")
    for node in launch_mod._setup(ctx):
        assert perform_substitutions(ctx, _prefix(node)) == "taskset -c 0,1"
    ctx = _context()
    for node in launch_mod._setup(ctx):
        assert perform_substitutions(ctx, _prefix(node)) == ""


# -- the parameter file's invariants ------------------------------------

def test_costmap_is_a_rolling_window_in_odom_that_autostarts():
    p = _costmap_params()
    assert p["rolling_window"] is True
    assert p["global_frame"] == "odom" and p["robot_base_frame"] == "base_link"
    assert p["autostart_node"] is True, "no lifecycle manager for one node"
    assert p["track_unknown_space"] is True, "'never seen' is not 'free'"


def test_window_covers_the_cameras_reach_and_its_cell():
    p = _costmap_params()
    assert p["width"] >= 6 and p["height"] >= 6, "3 m of camera clip each way"
    assert p["resolution"] == pytest.approx(0.05)


def test_footprint_covers_the_measured_robot():
    p = _costmap_params()
    xs = [x for x, _ in yaml.safe_load(p["footprint"])]
    ys = [y for _, y in yaml.safe_load(p["footprint"])]
    assert min(xs) <= -ROBOT_HALF_X and max(xs) >= ROBOT_HALF_X
    assert min(ys) <= -ROBOT_HALF_Y and max(ys) >= ROBOT_HALF_Y


def test_marking_and_clearing_are_split_by_floor_height():
    """One source marks from above the floor, the other clears with rays
    down to it. A single source cannot do both: either the floor becomes
    a wall or nothing ever clears."""
    voxel = _costmap_params()["voxel_layer"]
    sources = {n: voxel[n] for n in voxel["observation_sources"].split()}
    marking = [s for s in sources.values() if s["marking"]]
    clearing = [s for s in sources.values() if s["clearing"]]
    assert len(marking) == 1 and len(clearing) == 1
    assert not marking[0]["clearing"] and not clearing[0]["marking"]
    floor = -0.14  # the lowest the floor sits in base_link (highest stand)
    assert marking[0]["min_obstacle_height"] > floor + 0.05, "floor must not mark"
    assert clearing[0]["min_obstacle_height"] <= floor, "floor rays must clear"
    assert voxel["origin_z"] <= clearing[0]["min_obstacle_height"]


def test_ranges_match_the_camera():
    voxel = _costmap_params()["voxel_layer"]
    mark, clear = voxel["depth_mark"], voxel["depth_clear"]
    assert mark["obstacle_max_range"] <= 3.0, "the camera bringup clips depth at 3 m"
    assert mark["obstacle_min_range"] >= 0.3 and clear["raytrace_min_range"] >= 0.3
    assert clear["raytrace_max_range"] >= mark["obstacle_max_range"]


def test_voxel_grid_spans_floor_to_above_the_body():
    voxel = _costmap_params()["voxel_layer"]
    top = voxel["origin_z"] + voxel["z_voxels"] * voxel["z_resolution"]
    assert voxel["origin_z"] <= -0.2
    assert top >= voxel["depth_mark"]["max_obstacle_height"]


def test_inflation_reaches_the_footprints_corner():
    p = _costmap_params()
    corner = (ROBOT_HALF_X ** 2 + ROBOT_HALF_Y ** 2) ** 0.5
    assert p["inflation_layer"]["inflation_radius"] >= corner
