"""Tests for nav2.launch.py's OpaqueFunction and the nav2 parameter file.

The stack is stock nav2; what is ours is the composition (which servers,
managed in what order) and the configuration's invariants: no map anywhere,
every frame odom, the footprint and the observation sources the same as the
standalone costmap's. Those must hold without a camera and an odometry up.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters

PKG_DIR = Path(__file__).resolve().parents[1]
CONFIG = PKG_DIR / "config"

# The robot's half-extents at the home pose, legs included (see
# test_costmap_launch.py): the footprint must cover this box.
ROBOT_HALF_X, ROBOT_HALF_Y = 0.373, 0.230

# The follower must talk faster than the policy's cmd_vel watchdog
# (policy_node: watchdog_timeout_s = 0.2).
POLICY_WATCHDOG_S = 0.2


@pytest.fixture(scope="module")
def launch_mod():
    path = PKG_DIR / "launch" / "nav2.launch.py"
    spec = importlib.util.spec_from_file_location("nav2_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def params():
    with open(CONFIG / "nav2.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def costmap_params():
    with open(CONFIG / "costmap.yaml") as f:
        return yaml.safe_load(f)["/**/costmap"]["ros__parameters"]


def _context(**overrides):
    ctx = LaunchContext()
    defaults = {
        "nav2_params_file": str(CONFIG / "nav2.yaml"),
        "autostart": "true",
        "log_level": "info",
        "cpus": "",
        "launch-prefix": "",
    }
    defaults.update(overrides)
    ctx.launch_configurations.update(defaults)
    return ctx


def _nodes(launch_mod, ctx):
    return [a for a in launch_mod._setup(ctx) if isinstance(a, Node)]


def _params(node, ctx):
    return evaluate_parameters(ctx, node._Node__parameters)


def _prefix(node):
    return node._ExecuteLocal__process_description._Executable__prefix


# -- composition ---------------------------------------------------------------

def _declared_args(path):
    spec = importlib.util.spec_from_file_location(path.stem.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        a.name for a in module.generate_launch_description().entities
        if isinstance(a, DeclareLaunchArgument)
    }


def test_no_argument_name_shared_with_the_costmap_launch():
    """The bringup includes both launches side by side, and included
    launches share the parent's configurations: a name declared by
    costmap.launch.py first is what nav2.launch.py's default would silently
    become (seen for real: params_file -> costmap.yaml, DWB defaults at
    20 Hz). Only the names the bringup passes explicitly may be shared."""
    shared = _declared_args(PKG_DIR / "launch" / "nav2.launch.py") & _declared_args(
        PKG_DIR / "launch" / "costmap.launch.py"
    )
    assert shared == {"cpus"}

def test_every_lifecycle_node_is_launched_and_managed(launch_mod):
    ctx = _context()
    nodes = _nodes(launch_mod, ctx)
    names = [n._Node__node_name for n in nodes]
    for lifecycle in launch_mod.LIFECYCLE_NODES:
        assert lifecycle in names
    manager = next(n for n in nodes if n._Node__node_name == "lifecycle_manager_navigation")
    (p,) = _params(manager, ctx)
    assert list(p["node_names"]) == launch_mod.LIFECYCLE_NODES
    assert p["autostart"] is True


def test_costmap_owners_come_up_before_the_navigator(launch_mod):
    order = launch_mod.LIFECYCLE_NODES
    assert order.index("controller_server") < order.index("bt_navigator")
    assert order.index("planner_server") < order.index("bt_navigator")


def test_autostart_false_leaves_the_nodes_unconfigured(launch_mod):
    ctx = _context(autostart="false")
    manager = next(n for n in _nodes(launch_mod, ctx)
                   if n._Node__node_name == "lifecycle_manager_navigation")
    (p,) = _params(manager, ctx)
    assert p["autostart"] is False


def test_servers_read_the_params_file(launch_mod):
    ctx = _context()
    for n in _nodes(launch_mod, ctx):
        if n._Node__node_name in launch_mod.LIFECYCLE_NODES:
            (p,) = _params(n, ctx)
            assert str(p) == str(CONFIG / "nav2.yaml")


def test_cpu_pin_reaches_every_node(launch_mod):
    ctx = _context(cpus="0,1")
    for n in _nodes(launch_mod, ctx):
        assert perform_substitutions(ctx, _prefix(n)) == "taskset -c 0,1"
    ctx = _context()
    for n in _nodes(launch_mod, ctx):
        # No pin: the default `launch-prefix` configuration, which is empty.
        assert perform_substitutions(ctx, _prefix(n)) == ""


# -- configuration invariants --------------------------------------------------

def _leaf_params(params):
    """Every ros__parameters block, keyed by the top-level node name."""
    out = {}
    for name, block in params.items():
        while "ros__parameters" not in block:
            (block,) = block.values()
        out[name] = block["ros__parameters"]
    return out


def test_no_launch_substitutions_in_the_params_file():
    """`$(find-pkg-share ...)` is resolved by nav2_bringup's RewrittenYaml,
    which this launch does not use; the servers would read it as a path
    (seen for real: bt_navigator "Couldn't open input XML file")."""
    assert "$(" not in (CONFIG / "nav2.yaml").read_text()


def test_no_map_anywhere(params):
    assert "map_server" not in params
    assert "amcl" not in params
    for name, p in _leaf_params(params).items():
        for key in ("global_frame", "local_frame"):
            if key in p:
                assert p[key] == "odom", f"{name}.{key}"
        assert p.get("robot_base_frame", "base_link") == "base_link", name
        for plugin in p.get("plugins", []):
            assert p[plugin]["plugin"] != "nav2_costmap_2d::StaticLayer", name


def test_both_costmaps_roll_in_odom_with_the_robots_footprint(params):
    leaf = _leaf_params(params)
    for name in ("local_costmap", "global_costmap"):
        p = leaf[name]
        assert p["rolling_window"] is True, name
        assert p["global_frame"] == "odom", name
        assert p["track_unknown_space"] is True, name
        poly = yaml.safe_load(p["footprint"])
        xs, ys = [abs(x) for x, _ in poly], [abs(y) for _, y in poly]
        assert min(xs) >= ROBOT_HALF_X and min(ys) >= ROBOT_HALF_Y, name
    assert leaf["global_costmap"]["width"] >= leaf["local_costmap"]["width"]


def test_costmap_layers_match_the_standalone_costmap(params, costmap_params):
    """One perception, argued once (costmap.yaml): the nav2 windows must not
    silently drift from it -- same sources, heights, ranges and inflation."""
    leaf = _leaf_params(params)
    for name in ("local_costmap", "global_costmap"):
        p = leaf[name]
        for layer in p["plugins"]:
            ours, theirs = p[layer], costmap_params[layer]
            for key, value in theirs.items():
                if key == "publish_voxel_map":
                    continue  # the debug grid is the standalone map's alone
                assert ours[key] == value, f"{name}.{layer}.{key}"
    assert leaf["global_costmap"]["plugins"] == costmap_params["plugins"]
    local = leaf["local_costmap"]
    for key in ("width", "height", "resolution", "update_frequency"):
        assert local[key] == costmap_params[key], key


def test_local_costmap_checks_the_exact_footprint(params):
    """No inflation in the local costmap: Nav2 calls the inscribed cost a
    collision along the footprint's edges, which would turn the 0.76 x 0.48
    polygon into a 0.69 m circle and block every turn in a 1.8 m corridor
    (the file argues it). The follower must then not ask for the gradient."""
    leaf = _leaf_params(params)
    local = leaf["local_costmap"]
    assert "inflation_layer" not in local["plugins"]
    assert local["plugins"] == ["voxel_layer"]
    follow = leaf["controller_server"]["FollowPath"]
    assert follow["use_cost_regulated_linear_velocity_scaling"] is False
    assert follow["use_collision_detection"] is True
    # The planner keeps its gradient.
    assert "inflation_layer" in leaf["global_costmap"]["plugins"]


def test_follower_outruns_the_policy_watchdog(params):
    hz = _leaf_params(params)["controller_server"]["controller_frequency"]
    assert 1.0 / hz < POLICY_WATCHDOG_S


def test_goal_is_a_point_not_a_pose(params):
    """The VLM hands over XY; any final heading must count as reached."""
    checker = _leaf_params(params)["controller_server"]["goal_checker"]
    assert checker["yaw_goal_tolerance"] > 3.1416
    assert checker["xy_goal_tolerance"] <= 0.2


def test_follower_never_reverses_or_strafes(params):
    """The camera looks forward only, and the policy was driven with vx/wz."""
    c = _leaf_params(params)["controller_server"]
    assert c["FollowPath"]["allow_reversing"] is False
    assert c["min_y_velocity_threshold"] >= 0.5


def test_planner_may_cross_unknown_space(params):
    p = _leaf_params(params)["planner_server"]["GridBased"]
    assert p["allow_unknown"] is True


def test_observation_sources_are_the_pipelines_cloud(params):
    for name in ("local_costmap", "global_costmap"):
        layer = _leaf_params(params)[name]["voxel_layer"]
        for source in layer["observation_sources"].split():
            assert layer[source]["topic"] == "/wojtek/nav/points", f"{name}.{source}"
