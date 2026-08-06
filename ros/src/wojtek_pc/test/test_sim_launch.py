"""The simulation must be the robot bringup with the plant swapped.

The point of building the sim on ros2_control is that a run on the desk
exercises the same nodes, the same parameters and the same arming path as a
run on the robot. Nothing enforces that by itself: the two launches used to
be separate files, and their policy_node parameters had quietly drifted apart
(soft_start_s 0.5 vs 2.0, clamp_knee off vs on, the IMU mount rotation
missing in sim) -- so the simulation was validating a configuration the robot
never runs. These tests are the fence around that.

They call launch_common._launch_setup directly with a prepared context: the
OpaqueFunction body is where the nodes are actually composed, and going
through the full launch machinery would need a policy download.
"""

import importlib.util
from pathlib import Path

import pytest
from launch import LaunchContext
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters

from wojtek_bringup import launch_common

PKG_DIR = Path(__file__).resolve().parents[1]


class _FakePolicy:
    """What load_policy returns, without the Hugging Face round trip."""

    run_name = "test_policy"
    source = "dir:/tmp/test_policy"
    directory = "/tmp/test_policy"
    pd = {"kp": 40.0, "kd": 1.6, "max_torque": 9.0}


@pytest.fixture(autouse=True)
def _no_policy_download(monkeypatch):
    monkeypatch.setattr(
        launch_common, "load_policy", lambda *a, **k: _FakePolicy(),
    )


def _context(hardware, **overrides):
    """A context carrying the launch arguments _launch_setup reads."""
    ctx = LaunchContext()
    defaults = {
        "policy": "org/policy",
        "kp": "", "kd": "", "max_torque": "",
        "use_imu": "true",
        "dry_run": "false",
        "boot_pose": "home",
        "bag": "false", "bag_dir": "/tmp/bags", "bag_cpus": "",
        "rviz": "false", "rviz_config": "/tmp/x.rviz",
        "launch-prefix": "",
    }
    if hardware == "real":
        defaults.update({
            "bus": "spi", "can_baud": "8",
            "imu_bus": "/dev/i2c-1",
            "imu_addr_ag": "0x6A", "imu_addr_mag": "0x1C",
        })
    else:
        defaults.update({"hw": "mock", "model_xml": ""})
    defaults.update(overrides)
    ctx.launch_configurations.update(defaults)
    return ctx


def _nodes(hardware, **overrides):
    ctx = _context(hardware, **overrides)
    actions = launch_common._launch_setup(
        ctx, with_rviz=False, hardware=hardware,
    )
    return ctx, [a for a in actions if isinstance(a, Node)]


def _by_executable(nodes, executable):
    found = [n for n in nodes if executable in str(n._Node__node_executable)]
    assert len(found) == 1, f"expected exactly one {executable}, got {len(found)}"
    return found[0]


def _params(node, ctx):
    return evaluate_parameters(ctx, node._Node__parameters)


@pytest.mark.parametrize("executable", ["policy_node", "real_io_node"])
def test_control_parameters_are_identical_in_sim_and_real(executable):
    """The parameters that decide how the robot behaves must not depend on
    which plant is underneath. A deliberate exception belongs in this list,
    with the reason -- an empty list is the goal."""
    allowed_to_differ = set()

    ctx_real, real = _nodes("real")
    ctx_sim, sim = _nodes("sim")
    p_real = _params(_by_executable(real, executable), ctx_real)
    p_sim = _params(_by_executable(sim, executable), ctx_sim)

    assert len(p_real) == len(p_sim) == 1
    differing = {
        k for k in set(p_real[0]) | set(p_sim[0])
        if p_real[0].get(k) != p_sim[0].get(k)
    }
    assert differing <= allowed_to_differ, (
        f"{executable} parameters drifted between sim and real: {differing}"
    )


def test_sim_runs_the_robots_controller_configuration():
    """Same controllers, same rates, same file -- a sim-only copy of the yaml
    is how the joint order (which must match policy_meta.json) gets to differ
    from the robot's without anyone noticing."""
    ctx_real, real = _nodes("real")
    ctx_sim, sim = _nodes("sim")

    cm_real = _by_executable(real, "ros2_control_node")
    cm_sim = _by_executable(sim, "ros2_control_node")
    yaml_real = _params(cm_real, ctx_real)[1]
    yaml_sim = _params(cm_sim, ctx_sim)[1]
    assert yaml_real == yaml_sim
    assert "real_controllers.yaml" in str(yaml_sim)

    spawners_real = [
        n._Node__arguments for n in real
        if "spawner" in str(n._Node__node_executable)
    ]
    spawners_sim = [
        n._Node__arguments for n in sim
        if "spawner" in str(n._Node__node_executable)
    ]
    assert spawners_real == spawners_sim


def test_sim_loads_the_simulated_plant():
    ctx, nodes = _nodes("sim", hw="mock")
    description = _params(_by_executable(nodes, "ros2_control_node"), ctx)[0]
    assert "mock_components/GenericSystem" in description["robot_description"]
    assert "md80_hardware_interface" not in description["robot_description"]


def test_real_loads_the_md80_plant():
    ctx, nodes = _nodes("real")
    description = _params(_by_executable(nodes, "ros2_control_node"), ctx)[0]
    assert "md80_hardware_interface/MD80HardwareInterface" in (
        description["robot_description"]
    )


def test_ground_truth_and_static_tf_never_publish_the_same_transform():
    """With physics under it the plant knows the true base pose and publishes
    odom->base_link itself; the placeholder static transform must step aside
    or the two fight in the TF tree."""
    for hw, expected in (("mock", 1), ("mujoco", 0)):
        ctx = _context("sim", hw=hw)
        actions = launch_common._launch_setup(
            ctx, with_rviz=False, hardware="sim",
        )
        static = [
            a for a in actions
            if isinstance(a, Node)
            and "static_transform_publisher" in str(a._Node__node_executable)
            and (a.condition is None or a.condition.evaluate(ctx))
        ]
        assert len(static) == expected, f"hw:={hw}"


def test_sim_launch_keeps_the_arguments_its_callers_pass():
    """sim.sh and `ros2 run wojtek_bringup robot --sim` pass these by name;
    dropping one turns into a launch error at the worst moment."""
    spec = importlib.util.spec_from_file_location(
        "sim_launch", PKG_DIR / "launch" / "sim.launch.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ld = module.generate_launch_description()
    declared = {
        a.name for a in ld.entities if hasattr(a, "name") and hasattr(a, "default_value")
    }
    assert {
        "rviz", "policy", "camera", "camera_depth_hz", "camera_color_hz",
        "boot_pose", "hw", "model_xml",
    } <= declared
