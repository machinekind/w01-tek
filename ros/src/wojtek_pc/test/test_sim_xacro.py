"""The simulated component must expose exactly the robot's interfaces.

The broadcasters bind by sensor name and interface name (real_controllers.yaml
says sensor_name: imu), and the policy binds joints by name. If the two xacros
drift -- a joint renamed on one side, an interface added on the other -- the
simulation still comes up, just with a controller quietly refusing to claim
something. Comparing the generated URDFs is the cheapest way to notice.
"""

import subprocess
import xml.etree.ElementTree as ET

import pytest
from ament_index_python.packages import get_package_share_directory


def _ros2_control(package, relative, **args):
    """The <ros2_control> blocks of a xacro, generated as the launch does."""
    path = f"{get_package_share_directory(package)}/{relative}"
    cmd = ["xacro", path] + [f"{k}:={v}" for k, v in args.items()]
    generated = subprocess.run(
        cmd, check=True, capture_output=True, text=True,
    ).stdout
    return ET.fromstring(generated).findall("ros2_control")


@pytest.fixture(scope="module")
def real():
    return _ros2_control("wojtek_bringup", "urdf/wojtek_real.urdf.xacro")


@pytest.fixture(scope="module")
def sim():
    return _ros2_control("wojtek_pc", "urdf/wojtek_sim.urdf.xacro", hw="mock")


def _joints(blocks):
    return {
        j.get("name"): {
            ("command", c.get("name")) for c in j.findall("command_interface")
        } | {
            ("state", s.get("name")) for s in j.findall("state_interface")
        }
        for block in blocks for j in block.findall("joint")
    }


def _sensors(blocks):
    return {
        s.get("name"): {i.get("name") for i in s.findall("state_interface")}
        for block in blocks for s in block.findall("sensor")
    }


def test_same_joints_with_the_same_interfaces(real, sim):
    assert _joints(sim) == _joints(real)


def test_same_sensor_name_and_interfaces(real, sim):
    """sensor_name in real_controllers.yaml is shared by both bringups, so the
    name is a contract, not a label."""
    assert _sensors(sim) == _sensors(real)


def test_sim_declares_one_component_carrying_joints_and_sensor(sim):
    """The real robot has two drivers on two buses; the simulated plant owns a
    single physics state, so splitting it in two would mean sharing it between
    components."""
    assert len(sim) == 1
    assert sim[0].get("type") == "system"


def test_use_imu_false_drops_the_sensor_on_both_sides():
    sim = _ros2_control(
        "wojtek_pc", "urdf/wojtek_sim.urdf.xacro", hw="mock", use_imu="false",
    )
    real = _ros2_control(
        "wojtek_bringup", "urdf/wojtek_real.urdf.xacro", use_imu="false",
    )
    assert _sensors(sim) == _sensors(real) == {}
    assert _joints(sim) == _joints(real)


def test_tau_ff_adds_the_effort_command_on_both_sides():
    """A policy with the torque head gets an effort command per joint from
    the launch's tau_ff switch, and the simulated plant must take it: a
    controller spawned for an interface the plant does not export fails
    to activate, and the spawner takes every controller after it down."""
    sim = _ros2_control(
        "wojtek_pc", "urdf/wojtek_sim.urdf.xacro", hw="mock",
        tau_ff="true", tau_ff_scale="3.0",
    )
    real = _ros2_control(
        "wojtek_bringup", "urdf/wojtek_real.urdf.xacro", tau_ff="true",
    )
    assert _joints(sim) == _joints(real)
    assert all(("command", "effort") in ifs for ifs in _joints(sim).values())


def test_sim_servo_cap_leaves_the_torque_head_out():
    """The launch hands both xacros the drive cap, servo plus head. The
    plant clamps the two separately like the training sim, so the servo's
    param is the cap with the head removed and the head's scale rides next
    to it."""
    mujoco = _ros2_control(
        "wojtek_pc", "urdf/wojtek_sim.urdf.xacro", hw="mujoco",
        tau_ff="true", tau_ff_scale="3.0", max_torque="9.0",
    )
    for joint in mujoco[0].findall("joint"):
        params = {p.get("name"): p.text for p in joint.findall("param")}
        assert float(params["max_torque"]) == 6.0
        assert float(params["tau_ff_scale"]) == 3.0
    plain = _ros2_control(
        "wojtek_pc", "urdf/wojtek_sim.urdf.xacro", hw="mujoco", max_torque="9.0",
    )
    params = {p.get("name"): p.text for p in plain[0].find("joint").findall("param")}
    assert float(params["max_torque"]) == 9.0
    assert float(params["tau_ff_scale"]) == 0.0


def test_sim_plant_is_selected_by_hw(sim):
    plugin = sim[0].find("hardware/plugin").text
    assert plugin == "mock_components/GenericSystem"
    mujoco = _ros2_control(
        "wojtek_pc", "urdf/wojtek_sim.urdf.xacro", hw="mujoco",
    )
    assert mujoco[0].find("hardware/plugin").text == (
        "wojtek_mujoco_hardware_interface/MujocoHardwareInterface"
    )
