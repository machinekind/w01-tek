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


def test_sim_plant_is_selected_by_hw(sim):
    plugin = sim[0].find("hardware/plugin").text
    assert plugin == "mock_components/GenericSystem"
    mujoco = _ros2_control(
        "wojtek_pc", "urdf/wojtek_sim.urdf.xacro", hw="mujoco",
    )
    assert mujoco[0].find("hardware/plugin").text == (
        "wojtek_mujoco_hardware_interface/MujocoHardwareInterface"
    )
