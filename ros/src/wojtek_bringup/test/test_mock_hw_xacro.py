"""mock_hw:=true must swap the drivers, not the robot.

The mock exists to load the robot computer with the real node graph while
the motors are unpowered (ros/hw_tests/perf). It only does that if every
controller still claims what it claims on the robot: the broadcasters bind
by sensor name, the policy and the forward controllers by joint and
interface name. So the generated URDF must carry the same joints, interfaces
and sensor as the real one, with only the plugin changed.
"""

import subprocess
import xml.etree.ElementTree as ET

import pytest
from ament_index_python.packages import get_package_share_directory

XACRO = f"{get_package_share_directory('wojtek_bringup')}/urdf/wojtek_real.urdf.xacro"


def _ros2_control(**args):
    """The <ros2_control> blocks of the real xacro, generated as the launch does."""
    cmd = ["xacro", XACRO] + [f"{k}:={v}" for k, v in args.items()]
    generated = subprocess.run(
        cmd, check=True, capture_output=True, text=True,
    ).stdout
    return ET.fromstring(generated).findall("ros2_control")


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


def _plugins(blocks):
    return {b.find("hardware/plugin").text for b in blocks}


@pytest.mark.parametrize("tau_ff", ["false", "true"])
@pytest.mark.parametrize("use_imu", ["true", "false"])
def test_mock_carries_the_real_interfaces(tau_ff, use_imu):
    real = _ros2_control(tau_ff=tau_ff, use_imu=use_imu)
    mock = _ros2_control(tau_ff=tau_ff, use_imu=use_imu, mock_hw="true")
    assert _joints(mock) == _joints(real)
    assert _sensors(mock) == _sensors(real)


def test_mock_replaces_every_driver_plugin():
    mock = _ros2_control(mock_hw="true")
    assert _plugins(mock) == {"mock_components/GenericSystem"}
    assert "md80_hardware_interface/MD80HardwareInterface" in _plugins(_ros2_control())


def test_mock_is_off_by_default():
    assert "mock_components/GenericSystem" not in _plugins(_ros2_control())
