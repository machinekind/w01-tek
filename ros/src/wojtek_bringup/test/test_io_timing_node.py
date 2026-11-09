"""These tests check I/O timing and launch parameters.

They require rclpy and a built ROS workspace.
"""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

rclpy = pytest.importorskip("rclpy")

from launch import LaunchContext  # noqa: E402
from launch.actions import DeclareLaunchArgument  # noqa: E402
from launch.utilities import perform_substitutions  # noqa: E402
from launch_ros.actions import Node as LaunchNode  # noqa: E402
from launch_ros.utilities import evaluate_parameters  # noqa: E402
from rclpy.duration import Duration  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from wojtek_bringup import launch_common  # noqa: E402
from wojtek_bringup.real_io_node import RealIoNode  # noqa: E402


class _Recorder:
    """This publisher records message order."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def publish(self, msg):
        self.log.append((self.name, msg))


class _Raising:
    def __init__(self):
        self.calls = 0

    def publish(self, msg):
        self.calls += 1
        raise RuntimeError("publisher broke")


class _Logger:
    def __init__(self):
        self.errors, self.warnings = [], []

    def error(self, text, **kwargs):
        self.errors.append(text)

    def warning(self, text, **kwargs):
        self.warnings.append(text)

    warn = warning

    def info(self, text, **kwargs):
        pass


@pytest.fixture
def make_node():
    made = []

    def make(**params):
        args = ["--ros-args"]
        for key, value in params.items():
            args += ["-p", f"{key}:={str(value).lower()}"]
        rclpy.init(args=args)
        node = RealIoNode()
        made.append(node)
        log = []
        node._pub_cmd = _Recorder(log, "cmd")
        node._pub_effort = _Recorder(log, "effort")
        node._pub_abs = _Recorder(log, "abs")
        if node._timing_pub is not None:
            node._timing_pub = _Recorder(log, "timing")
        return node, log

    yield make
    for node in made:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _joints(node, age_ms=1.0):
    msg = JointState()
    msg.header.stamp = (
        node.get_clock().now() - Duration(nanoseconds=int(age_ms * 1e6))
    ).to_msg()
    msg.name = list(node.joint_names)
    msg.position = [0.0] * len(msg.name)  # The boot pose matches home.
    msg.velocity = [0.0] * len(msg.name)
    return msg


def _target(node, position=None, drop=None):
    msg = JointState()
    msg.header.stamp = node.get_clock().now().to_msg()
    names = list(node.joint_names)
    values = list(node.home_urdf) if position is None else list(position)
    if drop is not None:
        i = names.index(drop)
        del names[i], values[i]
    msg.name = names
    msg.position = [float(v) for v in values]
    return msg


def _timings(log):
    return [msg for name, msg in log if name == "timing"]


def _arm(node):
    node._on_joints(_joints(node))
    req = SimpleNamespace(data=True)
    res = node._srv_arm(req, SimpleNamespace(success=None, message=""))
    assert res.success, res.message


def test_nothing_is_published_when_publish_timing_is_off(make_node):
    node, log = make_node(publish_timing=False)
    assert node._timing_pub is None
    assert "/wojtek/io_timing" not in [p.topic_name for p in node.publishers]
    node._on_joints(_joints(node))
    node._on_targets(_target(node))
    assert _timings(log) == []


def test_the_publisher_exists_when_publish_timing_is_on(make_node):
    # The original publisher stays registered on the node.
    node, _ = make_node(publish_timing=True)
    assert "/wojtek/io_timing" in [p.topic_name for p in node.publishers]


def test_disarmed_target_is_timed_without_a_command(make_node):
    node, log = make_node(publish_timing=True)
    node._on_joints(_joints(node))
    target = _target(node)
    node._on_targets(target)
    [timing] = _timings(log)
    assert timing.armed is False
    assert timing.command_sent is False
    assert timing.target_stamp == target.header.stamp
    assert timing.target_age_ms >= 0.0
    assert [name for name, _ in log if name == "cmd"] == []


def test_armed_dry_run_sends_nothing_and_says_so(make_node):
    node, log = make_node(publish_timing=True, dry_run=True)
    _arm(node)
    node._on_targets(_target(node))
    [timing] = _timings(log)
    assert timing.armed is True
    assert timing.command_sent is False
    assert [name for name, _ in log if name == "cmd"] == []


def test_armed_command_is_published_before_its_timing(make_node):
    node, log = make_node(publish_timing=True)
    _arm(node)
    log.clear()
    node._on_targets(_target(node))
    assert [name for name, _ in log] == ["cmd", "timing"]
    timing = log[1][1]
    assert timing.armed is True
    assert timing.command_sent is True
    assert timing.target_age_ms >= 0.0


def test_non_finite_target_is_dropped_and_not_counted_as_sent(make_node):
    node, log = make_node(publish_timing=True)
    _arm(node)
    log.clear()
    position = list(node.home_urdf)
    position[0] = math.nan
    node._on_targets(_target(node, position=position))
    assert [name for name, _ in log] == ["timing"]
    assert log[0][1].armed is True
    assert log[0][1].command_sent is False


def test_target_missing_a_joint_is_rejected_but_still_timed(make_node):
    node, log = make_node(publish_timing=True)
    _arm(node)
    log.clear()
    logger = _Logger()
    node.get_logger = lambda: logger
    target = _target(node, drop=node.joint_names[3])
    node._on_targets(target)
    assert any("missing joint" in w for w in logger.warnings)
    assert [name for name, _ in log] == ["timing"]
    timing = log[0][1]
    assert timing.armed is True
    assert timing.command_sent is False
    assert timing.target_stamp == target.header.stamp
    assert math.isfinite(timing.target_age_ms)
    assert math.isfinite(timing.state_relay_ms)


def test_state_relay_is_nan_until_the_first_joint_state(make_node):
    node, log = make_node(publish_timing=True)
    node._on_targets(_target(node))
    node._on_joints(_joints(node, age_ms=5.0))
    node._on_targets(_target(node))
    before, after = _timings(log)
    assert math.isnan(before.state_relay_ms)
    assert math.isfinite(after.state_relay_ms)
    assert after.state_relay_ms >= 5.0


def test_a_failing_timing_publisher_is_logged_once_and_commands_continue(
    make_node,
):
    node, log = make_node(publish_timing=True)
    _arm(node)
    log.clear()
    broken = _Raising()
    node._timing_pub = broken
    logger = _Logger()
    node.get_logger = lambda: logger
    for _ in range(3):
        node._on_joints(_joints(node))
        node._on_targets(_target(node))
    assert broken.calls == 3
    assert len(logger.errors) == 1
    assert "io timing" in logger.errors[0]
    assert [name for name, _ in log].count("cmd") == 3


def _launch_context(**overrides):
    ctx = LaunchContext()
    description = launch_common.common_launch_description(
        with_rviz=False, bag_default="false", hardware="real",
    )
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument) and entity.default_value:
            ctx.launch_configurations[entity.name] = perform_substitutions(
                ctx, entity.default_value
            )
    ctx.launch_configurations.update(overrides)
    return ctx


def _fake_policy(*args, **kwargs):
    return SimpleNamespace(
        pd={"kp": 20.0, "kd": 0.5, "max_torque": 6.0},
        meta={},
        run_name="fake",
        source="test",
        directory=Path("/nonexistent/fake_policy"),
    )


@pytest.mark.parametrize("telemetry, expected", [("true", True), ("false", False)])
def test_launch_binds_real_io_publish_timing_to_telemetry(
    monkeypatch, telemetry, expected,
):
    monkeypatch.setattr(launch_common, "load_policy", _fake_policy)
    ctx = _launch_context(telemetry=telemetry, policy="org/fake")
    actions = launch_common._launch_setup(ctx, with_rviz=False, hardware="real")
    [real_io] = [
        a for a in actions
        if isinstance(a, LaunchNode) and a.node_executable == "real_io_node"
    ]
    params = evaluate_parameters(ctx, real_io._Node__parameters)
    merged = {}
    for entry in params:
        merged.update(entry)
    assert merged["publish_timing"] is expected
