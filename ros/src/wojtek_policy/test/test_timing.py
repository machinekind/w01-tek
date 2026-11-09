"""These tests cover sensor ages reported on /wojtek/policy_timing.

Run pytest ros/src/wojtek_policy/test/test_timing.py from the repository root.
The arithmetic tests run without ROS.
The callback tests require the ROS container.
The callback tests skip when rclpy is unavailable.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

import pytest  # noqa: E402

from wojtek_policy.timing import (  # noqa: E402
    InputSamples,
    age_ms,
    observation_ages,
    stamp_to_ns,
)

S = 1_000_000_000


def stamp(sec, nanosec):
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def test_stamp_to_ns():
    assert stamp_to_ns(stamp(0, 0)) == 0
    assert stamp_to_ns(stamp(12, 345)) == 12 * S + 345
    # Timestamps retain nanosecond precision at current epoch values.
    assert stamp_to_ns(stamp(1_762_000_000, 999_999_999)) == (
        1_762_000_000 * S + 999_999_999
    )


def test_age_ms():
    now = 1_762_000_000 * S
    assert age_ms(now, now) == 0.0
    assert age_ms(now, now - 20_000_000) == 20.0
    assert math.isclose(age_ms(now, now - 1_500), 0.0015)
    # Clock skew can produce a negative age.
    assert age_ms(now, now + 3_000_000) == -3.0


def test_age_unknown_is_nan():
    now = 1_762_000_000 * S
    assert math.isnan(age_ms(now, None))
    assert math.isnan(age_ms(now, 0))


def test_observation_ages():
    tick = 100 * S
    joints_age, joints_wait, imu_age = observation_ages(
        tick,
        joints_stamp_ns=tick - 12_000_000,
        joints_arrival_ns=tick - 9_000_000,
        imu_stamp_ns=tick - 4_000_000,
        uses_imu=True,
    )
    assert (joints_age, joints_wait, imu_age) == (12.0, 9.0, 4.0)
    # Joint age minus joint wait measures delivery time from the header stamp.
    assert joints_age - joints_wait == 3.0


def test_imu_age_nan_when_policy_is_imu_blind():
    tick = 100 * S
    _, _, imu_age = observation_ages(
        tick, tick - 1, tick - 1, imu_stamp_ns=tick - 4_000_000, uses_imu=False
    )
    assert math.isnan(imu_age)


def test_nan_before_first_sample():
    ages = observation_ages(100 * S, None, None, None, uses_imu=True)
    assert all(math.isnan(a) for a in ages)


def test_input_samples_ages():
    tick = 100 * S
    samples = InputSamples()
    assert all(math.isnan(a) for a in samples.ages(tick, uses_imu=True))
    samples.note_joints(arrival_ns=tick - 9_000_000,
                        header_stamp_ns=tick - 12_000_000)
    samples.note_imu(header_stamp_ns=tick - 4_000_000)
    assert samples.ages(tick, uses_imu=True) == (12.0, 9.0, 4.0)


# The callback tests run without ROS initialization or policy files.

JOINTS = [f"j{i}" for i in range(12)]
WORK_NS = 1_000_000  # Each field read advances the clock by this amount.


class WorkClock:
    """This clock advances only while the callback processes a message."""

    def __init__(self, start_ns):
        self.t = start_ns
        self.reads = []

    def now(self):
        self.reads.append(self.t)
        return SimpleNamespace(nanoseconds=self.t)


class SlowJointState:
    """Each read of name, position or velocity advances the test clock."""

    def __init__(self, clock, names, stamp_ns):
        self._clock = clock
        self.header = SimpleNamespace(stamp=stamp(stamp_ns // S, stamp_ns % S))
        self._name = list(names)
        self._position = [0.1] * len(names)
        self._velocity = [0.0] * len(names)

    def _work(self, value):
        self._clock.t += WORK_NS
        return value

    name = property(lambda self: self._work(self._name))
    position = property(lambda self: self._work(self._position))
    velocity = property(lambda self: self._work(self._velocity))


def node_stub(clock):
    return SimpleNamespace(
        joint_names=JOINTS,
        get_clock=lambda: clock,
        _samples=InputSamples(),
        _joints_arrival=None,
        _q_urdf=None,
        _dq_urdf=None,
    )


def policy_node():
    pytest.importorskip("rclpy")
    from wojtek_policy import policy_node as module
    return module.PolicyNode


def test_on_joints_takes_arrival_before_any_work():
    on_joints = policy_node()._on_joints
    entry = 100 * S
    clock = WorkClock(start_ns=entry)
    node = node_stub(clock)
    on_joints(node, SlowJointState(clock, JOINTS, stamp_ns=entry - 5_000_000))
    assert clock.t > entry
    # joints_wait_ms includes the time spent processing the message.
    assert node._samples.joints_arrival_ns == entry
    assert node._samples.joints_stamp_ns == entry - 5_000_000
    # The watchdog measures arrival after the callback processes the message.
    assert node._joints_arrival.nanoseconds == clock.t
    assert clock.reads == [entry, clock.t]


def test_on_joints_rejected_sample_keeps_the_last_good_one():
    on_joints = policy_node()._on_joints
    clock = WorkClock(start_ns=100 * S)
    node = node_stub(clock)
    on_joints(node, SlowJointState(clock, JOINTS, stamp_ns=99 * S))
    good = (node._samples.joints_arrival_ns, node._samples.joints_stamp_ns)
    watchdog = node._joints_arrival
    # A message with only passive joints leaves the accepted sample unchanged.
    on_joints(node, SlowJointState(clock, ["passive"], stamp_ns=101 * S))
    assert (node._samples.joints_arrival_ns,
            node._samples.joints_stamp_ns) == good
    assert node._joints_arrival is watchdog
