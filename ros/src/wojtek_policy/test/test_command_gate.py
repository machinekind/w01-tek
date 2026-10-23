"""The /cmd_vel dead-man of policy_node, as pure logic.

Run: pytest ros/src/wojtek_policy/test/test_command_gate.py (numpy only).

The gate holds the latest velocity command and hands it back as long as
the source keeps talking; once the source has been silent for longer than
the timeout, the velocities read zero while a commanded height (4-D
policies) is kept, so a dead planner stops the robot without dropping it.
"""

import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from wojtek_policy.command_gate import CommandGate  # noqa: E402


def test_zero_before_any_command():
    gate = CommandGate(timeout_s=0.5, fill=np.array([0.13]))
    cmd, expired = gate.current(now=10.0)
    np.testing.assert_array_equal(cmd, [0.0, 0.0, 0.0, 0.13])
    assert not expired  # nothing to expire yet: idle is not a fault


def test_fresh_command_passes_through():
    gate = CommandGate(timeout_s=0.5)
    gate.update(np.array([0.3, 0.0, 0.1]), now=1.0)
    cmd, expired = gate.current(now=1.4)
    np.testing.assert_array_equal(cmd, [0.3, 0.0, 0.1])
    assert not expired


def test_silence_zeroes_velocities_and_keeps_height():
    gate = CommandGate(timeout_s=0.5, fill=np.array([0.13]))
    gate.update(np.array([0.3, 0.1, 0.1, 0.16]), now=1.0)
    cmd, expired = gate.current(now=1.6)
    np.testing.assert_array_equal(cmd, [0.0, 0.0, 0.0, 0.16])
    assert expired


def test_exactly_at_timeout_is_still_fresh():
    gate = CommandGate(timeout_s=0.5)
    gate.update(np.array([0.3, 0.0, 0.0]), now=1.0)
    cmd, expired = gate.current(now=1.5)
    np.testing.assert_array_equal(cmd, [0.3, 0.0, 0.0])
    assert not expired


def test_new_command_rearms_after_expiry():
    gate = CommandGate(timeout_s=0.5)
    gate.update(np.array([0.3, 0.0, 0.0]), now=1.0)
    assert gate.current(now=2.0)[1]
    gate.update(np.array([0.2, 0.0, 0.0]), now=2.0)
    cmd, expired = gate.current(now=2.1)
    np.testing.assert_array_equal(cmd, [0.2, 0.0, 0.0])
    assert not expired


def test_timeout_zero_never_expires():
    gate = CommandGate(timeout_s=0.0)
    gate.update(np.array([0.3, 0.0, 0.0]), now=1.0)
    cmd, expired = gate.current(now=1e6)
    np.testing.assert_array_equal(cmd, [0.3, 0.0, 0.0])
    assert not expired


def test_update_does_not_alias_the_callers_array():
    gate = CommandGate(timeout_s=0.5)
    src = np.array([0.3, 0.0, 0.0])
    gate.update(src, now=1.0)
    src[0] = 9.0
    cmd, _ = gate.current(now=1.1)
    assert cmd[0] == 0.3
