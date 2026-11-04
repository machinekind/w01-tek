"""Pure tests for text command mapping, timeout and cancellation.

Run without ROS:
    pytest ros/src/wojtek_teleop/test/test_text_commander.py

The core emits velocity while active, one zero on stop/cancel/timeout,
then silence. A cancelled command must not resume when autonomy is enabled.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wojtek_teleop.text_command_state import CommandState  # noqa: E402

V = 0.3
W = 0.5
TIMEOUT = 2.0


def make_state(v_forward=V, w_turn=W, timeout=TIMEOUT):
    return CommandState(v_forward=v_forward, w_turn=w_turn,
                        command_timeout=timeout)


class TestMapping:
    def test_forward_maps_to_linear_x(self):
        s = make_state()
        assert s.handle("forward", now=0.0)
        assert s.twist_to_publish(now=0.0) == (V, 0.0)

    def test_left_maps_to_positive_angular_z(self):
        s = make_state()
        assert s.handle("left", now=0.0)
        assert s.twist_to_publish(now=0.0) == (0.0, W)

    def test_right_maps_to_negative_angular_z(self):
        s = make_state()
        assert s.handle("right", now=0.0)
        assert s.twist_to_publish(now=0.0) == (0.0, -W)

    def test_parameter_values_flow_through(self):
        s = make_state(v_forward=0.7, w_turn=1.1)
        s.handle("forward", now=0.0)
        assert s.twist_to_publish(now=0.0) == (0.7, 0.0)
        s.handle("left", now=0.0)
        assert s.twist_to_publish(now=0.0) == (0.0, 1.1)

    def test_command_with_surrounding_whitespace(self):
        # `ros2 topic pub ... "data: forward "` should still drive.
        s = make_state()
        assert s.handle("  forward\n", now=0.0)
        assert s.twist_to_publish(now=0.0) == (V, 0.0)


class TestTimeout:
    def test_active_until_deadline_then_one_zero_then_silence(self):
        s = make_state()
        s.handle("forward", now=10.0)
        assert s.twist_to_publish(now=10.0 + TIMEOUT) == (V, 0.0)
        assert s.twist_to_publish(now=10.0 + TIMEOUT + 0.01) == (0.0, 0.0)
        for dt in (0.02, 1.0, 100.0):
            assert s.twist_to_publish(now=10.0 + TIMEOUT + dt) is None

    def test_resend_before_deadline_extends_it(self):
        s = make_state()
        s.handle("forward", now=0.0)
        s.handle("forward", now=1.5)
        assert s.twist_to_publish(now=1.5 + TIMEOUT) == (V, 0.0)
        assert s.twist_to_publish(now=1.5 + TIMEOUT + 0.01) == (0.0, 0.0)

    def test_new_command_after_timeout_rearms(self):
        s = make_state()
        s.handle("forward", now=0.0)
        assert s.twist_to_publish(now=TIMEOUT + 0.1) == (0.0, 0.0)
        assert s.twist_to_publish(now=TIMEOUT + 0.2) is None
        assert s.handle("left", now=100.0)
        assert s.twist_to_publish(now=100.0) == (0.0, W)

    def test_new_command_while_active_remaps(self):
        s = make_state()
        s.handle("forward", now=0.0)
        s.handle("right", now=0.5)
        assert s.twist_to_publish(now=0.5) == (0.0, -W)


class TestStop:
    def test_explicit_stop_publishes_one_zero_then_silence(self):
        s = make_state()
        s.handle("forward", now=0.0)
        assert s.handle("stop", now=0.5)
        assert s.twist_to_publish(now=0.5) == (0.0, 0.0)
        assert s.twist_to_publish(now=0.6) is None

    def test_stop_while_idle_publishes_nothing(self):
        # A stop burst from idle would shout over whoever else is driving.
        s = make_state()
        assert s.handle("stop", now=0.0)
        assert s.twist_to_publish(now=0.0) is None

    def test_stop_after_timeout_stop_stays_silent(self):
        s = make_state()
        s.handle("forward", now=0.0)
        assert s.twist_to_publish(now=TIMEOUT + 0.1) == (0.0, 0.0)
        s.handle("stop", now=TIMEOUT + 0.2)
        assert s.twist_to_publish(now=TIMEOUT + 0.3) is None


class TestUnknownCommand:
    def test_unknown_returns_false_no_exception(self):
        s = make_state()
        assert not s.handle("backflip", now=0.0)
        assert s.twist_to_publish(now=0.0) is None

    def test_unknown_while_active_stops(self):
        s = make_state()
        s.handle("forward", now=0.0)
        assert not s.handle("banana", now=0.5)
        assert s.twist_to_publish(now=0.5) == (0.0, 0.0)
        assert s.twist_to_publish(now=0.6) is None

    def test_unknown_while_idle_stays_silent(self):
        s = make_state()
        assert not s.handle("banana", now=0.0)
        assert s.twist_to_publish(now=0.0) is None


class TestCancel:
    def test_cancel_forgets_command_before_deadline(self):
        s = make_state()
        s.handle("forward", now=0.0)
        assert s.twist_to_publish(now=0.1) == (V, 0.0)
        s.cancel()
        assert s.twist_to_publish(now=0.2) == (0.0, 0.0)
        # Re-enabling the mux must not revive this source's old command.
        assert s.twist_to_publish(now=0.8) is None
        assert s.twist_to_publish(now=1.5) is None

    def test_cancel_while_idle_does_not_publish(self):
        s = make_state()
        s.cancel()
        assert s.twist_to_publish(now=0.0) is None

    def test_only_a_new_command_can_restart_after_cancel(self):
        s = make_state()
        s.handle("forward", now=0.0)
        s.cancel()
        assert s.twist_to_publish(now=0.1) == (0.0, 0.0)
        assert s.handle("left", now=0.2)
        assert s.twist_to_publish(now=0.3) == (0.0, W)
