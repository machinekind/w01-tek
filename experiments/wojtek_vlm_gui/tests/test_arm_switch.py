"""The panel's arm switch against a mock node. No ROS runtime.

The switch is the operator's: the page calls the robot's services directly
on its own node and shows the robot's answer verbatim. Needs rclpy
importable (the wojtek_vlm_gui container); skipped elsewhere.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("rclpy")

from wojtek_vlm_gui import limits  # noqa: E402
from wojtek_vlm_gui.arm_switch import (  # noqa: E402
    ARM_SERVICE,
    LIE_DOWN_SERVICE,
    POLICY_SERVICE,
    STAND_UP_SERVICE,
    lie_down,
    set_armed,
    set_policy_enabled,
    stand_up,
)


def _node(available=True, success=True, message="armed", done=True):
    node = MagicMock()
    client = node.create_client.return_value
    client.wait_for_service.return_value = available
    future = client.call_async.return_value
    future.done.return_value = done
    future.result.return_value = MagicMock(success=success, message=message)
    return node, client, future


def test_the_services_are_the_ones_limits_allows():
    assert {ARM_SERVICE, POLICY_SERVICE, STAND_UP_SERVICE, LIE_DOWN_SERVICE} == set(limits.WRITABLE_SERVICES)


def test_arming_calls_the_arm_service_with_true_and_reports_the_answer():
    node, client, _ = _node(message="armed")

    ok, msg = set_armed(node, True, timeout=0.2)

    assert (ok, msg) == (True, "armed")
    assert client.call_async.call_args.args[0].data is True
    node.destroy_client.assert_called_once_with(client)


def test_disarming_sends_false():
    node, client, _ = _node(message="disarmed")

    ok, _ = set_armed(node, False, timeout=0.2)

    assert ok
    assert client.call_async.call_args.args[0].data is False


def test_a_refusal_is_returned_verbatim_not_raised():
    node, _, _ = _node(success=False, message="joint displacement from home pose ... refusing to arm")

    ok, msg = set_armed(node, True, timeout=0.2)

    assert not ok
    assert "refusing to arm" in msg


def test_missing_service_fails_with_a_clear_message():
    node, client, _ = _node(available=False)

    ok, msg = set_armed(node, True, timeout=0.2)

    assert not ok
    assert ARM_SERVICE in msg and "not available" in msg
    client.call_async.assert_not_called()
    node.destroy_client.assert_called_once_with(client)


def test_a_hung_call_times_out_and_still_cleans_up():
    node, client, future = _node(done=False)

    ok, msg = set_armed(node, True, timeout=0.05)

    assert not ok
    assert "timed out" in msg
    future.cancel.assert_called_once()
    node.destroy_client.assert_called_once_with(client)


def test_policy_switch_calls_the_enable_service():
    node, client, _ = _node(message="policy disabled")

    ok, msg = set_policy_enabled(node, False, timeout=0.2)

    assert (ok, msg) == (True, "policy disabled")
    assert node.create_client.call_args.args[1] == POLICY_SERVICE
    assert client.call_async.call_args.args[0].data is False


def test_stand_up_and_lie_down_call_their_trigger_services():
    from std_srvs.srv import Trigger

    node, client, _ = _node(message="stand_up: ramping over 4.0 s")
    ok, msg = stand_up(node, timeout=0.2)
    assert (ok, msg) == (True, "stand_up: ramping over 4.0 s")
    assert node.create_client.call_args.args == (Trigger, STAND_UP_SERVICE)

    node, client, _ = _node(message="armed -- disarm before ramping", success=False)
    ok, msg = lie_down(node, timeout=0.2)
    assert not ok and "disarm before ramping" in msg
    assert node.create_client.call_args.args == (Trigger, LIE_DOWN_SERVICE)
