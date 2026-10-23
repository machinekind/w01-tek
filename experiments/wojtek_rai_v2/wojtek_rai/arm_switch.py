"""The operator's arm/disarm switch for the panel.

Arming is a human action: the LLM has no arm tool and /wojtek/arm is on the
tools' forbidden list (limits.FORBIDDEN). The panel calls the service itself,
on the camera feed's node (spun on its own thread), and shows the robot's
answer. real_io refuses to arm unless the robot stands in the home pose, so
a refusal is normal and is reported verbatim, never raised.
"""

from __future__ import annotations

import time
from typing import Tuple

from std_srvs.srv import SetBool, Trigger

ARM_SERVICE = "/wojtek/arm"            # real_io: targets reach the motors only while armed
POLICY_SERVICE = "/wojtek/enable"      # policy_node: the RL gait computes targets only while enabled
STAND_UP_SERVICE = "/wojtek/stand_up"  # real_io: slow ramp to the home pose (refused while armed)
LIE_DOWN_SERVICE = "/wojtek/lie_down"  # real_io: slow ramp to the folded pose (refused while armed)
SERVICE_WAIT_S = 3.0
CALL_TIMEOUT_S = 5.0
POLL_S = 0.05


def _call(node, srv_type, service: str, req, timeout: float) -> Tuple[bool, str]:
    client = node.create_client(srv_type, service)
    try:
        if not client.wait_for_service(timeout_sec=min(timeout, SERVICE_WAIT_S)):
            return False, f"{service} not available (is the robot stack up and the link alive?)"
        future = client.call_async(req)
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                future.cancel()
                return False, f"{service} timed out after {timeout:.0f} s (link down?)"
            time.sleep(POLL_S)
        res = future.result()
        return bool(res.success), str(res.message)
    finally:
        # Never leave a client per click on the node.
        node.destroy_client(client)


def call_set_bool(node, service: str, value: bool, timeout: float = CALL_TIMEOUT_S) -> Tuple[bool, str]:
    """Call a std_srvs/SetBool service; (success, message from the robot or the failure)."""
    req = SetBool.Request()
    req.data = bool(value)
    return _call(node, SetBool, service, req, timeout)


def call_trigger(node, service: str, timeout: float = CALL_TIMEOUT_S) -> Tuple[bool, str]:
    """Call a std_srvs/Trigger service; (success, message from the robot or the failure)."""
    return _call(node, Trigger, service, Trigger.Request(), timeout)


def stand_up(node, timeout: float = CALL_TIMEOUT_S) -> Tuple[bool, str]:
    return call_trigger(node, STAND_UP_SERVICE, timeout)


def lie_down(node, timeout: float = CALL_TIMEOUT_S) -> Tuple[bool, str]:
    return call_trigger(node, LIE_DOWN_SERVICE, timeout)


def set_armed(node, armed: bool, timeout: float = CALL_TIMEOUT_S) -> Tuple[bool, str]:
    return call_set_bool(node, ARM_SERVICE, armed, timeout)


def set_policy_enabled(node, enabled: bool, timeout: float = CALL_TIMEOUT_S) -> Tuple[bool, str]:
    return call_set_bool(node, POLICY_SERVICE, enabled, timeout)
