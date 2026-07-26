"""Unit tests for the mid-level command parser and executor.

The executor is exercised against a kinematic unicycle: the body-frame
command it emits is integrated directly, which is exactly what the trained
velocity-tracking policy approximates.
"""

import math

import pytest

from wojtek_rl.midlevel import (
    Backward,
    Forward,
    MidLevelExecutor,
    Stop,
    Turn,
    parse_command,
)
from wojtek_rl.navigation import NavConfig, _wrap

DT = 0.02  # 50 Hz, matches the control loop


def make_executor(**kw) -> MidLevelExecutor:
    return MidLevelExecutor(NavConfig(), **kw)


def rollout(exec_, steps, x=0.0, y=0.0, yaw=0.0):
    """Integrate the executor's commands on a kinematic unicycle."""
    for _ in range(steps):
        vx, vy, wyaw = exec_.update(x, y, yaw)
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * DT
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * DT
        yaw = _wrap(yaw + wyaw * DT)
    return x, y, yaw


# --- parser -------------------------------------------------------------


def test_parse_turn_left():
    assert parse_command("turn_left 30") == Turn(math.radians(30))


def test_parse_turn_right_is_negative():
    assert parse_command("turn_right 45") == Turn(-math.radians(45))


def test_parse_forward():
    assert parse_command("forward 1.5") == Forward(1.5)


def test_parse_stop():
    assert parse_command("stop") == Stop()


def test_parse_tolerates_case_and_whitespace():
    assert parse_command("  Turn_Left 15 ") == Turn(math.radians(15))


@pytest.mark.parametrize(
    "text",
    ["", "fly 3", "turn_left", "forward 1 2", "forward -1", "forward abc", "stop 3", "turn_right 0"],
)
def test_parse_rejects(text):
    with pytest.raises(ValueError):
        parse_command(text)


# --- executor -----------------------------------------------------------


def test_idle_emits_zero():
    ex = make_executor()
    assert ex.update(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)
    assert not ex.active


def test_turn_left_90_converges():
    ex = make_executor()
    ex.submit(parse_command("turn_left 90"))
    _, _, yaw = rollout(ex, 500)
    assert abs(_wrap(yaw - math.pi / 2)) < 0.1
    assert not ex.active


def test_turn_right_yaw_rate_negative():
    ex = make_executor()
    ex.submit(parse_command("turn_right 45"))
    vx, vy, wyaw = ex.update(0.0, 0.0, 0.0)
    assert vx == 0.0 and vy == 0.0
    assert wyaw < 0.0


def test_forward_one_meter():
    ex = make_executor()
    ex.submit(parse_command("forward 1"))
    x, y, _ = rollout(ex, 500)
    assert abs(x - 1.0) < 0.12
    assert abs(y) < 0.12
    assert not ex.active


def test_sequencing_turn_then_forward():
    ex = make_executor()
    ex.submit(parse_command("turn_left 90"))
    ex.submit(parse_command("forward 1"))
    x, y, _ = rollout(ex, 1000)
    assert abs(x) < 0.15
    assert abs(y - 1.0) < 0.15
    assert not ex.active


def test_stop_clears_queue():
    ex = make_executor()
    ex.submit(parse_command("forward 2"))
    ex.submit(parse_command("turn_left 90"))
    rollout(ex, 10)
    assert ex.active
    ex.submit(Stop())
    assert not ex.active
    assert ex.update(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)


def test_queue_overflow_raises():
    ex = make_executor(queue_max=2)
    ex.submit(Forward(1.0))
    ex.submit(Forward(1.0))
    with pytest.raises(ValueError):
        ex.submit(Forward(1.0))


def test_commands_stay_above_policy_stand_threshold():
    """The policy freezes its gait below cmd speed ~0.05; an active command
    must never emit a crawl the robot would ignore (P-control deadlock)."""
    ex = make_executor()
    ex.submit(parse_command("turn_left 10"))
    x = y = yaw = 0.0
    while ex.active:
        vx, vy, wyaw = ex.update(x, y, yaw)
        if not ex.active:
            break
        speed = math.hypot(vx, vy) + 0.3 * abs(wyaw)
        assert speed > 0.05, f"emitted sub-stand command {vx, vy, wyaw}"
        yaw = _wrap(yaw + wyaw * DT)
    ex.submit(parse_command("forward 0.1"))
    for _ in range(2000):
        vx, vy, wyaw = ex.update(x, y, yaw)
        if not ex.active:
            break
        assert math.hypot(vx, vy) + 0.3 * abs(wyaw) > 0.05
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * DT
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * DT
        yaw = _wrap(yaw + wyaw * DT)
    assert not ex.active


def test_status_reports_active_command():
    ex = make_executor()
    ex.submit(parse_command("turn_left 30"))
    ex.update(0.0, 0.0, 0.0)
    s = ex.status()
    assert s["active"] == "turn_left 30deg"
    assert s["queued"] == 0
    assert s["remaining"] > 0.0


# --- backward + stall-abort (blocked robot) ------------------------------


def test_parse_backward():
    cmd = parse_command("backward 0.3")
    assert isinstance(cmd, Backward)
    assert cmd.meters == 0.3


def test_backward_half_meter():
    ex = make_executor()
    ex.submit(parse_command("backward 0.5"))
    x, y, _ = rollout(ex, 500)
    assert abs(x + 0.5) < 0.12
    assert abs(y) < 0.12
    assert not ex.active


def test_backward_emits_negative_vx_no_turn():
    ex = make_executor()
    ex.submit(parse_command("backward 0.3"))
    vx, vy, wyaw = ex.update(0.0, 0.0, 0.0)
    assert vx < 0.0
    assert vy == 0.0 and wyaw == 0.0


def test_blocked_forward_aborts_and_reports():
    """A wall (frozen pose) must stall-abort the command, not push forever."""
    ex = make_executor()
    ex.submit(parse_command("forward 1"))
    ex.submit(parse_command("turn_left 90"))  # queued behind the doomed move
    for _ in range(ex.STALL_TICKS + 5):
        ex.update(0.0, 0.0, 0.0)  # robot never moves
    assert ex.blocked == 1
    assert not ex.active, "queue must be dropped after a stall-abort"
    assert ex.update(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)


def test_blocked_turn_aborts():
    ex = make_executor()
    ex.submit(parse_command("turn_left 90"))
    for _ in range(ex.STALL_TICKS + 5):
        ex.update(0.0, 0.0, 0.0)  # yaw frozen: pivoting against furniture
    assert ex.blocked == 1
    assert not ex.active


def test_unobstructed_moves_do_not_count_blocked():
    ex = make_executor()
    ex.submit(parse_command("forward 1"))
    ex.submit(parse_command("turn_left 90"))
    rollout(ex, 1500)
    assert ex.blocked == 0
    assert not ex.active
