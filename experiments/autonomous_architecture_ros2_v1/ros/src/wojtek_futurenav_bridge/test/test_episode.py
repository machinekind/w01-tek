"""Model-free tests for the FutureNav episode state machine."""

import math

import pytest

from wojtek_futurenav_bridge.episode import FutureNavEpisode
from wojtek_futurenav_bridge.futurenav_http import FORWARD_STEP_M

DT = 0.02  # the 50 Hz command tick


class Walker:
    """Kinematic pose integrator: applies the commands the episode emits."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x, self.y, self.yaw = x, y, yaw

    def step(self, vx, vy, wyaw):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (c * vx - s * vy) * DT
        self.y += (s * vx + c * vy) * DT
        self.yaw += wyaw * DT


def run_until_idle(ep, walker, max_ticks=2000, frozen=False):
    """Tick until the executor drains (or the episode ends)."""
    for _ in range(max_ticks):
        cmd = ep.tick(walker.x, walker.y, walker.yaw)
        if cmd is None:
            return
        if not frozen:
            walker.step(*cmd)
        if ep.needs_decision or not ep.active:
            return
    raise AssertionError("executor never drained")


def start(instruction="walk to the sofa", **kwargs):
    ep = FutureNavEpisode(**kwargs)
    ep.start(instruction)
    return ep


def test_forward_action_walks_a_quarter_metre():
    ep = start()
    walker = Walker()
    assert ep.needs_decision
    ep.mark_decision_pending()
    ep.apply_action("MOVE_FORWARD")
    run_until_idle(ep, walker)
    assert ep.active
    assert ep.needs_decision
    dist = math.hypot(walker.x, walker.y)
    assert dist == pytest.approx(FORWARD_STEP_M, abs=0.08)


def test_turn_then_forward_changes_heading():
    ep = start()
    walker = Walker()
    ep.apply_action("TURN_LEFT")
    run_until_idle(ep, walker)
    assert walker.yaw > math.radians(8)  # 15 deg minus tolerance
    ep.apply_action("MOVE_FORWARD")
    run_until_idle(ep, walker)
    assert walker.y > 0.02  # walked along the new heading


def test_stop_finishes_done():
    ep = start()
    ep.apply_action("STOP")
    assert not ep.active
    end = ep.take_end()
    assert end.outcome == "done"
    assert ep.take_end() is None  # consumed


def test_unknown_action_aborts():
    ep = start()
    ep.apply_action("FLY")
    end = ep.take_end()
    assert end.outcome == "aborted"
    assert "unknown action" in end.reason


def test_step_budget_aborts():
    ep = start(max_steps=2)
    walker = Walker()
    for _ in range(2):
        ep.apply_action("MOVE_FORWARD")
        run_until_idle(ep, walker)
    ep.apply_action("MOVE_FORWARD")
    end = ep.take_end()
    assert end.outcome == "aborted"
    assert "step budget" in end.reason


def test_anti_spin_aborts():
    ep = start(max_rotation=3)
    walker = Walker()
    for _ in range(2):
        ep.apply_action("TURN_LEFT")
        run_until_idle(ep, walker)
    ep.apply_action("TURN_RIGHT")
    end = ep.take_end()
    assert end.outcome == "aborted"
    assert "spinning" in end.reason


def test_forward_resets_rotation_streak():
    ep = start(max_rotation=3)
    walker = Walker()
    for _ in range(2):
        ep.apply_action("TURN_LEFT")
        run_until_idle(ep, walker)
    ep.apply_action("MOVE_FORWARD")
    run_until_idle(ep, walker)
    for _ in range(2):
        ep.apply_action("TURN_LEFT")
        run_until_idle(ep, walker)
    assert ep.active  # streak restarted, no abort


def test_wedged_robot_aborts_after_blocked_streak():
    # The walker never moves: every move stall-aborts, and max_blocked=2
    # ends the episode on the second one.
    ep = start(max_blocked=2)
    walker = Walker()
    ep.apply_action("MOVE_FORWARD")
    run_until_idle(ep, walker, frozen=True)
    assert ep.active
    assert ep.executor.blocked == 1
    ep.apply_action("MOVE_FORWARD")
    run_until_idle(ep, walker, frozen=True)
    end = ep.take_end()
    assert end.outcome == "aborted"
    assert "wedged" in end.reason


def test_zeros_while_decision_pending():
    ep = start()
    ep.mark_decision_pending()
    assert ep.tick(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)


def test_cancel_and_late_reply_dropped():
    ep = start()
    ep.mark_decision_pending()
    ep.cancel("stop requested")
    end = ep.take_end()
    assert end.outcome == "cancelled"
    ep.apply_action("MOVE_FORWARD")  # the late HTTP reply
    assert not ep.active
    assert ep.tick(0.0, 0.0, 0.0) is None


def test_new_instruction_supersedes():
    ep = start("go to the tv")
    ep.apply_action("MOVE_FORWARD")
    ep.start("go to the sofa")
    end = ep.take_end()
    assert end.outcome == "cancelled"
    assert ep.active
    assert ep.instruction == "go to the sofa"
    assert ep.steps == 0


def test_failed_decision_aborts():
    ep = start()
    ep.mark_decision_pending()
    ep.fail_decision("connection refused")
    end = ep.take_end()
    assert end.outcome == "aborted"
    assert "decision failed" in end.reason


def test_empty_instruction_rejected():
    ep = FutureNavEpisode()
    with pytest.raises(ValueError):
        ep.start("   ")


def test_stale_epoch_reply_does_not_steer_the_next_episode():
    # Decision requested for episode A; the user supersedes with episode B
    # before the reply lands.  The stale reply must be dropped, and B must
    # still be waiting for its own (fresh) decision.
    ep = start("go to the tv")
    stale = ep.mark_decision_pending()
    ep.start("go to the sofa")
    ep.take_end()  # A's cancelled event
    ep.apply_action("MOVE_FORWARD", epoch=stale)
    assert ep.needs_decision  # nothing was submitted
    assert ep.steps == 0
    fresh = ep.mark_decision_pending()
    ep.apply_action("MOVE_FORWARD", epoch=fresh)
    assert ep.steps == 1
    assert ep.executor.active


def test_stale_epoch_failure_does_not_abort_the_next_episode():
    ep = start("go to the tv")
    stale = ep.mark_decision_pending()
    ep.start("go to the sofa")
    ep.take_end()
    ep.fail_decision("timeout", epoch=stale)
    assert ep.active
    assert ep.take_end() is None
