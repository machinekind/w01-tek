"""GotoController on a desk: no ROS, a fake costmap probe."""

import math

import pytest

from wojtek_nav.goto import GotoController, INSCRIBED, LETHAL

FREE = lambda x, y: 0  # noqa: E731


def _drive(ctrl, pose, goal_at=0.0, probe=FREE, dt=0.05, max_s=30.0):
    """Integrate the unicycle until the controller stops; return (pose, status, t)."""
    x, y, yaw = pose
    t = goal_at
    while t < goal_at + max_s:
        cmd = ctrl.step((x, y, yaw), t, probe)
        if cmd.status in ("reached", "blocked", "idle"):
            return (x, y, yaw), cmd.status, t
        x += cmd.vx * math.cos(yaw) * dt
        y += cmd.vx * math.sin(yaw) * dt
        yaw += cmd.wz * dt
        t += dt
    return (x, y, yaw), "timeout", t


def test_idle_until_a_goal_arrives():
    ctrl = GotoController()
    assert ctrl.step((0, 0, 0), 0.0, FREE).status == "idle"
    assert ctrl.goal is None


def test_drives_straight_to_a_goal_ahead_and_stops():
    ctrl = GotoController(goal_timeout=100)
    ctrl.set_goal(1.0, 0.0, 0.0)
    (x, y, _), status, t = _drive(ctrl, (0, 0, 0))
    assert status == "reached"
    assert abs(x - 1.0) < ctrl.reach_tolerance and abs(y) < 0.02
    # ~1 m at up to 0.3 m/s, slowing in: well under 10 s
    assert t < 10
    # Reached is sent once, then idle; the goal is consumed.
    assert ctrl.goal is None
    assert ctrl.step((x, y, 0), t, FREE).status == "idle"


def test_turns_in_place_first_when_the_goal_is_behind():
    ctrl = GotoController(goal_timeout=100)
    ctrl.set_goal(-1.0, 0.0, 0.0)
    cmd = ctrl.step((0, 0, 0), 0.0, FREE)
    assert cmd.status == "turning" and cmd.vx == 0.0 and abs(cmd.wz) == ctrl.w_max
    (x, y, yaw), status, _ = _drive(ctrl, (0, 0, 0))
    assert status == "reached"
    assert abs(x + 1.0) < ctrl.reach_tolerance


def test_steers_towards_a_goal_off_to_the_side():
    ctrl = GotoController(goal_timeout=100)
    ctrl.set_goal(1.0, 0.5, 0.0)
    (x, y, _), status, _ = _drive(ctrl, (0, 0, 0))
    assert status == "reached"
    assert math.hypot(x - 1.0, y - 0.5) < ctrl.reach_tolerance


def test_never_exceeds_the_limits():
    ctrl = GotoController(goal_timeout=100)
    ctrl.set_goal(3.0, 2.0, 0.0)
    x, y, yaw, t = 0.0, 0.0, 0.0, 0.0
    for _ in range(400):
        cmd = ctrl.step((x, y, yaw), t, FREE)
        assert 0.0 <= cmd.vx <= ctrl.v_max + 1e-9
        assert abs(cmd.wz) <= ctrl.w_max + 1e-9
        x += cmd.vx * math.cos(yaw) * 0.05
        y += cmd.vx * math.sin(yaw) * 0.05
        yaw += cmd.wz * 0.05
        t += 0.05


@pytest.mark.parametrize("cost", [INSCRIBED, LETHAL])
def test_stops_when_the_line_ahead_is_blocked(cost):
    """An obstacle the costmap remembers, right in the path: no planner,
    the robot stops and says so, and stays stopped."""
    ctrl = GotoController(goal_timeout=100)
    ctrl.set_goal(2.0, 0.0, 0.0)
    wall = lambda x, y: cost if x > 0.9 else 0  # noqa: E731
    (x, y, _), status, _ = _drive(ctrl, (0, 0, 0), probe=wall)
    assert status == "blocked"
    assert x < 0.9  # stopped short of it, by at least the lookahead
    assert ctrl.step((x, y, 0), 5.0, wall).status == "blocked"


def test_blocked_resumes_when_the_map_clears():
    ctrl = GotoController(goal_timeout=100)
    ctrl.set_goal(2.0, 0.0, 0.0)
    wall = lambda x, y: LETHAL if x > 0.9 else 0  # noqa: E731
    (x, y, yaw), status, t = _drive(ctrl, (0, 0, 0), probe=wall)
    assert status == "blocked"
    (x, y, _), status, _ = _drive(ctrl, (x, y, yaw), goal_at=t, probe=FREE)
    assert status == "reached" and abs(x - 2.0) < ctrl.reach_tolerance


def test_slows_near_things_but_keeps_going():
    ctrl = GotoController(goal_timeout=100)
    ctrl.set_goal(3.0, 0.0, 0.0)
    gradient = lambda x, y: ctrl.slow_cost if x > 0.9 else 0  # noqa: E731
    cmd_far = ctrl.step((0.0, 0.0, 0.0), 0.0, gradient)
    cmd_near = ctrl.step((0.7, 0.0, 0.0), 0.0, gradient)
    assert cmd_far.status == cmd_near.status == "driving"
    assert cmd_near.vx == pytest.approx(cmd_far.vx * 0.5)


def test_outside_the_window_counts_as_free():
    ctrl = GotoController(goal_timeout=100)
    ctrl.set_goal(2.0, 0.0, 0.0)
    assert ctrl.step((0, 0, 0), 0.0, lambda x, y: None).status == "driving"


def test_a_setpoint_expires_the_dead_man_stops_the_robot():
    """The VLM must keep talking: a goal older than goal_timeout is gone
    even if unreached, and the controller goes idle."""
    ctrl = GotoController(goal_timeout=3.0)
    ctrl.set_goal(5.0, 0.0, 0.0)
    assert ctrl.step((0, 0, 0), 2.9, FREE).status == "driving"
    assert ctrl.step((0, 0, 0), 3.1, FREE).status == "idle"
    assert ctrl.goal is None


def test_cancel_drops_the_setpoint_before_the_dead_man():
    """A stop from the brain or the console must not wait 3 s."""
    ctrl = GotoController(goal_timeout=3.0)
    ctrl.set_goal(5.0, 0.0, 0.0)
    assert ctrl.step((0, 0, 0), 0.5, FREE).status == "driving"
    ctrl.cancel()
    cmd = ctrl.step((0, 0, 0), 0.6, FREE)
    assert (cmd.vx, cmd.wz, cmd.status) == (0.0, 0.0, "idle")
    assert ctrl.goal is None
    # And a cancelled controller takes the next goal like a fresh one.
    ctrl.set_goal(1.0, 0.0, 1.0)
    assert ctrl.step((0, 0, 0), 1.0, FREE).status == "driving"


def test_a_new_goal_replaces_the_old_one_and_restarts_the_clock():
    ctrl = GotoController(goal_timeout=3.0)
    ctrl.set_goal(5.0, 0.0, 0.0)
    ctrl.set_goal(1.0, 0.0, 2.5)
    assert ctrl.goal == (1.0, 0.0)
    assert ctrl.step((0, 0, 0), 5.0, FREE).status == "driving"


def test_blocked_line_ahead_still_lets_the_robot_turn_towards_its_goal():
    """Mid-turn the heading sweeps across whatever stands beside the robot;
    that must not freeze it. Blocked is for a robot pointed at its goal."""
    ctrl = GotoController(goal_timeout=100)
    ctrl.set_goal(2.0, 0.0, 0.0)
    wall_to_the_left = lambda x, y: LETHAL if y > 0.3 else 0  # noqa: E731
    # Nose 30 deg left of the goal: the lookahead crosses the wall.
    cmd = ctrl.step((0.0, 0.0, 0.5), 0.0, wall_to_the_left)
    assert cmd.status == "turning" and cmd.vx == 0.0 and cmd.wz < 0
    # Aligned and the wall really is ahead: blocked.
    ahead = lambda x, y: LETHAL if x > 0.4 else 0  # noqa: E731
    assert ctrl.step((0.0, 0.0, 0.0), 0.0, ahead).status == "blocked"
