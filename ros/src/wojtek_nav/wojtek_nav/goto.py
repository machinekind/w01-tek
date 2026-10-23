"""Drive to a setpoint, straight, with the costmap as the safety veto.

Pure logic, no ROS: a `GotoController` is stepped with the robot's planar
pose, the current goal and a probe into the costmap, and answers with a
velocity command and a status word. The ROS node (goto_node.py) is a thin
shell around it, so the behaviour is testable on a desk.

The contract this implements (decided with the owner, 2026-09-25): the VLM
hands over the NEXT setpoint, a point in the odom frame ~1 m ahead, about
once a second; the robot walks straight at it. No local planner: the
strategy -- which way round the crate -- is the VLM's, from the picture.
What the robot keeps for itself is the veto: the costmap remembers what
the camera no longer sees, and if the footprint ahead would meet an
obstacle the robot stops and says so, until a new setpoint or a cleared
map lets it move. Strategy up there, reflexes down here.

Motion is non-holonomic on purpose even though the policy can strafe: the
camera looks where the body points, and a robot that walks sideways walks
blind. So: turn towards the setpoint first (in place when the bearing is
large), then walk, steering with a proportional yaw command.

Timing. A setpoint expires `goal_timeout` seconds after it arrived: the
VLM must keep talking to keep Wojtek walking, the same dead-man
text_commander has. Reaching the setpoint also stops the robot; there is
no "and then what" -- the next setpoint is the VLM's answer to the next
picture.
"""

import math
from dataclasses import dataclass

# Costmap cell values (nav_msgs/OccupancyGrid as nav2 publishes it):
# 100 lethal, 99 inscribed (an obstacle within the footprint's inscribed
# radius of this cell), lower = inflation gradient, 0 free, -1 never seen.
LETHAL = 100
INSCRIBED = 99


@dataclass(frozen=True)
class Command:
    vx: float
    wz: float
    status: str  # idle | turning | driving | blocked | reached


class GotoController:
    """Straight-line go-to with a costmap veto.

    probe(x, y) -> cost at that odom point, or None outside the window.
    """

    def __init__(self, v_max=0.3, w_max=0.5, k_yaw=1.5, turn_in_place_rad=0.6,
                 reach_tolerance=0.15, goal_timeout=3.0,
                 lookahead=(0.25, 0.5, 0.75), slow_cost=50, aligned_rad=0.1):
        self.v_max, self.w_max, self.k_yaw = v_max, w_max, k_yaw
        self.turn_in_place_rad = turn_in_place_rad
        self.aligned_rad = aligned_rad
        self.reach_tolerance = reach_tolerance
        self.goal_timeout = goal_timeout
        self.lookahead = tuple(lookahead)
        self.slow_cost = slow_cost
        self._goal = None       # (x, y) in odom
        self._goal_time = None  # when it arrived (s)
        self._done = False      # reached / expired: sent, waiting for the next

    # -- inputs ----------------------------------------------------------

    def set_goal(self, x, y, now):
        self._goal, self._goal_time, self._done = (float(x), float(y)), now, False

    @property
    def goal(self):
        return None if self._done else self._goal

    # -- the step --------------------------------------------------------

    def step(self, pose, now, probe):
        """pose = (x, y, yaw) in odom. Returns a Command."""
        if self._goal is None or self._done:
            return Command(0.0, 0.0, "idle")
        if now - self._goal_time > self.goal_timeout:
            self._done = True
            return Command(0.0, 0.0, "idle")

        x, y, yaw = pose
        gx, gy = self._goal
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        if dist < self.reach_tolerance:
            self._done = True
            return Command(0.0, 0.0, "reached")

        bearing = _wrap(math.atan2(dy, dx) - yaw)
        wz = max(-self.w_max, min(self.w_max, self.k_yaw * bearing))
        if abs(bearing) > self.turn_in_place_rad:
            return Command(0.0, wz, "turning")

        # The veto: the costmap's inflation already encodes "an obstacle
        # within the inscribed radius of this cell" as INSCRIBED, so a few
        # cells along the line the body is about to cover stand in for a
        # swept footprint. Beyond the window (None) is treated as free:
        # the window is 3 m each way and the lookahead is under 1 m.
        worst = 0
        for d in self.lookahead:
            if d > dist:
                break
            c = probe(x + d * math.cos(yaw), y + d * math.sin(yaw))
            if c is not None and c > worst:
                worst = c
        if worst >= INSCRIBED:
            # Blocked means "cannot ADVANCE". Turning on the spot is still
            # allowed while the nose is off the goal: the line ahead is
            # the current heading's, and mid-turn it sweeps across
            # whatever stands beside the robot (seen in the corridor: a
            # turn towards the far end froze against the side wall's
            # inflation). Only a robot pointed at its goal and still
            # blocked has nothing left to try.
            if abs(bearing) > self.aligned_rad:
                return Command(0.0, wz, "turning")
            return Command(0.0, 0.0, "blocked")

        # Slow into the goal, and slow near things.
        v = min(self.v_max, 0.6 * dist)
        if worst >= self.slow_cost:
            v *= 0.5
        return Command(v, wz, "driving")


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
