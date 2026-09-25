"""The Course dataclass and the shared scenario-design constants.

A scenario varies exactly ONE thing off these nominals (flat floor, model
friction, NOMINAL_SPEED, no disturbance) so a bad row has a single
interpretation -- see the package docstring.
"""

from dataclasses import dataclass

import numpy as np

NOMINAL_SPEED = 0.5  # m/s, every course unless the scenario IS about speed
SLIPPERY_FRICTION = 0.4  # sliding friction for the slippery rows
PUSH_VEL = 0.6  # m/s lateral impulse added to base qvel, push rows

HEIGHT_CMD = 0.125  # the anchor battery.py pins every scenario at


@dataclass(frozen=True)
class Course:
    """One benchmark scenario.

    `waypoints` are (K, 2) in the robot's *start* frame (origin at the base,
    +x along its initial heading), measured after the standing settle so a
    seed's reset pose cannot shift the course under it. `speeds` is one
    commanded speed per segment (len K-1), or a single-element tuple applied
    to every segment. `friction` overrides the contact sliding friction
    (None = the model's own value). `push_at_m` places one deterministic
    lateral impulse of PUSH_VEL at that arclength along the course.
    """

    name: str
    isolates: str
    waypoints: np.ndarray
    speeds: tuple[float, ...]
    friction: float | None = None
    push_at_m: float | None = None

    @property
    def segment_speeds(self) -> np.ndarray:
        n = len(self.waypoints) - 1
        if len(self.speeds) == 1:
            return np.full(n, self.speeds[0], dtype=float)
        if len(self.speeds) != n:
            raise ValueError(
                f"{self.name}: {len(self.speeds)} speeds for {n} segments"
            )
        return np.asarray(self.speeds, dtype=float)


@dataclass(frozen=True)
class SpinCourse:
    """A rotate-in-place scenario: no path, one held command.

    The robot gets [0, 0, wz, HEIGHT_CMD] from stand and must accumulate
    `turns` full rotations in the commanded direction within the step
    budget. There is nothing here for the pure-pursuit follower to do, so
    these rows get their own rollout and scoring (rotation and positional
    drift instead of cross-track and grip -- see scoring.spin_seed_result).
    """

    name: str
    isolates: str
    wz: float  # rad/s, signed: + spins left (CCW)
    turns: float = 1.0  # full rotations required for completion


Scenario = Course | SpinCourse
