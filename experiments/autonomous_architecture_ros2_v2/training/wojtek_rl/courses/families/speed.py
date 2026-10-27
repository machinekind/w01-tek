"""Speed scenarios: geometry shared with the nominal rows, only the command
changes -- so each row is attributable to speed alone."""

import numpy as np

from wojtek_rl.courses.families.geometry_paths import STRAIGHT_10M, circle
from wojtek_rl.courses.spec import Course

# 10 m line with a speed step every 2.5 m: the only course whose command
# changes mid-path.
_STEPS_WP = np.array([[0.0, 0.0], [2.5, 0.0], [5.0, 0.0], [7.5, 0.0], [10.0, 0.0]])

COURSES = [
    Course("straight_slow", "straight at 0.2 m/s", STRAIGHT_10M, (0.2,)),
    Course("straight_fast", "straight at 1.0 m/s", STRAIGHT_10M, (1.0,)),
    Course("circle_r2_fast", "R=2 m circle at 1.0 m/s", circle(2.0), (1.0,)),
    Course(
        "speed_steps_straight",
        "acceleration and braking along a path",
        _STEPS_WP,
        (0.2, 0.6, 1.0, 0.4),
    ),
]
