"""Path-geometry scenarios: nominal speed, dry floor -- only the shape varies.

To add a course: write a builder returning a Course, append it to COURSES.
Waypoints are in the robot's start frame; compose them from
wojtek_rl.courses.geometry and wrap open shapes in lead_in() so a curve
never starts on the first step of walking.
"""

import math

import numpy as np

from wojtek_rl.courses.geometry import arc, join, lead_in, line, sine_slalom
from wojtek_rl.courses.spec import NOMINAL_SPEED as NOM
from wojtek_rl.courses.spec import Course

STRAIGHT_10M = line(10.0)  # shared with the speed/floor/push families:
# their rows must differ from straight_10m in exactly one thing


def circle(radius, n=128):
    return lead_in(arc(radius, 2 * math.pi, n=n))


def _u_turn():
    return lead_in(join(
        line(3.0),
        arc(0.5, math.pi, start=(3.0, 0.0), heading=0.0, n=48),
        line(3.0, start=(3.0, 1.0), heading=math.pi),
    ))


COURSES = [
    Course("straight_10m", "heading hold, lateral drift", STRAIGHT_10M, (NOM,)),
    Course(
        "arc_r3_90deg",
        "gentle sustained curvature",
        lead_in(arc(3.0, math.pi / 2)),
        (NOM,),
    ),
    Course("circle_r2", "sustained mild yaw + forward", circle(2.0), (NOM,)),
    Course("circle_r075", "hard yaw near the turn cap", circle(0.75), (NOM,)),
    Course(
        "figure_eight_r15",
        "curvature sign reversal",
        lead_in(join(
            arc(1.5, 2 * math.pi, n=96),
            arc(1.5, -2 * math.pi, n=96),
        )),
        (NOM,),
    ),
    Course(
        "square_3m",
        "discrete heading steps, stop-turn-go",
        lead_in(np.array(
            [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0], [0.0, 0.0]]
        )),
        (NOM,),
    ),
    Course(
        "slalom_05m",
        "high-frequency steering",
        lead_in(sine_slalom(9.0, 0.5, 3.0)),
        (NOM,),
    ),
    Course("u_turn", "worst-case heading error", _u_turn(), (NOM,)),
]
