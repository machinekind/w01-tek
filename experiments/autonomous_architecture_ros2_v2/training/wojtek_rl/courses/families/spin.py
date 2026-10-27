"""Rotate-in-place scenarios: one held pure-spin command per row.

Same one-variable-per-row design as the rest of the catalogue, applied to
rotation: the left/right pair isolates chirality at the nominal rate (a
policy can be asymmetric -- the stiff_b keeper shipped unable to spin right
because nothing ever tested it), and the slow/fast rows isolate rate in one
direction. Rates are benchmark constants chosen on their own merits, not any
particular policy's trained command box -- a policy that cannot execute a
row scores 0 on it, which is the row working as intended.
"""

from wojtek_rl.courses.spec import SpinCourse

SPIN_NOMINAL = 0.8  # rad/s
SPIN_SLOW = 0.4
SPIN_FAST = 1.2

COURSES = [
    SpinCourse("spin_left", "pure spin CCW at 0.8 rad/s", wz=+SPIN_NOMINAL),
    SpinCourse("spin_right", "pure spin CW at 0.8 rad/s", wz=-SPIN_NOMINAL),
    SpinCourse("spin_slow", "precision spin, 0.4 rad/s CCW", wz=+SPIN_SLOW),
    SpinCourse("spin_fast", "fast spin, 1.2 rad/s CCW", wz=+SPIN_FAST),
]
