"""Disturbance scenarios: one deterministic lateral impulse mid-course.

The push rows differ from their straight baselines in exactly the impulse
(and from each other in exactly the speed) -- tests pin this."""

from wojtek_rl.courses.families.geometry_paths import STRAIGHT_10M
from wojtek_rl.courses.spec import NOMINAL_SPEED as NOM
from wojtek_rl.courses.spec import PUSH_VEL, Course

COURSES = [
    Course(
        "straight_push",
        f"{PUSH_VEL} m/s lateral impulse at 5 m",
        STRAIGHT_10M,
        (NOM,),
        push_at_m=5.0,
    ),
    Course(
        "straight_push_fast",
        f"{PUSH_VEL} m/s lateral impulse at 5 m, 1.0 m/s",
        STRAIGHT_10M,
        (1.0,),
        push_at_m=5.0,
    ),
]
