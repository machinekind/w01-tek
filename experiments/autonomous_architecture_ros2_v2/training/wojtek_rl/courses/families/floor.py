"""Floor-friction scenarios: geometry and speed stay nominal, only the
contact friction changes (applied to floor AND feet -- see
rollout.friction_geom_ids for why the floor alone would be a no-op)."""

from wojtek_rl.courses.families.geometry_paths import STRAIGHT_10M, circle
from wojtek_rl.courses.spec import NOMINAL_SPEED as NOM
from wojtek_rl.courses.spec import SLIPPERY_FRICTION, Course

COURSES = [
    Course(
        "straight_slippery",
        f"straight on mu={SLIPPERY_FRICTION}",
        STRAIGHT_10M,
        (NOM,),
        friction=SLIPPERY_FRICTION,
    ),
    Course(
        "circle_r1_slippery",
        f"turning on mu={SLIPPERY_FRICTION}",
        circle(1.0),
        (NOM,),
        friction=SLIPPERY_FRICTION,
    ),
]
