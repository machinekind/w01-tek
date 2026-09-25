"""Scenario families, one module each. Order here IS the report order.

To add a scenario: append a Course to the right family's COURSES (or add a
new family module and list it in FAMILY_MODULES). Names must stay unique --
course_catalogue() asserts it.
"""

from wojtek_rl.courses.families import disturbance, floor, geometry_paths, speed, spin
from wojtek_rl.courses.spec import Scenario

FAMILY_MODULES = (geometry_paths, speed, floor, disturbance, spin)


def course_catalogue() -> dict[str, Scenario]:
    """name -> Course, in family order. The fixed benchmark set.

    Split out (like battery.battery_scenarios) so tests and any future
    report section reuse the exact same course definitions.
    """
    catalogue: dict[str, Scenario] = {}
    for mod in FAMILY_MODULES:
        for course in mod.COURSES:
            assert course.name not in catalogue, f"duplicate course {course.name}"
            catalogue[course.name] = course
    return catalogue
