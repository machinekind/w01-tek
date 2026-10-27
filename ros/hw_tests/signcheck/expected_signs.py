#!/usr/bin/env python3
"""Expected encoder sign of every actuated joint when the robot goes from its
folded (lying) pose to its home (standing) pose, in the drives' sense: the
profile's poses mapped through its joint map. Usage:

    PYTHONPATH=ros/src/wojtek_policy python3 expected_signs.py [robot]
"""
import os
import sys

from wojtek_policy import poses, robots
from wojtek_policy.joint_map import JointMap

name = sys.argv[1] if len(sys.argv) > 1 else robots.DEFAULT_ROBOT
profile = robots.get(name)
config = os.path.join(os.path.dirname(poses.__file__), "..", "config", profile.joint_map)
jm = JointMap(config)
p = poses.for_robot(profile.training_robot)
names = poses.ACTUATOR_NAMES
delta = jm.to_urdf(names, p.home_ctrl) - jm.to_urdf(names, p.folded_ctrl)
print(f"{name}: encoder change lying -> standing (URDF = drive sense), rad")
for n, v in zip(names, delta):
    print(f"  {n:28s} {'+' if v > 0.01 else '-' if v < -0.01 else '0'}  {v:+.2f}")
