"""Named robot poses shared by the real bringup and the MuJoCo sim.

Poses are expressed in the policy/MuJoCo actuator convention; convert with
wojtek_policy.joint_map.JointMap where URDF angles are needed. They are
robot constants (the MJCF home keyframe, the actuator order, the folding
stops) -- independent of whichever policy is loaded, but dependent on
which legs are bolted on. `for_robot(training_robot)` returns the set for
one pair of legs; the robot profile (robots.py) names the training robot.

Two sets exist:

  "wojtek"      the stock legs. Home = the MJCF "home" keyframe; folded =
                hips straight, knees against the folding stop; passive
                four-bar joints (fourth/fifth) from the knee.
  "legs_v627"   the v6.27/v6.37a four-bar legs (training variant
                legs_v627; v6.37a has the same kinematics). Home = the
                levelled stance of the built training model; folded = the
                shins-flat pose on 35 mm standoffs under the base (the
                owner's definition of 2026-10-22); passive joints
                (fourth/fifth) from the lever (third) joint, the free foot
                joint at zero.
"""

from dataclasses import dataclass, field

import numpy as np

# Canonical actuated-joint order: the MJCF actuator order every policy,
# joint-target message, and pose in this stack uses. A property of the
# robot body, the same for every set of legs; policy_node cross-checks a
# loaded policy's actuator_names against this list.
ACTUATOR_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("rear_left", "rear_right", "front_right", "front_left")
    for joint in ("first", "second", "third")
]

# Standing home pose of the STOCK legs (the MJCF "home" keyframe ctrl),
# policy/MuJoCo convention: abduction 0, hip -0.2, knee 3.1 on every leg.
HOME_CTRL = np.array([0.0, -0.2, 3.1] * 4)

# Knee (third joint) angle against the folding mechanical stop of the stock
# legs, policy/MuJoCo convention. This is the model ctrlrange lower bound:
# the near-branch end of the four-bar range, i.e. the lower leg folded flat
# against the upper leg.
FOLDED_KNEE_RAD = 0.425


def folded_ctrl(joint_names, folded_knee_rad=FOLDED_KNEE_RAD):
    """Stock boot pose: hips straight, knees folded on the mechanical stop.

    Reproducible by hand, which makes it the reference pose for zeroing the
    MD80 encoders at power-on.
    """
    return np.array(
        [
            folded_knee_rad if n.endswith("third_joint") else 0.0
            for n in joint_names
        ]
    )


# Passive four-bar joints (fourth/fifth per leg) of the STOCK legs as a
# function of that leg's knee (third) joint, both in URDF convention:
#     urdf_passive = np.polyval(coeffs, urdf_third)
# The four-bar closure only exists in the MJCF (equality connect); the URDF
# leaves these joints dangling, so anyone publishing joint states for
# robot_state_publisher/RViz has to compute them (the sim gets them from
# MuJoCo directly; real_io_node uses this table on the real robot).
# Fitted by settling the MuJoCo closure over the full knee ctrl range with
# gravity off; max fit error 7e-5 rad.
PASSIVE_FROM_KNEE = {
    "rear_left_fourth_joint": [-1.1773459848e-05, 1.9013043239e-04, -1.0367019750e-03, 1.7043550976e-03, 3.6905369958e-04, 9.8215945687e-03, 3.0319463069e-02, 2.3409816954e-01, 2.3755523458e-01],  # noqa: E501
    "rear_left_fifth_joint": [1.1773459831e-05, -1.9013043234e-04, 1.0367019764e-03, -1.7043551100e-03, -3.6905365723e-04, -9.8215946427e-03, -3.0319462994e-02, -2.3409816959e-01, -2.3756032091e-01],  # noqa: E501
    "rear_right_fourth_joint": [1.1773459848e-05, 1.9013043239e-04, 1.0367019750e-03, 1.7043550976e-03, -3.6905369959e-04, 9.8215945687e-03, -3.0319463069e-02, 2.3409816954e-01, -2.3755523458e-01],  # noqa: E501
    "rear_right_fifth_joint": [-1.1773459831e-05, -1.9013043234e-04, -1.0367019764e-03, -1.7043551100e-03, 3.6905365724e-04, -9.8215946427e-03, 3.0319462994e-02, -2.3409816959e-01, 2.3756032091e-01],  # noqa: E501
    "front_right_fourth_joint": [-1.1773459848e-05, 1.9013043239e-04, -1.0367019750e-03, 1.7043550976e-03, 3.6905369959e-04, 9.8215945687e-03, 3.0319463069e-02, 2.3409816954e-01, 2.3755523458e-01],  # noqa: E501
    "front_right_fifth_joint": [1.1773459831e-05, -1.9013043234e-04, 1.0367019764e-03, -1.7043551099e-03, -3.6905365725e-04, -9.8215946427e-03, -3.0319462994e-02, -2.3409816959e-01, -2.3756032091e-01],  # noqa: E501
    "front_left_fourth_joint": [1.1773459848e-05, 1.9013043239e-04, 1.0367019750e-03, 1.7043550976e-03, -3.6905369959e-04, 9.8215945687e-03, -3.0319463069e-02, 2.3409816954e-01, -2.3755523458e-01],  # noqa: E501
    "front_left_fifth_joint": [-1.1773459831e-05, -1.9013043234e-04, -1.0367019764e-03, -1.7043551099e-03, 3.6905365725e-04, -9.8215946427e-03, 3.0319462994e-02, -2.3409816959e-01, 2.3756032091e-01],  # noqa: E501
}

# ---------------------------------------------------------------------------
# The v6.27 legs (training variant legs_v627; the v6.37a legs on the robot
# have the same kinematics and share these numbers).
#
# Home: the "home" keyframe of the built training model
# (wojtek_description/mujoco/legs_v627/scene_mjx.xml), i.e. the levelled
# stance from training/wojtek_rl/robots.py (stand_pose): the CAD "stanie"
# keyframe with the rear legs a tenth of a height rung longer and the front
# legs a tenth shorter, so the body stands level at 0.345 m. Front and rear
# legs are mounted mirrored (knees point at each other), hence the signs.
V627_HOME_CTRL = np.array([
    0.0, -2.2515898, -2.03678985,
    0.0, -2.2515898, -2.03678985,
    0.0, 2.26669567, 1.98215855,
    0.0, 2.26669567, 1.98215855,
])

# Folded = lying, the boot/zeroing pose. The robot zeroes its encoders at
# power-on, lying, before any torque exists, so this is the pose the
# unpowered robot rests in. The owner's definition (2026-10-22): the shin
# folded against the thigh (lever on its folding limit, 0.244), the leg
# turned at the hip until the line the shin and the foot pad rest on is
# horizontal, and the body carried on four 35 mm standoffs under the base
# -- which is exactly where the base bottom sits above that line
# (computed on the v6.37a meshes: hip axis 96 mm, base bottom 35 mm above
# the floor). The hip then stands at +-2.903 rad, 0.21 rad PAST the joint
# range of the exported model (2.695): the real hip stop has to allow it,
# or the shins cannot lie flat on any standoffs -- measure on the robot.
# The abduction has no stop and is zero by convention: with the body on
# the standoffs the legs carry no load and stay where they are put.
# Front and rear legs are alike (the same part, mounted mirrored).
V627_FOLDED_CTRL = np.array([
    0.0, -2.903, -0.244346,
    0.0, -2.903, -0.244346,
    0.0, 2.903, 0.244346,
    0.0, 2.903, 0.244346,
])

# Envelope every ramp and clip target must stay inside, policy convention:
# the built legs_v627 model's actuator ctrlrange, except the hip, which
# opens to +-2.95 so the lying pose above is reachable (the model's 2.695
# is the CAD export's range, and the lying pose needs 2.903; the sim's
# joint limit still clamps there until the model is re-exported). The
# lever range stops 13 degrees short of the four-bar dead points.
V627_CTRL_LOW = np.array([
    -3.1416, -2.95, -2.83267,
    -3.1416, -2.95, -2.83267,
    -3.1416, 1.53589, 0.244346,
    -3.1416, 1.53589, 0.244346,
])
V627_CTRL_HIGH = np.array([
    3.1416, -1.53589, -0.244346,
    3.1416, -1.53589, -0.244346,
    3.1416, 2.95, 2.83267,
    3.1416, 2.95, 2.83267,
])

# Passive four-bar joints of the v6.27 legs as a function of the lever
# (third) joint of the same leg, MuJoCo (policy) convention on both sides:
# real_io converts through the robot's joint map (joint_map_v627.yaml flips
# the lever of two legs, the URDF keeping the drives' sense of rotation).
# The four-bar hangs on the thigh, so
# the hip (second) joint does not enter. Fitted on 2026-10-22 by solving
# the loop closure of the built training model over the full lever range
# (240 points, Newton on the closure sites, degree 8); max fit error
# 8e-6 rad (fourth) / 8e-5 rad (fifth).
V627_PASSIVE_FROM_LEVER = {
    "rear_left_fourth_joint": [1.0328419107e-04, 1.1741429517e-03, 4.4270191552e-03, 3.9385662959e-04, -5.4753231292e-02, -1.8814112364e-01, -1.7423133271e-01, 1.1156745444e+00, -8.1098939368e-02],  # noqa: E501
    "rear_left_fifth_joint": [-6.3023199836e-04, -9.4662952994e-03, -6.2620802310e-02, -2.5075309448e-01, -6.7248410667e-01, -1.2431403914e+00, -1.4117714506e+00, 2.3296607515e-02, -5.9831230596e-01],  # noqa: E501
    "rear_right_fourth_joint": [-1.0328419107e-04, -1.1741429517e-03, -4.4270191552e-03, -3.9385662947e-04, 5.4753231292e-02, 1.8814112364e-01, 1.7423133271e-01, -1.1156745444e+00, 8.1098939368e-02],  # noqa: E501
    "rear_right_fifth_joint": [6.3023200155e-04, 9.4662953356e-03, 6.2620802478e-02, 2.5075309489e-01, 6.7248410724e-01, 1.2431403919e+00, 1.4117714508e+00, -2.3296607470e-02, 5.9831230596e-01],  # noqa: E501
    "front_right_fourth_joint": [-1.0324588092e-04, 1.1736497102e-03, -4.4243677176e-03, 3.8612243302e-04, 5.4766551907e-02, -1.8815491290e-01, 1.7423964472e-01, 1.1156719055e+00, 8.1099272849e-02],  # noqa: E501
    "front_right_fifth_joint": [6.2733428347e-04, -9.4317448323e-03, 6.2450398365e-02, -2.5030315090e-01, 6.7179587171e-01, -1.2425237030e+00, 1.4114599052e+00, 2.3376437080e-02, 5.9830448585e-01],  # noqa: E501
    "front_left_fourth_joint": [1.0324588093e-04, -1.1736497102e-03, 4.4243677178e-03, -3.8612243353e-04, -5.4766551906e-02, 1.8815491290e-01, -1.7423964472e-01, -1.1156719055e+00, -8.1099272849e-02],  # noqa: E501
    "front_left_fifth_joint": [-6.2733428228e-04, 9.4317448193e-03, -6.2450398307e-02, 2.5030315077e-01, -6.7179587154e-01, 1.2425237029e+00, -1.4114599051e+00, -2.3376437085e-02, -5.9830448585e-01],  # noqa: E501
}

# The v6.27 URDF has one more non-fixed joint per leg: the foot joint, a
# free hinge at the closure point (foot_link touches nothing and carries no
# mesh). It has no kinematic meaning, so it is published at zero.
V627_PASSIVE_CONSTANT = {
    f"{leg}_foot_joint": 0.0
    for leg in ("rear_left", "rear_right", "front_right", "front_left")
}


@dataclass(frozen=True)
class RobotPoses:
    """The named poses and the passive-joint table of one set of legs.

    All ctrl vectors are 12-wide in ACTUATOR_NAMES order, policy/MuJoCo
    convention. The passive tables are functions of the angle of the
    actuated joint named by `passive_driver`, in `passive_convention`:
    "urdf" (the stock table, fitted URDF-to-URDF) or "mjc" (the v6.27 table,
    fitted on the training model; the publisher converts both ends through
    the robot's joint map).
    """
    training_robot: str
    home_ctrl: np.ndarray
    # The boot/zeroing pose: reproducible by hand, where lie_down ends.
    folded_ctrl: np.ndarray
    # Envelope for ramps and clips (the model's actuator ctrlrange).
    ctrl_low: np.ndarray
    ctrl_high: np.ndarray
    # Passive joint name -> polynomial coefficients over the driving joint.
    passive_from_driver: dict
    # Which actuated joint of the same leg drives the passive ones.
    passive_driver: str
    # Passive joints published at a fixed angle.
    passive_constant: dict = field(default_factory=dict)
    # Convention of passive_from_driver's input and output: "urdf" or "mjc".
    passive_convention: str = "urdf"


# The stock legs' ctrlrange (wojtek_description/mujoco/wojtek_mjx.xml).
_STOCK_CTRL_LOW = np.array([-3.1416, -3.14, 0.425] * 4)
_STOCK_CTRL_HIGH = np.array([3.1416, 3.14, 5.8] * 4)


def for_robot(training_robot: str, folded_knee_rad: float = None) -> RobotPoses:
    """The poses of one set of legs, by training variant name.

    `training_robot` is RobotProfile.training_robot (robots.py). The stock
    legs take an optional folded knee angle, the one real_io_node exposes as
    a parameter; the other legs have no such knob.
    """
    if training_robot == "wojtek":
        knee = FOLDED_KNEE_RAD if folded_knee_rad is None else folded_knee_rad
        return RobotPoses(
            training_robot="wojtek",
            home_ctrl=HOME_CTRL.copy(),
            folded_ctrl=folded_ctrl(ACTUATOR_NAMES, knee),
            ctrl_low=_STOCK_CTRL_LOW.copy(),
            ctrl_high=_STOCK_CTRL_HIGH.copy(),
            passive_from_driver=dict(PASSIVE_FROM_KNEE),
            passive_driver="third_joint",
        )
    if training_robot == "legs_v627":
        return RobotPoses(
            training_robot="legs_v627",
            home_ctrl=V627_HOME_CTRL.copy(),
            folded_ctrl=V627_FOLDED_CTRL.copy(),
            ctrl_low=V627_CTRL_LOW.copy(),
            ctrl_high=V627_CTRL_HIGH.copy(),
            passive_from_driver=dict(V627_PASSIVE_FROM_LEVER),
            passive_driver="third_joint",
            passive_constant=dict(V627_PASSIVE_CONSTANT),
            passive_convention="mjc",
        )
    raise ValueError(
        f"no poses for training robot {training_robot!r} "
        "(known: 'wojtek', 'legs_v627')"
    )
