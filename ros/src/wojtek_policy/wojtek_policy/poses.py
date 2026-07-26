"""Named robot poses shared by the real bringup and the MuJoCo sim.

Poses are expressed in the policy/MuJoCo actuator convention; convert with
wojtek_policy.joint_map.JointMap where URDF angles are needed. These are
robot constants (the MJCF home keyframe and actuator order) -- independent
of whichever policy is loaded.
"""

import numpy as np

# Canonical actuated-joint order: the MJCF actuator order every policy,
# joint-target message, and pose in this stack uses. A property of the
# robot model, not of any one policy; policy_node cross-checks a loaded
# policy's actuator_names against this list.
ACTUATOR_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("rear_left", "rear_right", "front_right", "front_left")
    for joint in ("first", "second", "third")
]

# Standing home pose (the MJCF "home" keyframe ctrl), policy/MuJoCo
# convention: abduction 0, hip -0.2, knee 3.1 on every leg.
HOME_CTRL = np.array([0.0, -0.2, 3.1] * 4)

# Knee (third joint) angle against the folding mechanical stop, policy/MuJoCo
# convention. This is the model ctrlrange lower bound: the near-branch end of
# the four-bar range, i.e. the lower leg folded flat against the upper leg.
FOLDED_KNEE_RAD = 0.425


def folded_ctrl(joint_names, folded_knee_rad=FOLDED_KNEE_RAD):
    """Boot pose: hips straight, knees folded on the mechanical stop.

    Reproducible by hand, which makes it the reference pose for zeroing the
    MD80 encoders at power-on.
    """
    return np.array(
        [
            folded_knee_rad if n.endswith("third_joint") else 0.0
            for n in joint_names
        ]
    )


# Passive four-bar joints (fourth/fifth per leg) as a function of that leg's
# knee (third) joint, both in URDF convention:
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
