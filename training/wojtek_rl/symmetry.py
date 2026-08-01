"""Left/right mirror maps for Wojtek's observation and action spaces.

Mirroring is about the body xz-plane (y -> -y). Legs swap laterally
(rear_left <-> rear_right, front_right <-> front_left, paths.LEGS order).
Joint signs follow the model's axes in wojtek.xml: the abduction (first)
joints use axis (1,0,0) on BOTH sides, so a mirrored abduction angle
negates; the second/third joints use opposite z-axes on left vs right
legs (the builder already flipped them), so their mirrored angles keep
their sign. Vectors mirror as (x,-y,z); pseudo-vectors (angular rates)
as (-x,y,-z).

Everything here is plain numpy on names and static shapes - no model, no
env - so tests/unit can cover it. The env validates the assembled maps
against its real observation sizes at construction.
"""

import numpy as np

from wojtek_rl import height_scan

# paths.LEGS order (rear_left, rear_right, front_right, front_left) -> the
# laterally mirrored leg's index.
LEG_MIRROR = (1, 0, 3, 2)

# Per-leg actuated joints (first/abduction, second, third): sign of the
# mirrored angle, target, velocity or torque.
JOINT_SIGN = (-1.0, 1.0, 1.0)

# Observation catalog sizes (WojtekEnv._obs_catalog + the joystick env's
# command/phase additions). Must match the env's component shapes; the env
# asserts the assembled vector length at construction.
COMPONENT_SIZES = {
    "gyro": 3,
    "gravity": 3,
    "joint_pos": 12,
    "joint_vel": 12,
    "last_act": 12,
    "command": 4,
    "phase": 8,
    "linvel": 3,
    "base_height": 1,
    "actuator_force": 12,
    "foot_contact": 4,
    "height_scan": height_scan.SIZE,
    "height_scan_clean": height_scan.SIZE,
}


def joint_mirror():
    """(perm, sign) for a 12-vector in actuator order (LEGS-major)."""
    perm = np.array([leg * 3 + j for leg in LEG_MIRROR for j in range(3)])
    sign = np.array(JOINT_SIGN * 4)
    return perm, sign


def _leg_vector_mirror():
    """(perm, sign) for a per-leg 4-vector (contacts, air time, ...)."""
    return np.array(LEG_MIRROR), np.ones(4)


def _phase_mirror():
    """(perm, sign) for the 8-vector phase obs: cos(4 legs) ++ sin(4 legs).

    Swapping legs swaps their clock offsets; the master clock itself is
    unchanged, so the mirrored gait is the same schedule with left/right
    exchanged (trot stays a trot, the 4-beat walk reverses its lateral
    sequence).
    """
    leg = np.array(LEG_MIRROR)
    return np.concatenate([leg, leg + 4]), np.ones(8)


def component_mirror(name):
    """(perm, sign) arrays mirroring one observation catalog component."""
    jperm, jsign = joint_mirror()
    # y -> -y reverses each grid row's y index; heights are unsigned.
    sperm, ssign = height_scan.mirror_map()
    table = {
        # angular rate: pseudo-vector
        "gyro": (np.arange(3), np.array([-1.0, 1.0, -1.0])),
        # body-frame direction: vector
        "gravity": (np.arange(3), np.array([1.0, -1.0, 1.0])),
        "joint_pos": (jperm, jsign),
        "joint_vel": (jperm, jsign),
        "last_act": (jperm, jsign),
        # (vx, vy, wz, height)
        "command": (np.arange(4), np.array([1.0, -1.0, -1.0, 1.0])),
        "phase": _phase_mirror(),
        "linvel": (np.arange(3), np.array([1.0, -1.0, 1.0])),
        "base_height": (np.arange(1), np.ones(1)),
        "actuator_force": (jperm, jsign),
        "foot_contact": _leg_vector_mirror(),
        "height_scan": (sperm, ssign),
        "height_scan_clean": (sperm, ssign),
    }
    if name not in table:
        raise KeyError(
            f"no mirror map for obs component {name!r}; add it to "
            f"wojtek_rl.symmetry before training with symmetry on"
        )
    return table[name]


def obs_mirror(names):
    """(perm, sign) for a concatenated observation vector.

    `names` is the ordered component list (obs.state after the include
    filter, or obs.privileged). Applying `sign * obs[perm]` yields the
    observation the mirrored world would produce.
    """
    perms, signs, offset = [], [], 0
    for name in names:
        size = COMPONENT_SIZES[name]
        perm, sign = component_mirror(name)
        if len(perm) != size or len(sign) != size:
            raise ValueError(f"mirror map for {name!r} does not match size {size}")
        perms.append(np.asarray(perm) + offset)
        signs.append(np.asarray(sign, dtype=np.float32))
        offset += size
    return np.concatenate(perms), np.concatenate(signs)
