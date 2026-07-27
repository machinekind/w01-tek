"""Mirror-map properties, model-free (wojtek_rl.symmetry only)."""

import numpy as np
import pytest

from wojtek_rl import symmetry

# The two obs lists actually trained today: the stiff keeper's filtered
# actor list, and the full joystick state/privileged defaults.
STIFF_ACTOR = ["joint_pos", "joint_vel", "last_act", "command"]
FULL_STATE = [
    "gyro", "gravity", "joint_pos", "joint_vel", "last_act", "command",
    "phase",
]
PRIVILEGED = FULL_STATE + ["linvel", "foot_contact"]


def _apply(perm, sign, x):
    return sign * x[perm]


@pytest.mark.parametrize("names", [STIFF_ACTOR, FULL_STATE, PRIVILEGED])
def test_obs_mirror_is_involution(names):
    perm, sign = symmetry.obs_mirror(names)
    n = sum(symmetry.COMPONENT_SIZES[m] for m in names)
    assert perm.shape == sign.shape == (n,)
    # perm is a permutation, signs are unit
    assert sorted(perm.tolist()) == list(range(n))
    assert set(np.abs(sign).tolist()) == {1.0}
    # applying the mirror twice is the identity
    x = np.random.default_rng(0).normal(size=n)
    assert np.allclose(_apply(perm, sign, _apply(perm, sign, x)), x)


def test_action_mirror_swaps_legs_and_flips_abduction():
    perm, sign = symmetry.joint_mirror()
    a = np.arange(12.0)
    m = sign * a[perm]
    # rear_left (0,1,2) <-> rear_right (3,4,5): abduction negates,
    # second/third carry over unchanged
    assert m[0] == -a[3] and m[1] == a[4] and m[2] == a[5]
    assert m[3] == -a[0] and m[4] == a[1] and m[5] == a[2]
    # front_right (6,7,8) <-> front_left (9,10,11)
    assert m[6] == -a[9] and m[7] == a[10] and m[8] == a[11]
    assert np.allclose(sign * (sign * a[perm])[perm], a)


def test_command_mirror_flips_vy_and_wz_only():
    perm, sign = symmetry.component_mirror("command")
    cmd = np.array([0.4, 0.2, -1.0, 0.125])
    assert np.allclose(sign * cmd[perm], [0.4, -0.2, 1.0, 0.125])


def test_gyro_and_gravity_mirror_as_pseudo_and_true_vectors():
    perm, sign = symmetry.component_mirror("gyro")
    assert np.allclose(sign * np.array([1.0, 2.0, 3.0])[perm], [-1.0, 2.0, -3.0])
    perm, sign = symmetry.component_mirror("gravity")
    assert np.allclose(sign * np.array([1.0, 2.0, 3.0])[perm], [1.0, -2.0, 3.0])


def test_phase_mirror_swaps_leg_pairs_in_both_halves():
    perm, sign = symmetry.component_mirror("phase")
    x = np.arange(8.0)
    m = sign * x[perm]
    # cos block legs (RL,RR,FR,FL) -> (RR,RL,FL,FR); same for sin block
    assert m.tolist() == [1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0]


def test_unknown_component_raises():
    with pytest.raises(KeyError):
        symmetry.component_mirror("no_such_obs")
