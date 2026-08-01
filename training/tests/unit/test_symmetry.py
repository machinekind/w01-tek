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
# The terrain actor list with the height scan appended: 46 + 25 dims.
SCAN_ACTOR = [
    "gyro", "gravity", "joint_pos", "joint_vel", "last_act", "command",
    "height_scan",
]


def _apply(perm, sign, x):
    return sign * x[perm]


@pytest.mark.parametrize(
    "names", [STIFF_ACTOR, FULL_STATE, PRIVILEGED, SCAN_ACTOR]
)
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


# -- the terrain family's frozen actor layout ------------------------------

# terrain_blind_v3's obs.include applied to the joystick state list: the
# kp40 joint/action/command channels plus the IMU, 46 dims. Frozen for the
# whole terrain family (see the preset header).
TERRAIN_ACTOR = ["gyro", "gravity", "joint_pos", "joint_vel", "last_act", "command"]


def test_terrain_actor_layout_mirrors_every_block():
    """Pin the assembled 46-dim map to hand-written expectations, so an
    offset slip (component reorder, size change) cannot pass silently."""
    perm, sign = symmetry.obs_mirror(TERRAIN_ACTOR)
    assert perm.shape == sign.shape == (46,)

    gyro = np.array([1.0, 2.0, 3.0])
    grav = np.array([4.0, 5.0, 6.0])
    jpos = 10.0 + np.arange(12.0)
    jvel = 30.0 + np.arange(12.0)
    lact = 50.0 + np.arange(12.0)
    cmd = np.array([70.0, 71.0, 72.0, 73.0])
    m = sign * np.concatenate([gyro, grav, jpos, jvel, lact, cmd])[perm]

    np.testing.assert_allclose(m[0:3], [-1.0, 2.0, -3.0])  # gyro: pseudo-vector
    np.testing.assert_allclose(m[3:6], [4.0, -5.0, 6.0])  # gravity: vector
    # joint blocks: rear pair and front pair swap, abduction negates
    for base, blk in ((6, jpos), (18, jvel), (30, lact)):
        rl, rr, fr, fl = blk[0:3], blk[3:6], blk[6:9], blk[9:12]
        expected = np.concatenate([
            [-rr[0], rr[1], rr[2]], [-rl[0], rl[1], rl[2]],
            [-fl[0], fl[1], fl[2]], [-fr[0], fr[1], fr[2]],
        ])
        np.testing.assert_allclose(m[base : base + 12], expected)
    np.testing.assert_allclose(m[42:46], [70.0, -71.0, -72.0, 73.0])  # command


def test_scan_block_reverses_its_y_index_after_the_frozen_actor_layout():
    perm, sign = symmetry.obs_mirror(SCAN_ACTOR)
    assert perm.shape == sign.shape == (71,)
    scan = np.arange(25.0).reshape(5, 5)  # [ix, iy]
    x = np.concatenate([np.zeros(46), scan.reshape(-1)])
    m = (sign * x[perm])[46:71].reshape(5, 5)
    np.testing.assert_array_equal(m, scan[:, ::-1])
    np.testing.assert_array_equal(sign[46:71], np.ones(25))


# -- what the augmentation can and cannot train ----------------------------


def _signed_perm(perm, sign):
    n = len(perm)
    mat = np.zeros((n, n))
    mat[np.arange(n), perm] = sign
    return mat


def test_fixed_flag_world_mirror_is_invisible_to_the_policy():
    """The terrain_blind_v3 post-mortem, as an algebraic fact.

    The env's augmentation presents sigma(obs) and un-mirrors the action
    before a mirror-equivariant physics step, with the flag fixed per env
    (auto-reset never re-runs reset). For ANY policy -- however asymmetric
    -- the policy-frame (obs, action, reward) stream of a mirrored env is
    then IDENTICAL to a plain env's: the flag is unobservable, so PPO gets
    zero gradient toward pi(sigma s) = sigma pi(s). World mirroring cancels
    chirality of the WORLD; it cannot symmetrize the POLICY.

    That is how wojtek_terrain_blind_v3_20260728 trained with
    symmetry.enable=true and provably correct maps (mirror-wrapping the
    checkpoint swaps spin_left/spin_right scores exactly: -33/+124 deg to
    +127/-35) and still cannot turn one way. A fix must couple the two
    frames inside the LEARNER -- mirrored-transition duplication in the
    PPO batch, a symmetry loss, or per-step flag resampling -- and would
    make the streams below differ for an asymmetric policy.
    """
    perm, sign = symmetry.obs_mirror(TERRAIN_ACTOR)
    mo = _signed_perm(perm, sign)  # obs mirror
    ap, asn = symmetry.joint_mirror()
    ma = _signed_perm(ap, asn)  # action mirror
    n = len(perm)
    rng = np.random.default_rng(1)

    # Linear stand-in plant, mirror-equivariant by symmetrization -- the
    # property tests/integration/test_symmetry.py verifies for real MJX.
    a0 = rng.normal(size=(n, n)) / (2 * n)
    b0 = rng.normal(size=(n, 12)) / (2 * n)
    plant_a = (a0 + mo @ a0 @ mo) / 2.0
    plant_b = (b0 + mo @ b0 @ ma) / 2.0

    w = rng.normal(size=(12, n))

    def pi(obs):
        return np.tanh(w @ obs)

    x0 = rng.normal(size=n)
    # a random policy is genuinely asymmetric in its own frame
    assert np.abs(pi(mo @ x0) - ma @ pi(x0)).max() > 0.05

    def stream(mirrored):
        x = mo @ x0 if mirrored else x0.copy()
        out = []
        for _ in range(50):
            obs = mo @ x if mirrored else x
            act = pi(obs)  # what the learner records
            u = ma @ act if mirrored else act  # what the plant receives
            reward = x @ x + u @ u  # any sigma-invariant reward
            out.append((obs, act, reward))
            x = plant_a @ x + plant_b @ u
        return out

    for (o_p, a_p, r_p), (o_m, a_m, r_m) in zip(stream(False), stream(True)):
        np.testing.assert_allclose(o_m, o_p, atol=1e-12)
        np.testing.assert_allclose(a_m, a_p, atol=1e-12)
        np.testing.assert_allclose(r_m, r_p, atol=1e-12)
