"""Seam-switcher routing, model-free (wojtek_rl.seam only)."""

import jax.numpy as jp
import numpy as np
import pytest

from wojtek_rl import seam
from wojtek_rl import symmetry

# The terrain family's frozen actor layout (v4.2/v4.5/v4.6).
FAMILY_ACTOR = ["gyro", "gravity", "joint_pos", "joint_vel", "last_act", "command"]


def test_command_slice_family_layout():
    s = seam.command_slice(FAMILY_ACTOR)
    assert s == slice(42, 46)
    total = sum(symmetry.COMPONENT_SIZES[n] for n in FAMILY_ACTOR)
    assert total == 46


def test_command_slice_requires_command():
    with pytest.raises(ValueError):
        seam.command_slice(["gyro", "joint_pos"])


def _obs(cmd, n=46):
    state = np.zeros(n, np.float32)
    state[42:46] = cmd
    return {"state": jp.array(state), "privileged_state": jp.zeros(3)}


def _fixed(act):
    act = jp.array(act, jp.float32)

    def inf(obs, key):
        state = obs["state"]
        shape = state.shape[:-1] + act.shape
        return jp.broadcast_to(act, shape), {"who": float(act[0])}

    return inf


BASE = _fixed(np.full(12, 1.0))
SPIN = _fixed(np.full(12, 2.0))
CMD = seam.command_slice(FAMILY_ACTOR)


@pytest.mark.parametrize(
    "cmd,expect",
    [
        ([0.0, 0.0, 0.8, 0.15], 2.0),   # pure spin window
        ([0.0, 0.0, 0.0, 0.15], 2.0),   # stand (any height)
        ([0.4, 0.0, 0.0, 0.15], 1.0),   # walk
        ([0.0, 0.3, 0.0, 0.15], 1.0),   # strafe
        ([0.4, 0.0, 0.6, 0.15], 1.0),   # arc: wz with vx stays base
        ([1e-7, 0.0, 0.8, 0.15], 2.0),  # float dust still counts as zero
        ([0.01, 0.0, 0.8, 0.15], 1.0),  # a real command does not
    ],
)
def test_window_routing(cmd, expect):
    inf = seam.switch_inference(BASE, SPIN, CMD)
    act, _ = inf(_obs(cmd), None)
    assert act.shape == (12,)
    np.testing.assert_allclose(np.asarray(act), expect)


def test_batched_routing_is_per_row():
    inf = seam.switch_inference(BASE, SPIN, CMD)
    rows = np.stack(
        [
            np.asarray(_obs([0.0, 0.0, 0.8, 0.15])["state"]),
            np.asarray(_obs([0.5, 0.0, 0.0, 0.15])["state"]),
            np.asarray(_obs([0.0, 0.0, 0.0, 0.10])["state"]),
        ]
    )
    obs = {"state": jp.array(rows), "privileged_state": jp.zeros((3, 3))}
    act, _ = inf(obs, None)
    assert act.shape == (3, 12)
    np.testing.assert_allclose(np.asarray(act)[:, 0], [2.0, 1.0, 2.0])


def test_extras_come_from_base():
    inf = seam.switch_inference(BASE, SPIN, CMD)
    _, extras = inf(_obs([0.0, 0.0, 0.8, 0.15]), None)
    assert extras == {"who": 1.0}


def test_check_compatible_tuple_vs_list():
    base = {"env_config": {"obs": {"include": FAMILY_ACTOR},
                           "action_scale": (0.25, 0.5, 1.35)}}
    other = {"env_config": {"obs": {"include": list(FAMILY_ACTOR)},
                            "action_scale": [0.25, 0.5, 1.35]}}
    seam.check_compatible(base, other)


def test_check_compatible():
    base = {"env_config": {"obs": {"include": FAMILY_ACTOR}, "pd_kp": 40.0,
                           "action_scale": [0.25, 0.5, 1.35]}}
    good = {"env_config": {"obs": {"include": list(FAMILY_ACTOR)}, "pd_kp": 40.0,
                           "action_scale": [0.25, 0.5, 1.35]}}
    seam.check_compatible(base, good)
    with pytest.raises(ValueError, match="obs layouts"):
        seam.check_compatible(
            base, {"env_config": {"obs": {"include": FAMILY_ACTOR[:-1]}}}
        )
    with pytest.raises(ValueError, match="action contracts"):
        seam.check_compatible(
            base,
            {"env_config": {"obs": {"include": FAMILY_ACTOR}, "pd_kp": 80.0,
                            "action_scale": [0.25, 0.5, 1.35]}},
        )
