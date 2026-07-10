import jax
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from wojtek_rl import paths
from wojtek_rl.randomize import make_domain_randomize

GOLDEN_FIELDS = (
    "geom_friction",
    "body_mass",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
)

ALL_DISABLED = {
    "com_offset": {"enable": False},
    "joint_gains": {"enable": False},
    "dof": {"enable": False},
    "foot_friction": {"enable": False},
}


@pytest.fixture(scope="module")
def mj_model():
    return mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))


@pytest.fixture(scope="module")
def mjx_model(mj_model):
    return mjx.put_model(mj_model, impl="jax")


@pytest.fixture(scope="module")
def golden():
    return np.load(paths.PROJECT_DIR / "tests" / "data" / "randomize_golden.npz")


@pytest.fixture
def rng():
    return jax.random.split(jax.random.PRNGKey(0), 8)


def _assert_matches_golden(model_v, golden):
    assert int(golden["n_envs"]) == 8
    for field in GOLDEN_FIELDS:
        np.testing.assert_array_equal(np.array(getattr(model_v, field)), golden[field])


def test_golden_bitwise_default_dr_cfg(mj_model, mjx_model, rng, golden):
    randomize = make_domain_randomize(mj_model)
    model_v, _ = randomize(mjx_model, rng)
    _assert_matches_golden(model_v, golden)


def test_golden_bitwise_explicit_all_disabled(mj_model, mjx_model, rng, golden):
    randomize = make_domain_randomize(mj_model, ALL_DISABLED)
    model_v, _ = randomize(mjx_model, rng)
    _assert_matches_golden(model_v, golden)


def test_com_offset_varies_when_enabled(mj_model, mjx_model, rng):
    dr_cfg = {**ALL_DISABLED, "com_offset": {"enable": True, "xy": 0.02, "z": 0.01}}
    randomize = make_domain_randomize(mj_model, dr_cfg)
    model_v, in_axes = randomize(mjx_model, rng)

    root_id = mj_model.body("root").id
    ipos = np.array(model_v.body_ipos[:, root_id])
    assert not np.allclose(ipos[0], ipos[1])
    assert not np.allclose(ipos, 0.0)
    assert np.all(np.abs(ipos[:, 0]) <= 0.02 + 1e-8)
    assert np.all(np.abs(ipos[:, 1]) <= 0.02 + 1e-8)
    assert np.all(np.abs(ipos[:, 2]) <= 0.01 + 1e-8)
    assert in_axes.body_ipos == 0


def test_dof_fields_vary_when_enabled(mj_model, mjx_model, rng):
    dr_cfg = {**ALL_DISABLED, "dof": {"enable": True}}
    randomize = make_domain_randomize(mj_model, dr_cfg)
    model_v, in_axes = randomize(mjx_model, rng)

    baseline = mjx_model.dof_damping
    for field in ("dof_damping", "dof_armature", "dof_frictionloss"):
        arr = np.array(getattr(model_v, field))
        assert not np.allclose(arr[0], arr[1])
        assert getattr(in_axes, field) == 0
    assert not np.allclose(np.array(model_v.dof_damping[0]), np.array(baseline))


def test_foot_friction_varies_when_enabled(mj_model, mjx_model, rng):
    dr_cfg = {**ALL_DISABLED, "foot_friction": {"enable": True, "range": [0.8, 1.2]}}
    randomize = make_domain_randomize(mj_model, dr_cfg)
    model_v, in_axes = randomize(mjx_model, rng)

    foot_ids = [mj_model.geom(f"{leg}_foot_sphere").id for leg in paths.LEGS]
    friction = np.array(model_v.geom_friction[:, foot_ids, 0])
    assert not np.allclose(friction[0], friction[1])
    baseline = np.array(mjx_model.geom_friction[foot_ids, 0])
    assert not np.allclose(friction[0], baseline)
    assert in_axes.geom_friction == 0


def test_joint_gains_independent_when_enabled(mj_model, mjx_model, rng):
    dr_cfg = {
        **ALL_DISABLED,
        "joint_gains": {"enable": True, "gain_pct": 0.2, "kd_pct": 0.2},
    }
    randomize = make_domain_randomize(mj_model, dr_cfg)
    model_v, in_axes = randomize(mjx_model, rng)

    gain_col = np.array(model_v.actuator_gainprm[0, :, 0])
    kd_col = np.array(model_v.actuator_biasprm[0, :, 2])
    # Pre-expansion behavior was a single scalar broadcast to all 12 joints;
    # independent per-joint draws must not all collapse to the same value.
    assert not np.allclose(gain_col, gain_col[0])
    assert not np.allclose(kd_col, kd_col[0])
    assert np.all(gain_col >= (1 - 0.2) * 20.0 - 1e-6)  # kp=20, see build_model.py
    assert in_axes.actuator_gainprm == 0


def test_joint_gains_disabled_matches_single_scalar_broadcast(mj_model, mjx_model, rng):
    dr_cfg = {**ALL_DISABLED, "joint_gains": {"enable": False}}
    randomize = make_domain_randomize(mj_model, dr_cfg)
    model_v, _ = randomize(mjx_model, rng)

    gain_col = np.array(model_v.actuator_gainprm[0, :, 0])
    # Disabled path broadcasts one scalar to all 12 joints -> all equal.
    assert np.allclose(gain_col, gain_col[0])


def test_in_axes_marks_only_randomized_fields(mj_model, mjx_model, rng):
    randomize = make_domain_randomize(mj_model)
    model_v, in_axes = randomize(mjx_model, rng)
    assert in_axes.qpos0 is None
    assert in_axes.dof_damping is None
    assert in_axes.body_ipos is None
    assert in_axes.geom_friction == 0
    assert in_axes.body_mass == 0
    assert in_axes.actuator_gainprm == 0
    assert in_axes.actuator_biasprm == 0
    assert in_axes.actuator_forcerange == 0


def test_all_fields_enabled_in_axes(mj_model, mjx_model, rng):
    dr_cfg = {
        "com_offset": {"enable": True},
        "joint_gains": {"enable": True},
        "dof": {"enable": True},
        "foot_friction": {"enable": True},
    }
    randomize = make_domain_randomize(mj_model, dr_cfg)
    _, in_axes = randomize(mjx_model, rng)
    for field in (
        "geom_friction",
        "body_mass",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "body_ipos",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
    ):
        assert getattr(in_axes, field) == 0
    assert in_axes.qpos0 is None
