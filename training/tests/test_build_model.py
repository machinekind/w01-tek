import mujoco
import numpy as np

from wojtek_rl import build_model, paths


def _compiled():
    return build_model.build_spec().compile()


def test_actuator_and_constraint_counts():
    m = _compiled()
    assert m.nu == 12
    assert m.neq == 4
    assert m.nq == 31
    assert m.nv == 30


def test_total_mass_is_parameter():
    m = _compiled()
    assert np.isclose(m.body_mass.sum(), build_model.DEFAULT_TOTAL_MASS, atol=0.05)
    m2 = build_model.build_spec(total_mass=8.0).compile()
    assert np.isclose(m2.body_mass.sum(), 8.0, atol=0.05)


def test_actuators_are_pd_with_forcerange():
    m = _compiled()
    for i in range(m.nu):
        assert m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE
        assert m.actuator_gainprm[i, 0] == build_model.DEFAULT_KP
        assert m.actuator_biasprm[i, 1] == -build_model.DEFAULT_KP
        assert m.actuator_biasprm[i, 2] == -build_model.DEFAULT_KD
        assert np.allclose(
            m.actuator_forcerange[i],
            [-build_model.FORCERANGE, build_model.FORCERANGE],
        )


def test_collision_geoms_are_primitives_only():
    # 4 foot spheres + base box + 4 legs x 4 floor-only leg primitives.
    m = _compiled()
    active = [i for i in range(m.ngeom) if m.geom_contype[i] != 0]
    assert len(active) == 21
    assert all(
        m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH for i in active
    )
    feet = [i for i in active if m.geom_contype[i] == 1 and m.geom_conaffinity[i] == 1]
    assert len(feet) == 4  # feet pair with floor + each other, not the legs
    leg = [i for i in active if m.geom_contype[i] == 2]
    assert len(leg) == 16
    assert all(m.geom_conaffinity[i] == 0 for i in leg)  # floor-only


def test_timestep():
    m = _compiled()
    assert np.isclose(m.opt.timestep, 0.004)


def test_ego_camera():
    m = _compiled()
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
    assert cam_id >= 0
    assert m.cam_bodyid[cam_id] == m.body("root").id
    assert np.isclose(m.cam_fovy[cam_id], build_model.EGO_CAM["fovy"])
    # Optical axis (-z column of the camera frame) points along body +x,
    # pitched slightly down.
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, m.cam_quat[cam_id])
    optical_axis = -mat.reshape(3, 3)[:, 2]
    assert optical_axis @ np.array([1.0, 0.0, 0.0]) > 0.9
    assert optical_axis[2] < 0.0
