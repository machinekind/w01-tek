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
    # 4 foot spheres + 9 base chessboard cells + 4 legs x 4 floor-only leg
    # primitives. This count is the naconmax_per_env floor's geom count.
    m = _compiled()
    active = [i for i in range(m.ngeom) if m.geom_contype[i] != 0]
    assert len(active) == 4 + len(build_model.BASE_BOX_NAMES) + 16
    assert all(
        m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH for i in active
    )
    feet = [i for i in active if m.geom_contype[i] == 1 and m.geom_conaffinity[i] == 1]
    assert len(feet) == 4  # feet pair with floor + each other, not the legs
    leg = [i for i in active if m.geom_contype[i] == 2]
    assert len(leg) == 16
    assert all(m.geom_conaffinity[i] == 0 for i in leg)  # floor-only


def test_base_box_is_a_chessboard_inside_the_original_footprint():
    """One geom per filled cell, so the MJWarp heightfield cap of 50
    contacts per geom pair applies to each small cell separately. Every
    cell stays inside the original box's footprint, corner cells reach its
    outer faces up to the anti-double-claim shrink, and the fill pattern is
    the chessboard (adjacent cells never both filled)."""
    m = _compiled()
    hx, hy, hz = build_model.BASE_BOX_HALFSIZE
    nx, ny = build_model.BASE_BOX_GRID
    shrink = build_model.BASE_BOX_CELL_SHRINK
    names = build_model.BASE_BOX_NAMES
    assert len(names) == (nx * ny + 1) // 2
    for name in names:
        g = m.geom(name)
        assert np.allclose(g.size, [hx / nx - shrink, hy / ny - shrink, hz])
        assert g.contype == 1 and g.conaffinity == 15
        assert abs(g.pos[0]) + g.size[0] <= hx + 1e-9
        assert abs(g.pos[1]) + g.size[1] <= hy + 1e-9
        # Bottom face flush with the belly plate (body z=0): centered at the
        # origin the box used to stick 5 cm below the mesh and a lying robot
        # hovered on it.
        assert np.isclose(g.pos[2] - g.size[2], 0.0)
    # Corner cell (row 0, col 0) touches both outer faces up to the shrink.
    corner = m.geom(names[0])
    assert np.isclose(corner.pos[0] - corner.size[0], -hx + shrink)
    assert np.isclose(corner.pos[1] - corner.size[1], -hy + shrink)
    # Chessboard: no two filled cells share a grid edge.
    filled = {
        (int(n.split("r")[1].split("c")[0]), int(n.split("c")[1])) for n in names
    }
    for row, col in filled:
        assert (row, col + 1) not in filled and (row + 1, col) not in filled


def test_base_inertia_stays_on_the_full_box():
    """The split is a collision detail and moves no mass. Feeding a half box
    into the inertia formula would change the dynamics of every run with
    nothing in the model to show for it."""
    m = _compiled()
    root = m.body("root")
    lx, ly, lz = (2 * s for s in build_model.BASE_BOX_HALFSIZE)
    mass = float(root.mass[0])
    assert np.allclose(
        root.inertia,
        [
            mass / 12 * (ly**2 + lz**2),
            mass / 12 * (lx**2 + lz**2),
            mass / 12 * (lx**2 + ly**2),
        ],
    )


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
