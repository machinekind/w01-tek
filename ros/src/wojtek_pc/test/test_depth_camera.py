"""Renderer tests for the simulated D435 (wojtek_pc.depth_camera).

These need mujoco AND a working offscreen GL backend; they skip cleanly
where either is missing. Run for real inside the dev container:

    docker exec wojtek_robot python3 -m pytest /ros2_ws/src/wojtek_pc/test -q

The wall test is the acceptance test the issue asks for: a box at a known
distance must read back within tolerance, with the expected value computed
from the MODEL's camera pose (cam_xpos/cam_xmat) -- which is exactly the
"renders from the robot's head pose" criterion at unit level.
"""

import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

mujoco = pytest.importorskip("mujoco")

from wojtek_pc import camera_spec  # noqa: E402
from wojtek_pc.depth_camera import SimDepthCamera, inject_camera  # noqa: E402

REPO_ROS_SRC = PKG.parent  # .../ros/src


@pytest.fixture(scope="module")
def scene_xml(tmp_path_factory):
    """The sim's scene with meshdir rewritten to the source-tree meshes
    (mirrors mujoco_sim_node._prepare_model_xml, which rewrites to the
    installed share)."""
    src = PKG / "config"
    meshes = REPO_ROS_SRC / "wojtek_description" / "meshes"
    tmp = tmp_path_factory.mktemp("wojtek_mj")
    robot = (src / "wojtek_mjx.xml").read_text()
    robot = re.sub(r'meshdir="[^"]*"', f'meshdir="{meshes}"', robot)
    (tmp / "wojtek_mjx.xml").write_text(robot)
    shutil.copy(src / "scene_mjx.xml", tmp / "scene_mjx.xml")
    return str(tmp / "scene_mjx.xml")


def _camera_or_skip(model, **kwargs):
    try:
        cam = SimDepthCamera(model, **kwargs)
        cam.start()
    except Exception as e:  # no EGL/OSMesa on this host
        pytest.skip(f"no offscreen GL backend: {e}")
    return cam


def _home_state(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return data


def _model_with_wall(scene_xml, wall_x):
    spec = mujoco.MjSpec.from_file(scene_xml)
    inject_camera(spec)
    body = spec.worldbody.add_body()
    body.name = "test_wall"
    body.pos = [wall_x, 0.0, 1.0]
    geom = body.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [0.02, 2.0, 2.0]
    return spec.compile()


# Probe pixel for the wall tests: on the vertical centreline but ABOVE the
# principal point. The head-mounted camera sits ~0.2 m over the floor pitched
# 15 deg down, so the principal ray hits the FLOOR well inside the valid
# window; this row's ray runs slightly above horizontal instead -- it can
# only hit the wall (or nothing).
PROBE_ROW, PROBE_COL = 60, camera_spec.DEPTH_WIDTH // 2


def _expected_wall_depth(model, data, cam_id, wall_x, row=PROBE_ROW,
                         col=PROBE_COL):
    """Predicted 16UC1 depth at (row, col) for the plane x = wall_x, from
    the MODEL's own camera pose. MuJoCo's depth is distance along the
    optical axis (z-depth), so for pixel direction (u, v, 1) in the optical
    frame the wall is met at z = (wall_x - cam_x) / (u, v, 1)@axes_x."""
    fx, fy, cx, cy = camera_spec.intrinsics(
        camera_spec.DEPTH_WIDTH, camera_spec.DEPTH_HEIGHT
    )
    u, v = (col - cx) / fx, (row - cy) / fy
    r = data.cam_xmat[cam_id].reshape(3, 3)
    x_right, y_down, z_fwd = r[:, 0], -r[:, 1], -r[:, 2]
    denom = u * x_right[0] + v * y_down[0] + z_fwd[0]
    face = wall_x - 0.02  # the box's near face
    return (face - data.cam_xpos[cam_id][0]) / denom


class TestInjection:
    def test_camera_exists_with_spec_constants(self, scene_xml):
        spec = mujoco.MjSpec.from_file(scene_xml)
        inject_camera(spec)
        model = spec.compile()
        cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_spec.CAMERA_NAME
        )
        assert cam_id >= 0
        assert model.cam_fovy[cam_id] == pytest.approx(camera_spec.FOVY_DEG)
        assert model.cam_pos[cam_id] == pytest.approx(
            np.array(camera_spec.MOUNT_XYZ)
        )

    def test_camera_rides_the_free_base(self, scene_xml):
        spec = mujoco.MjSpec.from_file(scene_xml)
        inject_camera(spec)
        model = spec.compile()
        cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_spec.CAMERA_NAME
        )
        body_id = model.cam_bodyid[cam_id]
        # The owning body must be the free-jointed base, not the world.
        jnt = model.body_jntadr[body_id]
        assert jnt >= 0
        assert model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE

    def test_missing_camera_raises_loudly(self, scene_xml):
        model = mujoco.MjModel.from_xml_path(scene_xml)
        with pytest.raises(ValueError, match=camera_spec.CAMERA_NAME):
            SimDepthCamera(model)


class TestRender:
    def test_shape_and_encoding(self, scene_xml):
        model = _model_with_wall(scene_xml, wall_x=1.5)
        cam = _camera_or_skip(model)
        try:
            data = _home_state(model)
            depth = cam.render_depth(data.qpos, data.qvel)
            assert depth.shape == (camera_spec.DEPTH_HEIGHT,
                                   camera_spec.DEPTH_WIDTH)
            assert depth.dtype == np.uint16
            rgb = cam.render_color(data.qpos, data.qvel)
            assert rgb.shape == (camera_spec.COLOR_HEIGHT,
                                 camera_spec.COLOR_WIDTH, 3)
            assert rgb.dtype == np.uint8
        finally:
            cam.close()

    def test_wall_at_known_distance(self, scene_xml):
        wall_x = 1.5
        model = _model_with_wall(scene_xml, wall_x)
        cam = _camera_or_skip(model)
        try:
            data = _home_state(model)
            expected_m = _expected_wall_depth(model, data, cam.cam_id, wall_x)
            assert 0.5 < expected_m < 2.5  # sanity: inside the valid window
            depth = cam.render_depth(data.qpos, data.qvel)
            assert depth[PROBE_ROW, PROBE_COL] == pytest.approx(
                expected_m * 1000, abs=10
            )
        finally:
            cam.close()

    def test_no_return_is_zero(self, scene_xml):
        # No wall: the top rows look over the floor into the skybox, whose
        # z-buffer value is the far plane (~40 m) -- must come back as 0.
        spec = mujoco.MjSpec.from_file(scene_xml)
        inject_camera(spec)
        model = spec.compile()
        cam = _camera_or_skip(model)
        try:
            data = _home_state(model)
            depth = cam.render_depth(data.qpos, data.qvel)
            assert (depth[0, :] == 0).all()
        finally:
            cam.close()

    def test_out_of_window_clipping(self, scene_xml):
        # Wall beyond max range must read 0. Sampled on the TOP row: with
        # the 15 deg pitch and ~30 deg half-VFOV the top rays run near
        # horizontal, so unlike the principal ray they reach the far wall
        # instead of hitting the floor inside the valid window first.
        model = _model_with_wall(scene_xml, wall_x=5.0)
        cam = _camera_or_skip(model)
        try:
            data = _home_state(model)
            depth = cam.render_depth(data.qpos, data.qvel)
            assert depth[0, camera_spec.DEPTH_WIDTH // 2] == 0
        finally:
            cam.close()

    def test_camera_follows_the_base(self, scene_xml):
        # Move the base back: the wall must recede by the model-predicted
        # amount, with the expectation recomputed from cam_xpos/cam_xmat at
        # each pose.
        wall_x, shift = 1.0, 0.2
        model = _model_with_wall(scene_xml, wall_x)
        cam = _camera_or_skip(model)
        try:
            base = _home_state(model)
            probe = mujoco.MjData(model)
            readings, expected = [], []
            for dx in (0.0, -shift):
                qpos = base.qpos.copy()
                qpos[0] += dx
                probe.qpos[:] = qpos
                mujoco.mj_forward(model, probe)
                expected_m = _expected_wall_depth(
                    model, probe, cam.cam_id, wall_x
                )
                depth = cam.render_depth(qpos, base.qvel)
                assert depth[PROBE_ROW, PROBE_COL] == pytest.approx(
                    expected_m * 1000, abs=10
                )
                readings.append(int(depth[PROBE_ROW, PROBE_COL]))
                expected.append(expected_m)
            assert readings[1] - readings[0] == pytest.approx(
                (expected[1] - expected[0]) * 1000, abs=15
            )
        finally:
            cam.close()
