"""Unit tests for the D435 camera contract (wojtek_pc.camera_spec).

Pure math + numpy -- no ROS, no mujoco, no GL. Run anywhere:
    pytest ros/src/wojtek_pc/test/test_camera_spec.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wojtek_pc import camera_spec  # noqa: E402


class TestIntrinsics:
    def test_hfov_is_about_90_degrees(self):
        # The D435 depth stream is ~90 deg horizontal (fx 418 at 848x480).
        fx, _, _, _ = camera_spec.intrinsics(
            camera_spec.DEPTH_WIDTH, camera_spec.DEPTH_HEIGHT
        )
        hfov = math.degrees(2 * math.atan(camera_spec.DEPTH_WIDTH / (2 * fx)))
        assert abs(hfov - 90.0) < 2.0

    def test_fovy_matches_fy(self):
        # FOVY_DEG is derived from DEPTH_FY; inverting it must give fy back,
        # which is what guarantees the MJCF camera and CameraInfo agree.
        fy = 0.5 * camera_spec.DEPTH_HEIGHT / math.tan(
            math.radians(camera_spec.FOVY_DEG) / 2
        )
        assert fy == pytest.approx(camera_spec.DEPTH_FY, abs=1e-9)

    def test_principal_point_is_pixel_center(self):
        _, _, cx, cy = camera_spec.intrinsics(424, 240)
        assert cx == pytest.approx((424 - 1) / 2)
        assert cy == pytest.approx((240 - 1) / 2)

    def test_mjcf_optical_axis_matches_mount_pitch(self):
        # xyaxes: camera right x, camera up y; MJCF cameras look along -z.
        x = np.array(camera_spec.XYAXES[:3])
        y = np.array(camera_spec.XYAXES[3:])
        forward = -np.cross(x, y)
        pitch = camera_spec.MOUNT_PITCH_RAD
        expected = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
        assert forward == pytest.approx(expected, abs=1e-9)


class TestCameraInfoMsg:
    def _msg(self):
        pytest.importorskip("sensor_msgs")
        return camera_spec.camera_info_msg(
            camera_spec.DEPTH_WIDTH,
            camera_spec.DEPTH_HEIGHT,
            camera_spec.DEPTH_FRAME_ID,
        )

    def test_matrix_layout(self):
        msg = self._msg()
        fx, fy, cx, cy = camera_spec.intrinsics(424, 240)
        assert list(msg.k) == [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        assert list(msg.p) == [
            fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0,
        ]
        assert list(msg.r) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    def test_no_distortion(self):
        msg = self._msg()
        assert msg.distortion_model == "plumb_bob"
        assert list(msg.d) == [0.0] * 5

    def test_size_and_frame(self):
        msg = self._msg()
        assert (msg.width, msg.height) == (424, 240)
        assert msg.header.frame_id == "camera_depth_optical_frame"


class TestDepthToMm:
    def test_metres_to_millimetres_uint16(self):
        out = camera_spec.depth_to_mm(np.array([[1.234]]))
        assert out.dtype == np.uint16
        assert out[0, 0] == 1234

    def test_below_min_is_zero(self):
        assert camera_spec.depth_to_mm(np.array([[0.1]]))[0, 0] == 0

    def test_above_max_is_zero(self):
        # Includes the renderer's far-plane value for sky pixels (~40 m).
        assert camera_spec.depth_to_mm(np.array([[5.0]]))[0, 0] == 0
        assert camera_spec.depth_to_mm(np.array([[40.0]]))[0, 0] == 0

    def test_nonfinite_is_zero(self):
        out = camera_spec.depth_to_mm(np.array([[np.inf, np.nan, -np.inf]]))
        assert (out == 0).all()

    def test_window_bounds_inclusive(self):
        out = camera_spec.depth_to_mm(np.array([[0.3, 3.0]]))
        assert out[0, 0] == 300
        assert out[0, 1] == 3000

    def test_row_stride_matches_16uc1(self):
        h, w = camera_spec.DEPTH_HEIGHT, camera_spec.DEPTH_WIDTH
        out = camera_spec.depth_to_mm(np.full((h, w), 1.0))
        assert out.shape == (h, w)
        assert len(out.tobytes()) == w * h * 2
