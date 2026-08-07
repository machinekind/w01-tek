"""Unit tests for the ROS-free tracker math.

Everything here runs on synthetic geometry -- no mujoco, no
pupil-apriltags, no images -- so it stays in the fast dependency-light
tier.  The rendered end-to-end path is scripts/sim_rig_check.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from wojtek_benchmark import tracker  # noqa: E402


def _rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _synthetic_centers(T_cam_world, leg_x=2.0, leg_y=1.5):
    """Floor-tag centers as a camera at T_cam_world would see them."""
    world_pts = {
        "world_origin": np.zeros(3),
        "world_x": np.array([leg_x, 0.0, 0.0]),
        "world_y": np.array([0.0, leg_y, 0.0]),
    }
    return {
        role: (T_cam_world @ np.append(p, 1.0))[:3]
        for role, p in world_pts.items()
    }


def _example_T():
    T = tracker.se3(_rot_z(30.0) @ np.diag([1.0, -1.0, -1.0]), [0.3, -0.2, 2.5])
    return T


def test_solve_recovers_camera_pose():
    T_true = _example_T()
    centers = _synthetic_centers(T_true)
    T = tracker.solve_world_from_floor(centers, (2.0, 1.5))
    assert np.allclose(T, T_true, atol=1e-12)


def test_solve_refuses_missing_tag():
    centers = _synthetic_centers(_example_T())
    del centers["world_y"]
    with pytest.raises(tracker.CalibrationError, match="world_y"):
        tracker.solve_world_from_floor(centers, (2.0, 1.5))


def test_solve_refuses_unmeasured_legs():
    centers = _synthetic_centers(_example_T())
    with pytest.raises(tracker.CalibrationError, match="tape-measure"):
        tracker.solve_world_from_floor(centers, (None, None))


def test_solve_refuses_leg_length_mismatch():
    # Vision sees 2.0 m; the tape allegedly measured 2.1 m -> scaled print,
    # wrong size_m, or misplaced tag. > 1 % must refuse.
    centers = _synthetic_centers(_example_T())
    with pytest.raises(tracker.CalibrationError, match="leg x"):
        tracker.solve_world_from_floor(centers, (2.1, 1.5))


def test_solve_accepts_within_tolerance():
    centers = _synthetic_centers(_example_T())
    tracker.solve_world_from_floor(centers, (2.0 * 1.005, 1.5))


def test_solve_refuses_bent_L():
    T_true = _example_T()
    centers = _synthetic_centers(T_true)
    # Push world_y sideways: legs keep their lengths, the angle does not.
    p = np.array([0.15, 1.5 * np.cos(np.arcsin(0.1)), 0.0])
    centers["world_y"] = (T_true @ np.append(p, 1.0))[:3]
    with pytest.raises(tracker.CalibrationError, match="L angle"):
        tracker.solve_world_from_floor(centers, (2.0, 1.5))


def test_inv_se3():
    T = _example_T()
    assert np.allclose(tracker.inv_se3(T) @ T, np.eye(4), atol=1e-12)


def test_averaged_centers_ignores_dropped_roles():
    frames = [
        {"world_origin": [0.0, 0.0, 2.0], "world_x": [1.0, 0.0, 2.0]},
        {"world_origin": [0.2, 0.0, 2.0]},
    ]
    avg = tracker.averaged_centers(frames)
    assert np.allclose(avg["world_origin"], [0.1, 0.0, 2.0])
    assert np.allclose(avg["world_x"], [1.0, 0.0, 2.0])
    assert "world_y" not in avg


def test_tag_to_surface_is_x_flip():
    # Involution (applying it twice is identity) and x-preserving: the
    # detected tag yaw IS the surface yaw. sim_rig_check.py pins the same
    # constant against rendered ground truth.
    R = tracker.TAG_TO_SURFACE
    assert np.allclose(R @ R, np.eye(3))
    assert np.allclose(R @ np.array([1.0, 0, 0]), [1.0, 0, 0])
    assert np.isclose(np.linalg.det(R), 1.0)


def test_yaw_deg():
    assert np.isclose(tracker.yaw_deg(tracker.se3(_rot_z(40.0), [0, 0, 0])), 40.0)
