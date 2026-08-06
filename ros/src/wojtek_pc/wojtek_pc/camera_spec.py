"""Single source of truth for the simulated D435 camera (wojtek#91).

Everything that must agree across the MJCF camera, the URDF frames, the
published CameraInfo and the tests lives here: resolution, intrinsics, the
mount pose on the body, frame names and topic names. The numbers mirror the
real perception stack (wojtek_perception_bringup, config/d435.yaml): the
driver decimates 848x480 to 424x240, so fx there (~418) halves to ~209.

Conventions, chosen once:
  * fx == fy, cx/cy at (size-1)/2 (pixel-center convention; the 0.5 px
    difference vs size/2 is far below the D435's own noise).
  * FOVY_DEG is DERIVED from fy, never written independently -- the MJCF
    camera and the CameraInfo can therefore not drift apart.
  * The MJCF camera looks along body +x pitched MOUNT_PITCH_RAD down, same
    xyaxes convention as EGO_CAM in training/wojtek_rl/build_model.py.
  * ROS optical frame (z forward, x right, y down) = MJCF camera frame
    (looks along -z, x right, y up) rotated pi about its x axis.

This module is deliberately importable without ROS or mujoco: message
builders import sensor_msgs lazily so the intrinsics tests stay pure.
"""

import math

import numpy as np

# -- image geometry ----------------------------------------------------------
DEPTH_WIDTH, DEPTH_HEIGHT = 424, 240
DEPTH_FX = DEPTH_FY = 209.0          # 418 at 848x480, halved by decimation
# Colour is for the VLM: contract = topic + encoding, resolution is a knob.
COLOR_WIDTH, COLOR_HEIGHT = 640, 360
COLOR_ENCODING = "rgb8"

# Vertical FOV the render must use so that fy comes out at DEPTH_FY.
FOVY_DEG = math.degrees(2.0 * math.atan(DEPTH_HEIGHT / (2.0 * DEPTH_FY)))

# -- depth validity window (matches config/cloud_reduce.yaml on the robot) ---
DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 3.0

# -- mount pose in base_link, shared by the MJCF camera and the URDF chain ---
# Centre of the head bracket: the body mesh's forward section (x > 0.2) is
# the triangular head mount, tip at x=0.292, centred at y=0, z~0.067
# (measured from base_link.dae vertices). The lens sits 3 cm proud of the
# tip so the bracket's edges stay out of the frame.
MOUNT_XYZ = (0.32, 0.0, 0.07)
MOUNT_PITCH_RAD = math.radians(15.0)  # optical axis pitched down

# MJCF xyaxes (camera right, camera up) for a camera looking along body +x
# pitched MOUNT_PITCH_RAD down -- same construction as EGO_CAM (10 deg).
XYAXES = (
    0.0, -1.0, 0.0,
    math.sin(MOUNT_PITCH_RAD), 0.0, math.cos(MOUNT_PITCH_RAD),
)

CAMERA_NAME = "d435_depth"

# -- names the perception stack expects (realsense2_camera defaults with
#    camera_name=camera, camera_namespace=camera) ----------------------------
DEPTH_FRAME_ID = "camera_depth_optical_frame"
COLOR_FRAME_ID = "camera_color_optical_frame"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
DEPTH_INFO_TOPIC = "/camera/camera/depth/camera_info"
COLOR_TOPIC = "/camera/camera/color/image_raw"
COLOR_INFO_TOPIC = "/camera/camera/color/camera_info"


def intrinsics(width, height, fovy_deg=FOVY_DEG):
    """(fx, fy, cx, cy) for a render of the given size at fovy_deg."""
    fy = 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)
    return fy, fy, (width - 1) / 2.0, (height - 1) / 2.0


def depth_to_mm(depth_m, min_m=DEPTH_MIN_M, max_m=DEPTH_MAX_M):
    """Metric depth image -> 16UC1 millimetres, RealSense-style.

    Anything outside [min_m, max_m] -- including the renderer's far-plane
    value for sky pixels and any non-finite garbage -- becomes 0, the
    RealSense "no return" convention that cloud_reduce already filters.
    """
    depth_m = np.asarray(depth_m)
    valid = np.isfinite(depth_m) & (depth_m >= min_m) & (depth_m <= max_m)
    mm = np.where(valid, np.round(depth_m * 1000.0), 0.0)
    return np.clip(mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def xyaxes_to_quat(xyaxes=XYAXES):
    """MJCF xyaxes (right, up) -> wxyz quaternion, for MjSpec cameras."""
    import mujoco

    x = np.array(xyaxes[:3], dtype=float)
    y = np.array(xyaxes[3:], dtype=float)
    x /= np.linalg.norm(x)
    y -= x * (y @ x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).flatten())
    return quat


def camera_info_msg(width, height, frame_id, fovy_deg=FOVY_DEG):
    """sensor_msgs/CameraInfo for a distortion-free pinhole render.

    The caller sets header.stamp (it must be the physics-tick stamp the
    image was snapshotted at, shared with the Image message).
    """
    from sensor_msgs.msg import CameraInfo

    fx, fy, cx, cy = intrinsics(width, height, fovy_deg)
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg
