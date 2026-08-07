"""Detection and world-frame math for the AprilTag benchmark rig.

Deliberately ROS-free and sensor-agnostic: input is a grayscale image plus
pinhole intrinsics, output is rigid transforms.  The same code runs on the
sim render and on a real camera stream; nothing in here can tell which.

Conventions:
* Camera frame is the CV/ROS optical convention (x right, y down,
  z forward) -- what pupil-apriltags natively returns.
* The bench world frame is built from the three floor-tag CENTERS: origin
  at world_origin, x toward world_x, y toward world_y (orthogonalized),
  z = x cross y (up, if the tags lie on the floor and were placed
  counterclockwise as documented).  Tag yaw does not enter -- placement
  cannot rotate the frame, only translate it.
* Calibration must REFUSE rather than degrade: the measured-vs-detected
  leg-length check and the L right-angle check are physical-unit gates
  (meters, degrees), not tuned scores, per the course-benchmark design
  rules.
"""

from dataclasses import dataclass

import numpy as np

# Pose estimation runs at unit tag size and is scaled per tag afterwards:
# the estimator solves corners at +-size/2, so translation is exactly
# linear in size while rotation is size-free.  One detector pass covers a
# rig with mixed 160 mm / 80 mm tags.
_UNIT_TAG_SIZE = 1.0

DEFAULT_LEG_TOL_FRAC = 0.01     # 1 % of a tape-measured leg
DEFAULT_ANGLE_TOL_DEG = 3.0     # L squareness; placement error, not vision


class CalibrationError(RuntimeError):
    """The floor-tag geometry cannot be trusted; refuse, don't guess."""


# Detected-tag frame vs the physical tag surface: the apriltag convention
# has x right ON the tag face, y down, z out the BACK, while a mounted tag
# is naturally described x forward-on-the-robot, y left, z out the face
# (the surface normal).  The two differ by a 180 deg flip about x -- so x
# is shared, and a flat-mounted tag's detected yaw IS the robot's yaw.
# Verified against MuJoCo ground truth by sim_rig_check.py (relative
# rotation constant to < 0.05 deg across robot poses).
TAG_TO_SURFACE = np.diag([1.0, -1.0, -1.0])


@dataclass
class TagDetection:
    tag_id: int
    R: np.ndarray          # 3x3, tag frame -> camera optical frame
    t: np.ndarray          # 3, tag center in camera optical frame [m]
    corners: np.ndarray    # 4x2 image pixels
    decision_margin: float


def make_detector(nthreads=1):
    """pupil-apriltags detector for the rig family (lazy heavy import).

    quad_decimate stays 1.0 (no pre-downsampling): the robot tag lives near
    the decode limit of a few px per cell, and decimation would throw those
    pixels away for speed the tracker does not need at rig frame rates.
    """
    from pupil_apriltags import Detector

    return Detector(families="tag36h11", nthreads=nthreads, quad_decimate=1.0)


def detect(detector, gray, intrinsics, sizes_by_id):
    """Detect known rig tags and return {tag_id: TagDetection}.

    gray: uint8 (h, w).  intrinsics: (fx, fy, cx, cy).  sizes_by_id maps a
    tag id to its BLACK-square edge in meters (apriltags.yaml size_m);
    detections of ids not in it are ignored (someone else's tag in frame).
    """
    out = {}
    for d in detector.detect(
        gray, estimate_tag_pose=True, camera_params=tuple(intrinsics),
        tag_size=_UNIT_TAG_SIZE,
    ):
        size = sizes_by_id.get(d.tag_id)
        if size is None:
            continue
        out[d.tag_id] = TagDetection(
            tag_id=d.tag_id,
            R=np.asarray(d.pose_R, dtype=float),
            t=np.asarray(d.pose_t, dtype=float).ravel() * size,
            corners=np.asarray(d.corners, dtype=float),
            decision_margin=float(d.decision_margin),
        )
    return out


def se3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def inv_se3(T):
    R, t = T[:3, :3], T[:3, 3]
    return se3(R.T, -R.T @ t)


def solve_world_from_floor(
    centers, expected_legs,
    leg_tol_frac=DEFAULT_LEG_TOL_FRAC, angle_tol_deg=DEFAULT_ANGLE_TOL_DEG,
):
    """T_cam_world from the three floor-tag centers in the camera frame.

    centers: {"world_origin"|"world_x"|"world_y": (3,) camera-frame point}.
    expected_legs: (leg_x_m, leg_y_m), the tape-measured (or, in sim,
    configured) center-to-center distances.  None values REFUSE -- an
    unmeasured course must not calibrate, that is the whole point of
    recording the legs.
    """
    for role in ("world_origin", "world_x", "world_y"):
        if role not in centers:
            raise CalibrationError(f"floor tag {role!r} not detected")
    leg_x_m, leg_y_m = expected_legs
    if leg_x_m is None or leg_y_m is None:
        raise CalibrationError(
            "expected leg lengths are unset (distance_from_origin_m is null) "
            "-- tape-measure the course and record them before calibrating"
        )

    o = np.asarray(centers["world_origin"], dtype=float)
    vx = np.asarray(centers["world_x"], dtype=float) - o
    vy = np.asarray(centers["world_y"], dtype=float) - o

    for name, v, want in (("x", vx, leg_x_m), ("y", vy, leg_y_m)):
        got = float(np.linalg.norm(v))
        if abs(got - want) > leg_tol_frac * want:
            raise CalibrationError(
                f"leg {name}: vision measures {got:.4f} m, tape says "
                f"{want:.4f} m (> {leg_tol_frac:.1%}) -- wrong tag size, "
                "scaled print, or misplaced tag"
            )
    angle = np.degrees(np.arccos(np.clip(
        vx @ vy / (np.linalg.norm(vx) * np.linalg.norm(vy)), -1.0, 1.0
    )))
    if abs(angle - 90.0) > angle_tol_deg:
        raise CalibrationError(
            f"L angle is {angle:.2f} deg, not 90 +- {angle_tol_deg} "
            "-- floor tags misplaced"
        )

    ex = vx / np.linalg.norm(vx)
    ey = vy - ex * (ex @ vy)
    ey /= np.linalg.norm(ey)
    R = np.column_stack([ex, ey, np.cross(ex, ey)])
    return se3(R, o)


def averaged_centers(frames):
    """Average per-role camera-frame centers over several frames.

    frames: iterable of {role: (3,) point}.  Roles missing from a frame
    simply don't contribute; a role missing from EVERY frame is absent from
    the result (and solve_world_from_floor will refuse).
    """
    sums, counts = {}, {}
    for frame in frames:
        for role, p in frame.items():
            sums[role] = sums.get(role, 0.0) + np.asarray(p, dtype=float)
            counts[role] = counts.get(role, 0) + 1
    return {role: sums[role] / counts[role] for role in sums}


def yaw_deg(T):
    """Heading of T's x axis about world z, for scalar error reporting."""
    return float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
