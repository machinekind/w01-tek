"""Object-grounded perception and navigation on RAI's GroundingDINO service.

`find_objects`: detect named objects in the current camera image, report
bearing and distance (from the depth image) and their position in the map.
`go_to_object`: the same detection, then a Nav2 goal 0.6 m in front of the
best match, facing it, sent through `nav_tools._TimedNavMixin` so it gets the
same workspace box, permission check, deadline and cancel handle as
navigate_to_pose. Needs the wojtek_perception container (`/detection`,
rai_interfaces/srv/RAIGroundingDino) and, for go_to_object, wojtek_nav.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type

import numpy as np
from pydantic import BaseModel, Field
from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2.base import BaseROS2Tool
from rai.tools.ros2.navigation.nav2_blocking import NavigateToPoseBlockingTool
from rai_interfaces.srv import RAIGroundingDino
from rclpy.task import Future
from sensor_msgs.msg import CompressedImage, Image

from wojtek_rai import limits
from wojtek_rai.nav_tools import _TimedNavMixin

DETECTION_TIMEOUT_S = 40.0   # a cold GroundingDINO load on the 4 GB laptop GPU exceeds 15 s
IMAGE_TIMEOUT_S = 5.0
# The connector hands back its cached last message however old it is (its
# timeout only bounds the wait for the first message ever), so freshness is
# enforced here, on the connector's wall-clock arrival time: a frame older
# than MAX_FRAME_AGE_S is refused rather than fused with the robot's pose of
# now. Colour and depth must also be from the same moment.
MAX_FRAME_AGE_S = 0.5
MAX_PAIR_SKEW_S = 0.2
DEPTH_ROI_FRACTION = 0.5     # central part of the box used for the depth estimate
MIN_VALID_DEPTH_M = 0.2
MAX_VALID_DEPTH_M = 6.0
# realsense2_camera's depth->colour extrinsic (P_colour = R * P_depth + t): on
# the D435 the two imagers are ~15 mm apart. The sim's URDF puts both optical
# frames at one origin and publishes nothing here, which reads as t = 0.
DEPTH_TO_COLOR_EXTRINSICS_TOPIC = "/camera/camera/extrinsics/depth_to_color"
EXTRINSICS_TIMEOUT_S = 0.5
NO_EXTRINSICS = (0.0, 0.0, 0.0)
REPROJECTION_PASSES = 3
PARALLAX_SEED_PERCENTILE = 20
DEPTH_CONVERGED_M = 0.01
# Fallback for a colour frame without a frame_id: camera_link x in base_link
# (URDF camera_joint), optical axis taken as level. Only the TF path knows
# about the 15 deg mount pitch.
CAMERA_FORWARD_OFFSET_M = 0.32


@dataclass
class Sighting:
    name: str
    score: float
    bearing_rad: float        # + left
    distance_m: Optional[float]
    map_xy: Optional[tuple]   # (x, y) in the map frame, None without depth/TF


def _roi_depths(depth, u, v, w, h, fraction):
    sy, sx = depth.shape
    hw, hh = max(1.0, w * fraction / 2), max(1.0, h * fraction / 2)
    x0, x1 = int(max(0, u - hw)), int(min(sx, u + hw + 1))
    y0, y1 = int(max(0, v - hh)), int(min(sy, v + hh + 1))
    roi = depth[y0:y1, x0:x1]
    return roi[(roi > MIN_VALID_DEPTH_M) & (roi < MAX_VALID_DEPTH_M)]


def _box_depths(depth, u, v, w, h):
    """Valid depths in the central DEPTH_ROI_FRACTION of the box at (u, v), or
    in the whole box when that patch is empty: the robot's HIGH_ACCURACY preset
    fills ~65% of the depth image."""
    for fraction in (DEPTH_ROI_FRACTION, 1.0):
        valid = _roi_depths(depth, u, v, w, h, fraction)
        if valid.size:
            break
    return valid


def _median_depth(depth, u, v, w, h) -> Optional[float]:
    valid = _box_depths(depth, u, v, w, h)
    return float(np.median(valid)) if valid.size else None


def _object_depth(depth_arr, tx, ty, w_c, h_c, k_c, k_d, t=NO_EXTRINSICS) -> Optional[float]:
    """Median depth (m) of a colour detection: viewing ray (tx, ty) in the colour
    optical frame, box w_c x h_c colour pixels, re-projected into the depth image
    (intrinsics k_d) and, on the real D435, shifted by the depth->colour
    translation t (its rotation is ~identity)."""
    fx_c, fy_c = k_c[0], k_c[4]
    fx_d, fy_d, cx_d, cy_d = k_d[0], k_d[4], k_d[2], k_d[5]
    u_d, v_d = cx_d + fx_d * tx, cy_d + fy_d * ty
    w_d, h_d = w_c * fx_d / fx_c, h_c * fy_d / fy_c
    z = _median_depth(depth_arr, u_d, v_d, w_d, h_d)
    if z is None or not any(t):
        return z
    # Parallax: the colour ray lands fx_d*t/z pixels away in the depth image,
    # which for a small or near object is beside it. Seed the shift with the
    # near side of the naive patch (the detected object stands in front of its
    # surroundings), move the colour-frame point at that depth into the depth
    # frame and read the median there; repeat until it settles.
    z = float(np.percentile(_box_depths(depth_arr, u_d, v_d, w_d, h_d), PARALLAX_SEED_PERCENTILE))
    for _ in range(REPROJECTION_PASSES):
        x, y, zd = z * tx - t[0], z * ty - t[1], z - t[2]
        z_new = _median_depth(depth_arr, cx_d + fx_d * x / zd, cy_d + fy_d * y / zd, w_d, h_d)
        if z_new is None:
            break
        converged = abs(z_new - z) < DEPTH_CONVERGED_M
        z = z_new
        if converged:
            break
    return z


def _quat_rotate(q, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Rotate the vector v by the unit quaternion q (geometry_msgs Quaternion)."""
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    x, y, z = v
    # v' = v + 2 q_vec x (q_vec x v + w v)
    cx, cy, cz = qy * z - qz * y + qw * x, qz * x - qx * z + qw * y, qx * y - qy * x + qw * z
    return (x + 2.0 * (qy * cz - qz * cy), y + 2.0 * (qz * cx - qx * cz), z + 2.0 * (qx * cy - qy * cx))


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _as_image(colour):
    """The colour frame as sensor_msgs/Image for the detection service, which
    takes no CompressedImage: a JPEG frame (WiFi transport) is decoded here,
    keeping its header so the point still lands in the camera's frame."""
    if not isinstance(colour, CompressedImage):
        return colour
    import cv2  # heavy import, only needed on this path

    bgr = cv2.imdecode(np.frombuffer(colour.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"undecodable compressed colour frame ({colour.format!r})")
    img = Image()
    img.header = colour.header
    img.height, img.width = bgr.shape[0], bgr.shape[1]
    img.encoding, img.is_bigendian, img.step = "bgr8", 0, bgr.shape[1] * 3
    img.data = bgr.tobytes()
    return img


class _DetectMixin:
    """Shared detection pipeline for the two tools (needs a ROS2Connector)."""

    connector: ROS2Connector
    box_threshold: float = 0.35
    text_threshold: float = 0.45

    def _grab(self, topic: str, timeout: float = IMAGE_TIMEOUT_S, after: float = 0.0):
        """The connector's message on `topic` once one has arrived within
        MAX_FRAME_AGE_S (and after wall-clock `after`, to wait for the next
        frame). Raises RuntimeError when the stream has stalled."""
        deadline = time.monotonic() + timeout
        try:
            self.connector.receive_message(topic, timeout_sec=timeout)  # subscribes, waits for the first message
        except TimeoutError as exc:
            raise RuntimeError(f"no message from {topic} in {timeout:.0f} s (is the camera publishing?)") from exc
        while True:
            cached = self.connector.last_msg.get(topic)
            arrived = cached.timestamp if cached is not None else float("-inf")
            age = time.time() - arrived
            if age <= MAX_FRAME_AGE_S and arrived > after:
                return cached
            if time.monotonic() > deadline:
                raise RuntimeError(f"{topic} is stale: last frame {age:.1f} s old (camera stream or WiFi stalled)")
            time.sleep(0.05)

    def _grab_pair(self):
        """A colour and a depth frame taken at (nearly) the same moment."""
        colour, depth = self._grab(limits.COLOR_IMAGE_TOPIC), self._grab(limits.DEPTH_IMAGE_TOPIC)
        skew = abs(_stamp(colour.payload) - _stamp(depth.payload))
        if skew > MAX_PAIR_SKEW_S:
            # One stream ran ahead (dropped frames): wait for the next frame of the other.
            try:
                if _stamp(colour.payload) < _stamp(depth.payload):
                    colour = self._grab(limits.COLOR_IMAGE_TOPIC, after=colour.timestamp)
                else:
                    depth = self._grab(limits.DEPTH_IMAGE_TOPIC, after=depth.timestamp)
            except RuntimeError as exc:
                raise RuntimeError(f"colour/depth frames not synchronised (skew {skew:.2f} s): {exc}") from exc
            skew = abs(_stamp(colour.payload) - _stamp(depth.payload))
        if skew > MAX_PAIR_SKEW_S:
            raise RuntimeError(f"colour/depth frames not synchronised (skew {skew:.2f} s)")
        return colour.payload, depth.payload

    def _depth_to_color_translation(self) -> Tuple[float, float, float]:
        """t of the driver's depth->colour extrinsic, fetched once; (0, 0, 0)
        where nobody publishes it (the sim) or its message type is not installed."""
        cached = getattr(self, "_depth_to_color_t", None)
        if cached is not None:
            return cached
        try:
            ext = self.connector.receive_message(DEPTH_TO_COLOR_EXTRINSICS_TOPIC, timeout_sec=EXTRINSICS_TIMEOUT_S)
            t = tuple(float(v) for v in ext.payload.translation)
        except Exception:  # noqa: BLE001
            return NO_EXTRINSICS
        self._depth_to_color_t = t  # pydantic lets a private (underscore) attribute through
        return t

    def _call_detection(self, image, names: List[str]):
        client = self.connector.node.create_client(RAIGroundingDino, limits.DETECTION_SERVICE)
        try:
            if not client.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f"{limits.DETECTION_SERVICE} not available (is wojtek_perception up?)")
            req = RAIGroundingDino.Request()
            req.source_img = image
            req.classes = " , ".join(names)
            req.box_threshold = float(self.box_threshold)
            req.text_threshold = float(self.text_threshold)
            fut: Future = client.call_async(req)
            t0 = time.monotonic()
            while not fut.done():
                if time.monotonic() - t0 > DETECTION_TIMEOUT_S:
                    fut.cancel()
                    raise RuntimeError(f"detection service timed out after {DETECTION_TIMEOUT_S:.0f} s")
                time.sleep(0.05)
            return fut.result()
        finally:
            # On every path: each attempt over a flapping link would otherwise
            # leave a DDS reader/writer pair on the node.
            self.connector.node.destroy_client(client)

    def _detect(self, names: List[str]) -> Tuple[List[Sighting], Optional[str]]:
        """Sightings by descending score, and the first map-TF failure (None if
        every point with a depth was placed in the map)."""
        colour, depth = self._grab_pair()
        info = self._grab(limits.DEPTH_INFO_TOPIC).payload
        cinfo = self._grab(limits.COLOR_INFO_TOPIC).payload
        t_dc = self._depth_to_color_translation()
        resp = self._call_detection(_as_image(colour), names)

        depth_arr = _depth_array(depth)
        fx_c, fy_c, cx_c, cy_c = cinfo.k[0], cinfo.k[4], cinfo.k[2], cinfo.k[5]
        sightings: List[Sighting] = []
        tf_error: Optional[str] = None
        for det in resp.detections.detections:
            if not det.results:
                continue
            name = det.results[0].hypothesis.class_id
            score = float(det.results[0].hypothesis.score)
            u, v = det.bbox.center.position.x, det.bbox.center.position.y
            w, h = det.bbox.size_x, det.bbox.size_y
            # Colour and depth have different intrinsics/FOV: go through the
            # viewing ray (colour K) and re-project into the depth image (depth K).
            tx, ty = (u - cx_c) / fx_c, (v - cy_c) / fy_c
            z = _object_depth(depth_arr, tx, ty, w, h, cinfo.k, info.k, t_dc)
            bearing = -math.atan(tx)
            dist = None if z is None else z * math.sqrt(1.0 + tx * tx)
            map_xy = None
            if z is not None:
                map_xy, err = self._to_map((z * tx, z * ty, z), colour.header.frame_id)
                if err and tf_error is None:
                    tf_error = err
            sightings.append(Sighting(name, score, bearing, dist, map_xy))
        sightings.sort(key=lambda s: -s.score)
        return sightings, tf_error

    def _to_map(self, point: Tuple[float, float, float], frame_id: str) -> Tuple[Optional[tuple], Optional[str]]:
        """Point in the camera's optical frame (x right, y down, z forward) ->
        (x, y) in the map frame through TF, or (None, why not)."""
        if not frame_id:
            # No frame on the image: level optical axis from the mount point
            # in base_link (the pre-TF approximation, ~10 cm short at 3 m).
            x, _, z = point
            point, frame_id = (CAMERA_FORWARD_OFFSET_M + z, -x, 0.0), limits.BASE_FRAME
        try:
            tf = self.connector.get_transform(limits.MAP_FRAME, frame_id, timeout_sec=2.0)
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
        t = tf.transform.translation
        x, y, _ = _quat_rotate(tf.transform.rotation, point)
        return (t.x + x, t.y + y), None


def _depth_array(depth_msg):
    if depth_msg.encoding == "16UC1":
        return np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width) / 1000.0
    if depth_msg.encoding == "32FC1":
        return np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)
    raise RuntimeError(f"unsupported depth encoding {depth_msg.encoding}")


def _describe(s: Sighting) -> str:
    side = "left" if s.bearing_rad > 0.05 else "right" if s.bearing_rad < -0.05 else "ahead"
    dist = f"{s.distance_m:.2f} m" if s.distance_m else "distance unknown"
    pos = f", map ({s.map_xy[0]:.2f}, {s.map_xy[1]:.2f})" if s.map_xy else ""
    return f"{s.name} (score {s.score:.2f}): {dist}, {abs(math.degrees(s.bearing_rad)):.0f} deg {side}{pos}"


class FindObjectsInput(BaseModel):
    object_names: List[str] = Field(..., description="Natural-language names, e.g. ['ball', 'fire hydrant'].")


class FindObjectsTool(_DetectMixin, BaseROS2Tool):
    name: str = "find_objects"
    description: str = (
        "Detect the named objects in the current camera view and report each one's "
        "distance, bearing and map position. Use it to locate something before navigating."
    )
    args_schema: Type[FindObjectsInput] = FindObjectsInput

    def _run(self, object_names: List[str]) -> str:
        try:
            sightings, note = self._detect(object_names)
        except RuntimeError as exc:
            return f"find_objects failed: {exc}"
        if not sightings:
            return f"None of {object_names} detected in the current view."
        out = "\n".join(_describe(s) for s in sightings)
        if note and any(s.map_xy is None for s in sightings):
            out += f"\n(map position unavailable: {note})"
        return out


class GoToObjectInput(BaseModel):
    object_name: str = Field(..., description="What to walk to, e.g. 'ball' or 'fire hydrant'.")


class GoToObjectTool(_DetectMixin, _TimedNavMixin, NavigateToPoseBlockingTool):
    name: str = "go_to_object"
    description: str = (
        "Detect the named object in the current camera view and navigate to a spot "
        f"{limits.OBJECT_STANDOFF_M} m in front of it, facing it. Fails if the object is not "
        "visible; turn or move first in that case."
    )
    args_schema: Type[GoToObjectInput] = GoToObjectInput

    def _run(self, object_name: str) -> str:  # type: ignore[override]
        try:
            sightings, _ = self._detect([object_name])
        except RuntimeError as exc:
            return f"go_to_object refused: {exc}"
        best = next((s for s in sightings if s.map_xy and s.distance_m), None)
        if best is None:
            return f"{object_name!r} not detected with a usable distance; look around first."
        try:
            tf = self.connector.get_transform(limits.MAP_FRAME, limits.BASE_FRAME, timeout_sec=2.0)
        except Exception as exc:  # noqa: BLE001
            return f"no map pose: {exc}"
        rx, ry = tf.transform.translation.x, tf.transform.translation.y
        ox, oy = best.map_xy
        d = math.hypot(ox - rx, oy - ry)
        yaw = math.atan2(oy - ry, ox - rx)
        back = min(limits.OBJECT_STANDOFF_M, d)
        gx, gy = ox - back * math.cos(yaw), oy - back * math.sin(yaw)
        # _TimedNavMixin, not RAI's unbounded send_goal: the goal is refused
        # outside the workspace, bounded by NAV_GOAL_TIMEOUT_S and cancellable.
        result = self._navigate(gx, gy, yaw)
        return f"{_describe(best)}. Goal ({gx:.2f}, {gy:.2f}): {result}"


def build_perception_tools(connector: ROS2Connector, perms: dict, navigation: bool = True) -> List[Any]:
    """find_objects always; go_to_object only where Nav2 can be driven
    (navigation=False on a target without odometry)."""
    tools: List[Any] = [FindObjectsTool(connector=connector, **perms)]
    if navigation:
        # No workspace_bounds_* kwargs: rai-core 2.12 silently drops them; the
        # box is enforced in _TimedNavMixin._navigate (see nav_tools.py).
        tools.append(GoToObjectTool(
            connector=connector, frame_id=limits.MAP_FRAME, action_name=limits.NAV_ACTION, **perms
        ))
    return tools
