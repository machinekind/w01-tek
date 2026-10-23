"""find_objects / go_to_object against a fake connector: frame freshness,
camera geometry, depth/colour parallax, TF error reporting, client cleanup.
No ROS runtime, no GPU, no LLM.

Needs the rai package (runs inside the wojtek_rai container, or any interpreter
with rai-core); skipped elsewhere.
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

rai = pytest.importorskip("rai")

from wojtek_rai import limits  # noqa: E402
from wojtek_rai import perception_tools as pt  # noqa: E402
from wojtek_rai.perception_tools import (  # noqa: E402
    FindObjectsTool,
    GoToObjectTool,
    _object_depth,
    _quat_rotate,
)
from wojtek_rai.tools import _permissions  # noqa: E402

OPTICAL_FRAME = "camera_color_optical_frame"
PITCH = 0.2617994  # URDF camera_joint: 15 deg down


# --- fakes -------------------------------------------------------------------


class FakeClock:
    """Replaces the `time` module inside perception_tools: sleeping advances it."""

    def __init__(self):
        self.t = 1_000.0

    def time(self):
        return self.t

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def _header(stamp, frame_id=OPTICAL_FRAME):
    return SimpleNamespace(stamp=SimpleNamespace(sec=int(stamp), nanosec=int((stamp % 1) * 1e9)), frame_id=frame_id)


def _colour(stamp, frame_id=OPTICAL_FRAME):
    return SimpleNamespace(header=_header(stamp, frame_id), width=640, height=360, encoding="rgb8")


def _depth(stamp, arr):
    data = (arr * 1000.0).astype(np.uint16).tobytes()
    return SimpleNamespace(header=_header(stamp), width=arr.shape[1], height=arr.shape[0], encoding="16UC1", data=data)


def _info(fx, cx, cy):
    return SimpleNamespace(k=[fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0])


def _quat(x=0.0, y=0.0, z=0.0, w=1.0):
    return SimpleNamespace(x=x, y=y, z=z, w=w)


def _qmul(a, b):
    """Hamilton product a * b of geometry-style (x, y, z, w) quaternions."""
    return _quat(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    )


def _axis(axis, angle):
    s, c = math.sin(angle / 2), math.cos(angle / 2)
    return _quat(**{axis: s}, w=c)


def _transform(x, y, z, q):
    return SimpleNamespace(transform=SimpleNamespace(translation=SimpleNamespace(x=x, y=y, z=z), rotation=q))


def map_to_optical():
    """map == base_link; base_link -> camera_link pitched 15 deg down at
    (0.32, 0, 0.07); camera_link -> optical is rpy (-pi/2, 0, -pi/2)."""
    optical = _qmul(_axis("z", -math.pi / 2), _axis("x", -math.pi / 2))
    return _transform(0.32, 0.0, 0.07, _qmul(_axis("y", PITCH), optical))


def make_connector(clock, colour_age=0.0, depth_age=0.0, depth_arr=None, colour_stamp=None, depth_stamp=None):
    from rai.communication.ros2 import ROS2Connector
    from rai.communication.ros2.messages import ROS2Message

    fake = MagicMock(spec=ROS2Connector)
    if depth_arr is None:
        depth_arr = np.full((240, 424), 3.0)
    now = clock.time()
    fake.last_msg = {
        limits.COLOR_IMAGE_TOPIC: ROS2Message(
            payload=_colour(colour_stamp if colour_stamp is not None else now - colour_age),
            timestamp=now - colour_age),
        limits.DEPTH_IMAGE_TOPIC: ROS2Message(
            payload=_depth(depth_stamp if depth_stamp is not None else now - depth_age, depth_arr),
            timestamp=now - depth_age),
        limits.DEPTH_INFO_TOPIC: ROS2Message(payload=_info(210.0, 212.0, 120.0), timestamp=now),
        limits.COLOR_INFO_TOPIC: ROS2Message(payload=_info(420.0, 320.0, 180.0), timestamp=now),
    }

    def receive_message(topic, timeout_sec=1.0, **_):
        if topic not in fake.last_msg:
            raise TimeoutError(f"Message from {topic} not received in {timeout_sec} seconds")
        return fake.last_msg[topic]

    fake.receive_message.side_effect = receive_message
    fake.get_transform.return_value = map_to_optical()
    return fake


def _detection(name, u, v, w, h, score=0.9):
    return SimpleNamespace(
        results=[SimpleNamespace(hypothesis=SimpleNamespace(class_id=name, score=score))],
        bbox=SimpleNamespace(center=SimpleNamespace(position=SimpleNamespace(x=u, y=v)), size_x=w, size_y=h),
    )


def _service(connector, detections, *, available=True, done=True):
    """The /detection client on the fake node answers with `detections`."""
    client = connector.node.create_client.return_value
    client.wait_for_service.return_value = available
    fut = client.call_async.return_value
    fut.done.return_value = done
    fut.result.return_value = SimpleNamespace(detections=SimpleNamespace(detections=detections))
    return client, fut


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(pt, "time", fake)
    # A plain namespace stands in for the ROS request, which rejects fakes.
    monkeypatch.setattr(pt, "RAIGroundingDino", SimpleNamespace(Request=SimpleNamespace))
    return fake


def _find(connector):
    return FindObjectsTool(connector=connector, **_permissions())


def _goto(connector):
    return GoToObjectTool(
        connector=connector, frame_id=limits.MAP_FRAME, action_name=limits.NAV_ACTION, **_permissions()
    )


# --- 1. frame freshness ------------------------------------------------------


def test_grab_refuses_a_stale_cached_frame(clock):
    connector = make_connector(clock, colour_age=3.0)
    with pytest.raises(RuntimeError, match="stale"):
        _find(connector)._grab(limits.COLOR_IMAGE_TOPIC)


def test_grab_waits_for_a_frame_newer_than_after(clock):
    connector = make_connector(clock)
    first = connector.last_msg[limits.COLOR_IMAGE_TOPIC]
    with pytest.raises(RuntimeError, match="stale"):
        _find(connector)._grab(limits.COLOR_IMAGE_TOPIC, timeout=1.0, after=first.timestamp)


def test_go_to_object_reports_a_stale_camera_without_sending_a_goal(clock, monkeypatch):
    goals = []
    monkeypatch.setattr(pt.GoToObjectTool, "_navigate", lambda self, x, y, yaw: goals.append((x, y, yaw)) or "sent")
    connector = make_connector(clock, colour_age=3.0)
    _service(connector, [_detection("ball", 320, 180, 40, 40)])

    out = _goto(connector)._run("ball")

    assert "stale" in out
    assert goals == []
    connector.node.create_client.assert_not_called()


def test_detect_refuses_unsynchronised_colour_and_depth(clock):
    now = clock.time()
    connector = make_connector(clock, colour_stamp=now - 1.0, depth_stamp=now)
    _service(connector, [])
    with pytest.raises(RuntimeError, match="not synchronised"):
        _find(connector)._detect(["ball"])


# --- 2. camera geometry through TF ------------------------------------------


def test_quat_rotate_matches_the_urdf_camera_chain():
    q = map_to_optical().transform.rotation
    fwd = _quat_rotate(q, (0.0, 0.0, 1.0))  # optical z = camera forward, pitched down
    assert fwd == pytest.approx((math.cos(PITCH), 0.0, -math.sin(PITCH)), abs=1e-9)
    right = _quat_rotate(q, (1.0, 0.0, 0.0))  # optical x = camera right = -y
    assert right == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)


def test_to_map_accounts_for_the_mount_pitch(clock):
    tool = _find(make_connector(clock))

    centre, err = tool._to_map((0.0, 0.0, 3.0), OPTICAL_FRAME)
    assert err is None
    assert centre == pytest.approx((0.32 + 3.0 * math.cos(PITCH), 0.0), abs=1e-6)

    # A point on a horizontal ray sits above the image centre by tan(15 deg).
    level, _ = tool._to_map((0.0, -3.0 * math.tan(PITCH), 3.0), OPTICAL_FRAME)
    assert level == pytest.approx((0.32 + 3.0 / math.cos(PITCH), 0.0), abs=1e-6)
    tool.connector.get_transform.assert_called_with(limits.MAP_FRAME, OPTICAL_FRAME, timeout_sec=2.0)


def test_to_map_without_a_frame_id_falls_back_to_the_level_base_link_model(clock):
    connector = make_connector(clock)
    connector.get_transform.return_value = _transform(0.0, 0.0, 0.0, _quat())

    xy, err = _find(connector)._to_map((0.3, 0.0, 3.0), "")

    assert err is None
    assert xy == pytest.approx((pt.CAMERA_FORWARD_OFFSET_M + 3.0, -0.3))
    connector.get_transform.assert_called_with(limits.MAP_FRAME, limits.BASE_FRAME, timeout_sec=2.0)


# --- 3. depth/colour parallax --------------------------------------------------


def _near_object_scene():
    """Background at 3 m; a 10 px object at 0.5 m around depth column 235."""
    depth = np.full((240, 424), 3.0)
    depth[115:125, 230:240] = 0.5
    k_d, k_c = _info(210.0, 212.0, 120.0).k, _info(420.0, 424.0, 240.0).k
    # The colour ray that hits the object once the 15 mm baseline is applied:
    # u_d = cx_d + fx_d * (tx - t/z) = 235 at z = 0.5.
    tx = 23.0 / 210.0 + 0.015 / 0.5
    return depth, tx, k_c, k_d


def test_object_depth_corrects_for_the_depth_colour_baseline():
    depth, tx, k_c, k_d = _near_object_scene()

    naive = _object_depth(depth, tx, 0.0, 20.0, 20.0, k_c, k_d, t=(0.0, 0.0, 0.0))
    corrected = _object_depth(depth, tx, 0.0, 20.0, 20.0, k_c, k_d, t=(0.015, 0.0, 0.0))

    assert naive == pytest.approx(3.0)      # beside the object, on the wall
    assert corrected == pytest.approx(0.5)  # on the object


def test_object_depth_without_extrinsics_is_the_plain_reprojection():
    depth, _, k_c, k_d = _near_object_scene()
    # Ray straight at the object: both paths read the same pixels.
    tx = 23.0 / 210.0
    assert _object_depth(depth, tx, 0.0, 20.0, 20.0, k_c, k_d) == pytest.approx(0.5)
    assert _object_depth(depth, tx, 0.0, 20.0, 20.0, k_c, k_d, t=(0.0, 0.0, 0.0)) == pytest.approx(0.5)


def test_median_depth_falls_back_to_the_whole_box_when_the_centre_is_empty():
    depth = np.full((240, 424), 2.0)
    depth[105:135, 197:227] = 0.0  # no return in and around the middle of the box
    assert pt._median_depth(depth, 212, 120, 40, 40) == pytest.approx(2.0)
    assert pt._median_depth(depth, 212, 120, 10, 10) is None


def test_extrinsics_are_fetched_once_and_default_to_zero(clock):
    connector = make_connector(clock)
    tool = _find(connector)
    assert tool._depth_to_color_translation() == (0.0, 0.0, 0.0)  # nobody publishes them in the sim

    from rai.communication.ros2.messages import ROS2Message

    connector.last_msg[pt.DEPTH_TO_COLOR_EXTRINSICS_TOPIC] = ROS2Message(
        payload=SimpleNamespace(translation=[0.015, 0.0, 0.0]))
    assert tool._depth_to_color_translation() == (0.015, 0.0, 0.0)
    del connector.last_msg[pt.DEPTH_TO_COLOR_EXTRINSICS_TOPIC]
    assert tool._depth_to_color_translation() == (0.015, 0.0, 0.0)


# --- 4. TF errors do not stick to the tool instance ----------------------------


def test_tf_failure_is_reported_only_for_the_call_it_happened_in(clock):
    connector = make_connector(clock)
    _service(connector, [_detection("ball", 320, 180, 40, 40)])
    connector.get_transform.side_effect = [RuntimeError("map->base_link missing"), map_to_optical()]
    tool = _find(connector)

    first = tool._run(["ball"])
    second = tool._run(["ball"])

    assert "map position unavailable" in first and "map->base_link missing" in first
    assert "map position unavailable" not in second
    assert "map (" in second


# --- 5. the service client is always destroyed ---------------------------------


def test_detection_client_is_destroyed_when_the_service_is_absent(clock):
    connector = make_connector(clock)
    client, _ = _service(connector, [], available=False)

    out = _find(connector)._run(["ball"])

    assert "not available" in out
    connector.node.destroy_client.assert_called_once_with(client)


def test_detection_timeout_cancels_the_future_and_destroys_the_client(clock):
    connector = make_connector(clock)
    client, fut = _service(connector, [], done=False)

    out = _find(connector)._run(["ball"])

    assert "timed out" in out
    fut.cancel.assert_called_once()
    connector.node.destroy_client.assert_called_once_with(client)


# --- compressed colour (the physical robot over WiFi) -----------------------


def _compressed_colour(stamp, w=8, h=4, frame_id=OPTICAL_FRAME):
    import cv2
    from sensor_msgs.msg import CompressedImage

    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 90, dtype=np.uint8))
    assert ok
    msg = CompressedImage()
    msg.header.stamp.sec, msg.header.stamp.nanosec = int(stamp), int((stamp % 1) * 1e9)
    msg.header.frame_id = frame_id
    msg.format = "rgb8; jpeg compressed bgr8"
    msg.data = buf.tobytes()
    return msg


def test_detection_gets_a_decoded_image_when_the_colour_stream_is_compressed(clock):
    from rai.communication.ros2.messages import ROS2Message

    connector = make_connector(clock)
    now = clock.time()
    connector.last_msg[limits.COLOR_IMAGE_TOPIC] = ROS2Message(payload=_compressed_colour(now), timestamp=now)
    client, _ = _service(connector, [])

    _find(connector)._run(["ball"])

    req = client.call_async.call_args.args[0]
    img = req.source_img
    assert (img.encoding, img.width, img.height) == ("bgr8", 8, 4)
    assert img.header.frame_id == OPTICAL_FRAME
    assert len(img.data) == 8 * 4 * 3
