"""Subscription lifetime of the camera feed against a mock node. No ROS runtime.

The point of the feed on the real robot is that a raw colour reader exists
only while someone is looking, so these tests pin down when subscriptions are
created and destroyed. Needs rclpy/sensor_msgs importable (the wojtek_rai
container); skipped elsewhere.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("rclpy")

from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

from wojtek_rai import limits  # noqa: E402
from wojtek_rai.camera_feed import MAX_FRAME_AGE_S, CameraFeed, grab_once  # noqa: E402


def _rgb_image(w: int = 4, h: int = 2) -> Image:
    msg = Image()
    msg.width, msg.height, msg.encoding = w, h, "rgb8"
    msg.data = bytes(w * h * 3)
    return msg


class _NewSubscription:
    """create_subscription side effect: a fresh mock per call, all of them remembered."""

    def __init__(self) -> None:
        self.created = []

    def __call__(self, *_args, **_kwargs):
        sub = MagicMock(name=f"subscription{len(self.created)}")
        self.created.append(sub)
        return sub


@pytest.fixture
def node():
    fake = MagicMock()
    fake.create_subscription.side_effect = _NewSubscription()
    return fake


def feed_subscription(node):
    """The one subscription object the mock node handed out."""
    calls = node.create_subscription.side_effect.created
    assert len(calls) == 1
    return calls[0]


def _callback_of(node, topic):
    for call in node.create_subscription.call_args_list:
        if call.args[1] == topic:
            return call.args[2]
    raise AssertionError(f"no subscription on {topic}")


def test_feed_creates_no_subscription_until_enabled(node):
    feed = CameraFeed(node)
    node.create_subscription.assert_not_called()
    assert feed.latest() == (None, float("inf"), None, float("inf"))


def test_enable_color_subscribes_once_with_sensor_qos_and_disable_destroys(node):
    feed = CameraFeed(node)
    feed.enable_color(True)
    feed.enable_color(True)  # idempotent
    node.create_subscription.assert_called_once()
    msg_cls, topic, _cb, qos = node.create_subscription.call_args.args
    assert (msg_cls, topic, qos) == (Image, limits.COLOR_IMAGE_TOPIC, qos_profile_sensor_data)
    assert feed.is_enabled(limits.COLOR_IMAGE_TOPIC)

    feed.enable_color(False)
    feed.enable_color(False)  # idempotent
    node.destroy_subscription.assert_called_once_with(feed_subscription(node))
    assert not feed.is_enabled(limits.COLOR_IMAGE_TOPIC)


def test_disabling_a_stream_drops_its_cached_frame(node):
    feed = CameraFeed(node)
    feed.enable_depth(True)
    _callback_of(node, limits.DEPTH_IMAGE_TOPIC)(_rgb_image())
    assert feed.latest_msgs()[2] is not None
    feed.enable_depth(False)
    assert feed.latest_msgs()[2] is None


def test_latest_decodes_the_newest_frame_and_reports_its_age(node, monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr("wojtek_rai.camera_feed.time.time", lambda: clock["t"])
    feed = CameraFeed(node)
    feed.enable_color(True)
    _callback_of(node, limits.COLOR_IMAGE_TOPIC)(_rgb_image(4, 2))
    clock["t"] += 0.25
    color, age, depth, depth_age = feed.latest()
    assert color.shape == (2, 4, 3)
    assert age == pytest.approx(0.25)
    assert depth is None and depth_age == float("inf")


def test_fresh_refuses_stale_missing_and_disabled_frames(node, monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr("wojtek_rai.camera_feed.time.time", lambda: clock["t"])
    feed = CameraFeed(node)
    with pytest.raises(RuntimeError, match="not enabled"):
        feed.fresh(limits.COLOR_IMAGE_TOPIC)
    feed.enable_color(True)
    with pytest.raises(RuntimeError, match="no camera frame"):
        feed.fresh(limits.COLOR_IMAGE_TOPIC)
    frame = _rgb_image()
    _callback_of(node, limits.COLOR_IMAGE_TOPIC)(frame)
    assert feed.fresh(limits.COLOR_IMAGE_TOPIC) is frame
    clock["t"] += MAX_FRAME_AGE_S + 0.5
    with pytest.raises(RuntimeError, match="is the link up"):
        feed.fresh(limits.COLOR_IMAGE_TOPIC)


def test_grab_once_returns_the_first_message_and_destroys_its_subscription(node):
    frame, later = _rgb_image(), _rgb_image()
    sub = MagicMock(name="subscription")

    def deliver_two(_cls, _topic, cb, _qos):
        # The executor thread delivers frames as soon as the subscription exists.
        cb(frame)
        cb(later)
        return sub

    node.create_subscription.side_effect = deliver_two
    got = grab_once(node, limits.COLOR_IMAGE_TOPIC, Image, timeout=1.0)
    assert got is frame
    node.destroy_subscription.assert_called_once_with(sub)


def test_grab_once_times_out_with_a_clear_error_and_still_unsubscribes(node):
    sub = MagicMock(name="subscription")
    node.create_subscription.side_effect = None
    node.create_subscription.return_value = sub
    with pytest.raises(RuntimeError, match="is the camera up and the link alive"):
        grab_once(node, limits.COLOR_IMAGE_TOPIC, Image, timeout=0.01)
    node.destroy_subscription.assert_called_once_with(sub)


# --- compressed colour (the physical robot over WiFi) -----------------------


def _jpeg_image(w: int = 8, h: int = 4):
    """A CompressedImage carrying a JPEG of a uniform mid-grey frame."""
    import cv2
    import numpy as np
    from sensor_msgs.msg import CompressedImage

    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    msg = CompressedImage()
    msg.format = "rgb8; jpeg compressed bgr8"
    msg.data = buf.tobytes()
    return msg


def test_compressed_colour_topic_subscribes_with_the_compressed_message_type(node, monkeypatch):
    from sensor_msgs.msg import CompressedImage

    compressed = limits.COLOR_IMAGE_RAW_TOPIC + "/compressed"
    monkeypatch.setattr(limits, "COLOR_IMAGE_TOPIC", compressed)
    feed = CameraFeed(node)
    feed.enable_color(True)
    msg_cls, topic, _cb, qos = node.create_subscription.call_args.args
    assert (msg_cls, topic, qos) == (CompressedImage, compressed, qos_profile_sensor_data)


def test_to_rgb_decodes_a_jpeg_compressed_frame():
    from wojtek_rai.camera_feed import _to_rgb

    rgb = _to_rgb(_jpeg_image(8, 4))
    assert rgb.shape == (4, 8, 3)
    assert abs(int(rgb.mean()) - 128) <= 2
