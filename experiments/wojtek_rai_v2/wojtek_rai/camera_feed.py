"""Camera frames for the UI and the tools, with at most one reader per stream.

`CameraFeed` owns the colour/depth subscriptions of a node that is spun
elsewhere (`start_camera_feed` spins one on a daemon thread) and keeps the
newest raw sensor_msgs/Image of each stream with its arrival time. Nothing is
subscribed until `enable_color`/`enable_depth` is called, and switching a
stream off destroys its subscription: on the real robot the raw colour stream
is ~40 MB/s *per reader* over WiFi, so a reader may exist only while someone
is looking. The UI asks for `latest()` (decoded, for the sidebar); the tools
ask for `fresh()` and get an error, not a stale frame, when the link is down.

`grab_once` is the single-frame variant for a tool that looks rarely:
subscribe, wait for one message, unsubscribe. Neither helper goes through
RAI's `connector.receive_message`, which keeps its raw subscriber forever.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from wojtek_rai import limits

DEPTH_MAX_MM = 4000  # colour scale for the depth preview
# A cached frame older than this is refused by `fresh()`: the camera runs at
# 6-15 Hz, so anything older means the link (or the camera) is down.
MAX_FRAME_AGE_S = 1.0
GRAB_TIMEOUT_S = 5.0


def _message_class(topic: str):
    """The sensor_msgs type published on `topic`: CompressedImage for an
    image_transport `/compressed` topic, Image otherwise."""
    return CompressedImage if limits.is_compressed_topic(topic) else Image


def _decode_compressed(msg: CompressedImage) -> np.ndarray:
    """JPEG/PNG CompressedImage -> RGB array (OpenCV decodes to BGR)."""
    import cv2  # heavy import, only needed on this path

    bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"undecodable compressed image ({msg.format!r}, {len(msg.data)} bytes)")
    return bgr[:, :, ::-1]


def _to_rgb(msg: Any) -> np.ndarray:
    if isinstance(msg, CompressedImage):
        return _decode_compressed(msg)
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding == "rgb8":
        return arr.reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        return arr.reshape(msg.height, msg.width, 3)[:, :, ::-1]
    if msg.encoding == "mono8":
        return np.repeat(arr.reshape(msg.height, msg.width, 1), 3, axis=2)
    raise ValueError(f"unsupported colour encoding {msg.encoding}")


def _depth_to_rgb(msg: Image) -> np.ndarray:
    if msg.encoding == "16UC1":
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32)
    elif msg.encoding == "32FC1":
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width) * 1000.0
    else:
        raise ValueError(f"unsupported depth encoding {msg.encoding}")
    valid = depth > 0
    scaled = np.clip(depth / DEPTH_MAX_MM, 0.0, 1.0)
    # near = bright, far = dark, invalid = black
    gray = np.where(valid, (255 * (1.0 - scaled)).astype(np.uint8), 0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def grab_once(node: Node, topic: str, msg_cls: type, timeout: float = GRAB_TIMEOUT_S) -> Any:
    """One message from `topic`: subscribe (sensor-data QoS), wait, unsubscribe.

    `node` must be spun by an executor on another thread (the RAI connector's
    node and the camera-feed node both are). The subscription is destroyed on
    every exit path, so the stream flows for about one frame period per call.
    """
    got: list = []
    done = threading.Event()

    def _first(msg: Any) -> None:
        if not got:
            got.append(msg)
            done.set()

    sub = node.create_subscription(msg_cls, topic, _first, qos_profile_sensor_data)
    try:
        if not done.wait(timeout):
            raise RuntimeError(
                f"no message on {topic} in {timeout:.0f} s; is the camera up and the link alive?"
            )
        return got[0]
    finally:
        node.destroy_subscription(sub)


class CameraFeed:
    """Newest colour/depth frame of a node's streams; subscribed only on demand.

    `enable_*` are meant to be called from one thread (the UI); the message
    callbacks run on the executor thread and only touch `_msgs` under the lock.
    """

    def __init__(self, node: Node) -> None:
        self._node = node
        self._lock = threading.Lock()
        self._msgs: Dict[str, Tuple[Any, float]] = {}   # topic -> (raw msg, arrival time)
        self._subs: Dict[str, Any] = {}                  # topic -> live subscription

    @property
    def node(self) -> Node:
        return self._node

    # --- subscription lifetime --------------------------------------------

    def enable_color(self, on: bool) -> None:
        self._set_stream(limits.COLOR_IMAGE_TOPIC, on)

    def enable_depth(self, on: bool) -> None:
        self._set_stream(limits.DEPTH_IMAGE_TOPIC, on)

    def is_enabled(self, topic: str) -> bool:
        return topic in self._subs

    def _set_stream(self, topic: str, on: bool) -> None:
        sub = self._subs.get(topic)
        if on and sub is None:
            self._subs[topic] = self._node.create_subscription(
                _message_class(topic), topic, self._receiver(topic), qos_profile_sensor_data
            )
        elif not on and sub is not None:
            del self._subs[topic]
            self._node.destroy_subscription(sub)
            # Drop the cached frame: with no reader it would only get staler.
            with self._lock:
                self._msgs.pop(topic, None)

    def _receiver(self, topic: str) -> Callable[[Any], None]:
        def _on_msg(msg: Any) -> None:
            with self._lock:
                self._msgs[topic] = (msg, time.time())
        return _on_msg

    # --- frames -------------------------------------------------------------

    def _latest(self, topic: str) -> Tuple[Any, float]:
        """(raw msg or None, age in s; inf without a frame)."""
        with self._lock:
            entry = self._msgs.get(topic)
        if entry is None:
            return None, float("inf")
        msg, t = entry
        return msg, time.time() - t

    def latest_msgs(self) -> Tuple[Any, float, Any, float]:
        """(colour msg, colour_age_s, depth msg, depth_age_s); msgs may be None."""
        color, color_age = self._latest(limits.COLOR_IMAGE_TOPIC)
        depth, depth_age = self._latest(limits.DEPTH_IMAGE_TOPIC)
        return color, color_age, depth, depth_age

    def fresh(self, topic: str, max_age: float = MAX_FRAME_AGE_S) -> Any:
        """The newest message on `topic` if it arrived within `max_age`, else RuntimeError.

        Serving a stale frame would let the agent act on what the robot saw
        seconds ago; a flapping WiFi link must surface as an error instead.
        """
        if not self.is_enabled(topic):
            raise RuntimeError(f"{topic} is not enabled on the camera feed")
        msg, age = self._latest(topic)
        if msg is None:
            raise RuntimeError(f"no camera frame on {topic} yet; is the camera up and the link alive?")
        if age > max_age:
            raise RuntimeError(
                f"no fresh camera frame on {topic} (last one {age:.1f} s ago); is the link up?"
            )
        return msg

    def latest(self) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray], float]:
        """(colour, colour_age_s, depth, depth_age_s) decoded for the sidebar; frames may be None.

        Decoded here, at the UI's refresh rate, not per received frame.
        """
        color, color_age, depth, depth_age = self.latest_msgs()
        return (
            self._decode(color, _to_rgb), color_age,
            self._decode(depth, _depth_to_rgb), depth_age,
        )

    def _decode(self, msg: Any, convert: Callable[[Image], np.ndarray]) -> Optional[np.ndarray]:
        if msg is None:
            return None
        try:
            return convert(msg)
        except ValueError as exc:
            self._node.get_logger().warn(str(exc), throttle_duration_sec=10.0)
            return None


def start_camera_feed() -> CameraFeed:
    """Create the node, spin it on a daemon thread and wrap it. rclpy must be initialised.

    No stream is subscribed yet; the UI's toggles call `enable_color`/`enable_depth`.
    """
    node = rclpy.create_node("wojtek_rai_camera_feed")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, name="camera_feed_spin", daemon=True).start()
    return CameraFeed(node)
