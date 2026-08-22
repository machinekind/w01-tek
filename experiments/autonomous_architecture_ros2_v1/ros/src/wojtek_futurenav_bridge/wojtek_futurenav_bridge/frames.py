"""sensor_msgs Image -> JPEG base64, rclpy-free (duck-typed message).

Separate from the node shell so the conversion is unit-testable without a
ROS runtime; any object with height/width/step/encoding/data quacks enough.
"""

from __future__ import annotations

import base64
import io

import numpy as np


def image_to_jpeg_b64(msg, frame_px: int) -> str:
    """rgb8/bgr8 Image -> square JPEG of frame_px, base64-encoded.

    FutureNav was trained on square Habitat frames; the server resizes its
    input too, but resizing here keeps the payload small on a remote link.
    """
    from PIL import Image as PilImage  # deferred: heavy import, runtime-only

    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported encoding {msg.encoding!r}")
    if msg.step < msg.width * 3:
        raise ValueError(f"step {msg.step} too small for width {msg.width} rgb")
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    # step is bytes per row and may exceed width*3 (row padding).
    arr = arr.reshape(msg.height, msg.step)[:, : msg.width * 3]
    arr = arr.reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        arr = arr[:, :, ::-1]
    img = PilImage.fromarray(arr).resize((frame_px, frame_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()
