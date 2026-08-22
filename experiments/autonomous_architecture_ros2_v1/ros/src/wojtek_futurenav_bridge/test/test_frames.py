"""Image-to-JPEG conversion tests (duck-typed sensor_msgs Image)."""

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest

from wojtek_futurenav_bridge.frames import image_to_jpeg_b64


def fake_image(width=8, height=6, encoding="rgb8", pad=0):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = 200  # strongly red in rgb8 terms
    row = np.concatenate(
        [arr.reshape(height, -1), np.zeros((height, pad), dtype=np.uint8)], axis=1
    )
    return SimpleNamespace(
        width=width,
        height=height,
        step=width * 3 + pad,
        encoding=encoding,
        data=row.tobytes(),
    )


def decode(b64):
    from PIL import Image as PilImage

    return np.asarray(PilImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def test_rgb8_roundtrip_is_square_and_red():
    out = decode(image_to_jpeg_b64(fake_image(), frame_px=16))
    assert out.shape == (16, 16, 3)
    assert out[:, :, 0].mean() > 150
    assert out[:, :, 2].mean() < 60


def test_bgr8_channels_are_swapped():
    # The same buffer under bgr8: the "200" plane is blue, not red.
    out = decode(image_to_jpeg_b64(fake_image(encoding="bgr8"), frame_px=16))
    assert out[:, :, 2].mean() > 150
    assert out[:, :, 0].mean() < 60


def test_row_padding_is_stripped():
    out = decode(image_to_jpeg_b64(fake_image(pad=4), frame_px=16))
    assert out.shape == (16, 16, 3)
    assert out[:, :, 0].mean() > 150


def test_unknown_encoding_rejected():
    with pytest.raises(ValueError):
        image_to_jpeg_b64(fake_image(encoding="mono8"), frame_px=16)


def test_undersized_stride_rejected():
    msg = fake_image()
    msg.step = msg.width * 3 - 4  # malformed: row shorter than the pixels
    with pytest.raises(ValueError):
        image_to_jpeg_b64(msg, frame_px=16)
