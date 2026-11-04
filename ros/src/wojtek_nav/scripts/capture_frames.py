#!/usr/bin/env python3
"""Capture one synchronized colour/depth pair for point_bench.py.

Run inside the ROS container against a stationary simulation. Captures are
unlabelled: review the JPEG and annotate index.json before benchmarking.
"""

import argparse
from collections import deque
from io import BytesIO
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np


def stamp_seconds(msg):
    stamp = msg.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


def nearest_depth(colour, depths, max_skew):
    if not depths:
        return None
    stamp = stamp_seconds(colour)
    candidate = min(depths, key=lambda msg: abs(stamp_seconds(msg) - stamp))
    return candidate if abs(stamp_seconds(candidate) - stamp) <= max_skew else None


def depth_array(msg):
    """Read 16-bit millimetres, respecting ROS row padding and endianness."""
    if msg.encoding != "16UC1":
        raise ValueError(f"expected 16UC1 depth in millimetres, got {msg.encoding!r}")
    if msg.width <= 0 or msg.height <= 0 or msg.step < 2 * msg.width:
        raise ValueError("invalid depth dimensions or row stride")
    if len(msg.data) != msg.step * msg.height:
        raise ValueError("depth buffer length does not match its stride")
    dtype = np.dtype(">u2" if msg.is_bigendian else "<u2")
    return np.ndarray((msg.height, msg.width), dtype=dtype, buffer=bytes(msg.data),
                      strides=(msg.step, 2)).astype(np.uint16, copy=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--max-skew", type=float, default=0.1)
    parser.add_argument("--colour-topic", default="/camera/camera/color/image_raw/compressed")
    parser.add_argument("--depth-topic", default="/camera/camera/depth/image_rect_raw")
    parser.add_argument("--colour-info", default="/camera/camera/color/camera_info")
    parser.add_argument("--depth-info", default="/camera/camera/depth/camera_info")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    if not math.isfinite(args.max_skew) or args.max_skew < 0:
        parser.error("max-skew must be nonnegative and finite")

    from PIL import Image
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, CompressedImage, Image as RosImage

    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.json"
    entries = json.loads(index_path.read_text()) if index_path.exists() else []
    if not isinstance(entries, list):
        parser.error("index.json must contain a list")
    index = max((entry["i"] for entry in entries), default=-1) + 1
    rclpy.init(args=[])
    node = Node("wojtek_point_capture")
    frames = {"colour": None, "colour_info": None, "depth_info": None}
    depths = deque(maxlen=30)
    subscriptions = [
        node.create_subscription(CompressedImage, args.colour_topic,
                                 lambda msg: frames.update(colour=msg), qos_profile_sensor_data),
        node.create_subscription(RosImage, args.depth_topic, depths.append, qos_profile_sensor_data),
        node.create_subscription(CameraInfo, args.colour_info,
                                 lambda msg: frames.update(colour_info=msg), qos_profile_sensor_data),
        node.create_subscription(CameraInfo, args.depth_info,
                                 lambda msg: frames.update(depth_info=msg), qos_profile_sensor_data),
    ]
    try:
        deadline = time.monotonic() + args.timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if any(value is None for value in frames.values()):
                continue
            colour = frames["colour"]
            depth = nearest_depth(colour, depths, args.max_skew)
            if depth is None:
                continue
            jpeg = bytes(colour.data)
            with Image.open(BytesIO(jpeg)) as image:
                if image.format != "JPEG":
                    raise ValueError("colour stream must contain JPEG frames")
                width, height = image.size
            kc, kd = frames["colour_info"], frames["depth_info"]
            if (kc.width, kc.height) != (width, height):
                raise ValueError("colour CameraInfo dimensions do not match JPEG")
            if (kd.width, kd.height) != (depth.width, depth.height):
                raise ValueError("depth CameraInfo dimensions do not match image")
            values = depth_array(depth)
            # Exclusive creation preserves an existing fixture on accidental reuse.
            with (args.out / f"frame_{index:02d}.jpg").open("xb") as output:
                output.write(jpeg)
            with (args.out / f"frame_{index:02d}_depth.npy").open("xb") as output:
                np.save(output, values, allow_pickle=False)
            entries.append({
                "i": index, "colour_width": width, "colour_height": height,
                "colour_stamp": stamp_seconds(colour), "depth_stamp": stamp_seconds(depth),
                "colour_frame": colour.header.frame_id, "depth_frame": depth.header.frame_id,
                "k_colour": list(kc.k), "k_depth": list(kd.k),
                "annotation_status": "unreviewed", "truth": {},
            })
            with tempfile.NamedTemporaryFile(mode="w", dir=args.out, prefix=".index-",
                                             suffix=".json", delete=False) as output:
                json.dump(entries, output, indent=2)
                output.write("\n")
                temporary_index = Path(output.name)
            temporary_index.replace(index_path)
            print(f"Captured frame {index}; annotate {index_path} before benchmarking.")
            return 0
        raise RuntimeError("no synchronized colour/depth pair with CameraInfo before timeout")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
