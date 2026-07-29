"""Depth image -> coarse terrain reference cloud.

Reduces the depth image to a small fixed grid (default 8x8 = 64 points) by
taking the **median** of each patch. That is deliberately not an average:
depth edges produce flying pixels, and a patch straddling an obstacle edge
should report the dominant surface rather than a blend of two ranges that
exists nowhere.

What the reduction does and does not buy, measured on the bench (see
docs/perception/d435-noise.md):

- It does NOT denoise. Collapsing 6360 pixels into one point removed only
  1.8x of the temporal noise, where an uncorrelated sensor would give 80x.
  RealSense stereo noise is strongly correlated in space -- whole patches
  wobble together -- so no amount of downsampling flattens it. Temporal
  averaging from a moving viewpoint is the lever that works.
- It DOES fix coverage. Source fill was 74% (HIGH_ACCURACY trades fill for
  accuracy); the reduced grid comes out at ~100%, because a patch of
  thousands of pixels always has something valid in it. This is why the
  accurate-but-holey preset is the right choice upstream.

Reducing from the depth image rather than from a PointCloud2 is intentional:
the driver's cloud would be ~400k points serialised before we threw 99.98%
of them away.

Publishes an ORDERED cloud (height=grid_h, width=grid_w) with NaN in cells
that had too few valid pixels, so consumers that want a fixed-size tensor
get one and consumers that want a point list filter the NaNs.
"""

import warnings

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField


class CloudReduce(Node):
    def __init__(self):
        super().__init__("cloud_reduce")
        self.declare_parameter("grid_w", 8)
        self.declare_parameter("grid_h", 8)
        self.declare_parameter("min_range", 0.3)
        self.declare_parameter("max_range", 3.0)
        self.declare_parameter("min_valid_frac", 0.05)

        self._info = None
        self._warned_shape = False

        # Sensor-data QoS (best effort): compatible with the driver whether it
        # publishes best-effort or reliable.
        self.create_subscription(
            CameraInfo, "depth/camera_info", self._on_info, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, "depth/image", self._on_depth, qos_profile_sensor_data
        )
        self._pub = self.create_publisher(
            PointCloud2, "terrain_points", qos_profile_sensor_data
        )
        self.get_logger().info("cloud_reduce waiting for depth/camera_info")

    def _on_info(self, msg: CameraInfo) -> None:
        self._info = msg

    def _on_depth(self, msg: Image) -> None:
        if self._info is None:
            return
        depth = self._decode(msg)
        if depth is None:
            return

        gw = self.get_parameter("grid_w").value
        gh = self.get_parameter("grid_h").value
        lo = self.get_parameter("min_range").value
        hi = self.get_parameter("max_range").value
        min_frac = self.get_parameter("min_valid_frac").value

        h, w = depth.shape
        ph, pw = h // gh, w // gw
        if ph == 0 or pw == 0:
            self.get_logger().error(
                f"grid {gw}x{gh} is finer than the {w}x{h} depth image"
            )
            return

        # Crop the remainder so the patches tile exactly, then reshape so the
        # patch axes are separate and can be reduced in one pass.
        valid = (depth >= lo) & (depth <= hi)
        z = np.where(valid, depth, np.nan)[: gh * ph, : gw * pw]
        patches = z.reshape(gh, ph, gw, pw)

        # An all-invalid patch is the normal case (sky, out of range), not an
        # anomaly -- silence numpy's warning so it cannot spam the log at the
        # frame rate.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            z_med = np.nanmedian(patches, axis=(1, 3))
        frac = np.mean(np.isfinite(patches), axis=(1, 3))
        z_med = np.where(frac >= min_frac, z_med, np.nan)

        self._pub.publish(self._to_cloud(z_med, msg.header, ph, pw, gw, gh))

    def _decode(self, msg: Image):
        """Depth image -> float32 metres, or None if the encoding is unknown."""
        if msg.encoding == "16UC1":
            raw = np.frombuffer(msg.data, dtype=np.uint16)
            depth = raw.reshape(msg.height, msg.width).astype(np.float32) * 0.001
        elif msg.encoding == "32FC1":
            raw = np.frombuffer(msg.data, dtype=np.float32)
            depth = raw.reshape(msg.height, msg.width)
        else:
            self.get_logger().error(f"unsupported depth encoding {msg.encoding!r}")
            return None

        if (msg.width, msg.height) != (self._info.width, self._info.height):
            if not self._warned_shape:
                self.get_logger().warn(
                    f"depth {msg.width}x{msg.height} does not match camera_info "
                    f"{self._info.width}x{self._info.height}; deprojection will be "
                    "wrong. Post-processing that resizes the image (decimation) "
                    "must republish a matching camera_info."
                )
                self._warned_shape = True
            return None
        return depth

    def _to_cloud(self, z_med, header, ph, pw, gw, gh) -> PointCloud2:
        """Deproject patch centres through the camera intrinsics."""
        fx, fy = self._info.k[0], self._info.k[4]
        cx, cy = self._info.k[2], self._info.k[5]

        # Pixel coordinate of each patch centre.
        u = (np.arange(gw) + 0.5) * pw
        v = (np.arange(gh) + 0.5) * ph
        uu, vv = np.meshgrid(u, v)

        x = (uu - cx) / fx * z_med
        y = (vv - cy) / fy * z_med
        points = np.stack([x, y, z_med], axis=-1).astype(np.float32)

        msg = PointCloud2()
        msg.header = header          # depth optical frame, x right / y down / z fwd
        msg.height, msg.width = gh, gw
        msg.fields = [
            PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
            for i, n in enumerate(("x", "y", "z"))
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * gw
        msg.data = points.tobytes()
        msg.is_dense = False         # NaN where the patch had too little data
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = CloudReduce()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
