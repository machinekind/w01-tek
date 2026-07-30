"""Unit tests for the depth->grid reduction. No camera, no running graph.

Feeds synthetic depth images straight into the callbacks and inspects what
the node would have published.
"""

import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image

from wojtek_perception_bringup.cloud_reduce_node import CloudReduce

W, H = 424, 240
FX = FY = 209.0
CX, CY = W / 2.0, H / 2.0


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = CloudReduce()
    published = []
    n._pub.publish = published.append          # capture instead of publishing
    n.published = published
    info = CameraInfo()
    info.width, info.height = W, H
    info.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
    n._on_info(info)
    yield n
    n.destroy_node()


def _depth_msg(depth_m: np.ndarray) -> Image:
    msg = Image()
    msg.width, msg.height = W, H
    msg.encoding = "16UC1"
    msg.step = W * 2
    msg.data = (depth_m * 1000.0).astype(np.uint16).tobytes()
    return msg


def _points(msg):
    return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width, 3)


def test_flat_wall_reduces_to_that_range(node):
    node._on_depth(_depth_msg(np.full((H, W), 2.0, np.float32)))

    assert len(node.published) == 1
    cloud = node.published[0]
    assert (cloud.height, cloud.width) == (8, 8)
    xyz = _points(cloud)
    assert xyz.shape == (8, 8, 3)
    assert np.allclose(xyz[..., 2], 2.0, atol=1e-3)


def test_deprojection_is_centred_and_scales_with_range(node):
    node._on_depth(_depth_msg(np.full((H, W), 2.0, np.float32)))
    xyz = _points(node.published[0])

    # The grid straddles the principal point, so the two central columns sit
    # symmetrically either side of x=0 (same for rows about y=0).
    assert np.allclose(xyz[:, 3, 0], -xyz[:, 4, 0], atol=1e-4)
    assert np.allclose(xyz[3, :, 1], -xyz[4, :, 1], atol=1e-4)

    # Lateral offset is proportional to range: double the depth, double x.
    node.published.clear()
    node._on_depth(_depth_msg(np.full((H, W), 1.0, np.float32)))
    near = _points(node.published[0])
    assert np.allclose(near[..., 0] * 2.0, xyz[..., 0], atol=1e-4)


def test_out_of_range_returns_become_nan(node):
    depth = np.full((H, W), 2.0, np.float32)
    depth[:, : W // 8] = 0.05          # nearer than min_range
    depth[:, -W // 8:] = 9.0           # further than max_range
    node._on_depth(_depth_msg(depth))

    xyz = _points(node.published[0])
    assert np.isnan(xyz[:, 0, 2]).all()
    assert np.isnan(xyz[:, -1, 2]).all()
    assert np.isfinite(xyz[:, 1:-1, 2]).all()


def test_sparse_patch_below_min_valid_frac_is_dropped(node):
    """A patch with a couple of stray returns must not become a confident point."""
    depth = np.zeros((H, W), np.float32)
    depth[0, 0] = 2.0                  # one valid pixel in the first patch only
    node._on_depth(_depth_msg(depth))

    xyz = _points(node.published[0])
    assert np.isnan(xyz).all()


def test_median_ignores_a_flying_pixel_edge(node):
    """Median, not mean: an edge patch reports the dominant surface."""
    depth = np.full((H, W), 2.0, np.float32)
    patch_w = W // 8
    depth[:, :patch_w][:, : patch_w // 4] = 0.5     # a quarter of patch 0 is near

    node._on_depth(_depth_msg(depth))
    xyz = _points(node.published[0])
    # A mean would land near 1.63 m; the median stays on the dominant surface.
    assert np.allclose(xyz[:, 0, 2], 2.0, atol=1e-3)


def test_unknown_encoding_is_reported_not_crashed(node):
    msg = _depth_msg(np.full((H, W), 2.0, np.float32))
    msg.encoding = "mono8"
    node._on_depth(msg)
    assert node.published == []


def test_camera_info_mismatch_does_not_publish_wrong_geometry(node):
    """Decimation that resizes the image without a matching info must not
    silently deproject through stale intrinsics."""
    msg = _depth_msg(np.full((H, W), 2.0, np.float32))
    msg.width, msg.height = W // 2, H // 2
    msg.data = np.full((H // 2, W // 2), 2000, np.uint16).tobytes()
    node._on_depth(msg)
    assert node.published == []


def test_output_is_a_private_topic(node):
    """The published interface: ~/terrain_points, i.e. namespaced under the
    node like the rest of the stack's sensor output (magnetometer_broadcaster
    publishes ~/magnetic_field). Consumers hard-code this name."""
    assert node._pub.topic_name == "/cloud_reduce/terrain_points"
