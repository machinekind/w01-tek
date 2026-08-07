"""Rig ("tripod") camera for the simulated benchmark.

Same view-of-one-physics contract as wojtek_pc.sim_camera_node: the plant
inside ros2_control_node publishes /sim/qpos, this node mirrors it into its
own copy of the model -- the copy with the benchmark rig injected -- and
renders the fixed benchmark_rig_camera.  The rig camera is static; the
subscription exists because the *scene* moves: without qpos the robot (and
the tag on its back) would render frozen at spawn.

  subscribes  /sim/qpos                        (Float64MultiArray)
  publishes   /benchmark/camera/image_raw      mono8, full res -- the tracker's
              /benchmark/camera/camera_info    pinhole, zero distortion
              /benchmark/camera/preview        mono8, 1/4 res -- for HUMANS

mono8, not rgb8: the tracker grayscales anyway, and at 1080p the color
image would triple the DDS load for nothing (wojtek_perception_bringup's
README documents Python deserialization, not bandwidth, as the ceiling).

Look at /benchmark/camera/preview in Foxglove, not image_raw: the full-res
frame is ~2 MB and the foxglove_bridge websocket drops most of them, while
the preview is ~130 KB and survives.  Expect the configured hz only with
hardware GL; under llvmpipe (macOS Docker) a 1080p software render is the
bottleneck and the timer just runs late -- the node logs the achieved rate
so the slowdown is visible in its own words instead of as a mystery.
qpos goes in on the rendering side only -- the tag poses come back out
through pixels, which is what keeps the tracker sim-agnostic and the error
monitor's comparison against /sim/qpos non-circular.
"""

import os
import re
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from wojtek_pc import camera_spec

from wojtek_benchmark import sim_rig

FRAME_ID = "benchmark_camera_optical_frame"
IMAGE_TOPIC = "/benchmark/camera/image_raw"
INFO_TOPIC = "/benchmark/camera/camera_info"
PREVIEW_TOPIC = "/benchmark/camera/preview"
PREVIEW_STRIDE = 4


def _staged_scene(model_xml):
    """Stage the scene XMLs with meshdir rewritten to the installed meshes.

    Same move as wojtek_pc.sim_camera_node._staged_scene (kept local so
    wojtek_benchmark does not reach into another package's private helper).
    """
    scene = Path(model_xml)
    meshes = Path(get_package_share_directory("wojtek_description")) / "meshes"
    staged = Path(tempfile.mkdtemp(prefix="wojtek_benchmark_camera_"))
    for xml in scene.parent.glob("*.xml"):
        (staged / xml.name).write_text(
            re.sub(r'meshdir="[^"]*"', f'meshdir="{meshes}"', xml.read_text())
        )
    return str(staged / scene.name)


class BenchmarkCameraNode(Node):
    def __init__(self):
        super().__init__("wojtek_benchmark_camera")
        share = Path(get_package_share_directory("wojtek_benchmark"))
        self.declare_parameter("model_xml", "")
        self.declare_parameter("rig_config", str(share / "config" / "sim_rig.yaml"))
        self.declare_parameter("tags_config", str(share / "config" / "apriltags.yaml"))

        model_xml = self.get_parameter("model_xml").value
        if not model_xml:
            model_xml = os.path.join(
                get_package_share_directory("wojtek_pc"), "config", "scene_mjx.xml"
            )

        import mujoco

        rig_cfg = sim_rig.load_rig_config(self.get_parameter("rig_config").value)
        tags_cfg = sim_rig.load_tags_config(self.get_parameter("tags_config").value)
        cam = rig_cfg["camera"]
        self._size = (int(cam["width"]), int(cam["height"]))
        self._fovy = float(cam["fovy_deg"])

        self.model = sim_rig.load_model_with_rig(
            _staged_scene(model_xml), rig_cfg, tags_cfg
        )
        self.model.vis.global_.offwidth = max(
            self.model.vis.global_.offwidth, self._size[0]
        )
        self.model.vis.global_.offheight = max(
            self.model.vis.global_.offheight, self._size[1]
        )
        self._mujoco = mujoco
        self._data = mujoco.MjData(self.model)
        # Constructed here and used from timer callbacks: with rclpy's
        # default single-threaded executor both run on the spin thread, the
        # same thread-affinity deal sim_camera_node relies on.
        self._renderer = mujoco.Renderer(
            self.model, height=self._size[1], width=self._size[0]
        )

        self._qpos = None
        self._stamp = None
        self.create_subscription(
            Float64MultiArray, "sim/qpos", self._on_qpos, qos_profile_sensor_data
        )
        self._pub_image = self.create_publisher(
            Image, IMAGE_TOPIC, qos_profile_sensor_data
        )
        from sensor_msgs.msg import CameraInfo

        self._pub_info = self.create_publisher(
            CameraInfo, INFO_TOPIC, qos_profile_sensor_data
        )
        self._pub_preview = self.create_publisher(
            Image, PREVIEW_TOPIC, qos_profile_sensor_data
        )
        self._render_times = deque(maxlen=32)
        self._last_rate_log = time.monotonic()
        self._configured_hz = float(cam["hz"])
        self.create_timer(1.0 / self._configured_hz, self._render)
        self.get_logger().info(
            f"benchmark rig camera {self._size[0]}x{self._size[1]} @ "
            f"{cam['hz']:g} Hz mirroring /sim/qpos from {model_xml}"
        )

    def _on_qpos(self, msg):
        self._stamp = self.get_clock().now().to_msg()
        self._qpos = np.asarray(msg.data, dtype=np.float64)

    def _image_msg(self, gray, stamp):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = FRAME_ID
        msg.height, msg.width = gray.shape
        msg.encoding = "mono8"
        msg.is_bigendian = 0
        msg.step = gray.strides[0]
        msg.data = gray.tobytes()
        return msg

    def _render(self):
        if self._qpos is None:
            return
        if self._qpos.size != self.model.nq:
            self.get_logger().warning(
                f"/sim/qpos carries {self._qpos.size} values, the rig model "
                f"has {self.model.nq} -- same scene on both sides?",
                once=True,
            )
            return
        stamp = self._stamp
        t0 = time.monotonic()
        self._data.qpos[:] = self._qpos
        self._data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self._data)
        self._renderer.update_scene(self._data, camera=sim_rig.CAMERA_NAME)
        rgb = self._renderer.render()
        gray = np.ascontiguousarray(
            (rgb @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
        )

        self._pub_image.publish(self._image_msg(gray, stamp))
        self._pub_preview.publish(self._image_msg(
            np.ascontiguousarray(gray[::PREVIEW_STRIDE, ::PREVIEW_STRIDE]), stamp
        ))
        info = camera_spec.camera_info_msg(
            self._size[0], self._size[1], FRAME_ID, fovy_deg=self._fovy
        )
        info.header.stamp = stamp
        self._pub_info.publish(info)

        # Achieved rate, in the node's own log: under software GL the render
        # is the bottleneck and the timer just runs late -- say so instead of
        # letting a slow Foxglove image read as a transport mystery.
        self._render_times.append(time.monotonic() - t0)
        now = time.monotonic()
        if now - self._last_rate_log > 10.0 and len(self._render_times) >= 8:
            self._last_rate_log = now
            mean = sum(self._render_times) / len(self._render_times)
            self.get_logger().info(
                f"render {1000 * mean:.0f} ms/frame -> effective "
                f"{min(1.0 / mean, self._configured_hz):.1f} Hz "
                f"(configured {self._configured_hz:g} Hz)"
            )


def main():
    rclpy.init()
    node = BenchmarkCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._renderer.close()


if __name__ == "__main__":
    main()
