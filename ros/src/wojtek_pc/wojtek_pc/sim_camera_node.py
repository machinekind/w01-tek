"""Virtual D435 for the ros2_control simulation.

The physics now lives inside ros2_control_node (the MuJoCo hardware plugin),
and the renderer is Python, so the two cannot share one MjData. Instead the
plugin publishes the generalized position it just stepped -- /sim/qpos -- and
this node mirrors it into its own copy of the model, the one with the camera
injected, and renders from there. One physics state remains the source of
truth; this is a view of it, not a second simulation.

The camera contract (topics, encodings, intrinsics, frames) is
wojtek_pc.camera_spec, shared with the real perception stack, so
cloud_reduce/the planner/the VLM see in simulation exactly what they see on
the robot.

  subscribes  /sim/qpos  (Float64MultiArray, from the hardware plugin)
  publishes   /camera/camera/depth/image_rect_raw   16UC1 mm
              /camera/camera/depth/camera_info
              /camera/camera/color/image_raw        rgb8
              /camera/camera/color/camera_info

One deviation from the in-process renderer it replaces: image stamps are the
time the pose message arrived, not the physics tick that produced it. The two
differ by the transport latency of one small message on localhost, which is
below anything the depth consumers resolve -- but it is a deviation, and
docs/sim-test-contract.md says so.
"""

import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64MultiArray

from wojtek_pc import camera_spec


def _staged_scene(model_xml):
    """Copy the scene somewhere private, with meshdir pointing at the meshes.

    The MJX XMLs ship in wojtek_pc's share, the meshes in wojtek_description's,
    and the robot file's meshdir is relative to itself -- so it has to be
    rewritten before MuJoCo can load it. The hardware plugin does the same on
    its side (see MujocoPlant::load); this is the Python half.

    Anything else lying next to the scene comes along untouched, because a
    scene may bring assets of its own: scene_sim.xml's props/ holds the
    pictures its signs wear, and those are found relative to the file, not
    through meshdir.
    """
    scene = Path(model_xml)
    meshes = Path(get_package_share_directory("wojtek_description")) / "meshes"
    staged = Path(tempfile.mkdtemp(prefix="wojtek_sim_camera_"))
    for entry in scene.parent.iterdir():
        if entry.is_dir():
            shutil.copytree(entry, staged / entry.name)
        elif entry.suffix == ".xml":
            (staged / entry.name).write_text(
                re.sub(r'meshdir="[^"]*"', f'meshdir="{meshes}"', entry.read_text())
            )
        else:
            shutil.copy2(entry, staged / entry.name)
    return str(staged / scene.name)


class SimCameraNode(Node):
    def __init__(self):
        super().__init__("wojtek_sim_camera")
        self.declare_parameter("model_xml", "")
        self.declare_parameter("depth_hz", 15.0)
        self.declare_parameter("color_hz", 5.0)
        self.declare_parameter("depth_min_m", camera_spec.DEPTH_MIN_M)
        self.declare_parameter("depth_max_m", camera_spec.DEPTH_MAX_M)

        model_xml = self.get_parameter("model_xml").value
        if not model_xml or not os.path.exists(model_xml):
            raise RuntimeError(
                f"model_xml must point at the scene the plugin simulates, got "
                f"{model_xml!r}"
            )

        # Rendering needs a GL backend; physics does not, which is why this is
        # a separate node -- a broken backend takes the camera down and leaves
        # the control stack running.
        from wojtek_pc.depth_camera import SimDepthCamera, load_model_with_camera

        self.model = load_model_with_camera(_staged_scene(model_xml))
        self._cam = SimDepthCamera(
            self.model,
            min_m=self.get_parameter("depth_min_m").value,
            max_m=self.get_parameter("depth_max_m").value,
        )
        self._cam.start()

        self._qpos = None
        self._stamp = None
        self.create_subscription(
            Float64MultiArray, "sim/qpos", self._on_qpos, qos_profile_sensor_data
        )
        self._pub_depth = self.create_publisher(
            Image, camera_spec.DEPTH_TOPIC, qos_profile_sensor_data
        )
        self._pub_depth_info = self.create_publisher(
            CameraInfo, camera_spec.DEPTH_INFO_TOPIC, qos_profile_sensor_data
        )
        self._pub_color = self.create_publisher(
            Image, camera_spec.COLOR_TOPIC, qos_profile_sensor_data
        )
        self._pub_color_info = self.create_publisher(
            CameraInfo, camera_spec.COLOR_INFO_TOPIC, qos_profile_sensor_data
        )

        depth_hz = self.get_parameter("depth_hz").value
        color_hz = self.get_parameter("color_hz").value
        if depth_hz > 0:
            self.create_timer(1.0 / depth_hz, self._render_depth)
        if color_hz > 0:
            self.create_timer(1.0 / color_hz, self._render_color)
        self.get_logger().info(
            f"virtual camera mirroring /sim/qpos from {model_xml} "
            f"(depth {depth_hz:g} Hz, color {color_hz:g} Hz)"
        )

    def _on_qpos(self, msg):
        # Stamp on arrival: the pose is whatever the plugin last stepped.
        self._stamp = self.get_clock().now().to_msg()
        self._qpos = np.asarray(msg.data, dtype=np.float64)

    def _pose(self):
        if self._qpos is None or self._qpos.size != self.model.nq:
            if self._qpos is not None:
                self.get_logger().warning(
                    f"/sim/qpos carries {self._qpos.size} values, the render "
                    f"model has {self.model.nq} -- same scene on both sides?",
                    once=True,
                )
            return None
        return self._qpos, np.zeros(self.model.nv), self._stamp

    def _image_msg(self, array, stamp, frame_id, encoding):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height, msg.width = array.shape[0], array.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = array.strides[0]
        msg.data = array.tobytes()
        return msg

    def _publish(self, array, stamp, frame_id, encoding, image_pub, info_pub):
        image_pub.publish(self._image_msg(array, stamp, frame_id, encoding))
        info = camera_spec.camera_info_msg(array.shape[1], array.shape[0], frame_id)
        info.header.stamp = stamp
        info_pub.publish(info)

    def _render_depth(self):
        pose = self._pose()
        if pose is None:
            return
        qpos, qvel, stamp = pose
        self._publish(
            self._cam.render_depth(qpos, qvel), stamp,
            camera_spec.DEPTH_FRAME_ID, "16UC1",
            self._pub_depth, self._pub_depth_info,
        )

    def _render_color(self):
        pose = self._pose()
        if pose is None:
            return
        qpos, qvel, stamp = pose
        rgb = np.ascontiguousarray(self._cam.render_color(qpos, qvel))
        self._publish(
            rgb, stamp, camera_spec.COLOR_FRAME_ID, camera_spec.COLOR_ENCODING,
            self._pub_color, self._pub_color_info,
        )


def main():
    rclpy.init()
    node = SimCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cam.close()


if __name__ == "__main__":
    main()
