"""Interactive MuJoCo window mirroring the running simulation.

Same pattern as sim_camera_node: the physics lives in ros2_control_node
(the hardware plugin), which publishes /sim/qpos; this node mirrors that
into its own copy of the model and shows it in mujoco.viewer. A view of
the one physics state -- pan/orbit freely, nothing here feeds back into
the simulation.

    ros2 run wojtek_pc sim_viewer --ros-args -p model_xml:=<scene.xml>

Needs a display (GLFW); do NOT set MUJOCO_GL=egl for this one.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray

from wojtek_pc.sim_camera_node import _staged_scene


class SimViewerNode(Node):
    def __init__(self):
        super().__init__("wojtek_sim_viewer")
        self.declare_parameter("model_xml", "")
        self.qpos = None
        self.create_subscription(
            Float64MultiArray, "sim/qpos", self._on_qpos,
            qos_profile_sensor_data)

    def _on_qpos(self, msg):
        self.qpos = np.asarray(msg.data)


def main(args=None):
    import mujoco
    import mujoco.viewer

    rclpy.init(args=args)
    node = SimViewerNode()
    model_xml = node.get_parameter("model_xml").value
    if not model_xml:
        raise SystemExit("model_xml is required (the scene the plugin simulates)")
    model = mujoco.MjModel.from_xml_path(str(_staged_scene(model_xml)))
    data = mujoco.MjData(model)
    node.get_logger().info(f"mirroring /sim/qpos from {model_xml}")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.02)
                if node.qpos is not None and node.qpos.size == model.nq:
                    data.qpos[:] = node.qpos
                    mujoco.mj_forward(model, data)
                viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
