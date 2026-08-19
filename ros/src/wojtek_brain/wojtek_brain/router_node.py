"""Router node: final transcript in, routed intent out.

Rule router by default; set `model_path` to switch to the fine-tuned
encoder.  The transcript text passes through VERBATIM (#131 lesson: a
paraphrase drops the route), and low-confidence verdicts fall back to
`chat` — a wrong word is cheaper than a wrong walk.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from wojtek_agent_msgs.msg import RoutedIntent, Transcript

from .routing import EncoderRouter, RuleRouter, apply_confidence_floor


class RouterNode(Node):
    def __init__(self):
        super().__init__("wojtek_router")
        self.declare_parameter("model_path", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("min_confidence", 0.55)

        model_path = self.get_parameter("model_path").value
        if model_path:
            self.router = EncoderRouter(model_path, self.get_parameter("device").value)
            self.get_logger().info(f"encoder router: {model_path}")
        else:
            self.router = RuleRouter()
            self.get_logger().info("rule router (no model_path set)")
        self.min_confidence = self.get_parameter("min_confidence").value

        self.pub = self.create_publisher(RoutedIntent, "/wojtek/intent", 10)
        self.create_subscription(Transcript, "/wojtek/asr/final", self.on_final, 10)

    def on_final(self, msg: Transcript):
        intent, conf = self.router.classify(msg.text)
        intent, conf = apply_confidence_floor(intent, conf, self.min_confidence)
        out = RoutedIntent()
        out.header.stamp = self.get_clock().now().to_msg()
        out.utterance_id = msg.utterance_id
        out.text = msg.text
        out.intent = intent
        out.confidence = conf
        self.get_logger().info(f"{intent} ({conf:.2f}): {msg.text!r}")
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = RouterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
