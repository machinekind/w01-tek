"""ROS shell for the VLM brain: instruction in, exploration loop out.

    ros2 run wojtek_nav vlm_brain_node --ros-args -p instruction:="podejdź do fioletowego słupa" \
        -p url:=http://127.0.0.1:11434/v1 -p model:=qwen3-vl:30b-a3b-instruct

Inputs
  wojtek/vlm/instruction  std_msgs/String: a new task (replaces the current one).
                          The `instruction` parameter runs one task at start.
  camera colour image     the picture the model sees (JPEG-encoded here).
  wojtek/nav/pixel_status, wojtek/nav/pixel_target, wojtek/nav/status
                          the resolver's and goto's answers, TF odom->base_link.
Outputs
  wojtek/nav/pixel_goal   the verified pixel (PointStamped, picture stamp).
  wojtek/nav/goal         straight approach/explore steps in base_link.
  cmd_vel                 turning in place (the only direct motion).
  wojtek/vlm/status       std_msgs/String, JSON per step, latched.
  wojtek/vlm/annotated    the picture with the model's point drawn.

The model is any OpenAI-compatible chat endpoint with JSON-schema structured
output (vLLM, Ollama). The policy itself is wojtek_nav/vlm_brain.py.
"""

import base64
import io
import json
import math
import time
import urllib.request

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from PIL import Image as PILImage, ImageDraw
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from wojtek_nav.vlm_brain import (
    SCHEMA,
    SYSTEM,
    VERIFY_SCHEMA,
    Explorer,
    parse_answer,
    task_prompt,
    verify_prompt,
)

PIXEL_TERMINAL = ("reached", "blocked", "timeout", "no_frame", "no_depth", "no_tf")


class VlmBrainNode(Node):
    def __init__(self):
        super().__init__("wojtek_vlm_brain")
        p = self.declare_parameter
        p("instruction", "")
        p("url", "http://127.0.0.1:11434/v1")
        p("model", "qwen3-vl:30b-a3b-instruct")
        p("image_topic", "/camera/camera/color/image_raw")
        p("odom_frame", "odom")
        p("base_frame", "base_link")
        p("turn_deg", 45.0)
        p("turn_cmd_rad_s", 0.5)      # commanded; the gait delivers less
        p("turn_real_rad_s", 0.38)    # measured in the sim on the stiff gait
        p("done_within_m", 1.1)       # standoff 0.7 + goto's tolerance + a little
        p("approach_m", 1.0)
        p("max_steps", 30)
        p("max_s", 600.0)
        p("frame_max_age_s", 0.8)
        p("request_timeout_s", 180.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self._g = g
        self.frame = None
        self.pixel_status = None
        self.goto_status = None
        self.target = None
        self.instruction = g("instruction")
        self.new_instruction = None
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Image, g("image_topic"), lambda m: setattr(self, "frame", m), qos_profile_sensor_data)
        self.create_subscription(String, "wojtek/nav/pixel_status",
                                 lambda m: setattr(self, "pixel_status", m.data), latched)
        self.create_subscription(String, "wojtek/nav/status",
                                 lambda m: setattr(self, "goto_status", m.data), latched)
        self.create_subscription(PointStamped, "wojtek/nav/pixel_target",
                                 lambda m: setattr(self, "target", m), latched)
        self.create_subscription(String, "wojtek/vlm/instruction", self._on_instruction, 10)
        self.pub_pixel = self.create_publisher(PointStamped, "wojtek/nav/pixel_goal", 10)
        self.pub_goal = self.create_publisher(PoseStamped, "wojtek/nav/goal", 10)
        self.pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)
        self.pub_status = self.create_publisher(String, "wojtek/vlm/status", latched)
        self.pub_annot = self.create_publisher(Image, "wojtek/vlm/annotated", 1)
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self, spin_thread=False)
        self.step_no = 0
        self.get_logger().info(f"vlm brain up: {g('model')} at {g('url')}")

    # -- plumbing --------------------------------------------------------

    def _on_instruction(self, msg):
        self.new_instruction = msg.data.strip()

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def fresh_frame(self, timeout=15.0):
        end = time.time() + timeout
        while time.time() < end:
            self.spin(0.05)
            f = self.frame
            if f is not None and self.now_s() - Time.from_msg(f.header.stamp).nanoseconds * 1e-9 < self._g("frame_max_age_s"):
                return f
        raise RuntimeError("no fresh colour frame")

    def status(self, **kw):
        kw.update(step=self.step_no, instruction=self.instruction, t=round(time.time(), 2))
        self.pub_status.publish(String(data=json.dumps(kw, ensure_ascii=False)))
        self.get_logger().info(json.dumps(kw, ensure_ascii=False))

    def distance_to_target(self):
        if self.target is None:
            return None
        try:
            t = self.tf.lookup_transform(self._g("odom_frame"), self._g("base_frame"), Time()).transform.translation
        except Exception:  # noqa: BLE001 -- tf2 raises several unrelated types
            return None
        return math.hypot(self.target.point.x - t.x, self.target.point.y - t.y)

    # -- the model -------------------------------------------------------

    def ask(self, frame, text, schema, max_tokens=60):
        rgb = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        img = PILImage.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        body = {
            "model": self._g("model"), "temperature": 0.0, "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}},
                    {"type": "text", "text": text}]},
            ],
            "response_format": {"type": "json_schema", "json_schema": {"name": "nav", "schema": schema, "strict": True}},
        }
        req = urllib.request.Request(self._g("url") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self._g("request_timeout_s")) as r:
            out = json.load(r)
        return parse_answer(out["choices"][0]["message"]["content"]), time.perf_counter() - t0, img

    def annotate(self, img, frame, point=None, text=""):
        im = img.copy()
        dr = ImageDraw.Draw(im)
        if point is not None:
            x, y = point[0] / 1000.0 * im.width, point[1] / 1000.0 * im.height
            r = 14
            dr.ellipse([x - r, y - r, x + r, y + r], outline=(255, 40, 0), width=4)
            dr.line([x - 2 * r, y, x + 2 * r, y], fill=(255, 40, 0), width=2)
            dr.line([x, y - 2 * r, x, y + 2 * r], fill=(255, 40, 0), width=2)
        if text:
            dr.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
            dr.text((6, 6), text[:120], fill=(255, 255, 255))
        out = Image()
        out.header = frame.header
        out.height, out.width, out.encoding, out.step = im.height, im.width, "rgb8", im.width * 3
        out.data = np.asarray(im, dtype=np.uint8).tobytes()
        self.pub_annot.publish(out)

    # -- primitives ------------------------------------------------------

    def turn(self, deg):
        tw = Twist()
        tw.angular.z = self._g("turn_cmd_rad_s") * (1.0 if deg > 0 else -1.0)
        end = time.time() + abs(math.radians(deg)) / self._g("turn_real_rad_s")
        while time.time() < end:
            self.pub_cmd.publish(tw)
            self.spin(0.05)
        self.pub_cmd.publish(Twist())
        self.spin(1.0)

    def step_forward(self, metres, frame):
        goal = PoseStamped()
        goal.header.frame_id = self._g("base_frame")
        goal.header.stamp = frame.header.stamp
        goal.pose.position.x = float(metres)
        goal.pose.orientation.w = 1.0
        t0, last, seen_active = time.time(), 0.0, False
        while time.time() - t0 < 25.0:
            if time.time() - last >= 1.0:  # goto's dead-man is 3 s
                self.pub_goal.publish(goal)
                last = time.time()
            self.spin(0.05)
            if self.goto_status in ("driving", "turning", "blocked"):
                seen_active = True
            if seen_active and self.goto_status in ("reached", "idle"):
                break
            if self.goto_status == "blocked" and time.time() - t0 > 6.0:
                break
        self.spin(1.0)
        return "reached" if self.goto_status in ("reached", "idle") and seen_active else (self.goto_status or "timeout")

    def send_pixel(self, frame, point):
        m = PointStamped()
        m.header = frame.header
        m.point.x, m.point.y, m.point.z = float(point[0]), float(point[1]), 0.0
        self.pixel_status = None
        self.target = None
        self.pub_pixel.publish(m)
        t0 = time.time()
        while time.time() - t0 < 80.0:
            self.spin(0.1)
            if self.pixel_status in PIXEL_TERMINAL:
                return self.pixel_status
        return "timeout"

    # -- the loop --------------------------------------------------------

    def run_task(self, instruction):
        self.instruction = instruction
        ex = Explorer(self._g("turn_deg"), self._g("done_within_m"), self._g("approach_m"),
                      max_steps=int(self._g("max_steps")))
        self.step_no = 0
        t_start = time.time()
        action = ("look",)
        frame = None
        while time.time() - t_start < self._g("max_s") and self.new_instruction is None:
            kind = action[0]
            if kind in Explorer.TERMINAL:
                self.status(action="finished", result=kind)
                return kind
            if kind == "look":
                self.step_no += 1
                frame = self.fresh_frame()
                answer, dt, img = self.ask(frame, task_prompt(instruction), SCHEMA)
                self.annotate(img, frame, answer.get("point_2d") if answer.get("type") == "goal" else None,
                              f"#{self.step_no} {answer.get('type')} {answer.get('label', '')} ({dt:.1f}s)")
                self.status(action="ask", answer=answer, latency_s=round(dt, 2))
                action = ex.on_answer(answer)
            elif kind == "verify":
                v, dt, _ = self.ask(frame, verify_prompt(instruction, action[1]), VERIFY_SCHEMA, 20)
                self.status(action="verify", answer=v, latency_s=round(dt, 2))
                action = ex.on_verified(bool(v.get("visible")))
            elif kind == "goal":
                res = self.send_pixel(frame, action[1])
                dist = self.distance_to_target()
                self.status(action="pixel_goal", result=res, distance_to_target_m=None if dist is None else round(dist, 2))
                action = ex.on_goal_result(res, dist)
            elif kind in ("approach", "explore"):
                res = self.step_forward(action[1], frame)
                self.status(action=kind, metres=action[1], result=res)
                action = ex.on_move_result(res)
            elif kind == "search":
                self.turn(action[1])
                self.status(action="turn", deg=action[1], turned_total=ex.turned_deg)
                action = ex.turn_done()
        if self.new_instruction is not None:
            self.status(action="finished", result="replaced")
            return "replaced"
        self.status(action="finished", result="timeout")
        return "timeout"

    def serve(self):
        """One task from the parameter, then whatever arrives on the topic."""
        self.spin(1.5)
        if self.instruction:
            self.run_task(self.instruction)
        while rclpy.ok():
            self.spin(0.2)
            if self.new_instruction:
                task, self.new_instruction = self.new_instruction, None
                self.run_task(task)


def main():
    rclpy.init()
    node = VlmBrainNode()
    try:
        node.serve()
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_cmd.publish(Twist())
        node.spin(0.3)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
