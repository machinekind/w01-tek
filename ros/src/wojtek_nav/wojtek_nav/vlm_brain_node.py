"""ROS shell for the VLM brain: instruction in, exploration loop out.

    ros2 launch wojtek_nav brain.launch.py url:=http://<ollama host>:11434 \
        [model:=qwen3-vl:30b-a3b-instruct] [instruction:="podejdź do fioletowego słupa"]
    ros2 run wojtek_nav vlm_brain_node --ros-args -p url:=... -p instruction:=...

Inputs
  wojtek/vlm/instruction  std_msgs/String: a new task (replaces the current one).
                          An empty string or "stop" cancels: the goal in
                          flight is dropped (wojtek/nav/cancel), a turn in
                          progress gets its zero Twist, the brain goes idle.
                          The `instruction` parameter runs one task at start.
  camera colour image     the picture the model sees. By default the camera
                          node's own JPEG (<image_topic>/compressed,
                          image_transport's plugin): ~40-120 KB a frame
                          across the robot's wifi instead of the raw
                          image's 1-3 MB. compressed:=false takes the raw
                          image and encodes here (the sim without the
                          plugin, a robot without it).
  colour camera_info      the intrinsics that turn the model's pixel into a
                          bearing when the resolver has no depth for it
                          (<image_topic dir>/camera_info by default).
  wojtek/nav/pixel_status, wojtek/nav/pixel_target, wojtek/nav/status
                          the resolver's and goto's answers, TF odom->base_link
                          and odom->the camera's optical frame.
  wojtek/nav/cancel       a cancel from anyone else (the console's STOP, a
                          hand-typed one) ends the running task too; the
                          brain must not answer a stopped goto with a turn.
Outputs
  wojtek/nav/pixel_goal   the verified pixel (PointStamped, picture stamp).
  wojtek/nav/goal         approach/explore steps, in odom: fixed once, so a
                          re-send never depends on the picture's stamp
                          still being in goto's TF buffer.
  wojtek/nav/cancel       std_msgs/Empty on stop/replace: goto and the
                          resolver drop what they hold.
  cmd_vel                 turning in place (the only direct motion).
  wojtek/vlm/status       std_msgs/String, JSON per step, latched.
  wojtek/vlm/annotated    the picture with the model's point drawn.

The model is Qwen3-VL 30B-A3B (qwen3-vl:30b-a3b-instruct) served by Ollama
through its OpenAI-compatible chat endpoint with JSON-schema structured
output. `url` is the server's base URL, with or without /v1. The policy itself is wojtek_nav/vlm_brain.py.
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
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformListener

from wojtek_nav.vlm_brain import (
    DEFAULT_MODEL,
    DEFAULT_URL,
    SCHEMA,
    SYSTEM,
    VERIFY_SCHEMA,
    Explorer,
    chat_url,
    parse_answer,
    pixel_ray,
    ray_heading,
    step_along,
    task_prompt,
    verify_prompt,
)

PIXEL_TERMINAL = ("reached", "blocked", "timeout", "no_frame", "no_depth", "no_tf", "cancelled")
STOP_WORDS = ("", "stop", "stój", "stop.")


class Interrupted(Exception):
    """A new instruction (or a stop) arrived in the middle of a step."""


class VlmBrainNode(Node):
    def __init__(self):
        super().__init__("wojtek_vlm_brain")
        p = self.declare_parameter
        p("instruction", "")
        p("url", DEFAULT_URL)
        p("model", DEFAULT_MODEL)
        p("image_topic", "/camera/camera/color/image_raw")
        p("colour_info_topic", "")      # "": <image_topic dir>/camera_info
        p("compressed", True)           # <image_topic>/compressed, the camera's own JPEG
        p("odom_frame", "odom")
        p("base_frame", "base_link")
        p("turn_deg", 45.0)
        p("turn_cmd_rad_s", 0.5)      # commanded; the gait delivers less
        p("turn_real_rad_s", 0.38)    # measured in the sim on the stiff gait
        p("done_within_m", 1.1)       # standoff 0.7 + goto's tolerance + a little
        p("approach_m", 1.0)
        p("max_steps", 30)
        p("max_blind", 8)             # approaches in a row with no depth on the target
        p("min_progress_m", 0.25)     # a move shorter than this is a stall
        p("max_stalls", 3)            # stalls in a row end the task
        p("max_s", 600.0)
        p("frame_max_age_s", 0.8)
        p("request_timeout_s", 180.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self._g = g
        self.frame = None             # sensor_msgs Image or CompressedImage, the latest
        self.pixel_status = None
        self.goto_status = None
        self.target = None
        self.colour_info = None
        self.instruction = g("instruction")
        self.new_instruction = None   # None: nothing pending; "": stop; text: next task
        # Empty strings mean the defaults: a launch that forwards VLM_URL /
        # VLM_MODEL from the environment hands over "" when the variable is
        # set but blank (a .env line left empty, `export VLM_URL=`).
        self.model = g("model") or DEFAULT_MODEL
        self.endpoint = chat_url(g("url") or DEFAULT_URL)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        if g("compressed"):
            self.image_topic = g("image_topic").rstrip("/") + "/compressed"
            self.create_subscription(CompressedImage, self.image_topic,
                                     lambda m: setattr(self, "frame", m), qos_profile_sensor_data)
        else:
            self.image_topic = g("image_topic")
            self.create_subscription(Image, self.image_topic,
                                     lambda m: setattr(self, "frame", m), qos_profile_sensor_data)
        info_topic = g("colour_info_topic") or g("image_topic").rsplit("/", 1)[0] + "/camera_info"
        self.create_subscription(CameraInfo, info_topic,
                                 lambda m: setattr(self, "colour_info", m), qos_profile_sensor_data)
        self.create_subscription(String, "wojtek/nav/pixel_status",
                                 lambda m: setattr(self, "pixel_status", m.data), latched)
        self.create_subscription(String, "wojtek/nav/status",
                                 lambda m: setattr(self, "goto_status", m.data), latched)
        self.create_subscription(PointStamped, "wojtek/nav/pixel_target",
                                 lambda m: setattr(self, "target", m), latched)
        self.create_subscription(String, "wojtek/vlm/instruction", self._on_instruction, 10)
        self.create_subscription(Empty, "wojtek/nav/cancel", self._on_cancel, 10)
        self._running = False
        self._halting = False         # halt() is publishing its own cancel
        self.pub_pixel = self.create_publisher(PointStamped, "wojtek/nav/pixel_goal", 10)
        self.pub_goal = self.create_publisher(PoseStamped, "wojtek/nav/goal", 10)
        self.pub_cancel = self.create_publisher(Empty, "wojtek/nav/cancel", 10)
        self.pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)
        self.pub_status = self.create_publisher(String, "wojtek/vlm/status", latched)
        self.pub_annot = self.create_publisher(Image, "wojtek/vlm/annotated", 1)
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self, spin_thread=False)
        self.step_no = 0
        self.get_logger().info(
            f"vlm brain up: {self.model} at {self.endpoint}, pictures from {self.image_topic}")
        self.status(action="idle")

    # -- plumbing --------------------------------------------------------

    def _on_instruction(self, msg):
        text = msg.data.strip()
        self.new_instruction = "" if text.lower() in STOP_WORDS else text

    def _on_cancel(self, _msg):
        # Somebody stopped goto and the resolver: the task is over. Only
        # while a task runs (an idle brain publishes its own cancel and
        # must not chase its echo), never over a pending instruction (a
        # replacement's own cancel would otherwise wipe the new text), and
        # not while halt() is out: its cancel comes back on this very
        # subscription, and taken for a stop it turned a task that ended
        # in `error` into a latched `idle` 0.2 s later.
        if self._running and not self._halting and self.new_instruction is None:
            self.new_instruction = ""

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def check_interrupt(self):
        """Every waiting loop asks; a pending instruction ends the step."""
        if self.new_instruction is not None:
            raise Interrupted()

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def fresh_frame(self, timeout=15.0):
        end = time.time() + timeout
        while time.time() < end:
            self.spin(0.05)
            self.check_interrupt()
            f = self.frame
            if f is not None and self.now_s() - Time.from_msg(f.header.stamp).nanoseconds * 1e-9 < self._g("frame_max_age_s"):
                return f
        raise RuntimeError(f"no fresh colour frame on {self.image_topic}")

    def status(self, **kw):
        kw.update(step=self.step_no, instruction=self.instruction, t=round(time.time(), 2))
        self.pub_status.publish(String(data=json.dumps(kw, ensure_ascii=False)))
        self.get_logger().info(json.dumps(kw, ensure_ascii=False))

    def robot_pose(self):
        """(x, y, yaw) of base_link in odom, latest; None without TF."""
        try:
            t = self.tf.lookup_transform(self._g("odom_frame"), self._g("base_frame"), Time()).transform
        except Exception:  # noqa: BLE001 -- tf2 raises several unrelated types
            return None
        q = t.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.translation.x, t.translation.y, yaw

    def distance_to_target(self):
        pose = self.robot_pose()
        if self.target is None or pose is None:
            return None
        return math.hypot(self.target.point.x - pose[0], self.target.point.y - pose[1])

    def pixel_heading(self, frame, point):
        """The heading in odom of the viewing ray through the model's pixel
        (normalised (u, v)), with the camera's pose at the picture's stamp
        (the latest one if that has left the buffer); None without the
        intrinsics or TF."""
        info = self.colour_info
        if info is None or point is None:
            return None
        ray = pixel_ray(point[0] * info.width, point[1] * info.height, info.k)
        odom = self._g("odom_frame")
        for stamp in (Time.from_msg(frame.header.stamp), Time()):
            try:
                q = self.tf.lookup_transform(odom, info.header.frame_id, stamp).transform.rotation
                return ray_heading((q.x, q.y, q.z, q.w), ray)
            except Exception:  # noqa: BLE001 -- tf2 raises several unrelated types
                continue
        return None

    # -- the picture -----------------------------------------------------

    @staticmethod
    def jpeg_of(frame):
        """The frame as JPEG bytes: the camera's own when compressed, else
        encoded here at quality 90."""
        if isinstance(frame, CompressedImage):
            return bytes(frame.data)
        rgb = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        buf = io.BytesIO()
        PILImage.fromarray(rgb).save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    # -- the model -------------------------------------------------------

    def ask(self, frame, text, schema, max_tokens=60):
        """-> (the answer as a dict, seconds taken, the JPEG the model saw)."""
        jpeg = self.jpeg_of(frame)
        body = {
            "model": self.model, "temperature": 0.0, "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
                    {"type": "text", "text": text}]},
            ],
            "response_format": {"type": "json_schema", "json_schema": {"name": "nav", "schema": schema, "strict": True}},
        }
        req = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self._g("request_timeout_s")) as r:
            out = json.load(r)
        try:
            content = out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            # A 200 with no answer in it (no choices, a null message, a
            # body that is not an object): the task ends in `error` like
            # an unreachable server does, the node stays up.
            raise RuntimeError(f"malformed model response ({exc!r}): {str(out)[:200]}") from exc
        return parse_answer(content or ""), time.perf_counter() - t0, jpeg

    def annotate(self, jpeg, frame, point=None, text=""):
        """The picture the model answered on, with its point drawn, to
        wojtek/vlm/annotated. Decoded here, once per look; the verify call's
        picture is never decoded."""
        im = PILImage.open(io.BytesIO(jpeg)).convert("RGB")
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

    def halt(self):
        """Stop everything this brain set in motion: the goal at goto and
        the resolver (one cancel), and a turn of our own (one zero). The
        task is over by the time this is called, so the cancel's echo on
        our own subscription is not a new stop (_on_cancel)."""
        self._halting = True
        try:
            self.pub_cancel.publish(Empty())
            self.pub_cmd.publish(Twist())
            self.spin(0.3)
        finally:
            self._halting = False

    def turn(self, deg):
        tw = Twist()
        tw.angular.z = self._g("turn_cmd_rad_s") * (1.0 if deg > 0 else -1.0)
        end = time.time() + abs(math.radians(deg)) / self._g("turn_real_rad_s")
        while time.time() < end:
            self.pub_cmd.publish(tw)
            self.spin(0.05)
            self.check_interrupt()
        self.pub_cmd.publish(Twist())
        self.spin(1.0)

    def step_forward(self, metres, heading=None):
        """Walk `metres` from where the robot stands along `heading` (odom
        yaw; None: straight ahead). The setpoint is fixed in odom once, so
        the re-sends mean the same point however long the walk takes.
        -> (goto's final word, metres actually covered or None)."""
        start = self.robot_pose()
        if start is None:
            raise RuntimeError("no TF odom -> base_link for an approach step")
        if heading is None:
            heading = start[2]
        gx, gy = step_along(start, heading, metres)
        goal = PoseStamped()
        goal.header.frame_id = self._g("odom_frame")
        goal.pose.position.x, goal.pose.position.y = float(gx), float(gy)
        goal.pose.orientation.z, goal.pose.orientation.w = math.sin(heading / 2), math.cos(heading / 2)
        t0, last, seen_active = time.time(), 0.0, False
        while time.time() - t0 < 25.0:
            if time.time() - last >= 1.0:  # goto's dead-man is 3 s
                goal.header.stamp = self.get_clock().now().to_msg()
                self.pub_goal.publish(goal)
                last = time.time()
            self.spin(0.05)
            self.check_interrupt()
            if self.goto_status in ("driving", "turning", "blocked"):
                seen_active = True
            if seen_active and self.goto_status in ("reached", "idle"):
                break
            if self.goto_status == "blocked" and time.time() - t0 > 6.0:
                break
        self.spin(1.0)
        res = "reached" if self.goto_status in ("reached", "idle") and seen_active else (self.goto_status or "timeout")
        end = self.robot_pose()
        moved = None if end is None else math.hypot(end[0] - start[0], end[1] - start[1])
        return res, moved

    def send_pixel(self, frame, point):
        m = PointStamped()
        m.header = frame.header
        m.point.x, m.point.y, m.point.z = float(point[0]), float(point[1]), 0.0
        # A stop that landed during the model call is still unread (no spin
        # happens inside ask()); read it before anything goes out.
        self.spin(0.05)
        self.check_interrupt()
        self.pixel_status = None
        self.target = None
        self.pub_pixel.publish(m)
        t0 = time.time()
        while time.time() - t0 < 80.0:
            self.spin(0.1)
            self.check_interrupt()
            if self.pixel_status in PIXEL_TERMINAL:
                return self.pixel_status
        return "timeout"

    # -- the loop --------------------------------------------------------

    def run_task(self, instruction):
        self.instruction = instruction
        ex = Explorer(self._g("turn_deg"), self._g("done_within_m"), self._g("approach_m"),
                      max_steps=int(self._g("max_steps")), max_blind=int(self._g("max_blind")),
                      min_progress_m=self._g("min_progress_m"), max_stalls=int(self._g("max_stalls")))
        self.step_no = 0
        t_start = time.time()
        action = ("look",)
        frame = None
        self._running = True
        try:
            while time.time() - t_start < self._g("max_s"):
                # Deliver what arrived during a model call (ask() does not
                # spin), then honour it before the next step moves anything.
                self.spin(0.05)
                self.check_interrupt()
                kind = action[0]
                if kind in Explorer.TERMINAL:
                    if kind == "gave_up":
                        # Not an exception, but the operator gets the reason
                        # the same way: latched in `error`.
                        self.halt()
                        self.status(action="finished", result=kind, error=action[1])
                    else:
                        self.status(action="finished", result=kind)
                    return kind
                if kind == "look":
                    self.step_no += 1
                    frame = self.fresh_frame()
                    answer, dt, jpeg = self.ask(frame, task_prompt(instruction), SCHEMA)
                    self.annotate(jpeg, frame, answer.get("point_2d") if answer.get("type") == "goal" else None,
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
                elif kind == "approach":
                    _, metres, point, detour = action
                    heading = self.pixel_heading(frame, point)
                    if heading is not None:
                        heading += math.radians(detour)
                    res, moved = self.step_forward(metres, heading)
                    self.status(action=kind, metres=metres, result=res,
                                heading_deg=None if heading is None else round(math.degrees(heading), 1),
                                detour_deg=detour, moved_m=None if moved is None else round(moved, 2))
                    action = ex.on_move_result(res, moved)
                elif kind == "explore":
                    res, moved = self.step_forward(action[1])
                    self.status(action=kind, metres=action[1], result=res,
                                moved_m=None if moved is None else round(moved, 2))
                    action = ex.on_move_result(res, moved)
                elif kind == "search":
                    self.turn(action[1])
                    self.status(action="turn", deg=action[1], turned_total=ex.turned_deg)
                    action = ex.turn_done()
        except Interrupted:
            self.halt()
            result = "cancelled" if self.new_instruction == "" else "replaced"
            self.status(action="finished", result=result)
            return result
        except (RuntimeError, OSError, KeyError, ValueError) as exc:
            # No frame, the model unreachable or answering garbage: stop
            # what moves and say why, instead of dying under the operator.
            self.halt()
            self.status(action="finished", result="error", error=f"{type(exc).__name__}: {exc}")
            return "error"
        finally:
            self._running = False
        self.status(action="finished", result="timeout")
        return "timeout"

    def serve(self):
        """One task from the parameter, then whatever arrives on the topic."""
        self.spin(1.5)
        if self.instruction:
            self.run_task(self.instruction)
        while rclpy.ok():
            self.spin(0.2)
            if self.new_instruction is None:
                continue
            task, self.new_instruction = self.new_instruction, None
            if task:
                self.run_task(task)
            else:
                # A stop while idle: still worth one cancel, for a goal
                # somebody else left at goto. No Twist: an idle brain does
                # not shout over whichever source drives now.
                self.pub_cancel.publish(Empty())
                self.instruction = ""
                self.status(action="idle")


def main():
    rclpy.init()
    node = VlmBrainNode()
    try:
        node.serve()
    except KeyboardInterrupt:
        pass
    finally:
        # On Ctrl-C rclpy's own handler may already have shut the context
        # down; then there is nothing left to publish to (the launch's
        # SIGINT reaches goto too, which sends its own zero), so don't die
        # with a traceback over a stop that already happened.
        if rclpy.ok():
            try:
                node.pub_cancel.publish(Empty())
                node.pub_cmd.publish(Twist())
                node.spin(0.3)
            except Exception:  # noqa: BLE001 -- the teardown race, not a bug
                pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
