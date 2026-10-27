"""The page's ROS 2 side: a thin client of wojtek_nav's vlm_brain_node.

Publishes the operator's instruction on wojtek/vlm/instruction and the
cancel on wojtek/nav/cancel; subscribes to the brain's status JSON, the
two nav status words, the annotated picture, the camera's own JPEG
for the live view (limits.CAMERA_TOPIC, the stream the brain reads), the
raw depth (limits.DEPTH_TOPIC), the robot's load (limits.SYSMON_TOPIC,
from sysmon_node on the RPi) and the walker's guide (limits.GUIDE_TOPIC).

Every stored item carries a sequence number (`seq()`), so the web server
sends a client only what changed since its last look and drops frames a
slow browser could not take, instead of queueing them.
On STOP (`freeze()`) it also holds /cmd_vel at zero for a second: the
robot freezes where it stands (limits.FREEZE_*). A zero Twist is the only
thing it ever publishes there; it never drives.

`BrainClient` takes any node-like object (the tests hand it a mock);
`start_client` makes the real node and spins it on a daemon thread so
the web server never blocks on ROS.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Empty, String

from wojtek_vlm_gui import limits
from wojtek_vlm_gui.task_log import TaskLog, is_stop_word, parse_status

# The brain and goto publish their status latched (depth 1, reliable,
# transient local); a subscriber with the same profile gets the last one
# on connect instead of waiting for the next step.
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
ANNOTATED_ENCODING = "rgb8"


class BrainClient:
    def __init__(self, node, subscriber_wait_s: float = limits.SUBSCRIBER_WAIT_S,
                 poll_s: float = limits.SUBSCRIBER_POLL_S) -> None:
        self.node = node
        self._wait_s = float(subscriber_wait_s)
        self._poll_s = float(poll_s)
        self._lock = threading.Lock()
        self.log = TaskLog()
        self.goto_status: Optional[str] = None
        self.pixel_status: Optional[str] = None
        self._annotated: Optional[np.ndarray] = None   # the brain's newest picture
        # True from a picture's arrival until an ask line takes it.
        self._annotated_new = False
        # The picture of the newest ask line that got one, with its number.
        self._pinned: Optional[Tuple[int, np.ndarray]] = None
        self._jpeg: Optional[bytes] = None
        self._jpeg_at: float = 0.0
        self._sys: Optional[Dict[str, Any]] = None
        self._sys_at: float = 0.0
        self._depth: Optional[Tuple[int, int, bytes]] = None   # (width, height, uint16 LE)
        self._guide: Optional[Dict[str, Any]] = None
        self._seq: Dict[str, int] = {"log": 0, "nav": 0, "annotated": 0,
                                     "camera": 0, "depth": 0, "sys": 0, "guide": 0}
        self._pub_instruction = node.create_publisher(String, limits.INSTRUCTION_TOPIC, 10)
        self._pub_cancel = node.create_publisher(Empty, limits.CANCEL_TOPIC, 10)
        # Zeros only, and only while a STOP holds (freeze / _hold_zero).
        self._pub_zero = node.create_publisher(Twist, limits.CMD_VEL_TOPIC, 10)
        self._freeze_until = 0.0
        self._holding = False
        # How many STOPs have gone out so far. A task that was still waiting
        # for the brain when a STOP went out must not be sent after it.
        self._stops = 0
        # Before the brain's status: when a picture and its ask line are both
        # waiting, rclpy runs them in the order they were subscribed, and the
        # picture has to come first.
        node.create_subscription(Image, limits.ANNOTATED_TOPIC, self._on_annotated, 1)
        node.create_subscription(String, limits.VLM_STATUS_TOPIC, self._on_status, LATCHED)
        node.create_subscription(String, limits.GOTO_STATUS_TOPIC, self._on_goto, LATCHED)
        node.create_subscription(String, limits.PIXEL_STATUS_TOPIC, self._on_pixel, LATCHED)
        # The camera's JPEG: best-effort, depth 1, the sensor profile the
        # driver and the brain use; a late frame is dropped, never queued.
        node.create_subscription(CompressedImage, limits.CAMERA_TOPIC, self._on_camera,
                                 qos_profile_sensor_data)
        node.create_subscription(Image, limits.DEPTH_TOPIC, self._on_depth, qos_profile_sensor_data)
        # Best effort, matching sysmon_node's publisher (see there why).
        node.create_subscription(String, limits.SYSMON_TOPIC, self._on_sysmon, qos_profile_sensor_data)
        node.create_subscription(String, limits.GUIDE_TOPIC, self._on_guide, 1)   # volatile, see sysmon_node

    # -- out -------------------------------------------------------------

    def send(self, text: str) -> Tuple[bool, str]:
        """Publish a task. A stop word is a stop. Refuses, with a message,
        when no brain subscribes within the wait: never a silent no-op.
        A STOP that lands while the task waits for the brain drops the task
        with an error reply, instead of sending it after the stop."""
        text = text.strip()
        if is_stop_word(text):
            return self.freeze()
        with self._lock:
            stops = self._stops
        # The wait sleeps, so it runs outside the lock.
        if not self._brain_listening():
            return False, (f"no brain is listening on {limits.INSTRUCTION_TOPIC} "
                           "(is the session up with vlm:=true?)")
        # Check and publish under one lock, so a STOP cannot slip in between.
        with self._lock:
            if self._stops != stops:
                return False, f"not sent, STOP came first: {text}"
            self._pub_instruction.publish(String(data=text))
        return True, f"sent: {text}"

    def stop(self) -> Tuple[bool, str]:
        """The web console's order: the cancel first (goto and the resolver
        drop their goal even with no brain running), then the empty
        instruction (the brain halts and reports `cancelled`). Every STOP
        goes through here, so it also drops a task still waiting in send()."""
        with self._lock:
            self._stops += 1
            self._pub_cancel.publish(Empty())
            self._pub_instruction.publish(String(data=""))
        return True, "STOP sent"

    def freeze(self, hold_s: float = limits.FREEZE_HOLD_S,
               period_s: float = limits.FREEZE_PERIOD_S, start_hold=None) -> Tuple[bool, str]:
        """STOP: the robot freezes where it stands. The brain's stop first
        (goto and the resolver drop their goal, the brain halts), then a
        zero Twist on /cmd_vel at once, held for hold_s (limits.FREEZE_*
        says why zero velocity and not policy off or disarm). Never blocks:
        the hold runs on its own thread; a STOP during a hold extends it
        instead of starting a second one. `start_hold` replaces the thread
        starter in the tests."""
        self.stop()
        self._zero()
        with self._lock:
            self._freeze_until = time.monotonic() + float(hold_s)
            start = not self._holding
            self._holding = True
        if start:
            (start_hold or self._start_hold_thread)(float(period_s))
        return True, f"STOP: task and goal cancelled, /cmd_vel held at zero for {hold_s:g} s"

    def _zero(self) -> None:
        self._pub_zero.publish(Twist())   # all zeros; linear.z 0 = the policy's standing height

    def _start_hold_thread(self, period_s: float) -> None:
        threading.Thread(target=self._hold_zero, args=(period_s,),
                         name="wojtek_vlm_gui_freeze", daemon=True).start()

    def _hold_zero(self, period_s: float, sleep=time.sleep, clock=time.monotonic) -> None:
        """One zero Twist every period_s until the freeze runs out. The
        deadline is checked and the hold released under one lock, so a STOP
        that lands as the hold ends either extends this one or starts the
        next, never neither."""
        while True:
            with self._lock:
                if clock() >= self._freeze_until:
                    self._holding = False
                    return
            self._zero()
            sleep(period_s)

    def _brain_listening(self) -> bool:
        deadline = time.monotonic() + self._wait_s
        while True:
            if self.node.count_subscribers(limits.INSTRUCTION_TOPIC) > 0:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(self._poll_s)

    # -- in --------------------------------------------------------------

    def _on_status(self, msg) -> None:
        status = parse_status(msg.data)
        if status is None:
            return
        with self._lock:
            # The brain sends the picture just before the ask line of the
            # same step. Number it and write the number on the line, so the
            # page puts it under the right line even when it joins mid-task.
            # A picture goes to one ask only; an ask with no new picture
            # gets none.
            if status.get("action") == "ask" and self._annotated_new:
                self._seq["annotated"] += 1
                status["_picture"] = self._seq["annotated"]
                self._pinned = (self._seq["annotated"], self._annotated)
                self._annotated_new = False
            self.log.push(status)
            self._seq["log"] += 1

    def _on_goto(self, msg) -> None:
        with self._lock:
            self.goto_status = msg.data
            self._seq["nav"] += 1

    def _on_pixel(self, msg) -> None:
        with self._lock:
            self.pixel_status = msg.data
            self._seq["nav"] += 1

    def _on_annotated(self, msg) -> None:
        if msg.encoding != ANNOTATED_ENCODING:
            return
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, 3)
        with self._lock:
            self._annotated = img
            self._annotated_new = True

    def _on_camera(self, msg) -> None:
        with self._lock:
            self._jpeg = bytes(msg.data)
            self._jpeg_at = time.monotonic()
            self._seq["camera"] += 1

    def _on_depth(self, msg) -> None:
        if msg.encoding not in limits.DEPTH_ENCODINGS or msg.is_bigendian:
            return
        data = bytes(msg.data)
        if msg.step != msg.width * 2:   # padded rows: keep the pixels only
            data = b"".join(data[r * msg.step:r * msg.step + msg.width * 2] for r in range(msg.height))
        with self._lock:
            self._depth = (int(msg.width), int(msg.height), data)
            self._seq["depth"] += 1

    def _on_sysmon(self, msg) -> None:
        data = parse_status(msg.data)   # any JSON object; None for garbage
        if data is None:
            return
        with self._lock:
            self._sys = data
            self._sys_at = time.monotonic()
            self._seq["sys"] += 1

    def _on_guide(self, msg) -> None:
        data = parse_status(msg.data)
        if data is None:
            return
        with self._lock:
            self._guide = data
            self._seq["guide"] += 1

    def latest_guide(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._guide

    def seq(self) -> Dict[str, int]:
        """How many times each stored item changed; a web client compares
        it with what it last sent."""
        with self._lock:
            return dict(self._seq)

    def latest_depth(self) -> Optional[Tuple[int, int, bytes]]:
        with self._lock:
            return self._depth

    def latest_annotated(self) -> Optional[Tuple[int, np.ndarray]]:
        """The newest picture that went to an ask line, with the number that
        line carries; None before the first."""
        with self._lock:
            return self._pinned

    def latest_sysmon(self) -> Tuple[Optional[Dict[str, Any]], float]:
        """The robot's last load reading as published and its age in
        seconds (None, inf before the first one)."""
        with self._lock:
            if self._sys is None:
                return None, float("inf")
            return self._sys, time.monotonic() - self._sys_at

    def latest_jpeg(self) -> Tuple[Optional[bytes], float]:
        """The newest camera JPEG and its age in seconds (None, inf before
        the first frame)."""
        with self._lock:
            if self._jpeg is None:
                return None, float("inf")
            return self._jpeg, time.monotonic() - self._jpeg_at

    def latest(self) -> Dict[str, Any]:
        """A snapshot for one render: the task's lines, the nav words."""
        with self._lock:
            return {
                "instruction": self.log.instruction,
                "lines": list(self.log.lines),
                "finished": self.log.finished,
                "idle": self.log.idle,
                "goto": self.goto_status,
                "pixel": self.pixel_status,
            }


def start_client(node_name: str = "wojtek_vlm_gui") -> BrainClient:
    """The real node, spun on a daemon thread. The web server calls `send`,
    `stop` and `latest` from its own threads; rclpy allows publishing from
    any thread, and the callbacks land under the client's lock."""
    if not rclpy.ok():
        rclpy.init()
    node = Node(node_name)
    client = BrainClient(node)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, name="wojtek_vlm_gui_spin", daemon=True).start()
    return client
