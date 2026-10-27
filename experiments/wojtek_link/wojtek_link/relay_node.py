"""The relay: the DGX's ROS graph <-> the robot's, through the robot's own
foxglove_bridge (topics.py says what crosses and why nothing else can).

    python3 -m wojtek_link.relay_node [--url ws://<robot>:8765] [--send]
                                      [--camera-url ws://<camera Pi>:8765 | --in-camera]

Receive-only unless --send: the first contact with the robot reads its pose
and goto's word and publishes them on the DGX, nothing goes the other way.
With --send the DGX's goals, cancels and /cmd_vel reach the robot.

--camera-url opens a second websocket, to the camera Pi's read-only bridge
(camera_pi/): the pictures, the depth and the driver's optical frames come
IN from there, and nothing ever goes out on it. The two links come and go
independently; the robot's never carries a picture.

Runs on the DGX, in the robot's Docker image (rclpy + websockets), on the
DGX's own ROS domain with ROS_LOCALHOST_ONLY=1, next to the brain, the pixel
resolver and the operator page. Never on the robot: the robot is left
exactly as it is, the relay is one more client of a bridge it already runs.

What it does with the traffic
  IN   republished on the DGX with the robot's stamps; /wojtek/odom also
       becomes the DGX's odom -> base_link TF; every /tf_static the robot
       holds is merged into one latched message (a late subscriber on the
       DGX gets all of them, not just the last publisher's).
       The camera Pi's /tf_static adds only its optical chain below
       camera_link; the mount itself and the body are the robot's URDF
       (topics.merge_static).
  OUT  sent while the link is up, dropped while it is down -- never
       queued: a /cmd_vel from before a drop-out must not arrive after it.
       A goal's stamp is sent as zero, "now" on the robot: goto takes an
       odom goal as it is and a stamped one would be read against the
       robot's clock, which the DGX's need not match. Order is kept (one
       websocket), so a cancel still lands after the goal it cancels.
  status  JSON on /wojtek/link/status once a second: connected, counts,
       drops, kB/s in, the robot's clock offset as seen on the traffic
       (clock.py); the same for the camera Pi under "camera".

If the link drops, the robot's own dead-men stop it: policy_node zeroes
/cmd_vel after 0.5 s of silence, goto drops a setpoint after 3 s.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import threading
import time
from collections import defaultdict
from typing import Dict, Optional

import rclpy
import websockets
from geometry_msgs.msg import TransformStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.serialization import deserialize_message, serialize_message
from rclpy.signals import SignalHandlerOptions
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

from wojtek_link import protocol, topics
from wojtek_link.clock import OffsetEstimator

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
LATCHED_IN = ("/wojtek/nav/status",)      # goto publishes its word latched
OUT_QUEUE = 50                            # a few ticks of every OUT topic; older ones are dropped
MISSING_AFTER_S = 3.0                     # how long the robot's bridge gets to advertise what IN wants


def _stamp_s(msg) -> Optional[float]:
    header = getattr(msg, "header", None)
    if header is None:
        return None
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class Link:
    """One websocket and what the relay knows about it: the robot's bridge
    (IN, and OUT with --send) or the camera Pi's (IN only, never OUT)."""

    def __init__(self, name: str, url: str, wanted: Dict[str, str], send: bool = False) -> None:
        if send and name != "robot":
            raise ValueError(f"only the robot's link sends; {name!r} is receive-only")
        self.name, self.url, self.wanted, self.send = name, url, dict(wanted), send
        self.state = {"connected": False, "problems": [], "error": None, "since": None}
        self.clock = OffsetEstimator()
        self.counts_in: Dict[str, int] = defaultdict(int)
        self.bytes_in = 0
        self._rate_mark = (0, time.monotonic())
        self.out_queue: Optional[asyncio.Queue] = None

    def clocks_on(self, topic: str) -> bool:
        """Which stamps say where this machine's clock is. The robot: its
        pose (a camera relayed in the sim test is left out, a picture's
        stamp is its exposure). The camera Pi: its camera_info, the
        picture's stamp -- so the offset includes the driver's pipeline."""
        if self.name == "robot":
            return topic not in topics.IN_CAMERA
        return topic.endswith("/camera_info")

    def kb_s(self) -> float:
        n, t = self._rate_mark
        now = time.monotonic()
        self._rate_mark = (self.bytes_in, now)
        return round((self.bytes_in - n) / max(now - t, 1e-3) / 1000.0, 1)

    def status(self) -> dict:
        off = self.clock.offset_s()
        return {
            "url": self.url, "connected": self.state["connected"], "since": self.state["since"],
            "in": dict(self.counts_in), "kB_s_in": self.kb_s(),
            "clock_offset_ms": None if off is None else round(off * 1000.0, 1),
            "problems": self.state["problems"], "error": self.state["error"],
        }


class LinkNode(Node):
    def __init__(self, robot: Link, camera: Optional[Link], loop: asyncio.AbstractEventLoop) -> None:
        super().__init__("wojtek_link")
        self.robot, self.camera, self.loop = robot, camera, loop
        self.links = [ln for ln in (robot, camera) if ln is not None]
        wanted: Dict[str, str] = {}
        for ln in self.links:
            wanted.update(ln.wanted)
        self.types = {t: get_message(s) for t, s in {**wanted, **topics.OUT}.items()}
        self.pubs = {}
        for t in wanted:
            if t == "/tf_static" or t in LATCHED_IN:
                qos = LATCHED
            elif t in topics.IN_CAMERA:
                qos = qos_profile_sensor_data
            else:
                qos = 10
            self.pubs[t] = self.create_publisher(self.types[t], t, qos)
        self.tf = TransformBroadcaster(self)
        self.static: Dict[str, tuple] = {}         # child frame -> (link name, TransformStamped)
        self.static_refused: set = set()
        self.counts_out: Dict[str, int] = defaultdict(int)
        self.dropped_out: Dict[str, int] = defaultdict(int)
        if robot.send:
            for t in topics.OUT:
                self.create_subscription(self.types[t], t, lambda m, t=t: self._on_out(t, m), 10)
        self.pub_status = self.create_publisher(String, topics.STATUS_TOPIC, LATCHED)
        self.create_timer(1.0, self._publish_status)

    # -- OUT: the DGX's graph -> the robot (rclpy thread, handed to asyncio) --

    def _on_out(self, topic: str, msg) -> None:
        if topic == "/wojtek/nav/goal":
            msg.header.stamp.sec, msg.header.stamp.nanosec = 0, 0
        payload = serialize_message(msg)
        self.loop.call_soon_threadsafe(self._enqueue, topic, payload)

    def _enqueue(self, topic: str, payload: bytes) -> None:
        q = self.robot.out_queue
        if q is None or not self.robot.state["connected"]:
            self.dropped_out[topic] += 1
            return
        if q.full():
            try:
                q.get_nowait()
                self.dropped_out["(queue full)"] += 1
            except asyncio.QueueEmpty:
                pass
        q.put_nowait((topic, payload))

    # -- IN: the robot / the camera Pi -> the DGX's graph (asyncio thread) ----

    def on_in(self, link: Link, topic: str, payload: bytes) -> None:
        link.bytes_in += len(payload)
        msg = deserialize_message(payload, self.types[topic])
        stamp = _stamp_s(msg)
        if stamp is not None and link.clocks_on(topic):
            link.clock.add(time.time(), stamp)
        link.counts_in[topic] += 1
        if topic == "/tf_static":
            refused = set(topics.merge_static(self.static, msg.transforms, link.name)) - self.static_refused
            if refused:
                self.static_refused |= refused
                self.get_logger().warning(
                    f"{link.name}'s /tf_static may not set {sorted(refused)} (the robot's URDF owns the body "
                    f"and the mount; the camera Pi adds only its optical chain)")
            self.pubs[topic].publish(TFMessage(transforms=[t for _src, t in self.static.values()]))
            return
        self.pubs[topic].publish(msg)
        if topic == "/wojtek/odom":
            t = TransformStamped()
            t.header = msg.header
            t.child_frame_id = msg.child_frame_id
            p = msg.pose.pose
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (
                p.position.x, p.position.y, p.position.z)
            t.transform.rotation = p.orientation
            self.tf.sendTransform(t)

    # -- status ------------------------------------------------------------------

    def _publish_status(self) -> None:
        status = {"t": round(time.time(), 2), "send": self.robot.send, **self.robot.status(),
                  "out": dict(self.counts_out), "dropped_out": dict(self.dropped_out),
                  # base_link -> camera_link: the robot's URDF has the mount (fork PR #9)
                  "camera_mount": topics.MOUNT_FRAME in self.static,
                  "static_refused": sorted(self.static_refused)}
        if self.camera is not None:
            status["camera"] = self.camera.status()
        self.pub_status.publish(String(data=json.dumps(status)))


async def _session(node: LinkNode, link: Link, ws) -> None:
    """One connection: serverInfo, our OUT channels (the robot's link with
    --send only), IN subscriptions as the server's channels are advertised,
    then messages until it drops."""
    log = node.get_logger()
    first = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
    if first.get("op") != "serverInfo":
        raise RuntimeError(f"expected serverInfo, got {first.get('op')!r}")
    caps = first.get("capabilities", [])
    if "cdr" not in first.get("supportedEncodings", []):
        raise RuntimeError(f"the {link.name}'s bridge does not speak cdr")
    out_ids: Dict[str, int] = {}
    if link.send:
        if "clientPublish" not in caps:
            raise RuntimeError("the robot's bridge refuses client publishing (no clientPublish capability)")
        chans = [(i + 1, t, s) for i, (t, s) in enumerate(topics.OUT.items())]
        out_ids = {t: cid for cid, t, _s in chans}
        await ws.send(protocol.advertise(chans))
        link.out_queue = asyncio.Queue(maxsize=OUT_QUEUE)
    link.state.update(connected=True, error=None, since=round(time.time(), 2), problems=[])
    log.info(f"link up: {link.name} {link.url} ({ws.subprotocol}); "
             f"sending {'ON' if link.send else 'off'}")

    advertised: Dict[str, dict] = {}
    by_channel: Dict[int, str] = {}       # server channel id -> topic we subscribed it for
    by_sub: Dict[int, str] = {}           # our subscription id -> topic
    next_sub = [1]
    started = time.monotonic()
    reported_missing = [False]

    async def sender() -> None:
        while True:
            topic, payload = await link.out_queue.get()
            await ws.send(protocol.client_message(out_ids[topic], payload))
            node.counts_out[topic] += 1

    async def subscribe_new() -> None:
        ids, problems = protocol.match_channels(advertised, link.wanted)
        fresh = [(t, c) for t, c in ids.items() if c not in by_channel]
        if fresh:
            pairs = []
            for t, c in fresh:
                sid = next_sub[0]
                next_sub[0] += 1
                by_sub[sid], by_channel[c] = t, t
                pairs.append((sid, c))
            await ws.send(protocol.subscribe(pairs))
        still = [p for p in problems if "not advertised" not in p or time.monotonic() - started > MISSING_AFTER_S]
        link.state["problems"] = still
        if still and not reported_missing[0] and time.monotonic() - started > MISSING_AFTER_S:
            reported_missing[0] = True
            log.warning(f"the {link.name}'s bridge lacks: " + "; ".join(still))

    send_task = asyncio.create_task(sender()) if link.send else None
    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                await subscribe_new()          # re-check what is still missing
                continue
            if isinstance(raw, (bytes, bytearray)):
                parsed = protocol.parse_server_binary(raw)
                if parsed is None:
                    continue
                sid, _log_time, payload = parsed
                topic = by_sub.get(sid)
                if topic is not None:
                    try:
                        node.on_in(link, topic, payload)
                    except Exception as exc:  # noqa: BLE001 -- one bad message must not end the link
                        log.warning(f"dropping a {topic} message: {exc}", throttle_duration_sec=5.0)
                continue
            m = json.loads(raw)
            op = m.get("op")
            if op == "advertise":
                for ch in m.get("channels", []):
                    advertised[ch["topic"]] = ch
                await subscribe_new()
            elif op == "unadvertise":
                gone = set(m.get("channelIds", []))
                for topic, ch in list(advertised.items()):
                    if ch.get("id") in gone:
                        del advertised[topic]
                for c in gone:
                    t = by_channel.pop(c, None)
                    if t is not None:
                        for sid in [s for s, tt in by_sub.items() if tt == t]:
                            del by_sub[sid]
            elif op == "status" and m.get("level", 0) >= 1:
                log.warning(f"the {link.name}'s bridge says: {m.get('message')}", throttle_duration_sec=5.0)
    finally:
        if send_task is not None:
            send_task.cancel()
        link.state["connected"] = False
        link.out_queue = None


async def run_link(node: LinkNode, link: Link) -> None:
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(link.url, subprotocols=list(protocol.SUBPROTOCOLS),
                                          max_size=64 * 1024 * 1024, open_timeout=5,
                                          ping_interval=5, ping_timeout=5) as ws:
                backoff = 1.0
                await _session(node, link, ws)
        except (OSError, asyncio.TimeoutError, websockets.WebSocketException, RuntimeError) as exc:
            if link.state["error"] != f"{type(exc).__name__}: {exc}":
                node.get_logger().warning(f"link down ({link.name} {link.url}): {type(exc).__name__}: {exc}")
            link.state.update(connected=False, error=f"{type(exc).__name__}: {exc}")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2.0, 10.0)


async def run_all(node: LinkNode) -> None:
    """Each link on its own: the camera Pi dropping out leaves the robot's
    link up, and the other way round."""
    await asyncio.gather(*(run_link(node, ln) for ln in node.links))


def main() -> None:
    ap = argparse.ArgumentParser(description="DGX <-> Wojtek relay through the robot's foxglove_bridge.")
    ap.add_argument("--url", default=os.environ.get(topics.BRIDGE_URL_ENV, topics.DEFAULT_URL),
                    help=f"the robot's bridge ({topics.BRIDGE_URL_ENV}; default the robot's AP anchor)")
    ap.add_argument("--send", action="store_true",
                    help="also send the DGX's goals, cancels and /cmd_vel to the robot (default: receive only)")
    cam = ap.add_mutually_exclusive_group()
    cam.add_argument("--camera-url", default=os.environ.get(topics.CAMERA_URL_ENV) or None,
                     help=f"the camera Pi's read-only bridge ({topics.CAMERA_URL_ENV}): the pictures come IN "
                          "from there, nothing goes out on it")
    cam.add_argument("--in-camera", action="store_true",
                     help="the camera through the ROBOT's bridge -- the sim test only")
    args = ap.parse_args()
    if args.in_camera:
        args.camera_url = None      # the env's camera Pi would be a second source of the same topics
    robot_wants = {**topics.IN, **(topics.IN_CAMERA if args.in_camera else {})}
    robot = Link("robot", args.url, robot_wants, send=args.send)
    camera = Link("camera", args.camera_url, topics.CAMERA_PI) if args.camera_url else None
    # rclpy's own handlers would take SIGINT/SIGTERM, shut the ROS side
    # down and leave the websocket running: a relay that still holds the
    # robot's bridge (its subscriptions cost the robot) while relaying
    # nothing. Seen on the first test run: four of them after three pkills.
    # So the signals end the asyncio side, and the ROS side goes with it.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    node = LinkNode(robot, camera, loop)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    # Spun in short slices and joined before the ROS side is torn down: a
    # thread still inside spin() at teardown aborted the process now and
    # then ("terminate called without an active exception", 2 of 8 stops).
    stop_spin = threading.Event()

    def spin() -> None:
        while not stop_spin.is_set():
            executor.spin_once(timeout_sec=0.1)

    spinner = threading.Thread(target=spin, name="wojtek_link_spin", daemon=True)
    spinner.start()
    node.get_logger().info(f"relay: robot {robot.url}, IN {sorted(robot.wanted)}, "
                           f"OUT {sorted(topics.OUT) if args.send else 'off (receive only)'}")
    if camera is not None:
        node.get_logger().info(f"relay: camera Pi {camera.url}, IN {sorted(camera.wanted)}, OUT never")
    main_task = loop.create_task(run_all(node))
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, main_task.cancel)
    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        pass
    finally:
        # What the websockets left running (their keepalives, a closing
        # handshake) ends here, not in "Task was destroyed but it is
        # pending!" at loop.close() -- seen with two links on the robot.
        left = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in left:
            t.cancel()
        if left:
            loop.run_until_complete(asyncio.wait(left, timeout=2.0))
        node.get_logger().info("relay stopping: link closed, nothing more goes to the robot")
        stop_spin.set()
        spinner.join(timeout=2.0)
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        rclpy.try_shutdown()
        loop.close()


if __name__ == "__main__":
    main()
