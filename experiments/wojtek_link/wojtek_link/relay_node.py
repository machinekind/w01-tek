"""The relay: the DGX's ROS graph <-> the robot's, through the robot's own
foxglove_bridge (topics.py says what crosses and why nothing else can).

    python3 -m wojtek_link.relay_node [--url ws://<robot>:8765] [--send] [--in-camera]

Receive-only unless --send: the first contact with the robot reads its pose
and goto's word and publishes them on the DGX, nothing goes the other way.
With --send the DGX's goals, cancels and /cmd_vel reach the robot.

Runs on the DGX, in the robot's Docker image (rclpy + websockets), on the
DGX's own ROS domain with ROS_LOCALHOST_ONLY=1, next to the brain, the pixel
resolver and the operator page. Never on the robot: the robot is left
exactly as it is, the relay is one more client of a bridge it already runs.

What it does with the traffic
  IN   republished on the DGX with the robot's stamps; /wojtek/odom also
       becomes the DGX's odom -> base_link TF; every /tf_static the robot
       holds is merged into one latched message (a late subscriber on the
       DGX gets all of them, not just the last publisher's).
  OUT  sent while the link is up, dropped while it is down -- never
       queued: a /cmd_vel from before a drop-out must not arrive after it.
       A goal's stamp is sent as zero, "now" on the robot: goto takes an
       odom goal as it is and a stamped one would be read against the
       robot's clock, which the DGX's need not match. Order is kept (one
       websocket), so a cancel still lands after the goal it cancels.
  status  JSON on /wojtek/link/status once a second: connected, counts,
       drops, the robot's clock offset as seen on the traffic (clock.py).

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


class LinkNode(Node):
    def __init__(self, url: str, send: bool, in_camera: bool, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__("wojtek_link")
        self.url, self.send_enabled, self.loop = url, send, loop
        self.wanted_in: Dict[str, str] = dict(topics.IN)
        if in_camera:
            self.wanted_in.update(topics.IN_CAMERA)
        self.types = {t: get_message(s) for t, s in {**self.wanted_in, **topics.OUT}.items()}
        self.pubs = {}
        for t in self.wanted_in:
            if t == "/tf_static" or t in LATCHED_IN:
                qos = LATCHED
            elif t in topics.IN_CAMERA:
                qos = qos_profile_sensor_data
            else:
                qos = 10
            self.pubs[t] = self.create_publisher(self.types[t], t, qos)
        self.tf = TransformBroadcaster(self)
        self.static: Dict[str, TransformStamped] = {}
        self.clock = OffsetEstimator()
        self.counts_in: Dict[str, int] = defaultdict(int)
        self.counts_out: Dict[str, int] = defaultdict(int)
        self.dropped_out: Dict[str, int] = defaultdict(int)
        self.state = {"connected": False, "problems": [], "error": None, "since": None}
        self.out_queue: Optional[asyncio.Queue] = None
        if send:
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
        q = self.out_queue
        if q is None or not self.state["connected"]:
            self.dropped_out[topic] += 1
            return
        if q.full():
            try:
                q.get_nowait()
                self.dropped_out["(queue full)"] += 1
            except asyncio.QueueEmpty:
                pass
        q.put_nowait((topic, payload))

    # -- IN: the robot -> the DGX's graph (asyncio thread) --------------------

    def on_in(self, topic: str, payload: bytes) -> None:
        msg = deserialize_message(payload, self.types[topic])
        stamp = _stamp_s(msg)
        if stamp is not None and topic not in topics.IN_CAMERA:
            self.clock.add(time.time(), stamp)
        self.counts_in[topic] += 1
        if topic == "/tf_static":
            for t in msg.transforms:
                self.static[t.child_frame_id] = t
            self.pubs[topic].publish(TFMessage(transforms=list(self.static.values())))
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
        off = self.clock.offset_s()
        status = {
            "t": round(time.time(), 2), "url": self.url, "connected": self.state["connected"],
            "send": self.send_enabled, "since": self.state["since"],
            "in": dict(self.counts_in), "out": dict(self.counts_out), "dropped_out": dict(self.dropped_out),
            "clock_offset_ms": None if off is None else round(off * 1000.0, 1),
            "problems": self.state["problems"], "error": self.state["error"],
        }
        self.pub_status.publish(String(data=json.dumps(status)))


async def _session(node: LinkNode, ws) -> None:
    """One connection: serverInfo, our OUT channels, IN subscriptions as the
    robot's channels are advertised, then messages both ways until it drops."""
    log = node.get_logger()
    first = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
    if first.get("op") != "serverInfo":
        raise RuntimeError(f"expected serverInfo, got {first.get('op')!r}")
    caps = first.get("capabilities", [])
    if "cdr" not in first.get("supportedEncodings", []):
        raise RuntimeError("the robot's bridge does not speak cdr")
    out_ids: Dict[str, int] = {}
    if node.send_enabled:
        if "clientPublish" not in caps:
            raise RuntimeError("the robot's bridge refuses client publishing (no clientPublish capability)")
        chans = [(i + 1, t, s) for i, (t, s) in enumerate(topics.OUT.items())]
        out_ids = {t: cid for cid, t, _s in chans}
        await ws.send(protocol.advertise(chans))
    node.out_queue = asyncio.Queue(maxsize=OUT_QUEUE)
    node.state.update(connected=True, error=None, since=round(time.time(), 2), problems=[])
    log.info(f"link up: {node.url} ({ws.subprotocol}); sending {'ON' if node.send_enabled else 'off'}")

    advertised: Dict[str, dict] = {}
    by_channel: Dict[int, str] = {}       # server channel id -> topic we subscribed it for
    by_sub: Dict[int, str] = {}           # our subscription id -> topic
    next_sub = [1]
    started = time.monotonic()
    reported_missing = [False]

    async def sender() -> None:
        while True:
            topic, payload = await node.out_queue.get()
            await ws.send(protocol.client_message(out_ids[topic], payload))
            node.counts_out[topic] += 1

    async def subscribe_new() -> None:
        ids, problems = protocol.match_channels(advertised, node.wanted_in)
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
        node.state["problems"] = still
        if still and not reported_missing[0] and time.monotonic() - started > MISSING_AFTER_S:
            reported_missing[0] = True
            log.warning("the robot's bridge lacks: " + "; ".join(still))

    send_task = asyncio.create_task(sender()) if node.send_enabled else None
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
                        node.on_in(topic, payload)
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
                log.warning(f"robot's bridge says: {m.get('message')}", throttle_duration_sec=5.0)
    finally:
        if send_task is not None:
            send_task.cancel()
        node.state["connected"] = False
        node.out_queue = None


async def run_link(node: LinkNode) -> None:
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(node.url, subprotocols=list(protocol.SUBPROTOCOLS),
                                          max_size=64 * 1024 * 1024, open_timeout=5,
                                          ping_interval=5, ping_timeout=5) as ws:
                backoff = 1.0
                await _session(node, ws)
        except (OSError, asyncio.TimeoutError, websockets.WebSocketException, RuntimeError) as exc:
            if node.state["error"] != f"{type(exc).__name__}: {exc}":
                node.get_logger().warning(f"link down ({node.url}): {type(exc).__name__}: {exc}")
            node.state.update(connected=False, error=f"{type(exc).__name__}: {exc}")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2.0, 10.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="DGX <-> Wojtek relay through the robot's foxglove_bridge.")
    ap.add_argument("--url", default=os.environ.get("WOJTEK_BRIDGE_URL", topics.DEFAULT_URL),
                    help="the robot's bridge (WOJTEK_BRIDGE_URL; default the robot's AP anchor)")
    ap.add_argument("--send", action="store_true",
                    help="also send the DGX's goals, cancels and /cmd_vel to the robot (default: receive only)")
    ap.add_argument("--in-camera", action="store_true",
                    help="also relay the camera IN -- the sim test only; on the robot it comes from the camera Pi")
    args = ap.parse_args()
    # rclpy's own handlers would take SIGINT/SIGTERM, shut the ROS side
    # down and leave the websocket running: a relay that still holds the
    # robot's bridge (its subscriptions cost the robot) while relaying
    # nothing. Seen on the first test run: four of them after three pkills.
    # So the signals end the asyncio side, and the ROS side goes with it.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    node = LinkNode(args.url, args.send, args.in_camera, loop)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, name="wojtek_link_spin", daemon=True).start()
    node.get_logger().info(f"relay: {args.url}, IN {sorted(node.wanted_in)}, "
                           f"OUT {sorted(topics.OUT) if args.send else 'off (receive only)'}")
    main_task = loop.create_task(run_link(node))
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, main_task.cancel)
    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        pass
    finally:
        node.get_logger().info("relay stopping: link closed, nothing more goes to the robot")
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        rclpy.try_shutdown()
        loop.close()


if __name__ == "__main__":
    main()
