#!/usr/bin/env python3
"""Deck gateway -- the robot-side half of the deck panel.

    ros2 run wojtek_deck deck_gateway        # then open http://<robot>:8090

One HTTP port (parameter `port`, default 8090) carries three things:

  GET /              the panel (web/index.html and the files next to it)
  GET /ws            the command websocket: JSON both ways, see below
  GET /stream.mjpg   the colour camera as MJPEG (multipart), for the page's
                     <img> and for any detector on the handheld that wants
                     the same frames (OpenCV opens the URL directly)
  GET /det/          the detector's assets: the YOLOX network and the
                     onnxruntime-web runtime the page runs it with. Big
                     downloaded binaries, so they live in a store outside
                     the package (fetch_assets.sh), not in web/.

The page reads its charts from foxglove_bridge, not from here: this process
only carries what has to run on the robot, which is the dead-man. The
handheld sits on the far side of a wifi link; a dead-man on the handheld
cannot zero anything once that link is gone, and policy_node latches the
last /cmd_vel it saw. So the gate lives here (drive.py) and publishes
/cmd_vel itself: sticks stream in as normalized frames, and when they stop
the gate zeroes the motion for two seconds, then goes silent.

Websocket protocol (text frames, JSON):
  server -> page
    {"t":"hello", cmd_low, cmd_high, height_range, height_default,
                  bridge_port, policy}
    {"t":"avail", "svc": {key: bool}}          which services answer
    {"t":"svc", key, value, success, message}  a service call's verdict
    {"t":"status", drive: idle|live|deadman, height, cam_hz, clients}
  page -> server
    {"t":"cmd", vx, vy, yaw, [height]}         normalized sticks, >= 10 Hz
    {"t":"stop"}                               explicit stop
    {"t":"height", "delta": +-0.005}           step the held stance height
    {"t":"call", key, [value]}                 arm/enable (bool) and the
                                               Trigger services below
    {"t":"call", "key":"restart_stack"}        restart the robot's control
                                               stack (the systemd unit in
                                               the stack_unit parameter);
                                               refused unless the robot is
                                               lying, because the stack
                                               assumes the folded pose when
                                               it starts

Threading is the web_console pattern: rclpy spins in a background thread;
the ROS side hands data to the asyncio side with call_soon_threadsafe and
the asyncio side calls into the node (publishers, async service calls),
which rclpy allows from any thread.
"""
import asyncio
import io
import os
import shutil
import signal
import threading
import time
import json
from pathlib import Path

import rclpy
from aiohttp import web
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_srvs.srv import SetBool, Trigger

from wojtek_policy.policy_source import load_meta
from wojtek_deck.drive import DriveGate

# JPEG encoder, cheapest first. OpenCV encodes through libjpeg-turbo and
# releases the GIL while it works, so the asyncio side (the drive tick that
# publishes /cmd_vel) keeps running underneath. Pillow holds the GIL for
# most of the encode: on the RPi that stalled the tick to 6 Hz and, with
# the camera node next to it, starved the control stack until the MD80
# drives dropped to idle (2026-09-12). Pillow stays as the fallback.
try:
    import cv2
except ImportError:
    cv2 = None
try:
    from PIL import Image as PILImage
except ImportError:  # soft dep: no encoder at all = empty camera stream
    PILImage = None

# Fallbacks when no policy reference is set (or it fails to load) -- same
# values and role as in gamepad_teleop / web_console.
DEFAULT_CMD_LOW = (-0.6, -0.4, -0.7)
DEFAULT_CMD_HIGH = (0.6, 0.4, 0.7)
DEFAULT_HEIGHT_RANGE = (0.09, 0.17)
DEFAULT_HEIGHT = 0.125

# Same topic + encoding wojtek_pc/camera_spec.py pins for the sim camera and
# the real perception stack publishes. Repeated here (not imported) because
# wojtek_pc is PC-only and never reaches the robot.
DEFAULT_COLOR_TOPIC = "/camera/camera/color/image_raw"
COLOR_ENCODING = "rgb8"

DRIVE_TICK_HZ = 20.0     # /cmd_vel publish rate (same as the other teleops)
STATUS_HZ = 2.0          # status frames to the page
CMD_TIMEOUT_S = 0.5      # dead-man: sticks older than this = link gone
SILENCE_AFTER_S = 2.0    # zeroing burst length before going silent

# "Lying" for the stack restart: every joint within this of the folded pose,
# which is where the encoders read zero after a boot in that pose.
LYING_MAX_RAD = 0.35

SETBOOL_SERVICES = ("arm", "enable")
TRIGGER_SERVICES = ("zero", "stand_up", "lie_down", "reset",
                    "trick_paw_wave", "trick_bow", "trick_sit", "trick_shake")


def assets_store():
    """Where the detector's downloaded files live on this machine.

    The same shape as wojtek_policy's policy_store, and for the same reason:
    they are fetched binaries that must not be committed, so they sit in a
    gitignored directory next to the workspace's `src/` -- ros/deck_assets in
    a checkout, ~/wojtek_ws/deck_assets on the robot. WOJTEK_DECK_ASSETS wins
    when set, which is how a container names it, because a build there is not
    always --symlink-install and the walk up from this file finds no src/.

    Returns None when it cannot tell; the caller then serves the panel
    without a detector rather than refusing to start.
    """
    env = os.environ.get("WOJTEK_DECK_ASSETS", "").strip()
    if env:
        return Path(env).expanduser()
    # A symlink install resolves this file into src/; a copying install
    # (the robot's) into install/. Either way the store sits next to them.
    for parent in Path(__file__).resolve().parents:
        if parent.name in ("src", "install"):
            return parent.parent / "deck_assets"
    return None


class GatewayNode(Node):
    """ROS half: service clients, the /cmd_vel publisher, the camera tap."""

    def __init__(self, emit, want_frames):
        super().__init__("deck_gateway")
        self.emit = emit                # (dict) -> None, safe from ROS thread
        self.want_frames = want_frames  # () -> bool: anyone on /stream.mjpg?
        self.on_frame = None            # set by the server: (bytes) -> None

        self.declare_parameter("policy", "")
        self.declare_parameter("port", 8090)
        # Told to the page so it knows where foxglove_bridge listens.
        self.declare_parameter("bridge_port", 8765)
        self.declare_parameter("color_topic", DEFAULT_COLOR_TOPIC)
        self.declare_parameter("jpeg_quality", 80)
        # Most frames a second that get encoded for the stream. The camera
        # may run faster; the rest are dropped before they cost anything.
        # Encoding is the gateway's whole CPU bill, and on the RPi that
        # bill is paid by the same four cores as the control loop.
        self.declare_parameter("stream_hz", 10.0)
        # Take the camera node's own JPEG (image_transport's compressed
        # plugin, <color_topic>/compressed) instead of the raw image. The
        # camera node encodes in C++ and only while somebody subscribes;
        # the gateway then receives ~40 KB a frame instead of 0.9 MB and
        # encodes nothing. false = the raw image and the encoder below,
        # for a robot without the plugin.
        self.declare_parameter("compressed", True)
        # Where the detector's files are. Empty means "work it out", which
        # is right everywhere except a container that named it differently.
        self.declare_parameter("assets_dir", "")

        self._cli = {k: self.create_client(SetBool, f"wojtek/{k}")
                     for k in SETBOOL_SERVICES}
        self._cli.update({k: self.create_client(Trigger, f"wojtek/{k}")
                          for k in TRIGGER_SERVICES})
        self._pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)

        # The control stack's systemd unit, for the panel's restart button.
        # Empty disables the button (the simulation has no such unit).
        self.declare_parameter("stack_unit", "wojtek-robot.service")
        # Where the joints are, for the "is it lying" check before a
        # restart: the largest distance from zero, and when it was seen.
        self.joint_max_rad = None
        self._joint_stamp = 0.0
        self.create_subscription(JointState, "joint_states", self._on_joints, 10)

        self.cmd_low = list(DEFAULT_CMD_LOW)
        self.cmd_high = list(DEFAULT_CMD_HIGH)
        self.height_range = list(DEFAULT_HEIGHT_RANGE)
        self.height_default = DEFAULT_HEIGHT
        self.policy_name = ""
        self._load_meta()

        self._jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self._stream_period = 1.0 / max(
            0.1, float(self.get_parameter("stream_hz").value))
        self._last_encode = 0.0   # monotonic time of the last encoded frame
        self._frame_stamps = []   # wall times of the last encoded frames
        self.frames_seen = 0      # camera messages received (status field)
        self.frames_encoded = 0   # ... of which reached a viewer
        self._last_cb = None
        # The camera subscription exists only while somebody watches the
        # stream (set_camera_wanted). Receiving the raw image costs a third
        # of a core on the RPi whether or not a frame gets encoded, and the
        # gateway is resident in the robot service now, so an unwatched
        # panel must cost nothing.
        self._color_sub = None
        self._encoder = None
        if cv2 is not None or PILImage is not None:
            self._encoder = "OpenCV" if cv2 is not None else "Pillow"
            self.get_logger().info(f"camera JPEG encoder: {self._encoder}")
        else:
            self.get_logger().warning(
                "no JPEG encoder (neither OpenCV nor Pillow) -- camera "
                "stream will stay empty (apt install python3-opencv)")

    def _load_meta(self):
        ref = self.get_parameter("policy").value
        if not ref:
            self.get_logger().warning(
                "no policy reference set; driving with default command limits")
            return
        try:
            meta, source = load_meta(ref)
        except Exception as e:  # resolver/network/file -- stay drivable
            self.get_logger().warning(
                f"could not load policy contract {ref!r} ({e}); driving "
                "with default command limits")
            return
        self.cmd_low = [float(v) for v in meta["command_low"][:3]]
        self.cmd_high = [float(v) for v in meta["command_high"][:3]]
        if len(meta["command_low"]) >= 4:
            self.height_range = [
                float(meta["command_low"][3]), float(meta["command_high"][3])
            ]
        if meta.get("command_fill"):
            self.height_default = float(meta["command_fill"][0])
        self.policy_name = str(meta.get("run_name", ""))
        self.get_logger().info(
            f"command box from {meta['run_name']} ({source})")

    # -- camera (ROS thread) -------------------------------------------------
    def set_camera_wanted(self, wanted):
        """Subscribe to the camera while a viewer is on the stream."""
        if self._encoder is None:
            return
        if wanted and self._color_sub is None:
            # The camera publishes best-effort; a default-QoS subscription
            # would match nothing, so mirror the sensor-data profile.
            topic = self.get_parameter("color_topic").value
            if self.get_parameter("compressed").value:
                self._color_sub = self.create_subscription(
                    CompressedImage, topic + "/compressed",
                    self._on_compressed, qos_profile_sensor_data)
                self.get_logger().info(
                    "camera: viewer arrived, subscribing (compressed)")
            else:
                self._color_sub = self.create_subscription(
                    Image, topic, self._on_color, qos_profile_sensor_data)
                self.get_logger().info("camera: viewer arrived, subscribing")
        elif not wanted and self._color_sub is not None:
            self.destroy_subscription(self._color_sub)
            self._color_sub = None
            self._frame_stamps = []
            self.get_logger().info("camera: last viewer gone, unsubscribed")

    def _on_compressed(self, msg):
        """A JPEG from the camera node: pass it through, nothing to encode."""
        self.frames_seen += 1
        if self.on_frame is None or not self.want_frames():
            return
        if "jpeg" not in msg.format.lower():
            self.get_logger().warning(
                f"compressed camera format {msg.format!r} is not JPEG; set "
                "the camera's compressed format to jpeg", once=True)
            return
        now = time.monotonic()
        if now - self._last_encode < self._stream_period:
            return
        self._last_encode = now
        self._frame_stamps = [t for t in self._frame_stamps if now - t < 2.0]
        self._frame_stamps.append(now)
        self.frames_encoded += 1
        if self.frames_encoded == 1:
            self.get_logger().info(
                f"camera: first frame passed through ({len(msg.data) // 1024} "
                f"KB JPEG from the camera node)")
        self.on_frame(bytes(msg.data))

    def _on_color(self, msg):
        self.frames_seen += 1
        if self.on_frame is None or not self.want_frames():
            return
        if msg.encoding != COLOR_ENCODING:
            self.get_logger().warning(
                f"unsupported camera encoding {msg.encoding!r} "
                f"(want {COLOR_ENCODING})", once=True)
            return
        now = time.monotonic()
        if now - self._last_encode < self._stream_period:
            return  # over the stream rate: dropped before it costs anything
        try:
            jpeg = self._encode(msg)
        except Exception as e:  # noqa: BLE001 -- a bad frame must not kill
            # the spin thread (see web_console for the same rule)
            self.get_logger().warning(f"dropping camera frame: {e}", once=True)
            return
        self._last_encode = now
        if self._last_cb is not None and now - self._last_cb > 1.0:
            self.get_logger().warning(
                f"camera callback starved: {now - self._last_cb:.1f} s since "
                f"the previous frame (seen {self.frames_seen})")
        self._last_cb = now
        self._frame_stamps = [t for t in self._frame_stamps if now - t < 2.0]
        self._frame_stamps.append(now)
        self.frames_encoded += 1
        if self.frames_encoded == 1:
            self.get_logger().info(
                f"camera: first frame encoded ({msg.width}x{msg.height}, "
                f"{len(jpeg) // 1024} KB JPEG)")
        t0 = time.monotonic()
        self.on_frame(jpeg)
        dt = time.monotonic() - t0
        if dt > 0.05:
            self.get_logger().warning(f"handing a frame to the server took {dt*1000:.0f} ms")

    def _encode(self, msg):
        """One rgb8 Image message -> JPEG bytes."""
        if cv2 is not None:
            import numpy as np
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            ok, out = cv2.imencode(
                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            return out.tobytes()
        img = PILImage.frombuffer(
            "RGB", (msg.width, msg.height), msg.data, "raw", "RGB", 0, 1)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self._jpeg_quality)
        return buf.getvalue()

    def cam_hz(self):
        now = time.monotonic()
        recent = [t for t in self._frame_stamps if now - t < 2.0]
        return len(recent) / 2.0

    def _on_joints(self, msg):
        if msg.position:
            self.joint_max_rad = max(abs(p) for p in msg.position)
            self._joint_stamp = time.monotonic()

    def lying(self):
        """(ok, reason): may the control stack be restarted right now?"""
        if self.joint_max_rad is None or time.monotonic() - self._joint_stamp > 1.0:
            return False, "no fresh joint states"
        if self.joint_max_rad > LYING_MAX_RAD:
            return False, (f"robot is not lying: a joint is "
                           f"{self.joint_max_rad:.2f} rad from folded")
        return True, ""

    def stack_unit(self):
        unit = str(self.get_parameter("stack_unit").value).strip()
        return unit if unit and shutil.which("systemctl") else ""

    # -- commands (asyncio thread) ---------------------------------------------
    def availability(self):
        avail = {k: c.service_is_ready() for k, c in self._cli.items()}
        avail["restart_stack"] = bool(self.stack_unit())
        return avail

    def call(self, key, value=None):
        cli = self._cli[key]
        if not cli.service_is_ready():
            self.emit({"t": "svc", "key": key, "value": value,
                       "success": False, "message": "service unavailable"})
            return
        req = (SetBool.Request(data=bool(value)) if key in SETBOOL_SERVICES
               else Trigger.Request())
        fut = cli.call_async(req)

        def done(f, key=key, value=value):
            try:
                resp = f.result()
                self.emit({"t": "svc", "key": key, "value": value,
                           "success": bool(resp.success),
                           "message": resp.message})
            except Exception as e:  # noqa: BLE001 -- surface any RPC failure
                self.emit({"t": "svc", "key": key, "value": value,
                           "success": False, "message": str(e)})
        fut.add_done_callback(done)

    def publish_cmd(self, vx, vy, yaw, height):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(yaw)
        # Standing-height command; policy_node treats 0 as "use the default".
        t.linear.z = float(height)
        self._pub_cmd.publish(t)


@web.middleware
async def cross_origin_isolation(request, handler):
    """Let the page use shared memory.

    A browser only hands a page shared memory once the page promises it is
    not sharing a process with anything it did not ask for. These two
    headers are that promise. The detector runs on one thread today
    (det_worker.js says why), so nothing needs the memory yet; the headers
    stay because they are the one thing a multithreaded runtime will ask
    for, and they cost nothing. Everything the panel loads is same-origin,
    so nothing else has to change: the camera stream, the scripts and the
    bridge websocket are unaffected.

    A response that has already started (the MJPEG stream, the websocket)
    sent its headers inside the handler, so this only decorates the rest --
    which is fine, because the promise has to be on the page itself.
    """
    resp = await handler(request)
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    # The page's own files change with every deploy, and a browser left to
    # guess from Last-Modified will happily keep yesterday's deck.js for
    # hours. no-cache means "ask first": the browser still keeps a copy and
    # revalidates it with the ETag, which is one small round trip per file.
    # The detector assets are pinned by hash and never change under the
    # same name, so those are left to cache for as long as they like.
    if not request.path.startswith("/det/"):
        resp.headers.setdefault("Cache-Control", "no-cache")
    return resp


class Server:
    """asyncio half: HTTP + websocket + MJPEG, and the drive tick."""

    def __init__(self, node, web_dir, loop, assets_dir=None):
        self.node = node
        self.web_dir = web_dir
        self.assets_dir = assets_dir
        self.loop = loop
        self.clients = set()      # websocket connections
        self.streams = set()      # asyncio.Queue per MJPEG viewer
        self.gate = DriveGate(node.cmd_low, node.cmd_high, node.height_range,
                              node.height_default, timeout_s=CMD_TIMEOUT_S,
                              silence_after_s=SILENCE_AFTER_S)
        self._last_state = None

    # -- cross-thread entry points -----------------------------------------
    def emit(self, obj):
        self.loop.call_soon_threadsafe(self._broadcast, obj)

    def push_frame(self, jpeg):
        self.loop.call_soon_threadsafe(self._fanout_frame, jpeg)

    def want_frames(self):
        return bool(self.streams)

    def _broadcast(self, obj):
        if not self.clients:
            return
        data = json.dumps(obj)
        for ws in list(self.clients):
            if not ws.closed:
                asyncio.ensure_future(ws.send_str(data))

    def _fanout_frame(self, jpeg):
        for q in self.streams:
            # Latest frame wins: a slow viewer drops frames, never lags.
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(jpeg)

    # -- HTTP handlers -----------------------------------------------------
    async def index(self, request):
        return web.FileResponse(os.path.join(self.web_dir, "index.html"),
                                headers={"Cache-Control": "no-store"})

    async def stream(self, request):
        boundary = "wojtekframe"
        resp = web.StreamResponse(status=200, headers={
            "Content-Type": f"multipart/x-mixed-replace; boundary={boundary}",
            "Cache-Control": "no-store",
        })
        await resp.prepare(request)
        q = asyncio.Queue(maxsize=1)
        self.streams.add(q)
        self.node.set_camera_wanted(True)
        try:
            while True:
                jpeg = await q.get()
                await resp.write(
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg + b"\r\n")
        except (ConnectionResetError, asyncio.CancelledError,
                ConnectionError):
            pass
        finally:
            self.streams.discard(q)
            if not self.streams:
                self.node.set_camera_wanted(False)
        return resp

    async def websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=5.0)
        await ws.prepare(request)
        self.clients.add(ws)
        await ws.send_str(json.dumps(self.hello()))
        await ws.send_str(json.dumps(
            {"t": "avail", "svc": self.node.availability()}))
        await ws.send_str(json.dumps(self.status()))
        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT:
                    continue
                try:
                    self._on_message(json.loads(msg.data))
                except (ValueError, TypeError, KeyError):
                    continue
        finally:
            self.clients.discard(ws)
            if not self.clients:
                # Last page gone: whatever it was commanding stops now.
                self.gate.stop(self.loop.time())
        return ws

    def hello(self):
        return {
            "t": "hello",
            "cmd_low": self.node.cmd_low,
            "cmd_high": self.node.cmd_high,
            "height_range": self.node.height_range,
            "height_default": self.node.height_default,
            "bridge_port": int(self.node.get_parameter("bridge_port").value),
            "policy": self.node.policy_name,
        }

    def status(self):
        return {
            "t": "status",
            "drive": self.gate.state,
            "height": self.gate.height,
            "cam_hz": self.node.cam_hz(),
            "frames_seen": self.node.frames_seen,
            "frames_encoded": self.node.frames_encoded,
            "clients": len(self.clients),
            "viewers": len(self.streams),
        }

    def _on_message(self, msg):
        t = msg.get("t")
        now = self.loop.time()
        if t == "cmd":
            self.gate.command(now, msg.get("vx", 0), msg.get("vy", 0),
                              msg.get("yaw", 0), msg.get("height"))
        elif t == "stop":
            self.gate.stop(now)
        elif t == "height":
            h = self.gate.step_height(msg.get("delta", 0.0))
            self._broadcast({"t": "status", **{k: v for k, v in
                             self.status().items() if k != "t"},
                             "height": h})
        elif t == "call":
            key = msg.get("key")
            if key in SETBOOL_SERVICES or key in TRIGGER_SERVICES:
                self.node.call(key, msg.get("value"))
            elif key == "restart_stack":
                asyncio.ensure_future(self.restart_stack())

    async def restart_stack(self):
        """Restart the control stack's systemd unit, if the robot is lying.

        The stack assumes the folded pose when it starts (real_io zeroes
        there), so a restart mid-stand would leave every joint offset
        wrong. What makes this necessary at all: after a motor power cycle
        under a running controller the drives come back idle, keep
        answering on CAN, and nothing re-enables them. On the robot the
        gateway is a node of the same launch, so the restart takes this
        process down with the stack: the page sees the socket close and
        reconnects by itself once the service is back. A success verdict
        could therefore never reach the panel; what it gets instead is a
        message sent, and flushed, before systemctl is spawned. A refusal
        (sudo, a bad unit name) exits before anything stops, so that
        verdict does arrive.
        """
        async def verdict(success, message):
            msg = json.dumps({"t": "svc", "key": "restart_stack",
                              "value": None, "success": success,
                              "message": message})
            # Awaited rather than through _broadcast: that only schedules
            # the sends, and a SIGTERM may come before they run.
            await asyncio.gather(*(ws.send_str(msg) for ws in list(self.clients)
                                   if not ws.closed), return_exceptions=True)

        unit = self.node.stack_unit()
        if not unit:
            await verdict(False, "no control stack unit configured")
            return
        ok, why = self.node.lying()
        if not ok:
            await verdict(False, why)
            return
        self.gate.stop(self.loop.time())
        self.node.get_logger().warning(f"restarting {unit} on the panel's request")
        await verdict(True, f"restart requested; LINK drops and the panel "
                            f"reconnects when {unit} is back, ~30 s")
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "-n", "systemctl", "restart", unit,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE)
            _, err = await asyncio.wait_for(proc.communicate(), timeout=90)
        except Exception as e:  # noqa: BLE001 -- report, never crash the loop
            await verdict(False, f"restart failed: {e}")
            return
        if proc.returncode != 0:
            await verdict(False, f"systemctl exited {proc.returncode}: "
                                 f"{err.decode(errors='replace').strip()[:120]}")

    # -- periodic tasks ----------------------------------------------------
    async def drive_tick(self):
        while True:
            out = self.gate.tick(self.loop.time())
            if out is not None:
                self.node.publish_cmd(*out)
            if self.gate.state != self._last_state:
                if self.gate.state == "deadman":
                    self.node.get_logger().warning(
                        "pad frames stopped -- zeroing /cmd_vel")
                else:
                    self.node.get_logger().info(f"drive {self.gate.state}")
                self._last_state = self.gate.state
                self._broadcast(self.status())
            await asyncio.sleep(1.0 / DRIVE_TICK_HZ)

    async def status_tick(self):
        last_avail = None
        while True:
            if self.clients:
                avail = self.node.availability()
                if avail != last_avail:
                    self._broadcast({"t": "avail", "svc": avail})
                    last_avail = avail
                self._broadcast(self.status())
            await asyncio.sleep(1.0 / STATUS_HZ)

    def app(self):
        app = web.Application(middlewares=[cross_origin_isolation])
        app.router.add_get("/", self.index)
        app.router.add_get("/ws", self.websocket)
        app.router.add_get("/stream.mjpg", self.stream)
        # The detector's assets, before the catch-all below: aiohttp tries
        # routes in the order they were added, and "/" matches everything.
        #
        # follow_symlinks: deploy.sh builds the workspace with
        # `colcon --symlink-install`, so every file under the installed web/
        # is a symlink into src/, and the asset store on the robot is itself
        # a symlink. aiohttp's default refuses a file whose real path lies
        # outside the static root, which turned the whole page into 404s
        # (only index.html survived, served by FileResponse above). Both
        # directories hold only what this package and fetch_assets.sh put
        # there, so following links out of them is the intended layout.
        if self.assets_dir is not None:
            app.router.add_static("/det/", str(self.assets_dir),
                                  show_index=False, follow_symlinks=True)
        # css/js next to the page; no directory listing
        app.router.add_static("/", self.web_dir, show_index=False,
                              follow_symlinks=True)
        return app


def main():
    # No rclpy signal handlers: they would shut the ROS context down under
    # the spin thread and leave the web server running with no ROS behind
    # it -- a page that loads and never gets a frame, on a port the next
    # launch cannot take. The asyncio loop below owns the signals instead
    # and takes everything down in one go.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server = None

    def emit(obj):
        if server is not None:
            server.emit(obj)

    def want_frames():
        return server is not None and server.want_frames()

    node = GatewayNode(emit, want_frames)
    port = int(node.get_parameter("port").value)
    web_dir = os.path.join(get_package_share_directory("wojtek_deck"), "web")

    # The detector's files are optional. Without them the panel is the panel
    # it always was, minus the boxes, so say so once and carry on.
    param = str(node.get_parameter("assets_dir").value).strip()
    assets_dir = Path(param).expanduser() if param else assets_store()
    if assets_dir is None or not assets_dir.is_dir():
        where = (f"no detector assets in {assets_dir}" if assets_dir
                 else "cannot tell where the detector assets are (set "
                      "WOJTEK_DECK_ASSETS or the assets_dir parameter)")
        node.get_logger().warning(
            f"{where} -- the panel will run without detection. Fetch them "
            "with ros/src/wojtek_deck/fetch_assets.sh")
        assets_dir = None
    else:
        node.get_logger().info(f"detector assets from {assets_dir}")

    server = Server(node, web_dir, loop, assets_dir)
    node.on_frame = server.push_frame

    stopping = asyncio.Event()

    def stop(why):
        if not stopping.is_set():
            node.get_logger().info(f"deck gateway stopping ({why})")
            stopping.set()

    def spin():
        # If ROS goes away underneath us (an external shutdown), the server
        # must not outlive it: nothing behind it would ever send a frame.
        try:
            rclpy.spin(node)
        except Exception:  # noqa: BLE001 -- shutdown races raise here
            pass
        finally:
            loop.call_soon_threadsafe(stop, "ROS spin ended")

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()

    # Ctrl-C at a terminal, SIGTERM from a launch file or systemd: one path
    # for both, and it ends in the finally below.
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop, signal.Signals(sig).name)

    async def run():
        runner = web.AppRunner(server.app(), access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        node.get_logger().info(f"deck panel on http://0.0.0.0:{port}")
        ticks = asyncio.gather(server.drive_tick(), server.status_tick())
        try:
            await stopping.wait()
        finally:
            ticks.cancel()
            try:
                await ticks
            except asyncio.CancelledError:
                pass
            await runner.cleanup()

    try:
        loop.run_until_complete(run())
    finally:
        # Shut ROS down first so the spin thread wakes up and returns, then
        # wait for it. Exiting while it still sits inside the executor ends
        # in a C++ abort rather than a clean exit.
        rclpy.try_shutdown()
        spinner.join(timeout=3.0)


if __name__ == "__main__":
    main()
