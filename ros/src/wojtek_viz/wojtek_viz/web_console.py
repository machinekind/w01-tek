#!/usr/bin/env python3
"""Web operator console -- the browser twin of operator_console.py.

    ros2 run wojtek_viz web_console        # then open http://localhost:8080

Same manual-control surface as the Qt console (services / telemetry / joint
jog / drive pad), served as a single self-contained web page instead of an
X11 window -- so it works on macOS (no XQuartz), headless boxes, and phones
or tablets joined to the robot's AP. It talks to the SAME services/topics
the nodes already expose; nothing in wojtek_bringup / wojtek_policy changes.

The page also reads a bluetooth gamepad via the browser's Gamepad API (pair
the pad with the machine running the BROWSER, press any pad button on the
page): left stick vx/yaw, right stick strafe, A arms, D-pad height -- the
same mapping as wojtek_viz/gamepad_teleop.py, which is the in-container
(Linux) alternative. All pad traffic rides the existing cmd stream, so the
dead-man below covers it too.

One port (default 8080, parameter `port`): plain GET serves the page,
a websocket upgrade on the same port carries a small JSON protocol.

Control logic that matters for safety stays HERE, not in the browser:
  * Jog ramping runs server-side at JOG_TICK_HZ (the browser only moves
    set-points), same rates as the Qt console.
  * Drive commands are dead-man guarded: the page streams pad state while
    driving, and if updates stop (tab closed, wifi drop) for more than
    CMD_TIMEOUT_S the server zeroes /cmd_vel. The local Qt console does not
    need this; a phone on the robot's wifi does.

Threading mirrors the Qt console: rclpy spins in a background thread and
events cross into the asyncio (websocket) loop via call_soon_threadsafe.
"""
import asyncio
import http
import json
import os
import threading
import xml.etree.ElementTree as ET

import rclpy
import websockets
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

# The drive controls scale to the loaded policy's trained command box
# ([vx, vy, wz], per axis and asymmetric -- e.g. stiff_b trained on
# vx -0.8..+1.2, wz +-1.0) read from policy_meta.json deploy.command_low/
# high, same source policy_node clips against. These are only the fallbacks
# for a meta without a deploy box (they match policy.py's own defaults).
DEFAULT_CMD_LOW = (-0.6, -0.4, -0.7)
DEFAULT_CMD_HIGH = (0.6, 0.4, 0.7)
# Commanded standing height (m): trained envelope and default stance, also
# read from the meta (deploy.command_height*). Published on Twist linear.z
# (0 = use policy default).
DEFAULT_HEIGHT_RANGE = (0.09, 0.17)
DEFAULT_HEIGHT = 0.125

JOG_RAMP_RATE = 0.8      # rad/s -- how fast published targets chase set-points
JOG_TICK_HZ = 50.0
DRIVE_TICK_HZ = 20.0     # /cmd_vel publish rate while driving
TELEMETRY_HZ = 10.0      # joints/imu push rate to the browser
CMD_TIMEOUT_S = 0.5      # dead-man: zero /cmd_vel if the page goes silent
DEFAULT_JOINT_LIMIT = 3.14


class ConsoleNode(Node):
    """ROS half -- a port of operator_console.RosNode with the Qt signal
    plumbing replaced by a thread-safe `emit` into the asyncio loop."""

    def __init__(self, emit):
        super().__init__("web_console")
        self.emit = emit  # emit(dict) -- safe to call from ROS thread

        self._cli = {
            "arm": self.create_client(SetBool, "wojtek/arm"),
            "enable": self.create_client(SetBool, "wojtek/enable"),
            "zero": self.create_client(Trigger, "wojtek/zero"),
            "stand_up": self.create_client(Trigger, "wojtek/stand_up"),
            "lie_down": self.create_client(Trigger, "wojtek/lie_down"),
            "reset": self.create_client(Trigger, "wojtek/reset"),
        }
        self._pub_targets = self.create_publisher(JointState, "wojtek/joint_targets", 10)
        self._pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)

        self.cmd_low = list(DEFAULT_CMD_LOW)
        self.cmd_high = list(DEFAULT_CMD_HIGH)
        self.height_range = list(DEFAULT_HEIGHT_RANGE)
        self.height_default = DEFAULT_HEIGHT
        self.joint_names = self._load_meta()
        self.limits = {}          # joint -> (lower, upper), from URDF
        self.measured = {}        # joint -> measured position (rad)

        # Measured angles: prefer wojtek/joint_states_abs (real_io, absolute
        # URDF); in sim only plain joint_states exists (also absolute there).
        # On the real robot joint_states is boot-relative, so once abs has
        # been seen the raw topic is ignored. Same rule as the Qt console.
        self._have_abs = False
        self.create_subscription(
            JointState, "wojtek/joint_states_abs", self._on_joints_abs, 10)
        self.create_subscription(
            JointState, "joint_states", self._on_joints_raw, 10)
        self.create_subscription(
            Imu, "imu_sensor_broadcaster/imu", self._on_imu, 10)
        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "robot_description", self._on_urdf, latched)

        # latest telemetry, pushed to browsers on a slow tick (not per message)
        self.imu_rpy_gyro = None

    def _load_meta(self):
        try:
            share = get_package_share_directory("wojtek_policy")
            meta = json.loads(
                open(os.path.join(share, "config", "policy_meta.json")).read())
        except (PackageNotFoundError, OSError, ValueError) as e:
            self.get_logger().warning(f"no policy meta ({e}); jog builds from "
                                      "telemetry, drive uses default limits")
            return []
        deploy = meta.get("deploy", {})
        self.cmd_low = list(deploy.get("command_low", DEFAULT_CMD_LOW))[:3]
        self.cmd_high = list(deploy.get("command_high", DEFAULT_CMD_HIGH))[:3]
        self.height_range = [
            deploy.get("command_height_low", DEFAULT_HEIGHT_RANGE[0]),
            deploy.get("command_height_high", DEFAULT_HEIGHT_RANGE[1]),
        ]
        self.height_default = deploy.get("command_height", 0.0) or DEFAULT_HEIGHT
        return list(meta.get("actuator_names", []))

    # ---- telemetry callbacks (ROS thread) ----------------------------------
    def _on_joints_abs(self, msg):
        self._have_abs = True
        self._store_joints(msg)

    def _on_joints_raw(self, msg):
        if self._have_abs:
            return
        self._store_joints(msg)

    def _store_joints(self, msg):
        self.measured = dict(zip(msg.name, msg.position))
        if not self.joint_names:
            self.joint_names = list(msg.name)
            self.emit({"t": "config", **self.config()})

    def _on_imu(self, msg):
        import math
        q, w = msg.orientation, msg.angular_velocity
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.imu_rpy_gyro = ((roll, pitch, yaw), (w.x, w.y, w.z))

    def _on_urdf(self, msg):
        limits = {}
        try:
            root = ET.fromstring(msg.data)
            for j in root.iter("joint"):
                lim = j.find("limit")
                name = j.get("name")
                if name is not None and lim is not None and lim.get("lower") is not None:
                    limits[name] = (float(lim.get("lower")), float(lim.get("upper")))
        except ET.ParseError:
            return
        if limits:
            self.limits = limits
            self.emit({"t": "config", **self.config()})

    def config(self):
        return {
            "joints": self.joint_names,
            "limits": {n: self.limits.get(n, (-DEFAULT_JOINT_LIMIT, DEFAULT_JOINT_LIMIT))
                       for n in self.joint_names},
            "cmd_low": self.cmd_low,
            "cmd_high": self.cmd_high,
            "height_range": self.height_range,
            "height_default": self.height_default,
        }

    # ---- commands (called from the asyncio thread) --------------------------
    def availability(self):
        return {k: c.service_is_ready() for k, c in self._cli.items()}

    def call(self, key, value=None):
        cli = self._cli[key]
        if not cli.service_is_ready():
            self.emit({"t": "svc", "key": key, "success": False,
                       "message": "service unavailable"})
            return
        req = SetBool.Request(data=bool(value)) if key in ("arm", "enable") \
            else Trigger.Request()
        fut = cli.call_async(req)

        def done(f, key=key, value=value):
            try:
                resp = f.result()
                self.emit({"t": "svc", "key": key, "value": value,
                           "success": bool(resp.success), "message": resp.message})
            except Exception as e:  # noqa: BLE001 -- surface any RPC failure
                self.emit({"t": "svc", "key": key, "value": value,
                           "success": False, "message": str(e)})
        fut.add_done_callback(done)

    def publish_cmd(self, vx, vy, yaw, height=0.0):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(yaw)
        # Standing-height command; policy_node treats 0 as "use the default".
        t.linear.z = float(height)
        self._pub_cmd.publish(t)

    def publish_targets(self, names, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(p) for p in positions]
        self._pub_targets.publish(msg)


class Server:
    """Asyncio half: websocket protocol, jog/drive/telemetry ticks."""

    def __init__(self, node: ConsoleNode, page: bytes, loop):
        self.node = node
        self.page = page
        self.loop = loop
        self.clients = set()

        # drive state (dead-man guarded); height is an absolute set-point (m)
        # that holds across stop/drive-disable so the stance doesn't jump
        self.drive_on = False
        self.cmd = (0.0, 0.0, 0.0)          # normalized [-1, 1]
        self.height = node.height_default
        self.cmd_stamp = 0.0                # loop.time() of last cmd message

        # jog state (ramped server-side, mirrors the Qt console)
        self.jog_target = None              # {name: currently-published (rad)}
        self.jog_setpoint = {}              # {name: slider set-point (rad)}

    def emit(self, obj):
        """Broadcast to all pages; safe to call from the ROS thread."""
        self.loop.call_soon_threadsafe(self._broadcast, obj)

    def _broadcast(self, obj):
        if self.clients:
            websockets.broadcast(self.clients, json.dumps(obj))

    # ---- per-client ---------------------------------------------------------
    async def handler(self, ws):
        self.clients.add(ws)
        await ws.send(json.dumps({"t": "config", **self.node.config()}))
        await ws.send(json.dumps({"t": "avail", "svc": self.node.availability()}))
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                self._on_message(msg)
        finally:
            self.clients.discard(ws)
            if not self.clients:
                # last page gone: stop everything it may have been commanding
                self._drive_stop()
                self.jog_target = None

    def _on_message(self, msg):
        t = msg.get("t")
        if t == "call":
            key = msg.get("key")
            if key in ("arm", "enable", "zero", "stand_up", "lie_down", "reset"):
                self.node.call(key, msg.get("value"))
        elif t == "drive":
            if msg.get("on"):
                self.drive_on = True
            else:
                self._drive_stop()
        elif t == "cmd":
            self.cmd = (float(msg.get("vx", 0)), float(msg.get("vy", 0)),
                        float(msg.get("yaw", 0)))
            if "height" in msg:
                lo, hi = self.node.height_range
                self.height = min(max(float(msg["height"]), lo), hi)
            self.cmd_stamp = self.loop.time()
        elif t == "jog":
            if msg.get("on"):
                # start from measured so the robot doesn't jump -- refuse if
                # there's nothing measured yet (same rule as the Qt console)
                names = self.node.joint_names
                meas = self.node.measured
                if not names or not any(n in meas for n in names):
                    self._broadcast({"t": "jog_on", "ok": False,
                                     "message": "no measured joint angles yet"})
                    return
                self.jog_target = {n: meas.get(n, 0.0) for n in names}
                self.jog_setpoint = dict(self.jog_target)
                self._broadcast({"t": "jog_on", "ok": True,
                                 "targets": self.jog_target})
            else:
                self.jog_target = None
        elif t == "jog_set":
            if self.jog_target is not None:
                for n, v in (msg.get("targets") or {}).items():
                    if n in self.jog_target:
                        self.jog_setpoint[n] = float(v)

    def _drive_stop(self):
        if self.drive_on:
            # explicit stop; keep the height set-point so the stance holds
            self.node.publish_cmd(0.0, 0.0, 0.0, self.height)
        self.drive_on = False
        self.cmd = (0.0, 0.0, 0.0)

    # ---- periodic tasks ------------------------------------------------------
    async def drive_tick(self):
        while True:
            if self.drive_on:
                if self.loop.time() - self.cmd_stamp > CMD_TIMEOUT_S:
                    # dead-man: page went silent mid-drive -- zero the motion
                    # but hold the stance height
                    self.node.publish_cmd(0.0, 0.0, 0.0, self.height)
                else:
                    # normalized [-1,1] -> the trained (asymmetric) command
                    # box: positive stick scales by high, negative by low
                    lo, hi = self.node.cmd_low, self.node.cmd_high
                    vx, vy, yaw = (v * hi[i] if v >= 0 else v * -lo[i]
                                   for i, v in enumerate(self.cmd))
                    self.node.publish_cmd(vx, vy, yaw, self.height)
            await asyncio.sleep(1.0 / DRIVE_TICK_HZ)

    async def jog_tick(self):
        step = JOG_RAMP_RATE / JOG_TICK_HZ
        while True:
            if self.jog_target is not None:
                names = list(self.jog_target)
                for n in names:
                    d = self.jog_setpoint.get(n, self.jog_target[n]) - self.jog_target[n]
                    self.jog_target[n] += max(-step, min(step, d))
                self.node.publish_targets(names, [self.jog_target[n] for n in names])
            await asyncio.sleep(1.0 / JOG_TICK_HZ)

    async def telemetry_tick(self):
        while True:
            if self.clients:
                out = {"t": "telemetry", "meas": self.node.measured}
                if self.node.imu_rpy_gyro is not None:
                    rpy, gyro = self.node.imu_rpy_gyro
                    out["imu"] = {"rpy": rpy, "gyro": gyro}
                if self.jog_target is not None:
                    out["jog_cmd"] = self.jog_target
                self._broadcast(out)
            await asyncio.sleep(1.0 / TELEMETRY_HZ)

    async def avail_tick(self):
        last = None
        while True:
            if self.clients:
                avail = self.node.availability()
                if avail != last:
                    self._broadcast({"t": "avail", "svc": avail})
                    last = avail
            await asyncio.sleep(1.0)

    # ---- HTTP: serve the page on the same port -------------------------------
    def process_request(self, path, request_headers):
        if request_headers.get("Upgrade", "").lower() == "websocket":
            return None  # continue with the websocket handshake
        if path in ("/", "/index.html"):
            return (http.HTTPStatus.OK,
                    [("Content-Type", "text/html; charset=utf-8"),
                     ("Cache-Control", "no-store")],
                    self.page)
        return (http.HTTPStatus.NOT_FOUND, [("Content-Type", "text/plain")],
                b"not found\n")


def main():
    rclpy.init()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server = None  # set below; node.emit forwards into it once ready

    def emit(obj):
        if server is not None:
            server.emit(obj)

    node = ConsoleNode(emit)
    port = node.declare_parameter("port", 8080).value

    share = get_package_share_directory("wojtek_viz")
    page = open(os.path.join(share, "web", "index.html"), "rb").read()
    server = Server(node, page, loop)

    def spin():
        # On SIGINT rclpy's own handler shuts the context down under the
        # spinner, which then raises -- an expected teardown race, not an error.
        try:
            rclpy.spin(node)
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=spin, daemon=True).start()

    async def run():
        async with websockets.serve(server.handler, "0.0.0.0", port,
                                    process_request=server.process_request):
            node.get_logger().info(f"web console on http://localhost:{port}")
            await asyncio.gather(server.drive_tick(), server.jog_tick(),
                                 server.telemetry_tick(), server.avail_tick())

    try:
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()  # SIGINT may have shut the context down already


if __name__ == "__main__":
    main()
