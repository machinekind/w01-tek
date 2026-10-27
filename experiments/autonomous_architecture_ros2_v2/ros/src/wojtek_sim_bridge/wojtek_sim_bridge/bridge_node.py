"""World-side node: the MuJoCo room sim + browser bridge, on ROS topics.

Wraps the demo's RoomSim (walking policy, SCAN planner, renderers -- the
body emulator) as a self-stepping headless node. Physics runs at 50 Hz
whether or not a browser is attached, fixing the #131 defect where the sim
froze without a viewer; camera frames for the VLM are published HUD-free,
fixing the second one.

Publishes (world -> agent):
  /wojtek/exec/status              ExecStatus       25 Hz heartbeat
  /wojtek/camera/ego/compressed    CompressedImage  10 Hz, HUD-free
  /wojtek/camera/vln/compressed    CompressedImage   5 Hz, square VLN-CE
  /wojtek/map                      WorldMap          2 Hz occupancy grid
  /wojtek/audio/mic                AudioChunk       from the websocket mic

Subscribes (agent -> world/browser):
  /wojtek/tts/audio       AudioChunk  reply voice -> browser as binary
  /wojtek/asr/final       Transcript  -> {"type": "heard"}
  /wojtek/say             Sentence    -> say captions + chat_reply
  /wojtek/audio/speech_started Empty  -> {"type": "barge_in"}
  /wojtek/trace           String      -> {"type": "trace", ...}

Serves /wojtek/world/command (midlevel / goto / reset); the request is
applied on the sim's own thread so the executor is never touched from two
threads at once.

The websocket keeps room_app's single-viewer contract (newest connection
wins) so scenario.py drives takes against this bridge exactly as it drove
room_app. It replaces wojtek_voice's audio_bridge in a sim world: do not
run both, they'd double-publish the mic.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Empty, String

from wojtek_agent_msgs.msg import (
    AudioChunk,
    ExecStatus,
    Sentence,
    Transcript,
    WorldAck,
    WorldCmd,
    WorldMap,
)
from wojtek_agent_msgs.srv import WorldCommand

from wojtek_sim_bridge.protocol import (
    SAMPLE_RATE,
    SayAccumulator,
    parse_client_message,
    world_command_result,
)

CONTROL_HZ = 50.0
STATUS_EVERY = 2   # 25 Hz
EGO_EVERY = 5      # 10 Hz
VLN_EVERY = 10     # 5 Hz
MAP_EVERY = 25     # 2 Hz
WS_STATE_EVERY = 2 # 25 Hz, only while a viewer is attached


class SimBridgeNode(Node):
    def __init__(self):
        super().__init__("sim_bridge")
        self.declare_parameter("ws_port", 8010)
        self.declare_parameter("ws_host", "0.0.0.0")

        self.pub_status = self.create_publisher(
            ExecStatus, "/wojtek/exec/status", qos_profile_sensor_data)
        self.pub_ego = self.create_publisher(
            CompressedImage, "/wojtek/camera/ego/compressed", qos_profile_sensor_data)
        self.pub_vln = self.create_publisher(
            CompressedImage, "/wojtek/camera/vln/compressed", qos_profile_sensor_data)
        self.pub_map = self.create_publisher(WorldMap, "/wojtek/map", 10)
        self.pub_mic = self.create_publisher(
            AudioChunk, "/wojtek/audio/mic", qos_profile_sensor_data)

        self.create_subscription(AudioChunk, "/wojtek/tts/audio",
                                 self.on_tts_audio, 10)
        self.create_subscription(Transcript, "/wojtek/asr/final",
                                 self.on_heard, 10)
        self.create_subscription(Sentence, "/wojtek/say", self.on_say, 10)
        self.create_subscription(Empty, "/wojtek/audio/speech_started",
                                 self.on_speech_started, 10)
        self.create_subscription(String, "/wojtek/trace", self.on_trace, 10)

        self.create_service(WorldCommand, "/wojtek/world/command",
                            self.on_world_command)
        # Topic twin of the service: zenoh bridges topics across boxes but
        # not service replies (double-router hop, measured 2026-08-27).
        self.pub_ack = self.create_publisher(WorldAck, "/wojtek/world/ack", 10)
        self.create_subscription(WorldCmd, "/wojtek/world/cmd",
                                 self.on_world_cmd, 10)

        self.loop: asyncio.AbstractEventLoop | None = None
        self.sim = None
        self._say = SayAccumulator()
        self._client = None      # newest websocket wins (single viewer)
        self._client_gen = 0
        self._voice_on = False
        self._mic_seq = 0
        self._cmd_seq = 0        # accepted WorldCommands; fences proxy polls
        self._events: list[dict] = []
        self._audio_out: list[bytes] = []

    # -- service (rclpy thread -> sim thread) ----------------------------------

    def on_world_command(self, req, res):
        if self.loop is None or self.sim is None:
            res.ok, res.error = False, "world not running yet"
            return res
        fut = asyncio.run_coroutine_threadsafe(
            self._apply_command(req.kind, req.text, list(req.args)), self.loop)
        try:
            out = fut.result(timeout=5.0)
        except Exception as e:
            out = {"ok": False, "error": f"command failed: {e}"}
        res.ok = bool(out.get("ok"))
        res.error = str(out.get("error") or "")
        res.command = str(out.get("command") or "")
        res.cmd_seq = int(out.get("cmd_seq") or 0)
        return res

    def on_world_cmd(self, msg: WorldCmd):
        if self.loop is None or self.sim is None:
            out = {"ok": False, "error": "world not running yet"}
        else:
            fut = asyncio.run_coroutine_threadsafe(
                self._apply_command(msg.kind, msg.text, list(msg.args)), self.loop)
            try:
                out = fut.result(timeout=5.0)
            except Exception as e:
                out = {"ok": False, "error": f"command failed: {e}"}
        ack = WorldAck()
        ack.header.stamp = self.get_clock().now().to_msg()
        ack.req_seq = msg.req_seq
        ack.ok = bool(out.get("ok"))
        ack.error = str(out.get("error") or "")
        ack.command = str(out.get("command") or "")
        ack.cmd_seq = int(out.get("cmd_seq") or 0)
        self.pub_ack.publish(ack)

    async def _apply_command(self, kind: str, text: str, args: list[float]):
        out = world_command_result(kind, text, args, self.sim)
        if out.get("ok"):
            # Stamped on the sim thread, so the NEXT sim.step() -- and the
            # status message it produces -- already reflects this command.
            self._cmd_seq += 1
            out["cmd_seq"] = self._cmd_seq
        return out

    # -- agent -> browser fan-out (rclpy thread) --------------------------------

    def _send_json(self, payload: dict):
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._queue_event, payload)

    def _queue_event(self, payload: dict):
        # Runs on the asyncio loop; events ride the same queue as state ticks.
        # Bounded: with no viewer attached the queue would otherwise grow for
        # as long as the session runs.
        self._events.append(payload)
        del self._events[:-200]

    def on_tts_audio(self, msg: AudioChunk):
        import numpy as np
        pcm = np.asarray(msg.samples, np.int16).tobytes()
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._queue_audio, pcm)

    def _queue_audio(self, pcm: bytes):
        self._audio_out.append(pcm)
        del self._audio_out[:-100]

    def on_heard(self, msg: Transcript):
        self._send_json({"type": "heard", "text": msg.text})

    def on_say(self, msg: Sentence):
        for event in self._say.feed(msg.utterance_id, msg.text, msg.final,
                                    msg.source):
            self._send_json(event)

    def on_speech_started(self, _msg: Empty):
        self._say.drop()
        self._send_json({"type": "barge_in"})

    def on_trace(self, msg: String):
        try:
            event = json.loads(msg.data)
        except ValueError:
            return
        self._send_json({"type": "trace", **event})

    # -- websocket server (asyncio) ---------------------------------------------

    async def _ws_handler(self, sock):
        self._client_gen += 1
        gen = self._client_gen
        old, self._client = self._client, sock
        if old is not None:
            try:
                await old.send(json.dumps(
                    {"type": "error", "error": "another viewer took over"}))
                await old.close()
            except Exception:
                pass
        self.get_logger().info("viewer connected")
        try:
            async for packet in sock:
                if isinstance(packet, bytes):
                    if self._voice_on:
                        self._publish_mic(packet)
                    continue
                kind, val = parse_client_message(packet)
                if kind == "voice":
                    self._voice_on = val["on"]
                    await sock.send(json.dumps(
                        {"type": "voice_state", "on": self._voice_on}))
                elif kind == "command":
                    ack = self.sim.submit_command(val["text"])
                    await sock.send(json.dumps({"type": "command_ack", **ack}))
                elif kind == "reset":
                    self.sim.reset()
                elif kind == "unknown":
                    self.get_logger().warning(f"unknown ws message: {val}")
        except Exception as e:
            self.get_logger().info(f"viewer gone: {e}")
        finally:
            if self._client_gen == gen:
                self._client = None

    def _publish_mic(self, buf: bytes):
        import numpy as np
        msg = AudioChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.sample_rate = SAMPLE_RATE
        msg.channels = 1
        self._mic_seq += 1
        msg.seq = self._mic_seq
        msg.samples = np.frombuffer(buf, dtype=np.int16).tolist()
        self.pub_mic.publish(msg)

    # -- the world loop ------------------------------------------------------------

    async def run(self):
        self.loop = asyncio.get_running_loop()

        # Heavy import deferred to here: env (SCENE, WOJTEK_POLICY, ...) is
        # already set by the launch, and unit tests never touch it.
        from wojtek_rl import room_app
        self.sim = room_app.get_sim()
        self.get_logger().info("sim up; stepping at 50 Hz headless")

        import websockets
        host = self.get_parameter("ws_host").value
        port = int(self.get_parameter("ws_port").value)
        async with websockets.serve(self._ws_handler, host, port,
                                    max_size=2**20):
            self.get_logger().info(f"websocket bridge on :{port}")
            await self._sim_loop()

    async def _sim_loop(self):
        dt = 1.0 / CONTROL_HZ
        i = 0
        while rclpy.ok():
            t0 = self.loop.time()
            payload = self.sim.step()
            if i % STATUS_EVERY == 0:
                self._publish_status(payload)
            if i % EGO_EVERY == 0:
                self._publish_jpeg(self.pub_ego, self.sim.ego_jpeg(hud=False))
            if i % VLN_EVERY == 0:
                self._publish_jpeg(self.pub_vln, self.sim.vlm_frame_jpeg())
            if i % MAP_EVERY == 0:
                self._publish_map()
            if self._client is not None and i % WS_STATE_EVERY == 0:
                await self._ws_tick(payload)
            i += 1
            await asyncio.sleep(max(0.0, dt - (self.loop.time() - t0)))

    def _publish_status(self, payload: dict):
        msg = ExecStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.x, msg.y, msg.yaw = (float(payload["x"]), float(payload["y"]),
                                 float(payload["yaw"]))
        exec_status = payload.get("exec") or {}
        # executor.active is the property the controllers poll; the status
        # dict's "active" field is a human-readable describe string.
        msg.active = bool(self.sim.executor.active)
        msg.blocked = int(self.sim.executor.blocked)
        msg.resets = int(self.sim.resets)
        msg.sim_time = float(self.sim.sim_time)
        msg.exec_json = json.dumps(exec_status)
        msg.cmd_seq = self._cmd_seq
        self.pub_status.publish(msg)

    def _publish_jpeg(self, pub, b64: str):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = base64.b64decode(b64)
        pub.publish(msg)

    def _publish_map(self):
        omap = self.sim.omap
        msg = WorldMap()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.res = float(omap.res)
        msg.origin = [float(omap.origin[0]), float(omap.origin[1])]
        msg.shape = [int(omap.shape[0]), int(omap.shape[1])]
        msg.state = omap.state.reshape(-1).tolist()
        trail = omap.trail[-600:]
        msg.trail = [float(v) for xy in trail for v in xy]
        msg.ego_fovy_deg = float(self.sim._ego_fovy)
        self.pub_map.publish(msg)

    async def _ws_tick(self, payload: dict):
        sock = self._client
        if sock is None:
            return
        try:
            payload = dict(payload)
            payload["frame"], payload["ego"] = self.sim.render_pair()
            payload["map"] = self.sim.map_jpeg()
            events, self._events = self._events, []
            for event in events:
                await sock.send(json.dumps(event, ensure_ascii=False))
            audio, self._audio_out = self._audio_out, []
            for pcm in audio:
                await sock.send(pcm)
            await sock.send(json.dumps(payload))
        except Exception:
            self._client = None


def main():
    rclpy.init()
    node = SimBridgeNode()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        asyncio.run(node.run())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
