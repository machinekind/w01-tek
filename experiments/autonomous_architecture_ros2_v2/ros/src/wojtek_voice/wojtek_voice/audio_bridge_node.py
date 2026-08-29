"""Audio bridge: the laptop/stage browser is a dumb mic and speaker.

One websocket endpoint speaks the #131 wire protocol (the room_app mic
path), so the existing browser worklet works unchanged:

  browser -> binary frame  = 100 ms mono PCM16 @ 24 kHz  -> /wojtek/audio/mic
  browser -> JSON          = {"type": "mic", "on": bool}    gates publishing
  /wojtek/tts/audio        -> binary frame to the browser (playback)
  /wojtek/audio/speech_started -> {"type": "flush"}         barge-in: drop
                                                            queued playback

rclpy spins on a background thread; asyncio owns the main thread.  Messages
cross from executor callbacks into the loop via call_soon_threadsafe, and
from the websocket reader into publishers directly (rcl publish is safe off
the executor thread).
"""

from __future__ import annotations

import asyncio
import json
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Empty

from wojtek_agent_msgs.msg import AudioChunk, Sentence, Transcript

from .transport import SAMPLE_RATE


class AudioBridgeNode(Node):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__("wojtek_audio_bridge")
        self.declare_parameter("port", 8765)
        self.loop = loop
        self._ws_clients: set = set()
        self.mic_on = False
        self._seq = 0

        self.pub_mic = self.create_publisher(
            AudioChunk, "/wojtek/audio/mic", qos_profile_sensor_data
        )
        self.create_subscription(AudioChunk, "/wojtek/tts/audio", self.on_tts, 10)
        self.create_subscription(
            Empty, "/wojtek/audio/speech_started", self.on_speech_started, 10
        )
        # Mirrored to the browser for visibility only — the pipeline's
        # consumers read the topics, not the websocket.
        self.create_subscription(Transcript, "/wojtek/asr/final", self.on_final, 10)
        self.create_subscription(Sentence, "/wojtek/say", self.on_say, 10)

    # -- executor thread -> event loop ------------------------------------
    def on_tts(self, msg: AudioChunk):
        data = np.asarray(msg.samples, dtype=np.int16).tobytes()
        self.loop.call_soon_threadsafe(self._broadcast, data)

    def on_speech_started(self, _msg: Empty):
        self.loop.call_soon_threadsafe(
            self._broadcast, json.dumps({"type": "flush"})
        )

    def on_final(self, msg: Transcript):
        self.loop.call_soon_threadsafe(
            self._broadcast, json.dumps({"type": "transcript", "text": msg.text})
        )

    def on_say(self, msg: Sentence):
        if msg.text:
            self.loop.call_soon_threadsafe(
                self._broadcast,
                json.dumps({"type": "say", "text": msg.text, "source": msg.source}),
            )

    def _broadcast(self, payload):
        for ws in list(self._ws_clients):
            asyncio.ensure_future(self._send(ws, payload))

    async def _send(self, ws, payload):
        try:
            await ws.send(payload)
        except Exception:
            self._ws_clients.discard(ws)

    # -- websocket thread (event loop) -> publishers ----------------------
    def publish_mic(self, raw: bytes):
        if not self.mic_on or not raw:
            return
        msg = AudioChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.sample_rate = SAMPLE_RATE
        msg.channels = 1
        self._seq += 1
        msg.seq = self._seq
        msg.samples = np.frombuffer(raw, dtype=np.int16).tolist()
        self.pub_mic.publish(msg)

    async def handler(self, ws):
        self._ws_clients.add(ws)
        self.get_logger().info("browser connected")
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    self.publish_mic(message)
                else:
                    self.control(message)
        finally:
            self._ws_clients.discard(ws)
            self.get_logger().info("browser disconnected")

    def control(self, message: str):
        try:
            cmd = json.loads(message)
        except json.JSONDecodeError:
            self.get_logger().warning(f"bad control message: {message[:80]!r}")
            return
        if cmd.get("type") == "mic":
            self.mic_on = bool(cmd.get("on"))
            self.get_logger().info(f"mic {'on' if self.mic_on else 'off'}")


async def serve(node: AudioBridgeNode):
    import websockets

    port = node.get_parameter("port").value
    async with websockets.serve(node.handler, "0.0.0.0", port, max_size=2**20):
        node.get_logger().info(f"audio bridge listening on :{port}")
        await asyncio.Future()  # run until cancelled


def main(args=None):
    rclpy.init(args=args)
    loop = asyncio.new_event_loop()
    node = AudioBridgeNode(loop)

    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        loop.run_until_complete(serve(node))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
