"""VAD node: raw mic stream in, segmented utterances out.

Subscribes /wojtek/audio/mic (BestEffort — a dropped mic frame is stale
immediately) and publishes /wojtek/audio/speech (Reliable — an utterance
chunk must arrive) plus the /wojtek/audio/speech_started event that drives
barge-in.  The closing chunk carries the full utterance PCM, so the ASR
needs no reassembly (see AudioChunk.msg).
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Empty

from wojtek_agent_msgs.msg import AudioChunk

from .transport import UtteranceStream, VoiceSegmenter
from .vad_backends import make_vad


class VadNode(Node):
    def __init__(self):
        super().__init__("wojtek_vad")
        self.declare_parameter("backend", "silero")
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("silence_end_s", 0.7)
        self.declare_parameter("max_utterance_s", 20.0)

        seg = VoiceSegmenter(
            silence_end_s=self.get_parameter("silence_end_s").value,
            max_utterance_s=self.get_parameter("max_utterance_s").value,
        )
        backend = self.get_parameter("backend").value
        seg.vad = make_vad(
            backend, seg.sample_rate, self.get_parameter("threshold").value
        )
        self.stream = UtteranceStream(seg)
        self.get_logger().info(f"VAD backend: {backend}")

        self.pub_speech = self.create_publisher(AudioChunk, "/wojtek/audio/speech", 10)
        self.pub_started = self.create_publisher(Empty, "/wojtek/audio/speech_started", 10)
        self.create_subscription(
            AudioChunk, "/wojtek/audio/mic", self.on_mic, qos_profile_sensor_data
        )
        self._seq = 0
        self._rate = seg.sample_rate

    def on_mic(self, msg: AudioChunk):
        pcm = np.asarray(msg.samples, dtype=np.int16)
        if msg.sample_rate != self._rate:
            # The segmenter is configured for the bridge's rate; a mismatch is
            # a wiring error worth one loud line, not a silent resample.
            self.get_logger().error(
                f"mic frame at {msg.sample_rate} Hz, expected {self._rate}", once=True
            )
            return
        for event, uid, payload in self.stream.feed(pcm):
            if event == "started":
                self.pub_started.publish(Empty())
            elif event == "frame":
                self._publish_chunk(uid, payload, end=False)
            elif event == "ended":
                self._publish_chunk(uid, payload.pcm, end=True)
                self.get_logger().info(
                    f"utterance {payload.seconds:.1f}s ({payload.ended_on})"
                )
            # "aborted": too short to be speech; nothing downstream cares.

    def _publish_chunk(self, uid: str, pcm: np.ndarray, end: bool):
        msg = AudioChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.sample_rate = self._rate
        msg.channels = 1
        self._seq += 1
        msg.seq = self._seq
        msg.utterance_id = uid
        msg.end_of_utterance = end
        msg.samples = pcm.tolist()
        self.pub_speech.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
