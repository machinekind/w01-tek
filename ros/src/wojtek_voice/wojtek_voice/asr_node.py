"""ASR node: closed utterances in, final transcripts out.

Consumes the end_of_utterance chunk of /wojtek/audio/speech (which carries
the full PCM) and publishes /wojtek/asr/final.  Recognition is blocking, so
it runs on one worker thread fed by a queue — the subscription callback
never waits on the GPU, and utterances that pile up behind a slow decode are
processed in order rather than dropped.

Partial transcripts (/wojtek/asr/partial from the live chunks) are a planned
W2 addition; the message contract already supports them.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import rclpy
from rclpy.node import Node

from wojtek_agent_msgs.msg import AudioChunk, Transcript

from .asr_engine import WhisperEngine


class AsrNode(Node):
    def __init__(self):
        super().__init__("wojtek_asr")
        self.declare_parameter("model", "large-v3")
        self.declare_parameter("device", "auto")
        self.declare_parameter("compute", "default")
        self.declare_parameter("language", "pl")

        self.engine = WhisperEngine(
            model_size=self.get_parameter("model").value,
            device=self.get_parameter("device").value,
            compute_type=self.get_parameter("compute").value,
            language=self.get_parameter("language").value,
        )
        self.pub_final = self.create_publisher(Transcript, "/wojtek/asr/final", 10)
        self.create_subscription(AudioChunk, "/wojtek/audio/speech", self.on_chunk, 10)

        self._q: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def on_chunk(self, msg: AudioChunk):
        if not msg.end_of_utterance:
            return  # live chunks are for streaming consumers, not for us
        pcm = np.asarray(msg.samples, dtype=np.int16)
        self._q.put((msg.utterance_id, pcm, msg.sample_rate))

    def _work(self):
        while True:
            uid, pcm, rate = self._q.get()
            try:
                rec = self.engine.transcribe(pcm, rate)
            except Exception as e:  # a bad utterance must not kill the worker
                self.get_logger().warning(f"transcription failed for {uid}: {e}")
                continue
            if rec.dropped:
                self.get_logger().info(f"dropped low-confidence segments: {rec.dropped}")
            if not rec.text:
                continue
            out = Transcript()
            out.header.stamp = self.get_clock().now().to_msg()
            out.utterance_id = uid
            out.text = rec.text
            out.final = True
            out.confidence = rec.confidence
            out.language = self.engine.language
            self.get_logger().info(f"heard ({uid}): {rec.text!r}")
            self.pub_final.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = AsrNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
