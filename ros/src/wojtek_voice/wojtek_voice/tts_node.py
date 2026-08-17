"""TTS node: reply sentences in, playback audio frames out.

Subscribes /wojtek/say (already speakable text, sentence-chunked by the
brain), synthesizes with the configured engine and publishes
/wojtek/tts/audio at the browser rate.  /wojtek/audio/speech_started cuts
playback (barge-in).
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty

from wojtek_agent_msgs.msg import AudioChunk, Sentence

from .playback import SpeechSynthesizer
from .transport import SAMPLE_RATE
from .tts_engines import build_engine


class TtsNode(Node):
    def __init__(self):
        super().__init__("wojtek_tts")
        self.declare_parameter("engine", "chatterbox")
        self.declare_parameter("language", "pl")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("ref_wav", "")     # voice clone reference
        self.declare_parameter("ref_text", "")    # its exact transcript (F5)

        kind = self.get_parameter("engine").value
        kwargs = {}
        if kind != "silent":
            kwargs = dict(
                ref_wav=self.get_parameter("ref_wav").value,
                language=self.get_parameter("language").value,
                device=self.get_parameter("device").value,
            )
            if kind == "f5":
                kwargs["ref_text"] = self.get_parameter("ref_text").value
        engine = build_engine(kind, **kwargs)
        self.get_logger().info(f"TTS engine: {kind}")

        self.pub = self.create_publisher(AudioChunk, "/wojtek/tts/audio", 10)
        self._seq = 0
        self.synth = SpeechSynthesizer(
            engine,
            emit_frame=self._publish_frame,
            on_error=lambda text, e: self.get_logger().warning(
                f"synthesis failed for {text!r}: {e}"
            ),
        )
        self.synth.start()

        self.create_subscription(Sentence, "/wojtek/say", self.on_sentence, 10)
        self.create_subscription(
            Empty, "/wojtek/audio/speech_started", self.on_barge_in, 10
        )

    def on_sentence(self, msg: Sentence):
        if msg.text:  # a bare final marker carries no audio
            # The utterance id rides along so the published frames say which
            # question they answer; the latency probe needs that pairing to
            # measure the wait between a human speaking and hearing.
            self.synth.say(msg.text, tag=msg.utterance_id)

    def on_barge_in(self, _msg: Empty):
        self.synth.cancel()

    def _publish_frame(self, frame):
        msg = AudioChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.sample_rate = SAMPLE_RATE
        msg.channels = 1
        msg.utterance_id = self.synth.current_tag or ""
        self._seq += 1
        msg.seq = self._seq
        msg.samples = frame.tolist()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TtsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
