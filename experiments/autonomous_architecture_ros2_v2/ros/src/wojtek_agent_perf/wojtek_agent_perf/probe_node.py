"""Latency probe: listens to the whole agent pipeline, changes none of it.

A passive subscriber on the five topics a turn passes through.  It adds no
work to any node's callback, cannot deadlock the graph, and can be launched
or killed mid-session; the cost of measuring is one extra subscriber per
topic (and `/wojtek/tts/audio` is filtered to the first frame of each reply,
so the audio stream is not re-serialised for a stopwatch).

Output is `perf.span` JSONL in the demo stack's format:

    ros2 run wojtek_agent_perf probe --ros-args -p out:=/tmp/wojtek_perf.jsonl
    ./training/run.sh perf /tmp/wojtek_perf.jsonl

`vad_silence_s` must match the VAD node's `silence_end_s`; it is the one
stage nobody can observe from the outside (see latency.py).
"""

from __future__ import annotations

from pathlib import Path

import rclpy
from rclpy.node import Node

from wojtek_agent_msgs.msg import AudioChunk, RoutedIntent, Sentence, Transcript

from .latency import LatencyProbe, SpanWriter, stamp_to_s


class ProbeNode(Node):
    def __init__(self):
        super().__init__("wojtek_latency_probe")
        self.declare_parameter("out", "")
        self.declare_parameter("vad_silence_s", 0.7)
        self.declare_parameter("log_turns", True)

        self.probe = LatencyProbe(
            endpoint_wait_s=float(self.get_parameter("vad_silence_s").value)
        )
        self.log_turns = bool(self.get_parameter("log_turns").value)
        out = str(self.get_parameter("out").value)
        self.writer = (
            SpanWriter(Path(out).expanduser(),
                       on_error=lambda e: self.get_logger().warning(f"probe file: {e}"))
            if out
            else None
        )
        self.get_logger().info(
            f"latency probe -> {out or 'log only'} (endpoint {self.probe.endpoint_wait_s:g}s)"
        )

        self.create_subscription(AudioChunk, "/wojtek/audio/speech", self.on_speech, 10)
        self.create_subscription(Transcript, "/wojtek/asr/final", self.on_final, 10)
        self.create_subscription(RoutedIntent, "/wojtek/intent", self.on_intent, 10)
        self.create_subscription(Sentence, "/wojtek/say_en", self.on_say_en, 10)
        self.create_subscription(Sentence, "/wojtek/say", self.on_say, 10)
        self.create_subscription(AudioChunk, "/wojtek/tts/audio", self.on_audio, 50)

    # -- one handler per pipeline stage boundary ----------------------------

    def on_speech(self, msg: AudioChunk):
        if msg.end_of_utterance:   # the live chunks are not a stage boundary
            self._observe("speech_end", msg.utterance_id, msg.header.stamp)

    def on_final(self, msg: Transcript):
        if msg.final:
            self._observe("asr_final", msg.utterance_id, msg.header.stamp,
                          text=msg.text)

    def on_intent(self, msg: RoutedIntent):
        self._observe("intent", msg.utterance_id, msg.header.stamp)

    def on_say_en(self, msg: Sentence):
        self._observe("say_en_first", msg.utterance_id, msg.header.stamp)

    def on_say(self, msg: Sentence):
        self._observe("say_first", msg.utterance_id, msg.header.stamp)
        if msg.final:
            self._observe("say_final", msg.utterance_id, msg.header.stamp)

    def on_audio(self, msg: AudioChunk):
        self._observe("audio_first", msg.utterance_id, msg.header.stamp)

    def _observe(self, event: str, utterance_id: str, stamp, text: str = "") -> None:
        for record in self.probe.observe(event, utterance_id, stamp_to_s(stamp), text):
            if self.writer is not None:
                self.writer.write(record)
            if self.log_turns and record["stage"] == "voice.reply":
                turn = self.probe.turns.get(record["turn"])
                said = f" {turn.text!r}" if turn and turn.text else ""
                self.get_logger().info(
                    f"turn {record['turn']}: heard -> first sound "
                    f"{record['ms'] / 1000:.2f}s{said}"
                )

    def destroy_node(self):
        if self.writer is not None:
            self.writer.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ProbeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
