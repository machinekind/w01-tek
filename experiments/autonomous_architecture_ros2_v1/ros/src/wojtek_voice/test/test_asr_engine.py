"""Guard logic and input conversion, model-free."""

from dataclasses import dataclass

import numpy as np

from wojtek_voice.asr_engine import (
    MAX_NO_SPEECH_PROB,
    MIN_AVG_LOGPROB,
    filter_segments,
    pcm_to_whisper_input,
)


@dataclass
class Seg:
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = -0.3


class TestFilterSegments:
    def test_keeps_confident_speech(self):
        rec = filter_segments([Seg("cześć"), Seg("jestem Wojtek")])
        assert rec.text == "cześć jestem Wojtek"
        assert rec.dropped == []
        assert rec.confidence == -0.3

    def test_drops_probable_silence(self):
        # The phantom "6V" case: whisper decodes noise into a command.
        rec = filter_segments([Seg("6V", no_speech_prob=MAX_NO_SPEECH_PROB + 0.1)])
        assert rec.text == ""
        assert rec.dropped[0][0] == "6V"

    def test_drops_low_confidence(self):
        rec = filter_segments([Seg("blabla", avg_logprob=MIN_AVG_LOGPROB - 0.5)])
        assert rec.text == ""

    def test_mixed_keeps_only_good(self):
        rec = filter_segments(
            [Seg("idź do"), Seg("???", avg_logprob=-2.0), Seg("krzesła")]
        )
        assert rec.text == "idź do krzesła"
        assert len(rec.dropped) == 1

    def test_empty_input(self):
        rec = filter_segments([])
        assert rec.text == "" and rec.confidence == 0.0

    def test_missing_attrs_default_to_kept(self):
        @dataclass
        class Bare:
            text: str

        assert filter_segments([Bare("tak")]).text == "tak"


class TestPcmToWhisperInput:
    def test_resamples_and_normalises(self):
        pcm = (np.ones(24000) * 16384).astype(np.int16)  # 1 s at 24 kHz
        audio = pcm_to_whisper_input(pcm, 24000)
        assert audio.dtype == np.float32
        assert len(audio) == 16000
        assert abs(float(audio.mean()) - 0.5) < 0.01

    def test_16k_passthrough_length(self):
        pcm = np.zeros(16000, np.int16)
        assert len(pcm_to_whisper_input(pcm, 16000)) == 16000
