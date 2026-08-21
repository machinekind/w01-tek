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


# ---- backend selection + the transformers adapter ---------------------------


def test_backend_auto_picks_transformers_on_aarch64():
    """ctranslate2's aarch64 wheel is CPU-only (measured on a DGX Spark
    2026-08-21), so auto must route ARM boxes to the transformers engine."""
    from wojtek_voice.asr_engine import pick_backend

    assert pick_backend("auto", machine="aarch64") == "transformers"
    assert pick_backend("auto", machine="x86_64") == "faster-whisper"
    # An explicit request always wins over the platform rule.
    assert pick_backend("faster-whisper", machine="aarch64") == "faster-whisper"
    assert pick_backend("transformers", machine="x86_64") == "transformers"


def test_build_asr_engine_honours_the_backend():
    from wojtek_voice.asr_engine import (
        TransformersWhisperEngine,
        WhisperEngine,
        build_asr_engine,
    )

    eng = build_asr_engine("transformers", model_size="tiny", language="pl")
    assert isinstance(eng, TransformersWhisperEngine)
    assert eng.language == "pl"
    assert isinstance(build_asr_engine("faster-whisper"), WhisperEngine)


def test_transformers_segments_pass_the_same_guards():
    """Both backends must flow through filter_segments, so the phantom-'6V'
    incident stays fixed regardless of which engine decoded."""
    from wojtek_voice.asr_engine import _Segment, filter_segments

    rec = filter_segments([
        _Segment(text="idź do kuchni", no_speech_prob=0.1, avg_logprob=-0.3),
        _Segment(text="6V", no_speech_prob=0.9, avg_logprob=-0.2),      # silence
        _Segment(text="szszsz", no_speech_prob=0.2, avg_logprob=-1.7),  # guessing
    ])
    assert rec.text == "idź do kuchni"
    assert len(rec.dropped) == 2


def test_transformers_repo_resolution():
    from wojtek_voice.asr_engine import TransformersWhisperEngine

    assert TransformersWhisperEngine("large-v3")._repo() == "openai/whisper-large-v3"
    assert TransformersWhisperEngine("org/custom")._repo() == "org/custom"


def test_confident_repetition_loops_are_dropped():
    """A silence hallucination LOOPS confidently: measured 2026-08-21, the
    'że to jest w tym,' loop scored avg_logprob -0.25 -- better than real
    speech at -0.42 -- so only its redundancy gives it away. Whisper's own
    compression-ratio guard is the third gate."""
    from wojtek_voice.asr_engine import _Segment, compression_ratio, filter_segments

    loop = "że to jest w tym, " * 40
    assert compression_ratio(loop) > 2.4
    rec = filter_segments([_Segment(text=loop, no_speech_prob=0.1, avg_logprob=-0.25)])
    assert rec.text == "" and len(rec.dropped) == 1
    # Normal sentences compress poorly and must never trip it.
    ok = "Idę do lodówki po wodę dla ciebie."
    assert compression_ratio(ok) < 2.4
