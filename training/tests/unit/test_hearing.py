"""Unit tests for the spoken-instruction chain (wojtek_eval.hearing).

Nothing here touches real audio: `say`/`afconvert` are stood in for via
subprocess.run monkeypatching, and faster-whisper is stood in for via
sys.modules injection, so these run on any platform and without the
`eval` extra installed.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from wojtek_eval import hearing
from wojtek_eval.hearing import VOICES, Transcriber, hear_instruction, pick_voice, synthesize, wer

# -- wer -----------------------------------------------------------------


def test_wer_identical_is_zero():
    assert wer("go to the bed", "go to the bed") == 0.0


def test_wer_one_substitution_in_four_words():
    assert wer("go to the bed", "go to the couch") == 0.25


def test_wer_empty_ref_and_empty_hyp_is_zero():
    assert wer("", "") == 0.0


def test_wer_empty_ref_nonempty_hyp_is_one():
    assert wer("", "go left") == 1.0


def test_wer_empty_hyp_nonempty_ref_is_one():
    assert wer("go to the bed", "") == 1.0


def test_wer_ignores_case_and_punctuation():
    assert wer("Go to the bed!", "go to the bed") == 0.0


def test_wer_insertions_and_deletions():
    # ref has 3 words, hyp adds one extra word -> 1 insertion / 3
    assert wer("go to bed", "go to the bed") == pytest.approx(1 / 3)


# -- pick_voice ------------------------------------------------------------


def test_pick_voice_uses_rng_choice():
    class FakeRng:
        def __init__(self):
            self.seen = None

        def choice(self, seq):
            self.seen = seq
            return seq[2]

    rng = FakeRng()
    voice = pick_voice(rng)
    assert rng.seen == VOICES
    assert voice == VOICES[2]


# -- synthesize --------------------------------------------------------------


def _make_recording_run(fail_voiced_say: bool = False, fail_afconvert: bool = False):
    """Fake subprocess.run: records every command, writes the file the real
    tool would have written, and can simulate a `say -v` or afconvert failure.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "say":
            if fail_voiced_say and "-v" in cmd:
                raise subprocess.CalledProcessError(1, cmd, output="", stderr="voice not found")
            out_path = Path(cmd[cmd.index("-o") + 1])
            out_path.write_bytes(b"FORM....AIFF-fake")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "afconvert":
            if fail_afconvert:
                raise subprocess.CalledProcessError(1, cmd, output="", stderr="afconvert exploded")
            out_path = Path(cmd[-1])
            out_path.write_bytes(b"RIFF....WAVEfake")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    return calls, fake_run


def test_synthesize_invokes_say_then_afconvert_and_cleans_up_aiff(tmp_path, monkeypatch):
    calls, fake_run = _make_recording_run()
    monkeypatch.setattr(hearing.subprocess, "run", fake_run)

    wav_path = tmp_path / "out.wav"
    result = synthesize("hello world", wav_path, voice="Daniel", rate_wpm=210)

    assert result == wav_path
    assert wav_path.exists()
    assert not wav_path.with_suffix(".aiff").exists()

    assert len(calls) == 2
    say_cmd, afconvert_cmd = calls

    assert say_cmd[0] == "say"
    assert "-v" in say_cmd and say_cmd[say_cmd.index("-v") + 1] == "Daniel"
    assert "-r" in say_cmd and say_cmd[say_cmd.index("-r") + 1] == "210"
    assert say_cmd[-1] == "hello world"
    assert say_cmd[say_cmd.index("-o") + 1] == str(wav_path.with_suffix(".aiff"))

    assert afconvert_cmd[0] == "afconvert"
    assert "-f" in afconvert_cmd and afconvert_cmd[afconvert_cmd.index("-f") + 1] == "WAVE"
    assert "-d" in afconvert_cmd and afconvert_cmd[afconvert_cmd.index("-d") + 1] == "LEI16@16000"
    assert "-c" in afconvert_cmd and afconvert_cmd[afconvert_cmd.index("-c") + 1] == "1"
    assert afconvert_cmd[-1] == str(wav_path)


def test_synthesize_retries_without_voice_flag_on_say_failure(tmp_path, monkeypatch):
    calls, fake_run = _make_recording_run(fail_voiced_say=True)
    monkeypatch.setattr(hearing.subprocess, "run", fake_run)

    wav_path = tmp_path / "out.wav"
    result = synthesize("hi", wav_path, voice="NoSuchVoice")

    assert result == wav_path
    assert wav_path.exists()
    say_calls = [c for c in calls if c[0] == "say"]
    assert len(say_calls) == 2
    assert "-v" in say_calls[0]
    assert "-v" not in say_calls[1]


def test_synthesize_raises_runtime_error_when_say_fails_twice(tmp_path, monkeypatch):
    def always_fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="no voices at all")

    monkeypatch.setattr(hearing.subprocess, "run", always_fail)

    with pytest.raises(RuntimeError, match="no voices at all"):
        synthesize("hi", tmp_path / "out.wav")


def test_synthesize_raises_runtime_error_on_afconvert_failure(tmp_path, monkeypatch):
    calls, fake_run = _make_recording_run(fail_afconvert=True)
    monkeypatch.setattr(hearing.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="afconvert exploded"):
        synthesize("hi", tmp_path / "out.wav")


# -- Transcriber -------------------------------------------------------------


class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _FakeWhisperModel:
    instances = 0

    def __init__(self, model_size, device, compute_type):
        _FakeWhisperModel.instances += 1
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls: list[dict] = []

    def transcribe(self, path, beam_size=None, language=None, vad_filter=None):
        self.transcribe_calls.append(
            {"path": path, "beam_size": beam_size, "language": language, "vad_filter": vad_filter}
        )
        return [_FakeSegment(" go  "), _FakeSegment("to the bed ")], object()


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    """Inject a fake faster_whisper module so the lazy import in
    Transcriber._ensure_loaded succeeds without the real package installed."""
    _FakeWhisperModel.instances = 0
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = _FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    return fake_module


def test_transcriber_lazy_loads_and_joins_segments(fake_faster_whisper):
    t = Transcriber(model_size="tiny", device="cpu", compute_type="int8")
    assert _FakeWhisperModel.instances == 0  # not loaded yet

    text = t.transcribe(Path("/fake/path.wav"))

    assert text == "go to the bed"
    assert _FakeWhisperModel.instances == 1
    model = t._model
    assert model.model_size == "tiny"
    assert model.device == "cpu"
    assert model.compute_type == "int8"
    call = model.transcribe_calls[0]
    assert call["beam_size"] == 2
    assert call["language"] == "en"
    assert call["vad_filter"] is False


def test_transcriber_reuses_loaded_model(fake_faster_whisper):
    t = Transcriber()
    t.transcribe(Path("/fake/a.wav"))
    t.transcribe(Path("/fake/b.wav"))
    assert _FakeWhisperModel.instances == 1  # loaded once, stays resident


def test_transcriber_import_error_without_fake_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # simulate "not installed"
    t = Transcriber()
    with pytest.raises(ImportError):
        t.transcribe(Path("/fake/path.wav"))


# -- hear_instruction ---------------------------------------------------------


class _FakeRng:
    def choice(self, seq):
        return seq[0]


class _FakeTranscriber:
    def __init__(self, transcript):
        self.transcript = transcript
        self.seen_paths: list[Path] = []

    def transcribe(self, wav_path):
        self.seen_paths.append(wav_path)
        return self.transcript


def test_hear_instruction_wires_synthesize_and_transcribe(tmp_path, monkeypatch):
    synth_calls = []

    def fake_synthesize(text, wav_path, voice="Samantha", rate_wpm=190):
        synth_calls.append((text, wav_path, voice, rate_wpm))
        Path(wav_path).write_bytes(b"fake-wav")
        return wav_path

    monkeypatch.setattr(hearing, "synthesize", fake_synthesize)

    transcriber = _FakeTranscriber("go to the bed")
    result = hear_instruction("Go to the bed!", tmp_path, transcriber, _FakeRng())

    assert result["instruction"] == "Go to the bed!"
    assert result["voice"] == VOICES[0]
    assert result["transcript"] == "go to the bed"
    assert result["wer"] == 0.0
    assert Path(result["wav"]).exists()
    assert Path(result["wav"]).parent == tmp_path
    assert len(synth_calls) == 1
    assert transcriber.seen_paths == [Path(result["wav"])]


def test_hear_instruction_computes_nonzero_wer_on_mismatch(tmp_path, monkeypatch):
    def fake_synthesize(text, wav_path, voice="Samantha", rate_wpm=190):
        Path(wav_path).write_bytes(b"fake-wav")
        return wav_path

    monkeypatch.setattr(hearing, "synthesize", fake_synthesize)

    transcriber = _FakeTranscriber("go to the couch")
    result = hear_instruction("go to the bed", tmp_path, transcriber, _FakeRng())

    assert result["wer"] == pytest.approx(0.25)


def test_hear_instruction_slug_is_filesystem_safe(tmp_path, monkeypatch):
    def fake_synthesize(text, wav_path, voice="Samantha", rate_wpm=190):
        Path(wav_path).write_bytes(b"fake-wav")
        return wav_path

    monkeypatch.setattr(hearing, "synthesize", fake_synthesize)

    transcriber = _FakeTranscriber("go left")
    result = hear_instruction("Go left, then STOP!! (near the door)", tmp_path, transcriber, _FakeRng())

    wav_name = Path(result["wav"]).name
    assert wav_name.endswith(".wav")
    stem = wav_name[: -len(".wav")]
    assert all(c.isalnum() or c == "-" for c in stem)
