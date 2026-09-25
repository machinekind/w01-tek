"""The voice benchmark's logic, without a GPU or a voice model.

The bench exists to answer one question on new hardware ("is this engine
faster than real time here?"), so the parts worth testing are the ones that
turn timings into that answer.
"""

import pytest

from wojtek_rl.agent import tts_bench


class FakeWav:
    def __init__(self, n):
        self.shape = (1, n)


class FakeModel:
    """Generates `rtf` wall seconds per second of audio, deterministically."""

    sr = 24000

    def __init__(self, rtf=0.9, audio_s=2.0):
        self.rtf = rtf
        self.audio_s = audio_s
        self.calls = []
        self.prepared = []

    def prepare_conditionals(self, ref):
        self.prepared.append(ref)

    def generate(self, text, language_id=None):
        self.calls.append((text, language_id))
        return FakeWav(int(self.sr * self.audio_s))


def test_measure_reports_rtf_over_the_fixed_lines(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(tts_bench, "time_line", lambda m, t, lang: (1.8, 2.0))
    r = tts_bench.measure(model, tts_bench.LINES, "pl", reps=2)
    assert r["n"] == len(tts_bench.LINES) * 2
    assert r["rtf"] == pytest.approx(0.9)


def test_the_verdict_names_the_consequence_not_the_number():
    assert "comfortable" in tts_bench.verdict(0.5)
    assert "little margin" in tts_bench.verdict(0.95)
    slow = tts_bench.verdict(1.35)
    assert "SLOWER THAN REAL TIME" in slow
    # B / (RTF - 1) with a 1 s prebuffer: 1 / 0.35 = 2.9 s of speech.
    assert "2.9 s" in slow


def test_the_line_set_is_short_medium_and_long():
    """Cost tracks generated audio, not characters, so the set has to span
    lengths -- and it must stay stable, because the documented numbers were
    taken on it."""
    lengths = sorted(len(line) for line in tts_bench.LINES)
    assert lengths[0] < 25 and lengths[-1] > 80
    assert "Widzę kanapę. Hau hau!" in tts_bench.LINES


def test_variants_are_rejected_when_unknown():
    with pytest.raises(ValueError):
        tts_bench.apply_variant(FakeModel(), "make-it-fast-please")


def test_base_variant_touches_nothing():
    model = FakeModel()
    tts_bench.apply_variant(model, "base")      # must not raise or patch
    assert model.calls == []


def test_a_reference_is_encoded_once_at_load(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(tts_bench, "load_model",
                        lambda device, ref="": (model.prepared.append(ref) or model)
                        if ref else model)
    loaded = tts_bench.load_model("cuda", "/refs/voice.wav")
    assert loaded.prepared == ["/refs/voice.wav"]


def test_clock_sweep_degrades_when_it_cannot_set_clocks(monkeypatch, capsys):
    """A sweep needs privileges the box may not grant; that must be a
    warning, not a crash in the middle of a benchmark."""
    monkeypatch.setattr(tts_bench, "gpu_name", lambda: "Fake GPU")
    monkeypatch.setattr(tts_bench, "load_model", lambda device, ref="": FakeModel())
    monkeypatch.setattr(tts_bench, "time_line", lambda m, t, lang: (1.8, 2.0))
    monkeypatch.setattr(tts_bench, "set_clock", lambda mhz: False)
    assert tts_bench.main(["--clock-sweep", "1000,2000", "--reps", "1"]) == 0
    out = capsys.readouterr().out
    assert "cannot lock clocks" in out
    assert "RTF 0.90" in out


def test_json_output_carries_the_gpu_and_the_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_bench, "gpu_name", lambda: "GB10")
    monkeypatch.setattr(tts_bench, "load_model", lambda device, ref="": FakeModel())
    monkeypatch.setattr(tts_bench, "time_line", lambda m, t, lang: (2.7, 2.0))
    out = tmp_path / "bench.json"
    assert tts_bench.main(["--reps", "1", "--json", str(out)]) == 0
    import json

    data = json.loads(out.read_text())
    assert data["gpu"] == "GB10"
    assert data["variants"]["base"]["rtf"] == pytest.approx(1.35)
    assert set(data["first_piece"]) == set(tts_bench.FIRST_PIECES)
