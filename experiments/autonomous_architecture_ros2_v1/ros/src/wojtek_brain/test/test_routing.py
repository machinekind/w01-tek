"""Rule router behavior on realistic Polish utterances (ASR-style, lowercase)."""

import pytest

from wojtek_brain.routing import RuleRouter, apply_confidence_floor


@pytest.fixture
def router():
    return RuleRouter()


@pytest.mark.parametrize("text", [
    "cześć wojtek jak się masz",
    "kim jesteś",
    "opowiedz mi coś o sobie",
    "ile masz lat",
    "jaka jest stolica francji",
])
def test_chat(router, text):
    assert router.classify(text)[0] == "chat"


@pytest.mark.parametrize("text", [
    "idź do kuchni",
    "znajdź czerwoną piłkę",
    "podejdź do krzesła",
    "obejdź stół i stań przy drzwiach",
    "poszukaj misia",
    "obróć się w lewo",
    "wróć do mnie",
])
def test_nav(router, text):
    assert router.classify(text)[0] == "nav"


@pytest.mark.parametrize("text", [
    "co widzisz",
    "co teraz widzisz przed sobą",
    "opisz otoczenie",
    "rozejrzyj się",
    "widzisz gdzieś piłkę",
])
def test_visual(router, text):
    assert router.classify(text)[0] == "visual"


@pytest.mark.parametrize("text", [
    "stop",
    "stój",
    "przestań",
    "zatrzymaj się natychmiast",
    "dosyć tego",
])
def test_cancel(router, text):
    assert router.classify(text)[0] == "cancel"


def test_cancel_wins_over_nav(router):
    # "stop, nie idź tam" carries a nav verb but is a cancel.
    assert router.classify("stop nie idź tam")[0] == "cancel"


def test_system(router):
    assert router.classify("zresetuj się")[0] == "system"
    assert router.classify("mów głośniej")[0] == "system"


def test_empty_is_low_confidence_chat(router):
    intent, conf = router.classify("   ")
    assert intent == "chat" and conf == 0.0


class TestConfidenceFloor:
    def test_low_confidence_nav_downgrades_to_chat(self):
        assert apply_confidence_floor("nav", 0.4, 0.55) == ("chat", 0.4)

    def test_confident_nav_passes(self):
        assert apply_confidence_floor("nav", 0.85, 0.55) == ("nav", 0.85)

    def test_chat_never_downgrades(self):
        assert apply_confidence_floor("chat", 0.1, 0.55) == ("chat", 0.1)
