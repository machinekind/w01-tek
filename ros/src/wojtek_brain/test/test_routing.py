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


def test_status_questions_route_to_the_agent_not_the_blind_persona():
    """'Gdzie teraz jesteś i co robisz?' answered by Bielik produced a
    confident story about a garden on camera -- pose/status questions belong
    to the VLM agent, which has the camera and the pose."""
    r = RuleRouter()
    for text in (
        "Gdzie teraz jesteś i co robisz?",
        "gdzie jesteś?",
        "Gdzie się znajdujesz?",
        "co teraz robisz?",
        "Dokąd idziesz?",
    ):
        intent, conf = r.classify(text)
        assert intent == "visual", (text, intent)


def test_trick_asks_route_to_the_trick_lane():
    from wojtek_brain.routing import trick_name
    r = RuleRouter()
    for text, expect in (
        ("Siad!", "sit"),
        ("Wojtek, usiądź proszę", "sit"),
        ("daj łapę", "paw_wave"),
        ("ukłoń się", "bow"),
        ("otrząśnij się", "shake"),
        ("zrób siku pod drzewem", "pee"),
        ("pokaż jakąś sztuczkę", ""),
    ):
        intent, conf = r.classify(text)
        assert intent == "trick", (text, intent)
        assert trick_name(text) == expect, text


def test_trick_wins_over_nav_in_compound_asks():
    r = RuleRouter()
    assert r.classify("podejdź do drzewa i zrób siku")[0] == "trick"


def test_plain_nav_is_not_a_trick():
    from wojtek_brain.routing import trick_name
    assert trick_name("idź do okna") is None
    r = RuleRouter()
    assert r.classify("idź do okna")[0] == "nav"


def test_trick_lane_tolerates_whisper_spellings():
    from wojtek_brain.routing import trick_name
    assert trick_name("Wojtek, zrób siat.") == "sit"
