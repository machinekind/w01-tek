"""GoalManager: one goal at a time, lazy behaviours, derived states."""

from wojtek_agent.goals import GoalManager


class FakeBehaviour:
    """Stands in for VlmNavigator and SearchController alike."""

    def __init__(self, name):
        self.name = name
        self.rev = 0
        self._running = False
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self._terminal = "idle"

    @property
    def running(self):
        return self._running

    def start(self, text):
        self._running = True
        self.started.append(text)
        self.rev += 1
        return {"ok": True, "goal": text}

    def cancel(self, reason="user"):
        self._running = False
        self.cancelled.append(reason)
        self.rev += 1

    def finish(self, state):
        self._running = False
        self._terminal = state

    def status(self):
        state = "running" if self._running else self._terminal
        return {"state": state, "note": None, "reason": None}


def make_manager():
    nav, search = FakeBehaviour("nav"), FakeBehaviour("search")
    gm = GoalManager(navigator_factory=lambda: nav, search_factory=lambda: search)
    return gm, nav, search


def test_lazy_factories():
    built = []
    gm = GoalManager(
        navigator_factory=lambda: built.append("nav") or FakeBehaviour("nav"),
        search_factory=lambda: built.append("search") or FakeBehaviour("search"),
    )
    assert built == []
    gm.set_goal("the sofa", kind="navigate")
    assert built == ["nav"]  # search never constructed


def test_navigate_goal():
    gm, nav, _ = make_manager()
    ack = gm.set_goal("the sofa", kind="navigate")
    assert ack["ok"] and ack["kind"] == "navigate"
    assert nav.started == ["the sofa"]
    st = gm.status()
    assert st["state"] == "navigating"
    assert st["goal"] == "the sofa"


def test_search_goal_cancels_navigation():
    gm, nav, search = make_manager()
    gm.set_goal("the sofa", kind="navigate")
    gm.set_goal("red ball", kind="search")
    assert nav.cancelled == ["new goal"]
    assert search.started == ["red ball"]
    assert gm.status()["state"] == "searching"


def test_terminal_state_reported_after_finish():
    gm, _, search = make_manager()
    gm.set_goal("red ball", kind="search")
    search.finish("found")
    st = gm.status()
    assert st["state"] == "found"
    assert st["goal"] == "red ball"


def test_rejects_bad_input():
    gm, _, _ = make_manager()
    assert gm.set_goal("", kind="navigate")["ok"] is False
    assert gm.set_goal("x", kind="teleport")["ok"] is False


def test_failed_start_keeps_no_goal():
    class Broken(FakeBehaviour):
        def start(self, text):
            raise RuntimeError("no backend")

    gm = GoalManager(navigator_factory=lambda: Broken("nav"),
                     search_factory=lambda: FakeBehaviour("search"))
    ack = gm.set_goal("the sofa", kind="navigate")
    assert ack["ok"] is False
    assert gm.status()["goal"] is None


def test_trace_records_goal_set_and_cancel():
    from wojtek_agent.trace import Trace

    trace = Trace(None)
    nav, search = FakeBehaviour("nav"), FakeBehaviour("search")
    gm = GoalManager(lambda: nav, lambda: search, trace=trace)
    gm.set_goal("red ball", kind="search")
    gm.cancel("user")
    events = trace.recent()
    assert [e["kind"] for e in events] == ["goal.set", "goal.cancel"]
    assert events[0]["goal"] == "red ball" and events[0]["goal_kind"] == "search"
    assert events[0]["ok"] is True
    assert events[1]["cancelled"] == ["search"] and events[1]["reason"] == "user"


def test_trace_records_rejected_goal():
    from wojtek_agent.trace import Trace

    trace = Trace(None)
    gm = GoalManager(lambda: FakeBehaviour("nav"), lambda: FakeBehaviour("s"), trace=trace)
    gm.set_goal("", kind="navigate")
    assert trace.recent() == []  # rejected before the machine was touched


def test_describe_lines():
    gm, _, _ = make_manager()
    assert "no goal" in gm.describe()
    gm.set_goal("the sofa", kind="navigate")
    d = gm.describe()
    assert "navigate" in d and "the sofa" in d and "navigating" in d


# -- spoken outcomes -----------------------------------------------------------


def test_success_phrases_name_the_goal():
    from wojtek_agent.goals import outcome_phrase

    assert outcome_phrase("search", "found", "piłka", language="pl") == "Znalazłem: piłka! Hau hau!"
    assert "there" in outcome_phrase("navigate", "done", "the bed", language="en")


def test_navigation_giving_up_is_not_announced_as_arrival():
    """`done` covers both 'I arrived' and 'a guard stopped me' -- only the
    model saying done counts as success."""
    from wojtek_agent.goals import outcome_phrase

    arrived = outcome_phrase("navigate", "done", "łóżko", reason="vlm_done", language="pl")
    gave_up = outcome_phrase("navigate", "done", "łóżko", reason="max_rotation", language="pl")
    stuck = outcome_phrase("navigate", "done", "łóżko", reason="stuck", language="pl")
    assert arrived == "Jestem na miejscu! Hau!"
    assert gave_up == "Utknąłem, nie mogę tam dojść."
    assert stuck == gave_up


def test_failed_search_is_announced_too():
    from wojtek_agent.goals import outcome_phrase

    said = outcome_phrase("search", "not_found", "kanapa", language="pl")
    assert "Nie udało mi się znaleźć" in said and "kanapa" in said


def test_running_states_say_nothing():
    from wojtek_agent.goals import outcome_phrase

    for state in ("searching", "navigating", "idle", "scanning"):
        assert outcome_phrase("search", state, "x", language="pl") is None


def test_unknown_language_falls_back_to_english():
    from wojtek_agent.goals import outcome_phrase

    assert outcome_phrase("search", "found", "ball", language="xx").startswith("Found it")


def test_switch_phrases_name_both_goals():
    from wojtek_agent.goals import switch_phrase

    s = switch_phrase("search", "kanapa", "navigate", "podejdź do stołu", language="pl")
    assert "przestaję szukać: kanapa" in s and "podejdź do stołu" in s
    s2 = switch_phrase("search", "kanapa", "search", "piłka", language="pl")
    assert "przestaję szukać: kanapa" in s2 and "Zaczynam szukać: piłka" in s2


def test_no_switch_phrase_when_nothing_was_running():
    from wojtek_agent.goals import switch_phrase

    assert switch_phrase("", "", "search", "piłka") is None


def test_set_goal_records_what_it_interrupted():
    gm, nav, search = make_manager()
    gm.set_goal("kanapa", kind="search")
    assert gm.switched_from is None          # nothing was running
    gm.set_goal("stół", kind="navigate")
    assert gm.switched_from == ("search", "kanapa")


def test_switched_from_is_empty_after_a_finished_goal():
    gm, _, search = make_manager()
    gm.set_goal("kanapa", kind="search")
    search.finish("found")
    gm.set_goal("stół", kind="navigate")
    assert gm.switched_from is None          # it ended on its own, not interrupted
