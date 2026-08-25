"""The agent's goal state machine: one current goal, one active behaviour.

Wraps the two long-running behaviours behind a single get/set surface:
  navigate  VlmNavigator (FutureNav or any VlmClientProto backend) drives
            toward a language goal that names a place/object to reach
  search    SearchController explores frontiers until the object is found

Exactly one behaviour runs at a time; setting a new goal cancels the old
one. States are derived, not stored: idle / navigating / searching plus the
terminal report of the last run (done / found / not_found / error), so the
chat tools can always answer "what are you doing?" from status() without
polling loops.
"""

from __future__ import annotations

from loguru import logger

KINDS = ("navigate", "search")

# Spoken outcomes. A goal that ends in silence is indistinguishable from a
# goal still running, so the dog says how it went the moment it stops --
# success loudly, failure honestly. Templates rather than another model call:
# this fires at the end of a behaviour, where waiting a second for a
# generated sentence would land the confirmation after the moment has passed.
OUTCOMES = {
    "pl": {
        # No noun interpolation in Polish success/failure lines: the goal
        # string arrives in whatever case the user's sentence put it in
        # ("idź do telewizorA" -> target "telewizora"), and gluing a genitive
        # into a nominative slot produced "Znalazłem: telewizora" on camera.
        # Templates that need no inflection are always grammatical; the goal
        # itself is visible on screen and in the trace.
        ("search", "found"): "Hau hau! Znalazłem to, czego szukałem!",
        ("search", "not_found"): "Nie znalazłem tego, przeszukałem wszystko.",
        ("search", "error"): "Coś poszło nie tak podczas szukania.",
        ("navigate", "done"): "Jestem na miejscu! Hau!",
        ("navigate", "stuck"): "Utknąłem, nie mogę tam dojść.",
        ("navigate", "error"): "Nie dałem rady tam dojść.",
    },
    "en": {
        ("search", "found"): "Found it: {goal}! Woof woof!",
        ("search", "not_found"): "I could not find {goal}. I searched everywhere.",
        ("search", "error"): "Something went wrong while searching.",
        ("navigate", "done"): "I am there! Woof!",
        ("navigate", "stuck"): "I am stuck, I cannot get there.",
        ("navigate", "error"): "I could not get there.",
    },
}
# Said when a new goal preempts a running one. Switching in silence looks
# like the robot ignored you -- especially mid-search, where it just keeps
# walking and the new instruction seems lost.
SWITCH = {
    "pl": {
        ("search", "search"): "Ok, przestaję szukać: {old}. Zaczynam szukać: {new}!",
        ("search", "navigate"): "Ok, przestaję szukać: {old}. Idę: {new}!",
        ("navigate", "search"): "Ok, przerywam trasę. Zaczynam szukać: {new}!",
        ("navigate", "navigate"): "Ok, zmieniam trasę. Idę: {new}!",
    },
    "en": {
        ("search", "search"): "Okay, dropping the search for {old}. Looking for {new}!",
        ("search", "navigate"): "Okay, dropping the search for {old}. Heading: {new}!",
        ("navigate", "search"): "Okay, abandoning that route. Looking for {new}!",
        ("navigate", "navigate"): "Okay, new route. Heading: {new}!",
    },
}


def switch_phrase(old_kind: str, old_goal: str, new_kind: str, new_goal: str,
                  language: str = "pl") -> str | None:
    """What to say when a goal is replaced while still running."""
    if not old_goal or not old_kind:
        return None
    table = SWITCH.get(language) or SWITCH["en"]
    template = table.get((old_kind, new_kind))
    return template.format(old=old_goal, new=new_goal) if template else None


# Navigator reasons that mean "I gave up", not "I arrived". `vlm_done` is the
# model declaring arrival; everything else is a guard firing.
NAV_FAILURE_REASONS = {"max_rotation", "max_steps", "stuck", "vlm_stop"}
TERMINAL_STATES = {"found", "not_found", "done", "error", "stuck"}


def outcome_phrase(kind: str, state: str, goal: str, reason: str | None = None,
                   language: str = "pl") -> str | None:
    """What to say out loud when a goal stops. None = say nothing."""
    if state not in TERMINAL_STATES:
        return None
    if kind == "navigate" and state == "done" and reason in NAV_FAILURE_REASONS:
        state = "stuck"
    table = OUTCOMES.get(language) or OUTCOMES["en"]
    template = table.get((kind, state))
    return template.format(goal=goal or "") if template else None


class GoalManager:
    def __init__(self, navigator_factory, search_factory, trace=None, language: str = "pl"):
        """Factories, not instances: both stacks import lazily/heavy (the
        navigator may pull in a VLM backend), so nothing is built until the
        first goal arrives."""
        self._navigator_factory = navigator_factory
        self._search_factory = search_factory
        self.trace = trace
        self.language = language
        self._navigator = None
        self._search = None
        self._goal: str | None = None
        self._kind: str | None = None
        # (kind, goal) of the behaviour the last set_goal interrupted, if any.
        self.switched_from: tuple[str, str] | None = None

    # -- lazy behaviours ------------------------------------------------------

    def navigator(self):
        if self._navigator is None:
            self._navigator = self._navigator_factory()
        return self._navigator

    def search(self):
        if self._search is None:
            self._search = self._search_factory()
        return self._search

    @property
    def rev(self) -> int:
        """UI diffing counter: sum of the live behaviours' counters."""
        rev = 0
        if self._navigator is not None:
            rev += self._navigator.rev
        if self._search is not None:
            rev += self._search.rev
        return rev

    # -- get/set ----------------------------------------------------------------

    def set_goal(self, text: str, kind: str = "navigate",
                 nav_text: str | None = None) -> dict:
        """`text` is the user's wording VERBATIM (displayed, spoken about,
        used for search); `nav_text` optionally carries a dead-simple English
        imperative for the walking backend -- FutureNav is trained on English
        VLN-CE and the prompt navigators steer better in English too (both
        v3 castle/flat takes walked away from Polish-worded goals).
        """
        text = text.strip()
        if not text:
            return {"ok": False, "error": "empty goal"}
        if kind not in KINDS:
            return {"ok": False, "error": f"unknown goal kind {kind!r} (have {KINDS})"}
        # Remember what we are interrupting, so the caller can say so out loud.
        was = self.status()
        self.switched_from = (
            (was["kind"], was["goal"])
            if was["state"] in ("searching", "navigating")
            else None
        )
        self.cancel("new goal")
        try:
            ack = (
                self.navigator().start((nav_text or "").strip() or text)
                if kind == "navigate"
                else self.search().start(text)
            )
        except Exception as e:  # missing extra / backend misconfig
            logger.warning(f"goal start failed: {e}")
            return {"ok": False, "error": str(e)}
        if ack.get("ok"):
            self._goal, self._kind = text, kind
            # Announce the switch ourselves rather than asking the model to
            # mention it: a 4B routinely ignores that instruction, and the
            # user then hears "heading to the table" with no hint that the
            # search it was running has been abandoned.
            if self.switched_from and self.trace is not None:
                phrase = switch_phrase(*self.switched_from, kind, text, language=self.language)
                if phrase:
                    self.trace.add("goal.switch", text=phrase,
                                   old=list(self.switched_from), new=[kind, text])
        if self.trace is not None:
            self.trace.add("goal.set", goal=text, goal_kind=kind, ok=bool(ack.get("ok")),
                           error=ack.get("error"))
        return {**ack, "kind": kind}

    def cancel(self, reason: str = "user") -> None:
        cancelled = []
        if self._navigator is not None and self._navigator.running:
            self._navigator.cancel(reason)
            cancelled.append("navigate")
        if self._search is not None and self._search.running:
            self._search.cancel(reason)
            cancelled.append("search")
        if cancelled and self.trace is not None:
            self.trace.add("goal.cancel", reason=reason, cancelled=cancelled, goal=self._goal)

    def status(self) -> dict:
        """Current goal + derived state + the active behaviour's detail."""
        state, detail = "idle", None
        if self._search is not None and self._search.running:
            state, detail = "searching", self._search.status()
        elif self._navigator is not None and self._navigator.running:
            state, detail = "navigating", self._navigator.status()
        elif self._kind == "search" and self._search is not None:
            detail = self._search.status()
            state = detail["state"]  # found / not_found / error / idle
        elif self._kind == "navigate" and self._navigator is not None:
            detail = self._navigator.status()
            state = detail["state"]  # done / error / idle
        return {"goal": self._goal, "kind": self._kind, "state": state, "detail": detail}

    def describe(self) -> str:
        """One-line answer material for the chat loop."""
        s = self.status()
        if s["goal"] is None:
            return "no goal set; idle"
        line = f"goal: {s['kind']} -> {s['goal']!r}; state: {s['state']}"
        detail = s["detail"] or {}
        note = detail.get("note") or detail.get("reason")
        if note:
            line += f" ({note})"
        return line
