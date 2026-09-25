"""VlmNavigator with its state changes written to a Trace.

The navigator already exposes everything worth recording through
`_set_state` (state, step, reason, error) and `_last` (the decision that
caused it) -- it just keeps it in memory for the UI. Subclassing rather than
editing wojtek_rl.vlm_nav keeps the eval/demo navigator untouched: nothing
else needs a trace, and `_set_state` is a stable internal seam already used
by the eval subclass pattern.
"""

from __future__ import annotations

from wojtek_rl.vlm_nav import SYSTEM_PROMPT, VlmNavigator

# No step budget interactively. A benchmark caps steps so episodes stay
# comparable (and to bound SPL); a demo user asking for a route wants it
# finished, and a counter cannot tell "still working" from "thrashing". The
# run ends on evidence instead: the model answers done/stop, the user
# cancels, the agent spins in place (max_rotation), or every command comes
# back blocked (MAX_CONSECUTIVE_BLOCKED). Pass --vlm-max-steps to restore a
# cap when you want benchmark-shaped episodes.
NAV_MAX_STEPS = None
# Anti-spin, the one guard a route still needs: a run of turns with no forward
# move means the model has lost the thread. Generous enough for a full
# look-around (12 x 30 deg = one revolution) plus a correction.
NAV_MAX_ROTATION = 14

INSTRUCTION_PROMPT = (
    SYSTEM_PROMPT
    + """
The goal may be a multi-step route rather than a single destination, e.g.
"walk past the table, turn left at the doorway and stop next to the bed", or
"walk around the chair". Then:

- Carry it out IN ORDER, one step at a time, and say in `reasoning` which
  step you are on ("step 2 of 3: turning left at the doorway").
- Going "around" or "past" something means keeping it beside you as you move,
  not stopping at it.
- Answer `done` ONLY when the LAST step is complete. Reaching the first
  landmark is not the end of the route.
"""
)


class TracedVlmNavigator(VlmNavigator):
    def __init__(self, *args, trace=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = trace

    def _set_state(self, state, step=None, reason=None, error=None):
        super()._set_state(state, step=step, reason=reason, error=error)
        if self.trace is None:
            return
        x, y, yaw = self.sim.pose()
        import math

        self.trace.add(
            "nav.state",
            state=state,
            goal=self._goal,
            step=self._step,
            reason=reason,
            error=error,
            last=self._last,
            # What the navigation model itself emitted, when the backend
            # exposes it: FutureNav's raw action text and per-stage timings.
            # Without this the trace only shows the translated mid-level
            # command and the VLA is a black box.
            model=getattr(self.client, "last_meta", None),
            blocked=getattr(self.sim.executor, "blocked", None),
            pose=[round(x, 3), round(y, 3), round(math.degrees(yaw), 1)],
        )
