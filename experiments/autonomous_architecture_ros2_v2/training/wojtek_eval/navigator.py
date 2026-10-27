"""EvalNavigator: VlmNavigator + the eval-only `explore` action, and the
system prompt describing what Tier-A adds to the VLM's world (minimap HUD,
explore)."""

from __future__ import annotations

from wojtek_rl.vlm_nav import ACTIONS, SYSTEM_PROMPT, VlmDecision, VlmNavigator, decision_to_command

EVAL_ACTIONS = ACTIONS + ("explore",)

EVAL_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """
The bottom-right inset of each photo is a live top-down MINIMAP built from
your onboard depth camera (north is up, it only shows what you have already
seen): light gray = floor you have seen, red = obstacles, dark = unexplored,
cyan = frontier cells bordering unexplored space, green = your trail, yellow
arrow = you and your heading. Use it to stay oriented, avoid re-visiting
areas, and judge where unexplored space remains.

One extra action is available:
- `explore` (no amount) -- automatically turns and walks ~1 m toward the
  nearest unexplored frontier. Use it when the goal is nowhere in the camera
  view and you need to search other rooms, instead of wandering manually.

Task rules: DECIDING WHEN YOU HAVE ARRIVED IS THE MOST IMPORTANT PART OF THE
TASK. As soon as the target object occupies a large part of the camera view
(roughly within 1 m -- a chair, table or bed filling most of the frame),
issue `done` immediately: do not keep adjusting, do not walk past it. When
told to turn (e.g. "turn around"), execute the turn with turn_left/turn_right
(one 180, or two 90s) and then issue `done` right away -- do not walk
anywhere. For multi-part instructions ("go to X, then Y"), complete the parts
in order and issue `done` at the final one.

Anti-spin rules: never issue the same turn command more than TWICE in a row
-- your recent commands are listed each turn, check them. If your recent
turns already sum to ~360 degrees without the goal appearing, the goal is NOT
visible from this spot: use `explore` to move somewhere new. Walking forward
through open floor beats turning in place.
"""
)


class EvalNavigator(VlmNavigator):
    """Adds `explore` to the command mapping; everything else is inherited."""

    def _decision_to_command(self, decision: VlmDecision) -> str | None:
        if decision.action == "explore":
            return "explore"
        return decision_to_command(decision)
