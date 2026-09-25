"""The VLM brain: an exploration loop that looks until it sees the target.

Pure logic, no ROS and no HTTP: the prompts, the JSON schemas the model is
held to, the parser, and `Explorer`, the policy that turns the model's
answers and the robot's results into the next action. The node
(vlm_brain_node.py) does the talking; the tests feed this numbers.

The policy, decided with the owner on 2026-09-25, is *never a guessed
goal*:

  ask ──goal──► verify (a second yes/no question) ──yes──► pixel goal
   │                                    └──no──► search (turn, look again)
   ├──not_visible / turn──► search; after a full turn, explore 1 m forward
   └──done──► (ignored: arrival is the executive's call, below)

  pixel goal ──reached, target within done_within_m──► done
             ──reached, farther──► look again
             ──no_depth (target beyond the depth window)──► approach 1 m,
               then look again; approach blocked ──► search
             ──blocked──► search

The benchmark behind these choices (qwen3-vl 30B-A3B / 8B on nine sim
frames): pointing is fine, hallucinated goals on absent objects are the
failure to design against, and a thin target never "fills the view", so
`done` comes from the resolved target's distance, not from the model.
"""

import json
import math

# Where the node looks by default: a vLLM on the local machine (or the
# VLM_URL the launch reads from the environment) serving the 8B, the size
# the pointing benchmark found as accurate as the 30B-A3B and less prone
# to inventing objects (scripts/point_bench.py, 2026-09-25).
DEFAULT_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

SYSTEM = (
    "You are the navigation brain of a small quadruped robot. You see one photo "
    "from its forward camera, mounted 20 cm above the floor. Answer with exactly "
    "one JSON object and nothing else."
)

# The model's whole vocabulary. Held to it by the server's structured output.
SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["goal", "turn", "not_visible", "done"]},
        "point_2d": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                     "minItems": 2, "maxItems": 2},
        "label": {"type": "string"},
        "deg": {"type": "integer", "minimum": -180, "maximum": 180},
    },
    "required": ["type"],
    "additionalProperties": False,
}
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"visible": {"type": "boolean"}},
    "required": ["visible"],
    "additionalProperties": False,
}


def task_prompt(instruction):
    """The user's instruction verbatim (Polish is fine for Qwen3-VL), the
    answer format, and the rule that matters: not visible means not visible."""
    return (
        f"Zadanie od użytkownika: \"{instruction}\".\n"
        "If the target of the task is visible in this photo, reply "
        "{\"type\":\"goal\",\"point_2d\":[x,y],\"label\":\"<what>\"} with the point ON the target "
        "object (point_2d = [x, y], integers 0-1000 normalised to the image width and height).\n"
        "If the target is NOT in this photo, reply {\"type\":\"not_visible\"}. Never point at a "
        "different object instead of the target.\n"
        "If the robot already stands right in front of the target (it is close and fills much of "
        "the view), reply {\"type\":\"done\"}."
    )


def verify_prompt(instruction, label):
    """The second question before anything moves. It repeats the task so the
    model judges the object it pointed at against what was asked, not
    against its own label."""
    return (
        f"The task was: \"{instruction}\". Is the object \"{label}\" -- the target of that task, "
        "with every property the task names (kind, colour, size, where it is) -- actually visible "
        "in this photo? Answer {\"visible\": true} only if you can see it. A different object of "
        "another kind, colour or size does not count."
    )


def chat_url(base):
    """The chat-completions endpoint from however the server was named:
    `http://host:8000`, `http://host:8000/v1` and a trailing slash all
    land on `.../v1/chat/completions` (VLM_URL in .env is a base URL)."""
    base = base.strip().rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


def parse_answer(text):
    """The model's text to a dict; {"type": "unparsable"} when it is not JSON."""
    s = text.strip()
    try:
        return json.loads(s[s.find("{"):s.rfind("}") + 1])
    except (ValueError, TypeError):
        return {"type": "unparsable", "raw": s[:200]}


def normalised_point(answer):
    """(u, v) in [0, 1] from a goal answer, or None."""
    p = answer.get("point_2d")
    if not (isinstance(p, list) and len(p) == 2):
        return None
    try:
        u, v = float(p[0]) / 1000.0, float(p[1]) / 1000.0
    except (TypeError, ValueError):
        return None
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    return u, v


class Explorer:
    """The policy. Feed it events, get the next action.

    Actions: ("look",) ask the model with a fresh frame; ("verify", label)
    ask the yes/no question about the current answer; ("goal", (u, v))
    hand the pixel to the resolver; ("search", deg) turn in place;
    ("explore", metres) / ("approach", metres) walk straight; ("done",)
    and ("gave_up",) end the loop.
    """

    TERMINAL = ("done", "gave_up")

    def __init__(self, turn_deg=45.0, done_within_m=1.1, approach_m=1.0,
                 explore_m=1.0, max_steps=30):
        self.turn_deg = float(turn_deg)
        self.done_within_m = float(done_within_m)
        self.approach_m = float(approach_m)
        self.explore_m = float(explore_m)
        self.max_steps = int(max_steps)
        self.steps = 0
        self.turned_deg = 0.0
        self._point = None

    def _search(self):
        """Turn, and once a whole circle has shown nothing, step forward."""
        if abs(self.turned_deg) >= 360.0:
            self.turned_deg = 0.0
            return ("explore", self.explore_m)
        self.turned_deg += self.turn_deg
        return ("search", self.turn_deg)

    def on_answer(self, answer):
        """The model's answer to the task prompt."""
        self.steps += 1
        if self.steps > self.max_steps:
            return ("gave_up",)
        kind = answer.get("type")
        if kind == "goal":
            point = normalised_point(answer)
            if point is not None:
                self._point = point
                return ("verify", answer.get("label") or "the target")
        if kind == "turn" and isinstance(answer.get("deg"), (int, float)) and answer["deg"]:
            deg = float(answer["deg"])
            self.turned_deg += deg
            return ("search", deg)
        # not_visible, done (not the model's call), unparsable, a bad point
        return self._search()

    def on_verified(self, visible):
        """The yes/no answer about the point the model gave."""
        if visible and self._point is not None:
            self.turned_deg = 0.0
            return ("goal", self._point)
        self._point = None
        return self._search()

    def on_goal_result(self, result, distance_m=None):
        """pixel_goal_node's status word for the goal, and the robot's
        distance to the resolved object point (None if unknown)."""
        self._point = None
        if result == "reached":
            if distance_m is not None and distance_m < self.done_within_m:
                return ("done",)
            return ("look",)
        if result == "no_depth":
            return ("approach", self.approach_m)
        # blocked, timeout, no_frame, no_tf: a new heading, not the same push
        return self._search()

    def on_move_result(self, result):
        """goto's status after an approach/explore step."""
        if result == "reached":
            return ("look",)
        return self._search()

    def turn_done(self):
        return ("look",)
