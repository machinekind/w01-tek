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

  pixel goal ──reached/blocked, target within done_within_m──► done
             ──reached, farther──► look again
             ──no_depth (target beyond the depth window)──► approach 1 m
               ALONG THE PIXEL'S BEARING, then look again
             ──blocked──► look again (fresh depth: odometry drifts over a
               long walk, and the target may be right there); blocked
               twice in a row ──► search

  approach ──reached──► look again
           ──blocked──► look again; the next approach detours 40 deg off
             the bearing, alternating sides
  any move that covers < min_progress_m counts as a stall; max_stalls in a
  row, or more than max_blind approaches in a row that never bring the
  target into the depth window, end the task as `gave_up` with the reason.

The benchmark behind these choices (scripts/point_bench.py on nine sim
frames): pointing is fine, hallucinated goals on absent objects are the
failure to design against, and a thin target never "fills the view", so
`done` comes from the resolved target's distance, not from the model.

The approach bearing (machinekind/w01-tek#42): a target beyond the depth
window (3 m in the sim, the D435's trusted range on the robot) has no
depth, only a direction. Walking straight ahead instead of along that
direction closes nothing once a search turn has left the body pointing
elsewhere: the robot walks off the line, is blocked, turns 45 deg, walks
off again, and the pillar never comes into range.
"""

import json
import math

# Where the node looks by default: Ollama on the local machine (or the
# VLM_URL the launch reads from the environment) serving Qwen3-VL 30B-A3B.
DEFAULT_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen3-vl:30b-a3b-instruct"

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
    `http://host:11434`, `http://host:11434/v1` and a trailing slash all
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


def pixel_ray(u, v, k):
    """The viewing ray of pixel (u, v) (absolute, not normalised) in the
    camera's optical frame (x right, y down, z forward), at z = 1. k is the
    9-element row-major CameraInfo.k."""
    return (u - k[2]) / k[0], (v - k[5]) / k[4], 1.0


def rotate(q, v):
    """Vector v rotated by the unit quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    # v + 2w (q_v x v) + 2 q_v x (q_v x v)
    cx, cy, cz = y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]
    return (v[0] + 2 * (w * cx + y * cz - z * cy),
            v[1] + 2 * (w * cy + z * cx - x * cz),
            v[2] + 2 * (w * cz + x * cy - y * cx))


def ray_heading(q, ray):
    """The planar heading (yaw, rad) of a camera ray in a frame whose
    rotation from the camera is q -- odom for the approach. None when the
    ray points straight up or down."""
    x, y, _ = rotate(q, ray)
    if math.hypot(x, y) < 1e-9:
        return None
    return math.atan2(y, x)


def step_along(position, heading, metres):
    """The point `metres` from (x, y) `position` along `heading`."""
    return position[0] + metres * math.cos(heading), position[1] + metres * math.sin(heading)


class Explorer:
    """The policy. Feed it events, get the next action.

    Actions: ("look",) ask the model with a fresh frame; ("verify", label)
    ask the yes/no question about the current answer; ("goal", (u, v))
    hand the pixel to the resolver; ("search", deg) turn in place;
    ("explore", metres) walk straight ahead; ("approach", metres, (u, v),
    detour_deg) walk along the bearing of pixel (u, v), turned detour_deg
    off it (positive left); ("done",) and ("gave_up", reason) end the loop.
    """

    TERMINAL = ("done", "gave_up")

    def __init__(self, turn_deg=45.0, done_within_m=1.1, approach_m=1.0,
                 explore_m=1.0, max_steps=30, max_blind=8, min_progress_m=0.25,
                 max_stalls=3, detour_deg=40.0):
        self.turn_deg = float(turn_deg)
        self.done_within_m = float(done_within_m)
        self.approach_m = float(approach_m)
        self.explore_m = float(explore_m)
        self.max_steps = int(max_steps)
        self.max_blind = int(max_blind)
        self.min_progress_m = float(min_progress_m)
        self.max_stalls = int(max_stalls)
        self.detour_deg = float(detour_deg)
        self.steps = 0
        self.turned_deg = 0.0
        self.blind = 0        # approaches in a row with the target beyond depth
        self.stalls = 0       # moves in a row that covered < min_progress_m
        self.detour = 0.0     # deg off the bearing for the next approach
        self.goal_blocks = 0  # resolved goals in a row that ended blocked
        self._point = None
        self._moving = None   # "approach" / "explore": the move in flight

    def _search(self):
        """Turn, and once a whole circle has shown nothing, step forward."""
        if abs(self.turned_deg) >= 360.0:
            self.turned_deg = 0.0
            self._moving = "explore"
            return ("explore", self.explore_m)
        self.turned_deg += self.turn_deg
        return ("search", self.turn_deg)

    def on_answer(self, answer):
        """The model's answer to the task prompt."""
        self.steps += 1
        if self.steps > self.max_steps:
            return ("gave_up", f"step budget spent: {self.max_steps} looks without reaching the target")
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
        point, self._point = self._point, None
        if result in ("reached", "blocked"):
            # The depth resolved: the target is inside the window again.
            self.blind = 0
            if distance_m is not None and distance_m < self.done_within_m:
                return ("done",)
        if result == "reached":
            self.goal_blocks = 0
            return ("look",)
        if result == "blocked":
            # The setpoint was fixed in odom from a picture metres back; the
            # leg odometry under-reads a long walk (~30 % in the sim), so
            # the robot may stand at the target with odom saying otherwise.
            # A fresh picture resolves it again from here. Twice blocked is
            # a real obstacle: turn.
            self.goal_blocks += 1
            if self.goal_blocks == 1:
                return ("look",)
            self.goal_blocks = 0
            return self._search()
        if result == "no_depth" and point is not None:
            self.blind += 1
            if self.blind > self.max_blind:
                return ("gave_up", f"target never came within depth range: {self.max_blind} "
                                   f"approaches of {self.approach_m:g} m along its bearing")
            self._moving = "approach"
            return ("approach", self.approach_m, point, self.detour)
        # blocked, timeout, no_frame, no_tf: a new heading, not the same push
        return self._search()

    def on_move_result(self, result, moved_m=None):
        """goto's status after an approach/explore step, and how far the
        robot actually got (None if unknown)."""
        moving, self._moving = self._moving, None
        if moved_m is not None and moved_m < self.min_progress_m:
            self.stalls += 1
            if self.stalls >= self.max_stalls:
                return ("gave_up", f"no progress: {self.stalls} moves in a row covered "
                                   f"less than {self.min_progress_m:g} m each")
        else:
            self.stalls = 0
        if result == "reached":
            self.detour = 0.0
            return ("look",)
        if moving == "approach":
            # Something stands on the bearing. Looking again gives the
            # same bearing, so the next approach goes round it: left
            # first, then right, alternating while it stays blocked.
            self.detour = -self.detour if self.detour else self.detour_deg
            return ("look",)
        return self._search()

    def turn_done(self):
        return ("look",)
