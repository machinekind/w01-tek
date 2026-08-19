"""The agent's tool registry: what the chat loop can actually do.

Seven tools, deliberately few -- a 4B instruct model picks reliably from a
short flat list, not a deep API. Each tool returns a ToolResult whose text
(and optional images) is fed straight back into the model as the next
context block. Long-running behaviours (navigate/search) only START here and
return immediately; GoalManager owns their lifetime.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass, field

from wojtek_agent.spatial import PoseHistory, frontier_clusters, map_summary


@dataclass(frozen=True)
class ToolResult:
    text: str
    images: tuple[str, ...] = ()


@dataclass(frozen=True)
class Tool:
    name: str
    signature: str  # shown to the model, e.g. 'search {"object": "<name>"}'
    description: str
    fn: object = field(repr=False)  # async callable(args: dict) -> ToolResult


def _map_jpeg_b64(sim) -> str:
    """The agent-built occupancy map as a JPEG (robot, trail, frontiers)."""
    from PIL import Image

    img = Image.fromarray(sim.omap.map_image(sim.pose(), px=320))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_NAV_PREFIX_RE = re.compile(
    r"^(zmiana planów[!.,]?\s*)?(idź|pójdź|podejdź|chodź|jedź|biegnij|ruszaj|"
    r"zbliż się|go|walk|head)\s*(teraz\s+)?(do|w stronę|w kierunku|"
    r"przy|to|towards?)?\s*",
    re.IGNORECASE,
)


_QUESTION_RE = re.compile(
    r"^(co|kto|gdzie|jak|czy|kiedy|dlaczego|ile|what|where|how|why|who|which)\b",
    re.IGNORECASE,
)


def looks_like_question(text: str) -> bool:
    """A question routed into navigate/search becomes a goal named after
    the question (observed live: search for 'Co teraz robisz?...').  Bounce
    it back to the model instead."""
    t = text.strip()
    return t.endswith("?") or bool(_QUESTION_RE.match(t))


def nav_target_phrase(instruction: str) -> str:
    """'Idź do lodówki.' -> 'lodówki': the object phrase a search can use."""
    target = _NAV_PREFIX_RE.sub("", instruction).strip(" .!?")
    return target or instruction


def keep_full_instruction(instruction: str, user_text: str) -> bool:
    """True when the model collapsed a route into a bare noun phrase.

    The navigator is an instruction follower (VLN-CE style): "walk around the
    table, then stop by the door" is a two-step route, and handing it "the
    table" throws away both the manner and the second step. The chat model's
    job here is routing, not summarising, so when its argument is much shorter
    than what the user actually said, the user's words win.
    """
    if not user_text:
        return False
    user_words = len(user_text.split())
    return user_words >= 4 and len(instruction.split()) < max(3, 0.6 * user_words)


def _switch_note(goals, new_kind: str, new_goal: str) -> str:
    """Tell the model it just interrupted something, so its reply says so.

    Swapping goals in silence reads as "the robot ignored me" -- most visibly
    mid-search, where it keeps walking on the old target while the new
    instruction seems to vanish.
    """
    prev = getattr(goals, "switched_from", None)
    if not prev:
        return ""
    old_kind, old_goal = prev
    return (
        f"NOTE: this REPLACED a running {old_kind} goal ({old_goal!r}). Tell the "
        f"user you are dropping it and starting {new_goal!r} instead. "
    )


def build_tools(
    sim, goals, pose_history: PoseHistory, turn_context: dict | None = None,
    visibility_check=None,
) -> dict[str, Tool]:
    """turn_context carries this turn's raw user text (set by WojtekAgent), so
    navigate/search can fall back to the instruction as spoken.

    `visibility_check` is an optional async callable(target_text) -> bool|None
    scoring the CURRENT camera view.  When it returns False, navigate
    redirects to search: walking toward an object that is not in view is a
    guess, searching is the honest behaviour (user call, 2026-08-14).  True
    navigates; None (unavailable/error) keeps the old behaviour."""
    turn_context = turn_context if turn_context is not None else {}
    def _camera_frame() -> str:
        """Bare camera view. Deliberately hud-free: describing an inset the
        prompt never mentions makes a small VLM narrate the minimap as if it
        were furniture (it once "found" a toy inside its own map). Map
        questions have their own tool."""
        try:
            return sim.ego_jpeg(hud=False)
        except TypeError:  # sims whose ego_jpeg predates the flag
            return sim.ego_jpeg()

    async def look(args: dict) -> ToolResult:
        return ToolResult(
            text="Current head-camera view attached.",
            images=(_camera_frame(),),
        )

    async def get_map(args: dict) -> ToolResult:
        clusters = frontier_clusters(sim.omap)
        text = (
            "Occupancy map attached (top-down; light=seen floor, red=obstacle, "
            "cyan=unexplored frontier, green=your trail, yellow=you).\n"
            + map_summary(sim.omap, sim.pose(), clusters)
        )
        return ToolResult(text=text, images=(_map_jpeg_b64(sim),))

    async def route(args: dict) -> ToolResult:
        try:
            seconds = float(args.get("seconds", 3.0))
        except (TypeError, ValueError):
            seconds = 3.0
        seconds = min(max(seconds, 1.0), 120.0)
        return ToolResult(text=pose_history.describe(seconds))

    async def status(args: dict) -> ToolResult:
        x, y, yaw = sim.pose()
        import math

        lines = [
            f"pose: x={x:.2f} m, y={y:.2f} m, heading {math.degrees(yaw):.0f} deg",
            goals.describe(),
            f"executor: {sim.executor.status()}",
        ]
        return ToolResult(text="\n".join(lines))

    async def navigate(args: dict) -> ToolResult:
        instruction = str(args.get("instruction") or args.get("goal") or "").strip()
        spoken = str(turn_context.get("user_text", "") or "").strip()
        if not instruction or keep_full_instruction(instruction, spoken):
            instruction = spoken or instruction
        if looks_like_question(instruction):
            return ToolResult(
                text="that looks like a QUESTION, not a movement instruction -- "
                "answer it with 'say' (use `status` for questions about the "
                "current goal or progress)"
            )
        if visibility_check is not None:
            try:
                visible = await visibility_check(instruction)
            except Exception:
                visible = None  # a broken check must not block navigation
            if visible is False:
                target = nav_target_phrase(instruction)
                ack = goals.set_goal(target, kind="search")
                if not ack.get("ok"):
                    return ToolResult(text=f"search NOT started: {ack.get('error')}")
                return ToolResult(
                    text=_switch_note(goals, "search", target)
                    + f"{target!r} is NOT in the current view, so a SEARCH was "
                    "started instead of walking blind; tell the user you are "
                    "looking for it (runs in the background)"
                )
        ack = goals.set_goal(instruction, kind="navigate")
        if not ack.get("ok"):
            return ToolResult(text=f"navigation NOT started: {ack.get('error')}")
        return ToolResult(
            text=_switch_note(goals, "navigate", instruction)
            + f"navigation started, following: {instruction!r}. It runs in the "
            "background, step by step (the user can ask you for progress later)"
        )

    async def search(args: dict) -> ToolResult:
        target = str(args.get("object", args.get("target", "")) or "").strip()
        if looks_like_question(target):
            return ToolResult(
                text="that looks like a QUESTION, not an object to find -- "
                "answer it with 'say' (use `status` for goal questions)"
            )
        ack = goals.set_goal(target, kind="search")
        if not ack.get("ok"):
            return ToolResult(text=f"search NOT started: {ack.get('error')}")
        return ToolResult(
            text=_switch_note(goals, "search", target)
            + f"search for {target!r} started; you will scan the room, then "
            "explore unmapped areas until it is found (runs in the background)"
        )

    async def stop(args: dict) -> ToolResult:
        goals.cancel("chat stop")
        sim.submit_command("stop")
        return ToolResult(text="stopped: goal cancelled, robot standing")

    tools = [
        Tool(
            "look",
            "look {}",
            "See through your head camera right now. Use for any question about "
            "what is visible.",
            look,
        ),
        Tool(
            "map",
            "map {}",
            "Your self-built occupancy map + a summary of explored area and "
            "unexplored frontiers. Use for questions about the room layout or "
            "what you know so far.",
            get_map,
        ),
        Tool(
            "route",
            'route {"seconds": <1-120>}',
            "Describe your own movement over the last N seconds.",
            route,
        ),
        Tool(
            "status",
            "status {}",
            "Your pose, current goal and its state, and what your legs are doing.",
            status,
        ),
        Tool(
            "navigate",
            'navigate {"instruction": "<the user\'s route instruction, VERBATIM>"}',
            "Walk somewhere, or follow a route. COPY THE USER'S WHOLE "
            "INSTRUCTION into `instruction` word for word -- every step, "
            "direction and landmark ('walk around the table, then stop by the "
            "door'). Never shorten it to a single object name: the walking "
            "model follows the instruction itself, step by step. Starts in "
            "the background.",
            navigate,
        ),
        Tool(
            "search",
            'search {"object": "<what to find>"}',
            "Look for an object you cannot currently see: systematic scan + "
            "exploration of unmapped areas. Starts in the background.",
            search,
        ),
        Tool("stop", "stop {}", "Cancel the current goal and stand still.", stop),
    ]
    return {t.name: t for t in tools}


def tools_prompt(tools: dict[str, Tool]) -> str:
    lines = []
    for t in tools.values():
        lines.append(f"- {t.signature} -- {t.description}")
    return "\n".join(lines)
