"""Object search: frontier exploration scored by a VLM, with verification.

VLFM's skeleton (Yokoyama et al., ICRA 2024) sized for this stack: the
occupancy map the robot already builds (RoomSim.omap) supplies frontiers, a
value map accumulates per-view VLM scores splatted over the camera's FOV
cone, and the search commits to the best-value frontier until it is reached
(commitment, not per-step re-selection -- greedy re-selection oscillates).
Detections are never trusted from one frame: approach to close range and
re-detect k-of-n before declaring success (TriHelper's verify step; false
positives are the dominant real-world failure mode), and a failed verify
blacklists the spot so the search doesn't orbit a phantom.

The FSM is classical code; the VLM only scores views and verifies detections
(CogNav's lesson: let states be code, the model works inside them).

State flow:
    scanning -> selecting -> moving -> (detection) -> approaching ->
    verifying -> found | blacklist -> selecting ... -> not_found

No mujoco import; the sim is duck-typed exactly like VlmNavigator expects
(pose / submit_command / executor / ego_jpeg / omap), so unit tests drive the
whole FSM with a fake sim and a scripted scorer.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

import numpy as np
from loguru import logger

from wojtek_agent.parsing import iter_json_objects, strip_think
from wojtek_agent.spatial import FrontierCluster, frontier_clusters
from wojtek_rl.vlm_nav import _safe_err

SCAN_STOPS = 8
SCAN_TURN_DEG = 45.0
DETECT_SCORE = 5.0        # visible + score >= this -> worth approaching
DETECT_PERIOD_S = 1.5     # detection cadence while driving
POLL_S = 0.1
CMD_TIMEOUT_S = 30.0
MAX_WALL_S = 240.0
VERIFY_FRAMES = 3
VERIFY_MIN = 2
VERIFY_PAUSE_S = 0.3
APPROACH_STEP_M = 0.6
APPROACH_MAX_STEPS = 8
CLOSE_BBOX_FRAC = 0.30    # bbox width fraction of the image -> close enough
CENTER_TOL_DEG = 12.0
ASSUMED_TARGET_DIST_M = 1.0  # camera-only range guess for found/blacklist marks
BLACKLIST_R = 0.7
ATTEMPT_R = 0.5
DIST_LAMBDA = 0.5         # value points traded per metre of travel
EVENTS_MAX = 200          # debug event ring: commands, observations, states


def score_view_prompt(target: str) -> str:
    """The observer call's instruction — text in prompts/search_observer.txt.
    `description` comes FIRST in the contract -- a small VLM that must open
    with a verdict stops looking at the image (see parsing.py)."""
    from wojtek_agent.prompts import load

    return load("search_observer", target=target)



@dataclass(frozen=True)
class ViewScore:
    """One scored camera view."""

    description: str
    visible: bool
    score: float
    bbox: tuple[float, float, float, float] | None = None


def parse_view_score(obj: dict) -> ViewScore:
    """ViewScore from the observer's JSON (already extracted to a dict)."""
    bbox = obj.get("bbox_2d")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        bbox = tuple(float(v) for v in bbox)
    else:
        bbox = None
    score = obj.get("score", 0)
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    return ViewScore(
        description=str(obj.get("description", "") or ""),
        visible=bool(obj.get("target_visible", False)),
        score=max(0.0, min(10.0, score)),
        bbox=bbox,
    )


def make_score_view(llm):
    """Bind the observer call to an AgentLLM (or anything with .chat).

    A missing/garbled JSON reply degrades to 'nothing seen, score 0' rather
    than raising -- one bad observation must not kill a whole search.
    """
    from wojtek_agent.llm import user_message

    async def score_view(target: str, frame_b64: str) -> ViewScore:
        raw = await llm.chat([user_message(score_view_prompt(target), (frame_b64,))])
        try:
            obj = next(iter_json_objects(strip_think(raw)), None)
        except ValueError:
            obj = None
        if obj is None:
            logger.warning(f"observer reply unparseable: {raw[:120]!r}")
            return ViewScore(description="", visible=False, score=0.0)
        return parse_view_score(obj)

    return score_view


class ValueMap:
    """Semantic value grid over the same geometry as the OnlineMap.

    Each observation splats its score over the camera's FOV cone with cos^2
    axis weighting and fuses by confidence-weighted average (VLFM's update
    rule), so a view straight down a corridor counts more than the cone edge
    that grazed it. No occlusion ray-cast: max_range keeps splats local and
    decay() fades stale (possibly drift-misregistered) evidence.
    """

    def __init__(
        self,
        res: float,
        origin: tuple[float, float],
        shape: tuple[int, int],
        hfov_deg: float = 100.0,
        max_range: float = 4.0,
    ):
        self.res = res
        self.origin = origin
        self.shape = shape
        self.hfov = math.radians(hfov_deg)
        self.max_range = max_range
        self.value = np.zeros(shape, np.float32)
        self.conf = np.zeros(shape, np.float32)
        jj, ii = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        self._cx = origin[0] + (jj + 0.5) * res
        self._cy = origin[1] + (ii + 0.5) * res

    def splat(self, x: float, y: float, yaw: float, score: float) -> None:
        dx = self._cx - x
        dy = self._cy - y
        dist = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx) - yaw
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        half = self.hfov / 2
        mask = (dist <= self.max_range) & (np.abs(bearing) <= half)
        if not mask.any():
            return
        w = np.cos(bearing[mask] / half * (np.pi / 2)) ** 2
        v, c = self.value[mask], self.conf[mask]
        tot = w + c
        self.value[mask] = (w * score + c * v) / np.maximum(tot, 1e-9)
        self.conf[mask] = (w**2 + c**2) / np.maximum(tot, 1e-9)

    def decay(self, factor: float = 0.995) -> None:
        self.conf *= factor

    def value_near(self, x: float, y: float, radius_m: float = 0.3) -> float:
        """Confidence-weighted mean value in a disc (frontier scoring)."""
        dist = np.hypot(self._cx - x, self._cy - y)
        m = dist <= radius_m
        if not m.any():
            return 0.0
        c = self.conf[m]
        tot = float(c.sum())
        if tot < 1e-9:
            return 0.0
        return float((self.value[m] * c).sum() / tot)

    def confidence_near(self, x: float, y: float, radius_m: float = 0.3) -> float:
        dist = np.hypot(self._cx - x, self._cy - y)
        m = dist <= radius_m
        return float(self.conf[m].max()) if m.any() else 0.0


class SearchController:
    """Runs one object search as an asyncio task on the sim's event loop.

    `score_view(target, frame_b64) -> ViewScore` is injected (chat.py binds
    it to the AgentLLM); everything else is grid math and executor commands.
    Same event-loop contract as VlmNavigator: never steps the sim, so it
    pauses while no browser client drives the control loop.
    """

    def __init__(
        self,
        sim,
        score_view,
        hfov_deg: float = 100.0,
        max_wall_s: float = MAX_WALL_S,
        cmd_timeout_s: float = CMD_TIMEOUT_S,
        poll_s: float = POLL_S,
        verify_pause_s: float = VERIFY_PAUSE_S,
        detect_period_s: float = DETECT_PERIOD_S,
        frame_fn=None,
        trace=None,
    ):
        self.sim = sim
        self.score_view = score_view
        # The observer must see the BARE camera: it is never told about the
        # HUD minimap, so it reads the inset as part of the room and can
        # "detect" the target inside its own map (measured, see room_app's
        # ego_jpeg docstring). Callers pass a hud-free grabber; the default
        # keeps duck-typed sims (and the tests) working.
        self.frame_fn = frame_fn or sim.ego_jpeg
        self.trace = trace
        self.hfov_deg = hfov_deg
        self.max_wall_s = max_wall_s
        self.cmd_timeout_s = cmd_timeout_s
        self.poll_s = poll_s
        self.verify_pause_s = verify_pause_s
        self.detect_period_s = detect_period_s

        self.rev = 0
        self._task: asyncio.Task | None = None
        self._state = "idle"
        self._target: str | None = None
        self._note: str | None = None
        self._events: list[dict] = []
        self._blacklist: list[tuple[float, float]] = []
        self._attempted: list[tuple[float, float]] = []
        self._found_xy: tuple[float, float] | None = None
        self.value: ValueMap | None = None
        self._observations = 0

    # -- control -------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, target: str) -> dict:
        target = target.strip()
        if not target:
            return {"ok": False, "error": "empty search target"}
        if self.running:
            return {"ok": False, "error": f"already searching for {self._target!r}"}
        omap = self.sim.omap
        self._target = target
        self._events = []
        self._blacklist = []
        self._attempted = []
        self._found_xy = None
        self._observations = 0
        self.value = ValueMap(omap.res, omap.origin, omap.shape, hfov_deg=self.hfov_deg)
        self._set_state("scanning", note="initial look-around")
        self._task = asyncio.create_task(self._run(target))
        return {"ok": True, "target": target}

    def cancel(self, reason: str = "user"):
        if self._task is not None and not self._task.done():
            self._task.cancel()
            self.sim.submit_command("stop")
            logger.info(f"search cancelled ({reason})")
        if self._state not in ("found", "not_found", "error", "idle"):
            self._set_state("idle", note=f"cancelled ({reason})")

    def status(self) -> dict:
        return {
            "state": self._state,
            "target": self._target,
            "note": self._note,
            "observations": self._observations,
            "found_xy": list(self._found_xy) if self._found_xy else None,
            "blacklisted": len(self._blacklist),
            "attempted": len(self._attempted),
            "events": self._events[-20:],
        }

    def _set_state(self, state: str, note: str | None = None):
        self._state = state
        if note is not None:
            self._note = note
            self._event(f"[{state}] {note}")
        self.rev += 1

    def _event(self, text: str):
        self._events.append({"state": self._state, "text": text})
        del self._events[:-EVENTS_MAX]
        if self.trace is not None:
            x, y, yaw = self.sim.pose()
            self.trace.add(
                f"search.{self._state}",
                target=self._target,
                text=text,
                pose=[round(x, 3), round(y, 3), round(math.degrees(yaw), 1)],
            )

    # -- the FSM ---------------------------------------------------------------

    async def _run(self, target: str):
        try:
            await self._run_inner(target)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # never leave a zombie "searching" status
            logger.error(f"search crashed: {_safe_err(e)}")
            self._set_state("error", note=_safe_err(e))

    async def _run_inner(self, target: str):
        deadline = asyncio.get_event_loop().time() + self.max_wall_s

        # -- initial 360 look-around: seeds both the occupancy map (each
        # observation fuses depth via ego_jpeg) and the value map.
        for k in range(SCAN_STOPS):
            vs = await self._observe(target)
            if self._promising(vs) and await self._try_candidate(target, vs):
                return
            if k < SCAN_STOPS - 1:
                await self._do_cmd(f"turn_left {SCAN_TURN_DEG:g}")

        # -- frontier loop: commit to the best-value frontier, watch for the
        # target while driving, look around on arrival.
        while asyncio.get_event_loop().time() < deadline:
            self._set_state("selecting")
            goal = self._select_frontier()
            if goal is None:
                self._set_state("not_found", note="no unexplored frontiers left")
                return
            self._attempted.append((goal.x, goal.y))
            self._set_state("moving", note=f"heading to frontier at ({goal.x:.1f}, {goal.y:.1f})")
            det = await self._drive_and_watch(goal, target)
            if det is not None and await self._try_candidate(target, det):
                return
            if det is not None:
                continue  # verify failed; spot is blacklisted, pick again
            # Arrived (or gave up on the leg): sweep +-60 deg for a fresh look.
            for cmd in (None, "turn_left 60", "turn_right 120"):
                if cmd is not None:
                    await self._do_cmd(cmd)
                vs = await self._observe(target)
                if self._promising(vs) and await self._try_candidate(target, vs):
                    return
            self.value.decay(0.98)
        self._set_state("not_found", note="search time budget exhausted")

    def _promising(self, vs: ViewScore) -> bool:
        return vs.visible and vs.score >= DETECT_SCORE

    async def _try_candidate(self, target: str, vs: ViewScore) -> bool:
        """Approach + verify one detection; True ends the search as found."""
        self._set_state("approaching", note=f"possible {target}: {vs.description[:60]}")
        ok = await self._approach_and_verify(target, vs)
        if ok:
            self._found_xy = self._point_ahead(ASSUMED_TARGET_DIST_M)
            self._set_state("found", note=f"{target} confirmed")
            self.sim.submit_command("stop")
            return True
        spot = self._point_ahead(ASSUMED_TARGET_DIST_M)
        self._blacklist.append(spot)
        self._set_state("selecting", note="candidate did not verify; spot blacklisted")
        return False

    # -- observation -----------------------------------------------------------

    async def _observe(self, target: str) -> ViewScore:
        """Grab a frame (this also fuses depth into the occupancy map), score
        it, and splat the score into the value map at the current pose."""
        frame = self.frame_fn()
        vs = await self.score_view(target, frame)
        x, y, yaw = self.sim.pose()
        self.value.splat(x, y, yaw, vs.score)
        self._observations += 1
        flag = " VISIBLE" if vs.visible else ""
        bbox = f" bbox={[round(v) for v in vs.bbox]}" if vs.bbox else ""
        self._event(
            f"obs #{self._observations} @({x:.1f},{y:.1f},{math.degrees(yaw):.0f}deg) "
            f"score {vs.score:g}{flag}{bbox}: {vs.description[:70]}"
        )
        self.rev += 1
        return vs

    # -- frontier selection ------------------------------------------------------

    def _near_any(self, x: float, y: float, points, r: float) -> bool:
        return any(math.hypot(x - px, y - py) <= r for px, py in points)

    def _select_frontier(self) -> FrontierCluster | None:
        clusters = frontier_clusters(self.sim.omap)
        px, py, _ = self.sim.pose()
        best: FrontierCluster | None = None
        best_score = -math.inf
        any_conf = False
        skipped = 0
        for c in clusters:
            if self._near_any(c.x, c.y, self._blacklist, BLACKLIST_R):
                skipped += 1
                continue
            if self._near_any(c.x, c.y, self._attempted, ATTEMPT_R):
                skipped += 1
                continue
            dist = math.hypot(c.x - px, c.y - py)
            conf = self.value.confidence_near(c.x, c.y, radius_m=0.5)
            any_conf = any_conf or conf > 1e-3
            score = self.value.value_near(c.x, c.y, radius_m=0.5) - DIST_LAMBDA * dist
            if score > best_score:
                best, best_score = c, score
        self._event(
            f"frontiers: {len(clusters)} clusters, {skipped} skipped "
            f"(blacklist/attempted), best score {best_score:.2f}"
            + (" [FBE fallback: no value info]" if best is not None and not any_conf else "")
        )
        if best is not None and not any_conf:
            # Value map is uninformative here: plain nearest-frontier FBE is
            # the proven fallback (CoW), so re-pick by distance alone.
            best = min(
                (
                    c
                    for c in clusters
                    if not self._near_any(c.x, c.y, self._blacklist, BLACKLIST_R)
                    and not self._near_any(c.x, c.y, self._attempted, ATTEMPT_R)
                ),
                key=lambda c: math.hypot(c.x - px, c.y - py),
            )
        return best

    # -- motion ------------------------------------------------------------------

    async def _do_cmd(self, cmd: str) -> bool:
        """Submit one mid-level command and wait it out; False on rejection,
        timeout, or a blocked outcome (callers treat all three as 'that move
        didn't happen')."""
        ack = self.sim.submit_command(cmd)
        if not ack.get("ok"):
            self._event(f"cmd {cmd} -> rejected ({ack.get('error')})")
            return False
        self._event(f"cmd {cmd}")
        self.rev += 1
        blocked0 = getattr(self.sim.executor, "blocked", 0)
        waited = 0.0
        while self.sim.executor.active:
            await asyncio.sleep(self.poll_s)
            waited += self.poll_s
            if waited >= self.cmd_timeout_s:
                self.sim.submit_command("stop")
                self._event(f"cmd {cmd} -> timed out")
                return False
        if getattr(self.sim.executor, "blocked", 0) != blocked0:
            self._event(f"cmd {cmd} -> blocked")
            return False
        return True

    async def _drive_and_watch(self, goal: FrontierCluster, target: str) -> ViewScore | None:
        """Drive to the frontier, running detection at ~1/detect_period_s.

        Returns the detection that interrupted the drive, or None when the
        leg ended (arrived / blocked / timed out) without one.
        """
        executor = self.sim.executor
        goto = getattr(executor, "goto", None)
        if goto is not None:
            px, py, _ = self.sim.pose()
            goto(goal.x, goal.y, (px, py))
            self._event(f"goto ({goal.x:.1f},{goal.y:.1f}) via planner")
            return await self._watch_leg(target)
        # No planner: steer with turn+forward hops toward the frontier.
        for _ in range(6):
            x, y, yaw = self.sim.pose()
            dist = math.hypot(goal.x - x, goal.y - y)
            if dist < 0.4:
                return None
            bearing = math.degrees(
                (math.atan2(goal.y - y, goal.x - x) - yaw + math.pi) % (2 * math.pi) - math.pi
            )
            if abs(bearing) > 15:
                side = "turn_left" if bearing > 0 else "turn_right"
                if not await self._do_cmd(f"{side} {abs(bearing):.0f}"):
                    return None
            ack = self.sim.submit_command(f"forward {min(dist, 1.2):.2f}")
            if not ack.get("ok"):
                return None
            det = await self._watch_leg(target)
            if det is not None:
                return det
        return None

    async def _watch_leg(self, target: str) -> ViewScore | None:
        waited = 0.0
        since_detect = 0.0
        while self.sim.executor.active:
            await asyncio.sleep(self.poll_s)
            waited += self.poll_s
            since_detect += self.poll_s
            if waited >= self.cmd_timeout_s:
                self.sim.submit_command("stop")
                self._event("drive leg timed out")
                return None
            if since_detect >= self.detect_period_s:
                since_detect = 0.0
                vs = await self._observe(target)
                if self._promising(vs):
                    self.sim.submit_command("stop")
                    return vs
        return None

    # -- approach + verify ----------------------------------------------------------

    def _bbox_bearing_deg(self, bbox) -> float | None:
        """Horizontal bearing to the bbox center, +left, from 0-1000 coords."""
        if bbox is None:
            return None
        cx = (bbox[0] + bbox[2]) / 2.0 / 1000.0
        return (0.5 - cx) * self.hfov_deg

    async def _approach_and_verify(self, target: str, vs: ViewScore) -> bool:
        lost = 0
        for _ in range(APPROACH_MAX_STEPS):
            bearing = self._bbox_bearing_deg(vs.bbox)
            if bearing is not None and abs(bearing) > CENTER_TOL_DEG:
                side = "turn_left" if bearing > 0 else "turn_right"
                await self._do_cmd(f"{side} {min(abs(bearing), 90):.0f}")
            close = (
                vs.bbox is not None
                and (vs.bbox[2] - vs.bbox[0]) / 1000.0 >= CLOSE_BBOX_FRAC
            )
            advanced = True
            if not close:
                advanced = await self._do_cmd(f"forward {APPROACH_STEP_M:g}")
            if close or not advanced:
                # Point-blank (or can't get closer): settle it now.
                return await self._verify(target)
            vs = await self._observe(target)
            if not vs.visible:
                lost += 1
                if lost >= 2:
                    self._event("lost the candidate while approaching")
                    return False
            else:
                lost = 0
        return await self._verify(target)

    async def _verify(self, target: str) -> bool:
        """k-of-n stationary re-detection: one distant frame is never enough
        (VLM false positives dominate real-world search failures)."""
        self._set_state("verifying", note=f"double-checking the {target}")
        hits = 0
        for _ in range(VERIFY_FRAMES):
            await asyncio.sleep(self.verify_pause_s)
            vs = await self._observe(target)
            hits += int(vs.visible)
        self._event(f"verify: {hits}/{VERIFY_FRAMES} positive")
        return hits >= VERIFY_MIN

    def _point_ahead(self, dist_m: float) -> tuple[float, float]:
        x, y, yaw = self.sim.pose()
        return (x + math.cos(yaw) * dist_m, y + math.sin(yaw) * dist_m)
