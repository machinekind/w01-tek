"""Episode generation and scoring for the navigation eval.

Episodes are generated against the ORACLE grid + hand-annotated objects.json
(task #: goto / find / turn / multi), each with a guaranteed-reachable goal
and a geodesic shortest-path length -- the SPL denominator. Scoring follows
the Habitat/VLN conventions so numbers are comparable to the literature:

  SR      success rate: agent issued `done` within the goal radius
  SPL     SR weighted by ref_len / max(path_len, ref_len)
  SoftSPL progress-based variant (no done gate)
  DTG     final geodesic distance to goal (m)
  OSR     oracle success: any trail point entered the goal radius
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from wojtek_rl import paths
from wojtek_eval.gridmap import GridMap

GOTO_DIST = (1.2, 4.5)   # geodesic band for visible-ish goals
FIND_DIST = (3.0, 14.0)  # hidden goals: farther, and LOS must be blocked
MULTI_LEG1 = (1.0, 5.0)
MULTI_LEG2 = (1.5, 9.0)
MAX_STEPS = {"goto": 20, "find": 32, "turn": 6, "multi": 36}

GOTO_TEMPLATES = ["go to the {}", "walk to the {}", "head over to the {}"]
FIND_TEMPLATES = ["find the {}", "go find the {}", "look for the {}"]
TURN_TEMPLATES = ["turn around", "turn all the way around", "do a one-eighty"]
MULTI_TEMPLATES = ["go to the {}, then go to the {}", "first visit the {}, then find the {}"]


@dataclass
class Episode:
    id: str
    scene: str
    task: str  # goto | find | turn | multi
    instruction: str
    start: tuple[float, float, float]  # x, y, yaw
    goals: list[dict] = field(default_factory=list)  # [{name, pos, radius}]
    ref_len: float | None = None  # geodesic shortest-path length (m)
    spoken: bool = False
    audio: dict | None = None  # filled by the hearing chain

    @property
    def max_steps(self) -> int:
        return MAX_STEPS[self.task]


def load_objects(scene: str) -> list[dict]:
    return json.loads((paths.scene_dir(scene) / "objects.json").read_text())["objects"]


def _line_of_sight(
    grid: GridMap, a: tuple[float, float], b: tuple[float, float],
    ignore_near_b: float = 0.0,
) -> bool:
    """Straight segment a->b crosses no occupied cell (coarse Bresenham).

    ignore_near_b: occupied cells within this distance of b don't count --
    the goal OBJECT occupies cells around its own centroid, which would
    otherwise block every sight line to it.
    """
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / grid.res))
    for t in np.linspace(0.0, 1.0, n):
        x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        if math.hypot(x - b[0], y - b[1]) <= ignore_near_b:
            continue
        i, j = grid.world_to_cell(x, y)
        if not grid.in_bounds(i, j) or grid.occ[i, j]:
            return False
    return True


def _sample_start(grid, rng, dist_field, band, goal_pos, need_los=None,
                  goal_radius=1.0, tries=400):
    for _ in range(tries):
        x, y = grid.sample_free(rng, clearance=0.08)
        d = grid.distance_at(dist_field, x, y)
        if not (band[0] <= d <= band[1]):
            continue
        if need_los is not None:
            los = _line_of_sight(grid, (x, y), tuple(goal_pos),
                                 ignore_near_b=goal_radius)
            if los is not need_los:
                continue
        return (x, y, float(rng.uniform(-math.pi, math.pi))), d
    return None


def generate(
    scene: str,
    n_per_task: int,
    seed: int = 0,
    tasks: tuple[str, ...] = ("goto", "find", "turn", "multi"),
) -> list[Episode]:
    rng = np.random.default_rng(seed + hash(scene) % 10_000)
    grid = GridMap.load(paths.scene_dir(scene) / "occupancy.npz")
    objects = load_objects(scene)
    fields = {o["name"]: grid.geodesic([tuple(o["pos"])]) for o in objects}
    episodes: list[Episode] = []

    def add(task: str, instruction: str, start, goals, ref_len):
        episodes.append(
            Episode(
                id=f"{scene}-{task}-{len([e for e in episodes if e.task == task]):03d}",
                scene=scene,
                task=task,
                instruction=instruction,
                start=tuple(round(v, 3) for v in start),
                goals=goals,
                ref_len=round(ref_len, 3) if ref_len is not None else None,
            )
        )

    for task in tasks:
        made, guard = 0, 0
        while made < n_per_task and guard < n_per_task * 40:
            guard += 1
            if task == "turn":
                x, y = grid.sample_free(rng, clearance=0.12)
                yaw = float(rng.uniform(-math.pi, math.pi))
                add("turn", str(rng.choice(TURN_TEMPLATES)), (x, y, yaw), [], None)
                made += 1
                continue

            obj = objects[int(rng.integers(len(objects)))]
            name = str(rng.choice([obj["name"], *obj.get("aliases", [])]))
            goal = {"name": obj["name"], "pos": obj["pos"], "radius": obj["radius"]}
            if task == "goto":
                s = _sample_start(grid, rng, fields[obj["name"]], GOTO_DIST, obj["pos"], need_los=True, goal_radius=obj["radius"])
                if s is None:
                    continue
                add("goto", str(rng.choice(GOTO_TEMPLATES)).format(name), s[0], [goal], s[1])
            elif task == "find":
                s = _sample_start(grid, rng, fields[obj["name"]], FIND_DIST, obj["pos"], need_los=False, goal_radius=obj["radius"])
                if s is None:
                    continue
                add("find", str(rng.choice(FIND_TEMPLATES)).format(name), s[0], [goal], s[1])
            elif task == "multi":
                others = [o for o in objects if o["name"] != obj["name"]]
                obj2 = others[int(rng.integers(len(others)))]
                s = _sample_start(grid, rng, fields[obj["name"]], MULTI_LEG1, obj["pos"])
                if s is None:
                    continue
                d12 = grid.distance_at(fields[obj2["name"]], *obj["pos"])
                if not (MULTI_LEG2[0] <= d12 <= MULTI_LEG2[1]):
                    continue
                goal2 = {"name": obj2["name"], "pos": obj2["pos"], "radius": obj2["radius"]}
                add(
                    "multi",
                    str(rng.choice(MULTI_TEMPLATES)).format(obj["name"], obj2["name"]),
                    s[0],
                    [goal, goal2],
                    s[1] + d12,
                )
            made += 1
    return episodes


# -- scoring ---------------------------------------------------------------------


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def score_episode(ep: Episode, result: dict, grid: GridMap) -> dict:
    """result: state, reason, final_pose, path_length, trail, steps, blocked,
    collisions, failures, wall_s (from runner). Returns the metrics row."""
    done = result["state"] == "done" and result.get("reason") == "vlm_done"
    row = {
        "id": ep.id, "scene": ep.scene, "task": ep.task, "spoken": ep.spoken,
        "instruction": ep.instruction,
        "heard": (ep.audio or {}).get("transcript"),
        "wer": (ep.audio or {}).get("wer"),
        "steps": result["steps"], "path_len": round(result["path_length"], 2),
        "blocked": result["blocked"], "failures": result.get("failures", 0),
        "collisions": result.get("collisions", 0),
        "wall_s": round(result.get("wall_s", 0.0), 1),
        "end_state": f"{result['state']}/{result.get('reason')}",
    }
    x, y, yaw = result["final_pose"]

    if ep.task == "turn":
        x0, y0, yaw0 = ep.start
        turned = abs(_wrap(yaw - yaw0)) >= math.radians(140)
        stayed = math.hypot(x - x0, y - y0) <= 0.6
        row.update(success=bool(done and turned and stayed), spl=None, softspl=None,
                   dtg=None, oracle_success=bool(turned and stayed))
        return row

    final_goal = ep.goals[-1]
    gx, gy = final_goal["pos"]
    dist_field = grid.geodesic([(gx, gy)])
    dtg = grid.distance_at(dist_field, x, y)

    def near(px, py) -> bool:
        # Geodesic within radius, OR right next to the object with a clear
        # sight line: standing at the bed's edge counts even when walking
        # around it would take 2 m.
        if grid.distance_at(dist_field, px, py) <= final_goal["radius"]:
            return True
        return (
            math.hypot(px - gx, py - gy) <= final_goal["radius"]
            and _line_of_sight(grid, (px, py), (gx, gy),
                               ignore_near_b=final_goal["radius"] * 0.8)
        )

    near_final = near(x, y)

    visited_all = True
    if ep.task == "multi":
        first = ep.goals[0]
        visited_all = any(
            math.hypot(tx - first["pos"][0], ty - first["pos"][1]) <= first["radius"]
            for tx, ty in result["trail"]
        )

    success = bool(done and near_final and visited_all)
    oracle = bool(visited_all and any(near(tx, ty) for tx, ty in result["trail"]))
    ref = ep.ref_len or 0.0
    p = result["path_length"]
    spl = (ref / max(p, ref)) if (success and ref > 0) else 0.0
    d0 = grid.distance_at(dist_field, ep.start[0], ep.start[1])
    progress = max(0.0, 1.0 - dtg / d0) if d0 and math.isfinite(d0) and d0 > 0 else 0.0
    softspl = progress * (ref / max(p, ref)) if ref > 0 else 0.0
    row.update(
        success=success, oracle_success=oracle, dtg=round(float(dtg), 2),
        spl=round(spl, 3), softspl=round(softspl, 3),
    )
    return row


def summarize(rows: list[dict]) -> dict:
    """Aggregate means per task and overall (None-safe)."""

    def agg(sel: list[dict]) -> dict:
        def mean(key):
            vals = [r[key] for r in sel if r.get(key) is not None]
            return round(float(np.mean(vals)), 3) if vals else None

        return {
            "n": len(sel), "sr": mean("success"), "oracle_sr": mean("oracle_success"),
            "spl": mean("spl"), "softspl": mean("softspl"), "dtg": mean("dtg"),
            "steps": mean("steps"), "blocked": mean("blocked"),
            "collisions": mean("collisions"), "wer": mean("wer"),
        }

    out = {"overall": agg(rows)}
    for task in sorted({r["task"] for r in rows}):
        out[task] = agg([r for r in rows if r["task"] == task])
    for scene in sorted({r["scene"] for r in rows}):
        out[f"scene:{scene}"] = agg([r for r in rows if r["scene"] == scene])
    spoken = [r for r in rows if r["spoken"]]
    typed = [r for r in rows if not r["spoken"]]
    if spoken and typed:
        out["spoken"] = agg(spoken)
        out["typed"] = agg(typed)
    return out


def episodes_to_json(episodes: list[Episode]) -> str:
    return json.dumps([asdict(e) for e in episodes], indent=1)


def episodes_from_json(text: str) -> list[Episode]:
    """Reload a suite verbatim (runner --suite).

    Cross-model and cross-planner scoreboards are only comparable on the
    *same* episodes: regenerating changes the rng draws (and therefore the
    start poses, goals and spoken flags) whenever the generator or the task
    mix changes. Pin the suite instead of trusting a seed."""
    out = []
    for d in json.loads(text):
        d = dict(d)
        d["start"] = tuple(d["start"])
        out.append(Episode(**d))
    return out
