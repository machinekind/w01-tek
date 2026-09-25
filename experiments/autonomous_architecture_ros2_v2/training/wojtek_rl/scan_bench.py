"""A/B the SCAN local planner against the straight-march executor.

The failure this exists to measure: whatever the VLM backend is -- FutureNav,
Qwen3-VL, Claude -- the robot walks into furniture, because a mid-level
"forward 1.5" was executed as a straight line and nothing in the loop looked
at the depth channel while the robot was moving.

To measure the planner rather than the VLM, guidance here is an *oracle VLM*:
a scripted policy with perfect knowledge of where the goal is, emitting
exactly the commands the mid-level interface accepts -- turn to face the
goal, walk toward it, repeat. That is the best case for any VLM. Every
episode is generated so that the straight line from start to goal is blocked
by an obstacle but a route exists, which is exactly the situation the VLM
cannot resolve on its own and the local planner can.

    ./run.sh scan-bench --scene room --episodes 12
    ./run.sh scan-bench --scene apartment --episodes 8 --video

Outputs land in runs/scan_bench/<timestamp>_<scene>/ (gitignored): a
scoreboard JSON, per-episode rows, a top-down PNG of both trajectories, and
optionally an MP4 of the planner run.
"""

from __future__ import annotations

# Headless rendering backend; must precede any mujoco import.
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl" if sys.platform == "linux" else "cgl")

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from wojtek_eval.gridmap import GridMap
from wojtek_eval.kinsim import KinematicSim
from wojtek_rl import paths

GOAL_RADIUS = 0.5      # success if the robot stops this close to the goal
MAX_DECISIONS = 14     # oracle-VLM decisions per episode
STEP_M = 1.5           # how far one "forward" command asks for
TURN_DEADBAND_DEG = 12.0
PHYSICS_CMD_TIMEOUT_S = 20.0  # per mid-level command, physics tier


@dataclass
class Episode:
    idx: int
    start: tuple[float, float, float]
    goal: tuple[float, float]
    geodesic_m: float
    straight_m: float


@dataclass
class Result:
    idx: int
    mode: str
    success: bool
    collisions: int
    off_floor: int
    blocked: int
    path_m: float
    final_dist_m: float
    spl: float
    decisions: int
    wall_s: float
    falls: int = 0  # physics tier only


def generate_episodes(grid: GridMap, n: int, seed: int,
                      min_m: float = 1.5, max_m: float = 4.0) -> list[Episode]:
    """Start/goal pairs whose straight line is blocked but which are reachable.

    Straight-line-blocked is the whole point: on an open floor the straight
    march and the planner do the same thing, and the comparison says nothing.
    """
    rng = np.random.default_rng(seed)
    out: list[Episode] = []
    tries = 0
    while len(out) < n and tries < 4000:
        tries += 1
        sx, sy = grid.sample_free(rng, clearance=0.25)
        gx, gy = grid.sample_free(rng, clearance=0.20)
        straight = math.hypot(gx - sx, gy - sy)
        if not (min_m <= straight <= max_m):
            continue
        # Spread the suite out: a small room otherwise hands back the same
        # corridor four times and the scoreboard measures one situation.
        if any(math.hypot(sx - e.start[0], sy - e.start[1]) < 0.6 for e in out):
            continue
        if not _line_blocked(grid, (sx, sy), (gx, gy)):
            continue
        path = grid.shortest_path((sx, sy), (gx, gy))
        if path is None:
            continue
        geo = sum(
            math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])
        )
        if geo > 2.5 * straight:  # a detour that long is a different room
            continue
        yaw = math.atan2(gy - sy, gx - sx)
        out.append(Episode(len(out), (sx, sy, yaw), (gx, gy), geo, straight))
    if len(out) < n:
        logger.warning(f"only generated {len(out)}/{n} blocked-line episodes")
    return out


def _line_blocked(grid: GridMap, a, b) -> bool:
    """True if the straight segment crosses a cell the robot cannot stand in."""
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (0.5 * grid.res)))
    t = np.linspace(0.0, 1.0, n)
    xs = a[0] + t * (b[0] - a[0])
    ys = a[1] + t * (b[1] - a[1])
    return any(not grid.is_free(float(x), float(y)) for x, y in zip(xs, ys))


def oracle_command(pose, goal) -> str | None:
    """The command a perfect VLM would emit: face the goal, then walk at it."""
    x, y, yaw = pose
    dx, dy = goal[0] - x, goal[1] - y
    dist = math.hypot(dx, dy)
    if dist < GOAL_RADIUS:
        return None
    err = math.degrees((math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi)
    if abs(err) > TURN_DEADBAND_DEG:
        side = "turn_left" if err > 0 else "turn_right"
        return f"{side} {min(abs(err), 90.0):.0f}"
    return f"forward {min(dist, STEP_M):.2f}"


def run_episode(scene: str, ep: Episode, planner: bool,
                media_dir: Path | None = None) -> tuple[Result, list]:
    sim = KinematicSim(scene, start=ep.start, local_planner=planner, hud=False)
    trail: list[tuple[float, float]] = [(sim.x, sim.y)]
    frames: list = []
    if media_dir is not None:
        sim.step_hook = lambda s: frames.append(_media_frame(s, ep))
    t0 = time.time()
    decisions = 0
    try:
        for _ in range(MAX_DECISIONS):
            cmd = oracle_command(sim.pose(), ep.goal)
            if cmd is None:
                break
            decisions += 1
            ack = sim.submit_command(cmd)
            if not ack["ok"]:
                break
            _ = sim.executor.active  # deferred execution happens here
            trail.append((sim.x, sim.y))
        dist = math.hypot(sim.x - ep.goal[0], sim.y - ep.goal[1])
        success = dist < GOAL_RADIUS
        spl = (ep.geodesic_m / max(sim.path_length, ep.geodesic_m)) if success else 0.0
        res = Result(
            idx=ep.idx,
            mode="scan" if planner else "straight",
            success=success,
            collisions=sim.collisions,
            off_floor=sim.off_floor,
            blocked=sim.executor.blocked,
            path_m=round(sim.path_length, 3),
            final_dist_m=round(dist, 3),
            spl=round(spl, 3),
            decisions=decisions,
            wall_s=round(time.time() - t0, 1),
        )
        if frames and media_dir is not None:
            _write_video(frames, media_dir / f"ep{ep.idx:02d}.mp4")
        return res, trail
    finally:
        sim.close()


def run_episode_physics(scene: str, ep: Episode, planner: bool, policy: str,
                        media_dir: Path | None = None) -> tuple[Result, list]:
    """Same episode, same guidance, with legs: MuJoCo + the locomotion policy.

    Tier B of the roadmap. Collisions here are *real contacts* -- a robot geom
    touching scene geometry above the floor band -- not the kinematic tier's
    oracle-grid test. The two are not interchangeable: the grid is inflated by
    0.16 m, so a physical body can pass through cells the grid calls occupied,
    and counting those as hits would invent collisions that never happened.
    Falls are counted separately: tripping over a chair leg is a different
    failure from walking into it.
    """
    from wojtek_rl.room_app import CONTROL_HZ, RoomSim

    sim = RoomSim(paths.scene_xml(scene), policy, local_planner=planner)
    _place(sim, ep.start)
    contact = ContactMonitor(sim)
    trail: list[tuple[float, float]] = [ep.start[:2]]
    frames: list = []
    falls, path_m = 0, 0.0
    resets0 = sim.resets
    prev = np.array(ep.start[:2])
    t0 = time.time()
    decisions = 0
    try:
        for _ in range(MAX_DECISIONS):
            cmd = oracle_command(sim.pose(), ep.goal)
            if cmd is None:
                break
            decisions += 1
            if not sim.submit_command(cmd)["ok"]:
                break
            for tick in range(int(PHYSICS_CMD_TIMEOUT_S * CONTROL_HZ)):
                sim.step()
                x, y, _ = sim.pose()
                path_m += float(np.linalg.norm(np.array([x, y]) - prev))
                prev = np.array([x, y])
                contact.tick()
                if media_dir is not None and tick % 12 == 0:
                    frames.append(_physics_frame(sim))
                if sim.resets > resets0:  # the fall handler reset the pose
                    falls += 1
                    resets0 = sim.resets
                    break
                if not sim.executor.active:
                    break
            trail.append(sim.pose()[:2])
            if falls:
                break
        x, y, _ = sim.pose()
        dist = math.hypot(x - ep.goal[0], y - ep.goal[1])
        success = dist < GOAL_RADIUS and not falls
        res = Result(
            idx=ep.idx, mode=("scan" if planner else "straight") + "-phys",
            success=success, collisions=contact.events, off_floor=0,
            blocked=sim.executor.blocked, path_m=round(path_m, 3),
            final_dist_m=round(dist, 3),
            spl=round(ep.geodesic_m / max(path_m, ep.geodesic_m), 3) if success else 0.0,
            decisions=decisions, wall_s=round(time.time() - t0, 1), falls=falls,
        )
        if frames and media_dir is not None:
            _write_video(frames, media_dir / f"ep{ep.idx:02d}_physics.mp4")
        return res, trail
    finally:
        if sim.scan is not None:
            sim.scan.close()


class ContactMonitor:
    """Counts body-vs-furniture contact events during a physics rollout.

    A contact counts when a robot geom touches a non-robot geom above
    ``FLOOR_BAND_Z`` -- the same lower bound the occupancy map uses for "this
    is an obstacle, not floor". Feet on the ground sit below it. Rising edges
    only: leaning on a sofa for two seconds is one collision, not a hundred.
    """

    FLOOR_BAND_Z = 0.06

    def __init__(self, sim):
        import mujoco

        model = sim.model
        free = [
            j for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        root = int(model.body_rootid[model.jnt_bodyid[free[0]]])
        bodies = np.flatnonzero(np.asarray(model.body_rootid) == root)
        self._robot_geoms = set(
            np.flatnonzero(np.isin(np.asarray(model.geom_bodyid), bodies)).tolist()
        )
        self.sim = sim
        self.events = 0
        self.ticks_in_contact = 0
        self._touching = False

    def touching(self) -> bool:
        d = self.sim.data
        for c in range(d.ncon):
            con = d.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if (g1 in self._robot_geoms) == (g2 in self._robot_geoms):
                continue  # self-contact or scene-scene
            if float(con.pos[2]) > self.FLOOR_BAND_Z:
                return True
        return False

    def tick(self) -> None:
        now = self.touching()
        if now:
            self.ticks_in_contact += 1
            if not self._touching:
                self.events += 1
        self._touching = now


def _place(sim, start: tuple[float, float, float]) -> None:
    """Drop the robot at the episode start pose and let it settle."""
    sim.reset()
    x, y, yaw = start
    sim.data.qpos[0:2] = (x, y)
    sim.data.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    sim.data.qvel[:] = 0.0
    sim._mujoco.mj_forward(sim.model, sim.data)
    for _ in range(int(0.4 / sim.model.opt.timestep)):
        sim._mujoco.mj_step(sim.model, sim.data)
    sim.data.qvel[:] = 0.0
    sim._mujoco.mj_forward(sim.model, sim.data)
    sim.policy.reset()


def _physics_frame(sim) -> np.ndarray:
    """Chase view beside the planner's map (or the agent map without it)."""
    from PIL import Image

    sim.chase_renderer.update_scene(sim.data, camera=sim.cam)
    chase = sim.chase_renderer.render().copy()
    h = chase.shape[0]
    if sim.scan is not None:
        from wojtek_rl.scan.viz import plan_image

        panel = plan_image(sim.scan.map, sim.pose(), sim.scan.planner,
                           px=h, footprint=sim.scan.footprint)
    else:
        panel = np.asarray(
            Image.fromarray(sim.omap.map_image(sim.pose(), px=h)).resize((h, h))
        )
    out = np.zeros((h, chase.shape[1] + panel.shape[1], 3), np.uint8)
    out[:, : chase.shape[1]] = chase
    out[: panel.shape[0], chase.shape[1] :] = panel
    return out


def _media_frame(sim: KinematicSim, ep: Episode) -> np.ndarray:
    """Chase view beside the planner's own map, side by side."""
    from PIL import Image

    from wojtek_rl.scan.viz import plan_image

    chase = sim.frame_png()
    h = chase.shape[0]
    if sim.scan is not None:
        panel = plan_image(
            sim.scan.map, sim.pose(), sim.scan.executor.planner,
            px=h, footprint=sim.scan.footprint,
        )
    else:
        panel = np.asarray(
            Image.fromarray(sim.omap.map_image(sim.pose(), px=h)).resize((h, h))
        )
    out = np.zeros((h, chase.shape[1] + panel.shape[1], 3), np.uint8)
    out[:, : chase.shape[1]] = chase
    out[: panel.shape[0], chase.shape[1] :] = panel
    return out


def _write_video(frames: list, path: Path, fps: int = 5) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    w = min(f.shape[1] for f in frames)
    h = min(f.shape[0] for f in frames)
    imageio.mimwrite(str(path), [f[:h, :w] for f in frames], fps=fps, macro_block_size=1)


def trajectory_png(grid: GridMap, episodes: list[Episode], trails: dict, path: Path) -> None:
    """Both modes' trajectories over the oracle map: the picture of the result."""
    from PIL import Image

    img = np.full(grid.occ.shape + (3,), 40, np.uint8)
    img[grid.free] = (215, 215, 215)
    img[grid.occ] = (185, 60, 50)

    def draw(points, color):
        for x, y in points:
            i, j = grid.world_to_cell(x, y)
            if grid.in_bounds(i, j):
                img[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2] = color

    for ep in episodes:
        for mode, color in (
            ("straight", (240, 170, 60)), ("scan", (70, 200, 120)),
            ("straight-phys", (240, 170, 60)), ("scan-phys", (70, 200, 120)),
        ):
            key = (ep.idx, mode)
            if key in trails:
                draw(_densify(trails[key]), color)
        draw([ep.start[:2]], (90, 140, 250))
        draw([ep.goal], (240, 90, 200))
    scale = max(1, int(900 / max(img.shape[:2])))
    out = Image.fromarray(img[::-1]).resize(
        (img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST
    )
    out.save(path)


def _densify(trail, step: float = 0.03):
    pts = []
    for a, b in zip(trail, trail[1:]):
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        for t in np.linspace(0, 1, n):
            pts.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return pts or list(trail)


def summarize(rows: list[Result]) -> dict:
    out = {}
    for mode in ("straight", "scan", "straight-phys", "scan-phys"):
        sel = [r for r in rows if r.mode == mode]
        if not sel:
            continue
        out[mode] = {
            "episodes": len(sel),
            "success_rate": round(sum(r.success for r in sel) / len(sel), 3),
            "spl": round(float(np.mean([r.spl for r in sel])), 3),
            "collisions_total": sum(r.collisions for r in sel),
            # Per metre travelled, over the SUITE total -- a mode that barely
            # moves collects its hits in a very short path, and dividing by a
            # per-episode mean would flatter it by a factor of n.
            "collisions_per_m": round(
                sum(r.collisions for r in sel) / max(sum(r.path_m for r in sel), 1e-9), 2
            ),
            "collision_episodes": sum(1 for r in sel if r.collisions),
            "off_floor_total": sum(r.off_floor for r in sel),
            "blocked_total": sum(r.blocked for r in sel),
            "falls_total": sum(r.falls for r in sel),
            "path_m_mean": round(float(np.mean([r.path_m for r in sel])), 2),
            "final_dist_m_mean": round(float(np.mean([r.final_dist_m for r in sel])), 2),
            "wall_s_mean": round(float(np.mean([r.wall_s for r in sel])), 1),
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", default="room")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("both", "scan", "straight"), default="both")
    ap.add_argument(
        "--tier", choices=("kin", "physics"), default="kin",
        help="kin = kinematic base (fast, Tier A); physics = MuJoCo + the "
        "locomotion policy walking the same episodes (Tier B)",
    )
    ap.add_argument(
        "--policy", default=None,
        help="physics tier: policy reference (default: $WOJTEK_POLICY or the "
        "keeper in paths.DEFAULT_POLICY)",
    )
    ap.add_argument("--video", action="store_true", help="record an MP4 per SCAN episode")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    grid = GridMap.load(paths.scene_dir(args.scene) / "occupancy.npz")
    episodes = generate_episodes(grid, args.episodes, args.seed)
    if not episodes:
        raise SystemExit("no episodes generated: is the scene's occupancy grid built?")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or paths.PROJECT_DIR / f"runs/scan_bench/{stamp}_{args.scene}"
    out_dir.mkdir(parents=True, exist_ok=True)
    media = out_dir / "media" if args.video else None

    modes = [False, True] if args.mode == "both" else [args.mode == "scan"]
    rows: list[Result] = []
    trails: dict = {}
    for ep in episodes:
        for planner in modes:
            if args.tier == "physics":
                policy = args.policy or os.environ.get("WOJTEK_POLICY") or paths.DEFAULT_POLICY
                res, trail = run_episode_physics(
                    args.scene, ep, planner, policy, media if planner else None
                )
            else:
                res, trail = run_episode(
                    args.scene, ep, planner, media if planner else None
                )
            rows.append(res)
            trails[(ep.idx, res.mode)] = trail
            logger.info(
                f"ep{ep.idx:02d} {res.mode:14s} success={res.success} "
                f"coll={res.collisions} off_floor={res.off_floor} "
                f"falls={res.falls} blocked={res.blocked} "
                f"path={res.path_m:.2f} dist={res.final_dist_m:.2f} "
                f"({res.wall_s}s)"
            )

    summary = summarize(rows)
    (out_dir / "scoreboard.json").write_text(json.dumps(
        {
            "scene": args.scene,
            "seed": args.seed,
            "goal_radius_m": GOAL_RADIUS,
            "summary": summary,
            "episodes": [asdict(e) for e in episodes],
            "rows": [asdict(r) for r in rows],
        },
        indent=2, default=list,
    ))
    trajectory_png(grid, episodes, trails, out_dir / "trajectories.png")
    logger.success(f"scoreboard -> {out_dir}")
    for mode, s in summary.items():
        logger.info(f"{mode:8s} {json.dumps(s)}")


if __name__ == "__main__":
    main()
