"""Headless VLM navigation episode runner: no browser, full logging.

Runs one goal through the same RoomSim + VlmNavigator stack the room demo
uses, but pumps the 50 Hz control loop itself and records everything an
evaluation needs: per-decision JSONL (action, reasoning, pose), periodic
ego/chase frames, and a final summary with outcome + path length.

    ./run.sh nav-episode --goal "go to the bed" \
        --vlm-backend futurenav --vlm-url http://<gpu-host>:8100

Outputs land in runs/nav_episodes/<timestamp>_<backend>/ (gitignored).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from wojtek_rl.room_app import RoomSim  # noqa: F401  (sets MUJOCO_GL before mujoco)
from wojtek_rl import paths
from wojtek_rl.vlm_nav import VlmNavigator

PUMP_DT_S = 0.02          # 50 Hz, same cadence as the websocket loop
FRAME_EVERY_STEPS = 25    # 2 fps of saved frames


class RecordingClient:
    """Wraps any VlmClientProto; logs each decision with the current pose."""

    def __init__(self, inner, sim, out_dir: Path):
        self.inner = inner
        self.sim = sim
        self.log_path = out_dir / "decisions.jsonl"
        self.decisions = 0

    async def decide(self, goal, ego_b64, history, step, max_steps, pose):
        t0 = time.time()
        decision = await self.inner.decide(goal, ego_b64, history, step, max_steps, pose)
        record = {
            "step": step,
            "action": decision.action,
            "amount": decision.amount,
            "reasoning": decision.reasoning,
            "pose": list(pose),
            "latency_s": round(time.time() - t0, 2),
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self.decisions += 1
        logger.info(f"decision {step}: {decision.action} {decision.amount}")
        return decision


def build_client(args, sim, out_dir: Path):
    if args.vlm_backend == "futurenav":
        from wojtek_rl.futurenav_nav import (
            DEFAULT_FUTURENAV_URL,
            FUTURENAV_MAX_STEPS,
            FUTURENAV_TIMEOUT_S,
            FutureNavVlmClient,
        )

        client = FutureNavVlmClient(args.vlm_url or DEFAULT_FUTURENAV_URL)
        max_steps = args.max_steps or FUTURENAV_MAX_STEPS
        timeout_s = FUTURENAV_TIMEOUT_S
    else:
        from wojtek_rl.vlm_local import DEFAULT_LOCAL_MODEL, LOCAL_VLM_TIMEOUT_S, LocalVlmClient
        from wojtek_rl.vlm_nav import MAX_STEPS

        client = LocalVlmClient(args.vlm_model or DEFAULT_LOCAL_MODEL)
        max_steps = args.max_steps or MAX_STEPS
        timeout_s = LOCAL_VLM_TIMEOUT_S
    return RecordingClient(client, sim, out_dir), max_steps, timeout_s


async def run_episode(
    sim, nav, goal: str, out_dir: Path, goal_xy: tuple[float, float] | None = None,
    success_radius: float = 1.5,
) -> dict:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    x0, y0, _ = sim.pose()
    path_len = 0.0
    last = (x0, y0)
    tick = 0

    min_goal_dist = math.inf

    async def pump():
        nonlocal path_len, last, tick, min_goal_dist
        while True:
            sim.step()
            x, y, _ = sim.pose()
            path_len += math.hypot(x - last[0], y - last[1])
            last = (x, y)
            if goal_xy is not None:
                min_goal_dist = min(min_goal_dist, math.hypot(x - goal_xy[0], y - goal_xy[1]))
            if tick % FRAME_EVERY_STEPS == 0:
                chase_b64, ego_b64 = sim.render_pair()
                (frames_dir / f"{tick:06d}_chase.jpg").write_bytes(base64.b64decode(chase_b64))
                (frames_dir / f"{tick:06d}_ego.jpg").write_bytes(base64.b64decode(ego_b64))
            tick += 1
            await asyncio.sleep(PUMP_DT_S)

    t0 = time.time()
    pump_task = asyncio.create_task(pump())
    try:
        await nav._run(goal)
    finally:
        pump_task.cancel()

    x1, y1, yaw1 = sim.pose()
    status = nav.status()
    summary = {
        "goal": goal,
        "state": status["state"],
        "reason": status["reason"],
        "error": status["error"],
        "steps": status["step"],
        "max_steps": status["max_steps"],
        "duration_s": round(time.time() - t0, 1),
        "start_xy": [round(x0, 3), round(y0, 3)],
        "end_xy": [round(x1, 3), round(y1, 3)],
        "end_yaw_deg": round(math.degrees(yaw1), 1),
        "displacement_m": round(math.hypot(x1 - x0, y1 - y0), 3),
        "path_length_m": round(path_len, 3),
    }
    if goal_xy is not None:
        # VLN-CE-style outcome metrics, scaled to the room: SR (stopped within
        # the radius), oracle SR (came within the radius at any point), NE
        # (final distance to goal). Paper radius is 3 m on house-sized MP3D
        # scenes; this room is ~5 m across, hence the smaller default.
        end_dist = math.hypot(x1 - goal_xy[0], y1 - goal_xy[1])
        stopped = status["reason"] in ("vlm_done", "vlm_stop")
        summary.update({
            "goal_xy": [goal_xy[0], goal_xy[1]],
            "success_radius_m": success_radius,
            "dist_to_goal_m": round(end_dist, 3),
            "min_dist_to_goal_m": round(min_goal_dist, 3),
            "success": bool(stopped and end_dist <= success_radius),
            "oracle_success": bool(min_goal_dist <= success_radius),
        })
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--goal", required=True)
    p.add_argument("--vlm-backend", choices=("futurenav", "local"), default="futurenav")
    p.add_argument(
        "--vlm-cam",
        choices=("ego", "bench"),
        default=os.environ.get("VLM_CAM", "ego"),
        help="VLM viewpoint: ego = onboard ~10 cm cam; bench = VLN-CE-style 1.25 m cam",
    )
    p.add_argument(
        "--vlm-overlap",
        action="store_true",
        default=os.environ.get("VLM_OVERLAP", "") == "1",
        help="think about the next step while the current command executes (~2x rate)",
    )
    p.add_argument(
        "--goal-xy",
        default=None,
        help="ground-truth goal position 'X,Y' in world meters; enables SR/NE metrics",
    )
    p.add_argument(
        "--success-radius",
        type=float,
        default=1.5,
        help="success radius in meters around --goal-xy",
    )
    p.add_argument("--vlm-url", default=None, help="FutureNav action server URL")
    p.add_argument("--vlm-model", default=None)
    p.add_argument("--max-steps", type=int, default=None, help="override backend default")
    p.add_argument("--scene", type=Path, default=paths.ROOM_SCENE_XML)
    p.add_argument("--policy", type=Path, default=paths.POLICY_NPZ)
    p.add_argument("--out", type=Path, default=None, help="output dir (default runs/nav_episodes/<ts>)")
    args = p.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or (
        paths.PROJECT_DIR / "runs/nav_episodes" / f"{stamp}_{args.vlm_backend}_{args.vlm_cam}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = RoomSim(args.scene, args.policy, vlm_cam=args.vlm_cam)
    client, max_steps, timeout_s = build_client(args, sim, out_dir)
    is_futurenav = args.vlm_backend == "futurenav"
    from wojtek_rl.futurenav_nav import FUTURENAV_MAX_ROTATION
    nav = VlmNavigator(
        sim, client, max_steps=max_steps, vlm_timeout_s=timeout_s, overlap=args.vlm_overlap,
        vlnce_frame=is_futurenav,
        max_rotation=FUTURENAV_MAX_ROTATION if is_futurenav else None,
    )

    goal_xy = None
    if args.goal_xy:
        gx, gy = (float(v) for v in args.goal_xy.split(","))
        goal_xy = (gx, gy)

    summary = asyncio.run(
        run_episode(sim, nav, args.goal, out_dir, goal_xy=goal_xy,
                    success_radius=args.success_radius)
    )
    summary["backend"] = args.vlm_backend
    summary["vlm_cam"] = args.vlm_cam
    summary["overlap"] = args.vlm_overlap
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.success(f"episode {summary['state']} ({summary['reason']}): {json.dumps(summary)}")
    logger.info(f"artifacts: {out_dir}")


if __name__ == "__main__":
    main()
