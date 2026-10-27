"""Async evaluation runner: episodes x (KinematicSim + EvalNavigator + VLM).

Episodes run concurrently (default 4) against one shared OpenAIVlmClient so
the vLLM server batches decisions across episodes. Spoken episodes go through
the hearing chain FIRST (say -> faster-whisper on this machine), then the
transcript -- not the clean text -- becomes the VLM goal.

Everything lands in --out: episodes.json (the generated suite + audio),
results.jsonl (one scored row per episode, appended as they finish, so a
killed run keeps its partial results), summary.json, and frames/<episode>/
composite VLM frames + final agent map for the media pipeline.

Usage:
    python -m wojtek_eval.runner --base-url http://HOST:PORT --model MODEL \
        --scenes room apartment --per-task 6 --spoken-frac 0.5 \
        --out runs/nav_eval/night1 [--sample-frac 0.4] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

from wojtek_rl import paths
from wojtek_eval.episodes import Episode, generate, score_episode, summarize
from wojtek_eval.gridmap import GridMap


async def _run_episode(ep: Episode, client, out_dir: Path, save_frames: bool,
                       sim_kwargs: dict | None = None) -> dict:
    from wojtek_eval.kinsim import KinematicSim
    from wojtek_eval.navigator import EvalNavigator

    sim = KinematicSim(ep.scene, start=ep.start, **(sim_kwargs or {}))
    if save_frames:
        sim.frame_dir = out_dir / "frames" / ep.id
    goal = (ep.audio or {}).get("transcript") if ep.spoken else ep.instruction
    goal = (goal or ep.instruction).strip()
    nav = EvalNavigator(sim, client, max_steps=ep.max_steps, poll_s=0.01,
                        vlm_timeout_s=180.0)
    t0 = time.monotonic()
    nav.start(goal)
    try:
        while nav.running:
            await asyncio.sleep(0.05)
    finally:
        nav.cancel("runner cleanup") if nav.running else None
    status = nav.status()
    failures = sum(1 for h in getattr(nav, "history", []) if "failed" in h["result"])
    if save_frames:
        from PIL import Image

        sim.frame_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(sim.omap.map_image(sim.pose(), px=340)).save(
            sim.frame_dir / "final_map.png"
        )
    result = {
        "state": status["state"], "reason": status.get("reason"),
        "final_pose": sim.pose(), "path_length": sim.path_length,
        "trail": [(round(x, 2), round(y, 2)) for x, y in sim.omap.trail],
        "steps": status["step"], "blocked": sim.executor.blocked,
        # What the local planner is supposed to change: steps into an
        # obstacle (off_floor is the scan's own missing floor, not a hit).
        "collisions": sim.collisions, "off_floor": sim.off_floor,
        "failures": failures, "wall_s": time.monotonic() - t0,
        "history": getattr(nav, "history", []),
    }
    sim.close()
    return result


async def run_suite(episodes: list[Episode], client, out_dir: Path,
                    concurrency: int, frames_every: int,
                    sim_kwargs: dict | None = None) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    grids = {s: GridMap.load(paths.scene_dir(s) / "occupancy.npz")
             for s in {e.scene for e in episodes}}
    results_path = out_dir / "results.jsonl"
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict] = []
    done_n = 0

    async def one(i: int, ep: Episode):
        nonlocal done_n
        async with sem:
            try:
                res = await _run_episode(ep, client, out_dir,
                                         save_frames=(i % frames_every == 0),
                                         sim_kwargs=sim_kwargs)
                row = score_episode(ep, res, grids[ep.scene])
                row["history"] = res["history"]
                row["trail"] = res["trail"]
            except Exception as e:  # one broken episode must not kill the night
                logger.exception(f"{ep.id} crashed")
                row = {"id": ep.id, "scene": ep.scene, "task": ep.task,
                       "spoken": ep.spoken, "instruction": ep.instruction,
                       "success": False, "error": str(e)[:300]}
            rows.append(row)
            done_n += 1
            with results_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            logger.info(
                f"[{done_n}/{len(episodes)}] {ep.id} "
                f"{'OK ' if row.get('success') else 'no '}"
                f"(steps={row.get('steps')}, dtg={row.get('dtg')})"
            )

    await asyncio.gather(*(one(i, ep) for i, ep in enumerate(episodes)))
    return rows


def prepare_audio(episodes: list[Episode], out_dir: Path, seed: int) -> None:
    """Synthesize + transcribe spoken instructions up front (Mac CPU work,
    keeps GPU time pure VLM)."""
    from wojtek_eval.hearing import Transcriber, hear_instruction

    rng = np.random.default_rng(seed)
    tr = Transcriber("small")
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        if ep.spoken:
            ep.audio = hear_instruction(ep.instruction, audio_dir, tr, rng)
            logger.info(f"{ep.id}: heard {ep.audio['transcript']!r} (wer {ep.audio['wer']:.2f})")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("openai", "futurenav"), default="openai",
                   help="openai = vLLM-served chat VLM; futurenav = FutureNav-4B "
                   "action server (wojtek_rl/futurenav_server)")
    p.add_argument("--base-url", default=None, help="openai backend: vLLM URL")
    p.add_argument("--model", default=None, help="openai backend: model id")
    p.add_argument("--vlm-url", default=None, help="futurenav backend: action server URL")
    p.add_argument("--vlm-cam", choices=("ego", "bench"), default="ego",
                   help="camera the VLM sees (bench = VLN-CE-style 1.25 m mast)")
    p.add_argument(
        "--no-local-planner",
        action="store_true",
        help="execute forward commands as straight marches instead of routing "
        "them through the SCAN local planner (pre-SCAN baseline)",
    )
    p.add_argument("--no-hud", action="store_true",
                   help="clean frames without the minimap HUD (futurenav never saw HUDs)")
    p.add_argument(
        "--suite",
        type=Path,
        default=None,
        help="reuse an episodes.json from a previous run verbatim instead of "
        "generating (the only way two scoreboards are comparable)",
    )
    p.add_argument("--scenes", nargs="+", default=["room", "apartment"])
    p.add_argument("--per-task", type=int, default=6, help="episodes per task type per scene")
    p.add_argument("--sample-frac", type=float, default=1.0,
                   help="random fraction of the generated suite to actually run")
    p.add_argument("--spoken-frac", type=float, default=0.5)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.5,
                   help="greedy decoding loops on near-identical frames; a bit "
                   "of sampling keeps the policy from spinning in place")
    p.add_argument("--frames-every", type=int, default=3, help="save frames for every Nth episode")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true", help="generate + audio only, no VLM")
    args = p.parse_args(argv)

    episodes: list[Episode] = []
    if args.suite:
        from wojtek_eval.episodes import episodes_from_json

        episodes = episodes_from_json(args.suite.read_text())
        logger.info(f"loaded {len(episodes)} episodes verbatim from {args.suite}")
    else:
        for scene in args.scenes:
            episodes += generate(scene, args.per_task, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    if args.sample_frac < 1.0:
        k = max(1, int(len(episodes) * args.sample_frac))
        episodes = list(rng.choice(episodes, size=k, replace=False))
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    if args.suite:
        spoken_idx = np.array([e.spoken for e in episodes])
    else:
        spoken_idx = rng.random(len(episodes)) < args.spoken_frac
        for ep, sp in zip(episodes, spoken_idx):
            ep.spoken = bool(sp)
    logger.info(f"{len(episodes)} episodes ({int(spoken_idx.sum())} spoken) "
                f"across {args.scenes}")

    args.out.mkdir(parents=True, exist_ok=True)
    prepare_audio(episodes, args.out, args.seed)
    from wojtek_eval.episodes import episodes_to_json

    (args.out / "episodes.json").write_text(episodes_to_json(episodes))
    if args.dry_run:
        logger.info("dry run: suite + audio written, exiting before VLM")
        return

    import os

    sim_kwargs = {
        "vlm_cam": args.vlm_cam,
        "hud": not args.no_hud,
        "local_planner": not args.no_local_planner,
    }

    if args.backend == "futurenav":
        from wojtek_rl.futurenav_nav import DEFAULT_FUTURENAV_URL, FutureNavVlmClient

        client = FutureNavVlmClient(args.vlm_url or DEFAULT_FUTURENAV_URL)
        if args.concurrency != 1:
            # The action server holds one episode's state (frame history +
            # VGGT cache); interleaved episodes would silently corrupt it.
            logger.warning("futurenav backend is single-episode; forcing concurrency=1")
            args.concurrency = 1
    else:
        if not (args.base_url and args.model):
            p.error("--base-url and --model are required for the openai backend")
        from wojtek_eval.vlm_openai import OpenAIVlmClient

        client = OpenAIVlmClient(
            args.base_url, args.model, api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            temperature=args.temperature,
        )

    async def _main():
        try:
            rows = await run_suite(episodes, client, args.out,
                                   args.concurrency, args.frames_every,
                                   sim_kwargs=sim_kwargs)
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                await close()
        return rows

    rows = asyncio.run(_main())
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("summary:\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
