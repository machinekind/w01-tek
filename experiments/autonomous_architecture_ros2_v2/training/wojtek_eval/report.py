"""Turn a nav-eval run directory into human-facing artifacts: per-episode
GIFs from the saved VLM frames, trail-over-scene maps, and a markdown
scoreboard. The HTML sponsor page is assembled elsewhere from these pieces.

Usage:
    python -m wojtek_eval.report --run runs/nav_eval/night1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wojtek_rl import paths
from wojtek_eval.gridmap import GridMap


def episode_gif(frames_dir: Path, out: Path, ms_per_frame: int = 900) -> Path | None:
    from PIL import Image

    frames = sorted(frames_dir.glob("step_*.jpg"))
    if not frames:
        return None
    imgs = [Image.open(f) for f in frames]
    imgs[0].save(
        out, save_all=True, append_images=imgs[1:], duration=ms_per_frame, loop=0
    )
    return out


def trail_png(row: dict, grid: GridMap, out: Path, px: int = 420) -> Path:
    """Oracle map + episode trail + start/goal markers -- the judge's view."""
    from PIL import Image, ImageDraw

    H, W = grid.shape
    img = np.full((H, W, 3), 46, np.uint8)
    img[grid.free] = (228, 228, 228)
    img[grid.occ] = (160, 82, 72)
    im = Image.fromarray(img[::-1])
    scale = px / max(H, W)
    im = im.resize((int(W * scale), int(H * scale)), Image.NEAREST)
    dr = ImageDraw.Draw(im)

    def w2p(x, y):
        j = (x - grid.origin[0]) / grid.res * scale
        i = (H - (y - grid.origin[1]) / grid.res) * scale
        return (j, i)

    trail = row.get("trail") or []
    if len(trail) > 1:
        dr.line([w2p(x, y) for x, y in trail], fill=(50, 140, 60), width=3)
    if trail:
        x, y = trail[0]
        dr.ellipse([*np.subtract(w2p(x, y), 6), *np.add(w2p(x, y), 6)],
                   outline=(30, 90, 200), width=3)
    for g in row.get("goals", []):
        gx, gy = g["pos"]
        r = g["radius"] / grid.res * scale
        cx, cy = w2p(gx, gy)
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(220, 170, 40), width=3)
    ok = row.get("success")
    dr.text((8, 6), f"{row['id']}  {'SUCCESS' if ok else 'fail'}",
            fill=(20, 120, 40) if ok else (170, 40, 40))
    im.save(out)
    return out


def markdown_report(run_dir: Path) -> str:
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]

    def fmt(v):
        return "--" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))

    lines = [
        "# Wojtek navigation eval",
        "",
        f"Run: `{run_dir}`  |  episodes: {len(rows)}",
        "",
        "| slice | n | SR | oracle SR | SPL | SoftSPL | DTG (m) | steps | WER |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summary.items():
        lines.append(
            f"| {name} | {s['n']} | {fmt(s['sr'])} | {fmt(s['oracle_sr'])} | "
            f"{fmt(s['spl'])} | {fmt(s['softspl'])} | {fmt(s['dtg'])} | "
            f"{fmt(s['steps'])} | {fmt(s['wer'])} |"
        )
    lines += ["", "## Episodes", ""]
    for r in sorted(rows, key=lambda r: r["id"]):
        mark = "+" if r.get("success") else "-"
        heard = f" (heard: {r['heard']!r})" if r.get("heard") else ""
        lines.append(
            f"- [{mark}] `{r['id']}` \"{r['instruction']}\"{heard} -> "
            f"{r.get('end_state')} dtg={fmt(r.get('dtg'))} steps={r.get('steps')}"
        )
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--gifs", action="store_true", default=True)
    args = p.parse_args(argv)
    run_dir = args.run

    episodes = {e["id"]: e for e in json.loads((run_dir / "episodes.json").read_text())}
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    grids = {}
    media = run_dir / "media"
    media.mkdir(exist_ok=True)
    for row in rows:
        ep = episodes.get(row["id"], {})
        row["goals"] = ep.get("goals", [])
        scene = row["scene"]
        if scene not in grids:
            grids[scene] = GridMap.load(paths.scene_dir(scene) / "occupancy.npz")
        if row.get("trail"):
            trail_png(row, grids[scene], media / f"{row['id']}_trail.png")
        fdir = run_dir / "frames" / row["id"]
        if fdir.exists():
            episode_gif(fdir, media / f"{row['id']}.gif")
    report = markdown_report(run_dir)
    (run_dir / "report.md").write_text(report)
    print(f"wrote {run_dir}/report.md and {media}/ ({len(list(media.iterdir()))} files)")


if __name__ == "__main__":
    main()
