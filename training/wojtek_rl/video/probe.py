"""Render a checkpoint walking, with torque bars and the onboard depth view.

Run: ./run.sh video-probe --run runs/<name> --arena train --seconds 8
     ./run.sh video-probe --run runs/<name> --cell pyramid_stairs_5cm --vx 0.4

`--cell` films one cell of the measurement suite: the eval arena, the spawn
terrain-scan gives that cell's first course run, and the commanded speed held
at `--vx` so the robot walks out into the obstacle. Falling is a result, not a
failure of the tool -- the clip ends where the episode does.

Three overlays on one chase-camera frame:

* a bar per actuator against the env's effective torque limit, grouped per leg;
* the depth image from a camera placed at `height_scan.mask`'s mount, pitch
  and vertical field of view, so the render and the analytic mask share a
  frustum;
* the 25 scan points projected into that depth image, colored by
  `height_scan.visible_mask`. A run whose actor consumes `height_scan` also
  gets the held scan as a 5x5 heatmap.

A tool, not a measurement: nothing here reads or writes training config.
"""

import os
import sys

if sys.platform == "linux":  # headless GPU boxes; macOS uses its default GL
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import functools
from pathlib import Path

import numpy as np

from wojtek_rl import paths, terrain_scan, terrain_suite
from wojtek_rl.video import SceneView, write_video

HEIGHT_CMD = 0.125


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a checkpoint walking, with torque bars and the "
        "onboard depth view."
    )
    ap.add_argument("--run", default=None, help="required unless --list-cells")
    ap.add_argument(
        "--arena", choices=["flat", "train", "eval"], default=None,
        help="flat forces the flat scene; train/eval render on that terrain "
        "arena (terrain runs only). Default train, or eval with --cell",
    )
    ap.add_argument(
        "--cell", default=None,
        help="film one measurement cell of the eval arena, at the spawn "
        "terrain-scan uses for it (implies --arena eval)",
    )
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--vx", type=float, default=0.5, help="forward command, m/s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--scan", choices=["clean", "dark"], default="clean",
        help="dark feeds the actor a zero height scan, the blind fallback "
        "when the camera stops delivering",
    )
    ap.add_argument(
        "--camera", default="track",
        help="chase camera for the main view; terrain scenes also define "
        "track_far, higher and farther out",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--list-cells", action="store_true", help="print the cells and exit"
    )
    args = ap.parse_args()

    if args.list_cells:
        for line in terrain_scan.cell_lines():
            print(line)
        return
    if not args.run:
        ap.error("--run is required")
    if args.cell and args.arena:
        ap.error("--cell already selects the eval arena; drop --arena")
    cell = None
    if args.cell:
        cell = terrain_suite.CELLS_BY_NAME.get(args.cell)
        if cell is None:
            ap.error(f"unknown cell {args.cell!r}; --list-cells prints them all")
    arena = args.arena or ("eval" if cell else "train")

    import jax
    import jax.numpy as jp

    from wojtek_rl.battery import load_checkpoint_policy

    flat = arena == "flat"
    overrides = {"sim": {"backend": "jax", "num_envs": 1}}
    if not flat:
        # A pinned spawn pad: this renders one episode, and jitter only makes
        # it irreproducible.
        overrides["terrain"] = {
            "enable": True, "arena": arena, "pad_jitter": 0.0,
        }
    if cell is not None:
        overrides["terrain"]["spawn_yaw"] = False
        # The clip holds one command; the env's own resample would hand the
        # actor a different one halfway up the obstacle.
        overrides["command"] = {"resample_steps": 10**9}
    if args.scan == "dark":
        overrides["height_scan"] = {"dark": True}
    run, env, ckpt, inf = load_checkpoint_policy(
        Path(args.run), flat=flat, env_overrides=overrides
    )
    if cell is not None:
        terrain_scan.check_arena_of(env)
    if env._config.get("height_scan") is None:
        sys.exit(
            f"task {run.get('task')!r} has no height_scan config, so there is "
            "no mask geometry to place the depth camera at"
        )
    where = f"cell {cell.name}" if cell else f"arena {arena}"
    dark = "_dark" if args.scan == "dark" else ""
    out = Path(args.out) if args.out else (
        paths.PROJECT_DIR / "videos" / run["run_name"]
        / f"probe_{cell.name if cell else arena}{dark}.mp4"
    )
    print(f"scene: {env.xml_path}")

    # The heatmap shows what the actor is actually fed. _scan_live, not
    # _scan_enabled: the sample-and-hold buffers only exist on terrain.
    show_hold = env._scan_live and "height_scan" in env.actor_obs_names

    command = jp.array([args.vx, 0.0, 0.0, HEIGHT_CMD])
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(args.seed)
    if cell is None:
        state = jax.jit(env.reset)(rng)
    else:
        # Course run 0 of that cell: heading +x, the start offset furthest back
        # from the obstacle band, on the tile the scan measures.
        _, spawn, yaw, pad_h = terrain_scan.spawn_table(env, cell)
        state = jax.jit(functools.partial(terrain_scan.scan_reset, env))(
            rng, spawn[0], pad_h[0], yaw[0], command
        )
    n_steps = max(1, round(args.seconds / env.dt))
    every = max(1, round(1.0 / (args.fps * env.dt)))
    fps = 1.0 / (env.dt * every)

    frames, vels = [], []
    # No size: the showcase render is SceneView's own default.
    with SceneView(env, camera=args.camera, torque=True, onboard=True) as view:
        for i in range(n_steps):
            if "command" in state.info:
                state.info["command"] = command
            rng, act_rng = jax.random.split(rng)
            action, _ = inf(state.obs, act_rng)
            state = step(state, action)
            vels.append(float(env._local_linvel(state.data)[0]))
            if float(state.done):
                print(f"fell at step {i}")
                break
            if i % every:
                continue
            frames.append(
                view.frame(
                    state.data.qpos,
                    torque=state.data.actuator_force,
                    scan_hold=state.info["scan_hold"] if show_hold else None,
                    header=[
                        f"{run['run_name']} @ {ckpt.name}",
                        f"{where}   t {i * env.dt:5.2f} s",
                        f"cmd vx {args.vx:+.2f}   vx {vels[-1]:+.2f} m/s",
                    ],
                )
            )

    if not frames:
        sys.exit("no frames: the rollout ended before the first render")
    write_video(out, frames, fps=fps)
    print(
        f"commanded vx {args.vx:+.2f}  achieved forward vx {np.mean(vels):+.2f}"
    )
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
