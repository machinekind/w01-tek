"""Run the catalogue against a checkpoint: driver, plots, video, CLI table."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import jax
import numpy as np
from mujoco import mjx

from wojtek_rl.battery import load_checkpoint_policy
from wojtek_rl.courses.families import course_catalogue
from wojtek_rl.courses.follower import (
    GOAL_MIN_PROGRESS_M,
    GOAL_RADIUS_M,
    LOOKAHEAD_M,
    SPIN_ENTER_RAD,
    SPIN_EXIT_RAD,
    YAW_MAX,
)
from wojtek_rl.courses.rollout import course_rollout, friction_geom_ids, spin_rollout
from wojtek_rl.courses.scoring import (
    NOMINAL_HEIGHT_M,
    STANCE_HALFWIDTH_M,
    VIBRATION_CUTOFF_HZ,
    aggregate,
    seed_result,
    spin_seed_result,
)
from wojtek_rl.courses.spec import Course, SpinCourse
from wojtek_rl.video import SceneView, frame_size, write_video


def write_path_plot(out_png: Path, course: Course, path: np.ndarray, trails: list):
    """Overhead commanded-path-vs-actual plot, one per scenario."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Equal aspect is essential for judging path geometry, so size the figure
    # to the data instead: a 10 x 1 m course in a square figure is mostly
    # whitespace.
    box = np.concatenate([np.asarray(path)] + [t for t in trails if len(t)])
    span_x = max(float(np.ptp(box[:, 0])), 1.0)
    span_y = max(float(np.ptp(box[:, 1])), 1.0)
    width = 7.0
    height = float(np.clip(width * span_y / span_x + 1.4, 3.0, 9.0))

    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(path[:, 0], path[:, 1], "k--", lw=1.2, label="commanded path")
    for k, xy in enumerate(trails):
        if len(xy):
            ax.plot(xy[:, 0], xy[:, 1], lw=1.0, alpha=0.8, label=f"seed {k}")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{course.name} -- {course.isolates}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


class _NullCtx:
    """`with` wrapper yielding None, so the render path needs no branch."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def run_courses(
    run_dir: Path,
    seeds: int = 8,
    seed_base: int = 0,
    only: list[str] | None = None,
    video: bool = False,
    paths_plot: bool = False,
    out_dir: Path | None = None,
    video_size: tuple[int, int] = (640, 480),
    overlay_torque: bool = False,
    overlay_camera: bool = False,
) -> dict:
    """Run the course benchmark against `run_dir`'s latest checkpoint."""
    from wojtek_rl.build_model import FOOT_RADIUS

    run, env, ckpt, inf = load_checkpoint_policy(run_dir)
    if overlay_camera and env._config.get("height_scan") is None:
        raise SystemExit(
            f"task {run.get('task')!r} has no height_scan config, so there is "
            "no mask geometry to place the depth camera at"
        )
    foot_radius = FOOT_RADIUS
    catalogue = course_catalogue()
    names = only or list(catalogue)
    unknown = [n for n in names if n not in catalogue]
    if unknown:
        raise SystemExit(f"unknown course(s): {unknown}\nhave: {list(catalogue)}")

    floor_id = env.mj_model.geom("floor").id
    fric_geoms = friction_geom_ids(floor_id, env._foot_geom_ids)
    base_friction = env.mj_model.geom_friction[fric_geoms, 0].copy()
    # Equal-priority contacts take the max, so THIS is the nominal rows'
    # effective contact friction -- not the floor's value alone.
    effective_base = float(base_friction.max())
    out_dir = out_dir or (run_dir / "courses")

    results = {
        "run": run["run_name"],
        "checkpoint": ckpt.name,
        "seeds": seeds,
        "seed_base": seed_base,
        "follower": {
            "lookahead_m": LOOKAHEAD_M,
            "yaw_max": YAW_MAX,
            "spin_enter_rad": SPIN_ENTER_RAD,
            "spin_exit_rad": SPIN_EXIT_RAD,
            "goal_radius_m": GOAL_RADIUS_M,
            "goal_min_progress_m": GOAL_MIN_PROGRESS_M,
            "holonomic": False,
        },
        "normalizers": {
            "stance_halfwidth_m": STANCE_HALFWIDTH_M,
            "nominal_height_m": NOMINAL_HEIGHT_M,
            "vibration_cutoff_hz": VIBRATION_CUTOFF_HZ,
        },
        "courses": {},
    }

    def friction_key(n):
        f = getattr(catalogue[n], "friction", None)  # SpinCourse has none
        return -1.0 if f is None else f  # explicit: `or` would misread mu=0.0

    # Group by friction so the jitted step is rebuilt once per distinct
    # floor, not once per scenario: jax.jit closes over env._mjx_model, so a
    # friction change only takes effect through a fresh jit (and each one
    # costs a full trace + compile).
    current, reset, step = object(), None, None
    for name in sorted(names, key=friction_key):
        course = catalogue[name]
        want = getattr(course, "friction", None)  # None = model's own values
        if want != current:
            env.mj_model.geom_friction[fric_geoms, 0] = (
                base_friction if want is None else want
            )
            # Same in-place model swap battery.py's --alpha path uses.
            # env._make_data_fn keeps its old closure on purpose: it only
            # allocates, and no array shape changed here.
            env._mjx_model = mjx.put_model(env.mj_model, impl=env._backend)
            # Re-jit after the swap: the trace bakes the model in as
            # constants, so the previous executable still holds the old
            # friction. Verified end to end -- the slippery rows do move.
            reset, step = jax.jit(env.reset), jax.jit(env.step)
            current = want
        effective = effective_base if want is None else want

        trails, seed_rows = [], []
        course_path, course_length = None, 0.0
        # A machine with no usable GL (headless macOS, a container without
        # EGL) must not lose the whole benchmark over an optional video: warn,
        # drop video for the rest of the run, and keep the numbers, which are
        # the point. `video = False` also keeps this warning to one line.
        ctx = _NullCtx()
        if video:
            try:
                ctx = SceneView(
                    env, size=video_size, camera="track",
                    torque=overlay_torque, onboard=overlay_camera,
                )
            except Exception as exc:  # noqa: BLE001 -- any GL failure, same fallback
                print(f"warning: no video, renderer unavailable ({exc})")
                video = False
        with ctx as maybe_view:
            for s in range(seeds):
                # Video from the first iteration only: one clip per scenario
                # is the point, not eight of the same course.
                view = maybe_view if (video and s == 0) else None
                seed = seed_base + s
                if isinstance(course, SpinCourse):
                    rec, info = spin_rollout(
                        env, reset, step, inf, course, seed=seed, view=view,
                    )
                    seed_rows.append(spin_seed_result(rec, info, env.dt, course.wz))
                else:
                    rec, info = course_rollout(
                        env, reset, step, inf, course, foot_radius, seed=seed,
                        view=view,
                    )
                    seed_rows.append(seed_result(rec, info, env.dt))
                trails.append(rec.get("xy", np.empty((0, 2))))
                # A seed that fell while still standing never built a
                # Pursuit, so keep the first geometry we did get.
                if course_path is None and info.get("path") is not None:
                    course_path = info["path"]
                    course_length = info["total_length"]
                if view is not None and info["frames"]:
                    render_every = max(1, round(1 / (30 * env.dt)))
                    write_video(
                        out_dir / f"{name}.mp4",
                        info["frames"],
                        1.0 / (env.dt * render_every),
                    )

        entry = aggregate(seed_rows)
        entry["isolates"] = course.isolates
        entry["friction"] = effective
        entry["push_at_m"] = getattr(course, "push_at_m", None)
        entry["course_length_m"] = round(float(course_length), 3)
        results["courses"][name] = entry
        if paths_plot and course_path is not None:
            write_path_plot(out_dir / f"{name}_path.png", course, course_path, trails)
        print(_row(name, entry), flush=True)

    # Restore the model so a caller reusing `env` is not left on a slippery
    # floor (run_courses mutates it in place, as battery's alpha path does).
    env.mj_model.geom_friction[fric_geoms, 0] = base_friction
    # Scenarios ran grouped by friction; report them in catalogue order so
    # the table reads geometry -> speed -> floor -> disturbance.
    results["courses"] = {
        n: results["courses"][n] for n in catalogue if n in results["courses"]
    }
    return results


def _row(name, entry) -> str:
    b = entry.get("binding", "-")
    return (
        f"{name:<22} score {entry['score_median']:>8.2f} (worst "
        f"{entry['score_worst']:>8.2f})  bound by {b:<11} "
        f"falls {entry['falls']}/{entry['seeds']}  "
        f"done {entry['completed']}/{entry['seeds']}"
    )


def print_table(results: dict) -> None:
    print(f"\ncourses -- {results['run']} @ {results['checkpoint']} "
          f"({results['seeds']} seeds)")
    print(f"{'scenario':<22} {'median':>8} {'worst':>8}  {'binding':<11} "
          f"{'xte_rms':>8} {'v_err':>7} {'falls':>6} {'done':>6}")
    for name, e in results["courses"].items():
        raw = e.get("raw_median") or {}
        print(
            f"{name:<22} {e['score_median']:>8.2f} {e['score_worst']:>8.2f}  "
            f"{e.get('binding', '-'):<11} "
            f"{raw.get('xte_rms_m', float('nan')):>8.3f} "
            f"{raw.get('speed_err_rms', float('nan')):>7.3f} "
            f"{e['falls']:>3}/{e['seeds']} "
            f"{e['completed']:>3}/{e['seeds']}"
        )


def main():
    ap = argparse.ArgumentParser(
        description="Path-following course benchmark; see wojtek_rl.courses",
    )
    ap.add_argument("--run", help="run directory (not needed with --list)")
    ap.add_argument("--out", default=None, help="default: <run>/courses.json")
    ap.add_argument(
        "--seeds", type=int, default=8,
        help="rollouts per scenario (default 8); median and worst reported",
    )
    ap.add_argument(
        "--seed-base", type=int, default=0,
        help="offset added to every rollout seed (default 0); courses run "
        "CPU-deterministic, so a second base is the only way to get a "
        "replicate that isn't bit-identical to the first",
    )
    ap.add_argument(
        "--only", nargs="+", default=None,
        help="run just these scenarios (names from course_catalogue)",
    )
    ap.add_argument(
        "--video", action="store_true",
        help="render one mp4 per scenario (seed 0) into <run>/courses/",
    )
    ap.add_argument(
        "--video-size", type=frame_size, default=(640, 480), metavar="WxH",
        help="rendered frame size (default 640x480)",
    )
    ap.add_argument(
        "--overlay-torque", action="store_true",
        help="draw the per-actuator torque bars into the frame, against the "
        "env's effective limit",
    )
    ap.add_argument(
        "--overlay-camera", action="store_true",
        help="draw the onboard depth view as an inset, with the height-scan "
        "points projected onto it",
    )
    ap.add_argument(
        "--paths", action="store_true",
        help="write one overhead commanded-vs-actual PNG per scenario",
    )
    ap.add_argument(
        "--list", action="store_true", help="print the catalogue and exit"
    )
    args = ap.parse_args()

    if args.list:
        print(f"{'scenario':<22} {'length':>7} {'speed':>12}  isolates")
        for name, c in course_catalogue().items():
            if isinstance(c, SpinCourse):
                print(f"{name:<22} {c.turns:>4.1f} rev {abs(c.wz):>8g} rad/s"
                      f"  {c.isolates}")
                continue
            wp = np.asarray(c.waypoints, dtype=float)
            length = float(np.linalg.norm(np.diff(wp, axis=0), axis=1).sum())
            spd = np.unique(c.segment_speeds)
            speed = "/".join(f"{v:g}" for v in spd)
            extra = []
            if c.friction is not None:
                extra.append(f"mu={c.friction:g}")
            if c.push_at_m is not None:
                extra.append(f"push@{c.push_at_m:g}m")
            tail = f"{c.isolates}" + (f"  [{', '.join(extra)}]" if extra else "")
            print(f"{name:<22} {length:>5.1f} m {speed:>10} m/s  {tail}")
        return

    if not args.run:
        raise SystemExit("--run is required (or use --list)")
    run_dir = Path(args.run)
    results = run_courses(
        run_dir,
        seeds=args.seeds,
        seed_base=args.seed_base,
        only=args.only,
        video=args.video,
        paths_plot=args.paths,
        video_size=args.video_size,
        overlay_torque=args.overlay_torque,
        overlay_camera=args.overlay_camera,
    )
    print_table(results)
    out = Path(args.out) if args.out else run_dir / "courses.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    stamped = dict(results, timestamp=datetime.now().isoformat(timespec="seconds"))
    out.write_text(json.dumps(stamped, indent=2))
    print(f"\nwrote {out}")
