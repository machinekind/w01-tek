"""Score a checkpoint on the fixed terrain measurement course.

Run:
    ./run.sh terrain-scan --run runs/<name>
    ./run.sh terrain-scan --run runs/<name> \
        --baseline <HF_ORGANIZATION>/wojtek-terrain-v1

Writes ``runs/<name>/terrain_scan.json``. ``report`` reads that file instead of
recomputing it, which is how the flat battery already feeds ``report`` -- so a
scan can run on the cluster and a laptop can render the report from it.

What a run is (see terrain_suite for the numbers): the robot spawns at a tile
centre on a fixed heading, walks out until the base is OUT_RADIUS from the
centre -- one crossing -- then the commanded forward speed flips sign and it
walks back to within BACK_RADIUS, which is the second. Four crossings. A run
passes when all four finish inside the step budget with no fall.

Crossings, not distance walked: a stair tile is 3 m across and its treads only
occupy the band from 0.60 m to 1.25 m from the centre, so "half the commanded
distance" can be satisfied on the flat pad without ever meeting the obstacle.
Counting crossings also gives a failure a reason -- the robot either fell or
never reached OUT_RADIUS, and on a stair those are different problems.

The speed reverses rather than the robot turning. A 180 degree turn on a 13 cm
tread might be the hardest thing in the suite, and the number would then measure
turning instead of climbing. Every real preset trains forward speed from -0.8 to
1.2, so walking backwards is inside the trained command box.

Nothing is sampled. Eight headings, four start offsets, three speeds, one fixed
arena: two scans of one checkpoint return the same numbers.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from wojtek_rl import terrain, terrain_suite

# Fraction of the actuator torque cap that counts as saturated. The flat
# battery's number (battery.saturation_fractions), so the two are comparable.
SATURATION_FRAC = 0.85
# Commanded standing height, the same anchor value the flat battery pins. A run
# whose trained command box excludes it is flagged, not silently measured.
COMMAND_HEIGHT = 0.125
# A cell may not drop more than this many percentage points against the
# baseline keeper.
RELATIVE_DROP_LIMIT = 10.0
SCAN_SCHEMA = 1


# -- pure helpers: no jax, no checkpoint, unit-tested directly -----------------


def leg_sign(crossings):
    """+1 while walking out, -1 while walking back.

    The commanded forward speed carries the sign; the heading never changes. A
    180 degree turn on a 13 cm tread might be the hardest thing in the suite,
    and the number would then measure turning instead of climbing.
    """
    return 1.0 - 2.0 * (crossings % 2)


def crossing_progress(crossings, radius, running, xp=np):
    """Advance the crossing counter by at most one.

    An outbound leg finishes at OUT_RADIUS from the tile centre, an inbound leg
    at BACK_RADIUS. A run that has already fallen or finished never advances
    again, and the count stops at CROSSINGS. Pure, so the rule is testable
    without physics; `xp` is numpy or jax.numpy.
    """
    outbound = (crossings % 2) == 0
    leg_done = xp.where(
        outbound, radius >= terrain_suite.OUT_RADIUS, radius <= terrain_suite.BACK_RADIUS
    )
    return xp.minimum(crossings + (leg_done & running), terrain_suite.CROSSINGS)


def fall_progress(fell, done, running):
    """Sticky fall flag, recorded only while a run is still on its course.

    A run that has finished its fourth crossing keeps being stepped: the batch is
    one program over 32 environments and there is no way to drop one of them
    mid-loop. Its command still says "walk out", so it walks on, and without this
    gate a robot that falls over a hundred steps after completing the course
    would turn its own pass into a fall. The run is the four crossings; what
    happens after them is not the measurement.
    """
    return fell | (running & done)


@dataclass(frozen=True)
class CellResult:
    """One cell at one commanded speed, over RUNS_PER_CELL_SPEED fixed runs."""

    passed: int
    of: int
    falls: int
    timeouts: int
    crossings_mean: float
    saturation: float
    track_err: float
    clearance: float
    nacon_max: int
    steps: int


def cell_entry(cell, speed: float, result: CellResult) -> dict:
    """The JSON shape for one (cell, speed) pair, bar and provenance included."""
    bar, provenance = terrain_suite.threshold(cell, speed)
    return {
        "passed": result.passed,
        "of": result.of,
        "bar": bar,
        "bar_fraction": cell.bar,
        "provenance": provenance,
        "falls": result.falls,
        "timeouts": result.timeouts,
        "crossings_mean": round(result.crossings_mean, 3),
        "saturation": round(result.saturation, 4),
        "track_err": round(result.track_err, 4),
        "clearance": round(result.clearance, 4),
        "nacon_max": result.nacon_max,
        "steps": result.steps,
    }


def absolute_gate(cells: dict) -> dict:
    """Every gated cell against its bar. Tracked cells are reported, not gated."""
    failures = []
    checked = 0
    for name, per_speed in cells.items():
        for speed, r in per_speed.items():
            if r.get("bar") is None:
                continue
            checked += 1
            if r["passed"] < r["bar"]:
                failures.append(
                    {
                        "cell": name,
                        "speed": speed,
                        "passed": r["passed"],
                        "bar": r["bar"],
                        "provenance": r["provenance"],
                    }
                )
    return {
        "verdict": "fail" if failures else "pass",
        "checked": checked,
        "failures": failures,
    }


def relative_gate(scan: dict, baseline: dict | None) -> dict:
    """No cell drops more than RELATIVE_DROP_LIMIT points against the baseline.

    Refuses rather than reports when the two are not comparable:

    - a different arena, because scores from two terrains are not the same
      measurement;
    - a different engine, until the cross-engine spread is measured. Warp on GPU
      is float32 with a contact budget and CPU MuJoCo is float64 with none, so
      an engine gap and a precision gap are currently indistinguishable.

    A cell the baseline never measured counts as nothing to compare against, not
    as a failure -- otherwise every newly added cell fails its own first gate.
    """
    if baseline is None:
        return {"verdict": "no baseline", "notes": [
            "no --baseline given; the absolute bars are the only gate"
        ]}
    if baseline.get("arena") != scan.get("arena"):
        return {
            "verdict": "refused",
            "notes": [
                "baseline arena differs from this scan's; scores from two "
                f"terrains are not comparable ({baseline.get('arena')} vs "
                f"{scan.get('arena')})"
            ],
        }
    if baseline.get("engine") != scan.get("engine"):
        return {
            "verdict": "refused",
            "notes": [
                f"baseline engine {baseline.get('engine')!r} differs from "
                f"{scan.get('engine')!r}; the cross-engine spread has not been "
                "measured, so a drop cannot be attributed"
            ],
        }
    drops, unmatched = [], []
    base_cells = baseline.get("cells") or {}
    for name, per_speed in (scan.get("cells") or {}).items():
        for speed, r in per_speed.items():
            b = (base_cells.get(name) or {}).get(speed)
            if not b:
                unmatched.append(f"{name}@{speed}")
                continue
            now = 100.0 * r["passed"] / max(r["of"], 1)
            was = 100.0 * b["passed"] / max(b["of"], 1)
            if was - now > RELATIVE_DROP_LIMIT:
                drops.append(
                    {
                        "cell": name,
                        "speed": speed,
                        "was": round(was, 1),
                        "now": round(now, 1),
                        "drop": round(was - now, 1),
                    }
                )
    notes = [f"baseline: {baseline.get('run', '?')} @ {baseline.get('checkpoint', '?')}"]
    if unmatched:
        notes.append(
            f"{len(unmatched)} cell/speed pairs are new since the baseline and "
            "have nothing to compare against"
        )
    return {
        "verdict": "fail" if drops else "pass",
        "limit_points": RELATIVE_DROP_LIMIT,
        "drops": drops,
        "unmatched": unmatched,
        "notes": notes,
    }


def load_baseline(ref: str | None) -> dict | None:
    """A baseline scan from a local path or a Hugging Face reference.

    The baseline is an input, published with the keeper it came from, not a file
    this repository keeps: a best-ever number held in the repo hides which run
    set the bar, so a rejected policy could leave a bar behind that nobody can
    trace. `ref` is a path to a scan JSON, a directory holding one, or
    `org/name[@revision]`.
    """
    if not ref:
        return None
    path = Path(ref)
    if path.is_dir():
        path = path / "terrain_scan.json"
    if path.exists():
        return json.loads(path.read_text())
    from huggingface_hub import hf_hub_download

    repo_id, _, revision = ref.partition("@")
    local = hf_hub_download(repo_id, "terrain_scan.json", revision=revision or None)
    return json.loads(Path(local).read_text())


def command_box_warnings(run: dict, speeds=terrain_suite.SPEEDS) -> list[str]:
    """Commanded speeds and heights this run never trained on.

    Extrapolation is flagged, not silently measured. The code default for
    forward speed is +-0.6, so 0.7 is outside it -- every real preset trains
    -0.8 to 1.2, so this normally stays empty."""
    cmd = (run.get("env_config") or {}).get("command") or {}
    warnings = []
    vx = cmd.get("vx")
    if vx:
        for speed in speeds:
            if speed > vx[1] or -speed < vx[0]:
                warnings.append(
                    f"commanded vx +-{speed} is outside the trained box "
                    f"[{vx[0]}, {vx[1]}]; that cell is extrapolation"
                )
    height = cmd.get("height")
    if height and not (height[0] <= COMMAND_HEIGHT <= height[1]):
        warnings.append(
            f"commanded height {COMMAND_HEIGHT} is outside the trained box "
            f"[{height[0]}, {height[1]}]"
        )
    return warnings


# -- the rollout ---------------------------------------------------------------


def _spawn_table(env, cell) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(centre, spawn_xy, yaw, pad_height) for one cell's 32 runs.

    Offsets run along the heading, so what varies is where in the tread cycle
    the robot first meets the obstacle."""
    type_index = terrain.TYPES.index(cell.terrain_type)
    centre = np.asarray(env._terrain_origin_xy)[cell.row, type_index]
    pad_h = float(np.asarray(env._terrain_pad_h)[cell.row, type_index])
    yaw = np.array([r.yaw for r in terrain_suite.COURSE], dtype=np.float32)
    offset = np.array([r.offset for r in terrain_suite.COURSE], dtype=np.float32)
    heading = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
    spawn = centre[None, :] + heading * offset[:, None]
    centres = np.repeat(centre[None, :], len(terrain_suite.COURSE), axis=0)
    return (
        centres.astype(np.float32),
        spawn.astype(np.float32),
        yaw,
        np.full(len(terrain_suite.COURSE), pad_h, dtype=np.float32),
    )


def make_cell_runner(env, inf):
    """A jitted "score one cell at one speed" function.

    Batch shape is fixed at RUNS_PER_CELL_SPEED and the step budget is the only
    static argument, so exactly three programs get compiled -- one per commanded
    speed -- and each is dispatched once per cell. That is a requirement, not an
    optimization: warp allocates its contact pool when the data is created,
    sized by the number of environments, so a varying batch size means a new
    allocation. Which tile, which heading and which direction are numbers fed
    into the same program.
    """
    import functools

    import jax
    import jax.numpy as jp
    from mujoco import mjx

    n = terrain_suite.RUNS_PER_CELL_SPEED
    torque_cap = float(np.asarray(env.mj_model.actuator_forcerange[:, 1]).max())
    nv = env.mj_model.nv

    def reset_one(rng, spawn_xy, pad_h, yaw, height):
        """env.reset, then the deterministic pose written into qpos.

        Safe for the reason the training wrapper's teleport is safe: the next
        physics step recomputes kinematics from qpos, and no observation carries
        world position or heading. The joints are pinned to the height anchor
        with none of reset's noise, so the course is fully deterministic.
        """
        state = env.reset(rng)
        anchor = env._height_ctrl(height)
        qpos = env._home_qpos.at[env._qadr].set(anchor)
        qpos = qpos.at[0:2].set(spawn_xy)
        qpos = qpos.at[2].set(pad_h + height)
        qpos = qpos.at[3:7].set(
            jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
        )
        data = state.data.replace(qpos=qpos, qvel=jp.zeros(nv), ctrl=anchor)
        data = mjx.forward(env.mjx_model, data)
        info = dict(state.info)
        info["motor_targets"] = anchor
        return state.replace(data=data, obs=env._get_obs(data, info), info=info)

    def nacon_of(data):
        """Warp's live active-contact count, 0 on the jax backend (which has no
        contact budget to overflow)."""
        value = getattr(data._impl, "nacon", None)
        if value is None:
            return jp.zeros((), jp.int32)
        return jp.max(jp.asarray(value)).astype(jp.int32)

    @functools.partial(jax.jit, static_argnames=("budget",))
    def run(rng, centre, spawn_xy, pad_h, yaw, speed, height, budget):
        state = jax.vmap(reset_one, in_axes=(0, 0, 0, 0, None))(
            jax.random.split(rng, n), spawn_xy, pad_h, yaw, height
        )
        zeros = jp.zeros(n)

        def body(carry):
            i, state, rng, crossings, fell, sat, err, clr, counted, nacon = carry
            sign = leg_sign(crossings)
            command = jp.stack(
                [sign * speed, zeros, zeros, jp.full(n, height)], axis=-1
            )
            info = dict(state.info)
            info["command"] = command
            state = state.replace(info=info)
            rng, sub = jax.random.split(rng)
            action, _ = inf(state.obs, sub)
            was_running = ~(fell | (crossings >= terrain_suite.CROSSINGS))
            state = jax.vmap(env.step)(state, action)
            data = state.data

            radius = jp.linalg.norm(data.qpos[:, 0:2] - centre, axis=-1)
            crossings = crossing_progress(crossings, radius, was_running, xp=jp)
            fell = fall_progress(fell, state.done > 0.5, was_running)

            # Metrics skip the settle window and stop once a run is over.
            live = was_running & (i >= terrain_suite.SETTLE_STEPS)
            w = live.astype(jp.float32)
            force = jp.abs(data.actuator_force)
            sat = sat + w * jp.mean(
                (force > SATURATION_FRAC * torque_cap).astype(jp.float32), axis=-1
            )
            forward = jax.vmap(env._local_linvel)(data)[:, 0]
            err = err + w * jp.abs(sign * speed - forward)
            clr = clr + w * jp.mean(jax.vmap(env._foot_clearance)(data), axis=-1)
            return (
                i + 1, state, rng, crossings, fell, sat, err, clr,
                counted + w, jp.maximum(nacon, nacon_of(data)),
            )

        def cond(carry):
            i, _, _, crossings, fell, *_ = carry
            running = ~(fell | (crossings >= terrain_suite.CROSSINGS))
            return (i < budget) & jp.any(running)

        init = (
            jp.zeros((), jp.int32), state, rng,
            jp.zeros(n, jp.int32), jp.zeros(n, bool),
            zeros, zeros, zeros, zeros, jp.zeros((), jp.int32),
        )
        i, state, _, crossings, fell, sat, err, clr, counted, nacon = (
            jax.lax.while_loop(cond, body, init)
        )
        per_step = jp.maximum(counted, 1.0)
        return {
            "crossings": crossings,
            "fell": fell,
            "saturation": sat / per_step,
            "track_err": err / per_step,
            "clearance": clr / per_step,
            "counted": counted,
            "steps": i,
            "nacon_max": nacon,
        }

    return run


def reduce_runs(out) -> CellResult:
    """One cell's 32 run outcomes into its recorded numbers."""
    crossings = np.asarray(out["crossings"])
    fell = np.asarray(out["fell"], dtype=bool)
    finished = crossings >= terrain_suite.CROSSINGS
    return CellResult(
        passed=int((finished & ~fell).sum()),
        of=len(crossings),
        falls=int(fell.sum()),
        timeouts=int((~finished & ~fell).sum()),
        crossings_mean=float(crossings.mean()),
        saturation=float(np.asarray(out["saturation"]).mean()),
        track_err=float(np.asarray(out["track_err"]).mean()),
        clearance=float(np.asarray(out["clearance"]).mean()),
        nacon_max=int(out["nacon_max"]),
        steps=int(out["steps"]),
    )


def scan(
    run_dir: Path,
    *,
    backend: str = "auto",
    naconmax_per_env: int = 256,
    cell_names: list[str] | None = None,
    speeds=terrain_suite.SPEEDS,
    baseline_ref: str | None = None,
) -> dict:
    """Score `run_dir`'s latest checkpoint on the measurement course."""
    import jax

    from wojtek_rl.battery import load_checkpoint_policy

    cells = [
        c for c in terrain_suite.CELLS
        if cell_names is None or c.name in cell_names
    ]
    if cell_names:
        unknown = set(cell_names) - {c.name for c in terrain_suite.CELLS}
        if unknown:
            raise KeyError(f"unknown cell(s) {sorted(unknown)}")

    run, env, ckpt, inf = load_checkpoint_policy(
        run_dir,
        flat=False,
        env_overrides={
            "terrain": {
                "enable": True,
                "arena": "eval",
                # The spawn is written into qpos, so jitter and random yaw would
                # only make the course non-reproducible.
                "pad_jitter": 0.0,
                "spawn_yaw": False,
            },
            "sim": {
                "backend": backend,
                "num_envs": terrain_suite.RUNS_PER_CELL_SPEED,
                # Warp allocates its contact pool up front and drops overflow
                # silently. Generous here on purpose, and the recorded nacon_max
                # is what says whether it was enough.
                "naconmax_per_env": naconmax_per_env,
            },
            # The course drives the command itself; a mid-episode resample would
            # walk the robot off the tile.
            "command": {"resample_steps": 10**9},
        },
    )
    spec = json.loads(env._terrain_files["spec"].read_text())
    if spec["n_rows"] != len(terrain_suite.DIFFICULTIES):
        raise ValueError(
            f"the eval arena has {spec['n_rows']} rows, the suite defines "
            f"{len(terrain_suite.DIFFICULTIES)}. Rebuild it: "
            "`./training/run.sh build-terrain --arena eval`"
        )
    runner = make_cell_runner(env, inf)

    result = {
        "schema": SCAN_SCHEMA,
        "run": run["run_name"],
        "checkpoint": ckpt.name,
        "engine": env._backend,
        "arena": terrain_suite.arena_fingerprint(),
        "scene": Path(env.xml_path).name,
        "runs_per_cell_speed": terrain_suite.RUNS_PER_CELL_SPEED,
        "crossings_required": terrain_suite.CROSSINGS,
        "settle_steps": terrain_suite.SETTLE_STEPS,
        "saturation_threshold_frac": SATURATION_FRAC,
        "command_height": COMMAND_HEIGHT,
        "naconmax_per_env": naconmax_per_env,
        "warnings": command_box_warnings(run, speeds),
        "cells": {},
    }
    if cell_names:
        result["warnings"].append(
            f"partial scan: {len(cells)} of {len(terrain_suite.CELLS)} cells"
        )

    budgets = {s: terrain_suite.episode_budget(s, float(env.dt)) for s in speeds}
    t0 = time.perf_counter()
    env_steps = 0
    for cell in cells:
        centre, spawn, yaw, pad_h = _spawn_table(env, cell)
        per_speed = {}
        for speed in speeds:
            out = runner(
                jax.random.PRNGKey(
                    cell.row * 1000 + terrain.TYPES.index(cell.terrain_type)
                ),
                centre, spawn, pad_h, yaw, float(speed), COMMAND_HEIGHT,
                budget=budgets[speed],
            )
            out = jax.tree.map(np.asarray, out)
            reduced = reduce_runs(out)
            per_speed[f"{speed}"] = cell_entry(cell, speed, reduced)
            env_steps += reduced.steps * terrain_suite.RUNS_PER_CELL_SPEED
            print(
                f"{cell.name:34s} vx {speed:<4} "
                f"pass {reduced.passed:2d}/{reduced.of} "
                f"falls {reduced.falls:2d} timeouts {reduced.timeouts:2d} "
                f"steps {reduced.steps}"
            )
        result["cells"][cell.name] = per_speed

    wall = time.perf_counter() - t0
    result["perf"] = {
        "wall_clock_s": round(wall, 1),
        "env_steps": env_steps,
        "env_steps_per_s": round(env_steps / max(wall, 1e-9)),
    }
    nacon_max = max(
        (r["nacon_max"] for per in result["cells"].values() for r in per.values()),
        default=0,
    )
    capacity = naconmax_per_env * terrain_suite.RUNS_PER_CELL_SPEED
    result["contacts"] = {
        "nacon_max": nacon_max,
        "capacity": capacity,
        # Warp discards overflow silently, so a scan that hit the ceiling is not
        # a measurement of the policy.
        "overflow": bool(env._backend == "warp" and nacon_max >= capacity),
    }
    if result["contacts"]["overflow"]:
        result["warnings"].append(
            f"contact pool overflowed ({nacon_max} >= {capacity}); raise "
            "--naconmax-per-env and rescan, the numbers above are not valid"
        )
    baseline = load_baseline(baseline_ref)
    result["gate"] = {
        "absolute": absolute_gate(result["cells"]),
        "relative": relative_gate(result, baseline),
    }
    result["timestamp"] = datetime.now().isoformat(timespec="seconds")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="required unless --list-cells")
    ap.add_argument("--out", default=None, help="default: <run>/terrain_scan.json")
    ap.add_argument("--backend", choices=["auto", "warp", "jax"], default="auto")
    ap.add_argument(
        "--naconmax-per-env", type=int, default=256,
        help="warp contact pool per env; the scan records the peak so an "
             "overflow is visible instead of silent",
    )
    ap.add_argument(
        "--cells", default=None,
        help="comma-separated cell names for a partial scan (the cross-engine "
             "cross-check runs a few cells on MJX-jax)",
    )
    ap.add_argument(
        "--speeds", default=None,
        help=f"comma-separated commanded speeds (default "
             f"{','.join(str(s) for s in terrain_suite.SPEEDS)})",
    )
    ap.add_argument(
        "--baseline", default=None,
        help="previous keeper's scan: a path, a directory holding "
             "terrain_scan.json, or an HF reference org/name[@rev]",
    )
    ap.add_argument("--list-cells", action="store_true", help="print the cells and exit")
    args = ap.parse_args()

    if args.list_cells:
        for c in terrain_suite.CELLS:
            value, unit = terrain_suite.realized_dimension(c.terrain_type, c.difficulty)
            bar = "tracked" if c.tracked else f"{c.bar:.0%} ({terrain_suite.bar_count(c.bar)}/32)"
            print(f"{c.name:34s} row {c.row:2d}  d={c.difficulty:.6f}  "
                  f"{value:5.1f} {unit:3s}  {bar}")
        return
    if not args.run:
        ap.error("--run is required")

    result = scan(
        Path(args.run),
        backend=args.backend,
        naconmax_per_env=args.naconmax_per_env,
        cell_names=args.cells.split(",") if args.cells else None,
        speeds=(
            tuple(float(s) for s in args.speeds.split(","))
            if args.speeds else terrain_suite.SPEEDS
        ),
        baseline_ref=args.baseline,
    )
    out = Path(args.out) if args.out else Path(args.run) / "terrain_scan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    gate = result["gate"]
    print()
    for line in result["warnings"]:
        print(f"WARNING: {line}")
    print(f"absolute gate: {gate['absolute']['verdict']} "
          f"({len(gate['absolute']['failures'])} of "
          f"{gate['absolute']['checked']} gated cell/speed pairs below bar)")
    for f in gate["absolute"]["failures"]:
        print(f"  {f['cell']} @ vx {f['speed']}: {f['passed']} < {f['bar']} "
              f"[{f['provenance']}]")
    print(f"relative gate: {gate['relative']['verdict']}")
    for line in gate["relative"].get("notes", []):
        print(f"  {line}")
    for d in gate["relative"].get("drops", []):
        print(f"  {d['cell']} @ vx {d['speed']}: {d['was']}% -> {d['now']}% "
              f"({d['drop']} points)")
    perf = result["perf"]
    print(f"{perf['env_steps']:,} env steps in {perf['wall_clock_s']}s "
          f"({perf['env_steps_per_s']:,}/s)  -> {out}")


if __name__ == "__main__":
    main()
