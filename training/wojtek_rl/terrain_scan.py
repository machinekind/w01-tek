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
passes when all four finish inside its own step budget with no fall.

Those radii are Chebyshev, matching how the terrain is laid out (concentric
squares), so "cleared the obstacle band" means the same thing on all eight
headings. A diagonal walks sqrt(2) further to get there, so each run's deadline
is sized on its own distance -- see terrain_suite.OUT_RADIUS and run_deadlines.

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
arena: two scans of one checkpoint at the same ``--eval-seed`` return the same
numbers. A different ``--eval-seed`` redraws the rollout's noise on the same
course, which is how the score's test-retest spread gets measured.
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
# baseline keeper. Sized from the scan's own test-retest noise: three scans
# of one checkpoint on different --eval-seed draws moved a single cell/speed
# pair by up to 6 of 32 runs (18.75 points), so the old 10-point limit
# failed on noise alone (2026-07-27 validation report). The whole-course
# pass total moved only ~2% in the same test; a tighter gate should compare
# that total instead of single pairs.
RELATIVE_DROP_LIMIT = 25.0
# Warp contact pool per env: the env's own model-derived floor (22 geoms
# collide with the heightfield at 4 contacts per pair), and still 7x the
# per-env peak of 12 contacts measured on the GPU. The pool is allocated up
# front for the whole batch, so a bigger one is not free -- a 256 pool is what
# ran check-terrain out of device memory at 4096 envs. The scan records the peak
# it actually reached, so too small shows up as a recorded overflow.
DEFAULT_NACONMAX_PER_ENV = 88
SCAN_SCHEMA = 1


# -- pure helpers: no jax, no checkpoint, unit-tested directly -----------------


def leg_sign(crossings):
    """+1 while walking out, -1 while walking back.

    The commanded forward speed carries the sign; the heading never changes. A
    180 degree turn on a 13 cm tread might be the hardest thing in the suite,
    and the number would then measure turning instead of climbing.
    """
    return 1.0 - 2.0 * (crossings % 2)


def tile_distance(xy, centre, xp=np):
    """Chebyshev distance from the tile centre: max(|dx|, |dy|).

    Not the Euclidean radius. Every feature in the arena is a concentric square
    -- see terrain._cheby -- so a Euclidean radius would mean something different
    on every heading. See terrain_suite.OUT_RADIUS.
    """
    return xp.max(xp.abs(xy - centre), axis=-1)


def crossing_progress(crossings, distance, running, xp=np):
    """Advance the crossing counter by at most one.

    An outbound leg finishes at OUT_RADIUS from the tile centre, an inbound leg
    at BACK_RADIUS, both as the Chebyshev `distance` above. A run that has already
    fallen or finished never advances again, and the count stops at CROSSINGS.
    Pure, so the rule is testable without physics; `xp` is numpy or jax.numpy.
    """
    outbound = (crossings % 2) == 0
    leg_done = xp.where(
        outbound,
        distance >= terrain_suite.OUT_RADIUS,
        distance <= terrain_suite.BACK_RADIUS,
    )
    return xp.minimum(crossings + (leg_done & running), terrain_suite.CROSSINGS)


def still_running(i, crossings, fell, deadline):
    """Whether each run is still on its course at step `i`.

    A run stops when it falls, when it finishes its four crossings, or when it
    passes its own deadline. The deadline is per run because a diagonal heading
    walks sqrt(2) further to the same Chebyshev radius: one shared budget would
    hand an axis run that extra slack and make the speed a run has to sustain --
    and so the effective difficulty -- heading-dependent.
    """
    return ~(fell | (crossings >= terrain_suite.CROSSINGS)) & (i < deadline)


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
    measured: int  # runs that contributed metric steps
    nacon_max: int
    nefc_max: int
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
        # How many of the runs the three metrics above average over. A run that
        # fell inside the settle window contributes no steps, so a low number
        # here means those metrics describe the survivors, not the cell.
        "measured": result.measured,
        "nacon_max": result.nacon_max,
        "nefc_max": result.nefc_max,
        "steps": result.steps,
    }


def gated_pairs(cells=terrain_suite.CELLS, speeds=terrain_suite.SPEEDS) -> int:
    """How many (cell, speed) pairs a complete scan gates."""
    return sum(1 for c in cells if not c.tracked) * len(speeds)


def absolute_gate(cells: dict, expect_gated: int | None = None) -> dict:
    """Every gated cell against its bar. Tracked cells are reported, not gated.

    `expect_gated` is how many gated pairs a complete scan would have. A partial
    scan (`--cells`, or a crash part way) comes back `incomplete` rather than
    `pass`: "the four cells I measured are fine" must not read as "the policy
    passed".
    """
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
    if failures:
        verdict = "fail"
    elif expect_gated is not None and checked < expect_gated:
        verdict = "incomplete"
    else:
        verdict = "pass"
    return {
        "verdict": verdict,
        "checked": checked,
        "expected": expect_gated,
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


def cell_key(cell, eval_seed: int = 0):
    """One cell's rollout key: which draw of the policy's noise it runs on.

    Seed 0 has to return the base key unchanged -- every scan taken so far ran
    on it, and those numbers are the baselines the relative gate compares
    against. Any other seed is a fresh draw of the same course.
    """
    import jax

    base = jax.random.PRNGKey(
        cell.row * 1000 + terrain.TYPES.index(cell.terrain_type)
    )
    if eval_seed == 0:
        return base
    return jax.random.fold_in(base, eval_seed)


def baseline_seed_warnings(baseline: dict | None, eval_seed: int) -> list[str]:
    """Flag a baseline measured on a different noise draw.

    The two are still comparable -- same course, same policy input -- but part
    of any gap is then the scan's own test-retest spread rather than the
    policy. A baseline from before this option carries no seed and ran on 0.
    """
    if baseline is None:
        return []
    base_seed = int(baseline.get("eval_seed", 0))
    if base_seed == eval_seed:
        return []
    return [
        f"baseline was scanned at eval seed {base_seed}, this scan at "
        f"{eval_seed}; part of any difference between them is the scan's own "
        "test-retest spread, not the policy"
    ]


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


def check_arena(spec: dict) -> None:
    """Refuse to scan an arena that is not the one the suite defines.

    The recorded fingerprint comes from the suite's constants, and the gate
    compares two scans on it -- so an arena built with different parameters
    (`build-terrain --arena eval --seed 5`, or an arena from before a suite
    change) would produce numbers filed under a fingerprint that describes
    something else. The cells are defined by row index, so a row table that does
    not match is not a difficulty mismatch, it is a different terrain.
    """
    want = {
        "seed": terrain_suite.EVAL_SEED,
        "n_rows": len(terrain_suite.DIFFICULTIES),
        "ordered": terrain_suite.EVAL_ORDERED,
        "pad_radius": terrain_suite.EVAL_PAD_RADIUS,
        "n_steps": terrain.N_STEPS,
        "stair_platform_half": terrain.STAIR_PLATFORM_HALF,
        # build-terrain passes these three through for every arena kind, so
        # this gate is the only place a non-default eval geometry is caught.
        "tile_size": terrain.TILE_SIZE,
        "border": terrain.BORDER,
        "cell_size": terrain.CELL_SIZE,
    }
    wrong = {
        key: (spec.get(key), value)
        for key, value in want.items()
        if spec.get(key) != value
    }
    rows = [t["difficulty"] for t in spec.get("tiles", []) if t["col"] == 0]
    if len(rows) == len(terrain_suite.DIFFICULTIES) and not all(
        abs(a - b) < 1e-6 for a, b in zip(sorted(rows), terrain_suite.DIFFICULTIES)
    ):
        wrong["difficulties"] = (sorted(rows), list(terrain_suite.DIFFICULTIES))
    if wrong:
        detail = "; ".join(f"{k}: found {f!r}, expected {w!r}" for k, (f, w) in wrong.items())
        raise ValueError(
            f"the eval arena is not the measurement course ({detail}). Rebuild "
            "it: `./training/run.sh build-terrain --arena eval`"
        )


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
    centre = np.asarray(env._terrain.origin_xy)[cell.row, type_index]
    pad_h = float(np.asarray(env._terrain.pad_h)[cell.row, type_index])
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


def scan_reset(env, rng, spawn_xy, pad_h, yaw, command):
    """One run's starting state: env.reset, then the course's pose and command.

    Writing the pose into qpos is safe for the reason the training wrapper's
    teleport is safe -- the next physics step recomputes kinematics from qpos,
    and no observation carries world position or heading. The joints are pinned
    to the commanded height's stance anchor with none of reset's joint noise, so
    the course is fully deterministic.

    The command goes in BEFORE the observation is rebuilt. `command` is an
    observed component and env.reset samples its own random one, so leaving it
    would make the policy's very first action a response to a command the course
    never issued.
    """
    import jax.numpy as jp
    from mujoco import mjx

    state = env.reset(rng)
    height = command[3]
    anchor = env._height_ctrl(height)
    qpos = env._home_qpos.at[env._qadr].set(anchor)
    qpos = qpos.at[0:2].set(spawn_xy)
    qpos = qpos.at[2].set(pad_h + height)
    qpos = qpos.at[3:7].set(jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)]))
    data = state.data.replace(
        qpos=qpos, qvel=jp.zeros(env.mj_model.nv), ctrl=anchor
    )
    data = mjx.forward(env.mjx_model, data)
    info = dict(state.info)
    info["command"] = command
    info["motor_targets"] = anchor
    return state.replace(data=data, obs=env._get_obs(data, info), info=info)


def make_cell_runner(env, inf):
    """A jitted "score one cell at one speed" function.

    Batch shape is fixed at RUNS_PER_CELL_SPEED and the step budget is the only
    static argument, so exactly three programs get compiled -- one per commanded
    speed -- and each is dispatched once per cell. That is a requirement, not an
    optimization: warp allocates its contact pool when the data is created,
    sized by the number of environments, so a varying batch size means a new
    allocation. Which tile, which heading and which direction are numbers fed
    into the same program.

    Known perf ceiling, accepted for now: the cell axis could ride in the batch
    too (43 x 32 worlds is still one fixed shape), and the 129 sequential
    dispatches of a 32-world batch are why the measured scan runs ~200x below
    what the physics sustains at training batch sizes. Folding cells in changes
    nothing measured, only the wall clock, but it reshapes this whole rollout
    and needs GPU validation -- a follow-up, not a review-fix edit.
    """
    import functools

    import jax
    import jax.numpy as jp

    n = terrain_suite.RUNS_PER_CELL_SPEED
    torque_cap = float(np.asarray(env.mj_model.actuator_forcerange[:, 1]).max())

    reset_one = functools.partial(scan_reset, env)

    def nacon_of(data):
        """Warp's live active-contact count, 0 on the jax backend (which has no
        contact budget to overflow)."""
        value = getattr(data._impl, "nacon", None)
        if value is None:
            return jp.zeros((), jp.int32)
        return jp.max(jp.asarray(value)).astype(jp.int32)

    def nefc_of(data):
        """Warp's live constraint-row count, max over worlds; 0 on the jax
        backend. Rows past sim.njmax apply no force with no warning anywhere,
        so the peak has to be recorded to make an overflow visible -- the
        nacon counter above cannot see it (contacts vs rows are different
        budgets)."""
        value = getattr(data._impl, "nefc", None)
        # The jax impl keeps nefc as a static buffer size, not a per-step
        # count; only warp's per-world array is a measurement.
        if value is None or getattr(value, "ndim", 0) == 0:
            return jp.zeros((), jp.int32)
        return jp.max(jp.asarray(value)).astype(jp.int32)

    @functools.partial(jax.jit, static_argnames=("budget",))
    def run(rng, centre, spawn_xy, pad_h, yaw, speed, height, deadline, budget):
        zeros = jp.zeros(n)

        def command_at(crossings):
            """The course's command: forward speed signed by the leg, no lateral
            or yaw component, the fixed standing height."""
            return jp.stack(
                [leg_sign(crossings) * speed, zeros, zeros, jp.full(n, height)],
                axis=-1,
            )

        state = jax.vmap(reset_one)(
            jax.random.split(rng, n), spawn_xy, pad_h, yaw,
            command_at(jp.zeros(n, jp.int32)),  # the first leg walks out
        )

        def body(carry):
            i, state, rng, crossings, fell, sat, err, clr, counted, nacon, nefc = carry
            sign = leg_sign(crossings)
            command = command_at(crossings)
            info = dict(state.info)
            info["command"] = command
            state = state.replace(info=info)
            rng, sub = jax.random.split(rng)
            action, _ = inf(state.obs, sub)
            was_running = still_running(i, crossings, fell, deadline)
            state = jax.vmap(env.step)(state, action)
            data = state.data

            distance = tile_distance(data.qpos[:, 0:2], centre, xp=jp)
            crossings = crossing_progress(crossings, distance, was_running, xp=jp)
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
                jp.maximum(nefc, nefc_of(data)),
            )

        def cond(carry):
            i, _, _, crossings, fell, *_ = carry
            # `budget` is the hard stop (the longest run's deadline); the loop
            # normally ends earlier, when every run has finished, fallen or run
            # out of its own time.
            return (i < budget) & jp.any(still_running(i, crossings, fell, deadline))

        init = (
            jp.zeros((), jp.int32), state, rng,
            jp.zeros(n, jp.int32), jp.zeros(n, bool),
            zeros, zeros, zeros, zeros,
            jp.zeros((), jp.int32), jp.zeros((), jp.int32),
        )
        i, state, _, crossings, fell, sat, err, clr, counted, nacon, nefc = (
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
            "nefc_max": nefc,
        }

    return run


def reduce_runs(out) -> CellResult:
    """One cell's 32 run outcomes into its recorded numbers.

    The per-step metrics average over the runs that actually contributed steps,
    not over all 32. A run that fell inside the settle window has no metric steps
    at all, so its saturation, tracking error and clearance come back as zero --
    averaging those in drags the cell's numbers toward zero exactly where falls
    are common, which is the hard cells. On a cell where 20 of 32 runs fall early
    that reported a tracking error 2.7x better than the survivors', biased in the
    direction that makes hard terrain look easy.

    Pass, fall and timeout counts are over all 32 runs regardless: a fall is an
    outcome, not a missing measurement.
    """
    crossings = np.asarray(out["crossings"])
    fell = np.asarray(out["fell"], dtype=bool)
    finished = crossings >= terrain_suite.CROSSINGS
    measured = np.asarray(out["counted"]) > 0

    def mean_of(key):
        values = np.asarray(out[key])[measured]
        return float(values.mean()) if values.size else 0.0

    return CellResult(
        passed=int((finished & ~fell).sum()),
        of=len(crossings),
        falls=int(fell.sum()),
        timeouts=int((~finished & ~fell).sum()),
        crossings_mean=float(crossings.mean()),
        saturation=mean_of("saturation"),
        track_err=mean_of("track_err"),
        clearance=mean_of("clearance"),
        measured=int(measured.sum()),
        nacon_max=int(out["nacon_max"]),
        nefc_max=int(out["nefc_max"]),
        steps=int(out["steps"]),
    )


def scan(
    run_dir: Path,
    *,
    backend: str = "auto",
    naconmax_per_env: int = DEFAULT_NACONMAX_PER_ENV,
    njmax: int | None = None,
    cell_names: list[str] | None = None,
    speeds=terrain_suite.SPEEDS,
    baseline_ref: str | None = None,
    eval_seed: int = 0,
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

    # Resolved before the rollout on purpose: it is pure IO, and a bad
    # reference must not throw away a finished scan.
    baseline = load_baseline(baseline_ref)

    sim_overrides = {
        "backend": backend,
        "num_envs": terrain_suite.RUNS_PER_CELL_SPEED,
        # Warp allocates its contact pool up front and drops overflow
        # silently. The recorded nacon_max is what says whether the
        # budget was enough.
        "naconmax_per_env": naconmax_per_env,
    }
    if njmax is not None:
        sim_overrides["njmax"] = njmax
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
            "sim": sim_overrides,
            # The course drives the command itself; a mid-episode resample would
            # walk the robot off the tile.
            "command": {"resample_steps": 10**9},
        },
    )
    check_arena(json.loads(env._terrain.files["spec"].read_text()))
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
        "eval_seed": eval_seed,
        "naconmax_per_env": naconmax_per_env,
        "njmax": int(env._config.sim.njmax),
        "warnings": command_box_warnings(run, speeds),
        "cells": {},
    }
    if cell_names:
        result["warnings"].append(
            f"partial scan: {len(cells)} of {len(terrain_suite.CELLS)} cells"
        )
    if tuple(speeds) != tuple(terrain_suite.SPEEDS):
        result["warnings"].append(
            f"partial scan: speeds {sorted(speeds)} of "
            f"{sorted(terrain_suite.SPEEDS)}"
        )
    result["warnings"].extend(baseline_seed_warnings(baseline, eval_seed))

    dt = float(env.dt)
    budgets = {s: terrain_suite.episode_budget(s, dt) for s in speeds}
    deadlines = {
        s: np.asarray(terrain_suite.run_deadlines(s, dt), dtype=np.int32)
        for s in speeds
    }
    t0 = time.perf_counter()
    env_steps = 0
    for cell in cells:
        centre, spawn, yaw, pad_h = _spawn_table(env, cell)
        per_speed = {}
        for speed in speeds:
            out = runner(
                cell_key(cell, eval_seed),
                centre, spawn, pad_h, yaw, float(speed), COMMAND_HEIGHT,
                deadlines[speed], budget=budgets[speed],
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
    nefc_max = max(
        (r["nefc_max"] for per in result["cells"].values() for r in per.values()),
        default=0,
    )
    capacity = naconmax_per_env * terrain_suite.RUNS_PER_CELL_SPEED
    result["contacts"] = {
        "nacon_max": nacon_max,
        "capacity": capacity,
        # Warp discards overflow silently, so a scan that hit the ceiling is not
        # a measurement of the policy.
        "overflow": bool(env._backend == "warp" and nacon_max >= capacity),
        # The second budget: constraint rows per world. Rows past sim.njmax
        # apply no force, silently -- one contact costs 6 rows at condim=4
        # with a pyramidal cone, so this ceiling is nearer than it looks.
        "nefc_max": nefc_max,
        "njmax": result["njmax"],
        "rows_overflow": bool(
            env._backend == "warp" and nefc_max >= result["njmax"]
        ),
    }
    if result["contacts"]["overflow"]:
        result["warnings"].append(
            f"contact pool overflowed ({nacon_max} >= {capacity}); raise "
            "--naconmax-per-env and rescan, the numbers above are not valid"
        )
    if result["contacts"]["rows_overflow"]:
        result["warnings"].append(
            f"constraint rows overflowed ({nefc_max} >= {result['njmax']}); "
            "raise --njmax and rescan, the numbers above are not valid"
        )
    result["gate"] = {
        # The completeness reference is the FULL suite, whatever subset this
        # scan measured: "the pairs I measured are fine" must not read as
        # "the policy passed", on the speed axis as much as the cell axis.
        "absolute": absolute_gate(result["cells"], gated_pairs()),
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
        "--naconmax-per-env", type=int, default=DEFAULT_NACONMAX_PER_ENV,
        help=f"warp contact pool per env (default {DEFAULT_NACONMAX_PER_ENV}, "
             "the training value); the scan records the peak so an overflow is "
             "visible instead of silent",
    )
    ap.add_argument(
        "--njmax", type=int, default=None,
        help="warp constraint rows per world (default: the env's own, 320). "
             "Rows past it apply no force with no warning; the scan records "
             "the peak (nefc_max) so this overflow is visible too",
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
        "--eval-seed", type=int, default=0,
        help="which noise draw the rollout runs on: re-scan one checkpoint at "
             "1, 2, 3 to measure the score's test-retest spread. 0 (the "
             "default) is the historical stream every scan so far ran on. The "
             "arena and the course are fixed either way",
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
        njmax=args.njmax,
        cell_names=args.cells.split(",") if args.cells else None,
        speeds=(
            tuple(float(s) for s in args.speeds.split(","))
            if args.speeds else terrain_suite.SPEEDS
        ),
        baseline_ref=args.baseline,
        eval_seed=args.eval_seed,
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
          f"{gate['absolute']['checked']} gated cell/speed pairs below bar; "
          f"a complete scan gates {gate['absolute']['expected']})")
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
