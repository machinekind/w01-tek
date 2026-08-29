"""Load-and-step check for the generated terrain scene.

Proves scene_terrain.xml (many static box geoms plus one heightfield)
compiles, puts on the selected backend, and steps a batch of envs. Reports
model size, put_model and step timings, and contact/constraint-buffer usage
so the warp budgets can be sized from measurement rather than guesswork.

The batch spawns spread across the arena's tiles, hardest rows first, offset
into each tile's feature band -- a robot standing at the world origin touches
nothing but flat pad, and the budgets have to cover feet on treads, shins on
riser faces and the tumbles that follow. `--spawn origin` restores the old
single-pose batch (and is the fallback when the scene has no spec).

Everything lands in a JSON report. On any failure the report carries the
traceback, the optional sentinel gets "FAIL <reason>", and the process
exits nonzero; on success the sentinel gets "OK".

A heightfield contact overflow counts as a failure. Warp reports it from
device code and then drops the contacts, so a check that prints it measured
something other than the model. The count goes in the report either way.

CPU:  ./run.sh check-terrain --backend jax --num-envs 4 --steps 10
GPU:  ./run.sh check-terrain --backend warp --arena eval
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from wojtek_rl import paths

# Uniform half-width of the per-env actuator noise added to the home ctrl,
# in ctrl units. Small enough that every env stays near the standing pose.
CTRL_NOISE = 0.02
# Fixed seed for the ctrl noise so the batch is identical run to run.
SEED = 0
# Warp contact/constraint budgets for make_data, both deliberately generous:
# an undersized contact pool drops contacts silently on warp, and rows past
# njmax apply no force just as silently -- either would hide the very peaks
# this tool measures. The env defaults (32/320) size a real training run once
# the measured peaks are known.
NACONMAX_PER_ENV = 256
NJMAX = 1024
# How far from the tile centre a tile spawn is placed, as a CHEBYSHEV radius:
# past the 0.60 m platform, on the treads (which end at 1.25 m) and inside the
# scattered boxes' 1.40 m reach, so the feature band starts under the feet.
FEATURE_RADIUS = 0.9
# What MJWarp prints when a heightfield collision exceeds mjMAXCONPAIR and
# drops the contacts it cannot store. Those contacts are gone from the physics,
# so a run that prints this measured a different robot than the model says.
OVERFLOW_WARNING = "height field collision overflow"


def count_overflow_warnings(text: str) -> int:
    """Occurrences of the heightfield overflow warning in captured output."""
    return text.count(OVERFLOW_WARNING)


@contextlib.contextmanager
def capture_os_stdout(enabled: bool):
    """Collect writes to file descriptor 1, then re-emit them.

    The warning comes from ``wp.printf`` in device code, which writes to the
    process's stdout descriptor. Python never sees the string, so
    ``contextlib.redirect_stdout`` does not catch it and only ``dup2`` does.
    Everything captured is written out again afterwards, so redirecting hides
    nothing.

    Yields a one-element list; it holds the captured text once the block ends.
    Disabled, it yields an empty capture and leaves the descriptor alone.
    """
    holder = [""]
    if not enabled:
        yield holder
        return
    tmp = tempfile.TemporaryFile(mode="w+")
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(tmp.fileno(), 1)
        yield holder
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        tmp.seek(0)
        holder[0] = tmp.read()
        tmp.close()
        if holder[0]:
            sys.stdout.write(holder[0])
            sys.stdout.flush()


def _tile_spawns(arena: str, num_envs: int) -> np.ndarray | None:
    """(num_envs, 3) base positions over the arena's tiles, hardest rows first.

    Deterministic: env i takes tile i modulo the tile count (sorted by falling
    difficulty, so the hard rows fill first) and one of the eight course
    headings, rotated a step per lap so a tile is not always probed from the
    same side. The spawn sits FEATURE_RADIUS out in Chebyshev terms, and z
    rides the lookup grid's surface height there. None when the arena has no
    spec on disk (a custom --scene), which falls back to the origin spawn.
    """
    from wojtek_rl import terrain

    files = paths.terrain_paths(arena)
    if not (files["spec"].exists() and files["lookup"].exists()):
        return None
    spec = json.loads(files["spec"].read_text())
    tiles = sorted(spec["tiles"], key=lambda t: -t["difficulty"])
    npz = np.load(files["lookup"])
    cell_x = (float(npz["x_max"]) - float(npz["x_min"])) / (int(npz["ncol"]) - 1)
    cell_y = (float(npz["y_max"]) - float(npz["y_min"])) / (int(npz["nrow"]) - 1)

    spawns = np.zeros((num_envs, 3))
    for i in range(num_envs):
        t = tiles[i % len(tiles)]
        yaw = (np.pi / 4) * ((i + i // len(tiles)) % 8)
        stretch = 1.0 / max(abs(np.cos(yaw)), abs(np.sin(yaw)))
        spawns[i, 0] = t["origin"][0] + np.cos(yaw) * FEATURE_RADIUS * stretch
        spawns[i, 1] = t["origin"][1] + np.sin(yaw) * FEATURE_RADIUS * stretch
    spawns[:, 2] = terrain._bilinear(
        npz["lookup"], float(npz["x_min"]), cell_x, float(npz["y_min"]), cell_y,
        spawns[:, 0], spawns[:, 1],
    )
    return spawns


def _contact_dist(data):
    """Return the batched contact-distance array for either backend.

    The jax backend nests a Contact struct under _impl; warp flattens the
    same fields as contact__<name>. Both hold one distance per candidate
    contact slot.
    """
    impl = data._impl
    if hasattr(impl, "contact"):
        return impl.contact.dist
    return impl.contact__dist


def _native_ncon(data):
    """Warp's live active-contact count, or None on jax.

    Warp keeps nacon, the number of contacts actually written to the shared
    pool this step. The jax backend has no such per-step scalar; its ncon is
    the static candidate-buffer size, reported separately as capacity.
    """
    return getattr(data._impl, "nacon", None)


def _capacity(data, backend: str) -> dict:
    """Static contact-buffer capacity, named by the field it comes from."""
    impl = data._impl
    if backend == "warp":
        return {"field": "naconmax", "value": int(impl.naconmax)}
    return {"field": "ncon", "value": int(impl.ncon)}


def run_check(args) -> dict:
    import jax
    import jax.numpy as jp
    import mujoco
    from mujoco import mjx

    from wojtek_rl.base import make_data_fn, resolve_backend

    scene = Path(args.scene)
    backend = resolve_backend(args.backend)

    t = time.perf_counter()
    m = mujoco.MjModel.from_xml_path(str(scene))
    model_compile_s = time.perf_counter() - t

    result = {
        "status": "running",
        "scene": str(scene),
        "backend_requested": args.backend,
        "backend": backend,
        "num_envs": int(args.num_envs),
        "steps": int(args.steps),
        "model": {
            "ngeom": int(m.ngeom),
            "nbody": int(m.nbody),
            "nu": int(m.nu),
            "nq": int(m.nq),
            "nhfield": int(m.nhfield),
            "hfield_nrow": int(m.hfield_nrow[0]) if m.nhfield else 0,
            "hfield_ncol": int(m.hfield_ncol[0]) if m.nhfield else 0,
        },
        "model_compile_s": model_compile_s,
    }

    t = time.perf_counter()
    mjx_model = mjx.put_model(m, impl=backend)
    result["put_model_s"] = time.perf_counter() - t

    key = m.key("home")
    qpos_batch = np.tile(np.asarray(key.qpos), (args.num_envs, 1))
    spawns = _tile_spawns(args.arena, args.num_envs) if args.spawn == "tiles" else None
    if args.spawn == "tiles" and spawns is None:
        print("NOTE: no spec/lookup for this scene; spawning at the origin")
    if spawns is not None:
        qpos_batch[:, 0:2] = spawns[:, 0:2]
        # The home keyframe's base height, above the local surface instead of
        # above z = 0.
        qpos_batch[:, 2] = spawns[:, 2] + float(key.qpos[2])
    result["spawn"] = "tiles" if spawns is not None else "origin"
    result["naconmax_per_env"] = int(args.naconmax_per_env)
    result["njmax"] = int(args.njmax)
    qpos = jp.array(qpos_batch)

    rng = np.random.RandomState(SEED)
    noise = rng.uniform(-CTRL_NOISE, CTRL_NOISE, size=(args.num_envs, m.nu))
    ctrl = jp.array(key.ctrl[None, :] + noise)

    make_single = make_data_fn(
        backend, m, mjx_model, args.naconmax_per_env, args.njmax, args.num_envs
    )

    def init(qpos_row, ctrl_row):
        return make_single().replace(qpos=qpos_row, ctrl=ctrl_row)

    data = jax.vmap(init)(qpos, ctrl)
    step = jax.vmap(lambda d: mjx.step(mjx_model, d))

    # First step on its own. This call pays the step JIT/compile cost, kept
    # out of the steady-state rate below.
    step1 = jax.jit(step)
    t = time.perf_counter()
    jax.block_until_ready(step1(data))
    result["first_step_s"] = time.perf_counter() - t

    def body(d, _):
        d = step(d)
        out = {"active": jp.sum(_contact_dist(d) < 0.0, axis=-1)}  # per env
        nacon = _native_ncon(d)
        if nacon is not None:
            out["nacon"] = nacon
            # Per-world constraint rows, warp only. Rows past njmax apply no
            # force with no warning, so the peak is the only way to see it.
            out["nefc"] = jp.asarray(d._impl.nefc)
        return d, out

    @jax.jit
    def run(d):
        return jax.lax.scan(body, d, None, length=args.steps)

    # Both scans run inside the capture: they step the same physics, so a
    # warning from either one is the same finding. Only warp emits it.
    with capture_os_stdout(backend == "warp") as captured:
        jax.block_until_ready(run(data))  # compile the scan
        t = time.perf_counter()
        final, metrics = run(data)
        jax.block_until_ready((final, metrics))
        steady_s = time.perf_counter() - t

    result["steady_state_s"] = steady_s
    result["steps_per_s"] = args.steps / steady_s
    result["env_steps_per_s"] = args.steps * args.num_envs / steady_s

    active = np.asarray(metrics["active"])
    contacts = {
        "measure": "contacts with dist < 0, per env per step",
        "active_max": int(active.max()),
        "active_mean": float(active.mean()),
        "capacity": _capacity(data, backend),
    }
    if "nacon" in metrics:
        nacon = np.asarray(metrics["nacon"])
        contacts["nacon_max"] = int(nacon.max())
        contacts["nacon_mean"] = float(nacon.mean())
        contacts["overflow"] = bool(nacon.max() >= contacts["capacity"]["value"])
        nefc = np.asarray(metrics["nefc"])
        contacts["nefc_max"] = int(nefc.max())
        contacts["nefc_mean"] = float(nefc.mean())
        contacts["njmax"] = int(args.njmax)
        contacts["rows_overflow"] = bool(nefc.max() >= args.njmax)
    else:
        # jax computes every candidate contact and sizes rows exactly, so
        # there is no overflow to report; ncon is the static buffer size,
        # not a per-step count.
        contacts["overflow"] = False
    # Separate from the pool overflow above: this one is the per-geom-pair cap
    # inside the heightfield routine, which no counter in mjx.Data records.
    contacts["hfield_overflow_warnings"] = count_overflow_warnings(captured[0])
    result["contacts"] = contacts

    result["status"] = "ok"
    return result


def _write(out: Path, result: dict) -> None:
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
    except Exception:
        # Writing the report is best effort on the fail path; the sentinel
        # and exit code still carry the outcome.
        traceback.print_exc()


def _sentinel(path, text: str) -> None:
    if path:
        Path(path).write_text(text + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["auto", "warp", "jax"], default="auto")
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--spawn", choices=["tiles", "origin"], default="tiles",
                   help="tiles: spread the batch over the arena, hardest rows "
                        "first, offset into each tile's feature band; origin: "
                        "the old one-pose batch at the world origin")
    p.add_argument("--naconmax-per-env", type=int, default=NACONMAX_PER_ENV,
                   help=f"warp contact pool per env (default "
                        f"{NACONMAX_PER_ENV}, generous on purpose: an "
                        "undersized pool hides the peak this tool measures)")
    p.add_argument("--njmax", type=int, default=NJMAX,
                   help=f"warp constraint rows per world (default {NJMAX}, "
                        "generous on purpose; the training default 320 is "
                        "sized from the peak measured here)")
    p.add_argument("--arena", choices=paths.TERRAIN_KINDS, default="train",
                   help="which generated arena to check; also names the "
                        "default report so two arenas do not overwrite "
                        "each other")
    p.add_argument("--scene", default=None,
                   help="scene XML to check (default: the --arena one)")
    p.add_argument("--out", default=None)
    p.add_argument("--sentinel", default=None)
    args = p.parse_args()
    if args.scene is None:
        args.scene = str(paths.terrain_paths(args.arena)["scene"])
    elif args.spawn == "tiles":
        # A custom scene need not match the --arena spec the tile table would
        # come from, so don't place spawns from the wrong arena.
        print("NOTE: --scene given; spawning at the origin")
        args.spawn = "origin"
    if args.out is None:
        tag = "" if args.arena == "train" else f"_{args.arena}"
        args.out = str(paths.PROJECT_DIR / f"runs/check_terrain{tag}.json")

    out = Path(args.out)
    try:
        result = run_check(args)
        c = result["contacts"]
        dropped = c["hfield_overflow_warnings"]
        if dropped:
            reason = (
                f"{dropped} heightfield contact overflow warnings; contacts "
                "past the per-pair cap were dropped, so these numbers are not "
                "this model's physics"
            )
            result["status"] = "fail"
            result["error"] = reason
            _write(out, result)
            _sentinel(args.sentinel, f"FAIL {reason}")
            print(f"check-terrain FAILED: {reason} -> {out}", file=sys.stderr)
            sys.exit(1)
        _write(out, result)
        _sentinel(args.sentinel, "OK")
        rows = (
            f"  nefc max {c['nefc_max']}/{c['njmax']}" if "nefc_max" in c else ""
        )
        print(
            f"check-terrain OK  {result['backend']}  "
            f"spawn {result.get('spawn', '?')}  "
            f"{result['steps_per_s']:.1f} steps/s  "
            f"{result['env_steps_per_s']:,.0f} env-steps/s  "
            f"active ncon max {c['active_max']} mean {c['active_mean']:.1f}"
            f"{rows}  -> {out}"
        )
        sys.exit(0)
    except Exception:
        tb = traceback.format_exc()
        _write(out, {
            "status": "fail",
            "scene": args.scene,
            "backend_requested": args.backend,
            "error": tb,
        })
        reason = tb.strip().splitlines()[-1][:200]
        _sentinel(args.sentinel, f"FAIL {reason}")
        print(tb, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
