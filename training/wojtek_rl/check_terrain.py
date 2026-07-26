"""Load-and-step check for the generated terrain scene.

Proves scene_terrain.xml (many static box geoms plus one heightfield)
compiles, puts on the selected backend, and steps a batch of envs from the
home keyframe. Reports model size, put_model and step timings, and
contact-buffer usage so the warp contact budget can be sized from
measurement rather than guesswork.

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
# Warp contact/constraint budgets for make_data. naconmax_per_env is
# deliberately generous: an undersized pool drops contacts silently on warp
# and would hide the very number this tool measures. The env defaults
# (32/320) size a real training run once nacon is known.
NACONMAX_PER_ENV = 256
NJMAX = 320
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
    home_qpos = jp.array(key.qpos)

    rng = np.random.RandomState(SEED)
    noise = rng.uniform(-CTRL_NOISE, CTRL_NOISE, size=(args.num_envs, m.nu))
    ctrl = jp.array(key.ctrl[None, :] + noise)

    make_single = make_data_fn(
        backend, m, mjx_model, NACONMAX_PER_ENV, NJMAX, args.num_envs
    )

    def init(ctrl_row):
        return make_single().replace(qpos=home_qpos, ctrl=ctrl_row)

    data = jax.vmap(init)(ctrl)
    step = jax.vmap(lambda d: mjx.step(mjx_model, d))

    # First step on its own. This call pays the step JIT/compile cost, kept
    # out of the steady-state rate below.
    step1 = jax.jit(step)
    t = time.perf_counter()
    jax.block_until_ready(step1(data))
    result["first_step_s"] = time.perf_counter() - t

    def body(d, _):
        d = step(d)
        active = jp.sum(_contact_dist(d) < 0.0, axis=-1)  # touching, per env
        nacon = _native_ncon(d)
        return d, (active if nacon is None else (active, nacon))

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

    active, nacon = metrics if isinstance(metrics, tuple) else (metrics, None)
    active = np.asarray(active)
    contacts = {
        "measure": "contacts with dist < 0, per env per step",
        "active_max": int(active.max()),
        "active_mean": float(active.mean()),
        "capacity": _capacity(data, backend),
    }
    if nacon is not None:
        nacon = np.asarray(nacon)
        contacts["nacon_max"] = int(nacon.max())
        contacts["nacon_mean"] = float(nacon.mean())
        contacts["overflow"] = bool(nacon.max() >= contacts["capacity"]["value"])
    else:
        # jax computes every candidate contact, so there is no overflow to
        # report; ncon is the static buffer size, not a per-step count.
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
        print(
            f"check-terrain OK  {result['backend']}  "
            f"{result['steps_per_s']:.1f} steps/s  "
            f"{result['env_steps_per_s']:,.0f} env-steps/s  "
            f"active ncon max {c['active_max']} mean {c['active_mean']:.1f}  "
            f"-> {out}"
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
