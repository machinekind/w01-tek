"""IMU robustness grid: gyro-bias (and optionally gyro-noise) sweeps over
existing runs, scored on standing and straight-walk rollouts.

Run: python3 -m wojtek_rl.imu_grid --runs runs/<run1> [runs/<run2> ...]

Why it exists: the terrain v4.1 policy oscillated while STANDING on the
real robot -- a limit cycle at ~25 Hz (half the 50 Hz control rate),
driven by gyro noise and loop latency -- and nothing in the fixed battery
or the course benchmark perturbs the IMU, so nothing detected it in sim.
This grid is the controlled axis that was missing: it holds everything at
the course benchmark's nominals (flat floor, HEIGHT_CMD, no pushes) and
varies exactly one thing, the actor's gyro signal.

How the bias is applied: the env already owns the mechanism.  Every reset
writes info["gyro_bias"] (zeros unless the run trained the bias DR), and
_build_obs adds it to the ACTOR's gyro only -- the critic path and the
physics never see it.  The grid PINS that key to a fixed vector every
step, exactly like the course follower pins info["command"], so:
  - the sweep is deterministic (a fixed vector, not a reset-time draw),
  - it works on ANY checkpoint of this project, including runs that
    predate the bias DR (their info carries zeros -- same pytree, no
    retrace), and
  - no env rebuild or recompile per bias level.
Everything else the run trained with (white obs noise, latency and
encoder draws) stays exactly as stored in the run's config, so a cell is
"this policy, as deployed, plus a zero-rate gyro offset" -- the property
a real gyro has and the white training noise cannot represent.

The optional gyro white-noise axis (--noise-gyro) is different: noise
scales are baked into the jitted observation path from the env config, so
each value rebuilds the measurement env via load_checkpoint_policy's
env_overrides (one recompile per level).  Use it to probe the stability
MARGIN of a policy whose nominal cell looks clean: crank the noise past
the training value and watch which policy's vibration blows up first.

Metrics per cell, over the post-settle window:
  - vibration: battery.vibration_index (>5 Hz joint-velocity power),
  - band_20_25: battery.band_power_fraction over 20-25 Hz -- the
    near-Nyquist band where the real-robot limit cycle lived; a standing
    robot with a healthy filter keeps this near zero,
  - falls, and the fall time when a seed went down,
  - vx_err_rms on the walk scenario (bias must not silently break
    tracking even when nothing falls).

Output: runs/<run>/imu_grid/imu_grid.json per run, a table per run on
stdout, and one combined markdown report via --out.  No gates: the grid
is a measurement, compared across policies (e.g. terrain_blind_v4_1 vs
v5/v5_1) and against the bias=0 baseline rows it always includes.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from wojtek_rl.battery import (
    band_power_fraction,
    load_checkpoint_policy,
    make_lagged_rollout_fns,
    vibration_index,
)
from wojtek_rl.courses.spec import HEIGHT_CMD

SETTLE_STEPS = 50  # 1 s at ctrl_dt=0.02; reset transient, excluded from metrics
NYQUIST_BAND_HZ = (20.0, 25.1)  # the real-robot limit cycle sat at 25 Hz
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def bias_vector(axis: str, level: float) -> np.ndarray:
    """The pinned info["gyro_bias"] vector: `level` rad/s on one axis."""
    vec = np.zeros(3)
    vec[AXIS_INDEX[axis]] = level
    return vec


def grid_cells(bias_levels, axes) -> list:
    """[(axis, level)] with the zero level collapsed to one baseline cell:
    a zero bias is axis-less, and duplicating it per axis would just run
    the same rollout three times."""
    cells = [("-", 0.0)] if 0.0 in bias_levels else []
    return cells + [(a, b) for b in bias_levels if b != 0.0 for a in axes]


def scenario_metrics(qvel_hist: np.ndarray, dt: float) -> dict:
    """Spectral scores plus absolute scale for one surviving rollout.

    qvel_rms (rad/s over all joints) is the guard battery.py pairs with the
    vibration ratio: both spectral scores are FRACTIONS of total power, so a
    near-motionless stand can score high on microscopic buzz. A high ratio
    only means a real oscillation when qvel_rms says the joints actually
    move.
    """
    return {
        "vibration": round(vibration_index(qvel_hist, dt), 4),
        "band_20_25": round(
            band_power_fraction(qvel_hist, dt, *NYQUIST_BAND_HZ), 4
        ),
        "qvel_rms": round(float(np.sqrt(np.mean(qvel_hist**2))), 4),
    }


def _pin(state, cmd, bias, latency=None):
    """Pin command, gyro bias and (optionally) control latency for the
    next step.

    Same in-place-dict mechanism as the course follower's _hold_command
    (values replaced, structure unchanged -- no retrace): the command pin
    also zeroes steps_since_cmd so the env never resamples over us, the
    bias pin overwrites the reset-time draw with the cell's vector, and
    the latency pin overwrites info["ctrl_delay"] -- the per-episode
    substep-latency draw -- so a cell can measure the WORST-case constant
    latency instead of a random draw that only ~1/6 of seeds land on.
    """
    state.info["command"] = cmd
    state.info["steps_since_cmd"] = jp.zeros_like(state.info["steps_since_cmd"])
    state.info["gyro_bias"] = bias
    if latency is not None:
        state.info["ctrl_delay"] = jp.int32(latency)


def _rollout(env, reset, step, inf, cmd, n_steps, bias, seed, latency=None):
    """`n_steps` under a fixed command and pinned bias/latency.

    Returns (qvel_hist, vx_hist, fell_at) with the first SETTLE_STEPS
    excluded from the histories; fell_at is a step index or None.
    """
    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    qvel_hist, vx_hist = [], []
    for i in range(n_steps):
        _pin(state, cmd, bias, latency)
        rng, k = jax.random.split(rng)
        act, _ = inf(state.obs, k)
        state = step(state, act)
        if float(state.done):
            return None, None, i
        if i >= SETTLE_STEPS:
            d = state.data
            qvel_hist.append(np.asarray(d.qvel[env._vadr]))
            vx_hist.append(float(np.asarray(env._local_linvel(d))[0]))
    return np.asarray(qvel_hist), np.asarray(vx_hist), None


def _cell(env, reset, step, inf, bias, seeds, seed_base, stand_steps,
          walk_steps, walk_vx, dt, latency=None):
    """One (bias vector, latency) cell: stand and walk over `seeds`."""
    stand_cmd = jp.array([0.0, 0.0, 0.0, HEIGHT_CMD])
    walk_cmd = jp.array([walk_vx, 0.0, 0.0, HEIGHT_CMD])
    bias_jp = jp.asarray(bias)
    out = {}
    for scen, cmd, steps in (
        ("stand", stand_cmd, SETTLE_STEPS + stand_steps),
        ("walk", walk_cmd, SETTLE_STEPS + walk_steps),
    ):
        rows, fell = [], []
        for s in range(seeds):
            qvel, vx, fell_at = _rollout(
                env, reset, step, inf, cmd, steps, bias_jp, seed_base + s,
                latency,
            )
            if fell_at is not None:
                fell.append(round(fell_at * dt, 2))
                continue
            row = scenario_metrics(qvel, dt)
            if scen == "walk":
                row["vx_err_rms"] = round(
                    float(np.sqrt(np.mean((vx - walk_vx) ** 2))), 4
                )
            rows.append(row)
        agg = {"seeds": seeds, "falls": len(fell), "fell_at_s": fell}
        for key in rows[0] if rows else ():
            vals = [r[key] for r in rows]
            agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{key}_max"] = round(float(np.max(vals)), 4)
        out[scen] = agg
    return out


def _cell_row(cell: dict, noise, axis: str, bias: float, latency=None,
              vib=None) -> str:
    def _f(d, key):
        v = d.get(key)
        return f"{v:>7.3f}" if v is not None else "      -"

    st, wk = cell["stand"], cell["walk"]
    noise_s = "own" if noise is None else f"{noise:g}"
    lat_s = "own" if latency is None else str(latency)
    vib_s = "off" if vib is None else f"{vib:g}"
    return (
        f"{noise_s:>5} {vib_s:>5} {lat_s:>4} {axis:>4} {bias:>5.2f}  "
        f"{_f(st, 'vibration_mean')} {_f(st, 'band_20_25_mean')} "
        f"{_f(st, 'qvel_rms_mean')} "
        f"{st['falls']:>2}/{st['seeds']}  "
        f"{_f(wk, 'vibration_mean')} {_f(wk, 'vx_err_rms_mean')} "
        f"{wk['falls']:>2}/{wk['seeds']}"
    )


_HEADER = (
    f"{'noise':>5} {'vib':>5} {'lat':>4} {'axis':>4} {'bias':>5}  "
    f"{'st.vib':>7} {'st.band':>7} {'st.rms':>7} {'falls':>5}  "
    f"{'wk.vib':>7} {'wk.verr':>7} {'falls':>5}"
)


def run_grid(run_dir: Path, bias_levels, axes, noise_levels, seeds,
             seed_base, stand_sec, walk_sec, walk_vx,
             latency_levels=(None,), lag_tau=0.0) -> dict:
    """The full grid for one run; writes and returns its imu_grid.json.

    `lag_tau` > 0 swaps in battery's explicit-PD substep loop
    (make_lagged_rollout_fns). The joint torque then follows its target
    with a first-order delay, the way a real drive does. The env's own
    actuators apply torque instantly and cannot show that delay.
    """
    cells = []
    print(f"\nimu_grid -- {run_dir}"
          + (f" (lag_tau={lag_tau:g}s)" if lag_tau > 0 else ""))
    print(_HEADER)
    for noise, vib in noise_levels:
        overrides = {}
        if noise is not None:
            overrides.setdefault("obs_noise", {})["gyro"] = noise
        if vib is not None:
            # The resonator updates inside env.step, so its gain has to
            # enter through the env config. Pinning an info value, the way
            # bias and latency enter, cannot switch it on.
            overrides.setdefault("obs_noise", {})["gyro_vib"] = vib
        run, env, ckpt, inf = load_checkpoint_policy(
            run_dir, env_overrides=overrides or None
        )
        if lag_tau > 0:
            reset_fn, step_fn = make_lagged_rollout_fns(env, lag_tau)
            reset, step = jax.jit(reset_fn), jax.jit(step_fn)
        else:
            reset, step = jax.jit(env.reset), jax.jit(env.step)
        dt = env.dt
        stand_steps = int(round(stand_sec / dt))
        walk_steps = int(round(walk_sec / dt))
        for latency in latency_levels:
            for axis, bias in grid_cells(bias_levels, axes):
                cell = _cell(
                    env, reset, step, inf, bias_vector(axis, bias) if bias
                    else np.zeros(3), seeds, seed_base, stand_steps,
                    walk_steps, walk_vx, dt, latency,
                )
                entry = {"noise_gyro": noise, "vib_gain": vib,
                         "latency": latency, "axis": axis, "bias": bias,
                         **cell}
                cells.append(entry)
                print(_cell_row(cell, noise, axis, bias, latency, vib),
                      flush=True)
    results = {
        "run": run["run_name"],
        "checkpoint": ckpt.name,
        "seeds": seeds,
        "seed_base": seed_base,
        "stand_sec": stand_sec,
        "walk_sec": walk_sec,
        "walk_vx": walk_vx,
        "lag_tau": lag_tau,
        "band_hz": list(NYQUIST_BAND_HZ),
        "cells": cells,
    }
    out = run_dir / "imu_grid" / "imu_grid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    return results


def write_report(out: Path, all_results: list) -> None:
    """One markdown table over every run, for cross-policy comparison."""
    lines = [
        "# IMU robustness grid",
        "",
        "| run | noise | vib | lat | axis | bias | stand vib | stand band20-25 | "
        "stand qvel_rms | stand falls | walk vib | walk vx_err | walk falls |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for res in all_results:
        for c in res["cells"]:
            st, wk = c["stand"], c["walk"]

            def _m(d, key):
                v = d.get(key)
                return f"{v:.3f}" if v is not None else "-"

            noise = "own" if c["noise_gyro"] is None else f"{c['noise_gyro']:g}"
            vib = "off" if c.get("vib_gain") is None else f"{c['vib_gain']:g}"
            lat = "own" if c.get("latency") is None else str(c["latency"])
            lines.append(
                f"| {res['run']} | {noise} | {vib} | {lat} | {c['axis']} | {c['bias']:g} "
                f"| {_m(st, 'vibration_mean')} | {_m(st, 'band_20_25_mean')} "
                f"| {_m(st, 'qvel_rms_mean')} "
                f"| {st['falls']}/{st['seeds']} | {_m(wk, 'vibration_mean')} "
                f"| {_m(wk, 'vx_err_rms_mean')} | {wk['falls']}/{wk['seeds']} |"
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\nreport -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Gyro bias/noise robustness grid; see wojtek_rl.imu_grid",
    )
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories (e.g. runs/<name>)")
    ap.add_argument("--bias-levels", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2],
                    help="gyro bias magnitudes in rad/s (default: 0 0.05 0.1 "
                    "0.2); 0 is the baseline cell")
    ap.add_argument("--axes", nargs="+", default=["x", "y", "z"],
                    choices=list(AXIS_INDEX),
                    help="body axes to bias, one cell each (default: x y z)")
    ap.add_argument("--noise-gyro", type=float, nargs="+", default=None,
                    help="absolute white gyro-noise scales to sweep; each "
                    "value rebuilds the env (default: the run's own value "
                    "only)")
    ap.add_argument("--vib-gain", type=float, nargs="+", default=None,
                    help="gains for the gyro-vib feedback corruption "
                    "(obs_noise.gyro_vib): the policy's own torque jitter "
                    "comes back into its gyro through a resonator at half "
                    "the control rate. Each value rebuilds the env. Sweep "
                    "it to find the gain where standing goes unstable "
                    "(default: off)")
    ap.add_argument("--lag-tau", type=float, default=0.0,
                    help="first-order actuator-torque lag in seconds "
                    "(battery's explicit-PD path); 0 = the native ideal "
                    "actuators. One value per invocation, it recompiles "
                    "the whole grid")
    ap.add_argument("--latency-substeps", type=int, nargs="+", default=None,
                    help="pin the per-episode control-latency draw "
                    "(info['ctrl_delay']) to these substep counts, one grid "
                    "axis each; the random draw only lands a seed on the "
                    "worst case ~1/6 of the time (default: the env's own "
                    "draw only)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="rollouts per cell and scenario (default 3)")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--stand-sec", type=float, default=10.0,
                    help="measured standing window per rollout (default 10)")
    ap.add_argument("--walk-sec", type=float, default=10.0,
                    help="measured walking window per rollout (default 10)")
    ap.add_argument("--walk-vx", type=float, default=0.4,
                    help="walk scenario's commanded speed (default 0.4)")
    ap.add_argument("--out", default=None,
                    help="combined markdown report path (optional)")
    args = ap.parse_args()

    # One env build per (noise, vib) pair: baseline first, then each single
    # perturbation on its own so a cell isolates one mechanism.
    noise_levels = [(None, None)]
    noise_levels += [(n, None) for n in (args.noise_gyro or [])]
    noise_levels += [(None, v) for v in (args.vib_gain or [])]
    latency_levels = [None] + (args.latency_substeps or [])
    all_results = []
    for run in args.runs:
        all_results.append(run_grid(
            Path(run), args.bias_levels, args.axes, noise_levels,
            args.seeds, args.seed_base, args.stand_sec, args.walk_sec,
            args.walk_vx, latency_levels, args.lag_tau,
        ))
    if args.out:
        write_report(Path(args.out), all_results)


if __name__ == "__main__":
    main()
