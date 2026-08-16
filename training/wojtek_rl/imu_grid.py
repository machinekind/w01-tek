"""IMU robustness grid: gyro-bias and optional gyro-noise sweeps over
existing runs, scored on standing and straight-walk rollouts.

Run: python3 -m wojtek_rl.imu_grid --runs runs/<run1> [runs/<run2> ...]

The terrain v4.1 policy oscillated while standing on the real robot. The
oscillation was a limit cycle at about 25 Hz, half the 50 Hz control
rate, driven by gyro noise and loop latency. Nothing in the fixed battery
or the course benchmark perturbs the IMU, so the sim never showed it.
This grid adds the missing axis. It holds everything at the course
benchmark's nominals (flat floor, HEIGHT_CMD, no pushes) and varies only
the actor's gyro signal.

The bias rides on a mechanism the env already owns. Every reset writes
info["gyro_bias"], which holds zeros unless the run trained the bias DR,
and _build_obs adds it to the actor's gyro only. The critic path and the
physics never see it. The grid pins that key to a fixed vector every
step, exactly like the course follower pins info["command"]. That makes
the sweep deterministic, because a fixed vector replaces the reset-time
draw. It also works on any checkpoint of this project, because runs that
predate the bias DR carry zeros in the same pytree slot and nothing
retraces. No bias level needs an env rebuild or recompile. Everything
else the run trained with, white obs noise and the latency and encoder
draws, stays as stored in the run's config. A cell therefore measures
this policy, as deployed, plus a constant gyro offset. A real gyro has
such an offset, and the white training noise cannot represent it.

The optional white-noise axis (--noise-gyro) works differently. Noise
scales are baked into the jitted observation path from the env config, so
each value rebuilds the measurement env via load_checkpoint_policy's
env_overrides, one recompile per level. Use it to probe the stability
margin of a policy whose nominal cell looks clean. Raise the noise past
the training value and watch which policy's vibration blows up first.

Metrics per cell, over the post-settle window:
  - vibration: battery.vibration_index, the fraction of joint-velocity
    power above 5 Hz.
  - band_20_25: battery.band_power_fraction over 20-25 Hz, the band where
    the real-robot limit cycle lived. A standing robot with a healthy
    filter keeps this near zero.
  - falls, and the fall time when a seed went down.
  - vx_err_rms on the walk scenario. A bias must not silently break
    tracking even when nothing falls.

Output: runs/<run>/imu_grid/imu_grid.json per run, a table per run on
stdout, and one combined markdown report via --out. The grid has no
gates. It is a measurement. Compare cells across policies and against the
bias=0 baseline rows the grid always includes.
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

SETTLE_STEPS = 50  # the reset transient (1 s at ctrl_dt=0.02), excluded from metrics
NYQUIST_BAND_HZ = (20.0, 25.1)  # the real-robot limit cycle sat at 25 Hz
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def bias_vector(axis: str, level: float) -> np.ndarray:
    """Build the pinned info["gyro_bias"] vector, `level` rad/s on one axis."""
    vec = np.zeros(3)
    vec[AXIS_INDEX[axis]] = level
    return vec


def grid_cells(bias_levels, axes) -> list:
    """List the (axis, level) cells. The zero level collapses to one
    axis-less baseline cell, because a zero bias runs the same rollout on
    every axis."""
    cells = [("-", 0.0)] if 0.0 in bias_levels else []
    return cells + [(a, b) for b in bias_levels if b != 0.0 for a in axes]


def scenario_metrics(qvel_hist: np.ndarray, dt: float) -> dict:
    """Score one surviving rollout: the spectral scores plus absolute scale.

    Both spectral scores are fractions of total power, so a near-motionless
    stand can score high on microscopic buzz. qvel_rms (rad/s over all
    joints) is the guard against that, the same pairing battery.py uses. A
    high spectral score means a real oscillation only when qvel_rms shows
    the joints actually moving.
    """
    return {
        "vibration": round(vibration_index(qvel_hist, dt), 4),
        "band_20_25": round(
            band_power_fraction(qvel_hist, dt, *NYQUIST_BAND_HZ), 4
        ),
        "qvel_rms": round(float(np.sqrt(np.mean(qvel_hist**2))), 4),
    }


def _pin(state, cmd, bias, latency=None):
    """Pin command, gyro bias, and optionally control latency for the
    next step.

    This is the same in-place-dict mechanism as the course follower's
    _hold_command. Values change, structure does not, so nothing retraces.
    The command pin also zeroes steps_since_cmd so the env never resamples
    over us. The bias pin overwrites the reset-time draw with the cell's
    vector. The latency pin overwrites info["ctrl_delay"], the control
    latency the env drew for the episode, so a cell measures a chosen
    constant latency.
    The random draw lands on the worst case in only about one seed in six.
    """
    state.info["command"] = cmd
    state.info["steps_since_cmd"] = jp.zeros_like(state.info["steps_since_cmd"])
    state.info["gyro_bias"] = bias
    if latency is not None:
        state.info["ctrl_delay"] = jp.int32(latency)


def _rollout(env, reset, step, inf, cmd, n_steps, bias, seed, latency=None):
    """Roll out `n_steps` under a fixed command and pinned bias/latency.

    Returns (qvel_hist, vx_hist, fell_at). The histories exclude the
    first SETTLE_STEPS. fell_at is a step index, or None when the robot
    stayed up.
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
    """Run one (bias vector, latency) cell, a stand and a walk over `seeds`."""
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
    """Run the full grid for one run. Writes and returns its imu_grid.json.

    `lag_tau` > 0 swaps in the battery's explicit-PD substep loop
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
    """Write one markdown table over every run, for cross-policy comparison."""
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
                    help="gyro bias magnitudes in rad/s, where 0 is the "
                    "baseline cell (default: 0 0.05 0.1 0.2)")
    ap.add_argument("--axes", nargs="+", default=["x", "y", "z"],
                    choices=list(AXIS_INDEX),
                    help="body axes to bias, one cell each (default: x y z)")
    ap.add_argument("--noise-gyro", type=float, nargs="+", default=None,
                    help="absolute white gyro-noise scales to sweep. Each "
                    "value rebuilds the env (default: the run's own value "
                    "only)")
    ap.add_argument("--vib-gain", type=float, nargs="+", default=None,
                    help="gains for the vibration feedback on the actor's "
                    "gyro (obs_noise.gyro_vib). The policy's own torque jitter "
                    "comes back into its gyro through a resonator at half "
                    "the control rate. Each value rebuilds the env. Sweep "
                    "it to find the gain where standing goes unstable "
                    "(default: off)")
    ap.add_argument("--lag-tau", type=float, default=0.0,
                    help="first-order actuator-torque lag in seconds, via "
                    "the battery's explicit-PD path. 0 keeps the native "
                    "ideal actuators. Takes one value per invocation "
                    "because it recompiles the whole grid")
    ap.add_argument("--latency-substeps", type=int, nargs="+", default=None,
                    help="pin the control latency the env draws each episode "
                    "(info['ctrl_delay']) to these substep counts, one grid "
                    "axis each. The random draw lands on the worst case in "
                    "about one seed in six (default: the env's own draw "
                    "only)")
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

    # One env build per (noise, vib) pair. The baseline comes first, then
    # each perturbation on its own, so every cell isolates one mechanism.
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
