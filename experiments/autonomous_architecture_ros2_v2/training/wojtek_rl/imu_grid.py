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

The vibration gain (--vib-gain) is pinned again, the way the bias is.
The env copies obs_noise.gyro_vib into info["gyro_vib_gain"] at reset and
reads it there every step, so a cell just overwrites that scalar. Every
gain then runs inside one env build and one compile. That is what makes
--bisect-vib affordable. It bisects for the gain where the policy's stand
stops being motionless, a dozen or so stand rollouts on the same compiled
env. That critical gain is the policy's stability margin against the loop
from action to vibration to IMU. It is the number to compare across
policies. The battery's lagged step advances the same resonator, so a
gain acts in a lagged cell too.

The real robot does not hand the policy one fault at a time. It shakes,
its drives lag, and its gyro drifts, all at once. So the grid can compose
its axes. --lag-tau takes a list, and every lag value is run inside every
env build, next to the bias, latency and vibration cells that env build
already crosses. --cross-noise-vib adds the pairs of a noise level and a
vibration gain, on top of the single-mechanism cells. Only two of these
axes cost compile time now. Each noise level is one env build, and each
lag value is one JIT compile inside that build. A vibration gain costs
neither. Start from the single-mechanism cells to find where a policy is
weak, then compose only around that spot.

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
stdout, and one combined markdown report via --out. With --bisect-vib the
same JSON gains a "critical_gain" block. The grid has no gates. It is a
measurement. Compare cells across policies and against the bias=0
baseline rows the grid always includes.
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


def noise_vib_levels(noise_gyro=None, vib_gain=None, cross=False) -> list:
    """List the (noise, vib) pairs, one env build each.

    None means "leave that setting as the run trained it". The baseline,
    where both are left alone, always comes first, so every table starts
    from the row the others are read against.

    By default each level then appears on its own, so a cell that goes bad
    names the one thing that broke it. With `cross` on, every noise level
    is also paired with every vibration gain, because the real robot has
    both at the same time. Each pair is another env build and another
    compile, so the default keeps the grid cheap and the crossed grid is
    something you ask for.
    """
    noise = list(noise_gyro or [])
    vib = list(vib_gain or [])
    levels = [(None, None)]
    levels += [(n, None) for n in noise]
    levels += [(None, v) for v in vib]
    if cross:
        levels += [(n, v) for n in noise for v in vib]
    return levels


def env_builds(levels) -> list:
    """Group the (noise, vib) cells into the env builds that run them.

    Only the white-noise scale is baked into the compiled observation
    path. The vibration gain is pinned per step now, so every gain of one
    noise level shares that level's build. Returns (noise, [vib, ...])
    pairs in the order the levels first ask for them, so the untouched
    baseline build still leads.
    """
    builds = {}
    for noise, vib in levels:
        vibs = builds.setdefault(noise, [])
        if vib not in vibs:
            vibs.append(vib)
    return list(builds.items())


def lag_levels(lag_tau=None) -> list:
    """List the actuator-lag time constants to run inside one env build.

    0 means the env's own ideal actuators, and it leads when it is asked
    for, for the same reason the baseline leads above. Repeats are
    dropped, because each value costs a JIT compile. An empty list means
    the ideal actuators only.
    """
    out = []
    for tau in (lag_tau if lag_tau else [0.0]):
        tau = float(tau)
        if tau not in out:
            out.append(tau)
    return ([0.0] if 0.0 in out else []) + [t for t in out if t != 0.0]


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


def _pin(state, cmd, bias, latency=None, vib_gain=None):
    """Pin command, gyro bias, and optionally control latency and the
    vibration gain for the next step.

    This is the same in-place-dict mechanism as the course follower's
    _hold_command. Values change, structure does not, so nothing retraces.
    The command pin also zeroes steps_since_cmd so the env never resamples
    over us. The bias pin overwrites the reset-time draw with the cell's
    vector. The latency pin overwrites info["ctrl_delay"], the control
    latency the env drew for the episode, so a cell measures a chosen
    constant latency.
    The random draw lands on the worst case in only about one seed in six.
    The vibration pin overwrites info["gyro_vib_gain"], which reset filled
    from obs_noise.gyro_vib, so a cell measures a chosen gain on an env
    built for another one.
    """
    state.info["command"] = cmd
    state.info["steps_since_cmd"] = jp.zeros_like(state.info["steps_since_cmd"])
    state.info["gyro_bias"] = bias
    if latency is not None:
        state.info["ctrl_delay"] = jp.int32(latency)
    if vib_gain is not None:
        state.info["gyro_vib_gain"] = jp.float32(vib_gain)


def _rollout(env, reset, step, inf, cmd, n_steps, bias, seed, latency=None,
             vib_gain=None):
    """Roll out `n_steps` under a fixed command and pinned bias, latency
    and vibration gain.

    Returns (qvel_hist, vx_hist, fell_at). The histories exclude the
    first SETTLE_STEPS. fell_at is a step index, or None when the robot
    stayed up.
    """
    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    qvel_hist, vx_hist = [], []
    for i in range(n_steps):
        _pin(state, cmd, bias, latency, vib_gain)
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
          walk_steps, walk_vx, dt, latency=None, vib_gain=None):
    """Run one (bias vector, latency, vib gain) cell, a stand and a walk
    over `seeds`."""
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
                latency, vib_gain,
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


def _lag_str(lag) -> str:
    """A lag of 0 prints as "off", like an unset vib gain. It is the
    ideal-actuator cell."""
    return "off" if not lag else f"{lag:g}"


def _cell_row(cell: dict, noise, axis: str, bias: float, latency=None,
              vib=None, lag=0.0) -> str:
    def _f(d, key):
        v = d.get(key)
        return f"{v:>7.3f}" if v is not None else "      -"

    st, wk = cell["stand"], cell["walk"]
    noise_s = "own" if noise is None else f"{noise:g}"
    lat_s = "own" if latency is None else str(latency)
    vib_s = "off" if vib is None else f"{vib:g}"
    return (
        f"{noise_s:>5} {vib_s:>5} {lat_s:>4} {_lag_str(lag):>5} "
        f"{axis:>4} {bias:>5.2f}  "
        f"{_f(st, 'vibration_mean')} {_f(st, 'band_20_25_mean')} "
        f"{_f(st, 'qvel_rms_mean')} "
        f"{st['falls']:>2}/{st['seeds']}  "
        f"{_f(wk, 'vibration_mean')} {_f(wk, 'vx_err_rms_mean')} "
        f"{wk['falls']:>2}/{wk['seeds']}"
    )


_HEADER = (
    f"{'noise':>5} {'vib':>5} {'lat':>4} {'lag':>5} {'axis':>4} {'bias':>5}  "
    f"{'st.vib':>7} {'st.band':>7} {'st.rms':>7} {'falls':>5}  "
    f"{'wk.vib':>7} {'wk.verr':>7} {'falls':>5}"
)


def run_grid(run_dir: Path, bias_levels, axes, noise_levels, seeds,
             seed_base, stand_sec, walk_sec, walk_vx,
             latency_levels=(None,), lag_taus=(0.0,)) -> dict:
    """Run the full grid for one run. Writes and returns its imu_grid.json.

    `noise_levels` lists the (noise, vib) cells. env_builds groups them
    into one build per white-noise scale, because only that scale is baked
    into the compiled env. Inside a build the grid crosses `lag_taus` x
    the build's vibration gains x `latency_levels` x the bias cells, so a
    cell can carry several faults at once, the way the machine does.

    A lag of 0 keeps the env's own ideal actuators. A lag above 0 swaps in
    the battery's explicit-PD substep loop (make_lagged_rollout_fns), where
    the joint torque follows its target with a first-order delay, the way a
    real drive does. The env's actuators apply torque instantly and cannot
    show that delay. Each lag value is its own JIT compile in every env
    build, so a long lag list is slow to start. The lagged step advances
    the gyro-vib resonator on its own applied torque, so a vibration gain
    acts in a lagged cell too.
    """
    cells = []
    print(f"\nimu_grid -- {run_dir}")
    print(_HEADER)
    for noise, vibs in env_builds(noise_levels):
        overrides = None
        if noise is not None:
            overrides = {"obs_noise": {"gyro": noise}}
        run, env, ckpt, inf = load_checkpoint_policy(
            run_dir, env_overrides=overrides
        )
        dt = env.dt
        stand_steps = int(round(stand_sec / dt))
        walk_steps = int(round(walk_sec / dt))
        for lag_tau in lag_taus:
            if lag_tau > 0:
                reset_fn, step_fn = make_lagged_rollout_fns(env, lag_tau)
                reset, step = jax.jit(reset_fn), jax.jit(step_fn)
            else:
                reset, step = jax.jit(env.reset), jax.jit(env.step)
            for vib in vibs:
                for latency in latency_levels:
                    for axis, bias in grid_cells(bias_levels, axes):
                        cell = _cell(
                            env, reset, step, inf,
                            bias_vector(axis, bias) if bias else np.zeros(3),
                            seeds, seed_base, stand_steps, walk_steps,
                            walk_vx, dt, latency, vib,
                        )
                        entry = {"noise_gyro": noise, "vib_gain": vib,
                                 "latency": latency, "lag_tau": float(lag_tau),
                                 "axis": axis, "bias": bias, **cell}
                        cells.append(entry)
                        print(
                            _cell_row(cell, noise, axis, bias, latency, vib,
                                      lag_tau),
                            flush=True,
                        )
    results = {
        "run": run["run_name"],
        "checkpoint": ckpt.name,
        "seeds": seeds,
        "seed_base": seed_base,
        "stand_sec": stand_sec,
        "walk_sec": walk_sec,
        "walk_vx": walk_vx,
        "lag_taus": [float(t) for t in lag_taus],
        "band_hz": list(NYQUIST_BAND_HZ),
        "cells": cells,
    }
    write_run_json(run_dir, results)
    return results


def run_json_path(run_dir: Path) -> Path:
    return run_dir / "imu_grid" / "imu_grid.json"


def write_run_json(run_dir: Path, results: dict) -> dict:
    out = run_json_path(run_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    return results


def bisect_threshold(probe, lo, hi, tol, max_iters=20) -> dict:
    """Find the gain where `probe` flips from stable to unstable.

    `probe(gain)` answers whether that gain is unstable. It may return a
    plain bool, or a (unstable, qvel_rms) pair when it has a measured
    number to record. The search assumes the answer only ever flips once,
    from stable to unstable, as the gain rises, which is what a stability
    margin means.

    The bracket has to hold that flip. The low end must come back stable
    and the high end unstable. When it does not, "critical_gain" stays
    empty and "outcome" says which side the answer is on, either "above
    the bracket" or "below the bracket". A bracket that never flips has
    no threshold in it, and a guessed number would read like a measured
    one. Saying "above 0.5" is still a real result, so "reason" carries
    the gain and the number the failing end measured. Otherwise the
    bracket is halved until it is narrower than `tol`, and the critical
    gain is its midpoint.

    "probes" holds every question asked, in order, including the two
    bracket checks. Each entry is the gain, the answer, and whatever the
    probe measured there.
    """
    probes = []

    def ask(gain):
        answer = probe(gain)
        unstable, qvel_rms = (
            answer if isinstance(answer, tuple) else (answer, None)
        )
        unstable = bool(unstable)
        probes.append((round(float(gain), 6), unstable, qvel_rms))
        return unstable

    def measured():
        """What the last probe measured, for the reason line."""
        qvel_rms = probes[-1][2]
        return "" if qvel_rms is None else f", qvel_rms {qvel_rms:g}"

    lo, hi = float(lo), float(hi)
    result = {"lo": lo, "hi": hi, "tol": float(tol), "critical_gain": None,
              "probes": probes, "reason": None, "outcome": "bracketed"}
    if not lo < hi:
        result["outcome"] = "empty bracket"
        result["reason"] = f"lo {lo:g} is not below hi {hi:g}"
        return result
    if ask(lo):
        result["outcome"] = "below the bracket"
        result["reason"] = (
            f"lo probe unstable at gain {lo:g}{measured()}, so the critical "
            f"gain is below {lo:g}"
        )
        return result
    if not ask(hi):
        result["outcome"] = "above the bracket"
        result["reason"] = (
            f"hi probe stable at gain {hi:g}{measured()}, so the critical "
            f"gain is above {hi:g}"
        )
        return result
    for _ in range(max_iters):
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        if ask(mid):
            hi = mid
        else:
            lo = mid
    result.update(lo=round(lo, 6), hi=round(hi, 6),
                  critical_gain=round(0.5 * (lo + hi), 6))
    return result


def stand_probe(env, reset, step, inf, seeds, seed_base, stand_steps, dt,
                threshold):
    """Build the probe bisect_threshold asks. It answers whether the
    stand is unstable at a given vibration gain.

    One stand rollout per seed, at the pinned gain, no walk. A seed counts
    as unstable when it falls, or when its qvel_rms over the measured
    window is above `threshold`. The cell is unstable when any seed falls,
    or when a majority of the seeds are over the threshold. So one noisy
    seed out of three still counts as stable, and two do not.

    Also returns the mean qvel_rms of the seeds that stayed up, so the
    probe list shows how the stand grew from motionless to shaking.
    """
    stand_cmd = jp.array([0.0, 0.0, 0.0, HEIGHT_CMD])
    zero_bias = jp.zeros(3)
    n = SETTLE_STEPS + stand_steps

    def probe(gain):
        rms, falls = [], 0
        for s in range(seeds):
            qvel, _, fell_at = _rollout(
                env, reset, step, inf, stand_cmd, n, zero_bias,
                seed_base + s, None, gain,
            )
            if fell_at is not None:
                falls += 1
                continue
            rms.append(scenario_metrics(qvel, dt)["qvel_rms"])
        over = falls + sum(1 for r in rms if r > threshold)
        unstable = falls > 0 or 2 * over > seeds
        mean_rms = round(float(np.mean(rms)), 4) if rms else None
        return unstable, mean_rms

    return probe


def run_bisect(run_dir: Path, lo, hi, tol, threshold, seeds, seed_base,
               stand_sec) -> tuple:
    """Bisect the critical vibration gain of one run's policy.

    Returns (run name, the bisect_threshold result plus the settings that
    produced it). One env build and one compile cover the whole search,
    because the gain is pinned per step.
    """
    run, env, ckpt, inf = load_checkpoint_policy(run_dir)
    dt = env.dt
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    probe = stand_probe(
        env, reset, step, inf, seeds, seed_base,
        int(round(stand_sec / dt)), dt, threshold,
    )
    result = bisect_threshold(probe, lo, hi, tol)
    result.update({
        "checkpoint": ckpt.name,
        "threshold": float(threshold),
        "seeds": seeds,
        "seed_base": seed_base,
        "stand_sec": stand_sec,
    })
    return run["run_name"], result


def critical_gain_str(crit: dict) -> str:
    """The headline value, or where it is when the bracket missed it."""
    if crit["critical_gain"] is not None:
        return f"{crit['critical_gain']:g}"
    if crit.get("outcome") == "above the bracket":
        return f"above {crit['hi']:g}"
    if crit.get("outcome") == "below the bracket":
        return f"below {crit['lo']:g}"
    return "none"


def bisect_line(run_name: str, crit: dict) -> str:
    """One line per run, for stdout."""
    where = (
        f"bracket {crit['lo']:g}-{crit['hi']:g}"
        if crit["critical_gain"] is not None else crit["reason"]
    )
    return (
        f"critical vib gain -- {run_name}: {critical_gain_str(crit)}. "
        f"{where}. Threshold qvel_rms {crit['threshold']:g} rad/s, "
        f"{crit['seeds']} seeds, {len(crit['probes'])} probes."
    )


def merge_critical_gain(run_dir: Path, results, run_name: str,
                        crit: dict) -> dict:
    """Put `crit` under "critical_gain" in the run's imu_grid.json.

    `results` is the grid this invocation just ran, or None when only the
    bisection ran. In that case the file on disk is read and updated, so a
    bisection never throws away cells an earlier grid measured.
    """
    if results is None:
        path = run_json_path(run_dir)
        results = (
            json.loads(path.read_text()) if path.exists() else {"cells": []}
        )
    results.setdefault("run", run_name)
    results["critical_gain"] = crit
    return write_run_json(run_dir, results)


def _critical_gain_table(all_results: list) -> list:
    """The critical-gain summary, or nothing when no run was bisected.

    It leads the report because it is the one number to compare across
    policies. Each cell below it is a single measurement, and no single
    measurement is a verdict.
    """
    rows = [r for r in all_results if r.get("critical_gain")]
    if not rows:
        return []
    lines = [
        "## Critical gain per run",
        "",
        "The gyro-vib gain where the stand stops being motionless, found "
        "by bisection. Higher is a wider stability margin. A gain given as "
        "\"above\" or \"below\" a number means the bracket did not hold the "
        "flip, so the margin is on that side of it. The threshold is the "
        "standing qvel_rms in rad/s that counts as unstable, and it is a "
        "chosen number, so only compare runs bisected at the same one.",
        "",
        "| run | critical gain | bracket | threshold (qvel_rms) | seeds |",
        "|---|---|---|---|---|",
    ]
    for res in rows:
        c = res["critical_gain"]
        lines.append(
            f"| {res['run']} | {critical_gain_str(c)} "
            f"| {c['lo']:g}-{c['hi']:g} | {c['threshold']:g} | {c['seeds']} |"
        )
    return lines + [""]


def write_report(out: Path, all_results: list) -> None:
    """Write one markdown table over every run, for cross-policy comparison."""
    lines = ["# IMU robustness grid", ""]
    lines += _critical_gain_table(all_results)
    if any(res.get("cells") for res in all_results):
        lines += [
            "## Cells",
            "",
            "| run | noise | vib | lat | lag | axis | bias | stand vib | "
            "stand band20-25 | stand qvel_rms | stand falls | walk vib | "
            "walk vx_err | walk falls |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    for res in all_results:
        for c in res.get("cells", ()):
            st, wk = c["stand"], c["walk"]

            def _m(d, key):
                v = d.get(key)
                return f"{v:.3f}" if v is not None else "-"

            noise = "own" if c["noise_gyro"] is None else f"{c['noise_gyro']:g}"
            vib = "off" if c.get("vib_gain") is None else f"{c['vib_gain']:g}"
            lat = "own" if c.get("latency") is None else str(c["latency"])
            lag = _lag_str(c.get("lag_tau"))
            lines.append(
                f"| {res['run']} | {noise} | {vib} | {lat} | {lag} "
                f"| {c['axis']} | {c['bias']:g} "
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
                    "gyro (info['gyro_vib_gain'], set at reset from "
                    "obs_noise.gyro_vib). The policy's own torque jitter "
                    "comes back into its gyro through a resonator at half "
                    "the control rate. Each value is pinned per step, so no "
                    "value rebuilds the env. Sweep it to see where standing "
                    "goes unstable, or use --bisect-vib to find that gain "
                    "properly (default: off)")
    ap.add_argument("--cross-noise-vib", action="store_true",
                    help="also run every noise level paired with every "
                    "vibration gain, on top of the one-at-a-time cells. The "
                    "real robot has both at once. The pairs cost rollouts "
                    "and no env build (default: off)")
    ap.add_argument("--lag-tau", type=float, nargs="+", default=[0.0],
                    help="first-order actuator-torque lags in seconds, via "
                    "the battery's explicit-PD path. 0 keeps the native "
                    "ideal actuators. Every value runs inside every env "
                    "build, and each one costs a JIT compile there "
                    "(default: 0)")
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
    ap.add_argument("--bisect-vib", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="after the grid, bisect this bracket for the "
                    "critical vibration gain, the gain where the policy's "
                    "stand stops being motionless. LO must come back stable "
                    "and HI unstable, or the search reports that instead of "
                    "a number. Stand rollouts only, no walk (default: off)")
    ap.add_argument("--bisect-only", action="store_true",
                    help="skip the grid and only bisect. An existing "
                    "imu_grid.json keeps its cells. Only the critical_gain "
                    "block is rewritten (default: off)")
    ap.add_argument("--bisect-tol", type=float, default=0.02,
                    help="stop bisecting once the bracket is this narrow "
                    "(default 0.02)")
    ap.add_argument("--bisect-threshold", type=float, default=0.1,
                    help="a probed gain counts as unstable when the standing "
                    "qvel_rms is above this many rad/s in a majority of the "
                    "--seeds seeds, or when any seed falls. A quiet stand "
                    "measures 0.01 to 0.07 rad/s and a real limit cycle "
                    "reaches about 3, so 0.1 is clear of both. It is a "
                    "chosen number, so only compare policies bisected at the "
                    "same one (default 0.1)")
    ap.add_argument("--out", default=None,
                    help="combined markdown report path (optional)")
    args = ap.parse_args()

    if args.bisect_only and not args.bisect_vib:
        ap.error("--bisect-only needs --bisect-vib LO HI")
    # One env build per noise level, one JIT compile per lag value inside
    # each build. The vibration gain is pinned, so it costs neither.
    noise_levels = noise_vib_levels(
        args.noise_gyro, args.vib_gain, args.cross_noise_vib
    )
    lag_taus = lag_levels(args.lag_tau)
    latency_levels = [None] + (args.latency_substeps or [])
    all_results = []
    for run in args.runs:
        run_dir = Path(run)
        results = None
        if not args.bisect_only:
            results = run_grid(
                run_dir, args.bias_levels, args.axes, noise_levels,
                args.seeds, args.seed_base, args.stand_sec, args.walk_sec,
                args.walk_vx, latency_levels, lag_taus,
            )
        if args.bisect_vib:
            lo, hi = args.bisect_vib
            run_name, crit = run_bisect(
                run_dir, lo, hi, args.bisect_tol, args.bisect_threshold,
                args.seeds, args.seed_base, args.stand_sec,
            )
            print(bisect_line(run_name, crit), flush=True)
            results = merge_critical_gain(run_dir, results, run_name, crit)
        all_results.append(results)
    if args.out:
        write_report(Path(args.out), all_results)


if __name__ == "__main__":
    main()
