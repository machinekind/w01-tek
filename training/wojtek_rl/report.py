"""Evaluation report: battery + torque/power/foot-force/termination
metrics consolidated into one machine-comparable artifact per checkpoint.

Run: ./run.sh report --run runs/<name>

Writes runs/<name>/eval_report.json (everything downstream diffs against
this) and runs/<name>/eval_report.md (human-readable table). Reuses
battery.py's scenario rollouts -- report.py reduces the same per-step
signals (`actuator_force`, `qvel`, `base_accel`) that rollout() already
records, so `./run.sh report` costs about the same wall-clock as
`./run.sh battery`, not double.

Foot-force proxy: this env has no direct contact-force read (the battery's
own foot_contact/slip metrics use a geom-height heuristic, see
base.py._foot_contact) -- see `foot_force_proxy()` below for the documented
stand-in.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import jax
import numpy as np

from wojtek_rl.battery import (
    battery_scenarios,
    load_checkpoint_policy,
    rollout,
    scenario_result,
)

# -- pure metric functions ---------------------------------------------
# numpy arrays / plain dicts in, dict out -- no env, jax rollout or
# checkpoint needed, so these are unit-tested directly in test_report.py.


def torque_percentiles(actuator_force) -> dict:
    """p50/p90/p99 and max of |actuator_force| (N*m), pooled over every
    joint and step in `actuator_force` (shape (steps, n_joints) or (steps,)).
    """
    a = np.abs(np.asarray(actuator_force, dtype=float)).ravel()
    if a.size == 0:
        return {"p50": None, "p90": None, "p99": None, "max": None}
    return {
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def power_percentiles(actuator_force, joint_vel) -> dict:
    """p50/p90/p99 of |actuator_force * joint_vel| (W, per joint per step),
    plus the mean instantaneous total power (summed over joints, then
    averaged over steps). `actuator_force`/`joint_vel` must have matching
    shape (steps, n_joints) -- battery.rollout()'s `actuator_force`/`qvel`
    are already joint-aligned (both indexed in actuator order, see
    base.py._vadr).
    """
    force = np.asarray(actuator_force, dtype=float)
    vel = np.asarray(joint_vel, dtype=float)
    per_joint = np.abs(force * vel)
    if per_joint.size == 0:
        return {"p50": None, "p90": None, "p99": None, "mean_total": None}
    total = per_joint.sum(axis=-1) if per_joint.ndim > 1 else per_joint
    flat = per_joint.ravel()
    return {
        "p50": float(np.percentile(flat, 50)),
        "p90": float(np.percentile(flat, 90)),
        "p99": float(np.percentile(flat, 99)),
        "mean_total": float(total.mean()),
    }


# Foot-force proxy. The env has no ground-reaction-force sensor, so this uses
# the base IMU: peak baseline-subtracted vertical acceleration times total
# mass (F = m*a). It tracks impact loading in relative terms; it is not a
# calibrated force at the foot.
def foot_force_proxy(base_accel_z, total_mass: float, gravity: float = 9.81) -> dict:
    """Peak proxy force from base vertical-acceleration spikes. See the
    module-level note above for what this is (and isn't) measuring."""
    a = np.asarray(base_accel_z, dtype=float)
    if a.size == 0:
        return {"peak_accel_mps2": None, "peak_force_n": None}
    peak_accel = float(np.abs(np.abs(a) - gravity).max())
    return {
        "peak_accel_mps2": peak_accel,
        "peak_force_n": peak_accel * float(total_mass),
    }


def termination_summary(events: list) -> dict:
    """Reduce per-scenario termination events into a fall/reason breakdown.

    Each item of `events`: {"scenario", "fell_at", "height", "gravity_z",
    "min_height", "max_tilt_gz"}. `height`/`gravity_z` are the base height
    / local-frame gravity z-component (base.py._gravity_body) at the step
    that ended the scenario (None if it never fell -- see
    battery.rollout()'s `term` return). The fall reason is inferred from
    which of the env's two termination conditions (env.py's
    `fall = (qpos[2] < min_height) | (gravity[2] > max_tilt_gz)`) tripped:
    "height", "tilt", "both", or "unknown" if neither threshold explains a
    `done=True` fall (e.g. episode-length timeout counted as done upstream).
    """
    per_scenario = {}
    fall_reasons = {"height": 0, "tilt": 0, "both": 0, "unknown": 0}
    fall_count = 0
    for ev in events:
        name = ev["scenario"]
        if ev["fell_at"] is None:
            per_scenario[name] = {"fell": False, "fell_at": None, "reason": None}
            continue
        fall_count += 1
        is_height = ev["height"] is not None and ev["height"] < ev["min_height"]
        is_tilt = ev["gravity_z"] is not None and ev["gravity_z"] > ev["max_tilt_gz"]
        if is_height and is_tilt:
            reason = "both"
        elif is_height:
            reason = "height"
        elif is_tilt:
            reason = "tilt"
        else:
            reason = "unknown"
        fall_reasons[reason] += 1
        per_scenario[name] = {"fell": True, "fell_at": ev["fell_at"], "reason": reason}
    return {
        "scenarios_run": len(events),
        "fall_count": fall_count,
        "fall_reason_counts": fall_reasons,
        "per_scenario": per_scenario,
    }


def assemble_report(
    run_name: str,
    checkpoint: str,
    battery: dict,
    torque: dict,
    power: dict,
    foot_force: dict,
    termination: dict,
    timestamp: str | None = None,
) -> dict:
    """Merge the computed sections into the documented eval_report.json
    schema: run, checkpoint, battery, torque, power, foot_force_proxy,
    termination, timestamp."""
    return {
        "run": run_name,
        "checkpoint": checkpoint,
        "battery": battery,
        "torque": torque,
        "power": power,
        "foot_force_proxy": foot_force,
        "termination": termination,
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
    }


# -- report assembly (needs a checkpoint + jax rollout) ------------------


def build_report(run_dir: Path) -> dict:
    """Run the battery once, reducing its rollouts into the full report:
    battery table + torque/power percentiles + foot-force proxy +
    termination summary."""
    run, env, ckpt, inf, foot_radius = load_checkpoint_policy(run_dir)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    fall_cfg = env._config.fall
    total_mass = float(env.mj_model.body_mass.sum())
    gravity_mag = float(abs(env.mj_model.opt.gravity[2]))

    battery = {"run": run["run_name"], "checkpoint": ckpt.name}
    force_chunks, vel_chunks, accel_z_chunks = [], [], []
    term_events = []
    for name, (cmd_at, n) in battery_scenarios().items():
        rec, fell_at, term = rollout(env, reset, step, inf, cmd_at, n, foot_radius)
        battery[name] = scenario_result(name, rec, fell_at, env.dt)
        if rec["actuator_force"].size:
            force_chunks.append(rec["actuator_force"])
            vel_chunks.append(rec["qvel"])
            accel_z_chunks.append(rec["base_accel"][:, 2])
        term_events.append(
            {
                "scenario": name,
                "fell_at": fell_at,
                "height": term["height"] if term else None,
                "gravity_z": term["gravity_z"] if term else None,
                "min_height": float(fall_cfg.min_height),
                "max_tilt_gz": float(fall_cfg.max_tilt_gz),
            }
        )

    all_force = np.concatenate(force_chunks, axis=0) if force_chunks else np.zeros((0, 12))
    all_vel = np.concatenate(vel_chunks, axis=0) if vel_chunks else np.zeros((0, 12))
    all_accel_z = np.concatenate(accel_z_chunks, axis=0) if accel_z_chunks else np.zeros((0,))

    return assemble_report(
        run_name=run["run_name"],
        checkpoint=ckpt.name,
        battery=battery,
        torque=torque_percentiles(all_force),
        power=power_percentiles(all_force, all_vel),
        foot_force=foot_force_proxy(all_accel_z, total_mass, gravity_mag),
        termination=termination_summary(term_events),
    )


# -- markdown rendering ---------------------------------------------------


def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(report: dict) -> str:
    lines = [
        f"# Eval report: {report['run']}",
        "",
        f"- checkpoint: {report['checkpoint']}",
        f"- generated: {report['timestamp']}",
        "",
        "## Battery",
        "",
        "| scenario | fell_at | steps | metrics |",
        "|---|---|---|---|",
    ]
    for name, r in report["battery"].items():
        if name in ("run", "checkpoint"):
            continue
        extra = {k: v for k, v in r.items() if k not in ("fell_at", "steps")}
        metrics = ", ".join(f"{k}={v}" for k, v in extra.items())
        lines.append(f"| {name} | {_fmt(r.get('fell_at'))} | {r.get('steps', '-')} | {metrics or '-'} |")

    t = report["torque"]
    lines += [
        "",
        "## Torque (|actuator_force|, N*m)",
        "",
        "| p50 | p90 | p99 | max |",
        "|---|---|---|---|",
        f"| {_fmt(t['p50'])} | {_fmt(t['p90'])} | {_fmt(t['p99'])} | {_fmt(t['max'])} |",
    ]

    p = report["power"]
    lines += [
        "",
        "## Power (|actuator_force * joint_vel|, W)",
        "",
        "| p50 | p90 | p99 | mean total |",
        "|---|---|---|---|",
        f"| {_fmt(p['p50'])} | {_fmt(p['p90'])} | {_fmt(p['p99'])} | {_fmt(p['mean_total'])} |",
    ]

    f = report["foot_force_proxy"]
    lines += [
        "",
        "## Foot-force proxy",
        "",
        f"- peak base accel: {_fmt(f.get('peak_accel_mps2'))} m/s^2",
        f"- peak proxy force: {_fmt(f.get('peak_force_n'))} N",
        (
            "- method: peak baseline-subtracted |base IMU accelerometer "
            "z-axis| (linear-acceleration sensor), scaled by total robot "
            "mass (F=m*a). No direct contact-force sensor exists on this "
            "env -- this is a directional proxy, not a calibrated "
            "ground-reaction-force measurement."
        ),
    ]

    term = report["termination"]
    lines += [
        "",
        "## Termination",
        "",
        f"- fell in {term['fall_count']}/{term['scenarios_run']} scenarios",
        f"- reason counts: {term['fall_reason_counts']}",
        "",
        "| scenario | fell | fell_at | reason |",
        "|---|---|---|---|",
    ]
    for name, s in term["per_scenario"].items():
        lines.append(
            f"| {name} | {s['fell']} | {_fmt(s['fell_at'])} | {s['reason'] or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    report = build_report(run_dir)
    out_json = Path(args.out_json) if args.out_json else run_dir / "eval_report.json"
    out_md = Path(args.out_md) if args.out_md else run_dir / "eval_report.md"
    out_json.write_text(json.dumps(report, indent=2))
    out_md.write_text(render_markdown(report))
    print(json.dumps(report, indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
