"""Fixed evaluation battery for locomotion policies: one number-table per
run so iterations are comparable. Run: ./run.sh battery --run runs/<name>

Scenarios (all no-fall expected), matching the four video names in
docs/2026-07-12-reward-redesign.md:
  stand_to_trot_ramp — stand 100 steps, ramp vx 0->1.0 over 500 steps, hold
      150 steps; height pinned at 0.125 throughout. 750 steps.
  turn — stand 100 steps; vx=0.4 with wz swept -0.8->+0.8 over 400 steps;
      pure-spin hold vx=0, wz=0.8 for 250 steps. 750 steps.
  strafe — stand 100 steps; vy=+0.4 for 250 steps; vy=-0.4 for 250 steps.
      600 steps.
  walk_to_stop — vx=0.5 for 250 steps, then the command drops to all-zero
      and holds for 300 steps. 550 steps.

Metrics: falls; per-speed-band velocity tracking error; height error;
vibration index (>5 Hz joint-velocity power); foot slip; splay (p95
|abduction|) and torque-saturation fraction per joint group, every
scenario; gait purity (diagonal/lateral foot-contact correlation, per-foot
duty factor) over each scenario's moving window; angular tracking error
(turn only); lateral tracking error (strafe only); base-height
springiness -- peak-to-peak amplitude and 2-4 Hz spectral power fraction
(stand_to_trot_ramp hold window only); foot-planting step count, post-
switch clearance, and quiet-stance qvel (walk_to_stop only).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from wojtek_rl import paths

# Actuator order is per leg (abduction, hip, knee) tiled over paths.LEGS
# (rear_left, rear_right, front_right, front_left) -- see base.py._qadr and
# KNEE_ACTUATORS = (2, 5, 8, 11).
ABDUCTION_IDX = (0, 3, 6, 9)
HIP_IDX = (1, 4, 7, 10)
KNEE_IDX = (2, 5, 8, 11)


def vibration_index(qvel_hist, dt, cutoff_hz=5.0):
    v = qvel_hist - qvel_hist.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(v, axis=0)) ** 2
    freqs = np.fft.rfftfreq(v.shape[0], d=dt)
    total = power[freqs > 0.0].sum()
    return float(power[freqs > cutoff_hz].sum() / max(total, 1e-12))


def band_power_fraction(hist, dt, lo, hi):
    """Fraction of `hist`'s baseline-subtracted spectral power (rfft, same
    construction as vibration_index) falling in [lo, hi) Hz. Guards the
    same tiny-denominator case: a near-motionless signal has near-zero
    total power, so the ratio would blow up without the floor."""
    v = np.asarray(hist, dtype=float)
    v = v - v.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(v, axis=0)) ** 2
    freqs = np.fft.rfftfreq(v.shape[0], d=dt)
    total = power[freqs > 0.0].sum()
    band = power[(freqs >= lo) & (freqs < hi)].sum()
    return float(band / max(total, 1e-12))


def _contact_corr(c, i, j):
    a, b = c[:, i] - c[:, i].mean(), c[:, j] - c[:, j].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def diag_corr(contacts):
    """Correlation of diagonal foot pairs (LEGS order RL, RR, FR, FL):
    (0, 2) and (1, 3). High in trot, near zero in pace/pronk/skating."""
    c = contacts.astype(float)
    return 0.5 * (_contact_corr(c, 0, 2) + _contact_corr(c, 1, 3))


def lateral_corr(contacts):
    """Correlation of same-side foot pairs (LEGS order RL, RR, FR, FL):
    left (0, 3), right (1, 2). Mirrors diag_corr's pairing; high in pace,
    high alongside diag_corr in pronk."""
    c = contacts.astype(float)
    return 0.5 * (_contact_corr(c, 0, 3) + _contact_corr(c, 1, 2))


def duty_factor(contacts):
    """Mean contact fraction per foot, LEGS order (4,). Skating reads as
    duty ~= 1.0 on every foot with low diag/lateral correlation."""
    c = np.asarray(contacts, dtype=float)
    return [round(float(x), 3) for x in c.mean(axis=0)]


def abduction_p95(qpos_hist):
    """p95 of |abduction angle| (rad), pooled over steps and the four
    abduction joints (ABDUCTION_IDX). The anti-splay signal: a policy that
    holds abduction near zero has a low p95 even if it occasionally spikes."""
    q = np.asarray(qpos_hist, dtype=float)
    return float(np.percentile(np.abs(q[:, ABDUCTION_IDX]), 95))


def saturation_fractions(actuator_force_hist, torque_cap):
    """Fraction of (step, joint) samples over 85% of `torque_cap`, per leg-
    joint group (ABDUCTION_IDX/HIP_IDX/KNEE_IDX). `torque_cap <= 0` means
    the cap is unknown (should not happen given run_battery/build_report's
    fallback to the model's forcerange, but a hand-built test case might
    pass one) -- return None per group rather than a spurious 0.0."""
    f = np.asarray(actuator_force_hist, dtype=float)
    if torque_cap <= 0:
        return {"abduction": None, "hip": None, "knee": None}
    over = np.abs(f) > 0.85 * torque_cap
    return {
        "abduction": float(over[:, ABDUCTION_IDX].mean()),
        "hip": float(over[:, HIP_IDX].mean()),
        "knee": float(over[:, KNEE_IDX].mean()),
    }


def plant_step(contacts, switch_idx, hold=25):
    """Steps from `switch_idx` until all four feet are in contact and stay
    that way for `hold` consecutive steps; None if that window never
    occurs before `contacts` ends. A brief all-four touch that lifts a
    foot before `hold` steps elapse does not count -- the whole window
    must stay planted, not just its first step."""
    all_down = np.asarray(contacts, dtype=bool).all(axis=-1)
    n = len(all_down)
    for i in range(switch_idx, n - hold + 1):
        if all_down[i : i + hold].all():
            return i - switch_idx
    return None


def battery_scenarios():
    """name -> (cmd_at(i), n_steps). Split out so report.py (and eval.py's
    --scenario) can reuse the exact same battery scenarios. Height is
    pinned at 0.125 throughout every scenario -- the redesign drops the
    height command (docs/2026-07-12-reward-redesign.md), but the env still
    takes a 4-vector [vx, vy, wz, height], so cmd_at keeps sending the
    anchor value."""
    H = 0.125

    def stand_to_trot_ramp(i):
        vx = 0.0 if i < 100 else min(1.0, (i - 100) / 500)
        return jp.array([vx, 0.0, 0.0, H])

    def turn(i):
        if i < 100:
            vx, wz = 0.0, 0.0
        elif i < 500:
            vx = 0.4
            wz = -0.8 + 1.6 * (i - 100) / 399  # exact endpoints at i=100/499
        else:
            vx, wz = 0.0, 0.8  # pure-spin hold
        return jp.array([vx, 0.0, wz, H])

    def strafe(i):
        if i < 100:
            vy = 0.0
        elif i < 350:
            vy = 0.4
        else:
            vy = -0.4
        return jp.array([0.0, vy, 0.0, H])

    def walk_to_stop(i):
        vx = 0.5 if i < 250 else 0.0
        return jp.array([vx, 0.0, 0.0, H])

    return {
        "stand_to_trot_ramp": (stand_to_trot_ramp, 750),
        "turn": (turn, 750),
        "strafe": (strafe, 600),
        "walk_to_stop": (walk_to_stop, 550),
    }


def load_checkpoint_policy(run_dir: Path):
    """Load a run's measurement env + latest-checkpoint policy.

    Shared by run_battery and report.py, so "which checkpoint is latest"
    and "no random pushes while measuring" can't drift between the two.
    Returns (run, env, ckpt, inf, foot_radius) where inf = jax.jit(policy).
    """
    from wojtek_rl.build_model import FOOT_RADIUS
    from wojtek_rl.policy_io import load_policy
    from wojtek_rl.registry import make_env
    from wojtek_rl.train import build_ppo_params

    run = json.loads((run_dir / "run.json").read_text())
    # Measurement env: no random pushes (they contaminate vibration/slip/
    # stand metrics; robustness is trained, not measured here).
    env_cfg = dict(run.get("env_config") or {})
    env_cfg["push"] = {**env_cfg.get("push", {}), "enable": False}
    env = make_env(run.get("task", "joystick"), env_cfg)
    ckpt_dir = Path(run["checkpoint_dir"])
    if not ckpt_dir.exists():
        ckpt_dir = (run_dir / "checkpoints").resolve()
    ckpt = max(
        (p for p in ckpt_dir.iterdir() if p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    policy = load_policy(ckpt, env, build_ppo_params([], smoke=False))
    return run, env, ckpt, jax.jit(policy), FOOT_RADIUS


def rollout(env, reset, step, inf, cmd_at, n, foot_radius, seed=0):
    """Roll out `n` steps of `cmd_at` under `inf` in `env`.

    `reset`/`step` are `jax.jit(env.reset)`/`jax.jit(env.step)` -- passed in
    (rather than jitted here) so callers compile them once and reuse the
    same jitted callables across every scenario, as the original battery
    script did.

    Records the signals the battery scenarios need (cmd/state tracking,
    joint position/velocity, foot contact/clearance/slip, body-frame
    planar velocity + yaw rate) plus two extra per-step signals report.py
    reduces into torque/power/foot-force metrics (`actuator_force`,
    `base_accel`) -- harmless here since the battery's own scenario
    summary below only reads the fields it always has.

    Returns (rec, fell_at, term): `rec` maps signal name -> np.ndarray over
    the steps taken before any fall; `fell_at` is the step index of
    termination (None if the scenario completed); `term` is the state at
    termination ({"height", "gravity_z"}, used to classify the fall
    reason), or None if it never fell.
    """
    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    rec = {
        "cmd_vx": [], "vx": [], "vx_local": [], "cmd_vy": [], "vy_local": [],
        "cmd_wz": [], "wz": [], "cmd_h": [], "h": [],
        "qvel": [], "qpos": [], "contact": [], "foot_clearance": [],
        "slip": [], "actuator_force": [], "base_accel": [],
    }
    fell_at = None
    term = None
    for i in range(n):
        cmd = cmd_at(i)
        state.info["command"] = cmd
        rng, k = jax.random.split(rng)
        act, _ = inf(state.obs, k)
        state = step(state, act)
        d = state.data
        if float(state.done) and fell_at is None:
            fell_at = i
            term = {
                "height": float(d.qpos[2]),
                "gravity_z": float(env._gravity_body(d)[2]),
            }
            break
        gx = np.asarray(d.geom_xpos)[env._foot_geom_ids]
        c = gx[:, 2] < foot_radius + 0.005
        fv = np.asarray(d.sensordata)[np.asarray(env._foot_linvel_adr)]
        adr = env._sensor_adr["linear-acceleration"]
        local_linvel = np.asarray(env._local_linvel(d))
        rec["cmd_vx"].append(float(cmd[0]))
        rec["vx"].append(float(d.qvel[0]))
        rec["vx_local"].append(float(local_linvel[0]))
        rec["cmd_vy"].append(float(cmd[1]))
        rec["vy_local"].append(float(local_linvel[1]))
        rec["cmd_wz"].append(float(cmd[2]))
        rec["wz"].append(float(np.asarray(env._gyro(d))[2]))
        rec["cmd_h"].append(float(cmd[3]))
        rec["h"].append(float(d.qpos[2]))
        rec["qvel"].append(np.asarray(d.qvel[env._vadr]))
        rec["qpos"].append(np.asarray(d.qpos[env._qadr]))
        rec["contact"].append(c)
        rec["foot_clearance"].append(gx[:, 2] - foot_radius)
        rec["slip"].append(float((np.square(fv[:, :2]).sum(-1) * c).sum()))
        rec["actuator_force"].append(np.asarray(d.actuator_force))
        rec["base_accel"].append(np.asarray(d.sensordata[adr : adr + 3]))
    return {k: np.array(v) for k, v in rec.items()}, fell_at, term


def scenario_result(name, rec, fell_at, dt, torque_cap):
    """One scenario's battery.json entry, computed from its rollout().

    `torque_cap` is the env's effective actuator-force cap (N*m), used by
    the saturation metric -- see run_battery/build_report for how it's
    derived from the env.
    """
    r = {"fell_at": fell_at, "steps": len(rec["vx"])}
    if len(rec["vx"]) <= 50:
        return r

    moving = rec["cmd_vx"] > 0.05
    # velocity error by command band
    for lo, hi, band in [(0.05, 0.3, "vlow"), (0.3, 0.6, "vmid"), (0.6, 1.01, "vhigh")]:
        m = (rec["cmd_vx"] >= lo) & (rec["cmd_vx"] < hi)
        if m.sum() > 10:
            r[f"vel_err_{band}"] = round(float((rec["cmd_vx"][m] - rec["vx"][m]).mean()), 3)
    r["height_err_mean"] = round(float(np.abs(rec["cmd_h"] - rec["h"]).mean()), 4)
    r["vibration"] = round(vibration_index(rec["qvel"], dt), 3)
    # absolute motion scale: the vibration ratio is meaningless when
    # nearly motionless (tiny numerator over tiny denominator)
    r["qvel_rms"] = round(float(np.sqrt((rec["qvel"] ** 2).mean())), 3)
    r["slip_mean"] = round(float(rec["slip"].mean()), 4)
    if moving.sum() > 100:
        r["vel_err_overall"] = round(float((rec["cmd_vx"][moving] - rec["vx"][moving]).mean()), 3)

    # Anti-splay metrics: every scenario.
    r["splay_p95"] = round(abduction_p95(rec["qpos"]), 4)
    sat = saturation_fractions(rec["actuator_force"], torque_cap)
    r["saturation"] = {k: (round(v, 4) if v is not None else None) for k, v in sat.items()}

    # Gait purity over each scenario's own moving window (reuses `moving`,
    # so it lands on the ramp/hold for stand_to_trot_ramp, the vx=0.4
    # sweep for turn, and the pre-stop walk for walk_to_stop; strafe's
    # cmd_vx is always 0, so it has no moving window and gets none of
    # these -- side-stepping isn't the trot/pace/pronk/skating question).
    if moving.sum() > 100:
        c = rec["contact"][moving]
        r["diag_corr"] = round(diag_corr(c), 2)
        r["lateral_corr"] = round(lateral_corr(c), 2)
        r["duty_factor"] = duty_factor(c)

    if name == "turn":
        sweep, spin = slice(100, 500), slice(500, 750)
        # guarded: a fall inside the sweep truncates rec before these
        # slices exist in full.
        if len(rec["wz"]) >= 500:
            r["ang_vel_err_sweep"] = round(
                float(np.abs(rec["cmd_wz"][sweep] - rec["wz"][sweep]).mean()), 3
            )
        if len(rec["wz"]) >= 750:
            r["ang_vel_err_spin"] = round(
                float(np.abs(rec["cmd_wz"][spin] - rec["wz"][spin]).mean()), 3
            )

    if name == "strafe" and len(rec["vy_local"]) > 100:
        # both strafe windows pooled; the stand phase (steps 0:100) excluded
        w = slice(100, len(rec["vy_local"]))
        r["vy_err"] = round(float(np.abs(rec["cmd_vy"][w] - rec["vy_local"][w]).mean()), 3)

    if name == "stand_to_trot_ramp" and len(rec["h"]) >= 750:
        hold = rec["h"][600:750]
        r["height_pp"] = round(float(hold.max() - hold.min()), 4)
        r["height_band_power"] = round(band_power_fraction(hold, dt, 2.0, 4.0), 3)

    if name == "walk_to_stop":
        switch = 250
        r["steps_to_plant"] = plant_step(rec["contact"], switch, hold=25)
        if len(rec["foot_clearance"]) > switch:
            r["max_clearance_hold"] = round(float(rec["foot_clearance"][switch:].max()), 4)
        if len(rec["qvel"]) >= 150:
            tail = rec["qvel"][-150:]
            r["qvel_rms_hold"] = round(float(np.sqrt((tail ** 2).mean())), 3)

    return r


def torque_cap_of(env) -> float:
    """The env's effective actuator-force cap (N*m): the configured
    `max_torque` clamp if one is set, else the model's own forcerange.
    Shared by run_battery and report.py's build_report so the saturation
    metric can't drift between the two. Old runs predate `max_torque`
    entirely, so `.get` with a falsy default is required, not optional."""
    mt = env._config.get("max_torque", 0.0)
    return float(mt) if mt else float(env.mj_model.actuator_forcerange[:, 1].max())


def run_battery(run_dir: Path) -> dict:
    """Run the fixed scenario battery against `run_dir`'s checkpoint.

    Returns the same dict `main()` used to write (minus the `timestamp`
    stamp main() adds at write time). report.py imports this for the
    battery table half of eval_report.json.
    """
    run, env, ckpt, inf, foot_radius = load_checkpoint_policy(run_dir)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    torque_cap = torque_cap_of(env)
    results = {"run": run["run_name"], "checkpoint": ckpt.name}
    for name, (cmd_at, n) in battery_scenarios().items():
        rec, fell_at, _term = rollout(env, reset, step, inf, cmd_at, n, foot_radius)
        results[name] = scenario_result(name, rec, fell_at, env.dt, torque_cap)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = run_battery(Path(args.run))
    out = Path(args.out) if args.out else Path(args.run) / "battery.json"
    stamped = dict(results, timestamp=datetime.now().isoformat(timespec="seconds"))
    out.write_text(json.dumps(stamped, indent=2))
    print(json.dumps(stamped, indent=2))


if __name__ == "__main__":
    main()
