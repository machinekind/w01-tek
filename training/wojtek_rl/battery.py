"""Fixed evaluation battery for locomotion policies: one number-table per
run so iterations are comparable. Run: ./run.sh battery --run runs/<name>

Scenarios (all no-fall expected):
  ramp_mid / ramp_low / ramp_tall — stand 2 s, ramp 0->1.0 m/s over 10 s,
      hold 3 s, at three stance heights
  walk_trot — sustained walk 0.28, step to trot 0.75
  stand_heights — stand while height steps 0.125->0.095->0.165

Metrics: falls, per-speed-band velocity tracking error, height error,
vibration index (>5 Hz joint-velocity power), foot slip, gait purity
(diagonal-pair contact correlation in the trot band; should be high in
trot, low in walk).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from wojtek_rl import paths


def vibration_index(qvel_hist, dt, cutoff_hz=5.0):
    v = qvel_hist - qvel_hist.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(v, axis=0)) ** 2
    freqs = np.fft.rfftfreq(v.shape[0], d=dt)
    total = power[freqs > 0.0].sum()
    return float(power[freqs > cutoff_hz].sum() / max(total, 1e-12))


def diag_corr(contacts):
    """Correlation of diagonal foot pairs (LEGS order RL, RR, FR, FL)."""
    c = contacts.astype(float)
    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 1e-9 else 0.0
    return 0.5 * (corr(c[:, 0], c[:, 2]) + corr(c[:, 1], c[:, 3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from wojtek_rl.build_model import FOOT_RADIUS
    from wojtek_rl.policy_io import load_policy
    from wojtek_rl.registry import make_env
    from wojtek_rl.train import build_ppo_params

    run = json.loads((Path(args.run) / "run.json").read_text())
    # Measurement env: no random pushes (they contaminate vibration/slip/
    # stand metrics; robustness is trained, not measured here).
    env_cfg = dict(run.get("env_config") or {})
    env_cfg["push"] = {**env_cfg.get("push", {}), "enable": False}
    env = make_env(run.get("task", "joystick"), env_cfg)
    ckpt_dir = Path(run["checkpoint_dir"])
    if not ckpt_dir.exists():
        ckpt_dir = (Path(args.run) / "checkpoints").resolve()
    ckpt = max(
        (p for p in ckpt_dir.iterdir() if p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    policy = load_policy(ckpt, env, build_ppo_params([], smoke=False))
    reset, step, inf = jax.jit(env.reset), jax.jit(env.step), jax.jit(policy)

    def rollout(cmd_at, n, seed=0):
        rng = jax.random.PRNGKey(seed)
        state = reset(rng)
        rec = {"cmd_vx": [], "vx": [], "cmd_h": [], "h": [], "qvel": [], "contact": [], "slip": []}
        fell_at = None
        for i in range(n):
            state.info["command"] = cmd_at(i)
            rng, k = jax.random.split(rng)
            act, _ = inf(state.obs, k)
            state = step(state, act)
            if float(state.done) and fell_at is None:
                fell_at = i
                break
            d = state.data
            c = np.asarray(d.geom_xpos)[env._foot_geom_ids][:, 2] < FOOT_RADIUS + 0.005
            fv = np.asarray(d.sensordata)[np.asarray(env._foot_linvel_adr)]
            rec["cmd_vx"].append(float(cmd_at(i)[0]))
            rec["vx"].append(float(d.qvel[0]))
            rec["cmd_h"].append(float(cmd_at(i)[3]))
            rec["h"].append(float(d.qpos[2]))
            rec["qvel"].append(np.asarray(d.qvel[env._vadr]))
            rec["contact"].append(c)
            rec["slip"].append(float((np.square(fv[:, :2]).sum(-1) * c).sum()))
        return {k: np.array(v) for k, v in rec.items()}, fell_at

    def ramp(h):
        def cmd(i):
            vx = 0.0 if i < 100 else min(1.0, (i - 100) / 500)
            return jp.array([vx, 0.0, 0.0, h])
        return cmd, 750

    scenarios = {
        "ramp_mid": ramp(0.13),
        "ramp_low": ramp(0.10),
        "ramp_tall": ramp(0.165),
        "walk_trot": (
            lambda i: jp.array([0.28 if i < 300 else 0.75, 0.0, 0.0, 0.13]),
            600,
        ),
        "stand_heights": (
            lambda i: jp.array(
                [0.0, 0.0, 0.0, [0.125, 0.095, 0.165][min(i // 150, 2)]]
            ),
            450,
        ),
    }

    results = {"run": run["run_name"], "checkpoint": ckpt.name}
    for name, (cmd_at, n) in scenarios.items():
        rec, fell_at = rollout(cmd_at, n)
        r = {"fell_at": fell_at, "steps": len(rec["vx"])}
        if len(rec["vx"]) > 50:
            moving = rec["cmd_vx"] > 0.05
            # velocity error by command band
            for lo, hi, band in [(0.05, 0.3, "vlow"), (0.3, 0.6, "vmid"), (0.6, 1.01, "vhigh")]:
                m = (rec["cmd_vx"] >= lo) & (rec["cmd_vx"] < hi)
                if m.sum() > 10:
                    r[f"vel_err_{band}"] = round(float((rec["cmd_vx"][m] - rec["vx"][m]).mean()), 3)
            r["height_err_mean"] = round(float(np.abs(rec["cmd_h"] - rec["h"]).mean()), 4)
            r["vibration"] = round(vibration_index(rec["qvel"], env.dt), 3)
            # absolute motion scale: the vibration ratio is meaningless when
            # nearly motionless (tiny numerator over tiny denominator)
            r["qvel_rms"] = round(float(np.sqrt((rec["qvel"] ** 2).mean())), 3)
            r["slip_mean"] = round(float(rec["slip"].mean()), 4)
            if name == "walk_trot":
                r["diag_corr_walk"] = round(diag_corr(rec["contact"][100:300]), 2)
                r["diag_corr_trot"] = round(diag_corr(rec["contact"][350:]), 2)
            if name == "stand_heights":
                # Transitions are intentional motion; measure stillness only
                # in hold windows (60+ steps after each height switch) and
                # report settling separately. (The old all-steps qvel_rms
                # sent two iterations chasing a phantom tremble.)
                hold = np.zeros(len(rec["h"]), dtype=bool)
                for s0 in (0, 150, 300):
                    hold[s0 + 60 : s0 + 150] = True
                hold = hold[: len(rec["h"])]
                r["qvel_rms_hold"] = round(
                    float(np.sqrt((rec["qvel"][hold] ** 2).mean())), 3
                )
                settle = []
                for s0 in (150, 300):
                    err = np.abs(rec["h"][s0:s0 + 150] - rec["cmd_h"][s0:s0 + 150])
                    ok = np.where(err < 0.005)[0]
                    settle.append(int(ok[0]) if len(ok) else 150)
                r["settle_steps"] = settle
            if moving.sum() > 100:
                r["vel_err_overall"] = round(float((rec["cmd_vx"][moving] - rec["vx"][moving]).mean()), 3)
        results[name] = r

    out = Path(args.out) if args.out else Path(args.run) / "battery.json"
    stamped = dict(results, timestamp=datetime.now().isoformat(timespec="seconds"))
    out.write_text(json.dumps(stamped, indent=2))
    print(json.dumps(stamped, indent=2))


if __name__ == "__main__":
    main()
