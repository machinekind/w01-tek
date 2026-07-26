"""Turn a rollout into a score: physical normalizers, weakest-link min,
completion gate. See the package docstring for the full rationale."""

import numpy as np

from wojtek_rl.battery import vibration_index
from wojtek_rl.courses.spec import HEIGHT_CMD

# -- physical normalizers ---------------------------------------------------
# Home-keyframe references for the 14 kg model, the same ones battery.py's
# sim2real symptom metrics quote: feet sit +-0.174 m either side of the
# centreline, base z = 0.129 m. Used only as the denominators that turn a
# measured error into a dimensionless sub-score.
STANCE_HALFWIDTH_M = 0.174
NOMINAL_HEIGHT_M = 0.129
VIBRATION_CUTOFF_HZ = 5.0  # battery.vibration_index's default, restated here

# A sub-score is a ratio, so a near-zero error would send it to infinity and
# out of JSON. Cap it: anything this good is indistinguishable from perfect.
SUBSCORE_CAP = 1000.0

MIN_SCORE_STEPS = 50  # 1 s of course data: below this the RMS/spectral
# metrics are meaningless, so the seed just scores 0


def _ratio(reference, error):
    """`reference / error`, error floored at 1e-9 and the result capped at
    SUBSCORE_CAP so a near-perfect axis stays finite. Higher is better."""
    return round(float(min(SUBSCORE_CAP, reference / max(float(error), 1e-9))), 3)


def seed_result(rec, info, dt) -> dict:
    """One (scenario, seed) entry: raw metrics, sub-scores, and the score.

    A course that fell or never reached its final waypoint scores 0 -- the
    gate. Its raw metrics are still reported (over however far it got), so
    a failure is diagnosable rather than just a zero.
    """
    n = int(info["steps"])
    out = {
        "completed": bool(info["completed"]),
        "fell_at": info["fell_at"],
        "steps": n,
        "progress_m": round(float(rec["s"][-1]), 3) if n else 0.0,
        "course_length_m": round(float(info.get("total_length", 0.0)), 3),
    }
    if n <= MIN_SCORE_STEPS:  # too short for a meaningful spectrum or RMS
        out["score"] = 0.0
        out["subscores"] = None
        return out

    xte = np.asarray(rec["xte"], dtype=float)
    cmd_v = np.asarray(rec["cmd_v"], dtype=float)
    v_fwd = np.asarray(rec["v_fwd"], dtype=float)
    h = np.asarray(rec["h"], dtype=float)
    dist = float(np.asarray(rec["v_planar"], dtype=float).sum() * dt)
    slip = float(np.asarray(rec["slip_speed"], dtype=float).sum() * dt)
    vib = vibration_index(rec["qvel"], dt, cutoff_hz=VIBRATION_CUTOFF_HZ)

    # Speed is scored only where the follower asked for motion: the
    # yaw-in-place branch commands vx = 0, and charging "speed error" while
    # the robot is deliberately pivoting would penalise corners twice.
    moving = cmd_v > 0.05
    v_err = float(np.sqrt(np.square(cmd_v[moving] - v_fwd[moving]).mean())) if (
        moving.sum() > 10
    ) else 0.0
    v_ref = float(cmd_v[moving].mean()) if moving.sum() > 10 else 0.0

    raw = {
        "xte_rms_m": round(float(np.sqrt(np.square(xte).mean())), 4),
        "xte_p95_m": round(float(np.percentile(np.abs(xte), 95)), 4),
        "speed_err_rms": round(v_err, 4),
        "speed_cmd_mean": round(v_ref, 4),
        "height_err_rms_m": round(float(np.sqrt(np.square(h - HEIGHT_CMD).mean())), 4),
        "base_distance_m": round(dist, 3),
        "slip_distance_m": round(slip, 3),
        "vibration": round(vib, 4),
        "duration_s": round(n * dt, 2),
    }
    subs = {
        "tracking": _ratio(STANCE_HALFWIDTH_M, raw["xte_rms_m"]),
        "speed": _ratio(v_ref, v_err) if v_ref > 0 else SUBSCORE_CAP,
        "height": _ratio(NOMINAL_HEIGHT_M, raw["height_err_rms_m"]),
        "grip": _ratio(dist, slip),
        "smoothness": _ratio(1.0, vib),
    }
    binding = min(subs, key=subs.get)
    out.update(
        raw=raw,
        subscores=subs,
        binding=binding,
        score=round(min(subs.values()), 3) if out["completed"] else 0.0,
    )
    return out


def spin_seed_result(rec, info, dt, wz_cmd: float) -> dict:
    """One (spin scenario, seed) entry -- the SpinCourse analogue of
    seed_result.

    Same shape and same philosophy, different sub-scores: there is no path
    and no forward travel, so tracking/grip/speed are replaced by

        rotation  |wz_cmd| / RMS yaw-rate error   (the command normalizes)
        drift     STANCE_HALFWIDTH_M / max planar drift from the start

    height and smoothness are shared with seed_result. The gate is the
    same: a fall or an incomplete rotation scores 0.
    """
    n = int(info["steps"])
    required = float(info.get("total_length", 0.0))
    out = {
        "completed": bool(info["completed"]),
        "fell_at": info["fell_at"],
        "steps": n,
        "progress_m": round(float(rec["yaw_progress"][-1]), 3) if n else 0.0,
        "course_length_m": round(required, 3),  # radians for spin rows
    }
    if n <= MIN_SCORE_STEPS:
        out["score"] = 0.0
        out["subscores"] = None
        return out

    wz = np.asarray(rec["wz"], dtype=float)
    h = np.asarray(rec["h"], dtype=float)
    drift_max = float(np.asarray(rec["drift"], dtype=float).max())
    wz_err = float(np.sqrt(np.square(wz_cmd - wz).mean()))
    vib = vibration_index(rec["qvel"], dt, cutoff_hz=VIBRATION_CUTOFF_HZ)

    raw = {
        "yaw_rad_achieved": round(float(rec["yaw_progress"][-1]), 3),
        "wz_err_rms": round(wz_err, 4),
        "wz_cmd": round(float(wz_cmd), 3),
        "drift_max_m": round(drift_max, 4),
        "height_err_rms_m": round(float(np.sqrt(np.square(h - HEIGHT_CMD).mean())), 4),
        "vibration": round(vib, 4),
        "duration_s": round(n * dt, 2),
    }
    subs = {
        "rotation": _ratio(abs(wz_cmd), wz_err),
        "drift": _ratio(STANCE_HALFWIDTH_M, drift_max),
        "height": _ratio(NOMINAL_HEIGHT_M, raw["height_err_rms_m"]),
        "smoothness": _ratio(1.0, vib),
    }
    binding = min(subs, key=subs.get)
    out.update(
        raw=raw,
        subscores=subs,
        binding=binding,
        score=round(min(subs.values()), 3) if out["completed"] else 0.0,
    )
    return out


def aggregate(seeds: list[dict]) -> dict:
    """Median / worst across a scenario's seeds, plus the failure counts."""
    scores = [s["score"] for s in seeds]
    scored = [s for s in seeds if s.get("subscores")]
    agg = {
        "score_median": round(float(np.median(scores)), 3),
        "score_worst": round(float(np.min(scores)), 3),
        "seeds": len(seeds),
        "falls": sum(1 for s in seeds if s["fell_at"] is not None),
        "completed": sum(1 for s in seeds if s["completed"]),
    }
    if scored:
        agg["subscore_median"] = {
            k: round(float(np.median([s["subscores"][k] for s in scored])), 3)
            for k in scored[0]["subscores"]
        }
        agg["raw_median"] = {
            k: round(float(np.median([s["raw"][k] for s in scored])), 4)
            for k in scored[0]["raw"]
        }
        agg["binding"] = min(agg["subscore_median"], key=agg["subscore_median"].get)
    agg["per_seed"] = seeds
    return agg
