"""Aggregate the sim2real robustness grid's per-cell battery.json files into
one markdown comparison table.

Run: python3 -m wojtek_rl.grid_report --runs <run1> [<run2> ...] [--out FILE]

Reads runs/<run>/grid/battery_a<alpha>_lag<lag_ms>ms_env<tag>.json for every
run listed -- the files training/hpc/stiff_grid.slurm writes via battery.py's
--alpha/--lag-tau/--torque-envelope/--out (see wojtek_rl/battery.py's
apply_kt_miscalibration/make_lagged_rollout_fns/apply_torque_envelope).
`<tag>` is "none" (flat cap, no --torque-envelope passed) or
"<OMEGA_B>-<OMEGA_0>". Filenames without the `_env<tag>` segment (grid runs
predating this axis) are read too, treated as env "none" -- see CELL_RE.
Applies the stiffness ladder's gates 1-4 (see training/hpc/stiff_ladder.
slurm's run_gates) to each cell independently -- gate 5 (diminishing
returns vs a previous rung) does not apply to a single eval cell, there is
no "previous rung" here -- and reports mean track_err_rms over the 4
battery scenarios plus PASS/FAIL per cell. A missing cell (its battery run
crashed or was never submitted -- see stiff_grid.slurm's per-cell
WARN-and-continue policy) prints as MISSING, not a report crash.
"""

import argparse
import json
import re
from pathlib import Path

from wojtek_rl import paths

SCENARIOS = ["stand_to_trot_ramp", "turn", "strafe", "walk_to_stop"]
VEL_ERR_SCENARIOS = ["stand_to_trot_ramp", "turn", "walk_to_stop"]

# Keeper reference numbers, provenance wojtek_stiff_b_20260717_235321 job
# NNNNNNN -- see training/hpc/stiff_ladder.slurm's KEEPER_VIBRATION_JSON and
# gates 1-4 (this module applies the same four gates per grid cell).
KEEPER_VIBRATION = {
    "stand_to_trot_ramp": 0.247,
    "turn": 0.144,
    "strafe": 0.359,
    "walk_to_stop": 0.123,
}
VIBRATION_MULT = 1.3
VEL_ERR_LIMIT = 0.20
SATURATION_LIMIT = 0.05

# The `_env<tag>` segment is optional so grid runs from before this axis
# existed (bare battery_a<alpha>_lag<ms>ms.json, no envelope tag at all)
# still parse -- they get env "none" by construction (see find_cells): a
# grid run made before --torque-envelope existed only ever probed the
# flat cap. `<tag>` itself has no underscore (see stiff_grid.slurm's
# env_tag construction: "none" or "OMEGA_B-OMEGA_0"), so `[^_]+` cannot
# run past the `.json` this pattern anchors on.
CELL_RE = re.compile(r"battery_a([0-9.]+)_lag(\d+)ms(?:_env([^_]+))?\.json$")


def find_cells(run_dir: Path) -> dict:
    """{(alpha, lag_ms, env_tag): Path} for every grid-cell battery.json
    present under run_dir/grid. Empty dict if the run never got a grid
    pass. `env_tag` is "none" for both an explicit `_envnone` cell and a
    pre-envelope-axis cell with no `_env` segment at all -- both mean
    "flat cap, no perturbation on this axis"."""
    out = {}
    grid_dir = run_dir / "grid"
    if not grid_dir.exists():
        return out
    for p in sorted(grid_dir.glob("battery_a*_lag*ms*.json")):
        m = CELL_RE.search(p.name)
        if not m:
            continue
        alpha, lag_ms, env_tag = m.group(1), m.group(2), m.group(3)
        out[(float(alpha), int(lag_ms), env_tag or "none")] = p
    return out


def gate_cell(battery: dict) -> tuple:
    """(verdict, mean_track_err_rms, reasons) for one cell's battery dict,
    applying ladder gates 1-4: no falls; vel_err_overall/strafe vy_err <
    0.2; vibration <= 1.3x the keeper reference per scenario; torque
    saturation < 0.05 in every joint group/scenario. `mean_track_err_rms`
    is reported even on a FAIL cell when the data exists (useful context
    for why a cell failed some other gate)."""
    reasons = []

    for sc in SCENARIOS:
        if battery.get(sc, {}).get("fell_at") is not None:
            reasons.append(f"fell:{sc}")

    for sc in VEL_ERR_SCENARIOS:
        v = battery.get(sc, {}).get("vel_err_overall")
        if v is None or abs(v) >= VEL_ERR_LIMIT:
            reasons.append(f"vel_err_overall:{sc}={v}")
    vy = battery.get("strafe", {}).get("vy_err")
    if vy is None or abs(vy) >= VEL_ERR_LIMIT:
        reasons.append(f"vy_err:strafe={vy}")

    for sc in SCENARIOS:
        v = battery.get(sc, {}).get("vibration")
        ref = KEEPER_VIBRATION[sc]
        limit = VIBRATION_MULT * ref
        if v is None or v > limit:
            reasons.append(f"vibration:{sc}={v}>{limit:.4f}")

    max_sat = 0.0
    for sc in SCENARIOS:
        sat = battery.get(sc, {}).get("saturation") or {}
        for frac in sat.values():
            if frac is not None:
                max_sat = max(max_sat, frac)
    if max_sat >= SATURATION_LIMIT:
        reasons.append(f"saturation={max_sat}>={SATURATION_LIMIT}")

    errs = [battery.get(sc, {}).get("track_err_rms") for sc in SCENARIOS]
    mean_err = sum(errs) / len(errs) if all(e is not None for e in errs) else None

    verdict = "PASS" if not reasons else "FAIL"
    return verdict, mean_err, reasons


def kp_of(run_dir: Path):
    """The run's effective kp (run.json's post-customize stamp, train.py's
    kp_eff -- see wojtek_rl/train.py), used to rank runs by stiffness for
    the "stiffest surviving run" summary. None if run.json is missing or
    predates the kp stamp."""
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return None
    return json.loads(run_json.read_text()).get("kp")


def build_grid(run_names, runs_root: Path):
    """({run_name: {(alpha, lag_ms, env_tag): (verdict, mean_err, reasons)}},
    {run_name: kp}) for every run in `run_names`."""
    grid, kp_by_run = {}, {}
    for run_name in run_names:
        run_dir = runs_root / run_name
        cells = find_cells(run_dir)
        row = {}
        for key, path in cells.items():
            battery = json.loads(path.read_text())
            row[key] = gate_cell(battery)
        grid[run_name] = row
        kp_by_run[run_name] = kp_of(run_dir)
    return grid, kp_by_run


def _env_sort_key(tag: str):
    """Sort "none" first, then numerically by (omega_b, omega_0) rather
    than lexically ("15-28" before "5-10" would be wrong as strings)."""
    if tag == "none":
        return (0, 0.0, 0.0)
    try:
        omega_b, omega_0 = tag.split("-")
        return (1, float(omega_b), float(omega_0))
    except ValueError:
        return (2, 0.0, 0.0)  # malformed tag: sort last, don't crash the report


def render_markdown(grid: dict, kp_by_run: dict) -> str:
    all_keys = sorted({k for row in grid.values() for k in row})
    lags = sorted({lag for _, lag, _ in all_keys})
    alphas = sorted({a for a, _, _ in all_keys})
    envs = sorted({e for _, _, e in all_keys}, key=_env_sort_key)

    lines = [
        "# Robustness grid report",
        "",
        "Eval-only sim2real plant perturbations (alpha: Kt miscalibration; "
        "lag: actuator-bandwidth first-order torque lag, ms; envelope: "
        "speed-dependent driving-torque cap, \"none\" or \"OMEGA_B-OMEGA_0\" "
        "rad/s) -- see wojtek_rl/battery.py's apply_kt_miscalibration/"
        "make_lagged_rollout_fns/apply_torque_envelope and "
        "training/docs/configuration.md's \"Robustness grid (eval-only)\".",
        "",
        "Row = one run x alpha x envelope; columns = lags. Cell = mean "
        "track_err_rms over the 4 battery scenarios, then PASS/FAIL "
        "against the stiffness ladder's gates 1-4 (falls; "
        "vel_err_overall/strafe vy_err < 0.2; vibration <= 1.3x keeper ref "
        "per scenario -- stand_to_trot_ramp 0.247 / turn 0.144 / strafe "
        "0.359 / walk_to_stop 0.123; saturation max < 0.05). Provenance: "
        "keeper wojtek_stiff_b_20260717_235321, job NNNNNNN. MISSING = the "
        "cell's battery run crashed or was never submitted.",
        "",
    ]

    if not lags:
        lines.append("No grid cells found under any listed run's `grid/` directory.")
        return "\n".join(lines) + "\n"

    header = (
        "| run | kp | alpha | envelope | "
        + " | ".join(f"{lag}ms" for lag in lags) + " |"
    )
    sep = "|" + "---|" * (4 + len(lags))
    lines += [header, sep]
    for run_name, row in grid.items():
        kp = kp_by_run.get(run_name)
        kp_s = f"{kp:g}" if kp is not None else "-"
        for alpha in alphas:
            for env in envs:
                cells = []
                for lag in lags:
                    cell = row.get((alpha, lag, env))
                    if cell is None:
                        cells.append("MISSING")
                    else:
                        verdict, mean_err, _reasons = cell
                        err_s = f"{mean_err:.4f}" if mean_err is not None else "-"
                        cells.append(f"{err_s} {verdict}")
                lines.append(
                    f"| {run_name} | {kp_s} | {alpha:g} | {env} | "
                    + " | ".join(cells) + " |"
                )

    lines += [
        "",
        "## Stiffest run that stays PASS across all lags and envelopes, "
        "per alpha-world",
        "",
    ]
    for alpha in alphas:
        best_run, best_kp = None, None
        for run_name, row in grid.items():
            kp = kp_by_run.get(run_name)
            all_pass = all(
                row.get((alpha, lag, env), ("FAIL", None, None))[0] == "PASS"
                for lag in lags
                for env in envs
            )
            if all_pass and kp is not None and (best_kp is None or kp > best_kp):
                best_run, best_kp = run_name, kp
        if best_run:
            lines.append(f"- alpha={alpha:g}: **{best_run}** (kp={best_kp:g})")
        else:
            lines.append(
                f"- alpha={alpha:g}: none of the listed runs pass every "
                "lag/envelope"
            )

    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs", nargs="+", required=True,
        help="run names under training/runs/ (e.g. wojtek_stiff_kp60_...)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs_root = paths.PROJECT_DIR / "runs"
    grid, kp_by_run = build_grid(args.runs, runs_root)
    md = render_markdown(grid, kp_by_run)

    out = Path(args.out) if args.out else runs_root / "grid_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
