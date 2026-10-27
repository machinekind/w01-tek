"""CLI: identify actuator/joint parameters from an air-rig rosbag.

    ./training/run.sh sysid --bag <ros2 bag dir> [options]

The bag must be recorded with the robot held off the ground (belly on a
raised support, legs hanging freely); the fit replays the command stream
through batched fixed-base MJX rollouts and runs CMA-ES over the selected
physics parameters so the simulated joint positions match the recorded
ones. On-ground bags are not supported for fitting — the unobservable
contact/base state lets the optimizer launder init error into parameters
— use them only to validate an identified model. Workflow, bag recording
and result application: training/docs/sysid.md.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from loguru import logger

from wojtek_rl import paths
from wojtek_rl.sysid.bag import read_bag
from wojtek_rl.sysid.dataset import build_dataset
from wojtek_rl.sysid.mount import air_model, median_quat
from wojtek_rl.sysid.rollout import make_evaluator
from wojtek_rl.sysid.space import (
    ALL_PARAMS,
    DEFAULT_PARAMS,
    GROUPINGS,
    ParamSpace,
)

_EXTRA_HINT = "sysid needs the 'sysid' extra: cd training && uv sync --extra sysid"


def _parse_args():
    p = argparse.ArgumentParser(
        prog="sysid", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bag", required=True, help="ros2 bag directory (or file)")
    p.add_argument("--out", default="", help="output dir (default runs/sysid_<ts>)")
    p.add_argument("--xml", default=str(paths.SCENE_XML))
    p.add_argument(
        "--base-quat", default="",
        help="rig orientation as 'w,x,y,z' (world<-base); a level "
        "belly-on-support rig is the upright default and needs neither "
        "this nor an IMU — pass it only for a tilted rig (overrides the "
        "bag's IMU)",
    )
    p.add_argument(
        "--params", default=",".join(DEFAULT_PARAMS),
        help=f"comma list from {','.join(ALL_PARAMS)}; torque_scale is "
        "opt-in (it needs torque saturation in the bag)",
    )
    p.add_argument("--grouping", choices=GROUPINGS, default="per_type")
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--popsize", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.25, help="CMA-ES init step")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window-sec", type=float, default=2.0)
    p.add_argument(
        "--warmup-sec", type=float, default=0.3,
        help="unscored settle time per window (four-bar closure)",
    )
    p.add_argument("--stride-sec", type=float, default=1.0)
    p.add_argument("--max-windows", type=int, default=8)
    p.add_argument(
        "--ctrl-dt", type=float, default=0.0,
        help="command grid step, 0 = one command per physics step",
    )
    p.add_argument("--backend", default="auto", choices=("auto", "jax", "warp"))
    return p.parse_args()


def main():
    args = _parse_args()
    try:
        from cmaes import CMA
    except ImportError:
        raise SystemExit(_EXTRA_HINT)

    out = Path(
        args.out or f"runs/sysid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out.mkdir(parents=True, exist_ok=True)

    params = tuple(args.params.split(","))

    base_model = mujoco.MjModel.from_xml_path(args.xml)
    act_names = [base_model.actuator(i).name for i in range(base_model.nu)]
    sig = read_bag(args.bag, act_names)
    if args.base_quat:
        quat = np.array([float(v) for v in args.base_quat.split(",")])
    elif sig.quat is not None:
        quat = median_quat(sig.quat)
    else:
        quat = None
        logger.warning(
            "no --base-quat and no IMU in the bag; welding the base at the "
            "model's declared (upright) orientation — pass --base-quat if "
            "the rig held the robot differently"
        )
    mj_model = air_model(args.xml, quat=quat)
    logger.info(f"fixed-base model welded at quat {quat}")
    ds = build_dataset(
        mj_model, sig,
        ctrl_dt=args.ctrl_dt or None,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        warmup_sec=args.warmup_sec,
        max_windows=args.max_windows,
    )
    logger.info(
        f"dataset: {ds.cmd.shape[0]} windows x {ds.cmd.shape[1]} steps "
        f"@ {ds.ctrl_dt * 1e3:.1f} ms grid ({ds.n_substeps} substeps), "
        f"warmup {ds.warmup_steps} steps"
    )

    space = ParamSpace(mj_model, params=params, grouping=args.grouping)
    logger.info(f"search space: {space.dim} dims ({args.grouping} grouping)")
    ev = make_evaluator(
        mj_model, space, ds, backend=args.backend, popsize=args.popsize
    )
    logger.info(f"backend: {ev.backend}")

    u0 = space.default_genome()
    baseline = float(np.asarray(ev.losses(u0[None]))[0])
    logger.info(f"baseline rmse {baseline:.4f} rad with {space.describe(u0)}")

    opt = CMA(
        mean=u0,
        sigma=args.sigma,
        bounds=np.tile(np.array([0.0, 1.0]), (space.dim, 1)),
        population_size=args.popsize,
        seed=args.seed,
    )
    best_f, best_u = baseline, u0
    history = out / "history.csv"
    with open(history, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["generation", "best_rmse", "gen_best_rmse", "gen_mean_rmse"])
        for gen in range(args.generations):
            xs = [opt.ask() for _ in range(opt.population_size)]
            fs = np.asarray(ev.losses(np.array(xs)))
            opt.tell(list(zip(xs, fs.tolist())))
            i = int(np.argmin(fs))
            if fs[i] < best_f:
                best_f, best_u = float(fs[i]), np.array(xs[i])
            writer.writerow([gen, best_f, float(fs[i]), float(fs.mean())])
            logger.info(
                f"gen {gen:3d}  best {best_f:.4f}  "
                f"gen-best {fs[i]:.4f}  gen-mean {fs.mean():.4f}"
            )
            if opt.should_stop():
                logger.info("CMA-ES converged, stopping early")
                break

    result = {
        "rmse": best_f,
        "baseline_rmse": baseline,
        "params": space.describe(best_u),
        "baseline_params": space.describe(u0),
        "genome": best_u.tolist(),
        "config": vars(args),
    }
    (out / "best.json").write_text(json.dumps(result, indent=2) + "\n")
    _plot(out, ds, ev, space, best_u, u0, act_names)
    logger.info(f"best rmse {best_f:.4f} rad (baseline {baseline:.4f})")
    logger.info(f"identified params: {json.dumps(result['params'], indent=2)}")
    logger.info(f"results in {out}/ (best.json, history.csv, fit.png)")
    if "kp" in result["params"]:
        kp = np.mean(list(result["params"]["kp"].values()))
        kd = np.mean(list(result["params"].get("kd", {"": 1.0}).values()))
        logger.info(
            f"to bake into the model: ./training/run.sh build "
            f"--kp {kp:.1f} --kd {kd:.2f}  (or train with "
            f"++task.env.pd_kp={kp:.1f} ++task.env.pd_kd={kd:.2f})"
        )


def _plot(out, ds, ev, space, best_u, u0, act_names):
    """Measured vs simulated joint traces on the first window."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_best = np.asarray(ev.rollout(best_u))[0]
    q_base = np.asarray(ev.rollout(u0))[0]
    t = np.arange(ds.cmd.shape[1]) * ds.ctrl_dt
    fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(t, ds.meas[0][:, j], "k", lw=1.2, label="measured")
        ax.plot(t, q_base[:, j], "C1", lw=0.9, label="sim default")
        ax.plot(t, q_best[:, j], "C0", lw=0.9, label="sim identified")
        ax.axvspan(0, ds.warmup_steps * ds.ctrl_dt, color="gray", alpha=0.15)
        ax.set_title(act_names[j], fontsize=8)
    axes.flat[0].legend(fontsize=7)
    fig.supxlabel("time [s]")
    fig.supylabel("joint position [rad]")
    fig.tight_layout()
    fig.savefig(out / "fit.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
