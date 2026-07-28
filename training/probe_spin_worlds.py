"""Diagnostic: can the phase_f policy spin, and in which physics?

Rolls held pure-spin commands in (a) the plain measurement world and
(b) the world the foot-friction DR actually trained in (feet geom_priority
= 1, same mu), to separate "policy unlearned turning" from "policy turns
only in the DR-patched contact physics".
"""

import json
import math
import sys
from pathlib import Path

import jax
import numpy as np
from mujoco import mjx

from wojtek_rl import paths
from wojtek_rl.battery import load_checkpoint_policy
from wojtek_rl.courses.rollout import spin_rollout
from wojtek_rl.courses.spec import SpinCourse

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/wojtek_stiff_f_sym_20260726_seed1")

run, env, ckpt, inf = load_checkpoint_policy(RUN)
print(f"run {RUN.name} ckpt {ckpt.name}", flush=True)

SPINS = [SpinCourse("spin_left", "wz+", 0.8), SpinCourse("spin_right", "wz-", -0.8)]


def probe(env, label):
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    for spin in SPINS:
        rec, info = spin_rollout(env, reset, step, inf, spin, seed=0)
        yp = np.asarray(rec.get("yaw_progress", []))
        prog = math.degrees(float(yp[-1])) if yp.size else float("nan")
        print(
            f"[{label}] {spin.name:10s} wz={spin.wz:+.1f}  "
            f"progress {prog:+7.1f} deg  completed={info.get('completed')}  "
            f"fell={info.get('fell_at') is not None}",
            flush=True,
        )


probe(env, "plain")

foot_ids = [env.mj_model.geom(f"{leg}_foot_sphere").id for leg in paths.LEGS]
env.mj_model.geom_priority[foot_ids] = 1
env._mjx_model = mjx.put_model(env._mj_model, impl=env._backend)
probe(env, "foot-priority")
