"""Command-routed policy switcher: the distillation seam test (step 0).

Two frozen policies of the terrain family run as one composite: on a
pure-spin or stand command window the spin donor acts, everywhere else
the base (keeper) does. The window test is the flat_pitch_spin_exempt
construction from env.py -- vx and vy at exact zero -- extended to cover
stand (wz zero too), so the router here is the command half of the
planned DAgger router and nothing more. No training, no new network:
the composite exists to measure what a mid-episode style switch costs.

Both policies must share the actor observation layout and the action
contract; `attach_switch` refuses anything else. The privileged lists
may differ (v4.2 still carries phase, v4.5/v4.6 do not): brax's policy
network reads only obs["state"] and selects only the "state" normalizer
stats, so the value-side layout never enters inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import jax.numpy as jp

from wojtek_rl import symmetry

# Mirrors the flat_pitch_spin_exempt window test (env.py): the command
# samplers and the course follower's spin branch write exact zeros, so
# this is a window-type test, not a threshold tune.
PLANAR_TOL = 1e-6

# The action-contract fields a composite must agree on: a policy trained
# under one contract sends garbage through another's plant.
CONTRACT_KEYS = (
    "action_scale",
    "action_filter",
    "knee_target_max",
    "abduction_ctrl_limit",
    "pd_kp",
    "pd_kd",
    "max_torque",
)


def command_slice(actor_obs_names) -> slice:
    """Where the 4-dim command sits in the flat actor obs vector."""
    if "command" not in actor_obs_names:
        raise ValueError(
            f"actor obs {list(actor_obs_names)} has no 'command' component; "
            "the switcher routes on it"
        )
    start = 0
    for name in actor_obs_names:
        if name == "command":
            return slice(start, start + symmetry.COMPONENT_SIZES["command"])
        start += symmetry.COMPONENT_SIZES[name]
    raise AssertionError("unreachable")


def switch_inference(base_inf, spin_inf, cmd_slice: slice):
    """Compose two inference fns behind the command-window router.

    Batch-safe: the window mask broadcasts over any leading axes (the
    terrain scan batches cells, the battery rolls out single envs). Both
    nets evaluate every step and jp.where selects -- under jit that is
    two MLP passes, which the eval tools can afford, and it keeps the
    composite a pure function of (obs, key) like any other policy.
    Extras come from the acting branch's side only in the deterministic
    eval tools' sense: they never read extras, so base's are passed
    through unchanged.
    """

    def inf(obs, key):
        state = obs["state"] if isinstance(obs, Mapping) else obs
        cmd = state[..., cmd_slice]
        window = (jp.abs(cmd[..., 0]) < PLANAR_TOL) & (
            jp.abs(cmd[..., 1]) < PLANAR_TOL
        )
        base_act, extras = base_inf(obs, key)
        spin_act, _ = spin_inf(obs, key)
        return jp.where(window[..., None], spin_act, base_act), extras

    return inf


def _latest_checkpoint(run_dir: Path, run: dict) -> Path:
    """The same latest-checkpoint choice load_checkpoint_policy makes."""
    ckpt_dir = Path(run.get("checkpoint_dir", ""))
    if not ckpt_dir.exists():
        ckpt_dir = (run_dir / "checkpoints").resolve()
    return max(
        (p for p in ckpt_dir.iterdir() if p.name.isdigit()),
        key=lambda p: int(p.name),
    )


def _plain(value):
    """Sequences as lists, recursively: a config dict compares equal to
    its JSON round-trip (to_dict gives tuples, run.json gives lists)."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def check_compatible(base_run: dict, spin_run: dict) -> None:
    """Refuse a composite across obs layouts or action contracts."""
    base_env = base_run.get("env_config") or {}
    spin_env = spin_run.get("env_config") or {}
    base_obs = list((base_env.get("obs") or {}).get("include") or ())
    spin_obs = list((spin_env.get("obs") or {}).get("include") or ())
    if base_obs != spin_obs:
        raise ValueError(
            f"actor obs layouts differ: base {base_obs} vs spin {spin_obs}"
        )
    bad = [
        k
        for k in CONTRACT_KEYS
        if _plain(base_env.get(k)) != _plain(spin_env.get(k))
    ]
    if bad:
        detail = {
            k: (base_env.get(k), spin_env.get(k)) for k in bad
        }
        raise ValueError(f"action contracts differ on {detail}")


def attach_switch(run: dict, env, base_inf, spin_run_dir: Path, ppo_params):
    """Wrap `base_inf` with the spin donor loaded from `spin_run_dir`.

    Returns (run, inf): `run` gains a `seam` record (spin run name and
    checkpoint) so every results json downstream says what actually ran.
    The spin network is rebuilt against the BASE env's sizes -- legal
    because the actor layouts are asserted identical and inference never
    touches the value network (module docstring).
    """
    from wojtek_rl.policy_io import load_policy

    spin_run_dir = Path(spin_run_dir)
    spin_run = json.loads((spin_run_dir / "run.json").read_text())
    check_compatible(run, spin_run)
    spin_ckpt = _latest_checkpoint(spin_run_dir, spin_run)
    spin_inf = load_policy(spin_ckpt, env, ppo_params)
    inf = switch_inference(base_inf, spin_inf, command_slice(env.actor_obs_names))
    run = dict(
        run,
        seam={
            "router": "command-window (vx==0 and vy==0 -> spin policy)",
            "base_run": run.get("run_name"),
            "spin_run": spin_run.get("run_name"),
            "spin_checkpoint": spin_ckpt.name,
        },
    )
    return run, inf
