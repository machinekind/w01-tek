"""Export a Brax PPO checkpoint to a plain .npz + policy_meta.json for ROS.

The deploy stack (ros/, the wojtek_policy package) runs the policy with
numpy only -- no jax/brax on the robot. This script writes:

  policy.npz        norm_mean/norm_std + hidden_{i}_kernel/bias MLP layers
  policy_meta.json  the schema-2 deployment contract (see deploy_contract.py)

The contract is built from a live env constructed with the run's exact
env_config, so every number in it is the training-time value. Two
validations run before anything is written:

  1. the numpy MLP forward pass against the jitted brax inference fn
  2. the full deploy runtime (ros/.../wojtek_policy/policy.py, loaded via
     np_policy) against a reference pipeline computed from the env's own
     fields -- obs assembly, action filter, anchor, scale, and clip bounds
     all round-trip through the written artifacts

Run: ./run.sh export --run runs/<name> [--out <dir>]
Upload a keeper's artifacts next to its checkpoint on HF afterwards:
  hf upload <HF_ORGANIZATION>/<keeper> <out-dir> . --include "policy*"
"""

import argparse
import json
from pathlib import Path

import numpy as np

ACTION_SIZE = 12


def numpy_policy(params, obs):
    """Numpy mirror of the exported network. Keep in sync with the runner."""
    x = (obs - params["norm_mean"]) / params["norm_std"]
    i = 0
    while f"hidden_{i}_kernel" in params:
        x = x @ params[f"hidden_{i}_kernel"] + params[f"hidden_{i}_bias"]
        if f"hidden_{i + 1}_kernel" in params:  # SiLU on all but the last
            x = x * np.exp(-np.logaddexp(0.0, -x))
        i += 1
    loc = x[:ACTION_SIZE]
    return np.tanh(loc)


def build_env(run: dict):
    """The run's env, rebuilt for contract extraction on this host.

    sim.* is training-only (deploy_contract.py), so forcing the jax backend
    and one env changes nothing the contract reads -- and keeps the exporter
    off warp/GPU paths.
    """
    from wojtek_rl.registry import make_env

    env_config = dict(run.get("env_config", {}))
    sim = dict(env_config.get("sim", {}))
    sim.update(backend="jax", num_envs=1)
    env_config["sim"] = sim
    return make_env(run.get("task", "joystick"), env_config)


def validate_deploy_runtime(env, meta, out_dir, inference, key, priv_size):
    """Run the actual deploy runtime against an env-derived reference.

    Feeds both the same sensor streams for a rollout of steps; the deploy
    side goes through the written policy.npz + policy_meta.json, the
    reference through the env's own resolved fields and the brax inference
    fn. Catches every class of drift PR #30 hand-patched: obs assembly
    order, command fill, anchor, scale form, clip bounds, filter state.
    """
    from wojtek_rl.np_policy import load_policy_runtime

    policy = load_policy_runtime(out_dir / "policy.npz")

    home = np.asarray(env._home_ctrl, np.float32)
    anchor = np.asarray(meta["anchor_ctrl"], np.float32)
    scale = np.asarray(meta["action_scale"], np.float32)
    lo = np.asarray(meta["target_low"], np.float32)
    hi = np.asarray(meta["target_high"], np.float32)
    af = float(meta["action_filter"])
    cmd_lo = np.asarray(meta["command_low"], np.float32)
    cmd_hi = np.asarray(meta["command_high"], np.float32)
    fill = np.asarray(meta["command_fill"], np.float32)
    names = env.actor_obs_names

    rng = np.random.default_rng(0)
    last_act = np.zeros(ACTION_SIZE, np.float32)
    filt = np.zeros(ACTION_SIZE, np.float32)
    worst = 0.0
    for _ in range(32):
        gyro = rng.uniform(-2, 2, 3).astype(np.float32)
        grav = rng.uniform(-1, 1, 3).astype(np.float32)
        q = (home + rng.uniform(-0.4, 0.4, 12)).astype(np.float32)
        dq = rng.uniform(-6, 6, 12).astype(np.float32)
        cmd3 = rng.uniform(cmd_lo[:3], cmd_hi[:3]).astype(np.float32)

        parts = {
            "gyro": gyro,
            "gravity": grav,
            "joint_pos": q - home,
            "joint_vel": dq,
            "last_act": last_act,
            "command": np.concatenate([cmd3, fill]),
        }
        obs = np.concatenate([parts[n] for n in names]).astype(np.float32)
        ref_act, _ = inference(
            {"state": obs, "privileged_state": np.zeros(priv_size, np.float32)},
            key,
        )
        ref_act = np.asarray(ref_act, np.float32)
        last_act = ref_act
        filt = af * filt + (1.0 - af) * ref_act
        ref_targets = np.clip(anchor + filt * scale, lo, hi)

        got = policy.step(gyro, grav, q, dq, cmd3)
        worst = max(worst, float(np.max(np.abs(ref_targets - got))))
    assert worst < 1e-5, f"deploy runtime vs env reference: max |diff| = {worst}"
    print(f"validated deploy runtime vs env reference: max |diff| = {worst:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing run.json")
    ap.add_argument("--out", default=None, help="output dir (default <run>/deploy)")
    args = ap.parse_args()

    import functools

    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks

    from wojtek_rl import paths
    from wojtek_rl.deploy_contract import build_contract
    from wojtek_rl.train import build_ppo_params

    run_dir = Path(args.run)
    run = json.loads((run_dir / "run.json").read_text())
    ckpt_root = Path(run["checkpoint_dir"])
    if not ckpt_root.is_absolute():
        ckpt_root = paths.PROJECT_DIR / ckpt_root
    if not ckpt_root.exists():
        # run.json may carry the training host's absolute checkpoint path
        # (cluster runs); fall back to the run dir itself.
        ckpt_root = (run_dir / "checkpoints").resolve()
    ckpt = max(
        (p for p in ckpt_root.iterdir() if p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    out_dir = Path(args.out) if args.out else run_dir / "deploy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- contract from a live env built with the run's exact config ---------
    env = build_env(run)
    meta = build_contract(env, run, checkpoint=str(ckpt))

    # -- rebuild the inference fn exactly as eval.py does (sans env) --------
    ppo_params = build_ppo_params([], smoke=False)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )
    # Obs sizes vary per run -- read them from the checkpoint instead of
    # hardcoding, and cross-check against the contract.
    net_cfg = json.loads((ckpt / "ppo_network_config.json").read_text())
    obs_size = {
        k: tuple(v["shape"]) for k, v in net_cfg["observation_size"].items()
    }
    state_size = obs_size["state"][0]
    priv_size = obs_size["privileged_state"][0]
    assert state_size == meta["obs_size"], (
        f"checkpoint actor obs is {state_size}-D but the env config resolves "
        f"to {meta['obs_size']}-D ({meta['obs_layout']}) -- run.json and the "
        "checkpoint disagree"
    )
    ppo_network = network_factory(
        obs_size, ACTION_SIZE,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    params = ppo_checkpoint.load(str(ckpt))
    inference = jax.jit(make_inference_fn(params, deterministic=True))

    normalizer, policy_params = params[0], params[1]

    # -- flatten to numpy ----------------------------------------------------
    export = {
        "norm_mean": np.asarray(normalizer.mean["state"], np.float32),
        "norm_std": np.asarray(normalizer.std["state"], np.float32),
    }
    layers = policy_params["params"]
    for i, name in enumerate(sorted(layers, key=lambda n: int(n.split("_")[-1]))):
        export[f"hidden_{i}_kernel"] = np.asarray(layers[name]["kernel"], np.float32)
        export[f"hidden_{i}_bias"] = np.asarray(layers[name]["bias"], np.float32)

    # -- validate numpy vs brax ---------------------------------------------
    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(0)
    worst = 0.0
    for _ in range(32):
        obs = rng.uniform(-2.0, 2.0, state_size).astype(np.float32)
        ref, _ = inference(
            {"state": obs, "privileged_state": np.zeros(priv_size, np.float32)},
            key,
        )
        got = numpy_policy(export, obs)
        worst = max(worst, float(np.max(np.abs(np.asarray(ref) - got))))
    assert worst < 1e-4, f"numpy forward mismatch vs brax: {worst}"
    print(f"validated numpy vs brax inference: max |diff| = {worst:.2e}")

    np.savez(out_dir / "policy.npz", **export)
    (out_dir / "policy_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # -- validate the deploy runtime on the written artifacts ---------------
    validate_deploy_runtime(env, meta, out_dir, inference, key, priv_size)
    print(f"wrote {out_dir}/policy.npz and policy_meta.json (schema 2)")


if __name__ == "__main__":
    main()
