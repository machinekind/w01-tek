"""Export a Brax PPO checkpoint to a plain .npz for ROS deployment.

The deploy stack (ros/, the wojtek_policy package) runs the policy with numpy only -- no
jax/brax on the robot. This script flattens the checkpoint into:

  norm_mean/norm_std       observation normalizer for the "state" key (53,)
  hidden_{i}_kernel/bias   policy MLP layers (512/256/128 + final)
  plus a JSON metadata sidecar (home ctrl, ctrlrange, action scale, obs
  layout, gait clock) so the runner needs no wojtek_rl imports.

Run: ./run.sh export --run policies/fbb_v3 --out policies/fbb_v3/deploy
Validates the numpy forward pass against the brax inference fn before
writing anything.
"""

import argparse
import json
from pathlib import Path

import numpy as np


OBS_SIZE = 53
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing run.json")
    ap.add_argument("--out", default=None, help="output dir (default <run>/deploy)")
    args = ap.parse_args()

    import jax
    import mujoco
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks

    from wojtek_rl import paths
    from wojtek_rl.train import build_ppo_params

    run_dir = Path(args.run)
    run = json.loads((run_dir / "run.json").read_text())
    ckpt_root = Path(run["checkpoint_dir"])
    if not ckpt_root.is_absolute():
        ckpt_root = paths.PROJECT_DIR / ckpt_root
    ckpt = max(
        (p for p in ckpt_root.iterdir() if p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    out_dir = Path(args.out) if args.out else run_dir / "deploy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- rebuild the inference fn exactly as eval.py does (sans env) --------
    ppo_params = build_ppo_params([], smoke=False)
    import functools

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )
    # Obs sizes vary per run (e.g. the no-IMU actor is 48-D, fbb_v3 was
    # 53-D) -- read them from the checkpoint instead of hardcoding.
    net_cfg = json.loads((ckpt / "ppo_network_config.json").read_text())
    obs_size = {
        k: tuple(v["shape"]) for k, v in net_cfg["observation_size"].items()
    }
    state_size = obs_size["state"][0]
    priv_size = obs_size["privileged_state"][0]
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

    # -- metadata the runner needs (no wojtek_rl imports at deploy time) --------
    mj = mujoco.MjModel.from_xml_path(str(paths.SCENE_XML))
    key_home = mj.key("home")
    env_cfg = run.get("env_config", {})
    gait_freq = env_cfg.get("gait", {}).get("freq", [1.5, 2.5])
    # Actor obs = obs.state filtered by the obs.include whitelist (see
    # base.py). Fixed-width components; command absorbs the remainder so a
    # 3-D (vx,vy,wz) and a 4-D (+height) command both come out right.
    obs_cfg = env_cfg.get("obs", {})
    state_names = list(obs_cfg.get("state", []))
    include = obs_cfg.get("include", [])
    if include:
        state_names = [n for n in state_names if n in include]
    widths = {"gyro": 3, "gravity": 3, "joint_pos": 12, "joint_vel": 12,
              "last_act": 12, "phase": 8}
    cmd_size = state_size - sum(widths.get(n, 0) for n in state_names)
    obs_layout = [
        f"{n}:{widths.get(n, cmd_size)}" for n in state_names
    ] if state_names else [f"state:{state_size}"]
    meta = {
        "run_name": run["run_name"],
        "checkpoint": str(ckpt),
        "obs_size": state_size,
        "action_size": ACTION_SIZE,
        "obs_layout": obs_layout,
        "actuator_names": [mj.actuator(i).name for i in range(mj.nu)],
        "home_ctrl": np.asarray(key_home.ctrl).tolist(),
        "ctrl_low": mj.actuator_ctrlrange[:, 0].tolist(),
        "ctrl_high": mj.actuator_ctrlrange[:, 1].tolist(),
        "action_scale": float(env_cfg.get("action_scale", 0.5)),
        "ctrl_dt": float(env_cfg.get("ctrl_dt", 0.02)),
        "trot_phase": [0.0, np.pi, 0.0, np.pi],
        "gait_freq_hz": float(np.mean(gait_freq)),
        "knee_singularity": 3.2,
    }

    np.savez(out_dir / "policy.npz", **export)
    (out_dir / "policy_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_dir}/policy.npz and policy_meta.json")


if __name__ == "__main__":
    main()
