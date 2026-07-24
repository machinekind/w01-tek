"""Hydra-configured Brax PPO trainer for the four_bar_bot tasks.

Experiments are Hydra configs (wojtek_rl/conf): pick a task group or an
experiment preset and override anything from the CLI, e.g.

    ./run.sh train +experiment=getup
    ./run.sh train task=jump task.env.jump_at_steps=[40,80] seed=1
    ./run.sh train +experiment=run ppo.num_timesteps=3e8 run_name=fbb_run_v2

PPO hyper-parameters start from the playground's tuned Go1 config;
`task.ppo` then global `ppo` yaml/CLI blocks override them.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import functools
import json
import time
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

from wojtek_rl import paths


def _apply_ppo_overrides(p, overrides: dict) -> None:
    for k, v in (overrides or {}).items():
        if isinstance(v, dict):
            _apply_ppo_overrides(getattr(p, k), v)
        else:
            if isinstance(v, float) and v.is_integer() and not isinstance(
                getattr(p, k, None), float
            ):
                v = int(v)  # ppo.num_timesteps=3e8 arrives as float
            setattr(p, k, v)


def build_ppo_params(overrides, smoke: bool):
    """Playground Go1 PPO config + overrides.

    `overrides` is either a dict (hydra path) or a legacy ["k=v", ...] list
    (app.py / eval.py still call it that way).
    """
    from mujoco_playground.config import locomotion_params

    p = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")
    p.network_factory.policy_obs_key = "state"
    p.network_factory.value_obs_key = "privileged_state"
    if smoke:
        p.num_timesteps = 100_000
        p.num_envs = 64
        p.batch_size = 64
        p.num_minibatches = 4
        p.num_evals = 2
    if isinstance(overrides, dict):
        _apply_ppo_overrides(p, overrides)
    else:
        for kv in overrides or []:
            k, v = kv.split("=", 1)
            for cast in (int, float):
                try:
                    v = cast(v)
                    break
                except ValueError:
                    continue
            setattr(p, k, v)
    return p


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    from wojtek_rl.randomize import make_domain_randomize
    from wojtek_rl.registry import make_env

    task = cfg.task.name
    env_overrides = OmegaConf.to_container(cfg.task.env, resolve=True) or {}

    # PPO params resolve before the envs because the warp backend sizes its
    # contact buffer across the whole training batch, so the env config
    # needs the final num_envs at construction time.
    ppo_params = build_ppo_params({}, cfg.smoke)
    # Network group (conf/network) merges into the factory kwargs; explicit
    # ppo.network_factory.* overrides still win below.
    _apply_ppo_overrides(
        ppo_params.network_factory,
        OmegaConf.to_container(cfg.network, resolve=True) or {},
    )
    task_ppo_overrides = OmegaConf.to_container(cfg.task.ppo, resolve=True) or {}
    ppo_overrides = OmegaConf.to_container(cfg.ppo, resolve=True) or {}
    _apply_ppo_overrides(ppo_params, task_ppo_overrides)
    _apply_ppo_overrides(ppo_params, ppo_overrides)

    # The eval wrapper vmaps num_eval_envs worlds through the same env, so
    # the warp contact budget covers the larger of the two batches.
    env_overrides.setdefault("sim", {})["num_envs"] = int(
        max(ppo_params.num_envs, ppo_params.get("num_eval_envs", 0))
    )
    env = make_env(task, env_overrides)
    eval_env = make_env(task, env_overrides)
    print(f"actor obs ({len(env.actor_obs_names)} components): {env.actor_obs_names}")

    # Episode length follows the env config unless ppo yaml overrides it.
    # Smoke keeps episodes short too: brax's eval unrolls full episodes, so
    # a 1000-step episode dominates a CPU pipeline check.
    if "episode_length" not in task_ppo_overrides and "episode_length" not in ppo_overrides:
        ppo_params.episode_length = (
            min(200, env._config.episode_length) if cfg.smoke
            else env._config.episode_length
        )

    restore = cfg.restore
    if restore and not os.path.isabs(restore):
        restore = str(paths.PROJECT_DIR / restore)

    run_name = cfg.run_name or (
        ("smoke_" if cfg.smoke else "")
        + f"{task}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = paths.PROJECT_DIR / "runs" / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    training_params = dict(ppo_params)
    network_factory_cfg = training_params.pop("network_factory")
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **network_factory_cfg
    )
    dr_cfg = OmegaConf.to_container(cfg.dr, resolve=True)
    if cfg.domain_rand:
        training_params["randomization_fn"] = make_domain_randomize(
            env.mj_model, dr_cfg
        )
    else:
        enabled = [
            k for k, v in dr_cfg.items() if isinstance(v, dict) and v.get("enable")
        ]
        if enabled:
            raise ValueError(
                f"domain_rand=false disables the whole dr block, but "
                f"cfg.dr fields {enabled} have enable=true; either set "
                f"domain_rand=true or disable those fields too"
            )

    wb = None
    if cfg.wandb.enable:
        try:
            import wandb

            wb = wandb.init(
                project=cfg.wandb.project,
                name=run_name,
                config={
                    "hydra": OmegaConf.to_container(cfg, resolve=True),
                    "ppo": dict(ppo_params),
                    "env": env._config.to_dict(),
                },
            )
        except Exception as e:  # noqa: BLE001
            print(f"wandb disabled: {e}")

    t_last = [time.time(), 0]

    def progress(num_steps: int, metrics: dict) -> None:
        now = time.time()
        sps = (num_steps - t_last[1]) / max(now - t_last[0], 1e-9)
        t_last[0], t_last[1] = now, num_steps
        reward = metrics.get("eval/episode_reward", float("nan"))
        # avg_episode_length exposes die-and-reset reward hacking that the
        # reward number alone hides (trot_v1 post-mortem, 2026-07-05).
        ep_len = metrics.get("eval/avg_episode_length", float("nan"))
        line = (
            f"steps {num_steps:>12,}  reward {reward:8.2f}  "
            f"ep_len {ep_len:6.0f}  {sps:,.0f} steps/s"
        )
        # The eval metric is the eval env's level: that env resets to the
        # initial spawn distribution every evaluation, so this tracks the eval
        # spawn spread, not the training curriculum climbing (it hovers near
        # the init mean). The real curriculum signal is the training env's
        # level, which brax surfaces on a separate progress call under
        # ppo.log_training_metrics as episode/terrain_level_per_step (its
        # EpisodeMetricsLogger divides the `_per_step` metric by episode
        # length). Print both when present, labelled distinctly.
        level = metrics.get("eval/episode_terrain_level_per_step")
        if level is not None:
            line += f"  terrain_lvl {float(level):5.2f}"
        train_level = metrics.get("episode/terrain_level_per_step")
        if train_level is not None:
            line += f"  terrain_lvl_train {float(train_level):5.2f}"
        print(line)
        if wb is not None:
            wb.log({**metrics, "perf/steps_per_sec": sps}, step=num_steps)

    # Terrain runs need the curriculum auto-reset wrapper (promote/demote +
    # teleport on done); the flat path keeps brax's stock wrapper exactly.
    if getattr(env, "_terrain_enabled", False):
        from wojtek_rl.terrain_wrapper import wrap_for_terrain_brax_training

        wrap_env_fn = wrap_for_terrain_brax_training
    else:
        wrap_env_fn = wrapper.wrap_for_brax_training

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=cfg.seed,
        wrap_env_fn=wrap_env_fn,
        save_checkpoint_path=str(ckpt_dir),
        restore_checkpoint_path=restore,
        progress_fn=progress,
    )
    make_inference_fn, params, metrics = train_fn(
        environment=env, eval_env=eval_env
    )

    # Effective post-customize gains (uniform across actuators; DR excluded).
    kp_eff = float(env.mj_model.actuator_gainprm[0, 0])
    kd_eff = float(-env.mj_model.actuator_biasprm[0, 2])

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "task": task,
                "num_timesteps": int(ppo_params.num_timesteps),
                "final_reward": float(
                    metrics.get("eval/episode_reward", float("nan"))
                ),
                "checkpoint_dir": str(ckpt_dir),
                "env_config": env._config.to_dict(),
                "ppo_config": ppo_params.to_dict(),
                "hydra_config": OmegaConf.to_container(cfg, resolve=True),
                "kp": kp_eff,
                "kd": kd_eff,
            },
            indent=2,
            default=str,
        )
    )
    print(f"done -> {run_dir}")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
