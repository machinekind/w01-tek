"""DAgger distillation: one student regressed onto routed frozen teachers.

The student rolls out in the terrain training env; at every visited
state the teacher the router assigns to that env provides its pre-tanh
action mean (loc), and the student minimizes MSE against it. No reward
is optimized; the env's rewards are logged only.

Router, two modes (distill.routing.mode):
  command (per step, on the command channel the student observes):
    vx=vy=0 window (pure spin/stand) -> routing.command_window teacher
    any moving command               -> routing.moving teacher
  tile (per episode, simulation-side; the student never observes it):
    terrain_level == 0 (the flat row) -> routing.flat_row teacher
    terrain rows                      -> routing.types[tile type]
Tile-routed teachers are separable by the student only if its
observations can tell the tiles apart.

The loss targets the pre-tanh loc rather than the squashed action
because tanh saturation flattens the gradient exactly where a teacher
drives a joint hardest. With probability beta (redrawn per episode,
annealed to zero) an env is driven by its teacher instead of the
student, so the batch also contains states only a competent policy
reaches; the warm start covers the cold-start problem.

The run directory is a standard run: run.json + brax PPO checkpoints,
so courses/battery/terrain_scan/export work on the student unchanged.
The observation normalizer is frozen at the warm start's statistics.

Usage mirrors train.py:

    ./run.sh distill +experiment=terrain_distill seed=1 \
        run_name=wojtek_terrain_distill_s1
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
# The trainer is terrain-only, and terrain runs need warp's EPA scratch
# outside the XLA pool.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOC", "false")

import functools
import json
import time
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from wojtek_rl import paths


def command_router(routing: dict, teacher_names: list) -> tuple:
    """(window_idx, moving_idx) for routing.mode=command, validated.

    The window teacher labels every state whose command has vx and vy at
    exact zero (pure spin or stand; the flat_pitch_spin_exempt window
    construction), the moving teacher labels everything else. Both live
    in the command channel the student observes, so no label depends on
    anything the student cannot see.
    """
    by_name = {n: i for i, n in enumerate(teacher_names)}
    missing = [k for k in ("command_window", "moving") if not routing.get(k)]
    if missing:
        raise ValueError(f"routing.mode=command needs routing.{missing}")
    bad = {k: routing[k] for k in ("command_window", "moving")
           if routing[k] not in by_name}
    if bad:
        raise ValueError(f"routing names unknown teachers: {bad}")
    return by_name[routing["command_window"]], by_name[routing["moving"]]


def teacher_table(routing: dict, teacher_names: list, types: tuple) -> tuple:
    """(flat_idx, per-type idx list) from the routing config, validated.

    Every terrain type must be routed and every routed name must be a
    teacher: a hole would silently label a whole tile family with
    whatever teacher index fell out of a default.
    """
    by_name = {n: i for i, n in enumerate(teacher_names)}
    flat = routing.get("flat_row")
    if flat not in by_name:
        raise ValueError(f"routing.flat_row={flat!r} is not one of {teacher_names}")
    tmap = dict(routing.get("types") or {})
    missing = [t for t in types if t not in tmap]
    extra = [t for t in tmap if t not in types]
    if missing or extra:
        raise ValueError(
            f"routing.types must cover exactly {list(types)}; "
            f"missing {missing}, unknown {extra}"
        )
    bad = {t: n for t, n in tmap.items() if n not in by_name}
    if bad:
        raise ValueError(f"routing.types names unknown teachers: {bad}")
    return by_name[flat], [by_name[tmap[t]] for t in types]


def anneal(start: float, end: float, it: int, total: int) -> float:
    """Linear schedule from start to end over `total` iterations."""
    if total <= 1:
        return end
    frac = min(max(it / (total - 1), 0.0), 1.0)
    return start + (end - start) * frac


def select_rows(stacked, idx, xp=None):
    """Pick stacked[idx[n], n] from a (K, N, ...) stack via one-hot.

    gather with a data-dependent leading index compiles to a scatter on
    GPU; the einsum stays a plain matmul.
    """
    import jax.numpy as jp

    xp = xp or jp
    onehot = (idx[:, None] == xp.arange(stacked.shape[0])[None, :]).astype(
        stacked.dtype
    )
    return xp.einsum("kn...,nk->n...", stacked, onehot)


def latest_checkpoint(run_dir: Path, run: dict) -> Path:
    """The same latest-checkpoint choice battery.load_checkpoint_policy makes."""
    ckpt_dir = Path(run.get("checkpoint_dir", ""))
    if not ckpt_dir.exists():
        ckpt_dir = (run_dir / "checkpoints").resolve()
    return max(
        (p for p in ckpt_dir.iterdir() if p.name.isdigit()),
        key=lambda p: int(p.name),
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    import jax
    import jax.numpy as jp
    import numpy as np
    import optax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks

    from wojtek_rl import seam, terrain
    from wojtek_rl.randomize import make_domain_randomize
    from wojtek_rl.registry import make_env
    from wojtek_rl.terrain_wrapper import wrap_for_terrain_brax_training
    from wojtek_rl.train import build_ppo_params

    dcfg = OmegaConf.to_container(cfg.distill, resolve=True) or {}
    for key in ("teachers", "routing", "warm_start"):
        if not dcfg.get(key):
            raise ValueError(f"distill.{key} is required (preset terrain_distill)")

    task = cfg.task.name
    env_overrides = OmegaConf.to_container(cfg.task.env, resolve=True) or {}

    # PPO params carry the network factory, num_envs and the step budget;
    # the PPO update hyper-parameters are dead weight here. Resolution
    # order mirrors train.py so presets read the same.
    ppo_params = build_ppo_params({}, cfg.smoke)
    from wojtek_rl.train import _apply_ppo_overrides

    _apply_ppo_overrides(
        ppo_params.network_factory,
        OmegaConf.to_container(cfg.network, resolve=True) or {},
    )
    _apply_ppo_overrides(
        ppo_params, OmegaConf.to_container(cfg.task.ppo, resolve=True) or {}
    )
    _apply_ppo_overrides(
        ppo_params, OmegaConf.to_container(cfg.ppo, resolve=True) or {}
    )
    num_envs = int(ppo_params.num_envs)
    num_timesteps = int(ppo_params.num_timesteps)

    env_overrides.setdefault("sim", {})["num_envs"] = num_envs
    env = make_env(task, env_overrides)
    if not getattr(env, "_terrain_enabled", False):
        raise ValueError("distillation v1 runs on the terrain env only")
    if not env._terrain.flat_row:
        raise ValueError(
            "routing.flat_row needs terrain.flat_row=true (level 0 IS the "
            "flat row; without it level 0 is a terrain rung)"
        )
    episode_length = int(env._config.episode_length)
    action_size = env.action_size
    print(f"actor obs ({len(env.actor_obs_names)} components): {env.actor_obs_names}")

    # ---------------------------------------------------------- teachers
    teacher_names = list(dcfg["teachers"])
    teacher_runs = {}
    student_env_probe = {"env_config": env._config.to_dict()}
    for name in teacher_names:
        run_dir = paths.PROJECT_DIR / dcfg["teachers"][name]
        run = json.loads((run_dir / "run.json").read_text())
        # Same refusal the seam test uses: a teacher with another obs
        # layout or action contract would label garbage.
        seam.check_compatible(student_env_probe, run)
        teacher_runs[name] = (run_dir, run)

    routing_mode = str(dcfg["routing"].get("mode", "tile"))
    if routing_mode == "command":
        window_idx, moving_idx = command_router(dcfg["routing"], teacher_names)
    elif routing_mode == "tile":
        flat_idx, type_idx = teacher_table(
            dcfg["routing"], teacher_names, terrain.TYPES
        )
        type_idx = jp.array(type_idx)
    else:
        raise ValueError(f"routing.mode must be tile or command, got {routing_mode!r}")

    network_factory_cfg = dict(ppo_params.network_factory)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **network_factory_cfg
    )
    normalize = (
        running_statistics.normalize
        if ppo_params.get("normalize_observations", True)
        else lambda obs, _: obs
    )
    ppo_network = network_factory(
        env.observation_size, action_size, preprocess_observations_fn=normalize
    )
    apply_policy = ppo_network.policy_network.apply
    dist = ppo_network.parametric_action_distribution

    teacher_params = [
        ppo_checkpoint.load(str(latest_checkpoint(*teacher_runs[n])))
        for n in teacher_names
    ]

    # ----------------------------------------------------------- student
    warm_dir = paths.PROJECT_DIR / dcfg["warm_start"]
    warm_run = json.loads((warm_dir / "run.json").read_text())
    seam.check_compatible(student_env_probe, warm_run)
    warm_ckpt = latest_checkpoint(warm_dir, warm_run)
    warm_params = ppo_checkpoint.load(str(warm_ckpt))
    student_norm = warm_params[0]  # frozen (module docstring)
    policy_params = warm_params[1]
    value_params = warm_params[2]  # untrained ballast, kept for the format

    # ------------------------------------------------------ env wrapping
    key = jax.random.PRNGKey(int(cfg.seed))
    key, key_rand, key_reset, key_mix = jax.random.split(key, 4)
    v_rand_fn = None
    dr_cfg = OmegaConf.to_container(cfg.dr, resolve=True)
    if cfg.domain_rand:
        v_rand_fn = functools.partial(
            make_domain_randomize(env.mj_model, dr_cfg),
            rng=jax.random.split(key_rand, num_envs),
        )
    wenv = wrap_for_terrain_brax_training(
        env,
        episode_length=episode_length,
        action_repeat=1,
        randomization_fn=v_rand_fn,
    )

    # ------------------------------------------------------- run records
    run_name = cfg.run_name or (
        "distill_" + task + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = paths.PROJECT_DIR / "runs" / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_record = {
        "run_name": run_name,
        "task": task,
        "status": "running",
        "num_timesteps": num_timesteps,
        "final_reward": None,
        "checkpoint_dir": str(ckpt_dir),
        "env_config": env._config.to_dict(),
        "ppo_config": ppo_params.to_dict(),
        "hydra_config": OmegaConf.to_container(cfg, resolve=True),
        "kp": float(env.mj_model.actuator_gainprm[0, 0]),
        "kd": float(-env.mj_model.actuator_biasprm[0, 2]),
        "distill": {
            "teachers": {
                n: {"run": str(teacher_runs[n][0]),
                    "checkpoint": latest_checkpoint(*teacher_runs[n]).name}
                for n in teacher_names
            },
            "routing": dcfg["routing"],
            "warm_start": {"run": str(warm_dir), "checkpoint": warm_ckpt.name},
            "loss": "mse(pre-tanh loc)",
            "normalizer": "frozen from warm_start",
        },
    }

    def write_run_json() -> None:
        (run_dir / "run.json").write_text(
            json.dumps(run_record, indent=2, default=str)
        )

    write_run_json()

    def save_checkpoint(step: int, pol, ckpt_config) -> None:
        ppo_checkpoint.save(str(ckpt_dir), step, (student_norm, pol, value_params),
                            ckpt_config)

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

    # -------------------------------------------------------- iteration
    student_act = str(dcfg.get("student_act", "sample"))
    if student_act not in ("sample", "mode"):
        raise ValueError(f"distill.student_act must be sample or mode, got {student_act!r}")
    unroll = int(dcfg.get("unroll_length", 32))
    epochs = int(dcfg.get("epochs", 2))
    num_minibatches = int(dcfg.get("num_minibatches", 32))
    lr = float(dcfg.get("learning_rate", 3e-4))
    beta0 = float(dcfg.get("teacher_act_frac_start", 0.5))
    beta1 = float(dcfg.get("teacher_act_frac_end", 0.0))
    save_every = int(dcfg.get("save_every_steps", 25_000_000))
    steps_per_iter = unroll * num_envs
    num_iters = max(1, num_timesteps // steps_per_iter)
    batch = unroll * num_envs
    if batch % num_minibatches:
        raise ValueError(
            f"unroll_length*num_envs={batch} must divide by "
            f"num_minibatches={num_minibatches}"
        )

    # Cosine decay from learning_rate to learning_rate_end over every
    # update of the run; learning_rate_end unset means a constant rate.
    lr_end = dcfg.get("learning_rate_end")
    if lr_end:
        total_updates = num_iters * epochs * num_minibatches
        lr = optax.cosine_decay_schedule(
            lr, total_updates, alpha=float(lr_end) / float(lr)
        )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(dcfg.get("grad_clip", 1.0))),
        optax.adam(lr),
    )
    opt_state = optimizer.init(policy_params)

    if routing_mode == "command":

        def route(info):
            cmd = info["command"]
            window = (jp.abs(cmd[..., 0]) < seam.PLANAR_TOL) & (
                jp.abs(cmd[..., 1]) < seam.PLANAR_TOL
            )
            return jp.where(window, window_idx, moving_idx)

    else:

        def route(info):
            return jp.where(
                info["terrain_level"] == 0, flat_idx, type_idx[info["terrain_type"]]
            )

    def teacher_loc(obs):
        locs = jp.stack(
            [apply_policy(tp[0], tp[1], obs)[..., :action_size]
             for tp in teacher_params]
        )  # (K, N, A)
        return locs

    # Per-sample loss weight for command-window states (pure spin and
    # stand). The window's actions are the smallest in the batch (a slow
    # spin barely moves the joints), so unweighted MSE sacrifices them
    # first; 1.0 keeps plain MSE.
    window_w = float(dcfg.get("window_loss_weight", 1.0))

    def loss_fn(pol, state_batch, target_loc, weights):
        logits = apply_policy(student_norm, pol, {"state": state_batch})
        err = jp.mean(jp.square(logits[..., :action_size] - target_loc), axis=-1)
        return jp.sum(err * weights) / jp.sum(weights)

    grad_fn = jax.value_and_grad(loss_fn)

    def run_iteration(env_state, pol, opt_st, mix_mask, beta, k):
        def rollout_step(carry, _):
            env_state, mix_mask, k = carry
            k, k_act, k_mix = jax.random.split(k, 3)
            obs = env_state.obs
            t_loc = select_rows(teacher_loc(obs), route(env_state.info))
            logits = apply_policy(student_norm, pol, obs)
            if student_act == "mode":
                s_act = dist.mode(logits)
            else:
                s_act = dist.postprocess(
                    dist.sample_no_postprocessing(logits, k_act)
                )
            t_act = dist.postprocess(t_loc)
            act = jp.where(mix_mask[:, None], t_act, s_act)
            cmd = env_state.info["command"]
            window = (jp.abs(cmd[..., 0]) < seam.PLANAR_TOL) & (
                jp.abs(cmd[..., 1]) < seam.PLANAR_TOL
            )
            w = jp.where(window, window_w, 1.0)
            next_state = wenv.step(env_state, act)
            done = next_state.done > 0.5
            redraw = jax.random.bernoulli(k_mix, beta, (num_envs,))
            mix_mask = jp.where(done, redraw, mix_mask)
            data = (obs["state"], t_loc, w, next_state.done, next_state.reward)
            return (next_state, mix_mask, k), data

        (env_state, mix_mask, k), (S, L, W, D, R) = jax.lax.scan(
            rollout_step, (env_state, mix_mask, k), None, length=unroll
        )
        S = S.reshape(batch, -1)
        L = L.reshape(batch, action_size)
        W = W.reshape(batch)

        def epoch(carry, k_e):
            pol, opt_st = carry
            perm = jax.random.permutation(k_e, batch)
            Sp = S[perm].reshape(num_minibatches, -1, S.shape[-1])
            Lp = L[perm].reshape(num_minibatches, -1, action_size)
            Wp = W[perm].reshape(num_minibatches, -1)

            def mb(carry2, xs):
                pol, opt_st = carry2
                s, l, w = xs
                loss, grads = grad_fn(pol, s, l, w)
                updates, opt_st = optimizer.update(grads, opt_st, pol)
                pol = optax.apply_updates(pol, updates)
                return (pol, opt_st), loss

            (pol, opt_st), losses = jax.lax.scan(mb, (pol, opt_st), (Sp, Lp, Wp))
            return (pol, opt_st), jp.mean(losses)

        k, k_ep = jax.random.split(k)
        (pol, opt_st), ep_losses = jax.lax.scan(
            epoch, (pol, opt_st), jax.random.split(k_ep, epochs)
        )
        dones = jp.sum(D)
        metrics = {
            "loss": jp.mean(ep_losses),
            "reward_per_step": jp.mean(R),
            "dones": dones,
            "terrain_level": jp.mean(
                env_state.info["terrain_level"].astype(jp.float32)
            ),
            "teacher_frac": jp.mean(mix_mask.astype(jp.float32)),
        }
        return env_state, pol, opt_st, mix_mask, k, metrics

    run_iteration = jax.jit(run_iteration, donate_argnums=(0, 1, 2, 3))

    env_state = jax.jit(wenv.reset)(jax.random.split(key_reset, num_envs))
    # Router population at reset, for the log: label share per teacher.
    idx0 = np.asarray(route(env_state.info))
    share = {n: float((idx0 == i).mean()) for i, n in enumerate(teacher_names)}
    print(f"router share at reset: {share}")

    # observation_size as specs.Array leaves, the exact shape brax's own
    # trainer serializes: export_policy reads v["shape"] out of the
    # checkpoint json, and a plain tuple has none.
    from brax.training.acme import specs

    obs_spec = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jp.dtype("float32")), env_state.obs
    )
    ckpt_config = ppo_checkpoint.network_config(
        observation_size=obs_spec,
        action_size=action_size,
        normalize_observations=bool(ppo_params.get("normalize_observations", True)),
        network_factory=network_factory,
    )
    # Step 0 snapshot: the warm start itself. A run that dies early still
    # has a loadable checkpoint, and the eval tools get a baseline.
    save_checkpoint(0, policy_params, ckpt_config)

    mix_mask = jax.random.bernoulli(key_mix, beta0, (num_envs,))
    global_step, last_save, t0 = 0, 0, time.time()
    final_reward = float("nan")
    for it in range(num_iters):
        beta = anneal(beta0, beta1, it, num_iters)
        env_state, policy_params, opt_state, mix_mask, key, m = run_iteration(
            env_state, policy_params, opt_state, mix_mask, jp.float32(beta), key
        )
        global_step += steps_per_iter
        m = {k2: float(v) for k2, v in m.items()}
        sps = steps_per_iter / max(time.time() - t0, 1e-9)
        t0 = time.time()
        ep_len = steps_per_iter / max(m["dones"], 1.0)
        final_reward = m["reward_per_step"] * episode_length
        line = (
            f"steps {global_step:>12,}  loss {m['loss']:.5f}  "
            f"ep_len~{ep_len:6.0f}  terrain_lvl_train {m['terrain_level']:5.2f}  "
            f"beta {beta:.2f}  {sps:,.0f} steps/s"
        )
        print(line, flush=True)
        if wb is not None:
            wb.log(
                {
                    "distill/loss": m["loss"],
                    "distill/teacher_frac": m["teacher_frac"],
                    "distill/beta": beta,
                    "episode/terrain_level_per_step": m["terrain_level"],
                    "episode/reward_per_step": m["reward_per_step"],
                    "episode/approx_length": ep_len,
                    "perf/steps_per_sec": sps,
                },
                step=global_step,
            )
        if global_step - last_save >= save_every:
            save_checkpoint(global_step, policy_params, ckpt_config)
            last_save = global_step

    save_checkpoint(global_step, policy_params, ckpt_config)
    run_record["status"] = "complete"
    run_record["final_reward"] = float(final_reward)
    run_record["stopped_at_steps"] = int(global_step)
    write_run_json()
    print(f"done -> {run_dir}")
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
