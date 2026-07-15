"""Roll out a checkpoint under a fixed command and render a video.

Run: ./run.sh eval --run runs/<name> --x-vel 0.3 --out walk.mp4

Or run one of the battery's fixed command scripts instead of a constant
command (--x-vel/--y-vel/--yaw-vel/--height/--steps are ignored when set):

    ./run.sh eval --run runs/<name> --scenario turn

Default output name is <scenario>.mp4 unless --out overrides it. A bare
--out filename lands in videos/<run_name>/<timestamp>/ (gitignored); pass a
path containing a directory to override.
"""

import os
import sys

if sys.platform == "linux":  # headless GPU boxes; macOS uses its default GL
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from wojtek_rl import paths
from wojtek_rl.battery import battery_scenarios


def _leg_of(name):
    return next(
        (i for i, leg in enumerate(paths.LEGS) if name.startswith(leg)), 0
    )


def _cursor_strip(fig, axes, times):
    """Render `fig` once and return frame_at(t): the strip as an RGB array
    with a moving time cursor drawn inside each of `axes`."""
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    base = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    h = base.shape[0]
    # Window extents are bottom-up; image rows are top-down.
    spans = [
        (e.x0, e.x1, int(h - e.y1), int(h - e.y0))
        for e in (ax.get_window_extent() for ax in axes)
    ]
    plt.close(fig)

    def frame_at(t):
        img = base.copy()
        frac = (t - times[0]) / max(times[-1] - times[0], 1e-9)
        for x0, x1, r0, r1 in spans:
            col = int(x0 + frac * (x1 - x0))
            img[r0:r1, max(col - 1, 0):col + 1] = (220, 40, 40)
        return img

    return frame_at


def _torque_strip(times, torques, names, torque_cap, width, height=240):
    """Full-episode torque traces as an RGB strip, plus a t -> pixel map.

    One color per leg (three joints share a hue), dashed lines at the
    +-torque_cap clamp — the deploy-critical limit should be visible in
    every video (M's format, Discord 2026-07-10).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    for j, name in enumerate(names):
        leg = _leg_of(name)
        ax.plot(times, torques[:, j], color=colors[leg], linewidth=0.6,
                label=paths.LEGS[leg] if name.endswith("first_joint") else None)
    if torque_cap:
        ax.axhline(torque_cap, color="red", linestyle="--", linewidth=0.7)
        ax.axhline(-torque_cap, color="red", linestyle="--", linewidth=0.7)
        ax.set_ylim(-1.3 * torque_cap, 1.3 * torque_cap)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("t [s]", fontsize=8)
    ax.set_ylabel("torque [N·m]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper right", fontsize=6, ncol=4, framealpha=0.5)
    fig.tight_layout(pad=0.4)
    return _cursor_strip(fig, [ax], times)


def _joint_plot(times, target, state, name, width, height=300):
    """One joint's target-vs-state trace, full width and readable — the
    zoomed-in alternative to _joint_grid, selected with --joint."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.plot(times, state, color=colors[_leg_of(name)], linewidth=1.2,
            label="state")
    ax.plot(times, target, color="black", linewidth=0.9, linestyle="--",
            label="target")
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("t [s]", fontsize=9)
    ax.set_ylabel(f"{name} [rad]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.5)
    fig.tight_layout(pad=0.4)
    return _cursor_strip(fig, [ax], times)


def _joint_grid(times, targets, joints, names, width, height=360):
    """Per-joint target-vs-state traces as an RGB strip with a time cursor.

    Rows are the leg joints (abduction, hip, knee), columns the four legs;
    achieved qpos is solid in the leg's torque-strip color, the policy's
    motor target dashed black. Rows share a y-range so left/right asymmetry
    is visible at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_names = ("abduction", "hip", "knee")

    def row_of(n):
        return 0 if "first" in n else (1 if "second" in n else 2)

    colors = plt.get_cmap("tab10").colors
    dpi = 100
    fig, axes = plt.subplots(
        len(row_names), len(paths.LEGS),
        figsize=(width / dpi, height / dpi), dpi=dpi,
        sharex=True, sharey="row",
    )
    for j, name in enumerate(names):
        leg, row = _leg_of(name), row_of(name)
        ax = axes[row][leg]
        ax.plot(times, joints[:, j], color=colors[leg], linewidth=0.6,
                label="state")
        ax.plot(times, targets[:, j], color="black", linewidth=0.5,
                linestyle="--", label="target")
        ax.tick_params(labelsize=5)
    for leg, name in enumerate(paths.LEGS):
        axes[0][leg].set_title(
            "".join(w[0] for w in name.split("_")).upper(), fontsize=7
        )
    for row, r in enumerate(row_names):
        axes[row][0].set_ylabel(f"{r} [rad]", fontsize=6)
    axes[0][0].set_xlim(times[0], times[-1])
    axes[0][0].legend(loc="upper right", fontsize=5, framealpha=0.5)
    fig.tight_layout(pad=0.3)
    return _cursor_strip(fig, [ax for r in axes for ax in r], times)


def _label_bar(command, width, height=36):
    """Current command as a text bar above the render (cached per command)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vx, vy, wz = float(command[0]), float(command[1]), float(command[2])
    text = ("stand" if max(abs(vx), abs(vy), abs(wz)) < 0.05
            else f"vx {vx:+.2f}  vy {vy:+.2f}  wz {wz:+.2f}")
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("black")
    fig.text(0.02, 0.5, text, color="white", fontsize=10,
             va="center", family="monospace")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def _latest_checkpoint(ckpt_dir: Path) -> Path:
    steps = [p for p in ckpt_dir.iterdir() if p.name.isdigit()]
    return max(steps, key=lambda p: int(p.name))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--x-vel", type=float, default=0.3)
    ap.add_argument("--y-vel", type=float, default=0.0)
    ap.add_argument("--yaw-vel", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=0.125)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument(
        "--scenario",
        choices=sorted(battery_scenarios().keys()),
        default=None,
        help="run one of the battery's fixed command scripts instead of "
        "--x-vel/--y-vel/--yaw-vel/--height/--steps (all ignored when set)",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--no-plots",
        action="store_true",
        help="plain video without the command label, torque strip, and "
        "per-joint target-vs-state grid",
    )
    ap.add_argument(
        "--joint",
        default=None,
        help="plot target-vs-state for just this actuator as one full-width "
        "readable panel instead of the 3x4 grid, "
        "e.g. front_left_first_joint",
    )
    ap.add_argument(
        "--push",
        action="store_true",
        help="keep the training env's random pushes; default is push-free, "
        "matching the battery's measurement convention",
    )
    args = ap.parse_args()

    # Imported lazily: wojtek_rl.env and wojtek_rl.train pull in brax/mjx and may
    # not exist yet while other tasks are still landing, and this keeps
    # --help / arg parsing cheap and dependency-light.
    from wojtek_rl.policy_io import load_policy
    from wojtek_rl.registry import make_env
    from wojtek_rl.train import _apply_ppo_overrides, build_ppo_params

    run = json.loads((Path(args.run) / "run.json").read_text())
    task = run.get("task", "joystick")
    ppo_params = build_ppo_params([], smoke=False)
    # Replay the run's stored PPO config (network sizes etc.) so the rebuilt
    # network matches the checkpoint even for non-default `network` groups.
    stored = run.get("ppo_config")
    if isinstance(stored, dict) and isinstance(stored.get("network_factory"), dict):
        _apply_ppo_overrides(ppo_params.network_factory, stored["network_factory"])

    out_name = args.out or (f"{args.scenario}.mp4" if args.scenario else "walk.mp4")
    out = Path(out_name)
    if out.parent == Path("."):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = paths.PROJECT_DIR / "videos" / run["run_name"] / stamp / out.name
        out.parent.mkdir(parents=True, exist_ok=True)
    args.out = str(out)

    # Rebuild the exact training env: run.json stores the full env config,
    # which make_env re-applies over the task defaults. Random pushes are
    # off unless --push: a mid-video kick reads as a policy failure
    # (battery.py disables them for the same reason).
    env_cfg = dict(run.get("env_config") or {})
    if not args.push:
        env_cfg["push"] = {**env_cfg.get("push", {}), "enable": False}
    env = make_env(task, env_cfg)
    # run.json may carry the training host's absolute checkpoint path
    # (cluster runs); fall back to the run dir itself.
    ckpt_dir = Path(run["checkpoint_dir"])
    if not ckpt_dir.exists():
        ckpt_dir = (Path(args.run) / "checkpoints").resolve()  # orbax needs abs
    ckpt = _latest_checkpoint(ckpt_dir)
    policy = load_policy(ckpt, env, ppo_params)

    if args.scenario:
        cmd_at, n_steps = battery_scenarios()[args.scenario]
    else:
        command = jp.array([args.x_vel, args.y_vel, args.yaw_vel, args.height])
        cmd_at, n_steps = (lambda i: command), args.steps

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    inference = jax.jit(policy)

    rng = jax.random.PRNGKey(0)
    state = reset(rng)
    frames, vels, torques, targets, joints, frame_meta = [], [], [], [], [], []
    qadr = np.asarray(env._qadr)
    render_every = max(1, round(1 / (30 * env.dt)))
    fps = 1.0 / (env.dt * render_every)  # keeps video speed real-time

    import mujoco

    mj_model = env.mj_model
    data = mujoco.MjData(mj_model)
    with mujoco.Renderer(mj_model, height=480, width=640) as renderer:
        for i in range(n_steps):
            if task == "joystick":
                state.info["command"] = cmd_at(i)
            rng, act_rng = jax.random.split(rng)
            action, _ = inference(state.obs, act_rng)
            state = step(state, action)
            vels.append(float(state.data.qvel[0]))
            torques.append(np.asarray(state.data.actuator_force))
            # The joystick env applies ctrl one step late (action_delay);
            # info holds the target the policy just issued, which is the
            # deploy-relevant signal. getup/jump apply targets directly.
            targets.append(np.asarray(
                state.info.get("motor_targets", state.data.ctrl)
            ))
            joints.append(np.asarray(state.data.qpos)[qadr])
            if float(state.done):
                print(f"fell at step {i}")
                break
            if i % render_every == 0:
                data.qpos[:] = np.asarray(state.data.qpos)
                mujoco.mj_forward(mj_model, data)
                renderer.update_scene(data, camera="track")
                frames.append(renderer.render())
                frame_meta.append(
                    (i * env.dt, np.asarray(state.info["command"]))
                )

    if not args.no_plots and frames:
        names = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(mj_model.nu)
        ]
        times = np.arange(len(torques)) * env.dt
        strip_at = _torque_strip(
            times, np.stack(torques), names,
            float(env._config.get("max_torque", 0) or 0), frames[0].shape[1],
        )
        if args.joint:
            if args.joint not in names:
                sys.exit(f"unknown --joint {args.joint!r}; one of {names}")
            j = names.index(args.joint)
            grid_at = _joint_plot(
                times, np.stack(targets)[:, j], np.stack(joints)[:, j],
                args.joint, frames[0].shape[1],
            )
        else:
            grid_at = _joint_grid(
                times, np.stack(targets), np.stack(joints), names,
                frames[0].shape[1],
            )
        labels = {}
        for k, (t, cmd) in enumerate(frame_meta):
            key = tuple(np.round(cmd[:3], 2))
            if key not in labels:
                labels[key] = _label_bar(cmd, frames[0].shape[1])
            frames[k] = np.vstack(
                [labels[key], frames[k], strip_at(t), grid_at(t)]
            )

    import shutil

    import mediapy

    if shutil.which("ffmpeg") is None:
        # No system ffmpeg (common on a bare Mac); fall back to the binary
        # bundled with imageio-ffmpeg, which is already in the venv.
        import imageio_ffmpeg

        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

    mediapy.write_video(args.out, frames, fps=fps)
    if args.scenario:
        print(f"scenario {args.scenario}  mean vx {np.mean(vels):+.2f}")
    else:
        print(f"commanded vx {args.x_vel:+.2f}  achieved vx {np.mean(vels):+.2f}")
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
