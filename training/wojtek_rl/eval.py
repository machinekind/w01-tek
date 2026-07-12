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
    # which make_env re-applies over the task defaults.
    env = make_env(task, run.get("env_config"))
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
    frames, vels = [], []
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
            if float(state.done):
                print(f"fell at step {i}")
                break
            if i % render_every == 0:
                data.qpos[:] = np.asarray(state.data.qpos)
                mujoco.mj_forward(mj_model, data)
                renderer.update_scene(data, camera="track")
                frames.append(renderer.render())

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
