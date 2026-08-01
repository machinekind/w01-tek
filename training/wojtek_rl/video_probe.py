"""Render a checkpoint walking, with torque bars and the onboard depth view.

Run: ./run.sh video-probe --run runs/<name> --arena train --seconds 8
     ./run.sh video-probe --run runs/<name> --cell pyramid_stairs_5cm --vx 0.4

`--cell` films one cell of the measurement suite: the eval arena, the spawn
terrain-scan gives that cell's first course run, and the commanded speed held
at `--vx` so the robot walks out into the obstacle. Falling is a result, not a
failure of the tool -- the clip ends where the episode does.

Three overlays on one chase-camera frame:

* a bar per actuator against the env's effective torque limit, grouped per leg;
* the depth image from a camera placed at `height_scan.mask`'s mount, pitch
  and vertical field of view, so the render and the analytic mask share a
  frustum;
* the 25 scan points projected into that depth image, colored by
  `height_scan.visible_mask`. A run whose actor consumes `height_scan` also
  gets the held scan as a 5x5 heatmap.

A tool, not a measurement: nothing here reads or writes training config.
"""

import os
import sys

if sys.platform == "linux":  # headless GPU boxes; macOS uses its default GL
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import functools
import math
from pathlib import Path

import numpy as np

from wojtek_rl import height_scan, paths, terrain_scan, terrain_suite

HEIGHT_CMD = 0.125
DEPTH_WINDOW = (0.3, 3.0)  # colormap window, m
FRAME_W, FRAME_H = 960, 720
MARGIN = 12
INSET_W = 320
STRIP_H = 152
HEAT_CELL = 26
FONT_PX = 14
SCAN_CAM = "scan_probe"
ROOT_BODY = "root"

PANEL_RGBA = (18, 18, 22, 170)
TEXT_RGB = (236, 236, 240)
EDGE_RGB = (210, 210, 216)
LIMIT_RGB = (235, 70, 60)
VISIBLE_RGB = (70, 220, 110)
MASKED_RGB = (235, 90, 80)
OUT_OF_BAND_RGB = (40, 40, 46)


@functools.lru_cache(maxsize=1)
def _font():
    """Monospace face shipped with matplotlib, so no new font dependency."""
    import matplotlib
    from PIL import ImageFont

    ttf = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(str(ttf), FONT_PX)
    except OSError:
        return ImageFont.load_default()


@functools.lru_cache(maxsize=1)
def _leg_rgb():
    import matplotlib

    colors = matplotlib.colormaps["tab10"].colors
    return [tuple(round(255 * c) for c in colors[i]) for i in range(len(paths.LEGS))]


def render_model(env, mask):
    """(model, camera name) for rendering the env's scene.

    Adds a camera at the mask's mount and pitch unless the scene already
    carries one there, and widens the offscreen buffer to the frame size.
    """
    import mujoco

    from wojtek_rl.build_model import _xyaxes_to_quat

    pitch = math.radians(float(mask.pitch_deg))
    mount = [float(v) for v in mask.mount]
    # MuJoCo cameras look down -z with +y up: x right = -y body, up pitched
    # by the mask's angle, which is the frame visible_mask assumes.
    quat = _xyaxes_to_quat(
        [0.0, -1.0, 0.0, math.sin(pitch), 0.0, math.cos(pitch)]
    )
    fovy = float(mask.vfov_deg)
    m = env.mj_model
    root = m.body(ROOT_BODY).id
    camera = next(
        (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(m.ncam)
            if m.cam_bodyid[i] == root
            and np.allclose(m.cam_pos[i], mount, atol=1e-4)
            and np.allclose(m.cam_quat[i], quat, atol=1e-4)
            and abs(float(m.cam_fovy[i]) - fovy) < 1e-4
        ),
        None,
    )
    spec = mujoco.MjSpec.from_file(env.xml_path)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, FRAME_W)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, FRAME_H)
    if camera is None:
        spec.body(ROOT_BODY).add_camera(
            name=SCAN_CAM, pos=mount, quat=quat.tolist(), fovy=fovy
        )
        camera = SCAN_CAM
    return spec.compile(), camera


def depth_size(mask, width=INSET_W):
    """(w, h) whose aspect gives the render mask.hfov_deg horizontally while
    the camera's fovy carries mask.vfov_deg."""
    ratio = math.tan(math.radians(float(mask.hfov_deg)) / 2) / math.tan(
        math.radians(float(mask.vfov_deg)) / 2
    )
    return width, max(1, round(width / ratio))


def project(points, cam_pos, cam_mat, fovy, size):
    """(u, v, depth) of world `points` in a pinhole camera image."""
    w, h = size
    local = (np.asarray(points) - cam_pos) @ cam_mat
    depth = -local[:, 2]
    f = (h / 2) / math.tan(math.radians(fovy) / 2)
    ahead = np.maximum(depth, 1e-6)
    return w / 2 + f * local[:, 0] / ahead, h / 2 - f * local[:, 1] / ahead, depth


def depth_rgb(depth, window=DEPTH_WINDOW):
    """Depth image as RGB over a fixed window; out-of-band pixels stay flat."""
    import matplotlib

    lo, hi = window
    t = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    rgb = (matplotlib.colormaps["turbo"](t)[..., :3] * 255).astype(np.uint8)
    rgb[(depth < lo) | (depth > hi)] = OUT_OF_BAND_RGB
    return rgb


def _draw_torques(draw, torque, cap):
    """Signed bar per actuator, four groups of three, envelope at +-cap."""
    font = _font()
    colors = _leg_rgb()
    x0, x1 = MARGIN, FRAME_W - MARGIN
    y1 = FRAME_H - MARGIN
    y0 = y1 - STRIP_H
    draw.rectangle([x0, y0, x1, y1], fill=PANEL_RGBA)
    pad, group_gap, bar_gap = 14, 40, 6
    top, bot = y0 + 2 * FONT_PX + 14, y1 - 8
    mid = (top + bot) / 2
    half = (bot - top) / 2
    scale = 0.85 * half / cap
    n_leg, n_joint = len(paths.LEGS), len(torque) // len(paths.LEGS)
    group_w = ((x1 - x0) - 2 * pad - group_gap * (n_leg - 1)) / n_leg
    bar_w = (group_w - bar_gap * (n_joint - 1)) / n_joint
    for i, value in enumerate(np.asarray(torque)):
        leg, joint = divmod(i, n_joint)
        bx = x0 + pad + leg * (group_w + group_gap) + joint * (bar_w + bar_gap)
        tip = mid - float(np.clip(value * scale, -half, half))
        draw.rectangle(
            [bx, min(mid, tip), bx + bar_w, max(mid, tip)], fill=colors[leg]
        )
    draw.line([x0 + pad, mid, x1 - pad, mid], fill=EDGE_RGB, width=1)
    for sign in (-1, 1):
        y = mid - sign * cap * scale
        draw.line([x0 + pad, y, x1 - pad, y], fill=LIMIT_RGB, width=2)
    draw.text(
        (x0 + pad, y0 + 4), f"actuator torque, limit ±{cap:.1f} N·m",
        fill=TEXT_RGB, font=font,
    )
    for leg, name in enumerate(paths.LEGS):
        tag = "".join(w[0] for w in name.split("_")).upper()
        cx = x0 + pad + leg * (group_w + group_gap) + group_w / 2
        draw.text(
            (cx, y0 + FONT_PX + 8), tag, fill=colors[leg], font=font, anchor="ma"
        )


def _draw_inset(im, draw, depth, scan_uv, visible):
    """Depth inset in the top-right corner with the scan points on it.

    Returns its (x, y, w, h) rectangle.
    """
    from PIL import Image, ImageDraw

    h, w = depth.shape[:2]
    x0, y0 = FRAME_W - MARGIN - w, MARGIN
    tile = Image.fromarray(depth)
    tile_draw = ImageDraw.Draw(tile)
    for (u, v), vis in zip(scan_uv, visible):
        color = VISIBLE_RGB if vis else MASKED_RGB
        tile_draw.ellipse(
            [u - 3.5, v - 3.5, u + 3.5, v + 3.5], fill=color, outline=(15, 15, 18)
        )
    im.paste(tile, (x0, y0))
    draw.rectangle([x0 - 1, y0 - 1, x0 + w, y0 + h], outline=EDGE_RGB)
    lo, hi = DEPTH_WINDOW
    font = _font()
    draw.rectangle([x0, y0, x0 + w, y0 + FONT_PX + 6], fill=PANEL_RGBA)
    x = x0 + 5
    for text, color in (
        (f"depth {lo:.1f}-{hi:.1f} m ", TEXT_RGB),
        ("●seen ", VISIBLE_RGB),
        ("●masked", MASKED_RGB),
    ):
        draw.text((x, y0 + 3), text, fill=color, font=font)
        x += font.getlength(text)
    return x0, y0, w, h


def _draw_scan_hold(im, draw, scan, clip, inset):
    """The actor's held 5x5 scan below the depth inset, far row on top."""
    import matplotlib
    from PIL import Image

    grid = np.asarray(scan).reshape(height_scan.NX, height_scan.NY)
    # body_grid indexes ix*ny + iy with x forward and y left; the image wants
    # the far row first and +y on the left.
    grid = grid[::-1, ::-1]
    t = np.clip(grid / (2 * clip) + 0.5, 0.0, 1.0)
    rgb = (matplotlib.colormaps["coolwarm"](t)[..., :3] * 255).astype(np.uint8)
    size = (height_scan.NY * HEAT_CELL, height_scan.NX * HEAT_CELL)
    tile = Image.fromarray(rgb).resize(size, Image.NEAREST)
    x0 = FRAME_W - MARGIN - size[0]
    y0 = inset[1] + inset[3] + FONT_PX + 12
    im.paste(tile, (x0, y0))
    draw.rectangle([x0 - 1, y0 - 1, x0 + size[0], y0 + size[1]], outline=EDGE_RGB)
    draw.text(
        (x0, y0 - FONT_PX - 4), f"scan_hold ±{clip:.2f} m",
        fill=TEXT_RGB, font=_font(),
    )


def _draw_header(draw, lines):
    font = _font()
    step = FONT_PX + 5
    w = max(font.getlength(line) for line in lines) + 16
    draw.rectangle(
        [MARGIN, MARGIN, MARGIN + w, MARGIN + step * len(lines) + 8],
        fill=PANEL_RGBA,
    )
    for k, line in enumerate(lines):
        draw.text(
            (MARGIN + 8, MARGIN + 5 + k * step), line, fill=TEXT_RGB, font=font
        )


def compose(frame, depth, scan_uv, visible, torque, cap, scan_hold, clip, header):
    """One rendered frame with every overlay drawn onto it."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(frame)
    draw = ImageDraw.Draw(im, "RGBA")
    inset = _draw_inset(im, draw, depth, scan_uv, visible)
    if scan_hold is not None:
        _draw_scan_hold(im, draw, scan_hold, clip, inset)
    _draw_torques(draw, torque, cap)
    _draw_header(draw, header)
    return np.asarray(im)


def write_video(out: Path, frames, fps):
    import shutil

    import mediapy

    if shutil.which("ffmpeg") is None:
        # No system ffmpeg (common on a bare Mac); fall back to the binary
        # bundled with imageio-ffmpeg, which is already in the venv.
        import imageio_ffmpeg

        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    out.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(out), frames, fps=fps)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a checkpoint walking, with torque bars and the "
        "onboard depth view."
    )
    ap.add_argument("--run", default=None, help="required unless --list-cells")
    ap.add_argument(
        "--arena", choices=["flat", "train", "eval"], default=None,
        help="flat forces the flat scene; train/eval render on that terrain "
        "arena (terrain runs only). Default train, or eval with --cell",
    )
    ap.add_argument(
        "--cell", default=None,
        help="film one measurement cell of the eval arena, at the spawn "
        "terrain-scan uses for it (implies --arena eval)",
    )
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--vx", type=float, default=0.5, help="forward command, m/s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--camera", default="track",
        help="chase camera for the main view; terrain scenes also define "
        "track_far, higher and farther out",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--list-cells", action="store_true", help="print the cells and exit"
    )
    args = ap.parse_args()

    if args.list_cells:
        for line in terrain_scan.cell_lines():
            print(line)
        return
    if not args.run:
        ap.error("--run is required")
    if args.cell and args.arena:
        ap.error("--cell already selects the eval arena; drop --arena")
    cell = None
    if args.cell:
        cell = terrain_suite.CELLS_BY_NAME.get(args.cell)
        if cell is None:
            ap.error(f"unknown cell {args.cell!r}; --list-cells prints them all")
    arena = args.arena or ("eval" if cell else "train")

    import jax
    import jax.numpy as jp
    import mujoco

    from wojtek_rl.battery import load_checkpoint_policy, torque_cap_of

    flat = arena == "flat"
    overrides = {"sim": {"backend": "jax", "num_envs": 1}}
    if not flat:
        # A pinned spawn pad: this renders one episode, and jitter only makes
        # it irreproducible.
        overrides["terrain"] = {
            "enable": True, "arena": arena, "pad_jitter": 0.0,
        }
    if cell is not None:
        overrides["terrain"]["spawn_yaw"] = False
        # The clip holds one command; the env's own resample would hand the
        # actor a different one halfway up the obstacle.
        overrides["command"] = {"resample_steps": 10**9}
    run, env, ckpt, inf = load_checkpoint_policy(
        Path(args.run), flat=flat, env_overrides=overrides
    )
    if cell is not None:
        terrain_scan.check_arena_of(env)
    hs = env._config.get("height_scan")
    if hs is None:
        sys.exit(
            f"task {run.get('task')!r} has no height_scan config, so there is "
            "no mask geometry to place the depth camera at"
        )
    where = f"cell {cell.name}" if cell else f"arena {arena}"
    out = Path(args.out) if args.out else (
        paths.PROJECT_DIR / "videos" / run["run_name"]
        / f"probe_{cell.name if cell else arena}.mp4"
    )
    print(f"scene: {env.xml_path}")

    model, camera = render_model(env, hs.mask)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    size = depth_size(hs.mask)
    fovy = float(model.cam_fovy[cam_id])
    grid = height_scan.body_grid(hs.x_range, hs.y_range, hs.nx, hs.ny)
    mask = hs.mask
    mount = np.asarray(mask.mount, float)
    cap = torque_cap_of(env)
    # The heatmap shows what the actor is actually fed. _scan_live, not
    # _scan_enabled: the sample-and-hold buffers only exist on terrain.
    show_hold = env._scan_live and "height_scan" in env.actor_obs_names

    command = jp.array([args.vx, 0.0, 0.0, HEIGHT_CMD])
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(args.seed)
    if cell is None:
        state = jax.jit(env.reset)(rng)
    else:
        # Course run 0 of that cell: heading +x, the start offset furthest back
        # from the obstacle band, on the tile the scan measures.
        _, spawn, yaw, pad_h = terrain_scan.spawn_table(env, cell)
        state = jax.jit(functools.partial(terrain_scan.scan_reset, env))(
            rng, spawn[0], pad_h[0], yaw[0], command
        )
    n_steps = max(1, round(args.seconds / env.dt))
    every = max(1, round(1.0 / (args.fps * env.dt)))
    fps = 1.0 / (env.dt * every)

    frames, vels = [], []
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, height=FRAME_H, width=FRAME_W) as view, \
            mujoco.Renderer(model, height=size[1], width=size[0]) as depth_view:
        depth_view.enable_depth_rendering()
        for i in range(n_steps):
            if "command" in state.info:
                state.info["command"] = command
            rng, act_rng = jax.random.split(rng)
            action, _ = inf(state.obs, act_rng)
            state = step(state, action)
            vels.append(float(env._local_linvel(state.data)[0]))
            if float(state.done):
                print(f"fell at step {i}")
                break
            if i % every:
                continue
            qpos = np.asarray(state.data.qpos)
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            view.update_scene(data, camera=args.camera)
            frame = view.render()
            depth_view.update_scene(data, camera=camera)
            depth = depth_view.render()

            xy = height_scan.world_xy(
                grid, qpos[0:2], height_scan.yaw_from_quat(qpos[3:7], xp=np), xp=np
            )
            z = (
                np.asarray(env._terrain.height(xy)) if env._terrain_enabled
                else np.zeros(len(xy))
            )
            points = np.concatenate([xy, z[:, None]], axis=-1)
            visible = np.asarray(
                height_scan.visible_mask(
                    points, qpos[0:3], qpos[3:7], mount, mask.pitch_deg,
                    mask.hfov_deg, mask.vfov_deg, mask.min_depth, mask.max_depth,
                    xp=np,
                )
            )
            u, v, ahead = project(
                points, data.cam_xpos[cam_id],
                data.cam_xmat[cam_id].reshape(3, 3), fovy, size,
            )
            keep = ahead > 0.0
            frames.append(
                compose(
                    frame, depth_rgb(depth),
                    np.stack([u[keep], v[keep]], axis=-1), visible[keep],
                    state.data.actuator_force, cap,
                    np.asarray(state.info["scan_hold"]) if show_hold else None,
                    float(hs.clip),
                    [
                        f"{run['run_name']} @ {ckpt.name}",
                        f"{where}   t {i * env.dt:5.2f} s",
                        f"cmd vx {args.vx:+.2f}   vx {vels[-1]:+.2f} m/s",
                    ],
                )
            )

    if not frames:
        sys.exit("no frames: the rollout ended before the first render")
    write_video(out, frames, fps=fps)
    print(
        f"commanded vx {args.vx:+.2f}  achieved forward vx {np.mean(vels):+.2f}"
    )
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
