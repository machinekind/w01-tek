"""In-frame overlays: header panel, torque strip, depth inset, scan heatmap.

Every function draws the values it is handed onto a PIL image of any size, so
nothing here needs a model, an env or a renderer. Panel geometry is scaled off
a reference frame size, so a smaller video gets proportionally smaller panels.
"""

import functools
from pathlib import Path

import numpy as np

from wojtek_rl import paths

DEPTH_WINDOW = (0.3, 3.0)  # colormap window, m
REF_W, REF_H = 960, 720  # the size the panel constants below were drawn at
MARGIN = 12
INSET_W = 320
STRIP_H = 152
HEAT_CELL = 26
FONT_PX = 14
DOT_R = 3.5

PANEL_RGBA = (18, 18, 22, 170)
TEXT_RGB = (236, 236, 240)
EDGE_RGB = (210, 210, 216)
LIMIT_RGB = (235, 70, 60)
VISIBLE_RGB = (70, 220, 110)
MASKED_RGB = (235, 90, 80)
OUT_OF_BAND_RGB = (40, 40, 46)


@functools.lru_cache(maxsize=4)
def _font(px=FONT_PX):
    """Monospace face shipped with matplotlib, so no new font dependency."""
    import matplotlib
    from PIL import ImageFont

    ttf = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(str(ttf), px)
    except OSError:
        return ImageFont.load_default()


@functools.lru_cache(maxsize=1)
def _leg_rgb():
    import matplotlib

    colors = matplotlib.colormaps["tab10"].colors
    return [tuple(round(255 * c) for c in colors[i]) for i in range(len(paths.LEGS))]


def scale_of(size):
    """Panel scale for a frame of `size`, 1.0 at the reference size."""
    w, h = size
    return min(w / REF_W, h / REF_H)


def inset_width(size):
    """Depth-inset width for a frame of `size`; the depth render must match."""
    return max(80, round(INSET_W * scale_of(size)))


def font_px(size):
    """Text size for a frame of `size`; a small frame gets a smaller face, so
    the labels stay inside the panels they belong to."""
    return max(9, round(FONT_PX * scale_of(size)))


def depth_rgb(depth, window=DEPTH_WINDOW):
    """Depth image as RGB over a fixed window; out-of-band pixels stay flat."""
    import matplotlib

    lo, hi = window
    t = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    rgb = (matplotlib.colormaps["turbo"](t)[..., :3] * 255).astype(np.uint8)
    rgb[(depth < lo) | (depth > hi)] = OUT_OF_BAND_RGB
    return rgb


def draw_torques(im, torque, cap):
    """Signed bar per actuator, four groups of three, envelope at +-cap."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(im, "RGBA")
    px = font_px(im.size)
    font = _font(px)
    colors = _leg_rgb()
    w, h = im.size
    strip_h = max(round(STRIP_H * scale_of(im.size)), 2 * px + 40)
    x0, x1 = MARGIN, w - MARGIN
    y1 = h - MARGIN
    y0 = y1 - strip_h
    draw.rectangle([x0, y0, x1, y1], fill=PANEL_RGBA)
    pad, group_gap, bar_gap = 14, 40, 6
    top, bot = y0 + 2 * px + 14, y1 - 8
    mid = (top + bot) / 2
    half = (bot - top) / 2
    scale = 0.85 * half / cap
    torque = np.asarray(torque)
    n_leg, n_joint = len(paths.LEGS), len(torque) // len(paths.LEGS)
    group_w = ((x1 - x0) - 2 * pad - group_gap * (n_leg - 1)) / n_leg
    bar_w = (group_w - bar_gap * (n_joint - 1)) / n_joint
    for i, value in enumerate(torque):
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
            (cx, y0 + px + 8), tag, fill=colors[leg], font=font, anchor="ma"
        )


def draw_depth_inset(im, depth, scan_uv=(), visible=()):
    """Depth inset in the top-right corner with the scan points on it.

    Returns its (x, y, w, h) rectangle.
    """
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(im, "RGBA")
    h, w = depth.shape[:2]
    x0, y0 = im.size[0] - MARGIN - w, MARGIN
    r = max(2.0, DOT_R * scale_of(im.size))
    tile = Image.fromarray(depth)
    tile_draw = ImageDraw.Draw(tile)
    for (u, v), vis in zip(scan_uv, visible):
        color = VISIBLE_RGB if vis else MASKED_RGB
        tile_draw.ellipse([u - r, v - r, u + r, v + r], fill=color, outline=(15, 15, 18))
    im.paste(tile, (x0, y0))
    draw.rectangle([x0 - 1, y0 - 1, x0 + w, y0 + h], outline=EDGE_RGB)
    lo, hi = DEPTH_WINDOW
    px = font_px(im.size)
    font = _font(px)
    draw.rectangle([x0, y0, x0 + w, y0 + px + 6], fill=PANEL_RGBA)
    x = x0 + 5
    for text, color in (
        (f"depth {lo:.1f}-{hi:.1f} m ", TEXT_RGB),
        ("●seen ", VISIBLE_RGB),
        ("●masked", MASKED_RGB),
    ):
        draw.text((x, y0 + 3), text, fill=color, font=font)
        x += font.getlength(text)
    return x0, y0, w, h


def draw_scan_hold(im, grid, clip, top):
    """The actor's held scan as a heatmap under `top`, right-aligned.

    `grid` is already oriented for display: far row first, +y on the left.
    """
    import matplotlib
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(im, "RGBA")
    grid = np.asarray(grid)
    cell = max(10, round(HEAT_CELL * scale_of(im.size)))
    t = np.clip(grid / (2 * clip) + 0.5, 0.0, 1.0)
    rgb = (matplotlib.colormaps["coolwarm"](t)[..., :3] * 255).astype(np.uint8)
    size = (grid.shape[1] * cell, grid.shape[0] * cell)
    tile = Image.fromarray(rgb).resize(size, Image.NEAREST)
    x0 = im.size[0] - MARGIN - size[0]
    im.paste(tile, (x0, top))
    draw.rectangle([x0 - 1, top - 1, x0 + size[0], top + size[1]], outline=EDGE_RGB)
    px = font_px(im.size)
    draw.text(
        (x0, top - px - 4), f"scan_hold ±{clip:.2f} m",
        fill=TEXT_RGB, font=_font(px),
    )


def draw_header(im, lines):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(im, "RGBA")
    px = font_px(im.size)
    font = _font(px)
    step = px + 5
    w = max(font.getlength(line) for line in lines) + 16
    draw.rectangle(
        [MARGIN, MARGIN, MARGIN + w, MARGIN + step * len(lines) + 8],
        fill=PANEL_RGBA,
    )
    for k, line in enumerate(lines):
        draw.text(
            (MARGIN + 8, MARGIN + 5 + k * step), line, fill=TEXT_RGB, font=font
        )


def compose(
    frame, *, depth=None, scan_uv=(), visible=(), torque=None, cap=None,
    scan_hold=None, clip=None, header=None,
):
    """One rendered frame with every overlay it was given drawn onto it."""
    from PIL import Image

    im = Image.fromarray(frame)
    inset = None
    if depth is not None:
        inset = draw_depth_inset(im, depth, scan_uv, visible)
    if scan_hold is not None:
        px = font_px(im.size)
        # Below the inset when there is one, else where the inset would start.
        top = MARGIN + px + 8 if inset is None else inset[1] + inset[3] + px + 12
        draw_scan_hold(im, scan_hold, clip, top)
    if torque is not None:
        draw_torques(im, torque, cap)
    if header:
        draw_header(im, header)
    return np.asarray(im)
