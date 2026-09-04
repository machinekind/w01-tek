#!/usr/bin/env python3
"""Draw the two picture textures the simulation's props wear.

Every other prop in scene_sim.xml is a plain coloured shape, because a plain
coloured shape is all the detector needs to name a fire hydrant or a traffic
light. Two of them are different: a stop sign is a stop sign because it says
STOP, and a clock is a clock because it has numbers and hands. Those two need
a picture, and this is where the pictures come from.

The results are committed next to this file, so nobody has to run it to use
the simulation. Run it when you want to change how a sign looks:

    python3 ros/src/wojtek_pc/config/props/make_textures.py

Pillow is the only requirement, and any font will do -- the letters are large
and the detector reads shapes, not typefaces.
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
SIZE = 256

# Whatever bold face the machine happens to have; the last resort is the one
# Pillow carries itself, which is why this list may end without a match.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
)


def font(points):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, points)
    return ImageFont.load_default(size=points)


def octagon(centre, radius, start_deg=22.5):
    return [
        (
            centre + radius * math.cos(math.radians(start_deg + i * 45)),
            centre + radius * math.sin(math.radians(start_deg + i * 45)),
        )
        for i in range(8)
    ]


def stop_sign(size=SIZE):
    """A road stop sign, drawn to the edges of a square.

    The prop is a flat square plate, so the corners around the octagon show.
    They are painted near-black, which is what makes the sign work at three
    metres: red corners turned the whole plate into one red square as the
    picture got small, and the detector stopped seeing an octagon.
    """
    img = Image.new("RGB", (size, size), (28, 28, 32))
    d = ImageDraw.Draw(img)
    d.polygon(octagon(size / 2, size / 2), fill=(196, 26, 32))
    ring = octagon(size / 2, size / 2 * 0.86)
    d.line(ring + [ring[0]], fill=(245, 245, 245), width=max(3, size // 40))
    d.text(
        (size / 2, size / 2), "STOP",
        font=font(int(size * 0.30)), fill=(250, 250, 250), anchor="mm",
    )
    img.save(HERE / "stop_sign.png")


def clock(size=SIZE):
    """A round wall clock: rim, hours 1 to 12, and hands at about ten past two.

    Like the stop sign, the square corners left over around the dial are
    painted near-black so what carries to the camera is a round face and not
    a white square.
    """
    img = Image.new("RGB", (size, size), (28, 28, 32))
    d = ImageDraw.Draw(img)
    c = size / 2
    d.ellipse([2, 2, size - 3, size - 3], fill=(235, 233, 226),
              outline=(40, 40, 45), width=max(4, size // 30))
    numerals = font(int(size * 0.11))
    for hour in range(1, 13):
        a = math.radians(hour * 30 - 90)
        d.text(
            (c + 0.78 * c * math.cos(a), c + 0.78 * c * math.sin(a)),
            str(hour), font=numerals, fill=(30, 30, 35), anchor="mm",
        )
    for length, width, degrees in ((0.42, 7, -30), (0.62, 5, 60)):
        a = math.radians(degrees)
        d.line(
            [c, c, c + length * c * math.cos(a), c + length * c * math.sin(a)],
            fill=(20, 20, 25), width=width,
        )
    d.ellipse([c - 6, c - 6, c + 6, c + 6], fill=(20, 20, 25))
    img.save(HERE / "clock.png")


if __name__ == "__main__":
    stop_sign()
    clock()
    print(f"wrote stop_sign.png and clock.png in {HERE}")
