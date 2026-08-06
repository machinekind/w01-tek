"""Canonical tag36h11 bitmap rendering.

Reimplements apriltag_to_image() from AprilRobotics/apriltag for the
tag36h11 family, so the printed tags are pixel-identical to the official
apriltag-imgs PNGs -- including orientation.  Orientation matters: the
detector reports pose relative to the canonical tag frame, so a rotated
rendering would silently rotate every ground-truth yaw.  The unit tests
compare this rendering against fixtures extracted from the official
images; do not "simplify" the bit layout.
"""

from .tag36h11_data import BIT_POSITIONS, CODES, NBITS, TOTAL_WIDTH, WIDTH_AT_BORDER

FAMILY = "tag36h11"

_BORDER_START = (TOTAL_WIDTH - WIDTH_AT_BORDER) // 2


def render_bitmap(tag_id: int):
    """Return the TOTAL_WIDTH x TOTAL_WIDTH cell grid for one tag.

    grid[row][col] is True where the cell is white; row 0 is the top of the
    canonical tag.  The grid includes the one-cell white quiet zone, so the
    black border square spans WIDTH_AT_BORDER cells of it.
    """
    if not 0 <= tag_id < len(CODES):
        raise ValueError(f"{FAMILY} has ids 0..{len(CODES) - 1}, got {tag_id}")
    code = CODES[tag_id]
    grid = [[False] * TOTAL_WIDTH for _ in range(TOTAL_WIDTH)]
    for i in range(TOTAL_WIDTH):
        grid[0][i] = grid[TOTAL_WIDTH - 1][i] = True
        grid[i][0] = grid[i][TOTAL_WIDTH - 1] = True
    for i, (bx, by) in enumerate(BIT_POSITIONS):
        if code & (1 << (NBITS - 1 - i)):
            grid[by + _BORDER_START][bx + _BORDER_START] = True
    return grid
