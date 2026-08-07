"""Minimal deterministic PNG encoder (stdlib only).

Exists so sim_rig can hand MuJoCo texture files without pulling in PIL:
the sim container's Python has zlib and nothing image-shaped, and the
textures are nearest-neighbor tag bitmaps where an encoder is 30 lines.
Output is byte-deterministic (fixed zlib level, no ancillary chunks).
"""

import struct
import zlib


def write_rgb(rows) -> bytes:
    """rows: sequence of rows, each a sequence of (r, g, b) uint8 tuples."""
    height = len(rows)
    width = len(rows[0])
    raw = b"".join(
        b"\x00" + bytes(c for px in row for c in px) for row in rows
    )

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
