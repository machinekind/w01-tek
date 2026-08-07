"""Unit tests for the mujoco-free parts of the sim rig.

Injection itself (MjSpec, textures, rendering) is exercised by
scripts/sim_rig_check.py, which needs mujoco + a GL backend; these tests
guard the pure config/geometry/encoding pieces on any machine.
"""

import struct
import sys
import zlib
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from wojtek_benchmark import png, sim_rig, tag36h11  # noqa: E402


def test_rig_config_loads_and_legs_derive():
    cfg = sim_rig.load_rig_config()
    leg_x, leg_y = sim_rig.leg_lengths(cfg)
    assert leg_x > 0 and leg_y > 0


def test_world_frame_is_orthonormal_and_at_origin_tag():
    cfg = sim_rig.load_rig_config()
    T = sim_rig.world_frame_in_sim(cfg)
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert R[2, 2] > 0.99, "bench world z must point up"
    o = cfg["floor_tags"]["world_origin"]["center_xy"]
    assert np.allclose(T[:3, 3], [o[0], o[1], 0.0])


def test_tags_config_rejects_wrong_family(tmp_path):
    bad = tmp_path / "tags.yaml"
    bad.write_text("family: tag25h9\ntags: []\n")
    try:
        sim_rig.load_tags_config(bad)
    except ValueError:
        return
    raise AssertionError("wrong family must raise")


def test_texture_rows_scale_and_binarize():
    rows = sim_rig.tag_texture_rows(0, px_per_cell=4)
    n = tag36h11.TOTAL_WIDTH * 4
    assert len(rows) == n and len(rows[0]) == n
    values = {px for row in rows for px in row}
    assert values <= {(0, 0, 0), (255, 255, 255)}
    # Quiet-zone ring is white in the texture too.
    assert rows[0][0] == (255, 255, 255)


def test_png_roundtrip():
    rows = [[(255, 0, 0), (0, 255, 0)], [(0, 0, 255), (255, 255, 255)]]
    blob = png.write_rgb(rows)
    assert blob.startswith(b"\x89PNG\r\n\x1a\n")
    w, h = struct.unpack(">II", blob[16:24])
    assert (w, h) == (2, 2)
    # Decode the IDAT payload and check the raw filtered scanlines.
    idat_len = struct.unpack(">I", blob[33:37])[0]
    raw = zlib.decompress(blob[41:41 + idat_len])
    assert raw == b"\x00\xff\x00\x00\x00\xff\x00" + b"\x00\x00\x00\xff\xff\xff\xff"
    assert png.write_rgb(rows) == blob, "encoder must be deterministic"
