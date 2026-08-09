"""Guards for the printable-tag pipeline.

Two contracts are enforced here:

1. Rendering is canonical: bitmaps must match fixtures extracted from the
   official AprilRobotics/apriltag-imgs PNGs (test/fixtures/*.txt, one line
   per row, 1 = white).  A rotated or mirrored rendering would still detect,
   but every ground-truth pose would be silently wrong.
2. The committed tags/*.pdf match config/apriltags.yaml byte-for-byte, so
   the sheets people print are provably the config that calibration and
   detection will read.
"""

from pathlib import Path

import pytest

import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from wojtek_benchmark import generate_tags, tag36h11  # noqa: E402

FIXTURES = sorted((PACKAGE_ROOT / "test" / "fixtures").glob("tag36_11_*.txt"))


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_bitmap_matches_official_apriltag_imgs(fixture):
    tag_id = int(fixture.stem.split("_")[-1])
    want = [[c == "1" for c in line] for line in fixture.read_text().split()]
    assert tag36h11.render_bitmap(tag_id) == want


def test_fixtures_cover_every_configured_tag():
    configured = {t["id"] for t in generate_tags.load_config(generate_tags.DEFAULT_CONFIG)}
    covered = {int(p.stem.split("_")[-1]) for p in FIXTURES}
    missing = configured - covered
    assert not missing, (
        f"add official-image fixtures for ids {sorted(missing)}: extract from "
        "https://github.com/AprilRobotics/apriltag-imgs tag36h11/"
    )


def test_committed_pdfs_match_config(tmp_path):
    expected = generate_tags.render_all(generate_tags.DEFAULT_CONFIG)
    committed = {p.name: p.read_bytes() for p in generate_tags.DEFAULT_OUT_DIR.glob("*.pdf")}
    assert sorted(committed) == sorted(expected), (
        "tags/ out of sync with apriltags.yaml -- regenerate with scripts/generate_tags.py"
    )
    for name, blob in expected.items():
        assert committed[name] == blob, (
            f"{name} differs from what apriltags.yaml generates -- "
            "regenerate with scripts/generate_tags.py"
        )


def test_generation_is_deterministic():
    a = generate_tags.render_all(generate_tags.DEFAULT_CONFIG)
    b = generate_tags.render_all(generate_tags.DEFAULT_CONFIG)
    assert a == b
