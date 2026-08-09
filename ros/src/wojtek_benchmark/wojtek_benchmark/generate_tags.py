"""Generate the committed print-ready AprilTag PDFs from config/apriltags.yaml.

The PDFs in tags/ are generated artifacts under the same contract as the
generated MJCF models: committed so a phone or print shop can use them
without a dev environment, regenerated (never hand-edited) after editing
the yaml, and diffed against it via --check.

The PDF is written by hand rather than through a library so the output is
byte-deterministic: no timestamps, no library-version metadata, no
compression.  Byte-identical output for identical yaml input is what lets
the T0 gate prove the committed sheets match the config.

Each A4 sheet carries one tag at exact physical size plus everything needed
to trust it standalone: id, role, black-edge size, a 100 mm scale-check bar
(printers silently rescale), and center tick marks on all four sides for
tape-measuring center-to-center distances at course layout time.
"""

import argparse
import sys
from pathlib import Path

import yaml

from . import tag36h11

MM = 72.0 / 25.4  # PDF user-space points per millimeter

PAGE_W_MM = 210.0
PAGE_H_MM = 297.0
MARGIN_MM = 5.0        # printable-area allowance around the tag square
TAG_TOP_MM = 20.0      # gap from the top page edge to the tag
TEXT_X_MM = 20.0
RULER_Y_MM = 20.0
RULER_LEN_MM = 100.0

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "apriltags.yaml"
DEFAULT_OUT_DIR = PACKAGE_ROOT / "tags"


class ConfigError(ValueError):
    pass


def load_config(path: Path):
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("family") != tag36h11.FAMILY:
        raise ConfigError(f"family must be {tag36h11.FAMILY!r}, got {cfg.get('family')!r}")
    tags = cfg.get("tags")
    if not tags:
        raise ConfigError("no tags defined")
    seen_ids, seen_roles = set(), set()
    for t in tags:
        tag_id, role, size_m = t.get("id"), t.get("role"), t.get("size_m")
        if not isinstance(tag_id, int) or not 0 <= tag_id < len(tag36h11.CODES):
            raise ConfigError(f"bad id {tag_id!r}")
        if not isinstance(role, str) or not role:
            raise ConfigError(f"id {tag_id}: bad role {role!r}")
        if not isinstance(size_m, float) or size_m <= 0:
            raise ConfigError(f"id {tag_id}: size_m must be a positive float, got {size_m!r}")
        dist = t.get("distance_from_origin_m")
        if dist is not None and (not isinstance(dist, (int, float)) or dist <= 0):
            raise ConfigError(f"id {tag_id}: bad distance_from_origin_m {dist!r}")
        if tag_id in seen_ids:
            raise ConfigError(f"duplicate id {tag_id}")
        if role in seen_roles:
            raise ConfigError(f"duplicate role {role!r}")
        seen_ids.add(tag_id)
        seen_roles.add(role)
        total_mm = size_m * 1000.0 * tag36h11.TOTAL_WIDTH / tag36h11.WIDTH_AT_BORDER
        if total_mm > PAGE_W_MM - 2 * MARGIN_MM:
            raise ConfigError(
                f"id {tag_id}: size_m {size_m} needs a {total_mm:.0f} mm printed square "
                f"(black edge * {tag36h11.TOTAL_WIDTH}/{tag36h11.WIDTH_AT_BORDER}); "
                f"max on A4 is {(PAGE_W_MM - 2 * MARGIN_MM) * 0.8:.0f} mm black edge"
            )
    return tags


def pdf_filename(tag) -> str:
    size_mm = tag["size_m"] * 1000.0
    return f"{tag36h11.FAMILY}_id{tag['id']:02d}_{tag['role'].replace('_', '-')}_{size_mm:g}mm.pdf"


# ---------- minimal deterministic PDF ----------

def _fmt(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _text(x_mm, y_mm, size_pt, s, gray=0.0):
    esc = s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    return (
        f"BT {_fmt(gray)} g /F1 {_fmt(size_pt)} Tf "
        f"1 0 0 1 {_fmt(x_mm * MM)} {_fmt(y_mm * MM)} Tm ({esc}) Tj ET"
    )


def _rect(x_mm, y_mm, w_mm, h_mm) -> str:
    return f"{_fmt(x_mm * MM)} {_fmt(y_mm * MM)} {_fmt(w_mm * MM)} {_fmt(h_mm * MM)} re"


def _line(x1_mm, y1_mm, x2_mm, y2_mm) -> str:
    return f"{_fmt(x1_mm * MM)} {_fmt(y1_mm * MM)} m {_fmt(x2_mm * MM)} {_fmt(y2_mm * MM)} l"


def _pdf_document(content: bytes) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 "
            + f"{_fmt(PAGE_W_MM * MM)} {_fmt(PAGE_H_MM * MM)}".encode()
            + b"] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


def build_pdf(tag) -> bytes:
    tag_id, role = tag["id"], tag["role"]
    size_mm = tag["size_m"] * 1000.0
    cell = size_mm / tag36h11.WIDTH_AT_BORDER
    total = cell * tag36h11.TOTAL_WIDTH
    x0 = (PAGE_W_MM - total) / 2.0
    y_top = PAGE_H_MM - TAG_TOP_MM
    y0 = y_top - total
    cx, cy = x0 + total / 2.0, y0 + total / 2.0
    grid = tag36h11.render_bitmap(tag_id)

    ops = []

    # Tag: black cells only (paper is the white). Cells share exact edges;
    # print rasterization at 300+ dpi does not open gaps between them.
    ops.append("0 g")
    for row in range(tag36h11.TOTAL_WIDTH):
        for col in range(tag36h11.TOTAL_WIDTH):
            if not grid[row][col]:
                ops.append(_rect(x0 + col * cell, y_top - (row + 1) * cell, cell, cell))
    ops.append("f")

    # Center tick marks outside the printed square (light gray so they cannot
    # read as tag structure): for lining up tape measures with the tag center.
    ops.append("0.6 G 0.4 w")
    for tick in (
        _line(cx, y_top + 2, cx, y_top + 7),
        _line(cx, y0 - 2, cx, y0 - 7),
        _line(x0 - 2, cy, x0 - 7, cy),
        _line(x0 + total + 2, cy, x0 + total + 7, cy),
    ):
        ops.append(tick)
    ops.append("S")

    ops.append(_text(TEXT_X_MM, 60, 14, f"AprilTag {tag36h11.FAMILY}  -  id {tag_id}  -  {role}"))
    ops.append(_text(
        TEXT_X_MM, 52, 10,
        f"Black square edge: {size_mm:g} mm  (size_m: {tag['size_m']:g} in config/apriltags.yaml)",
    ))
    ops.append(_text(
        TEXT_X_MM, 46, 10,
        "Print at 100% scale (no fit-to-page). The bar below must measure exactly 100 mm;",
    ))
    ops.append(_text(
        TEXT_X_MM, 40, 10,
        "if it does not, the print was scaled: reprint, or set size_m to the measured black edge.",
    ))
    ops.append(_text(
        TEXT_X_MM, 34, 10,
        "Generated by wojtek_benchmark scripts/generate_tags.py. Do not hand-edit; regenerate.",
        gray=0.45,
    ))

    # 100 mm scale-check ruler: minor tick each 10 mm, major at 0/50/100.
    ops.append("0 G 0.6 w")
    ops.append(_line(TEXT_X_MM, RULER_Y_MM, TEXT_X_MM + RULER_LEN_MM, RULER_Y_MM))
    for i in range(11):
        x = TEXT_X_MM + 10.0 * i
        ops.append(_line(x, RULER_Y_MM, x, RULER_Y_MM + (4.0 if i % 5 == 0 else 2.5)))
    ops.append("S")
    ops.append(_text(TEXT_X_MM + RULER_LEN_MM + 3, RULER_Y_MM - 1, 10, "100 mm"))

    return _pdf_document("\n".join(ops).encode("ascii"))


# ---------- CLI ----------

def render_all(config_path: Path):
    return {pdf_filename(t): build_pdf(t) for t in load_config(config_path)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--check", action="store_true",
        help="verify out-dir matches the config exactly (no write); exit 1 on drift",
    )
    args = ap.parse_args(argv)

    try:
        expected = render_all(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 1

    if args.check:
        drift = []
        on_disk = {p.name for p in args.out_dir.glob("*.pdf")}
        for name in sorted(on_disk - expected.keys()):
            drift.append(f"stale (not in yaml): {name}")
        for name, blob in sorted(expected.items()):
            path = args.out_dir / name
            if not path.exists():
                drift.append(f"missing: {name}")
            elif path.read_bytes() != blob:
                drift.append(f"content differs from yaml: {name}")
        for d in drift:
            print(d, file=sys.stderr)
        if drift:
            print("regenerate with scripts/generate_tags.py", file=sys.stderr)
        return 1 if drift else 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in sorted(expected.items()):
        (args.out_dir / name).write_bytes(blob)
        print(f"wrote {args.out_dir / name}")
    for stale in sorted(p for p in args.out_dir.glob("*.pdf") if p.name not in expected):
        stale.unlink()
        print(f"removed stale {stale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
