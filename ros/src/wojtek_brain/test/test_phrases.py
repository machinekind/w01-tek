"""The canned-phrase bank: every kind present, sampled without immediate
repeats, and clean of the two on-camera failure modes (glued noun cases and
onomatopoeia the TTS renders as noises)."""

import re

from wojtek_brain import phrases


def test_every_kind_has_variants():
    for kind in phrases.KINDS:
        lines = phrases.variants(kind)
        assert len(lines) >= 2, f"{kind} needs variants to sample from"
        assert all(ln.strip() == ln and ln for ln in lines)


def test_shuffle_bag_plays_everything_before_any_repeat():
    for kind in phrases.KINDS:
        phrases._bags.pop(kind, None)
        n = len(phrases.variants(kind))
        drawn = [phrases.sample(kind) for _ in range(n)]
        assert sorted(drawn) == sorted(phrases.variants(kind))


def test_shuffle_bag_never_repeats_across_the_reshuffle():
    for kind in phrases.KINDS:
        phrases._bags.pop(kind, None)
        last = None
        for _ in range(len(phrases.variants(kind)) * 5):
            nxt = phrases.sample(kind, avoid=last)
            if len(phrases.variants(kind)) > 1:
                assert nxt != last
            last = nxt


def test_no_onomatopoeia_and_no_interpolation():
    for line in phrases.all_phrases():
        assert not re.search(r"\b(hau|woof|wrrr)\b", line, re.I), line
        assert "{" not in line and "}" not in line, line


def test_every_node_module_that_uses_phrases_imports_it():
    """bielik crashed live on its first nav ack: `phrases.sample` was added
    by a patch whose import anchor silently missed. Static check: any module
    referencing `phrases.` must import it (nodes import rclpy, so they
    cannot be imported here -- parse instead)."""
    import ast
    from pathlib import Path

    pkg = Path(__file__).parents[1] / "wojtek_brain"
    for py in pkg.glob("*.py"):
        src = py.read_text()
        if "phrases." not in src or py.name == "phrases.py":
            continue
        tree = ast.parse(src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported |= {a.asname or a.name for a in node.names}
            elif isinstance(node, ast.Import):
                imported |= {(a.asname or a.name).split(".")[0] for a in node.names}
        assert "phrases" in imported, f"{py.name} uses phrases without importing it"
