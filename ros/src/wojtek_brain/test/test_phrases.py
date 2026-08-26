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


def test_sample_avoids_immediate_repeat():
    for kind in phrases.KINDS:
        last = phrases.sample(kind)
        for _ in range(20):
            nxt = phrases.sample(kind, avoid=last)
            assert nxt != last
            last = nxt


def test_no_onomatopoeia_and_no_interpolation():
    for line in phrases.all_phrases():
        assert not re.search(r"\b(hau|woof|wrrr)\b", line, re.I), line
        assert "{" not in line and "}" not in line, line
