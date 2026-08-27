"""The canned-phrase bank: prerecorded Polish for the frequent moments.

Most of what the dog says out loud is one of a dozen situations -- "on my
way", "still searching", "found it". Each gets a handful of fixed variants
(prompts/bielik/phrases/<kind>.txt, one per line) sampled at random, so the
robot answers instantly and never twice in a row with the same words.

Why fixed text matters: the TTS server caches synthesized lines by EXACT
text, and the stack pre-synthesizes every line here at startup -- a canned
phrase then costs a cache hit (~0.1 s measured) instead of a synthesis
(2-4 s). That is the whole latency trick: no new audio path, just stable
strings.

House rules for the bank: no noun interpolation (Polish case endings made
"Znalazłem: telewizora" on camera) and no onomatopoeia (hau/woof -- the TTS
renders them as noises that sound like a fault, user call 2026-08-26).
"""

from __future__ import annotations

import random
from pathlib import Path

_DIR = Path(__file__).parent / "prompts" / "bielik" / "phrases"

KINDS = ("intro", "nav_ack", "cancel_ack", "search_ack", "search_progress",
         "nav_progress", "search_found", "search_not_found", "nav_done",
         "nav_stuck", "goal_error", "switch", "trick_ack")

_cache: dict[str, list[str]] = {}
_bags: dict[str, list[str]] = {}


def variants(kind: str) -> list[str]:
    lines = _cache.get(kind)
    if lines is None:
        path = _DIR / f"{kind}.txt"
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        _cache[kind] = lines
    return lines


def sample(kind: str, avoid: str | None = None) -> str:
    """Shuffle-bag sampling: every variant plays once before any repeats.

    Plain random.choice said "moj popisowy numer" two times out of four on
    camera; a bag guarantees maximum spacing between reruns. `avoid` guards
    the reshuffle boundary (the new bag must not open with the line the old
    one closed on).
    """
    bag = _bags.get(kind)
    if not bag:
        bag = variants(kind).copy()
        random.shuffle(bag)
        if avoid and len(bag) > 1 and bag[-1] == avoid:
            bag[0], bag[-1] = bag[-1], bag[0]
        _bags[kind] = bag
    return bag.pop()


def all_phrases() -> list[str]:
    """Every line of every kind -- the TTS prewarm list."""
    out: list[str] = []
    for kind in KINDS:
        out.extend(variants(kind))
    return out


def main() -> None:
    """`python3 -m wojtek_brain.phrases` prints the prewarm list."""
    for line in all_phrases():
        print(line)


if __name__ == "__main__":
    main()
