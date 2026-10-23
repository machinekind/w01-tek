"""Two guards the experiment contract requires. Model-free, no git needed.

1. Nothing in this experiment carries a secret-shaped value or the identity of
   private infrastructure (the repository is public).
2. Nothing outside the experiment references it (delete-in-one-rm rule);
   checked only when the surrounding repository is visible (host runs).
"""

import re
from pathlib import Path

import pytest

EXPERIMENT = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT.parent.parent
EXPERIMENT_NAME = EXPERIMENT.name

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),                 # OpenAI-style keys
    re.compile(r"AKIA[0-9A-Z]{16}"),                       # AWS access key ids
    re.compile(r"hf_[A-Za-z0-9]{20,}"),                    # Hugging Face tokens
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[=:]\s*['\"][^'\"\s]{8,}"),
    re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b"),              # bare IPv4 addresses
    re.compile(r"\b[\w.-]+@[\w-]+\.[\w.-]+\b"),            # e-mail addresses
]
ALLOWED_IPS = {"127.0.0.1", "0.0.0.0"}
TEXT_SUFFIXES = {".py", ".sh", ".toml", ".yaml", ".yml", ".json", ".md", ".repos", ".ini", ""}
SKIP_DIRS = {"__pycache__", ".pytest_cache", "runs"}


def _experiment_files():
    out = []
    for path in EXPERIMENT.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if SKIP_DIRS & set(path.relative_to(EXPERIMENT).parts):
            continue
        out.append(path)
    return sorted(out)


@pytest.mark.parametrize("path", _experiment_files(), ids=lambda p: str(p.relative_to(EXPERIMENT)))
def test_no_secret_shaped_values(path):
    if path.resolve() == Path(__file__).resolve():
        return
    text = path.read_text(errors="replace")
    for pat in SECRET_PATTERNS:
        for m in pat.finditer(text):
            if m.group(0) in ALLOWED_IPS:
                continue
            raise AssertionError(f"{path.relative_to(EXPERIMENT)}: secret-shaped value {m.group(0)!r}")


def test_nothing_outside_the_experiment_references_it():
    scan_roots = [REPO_ROOT / "ros", REPO_ROOT / "training", REPO_ROOT / "docs"]
    if not all(r.is_dir() for r in scan_roots):
        pytest.skip("repository not visible from here (container run)")
    candidates = list(REPO_ROOT.glob("*.sh")) + list(REPO_ROOT.glob("*.md"))
    for root in scan_roots:
        candidates += [p for p in root.rglob("*") if p.is_file() and p.suffix in TEXT_SUFFIXES]
    hits = []
    for path in candidates:
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            if EXPERIMENT_NAME in path.read_text(errors="replace"):
                hits.append(str(path.relative_to(REPO_ROOT)))
        except OSError:
            continue
    assert hits == [], f"outside references to {EXPERIMENT_NAME}: {hits}"
