"""Hygiene checks for the job scripts in jobs/.

These scripts are plain bash. They declare their inputs with ordinary shell
parameter expansion, so the declaration is the code that applies it and there
is no separate contract file to keep in step.

What is checked here is what this repository cares about: the scripts parse,
they declare their inputs before doing any work, and they carry no scheduler,
host, or transfer commands. How a dispatcher reads a declaration is not this
repository's concern.

Scripts are discovered by glob, so a new one is covered without editing this
file. Names starting with an underscore are sourced helpers, not job scripts.
"""

import re
import subprocess
from pathlib import Path

import pytest

JOBS_DIR = Path(__file__).resolve().parents[2] / "jobs"

# : "${NAME:?message}"  or  : "${NAME:=default}"  or
# export NAME="${NAME:-default}"
REQUIRED_RE = re.compile(r'^\s*:\s*"\$\{([A-Za-z_][A-Za-z0-9_]*):\?', re.M)
OPTIONAL_RE = re.compile(r'^\s*:\s*"\$\{([A-Za-z_][A-Za-z0-9_]*):=', re.M)
EXPORTED_RE = re.compile(
    r'^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)="\$\{\1:-', re.M
)

# Anything that would tie a job script to one machine or one scheduler.
FORBIDDEN_RE = re.compile(
    r"#SBATCH|\bSLURM_|\bsbatch\b|\bsqueue\b|\bsacct\b|\bscancel\b|"
    r"\bsrun\b|\brsync\b|\bscp\b|^\s*ssh\s",
    re.M,
)

# An absolute path outside this repository names somebody's machine. The
# scheduler check above cannot catch it: a hard-coded storage path is plain
# shell. Only /tmp and /dev are machine independent.
ABSOLUTE_PATH_RE = re.compile(
    r"""["'=:\s](/(?!tmp/|tmp["'\s]|dev/)[A-Za-z][A-Za-z0-9._-]*(?:/[A-Za-z0-9._*-]+)+)""",
)

# A host name, a user@host, or a scheduler queue named in passing.
SITE_VALUE_RE = re.compile(
    r"[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"|--partition[= ]|--account[= ]|--qos[= ]",
)


def payloads():
    return sorted(p for p in JOBS_DIR.glob("*.sh") if not p.name.startswith("_"))


def test_payloads_exist():
    assert payloads(), f"no job scripts found in {JOBS_DIR}"


@pytest.mark.parametrize("path", payloads(), ids=lambda p: p.name)
def test_parses(path: Path):
    proc = subprocess.run(
        ["bash", "-n", str(path)], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("path", payloads(), ids=lambda p: p.name)
def test_declares_its_inputs(path: Path):
    text = path.read_text()
    declared = (
        set(REQUIRED_RE.findall(text))
        | set(OPTIONAL_RE.findall(text))
        | set(EXPORTED_RE.findall(text))
    )
    assert declared, f"{path.name} declares no inputs"


@pytest.mark.parametrize("path", payloads(), ids=lambda p: p.name)
def test_declares_before_working(path: Path):
    """Every declaration comes before the first command that does something.

    A declaration after the first real step means a run can get half way in
    and only then discover that a required input is missing.
    """
    lines = path.read_text().splitlines()
    first_work = None
    last_decl = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if (
            REQUIRED_RE.match(line)
            or OPTIONAL_RE.match(line)
            or EXPORTED_RE.match(line)
        ):
            last_decl = i
            continue
        if first_work is None and re.match(
            r"^(run_main|python3|\./|mkdir|curl|[A-Z_]+=\$\()", stripped
        ):
            first_work = i
    if first_work is not None and last_decl is not None:
        assert last_decl < first_work, (
            f"{path.name}: an input is declared at line {last_decl + 1}, "
            f"after work starts at line {first_work + 1}"
        )


@pytest.mark.parametrize("path", payloads(), ids=lambda p: p.name)
def test_no_scheduler_or_transfer_commands(path: Path):
    hits = FORBIDDEN_RE.findall(path.read_text())
    assert not hits, f"{path.name} refers to a scheduler or a remote host: {hits}"


@pytest.mark.parametrize("path", payloads(), ids=lambda p: p.name)
def test_no_site_values(path: Path):
    """No absolute path or host name belonging to one machine.

    A job script runs wherever the repository is checked out. A path like
    /scratch/user/runs works on exactly one machine and names its owner, and
    nothing else in this suite would catch it.
    """
    text = path.read_text()
    paths = [m for m in ABSOLUTE_PATH_RE.findall(text)]
    assert not paths, f"{path.name} hard-codes an absolute path: {paths}"
    site = SITE_VALUE_RE.findall(text)
    assert not site, f"{path.name} names a host or a scheduler queue: {site}"
