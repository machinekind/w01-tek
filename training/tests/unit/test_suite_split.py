"""Guard the unit/integration split.

`./run.sh test` runs tests/unit only, and its whole point is that it finishes
in seconds so it is usable in an edit loop. That holds only while nothing in
tests/unit builds or steps an MJX model: instantiating an env, putting a model
on device, or parsing the scene XML costs seconds to tens of seconds each.
A new test that needs any of those belongs in tests/integration.
"""

import re
from pathlib import Path

# Written in pieces so this guard does not match its own source.
FORBIDDEN = {
    "env instantiation": r"Wojtek[A-Za-z]*\(\)",
    "registry env": r"make" + r"_env\(",
    "model upload": r"mjx\." + r"put_model",
    "scene parse": r"MjModel\.from" + r"_xml_[a-z]+",
    "model build": r"build" + r"_model\.build",
    "checkpoint load": r"load" + r"_checkpoint_policy",
}

UNIT_DIR = Path(__file__).parent


def test_no_unit_test_builds_or_steps_a_model():
    offenders = {}
    for path in sorted(UNIT_DIR.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        source = path.read_text()
        hits = [label for label, pat in FORBIDDEN.items() if re.search(pat, source)]
        if hits:
            offenders[path.name] = hits
    assert not offenders, (
        f"these tests/unit files touch a real model: {offenders}. Move them to "
        "tests/integration -- see this module's docstring."
    )


def test_the_split_directories_both_exist_and_are_populated():
    integration = UNIT_DIR.parent / "integration"
    assert integration.is_dir()
    assert len(list(UNIT_DIR.glob("test_*.py"))) > 5
    assert len(list(integration.glob("test_*.py"))) > 5
