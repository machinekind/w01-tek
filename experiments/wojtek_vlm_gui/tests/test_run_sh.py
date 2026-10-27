"""run.sh hands the container the same ROS_LOCALHOST_ONLY that ros/sim.sh
gives wojtek_robot: from this shell, else the repo-root .env, else ros/.env.
And `run.sh gui` makes the page listen where the platform needs it.
Model-free: a copy of run.sh in a temporary tree, with a fake `docker` on
PATH that writes down what it was given. No Docker, no ROS."""

import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

EXPERIMENT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

FAKE_DOCKER = """#!/usr/bin/env bash
echo "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY-<unset>} args=$*" >> "$DOCKER_LOG"
"""


@pytest.fixture
def tree(tmp_path):
    """tmp/experiments/wojtek_vlm_gui/run.sh, the way the repository lays it
    out, so run.sh's ../../.env and ../../ros/.env land inside tmp."""
    exp = tmp_path / "experiments" / "wojtek_vlm_gui"
    (exp / "docker").mkdir(parents=True)
    shutil.copy(EXPERIMENT / "run.sh", exp / "run.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    return tmp_path


def _up(tree, shell_value=None, command=("up",)):
    """`run.sh <command>`, `up` by default; returns (what docker saw for
    ROS_LOCALHOST_ONLY, the args of the last docker call)."""
    log = tree / "docker.log"
    env = dict(os.environ)
    env.pop("ROS_LOCALHOST_ONLY", None)   # the container's own compose env sets it
    if shell_value is not None:
        env["ROS_LOCALHOST_ONLY"] = shell_value
    env["PATH"] = f"{tree / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["DOCKER_LOG"] = str(log)
    run = subprocess.run(["bash", str(tree / "experiments" / "wojtek_vlm_gui" / "run.sh"), *command],
                         env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    line = log.read_text().splitlines()[-1]
    value, _, args = line.partition(" args=")
    return value.split("=", 1)[1], args


def test_root_env_sets_localhost_only(tree):
    (tree / ".env").write_text("VLM_URL=x\nROS_LOCALHOST_ONLY=1\n")
    value, args = _up(tree)
    assert value == "1"
    assert "compose" in args.split() and "up" in args.split()


def test_shell_value_wins_over_env_file(tree):
    (tree / ".env").write_text("ROS_LOCALHOST_ONLY=1\n")
    assert _up(tree, shell_value="0")[0] == "0"


def test_ros_env_is_the_fallback(tree):
    (tree / "ros").mkdir()
    (tree / "ros" / ".env").write_text("ROS_LOCALHOST_ONLY=1\n")
    assert _up(tree)[0] == "1"


def test_quoted_value_is_unquoted(tree):
    (tree / ".env").write_text('ROS_LOCALHOST_ONLY="1"\n')
    assert _up(tree)[0] == "1"


def test_a_windows_line_end_is_cut_off(tree):
    (tree / ".env").write_text("ROS_LOCALHOST_ONLY=1\r\n")
    assert _up(tree)[0] == "1"


@pytest.mark.parametrize("env_text", [None, "VLM_URL=x\n", "ROS_LOCALHOST_ONLY=\n"],
                         ids=["no .env", "no key", "empty value"])
def test_missing_or_empty_key_leaves_it_unset(tree, env_text):
    # Unset, so compose's own default applies. Also guards against grep
    # finding nothing and set -e with pipefail stopping the script.
    if env_text is not None:
        (tree / ".env").write_text(env_text)
    assert _up(tree)[0] == "<unset>"


def test_gui_listens_where_the_platform_needs_and_the_operators_host_wins(tree):
    # On a Mac the container is on Docker's bridge, so the server must listen
    # on the container's own address (compose.mac.yaml keeps the port to the
    # Mac). On Linux the server's own default applies. Either way an --host
    # the operator types comes last, and argparse keeps the last one.
    _, args = _up(tree, command=("gui", "--host", "127.0.0.1"))
    words = args.split()
    assert words[:2] == ["exec", "-i"] and "wojtek_vlm_gui.server" in words
    if platform.system() == "Darwin":
        assert words[-4:] == ["--host", "0.0.0.0", "--host", "127.0.0.1"]
    else:
        assert words.count("--host") == 1 and words[-2:] == ["--host", "127.0.0.1"]


def test_the_mac_publishes_the_page_on_its_own_loopback_only():
    text = (EXPERIMENT / "docker" / "compose.mac.yaml").read_text()
    ports = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("- ") and "8501" in ln]
    assert ports == ['- "127.0.0.1:8501:8501"']
