"""ros/docker/forward_env.sh: what sim.sh and dev.sh carry into the container.

The helper reads a short list of personal values from the host environment or
the repo-root .env and hands them to `docker exec`. These tests source the
real file from a scratch directory with a made-up .env next to it, the way
sim.sh does, and look at what comes out: the DOCKER_ENV array and the
environment the docker command would inherit.

Three things are fenced here. A token's value never goes into the docker
command line, where ps shows it to every user on the box. A .env saved on
Windows (a carriage return on every line) or with spaces after a value comes
out clean. And a Mac's MUJOCO_GL=cgl, which MuJoCo on Linux refuses, stays
on the host.

Plain bash and the standard library, no ROS. The script lives outside
ros/src, so inside the container (only ros/src is mounted) and on the robot
(deploy.sh ships ros/src only) these tests skip.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "docker" / "forward_env.sh"

pytestmark = [
    pytest.mark.skipif(
        not SCRIPT.is_file(), reason="ros/docker is not here (only ros/src is)"
    ),
    pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash"),
]

# Source the helper like sim.sh does, then print the array and the
# environment between two markers so the script's own messages stay apart.
_PROBE = (
    "set -eo pipefail; DOCKER_ENV=(); {prelude}"
    ' . "$1";'
    ' echo "--DOCKER_ENV--";'
    ' printf "%s\\n" ${{DOCKER_ENV[@]+"${{DOCKER_ENV[@]}}"}};'
    ' echo "--ENV--"; env'
)


def run(tmp_path, env_bytes=b"", extra_env=None, prelude=""):
    """Source forward_env.sh from tmp/a/b with env_bytes as tmp/.env.

    Returns (messages, items, forwarded): what the script printed, the
    DOCKER_ENV items in order, and {name: value} for each forwarded name as
    the docker command's environment holds it.
    """
    (tmp_path / ".env").write_bytes(env_bytes)
    cwd = tmp_path / "a" / "b"
    cwd.mkdir(parents=True)
    # Only PATH from the host, so the developer's own HF_* or VLM_* cannot
    # leak into the result.
    env = {"PATH": os.environ["PATH"], **(extra_env or {})}
    out = subprocess.run(
        ["bash", "-c", _PROBE.format(prelude=prelude), "_", str(SCRIPT)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    messages, rest = out.split("--DOCKER_ENV--\n", 1)
    items_text, env_text = rest.split("--ENV--\n", 1)
    # printf with an empty array still prints one empty line.
    items = [item for item in items_text.splitlines() if item]
    dumped = dict(
        line.split("=", 1) for line in env_text.splitlines() if "=" in line
    )
    names = [items[i + 1] for i in range(0, len(items) - 1, 2) if items[i] == "-e"]
    forwarded = {name: dumped.get(name) for name in names}
    return messages, items, forwarded, out


def test_token_value_not_in_docker_argv(tmp_path):
    _, items, forwarded, _ = run(tmp_path, b'HF_TOKEN="hf_test_value"\n')
    assert items == ["-e", "HF_TOKEN"]
    assert not any("hf_test_value" in item for item in items)
    # The value still reaches docker, through its environment.
    assert forwarded == {"HF_TOKEN": "hf_test_value"}


def test_unexported_shell_var_is_exported(tmp_path):
    # A plain shell variable, not exported: a bare -e would send nothing for
    # it unless the helper exports it.
    _, items, forwarded, _ = run(tmp_path, prelude="MUJOCO_GL=egl;")
    assert items == ["-e", "MUJOCO_GL"]
    assert forwarded == {"MUJOCO_GL": "egl"}


def test_path_policy_not_forwarded(tmp_path):
    messages, items, _, _ = run(tmp_path, b"WOJTEK_POLICY=runs/x/export\n")
    assert "WOJTEK_POLICY" not in items
    assert "is a path" in messages


def test_crlf_and_padded_values_are_cleaned(tmp_path):
    env_bytes = (
        b"HF_ORGANIZATION=myorg\r\n"
        b"HF_TOKEN=\r\n"
        b'WOJTEK_POLICY="myorg/pol@v1"\r\n'
        b"VLM_URL=   \n"
        b"VLM_MODEL=qwen3-vl:30b \r\n"
        b"ROS_LOCALHOST_ONLY=1\r\n"
    )
    messages, _, forwarded, out = run(tmp_path, env_bytes)
    assert forwarded == {
        "HF_ORGANIZATION": "myorg",
        "WOJTEK_POLICY": "myorg/pol@v1",
        "VLM_MODEL": "qwen3-vl:30b",
        "ROS_LOCALHOST_ONLY": "1",
    }
    assert "\r" not in out
    assert "is a path" not in messages


def test_quoted_lf_values_unchanged(tmp_path):
    env_bytes = b"HF_ORGANIZATION=\"myorg\"\nVLM_URL='http://localhost:11434'\n"
    _, _, forwarded, _ = run(tmp_path, env_bytes)
    assert forwarded == {
        "HF_ORGANIZATION": "myorg",
        "VLM_URL": "http://localhost:11434",
    }


def test_host_environment_wins(tmp_path):
    env_bytes = b"HF_ORGANIZATION=\"myorg\"\nVLM_URL='http://localhost:11434'\n"
    _, _, forwarded, _ = run(
        tmp_path, env_bytes, extra_env={"HF_ORGANIZATION": "hostorg"}
    )
    assert forwarded["HF_ORGANIZATION"] == "hostorg"


@pytest.mark.parametrize("value", ["egl", "EGL", "osmesa", "glx", "glfw", "disabled"])
def test_linux_gl_backends_are_forwarded(tmp_path, value):
    messages, items, forwarded, _ = run(tmp_path, extra_env={"MUJOCO_GL": value})
    assert items == ["-e", "MUJOCO_GL"]
    assert forwarded == {"MUJOCO_GL": value}
    assert "MUJOCO_GL" not in messages


@pytest.mark.parametrize("value", ["cgl", "CGL", "wgl"])
def test_other_gl_backends_stay_on_the_host(tmp_path, value):
    messages, items, _, _ = run(tmp_path, extra_env={"MUJOCO_GL": value})
    assert items == []
    assert f"MUJOCO_GL={value} does not work in the Linux container" in messages


def test_nothing_set_forwards_nothing(tmp_path):
    messages, items, _, _ = run(tmp_path)
    assert items == []
    assert messages == ""
