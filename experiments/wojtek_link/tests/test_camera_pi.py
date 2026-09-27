"""The camera Pi's run files agree with what the relay reads. Pure: the
YAML is read as text (no PyYAML needed), the script only syntax-checked."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from wojtek_link import topics

CAMERA_PI = Path(__file__).resolve().parent.parent / "camera_pi"


def _list_under(text: str, key: str) -> list:
    """The '- item' lines of a block list `key:` in a YAML file."""
    m = re.search(rf"^(\s*){key}:\s*\n((?:\1\s+-\s.*\n)+)", text, re.M)
    assert m, f"no list {key!r}"
    return [line.split("-", 1)[1].strip().strip("'\"") for line in m.group(2).splitlines()]


def test_the_bridge_serves_exactly_what_the_relay_reads():
    text = (CAMERA_PI / "bridge.yaml").read_text()
    served = {p.removeprefix("^").removesuffix("$") for p in _list_under(text, "topic_whitelist")}
    assert served == set(topics.CAMERA_PI)


def test_the_bridge_is_read_only():
    text = (CAMERA_PI / "bridge.yaml").read_text()
    caps = re.search(r"^\s*capabilities:\s*\[(.*)\]\s*$", text, re.M)
    assert caps, "capabilities must be set: the bridge's default includes clientPublish and services"
    names = {c.strip().strip("'\"") for c in caps.group(1).split(",") if c.strip()}
    assert names, "an empty capabilities list aborts foxglove_bridge 3.5.0"
    assert not names & {"clientPublish", "services", "parameters", "parametersSubscribe", "assets"}


def test_the_driver_runs_the_pi_safe_set_at_the_wifi_s_rates():
    text = (CAMERA_PI / "realsense.yaml").read_text()

    def val(key):
        return re.search(rf"^\s*{re.escape(key)}:\s*(\S+)", text, re.M).group(1)

    for off in ("pointcloud.enable", "align_depth.enable", "enable_rgbd", "enable_sync",
                "enable_infra1", "enable_infra2"):
        assert val(off) == "false", off           # the set that SIGSEGVs the driver on a Pi stays off
    w, h, fps = map(int, val("depth_module.depth_profile").split("x"))
    assert (w, h) == (424, 240)                   # the sensor's own, no decimation on the Pi
    assert 1000.0 / fps / 2 <= 100.0              # every colour frame within pixel_goal_node's max_skew_s
    assert val("rgb_camera.color_profile").startswith("640x480x")


def test_the_run_script_parses_and_keeps_the_graph_local():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    subprocess.run([bash, "-n", str(CAMERA_PI / "run.sh")], check=True)
    text = (CAMERA_PI / "run.sh").read_text()
    assert "export ROS_LOCALHOST_ONLY=1" in text   # the camera Pi's graph never joins the LAN's
    # it installs and reconfigures nothing: sudo, apt or systemctl only ever in advice it prints
    for line in text.splitlines():
        code = line.strip()
        if code.startswith("#") or not re.search(r"\b(apt|apt-get|sudo|systemctl|usermod)\b", code):
            continue
        assert code.startswith("echo "), line
