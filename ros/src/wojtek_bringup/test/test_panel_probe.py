"""The Deck's panel icon opens only a host that really serves the panel.

ros/deploy/deck/deck_panel.sh looks for the machine serving the wojtek_deck
panel: the robot, the PC's address, then every host on the Deck's networks.
A host counts only when its page carries the panel's title, so a home router
or some other service listening on port 8090 is not opened by mistake. The
first test keeps that title and the real page from drifting apart; the
second runs the probe against a real page, a different page, a socket that
never answers and a closed port.

The script lives outside ros/src, so inside the ROS container (only ros/src
is mounted) and on the robot (deploy.sh ships ros/src only) these tests
skip. The probe needs GNU `timeout`, which macOS does not have, so the
second test runs on a Linux host.
"""

import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROS = Path(__file__).resolve().parents[3]
SCRIPT = ROS / "deploy" / "deck" / "deck_panel.sh"
PAGE = ROS / "src" / "wojtek_deck" / "web" / "index.html"

pytestmark = pytest.mark.skipif(
    not SCRIPT.is_file(), reason="ros/deploy is not here (only ros/src is)"
)


def _marker():
    found = re.search(r"^MARKER='([^']*)'", SCRIPT.read_text(), re.MULTILINE)
    assert found, "deck_panel.sh no longer sets MARKER='...'"
    return found.group(1)


def test_marker_is_the_page_title():
    assert _marker() in PAGE.read_text()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_listening(port, deadline_s=10.0):
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"nothing came up on port {port}")


def _serve(directory, port):
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=directory,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_until_listening(port)
    return proc


def _probe(port):
    return subprocess.run(
        ["bash", str(SCRIPT), "--probe", "127.0.0.1", str(port)],
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.mark.skipif(
    shutil.which("timeout") is None or shutil.which("bash") is None,
    reason="needs bash and GNU timeout (macOS has no timeout)",
)
def test_probe_accepts_only_the_panel_page(tmp_path):
    panel_dir = tmp_path / "panel"
    other_dir = tmp_path / "other"
    panel_dir.mkdir()
    other_dir.mkdir()
    shutil.copy(PAGE, panel_dir / "index.html")
    (other_dir / "index.html").write_text(
        "<html><head><title>Other thing</title></head></html>\n"
    )

    procs = []
    silent = socket.socket()
    try:
        port_panel, port_other = _free_port(), _free_port()
        procs.append(_serve(panel_dir, port_panel))
        procs.append(_serve(other_dir, port_other))
        # Accepts the connection (the kernel does that) and never says a word.
        silent.bind(("127.0.0.1", 0))
        silent.listen(8)
        port_silent = silent.getsockname()[1]
        port_closed = _free_port()

        result = _probe(port_panel)
        assert (result.returncode, result.stdout.strip()) == (0, "panel")

        for port in (port_other, port_closed):
            result = _probe(port)
            assert (result.returncode, result.stdout.strip()) == (1, "no")

        start = time.monotonic()
        result = _probe(port_silent)
        assert (result.returncode, result.stdout.strip()) == (1, "no")
        assert time.monotonic() - start < 5.0
    finally:
        silent.close()
        for proc in procs:
            proc.terminate()
            proc.wait(timeout=5)
