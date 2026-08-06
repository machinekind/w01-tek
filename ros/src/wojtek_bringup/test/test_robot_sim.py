"""In sim mode the launch owns the console and the pad; robot.py must not.

sim.launch.py carries console:=web|qt|none and gamepad:=true since sim.sh was
folded into it. robot.py used to also spawn its own console (Qt by default)
and its own gamepad.launch.py in sim mode, so a plain `robot --sim` came up
with two consoles -- the launch's web one and the script's Qt one. These
tests are the fence: in sim mode robot.py starts processes for the session
(the sim launch, viz), never a console or pad of its own, and its flags
arrive as launch arguments instead.
"""

import subprocess
import sys

import pytest

from wojtek_bringup import robot
from wojtek_bringup.robot import sim_session_args


class _Args:
    def __init__(self, **kw):
        self.no_console = kw.get("no_console", False)
        self.web_console = kw.get("web_console", False)
        self.gamepad = kw.get("gamepad", False)


def test_default_leaves_the_console_choice_to_the_launch():
    # No token at all: sim.launch.py's own default (web) is the default.
    assert sim_session_args(_Args()) == []


def test_no_console_translates_to_console_none():
    assert sim_session_args(_Args(no_console=True)) == ["console:=none"]


def test_web_console_translates_to_console_web():
    assert sim_session_args(_Args(web_console=True)) == ["console:=web"]


def test_no_console_wins_over_web_console():
    args = _Args(no_console=True, web_console=True)
    assert sim_session_args(args) == ["console:=none"]


def test_gamepad_translates_to_launch_argument():
    assert sim_session_args(_Args(gamepad=True)) == ["gamepad:=true"]


class _FakeProc:
    def wait(self, timeout=None):
        return 0

    def send_signal(self, sig):
        pass


@pytest.fixture
def spawned(monkeypatch):
    """Run robot.main() with Popen recorded instead of executed."""
    commands = []

    def fake_popen(cmd, *a, **kw):
        commands.append(cmd)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    def run(argv):
        monkeypatch.setattr(sys, "argv", ["robot"] + argv)
        robot.main()
        return commands

    return run


def _sim_launch(commands):
    launches = [c for c in commands if "sim.launch.py" in c]
    assert len(launches) == 1, commands
    return launches[0]


def test_plain_sim_spawns_no_console_process(spawned):
    commands = spawned(["--sim", "--no-viz"])
    _sim_launch(commands)
    consoles = [
        c for c in commands
        if "run" in c and ("console" in c or "web_console" in c)
    ]
    assert consoles == [], "robot.py spawned a console on top of the launch's"


def test_sim_web_console_flag_reaches_the_launch(spawned):
    commands = spawned(["--sim", "--no-viz", "--web-console"])
    assert "console:=web" in _sim_launch(commands)
    assert len(commands) == 1, commands


def test_sim_gamepad_is_a_launch_argument_not_a_process(spawned):
    commands = spawned(["--sim", "--no-viz", "--gamepad"])
    assert "gamepad:=true" in _sim_launch(commands)
    gamepads = [c for c in commands if "gamepad.launch.py" in c]
    assert gamepads == [], "robot.py spawned gamepad.launch.py next to the launch's"


def test_explicit_passthrough_tokens_come_after_the_translated_flags(spawned):
    # A hand-typed console:= must win over the flag translation; ros2 launch
    # lets the last occurrence of an argument win.
    commands = spawned(["--sim", "--no-viz", "--no-console", "console:=qt"])
    cmd = _sim_launch(commands)
    assert cmd.index("console:=none") < cmd.index("console:=qt")
