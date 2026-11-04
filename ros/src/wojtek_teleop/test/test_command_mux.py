"""Model-free ownership scenarios; no ROS, simulator or hardware required."""

from itertools import permutations
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wojtek_teleop.command_mux import CommandMux  # noqa: E402


FORWARD = (0.3, 0.0, 0.125, 0.0, 0.0, 0.0)
TURN = (0.0, 0.0, 0.125, 0.0, 0.0, 0.4)
STOP = (0.0, 0.0, 0.125, 0.0, 0.0, 0.0)


def test_startup_is_silent_and_autonomy_requires_explicit_enable():
    mux = CommandMux()
    assert mux.step(0).command is None
    assert not mux.receive("nav", FORWARD, 0.1)
    assert mux.step(0.2).command is None
    assert mux.set_autonomy_enabled(True, 0.3)
    # An input received while inhibited is never played on enable.
    assert mux.step(0.4).command is None
    assert mux.receive("nav", FORWARD, 0.5)
    assert mux.step(0.6).command == FORWARD


@pytest.mark.parametrize("sources", list(permutations(("pad", "deck", "console", "external"))))
def test_concurrent_manual_inputs_have_a_deterministic_winner(sources):
    mux = CommandMux()
    for source in sources:
        mux.receive(source, FORWARD if source == "pad" else TURN, 0.0)
    assert mux.owner == "pad"
    assert mux.step(0.1).command == FORWARD


def test_same_owner_refreshes_its_lease():
    mux = CommandMux()
    mux.receive("deck", FORWARD, 0)
    assert mux.step(0.4).command == FORWARD
    mux.receive("deck", TURN, 0.45)
    assert mux.step(0.9).command == TURN
    assert mux.step(0.951).command == STOP
    assert mux.step(1.0).command is None


def test_zero_commands_keep_the_lease_until_the_source_goes_silent():
    mux = CommandMux()
    mux.receive("pad", STOP, 0)
    assert not mux.receive("deck", FORWARD, 0.1)
    mux.receive("pad", STOP, 0.4)
    assert not mux.receive("deck", FORWARD, 0.8)
    assert mux.step(0.91).command == (0.0,) * 6
    assert mux.owner is None
    assert mux.step(1.0).command is None


def test_manual_takeover_cancels_and_latches_autonomy_off():
    mux = CommandMux()
    mux.set_autonomy_enabled(True, 0)
    mux.receive("nav", FORWARD, 0.01)
    assert mux.step(0.02).command == FORWARD
    mux.receive("external", TURN, 0.1)
    decision = mux.step(0.11)
    assert decision.command == TURN
    assert decision.cancel_autonomy
    assert not mux.autonomy_enabled
    assert not mux.receive("brain", FORWARD, 0.2)
    assert not mux.set_autonomy_enabled(True, 0.3)
    assert mux.reason == "manual_owner_active"
    assert mux.step(0.7).command == STOP
    assert not mux.receive("nav", FORWARD, 0.8)
    assert mux.set_autonomy_enabled(True, 0.9)
    assert mux.step(1.0).command is None
    assert mux.receive("nav", FORWARD, 1.01)
    assert mux.step(1.02).command == FORWARD


def test_repeated_manual_commands_do_not_repeat_cancel():
    mux = CommandMux()
    mux.receive("pad", FORWARD, 0)
    assert mux.step(0.01).cancel_autonomy
    mux.receive("pad", TURN, 0.1)
    assert not mux.step(0.11).cancel_autonomy


def test_expiry_has_one_stop_then_silence_and_no_waiting_source_replay():
    mux = CommandMux()
    mux.receive("pad", FORWARD, 0)
    mux.step(0.01)
    assert not mux.receive("deck", TURN, 0.4)
    assert mux.step(0.5).command == STOP
    assert mux.step(0.51).command is None
    assert mux.owner is None
    assert mux.receive("deck", TURN, 0.6)
    assert mux.step(0.61).command == TURN


def test_arrival_after_expiry_cannot_hide_the_stop_transition():
    mux = CommandMux()
    mux.receive("pad", FORWARD, 0)
    mux.step(0.01)
    mux.receive("deck", TURN, 0.6)
    assert mux.step(0.61).command == STOP
    assert mux.step(0.62).command == TURN


def test_disabling_autonomy_cancels_and_clears_its_owner():
    mux = CommandMux()
    mux.set_autonomy_enabled(True, 0)
    mux.receive("brain", TURN, 0.01)
    mux.step(0.02)
    assert mux.set_autonomy_enabled(False, 0.03)
    decision = mux.step(0.04)
    assert decision.cancel_autonomy
    assert decision.command == STOP
    assert mux.owner is None
    assert mux.step(0.05).command is None


def test_cancel_echo_does_not_republish_or_remove_manual_owner():
    mux = CommandMux()
    mux.receive("pad", FORWARD, 0)
    assert mux.step(0.01).cancel_autonomy
    mux.cancel_autonomy(0.02, notify=False)
    decision = mux.step(0.03)
    assert not decision.cancel_autonomy
    assert decision.command == FORWARD
    assert mux.owner == "pad"


def test_enable_cannot_overtake_an_unpublished_cancel():
    mux = CommandMux()
    assert mux.set_autonomy_enabled(False, 0)
    assert not mux.set_autonomy_enabled(True, 0.001)
    assert mux.reason == "cancel_pending"
    assert not mux.autonomy_enabled
    assert mux.step(0.05).cancel_autonomy
    assert mux.set_autonomy_enabled(True, 0.06)


def test_external_cancel_inhibits_autonomy_without_echo():
    mux = CommandMux()
    mux.set_autonomy_enabled(True, 0)
    mux.receive("nav", FORWARD, 0.01)
    mux.step(0.02)
    mux.cancel_autonomy(0.03, notify=False)
    decision = mux.step(0.04)
    assert decision.command == STOP
    assert not decision.cancel_autonomy
    assert not mux.receive("nav", TURN, 0.05)


def test_disabling_autonomy_keeps_manual_control():
    mux = CommandMux()
    mux.receive("console", FORWARD, 0)
    mux.step(0.01)
    mux.set_autonomy_enabled(False, 0.02)
    decision = mux.step(0.03)
    assert decision.command == FORWARD
    assert decision.cancel_autonomy


@pytest.mark.parametrize("command", [
    (float("nan"), 0, 0, 0, 0, 0),
    (0, float("inf"), 0, 0, 0, 0),
    (0, 0, float("-inf"), 0, 0, 0),
    (0, 0, 0, 0, 0, "bad"),
    (0, 0),
    None,
])
def test_bad_commands_do_not_replace_or_refresh_owner(command):
    mux = CommandMux()
    mux.receive("deck", FORWARD, 0)
    mux.step(0.01)
    assert not mux.receive("deck", command, 0.4)
    assert mux.step(0.49).command == FORWARD
    assert mux.step(0.5).command == STOP
    assert mux.status()["rejected"] == 1


def test_invalid_higher_priority_packet_cannot_take_over():
    mux = CommandMux()
    mux.set_autonomy_enabled(True, 0)
    mux.receive("nav", FORWARD, 0.01)
    assert not mux.receive("pad", (float("nan"),) * 6, 0.02)
    assert mux.autonomy_enabled
    assert mux.owner == "nav"
    assert not mux.step(0.03).cancel_autonomy


def test_unknown_source_is_rejected():
    mux = CommandMux()
    assert not mux.receive("unregistered", FORWARD, 0)
    assert mux.owner is None
    assert mux.step(0.1).command is None


def test_autonomous_sources_also_have_priority_and_no_replay():
    mux = CommandMux()
    mux.set_autonomy_enabled(True, 0)
    mux.receive("text", FORWARD, 0.01)
    mux.receive("nav", FORWARD, 0.02)
    mux.receive("brain", TURN, 0.03)
    assert mux.owner == "brain"
    mux.step(0.04)
    assert not mux.receive("nav", FORWARD, 0.4)
    assert mux.step(0.54).command == STOP
    assert mux.step(0.55).command is None


def test_timeout_preserves_the_last_emitted_height_channel():
    mux = CommandMux()
    command = (0.3, -0.1, 0.345, 0.0, 0.0, 0.2)
    mux.receive("external", command, 0)
    mux.step(0.01)
    assert mux.step(0.5).command == (0.0, 0.0, 0.345, 0.0, 0.0, 0.0)


def test_clock_regression_discards_command_and_inhibits_autonomy():
    mux = CommandMux()
    mux.set_autonomy_enabled(True, 10)
    mux.receive("nav", FORWARD, 10.01)
    mux.step(10.02)
    decision = mux.step(1)
    assert decision.command == STOP
    assert decision.cancel_autonomy
    assert not mux.autonomy_enabled
    assert mux.step(1.1).command is None


@pytest.mark.parametrize("timeout", [0, -0.5, float("inf"), float("nan")])
def test_invalid_timeouts_are_refused(timeout):
    with pytest.raises(ValueError):
        CommandMux(timeout)


def test_nonfinite_clock_is_refused():
    with pytest.raises(ValueError):
        CommandMux().step(float("nan"))
