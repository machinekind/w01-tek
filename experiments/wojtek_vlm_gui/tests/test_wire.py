"""The websocket protocol between server.py and the JS page. Pure."""

import json
import struct

import pytest

from wojtek_vlm_gui import wire


def test_jpeg_frames_carry_their_tag_first():
    assert wire.pack_jpeg(wire.TAG_CAMERA, b"\xff\xd8x") == b"\x00\xff\xd8x"


def test_annotated_frames_carry_the_number_of_their_ask_line():
    frame = wire.pack_annotated(7, b"\xff\xd8x")
    assert frame[0] == wire.TAG_ANNOTATED
    assert struct.unpack_from("<I", frame, 1)[0] == 7   # the page reads it as a u32 LE at offset 1
    assert frame[5:] == b"\xff\xd8x"
    assert wire.unpack_annotated(frame) == (7, b"\xff\xd8x")


def test_depth_frame_pixels_start_at_an_even_offset_and_round_trip():
    pixels = struct.pack("<6H", 0, 200, 1000, 4000, 65535, 7)
    frame = wire.pack_depth(3, 2, pixels)
    assert frame[0] == wire.TAG_DEPTH and wire.DEPTH_HEADER.size == 8
    assert wire.unpack_depth(frame) == (3, 2, pixels)
    # the page views the pixels as a Uint16Array at offset 8 (must be even)
    assert struct.unpack_from("<6H", frame, 8)[2] == 1000


def test_depth_frame_refuses_a_size_mismatch():
    with pytest.raises(ValueError):
        wire.pack_depth(3, 2, b"\x00" * 10)


def test_log_message_formats_lines_and_colours_the_finish():
    snap = {
        "instruction": "go to the pillar",
        "lines": [
            {"action": "ask", "step": 1, "answer": {"type": "not_visible"}, "latency_s": 1.3, "_picture": 3},
            {"action": "finished", "step": 2, "result": "done"},
        ],
        "finished": {"action": "finished", "step": 2, "result": "done"},
    }
    msg = wire.log_message(snap)
    json.dumps(msg)   # the picture never reaches the JSON
    assert msg["t"] == "log" and msg["instruction"] == "go to the pillar"
    assert msg["lines"][0]["text"].startswith("#1 ask")
    assert msg["lines"][0]["s"]["answer"] == {"type": "not_visible"} and "_picture" not in msg["lines"][0]["s"]
    assert msg["lines"][0]["picture"] == 3 and msg["lines"][1]["picture"] is None
    assert msg["finished"] == {"text": "finished: done", "ok": True}
    bad = wire.log_message({"lines": [], "finished": {"action": "finished", "result": "error", "error": "x"}})
    assert bad["finished"] == {"text": "finished: error -- x", "ok": False}


def test_parse_command_accepts_the_protocol_and_nothing_else():
    assert wire.parse_command('{"t": "stop"}') == {"t": "stop"}
    assert wire.parse_command('{"t": "task", "text": "go"}') == {"t": "task", "text": "go"}
    assert wire.parse_command('{"t": "svc", "name": "arm"}') == {"t": "svc", "name": "arm"}
    long = wire.parse_command(json.dumps({"t": "task", "text": "x" * 5000}))
    assert len(long["text"]) == wire.MAX_TASK_CHARS
    for junk in ("", "nope", "[]", '{"t": "svc", "name": "reboot"}', '{"t": "task", "text": 5}',
                 '{"t": "cmd_vel", "x": 1}'):
        assert wire.parse_command(junk) is None


def test_stop_and_only_stop_jumps_the_queue():
    assert wire.is_urgent(wire.parse_command('{"t": "stop"}'))
    assert not wire.is_urgent(wire.parse_command('{"t": "task", "text": "go"}'))
    assert not wire.is_urgent(wire.parse_command('{"t": "svc", "name": "arm"}'))


def test_the_page_can_reach_only_the_operator_services():
    assert set(wire.SERVICES) == {"stand_up", "lie_down", "arm", "disarm", "policy_on", "policy_off"}


def test_origin_allowed_takes_the_pages_own_address_and_nothing_else():
    for origin, host in [("http://localhost:8501", "localhost:8501"),
                         ("http://LOCALHOST:8501", "localhost:8501"),
                         ("http://127.0.0.1:8501", "127.0.0.1:8501"),
                         ("http://[::1]:8501", "[::1]:8501")]:
        assert wire.origin_allowed(origin, host) is True, (origin, host)
    for origin, host in [("http://evil.example", "localhost:8501"),
                         ("http://localhost:8501.evil.example", "localhost:8501"),
                         ("http://localhost:8502", "localhost:8501"),
                         ("http://127.0.0.1:8501", "localhost:8501"),
                         ("null", "localhost:8501"),
                         ("", "localhost:8501"),
                         ("file://", "localhost:8501"),
                         ("http://localhost:8501", None),
                         ("http://localhost:8501", "")]:
        assert wire.origin_allowed(origin, host) is False, (origin, host)


def test_a_client_without_an_origin_header_is_not_a_browser_and_passes():
    assert wire.origin_allowed(None, "localhost:8501") is True
