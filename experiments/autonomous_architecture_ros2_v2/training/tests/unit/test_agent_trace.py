"""Session trace: JSONL on disk, bounded ring in memory, never fatal."""

import json

from wojtek_rl.agent.trace import TEXT_MAX, Trace, default_trace_path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_writes_jsonl_and_ring(tmp_path):
    t = Trace(tmp_path / "trace.jsonl")
    t.add("chat.ask", text="hello")
    t.add("chat.say", say="woof")
    t.close()
    rows = read_jsonl(tmp_path / "trace.jsonl")
    assert [r["kind"] for r in rows] == ["chat.ask", "chat.say"]
    assert rows[0]["text"] == "hello" and rows[1]["say"] == "woof"
    assert [r["seq"] for r in rows] == [1, 2]
    assert all("wall" in r and "t" in r for r in rows)
    assert [e["kind"] for e in t.recent()] == ["chat.ask", "chat.say"]


def test_creates_parent_directory(tmp_path):
    path = tmp_path / "deep" / "nested" / "trace.jsonl"
    Trace(path).add("session.start", scene="room")
    assert read_jsonl(path)[0]["scene"] == "room"


def test_ring_is_bounded_but_file_is_not(tmp_path):
    t = Trace(tmp_path / "t.jsonl", ring_max=3)
    for i in range(10):
        t.add("nav.state", step=i)
    t.close()
    assert [e["step"] for e in t.recent()] == [7, 8, 9]
    assert len(read_jsonl(tmp_path / "t.jsonl")) == 10


def test_recent_filters_by_kind_prefix(tmp_path):
    t = Trace(tmp_path / "t.jsonl")
    t.add("chat.ask", text="a")
    t.add("search.moving", text="b")
    t.add("chat.say", say="c")
    assert [e["kind"] for e in t.recent(kind_prefix="chat")] == ["chat.ask", "chat.say"]
    assert [e["kind"] for e in t.recent(kind_prefix="search")] == ["search.moving"]


def test_recent_limit_returns_tail(tmp_path):
    t = Trace(tmp_path / "t.jsonl")
    for i in range(5):
        t.add("nav.state", step=i)
    assert [e["step"] for e in t.recent(limit=2)] == [3, 4]


def test_long_text_is_clipped(tmp_path):
    t = Trace(tmp_path / "t.jsonl")
    event = t.add("chat.llm", raw="x" * (TEXT_MAX + 500))
    assert len(event["raw"]) < TEXT_MAX + 100
    assert "more chars" in event["raw"]


def test_memory_only_when_no_path():
    t = Trace(None)
    t.add("chat.ask", text="hi")
    assert len(t.recent()) == 1


def test_unwritable_path_degrades_to_ring(tmp_path):
    clash = tmp_path / "afile"
    clash.write_text("not a directory")
    t = Trace(clash / "trace.jsonl")  # parent is a file -> mkdir fails
    t.add("chat.ask", text="still recorded")
    assert [e["text"] for e in t.recent()] == ["still recorded"]


def test_non_serializable_value_does_not_raise(tmp_path):
    t = Trace(tmp_path / "t.jsonl")
    t.add("chat.tool", args={"obj": object()})  # default=str handles it
    t.close()
    assert len(read_jsonl(tmp_path / "t.jsonl")) == 1


def test_default_trace_path_shape(tmp_path):
    p = default_trace_path(tmp_path, "apartment")
    assert p.parent == tmp_path / "agent_traces"
    assert p.name.startswith("apartment_") and p.suffix == ".jsonl"


# -- live streaming ------------------------------------------------------------


def test_subscriber_sees_every_event():
    """The trace is the one place every decision passes through, so the UI
    streams it rather than instrumenting each subsystem twice."""
    t = Trace(None)
    seen = []
    t.subscribe(seen.append)
    t.add("chat.ask", text="hi")
    t.add("nav.state", state="thinking")
    assert [e["kind"] for e in seen] == ["chat.ask", "nav.state"]


def test_unsubscribe_stops_delivery():
    t = Trace(None)
    seen = []
    listener = seen.append
    t.subscribe(listener)
    t.add("chat.ask", text="a")
    t.unsubscribe(listener)
    t.add("chat.ask", text="b")
    assert len(seen) == 1


def test_a_broken_listener_is_dropped_not_fatal():
    """A disconnected websocket must not take tracing down with it."""
    t = Trace(None)
    good = []

    def boom(_e):
        raise RuntimeError("socket closed")

    t.subscribe(boom)
    t.subscribe(good.append)
    t.add("chat.ask", text="a")   # boom raises here and is unsubscribed
    t.add("chat.ask", text="b")
    assert len(good) == 2
    assert len(t._listeners) == 1
    assert len(t.recent()) == 2   # both still recorded
