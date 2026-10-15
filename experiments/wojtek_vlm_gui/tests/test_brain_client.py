"""The page's ROS 2 client against a mock node. No ROS runtime.

What it guards: the page writes only the instruction, the cancel and, on
STOP, zero Twists on /cmd_vel (never a non-zero one, never a goal), reads
only the brain's and goto's status and the annotated picture and the
camera's JPEG, refuses a task nobody listens to, and stops in the web
console's order before it freezes. Needs rclpy importable (the
wojtek_vlm_gui container); skipped elsewhere.
"""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("rclpy")

from geometry_msgs.msg import Twist  # noqa: E402
from sensor_msgs.msg import CompressedImage, Image  # noqa: E402
from std_msgs.msg import Empty, String  # noqa: E402

from wojtek_vlm_gui import limits  # noqa: E402
from wojtek_vlm_gui.brain_client import BrainClient  # noqa: E402


def _client(listening=True):
    """A mock node that hands out one publisher mock per topic, so the
    order and the target of every publish can be checked."""
    node = MagicMock()
    pubs = {}
    node.create_publisher.side_effect = lambda _t, topic, _d: pubs.setdefault(topic, MagicMock())
    node.count_subscribers.return_value = 1 if listening else 0
    return BrainClient(node, subscriber_wait_s=0.05, poll_s=0.01), node, pubs


def test_the_only_publishers_are_the_instruction_the_cancel_and_the_zero_cmd_vel():
    _, node, _ = _client()
    created = {c.args[1]: c.args[0] for c in node.create_publisher.call_args_list}
    assert created == {limits.INSTRUCTION_TOPIC: String, limits.CANCEL_TOPIC: Empty,
                       limits.CMD_VEL_TOPIC: Twist}
    assert set(created) == set(limits.WRITABLE_TOPICS) | set(limits.ZERO_ONLY_TOPICS)
    for topic in limits.NEVER_PUBLISHED:
        assert topic not in created


def _is_zero(twist):
    return all(v == 0.0 for v in (twist.linear.x, twist.linear.y, twist.linear.z,
                                  twist.angular.x, twist.angular.y, twist.angular.z))


def test_freeze_cancels_first_then_zeroes_cmd_vel_and_starts_one_hold():
    order, holds = [], []
    client, _, pubs = _client()
    pubs[limits.CANCEL_TOPIC].publish.side_effect = lambda m: order.append("cancel")
    pubs[limits.INSTRUCTION_TOPIC].publish.side_effect = lambda m: order.append(f"instruction:{m.data!r}")
    pubs[limits.CMD_VEL_TOPIC].publish.side_effect = lambda m: order.append(
        "zero" if _is_zero(m) else "NON-ZERO")

    ok, text = client.freeze(start_hold=holds.append)

    assert ok and "zero" in text
    assert order == ["cancel", "instruction:''", "zero"]
    assert holds == [limits.FREEZE_PERIOD_S]


def test_a_stop_during_a_hold_extends_it_instead_of_starting_a_second():
    holds = []
    client, _, _ = _client()
    client.freeze(start_hold=holds.append)
    first_until = client._freeze_until
    client.freeze(start_hold=holds.append)
    assert len(holds) == 1 and client._freeze_until >= first_until


def test_the_hold_publishes_zeros_until_it_runs_out_and_then_releases():
    client, _, pubs = _client()
    clock = {"t": 100.0}
    client._freeze_until = 100.0 + 1.0
    client._holding = True

    def sleep(dt):
        clock["t"] += dt

    client._hold_zero(0.05, sleep=sleep, clock=lambda: clock["t"])

    sent = [c.args[0] for c in pubs[limits.CMD_VEL_TOPIC].publish.call_args_list]
    assert 19 <= len(sent) <= 21 and all(_is_zero(t) for t in sent)
    assert client._holding is False
    # a STOP after the hold ended starts a fresh one
    holds = []
    client.freeze(start_hold=holds.append)
    assert holds == [limits.FREEZE_PERIOD_S]


def test_nothing_the_client_does_puts_a_non_zero_twist_on_cmd_vel():
    client, _, pubs = _client()
    client.send("go to the pillar")
    client.stop()
    client.freeze(start_hold=lambda _p: None)
    client.send("stop")
    for call in pubs[limits.CMD_VEL_TOPIC].publish.call_args_list:
        assert _is_zero(call.args[0])


def test_the_subscriptions_are_the_status_topics_the_pictures_and_the_robot_load():
    _, node, _ = _client()
    topics = {c.args[1] for c in node.create_subscription.call_args_list}
    assert topics == {limits.VLM_STATUS_TOPIC, limits.GOTO_STATUS_TOPIC,
                      limits.PIXEL_STATUS_TOPIC, limits.ANNOTATED_TOPIC, limits.CAMERA_TOPIC,
                      limits.DEPTH_TOPIC, limits.SYSMON_TOPIC, limits.GUIDE_TOPIC}
    assert limits.CAMERA_TOPIC.endswith("/compressed")   # the JPEG, never the raw image


def test_the_camera_jpeg_is_kept_latest_with_its_age():
    client, _, _ = _client()
    assert client.latest_jpeg() == (None, float("inf"))
    for payload in (b"\xff\xd8first", b"\xff\xd8second"):
        msg = CompressedImage()
        msg.format, msg.data = "jpeg", payload
        client._on_camera(msg)
    jpeg, age = client.latest_jpeg()
    assert jpeg == b"\xff\xd8second" and 0.0 <= age < 1.0


def test_send_publishes_the_instruction_as_typed():
    client, _, pubs = _client()
    ok, msg = client.send("  go to the purple pillar ")
    assert ok and "go to the purple pillar" in msg
    assert pubs[limits.INSTRUCTION_TOPIC].publish.call_args.args[0].data == "go to the purple pillar"
    pubs[limits.CANCEL_TOPIC].publish.assert_not_called()


def test_send_refuses_when_no_brain_listens_and_publishes_nothing():
    client, _, pubs = _client(listening=False)
    ok, msg = client.send("go to the purple pillar")
    assert not ok
    assert limits.INSTRUCTION_TOPIC in msg and "vlm:=true" in msg
    pubs[limits.INSTRUCTION_TOPIC].publish.assert_not_called()


def test_stop_sends_the_cancel_first_then_the_empty_instruction():
    order = []
    client, _, pubs = _client()
    pubs[limits.CANCEL_TOPIC].publish.side_effect = lambda m: order.append(("cancel", m))
    pubs[limits.INSTRUCTION_TOPIC].publish.side_effect = lambda m: order.append(("instruction", m.data))

    ok, _ = client.stop()

    assert ok
    assert [o[0] for o in order] == ["cancel", "instruction"]
    assert order[1][1] == ""


def test_a_stop_word_typed_as_a_task_is_a_stop_and_a_freeze_even_with_no_brain(monkeypatch):
    client, _, pubs = _client(listening=False)
    monkeypatch.setattr(client, "_start_hold_thread", lambda _p: None)
    ok, msg = client.send("stop")
    assert ok and msg.startswith("STOP")
    pubs[limits.CANCEL_TOPIC].publish.assert_called_once()
    assert pubs[limits.INSTRUCTION_TOPIC].publish.call_args.args[0].data == ""
    assert _is_zero(pubs[limits.CMD_VEL_TOPIC].publish.call_args.args[0])


def _record_instructions(pubs):
    sent = []
    pubs[limits.INSTRUCTION_TOPIC].publish.side_effect = lambda m: sent.append(m.data)
    return sent


def _stop_on_the_second_poll(client, node, stop):
    """The brain is not there on the first look; on the second the
    operator's STOP lands, and from then on the brain listens."""
    client._wait_s = 5.0   # the wait must not run out before the second look
    polls = {"n": 0}

    def count_subscribers(_topic):
        polls["n"] += 1
        if polls["n"] == 1:
            return 0
        if polls["n"] == 2:
            stop()
        return 1

    node.count_subscribers.side_effect = count_subscribers


def _nothing_carries(pubs, text):
    for pub in pubs.values():
        for call in pub.publish.call_args_list:
            assert getattr(call.args[0], "data", None) != text


def test_a_stop_during_the_subscriber_wait_drops_the_task_instead_of_sending_it_after():
    client, node, pubs = _client(listening=False)
    sent = _record_instructions(pubs)
    _stop_on_the_second_poll(client, node, lambda: client.freeze(start_hold=lambda _p: None))

    ok, msg = client.send("go to the pillar")

    assert not ok and "STOP" in msg
    assert sent == [""]   # the stop's empty instruction, and nothing after it
    pubs[limits.CANCEL_TOPIC].publish.assert_called_once()
    _nothing_carries(pubs, "go to the pillar")


def test_a_service_stop_during_the_subscriber_wait_also_drops_the_task():
    # Disarm and Lie down call stop() directly, not freeze().
    client, node, pubs = _client(listening=False)
    sent = _record_instructions(pubs)
    _stop_on_the_second_poll(client, node, client.stop)

    ok, _ = client.send("go to the pillar")

    assert not ok
    assert sent == [""]
    _nothing_carries(pubs, "go to the pillar")


def test_a_stop_before_the_task_does_not_block_the_next_task():
    client, _, pubs = _client()
    client.freeze(start_hold=lambda _p: None)
    ok, _ = client.send("go to the box")
    assert ok
    assert pubs[limits.INSTRUCTION_TOPIC].publish.call_args.args[0].data == "go to the box"


def test_status_messages_land_in_the_log_in_order_and_per_task():
    client, _, _ = _client()

    def status(**kw):
        client._on_status(String(data=json.dumps(kw)))

    status(action="ask", step=1, instruction="go to the pillar", t=1.0, answer={"type": "not_visible"})
    status(action="turn", step=1, instruction="go to the pillar", t=1.5, deg=45.0, turned_total=45.0)
    status(action="finished", step=2, instruction="go to the pillar", t=2.0, result="cancelled")
    status(action="ask", step=1, instruction="go to the box", t=3.0, answer={"type": "goal"})
    client._on_status(String(data="garbage"))  # ignored, never raises

    snap = client.latest()
    assert snap["instruction"] == "go to the box"
    assert [s["action"] for s in snap["lines"]] == ["ask"]
    assert snap["finished"] is None and not snap["idle"]


def _picture(client):
    msg = Image()
    msg.encoding, msg.height, msg.width = "rgb8", 2, 3
    msg.data = bytes(range(2 * 3 * 3))
    client._on_annotated(msg)


def _line(client, action, step):
    client._on_status(String(data=json.dumps({"action": action, "step": step, "instruction": "x", "t": 0})))


def test_the_annotated_picture_is_numbered_for_its_ask_line():
    client, _, _ = _client()
    _picture(client)
    _line(client, "ask", 1)
    _line(client, "verify", 1)

    lines = client.latest()["lines"]
    assert lines[0]["_picture"] == 1
    assert "_picture" not in lines[1]
    number, img = client.latest_annotated()
    assert number == 1
    assert isinstance(img, np.ndarray) and img.shape == (2, 3, 3) and img[1, 2, 2] == 17
    assert client.seq()["annotated"] == 1


def test_a_picture_alone_is_not_sent_until_its_ask():
    client, _, _ = _client()
    _picture(client)
    assert client.seq()["annotated"] == 0
    assert client.latest_annotated() is None


def test_a_picture_goes_to_one_ask_only():
    client, _, _ = _client()
    _picture(client)
    _line(client, "ask", 1)
    _line(client, "ask", 2)
    lines = client.latest()["lines"]
    assert "_picture" not in lines[1] and client.seq()["annotated"] == 1
    _picture(client)
    _line(client, "ask", 3)
    assert client.latest()["lines"][2]["_picture"] == 2


def test_the_picture_is_subscribed_before_the_status():
    _, node, _ = _client()
    topics = [c.args[1] for c in node.create_subscription.call_args_list]
    assert topics.index(limits.ANNOTATED_TOPIC) < topics.index(limits.VLM_STATUS_TOPIC)


def test_a_picture_in_another_encoding_is_ignored():
    client, _, _ = _client()
    msg = Image()
    msg.encoding, msg.height, msg.width = "bgr8", 1, 1
    msg.data = bytes(3)
    client._on_annotated(msg)
    assert client._annotated is None


def test_the_robot_load_is_kept_latest_with_its_age_and_garbage_is_ignored():
    client, _, _ = _client()
    assert client.latest_sysmon() == (None, float("inf"))
    client._on_sysmon(String(data="not json"))
    assert client.latest_sysmon()[0] is None
    client._on_sysmon(String(data=json.dumps({"host": "robot-core", "cores": [50.0, 1.0, 0.0, 0.0],
                                              "isolated": [2, 3], "load": [1.0, 2.0, 3.0],
                                              "mem_used_mb": 1000, "mem_total_mb": 7800, "temp_c": 51.0})))
    data, age = client.latest_sysmon()
    assert data["host"] == "robot-core" and 0.0 <= age < 1.0


def _depth_msg(width, height, step, encoding="16UC1", fill=b"\x01\x02"):
    msg = Image()
    msg.encoding, msg.width, msg.height, msg.step, msg.is_bigendian = encoding, width, height, step, 0
    msg.data = fill * (step * height // 2)
    return msg


def test_depth_is_kept_as_packed_uint16_and_bumps_its_sequence():
    client, _, _ = _client()
    before = client.seq()["depth"]
    client._on_depth(_depth_msg(3, 2, 6))
    w, h, data = client.latest_depth()
    assert (w, h, len(data)) == (3, 2, 12) and client.seq()["depth"] == before + 1


def test_padded_depth_rows_are_trimmed_and_other_encodings_ignored():
    client, _, _ = _client()
    client._on_depth(_depth_msg(3, 2, 8))          # 2 pad bytes a row
    assert len(client.latest_depth()[2]) == 12
    client._on_depth(_depth_msg(3, 2, 6, encoding="32FC1"))
    assert client.seq()["depth"] == 1


def test_nav_status_words_are_kept_latest():
    client, _, _ = _client()
    client._on_goto(String(data="driving"))
    client._on_pixel(String(data="sent"))
    client._on_goto(String(data="reached"))
    snap = client.latest()
    assert (snap["goto"], snap["pixel"]) == ("reached", "sent")
