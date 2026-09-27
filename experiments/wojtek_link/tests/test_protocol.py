"""The Foxglove frames the relay reads and writes. Pure."""

import json
import struct

from wojtek_link import protocol


def test_a_server_message_frame_is_opcode_subscription_logtime_payload():
    frame = struct.pack("<BIQ", 1, 7, 1_790_467_326_081_395_980) + b"\x00\x01\x00\x00cdr"
    assert protocol.parse_server_binary(frame) == (7, 1_790_467_326_081_395_980, b"\x00\x01\x00\x00cdr")


def test_other_binary_opcodes_and_short_frames_are_not_messages():
    assert protocol.parse_server_binary(struct.pack("<BIQ", 2, 7, 0) + b"x") is None
    assert protocol.parse_server_binary(b"\x01\x00\x00") is None


def test_a_client_message_is_opcode_channel_payload():
    assert protocol.client_message(3, b"abc") == b"\x01\x03\x00\x00\x00abc"


def test_subscribe_and_advertise_are_the_protocol_s_json():
    assert json.loads(protocol.subscribe([(1, 23), (2, 34)])) == {
        "op": "subscribe", "subscriptions": [{"id": 1, "channelId": 23}, {"id": 2, "channelId": 34}]}
    assert json.loads(protocol.advertise([(1, "/cmd_vel", "geometry_msgs/msg/Twist")])) == {
        "op": "advertise", "channels": [{"id": 1, "topic": "/cmd_vel", "encoding": "cdr",
                                         "schemaName": "geometry_msgs/msg/Twist"}]}


def test_channels_are_matched_by_topic_type_and_encoding():
    advertised = {  # as the robot's bridge 3.5.0 advertised them in the sim
        "/wojtek/odom": {"id": 23, "encoding": "cdr", "schemaName": "nav_msgs/msg/Odometry"},
        "/tf_static": {"id": 34, "encoding": "cdr", "schemaName": "tf2_msgs/msg/TFMessage"},
        "/wojtek/nav/status": {"id": 29, "encoding": "json", "schemaName": "std_msgs/msg/String"},
        "/x": {"id": 5, "encoding": "cdr", "schemaName": "std_msgs/msg/Empty"},
    }
    wanted = {"/wojtek/odom": "nav_msgs/msg/Odometry", "/tf_static": "tf2_msgs/msg/TFMessage",
              "/wojtek/nav/status": "std_msgs/msg/String", "/x": "std_msgs/msg/String",
              "/missing": "std_msgs/msg/String"}
    ids, problems = protocol.match_channels(advertised, wanted)
    assert ids == {"/wojtek/odom": 23, "/tf_static": 34}
    assert len(problems) == 3
    assert any("/missing: not advertised" in p for p in problems)
    assert any("/x: type std_msgs/msg/Empty" in p for p in problems)
    assert any("/wojtek/nav/status: encoding json" in p for p in problems)
