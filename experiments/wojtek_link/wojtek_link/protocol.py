"""The Foxglove WebSocket protocol, the part the relay speaks. Pure; tested.

Measured against foxglove_bridge 3.5.0 (jazzy, the robot's): it picks the
subprotocol "foxglove.sdk.v1", advertises the capability clientPublish, and
carries ROS messages as "cdr" -- the rclpy serialized bytes, the 4-byte CDR
encapsulation header included.

Server -> client
  text   {"op": "serverInfo" | "advertise" | "unadvertise" | ...}
  binary 0x01, subscription id u32 LE, log time ns u64 LE, payload
Client -> server
  text   {"op": "subscribe", "subscriptions": [{"id", "channelId"}]}
         {"op": "advertise", "channels": [{"id", "topic", "encoding", "schemaName"}]}
  binary 0x01, client channel id u32 LE, payload
"""

from __future__ import annotations

import json
import struct
from typing import Dict, Iterable, List, Optional, Tuple

SUBPROTOCOLS = ("foxglove.sdk.v1", "foxglove.websocket.v1")
OP_MESSAGE_DATA = 0x01
_SERVER_HEADER = struct.Struct("<BIQ")     # opcode, subscription id, log time
_CLIENT_HEADER = struct.Struct("<BI")      # opcode, channel id


def parse_server_binary(frame: bytes) -> Optional[Tuple[int, int, bytes]]:
    """(subscription id, log time ns, payload) of a message-data frame; None
    for any other binary opcode or a frame too short to be one."""
    if len(frame) < _SERVER_HEADER.size or frame[0] != OP_MESSAGE_DATA:
        return None
    _op, sub_id, log_time = _SERVER_HEADER.unpack_from(frame)
    return sub_id, log_time, bytes(frame[_SERVER_HEADER.size:])


def client_message(channel_id: int, payload: bytes) -> bytes:
    return _CLIENT_HEADER.pack(OP_MESSAGE_DATA, channel_id) + payload


def subscribe(pairs: Iterable[Tuple[int, int]]) -> str:
    """pairs: (our subscription id, the server's channel id)."""
    return json.dumps({"op": "subscribe",
                       "subscriptions": [{"id": s, "channelId": c} for s, c in pairs]})


def advertise(channels: Iterable[Tuple[int, str, str]]) -> str:
    """channels: (our channel id, topic, ROS type)."""
    return json.dumps({"op": "advertise", "channels": [
        {"id": cid, "topic": topic, "encoding": "cdr", "schemaName": schema}
        for cid, topic, schema in channels]})


def match_channels(advertised: Dict[str, dict], wanted: Dict[str, str]) -> Tuple[Dict[str, int], List[str]]:
    """The server's channel id for every wanted topic it advertises with the
    right type and the cdr encoding, and the problems for the rest (missing,
    another type, another encoding) as readable lines."""
    ids: Dict[str, int] = {}
    problems: List[str] = []
    for topic, schema in wanted.items():
        ch = advertised.get(topic)
        if ch is None:
            problems.append(f"{topic}: not advertised by the robot's bridge")
        elif ch.get("schemaName") != schema:
            problems.append(f"{topic}: type {ch.get('schemaName')} on the robot, the link expects {schema}")
        elif ch.get("encoding") != "cdr":
            problems.append(f"{topic}: encoding {ch.get('encoding')}, the link speaks cdr")
        else:
            ids[topic] = int(ch["id"])
    return ids, problems
