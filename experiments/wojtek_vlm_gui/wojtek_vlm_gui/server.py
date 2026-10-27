"""The operator page's server: one ROS 2 node, one port, a JS page.

    python3 -m wojtek_vlm_gui.server [--port 8501] [--host 127.0.0.1]   ->  http://localhost:8501

By default it listens on this computer only. On a Mac, `run.sh gui` passes
--host 0.0.0.0, because there the container sits on Docker's bridge network;
the ports line in docker/compose.mac.yaml then decides who reaches it (the
Mac itself only). The websocket also refuses a handshake whose Origin is not
the page's own address, so another web page open in the operator's browser
cannot drive the robot.

Runs on the PC, never on the robot: the RPi only publishes (the camera, its
depth, sysmon_node's load). GET / serves web/index.html and its script; the
websocket on /ws carries the protocol in wire.py. Per connection, a sender
loop looks at the client's sequence numbers every SEND_TICK_S and sends
only what changed -- the newest camera and depth frame, never a backlog --
so a slow browser drops frames instead of lagging behind the robot.

What the page may do on the robot is what the operator page always did:
the instruction and the cancel (BrainClient), the four operator services
(arm_switch). Disarm and Lie down also STOP the brain, first, before their
service call. STOP itself freezes the robot (BrainClient.freeze: cancel,
then /cmd_vel held at zero) and is handled the moment it arrives: every
other command runs as its own task, so a STOP never waits behind a slow
service call or a task send.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import uvicorn
from PIL import Image as PILImage
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from wojtek_vlm_gui import arm_switch, wire
from wojtek_vlm_gui.brain_client import BrainClient, start_client

WEB_DIR = Path(__file__).resolve().parent / "web"
SEND_TICK_S = 0.02            # 50 Hz look at the client's state; the camera is 20 fps
SYS_EVERY_S = 1.0             # the load panel; its age goes with it
ANNOTATED_JPEG_QUALITY = 85

# name -> (call on the node, also STOP the brain)
SERVICE_CALLS: Dict[str, Tuple[Callable, bool]] = {
    "stand_up": (arm_switch.stand_up, False),
    "lie_down": (arm_switch.lie_down, True),
    "arm": (lambda n: arm_switch.set_armed(n, True), False),
    "disarm": (lambda n: arm_switch.set_armed(n, False), True),
    "policy_on": (lambda n: arm_switch.set_policy_enabled(n, True), False),
    "policy_off": (lambda n: arm_switch.set_policy_enabled(n, False), False),
}
assert set(SERVICE_CALLS) == set(wire.SERVICES)


def _jpeg_of(rgb) -> bytes:
    buf = io.BytesIO()
    PILImage.fromarray(rgb, "RGB").save(buf, "JPEG", quality=ANNOTATED_JPEG_QUALITY)
    return buf.getvalue()


async def run_command(client: BrainClient, cmd) -> dict:
    """Blocking ROS calls (a subscriber wait, a service round trip) go to a
    worker thread so the event loop keeps streaming frames. STOP never gets
    here: ws_endpoint answers it inline (see there). Disarm and Lie down
    stop the brain before their call, not after it: a page that closes
    during the call cancels this task at the await, while the robot's call
    still goes through in its thread."""
    if cmd["t"] == "task":
        ok, text = await asyncio.to_thread(client.send, cmd["text"])
    else:
        call, stops_brain = SERVICE_CALLS[cmd["name"]]
        if stops_brain:
            client.stop()   # first: a page closed mid-call cancels this task at the await below
        ok, text = await asyncio.to_thread(call, client.node)
        if stops_brain:
            text = f"{text} (brain STOP sent)"
        text = f"{cmd['name']}: {text}"
    return wire.reply_message(ok, text)


def make_app(client: BrainClient) -> Starlette:
    async def index(_request):
        return FileResponse(WEB_DIR / "index.html")

    async def health(_request):
        """How often each stream arrived and how old the newest one is: the
        first place to look when a panel on the page stops moving."""
        _data, sys_age = client.latest_sysmon()
        _jpeg, cam_age = client.latest_jpeg()
        return JSONResponse({"seq": client.seq(),
                             "sys_age_s": None if sys_age == float("inf") else round(sys_age, 2),
                             "camera_age_s": None if cam_age == float("inf") else round(cam_age, 2)})

    async def sender(ws: WebSocket) -> None:
        sent: Dict[str, int] = {}
        loop = asyncio.get_running_loop()
        next_sys = 0.0
        while True:
            seq = client.seq()
            if seq["camera"] != sent.get("camera"):
                jpeg, _age = client.latest_jpeg()
                if jpeg is not None:
                    await ws.send_bytes(wire.pack_jpeg(wire.TAG_CAMERA, jpeg))
                sent["camera"] = seq["camera"]
            if seq["depth"] != sent.get("depth"):
                depth = client.latest_depth()
                if depth is not None:
                    await ws.send_bytes(wire.pack_depth(*depth))
                sent["depth"] = seq["depth"]
            # The picture goes out before the log, so it usually reaches the
            # page before its ask line. When the line wins anyway (it landed
            # between the two reads), the page still puts the picture under
            # it when it comes: they meet by number.
            if seq["annotated"] != sent.get("annotated"):
                pinned = client.latest_annotated()
                if pinned is not None:
                    picture, rgb = pinned
                    jpeg = await asyncio.to_thread(_jpeg_of, rgb)
                    await ws.send_bytes(wire.pack_annotated(picture, jpeg))
                sent["annotated"] = seq["annotated"]
            if seq["log"] != sent.get("log") or seq["nav"] != sent.get("nav"):
                snap = client.latest()
                await ws.send_text(json.dumps(wire.log_message(snap)))
                await ws.send_text(json.dumps(wire.nav_message(snap)))
                sent["log"], sent["nav"] = seq["log"], seq["nav"]
            if seq["guide"] != sent.get("guide"):
                guide = client.latest_guide()
                if guide is not None:
                    await ws.send_text(json.dumps(wire.guide_message(guide)))
                sent["guide"] = seq["guide"]
            if loop.time() >= next_sys:
                data, age = client.latest_sysmon()
                if data is not None:
                    await ws.send_text(json.dumps(wire.sys_message(data, age)))
                next_sys = loop.time() + SYS_EVERY_S
            await asyncio.sleep(SEND_TICK_S)

    async def answer(ws: WebSocket, cmd) -> None:
        reply = await run_command(client, cmd)
        try:
            await ws.send_text(json.dumps(reply))
        except (WebSocketDisconnect, RuntimeError):
            pass   # the page went away while the call ran; the call itself happened

    async def ws_endpoint(ws: WebSocket) -> None:
        origin, host = ws.headers.get("origin"), ws.headers.get("host")
        if not wire.origin_allowed(origin, host):
            # Another web page in the operator's browser, not ours: refuse
            # the handshake. Starlette turns a close before accept into an
            # HTTP 403.
            client.node.get_logger().warning(f"websocket refused: origin {origin!r} for host {host!r}")
            await ws.close(code=1008)
            return
        await ws.accept()
        send_task = asyncio.create_task(sender(ws))
        running = set()
        try:
            while True:
                cmd = wire.parse_command(await ws.receive_text())
                if cmd is None:
                    continue
                if wire.is_urgent(cmd):
                    # Inline and first: freeze() only publishes and starts
                    # its hold thread, it never blocks the loop. An Arm or a
                    # task still waiting on the robot does not delay it.
                    ok, text = client.freeze()
                    await ws.send_text(json.dumps(wire.reply_message(ok, text)))
                    continue
                task = asyncio.create_task(answer(ws, cmd))
                running.add(task)
                task.add_done_callback(running.discard)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            send_task.cancel()
            for task in running:
                task.cancel()

    return Starlette(routes=[
        Route("/", index),
        Route("/health", health),
        Mount("/web", StaticFiles(directory=WEB_DIR), name="web"),
        WebSocketRoute("/ws", ws_endpoint),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="The Wojtek VLM operator page (JS) and its ROS 2 node.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on; 127.0.0.1 keeps the page to this computer, "
                         "0.0.0.0 opens it to the network")
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args()
    client = start_client()
    uvicorn.run(make_app(client), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
