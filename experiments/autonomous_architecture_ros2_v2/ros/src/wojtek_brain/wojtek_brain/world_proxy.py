"""Duck-typed stand-in for the demo's RoomSim, fed by ROS topics.

The whole agent stack (WojtekAgent tools, GoalManager, VlmNavigator,
SearchController) touches the world through nine members: pose(),
submit_command(), executor.active/.blocked/.goto()/.status(), ego_jpeg(),
vlm_frame_jpeg(), omap, resets (plus sim_time and pose_history). This module
implements exactly that surface over message caches, so the agent code runs
unmodified whether the world is an in-process MuJoCo sim or a robot on the
other end of a wire.

rclpy-free on purpose: the node layer (vlm_agent_node) owns subscriptions and
the service client, and feeds this object plain Python values. That keeps the
protocol logic testable without a ROS graph, in the house unit-test style.

Threading contract: on_*() setters are called from the rclpy spin thread;
readers run on the agent's asyncio thread. Every cross-thread value is either
swapped atomically (attribute rebind) or immutable once published.
"""

from __future__ import annotations

import json
import threading
from typing import Callable

# kind, text, args -> ack dict. The node layer implements this as a blocking
# WorldCommand service call; tests hand in a lambda.
CommandFn = Callable[[str, str, list[float]], dict]

FRAME_WAIT_S = 5.0  # first-frame grace at startup; afterwards reads are instant


class ExecutorView:
    """The executor the controllers poll, reconstructed from ExecStatus.

    `active` is sequence-fenced: an accepted command returns a cmd_seq, and
    until a status snapshot stamped with that seq (or later) arrives, the
    command counts as running no matter what the snapshot says. Without the
    fence a controller polls a pre-command snapshot, concludes the command
    already finished, and machine-guns the next one 3 ms later (seen live,
    2026-08-26: the robot 'searched' a flat while moving 6 cm).
    """

    def __init__(self, command_fn: CommandFn):
        self._command_fn = command_fn
        self._raw_active = False
        self._status_seq = 0
        self._acked_seq = 0
        self.blocked = 0
        self._status_json = "{}"

    @property
    def active(self) -> bool:
        return self._raw_active or self._status_seq < self._acked_seq

    def note_ack(self, ack: dict) -> None:
        seq = int(ack.get("cmd_seq") or 0)
        if ack.get("ok") and seq:
            self._acked_seq = max(self._acked_seq, seq)

    def status(self) -> dict:
        try:
            return json.loads(self._status_json)
        except (TypeError, ValueError):
            return {}

    def goto(self, x: float, y: float, from_xy: tuple[float, float]) -> None:
        # from_xy is ignored on the wire: the world knows its own pose better
        # than a snapshot the agent took a status message ago.
        self.note_ack(self._command_fn("goto", "", [float(x), float(y)]))


class WorldProxy:
    """RoomSim look-alike over topic caches and one command service."""

    def __init__(self, command_fn: CommandFn, pose_history=None,
                 map_factory=None):
        self.executor = ExecutorView(command_fn)
        self._command_fn = command_fn
        self.resets = 0
        self.sim_time = 0.0
        self._pose = (0.0, 0.0, 0.0)
        self._ego_fovy = 100.0  # overwritten by the first WorldMap message
        self.omap = None
        self._map_factory = map_factory  # tests inject a fake OnlineMap class
        if pose_history is None:
            from wojtek_rl.agent.spatial import PoseHistory
            pose_history = PoseHistory()
        self.pose_history = pose_history
        self._ego_b64: str | None = None
        self._vln_b64: str | None = None
        self._ego_ready = threading.Event()
        self._vln_ready = threading.Event()

    # -- reads used by tools and controllers ---------------------------------

    def pose(self) -> tuple[float, float, float]:
        return self._pose

    def submit_command(self, text: str) -> dict:
        ack = self._command_fn("midlevel", text, [])
        self.executor.note_ack(ack)
        return ack

    def ego_jpeg(self, hud: bool = True) -> str:
        # The HUD flag is accepted for drop-in compatibility but every frame
        # on the wire is HUD-free: VLM-facing frames must not carry the
        # minimap (agentic-ros2.md, simulator strategy).
        if not self._ego_ready.wait(FRAME_WAIT_S):
            raise RuntimeError("no ego camera frame received yet")
        return self._ego_b64

    def vlm_frame_jpeg(self) -> str:
        if not self._vln_ready.wait(FRAME_WAIT_S):
            raise RuntimeError("no VLN camera frame received yet")
        return self._vln_b64

    # -- feeds from the node layer (rclpy spin thread) ------------------------

    def on_status(self, x: float, y: float, yaw: float, active: bool,
                  blocked: int, resets: int, sim_time: float,
                  exec_json: str, cmd_seq: int = 0) -> None:
        self._pose = (float(x), float(y), float(yaw))
        self.executor._raw_active = bool(active)
        self.executor._status_seq = max(self.executor._status_seq, int(cmd_seq))
        self.executor.blocked = int(blocked)
        self.executor._status_json = exec_json
        self.resets = int(resets)
        self.sim_time = float(sim_time)
        self.pose_history.add(float(sim_time), float(x), float(y), float(yaw))

    def on_map(self, res: float, origin: tuple[float, float],
               shape: tuple[int, int], state: bytes, trail: list[float],
               ego_fovy_deg: float) -> None:
        import numpy as np

        if ego_fovy_deg > 0:
            self._ego_fovy = float(ego_fovy_deg)
        if self.omap is None or self.omap.shape != tuple(shape):
            factory = self._map_factory
            if factory is None:
                from wojtek_eval.mapping import OnlineMap
                factory = OnlineMap
            self.omap = factory(res=float(res),
                                origin=(float(origin[0]), float(origin[1])),
                                shape=(int(shape[0]), int(shape[1])))
        grid = np.frombuffer(bytes(state), dtype=np.uint8)
        if grid.size == int(shape[0]) * int(shape[1]):
            # Rebind, don't mutate in place: readers on the agent thread then
            # always see one consistent generation of the grid.
            self.omap.state = grid.reshape(int(shape[0]), int(shape[1])).copy()
        pairs = list(zip(trail[0::2], trail[1::2]))
        self.omap.trail = [(float(x), float(y)) for x, y in pairs]

    def on_ego_jpeg(self, jpeg: bytes) -> None:
        import base64
        self._ego_b64 = base64.b64encode(bytes(jpeg)).decode("ascii")
        self._ego_ready.set()

    def on_vln_jpeg(self, jpeg: bytes) -> None:
        import base64
        self._vln_b64 = base64.b64encode(bytes(jpeg)).decode("ascii")
        self._vln_ready.set()
