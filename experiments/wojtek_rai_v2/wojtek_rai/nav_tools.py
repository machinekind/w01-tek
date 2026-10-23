"""Navigation tools: RAI's Nav2 tools plus a named-place goal on top of them.

Requires the wojtek_nav container (`run.sh nav launch`): action
`navigate_to_pose`, TF map->base_link, /map. Without it the tools return an
error string the agent can read, and `walk` still works.

The Nav2 goal itself goes through `_TimedNavMixin._navigate`, not RAI's
`NavigateToPoseBlockingTool._run`: that one calls `ActionClient.send_goal`
with no timeout, so a goal stuck in recovery or a result lost in a WiFi flap
blocks the agent forever. Here a goal is bounded by `limits.NAV_GOAL_TIMEOUT_S`
and cancelled on expiry, the handle in flight is kept so `stop` can cancel it,
and the workspace-bounds check is done here too (rai-core 2.12 ignores the
`workspace_bounds_*` kwargs: `BaseROS2Tool` is `extra="ignore"`).
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Type

import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from langchain_core.tools import BaseTool
from nav2_msgs.action import NavigateToPose
from pydantic import BaseModel, Field
from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2.base import BaseROS2Tool
from rai.tools.ros2.navigation.nav2 import GetOccupancyGridTool
from rai.tools.ros2.navigation.nav2_blocking import (
    GetCurrentPoseTool,
    NavigateToPoseBlockingTool,
    _get_error_message,
)
from rclpy.action import ActionClient
from tf_transformations import quaternion_from_euler

from wojtek_rai import limits

# How long to wait for the action server and for its goal response. Belongs in
# limits.py next to NAV_GOAL_TIMEOUT_S; picked up from there once it lands.
NAV_SERVER_TIMEOUT_S: float = getattr(limits, "NAV_SERVER_TIMEOUT_S", 5.0)

# The goal handle of the Nav2 goal in flight (one agent, one goal at a time),
# so `stop` can cancel it. Cleared when `_navigate` returns.
_ACTIVE: Dict[str, Any] = {"handle": None}


def cancel_active_nav_goal() -> bool:
    """Cancel the Nav2 goal in flight, if any. Returns whether one was cancelled."""
    handle = _ACTIVE.get("handle")
    if handle is None:
        return False
    try:
        handle.cancel_goal_async()
    except Exception:  # noqa: BLE001 -- best effort; the caller stops the robot anyway
        return False
    return True


def _wait(future: Any, timeout_s: float) -> bool:
    """Block until `future` is done (the connector's executor spins it) or the timeout."""
    done = threading.Event()
    future.add_done_callback(lambda _f: done.set())
    return done.wait(timeout_s)


def _outside_workspace(x: float, y: float) -> bool:
    lo, hi = limits.WORKSPACE_MIN, limits.WORKSPACE_MAX
    return not (lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1])


class _TimedNavMixin:
    """Send one NavigateToPose goal and block on it for at most NAV_GOAL_TIMEOUT_S.

    Plain class (no pydantic fields), mixed in FIRST before
    NavigateToPoseBlockingTool: it overrides `_run(x, y, z, yaw)`, so a
    subclass with its own `_run(place|object_name)` reaches the guard and the
    deadline either through `self._navigate(...)` or through
    `super()._run(x=..., y=..., z=..., yaw=...)`, never RAI's unbounded
    `send_goal`. Every failure comes back as a string the agent can read,
    matching the parent's contract.
    """

    connector: ROS2Connector
    frame_id: str
    action_name: str

    def _run(self, x: float, y: float, z: float, yaw: float) -> str:  # type: ignore[override]
        return self._navigate(x, y, yaw)

    def _goal(self, x: float, y: float, yaw: float) -> Any:
        # Same PoseStamped as NavigateToPoseBlockingTool._run builds.
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.connector.node.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        quat = quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        goal = NavigateToPose.Goal()
        goal.pose = pose
        return goal

    def _navigate(self, x: float, y: float, yaw: float) -> str:
        # RAI's tool never checks its own permission lists for the action.
        if not self.is_writable(self.action_name):  # type: ignore[attr-defined]
            return f"action {self.action_name!r} is not writable; refused"
        if _outside_workspace(x, y):
            return (
                f"goal ({x:.2f}, {y:.2f}) is outside the workspace "
                f"{limits.WORKSPACE_MIN[:2]}..{limits.WORKSPACE_MAX[:2]}; refused"
            )
        try:
            return self._send_and_wait(x, y, yaw)
        except Exception as e:  # noqa: BLE001 -- the agent reads the error string
            return f"Navigate to pose action failed with exception: {type(e).__name__}: {e}"
        finally:
            _ACTIVE["handle"] = None

    def _send_and_wait(self, x: float, y: float, yaw: float) -> str:
        client = ActionClient(self.connector.node, NavigateToPose, self.action_name)
        if not client.wait_for_server(timeout_sec=NAV_SERVER_TIMEOUT_S):
            return (
                f"{self.action_name} action server not available "
                "(is the wojtek_nav container running?)"
            )
        goal_future = client.send_goal_async(self._goal(x, y, yaw))
        if not _wait(goal_future, NAV_SERVER_TIMEOUT_S):
            return f"no goal response from {self.action_name} within {NAV_SERVER_TIMEOUT_S:.0f} s"
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return "goal rejected by Nav2"
        _ACTIVE["handle"] = goal_handle
        result_future = goal_handle.get_result_async()
        if not _wait(result_future, limits.NAV_GOAL_TIMEOUT_S):
            goal_handle.cancel_goal_async()
            return (
                f"navigation timed out after {limits.NAV_GOAL_TIMEOUT_S:.0f} s; "
                "goal cancelled, the robot is stopping"
            )
        response = result_future.result()
        if response is None:
            return "Navigate to pose action failed. Please try again."
        if response.status != GoalStatus.STATUS_SUCCEEDED:
            return f"Navigate to pose action failed. {_get_error_message(response)}"
        return "Navigate to pose successful."


class GetMapPoseTool(GetCurrentPoseTool):
    name: str = "get_map_pose"
    description: str = (
        "The robot's current position (x, y in metres) and heading (yaw) in the map "
        "frame. Use it before planning a navigation goal and when asked where you are."
    )


class NavigateToPoseTool(_TimedNavMixin, NavigateToPoseBlockingTool):
    name: str = "navigate_to_pose"
    description: str = (
        "Navigate autonomously to a goal (x, y in metres, yaw in radians) in the map "
        "frame, avoiding obstacles. Goals must lie within x, y in "
        f"[{limits.WORKSPACE_MIN[0]:g}, {limits.WORKSPACE_MAX[0]:g}] m. Blocks until arrival, "
        f"failure, or {limits.NAV_GOAL_TIMEOUT_S:.0f} s, after which the goal is cancelled. "
        "Prefer this over 'walk' for anything farther than one metre."
    )


class GetMapImageTool(GetOccupancyGridTool):
    name: str = "get_map_image"
    description: str = (
        "The current map as an image: white free, black obstacles, grey unknown, "
        "the robot as a red dot. Look at it to decide where to go or why a goal failed."
    )


class CancelNavigationTool(BaseROS2Tool):
    name: str = "cancel_navigation"
    description: str = (
        "Cancel the Nav2 goal in progress, if any; Nav2 then stops the robot. "
        "Use it when the user says stop while a navigation goal is active."
    )

    def _run(self) -> str:
        if cancel_active_nav_goal():
            return "Navigation goal cancelled."
        return "No navigation goal in progress."


NO_REGISTRY = ("", "none")


def load_places(path: str | Path = limits.PLACES_FILE) -> Dict[str, Dict[str, float]]:
    """Named goals from the YAML registry, standoff applied: {name: {x, y, yaw}}.
    WOJTEK_RAI_PLACES=none (or empty) means no registry at all: the physical
    robot has no map yet, so there is nothing to name."""
    if str(path).strip().lower() in NO_REGISTRY:
        return {}
    raw = yaml.safe_load(Path(path).read_text())
    default_standoff = float(raw.get("standoff", 0.0))
    out: Dict[str, Dict[str, float]] = {}
    for name, p in (raw.get("places") or {}).items():
        x, y = float(p["x"]), float(p["y"])
        standoff = float(p.get("standoff", default_standoff))
        dist = math.hypot(x, y)
        if standoff > 0 and dist > standoff:
            # Pull the goal back towards the origin so the robot stops in front.
            x, y = x * (dist - standoff) / dist, y * (dist - standoff) / dist
        yaw = float(p["yaw"]) if "yaw" in p else math.atan2(float(p["y"]) - y, float(p["x"]) - x)
        out[name] = {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 3)}
    return out


class GoToPlaceInput(BaseModel):
    place: str = Field(..., description="Name of a known place, e.g. 'hydrant' or 'home'.")


class GoToPlaceTool(_TimedNavMixin, NavigateToPoseBlockingTool):
    """Same Nav2 action as navigate_to_pose, goal resolved from the registry."""

    name: str = "go_to_place"
    description: str = "Navigate to a known named place. Known places: {places}."
    args_schema: Type[GoToPlaceInput] = GoToPlaceInput
    places: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.description = self.description.format(places=", ".join(sorted(self.places)) or "none")

    def _run(self, place: str) -> str:  # type: ignore[override]
        key = place.strip().lower().replace(" ", "_")
        if key not in self.places:
            return f"Unknown place {place!r}. Known: {', '.join(sorted(self.places))}."
        g = self.places[key]
        return self._navigate(g["x"], g["y"], g["yaw"])


def build_nav_tools(connector: ROS2Connector, perms: Dict[str, List[str]]) -> List[BaseTool]:
    places = load_places(limits.PLACES_FILE)
    # No workspace_bounds_* kwargs: rai-core 2.12 has no such fields and
    # silently dropped them; the box is enforced in _TimedNavMixin._navigate.
    tools: List[BaseTool] = [
        GetMapPoseTool(
            connector=connector, frame_id=limits.MAP_FRAME, robot_frame_id=limits.BASE_FRAME, **perms
        ),
        NavigateToPoseTool(
            connector=connector, frame_id=limits.MAP_FRAME, action_name=limits.NAV_ACTION, **perms
        ),
        CancelNavigationTool(connector=connector, **perms),
        GetMapImageTool(connector=connector, **perms),
    ]
    if places:
        # Without a registry the tool would only ever answer "unknown place";
        # better not to offer it than to have the LLM guess names.
        tools.insert(2, GoToPlaceTool(
            connector=connector, frame_id=limits.MAP_FRAME, action_name=limits.NAV_ACTION,
            places=places, **perms,
        ))
    return tools
