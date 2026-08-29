"""Wojtek's conversational agent: chat + tools + goal state machine + search.

The layer above VlmNavigator/ScanExecutor: a small instruct VLM (Qwen3-VL-4B
by default) receives user text, decides between answering directly and
calling one of a fixed set of tools (camera view, occupancy map, route
history, navigate, search, stop), and replies in character as a happy robot
dog. Long-running behaviours (navigate via FutureNav, object search via the
frontier/value-map controller) run as background asyncio tasks owned by
GoalManager; the chat turn only starts/inspects them.

Module map:
  llm.py      OpenAI-compatible chat client (vLLM etc.), multi-image messages
  parsing.py  tolerant reply parsing (thought-first JSON contract)
  spatial.py  pose history + occupancy-map text summaries + frontier clusters
  search.py   VLFM-style object search: value map + verify loop (FSM)
  goals.py    the goal state machine (idle / navigating / searching)
  tools.py    tool registry binding sim + goals to the chat loop
  chat.py     WojtekAgent: persona prompt, tool loop, context management

Stage timings live one level up in `wojtek_rl.perf` (stdlib-only, so any
module here or in wojtek_eval can time itself without an import cycle); they
are written into this package's session trace and read by
`wojtek_rl.perf_report`.
"""

from wojtek_rl.agent.chat import WojtekAgent
from wojtek_rl.agent.goals import GoalManager
from wojtek_rl.agent.llm import AgentLLM
from wojtek_rl.agent.search import SearchController
from wojtek_rl.agent.spatial import PoseHistory

__all__ = ["AgentLLM", "GoalManager", "PoseHistory", "SearchController", "WojtekAgent"]
