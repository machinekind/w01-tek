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

The re-exports below are resolved lazily (PEP 562).  Importing them eagerly
makes this package unimportable from inside the cycle it participates in:
`wojtek_rl.vlm_nav` loads a prompt from `wojtek_agent.prompts`, while
`wojtek_agent.chat` imports `wojtek_rl.vlm_nav` — so an eager `__init__`
resolves only when something happens to import this package first.
"""

import importlib
from typing import TYPE_CHECKING

_LAZY = {
    "WojtekAgent": "chat",
    "GoalManager": "goals",
    "AgentLLM": "llm",
    "SearchController": "search",
    "PoseHistory": "spatial",
}

__all__ = ["AgentLLM", "GoalManager", "PoseHistory", "SearchController", "WojtekAgent"]

if TYPE_CHECKING:  # keep the names resolvable for type checkers and IDEs
    from wojtek_agent.chat import WojtekAgent
    from wojtek_agent.goals import GoalManager
    from wojtek_agent.llm import AgentLLM
    from wojtek_agent.search import SearchController
    from wojtek_agent.spatial import PoseHistory


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"wojtek_agent.{module}"), name)


def __dir__():
    return sorted(__all__)
