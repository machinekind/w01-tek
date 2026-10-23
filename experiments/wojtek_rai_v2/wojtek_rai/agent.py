"""Assemble the RAI ReAct agent for Wojtek."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain_core.language_models import BaseChatModel
from rai import get_llm_model
from rai.initialization.model_initialization import get_llm_model_config_and_vendor
from rai.agents.langchain import ReActAgent
from rai.communication.ros2 import ROS2Connector
from rai_whoami import EmbodimentInfo

from wojtek_rai.tools import build_tools

EMBODIMENT_PATH = Path(__file__).with_name("embodiment.json")

# WOJTEK_RAI_READONLY=1: no walk/stop/stand_up/lie_down tools at all. Use it
# for the first sessions against the physical robot.
READ_ONLY = os.environ.get("WOJTEK_RAI_READONLY", "").strip() in ("1", "true", "yes")


READ_ONLY_RULE = (
    "Read-only session: you have no movement tools at all. Observe with the "
    "camera, report, and tell the user plainly when asked to move that you cannot."
)


def load_embodiment() -> EmbodimentInfo:
    info = EmbodimentInfo.from_file(EMBODIMENT_PATH)
    if READ_ONLY:
        # to_langchain() returns a message object, so the rule is added to the
        # embodiment itself rather than concatenated onto the prompt.
        info = info.model_copy(update={"rules": [*(info.rules or []), READ_ONLY_RULE]})
    return info


# Ollama context window for the agent. Ollama's default on a big GPU is the
# model's full window (262k for qwen3-vl), which reserves tens of GB of KV
# cache; a ReAct turn with a few images fits comfortably in 32k.
OLLAMA_NUM_CTX = 32768


def _llm_kwargs() -> dict:
    """Vendor-specific model options. Ollama: no chain-of-thought and a
    bounded context. reasoning=False only works on non-thinking builds
    (config.toml uses the *-instruct tags): Ollama 0.34 ignores it for the
    thinking tags, and a 1-4k token hidden preamble per step made every tool
    call 8-70 s slower without helping tool discipline."""
    _, vendor = get_llm_model_config_and_vendor("complex_model")
    if vendor == "ollama":
        return {"reasoning": False, "num_ctx": OLLAMA_NUM_CTX}
    return {}


def build_llm() -> BaseChatModel:
    return get_llm_model("complex_model", streaming=True, **_llm_kwargs())


def build_agent(connector: ROS2Connector, llm: Optional[BaseChatModel] = None) -> ReActAgent:
    """Direct-mode agent (no HRI connectors): the caller drives it, e.g. streamlit."""
    return ReActAgent(
        target_connectors={},
        llm=llm or build_llm(),
        system_prompt=load_embodiment().to_langchain(),
        tools=build_tools(connector, read_only=READ_ONLY),
    )
