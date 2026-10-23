"""One-shot terminal chat with the agent, for scripted checks and logs.

    ./experiments/wojtek_rai_v2/run.sh chat "stand up and walk forward for 2 seconds"
    ./experiments/wojtek_rai_v2/run.sh chat --debug "..."   # every LLM call: prompt + raw reply

Prints every tool call with its arguments and result, then the final answer.
With --debug, LangChain's global debug mode dumps each chain/LLM/tool
invocation (system prompt included) to stderr -- the place to look when the
model answers in prose instead of calling a tool.
"""

from __future__ import annotations

import sys

import rclpy
from langchain_core.messages import HumanMessage
from rai.communication.ros2 import ROS2Connector

from wojtek_rai.agent import build_agent
from wojtek_rai.stream import TurnEvents, run_turn, text_of


def main(argv: list[str]) -> int:
    debug = "--debug" in argv
    argv = [a for a in argv if a != "--debug"]
    if not argv:
        print("usage: python -m wojtek_rai.chat [--debug] <instruction>", file=sys.stderr)
        return 2
    if debug:
        from langchain_core.globals import set_debug

        set_debug(True)
    prompt = " ".join(argv)
    rclpy.init()
    connector = ROS2Connector(node_name="wojtek_rai_chat", executor_type="multi_threaded")
    try:
        graph = build_agent(connector).agent
        events = TurnEvents(
            on_tool_calls=lambda calls: [
                print(f"[tool call] {c['name']}({c['args']})", flush=True) for c in calls
            ],
            on_tool_result=lambda m: print(f"[tool result] {text_of(m.content)[:300]}", flush=True),
        )
        result = run_turn(graph, [HumanMessage(content=prompt)], events)
        for t in result.assistant_texts:
            print(f"[assistant] {t[:600]}")
        print(f"\n[answer] {result.final_text}")
        return 0
    finally:
        connector.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
