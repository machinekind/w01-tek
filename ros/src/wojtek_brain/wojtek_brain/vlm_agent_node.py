"""The Qwen agent as a ROS node: the W3 convergence piece.

Runs the demo's agent stack (WojtekAgent + GoalManager + VlmNavigator +
SearchController, imported from wojtek_rl unmodified) against a WorldProxy
instead of an in-process MuJoCo sim. Everything the robot will run lives on
this side of the wire; the world -- sim today, hardware later -- is reached
only through topics and one command service.

Wiring (see docs/plans/agentic-ros2.md):
  in : /wojtek/intent            RoutedIntent   nav/visual/chat are ours
                                                (Qwen authors, Bielik only
                                                translates); cancel aborts;
                                                system is ignored in v1
  in : /wojtek/audio/speech_started  Empty      barge-in: stop publishing the
                                                rest of the current reply
  in : /wojtek/exec/status       ExecStatus     pose + executor + resets
  in : /wojtek/map               WorldMap       agent-built occupancy grid
  in : /wojtek/camera/ego/compressed  CompressedImage  HUD-free ego view
  in : /wojtek/camera/vln/compressed  CompressedImage  square VLN-CE frame
  out: /wojtek/say_en            Sentence       English; Bielik renders Polish
  out: /wojtek/trace             String         trace events as JSON
  srv: /wojtek/world/command     WorldCommand   midlevel / goto / reset

Nav intents get NO reply text from here: Bielik already spoke the canned ack,
and the goal watcher announces the outcome when the behaviour ends -- a
second confirmation sentence would be spoken twice. Visual intents answer in
full.

Threading: rclpy spins on a daemon thread; the agent, its controllers and the
goal watcher live on the asyncio loop in main, the same split the audio
bridge uses. Blocking service calls from the agent thread cost single-digit
milliseconds on the measured 0.6 ms wire.
"""

from __future__ import annotations

import asyncio
import json
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Empty, String

from wojtek_agent_msgs.msg import ExecStatus, RoutedIntent, Sentence, WorldMap
from wojtek_agent_msgs.srv import WorldCommand

from wojtek_brain.sentences import speakable, split_sentences
from wojtek_brain.world_proxy import WorldProxy

CHAT_TURN_TIMEOUT_S = 90.0
SERVICE_TIMEOUT_S = 10.0
GOAL_POLL_S = 0.2


class VlmAgentNode(Node):
    def __init__(self):
        super().__init__("vlm_agent")
        self.declare_parameter("agent_url", "http://127.0.0.1:8090/v1")
        self.declare_parameter("agent_model", "")
        self.declare_parameter("vlm_backend", "futurenav")
        self.declare_parameter("vlm_url", "")
        self.declare_parameter("vlm_max_steps", 0)
        self.declare_parameter("trace_path", "")
        self.declare_parameter("forward_scale", 0.0)

        self.pub_say = self.create_publisher(Sentence, "/wojtek/say_en", 10)
        self.pub_trace = self.create_publisher(String, "/wojtek/trace", 10)
        self.cli_world = self.create_client(WorldCommand, "/wojtek/world/command")

        self.create_subscription(RoutedIntent, "/wojtek/intent", self.on_intent, 10)
        self.create_subscription(Empty, "/wojtek/audio/speech_started",
                                 self.on_speech_started, 10)
        self.create_subscription(ExecStatus, "/wojtek/exec/status",
                                 self.on_exec_status, qos_profile_sensor_data)
        self.create_subscription(WorldMap, "/wojtek/map", self.on_map, 10)
        self.create_subscription(CompressedImage, "/wojtek/camera/ego/compressed",
                                 self.on_ego, qos_profile_sensor_data)
        self.create_subscription(CompressedImage, "/wojtek/camera/vln/compressed",
                                 self.on_vln, qos_profile_sensor_data)

        self.loop: asyncio.AbstractEventLoop | None = None
        self.proxy = WorldProxy(self.world_command)
        self._chat_task: asyncio.Task | None = None
        self._last_utterance = ""
        self._say_seq = 0
        self._barge_gen = 0
        self._agent = None
        self._goals = None
        self._trace = None

    # -- world command service (blocking, called from the agent thread) -------

    def world_command(self, kind: str, text: str, args: list[float]) -> dict:
        req = WorldCommand.Request(kind=kind, text=text, args=args)
        if not self.cli_world.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
            return {"ok": False, "error": "world command service unavailable"}
        future = self.cli_world.call_async(req)
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(SERVICE_TIMEOUT_S):
            return {"ok": False, "error": "world command timed out"}
        res = future.result()
        out = {"ok": bool(res.ok), "cmd_seq": int(res.cmd_seq)}
        if res.error:
            out["error"] = res.error
        if res.command:
            out["command"] = res.command
        return out

    # -- subscriptions (rclpy spin thread) ------------------------------------

    def on_exec_status(self, msg: ExecStatus):
        self.proxy.on_status(msg.x, msg.y, msg.yaw, msg.active, msg.blocked,
                             msg.resets, msg.sim_time, msg.exec_json,
                             msg.cmd_seq)

    def on_map(self, msg: WorldMap):
        self.proxy.on_map(msg.res, tuple(msg.origin), tuple(msg.shape),
                          bytes(msg.state), list(msg.trail), msg.ego_fovy_deg)

    def on_ego(self, msg: CompressedImage):
        self.proxy.on_ego_jpeg(bytes(msg.data))

    def on_vln(self, msg: CompressedImage):
        self.proxy.on_vln_jpeg(bytes(msg.data))

    def on_speech_started(self, _msg: Empty):
        # The human is talking over us. Whatever reply is mid-publish is
        # stale; Bielik and TTS drop their queues on the same signal.
        self._barge_gen += 1

    def on_intent(self, msg: RoutedIntent):
        if self.loop is None:
            return
        if msg.intent in ("nav", "visual", "chat", "cancel"):
            self.loop.call_soon_threadsafe(self._dispatch, msg.intent,
                                           msg.text, msg.utterance_id)

    # -- agent side (asyncio thread) -------------------------------------------

    def _dispatch(self, intent: str, text: str, utterance_id: str):
        self._last_utterance = utterance_id
        if self._chat_task is not None and not self._chat_task.done():
            self._chat_task.cancel()
        if intent == "cancel":
            if self._goals is not None:
                self._goals.cancel("user")
            return
        self._chat_task = self.loop.create_task(
            self._chat_turn(intent, text, utterance_id))

    async def _chat_turn(self, intent: str, text: str, utterance_id: str):
        from wojtek_rl import perf

        if perf.current_turn() is None:
            perf.start_turn("text", chars=len(text))
        agent = self._build_agent()
        try:
            result = await asyncio.wait_for(agent.ask(text, voice=True),
                                            CHAT_TURN_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            self.get_logger().warning(
                f"chat turn timed out after {CHAT_TURN_TIMEOUT_S:g}s")
            return
        except Exception as e:
            self.get_logger().error(f"chat turn failed: {e}")
            return
        # Nav replies stay silent: Bielik's canned ack already played, and
        # the goal watcher announces how it ended.
        if intent != "nav" and result.get("ok"):
            self.publish_say(result.get("say", ""), utterance_id)

    def publish_say(self, text: str, utterance_id: str):
        text = speakable(text or "")
        if not text:
            return
        gen = self._barge_gen
        parts = split_sentences(text) or [text]
        for i, part in enumerate(parts):
            if self._barge_gen != gen:
                return  # user started talking; the rest is stale
            self._say_seq += 1
            msg = Sentence(utterance_id=utterance_id, seq=self._say_seq,
                           text=part, final=(i == len(parts) - 1), source="qwen")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_say.publish(msg)

    # -- goal outcome watcher ---------------------------------------------------

    async def watch_goals(self):
        """Announce terminal goal states; silence looks like a hung robot."""
        from wojtek_rl.agent.goals import TERMINAL_STATES, outcome_phrase

        rev, announced = -1, None
        # rclpy.ok() gate: SIGTERM lands in rclpy's signal handler, which
        # shuts the context down but kills no loop -- without this check the
        # process survives pkill and doubles on the next launch (seen live).
        while rclpy.ok():
            await asyncio.sleep(GOAL_POLL_S)
            goals = self._goals
            if goals is None or goals.rev == rev:
                continue
            rev = goals.rev
            status = goals.status()
            key = (status.get("kind"), status.get("state"), status.get("goal"))
            if status.get("state") in TERMINAL_STATES and key != announced:
                announced = key
                phrase = outcome_phrase(
                    status.get("kind") or "", status.get("state") or "",
                    status.get("goal") or "",
                    reason=(status.get("detail") or {}).get("reason"),
                    language="en",
                )
                if phrase:
                    self.get_logger().info(f"announcing outcome: {phrase!r}")
                    if self._trace is not None:
                        self._trace.add("goal.announce", text=phrase,
                                        goal_state=status.get("state"))
                    self.publish_say(phrase, self._last_utterance)

    # -- lazy construction (heavy imports on first intent, like room_app) ------

    def _build_agent(self):
        if self._agent is not None:
            return self._agent
        import os

        from wojtek_rl import perf
        from wojtek_rl.agent.chat import WojtekAgent
        from wojtek_rl.agent.goals import GoalManager
        from wojtek_rl.agent.llm import (
            DEFAULT_AGENT_MODEL,
            DEFAULT_AGENT_URL,
            AgentLLM,
        )
        from wojtek_rl.agent.nav import TracedVlmNavigator
        from wojtek_rl.agent.search import SearchController, make_score_view
        from wojtek_rl.agent.tools import build_tools
        from wojtek_rl.agent.trace import Trace

        p = lambda name: self.get_parameter(name).value
        if float(p("forward_scale") or 0) > 0:
            os.environ["WOJTEK_NAV_FORWARD_SCALE"] = str(p("forward_scale"))

        trace = Trace(p("trace_path") or None)
        perf.bind(trace)
        self._trace = trace

        def on_trace(event: dict):
            msg = String(data=json.dumps(event, ensure_ascii=False))
            self.pub_trace.publish(msg)
            # Speak the goal switch immediately, before the model's reply:
            # the robot is already turning away from the old target.
            if event.get("kind") == "goal.switch" and event.get("text"):
                self.loop.call_soon_threadsafe(
                    self.publish_say, event["text"], self._last_utterance)

        trace.subscribe(on_trace)

        llm = AgentLLM(base_url=p("agent_url") or DEFAULT_AGENT_URL,
                       model=p("agent_model") or DEFAULT_AGENT_MODEL)

        def navigator_factory():
            backend = p("vlm_backend")
            if backend == "futurenav":
                from wojtek_rl.futurenav_nav import (
                    DEFAULT_FUTURENAV_URL,
                    FUTURENAV_MAX_ROTATION,
                    FUTURENAV_MAX_STEPS,
                    FUTURENAV_TIMEOUT_S,
                    FutureNavVlmClient,
                )
                client = FutureNavVlmClient(p("vlm_url") or DEFAULT_FUTURENAV_URL)
                return TracedVlmNavigator(
                    self.proxy, client, max_steps=FUTURENAV_MAX_STEPS,
                    vlm_timeout_s=FUTURENAV_TIMEOUT_S, overlap=True,
                    vlnce_frame=True, max_rotation=FUTURENAV_MAX_ROTATION,
                    trace=trace,
                )
            # Any OpenAI-compatible server; defaults to the chat agent's own
            # endpoint so one served model drives navigation too.
            from wojtek_eval.vlm_openai import OpenAIVlmClient
            from wojtek_rl.agent.nav import (
                INSTRUCTION_PROMPT,
                NAV_MAX_ROTATION,
                NAV_MAX_STEPS,
            )
            from wojtek_rl.vlm_nav import ACTIONS
            client = OpenAIVlmClient(
                base_url=p("vlm_url") or p("agent_url") or DEFAULT_AGENT_URL,
                model=p("agent_model") or DEFAULT_AGENT_MODEL,
                system_prompt=INSTRUCTION_PROMPT, actions=ACTIONS,
            )
            return TracedVlmNavigator(
                self.proxy, client,
                max_steps=int(p("vlm_max_steps") or 0) or NAV_MAX_STEPS,
                max_rotation=NAV_MAX_ROTATION, trace=trace,
            )

        score_view = make_score_view(llm)

        def search_factory():
            return SearchController(
                self.proxy, score_view, hfov_deg=self.proxy._ego_fovy,
                frame_fn=lambda: self.proxy.ego_jpeg(hud=False), trace=trace,
            )

        self._goals = GoalManager(navigator_factory=navigator_factory,
                                  search_factory=search_factory,
                                  trace=trace, language="en")

        turn_context: dict = {}

        async def visibility_check(target: str):
            vs = await score_view(target, self.proxy.ego_jpeg(hud=False))
            return vs.visible

        tools = build_tools(self.proxy, self._goals, self.proxy.pose_history,
                            turn_context=turn_context,
                            visibility_check=visibility_check)
        # English out: Bielik owns the Polish, exactly the /wojtek/say_en
        # contract the demo's speak_llm mirrored in-process.
        self._agent = WojtekAgent(llm, tools, trace=trace,
                                  turn_context=turn_context,
                                  lang_mode="direct", reply_language="en")
        self.get_logger().info("agent built: reply_language=en via Bielik")
        return self._agent

    async def run(self):
        self.loop = asyncio.get_running_loop()
        await self.watch_goals()


def main():
    rclpy.init()
    node = VlmAgentNode()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        asyncio.run(node.run())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
