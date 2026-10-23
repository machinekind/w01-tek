"""Benchmark one ReAct step of the agent's LLM, reproducibly.

    ./experiments/wojtek_rai_v2/run.sh bench                       # baseline: agent LLM, 2 images, tools
    ./experiments/wojtek_rai_v2/run.sh bench --model qwen3-vl:8b
    ./experiments/wojtek_rai_v2/run.sh bench --images 0 --no-tools
    ./experiments/wojtek_rai_v2/run.sh bench --history-steps 4 --runs 3 --json out.json

Builds the LLM exactly as the agent does (wojtek_rai.agent.build_llm), binds
the same tool schemas (wojtek_rai.tools.build_tools on a ROS2Connector), uses
the embodiment system prompt, and sends a synthetic history shaped like a
real turn: user instruction, AI tool call, tool result, N camera frames as
base64 image blocks in HumanMessages, a final user question. The response is
streamed so time-to-first-token is measured; prompt/generation token counts
and durations come from Ollama's fields in the AIMessage response_metadata.

The frames are real (/camera/camera/color/image_raw) when the simulation is
up, else a generated 640x360 PNG -- either way the same bytes are reused for
every image and every run, so runs are comparable.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, List, Optional

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

INSTRUCTION = "Look around and tell me whether the path ahead of you is clear."
QUESTION = "Based on what you saw, what is directly in front of you? One sentence."
CAMERA_TOOL = "get_ros2_image"
FRAME_WAIT_S = 3.0
GENERATED_W, GENERATED_H = 640, 360


# ---------------------------------------------------------------- images


def _generated_png() -> bytes:
    """A 640x360 gradient with a few shapes: not a camera frame, but as many
    pixels (and therefore vision tokens) as one."""
    import numpy as np
    from PIL import Image, ImageDraw

    y, x = np.mgrid[0:GENERATED_H, 0:GENERATED_W]
    rgb = np.stack(
        [(x * 255 // GENERATED_W), (y * 255 // GENERATED_H), np.full_like(x, 96)], axis=2
    ).astype(np.uint8)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle((80, 120, 260, 300), fill=(200, 40, 40))
    draw.ellipse((380, 60, 560, 240), fill=(40, 200, 80))
    draw.line((0, 330, GENERATED_W, 330), fill=(255, 255, 255), width=4)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _camera_png(wait_s: float) -> Optional[bytes]:
    """Newest colour frame from the simulation, or None if none arrives.
    Spins a throwaway node on this thread (camera_feed's daemon-thread spinner
    cannot be joined cleanly before rclpy shuts down)."""
    import rclpy
    from PIL import Image
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image as ImageMsg

    from wojtek_rai import limits
    from wojtek_rai.camera_feed import _to_rgb

    got: list = []
    node = rclpy.create_node("wojtek_rai_bench_camera")
    node.create_subscription(ImageMsg, limits.COLOR_IMAGE_TOPIC, got.append, qos_profile_sensor_data)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        deadline = time.monotonic() + wait_s
        while not got and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if not got:
            return None
        buf = io.BytesIO()
        Image.fromarray(_to_rgb(got[-1])).save(buf, format="PNG")
        return buf.getvalue()
    finally:
        executor.remove_node(node)
        node.destroy_node()


def load_frame(source: str) -> tuple[bytes, str]:
    """(png bytes, where they came from). source: auto | camera | generated."""
    if source in ("auto", "camera"):
        png = _camera_png(FRAME_WAIT_S)
        if png is not None:
            return png, "camera"
        if source == "camera":
            raise SystemExit("no frame on the colour topic; is the simulation up?")
    return _generated_png(), "generated"


def image_message(png: bytes, index: int) -> HumanMessage:
    b64 = base64.b64encode(png).decode("ascii")
    return HumanMessage(
        content=[
            {"type": "text", "text": f"Camera frame {index + 1}:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    )


# ---------------------------------------------------------------- history


def build_history(png: bytes, images: int, history_steps: int, nonce: str = "") -> List[BaseMessage]:
    """User instruction, then `history_steps` (AI tool call + tool result)
    pairs, then `images` camera frames, then the closing question."""
    first = f"[session {nonce}] {INSTRUCTION}" if nonce else INSTRUCTION
    msgs: List[BaseMessage] = [HumanMessage(content=first)]
    for step in range(history_steps):
        call_id = f"call_{step:03d}"
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": CAMERA_TOOL, "args": {}, "type": "tool_call"}],
            )
        )
        msgs.append(
            ToolMessage(
                content="Image received successfully; it is attached below.",
                tool_call_id=call_id,
                name=CAMERA_TOOL,
            )
        )
    for i in range(images):
        msgs.append(image_message(png, i))
    msgs.append(HumanMessage(content=QUESTION))
    return msgs


# ---------------------------------------------------------------- one run


@dataclass
class RunStats:
    run: int
    ttft_s: float
    total_s: float
    prompt_tokens: Optional[int]
    gen_tokens: Optional[int]
    prompt_eval_s: Optional[float]
    eval_s: Optional[float]
    load_s: Optional[float]
    server_s: Optional[float]  # Ollama total_duration: client total minus this is queueing + transport
    prompt_tok_s: Optional[float]
    gen_tok_s: Optional[float]
    chunks: int
    tool_calls: int
    reasoning_chars: int
    hidden_chunks: int  # chunks with no content/reasoning/tool-call: thinking Ollama emits but langchain-ollama drops when reasoning=False
    text: str


def _ns(meta: dict, key: str) -> Optional[float]:
    v = meta.get(key)
    return None if v is None else v / 1e9


def _rate(tokens: Optional[int], seconds: Optional[float]) -> Optional[float]:
    if tokens is None or not seconds:
        return None
    return tokens / seconds


def one_run(llm, messages: List[BaseMessage], run: int) -> RunStats:
    t0 = time.perf_counter()
    first: Optional[float] = None
    acc: Optional[AIMessageChunk] = None
    chunks = 0
    hidden = 0
    for chunk in llm.stream(messages):
        chunks += 1
        if first is None and (chunk.content or chunk.tool_call_chunks):
            first = time.perf_counter()
        if not (chunk.content or chunk.tool_call_chunks
                or (chunk.additional_kwargs or {}).get("reasoning_content")):
            hidden += 1
        acc = chunk if acc is None else acc + chunk
    t1 = time.perf_counter()
    if first is None:
        first = t1
    meta: dict = dict(getattr(acc, "response_metadata", None) or {})
    usage = getattr(acc, "usage_metadata", None) or {}
    prompt_tokens = meta.get("prompt_eval_count", usage.get("input_tokens"))
    gen_tokens = meta.get("eval_count", usage.get("output_tokens"))
    prompt_eval_s = _ns(meta, "prompt_eval_duration")
    eval_s = _ns(meta, "eval_duration")
    text = acc.content if acc is not None else ""
    if not isinstance(text, str):
        text = json.dumps(text)[:200]
    tool_calls = list(getattr(acc, "tool_calls", None) or [])
    reasoning = (getattr(acc, "additional_kwargs", None) or {}).get("reasoning_content") or ""
    return RunStats(
        run=run,
        ttft_s=round(first - t0, 3),
        total_s=round(t1 - t0, 3),
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        prompt_eval_s=None if prompt_eval_s is None else round(prompt_eval_s, 3),
        eval_s=None if eval_s is None else round(eval_s, 3),
        load_s=None if _ns(meta, "load_duration") is None else round(_ns(meta, "load_duration"), 3),
        server_s=None if _ns(meta, "total_duration") is None else round(_ns(meta, "total_duration"), 3),
        prompt_tok_s=None if _rate(prompt_tokens, prompt_eval_s) is None else round(_rate(prompt_tokens, prompt_eval_s), 1),
        gen_tok_s=None if _rate(gen_tokens, eval_s) is None else round(_rate(gen_tokens, eval_s), 1),
        chunks=chunks,
        tool_calls=len(tool_calls),
        reasoning_chars=len(reasoning),
        hidden_chunks=hidden,
        text=(text[:160] + ("..." if len(text) > 160 else "")) if text else
             (f"[tool calls: {[c['name'] for c in tool_calls]}]" if tool_calls else ""),
    )


# ---------------------------------------------------------------- main


def _mean(values: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in values if v is not None]
    return round(statistics.fmean(xs), 3) if xs else None


def _configure_llm(model: Optional[str], num_ctx: Optional[int]):
    """The agent's LLM, with the CLI overrides applied on a copy so every
    other option (reasoning, streaming, base_url) stays as the agent sets it."""
    from wojtek_rai.agent import build_llm

    llm = build_llm()
    update: dict[str, Any] = {}
    if model:
        update["model"] = model
    if num_ctx is not None:
        update["num_ctx"] = num_ctx
    if update:
        if not all(hasattr(llm, k) for k in update):
            raise SystemExit(f"{type(llm).__name__} has no field for {update}; "
                             "--model/--num-ctx are Ollama options")
        llm = llm.model_copy(update=update)
    return llm


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="Ollama model name (default: the agent's complex_model)")
    ap.add_argument("--num-ctx", type=int, help="Ollama num_ctx (default: the agent's)")
    ap.add_argument("--images", type=int, default=2, help="camera frames in the history (default 2)")
    ap.add_argument("--history-steps", type=int, default=1,
                    help="AI tool-call + tool-result pairs before the frames (default 1)")
    ap.add_argument("--no-tools", action="store_true", help="do not bind the agent's tools")
    ap.add_argument("--runs", type=int, default=2, help="repetitions (default 2)")
    ap.add_argument("--cold", action="store_true",
                    help="unique nonce in the first message per run: defeats Ollama's prefix "
                         "cache so the whole prompt is evaluated (what a new turn costs)")
    ap.add_argument("--frame", choices=("auto", "camera", "generated"), default="auto",
                    help="frame source: camera if the sim is up, else generated (default auto)")
    ap.add_argument("--embodiment", metavar="PATH",
                    help="embodiment JSON to use instead of the agent's (same schema)")
    ap.add_argument("--json", metavar="PATH", help="also write the summary to this file")
    ap.add_argument("--quiet", action="store_true", help="only the summary lines")
    args = ap.parse_args(argv)
    if args.runs < 1 or args.images < 0 or args.history_steps < 0:
        ap.error("--runs >= 1, --images >= 0, --history-steps >= 0")

    import rclpy
    from rai.communication.ros2 import ROS2Connector

    from wojtek_rai.agent import load_embodiment
    from wojtek_rai.tools import build_tools

    rclpy.init()
    connector = ROS2Connector(node_name="wojtek_rai_bench", executor_type="multi_threaded")
    try:
        llm = _configure_llm(args.model, args.num_ctx)
        tools = [] if args.no_tools else build_tools(connector)
        bound = llm.bind_tools(tools) if tools else llm
        png, frame_source = load_frame(args.frame) if args.images else (b"", "none")
        if args.embodiment:
            from rai_whoami import EmbodimentInfo

            system = EmbodimentInfo.from_file(args.embodiment).to_langchain()
        else:
            system = load_embodiment().to_langchain()

        def messages_for(run: int) -> List[BaseMessage]:
            nonce = f"{time.time_ns():x}-{run}" if args.cold else ""
            return [system] + build_history(png, args.images, args.history_steps, nonce)

        messages = messages_for(0)

        config = {
            "model": getattr(llm, "model", None),
            "num_ctx": getattr(llm, "num_ctx", None),
            "reasoning": getattr(llm, "reasoning", None),
            "base_url": getattr(llm, "base_url", None),
            "tools": len(tools),
            "tool_names": [t.name for t in tools],
            "images": args.images,
            "frame_source": frame_source,
            "frame_bytes": len(png),
            "history_steps": args.history_steps,
            "cold": args.cold,
            "embodiment": args.embodiment or "agent",
            "messages": len(messages),
            "runs": args.runs,
        }
        if not args.quiet:
            print(f"[bench] {json.dumps({k: v for k, v in config.items() if k != 'tool_names'})}",
                  flush=True)

        runs: List[RunStats] = []
        for i in range(args.runs):
            stats = one_run(bound, messages_for(i + 1), i + 1)
            runs.append(stats)
            if not args.quiet:
                print(f"[run {stats.run}] ttft={stats.ttft_s}s total={stats.total_s}s "
                      f"prompt={stats.prompt_tokens} gen={stats.gen_tokens} "
                      f"prompt_eval={stats.prompt_eval_s}s ({stats.prompt_tok_s} tok/s) "
                      f"gen={stats.gen_tok_s} tok/s load={stats.load_s}s server={stats.server_s}s "
                      f"reasoning_chars={stats.reasoning_chars} hidden_chunks={stats.hidden_chunks} | {stats.text}", flush=True)

        summary = {
            "config": config,
            "runs": [asdict(r) for r in runs],
            "mean": {
                "ttft_s": _mean([r.ttft_s for r in runs]),
                "total_s": _mean([r.total_s for r in runs]),
                "server_s": _mean([r.server_s for r in runs]),
                "prompt_tokens": _mean([r.prompt_tokens for r in runs]),
                "gen_tokens": _mean([r.gen_tokens for r in runs]),
                "prompt_eval_s": _mean([r.prompt_eval_s for r in runs]),
                "prompt_tok_s": _mean([r.prompt_tok_s for r in runs]),
                "gen_tok_s": _mean([r.gen_tok_s for r in runs]),
                "reasoning_chars": _mean([r.reasoning_chars for r in runs]),
                "hidden_chunks": _mean([r.hidden_chunks for r in runs]),
            },
        }
        print(json.dumps(summary, indent=2))
        m = summary["mean"]
        print(
            f"| {config['model']} | {'cold' if args.cold else 'warm'} | tools={config['tools']} | images={args.images} "
            f"| steps={args.history_steps} | prompt_tok={m['prompt_tokens']} "
            f"| ttft={m['ttft_s']}s | total={m['total_s']}s (server {m['server_s']}s) "
            f"| prompt_eval={m['prompt_eval_s']}s ({m['prompt_tok_s']} tok/s) "
            f"| gen={m['gen_tok_s']} tok/s | gen_tok={m['gen_tokens']} hidden_chunks={m['hidden_chunks']} | runs={args.runs} |"
        )
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
        return 0
    finally:
        connector.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
