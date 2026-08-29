# wojtek_brain

Brain nodes for the agentic stack ([design](../../../docs/plans/agentic-ros2.md)):

| node | in | out |
|---|---|---|
| `router` | `/wojtek/asr/final` | `/wojtek/intent` (chat / nav / visual / cancel / system) |
| `bielik` | `/wojtek/intent`, `/wojtek/say_en`, `/wojtek/audio/speech_started` | `/wojtek/say` (sentence stream to TTS) |

The router runs Polish keyword rules until a fine-tuned encoder lands
(`model_path` parameter).  Train one with:

```bash
ROUTER_GEN_URL=http://127.0.0.1:8091 ROUTER_GEN_MODEL=<served-model> \
  python3 tools/gen_router_dataset.py > router_dataset.jsonl
ROUTER_DATA=router_dataset.jsonl ROUTER_OUT=router_model python3 tools/train_router.py
```

Compare `ROUTER_BASE=allegro/herbert-base-cased` (default) against
mmBERT/EuroBERT/ModernBERT on the same dataset before switching the default.

`bielik` streams sentences to TTS the moment punctuation closes them, speaks
canned acknowledgements for `nav` (the Qwen agent walks and reports), and
translates the agent's English (`/wojtek/say_en`) to Polish.  Barge-in
(`speech_started`) cancels generation between tokens and drops the queue.
The LLM itself is vLLM in a separate process; this node is an HTTP client
(`requirements.txt` → deployment venv).

Tests (model- and rclpy-free):

```bash
python -m pytest ros/src/wojtek_brain/test -q
```
