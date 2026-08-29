"""Editable prompt store: every Qwen-facing prompt lives in prompts/qwen/*.txt.

These all land on the Qwen3-VL agent and MUST stay in English (the model
reasons/tools best in English; only the spoken sentence leaves English).
The Polish Bielik prompts live in ros/src/wojtek_brain/.../prompts/bielik/.

Edit the .txt files to change what the models are told — no code change,
just restart the app.  Placeholders like {target} or {language} are plain
tokens substituted with str.replace (NOT str.format), so literal JSON
braces in the prompt text are safe.

The files:
  persona.txt          Wojtek's character (Qwen chat agent)
  contract.txt         the one-JSON-object reply contract
  rules.txt            tool-usage rules (when to look/map/navigate/search)
  voice_style.txt      extra style rules for spoken replies
  text_language.txt    language policy for typed replies
  voice_language.txt   language policy for spoken replies      {language}
  translate_style.txt  style when the translate mode is active
  translate.txt        the standalone translation call         {language} {text}
  search_observer.txt  the search observer's scoring call      {target}
  nav_system.txt       the VLM navigator's system prompt       {MIN_TURN_DEG} {MAX_TURN_DEG} {MIN_FORWARD_M} {MAX_FORWARD_M} {MAX_BACKWARD_M}
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts" / "qwen"


def load(name: str, **tokens: object) -> str:
    """Read prompts/<name>.txt and substitute {token} markers.

    Plain replace, not format: prompt files carry literal JSON braces.
    """
    text = (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")
    for key, value in tokens.items():
        text = text.replace("{" + key + "}", str(value))
    return text
