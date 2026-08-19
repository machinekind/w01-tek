# Prompts, split by the model they land on

- `qwen/` — everything the **Qwen3-VL agent** sees (chat persona, tool
  rules, reply contract, search observer, VLM-navigator system prompt).
  **Keep these in ENGLISH**: the 4B model reasons and follows tools best in
  English; only the spoken `say` sentence leaves English, and that is
  controlled by `qwen/voice_language.txt`, not by translating the prompts.
- The **Bielik** prompts live with the ROS node that uses them:
  `ros/src/wojtek_brain/wojtek_brain/prompts/bielik/`. **Keep those in
  POLISH** — Bielik is Polish-native and speaks directly to the user.

Placeholders like `{target}` or `{language}` are plain replace-tokens
(str.replace, not str.format) — keep the token, edit everything around it;
literal JSON braces in the text are safe.  Edits take effect on the next
app/node restart; nothing is compiled in.
