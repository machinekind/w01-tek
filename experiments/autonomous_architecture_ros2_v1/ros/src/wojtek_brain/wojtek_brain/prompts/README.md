# Bielik prompts — POLISH

Everything in `bielik/` goes to the Bielik LLM, which talks directly to the
user: keep it Polish.  `nav_acks.txt` / `cancel_acks.txt` are one line per
variant, picked round-robin.  The English-side (Qwen3-VL agent) prompts
live in `training/wojtek_agent/prompts/qwen/`.

Edits take effect on the next node restart (colcon reinstalls the files via
package_data on rebuild; with --symlink-install a restart is enough).
