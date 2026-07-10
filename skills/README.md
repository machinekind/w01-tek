# Claude Code skills (opt-in)

These skills are deliberately **not** auto-discovered: Claude Code only scans `.claude/skills/`,
which is gitignored here. To opt in, symlink (or copy) the ones you want into your local
`.claude/skills/`:

```bash
mkdir -p .claude/skills
ln -s ../../skills/brax-locomotion-training .claude/skills/
```

Or point Claude at one ad hoc: "read skills/<name>/SKILL.md and follow it".
