#!/usr/bin/env python3
"""Check local Markdown links/anchors and documented runtime defaults.

Supports CommonMark links, images and headings, plus tables and strikethrough.
Code examples are excluded by the parser. Explicit HTML <a id/name> anchors
are recognized; raw HTML links and other GitHub extensions are not checked.
"""

import ast
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = MarkdownIt("commonmark").enable(["table", "strikethrough"])


def inline_tokens(tokens):
    for token in tokens:
        if token.type == "inline":
            yield from token.children or []
        elif token.type == "html_block":
            yield token


def heading_text(tokens):
    """Rendered heading text, excluding link destinations and markup."""
    parts = []
    for token in tokens:
        if token.type in ("text", "code_inline"):
            parts.append(token.content)
        elif token.type in ("softbreak", "hardbreak"):
            parts.append(" ")
        elif token.type == "image":
            parts.append(heading_text(token.children or []))
    return "".join(parts)


class ExplicitAnchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.ids.update(value for key, value in attrs if key in ("id", "name") and value)


def heading_ids(text):
    tokens = MARKDOWN.parse(text)
    ids = set()
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        heading = heading_text(tokens[index + 1].children or []).strip()
        base = re.sub(r"[^\w\s-]", "", heading.lower()).replace(" ", "-")
        slug, count = base, 0
        while slug in ids:
            count += 1
            slug = f"{base}-{count}"
        ids.add(slug)
    explicit = ExplicitAnchors()
    for token in inline_tokens(tokens):
        if token.type in ("html_inline", "html_block"):
            explicit.feed(token.content)
    explicit.close()
    return ids | explicit.ids


def check_links(path, root=ROOT):
    errors = []
    tokens = MARKDOWN.parse(path.read_text(encoding="utf-8"))
    for token in inline_tokens(tokens):
        if token.type == "link_open":
            target = token.attrGet("href")
        elif token.type == "image":
            target = token.attrGet("src")
        else:
            continue
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("//"):
            continue
        link, _, anchor = target.partition("#")
        if link.startswith("/"):
            dest = (root / unquote(link).lstrip("/")).resolve()
        else:
            dest = (path.parent / unquote(link)).resolve() if link else path
        if not dest.exists():
            errors.append(f"{path.relative_to(root)}: missing {target}")
        elif anchor and dest.suffix == ".md" and unquote(anchor) not in heading_ids(dest.read_text(encoding="utf-8")):
            errors.append(f"{path.relative_to(root)}: missing anchor {target}")
    return errors


def check_defaults(root=ROOT):
    source = root / "ros/src/wojtek_bringup/wojtek_bringup/launch_common.py"
    defaults = {}
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "DeclareLaunchArgument":
            if node.args and isinstance(node.args[0], ast.Constant):
                for kw in node.keywords:
                    if kw.arg == "default_value" and isinstance(kw.value, ast.Constant):
                        defaults[node.args[0].value] = kw.value.value
    required = {"bus", "can_baud", "boot_pose", "use_imu"}
    missing = required - defaults.keys()
    if missing:
        return [f"cannot read constant launch defaults: {', '.join(sorted(missing))}; update check_docs.py"]
    config = (root / "ros/src/wojtek_bringup/config/real_controllers.yaml").read_text()
    rates = {}
    for name in ("controller_manager", "joint_state_broadcaster", "imu_sensor_broadcaster", "magnetometer_broadcaster"):
        section = re.search(rf"(?ms)^{name}:\n(.*?)(?=^\w|\Z)", config)
        match = re.search(r"^\s+update_rate:\s*(\d+)", section.group(1), re.M) if section else None
        if not match:
            return [f"cannot read {name}.update_rate; update check_docs.py"]
        rates[name] = match.group(1)
    expected = [
        f"| Controller manager | {rates['controller_manager']} Hz |",
        f"| Magnetometer broadcaster | {rates['magnetometer_broadcaster']} Hz |",
        f"| CAN | CANdle {str(defaults['bus']).upper()}, {defaults['can_baud']} Mbit/s default |",
        f"| Manual launch boot pose | `{defaults['boot_pose']}` |",
    ]
    if rates["joint_state_broadcaster"] == rates["imu_sensor_broadcaster"]:
        expected.append(f"| Joint and IMU broadcasters | {rates['joint_state_broadcaster']} Hz each |")
    else:
        return ["joint/IMU rates now differ; update compatibility table and check_docs.py"]
    expected.append(f"| IMU | I2C {'enabled' if defaults['use_imu'] == 'true' else 'disabled'} by default |")
    doc = (root / "docs/compatibility.md").read_text()
    return [f"docs/compatibility.md: expected {row}" for row in expected if row not in doc]


def main():
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
    ).decode().split("\0")
    paths = sorted({ROOT / f for f in files if f.endswith(".md") and (ROOT / f).is_file()})
    errors = [error for path in paths for error in check_links(path)] + check_defaults()
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f"Checked {len(paths)} Markdown files and runtime defaults.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
