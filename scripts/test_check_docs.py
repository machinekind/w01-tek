"""Regression tests for the documentation checker's supported Markdown syntax."""

from pathlib import Path
import tempfile
import unittest

import check_docs


class DocumentationChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def links(self, text):
        path = self.write("docs/index.md", text)
        return check_docs.check_links(path, self.root)

    def test_valid_local_links_and_duplicate_heading_anchors(self):
        self.write("guide.md", "# Setup\n# Setup\n<a id='manual'></a>\n")
        self.assertEqual(self.links(
            "[first](../guide.md#setup) [second](/guide.md#setup-1) "
            "[custom](../guide.md#manual)\n# Local\n[self](#local)"
        ), [])

    def test_missing_file_and_anchor_have_actionable_errors(self):
        self.write("guide.md", "# Setup\n")
        self.assertEqual(self.links("[file](lost.md) [anchor](../guide.md#absent)"), [
            "docs/index.md: missing lost.md",
            "docs/index.md: missing anchor ../guide.md#absent",
        ])

    def test_encoded_paths_and_angle_bracket_destinations(self):
        self.write("a guide.md", "# Setup\n")
        self.assertEqual(self.links(
            '[encoded](../a%20guide.md#setup) [space](<../a guide.md#setup> "Guide")'
        ), [])

    def test_external_links_are_not_treated_as_local_paths(self):
        self.assertEqual(self.links(
            "[web](https://example.invalid/page) [mail](mailto:a@example.invalid) "
            "[relative](//example.invalid/page)"
        ), [])

    def test_fenced_examples_are_neither_links_nor_heading_anchors(self):
        examples = (
            "```markdown\n# Example\n[broken](missing.md)\n```\n",
            "~~~markdown\n# Example\n[broken](missing.md)\n~~~\n",
            "  ````markdown\n```\n# Example\n[broken](missing.md)\n```\n  ````\n",
            "```markdown\n# Example\n[broken](missing.md)\n",
        )
        for index, example in enumerate(examples):
            with self.subTest(example=example):
                self.assertEqual(self.links(example), [])
                self.assertNotIn("example", check_docs.heading_ids(example))
                if index < len(examples) - 1:
                    self.assertEqual(self.links(example + "[real](broken.md)"), [
                        "docs/index.md: missing broken.md",
                    ])

    def test_link_cannot_target_a_heading_inside_a_fence(self):
        self.write("guide.md", "```text\n# Example\n```\n# Actual\n")
        self.assertEqual(self.links("[example](../guide.md#example)"), [
            "docs/index.md: missing anchor ../guide.md#example",
        ])
        self.assertEqual(self.links("[actual](../guide.md#actual)"), [])

    def test_inline_code_examples_are_ignored_but_real_links_checked(self):
        self.assertEqual(self.links(
            "`[example](missing.md)` and `` `[another](absent.md)` `` "
            "[real](broken.md)"
        ), ["docs/index.md: missing broken.md"])
        self.assertIn("use-cmd_vel", check_docs.heading_ids("# Use `cmd_vel`"))

    def test_paired_backticks_on_one_or_multiple_lines_are_valid_code(self):
        for example in (
            "stray ` here [example](missing.md) and later ` end",
            "stray ` here\n[example](missing.md) and later ` end",
            "``multi\n`[example](missing.md)`\nline``",
        ):
            with self.subTest(example=example):
                self.assertEqual(self.links(example + "\n\n[real](broken.md)"), [
                    "docs/index.md: missing broken.md",
                ])

    def test_backticks_cannot_hide_links_across_block_boundaries(self):
        examples = (
            "stray ` here\n\n[real](missing.md) and later ` end",
            "stray ` here\n# [real](missing.md) and later ` end",
            "- stray ` here\n- [real](missing.md) and later ` end",
            "| A | B |\n|---|---|\n| `start | [real](missing.md) `end |",
        )
        for example in examples:
            with self.subTest(example=example):
                self.assertEqual(self.links(example), ["docs/index.md: missing missing.md"])

    def test_reference_links_and_images_resolve_definitions(self):
        self.write("guide.md", "# Setup\n")
        self.write("photo.png", "image fixture")
        self.assertEqual(self.links(
            '[guide][manual] [manual][] [manual] ![image][photo] [lost][absent]\n\n'
            '[manual]: <../guide.md#setup> "Setup guide"\n'
            '[photo]:\n  ../photo.png\n'
            '[absent]: missing.md\n'
        ), ["docs/index.md: missing missing.md"])
        self.assertEqual(self.links("![image][photo]\n\n[photo]: missing.png\n"), [
            "docs/index.md: missing missing.png",
        ])
        self.assertEqual(self.links("[guide][manual]\n\n[manual]: ../guide.md#absent\n"), [
            "docs/index.md: missing anchor ../guide.md#absent",
        ])

    def test_unused_reference_definitions_and_code_are_not_links(self):
        self.assertEqual(self.links(
            "`[literal][unused]`\n\n"
            "    [also literal](missing.md)\n\n"
            "[unused]: absent.md\n"
        ), [])

    def test_setext_and_formatted_headings_use_visible_text(self):
        self.write("guide.md", (
            "Title\n=====\n\nSubtitle\n--------\n\n"
            "# Heading with [a link](other.md)\n"
            "## *Styled* `cmd_vel` and ~~old~~\n"
            "## A ![small *image*](image.png) &amp; <em>text</em>\n"
        ))
        for anchor in ("title", "subtitle", "heading-with-a-link",
                       "styled-cmd_vel-and-old", "a-small-image--text"):
            with self.subTest(anchor=anchor):
                self.assertEqual(self.links(f"[heading](../guide.md#{anchor})"), [])

    def test_generated_heading_ids_do_not_collide(self):
        self.assertEqual(check_docs.heading_ids("# Title\n# Title\n# Title-1\n"), {
            "title", "title-1", "title-1-1",
        })

    def test_explicit_html_anchors_ignore_code_and_allow_other_attributes(self):
        self.write("guide.md", (
            "<a class='anchor' id='manual'></a>\n\n"
            "<a name='legacy'></a>\n\n"
            "`<a id='inline-example'></a>`\n\n"
            "```html\n<a id='fenced-example'></a>\n```\n"
        ))
        self.assertEqual(self.links("[manual](../guide.md#manual) [legacy](../guide.md#legacy)"), [])
        for anchor in ("inline-example", "fenced-example"):
            with self.subTest(anchor=anchor):
                self.assertEqual(self.links(f"[example](../guide.md#{anchor})"), [
                    f"docs/index.md: missing anchor ../guide.md#{anchor}",
                ])

    def defaults_fixture(self, *, rate=100, bus="spi", literal_bus=True):
        bus_value = repr(bus) if literal_bus else "get_bus()"
        self.write("ros/src/wojtek_bringup/wojtek_bringup/launch_common.py", "\n".join([
            f"DeclareLaunchArgument('bus', default_value={bus_value})",
            "DeclareLaunchArgument('can_baud', default_value='1')",
            "DeclareLaunchArgument('boot_pose', default_value='home')",
            "DeclareLaunchArgument('use_imu', default_value='true')",
        ]))
        self.write("ros/src/wojtek_bringup/config/real_controllers.yaml", "".join(
            f"{name}:\n  ros__parameters:\n    update_rate: {value}\n"
            for name, value in (
                ("controller_manager", rate), ("joint_state_broadcaster", 50),
                ("imu_sensor_broadcaster", 50), ("magnetometer_broadcaster", 100),
            )
        ))
        self.write("docs/compatibility.md", "\n".join([
            "| Controller manager | 100 Hz |",
            "| Joint and IMU broadcasters | 50 Hz each |",
            "| Magnetometer broadcaster | 100 Hz |",
            "| CAN | CANdle SPI, 1 Mbit/s default |",
            "| Manual launch boot pose | `home` |",
            "| IMU | I2C enabled by default |",
        ]))

    def test_runtime_defaults_match_the_documented_table(self):
        self.defaults_fixture()
        self.assertEqual(check_docs.check_defaults(self.root), [])

    def test_runtime_rate_and_launch_default_drift_are_reported(self):
        self.defaults_fixture(rate=200, bus="usb")
        self.assertEqual(check_docs.check_defaults(self.root), [
            "docs/compatibility.md: expected | Controller manager | 200 Hz |",
            "docs/compatibility.md: expected | CAN | CANdle USB, 1 Mbit/s default |",
        ])

    def test_dynamic_launch_default_has_a_diagnostic_instead_of_key_error(self):
        self.defaults_fixture(literal_bus=False)
        self.assertEqual(check_docs.check_defaults(self.root), [
            "cannot read constant launch defaults: bus; update check_docs.py",
        ])


if __name__ == "__main__":
    unittest.main()
