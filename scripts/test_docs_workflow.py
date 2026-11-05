"""Exercise the workflow's whitespace step against temporary Git histories."""

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/docs.yml"


class WhitespaceWorkflowChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Workflow test")
        self.git("config", "user.email", "test@example.invalid")

    def git(self, *args):
        return subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
            cwd=self.root, env=self.env, check=True, capture_output=True, text=True,
        ).stdout.strip()

    def commit(self, name, content):
        (self.root / name).write_text(content)
        self.git("add", name)
        self.git("commit", "-qm", "test change")
        return self.git("rev-parse", "HEAD")

    def check(self, base="", head=""):
        # Run the actual shell step, so changing the workflow range is covered.
        step = WORKFLOW.read_text().split("- name: Check whitespace\n", 1)[1]
        script = textwrap.dedent(step.split("run: |\n", 1)[1])
        return subprocess.run(
            ["bash", "-e", "-c", script], cwd=self.root,
            env=dict(self.env, PR_BASE=base, PR_HEAD=head),
            capture_output=True, text=True,
        )

    def test_push_detects_whitespace_brought_in_by_merge(self):
        self.commit("initial", "clean\n")
        self.git("switch", "-qc", "feature")
        self.commit("added", "bad   \n")
        self.git("switch", "-q", "main")
        self.git("merge", "--no-ff", "feature", "-m", "merge feature")
        result = self.check()
        self.assertEqual(result.returncode, 2)
        self.assertIn("trailing whitespace", result.stdout)

    def test_pr_ignores_base_only_whitespace_cleanup(self):
        self.commit("existing", "legacy   \n")
        self.git("branch", "feature")
        base = self.commit("existing", "legacy\n")
        self.git("switch", "-q", "feature")
        head = self.commit("unrelated", "clean\n")
        result = self.check(base, head)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_pr_rejects_its_own_whitespace(self):
        base = self.commit("initial", "clean\n")
        self.git("switch", "-qc", "feature")
        head = self.commit("added", "bad   \n")
        result = self.check(base, head)
        self.assertEqual(result.returncode, 2)
        self.assertIn("trailing whitespace", result.stdout)

    def test_clean_push_passes(self):
        self.commit("initial", "clean\n")
        self.commit("added", "also clean\n")
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
