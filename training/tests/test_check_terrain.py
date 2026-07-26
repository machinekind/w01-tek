"""The heightfield-overflow guardrail in check-terrain.

The warning it looks for is printed from MJWarp device code, so producing a
real one needs a CUDA GPU. Everything here works on the string and the
descriptor plumbing instead: the counter, the dup2 capture, and main()'s exit
path when the count is not zero.
"""

import json
import os
import subprocess
import sys

import pytest

from wojtek_rl import check_terrain, paths

WARNING = check_terrain.OVERFLOW_WARNING


def test_counter_finds_every_occurrence():
    log = (
        "put_model 1.2s\n"
        f"Warning: {WARNING} at cell 12\n"
        "step 0\n"
        f"Warning: {WARNING} at cell 13\n"
    )
    assert check_terrain.count_overflow_warnings(log) == 2
    assert check_terrain.count_overflow_warnings("clean run, 200 steps\n") == 0


def test_capture_takes_descriptor_writes_and_puts_them_back(capfd):
    """A write straight to fd 1 bypasses sys.stdout, which is the whole reason
    the capture uses dup2 instead of redirect_stdout."""
    with check_terrain.capture_os_stdout(True) as captured:
        os.write(1, f"Warning: {WARNING}\n".encode())
    assert check_terrain.count_overflow_warnings(captured[0]) == 1
    # Re-emitted, so redirecting hid nothing.
    assert WARNING in capfd.readouterr().out


def test_capture_disabled_leaves_stdout_alone(capfd):
    with check_terrain.capture_os_stdout(False) as captured:
        print("ordinary output")
    assert captured[0] == ""
    assert "ordinary output" in capfd.readouterr().out


def test_main_fails_on_any_warning(tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    sentinel = tmp_path / "sentinel"

    def fake_check(args):
        return {
            "status": "ok",
            "backend": "warp",
            "steps_per_s": 100.0,
            "env_steps_per_s": 1e5,
            "contacts": {
                "active_max": 40,
                "active_mean": 12.0,
                "hfield_overflow_warnings": 7,
            },
        }

    monkeypatch.setattr(check_terrain, "run_check", fake_check)
    monkeypatch.setattr(
        sys, "argv",
        ["check-terrain", "--out", str(out), "--sentinel", str(sentinel)],
    )
    with pytest.raises(SystemExit) as exc:
        check_terrain.main()
    assert exc.value.code == 1
    report = json.loads(out.read_text())
    assert report["status"] == "fail"
    assert "7 heightfield contact overflow" in report["error"]
    assert sentinel.read_text().startswith("FAIL")


def test_main_passes_when_there_are_no_warnings(tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    sentinel = tmp_path / "sentinel"

    def fake_check(args):
        return {
            "status": "ok",
            "backend": "jax",
            "steps_per_s": 100.0,
            "env_steps_per_s": 1e5,
            "contacts": {
                "active_max": 40,
                "active_mean": 12.0,
                "hfield_overflow_warnings": 0,
            },
        }

    monkeypatch.setattr(check_terrain, "run_check", fake_check)
    monkeypatch.setattr(
        sys, "argv",
        ["check-terrain", "--out", str(out), "--sentinel", str(sentinel)],
    )
    with pytest.raises(SystemExit) as exc:
        check_terrain.main()
    assert exc.value.code == 0
    assert sentinel.read_text().strip() == "OK"


def test_smoke_path_greps_for_the_same_string():
    """run.sh cannot import the constant, so the two copies are checked here."""
    assert WARNING in (paths.PROJECT_DIR / "run.sh").read_text()


def _patched_run_sh(tmp_path, stub_body: str):
    """run.sh with its interpreter swapped for a stub, so the smoke branch can
    be driven without training anything."""
    stub = tmp_path / "fake_python.sh"
    stub.write_text("#!/usr/bin/env bash\n" + stub_body + "\n")
    stub.chmod(0o755)
    script = tmp_path / "run.sh"
    script.write_text(
        (paths.PROJECT_DIR / "run.sh").read_text().replace(
            "PY=.venv/bin/python", f"PY={stub}"
        )
    )
    script.chmod(0o755)
    return script


def test_smoke_fails_when_the_warning_appears_in_the_run_output(tmp_path):
    script = _patched_run_sh(tmp_path, f'echo "Warning: {WARNING} [cell 7]"')
    # run.sh keeps the log on failure, on purpose -- its message names the file.
    # TMPDIR points it at tmp_path so the test does not litter the real one.
    result = subprocess.run(
        [str(script), "smoke"],
        capture_output=True,
        text=True,
        env={**os.environ, "TMPDIR": str(tmp_path)},
    )
    assert result.returncode == 1
    assert "smoke FAILED" in result.stderr
    assert list(tmp_path.glob("wojtek_smoke.*"))  # the log it points at exists


def test_smoke_passes_a_clean_run_through(tmp_path):
    script = _patched_run_sh(tmp_path, 'echo "done -> runs/x"')
    result = subprocess.run([str(script), "smoke"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "done -> runs/x" in result.stdout


def test_smoke_keeps_the_trainer_exit_code(tmp_path):
    script = _patched_run_sh(tmp_path, 'echo "boom" >&2; exit 3')
    result = subprocess.run([str(script), "smoke"], capture_output=True, text=True)
    assert result.returncode == 3
