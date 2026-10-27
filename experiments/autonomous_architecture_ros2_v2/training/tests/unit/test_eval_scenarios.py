from wojtek_rl.battery import battery_scenarios
from wojtek_rl.eval import eval_scenarios


# -- eval_scenarios: battery_scenarios() plus the eval-only demo_sequence,
# without mutating battery.py's fixed comparison battery.


def test_eval_scenarios_includes_battery_scenarios_unchanged():
    # battery_scenarios() builds fresh closures per call, so compare
    # behavior (n_steps and sampled cmd_at outputs), not function identity.
    battery = battery_scenarios()
    evals = eval_scenarios()
    assert set(battery.keys()) <= set(evals.keys())
    for name, (cmd_at, n) in battery.items():
        e_cmd_at, e_n = evals[name]
        assert e_n == n
        for i in (0, n // 2, n - 1):
            assert list(map(float, e_cmd_at(i))) == list(map(float, cmd_at(i)))


def test_eval_scenarios_adds_demo_sequence_only():
    assert set(eval_scenarios().keys()) - set(battery_scenarios().keys()) == {
        "demo_sequence",
    }


# cmd_at returns jp.array (float32 by default in this project -- no x64
# config anywhere), so 0.5/0.7-scale values carry ~1e-7 rounding error
# against a plain Python float; compare with a tolerance loose enough to
# absorb that but tight enough to catch a real logic error (test_battery.py
# uses the same tolerance for the same reason).
def _close(a, b, tol=1e-5):
    return abs(a - b) < tol


def test_demo_sequence_step_count():
    _, n = eval_scenarios()["demo_sequence"]
    assert n == 1200


def test_demo_sequence_windows():
    cmd, _ = eval_scenarios()["demo_sequence"]

    # 0-3s: stand
    assert float(cmd(0)[0]) == 0.0
    assert float(cmd(0)[2]) == 0.0
    assert float(cmd(149)[0]) == 0.0
    assert float(cmd(149)[2]) == 0.0

    # 3-9s: trot forward
    assert _close(float(cmd(150)[0]), 0.5)
    assert float(cmd(150)[2]) == 0.0
    assert _close(float(cmd(449)[0]), 0.5)

    # 9-15s: turn in place
    assert float(cmd(450)[0]) == 0.0
    assert _close(float(cmd(450)[2]), 0.7)
    assert _close(float(cmd(749)[2]), 0.7)

    # 15-21s: trot forward
    assert _close(float(cmd(750)[0]), 0.5)
    assert float(cmd(750)[2]) == 0.0
    assert _close(float(cmd(1049)[0]), 0.5)

    # 21-24s: stand
    assert float(cmd(1050)[0]) == 0.0
    assert float(cmd(1050)[2]) == 0.0
    assert float(cmd(1199)[0]) == 0.0
    assert float(cmd(1199)[2]) == 0.0

    # height pinned throughout, same anchor as battery_scenarios()
    for i in (0, 150, 450, 750, 1050, 1199):
        assert float(cmd(i)[3]) == 0.125
    # vy always zero: demo_sequence has no strafe leg
    for i in (0, 150, 450, 750, 1050, 1199):
        assert float(cmd(i)[1]) == 0.0
