import numpy as np

from wojtek_rl.report import (
    assemble_report,
    foot_force_proxy,
    power_percentiles,
    render_markdown,
    termination_summary,
    torque_percentiles,
)


def test_torque_percentiles_known_values():
    # 4 steps, 2 joints; |force| = [[0,10],[2,20],[4,40],[100,100]]
    force = np.array([[0, -10], [2, 20], [-4, 40], [100, 100]])
    r = torque_percentiles(force)
    flat = np.abs(force).ravel()
    assert r["p50"] == float(np.percentile(flat, 50))
    assert r["p90"] == float(np.percentile(flat, 90))
    assert r["p99"] == float(np.percentile(flat, 99))
    assert r["max"] == 100.0


def test_torque_percentiles_empty():
    r = torque_percentiles(np.zeros((0, 12)))
    assert r == {"p50": None, "p90": None, "p99": None, "max": None}


def test_power_percentiles_known_values():
    # force * vel per joint per step; check the mean-total-power reduction
    # explicitly on a tiny hand-computed case.
    force = np.array([[2.0, 4.0], [1.0, 1.0]])
    vel = np.array([[3.0, 1.0], [2.0, 2.0]])
    r = power_percentiles(force, vel)
    per_joint = np.abs(force * vel)  # [[6,4],[2,2]]
    total = per_joint.sum(axis=-1)  # [10, 4]
    assert r["mean_total"] == float(total.mean())
    flat = per_joint.ravel()
    assert r["p50"] == float(np.percentile(flat, 50))
    assert r["p99"] == float(np.percentile(flat, 99))


def test_power_percentiles_empty():
    r = power_percentiles(np.zeros((0, 12)), np.zeros((0, 12)))
    assert r == {"p50": None, "p90": None, "p99": None, "mean_total": None}


def test_foot_force_proxy_known_values():
    # baseline g=9.81; steady standing (no spike) then a clear impact spike.
    accel_z = np.array([9.81, 9.81, 20.0, -9.81])
    r = foot_force_proxy(accel_z, total_mass=10.0, gravity=9.81)
    expected_peak = max(abs(abs(v) - 9.81) for v in accel_z)  # from the 20.0 sample
    assert r["peak_accel_mps2"] == expected_peak
    assert r["peak_force_n"] == expected_peak * 10.0


def test_foot_force_proxy_empty():
    r = foot_force_proxy(np.zeros((0,)), total_mass=10.0)
    assert r == {"peak_accel_mps2": None, "peak_force_n": None}


def test_termination_summary_known_fell_and_survived_set():
    events = [
        {
            "scenario": "ramp_mid",
            "fell_at": None,
            "height": None,
            "gravity_z": None,
            "min_height": 0.06,
            "max_tilt_gz": -0.4,
        },
        {
            "scenario": "ramp_low",
            "fell_at": 42,
            "height": 0.03,  # below min_height -> height fall
            "gravity_z": -0.9,  # not over max_tilt_gz -> not tilt
            "min_height": 0.06,
            "max_tilt_gz": -0.4,
        },
        {
            "scenario": "ramp_tall",
            "fell_at": 100,
            "height": 0.10,  # above min_height -> not height
            "gravity_z": -0.1,  # over max_tilt_gz -> tilt fall
            "min_height": 0.06,
            "max_tilt_gz": -0.4,
        },
        {
            "scenario": "walk_trot",
            "fell_at": 7,
            "height": 0.02,  # both conditions trip
            "gravity_z": 0.0,
            "min_height": 0.06,
            "max_tilt_gz": -0.4,
        },
    ]
    s = termination_summary(events)
    assert s["scenarios_run"] == 4
    assert s["fall_count"] == 3
    assert s["fall_reason_counts"] == {"height": 1, "tilt": 1, "both": 1, "unknown": 0}
    assert s["per_scenario"]["ramp_mid"] == {"fell": False, "fell_at": None, "reason": None}
    assert s["per_scenario"]["ramp_low"]["reason"] == "height"
    assert s["per_scenario"]["ramp_tall"]["reason"] == "tilt"
    assert s["per_scenario"]["walk_trot"]["reason"] == "both"
    assert s["per_scenario"]["ramp_low"]["fell_at"] == 42


def test_termination_summary_all_survived():
    events = [
        {"scenario": f"s{i}", "fell_at": None, "height": None, "gravity_z": None,
         "min_height": 0.06, "max_tilt_gz": -0.4}
        for i in range(3)
    ]
    s = termination_summary(events)
    assert s["fall_count"] == 0
    assert s["fall_reason_counts"] == {"height": 0, "tilt": 0, "both": 0, "unknown": 0}
    assert all(not v["fell"] for v in s["per_scenario"].values())


def test_assemble_report_schema_keys():
    report = assemble_report(
        run_name="probe",
        checkpoint="1000",
        battery={"run": "probe", "checkpoint": "1000", "ramp_mid": {"fell_at": None}},
        torque={"p50": 1.0, "p90": 2.0, "p99": 3.0, "max": 4.0},
        power={"p50": 1.0, "p90": 2.0, "p99": 3.0, "mean_total": 4.0},
        foot_force={"peak_accel_mps2": 1.0, "peak_force_n": 10.0},
        termination={"scenarios_run": 1, "fall_count": 0,
                     "fall_reason_counts": {}, "per_scenario": {}},
        timestamp="2026-07-10T00:00:00",
    )
    assert set(report.keys()) == {
        "run", "checkpoint", "battery", "torque", "power",
        "foot_force_proxy", "termination", "timestamp",
    }
    assert report["run"] == "probe"
    assert report["checkpoint"] == "1000"
    assert report["timestamp"] == "2026-07-10T00:00:00"


def test_assemble_report_default_timestamp_when_omitted():
    report = assemble_report(
        run_name="probe", checkpoint="1", battery={}, torque={}, power={},
        foot_force={}, termination={},
    )
    assert isinstance(report["timestamp"], str) and report["timestamp"]


def test_render_markdown_smoke():
    report = assemble_report(
        run_name="probe",
        checkpoint="1000",
        battery={
            "run": "probe",
            "checkpoint": "1000",
            "ramp_mid": {"fell_at": None, "steps": 750, "vibration": 0.1},
        },
        torque={"p50": 1.0, "p90": 2.0, "p99": 3.0, "max": 4.0},
        power={"p50": 1.0, "p90": 2.0, "p99": 3.0, "mean_total": 4.0},
        foot_force={"peak_accel_mps2": 1.234, "peak_force_n": 12.3},
        termination=termination_summary(
            [{"scenario": "ramp_mid", "fell_at": None, "height": None,
              "gravity_z": None, "min_height": 0.06, "max_tilt_gz": -0.4}]
        ),
        timestamp="2026-07-10T00:00:00",
    )
    md = render_markdown(report)
    assert "# Eval report: probe" in md
    assert "## Battery" in md
    assert "## Torque" in md
    assert "## Power" in md
    assert "## Foot-force proxy" in md
    assert "## Termination" in md
    assert "ramp_mid" in md
