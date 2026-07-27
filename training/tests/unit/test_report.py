import numpy as np

from wojtek_rl import report
from wojtek_rl.report import (
    assemble_report,
    foot_force_proxy,
    power_percentiles,
    render_markdown,
    termination_summary,
    torque_by_speed,
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
            "scenario": "stand_to_trot_ramp",
            "fell_at": None,
            "height": None,
            "gravity_z": None,
            "min_height": 0.06,
            "max_tilt_gz": -0.4,
        },
        {
            "scenario": "turn",
            "fell_at": 42,
            "height": 0.03,  # below min_height -> height fall
            "gravity_z": -0.9,  # not over max_tilt_gz -> not tilt
            "min_height": 0.06,
            "max_tilt_gz": -0.4,
        },
        {
            "scenario": "strafe",
            "fell_at": 100,
            "height": 0.10,  # above min_height -> not height
            "gravity_z": -0.1,  # over max_tilt_gz -> tilt fall
            "min_height": 0.06,
            "max_tilt_gz": -0.4,
        },
        {
            "scenario": "walk_to_stop",
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
    assert s["per_scenario"]["stand_to_trot_ramp"] == {"fell": False, "fell_at": None, "reason": None}
    assert s["per_scenario"]["turn"]["reason"] == "height"
    assert s["per_scenario"]["strafe"]["reason"] == "tilt"
    assert s["per_scenario"]["walk_to_stop"]["reason"] == "both"
    assert s["per_scenario"]["turn"]["fell_at"] == 42


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
        battery={"run": "probe", "checkpoint": "1000", "stand_to_trot_ramp": {"fell_at": None}},
        torque={"p50": 1.0, "p90": 2.0, "p99": 3.0, "max": 4.0},
        power={"p50": 1.0, "p90": 2.0, "p99": 3.0, "mean_total": 4.0},
        foot_force={"peak_accel_mps2": 1.0, "peak_force_n": 10.0},
        termination={"scenarios_run": 1, "fall_count": 0,
                     "fall_reason_counts": {}, "per_scenario": {}},
        torque_by_speed={"0.0-0.2": {"p90": 1.0, "p99": 2.0, "n": 5}},
        timestamp="2026-07-10T00:00:00",
    )
    assert set(report.keys()) == {
        "run", "checkpoint", "battery_scene", "battery", "torque", "power",
        "foot_force_proxy", "termination", "torque_by_speed", "terrain",
        "timestamp",
    }
    # A run with no terrain scan says so, rather than omitting the section.
    assert report["terrain"] is None
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
            "stand_to_trot_ramp": {"fell_at": None, "steps": 750, "vibration": 0.1},
        },
        torque={"p50": 1.0, "p90": 2.0, "p99": 3.0, "max": 4.0},
        power={"p50": 1.0, "p90": 2.0, "p99": 3.0, "mean_total": 4.0},
        foot_force={"peak_accel_mps2": 1.234, "peak_force_n": 12.3},
        termination=termination_summary(
            [{"scenario": "stand_to_trot_ramp", "fell_at": None, "height": None,
              "gravity_z": None, "min_height": 0.06, "max_tilt_gz": -0.4}]
        ),
        torque_by_speed={"0.0-0.2": {"p90": 1.0, "p99": 2.0, "n": 5}},
        timestamp="2026-07-10T00:00:00",
    )
    md = render_markdown(report)
    assert "# Eval report: probe" in md
    assert "## Battery" in md
    assert "## Torque" in md
    assert "## Torque by achieved speed" in md
    assert "## Power" in md
    assert "## Foot-force proxy" in md
    assert "## Termination" in md
    assert "stand_to_trot_ramp" in md
    assert "0.0-0.2" in md


# -- torque_by_speed ----------------------------------------------------------


def test_torque_by_speed_known_values():
    # 4 steps, 2 joints. speed = hypot(vx_local, vy_local):
    #   step0: hypot(0.1, 0.0) = 0.1  -> bin [0.0, 0.2)
    #   step1: hypot(0.3, 0.0) = 0.3  -> bin [0.2, 0.4)
    #   step2: hypot(0.3, 0.4) = 0.5  -> bin [0.4, 0.6)
    #   step3: hypot(0.6, 0.8) = 1.0  -> bin [1.0, inf)  (>= edge)
    vx_local = np.array([0.1, 0.3, 0.3, 0.6])
    vy_local = np.array([0.0, 0.0, 0.4, 0.8])
    force = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    r = torque_by_speed(vx_local, vy_local, force)
    assert r["0.0-0.2"] == {
        "p90": float(np.percentile([1.0, 2.0], 90)),
        "p99": float(np.percentile([1.0, 2.0], 99)),
        "n": 1,
    }
    assert r["0.2-0.4"] == {
        "p90": float(np.percentile([3.0, 4.0], 90)),
        "p99": float(np.percentile([3.0, 4.0], 99)),
        "n": 1,
    }
    assert r["0.4-0.6"] == {
        "p90": float(np.percentile([5.0, 6.0], 90)),
        "p99": float(np.percentile([5.0, 6.0], 99)),
        "n": 1,
    }
    assert r["0.6-0.8"] == {"p90": None, "p99": None, "n": 0}
    assert r["0.8-1.0"] == {"p90": None, "p99": None, "n": 0}
    assert r["1.0+"] == {
        "p90": float(np.percentile([7.0, 8.0], 90)),
        "p99": float(np.percentile([7.0, 8.0], 99)),
        "n": 1,
    }


def test_torque_by_speed_empty():
    r = torque_by_speed(np.zeros(0), np.zeros(0), np.zeros((0, 12)))
    assert all(v == {"p90": None, "p99": None, "n": 0} for v in r.values())
    assert set(r.keys()) == {
        "0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0", "1.0+",
    }


# -- terrain section ------------------------------------------------------------


def _scan_doc():
    """A terrain-scan document shaped like the one terrain_scan writes."""
    from wojtek_rl import terrain_scan, terrain_suite

    cell = terrain_suite.CELLS_BY_NAME["pyramid_stairs_5cm"]
    tracked = terrain_suite.CELLS_BY_NAME["pyramid_stairs_9cm"]
    reduced = terrain_scan.CellResult(
        passed=29, of=32, falls=2, timeouts=1, crossings_mean=3.8,
        saturation=0.12, track_err=0.05, clearance=0.01, measured=30,
        nacon_max=88, nefc_max=300, steps=1036,
    )
    return {
        "run": "probe", "checkpoint": "1000", "engine": "warp",
        "arena": terrain_suite.arena_fingerprint(),
        "runs_per_cell_speed": 32,
        "cells": {
            cell.name: {"0.4": terrain_scan.cell_entry(cell, 0.4, reduced),
                        "0.7": terrain_scan.cell_entry(cell, 0.7, reduced)},
            tracked.name: {"0.4": terrain_scan.cell_entry(tracked, 0.4, reduced)},
        },
        "gate": {
            "absolute": {"verdict": "pass", "checked": 2, "failures": []},
            "relative": {"verdict": "no baseline", "notes": ["no --baseline given"]},
        },
    }


def test_terrain_section_says_so_when_there_is_no_scan():
    lines = report.render_terrain_markdown(None)
    assert "## Terrain" in lines
    assert any("no terrain scan" in line for line in lines)


def test_terrain_section_renders_a_scan():
    lines = report.render_terrain_markdown(_scan_doc())
    text = "\n".join(lines)
    assert "## Terrain" in text
    assert "engine: warp" in text
    # one row per cell and speed, with the bar and where its number came from
    assert "| pyramid_stairs_5cm | 0.4 | 29 | 32 | 26 | plan |" in text
    assert "| pyramid_stairs_5cm | 0.7 | 29 | 32 | 26 | provisional |" in text
    assert "| pyramid_stairs_9cm | 0.4 | 29 | 32 | - | tracked |" in text
    assert "### Gate" in text
    # how many runs the per-step metrics average over, so a thin sample shows
    assert "| 30 |" in text
    assert "measured` is how many runs" in text
    # the scan's own provenance renders, so a stale scan is visible
    assert "scan: run probe, checkpoint 1000" in text


def test_terrain_section_warns_when_the_scan_is_for_another_checkpoint():
    """The report renders whatever scan file sits in the run dir, and it picks
    the newest checkpoint itself -- the two can disagree."""
    lines = report.render_terrain_markdown(_scan_doc(), report_checkpoint="2000")
    assert any("scan measured checkpoint 1000" in line for line in lines)
    lines = report.render_terrain_markdown(_scan_doc(), report_checkpoint="1000")
    assert not any("scan measured checkpoint" in line for line in lines)


def test_terrain_section_tolerates_an_incomplete_document():
    """A scan that crashed part way, or an older schema, must not take the whole
    report down with it."""
    for doc in ({}, {"cells": {}}, {"cells": None, "arena": {}},
                {"cells": {"c": {"0.4": {}}}}):
        lines = report.render_terrain_markdown(doc or None)
        assert any("Terrain" in line for line in lines)


def test_full_report_carries_the_scan_and_the_battery_scene():
    doc = _scan_doc()
    rendered = report.render_markdown(
        report.assemble_report(
            run_name="probe", checkpoint="1000",
            battery={"run": "probe", "checkpoint": "1000",
                     "stand_to_trot_ramp": {"fell_at": None, "steps": 750}},
            torque={"p50": 1.0, "p90": 2.0, "p99": 3.0, "max": 4.0},
            power={"p50": 1.0, "p90": 2.0, "p99": 3.0, "mean_total": 4.0},
            foot_force={"peak_accel_mps2": 1.0, "peak_force_n": 10.0},
            termination={"scenarios_run": 1, "fall_count": 0,
                         "fall_reason_counts": {}, "per_scenario": {}},
            battery_scene="scene_mjx.xml", terrain=doc,
        )
    )
    # the report says which scene produced the battery half
    assert "scene: scene_mjx.xml" in rendered
    assert "flat comparison" in rendered
    assert "engine: warp" in rendered
