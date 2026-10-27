"""The machine panel's parsers, on strings. No /proc, no ROS."""

from wojtek_vlm_gui import sysmon

STAT_T0 = """cpu  400 20 100 1000 10 0 20 0 0 0
cpu0 100 10 50 200 5 0 10 0 0 0
cpu1 100 10 50 300 5 0 10 0 0 0
cpu2 0 0 0 250 0 0 0 0 0 0
intr 12345
ctxt 6789
"""
STAT_T1 = """cpu  500 20 150 1050 10 0 20 0 0 0
cpu0 150 10 75 225 5 0 10 0 0 0
cpu1 100 10 50 400 5 0 10 0 0 0
cpu2 50 0 0 250 0 0 0 0 0 0
"""


def test_per_core_usage_is_busy_over_elapsed_between_two_samples():
    usage = sysmon.core_usage(sysmon.parse_stat(STAT_T0), sysmon.parse_stat(STAT_T1))
    assert len(usage) == 3
    assert usage[0] == 75.0     # 75 busy of 100 elapsed
    assert usage[1] == 0.0      # only idle moved
    assert usage[2] == 100.0    # only busy moved


def test_first_sample_is_since_boot_and_never_divides_by_zero():
    now = sysmon.parse_stat(STAT_T1)
    assert sysmon.core_usage({}, now)[1] == 100.0 * 170 / 575
    assert sysmon.core_usage(now, now) == [0.0, 0.0, 0.0]


def test_the_other_parsers():
    assert sysmon.parse_loadavg("5.62 5.10 5.43 4/400 10404\n") == (5.62, 5.10, 5.43)
    used, total = sysmon.parse_meminfo("MemTotal:  7988408 kB\nMemFree: 1 kB\nMemAvailable: 6896460 kB\n")
    assert (used, total) == ((7988408 - 6896460) // 1024, 7988408 // 1024)
    assert sysmon.parse_temp("51121\n") == 51.121
    assert sysmon.parse_temp("") is None
    assert sysmon.parse_isolated("2-3\n") == [2, 3]
    assert sysmon.parse_isolated("1,3") == [1, 3]
    assert sysmon.parse_isolated("\n") == []


def test_rt_cores_are_labelled_reserved():
    assert sysmon.core_label(2, [2, 3]) == "core 2 (RT, reserved)"
    assert sysmon.core_label(0, [2, 3]) == "core 0"


def test_a_live_read_never_raises_and_names_the_host(tmp_path, monkeypatch):
    monkeypatch.setattr(sysmon, "STAT_PATH", str(tmp_path / "missing"))
    monkeypatch.setattr(sysmon, "LOADAVG_PATH", str(tmp_path / "missing"))
    monkeypatch.setattr(sysmon, "MEMINFO_PATH", str(tmp_path / "missing"))
    monkeypatch.setattr(sysmon, "THERMAL_PATH", str(tmp_path / "missing"))
    monkeypatch.setattr(sysmon, "ISOLATED_PATH", str(tmp_path / "missing"))
    snap = sysmon.SysMon(host="bench").read()
    assert snap.host == "bench" and snap.cores == [] and snap.temp_c is None


def test_the_snapshot_survives_the_json_round_trip_and_tolerates_missing_fields():
    snap = sysmon.Snapshot(host="robot-core", cores=[50.0, 1.0, 0.0, 0.0], isolated=[2, 3],
                           load=(1.0, 2.0, 3.0), mem_used_mb=1000, mem_total_mb=7800, temp_c=51.0)
    assert sysmon.snapshot_from_dict(sysmon.snapshot_to_dict(snap)) == snap
    bare = sysmon.snapshot_from_dict({"host": "x"})
    assert bare.cores == [] and bare.temp_c is None and bare.load == (0.0, 0.0, 0.0)
