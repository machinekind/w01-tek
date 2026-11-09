"""These tests check timestamp arithmetic for /wojtek/io_timing without ROS."""

from types import SimpleNamespace

import pytest

from wojtek_bringup.timing import age_ms, stamp_ns


def _time(sec, nanosec):
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def test_stamp_ns_accepts_time_and_int():
    assert stamp_ns(_time(3, 250)) == 3_000_000_250
    assert stamp_ns(42) == 42


def test_age_across_a_second_boundary():
    assert age_ms(_time(11, 2_000_000), _time(10, 999_000_000)) == pytest.approx(3.0)


def test_age_mixes_int_and_time():
    assert age_ms(5_012_500_000, _time(5, 10_000_000)) == pytest.approx(2.5)


def test_age_is_negative_when_stamp_is_later():
    assert age_ms(_time(1, 0), _time(1, 1_000_000)) == pytest.approx(-1.0)


def test_large_epoch_stamps_keep_microsecond_precision():
    # Epoch timestamps retain microsecond precision.
    now = _time(1_762_000_000, 123_456_789)
    then = _time(1_762_000_000, 123_455_789)
    assert age_ms(now, then) == pytest.approx(0.001, abs=1e-9)
