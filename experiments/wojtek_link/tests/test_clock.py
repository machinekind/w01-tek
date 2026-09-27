"""The robot's clock offset read off the traffic. Pure."""

import pytest

from wojtek_link.clock import OffsetEstimator


def test_the_offset_is_the_fastest_delivery_over_the_window():
    est = OffsetEstimator(window_s=10.0)
    assert est.offset_s() is None
    # the robot's clock 2.0 s behind the DGX's, deliveries of 30, 8 and 15 ms
    for recv, delay in ((100.0, 0.030), (100.1, 0.008), (100.2, 0.015)):
        est.add(recv, recv - 2.0 - delay)
    assert est.offset_s() == pytest.approx(2.008)


def test_unstamped_messages_are_ignored():
    est = OffsetEstimator()
    est.add(100.0, 0.0)
    assert est.offset_s() is None


def test_old_samples_leave_the_window():
    est = OffsetEstimator(window_s=5.0)
    est.add(100.0, 100.0 - 0.001)      # a lucky fast one, long ago
    est.add(110.0, 110.0 - 0.050)
    assert est.offset_s() == pytest.approx(0.050)


def test_a_robot_clock_ahead_gives_a_negative_offset():
    est = OffsetEstimator()
    est.add(100.0, 100.5 - 0.004)       # the robot 0.5 s ahead
    assert est.offset_s() == pytest.approx(-0.496)
