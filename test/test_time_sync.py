# Copyright 2026 ZERO-SNU

import math

import pytest

from particle_filter.time_sync import (
    ClockEpochLatch,
    is_clock_epoch_reset,
    relative_planar_motion,
)


def test_relative_motion_is_expressed_in_previous_sensor_frame():
    # A rear-facing laser has yaw pi.  Vehicle-forward world motion therefore
    # appears as negative local laser-x and is rotated back correctly by MCL.
    dx, dy, dyaw = relative_planar_motion(
        0.0, 0.0, math.pi,
        1.0, 0.0, math.pi,
    )
    assert dx == pytest.approx(-1.0)
    assert dy == pytest.approx(0.0, abs=1e-12)
    assert dyaw == pytest.approx(0.0)


def test_relative_yaw_wraps_across_pi():
    _, _, dyaw = relative_planar_motion(
        0.0, 0.0, math.radians(179.0),
        0.0, 0.0, math.radians(-179.0),
    )
    assert dyaw == pytest.approx(math.radians(2.0))


def test_relative_translation_rotates_into_previous_frame():
    dx, dy, _ = relative_planar_motion(
        2.0, 3.0, math.pi / 2.0,
        2.0, 4.0, math.pi / 2.0,
    )
    assert dx == pytest.approx(1.0)
    assert dy == pytest.approx(0.0, abs=1e-12)


def test_small_backward_jump_is_only_out_of_order():
    assert not is_clock_epoch_reset(10_000_000_000, 9_950_000_000, 1.0)


def test_large_backward_jump_starts_new_clock_epoch():
    assert is_clock_epoch_reset(10_000_000_000, 8_000_000_000, 1.0)


def test_clock_epoch_fault_stays_latched_until_restart():
    guard = ClockEpochLatch(1.0)
    assert guard.observe(10_000_000_000, 10_100_000_000)
    assert not guard.observe(10_100_000_000, 8_000_000_000)
    assert guard.faulted
    assert not guard.observe(0, 20_000_000_000)

    restarted_guard = ClockEpochLatch(1.0)
    assert restarted_guard.observe(0, 20_000_000_000)
