# Copyright 2026 ZERO-SNU

import math


def relative_planar_motion(previous_x, previous_y, previous_yaw,
                           current_x, current_y, current_yaw):
    """Return current pose relative to the previous planar frame."""
    dx = current_x - previous_x
    dy = current_y - previous_y
    cos_yaw = math.cos(previous_yaw)
    sin_yaw = math.sin(previous_yaw)
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    raw_yaw = current_yaw - previous_yaw
    delta_yaw = math.atan2(math.sin(raw_yaw), math.cos(raw_yaw))
    return local_x, local_y, delta_yaw


class ClockEpochLatch:
    """Latch a backward ROS-time epoch change until process restart."""

    def __init__(self, reset_threshold_seconds):
        self.reset_threshold_seconds = reset_threshold_seconds
        self.faulted = False

    def observe(self, previous_stamp_ns, current_stamp_ns):
        if self.faulted:
            return False
        if is_clock_epoch_reset(
                previous_stamp_ns, current_stamp_ns,
                self.reset_threshold_seconds):
            self.faulted = True
            return False
        return True


def is_clock_epoch_reset(previous_stamp_ns, current_stamp_ns,
                         reset_threshold_seconds):
    return (previous_stamp_ns > 0 and current_stamp_ns < previous_stamp_ns and
            (previous_stamp_ns - current_stamp_ns) * 1e-9 >=
            reset_threshold_seconds)
