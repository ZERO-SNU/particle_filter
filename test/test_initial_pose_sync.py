# Copyright 2026 ZERO-SNU

from threading import Lock
from types import SimpleNamespace
import sys

import numpy as np

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, TransformStamped

# These tests exercise callback/state semantics without constructing the GPU
# range backend.  The real vehicle provides range_libc at runtime.
sys.modules.setdefault('range_libc', SimpleNamespace())

from particle_filter.particle_filter import ParticleFiler  # noqa: E402


class Recorder:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class Logger:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass

    def error(self, _message):
        pass


def test_operational_pose_is_published_when_visualization_is_disabled():
    fake = SimpleNamespace(
        inferred_pose=np.array([1.0, 2.0, 0.3]),
        estimate_stamp=Time(sec=12, nanosec=34),
        DO_VIZ=False,
        pose_pub=Recorder(),
        legacy_pose_pub=Recorder(),
        laser_pose_to_base_pose=lambda pose: pose,
    )

    ParticleFiler.visualize(fake)

    assert len(fake.pose_pub.messages) == 1
    assert fake.pose_pub.messages[0].header.frame_id == 'map'
    assert fake.pose_pub.messages[0].header.stamp.sec == 12
    assert fake.legacy_pose_pub.messages == []


def test_manual_pose_updates_tf_but_not_scan_synchronized_outputs():
    transform = TransformStamped()
    transform.header.stamp = Time(sec=20, nanosec=50)
    transform.transform.rotation.w = 1.0
    calls = []

    class TfBuffer:
        def lookup_transform(self, target, source, stamp, timeout):
            calls.append(('lookup', target, source, stamp, timeout))
            return transform

    fake = SimpleNamespace(
        MAX_PARTICLES=8,
        particles=np.zeros((8, 3)),
        weights=np.zeros(8),
        state_lock=Lock(),
        LASER_FRAME='laser',
        TF_LOOKUP_TIMEOUT=0.05,
        tf_buffer=TfBuffer(),
        get_logger=lambda: Logger(),
        _reset_scan_epoch=lambda: calls.append(('reset',)),
        publish_tf=lambda *args, **kwargs: calls.append(
            ('publish_tf', args, kwargs)),
        visualize=lambda **kwargs: calls.append(('visualize', kwargs)),
    )
    pose = Pose()
    pose.position.x = 3.0
    pose.position.y = -1.0
    pose.orientation.w = 1.0

    ParticleFiler.initialize_particles_pose(fake, pose)

    assert calls[0] == ('reset',)
    tf_call = next(call for call in calls if call[0] == 'publish_tf')
    assert tf_call[2]['publish_odom'] is False
    assert ('visualize', {'publish_operational_pose': False}) in calls
    assert fake.estimate_stamp.sec == 20
    assert np.allclose(fake.inferred_pose, [3.0, -1.0, 0.0])
    assert np.isclose(np.sum(fake.weights), 1.0)


def test_manual_pose_reset_discards_old_motion_and_health_baselines():
    parameters = {
        'jump_gate_margin_m': SimpleNamespace(value=0.5),
        'jump_gate_max_holds': SimpleNamespace(value=5),
    }
    fake = SimpleNamespace(
        last_motion_tf=object(),
        last_motion_stamp_ns=123,
        estimate_stamp=object(),
        inferred_pose=np.ones(3),
        _health_prev_inferred=np.ones(3),
        _gate_prev_odom=np.ones(3),
        get_parameter=lambda name: parameters[name],
    )

    ParticleFiler._reset_scan_epoch(fake)

    assert fake.last_motion_tf is None
    assert fake.last_motion_stamp_ns == 0
    assert fake.estimate_stamp is None
    assert fake.inferred_pose is None
    assert fake._health_prev_inferred is None
    assert fake._gate_prev_odom is None


def test_clicked_pose_accepts_only_valid_map_pose():
    accepted = []
    transform = TransformStamped()
    transform.transform.translation.x = 0.165
    transform.transform.rotation.z = 1.0
    transform.transform.rotation.w = 0.0
    fake = SimpleNamespace(
        clock_epoch_latch=SimpleNamespace(faulted=False),
        BASE_FRAME='base_link',
        LASER_FRAME='laser',
        TF_LOOKUP_TIMEOUT=0.05,
        tf_buffer=SimpleNamespace(
            lookup_transform=lambda *_args, **_kwargs: transform),
        get_logger=lambda: Logger(),
        base_pose_to_laser_pose=ParticleFiler.base_pose_to_laser_pose,
        initialize_particles_pose=lambda pose: accepted.append(pose),
    )
    message = PoseWithCovarianceStamped()
    message.header.frame_id = '/map'
    message.pose.pose.position.x = 1.0
    message.pose.pose.orientation.w = 1.0

    ParticleFiler.clicked_pose(fake, message)
    assert len(accepted) == 1
    assert np.isclose(accepted[0].position.x, 1.165)
    assert np.isclose(accepted[0].orientation.z, 1.0)

    message.header.frame_id = 'odom'
    ParticleFiler.clicked_pose(fake, message)
    assert len(accepted) == 1


def test_initial_pose_converts_vehicle_heading_to_laser_heading():
    base_pose = Pose()
    base_pose.position.x = 1.0
    base_pose.position.y = 2.0
    base_pose.orientation.w = 1.0
    base_to_laser = TransformStamped()
    base_to_laser.transform.translation.x = 0.165
    base_to_laser.transform.rotation.z = 1.0
    base_to_laser.transform.rotation.w = 0.0

    laser_pose = ParticleFiler.base_pose_to_laser_pose(base_pose, base_to_laser)

    assert np.isclose(laser_pose.position.x, 1.165)
    assert np.isclose(laser_pose.position.y, 2.0)
    assert np.isclose(laser_pose.orientation.z, 1.0)
    assert np.isclose(laser_pose.orientation.w, 0.0, atol=1e-6)


def test_operational_pose_converts_laser_heading_back_to_vehicle_heading():
    base_to_laser = TransformStamped()
    base_to_laser.transform.translation.x = 0.165
    base_to_laser.transform.rotation.z = 1.0
    base_to_laser.transform.rotation.w = 0.0

    base_pose = ParticleFiler.laser_pose_to_base_pose_with_tf(
        np.array([1.165, 2.0, np.pi]), base_to_laser)

    assert np.allclose(base_pose, [1.0, 2.0, 0.0], atol=1e-6)


def test_manual_pose_cannot_clear_latched_clock_fault():
    accepted = []
    fake = SimpleNamespace(
        clock_epoch_latch=SimpleNamespace(faulted=True),
        get_logger=lambda: Logger(),
        initialize_particles_pose=lambda pose: accepted.append(pose),
    )
    message = PoseWithCovarianceStamped()
    message.header.frame_id = 'map'
    message.pose.pose.orientation.w = 1.0

    ParticleFiler.clicked_pose(fake, message)
    assert accepted == []
