# MIT License

# Copyright (c) 2020 Hongrui Zheng, Corey Walsh

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ros2 python
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# libraries
import numpy as np
import range_libc
import time
from threading import Lock
from particle_filter import utils as Utils
import atexit
import os

# TF
# import tf.transformations
# import tf
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
import tf_transformations
from particle_filter.pf_health import n_eff_ratio, DriftWindow, JumpGate
from particle_filter.time_sync import ClockEpochLatch, relative_planar_motion

# messages
from std_msgs.msg import String, Header, Float32MultiArray
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseStamped,
    PoseArray,
    Quaternion,
    PolygonStamped,
    Polygon,
    Point32,
    PoseWithCovarianceStamped,
    PointStamped,
    TransformStamped,
)
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from ros2node.api import get_node_names # gym_bridge_launch.py 작동 시 라이다 180도 적용 코드 자동 제외
"""
These flags indicate several variants of the sensor model. Only one of them is used at a time.
"""
VAR_NO_EVAL_SENSOR_MODEL = 0
VAR_CALC_RANGE_MANY_EVAL_SENSOR = 1
VAR_REPEAT_ANGLES_EVAL_SENSOR = 2
VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT = 3
VAR_RADIAL_CDDT_OPTIMIZATIONS = 4

logger_file = open((os.path.expanduser("~")+time.strftime('/wp-%Y-%m-%d-%H-%M-%S',time.gmtime())+".csv"),'w')
logger_file.write('# x_m,y_m,yaw_rad\n')


class ParticleFiler(Node):
    """
    This class implements Monte Carlo Localization based on odometry and a laser scanner.
    """

    def __init__(self):
        super().__init__("particle_filter")

        # declare parameters
        self.declare_parameter("angle_step")
        self.declare_parameter("max_particles")
        self.declare_parameter("max_viz_particles")
        self.declare_parameter("squash_factor")
        self.declare_parameter("max_range")
        self.declare_parameter("theta_discretization")
        self.declare_parameter("range_method")
        self.declare_parameter("rangelib_variant")
        self.declare_parameter("fine_timing")
        self.declare_parameter("publish_odom")
        self.declare_parameter("viz")
        self.declare_parameter("sim_mode", False)
        # 스캔 배열 반전(구 'for RPLiDAR' 코드). 이 차량(urg/Hokuyo 270° FOV)은 불필요 —
        # 켜면 절반-스왑이 180°가 아닌 ~135° 각도 왜곡이 되어 위치추정이 발산한다 (2026-07-23 실차 사고).
        # RPLiDAR를 180° 뒤집어 장착한 차량에서만 true.
        self.declare_parameter("scan_rotate_180", False)
        self.declare_parameter("z_short")
        self.declare_parameter("z_max")
        self.declare_parameter("z_rand")
        self.declare_parameter("z_hit")
        self.declare_parameter("sigma_hit")
        self.declare_parameter("motion_dispersion_x")
        self.declare_parameter("motion_dispersion_y")
        self.declare_parameter("motion_dispersion_theta")
        self.declare_parameter("scan_topic")
        self.declare_parameter("odometry_topic")
        self.declare_parameter("laser_frame", "laser")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tf_lookup_timeout", 0.05)
        self.declare_parameter("max_future_stamp", 0.05)
        self.declare_parameter("clock_reset_threshold", 1.0)
        # [P0-3] 헬스 지표 발행 — 주행 로직 무영향(발행만). Phase 3 FSM·재보정 감지 입력.
        self.declare_parameter("publish_health", True)
        # [08-24] N_eff비가 이 값 아래로 내려가면 발산 의심 경고. 0 = 끔.
        self.declare_parameter("health_warn_neff", 0.10)
        self.declare_parameter("health_window_m", 5.0)
        # [P0-5] pose 점프 게이트 — 물리 불가능 점프(odom 이동량+margin 초과) 시
        # last-good pose를 유지 발행. false = 끔(기존 동작). 파티클·가중치는 건드리지
        # 않음(발행 pose만 게이트) — 연속 max_holds 초과 시 새 추정 수용(재수렴 허용).
        self.declare_parameter("jump_gate_enable", False)
        self.declare_parameter("jump_gate_margin_m", 0.5)
        self.declare_parameter("jump_gate_max_holds", 5)
        # [08-24] hold 중 last-good 처리: 'freeze'(구 동작, 기본) | 'dead_reckon'.
        # freeze 는 hold 가 시작되면 차가 움직이는 한 max_holds 까지 반드시 얼고 누적분을
        # 한 번에 점프한다(car1 4.3 m/s: 0.4 s 동결 → 2.93 m 순간이동). dead_reckon 은
        # hold 중 발행 pose 를 odom 으로 전진시켜 동결이 없고 풀릴 때 점프가 원래 오차만큼만.
        # 차량별 거동 보존을 위해 기본은 freeze — car1 프로파일(particle_filter: 섹션)에서 켠다.
        self.declare_parameter("jump_gate_hold_mode", "freeze")

        # parameters
        self.ANGLE_STEP = self.get_parameter("angle_step").value
        self.MAX_PARTICLES = self.get_parameter("max_particles").value
        self.MAX_VIZ_PARTICLES = self.get_parameter("max_viz_particles").value
        self.INV_SQUASH_FACTOR = 1.0 / self.get_parameter("squash_factor").value
        self.MAX_RANGE_METERS = self.get_parameter("max_range").value
        self.THETA_DISCRETIZATION = self.get_parameter("theta_discretization").value
        self.WHICH_RM = self.get_parameter("range_method").value
        self.RANGELIB_VAR = self.get_parameter("rangelib_variant").value
        self.SHOW_FINE_TIMING = self.get_parameter("fine_timing").value
        self.PUBLISH_ODOM = self.get_parameter("publish_odom").value
        self.DO_VIZ = self.get_parameter("viz").value
        self.SIM_MODE = self.get_parameter("sim_mode").value
        self.SCAN_ROTATE_180 = self.get_parameter("scan_rotate_180").value
        self.LASER_FRAME = str(self.get_parameter("laser_frame").value)
        self.BASE_FRAME = str(self.get_parameter("base_frame").value)
        self.TF_LOOKUP_TIMEOUT = float(self.get_parameter("tf_lookup_timeout").value)
        self.MAX_FUTURE_STAMP = float(self.get_parameter("max_future_stamp").value)
        self.clock_epoch_latch = ClockEpochLatch(
            float(self.get_parameter("clock_reset_threshold").value))

        # sensor model constants
        self.Z_SHORT = self.get_parameter("z_short").value
        self.Z_MAX = self.get_parameter("z_max").value
        self.Z_RAND = self.get_parameter("z_rand").value
        self.Z_HIT = self.get_parameter("z_hit").value
        self.SIGMA_HIT = self.get_parameter("sigma_hit").value

        # motion model constants
        self.PUBLISH_HEALTH = bool(self.get_parameter("publish_health").value)
        self.HEALTH_WARN_NEFF = float(self.get_parameter("health_warn_neff").value)
        self._health_window = DriftWindow(float(self.get_parameter("health_window_m").value))
        self.JUMP_GATE_ENABLE = bool(self.get_parameter("jump_gate_enable").value)
        self._jump_gate = JumpGate(
            float(self.get_parameter("jump_gate_margin_m").value),
            int(self.get_parameter("jump_gate_max_holds").value),
            str(self.get_parameter("jump_gate_hold_mode").value))
        self._gate_prev_odom = None
        self._health_prev_inferred = None
        self._health_raw_w_mean = 0.0
        self.MOTION_DISPERSION_X = self.get_parameter("motion_dispersion_x").value
        self.MOTION_DISPERSION_Y = self.get_parameter("motion_dispersion_y").value
        self.MOTION_DISPERSION_THETA = self.get_parameter(
            "motion_dispersion_theta"
        ).value

        # various data containers used in the MCL algorithm
        self.MAX_RANGE_PX = None
        self.odometry_data = np.array([0.0, 0.0, 0.0])
        self.laser = None
        self.iters = 0
        self.map_info = None
        self.map_initialized = False
        self.lidar_initialized = False
        self.odom_initialized = False
        self.last_pose = None
        self.laser_angles = None
        self.downsampled_angles = None
        self.range_method = None
        self.last_time = None
        self.last_stamp = None
        self.estimate_stamp = None
        self.last_motion_tf = None
        self.last_motion_stamp_ns = 0
        self._last_tf_warning = 0.0
        self._last_latency_log = 0.0
        self.first_sensor_update = True
        self.state_lock = Lock()
        # The GPU scan callback can run continuously.  Keep RViz initial-pose
        # delivery on the executor's second thread; the existing particle
        # initialization itself remains serialized by state_lock.
        self.manual_reset_group = MutuallyExclusiveCallbackGroup()

        # cache this to avoid memory allocation in motion model
        self.local_deltas = np.zeros((self.MAX_PARTICLES, 3))

        # cache this for the sensor model computation
        self.queries = None
        self.ranges = None
        self.tiled_angles = None
        self.sensor_model_table = None

        # particle poses and weights
        self.inferred_pose = None
        self.particle_indices = np.arange(self.MAX_PARTICLES)
        self.particles = np.zeros((self.MAX_PARTICLES, 3))
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)

        # [Orin 최적화] systematic resampling 재사용 버퍼 — 매 스텝 신규 할당 제거 (08-15 실차 검증: 1.106→0.372 ms)
        self._resample_base = (
            np.arange(self.MAX_PARTICLES, dtype=np.float64)
            / float(self.MAX_PARTICLES)
        )
        self._resample_positions = np.empty(self.MAX_PARTICLES, dtype=np.float64)
        self._resample_cdf = np.empty(self.MAX_PARTICLES, dtype=np.float64)
        self._resample_particles = np.empty_like(self.particles)

        # initialize the state
        self.smoothing = Utils.CircularArray(10)
        self.timer = Utils.Timer(10)
        # map service client
        self.map_client = self.create_client(GetMap, "/map_server/map")
        self.get_omap()
        # [Orin 최적화] fused GPU 센서모델(variant 3) capability probe — 기동 시 1회 판정.
        # rmgpu + fused 지원 .so 조합에서만 활성. 그 외(스톡 .so, CPU 방식)는 variant 2로 강등해
        # 런타임 매 스텝 실패-재계산 회귀를 차단한다.
        if self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT:
            if self.WHICH_RM == "rmgpu":
                supports_fused = getattr(
                    self.range_method, "supports_fused_sensor_model", None
                )
                if supports_fused is None or not supports_fused():
                    self.get_logger().warning(
                        "Installed RangeLib lacks the fused GPU sensor model; "
                        "falling back to rangelib_variant 2"
                    )
                    self.RANGELIB_VAR = VAR_REPEAT_ANGLES_EVAL_SENSOR
                else:
                    self.get_logger().info(
                        "Using fused GPU ray-casting, likelihood, and weight reduction"
                    )
            else:
                self.get_logger().warning(
                    "rangelib_variant 3 (fused) is rmgpu-only; range_method=%s → "
                    "falling back to rangelib_variant 2" % self.WHICH_RM
                )
                self.RANGELIB_VAR = VAR_REPEAT_ANGLES_EVAL_SENSOR
        self.precompute_sensor_model()
        self.initialize_global()

        # keep track of speed from input odom
        self.current_speed = 0.0

        # Pub Subs
        # these topics are for visualization
        self.pose_pub = self.create_publisher(PoseStamped, "/pf/pose", 1)
        self.legacy_pose_pub = self.create_publisher(
            PoseStamped, "/pf/viz/inferred_pose", 1)
        self.particle_pub = self.create_publisher(PoseArray, "/pf/viz/particles", 1)
        self.pub_fake_scan = self.create_publisher(LaserScan, "/pf/viz/fake_scan", 1)
        self.rect_pub = self.create_publisher(PolygonStamped, "/pf/viz/poly1", 1)
        
        # 환경(sim/real) 판별은 sim_mode 파라미터로만 한다 (P2: 코드 내 자동감지 금지).
        # 시뮬 launch가 sim_mode:=true 를 주입한다 (기본 False = 실차).

        if self.PUBLISH_ODOM:
            self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # these topics are for coordinate space things
        # [08-17 TF 세트] map→odom 합성에 필요한 odom→laser 실시간 lookup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub_tf = TransformBroadcaster(self)

        # these topics are to receive data from the racecar
        # [P0-3] 헬스 지표 발행자
        if self.PUBLISH_HEALTH:
            self.health_pub = self.create_publisher(Float32MultiArray, "/pf/health", 1)
        self.laser_sub = self.create_subscription(
            LaserScan, self.get_parameter("scan_topic").value, self.lidarCB, 1
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.get_parameter("odometry_topic").value, self.odomCB2, 1
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.clicked_pose, 1,
            callback_group=self.manual_reset_group
        )
        self.click_sub = self.create_subscription(
            PointStamped, "/clicked_point", self.clicked_pose, 1,
            callback_group=self.manual_reset_group
        )

        self.get_logger().info("Finished initializing, waiting on messages...")

    def get_omap(self):
        """
        Fetch the occupancy grid map from the map_server instance, and initialize the correct
        RangeLibc method. Also stores a matrix which indicates the permissible region of the map
        """

        while not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Get map service not available, waiting...")
        req = GetMap.Request()
        future = self.map_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        map_msg = future.result().map
        self.map_info = map_msg.info

        oMap = range_libc.PyOMap(map_msg)
        self.MAX_RANGE_PX = int(self.MAX_RANGE_METERS / self.map_info.resolution)

        # initialize range method
        self.get_logger().info("Initializing range method: " + self.WHICH_RM)
        if self.WHICH_RM == "bl":
            self.range_method = range_libc.PyBresenhamsLine(oMap, self.MAX_RANGE_PX)
        elif "cddt" in self.WHICH_RM:
            self.range_method = range_libc.PyCDDTCast(
                oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION
            )
            if self.WHICH_RM == "pcddt":
                self.get_logger().info("Pruning...")
                self.range_method.prune()
        elif self.WHICH_RM == "rm":
            self.range_method = range_libc.PyRayMarching(oMap, self.MAX_RANGE_PX)
        elif self.WHICH_RM == "rmgpu":
            # [08-24] GPU 레이마칭 실패 시 **CPU 로 안전 강등**한다.
            # 이전에는 예외가 그대로 올라가 PF 가 아예 안 떴다 — 대회장에서 GPU rangelib
            # 이 빌드돼 있지 않거나 CUDA 가 안 잡히면 측위 자체가 죽는다는 뜻이다.
            # 강등은 조용히 하지 않는다(성능이 크게 달라지므로 반드시 로그로 남긴다).
            try:
                self.range_method = range_libc.PyRayMarchingGPU(oMap, self.MAX_RANGE_PX)
            except Exception as ex:                              # noqa: BLE001
                self.get_logger().error(
                    'rmgpu(PyRayMarchingGPU) 초기화 실패 — CPU rm 으로 강등한다: %r. '
                    'GPU 를 쓰려면 scripts/install_rangelib_orin_sm87.sh 로 range_libc 를 '
                    '다시 빌드할 것. 강등 상태에서는 max_particles·angle_step 을 GPU 값 '
                    '그대로 두면 PF 가 스캔 속도를 못 따라간다.' % (ex,))
                self.WHICH_RM = "rm"
                self.range_method = range_libc.PyRayMarching(oMap, self.MAX_RANGE_PX)
        elif self.WHICH_RM == "glt":
            self.range_method = range_libc.PyGiantLUTCast(
                oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION
            )
        self.get_logger().info("Done loading map")

        # 0: permissible, -1: unmapped, 100: blocked
        array_255 = np.array(map_msg.data).reshape(
            (map_msg.info.height, map_msg.info.width)
        )

        # 0: not permissible, 1: permissible
        self.permissible_region = np.zeros_like(array_255, dtype=bool)
        self.permissible_region[array_255 == 0] = 1
        self.map_initialized = True

    def publish_tf(self, pose, stamp=None, odom_to_base=None,
                   publish_odom=True):
        """Publish map->odom from the PF's map->base_link estimate."""
        if self.clock_epoch_latch.faulted:
            return
        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        if odom_to_base is None:
            try:
                odom_to_base = self.tf_buffer.lookup_transform(
                    'odom', self.BASE_FRAME,
                    rclpy.time.Time.from_msg(stamp),
                    timeout=Duration(seconds=self.TF_LOOKUP_TIMEOUT))
            except TransformException as ex:
                self.get_logger().warn(
                    'Cannot publish map -> odom until scan-time odom -> base_link is available: %s' % ex)
                return

        q = odom_to_base.transform.rotation
        odom_base_yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w])[2]
        map_odom_yaw = pose[2] - odom_base_yaw
        odom_base_x = odom_to_base.transform.translation.x
        odom_base_y = odom_to_base.transform.translation.y

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        # T_map_odom = T_map_base * inverse(T_odom_base).
        t.transform.translation.x = pose[0] - (
            np.cos(map_odom_yaw) * odom_base_x -
            np.sin(map_odom_yaw) * odom_base_y)
        t.transform.translation.y = pose[1] - (
            np.sin(map_odom_yaw) * odom_base_x +
            np.cos(map_odom_yaw) * odom_base_y)
        t.transform.translation.z = 0.0
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, map_odom_yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.pub_tf.sendTransform(t)
        # also publish odometry to facilitate getting the localization pose
        if self.PUBLISH_ODOM and publish_odom:
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = "map"
            odom.pose.pose.position.x = pose[0]
            odom.pose.pose.position.y = pose[1]
            odom.pose.pose.orientation = Utils.angle_to_quaternion(pose[2])
            cov_mat = np.cov(
                self.particles, rowvar=False, ddof=0, aweights=self.weights
            ).flatten()
            odom.pose.covariance[: cov_mat.shape[0]] = cov_mat
            odom.twist.twist.linear.x = self.current_speed
            self.odom_pub.publish(odom)

        return

    def visualize(self, publish_operational_pose=True):
        """
        Publish various visualization messages.

        ``/pf/pose`` is an operational, scan-synchronized topic.  A manual
        ``/initialpose`` may update RViz immediately, but must not inject a
        pose stamp for which lattice has no obstacle-map snapshot.
        """
        if isinstance(self.inferred_pose, np.ndarray):
            ps = PoseStamped()
            ps.header.stamp = self.estimate_stamp
            ps.header.frame_id = "map"
            ps.pose.position.x = self.inferred_pose[0]
            ps.pose.position.y = self.inferred_pose[1]
            ps.pose.orientation = Utils.angle_to_quaternion(self.inferred_pose[2])
            if publish_operational_pose:
                # Never subscription-gate operational state.
                self.pose_pub.publish(ps)
            if self.DO_VIZ:
                self.legacy_pose_pub.publish(ps)

        if not self.DO_VIZ:
            return

        if self.particle_pub.get_subscription_count() > 0:
            # publish a downsampled version of the particle distribution to avoid a lot of latency
            if self.MAX_PARTICLES > self.MAX_VIZ_PARTICLES:
                # randomly downsample particles
                proposal_indices = np.random.choice(
                    self.particle_indices, self.MAX_VIZ_PARTICLES, p=self.weights
                )
                # proposal_indices = np.random.choice(self.particle_indices, self.MAX_VIZ_PARTICLES)
                self.publish_particles(self.particles[proposal_indices, :])
            else:
                self.publish_particles(self.particles)

        if self.pub_fake_scan.get_subscription_count() > 0 and isinstance(
            self.ranges, np.ndarray
        ):
            # Preserve the proven vehicle-pose sensor convention used by the
            # existing map/scan calibration.
            self.viz_queries[:, 0] = self.inferred_pose[0]
            self.viz_queries[:, 1] = self.inferred_pose[1]
            self.viz_queries[:, 2] = self.downsampled_angles + self.inferred_pose[2]
            self.range_method.calc_range_many(self.viz_queries, self.viz_ranges)
            self.publish_scan(self.downsampled_angles, self.viz_ranges)

    def publish_particles(self, particles):
        # publish the given particles as a PoseArray object
        pa = PoseArray()
        pa.header.stamp = self.estimate_stamp
        pa.header.frame_id = "map"
        pa.poses = Utils.particles_to_poses(particles)
        self.particle_pub.publish(pa)

    def publish_scan(self, angles, ranges):
        # publish the given angels and ranges as a laser scan message
        ls = LaserScan()
        ls.header.stamp = self.estimate_stamp
        # 08-17 TF 세트: /laser_pf 프레임 폐지 — 물리 laser 프레임 기준으로 발행
        ls.header.frame_id = self.LASER_FRAME
        ls.angle_min = np.min(angles)
        ls.angle_max = np.max(angles)
        ls.angle_increment = np.abs(angles[0] - angles[1])
        ls.range_min = 0
        ls.range_max = np.max(ranges)
        ls.ranges = ranges
        self.pub_fake_scan.publish(ls)

    def lidarCB(self, msg):
        """
        Initializes reused buffers, and stores the relevant laser scanner data for later use.
        """
        if msg.header.frame_id.lstrip('/') != self.LASER_FRAME:
            self.get_logger().warn(
                "Skipping scan frame '%s'; expected '%s'" % (
                    msg.header.frame_id, self.LASER_FRAME))
            return
        if not isinstance(self.laser_angles, np.ndarray):
            self.get_logger().info("...Received first LiDAR message")
            self.laser_angles = np.linspace(
                msg.angle_min, msg.angle_max, len(msg.ranges)
            )
            self.downsampled_angles = np.copy(
                self.laser_angles[0 :: self.ANGLE_STEP]
            ).astype(np.float32)
            self.viz_queries = np.zeros(
                (self.downsampled_angles.shape[0], 3), dtype=np.float32
            )
            self.viz_ranges = np.zeros(
                self.downsampled_angles.shape[0], dtype=np.float32
            )
            self.get_logger().info(str(self.downsampled_angles.shape[0]))

        # store the necessary scanner information for later processing
        self.downsampled_ranges = np.array(msg.ranges[:: self.ANGLE_STEP])
        # 스캔 배열 반전은 sim/real이 아니라 '장착' 문제 — scan_rotate_180 파라미터로만 결정.
        # 기존 구형 차량 코드에서 half-swap이 360° 스캔에만 맞으며,
        # 270° FOV에서는 약 135°로 왜곡되어 localization을 발산시킨다.
        if self.SCAN_ROTATE_180:
            # rotate the downsampled scan by 180° in angle space
            angle_step = msg.angle_increment * self.ANGLE_STEP
            shift = int(np.round(np.pi / angle_step))
            self.downsampled_ranges = np.roll(self.downsampled_ranges, shift)
            if not self.lidar_initialized:  # 매 스캔 로그 스팸 방지 — 최초 1회만
                self.get_logger().info(
                    f"scan_rotate_180: rotating downsampled scan by {shift} indices"
                )
        self.lidar_initialized = True
        self.update(msg.header.stamp)

    def odomCB(self, msg):
        """
        Store deltas between consecutive odometry messages in the coordinate space of the car.

        Odometry data is accumulated via dead reckoning, so it is very inaccurate on its own.
        """
        position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        orientation = Utils.quaternion_to_angle(msg.pose.pose.orientation)
        pose = np.array([position[0], position[1], orientation])
        self.current_speed = msg.twist.twist.linear.x

        if isinstance(self.last_pose, np.ndarray):
            # changes in x,y,theta in local coordinate system of the car
            rot = Utils.rotation_matrix(-self.last_pose[2])
            delta = np.array([position - self.last_pose[0:2]]).transpose()
            local_delta = (rot * delta).transpose()

            self.odometry_data = np.array(
                [local_delta[0, 0], local_delta[0, 1], orientation - self.last_pose[2]]
            )
            self.last_pose = pose
            self.last_stamp = msg.header.stamp
            self.odom_initialized = True
        else:
            self.get_logger().info("...Received first Odometry message")
            self.last_pose = pose

        # this topic is slower than lidar, so update every time we receive a message
        # self.update()

    def odomCB2(self, msg):
        """
        Cache odometry health and speed.  Particle motion is applied once in
        update() from exact odom->base_link transforms at scan timestamps.
        """
        if self.clock_epoch_latch.faulted:
            return
        self.current_speed = msg.twist.twist.linear.x
        self.last_stamp = msg.header.stamp
        if not self.odom_initialized:
            self.get_logger().info("...Received first Odometry message CB2")
        self.odom_initialized = True

    def clicked_pose(self, msg):
        """
        Receive pose messages from RViz and initialize the particle distribution in response.
        """
        if self.clock_epoch_latch.faulted:
            self.get_logger().error(
                'Ignoring manual initialization after a clock epoch fault; '
                'restart PF, lattice_planner, and Pure Pursuit')
            return
        if isinstance(msg, PointStamped):
            self.initialize_global()
        elif isinstance(msg, PoseWithCovarianceStamped):
            frame = msg.header.frame_id.lstrip('/')
            pose = msg.pose.pose
            values = np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w], dtype=float)
            quaternion_norm = float(np.linalg.norm(values[3:]))
            if frame != 'map':
                self.get_logger().error(
                    "Ignoring 2D Pose Estimate in frame '%s'; expected 'map'" %
                    msg.header.frame_id)
                return
            if not np.all(np.isfinite(values)) or quaternion_norm < 1e-6:
                self.get_logger().error(
                    'Ignoring 2D Pose Estimate with a non-finite or invalid pose')
                return
            self.get_logger().info(
                '2D Pose Estimate accepted as map->%s seed' % self.BASE_FRAME)
            self.initialize_particles_pose(pose)

    def _reset_scan_epoch(self):
        """A manual relocalization must not reuse motion from the old pose."""
        self.last_motion_tf = None
        self.last_motion_stamp_ns = 0
        self.estimate_stamp = None
        self.inferred_pose = None
        self._health_prev_inferred = None
        self._gate_prev_odom = None
        self._jump_gate = JumpGate(
            float(self.get_parameter('jump_gate_margin_m').value),
            int(self.get_parameter('jump_gate_max_holds').value),
            str(self.get_parameter('jump_gate_hold_mode').value))

    def initialize_particles_pose(self, pose):
        """
        Initialize particles in the general region of the provided pose.
        """
        self.get_logger().info("SETTING POSE")
        self.get_logger().info(str([pose.position.x, pose.position.y]))
        yaw = Utils.quaternion_to_angle(pose.orientation)
        with self.state_lock:
            # This is the original direct RViz seed: no frame conversion or
            # artificial hold.  Lock the whole reset/publication transaction
            # so a scan cannot overwrite it halfway through.
            self._reset_scan_epoch()
            self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
            self.particles[:, 0] = pose.position.x + np.random.normal(
                loc=0.0, scale=0.5, size=self.MAX_PARTICLES
            )
            self.particles[:, 1] = pose.position.y + np.random.normal(
                loc=0.0, scale=0.5, size=self.MAX_PARTICLES
            )
            self.particles[:, 2] = yaw + np.random.normal(
                loc=0.0, scale=0.4, size=self.MAX_PARTICLES)
            # A manual RViz estimate is an explicit operator reset, not a scan
            # projection.  Show it immediately using the freshest available
            # odom->base_link transform; normal PF/dynamic-map processing still
            # uses exact scan timestamps only.
            self.inferred_pose = np.array([
                pose.position.x,
                pose.position.y,
                yaw])
            try:
                odom_to_base = self.tf_buffer.lookup_transform(
                    'odom', self.BASE_FRAME, rclpy.time.Time(),
                    timeout=Duration(seconds=self.TF_LOOKUP_TIMEOUT))
            except TransformException as ex:
                self.get_logger().warn(
                    '2D Pose Estimate stored, but odom -> base_link is unavailable: %s' % ex)
                return
            self.estimate_stamp = odom_to_base.header.stamp
            # This latest-TF lookup is isolated to the explicit operator reset.
            # Do not publish operational pose/odom until an exact scan epoch has
            # produced the matching PF/map snapshot.
            self.publish_tf(
                self.inferred_pose, self.estimate_stamp, odom_to_base,
                publish_odom=False)
            self.visualize(publish_operational_pose=False)

    def initialize_global(self):
        """
        Spread the particle distribution over the permissible region of the state space.
        """
        self._reset_scan_epoch()
        self.get_logger().info("GLOBAL INITIALIZATION")
        # randomize over grid coordinate space
        self.state_lock.acquire()
        permissible_x, permissible_y = np.where(self.permissible_region == 1)
        indices = np.random.randint(0, len(permissible_x), size=self.MAX_PARTICLES)

        permissible_states = np.zeros((self.MAX_PARTICLES, 3))
        permissible_states[:, 0] = permissible_y[indices]
        permissible_states[:, 1] = permissible_x[indices]
        permissible_states[:, 2] = np.random.random(self.MAX_PARTICLES) * np.pi * 2.0

        Utils.map_to_world(permissible_states, self.map_info)
        self.particles = permissible_states
        self.weights[:] = 1.0 / self.MAX_PARTICLES
        self.state_lock.release()

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model. For each discrete computed
        range value, this provides the probability of measuring any (discrete) range.

        This table is indexed by the sensor model at runtime by discretizing the measurements
        and computed ranges from RangeLibc.
        """
        self.get_logger().info("Precomputing sensor model")
        # sensor model constants
        z_short = self.Z_SHORT
        z_max = self.Z_MAX
        z_rand = self.Z_RAND
        z_hit = self.Z_HIT
        sigma_hit = self.SIGMA_HIT

        table_width = int(self.MAX_RANGE_PX) + 1
        self.sensor_model_table = np.zeros((table_width, table_width))

        t = time.time()
        # d is the computed range from RangeLibc
        for d in range(table_width):
            norm = 0.0
            sum_unkown = 0.0
            # r is the observed range from the lidar unit
            for r in range(table_width):
                prob = 0.0
                z = float(r - d)
                # reflects from the intended object
                prob += (
                    z_hit
                    * np.exp(-(z * z) / (2.0 * sigma_hit * sigma_hit))
                    / (sigma_hit * np.sqrt(2.0 * np.pi))
                )

                # observed range is less than the predicted range - short reading
                if r < d:
                    prob += 2.0 * z_short * (d - r) / float(d)

                # erroneous max range measurement
                if int(r) == int(self.MAX_RANGE_PX):
                    prob += z_max

                # random measurement
                if r < int(self.MAX_RANGE_PX):
                    prob += z_rand * 1.0 / float(self.MAX_RANGE_PX)

                norm += prob
                self.sensor_model_table[int(r), int(d)] = prob

            # normalize
            self.sensor_model_table[:, int(d)] /= norm

        # upload the sensor model to RangeLib for ultra fast resolution
        if self.RANGELIB_VAR > 0:
            self.range_method.set_sensor_model(self.sensor_model_table)

    def motion_model(self, proposal_dist, action):
        """
        The motion model applies the odometry to the particle distribution. Since there the odometry
        data is inaccurate, the motion model mixes in gaussian noise to spread out the distribution.

        Vectorized motion model. Computing the motion model over all particles is thousands of times
        faster than doing it for each particle individually due to vectorization and reduction in
        function call overhead

        TODO this could be better, but it works for now
            - fixed random noise is not very realistic
            - ackermann model provides bad estimates at high speed
        """
        # rotate the action into the coordinate space of each particle
        # t1 = time.time()
        cosines = np.cos(proposal_dist[:, 2])
        sines = np.sin(proposal_dist[:, 2])

        self.local_deltas[:, 0] = cosines * action[0] - sines * action[1]
        self.local_deltas[:, 1] = sines * action[0] + cosines * action[1]
        self.local_deltas[:, 2] = action[2]

        proposal_dist[:, :] += self.local_deltas
        proposal_dist[:, 0] += np.random.normal(
            loc=0.0, scale=self.MOTION_DISPERSION_X, size=self.MAX_PARTICLES
        )
        proposal_dist[:, 1] += np.random.normal(
            loc=0.0, scale=self.MOTION_DISPERSION_Y, size=self.MAX_PARTICLES
        )
        proposal_dist[:, 2] += np.random.normal(
            loc=0.0, scale=self.MOTION_DISPERSION_THETA, size=self.MAX_PARTICLES
        )

    def sensor_model(self, proposal_dist, obs, weights):
        """
        This function computes a probablistic weight for each particle in the proposal distribution.
        These weights represent how probable each proposed (x,y,theta) pose is given the measured
        ranges from the lidar scanner.

        There are 4 different variants using various features of RangeLibc for demonstration purposes.
        - VAR_REPEAT_ANGLES_EVAL_SENSOR is the most stable, and is very fast.
        - VAR_NO_EVAL_SENSOR_MODEL directly indexes the precomputed sensor model. This is slow
                                   but it demonstrates what self.range_method.eval_sensor_model does
        - VAR_RADIAL_CDDT_OPTIMIZATIONS is only compatible with CDDT or PCDDT, it implments the radial
                                        optimizations to CDDT which simultaneously performs ray casting
                                        in two directions, reducing the amount of work by roughly a third
        """

        num_rays = self.downsampled_angles.shape[0]
        # Existing map/scan calibration treats particle heading as the raycast
        # heading.  Do not add the static laser yaw here: doing so applies pi
        # twice and flips the estimate on the first scan after /initialpose.
        sensor_dist = proposal_dist
        # only allocate buffers once to avoid slowness
        if self.first_sensor_update:
            if self.RANGELIB_VAR <= 1:
                self.queries = np.zeros(
                    (num_rays * self.MAX_PARTICLES, 3), dtype=np.float32
                )
            else:
                self.queries = np.zeros((self.MAX_PARTICLES, 3), dtype=np.float32)

            self.ranges = np.zeros(num_rays * self.MAX_PARTICLES, dtype=np.float32)
            self.tiled_angles = np.tile(self.downsampled_angles, self.MAX_PARTICLES)
            self.first_sensor_update = False

        if self.RANGELIB_VAR == VAR_RADIAL_CDDT_OPTIMIZATIONS:
            if "cddt" in self.WHICH_RM:
                self.queries[:, :] = sensor_dist[:, :]
                self.range_method.calc_range_many_radial_optimized(
                    num_rays,
                    self.downsampled_angles[0],
                    self.downsampled_angles[-1],
                    self.queries,
                    self.ranges,
                )

                # evaluate the sensor model
                self.range_method.eval_sensor_model(
                    obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES
                )
                # apply the squash factor
                self.weights = np.power(self.weights, self.INV_SQUASH_FACTOR)
            else:
                self.get_logger().info(
                    "Cannot use radial optimizations with non-CDDT based methods, use rangelib_variant 2"
                )
        elif self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT:
            if self.SHOW_FINE_TIMING:
                t_start = time.time()
            self.queries[:, :] = sensor_dist[:, :]
            # fused GPU 경로 — 실패(bool False) 시 variant 2로 영구 강등 후 그 스텝은 재계산(정확성 보존)
            fused_ok = self.range_method.calc_range_repeat_angles_eval_sensor_model(
                self.queries, self.downsampled_angles, obs, self.weights
            )
            if not fused_ok:
                self.get_logger().error(
                    "Fused GPU sensor model failed; switching to variant 2"
                )
                self.RANGELIB_VAR = VAR_REPEAT_ANGLES_EVAL_SENSOR
                self.range_method.calc_range_repeat_angles(
                    self.queries, self.downsampled_angles, self.ranges
                )
                self.range_method.eval_sensor_model(
                    obs,
                    self.ranges,
                    self.weights,
                    num_rays,
                    self.MAX_PARTICLES,
                )
            if self.SHOW_FINE_TIMING:
                t_fused = time.time()
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
            if self.SHOW_FINE_TIMING:
                t_squash = time.time()
                t_total = (t_squash - t_start) / 100.0
            if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
                self.get_logger().info(
                    str(
                        [
                            "sensor_model fused:",
                            np.round((t_fused - t_start) / t_total, 2),
                            "squash:",
                            np.round((t_squash - t_fused) / t_total, 2),
                        ]
                    )
                )
        elif self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR:
            if self.SHOW_FINE_TIMING:
                t_start = time.time()
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            self.queries[:, :] = sensor_dist[:, :]
            if self.SHOW_FINE_TIMING:
                t_init = time.time()
            self.range_method.calc_range_repeat_angles(
                self.queries, self.downsampled_angles, self.ranges
            )
            if self.SHOW_FINE_TIMING:
                t_range = time.time()
            # evaluate the sensor model on the GPU
            self.range_method.eval_sensor_model(
                obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES
            )
            if self.SHOW_FINE_TIMING:
                t_eval = time.time()
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
            if self.SHOW_FINE_TIMING:
                t_squash = time.time()
                t_total = (t_squash - t_start) / 100.0

            if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
                self.get_logger().info(
                    str(
                        [
                            "sensor_model: init: ",
                            np.round((t_init - t_start) / t_total, 2),
                            "range:",
                            np.round((t_range - t_init) / t_total, 2),
                            "eval:",
                            np.round((t_eval - t_range) / t_total, 2),
                            "squash:",
                            np.round((t_squash - t_eval) / t_total, 2),
                        ]
                    )
                )
        elif self.RANGELIB_VAR == VAR_CALC_RANGE_MANY_EVAL_SENSOR:
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            # this part is inefficient since it requires a lot of effort to construct this redundant array
            self.queries[:, 0] = np.repeat(sensor_dist[:, 0], num_rays)
            self.queries[:, 1] = np.repeat(sensor_dist[:, 1], num_rays)
            self.queries[:, 2] = np.repeat(sensor_dist[:, 2], num_rays)
            self.queries[:, 2] += self.tiled_angles

            self.range_method.calc_range_many(self.queries, self.ranges)

            # evaluate the sensor model on the GPU
            self.range_method.eval_sensor_model(
                obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES
            )
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
        elif self.RANGELIB_VAR == VAR_NO_EVAL_SENSOR_MODEL:
            # this version directly uses the sensor model in Python, at a significant computational cost
            self.queries[:, 0] = np.repeat(sensor_dist[:, 0], num_rays)
            self.queries[:, 1] = np.repeat(sensor_dist[:, 1], num_rays)
            self.queries[:, 2] = np.repeat(sensor_dist[:, 2], num_rays)
            self.queries[:, 2] += self.tiled_angles

            # compute the ranges for all the particles in a single functon call
            self.range_method.calc_range_many(self.queries, self.ranges)

            # resolve the sensor model by discretizing and indexing into the precomputed table
            obs /= float(self.map_info.resolution)
            ranges = self.ranges / float(self.map_info.resolution)
            obs[obs > self.MAX_RANGE_PX] = self.MAX_RANGE_PX
            ranges[ranges > self.MAX_RANGE_PX] = self.MAX_RANGE_PX

            intobs = np.rint(obs).astype(np.uint16)
            intrng = np.rint(ranges).astype(np.uint16)

            # compute the weight for each particle
            for i in range(self.MAX_PARTICLES):
                weight = np.product(
                    self.sensor_model_table[
                        intobs, intrng[i * num_rays : (i + 1) * num_rays]
                    ]
                )
                weight = np.power(weight, self.INV_SQUASH_FACTOR)
                weights[i] = weight
        else:
            self.get_logger().info("PLEASE SET rangelib_variant PARAM to 0-4")

    def MCL(self, a, o):
        """
        Performs one step of Monte Carlo Localization.
            1. resample particle distribution to form the proposal distribution
            2. apply the motion model
            3. apply the sensor model
            4. normalize particle weights

        This is in the critical path of code execution, so it is optimized for speed.
        """
        if self.SHOW_FINE_TIMING:
            t = time.time()
        # draw the proposal distribution from the old particles
        # [Orin 최적화] multinomial(np.random.choice) → systematic resampling + 사전할당 버퍼.
        # 동일 확률분포에서 저분산 추출, 배열 신규 할당 없음 (08-15 실차 검증 2.97배 단축)
        np.cumsum(self.weights, out=self._resample_cdf)
        self._resample_cdf[-1] = 1.0
        np.add(
            self._resample_base,
            np.random.random() / float(self.MAX_PARTICLES),
            out=self._resample_positions,
        )
        proposal_indices = np.searchsorted(
            self._resample_cdf, self._resample_positions, side="left"
        )
        np.take(
            self.particles,
            proposal_indices,
            axis=0,
            out=self._resample_particles,
            mode="clip",
        )
        proposal_distribution = self._resample_particles
        if self.SHOW_FINE_TIMING:
            t_propose = time.time()

        # Motion is one exact odom->base_link delta per successful scan epoch.
        self.motion_model(proposal_distribution, a)
        if self.SHOW_FINE_TIMING:
            t_motion = time.time()

        # compute the sensor model
        self.sensor_model(proposal_distribution, o, self.weights)
        if self.SHOW_FINE_TIMING:
            t_sensor = time.time()

        # normalize importance weights
        self._health_raw_w_mean = float(np.mean(self.weights))   # [P0-3] 정규화 전 평균 likelihood
        self.weights /= np.sum(self.weights)
        if self.SHOW_FINE_TIMING:
            t_norm = time.time()
            t_total = (t_norm - t) / 100.0

        if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
            self.get_logger().info(
                str(
                    [
                        "MCL: propose: ",
                        np.round((t_propose - t) / t_total, 2),
                        "motion:",
                        np.round((t_motion - t_propose) / t_total, 2),
                        "sensor:",
                        np.round((t_sensor - t_motion) / t_total, 2),
                        "norm:",
                        np.round((t_norm - t_sensor) / t_total, 2),
                    ]
                )
            )

        # save the particles — 버퍼 스왑 (proposal은 _resample_particles를 가리키므로 재할당 없이 교대)
        self.particles, self._resample_particles = (
            proposal_distribution,
            self.particles,
        )

    def expected_pose(self):
        # returns the expected value of the pose given the particle distribution
        return np.dot(self.particles.transpose(), self.weights)

    @staticmethod
    def relative_odom_motion(previous, current):
        p = previous.transform.translation
        c = current.transform.translation
        pq = previous.transform.rotation
        cq = current.transform.rotation
        previous_yaw = tf_transformations.euler_from_quaternion(
            [pq.x, pq.y, pq.z, pq.w])[2]
        current_yaw = tf_transformations.euler_from_quaternion(
            [cq.x, cq.y, cq.z, cq.w])[2]
        return np.array(relative_planar_motion(
            p.x, p.y, previous_yaw, c.x, c.y, current_yaw))

    def _publish_health(self):
        """[P0-3] /pf/health 발행 + 발산 자가 경고.

        data = [N_eff비, 평균 likelihood, 직전 추정 대비 점프[m], drift_scale, drift_yaw]
        (레이아웃은 pf_health.py). 08-24 이전에는 도달 불가 코드에 있어 한 번도 안 나갔다."""
        if not self.PUBLISH_HEALTH or self.inferred_pose is None:
            return
        jump = 0.0
        if self._health_prev_inferred is not None:
            jump = float(np.hypot(
                self.inferred_pose[0] - self._health_prev_inferred[0],
                self.inferred_pose[1] - self._health_prev_inferred[1]))
        self._health_prev_inferred = np.copy(self.inferred_pose)
        if self.last_pose is not None:
            self._health_window.push(
                (float(self.inferred_pose[0]), float(self.inferred_pose[1]),
                 float(self.inferred_pose[2])),
                (float(self.last_pose[0]), float(self.last_pose[1]),
                 float(self.last_pose[2])))
        scale, dyaw = self._health_window.drift()
        neff = float(n_eff_ratio(self.weights))
        m = Float32MultiArray()
        m.data = [neff, float(self._health_raw_w_mean), jump, float(scale), float(dyaw)]
        self.health_pub.publish(m)

        # 발산 자가 경고 — N_eff 가 바닥이면 파티클이 한 점에 몰려 다른 가설을 못 본다.
        # 이 맵은 주행선 2 m 안이 28~78% 미탐사라 우도가 다봉이고(1~3 m 떨어진 봉우리
        # 5개 실측), 한 번 엉뚱한 봉우리로 가면 스스로 못 빠져나온다. 소리는 내야 한다.
        if self.HEALTH_WARN_NEFF > 0.0 and neff < self.HEALTH_WARN_NEFF:
            self.get_logger().warn(
                'PF 발산 의심: N_eff비 %.3f < %.3f — 추정이 한 가설에 갇혔을 수 있다. '
                '맵 미탐사 구간에서는 잘못된 자리에도 스캔이 맞는다(자기모순 없음). '
                '속도를 줄이고 RViz 로 위치를 확인할 것.' % (neff, self.HEALTH_WARN_NEFF),
                throttle_duration_sec=2.0)

    def update(self, scan_stamp):
        """
        Update at the LaserScan epoch only.  Arrival-order odometry is never
        mixed with a different scan timestamp.
        """
        if self.clock_epoch_latch.faulted:
            return
        scan_ns = int(scan_stamp.sec) * 1000000000 + int(scan_stamp.nanosec)
        if scan_ns == 0:
            self.get_logger().warn('Skipping PF update with zero scan stamp')
            return
        if not self.clock_epoch_latch.observe(self.last_motion_stamp_ns, scan_ns):
            self.inferred_pose = None
            self.estimate_stamp = None
            self.last_motion_tf = None
            self.odom_initialized = False
            self.get_logger().fatal(
                'ROS time moved backward; restart PF, lattice_planner, and Pure Pursuit')
            return
        if not (self.lidar_initialized and self.odom_initialized and self.map_initialized):
            return
        scan_age = (self.get_clock().now().nanoseconds - scan_ns) * 1e-9
        if time.monotonic() - self._last_latency_log >= 1.0:
            self.get_logger().info(
                '[latency] PF scan arrival: %.1f ms' % (scan_age * 1000.0))
            self._last_latency_log = time.monotonic()
        if scan_age < -self.MAX_FUTURE_STAMP:
            self.get_logger().warn(
                'Skipping PF scan from the future: %.3f s' % scan_age)
            return
        if self.last_motion_stamp_ns and scan_ns <= self.last_motion_stamp_ns:
            self.get_logger().warn('Skipping duplicate/out-of-order scan')
            return
        try:
            odom_to_base = self.tf_buffer.lookup_transform(
                'odom', self.BASE_FRAME, rclpy.time.Time.from_msg(scan_stamp),
                timeout=Duration(seconds=self.TF_LOOKUP_TIMEOUT))
        except TransformException as ex:
            if time.monotonic() - self._last_tf_warning >= 1.0:
                self.get_logger().warn(
                    'Skipping PF scan: exact odom -> base_link transform '
                    'unavailable: %s' % ex)
                self._last_tf_warning = time.monotonic()
            return
        action = np.zeros(3)
        if self.last_motion_tf is not None:
            action = self.relative_odom_motion(self.last_motion_tf, odom_to_base)
        if self.state_lock.locked():
            return
        with self.state_lock:
            self.timer.tick()
            self.iters += 1
            started = time.time()
            observation = np.copy(self.downsampled_ranges).astype(np.float32)
            self.MCL(action, observation)
            self.inferred_pose = self.expected_pose()
            self.estimate_stamp = scan_stamp
            self.last_motion_tf = odom_to_base
            self.last_motion_stamp_ns = scan_ns
            q = odom_to_base.transform.rotation
            base_yaw = tf_transformations.euler_from_quaternion(
                [q.x, q.y, q.z, q.w])[2]
            self.last_pose = np.array([
                odom_to_base.transform.translation.x,
                odom_to_base.transform.translation.y, base_yaw])
            if self.JUMP_GATE_ENABLE:
                gated, held = self._jump_gate.check(
                    (float(self.inferred_pose[0]), float(self.inferred_pose[1]),
                     float(self.inferred_pose[2])), float(np.hypot(action[0], action[1])),
                    action=(float(action[0]), float(action[1]), float(action[2])))
                if held:
                    self.get_logger().warn(
                        'pose 점프 게이트: 추정 점프 차단 — last-good 유지 (연속 %d회, %s)'
                        % (self._jump_gate._holds, self._jump_gate.hold_mode),
                        throttle_duration_sec=1.0)
                    self.inferred_pose = np.array(gated)
            finished = time.time()
            # Keep outputs in the same transaction as MCL so the manual
            # callback cannot reset stamp/pose between calculation and RViz.
            logger_file.write('%f, %f, %f\n' % tuple(self.inferred_pose))
            self.publish_tf(self.inferred_pose, scan_stamp, odom_to_base)
            # [08-24 결함 수정] 헬스 발행이 아래 **도달 불가** 구간(구 arrival-order 코드)에만
            # 있어서 `/pf/health` 가 한 번도 나가지 않았다 — 실차 백 15개 전부 0건.
            # 그래서 08-21 이후 "PF 발산을 /pf/health 로 판단하라"는 절차가 통째로 무력했고,
            # PP_problem_retry(car6) 처럼 PF 가 2 m 어긋난 채 자기모순 없이(fit 0.83) 계속
            # 달리는 상황을 아무도 감지하지 못했다. 살아 있는 경로로 옮긴다.
            self._publish_health()
            self.smoothing.append(1.0 / max(finished - started, 1e-6))
            self.visualize()
        return

        # Legacy arrival-order implementation retained below only as reference.
        if self.lidar_initialized and self.odom_initialized and self.map_initialized:
            if self.state_lock.locked():
                self.get_logger().info("Concurrency error avoided")
            else:
                self.state_lock.acquire()
                self.timer.tick()
                self.iters += 1

                t1 = time.time()
                observation = np.copy(self.downsampled_ranges).astype(np.float32)
                action = np.copy(self.odometry_data)
                self.odometry_data = np.zeros(3)

                # run the MCL update algorithm
                self.MCL(action, observation)

                # compute the expected value of the robot pose
                self.inferred_pose = self.expected_pose()

                # [P0-5] pose 점프 게이트 — 발행 추정만 보정(파티클 무개입)
                if self.JUMP_GATE_ENABLE:
                    odom_step = 0.0
                    if self._gate_prev_odom is not None:
                        odom_step = float(np.hypot(
                            self.last_pose[0] - self._gate_prev_odom[0],
                            self.last_pose[1] - self._gate_prev_odom[1]))
                    self._gate_prev_odom = np.copy(self.last_pose)
                    gated, held = self._jump_gate.check(
                        (float(self.inferred_pose[0]), float(self.inferred_pose[1]),
                         float(self.inferred_pose[2])), odom_step)
                    if held:
                        self.get_logger().warn(
                            "pose 점프 게이트: 추정 점프 차단 — last-good 유지 "
                            "(연속 %d회)" % self._jump_gate._holds)
                        self.inferred_pose = np.array(gated)
                self.state_lock.release()
                t2 = time.time()

                # log inferred pose (x, y, yaw)
                logger_file.write('%f, %f, %f\n' %(self.inferred_pose[0], self.inferred_pose[1], self.inferred_pose[2]))

                # publish transformation frame based on inferred pose
                self.publish_tf(self.inferred_pose, self.last_stamp)

                # [P0-3] 헬스 지표 발행 — 레이아웃은 pf_health.py 참조
                if self.PUBLISH_HEALTH:
                    jump = 0.0
                    if self._health_prev_inferred is not None:
                        jump = float(np.hypot(
                            self.inferred_pose[0] - self._health_prev_inferred[0],
                            self.inferred_pose[1] - self._health_prev_inferred[1]))
                    self._health_prev_inferred = np.copy(self.inferred_pose)
                    self._health_window.push(
                        (float(self.inferred_pose[0]), float(self.inferred_pose[1]),
                         float(self.inferred_pose[2])),
                        (float(self.last_pose[0]), float(self.last_pose[1]),
                         float(self.last_pose[2])))
                    scale, dyaw = self._health_window.drift()
                    m = Float32MultiArray()
                    m.data = [float(n_eff_ratio(self.weights)),
                              float(self._health_raw_w_mean),
                              jump, float(scale), float(dyaw)]
                    self.health_pub.publish(m)

                # this is for tracking particle filter speed
                ips = 1.0 / (t2 - t1)
                self.smoothing.append(ips)
                if self.iters % 10 == 0:
                    self.get_logger().info(
                        str(
                            [
                                "iters per sec:",
                                int(self.timer.fps()),
                                " possible:",
                                int(self.smoothing.mean()),
                            ]
                        )
                    )

                self.visualize()


# import argparse
# import sys
# parser = argparse.ArgumentParser(description='Particle filter.')
# parser.add_argument('--config', help='Path to yaml file containing config parameters. Helpful for calling node directly with Python for profiling.')

# def load_params_from_yaml(fp):
#     from yaml import load
#     with open(fp, 'r') as infile:
#         yaml_data = load(infile)
#         for param in yaml_data:
#             print 'param:', param, ':', yaml_data[param]
#             rospy.set_param('~'+param, yaml_data[param])

# # this function can be used to generate flame graphs easily
# def make_flamegraph(filterx=None):
#     import flamegraph, os
#     perf_log_path = os.path.join(os.path.dirname(__file__), '../tmp/perf.log')
#     flamegraph.start_profile_thread(fd=open(perf_log_path, 'w'),
#                                     filter=filterx,
#                                     interval=0.001)

def shutdown():
    print("writting log files...")
    logger_file.close()

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFiler()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(pf)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        pf.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    atexit.register(shutdown)
    main()

# if __name__=='__main__':
#     rospy.init_node('particle_filter')

#     args,_ = parser.parse_known_args()
#     if args.config:
#         load_params_from_yaml(args.config)

#     # make_flamegraph(r'update')

#     pf = ParticleFiler()
#     rospy.spin()
