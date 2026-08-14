"""PF 헬스 지표·odom↔map 잔차 계산 (P0-3) — rclpy 무의존 순수 로직.

발행 배열 레이아웃 (/pf/health, std_msgs/Float32MultiArray):
  [0] n_eff_ratio      유효 입자 비율 1/(N·Σw²), 1.0=균등(건강) → 0=한 점 몰림(발산 의심)
  [1] mean_likelihood  센서 모델 원시 가중치 평균(정규화 전) — 낮으면 스캔↔맵 불일치
  [2] pose_jump_m      직전 추정 대비 위치 점프 [m] — 이동량 대비 과대면 추정 튐
  [3] drift_scale      최근 창(health_window_m)에서 PF 이동거리/odom 이동거리
                       (1.0=일치, ≠1 지속 = speed_to_erpm_gain 재보정 신호 → OPERATIONS §3-5)
  [4] drift_yaw_rad    같은 창의 PF yaw 변화 − odom yaw 변화 누적 [rad]

소비자: Phase 3a race_monitor / 3b 복구 FSM (발산 판정), 수동 진단(ros2 topic echo).
실행: python3 particle_filter/pf_health.py  → 자가 검증.

self-test: python3 pf_health.py
"""
import math
from collections import deque


def n_eff_ratio(weights):
    """유효 입자 비율 = 1/(N·Σw²). weights는 정규화된 배열(합 1)."""
    n = len(weights)
    if n == 0:
        return 0.0
    ssq = float(sum(w * w for w in weights))
    if ssq <= 0.0:
        return 0.0
    return 1.0 / (n * ssq)


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class DriftWindow:
    """PF·odom 증분을 odom 이동거리 기준 최근 window_m 만큼 유지하며
    스케일(거리비)·yaw 잔차를 계산한다."""

    def __init__(self, window_m=5.0):
        self.window_m = float(window_m)
        self._steps = deque()          # (pf_d, odom_d, dyaw_resid)
        self._pf_sum = 0.0
        self._odom_sum = 0.0
        self._yaw_sum = 0.0
        self._last_pf = None           # (x, y, yaw)
        self._last_odom = None

    def push(self, pf_pose, odom_pose):
        """pf_pose/odom_pose = (x, y, yaw). 갱신 주기마다 호출."""
        if self._last_pf is not None:
            pf_d = math.hypot(pf_pose[0] - self._last_pf[0],
                              pf_pose[1] - self._last_pf[1])
            od_d = math.hypot(odom_pose[0] - self._last_odom[0],
                              odom_pose[1] - self._last_odom[1])
            dyaw = wrap_pi(wrap_pi(pf_pose[2] - self._last_pf[2])
                           - wrap_pi(odom_pose[2] - self._last_odom[2]))
            self._steps.append((pf_d, od_d, dyaw))
            self._pf_sum += pf_d
            self._odom_sum += od_d
            self._yaw_sum += dyaw
            while self._odom_sum > self.window_m and len(self._steps) > 1:
                p, o, y = self._steps.popleft()
                self._pf_sum -= p
                self._odom_sum -= o
                self._yaw_sum -= y
        self._last_pf = tuple(pf_pose)
        self._last_odom = tuple(odom_pose)

    def drift(self):
        """(drift_scale, drift_yaw_rad). 창이 아직 짧으면(정지·초기) scale=1.0."""
        if self._odom_sum < 0.5:       # 이동이 거의 없으면 비율이 무의미
            return 1.0, self._yaw_sum
        return self._pf_sum / self._odom_sum, self._yaw_sum


def _self_test():
    # ① N_eff: 균등 → 1.0, 한 점 몰림 → 1/N
    n = 100
    assert abs(n_eff_ratio([1.0 / n] * n) - 1.0) < 1e-9
    w = [0.0] * n
    w[3] = 1.0
    assert abs(n_eff_ratio(w) - 1.0 / n) < 1e-9
    # ② 드리프트 창: odom이 실제보다 3% 길게 적산(gain 과소) → scale ≈ 1/1.03
    dw = DriftWindow(window_m=5.0)
    for i in range(400):
        s = i * 0.03
        dw.push((s, 0.0, 0.0), (s * 1.03, 0.0, 0.0))
    sc, dy = dw.drift()
    assert abs(sc - 1.0 / 1.03) < 1e-3, sc
    assert abs(dy) < 1e-9
    # ③ yaw 잔차: PF가 창 동안 0.1 rad 더 회전
    dw = DriftWindow(window_m=100.0)   # 창 넉넉히 — 누적 전체 관찰
    for i in range(101):
        s = i * 0.05
        dw.push((s, 0.0, 0.001 * i), (s, 0.0, 0.0))
    _, dy = dw.drift()
    assert abs(dy - 0.1) < 1e-6, dy
    # ④ 창 롤링: 오래된 구간이 빠져나가 최근 창만 반영
    dw = DriftWindow(window_m=2.0)
    for i in range(200):               # 전반 100스텝 scale 1.1, 후반 1.0
        s = i * 0.05
        pf = s if i < 100 else 5.0 * 1.0 + (s - 5.0)
        dw.push((pf * (1.1 if i < 100 else 1.0), 0.0, 0.0), (s, 0.0, 0.0))
    sc, _ = dw.drift()
    assert abs(sc - 1.0) < 0.02, sc
    # ⑤ 랩 경계 yaw 랩핑
    dw = DriftWindow(window_m=10.0)
    dw.push((0.0, 0.0, 3.1), (0.0, 0.0, 3.1))
    dw.push((1.0, 0.0, -3.1), (1.0, 0.0, -3.1))   # +π 근처 → −π 근처 (연속 회전)
    _, dy = dw.drift()
    assert abs(dy) < 1e-9
    print("pf_health 자가 검증 5/5 통과 ✅ (N_eff·스케일·yaw잔차·창 롤링·랩핑)")


if __name__ == "__main__":
    _self_test()
