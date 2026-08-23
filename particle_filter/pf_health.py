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


class JumpGate:
    """[P0-5] pose 점프 게이트 — 물리적으로 불가능한 추정 점프를 차단하고
    last-good pose를 유지한다. 연속 max_holds 초과 시 새 추정을 수용
    (진짜 재수렴/킥냅에서 옛 pose에 갇히지 않게).

    hold_mode:
      'freeze'      (기본, 구 동작) hold 중 last-good 을 **그 자리에 고정**한다.
      'dead_reckon' hold 중 last-good 을 odom 이동량(action)만큼 **전진시킨다**.

    [08-24 car1 실측] 'freeze' 의 결함: hold 가 시작되면 last-good 은 멈춰 있는데
    차는 계속 가므로, 다음 갱신의 새 추정은 '고정점 + 한 스텝 odom + 여유' 창을
    **반드시** 벗어난다 → max_holds 까지 연속 hold 가 강제되고, 풀릴 때 누적분이
    한 번에 점프한다. 4.3 m/s 직선에서 0.4 s 동결 뒤 2.93 m 순간이동이 그것이다.
    'dead_reckon' 은 창이 차와 함께 움직이므로 추정이 창 안으로 돌아오면 즉시
    채택되고, 그 사이 발행 pose 도 멈추지 않는다."""

    def __init__(self, margin_m=0.5, max_holds=5, hold_mode='freeze'):
        self.margin_m = float(margin_m)
        self.max_holds = int(max_holds)
        self.hold_mode = str(hold_mode)
        if self.hold_mode not in ('freeze', 'dead_reckon'):
            raise ValueError("jump_gate_hold_mode must be 'freeze' or 'dead_reckon', got %r"
                             % hold_mode)
        self._last_good = None
        self._holds = 0

    @staticmethod
    def _advance(pose, action):
        """pose=(x,y,yaw) 를 차 기준 odom 이동 action=(dx,dy,dyaw) 만큼 전진."""
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        return (x + c * action[0] - s * action[1],
                y + s * action[0] + c * action[1],
                yaw + action[2])

    def check(self, new_pose, odom_step_m, action=None):
        """new_pose=(x,y,yaw), odom_step_m=직전 갱신 이후 odom 이동량,
        action=(dx,dy,dyaw) 차 기준 odom 이동(dead_reckon 에 필요; 없으면 freeze 동작).

        반환 (채택 pose, held). held=True면 반환 pose 를 대신 발행하라는 뜻."""
        if self._last_good is None:
            self._last_good = tuple(new_pose)
            return tuple(new_pose), False
        jump = math.hypot(new_pose[0] - self._last_good[0],
                          new_pose[1] - self._last_good[1])
        if jump <= abs(odom_step_m) + self.margin_m:
            self._last_good = tuple(new_pose)
            self._holds = 0
            return tuple(new_pose), False
        self._holds += 1
        if self._holds > self.max_holds:
            # 연속 초과 — 새 추정 수용(재수렴 간주) 후 게이트 재무장
            self._last_good = tuple(new_pose)
            self._holds = 0
            return tuple(new_pose), False
        if self.hold_mode == 'dead_reckon' and action is not None:
            self._last_good = self._advance(self._last_good, action)
        return self._last_good, True


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
    # ⑥ JumpGate: 정상 이동 통과
    g = JumpGate(margin_m=0.5, max_holds=3)
    p0, h = g.check((0.0, 0.0, 0.0), 0.0)
    assert not h
    p1, h = g.check((0.3, 0.0, 0.0), 0.25)      # 0.3 ≤ 0.25+0.5
    assert not h and p1 == (0.3, 0.0, 0.0)
    # ⑦ 점프 차단 → last-good 유지
    p2, h = g.check((5.0, 5.0, 0.0), 0.25)
    assert h and p2 == (0.3, 0.0, 0.0)
    # ⑧ 연속 초과 시 수용(재수렴) 후 재무장
    for _ in range(2):
        p3, h = g.check((5.0, 5.0, 0.0), 0.25)
        assert h
    p4, h = g.check((5.0, 5.0, 0.0), 0.25)      # 4번째 = max_holds(3) 초과 → 수용
    assert not h and p4 == (5.0, 5.0, 0.0)
    # ⑨ 수용 후 정상 추적 재개
    p5, h = g.check((5.2, 5.0, 0.0), 0.25)
    assert not h and p5 == (5.2, 5.0, 0.0)
    # ⑩ [08-24] 'freeze' 의 결함 재현: 차가 0.4 m/갱신으로 계속 가는데 추정이 0.9 m
    #    앞서 버리면(0.4+0.5 초과), 고정된 last-good 대비 간극이 매 갱신 0.4 씩 벌어져
    #    max_holds 까지 **반드시** 얼고 누적분이 한 번에 점프한다 (car1 76 s: 2.93 m).
    def _run(mode):
        g_ = JumpGate(margin_m=0.5, max_holds=5, hold_mode=mode)
        g_.check((0.0, 0.0, 0.0), 0.0)
        x_true, pub, held = 0.0, [0.0], 0
        for _ in range(8):
            x_true += 0.4
            est = x_true + 0.95                # 추정이 0.95 m 앞서 있다(재수렴)
            pose, h = g_.check((est, 0.0, 0.0), 0.4, action=(0.4, 0.0, 0.0))
            held += int(h)
            pub.append(pose[0])
        jumps = [b - a for a, b in zip(pub, pub[1:])]
        return held, pub, max(jumps)
    held_f, pub_f, jump_f = _run('freeze')
    assert held_f == 5, held_f                 # max_holds 까지 전부 얼었다
    assert pub_f[1] == pub_f[5], pub_f         # 동결: 발행 pose 가 5갱신 동안 같은 값
    assert jump_f > 2.5, jump_f                # 풀릴 때 0.95 + 5×0.4 ≈ 2.95 m 순간이동
    # ⑪ 'dead_reckon': hold 중 last-good 이 odom 만큼 전진 → 발행 pose 가 멈추지 않고,
    #    풀릴 때 점프는 원래 오차 0.95 m 뿐이다(누적분이 안 붙는다). hold 횟수는 같다 —
    #    추정이 창 밖에 **계속** 있는 한 max_holds 까지 버티는 것이 게이트의 본업이다.
    held_d, pub_d, jump_d = _run('dead_reckon')
    assert held_d == 5, held_d
    assert all(b > a for a, b in zip(pub_d, pub_d[1:])), pub_d   # 동결 없음
    assert jump_d < 1.5, jump_d          # 0.95 + 그 스텝 정상 이동 0.4 (freeze 는 2.95)
    # ⑫ dead_reckon 도 진짜 점프(5 m 킥냅)는 여전히 max_holds 동안 막고 그 뒤 수용한다
    gk = JumpGate(margin_m=0.5, max_holds=3, hold_mode='dead_reckon')
    gk.check((0.0, 0.0, 0.0), 0.0)
    holds = [gk.check((9.0, 9.0, 0.0), 0.1, action=(0.1, 0.0, 0.0))[1] for _ in range(4)]
    assert holds == [True, True, True, False], holds
    # 헤딩 회전도 dead-reckon 된다
    gr = JumpGate(hold_mode='dead_reckon')
    gr.check((0.0, 0.0, 0.0), 0.0)
    pose, h = gr.check((9.0, 0.0, 0.0), 0.1, action=(0.1, 0.0, 0.5))
    assert h and abs(pose[2] - 0.5) < 1e-9 and abs(pose[0] - 0.1) < 1e-9
    try:
        JumpGate(hold_mode='bogus')
        raise AssertionError('bogus hold_mode 가 통과했다')
    except ValueError:
        pass
    print("pf_health 자가 검증 12/12 통과 ✅ "
          "(N_eff·스케일·yaw잔차·창 롤링·랩핑·게이트 통과/차단/해제/재무장·"
          "freeze 동결 재현·dead_reckon 비동결·킥냅 유지)")


if __name__ == "__main__":
    _self_test()
