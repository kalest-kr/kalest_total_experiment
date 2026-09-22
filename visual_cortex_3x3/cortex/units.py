"""units.py -- 단위 규약과 물리 상수.

전도도 모드(`conductance_lif`)의 단위는 다음으로 **통일**한다.

======== ======== ====================================================
기호     단위     설명
======== ======== ====================================================
V        mV       막전위
t, tau   ms       시간, 시정수
C        pF       막 용량
g        nS       전도도 (누설, 시냅스, 구획 결합)
I        pA       전류
======== ======== ====================================================

이 조합은 서로 정합적이다::

    [nS] * [mV] = 1e-9 S * 1e-3 V = 1e-12 A = 1 pA
    [pF] * [mV] / [ms] = 1e-12 F * 1e-3 V / 1e-3 s = 1e-12 A = 1 pA

따라서 ``C dV/dt = g (E - V) + I`` 를 pF, mV, ms, nS, pA 로 그대로 쓸 수 있고
별도의 환산 계수가 필요 없다. 코드 어디에서도 SI 기본 단위로 되돌리지 않는다.

``sum_threshold`` 모드의 s_j, theta_j, P 는 **무차원**이다. 이 모드의 합산값을
막전위(mV)로 해석하지 않는다. 두 모드의 단위는 서로 호환되지 않으며 변환
계수를 임의로 도입하지 않는다.

공간 단위:

* ``cortical_xyz_mm`` : 피질 모형 좌표, mm
* ``visual_field_xy`` : 시야 좌표, degree (시야각). 픽셀→도 변환은
  영상 크기와 FOV 설정으로 계산한다 (:mod:`cortex.retinotopy`).
* 두 공간의 거리를 서로 섞어 쓰지 않는다.

발화율은 Hz(=1/s)이고 dt 는 ms 다. dt 동안의 기대 사건 수는 ``rate_hz * dt_ms / 1000``
이며, 이 변환을 거치지 않고 Hz 를 ms 격자에 직접 쓰지 않는다.
"""

from __future__ import annotations

#: 1초 = 1000 ms
MS_PER_S: float = 1000.0

#: 발화율(Hz)과 dt(ms)로부터 dt 동안의 기대 사건 수
def expected_events(rate_hz: float, dt_ms: float) -> float:
    """rate_hz [1/s] 와 dt_ms [ms] 로부터 dt 동안의 기대 사건 수(무차원)."""
    return float(rate_hz) * float(dt_ms) / MS_PER_S


#: 문서/manifest 에 기록할 단위 표
UNIT_TABLE: dict[str, str] = {
    "voltage": "mV",
    "time": "ms",
    "capacitance": "pF",
    "conductance": "nS",
    "current": "pA",
    "rate": "Hz",
    "cortical_position": "mm",
    "visual_field": "deg",
    "image_position": "px",
    "sum_threshold_activation": "dimensionless",
    "synaptic_weight_conductance_mode": "nS",
    "synaptic_weight_sum_mode": "dimensionless",
}

#: NMDA Mg2+ 차단 식의 기본 계수 (Jahr & Stevens 1990 형태). 모형 파라미터다.
MG_BLOCK_MG_MM: float = 1.0      # [Mg2+] in mM
MG_BLOCK_SLOPE: float = 0.062    # 1/mV
MG_BLOCK_SCALE: float = 3.57     # mM

__all__ = [
    "MS_PER_S",
    "expected_events",
    "UNIT_TABLE",
    "MG_BLOCK_MG_MM",
    "MG_BLOCK_SLOPE",
    "MG_BLOCK_SCALE",
]
