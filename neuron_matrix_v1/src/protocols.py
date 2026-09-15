"""protocols.py -- V2/V4/IT 확장용 입출력 프로토콜 (설계 설명만, **미구현**).

명세 [1]: "V2·V4·IT 는 향후 확장을 위한 입출력 프로토콜과 설계 설명까지만
제공한다. 구현하지 않은 기능은 명시하고, 이름만 붙인 항등 함수로 완성된
피질을 가장하지 않는다."

따라서 이 모듈의 모든 단계는 호출하면 ``NotImplementedError`` 를 낸다.
항등 함수를 넣어 "구현된 것처럼" 보이게 하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

IMPLEMENTED_AREAS = ("V1",)
NOT_IMPLEMENTED_AREAS = ("V2", "V4", "IT", "L5", "L6")


@dataclass(frozen=True)
class AreaIO:
    """영역 간 입출력 규약 (설계 문서용 값 객체).

    Attributes
    ----------
    area : str
    input_spec : str
        받는 배열의 의미/shape/단위.
    output_spec : str
        내보내는 배열의 의미/shape/단위.
    observation_tick : str
        어느 시점을 그 영역의 반응으로 읽는지.
    teacher_interface : str
        교정 신호가 들어오는 경로.
    status : str
        "implemented" 또는 "not_implemented".
    """

    area: str
    input_spec: str
    output_spec: str
    observation_tick: str
    teacher_interface: str
    status: str


AREA_PROTOCOLS: dict[str, AreaIO] = {
    "V1": AreaIO(
        area="V1",
        input_spec="L4 ExternalDrive: (P*2,) 정규화 ON/OFF 값 [0,1], tick 0 주입",
        output_spec="L2 관찰 활동: (M,) 0/1, tick 2 / L3 보조 출력: (M,) 0/1, tick 3",
        observation_tick="L4=0, INH=1, L2=2, L3=3",
        teacher_interface="L1Relay.route({l2_neuron_id: correction}) -> b 의 correction_input",
        status="implemented",
    ),
    "V2": AreaIO(
        area="V2",
        input_spec="(설계) V1 L2/L3 의 (M,) 0/1 활동을 지연 d>=1 의 흥분 연결로 수신",
        output_spec="(설계) 윤곽 결합 후보 뉴런의 0/1 활동",
        observation_tick="(설계) V1 L3 관찰 tick + 1 이상",
        teacher_interface="(설계) 같은 L1Relay 규약, 목적지 집합만 V2 로 확장",
        status="not_implemented",
    ),
    "V4": AreaIO(
        area="V4",
        input_spec="(설계) V2 활동 (M2,) 0/1",
        output_spec="(설계) 형태 부분 뉴런의 0/1 활동",
        observation_tick="(설계) V2 관찰 tick + 1 이상",
        teacher_interface="(설계) 동일 규약",
        status="not_implemented",
    ),
    "IT": AreaIO(
        area="IT",
        input_spec="(설계) V4 활동 (M3,) 0/1",
        output_spec="(설계) 물체 범주 뉴런의 0/1 활동",
        observation_tick="(설계) V4 관찰 tick + 1 이상",
        teacher_interface="(설계) 동일 규약. 단, IT 최종 오차 하나에서 하위 영역의 "
                          "뉴런별 목표를 자동 역산하는 기능은 이 프로젝트에 없다.",
        status="not_implemented",
    ),
}


class AreaStage(Protocol):
    """향후 영역 단계가 따라야 할 인터페이스 (구현체 없음)."""

    def forward(self, upstream_activity: np.ndarray) -> np.ndarray: ...
    def observation_tick(self) -> int: ...


class NotImplementedStage:
    """미구현 영역 자리표시자. 호출하면 예외를 낸다 (항등 함수가 아니다)."""

    def __init__(self, area: str) -> None:
        if area not in AREA_PROTOCOLS:
            raise KeyError(area)
        self.area = area
        self.spec = AREA_PROTOCOLS[area]

    def forward(self, upstream_activity: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{self.area} 는 이 프로젝트에서 구현하지 않았다. "
            f"입출력 프로토콜 설명만 제공한다: {self.spec}"
        )

    def observation_tick(self) -> int:
        raise NotImplementedError(f"{self.area} 미구현")


def status_report() -> dict[str, Any]:
    return {
        "implemented_areas": list(IMPLEMENTED_AREAS),
        "not_implemented_areas": list(NOT_IMPLEMENTED_AREAS),
        "protocols": {k: vars(v) for k, v in AREA_PROTOCOLS.items()},
        "note_ko": (
            "구현된 것은 V1 의 L4/INH중계/L2/L3 뿐이다. V2/V4/IT/L5/L6 는 "
            "입출력 프로토콜과 설계 설명만 있으며 호출하면 NotImplementedError 를 낸다. "
            "100만 화소 입력 지원은 100만 뉴런 학습 완료를 뜻하지 않는다."
        ),
    }
