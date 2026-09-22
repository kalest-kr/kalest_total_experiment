"""ids.py -- 안정적인 정수 ID 와 이름 레지스트리.

뉴런·시냅스·영역·층·세포 유형·수용체·구획을 **정수 ID** 로 다룬다.
이웃 뉴런 객체를 다른 객체 안에 재귀적으로 넣지 않는다 (명세 2절).

ID 규약
-------
* ``neuron_id`` : 0..N-1. 전역 상태 배열의 행 인덱스와 **동일**하다.
* ``synapse_id`` : 0..K-1. edge table 의 행 인덱스와 동일하다.
* ``event_id``   : 실행(run) 안에서 단조 증가하는 정수. 체크포인트에서 이어진다.
* ``spike_id``   : 발화 1건의 고유 ID. 도착 이벤트의 ``parent_spike_id`` 가 된다.

영역/층/세포유형/수용체 이름은 :class:`Registry` 로 정수에 매핑한다. 매핑은
manifest 에 저장되어 재현 시 같은 정수를 복원한다. Python 의 실행별 hash 에
의존하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

#: 구획 인덱스. 피라미드 세포는 3개, 나머지 유형은 soma 만 사용한다.
COMPARTMENT_NAMES: tuple[str, ...] = ("soma", "basal", "apical")
COMPARTMENT_INDEX: dict[str, int] = {n: i for i, n in enumerate(COMPARTMENT_NAMES)}
N_COMPARTMENTS: int = len(COMPARTMENT_NAMES)

#: 피질 층 이름. L2/3 을 연산에서 묶더라도 조회·집계에서는 구분을 유지한다.
CORTICAL_LAYERS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "L6")

#: 비피질 구조의 "층" 자리표시 이름 (망막/LGN)
NONCORTICAL_LAYERS: tuple[str, ...] = ("L_input", "L_relay")

#: 이벤트 종류
EVENT_TYPES: tuple[str, ...] = (
    "external_drive",   # 망막/외부 자극 주입
    "synaptic_arrival", # 시냅스 도착
    "spike",            # 발화
    "weight_update",    # 가중치 변경
    "threshold_update", # 임계값 변경
)


class Registry:
    """이름 <-> 정수 ID 양방향 레지스트리 (삽입 순서로 ID 부여)."""

    def __init__(self, names: Iterable[str] = ()) -> None:
        self._names: list[str] = []
        self._index: dict[str, int] = {}
        for n in names:
            self.add(n)

    def add(self, name: str) -> int:
        """이름을 등록하고 ID 를 돌려준다. 이미 있으면 기존 ID."""
        if name in self._index:
            return self._index[name]
        idx = len(self._names)
        self._names.append(name)
        self._index[name] = idx
        return idx

    def id_of(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError as exc:
            raise KeyError(
                f"등록되지 않은 이름: {name!r} (등록된 것: {self._names})"
            ) from exc

    def name_of(self, idx: int) -> str:
        return self._names[int(idx)]

    def has(self, name: str) -> bool:
        return name in self._index

    def names(self) -> list[str]:
        return list(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def to_dict(self) -> dict[str, int]:
        return dict(self._index)

    @staticmethod
    def from_dict(mapping: dict[str, int]) -> "Registry":
        """manifest 에서 복원. 저장된 정수값을 그대로 보존한다."""
        reg = Registry()
        reg._names = [""] * (max(mapping.values()) + 1 if mapping else 0)
        for name, idx in mapping.items():
            reg._names[int(idx)] = name
            reg._index[name] = int(idx)
        return reg


@dataclass
class IdSpace:
    """한 실행에서 쓰는 모든 이름 레지스트리 묶음."""

    areas: Registry = field(default_factory=Registry)
    layers: Registry = field(default_factory=lambda: Registry(
        list(CORTICAL_LAYERS) + list(NONCORTICAL_LAYERS)))
    cell_types: Registry = field(default_factory=Registry)
    receptors: Registry = field(default_factory=Registry)
    compartments: Registry = field(default_factory=lambda: Registry(COMPARTMENT_NAMES))
    event_types: Registry = field(default_factory=lambda: Registry(EVENT_TYPES))
    plasticity_rules: Registry = field(default_factory=lambda: Registry(("none", "stdp")))

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            "areas": self.areas.to_dict(),
            "layers": self.layers.to_dict(),
            "cell_types": self.cell_types.to_dict(),
            "receptors": self.receptors.to_dict(),
            "compartments": self.compartments.to_dict(),
            "event_types": self.event_types.to_dict(),
            "plasticity_rules": self.plasticity_rules.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict[str, dict[str, int]]) -> "IdSpace":
        return IdSpace(
            areas=Registry.from_dict(d["areas"]),
            layers=Registry.from_dict(d["layers"]),
            cell_types=Registry.from_dict(d["cell_types"]),
            receptors=Registry.from_dict(d["receptors"]),
            compartments=Registry.from_dict(d["compartments"]),
            event_types=Registry.from_dict(d["event_types"]),
            plasticity_rules=Registry.from_dict(d["plasticity_rules"]),
        )


class IdCounter:
    """event_id / spike_id 처럼 실행 전체에서 단조 증가하는 카운터.

    체크포인트에 현재 값을 저장하고 재개 시 이어받아 ID 가 중복되지 않게 한다.
    """

    def __init__(self, start: int = 0) -> None:
        self._next = int(start)

    def take(self, n: int = 1) -> int:
        """n 개를 예약하고 **시작 ID** 를 돌려준다 (부작용: 카운터 증가)."""
        if n < 0:
            raise ValueError("n 은 0 이상이어야 한다")
        start = self._next
        self._next += int(n)
        return start

    @property
    def value(self) -> int:
        return self._next

    def restore(self, value: int) -> None:
        self._next = int(value)


__all__ = [
    "COMPARTMENT_NAMES", "COMPARTMENT_INDEX", "N_COMPARTMENTS",
    "CORTICAL_LAYERS", "NONCORTICAL_LAYERS", "EVENT_TYPES",
    "Registry", "IdSpace", "IdCounter",
]
