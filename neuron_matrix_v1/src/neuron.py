"""neuron.py -- 3x3 행렬을 상태의 "기준 저장소"로 사용하는 뉴런 구현.

행렬 배치 (행 기준, 명세 [2]절):

    M_i = [[x_i, y_i, z_i],
           [u_i, threshold_i, P_i],
           [E_i, I_i,        b_i]]

설계 결정 (비차단적 선택, config.json 의 ``implementation_notes`` 에도 기록):

* 모든 뉴런의 3x3 행렬은 ``NeuronStateStore`` 가 소유한 하나의
  ``(N, 3, 3) float64`` 배열 위의 **뷰(view)** 다. 따라서
  ``neuron.M`` 은 명세대로 shape ``(3,3)``, dtype ``float64`` 이면서
  동시에 엔진이 전체 모집단을 벡터화해 계산할 수 있다.
  값은 한 곳(배열)에만 저장되므로 중복 저장으로 인한 불일치가 없다.
* ``threshold`` / ``u`` / ``P`` / ``E`` / ``I`` / ``b`` 는 모두 행렬 셀을
  읽고 쓰는 property 다. 별도 필드를 두지 않는다.
* ``b_base`` (고정 파라미터) 와 ``correction_input`` (일시 교정) 은 행렬 밖의
  값이다. 행렬의 ``b`` 셀은 항상 이 둘의 **합**으로 계산되어 기록된다
  (명세 [2]: "b: 해당 시점의 기저 입력과 외부 교정 입력의 합").
  즉 같은 값을 두 곳에 중복 저장하는 것이 아니라 성분과 합의 관계다.
* ``external_sensory_drive`` 는 L4 에 주입되는 비음수 감각값이며 행렬 밖의
  일시 상태다. E 셀에는 명세 [4]에 따라 이미 합산되어 들어간다.

단위: 현재 모델의 u/E/I/b/threshold/P 는 모두 **무차원**이다. 막전위(mV)나
전도도(nS)가 아니다. x,y,z 는 ``position_space`` 메타데이터가 정의하는
모형 좌표이며 실제 조직 측정값이 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 행렬 셀 인덱스 상수 (행, 열)
CELL_X = (0, 0)
CELL_Y = (0, 1)
CELL_Z = (0, 2)
CELL_U = (1, 0)
CELL_THRESHOLD = (1, 1)
CELL_P = (1, 2)
CELL_E = (2, 0)
CELL_I = (2, 1)
CELL_B = (2, 2)

CELL_MEANING: dict[str, str] = {
    "(0,0)": "x: 모형 피질 좌표 x",
    "(0,1)": "y: 모형 피질 좌표 y",
    "(0,2)": "z: 모형 피질 좌표 z (층 깊이, 설계값)",
    "(1,0)": "u: 현재 시점 발화 판단 입력 (무차원)",
    "(1,1)": "threshold: 발화 임계값 (무차원)",
    "(1,2)": "P: 송신 신호에 곱하는 뉴런별 전달 계수",
    "(2,0)": "E: 가중치 적용 흥분 입력 합 + external_sensory_drive",
    "(2,1)": "I: 가중치 적용 억제 입력 크기 합 (양수 저장)",
    "(2,2)": "b: b_base + correction_input",
}

# 행렬 밖에 두는 메타데이터 목록 (명세 [2])
METADATA_FIELDS = (
    "neuron_id",
    "name",
    "area",
    "layer",
    "cell_type",
    "preferred_orientation_deg",
    "receptive_field",
    "position_space",
    "fired",
    "observation_tick",
    "incoming_connection_ids",
    "outgoing_connection_ids",
)


@dataclass
class ReceptiveField:
    """원본 영상 좌표계에서 정의되는 수용장.

    주의: 피질 모형 좌표 (x,y,z) 와 완전히 별개의 좌표계다.

    Attributes
    ----------
    center_x, center_y : float
        원본 영상 픽셀 좌표 (열, 행) 기준 중심.
    radius_px : float
        모형 수용장 반경 (픽셀). ``max(2, 2*sigma_pool)`` 의 **설계값**이며
        생물학적 측정값이 아니다.
    sigma_pool_px : float
        해당 격자점의 안티앨리어싱/풀링 가우시안 폭 (픽셀).
    eccentricity_px : float
        시야 중심에서의 거리 (픽셀).
    units : str
        좌표 단위 설명 문자열.
    """

    center_x: float
    center_y: float
    radius_px: float
    sigma_pool_px: float = 0.0
    eccentricity_px: float = 0.0
    units: str = "source_image_pixels"

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_x": float(self.center_x),
            "center_y": float(self.center_y),
            "radius_px": float(self.radius_px),
            "sigma_pool_px": float(self.sigma_pool_px),
            "eccentricity_px": float(self.eccentricity_px),
            "units": self.units,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ReceptiveField":
        return ReceptiveField(
            center_x=d["center_x"],
            center_y=d["center_y"],
            radius_px=d["radius_px"],
            sigma_pool_px=d.get("sigma_pool_px", 0.0),
            eccentricity_px=d.get("eccentricity_px", 0.0),
            units=d.get("units", "source_image_pixels"),
        )


class NeuronStateStore:
    """모든 뉴런의 3x3 상태 행렬을 소유하는 ``(N,3,3) float64`` 배열.

    ``Neuron.M`` 은 이 배열의 행 뷰이므로 스칼라 접근과 벡터 연산이
    같은 메모리를 공유한다. 부작용: ``allocate`` 는 배열을 재할당하므로
    기존 뷰가 무효화된다. 회로 구성이 끝난 뒤에는 호출하지 않는다.
    """

    def __init__(self) -> None:
        self.states: np.ndarray = np.zeros((0, 3, 3), dtype=np.float64)
        # 행렬 밖 상태도 같은 방식으로 배열 1개씩만 둔다 (중복 저장 없음).
        self.b_base: np.ndarray = np.zeros(0, dtype=np.float64)
        self.correction_input: np.ndarray = np.zeros(0, dtype=np.float64)
        self.external_drive: np.ndarray = np.zeros(0, dtype=np.float64)
        self.fired: np.ndarray = np.zeros(0, dtype=np.int8)
        self.observation_tick: np.ndarray = np.zeros(0, dtype=np.int64)

    def allocate(self, n: int) -> None:
        """뉴런 n 개분의 상태 배열을 새로 할당한다 (기존 값은 버려진다)."""
        self.states = np.zeros((n, 3, 3), dtype=np.float64)
        self.b_base = np.zeros(n, dtype=np.float64)
        self.correction_input = np.zeros(n, dtype=np.float64)
        self.external_drive = np.zeros(n, dtype=np.float64)
        self.fired = np.zeros(n, dtype=np.int8)
        self.observation_tick = np.zeros(n, dtype=np.int64)

    # --- 벡터 뷰 (엔진이 사용) -------------------------------------------
    @property
    def u(self) -> np.ndarray:
        """shape (N,) float64 뷰."""
        return self.states[:, 1, 0]

    @property
    def threshold(self) -> np.ndarray:
        return self.states[:, 1, 1]

    @property
    def P(self) -> np.ndarray:
        return self.states[:, 1, 2]

    @property
    def E(self) -> np.ndarray:
        return self.states[:, 2, 0]

    @property
    def I(self) -> np.ndarray:  # noqa: E743  (명세의 기호를 유지)
        return self.states[:, 2, 1]

    @property
    def b(self) -> np.ndarray:
        return self.states[:, 2, 2]


@dataclass
class Neuron:
    """뉴런 한 개. 상태는 ``self.M`` (3x3 행렬 뷰) 에만 저장한다.

    행렬 밖 상태(``fired``, ``b_base``, ``correction_input``,
    ``external_sensory_drive``, ``observation_tick``)도 ``NeuronStateStore`` 의
    배열 1개씩에만 저장되고 여기서는 property 로 그 셀을 읽고 쓴다.
    따라서 어떤 값도 두 곳에 중복 저장되지 않는다.

    Parameters
    ----------
    neuron_id : int
        ``NeuronStateStore.states`` 의 행 인덱스와 동일한 정수.
    name : str
        등록 순서와 무관한 안정적 식별자 (예 ``"L2/ori90/k3/j7"``).
    store : NeuronStateStore
    """

    neuron_id: int
    name: str
    store: "NeuronStateStore"
    area: str = "V1"
    layer: str = "L4"
    cell_type: str = "excitatory"
    preferred_orientation_deg: float | None = None
    receptive_field: ReceptiveField | None = None
    position_space: str = "model_cortical_units (설계값, 실제 조직 측정 아님)"
    incoming_connection_ids: list[int] = field(default_factory=list)
    outgoing_connection_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.M.shape != (3, 3):
            raise ValueError(f"뉴런 행렬 shape 는 (3,3) 이어야 한다: {self.M.shape}")
        if self.M.dtype != np.float64:
            raise ValueError(f"뉴런 행렬 dtype 는 float64 이어야 한다: {self.M.dtype}")

    @property
    def M(self) -> np.ndarray:
        """이 뉴런의 3x3 상태 행렬 뷰 (shape (3,3), dtype float64)."""
        return self.store.states[self.neuron_id]

    @property
    def fired(self) -> int:
        """현재 발화 q (0/1). 행렬 밖 메타데이터."""
        return int(self.store.fired[self.neuron_id])

    @fired.setter
    def fired(self, value: int) -> None:
        self.store.fired[self.neuron_id] = int(value)

    @property
    def b_base(self) -> float:
        return float(self.store.b_base[self.neuron_id])

    @b_base.setter
    def b_base(self, value: float) -> None:
        self.store.b_base[self.neuron_id] = float(value)

    @property
    def correction_input(self) -> float:
        return float(self.store.correction_input[self.neuron_id])

    @correction_input.setter
    def correction_input(self, value: float) -> None:
        self.store.correction_input[self.neuron_id] = float(value)

    @property
    def external_sensory_drive(self) -> float:
        return float(self.store.external_drive[self.neuron_id])

    @external_sensory_drive.setter
    def external_sensory_drive(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("external_sensory_drive 는 비음수여야 한다")
        self.store.external_drive[self.neuron_id] = float(value)

    @property
    def observation_tick(self) -> int:
        return int(self.store.observation_tick[self.neuron_id])

    @observation_tick.setter
    def observation_tick(self, value: int) -> None:
        self.store.observation_tick[self.neuron_id] = int(value)

    # --- 위치 -------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        """모형 피질 좌표 (x,y,z), shape (3,)."""
        return self.M[0]

    def set_position(self, x: float, y: float, z: float) -> None:
        self.M[0, 0] = x
        self.M[0, 1] = y
        self.M[0, 2] = z

    # --- 발화 판단 관련 ---------------------------------------------------
    @property
    def u(self) -> float:
        return float(self.M[CELL_U])

    @u.setter
    def u(self, value: float) -> None:
        self.M[CELL_U] = value

    @property
    def threshold(self) -> float:
        return float(self.M[CELL_THRESHOLD])

    @threshold.setter
    def threshold(self, value: float) -> None:
        self.M[CELL_THRESHOLD] = value

    @property
    def transmission(self) -> float:
        """P: 송신 신호에 곱하는 전달 계수."""
        return float(self.M[CELL_P])

    @transmission.setter
    def transmission(self, value: float) -> None:
        self.M[CELL_P] = value

    @property
    def excitatory_sum(self) -> float:
        return float(self.M[CELL_E])

    @excitatory_sum.setter
    def excitatory_sum(self, value: float) -> None:
        self.M[CELL_E] = value

    @property
    def inhibitory_sum(self) -> float:
        return float(self.M[CELL_I])

    @inhibitory_sum.setter
    def inhibitory_sum(self, value: float) -> None:
        if value < 0:
            raise ValueError("I 는 크기 합이므로 음수가 될 수 없다")
        self.M[CELL_I] = value

    @property
    def bias(self) -> float:
        """b = b_base + correction_input."""
        return float(self.M[CELL_B])

    @bias.setter
    def bias(self, value: float) -> None:
        self.M[CELL_B] = value

    # --- 일시 상태 --------------------------------------------------------
    def reset_transient(self) -> None:
        """u, E, I, b, 발화, 교정 입력, 감각 주입을 초기화한다.

        부작용: 행렬의 u/E/I/b 셀을 덮어쓴다. threshold, P, 위치는 유지한다.
        초기화 후 b 는 b_base 로 돌아간다 (명세 [5]).
        """
        M = self.M
        M[CELL_U] = 0.0
        M[CELL_E] = 0.0
        M[CELL_I] = 0.0
        self.correction_input = 0.0
        self.store.external_drive[self.neuron_id] = 0.0
        M[CELL_B] = self.b_base
        self.fired = 0

    def metadata(self) -> dict[str, Any]:
        """행렬 밖 메타데이터 dict (진단/저장용, 부작용 없음)."""
        return {
            "neuron_id": self.neuron_id,
            "name": self.name,
            "area": self.area,
            "layer": self.layer,
            "cell_type": self.cell_type,
            "preferred_orientation_deg": self.preferred_orientation_deg,
            "receptive_field": (
                self.receptive_field.to_dict() if self.receptive_field else None
            ),
            "position_space": self.position_space,
            "fired": int(self.fired),
            "observation_tick": self.observation_tick,
            "incoming_connection_ids": list(self.incoming_connection_ids),
            "outgoing_connection_ids": list(self.outgoing_connection_ids),
        }


class Population:
    """뉴런 집합. 이름 -> 인덱스 사상과 상태 저장소를 함께 관리한다."""

    def __init__(self) -> None:
        self.store = NeuronStateStore()
        self.neurons: list[Neuron] = []
        self._by_name: dict[str, int] = {}
        self._specs: list[dict[str, Any]] = []
        self._finalized = False

    # --- 구성 -------------------------------------------------------------
    def declare(self, name: str, **kwargs: Any) -> None:
        """뉴런 1개를 등록 예약한다. 실제 행렬은 ``finalize`` 에서 만들어진다."""
        if self._finalized:
            raise RuntimeError("finalize() 이후에는 뉴런을 추가할 수 없다")
        if name in self._by_name:
            raise ValueError(f"중복 뉴런 이름: {name}")
        self._by_name[name] = len(self._specs)
        self._specs.append({"name": name, **kwargs})

    def finalize(self) -> None:
        """상태 배열을 할당하고 Neuron 객체(행렬 뷰)를 만든다."""
        n = len(self._specs)
        self.store.allocate(n)
        self.neurons = []
        for idx, spec in enumerate(self._specs):
            spec = dict(spec)
            pos = spec.pop("position", (0.0, 0.0, 0.0))
            threshold = spec.pop("threshold", 0.5)
            transmission = spec.pop("transmission", 1.0)
            b_base = spec.pop("b_base", 0.0)
            observation_tick = spec.pop("observation_tick", 0)
            neuron = Neuron(neuron_id=idx, store=self.store, **spec)
            neuron.b_base = b_base
            neuron.observation_tick = observation_tick
            neuron.set_position(*pos)
            neuron.threshold = threshold
            neuron.transmission = transmission
            neuron.bias = b_base
            self.neurons.append(neuron)
        self._finalized = True

    # --- 조회 -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.neurons) if self._finalized else len(self._specs)

    def index(self, name: str) -> int:
        return self._by_name[name]

    def by_name(self, name: str) -> Neuron:
        return self.neurons[self._by_name[name]]

    def __getitem__(self, key: int | str) -> Neuron:
        if isinstance(key, str):
            return self.by_name(key)
        return self.neurons[key]

    def names(self) -> list[str]:
        return [n.name for n in self.neurons]

    def ids_in_layer(self, layer: str) -> list[int]:
        return [n.neuron_id for n in self.neurons if n.layer == layer]

    def reset_transient(self) -> None:
        """모든 뉴런의 일시 상태 초기화 (벡터화). 학습 가중치와 무관하다."""
        st = self.store
        st.states[:, 1, 0] = 0.0   # u
        st.states[:, 2, 0] = 0.0   # E
        st.states[:, 2, 1] = 0.0   # I
        st.correction_input[:] = 0.0
        st.external_drive[:] = 0.0
        st.fired[:] = 0
        st.states[:, 2, 2] = st.b_base

    # --- 벡터 파라미터 ----------------------------------------------------
    def b_base_vector(self) -> np.ndarray:
        """shape (N,) float64 뷰 (복사가 아니다)."""
        return self.store.b_base

    def observation_tick_vector(self) -> np.ndarray:
        """shape (N,) int64 뷰."""
        return self.store.observation_tick

    def fired_vector(self) -> np.ndarray:
        return self.store.fired
