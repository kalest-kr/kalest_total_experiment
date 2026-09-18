"""records.py -- 3x3 뉴런 기록 인터페이스와 타입이 정해진 상태 배열.

명세 2절의 배치를 **그대로** 구현한다 (행/열은 사용자 설명 기준 1부터,
Python 인덱스는 0부터).

===========  ==============  =================================================
사용자 표기  Python 인덱스   의미
===========  ==============  =================================================
1행 1열      ``[0][0]``      x 좌표, mm
1행 2열      ``[0][1]``      y 좌표, mm
1행 3열      ``[0][2]``      z 좌표, mm
2행 1열      ``[1][0]``      **출력 연결 목록** (:class:`OutgoingConnectionsView`)
2행 2열      ``[1][1]``      현재 발화 임계값 θ
2행 3열      ``[1][2]``      기저 출력 이득 P
3행 1열      ``[2][0]``      입력 이벤트 로그 (:class:`InputLogView`)
3행 2열      ``[2][1]``      동적 상태 참조 (:class:`DynamicStateView`)
3행 3열      ``[2][2]``      메타데이터 참조 (:class:`MetadataView`)
===========  ==============  =================================================

반드시 지킨 제약
----------------
* ``[1][0]`` 에 **입력 총합을 저장하지 않는다.** 그 칸은 출력 연결 목록이다.
  합산값은 :mod:`cortex.dynamics` 의 지역 변수이며, 모니터링이 필요하면
  진단 기록에 단위와 함께 남기되 진실의 원본으로 중복 관리하지 않는다.
* ``[2][0]`` 의 입력 로그는 발화 후 삭제하지 않는다. 현재 연산에 필요한
  **작업 버퍼**(전도도/활성화 누적)는 :class:`NeuronArrays` 의 별도 배열이며
  장기 로그와 분리되어 있다.
* 이 구조는 서로 다른 타입을 담는 **기록 인터페이스**다. ``float32[3,3]``
  수치 행렬이 아니다. :meth:`NeuronRecord.as_matrix` 는 object 배열을 준다.
* 3x3 뷰는 :class:`NeuronArrays` 의 **같은 메모리를 참조**한다. 미러 복사본을
  만들지 않으므로 값 불일치가 생기지 않는다 (쓰기도 배열에 반영된다).
* 이웃 뉴런 객체를 재귀적으로 담지 않는다. 안정적인 정수 ID 만 쓴다.
* ``P`` 는 **출력 이득**이다. 방출 확률이 필요하면 ``release_probability`` 로
  따로 이름 붙이고 RNG 와 역할을 분리한다 (이번 구현에서는 미사용,
  IMPLEMENTATION_STATUS.md 참조).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .ids import COMPARTMENT_NAMES, IdSpace, N_COMPARTMENTS

if TYPE_CHECKING:  # pragma: no cover - 타입 검사 전용
    from .events import EventLog
    from .synapses import SynapseTable


class NeuronArrays:
    """모든 뉴런의 상태를 담는 타입이 정해진 배열 묶음 (Structure-of-Arrays).

    행 인덱스 == ``neuron_id`` 다. 배열은 한 번만 할당되고 이후 shape 가
    바뀌지 않는다 (뷰가 무효화되지 않게 하기 위함).

    Parameters
    ----------
    n : int
        뉴런 수.
    n_receptors : int
        수용체 종류 수. 전도도 배열이 ``(n, 3, n_receptors)`` 가 된다.
    """

    def __init__(self, n: int, n_receptors: int) -> None:
        n = int(n)
        r = int(n_receptors)
        self.n = n
        self.n_receptors = r

        # --- 구조 (고정) ------------------------------------------------
        self.position_mm = np.zeros((n, 3), dtype=np.float64)     # [0][0..2]
        self.threshold = np.zeros(n, dtype=np.float64)            # [1][1]
        self.output_gain_P = np.ones(n, dtype=np.float64)         # [1][2]
        self.area_id = np.full(n, -1, dtype=np.int32)
        self.layer_id = np.full(n, -1, dtype=np.int32)
        self.cell_type_id = np.full(n, -1, dtype=np.int32)
        self.dale_sign = np.zeros(n, dtype=np.int8)               # +1 흥분 / -1 억제
        self.has_compartment = np.zeros((n, N_COMPARTMENTS), dtype=bool)
        self.has_compartment[:, 0] = True                          # soma 는 항상 존재

        # --- 세포 유형 파라미터 (뉴런별로 펼쳐 둔다) ---------------------
        self.C_pF = np.ones((n, N_COMPARTMENTS), dtype=np.float64)
        self.gL_nS = np.ones((n, N_COMPARTMENTS), dtype=np.float64)
        self.EL_mV = np.full((n, N_COMPARTMENTS), -70.0, dtype=np.float64)
        self.g_couple_nS = np.zeros((n, N_COMPARTMENTS), dtype=np.float64)
        # g_couple_nS[:, 1] = soma-basal, [:, 2] = soma-apical, [:, 0] 미사용
        self.V_reset_mV = np.full(n, -65.0, dtype=np.float64)
        self.t_ref_ms = np.zeros(n, dtype=np.float64)
        self.target_rate_hz = np.zeros(n, dtype=np.float64)
        self.refractory_discards_input = np.zeros(n, dtype=bool)

        # --- 동적 상태 ([2][1] 이 참조) ----------------------------------
        self.V_mV = np.full((n, N_COMPARTMENTS), -70.0, dtype=np.float64)
        self.g_nS = np.zeros((n, N_COMPARTMENTS, r), dtype=np.float64)
        self.refractory_until_ms = np.full(n, -np.inf, dtype=np.float64)
        self.last_spike_ms = np.full(n, -np.inf, dtype=np.float64)
        self.spike_count = np.zeros(n, dtype=np.int64)
        self.rate_estimate_hz = np.zeros(n, dtype=np.float64)
        self.trace_pre = np.zeros(n, dtype=np.float64)    # STDP 전 흔적
        self.trace_post = np.zeros(n, dtype=np.float64)   # STDP 후 흔적
        #: 사용자/실험이 직접 주는 지속 외부 전류 [pA]. 엔진은 이 배열을 덮어쓰지
        #: 않는다. 망막 구동 전류는 엔진 내부에서 별도로 더해진다.
        self.Iext_pA = np.zeros((n, N_COMPARTMENTS), dtype=np.float64)

        # --- sum_threshold 모드의 작업 버퍼 -----------------------------
        # 장기 로그와 분리된 "현재 처리 구간" 누적값이다. 매 구간 시작에 0 으로
        # 초기화되며, 전체 과거 로그의 합을 여기에 넣지 않는다.
        self.interval_activation = np.zeros(n, dtype=np.float64)

        # --- 발화 판정 직전 상태 스냅샷 (진단용, 판정 전에 채워진다) ------
        self.last_decision_value = np.zeros(n, dtype=np.float64)
        self.last_decision_threshold = np.zeros(n, dtype=np.float64)
        self.fired = np.zeros(n, dtype=np.int8)

        # --- 로그 처리 위치 --------------------------------------------
        # 같은 이벤트를 같은 상태에 두 번 주입하지 않기 위한 위치 표시.
        self.last_consumed_event = np.full(n, -1, dtype=np.int64)

        # --- 시야/특징 메타데이터 ([2][2] 가 참조) -----------------------
        self.visual_field_xy_deg = np.full((n, 2), np.nan, dtype=np.float64)
        self.rf_sigma_deg = np.full(n, np.nan, dtype=np.float64)
        self.pref_orientation_rad = np.full(n, np.nan, dtype=np.float64)
        self.pref_phase_rad = np.full(n, np.nan, dtype=np.float64)
        self.ocular_dominance = np.zeros(n, dtype=np.float64)   # -1 왼눈 .. +1 오른눈
        self.eye_id = np.full(n, -1, dtype=np.int8)             # -1 해당 없음
        self.on_off = np.zeros(n, dtype=np.int8)                # +1 ON, -1 OFF, 0 해당없음
        self.channel_id = np.full(n, -1, dtype=np.int32)        # 망막 채널 인덱스
        self.hypercolumn_uv = np.full((n, 2), np.nan, dtype=np.float64)
        #: 영역 표면 좌표 (u,v) [mm]. 영역별 오프셋을 뺀 국소 좌표다.
        self.surface_uv_mm = np.full((n, 2), np.nan, dtype=np.float64)
        #: 피라미드 첨단수상돌기 다발이 도달하는 깊이 [mm] (L1 접점 모형).
        #: 세포체가 없는 층이라도 apical 접점의 공간 위치를 갖게 한다.
        self.apical_depth_mm = np.full(n, np.nan, dtype=np.float64)

        # --- 출력 연결 인덱스 ([1][0] 이 참조) ---------------------------
        # CSR 형태: out_syn[out_ptr[i]:out_ptr[i+1]] 가 뉴런 i 의 synapse_id 들.
        self.out_ptr = np.zeros(n + 1, dtype=np.int64)
        self.out_syn = np.zeros(0, dtype=np.int64)
        self._outgoing_built = False

    # ------------------------------------------------------------------
    def set_outgoing_index(self, out_ptr: np.ndarray, out_syn: np.ndarray) -> None:
        """시냅스 테이블에서 만든 CSR 출력 인덱스를 연결한다."""
        out_ptr = np.asarray(out_ptr, dtype=np.int64)
        if out_ptr.shape != (self.n + 1,):
            raise ValueError(f"out_ptr shape 는 ({self.n + 1},) 여야 한다: {out_ptr.shape}")
        self.out_ptr = out_ptr
        self.out_syn = np.asarray(out_syn, dtype=np.int64)
        self._outgoing_built = True

    @property
    def outgoing_built(self) -> bool:
        return self._outgoing_built

    def nbytes(self) -> int:
        total = 0
        for v in vars(self).values():
            if isinstance(v, np.ndarray):
                total += int(v.nbytes)
        return total


# ----------------------------------------------------------------------
# 3x3 칸을 채우는 참조 뷰들
# ----------------------------------------------------------------------
class OutgoingConnectionsView:
    """3x3 ``[1][0]``: 이 뉴런의 **출력 연결 목록**.

    시냅스 객체를 복사해 담지 않고 ``synapse_id`` 배열을 참조한다.
    한 뉴런은 여러 출력 연결을 가질 수 있다.
    """

    __slots__ = ("_pop", "_i")

    def __init__(self, population: "NeuronPopulation", neuron_id: int) -> None:
        self._pop = population
        self._i = int(neuron_id)

    @property
    def synapse_ids(self) -> np.ndarray:
        """shape (m,) int64. 배열의 사본이 아니라 뷰(slice)다."""
        a = self._pop.arrays
        if not a.outgoing_built:
            raise RuntimeError(
                "출력 연결 인덱스가 아직 만들어지지 않았다. "
                "SynapseTable.build_indices() 후 NeuronArrays.set_outgoing_index() 를 호출하라."
            )
        lo, hi = int(a.out_ptr[self._i]), int(a.out_ptr[self._i + 1])
        return a.out_syn[lo:hi]

    def __len__(self) -> int:
        return int(self.synapse_ids.size)

    def rows(self) -> list[dict[str, Any]]:
        """연결별 전체 필드를 dict 목록으로 (조회용, 부작용 없음)."""
        syn = self._pop.synapses
        if syn is None:
            raise RuntimeError("SynapseTable 이 population 에 연결되어 있지 않다")
        return [syn.row(int(s)) for s in self.synapse_ids]

    def targets(self) -> np.ndarray:
        syn = self._pop.synapses
        if syn is None:
            raise RuntimeError("SynapseTable 이 population 에 연결되어 있지 않다")
        return syn.dst_id[self.synapse_ids]

    def __repr__(self) -> str:
        try:
            n = len(self)
        except RuntimeError:
            n = -1
        return f"<OutgoingConnectionsView neuron={self._i} n_synapses={n}>"


class InputLogView:
    """3x3 ``[2][0]``: 이 뉴런의 **입력 이벤트 로그 조회 뷰**.

    모든 뉴런이 무한 길이 Python 리스트를 들고 있지 않도록, 실제 저장은
    :class:`cortex.events.EventLog` 의 연속 배열(또는 디스크 청크)에 있고
    여기서는 인덱스로 조회만 한다. **발화해도 지워지지 않는다.**
    """

    __slots__ = ("_pop", "_i")

    def __init__(self, population: "NeuronPopulation", neuron_id: int) -> None:
        self._pop = population
        self._i = int(neuron_id)

    def _log(self) -> "EventLog":
        log = self._pop.event_log
        if log is None:
            raise RuntimeError(
                "EventLog 가 population 에 연결되어 있지 않다. "
                "NeuronPopulation.attach(event_log=...) 를 먼저 호출하라."
            )
        return log

    def count(self) -> int:
        return self._log().count_for_neuron(self._i)

    def rows(self, t_from_ms: float | None = None, t_to_ms: float | None = None,
             limit: int | None = None) -> dict[str, np.ndarray]:
        """시간 구간의 도착 이벤트를 열 배열 dict 로 돌려준다 (부작용 없음)."""
        return self._log().query_neuron(self._i, t_from_ms, t_to_ms, limit)

    @property
    def last_consumed_event(self) -> int:
        """이 뉴런의 상태에 이미 반영된 마지막 event_id (-1 이면 없음)."""
        return int(self._pop.arrays.last_consumed_event[self._i])

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        try:
            c = self.count()
        except RuntimeError:
            c = -1
        return (f"<InputLogView neuron={self._i} n_events={c} "
                f"last_consumed={int(self._pop.arrays.last_consumed_event[self._i])}>")


class DynamicStateView:
    """3x3 ``[2][1]``: 막전위·전도도·불응기·흔적 등 동적 상태 **참조**.

    모든 속성은 :class:`NeuronArrays` 를 읽고 쓴다 (사본 아님).
    """

    __slots__ = ("_pop", "_i")

    def __init__(self, population: "NeuronPopulation", neuron_id: int) -> None:
        self._pop = population
        self._i = int(neuron_id)

    # 막전위 --------------------------------------------------------
    @property
    def V_mV(self) -> np.ndarray:
        """shape (3,) 뷰. 인덱스는 soma/basal/apical."""
        return self._pop.arrays.V_mV[self._i]

    @property
    def V_soma_mV(self) -> float:
        return float(self._pop.arrays.V_mV[self._i, 0])

    @V_soma_mV.setter
    def V_soma_mV(self, value: float) -> None:
        self._pop.arrays.V_mV[self._i, 0] = float(value)

    # 전도도 --------------------------------------------------------
    @property
    def g_nS(self) -> np.ndarray:
        """shape (3, n_receptors) 뷰. 값은 비음수여야 한다."""
        return self._pop.arrays.g_nS[self._i]

    def g_of(self, compartment: str, receptor: str) -> float:
        ids = self._pop.ids
        c = ids.compartments.id_of(compartment)
        r = ids.receptors.id_of(receptor)
        return float(self._pop.arrays.g_nS[self._i, c, r])

    # 기타 ----------------------------------------------------------
    @property
    def refractory_until_ms(self) -> float:
        return float(self._pop.arrays.refractory_until_ms[self._i])

    @property
    def last_spike_ms(self) -> float:
        return float(self._pop.arrays.last_spike_ms[self._i])

    @property
    def rate_estimate_hz(self) -> float:
        return float(self._pop.arrays.rate_estimate_hz[self._i])

    @property
    def trace_pre(self) -> float:
        return float(self._pop.arrays.trace_pre[self._i])

    @property
    def trace_post(self) -> float:
        return float(self._pop.arrays.trace_post[self._i])

    @property
    def interval_activation(self) -> float:
        """sum_threshold 모드의 **현재 처리 구간** 누적값 (작업 버퍼, 무차원).

        장기 로그의 총합이 아니다. 매 구간 시작에 0 으로 초기화된다.
        """
        return float(self._pop.arrays.interval_activation[self._i])

    @property
    def fired(self) -> int:
        return int(self._pop.arrays.fired[self._i])

    def snapshot(self) -> dict[str, Any]:
        """읽기 전용 사본 (측정 함수용). 모델 상태를 바꾸지 않는다."""
        a = self._pop.arrays
        i = self._i
        return {
            "V_mV": a.V_mV[i].copy(),
            "g_nS": a.g_nS[i].copy(),
            "refractory_until_ms": float(a.refractory_until_ms[i]),
            "last_spike_ms": float(a.last_spike_ms[i]),
            "rate_estimate_hz": float(a.rate_estimate_hz[i]),
            "trace_pre": float(a.trace_pre[i]),
            "trace_post": float(a.trace_post[i]),
            "interval_activation": float(a.interval_activation[i]),
            "fired": int(a.fired[i]),
            "last_decision_value": float(a.last_decision_value[i]),
            "last_decision_threshold": float(a.last_decision_threshold[i]),
        }

    def __repr__(self) -> str:
        a = self._pop.arrays
        return (f"<DynamicStateView neuron={self._i} V_soma={a.V_mV[self._i, 0]:.3f}mV "
                f"g_sum={a.g_nS[self._i].sum():.4f}nS>")


class MetadataView:
    """3x3 ``[2][2]``: 영역·층·세포 유형·수용장·선호 특징 등 **메타데이터 참조**."""

    __slots__ = ("_pop", "_i")

    def __init__(self, population: "NeuronPopulation", neuron_id: int) -> None:
        self._pop = population
        self._i = int(neuron_id)

    @property
    def area(self) -> str:
        return self._pop.ids.areas.name_of(self._pop.arrays.area_id[self._i])

    @property
    def layer(self) -> str:
        return self._pop.ids.layers.name_of(self._pop.arrays.layer_id[self._i])

    @property
    def cell_type(self) -> str:
        return self._pop.ids.cell_types.name_of(self._pop.arrays.cell_type_id[self._i])

    @property
    def dale(self) -> str:
        return "excitatory" if self._pop.arrays.dale_sign[self._i] > 0 else "inhibitory"

    @property
    def compartments(self) -> tuple[str, ...]:
        mask = self._pop.arrays.has_compartment[self._i]
        return tuple(name for name, ok in zip(COMPARTMENT_NAMES, mask) if bool(ok))

    @property
    def visual_field_xy_deg(self) -> np.ndarray:
        """shape (2,) 시야 좌표 [deg]. 피질 mm 좌표와 **다른 공간**이다."""
        return self._pop.arrays.visual_field_xy_deg[self._i]

    @property
    def receptive_field(self) -> dict[str, Any]:
        a = self._pop.arrays
        i = self._i
        return {
            "center_xy_deg": a.visual_field_xy_deg[i].tolist(),
            "sigma_deg": float(a.rf_sigma_deg[i]),
            "space": "visual_field_deg",
        }

    @property
    def preferred(self) -> dict[str, Any]:
        a = self._pop.arrays
        i = self._i
        return {
            "orientation_rad": float(a.pref_orientation_rad[i]),
            "phase_rad": float(a.pref_phase_rad[i]),
            "on_off": int(a.on_off[i]),
            "channel_id": int(a.channel_id[i]),
            "ocular_dominance": float(a.ocular_dominance[i]),
            "eye_id": int(a.eye_id[i]),
            "hypercolumn_uv": a.hypercolumn_uv[i].tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "neuron_id": self._i,
            "area": self.area,
            "layer": self.layer,
            "cell_type": self.cell_type,
            "dale": self.dale,
            "compartments": list(self.compartments),
            "cortical_xyz_mm": self._pop.arrays.position_mm[self._i].tolist(),
            "receptive_field_parameters": self.receptive_field,
            "preferred": self.preferred,
        }

    def __repr__(self) -> str:
        return (f"<MetadataView neuron={self._i} {self.area}/{self.layer}/"
                f"{self.cell_type} ({self.dale})>")


# ----------------------------------------------------------------------
@dataclass(frozen=True)
class NeuronRecord:
    """뉴런 1개의 3x3 기록 인터페이스.

    ``population`` 의 배열을 참조하는 **가벼운 핸들**이다. 상태를 복사해 갖지
    않으므로 배열을 바꾸면 이 객체를 통해 본 값도 즉시 바뀐다.
    """

    population: "NeuronPopulation"
    neuron_id: int

    # --- 1행 ----------------------------------------------------------
    @property
    def x_mm(self) -> float:
        return float(self.population.arrays.position_mm[self.neuron_id, 0])

    @property
    def y_mm(self) -> float:
        return float(self.population.arrays.position_mm[self.neuron_id, 1])

    @property
    def z_mm(self) -> float:
        return float(self.population.arrays.position_mm[self.neuron_id, 2])

    # --- 2행 ----------------------------------------------------------
    @property
    def outgoing(self) -> OutgoingConnectionsView:
        return OutgoingConnectionsView(self.population, self.neuron_id)

    @property
    def threshold(self) -> float:
        """[1][1] 현재 발화 임계값 θ. 단위는 엔진 모드에 따른다 (mV 또는 무차원)."""
        return float(self.population.arrays.threshold[self.neuron_id])

    @threshold.setter
    def threshold(self, value: float) -> None:
        self.population.arrays.threshold[self.neuron_id] = float(value)

    @property
    def output_gain_P(self) -> float:
        """[1][2] 기저 출력 이득 P. 방출 확률이 아니다."""
        return float(self.population.arrays.output_gain_P[self.neuron_id])

    @output_gain_P.setter
    def output_gain_P(self, value: float) -> None:
        self.population.arrays.output_gain_P[self.neuron_id] = float(value)

    # --- 3행 ----------------------------------------------------------
    @property
    def input_log(self) -> InputLogView:
        return InputLogView(self.population, self.neuron_id)

    @property
    def dynamic_state(self) -> DynamicStateView:
        return DynamicStateView(self.population, self.neuron_id)

    @property
    def metadata(self) -> MetadataView:
        return MetadataView(self.population, self.neuron_id)

    # --- 3x3 조회 ------------------------------------------------------
    def as_matrix(self) -> np.ndarray:
        """명세 2절 배치의 3x3 **object 배열**.

        Returns
        -------
        np.ndarray, shape (3,3), dtype=object

        각 칸의 타입::

            [[float, float, float],
             [OutgoingConnectionsView, float, float],
             [InputLogView, DynamicStateView, MetadataView]]

        수치 행렬이 아니다. ``float32[3,3]`` 으로 바꾸려 하지 말 것.
        """
        m = np.empty((3, 3), dtype=object)
        m[0, 0] = self.x_mm
        m[0, 1] = self.y_mm
        m[0, 2] = self.z_mm
        m[1, 0] = self.outgoing
        m[1, 1] = self.threshold
        m[1, 2] = self.output_gain_P
        m[2, 0] = self.input_log
        m[2, 1] = self.dynamic_state
        m[2, 2] = self.metadata
        return m

    def as_matrix_description(self) -> list[list[str]]:
        """각 칸이 무엇인지 사람이 읽는 설명 (검증·문서용)."""
        return [
            ["x_mm", "y_mm", "z_mm"],
            ["outgoing_connection_list", "threshold_theta", "output_gain_P"],
            ["input_event_log_view", "dynamic_state_ref", "metadata_ref"],
        ]

    def describe(self) -> dict[str, Any]:
        """JSON 으로 저장 가능한 요약 (뷰 대신 요약값을 넣는다)."""
        return {
            "neuron_id": self.neuron_id,
            "cortical_xyz_mm": [self.x_mm, self.y_mm, self.z_mm],
            "n_outgoing": len(self.outgoing) if self.population.arrays.outgoing_built else None,
            "threshold": self.threshold,
            "output_gain_P": self.output_gain_P,
            "n_input_events": (self.input_log.count()
                               if self.population.event_log is not None else None),
            "dynamic_state": self.dynamic_state.snapshot(),
            "metadata": self.metadata.to_dict(),
        }


class NeuronPopulation:
    """뉴런 집단: 상태 배열 + ID 레지스트리 + (선택) 시냅스/로그 부착.

    3x3 기록 인터페이스는 :meth:`record` 로 얻는다.
    """

    def __init__(self, arrays: NeuronArrays, ids: IdSpace) -> None:
        self.arrays = arrays
        self.ids = ids
        self.synapses: "SynapseTable | None" = None
        self.event_log: "EventLog | None" = None
        self._name_of: list[str] | None = None

    def attach(self, synapses: "SynapseTable | None" = None,
               event_log: "EventLog | None" = None) -> None:
        """시냅스 테이블과 이벤트 로그를 연결한다 (부작용: 참조 설정)."""
        if synapses is not None:
            self.synapses = synapses
        if event_log is not None:
            self.event_log = event_log

    def __len__(self) -> int:
        return self.arrays.n

    def record(self, neuron_id: int) -> NeuronRecord:
        """뉴런 1개의 3x3 기록 인터페이스 핸들."""
        i = int(neuron_id)
        if not (0 <= i < self.arrays.n):
            raise IndexError(f"neuron_id 범위를 벗어났다: {neuron_id} (0..{self.arrays.n - 1})")
        return NeuronRecord(self, i)

    def records(self, neuron_ids: Sequence[int]) -> list[NeuronRecord]:
        return [self.record(i) for i in neuron_ids]

    # --- 집합 조회 ------------------------------------------------------
    def select(self, area: str | None = None, layer: str | None = None,
               cell_type: str | None = None) -> np.ndarray:
        """조건에 맞는 neuron_id 배열 (정렬됨, 부작용 없음)."""
        a = self.arrays
        mask = np.ones(a.n, dtype=bool)
        if area is not None:
            mask &= a.area_id == self.ids.areas.id_of(area)
        if layer is not None:
            mask &= a.layer_id == self.ids.layers.id_of(layer)
        if cell_type is not None:
            mask &= a.cell_type_id == self.ids.cell_types.id_of(cell_type)
        return np.nonzero(mask)[0]

    def counts_by_area_layer_type(self) -> dict[str, int]:
        """('area/layer/cell_type' -> 개수) 집계. 검증 9번에서 사용."""
        a = self.arrays
        out: dict[str, int] = {}
        for i in range(a.n):
            key = (f"{self.ids.areas.name_of(a.area_id[i])}/"
                   f"{self.ids.layers.name_of(a.layer_id[i])}/"
                   f"{self.ids.cell_types.name_of(a.cell_type_id[i])}")
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))


__all__ = [
    "NeuronArrays", "NeuronPopulation", "NeuronRecord",
    "OutgoingConnectionsView", "InputLogView", "DynamicStateView", "MetadataView",
]
