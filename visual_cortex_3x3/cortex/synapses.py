"""synapses.py -- 희소 edge table 과 인덱스.

명세 3절의 필드를 모두 갖춘다::

    synapse_id, src_id, dst_id, src_area, dst_area, target_layer,
    target_compartment, receptor_type, weight, weight_unit,
    base_delay_ms, effective_delay_steps, plasticity_rule,
    plasticity_state_id, active

설계 선택
---------
* 열 단위 NumPy 배열 + ``scipy.sparse`` CSR 인덱스. 전체 N×N 밀집 행렬을
  만들지 않는다.
* ``weight`` 는 **비음수 크기**다. 흥분/억제 부호는 **발신 뉴런의 Dale 유형**이
  결정하며, 학습으로 부호가 뒤집히지 않는다 (``clip(w, 0, w_max)``).
  전도도 모드에서 억제성 시냅스의 Δg 를 음수로 만들지 않는다. 실제 전류 방향은
  역전전위 ``E_rev`` 와 막전위가 결정한다.
* ``weight_unit`` 은 엔진 모드에 따른다: ``conductance_lif`` 는 nS,
  ``sum_threshold`` 는 무차원.
* ``plasticity_state_id`` 는 가소성 규칙별 상태 배열의 행 인덱스다 (-1 이면 없음).
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # scipy 는 필수 의존성이다. 없으면 즉시 알린다.
    from scipy import sparse as _sparse
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "SciPy 가 필요하다. `python -m pip install -r requirements.txt` 를 실행하라."
    ) from _exc


class SynapseTable:
    """희소 시냅스 테이블.

    ``add_block`` 으로 규칙별 시냅스를 덩어리로 추가하고, 모두 추가한 뒤
    ``build_indices()`` 를 호출해 CSR 인덱스를 만든다.
    """

    COLUMNS: tuple[str, ...] = (
        "synapse_id", "src_id", "dst_id", "src_area", "dst_area",
        "target_layer", "target_compartment", "receptor_type",
        "weight", "weight_unit", "base_delay_ms", "effective_delay_steps",
        "plasticity_rule", "plasticity_state_id", "active",
    )

    def __init__(self, n_neurons: int, weight_unit: str) -> None:
        self.n_neurons = int(n_neurons)
        self.weight_unit = str(weight_unit)
        self._blocks: list[dict[str, np.ndarray]] = []
        self._built = False

        self.src_id = np.zeros(0, dtype=np.int32)
        self.dst_id = np.zeros(0, dtype=np.int32)
        self.src_area = np.zeros(0, dtype=np.int16)
        self.dst_area = np.zeros(0, dtype=np.int16)
        self.target_layer = np.zeros(0, dtype=np.int16)
        self.target_compartment = np.zeros(0, dtype=np.int8)
        self.receptor_type = np.zeros(0, dtype=np.int8)
        self.weight = np.zeros(0, dtype=np.float64)
        self.base_delay_ms = np.zeros(0, dtype=np.float64)
        self.effective_delay_steps = np.zeros(0, dtype=np.int32)
        self.plasticity_rule = np.zeros(0, dtype=np.int8)
        self.plasticity_state_id = np.zeros(0, dtype=np.int64)
        self.active = np.zeros(0, dtype=bool)
        self.rule_index = np.zeros(0, dtype=np.int16)   # 어느 배선 규칙에서 나왔는지

        # 발신 뉴런의 Dale 부호 (+1/-1). 전류 부호가 아니라 **연결 유형**이다.
        self.src_dale_sign = np.zeros(0, dtype=np.int8)

        # CSR 인덱스
        self.in_ptr = np.zeros(self.n_neurons + 1, dtype=np.int64)
        self.in_syn = np.zeros(0, dtype=np.int64)
        self.out_ptr = np.zeros(self.n_neurons + 1, dtype=np.int64)
        self.out_syn = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------
    def add_block(self, *, src_id: np.ndarray, dst_id: np.ndarray,
                  src_area: np.ndarray, dst_area: np.ndarray,
                  target_layer: np.ndarray, target_compartment: int,
                  receptor_type: int, weight: np.ndarray,
                  base_delay_ms: np.ndarray, effective_delay_steps: np.ndarray,
                  plasticity_rule: int, src_dale_sign: np.ndarray,
                  rule_index: int) -> int:
        """시냅스 여러 개를 한 번에 추가한다 (부작용: 내부 블록 목록 확장)."""
        if self._built:
            raise RuntimeError("build_indices() 이후에는 시냅스를 추가할 수 없다")
        w = np.asarray(weight, dtype=np.float64)
        if np.any(w < 0.0):
            raise ValueError(
                "weight 는 비음수 크기여야 한다. 흥분/억제 부호는 발신 뉴런의 "
                "Dale 유형이 결정한다."
            )
        d = np.asarray(effective_delay_steps, dtype=np.int32)
        if np.any(d < 1):
            raise ValueError("effective_delay_steps 는 1 이상이어야 한다 (0 지연 재귀 방지)")
        n = w.size
        self._blocks.append({
            "src_id": np.asarray(src_id, dtype=np.int32),
            "dst_id": np.asarray(dst_id, dtype=np.int32),
            "src_area": np.asarray(src_area, dtype=np.int16),
            "dst_area": np.asarray(dst_area, dtype=np.int16),
            "target_layer": np.asarray(target_layer, dtype=np.int16),
            "target_compartment": np.full(n, int(target_compartment), dtype=np.int8),
            "receptor_type": np.full(n, int(receptor_type), dtype=np.int8),
            "weight": w,
            "base_delay_ms": np.asarray(base_delay_ms, dtype=np.float64),
            "effective_delay_steps": d,
            "plasticity_rule": np.full(n, int(plasticity_rule), dtype=np.int8),
            "active": np.ones(n, dtype=bool),
            "src_dale_sign": np.asarray(src_dale_sign, dtype=np.int8),
            "rule_index": np.full(n, int(rule_index), dtype=np.int16),
        })
        return n

    def pending_count(self) -> int:
        return int(sum(b["weight"].size for b in self._blocks))

    def build_indices(self) -> None:
        """블록을 이어 붙이고 in/out CSR 인덱스를 만든다."""
        if self._built:
            return
        if not self._blocks:
            self._built = True
            self.plasticity_state_id = np.zeros(0, dtype=np.int64)
            return
        keys = [k for k in self._blocks[0]]
        for k in keys:
            setattr(self, k, np.concatenate([b[k] for b in self._blocks]))
        self._blocks = []
        k_total = self.weight.size
        self.plasticity_state_id = np.where(
            self.plasticity_rule > 0, np.arange(k_total, dtype=np.int64), -1
        )

        order_in = np.argsort(self.dst_id.astype(np.int64), kind="stable")
        self.in_syn = order_in.astype(np.int64)
        counts = np.bincount(self.dst_id.astype(np.int64), minlength=self.n_neurons)
        self.in_ptr = np.zeros(self.n_neurons + 1, dtype=np.int64)
        np.cumsum(counts, out=self.in_ptr[1:])

        order_out = np.argsort(self.src_id.astype(np.int64), kind="stable")
        self.out_syn = order_out.astype(np.int64)
        counts_o = np.bincount(self.src_id.astype(np.int64), minlength=self.n_neurons)
        self.out_ptr = np.zeros(self.n_neurons + 1, dtype=np.int64)
        np.cumsum(counts_o, out=self.out_ptr[1:])
        self._built = True

    @property
    def built(self) -> bool:
        return self._built

    def __len__(self) -> int:
        return int(self.weight.size) if self._built else self.pending_count()

    @property
    def n_synapses(self) -> int:
        return len(self)

    # ------------------------------------------------------------------
    def outgoing_of(self, neuron_id: int) -> np.ndarray:
        i = int(neuron_id)
        return self.out_syn[self.out_ptr[i]:self.out_ptr[i + 1]]

    def incoming_of(self, neuron_id: int) -> np.ndarray:
        i = int(neuron_id)
        return self.in_syn[self.in_ptr[i]:self.in_ptr[i + 1]]

    def row(self, synapse_id: int) -> dict[str, Any]:
        s = int(synapse_id)
        return {
            "synapse_id": s,
            "src_id": int(self.src_id[s]),
            "dst_id": int(self.dst_id[s]),
            "src_area": int(self.src_area[s]),
            "dst_area": int(self.dst_area[s]),
            "target_layer": int(self.target_layer[s]),
            "target_compartment": int(self.target_compartment[s]),
            "receptor_type": int(self.receptor_type[s]),
            "weight": float(self.weight[s]),
            "weight_unit": self.weight_unit,
            "base_delay_ms": float(self.base_delay_ms[s]),
            "effective_delay_steps": int(self.effective_delay_steps[s]),
            "plasticity_rule": int(self.plasticity_rule[s]),
            "plasticity_state_id": int(self.plasticity_state_id[s]),
            "active": bool(self.active[s]),
            "src_dale_sign": int(self.src_dale_sign[s]),
            "rule_index": int(self.rule_index[s]),
        }

    def connectivity_matrix(self) -> "_sparse.csr_matrix":
        """(N,N) CSR 희소 행렬. 값은 **부호를 적용한** 가중치다 (시각화/집계용).

        이 행렬은 계산 경로에서 쓰지 않는다. 계산은 이벤트 기반이다.
        """
        data = self.weight * self.src_dale_sign.astype(np.float64)
        return _sparse.csr_matrix(
            (data, (self.src_id.astype(np.int64), self.dst_id.astype(np.int64))),
            shape=(self.n_neurons, self.n_neurons),
        )

    # --- Dale 제약과 범위 ----------------------------------------------
    def clip_weights(self, w_min: float, w_max: float) -> dict[str, int]:
        """가중치를 [w_min, w_max] 로 자른다. 부호(연결 유형)는 바뀌지 않는다.

        Returns
        -------
        dict : 하한/상한에 걸린 시냅스 수
        """
        if w_min < 0.0:
            raise ValueError("w_min 은 0 이상이어야 한다 (Dale 부호 보존)")
        lo = int(np.count_nonzero(self.weight < w_min))
        hi = int(np.count_nonzero(self.weight > w_max))
        np.clip(self.weight, w_min, w_max, out=self.weight)
        return {"clipped_low": lo, "clipped_high": hi}

    def dale_violations(self, neuron_dale_sign: np.ndarray) -> np.ndarray:
        """발신 뉴런의 Dale 부호와 시냅스에 기록된 부호가 다른 행을 찾는다."""
        expected = np.asarray(neuron_dale_sign, dtype=np.int8)[self.src_id]
        return np.nonzero(expected != self.src_dale_sign)[0]

    def summary(self, id_space: Any = None) -> dict[str, Any]:
        """집계 요약 (검증 9번, 보고서용)."""
        if not self._built:
            return {"built": False, "pending": self.pending_count()}
        out: dict[str, Any] = {
            "built": True,
            "n_synapses": int(self.weight.size),
            "n_active": int(np.count_nonzero(self.active)),
            "weight_unit": self.weight_unit,
            "weight_min": float(self.weight.min()) if self.weight.size else None,
            "weight_max": float(self.weight.max()) if self.weight.size else None,
            "weight_mean": float(self.weight.mean()) if self.weight.size else None,
            "n_excitatory_sources": int(np.count_nonzero(self.src_dale_sign > 0)),
            "n_inhibitory_sources": int(np.count_nonzero(self.src_dale_sign < 0)),
            "delay_steps_min": int(self.effective_delay_steps.min()) if self.weight.size else None,
            "delay_steps_max": int(self.effective_delay_steps.max()) if self.weight.size else None,
            "base_delay_ms_min": float(self.base_delay_ms.min()) if self.weight.size else None,
            "base_delay_ms_max": float(self.base_delay_ms.max()) if self.weight.size else None,
            "n_by_rule_index": {int(k): int(v) for k, v in
                                zip(*np.unique(self.rule_index, return_counts=True))},
        }
        return out

    def state_dict(self) -> dict[str, np.ndarray]:
        """체크포인트용 배열 묶음 (가중치와 활성 플래그 포함)."""
        return {
            "src_id": self.src_id, "dst_id": self.dst_id,
            "src_area": self.src_area, "dst_area": self.dst_area,
            "target_layer": self.target_layer,
            "target_compartment": self.target_compartment,
            "receptor_type": self.receptor_type, "weight": self.weight,
            "base_delay_ms": self.base_delay_ms,
            "effective_delay_steps": self.effective_delay_steps,
            "plasticity_rule": self.plasticity_rule,
            "plasticity_state_id": self.plasticity_state_id,
            "active": self.active, "src_dale_sign": self.src_dale_sign,
            "rule_index": self.rule_index,
        }

    def load_state_dict(self, d: dict[str, np.ndarray]) -> None:
        for k, v in d.items():
            setattr(self, k, np.asarray(v))
        self._blocks = []
        self._built = False
        self.build_indices()


__all__ = ["SynapseTable"]
