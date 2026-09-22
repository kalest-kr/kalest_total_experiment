"""connections.py -- 지속되는 연결 목록 ``ConnectionTable``.

명세 [3]-A. 각 연결은
``connection_id, pre_id, post_id, kind, weight, delay_ticks, trainable`` 을
저장한다. weight 는 **비음수 크기**이고 부호는 kind 가 결정한다.

설계 결정:

* 내부 저장은 NumPy 배열(구조 분해형 SoA)이다. 학습/합산이 전부 벡터화된다.
* 합산 순서를 뉴런 등록 순서와 무관하게 만들기 위해 ``finalize`` 에서
  ``(post_name, kind, pre_name, 생성순번)`` 사전식 정렬 순열을 계산해 둔다.
  E/I 누적은 항상 이 순열 순서로 수행하므로, 같은 논리 회로를 다른 순서로
  등록해도 부동소수점 합까지 동일하다 (명세 [5], T4).
* ``weight_version`` 은 가중치가 실제로 바뀔 때만 증가한다. 영상 1장의
  자유 실행 중에는 변하지 않는다 (명세 [3]-A).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

KIND_EXCITATORY = 0
KIND_INHIBITORY = 1
KIND_NAMES = {KIND_EXCITATORY: "excitatory", KIND_INHIBITORY: "inhibitory"}
KIND_CODES = {"excitatory": KIND_EXCITATORY, "inhibitory": KIND_INHIBITORY}
KIND_SIGN = np.array([+1.0, -1.0], dtype=np.float64)  # sign_k (명세 [9])


class ConnectionTable:
    """지속되는 연결 목록.

    부작용 정책: ``add`` / ``finalize`` / ``set_weights`` / ``apply_delta``
    만 상태를 바꾼다. 조회 메서드는 부작용이 없다.
    """

    def __init__(self, n_neurons: int, neuron_names: list[str] | None = None) -> None:
        self.n_neurons = int(n_neurons)
        self.neuron_names = list(neuron_names) if neuron_names is not None else [
            str(i) for i in range(n_neurons)
        ]
        self._rows: list[dict[str, Any]] = []
        self._finalized = False
        self.weight_version = 0

        # finalize 이후 채워지는 배열들
        self.connection_id: np.ndarray = np.zeros(0, dtype=np.int64)
        self.pre_id: np.ndarray = np.zeros(0, dtype=np.int64)
        self.post_id: np.ndarray = np.zeros(0, dtype=np.int64)
        self.kind: np.ndarray = np.zeros(0, dtype=np.int8)
        self.weight: np.ndarray = np.zeros(0, dtype=np.float64)
        self.delay_ticks: np.ndarray = np.zeros(0, dtype=np.int64)
        self.trainable: np.ndarray = np.zeros(0, dtype=bool)
        self.sum_order: np.ndarray = np.zeros(0, dtype=np.int64)
        self._exc_order: np.ndarray = np.zeros(0, dtype=np.int64)
        self._inh_order: np.ndarray = np.zeros(0, dtype=np.int64)
        self._delay_groups: list[tuple[int, np.ndarray]] = []

    # --- 구성 -------------------------------------------------------------
    def add(
        self,
        pre_id: int,
        post_id: int,
        kind: str,
        weight: float,
        delay_ticks: int,
        trainable: bool = False,
    ) -> int:
        """연결 1개를 추가하고 connection_id 를 돌려준다.

        검증 (명세 [12] T1): 음수 가중치, 범위 밖 뉴런 ID, 잘못된 연결 종류,
        0 이하 지연을 모두 ``ValueError`` 로 거부한다.
        """
        if self._finalized:
            raise RuntimeError("finalize() 이후에는 연결을 추가할 수 없다")
        if kind not in KIND_CODES:
            raise ValueError(f"잘못된 연결 종류: {kind!r} (excitatory/inhibitory 만 허용)")
        if not (0 <= int(pre_id) < self.n_neurons):
            raise ValueError(f"잘못된 pre_id: {pre_id}")
        if not (0 <= int(post_id) < self.n_neurons):
            raise ValueError(f"잘못된 post_id: {post_id}")
        w = float(weight)
        if not np.isfinite(w) or w < 0.0:
            raise ValueError(f"weight 는 유한한 비음수 크기여야 한다: {weight}")
        d = int(delay_ticks)
        if d < 1:
            raise ValueError(f"delay_ticks 는 1 이상의 정수여야 한다: {delay_ticks}")
        cid = len(self._rows)
        self._rows.append(
            {
                "connection_id": cid,
                "pre_id": int(pre_id),
                "post_id": int(post_id),
                "kind": KIND_CODES[kind],
                "weight": w,
                "delay_ticks": d,
                "trainable": bool(trainable),
            }
        )
        return cid

    def finalize(self) -> None:
        """배열 변환 + 안정적 합산 순열 계산. 이후 구조는 고정된다."""
        n = len(self._rows)
        self.connection_id = np.arange(n, dtype=np.int64)
        self.pre_id = np.array([r["pre_id"] for r in self._rows], dtype=np.int64)
        self.post_id = np.array([r["post_id"] for r in self._rows], dtype=np.int64)
        self.kind = np.array([r["kind"] for r in self._rows], dtype=np.int8)
        self.weight = np.array([r["weight"] for r in self._rows], dtype=np.float64)
        self.delay_ticks = np.array([r["delay_ticks"] for r in self._rows], dtype=np.int64)
        self.trainable = np.array([r["trainable"] for r in self._rows], dtype=bool)

        # 등록 순서 독립적인 합산 순열: (post_name, kind, pre_name, cid) 사전식
        keys = [
            (self.neuron_names[self.post_id[i]], int(self.kind[i]),
             self.neuron_names[self.pre_id[i]], int(i))
            for i in range(n)
        ]
        self.sum_order = np.array(
            sorted(range(n), key=lambda i: keys[i]), dtype=np.int64
        )
        so = self.sum_order
        self._exc_order = so[self.kind[so] == KIND_EXCITATORY]
        self._inh_order = so[self.kind[so] == KIND_INHIBITORY]

        # 지연별 연결 그룹 (전송 벡터화용). 순서는 connection_id 오름차순.
        self._delay_groups = []
        for d in sorted(set(int(x) for x in self.delay_ticks.tolist())):
            idx = np.nonzero(self.delay_ticks == d)[0]
            self._delay_groups.append((d, idx.astype(np.int64)))

        self._finalized = True

    # --- 조회 (부작용 없음) ------------------------------------------------
    def __len__(self) -> int:
        return len(self._rows)

    @property
    def n_connections(self) -> int:
        return len(self._rows)

    @property
    def exc_order(self) -> np.ndarray:
        return self._exc_order

    @property
    def inh_order(self) -> np.ndarray:
        return self._inh_order

    @property
    def delay_groups(self) -> list[tuple[int, np.ndarray]]:
        return self._delay_groups

    def sign(self) -> np.ndarray:
        """shape (K,) float64: excitatory=+1, inhibitory=-1."""
        return KIND_SIGN[self.kind]

    def kind_name(self, cid: int) -> str:
        return KIND_NAMES[int(self.kind[cid])]

    def incoming(self, post_id: int) -> np.ndarray:
        """해당 뉴런으로 들어오는 connection_id 배열 (정렬됨)."""
        return np.nonzero(self.post_id == int(post_id))[0]

    def outgoing(self, pre_id: int) -> np.ndarray:
        return np.nonzero(self.pre_id == int(pre_id))[0]

    def row(self, cid: int) -> dict[str, Any]:
        i = int(cid)
        return {
            "connection_id": i,
            "pre_id": int(self.pre_id[i]),
            "post_id": int(self.post_id[i]),
            "pre_name": self.neuron_names[self.pre_id[i]],
            "post_name": self.neuron_names[self.post_id[i]],
            "kind": KIND_NAMES[int(self.kind[i])],
            "weight": float(self.weight[i]),
            "delay_ticks": int(self.delay_ticks[i]),
            "trainable": bool(self.trainable[i]),
        }

    def validate_event_source(self, connection_id: int, source_id: int) -> None:
        """이벤트의 source_id 가 연결 목록의 pre_id 와 일치하는지 검증한다."""
        cid = int(connection_id)
        if not (0 <= cid < self.n_connections):
            raise ValueError(f"존재하지 않는 connection_id: {connection_id}")
        if int(self.pre_id[cid]) != int(source_id):
            raise ValueError(
                f"connection {cid} 의 pre_id={int(self.pre_id[cid])} 와 "
                f"이벤트 source_id={source_id} 가 다르다"
            )

    # --- 가중치 변경 (학습 전용) -------------------------------------------
    def set_weights(self, new_weight: np.ndarray) -> None:
        """가중치 전체 교체. weight_version 을 1 증가시킨다 (부작용 있음)."""
        w = np.asarray(new_weight, dtype=np.float64)
        if w.shape != self.weight.shape:
            raise ValueError("가중치 shape 불일치")
        if np.any(w < 0.0):
            raise ValueError("가중치는 비음수여야 한다 (clipping 후에도)")
        self.weight = w
        self.weight_version += 1

    def snapshot_weights(self) -> np.ndarray:
        """현재 가중치의 복사본 (부작용 없음)."""
        return self.weight.copy()

    def restore_weights(self, weights: np.ndarray, version: int | None = None) -> None:
        """과거 가중치 버전 복원 (진단용). version 을 주면 그대로 복원한다."""
        self.weight = np.asarray(weights, dtype=np.float64).copy()
        if version is not None:
            self.weight_version = int(version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_neurons": self.n_neurons,
            "neuron_names": self.neuron_names,
            "connection_id": self.connection_id.tolist(),
            "pre_id": self.pre_id.tolist(),
            "post_id": self.post_id.tolist(),
            "kind": self.kind.tolist(),
            "weight": self.weight.tolist(),
            "delay_ticks": self.delay_ticks.tolist(),
            "trainable": self.trainable.tolist(),
            "weight_version": self.weight_version,
        }

    @staticmethod
    def from_arrays(
        n_neurons: int,
        neuron_names: list[str],
        pre_id: Iterable[int],
        post_id: Iterable[int],
        kind: Iterable[int],
        weight: Iterable[float],
        delay_ticks: Iterable[int],
        trainable: Iterable[bool],
        weight_version: int = 0,
    ) -> "ConnectionTable":
        table = ConnectionTable(n_neurons, neuron_names)
        for p, q, k, w, d, t in zip(pre_id, post_id, kind, weight, delay_ticks, trainable):
            table.add(int(p), int(q), KIND_NAMES[int(k)], float(w), int(d), bool(t))
        table.finalize()
        table.weight_version = int(weight_version)
        return table
