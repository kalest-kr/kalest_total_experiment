"""local_learning.py -- 이번에 채택한 국소 학습 기준 규칙.

명세 [9]. **제한된 퍼셉트론형 기준 규칙**이며 생화학적 학습 법칙이나
손실 하강이 보장된 심층 알고리즘이 아니다.

영상의 자유 실행을 완료한 뒤, L2 뉴런 i 의 관찰 시점에 대해::

    d_i        = target_i - q_i_free
    e_k        = s_k_free                       # 그 시점 그 연결의 도착값 합
    sign_k     = +1 (excitatory) / -1 (inhibitory)
    Z_i        = epsilon + sum(e_k**2, trainable inputs to i)
    G_k        = learning_rate * d_i * sign_k * e_k / Z_i
    weight_new = clip(weight_old + G_k, 0, weight_max)

구현 조건 (명세 [9] 주의사항)
-----------------------------
1. ``e`` 는 이번 관찰 시점 도착값이다. postsynaptic q 를 다시 곱하지 않는다.
   발화하지 못한 뉴런도 학습한다.
2. ``e`` 는 STDP 나 장기 누적 흔적이 아니다. 이전 영상 값을 누적하지 않는다.
3. ``d`` 는 이진 활동 차이다. 로짓 오차나 손실 기울기가 아니다.
4. 모든 G 를 같은 자유 실행 스냅샷에서 계산한 뒤 한꺼번에 적용한다.
5. 한 영상의 순방향 계산 도중에는 가중치를 바꾸지 않는다.
6. threshold/P/b_base 는 학습하지 않고 가중치 감쇠도 쓰지 않는다.
7. 도착 입력이 전부 0 이면 갱신은 0 이다 (임계값/기저 입력을 자동 조정하지 않는다).
8. Z 는 해당 수신 뉴런의 지역 입력만 사용한다.
9. 원래 G 와 clipping 후 실제 변화량을 모두 집계한다.
10. 국소 함수는 해당 뉴런의 d, 도착 입력, 연결 속성, 설정만 받는다.
    전체 라벨·다른 뉴런의 오차·감사 로그에 접근하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .connections import ConnectionTable


@dataclass
class FreeRunSnapshot:
    """자유 실행에서 학습에 필요한 값만 담은 스냅샷 (부작용 없음).

    Attributes
    ----------
    arrivals : (K,) float64
        L2 관찰 시점의 연결별 도착값 ``s_k``.
    q : (N,) int8
        각 뉴런의 관찰 시점 발화.
    u : (N,) float64
    weight_version : int
    """

    arrivals: np.ndarray
    q: np.ndarray
    u: np.ndarray
    weight_version: int

    @staticmethod
    def from_trial(result, observation_tick: int) -> "FreeRunSnapshot":
        return FreeRunSnapshot(
            arrivals=result.arrivals_at(observation_tick),
            q=result.observed_q,
            u=result.observed_u,
            weight_version=result.weight_version,
        )


@dataclass
class UpdateBatch:
    """계산된 갱신량과 집계 통계."""

    raw_G: np.ndarray            # (K,) 원래 G (clipping 전)
    applied_delta: np.ndarray    # (K,) 실제 적용된 변화량 (clipping 후)
    new_weight: np.ndarray       # (K,)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningConfig:
    epsilon: float = 1e-8
    learning_rate: float = 0.05
    weight_max: float = 1.0


def compute_local_update_for_neuron(
    d_i: float,
    e: np.ndarray,
    sign: np.ndarray,
    trainable: np.ndarray,
    cfg: LearningConfig,
) -> np.ndarray:
    """**순수 국소 함수**: 뉴런 1개의 갱신량 G 를 계산한다.

    인자는 해당 뉴런의 오차 ``d_i``, 그 뉴런에 도착한 입력 ``e``,
    그 연결들의 속성(``sign``, ``trainable``), 설정뿐이다.
    다른 뉴런의 오차·전체 라벨·감사 로그에 접근하지 않는다.

    Parameters
    ----------
    d_i : float
        target - q_free (이진 활동 차이).
    e : (m,) float64
        그 뉴런의 입력 연결별 도착값.
    sign : (m,) float64
        +1 흥분 / -1 억제.
    trainable : (m,) bool
    cfg : LearningConfig

    Returns
    -------
    (m,) float64 : clipping **전**의 G. 부작용 없음.
    """
    e = np.asarray(e, dtype=np.float64)
    tr = np.asarray(trainable, dtype=bool)
    Z = cfg.epsilon + float(np.sum((e[tr]) ** 2)) if tr.any() else cfg.epsilon
    G = cfg.learning_rate * float(d_i) * np.asarray(sign, dtype=np.float64) * e / Z
    return np.where(tr, G, 0.0)


class LocalLearner:
    """국소 규칙 배치 구현 (수학적으로 뉴런별 순수 함수와 동일).

    Parameters
    ----------
    table : ConnectionTable
    cfg : LearningConfig
    """

    def __init__(self, table: ConnectionTable, cfg: LearningConfig) -> None:
        self.table = table
        self.cfg = cfg
        self._sign = table.sign()
        self._post = table.post_id
        self._trainable = table.trainable
        self._n = table.n_neurons
        self.total_clipped = 0
        self.total_nonzero_G = 0

    def compute_updates(
        self, free_snapshot: FreeRunSnapshot, local_errors: dict[int, float] | tuple
    ) -> UpdateBatch:
        """자유 실행 스냅샷 + 뉴런별 오차 -> 갱신 배치. **가중치를 바꾸지 않는다.**

        Parameters
        ----------
        free_snapshot : FreeRunSnapshot
        local_errors : {neuron_id: d} 또는 (ids, d_values)

        Returns
        -------
        UpdateBatch
        """
        e = np.asarray(free_snapshot.arrivals, dtype=np.float64)
        if e.shape[0] != self.table.n_connections:
            raise ValueError("arrivals 길이가 연결 수와 다르다")

        d_vec = np.zeros(self._n, dtype=np.float64)
        if isinstance(local_errors, tuple):
            ids, vals = local_errors
            d_vec[np.asarray(ids, dtype=np.int64)] = np.asarray(vals, dtype=np.float64)
        else:
            for nid, dv in local_errors.items():
                d_vec[int(nid)] = float(dv)

        tr = self._trainable
        e_tr = np.where(tr, e, 0.0)
        # Z_i: 해당 수신 뉴런의 학습 대상 지역 입력만 사용
        Z = self.cfg.epsilon + np.bincount(
            self._post, weights=e_tr * e_tr, minlength=self._n
        )
        G = self.cfg.learning_rate * d_vec[self._post] * self._sign * e_tr / Z[self._post]
        G = np.where(tr, G, 0.0)

        w_old = self.table.weight
        w_new = np.clip(w_old + G, 0.0, self.cfg.weight_max)
        applied = w_new - w_old

        nonzero = np.count_nonzero(G)
        clipped = int(np.count_nonzero((G != 0.0) & (np.abs(applied - G) > 1e-15)))
        stats = {
            "n_nonzero_G": int(nonzero),
            "n_clipped": clipped,
            "clipping_fraction": (clipped / nonzero) if nonzero else 0.0,
            "sum_abs_raw_G": float(np.abs(G).sum()),
            "sum_abs_applied": float(np.abs(applied).sum()),
            "max_abs_raw_G": float(np.abs(G).max()) if G.size else 0.0,
            "n_active_l2_errors": int(np.count_nonzero(d_vec)),
        }
        return UpdateBatch(raw_G=G, applied_delta=applied, new_weight=w_new, stats=stats)

    def apply_updates(self, updates: UpdateBatch) -> dict[str, Any]:
        """갱신을 한꺼번에 적용한다 (**부작용**: 가중치와 weight_version 변경).

        갱신량이 모두 정확히 0 이면 가중치를 비트 단위로 보존하기 위해
        ``set_weights`` 를 호출하지 않는다 (감쇠가 없으므로 지속 변화도 없다).
        """
        if not np.any(updates.applied_delta):
            return {"changed": False, **updates.stats}
        self.table.set_weights(updates.new_weight)
        self.total_clipped += updates.stats["n_clipped"]
        self.total_nonzero_G += updates.stats["n_nonzero_G"]
        return {"changed": True, **updates.stats}

    # --- 참조 구현 (검사용) -----------------------------------------------
    def compute_updates_reference(
        self, free_snapshot: FreeRunSnapshot, local_errors: dict[int, float]
    ) -> np.ndarray:
        """뉴런별 순수 국소 함수만 사용하는 느린 참조 구현 (검증용)."""
        e = np.asarray(free_snapshot.arrivals, dtype=np.float64)
        G = np.zeros_like(e)
        for nid, d_i in local_errors.items():
            idx = self.table.incoming(int(nid))
            if idx.size == 0:
                continue
            G[idx] = compute_local_update_for_neuron(
                float(d_i), e[idx], self._sign[idx], self._trainable[idx], self.cfg
            )
        return G
