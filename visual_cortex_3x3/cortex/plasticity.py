"""plasticity.py -- 국소 가소성(STDP)과 항상성 임계 적응.

명세 8절.

STDP 수식 (실제 구현)
---------------------
전/후 흔적은 스텝마다 지수 감쇠한다::

    x_pre[i]  <- x_pre[i]  * exp(-dt/tau_plus)
    x_post[j] <- x_post[j] * exp(-dt/tau_minus)

발화한 뉴런의 흔적은 1 만큼 증가한다 (all-to-all pair 방식).
시냅스 s (i -> j) 의 갱신은::

    j 가 발화하면 (LTP):  dw_s += A_plus  * x_pre[i]
    i 가 발화하면 (LTD):  dw_s -= A_minus * x_post[j]

동시 발화 처리 (``simultaneous_policy``)

* ``both``       : 같은 스텝의 전/후 발화 모두에 대해 위 두 항을 적용한다.
                   두 항 모두 **흔적 증가 이전** 값을 쓴다 (자기 자신의 스파이크가
                   자기 갱신에 들어가지 않는다).
* ``pre_first``  : 전 흔적을 먼저 증가시킨 뒤 LTP 를 계산한다 (동시 발화 시 LTP 우세).
* ``post_first`` : 후 흔적을 먼저 증가시킨 뒤 LTD 를 계산한다 (동시 발화 시 LTD 우세).

갱신 순서 ``update_order`` 는 LTP/LTD 중 어느 쪽을 먼저 누적할지 정한다.
가중치는 비음수 크기이므로 ``clip(w, weight_min>=0, weight_max)`` 로 잘리며
**흥분 연결이 억제 연결로 뒤집히지 않는다**.

이것은 "동시 발화 곱" 이 아니라 전/후 사건의 **시간 차**를 흔적으로 표현한
pair-based STDP 다.

항상성 임계 적응
----------------
활동률 추정은 지수 이동 평균이다 (window_ms 가 누적 시간)::

    r[i] <- r[i] + (dt/window_ms) * (spike[i]/(dt/1000) - r[i])      [Hz]
    theta[i] <- clip(theta[i] + eta_theta*(r[i]-target[i])*(dt/window_ms), lo, hi)

상하한은 엔진 모드에 따라 mV(``theta_min_mV``/``theta_max_mV``) 또는 무차원
(``theta_min_sum``/``theta_max_sum``) 을 쓴다. 목표 활동률은 세포 유형별로
다를 수 있다.

**이웃 임계값 평균화는 기본값이 아니다.** 소거(ablation) 실험 옵션으로만 두며,
이것을 "정답 교정"이라고 부르지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .units import MS_PER_S


@dataclass
class UpdateRecord:
    """한 스텝의 실제 변화량 (updates.h5 에 그대로 저장된다)."""

    step: int
    time_ms: float
    n_synapses_changed: int = 0
    ltp_sum: float = 0.0
    ltd_sum: float = 0.0
    decay_sum: float = 0.0
    clipped_low: int = 0
    clipped_high: int = 0
    weight_delta_sum: float = 0.0
    weight_delta_abs_sum: float = 0.0
    theta_delta_sum: float = 0.0
    theta_delta_abs_sum: float = 0.0
    n_theta_clipped: int = 0
    teacher_derived: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dict(vars(self))
        d.pop("extra")
        d.update(self.extra)
        return d


class StdpHomeostasis:
    """STDP + 항상성 임계 적응 모듈.

    Parameters
    ----------
    cfg : 해석된 설정
    table : :class:`cortex.synapses.SynapseTable`
    arrays : :class:`cortex.records.NeuronArrays`
    ids : :class:`cortex.ids.IdSpace`
    """

    def __init__(self, cfg: dict[str, Any], table: Any, arrays: Any, ids: Any) -> None:
        self.cfg = cfg
        self.table = table
        self.a = arrays
        self.ids = ids
        lr = cfg["learning"]
        self.task_enabled = bool(lr["task_learning_enabled"])
        self.decay_enabled = bool(lr["weight_decay_enabled"])
        self.theta_enabled = bool(lr["threshold_adaptation_enabled"])
        s = lr["stdp"]
        self.A_plus = float(s["A_plus"])
        self.A_minus = float(s["A_minus"])
        self.tau_plus = float(s["tau_plus_ms"])
        self.tau_minus = float(s["tau_minus_ms"])
        self.w_min = float(s["weight_min"])
        self.w_max = float(s["weight_max"])
        self.decay_per_ms = float(s["weight_decay_per_ms"])
        self.simultaneous_policy = s["simultaneous_policy"]
        self.update_order = s["update_order"]
        self.apply_to_inhibitory = bool(s["apply_to_inhibitory"])

        h = lr["homeostasis"]
        self.eta_theta = float(h["eta_theta"])
        self.window_ms = float(h["window_ms"])
        mode = cfg["engine"]["mode"]
        if mode == "conductance_lif":
            self.theta_lo = float(h["theta_min_mV"])
            self.theta_hi = float(h["theta_max_mV"])
        else:
            self.theta_lo = float(h["theta_min_sum"])
            self.theta_hi = float(h["theta_max_sum"])
        self.per_cell_type_target = bool(h["per_cell_type_target"])

        nb = lr["neighbor_theta_averaging"]
        self.neighbor_enabled = bool(nb["enabled"])
        self.neighbor_radius_mm = float(nb["radius_mm"])
        self.neighbor_strength = float(nb["strength"])
        self._neighbor_index: Any = None

        # 학습 대상 시냅스 마스크
        stdp_id = ids.plasticity_rules.id_of("stdp")
        self.plastic = table.plasticity_rule == stdp_id
        if not self.apply_to_inhibitory:
            self.plastic = self.plastic & (table.src_dale_sign > 0)
        self.plastic_idx = np.nonzero(self.plastic)[0]

        self.records: list[UpdateRecord] = []
        self.total_weight_delta = 0.0
        self.total_theta_delta = 0.0

    # ------------------------------------------------------------------
    def on_step(self, engine: Any, spiked_ids: np.ndarray, dt_ms: float) -> UpdateRecord:
        """7단계에서 호출된다. 흔적/가중치/임계값을 갱신한다 (부작용 있음)."""
        a = self.a
        rec = UpdateRecord(step=engine.step_index, time_ms=engine.time_ms)

        # 흔적 감쇠 (스텝 시작 상태 -> 현재)
        a.trace_pre *= np.exp(-dt_ms / self.tau_plus)
        a.trace_post *= np.exp(-dt_ms / self.tau_minus)

        pre_before = a.trace_pre.copy()
        post_before = a.trace_post.copy()
        if self.simultaneous_policy == "pre_first" and spiked_ids.size:
            pre_before = pre_before.copy()
            pre_before[spiked_ids] += 1.0
        if self.simultaneous_policy == "post_first" and spiked_ids.size:
            post_before = post_before.copy()
            post_before[spiked_ids] += 1.0

        if self.task_enabled and self.plastic_idx.size and spiked_ids.size:
            dw = np.zeros(self.table.weight.size, dtype=np.float64)
            steps = (("ltp", "ltd") if self.update_order == "post_then_pre"
                     else ("ltd", "ltp"))
            for which in steps:
                if which == "ltp":
                    syn = self._incoming_plastic(spiked_ids)
                    if syn.size:
                        contrib = self.A_plus * pre_before[self.table.src_id[syn]]
                        np.add.at(dw, syn, contrib)
                        rec.ltp_sum += float(contrib.sum())
                else:
                    syn = self._outgoing_plastic(spiked_ids)
                    if syn.size:
                        contrib = self.A_minus * post_before[self.table.dst_id[syn]]
                        np.add.at(dw, syn, -contrib)
                        rec.ltd_sum += float(contrib.sum())
            if self.decay_enabled and self.decay_per_ms > 0.0:
                d = self.decay_per_ms * dt_ms * self.table.weight
                d[~self.plastic] = 0.0
                dw -= d
                rec.decay_sum += float(d.sum())
            before = self.table.weight.copy()
            self.table.weight += dw
            lo = int(np.count_nonzero(self.table.weight < self.w_min))
            hi = int(np.count_nonzero(self.table.weight > self.w_max))
            np.clip(self.table.weight, self.w_min, self.w_max, out=self.table.weight)
            delta = self.table.weight - before
            rec.clipped_low, rec.clipped_high = lo, hi
            rec.n_synapses_changed = int(np.count_nonzero(delta))
            rec.weight_delta_sum = float(delta.sum())
            rec.weight_delta_abs_sum = float(np.abs(delta).sum())
            self.total_weight_delta += rec.weight_delta_abs_sum

        # 흔적 증가 (갱신 뒤에 반영)
        if spiked_ids.size:
            if self.simultaneous_policy != "pre_first":
                a.trace_pre[spiked_ids] += 1.0
            else:
                a.trace_pre[:] = pre_before
            if self.simultaneous_policy != "post_first":
                a.trace_post[spiked_ids] += 1.0
            else:
                a.trace_post[:] = post_before

        # 활동률 추정과 임계 적응 (교사 항과 독립인 항목)
        spike_ind = np.zeros(a.n, dtype=np.float64)
        if spiked_ids.size:
            spike_ind[spiked_ids] = 1.0
        inst_hz = spike_ind / (dt_ms / MS_PER_S)
        alpha = dt_ms / self.window_ms
        a.rate_estimate_hz += alpha * (inst_hz - a.rate_estimate_hz)

        if self.theta_enabled:
            target = a.target_rate_hz if self.per_cell_type_target else float(
                np.mean(a.target_rate_hz))
            before_t = a.threshold.copy()
            a.threshold += self.eta_theta * (a.rate_estimate_hz - target) * alpha
            n_clip = int(np.count_nonzero(
                (a.threshold < self.theta_lo) | (a.threshold > self.theta_hi)))
            np.clip(a.threshold, self.theta_lo, self.theta_hi, out=a.threshold)
            if self.neighbor_enabled and self.neighbor_strength > 0.0:
                self._apply_neighbor_averaging()
            dtheta = a.threshold - before_t
            rec.theta_delta_sum = float(dtheta.sum())
            rec.theta_delta_abs_sum = float(np.abs(dtheta).sum())
            rec.n_theta_clipped = n_clip
            self.total_theta_delta += rec.theta_delta_abs_sum

        rec.teacher_derived = False   # 이 모듈의 항은 모두 교사와 무관하다
        self.records.append(rec)
        return rec

    # ------------------------------------------------------------------
    def _incoming_plastic(self, dst_ids: np.ndarray) -> np.ndarray:
        t = self.table
        if dst_ids.size == 0:
            return np.zeros(0, dtype=np.int64)
        parts = [t.in_syn[t.in_ptr[j]:t.in_ptr[j + 1]] for j in dst_ids]
        syn = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        return syn[self.plastic[syn]] if syn.size else syn

    def _outgoing_plastic(self, src_ids: np.ndarray) -> np.ndarray:
        t = self.table
        if src_ids.size == 0:
            return np.zeros(0, dtype=np.int64)
        parts = [t.out_syn[t.out_ptr[i]:t.out_ptr[i + 1]] for i in src_ids]
        syn = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        return syn[self.plastic[syn]] if syn.size else syn

    def _apply_neighbor_averaging(self) -> None:
        """소거 실험 전용: 이웃 임계값을 일부 평균화한다.

        기본값은 꺼져 있다. 이것을 "정답 교정"이라고 부르지 않는다.
        """
        from scipy.spatial import cKDTree

        a = self.a
        if self._neighbor_index is None:
            tree = cKDTree(a.position_mm)
            self._neighbor_index = tree.query_ball_tree(tree, r=self.neighbor_radius_mm)
        means = np.array([
            float(a.threshold[idx].mean()) if idx else float(a.threshold[i])
            for i, idx in enumerate(self._neighbor_index)
        ])
        a.threshold += self.neighbor_strength * (means - a.threshold)

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "rule": "stdp_homeostasis",
            "n_plastic_synapses": int(self.plastic_idx.size),
            "task_learning_enabled": self.task_enabled,
            "weight_decay_enabled": self.decay_enabled,
            "threshold_adaptation_enabled": self.theta_enabled,
            "neighbor_theta_averaging_enabled": self.neighbor_enabled,
            "total_abs_weight_delta": self.total_weight_delta,
            "total_abs_theta_delta": self.total_theta_delta,
            "n_update_records": len(self.records),
            "simultaneous_policy": self.simultaneous_policy,
            "update_order": self.update_order,
            "apply_to_inhibitory": self.apply_to_inhibitory,
            "teacher_independent_terms_ko": (
                "STDP·감쇠·항상성 항은 교사 신호와 무관하게 동작한다. "
                "0교정 검사에서 이 항들의 변화는 0 이 아닐 수 있으며, "
                "교사 유래 항과 분리해 비교해야 한다."
            ),
        }


class NoPlasticity:
    """``learning.mode = none`` 용 고정 기준. 아무 것도 바꾸지 않는다."""

    def __init__(self) -> None:
        self.records: list[UpdateRecord] = []

    def on_step(self, engine: Any, spiked_ids: np.ndarray, dt_ms: float) -> UpdateRecord:
        """가중치·임계값·흔적을 **바꾸지 않는다**. 활동률 추정만 갱신한다.

        활동률은 보고용 진단값이며 임계 적응에 쓰이지 않는다.
        """
        a = engine.a
        spike_ind = np.zeros(a.n, dtype=np.float64)
        if spiked_ids.size:
            spike_ind[spiked_ids] = 1.0
        inst_hz = spike_ind / (dt_ms / MS_PER_S)
        alpha = dt_ms / max(float(engine.cfg["learning"]["homeostasis"]["window_ms"]), dt_ms)
        a.rate_estimate_hz += alpha * (inst_hz - a.rate_estimate_hz)
        return UpdateRecord(step=engine.step_index, time_ms=engine.time_ms)

    def summary(self) -> dict[str, Any]:
        return {"rule": "none", "note_ko": "가중치·임계값을 바꾸지 않는 고정 기준이다."}


def make(cfg: dict[str, Any], table: Any, arrays: Any, ids: Any) -> Any:
    """설정의 ``learning.mode`` 에 맞는 가소성 모듈을 만든다.

    ``rao_reference`` 는 **다른 엔진**이므로 여기서 만들지 않는다
    (:mod:`cortex.predictive_coding`). 스파이크 엔진에는 학습 없음을 준다.
    """
    mode = cfg["learning"]["mode"]
    if mode == "stdp_homeostasis":
        return StdpHomeostasis(cfg, table, arrays, ids)
    return NoPlasticity()


__all__ = ["UpdateRecord", "StdpHomeostasis", "NoPlasticity", "make"]
