"""dynamics.py -- 공통 시간 루프와 두 동작 모드.

명세 4, 9절. 한 스텝은 **항상** 다음 순서다.

1. 외부 자극과 예약된 지연 사건 수집
2. 도착 사건을 입력 로그에 기록
3. 새 사건을 **한 번만** 수용체/작업 버퍼에 반영
4. 시냅스·막전위·구획 상태를 dt 만큼 갱신
5. 발화와 불응기 판정, 필요 시 리셋
6. 출력 목록에 따라 도착 시각이 정해진 사건 예약
7. 가소성 흔적과 학습 규칙 갱신
8. 선택된 상태와 실제 가중치/임계 변화 기록

**숨은 순서 의존성 없음**: 한 스텝의 모든 발화는 같은 (스텝 시작 시점의)
가중치로 판정·예약되고, 학습 갱신은 7단계에서 한꺼번에 적용된다.

모드 A: ``sum_threshold``
-------------------------
무차원 합산 모델이다. 생물학적 막전위 모델이 **아니다**::

    s_j = sum(effective_contribution of arrivals for neuron j in this interval)
    q_j = 1 if s_j >= theta_j else 0

한 처리 구간의 입력을 **모두 모은 뒤** 판정하므로 입력 목록 순서가 결과를
바꾸지 않는다 (누적은 synapse_id 순으로 정렬해 수행한다).
경계 조건: 입력이 0 이어도 ``theta_j <= 0`` 이면 발화한다.

모드 B: ``conductance_lif``
---------------------------
구획별 전도도 기반 LIF. 적분은 **후향 오일러(반암시적)** 이며 구획 결합까지
포함한 3x3 선형계를 뉴런마다 닫힌 형태로 푼다. 무조건 안정이므로 발산을
가리기 위한 임의 전압 클리핑을 넣지 않았다. 실제 이산식은 equations.md 참조.

전도도는 도착 사건으로 **한 번만** 증가하고, 그 뒤 수용체별 시정수로 감쇠하며
효과는 이후 스텝에도 상태로 남는다. 새로 도착한 사건과 이전 입력 때문에 남아
있는 상태를 구분한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .anatomy import Anatomy
from .events import DelayQueue, EventLog
from .ids import COMPARTMENT_INDEX, IdCounter
from .records import NeuronArrays
from .synapses import SynapseTable
from .units import MG_BLOCK_MG_MM, MG_BLOCK_SCALE, MG_BLOCK_SLOPE, MS_PER_S


@dataclass
class StepReport:
    """한 스텝의 요약 (기록/진단용)."""

    step: int
    time_ms: float
    n_arrivals: int
    n_spikes: int
    n_scheduled: int
    mean_V_soma_mV: float = float("nan")
    max_abs_V_mV: float = float("nan")
    n_nonfinite: int = 0
    decided: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExternalDrive:
    """망막(또는 지정 뉴런)에 주는 외생 입력.

    Attributes
    ----------
    neuron_ids : (M,) int64
    current_pA : (M,) float64
        ``rate`` 모드에서 쓰는 지속 전류 (conductance 모드).
    sum_contribution : (M,) float64
        ``sum_threshold`` 모드에서 매 스텝 더할 무차원 기여.
    forced_spike_steps : dict[int, np.ndarray] | None
        ``poisson`` 모드에서 강제 발화시킬 (스텝 -> 뉴런 ID 배열).
    """

    neuron_ids: np.ndarray
    current_pA: np.ndarray | None = None
    sum_contribution: np.ndarray | None = None
    forced_spike_steps: dict[int, np.ndarray] | None = None
    n_multi_event_truncated: int = 0


class Engine:
    """공통 시간 루프 엔진.

    Parameters
    ----------
    cfg : 해석된 설정
    anat : :class:`cortex.anatomy.Anatomy`
    table : :class:`cortex.synapses.SynapseTable`
    event_log : :class:`cortex.events.EventLog`
    plasticity : 객체 | None
        ``on_step(engine, spiked, dt_ms)`` 와 ``on_spikes(...)`` 를 제공하는
        학습 모듈 (:mod:`cortex.plasticity`). None 이면 학습 없음.
    recorder : 객체 | None
        ``record_step(...)`` 을 제공하는 기록기. 계산을 바꾸지 않는다.
    """

    def __init__(self, cfg: dict[str, Any], anat: Anatomy, table: SynapseTable,
                 event_log: EventLog, plasticity: Any = None,
                 recorder: Any = None) -> None:
        self.cfg = cfg
        self.anat = anat
        self.pop = anat.population
        self.a: NeuronArrays = anat.population.arrays
        self.table = table
        self.log = event_log
        self.plasticity = plasticity
        self.recorder = recorder

        eng = cfg["engine"]
        self.mode = eng["mode"]
        self.dt_ms = float(eng["dt_ms"])
        self.weight_application = eng["weight_application"]
        self.interval_steps = int(eng["sum_threshold_interval_steps"])

        self.queue = DelayQueue()
        self.step_index = 0
        self.time_ms = 0.0
        self.sample_id = 0
        self.episode_id = 0
        self.event_counter = IdCounter(0)
        self.spike_counter = IdCounter(0)

        # 수용체 상수 (인덱스 = receptor_id)
        r_names = anat.ids.receptors.names()
        self.receptor_names = r_names
        self.tau_r = np.array([cfg["receptors"][n]["tau_ms"] for n in r_names], np.float64)
        self.E_rev = np.array([cfg["receptors"][n]["E_rev_mV"] for n in r_names], np.float64)
        self.mg_block = np.array([bool(cfg["receptors"][n]["mg_block"]) for n in r_names])
        self.decay = np.exp(-self.dt_ms / self.tau_r)
        # 스텝 동안의 시간 평균 계수: (tau/dt)(1-exp(-dt/tau))
        self.mean_factor = (self.tau_r / self.dt_ms) * (1.0 - self.decay)

        self.drive: ExternalDrive | None = None
        # 망막 구동 전류는 사용자가 직접 설정한 Iext_pA 와 **더해진다**.
        # 엔진이 Iext_pA 를 매 스텝 0 으로 덮어쓰지 않는다.
        self._drive_I_pA = np.zeros((self.a.n, 3), dtype=np.float64)
        self.n_nonfinite_total = 0
        self.step_reports: list[StepReport] = []
        self._interval_counter = 0

    # ------------------------------------------------------------------
    # 상태 초기화
    # ------------------------------------------------------------------
    def reset_transient(self, *, voltages: bool = True, conductances: bool = True,
                        event_queue: bool = True, traces: bool = False,
                        thresholds: bool = False) -> None:
        """표본 사이 초기화. **가중치는 절대 지우지 않는다.**

        무엇을 리셋하고 무엇을 유지할지는 ``engine.reset_between_samples`` 설정이
        결정하며, 독립 영상 실험과 연속 시퀀스 실험을 구분한다.
        """
        a = self.a
        if voltages:
            a.V_mV[:, :] = a.EL_mV
            a.refractory_until_ms[:] = -np.inf
            a.fired[:] = 0
        if conductances:
            a.g_nS[:, :, :] = 0.0
        if event_queue:
            self.queue.clear()
        if traces:
            a.trace_pre[:] = 0.0
            a.trace_post[:] = 0.0
            a.rate_estimate_hz[:] = 0.0
        if thresholds:
            raise ValueError(
                "임계값 초기화는 학습 결과를 지우는 동작이다. reset_parameters() 를 "
                "명시적으로 호출하라."
            )
        a.interval_activation[:] = 0.0
        self._interval_counter = 0
        self.step_index = 0
        self.time_ms = 0.0

    def set_external_drive(self, drive: ExternalDrive | None) -> None:
        self.drive = drive

    # ------------------------------------------------------------------
    # 한 스텝
    # ------------------------------------------------------------------
    def step(self) -> StepReport:
        """명세 9절의 8단계를 순서대로 수행한다."""
        a = self.a
        t = self.time_ms
        n = self.step_index

        # --- 1. 예약 사건과 외부 자극 수집 -----------------------------
        payload = self.queue.pop(n)
        n_arrivals = 0
        if payload:
            order = np.argsort(payload["synapse_id"], kind="stable")
            payload = {k: v[order] for k, v in payload.items()}
            n_arrivals = int(payload["synapse_id"].size)

        # --- 2. 도착 사건을 입력 로그에 기록 ---------------------------
        if n_arrivals:
            syn = payload["synapse_id"]
            dst = self.table.dst_id[syn].astype(np.int64)
            comp = self.table.target_compartment[syn]
            if self.weight_application == "arrival":
                w_used = self.table.weight[syn]
                amount = w_used * payload["gain_snapshot"]
            else:
                w_used = payload["weight_snapshot"]
                amount = payload["amount"]
            ev_start = self.event_counter.take(n_arrivals)
            ev_ids = np.arange(ev_start, ev_start + n_arrivals, dtype=np.int64)
            self.log.append(
                event_id=ev_ids, parent_spike_id=payload["spike_id"],
                sample_id=self.sample_id, episode_id=self.episode_id,
                src_id=self.table.src_id[syn], dst_id=dst, synapse_id=syn,
                emit_time_ms=payload["emit_time_ms"],
                arrival_time_ms=np.full(n_arrivals, t),
                arrival_step=np.full(n_arrivals, n, dtype=np.int64),
                weight_snapshot=w_used, source_gain_snapshot=payload["gain_snapshot"],
                target_compartment=comp, event_type="synaptic_arrival",
                amount=amount,
                amount_unit="nS" if self.mode == "conductance_lif" else "dimensionless",
            )
            a.last_consumed_event[dst] = np.maximum(
                a.last_consumed_event[dst], ev_ids[-1])

        # --- 3. 새 사건을 한 번만 반영 ----------------------------------
        if n_arrivals:
            self._apply_arrivals(payload, amount, t)

        if self.mode == "conductance_lif":
            rep = self._step_conductance(n, t)
        else:
            rep = self._step_sum_threshold(n, t)
        rep.n_arrivals = n_arrivals
        rep.n_scheduled = self.queue.pending_count()

        # --- 8. 기록 ----------------------------------------------------
        if self.recorder is not None:
            self.recorder.record_step(self, rep)
        self.step_reports.append(rep)

        self.step_index += 1
        self.time_ms += self.dt_ms
        return rep

    # ------------------------------------------------------------------
    def _apply_arrivals(self, payload: dict[str, np.ndarray],
                        amount: np.ndarray, t: float) -> None:
        """도착값을 작업 버퍼(전도도 또는 구간 누적)에 **한 번만** 더한다."""
        a = self.a
        syn = payload["synapse_id"]
        dst = self.table.dst_id[syn].astype(np.int64)
        if self.mode == "conductance_lif":
            comp = self.table.target_compartment[syn].astype(np.int64)
            rec = self.table.receptor_type[syn].astype(np.int64)
            amt = np.maximum(amount, 0.0)  # 억제성 Δg 를 음수로 만들지 않는다
            # 불응기 동안 soma 입력을 버리는 세포 유형 처리
            drop = (a.refractory_discards_input[dst] & (comp == 0)
                    & (t < a.refractory_until_ms[dst]))
            if drop.any():
                keep = ~drop
                dst, comp, rec, amt = dst[keep], comp[keep], rec[keep], amt[keep]
            if dst.size:
                np.add.at(a.g_nS, (dst, comp, rec), amt)
        else:
            sign = self.table.src_dale_sign[syn].astype(np.float64)
            np.add.at(a.interval_activation, dst, amount * sign)

    # ------------------------------------------------------------------
    def _step_conductance(self, n: int, t: float) -> StepReport:
        a = self.a
        dt = self.dt_ms

        # 망막 구동 전류 (rate 모드). 사용자가 설정한 a.Iext_pA 는 유지된다.
        self._drive_I_pA[:, :] = 0.0
        if self.drive is not None and self.drive.current_pA is not None:
            self._drive_I_pA[self.drive.neuron_ids, 0] = self.drive.current_pA
        I_total = a.Iext_pA + self._drive_I_pA

        # --- 4. 시냅스/막전위/구획 상태 갱신 ---------------------------
        g0 = a.g_nS                                   # (N,3,R) 도착 반영 직후
        g_bar = g0 * self.mean_factor[None, None, :]  # 스텝 동안의 시간 평균

        V_pre = a.V_mV                                # (N,3)
        block = np.ones_like(g_bar)
        if self.mg_block.any():
            b = 1.0 / (1.0 + np.exp(-MG_BLOCK_SLOPE * V_pre) * MG_BLOCK_MG_MM / MG_BLOCK_SCALE)
            block[:, :, self.mg_block] = b[:, :, None]
        g_eff = g_bar * block                          # (N,3,R)
        g_sum = g_eff.sum(axis=2)                      # (N,3)
        g_E = (g_eff * self.E_rev[None, None, :]).sum(axis=2)

        g_sb = a.g_couple_nS[:, COMPARTMENT_INDEX["basal"]]
        g_sa = a.g_couple_nS[:, COMPARTMENT_INDEX["apical"]]
        c_dt = a.C_pF / dt                             # (N,3)

        diag = c_dt + a.gL_nS + g_sum
        a_s = diag[:, 0] + g_sb + g_sa
        a_b = diag[:, 1] + g_sb
        a_a = diag[:, 2] + g_sa

        rhs = c_dt * V_pre + a.gL_nS * a.EL_mV + g_E + I_total
        denom = a_s - (g_sb * g_sb) / a_b - (g_sa * g_sa) / a_a
        num = rhs[:, 0] + g_sb * rhs[:, 1] / a_b + g_sa * rhs[:, 2] / a_a
        V_s = num / denom
        V_b = (rhs[:, 1] + g_sb * V_s) / a_b
        V_a = (rhs[:, 2] + g_sa * V_s) / a_a

        V_new = np.stack([V_s, V_b, V_a], axis=1)
        absent = ~a.has_compartment
        V_new[absent] = a.EL_mV[absent]

        # 전도도 감쇠 (도착 효과는 상태로 남는다)
        a.g_nS *= self.decay[None, None, :]

        # --- 5. 발화/불응기 판정 ---------------------------------------
        refractory = t < a.refractory_until_ms
        V_new[refractory, 0] = a.V_reset_mV[refractory]
        a.V_mV[:, :] = V_new
        a.last_decision_value[:] = a.V_mV[:, 0]
        a.last_decision_threshold[:] = a.threshold

        spiked = (a.V_mV[:, 0] >= a.threshold) & (~refractory)
        if self.drive is not None and self.drive.forced_spike_steps is not None:
            forced = self.drive.forced_spike_steps.get(n)
            if forced is not None and forced.size:
                spiked[forced] = True
        spiked_ids = np.nonzero(spiked)[0]
        if spiked_ids.size:
            a.V_mV[spiked_ids, 0] = a.V_reset_mV[spiked_ids]
            a.refractory_until_ms[spiked_ids] = t + a.t_ref_ms[spiked_ids]
            a.last_spike_ms[spiked_ids] = t
            a.spike_count[spiked_ids] += 1
        a.fired[:] = 0
        a.fired[spiked_ids] = 1

        nonfinite = int(np.count_nonzero(~np.isfinite(a.V_mV)))
        self.n_nonfinite_total += nonfinite

        # --- 6. 출력 사건 예약 -----------------------------------------
        self._emit(spiked_ids, n, t)
        # --- 7. 가소성 --------------------------------------------------
        if self.plasticity is not None:
            self.plasticity.on_step(self, spiked_ids, dt)

        return StepReport(
            step=n, time_ms=t, n_arrivals=0, n_spikes=int(spiked_ids.size),
            n_scheduled=0,
            mean_V_soma_mV=float(np.nanmean(a.V_mV[:, 0])) if a.n else float("nan"),
            max_abs_V_mV=float(np.nanmax(np.abs(a.V_mV))) if a.n else float("nan"),
            n_nonfinite=nonfinite,
        )

    # ------------------------------------------------------------------
    def _step_sum_threshold(self, n: int, t: float) -> StepReport:
        a = self.a
        if self.drive is not None and self.drive.sum_contribution is not None:
            np.add.at(a.interval_activation, self.drive.neuron_ids,
                      self.drive.sum_contribution)

        self._interval_counter += 1
        if self._interval_counter < self.interval_steps:
            return StepReport(step=n, time_ms=t, n_arrivals=0, n_spikes=0,
                              n_scheduled=0, decided=False)
        self._interval_counter = 0

        a.last_decision_value[:] = a.interval_activation
        a.last_decision_threshold[:] = a.threshold
        spiked = a.interval_activation >= a.threshold
        if self.drive is not None and self.drive.forced_spike_steps is not None:
            forced = self.drive.forced_spike_steps.get(n)
            if forced is not None and forced.size:
                spiked[forced] = True
        spiked_ids = np.nonzero(spiked)[0]
        a.fired[:] = 0
        a.fired[spiked_ids] = 1
        if spiked_ids.size:
            a.last_spike_ms[spiked_ids] = t
            a.spike_count[spiked_ids] += 1
        # 작업 버퍼만 비운다. 장기 입력 로그는 그대로 남는다.
        a.interval_activation[:] = 0.0

        self._emit(spiked_ids, n, t)
        if self.plasticity is not None:
            self.plasticity.on_step(self, spiked_ids, self.dt_ms)
        return StepReport(step=n, time_ms=t, n_arrivals=0,
                          n_spikes=int(spiked_ids.size), n_scheduled=0)

    # ------------------------------------------------------------------
    def _emit(self, spiked_ids: np.ndarray, n: int, t: float) -> None:
        """발화 뉴런의 출력 사건을 지연에 따라 예약한다 (6단계).

        가중치/이득은 설정 ``engine.weight_application`` 에 따라 발신 시점
        스냅샷(기본) 또는 도착 시점 값을 쓴다. 실제 적용한 값을 이벤트에 남긴다.
        """
        if spiked_ids.size == 0:
            return
        a = self.a
        sp_start = self.spike_counter.take(int(spiked_ids.size))
        spike_ids = np.arange(sp_start, sp_start + spiked_ids.size, dtype=np.int64)

        ev_start = self.event_counter.take(int(spiked_ids.size))
        self.log.append(
            event_id=np.arange(ev_start, ev_start + spiked_ids.size, dtype=np.int64),
            parent_spike_id=spike_ids, sample_id=self.sample_id,
            episode_id=self.episode_id, src_id=spiked_ids,
            dst_id=np.full(spiked_ids.size, -1, dtype=np.int64),
            synapse_id=np.full(spiked_ids.size, -1, dtype=np.int64),
            emit_time_ms=np.full(spiked_ids.size, t),
            arrival_time_ms=np.full(spiked_ids.size, t),
            arrival_step=np.full(spiked_ids.size, n, dtype=np.int64),
            weight_snapshot=np.zeros(spiked_ids.size),
            source_gain_snapshot=a.output_gain_P[spiked_ids],
            target_compartment=np.full(spiked_ids.size, -1, dtype=np.int8),
            event_type="spike", amount=np.ones(spiked_ids.size), amount_unit="spike",
        )

        spike_of = np.zeros(a.n, dtype=np.int64)
        spike_of[spiked_ids] = spike_ids
        out_ptr, out_syn = self.table.out_ptr, self.table.out_syn
        counts = out_ptr[spiked_ids + 1] - out_ptr[spiked_ids]
        total = int(counts.sum())
        if total == 0:
            return
        syn = np.concatenate([out_syn[out_ptr[i]:out_ptr[i + 1]] for i in spiked_ids])
        src = self.table.src_id[syn].astype(np.int64)
        active = self.table.active[syn]
        if not active.all():
            syn, src = syn[active], src[active]
        if syn.size == 0:
            return
        gain = a.output_gain_P[src]
        w = self.table.weight[syn]
        amount = w * gain
        steps = self.table.effective_delay_steps[syn].astype(np.int64)
        arrival = n + steps
        emit_time = np.full(syn.size, t)
        sp = spike_of[src]

        for st in np.unique(arrival):
            m = arrival == st
            self.queue.schedule(int(st), {
                "synapse_id": syn[m],
                "amount": amount[m],
                "weight_snapshot": w[m],
                "gain_snapshot": gain[m],
                "emit_time_ms": emit_time[m],
                "spike_id": sp[m],
            })

    # ------------------------------------------------------------------
    def run(self, n_steps: int, on_step: Callable[[StepReport], None] | None = None
            ) -> list[StepReport]:
        """n_steps 만큼 실행한다. 진행 콜백은 계산을 바꾸지 않아야 한다."""
        out: list[StepReport] = []
        for _ in range(int(n_steps)):
            rep = self.step()
            out.append(rep)
            if on_step is not None:
                on_step(rep)
        return out

    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """체크포인트용 상태 (가중치·전도도·흔적·이벤트 큐·시간)."""
        a = self.a
        return {
            "V_mV": a.V_mV.copy(), "g_nS": a.g_nS.copy(),
            "threshold": a.threshold.copy(),
            "refractory_until_ms": a.refractory_until_ms.copy(),
            "last_spike_ms": a.last_spike_ms.copy(),
            "spike_count": a.spike_count.copy(),
            "rate_estimate_hz": a.rate_estimate_hz.copy(),
            "trace_pre": a.trace_pre.copy(), "trace_post": a.trace_post.copy(),
            "interval_activation": a.interval_activation.copy(),
            "last_consumed_event": a.last_consumed_event.copy(),
            "weight": self.table.weight.copy(),
            "queue": self.queue.state_dict(),
            "step_index": self.step_index, "time_ms": self.time_ms,
            "sample_id": self.sample_id, "episode_id": self.episode_id,
            "event_counter": self.event_counter.value,
            "spike_counter": self.spike_counter.value,
            "interval_counter": self._interval_counter,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        a = self.a
        for k in ("V_mV", "g_nS", "threshold", "refractory_until_ms", "last_spike_ms",
                  "spike_count", "rate_estimate_hz", "trace_pre", "trace_post",
                  "interval_activation", "last_consumed_event"):
            getattr(a, k)[...] = np.asarray(d[k])
        self.table.weight[...] = np.asarray(d["weight"])
        self.queue.load_state_dict(d["queue"])
        self.step_index = int(d["step_index"])
        self.time_ms = float(d["time_ms"])
        self.sample_id = int(d["sample_id"])
        self.episode_id = int(d["episode_id"])
        self.event_counter.restore(int(d["event_counter"]))
        self.spike_counter.restore(int(d["spike_counter"]))
        self._interval_counter = int(d["interval_counter"])

    def read_only_snapshot(self) -> dict[str, np.ndarray]:
        """측정 함수용 깊은 복사. 모델/RNG 를 바꾸지 않는다 (검증 10번)."""
        a = self.a
        return {
            "V_mV": a.V_mV.copy(), "g_nS": a.g_nS.copy(),
            "threshold": a.threshold.copy(), "weight": self.table.weight.copy(),
            "trace_pre": a.trace_pre.copy(), "trace_post": a.trace_post.copy(),
            "last_consumed_event": a.last_consumed_event.copy(),
            "rate_estimate_hz": a.rate_estimate_hz.copy(),
        }


def settle_chain(engine: Engine, *, n_steps: int, drive: ExternalDrive | None,
                 initial_state: dict[str, Any] | None = None,
                 weight_snapshot: np.ndarray | None = None,
                 freeze_learning: bool = False) -> dict[str, Any]:
    """명시적 입력·초기 상태·가중치 스냅샷·길이로 회로를 정착시킨다.

    명세 9절: 자유/유도 비교에서 초기 막전위·전도도·흔적·대기 이벤트·난수 상태·
    시간 길이·회로 마스크를 **일치시키고 교사/문맥 항만** 바꾸기 위한 진입점이다.
    두 조건 사이에서 학습 업데이트를 먼저 적용하지 않는다 (``freeze_learning``).

    Returns
    -------
    dict : 스파이크 회로의 **안정화 지표** (연속값 Rao 의 에너지 수렴 지표와 다르다)
    """
    if initial_state is not None:
        engine.load_state_dict(initial_state)
    if weight_snapshot is not None:
        engine.table.weight[...] = np.asarray(weight_snapshot)
    saved = engine.plasticity
    if freeze_learning:
        engine.plasticity = None
    engine.set_external_drive(drive)
    try:
        reports = engine.run(int(n_steps))
    finally:
        engine.plasticity = saved

    spikes = np.array([r.n_spikes for r in reports], dtype=np.float64)
    half = max(1, len(spikes) // 2)
    return {
        "n_steps": int(n_steps),
        "total_spikes": int(spikes.sum()),
        "spikes_first_half": float(spikes[:half].mean()) if spikes.size else 0.0,
        "spikes_second_half": float(spikes[half:].mean()) if spikes.size else 0.0,
        "spike_rate_drift": (float(spikes[half:].mean() - spikes[:half].mean())
                             if spikes.size else 0.0),
        "max_abs_V_mV": float(np.nanmax([r.max_abs_V_mV for r in reports]))
        if reports else float("nan"),
        "n_nonfinite": int(sum(r.n_nonfinite for r in reports)),
        "pending_events": engine.queue.pending_count(),
        "metric_kind": "spiking_circuit_settling",
        "note_ko": ("스파이크 회로의 안정화 지표다. Rao 연속값 모델의 에너지 수렴 "
                    "지표와 같은 양이 아니므로 섞어 비교하지 않는다."),
    }


def rates_to_drive(cfg: dict[str, Any], neuron_ids: np.ndarray,
                   rates_hz: np.ndarray, n_steps: int,
                   rng: np.random.Generator) -> ExternalDrive:
    """발화율 -> :class:`ExternalDrive`.

    ``rate`` 모드는 지속 전류(또는 무차원 기여)로, ``poisson`` 모드는 미리 뽑아
    둔 강제 발화 스텝으로 만든다. 미리 뽑아 두면 대조군에서 **동일한 외생
    사건을 재생**할 수 있다.
    """
    d = cfg["retina"]["drive"]
    ids = np.asarray(neuron_ids, dtype=np.int64)
    r = np.asarray(rates_hz, dtype=np.float64)
    if d["mode"] == "rate":
        if cfg["engine"]["mode"] == "conductance_lif":
            return ExternalDrive(ids, current_pA=r * float(d["current_per_hz_pA"]))
        contrib = r * float(cfg["engine"]["dt_ms"]) / MS_PER_S * float(d["sum_mode_scale"])
        return ExternalDrive(ids, sum_contribution=contrib)

    lam = r * float(cfg["engine"]["dt_ms"]) / MS_PER_S
    forced: dict[int, np.ndarray] = {}
    truncated = 0
    for s in range(int(n_steps)):
        c = rng.poisson(lam)
        nz = np.nonzero(c > 0)[0]
        if nz.size:
            forced[s] = ids[nz]
            truncated += int((c[nz] - 1).sum())
    return ExternalDrive(ids, forced_spike_steps=forced,
                         n_multi_event_truncated=truncated)


__all__ = ["StepReport", "ExternalDrive", "Engine", "settle_chain", "rates_to_drive"]
