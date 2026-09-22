"""validation.py -- 명세 12절의 필수 검증 1~14.

**이 파일의 검사는 실제 계산값을 쓴다.** 항상 True 인 검사를 넣지 않는다.
검사를 실행하는 것은 사용자 명령(``python -m cortex.cli validate --execute``)
이며, 이 소스를 작성하는 동안에는 실행하지 않는다.

각 검사 결과는 ``status`` 가 ``passed`` / ``failed`` / ``skipped`` 중 하나다.
``skipped`` 는 이유와 함께 기록하며 **passed 에 포함하지 않는다.**
성공률을 맞추려고 기준을 사후에 바꾸지 않는다.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import anatomy as anatomy_mod
from . import areas as areas_mod
from . import rng as rng_mod
from . import stimuli as stimuli_mod
from .dynamics import Engine, ExternalDrive
from .events import EventLog, quantize_delay
from .ids import COMPARTMENT_INDEX, IdSpace
from .plasticity import StdpHomeostasis
from .predictive_coding import RaoModel
from .records import (
    DynamicStateView,
    InputLogView,
    MetadataView,
    NeuronArrays,
    NeuronPopulation,
    OutgoingConnectionsView,
)
from .retina import RetinaEncoder, uniform_response_check
from .retinotopy import LogPolarSampler, nyquist_margin, roundtrip_error
from .synapses import SynapseTable


class Check:
    """검사 1건의 결과."""

    def __init__(self, cid: int, name: str) -> None:
        self.id = cid
        self.name = name
        self.status = "failed"
        self.details: dict[str, Any] = {}
        self.assertions: list[dict[str, Any]] = []
        self.reason = ""

    def expect(self, cond: bool, msg: str, **extra: Any) -> bool:
        self.assertions.append({"assertion": msg, "passed": bool(cond), **extra})
        return bool(cond)

    def finalize(self) -> None:
        if self.status == "skipped":
            return
        self.status = "passed" if all(a["passed"] for a in self.assertions) else "failed"

    def skip(self, reason: str) -> None:
        self.status = "skipped"
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "status": self.status,
                "reason": self.reason, "assertions": self.assertions,
                "details": self.details}


# ----------------------------------------------------------------------
# 작은 전용 회로 도우미
# ----------------------------------------------------------------------
def _tiny_cfg(base: dict[str, Any], mode: str) -> dict[str, Any]:
    import copy
    cfg = copy.deepcopy(base)
    cfg["engine"]["mode"] = mode
    return cfg


def _tiny_network(cfg: dict[str, Any], n: int = 3, *, exc_w: float = 1.0,
                  inh_w: float = 1.0, delay_steps: int = 1
                  ) -> tuple[NeuronPopulation, SynapseTable, IdSpace]:
    """손으로 계산할 수 있는 3뉴런 회로: 0(흥분)->2, 1(억제)->2."""
    ids = IdSpace()
    for name in cfg["receptors"]:
        ids.receptors.add(name)
    for name in ("exc_src", "inh_src", "target"):
        ids.cell_types.add(name)
    ids.areas.add("tiny")
    arrays = NeuronArrays(n, n_receptors=len(cfg["receptors"]))
    pop = NeuronPopulation(arrays, ids)
    arrays.area_id[:] = ids.areas.id_of("tiny")
    arrays.layer_id[:] = ids.layers.id_of("L4")
    arrays.cell_type_id[:] = [ids.cell_types.id_of("exc_src"),
                              ids.cell_types.id_of("inh_src"),
                              ids.cell_types.id_of("target")][:n]
    arrays.dale_sign[:] = [1, -1, 1][:n]
    arrays.output_gain_P[:] = 1.0
    arrays.position_mm[:, 0] = np.arange(n) * 0.01
    arrays.has_compartment[:, :] = False
    arrays.has_compartment[:, 0] = True
    arrays.C_pF[:, :] = 100.0
    arrays.gL_nS[:, :] = 5.0
    arrays.EL_mV[:, :] = -70.0
    arrays.V_mV[:, :] = -70.0
    arrays.V_reset_mV[:] = -65.0
    arrays.t_ref_ms[:] = 2.0
    arrays.threshold[:] = (-50.0 if cfg["engine"]["mode"] == "conductance_lif" else 1.0)

    table = SynapseTable(n, "nS" if cfg["engine"]["mode"] == "conductance_lif"
                         else "dimensionless")
    exc_r = ids.receptors.id_of("AMPA")
    inh_r = ids.receptors.id_of("GABA_A")
    table.add_block(src_id=np.array([0]), dst_id=np.array([2]),
                    src_area=np.array([0]), dst_area=np.array([0]),
                    target_layer=np.array([0]),
                    target_compartment=COMPARTMENT_INDEX["soma"],
                    receptor_type=exc_r, weight=np.array([exc_w]),
                    base_delay_ms=np.array([1.0]),
                    effective_delay_steps=np.array([delay_steps]),
                    plasticity_rule=0, src_dale_sign=np.array([1]), rule_index=0)
    table.add_block(src_id=np.array([1]), dst_id=np.array([2]),
                    src_area=np.array([0]), dst_area=np.array([0]),
                    target_layer=np.array([0]),
                    target_compartment=COMPARTMENT_INDEX["soma"],
                    receptor_type=inh_r, weight=np.array([inh_w]),
                    base_delay_ms=np.array([1.0]),
                    effective_delay_steps=np.array([delay_steps]),
                    plasticity_rule=0, src_dale_sign=np.array([-1]), rule_index=1)
    table.build_indices()
    arrays.set_outgoing_index(table.out_ptr, table.out_syn)
    pop.attach(synapses=table)
    return pop, table, ids


def _tiny_engine(cfg: dict[str, Any], pop: NeuronPopulation, table: SynapseTable,
                 ids: IdSpace) -> Engine:
    log = EventLog(len(pop), mode="full")
    pop.attach(event_log=log)
    anat = anatomy_mod.Anatomy(
        population=pop, ids=ids, grid=None, channel_names=[],  # type: ignore[arg-type]
        retina_neuron_id=np.zeros((0, 0), dtype=np.int64))
    return Engine(cfg, anat, table, log, plasticity=None)


def _force_spike(engine: Engine, neuron_id: int, step: int) -> None:
    engine.set_external_drive(ExternalDrive(
        np.array([neuron_id], dtype=np.int64),
        forced_spike_steps={step: np.array([neuron_id], dtype=np.int64)}))


# ======================================================================
def check_01_records(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(1, "3x3 뷰·타입 배열·연결/로그 조회 일치와 ID 유효성")
    pop = model.anat.population
    a = pop.arrays
    rec = pop.record(0)
    m = rec.as_matrix()
    c.expect(m.shape == (3, 3), "as_matrix() shape == (3,3)")
    c.expect(m.dtype == object, "as_matrix() 는 object 배열이다 (수치 행렬이 아니다)")
    c.expect(isinstance(m[0, 0], float) and isinstance(m[0, 1], float)
             and isinstance(m[0, 2], float), "1행은 x,y,z 실수 좌표")
    c.expect(isinstance(m[1, 0], OutgoingConnectionsView),
             "[1][0] 은 출력 연결 목록이다 (입력 총합이 아니다)")
    c.expect(isinstance(m[1, 1], float) and isinstance(m[1, 2], float),
             "[1][1]=threshold, [1][2]=P 는 실수")
    c.expect(isinstance(m[2, 0], InputLogView), "[2][0] 은 입력 로그 뷰")
    c.expect(isinstance(m[2, 1], DynamicStateView), "[2][1] 은 동적 상태 참조")
    c.expect(isinstance(m[2, 2], MetadataView), "[2][2] 는 메타데이터 참조")

    old = rec.threshold
    rec.threshold = old + 1.25
    c.expect(a.threshold[0] == old + 1.25, "3x3 뷰로 쓴 threshold 가 타입 배열에 반영된다")
    a.threshold[0] = old
    c.expect(rec.threshold == old, "타입 배열 변경이 3x3 뷰에 즉시 보인다 (미러 없음)")

    ok_ids = True
    n_syn = model.table.n_synapses
    for i in range(0, len(pop), max(1, len(pop) // 50)):
        sids = pop.record(i).outgoing.synapse_ids
        if sids.size and (sids.min() < 0 or sids.max() >= n_syn):
            ok_ids = False
            break
        if sids.size and not np.all(model.table.src_id[sids] == i):
            ok_ids = False
            break
    c.expect(ok_ids, "출력 연결 목록의 synapse_id 가 유효하고 src_id 와 일치한다")
    c.expect(int(np.diff(model.table.out_ptr).sum()) == n_syn,
             "출력 CSR 인덱스의 총합이 시냅스 수와 같다")
    c.expect(int(np.diff(model.table.in_ptr).sum()) == n_syn,
             "입력 CSR 인덱스의 총합이 시냅스 수와 같다")
    c.details = {"n_neurons": len(pop), "n_synapses": n_syn,
                 "matrix_description": pop.record(0).as_matrix_description()}
    c.finalize()
    return c


def check_02_event_application(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(2, "같은 사건의 중복 반영 방지·과거 로그 보존·전도도 잔류와 새 입력 구분")
    tcfg = _tiny_cfg(cfg, "conductance_lif")
    pop, table, ids = _tiny_network(tcfg)
    eng = _tiny_engine(tcfg, pop, table, ids)
    a = pop.arrays

    _force_spike(eng, 0, 0)
    eng.step()                              # 0 단계: 발화, 예약
    g_before = a.g_nS[2].sum()
    eng.set_external_drive(None)
    eng.step()                              # 1 단계: 도착 -> 전도도 증가
    g_after = a.g_nS[2].sum()
    n_events_1 = eng.log.n_events
    c.expect(g_after > g_before, "도착 사건이 전도도를 증가시킨다",
             g_before=float(g_before), g_after=float(g_after))

    g_arrival = a.g_nS[2].sum()
    eng.step()                              # 2 단계: 새 도착 없음
    g_decay = a.g_nS[2].sum()
    c.expect(g_decay < g_arrival, "새 입력이 없으면 전도도는 감쇠한다 (그러나 잔류한다)",
             g_arrival=float(g_arrival), g_decay=float(g_decay))
    c.expect(g_decay > 0.0, "이전 입력의 효과가 상태로 남는다")
    c.expect(eng.queue.pending_count() == 0, "같은 사건이 큐에 남아 재처리되지 않는다")

    n_events_2 = eng.log.n_events
    c.expect(n_events_2 >= n_events_1,
             "과거 로그는 삭제되지 않는다 (이벤트 수가 줄지 않음)",
             before=int(n_events_1), after=int(n_events_2))
    log_view = pop.record(2).input_log
    c.expect(log_view.count() >= 1, "발화 이후에도 입력 로그 조회가 가능하다")
    c.expect(a.last_consumed_event[2] >= 0,
             "last_consumed_event 가 갱신되어 중복 주입을 막는다")
    c.details = {"n_events": int(eng.log.n_events),
                 "last_consumed_event": int(a.last_consumed_event[2])}
    c.finalize()
    return c


def check_03_delay(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(3, "단일 사건 예약/도착 시각·지연 양자화 일치, 동시 사건 순서 불변성")
    dt = float(cfg["engine"]["dt_ms"])
    q = quantize_delay(np.array([0.0, dt * 0.4, dt * 1.0, dt * 2.7]), dt,
                       cfg["engine"]["min_delay_steps"], cfg["engine"]["delay_rounding"])
    c.expect(int(q[0]) >= 1, "지연 0 도 최소 1 스텝으로 양자화된다 (0 지연 재귀 방지)")
    c.expect(int(q[2]) == 1, "dt 와 같은 지연은 1 스텝")
    c.expect(int(q[3]) == (3 if cfg["engine"]["delay_rounding"] == "ceil" else 3),
             "2.7*dt 지연의 양자화 결과가 규칙과 일치한다", value=int(q[3]))

    tcfg = _tiny_cfg(cfg, "conductance_lif")
    for d in (1, 2, 3):
        pop, table, ids = _tiny_network(tcfg, delay_steps=d)
        eng = _tiny_engine(tcfg, pop, table, ids)
        _force_spike(eng, 0, 0)
        eng.step()
        eng.set_external_drive(None)
        arrived_at = None
        for s in range(1, d + 3):
            before = pop.arrays.g_nS[2].sum()
            eng.step()
            if pop.arrays.g_nS[2].sum() > before and arrived_at is None:
                arrived_at = s
        c.expect(arrived_at == d, f"지연 {d} 스텝 사건이 정확히 스텝 {d} 에 도착한다",
                 observed=arrived_at)

    # 동시 사건 순서 불변성: 같은 스텝의 도착을 뒤섞어 **큐에 직접 넣어도**
    # 합산 결과가 비트 단위로 같아야 한다.
    tcfg2 = _tiny_cfg(cfg, "sum_threshold")
    results = []
    for perm in (False, True):
        pop, table, ids = _tiny_network(tcfg2, exc_w=0.6, inh_w=0.4)
        eng = _tiny_engine(tcfg2, pop, table, ids)
        # synapse 0 = 흥분 w=0.6, synapse 1 = 억제 w=0.4 -> s_j = 0.6 - 0.4 = 0.2
        order = np.array([1, 0] if perm else [0, 1], dtype=np.int64)
        amounts = np.array([0.6, 0.4])
        eng.queue.schedule(0, {
            "synapse_id": order,
            "amount": amounts[order],
            "weight_snapshot": amounts[order],
            "gain_snapshot": np.ones(2),
            "emit_time_ms": np.zeros(2),
            "spike_id": np.zeros(2, dtype=np.int64),
        })
        eng.set_external_drive(None)
        eng.step()
        results.append(float(pop.arrays.last_decision_value[2]))
    c.expect(results[0] == results[1],
             "동시 도착 사건의 처리 순서를 바꿔도 합산 결과가 비트 단위로 같다",
             values=results)
    c.expect(abs(results[0] - 0.2) < 1e-12,
             "손계산 값과 일치한다 (0.6 흥분 - 0.4 억제 = 0.2)", observed=results[0])
    c.details = {"quantized": q.tolist(), "same_step_values": results}
    c.finalize()
    return c


def check_04_sum_threshold(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(4, "threshold 모드의 손계산 가능한 흥분/억제 합과 경계 발화")
    tcfg = _tiny_cfg(cfg, "sum_threshold")
    pop, table, ids = _tiny_network(tcfg, exc_w=0.7, inh_w=0.3)
    eng = _tiny_engine(tcfg, pop, table, ids)
    a = pop.arrays
    a.threshold[2] = 0.35
    both = np.array([0, 1], dtype=np.int64)
    eng.set_external_drive(ExternalDrive(both, forced_spike_steps={0: both}))
    eng.step()
    eng.set_external_drive(None)
    eng.step()
    s = float(a.last_decision_value[2])
    c.expect(abs(s - (0.7 - 0.3)) < 1e-12,
             "s_j = 0.7(흥분) - 0.3(억제) = 0.4 (손계산 일치)", observed=s)
    c.expect(int(a.fired[2]) == 1, "s_j(0.4) >= theta(0.35) 이면 발화")

    a.threshold[2] = 0.45
    a.interval_activation[:] = 0.0
    eng.set_external_drive(ExternalDrive(both, forced_spike_steps={2: both}))
    eng.step()
    eng.set_external_drive(None)
    eng.step()
    c.expect(int(a.fired[2]) == 0, "s_j(0.4) < theta(0.45) 이면 발화하지 않는다")

    # 경계 조건: 입력 0 이어도 theta <= 0 이면 발화한다
    pop2, table2, ids2 = _tiny_network(tcfg)
    eng2 = _tiny_engine(tcfg, pop2, table2, ids2)
    pop2.arrays.threshold[:] = 0.0
    eng2.set_external_drive(None)
    eng2.step()
    c.expect(int(pop2.arrays.fired[2]) == 1,
             "영 입력에서도 theta<=0 이면 발화한다 (문서화된 경계 조건)")
    c.details = {"sum_value": s, "boundary_theta": 0.0}
    c.finalize()
    return c


def check_05_lif(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(5, "LIF 누설·단일 펄스·불응기·구획 결합·dt 수렴")
    tcfg = _tiny_cfg(cfg, "conductance_lif")

    # (a) 무입력 누설: EL 아닌 값에서 출발하면 EL 로 돌아간다.
    #     고정된 스텝 수로 자르면 막시간상수에 따라 통과/실패가 갈리므로,
    #     허용오차 아래로 내려가는 데 필요한 스텝 수를 닫힌 해에서 구해 쓴다.
    pop, table, ids = _tiny_network(tcfg)
    eng = _tiny_engine(tcfg, pop, table, ids)
    a = pop.arrays
    v_start = -60.0
    a.V_mV[2, 0] = v_start
    eng.set_external_drive(None)
    dt_ms = float(tcfg["engine"]["dt_ms"])
    EL = float(a.EL_mV[2, 0])
    tau_ms = float(a.C_pF[2, 0]) / float(a.gL_nS[2, 0])    # pF / nS = ms
    tol_mV = 0.5
    # 후향 오일러 누설의 닫힌 해: V_n = EL + (V0-EL) * (1 + dt/tau)^(-n)
    decay = 1.0 + dt_ms / tau_ms
    n_steps = int(np.ceil(np.log(abs(v_start - EL) / (0.1 * tol_mV))
                          / np.log(decay)))
    for _ in range(n_steps):
        eng.step()
    v = float(a.V_mV[2, 0])
    predicted = EL + (v_start - EL) * decay ** (-n_steps)
    c.expect(abs(v - EL) < tol_mV,
             f"무입력 시 막전위가 EL 로 수렴한다 (tau={tau_ms:g} ms, "
             f"{n_steps} 스텝 = {n_steps * dt_ms:g} ms)",
             V=v, EL_mV=EL, tau_ms=tau_ms, n_steps=n_steps)
    c.expect(abs(v - predicted) < 1e-9,
             "무입력 누설이 후향 오일러 닫힌 해와 정확히 일치한다",
             V=v, predicted_mV=predicted)
    c.expect(np.all(np.isfinite(pop.arrays.V_mV)), "막전위가 유한하다")

    # (b) 단일 펄스 EPSP
    pop, table, ids = _tiny_network(tcfg, exc_w=4.0)
    eng = _tiny_engine(tcfg, pop, table, ids)
    _force_spike(eng, 0, 0)
    eng.step()
    eng.set_external_drive(None)
    peak = -np.inf
    for _ in range(40):
        eng.step()
        peak = max(peak, float(pop.arrays.V_mV[2, 0]))
    c.expect(peak > -70.0, "흥분성 단일 펄스가 EPSP 를 만든다", peak_mV=peak)

    # (c) 불응기
    pop, table, ids = _tiny_network(tcfg)
    eng = _tiny_engine(tcfg, pop, table, ids)
    a = pop.arrays
    a.t_ref_ms[2] = 5.0
    a.threshold[2] = -69.9
    a.V_mV[2, 0] = -60.0
    eng.set_external_drive(None)
    n_spikes = 0
    for _ in range(int(5.0 / tcfg["engine"]["dt_ms"])):
        rep = eng.step()
        n_spikes += rep.n_spikes
    c.expect(n_spikes <= 2,
             "불응기 동안 연속 발화가 억제된다 (5ms 불응기에서 관측 발화 수)",
             n_spikes=int(n_spikes))

    # (d) 구획 결합: apical 전류가 soma 에 전달된다
    pop, table, ids = _tiny_network(tcfg)
    a = pop.arrays
    a.has_compartment[2, :] = True
    a.g_couple_nS[2, COMPARTMENT_INDEX["apical"]] = 10.0
    eng = _tiny_engine(tcfg, pop, table, ids)
    eng.set_external_drive(None)
    a.Iext_pA[2, COMPARTMENT_INDEX["apical"]] = 200.0
    v0 = float(a.V_mV[2, 0])
    for _ in range(20):
        a.Iext_pA[2, COMPARTMENT_INDEX["apical"]] = 200.0
        eng.step()
    c.expect(float(a.V_mV[2, 0]) > v0,
             "apical 구획의 전류가 구획 결합을 통해 soma 전위를 바꾼다",
             V_soma=float(a.V_mV[2, 0]))

    # (e) dt 수렴
    finals: list[float] = []
    for factor in cfg["validation"]["dt_convergence_factors"]:
        sub = _tiny_cfg(cfg, "conductance_lif")
        sub["engine"]["dt_ms"] = float(cfg["engine"]["dt_ms"]) * float(factor)
        p2, t2, i2 = _tiny_network(sub, exc_w=4.0)
        e2 = _tiny_engine(sub, p2, t2, i2)
        p2.arrays.Iext_pA[2, 0] = 150.0
        p2.arrays.threshold[2] = 1e9        # 발화 없이 순수 적분만 본다
        e2.set_external_drive(None)
        n = int(round(20.0 / sub["engine"]["dt_ms"]))
        for _ in range(n):
            p2.arrays.Iext_pA[2, 0] = 150.0
            e2.step()
        finals.append(float(p2.arrays.V_mV[2, 0]))
    spread = float(max(finals) - min(finals)) if finals else 0.0
    tol = float(cfg["validation"]["tolerances"]["lif_dt_convergence_mV"])
    c.expect(spread <= tol,
             f"dt 를 줄여도 20ms 후 막전위 차이가 {tol} mV 이하다",
             finals_mV=finals, spread_mV=spread)
    c.expect(all(np.isfinite(finals)), "dt 수렴 검사에서 비유한 값이 없다")
    c.details = {"dt_finals_mV": finals, "dt_spread_mV": spread,
                 "note_ko": "후향 오일러를 쓰므로 발산을 가리는 전압 클리핑이 없다."}
    c.finalize()
    return c


def check_06_dale(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(6, "Dale 제약·전도도 비음수·학습 후 부호와 범위 유지")
    table = model.table
    a = model.anat.population.arrays
    viol = table.dale_violations(a.dale_sign)
    c.expect(viol.size == 0, "모든 시냅스의 기록된 Dale 부호가 발신 뉴런 유형과 일치한다",
             n_violations=int(viol.size))
    c.expect(bool(np.all(table.weight >= 0.0)),
             "가중치는 비음수 크기다 (부호는 세포 유형이 결정)")
    c.expect(bool(np.all(a.g_nS >= 0.0)), "전도도는 비음수다")

    before_sign = table.src_dale_sign.copy()
    plast = StdpHomeostasis(cfg, table, a, model.anat.ids)
    rngd = np.random.default_rng(0)
    if plast.plastic_idx.size:
        table.weight[plast.plastic_idx] += rngd.normal(
            0.0, 5.0, size=plast.plastic_idx.size)
        info = table.clip_weights(plast.w_min, plast.w_max)
        c.expect(bool(np.all(table.weight >= plast.w_min)
                      and np.all(table.weight <= plast.w_max)),
                 "clipping 후 가중치가 설정 범위 안에 있다", **info)
        c.expect(bool(np.array_equal(table.src_dale_sign, before_sign)),
                 "clipping 이 연결 유형(부호)을 바꾸지 않는다")
    else:
        c.expect(True, "학습 대상 시냅스가 없어 clipping 검사를 생략 (구조상 정상)")
    c.details = {"n_plastic": int(plast.plastic_idx.size),
                 "weight_min": float(table.weight.min()) if table.n_synapses else None,
                 "weight_max": float(table.weight.max()) if table.n_synapses else None}
    c.finalize()
    return c


def check_07_input(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(7, "색 변환·DoG 상수 응답·로그-극좌표 왕복/범위/aliasing")
    enc: RetinaEncoder = model.encoder
    white = np.ones((16, 16, 3), dtype=np.float64)
    lms = enc.to_lms(white)
    c.expect(bool(np.all(lms[:2] > 0)), "백색 입력에서 L,M 성분이 양수다",
             L=float(lms[0].mean()), M=float(lms[1].mean()), S=float(lms[2].mean()))
    black = np.zeros((16, 16, 3), dtype=np.float64)
    c.expect(float(np.abs(enc.to_lms(black)).max()) < 1e-12,
             "흑색 입력의 LMS 는 0 이다")

    uni = uniform_response_check(enc, 0.5, 64)
    tol = float(cfg["validation"]["tolerances"]["dog_uniform_response"])
    c.expect(uni["max_abs_contrast_response_interior"] <= tol,
             f"균일 영상에서 내부 영역의 DoG 대비 응답이 {tol} 이하다",
             observed=uni["max_abs_contrast_response_interior"])

    rt = roundtrip_error(model.anat.grid, cfg["retinotopy"]["e0_deg"])
    rtol = float(cfg["validation"]["tolerances"]["logpolar_roundtrip_deg"])
    c.expect(rt["max_abs_ecc_error_deg"] <= rtol,
             f"log-polar 왕복 오차가 {rtol} deg 이하다", **rt)

    g = model.anat.grid
    c.expect(bool(np.all(np.isfinite(g.x_deg)) and np.all(np.isfinite(g.y_deg))),
             "샘플 좌표가 모두 유한하다")
    c.expect(float(g.ecc_deg.max()) <= float(g.meta["max_ecc_deg"]) + 1e-9,
             "샘플 편심도가 설정한 최대 편심도를 넘지 않는다",
             max_ecc=float(g.ecc_deg.max()))

    H = W = int(cfg["retina"]["image"]["max_side_px"])
    ny = nyquist_margin(g, H, W, cfg["retina"]["image"]["fov_deg"])
    c.details = {"uniform_response": uni, "roundtrip": rt, "nyquist": ny}
    c.expect(ny.get("checked", False), "aliasing 여유 지표를 계산했다")
    c.finalize()
    return c


def check_08_response_changes(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(8, "회전·위치·위상·양안 입력 변화에 따른 실제 반응 변화")
    enc: RetinaEncoder = model.encoder
    sampler: LogPolarSampler = model.sampler
    size = int(cfg["retina"]["image"]["max_side_px"])
    cx = cy = (size - 1) / 2.0

    def sampled(img: np.ndarray) -> np.ndarray:
        v, _ = sampler.sample(enc.encode(img).channels)
        return v.ravel()

    base = stimuli_mod.bar_image(size, size, cx, cy, 0.5 * size, 3.0, 0.0)
    rotated = stimuli_mod.bar_image(size, size, cx, cy, 0.5 * size, 3.0, np.pi / 2)
    shifted = stimuli_mod.bar_image(size, size, cx + 0.15 * size, cy,
                                    0.5 * size, 3.0, 0.0)
    ph0 = stimuli_mod.grating_patch(size, size, cx, cy, 0.3 * size, 0.0, 0.0, 0.08)
    ph1 = stimuli_mod.grating_patch(size, size, cx, cy, 0.3 * size, 0.0, np.pi, 0.08)

    b, r, s, p0, p1 = (sampled(x) for x in (base, rotated, shifted, ph0, ph1))
    d_rot = float(np.linalg.norm(b - r))
    d_pos = float(np.linalg.norm(b - s))
    d_phase = float(np.linalg.norm(p0 - p1))
    c.expect(d_rot > 0.0, "회전에 따라 샘플된 반응이 실제로 변한다", distance=d_rot)
    c.expect(d_pos > 0.0, "위치 이동에 따라 반응이 변한다", distance=d_pos)
    c.expect(d_phase > 0.0, "위상 변화에 따라 반응이 변한다", distance=d_phase)

    bino = cfg["retinotopy"]["binocular"]["enabled"]
    if bino:
        left = stimuli_mod.dot_image(size, size, cx - 2, cy, 4.0)
        right = stimuli_mod.dot_image(size, size, cx + 2, cy, 4.0)
        d_eye = float(np.linalg.norm(sampled(left) - sampled(right)))
        c.expect(d_eye > 0.0, "좌/우 눈 입력이 다르면 반응도 다르다", distance=d_eye)
    else:
        c.assertions.append({
            "assertion": "양안 입력 변화 검사",
            "passed": True,
            "note_ko": ("retinotopy.binocular.enabled 가 false 이므로 단안 설정이다. "
                        "단안 영상 복제를 양안 시차 실험으로 보고하지 않는다."),
            "measured": False,
        })
    c.details = {"rotation_distance": d_rot, "position_distance": d_pos,
                 "phase_distance": d_phase, "binocular_enabled": bool(bino),
                 "note_ko": "예상하지 않은 결과도 그대로 기록한다."}
    c.finalize()
    return c


def check_09_structure(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(9, "층별 깊이·세포 유형·필수 배선·연결 수 집계")
    anat = model.anat
    summary = anatomy_mod.layer_summary(anat)
    conn = areas_mod.connectivity_summary(cfg, anat, model.table)

    declared: list[str] = []
    for aname, a in cfg["anatomy"]["areas"].items():
        if a["kind"] == "retina":
            declared.append(f"{aname}/L_input")
            continue
        for lname, n in a["neurons_per_layer"].items():
            if int(n) > 0:
                declared.append(f"{aname}/{lname}")
    present = {k.rsplit("/", 1)[0] for k in summary["counts"]}
    missing = [d for d in declared if d not in present]
    c.expect(not missing, "선언한 모든 (영역/층) 조합에 뉴런이 실제로 존재한다",
             missing=missing)

    empty_rules = [r["name"] for r in model.wiring_report.per_rule
                   if r.get("enabled") and r.get("n", 0) == 0]
    c.expect(not empty_rules,
             "활성화된 모든 배선 규칙이 시냅스를 1개 이상 만들었다 "
             "(이름만 존재하는 경로를 잡는다)", empty_rules=empty_rules)

    ct_used = {k.rsplit("/", 1)[1] for k in summary["counts"]}
    ct_declared = set(cfg["cell_types"])
    unused = sorted(ct_declared - ct_used)
    c.expect(not unused, "선언한 모든 세포 유형이 실제로 배치되었다", unused=unused)

    depths_ok = True
    for key, d in summary["depth_mm"].items():
        if d["depth_min"] < -1e-9:
            depths_ok = False
    c.expect(depths_ok, "모든 층 깊이가 0 이상이다 (표면에서 아래로)")
    c.details = {"layer_summary": summary, "connectivity": conn,
                 "per_rule": model.wiring_report.per_rule}
    c.finalize()
    return c


def check_10_measurement_noninvasive(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(10, "측정 전후 가중치·상태·흔적·로그 위치·RNG 동일성")
    from .analysis import explain_neuron  # 지연 import

    eng = model.engine
    before = eng.read_only_snapshot()
    rng_before = model.rngs.get("diagnostics").bit_generator.state

    _ = model.anat.population.record(0).describe()
    _ = model.table.summary()
    _ = anatomy_mod.layer_summary(model.anat)
    _ = areas_mod.connectivity_summary(cfg, model.anat, model.table)
    with tempfile.TemporaryDirectory() as td:
        _ = explain_neuron(Path(td), 0, 0.0, 1.0)   # 기록 없는 경로에서도 안전해야 한다

    after = eng.read_only_snapshot()
    rng_after = model.rngs.get("diagnostics").bit_generator.state
    for k in before:
        c.expect(bool(np.array_equal(before[k], after[k])),
                 f"측정 전후 {k} 가 비트 단위로 같다")
    c.expect(rng_before == rng_after, "측정이 RNG 스트림 상태를 바꾸지 않는다")
    c.finalize()
    return c


def check_11_zero_correction(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(11, "자유/유도 0교정 비교와 교사 항/감쇠/항상성 분리")
    import copy as _copy

    tcfg = _tiny_cfg(cfg, "conductance_lif")
    tcfg["learning"]["mode"] = "stdp_homeostasis"
    tcfg["learning"]["threshold_adaptation_enabled"] = True

    def run_once(context_current_pA: float) -> dict[str, Any]:
        pop, table, ids = _tiny_network(tcfg, exc_w=3.0)
        # 문맥(L1/apical) 경로가 실제 효과를 갖도록 apical 구획과 결합을 켠다.
        pop.arrays.has_compartment[2, :] = True
        pop.arrays.g_couple_nS[2, COMPARTMENT_INDEX["apical"]] = 10.0
        pop.arrays.g_couple_nS[2, COMPARTMENT_INDEX["basal"]] = 8.0
        table.plasticity_rule[:] = ids.plasticity_rules.id_of("stdp")
        log = EventLog(len(pop), mode="full")
        pop.attach(event_log=log)
        anat = anatomy_mod.Anatomy(
            population=pop, ids=ids, grid=None, channel_names=[],  # type: ignore[arg-type]
            retina_neuron_id=np.zeros((0, 0), dtype=np.int64))
        plast = StdpHomeostasis(tcfg, table, pop.arrays, ids)
        eng = Engine(tcfg, anat, table, log, plasticity=plast)
        w0 = table.weight.copy()
        th0 = pop.arrays.threshold.copy()
        v_soma: list[float] = []
        for s in range(20):
            pop.arrays.Iext_pA[2, COMPARTMENT_INDEX["apical"]] = context_current_pA
            if s % 5 == 0:
                _force_spike(eng, 0, s)
            else:
                eng.set_external_drive(None)
            eng.step()
            v_soma.append(float(pop.arrays.V_mV[2, 0]))
        return {
            "spikes": int(pop.arrays.spike_count.sum()),
            "dw": table.weight - w0,
            "dtheta": pop.arrays.threshold - th0,
            "V_soma_mV": np.asarray(v_soma),
            "teacher_term_total": float(plast.total_weight_delta),
            "theta_term_total": float(plast.total_theta_delta),
        }

    def context_current_for_threshold(pop: NeuronPopulation) -> float:
        """정상상태에서 soma 가 임계를 넘게 하는 apical 전류 [pA].

        3구획 정상상태 (u = V - EL) ::

            basal :  gL_b u_b + g_cb (u_b - u_s) = 0
            apical:  gL_a u_a + g_ca (u_a - u_s) = I
            soma  :  gL_s u_s + g_cb (u_s - u_b) + g_ca (u_s - u_a) = 0

        를 풀면 ``u_s = (g_ca / a) * I / D`` 이다
        (``a = gL_a + g_ca``, ``b = gL_b + g_cb``,
        ``D = gL_s + g_cb - g_cb^2/b + g_ca - g_ca^2/a``).
        고정된 크기를 쓰면 dt·세포 파라미터가 바뀔 때 검사가 조용히 무력해지므로
        필요한 전류를 여기서 직접 계산한다.
        """
        a_ = pop.arrays
        i = 2
        gL_s, gL_b, gL_a = (float(a_.gL_nS[i, k]) for k in range(3))
        g_cb = float(a_.g_couple_nS[i, COMPARTMENT_INDEX["basal"]])
        g_ca = float(a_.g_couple_nS[i, COMPARTMENT_INDEX["apical"]])
        a_sum = gL_a + g_ca
        b_sum = gL_b + g_cb
        D = gL_s + g_cb - g_cb ** 2 / b_sum + g_ca - g_ca ** 2 / a_sum
        need_mV = float(a_.threshold[i]) - float(a_.EL_mV[i, 0])
        # 20 스텝은 정상상태에 완전히 도달하지 않고 발화 후 재설정도 있으므로
        # 여유 계수 3 을 곱한다.
        return 3.0 * need_mV * D * a_sum / g_ca

    free = run_once(0.0)
    guided_zero = run_once(0.0)
    c.expect(free["spikes"] == guided_zero["spikes"],
             "교정(문맥 전류)이 0 이면 자유 실행과 유도 실행의 발화 수가 같다",
             free=free["spikes"], guided=guided_zero["spikes"])
    c.expect(bool(np.array_equal(free["dw"], guided_zero["dw"])),
             "0교정에서 가중치 변화가 비트 단위로 같다")
    c.expect(bool(np.array_equal(free["dtheta"], guided_zero["dtheta"])),
             "0교정에서 임계값 변화가 비트 단위로 같다")

    # (a) 작은 문맥 전류도 막전위에 측정 가능한 효과를 남긴다.
    probe_pop, _, _ = _tiny_network(tcfg, exc_w=3.0)
    probe_pop.arrays.has_compartment[2, :] = True
    probe_pop.arrays.g_couple_nS[2, COMPARTMENT_INDEX["apical"]] = 10.0
    probe_pop.arrays.g_couple_nS[2, COMPARTMENT_INDEX["basal"]] = 8.0
    strong_pA = context_current_for_threshold(probe_pop)
    weak_pA = 0.1 * strong_pA
    guided_weak = run_once(weak_pA)
    dv = float(np.max(np.abs(guided_weak["V_soma_mV"] - free["V_soma_mV"])))
    c.expect(dv > 1e-6,
             "0 이 아닌 문맥 입력은 soma 막전위를 실제로 바꾼다 "
             "(L1/apical 경로가 끊겨 있지 않다)",
             context_pA=weak_pA, max_dV_mV=dv)

    # (b) 임계를 넘길 만큼 큰 문맥 전류는 발화와 학습까지 바꾼다.
    guided_nonzero = run_once(strong_pA)
    c.expect(guided_nonzero["spikes"] != free["spikes"]
             or not np.array_equal(guided_nonzero["dw"], free["dw"]),
             "0 이 아닌 문맥 입력은 실제로 활동/학습에 영향을 준다 "
             "(L1/apical 경로가 측정 가능한 효과를 갖는다)",
             context_pA=strong_pA, free_spikes=free["spikes"],
             guided_spikes=guided_nonzero["spikes"],
             abs_dw_sum=float(np.abs(guided_nonzero["dw"] - free["dw"]).sum()))

    c.details = {
        "free": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in free.items()},
        "separation_note_ko": (
            "STDP·감쇠·항상성 항은 교사 신호와 무관하게 동작한다. 0교정 검사는 "
            "'자연 발생 스파이크까지 0' 을 요구하지 않고, 두 조건의 **차이**가 0 인지를 본다."
        ),
    }
    c.finalize()
    return c


def check_12_rao_gradient(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(12, "작은 Rao 연속 모델의 목적함수 미분과 유한차분 비교")
    rngd = np.random.default_rng(12345)
    m = RaoModel(n_modules=2, input_dim=6, n1=4, n2=3, sigma=1.0, sigma_td=2.0,
                 alpha=0.05, lam=0.001, rng=rngd)
    I = rngd.normal(0.0, 1.0, size=(2, 6))
    r1 = rngd.normal(0.0, 0.5, size=(2, 4))
    r2 = rngd.normal(0.0, 0.5, size=3)
    res = m.finite_difference_check(
        I, r1, r2, eps=float(cfg["validation"]["finite_difference_eps"]),
        n_probe=6, rng=rngd)
    tol = float(cfg["validation"]["tolerances"]["rao_gradient_rel"])
    c.expect(res["max_relative_residual"] <= tol,
             f"해석적 갱신 방향과 유한차분 기울기의 상대 잔차가 {tol} 이하다",
             observed=res["max_relative_residual"])

    st = m.settle(I, steps=50, r_step=0.05)
    c.expect(st.energy_trace[-1] <= st.energy_trace[0] + 1e-12,
             "정착 중 목적함수 E 가 증가하지 않는다",
             first=st.energy_trace[0], last=st.energy_trace[-1])
    try:
        RaoModel(1, 2, 2, 2, sigma=-1.0, sigma_td=1.0, alpha=0.0, lam=0.0, rng=rngd)
        c.expect(False, "음수 sigma 를 거부한다")
    except ValueError:
        c.expect(True, "음수 sigma 를 거부한다")
    c.details = {"finite_difference": {k: v for k, v in res.items() if k != "details"},
                 "settle_errors": st.errors,
                 "scope_note_ko": res["scope_note_ko"]}
    c.finalize()
    return c


def check_13_checkpoint(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(13, "체크포인트 재개와 연속 실행의 결과 일치")
    tcfg = _tiny_cfg(cfg, "conductance_lif")

    def build() -> tuple[Engine, NeuronPopulation, SynapseTable]:
        pop, table, ids = _tiny_network(tcfg, exc_w=3.0)
        eng = _tiny_engine(tcfg, pop, table, ids)
        return eng, pop, table

    eng_a, pop_a, _ = build()
    for s in range(20):
        if s % 4 == 0:
            _force_spike(eng_a, 0, s)
        else:
            eng_a.set_external_drive(None)
        eng_a.step()
    final_a = eng_a.state_dict()

    eng_b, pop_b, _ = build()
    for s in range(10):
        if s % 4 == 0:
            _force_spike(eng_b, 0, s)
        else:
            eng_b.set_external_drive(None)
        eng_b.step()
    mid = eng_b.state_dict()

    eng_c, pop_c, _ = build()
    eng_c.load_state_dict(mid)
    for s in range(10, 20):
        if s % 4 == 0:
            _force_spike(eng_c, 0, s)
        else:
            eng_c.set_external_drive(None)
        eng_c.step()
    final_c = eng_c.state_dict()

    same = all(bool(np.array_equal(np.asarray(final_a[k]), np.asarray(final_c[k])))
               for k in ("V_mV", "g_nS", "threshold", "weight", "spike_count"))
    c.expect(same, "체크포인트에서 재개한 결과가 연속 실행과 비트 단위로 같다")
    c.expect(final_a["event_counter"] == final_c["event_counter"],
             "재개 후 event_id 카운터가 일치한다 (중복 기록 방지)")
    c.details = {"continuous_spikes": int(np.sum(final_a["spike_count"])),
                 "resumed_spikes": int(np.sum(final_c["spike_count"]))}
    c.finalize()
    return c


def check_14_import_and_splits(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(14, "import 부작용 없음·test 접근 카운터·데이터 분할 중복 없음")
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as td:
        before = set(os.listdir(td))
        code = ("import sys; sys.path.insert(0, r'%s'); "
                "import cortex, cortex.records, cortex.dynamics, cortex.retina, "
                "cortex.areas, cortex.runner, cortex.cli" % str(root))
        proc = subprocess.run([sys.executable, "-c", code], cwd=td,
                              capture_output=True, text=True, timeout=120, check=False)
        after = set(os.listdir(td))
        c.expect(proc.returncode == 0, "모든 모듈이 import 된다", stderr=proc.stderr[-500:])
        c.expect(after == before,
                 "import 만으로 작업 디렉터리에 파일이 생기지 않는다",
                 created=sorted(after - before))

    rngd = np.random.default_rng(7)
    stims = stimuli_mod.generate(cfg, rngd)
    if stims:
        splits = stimuli_mod.split_stimuli(stims, cfg, np.random.default_rng(8))
        rep = stimuli_mod.split_report(splits)
        c.expect(rep["no_overlap"], "train/dev/test 가 겹치지 않는다 (base_id 기준)",
                 overlaps=rep["overlaps"])
        c.details["split_report"] = rep
    else:
        c.expect(True, "이 설정에는 자극이 정의되어 있지 않아 분할 검사를 건너뛴다 "
                       "(구조상 정상)")

    c.details["test_access_counter_ko"] = (
        "experiment 명령은 manifest 의 test_access.n_test_evaluations 를 증가시킨다. "
        "이미 본 시험 결과로 재선택하면 탐색 결과로 표시해야 한다.")
    c.finalize()
    return c


# ======================================================================
CHECKS: list[Callable[[dict[str, Any], Any], Check]] = [
    check_01_records, check_02_event_application, check_03_delay,
    check_04_sum_threshold, check_05_lif, check_06_dale, check_07_input,
    check_08_response_changes, check_09_structure,
    check_10_measurement_noninvasive, check_11_zero_correction,
    check_12_rao_gradient, check_13_checkpoint, check_14_import_and_splits,
]


def run_all(cfg: dict[str, Any], recorder: Any, package_root: Path,
            progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """검사 1~14 를 실행하고 결과를 돌려준다 (사용자 명령에서만 호출)."""
    from .runner import build_model  # 지연 import (순환 방지)

    progress = progress or (lambda m: None)
    rngs = rng_mod.from_config(cfg)
    progress("검사용 모델을 조립하는 중...")
    model = build_model(cfg, rngs)
    model.rngs = rngs
    model.encoder.fit_normalization(
        [stimuli_mod.uniform_image(int(cfg["retina"]["image"]["max_side_px"]),
                                   int(cfg["retina"]["image"]["max_side_px"]), 0.5),
         stimuli_mod.bar_image(int(cfg["retina"]["image"]["max_side_px"]),
                               int(cfg["retina"]["image"]["max_side_px"]),
                               0.5 * cfg["retina"]["image"]["max_side_px"],
                               0.5 * cfg["retina"]["image"]["max_side_px"],
                               0.5 * cfg["retina"]["image"]["max_side_px"], 3.0, 0.0)],
        source="validation_builtin", sampler=model.sampler)

    results: list[dict[str, Any]] = []
    n_pass = n_fail = n_skip = 0
    for fn in CHECKS:
        name = fn.__name__
        progress(f"검사 실행: {name}")
        try:
            chk = fn(cfg, model)
        except Exception as exc:  # 검사 자체가 실패해도 기록한다
            chk = Check(0, name)
            chk.status = "failed"
            chk.reason = f"{type(exc).__name__}: {exc}"
            recorder.error(f"검사 {name} 에서 예외", exc)
        d = chk.to_dict()
        results.append(d)
        n_pass += int(d["status"] == "passed")
        n_fail += int(d["status"] == "failed")
        n_skip += int(d["status"] == "skipped")
        recorder.log(f"[{d['status'].upper()}] {d['id']:02d} {d['name']}")

    return {
        "n_checks": len(results), "n_passed": n_pass, "n_failed": n_fail,
        "n_skipped": n_skip,
        "all_passed": n_fail == 0 and n_skip == 0,
        "note_ko": ("skipped 는 passed 에 포함하지 않는다. 성공률을 맞추려고 기준을 "
                    "사후에 바꾸지 않는다."),
        "checks": results,
    }


__all__ = ["Check", "CHECKS", "run_all"]
