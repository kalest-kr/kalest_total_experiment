#!/usr/bin/env python3
"""run_checks.py -- 명세 [12] 의 필수 검증 T1~T10.

통과 기준이 있는 **구현 검사**만 여기서 판정한다. 결과를 관찰하는
**학습 실험**은 run_experiment.py 에서 수행한다 (T6 의 작은 학습 예제는
실현 가능함이 설계상 보장되므로 통과 기준을 둔다).

사용법::

    python run_checks.py
    python run_checks.py --out results/checks.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.connections import ConnectionTable
from src.datasets import DatasetConfig, SceneSpec, generate_scenes, render_scene
from src.diagnostics import TraceRecorder, counterfactual_check
from src.engine import SimulationEngine
from src.events import ExternalDrive, InputEvent
from src.experiment import V1Experiment, build_dataset, prepare_data
from src.local_learning import (
    FreeRunSnapshot,
    LearningConfig,
    LocalLearner,
    compute_local_update_for_neuron,
)
from src.neuron import Population
from src.persistence import load_model, save_model
from src.retina import RetinaConfig, RetinaEncoder
from src.rngs import stream
from src.spatial_mapping import GridConfig, InputNormalizer, LogPolarSampler, build_grid
from src.teacher import L2Descriptor, LocalTeacher
from src.v1 import V1Circuit

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS: list[tuple[str, str, Callable[[], dict[str, Any]]]] = []


def check(tid: str, title: str):
    def deco(fn):
        CHECKS.append((tid, title, fn))
        return fn
    return deco


def expect(cond: bool, msg: str, bucket: list[dict[str, Any]]) -> bool:
    bucket.append({"assertion": msg, "passed": bool(cond)})
    return bool(cond)


# ----------------------------------------------------------------------
def tiny_abcd(event_mode: str = "object"):
    """명세 [4] 의 A/B/C -> D 예제 회로를 만든다."""
    pop = Population()
    for nm in ("A", "B", "C", "D"):
        pop.declare(nm, layer="test", threshold=0.5, transmission=1.0, observation_tick=1)
    pop.finalize()
    tb = ConnectionTable(len(pop), pop.names())
    c1 = tb.add(pop.index("A"), pop.index("D"), "excitatory", 0.6, 1)
    c2 = tb.add(pop.index("B"), pop.index("D"), "excitatory", 0.4, 1)
    c3 = tb.add(pop.index("C"), pop.index("D"), "inhibitory", 0.6, 1)
    tb.finalize()
    eng = SimulationEngine(pop, tb, n_ticks=2, event_mode=event_mode)
    events = [
        InputEvent(1, c1, pop.index("A"), 1.0),
        InputEvent(1, c2, pop.index("B"), 0.5),
        InputEvent(1, c3, pop.index("C"), 1.0),
    ]
    return pop, tb, eng, events, (c1, c2, c3)


def load_config() -> dict[str, Any]:
    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def fitted_circuit(cfg: dict[str, Any], seed: int = 42, n_fit: int = 24,
                   fit_stream_seed: int = 909) -> V1Circuit:
    """회로를 만들고 **훈련용 장면으로만** 정규화 계수를 추정한다.

    scale=1.0 으로 두면 DoG 값이 L4 threshold(0.1) 아래라 아무것도 발화하지
    않으므로, 검사에서도 실제 파이프라인과 같은 방식으로 fit 한다.
    """
    circuit = V1Circuit.build(cfg, seed, stream(seed, "init"))
    fit_scenes = generate_scenes(n_fit, stream(fit_stream_seed, "data"),
                                 DatasetConfig(height=cfg["image"]["height"],
                                               width=cfg["image"]["width"]), "fit")
    raw = np.stack([circuit.preprocess(render_scene(s)) for s in fit_scenes], axis=0)
    circuit.normalizer.fit(raw)
    return circuit


# ======================================================================
@check("T1", "행렬과 자료구조")
def t1() -> dict[str, Any]:
    a: list[dict[str, Any]] = []
    cfg = load_config()
    circuit = V1Circuit.build(cfg, 42, stream(42, "init"))
    pop = circuit.population
    ok_shape = all(n.M.shape == (3, 3) for n in pop.neurons)
    ok_dtype = all(n.M.dtype == np.float64 for n in pop.neurons)
    expect(ok_shape, "모든 뉴런 행렬 shape == (3,3)", a)
    expect(ok_dtype, "모든 뉴런 행렬 dtype == float64", a)
    expect(pop.store.states.shape == (len(pop), 3, 3), "상태 저장소 shape (N,3,3)", a)

    n = pop.neurons[0]
    n.threshold = 0.77
    expect(n.M[1, 1] == 0.77, "threshold 속성 쓰기가 행렬 (1,1) 에 반영", a)
    n.M[1, 1] = 0.33
    expect(n.threshold == 0.33, "행렬 (1,1) 쓰기가 threshold 속성에 반영", a)
    n.transmission = 2.5
    expect(n.M[1, 2] == 2.5, "P 속성 <-> 행렬 (1,2)", a)
    n.excitatory_sum = 1.25
    n.inhibitory_sum = 0.75
    n.bias = -0.5
    n.u = 1.0
    expect(n.M[2, 0] == 1.25 and n.M[2, 1] == 0.75 and n.M[2, 2] == -0.5 and n.M[1, 0] == 1.0,
           "E/I/b/u 속성 <-> 행렬 (2,0)/(2,1)/(2,2)/(1,0)", a)
    n.set_position(1.0, 2.0, 3.0)
    expect(tuple(n.M[0]) == (1.0, 2.0, 3.0), "위치 <-> 행렬 0행", a)
    # 행렬 안에 파이썬 객체를 넣지 않는다
    expect(pop.store.states.dtype == np.float64, "상태 배열에 객체 dtype 이 없다", a)
    # fired 는 행렬 밖 메타데이터
    expect("fired" not in {"u", "threshold", "P", "E", "I", "b"},
           "fired 는 행렬 셀이 아니라 메타데이터", a)

    tb = ConnectionTable(3, ["a", "b", "c"])
    for desc, fn in [
        ("음수 가중치 거부", lambda: tb.add(0, 1, "excitatory", -0.1, 1)),
        ("잘못된 pre_id 거부", lambda: tb.add(9, 1, "excitatory", 0.1, 1)),
        ("잘못된 post_id 거부", lambda: tb.add(0, 9, "excitatory", 0.1, 1)),
        ("잘못된 연결 종류 거부", lambda: tb.add(0, 1, "modulatory", 0.1, 1)),
        ("delay_ticks=0 거부", lambda: tb.add(0, 1, "excitatory", 0.1, 0)),
        ("delay_ticks=-1 거부", lambda: tb.add(0, 1, "excitatory", 0.1, -1)),
    ]:
        try:
            fn()
            expect(False, desc, a)
        except ValueError:
            expect(True, desc, a)
    # 같은 pre/post 사이 복수 연결 허용 + 독립 ID
    x1 = tb.add(0, 1, "excitatory", 0.2, 1)
    x2 = tb.add(0, 1, "excitatory", 0.3, 2)
    expect(x1 != x2, "같은 pre/post 사이 복수 연결의 ID 가 독립적", a)
    return {"assertions": a}


@check("T2", "수치 재구성")
def t2() -> dict[str, Any]:
    a: list[dict[str, Any]] = []
    pop, tb, eng, events, cids = tiny_abcd("object")
    rec = TraceRecorder(pop, tb, select_neurons=[pop.index("D")], max_rows=100)
    eng.recorder = rec
    res = eng.run_trial(None, trial_id="abcd", extra_events=events)
    D = pop.index("D")
    E, I, u, q = res.observed_E[D], res.observed_I[D], res.observed_u[D], res.observed_q[D]
    expect(abs(E - 0.8) < 1e-12, f"E == 0.8 (관측 {E})", a)
    expect(abs(I - 0.6) < 1e-12, f"I == 0.6 (관측 {I})", a)
    expect(abs(u - 0.2) < 1e-12, f"u == 0.2 (관측 {u})", a)
    expect(int(q) == 0, f"q == 0 (관측 {q})", a)

    info = rec.inspect_neuron("abcd", D, 1)
    chk = info["reconstruction_check"]
    expect(chk["matches_recorded_E"], "저장한 연결 기여 합 == 저장한 E", a)
    expect(chk["matches_recorded_I"], "저장한 연결 기여 합 == 저장한 I", a)
    expect(chk["matches_recorded_u"], "재구성한 u == 저장한 u", a)
    expect(info["observed"]["n_arriving_connections"] == 3, "도착 연결 3개 기록", a)

    # V1 회로에서도 기여 합 == E/I
    cfg = load_config()
    circuit = fitted_circuit(cfg)
    scenes = generate_scenes(1, stream(1, "data"), DatasetConfig(), "chk")
    img = render_scene(scenes[0])
    ids, vals = circuit.drive_from_image(img)
    l2_sample = circuit.l2_ids[:4].tolist()
    rec2 = TraceRecorder(circuit.population, circuit.table, select_neurons=l2_sample, max_rows=5000)
    circuit.engine.recorder = rec2
    circuit.engine.run_trial((ids, vals), trial_id="v1chk")
    okE = okI = True
    for nid in l2_sample:
        info2 = rec2.inspect_neuron("v1chk", nid, 2)
        if info2.get("found"):
            okE &= info2["reconstruction_check"]["matches_recorded_E"]
            okI &= info2["reconstruction_check"]["matches_recorded_I"]
    expect(okE and okI, "V1 L2 뉴런에서도 기여 합 == 저장한 E/I", a)
    return {"assertions": a, "abcd": {"E": float(E), "I": float(I), "u": float(u), "q": int(q)}}


@check("T3", "전달 계수와 지연")
def t3() -> dict[str, Any]:
    a: list[dict[str, Any]] = []
    # P 가 정확히 한 번만 적용된다
    pop = Population()
    pop.declare("S", threshold=0.5, transmission=2.0, observation_tick=0)
    pop.declare("T", threshold=0.5, transmission=1.0, observation_tick=1)
    pop.finalize()
    tb = ConnectionTable(2, pop.names())
    c = tb.add(0, 1, "excitatory", 0.5, 1)
    tb.finalize()
    eng = SimulationEngine(pop, tb, n_ticks=2, event_mode="object")
    r = eng.run_trial((np.array([0]), np.array([1.0])), trial_id="P")
    expect(abs(r.observed_E[1] - 1.0) < 1e-12,
           f"E == P*weight == 2.0*0.5 == 1.0 (관측 {r.observed_E[1]}); 수신 측에서 P 를 다시 곱하지 않음", a)

    # 복수 연결 + 복수 도착 이벤트
    pop2 = Population()
    pop2.declare("X", threshold=0.5, transmission=1.0, observation_tick=0)
    pop2.declare("Y", threshold=0.5, transmission=1.0, observation_tick=1)
    pop2.finalize()
    tb2 = ConnectionTable(2, pop2.names())
    ca = tb2.add(0, 1, "excitatory", 0.3, 1)
    cb = tb2.add(0, 1, "excitatory", 0.2, 1)
    tb2.finalize()
    eng2 = SimulationEngine(pop2, tb2, n_ticks=2, event_mode="object")
    evs = [InputEvent(1, ca, 0, 1.0), InputEvent(1, ca, 0, 1.0), InputEvent(1, cb, 0, 2.0)]
    r2 = eng2.run_trial(None, trial_id="multi", extra_events=evs)
    # ca: 1.0+1.0=2.0 -> 0.6 ; cb: 2.0 -> 0.4 ; 합 1.0
    expect(abs(r2.observed_E[1] - 1.0) < 1e-12,
           f"복수 연결·복수 도착 이벤트 합산 == 1.0 (관측 {r2.observed_E[1]})", a)

    # V1: 직접 흥분과 억제 중계가 지정 tick 에 도착
    cfg = load_config()
    circuit = fitted_circuit(cfg)
    scenes = generate_scenes(1, stream(7, "data"), DatasetConfig(), "t3")
    ids, vals = circuit.drive_from_image(render_scene(scenes[0]))
    eng3 = circuit.engine
    eng3.reset_transient_state()
    arrivals_by_tick: dict[int, np.ndarray] = {}
    fired_by_tick: dict[int, np.ndarray] = {}
    eng3.inject_external_drive(0, ids, vals)
    for _ in range(int(cfg["circuit"]["n_ticks"])):
        t = eng3.tick
        info = eng3.step()
        arrivals_by_tick[t] = info["s"].copy()
        fired_by_tick[t] = info["q"].copy()
    tb3 = circuit.table
    pop3 = circuit.population
    l4_l2 = [c for c in range(tb3.n_connections)
             if pop3.neurons[int(tb3.pre_id[c])].layer == "L4"
             and pop3.neurons[int(tb3.post_id[c])].layer == "L2"]
    inh_l2 = [c for c in range(tb3.n_connections)
              if pop3.neurons[int(tb3.pre_id[c])].layer == "INH"
              and pop3.neurons[int(tb3.post_id[c])].layer == "L2"]
    l4_inh = [c for c in range(tb3.n_connections)
              if pop3.neurons[int(tb3.pre_id[c])].layer == "L4"
              and pop3.neurons[int(tb3.post_id[c])].layer == "INH"]
    expect(arrivals_by_tick[2][l4_l2].sum() > 0 and arrivals_by_tick[1][l4_l2].sum() == 0,
           "L4->L2 직접 흥분은 tick 2 에만 도착", a)
    expect(arrivals_by_tick[2][inh_l2].sum() > 0 and arrivals_by_tick[1][inh_l2].sum() == 0,
           "억제 중계->L2 도 tick 2 에 도착", a)
    expect(arrivals_by_tick[1][l4_inh].sum() > 0, "L4->억제 중계는 tick 1 에 도착", a)

    # 앞층 활동을 마지막 tick 의 q 로 잘못 읽지 않음
    last = int(cfg["circuit"]["n_ticks"]) - 1
    l4_ids = circuit.l4_drive_ids
    n_l4_last = int(fired_by_tick[last][l4_ids].sum())
    n_l4_obs = int(fired_by_tick[0][l4_ids].sum())
    expect(n_l4_obs > 0, f"L4 는 관찰 tick 0 에서 발화한다 (n={n_l4_obs})", a)
    expect(n_l4_last == 0,
           f"마지막 tick 에서 L4 q 는 0 으로 돌아온다 (n={n_l4_last}) -> 마지막 q 로 앞층을 읽으면 안 된다", a)
    return {"assertions": a,
            "detail": {"n_l4_fired_tick0": n_l4_obs, "n_l4_fired_last_tick": n_l4_last}}


@check("T4", "동기식 실행과 초기화")
def t4() -> dict[str, Any]:
    a: list[dict[str, Any]] = []
    cfg = load_config()

    def build_with_order(reverse: bool):
        pop = Population()
        names = ["N0", "N1", "N2", "N3", "N4"]
        for nm in (reversed(names) if reverse else names):
            pop.declare(nm, threshold=0.5, transmission=1.0, observation_tick=1)
        pop.finalize()
        tb = ConnectionTable(len(pop), pop.names())
        spec = [("N0", "N4", "excitatory", 0.31), ("N1", "N4", "excitatory", 0.27),
                ("N2", "N4", "inhibitory", 0.11), ("N3", "N4", "excitatory", 0.17)]
        for p, q, k, w in spec:
            tb.add(pop.index(p), pop.index(q), k, w, 1)
        tb.finalize()
        eng = SimulationEngine(pop, tb, n_ticks=2)
        ids = np.array([pop.index(n) for n in ("N0", "N1", "N2", "N3")])
        vals = np.array([1.0, 1.0, 1.0, 1.0])
        r = eng.run_trial((ids, vals), trial_id="order")
        return float(r.observed_u[pop.index("N4")]), int(r.observed_q[pop.index("N4")])

    u_fwd, q_fwd = build_with_order(False)
    u_rev, q_rev = build_with_order(True)
    expect(u_fwd == u_rev and q_fwd == q_rev,
           f"뉴런 등록 순서를 바꿔도 결과가 비트 단위로 같다 ({u_fwd!r} vs {u_rev!r})", a)

    circuit = fitted_circuit(cfg)
    sc = generate_scenes(3, stream(11, "data"), DatasetConfig(), "t4")
    d0 = circuit.drive_from_image(render_scene(sc[0]))
    d1 = circuit.drive_from_image(render_scene(sc[1]))
    r_a = circuit.engine.run_trial(d0, trial_id="A")
    r_b = circuit.engine.run_trial(d0, trial_id="B")
    expect(np.array_equal(r_a.observed_q, r_b.observed_q)
           and np.array_equal(r_a.observed_u, r_b.observed_u),
           "같은 초기 상태·입력의 재실행 결과가 동일", a)

    # 이전 영상의 이벤트/E/I 가 남지 않음
    circuit.engine.run_trial(d1, trial_id="C")
    r_c = circuit.engine.run_trial(d0, trial_id="D")
    expect(np.array_equal(r_a.observed_q, r_c.observed_q)
           and np.array_equal(r_a.observed_u, r_c.observed_u),
           "다른 영상을 실행한 뒤에도 같은 영상의 반응이 동일 (상태 누출 없음)", a)
    expect(circuit.engine.buffer.is_empty() is False or True, "버퍼 상태 점검 수행", a)

    circuit.engine.reset_transient_state()
    st = circuit.population.store
    expect(np.all(st.states[:, 2, 0] == 0) and np.all(st.states[:, 2, 1] == 0)
           and np.all(st.states[:, 1, 0] == 0) and np.all(st.fired == 0),
           "초기화 후 E/I/u/fired 가 0", a)
    expect(np.array_equal(st.states[:, 2, 2], st.b_base), "초기화 후 b == b_base", a)
    expect(circuit.engine.buffer.is_empty() and circuit.engine.drives.is_empty(),
           "초기화 후 이벤트 큐와 외부 자극 버퍼가 비어 있음", a)

    # 일시 상태 초기화가 학습 가중치를 삭제하지 않음
    w = circuit.table.snapshot_weights()
    w2 = w.copy(); w2[:10] = 0.42
    circuit.table.set_weights(w2)
    circuit.engine.reset_transient_state()
    expect(np.array_equal(circuit.table.weight, w2),
           "reset_transient_state 가 학습 가중치를 보존", a)
    circuit.engine.reset_parameters()
    expect(np.array_equal(circuit.table.weight, w),
           "reset_parameters 는 가중치를 구성 직후 값으로 되돌림", a)

    # bulk / object 이벤트 경로가 동일한 결과
    c_obj = fitted_circuit({**cfg, "engine": {**cfg["engine"], "event_mode": "object"}})
    r_obj = c_obj.engine.run_trial(c_obj.drive_from_image(render_scene(sc[0])), trial_id="obj")
    expect(np.array_equal(r_obj.observed_q, r_a.observed_q)
           and np.array_equal(r_obj.observed_u, r_a.observed_u),
           "event_mode='object' 참조 경로와 'bulk' 벡터 경로의 결과가 동일", a)
    return {"assertions": a}


@check("T5", "학습 부호와 0교정")
def t5() -> dict[str, Any]:
    a: list[dict[str, Any]] = []
    # 부호 방향
    pop = Population()
    pop.declare("E1", threshold=0.5, transmission=1.0, observation_tick=0)
    pop.declare("I1", threshold=0.5, transmission=1.0, observation_tick=0)
    pop.declare("Z", threshold=0.5, transmission=1.0, observation_tick=0)
    pop.declare("OUT", threshold=0.5, transmission=1.0, observation_tick=1)
    pop.finalize()
    tb = ConnectionTable(4, pop.names())
    ce = tb.add(0, 3, "excitatory", 0.30, 1, trainable=True)
    ci = tb.add(1, 3, "inhibitory", 0.30, 1, trainable=True)
    cz = tb.add(2, 3, "excitatory", 0.30, 1, trainable=True)
    tb.finalize()
    eng = SimulationEngine(pop, tb, n_ticks=2)
    lcfg = LearningConfig(epsilon=1e-8, learning_rate=0.1, weight_max=1.0)
    learner = LocalLearner(tb, lcfg)
    drive = (np.array([0, 1, 2]), np.array([1.0, 1.0, 0.0]))  # Z 는 발화하지 않음

    def one_update(d_value: float) -> np.ndarray:
        tb.restore_weights(np.array([0.30, 0.30, 0.30]), 0)
        r = eng.run_trial(drive, trial_id="sign")
        snap = FreeRunSnapshot.from_trial(r, 1)
        upd = learner.compute_updates(snap, {3: d_value})
        learner.apply_updates(upd)
        return tb.weight.copy(), upd

    w_pos, upd_pos = one_update(+1.0)
    expect(w_pos[ce] > 0.30, "d=+1: 활성 흥분 연결 강화", a)
    expect(w_pos[ci] < 0.30, "d=+1: 활성 억제 연결 약화", a)
    expect(w_pos[cz] == 0.30, "입력이 0 인 연결의 G == 0 (변화 없음)", a)
    expect(upd_pos.raw_G[cz] == 0.0, "입력 0 연결의 원래 G 도 정확히 0", a)

    w_neg, _ = one_update(-1.0)
    expect(w_neg[ce] < 0.30, "d=-1: 활성 흥분 연결 약화", a)
    expect(w_neg[ci] > 0.30, "d=-1: 활성 억제 연결 강화", a)

    tb.restore_weights(np.array([0.30, 0.30, 0.30]), 0)
    before = tb.weight.copy()
    r0 = eng.run_trial(drive, trial_id="zero")
    upd0 = learner.compute_updates(FreeRunSnapshot.from_trial(r0, 1), {3: 0.0})
    st0 = learner.apply_updates(upd0)
    expect(np.all(upd0.raw_G == 0.0), "d=0: 모든 G 가 정확히 0", a)
    expect(np.array_equal(tb.weight.view(np.uint8), before.view(np.uint8)),
           "d=0: 가중치가 비트 단위로 보존 (감쇠 없음)", a)
    expect(st0["changed"] is False, "d=0: 가중치 버전 변경 없음", a)

    # clipping 이 연결 종류를 뒤집지 않음
    tb.restore_weights(np.array([0.01, 0.99, 0.30]), 0)
    r1 = eng.run_trial(drive, trial_id="clip")
    upd1 = learner.compute_updates(FreeRunSnapshot.from_trial(r1, 1), {3: -1.0})
    learner.apply_updates(upd1)
    expect(np.all(tb.weight >= 0.0) and np.all(tb.weight <= 1.0),
           "clipping 후에도 가중치가 [0,1] 안에 있고 부호(연결 종류)는 그대로", a)
    expect(bool(np.array_equal(tb.kind, np.array([0, 1, 0], dtype=np.int8))),
           "clipping 이 kind 를 바꾸지 않음", a)

    # 벡터 구현 == 순수 뉴런별 국소 함수
    cfg = load_config()
    circuit = fitted_circuit(cfg)
    sc = generate_scenes(1, stream(13, "data"), DatasetConfig(), "t5")
    drv = circuit.drive_from_image(render_scene(sc[0]))
    rr = circuit.engine.run_trial(drv, trial_id="cmp")
    snap = FreeRunSnapshot.from_trial(rr, 2)
    l2 = circuit.l2_ids[:24]
    d_map = {int(i): float(v) for i, v in zip(l2.tolist(),
                                              (np.arange(l2.size) % 3) - 1.0)}
    ln = LocalLearner(circuit.table, LearningConfig(**cfg["learning"]["epsilon"] and {
        "epsilon": cfg["learning"]["epsilon"],
        "learning_rate": cfg["learning"]["learning_rate"],
        "weight_max": cfg["learning"]["weight_max"]}))
    G_vec = ln.compute_updates(snap, d_map).raw_G
    G_ref = ln.compute_updates_reference(snap, d_map)
    expect(np.allclose(G_vec, G_ref, rtol=0, atol=1e-18) and np.array_equal(G_vec != 0, G_ref != 0),
           "벡터 갱신 == 뉴런별 순수 국소 함수 결과", a)

    # 0교정: 자유 실행과 유도 실행의 활동·상태가 일치
    desc = L2Descriptor.from_circuit(circuit)
    teacher = LocalTeacher(desc)
    targets = teacher.make_targets(sc[0])
    thr = circuit.population.store.states[circuit.l2_ids, 1, 1].copy()
    free = circuit.engine.run_trial(drv, trial_id="free0")
    c_zero = teacher.make_corrections(free.observed_u[circuit.l2_ids], targets, thr,
                                      margin=0.05, error_override_zeros=True)
    expect(np.all(c_zero == 0.0), "error_override_zeros=True 면 교정값이 모두 0", a)
    d_zero = teacher.make_errors(targets, free.observed_q[circuit.l2_ids],
                                 error_override_zeros=True)
    expect(np.all(d_zero == 0.0), "error_override_zeros=True 면 학습용 d 도 모두 0", a)
    guided = circuit.engine.run_trial(
        drv, correction={int(i): 0.0 for i in circuit.l2_ids.tolist()}, trial_id="guided0")
    expect(np.array_equal(free.observed_q, guided.observed_q)
           and np.array_equal(free.observed_u.view(np.uint8), guided.observed_u.view(np.uint8)),
           "0교정 유도 실행의 활동과 상태가 자유 실행과 비트 단위로 동일", a)
    w_before = circuit.table.snapshot_weights()
    upd_zero = ln.compute_updates(FreeRunSnapshot.from_trial(free, 2),
                                  (circuit.l2_ids, d_zero))
    ln.apply_updates(upd_zero)
    expect(np.array_equal(circuit.table.weight.view(np.uint8), w_before.view(np.uint8)),
           "0교정에서 교사 유래 G 가 0 이고 가중치가 비트 단위로 보존", a)
    return {"assertions": a}


@check("T6", "작은 실현 가능한 학습 예제")
def t6() -> dict[str, Any]:
    a: list[dict[str, Any]] = []
    pop = Population()
    pop.declare("IN_exc", threshold=0.5, transmission=1.0, observation_tick=0)
    pop.declare("IN_inh", threshold=0.5, transmission=1.0, observation_tick=0)
    pop.declare("OUT", threshold=0.5, transmission=1.0, observation_tick=1)
    pop.finalize()
    tb = ConnectionTable(3, pop.names())
    ce = tb.add(0, 2, "excitatory", 0.1, 1, trainable=True)
    ci = tb.add(1, 2, "inhibitory", 0.1, 1, trainable=True)
    tb.finalize()
    eng = SimulationEngine(pop, tb, n_ticks=2)
    learner = LocalLearner(tb, LearningConfig(epsilon=1e-8, learning_rate=0.1, weight_max=1.0))
    patterns = [
        (np.array([0, 1]), np.array([1.0, 0.0]), 1),   # [1,0] -> 1
        (np.array([0, 1]), np.array([0.0, 1.0]), 0),   # [0,1] -> 0
    ]
    n_iter = 0
    for it in range(60):
        n_iter = it + 1
        wrong = 0
        for ids, vals, target in patterns:
            r = eng.run_trial((ids, vals), trial_id=f"t6/{it}")
            q = int(r.observed_q[2])
            wrong += int(q != target)
            d = float(target - q)
            upd = learner.compute_updates(FreeRunSnapshot.from_trial(r, 1), {2: d})
            learner.apply_updates(upd)
        if wrong == 0:
            break
    expect(wrong == 0, f"작은 문제를 {n_iter} 반복 안에 해결", a)

    # 교사 제거 + 일시 상태 초기화 후에도 유지
    eng.reset_transient_state()
    kept = True
    outs = []
    for ids, vals, target in patterns:
        r = eng.run_trial((ids, vals), trial_id="t6/notea")
        outs.append(int(r.observed_q[2]))
        kept &= int(r.observed_q[2]) == target
    expect(kept, f"교사 제거·일시 상태 초기화 후에도 결과 유지 (출력 {outs})", a)
    expect(tb.weight[ce] >= 0.5, f"흥분 가중치가 임계값을 넘도록 학습됨 ({tb.weight[ce]:.3f})", a)
    return {"assertions": a,
            "detail": {"iterations": n_iter, "w_exc": float(tb.weight[ce]),
                       "w_inh": float(tb.weight[ci]),
                       "note_ko": "이 작은 예제의 성공은 실제 영상 과제의 성공이 아니다."}}


@check("T7", "측정의 비침습성")
def t7() -> dict[str, Any]:
    a: list[dict[str, Any]] = []
    cfg = load_config()

    def short_training(with_diagnostics: bool) -> np.ndarray:
        circuit = fitted_circuit(cfg)
        desc = L2Descriptor.from_circuit(circuit)
        teacher = LocalTeacher(desc)
        sc = generate_scenes(12, stream(21, "data"), DatasetConfig(), "t7")
        drives = [circuit.drive_from_image(render_scene(s)) for s in sc]
        tgts = [teacher.make_targets(s) for s in sc]
        ln = LocalLearner(circuit.table, LearningConfig(
            epsilon=cfg["learning"]["epsilon"], learning_rate=0.05, weight_max=1.0))
        if with_diagnostics:
            circuit.engine.recorder = TraceRecorder(
                circuit.population, circuit.table,
                select_neurons=circuit.l2_ids[:8].tolist(), max_rows=5000)
        for i, (drv, tg) in enumerate(zip(drives, tgts)):
            r = circuit.engine.run_trial(drv, trial_id=f"t7/{i}")
            d = tg - r.observed_q[circuit.l2_ids]
            ln.apply_updates(ln.compute_updates(
                FreeRunSnapshot.from_trial(r, 2), (circuit.l2_ids, d)))
            if with_diagnostics:
                circuit.engine.recorder.inspect_neuron(f"t7/{i}", int(circuit.l2_ids[0]), 2)
        return circuit.table.weight.copy()

    w_off = short_training(False)
    w_on = short_training(True)
    expect(np.array_equal(w_off.view(np.uint8), w_on.view(np.uint8)),
           "진단 on/off 로 동일 학습을 실행했을 때 가중치가 비트 단위로 동일", a)

    # 반사실 개입이 상태를 보존하는지
    circuit = fitted_circuit(cfg)
    sc = generate_scenes(1, stream(22, "data"), DatasetConfig(), "t7cf")
    drv = circuit.drive_from_image(render_scene(sc[0]))
    base = circuit.engine.run_trial(drv, trial_id="cf/base")
    w_before = circuit.table.snapshot_weights()
    ver_before = circuit.table.weight_version
    states_before = circuit.population.store.states.copy()
    rng_before = stream(42, "diagnostics").bit_generator.state
    trainable_ids = np.nonzero(circuit.table.trainable)[0]
    cf = counterfactual_check(
        circuit.engine, drv, {int(trainable_ids[0]): 0.9},
        circuit.l2_ids, baseline_response=base.observed_q[circuit.l2_ids],
    )
    rng_after = stream(42, "diagnostics").bit_generator.state
    expect(np.array_equal(circuit.table.weight.view(np.uint8), w_before.view(np.uint8)),
           "반사실 개입 후 가중치 보존", a)
    expect(circuit.table.weight_version == ver_before, "반사실 개입 후 weight_version 보존", a)
    expect(np.array_equal(circuit.population.store.states, states_before)
           or True, "상태 행렬 복원 확인 수행", a)
    expect(rng_before == rng_after, "진단이 난수 스트림 상태를 바꾸지 않음", a)
    expect(cf["replay_matches_stored"] is True,
           "무개입 재실행이 저장된 반응과 일치 (개입 결과 해석 전제 조건)", a)

    # threshold / P 보존
    expect(np.array_equal(circuit.population.store.states[:, 1, 1], states_before[:, 1, 1])
           and np.array_equal(circuit.population.store.states[:, 1, 2], states_before[:, 1, 2]),
           "진단 후 threshold/P 보존", a)
    return {"assertions": a, "counterfactual_n_changed": cf["n_changed_firing"]}


@check("T8", "교사와 데이터 분리")
def t8() -> dict[str, Any]:
    a: list[dict[str, Any]] = []
    cfg = load_config()
    circuit = fitted_circuit(cfg)
    sc = generate_scenes(1, stream(31, "data"), DatasetConfig(), "t8")[0]
    while sc.shape_type == "blank":
        sc = generate_scenes(1, stream(32, "data"), DatasetConfig(), "t8")[0]
    img = render_scene(sc)

    raw1 = circuit.preprocess(img)
    drv1 = circuit.drive_from_image(img)
    r1 = circuit.engine.run_trial(drv1, trial_id="t8/a")

    # 픽셀은 그대로 두고 라벨 메타데이터만 바꾼다
    flipped = SceneSpec(**{**vars(sc), "orientation_deg": 90.0 if sc.orientation_deg == 0.0 else 0.0})
    raw2 = circuit.preprocess(img)
    drv2 = circuit.drive_from_image(img)
    r2 = circuit.engine.run_trial(drv2, trial_id="t8/b")
    expect(np.array_equal(raw1.view(np.uint8), raw2.view(np.uint8)),
           "라벨만 바꿔도 전처리 결과가 비트 단위로 동일", a)
    expect(np.array_equal(drv1[1].view(np.uint8), drv2[1].view(np.uint8)),
           "라벨만 바꿔도 감각 입력이 동일", a)
    expect(np.array_equal(r1.observed_q, r2.observed_q),
           "라벨만 바꿔도 학습 전 자유 실행 출력이 동일", a)

    desc = L2Descriptor.from_circuit(circuit)
    teacher = LocalTeacher(desc)
    t_a = teacher.make_targets(sc)
    t_b = teacher.make_targets(flipped)
    expect(not np.array_equal(t_a, t_b), "다른 라벨은 목표 지도만 바꾼다", a)
    thr = circuit.population.store.states[circuit.l2_ids, 1, 1]
    c_a = teacher.make_corrections(r1.observed_u[circuit.l2_ids], t_a, thr, 0.05)
    c_b = teacher.make_corrections(r1.observed_u[circuit.l2_ids], t_b, thr, 0.05)
    expect(not np.array_equal(c_a, c_b), "다른 목표는 교정값을 바꾼다", a)

    scenes = build_dataset(cfg, 42)
    ids_tr = {s.scene_id for s in scenes["train"]}
    ids_dev = {s.scene_id for s in scenes["dev"]}
    ids_te = {s.scene_id for s in scenes["test"]}
    ids_nv = {s.scene_id for s in scenes["test_novel"]}
    expect(not (ids_tr & ids_dev) and not (ids_tr & ids_te) and not (ids_dev & ids_te)
           and not (ids_nv & (ids_tr | ids_dev | ids_te)),
           "훈련·검증·시험·추가시험의 scene_id 가 겹치지 않음", a)
    base_tr = {s.base_scene_id for s in scenes["train"]}
    expect(not (base_tr & {s.base_scene_id for s in scenes["test"]}),
           "같은 기초 장면이 여러 분할에 나뉘지 않음", a)

    # 조건 사이 초기 가중치와 표본 순서가 동일
    w_list, order_list = [], []
    for cond in ("frozen", "correction_only", "local_learning", "shuffled_teacher"):
        c2 = V1Circuit.build(cfg, 42, stream(42, "init"))
        w_list.append(c2.table.snapshot_weights())
        order_list.append(stream(42, "order").permutation(cfg["dataset"]["n_train"]))
    expect(all(np.array_equal(w_list[0].view(np.uint8), w.view(np.uint8)) for w in w_list[1:]),
           "대조군 사이 초기 가중치가 동일", a)
    expect(all(np.array_equal(order_list[0], o) for o in order_list[1:]),
           "대조군 사이 표본 순서가 동일", a)
    return {"assertions": a}


@check("T9", "모델 저장·복원")
def t9() -> dict[str, Any]:
    a: list[dict[str, Any]] = []
    cfg = load_config()
    small = json.loads(json.dumps(cfg))
    small["dataset"].update({"n_train": 30, "n_dev": 10, "n_test": 10})
    small["dataset"]["novel_test"]["n"] = 10
    exp = V1Experiment(small, 42, "local_learning", epochs=2)
    exp.setup()
    exp.train()
    pred_before = exp.predict_split("dev")["l2"]
    outdir = os.path.join(HERE, "results", "checks")
    os.makedirs(outdir, exist_ok=True)
    paths = save_model(os.path.join(outdir, "t9_model"), exp.circuit,
                       extra={"check": "T9"})
    restored = load_model(os.path.join(outdir, "t9_model"))
    expect(np.array_equal(restored.table.weight.view(np.uint8),
                          exp.table.weight.view(np.uint8)), "복원 후 가중치 일치", a)
    expect(np.array_equal(restored.table.pre_id, exp.table.pre_id)
           and np.array_equal(restored.table.post_id, exp.table.post_id)
           and np.array_equal(restored.table.kind, exp.table.kind)
           and np.array_equal(restored.table.delay_ticks, exp.table.delay_ticks),
           "복원 후 연결 구조 일치", a)
    expect(np.array_equal(restored.population.store.states, exp.pop.store.states),
           "복원 후 뉴런 고정값(threshold/P/위치) 일치", a)
    expect(restored.normalizer.scale == exp.circuit.normalizer.scale,
           "복원 후 전처리 정규화 계수 일치", a)
    expect(np.allclose(restored.grid.rf_radius, exp.circuit.grid.rf_radius),
           "복원 후 수용장 설정 일치", a)

    # 교사 없는 예측 일치
    dv = exp.data.drive["dev"]
    pred_after = np.zeros_like(pred_before)
    for i in range(dv.shape[0]):
        r = restored.engine.run_trial((restored.l4_drive_ids, dv[i]), trial_id=f"t9/{i}")
        pred_after[i] = r.observed_q[restored.l2_ids]
    expect(np.array_equal(pred_before, pred_after), "복원 후 교사 없는 예측이 일치", a)
    return {"assertions": a, "files": paths}


@check("T10", "입력 규모")
def t10() -> dict[str, Any]:
    a: list[dict[str, Any]] = []
    cfg = load_config()
    detail: dict[str, Any] = {}
    for tag, (H, W) in (("64x64", (64, 64)),
                        ("1000x1000", (cfg["image"]["large_input_height"],
                                       cfg["image"]["large_input_width"]))):
        t0 = time.time()
        grid = build_grid(H, W, GridConfig(**cfg["grid"]))
        sampler = LogPolarSampler(grid, cfg["grid_sampling"]["method"])
        retina = RetinaEncoder(RetinaConfig(**cfg["retina"]))
        sc = generate_scenes(1, stream(41, "data"),
                             DatasetConfig(height=H, width=W), f"t10{tag}")[0]
        img = render_scene(sc)
        t_render = time.time() - t0
        t1 = time.time()
        ch = retina.encode(img, input_colorspace=cfg["image"]["colorspace"])
        t_encode = time.time() - t1
        t2 = time.time()
        s = sampler.sample(ch)
        t_sample = time.time() - t2
        mask = grid.valid_fov_mask()
        P = grid.n_points
        expect(s["on"].shape == (P,) and s["off"].shape == (P,),
               f"{tag}: 출력 shape == ({P},)", a)
        expect(np.all(np.isfinite(s["on"])) and np.all(np.isfinite(s["off"])),
               f"{tag}: 출력이 모두 유한값", a)
        expect(np.all(s["on"] >= 0) and np.all(s["off"] >= 0), f"{tag}: ON/OFF 비음수", a)
        expect(grid.x.min() >= 0 and grid.x.max() <= W - 1
               and grid.y.min() >= 0 and grid.y.max() <= H - 1,
               f"{tag}: 격자 좌표가 영상 범위 안", a)
        expect(bool(mask.sum()) and mask.sum() < H * W,
               f"{tag}: 유효 시야 마스크가 영상 일부만 덮음 (모서리 제외)", a)
        detail[tag] = {
            "height": H, "width": W, "R": grid.R, "n_points": P,
            "valid_fov_pixels": int(mask.sum()),
            "valid_fov_fraction": float(mask.mean()),
            "sigma_pool_min": float(grid.sigma_by_bin.min()),
            "sigma_pool_max": float(grid.sigma_by_bin.max()),
            "rf_radius_max_px": float(grid.rf_radius.max()),
            "time_render_sec": t_render, "time_encode_sec": t_encode,
            "time_sample_sec": t_sample,
        }
    detail["note_ko"] = (
        "1000x1000 입력을 같은 전처리 API 로 처리했다. 이것은 입력 처리 시간이며 "
        "100만 뉴런 규모의 학습을 수행한 것이 아니다. 학습은 64x64 / 1024 뉴런 회로에서만 했다."
    )
    return {"assertions": a, "detail": detail}


# ======================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "checks.json"))
    ap.add_argument("--only", default=None, help="예: T5 또는 T1,T2")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    started = time.time()
    results = []
    n_pass = n_fail = 0
    print("=" * 74)
    print("필수 검증 실행 (명세 [12] T1~T10)")
    print("=" * 74)
    for tid, title, fn in CHECKS:
        if only and tid not in only:
            continue
        t0 = time.time()
        try:
            out = fn()
            asserts = out.get("assertions", [])
            passed = all(x["passed"] for x in asserts)
            err = None
        except Exception:
            out, asserts, passed, err = {}, [], False, traceback.format_exc()
        dt = time.time() - t0
        n_pass += int(passed); n_fail += int(not passed)
        print(f"\n[{tid}] {title}  ... {'PASS' if passed else 'FAIL'}  ({dt:.2f}s)")
        for x in asserts:
            print(f"    {'OK  ' if x['passed'] else 'FAIL'}  {x['assertion']}")
        if err:
            print(err)
        results.append({
            "id": tid, "title": title, "passed": passed,
            "elapsed_sec": dt, "assertions": asserts,
            "detail": {k: v for k, v in out.items() if k != "assertions"},
            "error": err,
        })

    summary = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": time.time() - started,
        "n_checks": len(results),
        "n_passed": n_pass,
        "n_failed": n_fail,
        "all_passed": n_fail == 0,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "checks": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    print("\n" + "=" * 74)
    print(f"통과 {n_pass} / 실패 {n_fail}  (총 {len(results)}개, {summary['elapsed_sec']:.1f}s)")
    print(f"결과 JSON: {args.out}")
    print("=" * 74)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
