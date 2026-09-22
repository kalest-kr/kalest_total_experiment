"""두 동작 모드와 시간 루프 테스트 (명세 4, 9절)."""

from __future__ import annotations

import numpy as np

from cortex.dynamics import ExternalDrive
from cortex.validation import _tiny_cfg, _tiny_engine, _tiny_network


def _force(eng, nid, step):
    eng.set_external_drive(ExternalDrive(
        np.array([nid], dtype=np.int64),
        forced_spike_steps={step: np.array([nid], dtype=np.int64)}))


def test_sum_threshold_hand_calculation(minimal_cfg):
    cfg = _tiny_cfg(minimal_cfg, "sum_threshold")
    pop, table, ids = _tiny_network(cfg, exc_w=0.7, inh_w=0.3)
    eng = _tiny_engine(cfg, pop, table, ids)
    pop.arrays.threshold[2] = 0.35
    both = np.array([0, 1], dtype=np.int64)
    eng.set_external_drive(ExternalDrive(both, forced_spike_steps={0: both}))
    eng.step()
    eng.set_external_drive(None)
    eng.step()
    assert abs(float(pop.arrays.last_decision_value[2]) - 0.4) < 1e-12
    assert int(pop.arrays.fired[2]) == 1


def test_sum_threshold_zero_input_boundary(minimal_cfg):
    cfg = _tiny_cfg(minimal_cfg, "sum_threshold")
    pop, table, ids = _tiny_network(cfg)
    eng = _tiny_engine(cfg, pop, table, ids)
    pop.arrays.threshold[:] = 0.0
    eng.set_external_drive(None)
    eng.step()
    assert int(pop.arrays.fired[2]) == 1, "theta<=0 이면 영 입력에서도 발화한다"


def test_conductance_leak_to_EL(minimal_cfg):
    cfg = _tiny_cfg(minimal_cfg, "conductance_lif")
    pop, table, ids = _tiny_network(cfg)
    eng = _tiny_engine(cfg, pop, table, ids)
    pop.arrays.V_mV[2, 0] = -55.0
    eng.set_external_drive(None)
    for _ in range(200):
        eng.step()
    assert abs(float(pop.arrays.V_mV[2, 0]) + 70.0) < 0.2


def test_inhibitory_delta_g_is_non_negative(minimal_cfg):
    cfg = _tiny_cfg(minimal_cfg, "conductance_lif")
    pop, table, ids = _tiny_network(cfg, inh_w=2.0)
    eng = _tiny_engine(cfg, pop, table, ids)
    _force(eng, 1, 0)
    eng.step()
    eng.set_external_drive(None)
    eng.step()
    assert np.all(pop.arrays.g_nS >= 0.0), "억제성 Δg 를 음수로 만들지 않는다"


def test_no_double_application(minimal_cfg):
    cfg = _tiny_cfg(minimal_cfg, "conductance_lif")
    pop, table, ids = _tiny_network(cfg, exc_w=2.0)
    eng = _tiny_engine(cfg, pop, table, ids)
    _force(eng, 0, 0)
    eng.step()
    eng.set_external_drive(None)
    eng.step()
    g1 = float(pop.arrays.g_nS[2].sum())
    eng.step()
    g2 = float(pop.arrays.g_nS[2].sum())
    assert g2 < g1, "같은 사건이 두 번 반영되지 않는다 (감쇠만 남는다)"


def test_step_order_increments_time(minimal_cfg):
    cfg = _tiny_cfg(minimal_cfg, "conductance_lif")
    pop, table, ids = _tiny_network(cfg)
    eng = _tiny_engine(cfg, pop, table, ids)
    eng.set_external_drive(None)
    t0, s0 = eng.time_ms, eng.step_index
    eng.step()
    assert eng.step_index == s0 + 1
    assert abs(eng.time_ms - (t0 + cfg["engine"]["dt_ms"])) < 1e-12


def test_checkpoint_roundtrip(minimal_cfg):
    cfg = _tiny_cfg(minimal_cfg, "conductance_lif")
    pop, table, ids = _tiny_network(cfg, exc_w=3.0)
    eng = _tiny_engine(cfg, pop, table, ids)
    for s in range(8):
        if s % 3 == 0:
            _force(eng, 0, s)
        else:
            eng.set_external_drive(None)
        eng.step()
    st = eng.state_dict()

    pop2, table2, ids2 = _tiny_network(cfg, exc_w=3.0)
    eng2 = _tiny_engine(cfg, pop2, table2, ids2)
    eng2.load_state_dict(st)
    assert np.array_equal(eng2.a.V_mV, eng.a.V_mV)
    assert eng2.event_counter.value == eng.event_counter.value


def test_read_only_snapshot_does_not_alias(minimal_cfg):
    cfg = _tiny_cfg(minimal_cfg, "conductance_lif")
    pop, table, ids = _tiny_network(cfg)
    eng = _tiny_engine(cfg, pop, table, ids)
    snap = eng.read_only_snapshot()
    pop.arrays.V_mV[0, 0] = 999.0
    assert snap["V_mV"][0, 0] != 999.0
