"""이벤트 로그와 지연 양자화 테스트 (명세 3절)."""

from __future__ import annotations

import numpy as np
import pytest

from cortex.events import CapacityExceeded, DelayQueue, EventLog, quantize_delay


def test_min_delay_enforced():
    q = quantize_delay(np.array([0.0, 0.1]), 1.0, min_steps=1)
    assert np.all(q >= 1)


def test_zero_min_delay_rejected():
    with pytest.raises(ValueError):
        quantize_delay(np.array([1.0]), 1.0, min_steps=0)


def test_ceil_vs_round():
    assert int(quantize_delay(np.array([1.2]), 1.0, 1, "ceil")[0]) == 2
    assert int(quantize_delay(np.array([1.2]), 1.0, 1, "round")[0]) == 1


def test_queue_pops_once():
    q = DelayQueue()
    q.schedule(3, {"synapse_id": np.array([1, 2])})
    got = q.pop(3)
    assert got["synapse_id"].tolist() == [1, 2]
    assert q.pop(3) == {}


def test_capacity_exceeded_raises():
    log = EventLog(4, mode="full", max_events=2)
    kw = dict(parent_spike_id=np.zeros(3, np.int64), sample_id=0, episode_id=0,
              src_id=np.zeros(3, np.int64), dst_id=np.arange(3),
              synapse_id=np.arange(3), emit_time_ms=np.zeros(3),
              arrival_time_ms=np.zeros(3), arrival_step=np.zeros(3, np.int64),
              weight_snapshot=np.ones(3), source_gain_snapshot=np.ones(3),
              target_compartment=np.zeros(3, np.int8),
              event_type="synaptic_arrival", amount=np.ones(3),
              amount_unit="nS")
    with pytest.raises(CapacityExceeded):
        log.append(event_id=np.arange(3), **kw)


def test_selected_mode_counts_skipped():
    log = EventLog(4, mode="selected", selected_neurons=[0],
                   selection_criterion="테스트")
    n = log.append(event_id=np.arange(3), parent_spike_id=np.zeros(3, np.int64),
                   sample_id=0, episode_id=0, src_id=np.zeros(3, np.int64),
                   dst_id=np.array([0, 1, 2]), synapse_id=np.arange(3),
                   emit_time_ms=np.zeros(3), arrival_time_ms=np.zeros(3),
                   arrival_step=np.zeros(3, np.int64), weight_snapshot=np.ones(3),
                   source_gain_snapshot=np.ones(3),
                   target_compartment=np.zeros(3, np.int8),
                   event_type="synaptic_arrival", amount=np.ones(3),
                   amount_unit="nS")
    assert n == 1
    assert log.n_skipped_by_mode == 2, "버린 건수를 조용히 숨기지 않는다"


def test_query_neuron_index():
    log = EventLog(3, mode="full")
    log.append(event_id=np.arange(2), parent_spike_id=np.zeros(2, np.int64),
               sample_id=0, episode_id=0, src_id=np.zeros(2, np.int64),
               dst_id=np.array([1, 1]), synapse_id=np.arange(2),
               emit_time_ms=np.zeros(2), arrival_time_ms=np.array([1.0, 5.0]),
               arrival_step=np.zeros(2, np.int64), weight_snapshot=np.ones(2),
               source_gain_snapshot=np.ones(2),
               target_compartment=np.zeros(2, np.int8),
               event_type="synaptic_arrival", amount=np.ones(2), amount_unit="nS")
    assert log.count_for_neuron(1) == 2
    rows = log.query_neuron(1, 0.0, 2.0)
    assert rows["arrival_time_ms"].tolist() == [1.0]
