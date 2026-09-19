"""3x3 기록 인터페이스 테스트 (명세 2절)."""

from __future__ import annotations

import numpy as np

from cortex.records import (
    DynamicStateView,
    InputLogView,
    MetadataView,
    OutgoingConnectionsView,
)


def test_matrix_layout(minimal_model):
    rec = minimal_model.anat.population.record(0)
    m = rec.as_matrix()
    assert m.shape == (3, 3)
    assert m.dtype == object
    assert isinstance(m[1, 0], OutgoingConnectionsView), "[1][0] 은 출력 연결 목록"
    assert isinstance(m[2, 0], InputLogView)
    assert isinstance(m[2, 1], DynamicStateView)
    assert isinstance(m[2, 2], MetadataView)


def test_no_input_sum_in_cell_1_0(minimal_model):
    """[1][0] 에 입력 총합이 들어가지 않는다."""
    rec = minimal_model.anat.population.record(0)
    cell = rec.as_matrix()[1, 0]
    assert not isinstance(cell, (int, float, np.floating))


def test_view_and_array_share_state(minimal_model):
    a = minimal_model.anat.population.arrays
    rec = minimal_model.anat.population.record(3)
    rec.threshold = 1.2345
    assert a.threshold[3] == 1.2345
    a.threshold[3] = -7.5
    assert rec.threshold == -7.5


def test_outgoing_ids_are_valid(minimal_model):
    t = minimal_model.table
    for i in (0, 1, 2):
        sids = minimal_model.anat.population.record(i).outgoing.synapse_ids
        if sids.size:
            assert sids.min() >= 0 and sids.max() < t.n_synapses
            assert np.all(t.src_id[sids] == i)


def test_metadata_has_separate_spaces(minimal_model):
    md = minimal_model.anat.population.record(0).metadata.to_dict()
    assert "cortical_xyz_mm" in md
    assert md["receptive_field_parameters"]["space"] == "visual_field_deg"


def test_state_arrays_are_float64(minimal_model):
    a = minimal_model.anat.population.arrays
    assert a.V_mV.dtype == np.float64
    assert a.g_nS.dtype == np.float64
    assert a.position_mm.dtype == np.float64
