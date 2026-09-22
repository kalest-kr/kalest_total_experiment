"""가소성과 Rao 참조 모델 테스트 (명세 8절)."""

from __future__ import annotations

import numpy as np
import pytest

from cortex.predictive_coding import RaoModel
from cortex.synapses import SynapseTable


def test_rao_rejects_non_positive_sigma():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        RaoModel(1, 4, 3, 2, sigma=0.0, sigma_td=1.0, alpha=0.0, lam=0.0, rng=rng)
    with pytest.raises(ValueError):
        RaoModel(1, 4, 3, 2, sigma=1.0, sigma_td=-1.0, alpha=0.0, lam=0.0, rng=rng)


def test_rao_gradients_match_finite_difference():
    rng = np.random.default_rng(1)
    m = RaoModel(2, 5, 4, 3, sigma=1.0, sigma_td=2.0, alpha=0.05, lam=0.001, rng=rng)
    I = rng.normal(size=(2, 5))
    r1 = rng.normal(size=(2, 4)) * 0.3
    r2 = rng.normal(size=3) * 0.3
    res = m.finite_difference_check(I, r1, r2, eps=1e-6, n_probe=6, rng=rng)
    assert res["max_relative_residual"] < 1e-4


def test_rao_settle_decreases_energy():
    rng = np.random.default_rng(2)
    m = RaoModel(1, 6, 4, 3, sigma=1.0, sigma_td=2.0, alpha=0.05, lam=0.001, rng=rng)
    I = rng.normal(size=(1, 6))
    st = m.settle(I, steps=60, r_step=0.05)
    assert st.energy_trace[-1] <= st.energy_trace[0]
    assert "image_reconstruction_error_l2" in st.errors
    assert "interarea_representation_error_l2" in st.errors


def test_rao_settle_requires_frozen_U():
    rng = np.random.default_rng(3)
    m = RaoModel(1, 4, 3, 2, sigma=1.0, sigma_td=1.0, alpha=0.0, lam=0.0, rng=rng)
    with pytest.raises(ValueError):
        m.settle(np.zeros((1, 4)), steps=1, r_step=0.01, freeze_U=False)


def test_negative_weight_rejected():
    t = SynapseTable(2, "nS")
    with pytest.raises(ValueError):
        t.add_block(src_id=np.array([0]), dst_id=np.array([1]),
                    src_area=np.array([0]), dst_area=np.array([0]),
                    target_layer=np.array([0]), target_compartment=0,
                    receptor_type=0, weight=np.array([-1.0]),
                    base_delay_ms=np.array([1.0]),
                    effective_delay_steps=np.array([1]), plasticity_rule=0,
                    src_dale_sign=np.array([1]), rule_index=0)


def test_clip_preserves_sign(minimal_model):
    t = minimal_model.table
    before = t.src_dale_sign.copy()
    t.weight[:] = np.linspace(-5, 20, t.n_synapses)
    t.clip_weights(0.0, 6.0)
    assert np.all(t.weight >= 0.0) and np.all(t.weight <= 6.0)
    assert np.array_equal(t.src_dale_sign, before)


def test_dale_no_violations(minimal_model):
    a = minimal_model.anat.population.arrays
    assert minimal_model.table.dale_violations(a.dale_sign).size == 0
