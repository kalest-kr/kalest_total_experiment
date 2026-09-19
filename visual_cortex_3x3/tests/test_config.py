"""설정 스키마·병합·검증 테스트."""

from __future__ import annotations

import json

import pytest

from cortex.config import (
    ConfigError,
    config_hash,
    count_neurons,
    derived_retina_neuron_count,
    estimate_sizes,
    resolve,
)


def test_unknown_key_is_rejected(minimal_cfg):
    raw = {"engine": {"dt_mss": 1.0}}
    with pytest.raises(ConfigError) as e:
        resolve(raw)
    assert "dt_mss" in str(e.value)


def test_free_form_layer_names_allowed():
    cfg = resolve({
        "cell_types": {"x": {"dale": "excitatory", "compartments": ["soma"]}},
        "anatomy": {"areas": {"A": {
            "kind": "cortex",
            "neurons_per_layer": {"L4": 4},
            "cell_type_fractions": {"L4": {"x": 1.0}}}}},
    })
    assert cfg["anatomy"]["areas"]["A"]["neurons_per_layer"]["L4"] == 4


def test_cell_type_fraction_must_sum_to_one():
    with pytest.raises(ConfigError):
        resolve({
            "cell_types": {"x": {"dale": "excitatory", "compartments": ["soma"]},
                           "y": {"dale": "inhibitory", "compartments": ["soma"]}},
            "anatomy": {"areas": {"A": {
                "kind": "cortex", "neurons_per_layer": {"L4": 4},
                "cell_type_fractions": {"L4": {"x": 0.5, "y": 0.2}}}}},
        })


def test_min_delay_must_be_at_least_one():
    with pytest.raises(ConfigError):
        resolve({"engine": {"min_delay_steps": 0}})


def test_non_full_recording_requires_criterion():
    with pytest.raises(ConfigError):
        resolve({"recording": {"mode": "selected", "selection_criterion": ""}})


def test_rao_sigma_must_be_positive():
    with pytest.raises(ConfigError):
        resolve({"learning": {"rao": {"sigma": 0.0}}})


def test_config_hash_is_stable(minimal_cfg):
    a = config_hash(minimal_cfg)
    b = config_hash(json.loads(json.dumps(minimal_cfg)))
    assert a == b


def test_retina_neurons_are_derived(minimal_cfg):
    n, info = derived_retina_neuron_count(minimal_cfg)
    assert n == info["n_channels"] * info["n_samples"]
    assert count_neurons(minimal_cfg) > n


def test_estimate_does_not_predict_runtime(minimal_cfg):
    est = estimate_sizes(minimal_cfg).to_dict()
    assert est["runtime_seconds_estimated"] is None
    assert est["n_steps"] > 0
