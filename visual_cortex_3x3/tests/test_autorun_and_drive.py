"""자동 실행(run-all)과 구동/전달 진단 테스트.

이 파일의 테스트는 **회귀 방지용**이다. 각 테스트는 실제로 고친 결함 하나에
대응한다. 시뮬레이션을 돌리더라도 최소 규모로만 돌린다.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from cortex.autorun import ALL_STAGES, prepare_config, resolve_backend
from cortex.config import drive_headroom_warnings
from cortex.records import NeuronArrays, NeuronPopulation
from cortex.ids import IdSpace


# ----------------------------------------------------------------------
# 3x3 기록 인터페이스: frozen dataclass 가 property setter 를 막던 결함
# ----------------------------------------------------------------------
def test_record_property_setters_work_under_frozen_dataclass():
    pop = NeuronPopulation(NeuronArrays(3, 3), IdSpace())
    rec = pop.record(0)
    rec.threshold = 1.25
    rec.output_gain_P = 2.5
    assert pop.arrays.threshold[0] == 1.25
    assert pop.arrays.output_gain_P[0] == 2.5


@pytest.mark.parametrize("name", ["population", "neuron_id", "not_a_field"])
def test_record_identity_fields_stay_frozen(name):
    pop = NeuronPopulation(NeuronArrays(3, 3), IdSpace())
    rec = pop.record(0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(rec, name, 1)


# ----------------------------------------------------------------------
# 망막 정규화
# ----------------------------------------------------------------------
def test_normalize_keeps_shape_for_grid_samples(minimal_model):
    """normalize 를 (C,S) 격자 샘플에 써도 축이 늘지 않는다."""
    enc = minimal_model.encoder
    img = np.zeros((32, 32), dtype=np.float64)
    img[10:20, 10:20] = 1.0
    enc.fit_normalization([img], source="test", sampler=minimal_model.sampler)
    values, _ = minimal_model.sampler.sample(enc.encode(img).channels)
    assert enc.normalize(values).shape == values.shape
    assert enc.normalize(enc.encode(img).channels).shape == enc.encode(img).channels.shape


def test_normalize_rejects_wrong_channel_axis(minimal_model):
    enc = minimal_model.encoder
    img = np.zeros((32, 32), dtype=np.float64)
    img[8:16, 8:16] = 1.0
    enc.fit_normalization([img], source="test", sampler=minimal_model.sampler)
    with pytest.raises(ValueError):
        enc.normalize(np.zeros((enc.scale.size + 1, 5)))


def test_fit_normalization_ignores_zeros_instead_of_returning_nan(minimal_model):
    """0 화소가 있어도 스케일이 1.0 로 무너지지 않는다 (nanpercentile)."""
    enc = minimal_model.encoder
    img = np.zeros((32, 32), dtype=np.float64)
    img[12:20, 12:20] = 1.0                       # 대부분이 0 인 영상
    info = enc.fit_normalization([img], source="test", sampler=minimal_model.sampler)
    assert info["fit_representation"] == "grid_samples"
    assert enc.scale.max() > 0.0
    assert not np.allclose(enc.scale, 1.0), "모든 채널이 degenerate 로 떨어지면 안 된다"


def test_fit_normalization_floors_empty_channels(minimal_model):
    """신호가 없는 채널이 수치 잔차만으로 최대 세기까지 증폭되지 않는다."""
    enc = minimal_model.encoder
    img = np.zeros((32, 32), dtype=np.float64)
    img[12:20, 12:20] = 1.0                       # 회색조 -> 색 대립 채널은 0
    values, _ = minimal_model.sampler.sample(enc.encode(img).channels)

    info = enc.fit_normalization([img], source="test", sampler=minimal_model.sampler)
    assert info["floored_channels"], "색 대립 채널은 바닥에 걸려야 한다"
    assert enc.scale.min() >= info["min_scale_ratio"] * enc.scale.max()
    with_floor = enc.normalize(values)

    enc.fit_normalization([img], source="test", sampler=minimal_model.sampler,
                          min_scale_ratio=0.0)
    without_floor = enc.normalize(values)

    ratio = info["min_scale_ratio"]
    for c in info["floored_channels"]:
        assert without_floor[c].max() > 0.9, "바닥이 없으면 잔차가 최대까지 포화한다"
        assert with_floor[c].max() < 0.9, "바닥을 두면 포화하지 않는다"
        assert with_floor[c].max() < without_floor[c].max()
        # 바닥은 증폭 상한을 정한다: 가장 센 채널 대비 신호 비율에 비례한 값만 남는다.
        share = values[c].max() / values.max()
        assert with_floor[c].max() <= share / ratio + 1e-9


# ----------------------------------------------------------------------
# 구동 여유 진단
# ----------------------------------------------------------------------
def test_drive_headroom_flags_unreachable_threshold(minimal_cfg):
    cfg = copy.deepcopy(minimal_cfg)
    assert drive_headroom_warnings(cfg) == [], "배포 설정은 임계에 닿아야 한다"
    cfg["retina"]["drive"]["sum_mode_scale"] = 1.0
    warnings = drive_headroom_warnings(cfg)
    assert warnings and "retinal_ganglion" in warnings[0]


def test_drive_headroom_checks_current_in_lif_mode(v1_cfg):
    cfg = copy.deepcopy(v1_cfg)
    cfg["retina"]["drive"]["current_per_hz_pA"] = 1e-6
    warnings = drive_headroom_warnings(cfg)
    assert warnings and "pA" in warnings[0]


# ----------------------------------------------------------------------
# 층간 배선: 표면 좌표 반경
# ----------------------------------------------------------------------
def test_translaminar_rules_make_synapses(minimal_model, minimal_cfg):
    """이름만 존재하는 경로가 없어야 한다 (검증 9 와 같은 조건)."""
    empty = [r["name"] for r in minimal_model.wiring_report.per_rule if r["n"] == 0]
    assert empty == []


def test_surface_radius_reaches_across_layers(minimal_cfg):
    """radius_space=surface 는 깊이 차이를 무시하고 같은 기둥 안을 잇는다."""
    from cortex import rng as rng_mod
    from cortex.runner import build_model

    cfg = copy.deepcopy(minimal_cfg)
    target = "V1_L2L3->L5"
    for r in cfg["wiring"]["rules"]:
        if r["name"] == target:
            assert r["radius_space"] == "surface"
            r["radius_space"] = "cortical_3d"
    model = build_model(cfg, rng_mod.from_config(cfg))
    n_3d = next(r["n"] for r in model.wiring_report.per_rule if r["name"] == target)
    model2 = build_model(minimal_cfg, rng_mod.from_config(minimal_cfg))
    n_surface = next(r["n"] for r in model2.wiring_report.per_rule if r["name"] == target)
    assert n_3d == 0 and n_surface > 0


# ----------------------------------------------------------------------
# 전달 여유와 침묵 영역
# ----------------------------------------------------------------------
def test_transmission_headroom_reports_each_excitatory_rule(v1_cfg):
    from cortex import rng as rng_mod
    from cortex.runner import build_model, transmission_headroom

    model = build_model(v1_cfg, rng_mod.from_config(v1_cfg))
    head = transmission_headroom(v1_cfg, model)
    assert head["applicable"] and head["rules"]
    retina_rows = [r for r in head["rules"] if r["presyn_area"] == "Retina"]
    assert retina_rows, "망막 출력 규칙이 보고에 있어야 한다"
    # 망막 상한은 불응기가 아니라 외부 구동이 정한다.
    drive = v1_cfg["retina"]["drive"]
    cap = drive["baseline_rate_hz"] + drive["gain"] * drive["max_rate_hz"]
    assert all(r["presyn_rate_max_hz"] <= cap + 1e-6 for r in retina_rows)


def test_transmission_headroom_not_applicable_in_sum_mode(minimal_cfg):
    from cortex import rng as rng_mod
    from cortex.runner import build_model, transmission_headroom

    model = build_model(minimal_cfg, rng_mod.from_config(minimal_cfg))
    assert transmission_headroom(minimal_cfg, model)["applicable"] is False


def test_silent_area_report_names_silent_areas():
    from cortex.runner import SampleResult, silent_area_report

    def mk(spikes):
        return SampleResult(sample_index=0, stimulus_id="s", label="l",
                            n_spikes_total=sum(spikes.values()), spikes_by_area=spikes,
                            mean_rate_hz_by_area={}, first_spike_step_by_area={},
                            silent_fraction_by_area={}, n_steps=1, n_events=0)

    rep = silent_area_report([mk({"Retina": 10, "V1": 0}), mk({"Retina": 3, "V1": 0})])
    assert rep["silent_areas"] == ["V1"]
    assert rep["active_areas"] == ["Retina"]
    assert rep["all_silent"] is False


# ----------------------------------------------------------------------
# 자동 실행 설정 처리
# ----------------------------------------------------------------------
def test_resolve_backend_reports_downgrade(minimal_cfg):
    cfg = copy.deepcopy(minimal_cfg)
    cfg["recording"]["backend"] = "hdf5"
    chosen, note = resolve_backend(cfg, "auto")
    assert chosen in ("hdf5", "npz")
    assert (chosen == "npz") == bool(note), "바꿨으면 반드시 알려야 한다"
    assert resolve_backend(cfg, "npz") == ("npz", "")


def test_prepare_config_records_every_override(project_root):
    cfg, notes = prepare_config(str(project_root / "configs" / "minimal.json"),
                                backend="npz", max_stimuli=3, duration_ms=25.0,
                                seed=7)
    assert cfg["experiment"]["max_stimuli"] == 3
    assert cfg["engine"]["duration_ms"] == 25.0
    assert cfg["seeds"]["master"] == 7
    assert len(notes) >= 3, "덮어쓴 항목은 조용히 넘어가지 않는다"


def test_stage_names_are_unique_and_ordered():
    assert len(set(ALL_STAGES)) == len(ALL_STAGES)
    assert ALL_STAGES[0] == "config_check"
    assert ALL_STAGES.index("validate") < ALL_STAGES.index("simulate")
    assert ALL_STAGES.index("simulate") < ALL_STAGES.index("report")
