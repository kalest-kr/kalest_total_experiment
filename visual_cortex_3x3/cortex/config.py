"""config.py -- 설정 스키마, 기본값, 병합과 검증.

설계 선택 (근거)
----------------
설정은 **중첩 dict** 로 다루고, :data:`DEFAULTS` 에 모든 스칼라 기본값을 선언한다.
사용자 JSON 을 DEFAULTS 위에 깊은 병합(deep merge)하고, **DEFAULTS 에 없는 키는
오류**로 처리한다. 오타가 조용히 무시되어 "설정했다고 생각했지만 반영되지 않는"
상황을 막기 위해서다. 영역/배선처럼 개수가 가변인 부분은 항목 하나하나에
템플릿 기본값을 적용한다 (``_AREA_TEMPLATE``, ``_WIRING_TEMPLATE``).

해석된 전체 설정(resolved config)은 manifest.json 에 그대로 저장된다. 따라서
실행 기록만 보고도 어떤 값이 실제로 쓰였는지 알 수 있다.

단위는 키 이름에 붙인다 (``dt_ms``, ``C_pF``, ``gL_nS``, ``E_rev_mV``,
``sigma_deg``, ``extent_mm``). 단위 없는 수치를 새로 추가하지 말 것.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .ids import CORTICAL_LAYERS


class ConfigError(ValueError):
    """설정 파일이 스키마를 위반했을 때."""


# ----------------------------------------------------------------------
# 기본값
# ----------------------------------------------------------------------
#: 세포 유형 1개의 기본 파라미터. 값은 **모형 파라미터**이며 특정 실험의
#: 측정값이 아니다 (BIOLOGY_AND_ASSUMPTIONS.md 참조).
_CELL_TYPE_TEMPLATE: dict[str, Any] = {
    "dale": "excitatory",                 # excitatory | inhibitory
    "compartments": ["soma"],             # soma / basal / apical 부분집합, soma 필수
    "C_pF": {"soma": 200.0, "basal": 100.0, "apical": 100.0},
    "gL_nS": {"soma": 10.0, "basal": 5.0, "apical": 5.0},
    "EL_mV": {"soma": -70.0, "basal": -70.0, "apical": -70.0},
    "g_couple_nS": {"soma_basal": 8.0, "soma_apical": 4.0},
    "V_th_mV": -50.0,
    "V_reset_mV": -65.0,
    "t_ref_ms": 2.0,
    "target_rate_hz": 5.0,                # 항상성 임계 적응의 목표 활동률
    "sum_threshold_theta": 1.0,           # sum_threshold 모드의 초기 theta (무차원)
    "output_gain_P": 1.0,                 # 3x3 [1][2], 기본 범위 0~1 밖의 값은 경고
    "refractory_input_policy": "accumulate",  # accumulate | discard (soma 시냅스 입력 처리)
}

#: 영역 1개의 기본 구조.
_AREA_TEMPLATE: dict[str, Any] = {
    "kind": "cortex",                      # cortex | retina | thalamus
    "hierarchy_level": 0.0,                # 표시용. 배선을 강제하지 않는다.
    "surface_origin_mm": [0.0, 0.0],       # 피질 표면 좌표계의 영역 오프셋
    "surface_extent_mm": [4.0, 4.0],
    "layer_thickness_mm": {"L1": 0.10, "L2": 0.25, "L3": 0.35,
                           "L4": 0.30, "L5": 0.40, "L6": 0.40},
    "depth_origin_mm": 0.0,                # 피질 표면(z=0)에서 아래로 증가
    "neurons_per_layer": {},               # {"L4": 128, ...}. 없으면 0
    "cell_type_fractions": {},             # {"L4": {"spiny_stellate":0.8,"PV":0.2}}
    "visual_field": {
        "max_ecc_deg": 8.0,
        "rf_sigma_deg": 0.25,              # 이 영역 뉴런의 기본 수용장 크기
        "rf_sigma_growth_per_deg": 0.06,   # 편심도에 따른 수용장 확대
    },
    "notes": "",
}

#: 배선 규칙 1개의 기본값.
_WIRING_TEMPLATE: dict[str, Any] = {
    "name": "",
    "enabled": True,
    "src": {"area": "", "layer": [], "cell_type": []},
    "dst": {"area": "", "layer": [], "cell_type": []},
    "target_compartment": "basal",         # soma | basal | apical
    "receptor": "AMPA",
    "rule": "rf_knn",                      # rf_knn | local_radius | all_to_all_sampled
    "k": 12,                               # rf_knn 후보 수
    "radius_mm": 0.3,                      # local_radius 반경 (피질 mm)
    "rf_match_sigma_deg": 0.4,             # 시야 위치 대응 허용폭
    "probability": 0.5,                    # 후보 중 실제로 만들 확률
    "max_synapses_per_target": 0,          # 0 이면 제한 없음
    "weight": {"dist": "lognormal", "median": 1.0, "sigma": 0.35,
               "min": 0.0, "max": 8.0},
    "conduction_velocity_mm_per_ms": 0.3,
    "synaptic_delay_ms": 0.8,
    "use_straight_line_distance": True,    # 축삭 길이 근사 (표시 대상)
    "plasticity_rule": "none",             # none | stdp
    # Gabor 모양으로 초기 가중치를 정할지. True 면 manifest 에 기록된다.
    # 음의 필터 계수는 음의 전도도가 아니라 반대 극성(OFF) 입력으로 구현한다.
    "gabor_initialized": False,
    "note": "",
}

DEFAULTS: dict[str, Any] = {
    "meta": {
        "name": "unnamed",
        "description": "",
        "species_assumption": "primate_visual_cortex_model",
        "notes": [],
    },
    "seeds": {
        "master": 12345,
        # 고정된 이름의 독립 스트림. 조건을 추가해도 기존 조건의 초기 상태가
        # 바뀌지 않도록 이름->정수 매핑을 고정한다 (rng.py).
        "stream_offsets": {
            "data": 101, "wiring": 202, "weights": 303, "input_noise": 404,
            "learning_order": 505, "diagnostics": 606, "readout": 707,
            "split": 808, "stimulus": 909,
        },
    },
    "engine": {
        "mode": "conductance_lif",          # sum_threshold | conductance_lif
        "dt_ms": 0.5,
        "duration_ms": 200.0,
        "min_delay_steps": 1,
        "delay_rounding": "ceil",           # ceil | round
        "weight_application": "emit",       # emit | arrival (기본: 발신 시점 스냅샷)
        "sum_threshold_interval_steps": 1,  # sum_threshold 모드의 처리 구간 길이
        "reset_between_samples": {
            "voltages": True, "conductances": True, "event_queue": True,
            "traces": False, "thresholds": False, "weights": False,
            "input_log": False,
        },
        "sequence_mode": False,             # True 면 표본 간 상태를 유지 (동영상)
    },
    "receptors": {
        "AMPA":   {"tau_ms": 2.0,   "E_rev_mV": 0.0,   "kind": "excitatory", "mg_block": False},
        "NMDA":   {"tau_ms": 100.0, "E_rev_mV": 0.0,   "kind": "excitatory", "mg_block": True},
        "GABA_A": {"tau_ms": 6.0,   "E_rev_mV": -70.0, "kind": "inhibitory", "mg_block": False},
    },
    "cell_types": {},                        # 이름 -> _CELL_TYPE_TEMPLATE 병합
    "anatomy": {
        "areas": {},                         # 이름 -> _AREA_TEMPLATE 병합
        "area_gap_mm": 1.0,                  # 영역 사이 표면 좌표 간격 (배치용)
        "jitter_mm": 0.02,                   # 뉴런 위치 난수 흔들기
    },
    "wiring": {
        "rules": [],
        "dale_enforced": True,
        "allow_self_connection": False,
        "max_total_synapses": 5_000_000,     # 용량 한도. 초과 시 명시적으로 중단
    },
    "retina": {
        "image": {
            "max_side_px": 1024,
            "fov_deg": 20.0,                 # 영상 긴 변이 덮는 시야각
            "input_colorspace": "srgb",      # srgb | linear
        },
        "color": {
            "use_lms": True,
            "lms_matrix": "hunt_pointer_estevez_d65",  # 또는 "stockman_sharpe_lms"
            "opponent_channels": ["luminance", "red_green", "blue_yellow"],
        },
        "dog": {
            "center_sigma_px": 1.0,
            "surround_sigma_px": 3.0,
            "truncate": 4.0,
            "boundary_mode": "reflect",
            "normalize_each_kernel_to_unit_sum": True,
        },
        "channels": {
            "on_off_split": True,            # 6 채널 (3 대립 x ON/OFF)
            "keep_lowpass_lms": True,        # V4 색 경로용 저주파 LMS 3채널 추가
            "lowpass_sigma_px": 8.0,
        },
        "drive": {
            "mode": "rate",                  # rate | poisson
            "max_rate_hz": 60.0,
            "gain": 1.0,
            "baseline_rate_hz": 0.0,
            # rate 모드에서 발화율(Hz)을 망막 뉴런의 외부 전류(pA)로 바꾸는 계수.
            # 모형 파라미터이며 측정값이 아니다.
            "current_per_hz_pA": 2.0,
            # sum_threshold 모드에서 발화율을 무차원 기여로 바꾸는 계수.
            "sum_mode_scale": 1.0,
        },
    },
    "retinotopy": {
        "mapping": "log_polar",              # log_polar | uniform_control
        "e0_deg": 0.5,                       # rho = log(1 + ecc/e0)
        "eccentricity_unit": "deg",
        "n_radial": 12,
        "n_angular": 24,
        "fovea_patch": {
            "enabled": True,
            "radius_deg": 0.5,
            "grid": 4,                       # 4x4 Cartesian 격자
        },
        "sampling": {
            "lowpass_before_sampling": True,
            "sigma_scale": 0.5,              # sigma_px = sigma_scale * 셀 크기
            "min_sigma_px": 0.5,
            "interpolation_order": 1,
        },
        "uniform_control": {
            "match_total_samples": True,
            "grid": 0,                       # 0 이면 총 표본 수에서 자동 계산
        },
        "binocular": {
            "enabled": False,
            "interocular_shift_deg": 0.0,
        },
    },
    "v1": {
        "n_orientations": 12,                # 0~165도, 15도 간격
        "orientation_step_deg": 15.0,
        "phases_rad": [0.0, -1.5707963267948966],
        "pinwheel": {
            "enabled": True,
            "hypercolumn_mm": 0.8,
            "n_pinwheels_per_mm2": 3.0,
        },
        "ocular_dominance": {
            "enabled": True,
            "column_width_mm": 0.4,
            "strength": 0.6,                 # 0=양안 동일, 1=완전 단안
        },
        "gabor_init": {
            "enabled": False,                # 초기 배선을 Gabor 로 정하면 기록된다
            "sigma_deg": 0.25,
            "aspect": 1.6,
            "cycles_per_deg": 2.0,
        },
        "fixed_gabor_reference": {
            "enabled": False,                # 학습 아님. 대조 경로.
            "sigma_deg": 0.25,
            "aspect": 1.6,
            "cycles_per_deg": 2.0,
            "energy_eps": 1e-6,
        },
    },
    "learning": {
        "mode": "none",                      # none | stdp_homeostasis | rao_reference
        "task_learning_enabled": True,
        "weight_decay_enabled": False,
        "threshold_adaptation_enabled": False,
        "stdp": {
            "A_plus": 0.01,
            "A_minus": 0.012,
            "tau_plus_ms": 20.0,
            "tau_minus_ms": 20.0,
            "weight_min": 0.0,
            "weight_max": 8.0,
            "weight_decay_per_ms": 0.0,
            "simultaneous_policy": "both",   # both | pre_first | post_first
            "update_order": "post_then_pre", # 문서화된 갱신 순서
            "apply_to_inhibitory": False,
        },
        "homeostasis": {
            "eta_theta": 0.002,
            "window_ms": 500.0,
            "theta_min_mV": -60.0,
            "theta_max_mV": -35.0,
            "theta_min_sum": 0.05,
            "theta_max_sum": 50.0,
            "per_cell_type_target": True,
        },
        "neighbor_theta_averaging": {
            "enabled": False,                # 소거(ablation) 실험 전용 옵션
            "radius_mm": 0.15,
            "strength": 0.0,
        },
        "rao": {
            "n_levels": 2,
            "level_sizes": [32, 16],
            "sigma": 1.0,
            "sigma_td": 2.0,
            "alpha": 0.05,
            "lambda_u": 0.001,
            "settle_steps": 30,
            "r_step": 0.05,
            "u_step": 0.002,
            "freeze_U_during_settle": True,
        },
        "apical_error_coupling": {
            "enabled": False,                # Rao 오차 -> apical 전류 (가정 문서화 필요)
            "gain_pA_per_unit": 0.0,
            "split_sign": True,
        },
    },
    "readout": {
        "enabled": False,
        "source_area": "IT",
        "source_layer": ["L2", "L3"],
        "window_ms": [50.0, 200.0],
        "classifier": "ridge",               # ridge | logistic
        "l2": 1.0,
        "n_classes": 7,
        "note": "분류 readout 은 별도 모듈이며 피질 학습과 분리된다.",
    },
    "recording": {
        "mode": "full",                      # full | selected | summary
        "backend": "hdf5",                   # hdf5 | npz
        "selected_neurons": [],
        "selected_areas": [],
        "max_events": 20_000_000,
        "max_state_samples": 2_000_000,
        "state_sample_every_steps": 1,
        "compression": "gzip",
        "compression_level": 4,
        "chunk_rows": 4096,
        "flush_every_steps": 200,
        "selection_criterion": "",           # summary/selected 모드에서 필수
    },
    "checkpoint": {
        "enabled": True,
        "every_steps": 0,                    # 0 이면 표본 경계에서만
        "every_samples": 1,
        "keep_last": 3,
    },
    "experiment": {
        "protocol": "single_pass",           # single_pass | sweep | train_dev_test
        "stimuli": [],                       # stimuli.py 가 해석하는 명세 목록
        "n_samples": 1,
        "splits": {"train": 0.6, "dev": 0.2, "test": 0.2, "stratified": True},
        "conditions": [],                    # 대조군 정의
        "limits": {
            "max_runtime_minutes": 0,        # 0 이면 제한 없음
            "max_disk_mb": 4096,
            "max_ram_mb": 8192,
        },
    },
    "validation": {
        "run_structural_checks": True,
        "stop_experiment_on_failure": True,
        "dt_convergence_factors": [1.0, 0.5, 0.25],
        "finite_difference_eps": 1e-6,
        "tolerances": {
            "matrix_view": 0.0,
            "delay_steps": 0.0,
            "lif_dt_convergence_mV": 0.5,
            "logpolar_roundtrip_deg": 0.05,
            "dog_uniform_response": 1e-6,
            "rao_gradient_rel": 1e-4,
        },
    },
}


# ----------------------------------------------------------------------
# 병합과 검증
# ----------------------------------------------------------------------
#: 하위 키가 **자유 형식**인 경로. 여기서는 기본값에 없는 키도 허용한다.
#: ``*`` 는 임의의 한 단계를 뜻한다. 이 목록에 없는 경로의 낯선 키는 오류다
#: (오타가 조용히 무시되는 것을 막기 위함).
_OPEN_PATH_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("seeds", "stream_offsets"),
    ("anatomy", "areas", "*", "neurons_per_layer"),
    ("anatomy", "areas", "*", "cell_type_fractions"),
    ("anatomy", "areas", "*", "cell_type_fractions", "*"),
    ("anatomy", "areas", "*", "layer_thickness_mm"),
)


def _is_open_path(path: str) -> bool:
    if not path:
        return False
    segs = tuple(path.split("."))
    for pat in _OPEN_PATH_PATTERNS:
        if len(pat) != len(segs):
            continue
        if all(p == "*" or p == s for p, s in zip(pat, segs)):
            return True
    return False


def _deep_merge(base: dict[str, Any], override: dict[str, Any],
                path: str = "") -> dict[str, Any]:
    """base 위에 override 를 깊은 병합.

    base 에 없는 키는 ConfigError 다. 단 :data:`_OPEN_PATH_PATTERNS` 에 해당하는
    자유 형식 경로(층 이름, 세포 유형 이름, RNG 스트림 이름 등)에서는 허용한다.
    """
    out = copy.deepcopy(base)
    open_here = _is_open_path(path)
    for key, val in override.items():
        here = f"{path}.{key}" if path else key
        if key not in out:
            if open_here:
                out[key] = copy.deepcopy(val)
                continue
            raise ConfigError(
                f"알 수 없는 설정 키: '{here}'. 오타이거나 지원하지 않는 항목이다. "
                f"사용 가능한 키: {sorted(out.keys())}"
            )
        if isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val, here)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _merge_free_dict(template: dict[str, Any], override: dict[str, Any],
                     path: str) -> dict[str, Any]:
    """항목 수가 가변인 dict (영역/세포유형) 에 템플릿을 적용한다."""
    return _deep_merge(template, override, path)


def resolve(user_config: dict[str, Any]) -> dict[str, Any]:
    """사용자 설정을 기본값 위에 병합하고 전체를 검증한다 (부작용 없음).

    Returns
    -------
    dict : 완전히 해석된 설정. manifest.json 에 그대로 저장된다.
    """
    free_sections = {
        "cell_types": user_config.get("cell_types", {}),
        "areas": user_config.get("anatomy", {}).get("areas", {}),
    }
    stripped = copy.deepcopy(user_config)
    stripped.pop("cell_types", None)
    if "anatomy" in stripped:
        stripped["anatomy"] = {k: v for k, v in stripped["anatomy"].items() if k != "areas"}
    wiring_rules = None
    if "wiring" in stripped and "rules" in stripped["wiring"]:
        wiring_rules = stripped["wiring"].pop("rules")

    cfg = _deep_merge(DEFAULTS, stripped)

    cfg["cell_types"] = {
        name: _merge_free_dict(_CELL_TYPE_TEMPLATE, spec, f"cell_types.{name}")
        for name, spec in free_sections["cell_types"].items()
    }
    cfg["anatomy"]["areas"] = {
        name: _merge_free_dict(_AREA_TEMPLATE, spec, f"anatomy.areas.{name}")
        for name, spec in free_sections["areas"].items()
    }
    if wiring_rules is not None:
        cfg["wiring"]["rules"] = [
            _merge_free_dict(_WIRING_TEMPLATE, r, f"wiring.rules[{i}]")
            for i, r in enumerate(wiring_rules)
        ]

    validate(cfg)
    return cfg


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate(cfg: dict[str, Any]) -> None:
    """해석된 설정의 구조·값 범위를 검사한다. 위반 시 ConfigError."""
    eng = cfg["engine"]
    _require(eng["mode"] in ("sum_threshold", "conductance_lif"),
             f"engine.mode 는 sum_threshold 또는 conductance_lif 여야 한다: {eng['mode']!r}")
    _require(eng["dt_ms"] > 0, "engine.dt_ms 는 양수여야 한다")
    _require(eng["duration_ms"] > 0, "engine.duration_ms 는 양수여야 한다")
    _require(eng["min_delay_steps"] >= 1,
             "engine.min_delay_steps 는 1 이상이어야 한다 (0 지연 재귀 방지)")
    _require(eng["delay_rounding"] in ("ceil", "round"),
             "engine.delay_rounding 은 ceil 또는 round")
    _require(eng["weight_application"] in ("emit", "arrival"),
             "engine.weight_application 은 emit 또는 arrival")
    _require(eng["sum_threshold_interval_steps"] >= 1,
             "engine.sum_threshold_interval_steps 는 1 이상")

    for rname, r in cfg["receptors"].items():
        _require(r["tau_ms"] > 0, f"receptors.{rname}.tau_ms 는 양수여야 한다")
        _require(r["kind"] in ("excitatory", "inhibitory"),
                 f"receptors.{rname}.kind 는 excitatory/inhibitory")

    _require(len(cfg["cell_types"]) > 0, "cell_types 가 비어 있다")
    for cname, c in cfg["cell_types"].items():
        _require(c["dale"] in ("excitatory", "inhibitory"),
                 f"cell_types.{cname}.dale 는 excitatory/inhibitory")
        _require("soma" in c["compartments"],
                 f"cell_types.{cname}.compartments 에 soma 가 없다")
        for comp in c["compartments"]:
            _require(comp in ("soma", "basal", "apical"),
                     f"cell_types.{cname}: 알 수 없는 구획 {comp!r}")
            _require(c["C_pF"][comp] > 0, f"cell_types.{cname}.C_pF[{comp}] > 0 이어야 한다")
            _require(c["gL_nS"][comp] > 0, f"cell_types.{cname}.gL_nS[{comp}] > 0")
        _require(c["t_ref_ms"] >= 0, f"cell_types.{cname}.t_ref_ms >= 0")
        _require(c["V_th_mV"] > c["V_reset_mV"],
                 f"cell_types.{cname}: V_th_mV 가 V_reset_mV 보다 커야 한다")
        _require(c["refractory_input_policy"] in ("accumulate", "discard"),
                 f"cell_types.{cname}.refractory_input_policy 는 accumulate/discard")

    areas = cfg["anatomy"]["areas"]
    _require(len(areas) > 0, "anatomy.areas 가 비어 있다")
    for aname, a in areas.items():
        _require(a["kind"] in ("cortex", "retina", "thalamus"),
                 f"anatomy.areas.{aname}.kind 는 cortex/retina/thalamus")
        for lname, n in a["neurons_per_layer"].items():
            _require(int(n) >= 0, f"{aname}.{lname} 뉴런 수는 0 이상")
            fr = a["cell_type_fractions"].get(lname)
            if int(n) > 0:
                _require(fr is not None and len(fr) > 0,
                         f"anatomy.areas.{aname}.cell_type_fractions.{lname} 가 필요하다")
                s = float(sum(fr.values()))
                _require(abs(s - 1.0) < 1e-6,
                         f"{aname}.{lname} 세포 유형 비율 합이 1 이 아니다: {s}")
                for ct in fr:
                    _require(ct in cfg["cell_types"],
                             f"{aname}.{lname} 의 세포 유형 {ct!r} 이 cell_types 에 없다")
        if a["kind"] == "cortex":
            total = sum(a["layer_thickness_mm"].get(l, 0.0) for l in CORTICAL_LAYERS)
            _require(total > 0, f"anatomy.areas.{aname}: 층 두께 합이 0 이다")

    for i, r in enumerate(cfg["wiring"]["rules"]):
        tag = f"wiring.rules[{i}] ({r['name'] or 'unnamed'})"
        _require(r["src"]["area"] in areas, f"{tag}: 알 수 없는 src.area {r['src']['area']!r}")
        _require(r["dst"]["area"] in areas, f"{tag}: 알 수 없는 dst.area {r['dst']['area']!r}")
        _require(r["receptor"] in cfg["receptors"], f"{tag}: 알 수 없는 receptor")
        _require(r["target_compartment"] in ("soma", "basal", "apical"),
                 f"{tag}: 알 수 없는 target_compartment")
        _require(r["rule"] in ("rf_knn", "local_radius", "all_to_all_sampled"),
                 f"{tag}: 알 수 없는 rule {r['rule']!r}")
        _require(0.0 <= r["probability"] <= 1.0, f"{tag}: probability 는 [0,1]")
        _require(r["conduction_velocity_mm_per_ms"] > 0, f"{tag}: 전도속도는 양수")
        _require(r["synaptic_delay_ms"] >= 0, f"{tag}: 시냅스 지연은 0 이상")
        _require(r["plasticity_rule"] in ("none", "stdp"), f"{tag}: 알 수 없는 가소성 규칙")
        _require(isinstance(r["gabor_initialized"], bool),
                 f"{tag}: gabor_initialized 는 true/false")
        w = r["weight"]
        _require(w["dist"] in ("lognormal", "normal", "constant"), f"{tag}: 알 수 없는 weight.dist")
        _require(w["min"] >= 0.0, f"{tag}: weight.min 은 0 이상 (Dale 부호는 세포 유형이 결정)")
        _require(w["max"] > w["min"], f"{tag}: weight.max > weight.min")

    ret = cfg["retina"]
    _require(ret["image"]["fov_deg"] > 0, "retina.image.fov_deg 는 양수")
    _require(ret["image"]["input_colorspace"] in ("srgb", "linear"),
             "retina.image.input_colorspace 는 srgb 또는 linear")
    _require(ret["dog"]["surround_sigma_px"] > ret["dog"]["center_sigma_px"] > 0,
             "retina.dog: 0 < center_sigma_px < surround_sigma_px 여야 한다")
    _require(ret["drive"]["mode"] in ("rate", "poisson"),
             "retina.drive.mode 는 rate 또는 poisson")
    _require(ret["drive"]["max_rate_hz"] > 0, "retina.drive.max_rate_hz 는 양수")
    _require(ret["drive"]["current_per_hz_pA"] >= 0,
             "retina.drive.current_per_hz_pA 는 0 이상")
    _require(ret["color"]["lms_matrix"] in ("hunt_pointer_estevez_d65", "stockman_sharpe_lms"),
             "retina.color.lms_matrix 값이 지원 목록에 없다")

    rt = cfg["retinotopy"]
    _require(rt["mapping"] in ("log_polar", "uniform_control"),
             "retinotopy.mapping 은 log_polar 또는 uniform_control")
    _require(rt["e0_deg"] > 0, "retinotopy.e0_deg 는 양수 (log(1+ecc/e0) 의 특이점 회피)")
    _require(rt["n_radial"] >= 1 and rt["n_angular"] >= 1,
             "retinotopy.n_radial/n_angular 는 1 이상")
    _require(rt["eccentricity_unit"] in ("deg", "px"),
             "retinotopy.eccentricity_unit 은 deg 또는 px")
    _require(rt["sampling"]["interpolation_order"] in (0, 1, 3),
             "retinotopy.sampling.interpolation_order 는 0/1/3")

    v1 = cfg["v1"]
    _require(v1["n_orientations"] >= 1, "v1.n_orientations 는 1 이상")
    _require(len(v1["phases_rad"]) >= 1, "v1.phases_rad 가 비어 있다")

    lr = cfg["learning"]
    _require(lr["mode"] in ("none", "stdp_homeostasis", "rao_reference"),
             f"learning.mode 값이 잘못되었다: {lr['mode']!r}")
    _require(lr["stdp"]["tau_plus_ms"] > 0 and lr["stdp"]["tau_minus_ms"] > 0,
             "learning.stdp 시정수는 양수")
    _require(lr["stdp"]["weight_max"] > lr["stdp"]["weight_min"] >= 0.0,
             "learning.stdp 가중치 범위가 잘못되었다 (하한 0 이상, 상한 > 하한)")
    _require(lr["stdp"]["simultaneous_policy"] in ("both", "pre_first", "post_first"),
             "learning.stdp.simultaneous_policy 값이 잘못되었다")
    rao = lr["rao"]
    _require(rao["sigma"] > 0 and rao["sigma_td"] > 0,
             "learning.rao.sigma / sigma_td 는 양수여야 한다 (분산 계수)")
    _require(rao["n_levels"] == len(rao["level_sizes"]),
             "learning.rao.n_levels 와 level_sizes 길이가 다르다")
    _require(rao["alpha"] >= 0 and rao["lambda_u"] >= 0,
             "learning.rao.alpha / lambda_u 는 0 이상")

    rec = cfg["recording"]
    _require(rec["mode"] in ("full", "selected", "summary"),
             "recording.mode 는 full/selected/summary")
    _require(rec["backend"] in ("hdf5", "npz"), "recording.backend 는 hdf5 또는 npz")
    if rec["mode"] != "full":
        _require(bool(rec["selection_criterion"]),
                 "recording.mode 가 full 이 아니면 selection_criterion 을 반드시 적어야 한다 "
                 "(무엇을 기록하지 않는지 manifest 에 남긴다)")
    _require(rec["state_sample_every_steps"] >= 1,
             "recording.state_sample_every_steps 는 1 이상")

    _require(cfg["wiring"]["max_total_synapses"] > 0, "wiring.max_total_synapses 는 양수")


def load(path: str | Path) -> dict[str, Any]:
    """JSON 설정 파일을 읽어 해석한다. 파일이 없으면 친절한 한국어 오류."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(
            f"설정 파일을 찾을 수 없다: {p}\n"
            f"  - 경로를 확인하거나 configs/ 폴더의 예시 파일을 사용하라.\n"
            f"  - 예: python -m cortex.cli inspect-config --config configs/minimal.json"
        )
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"설정 파일 JSON 구문 오류 ({p}): {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"설정 파일의 최상위는 객체여야 한다: {p}")
    cfg = resolve(raw)
    cfg["meta"]["source_path"] = str(p.resolve())
    return cfg


@dataclass(frozen=True)
class SizeEstimate:
    """실행 전에 계산하는 규모 추정. **런타임(초)은 추정하지 않는다.**"""

    n_neurons: int
    n_synapses_estimated: int
    n_steps: int
    events_estimated: int
    state_rows_estimated: int
    ram_mb_estimated: float
    disk_mb_estimated: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_neurons": self.n_neurons,
            "n_synapses_estimated": self.n_synapses_estimated,
            "n_steps": self.n_steps,
            "events_estimated": self.events_estimated,
            "state_rows_estimated": self.state_rows_estimated,
            "ram_mb_estimated": round(self.ram_mb_estimated, 2),
            "disk_mb_estimated": round(self.disk_mb_estimated, 2),
            "runtime_seconds_estimated": None,
            "runtime_note_ko": "실제 런타임은 측정 전에 확정하지 않는다.",
            "notes": list(self.notes),
        }


def derived_retina_neuron_count(cfg: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """망막 영역의 뉴런 수는 (채널 수 x 시야 샘플 수) 로 **유도**된다.

    설정의 ``neurons_per_layer`` 값 대신 이 값을 쓰며, 실제로 쓴 값과 유도 근거를
    manifest 에 남긴다. 순환 import 를 피하려고 지연 import 한다.
    """
    from .retinotopy import build_grid  # 지연 import
    from .retina import OPPONENT_NAMES

    grid = build_grid(cfg)
    ch = cfg["retina"]["channels"]
    n_ch = (len(OPPONENT_NAMES) * (2 if ch["on_off_split"] else 1)
            + (3 if ch["keep_lowpass_lms"] else 0))
    n = int(n_ch * grid.n_samples)
    return n, {"n_channels": n_ch, "n_samples": int(grid.n_samples),
               "grid_summary": grid.summary()}


def count_neurons(cfg: dict[str, Any]) -> int:
    """망막 영역은 유도값, 나머지는 설정값으로 총 뉴런 수를 센다."""
    total = 0
    for a in cfg["anatomy"]["areas"].values():
        if a["kind"] == "retina":
            total += derived_retina_neuron_count(cfg)[0]
        else:
            total += int(sum(int(n) for n in a["neurons_per_layer"].values()))
    return int(total)


def estimate_sizes(cfg: dict[str, Any]) -> SizeEstimate:
    """뉴런/시냅스/이벤트/용량의 **상한 성격 추정**. 실행 전에 계산한다."""
    n_neurons = count_neurons(cfg)
    n_steps = int(round(cfg["engine"]["duration_ms"] / cfg["engine"]["dt_ms"]))
    n_samples = max(1, int(cfg["experiment"]["n_samples"]))

    areas = cfg["anatomy"]["areas"]
    n_syn = 0
    notes: list[str] = []
    for r in cfg["wiring"]["rules"]:
        if not r["enabled"]:
            continue
        dst = areas[r["dst"]["area"]]
        if dst["kind"] == "retina":
            dst_n = derived_retina_neuron_count(cfg)[0]
        else:
            dst_n = sum(int(n) for lname, n in dst["neurons_per_layer"].items()
                        if (not r["dst"]["layer"]) or lname in r["dst"]["layer"])
        per_target = r["k"] if r["rule"] == "rf_knn" else r["k"]
        if r["max_synapses_per_target"] > 0:
            per_target = min(per_target, r["max_synapses_per_target"])
        n_syn += int(dst_n * per_target * r["probability"])
    if n_syn > cfg["wiring"]["max_total_synapses"]:
        notes.append(
            f"추정 시냅스 수 {n_syn} 가 wiring.max_total_synapses "
            f"({cfg['wiring']['max_total_synapses']}) 를 넘는다. 실행은 한도에서 중단된다."
        )

    # 이벤트 수 추정: 평균 활동률 가정 (모형 가정이며 측정값이 아니다)
    assumed_rate_hz = 5.0
    spikes = n_neurons * assumed_rate_hz * cfg["engine"]["duration_ms"] / 1000.0 * n_samples
    fanout = (n_syn / max(1, n_neurons))
    events = int(spikes * max(1.0, fanout))
    notes.append(f"이벤트 수 추정은 평균 {assumed_rate_hz} Hz 가정에서 나온 값이다 (모형 가정).")

    bytes_per_event = 88      # DATA_SCHEMA.md 의 이벤트 레코드 크기
    bytes_per_state_row = 64
    rec = cfg["recording"]
    state_rows = 0
    if rec["mode"] != "summary":
        n_rec = n_neurons if rec["mode"] == "full" else max(1, len(rec["selected_neurons"]))
        state_rows = int(n_rec * (n_steps / rec["state_sample_every_steps"]) * n_samples)

    disk_mb = (events * bytes_per_event + state_rows * bytes_per_state_row) / 1e6
    ram_mb = (n_neurons * 3 * 8 * 6 + n_syn * 8 * 10) / 1e6 + 64.0

    return SizeEstimate(
        n_neurons=n_neurons,
        n_synapses_estimated=n_syn,
        n_steps=n_steps,
        events_estimated=events,
        state_rows_estimated=state_rows,
        ram_mb_estimated=ram_mb,
        disk_mb_estimated=disk_mb,
        notes=notes,
    )


def config_hash(cfg: dict[str, Any]) -> str:
    """해석된 설정의 sha256 (키 정렬, UTF-8)."""
    import hashlib
    payload = json.dumps(cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def iter_area_layers(cfg: dict[str, Any]) -> Iterable[tuple[str, str, int]]:
    """(area, layer, n_neurons) 를 결정적 순서로 열거한다."""
    for aname in sorted(cfg["anatomy"]["areas"]):
        a = cfg["anatomy"]["areas"][aname]
        for lname in sorted(a["neurons_per_layer"]):
            yield aname, lname, int(a["neurons_per_layer"][lname])


__all__ = [
    "ConfigError", "DEFAULTS", "resolve", "validate", "load",
    "SizeEstimate", "estimate_sizes", "count_neurons", "config_hash",
    "derived_retina_neuron_count",
    "iter_area_layers",
]
