

def _engine_lif(duration_ms: float = 200.0) -> dict[str, Any]:
    return {
        "mode": "conductance_lif", "dt_ms": 0.5, "duration_ms": duration_ms,
        "min_delay_steps": 1, "delay_rounding": "ceil",
        "weight_application": "emit",
        "reset_between_samples": {
            "voltages": True, "conductances": True, "event_queue": True,
            "traces": True, "thresholds": False, "weights": False,
            "input_log": False,
        },
        "sequence_mode": False,
    }


def _v1_block() -> dict[str, Any]:
    return {
        "n_orientations": 12, "orientation_step_deg": 15.0,
        "phases_rad": [0.0, -1.5707963267948966],
        "pinwheel": {"enabled": True, "hypercolumn_mm": 0.8,
                     "n_pinwheels_per_mm2": 3.0},
        "ocular_dominance": {"enabled": True, "column_width_mm": 0.4,
                             "strength": 0.6},
        "gabor_init": {"enabled": True, "sigma_deg": 0.35, "aspect": 1.6,
                       "cycles_per_deg": 1.5},
        "fixed_gabor_reference": {"enabled": True, "sigma_deg": 0.35,
                                  "aspect": 1.6, "cycles_per_deg": 1.5,
                                  "energy_eps": 1e-06},
    }


def _validation_block() -> dict[str, Any]:
    return {
        "dt_convergence_factors": [1.0, 0.5, 0.25],
        "tolerances": {"lif_dt_convergence_mV": 0.6,
                       "logpolar_roundtrip_deg": 0.05,
                       "dog_uniform_response": 1e-06,
                       "rao_gradient_rel": 0.0001},
    }


def _config_v1_small() -> dict[str, Any]:
    """configs/v1_small.json 과 동일한 내용 (작은 영상 + V1 미세회로)."""
    return {
        "meta": {
            "name": "v1_small",
            "description": "작은 영상 + V1 미세회로. 방향·위상·주변 문맥 검사를 위한 설정.",
            "species_assumption": "primate_visual_cortex_model",
            "notes": [
                "12개 방향 x 2 위상은 계산을 위한 이산화다. 피질이 정확히 24개 균일 채널이라는 주장이 아니다.",
                "fixed_gabor_reference 는 학습이 아니라 고정 특징 추출 대조 경로다.",
            ],
        },
        "seeds": {"master": 20240202},
        "engine": _engine_lif(200.0),
        "cell_types": _CELL_TYPES,
        "retina": {
            "image": {"max_side_px": 96, "fov_deg": 12.0,
                      "input_colorspace": "srgb"},
            "channels": {"on_off_split": True, "keep_lowpass_lms": True,
                         "lowpass_sigma_px": 8.0},
            "drive": {"mode": "rate", "max_rate_hz": 60.0, "gain": 1.0,
                      "baseline_rate_hz": 0.5, "current_per_hz_pA": 2.5,
                      "sum_mode_scale": 1.0},
        },
        "retinotopy": {
            "mapping": "log_polar", "e0_deg": 0.5, "n_radial": 5,
            "n_angular": 10,
            "fovea_patch": {"enabled": True, "radius_deg": 0.5, "grid": 3},
            "sampling": {"lowpass_before_sampling": True, "sigma_scale": 0.5,
                         "min_sigma_px": 0.5, "interpolation_order": 1},
            "binocular": {"enabled": False, "interocular_shift_deg": 0.0},
        },
        "v1": _v1_block(),
        "anatomy": {
            "area_gap_mm": 1.0, "jitter_mm": 0.02,
            "areas": {
                "Retina": {
                    "kind": "retina", "surface_extent_mm": [1.5, 1.5],
                    "neurons_per_layer": {"L_input": 0},
                    "cell_type_fractions": {"L_input": {"retinal_ganglion": 1.0}},
                    "visual_field": {"max_ecc_deg": 5.0, "rf_sigma_deg": 0.25,
                                     "rf_sigma_growth_per_deg": 0.06},
                    "notes": "뉴런 수는 (채널 수 x 시야 샘플 수)로 유도된다.",
                },
                "LGN": {
                    "kind": "thalamus", "surface_extent_mm": [1.5, 1.5],
                    "neurons_per_layer": {"L_relay": 96},
                    "cell_type_fractions": {"L_relay": {"lgn_relay": 1.0}},
                    "visual_field": {"max_ecc_deg": 5.0, "rf_sigma_deg": 0.3,
                                     "rf_sigma_growth_per_deg": 0.07},
                },
                "V1": _cortical_area(
                    1.0, [2.4, 2.4],
                    {"L1": 12, "L2": 96, "L3": 96, "L4": 144, "L5": 64, "L6": 64},
                    5.0, 0.35, 0.08,
                    "hypercolumn 미세 좌표와 방향 지도가 배치·배선에 반영된다."),
            },
        },
        "wiring": {
            "max_total_synapses": 800000,
            "rules": [
                _rule("Retina->LGN", "Retina", ["L_input"], "LGN", ["L_relay"],
                      comp="soma", k=6, sigma=0.6, prob=0.8, median=1.4, vel=1.0,
                      delay=1.0, note="망막 -> LGN 중계 (시야 위치 대응)"),
                _rule("LGN->V1_L4", "LGN", ["L_relay"], "V1", ["L4"], k=12,
                      sigma=0.7, prob=0.6, median=1.2, vel=0.5, delay=1.2,
                      dst_types=["spiny_stellate"], plastic="stdp", gabor=True,
                      note="Gabor 모양으로 초기 배선을 정했다. 이 선택성은 학습된 것이 아니다."),
                _rule("LGN->V1_L4_PV", "LGN", ["L_relay"], "V1", ["L4"],
                      comp="soma", k=8, sigma=0.8, prob=0.4, median=1.0, vel=0.5,
                      delay=1.1, dst_types=["pv_basket"],
                      note="전방향 억제 (feedforward inhibition)"),
                _rule("LGN->V1_L6", "LGN", ["L_relay"], "V1", ["L6"], k=4,
                      sigma=1.0, prob=0.25, median=0.5, vel=0.5, delay=1.4,
                      dst_types=["pyramidal"],
                      note="L4 외 층으로 가는 약한 상향 입력 (혼합 허용)"),
                *_local_microcircuit("V1"),
                _rule("V1_L6->LGN_feedback", "V1", ["L6"], "LGN", ["L_relay"],
                      comp="soma", k=5, sigma=1.2, prob=0.35, median=0.4,
                      vel=0.5, delay=2.5, src_types=["pyramidal"],
                      note="L6 -> LGN 피드백. 재귀 경로가 시간 진화에 영향을 준다."),
            ],
        },
        "learning": {
            "mode": "none",
            "stdp": {"A_plus": 0.01, "A_minus": 0.012, "tau_plus_ms": 20.0,
                     "tau_minus_ms": 20.0, "weight_min": 0.0, "weight_max": 6.0},
            "homeostasis": {"eta_theta": 0.002, "window_ms": 500.0,
                            "theta_min_mV": -60.0, "theta_max_mV": -35.0},
        },
        "readout": {"enabled": False, "source_area": "V1",
                    "source_layer": ["L2", "L3"], "n_classes": 4},
        "recording": {
            "mode": "selected", "backend": "hdf5",
            "selection_criterion": "영역마다 균등 간격으로 뽑은 표본 뉴런만 상태를 기록한다. "
                                   "이벤트는 선택 뉴런에 도착/발신한 것만 저장된다.",
            "selected_areas": ["V1", "LGN"],
            "state_sample_every_steps": 2,
            "max_events": 8000000, "max_state_samples": 800000,
            "flush_every_steps": 100,
        },
        "checkpoint": {"enabled": True, "every_samples": 4, "keep_last": 3},
        "experiment": {
            "protocol": "single_pass", "n_samples": 1,
            "stimuli": [
                {"kind": "uniform", "n": 1,
                 "params": {"size_px": 96, "value": 0.5}},
                {"kind": "orientation_sweep", "n": 12,
                 "params": {"size_px": 96, "n_orientations": 12,
                            "radius_px": 28.0, "cycles_per_px": 0.06,
                            "contrast": 0.4}},
                {"kind": "phase_sweep", "n": 8,
                 "params": {"size_px": 96, "n_phases": 8, "radius_px": 28.0,
                            "cycles_per_px": 0.06}},
                {"kind": "contrast_sweep", "n": 5,
                 "params": {"size_px": 96, "min": 0.05, "max": 0.45,
                            "radius_px": 28.0, "cycles_per_px": 0.06}},
                {"kind": "length_sweep", "n": 6,
                 "params": {"size_px": 96, "min_px": 4.0, "max_px": 70.0,
                            "width_px": 3.0}},
                {"kind": "center_surround", "n": 1,
                 "params": {"size_px": 96, "n_orientations": 4,
                            "radius_px": 14.0, "outer_px": 40.0,
                            "cycles_per_px": 0.06}},
            ],
            "limits": {"max_disk_mb": 2048, "max_ram_mb": 4096},
        },
        "validation": _validation_block(),
    }


def _config_hierarchy_small() -> dict[str, Any]:
    """configs/hierarchy_small.json 과 동일한 내용 (LGN + V1/V2/V3/V4/IT)."""
    counts_small = {"L1": 8, "L2": 48, "L3": 48, "L4": 64, "L5": 32, "L6": 32}
    v1 = _config_v1_small()
    rules: list[dict[str, Any]] = [
        _rule("Retina->LGN", "Retina", ["L_input"], "LGN", ["L_relay"],
              comp="soma", k=6, sigma=0.6, prob=0.8, median=1.4, vel=1.0,
              delay=1.0, enabled=True),
        _rule("LGN->V1_L4", "LGN", ["L_relay"], "V1", ["L4"], k=12, sigma=0.7,
              prob=0.6, median=1.2, vel=0.5, delay=1.2, enabled=True,
              dst_types=["spiny_stellate"], plastic="stdp", gabor=True,
              note="Gabor 모양 초기 배선 (학습된 선택성이 아니다)"),
        _rule("LGN->V1_L4_PV", "LGN", ["L_relay"], "V1", ["L4"], comp="soma",
              k=8, sigma=0.8, prob=0.4, median=1.0, vel=0.5, delay=1.1,
              enabled=True, dst_types=["pv_basket"]),
        _rule("LGN->V1_L6", "LGN", ["L_relay"], "V1", ["L6"], k=4, sigma=1.0,
              prob=0.25, median=0.5, vel=0.5, delay=1.4, enabled=True,
              dst_types=["pyramidal"]),
    ]
    # V1 의 미세회로는 v1_small 과 동일한 규칙(enabled 키 없음)을 그대로 쓰고,
    # 상위 영역은 같은 모양의 규칙을 enabled=True 로 복제한다.
    rules += [r for r in v1["wiring"]["rules"]
              if r["name"].startswith("V1_") and "LGN" not in r["name"]]
    for area in ("V2", "V3", "V4", "IT"):
        rules += _local_microcircuit(area)
    # 기본 물체 인식 경로: V1 <-> V2 <-> V4 <-> IT
    rules += _feedforward("V1", "V2", sigma=0.9, delay=2.0)
    rules += _feedforward("V2", "V4", sigma=1.4, delay=2.4)
    rules += _feedforward("V4", "IT", sigma=2.2, delay=2.8)
    rules += _feedback("V1", "V2", sigma=1.2, delay=3.0)
    rules += _feedback("V2", "V4", sigma=1.8, delay=3.4)
    rules += _feedback("V4", "IT", sigma=2.6, delay=3.8)
    # 선택 경로: V2 <-> V3 <-> V4 (기본 활성화, 설정으로 끌 수 있다)
    rules += _feedforward("V2", "V3", sigma=1.2, delay=2.2)
    rules += _feedforward("V3", "V4", sigma=1.6, delay=2.4)
    rules += _feedback("V2", "V3", sigma=1.6, delay=3.2)
    rules += _feedback("V3", "V4", sigma=2.0, delay=3.4)
    rules += [
        _rule("V1->V4_bypass", "V1", ["L2", "L3"], "V4", ["L4"], k=5, sigma=1.6,
              prob=0.12, median=0.6, vel=0.5, delay=2.6, enabled=True,
              src_types=["pyramidal"], dst_types=["spiny_stellate"],
              note="영역 간 우회 경로. 집단 수준 경향의 예외를 허용한다."),
        _rule("V1_L6->LGN_feedback", "V1", ["L6"], "LGN", ["L_relay"],
              comp="soma", k=5, sigma=1.2, prob=0.35, median=0.4, vel=0.5,
              delay=2.5, enabled=True, src_types=["pyramidal"]),
    ]
    return {
        "meta": {
            "name": "hierarchy_small",
            "description": "LGN 과 지정한 모든 피질 영역(V1,V2,V3,V4,IT)을 포함한 작은 연결망.",
            "species_assumption": "primate_visual_cortex_model",
            "notes": [
                "기본 물체 인식 경로는 V1<->V2<->V4<->IT 이고, V2<->V3<->V4 경로는 설정으로 켤 수 있다.",
                "모든 정보가 V1->V2->V3->V4->IT 한 사슬만 거친다고 가정하지 않는다.",
                "영역별 뉴런 수·층 두께·연결 확률은 출처가 명시되지 않은 모형 파라미터다.",
                "이 기능 배분은 연구용 작업 가설이다. 같은 영역이 여러 시각 특성에 관여할 수 있다.",
            ],
        },
        "seeds": {"master": 20240303},
        "engine": _engine_lif(250.0),
        "cell_types": _CELL_TYPES,
        "retina": {
            "image": {"max_side_px": 96, "fov_deg": 12.0,
                      "input_colorspace": "srgb"},
            "channels": {"on_off_split": True, "keep_lowpass_lms": True,
                         "lowpass_sigma_px": 8.0},
            "drive": {"mode": "rate", "max_rate_hz": 60.0, "gain": 1.0,
                      "baseline_rate_hz": 0.5, "current_per_hz_pA": 2.5,
                      "sum_mode_scale": 1.0},
        },
        "retinotopy": {
            "mapping": "log_polar", "e0_deg": 0.5, "n_radial": 4,
            "n_angular": 8,
            "fovea_patch": {"enabled": True, "radius_deg": 0.5, "grid": 3},
            "sampling": {"lowpass_before_sampling": True, "sigma_scale": 0.5,
                         "min_sigma_px": 0.5, "interpolation_order": 1},
        },
        "v1": _v1_block(),
        "anatomy": {
            "area_gap_mm": 1.0, "jitter_mm": 0.02,
            "areas": {
                "Retina": {
                    "kind": "retina", "surface_extent_mm": [1.5, 1.5],
                    "neurons_per_layer": {"L_input": 0},
                    "cell_type_fractions": {"L_input": {"retinal_ganglion": 1.0}},
                    "visual_field": {"max_ecc_deg": 5.0, "rf_sigma_deg": 0.25,
                                     "rf_sigma_growth_per_deg": 0.06},
                },
                "LGN": {
                    "kind": "thalamus", "surface_extent_mm": [1.5, 1.5],
                    "neurons_per_layer": {"L_relay": 72},
                    "cell_type_fractions": {"L_relay": {"lgn_relay": 1.0}},
                    "visual_field": {"max_ecc_deg": 5.0, "rf_sigma_deg": 0.3,
                                     "rf_sigma_growth_per_deg": 0.07},
                },
                "V1": _cortical_area(1.0, [2.0, 2.0],
                                     {"L1": 10, "L2": 72, "L3": 72, "L4": 96,
                                      "L5": 48, "L6": 48}, 5.0, 0.35, 0.08,
                                     "국소 수용장, 방향/위상 채널"),
                "V2": _cortical_area(2.0, [1.8, 1.8], counts_small, 5.0, 0.7,
                                     0.14,
                                     "더 넓은 범위의 방향/위상 반응 통합과 재귀 상호작용"),
                "V3": _cortical_area(3.0, [1.6, 1.6], counts_small, 5.0, 1.0,
                                     0.18,
                                     "공간 통합. 운동 기능 평가에는 프레임 시퀀스가 필요하다."),
                "V4": _cortical_area(4.0, [1.6, 1.6], counts_small, 5.0, 1.4,
                                     0.22,
                                     "저주파 색 정보와 형태 정보가 합류하는 지점, 넓어진 수용장"),
                "IT": _cortical_area(5.0, [1.4, 1.4], counts_small, 5.0, 2.2,
                                     0.3,
                                     "집단 활동을 특징 벡터로 제공한다. 한 뉴런이 곧 카테고리 의미가 아니다."),
            },
        },
        "wiring": {"max_total_synapses": 2000000, "rules": rules},
        "learning": {
            "mode": "none",
            "stdp": {"A_plus": 0.008, "A_minus": 0.01, "tau_plus_ms": 20.0,
                     "tau_minus_ms": 20.0, "weight_min": 0.0, "weight_max": 6.0},
            "homeostasis": {"eta_theta": 0.002, "window_ms": 500.0,
                            "theta_min_mV": -60.0, "theta_max_mV": -35.0},
            "rao": {"n_levels": 2, "level_sizes": [32, 16], "sigma": 1.0,
                    "sigma_td": 2.0, "alpha": 0.05, "lambda_u": 0.001,
                    "settle_steps": 40, "r_step": 0.05, "u_step": 0.002},
        },
        "readout": {
            "enabled": True, "source_area": "IT", "source_layer": ["L2", "L3"],
            "classifier": "ridge", "l2": 1.0, "n_classes": 7,
        },
        "recording": {
            "mode": "selected", "backend": "hdf5",
            "selection_criterion": "영역마다 균등 간격으로 뽑은 표본 뉴런의 상태와 그 뉴런 "
                                   "관련 이벤트만 저장한다. 전체 뉴런의 매 스텝 상태는 저장하지 않는다.",
            "selected_areas": ["V1", "V2", "V4", "IT"],
            "state_sample_every_steps": 4,
            "max_events": 12000000, "max_state_samples": 1000000,
            "flush_every_steps": 200,
        },
        "checkpoint": {"enabled": True, "every_samples": 8, "keep_last": 3},
        "experiment": {
            "protocol": "train_dev_test", "n_samples": 1,
            "splits": {"train": 0.6, "dev": 0.2, "test": 0.2,
                       "stratified": True},
            "stimuli": [
                {"kind": "shape_classes", "params": {
                    "size_px": 96,
                    "classes": ["circle", "triangle", "square",
                                "bar_horizontal", "bar_vertical", "cross",
                                "blank"],
                    "n_per_class": 6, "n_variants": 2, "radius_px": 18.0,
                    "contrast": 0.4}},
                {"kind": "color_illumination",
                 "params": {"size_px": 96, "radius_px": 18.0}},
                {"kind": "transform",
                 "params": {"size_px": 96, "label": "square",
                            "radius_px": 15.0}},
            ],
            "limits": {"max_disk_mb": 4096, "max_ram_mb": 8192},
        },
        "validation": _validation_block(),
    }


def _config_megapixel_input() -> dict[str, Any]:
    """configs/megapixel_input.json 과 동일한 내용 (1024x1024 입력 점검)."""
    mp = copy.deepcopy(_config_v1_small())
    mp["meta"] = {
        "name": "megapixel_input",
        "description": "약 100만 화소(1024x1024) 입력과 제한된 피질 표본 수. "
                       "영상 화소 수와 피질 뉴런 수를 분리하는 것을 보이는 설정.",
        "species_assumption": "primate_visual_cortex_model",
        "notes": [
            "100만 화소 입력을 처리한다고 해서 100만 개 뉴런 객체를 만들지 않는다.",
            "피질 표본 수는 retinotopy 의 격자 크기로 정해지며 영상 크기와 독립이다.",
            "주변부 화소 수를 생물학 상수로 고정하지 않는다. 셀 면적과 px_per_deg 에서 계산한 추정값을 기록할 뿐이다.",
        ],
    }
    mp["seeds"] = {"master": 20240404}
    mp["engine"] = _engine_lif(120.0)
    mp["retina"]["image"] = {"max_side_px": 1024, "fov_deg": 30.0,
                             "input_colorspace": "srgb"}
    mp["retina"]["channels"] = {"on_off_split": True, "keep_lowpass_lms": True,
                                "lowpass_sigma_px": 24.0}
    mp["retina"]["dog"] = {"center_sigma_px": 2.0, "surround_sigma_px": 6.0,
                           "truncate": 4.0, "boundary_mode": "reflect",
                           "normalize_each_kernel_to_unit_sum": True}
    mp["retinotopy"] = {
        "mapping": "log_polar", "e0_deg": 0.5, "n_radial": 6, "n_angular": 12,
        "fovea_patch": {"enabled": True, "radius_deg": 0.5, "grid": 3},
        "sampling": {"lowpass_before_sampling": True, "sigma_scale": 0.5,
                     "min_sigma_px": 0.5, "interpolation_order": 1},
    }
    areas = mp["anatomy"]["areas"]
    areas["Retina"]["visual_field"] = {"max_ecc_deg": 14.0, "rf_sigma_deg": 0.25,
                                       "rf_sigma_growth_per_deg": 0.06}
    areas["LGN"]["neurons_per_layer"] = {"L_relay": 96}
    areas["LGN"]["visual_field"] = {"max_ecc_deg": 14.0, "rf_sigma_deg": 0.35,
                                    "rf_sigma_growth_per_deg": 0.08}
    areas["V1"]["neurons_per_layer"] = {"L1": 8, "L2": 48, "L3": 48, "L4": 72,
                                        "L5": 32, "L6": 32}
    areas["V1"]["visual_field"] = {"max_ecc_deg": 14.0, "rf_sigma_deg": 0.45,
                                   "rf_sigma_growth_per_deg": 0.1}
    mp["wiring"]["max_total_synapses"] = 1500000
    mp["recording"] = {
        "mode": "summary", "backend": "hdf5",
        "selection_criterion": "대규모 입력 점검 설정이므로 이벤트는 집계만 저장하고 "
                               "개별 사건은 저장하지 않는다. 뉴런별 상태도 저장하지 않는다. "
                               "재현 범위: 설정·시드·코드 해시로 같은 결과를 다시 만들 수 있으나 "
                               "개별 사건 추적은 불가능하다.",
        "selected_areas": ["V1"], "state_sample_every_steps": 10,
        "max_events": 2000000, "max_state_samples": 100000,
        "flush_every_steps": 200,
    }
    mp["experiment"] = {
        "protocol": "single_pass", "n_samples": 1,
        "stimuli": [
            {"kind": "uniform", "n": 1,
             "params": {"size_px": 1024, "value": 0.5}},
            {"kind": "bar", "n": 2,
             "params": {"size_px": 1024, "length_px": 400.0, "width_px": 12.0}},
            {"kind": "shape_classes", "params": {
                "size_px": 1024, "classes": ["circle", "square"],
                "n_per_class": 1, "n_variants": 1, "radius_px": 160.0}},
        ],
        "limits": {"max_disk_mb": 2048, "max_ram_mb": 8192},
    }
    return mp


#: 내장 설정 이름 -> 원시(해석 전) 설정을 만드는 함수
BUILTIN_CONFIGS: dict[str, Any] = {
    "minimal": _config_minimal,
    "v1_small": _config_v1_small,
    "hierarchy_small": _config_hierarchy_small,
    "megapixel_input": _config_megapixel_input,
}


def load_config_by_name_or_path(name_or_path: str) -> dict[str, Any]:
    """내장 설정 이름 또는 JSON 파일 경로에서 설정을 읽어 해석한다.

    이름(`minimal`, `v1_small`, `hierarchy_small`, `megapixel_input`)을 먼저
    보고, 아니면 파일 경로로 취급한다. 둘 다 아니면 친절한 한국어 오류를 낸다.
    """
    key = str(name_or_path).strip()
    stem = Path(key).stem
    if key in BUILTIN_CONFIGS:
        cfg = resolve_config(BUILTIN_CONFIGS[key]())
        cfg["meta"]["source_path"] = f"<builtin:{key}>"
        return cfg
    if Path(key).is_file():
        return load_config(key)
    if stem in BUILTIN_CONFIGS:
        cfg = resolve_config(BUILTIN_CONFIGS[stem]())
        cfg["meta"]["source_path"] = f"<builtin:{stem}>"
        return cfg
    raise ConfigError(
        f"설정을 찾을 수 없다: {name_or_path!r}\n"
        f"  - 내장 설정 이름: {', '.join(sorted(BUILTIN_CONFIGS))}\n"
        f"  - 또는 JSON 파일의 전체 경로를 주어라 (공백/한글 경로 가능)."
    )
