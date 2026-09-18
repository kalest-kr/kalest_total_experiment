# ============================================================================
# 섹션: builtin configs  —  내장 설정 4종 (JSON 파일 없이 실행 가능)
#   (원래 파일: configs/minimal.json, v1_small.json, hierarchy_small.json,
#               megapixel_input.json)
# ============================================================================
"""내장 설정.

원래 저장소의 `configs/*.json` 과 **해석 결과가 동일**하도록 코드로 구성했다.
(동일성은 `python cortex_all_in_one.py selftest` 로 확인할 수 있다. JSON 파일이
옆에 있을 때만 비교하며, 없으면 건너뛴다.)

값은 전부 **모형 파라미터**다. 출처가 있는 측정값이 아니다
(BIOLOGY_AND_ASSUMPTIONS.md 참조).
"""

_CELL_TYPES: dict[str, Any] = {
    "retinal_ganglion": {
        "dale": "excitatory", "compartments": ["soma"],
        "C_pF": {"soma": 80.0, "basal": 100.0, "apical": 100.0},
        "gL_nS": {"soma": 6.0, "basal": 5.0, "apical": 5.0},
        "EL_mV": {"soma": -65.0, "basal": -70.0, "apical": -70.0},
        "V_th_mV": -50.0, "V_reset_mV": -62.0, "t_ref_ms": 1.5,
        "target_rate_hz": 10.0, "sum_threshold_theta": 0.5,
    },
    "lgn_relay": {
        "dale": "excitatory", "compartments": ["soma"],
        "C_pF": {"soma": 100.0, "basal": 100.0, "apical": 100.0},
        "gL_nS": {"soma": 7.0, "basal": 5.0, "apical": 5.0},
        "EL_mV": {"soma": -68.0, "basal": -70.0, "apical": -70.0},
        "V_th_mV": -50.0, "V_reset_mV": -63.0, "t_ref_ms": 2.0,
        "target_rate_hz": 8.0, "sum_threshold_theta": 0.6,
    },
    "spiny_stellate": {
        "dale": "excitatory", "compartments": ["soma", "basal"],
        "C_pF": {"soma": 150.0, "basal": 90.0, "apical": 90.0},
        "gL_nS": {"soma": 8.0, "basal": 5.0, "apical": 5.0},
        "EL_mV": {"soma": -70.0, "basal": -70.0, "apical": -70.0},
        "g_couple_nS": {"soma_basal": 10.0, "soma_apical": 0.0},
        "V_th_mV": -50.0, "V_reset_mV": -65.0, "t_ref_ms": 2.0,
        "target_rate_hz": 5.0, "sum_threshold_theta": 1.0,
    },
    "pyramidal": {
        "dale": "excitatory", "compartments": ["soma", "basal", "apical"],
        "C_pF": {"soma": 200.0, "basal": 120.0, "apical": 120.0},
        "gL_nS": {"soma": 10.0, "basal": 6.0, "apical": 6.0},
        "EL_mV": {"soma": -70.0, "basal": -70.0, "apical": -70.0},
        "g_couple_nS": {"soma_basal": 9.0, "soma_apical": 4.0},
        "V_th_mV": -50.0, "V_reset_mV": -65.0, "t_ref_ms": 2.5,
        "target_rate_hz": 4.0, "sum_threshold_theta": 1.0,
    },
    "pv_basket": {
        "dale": "inhibitory", "compartments": ["soma"],
        "C_pF": {"soma": 100.0, "basal": 80.0, "apical": 80.0},
        "gL_nS": {"soma": 10.0, "basal": 5.0, "apical": 5.0},
        "EL_mV": {"soma": -65.0, "basal": -70.0, "apical": -70.0},
        "V_th_mV": -48.0, "V_reset_mV": -60.0, "t_ref_ms": 1.0,
        "target_rate_hz": 12.0, "sum_threshold_theta": 0.8,
    },
    "sst_martinotti": {
        "dale": "inhibitory", "compartments": ["soma"],
        "C_pF": {"soma": 110.0, "basal": 80.0, "apical": 80.0},
        "gL_nS": {"soma": 8.0, "basal": 5.0, "apical": 5.0},
        "EL_mV": {"soma": -66.0, "basal": -70.0, "apical": -70.0},
        "V_th_mV": -50.0, "V_reset_mV": -62.0, "t_ref_ms": 2.0,
        "target_rate_hz": 6.0, "sum_threshold_theta": 0.9,
    },
    "l1_inhibitory": {
        "dale": "inhibitory", "compartments": ["soma"],
        "C_pF": {"soma": 90.0, "basal": 80.0, "apical": 80.0},
        "gL_nS": {"soma": 8.0, "basal": 5.0, "apical": 5.0},
        "EL_mV": {"soma": -66.0, "basal": -70.0, "apical": -70.0},
        "V_th_mV": -50.0, "V_reset_mV": -62.0, "t_ref_ms": 2.0,
        "target_rate_hz": 5.0, "sum_threshold_theta": 0.9,
    },
}

_FF_NOTE = ("상향은 주로 상부층(L2/3) 기원, 중간층(L4) 표적이라는 집단 수준 경향을 "
            "쓴다. 개별 시냅스의 완전한 목록이 아니며 기원/종말층 혼합을 허용한다 "
            "(Markov 등, Anatomy of hierarchy).")
_FB_NOTE = ("하향은 L4 밖(주로 L1/apical, L5)을 표적으로 한다. 같은 근거의 집단 수준 "
            "경향이며 예외를 금지하는 규칙이 아니다.")


def _rule(name, src_area, src_layers, dst_area, dst_layers, *, comp="basal",
          receptor="AMPA", kind="rf_knn", k=8, radius=0.4, sigma=0.5, prob=0.5,
          median=1.0, wsigma=0.3, wmax=6.0, vel=0.3, delay=0.8,
          src_types=None, dst_types=None, plastic="none", gabor=False,
          enabled=None, note="") -> dict[str, Any]:
    """배선 규칙 하나를 만든다 (설정 dict). 해석은 resolve_config 가 한다."""
    r: dict[str, Any] = {"name": name}
    if enabled is not None:
        r["enabled"] = bool(enabled)
    r.update({
        "src": {"area": src_area, "layer": list(src_layers),
                "cell_type": list(src_types or [])},
        "dst": {"area": dst_area, "layer": list(dst_layers),
                "cell_type": list(dst_types or [])},
        "target_compartment": comp, "receptor": receptor, "rule": kind,
        "k": k, "radius_mm": radius, "rf_match_sigma_deg": sigma,
        "probability": prob,
        "weight": {"dist": "lognormal", "median": median, "sigma": wsigma,
                   "min": 0.0, "max": wmax},
        "conduction_velocity_mm_per_ms": vel, "synaptic_delay_ms": delay,
        "plasticity_rule": plastic, "gabor_initialized": gabor, "note": note,
    })
    return r


def _cortical_area(level, extent, counts, max_ecc, rf_sigma, growth,
                   note="") -> dict[str, Any]:
    return {
        "kind": "cortex", "hierarchy_level": level,
        "surface_extent_mm": list(extent),
        "layer_thickness_mm": {"L1": 0.10, "L2": 0.22, "L3": 0.33,
                               "L4": 0.30, "L5": 0.38, "L6": 0.37},
        "neurons_per_layer": dict(counts),
        "cell_type_fractions": {
            "L1": {"l1_inhibitory": 1.0},
            "L2": {"pyramidal": 0.72, "pv_basket": 0.18, "sst_martinotti": 0.10},
            "L3": {"pyramidal": 0.72, "pv_basket": 0.18, "sst_martinotti": 0.10},
            "L4": {"spiny_stellate": 0.72, "pv_basket": 0.20, "sst_martinotti": 0.08},
            "L5": {"pyramidal": 0.82, "pv_basket": 0.18},
            "L6": {"pyramidal": 0.82, "pv_basket": 0.18},
        },
        "visual_field": {"max_ecc_deg": max_ecc, "rf_sigma_deg": rf_sigma,
                         "rf_sigma_growth_per_deg": growth},
        "notes": note,
    }


def _local_microcircuit(area: str, scale: float = 1.0,
                        enabled=None) -> list[dict[str, Any]]:
    """한 피질 영역 내부의 층간·억제·재귀 배선 (모형 규칙)."""
    e = enabled
    return [
        _rule(f"{area}_L4->L2L3", area, ["L4"], area, ["L2", "L3"],
              kind="local_radius", radius=0.45, k=8, prob=0.5,
              median=1.1 * scale, vel=0.2, delay=0.8,
              src_types=["spiny_stellate"], dst_types=["pyramidal"],
              plastic="stdp", enabled=e, note="L4 -> L2/3 상행"),
        _rule(f"{area}_L4->L4_PV", area, ["L4"], area, ["L4"],
              comp="soma", kind="local_radius", radius=0.35, k=6, prob=0.55,
              median=1.0 * scale, vel=0.2, delay=0.6, enabled=e,
              src_types=["spiny_stellate"], dst_types=["pv_basket"]),
        _rule(f"{area}_L4_PV->L4", area, ["L4"], area, ["L4"],
              comp="soma", receptor="GABA_A", kind="local_radius",
              radius=0.35, k=6, prob=0.65, median=1.4 * scale, vel=0.2,
              delay=0.5, src_types=["pv_basket"], dst_types=["spiny_stellate"],
              enabled=e, note="PV 는 soma 표적 억제"),
        _rule(f"{area}_L2L3_lateral", area, ["L2", "L3"], area, ["L2", "L3"],
              kind="local_radius", radius=0.6, k=6, prob=0.3,
              median=0.7 * scale, vel=0.15, delay=1.4,
              src_types=["pyramidal"], dst_types=["pyramidal"],
              plastic="stdp", enabled=e, note="수평 재귀 연결"),
        _rule(f"{area}_L2L3->PV", area, ["L2", "L3"], area, ["L2", "L3"],
              comp="soma", kind="local_radius", radius=0.5, k=5, prob=0.5,
              median=0.9 * scale, vel=0.2, delay=0.6, enabled=e,
              src_types=["pyramidal"], dst_types=["pv_basket"]),
        _rule(f"{area}_PV->L2L3", area, ["L2", "L3"], area, ["L2", "L3"],
              comp="soma", receptor="GABA_A", kind="local_radius",
              radius=0.5, k=6, prob=0.6, median=1.5 * scale, vel=0.2,
              delay=0.5, src_types=["pv_basket"], dst_types=["pyramidal"],
              enabled=e),
        _rule(f"{area}_L2L3->SST", area, ["L2", "L3"], area, ["L2", "L3"],
              comp="soma", kind="local_radius", radius=0.5, k=4, prob=0.4,
              median=0.7 * scale, vel=0.2, delay=0.7, enabled=e,
              src_types=["pyramidal"], dst_types=["sst_martinotti"]),
        _rule(f"{area}_SST->apical", area, ["L2", "L3"], area, ["L2", "L3"],
              comp="apical", receptor="GABA_A", kind="local_radius",
              radius=0.7, k=5, prob=0.5, median=1.0 * scale, vel=0.15,
              delay=0.8, src_types=["sst_martinotti"], dst_types=["pyramidal"],
              enabled=e, note="SST 는 첨단수상돌기(apical) 표적 억제"),
        _rule(f"{area}_L2L3->L5", area, ["L2", "L3"], area, ["L5"],
              kind="local_radius", radius=0.6, k=5, prob=0.45,
              median=0.9 * scale, vel=0.2, delay=0.9, enabled=e,
              src_types=["pyramidal"], dst_types=["pyramidal"]),
        _rule(f"{area}_L5->L6", area, ["L5"], area, ["L6"],
              kind="local_radius", radius=0.6, k=4, prob=0.45,
              median=0.8 * scale, vel=0.2, delay=0.9, enabled=e,
              src_types=["pyramidal"], dst_types=["pyramidal"]),
        _rule(f"{area}_L6->L4", area, ["L6"], area, ["L4"],
              kind="local_radius", radius=0.6, k=4, prob=0.35,
              median=0.5 * scale, vel=0.2, delay=1.1, enabled=e,
              src_types=["pyramidal"], dst_types=["spiny_stellate"],
              note="층내 피드백 (L6 -> L4)"),
        _rule(f"{area}_L5->L1_inh", area, ["L5"], area, ["L1"],
              comp="soma", kind="local_radius", radius=0.9, k=3, prob=0.45,
              median=0.7 * scale, vel=0.2, delay=1.0, enabled=e,
              src_types=["pyramidal"], dst_types=["l1_inhibitory"],
              note="L1 은 세포체 밀도가 낮지만 비어 있지 않다"),
        _rule(f"{area}_L1_inh->apical", area, ["L1"], area, ["L2", "L3", "L5"],
              comp="apical", receptor="GABA_A", kind="local_radius",
              radius=0.9, k=6, prob=0.5, median=0.9 * scale, vel=0.15,
              delay=0.8, src_types=["l1_inhibitory"], dst_types=["pyramidal"],
              enabled=e, note="L1 억제 -> 피라미드 apical 구획"),
    ]


def _feedforward(a: str, b: str, *, sigma: float, prob: float = 0.35,
                 median: float = 1.0, k: int = 10, delay: float = 2.0,
                 plastic: str = "stdp") -> list[dict[str, Any]]:
    """a(하위) -> b(상위) 상향 경로."""
    return [
        _rule(f"{a}->{b}_FF_main", a, ["L2", "L3"], b, ["L4"], k=k, sigma=sigma,
              prob=prob, median=median, vel=0.5, delay=delay, enabled=True,
              src_types=["pyramidal"], dst_types=["spiny_stellate"],
              plastic=plastic, note=_FF_NOTE),
        _rule(f"{a}->{b}_FF_mixed", a, ["L5"], b, ["L4", "L3"], k=5,
              sigma=sigma * 1.2, prob=prob * 0.4, median=median * 0.5, vel=0.5,
              delay=delay * 1.1, enabled=True, src_types=["pyramidal"],
              dst_types=["spiny_stellate", "pyramidal"],
              note="기원/종말층 혼합을 허용하는 소수 경로."),
        _rule(f"{a}->{b}_FF_inh", a, ["L2", "L3"], b, ["L4"], comp="soma", k=6,
              sigma=sigma, prob=prob * 0.6, median=median * 0.8, vel=0.5,
              delay=delay, enabled=True, src_types=["pyramidal"],
              dst_types=["pv_basket"], note="상향 경로에 딸린 전방향 억제."),
    ]


def _feedback(a: str, b: str, *, sigma: float, prob: float = 0.25,
              median: float = 0.5, k: int = 8,
              delay: float = 3.0) -> list[dict[str, Any]]:
    """b(상위) -> a(하위) 하향. L4 를 피해 apical/L5 를 표적으로 한다."""
    return [
        _rule(f"{b}->{a}_FB_apical", b, ["L5", "L6"], a, ["L2", "L3"],
              comp="apical", k=k, sigma=sigma, prob=prob, median=median,
              vel=0.5, delay=delay, enabled=True, src_types=["pyramidal"],
              dst_types=["pyramidal"], note=_FB_NOTE),
        _rule(f"{b}->{a}_FB_L1", b, ["L5", "L6"], a, ["L1"], comp="soma", k=4,
              sigma=sigma * 1.2, prob=prob * 0.6, median=median * 0.8, vel=0.5,
              delay=delay, enabled=True, src_types=["pyramidal"],
              dst_types=["l1_inhibitory"], note="L1 억제 뉴런을 경유하는 하향 경로."),
        _rule(f"{b}->{a}_FB_L5", b, ["L5", "L6"], a, ["L5"], k=4,
              sigma=sigma * 1.2, prob=prob * 0.6, median=median * 0.7, vel=0.5,
              delay=delay * 1.1, enabled=True, src_types=["pyramidal"],
              dst_types=["pyramidal"], note=_FB_NOTE),
    ]
