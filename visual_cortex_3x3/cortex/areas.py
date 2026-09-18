"""areas.py -- 영역 간/영역 내 배선 생성.

명세 6절.

배선 규칙
---------
규칙 하나는 ``(src 영역/층/세포유형) -> (dst 영역/층/세포유형)`` 집합 사이에
시냅스를 만든다. 후보 선택 방식은 세 가지다.

``rf_knn``
    **시야 좌표(deg)** 에서 dst 뉴런의 수용장 중심에 가까운 src 후보 k 개를
    ``cKDTree`` 로 찾는다. 원시 3D 거리만 쓰지 않고 시야 위치 대응과
    영역·층·세포유형 마스크를 함께 쓴다. ``rf_match_sigma_deg`` 로 거리에 따른
    연결 확률 감쇠를 준다. 상위 영역일수록 이 값이 커져 수용장이 넓어진다.
``local_radius``
    **피질 좌표(mm)** 에서 반경 안의 후보를 찾는다 (국소/수평 연결용).
``all_to_all_sampled``
    후보 전체에서 확률로 뽑는다 (작은 설정 전용).

전체 N×N 거리 행렬을 만들지 않는다. 언제나 KD-tree 질의로 후보만 본다.

지연
----
``delay_ms = synaptic_delay_ms + distance_mm / conduction_velocity_mm_per_ms``
직선 3D 거리를 축삭 길이로 쓰는 근사이며 ``use_straight_line_distance`` 로
표시한다. 연속 지연(ms)과 dt 격자로 양자화한 스텝 수를 **둘 다** 기록한다.
최소 지연은 1 스텝이므로 0 지연 재귀가 생기지 않는다.

Gabor 초기 배선
---------------
``gabor_initialized`` 규칙은 dst 의 선호 방향/위상과 src 의 시야 위치로 Gabor
계수를 계산해 초기 가중치를 정한다. **음의 계수를 음의 전도도로 만들지 않는다.**
ON 극성 src 는 계수의 양수부, OFF 극성 src 는 음수부의 절댓값을 쓴다.
억제성 src 는 절댓값을 쓰고 부호는 Dale 유형이 만든다. 이 초기화를 사용하면
"방향 선택성을 학습했다"고 말하지 않는다 (manifest 에 기록된다).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .anatomy import Anatomy
from .events import CapacityExceeded, quantize_delay
from .ids import COMPARTMENT_INDEX
from .synapses import SynapseTable


@dataclass
class WiringReport:
    """배선 결과 집계 (manifest/보고서에 그대로 들어간다)."""

    n_synapses: int
    per_rule: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"n_synapses": self.n_synapses, "per_rule": self.per_rule,
                "notes": self.notes}


def _mask_for(anat: Anatomy, spec: dict[str, Any]) -> np.ndarray:
    a = anat.population.arrays
    ids = anat.ids
    m = a.area_id == ids.areas.id_of(spec["area"])
    if spec["layer"]:
        lm = np.zeros_like(m)
        for lname in spec["layer"]:
            lm |= a.layer_id == ids.layers.id_of(lname)
        m &= lm
    if spec["cell_type"]:
        cm = np.zeros_like(m)
        for cname in spec["cell_type"]:
            cm |= a.cell_type_id == ids.cell_types.id_of(cname)
        m &= cm
    return m


def _sample_weights(rule: dict[str, Any], n: int,
                    rng: np.random.Generator) -> np.ndarray:
    w = rule["weight"]
    if w["dist"] == "constant":
        vals = np.full(n, float(w["median"]), dtype=np.float64)
    elif w["dist"] == "normal":
        vals = rng.normal(float(w["median"]), float(w["sigma"]), size=n)
    else:  # lognormal: median 이 exp(mu) 가 되도록 mu = log(median)
        vals = rng.lognormal(np.log(max(float(w["median"]), 1e-12)),
                             float(w["sigma"]), size=n)
    return np.clip(vals, float(w["min"]), float(w["max"]))


def gabor_coefficient(dx_deg: np.ndarray, dy_deg: np.ndarray,
                      orientation_rad: np.ndarray, phase_rad: np.ndarray,
                      sigma_deg: float, aspect: float,
                      cycles_per_deg: float) -> np.ndarray:
    """원본 시야 Cartesian 좌표에서 평가한 Gabor 계수.

    **왜곡된 로그-극좌표 배열 위의 직선 필터가 아니다.** 각 수용장 안에서
    원본 시야 좌표 차이를 그대로 쓴다.
    """
    ct, st = np.cos(orientation_rad), np.sin(orientation_rad)
    xr = dx_deg * ct + dy_deg * st
    yr = -dx_deg * st + dy_deg * ct
    env = np.exp(-0.5 * ((xr / sigma_deg) ** 2
                         + (yr / (sigma_deg * aspect)) ** 2))
    return env * np.cos(2.0 * np.pi * cycles_per_deg * xr + phase_rad)


def build(cfg: dict[str, Any], anat: Anatomy,
          rng_wiring: np.random.Generator,
          rng_weights: np.random.Generator) -> tuple[SynapseTable, WiringReport]:
    """배선 규칙을 모두 적용해 시냅스 테이블을 만든다.

    Parameters
    ----------
    rng_wiring : 후보 선택/확률 샘플링 스트림
    rng_weights : 초기 가중치 스트림 (분리해 두면 가중치만 바꾸는 대조군이 쉽다)

    Raises
    ------
    CapacityExceeded
        ``wiring.max_total_synapses`` 를 넘으면 조용히 자르지 않고 중단한다.
    """
    a = anat.population.arrays
    ids = anat.ids
    dt = float(cfg["engine"]["dt_ms"])
    min_steps = int(cfg["engine"]["min_delay_steps"])
    rounding = cfg["engine"]["delay_rounding"]
    weight_unit = "nS" if cfg["engine"]["mode"] == "conductance_lif" else "dimensionless"
    table = SynapseTable(len(anat.population), weight_unit)
    report = WiringReport(n_synapses=0)
    max_total = int(cfg["wiring"]["max_total_synapses"])
    allow_self = bool(cfg["wiring"]["allow_self_connection"])
    gi = cfg["v1"]["gabor_init"]

    for r_idx, rule in enumerate(cfg["wiring"]["rules"]):
        if not rule["enabled"]:
            report.per_rule.append({"name": rule["name"], "enabled": False, "n": 0})
            continue
        src_mask = _mask_for(anat, rule["src"])
        dst_mask = _mask_for(anat, rule["dst"])
        src_ids = np.nonzero(src_mask)[0]
        dst_ids = np.nonzero(dst_mask)[0]
        if src_ids.size == 0 or dst_ids.size == 0:
            report.per_rule.append({
                "name": rule["name"], "enabled": True, "n": 0,
                "warning": "src 또는 dst 집합이 비어 있다 (배선 0개)",
                "n_src": int(src_ids.size), "n_dst": int(dst_ids.size)})
            continue

        pairs_src, pairs_dst = _candidate_pairs(rule, a, src_ids, dst_ids, rng_wiring)
        if pairs_src.size == 0:
            report.per_rule.append({"name": rule["name"], "enabled": True, "n": 0,
                                    "warning": "후보가 선택되지 않았다"})
            continue
        if not allow_self:
            keep = pairs_src != pairs_dst
            pairs_src, pairs_dst = pairs_src[keep], pairs_dst[keep]

        # 확률 적용
        if rule["probability"] < 1.0:
            keep = rng_wiring.random(pairs_src.size) < float(rule["probability"])
            pairs_src, pairs_dst = pairs_src[keep], pairs_dst[keep]
        if pairs_src.size == 0:
            report.per_rule.append({"name": rule["name"], "enabled": True, "n": 0,
                                    "warning": "확률 적용 후 후보가 남지 않았다"})
            continue

        if rule["max_synapses_per_target"] > 0:
            pairs_src, pairs_dst = _cap_per_target(
                pairs_src, pairs_dst, int(rule["max_synapses_per_target"]))

        n_new = int(pairs_src.size)
        if table.pending_count() + n_new > max_total:
            raise CapacityExceeded(
                f"배선 규칙 '{rule['name']}' 에서 시냅스 한도를 넘었다: "
                f"{table.pending_count()} + {n_new} > {max_total}. "
                f"wiring.max_total_synapses 를 늘리거나 규칙의 k/probability 를 줄여라."
            )

        # 가중치
        gabor_used = False
        if rule["gabor_initialized"]:
            w = _gabor_weights(a, pairs_src, pairs_dst, gi)
            wcfg = rule["weight"]
            w = np.clip(w * float(wcfg["median"]), float(wcfg["min"]), float(wcfg["max"]))
            gabor_used = True
        else:
            w = _sample_weights(rule, n_new, rng_weights)

        # 지연: 3D 피질 거리 기반 (직선 근사)
        d3 = np.linalg.norm(a.position_mm[pairs_src] - a.position_mm[pairs_dst], axis=1)
        delay_ms = (float(rule["synaptic_delay_ms"])
                    + d3 / float(rule["conduction_velocity_mm_per_ms"]))
        steps = quantize_delay(delay_ms, dt, min_steps, rounding)

        table.add_block(
            src_id=pairs_src, dst_id=pairs_dst,
            src_area=a.area_id[pairs_src], dst_area=a.area_id[pairs_dst],
            target_layer=a.layer_id[pairs_dst],
            target_compartment=COMPARTMENT_INDEX[rule["target_compartment"]],
            receptor_type=ids.receptors.id_of(rule["receptor"]),
            weight=w, base_delay_ms=delay_ms, effective_delay_steps=steps,
            plasticity_rule=ids.plasticity_rules.id_of(rule["plasticity_rule"]),
            src_dale_sign=a.dale_sign[pairs_src], rule_index=r_idx,
        )
        report.n_synapses += n_new
        report.per_rule.append({
            "name": rule["name"], "enabled": True, "n": n_new,
            "n_src": int(src_ids.size), "n_dst": int(dst_ids.size),
            "rule": rule["rule"], "receptor": rule["receptor"],
            "target_compartment": rule["target_compartment"],
            "plasticity_rule": rule["plasticity_rule"],
            "gabor_initialized": gabor_used,
            "weight_min": float(w.min()), "weight_max": float(w.max()),
            "weight_mean": float(w.mean()),
            "delay_ms_min": float(delay_ms.min()), "delay_ms_max": float(delay_ms.max()),
            "delay_steps_min": int(steps.min()), "delay_steps_max": int(steps.max()),
            "straight_line_distance_approximation": bool(rule["use_straight_line_distance"]),
            "mean_synapses_per_target": float(n_new / max(1, dst_ids.size)),
        })

    table.build_indices()
    anat.population.arrays.set_outgoing_index(table.out_ptr, table.out_syn)
    anat.population.attach(synapses=table)

    if any(x.get("gabor_initialized") for x in report.per_rule):
        report.notes.append(
            "일부 규칙의 초기 가중치를 Gabor 모양으로 정했다. 이 경로의 방향 선택성을 "
            "'학습되었다'고 보고하지 않는다."
        )
    report.notes.append(
        "지연은 직선 3D 거리를 축삭 길이로 쓰는 근사를 포함한다. 연속 지연(ms)과 "
        "양자화 스텝을 모두 저장했다."
    )
    return table, report


def _candidate_pairs(rule: dict[str, Any], a: Any, src_ids: np.ndarray,
                     dst_ids: np.ndarray,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """규칙별 (src, dst) 후보 쌍. 전체 거리 행렬을 만들지 않는다."""
    kind = rule["rule"]
    if kind == "all_to_all_sampled":
        s = np.repeat(src_ids, dst_ids.size)
        d = np.tile(dst_ids, src_ids.size)
        return s, d

    if kind == "local_radius":
        pts_src = a.position_mm[src_ids]
        pts_dst = a.position_mm[dst_ids]
        tree = cKDTree(pts_src)
        neigh = tree.query_ball_point(pts_dst, r=float(rule["radius_mm"]))
        s_list, d_list = [], []
        for j, cand in enumerate(neigh):
            if not cand:
                continue
            c = np.asarray(cand, dtype=np.int64)
            if c.size > rule["k"] > 0:
                c = rng.choice(c, size=int(rule["k"]), replace=False)
            s_list.append(src_ids[c])
            d_list.append(np.full(c.size, dst_ids[j], dtype=np.int64))
        if not s_list:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(s_list), np.concatenate(d_list)

    # rf_knn: 시야 좌표(deg) 에서 k 최근접 + 거리 감쇠 확률
    vf_src = a.visual_field_xy_deg[src_ids]
    vf_dst = a.visual_field_xy_deg[dst_ids]
    finite = np.isfinite(vf_src).all(axis=1)
    if not finite.any():
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    src_ids = src_ids[finite]
    vf_src = vf_src[finite]
    tree = cKDTree(vf_src)
    k = min(int(rule["k"]), src_ids.size)
    dist, idx = tree.query(vf_dst, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    sigma = float(rule["rf_match_sigma_deg"])
    p = np.exp(-0.5 * (dist / max(sigma, 1e-9)) ** 2)
    keep = rng.random(p.shape) < p
    rows, cols = np.nonzero(keep)
    return src_ids[idx[rows, cols]], dst_ids[rows]


def _cap_per_target(src: np.ndarray, dst: np.ndarray, cap: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """dst 당 시냅스 수를 cap 으로 제한한다 (앞쪽 우선, 결정적)."""
    order = np.argsort(dst, kind="stable")
    s, d = src[order], dst[order]
    keep = np.zeros(d.size, dtype=bool)
    if d.size == 0:
        return s, d
    start = 0
    for i in range(1, d.size + 1):
        if i == d.size or d[i] != d[start]:
            keep[start: min(i, start + cap)] = True
            start = i
    return s[keep], d[keep]


def _gabor_weights(a: Any, src: np.ndarray, dst: np.ndarray,
                   gi: dict[str, Any]) -> np.ndarray:
    """Gabor 계수에서 **비음수** 가중치 크기를 만든다.

    음의 계수를 음의 전도도로 만들지 않는다:

    * src 가 ON 극성(``on_off=+1``)  -> ``max(coef, 0)``
    * src 가 OFF 극성(``on_off=-1``) -> ``max(-coef, 0)``
    * 극성이 없는 src               -> ``|coef|`` (부호는 Dale 유형이 만든다)
    """
    dx = a.visual_field_xy_deg[src, 0] - a.visual_field_xy_deg[dst, 0]
    dy = a.visual_field_xy_deg[src, 1] - a.visual_field_xy_deg[dst, 1]
    ori = np.nan_to_num(a.pref_orientation_rad[dst], nan=0.0)
    ph = np.nan_to_num(a.pref_phase_rad[dst], nan=0.0)
    coef = gabor_coefficient(dx, dy, ori, ph, float(gi["sigma_deg"]),
                             float(gi["aspect"]), float(gi["cycles_per_deg"]))
    pol = a.on_off[src]
    w = np.abs(coef)
    w = np.where(pol > 0, np.maximum(coef, 0.0), w)
    w = np.where(pol < 0, np.maximum(-coef, 0.0), w)
    return w


def connectivity_summary(cfg: dict[str, Any], anat: Anatomy,
                         table: SynapseTable) -> dict[str, Any]:
    """층·세포유형·영역별 연결 집계 (검증 9번).

    "세포 유형/층 이름만 존재하는 구현"을 잡기 위해, 각 규칙이 실제로 만든
    시냅스 수와 층별 in/out degree 를 함께 낸다.
    """
    a = anat.population.arrays
    ids = anat.ids
    out: dict[str, Any] = {"synapses": table.summary(), "by_pathway": {}, "degree": {}}
    if not table.built or table.n_synapses == 0:
        out["warning"] = "시냅스가 하나도 만들어지지 않았다"
        return out
    key = (a.area_id[table.src_id].astype(np.int64) * 10_000
           + a.layer_id[table.src_id].astype(np.int64) * 100
           + a.area_id[table.dst_id].astype(np.int64))
    uniq, counts = np.unique(key, return_counts=True)
    for k, c in zip(uniq, counts):
        sa, sl, da = int(k // 10_000), int((k // 100) % 100), int(k % 100)
        name = (f"{ids.areas.name_of(sa)}/{ids.layers.name_of(sl)}"
                f" -> {ids.areas.name_of(da)}")
        out["by_pathway"][name] = int(c)
    in_deg = np.diff(table.in_ptr)
    out_deg = np.diff(table.out_ptr)
    for aname in ids.areas.names():
        m = a.area_id == ids.areas.id_of(aname)
        if not m.any():
            continue
        out["degree"][aname] = {
            "n_neurons": int(m.sum()),
            "in_degree_mean": float(in_deg[m].mean()),
            "in_degree_max": int(in_deg[m].max()),
            "out_degree_mean": float(out_deg[m].mean()),
            "out_degree_max": int(out_deg[m].max()),
            "n_isolated_no_input": int(np.count_nonzero(in_deg[m] == 0)),
            "n_isolated_no_output": int(np.count_nonzero(out_deg[m] == 0)),
        }
    return out


__all__ = ["WiringReport", "build", "gabor_coefficient", "connectivity_summary"]
