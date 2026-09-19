"""anatomy.py -- 3D 배치, 층, 세포 유형, 방향/안구우세 지도.

명세 6, 7절.

좌표계
------
* 영역별 **표면 좌표** ``(u, v)`` [mm] 와 **깊이** ``d`` [mm] 를 둔다.
* 전역 피질 좌표는 ``x = area_origin_x + u``, ``y = area_origin_y + v``,
  ``z = -depth`` (표면이 z=0, 아래로 음수) 로 만든다. 영역은 ``area_gap_mm``
  간격으로 나란히 배치한다. 이 배치는 **모형 좌표**이며 실제 뇌의 해부학적
  위치가 아니다.
* 피질 표면 ``u`` 축을 로그-극좌표의 ``rho`` 축에, ``v`` 축을 각도 ``theta`` 축에
  대응시킨다. 그 결과 중심시야가 더 넓은 피질 면적을 차지한다(중심시야 확대).
  이것은 널리 쓰이는 V1 사상 모형이며 개별 뇌의 측정 지도가 아니다.

방향 지도와 안구 우세 지도
--------------------------
방향 지도는 **무작위 위상 평면파 중첩**으로 만든다. ``z(u,v) = Σ exp(i(k·r+ψ))``
에서 ``θ = 0.5 * arg z`` 를 쓰면 pinwheel 특이점이 자연히 생긴다 (Rojer &
Schwartz 계열 모형). 뉴런의 **위치가 선호 방향을 결정**하므로 pinwheel 이 실제
배치와 배선에 반영된다. 같은 좌표에 모든 방향 뉴런을 겹쳐 놓지 않는다.

안구 우세 지도는 **별도 속성**이며 ``u`` 방향 줄무늬로 만든다. 방향 지도와
독립이다. 단안 영상을 복제해 넣은 양안 입력은 실제 양안 시차 실험이 아니다.

L1 처리
-------
L1 은 세포체 밀도가 낮은 층으로 두되 **빈 공간이 아니다**. 설정에서 소수의
억제성 뉴런을 두고, 피라미드 세포의 ``apical_depth_mm`` 를 L1 깊이에 놓아
피드백이 apical 구획에 접점을 갖게 한다. L1 을 "독립 정답 계산기"로 쓰지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import derived_retina_neuron_count
from .ids import COMPARTMENT_INDEX, CORTICAL_LAYERS, IdSpace, N_COMPARTMENTS
from .records import NeuronArrays, NeuronPopulation
from .retina import OPPONENT_NAMES
from .retinotopy import SamplingGrid, build_grid, ecc_to_rho, rho_to_ecc


@dataclass
class Anatomy:
    """구축된 해부 구조."""

    population: NeuronPopulation
    ids: IdSpace
    grid: SamplingGrid
    channel_names: list[str]
    retina_neuron_id: np.ndarray           # (n_channels, n_samples) -> neuron_id, 없으면 -1
    area_slices: dict[str, tuple[int, int]] = field(default_factory=dict)
    layer_depth_center_mm: dict[str, dict[str, float]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return len(self.population)


def orientation_map(u: np.ndarray, v: np.ndarray, hypercolumn_mm: float,
                    rng: np.random.Generator, n_waves: int = 12) -> np.ndarray:
    """무작위 위상 평면파 중첩으로 만든 방향 지도. 반환값은 [0, pi).

    pinwheel 특이점이 자연히 생긴다. 이것은 **모형**이며 특정 동물의 측정
    지도가 아니다.
    """
    k = 2.0 * np.pi / float(hypercolumn_mm)
    z = np.zeros(u.shape, dtype=np.complex128)
    for j in range(int(n_waves)):
        phi = np.pi * j / float(n_waves)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        psi = rng.uniform(0.0, 2.0 * np.pi)
        kx, ky = k * np.cos(phi), k * np.sin(phi)
        z += np.exp(1j * (sign * (kx * u + ky * v) + psi))
    theta = 0.5 * np.angle(z)
    return np.mod(theta, np.pi)


def ocular_dominance_map(u: np.ndarray, column_width_mm: float,
                         strength: float) -> np.ndarray:
    """u 방향 줄무늬 안구 우세 지도. 반환값 [-strength, +strength].

    방향 지도와 **독립된 별도 속성**이다.
    """
    if column_width_mm <= 0:
        return np.zeros_like(u)
    return float(strength) * np.tanh(3.0 * np.sin(2.0 * np.pi * u / float(column_width_mm)))


def _layer_depths(area_cfg: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """층 이름 -> (상단 깊이, 하단 깊이) [mm]. 표면에서 아래로 누적."""
    out: dict[str, tuple[float, float]] = {}
    d = float(area_cfg["depth_origin_mm"])
    for lname in CORTICAL_LAYERS:
        t = float(area_cfg["layer_thickness_mm"].get(lname, 0.0))
        out[lname] = (d, d + t)
        d += t
    return out


def build(cfg: dict[str, Any], rng: np.random.Generator) -> Anatomy:
    """설정에서 뉴런 집단과 공간 배치를 만든다.

    Parameters
    ----------
    cfg : 해석된 설정
    rng : ``wiring`` 스트림의 Generator (배치 난수)

    Returns
    -------
    Anatomy

    부작용: 없음 (새 객체를 만들어 반환). RNG 상태는 소비된다.
    """
    ids = IdSpace()
    for name in cfg["receptors"]:
        ids.receptors.add(name)
    for name in cfg["cell_types"]:
        ids.cell_types.add(name)
    for name in sorted(cfg["anatomy"]["areas"]):
        ids.areas.add(name)

    grid = build_grid(cfg)
    ch_cfg = cfg["retina"]["channels"]
    channel_names: list[str] = []
    for base in OPPONENT_NAMES:
        if ch_cfg["on_off_split"]:
            channel_names += [f"{base}_ON", f"{base}_OFF"]
        else:
            channel_names.append(base)
    if ch_cfg["keep_lowpass_lms"]:
        channel_names += ["lowpass_L", "lowpass_M", "lowpass_S"]
    n_channels = len(channel_names)

    # --- 영역별 뉴런 수 확정 -------------------------------------------
    plan: list[tuple[str, str, str, int]] = []   # (area, layer, cell_type, n)
    retina_area: str | None = None
    for aname in sorted(cfg["anatomy"]["areas"]):
        a = cfg["anatomy"]["areas"][aname]
        if a["kind"] == "retina":
            retina_area = aname
            n_total, _ = derived_retina_neuron_count(cfg)
            ct = a["cell_type_fractions"].get("L_input")
            if not ct:
                raise ValueError(
                    f"망막 영역 {aname} 에 cell_type_fractions.L_input 이 필요하다"
                )
            if len(ct) != 1:
                raise ValueError(
                    f"망막 영역 {aname} 은 세포 유형 1개만 허용한다 (현재 {list(ct)})"
                )
            plan.append((aname, "L_input", next(iter(ct)), n_total))
            continue
        for lname in sorted(a["neurons_per_layer"]):
            n = int(a["neurons_per_layer"][lname])
            if n <= 0:
                continue
            fr = a["cell_type_fractions"][lname]
            names = sorted(fr)
            counts = _split_counts(n, [float(fr[k]) for k in names])
            for cname, c in zip(names, counts):
                if c > 0:
                    plan.append((aname, lname, cname, c))

    n_total = int(sum(p[3] for p in plan))
    arrays = NeuronArrays(n_total, n_receptors=len(cfg["receptors"]))
    pop = NeuronPopulation(arrays, ids)

    # --- 영역 표면 오프셋 ------------------------------------------------
    offsets: dict[str, tuple[float, float]] = {}
    x_cursor = 0.0
    gap = float(cfg["anatomy"]["area_gap_mm"])
    for aname in sorted(cfg["anatomy"]["areas"]):
        a = cfg["anatomy"]["areas"][aname]
        ox = float(a["surface_origin_mm"][0]) + x_cursor
        oy = float(a["surface_origin_mm"][1])
        offsets[aname] = (ox, oy)
        x_cursor = ox + float(a["surface_extent_mm"][0]) + gap

    jitter = float(cfg["anatomy"]["jitter_mm"])
    v1cfg = cfg["v1"]
    n_ori = int(v1cfg["n_orientations"])
    ori_step = float(v1cfg["orientation_step_deg"])
    phases = [float(p) for p in v1cfg["phases_rad"]]

    retina_index = np.full((n_channels, grid.n_samples), -1, dtype=np.int64)
    area_slices: dict[str, tuple[int, int]] = {}
    layer_depth_center: dict[str, dict[str, float]] = {}

    cursor = 0
    for aname, lname, cname, n in plan:
        a = cfg["anatomy"]["areas"][aname]
        ct = cfg["cell_types"][cname]
        lo, hi = cursor, cursor + n
        cursor = hi
        sl = slice(lo, hi)
        aid = ids.areas.id_of(aname)
        lid = ids.layers.id_of(lname)
        cid = ids.cell_types.id_of(cname)

        arrays.area_id[sl] = aid
        arrays.layer_id[sl] = lid
        arrays.cell_type_id[sl] = cid
        arrays.dale_sign[sl] = 1 if ct["dale"] == "excitatory" else -1
        arrays.output_gain_P[sl] = float(ct["output_gain_P"])
        arrays.V_reset_mV[sl] = float(ct["V_reset_mV"])
        arrays.t_ref_ms[sl] = float(ct["t_ref_ms"])
        arrays.target_rate_hz[sl] = float(ct["target_rate_hz"])
        arrays.refractory_discards_input[sl] = ct["refractory_input_policy"] == "discard"
        for comp in ct["compartments"]:
            ci = COMPARTMENT_INDEX[comp]
            arrays.has_compartment[sl, ci] = True
            arrays.C_pF[sl, ci] = float(ct["C_pF"][comp])
            arrays.gL_nS[sl, ci] = float(ct["gL_nS"][comp])
            arrays.EL_mV[sl, ci] = float(ct["EL_mV"][comp])
        arrays.g_couple_nS[sl, COMPARTMENT_INDEX["basal"]] = (
            float(ct["g_couple_nS"]["soma_basal"])
            if "basal" in ct["compartments"] else 0.0)
        arrays.g_couple_nS[sl, COMPARTMENT_INDEX["apical"]] = (
            float(ct["g_couple_nS"]["soma_apical"])
            if "apical" in ct["compartments"] else 0.0)
        arrays.V_mV[sl, :] = arrays.EL_mV[sl, :]

        if cfg["engine"]["mode"] == "conductance_lif":
            arrays.threshold[sl] = float(ct["V_th_mV"])
        else:
            arrays.threshold[sl] = float(ct["sum_threshold_theta"])

        # --- 공간 배치 --------------------------------------------------
        ox, oy = offsets[aname]
        ext_u, ext_v = (float(a["surface_extent_mm"][0]),
                        float(a["surface_extent_mm"][1]))
        if a["kind"] == "retina":
            # 망막 뉴런은 (채널, 시야 샘플) 격자에 1:1 대응한다.
            s_idx = np.tile(np.arange(grid.n_samples), n_channels)
            c_idx = np.repeat(np.arange(n_channels), grid.n_samples)
            if s_idx.size != n:
                raise ValueError(
                    f"망막 뉴런 수 불일치: 유도값 {s_idx.size} vs 배치 {n}")
            retina_index[c_idx, s_idx] = np.arange(lo, hi)
            arrays.visual_field_xy_deg[sl, 0] = grid.x_deg[s_idx]
            arrays.visual_field_xy_deg[sl, 1] = grid.y_deg[s_idx]
            arrays.rf_sigma_deg[sl] = grid.rf_sigma_deg[s_idx]
            arrays.channel_id[sl] = c_idx
            on_off = np.zeros(n, dtype=np.int8)
            for k, nm in enumerate(channel_names):
                if nm.endswith("_ON"):
                    on_off[c_idx == k] = 1
                elif nm.endswith("_OFF"):
                    on_off[c_idx == k] = -1
            arrays.on_off[sl] = on_off
            # 망막은 시야 좌표를 그대로 표면 좌표로 쓴다 (피질 사상 아님).
            u = (grid.x_deg[s_idx] - grid.x_deg.min())
            v = (grid.y_deg[s_idx] - grid.y_deg.min())
            scale = max(1e-9, max(u.max() if u.size else 1.0, v.max() if v.size else 1.0))
            u = u / scale * ext_u
            v = v / scale * ext_v
            depth = np.zeros(n)
        else:
            u = rng.uniform(0.0, ext_u, size=n)
            v = rng.uniform(0.0, ext_v, size=n)
            if a["kind"] == "cortex":
                depths = _layer_depths(a)
                d0, d1 = depths[lname]
                depth = rng.uniform(d0, d1, size=n)
                layer_depth_center.setdefault(aname, {})[lname] = float((d0 + d1) / 2.0)
                arrays.apical_depth_mm[sl] = float(depths["L1"][0] + depths["L1"][1]) / 2.0
            else:  # thalamus
                depth = rng.uniform(0.0, 0.3, size=n)
            # 표면 -> 시야 (로그-극좌표 역사상): 중심시야 확대가 생긴다.
            max_ecc = float(a["visual_field"]["max_ecc_deg"])
            e0 = float(cfg["retinotopy"]["e0_deg"])
            rho_max = ecc_to_rho(max_ecc, e0)
            rho = rho_max * (u / max(ext_u, 1e-9))
            theta = 2.0 * np.pi * (v / max(ext_v, 1e-9))
            ecc = rho_to_ecc(rho, e0)
            arrays.visual_field_xy_deg[sl, 0] = ecc * np.cos(theta)
            arrays.visual_field_xy_deg[sl, 1] = ecc * np.sin(theta)
            arrays.rf_sigma_deg[sl] = (
                float(a["visual_field"]["rf_sigma_deg"])
                + float(a["visual_field"]["rf_sigma_growth_per_deg"]) * ecc)

        arrays.surface_uv_mm[sl, 0] = u
        arrays.surface_uv_mm[sl, 1] = v
        arrays.position_mm[sl, 0] = ox + u + rng.normal(0.0, jitter, size=n)
        arrays.position_mm[sl, 1] = oy + v + rng.normal(0.0, jitter, size=n)
        arrays.position_mm[sl, 2] = -depth

        # --- V1 방향/위상/안구 우세 -------------------------------------
        if aname == "V1" and a["kind"] == "cortex":
            pw = v1cfg["pinwheel"]
            if pw["enabled"]:
                theta_map = orientation_map(u, v, float(pw["hypercolumn_mm"]), rng)
            else:
                theta_map = rng.uniform(0.0, np.pi, size=n)
            # 12개 방향 구간으로 이산화 (계산을 위한 이산화이며, 피질이 정확히
            # 24개 균일 채널이라는 주장이 아니다)
            step_rad = np.deg2rad(ori_step)
            bin_idx = np.rint(theta_map / step_rad).astype(int) % n_ori
            is_exc = ct["dale"] == "excitatory"
            if lname in ("L2", "L3", "L4", "L5", "L6") and is_exc:
                arrays.pref_orientation_rad[sl] = bin_idx * step_rad
                hc = float(pw["hypercolumn_mm"]) if pw["enabled"] else 1.0
                fu = np.mod(u, hc) / hc
                fv = np.mod(v, hc) / hc
                arrays.hypercolumn_uv[sl, 0] = fu
                arrays.hypercolumn_uv[sl, 1] = fv
                p_idx = np.floor(fv * len(phases)).astype(int) % len(phases)
                arrays.pref_phase_rad[sl] = np.asarray(phases)[p_idx]
            od = v1cfg["ocular_dominance"]
            if od["enabled"]:
                arrays.ocular_dominance[sl] = ocular_dominance_map(
                    u, float(od["column_width_mm"]), float(od["strength"]))

        s0, s1 = area_slices.get(aname, (lo, hi))
        area_slices[aname] = (min(s0, lo), max(s1, hi))

    meta = {
        "n_neurons": n_total,
        "n_channels": n_channels,
        "channel_names": channel_names,
        "retina_area": retina_area,
        "area_offsets_mm": {k: list(v) for k, v in offsets.items()},
        "layer_thickness_total_mm": {
            aname: float(sum(a["layer_thickness_mm"].get(l, 0.0) for l in CORTICAL_LAYERS))
            for aname, a in cfg["anatomy"]["areas"].items() if a["kind"] == "cortex"
        },
        "grid_summary": grid.summary(),
        "plan": [{"area": p[0], "layer": p[1], "cell_type": p[2], "n": p[3]} for p in plan],
        "orientation_map_model_ko": (
            "무작위 위상 평면파 중첩 모형 (Rojer & Schwartz 계열). 뉴런 위치가 선호 "
            "방향을 결정하므로 pinwheel 이 배선에 반영된다. 특정 동물의 측정 지도가 아니다."
        ),
        "cortical_mapping_note_ko": (
            "피질 표면 u 축을 rho, v 축을 theta 에 대응시킨 로그-극좌표 사상 모형이다. "
            "개별 뇌의 측정 지도가 아니다."
        ),
        "l1_note_ko": (
            "L1 은 세포체 밀도가 낮은 층으로 두고, 피라미드의 apical_depth_mm 를 "
            "L1 깊이에 두어 피드백이 apical 구획에 접점을 갖게 했다. 독립 정답 계산기가 아니다."
        ),
    }
    return Anatomy(
        population=pop, ids=ids, grid=grid, channel_names=channel_names,
        retina_neuron_id=retina_index, area_slices=area_slices,
        layer_depth_center_mm=layer_depth_center, meta=meta,
    )


def _split_counts(total: int, fractions: list[float]) -> list[int]:
    """비율을 정수 개수로 나눈다 (최대 잔여법, 결정적)."""
    raw = [total * f for f in fractions]
    base = [int(np.floor(r)) for r in raw]
    rest = total - sum(base)
    order = np.argsort([-(r - b) for r, b in zip(raw, base)], kind="stable")
    for i in range(rest):
        base[int(order[i % len(base)])] += 1
    return base


def layer_summary(anat: Anatomy) -> dict[str, Any]:
    """층별 깊이·세포 유형·개수 집계 (검증 9번)."""
    a = anat.population.arrays
    ids = anat.ids
    out: dict[str, Any] = {"counts": anat.population.counts_by_area_layer_type(),
                           "depth_mm": {}}
    for i_area, aname in enumerate(ids.areas.names()):
        for lname in ids.layers.names():
            m = (a.area_id == i_area) & (a.layer_id == ids.layers.id_of(lname))
            if not m.any():
                continue
            z = -a.position_mm[m, 2]
            out["depth_mm"][f"{aname}/{lname}"] = {
                "n": int(m.sum()),
                "depth_min": float(z.min()),
                "depth_max": float(z.max()),
                "depth_mean": float(z.mean()),
            }
    return out


__all__ = ["Anatomy", "build", "orientation_map", "ocular_dominance_map",
           "layer_summary"]
