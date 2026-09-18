"""retinotopy.py -- 불균일(로그-극좌표) 샘플링과 시야<->영상 좌표 변환.

명세 5.2절.

사상
----
``rho = log(1 + ecc / e0)`` 와 그 역 ``ecc = e0 * (exp(rho) - 1)``.
로그를 두 번 적용하지 않는다.

편심도 단위는 **시야각(deg)** 으로 고정한다. 픽셀↔시야각 변환은 영상 긴 변의
픽셀 수와 ``fov_deg`` 로 계산한다::

    px_per_deg = max(H, W) / fov_deg

세 좌표계를 **분리**한다.

* ``image_px``        : 원본 영상 픽셀 (H, W)
* ``visual_field_xy`` : 시선 중심 기준 시야각 [deg]
* ``cortical_xyz_mm`` : 피질 모형 좌표 [mm] (:mod:`cortex.anatomy` 가 부여)

서로 다른 공간의 거리를 섞어 쓰지 않는다.

중요한 구현 선택
----------------
* **왜곡된 로그-극좌표 배열 위에서 직선 필터를 적용하지 않는다.** 샘플 위치를
  원본 영상 좌표로 계산한 뒤 원본 영상에서 값을 읽는다. 방향 필터도 원본
  Cartesian 좌표에서 정의한다 (:mod:`cortex.areas` 의 Gabor 초기화).
* 중심 특이점은 반경 ``fovea_patch.radius_deg`` 안쪽을 작은 Cartesian 격자로
  대체해 처리한다 (채택한 근사이며 명시한다).
* 각도 경계의 주기성은 샘플 위치를 각도에서 직접 계산하므로 자연히 처리된다.
* 편심도가 커질수록 셀이 커지고, 샘플링 전에 셀 크기에 맞춘 저역통과를 적용해
  aliasing 을 줄인다. 반경(deg), 면적(deg^2), 픽셀 개수는 각각 따로 기록한다.
  "주변부는 항상 10만 화소" 같은 수치를 생물학 상수로 고정하지 않는다.
* 영상 밖 샘플은 ``valid`` 마스크로 표시하고 값은 0 으로 둔다. 개수를 기록한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage


def px_per_deg(height: int, width: int, fov_deg: float) -> float:
    """영상 긴 변이 ``fov_deg`` 를 덮는다고 보고 픽셀/도 를 계산한다."""
    return float(max(int(height), int(width))) / float(fov_deg)


def ecc_to_rho(ecc_deg: np.ndarray | float, e0_deg: float) -> np.ndarray:
    """rho = log(1 + ecc/e0). ecc >= 0, e0 > 0."""
    return np.log1p(np.asarray(ecc_deg, dtype=np.float64) / float(e0_deg))


def rho_to_ecc(rho: np.ndarray | float, e0_deg: float) -> np.ndarray:
    """역변환 ecc = e0*(exp(rho)-1)."""
    return float(e0_deg) * np.expm1(np.asarray(rho, dtype=np.float64))


@dataclass
class SamplingGrid:
    """시야 샘플 격자. 모든 배열 길이는 S (샘플 수).

    Attributes
    ----------
    x_deg, y_deg : (S,) float64
        시선 중심 기준 시야 좌표.
    ecc_deg, theta_rad, rho : (S,) float64
    sigma_deg : (S,) float64
        저역통과/수용장 폭 (시야각 기준).
    rf_sigma_deg : (S,) float64
        모형 수용장 표준편차. 편심도에 따라 커진다 (설계값).
    cell_area_deg2 : (S,) float64
        이 샘플이 담당하는 시야 면적 [deg^2].
    kind : (S,) int8
        0 = 중심 Cartesian 패치, 1 = 로그-극좌표, 2 = 균일 대조군
    radial_bin, angular_bin : (S,) int32
        로그-극좌표 샘플의 격자 인덱스. 그 외는 -1.
    """

    x_deg: np.ndarray
    y_deg: np.ndarray
    ecc_deg: np.ndarray
    theta_rad: np.ndarray
    rho: np.ndarray
    sigma_deg: np.ndarray
    rf_sigma_deg: np.ndarray
    cell_area_deg2: np.ndarray
    kind: np.ndarray
    radial_bin: np.ndarray
    angular_bin: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.x_deg.size)

    def to_pixels(self, height: int, width: int, fov_deg: float,
                  gaze_center_px: tuple[float, float] | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """시야 좌표 -> 영상 픽셀 좌표 (col, row) 와 영상 안 여부 마스크."""
        ppd = px_per_deg(height, width, fov_deg)
        if gaze_center_px is None:
            cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        else:
            cx, cy = float(gaze_center_px[0]), float(gaze_center_px[1])
        col = cx + self.x_deg * ppd
        row = cy + self.y_deg * ppd
        valid = (col >= 0) & (col <= width - 1) & (row >= 0) & (row <= height - 1)
        return col, row, valid

    def summary(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_fovea_patch": int(np.count_nonzero(self.kind == 0)),
            "n_logpolar": int(np.count_nonzero(self.kind == 1)),
            "n_uniform": int(np.count_nonzero(self.kind == 2)),
            "ecc_deg_min": float(self.ecc_deg.min()) if self.n_samples else None,
            "ecc_deg_max": float(self.ecc_deg.max()) if self.n_samples else None,
            "sigma_deg_min": float(self.sigma_deg.min()) if self.n_samples else None,
            "sigma_deg_max": float(self.sigma_deg.max()) if self.n_samples else None,
            "rf_sigma_deg_min": float(self.rf_sigma_deg.min()) if self.n_samples else None,
            "rf_sigma_deg_max": float(self.rf_sigma_deg.max()) if self.n_samples else None,
            "cell_area_deg2_total": float(self.cell_area_deg2.sum()) if self.n_samples else None,
            **self.meta,
        }


def x_len(arrays: list[np.ndarray]) -> int:
    """리스트에 쌓인 좌표 배열들의 총 길이."""
    return int(sum(int(a.size) for a in arrays))


def build_grid(cfg: dict[str, Any]) -> SamplingGrid:
    """설정에서 샘플 격자를 만든다 (부작용 없음).

    ``retinotopy.mapping`` 이 ``uniform_control`` 이면 **총 샘플 수를 맞춘**
    균일 격자를 만든다 (명세 5.2 마지막 항목).
    """
    rt = cfg["retinotopy"]
    areas = cfg["anatomy"]["areas"]
    max_ecc = float(max(a["visual_field"]["max_ecc_deg"] for a in areas.values()))
    # 수용장 크기 규칙은 **망막 영역**의 설정을 쓴다 (없으면 이름 순 첫 영역).
    retina_areas = [a for a in areas.values() if a["kind"] == "retina"]
    rf_source = retina_areas[0] if retina_areas else areas[sorted(areas)[0]]
    e0 = float(rt["e0_deg"])
    n_r = int(rt["n_radial"])
    n_a = int(rt["n_angular"])
    fov = rt["fovea_patch"]
    sig = rt["sampling"]

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sig_deg: list[np.ndarray] = []
    area: list[np.ndarray] = []
    kinds: list[np.ndarray] = []
    rbin: list[np.ndarray] = []
    abin: list[np.ndarray] = []

    inner_deg = 0.0
    if fov["enabled"]:
        inner_deg = float(fov["radius_deg"])
        g = int(fov["grid"])
        step = 2.0 * inner_deg / g
        coords = (np.arange(g) + 0.5) * step - inner_deg
        gx, gy = np.meshgrid(coords, coords, indexing="xy")
        gx, gy = gx.ravel(), gy.ravel()
        inside = (gx ** 2 + gy ** 2) <= inner_deg ** 2
        gx, gy = gx[inside], gy[inside]
        xs.append(gx)
        ys.append(gy)
        s = np.full(gx.size, max(float(sig["sigma_scale"]) * step, 1e-6))
        sig_deg.append(s)
        area.append(np.full(gx.size, step * step))
        kinds.append(np.zeros(gx.size, dtype=np.int8))
        rbin.append(np.full(gx.size, -1, dtype=np.int32))
        abin.append(np.full(gx.size, -1, dtype=np.int32))

    if rt["mapping"] == "log_polar":
        rho_lo = ecc_to_rho(inner_deg, e0)
        rho_hi = ecc_to_rho(max_ecc, e0)
        edges = np.linspace(rho_lo, rho_hi, n_r + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ecc_c = rho_to_ecc(centers, e0)
        ecc_edges = rho_to_ecc(edges, e0)
        d_ecc = np.diff(ecc_edges)                      # (n_r,) 반경 방향 셀 폭 [deg]
        thetas = 2.0 * np.pi * (np.arange(n_a) + 0.5) / n_a
        arc = ecc_c * 2.0 * np.pi / n_a                 # 접선 방향 셀 폭 [deg]
        cell_sigma = float(sig["sigma_scale"]) * np.maximum(d_ecc, arc)
        # 셀 면적: 환형 구간을 각도 수로 나눈 값 [deg^2]
        ring_area = np.pi * (ecc_edges[1:] ** 2 - ecc_edges[:-1] ** 2) / n_a

        rr = np.repeat(np.arange(n_r), n_a)
        aa = np.tile(np.arange(n_a), n_r)
        ec = ecc_c[rr]
        th = thetas[aa]
        xs.append(ec * np.cos(th))
        ys.append(ec * np.sin(th))
        sig_deg.append(cell_sigma[rr])
        area.append(ring_area[rr])
        kinds.append(np.ones(rr.size, dtype=np.int8))
        rbin.append(rr.astype(np.int32))
        abin.append(aa.astype(np.int32))
        mapping_meta = {"mapping": "log_polar", "e0_deg": e0,
                        "rho_range": [float(rho_lo), float(rho_hi)]}
    else:
        # 총 샘플 수를 맞추기 위해 log-polar 쪽이 만들었을 샘플 수를 센다.
        n_fovea = int(x_len(xs))
        n_target = n_r * n_a + n_fovea
        grid = int(rt["uniform_control"]["grid"])
        if grid <= 0:
            grid = int(round(np.sqrt(4.0 * n_target / np.pi)))
        step = 2.0 * max_ecc / grid
        coords = (np.arange(grid) + 0.5) * step - max_ecc
        gx, gy = np.meshgrid(coords, coords, indexing="xy")
        gx, gy = gx.ravel(), gy.ravel()
        inside = (gx ** 2 + gy ** 2) <= max_ecc ** 2
        gx, gy = gx[inside], gy[inside]
        # 균일 대조군은 중심 패치를 따로 두지 않고 전체를 균일 격자로 덮는다.
        xs = [gx]
        ys = [gy]
        sig_deg = [np.full(gx.size, float(sig["sigma_scale"]) * step)]
        area = [np.full(gx.size, step * step)]
        kinds = [np.full(gx.size, 2, dtype=np.int8)]
        rbin = [np.full(gx.size, -1, dtype=np.int32)]
        abin = [np.full(gx.size, -1, dtype=np.int32)]
        mapping_meta = {"mapping": "uniform_control", "grid": grid,
                        "target_n_samples": int(n_target)}

    x = np.concatenate(xs) if xs else np.zeros(0)
    y = np.concatenate(ys) if ys else np.zeros(0)
    sdeg = np.concatenate(sig_deg) if sig_deg else np.zeros(0)
    ar = np.concatenate(area) if area else np.zeros(0)
    kd = np.concatenate(kinds) if kinds else np.zeros(0, np.int8)
    rb = np.concatenate(rbin) if rbin else np.zeros(0, np.int32)
    ab = np.concatenate(abin) if abin else np.zeros(0, np.int32)

    ecc = np.hypot(x, y)
    theta = np.arctan2(y, x) % (2.0 * np.pi)
    rho = ecc_to_rho(ecc, e0)

    # 모형 수용장 크기: 기본 sigma + 편심도 비례 증가 (설계값, 측정값 아님)
    vf = rf_source["visual_field"]
    rf_sigma = (float(vf["rf_sigma_deg"])
                + float(vf["rf_sigma_growth_per_deg"]) * ecc)

    return SamplingGrid(
        x_deg=x, y_deg=y, ecc_deg=ecc, theta_rad=theta, rho=rho,
        sigma_deg=sdeg, rf_sigma_deg=rf_sigma, cell_area_deg2=ar,
        kind=kd, radial_bin=rb, angular_bin=ab,
        meta={
            "max_ecc_deg": max_ecc,
            "fovea_patch_radius_deg": inner_deg,
            "fovea_patch_enabled": bool(fov["enabled"]),
            "approximation_note_ko": (
                "중심 특이점은 반경 안쪽을 Cartesian 격자로 대체하는 근사로 처리했다. "
                "각도 주기성은 샘플 위치를 각도에서 직접 계산해 처리한다."
            ),
            **mapping_meta,
        },
    )


class LogPolarSampler:
    """격자 위치에서 채널값을 읽는다 (저역통과 후 보간).

    구현: 반경 bin 마다 σ 가 하나이므로 σ 값별로 채널을 **한 번씩만** 흐리고,
    같은 σ 를 쓰는 샘플들을 한꺼번에 보간한다. 샘플마다 거대한 새 영상을
    반복 생성하지 않는다.
    """

    def __init__(self, grid: SamplingGrid, cfg: dict[str, Any]) -> None:
        self.grid = grid
        self.cfg = cfg
        s = cfg["retinotopy"]["sampling"]
        self.lowpass = bool(s["lowpass_before_sampling"])
        self.min_sigma_px = float(s["min_sigma_px"])
        self.order = int(s["interpolation_order"])
        self.fov_deg = float(cfg["retina"]["image"]["fov_deg"])
        self.boundary_mode = cfg["retina"]["dog"]["boundary_mode"]

    def sample(self, channels: np.ndarray,
               gaze_center_px: tuple[float, float] | None = None
               ) -> tuple[np.ndarray, dict[str, Any]]:
        """(C,H,W) 채널 -> (C,S) 샘플값과 메타데이터.

        Returns
        -------
        values : (C, S) float64
            영상 밖 샘플은 0 이고 meta['valid'] 마스크로 표시된다.
        meta : dict
        """
        ch = np.asarray(channels, dtype=np.float64)
        if ch.ndim != 3:
            raise ValueError(f"channels 는 (C,H,W) 여야 한다: {ch.shape}")
        C, H, W = ch.shape
        col, row, valid = self.grid.to_pixels(H, W, self.fov_deg, gaze_center_px)
        ppd = px_per_deg(H, W, self.fov_deg)
        sigma_px = np.maximum(self.grid.sigma_deg * ppd, self.min_sigma_px)

        values = np.zeros((C, self.grid.n_samples), dtype=np.float64)
        # σ 를 소수 3자리로 묶어 흐림 횟수를 줄인다 (사용한 σ 는 메타에 기록)
        keys = np.round(sigma_px, 3)
        used_sigmas = np.unique(keys)
        for sg in used_sigmas:
            idx = np.nonzero(keys == sg)[0]
            if idx.size == 0:
                continue
            coords = np.stack([row[idx], col[idx]], axis=0)
            for c in range(C):
                plane = ch[c]
                if self.lowpass and sg > 0:
                    plane = ndimage.gaussian_filter(
                        plane, float(sg), mode=self.boundary_mode, truncate=4.0)
                values[c, idx] = ndimage.map_coordinates(
                    plane, coords, order=self.order, mode="constant", cval=0.0)
        values[:, ~valid] = 0.0
        values = np.maximum(values, 0.0)

        meta = {
            "n_samples": int(self.grid.n_samples),
            "n_outside_image": int(np.count_nonzero(~valid)),
            "px_per_deg": float(ppd),
            "sigma_px_used": [float(s) for s in used_sigmas],
            "sigma_px_min": float(sigma_px.min()) if sigma_px.size else None,
            "sigma_px_max": float(sigma_px.max()) if sigma_px.size else None,
            "n_pixels_per_cell_estimate": (self.grid.cell_area_deg2 * ppd * ppd).tolist(),
            "interpolation_order": self.order,
            "valid": valid,
            "note_ko": ("셀당 픽셀 개수는 면적(deg^2)과 px_per_deg 로 계산한 추정이며 "
                        "생물학 상수가 아니다."),
        }
        return values, meta


def roundtrip_error(grid: SamplingGrid, e0_deg: float) -> dict[str, Any]:
    """log-polar 정변환/역변환 왕복 오차 (검증 7번). 측정만 한다."""
    rho = ecc_to_rho(grid.ecc_deg, e0_deg)
    ecc_back = rho_to_ecc(rho, e0_deg)
    err = np.abs(ecc_back - grid.ecc_deg)
    x_back = ecc_back * np.cos(grid.theta_rad)
    y_back = ecc_back * np.sin(grid.theta_rad)
    xy_err = np.hypot(x_back - grid.x_deg, y_back - grid.y_deg)
    return {
        "max_abs_ecc_error_deg": float(err.max()) if err.size else 0.0,
        "mean_abs_ecc_error_deg": float(err.mean()) if err.size else 0.0,
        "max_abs_xy_error_deg": float(xy_err.max()) if xy_err.size else 0.0,
        "n_samples": int(grid.n_samples),
    }


def nyquist_margin(grid: SamplingGrid, height: int, width: int,
                   fov_deg: float) -> dict[str, Any]:
    """샘플 간격 대비 저역통과 폭의 여유를 본다 (aliasing 진단, 판정하지 않음).

    로그-극좌표 격자에서 이웃 샘플 간 최소 시야 거리와 σ 를 비교한다.
    σ >= 0.5 * 간격 이면 대략 Nyquist 여유가 있다고 본다 (경험적 기준이며
    엄밀한 증명이 아니다).
    """
    n = grid.n_samples
    if n < 2:
        return {"n_samples": n, "checked": False}
    from scipy.spatial import cKDTree

    pts = np.stack([grid.x_deg, grid.y_deg], axis=1)
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    spacing = d[:, 1]
    ratio = np.divide(grid.sigma_deg, np.maximum(spacing, 1e-12))
    return {
        "n_samples": n,
        "checked": True,
        "spacing_deg_min": float(spacing.min()),
        "spacing_deg_max": float(spacing.max()),
        "sigma_over_spacing_min": float(ratio.min()),
        "sigma_over_spacing_median": float(np.median(ratio)),
        "n_below_half": int(np.count_nonzero(ratio < 0.5)),
        "criterion_note_ko": ("sigma/이웃간격 >= 0.5 를 경험적 여유 기준으로 본다. "
                              "엄밀한 aliasing 증명이 아니다."),
    }


__all__ = [
    "px_per_deg", "ecc_to_rho", "rho_to_ecc", "SamplingGrid", "build_grid",
    "LogPolarSampler", "roundtrip_error", "nyquist_margin",
]
