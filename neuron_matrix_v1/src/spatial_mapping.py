"""spatial_mapping.py -- 비균일(로그-극좌표) 격자 저역통과 + 표본 추출.

명세 [6]. 격자 정의:

    rho_k   = (k + 0.5) / radial_bins
    r_k     = r0 * (exp(rho_k * log(1 + R/r0)) - 1)
    angle_j = 2*pi*(j + 0.5) / angular_bins
    x_kj    = cx + r_k*cos(angle_j)
    y_kj    = cy + r_k*sin(angle_j)

반경 경계는 같은 함수를 rho = k/radial_bins 에서 평가해 Δr 을 얻는다.

    sigma_pool = max(0.5, 0.5*max(delta_r, r_k*2*pi/angular_bins))

기본 시야는 영상에 **내접한 원** (반경 R = min(W-1,H-1)/2) 이다.
바깥 모서리는 처리되지 않으며 마스크와 그림에 표시한다.

구현 선택 (기록): 근사를 쓰지 않는 **direct** 경로를 기본으로 한다.
반경 bin 마다 sigma_pool 이 하나이므로 채널당 ``radial_bins`` 회의
``scipy.ndimage.gaussian_filter`` 만 수행하고, 그 흐린 영상에서 이중선형
보간으로 값을 읽는다. 표본마다 새 거대 영상을 반복 생성하지 않도록
한 영상당 흐린 영상은 bin 당 1회만 만든다.

주의: 로그-극좌표 배열의 (k, j) 인덱스는 원본 영상의 선 기울기와 다르다.
방향 라벨과 목표 생성은 **원본 영상 좌표계**에서만 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class GridConfig:
    radial_bins: int = 8
    angular_bins: int = 16
    r0_px: float = 2.0
    truncate: float = 3.0
    mode: str = "reflect"
    rf_radius_rule: str = "max(2, 2*sigma_pool)"


@dataclass
class LogPolarGrid:
    """영상 크기에 대해 확정된 격자 기하 (부작용 없는 값 객체).

    Attributes
    ----------
    height, width : int
    cx, cy : float
        시야 중심 (원본 영상 픽셀 좌표).
    R : float
        내접 원 반경 = min(W-1, H-1)/2.
    n_points : int
        radial_bins * angular_bins.
    radial_bin, angular_bin : (P,) int64
    r_k : (P,) float64
        각 표본의 편심도(픽셀).
    x, y : (P,) float64
        원본 영상 좌표 (x=열, y=행).
    sigma_pool : (P,) float64
    rf_radius : (P,) float64
        모형 수용장 반경 (설계값).
    delta_r : (radial_bins,) float64
    sigma_by_bin : (radial_bins,) float64
    """

    height: int
    width: int
    cx: float
    cy: float
    R: float
    cfg: GridConfig
    radial_bin: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    angular_bin: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    r_k: np.ndarray = field(default_factory=lambda: np.zeros(0))
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma_pool: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rf_radius: np.ndarray = field(default_factory=lambda: np.zeros(0))
    delta_r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma_by_bin: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_points(self) -> int:
        return int(self.x.size)

    def valid_fov_mask(self) -> np.ndarray:
        """(H,W) bool: 기본 유효 시야(내접 원) 마스크."""
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        return ((xx - self.cx) ** 2 + (yy - self.cy) ** 2) <= self.R**2

    def to_dict(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "cx": self.cx,
            "cy": self.cy,
            "R": self.R,
            "radial_bins": self.cfg.radial_bins,
            "angular_bins": self.cfg.angular_bins,
            "r0_px": self.cfg.r0_px,
            "n_points": self.n_points,
            "r_k": self.r_k.tolist(),
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "sigma_pool": self.sigma_pool.tolist(),
            "rf_radius": self.rf_radius.tolist(),
            "delta_r": self.delta_r.tolist(),
            "sigma_by_bin": self.sigma_by_bin.tolist(),
            "rf_radius_rule": self.cfg.rf_radius_rule,
        }


def build_grid(height: int, width: int, cfg: GridConfig) -> LogPolarGrid:
    """영상 크기에서 로그-극좌표 격자를 만든다 (부작용 없음)."""
    H, W = int(height), int(width)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    R = min(W - 1, H - 1) / 2.0
    r0 = float(cfg.r0_px)
    K, J = int(cfg.radial_bins), int(cfg.angular_bins)
    log_term = np.log(1.0 + R / r0)

    def radius_of(rho: np.ndarray | float) -> np.ndarray:
        return r0 * (np.exp(np.asarray(rho, dtype=np.float64) * log_term) - 1.0)

    k = np.arange(K, dtype=np.float64)
    r_center = radius_of((k + 0.5) / K)                 # (K,)
    r_edge = radius_of(np.arange(K + 1, dtype=np.float64) / K)   # (K+1,)
    delta_r = np.diff(r_edge)                            # (K,)
    arc = r_center * 2.0 * np.pi / J
    sigma_by_bin = np.maximum(0.5, 0.5 * np.maximum(delta_r, arc))

    kk = np.repeat(np.arange(K, dtype=np.int64), J)
    jj = np.tile(np.arange(J, dtype=np.int64), K)
    angles = 2.0 * np.pi * (jj + 0.5) / J
    r_pt = r_center[kk]
    x = cx + r_pt * np.cos(angles)
    y = cy + r_pt * np.sin(angles)
    sigma_pt = sigma_by_bin[kk]
    rf_radius = np.maximum(2.0, 2.0 * sigma_pt)

    return LogPolarGrid(
        height=H, width=W, cx=cx, cy=cy, R=R, cfg=cfg,
        radial_bin=kk, angular_bin=jj, r_k=r_pt, x=x, y=y,
        sigma_pool=sigma_pt, rf_radius=rf_radius,
        delta_r=delta_r, sigma_by_bin=sigma_by_bin,
    )


def bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(H,W) 영상에서 (P,) 좌표의 이중선형 보간값. 경계는 클램프."""
    H, W = img.shape
    x = np.clip(x, 0.0, W - 1.0)
    y = np.clip(y, 0.0, H - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx = x - x0
    fy = y - y0
    v = (
        img[y0, x0] * (1 - fx) * (1 - fy)
        + img[y0, x1] * fx * (1 - fy)
        + img[y1, x0] * (1 - fx) * fy
        + img[y1, x1] * fx * fy
    )
    return v


class LogPolarSampler:
    """비균일 격자 저역통과 + 표본 추출기.

    Parameters
    ----------
    grid : LogPolarGrid

    Notes
    -----
    * ``method="direct"``: 반경 bin 별 sigma_pool 로 채널을 흐린 뒤 이중선형 보간.
      근사 없음. 사용한 폭은 ``grid.sigma_by_bin`` 에 기록되어 있다.
    * 부작용 없음.
    """

    def __init__(self, grid: LogPolarGrid, method: str = "direct") -> None:
        if method != "direct":
            raise ValueError("현재 구현된 방법은 'direct' 뿐이다")
        self.grid = grid
        self.method = method

    def sample(self, channels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """ON/OFF 영상 -> 격자 표본.

        Parameters
        ----------
        channels : dict
            ``{"on": (H,W) float64, "off": (H,W) float64}``.

        Returns
        -------
        dict
            ``{"on": (P,) float64, "off": (P,) float64}``, P = n_points.
            값은 비음수.
        """
        g = self.grid
        out: dict[str, np.ndarray] = {}
        for name in ("on", "off"):
            img = np.asarray(channels[name], dtype=np.float64)
            if img.shape != (g.height, g.width):
                raise ValueError(
                    f"채널 '{name}' shape {img.shape} 가 격자 {(g.height, g.width)} 와 다르다"
                )
            vals = np.zeros(g.n_points, dtype=np.float64)
            for k, sigma in enumerate(g.sigma_by_bin):
                mask = g.radial_bin == k
                blurred = gaussian_filter(
                    img, float(sigma), mode=g.cfg.mode, truncate=g.cfg.truncate
                )
                vals[mask] = bilinear_sample(blurred, g.x[mask], g.y[mask])
            out[name] = np.maximum(vals, 0.0)
        return out


class InputNormalizer:
    """L4 입력 크기 정규화 계수 (훈련 데이터로만 추정).

    규칙 (명세 [6]): 훈련 ON/OFF 표본의 **양수값 95 백분위**로 나눈 뒤 [0,1] 로 제한.
    모든 값이 0 이면 scale=1.0 으로 두고 ``degenerate=True`` 로 기록한다
    (0 으로 나누지 않으며, 조용히 임계값을 바꾸지도 않는다).
    """

    def __init__(self, percentile: float = 95.0) -> None:
        self.percentile = float(percentile)
        self.scale: float = 1.0
        self.degenerate: bool = False
        self.fitted: bool = False
        self.n_positive_values: int = 0

    def fit(self, samples: np.ndarray) -> "InputNormalizer":
        """samples: (n_train, P*2) 또는 임의 shape 의 ON/OFF 표본 값 배열.

        부작용: self.scale / self.fitted 를 설정한다.
        """
        v = np.asarray(samples, dtype=np.float64).ravel()
        pos = v[v > 0.0]
        self.n_positive_values = int(pos.size)
        if pos.size == 0:
            self.scale = 1.0
            self.degenerate = True
        else:
            s = float(np.percentile(pos, self.percentile))
            if not np.isfinite(s) or s <= 0.0:
                self.scale = 1.0
                self.degenerate = True
            else:
                self.scale = s
                self.degenerate = False
        self.fitted = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """[0,1] 로 제한한 정규화 값 (부작용 없음)."""
        if not self.fitted:
            raise RuntimeError("InputNormalizer 가 훈련 데이터로 fit 되지 않았다")
        return np.clip(np.asarray(values, dtype=np.float64) / self.scale, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "percentile": self.percentile,
            "scale": self.scale,
            "degenerate": self.degenerate,
            "fitted": self.fitted,
            "n_positive_values": self.n_positive_values,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "InputNormalizer":
        n = InputNormalizer(d.get("percentile", 95.0))
        n.scale = float(d["scale"])
        n.degenerate = bool(d["degenerate"])
        n.fitted = bool(d["fitted"])
        n.n_positive_values = int(d.get("n_positive_values", 0))
        return n
