"""datasets.py -- 합성 선 자극 생성 (가로선/세로선/빈 화면).

명세 [8]. 이미지와 **별도로** 장면 메타데이터를 만든다. 학생(감각 경로)은
이미지 픽셀만 받고, 메타데이터는 교사와 평가기만 읽는다.

렌더링 식 (명세 요구: 픽셀 생성식 명시)
---------------------------------------
배경 밝기 ``bg`` = 0.5. 선분 [p0, p1], 선폭 ``line_width`` (픽셀, 전체 폭),
극성 ``polarity`` (+1 밝은 선 / -1 어두운 선), 진폭 ``contrast``:

    dist(x,y)  = 점 (x+0, y+0) 에서 선분까지의 유클리드 거리 (픽셀)
    coverage   = clip(line_width/2 + 0.5 - dist, 0, 1)      # 안티앨리어싱
    image(x,y) = clip(bg + polarity * contrast * coverage, 0, 1)

``coverage`` 는 선 경계에서 1픽셀 폭으로 선형 감쇠하는 근사적 커버리지다.
정확한 면적 적분이 아니며, 이는 설계 선택으로 기록한다.

이미지는 [0,1] float64 회색조이고 기본적으로 **sRGB** 값으로 해석한다
(``RetinaEncoder.encode(..., input_colorspace="srgb")``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass
class SceneSpec:
    """한 장면의 메타데이터 (픽셀과 분리).

    Attributes
    ----------
    scene_id : str
        분할 간 겹치지 않는 고유 ID.
    base_scene_id : str
        파생 영상을 만들 때 같은 기초 장면을 한 분할에 묶기 위한 키.
        현재 데이터는 파생 변형을 만들지 않으므로 scene_id 와 같다.
    shape_type : {"horizontal_line", "vertical_line", "blank"}
    segment_endpoints : ((x0,y0),(x1,y1)) | None
        원본 영상 픽셀 좌표의 **유한 선분** 양끝. blank 면 None.
    orientation_deg : float | None
        원본 영상 좌표계 기준 방향 (0=가로, 90=세로). blank 면 None.
    contrast : float
        배경 0.5 대비 진폭. Michelson 대비 = contrast/0.5.
    polarity : int
        +1 밝은 선, -1 어두운 선. blank 면 0.
    line_width : float
        전체 선폭 (픽셀).
    """

    scene_id: str
    base_scene_id: str
    shape_type: str
    segment_endpoints: tuple[tuple[float, float], tuple[float, float]] | None
    orientation_deg: float | None
    contrast: float
    polarity: int
    line_width: float
    height: int = 64
    width: int = 64

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "base_scene_id": self.base_scene_id,
            "shape_type": self.shape_type,
            "segment_endpoints": (
                [list(self.segment_endpoints[0]), list(self.segment_endpoints[1])]
                if self.segment_endpoints
                else None
            ),
            "orientation_deg": self.orientation_deg,
            "contrast": self.contrast,
            "polarity": self.polarity,
            "line_width": self.line_width,
            "height": self.height,
            "width": self.width,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "SceneSpec":
        ep = d["segment_endpoints"]
        return SceneSpec(
            scene_id=d["scene_id"],
            base_scene_id=d["base_scene_id"],
            shape_type=d["shape_type"],
            segment_endpoints=(tuple(ep[0]), tuple(ep[1])) if ep else None,
            orientation_deg=d["orientation_deg"],
            contrast=d["contrast"],
            polarity=d["polarity"],
            line_width=d["line_width"],
            height=d.get("height", 64),
            width=d.get("width", 64),
        )


@dataclass
class DatasetConfig:
    height: int = 64
    width: int = 64
    background: float = 0.5
    p_horizontal: float = 0.40
    p_vertical: float = 0.40
    p_blank: float = 0.20
    contrast_min: float = 0.25
    contrast_max: float = 0.45
    line_width_min: float = 1.0
    line_width_max: float = 2.5
    length_frac_min: float = 0.35
    length_frac_max: float = 0.95
    center_radius_frac: float = 0.70
    center_radius_min_frac: float = 0.0
    orientations_deg: tuple[float, ...] = (0.0, 90.0)


def point_segment_distance(
    px: np.ndarray, py: np.ndarray, p0: tuple[float, float], p1: tuple[float, float]
) -> np.ndarray:
    """점 (px,py) 에서 **유한 선분** p0-p1 까지의 최소 거리 (무한 직선 아님).

    Returns (broadcast 된) 거리 배열. 부작용 없음.
    """
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    denom = dx * dx + dy * dy
    if denom <= 0.0:
        return np.hypot(px - x0, py - y0)
    t = ((px - x0) * dx + (py - y0) * dy) / denom
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def render_scene(spec: SceneSpec, background: float = 0.5) -> np.ndarray:
    """장면 -> (H,W) float64 [0,1] 회색조 이미지 (sRGB 값으로 해석). 부작용 없음."""
    H, W = spec.height, spec.width
    img = np.full((H, W), float(background), dtype=np.float64)
    if spec.shape_type == "blank" or spec.segment_endpoints is None:
        return img
    yy, xx = np.mgrid[0:H, 0:W]
    dist = point_segment_distance(
        xx.astype(np.float64), yy.astype(np.float64), *spec.segment_endpoints
    )
    coverage = np.clip(spec.line_width / 2.0 + 0.5 - dist, 0.0, 1.0)
    img = img + spec.polarity * spec.contrast * coverage
    return np.clip(img, 0.0, 1.0)


def generate_scenes(
    n: int,
    rng: np.random.Generator,
    cfg: DatasetConfig,
    prefix: str,
    contrast_range: tuple[float, float] | None = None,
    center_radius_range_frac: tuple[float, float] | None = None,
) -> list[SceneSpec]:
    """장면 n 개를 생성한다 (이미지는 만들지 않는다).

    선 중심은 유효 시야(내접 원) 안, 반경 ``center_radius_frac * R`` 이내에서
    뽑는다. ``contrast_range`` / ``center_radius_range_frac`` 를 주면
    독립 추가 시험용으로 다른 범위를 쓴다.
    """
    H, W = cfg.height, cfg.width
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    R = min(W - 1, H - 1) / 2.0
    cmin, cmax = contrast_range or (cfg.contrast_min, cfg.contrast_max)
    rmin_f, rmax_f = center_radius_range_frac or (
        cfg.center_radius_min_frac,
        cfg.center_radius_frac,
    )

    kinds = np.array(["horizontal_line", "vertical_line", "blank"])
    probs = np.array([cfg.p_horizontal, cfg.p_vertical, cfg.p_blank], dtype=np.float64)
    probs = probs / probs.sum()

    scenes: list[SceneSpec] = []
    for i in range(int(n)):
        sid = f"{prefix}-{i:05d}"
        kind = str(rng.choice(kinds, p=probs))
        if kind == "blank":
            scenes.append(
                SceneSpec(sid, sid, "blank", None, None, 0.0, 0, 0.0, H, W)
            )
            continue
        # 중심: 유효 시야 안에서 균일 면적 샘플링
        u = rng.uniform(rmin_f**2, rmax_f**2)
        rad = R * np.sqrt(u)
        ang = rng.uniform(0.0, 2.0 * np.pi)
        mx = cx + rad * np.cos(ang)
        my = cy + rad * np.sin(ang)
        length = R * 2.0 * rng.uniform(cfg.length_frac_min, cfg.length_frac_max)
        contrast = float(rng.uniform(cmin, cmax))
        polarity = int(rng.choice([-1, 1]))
        lw = float(rng.uniform(cfg.line_width_min, cfg.line_width_max))
        if kind == "horizontal_line":
            ori = 0.0
            dx, dy = length / 2.0, 0.0
        else:
            ori = 90.0
            dx, dy = 0.0, length / 2.0
        p0 = (float(mx - dx), float(my - dy))
        p1 = (float(mx + dx), float(my + dy))
        scenes.append(SceneSpec(sid, sid, kind, (p0, p1), ori, contrast, polarity, lw, H, W))
    return scenes


def split_summary(scenes: Iterable[SceneSpec]) -> dict[str, Any]:
    """분할 요약 통계 (표본 수, 방향/극성/빈 화면 비율, 위치·대비 범위)."""
    scenes = list(scenes)
    n = len(scenes)
    if n == 0:
        return {"n": 0}
    kinds: dict[str, int] = {}
    for s in scenes:
        kinds[s.shape_type] = kinds.get(s.shape_type, 0) + 1
    lines = [s for s in scenes if s.shape_type != "blank"]
    contrasts = [s.contrast for s in lines]
    pol = {"+1": sum(1 for s in lines if s.polarity > 0),
           "-1": sum(1 for s in lines if s.polarity < 0)}
    centers = [
        ((s.segment_endpoints[0][0] + s.segment_endpoints[1][0]) / 2.0,
         (s.segment_endpoints[0][1] + s.segment_endpoints[1][1]) / 2.0)
        for s in lines
    ]
    return {
        "n": n,
        "counts_by_shape": kinds,
        "fraction_by_shape": {k: v / n for k, v in kinds.items()},
        "n_lines": len(lines),
        "polarity_counts": pol,
        "contrast_min": float(min(contrasts)) if contrasts else None,
        "contrast_max": float(max(contrasts)) if contrasts else None,
        "michelson_contrast_range": (
            [float(min(contrasts)) / 0.5, float(max(contrasts)) / 0.5] if contrasts else None
        ),
        "center_x_range": (
            [float(min(c[0] for c in centers)), float(max(c[0] for c in centers))]
            if centers else None
        ),
        "center_y_range": (
            [float(min(c[1] for c in centers)), float(max(c[1] for c in centers))]
            if centers else None
        ),
        "line_width_range": (
            [float(min(s.line_width for s in lines)), float(max(s.line_width for s in lines))]
            if lines else None
        ),
        "orientation_counts": {
            "0": sum(1 for s in lines if s.orientation_deg == 0.0),
            "90": sum(1 for s in lines if s.orientation_deg == 90.0),
        },
    }


def make_splits(
    seed_rng: np.random.Generator,
    cfg: DatasetConfig,
    n_train: int,
    n_dev: int,
    n_test: int,
) -> dict[str, list[SceneSpec]]:
    """훈련/검증/시험 분할 생성. scene_id 접두사가 달라 절대 겹치지 않는다."""
    return {
        "train": generate_scenes(n_train, seed_rng, cfg, "train"),
        "dev": generate_scenes(n_dev, seed_rng, cfg, "dev"),
        "test": generate_scenes(n_test, seed_rng, cfg, "test"),
    }
