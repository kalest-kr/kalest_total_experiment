"""stimuli.py -- 자극 생성기.

명세 11절. 모든 영상은 ``(H, W, 3) float64``, 값 범위 [0,1] 이고 **sRGB 값**으로
해석된다. 도형은 픽셀당 supersampling 으로 안티앨리어싱한다.

제공하는 자극

* ``uniform``            : 균일 영상 (DoG 상수 응답 검사용)
* ``dot``                : 한 점
* ``bar``                : 방향/위상/명암/길이를 지정한 막대
* ``orientation_sweep``  : 방향 스윕
* ``phase_sweep``        : 위상 스윕
* ``contrast_sweep``     : 명암 스윕
* ``length_sweep``       : 종단 억제(end-stopping) 용 길이 변화 막대
* ``center_surround``    : 중앙-주변 방향 대비
* ``shape_classes``      : 원·삼각형·사각형 등 분류용 도형
* ``color_illumination`` : 색/조명 변환
* ``transform``          : 위치/크기 변환
* ``binocular``          : 좌/우 영상 쌍 (실제 두 눈 입력이 있는 설정에서만 사용)
* ``motion``             : 프레임 시퀀스 (정지영상 실행의 운동 지표는 not_applicable)

**중요**: 단안 영상을 복제한 양안 입력은 실제 양안 시차 실험이 아니다.
운동 기능은 프레임 시퀀스와 시간적 회로가 있어야 평가할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

BACKGROUND: float = 0.5


@dataclass
class Stimulus:
    """자극 1개.

    Attributes
    ----------
    stimulus_id : str
    base_id : str
        같은 기초 자극의 변형들을 묶는 키. 분할이 이 키를 넘나들지 않게 한다.
    label : str
        분류 라벨 (없으면 "").
    frames : list[np.ndarray]
        길이 1 이면 정지 영상, 2 이상이면 시퀀스.
    eyes : dict[str, np.ndarray] | None
        양안 입력. None 이면 단안.
    params : dict
        생성 파라미터 (방향/위상/명암/크기 등).
    """

    stimulus_id: str
    base_id: str
    label: str
    frames: list[np.ndarray]
    eyes: dict[str, np.ndarray] | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def image(self) -> np.ndarray:
        return self.frames[0]

    @property
    def is_sequence(self) -> bool:
        return len(self.frames) > 1

    @property
    def is_binocular(self) -> bool:
        return self.eyes is not None

    def meta(self) -> dict[str, Any]:
        return {
            "stimulus_id": self.stimulus_id, "base_id": self.base_id,
            "label": self.label, "n_frames": len(self.frames),
            "binocular": self.is_binocular, "params": self.params,
        }


# ----------------------------------------------------------------------
# 래스터화 도우미
# ----------------------------------------------------------------------
def _grid(h: int, w: int, ss: int) -> tuple[np.ndarray, np.ndarray]:
    """supersampling 격자의 (x, y) 좌표. shape (h*ss, w*ss)."""
    ys = (np.arange(h * ss) + 0.5) / ss - 0.5
    xs = (np.arange(w * ss) + 0.5) / ss - 0.5
    return np.meshgrid(xs, ys, indexing="xy")


def _downsample(mask: np.ndarray, ss: int) -> np.ndarray:
    h, w = mask.shape[0] // ss, mask.shape[1] // ss
    return mask.reshape(h, ss, w, ss).mean(axis=(1, 3))


def point_in_polygon(px: np.ndarray, py: np.ndarray,
                     verts: np.ndarray) -> np.ndarray:
    """짝수-홀수 규칙 다각형 내부 판정 (벡터화)."""
    inside = np.zeros(px.shape, dtype=bool)
    n = verts.shape[0]
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        cond = ((y1 > py) != (y2 > py))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (py - y1) / np.where(y2 == y1, np.nan, (y2 - y1)) + x1
        hit = cond & (px < np.nan_to_num(xint, nan=-np.inf))
        inside ^= hit
    return inside


def _colorize(coverage: np.ndarray, contrast: float, polarity: int,
              color: Sequence[float] | None = None) -> np.ndarray:
    """커버리지 -> (H,W,3) 영상."""
    h, w = coverage.shape
    img = np.full((h, w, 3), BACKGROUND, dtype=np.float64)
    if color is None:
        delta = polarity * contrast * coverage
        img += delta[:, :, None]
    else:
        c = np.asarray(color, dtype=np.float64)
        img += coverage[:, :, None] * (c[None, None, :] - BACKGROUND) * contrast * 2.0
    return np.clip(img, 0.0, 1.0)


def uniform_image(h: int, w: int, value: float = BACKGROUND) -> np.ndarray:
    return np.full((h, w, 3), float(value), dtype=np.float64)


def dot_image(h: int, w: int, cx: float, cy: float, radius_px: float,
              contrast: float = 0.4, polarity: int = 1, ss: int = 4) -> np.ndarray:
    gx, gy = _grid(h, w, ss)
    m = ((gx - cx) ** 2 + (gy - cy) ** 2) <= radius_px ** 2
    return _colorize(_downsample(m.astype(np.float64), ss), contrast, polarity)


def bar_image(h: int, w: int, cx: float, cy: float, length_px: float,
              width_px: float, orientation_rad: float, contrast: float = 0.4,
              polarity: int = 1, ss: int = 4) -> np.ndarray:
    gx, gy = _grid(h, w, ss)
    dx, dy = gx - cx, gy - cy
    ct, st = np.cos(orientation_rad), np.sin(orientation_rad)
    xr = dx * ct + dy * st
    yr = -dx * st + dy * ct
    m = (np.abs(xr) <= length_px / 2.0) & (np.abs(yr) <= width_px / 2.0)
    return _colorize(_downsample(m.astype(np.float64), ss), contrast, polarity)


def grating_patch(h: int, w: int, cx: float, cy: float, radius_px: float,
                  orientation_rad: float, phase_rad: float,
                  cycles_per_px: float, contrast: float = 0.4,
                  ss: int = 2) -> np.ndarray:
    """원형 창을 가진 사인 격자 (위상 스윕/명암 반전 검사용)."""
    gx, gy = _grid(h, w, ss)
    dx, dy = gx - cx, gy - cy
    ct, st = np.cos(orientation_rad), np.sin(orientation_rad)
    xr = dx * ct + dy * st
    wave = np.cos(2.0 * np.pi * cycles_per_px * xr + phase_rad)
    window = ((dx ** 2 + dy ** 2) <= radius_px ** 2).astype(np.float64)
    field = _downsample(wave * window, ss)
    img = np.clip(BACKGROUND + contrast * field[:, :, None], 0.0, 1.0)
    return np.repeat(img, 3, axis=2) if img.shape[2] == 1 else img


def polygon_image(h: int, w: int, verts: np.ndarray, contrast: float = 0.4,
                  polarity: int = 1, ss: int = 4,
                  color: Sequence[float] | None = None) -> np.ndarray:
    gx, gy = _grid(h, w, ss)
    m = point_in_polygon(gx, gy, np.asarray(verts, dtype=np.float64))
    return _colorize(_downsample(m.astype(np.float64), ss), contrast, polarity, color)


def regular_polygon(cx: float, cy: float, radius: float, n_sides: int,
                    rotation_rad: float = 0.0) -> np.ndarray:
    ang = rotation_rad + 2.0 * np.pi * np.arange(n_sides) / n_sides
    return np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=1)


def cross_image(h: int, w: int, cx: float, cy: float, arm_px: float,
                width_px: float, contrast: float = 0.4, polarity: int = 1,
                ss: int = 4) -> np.ndarray:
    a = bar_image(h, w, cx, cy, arm_px, width_px, 0.0, contrast, polarity, ss)
    b = bar_image(h, w, cx, cy, arm_px, width_px, np.pi / 2, contrast, polarity, ss)
    if polarity > 0:
        return np.maximum(a, b)
    return np.minimum(a, b)


def center_surround_image(h: int, w: int, center_ori_rad: float,
                          surround_ori_rad: float, radius_px: float,
                          outer_px: float, cycles_per_px: float,
                          contrast: float = 0.4, ss: int = 2) -> np.ndarray:
    """중앙-주변 방향 대비 자극."""
    gx, gy = _grid(h, w, ss)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dx, dy = gx - cx, gy - cy
    r2 = dx ** 2 + dy ** 2
    def wave(theta: float) -> np.ndarray:
        xr = dx * np.cos(theta) + dy * np.sin(theta)
        return np.cos(2.0 * np.pi * cycles_per_px * xr)
    field = np.where(r2 <= radius_px ** 2, wave(center_ori_rad),
                     np.where(r2 <= outer_px ** 2, wave(surround_ori_rad), 0.0))
    f = _downsample(field, ss)
    img = np.clip(BACKGROUND + contrast * f, 0.0, 1.0)
    return np.repeat(img[:, :, None], 3, axis=2)


# ----------------------------------------------------------------------
# 자극 집합 생성
# ----------------------------------------------------------------------
DEFAULT_SHAPE_CLASSES: tuple[str, ...] = (
    "circle", "triangle", "square", "bar_horizontal", "bar_vertical", "cross", "blank",
)


def generate(cfg: dict[str, Any], rng: np.random.Generator) -> list[Stimulus]:
    """설정 ``experiment.stimuli`` 목록에서 자극을 만든다 (부작용 없음)."""
    h = int(cfg["retina"]["image"]["max_side_px"])
    w = h
    out: list[Stimulus] = []
    for si, spec in enumerate(cfg["experiment"]["stimuli"]):
        kind = spec["kind"]
        n = int(spec.get("n", 1))
        p = dict(spec.get("params", {}))
        size = int(p.pop("size_px", h))
        hh = ww = size
        if kind == "uniform":
            for i in range(n):
                v = float(p.get("value", BACKGROUND))
                out.append(Stimulus(f"s{si}_uniform_{i}", f"b{si}_uniform_{i}", "blank",
                                    [uniform_image(hh, ww, v)],
                                    params={"kind": kind, "value": v}))
        elif kind == "dot":
            for i in range(n):
                cx = rng.uniform(0.2, 0.8) * ww
                cy = rng.uniform(0.2, 0.8) * hh
                out.append(Stimulus(f"s{si}_dot_{i}", f"b{si}_dot_{i}", "dot",
                                    [dot_image(hh, ww, cx, cy,
                                               float(p.get("radius_px", 3.0)),
                                               float(p.get("contrast", 0.4)))],
                                    params={"kind": kind, "cx": cx, "cy": cy}))
        elif kind == "bar":
            for i in range(n):
                ori = rng.uniform(0.0, np.pi)
                cx = rng.uniform(0.3, 0.7) * ww
                cy = rng.uniform(0.3, 0.7) * hh
                out.append(Stimulus(
                    f"s{si}_bar_{i}", f"b{si}_bar_{i}", "bar",
                    [bar_image(hh, ww, cx, cy, float(p.get("length_px", 0.5 * ww)),
                               float(p.get("width_px", 3.0)), ori,
                               float(p.get("contrast", 0.4)))],
                    params={"kind": kind, "orientation_rad": float(ori),
                            "cx": cx, "cy": cy}))
        elif kind in ("orientation_sweep", "phase_sweep", "contrast_sweep",
                      "length_sweep"):
            out += _sweep(kind, si, hh, ww, n, p)
        elif kind == "center_surround":
            n_ori = int(p.get("n_orientations", 4))
            for a in range(n_ori):
                for b in range(n_ori):
                    t1 = np.pi * a / n_ori
                    t2 = np.pi * b / n_ori
                    out.append(Stimulus(
                        f"s{si}_cs_{a}_{b}", f"b{si}_cs_{a}_{b}", "center_surround",
                        [center_surround_image(
                            hh, ww, t1, t2, float(p.get("radius_px", 0.12 * ww)),
                            float(p.get("outer_px", 0.4 * ww)),
                            float(p.get("cycles_per_px", 0.08)),
                            float(p.get("contrast", 0.4)))],
                        params={"kind": kind, "center_ori_rad": t1,
                                "surround_ori_rad": t2}))
        elif kind == "shape_classes":
            out += _shape_classes(si, hh, ww, p, rng)
        elif kind == "color_illumination":
            out += _color_illumination(si, hh, ww, p, rng)
        elif kind == "transform":
            out += _transforms(si, hh, ww, p, rng)
        elif kind == "binocular":
            out += _binocular(si, hh, ww, p, rng)
        elif kind == "motion":
            out += _motion(si, hh, ww, p, rng)
        else:
            raise ValueError(f"알 수 없는 자극 종류: {kind!r}")
    return out


def _sweep(kind: str, si: int, h: int, w: int, n: int,
           p: dict[str, Any]) -> list[Stimulus]:
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    out: list[Stimulus] = []
    if kind == "orientation_sweep":
        n_ori = int(p.get("n_orientations", 12))
        for i in range(n_ori):
            ori = np.pi * i / n_ori
            out.append(Stimulus(
                f"s{si}_ori_{i}", f"b{si}_ori", "grating",
                [grating_patch(h, w, cx, cy, float(p.get("radius_px", 0.3 * w)), ori,
                               float(p.get("phase_rad", 0.0)),
                               float(p.get("cycles_per_px", 0.08)),
                               float(p.get("contrast", 0.4)))],
                params={"kind": kind, "orientation_rad": float(ori),
                        "orientation_deg": float(np.rad2deg(ori))}))
    elif kind == "phase_sweep":
        n_ph = int(p.get("n_phases", 8))
        for i in range(n_ph):
            ph = 2.0 * np.pi * i / n_ph
            out.append(Stimulus(
                f"s{si}_phase_{i}", f"b{si}_phase", "grating",
                [grating_patch(h, w, cx, cy, float(p.get("radius_px", 0.3 * w)),
                               float(p.get("orientation_rad", 0.0)), ph,
                               float(p.get("cycles_per_px", 0.08)),
                               float(p.get("contrast", 0.4)))],
                params={"kind": kind, "phase_rad": float(ph)}))
    elif kind == "contrast_sweep":
        for i, c in enumerate(np.linspace(float(p.get("min", 0.05)),
                                          float(p.get("max", 0.45)), n)):
            out.append(Stimulus(
                f"s{si}_contrast_{i}", f"b{si}_contrast", "grating",
                [grating_patch(h, w, cx, cy, float(p.get("radius_px", 0.3 * w)),
                               float(p.get("orientation_rad", 0.0)),
                               float(p.get("phase_rad", 0.0)),
                               float(p.get("cycles_per_px", 0.08)), float(c))],
                params={"kind": kind, "contrast": float(c)}))
    else:  # length_sweep (종단 억제)
        for i, L in enumerate(np.linspace(float(p.get("min_px", 4.0)),
                                          float(p.get("max_px", 0.8 * w)), n)):
            out.append(Stimulus(
                f"s{si}_len_{i}", f"b{si}_len", "bar",
                [bar_image(h, w, cx, cy, float(L), float(p.get("width_px", 3.0)),
                           float(p.get("orientation_rad", 0.0)),
                           float(p.get("contrast", 0.4)))],
                params={"kind": kind, "length_px": float(L)}))
    return out


def _shape_classes(si: int, h: int, w: int, p: dict[str, Any],
                   rng: np.random.Generator) -> list[Stimulus]:
    classes = list(p.get("classes", DEFAULT_SHAPE_CLASSES))
    n_per = int(p.get("n_per_class", 8))
    n_var = int(p.get("n_variants", 1))
    radius = float(p.get("radius_px", 0.18 * w))
    contrast = float(p.get("contrast", 0.4))
    out: list[Stimulus] = []
    for label in classes:
        for b in range(n_per):
            base = f"b{si}_{label}_{b}"
            bcx = rng.uniform(0.35, 0.65) * w
            bcy = rng.uniform(0.35, 0.65) * h
            brot = rng.uniform(0.0, 2.0 * np.pi)
            for v in range(n_var):
                jx = bcx + rng.normal(0.0, 0.02 * w)
                jy = bcy + rng.normal(0.0, 0.02 * h)
                rr = radius * float(rng.uniform(0.9, 1.1))
                img = _render_shape(label, h, w, jx, jy, rr, brot, contrast)
                out.append(Stimulus(f"{base}_v{v}", base, label, [img],
                                    params={"kind": "shape_classes", "cx": jx,
                                            "cy": jy, "radius_px": rr,
                                            "rotation_rad": float(brot),
                                            "variant": v}))
    return out


def _render_shape(label: str, h: int, w: int, cx: float, cy: float,
                  radius: float, rot: float, contrast: float) -> np.ndarray:
    if label == "circle":
        return dot_image(h, w, cx, cy, radius, contrast, 1)
    if label == "triangle":
        return polygon_image(h, w, regular_polygon(cx, cy, radius, 3, rot), contrast, 1)
    if label == "square":
        return polygon_image(h, w, regular_polygon(cx, cy, radius, 4, rot), contrast, 1)
    if label == "bar_horizontal":
        return bar_image(h, w, cx, cy, 2.4 * radius, 0.35 * radius, 0.0, contrast, 1)
    if label == "bar_vertical":
        return bar_image(h, w, cx, cy, 2.4 * radius, 0.35 * radius, np.pi / 2, contrast, 1)
    if label == "cross":
        return cross_image(h, w, cx, cy, 2.2 * radius, 0.35 * radius, contrast, 1)
    if label == "blank":
        return uniform_image(h, w)
    raise ValueError(f"알 수 없는 도형 라벨: {label!r}")


def _color_illumination(si: int, h: int, w: int, p: dict[str, Any],
                        rng: np.random.Generator) -> list[Stimulus]:
    """색과 조명을 따로 바꾼 자극.

    조명 변화는 곱셈 이득으로 모형화했다. 이것만으로 **색채 항상성이 구현되었다고
    말하지 않는다.** 검증에는 조명 변화 조건의 실제 측정이 필요하다.
    """
    colors = p.get("colors", [[1, 0.2, 0.2], [0.2, 1, 0.2], [0.2, 0.2, 1],
                              [0.9, 0.9, 0.3]])
    gains = p.get("illumination_gains", [0.7, 1.0, 1.3])
    radius = float(p.get("radius_px", 0.18 * w))
    out: list[Stimulus] = []
    for ci, col in enumerate(colors):
        base = f"b{si}_color_{ci}"
        cx = rng.uniform(0.4, 0.6) * w
        cy = rng.uniform(0.4, 0.6) * h
        for gi, g in enumerate(gains):
            img = polygon_image(h, w, regular_polygon(cx, cy, radius, 24), 0.45, 1,
                                color=col)
            img = np.clip(img * float(g), 0.0, 1.0)
            out.append(Stimulus(f"{base}_g{gi}", base, f"color{ci}", [img],
                                params={"kind": "color_illumination", "color": list(col),
                                        "illumination_gain": float(g),
                                        "note_ko": "조명 변화는 곱셈 이득 모형이다."}))
    return out


def _transforms(si: int, h: int, w: int, p: dict[str, Any],
                rng: np.random.Generator) -> list[Stimulus]:
    label = str(p.get("label", "square"))
    positions = p.get("positions", [[0.35, 0.5], [0.5, 0.5], [0.65, 0.5]])
    scales = p.get("scales", [0.7, 1.0, 1.4])
    radius = float(p.get("radius_px", 0.15 * w))
    out: list[Stimulus] = []
    base = f"b{si}_transform_{label}"
    for pi, (fx, fy) in enumerate(positions):
        for si2, sc in enumerate(scales):
            img = _render_shape(label, h, w, fx * w, fy * h, radius * float(sc),
                                0.0, 0.4)
            out.append(Stimulus(f"{base}_p{pi}_s{si2}", base, label, [img],
                                params={"kind": "transform", "position": [fx, fy],
                                        "scale": float(sc)}))
    return out


def _binocular(si: int, h: int, w: int, p: dict[str, Any],
               rng: np.random.Generator) -> list[Stimulus]:
    """좌/우 영상 쌍. ``disparity_px`` 가 0 이면 **복제이며 시차 실험이 아니다**."""
    disparities = p.get("disparity_px", [0.0, 2.0, 4.0])
    radius = float(p.get("radius_px", 0.15 * w))
    out: list[Stimulus] = []
    for di, d in enumerate(disparities):
        cx, cy = 0.5 * w, 0.5 * h
        left = dot_image(h, w, cx - float(d) / 2, cy, radius, 0.4, 1)
        right = dot_image(h, w, cx + float(d) / 2, cy, radius, 0.4, 1)
        out.append(Stimulus(
            f"s{si}_bino_{di}", f"b{si}_bino_{di}", "binocular", [left],
            eyes={"left": left, "right": right},
            params={"kind": "binocular", "disparity_px": float(d),
                    "note_ko": ("disparity_px=0 은 단안 영상 복제이며 실제 양안 시차 "
                                "실험으로 보고하지 않는다.")}))
    return out


def _motion(si: int, h: int, w: int, p: dict[str, Any],
            rng: np.random.Generator) -> list[Stimulus]:
    """막대가 이동하는 프레임 시퀀스. 운동 지표는 이 자극에서만 의미가 있다."""
    n_frames = int(p.get("n_frames", 8))
    speed = float(p.get("speed_px_per_frame", 3.0))
    ori = float(p.get("orientation_rad", 0.0))
    out: list[Stimulus] = []
    for d, direction in enumerate(p.get("directions_rad", [0.0, np.pi])):
        frames = []
        for f in range(n_frames):
            off = (f - n_frames / 2.0) * speed
            cx = 0.5 * w + off * np.cos(direction)
            cy = 0.5 * h + off * np.sin(direction)
            frames.append(bar_image(h, w, cx, cy, float(p.get("length_px", 0.4 * w)),
                                    float(p.get("width_px", 3.0)), ori, 0.4))
        out.append(Stimulus(f"s{si}_motion_{d}", f"b{si}_motion_{d}", "motion",
                            frames,
                            params={"kind": "motion", "direction_rad": float(direction),
                                    "speed_px_per_frame": speed,
                                    "n_frames": n_frames}))
    return out


# ----------------------------------------------------------------------
def split_stimuli(stimuli: Sequence[Stimulus], cfg: dict[str, Any],
                  rng: np.random.Generator) -> dict[str, list[Stimulus]]:
    """train/dev/test 분할. **같은 base_id 의 변형은 한 분할에 묶인다.**

    ``splits.stratified`` 면 라벨별로 비율을 맞춘다.
    """
    sp = cfg["experiment"]["splits"]
    fr = {k: float(sp[k]) for k in ("train", "dev", "test")}
    total = sum(fr.values())
    if total <= 0:
        raise ValueError("splits 비율 합이 0 이다")
    fr = {k: v / total for k, v in fr.items()}

    by_base: dict[str, list[Stimulus]] = {}
    for s in stimuli:
        by_base.setdefault(s.base_id, []).append(s)
    bases = sorted(by_base)
    labels = {b: by_base[b][0].label for b in bases}

    out: dict[str, list[Stimulus]] = {"train": [], "dev": [], "test": []}
    groups: dict[str, list[str]] = {}
    for b in bases:
        key = labels[b] if sp["stratified"] else "_all"
        groups.setdefault(key, []).append(b)

    for key in sorted(groups):
        gb = groups[key]
        order = rng.permutation(len(gb))
        gb = [gb[i] for i in order]
        n = len(gb)
        n_tr = int(round(fr["train"] * n))
        n_dev = int(round(fr["dev"] * n))
        n_tr = min(n_tr, n)
        n_dev = min(n_dev, n - n_tr)
        parts = {"train": gb[:n_tr], "dev": gb[n_tr:n_tr + n_dev],
                 "test": gb[n_tr + n_dev:]}
        for split, bl in parts.items():
            for b in bl:
                out[split] += by_base[b]
    return out


def split_report(splits: dict[str, list[Stimulus]]) -> dict[str, Any]:
    """분할 요약과 **중복 검사** (검증 14번)."""
    bases = {k: {s.base_id for s in v} for k, v in splits.items()}
    ids = {k: {s.stimulus_id for s in v} for k, v in splits.items()}
    overlaps = {}
    keys = list(splits)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            overlaps[f"{a}&{b}_base_ids"] = sorted(bases[a] & bases[b])
            overlaps[f"{a}&{b}_stimulus_ids"] = sorted(ids[a] & ids[b])
    labels = {}
    for k, v in splits.items():
        c: dict[str, int] = {}
        for s in v:
            c[s.label] = c.get(s.label, 0) + 1
        labels[k] = dict(sorted(c.items()))
    return {
        "counts": {k: len(v) for k, v in splits.items()},
        "n_base_ids": {k: len(v) for k, v in bases.items()},
        "labels": labels,
        "overlaps": overlaps,
        "no_overlap": all(len(v) == 0 for v in overlaps.values()),
    }


__all__ = [
    "BACKGROUND", "Stimulus", "uniform_image", "dot_image", "bar_image",
    "grating_patch", "polygon_image", "regular_polygon", "cross_image",
    "center_surround_image", "point_in_polygon", "DEFAULT_SHAPE_CLASSES",
    "generate", "split_stimuli", "split_report",
]
