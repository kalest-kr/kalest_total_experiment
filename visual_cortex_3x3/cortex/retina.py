"""retina.py -- 영상 입력, 색 변환, 중심-주변 DoG, ON/OFF 분리, 입력 구동.

명세 5.1절.

색 변환 경로
------------
``uint8 -> [0,1] -> sRGB 선형화 -> linear RGB -> XYZ -> LMS(근사) -> 대립 채널``

**반드시 지킨 구분**

* ``RGB_TO_XYZ`` 는 RGB→XYZ 행렬이다. 이것을 LMS 행렬이라고 부르지 않는다.
  LMS 로 가려면 ``XYZ_TO_LMS_*`` 를 **추가로** 곱해야 한다.
* 3채널 RGB 입력에서 실제 원추세포 스펙트럼 응답을 **완전히 복원할 수 없다**.
  이 모듈이 만드는 LMS 는 표준 관찰자 기반의 선형 근사이며, 촬영 장치의
  스펙트럼 감도와 조명을 모르는 상태에서의 추정이다.
* 행렬 계수는 널리 인용되는 값을 그대로 적었고 정규화 조건을 함께 기록했다.
  1차 출처 원문을 이번 작업에서 직접 대조 확인하지 않았으므로
  BIOLOGY_AND_ASSUMPTIONS.md 에 **가정**으로 표시한다.

대비 경로와 저주파 경로
-----------------------
기본 대비 경로는 대립 3채널 × ON/OFF = **6개의 비음수 채널**이다.
DoG 만 쓰면 균일한 색/저주파 정보가 사라지므로, V4 색 과제를 위해
저주파 LMS 3채널을 **별도 경로**로 유지하는 옵션을 둔다
(``retina.channels.keep_lowpass_lms``). 합류 지점은 배선 규칙에서
``channel_id`` 로 지정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

from .units import MS_PER_S

# ----------------------------------------------------------------------
# 색 변환 행렬 (실제 수치와 정규화 조건을 코드에 남긴다)
# ----------------------------------------------------------------------
#: 선형 sRGB(D65) -> CIE XYZ. IEC 61966-2-1 의 sRGB 원색/백색점에서 유도된 값.
#: 행 합은 D65 백색점 (0.95047, 1.00000, 1.08883) 이 된다.
#: **이것은 XYZ 행렬이며 LMS 행렬이 아니다.**
RGB_TO_XYZ: np.ndarray = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

#: XYZ -> LMS, Hunt-Pointer-Estevez 행렬을 **D65 에 정규화**한 형태.
XYZ_TO_LMS_HPE_D65: np.ndarray = np.array([
    [0.4002, 0.7076, -0.0808],
    [-0.2263, 1.1653, 0.0457],
    [0.0, 0.0, 0.9182],
], dtype=np.float64)

#: XYZ -> LMS, Stockman & Sharpe 계열 원추 기본함수에서 유도되어 널리 인용되는 행렬.
XYZ_TO_LMS_STOCKMAN_SHARPE: np.ndarray = np.array([
    [0.210576, 0.855098, -0.0396983],
    [-0.417076, 1.177260, 0.0786283],
    [0.0, 0.0, 0.5168350],
], dtype=np.float64)

LMS_MATRICES: dict[str, np.ndarray] = {
    "hunt_pointer_estevez_d65": XYZ_TO_LMS_HPE_D65,
    "stockman_sharpe_lms": XYZ_TO_LMS_STOCKMAN_SHARPE,
}

MATRIX_PROVENANCE: dict[str, str] = {
    "RGB_TO_XYZ": (
        "선형 sRGB(D65) -> CIE XYZ. IEC 61966-2-1 의 sRGB 원색과 D65 백색점에서 "
        "유도되는 표준 계수. 정규화: 행 합 = D65 백색점."
    ),
    "hunt_pointer_estevez_d65": (
        "Hunt-Pointer-Estevez 원추 응답 행렬의 D65 정규화 형태. "
        "1차 출처 원문을 이번 작업에서 직접 대조하지 않았으므로 **가정**으로 표시한다."
    ),
    "stockman_sharpe_lms": (
        "Stockman & Sharpe 계열 2도 원추 기본함수에서 유도되어 널리 인용되는 XYZ->LMS 행렬. "
        "1차 출처 원문을 이번 작업에서 직접 대조하지 않았으므로 **가정**으로 표시한다."
    ),
}

#: 대립 채널 정의 (모형 선택이며 특정 망막 신경절 세포의 측정 가중치가 아니다)
OPPONENT_MATRIX: np.ndarray = np.array([
    [0.6, 0.4, 0.0],     # luminance   ~ L+M
    [1.0, -1.0, 0.0],    # red_green   ~ L-M
    [-0.5, -0.5, 1.0],   # blue_yellow ~ S-(L+M)/2
], dtype=np.float64)

OPPONENT_NAMES: tuple[str, ...] = ("luminance", "red_green", "blue_yellow")


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB 전달함수 역변환. 입력/출력 모두 [0,1]."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_float01(image: np.ndarray) -> np.ndarray:
    """uint8/uint16/float 입력을 [0,1] float64 로 실수화한다.

    float 입력이 [0,1] 을 벗어나면 조용히 스케일을 추정하지 않고 오류를 낸다.
    """
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float64) / 65535.0
    arr = arr.astype(np.float64)
    if arr.size and (arr.min() < -1e-9 or arr.max() > 1.0 + 1e-9):
        raise ValueError(
            f"float 영상은 [0,1] 범위여야 한다 (관측 [{arr.min():.4g}, {arr.max():.4g}]). "
            f"uint8 로 넘기거나 미리 정규화하라."
        )
    return np.clip(arr, 0.0, 1.0)


def gaussian_kernel_1d(sigma: float, truncate: float) -> np.ndarray:
    """합이 1 이 되도록 정규화한 1D 가우시안 커널."""
    radius = int(max(1, round(truncate * float(sigma))))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum()


@dataclass
class RetinaOutput:
    """전처리 결과.

    Attributes
    ----------
    channels : (C, H, W) float64, 비음수
        대비 경로 6채널 (+ 선택적 저주파 LMS 3채널).
    channel_names : list[str]
    lms : (3, H, W) float64
        선형 LMS 근사값 (진단/시각화용).
    opponent : (3, H, W) float64
        DoG 이전의 대립 신호.
    meta : dict
    """

    channels: np.ndarray
    channel_names: list[str]
    lms: np.ndarray
    opponent: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


class RetinaEncoder:
    """색 변환 + DoG + ON/OFF + (선택) 저주파 LMS 경로.

    부작용: :meth:`fit_normalization` 만 내부 스케일을 바꾼다. :meth:`encode` 는
    순수 함수다.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        rc = cfg["retina"]
        self.input_colorspace = rc["image"]["input_colorspace"]
        self.use_lms = bool(rc["color"]["use_lms"])
        self.lms_matrix_name = rc["color"]["lms_matrix"]
        self.xyz_to_lms = LMS_MATRICES[self.lms_matrix_name]
        self.dog = rc["dog"]
        self.on_off_split = bool(rc["channels"]["on_off_split"])
        self.keep_lowpass_lms = bool(rc["channels"]["keep_lowpass_lms"])
        self.lowpass_sigma_px = float(rc["channels"]["lowpass_sigma_px"])

        #: 정규화 스케일. **훈련 데이터로만** 추정한다 (fit_normalization).
        self.scale: np.ndarray | None = None
        self.normalization_fitted = False
        self.normalization_source = ""

    # ------------------------------------------------------------------
    def channel_names(self) -> list[str]:
        names: list[str] = []
        for base in OPPONENT_NAMES:
            if self.on_off_split:
                names += [f"{base}_ON", f"{base}_OFF"]
            else:
                names.append(base)
        if self.keep_lowpass_lms:
            names += ["lowpass_L", "lowpass_M", "lowpass_S"]
        return names

    @property
    def n_channels(self) -> int:
        return len(self.channel_names())

    # ------------------------------------------------------------------
    def to_lms(self, image: np.ndarray) -> np.ndarray:
        """(H,W,3) 또는 (H,W) 영상 -> (3,H,W) LMS 근사.

        회색조 입력은 R=G=B 로 확장한다 (색 정보가 없다는 사실을 meta 에 남긴다).
        """
        arr = to_float01(image)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"지원하지 않는 영상 shape: {arr.shape}")
        rgb = arr[:, :, :3]
        lin = srgb_to_linear(rgb) if self.input_colorspace == "srgb" else rgb
        xyz = lin @ RGB_TO_XYZ.T                      # (H,W,3)
        if not self.use_lms:
            return np.moveaxis(xyz, 2, 0)
        lms = xyz @ self.xyz_to_lms.T
        return np.moveaxis(lms, 2, 0)                  # (3,H,W)

    def opponent_signals(self, lms: np.ndarray) -> np.ndarray:
        """(3,H,W) LMS -> (3,H,W) 대립 신호 (luminance, red_green, blue_yellow)."""
        flat = lms.reshape(3, -1)
        return (OPPONENT_MATRIX @ flat).reshape(lms.shape)

    def difference_of_gaussians(self, plane: np.ndarray) -> np.ndarray:
        """정규화된 가우시안 두 개의 차. 균일 입력의 응답은 경계 오차 수준이다."""
        c = ndimage.gaussian_filter(
            plane, self.dog["center_sigma_px"], mode=self.dog["boundary_mode"],
            truncate=self.dog["truncate"])
        s = ndimage.gaussian_filter(
            plane, self.dog["surround_sigma_px"], mode=self.dog["boundary_mode"],
            truncate=self.dog["truncate"])
        return c - s

    def encode(self, image: np.ndarray) -> RetinaOutput:
        """영상 1장 -> 비음수 채널 스택 (부작용 없음)."""
        lms = self.to_lms(image)
        opp = self.opponent_signals(lms)
        planes: list[np.ndarray] = []
        for i in range(3):
            d = self.difference_of_gaussians(opp[i])
            if self.on_off_split:
                planes.append(np.maximum(d, 0.0))
                planes.append(np.maximum(-d, 0.0))
            else:
                planes.append(d)
        if self.keep_lowpass_lms:
            for i in range(3):
                lp = ndimage.gaussian_filter(
                    lms[i], self.lowpass_sigma_px, mode=self.dog["boundary_mode"],
                    truncate=self.dog["truncate"])
                planes.append(np.maximum(lp, 0.0))
        channels = np.stack(planes, axis=0)
        return RetinaOutput(
            channels=channels,
            channel_names=self.channel_names(),
            lms=lms,
            opponent=opp,
            meta={
                "input_colorspace": self.input_colorspace,
                "lms_matrix": self.lms_matrix_name,
                "lms_matrix_provenance": MATRIX_PROVENANCE[self.lms_matrix_name],
                "rgb_to_xyz_provenance": MATRIX_PROVENANCE["RGB_TO_XYZ"],
                "dog": dict(self.dog),
                "grayscale_input_expanded": bool(np.asarray(image).ndim == 2),
                "cone_spectra_note_ko": (
                    "3채널 RGB 에서 실제 원추세포 스펙트럼 응답을 복원한 것이 아니다. "
                    "표준 관찰자 기반의 선형 근사다."
                ),
            },
        )

    # ------------------------------------------------------------------
    def fit_normalization(self, images: Sequence[np.ndarray], percentile: float = 99.0,
                          source: str = "train") -> dict[str, Any]:
        """채널별 스케일을 **훈련 영상으로만** 추정한다 (부작용: self.scale 설정).

        dev/test 영상으로 다시 추정하지 않는다. 추정에 쓴 분할 이름을 기록한다.
        """
        if not images:
            raise ValueError("정규화 추정에 쓸 영상이 없다")
        acc: list[np.ndarray] = []
        for img in images:
            out = self.encode(img)
            acc.append(out.channels.reshape(out.channels.shape[0], -1))
        allv = np.concatenate(acc, axis=1)
        scale = np.percentile(np.where(allv > 0, allv, np.nan), percentile, axis=1)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        self.scale = scale.astype(np.float64)
        self.normalization_fitted = True
        self.normalization_source = source
        return {
            "percentile": float(percentile),
            "source_split": source,
            "n_images": len(images),
            "scale": self.scale.tolist(),
            "degenerate_channels": [i for i, s in enumerate(self.scale) if s == 1.0],
        }

    def normalize(self, channels: np.ndarray) -> np.ndarray:
        """추정한 스케일로 나누고 [0,1] 로 자른다.

        스케일을 아직 추정하지 않았다면 오류를 낸다 (조용히 1.0 을 쓰지 않는다).
        """
        if not self.normalization_fitted or self.scale is None:
            raise RuntimeError(
                "채널 정규화 스케일이 아직 추정되지 않았다. "
                "RetinaEncoder.fit_normalization(train_images) 를 먼저 호출하라."
            )
        return np.clip(channels / self.scale[:, None, None], 0.0, 1.0)

    def normalization_state(self) -> dict[str, Any]:
        return {
            "fitted": self.normalization_fitted,
            "source_split": self.normalization_source,
            "scale": None if self.scale is None else self.scale.tolist(),
        }


# ----------------------------------------------------------------------
# 입력 구동 (rate / Poisson)
# ----------------------------------------------------------------------
@dataclass
class ExogenousEventSequence:
    """외생 입력 사건열. 대조군에서 **그대로 재생**할 수 있도록 저장한다.

    Attributes
    ----------
    step : (E,) int64
    neuron_id : (E,) int32
    count : (E,) int32
        해당 스텝의 사건 수 (Poisson 모드에서 1보다 클 수 있다).
    mode : str
    rng_stream : str
    """

    step: np.ndarray
    neuron_id: np.ndarray
    count: np.ndarray
    mode: str
    rng_stream: str

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step, "neuron_id": self.neuron_id,
                "count": self.count, "mode": self.mode, "rng_stream": self.rng_stream}

    @staticmethod
    def from_state_dict(d: dict[str, Any]) -> "ExogenousEventSequence":
        return ExogenousEventSequence(
            np.asarray(d["step"], dtype=np.int64),
            np.asarray(d["neuron_id"], dtype=np.int32),
            np.asarray(d["count"], dtype=np.int32),
            str(d["mode"]), str(d["rng_stream"]),
        )


class InputDriver:
    """샘플된 채널값 -> 입력 뉴런 구동.

    ``rate`` 모드: 연속 발화율(Hz)을 그대로 외부 전류/활성화로 쓴다.
    ``poisson`` 모드: dt 동안의 사건 수를 ``Poisson(rate*dt/1000)`` 로 뽑는다.

    Hz 와 ms 를 혼동하지 않는다. 변환은 :func:`cortex.units.expected_events`.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        d = cfg["retina"]["drive"]
        self.mode = d["mode"]
        self.max_rate_hz = float(d["max_rate_hz"])
        self.gain = float(d["gain"])
        self.baseline_rate_hz = float(d["baseline_rate_hz"])
        self.dt_ms = float(cfg["engine"]["dt_ms"])

    def rates_hz(self, normalized_values: np.ndarray) -> np.ndarray:
        """[0,1] 정규화 채널값 -> 발화율 [Hz] (비음수)."""
        v = np.asarray(normalized_values, dtype=np.float64)
        return np.maximum(0.0, self.baseline_rate_hz + self.gain * self.max_rate_hz * v)

    def sample_counts(self, rates_hz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """dt 동안의 사건 수. rate 모드에서는 결정적 기대값(반올림 없음)이다."""
        lam = np.asarray(rates_hz, dtype=np.float64) * self.dt_ms / MS_PER_S
        if self.mode == "rate":
            return lam
        return rng.poisson(lam).astype(np.float64)

    def build_sequence(self, rates_hz: np.ndarray, neuron_ids: np.ndarray,
                       n_steps: int, rng: np.random.Generator,
                       rng_stream: str) -> ExogenousEventSequence:
        """전체 스텝의 외생 사건열을 **미리** 만든다.

        미리 만들어 두면 자유/유도 대조 실행에서 동일한 외생 사건을 그대로
        재생할 수 있다 (명세 9절).
        """
        steps: list[np.ndarray] = []
        ids: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        for s in range(int(n_steps)):
            c = self.sample_counts(rates_hz, rng)
            nz = np.nonzero(c > 0)[0]
            if nz.size == 0:
                continue
            steps.append(np.full(nz.size, s, dtype=np.int64))
            ids.append(np.asarray(neuron_ids, dtype=np.int32)[nz])
            counts.append(c[nz].astype(np.float64))
        if not steps:
            return ExogenousEventSequence(
                np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int32),
                self.mode, rng_stream)
        return ExogenousEventSequence(
            np.concatenate(steps),
            np.concatenate(ids).astype(np.int32),
            np.concatenate(counts).astype(np.int32) if self.mode == "poisson"
            else np.concatenate(counts).astype(np.float64).astype(np.int32),
            self.mode, rng_stream,
        )


def uniform_response_check(encoder: RetinaEncoder, value: float = 0.5,
                           size: int = 64) -> dict[str, Any]:
    """균일 영상에서 대비 경로의 잔여 응답을 측정한다 (검증 7번).

    이상적으로는 0 이어야 하며 실제로는 경계 처리에서 작은 값이 남는다.
    이 함수는 **측정만** 하고 통과/실패를 판정하지 않는다.
    """
    img = np.full((size, size, 3), float(value), dtype=np.float64)
    out = encoder.encode(img)
    n_contrast = 6 if encoder.on_off_split else 3
    contrast = out.channels[:n_contrast]
    inner = contrast[:, size // 4: 3 * size // 4, size // 4: 3 * size // 4]
    return {
        "uniform_value": float(value),
        "size": int(size),
        "max_abs_contrast_response": float(np.abs(contrast).max()),
        "max_abs_contrast_response_interior": float(np.abs(inner).max()),
        "mean_abs_contrast_response": float(np.abs(contrast).mean()),
        "boundary_mode": encoder.dog["boundary_mode"],
        "note_ko": ("경계 처리 때문에 가장자리에 잔여 응답이 남을 수 있다. "
                    "interior 값이 본질적인 DoG 상수 응답 오차에 가깝다."),
    }


def load_image(path: str) -> np.ndarray:
    """Pillow 로 영상을 읽어 (H,W,3) uint8 로 돌려준다.

    이 함수는 **파일을 읽기만** 한다. 다운로드하거나 데이터를 만들지 않는다.
    """
    from PIL import Image  # 지연 import: 모듈 import 부작용 방지

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


__all__ = [
    "RGB_TO_XYZ", "XYZ_TO_LMS_HPE_D65", "XYZ_TO_LMS_STOCKMAN_SHARPE",
    "LMS_MATRICES", "MATRIX_PROVENANCE", "OPPONENT_MATRIX", "OPPONENT_NAMES",
    "srgb_to_linear", "to_float01", "gaussian_kernel_1d",
    "RetinaOutput", "RetinaEncoder", "InputDriver", "ExogenousEventSequence",
    "uniform_response_check", "load_image",
]
