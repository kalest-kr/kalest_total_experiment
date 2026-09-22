"""retina.py -- 명암(휘도) 경로 전처리: 회색조 변환 + DoG 중심-주변 + ON/OFF 분리.

명세 [6]. **이것은 완전한 망막 색채 회로가 아니다.** 휘도 한 경로만 구현한다.

권장 기본 처리 (config 로 변경 가능):

1. 입력을 [0,1] 실수로 변환 (uint8 은 /255).
2. RGB 는 sRGB 선형화 후 Rec.709 휘도 근사로 회색조.
   float 입력이 선형값인지 sRGB 인지는 ``input_colorspace`` 인자로 **명시**한다.
3. 정규화된 가우시안 두 개의 차 (sigma_center=1, sigma_surround=3,
   mode='reflect', truncate=3).
4. ON = max(DoG, 0), OFF = max(-DoG, 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

# Rec.709 / sRGB 휘도 계수 (선형 RGB 에 적용)
LUMA_REC709 = (0.2126, 0.7152, 0.0722)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB [0,1] -> 선형 [0,1] (IEC 61966-2-1 전달함수)."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_float01(image: np.ndarray) -> np.ndarray:
    """uint8/uint16/float 입력을 [0,1] float64 로 변환한다.

    float 입력은 범위를 검사하고 [0,1] 밖이면 ValueError 를 낸다
    (조용한 스케일 추정을 하지 않는다).
    """
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float64) / 65535.0
    arr = arr.astype(np.float64)
    if arr.size and (arr.min() < -1e-9 or arr.max() > 1.0 + 1e-9):
        raise ValueError(
            f"float 이미지는 [0,1] 범위여야 한다 (관측 [{arr.min():.4g},{arr.max():.4g}])"
        )
    return np.clip(arr, 0.0, 1.0)


@dataclass
class RetinaConfig:
    sigma_center: float = 1.0
    sigma_surround: float = 3.0
    truncate: float = 3.0
    mode: str = "reflect"
    channel_order: str = "RGB"


class RetinaEncoder:
    """휘도 DoG ON/OFF 부호기.

    부작용 없음 (순수 함수형). 학습되는 파라미터가 없다.
    """

    def __init__(self, cfg: RetinaConfig | None = None) -> None:
        self.cfg = cfg or RetinaConfig()

    # --- 회색조 ------------------------------------------------------------
    def to_gray(self, image: np.ndarray, input_colorspace: str = "srgb") -> np.ndarray:
        """입력을 선형 휘도 회색조 (H,W) float64 [0,1] 로 변환한다.

        Parameters
        ----------
        image : np.ndarray
            (H,W) 또는 (H,W,3) 또는 (H,W,4). 채널 순서는 cfg.channel_order.
        input_colorspace : {"srgb", "linear"}
            **반드시 명시**한다. "srgb" 면 선형화를 거친다.

        Returns
        -------
        (H, W) float64, [0,1] 선형 휘도.
        """
        if input_colorspace not in ("srgb", "linear"):
            raise ValueError("input_colorspace 는 'srgb' 또는 'linear' 여야 한다")
        arr = to_float01(image)
        if arr.ndim == 2:
            return srgb_to_linear(arr) if input_colorspace == "srgb" else arr
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"지원하지 않는 이미지 shape: {arr.shape}")
        rgb = arr[:, :, :3]
        if self.cfg.channel_order.upper() == "BGR":
            rgb = rgb[:, :, ::-1]
        if input_colorspace == "srgb":
            rgb = srgb_to_linear(rgb)
        w = np.array(LUMA_REC709, dtype=np.float64)
        return rgb @ w

    # --- DoG / ON-OFF ------------------------------------------------------
    def dog(self, gray: np.ndarray) -> np.ndarray:
        """정규화된 가우시안 두 개의 차. 반환 shape 는 입력과 같다."""
        c = gaussian_filter(
            gray, self.cfg.sigma_center, mode=self.cfg.mode, truncate=self.cfg.truncate
        )
        s = gaussian_filter(
            gray, self.cfg.sigma_surround, mode=self.cfg.mode, truncate=self.cfg.truncate
        )
        return c - s

    def encode(
        self, image: np.ndarray, input_colorspace: str = "srgb"
    ) -> dict[str, np.ndarray]:
        """이미지 -> ON/OFF 채널.

        Returns
        -------
        dict with keys
            ``gray``  : (H,W) float64 선형 휘도
            ``dog``   : (H,W) float64
            ``on``    : (H,W) float64, >= 0
            ``off``   : (H,W) float64, >= 0

        부작용 없음. 단위: 무차원 대비값.
        """
        gray = self.to_gray(image, input_colorspace=input_colorspace)
        d = self.dog(gray)
        return {
            "gray": gray,
            "dog": d,
            "on": np.maximum(d, 0.0),
            "off": np.maximum(-d, 0.0),
        }
