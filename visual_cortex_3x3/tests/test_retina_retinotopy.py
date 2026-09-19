"""색 변환·DoG·로그-극좌표 테스트 (명세 5절)."""

from __future__ import annotations

import numpy as np

from cortex.retina import (
    RGB_TO_XYZ,
    XYZ_TO_LMS_HPE_D65,
    RetinaEncoder,
    srgb_to_linear,
    to_float01,
    uniform_response_check,
)
from cortex.retinotopy import (
    build_grid,
    ecc_to_rho,
    px_per_deg,
    rho_to_ecc,
    roundtrip_error,
)


def test_matrices_are_distinct():
    assert not np.allclose(RGB_TO_XYZ, XYZ_TO_LMS_HPE_D65), \
        "RGB->XYZ 행렬을 LMS 행렬이라고 부르면 안 된다"


def test_srgb_linearization_endpoints():
    assert abs(float(srgb_to_linear(np.array(0.0)))) < 1e-12
    assert abs(float(srgb_to_linear(np.array(1.0))) - 1.0) < 1e-12
    assert float(srgb_to_linear(np.array(0.5))) < 0.5


def test_uint8_conversion():
    a = to_float01(np.array([[0, 255]], dtype=np.uint8))
    assert a.tolist() == [[0.0, 1.0]]


def test_dog_uniform_interior_is_small(v1_cfg):
    enc = RetinaEncoder(v1_cfg)
    res = uniform_response_check(enc, 0.5, 64)
    assert res["max_abs_contrast_response_interior"] < 1e-6


def test_channel_count(v1_cfg):
    enc = RetinaEncoder(v1_cfg)
    assert enc.n_channels == 9   # 대립 3 x ON/OFF + 저주파 LMS 3


def test_normalization_requires_fit(v1_cfg):
    enc = RetinaEncoder(v1_cfg)
    try:
        enc.normalize(np.zeros((enc.n_channels, 4)))
    except RuntimeError:
        return
    raise AssertionError("fit 없이 normalize 가 통과하면 안 된다")


def test_logpolar_roundtrip():
    e0 = 0.5
    ecc = np.array([0.0, 0.5, 2.0, 8.0])
    assert np.allclose(rho_to_ecc(ecc_to_rho(ecc, e0), e0), ecc, atol=1e-12)


def test_grid_roundtrip_small(v1_cfg):
    g = build_grid(v1_cfg)
    err = roundtrip_error(g, v1_cfg["retinotopy"]["e0_deg"])
    assert err["max_abs_ecc_error_deg"] < 1e-9


def test_px_per_deg_uses_long_side():
    assert abs(px_per_deg(100, 200, 20.0) - 10.0) < 1e-12


def test_grid_samples_inside_image(v1_cfg):
    g = build_grid(v1_cfg)
    H = W = v1_cfg["retina"]["image"]["max_side_px"]
    col, row, valid = g.to_pixels(H, W, v1_cfg["retina"]["image"]["fov_deg"])
    assert valid.all(), "설정한 최대 편심도의 샘플이 영상 안에 있어야 한다"
