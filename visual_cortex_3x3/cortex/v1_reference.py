"""v1_reference.py -- 고정 Gabor 대조 경로와 위상 불변 에너지.

명세 7절.

**이 경로에서 방향 선택성이 "학습되었다"고 말하지 않는다.** 필터 계수를 사람이
정해 넣은 고정 특징 추출기이며, 피질 회로 경로(:mod:`cortex.areas` 의 배선과
:mod:`cortex.dynamics` 의 동역학)와 **혼용해 보고하지 않는다**.

참조 에너지::

    energy = sqrt(r0**2 + rq**2 + eps**2) - eps

여기서 ``r0``, ``rq`` 는 위상 0 과 -pi/2 필터의 응답이다. 유한 필터, 경계 처리,
공간 왜곡 조건에서 **완벽한 위상 불변성을 보장하지 않는다.** 위상 스윕과 명암
반전 검사로 실제 불변성을 측정해야 한다 (:func:`phase_sweep_response`).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .areas import gabor_coefficient
from .retinotopy import SamplingGrid


class FixedGaborReference:
    """격자 샘플 위에서 계산하는 고정 Gabor 특징 추출기.

    필터는 **원본 시야 Cartesian 좌표**에서 정의하고 각 수용장 안에서 평가한다.
    왜곡된 로그-극좌표 배열 위에 직선 필터를 올려 쓰지 않는다.
    """

    def __init__(self, grid: SamplingGrid, cfg: dict[str, Any]) -> None:
        ref = cfg["v1"]["fixed_gabor_reference"]
        self.enabled = bool(ref["enabled"])
        self.sigma_deg = float(ref["sigma_deg"])
        self.aspect = float(ref["aspect"])
        self.cycles_per_deg = float(ref["cycles_per_deg"])
        self.eps = float(ref["energy_eps"])
        self.grid = grid
        self.n_orientations = int(cfg["v1"]["n_orientations"])
        self.orientation_step_deg = float(cfg["v1"]["orientation_step_deg"])
        self.phases = [float(p) for p in cfg["v1"]["phases_rad"]]
        self.orientations_rad = np.deg2rad(
            np.arange(self.n_orientations) * self.orientation_step_deg)

    def filter_bank(self, centers: np.ndarray) -> np.ndarray:
        """(n_centers, n_orientations, n_phases, n_samples) 필터 계수.

        메모리 사용이 크므로 작은 설정에서만 쓴다. 필요한 중심만 넘겨라.
        """
        centers = np.asarray(centers, dtype=np.float64)
        n_c = centers.shape[0]
        dx = self.grid.x_deg[None, :] - centers[:, 0:1]
        dy = self.grid.y_deg[None, :] - centers[:, 1:2]
        out = np.zeros((n_c, self.n_orientations, len(self.phases),
                        self.grid.n_samples), dtype=np.float64)
        for oi, ori in enumerate(self.orientations_rad):
            for pi, ph in enumerate(self.phases):
                out[:, oi, pi, :] = gabor_coefficient(
                    dx, dy, np.full_like(dx, ori), np.full_like(dx, ph),
                    self.sigma_deg, self.aspect, self.cycles_per_deg)
        return out

    def responses(self, sampled_channel: np.ndarray, centers: np.ndarray
                  ) -> dict[str, np.ndarray]:
        """샘플된 1채널 신호 -> 필터 응답과 에너지.

        Parameters
        ----------
        sampled_channel : (S,) float64
        centers : (n_centers, 2) float64  시야 좌표 [deg]

        Returns
        -------
        dict : ``linear`` (n_c, n_ori, n_phase), ``energy`` (n_c, n_ori)
        """
        bank = self.filter_bank(centers)
        lin = np.einsum("copn,n->cop", bank, np.asarray(sampled_channel, float))
        if lin.shape[2] >= 2:
            r0, rq = lin[:, :, 0], lin[:, :, 1]
        else:
            r0, rq = lin[:, :, 0], np.zeros_like(lin[:, :, 0])
        energy = np.sqrt(r0 ** 2 + rq ** 2 + self.eps ** 2) - self.eps
        return {
            "linear": lin, "energy": energy,
            "orientations_rad": self.orientations_rad,
        }


def phase_sweep_response(ref: FixedGaborReference, sampled_by_phase: Sequence[np.ndarray],
                         centers: np.ndarray) -> dict[str, Any]:
    """위상 스윕에서 선형 응답과 에너지의 변동을 **측정**한다.

    완벽한 위상 불변성을 주장하지 않는다. 변동 계수를 그대로 보고한다.
    """
    lins, ens = [], []
    for s in sampled_by_phase:
        r = ref.responses(s, centers)
        lins.append(r["linear"][:, :, 0])
        ens.append(r["energy"])
    lin = np.stack(lins, axis=0)      # (n_phase, n_c, n_ori)
    en = np.stack(ens, axis=0)
    def cv(x: np.ndarray) -> np.ndarray:
        m = np.abs(x).mean(axis=0)
        s = x.std(axis=0)
        return s / np.maximum(m, 1e-12)
    return {
        "n_phases": int(lin.shape[0]),
        "linear_cv_mean": float(np.nanmean(cv(lin))),
        "energy_cv_mean": float(np.nanmean(cv(en))),
        "linear_range_mean": float(np.nanmean(lin.max(axis=0) - lin.min(axis=0))),
        "energy_range_mean": float(np.nanmean(en.max(axis=0) - en.min(axis=0))),
        "note_ko": ("유한 필터·경계·공간 왜곡 조건에서 완벽한 위상 불변성을 보장하지 "
                    "않는다. 위 변동 계수는 측정값이다."),
    }


def contrast_reversal_response(ref: FixedGaborReference, sampled: np.ndarray,
                               sampled_reversed: np.ndarray,
                               centers: np.ndarray) -> dict[str, Any]:
    """명암 반전 자극에서 선형 응답과 에너지의 변화를 측정한다."""
    a = ref.responses(sampled, centers)
    b = ref.responses(sampled_reversed, centers)
    return {
        "linear_sign_flip_fraction": float(np.mean(
            np.sign(a["linear"][:, :, 0]) != np.sign(b["linear"][:, :, 0]))),
        "energy_relative_change_mean": float(np.nanmean(
            np.abs(b["energy"] - a["energy"]) / np.maximum(np.abs(a["energy"]), 1e-12))),
        "note_ko": "에너지가 명암 반전에 얼마나 둔감한지 측정한 값이다.",
    }


def orientation_tuning(energy: np.ndarray, orientations_rad: np.ndarray
                       ) -> dict[str, Any]:
    """에너지 (n_c, n_ori) 에서 방향 튜닝 지표를 계산한다.

    OSI = (R_pref - R_orth) / (R_pref + R_orth). 측정 지표이며 기준을 사후에
    바꾸지 않는다.
    """
    e = np.asarray(energy, dtype=np.float64)
    n_ori = e.shape[1]
    pref_idx = np.argmax(e, axis=1)
    pref = e[np.arange(e.shape[0]), pref_idx]
    orth_idx = (pref_idx + n_ori // 2) % n_ori
    orth = e[np.arange(e.shape[0]), orth_idx]
    osi = (pref - orth) / np.maximum(pref + orth, 1e-12)
    return {
        "preferred_orientation_rad": orientations_rad[pref_idx].tolist(),
        "osi": osi.tolist(),
        "osi_mean": float(np.nanmean(osi)),
        "n_centers": int(e.shape[0]),
        "learned": False,
        "note_ko": "고정 Gabor 대조 경로의 튜닝이다. 학습된 선택성이 아니다.",
    }


__all__ = ["FixedGaborReference", "phase_sweep_response",
           "contrast_reversal_response", "orientation_tuning"]
