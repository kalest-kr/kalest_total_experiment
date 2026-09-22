"""teacher.py -- 외부 교사(목표 활동 지도)와 L1 교정 전달 인터페이스.

명세 [8], [10].

**정보상의 가정 (반드시 명시)**: 이 교사는 도형의 각도·좌표를 이미 알고 있고,
어느 뉴런에 어떤 값을 보낼지도 사람이 설계한 규칙으로 알고 있다.
이것은 최종 분류 오차 하나에서 자동 역산한 결과가 **아니다**.
학생(감각 경로)은 이미지 픽셀만 받는다.

L2 뉴런 i 의 이진 목표:

* 빈 화면 -> 0
* d_i = 선분(무한 직선 아님)과 수용장 중심의 최소 거리
* 방향 차이는 180° 주기의 최소 차이
* ``d_i <= rf_radius_i/2`` 이고 ``방향차 <= 15°`` 이면 1, 아니면 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .datasets import SceneSpec, point_segment_distance


def orientation_difference_deg(a: np.ndarray | float, b: float, period: float = 180.0) -> np.ndarray:
    """180° 주기의 최소 방향 차이 (도)."""
    d = np.abs(np.asarray(a, dtype=np.float64) - float(b)) % period
    return np.minimum(d, period - d)


@dataclass
class L2Descriptor:
    """교사가 읽는 L2 뉴런 기술자 (감각 경로와 분리)."""

    neuron_ids: np.ndarray        # (M,) int64
    rf_center_x: np.ndarray       # (M,) float64, 원본 영상 좌표
    rf_center_y: np.ndarray       # (M,) float64
    rf_radius: np.ndarray         # (M,) float64
    orientation_deg: np.ndarray   # (M,) float64

    @staticmethod
    def from_circuit(circuit) -> "L2Descriptor":
        pop = circuit.population
        ids = circuit.l2_ids
        cx = np.array([pop.neurons[i].receptive_field.center_x for i in ids])
        cy = np.array([pop.neurons[i].receptive_field.center_y for i in ids])
        rr = np.array([pop.neurons[i].receptive_field.radius_px for i in ids])
        ori = circuit.l2_orientation.copy()
        return L2Descriptor(ids, cx, cy, rr, ori)


class LocalTeacher:
    """뉴런별 목표 활동 지도를 만드는 외부 교사.

    Parameters
    ----------
    descriptor : L2Descriptor
    orientation_tolerance_deg : float
    distance_divisor : float
        ``d_i <= rf_radius_i / distance_divisor`` (기본 2.0).
    """

    def __init__(
        self,
        descriptor: L2Descriptor,
        orientation_tolerance_deg: float = 15.0,
        distance_divisor: float = 2.0,
        orientation_period_deg: float = 180.0,
    ) -> None:
        self.desc = descriptor
        self.tol = float(orientation_tolerance_deg)
        self.div = float(distance_divisor)
        self.period = float(orientation_period_deg)

    def make_targets(
        self, scene_metadata: SceneSpec, neurons: L2Descriptor | None = None
    ) -> np.ndarray:
        """장면 메타데이터 -> L2 목표 지도.

        Returns
        -------
        np.ndarray, shape (M,), dtype int8, 값 0/1
            ``descriptor.neuron_ids`` 순서와 일치한다. 부작용 없음.
        """
        d = neurons or self.desc
        M = d.neuron_ids.size
        if scene_metadata.shape_type == "blank" or scene_metadata.segment_endpoints is None:
            return np.zeros(M, dtype=np.int8)
        p0, p1 = scene_metadata.segment_endpoints
        dist = point_segment_distance(d.rf_center_x, d.rf_center_y, p0, p1)
        ori_diff = orientation_difference_deg(
            d.orientation_deg, float(scene_metadata.orientation_deg), self.period
        )
        hit = (dist <= d.rf_radius / self.div) & (ori_diff <= self.tol)
        return hit.astype(np.int8)

    def make_targets_batch(self, scenes: Sequence[SceneSpec]) -> np.ndarray:
        """(n_scenes, M) int8 목표 행렬."""
        return np.stack([self.make_targets(s) for s in scenes], axis=0)

    # --- 활동 교정값 -------------------------------------------------------
    def make_corrections(
        self,
        u_free: np.ndarray,
        targets: np.ndarray,
        thresholds: np.ndarray,
        margin: float = 0.05,
        error_override_zeros: bool = False,
    ) -> np.ndarray:
        """자유 실행 u 와 목표로부터 교정 입력 c_i 를 만든다 (명세 [10]).

        Returns (M,) float64. ``error_override_zeros`` 면 전부 0.
        """
        if error_override_zeros:
            return np.zeros_like(np.asarray(u_free, dtype=np.float64))
        u = np.asarray(u_free, dtype=np.float64)
        t = np.asarray(targets)
        th = np.asarray(thresholds, dtype=np.float64)
        pos = np.maximum(0.0, th + margin - u)
        neg = np.minimum(0.0, th - margin - u)
        return np.where(t == 1, pos, neg)

    def make_errors(
        self, targets: np.ndarray, q_free: np.ndarray, error_override_zeros: bool = False
    ) -> np.ndarray:
        """d_i = target_i - q_i_free (이진 활동 차이). 로짓 오차나 손실 기울기가 아니다."""
        if error_override_zeros:
            return np.zeros(np.asarray(targets).shape, dtype=np.float64)
        return np.asarray(targets, dtype=np.float64) - np.asarray(q_free, dtype=np.float64)


class L1Relay:
    """교정 신호가 들어오는 **논리적 인터페이스** (명세 [7], [8]).

    새 피라미드 세포체 집단을 만들지 않는다. 교사가 계산한
    ``{destination_neuron_id: correction_value}`` 를 해당 지역 모듈에 전달할 뿐이다.
    """

    def __init__(self, allowed_destinations: np.ndarray) -> None:
        self._allowed = set(int(x) for x in np.asarray(allowed_destinations).tolist())
        self.n_routed = 0

    def route(self, corrections: Mapping[int, float] | tuple[np.ndarray, np.ndarray]) -> dict[int, float]:
        """교정 사상 검증 후 그대로 전달한다 (부작용: 카운터 증가).

        허용되지 않은 목적지가 있으면 ValueError.
        """
        if isinstance(corrections, tuple):
            ids, vals = corrections
            items = {int(i): float(v) for i, v in zip(np.asarray(ids).tolist(),
                                                     np.asarray(vals).tolist())}
        else:
            items = {int(k): float(v) for k, v in corrections.items()}
        bad = [k for k in items if k not in self._allowed]
        if bad:
            raise ValueError(f"L1Relay 가 허용하지 않는 목적지 뉴런: {bad[:5]}")
        self.n_routed += len(items)
        return items


def teacher_feasibility_report(
    targets: np.ndarray, sensory_bits: np.ndarray, scene_ids: Sequence[str]
) -> dict[str, Any]:
    """식별 불가능성 후보 집계.

    같은 학생 입력(감각 비트)에 서로 다른 교사 목표가 붙은 표본 쌍을 센다.
    "모든 교사 목표가 학생에게 실현 가능하다"고 가정하지 않기 위한 진단이다.

    Parameters
    ----------
    targets : (n_scenes, M) int
    sensory_bits : (n_scenes, D) int
        학생이 실제로 받는 L4 발화 패턴 (이진화 후).

    Returns
    -------
    dict : 충돌 그룹 수, 충돌 표본 수, 충돌 비트 수 등
    """
    keys: dict[bytes, list[int]] = {}
    for i in range(sensory_bits.shape[0]):
        k = np.packbits(sensory_bits[i].astype(np.uint8)).tobytes()
        keys.setdefault(k, []).append(i)
    n_groups = 0
    n_samples = 0
    n_conflicting_bits = 0
    for idxs in keys.values():
        if len(idxs) < 2:
            continue
        block = targets[idxs]
        varying = (block.max(axis=0) != block.min(axis=0))
        if varying.any():
            n_groups += 1
            n_samples += len(idxs)
            n_conflicting_bits += int(varying.sum())
    return {
        "n_distinct_sensory_patterns": len(keys),
        "n_conflicting_groups": n_groups,
        "n_samples_in_conflicting_groups": n_samples,
        "n_conflicting_target_bits": n_conflicting_bits,
        "note_ko": (
            "같은 L4 발화 패턴에 서로 다른 목표가 붙은 경우. 전처리·이진화·수용장 제한으로 "
            "생기는 식별 불가능성 후보이며, 이 표본들은 학생이 원리적으로 구분할 수 없다."
        ),
    }
