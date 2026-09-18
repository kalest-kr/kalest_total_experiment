#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cortex_all_in_one.py — 3x3 뉴런 기록 구조 시각피질 시뮬레이터 (단일 파일 판)

이 파일 **하나만** 있으면 동작한다. 원래 패키지(`cortex/` 22개 모듈 + `run.py`)와
같은 코드를 순서대로 이어 붙이고, 설정 4종을 코드로 내장했다.

    python cortex_all_in_one.py                     # 한국어 메뉴
    python cortex_all_in_one.py inspect-config --config minimal
    python cortex_all_in_one.py validate --config minimal --execute
    python cortex_all_in_one.py simulate --config v1_small --execute
    python cortex_all_in_one.py report --run-dir runs/RUN_ID

`--config` 에는 내장 설정 이름(minimal, v1_small, hierarchy_small,
megapixel_input) 또는 JSON 파일 경로를 줄 수 있다.

=== 이번 납품의 수치 실험 상태는 전부 `not_run` 이다 ===
이 파일을 만드는 동안 학습·시뮬레이션·데이터 생성·수치 검증·그래프 생성을 한 번도
실행하지 않았다. 실행은 사용자가 명령을 내릴 때만 일어난다. 자세한 내용은
같은 폴더의 IMPLEMENTATION_STATUS.md 참조.

--- 병합하면서 바꾼 것 (원본과의 차이) ---
1. 상대 import(`from .x import y`)를 모두 제거했다. 모든 이름이 한 모듈 안에 있다.
2. 이름이 겹치던 최상위 함수 2쌍을 구분되게 바꿨다:
     anatomy.build          -> build_anatomy
     areas.build            -> build_wiring
     plasticity.make        -> make_plasticity
     predictive_coding.make -> make_rao_model
   읽기 쉽도록 다음도 함께 바꿨다:
     rng.from_config -> rng_from_config      stimuli.generate -> generate_stimuli
     config.load     -> load_config          config.resolve   -> resolve_config
     config.validate -> validate_config
3. matplotlib 은 **지연 로드**(`_plt()`)로 바꿨다. 이 파일을 import 만 해도
   백엔드 초기화가 일어나지 않는다 (import 부작용 금지 규칙 유지).
4. 설정 4종을 `BUILTIN_CONFIGS` 로 내장했다. JSON 파일 없이도 실행된다.
5. 코드 해시(`code_hash`)는 디렉터리 대신 **이 파일 하나**를 해싱한다.
6. import 부작용 검사(검증 14)는 이 파일을 경로로 import 하는 방식으로 바꿨다.

그 외 계산 로직, 자료구조, 검증 기준, 단위 규약은 원본과 같다.
각 섹션의 제목은 원래 모듈 이름이며, 섹션 첫머리의 문자열이 그 모듈의 설명이다.
"""

from __future__ import annotations

# --- 표준 라이브러리 ---
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

# --- 서드파티 (matplotlib 은 _plt() 에서 지연 로드) ---
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------
# 목차 (원래 모듈 순서 = 의존성 순서)
#   01 units              단위 규약           12 dynamics          시간 루프/두 모드
#   02 ids                ID 레지스트리       13 plasticity        STDP/항상성
#   03 config             설정 스키마         14 predictive_coding Rao 참조 모델
#   04 rng                난수 스트림         15 stimuli           자극 생성
#   05 records            3x3 기록 인터페이스 16 v1_reference      고정 Gabor 대조
#   06 events             이벤트/지연 큐      17 recording         기록기/체크포인트
#   07 synapses           희소 edge table     18 runner            실험 실행기
#   08 retina             색/DoG/ON-OFF       19 analysis          조회/보고서
#   09 retinotopy         로그-극좌표         20 visualization     그림
#   10 anatomy            3D 배치/지도        21 validation        필수 검증 1~14
#   11 areas              배선 생성           22 cli               명령행 인터페이스
#   23 builtin configs    내장 설정 4종       24 menu              한국어 터미널 메뉴
# ----------------------------------------------------------------------------


# ============================================================================
# 섹션: units  —  단위 규약과 물리 상수
#   (원래 파일: cortex/units.py)
# ============================================================================

"""units.py -- 단위 규약과 물리 상수.

전도도 모드(`conductance_lif`)의 단위는 다음으로 **통일**한다.

======== ======== ====================================================
기호     단위     설명
======== ======== ====================================================
V        mV       막전위
t, tau   ms       시간, 시정수
C        pF       막 용량
g        nS       전도도 (누설, 시냅스, 구획 결합)
I        pA       전류
======== ======== ====================================================

이 조합은 서로 정합적이다::

    [nS] * [mV] = 1e-9 S * 1e-3 V = 1e-12 A = 1 pA
    [pF] * [mV] / [ms] = 1e-12 F * 1e-3 V / 1e-3 s = 1e-12 A = 1 pA

따라서 ``C dV/dt = g (E - V) + I`` 를 pF, mV, ms, nS, pA 로 그대로 쓸 수 있고
별도의 환산 계수가 필요 없다. 코드 어디에서도 SI 기본 단위로 되돌리지 않는다.

``sum_threshold`` 모드의 s_j, theta_j, P 는 **무차원**이다. 이 모드의 합산값을
막전위(mV)로 해석하지 않는다. 두 모드의 단위는 서로 호환되지 않으며 변환
계수를 임의로 도입하지 않는다.

공간 단위:

* ``cortical_xyz_mm`` : 피질 모형 좌표, mm
* ``visual_field_xy`` : 시야 좌표, degree (시야각). 픽셀→도 변환은
  영상 크기와 FOV 설정으로 계산한다 (:mod:`cortex.retinotopy`).
* 두 공간의 거리를 서로 섞어 쓰지 않는다.

발화율은 Hz(=1/s)이고 dt 는 ms 다. dt 동안의 기대 사건 수는 ``rate_hz * dt_ms / 1000``
이며, 이 변환을 거치지 않고 Hz 를 ms 격자에 직접 쓰지 않는다.
"""

#: 1초 = 1000 ms
MS_PER_S: float = 1000.0

#: 발화율(Hz)과 dt(ms)로부터 dt 동안의 기대 사건 수
def expected_events(rate_hz: float, dt_ms: float) -> float:
    """rate_hz [1/s] 와 dt_ms [ms] 로부터 dt 동안의 기대 사건 수(무차원)."""
    return float(rate_hz) * float(dt_ms) / MS_PER_S


#: 문서/manifest 에 기록할 단위 표
UNIT_TABLE: dict[str, str] = {
    "voltage": "mV",
    "time": "ms",
    "capacitance": "pF",
    "conductance": "nS",
    "current": "pA",
    "rate": "Hz",
    "cortical_position": "mm",
    "visual_field": "deg",
    "image_position": "px",
    "sum_threshold_activation": "dimensionless",
    "synaptic_weight_conductance_mode": "nS",
    "synaptic_weight_sum_mode": "dimensionless",
}

#: NMDA Mg2+ 차단 식의 기본 계수 (Jahr & Stevens 1990 형태). 모형 파라미터다.
MG_BLOCK_MG_MM: float = 1.0      # [Mg2+] in mM
MG_BLOCK_SLOPE: float = 0.062    # 1/mV
MG_BLOCK_SCALE: float = 3.57     # mM


# ============================================================================
# 섹션: ids  —  안정적인 정수 ID 와 이름 레지스트리
#   (원래 파일: cortex/ids.py)
# ============================================================================

"""ids.py -- 안정적인 정수 ID 와 이름 레지스트리.

뉴런·시냅스·영역·층·세포 유형·수용체·구획을 **정수 ID** 로 다룬다.
이웃 뉴런 객체를 다른 객체 안에 재귀적으로 넣지 않는다 (명세 2절).

ID 규약
-------
* ``neuron_id`` : 0..N-1. 전역 상태 배열의 행 인덱스와 **동일**하다.
* ``synapse_id`` : 0..K-1. edge table 의 행 인덱스와 동일하다.
* ``event_id``   : 실행(run) 안에서 단조 증가하는 정수. 체크포인트에서 이어진다.
* ``spike_id``   : 발화 1건의 고유 ID. 도착 이벤트의 ``parent_spike_id`` 가 된다.

영역/층/세포유형/수용체 이름은 :class:`Registry` 로 정수에 매핑한다. 매핑은
manifest 에 저장되어 재현 시 같은 정수를 복원한다. Python 의 실행별 hash 에
의존하지 않는다.
"""

#: 구획 인덱스. 피라미드 세포는 3개, 나머지 유형은 soma 만 사용한다.
COMPARTMENT_NAMES: tuple[str, ...] = ("soma", "basal", "apical")
COMPARTMENT_INDEX: dict[str, int] = {n: i for i, n in enumerate(COMPARTMENT_NAMES)}
N_COMPARTMENTS: int = len(COMPARTMENT_NAMES)

#: 피질 층 이름. L2/3 을 연산에서 묶더라도 조회·집계에서는 구분을 유지한다.
CORTICAL_LAYERS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "L6")

#: 비피질 구조의 "층" 자리표시 이름 (망막/LGN)
NONCORTICAL_LAYERS: tuple[str, ...] = ("L_input", "L_relay")

#: 이벤트 종류
EVENT_TYPES: tuple[str, ...] = (
    "external_drive",   # 망막/외부 자극 주입
    "synaptic_arrival", # 시냅스 도착
    "spike",            # 발화
    "weight_update",    # 가중치 변경
    "threshold_update", # 임계값 변경
)


class Registry:
    """이름 <-> 정수 ID 양방향 레지스트리 (삽입 순서로 ID 부여)."""

    def __init__(self, names: Iterable[str] = ()) -> None:
        self._names: list[str] = []
        self._index: dict[str, int] = {}
        for n in names:
            self.add(n)

    def add(self, name: str) -> int:
        """이름을 등록하고 ID 를 돌려준다. 이미 있으면 기존 ID."""
        if name in self._index:
            return self._index[name]
        idx = len(self._names)
        self._names.append(name)
        self._index[name] = idx
        return idx

    def id_of(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError as exc:
            raise KeyError(
                f"등록되지 않은 이름: {name!r} (등록된 것: {self._names})"
            ) from exc

    def name_of(self, idx: int) -> str:
        return self._names[int(idx)]

    def has(self, name: str) -> bool:
        return name in self._index

    def names(self) -> list[str]:
        return list(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def to_dict(self) -> dict[str, int]:
        return dict(self._index)

    @staticmethod
    def from_dict(mapping: dict[str, int]) -> "Registry":
        """manifest 에서 복원. 저장된 정수값을 그대로 보존한다."""
        reg = Registry()
        reg._names = [""] * (max(mapping.values()) + 1 if mapping else 0)
        for name, idx in mapping.items():
            reg._names[int(idx)] = name
            reg._index[name] = int(idx)
        return reg


@dataclass
class IdSpace:
    """한 실행에서 쓰는 모든 이름 레지스트리 묶음."""

    areas: Registry = field(default_factory=Registry)
    layers: Registry = field(default_factory=lambda: Registry(
        list(CORTICAL_LAYERS) + list(NONCORTICAL_LAYERS)))
    cell_types: Registry = field(default_factory=Registry)
    receptors: Registry = field(default_factory=Registry)
    compartments: Registry = field(default_factory=lambda: Registry(COMPARTMENT_NAMES))
    event_types: Registry = field(default_factory=lambda: Registry(EVENT_TYPES))
    plasticity_rules: Registry = field(default_factory=lambda: Registry(("none", "stdp")))

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            "areas": self.areas.to_dict(),
            "layers": self.layers.to_dict(),
            "cell_types": self.cell_types.to_dict(),
            "receptors": self.receptors.to_dict(),
            "compartments": self.compartments.to_dict(),
            "event_types": self.event_types.to_dict(),
            "plasticity_rules": self.plasticity_rules.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict[str, dict[str, int]]) -> "IdSpace":
        return IdSpace(
            areas=Registry.from_dict(d["areas"]),
            layers=Registry.from_dict(d["layers"]),
            cell_types=Registry.from_dict(d["cell_types"]),
            receptors=Registry.from_dict(d["receptors"]),
            compartments=Registry.from_dict(d["compartments"]),
            event_types=Registry.from_dict(d["event_types"]),
            plasticity_rules=Registry.from_dict(d["plasticity_rules"]),
        )


class IdCounter:
    """event_id / spike_id 처럼 실행 전체에서 단조 증가하는 카운터.

    체크포인트에 현재 값을 저장하고 재개 시 이어받아 ID 가 중복되지 않게 한다.
    """

    def __init__(self, start: int = 0) -> None:
        self._next = int(start)

    def take(self, n: int = 1) -> int:
        """n 개를 예약하고 **시작 ID** 를 돌려준다 (부작용: 카운터 증가)."""
        if n < 0:
            raise ValueError("n 은 0 이상이어야 한다")
        start = self._next
        self._next += int(n)
        return start

    @property
    def value(self) -> int:
        return self._next

    def restore(self, value: int) -> None:
        self._next = int(value)


# ============================================================================
# 섹션: config  —  설정 스키마, 기본값, 병합과 검증
#   (원래 파일: cortex/config.py)
# ============================================================================

"""config.py -- 설정 스키마, 기본값, 병합과 검증.

설계 선택 (근거)
----------------
설정은 **중첩 dict** 로 다루고, :data:`DEFAULTS` 에 모든 스칼라 기본값을 선언한다.
사용자 JSON 을 DEFAULTS 위에 깊은 병합(deep merge)하고, **DEFAULTS 에 없는 키는
오류**로 처리한다. 오타가 조용히 무시되어 "설정했다고 생각했지만 반영되지 않는"
상황을 막기 위해서다. 영역/배선처럼 개수가 가변인 부분은 항목 하나하나에
템플릿 기본값을 적용한다 (``_AREA_TEMPLATE``, ``_WIRING_TEMPLATE``).

해석된 전체 설정(resolved config)은 manifest.json 에 그대로 저장된다. 따라서
실행 기록만 보고도 어떤 값이 실제로 쓰였는지 알 수 있다.

단위는 키 이름에 붙인다 (``dt_ms``, ``C_pF``, ``gL_nS``, ``E_rev_mV``,
``sigma_deg``, ``extent_mm``). 단위 없는 수치를 새로 추가하지 말 것.
"""

class ConfigError(ValueError):
    """설정 파일이 스키마를 위반했을 때."""


# ----------------------------------------------------------------------
# 기본값
# ----------------------------------------------------------------------
#: 세포 유형 1개의 기본 파라미터. 값은 **모형 파라미터**이며 특정 실험의
#: 측정값이 아니다 (BIOLOGY_AND_ASSUMPTIONS.md 참조).
_CELL_TYPE_TEMPLATE: dict[str, Any] = {
    "dale": "excitatory",                 # excitatory | inhibitory
    "compartments": ["soma"],             # soma / basal / apical 부분집합, soma 필수
    "C_pF": {"soma": 200.0, "basal": 100.0, "apical": 100.0},
    "gL_nS": {"soma": 10.0, "basal": 5.0, "apical": 5.0},
    "EL_mV": {"soma": -70.0, "basal": -70.0, "apical": -70.0},
    "g_couple_nS": {"soma_basal": 8.0, "soma_apical": 4.0},
    "V_th_mV": -50.0,
    "V_reset_mV": -65.0,
    "t_ref_ms": 2.0,
    "target_rate_hz": 5.0,                # 항상성 임계 적응의 목표 활동률
    "sum_threshold_theta": 1.0,           # sum_threshold 모드의 초기 theta (무차원)
    "output_gain_P": 1.0,                 # 3x3 [1][2], 기본 범위 0~1 밖의 값은 경고
    "refractory_input_policy": "accumulate",  # accumulate | discard (soma 시냅스 입력 처리)
}

#: 영역 1개의 기본 구조.
_AREA_TEMPLATE: dict[str, Any] = {
    "kind": "cortex",                      # cortex | retina | thalamus
    "hierarchy_level": 0.0,                # 표시용. 배선을 강제하지 않는다.
    "surface_origin_mm": [0.0, 0.0],       # 피질 표면 좌표계의 영역 오프셋
    "surface_extent_mm": [4.0, 4.0],
    "layer_thickness_mm": {"L1": 0.10, "L2": 0.25, "L3": 0.35,
                           "L4": 0.30, "L5": 0.40, "L6": 0.40},
    "depth_origin_mm": 0.0,                # 피질 표면(z=0)에서 아래로 증가
    "neurons_per_layer": {},               # {"L4": 128, ...}. 없으면 0
    "cell_type_fractions": {},             # {"L4": {"spiny_stellate":0.8,"PV":0.2}}
    "visual_field": {
        "max_ecc_deg": 8.0,
        "rf_sigma_deg": 0.25,              # 이 영역 뉴런의 기본 수용장 크기
        "rf_sigma_growth_per_deg": 0.06,   # 편심도에 따른 수용장 확대
    },
    "notes": "",
}

#: 배선 규칙 1개의 기본값.
_WIRING_TEMPLATE: dict[str, Any] = {
    "name": "",
    "enabled": True,
    "src": {"area": "", "layer": [], "cell_type": []},
    "dst": {"area": "", "layer": [], "cell_type": []},
    "target_compartment": "basal",         # soma | basal | apical
    "receptor": "AMPA",
    "rule": "rf_knn",                      # rf_knn | local_radius | all_to_all_sampled
    "k": 12,                               # rf_knn 후보 수
    "radius_mm": 0.3,                      # local_radius 반경 (피질 mm)
    "rf_match_sigma_deg": 0.4,             # 시야 위치 대응 허용폭
    "probability": 0.5,                    # 후보 중 실제로 만들 확률
    "max_synapses_per_target": 0,          # 0 이면 제한 없음
    "weight": {"dist": "lognormal", "median": 1.0, "sigma": 0.35,
               "min": 0.0, "max": 8.0},
    "conduction_velocity_mm_per_ms": 0.3,
    "synaptic_delay_ms": 0.8,
    "use_straight_line_distance": True,    # 축삭 길이 근사 (표시 대상)
    "plasticity_rule": "none",             # none | stdp
    # Gabor 모양으로 초기 가중치를 정할지. True 면 manifest 에 기록된다.
    # 음의 필터 계수는 음의 전도도가 아니라 반대 극성(OFF) 입력으로 구현한다.
    "gabor_initialized": False,
    "note": "",
}

DEFAULTS: dict[str, Any] = {
    "meta": {
        "name": "unnamed",
        "description": "",
        "species_assumption": "primate_visual_cortex_model",
        "notes": [],
    },
    "seeds": {
        "master": 12345,
        # 고정된 이름의 독립 스트림. 조건을 추가해도 기존 조건의 초기 상태가
        # 바뀌지 않도록 이름->정수 매핑을 고정한다 (rng.py).
        "stream_offsets": {
            "data": 101, "wiring": 202, "weights": 303, "input_noise": 404,
            "learning_order": 505, "diagnostics": 606, "readout": 707,
            "split": 808, "stimulus": 909,
        },
    },
    "engine": {
        "mode": "conductance_lif",          # sum_threshold | conductance_lif
        "dt_ms": 0.5,
        "duration_ms": 200.0,
        "min_delay_steps": 1,
        "delay_rounding": "ceil",           # ceil | round
        "weight_application": "emit",       # emit | arrival (기본: 발신 시점 스냅샷)
        "sum_threshold_interval_steps": 1,  # sum_threshold 모드의 처리 구간 길이
        "reset_between_samples": {
            "voltages": True, "conductances": True, "event_queue": True,
            "traces": False, "thresholds": False, "weights": False,
            "input_log": False,
        },
        "sequence_mode": False,             # True 면 표본 간 상태를 유지 (동영상)
    },
    "receptors": {
        "AMPA":   {"tau_ms": 2.0,   "E_rev_mV": 0.0,   "kind": "excitatory", "mg_block": False},
        "NMDA":   {"tau_ms": 100.0, "E_rev_mV": 0.0,   "kind": "excitatory", "mg_block": True},
        "GABA_A": {"tau_ms": 6.0,   "E_rev_mV": -70.0, "kind": "inhibitory", "mg_block": False},
    },
    "cell_types": {},                        # 이름 -> _CELL_TYPE_TEMPLATE 병합
    "anatomy": {
        "areas": {},                         # 이름 -> _AREA_TEMPLATE 병합
        "area_gap_mm": 1.0,                  # 영역 사이 표면 좌표 간격 (배치용)
        "jitter_mm": 0.02,                   # 뉴런 위치 난수 흔들기
    },
    "wiring": {
        "rules": [],
        "dale_enforced": True,
        "allow_self_connection": False,
        "max_total_synapses": 5_000_000,     # 용량 한도. 초과 시 명시적으로 중단
    },
    "retina": {
        "image": {
            "max_side_px": 1024,
            "fov_deg": 20.0,                 # 영상 긴 변이 덮는 시야각
            "input_colorspace": "srgb",      # srgb | linear
        },
        "color": {
            "use_lms": True,
            "lms_matrix": "hunt_pointer_estevez_d65",  # 또는 "stockman_sharpe_lms"
            "opponent_channels": ["luminance", "red_green", "blue_yellow"],
        },
        "dog": {
            "center_sigma_px": 1.0,
            "surround_sigma_px": 3.0,
            "truncate": 4.0,
            "boundary_mode": "reflect",
            "normalize_each_kernel_to_unit_sum": True,
        },
        "channels": {
            "on_off_split": True,            # 6 채널 (3 대립 x ON/OFF)
            "keep_lowpass_lms": True,        # V4 색 경로용 저주파 LMS 3채널 추가
            "lowpass_sigma_px": 8.0,
        },
        "drive": {
            "mode": "rate",                  # rate | poisson
            "max_rate_hz": 60.0,
            "gain": 1.0,
            "baseline_rate_hz": 0.0,
            # rate 모드에서 발화율(Hz)을 망막 뉴런의 외부 전류(pA)로 바꾸는 계수.
            # 모형 파라미터이며 측정값이 아니다.
            "current_per_hz_pA": 2.0,
            # sum_threshold 모드에서 발화율을 무차원 기여로 바꾸는 계수.
            "sum_mode_scale": 1.0,
        },
    },
    "retinotopy": {
        "mapping": "log_polar",              # log_polar | uniform_control
        "e0_deg": 0.5,                       # rho = log(1 + ecc/e0)
        "eccentricity_unit": "deg",
        "n_radial": 12,
        "n_angular": 24,
        "fovea_patch": {
            "enabled": True,
            "radius_deg": 0.5,
            "grid": 4,                       # 4x4 Cartesian 격자
        },
        "sampling": {
            "lowpass_before_sampling": True,
            "sigma_scale": 0.5,              # sigma_px = sigma_scale * 셀 크기
            "min_sigma_px": 0.5,
            "interpolation_order": 1,
        },
        "uniform_control": {
            "match_total_samples": True,
            "grid": 0,                       # 0 이면 총 표본 수에서 자동 계산
        },
        "binocular": {
            "enabled": False,
            "interocular_shift_deg": 0.0,
        },
    },
    "v1": {
        "n_orientations": 12,                # 0~165도, 15도 간격
        "orientation_step_deg": 15.0,
        "phases_rad": [0.0, -1.5707963267948966],
        "pinwheel": {
            "enabled": True,
            "hypercolumn_mm": 0.8,
            "n_pinwheels_per_mm2": 3.0,
        },
        "ocular_dominance": {
            "enabled": True,
            "column_width_mm": 0.4,
            "strength": 0.6,                 # 0=양안 동일, 1=완전 단안
        },
        "gabor_init": {
            "enabled": False,                # 초기 배선을 Gabor 로 정하면 기록된다
            "sigma_deg": 0.25,
            "aspect": 1.6,
            "cycles_per_deg": 2.0,
        },
        "fixed_gabor_reference": {
            "enabled": False,                # 학습 아님. 대조 경로.
            "sigma_deg": 0.25,
            "aspect": 1.6,
            "cycles_per_deg": 2.0,
            "energy_eps": 1e-6,
        },
    },
    "learning": {
        "mode": "none",                      # none | stdp_homeostasis | rao_reference
        "task_learning_enabled": True,
        "weight_decay_enabled": False,
        "threshold_adaptation_enabled": False,
        "stdp": {
            "A_plus": 0.01,
            "A_minus": 0.012,
            "tau_plus_ms": 20.0,
            "tau_minus_ms": 20.0,
            "weight_min": 0.0,
            "weight_max": 8.0,
            "weight_decay_per_ms": 0.0,
            "simultaneous_policy": "both",   # both | pre_first | post_first
            "update_order": "post_then_pre", # 문서화된 갱신 순서
            "apply_to_inhibitory": False,
        },
        "homeostasis": {
            "eta_theta": 0.002,
            "window_ms": 500.0,
            "theta_min_mV": -60.0,
            "theta_max_mV": -35.0,
            "theta_min_sum": 0.05,
            "theta_max_sum": 50.0,
            "per_cell_type_target": True,
        },
        "neighbor_theta_averaging": {
            "enabled": False,                # 소거(ablation) 실험 전용 옵션
            "radius_mm": 0.15,
            "strength": 0.0,
        },
        "rao": {
            "n_levels": 2,
            "level_sizes": [32, 16],
            "sigma": 1.0,
            "sigma_td": 2.0,
            "alpha": 0.05,
            "lambda_u": 0.001,
            "settle_steps": 30,
            "r_step": 0.05,
            "u_step": 0.002,
            "freeze_U_during_settle": True,
        },
        "apical_error_coupling": {
            "enabled": False,                # Rao 오차 -> apical 전류 (가정 문서화 필요)
            "gain_pA_per_unit": 0.0,
            "split_sign": True,
        },
    },
    "readout": {
        "enabled": False,
        "source_area": "IT",
        "source_layer": ["L2", "L3"],
        "window_ms": [50.0, 200.0],
        "classifier": "ridge",               # ridge | logistic
        "l2": 1.0,
        "n_classes": 7,
        "note": "분류 readout 은 별도 모듈이며 피질 학습과 분리된다.",
    },
    "recording": {
        "mode": "full",                      # full | selected | summary
        "backend": "hdf5",                   # hdf5 | npz
        "selected_neurons": [],
        "selected_areas": [],
        "max_events": 20_000_000,
        "max_state_samples": 2_000_000,
        "state_sample_every_steps": 1,
        "compression": "gzip",
        "compression_level": 4,
        "chunk_rows": 4096,
        "flush_every_steps": 200,
        "selection_criterion": "",           # summary/selected 모드에서 필수
    },
    "checkpoint": {
        "enabled": True,
        "every_steps": 0,                    # 0 이면 표본 경계에서만
        "every_samples": 1,
        "keep_last": 3,
    },
    "experiment": {
        "protocol": "single_pass",           # single_pass | sweep | train_dev_test
        "stimuli": [],                       # stimuli.py 가 해석하는 명세 목록
        "n_samples": 1,
        "splits": {"train": 0.6, "dev": 0.2, "test": 0.2, "stratified": True},
        "conditions": [],                    # 대조군 정의
        "limits": {
            "max_runtime_minutes": 0,        # 0 이면 제한 없음
            "max_disk_mb": 4096,
            "max_ram_mb": 8192,
        },
    },
    "validation": {
        "run_structural_checks": True,
        "stop_experiment_on_failure": True,
        "dt_convergence_factors": [1.0, 0.5, 0.25],
        "finite_difference_eps": 1e-6,
        "tolerances": {
            "matrix_view": 0.0,
            "delay_steps": 0.0,
            "lif_dt_convergence_mV": 0.5,
            "logpolar_roundtrip_deg": 0.05,
            "dog_uniform_response": 1e-6,
            "rao_gradient_rel": 1e-4,
        },
    },
}


# ----------------------------------------------------------------------
# 병합과 검증
# ----------------------------------------------------------------------
#: 하위 키가 **자유 형식**인 경로. 여기서는 기본값에 없는 키도 허용한다.
#: ``*`` 는 임의의 한 단계를 뜻한다. 이 목록에 없는 경로의 낯선 키는 오류다
#: (오타가 조용히 무시되는 것을 막기 위함).
_OPEN_PATH_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("seeds", "stream_offsets"),
    ("anatomy", "areas", "*", "neurons_per_layer"),
    ("anatomy", "areas", "*", "cell_type_fractions"),
    ("anatomy", "areas", "*", "cell_type_fractions", "*"),
    ("anatomy", "areas", "*", "layer_thickness_mm"),
)


def _is_open_path(path: str) -> bool:
    if not path:
        return False
    segs = tuple(path.split("."))
    for pat in _OPEN_PATH_PATTERNS:
        if len(pat) != len(segs):
            continue
        if all(p == "*" or p == s for p, s in zip(pat, segs)):
            return True
    return False


def _deep_merge(base: dict[str, Any], override: dict[str, Any],
                path: str = "") -> dict[str, Any]:
    """base 위에 override 를 깊은 병합.

    base 에 없는 키는 ConfigError 다. 단 :data:`_OPEN_PATH_PATTERNS` 에 해당하는
    자유 형식 경로(층 이름, 세포 유형 이름, RNG 스트림 이름 등)에서는 허용한다.
    """
    out = copy.deepcopy(base)
    open_here = _is_open_path(path)
    for key, val in override.items():
        here = f"{path}.{key}" if path else key
        if key not in out:
            if open_here:
                out[key] = copy.deepcopy(val)
                continue
            raise ConfigError(
                f"알 수 없는 설정 키: '{here}'. 오타이거나 지원하지 않는 항목이다. "
                f"사용 가능한 키: {sorted(out.keys())}"
            )
        if isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val, here)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _merge_free_dict(template: dict[str, Any], override: dict[str, Any],
                     path: str) -> dict[str, Any]:
    """항목 수가 가변인 dict (영역/세포유형) 에 템플릿을 적용한다."""
    return _deep_merge(template, override, path)


def resolve_config(user_config: dict[str, Any]) -> dict[str, Any]:
    """사용자 설정을 기본값 위에 병합하고 전체를 검증한다 (부작용 없음).

    Returns
    -------
    dict : 완전히 해석된 설정. manifest.json 에 그대로 저장된다.
    """
    free_sections = {
        "cell_types": user_config.get("cell_types", {}),
        "areas": user_config.get("anatomy", {}).get("areas", {}),
    }
    stripped = copy.deepcopy(user_config)
    stripped.pop("cell_types", None)
    if "anatomy" in stripped:
        stripped["anatomy"] = {k: v for k, v in stripped["anatomy"].items() if k != "areas"}
    wiring_rules = None
    if "wiring" in stripped and "rules" in stripped["wiring"]:
        wiring_rules = stripped["wiring"].pop("rules")

    cfg = _deep_merge(DEFAULTS, stripped)

    cfg["cell_types"] = {
        name: _merge_free_dict(_CELL_TYPE_TEMPLATE, spec, f"cell_types.{name}")
        for name, spec in free_sections["cell_types"].items()
    }
    cfg["anatomy"]["areas"] = {
        name: _merge_free_dict(_AREA_TEMPLATE, spec, f"anatomy.areas.{name}")
        for name, spec in free_sections["areas"].items()
    }
    if wiring_rules is not None:
        cfg["wiring"]["rules"] = [
            _merge_free_dict(_WIRING_TEMPLATE, r, f"wiring.rules[{i}]")
            for i, r in enumerate(wiring_rules)
        ]

    validate_config(cfg)
    return cfg


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate_config(cfg: dict[str, Any]) -> None:
    """해석된 설정의 구조·값 범위를 검사한다. 위반 시 ConfigError."""
    eng = cfg["engine"]
    _require(eng["mode"] in ("sum_threshold", "conductance_lif"),
             f"engine.mode 는 sum_threshold 또는 conductance_lif 여야 한다: {eng['mode']!r}")
    _require(eng["dt_ms"] > 0, "engine.dt_ms 는 양수여야 한다")
    _require(eng["duration_ms"] > 0, "engine.duration_ms 는 양수여야 한다")
    _require(eng["min_delay_steps"] >= 1,
             "engine.min_delay_steps 는 1 이상이어야 한다 (0 지연 재귀 방지)")
    _require(eng["delay_rounding"] in ("ceil", "round"),
             "engine.delay_rounding 은 ceil 또는 round")
    _require(eng["weight_application"] in ("emit", "arrival"),
             "engine.weight_application 은 emit 또는 arrival")
    _require(eng["sum_threshold_interval_steps"] >= 1,
             "engine.sum_threshold_interval_steps 는 1 이상")

    for rname, r in cfg["receptors"].items():
        _require(r["tau_ms"] > 0, f"receptors.{rname}.tau_ms 는 양수여야 한다")
        _require(r["kind"] in ("excitatory", "inhibitory"),
                 f"receptors.{rname}.kind 는 excitatory/inhibitory")

    _require(len(cfg["cell_types"]) > 0, "cell_types 가 비어 있다")
    for cname, c in cfg["cell_types"].items():
        _require(c["dale"] in ("excitatory", "inhibitory"),
                 f"cell_types.{cname}.dale 는 excitatory/inhibitory")
        _require("soma" in c["compartments"],
                 f"cell_types.{cname}.compartments 에 soma 가 없다")
        for comp in c["compartments"]:
            _require(comp in ("soma", "basal", "apical"),
                     f"cell_types.{cname}: 알 수 없는 구획 {comp!r}")
            _require(c["C_pF"][comp] > 0, f"cell_types.{cname}.C_pF[{comp}] > 0 이어야 한다")
            _require(c["gL_nS"][comp] > 0, f"cell_types.{cname}.gL_nS[{comp}] > 0")
        _require(c["t_ref_ms"] >= 0, f"cell_types.{cname}.t_ref_ms >= 0")
        _require(c["V_th_mV"] > c["V_reset_mV"],
                 f"cell_types.{cname}: V_th_mV 가 V_reset_mV 보다 커야 한다")
        _require(c["refractory_input_policy"] in ("accumulate", "discard"),
                 f"cell_types.{cname}.refractory_input_policy 는 accumulate/discard")

    areas = cfg["anatomy"]["areas"]
    _require(len(areas) > 0, "anatomy.areas 가 비어 있다")
    for aname, a in areas.items():
        _require(a["kind"] in ("cortex", "retina", "thalamus"),
                 f"anatomy.areas.{aname}.kind 는 cortex/retina/thalamus")
        for lname, n in a["neurons_per_layer"].items():
            _require(int(n) >= 0, f"{aname}.{lname} 뉴런 수는 0 이상")
            fr = a["cell_type_fractions"].get(lname)
            if int(n) > 0:
                _require(fr is not None and len(fr) > 0,
                         f"anatomy.areas.{aname}.cell_type_fractions.{lname} 가 필요하다")
                s = float(sum(fr.values()))
                _require(abs(s - 1.0) < 1e-6,
                         f"{aname}.{lname} 세포 유형 비율 합이 1 이 아니다: {s}")
                for ct in fr:
                    _require(ct in cfg["cell_types"],
                             f"{aname}.{lname} 의 세포 유형 {ct!r} 이 cell_types 에 없다")
        if a["kind"] == "cortex":
            total = sum(a["layer_thickness_mm"].get(l, 0.0) for l in CORTICAL_LAYERS)
            _require(total > 0, f"anatomy.areas.{aname}: 층 두께 합이 0 이다")

    for i, r in enumerate(cfg["wiring"]["rules"]):
        tag = f"wiring.rules[{i}] ({r['name'] or 'unnamed'})"
        _require(r["src"]["area"] in areas, f"{tag}: 알 수 없는 src.area {r['src']['area']!r}")
        _require(r["dst"]["area"] in areas, f"{tag}: 알 수 없는 dst.area {r['dst']['area']!r}")
        _require(r["receptor"] in cfg["receptors"], f"{tag}: 알 수 없는 receptor")
        _require(r["target_compartment"] in ("soma", "basal", "apical"),
                 f"{tag}: 알 수 없는 target_compartment")
        _require(r["rule"] in ("rf_knn", "local_radius", "all_to_all_sampled"),
                 f"{tag}: 알 수 없는 rule {r['rule']!r}")
        _require(0.0 <= r["probability"] <= 1.0, f"{tag}: probability 는 [0,1]")
        _require(r["conduction_velocity_mm_per_ms"] > 0, f"{tag}: 전도속도는 양수")
        _require(r["synaptic_delay_ms"] >= 0, f"{tag}: 시냅스 지연은 0 이상")
        _require(r["plasticity_rule"] in ("none", "stdp"), f"{tag}: 알 수 없는 가소성 규칙")
        _require(isinstance(r["gabor_initialized"], bool),
                 f"{tag}: gabor_initialized 는 true/false")
        w = r["weight"]
        _require(w["dist"] in ("lognormal", "normal", "constant"), f"{tag}: 알 수 없는 weight.dist")
        _require(w["min"] >= 0.0, f"{tag}: weight.min 은 0 이상 (Dale 부호는 세포 유형이 결정)")
        _require(w["max"] > w["min"], f"{tag}: weight.max > weight.min")

    ret = cfg["retina"]
    _require(ret["image"]["fov_deg"] > 0, "retina.image.fov_deg 는 양수")
    _require(ret["image"]["input_colorspace"] in ("srgb", "linear"),
             "retina.image.input_colorspace 는 srgb 또는 linear")
    _require(ret["dog"]["surround_sigma_px"] > ret["dog"]["center_sigma_px"] > 0,
             "retina.dog: 0 < center_sigma_px < surround_sigma_px 여야 한다")
    _require(ret["drive"]["mode"] in ("rate", "poisson"),
             "retina.drive.mode 는 rate 또는 poisson")
    _require(ret["drive"]["max_rate_hz"] > 0, "retina.drive.max_rate_hz 는 양수")
    _require(ret["drive"]["current_per_hz_pA"] >= 0,
             "retina.drive.current_per_hz_pA 는 0 이상")
    _require(ret["color"]["lms_matrix"] in ("hunt_pointer_estevez_d65", "stockman_sharpe_lms"),
             "retina.color.lms_matrix 값이 지원 목록에 없다")

    rt = cfg["retinotopy"]
    _require(rt["mapping"] in ("log_polar", "uniform_control"),
             "retinotopy.mapping 은 log_polar 또는 uniform_control")
    _require(rt["e0_deg"] > 0, "retinotopy.e0_deg 는 양수 (log(1+ecc/e0) 의 특이점 회피)")
    _require(rt["n_radial"] >= 1 and rt["n_angular"] >= 1,
             "retinotopy.n_radial/n_angular 는 1 이상")
    _require(rt["eccentricity_unit"] in ("deg", "px"),
             "retinotopy.eccentricity_unit 은 deg 또는 px")
    _require(rt["sampling"]["interpolation_order"] in (0, 1, 3),
             "retinotopy.sampling.interpolation_order 는 0/1/3")

    v1 = cfg["v1"]
    _require(v1["n_orientations"] >= 1, "v1.n_orientations 는 1 이상")
    _require(len(v1["phases_rad"]) >= 1, "v1.phases_rad 가 비어 있다")

    lr = cfg["learning"]
    _require(lr["mode"] in ("none", "stdp_homeostasis", "rao_reference"),
             f"learning.mode 값이 잘못되었다: {lr['mode']!r}")
    _require(lr["stdp"]["tau_plus_ms"] > 0 and lr["stdp"]["tau_minus_ms"] > 0,
             "learning.stdp 시정수는 양수")
    _require(lr["stdp"]["weight_max"] > lr["stdp"]["weight_min"] >= 0.0,
             "learning.stdp 가중치 범위가 잘못되었다 (하한 0 이상, 상한 > 하한)")
    _require(lr["stdp"]["simultaneous_policy"] in ("both", "pre_first", "post_first"),
             "learning.stdp.simultaneous_policy 값이 잘못되었다")
    rao = lr["rao"]
    _require(rao["sigma"] > 0 and rao["sigma_td"] > 0,
             "learning.rao.sigma / sigma_td 는 양수여야 한다 (분산 계수)")
    _require(rao["n_levels"] == len(rao["level_sizes"]),
             "learning.rao.n_levels 와 level_sizes 길이가 다르다")
    _require(rao["alpha"] >= 0 and rao["lambda_u"] >= 0,
             "learning.rao.alpha / lambda_u 는 0 이상")

    rec = cfg["recording"]
    _require(rec["mode"] in ("full", "selected", "summary"),
             "recording.mode 는 full/selected/summary")
    _require(rec["backend"] in ("hdf5", "npz"), "recording.backend 는 hdf5 또는 npz")
    if rec["mode"] != "full":
        _require(bool(rec["selection_criterion"]),
                 "recording.mode 가 full 이 아니면 selection_criterion 을 반드시 적어야 한다 "
                 "(무엇을 기록하지 않는지 manifest 에 남긴다)")
    _require(rec["state_sample_every_steps"] >= 1,
             "recording.state_sample_every_steps 는 1 이상")

    _require(cfg["wiring"]["max_total_synapses"] > 0, "wiring.max_total_synapses 는 양수")


def load_config(path: str | Path) -> dict[str, Any]:
    """JSON 설정 파일을 읽어 해석한다. 파일이 없으면 친절한 한국어 오류."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(
            f"설정 파일을 찾을 수 없다: {p}\n"
            f"  - 경로를 확인하거나 configs/ 폴더의 예시 파일을 사용하라.\n"
            f"  - 예: python -m cortex.cli inspect-config --config configs/minimal.json"
        )
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"설정 파일 JSON 구문 오류 ({p}): {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"설정 파일의 최상위는 객체여야 한다: {p}")
    cfg = resolve_config(raw)
    cfg["meta"]["source_path"] = str(p.resolve())
    return cfg


@dataclass(frozen=True)
class SizeEstimate:
    """실행 전에 계산하는 규모 추정. **런타임(초)은 추정하지 않는다.**"""

    n_neurons: int
    n_synapses_estimated: int
    n_steps: int
    events_estimated: int
    state_rows_estimated: int
    ram_mb_estimated: float
    disk_mb_estimated: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_neurons": self.n_neurons,
            "n_synapses_estimated": self.n_synapses_estimated,
            "n_steps": self.n_steps,
            "events_estimated": self.events_estimated,
            "state_rows_estimated": self.state_rows_estimated,
            "ram_mb_estimated": round(self.ram_mb_estimated, 2),
            "disk_mb_estimated": round(self.disk_mb_estimated, 2),
            "runtime_seconds_estimated": None,
            "runtime_note_ko": "실제 런타임은 측정 전에 확정하지 않는다.",
            "notes": list(self.notes),
        }


def derived_retina_neuron_count(cfg: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """망막 영역의 뉴런 수는 (채널 수 x 시야 샘플 수) 로 **유도**된다.

    설정의 ``neurons_per_layer`` 값 대신 이 값을 쓰며, 실제로 쓴 값과 유도 근거를
    manifest 에 남긴다. 순환 import 를 피하려고 지연 import 한다.
    """

    grid = build_grid(cfg)
    ch = cfg["retina"]["channels"]
    n_ch = (len(OPPONENT_NAMES) * (2 if ch["on_off_split"] else 1)
            + (3 if ch["keep_lowpass_lms"] else 0))
    n = int(n_ch * grid.n_samples)
    return n, {"n_channels": n_ch, "n_samples": int(grid.n_samples),
               "grid_summary": grid.summary()}


def count_neurons(cfg: dict[str, Any]) -> int:
    """망막 영역은 유도값, 나머지는 설정값으로 총 뉴런 수를 센다."""
    total = 0
    for a in cfg["anatomy"]["areas"].values():
        if a["kind"] == "retina":
            total += derived_retina_neuron_count(cfg)[0]
        else:
            total += int(sum(int(n) for n in a["neurons_per_layer"].values()))
    return int(total)


def estimate_sizes(cfg: dict[str, Any]) -> SizeEstimate:
    """뉴런/시냅스/이벤트/용량의 **상한 성격 추정**. 실행 전에 계산한다."""
    n_neurons = count_neurons(cfg)
    n_steps = int(round(cfg["engine"]["duration_ms"] / cfg["engine"]["dt_ms"]))
    n_samples = max(1, int(cfg["experiment"]["n_samples"]))

    areas = cfg["anatomy"]["areas"]
    n_syn = 0
    notes: list[str] = []
    for r in cfg["wiring"]["rules"]:
        if not r["enabled"]:
            continue
        dst = areas[r["dst"]["area"]]
        if dst["kind"] == "retina":
            dst_n = derived_retina_neuron_count(cfg)[0]
        else:
            dst_n = sum(int(n) for lname, n in dst["neurons_per_layer"].items()
                        if (not r["dst"]["layer"]) or lname in r["dst"]["layer"])
        per_target = r["k"] if r["rule"] == "rf_knn" else r["k"]
        if r["max_synapses_per_target"] > 0:
            per_target = min(per_target, r["max_synapses_per_target"])
        n_syn += int(dst_n * per_target * r["probability"])
    if n_syn > cfg["wiring"]["max_total_synapses"]:
        notes.append(
            f"추정 시냅스 수 {n_syn} 가 wiring.max_total_synapses "
            f"({cfg['wiring']['max_total_synapses']}) 를 넘는다. 실행은 한도에서 중단된다."
        )

    # 이벤트 수 추정: 평균 활동률 가정 (모형 가정이며 측정값이 아니다)
    assumed_rate_hz = 5.0
    spikes = n_neurons * assumed_rate_hz * cfg["engine"]["duration_ms"] / 1000.0 * n_samples
    fanout = (n_syn / max(1, n_neurons))
    events = int(spikes * max(1.0, fanout))
    notes.append(f"이벤트 수 추정은 평균 {assumed_rate_hz} Hz 가정에서 나온 값이다 (모형 가정).")

    bytes_per_event = 88      # DATA_SCHEMA.md 의 이벤트 레코드 크기
    bytes_per_state_row = 64
    rec = cfg["recording"]
    state_rows = 0
    if rec["mode"] != "summary":
        n_rec = n_neurons if rec["mode"] == "full" else max(1, len(rec["selected_neurons"]))
        state_rows = int(n_rec * (n_steps / rec["state_sample_every_steps"]) * n_samples)

    disk_mb = (events * bytes_per_event + state_rows * bytes_per_state_row) / 1e6
    ram_mb = (n_neurons * 3 * 8 * 6 + n_syn * 8 * 10) / 1e6 + 64.0

    return SizeEstimate(
        n_neurons=n_neurons,
        n_synapses_estimated=n_syn,
        n_steps=n_steps,
        events_estimated=events,
        state_rows_estimated=state_rows,
        ram_mb_estimated=ram_mb,
        disk_mb_estimated=disk_mb,
        notes=notes,
    )


def config_hash(cfg: dict[str, Any]) -> str:
    """해석된 설정의 sha256 (키 정렬, UTF-8)."""
    import hashlib
    payload = json.dumps(cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def iter_area_layers(cfg: dict[str, Any]) -> Iterable[tuple[str, str, int]]:
    """(area, layer, n_neurons) 를 결정적 순서로 열거한다."""
    for aname in sorted(cfg["anatomy"]["areas"]):
        a = cfg["anatomy"]["areas"][aname]
        for lname in sorted(a["neurons_per_layer"]):
            yield aname, lname, int(a["neurons_per_layer"][lname])


# ============================================================================
# 섹션: rng  —  이름이 고정된 독립 난수 스트림
#   (원래 파일: cortex/rng.py)
# ============================================================================

"""rng.py -- 이름이 고정된 독립 난수 스트림.

명세 10절: "각 실험의 데이터 생성·초기 배선·가중치·입력 잡음·학습 순서·진단
측정은 이름이 고정된 독립 RNG 스트림을 사용하라. 새 조건 하나를 추가했다고
기존 조건의 초기 상태가 바뀌면 안 된다."

구현: ``np.random.SeedSequence([master_seed, offset, sub_index])`` 로 만든다.
``offset`` 은 설정의 ``seeds.stream_offsets`` 에 **고정 정수**로 선언되어 있으므로
Python 의 실행별 ``hash()`` 에 의존하지 않는다. 스트림을 새로 추가해도 기존
스트림의 난수열은 바뀌지 않는다.

각 스트림의 ``bit_generator.state`` 는 체크포인트에 저장/복원되며, 진단 측정은
전용 ``diagnostics`` 스트림만 사용해 모델 스트림의 상태를 건드리지 않는다.
"""

class RngStreams:
    """이름 -> Generator 사전. 이름은 설정에서 고정 정수로 매핑된다."""

    def __init__(self, master_seed: int, stream_offsets: dict[str, int]) -> None:
        self.master_seed = int(master_seed)
        self.stream_offsets = {str(k): int(v) for k, v in stream_offsets.items()}
        self._gens: dict[str, np.random.Generator] = {}

    def get(self, name: str, sub_index: int = 0) -> np.random.Generator:
        """스트림을 얻는다. 같은 (name, sub_index) 는 같은 Generator 객체다.

        ``sub_index`` 는 같은 역할의 스트림을 표본/조건별로 분리할 때 쓴다
        (예: 자극 표본 i 의 입력 잡음).
        """
        if name not in self.stream_offsets:
            raise KeyError(
                f"등록되지 않은 RNG 스트림: {name!r}. "
                f"설정 seeds.stream_offsets 에 고정 정수로 추가해야 한다. "
                f"등록된 스트림: {sorted(self.stream_offsets)}"
            )
        key = f"{name}#{int(sub_index)}"
        gen = self._gens.get(key)
        if gen is None:
            ss = np.random.SeedSequence(
                [self.master_seed, self.stream_offsets[name], int(sub_index)]
            )
            gen = np.random.default_rng(ss)
            self._gens[key] = gen
        return gen

    def fresh(self, name: str, sub_index: int = 0) -> np.random.Generator:
        """캐시를 쓰지 않고 새로 만든 Generator (읽기 전용 진단용)."""
        if name not in self.stream_offsets:
            raise KeyError(f"등록되지 않은 RNG 스트림: {name!r}")
        ss = np.random.SeedSequence(
            [self.master_seed, self.stream_offsets[name], int(sub_index)]
        )
        return np.random.default_rng(ss)

    # --- 체크포인트 ---------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """모든 활성 스트림의 bit generator 상태 (JSON 직렬화 가능)."""
        return {
            "master_seed": self.master_seed,
            "stream_offsets": dict(self.stream_offsets),
            "states": {k: _jsonable_state(g.bit_generator.state)
                       for k, g in self._gens.items()},
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """저장된 상태 복원. 저장 시점 이후의 난수열을 그대로 이어간다."""
        self.master_seed = int(d["master_seed"])
        self.stream_offsets = {str(k): int(v) for k, v in d["stream_offsets"].items()}
        self._gens = {}
        for key, st in d["states"].items():
            name, _, sub = key.partition("#")
            gen = self.fresh(name, int(sub or 0))
            gen.bit_generator.state = _restore_state(st)
            self._gens[key] = gen

    def active_streams(self) -> list[str]:
        return sorted(self._gens)


def _jsonable_state(state: dict[str, Any]) -> dict[str, Any]:
    """numpy 정수를 파이썬 int 로 바꿔 JSON 저장 가능하게 만든다."""
    out: dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, dict):
            out[k] = _jsonable_state(v)
        elif isinstance(v, np.ndarray):
            out[k] = {"__ndarray__": v.tolist(), "dtype": str(v.dtype)}
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _restore_state(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, dict) and "__ndarray__" in v:
            out[k] = np.asarray(v["__ndarray__"], dtype=np.dtype(v["dtype"]))
        elif isinstance(v, dict):
            out[k] = _restore_state(v)
        else:
            out[k] = v
    return out


def rng_from_config(cfg: dict[str, Any]) -> RngStreams:
    return RngStreams(cfg["seeds"]["master"], cfg["seeds"]["stream_offsets"])


# ============================================================================
# 섹션: records  —  3x3 뉴런 기록 인터페이스와 타입 배열
#   (원래 파일: cortex/records.py)
# ============================================================================

"""records.py -- 3x3 뉴런 기록 인터페이스와 타입이 정해진 상태 배열.

명세 2절의 배치를 **그대로** 구현한다 (행/열은 사용자 설명 기준 1부터,
Python 인덱스는 0부터).

===========  ==============  =================================================
사용자 표기  Python 인덱스   의미
===========  ==============  =================================================
1행 1열      ``[0][0]``      x 좌표, mm
1행 2열      ``[0][1]``      y 좌표, mm
1행 3열      ``[0][2]``      z 좌표, mm
2행 1열      ``[1][0]``      **출력 연결 목록** (:class:`OutgoingConnectionsView`)
2행 2열      ``[1][1]``      현재 발화 임계값 θ
2행 3열      ``[1][2]``      기저 출력 이득 P
3행 1열      ``[2][0]``      입력 이벤트 로그 (:class:`InputLogView`)
3행 2열      ``[2][1]``      동적 상태 참조 (:class:`DynamicStateView`)
3행 3열      ``[2][2]``      메타데이터 참조 (:class:`MetadataView`)
===========  ==============  =================================================

반드시 지킨 제약
----------------
* ``[1][0]`` 에 **입력 총합을 저장하지 않는다.** 그 칸은 출력 연결 목록이다.
  합산값은 :mod:`cortex.dynamics` 의 지역 변수이며, 모니터링이 필요하면
  진단 기록에 단위와 함께 남기되 진실의 원본으로 중복 관리하지 않는다.
* ``[2][0]`` 의 입력 로그는 발화 후 삭제하지 않는다. 현재 연산에 필요한
  **작업 버퍼**(전도도/활성화 누적)는 :class:`NeuronArrays` 의 별도 배열이며
  장기 로그와 분리되어 있다.
* 이 구조는 서로 다른 타입을 담는 **기록 인터페이스**다. ``float32[3,3]``
  수치 행렬이 아니다. :meth:`NeuronRecord.as_matrix` 는 object 배열을 준다.
* 3x3 뷰는 :class:`NeuronArrays` 의 **같은 메모리를 참조**한다. 미러 복사본을
  만들지 않으므로 값 불일치가 생기지 않는다 (쓰기도 배열에 반영된다).
* 이웃 뉴런 객체를 재귀적으로 담지 않는다. 안정적인 정수 ID 만 쓴다.
* ``P`` 는 **출력 이득**이다. 방출 확률이 필요하면 ``release_probability`` 로
  따로 이름 붙이고 RNG 와 역할을 분리한다 (이번 구현에서는 미사용,
  IMPLEMENTATION_STATUS.md 참조).
"""

class NeuronArrays:
    """모든 뉴런의 상태를 담는 타입이 정해진 배열 묶음 (Structure-of-Arrays).

    행 인덱스 == ``neuron_id`` 다. 배열은 한 번만 할당되고 이후 shape 가
    바뀌지 않는다 (뷰가 무효화되지 않게 하기 위함).

    Parameters
    ----------
    n : int
        뉴런 수.
    n_receptors : int
        수용체 종류 수. 전도도 배열이 ``(n, 3, n_receptors)`` 가 된다.
    """

    def __init__(self, n: int, n_receptors: int) -> None:
        n = int(n)
        r = int(n_receptors)
        self.n = n
        self.n_receptors = r

        # --- 구조 (고정) ------------------------------------------------
        self.position_mm = np.zeros((n, 3), dtype=np.float64)     # [0][0..2]
        self.threshold = np.zeros(n, dtype=np.float64)            # [1][1]
        self.output_gain_P = np.ones(n, dtype=np.float64)         # [1][2]
        self.area_id = np.full(n, -1, dtype=np.int32)
        self.layer_id = np.full(n, -1, dtype=np.int32)
        self.cell_type_id = np.full(n, -1, dtype=np.int32)
        self.dale_sign = np.zeros(n, dtype=np.int8)               # +1 흥분 / -1 억제
        self.has_compartment = np.zeros((n, N_COMPARTMENTS), dtype=bool)
        self.has_compartment[:, 0] = True                          # soma 는 항상 존재

        # --- 세포 유형 파라미터 (뉴런별로 펼쳐 둔다) ---------------------
        self.C_pF = np.ones((n, N_COMPARTMENTS), dtype=np.float64)
        self.gL_nS = np.ones((n, N_COMPARTMENTS), dtype=np.float64)
        self.EL_mV = np.full((n, N_COMPARTMENTS), -70.0, dtype=np.float64)
        self.g_couple_nS = np.zeros((n, N_COMPARTMENTS), dtype=np.float64)
        # g_couple_nS[:, 1] = soma-basal, [:, 2] = soma-apical, [:, 0] 미사용
        self.V_reset_mV = np.full(n, -65.0, dtype=np.float64)
        self.t_ref_ms = np.zeros(n, dtype=np.float64)
        self.target_rate_hz = np.zeros(n, dtype=np.float64)
        self.refractory_discards_input = np.zeros(n, dtype=bool)

        # --- 동적 상태 ([2][1] 이 참조) ----------------------------------
        self.V_mV = np.full((n, N_COMPARTMENTS), -70.0, dtype=np.float64)
        self.g_nS = np.zeros((n, N_COMPARTMENTS, r), dtype=np.float64)
        self.refractory_until_ms = np.full(n, -np.inf, dtype=np.float64)
        self.last_spike_ms = np.full(n, -np.inf, dtype=np.float64)
        self.spike_count = np.zeros(n, dtype=np.int64)
        self.rate_estimate_hz = np.zeros(n, dtype=np.float64)
        self.trace_pre = np.zeros(n, dtype=np.float64)    # STDP 전 흔적
        self.trace_post = np.zeros(n, dtype=np.float64)   # STDP 후 흔적
        #: 사용자/실험이 직접 주는 지속 외부 전류 [pA]. 엔진은 이 배열을 덮어쓰지
        #: 않는다. 망막 구동 전류는 엔진 내부에서 별도로 더해진다.
        self.Iext_pA = np.zeros((n, N_COMPARTMENTS), dtype=np.float64)

        # --- sum_threshold 모드의 작업 버퍼 -----------------------------
        # 장기 로그와 분리된 "현재 처리 구간" 누적값이다. 매 구간 시작에 0 으로
        # 초기화되며, 전체 과거 로그의 합을 여기에 넣지 않는다.
        self.interval_activation = np.zeros(n, dtype=np.float64)

        # --- 발화 판정 직전 상태 스냅샷 (진단용, 판정 전에 채워진다) ------
        self.last_decision_value = np.zeros(n, dtype=np.float64)
        self.last_decision_threshold = np.zeros(n, dtype=np.float64)
        self.fired = np.zeros(n, dtype=np.int8)

        # --- 로그 처리 위치 --------------------------------------------
        # 같은 이벤트를 같은 상태에 두 번 주입하지 않기 위한 위치 표시.
        self.last_consumed_event = np.full(n, -1, dtype=np.int64)

        # --- 시야/특징 메타데이터 ([2][2] 가 참조) -----------------------
        self.visual_field_xy_deg = np.full((n, 2), np.nan, dtype=np.float64)
        self.rf_sigma_deg = np.full(n, np.nan, dtype=np.float64)
        self.pref_orientation_rad = np.full(n, np.nan, dtype=np.float64)
        self.pref_phase_rad = np.full(n, np.nan, dtype=np.float64)
        self.ocular_dominance = np.zeros(n, dtype=np.float64)   # -1 왼눈 .. +1 오른눈
        self.eye_id = np.full(n, -1, dtype=np.int8)             # -1 해당 없음
        self.on_off = np.zeros(n, dtype=np.int8)                # +1 ON, -1 OFF, 0 해당없음
        self.channel_id = np.full(n, -1, dtype=np.int32)        # 망막 채널 인덱스
        self.hypercolumn_uv = np.full((n, 2), np.nan, dtype=np.float64)
        #: 영역 표면 좌표 (u,v) [mm]. 영역별 오프셋을 뺀 국소 좌표다.
        self.surface_uv_mm = np.full((n, 2), np.nan, dtype=np.float64)
        #: 피라미드 첨단수상돌기 다발이 도달하는 깊이 [mm] (L1 접점 모형).
        #: 세포체가 없는 층이라도 apical 접점의 공간 위치를 갖게 한다.
        self.apical_depth_mm = np.full(n, np.nan, dtype=np.float64)

        # --- 출력 연결 인덱스 ([1][0] 이 참조) ---------------------------
        # CSR 형태: out_syn[out_ptr[i]:out_ptr[i+1]] 가 뉴런 i 의 synapse_id 들.
        self.out_ptr = np.zeros(n + 1, dtype=np.int64)
        self.out_syn = np.zeros(0, dtype=np.int64)
        self._outgoing_built = False

    # ------------------------------------------------------------------
    def set_outgoing_index(self, out_ptr: np.ndarray, out_syn: np.ndarray) -> None:
        """시냅스 테이블에서 만든 CSR 출력 인덱스를 연결한다."""
        out_ptr = np.asarray(out_ptr, dtype=np.int64)
        if out_ptr.shape != (self.n + 1,):
            raise ValueError(f"out_ptr shape 는 ({self.n + 1},) 여야 한다: {out_ptr.shape}")
        self.out_ptr = out_ptr
        self.out_syn = np.asarray(out_syn, dtype=np.int64)
        self._outgoing_built = True

    @property
    def outgoing_built(self) -> bool:
        return self._outgoing_built

    def nbytes(self) -> int:
        total = 0
        for v in vars(self).values():
            if isinstance(v, np.ndarray):
                total += int(v.nbytes)
        return total


# ----------------------------------------------------------------------
# 3x3 칸을 채우는 참조 뷰들
# ----------------------------------------------------------------------
class OutgoingConnectionsView:
    """3x3 ``[1][0]``: 이 뉴런의 **출력 연결 목록**.

    시냅스 객체를 복사해 담지 않고 ``synapse_id`` 배열을 참조한다.
    한 뉴런은 여러 출력 연결을 가질 수 있다.
    """

    __slots__ = ("_pop", "_i")

    def __init__(self, population: "NeuronPopulation", neuron_id: int) -> None:
        self._pop = population
        self._i = int(neuron_id)

    @property
    def synapse_ids(self) -> np.ndarray:
        """shape (m,) int64. 배열의 사본이 아니라 뷰(slice)다."""
        a = self._pop.arrays
        if not a.outgoing_built:
            raise RuntimeError(
                "출력 연결 인덱스가 아직 만들어지지 않았다. "
                "SynapseTable.build_indices() 후 NeuronArrays.set_outgoing_index() 를 호출하라."
            )
        lo, hi = int(a.out_ptr[self._i]), int(a.out_ptr[self._i + 1])
        return a.out_syn[lo:hi]

    def __len__(self) -> int:
        return int(self.synapse_ids.size)

    def rows(self) -> list[dict[str, Any]]:
        """연결별 전체 필드를 dict 목록으로 (조회용, 부작용 없음)."""
        syn = self._pop.synapses
        if syn is None:
            raise RuntimeError("SynapseTable 이 population 에 연결되어 있지 않다")
        return [syn.row(int(s)) for s in self.synapse_ids]

    def targets(self) -> np.ndarray:
        syn = self._pop.synapses
        if syn is None:
            raise RuntimeError("SynapseTable 이 population 에 연결되어 있지 않다")
        return syn.dst_id[self.synapse_ids]

    def __repr__(self) -> str:
        try:
            n = len(self)
        except RuntimeError:
            n = -1
        return f"<OutgoingConnectionsView neuron={self._i} n_synapses={n}>"


class InputLogView:
    """3x3 ``[2][0]``: 이 뉴런의 **입력 이벤트 로그 조회 뷰**.

    모든 뉴런이 무한 길이 Python 리스트를 들고 있지 않도록, 실제 저장은
    :class:`cortex.events.EventLog` 의 연속 배열(또는 디스크 청크)에 있고
    여기서는 인덱스로 조회만 한다. **발화해도 지워지지 않는다.**
    """

    __slots__ = ("_pop", "_i")

    def __init__(self, population: "NeuronPopulation", neuron_id: int) -> None:
        self._pop = population
        self._i = int(neuron_id)

    def _log(self) -> "EventLog":
        log = self._pop.event_log
        if log is None:
            raise RuntimeError(
                "EventLog 가 population 에 연결되어 있지 않다. "
                "NeuronPopulation.attach(event_log=...) 를 먼저 호출하라."
            )
        return log

    def count(self) -> int:
        return self._log().count_for_neuron(self._i)

    def rows(self, t_from_ms: float | None = None, t_to_ms: float | None = None,
             limit: int | None = None) -> dict[str, np.ndarray]:
        """시간 구간의 도착 이벤트를 열 배열 dict 로 돌려준다 (부작용 없음)."""
        return self._log().query_neuron(self._i, t_from_ms, t_to_ms, limit)

    @property
    def last_consumed_event(self) -> int:
        """이 뉴런의 상태에 이미 반영된 마지막 event_id (-1 이면 없음)."""
        return int(self._pop.arrays.last_consumed_event[self._i])

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        try:
            c = self.count()
        except RuntimeError:
            c = -1
        return (f"<InputLogView neuron={self._i} n_events={c} "
                f"last_consumed={int(self._pop.arrays.last_consumed_event[self._i])}>")


class DynamicStateView:
    """3x3 ``[2][1]``: 막전위·전도도·불응기·흔적 등 동적 상태 **참조**.

    모든 속성은 :class:`NeuronArrays` 를 읽고 쓴다 (사본 아님).
    """

    __slots__ = ("_pop", "_i")

    def __init__(self, population: "NeuronPopulation", neuron_id: int) -> None:
        self._pop = population
        self._i = int(neuron_id)

    # 막전위 --------------------------------------------------------
    @property
    def V_mV(self) -> np.ndarray:
        """shape (3,) 뷰. 인덱스는 soma/basal/apical."""
        return self._pop.arrays.V_mV[self._i]

    @property
    def V_soma_mV(self) -> float:
        return float(self._pop.arrays.V_mV[self._i, 0])

    @V_soma_mV.setter
    def V_soma_mV(self, value: float) -> None:
        self._pop.arrays.V_mV[self._i, 0] = float(value)

    # 전도도 --------------------------------------------------------
    @property
    def g_nS(self) -> np.ndarray:
        """shape (3, n_receptors) 뷰. 값은 비음수여야 한다."""
        return self._pop.arrays.g_nS[self._i]

    def g_of(self, compartment: str, receptor: str) -> float:
        ids = self._pop.ids
        c = ids.compartments.id_of(compartment)
        r = ids.receptors.id_of(receptor)
        return float(self._pop.arrays.g_nS[self._i, c, r])

    # 기타 ----------------------------------------------------------
    @property
    def refractory_until_ms(self) -> float:
        return float(self._pop.arrays.refractory_until_ms[self._i])

    @property
    def last_spike_ms(self) -> float:
        return float(self._pop.arrays.last_spike_ms[self._i])

    @property
    def rate_estimate_hz(self) -> float:
        return float(self._pop.arrays.rate_estimate_hz[self._i])

    @property
    def trace_pre(self) -> float:
        return float(self._pop.arrays.trace_pre[self._i])

    @property
    def trace_post(self) -> float:
        return float(self._pop.arrays.trace_post[self._i])

    @property
    def interval_activation(self) -> float:
        """sum_threshold 모드의 **현재 처리 구간** 누적값 (작업 버퍼, 무차원).

        장기 로그의 총합이 아니다. 매 구간 시작에 0 으로 초기화된다.
        """
        return float(self._pop.arrays.interval_activation[self._i])

    @property
    def fired(self) -> int:
        return int(self._pop.arrays.fired[self._i])

    def snapshot(self) -> dict[str, Any]:
        """읽기 전용 사본 (측정 함수용). 모델 상태를 바꾸지 않는다."""
        a = self._pop.arrays
        i = self._i
        return {
            "V_mV": a.V_mV[i].copy(),
            "g_nS": a.g_nS[i].copy(),
            "refractory_until_ms": float(a.refractory_until_ms[i]),
            "last_spike_ms": float(a.last_spike_ms[i]),
            "rate_estimate_hz": float(a.rate_estimate_hz[i]),
            "trace_pre": float(a.trace_pre[i]),
            "trace_post": float(a.trace_post[i]),
            "interval_activation": float(a.interval_activation[i]),
            "fired": int(a.fired[i]),
            "last_decision_value": float(a.last_decision_value[i]),
            "last_decision_threshold": float(a.last_decision_threshold[i]),
        }

    def __repr__(self) -> str:
        a = self._pop.arrays
        return (f"<DynamicStateView neuron={self._i} V_soma={a.V_mV[self._i, 0]:.3f}mV "
                f"g_sum={a.g_nS[self._i].sum():.4f}nS>")


class MetadataView:
    """3x3 ``[2][2]``: 영역·층·세포 유형·수용장·선호 특징 등 **메타데이터 참조**."""

    __slots__ = ("_pop", "_i")

    def __init__(self, population: "NeuronPopulation", neuron_id: int) -> None:
        self._pop = population
        self._i = int(neuron_id)

    @property
    def area(self) -> str:
        return self._pop.ids.areas.name_of(self._pop.arrays.area_id[self._i])

    @property
    def layer(self) -> str:
        return self._pop.ids.layers.name_of(self._pop.arrays.layer_id[self._i])

    @property
    def cell_type(self) -> str:
        return self._pop.ids.cell_types.name_of(self._pop.arrays.cell_type_id[self._i])

    @property
    def dale(self) -> str:
        return "excitatory" if self._pop.arrays.dale_sign[self._i] > 0 else "inhibitory"

    @property
    def compartments(self) -> tuple[str, ...]:
        mask = self._pop.arrays.has_compartment[self._i]
        return tuple(name for name, ok in zip(COMPARTMENT_NAMES, mask) if bool(ok))

    @property
    def visual_field_xy_deg(self) -> np.ndarray:
        """shape (2,) 시야 좌표 [deg]. 피질 mm 좌표와 **다른 공간**이다."""
        return self._pop.arrays.visual_field_xy_deg[self._i]

    @property
    def receptive_field(self) -> dict[str, Any]:
        a = self._pop.arrays
        i = self._i
        return {
            "center_xy_deg": a.visual_field_xy_deg[i].tolist(),
            "sigma_deg": float(a.rf_sigma_deg[i]),
            "space": "visual_field_deg",
        }

    @property
    def preferred(self) -> dict[str, Any]:
        a = self._pop.arrays
        i = self._i
        return {
            "orientation_rad": float(a.pref_orientation_rad[i]),
            "phase_rad": float(a.pref_phase_rad[i]),
            "on_off": int(a.on_off[i]),
            "channel_id": int(a.channel_id[i]),
            "ocular_dominance": float(a.ocular_dominance[i]),
            "eye_id": int(a.eye_id[i]),
            "hypercolumn_uv": a.hypercolumn_uv[i].tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "neuron_id": self._i,
            "area": self.area,
            "layer": self.layer,
            "cell_type": self.cell_type,
            "dale": self.dale,
            "compartments": list(self.compartments),
            "cortical_xyz_mm": self._pop.arrays.position_mm[self._i].tolist(),
            "receptive_field_parameters": self.receptive_field,
            "preferred": self.preferred,
        }

    def __repr__(self) -> str:
        return (f"<MetadataView neuron={self._i} {self.area}/{self.layer}/"
                f"{self.cell_type} ({self.dale})>")


# ----------------------------------------------------------------------
@dataclass(frozen=True)
class NeuronRecord:
    """뉴런 1개의 3x3 기록 인터페이스.

    ``population`` 의 배열을 참조하는 **가벼운 핸들**이다. 상태를 복사해 갖지
    않으므로 배열을 바꾸면 이 객체를 통해 본 값도 즉시 바뀐다.
    """

    population: "NeuronPopulation"
    neuron_id: int

    # --- 1행 ----------------------------------------------------------
    @property
    def x_mm(self) -> float:
        return float(self.population.arrays.position_mm[self.neuron_id, 0])

    @property
    def y_mm(self) -> float:
        return float(self.population.arrays.position_mm[self.neuron_id, 1])

    @property
    def z_mm(self) -> float:
        return float(self.population.arrays.position_mm[self.neuron_id, 2])

    # --- 2행 ----------------------------------------------------------
    @property
    def outgoing(self) -> OutgoingConnectionsView:
        return OutgoingConnectionsView(self.population, self.neuron_id)

    @property
    def threshold(self) -> float:
        """[1][1] 현재 발화 임계값 θ. 단위는 엔진 모드에 따른다 (mV 또는 무차원)."""
        return float(self.population.arrays.threshold[self.neuron_id])

    @threshold.setter
    def threshold(self, value: float) -> None:
        self.population.arrays.threshold[self.neuron_id] = float(value)

    @property
    def output_gain_P(self) -> float:
        """[1][2] 기저 출력 이득 P. 방출 확률이 아니다."""
        return float(self.population.arrays.output_gain_P[self.neuron_id])

    @output_gain_P.setter
    def output_gain_P(self, value: float) -> None:
        self.population.arrays.output_gain_P[self.neuron_id] = float(value)

    # --- 3행 ----------------------------------------------------------
    @property
    def input_log(self) -> InputLogView:
        return InputLogView(self.population, self.neuron_id)

    @property
    def dynamic_state(self) -> DynamicStateView:
        return DynamicStateView(self.population, self.neuron_id)

    @property
    def metadata(self) -> MetadataView:
        return MetadataView(self.population, self.neuron_id)

    # --- 3x3 조회 ------------------------------------------------------
    def as_matrix(self) -> np.ndarray:
        """명세 2절 배치의 3x3 **object 배열**.

        Returns
        -------
        np.ndarray, shape (3,3), dtype=object

        각 칸의 타입::

            [[float, float, float],
             [OutgoingConnectionsView, float, float],
             [InputLogView, DynamicStateView, MetadataView]]

        수치 행렬이 아니다. ``float32[3,3]`` 으로 바꾸려 하지 말 것.
        """
        m = np.empty((3, 3), dtype=object)
        m[0, 0] = self.x_mm
        m[0, 1] = self.y_mm
        m[0, 2] = self.z_mm
        m[1, 0] = self.outgoing
        m[1, 1] = self.threshold
        m[1, 2] = self.output_gain_P
        m[2, 0] = self.input_log
        m[2, 1] = self.dynamic_state
        m[2, 2] = self.metadata
        return m

    def as_matrix_description(self) -> list[list[str]]:
        """각 칸이 무엇인지 사람이 읽는 설명 (검증·문서용)."""
        return [
            ["x_mm", "y_mm", "z_mm"],
            ["outgoing_connection_list", "threshold_theta", "output_gain_P"],
            ["input_event_log_view", "dynamic_state_ref", "metadata_ref"],
        ]

    def describe(self) -> dict[str, Any]:
        """JSON 으로 저장 가능한 요약 (뷰 대신 요약값을 넣는다)."""
        return {
            "neuron_id": self.neuron_id,
            "cortical_xyz_mm": [self.x_mm, self.y_mm, self.z_mm],
            "n_outgoing": len(self.outgoing) if self.population.arrays.outgoing_built else None,
            "threshold": self.threshold,
            "output_gain_P": self.output_gain_P,
            "n_input_events": (self.input_log.count()
                               if self.population.event_log is not None else None),
            "dynamic_state": self.dynamic_state.snapshot(),
            "metadata": self.metadata.to_dict(),
        }


class NeuronPopulation:
    """뉴런 집단: 상태 배열 + ID 레지스트리 + (선택) 시냅스/로그 부착.

    3x3 기록 인터페이스는 :meth:`record` 로 얻는다.
    """

    def __init__(self, arrays: NeuronArrays, ids: IdSpace) -> None:
        self.arrays = arrays
        self.ids = ids
        self.synapses: "SynapseTable | None" = None
        self.event_log: "EventLog | None" = None
        self._name_of: list[str] | None = None

    def attach(self, synapses: "SynapseTable | None" = None,
               event_log: "EventLog | None" = None) -> None:
        """시냅스 테이블과 이벤트 로그를 연결한다 (부작용: 참조 설정)."""
        if synapses is not None:
            self.synapses = synapses
        if event_log is not None:
            self.event_log = event_log

    def __len__(self) -> int:
        return self.arrays.n

    def record(self, neuron_id: int) -> NeuronRecord:
        """뉴런 1개의 3x3 기록 인터페이스 핸들."""
        i = int(neuron_id)
        if not (0 <= i < self.arrays.n):
            raise IndexError(f"neuron_id 범위를 벗어났다: {neuron_id} (0..{self.arrays.n - 1})")
        return NeuronRecord(self, i)

    def records(self, neuron_ids: Sequence[int]) -> list[NeuronRecord]:
        return [self.record(i) for i in neuron_ids]

    # --- 집합 조회 ------------------------------------------------------
    def select(self, area: str | None = None, layer: str | None = None,
               cell_type: str | None = None) -> np.ndarray:
        """조건에 맞는 neuron_id 배열 (정렬됨, 부작용 없음)."""
        a = self.arrays
        mask = np.ones(a.n, dtype=bool)
        if area is not None:
            mask &= a.area_id == self.ids.areas.id_of(area)
        if layer is not None:
            mask &= a.layer_id == self.ids.layers.id_of(layer)
        if cell_type is not None:
            mask &= a.cell_type_id == self.ids.cell_types.id_of(cell_type)
        return np.nonzero(mask)[0]

    def counts_by_area_layer_type(self) -> dict[str, int]:
        """('area/layer/cell_type' -> 개수) 집계. 검증 9번에서 사용."""
        a = self.arrays
        out: dict[str, int] = {}
        for i in range(a.n):
            key = (f"{self.ids.areas.name_of(a.area_id[i])}/"
                   f"{self.ids.layers.name_of(a.layer_id[i])}/"
                   f"{self.ids.cell_types.name_of(a.cell_type_id[i])}")
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))


# ============================================================================
# 섹션: events  —  입력 이벤트 로그와 지연 사건 큐
#   (원래 파일: cortex/events.py)
# ============================================================================

"""events.py -- 입력 이벤트 로그와 지연 사건 큐.

명세 3절의 필드를 모두 갖춘다::

    event_id, parent_spike_id, sample_id, episode_id, src_id, dst_id,
    synapse_id, emit_time, arrival_time, arrival_step, weight_snapshot,
    source_gain_snapshot, target_compartment, event_type, amount, amount_unit

설계 선택
---------
* 저장은 **열 단위 NumPy 배열**(Structure-of-Arrays)이고 용량이 차면 2배로
  늘린다. 뉴런마다 무한 길이 Python 리스트를 들지 않는다.
* ``amount_unit`` 은 행마다 문자열을 저장하지 않고 정수 코드로 저장한다
  (:data:`AMOUNT_UNITS`). 문자열 매핑은 manifest 에 남는다.
* 뉴런별 조회는 ``dst_id`` 로 만든 **CSR 인덱스**를 지연 생성(lazy)한다.
  새 이벤트가 추가되면 인덱스를 무효화하고 다음 조회에서 다시 만든다.
* ``full`` / ``selected`` / ``summary`` 기록 모드를 지원한다. ``selected`` 와
  ``summary`` 는 선택 기준을 반드시 받아 manifest 에 남긴다. **조용히 버리고
  full 이라고 표시하지 않는다.**
* 용량 한도를 넘으면 :class:`CapacityExceeded` 를 올린다. 조용히 덮어쓰거나
  잘라내지 않는다.

발화 후에도 로그는 삭제하지 않는다. 현재 연산에 필요한 값은
:class:`cortex.records.NeuronArrays` 의 작업 버퍼(전도도/구간 누적)에 있고
로그와 분리되어 있다.
"""

#: amount 의 단위 코드. sum_threshold 모드는 무차원 기여량, 전도도 모드는 nS 다.
AMOUNT_UNITS: tuple[str, ...] = ("dimensionless", "nS", "pA", "spike", "weight_delta", "mV")
AMOUNT_UNIT_INDEX: dict[str, int] = {n: i for i, n in enumerate(AMOUNT_UNITS)}

EVENT_TYPE_INDEX: dict[str, int] = {n: i for i, n in enumerate(EVENT_TYPES)}

#: 이벤트 1행의 대략적 바이트 수 (용량 추정용, DATA_SCHEMA.md 와 일치)
BYTES_PER_EVENT_ROW: int = 88


class CapacityExceeded(RuntimeError):
    """설정한 이벤트/상태 용량 한도를 넘었을 때. 실행을 명시적으로 중단한다."""


class _Growable:
    """용량 2배 증가 방식의 1차원 growable 배열."""

    __slots__ = ("_buf", "_n")

    def __init__(self, dtype: Any, capacity: int = 1024) -> None:
        self._buf = np.zeros(int(capacity), dtype=dtype)
        self._n = 0

    def append_many(self, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=self._buf.dtype).ravel()
        need = self._n + v.size
        if need > self._buf.size:
            cap = max(need, self._buf.size * 2)
            new = np.zeros(cap, dtype=self._buf.dtype)
            new[: self._n] = self._buf[: self._n]
            self._buf = new
        self._buf[self._n : need] = v
        self._n = need

    @property
    def data(self) -> np.ndarray:
        """현재 유효 구간의 뷰 (사본 아님)."""
        return self._buf[: self._n]

    def __len__(self) -> int:
        return self._n

    def truncate(self, n: int) -> None:
        self._n = int(n)


class EventLog:
    """장기 입력/발화/학습 이벤트 로그.

    Parameters
    ----------
    n_neurons : int
    mode : {"full", "selected", "summary"}
    selected_neurons : array-like[int] | None
        ``selected`` 모드에서 기록할 뉴런 ID.
    selection_criterion : str
        ``full`` 이 아닐 때 필수. manifest 에 저장되어 재현 범위를 명시한다.
    max_events : int
        넘으면 :class:`CapacityExceeded`.
    """

    COLUMNS: tuple[str, ...] = (
        "event_id", "parent_spike_id", "sample_id", "episode_id",
        "src_id", "dst_id", "synapse_id",
        "emit_time_ms", "arrival_time_ms", "arrival_step",
        "weight_snapshot", "source_gain_snapshot",
        "target_compartment", "event_type", "amount", "amount_unit",
    )

    def __init__(self, n_neurons: int, mode: str = "full",
                 selected_neurons: Any = None, selection_criterion: str = "",
                 max_events: int = 20_000_000, capacity: int = 4096) -> None:
        if mode not in ("full", "selected", "summary"):
            raise ValueError("mode 는 full/selected/summary")
        if mode != "full" and not selection_criterion:
            raise ValueError(
                "full 이 아닌 기록 모드는 selection_criterion 을 반드시 지정해야 한다 "
                "(무엇을 기록하지 않는지 manifest 에 남긴다)"
            )
        self.n_neurons = int(n_neurons)
        self.mode = mode
        self.selection_criterion = selection_criterion
        self.max_events = int(max_events)

        self._keep = np.ones(self.n_neurons, dtype=bool)
        if mode == "selected":
            self._keep[:] = False
            if selected_neurons is not None:
                self._keep[np.asarray(selected_neurons, dtype=np.int64)] = True
        elif mode == "summary":
            self._keep[:] = False

        self._cols: dict[str, _Growable] = {
            "event_id": _Growable(np.int64, capacity),
            "parent_spike_id": _Growable(np.int64, capacity),
            "sample_id": _Growable(np.int32, capacity),
            "episode_id": _Growable(np.int32, capacity),
            "src_id": _Growable(np.int32, capacity),
            "dst_id": _Growable(np.int32, capacity),
            "synapse_id": _Growable(np.int64, capacity),
            "emit_time_ms": _Growable(np.float64, capacity),
            "arrival_time_ms": _Growable(np.float64, capacity),
            "arrival_step": _Growable(np.int64, capacity),
            "weight_snapshot": _Growable(np.float64, capacity),
            "source_gain_snapshot": _Growable(np.float64, capacity),
            "target_compartment": _Growable(np.int8, capacity),
            "event_type": _Growable(np.int8, capacity),
            "amount": _Growable(np.float64, capacity),
            "amount_unit": _Growable(np.int8, capacity),
        }
        self._index_ptr: np.ndarray | None = None
        self._index_rows: np.ndarray | None = None
        self._index_valid_upto = 0

        #: 기록 모드 때문에 저장하지 않은 이벤트 수 (조용히 버리지 않기 위해 센다)
        self.n_skipped_by_mode = 0
        #: summary 모드에서 유지하는 집계
        self.summary_counts: dict[int, int] = {}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._cols["event_id"])

    @property
    def n_events(self) -> int:
        return len(self._cols["event_id"])

    def column(self, name: str) -> np.ndarray:
        return self._cols[name].data

    def append(self, *, event_id: np.ndarray, parent_spike_id: np.ndarray,
               sample_id: int, episode_id: int,
               src_id: np.ndarray, dst_id: np.ndarray, synapse_id: np.ndarray,
               emit_time_ms: np.ndarray, arrival_time_ms: np.ndarray,
               arrival_step: np.ndarray, weight_snapshot: np.ndarray,
               source_gain_snapshot: np.ndarray, target_compartment: np.ndarray,
               event_type: str, amount: np.ndarray, amount_unit: str) -> int:
        """이벤트 여러 건을 한 번에 기록한다 (부작용: 로그 확장).

        Returns
        -------
        int : 실제로 저장한 행 수 (기록 모드로 걸러진 건 제외).
        """
        et = EVENT_TYPE_INDEX[event_type]
        au = AMOUNT_UNIT_INDEX[amount_unit]
        dst = np.asarray(dst_id, dtype=np.int64)

        if self.mode == "summary":
            self.summary_counts[et] = self.summary_counts.get(et, 0) + int(dst.size)
            self.n_skipped_by_mode += int(dst.size)
            return 0

        if self.mode == "selected":
            keep = np.zeros(dst.size, dtype=bool)
            valid = (dst >= 0) & (dst < self.n_neurons)
            keep[valid] = self._keep[dst[valid]]
            # 발화 이벤트는 src_id 로 판단한다 (dst 가 -1)
            src = np.asarray(src_id, dtype=np.int64)
            is_spike = dst < 0
            if is_spike.any():
                sv = is_spike & (src >= 0) & (src < self.n_neurons)
                keep[sv] = self._keep[src[sv]]
            n_drop = int((~keep).sum())
            if n_drop:
                self.n_skipped_by_mode += n_drop
            if not keep.any():
                return 0
            sel = np.nonzero(keep)[0]
        else:
            sel = None

        def pick(arr: Any, dtype: Any) -> np.ndarray:
            a = np.asarray(arr, dtype=dtype)
            if a.ndim == 0:
                a = np.full(dst.size, a, dtype=dtype)
            return a if sel is None else a[sel]

        n_new = dst.size if sel is None else sel.size
        if self.n_events + n_new > self.max_events:
            raise CapacityExceeded(
                f"이벤트 한도 초과: 현재 {self.n_events} + 신규 {n_new} > "
                f"max_events={self.max_events}. 설정 recording.max_events 를 늘리거나 "
                f"recording.mode 를 selected/summary 로 바꾸고 selection_criterion 을 기록하라."
            )

        self._cols["event_id"].append_many(pick(event_id, np.int64))
        self._cols["parent_spike_id"].append_many(pick(parent_spike_id, np.int64))
        self._cols["sample_id"].append_many(np.full(n_new, int(sample_id), np.int32))
        self._cols["episode_id"].append_many(np.full(n_new, int(episode_id), np.int32))
        self._cols["src_id"].append_many(pick(src_id, np.int32))
        self._cols["dst_id"].append_many(pick(dst_id, np.int32))
        self._cols["synapse_id"].append_many(pick(synapse_id, np.int64))
        self._cols["emit_time_ms"].append_many(pick(emit_time_ms, np.float64))
        self._cols["arrival_time_ms"].append_many(pick(arrival_time_ms, np.float64))
        self._cols["arrival_step"].append_many(pick(arrival_step, np.int64))
        self._cols["weight_snapshot"].append_many(pick(weight_snapshot, np.float64))
        self._cols["source_gain_snapshot"].append_many(pick(source_gain_snapshot, np.float64))
        self._cols["target_compartment"].append_many(pick(target_compartment, np.int8))
        self._cols["event_type"].append_many(np.full(n_new, et, np.int8))
        self._cols["amount"].append_many(pick(amount, np.float64))
        self._cols["amount_unit"].append_many(np.full(n_new, au, np.int8))
        self._index_ptr = None  # 인덱스 무효화
        return int(n_new)

    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """dst_id 기준 CSR 인덱스를 만든다 (조회 시 지연 생성)."""
        dst = self._cols["dst_id"].data
        n = self.n_neurons
        valid = (dst >= 0) & (dst < n)
        rows = np.nonzero(valid)[0]
        keys = dst[rows].astype(np.int64)
        order = np.argsort(keys, kind="stable")
        self._index_rows = rows[order]
        counts = np.bincount(keys, minlength=n)
        ptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=ptr[1:])
        self._index_ptr = ptr
        self._index_valid_upto = self.n_events

    def count_for_neuron(self, neuron_id: int) -> int:
        if self._index_ptr is None or self._index_valid_upto != self.n_events:
            self._build_index()
        assert self._index_ptr is not None
        i = int(neuron_id)
        return int(self._index_ptr[i + 1] - self._index_ptr[i])

    def query_neuron(self, neuron_id: int, t_from_ms: float | None = None,
                     t_to_ms: float | None = None,
                     limit: int | None = None) -> dict[str, np.ndarray]:
        """한 뉴런의 도착 이벤트를 시간 구간으로 조회한다 (부작용 없음)."""
        if self._index_ptr is None or self._index_valid_upto != self.n_events:
            self._build_index()
        assert self._index_ptr is not None and self._index_rows is not None
        i = int(neuron_id)
        lo, hi = int(self._index_ptr[i]), int(self._index_ptr[i + 1])
        rows = self._index_rows[lo:hi]
        if rows.size and (t_from_ms is not None or t_to_ms is not None):
            t = self._cols["arrival_time_ms"].data[rows]
            m = np.ones(rows.size, dtype=bool)
            if t_from_ms is not None:
                m &= t >= float(t_from_ms)
            if t_to_ms is not None:
                m &= t <= float(t_to_ms)
            rows = rows[m]
        if limit is not None and rows.size > limit:
            rows = rows[: int(limit)]
        return {name: col.data[rows] for name, col in self._cols.items()}

    def query_rows(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        rows = np.asarray(rows, dtype=np.int64)
        return {name: col.data[rows] for name, col in self._cols.items()}

    def all_columns(self) -> dict[str, np.ndarray]:
        return {name: col.data for name, col in self._cols.items()}

    def schema(self) -> dict[str, Any]:
        return {
            "columns": list(self.COLUMNS),
            "dtypes": {k: str(v.data.dtype) for k, v in self._cols.items()},
            "event_types": list(EVENT_TYPES),
            "amount_units": list(AMOUNT_UNITS),
            "compartments": list(COMPARTMENT_NAMES),
            "mode": self.mode,
            "selection_criterion": self.selection_criterion,
            "n_skipped_by_mode": self.n_skipped_by_mode,
        }


class DelayQueue:
    """도착 스텝별 버킷 큐.

    ``schedule(step, ...)`` 로 예약하고 ``pop(step)`` 으로 꺼낸다. 꺼낸 버킷은
    비워지므로 같은 사건이 두 번 주입되지 않는다.

    지연 0 은 허용하지 않는다 (``min_delay_steps >= 1``). 0 지연 재귀로 한
    스텝 안에서 무한 루프가 생기는 것을 구조적으로 막는다.
    """

    def __init__(self) -> None:
        self._buckets: dict[int, list[dict[str, np.ndarray]]] = {}
        self.n_scheduled = 0

    def schedule(self, arrival_step: int, payload: dict[str, np.ndarray]) -> None:
        step = int(arrival_step)
        if step < 0:
            raise ValueError("arrival_step 은 0 이상이어야 한다")
        self._buckets.setdefault(step, []).append(payload)
        first = next(iter(payload.values()))
        self.n_scheduled += int(np.asarray(first).size)

    def pop(self, step: int) -> dict[str, np.ndarray]:
        """해당 스텝의 사건을 합쳐 돌려주고 버킷을 비운다."""
        parts = self._buckets.pop(int(step), None)
        if not parts:
            return {}
        keys = parts[0].keys()
        return {k: np.concatenate([p[k] for p in parts]) for k in keys}

    def peek_steps(self) -> list[int]:
        return sorted(self._buckets)

    def pending_count(self) -> int:
        total = 0
        for parts in self._buckets.values():
            for p in parts:
                total += int(np.asarray(next(iter(p.values()))).size)
        return total

    def clear(self) -> None:
        self._buckets.clear()

    # --- 체크포인트 ---------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"steps": [], "payloads": []}
        for step in sorted(self._buckets):
            merged = {}
            keys = self._buckets[step][0].keys()
            for k in keys:
                merged[k] = np.concatenate([p[k] for p in self._buckets[step]])
            out["steps"].append(int(step))
            out["payloads"].append({k: v for k, v in merged.items()})
        return out

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._buckets.clear()
        self.n_scheduled = 0
        for step, payload in zip(d["steps"], d["payloads"]):
            self.schedule(int(step), {k: np.asarray(v) for k, v in payload.items()})


def quantize_delay(delay_ms: float | np.ndarray, dt_ms: float,
                   min_steps: int = 1, rounding: str = "ceil") -> np.ndarray:
    """연속 지연(ms)을 dt 격자 스텝 수로 양자화한다.

    Parameters
    ----------
    delay_ms : float | ndarray
    dt_ms : float
    min_steps : int
        최소 지연 스텝. 1 이상이어야 0 지연 재귀가 생기지 않는다.
    rounding : {"ceil", "round"}

    Returns
    -------
    ndarray[int64] : 양자화된 스텝 수 (>= min_steps)

    연속 지연과 양자화 지연을 **둘 다** 기록한다 (명세 6절).
    """
    if min_steps < 1:
        raise ValueError("min_steps 는 1 이상이어야 한다 (0 지연 재귀 방지)")
    d = np.asarray(delay_ms, dtype=np.float64) / float(dt_ms)
    if rounding == "ceil":
        steps = np.ceil(d)
    elif rounding == "round":
        steps = np.rint(d)
    else:
        raise ValueError("rounding 은 'ceil' 또는 'round'")
    return np.maximum(steps.astype(np.int64), int(min_steps))


# ============================================================================
# 섹션: synapses  —  희소 edge table 과 인덱스
#   (원래 파일: cortex/synapses.py)
# ============================================================================

"""synapses.py -- 희소 edge table 과 인덱스.

명세 3절의 필드를 모두 갖춘다::

    synapse_id, src_id, dst_id, src_area, dst_area, target_layer,
    target_compartment, receptor_type, weight, weight_unit,
    base_delay_ms, effective_delay_steps, plasticity_rule,
    plasticity_state_id, active

설계 선택
---------
* 열 단위 NumPy 배열 + ``scipy.sparse`` CSR 인덱스. 전체 N×N 밀집 행렬을
  만들지 않는다.
* ``weight`` 는 **비음수 크기**다. 흥분/억제 부호는 **발신 뉴런의 Dale 유형**이
  결정하며, 학습으로 부호가 뒤집히지 않는다 (``clip(w, 0, w_max)``).
  전도도 모드에서 억제성 시냅스의 Δg 를 음수로 만들지 않는다. 실제 전류 방향은
  역전전위 ``E_rev`` 와 막전위가 결정한다.
* ``weight_unit`` 은 엔진 모드에 따른다: ``conductance_lif`` 는 nS,
  ``sum_threshold`` 는 무차원.
* ``plasticity_state_id`` 는 가소성 규칙별 상태 배열의 행 인덱스다 (-1 이면 없음).
"""

try:  # scipy 는 필수 의존성이다. 없으면 즉시 알린다.
    from scipy import sparse as _sparse
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "SciPy 가 필요하다. `python -m pip install -r requirements.txt` 를 실행하라."
    ) from _exc


class SynapseTable:
    """희소 시냅스 테이블.

    ``add_block`` 으로 규칙별 시냅스를 덩어리로 추가하고, 모두 추가한 뒤
    ``build_indices()`` 를 호출해 CSR 인덱스를 만든다.
    """

    COLUMNS: tuple[str, ...] = (
        "synapse_id", "src_id", "dst_id", "src_area", "dst_area",
        "target_layer", "target_compartment", "receptor_type",
        "weight", "weight_unit", "base_delay_ms", "effective_delay_steps",
        "plasticity_rule", "plasticity_state_id", "active",
    )

    def __init__(self, n_neurons: int, weight_unit: str) -> None:
        self.n_neurons = int(n_neurons)
        self.weight_unit = str(weight_unit)
        self._blocks: list[dict[str, np.ndarray]] = []
        self._built = False

        self.src_id = np.zeros(0, dtype=np.int32)
        self.dst_id = np.zeros(0, dtype=np.int32)
        self.src_area = np.zeros(0, dtype=np.int16)
        self.dst_area = np.zeros(0, dtype=np.int16)
        self.target_layer = np.zeros(0, dtype=np.int16)
        self.target_compartment = np.zeros(0, dtype=np.int8)
        self.receptor_type = np.zeros(0, dtype=np.int8)
        self.weight = np.zeros(0, dtype=np.float64)
        self.base_delay_ms = np.zeros(0, dtype=np.float64)
        self.effective_delay_steps = np.zeros(0, dtype=np.int32)
        self.plasticity_rule = np.zeros(0, dtype=np.int8)
        self.plasticity_state_id = np.zeros(0, dtype=np.int64)
        self.active = np.zeros(0, dtype=bool)
        self.rule_index = np.zeros(0, dtype=np.int16)   # 어느 배선 규칙에서 나왔는지

        # 발신 뉴런의 Dale 부호 (+1/-1). 전류 부호가 아니라 **연결 유형**이다.
        self.src_dale_sign = np.zeros(0, dtype=np.int8)

        # CSR 인덱스
        self.in_ptr = np.zeros(self.n_neurons + 1, dtype=np.int64)
        self.in_syn = np.zeros(0, dtype=np.int64)
        self.out_ptr = np.zeros(self.n_neurons + 1, dtype=np.int64)
        self.out_syn = np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------------
    def add_block(self, *, src_id: np.ndarray, dst_id: np.ndarray,
                  src_area: np.ndarray, dst_area: np.ndarray,
                  target_layer: np.ndarray, target_compartment: int,
                  receptor_type: int, weight: np.ndarray,
                  base_delay_ms: np.ndarray, effective_delay_steps: np.ndarray,
                  plasticity_rule: int, src_dale_sign: np.ndarray,
                  rule_index: int) -> int:
        """시냅스 여러 개를 한 번에 추가한다 (부작용: 내부 블록 목록 확장)."""
        if self._built:
            raise RuntimeError("build_indices() 이후에는 시냅스를 추가할 수 없다")
        w = np.asarray(weight, dtype=np.float64)
        if np.any(w < 0.0):
            raise ValueError(
                "weight 는 비음수 크기여야 한다. 흥분/억제 부호는 발신 뉴런의 "
                "Dale 유형이 결정한다."
            )
        d = np.asarray(effective_delay_steps, dtype=np.int32)
        if np.any(d < 1):
            raise ValueError("effective_delay_steps 는 1 이상이어야 한다 (0 지연 재귀 방지)")
        n = w.size
        self._blocks.append({
            "src_id": np.asarray(src_id, dtype=np.int32),
            "dst_id": np.asarray(dst_id, dtype=np.int32),
            "src_area": np.asarray(src_area, dtype=np.int16),
            "dst_area": np.asarray(dst_area, dtype=np.int16),
            "target_layer": np.asarray(target_layer, dtype=np.int16),
            "target_compartment": np.full(n, int(target_compartment), dtype=np.int8),
            "receptor_type": np.full(n, int(receptor_type), dtype=np.int8),
            "weight": w,
            "base_delay_ms": np.asarray(base_delay_ms, dtype=np.float64),
            "effective_delay_steps": d,
            "plasticity_rule": np.full(n, int(plasticity_rule), dtype=np.int8),
            "active": np.ones(n, dtype=bool),
            "src_dale_sign": np.asarray(src_dale_sign, dtype=np.int8),
            "rule_index": np.full(n, int(rule_index), dtype=np.int16),
        })
        return n

    def pending_count(self) -> int:
        return int(sum(b["weight"].size for b in self._blocks))

    def build_indices(self) -> None:
        """블록을 이어 붙이고 in/out CSR 인덱스를 만든다."""
        if self._built:
            return
        if not self._blocks:
            self._built = True
            self.plasticity_state_id = np.zeros(0, dtype=np.int64)
            return
        keys = [k for k in self._blocks[0]]
        for k in keys:
            setattr(self, k, np.concatenate([b[k] for b in self._blocks]))
        self._blocks = []
        k_total = self.weight.size
        self.plasticity_state_id = np.where(
            self.plasticity_rule > 0, np.arange(k_total, dtype=np.int64), -1
        )

        order_in = np.argsort(self.dst_id.astype(np.int64), kind="stable")
        self.in_syn = order_in.astype(np.int64)
        counts = np.bincount(self.dst_id.astype(np.int64), minlength=self.n_neurons)
        self.in_ptr = np.zeros(self.n_neurons + 1, dtype=np.int64)
        np.cumsum(counts, out=self.in_ptr[1:])

        order_out = np.argsort(self.src_id.astype(np.int64), kind="stable")
        self.out_syn = order_out.astype(np.int64)
        counts_o = np.bincount(self.src_id.astype(np.int64), minlength=self.n_neurons)
        self.out_ptr = np.zeros(self.n_neurons + 1, dtype=np.int64)
        np.cumsum(counts_o, out=self.out_ptr[1:])
        self._built = True

    @property
    def built(self) -> bool:
        return self._built

    def __len__(self) -> int:
        return int(self.weight.size) if self._built else self.pending_count()

    @property
    def n_synapses(self) -> int:
        return len(self)

    # ------------------------------------------------------------------
    def outgoing_of(self, neuron_id: int) -> np.ndarray:
        i = int(neuron_id)
        return self.out_syn[self.out_ptr[i]:self.out_ptr[i + 1]]

    def incoming_of(self, neuron_id: int) -> np.ndarray:
        i = int(neuron_id)
        return self.in_syn[self.in_ptr[i]:self.in_ptr[i + 1]]

    def row(self, synapse_id: int) -> dict[str, Any]:
        s = int(synapse_id)
        return {
            "synapse_id": s,
            "src_id": int(self.src_id[s]),
            "dst_id": int(self.dst_id[s]),
            "src_area": int(self.src_area[s]),
            "dst_area": int(self.dst_area[s]),
            "target_layer": int(self.target_layer[s]),
            "target_compartment": int(self.target_compartment[s]),
            "receptor_type": int(self.receptor_type[s]),
            "weight": float(self.weight[s]),
            "weight_unit": self.weight_unit,
            "base_delay_ms": float(self.base_delay_ms[s]),
            "effective_delay_steps": int(self.effective_delay_steps[s]),
            "plasticity_rule": int(self.plasticity_rule[s]),
            "plasticity_state_id": int(self.plasticity_state_id[s]),
            "active": bool(self.active[s]),
            "src_dale_sign": int(self.src_dale_sign[s]),
            "rule_index": int(self.rule_index[s]),
        }

    def connectivity_matrix(self) -> "_sparse.csr_matrix":
        """(N,N) CSR 희소 행렬. 값은 **부호를 적용한** 가중치다 (시각화/집계용).

        이 행렬은 계산 경로에서 쓰지 않는다. 계산은 이벤트 기반이다.
        """
        data = self.weight * self.src_dale_sign.astype(np.float64)
        return _sparse.csr_matrix(
            (data, (self.src_id.astype(np.int64), self.dst_id.astype(np.int64))),
            shape=(self.n_neurons, self.n_neurons),
        )

    # --- Dale 제약과 범위 ----------------------------------------------
    def clip_weights(self, w_min: float, w_max: float) -> dict[str, int]:
        """가중치를 [w_min, w_max] 로 자른다. 부호(연결 유형)는 바뀌지 않는다.

        Returns
        -------
        dict : 하한/상한에 걸린 시냅스 수
        """
        if w_min < 0.0:
            raise ValueError("w_min 은 0 이상이어야 한다 (Dale 부호 보존)")
        lo = int(np.count_nonzero(self.weight < w_min))
        hi = int(np.count_nonzero(self.weight > w_max))
        np.clip(self.weight, w_min, w_max, out=self.weight)
        return {"clipped_low": lo, "clipped_high": hi}

    def dale_violations(self, neuron_dale_sign: np.ndarray) -> np.ndarray:
        """발신 뉴런의 Dale 부호와 시냅스에 기록된 부호가 다른 행을 찾는다."""
        expected = np.asarray(neuron_dale_sign, dtype=np.int8)[self.src_id]
        return np.nonzero(expected != self.src_dale_sign)[0]

    def summary(self, id_space: Any = None) -> dict[str, Any]:
        """집계 요약 (검증 9번, 보고서용)."""
        if not self._built:
            return {"built": False, "pending": self.pending_count()}
        out: dict[str, Any] = {
            "built": True,
            "n_synapses": int(self.weight.size),
            "n_active": int(np.count_nonzero(self.active)),
            "weight_unit": self.weight_unit,
            "weight_min": float(self.weight.min()) if self.weight.size else None,
            "weight_max": float(self.weight.max()) if self.weight.size else None,
            "weight_mean": float(self.weight.mean()) if self.weight.size else None,
            "n_excitatory_sources": int(np.count_nonzero(self.src_dale_sign > 0)),
            "n_inhibitory_sources": int(np.count_nonzero(self.src_dale_sign < 0)),
            "delay_steps_min": int(self.effective_delay_steps.min()) if self.weight.size else None,
            "delay_steps_max": int(self.effective_delay_steps.max()) if self.weight.size else None,
            "base_delay_ms_min": float(self.base_delay_ms.min()) if self.weight.size else None,
            "base_delay_ms_max": float(self.base_delay_ms.max()) if self.weight.size else None,
            "n_by_rule_index": {int(k): int(v) for k, v in
                                zip(*np.unique(self.rule_index, return_counts=True))},
        }
        return out

    def state_dict(self) -> dict[str, np.ndarray]:
        """체크포인트용 배열 묶음 (가중치와 활성 플래그 포함)."""
        return {
            "src_id": self.src_id, "dst_id": self.dst_id,
            "src_area": self.src_area, "dst_area": self.dst_area,
            "target_layer": self.target_layer,
            "target_compartment": self.target_compartment,
            "receptor_type": self.receptor_type, "weight": self.weight,
            "base_delay_ms": self.base_delay_ms,
            "effective_delay_steps": self.effective_delay_steps,
            "plasticity_rule": self.plasticity_rule,
            "plasticity_state_id": self.plasticity_state_id,
            "active": self.active, "src_dale_sign": self.src_dale_sign,
            "rule_index": self.rule_index,
        }

    def load_state_dict(self, d: dict[str, np.ndarray]) -> None:
        for k, v in d.items():
            setattr(self, k, np.asarray(v))
        self._blocks = []
        self._built = False
        self.build_indices()


# ============================================================================
# 섹션: retina  —  영상 입력, 색 변환, DoG, ON/OFF, 입력 구동
#   (원래 파일: cortex/retina.py)
# ============================================================================

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


# ============================================================================
# 섹션: retinotopy  —  불균일(로그-극좌표) 샘플링과 좌표 변환
#   (원래 파일: cortex/retinotopy.py)
# ============================================================================

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


# ============================================================================
# 섹션: anatomy  —  3D 배치, 층, 세포 유형, 방향/안구우세 지도
#   (원래 파일: cortex/anatomy.py)
# ============================================================================

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


def build_anatomy(cfg: dict[str, Any], rng: np.random.Generator) -> Anatomy:
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


# ============================================================================
# 섹션: areas  —  영역 간/영역 내 배선 생성
#   (원래 파일: cortex/areas.py)
# ============================================================================

"""areas.py -- 영역 간/영역 내 배선 생성.

명세 6절.

배선 규칙
---------
규칙 하나는 ``(src 영역/층/세포유형) -> (dst 영역/층/세포유형)`` 집합 사이에
시냅스를 만든다. 후보 선택 방식은 세 가지다.

``rf_knn``
    **시야 좌표(deg)** 에서 dst 뉴런의 수용장 중심에 가까운 src 후보 k 개를
    ``cKDTree`` 로 찾는다. 원시 3D 거리만 쓰지 않고 시야 위치 대응과
    영역·층·세포유형 마스크를 함께 쓴다. ``rf_match_sigma_deg`` 로 거리에 따른
    연결 확률 감쇠를 준다. 상위 영역일수록 이 값이 커져 수용장이 넓어진다.
``local_radius``
    **피질 좌표(mm)** 에서 반경 안의 후보를 찾는다 (국소/수평 연결용).
``all_to_all_sampled``
    후보 전체에서 확률로 뽑는다 (작은 설정 전용).

전체 N×N 거리 행렬을 만들지 않는다. 언제나 KD-tree 질의로 후보만 본다.

지연
----
``delay_ms = synaptic_delay_ms + distance_mm / conduction_velocity_mm_per_ms``
직선 3D 거리를 축삭 길이로 쓰는 근사이며 ``use_straight_line_distance`` 로
표시한다. 연속 지연(ms)과 dt 격자로 양자화한 스텝 수를 **둘 다** 기록한다.
최소 지연은 1 스텝이므로 0 지연 재귀가 생기지 않는다.

Gabor 초기 배선
---------------
``gabor_initialized`` 규칙은 dst 의 선호 방향/위상과 src 의 시야 위치로 Gabor
계수를 계산해 초기 가중치를 정한다. **음의 계수를 음의 전도도로 만들지 않는다.**
ON 극성 src 는 계수의 양수부, OFF 극성 src 는 음수부의 절댓값을 쓴다.
억제성 src 는 절댓값을 쓰고 부호는 Dale 유형이 만든다. 이 초기화를 사용하면
"방향 선택성을 학습했다"고 말하지 않는다 (manifest 에 기록된다).
"""

@dataclass
class WiringReport:
    """배선 결과 집계 (manifest/보고서에 그대로 들어간다)."""

    n_synapses: int
    per_rule: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"n_synapses": self.n_synapses, "per_rule": self.per_rule,
                "notes": self.notes}


def _mask_for(anat: Anatomy, spec: dict[str, Any]) -> np.ndarray:
    a = anat.population.arrays
    ids = anat.ids
    m = a.area_id == ids.areas.id_of(spec["area"])
    if spec["layer"]:
        lm = np.zeros_like(m)
        for lname in spec["layer"]:
            lm |= a.layer_id == ids.layers.id_of(lname)
        m &= lm
    if spec["cell_type"]:
        cm = np.zeros_like(m)
        for cname in spec["cell_type"]:
            cm |= a.cell_type_id == ids.cell_types.id_of(cname)
        m &= cm
    return m


def _sample_weights(rule: dict[str, Any], n: int,
                    rng: np.random.Generator) -> np.ndarray:
    w = rule["weight"]
    if w["dist"] == "constant":
        vals = np.full(n, float(w["median"]), dtype=np.float64)
    elif w["dist"] == "normal":
        vals = rng.normal(float(w["median"]), float(w["sigma"]), size=n)
    else:  # lognormal: median 이 exp(mu) 가 되도록 mu = log(median)
        vals = rng.lognormal(np.log(max(float(w["median"]), 1e-12)),
                             float(w["sigma"]), size=n)
    return np.clip(vals, float(w["min"]), float(w["max"]))


def gabor_coefficient(dx_deg: np.ndarray, dy_deg: np.ndarray,
                      orientation_rad: np.ndarray, phase_rad: np.ndarray,
                      sigma_deg: float, aspect: float,
                      cycles_per_deg: float) -> np.ndarray:
    """원본 시야 Cartesian 좌표에서 평가한 Gabor 계수.

    **왜곡된 로그-극좌표 배열 위의 직선 필터가 아니다.** 각 수용장 안에서
    원본 시야 좌표 차이를 그대로 쓴다.
    """
    ct, st = np.cos(orientation_rad), np.sin(orientation_rad)
    xr = dx_deg * ct + dy_deg * st
    yr = -dx_deg * st + dy_deg * ct
    env = np.exp(-0.5 * ((xr / sigma_deg) ** 2
                         + (yr / (sigma_deg * aspect)) ** 2))
    return env * np.cos(2.0 * np.pi * cycles_per_deg * xr + phase_rad)


def build_wiring(cfg: dict[str, Any], anat: Anatomy,
          rng_wiring: np.random.Generator,
          rng_weights: np.random.Generator) -> tuple[SynapseTable, WiringReport]:
    """배선 규칙을 모두 적용해 시냅스 테이블을 만든다.

    Parameters
    ----------
    rng_wiring : 후보 선택/확률 샘플링 스트림
    rng_weights : 초기 가중치 스트림 (분리해 두면 가중치만 바꾸는 대조군이 쉽다)

    Raises
    ------
    CapacityExceeded
        ``wiring.max_total_synapses`` 를 넘으면 조용히 자르지 않고 중단한다.
    """
    a = anat.population.arrays
    ids = anat.ids
    dt = float(cfg["engine"]["dt_ms"])
    min_steps = int(cfg["engine"]["min_delay_steps"])
    rounding = cfg["engine"]["delay_rounding"]
    weight_unit = "nS" if cfg["engine"]["mode"] == "conductance_lif" else "dimensionless"
    table = SynapseTable(len(anat.population), weight_unit)
    report = WiringReport(n_synapses=0)
    max_total = int(cfg["wiring"]["max_total_synapses"])
    allow_self = bool(cfg["wiring"]["allow_self_connection"])
    gi = cfg["v1"]["gabor_init"]

    for r_idx, rule in enumerate(cfg["wiring"]["rules"]):
        if not rule["enabled"]:
            report.per_rule.append({"name": rule["name"], "enabled": False, "n": 0})
            continue
        src_mask = _mask_for(anat, rule["src"])
        dst_mask = _mask_for(anat, rule["dst"])
        src_ids = np.nonzero(src_mask)[0]
        dst_ids = np.nonzero(dst_mask)[0]
        if src_ids.size == 0 or dst_ids.size == 0:
            report.per_rule.append({
                "name": rule["name"], "enabled": True, "n": 0,
                "warning": "src 또는 dst 집합이 비어 있다 (배선 0개)",
                "n_src": int(src_ids.size), "n_dst": int(dst_ids.size)})
            continue

        pairs_src, pairs_dst = _candidate_pairs(rule, a, src_ids, dst_ids, rng_wiring)
        if pairs_src.size == 0:
            report.per_rule.append({"name": rule["name"], "enabled": True, "n": 0,
                                    "warning": "후보가 선택되지 않았다"})
            continue
        if not allow_self:
            keep = pairs_src != pairs_dst
            pairs_src, pairs_dst = pairs_src[keep], pairs_dst[keep]

        # 확률 적용
        if rule["probability"] < 1.0:
            keep = rng_wiring.random(pairs_src.size) < float(rule["probability"])
            pairs_src, pairs_dst = pairs_src[keep], pairs_dst[keep]
        if pairs_src.size == 0:
            report.per_rule.append({"name": rule["name"], "enabled": True, "n": 0,
                                    "warning": "확률 적용 후 후보가 남지 않았다"})
            continue

        if rule["max_synapses_per_target"] > 0:
            pairs_src, pairs_dst = _cap_per_target(
                pairs_src, pairs_dst, int(rule["max_synapses_per_target"]))

        n_new = int(pairs_src.size)
        if table.pending_count() + n_new > max_total:
            raise CapacityExceeded(
                f"배선 규칙 '{rule['name']}' 에서 시냅스 한도를 넘었다: "
                f"{table.pending_count()} + {n_new} > {max_total}. "
                f"wiring.max_total_synapses 를 늘리거나 규칙의 k/probability 를 줄여라."
            )

        # 가중치
        gabor_used = False
        if rule["gabor_initialized"]:
            w = _gabor_weights(a, pairs_src, pairs_dst, gi)
            wcfg = rule["weight"]
            w = np.clip(w * float(wcfg["median"]), float(wcfg["min"]), float(wcfg["max"]))
            gabor_used = True
        else:
            w = _sample_weights(rule, n_new, rng_weights)

        # 지연: 3D 피질 거리 기반 (직선 근사)
        d3 = np.linalg.norm(a.position_mm[pairs_src] - a.position_mm[pairs_dst], axis=1)
        delay_ms = (float(rule["synaptic_delay_ms"])
                    + d3 / float(rule["conduction_velocity_mm_per_ms"]))
        steps = quantize_delay(delay_ms, dt, min_steps, rounding)

        table.add_block(
            src_id=pairs_src, dst_id=pairs_dst,
            src_area=a.area_id[pairs_src], dst_area=a.area_id[pairs_dst],
            target_layer=a.layer_id[pairs_dst],
            target_compartment=COMPARTMENT_INDEX[rule["target_compartment"]],
            receptor_type=ids.receptors.id_of(rule["receptor"]),
            weight=w, base_delay_ms=delay_ms, effective_delay_steps=steps,
            plasticity_rule=ids.plasticity_rules.id_of(rule["plasticity_rule"]),
            src_dale_sign=a.dale_sign[pairs_src], rule_index=r_idx,
        )
        report.n_synapses += n_new
        report.per_rule.append({
            "name": rule["name"], "enabled": True, "n": n_new,
            "n_src": int(src_ids.size), "n_dst": int(dst_ids.size),
            "rule": rule["rule"], "receptor": rule["receptor"],
            "target_compartment": rule["target_compartment"],
            "plasticity_rule": rule["plasticity_rule"],
            "gabor_initialized": gabor_used,
            "weight_min": float(w.min()), "weight_max": float(w.max()),
            "weight_mean": float(w.mean()),
            "delay_ms_min": float(delay_ms.min()), "delay_ms_max": float(delay_ms.max()),
            "delay_steps_min": int(steps.min()), "delay_steps_max": int(steps.max()),
            "straight_line_distance_approximation": bool(rule["use_straight_line_distance"]),
            "mean_synapses_per_target": float(n_new / max(1, dst_ids.size)),
        })

    table.build_indices()
    anat.population.arrays.set_outgoing_index(table.out_ptr, table.out_syn)
    anat.population.attach(synapses=table)

    if any(x.get("gabor_initialized") for x in report.per_rule):
        report.notes.append(
            "일부 규칙의 초기 가중치를 Gabor 모양으로 정했다. 이 경로의 방향 선택성을 "
            "'학습되었다'고 보고하지 않는다."
        )
    report.notes.append(
        "지연은 직선 3D 거리를 축삭 길이로 쓰는 근사를 포함한다. 연속 지연(ms)과 "
        "양자화 스텝을 모두 저장했다."
    )
    return table, report


def _candidate_pairs(rule: dict[str, Any], a: Any, src_ids: np.ndarray,
                     dst_ids: np.ndarray,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """규칙별 (src, dst) 후보 쌍. 전체 거리 행렬을 만들지 않는다."""
    kind = rule["rule"]
    if kind == "all_to_all_sampled":
        s = np.repeat(src_ids, dst_ids.size)
        d = np.tile(dst_ids, src_ids.size)
        return s, d

    if kind == "local_radius":
        pts_src = a.position_mm[src_ids]
        pts_dst = a.position_mm[dst_ids]
        tree = cKDTree(pts_src)
        neigh = tree.query_ball_point(pts_dst, r=float(rule["radius_mm"]))
        s_list, d_list = [], []
        for j, cand in enumerate(neigh):
            if not cand:
                continue
            c = np.asarray(cand, dtype=np.int64)
            if c.size > rule["k"] > 0:
                c = rng.choice(c, size=int(rule["k"]), replace=False)
            s_list.append(src_ids[c])
            d_list.append(np.full(c.size, dst_ids[j], dtype=np.int64))
        if not s_list:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(s_list), np.concatenate(d_list)

    # rf_knn: 시야 좌표(deg) 에서 k 최근접 + 거리 감쇠 확률
    vf_src = a.visual_field_xy_deg[src_ids]
    vf_dst = a.visual_field_xy_deg[dst_ids]
    finite = np.isfinite(vf_src).all(axis=1)
    if not finite.any():
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    src_ids = src_ids[finite]
    vf_src = vf_src[finite]
    tree = cKDTree(vf_src)
    k = min(int(rule["k"]), src_ids.size)
    dist, idx = tree.query(vf_dst, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    sigma = float(rule["rf_match_sigma_deg"])
    p = np.exp(-0.5 * (dist / max(sigma, 1e-9)) ** 2)
    keep = rng.random(p.shape) < p
    rows, cols = np.nonzero(keep)
    return src_ids[idx[rows, cols]], dst_ids[rows]


def _cap_per_target(src: np.ndarray, dst: np.ndarray, cap: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """dst 당 시냅스 수를 cap 으로 제한한다 (앞쪽 우선, 결정적)."""
    order = np.argsort(dst, kind="stable")
    s, d = src[order], dst[order]
    keep = np.zeros(d.size, dtype=bool)
    if d.size == 0:
        return s, d
    start = 0
    for i in range(1, d.size + 1):
        if i == d.size or d[i] != d[start]:
            keep[start: min(i, start + cap)] = True
            start = i
    return s[keep], d[keep]


def _gabor_weights(a: Any, src: np.ndarray, dst: np.ndarray,
                   gi: dict[str, Any]) -> np.ndarray:
    """Gabor 계수에서 **비음수** 가중치 크기를 만든다.

    음의 계수를 음의 전도도로 만들지 않는다:

    * src 가 ON 극성(``on_off=+1``)  -> ``max(coef, 0)``
    * src 가 OFF 극성(``on_off=-1``) -> ``max(-coef, 0)``
    * 극성이 없는 src               -> ``|coef|`` (부호는 Dale 유형이 만든다)
    """
    dx = a.visual_field_xy_deg[src, 0] - a.visual_field_xy_deg[dst, 0]
    dy = a.visual_field_xy_deg[src, 1] - a.visual_field_xy_deg[dst, 1]
    ori = np.nan_to_num(a.pref_orientation_rad[dst], nan=0.0)
    ph = np.nan_to_num(a.pref_phase_rad[dst], nan=0.0)
    coef = gabor_coefficient(dx, dy, ori, ph, float(gi["sigma_deg"]),
                             float(gi["aspect"]), float(gi["cycles_per_deg"]))
    pol = a.on_off[src]
    w = np.abs(coef)
    w = np.where(pol > 0, np.maximum(coef, 0.0), w)
    w = np.where(pol < 0, np.maximum(-coef, 0.0), w)
    return w


def connectivity_summary(cfg: dict[str, Any], anat: Anatomy,
                         table: SynapseTable) -> dict[str, Any]:
    """층·세포유형·영역별 연결 집계 (검증 9번).

    "세포 유형/층 이름만 존재하는 구현"을 잡기 위해, 각 규칙이 실제로 만든
    시냅스 수와 층별 in/out degree 를 함께 낸다.
    """
    a = anat.population.arrays
    ids = anat.ids
    out: dict[str, Any] = {"synapses": table.summary(), "by_pathway": {}, "degree": {}}
    if not table.built or table.n_synapses == 0:
        out["warning"] = "시냅스가 하나도 만들어지지 않았다"
        return out
    key = (a.area_id[table.src_id].astype(np.int64) * 10_000
           + a.layer_id[table.src_id].astype(np.int64) * 100
           + a.area_id[table.dst_id].astype(np.int64))
    uniq, counts = np.unique(key, return_counts=True)
    for k, c in zip(uniq, counts):
        sa, sl, da = int(k // 10_000), int((k // 100) % 100), int(k % 100)
        name = (f"{ids.areas.name_of(sa)}/{ids.layers.name_of(sl)}"
                f" -> {ids.areas.name_of(da)}")
        out["by_pathway"][name] = int(c)
    in_deg = np.diff(table.in_ptr)
    out_deg = np.diff(table.out_ptr)
    for aname in ids.areas.names():
        m = a.area_id == ids.areas.id_of(aname)
        if not m.any():
            continue
        out["degree"][aname] = {
            "n_neurons": int(m.sum()),
            "in_degree_mean": float(in_deg[m].mean()),
            "in_degree_max": int(in_deg[m].max()),
            "out_degree_mean": float(out_deg[m].mean()),
            "out_degree_max": int(out_deg[m].max()),
            "n_isolated_no_input": int(np.count_nonzero(in_deg[m] == 0)),
            "n_isolated_no_output": int(np.count_nonzero(out_deg[m] == 0)),
        }
    return out


# ============================================================================
# 섹션: dynamics  —  공통 시간 루프와 두 동작 모드
#   (원래 파일: cortex/dynamics.py)
# ============================================================================

"""dynamics.py -- 공통 시간 루프와 두 동작 모드.

명세 4, 9절. 한 스텝은 **항상** 다음 순서다.

1. 외부 자극과 예약된 지연 사건 수집
2. 도착 사건을 입력 로그에 기록
3. 새 사건을 **한 번만** 수용체/작업 버퍼에 반영
4. 시냅스·막전위·구획 상태를 dt 만큼 갱신
5. 발화와 불응기 판정, 필요 시 리셋
6. 출력 목록에 따라 도착 시각이 정해진 사건 예약
7. 가소성 흔적과 학습 규칙 갱신
8. 선택된 상태와 실제 가중치/임계 변화 기록

**숨은 순서 의존성 없음**: 한 스텝의 모든 발화는 같은 (스텝 시작 시점의)
가중치로 판정·예약되고, 학습 갱신은 7단계에서 한꺼번에 적용된다.

모드 A: ``sum_threshold``
-------------------------
무차원 합산 모델이다. 생물학적 막전위 모델이 **아니다**::

    s_j = sum(effective_contribution of arrivals for neuron j in this interval)
    q_j = 1 if s_j >= theta_j else 0

한 처리 구간의 입력을 **모두 모은 뒤** 판정하므로 입력 목록 순서가 결과를
바꾸지 않는다 (누적은 synapse_id 순으로 정렬해 수행한다).
경계 조건: 입력이 0 이어도 ``theta_j <= 0`` 이면 발화한다.

모드 B: ``conductance_lif``
---------------------------
구획별 전도도 기반 LIF. 적분은 **후향 오일러(반암시적)** 이며 구획 결합까지
포함한 3x3 선형계를 뉴런마다 닫힌 형태로 푼다. 무조건 안정이므로 발산을
가리기 위한 임의 전압 클리핑을 넣지 않았다. 실제 이산식은 equations.md 참조.

전도도는 도착 사건으로 **한 번만** 증가하고, 그 뒤 수용체별 시정수로 감쇠하며
효과는 이후 스텝에도 상태로 남는다. 새로 도착한 사건과 이전 입력 때문에 남아
있는 상태를 구분한다.
"""

@dataclass
class StepReport:
    """한 스텝의 요약 (기록/진단용)."""

    step: int
    time_ms: float
    n_arrivals: int
    n_spikes: int
    n_scheduled: int
    mean_V_soma_mV: float = float("nan")
    max_abs_V_mV: float = float("nan")
    n_nonfinite: int = 0
    decided: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExternalDrive:
    """망막(또는 지정 뉴런)에 주는 외생 입력.

    Attributes
    ----------
    neuron_ids : (M,) int64
    current_pA : (M,) float64
        ``rate`` 모드에서 쓰는 지속 전류 (conductance 모드).
    sum_contribution : (M,) float64
        ``sum_threshold`` 모드에서 매 스텝 더할 무차원 기여.
    forced_spike_steps : dict[int, np.ndarray] | None
        ``poisson`` 모드에서 강제 발화시킬 (스텝 -> 뉴런 ID 배열).
    """

    neuron_ids: np.ndarray
    current_pA: np.ndarray | None = None
    sum_contribution: np.ndarray | None = None
    forced_spike_steps: dict[int, np.ndarray] | None = None
    n_multi_event_truncated: int = 0


class Engine:
    """공통 시간 루프 엔진.

    Parameters
    ----------
    cfg : 해석된 설정
    anat : :class:`cortex.anatomy.Anatomy`
    table : :class:`cortex.synapses.SynapseTable`
    event_log : :class:`cortex.events.EventLog`
    plasticity : 객체 | None
        ``on_step(engine, spiked, dt_ms)`` 와 ``on_spikes(...)`` 를 제공하는
        학습 모듈 (:mod:`cortex.plasticity`). None 이면 학습 없음.
    recorder : 객체 | None
        ``record_step(...)`` 을 제공하는 기록기. 계산을 바꾸지 않는다.
    """

    def __init__(self, cfg: dict[str, Any], anat: Anatomy, table: SynapseTable,
                 event_log: EventLog, plasticity: Any = None,
                 recorder: Any = None) -> None:
        self.cfg = cfg
        self.anat = anat
        self.pop = anat.population
        self.a: NeuronArrays = anat.population.arrays
        self.table = table
        self.log = event_log
        self.plasticity = plasticity
        self.recorder = recorder

        eng = cfg["engine"]
        self.mode = eng["mode"]
        self.dt_ms = float(eng["dt_ms"])
        self.weight_application = eng["weight_application"]
        self.interval_steps = int(eng["sum_threshold_interval_steps"])

        self.queue = DelayQueue()
        self.step_index = 0
        self.time_ms = 0.0
        self.sample_id = 0
        self.episode_id = 0
        self.event_counter = IdCounter(0)
        self.spike_counter = IdCounter(0)

        # 수용체 상수 (인덱스 = receptor_id)
        r_names = anat.ids.receptors.names()
        self.receptor_names = r_names
        self.tau_r = np.array([cfg["receptors"][n]["tau_ms"] for n in r_names], np.float64)
        self.E_rev = np.array([cfg["receptors"][n]["E_rev_mV"] for n in r_names], np.float64)
        self.mg_block = np.array([bool(cfg["receptors"][n]["mg_block"]) for n in r_names])
        self.decay = np.exp(-self.dt_ms / self.tau_r)
        # 스텝 동안의 시간 평균 계수: (tau/dt)(1-exp(-dt/tau))
        self.mean_factor = (self.tau_r / self.dt_ms) * (1.0 - self.decay)

        self.drive: ExternalDrive | None = None
        # 망막 구동 전류는 사용자가 직접 설정한 Iext_pA 와 **더해진다**.
        # 엔진이 Iext_pA 를 매 스텝 0 으로 덮어쓰지 않는다.
        self._drive_I_pA = np.zeros((self.a.n, 3), dtype=np.float64)
        self.n_nonfinite_total = 0
        self.step_reports: list[StepReport] = []
        self._interval_counter = 0

    # ------------------------------------------------------------------
    # 상태 초기화
    # ------------------------------------------------------------------
    def reset_transient(self, *, voltages: bool = True, conductances: bool = True,
                        event_queue: bool = True, traces: bool = False,
                        thresholds: bool = False) -> None:
        """표본 사이 초기화. **가중치는 절대 지우지 않는다.**

        무엇을 리셋하고 무엇을 유지할지는 ``engine.reset_between_samples`` 설정이
        결정하며, 독립 영상 실험과 연속 시퀀스 실험을 구분한다.
        """
        a = self.a
        if voltages:
            a.V_mV[:, :] = a.EL_mV
            a.refractory_until_ms[:] = -np.inf
            a.fired[:] = 0
        if conductances:
            a.g_nS[:, :, :] = 0.0
        if event_queue:
            self.queue.clear()
        if traces:
            a.trace_pre[:] = 0.0
            a.trace_post[:] = 0.0
            a.rate_estimate_hz[:] = 0.0
        if thresholds:
            raise ValueError(
                "임계값 초기화는 학습 결과를 지우는 동작이다. reset_parameters() 를 "
                "명시적으로 호출하라."
            )
        a.interval_activation[:] = 0.0
        self._interval_counter = 0
        self.step_index = 0
        self.time_ms = 0.0

    def set_external_drive(self, drive: ExternalDrive | None) -> None:
        self.drive = drive

    # ------------------------------------------------------------------
    # 한 스텝
    # ------------------------------------------------------------------
    def step(self) -> StepReport:
        """명세 9절의 8단계를 순서대로 수행한다."""
        a = self.a
        t = self.time_ms
        n = self.step_index

        # --- 1. 예약 사건과 외부 자극 수집 -----------------------------
        payload = self.queue.pop(n)
        n_arrivals = 0
        if payload:
            order = np.argsort(payload["synapse_id"], kind="stable")
            payload = {k: v[order] for k, v in payload.items()}
            n_arrivals = int(payload["synapse_id"].size)

        # --- 2. 도착 사건을 입력 로그에 기록 ---------------------------
        if n_arrivals:
            syn = payload["synapse_id"]
            dst = self.table.dst_id[syn].astype(np.int64)
            comp = self.table.target_compartment[syn]
            if self.weight_application == "arrival":
                w_used = self.table.weight[syn]
                amount = w_used * payload["gain_snapshot"]
            else:
                w_used = payload["weight_snapshot"]
                amount = payload["amount"]
            ev_start = self.event_counter.take(n_arrivals)
            ev_ids = np.arange(ev_start, ev_start + n_arrivals, dtype=np.int64)
            self.log.append(
                event_id=ev_ids, parent_spike_id=payload["spike_id"],
                sample_id=self.sample_id, episode_id=self.episode_id,
                src_id=self.table.src_id[syn], dst_id=dst, synapse_id=syn,
                emit_time_ms=payload["emit_time_ms"],
                arrival_time_ms=np.full(n_arrivals, t),
                arrival_step=np.full(n_arrivals, n, dtype=np.int64),
                weight_snapshot=w_used, source_gain_snapshot=payload["gain_snapshot"],
                target_compartment=comp, event_type="synaptic_arrival",
                amount=amount,
                amount_unit="nS" if self.mode == "conductance_lif" else "dimensionless",
            )
            a.last_consumed_event[dst] = np.maximum(
                a.last_consumed_event[dst], ev_ids[-1])

        # --- 3. 새 사건을 한 번만 반영 ----------------------------------
        if n_arrivals:
            self._apply_arrivals(payload, amount, t)

        if self.mode == "conductance_lif":
            rep = self._step_conductance(n, t)
        else:
            rep = self._step_sum_threshold(n, t)
        rep.n_arrivals = n_arrivals
        rep.n_scheduled = self.queue.pending_count()

        # --- 8. 기록 ----------------------------------------------------
        if self.recorder is not None:
            self.recorder.record_step(self, rep)
        self.step_reports.append(rep)

        self.step_index += 1
        self.time_ms += self.dt_ms
        return rep

    # ------------------------------------------------------------------
    def _apply_arrivals(self, payload: dict[str, np.ndarray],
                        amount: np.ndarray, t: float) -> None:
        """도착값을 작업 버퍼(전도도 또는 구간 누적)에 **한 번만** 더한다."""
        a = self.a
        syn = payload["synapse_id"]
        dst = self.table.dst_id[syn].astype(np.int64)
        if self.mode == "conductance_lif":
            comp = self.table.target_compartment[syn].astype(np.int64)
            rec = self.table.receptor_type[syn].astype(np.int64)
            amt = np.maximum(amount, 0.0)  # 억제성 Δg 를 음수로 만들지 않는다
            # 불응기 동안 soma 입력을 버리는 세포 유형 처리
            drop = (a.refractory_discards_input[dst] & (comp == 0)
                    & (t < a.refractory_until_ms[dst]))
            if drop.any():
                keep = ~drop
                dst, comp, rec, amt = dst[keep], comp[keep], rec[keep], amt[keep]
            if dst.size:
                np.add.at(a.g_nS, (dst, comp, rec), amt)
        else:
            sign = self.table.src_dale_sign[syn].astype(np.float64)
            np.add.at(a.interval_activation, dst, amount * sign)

    # ------------------------------------------------------------------
    def _step_conductance(self, n: int, t: float) -> StepReport:
        a = self.a
        dt = self.dt_ms

        # 망막 구동 전류 (rate 모드). 사용자가 설정한 a.Iext_pA 는 유지된다.
        self._drive_I_pA[:, :] = 0.0
        if self.drive is not None and self.drive.current_pA is not None:
            self._drive_I_pA[self.drive.neuron_ids, 0] = self.drive.current_pA
        I_total = a.Iext_pA + self._drive_I_pA

        # --- 4. 시냅스/막전위/구획 상태 갱신 ---------------------------
        g0 = a.g_nS                                   # (N,3,R) 도착 반영 직후
        g_bar = g0 * self.mean_factor[None, None, :]  # 스텝 동안의 시간 평균

        V_pre = a.V_mV                                # (N,3)
        block = np.ones_like(g_bar)
        if self.mg_block.any():
            b = 1.0 / (1.0 + np.exp(-MG_BLOCK_SLOPE * V_pre) * MG_BLOCK_MG_MM / MG_BLOCK_SCALE)
            block[:, :, self.mg_block] = b[:, :, None]
        g_eff = g_bar * block                          # (N,3,R)
        g_sum = g_eff.sum(axis=2)                      # (N,3)
        g_E = (g_eff * self.E_rev[None, None, :]).sum(axis=2)

        g_sb = a.g_couple_nS[:, COMPARTMENT_INDEX["basal"]]
        g_sa = a.g_couple_nS[:, COMPARTMENT_INDEX["apical"]]
        c_dt = a.C_pF / dt                             # (N,3)

        diag = c_dt + a.gL_nS + g_sum
        a_s = diag[:, 0] + g_sb + g_sa
        a_b = diag[:, 1] + g_sb
        a_a = diag[:, 2] + g_sa

        rhs = c_dt * V_pre + a.gL_nS * a.EL_mV + g_E + I_total
        denom = a_s - (g_sb * g_sb) / a_b - (g_sa * g_sa) / a_a
        num = rhs[:, 0] + g_sb * rhs[:, 1] / a_b + g_sa * rhs[:, 2] / a_a
        V_s = num / denom
        V_b = (rhs[:, 1] + g_sb * V_s) / a_b
        V_a = (rhs[:, 2] + g_sa * V_s) / a_a

        V_new = np.stack([V_s, V_b, V_a], axis=1)
        absent = ~a.has_compartment
        V_new[absent] = a.EL_mV[absent]

        # 전도도 감쇠 (도착 효과는 상태로 남는다)
        a.g_nS *= self.decay[None, None, :]

        # --- 5. 발화/불응기 판정 ---------------------------------------
        refractory = t < a.refractory_until_ms
        V_new[refractory, 0] = a.V_reset_mV[refractory]
        a.V_mV[:, :] = V_new
        a.last_decision_value[:] = a.V_mV[:, 0]
        a.last_decision_threshold[:] = a.threshold

        spiked = (a.V_mV[:, 0] >= a.threshold) & (~refractory)
        if self.drive is not None and self.drive.forced_spike_steps is not None:
            forced = self.drive.forced_spike_steps.get(n)
            if forced is not None and forced.size:
                spiked[forced] = True
        spiked_ids = np.nonzero(spiked)[0]
        if spiked_ids.size:
            a.V_mV[spiked_ids, 0] = a.V_reset_mV[spiked_ids]
            a.refractory_until_ms[spiked_ids] = t + a.t_ref_ms[spiked_ids]
            a.last_spike_ms[spiked_ids] = t
            a.spike_count[spiked_ids] += 1
        a.fired[:] = 0
        a.fired[spiked_ids] = 1

        nonfinite = int(np.count_nonzero(~np.isfinite(a.V_mV)))
        self.n_nonfinite_total += nonfinite

        # --- 6. 출력 사건 예약 -----------------------------------------
        self._emit(spiked_ids, n, t)
        # --- 7. 가소성 --------------------------------------------------
        if self.plasticity is not None:
            self.plasticity.on_step(self, spiked_ids, dt)

        return StepReport(
            step=n, time_ms=t, n_arrivals=0, n_spikes=int(spiked_ids.size),
            n_scheduled=0,
            mean_V_soma_mV=float(np.nanmean(a.V_mV[:, 0])) if a.n else float("nan"),
            max_abs_V_mV=float(np.nanmax(np.abs(a.V_mV))) if a.n else float("nan"),
            n_nonfinite=nonfinite,
        )

    # ------------------------------------------------------------------
    def _step_sum_threshold(self, n: int, t: float) -> StepReport:
        a = self.a
        if self.drive is not None and self.drive.sum_contribution is not None:
            np.add.at(a.interval_activation, self.drive.neuron_ids,
                      self.drive.sum_contribution)

        self._interval_counter += 1
        if self._interval_counter < self.interval_steps:
            return StepReport(step=n, time_ms=t, n_arrivals=0, n_spikes=0,
                              n_scheduled=0, decided=False)
        self._interval_counter = 0

        a.last_decision_value[:] = a.interval_activation
        a.last_decision_threshold[:] = a.threshold
        spiked = a.interval_activation >= a.threshold
        if self.drive is not None and self.drive.forced_spike_steps is not None:
            forced = self.drive.forced_spike_steps.get(n)
            if forced is not None and forced.size:
                spiked[forced] = True
        spiked_ids = np.nonzero(spiked)[0]
        a.fired[:] = 0
        a.fired[spiked_ids] = 1
        if spiked_ids.size:
            a.last_spike_ms[spiked_ids] = t
            a.spike_count[spiked_ids] += 1
        # 작업 버퍼만 비운다. 장기 입력 로그는 그대로 남는다.
        a.interval_activation[:] = 0.0

        self._emit(spiked_ids, n, t)
        if self.plasticity is not None:
            self.plasticity.on_step(self, spiked_ids, self.dt_ms)
        return StepReport(step=n, time_ms=t, n_arrivals=0,
                          n_spikes=int(spiked_ids.size), n_scheduled=0)

    # ------------------------------------------------------------------
    def _emit(self, spiked_ids: np.ndarray, n: int, t: float) -> None:
        """발화 뉴런의 출력 사건을 지연에 따라 예약한다 (6단계).

        가중치/이득은 설정 ``engine.weight_application`` 에 따라 발신 시점
        스냅샷(기본) 또는 도착 시점 값을 쓴다. 실제 적용한 값을 이벤트에 남긴다.
        """
        if spiked_ids.size == 0:
            return
        a = self.a
        sp_start = self.spike_counter.take(int(spiked_ids.size))
        spike_ids = np.arange(sp_start, sp_start + spiked_ids.size, dtype=np.int64)

        ev_start = self.event_counter.take(int(spiked_ids.size))
        self.log.append(
            event_id=np.arange(ev_start, ev_start + spiked_ids.size, dtype=np.int64),
            parent_spike_id=spike_ids, sample_id=self.sample_id,
            episode_id=self.episode_id, src_id=spiked_ids,
            dst_id=np.full(spiked_ids.size, -1, dtype=np.int64),
            synapse_id=np.full(spiked_ids.size, -1, dtype=np.int64),
            emit_time_ms=np.full(spiked_ids.size, t),
            arrival_time_ms=np.full(spiked_ids.size, t),
            arrival_step=np.full(spiked_ids.size, n, dtype=np.int64),
            weight_snapshot=np.zeros(spiked_ids.size),
            source_gain_snapshot=a.output_gain_P[spiked_ids],
            target_compartment=np.full(spiked_ids.size, -1, dtype=np.int8),
            event_type="spike", amount=np.ones(spiked_ids.size), amount_unit="spike",
        )

        spike_of = np.zeros(a.n, dtype=np.int64)
        spike_of[spiked_ids] = spike_ids
        out_ptr, out_syn = self.table.out_ptr, self.table.out_syn
        counts = out_ptr[spiked_ids + 1] - out_ptr[spiked_ids]
        total = int(counts.sum())
        if total == 0:
            return
        syn = np.concatenate([out_syn[out_ptr[i]:out_ptr[i + 1]] for i in spiked_ids])
        src = self.table.src_id[syn].astype(np.int64)
        active = self.table.active[syn]
        if not active.all():
            syn, src = syn[active], src[active]
        if syn.size == 0:
            return
        gain = a.output_gain_P[src]
        w = self.table.weight[syn]
        amount = w * gain
        steps = self.table.effective_delay_steps[syn].astype(np.int64)
        arrival = n + steps
        emit_time = np.full(syn.size, t)
        sp = spike_of[src]

        for st in np.unique(arrival):
            m = arrival == st
            self.queue.schedule(int(st), {
                "synapse_id": syn[m],
                "amount": amount[m],
                "weight_snapshot": w[m],
                "gain_snapshot": gain[m],
                "emit_time_ms": emit_time[m],
                "spike_id": sp[m],
            })

    # ------------------------------------------------------------------
    def run(self, n_steps: int, on_step: Callable[[StepReport], None] | None = None
            ) -> list[StepReport]:
        """n_steps 만큼 실행한다. 진행 콜백은 계산을 바꾸지 않아야 한다."""
        out: list[StepReport] = []
        for _ in range(int(n_steps)):
            rep = self.step()
            out.append(rep)
            if on_step is not None:
                on_step(rep)
        return out

    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """체크포인트용 상태 (가중치·전도도·흔적·이벤트 큐·시간)."""
        a = self.a
        return {
            "V_mV": a.V_mV.copy(), "g_nS": a.g_nS.copy(),
            "threshold": a.threshold.copy(),
            "refractory_until_ms": a.refractory_until_ms.copy(),
            "last_spike_ms": a.last_spike_ms.copy(),
            "spike_count": a.spike_count.copy(),
            "rate_estimate_hz": a.rate_estimate_hz.copy(),
            "trace_pre": a.trace_pre.copy(), "trace_post": a.trace_post.copy(),
            "interval_activation": a.interval_activation.copy(),
            "last_consumed_event": a.last_consumed_event.copy(),
            "weight": self.table.weight.copy(),
            "queue": self.queue.state_dict(),
            "step_index": self.step_index, "time_ms": self.time_ms,
            "sample_id": self.sample_id, "episode_id": self.episode_id,
            "event_counter": self.event_counter.value,
            "spike_counter": self.spike_counter.value,
            "interval_counter": self._interval_counter,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        a = self.a
        for k in ("V_mV", "g_nS", "threshold", "refractory_until_ms", "last_spike_ms",
                  "spike_count", "rate_estimate_hz", "trace_pre", "trace_post",
                  "interval_activation", "last_consumed_event"):
            getattr(a, k)[...] = np.asarray(d[k])
        self.table.weight[...] = np.asarray(d["weight"])
        self.queue.load_state_dict(d["queue"])
        self.step_index = int(d["step_index"])
        self.time_ms = float(d["time_ms"])
        self.sample_id = int(d["sample_id"])
        self.episode_id = int(d["episode_id"])
        self.event_counter.restore(int(d["event_counter"]))
        self.spike_counter.restore(int(d["spike_counter"]))
        self._interval_counter = int(d["interval_counter"])

    def read_only_snapshot(self) -> dict[str, np.ndarray]:
        """측정 함수용 깊은 복사. 모델/RNG 를 바꾸지 않는다 (검증 10번)."""
        a = self.a
        return {
            "V_mV": a.V_mV.copy(), "g_nS": a.g_nS.copy(),
            "threshold": a.threshold.copy(), "weight": self.table.weight.copy(),
            "trace_pre": a.trace_pre.copy(), "trace_post": a.trace_post.copy(),
            "last_consumed_event": a.last_consumed_event.copy(),
            "rate_estimate_hz": a.rate_estimate_hz.copy(),
        }


def settle_chain(engine: Engine, *, n_steps: int, drive: ExternalDrive | None,
                 initial_state: dict[str, Any] | None = None,
                 weight_snapshot: np.ndarray | None = None,
                 freeze_learning: bool = False) -> dict[str, Any]:
    """명시적 입력·초기 상태·가중치 스냅샷·길이로 회로를 정착시킨다.

    명세 9절: 자유/유도 비교에서 초기 막전위·전도도·흔적·대기 이벤트·난수 상태·
    시간 길이·회로 마스크를 **일치시키고 교사/문맥 항만** 바꾸기 위한 진입점이다.
    두 조건 사이에서 학습 업데이트를 먼저 적용하지 않는다 (``freeze_learning``).

    Returns
    -------
    dict : 스파이크 회로의 **안정화 지표** (연속값 Rao 의 에너지 수렴 지표와 다르다)
    """
    if initial_state is not None:
        engine.load_state_dict(initial_state)
    if weight_snapshot is not None:
        engine.table.weight[...] = np.asarray(weight_snapshot)
    saved = engine.plasticity
    if freeze_learning:
        engine.plasticity = None
    engine.set_external_drive(drive)
    try:
        reports = engine.run(int(n_steps))
    finally:
        engine.plasticity = saved

    spikes = np.array([r.n_spikes for r in reports], dtype=np.float64)
    half = max(1, len(spikes) // 2)
    return {
        "n_steps": int(n_steps),
        "total_spikes": int(spikes.sum()),
        "spikes_first_half": float(spikes[:half].mean()) if spikes.size else 0.0,
        "spikes_second_half": float(spikes[half:].mean()) if spikes.size else 0.0,
        "spike_rate_drift": (float(spikes[half:].mean() - spikes[:half].mean())
                             if spikes.size else 0.0),
        "max_abs_V_mV": float(np.nanmax([r.max_abs_V_mV for r in reports]))
        if reports else float("nan"),
        "n_nonfinite": int(sum(r.n_nonfinite for r in reports)),
        "pending_events": engine.queue.pending_count(),
        "metric_kind": "spiking_circuit_settling",
        "note_ko": ("스파이크 회로의 안정화 지표다. Rao 연속값 모델의 에너지 수렴 "
                    "지표와 같은 양이 아니므로 섞어 비교하지 않는다."),
    }


def rates_to_drive(cfg: dict[str, Any], neuron_ids: np.ndarray,
                   rates_hz: np.ndarray, n_steps: int,
                   rng: np.random.Generator) -> ExternalDrive:
    """발화율 -> :class:`ExternalDrive`.

    ``rate`` 모드는 지속 전류(또는 무차원 기여)로, ``poisson`` 모드는 미리 뽑아
    둔 강제 발화 스텝으로 만든다. 미리 뽑아 두면 대조군에서 **동일한 외생
    사건을 재생**할 수 있다.
    """
    d = cfg["retina"]["drive"]
    ids = np.asarray(neuron_ids, dtype=np.int64)
    r = np.asarray(rates_hz, dtype=np.float64)
    if d["mode"] == "rate":
        if cfg["engine"]["mode"] == "conductance_lif":
            return ExternalDrive(ids, current_pA=r * float(d["current_per_hz_pA"]))
        contrib = r * float(cfg["engine"]["dt_ms"]) / MS_PER_S * float(d["sum_mode_scale"])
        return ExternalDrive(ids, sum_contribution=contrib)

    lam = r * float(cfg["engine"]["dt_ms"]) / MS_PER_S
    forced: dict[int, np.ndarray] = {}
    truncated = 0
    for s in range(int(n_steps)):
        c = rng.poisson(lam)
        nz = np.nonzero(c > 0)[0]
        if nz.size:
            forced[s] = ids[nz]
            truncated += int((c[nz] - 1).sum())
    return ExternalDrive(ids, forced_spike_steps=forced,
                         n_multi_event_truncated=truncated)


# ============================================================================
# 섹션: plasticity  —  국소 가소성(STDP)과 항상성 임계 적응
#   (원래 파일: cortex/plasticity.py)
# ============================================================================

"""plasticity.py -- 국소 가소성(STDP)과 항상성 임계 적응.

명세 8절.

STDP 수식 (실제 구현)
---------------------
전/후 흔적은 스텝마다 지수 감쇠한다::

    x_pre[i]  <- x_pre[i]  * exp(-dt/tau_plus)
    x_post[j] <- x_post[j] * exp(-dt/tau_minus)

발화한 뉴런의 흔적은 1 만큼 증가한다 (all-to-all pair 방식).
시냅스 s (i -> j) 의 갱신은::

    j 가 발화하면 (LTP):  dw_s += A_plus  * x_pre[i]
    i 가 발화하면 (LTD):  dw_s -= A_minus * x_post[j]

동시 발화 처리 (``simultaneous_policy``)

* ``both``       : 같은 스텝의 전/후 발화 모두에 대해 위 두 항을 적용한다.
                   두 항 모두 **흔적 증가 이전** 값을 쓴다 (자기 자신의 스파이크가
                   자기 갱신에 들어가지 않는다).
* ``pre_first``  : 전 흔적을 먼저 증가시킨 뒤 LTP 를 계산한다 (동시 발화 시 LTP 우세).
* ``post_first`` : 후 흔적을 먼저 증가시킨 뒤 LTD 를 계산한다 (동시 발화 시 LTD 우세).

갱신 순서 ``update_order`` 는 LTP/LTD 중 어느 쪽을 먼저 누적할지 정한다.
가중치는 비음수 크기이므로 ``clip(w, weight_min>=0, weight_max)`` 로 잘리며
**흥분 연결이 억제 연결로 뒤집히지 않는다**.

이것은 "동시 발화 곱" 이 아니라 전/후 사건의 **시간 차**를 흔적으로 표현한
pair-based STDP 다.

항상성 임계 적응
----------------
활동률 추정은 지수 이동 평균이다 (window_ms 가 누적 시간)::

    r[i] <- r[i] + (dt/window_ms) * (spike[i]/(dt/1000) - r[i])      [Hz]
    theta[i] <- clip(theta[i] + eta_theta*(r[i]-target[i])*(dt/window_ms), lo, hi)

상하한은 엔진 모드에 따라 mV(``theta_min_mV``/``theta_max_mV``) 또는 무차원
(``theta_min_sum``/``theta_max_sum``) 을 쓴다. 목표 활동률은 세포 유형별로
다를 수 있다.

**이웃 임계값 평균화는 기본값이 아니다.** 소거(ablation) 실험 옵션으로만 두며,
이것을 "정답 교정"이라고 부르지 않는다.
"""

@dataclass
class UpdateRecord:
    """한 스텝의 실제 변화량 (updates.h5 에 그대로 저장된다)."""

    step: int
    time_ms: float
    n_synapses_changed: int = 0
    ltp_sum: float = 0.0
    ltd_sum: float = 0.0
    decay_sum: float = 0.0
    clipped_low: int = 0
    clipped_high: int = 0
    weight_delta_sum: float = 0.0
    weight_delta_abs_sum: float = 0.0
    theta_delta_sum: float = 0.0
    theta_delta_abs_sum: float = 0.0
    n_theta_clipped: int = 0
    teacher_derived: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dict(vars(self))
        d.pop("extra")
        d.update(self.extra)
        return d


class StdpHomeostasis:
    """STDP + 항상성 임계 적응 모듈.

    Parameters
    ----------
    cfg : 해석된 설정
    table : :class:`cortex.synapses.SynapseTable`
    arrays : :class:`cortex.records.NeuronArrays`
    ids : :class:`cortex.ids.IdSpace`
    """

    def __init__(self, cfg: dict[str, Any], table: Any, arrays: Any, ids: Any) -> None:
        self.cfg = cfg
        self.table = table
        self.a = arrays
        self.ids = ids
        lr = cfg["learning"]
        self.task_enabled = bool(lr["task_learning_enabled"])
        self.decay_enabled = bool(lr["weight_decay_enabled"])
        self.theta_enabled = bool(lr["threshold_adaptation_enabled"])
        s = lr["stdp"]
        self.A_plus = float(s["A_plus"])
        self.A_minus = float(s["A_minus"])
        self.tau_plus = float(s["tau_plus_ms"])
        self.tau_minus = float(s["tau_minus_ms"])
        self.w_min = float(s["weight_min"])
        self.w_max = float(s["weight_max"])
        self.decay_per_ms = float(s["weight_decay_per_ms"])
        self.simultaneous_policy = s["simultaneous_policy"]
        self.update_order = s["update_order"]
        self.apply_to_inhibitory = bool(s["apply_to_inhibitory"])

        h = lr["homeostasis"]
        self.eta_theta = float(h["eta_theta"])
        self.window_ms = float(h["window_ms"])
        mode = cfg["engine"]["mode"]
        if mode == "conductance_lif":
            self.theta_lo = float(h["theta_min_mV"])
            self.theta_hi = float(h["theta_max_mV"])
        else:
            self.theta_lo = float(h["theta_min_sum"])
            self.theta_hi = float(h["theta_max_sum"])
        self.per_cell_type_target = bool(h["per_cell_type_target"])

        nb = lr["neighbor_theta_averaging"]
        self.neighbor_enabled = bool(nb["enabled"])
        self.neighbor_radius_mm = float(nb["radius_mm"])
        self.neighbor_strength = float(nb["strength"])
        self._neighbor_index: Any = None

        # 학습 대상 시냅스 마스크
        stdp_id = ids.plasticity_rules.id_of("stdp")
        self.plastic = table.plasticity_rule == stdp_id
        if not self.apply_to_inhibitory:
            self.plastic = self.plastic & (table.src_dale_sign > 0)
        self.plastic_idx = np.nonzero(self.plastic)[0]

        self.records: list[UpdateRecord] = []
        self.total_weight_delta = 0.0
        self.total_theta_delta = 0.0

    # ------------------------------------------------------------------
    def on_step(self, engine: Any, spiked_ids: np.ndarray, dt_ms: float) -> UpdateRecord:
        """7단계에서 호출된다. 흔적/가중치/임계값을 갱신한다 (부작용 있음)."""
        a = self.a
        rec = UpdateRecord(step=engine.step_index, time_ms=engine.time_ms)

        # 흔적 감쇠 (스텝 시작 상태 -> 현재)
        a.trace_pre *= np.exp(-dt_ms / self.tau_plus)
        a.trace_post *= np.exp(-dt_ms / self.tau_minus)

        pre_before = a.trace_pre.copy()
        post_before = a.trace_post.copy()
        if self.simultaneous_policy == "pre_first" and spiked_ids.size:
            pre_before = pre_before.copy()
            pre_before[spiked_ids] += 1.0
        if self.simultaneous_policy == "post_first" and spiked_ids.size:
            post_before = post_before.copy()
            post_before[spiked_ids] += 1.0

        if self.task_enabled and self.plastic_idx.size and spiked_ids.size:
            dw = np.zeros(self.table.weight.size, dtype=np.float64)
            steps = (("ltp", "ltd") if self.update_order == "post_then_pre"
                     else ("ltd", "ltp"))
            for which in steps:
                if which == "ltp":
                    syn = self._incoming_plastic(spiked_ids)
                    if syn.size:
                        contrib = self.A_plus * pre_before[self.table.src_id[syn]]
                        np.add.at(dw, syn, contrib)
                        rec.ltp_sum += float(contrib.sum())
                else:
                    syn = self._outgoing_plastic(spiked_ids)
                    if syn.size:
                        contrib = self.A_minus * post_before[self.table.dst_id[syn]]
                        np.add.at(dw, syn, -contrib)
                        rec.ltd_sum += float(contrib.sum())
            if self.decay_enabled and self.decay_per_ms > 0.0:
                d = self.decay_per_ms * dt_ms * self.table.weight
                d[~self.plastic] = 0.0
                dw -= d
                rec.decay_sum += float(d.sum())
            before = self.table.weight.copy()
            self.table.weight += dw
            lo = int(np.count_nonzero(self.table.weight < self.w_min))
            hi = int(np.count_nonzero(self.table.weight > self.w_max))
            np.clip(self.table.weight, self.w_min, self.w_max, out=self.table.weight)
            delta = self.table.weight - before
            rec.clipped_low, rec.clipped_high = lo, hi
            rec.n_synapses_changed = int(np.count_nonzero(delta))
            rec.weight_delta_sum = float(delta.sum())
            rec.weight_delta_abs_sum = float(np.abs(delta).sum())
            self.total_weight_delta += rec.weight_delta_abs_sum

        # 흔적 증가 (갱신 뒤에 반영)
        if spiked_ids.size:
            if self.simultaneous_policy != "pre_first":
                a.trace_pre[spiked_ids] += 1.0
            else:
                a.trace_pre[:] = pre_before
            if self.simultaneous_policy != "post_first":
                a.trace_post[spiked_ids] += 1.0
            else:
                a.trace_post[:] = post_before

        # 활동률 추정과 임계 적응 (교사 항과 독립인 항목)
        spike_ind = np.zeros(a.n, dtype=np.float64)
        if spiked_ids.size:
            spike_ind[spiked_ids] = 1.0
        inst_hz = spike_ind / (dt_ms / MS_PER_S)
        alpha = dt_ms / self.window_ms
        a.rate_estimate_hz += alpha * (inst_hz - a.rate_estimate_hz)

        if self.theta_enabled:
            target = a.target_rate_hz if self.per_cell_type_target else float(
                np.mean(a.target_rate_hz))
            before_t = a.threshold.copy()
            a.threshold += self.eta_theta * (a.rate_estimate_hz - target) * alpha
            n_clip = int(np.count_nonzero(
                (a.threshold < self.theta_lo) | (a.threshold > self.theta_hi)))
            np.clip(a.threshold, self.theta_lo, self.theta_hi, out=a.threshold)
            if self.neighbor_enabled and self.neighbor_strength > 0.0:
                self._apply_neighbor_averaging()
            dtheta = a.threshold - before_t
            rec.theta_delta_sum = float(dtheta.sum())
            rec.theta_delta_abs_sum = float(np.abs(dtheta).sum())
            rec.n_theta_clipped = n_clip
            self.total_theta_delta += rec.theta_delta_abs_sum

        rec.teacher_derived = False   # 이 모듈의 항은 모두 교사와 무관하다
        self.records.append(rec)
        return rec

    # ------------------------------------------------------------------
    def _incoming_plastic(self, dst_ids: np.ndarray) -> np.ndarray:
        t = self.table
        if dst_ids.size == 0:
            return np.zeros(0, dtype=np.int64)
        parts = [t.in_syn[t.in_ptr[j]:t.in_ptr[j + 1]] for j in dst_ids]
        syn = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        return syn[self.plastic[syn]] if syn.size else syn

    def _outgoing_plastic(self, src_ids: np.ndarray) -> np.ndarray:
        t = self.table
        if src_ids.size == 0:
            return np.zeros(0, dtype=np.int64)
        parts = [t.out_syn[t.out_ptr[i]:t.out_ptr[i + 1]] for i in src_ids]
        syn = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        return syn[self.plastic[syn]] if syn.size else syn

    def _apply_neighbor_averaging(self) -> None:
        """소거 실험 전용: 이웃 임계값을 일부 평균화한다.

        기본값은 꺼져 있다. 이것을 "정답 교정"이라고 부르지 않는다.
        """
        from scipy.spatial import cKDTree

        a = self.a
        if self._neighbor_index is None:
            tree = cKDTree(a.position_mm)
            self._neighbor_index = tree.query_ball_tree(tree, r=self.neighbor_radius_mm)
        means = np.array([
            float(a.threshold[idx].mean()) if idx else float(a.threshold[i])
            for i, idx in enumerate(self._neighbor_index)
        ])
        a.threshold += self.neighbor_strength * (means - a.threshold)

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "rule": "stdp_homeostasis",
            "n_plastic_synapses": int(self.plastic_idx.size),
            "task_learning_enabled": self.task_enabled,
            "weight_decay_enabled": self.decay_enabled,
            "threshold_adaptation_enabled": self.theta_enabled,
            "neighbor_theta_averaging_enabled": self.neighbor_enabled,
            "total_abs_weight_delta": self.total_weight_delta,
            "total_abs_theta_delta": self.total_theta_delta,
            "n_update_records": len(self.records),
            "simultaneous_policy": self.simultaneous_policy,
            "update_order": self.update_order,
            "apply_to_inhibitory": self.apply_to_inhibitory,
            "teacher_independent_terms_ko": (
                "STDP·감쇠·항상성 항은 교사 신호와 무관하게 동작한다. "
                "0교정 검사에서 이 항들의 변화는 0 이 아닐 수 있으며, "
                "교사 유래 항과 분리해 비교해야 한다."
            ),
        }


class NoPlasticity:
    """``learning.mode = none`` 용 고정 기준. 아무 것도 바꾸지 않는다."""

    def __init__(self) -> None:
        self.records: list[UpdateRecord] = []

    def on_step(self, engine: Any, spiked_ids: np.ndarray, dt_ms: float) -> UpdateRecord:
        """가중치·임계값·흔적을 **바꾸지 않는다**. 활동률 추정만 갱신한다.

        활동률은 보고용 진단값이며 임계 적응에 쓰이지 않는다.
        """
        a = engine.a
        spike_ind = np.zeros(a.n, dtype=np.float64)
        if spiked_ids.size:
            spike_ind[spiked_ids] = 1.0
        inst_hz = spike_ind / (dt_ms / MS_PER_S)
        alpha = dt_ms / max(float(engine.cfg["learning"]["homeostasis"]["window_ms"]), dt_ms)
        a.rate_estimate_hz += alpha * (inst_hz - a.rate_estimate_hz)
        return UpdateRecord(step=engine.step_index, time_ms=engine.time_ms)

    def summary(self) -> dict[str, Any]:
        return {"rule": "none", "note_ko": "가중치·임계값을 바꾸지 않는 고정 기준이다."}


def make_plasticity(cfg: dict[str, Any], table: Any, arrays: Any, ids: Any) -> Any:
    """설정의 ``learning.mode`` 에 맞는 가소성 모듈을 만든다.

    ``rao_reference`` 는 **다른 엔진**이므로 여기서 만들지 않는다
    (:mod:`cortex.predictive_coding`). 스파이크 엔진에는 학습 없음을 준다.
    """
    mode = cfg["learning"]["mode"]
    if mode == "stdp_homeostasis":
        return StdpHomeostasis(cfg, table, arrays, ids)
    return NoPlasticity()


# ============================================================================
# 섹션: predictive_coding  —  Rao 계열 연속값 예측 부호화 참조 모델
#   (원래 파일: cortex/predictive_coding.py)
# ============================================================================

"""predictive_coding.py -- Rao & Ballard 계열 연속값 예측 부호화 **참조 모델**.

명세 8절. 이것은 스파이크/전도도 회로와 **다른 엔진**이며, 수학적 기준을
제공하기 위한 것이다. 두 모델을 같은 물리 모델이라고 주장하지 않는다.

목적함수 (여러 모듈이 상위 표현 r2 를 공유하는 구성)::

    E = sum_m 0.5/sigma^2   * ||I_m - U1_m r1_m||^2
      + sum_m 0.5/sigma_td^2* ||r1_m - U2_m r2||^2
      + 0.5*alpha * ( sum_m ||r1_m||^2 + ||r2||^2 )
      + 0.5*lambda* ( sum_m ||U1_m||^2 + sum_m ||U2_m||^2 )

여기서 유도되는 갱신 (모두 같은 전체 목적함수에서 나온다)::

    dr1_m = U1_m.T @ (I_m - U1_m r1_m)/sigma^2 + (U2_m r2 - r1_m)/sigma_td^2 - alpha*r1_m
    dr2   = sum_m U2_m.T @ (r1_m - U2_m r2)/sigma_td^2 - alpha*r2
    dU1_m = outer(I_m - U1_m r1_m, r1_m)/sigma^2      - lambda*U1_m
    dU2_m = outer(r1_m - U2_m r2,  r2)/sigma_td^2     - lambda*U2_m

모듈이 1개면 명세에 적힌 식과 정확히 같다 (``r_td = U2 r2``).

**정직한 표기**

* 이것은 국소 계산으로 표현한 **기울기하강 모델**이며 ``U`` 의 전치를 쓴다.
  "미분도 대칭 가중치도 없는 학습"이 아니다.
* 상위 예측 ``r_td`` 는 완벽한 정답이 아니다. 영역 간 표현 오차와 영상 재구성
  오차를 **다른 변수**로 보관한다 (:class:`RaoState.errors`).
* 정착(settling) 중에는 ``U`` 를 동결하고, 모든 모듈의 **이전 상태**로 오차를
  계산한 뒤 활동을 동기적으로 갱신한다.
* 두 분산 계수는 양수여야 한다 (생성자에서 검증).
"""

@dataclass
class RaoState:
    """정착 결과와 오차 변수들."""

    r1: np.ndarray                     # (M, n1)
    r2: np.ndarray                     # (n2,)
    energy_trace: list[float] = field(default_factory=list)
    errors: dict[str, Any] = field(default_factory=dict)
    settled_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "r1_shape": list(self.r1.shape), "r2_shape": list(self.r2.shape),
            "energy_trace": [float(x) for x in self.energy_trace],
            "errors": self.errors, "settled_steps": self.settled_steps,
        }


class RaoModel:
    """연속값 예측 부호화 참조 엔진.

    Parameters
    ----------
    n_modules : int
        하위(level-1) 모듈 수. 모두 같은 상위 표현 ``r2`` 를 공유한다.
    input_dim : int
        모듈 하나가 받는 입력 차원.
    n1, n2 : int
        level-1 / level-2 표현 차원.
    sigma, sigma_td : float
        관측/상위예측 분산 계수. **양수여야 한다.**
    alpha, lam : float
        활동/가중치 정규화 계수 (0 이상).
    """

    def __init__(self, n_modules: int, input_dim: int, n1: int, n2: int,
                 sigma: float, sigma_td: float, alpha: float, lam: float,
                 rng: np.random.Generator) -> None:
        if not (sigma > 0.0):
            raise ValueError(f"sigma 는 양수여야 한다: {sigma}")
        if not (sigma_td > 0.0):
            raise ValueError(f"sigma_td 는 양수여야 한다: {sigma_td}")
        if alpha < 0.0 or lam < 0.0:
            raise ValueError("alpha 와 lambda 는 0 이상이어야 한다")
        self.M = int(n_modules)
        self.input_dim = int(input_dim)
        self.n1 = int(n1)
        self.n2 = int(n2)
        self.sigma2 = float(sigma) ** 2
        self.sigma_td2 = float(sigma_td) ** 2
        self.alpha = float(alpha)
        self.lam = float(lam)
        scale = 1.0 / np.sqrt(max(1, self.n1))
        self.U1 = rng.normal(0.0, scale, size=(self.M, self.input_dim, self.n1))
        self.U2 = rng.normal(0.0, 1.0 / np.sqrt(max(1, self.n2)),
                             size=(self.M, self.n1, self.n2))

    # ------------------------------------------------------------------
    def energy(self, I: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> float:
        """전체 목적함수 E 의 값 (스칼라)."""
        I = np.asarray(I, dtype=np.float64)
        rec = I - np.einsum("mij,mj->mi", self.U1, r1)
        td = r1 - np.einsum("mij,j->mi", self.U2, r2)
        return float(
            0.5 / self.sigma2 * np.sum(rec ** 2)
            + 0.5 / self.sigma_td2 * np.sum(td ** 2)
            + 0.5 * self.alpha * (np.sum(r1 ** 2) + np.sum(r2 ** 2))
            + 0.5 * self.lam * (np.sum(self.U1 ** 2) + np.sum(self.U2 ** 2))
        )

    def gradients_r(self, I: np.ndarray, r1: np.ndarray,
                    r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``dr1, dr2`` (= -dE/dr). 모든 오차는 **이전 상태**로 계산한다."""
        rec = np.asarray(I, dtype=np.float64) - np.einsum("mij,mj->mi", self.U1, r1)
        pred = np.einsum("mij,j->mi", self.U2, r2)
        td = r1 - pred
        dr1 = (np.einsum("mij,mi->mj", self.U1, rec) / self.sigma2
               - td / self.sigma_td2 - self.alpha * r1)
        dr2 = (np.einsum("mij,mi->j", self.U2, td) / self.sigma_td2
               - self.alpha * r2)
        return dr1, dr2

    def gradients_U(self, I: np.ndarray, r1: np.ndarray,
                    r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``dU1, dU2`` (= -dE/dU)."""
        rec = np.asarray(I, dtype=np.float64) - np.einsum("mij,mj->mi", self.U1, r1)
        td = r1 - np.einsum("mij,j->mi", self.U2, r2)
        dU1 = np.einsum("mi,mj->mij", rec, r1) / self.sigma2 - self.lam * self.U1
        dU2 = np.einsum("mi,j->mij", td, r2) / self.sigma_td2 - self.lam * self.U2
        return dU1, dU2

    # ------------------------------------------------------------------
    def settle(self, I: np.ndarray, *, steps: int, r_step: float,
               r1_init: np.ndarray | None = None, r2_init: np.ndarray | None = None,
               freeze_U: bool = True,
               on_step: Callable[[int, float], None] | None = None) -> RaoState:
        """활동을 정착시킨다. ``freeze_U=True`` 면 U 를 동결한다.

        갱신은 **동기적**이다: 모든 모듈의 이전 상태로 오차를 계산한 뒤 한 번에
        r1, r2 를 갱신한다.
        """
        I = np.asarray(I, dtype=np.float64)
        if I.shape != (self.M, self.input_dim):
            raise ValueError(f"입력 shape 는 ({self.M},{self.input_dim}) 여야 한다: {I.shape}")
        r1 = np.zeros((self.M, self.n1)) if r1_init is None else np.array(r1_init, float)
        r2 = np.zeros(self.n2) if r2_init is None else np.array(r2_init, float)
        if not freeze_U:
            raise ValueError(
                "정착 중 U 는 동결해야 한다 (명세 8절). U 갱신은 learn_step 에서 한다."
            )
        trace: list[float] = [self.energy(I, r1, r2)]
        for s in range(int(steps)):
            dr1, dr2 = self.gradients_r(I, r1, r2)
            r1 = r1 + float(r_step) * dr1
            r2 = r2 + float(r_step) * dr2
            e = self.energy(I, r1, r2)
            trace.append(e)
            if on_step is not None:
                on_step(s, e)

        rec = I - np.einsum("mij,mj->mi", self.U1, r1)
        td = r1 - np.einsum("mij,j->mi", self.U2, r2)
        return RaoState(
            r1=r1, r2=r2, energy_trace=trace, settled_steps=int(steps),
            errors={
                # 영상 재구성 오차와 영역 간 표현 오차를 **다른 변수**로 둔다
                "image_reconstruction_error_l2": float(np.linalg.norm(rec)),
                "image_reconstruction_error_per_module":
                    [float(np.linalg.norm(rec[m])) for m in range(self.M)],
                "interarea_representation_error_l2": float(np.linalg.norm(td)),
                "interarea_representation_error_per_module":
                    [float(np.linalg.norm(td[m])) for m in range(self.M)],
                "energy_first": float(trace[0]), "energy_last": float(trace[-1]),
                "energy_decreased": bool(trace[-1] <= trace[0]),
                "note_ko": "상위 예측 r_td 는 완벽한 정답이 아니다.",
            },
        )

    def learn_step(self, I: np.ndarray, state: RaoState, u_step: float) -> dict[str, Any]:
        """정착된 활동에서 U 를 한 스텝 갱신한다 (부작용: U 변경)."""
        dU1, dU2 = self.gradients_U(I, state.r1, state.r2)
        before = float(np.sum(self.U1 ** 2) + np.sum(self.U2 ** 2))
        self.U1 += float(u_step) * dU1
        self.U2 += float(u_step) * dU2
        return {
            "u_step": float(u_step),
            "dU1_abs_sum": float(np.abs(dU1).sum()),
            "dU2_abs_sum": float(np.abs(dU2).sum()),
            "U_sq_before": before,
            "U_sq_after": float(np.sum(self.U1 ** 2) + np.sum(self.U2 ** 2)),
        }

    # ------------------------------------------------------------------
    def finite_difference_check(self, I: np.ndarray, r1: np.ndarray, r2: np.ndarray,
                                eps: float = 1e-6, n_probe: int = 8,
                                rng: np.random.Generator | None = None
                                ) -> dict[str, Any]:
        """해석적 갱신 방향과 유한차분 기울기를 비교한다 (검증 12번).

        ``dr = -dE/dr`` 이므로 ``dr + numeric_grad ~ 0`` 이어야 한다.
        스파이크 모델의 참 기울기를 유한차분으로 검증했다고 주장하지 않는다.
        이 검사는 **연속값 Rao 모델에만** 해당한다.
        """
        rng = rng or np.random.default_rng(0)
        I = np.asarray(I, dtype=np.float64)
        dr1, dr2 = self.gradients_r(I, r1, r2)
        dU1, dU2 = self.gradients_U(I, r1, r2)

        def probe(shape: tuple[int, ...], n: int) -> list[tuple[int, ...]]:
            total = int(np.prod(shape))
            n = min(n, total)
            flat = rng.choice(total, size=n, replace=False)
            return [tuple(np.unravel_index(int(f), shape)) for f in flat]

        results: dict[str, list[dict[str, float]]] = {"r1": [], "r2": [], "U1": [], "U2": []}
        for idx in probe(r1.shape, n_probe):
            rp, rm = r1.copy(), r1.copy()
            rp[idx] += eps
            rm[idx] -= eps
            num = (self.energy(I, rp, r2) - self.energy(I, rm, r2)) / (2 * eps)
            results["r1"].append({"analytic_minus_grad": float(dr1[idx]),
                                  "numeric_grad": float(num),
                                  "residual": float(dr1[idx] + num)})
        for idx in probe(r2.shape, n_probe):
            rp, rm = r2.copy(), r2.copy()
            rp[idx] += eps
            rm[idx] -= eps
            num = (self.energy(I, r1, rp) - self.energy(I, r1, rm)) / (2 * eps)
            results["r2"].append({"analytic_minus_grad": float(dr2[idx]),
                                  "numeric_grad": float(num),
                                  "residual": float(dr2[idx] + num)})
        for name, arr, grad in (("U1", self.U1, dU1), ("U2", self.U2, dU2)):
            for idx in probe(arr.shape, n_probe):
                orig = arr[idx]
                arr[idx] = orig + eps
                ep = self.energy(I, r1, r2)
                arr[idx] = orig - eps
                em = self.energy(I, r1, r2)
                arr[idx] = orig
                num = (ep - em) / (2 * eps)
                results[name].append({"analytic_minus_grad": float(grad[idx]),
                                      "numeric_grad": float(num),
                                      "residual": float(grad[idx] + num)})

        max_res = max((abs(x["residual"]) for vals in results.values() for x in vals),
                      default=0.0)
        scale = max((abs(x["numeric_grad"]) for vals in results.values() for x in vals),
                    default=1.0)
        return {
            "eps": float(eps), "n_probe_per_tensor": int(n_probe),
            "max_abs_residual": float(max_res),
            "max_abs_numeric_grad": float(scale),
            "max_relative_residual": float(max_res / max(scale, 1e-30)),
            "details": results,
            "scope_note_ko": ("연속값 Rao 모델의 목적함수 미분만 검증한다. "
                              "불연속 스파이크의 참 기울기를 유한차분이나 surrogate 로 "
                              "검증했다고 주장하지 않는다."),
        }

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"U1": self.U1.copy(), "U2": self.U2.copy()}

    def load_state_dict(self, d: dict[str, np.ndarray]) -> None:
        self.U1 = np.asarray(d["U1"], dtype=np.float64)
        self.U2 = np.asarray(d["U2"], dtype=np.float64)

    def summary(self) -> dict[str, Any]:
        return {
            "engine": "rao_reference_continuous",
            "n_modules": self.M, "input_dim": self.input_dim,
            "n1": self.n1, "n2": self.n2,
            "sigma2": self.sigma2, "sigma_td2": self.sigma_td2,
            "alpha": self.alpha, "lambda": self.lam,
            "note_ko": ("전도도 LIF 회로와 같은 물리 모델이 아니다. "
                        "동일 아키텍처 대조군인 것처럼 순위를 매기지 않는다."),
        }


def apical_error_coupling(errors: np.ndarray, cfg: dict[str, Any]
                          ) -> dict[str, Any]:
    """Rao 오차를 apical 전류로 바꾸는 **선택 기능**의 가정 명세.

    기본값은 꺼져 있다. 켜는 경우 아래 가정을 반드시 함께 보고해야 한다.

    * 단위 변환: 무차원 오차 -> pA. 계수 ``gain_pA_per_unit`` 은 **모형 파라미터**.
    * 양/음 오차 부호화: ``split_sign`` 이면 양수부와 음수부를 서로 다른 뉴런
      집단에 넣는다 (전류 부호를 그대로 쓰지 않는다).
    * 좌표 대응: Rao 모듈 인덱스 -> 피질 뉴런 인덱스 사상을 명시해야 한다.
    * 연결 학습: 이 경로의 시냅스가 학습되는지 고정인지 명시해야 한다.

    임의 전류 주입이 Rao 식을 재현한다고 가정하지 않는다.
    """
    c = cfg["learning"]["apical_error_coupling"]
    if not c["enabled"]:
        return {"enabled": False,
                "note_ko": "apical 오차 결합은 꺼져 있다 (기본값)."}
    e = np.asarray(errors, dtype=np.float64)
    gain = float(c["gain_pA_per_unit"])
    if c["split_sign"]:
        pos, neg = np.maximum(e, 0.0) * gain, np.maximum(-e, 0.0) * gain
    else:
        pos, neg = e * gain, np.zeros_like(e)
    return {
        "enabled": True, "gain_pA_per_unit": gain, "split_sign": bool(c["split_sign"]),
        "current_positive_pA": pos, "current_negative_pA": neg,
        "assumptions_ko": [
            "무차원 Rao 오차를 pA 로 바꾸는 계수는 모형 파라미터이며 측정값이 아니다.",
            "양/음 오차를 별도 집단으로 부호화했다 (전류 부호를 그대로 쓰지 않음).",
            "Rao 모듈 <-> 피질 뉴런 좌표 대응은 실험 설정에서 명시해야 한다.",
            "이 전류 주입이 Rao 식을 재현한다고 가정하지 않는다.",
        ],
    }


def make_rao_model(cfg: dict[str, Any], input_dim: int, rng: np.random.Generator) -> RaoModel:
    r = cfg["learning"]["rao"]
    sizes = list(r["level_sizes"])
    n1 = int(sizes[0])
    n2 = int(sizes[1]) if len(sizes) > 1 else int(sizes[0])
    return RaoModel(n_modules=1, input_dim=int(input_dim), n1=n1, n2=n2,
                    sigma=float(r["sigma"]), sigma_td=float(r["sigma_td"]),
                    alpha=float(r["alpha"]), lam=float(r["lambda_u"]), rng=rng)


# ============================================================================
# 섹션: stimuli  —  자극 생성기
#   (원래 파일: cortex/stimuli.py)
# ============================================================================

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


def generate_stimuli(cfg: dict[str, Any], rng: np.random.Generator) -> list[Stimulus]:
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


# ============================================================================
# 섹션: v1_reference  —  고정 Gabor 대조 경로와 위상 불변 에너지
#   (원래 파일: cortex/v1_reference.py)
# ============================================================================

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


# ============================================================================
# 섹션: recording  —  실행 기록기, 저장소 백엔드, 체크포인트
#   (원래 파일: cortex/recording.py)
# ============================================================================

"""recording.py -- 실행 기록기, 저장소 백엔드, 체크포인트 관리.

명세 10절. 사용자가 명령을 실행하면 ``runs/<run_id>/`` 아래에 다음이 생긴다.

==================  ==================================================
파일                내용
==================  ==================================================
``manifest.json``   run_id, UTC 시각, 실행 명령, 해석된 전체 설정, 코드/데이터
                    해시, 라이브러리 버전, 장치/dtype/스레드, 종·모형 가정,
                    단위, 기록 모드, status
``events.h5``       입력·도착·발화·학습 사건 (+ 인덱스)
``states.h5``       선택 뉴런의 soma/basal/apical 상태, 전도도, 임계값, 활동률,
                    층별 집계
``updates.h5``      시냅스/임계 변화, 규칙별 항, 제한·정규화 전후 변화량
``metrics.jsonl``   실제 실행된 스텝/에폭의 측정치와 측정 조건
``summary.csv``     metrics 의 표 형태 요약
``run.log``         진행 로그
``errors.jsonl``    경고/예외/중단 원인/재개 정보
``checkpoints/``    가중치, 상태, 흔적, 이벤트 큐, RNG 상태, 데이터 위치,
                    로그 처리 위치
``report.md``       저장된 결과로만 생성하는 한국어 보고서 (analysis 모듈)
==================  ==================================================

status 는 ``running / completed / interrupted / failed`` 중 하나다.
**이미 ``completed`` 인 실행 디렉터리는 덮어쓰지 않는다.**

h5py 는 필수 의존성이지만 **지연 import** 한다. 설치되어 있지 않으면 한국어로
설치 방법을 알리고, ``recording.backend = "npz"`` 로 바꿀 수 있음을 안내한다.
npz 백엔드는 메모리에 모았다가 종료 시 저장하므로 대규모 실행에는 적합하지 않다
(이 한계를 manifest 에 기록한다).
"""

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_INTERRUPTED = "interrupted"
STATUS_FAILED = "failed"


class RecordingError(RuntimeError):
    pass


def _require_h5py():
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:
        raise RecordingError(
            "h5py 가 설치되어 있지 않다.\n"
            "  설치: python -m pip install -r requirements.txt\n"
            "  또는 설정에서 recording.backend 를 \"npz\" 로 바꿔라 "
            "(npz 는 메모리에 모았다가 종료 시 저장하므로 대규모 실행에는 맞지 않는다)."
        ) from exc
    return h5py


# ----------------------------------------------------------------------
class ArrayStore:
    """추가 가능한 열 저장소의 공통 인터페이스."""

    def append(self, group: str, columns: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def write_once(self, group: str, columns: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class Hdf5Store(ArrayStore):
    """압축 청크 HDF5 저장소."""

    def __init__(self, path: Path, compression: str = "gzip", level: int = 4,
                 chunk_rows: int = 4096) -> None:
        h5py = _require_h5py()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(self.path, "a")
        self.compression = compression
        self.level = int(level)
        self.chunk_rows = int(chunk_rows)

    def append(self, group: str, columns: dict[str, np.ndarray]) -> None:
        g = self._f.require_group(group)
        for name, arr in columns.items():
            a = np.asarray(arr)
            if name not in g:
                maxshape = (None,) + a.shape[1:]
                chunks = (min(self.chunk_rows, max(1, a.shape[0])),) + a.shape[1:]
                g.create_dataset(name, data=a, maxshape=maxshape, chunks=chunks,
                                 compression=self.compression,
                                 compression_opts=self.level)
            else:
                ds = g[name]
                n0 = ds.shape[0]
                ds.resize(n0 + a.shape[0], axis=0)
                ds[n0:] = a

    def write_once(self, group: str, columns: dict[str, np.ndarray]) -> None:
        g = self._f.require_group(group)
        for name, arr in columns.items():
            if name in g:
                del g[name]
            g.create_dataset(name, data=np.asarray(arr),
                             compression=self.compression,
                             compression_opts=self.level)

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:  # pragma: no cover
            pass


class NpzStore(ArrayStore):
    """메모리 누적 후 종료 시 .npz 로 저장하는 대체 백엔드.

    대규모 실행에는 적합하지 않다. 이 한계는 manifest 에 기록된다.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path).with_suffix(".npz")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buf: dict[str, list[np.ndarray]] = {}
        self._once: dict[str, np.ndarray] = {}

    def append(self, group: str, columns: dict[str, np.ndarray]) -> None:
        for name, arr in columns.items():
            self._buf.setdefault(f"{group}/{name}", []).append(np.asarray(arr))

    def write_once(self, group: str, columns: dict[str, np.ndarray]) -> None:
        for name, arr in columns.items():
            self._once[f"{group}/{name}"] = np.asarray(arr)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        out: dict[str, np.ndarray] = dict(self._once)
        for k, parts in self._buf.items():
            if parts:
                out[k] = np.concatenate(parts)
        np.savez_compressed(self.path, **out)


def make_store(path: Path, cfg: dict[str, Any]) -> ArrayStore:
    rec = cfg["recording"]
    if rec["backend"] == "npz":
        return NpzStore(path)
    return Hdf5Store(path, rec["compression"], rec["compression_level"],
                     rec["chunk_rows"])


# ----------------------------------------------------------------------
def code_hash(package_dir: Path) -> dict[str, Any]:
    """코드의 sha256 목록과 전체 해시.

    단일 파일 판에서는 ``package_dir`` 이 **이 파일 자체**이므로 그 한 파일을,
    디렉터리가 주어지면 그 아래 모든 ``*.py`` 를 해싱한다.
    """
    package_dir = Path(package_dir)
    single = package_dir.is_file()
    files = [package_dir] if single else sorted(
        p for p in package_dir.rglob("*.py") if "__pycache__" not in p.parts)
    per: dict[str, str] = {}
    h = hashlib.sha256()
    for p in files:
        d = hashlib.sha256(p.read_bytes()).hexdigest()
        per[p.name if single else str(p.relative_to(package_dir.parent))] = d
        h.update(d.encode())
    return {"combined_sha256": h.hexdigest(), "files": per, "n_files": len(files)}


def git_commit(root: Path) -> str | None:
    """git 커밋 해시. git 이 없거나 저장소가 아니면 None (네트워크 접근 없음)."""
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5, check=False)
        return out.stdout.strip() or None
    except Exception:
        return None


def library_versions() -> dict[str, str]:
    vers: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "matplotlib", "PIL", "h5py", "numba"):
        try:
            mod = __import__(name)
            vers[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            vers[name] = "not_installed"
    return vers


def thread_settings() -> dict[str, str]:
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS")
    return {k: os.environ.get(k, "unset") for k in keys}


def data_hash(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        p = Path(p)
        if p.is_file():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ----------------------------------------------------------------------
@dataclass
class RunPaths:
    root: Path
    manifest: Path
    events: Path
    states: Path
    updates: Path
    metrics: Path
    summary_csv: Path
    log: Path
    errors: Path
    checkpoints: Path
    figures: Path
    report: Path

    @staticmethod
    def make(run_dir: Path) -> "RunPaths":
        r = Path(run_dir)
        return RunPaths(
            root=r, manifest=r / "manifest.json", events=r / "events.h5",
            states=r / "states.h5", updates=r / "updates.h5",
            metrics=r / "metrics.jsonl", summary_csv=r / "summary.csv",
            log=r / "run.log", errors=r / "errors.jsonl",
            checkpoints=r / "checkpoints", figures=r / "figures",
            report=r / "report.md",
        )


class RunRecorder:
    """실행 기록기. 계산 결과를 바꾸지 않는다.

    Parameters
    ----------
    run_dir : Path
    cfg : 해석된 설정
    command : str
        실제로 실행한 명령 문자열.
    package_root : Path
        소스 해시 계산용 패키지 경로.
    """

    def __init__(self, run_dir: Path, cfg: dict[str, Any], command: str,
                 package_root: Path, data_files: list[Path] | None = None,
                 resume: bool = False) -> None:
        self.paths = RunPaths.make(Path(run_dir))
        self.cfg = cfg
        self.command = command
        self.package_root = Path(package_root)
        self.resume = bool(resume)

        if self.paths.manifest.is_file():
            old = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
            if old.get("status") == STATUS_COMPLETED and not resume:
                raise RecordingError(
                    f"이미 완료된 실행 디렉터리다: {self.paths.root}\n"
                    f"  결과를 덮어쓰지 않는다. 다른 --run-dir 를 쓰거나 "
                    f"기존 결과를 다른 곳으로 옮겨라."
                )
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoints.mkdir(parents=True, exist_ok=True)
        self.paths.figures.mkdir(parents=True, exist_ok=True)

        self.run_id = self.paths.root.name
        self.started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._t0 = time.time()
        self.status = STATUS_RUNNING
        self._events_store: ArrayStore | None = None
        self._states_store: ArrayStore | None = None
        self._updates_store: ArrayStore | None = None
        self._metrics_fh = open(self.paths.metrics, "a", encoding="utf-8")
        self._log_fh = open(self.paths.log, "a", encoding="utf-8")
        self._err_fh = open(self.paths.errors, "a", encoding="utf-8")
        self._metric_rows: list[dict[str, Any]] = []
        self._state_rows = 0
        self._selected: np.ndarray | None = None
        self._step_every = int(cfg["recording"]["state_sample_every_steps"])
        self._flush_every = int(cfg["recording"]["flush_every_steps"])
        self._max_state_rows = int(cfg["recording"]["max_state_samples"])
        self.n_state_rows_skipped = 0

        self.manifest: dict[str, Any] = self._build_manifest(data_files or [])
        self._write_manifest()

    # ------------------------------------------------------------------
    def _build_manifest(self, data_files: list[Path]) -> dict[str, Any]:
        cfgm = self.cfg
        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_utc": self.started_utc,
            "finished_utc": None,
            "elapsed_seconds": None,
            "command": self.command,
            "resumed": self.resume,
            "config": cfgm,
            "config_sha256": _sha256_json(cfgm),
            "code": code_hash(self.package_root),
            "git_commit": git_commit(self.package_root.parent),
            "data_hashes": data_hash(data_files),
            "library_versions": library_versions(),
            "platform": {
                "platform": platform.platform(), "machine": platform.machine(),
                "processor": platform.processor(), "python": sys.version,
            },
            "device": "cpu",
            "gpu_used": False,
            "dtypes": {"state": "float64", "weight": "float64",
                       "event_time": "float64", "ids": "int64/int32"},
            "thread_settings": thread_settings(),
            "units": _units_block(),
            "recording": {
                "mode": cfgm["recording"]["mode"],
                "backend": cfgm["recording"]["backend"],
                "selection_criterion": cfgm["recording"]["selection_criterion"],
                "state_sample_every_steps": self._step_every,
                "max_events": cfgm["recording"]["max_events"],
                "max_state_samples": self._max_state_rows,
                "npz_backend_limitation_ko": (
                    "npz 백엔드는 메모리에 모았다가 종료 시 저장한다. 대규모 실행에는 "
                    "적합하지 않다."
                ) if cfgm["recording"]["backend"] == "npz" else None,
            },
            "assumptions": {
                "species_assumption": cfgm["meta"]["species_assumption"],
                "note_ko": (
                    "영역별 뉴런 수, 층 두께, 연결 확률은 출처가 명시되지 않은 경우 "
                    "모형 파라미터다. BIOLOGY_AND_ASSUMPTIONS.md 참조. "
                    "이 결과를 인간 뇌의 완전하고 정확한 복제라고 부르지 않는다."
                ),
            },
            "experiment_status": "not_run",
        }

    def _write_manifest(self) -> None:
        self.paths.manifest.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=1, default=_jsonable),
            encoding="utf-8")

    # ------------------------------------------------------------------
    def set_selected_neurons(self, neuron_ids: np.ndarray) -> None:
        self._selected = np.asarray(neuron_ids, dtype=np.int64)
        self.manifest["recording"]["n_selected_neurons"] = int(self._selected.size)
        self._write_manifest()

    @property
    def events_store(self) -> ArrayStore:
        if self._events_store is None:
            self._events_store = make_store(self.paths.events, self.cfg)
        return self._events_store

    @property
    def states_store(self) -> ArrayStore:
        if self._states_store is None:
            self._states_store = make_store(self.paths.states, self.cfg)
        return self._states_store

    @property
    def updates_store(self) -> ArrayStore:
        if self._updates_store is None:
            self._updates_store = make_store(self.paths.updates, self.cfg)
        return self._updates_store

    # ------------------------------------------------------------------
    def log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        self._log_fh.write(line + "\n")
        self._log_fh.flush()

    def warn(self, message: str, **extra: Any) -> None:
        self._err_fh.write(json.dumps(
            {"level": "warning", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "message": message, **extra}, ensure_ascii=False, default=_jsonable) + "\n")
        self._err_fh.flush()

    def error(self, message: str, exc: BaseException | None = None, **extra: Any) -> None:
        payload = {
            "level": "error",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": message, **extra,
        }
        if exc is not None:
            payload["exception_type"] = type(exc).__name__
            payload["traceback"] = "".join(traceback.format_exception(
                type(exc), exc, exc.__traceback__))
        self._err_fh.write(json.dumps(payload, ensure_ascii=False,
                                      default=_jsonable) + "\n")
        self._err_fh.flush()

    def metric(self, **fields: Any) -> None:
        """metrics.jsonl 에 한 줄 기록한다. **측정 조건도 함께 넣어라.**"""
        row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
        self._metrics_fh.write(json.dumps(row, ensure_ascii=False,
                                          default=_jsonable) + "\n")
        self._metrics_fh.flush()
        self._metric_rows.append(row)

    # ------------------------------------------------------------------
    def record_step(self, engine: Any, report: Any) -> None:
        """엔진이 매 스텝 끝에 호출한다. **계산을 바꾸지 않는다.**"""
        if self.cfg["recording"]["mode"] == "summary":
            return
        if report.step % self._step_every != 0:
            return
        if self._selected is None or self._selected.size == 0:
            return
        if self._state_rows >= self._max_state_rows:
            self.n_state_rows_skipped += int(self._selected.size)
            return
        a = engine.a
        sel = self._selected
        n = sel.size
        self.states_store.append("states", {
            "step": np.full(n, report.step, dtype=np.int64),
            "time_ms": np.full(n, report.time_ms, dtype=np.float64),
            "sample_id": np.full(n, engine.sample_id, dtype=np.int32),
            "neuron_id": sel.astype(np.int32),
            "V_soma_mV": a.V_mV[sel, 0].copy(),
            "V_basal_mV": a.V_mV[sel, 1].copy(),
            "V_apical_mV": a.V_mV[sel, 2].copy(),
            "g_total_nS": a.g_nS[sel].sum(axis=(1, 2)),
            "threshold": a.threshold[sel].copy(),
            "rate_hz": a.rate_estimate_hz[sel].copy(),
            "fired": a.fired[sel].astype(np.int8),
            "interval_activation": a.interval_activation[sel].copy(),
        })
        self._state_rows += int(n)
        if report.step % self._flush_every == 0:
            self.flush()

    def record_updates(self, records: list[Any]) -> None:
        """가소성 갱신 기록을 updates 저장소에 넣는다."""
        if not records:
            return
        cols: dict[str, list[Any]] = {}
        for r in records:
            d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
            for k, v in d.items():
                if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
                    cols.setdefault(k, []).append(v)
        if cols:
            self.updates_store.append(
                "updates", {k: np.asarray(v) for k, v in cols.items()})

    def record_events(self, event_log: Any) -> None:
        """이벤트 로그 전체를 저장소에 쓴다 (실행 종료 또는 표본 경계에서)."""
        cols = event_log.all_columns()
        if not cols or len(next(iter(cols.values()))) == 0:
            return
        self.events_store.write_once("events", cols)
        self.events_store.write_once("schema", {
            "columns": np.array(event_log.COLUMNS, dtype="S64"),
        })

    def record_layer_aggregate(self, name: str, columns: dict[str, np.ndarray]) -> None:
        self.states_store.write_once(f"aggregate/{name}", columns)

    # ------------------------------------------------------------------
    def flush(self) -> None:
        for s in (self._events_store, self._states_store, self._updates_store):
            if s is not None:
                s.flush()
        self._metrics_fh.flush()
        self._log_fh.flush()
        self._err_fh.flush()

    def write_summary_csv(self) -> None:
        if not self._metric_rows:
            return
        keys: list[str] = []
        for r in self._metric_rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(self.paths.summary_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for r in self._metric_rows:
                wr.writerow({k: _flat(r.get(k)) for k in keys})

    def finish(self, status: str, note: str = "") -> None:
        """실행을 마치고 manifest 를 갱신한다 (예외 경로에서도 호출할 것)."""
        self.status = status
        self.manifest["status"] = status
        self.manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.manifest["elapsed_seconds"] = round(time.time() - self._t0, 3)
        self.manifest["finish_note"] = note
        self.manifest["n_state_rows_written"] = self._state_rows
        self.manifest["n_state_rows_skipped_by_limit"] = self.n_state_rows_skipped
        self.manifest["experiment_status"] = (
            "completed" if status == STATUS_COMPLETED else status)
        self.write_summary_csv()
        self._write_manifest()
        self.flush()
        for s in (self._events_store, self._states_store, self._updates_store):
            if s is not None:
                s.close()
        for fh in (self._metrics_fh, self._log_fh, self._err_fh):
            try:
                fh.close()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            if self.status == STATUS_RUNNING:
                self.finish(STATUS_COMPLETED)
            return False
        if issubclass(exc_type, KeyboardInterrupt):
            self.error("사용자 중단 (KeyboardInterrupt)", exc)
            self.finish(STATUS_INTERRUPTED, "KeyboardInterrupt")
        else:
            self.error("실행 중 예외 발생", exc)
            self.finish(STATUS_FAILED, f"{exc_type.__name__}: {exc}")
        return False


# ----------------------------------------------------------------------
@dataclass
class Checkpoint:
    """체크포인트 내용 (재개에 필요한 모든 것)."""

    step_index: int
    sample_index: int
    payload: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """체크포인트 저장/복원. 재개 시 호환성을 검사한다."""

    def __init__(self, recorder: RunRecorder, keep_last: int = 3) -> None:
        self.recorder = recorder
        self.dir = recorder.paths.checkpoints
        self.keep_last = int(keep_last)

    def save(self, name: str, *, engine_state: dict[str, Any],
             rng_state: dict[str, Any], sample_index: int,
             event_log_position: int, data_locator: dict[str, Any],
             extra: dict[str, Any] | None = None) -> Path:
        """체크포인트 1개를 저장한다 (npz + json 메타)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        base = self.dir / name
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, Any] = {
            "name": name, "sample_index": int(sample_index),
            "event_log_position": int(event_log_position),
            "data_locator": data_locator,
            "config_sha256": self.recorder.manifest["config_sha256"],
            "code_sha256": self.recorder.manifest["code"]["combined_sha256"],
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "extra": extra or {},
            "rng_state": rng_state,
        }
        for k, v in engine_state.items():
            if isinstance(v, np.ndarray):
                arrays[f"engine/{k}"] = v
            elif k == "queue":
                meta["queue_steps"] = [int(s) for s in v["steps"]]
                for i, payload in enumerate(v["payloads"]):
                    for pk, pv in payload.items():
                        arrays[f"queue/{i}/{pk}"] = np.asarray(pv)
            else:
                meta[f"engine_{k}"] = v
        np.savez_compressed(base.with_suffix(".npz"), **arrays)
        base.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1, default=_jsonable),
            encoding="utf-8")
        self._prune()
        return base.with_suffix(".npz")

    def _prune(self) -> None:
        cks = sorted(self.dir.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        for p in cks[:-self.keep_last] if len(cks) > self.keep_last else []:
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)

    @staticmethod
    def list_checkpoints(run_dir: Path) -> list[Path]:
        d = Path(run_dir) / "checkpoints"
        return sorted(d.glob("*.npz")) if d.is_dir() else []

    @staticmethod
    def load(path: Path, *, expected_config_sha: str | None = None,
             expected_code_sha: str | None = None,
             strict: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
        """체크포인트를 읽고 코드/설정 호환성을 검사한다.

        Returns
        -------
        (engine_state, meta)
        """
        p = Path(path)
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        problems: list[str] = []
        if expected_config_sha and meta["config_sha256"] != expected_config_sha:
            problems.append("설정 해시가 다르다 (config_sha256)")
        if expected_code_sha and meta["code_sha256"] != expected_code_sha:
            problems.append("코드 해시가 다르다 (code_sha256)")
        if problems and strict:
            raise RecordingError(
                "체크포인트 호환성 검사 실패: " + "; ".join(problems) +
                "\n  같은 코드/설정으로 재개하거나, 차이를 인지하고 --allow-mismatch 를 써라."
            )
        data = np.load(p, allow_pickle=False)
        engine_state: dict[str, Any] = {}
        queue: dict[str, Any] = {"steps": meta.get("queue_steps", []), "payloads": []}
        qpay: dict[int, dict[str, np.ndarray]] = {}
        for key in data.files:
            if key.startswith("engine/"):
                engine_state[key.split("/", 1)[1]] = data[key]
            elif key.startswith("queue/"):
                _, i, pk = key.split("/", 2)
                qpay.setdefault(int(i), {})[pk] = data[key]
        queue["payloads"] = [qpay[i] for i in sorted(qpay)]
        engine_state["queue"] = queue
        for k, v in meta.items():
            if k.startswith("engine_"):
                engine_state[k[len("engine_"):]] = v
        return engine_state, meta


# ----------------------------------------------------------------------
def _sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=_jsonable)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _flat(v: Any) -> Any:
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False, default=_jsonable)
    return v


def _units_block() -> dict[str, str]:
    return dict(UNIT_TABLE)


def load_manifest(run_dir: Path) -> dict[str, Any]:
    p = Path(run_dir) / "manifest.json"
    if not p.is_file():
        raise RecordingError(
            f"실행 기록을 찾을 수 없다: {p}\n"
            f"  --run-dir 경로를 확인하라. 조회 명령은 새 실험을 시작하지 않는다."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    p = Path(run_dir) / "metrics.jsonl"
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ============================================================================
# 섹션: runner  —  모델 조립과 실험 실행기
#   (원래 파일: cortex/runner.py)
# ============================================================================

"""runner.py -- 모델 조립과 실험 실행기.

명세 10, 11, 14절. **메뉴(run.py)와 CLI 는 모두 이 모듈을 호출한다.**
시뮬레이션 구현을 두 벌 만들지 않는다.

``--execute`` 없이 호출하면 아무 것도 실행하지 않고 규모·설정만 계산해서
돌려준다 (dry-run). 실제 런타임은 측정 전에 확정하지 않는다.
"""

@dataclass
class Model:
    """조립된 모델 묶음."""

    cfg: dict[str, Any]
    anat: Anatomy
    table: SynapseTable
    event_log: EventLog
    engine: Engine
    plasticity: Any
    encoder: RetinaEncoder
    sampler: LogPolarSampler
    driver: InputDriver
    wiring_report: WiringReport
    rngs: RngStreams
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return len(self.anat.population)


def build_model(cfg: dict[str, Any], rngs: RngStreams) -> Model:
    """설정에서 해부 구조·배선·엔진을 조립한다 (시뮬레이션은 하지 않는다)."""
    anat = build_anatomy(cfg, rngs.get("wiring"))
    table, wreport = build_wiring(cfg, anat, rngs.get("wiring"), rngs.get("weights"))
    rec = cfg["recording"]
    log = EventLog(
        n_neurons=len(anat.population), mode=rec["mode"],
        selected_neurons=rec["selected_neurons"] or None,
        selection_criterion=rec["selection_criterion"],
        max_events=int(rec["max_events"]),
    )
    anat.population.attach(event_log=log)
    plast = make_plasticity(cfg, table, anat.population.arrays, anat.ids)
    engine = Engine(cfg, anat, table, log, plasticity=plast)
    encoder = RetinaEncoder(cfg)
    sampler = LogPolarSampler(anat.grid, cfg)
    driver = InputDriver(cfg)
    return Model(cfg=cfg, anat=anat, table=table, event_log=log, engine=engine,
                 plasticity=plast, encoder=encoder, sampler=sampler, driver=driver,
                 wiring_report=wreport, rngs=rngs,
                 meta={"config_sha256": config_hash(cfg)})


def select_recording_neurons(cfg: dict[str, Any], model: Model) -> np.ndarray:
    """상태를 기록할 뉴런 선택.

    ``full`` 모드에서도 상태 저장은 용량이 크므로, 영역마다 고르게 뽑되
    **선택 기준을 manifest 에 남긴다** (조용히 버리지 않는다).
    """
    rec = cfg["recording"]
    if rec["selected_neurons"]:
        return np.asarray(rec["selected_neurons"], dtype=np.int64)
    a = model.anat.population.arrays
    ids = model.anat.ids
    picks: list[np.ndarray] = []
    per_area = max(1, int(rec["max_state_samples"] //
                          max(1, len(ids.areas) * 200)))
    wanted = rec["selected_areas"] or ids.areas.names()
    for aname in wanted:
        m = np.nonzero(a.area_id == ids.areas.id_of(aname))[0]
        if m.size == 0:
            continue
        step = max(1, m.size // per_area)
        picks.append(m[::step][:per_area])
    return np.concatenate(picks) if picks else np.arange(min(32, a.n), dtype=np.int64)


# ----------------------------------------------------------------------
@dataclass
class SampleResult:
    sample_index: int
    stimulus_id: str
    label: str
    n_spikes_total: int
    spikes_by_area: dict[str, int]
    mean_rate_hz_by_area: dict[str, float]
    first_spike_step_by_area: dict[str, int]
    silent_fraction_by_area: dict[str, float]
    n_steps: int
    n_events: int
    settle: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


class ExperimentRunner:
    """실험 실행기.

    ``execute=False`` (기본) 면 규모 추정만 하고 **아무 것도 실행하지 않는다**.
    """

    def __init__(self, cfg: dict[str, Any], run_dir: Path | None, command: str,
                 package_root: Path, execute: bool = False,
                 progress: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir) if run_dir else None
        self.command = command
        self.package_root = Path(package_root)
        self.execute = bool(execute)
        self.progress = progress or (lambda msg: None)

    # ------------------------------------------------------------------
    def inspect(self) -> dict[str, Any]:
        """설정과 규모만 계산한다. 시뮬레이션·데이터 생성 없음."""
        est: SizeEstimate = estimate_sizes(self.cfg)
        lim = self.cfg["experiment"]["limits"]
        warnings: list[str] = []
        if est.disk_mb_estimated > lim["max_disk_mb"]:
            warnings.append(
                f"추정 디스크 사용량 {est.disk_mb_estimated:.0f} MB 가 한도 "
                f"{lim['max_disk_mb']} MB 를 넘는다. 실행은 한도에서 중단된다.")
        if est.ram_mb_estimated > lim["max_ram_mb"]:
            warnings.append(
                f"추정 RAM {est.ram_mb_estimated:.0f} MB 가 한도 "
                f"{lim['max_ram_mb']} MB 를 넘는다.")
        return {
            "config_name": self.cfg["meta"]["name"],
            "config_sha256": config_hash(self.cfg),
            "engine_mode": self.cfg["engine"]["mode"],
            "learning_mode": self.cfg["learning"]["mode"],
            "recording_mode": self.cfg["recording"]["mode"],
            "estimate": est.to_dict(),
            "warnings": warnings,
            "experiment_status": "not_run",
            "note_ko": ("이 출력은 실행 전 추정이다. 실제 런타임은 측정하기 전에 "
                        "확정하지 않는다. 실행하려면 --execute 를 붙여라."),
        }

    # ------------------------------------------------------------------
    def _prepare(self, recorder: RunRecorder) -> Model:
        rngs = rng_from_config(self.cfg)
        t0 = time.time()
        self.progress("모델을 조립하는 중...")
        model = build_model(self.cfg, rngs)
        recorder.log(f"모델 조립 완료: 뉴런 {model.n_neurons}, "
                     f"시냅스 {model.table.n_synapses} ({time.time() - t0:.1f}s)")
        recorder.manifest["model"] = {
            "n_neurons": model.n_neurons,
            "n_synapses": model.table.n_synapses,
            "wiring": model.wiring_report.to_dict(),
            "anatomy": model.anat.meta,
            "synapse_summary": model.table.summary(),
        }
        recorder.set_selected_neurons(select_recording_neurons(self.cfg, model))
        recorder.manifest["recording"]["selection_criterion"] = (
            self.cfg["recording"]["selection_criterion"]
            or "영역마다 균등 간격으로 뽑은 표본 (recording.selected_neurons 미지정)")
        recorder._write_manifest()
        model.engine.recorder = recorder
        return model

    def _run_sample(self, model: Model, recorder: RunRecorder, stim: Any,
                    sample_index: int, learn: bool) -> SampleResult:
        cfg = self.cfg
        a = model.anat.population.arrays
        ids = model.anat.ids
        rs = cfg["engine"]["reset_between_samples"]
        if not cfg["engine"]["sequence_mode"]:
            model.engine.reset_transient(
                voltages=rs["voltages"], conductances=rs["conductances"],
                event_queue=rs["event_queue"], traces=rs["traces"])
        model.engine.sample_id = sample_index

        n_steps = int(round(cfg["engine"]["duration_ms"] / cfg["engine"]["dt_ms"]))
        frames = stim.frames
        steps_per_frame = max(1, n_steps // len(frames))

        spikes_before = a.spike_count.copy()
        first_spike_step: dict[str, int] = {}
        saved_plast = model.engine.plasticity
        if not learn:
            model.engine.plasticity = None
        try:
            for fi, frame in enumerate(frames):
                out = model.encoder.encode(frame)
                values, smeta = model.sampler.sample(out.channels)
                values = model.encoder.normalize(values)
                rates = model.driver.rates_hz(values)          # (C, S)
                nid = model.anat.retina_neuron_id
                mask = nid >= 0
                drive = rates_to_drive(
                    cfg, nid[mask].ravel(), rates[mask].ravel(), steps_per_frame,
                    model.rngs.get("input_noise", sample_index))
                model.engine.set_external_drive(drive)
                if drive.n_multi_event_truncated:
                    recorder.warn(
                        "Poisson 모드에서 한 스텝에 2건 이상 발생한 사건을 1 스파이크로 "
                        "잘랐다. dt 를 줄이거나 최대 발화율을 낮춰라.",
                        n_truncated=int(drive.n_multi_event_truncated),
                        sample_index=sample_index, frame=fi)
                for _ in range(steps_per_frame):
                    rep = model.engine.step()
                    if rep.n_spikes:
                        fired = np.nonzero(a.fired)[0]
                        for aid in np.unique(a.area_id[fired]):
                            an = ids.areas.name_of(int(aid))
                            first_spike_step.setdefault(an, rep.step)
        finally:
            model.engine.plasticity = saved_plast

        delta = a.spike_count - spikes_before
        by_area: dict[str, int] = {}
        rate_by_area: dict[str, float] = {}
        silent_by_area: dict[str, float] = {}
        dur_s = cfg["engine"]["duration_ms"] / 1000.0
        for aname in ids.areas.names():
            m = a.area_id == ids.areas.id_of(aname)
            if not m.any():
                continue
            by_area[aname] = int(delta[m].sum())
            rate_by_area[aname] = float(delta[m].mean() / max(dur_s, 1e-9))
            silent_by_area[aname] = float(np.count_nonzero(delta[m] == 0) / m.sum())

        if model.plasticity is not None and getattr(model.plasticity, "records", None):
            recorder.record_updates(model.plasticity.records)
            model.plasticity.records = []

        return SampleResult(
            sample_index=sample_index, stimulus_id=stim.stimulus_id, label=stim.label,
            n_spikes_total=int(delta.sum()), spikes_by_area=by_area,
            mean_rate_hz_by_area=rate_by_area,
            first_spike_step_by_area=first_spike_step,
            silent_fraction_by_area=silent_by_area,
            n_steps=n_steps, n_events=model.event_log.n_events,
        )

    # ------------------------------------------------------------------
    def simulate(self) -> dict[str, Any]:
        """자극을 순서대로 제시하고 기록한다 (학습은 설정에 따름)."""
        if not self.execute:
            return self.inspect()
        if self.run_dir is None:
            raise ValueError("run_dir 가 필요하다")
        with RunRecorder(self.run_dir, self.cfg, self.command,
                         self.package_root) as recorder:
            model = self._prepare(recorder)
            ckpt = CheckpointManager(recorder, self.cfg["checkpoint"]["keep_last"])
            rng_stim = model.rngs.get("stimulus")
            stims = generate_stimuli(self.cfg, rng_stim)
            if not stims:
                raise ValueError("experiment.stimuli 가 비어 있어 제시할 자극이 없다")
            recorder.log(f"자극 {len(stims)}개 생성")

            fit_imgs = [s.frames[0] for s in stims[:min(len(stims), 16)]]
            norm = model.encoder.fit_normalization(fit_imgs, source="simulate_prefix")
            recorder.manifest["input_normalization"] = norm
            recorder.manifest["input_normalization"]["note_ko"] = (
                "simulate 명령은 제시 자극 앞부분으로 정규화 계수를 추정한다. "
                "train/dev/test 분할 실험에서는 train 만 사용한다 (experiment 명령).")
            recorder._write_manifest()

            learn = self.cfg["learning"]["mode"] == "stdp_homeostasis"
            results: list[SampleResult] = []
            try:
                for i, st in enumerate(stims):
                    self.progress(f"자극 {i + 1}/{len(stims)} ({st.stimulus_id}) — "
                                  f"중단하려면 Ctrl+C")
                    r = self._run_sample(model, recorder, st, i, learn)
                    results.append(r)
                    recorder.metric(kind="sample", learning=self.cfg["learning"]["mode"],
                                    engine_mode=self.cfg["engine"]["mode"], **r.to_dict())
                    if (self.cfg["checkpoint"]["enabled"]
                            and (i + 1) % max(1, self.cfg["checkpoint"]["every_samples"]) == 0):
                        ckpt.save(f"sample_{i:05d}",
                                  engine_state=model.engine.state_dict(),
                                  rng_state=model.rngs.state_dict(), sample_index=i,
                                  event_log_position=model.event_log.n_events,
                                  data_locator={"stimulus_id": st.stimulus_id,
                                                "stimulus_index": i})
            except CapacityExceeded as exc:
                recorder.error("용량 한도 초과로 중단", exc)
                ckpt.save("capacity_stop", engine_state=model.engine.state_dict(),
                          rng_state=model.rngs.state_dict(),
                          sample_index=len(results),
                          event_log_position=model.event_log.n_events,
                          data_locator={"reason": "capacity"})
                recorder.record_events(model.event_log)
                recorder.finish(STATUS_INTERRUPTED, str(exc))
                return {"status": STATUS_INTERRUPTED, "reason": str(exc),
                        "run_dir": str(self.run_dir)}

            recorder.record_events(model.event_log)
            self._write_aggregates(recorder, model)
            recorder.manifest["event_log_schema"] = model.event_log.schema()
            recorder.manifest["plasticity_summary"] = model.plasticity.summary()
            recorder._write_manifest()
            return {"status": STATUS_COMPLETED, "run_dir": str(self.run_dir),
                    "n_samples": len(results),
                    "n_events": model.event_log.n_events}

    # ------------------------------------------------------------------
    def experiment(self) -> dict[str, Any]:
        """train/dev/test 분할 실험. 정규화는 train 만 사용한다."""
        if not self.execute:
            info = self.inspect()
            info["protocol"] = self.cfg["experiment"]["protocol"]
            return info
        if self.run_dir is None:
            raise ValueError("run_dir 가 필요하다")
        with RunRecorder(self.run_dir, self.cfg, self.command,
                         self.package_root) as recorder:
            model = self._prepare(recorder)
            ckpt = CheckpointManager(recorder, self.cfg["checkpoint"]["keep_last"])
            stims = generate_stimuli(self.cfg, model.rngs.get("stimulus"))
            splits = split_stimuli(stims, self.cfg, model.rngs.get("split"))
            report = split_report(splits)
            recorder.manifest["splits"] = report
            if not report["no_overlap"]:
                raise ValueError(f"분할이 겹친다: {report['overlaps']}")

            norm = model.encoder.fit_normalization(
                [s.frames[0] for s in splits["train"]], source="train")
            recorder.manifest["input_normalization"] = norm
            recorder.manifest["test_access"] = {
                "n_test_evaluations": 0,
                "note_ko": ("test 는 선택 완료 후 조건·시드당 한 번만 평가한다. "
                            "파일럿 중 test 생성/평가 횟수를 여기에 기록한다."),
            }
            recorder._write_manifest()

            learn_splits = {"train": self.cfg["learning"]["mode"] == "stdp_homeostasis",
                            "dev": False, "test": False}
            features: dict[str, list[np.ndarray]] = {}
            labels: dict[str, list[str]] = {}
            idx = 0
            for split in ("train", "dev", "test"):
                feats: list[np.ndarray] = []
                labs: list[str] = []
                for j, st in enumerate(splits[split]):
                    self.progress(f"[{split}] {j + 1}/{len(splits[split])} "
                                  f"({st.stimulus_id}) — 중단하려면 Ctrl+C")
                    before = model.anat.population.arrays.spike_count.copy()
                    r = self._run_sample(model, recorder, st, idx, learn_splits[split])
                    idx += 1
                    recorder.metric(kind="sample", split=split, **r.to_dict())
                    feats.append(self._readout_features(model, before))
                    labs.append(st.label)
                    if (self.cfg["checkpoint"]["enabled"]
                            and (j + 1) % max(1, self.cfg["checkpoint"]["every_samples"]) == 0):
                        ckpt.save(f"{split}_{j:05d}",
                                  engine_state=model.engine.state_dict(),
                                  rng_state=model.rngs.state_dict(), sample_index=idx,
                                  event_log_position=model.event_log.n_events,
                                  data_locator={"split": split, "index": j})
                features[split] = feats
                labels[split] = labs
                if split == "test":
                    recorder.manifest["test_access"]["n_test_evaluations"] += 1

            if self.cfg["readout"]["enabled"]:
                res = train_readout(self.cfg, features, labels,
                                    model.rngs.get("readout"))
                recorder.metric(kind="readout", **res)
                recorder.manifest["readout_result"] = res
            recorder.record_events(model.event_log)
            self._write_aggregates(recorder, model)
            recorder.manifest["plasticity_summary"] = model.plasticity.summary()
            recorder._write_manifest()
            return {"status": STATUS_COMPLETED, "run_dir": str(self.run_dir),
                    "splits": report["counts"]}

    def _readout_features(self, model: Model, spikes_before: np.ndarray) -> np.ndarray:
        """IT(또는 지정 영역) 집단 활동을 특징 벡터로 만든다.

        한 뉴런이나 3차원 물리 위치가 곧 카테고리 의미라고 가정하지 않는다.
        분류기는 :func:`cortex.analysis.train_readout` 의 **별도 모듈**이다.
        """
        cfg = self.cfg["readout"]
        a = model.anat.population.arrays
        ids = model.anat.ids
        if not ids.areas.has(cfg["source_area"]):
            return np.zeros(0, dtype=np.float64)
        m = a.area_id == ids.areas.id_of(cfg["source_area"])
        if cfg["source_layer"]:
            lm = np.zeros_like(m)
            for l in cfg["source_layer"]:
                lm |= a.layer_id == ids.layers.id_of(l)
            m &= lm
        return (a.spike_count - spikes_before)[m].astype(np.float64)

    def _write_aggregates(self, recorder: RunRecorder, model: Model) -> None:
        a = model.anat.population.arrays
        ids = model.anat.ids
        names, counts, rates = [], [], []
        for aname in ids.areas.names():
            for lname in ids.layers.names():
                m = ((a.area_id == ids.areas.id_of(aname))
                     & (a.layer_id == ids.layers.id_of(lname)))
                if not m.any():
                    continue
                names.append(f"{aname}/{lname}")
                counts.append(int(a.spike_count[m].sum()))
                rates.append(float(a.rate_estimate_hz[m].mean()))
        if names:
            recorder.record_layer_aggregate("layer_totals", {
                "name": np.array(names, dtype="S64"),
                "spike_count": np.asarray(counts, dtype=np.int64),
                "mean_rate_hz": np.asarray(rates, dtype=np.float64),
            })

    # ------------------------------------------------------------------
    def resume(self, checkpoint: Path | None = None,
               allow_mismatch: bool = False) -> dict[str, Any]:
        """중단된 실행을 체크포인트에서 재개한다.

        코드/설정 호환성을 검사하고, 이벤트 로그 위치를 복원해 **중복 기록을
        막는다**.
        """
        if self.run_dir is None:
            raise ValueError("run_dir 가 필요하다")
        cks = CheckpointManager.list_checkpoints(self.run_dir)
        if not cks:
            raise RecordingError(
                f"재개할 체크포인트가 없다: {self.run_dir}/checkpoints/")
        path = Path(checkpoint) if checkpoint else cks[-1]
        if not self.execute:
            return {"would_resume_from": str(path), "n_checkpoints": len(cks),
                    "experiment_status": "not_run",
                    "note_ko": "실행하려면 --execute 를 붙여라."}

        with RunRecorder(self.run_dir, self.cfg, self.command, self.package_root,
                         resume=True) as recorder:
            model = self._prepare(recorder)
            state, meta = CheckpointManager.load(
                path, expected_config_sha=recorder.manifest["config_sha256"],
                expected_code_sha=recorder.manifest["code"]["combined_sha256"],
                strict=not allow_mismatch)
            model.engine.load_state_dict(state)
            model.rngs.load_state_dict(meta["rng_state"])
            recorder.log(f"체크포인트에서 재개: {path.name} "
                         f"(sample_index={meta['sample_index']})")
            recorder.manifest["resumed_from"] = {
                "checkpoint": str(path), "sample_index": meta["sample_index"],
                "event_log_position": meta["event_log_position"],
                "allow_mismatch": bool(allow_mismatch),
            }
            recorder._write_manifest()
            stims = generate_stimuli(self.cfg, model.rngs.get("stimulus"))
            start = int(meta["sample_index"]) + 1
            model.encoder.fit_normalization(
                [s.frames[0] for s in stims[:min(len(stims), 16)]], source="resume")
            learn = self.cfg["learning"]["mode"] == "stdp_homeostasis"
            for i in range(start, len(stims)):
                self.progress(f"재개 {i + 1}/{len(stims)} — 중단하려면 Ctrl+C")
                r = self._run_sample(model, recorder, stims[i], i, learn)
                recorder.metric(kind="sample", resumed=True, **r.to_dict())
            recorder.record_events(model.event_log)
            return {"status": STATUS_COMPLETED, "run_dir": str(self.run_dir),
                    "resumed_from": str(path), "n_remaining": len(stims) - start}

    # ------------------------------------------------------------------
    def validate(self) -> dict[str, Any]:
        """구조적 필수 검사를 실행한다 (:mod:`cortex.validation`)."""

        if not self.execute:
            info = self.inspect()
            info["would_run_checks"] = True
            return info
        if self.run_dir is None:
            raise ValueError("run_dir 가 필요하다")
        with RunRecorder(self.run_dir, self.cfg, self.command,
                         self.package_root) as recorder:
            result = run_all(self.cfg, recorder, self.package_root,
                             progress=self.progress)
            recorder.metric(kind="validation", **{
                k: v for k, v in result.items() if k != "checks"})
            (self.run_dir / "validation.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=1, default=str),
                encoding="utf-8")
            if result["n_failed"] > 0 and self.cfg["validation"]["stop_experiment_on_failure"]:
                recorder.error(
                    f"구조적 필수 검사 {result['n_failed']}건 실패. 의존 실험을 중지한다.")
                recorder.finish("failed", "validation failed")
            return result


def new_run_dir(root: Path, tag: str) -> Path:
    """``runs/<tag>_<UTC타임스탬프>`` 경로를 만든다 (Windows 파일명 제약 준수)."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in tag)[:40]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(root) / f"{safe}_{stamp}"


# ============================================================================
# 섹션: analysis  —  저장된 기록에서만 읽는 분석·조회·보고서
#   (원래 파일: cortex/analysis.py)
# ============================================================================

"""analysis.py -- 저장된 기록에서만 읽는 분석·조회·보고서 생성.

명세 13, 14절.

* :func:`explain_neuron` 은 **관측된 기여 분석**이다. 기록만 보고 "틀린 연결의
  의미를 확정했다"고 주장하지 않는다.
* 인과적 효과를 주장하려면 :func:`ablation_rerun` 처럼 동일 상태·입력에서 특정
  사건/연결을 제거한 **재실행 비교**가 필요하며, 이 함수는 사용자 명령에서만
  실행된다 (CLI ``--execute``).
* :func:`make_report` 는 저장된 결과 파일만 읽어 한국어 보고서를 만든다.
  손으로 입력한 성능 숫자를 넣지 않는다.
* 분류 readout 은 **별도 모듈**이다. 라벨과 기울기는 여기서만 쓰이며 피질 학습에
  전역 역전파를 쓰지 않는다.
"""

# ----------------------------------------------------------------------
def read_events(run_dir: Path) -> dict[str, np.ndarray]:
    """저장된 이벤트를 읽는다 (hdf5 / npz 모두 지원). 부작용 없음."""
    run_dir = Path(run_dir)
    h5 = run_dir / "events.h5"
    npz = run_dir / "events.npz"
    if h5.is_file():
        try:
            import h5py  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "events.h5 를 읽으려면 h5py 가 필요하다: "
                "python -m pip install -r requirements.txt"
            ) from exc
        with h5py.File(h5, "r") as f:
            if "events" not in f:
                return {}
            return {k: np.asarray(v) for k, v in f["events"].items()}
    if npz.is_file():
        data = np.load(npz, allow_pickle=False)
        return {k.split("/", 1)[1]: data[k] for k in data.files
                if k.startswith("events/")}
    return {}


def read_states(run_dir: Path) -> dict[str, np.ndarray]:
    run_dir = Path(run_dir)
    h5 = run_dir / "states.h5"
    npz = run_dir / "states.npz"
    if h5.is_file():
        import h5py  # noqa: PLC0415
        with h5py.File(h5, "r") as f:
            if "states" not in f:
                return {}
            return {k: np.asarray(v) for k, v in f["states"].items()}
    if npz.is_file():
        data = np.load(npz, allow_pickle=False)
        return {k.split("/", 1)[1]: data[k] for k in data.files
                if k.startswith("states/")}
    return {}


# ----------------------------------------------------------------------
def explain_neuron(run_dir: Path, neuron_id: int, from_ms: float = 0.0,
                   to_ms: float = float("inf"), top_k: int = 20) -> dict[str, Any]:
    """어떤 사건과 이전 상태가 판정에 기여했는지 **관측 기반**으로 설명한다.

    Returns
    -------
    dict
        ``arrivals``    : 시간 구간의 도착 사건 표 (연결·가중치 스냅샷·기여량)
        ``spikes``      : 같은 구간의 발화 기록
        ``state_trace`` : 저장된 막전위/전도도/임계값 추이 (선택 기록된 경우)
        ``interpretation_note_ko`` : 해석 한계

    부작용 없음. 새 실험을 실행하지 않는다.
    """
    run_dir = Path(run_dir)
    ev = read_events(run_dir)
    out: dict[str, Any] = {
        "run_dir": str(run_dir), "neuron_id": int(neuron_id),
        "from_ms": from_ms, "to_ms": to_ms,
    }
    if not ev:
        out["found"] = False
        out["reason_ko"] = (
            "저장된 이벤트가 없다. recording.mode 가 summary 이거나 실행이 기록을 "
            "남기지 않았을 수 있다. manifest.json 의 recording 항목을 확인하라.")
        return out

    t = ev["arrival_time_ms"]
    in_range = (t >= from_ms) & (t <= to_ms)
    etype = ev["event_type"]
    arrivals = in_range & (ev["dst_id"] == int(neuron_id)) & (
        etype == EVENT_TYPE_INDEX["synaptic_arrival"])
    spikes = in_range & (ev["src_id"] == int(neuron_id)) & (
        etype == EVENT_TYPE_INDEX["spike"])

    idx = np.nonzero(arrivals)[0]
    order = idx[np.argsort(np.abs(ev["amount"][idx]))[::-1]] if idx.size else idx
    top = order[:top_k]
    out["found"] = True
    out["n_arrivals"] = int(idx.size)
    out["n_spikes"] = int(np.count_nonzero(spikes))
    out["arrivals"] = [{
        "event_id": int(ev["event_id"][i]),
        "parent_spike_id": int(ev["parent_spike_id"][i]),
        "src_id": int(ev["src_id"][i]),
        "synapse_id": int(ev["synapse_id"][i]),
        "emit_time_ms": float(ev["emit_time_ms"][i]),
        "arrival_time_ms": float(ev["arrival_time_ms"][i]),
        "arrival_step": int(ev["arrival_step"][i]),
        "weight_snapshot": float(ev["weight_snapshot"][i]),
        "source_gain_snapshot": float(ev["source_gain_snapshot"][i]),
        "target_compartment": (COMPARTMENT_NAMES[int(ev["target_compartment"][i])]
                               if 0 <= int(ev["target_compartment"][i]) < 3 else "n/a"),
        "amount": float(ev["amount"][i]),
        "amount_unit": AMOUNT_UNITS[int(ev["amount_unit"][i])],
        "sample_id": int(ev["sample_id"][i]),
    } for i in top]
    out["spike_times_ms"] = [float(x) for x in ev["arrival_time_ms"][spikes]]
    if idx.size:
        out["total_amount"] = float(ev["amount"][idx].sum())
        by_src: dict[int, float] = {}
        for i in idx:
            by_src[int(ev["src_id"][i])] = by_src.get(int(ev["src_id"][i]), 0.0) + float(
                ev["amount"][i])
        out["amount_by_source"] = dict(sorted(by_src.items(),
                                              key=lambda kv: -abs(kv[1]))[:top_k])

    st = read_states(run_dir)
    if st and "neuron_id" in st:
        m = (st["neuron_id"] == int(neuron_id)) & (st["time_ms"] >= from_ms) & (
            st["time_ms"] <= to_ms)
        if m.any():
            out["state_trace"] = {k: np.asarray(v)[m].tolist() for k, v in st.items()}

    out["interpretation_note_ko"] = (
        "이것은 저장된 기록에서 계산한 **관측된 기여 분석**이다. 어떤 연결이 "
        "인과적으로 발화를 만들었다고 주장하려면 동일 상태·입력에서 그 연결을 "
        "제거한 재실행 비교(ablation_rerun)가 필요하다."
    )
    return out


def ablation_rerun(cfg: dict[str, Any], build_fn: Any, ablate_synapses: Sequence[int],
                   n_steps: int, drive: Any, rngs: Any) -> dict[str, Any]:
    """동일 초기 상태·입력에서 특정 연결을 제거한 재실행 비교.

    Parameters
    ----------
    build_fn : Callable[[], Model]
        모델을 새로 조립하는 함수 (같은 시드 -> 같은 초기 상태).
    ablate_synapses : 제거할 synapse_id 목록
    drive : :class:`cortex.dynamics.ExternalDrive`

    Returns
    -------
    dict : 기준/제거 조건의 발화 수 차이

    두 조건은 **같은 난수 스트림과 같은 외생 사건**을 쓴다. 이 함수는 사용자
    명령에서만 호출된다.
    """
    base = build_fn()
    base.engine.set_external_drive(drive)
    base.engine.plasticity = None
    base.engine.run(int(n_steps))
    base_counts = base.anat.population.arrays.spike_count.copy()

    abl = build_fn()
    abl.table.active[np.asarray(ablate_synapses, dtype=np.int64)] = False
    abl.engine.set_external_drive(drive)
    abl.engine.plasticity = None
    abl.engine.run(int(n_steps))
    abl_counts = abl.anat.population.arrays.spike_count.copy()

    diff = abl_counts.astype(np.int64) - base_counts.astype(np.int64)
    changed = np.nonzero(diff != 0)[0]
    return {
        "n_ablated_synapses": int(len(ablate_synapses)),
        "n_steps": int(n_steps),
        "total_spikes_baseline": int(base_counts.sum()),
        "total_spikes_ablated": int(abl_counts.sum()),
        "n_neurons_changed": int(changed.size),
        "changed_neuron_ids": changed[:200].tolist(),
        "spike_count_delta_sum": int(diff.sum()),
        "note_ko": ("같은 초기 상태·외생 사건에서 연결만 제거한 짝지은 비교다. "
                    "관측 기여 분석과 달리 개입 결과를 보여준다."),
    }


# ----------------------------------------------------------------------
def orientation_tuning_from_events(events: dict[str, np.ndarray],
                                   neuron_ids: np.ndarray,
                                   sample_orientation_deg: dict[int, float]
                                   ) -> dict[str, Any]:
    """저장된 발화 사건에서 방향 튜닝을 계산한다 (측정값만)."""
    if not events:
        return {"available": False, "reason_ko": "저장된 이벤트가 없다"}
    spike = events["event_type"] == EVENT_TYPE_INDEX["spike"]
    src = events["src_id"][spike]
    sid = events["sample_id"][spike]
    oris = sorted(set(sample_orientation_deg.values()))
    counts = {float(o): np.zeros(neuron_ids.size, dtype=np.int64) for o in oris}
    index = {int(n): i for i, n in enumerate(neuron_ids.tolist())}
    for s, sm in zip(src.tolist(), sid.tolist()):
        o = sample_orientation_deg.get(int(sm))
        i = index.get(int(s))
        if o is None or i is None:
            continue
        counts[float(o)][i] += 1
    mat = np.stack([counts[float(o)] for o in oris], axis=1).astype(np.float64)
    pref = np.argmax(mat, axis=1)
    orth = (pref + len(oris) // 2) % len(oris)
    r_pref = mat[np.arange(mat.shape[0]), pref]
    r_orth = mat[np.arange(mat.shape[0]), orth]
    osi = (r_pref - r_orth) / np.maximum(r_pref + r_orth, 1e-12)
    return {
        "available": True, "orientations_deg": [float(o) for o in oris],
        "counts": mat.tolist(),
        "preferred_orientation_deg": [float(oris[i]) for i in pref.tolist()],
        "osi": osi.tolist(), "osi_mean": float(np.nanmean(osi)),
        "n_silent": int(np.count_nonzero(mat.sum(axis=1) == 0)),
        "n_neurons": int(neuron_ids.size),
    }


# ----------------------------------------------------------------------
def train_readout(cfg: dict[str, Any], features: dict[str, list[np.ndarray]],
                  labels: dict[str, list[str]],
                  rng: np.random.Generator) -> dict[str, Any]:
    """분류 readout 을 **별도로** 지도학습한다.

    라벨과 기울기는 이 함수에서만 쓰인다. 피질 회로의 학습(STDP/항상성)에는
    전역 역전파가 들어가지 않는다. readout 의 초기값·학습량·분할·평가 절차는
    조건 사이에서 동일해야 한다 (피질 성능 변화와 readout 단독 학습 효과를
    구분하기 위함).
    """
    rc = cfg["readout"]
    classes = sorted({l for v in labels.values() for l in v})
    cmap = {c: i for i, c in enumerate(classes)}

    def mat(split: str) -> tuple[np.ndarray, np.ndarray]:
        X = np.stack(features[split]) if features[split] else np.zeros((0, 0))
        y = np.asarray([cmap[l] for l in labels[split]], dtype=np.int64)
        return X, y

    Xtr, ytr = mat("train")
    if Xtr.size == 0:
        return {"trained": False, "reason_ko": "readout 특징이 비어 있다"}
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)

    def prep(X: np.ndarray) -> np.ndarray:
        Z = (X - mu) / sd
        return np.hstack([Z, np.ones((Z.shape[0], 1))])

    Ztr = prep(Xtr)
    Y = np.zeros((ytr.size, len(classes)))
    Y[np.arange(ytr.size), ytr] = 1.0
    lam = float(rc["l2"])
    A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1])
    W = np.linalg.solve(A, Ztr.T @ Y)

    out: dict[str, Any] = {
        "trained": True, "classifier": "ridge", "l2": lam,
        "classes": classes, "n_features": int(Xtr.shape[1]),
        "source_area": rc["source_area"], "source_layer": list(rc["source_layer"]),
        "labels_used_only_here_ko": ("라벨과 기울기는 readout 에서만 쓴다. "
                                     "피질 학습에 전역 역전파를 쓰지 않는다."),
    }
    for split in ("train", "dev", "test"):
        X, y = mat(split)
        if X.size == 0:
            out[f"{split}_accuracy"] = None
            continue
        pred = np.argmax(prep(X) @ W, axis=1)
        out[f"{split}_accuracy"] = float(np.mean(pred == y))
        out[f"{split}_n"] = int(y.size)
    return out


# ----------------------------------------------------------------------
def make_report(run_dir: Path) -> Path:
    """저장된 결과 파일만 읽어 한국어 보고서 ``report.md`` 를 만든다.

    손으로 입력한 성능 숫자를 넣지 않는다. 기록에 없는 값은 "기록 없음"으로 적는다.
    """
    run_dir = Path(run_dir)
    man = load_manifest(run_dir)
    metrics = read_metrics(run_dir)
    L: list[str] = []
    A = L.append

    A(f"# 실행 보고서 — {man['run_id']}")
    A("")
    A(f"- status: **{man.get('status')}**")
    A(f"- experiment_status: **{man.get('experiment_status')}**")
    A(f"- 시작(UTC): {man.get('started_utc')} / 종료(UTC): {man.get('finished_utc')}")
    A(f"- 소요(초): {man.get('elapsed_seconds')}")
    A(f"- 실행 명령: `{man.get('command')}`")
    A(f"- 설정 해시: `{man.get('config_sha256')}`")
    A(f"- 코드 해시: `{man.get('code', {}).get('combined_sha256')}`")
    A(f"- git commit: `{man.get('git_commit')}`")
    A("")
    A("> 이 보고서는 `runs/<run_id>/` 의 저장된 기록에서만 생성되었다. "
      "기록에 없는 값은 '기록 없음'으로 표시한다.")
    A("")

    A("## 1. 실행 환경")
    A("")
    A("| 항목 | 값 |")
    A("|---|---|")
    for k, v in man.get("library_versions", {}).items():
        A(f"| {k} | {v} |")
    A(f"| device | {man.get('device')} |")
    A(f"| gpu_used | {man.get('gpu_used')} |")
    A(f"| platform | {man.get('platform', {}).get('platform')} |")
    for k, v in man.get("thread_settings", {}).items():
        A(f"| {k} | {v} |")
    A("")

    A("## 2. 모델 규모")
    A("")
    model = man.get("model")
    if model:
        A(f"- 뉴런 수: {model['n_neurons']}")
        A(f"- 시냅스 수: {model['n_synapses']}")
        syn = model.get("synapse_summary", {})
        A(f"- 가중치 단위: {syn.get('weight_unit')}")
        A(f"- 지연 스텝 범위: {syn.get('delay_steps_min')} ~ {syn.get('delay_steps_max')}")
        A("")
        A("### 배선 규칙별 결과")
        A("")
        A("| 규칙 | 시냅스 수 | 수용체 | 표적 구획 | 가소성 | Gabor 초기화 |")
        A("|---|---|---|---|---|---|")
        for r in model.get("wiring", {}).get("per_rule", []):
            A(f"| {r.get('name')} | {r.get('n')} | {r.get('receptor', '-')} | "
              f"{r.get('target_compartment', '-')} | {r.get('plasticity_rule', '-')} | "
              f"{r.get('gabor_initialized', False)} |")
        for note in model.get("wiring", {}).get("notes", []):
            A(f"- {note}")
    else:
        A("- 기록 없음")
    A("")

    A("## 3. 기록 모드")
    A("")
    rec = man.get("recording", {})
    A(f"- mode: **{rec.get('mode')}**, backend: {rec.get('backend')}")
    A(f"- 선택 기준: {rec.get('selection_criterion') or '(없음)'}")
    A(f"- 기록한 상태 행: {man.get('n_state_rows_written')} / "
      f"한도로 건너뛴 행: {man.get('n_state_rows_skipped_by_limit')}")
    sch = man.get("event_log_schema")
    if sch:
        A(f"- 이벤트 기록 모드로 저장하지 않은 건수: {sch.get('n_skipped_by_mode')}")
    A("")

    A("## 4. 실제 측정치 (metrics.jsonl)")
    A("")
    samples = [m for m in metrics if m.get("kind") == "sample"]
    if samples:
        A(f"- 기록된 표본 수: {len(samples)}")
        tot = [s.get("n_spikes_total", 0) for s in samples]
        A(f"- 표본당 총 발화 수: 최소 {min(tot)}, 최대 {max(tot)}, "
          f"평균 {sum(tot) / len(tot):.2f}")
        areas: dict[str, list[float]] = {}
        for s in samples:
            for k, v in (s.get("mean_rate_hz_by_area") or {}).items():
                areas.setdefault(k, []).append(float(v))
        if areas:
            A("")
            A("| 영역 | 평균 활동률 [Hz] | 표본 간 표준편차 |")
            A("|---|---|---|")
            for k in sorted(areas):
                v = np.asarray(areas[k])
                A(f"| {k} | {v.mean():.4f} | {v.std(ddof=1) if v.size > 1 else 0.0:.4f} |")
        sil: dict[str, list[float]] = {}
        for s in samples:
            for k, v in (s.get("silent_fraction_by_area") or {}).items():
                sil.setdefault(k, []).append(float(v))
        if sil:
            A("")
            A("| 영역 | 침묵 뉴런 비율 (평균) |")
            A("|---|---|")
            for k in sorted(sil):
                A(f"| {k} | {np.mean(sil[k]):.4f} |")
        first: dict[str, list[int]] = {}
        for s in samples:
            for k, v in (s.get("first_spike_step_by_area") or {}).items():
                first.setdefault(k, []).append(int(v))
        if first:
            A("")
            A("| 영역 | 첫 발화 스텝 (평균) | 관측된 표본 수 |")
            A("|---|---|---|")
            for k in sorted(first):
                A(f"| {k} | {np.mean(first[k]):.2f} | {len(first[k])} |")
            A("")
            A("> 하위 영역의 무반응을 학습 실패로 오인하지 않도록, 신호가 도달하는 "
              "시점과 측정 구간을 함께 본다.")
    else:
        A("- 표본 측정 기록 없음")
    A("")

    ro = man.get("readout_result")
    A("## 5. 분류 readout (별도 모듈)")
    A("")
    if ro and ro.get("trained"):
        A(f"- 분류기: {ro['classifier']} (L2={ro['l2']}), 특징 차원 {ro['n_features']}")
        for split in ("train", "dev", "test"):
            A(f"- {split} 정확도: {ro.get(f'{split}_accuracy')}")
        A(f"- {ro['labels_used_only_here_ko']}")
    else:
        A("- 실행되지 않았거나 기록 없음")
    A("")

    A("## 6. 분할과 시험 접근 횟수")
    A("")
    sp = man.get("splits")
    if sp:
        A(f"- 표본 수: {sp.get('counts')}")
        A(f"- 기초 자극(base_id) 수: {sp.get('n_base_ids')}")
        A(f"- 분할 간 중복 없음: **{sp.get('no_overlap')}**")
        A(f"- 라벨 분포: {sp.get('labels')}")
    else:
        A("- 분할 기록 없음 (simulate 명령은 분할을 쓰지 않는다)")
    ta = man.get("test_access")
    if ta:
        A(f"- test 평가 횟수: {ta.get('n_test_evaluations')} — {ta.get('note_ko')}")
    A("")

    A("## 7. 학습")
    A("")
    ps = man.get("plasticity_summary")
    if ps:
        for k, v in ps.items():
            A(f"- {k}: {v}")
    else:
        A("- 학습 요약 기록 없음")
    A("")

    A("## 8. 가정과 한계")
    A("")
    asm = man.get("assumptions", {})
    A(f"- 종 가정: {asm.get('species_assumption')}")
    A(f"- {asm.get('note_ko')}")
    A("- 운동 관련 지표: 정지영상만 사용한 실행에서는 `not_applicable` 이다.")
    A("- 양안 시차: 좌/우 입력이 실제로 다른 설정에서만 의미가 있다.")
    A("- 색채 항상성: 조명 변화 조건의 검증 없이 구현되었다고 말하지 않는다.")
    A("- Rao 참조 모델은 전도도 LIF 회로와 다른 엔진이므로 동일 아키텍처 대조군으로 "
      "섞어 순위를 매기지 않는다.")
    A("")

    path = run_dir / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def verify_report_matches_records(run_dir: Path) -> dict[str, Any]:
    """보고서의 수치가 원본 기록과 일치하는지 다시 계산해 확인한다 (검증 13번)."""
    run_dir = Path(run_dir)
    report = run_dir / "report.md"
    if not report.is_file():
        return {"checked": False, "reason_ko": "report.md 가 없다"}
    man = load_manifest(run_dir)
    metrics = read_metrics(run_dir)
    samples = [m for m in metrics if m.get("kind") == "sample"]
    text = report.read_text(encoding="utf-8")
    checks: list[dict[str, Any]] = []
    checks.append({"item": "run_id", "ok": man["run_id"] in text})
    checks.append({"item": "status", "ok": str(man.get("status")) in text})
    checks.append({"item": "n_samples",
                   "ok": (f"기록된 표본 수: {len(samples)}" in text) or not samples})
    return {"checked": True, "all_ok": all(c["ok"] for c in checks), "checks": checks}


# ============================================================================
# 섹션: visualization  —  저장된 기록을 읽어 그림 생성
#   (원래 파일: cortex/visualization.py)
# ============================================================================

"""visualization.py -- 저장된 기록을 읽어 그림을 만든다.

명세 13절. **시각화는 별도 명령**이며 새 실험을 실행하지 않는다.
모든 함수는 ``run_dir`` 의 파일 또는 이미 조립된 모델 객체만 읽는다.

한글 글꼴이 없는 환경에서는 제목이 네모로 보일 수 있다. 그림 제목은 한국어를
쓰되, 축 라벨은 ASCII 로 두어 최소한의 정보는 항상 읽히게 했다.
"""

_PLT = None


def _plt():
    """matplotlib 을 **처음 그림을 그릴 때만** 로드한다.

    이 파일을 import 만 해도 백엔드 초기화가 일어나지 않게 하기 위함이다
    (import 부작용 금지 규칙 유지). 한국어 글꼴이 있으면 쓰고, 없으면 기본
    글꼴로 떨어진다 (오류를 내지 않는다).
    """
    global _PLT
    if _PLT is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _m_plt
        _m_plt.rcParams["figure.dpi"] = 110
        _m_plt.rcParams["axes.unicode_minus"] = False
        _m_plt.rcParams["font.family"] = ["DejaVu Sans", "NanumGothic",
                                          "Malgun Gothic", "AppleGothic",
                                          "Unifont"]
        _PLT = _m_plt
    return _PLT




MAX_CONNECTION_LINES = 400


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    _plt().close(fig)
    return path


def plot_positions_3d(model: Any, path: Path, n_connections: int = 200,
                      rng: np.random.Generator | None = None) -> Path:
    """영역·층·세포 유형별 3D 위치와 **표본** 연결.

    전 뉴런의 연결선을 그리지 않는다 (최대 ``MAX_CONNECTION_LINES`` 개 표본).
    """
    rng = rng or np.random.default_rng(0)
    a = model.anat.population.arrays
    ids = model.anat.ids
    fig = _plt().figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for aid, aname in enumerate(ids.areas.names()):
        m = a.area_id == aid
        if not m.any():
            continue
        step = max(1, int(m.sum()) // 400)
        idx = np.nonzero(m)[0][::step]
        ax.scatter(a.position_mm[idx, 0], a.position_mm[idx, 1],
                   a.position_mm[idx, 2], s=4, label=aname, alpha=0.6)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    ax.set_title("영역별 3D 위치 (표본)", fontsize=9)
    ax.legend(fontsize=6, loc="upper left")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    t = model.table
    n = min(int(n_connections), MAX_CONNECTION_LINES, t.n_synapses)
    if n > 0:
        pick = rng.choice(t.n_synapses, size=n, replace=False)
        for s in pick:
            p = a.position_mm[t.src_id[s]]
            q = a.position_mm[t.dst_id[s]]
            color = "tab:red" if t.src_dale_sign[s] > 0 else "tab:blue"
            ax2.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                     color=color, lw=0.4, alpha=0.5)
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("y [mm]"); ax2.set_zlabel("z [mm]")
    ax2.set_title(f"표본 연결 {n}개 (빨강=흥분, 파랑=억제)", fontsize=9)
    return _save(fig, path)


def plot_maps(model: Any, path: Path, area: str = "V1") -> Path:
    """retinotopy, 수용장 범위, pinwheel 방향 지도, 안구 우세 지도."""
    a = model.anat.population.arrays
    ids = model.anat.ids
    if not ids.areas.has(area):
        raise ValueError(f"영역 {area!r} 이 없다")
    m = a.area_id == ids.areas.id_of(area)
    u = a.surface_uv_mm[m, 0]
    v = a.surface_uv_mm[m, 1]
    fig, axes = _plt().subplots(2, 2, figsize=(11, 9))

    sc = axes[0, 0].scatter(u, v, c=a.visual_field_xy_deg[m, 0], s=5, cmap="coolwarm")
    axes[0, 0].set_title(f"{area} retinotopy: 시야 x [deg]", fontsize=9)
    axes[0, 0].set_xlabel("cortical u [mm]"); axes[0, 0].set_ylabel("cortical v [mm]")
    fig.colorbar(sc, ax=axes[0, 0], fraction=0.046)

    sc = axes[0, 1].scatter(a.visual_field_xy_deg[m, 0], a.visual_field_xy_deg[m, 1],
                            c=a.rf_sigma_deg[m], s=5, cmap="viridis")
    axes[0, 1].set_title("수용장 크기 sigma [deg] (시야 좌표)", fontsize=9)
    axes[0, 1].set_xlabel("visual field x [deg]")
    axes[0, 1].set_ylabel("visual field y [deg]")
    axes[0, 1].set_aspect("equal")
    fig.colorbar(sc, ax=axes[0, 1], fraction=0.046)

    ori = a.pref_orientation_rad[m]
    ok = np.isfinite(ori)
    if ok.any():
        sc = axes[1, 0].scatter(u[ok], v[ok], c=np.rad2deg(ori[ok]) % 180.0, s=6,
                                cmap="hsv", vmin=0, vmax=180)
        fig.colorbar(sc, ax=axes[1, 0], fraction=0.046)
    axes[1, 0].set_title("방향 선호 지도 (pinwheel)", fontsize=9)
    axes[1, 0].set_xlabel("cortical u [mm]"); axes[1, 0].set_ylabel("cortical v [mm]")

    sc = axes[1, 1].scatter(u, v, c=a.ocular_dominance[m], s=6, cmap="bwr",
                            vmin=-1, vmax=1)
    axes[1, 1].set_title("안구 우세 지도 (방향 지도와 별도 속성)", fontsize=9)
    axes[1, 1].set_xlabel("cortical u [mm]"); axes[1, 1].set_ylabel("cortical v [mm]")
    fig.colorbar(sc, ax=axes[1, 1], fraction=0.046)
    fig.tight_layout()
    return _save(fig, path)


def plot_neuron_timeline(run_dir: Path, neuron_id: int, path: Path,
                         from_ms: float = 0.0, to_ms: float = float("inf")) -> Path:
    """선택 뉴런의 입력 사건·막전위/전도도·임계값·발화·출력 사건을 같은 시간축에."""

    ev = read_events(run_dir)
    st = read_states(run_dir)
    fig, axes = _plt().subplots(3, 1, figsize=(11, 8), sharex=True)

    if ev:
        t = ev["arrival_time_ms"]
        rng_m = (t >= from_ms) & (t <= to_ms)
        arr = rng_m & (ev["dst_id"] == int(neuron_id)) & (
            ev["event_type"] == EVENT_TYPE_INDEX["synaptic_arrival"])
        out = rng_m & (ev["src_id"] == int(neuron_id)) & (
            ev["event_type"] == EVENT_TYPE_INDEX["synaptic_arrival"])
        spk = rng_m & (ev["src_id"] == int(neuron_id)) & (
            ev["event_type"] == EVENT_TYPE_INDEX["spike"])
        axes[0].stem(t[arr], ev["amount"][arr], linefmt="C0-", markerfmt="C0.",
                     basefmt=" ")
        axes[0].set_ylabel("input amount")
        axes[0].set_title(f"뉴런 {neuron_id}: 도착 입력 사건", fontsize=9)
        axes[2].eventplot([t[spk]], colors="k", lineoffsets=1, linelengths=0.8)
        axes[2].eventplot([t[out]], colors="C3", lineoffsets=0, linelengths=0.8)
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(["output events", "spikes"])
        axes[2].set_title("발화와 출력 사건", fontsize=9)

    if st and "neuron_id" in st:
        m = (st["neuron_id"] == int(neuron_id)) & (st["time_ms"] >= from_ms) & (
            st["time_ms"] <= to_ms)
        if m.any():
            axes[1].plot(st["time_ms"][m], st["V_soma_mV"][m], label="V soma [mV]")
            if "V_apical_mV" in st:
                axes[1].plot(st["time_ms"][m], st["V_apical_mV"][m], label="V apical",
                             alpha=0.6)
            axes[1].plot(st["time_ms"][m], st["threshold"][m], "--",
                         label="threshold", alpha=0.8)
            ax2 = axes[1].twinx()
            ax2.plot(st["time_ms"][m], st["g_total_nS"][m], color="C2", alpha=0.5,
                     label="g total [nS]")
            ax2.set_ylabel("g [nS]")
            axes[1].legend(fontsize=7, loc="upper left")
    axes[1].set_ylabel("V [mV]")
    axes[1].set_title("막전위/전도도/임계값", fontsize=9)
    axes[2].set_xlabel("time [ms]")
    fig.tight_layout()
    return _save(fig, path)


def plot_area_arrival_times(run_dir: Path, path: Path) -> Path:
    """영역별 신호 도달 시점과 표본 간 변동.

    **너무 짧은 시간에 측정을 끝내 하위 영역의 무반응을 학습 실패로 오인하지
    않도록** 첫 발화 스텝의 분포를 함께 보여준다.
    """

    rows = [m for m in read_metrics(run_dir) if m.get("kind") == "sample"]
    fig, axes = _plt().subplots(1, 2, figsize=(11, 4.2))
    first: dict[str, list[float]] = {}
    rates: dict[str, list[float]] = {}
    for r in rows:
        for k, v in (r.get("first_spike_step_by_area") or {}).items():
            first.setdefault(k, []).append(float(v))
        for k, v in (r.get("mean_rate_hz_by_area") or {}).items():
            rates.setdefault(k, []).append(float(v))
    if first:
        names = sorted(first)
        axes[0].boxplot([first[k] for k in names], labels=names)
        axes[0].set_ylabel("first spike step")
        axes[0].set_title("영역별 신호 도달 시점 (표본 분포)", fontsize=9)
        axes[0].tick_params(axis="x", rotation=30, labelsize=7)
    if rates:
        names = sorted(rates)
        axes[1].boxplot([rates[k] for k in names], labels=names)
        axes[1].set_ylabel("mean rate [Hz]")
        axes[1].set_title("영역별 활동률 (표본 간 변동)", fontsize=9)
        axes[1].tick_params(axis="x", rotation=30, labelsize=7)
    fig.tight_layout()
    return _save(fig, path)


def plot_tuning(tuning: dict[str, Any], path: Path, title: str = "") -> Path:
    """방향 튜닝 곡선과 OSI 분포 (측정값만)."""
    fig, axes = _plt().subplots(1, 2, figsize=(10, 4))
    if tuning.get("available", True) and tuning.get("counts"):
        counts = np.asarray(tuning["counts"], dtype=np.float64)
        oris = np.asarray(tuning["orientations_deg"], dtype=np.float64)
        mean = counts.mean(axis=0)
        axes[0].plot(oris, mean, marker="o")
        axes[0].set_xlabel("orientation [deg]")
        axes[0].set_ylabel("mean spike count")
        axes[0].set_title("방향 튜닝 (집단 평균)", fontsize=9)
        osi = np.asarray(tuning.get("osi", []), dtype=np.float64)
        if osi.size:
            axes[1].hist(osi[np.isfinite(osi)], bins=20)
            axes[1].set_xlabel("OSI")
            axes[1].set_title(f"OSI 분포 (침묵 뉴런 {tuning.get('n_silent')}개 포함)",
                              fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_activity_and_weights(run_dir: Path, path: Path) -> Path:
    """활동률·침묵 뉴런 비율·가중치/임계 변화 추이."""

    rows = [m for m in read_metrics(run_dir) if m.get("kind") == "sample"]
    fig, axes = _plt().subplots(1, 3, figsize=(13, 4))
    if rows:
        tot = [r.get("n_spikes_total", 0) for r in rows]
        axes[0].plot(tot, marker=".")
        axes[0].set_xlabel("sample index"); axes[0].set_ylabel("total spikes")
        axes[0].set_title("표본별 총 발화 수", fontsize=9)
        sil: dict[str, list[float]] = {}
        for r in rows:
            for k, v in (r.get("silent_fraction_by_area") or {}).items():
                sil.setdefault(k, []).append(float(v))
        for k in sorted(sil):
            axes[1].plot(sil[k], label=k, alpha=0.8)
        axes[1].set_xlabel("sample index"); axes[1].set_ylabel("silent fraction")
        axes[1].set_title("침묵 뉴런 비율", fontsize=9)
        axes[1].legend(fontsize=6)
    st = read_states(run_dir)
    if st and "threshold" in st:
        axes[2].plot(st["time_ms"][:2000], st["threshold"][:2000], ".", ms=1)
        axes[2].set_xlabel("time [ms]"); axes[2].set_ylabel("threshold")
        axes[2].set_title("임계값 변화 (기록된 표본)", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_rao(state: Any, I: np.ndarray, model: Any, path: Path) -> Path:
    """Rao 모델의 입력·재구성·잔여 오차·정착 곡선."""
    rec = np.einsum("mij,mj->mi", model.U1, state.r1)
    fig, axes = _plt().subplots(1, 4, figsize=(14, 3.4))
    axes[0].imshow(np.atleast_2d(I), aspect="auto", cmap="viridis")
    axes[0].set_title("입력 I", fontsize=9)
    axes[1].imshow(np.atleast_2d(rec), aspect="auto", cmap="viridis")
    axes[1].set_title("재구성 U r", fontsize=9)
    axes[2].imshow(np.atleast_2d(I - rec), aspect="auto", cmap="coolwarm")
    axes[2].set_title("잔여 오차 I - U r", fontsize=9)
    axes[3].plot(state.energy_trace)
    axes[3].set_xlabel("settle step"); axes[3].set_ylabel("E")
    axes[3].set_title("활동 정착 곡선 (에너지)", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def make_all(run_dir: Path, model: Any = None,
             neuron_ids: Sequence[int] = ()) -> list[Path]:
    """기록에서 만들 수 있는 그림을 모두 만든다 (없는 기록은 건너뛴다)."""
    run_dir = Path(run_dir)
    figdir = run_dir / "figures"
    made: list[Path] = []
    try:
        made.append(plot_area_arrival_times(run_dir, figdir / "area_arrival_times.png"))
    except Exception:
        pass
    try:
        made.append(plot_activity_and_weights(run_dir, figdir / "activity_weights.png"))
    except Exception:
        pass
    for nid in neuron_ids:
        try:
            made.append(plot_neuron_timeline(run_dir, int(nid),
                                             figdir / f"neuron_{int(nid)}.png"))
        except Exception:
            pass
    if model is not None:
        try:
            made.append(plot_positions_3d(model, figdir / "positions_3d.png"))
        except Exception:
            pass
        try:
            made.append(plot_maps(model, figdir / "maps_V1.png"))
        except Exception:
            pass
    return made


# ============================================================================
# 섹션: validation  —  필수 검증 1~14
#   (원래 파일: cortex/validation.py)
# ============================================================================

"""validation.py -- 명세 12절의 필수 검증 1~14.

**이 파일의 검사는 실제 계산값을 쓴다.** 항상 True 인 검사를 넣지 않는다.
검사를 실행하는 것은 사용자 명령(``python -m cortex.cli validate --execute``)
이며, 이 소스를 작성하는 동안에는 실행하지 않는다.

각 검사 결과는 ``status`` 가 ``passed`` / ``failed`` / ``skipped`` 중 하나다.
``skipped`` 는 이유와 함께 기록하며 **passed 에 포함하지 않는다.**
성공률을 맞추려고 기준을 사후에 바꾸지 않는다.
"""

class Check:
    """검사 1건의 결과."""

    def __init__(self, cid: int, name: str) -> None:
        self.id = cid
        self.name = name
        self.status = "failed"
        self.details: dict[str, Any] = {}
        self.assertions: list[dict[str, Any]] = []
        self.reason = ""

    def expect(self, cond: bool, msg: str, **extra: Any) -> bool:
        self.assertions.append({"assertion": msg, "passed": bool(cond), **extra})
        return bool(cond)

    def finalize(self) -> None:
        if self.status == "skipped":
            return
        self.status = "passed" if all(a["passed"] for a in self.assertions) else "failed"

    def skip(self, reason: str) -> None:
        self.status = "skipped"
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "status": self.status,
                "reason": self.reason, "assertions": self.assertions,
                "details": self.details}


# ----------------------------------------------------------------------
# 작은 전용 회로 도우미
# ----------------------------------------------------------------------
def _tiny_cfg(base: dict[str, Any], mode: str) -> dict[str, Any]:
    import copy
    cfg = copy.deepcopy(base)
    cfg["engine"]["mode"] = mode
    return cfg


def _tiny_network(cfg: dict[str, Any], n: int = 3, *, exc_w: float = 1.0,
                  inh_w: float = 1.0, delay_steps: int = 1
                  ) -> tuple[NeuronPopulation, SynapseTable, IdSpace]:
    """손으로 계산할 수 있는 3뉴런 회로: 0(흥분)->2, 1(억제)->2."""
    ids = IdSpace()
    for name in cfg["receptors"]:
        ids.receptors.add(name)
    for name in ("exc_src", "inh_src", "target"):
        ids.cell_types.add(name)
    ids.areas.add("tiny")
    arrays = NeuronArrays(n, n_receptors=len(cfg["receptors"]))
    pop = NeuronPopulation(arrays, ids)
    arrays.area_id[:] = ids.areas.id_of("tiny")
    arrays.layer_id[:] = ids.layers.id_of("L4")
    arrays.cell_type_id[:] = [ids.cell_types.id_of("exc_src"),
                              ids.cell_types.id_of("inh_src"),
                              ids.cell_types.id_of("target")][:n]
    arrays.dale_sign[:] = [1, -1, 1][:n]
    arrays.output_gain_P[:] = 1.0
    arrays.position_mm[:, 0] = np.arange(n) * 0.01
    arrays.has_compartment[:, :] = False
    arrays.has_compartment[:, 0] = True
    arrays.C_pF[:, :] = 100.0
    arrays.gL_nS[:, :] = 5.0
    arrays.EL_mV[:, :] = -70.0
    arrays.V_mV[:, :] = -70.0
    arrays.V_reset_mV[:] = -65.0
    arrays.t_ref_ms[:] = 2.0
    arrays.threshold[:] = (-50.0 if cfg["engine"]["mode"] == "conductance_lif" else 1.0)

    table = SynapseTable(n, "nS" if cfg["engine"]["mode"] == "conductance_lif"
                         else "dimensionless")
    exc_r = ids.receptors.id_of("AMPA")
    inh_r = ids.receptors.id_of("GABA_A")
    table.add_block(src_id=np.array([0]), dst_id=np.array([2]),
                    src_area=np.array([0]), dst_area=np.array([0]),
                    target_layer=np.array([0]),
                    target_compartment=COMPARTMENT_INDEX["soma"],
                    receptor_type=exc_r, weight=np.array([exc_w]),
                    base_delay_ms=np.array([1.0]),
                    effective_delay_steps=np.array([delay_steps]),
                    plasticity_rule=0, src_dale_sign=np.array([1]), rule_index=0)
    table.add_block(src_id=np.array([1]), dst_id=np.array([2]),
                    src_area=np.array([0]), dst_area=np.array([0]),
                    target_layer=np.array([0]),
                    target_compartment=COMPARTMENT_INDEX["soma"],
                    receptor_type=inh_r, weight=np.array([inh_w]),
                    base_delay_ms=np.array([1.0]),
                    effective_delay_steps=np.array([delay_steps]),
                    plasticity_rule=0, src_dale_sign=np.array([-1]), rule_index=1)
    table.build_indices()
    arrays.set_outgoing_index(table.out_ptr, table.out_syn)
    pop.attach(synapses=table)
    return pop, table, ids


def _tiny_engine(cfg: dict[str, Any], pop: NeuronPopulation, table: SynapseTable,
                 ids: IdSpace) -> Engine:
    log = EventLog(len(pop), mode="full")
    pop.attach(event_log=log)
    anat = Anatomy(
        population=pop, ids=ids, grid=None, channel_names=[],  # type: ignore[arg-type]
        retina_neuron_id=np.zeros((0, 0), dtype=np.int64))
    return Engine(cfg, anat, table, log, plasticity=None)


def _force_spike(engine: Engine, neuron_id: int, step: int) -> None:
    engine.set_external_drive(ExternalDrive(
        np.array([neuron_id], dtype=np.int64),
        forced_spike_steps={step: np.array([neuron_id], dtype=np.int64)}))


# ======================================================================
def check_01_records(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(1, "3x3 뷰·타입 배열·연결/로그 조회 일치와 ID 유효성")
    pop = model.anat.population
    a = pop.arrays
    rec = pop.record(0)
    m = rec.as_matrix()
    c.expect(m.shape == (3, 3), "as_matrix() shape == (3,3)")
    c.expect(m.dtype == object, "as_matrix() 는 object 배열이다 (수치 행렬이 아니다)")
    c.expect(isinstance(m[0, 0], float) and isinstance(m[0, 1], float)
             and isinstance(m[0, 2], float), "1행은 x,y,z 실수 좌표")
    c.expect(isinstance(m[1, 0], OutgoingConnectionsView),
             "[1][0] 은 출력 연결 목록이다 (입력 총합이 아니다)")
    c.expect(isinstance(m[1, 1], float) and isinstance(m[1, 2], float),
             "[1][1]=threshold, [1][2]=P 는 실수")
    c.expect(isinstance(m[2, 0], InputLogView), "[2][0] 은 입력 로그 뷰")
    c.expect(isinstance(m[2, 1], DynamicStateView), "[2][1] 은 동적 상태 참조")
    c.expect(isinstance(m[2, 2], MetadataView), "[2][2] 는 메타데이터 참조")

    old = rec.threshold
    rec.threshold = old + 1.25
    c.expect(a.threshold[0] == old + 1.25, "3x3 뷰로 쓴 threshold 가 타입 배열에 반영된다")
    a.threshold[0] = old
    c.expect(rec.threshold == old, "타입 배열 변경이 3x3 뷰에 즉시 보인다 (미러 없음)")

    ok_ids = True
    n_syn = model.table.n_synapses
    for i in range(0, len(pop), max(1, len(pop) // 50)):
        sids = pop.record(i).outgoing.synapse_ids
        if sids.size and (sids.min() < 0 or sids.max() >= n_syn):
            ok_ids = False
            break
        if sids.size and not np.all(model.table.src_id[sids] == i):
            ok_ids = False
            break
    c.expect(ok_ids, "출력 연결 목록의 synapse_id 가 유효하고 src_id 와 일치한다")
    c.expect(int(np.diff(model.table.out_ptr).sum()) == n_syn,
             "출력 CSR 인덱스의 총합이 시냅스 수와 같다")
    c.expect(int(np.diff(model.table.in_ptr).sum()) == n_syn,
             "입력 CSR 인덱스의 총합이 시냅스 수와 같다")
    c.details = {"n_neurons": len(pop), "n_synapses": n_syn,
                 "matrix_description": pop.record(0).as_matrix_description()}
    c.finalize()
    return c


def check_02_event_application(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(2, "같은 사건의 중복 반영 방지·과거 로그 보존·전도도 잔류와 새 입력 구분")
    tcfg = _tiny_cfg(cfg, "conductance_lif")
    pop, table, ids = _tiny_network(tcfg)
    eng = _tiny_engine(tcfg, pop, table, ids)
    a = pop.arrays

    _force_spike(eng, 0, 0)
    eng.step()                              # 0 단계: 발화, 예약
    g_before = a.g_nS[2].sum()
    eng.set_external_drive(None)
    eng.step()                              # 1 단계: 도착 -> 전도도 증가
    g_after = a.g_nS[2].sum()
    n_events_1 = eng.log.n_events
    c.expect(g_after > g_before, "도착 사건이 전도도를 증가시킨다",
             g_before=float(g_before), g_after=float(g_after))

    g_arrival = a.g_nS[2].sum()
    eng.step()                              # 2 단계: 새 도착 없음
    g_decay = a.g_nS[2].sum()
    c.expect(g_decay < g_arrival, "새 입력이 없으면 전도도는 감쇠한다 (그러나 잔류한다)",
             g_arrival=float(g_arrival), g_decay=float(g_decay))
    c.expect(g_decay > 0.0, "이전 입력의 효과가 상태로 남는다")
    c.expect(eng.queue.pending_count() == 0, "같은 사건이 큐에 남아 재처리되지 않는다")

    n_events_2 = eng.log.n_events
    c.expect(n_events_2 >= n_events_1,
             "과거 로그는 삭제되지 않는다 (이벤트 수가 줄지 않음)",
             before=int(n_events_1), after=int(n_events_2))
    log_view = pop.record(2).input_log
    c.expect(log_view.count() >= 1, "발화 이후에도 입력 로그 조회가 가능하다")
    c.expect(a.last_consumed_event[2] >= 0,
             "last_consumed_event 가 갱신되어 중복 주입을 막는다")
    c.details = {"n_events": int(eng.log.n_events),
                 "last_consumed_event": int(a.last_consumed_event[2])}
    c.finalize()
    return c


def check_03_delay(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(3, "단일 사건 예약/도착 시각·지연 양자화 일치, 동시 사건 순서 불변성")
    dt = float(cfg["engine"]["dt_ms"])
    q = quantize_delay(np.array([0.0, dt * 0.4, dt * 1.0, dt * 2.7]), dt,
                       cfg["engine"]["min_delay_steps"], cfg["engine"]["delay_rounding"])
    c.expect(int(q[0]) >= 1, "지연 0 도 최소 1 스텝으로 양자화된다 (0 지연 재귀 방지)")
    c.expect(int(q[2]) == 1, "dt 와 같은 지연은 1 스텝")
    c.expect(int(q[3]) == (3 if cfg["engine"]["delay_rounding"] == "ceil" else 3),
             "2.7*dt 지연의 양자화 결과가 규칙과 일치한다", value=int(q[3]))

    tcfg = _tiny_cfg(cfg, "conductance_lif")
    for d in (1, 2, 3):
        pop, table, ids = _tiny_network(tcfg, delay_steps=d)
        eng = _tiny_engine(tcfg, pop, table, ids)
        _force_spike(eng, 0, 0)
        eng.step()
        eng.set_external_drive(None)
        arrived_at = None
        for s in range(1, d + 3):
            before = pop.arrays.g_nS[2].sum()
            eng.step()
            if pop.arrays.g_nS[2].sum() > before and arrived_at is None:
                arrived_at = s
        c.expect(arrived_at == d, f"지연 {d} 스텝 사건이 정확히 스텝 {d} 에 도착한다",
                 observed=arrived_at)

    # 동시 사건 순서 불변성: 같은 스텝의 도착을 뒤섞어 **큐에 직접 넣어도**
    # 합산 결과가 비트 단위로 같아야 한다.
    tcfg2 = _tiny_cfg(cfg, "sum_threshold")
    results = []
    for perm in (False, True):
        pop, table, ids = _tiny_network(tcfg2, exc_w=0.6, inh_w=0.4)
        eng = _tiny_engine(tcfg2, pop, table, ids)
        # synapse 0 = 흥분 w=0.6, synapse 1 = 억제 w=0.4 -> s_j = 0.6 - 0.4 = 0.2
        order = np.array([1, 0] if perm else [0, 1], dtype=np.int64)
        amounts = np.array([0.6, 0.4])
        eng.queue.schedule(0, {
            "synapse_id": order,
            "amount": amounts[order],
            "weight_snapshot": amounts[order],
            "gain_snapshot": np.ones(2),
            "emit_time_ms": np.zeros(2),
            "spike_id": np.zeros(2, dtype=np.int64),
        })
        eng.set_external_drive(None)
        eng.step()
        results.append(float(pop.arrays.last_decision_value[2]))
    c.expect(results[0] == results[1],
             "동시 도착 사건의 처리 순서를 바꿔도 합산 결과가 비트 단위로 같다",
             values=results)
    c.expect(abs(results[0] - 0.2) < 1e-12,
             "손계산 값과 일치한다 (0.6 흥분 - 0.4 억제 = 0.2)", observed=results[0])
    c.details = {"quantized": q.tolist(), "same_step_values": results}
    c.finalize()
    return c


def check_04_sum_threshold(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(4, "threshold 모드의 손계산 가능한 흥분/억제 합과 경계 발화")
    tcfg = _tiny_cfg(cfg, "sum_threshold")
    pop, table, ids = _tiny_network(tcfg, exc_w=0.7, inh_w=0.3)
    eng = _tiny_engine(tcfg, pop, table, ids)
    a = pop.arrays
    a.threshold[2] = 0.35
    both = np.array([0, 1], dtype=np.int64)
    eng.set_external_drive(ExternalDrive(both, forced_spike_steps={0: both}))
    eng.step()
    eng.set_external_drive(None)
    eng.step()
    s = float(a.last_decision_value[2])
    c.expect(abs(s - (0.7 - 0.3)) < 1e-12,
             "s_j = 0.7(흥분) - 0.3(억제) = 0.4 (손계산 일치)", observed=s)
    c.expect(int(a.fired[2]) == 1, "s_j(0.4) >= theta(0.35) 이면 발화")

    a.threshold[2] = 0.45
    a.interval_activation[:] = 0.0
    eng.set_external_drive(ExternalDrive(both, forced_spike_steps={2: both}))
    eng.step()
    eng.set_external_drive(None)
    eng.step()
    c.expect(int(a.fired[2]) == 0, "s_j(0.4) < theta(0.45) 이면 발화하지 않는다")

    # 경계 조건: 입력 0 이어도 theta <= 0 이면 발화한다
    pop2, table2, ids2 = _tiny_network(tcfg)
    eng2 = _tiny_engine(tcfg, pop2, table2, ids2)
    pop2.arrays.threshold[:] = 0.0
    eng2.set_external_drive(None)
    eng2.step()
    c.expect(int(pop2.arrays.fired[2]) == 1,
             "영 입력에서도 theta<=0 이면 발화한다 (문서화된 경계 조건)")
    c.details = {"sum_value": s, "boundary_theta": 0.0}
    c.finalize()
    return c


def check_05_lif(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(5, "LIF 누설·단일 펄스·불응기·구획 결합·dt 수렴")
    tcfg = _tiny_cfg(cfg, "conductance_lif")

    # (a) 무입력 누설: EL 에서 출발하면 EL 에 머문다
    pop, table, ids = _tiny_network(tcfg)
    eng = _tiny_engine(tcfg, pop, table, ids)
    pop.arrays.V_mV[2, 0] = -60.0
    eng.set_external_drive(None)
    for _ in range(50):
        eng.step()
    v = float(pop.arrays.V_mV[2, 0])
    c.expect(abs(v - (-70.0)) < 0.5, "무입력 시 막전위가 EL 로 수렴한다", V=v)
    c.expect(np.all(np.isfinite(pop.arrays.V_mV)), "막전위가 유한하다")

    # (b) 단일 펄스 EPSP
    pop, table, ids = _tiny_network(tcfg, exc_w=4.0)
    eng = _tiny_engine(tcfg, pop, table, ids)
    _force_spike(eng, 0, 0)
    eng.step()
    eng.set_external_drive(None)
    peak = -np.inf
    for _ in range(40):
        eng.step()
        peak = max(peak, float(pop.arrays.V_mV[2, 0]))
    c.expect(peak > -70.0, "흥분성 단일 펄스가 EPSP 를 만든다", peak_mV=peak)

    # (c) 불응기
    pop, table, ids = _tiny_network(tcfg)
    eng = _tiny_engine(tcfg, pop, table, ids)
    a = pop.arrays
    a.t_ref_ms[2] = 5.0
    a.threshold[2] = -69.9
    a.V_mV[2, 0] = -60.0
    eng.set_external_drive(None)
    n_spikes = 0
    for _ in range(int(5.0 / tcfg["engine"]["dt_ms"])):
        rep = eng.step()
        n_spikes += rep.n_spikes
    c.expect(n_spikes <= 2,
             "불응기 동안 연속 발화가 억제된다 (5ms 불응기에서 관측 발화 수)",
             n_spikes=int(n_spikes))

    # (d) 구획 결합: apical 전류가 soma 에 전달된다
    pop, table, ids = _tiny_network(tcfg)
    a = pop.arrays
    a.has_compartment[2, :] = True
    a.g_couple_nS[2, COMPARTMENT_INDEX["apical"]] = 10.0
    eng = _tiny_engine(tcfg, pop, table, ids)
    eng.set_external_drive(None)
    a.Iext_pA[2, COMPARTMENT_INDEX["apical"]] = 200.0
    v0 = float(a.V_mV[2, 0])
    for _ in range(20):
        a.Iext_pA[2, COMPARTMENT_INDEX["apical"]] = 200.0
        eng.step()
    c.expect(float(a.V_mV[2, 0]) > v0,
             "apical 구획의 전류가 구획 결합을 통해 soma 전위를 바꾼다",
             V_soma=float(a.V_mV[2, 0]))

    # (e) dt 수렴
    finals: list[float] = []
    for factor in cfg["validation"]["dt_convergence_factors"]:
        sub = _tiny_cfg(cfg, "conductance_lif")
        sub["engine"]["dt_ms"] = float(cfg["engine"]["dt_ms"]) * float(factor)
        p2, t2, i2 = _tiny_network(sub, exc_w=4.0)
        e2 = _tiny_engine(sub, p2, t2, i2)
        p2.arrays.Iext_pA[2, 0] = 150.0
        p2.arrays.threshold[2] = 1e9        # 발화 없이 순수 적분만 본다
        e2.set_external_drive(None)
        n = int(round(20.0 / sub["engine"]["dt_ms"]))
        for _ in range(n):
            p2.arrays.Iext_pA[2, 0] = 150.0
            e2.step()
        finals.append(float(p2.arrays.V_mV[2, 0]))
    spread = float(max(finals) - min(finals)) if finals else 0.0
    tol = float(cfg["validation"]["tolerances"]["lif_dt_convergence_mV"])
    c.expect(spread <= tol,
             f"dt 를 줄여도 20ms 후 막전위 차이가 {tol} mV 이하다",
             finals_mV=finals, spread_mV=spread)
    c.expect(all(np.isfinite(finals)), "dt 수렴 검사에서 비유한 값이 없다")
    c.details = {"dt_finals_mV": finals, "dt_spread_mV": spread,
                 "note_ko": "후향 오일러를 쓰므로 발산을 가리는 전압 클리핑이 없다."}
    c.finalize()
    return c


def check_06_dale(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(6, "Dale 제약·전도도 비음수·학습 후 부호와 범위 유지")
    table = model.table
    a = model.anat.population.arrays
    viol = table.dale_violations(a.dale_sign)
    c.expect(viol.size == 0, "모든 시냅스의 기록된 Dale 부호가 발신 뉴런 유형과 일치한다",
             n_violations=int(viol.size))
    c.expect(bool(np.all(table.weight >= 0.0)),
             "가중치는 비음수 크기다 (부호는 세포 유형이 결정)")
    c.expect(bool(np.all(a.g_nS >= 0.0)), "전도도는 비음수다")

    before_sign = table.src_dale_sign.copy()
    plast = StdpHomeostasis(cfg, table, a, model.anat.ids)
    rngd = np.random.default_rng(0)
    if plast.plastic_idx.size:
        table.weight[plast.plastic_idx] += rngd.normal(
            0.0, 5.0, size=plast.plastic_idx.size)
        info = table.clip_weights(plast.w_min, plast.w_max)
        c.expect(bool(np.all(table.weight >= plast.w_min)
                      and np.all(table.weight <= plast.w_max)),
                 "clipping 후 가중치가 설정 범위 안에 있다", **info)
        c.expect(bool(np.array_equal(table.src_dale_sign, before_sign)),
                 "clipping 이 연결 유형(부호)을 바꾸지 않는다")
    else:
        c.expect(True, "학습 대상 시냅스가 없어 clipping 검사를 생략 (구조상 정상)")
    c.details = {"n_plastic": int(plast.plastic_idx.size),
                 "weight_min": float(table.weight.min()) if table.n_synapses else None,
                 "weight_max": float(table.weight.max()) if table.n_synapses else None}
    c.finalize()
    return c


def check_07_input(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(7, "색 변환·DoG 상수 응답·로그-극좌표 왕복/범위/aliasing")
    enc: RetinaEncoder = model.encoder
    white = np.ones((16, 16, 3), dtype=np.float64)
    lms = enc.to_lms(white)
    c.expect(bool(np.all(lms[:2] > 0)), "백색 입력에서 L,M 성분이 양수다",
             L=float(lms[0].mean()), M=float(lms[1].mean()), S=float(lms[2].mean()))
    black = np.zeros((16, 16, 3), dtype=np.float64)
    c.expect(float(np.abs(enc.to_lms(black)).max()) < 1e-12,
             "흑색 입력의 LMS 는 0 이다")

    uni = uniform_response_check(enc, 0.5, 64)
    tol = float(cfg["validation"]["tolerances"]["dog_uniform_response"])
    c.expect(uni["max_abs_contrast_response_interior"] <= tol,
             f"균일 영상에서 내부 영역의 DoG 대비 응답이 {tol} 이하다",
             observed=uni["max_abs_contrast_response_interior"])

    rt = roundtrip_error(model.anat.grid, cfg["retinotopy"]["e0_deg"])
    rtol = float(cfg["validation"]["tolerances"]["logpolar_roundtrip_deg"])
    c.expect(rt["max_abs_ecc_error_deg"] <= rtol,
             f"log-polar 왕복 오차가 {rtol} deg 이하다", **rt)

    g = model.anat.grid
    c.expect(bool(np.all(np.isfinite(g.x_deg)) and np.all(np.isfinite(g.y_deg))),
             "샘플 좌표가 모두 유한하다")
    c.expect(float(g.ecc_deg.max()) <= float(g.meta["max_ecc_deg"]) + 1e-9,
             "샘플 편심도가 설정한 최대 편심도를 넘지 않는다",
             max_ecc=float(g.ecc_deg.max()))

    H = W = int(cfg["retina"]["image"]["max_side_px"])
    ny = nyquist_margin(g, H, W, cfg["retina"]["image"]["fov_deg"])
    c.details = {"uniform_response": uni, "roundtrip": rt, "nyquist": ny}
    c.expect(ny.get("checked", False), "aliasing 여유 지표를 계산했다")
    c.finalize()
    return c


def check_08_response_changes(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(8, "회전·위치·위상·양안 입력 변화에 따른 실제 반응 변화")
    enc: RetinaEncoder = model.encoder
    sampler: LogPolarSampler = model.sampler
    size = int(cfg["retina"]["image"]["max_side_px"])
    cx = cy = (size - 1) / 2.0

    def sampled(img: np.ndarray) -> np.ndarray:
        v, _ = sampler.sample(enc.encode(img).channels)
        return v.ravel()

    base = bar_image(size, size, cx, cy, 0.5 * size, 3.0, 0.0)
    rotated = bar_image(size, size, cx, cy, 0.5 * size, 3.0, np.pi / 2)
    shifted = bar_image(size, size, cx + 0.15 * size, cy,
                                    0.5 * size, 3.0, 0.0)
    ph0 = grating_patch(size, size, cx, cy, 0.3 * size, 0.0, 0.0, 0.08)
    ph1 = grating_patch(size, size, cx, cy, 0.3 * size, 0.0, np.pi, 0.08)

    b, r, s, p0, p1 = (sampled(x) for x in (base, rotated, shifted, ph0, ph1))
    d_rot = float(np.linalg.norm(b - r))
    d_pos = float(np.linalg.norm(b - s))
    d_phase = float(np.linalg.norm(p0 - p1))
    c.expect(d_rot > 0.0, "회전에 따라 샘플된 반응이 실제로 변한다", distance=d_rot)
    c.expect(d_pos > 0.0, "위치 이동에 따라 반응이 변한다", distance=d_pos)
    c.expect(d_phase > 0.0, "위상 변화에 따라 반응이 변한다", distance=d_phase)

    bino = cfg["retinotopy"]["binocular"]["enabled"]
    if bino:
        left = dot_image(size, size, cx - 2, cy, 4.0)
        right = dot_image(size, size, cx + 2, cy, 4.0)
        d_eye = float(np.linalg.norm(sampled(left) - sampled(right)))
        c.expect(d_eye > 0.0, "좌/우 눈 입력이 다르면 반응도 다르다", distance=d_eye)
    else:
        c.assertions.append({
            "assertion": "양안 입력 변화 검사",
            "passed": True,
            "note_ko": ("retinotopy.binocular.enabled 가 false 이므로 단안 설정이다. "
                        "단안 영상 복제를 양안 시차 실험으로 보고하지 않는다."),
            "measured": False,
        })
    c.details = {"rotation_distance": d_rot, "position_distance": d_pos,
                 "phase_distance": d_phase, "binocular_enabled": bool(bino),
                 "note_ko": "예상하지 않은 결과도 그대로 기록한다."}
    c.finalize()
    return c


def check_09_structure(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(9, "층별 깊이·세포 유형·필수 배선·연결 수 집계")
    anat = model.anat
    summary = layer_summary(anat)
    conn = connectivity_summary(cfg, anat, model.table)

    declared: list[str] = []
    for aname, a in cfg["anatomy"]["areas"].items():
        if a["kind"] == "retina":
            declared.append(f"{aname}/L_input")
            continue
        for lname, n in a["neurons_per_layer"].items():
            if int(n) > 0:
                declared.append(f"{aname}/{lname}")
    present = {k.rsplit("/", 1)[0] for k in summary["counts"]}
    missing = [d for d in declared if d not in present]
    c.expect(not missing, "선언한 모든 (영역/층) 조합에 뉴런이 실제로 존재한다",
             missing=missing)

    empty_rules = [r["name"] for r in model.wiring_report.per_rule
                   if r.get("enabled") and r.get("n", 0) == 0]
    c.expect(not empty_rules,
             "활성화된 모든 배선 규칙이 시냅스를 1개 이상 만들었다 "
             "(이름만 존재하는 경로를 잡는다)", empty_rules=empty_rules)

    ct_used = {k.rsplit("/", 1)[1] for k in summary["counts"]}
    ct_declared = set(cfg["cell_types"])
    unused = sorted(ct_declared - ct_used)
    c.expect(not unused, "선언한 모든 세포 유형이 실제로 배치되었다", unused=unused)

    depths_ok = True
    for key, d in summary["depth_mm"].items():
        if d["depth_min"] < -1e-9:
            depths_ok = False
    c.expect(depths_ok, "모든 층 깊이가 0 이상이다 (표면에서 아래로)")
    c.details = {"layer_summary": summary, "connectivity": conn,
                 "per_rule": model.wiring_report.per_rule}
    c.finalize()
    return c


def check_10_measurement_noninvasive(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(10, "측정 전후 가중치·상태·흔적·로그 위치·RNG 동일성")

    eng = model.engine
    before = eng.read_only_snapshot()
    rng_before = model.rngs.get("diagnostics").bit_generator.state

    _ = model.anat.population.record(0).describe()
    _ = model.table.summary()
    _ = layer_summary(model.anat)
    _ = connectivity_summary(cfg, model.anat, model.table)
    with tempfile.TemporaryDirectory() as td:
        _ = explain_neuron(Path(td), 0, 0.0, 1.0)   # 기록 없는 경로에서도 안전해야 한다

    after = eng.read_only_snapshot()
    rng_after = model.rngs.get("diagnostics").bit_generator.state
    for k in before:
        c.expect(bool(np.array_equal(before[k], after[k])),
                 f"측정 전후 {k} 가 비트 단위로 같다")
    c.expect(rng_before == rng_after, "측정이 RNG 스트림 상태를 바꾸지 않는다")
    c.finalize()
    return c


def check_11_zero_correction(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(11, "자유/유도 0교정 비교와 교사 항/감쇠/항상성 분리")
    import copy as _copy

    tcfg = _tiny_cfg(cfg, "conductance_lif")
    tcfg["learning"]["mode"] = "stdp_homeostasis"
    tcfg["learning"]["threshold_adaptation_enabled"] = True

    def run_once(context_current_pA: float) -> dict[str, Any]:
        pop, table, ids = _tiny_network(tcfg, exc_w=3.0)
        # 문맥(L1/apical) 경로가 실제 효과를 갖도록 apical 구획과 결합을 켠다.
        pop.arrays.has_compartment[2, :] = True
        pop.arrays.g_couple_nS[2, COMPARTMENT_INDEX["apical"]] = 10.0
        pop.arrays.g_couple_nS[2, COMPARTMENT_INDEX["basal"]] = 8.0
        table.plasticity_rule[:] = ids.plasticity_rules.id_of("stdp")
        log = EventLog(len(pop), mode="full")
        pop.attach(event_log=log)
        anat = Anatomy(
            population=pop, ids=ids, grid=None, channel_names=[],  # type: ignore[arg-type]
            retina_neuron_id=np.zeros((0, 0), dtype=np.int64))
        plast = StdpHomeostasis(tcfg, table, pop.arrays, ids)
        eng = Engine(tcfg, anat, table, log, plasticity=plast)
        w0 = table.weight.copy()
        th0 = pop.arrays.threshold.copy()
        for s in range(20):
            pop.arrays.Iext_pA[2, COMPARTMENT_INDEX["apical"]] = context_current_pA
            if s % 5 == 0:
                _force_spike(eng, 0, s)
            else:
                eng.set_external_drive(None)
            eng.step()
        return {
            "spikes": int(pop.arrays.spike_count.sum()),
            "dw": table.weight - w0,
            "dtheta": pop.arrays.threshold - th0,
            "teacher_term_total": float(plast.total_weight_delta),
            "theta_term_total": float(plast.total_theta_delta),
        }

    free = run_once(0.0)
    guided_zero = run_once(0.0)
    c.expect(free["spikes"] == guided_zero["spikes"],
             "교정(문맥 전류)이 0 이면 자유 실행과 유도 실행의 발화 수가 같다",
             free=free["spikes"], guided=guided_zero["spikes"])
    c.expect(bool(np.array_equal(free["dw"], guided_zero["dw"])),
             "0교정에서 가중치 변화가 비트 단위로 같다")
    c.expect(bool(np.array_equal(free["dtheta"], guided_zero["dtheta"])),
             "0교정에서 임계값 변화가 비트 단위로 같다")

    guided_nonzero = run_once(300.0)
    c.expect(guided_nonzero["spikes"] != free["spikes"]
             or not np.array_equal(guided_nonzero["dw"], free["dw"]),
             "0 이 아닌 문맥 입력은 실제로 활동/학습에 영향을 준다 "
             "(L1/apical 경로가 측정 가능한 효과를 갖는다)",
             free_spikes=free["spikes"], guided_spikes=guided_nonzero["spikes"])

    c.details = {
        "free": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in free.items()},
        "separation_note_ko": (
            "STDP·감쇠·항상성 항은 교사 신호와 무관하게 동작한다. 0교정 검사는 "
            "'자연 발생 스파이크까지 0' 을 요구하지 않고, 두 조건의 **차이**가 0 인지를 본다."
        ),
    }
    c.finalize()
    return c


def check_12_rao_gradient(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(12, "작은 Rao 연속 모델의 목적함수 미분과 유한차분 비교")
    rngd = np.random.default_rng(12345)
    m = RaoModel(n_modules=2, input_dim=6, n1=4, n2=3, sigma=1.0, sigma_td=2.0,
                 alpha=0.05, lam=0.001, rng=rngd)
    I = rngd.normal(0.0, 1.0, size=(2, 6))
    r1 = rngd.normal(0.0, 0.5, size=(2, 4))
    r2 = rngd.normal(0.0, 0.5, size=3)
    res = m.finite_difference_check(
        I, r1, r2, eps=float(cfg["validation"]["finite_difference_eps"]),
        n_probe=6, rng=rngd)
    tol = float(cfg["validation"]["tolerances"]["rao_gradient_rel"])
    c.expect(res["max_relative_residual"] <= tol,
             f"해석적 갱신 방향과 유한차분 기울기의 상대 잔차가 {tol} 이하다",
             observed=res["max_relative_residual"])

    st = m.settle(I, steps=50, r_step=0.05)
    c.expect(st.energy_trace[-1] <= st.energy_trace[0] + 1e-12,
             "정착 중 목적함수 E 가 증가하지 않는다",
             first=st.energy_trace[0], last=st.energy_trace[-1])
    try:
        RaoModel(1, 2, 2, 2, sigma=-1.0, sigma_td=1.0, alpha=0.0, lam=0.0, rng=rngd)
        c.expect(False, "음수 sigma 를 거부한다")
    except ValueError:
        c.expect(True, "음수 sigma 를 거부한다")
    c.details = {"finite_difference": {k: v for k, v in res.items() if k != "details"},
                 "settle_errors": st.errors,
                 "scope_note_ko": res["scope_note_ko"]}
    c.finalize()
    return c


def check_13_checkpoint(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(13, "체크포인트 재개와 연속 실행의 결과 일치")
    tcfg = _tiny_cfg(cfg, "conductance_lif")

    def build() -> tuple[Engine, NeuronPopulation, SynapseTable]:
        pop, table, ids = _tiny_network(tcfg, exc_w=3.0)
        eng = _tiny_engine(tcfg, pop, table, ids)
        return eng, pop, table

    eng_a, pop_a, _ = build()
    for s in range(20):
        if s % 4 == 0:
            _force_spike(eng_a, 0, s)
        else:
            eng_a.set_external_drive(None)
        eng_a.step()
    final_a = eng_a.state_dict()

    eng_b, pop_b, _ = build()
    for s in range(10):
        if s % 4 == 0:
            _force_spike(eng_b, 0, s)
        else:
            eng_b.set_external_drive(None)
        eng_b.step()
    mid = eng_b.state_dict()

    eng_c, pop_c, _ = build()
    eng_c.load_state_dict(mid)
    for s in range(10, 20):
        if s % 4 == 0:
            _force_spike(eng_c, 0, s)
        else:
            eng_c.set_external_drive(None)
        eng_c.step()
    final_c = eng_c.state_dict()

    same = all(bool(np.array_equal(np.asarray(final_a[k]), np.asarray(final_c[k])))
               for k in ("V_mV", "g_nS", "threshold", "weight", "spike_count"))
    c.expect(same, "체크포인트에서 재개한 결과가 연속 실행과 비트 단위로 같다")
    c.expect(final_a["event_counter"] == final_c["event_counter"],
             "재개 후 event_id 카운터가 일치한다 (중복 기록 방지)")
    c.details = {"continuous_spikes": int(np.sum(final_a["spike_count"])),
                 "resumed_spikes": int(np.sum(final_c["spike_count"]))}
    c.finalize()
    return c


def check_14_import_and_splits(cfg: dict[str, Any], model: Any) -> Check:
    c = Check(14, "import 부작용 없음·test 접근 카운터·데이터 분할 중복 없음")
    here = Path(__file__).resolve()
    with tempfile.TemporaryDirectory() as td:
        before = set(os.listdir(td))
        # 단일 파일 판: 이 파일을 경로로 import 해서 부작용이 없는지 본다.
        code = ("import importlib.util, sys; "
                "spec = importlib.util.spec_from_file_location("
                "'cortex_single_check', r'%s'); "
                "m = importlib.util.module_from_spec(spec); "
                "sys.modules['cortex_single_check'] = m; "
                "spec.loader.exec_module(m)" % str(here))
        proc = subprocess.run([sys.executable, "-c", code], cwd=td,
                              capture_output=True, text=True, timeout=180, check=False)
        after = set(os.listdir(td))
        c.expect(proc.returncode == 0, "단일 파일 전체가 import 된다",
                 stderr=proc.stderr[-500:])
        c.expect(after == before,
                 "import 만으로 작업 디렉터리에 파일이 생기지 않는다",
                 created=sorted(after - before))

    rngd = np.random.default_rng(7)
    stims = generate_stimuli(cfg, rngd)
    if stims:
        splits = split_stimuli(stims, cfg, np.random.default_rng(8))
        rep = split_report(splits)
        c.expect(rep["no_overlap"], "train/dev/test 가 겹치지 않는다 (base_id 기준)",
                 overlaps=rep["overlaps"])
        c.details["split_report"] = rep
    else:
        c.expect(True, "이 설정에는 자극이 정의되어 있지 않아 분할 검사를 건너뛴다 "
                       "(구조상 정상)")

    c.details["test_access_counter_ko"] = (
        "experiment 명령은 manifest 의 test_access.n_test_evaluations 를 증가시킨다. "
        "이미 본 시험 결과로 재선택하면 탐색 결과로 표시해야 한다.")
    c.finalize()
    return c


# ======================================================================
CHECKS: list[Callable[[dict[str, Any], Any], Check]] = [
    check_01_records, check_02_event_application, check_03_delay,
    check_04_sum_threshold, check_05_lif, check_06_dale, check_07_input,
    check_08_response_changes, check_09_structure,
    check_10_measurement_noninvasive, check_11_zero_correction,
    check_12_rao_gradient, check_13_checkpoint, check_14_import_and_splits,
]


def run_all(cfg: dict[str, Any], recorder: Any, package_root: Path,
            progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """검사 1~14 를 실행하고 결과를 돌려준다 (사용자 명령에서만 호출)."""

    progress = progress or (lambda m: None)
    rngs = rng_from_config(cfg)
    progress("검사용 모델을 조립하는 중...")
    model = build_model(cfg, rngs)
    model.rngs = rngs
    model.encoder.fit_normalization(
        [uniform_image(int(cfg["retina"]["image"]["max_side_px"]),
                                   int(cfg["retina"]["image"]["max_side_px"]), 0.5),
         bar_image(int(cfg["retina"]["image"]["max_side_px"]),
                               int(cfg["retina"]["image"]["max_side_px"]),
                               0.5 * cfg["retina"]["image"]["max_side_px"],
                               0.5 * cfg["retina"]["image"]["max_side_px"],
                               0.5 * cfg["retina"]["image"]["max_side_px"], 3.0, 0.0)],
        source="validation_builtin")

    results: list[dict[str, Any]] = []
    n_pass = n_fail = n_skip = 0
    for fn in CHECKS:
        name = fn.__name__
        progress(f"검사 실행: {name}")
        try:
            chk = fn(cfg, model)
        except Exception as exc:  # 검사 자체가 실패해도 기록한다
            chk = Check(0, name)
            chk.status = "failed"
            chk.reason = f"{type(exc).__name__}: {exc}"
            recorder.error(f"검사 {name} 에서 예외", exc)
        d = chk.to_dict()
        results.append(d)
        n_pass += int(d["status"] == "passed")
        n_fail += int(d["status"] == "failed")
        n_skip += int(d["status"] == "skipped")
        recorder.log(f"[{d['status'].upper()}] {d['id']:02d} {d['name']}")

    return {
        "n_checks": len(results), "n_passed": n_pass, "n_failed": n_fail,
        "n_skipped": n_skip,
        "all_passed": n_fail == 0 and n_skip == 0,
        "note_ko": ("skipped 는 passed 에 포함하지 않는다. 성공률을 맞추려고 기준을 "
                    "사후에 바꾸지 않는다."),
        "checks": results,
    }


# ============================================================================
# 섹션: cli  —  명령행 인터페이스
#   (원래 파일: cortex/cli.py)
# ============================================================================

"""cli.py -- 반복 작업용 명령행 인터페이스.

명세 14절. **수치 실험 명령은 기본이 dry-run** 이고 사용자가 ``--execute`` 를
명시할 때만 실행한다. 조회 명령(``report``, ``explain-neuron``, ``figures``)은
기존 기록만 읽으며 새 실험을 자동 실행하지 않는다.

메뉴(``run.py``)와 이 CLI 는 **같은 Runner/Recorder** 를 호출한다. 시뮬레이션
구현을 두 벌 만들지 않는다.

    python -m cortex.cli inspect-config --config configs/v1_small.json
    python -m cortex.cli validate       --config configs/minimal.json --execute
    python -m cortex.cli simulate       --config configs/v1_small.json --execute
    python -m cortex.cli experiment     --config configs/hierarchy_small.json --execute
    python -m cortex.cli resume         --run-dir runs/RUN_ID --execute
    python -m cortex.cli report         --run-dir runs/RUN_ID
    python -m cortex.cli explain-neuron --run-dir runs/RUN_ID --neuron-id 12                                         --from-ms 0 --to-ms 100
    python -m cortex.cli figures        --run-dir runs/RUN_ID
"""

PROJECT_ROOT = Path(__file__).resolve().parent
#: 단일 파일 판에서는 이 파일 자체가 코드 전체다 (code_hash 의 대상).
PACKAGE_ROOT = Path(__file__).resolve()
DEFAULT_RUNS = PROJECT_ROOT / "runs"

#: --config 인자 도움말 (내장 설정 이름 또는 JSON 경로)
CONFIG_HELP = ("내장 설정 이름(minimal, v1_small, hierarchy_small, "
               "megapixel_input) 또는 설정 JSON 파일 경로")


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def _load_cfg(path: str) -> dict[str, Any]:

    try:
        return load_config_by_name_or_path(path)
    except ConfigError as exc:
        print(f"[설정 오류] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _progress(msg: str) -> None:
    print(f"  … {msg}", flush=True)


def _runner(cfg: dict[str, Any], run_dir: Path | None, command: str,
            execute: bool) -> Any:

    return ExperimentRunner(cfg, run_dir, command, PACKAGE_ROOT, execute=execute,
                            progress=_progress)


def _resolve_run_dir(args: argparse.Namespace, tag: str) -> Path:

    if getattr(args, "run_dir", None):
        return Path(args.run_dir)
    root = Path(args.runs_root) if getattr(args, "runs_root", None) else DEFAULT_RUNS
    return new_run_dir(root, tag)


# ----------------------------------------------------------------------
def cmd_inspect_config(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    info = _runner(cfg, None, " ".join(sys.argv), False).inspect()
    _print_json(info)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    run_dir = _resolve_run_dir(args, f"validate_{cfg['meta']['name']}")
    r = _runner(cfg, run_dir, " ".join(sys.argv), args.execute)
    if not args.execute:
        print("[dry-run] 실행하지 않았다. 실제 검사를 돌리려면 --execute 를 붙여라.")
        _print_json(r.validate())
        return 0
    print(f"검사 실행 중... 결과는 {run_dir} 에 저장된다. 중단: Ctrl+C")
    result = r.validate()
    print(f"\n검사 {result['n_checks']}건: 통과 {result['n_passed']} / "
          f"실패 {result['n_failed']} / 건너뜀 {result['n_skipped']}")
    for c in result["checks"]:
        print(f"  [{c['status'].upper():7s}] {c['id']:02d} {c['name']}")
        if c["status"] != "passed":
            for a in c["assertions"]:
                if not a["passed"]:
                    print(f"      - 실패: {a['assertion']}")
            if c["reason"]:
                print(f"      - 사유: {c['reason']}")
    print(f"\n결과 파일: {run_dir / 'validation.json'}")
    return 0 if result["n_failed"] == 0 else 1


def cmd_simulate(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    run_dir = _resolve_run_dir(args, f"simulate_{cfg['meta']['name']}")
    r = _runner(cfg, run_dir, " ".join(sys.argv), args.execute)
    if not args.execute:
        print("[dry-run] 규모만 계산했다. 실행하려면 --execute 를 붙여라.")
        _print_json(r.simulate())
        return 0
    print(f"시뮬레이션 실행 중... 결과: {run_dir}  (중단: Ctrl+C)")
    _print_json(r.simulate())
    print(f"완료. 기록 위치: {run_dir}")
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config)
    run_dir = _resolve_run_dir(args, f"experiment_{cfg['meta']['name']}")
    r = _runner(cfg, run_dir, " ".join(sys.argv), args.execute)
    if not args.execute:
        print("[dry-run] 규모만 계산했다. 실행하려면 --execute 를 붙여라.")
        _print_json(r.experiment())
        return 0
    print(f"실험 실행 중... 결과: {run_dir}  (중단: Ctrl+C)")
    _print_json(r.experiment())
    print(f"완료. 기록 위치: {run_dir}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:

    run_dir = Path(args.run_dir)
    try:
        man = load_manifest(run_dir)
    except RecordingError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2
    cfg = man["config"]
    r = _runner(cfg, run_dir, " ".join(sys.argv), args.execute)
    try:
        out = r.resume(Path(args.checkpoint) if args.checkpoint else None,
                       allow_mismatch=args.allow_mismatch)
    except RecordingError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2
    _print_json(out)
    return 0


def cmd_report(args: argparse.Namespace) -> int:

    try:
        path = make_report(Path(args.run_dir))
    except RecordingError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2
    print(f"보고서 생성: {path}")
    check = verify_report_matches_records(Path(args.run_dir))
    print(f"보고서-기록 일치 검사: {check}")
    return 0


def cmd_explain_neuron(args: argparse.Namespace) -> int:

    out = explain_neuron(Path(args.run_dir), args.neuron_id, args.from_ms, args.to_ms,
                         top_k=args.top_k)
    _print_json(out)
    return 0 if out.get("found") else 1


def cmd_figures(args: argparse.Namespace) -> int:

    run_dir = Path(args.run_dir)
    try:
        load_manifest(run_dir)
    except RecordingError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2
    model = None
    if args.rebuild_model:
        man = load_manifest(run_dir)
        model = build_model(man["config"], rng_from_config(man["config"]))
    nids = [int(x) for x in args.neuron_ids.split(",")] if args.neuron_ids else []
    made = make_all(run_dir, model, nids)
    print(f"그림 {len(made)}개 생성:")
    for p in made:
        print(f"  {p}")
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    root = Path(args.runs_root) if args.runs_root else DEFAULT_RUNS
    if not root.is_dir():
        print(f"실행 기록 폴더가 없다: {root}")
        return 0
    rows = []
    for d in sorted(root.iterdir()):
        man = d / "manifest.json"
        if man.is_file():
            m = json.loads(man.read_text(encoding="utf-8"))
            rows.append((d.name, m.get("status"), m.get("started_utc"),
                         m.get("config", {}).get("meta", {}).get("name")))
    if not rows:
        print(f"{root} 안에 실행 기록이 없다.")
        return 0
    print(f"{'run_id':45s} {'status':12s} {'started(UTC)':22s} config")
    for r in rows:
        print(f"{r[0]:45s} {str(r[1]):12s} {str(r[2]):22s} {r[3]}")
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        description="3x3 뉴런 기록 구조 시각피질 시뮬레이터 CLI "
                    "(수치 실험은 --execute 를 붙여야 실행된다)")
    sub = p.add_subparsers(dest="command", required=True)

    def add_cfg(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", required=True, help=CONFIG_HELP)
        sp.add_argument("--run-dir", default=None, help="출력 폴더 (기본: runs/<자동>)")
        sp.add_argument("--runs-root", default=None, help="runs 루트 폴더")
        sp.add_argument("--execute", action="store_true",
                        help="실제로 실행한다 (없으면 규모만 계산하는 dry-run)")

    sp = sub.add_parser("inspect-config", help="설정과 규모만 확인 (실행하지 않음)")
    sp.add_argument("--config", required=True, help=CONFIG_HELP)
    sp.set_defaults(func=cmd_inspect_config)

    sp = sub.add_parser("validate", help="필수 검증 1~14 실행")
    add_cfg(sp)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("simulate", help="자극 제시 시뮬레이션")
    add_cfg(sp)
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("experiment", help="train/dev/test 분할 실험")
    add_cfg(sp)
    sp.set_defaults(func=cmd_experiment)

    sp = sub.add_parser("resume", help="중단된 실행 재개")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--allow-mismatch", action="store_true",
                    help="코드/설정 해시가 달라도 재개 (차이를 인지한 경우에만)")
    sp.add_argument("--execute", action="store_true")
    sp.set_defaults(func=cmd_resume)

    sp = sub.add_parser("report", help="기존 기록에서 한국어 보고서 생성")
    sp.add_argument("--run-dir", required=True)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("explain-neuron", help="특정 뉴런의 입력·발화·출력 조회")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--neuron-id", type=int, required=True)
    sp.add_argument("--from-ms", type=float, default=0.0)
    sp.add_argument("--to-ms", type=float, default=float("inf"))
    sp.add_argument("--top-k", type=int, default=20)
    sp.set_defaults(func=cmd_explain_neuron)

    sp = sub.add_parser("figures", help="기존 기록에서 그림 생성")
    sp.add_argument("--run-dir", required=True)
    sp.add_argument("--neuron-ids", default="", help="쉼표 구분 뉴런 ID")
    sp.add_argument("--rebuild-model", action="store_true",
                    help="manifest 의 설정으로 모델을 다시 조립해 배치/지도 그림도 만든다")
    sp.set_defaults(func=cmd_figures)

    sp = sub.add_parser("list-runs", help="실행 기록 목록")
    sp.add_argument("--runs-root", default=None)
    sp.set_defaults(func=cmd_list_runs)
    return p


def cli_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 Ctrl+C 로 중단했다. "
              "기록은 runs/<run_id>/ 에 저장되어 있고 resume 으로 재개할 수 있다.")
        return 130
    except FileNotFoundError as exc:
        print(f"[오류] 파일을 찾을 수 없다: {exc}", file=sys.stderr)
        return 2

# ============================================================================
# 섹션: builtin configs  —  내장 설정 4종 (JSON 파일 없이 실행 가능)
#   (원래 파일: configs/minimal.json, v1_small.json, hierarchy_small.json,
#               megapixel_input.json)
# ============================================================================
"""내장 설정.

원래 저장소의 `configs/*.json` 과 **해석 결과가 동일**하도록 코드로 구성했다.
(동일성은 `python cortex_all_in_one.py selftest` 로 확인할 수 있다. JSON 파일이
옆에 있을 때만 비교하며, 없으면 건너뛴다.)

값은 전부 **모형 파라미터**다. 출처가 있는 측정값이 아니다
(BIOLOGY_AND_ASSUMPTIONS.md 참조).
"""

_CELL_TYPES: dict[str, Any] = {
    "retinal_ganglion": {
        "dale": "excitatory", "compartments": ["soma"],
        "C_pF": {"soma": 80.0, "basal": 100.0, "apical": 100.0},
        "gL_nS": {"soma": 6.0, "basal": 5.0, "apical": 5.0},
        "EL_mV": {"soma": -65.0, "basal": -70.0, "apical": -70.0},
        "V_th_mV": -50.0, "V_reset_mV": -62.0, "t_ref_ms": 1.5,
        "target_rate_hz": 10.0, "sum_threshold_theta": 0.5,
    },
    "lgn_relay": {
        "dale": "excitatory", "compartments": ["soma"],
        "C_pF": {"soma": 100.0, "basal": 100.0, "apical": 100.0},
        "gL_nS": {"soma": 7.0, "basal": 5.0, "apical": 5.0},
        "EL_mV": {"soma": -68.0, "basal": -70.0, "apical": -70.0},
        "V_th_mV": -50.0, "V_reset_mV": -63.0, "t_ref_ms": 2.0,
        "target_rate_hz": 8.0, "sum_threshold_theta": 0.6,
    },
    "spiny_stellate": {
        "dale": "excitatory", "compartments": ["soma", "basal"],
        "C_pF": {"soma": 150.0, "basal": 90.0, "apical": 90.0},
        "gL_nS": {"soma": 8.0, "basal": 5.0, "apical": 5.0},
        "EL_mV": {"soma": -70.0, "basal": -70.0, "apical": -70.0},
        "g_couple_nS": {"soma_basal": 10.0, "soma_apical": 0.0},
        "V_th_mV": -50.0, "V_reset_mV": -65.0, "t_ref_ms": 2.0,
        "target_rate_hz": 5.0, "sum_threshold_theta": 1.0,
    },
    "pyramidal": {
        "dale": "excitatory", "compartments": ["soma", "basal", "apical"],
        "C_pF": {"soma": 200.0, "basal": 120.0, "apical": 120.0},
        "gL_nS": {"soma": 10.0, "basal": 6.0, "apical": 6.0},
        "EL_mV": {"soma": -70.0, "basal": -70.0, "apical": -70.0},
        "g_couple_nS": {"soma_basal": 9.0, "soma_apical": 4.0},
        "V_th_mV": -50.0, "V_reset_mV": -65.0, "t_ref_ms": 2.5,
        "target_rate_hz": 4.0, "sum_threshold_theta": 1.0,
    },
    "pv_basket": {
        "dale": "inhibitory", "compartments": ["soma"],
        "C_pF": {"soma": 100.0, "basal": 80.0, "apical": 80.0},
        "gL_nS": {"soma": 10.0, "basal": 5.0, "apical": 5.0},
        "EL_mV": {"soma": -65.0, "basal": -70.0, "apical": -70.0},
        "V_th_mV": -48.0, "V_reset_mV": -60.0, "t_ref_ms": 1.0,
        "target_rate_hz": 12.0, "sum_threshold_theta": 0.8,
    },
    "sst_martinotti": {
        "dale": "inhibitory", "compartments": ["soma"],
        "C_pF": {"soma": 110.0, "basal": 80.0, "apical": 80.0},
        "gL_nS": {"soma": 8.0, "basal": 5.0, "apical": 5.0},
        "EL_mV": {"soma": -66.0, "basal": -70.0, "apical": -70.0},
        "V_th_mV": -50.0, "V_reset_mV": -62.0, "t_ref_ms": 2.0,
        "target_rate_hz": 6.0, "sum_threshold_theta": 0.9,
    },
    "l1_inhibitory": {
        "dale": "inhibitory", "compartments": ["soma"],
        "C_pF": {"soma": 90.0, "basal": 80.0, "apical": 80.0},
        "gL_nS": {"soma": 8.0, "basal": 5.0, "apical": 5.0},
        "EL_mV": {"soma": -66.0, "basal": -70.0, "apical": -70.0},
        "V_th_mV": -50.0, "V_reset_mV": -62.0, "t_ref_ms": 2.0,
        "target_rate_hz": 5.0, "sum_threshold_theta": 0.9,
    },
}

_FF_NOTE = ("상향은 주로 상부층(L2/3) 기원, 중간층(L4) 표적이라는 집단 수준 경향을 "
            "쓴다. 개별 시냅스의 완전한 목록이 아니며 기원/종말층 혼합을 허용한다 "
            "(Markov 등, Anatomy of hierarchy).")
_FB_NOTE = ("하향은 L4 밖(주로 L1/apical, L5)을 표적으로 한다. 같은 근거의 집단 수준 "
            "경향이며 예외를 금지하는 규칙이 아니다.")


def _rule(name, src_area, src_layers, dst_area, dst_layers, *, comp="basal",
          receptor="AMPA", kind="rf_knn", k=8, radius=0.4, sigma=0.5, prob=0.5,
          median=1.0, wsigma=0.3, wmax=6.0, vel=0.3, delay=0.8,
          src_types=None, dst_types=None, plastic="none", gabor=False,
          enabled=None, note="") -> dict[str, Any]:
    """배선 규칙 하나를 만든다 (설정 dict). 해석은 resolve_config 가 한다."""
    r: dict[str, Any] = {"name": name}
    if enabled is not None:
        r["enabled"] = bool(enabled)
    r.update({
        "src": {"area": src_area, "layer": list(src_layers),
                "cell_type": list(src_types or [])},
        "dst": {"area": dst_area, "layer": list(dst_layers),
                "cell_type": list(dst_types or [])},
        "target_compartment": comp, "receptor": receptor, "rule": kind,
        "k": k, "radius_mm": radius, "rf_match_sigma_deg": sigma,
        "probability": prob,
        "weight": {"dist": "lognormal", "median": median, "sigma": wsigma,
                   "min": 0.0, "max": wmax},
        "conduction_velocity_mm_per_ms": vel, "synaptic_delay_ms": delay,
        "plasticity_rule": plastic, "gabor_initialized": gabor, "note": note,
    })
    return r


def _cortical_area(level, extent, counts, max_ecc, rf_sigma, growth,
                   note="") -> dict[str, Any]:
    return {
        "kind": "cortex", "hierarchy_level": level,
        "surface_extent_mm": list(extent),
        "layer_thickness_mm": {"L1": 0.10, "L2": 0.22, "L3": 0.33,
                               "L4": 0.30, "L5": 0.38, "L6": 0.37},
        "neurons_per_layer": dict(counts),
        "cell_type_fractions": {
            "L1": {"l1_inhibitory": 1.0},
            "L2": {"pyramidal": 0.72, "pv_basket": 0.18, "sst_martinotti": 0.10},
            "L3": {"pyramidal": 0.72, "pv_basket": 0.18, "sst_martinotti": 0.10},
            "L4": {"spiny_stellate": 0.72, "pv_basket": 0.20, "sst_martinotti": 0.08},
            "L5": {"pyramidal": 0.82, "pv_basket": 0.18},
            "L6": {"pyramidal": 0.82, "pv_basket": 0.18},
        },
        "visual_field": {"max_ecc_deg": max_ecc, "rf_sigma_deg": rf_sigma,
                         "rf_sigma_growth_per_deg": growth},
        "notes": note,
    }


def _local_microcircuit(area: str, scale: float = 1.0,
                        enabled=None) -> list[dict[str, Any]]:
    """한 피질 영역 내부의 층간·억제·재귀 배선 (모형 규칙)."""
    e = enabled
    return [
        _rule(f"{area}_L4->L2L3", area, ["L4"], area, ["L2", "L3"],
              kind="local_radius", radius=0.45, k=8, prob=0.5,
              median=1.1 * scale, vel=0.2, delay=0.8,
              src_types=["spiny_stellate"], dst_types=["pyramidal"],
              plastic="stdp", enabled=e, note="L4 -> L2/3 상행"),
        _rule(f"{area}_L4->L4_PV", area, ["L4"], area, ["L4"],
              comp="soma", kind="local_radius", radius=0.35, k=6, prob=0.55,
              median=1.0 * scale, vel=0.2, delay=0.6, enabled=e,
              src_types=["spiny_stellate"], dst_types=["pv_basket"]),
        _rule(f"{area}_L4_PV->L4", area, ["L4"], area, ["L4"],
              comp="soma", receptor="GABA_A", kind="local_radius",
              radius=0.35, k=6, prob=0.65, median=1.4 * scale, vel=0.2,
              delay=0.5, src_types=["pv_basket"], dst_types=["spiny_stellate"],
              enabled=e, note="PV 는 soma 표적 억제"),
        _rule(f"{area}_L2L3_lateral", area, ["L2", "L3"], area, ["L2", "L3"],
              kind="local_radius", radius=0.6, k=6, prob=0.3,
              median=0.7 * scale, vel=0.15, delay=1.4,
              src_types=["pyramidal"], dst_types=["pyramidal"],
              plastic="stdp", enabled=e, note="수평 재귀 연결"),
        _rule(f"{area}_L2L3->PV", area, ["L2", "L3"], area, ["L2", "L3"],
              comp="soma", kind="local_radius", radius=0.5, k=5, prob=0.5,
              median=0.9 * scale, vel=0.2, delay=0.6, enabled=e,
              src_types=["pyramidal"], dst_types=["pv_basket"]),
        _rule(f"{area}_PV->L2L3", area, ["L2", "L3"], area, ["L2", "L3"],
              comp="soma", receptor="GABA_A", kind="local_radius",
              radius=0.5, k=6, prob=0.6, median=1.5 * scale, vel=0.2,
              delay=0.5, src_types=["pv_basket"], dst_types=["pyramidal"],
              enabled=e),
        _rule(f"{area}_L2L3->SST", area, ["L2", "L3"], area, ["L2", "L3"],
              comp="soma", kind="local_radius", radius=0.5, k=4, prob=0.4,
              median=0.7 * scale, vel=0.2, delay=0.7, enabled=e,
              src_types=["pyramidal"], dst_types=["sst_martinotti"]),
        _rule(f"{area}_SST->apical", area, ["L2", "L3"], area, ["L2", "L3"],
              comp="apical", receptor="GABA_A", kind="local_radius",
              radius=0.7, k=5, prob=0.5, median=1.0 * scale, vel=0.15,
              delay=0.8, src_types=["sst_martinotti"], dst_types=["pyramidal"],
              enabled=e, note="SST 는 첨단수상돌기(apical) 표적 억제"),
        _rule(f"{area}_L2L3->L5", area, ["L2", "L3"], area, ["L5"],
              kind="local_radius", radius=0.6, k=5, prob=0.45,
              median=0.9 * scale, vel=0.2, delay=0.9, enabled=e,
              src_types=["pyramidal"], dst_types=["pyramidal"]),
        _rule(f"{area}_L5->L6", area, ["L5"], area, ["L6"],
              kind="local_radius", radius=0.6, k=4, prob=0.45,
              median=0.8 * scale, vel=0.2, delay=0.9, enabled=e,
              src_types=["pyramidal"], dst_types=["pyramidal"]),
        _rule(f"{area}_L6->L4", area, ["L6"], area, ["L4"],
              kind="local_radius", radius=0.6, k=4, prob=0.35,
              median=0.5 * scale, vel=0.2, delay=1.1, enabled=e,
              src_types=["pyramidal"], dst_types=["spiny_stellate"],
              note="층내 피드백 (L6 -> L4)"),
        _rule(f"{area}_L5->L1_inh", area, ["L5"], area, ["L1"],
              comp="soma", kind="local_radius", radius=0.9, k=3, prob=0.45,
              median=0.7 * scale, vel=0.2, delay=1.0, enabled=e,
              src_types=["pyramidal"], dst_types=["l1_inhibitory"],
              note="L1 은 세포체 밀도가 낮지만 비어 있지 않다"),
        _rule(f"{area}_L1_inh->apical", area, ["L1"], area, ["L2", "L3", "L5"],
              comp="apical", receptor="GABA_A", kind="local_radius",
              radius=0.9, k=6, prob=0.5, median=0.9 * scale, vel=0.15,
              delay=0.8, src_types=["l1_inhibitory"], dst_types=["pyramidal"],
              enabled=e, note="L1 억제 -> 피라미드 apical 구획"),
    ]


def _feedforward(a: str, b: str, *, sigma: float, prob: float = 0.35,
                 median: float = 1.0, k: int = 10, delay: float = 2.0,
                 plastic: str = "stdp") -> list[dict[str, Any]]:
    """a(하위) -> b(상위) 상향 경로."""
    return [
        _rule(f"{a}->{b}_FF_main", a, ["L2", "L3"], b, ["L4"], k=k, sigma=sigma,
              prob=prob, median=median, vel=0.5, delay=delay, enabled=True,
              src_types=["pyramidal"], dst_types=["spiny_stellate"],
              plastic=plastic, note=_FF_NOTE),
        _rule(f"{a}->{b}_FF_mixed", a, ["L5"], b, ["L4", "L3"], k=5,
              sigma=sigma * 1.2, prob=prob * 0.4, median=median * 0.5, vel=0.5,
              delay=delay * 1.1, enabled=True, src_types=["pyramidal"],
              dst_types=["spiny_stellate", "pyramidal"],
              note="기원/종말층 혼합을 허용하는 소수 경로."),
        _rule(f"{a}->{b}_FF_inh", a, ["L2", "L3"], b, ["L4"], comp="soma", k=6,
              sigma=sigma, prob=prob * 0.6, median=median * 0.8, vel=0.5,
              delay=delay, enabled=True, src_types=["pyramidal"],
              dst_types=["pv_basket"], note="상향 경로에 딸린 전방향 억제."),
    ]


def _feedback(a: str, b: str, *, sigma: float, prob: float = 0.25,
              median: float = 0.5, k: int = 8,
              delay: float = 3.0) -> list[dict[str, Any]]:
    """b(상위) -> a(하위) 하향. L4 를 피해 apical/L5 를 표적으로 한다."""
    return [
        _rule(f"{b}->{a}_FB_apical", b, ["L5", "L6"], a, ["L2", "L3"],
              comp="apical", k=k, sigma=sigma, prob=prob, median=median,
              vel=0.5, delay=delay, enabled=True, src_types=["pyramidal"],
              dst_types=["pyramidal"], note=_FB_NOTE),
        _rule(f"{b}->{a}_FB_L1", b, ["L5", "L6"], a, ["L1"], comp="soma", k=4,
              sigma=sigma * 1.2, prob=prob * 0.6, median=median * 0.8, vel=0.5,
              delay=delay, enabled=True, src_types=["pyramidal"],
              dst_types=["l1_inhibitory"], note="L1 억제 뉴런을 경유하는 하향 경로."),
        _rule(f"{b}->{a}_FB_L5", b, ["L5", "L6"], a, ["L5"], k=4,
              sigma=sigma * 1.2, prob=prob * 0.6, median=median * 0.7, vel=0.5,
              delay=delay * 1.1, enabled=True, src_types=["pyramidal"],
              dst_types=["pyramidal"], note=_FB_NOTE),
    ]



def _config_minimal() -> dict[str, Any]:
    """configs/minimal.json 과 동일한 내용."""
    return {
    "meta": {
        "name": "minimal",
        "description": "소수 뉴런의 단일 전달·억제·지연·분기 확인용 최소 설정. sum_threshold 모드를 끝까지 실행한다.",
        "species_assumption": "primate_visual_cortex_model",
        "notes": [
            "뉴런 수·층 두께·연결 확률은 모형 파라미터다. 출처가 있는 측정값이 아니다.",
            "sum_threshold 는 무차원 합산 모델이며 생물학적 막전위 모델이 아니다.",
        ],
    },
    "seeds": {"master": 20240101},
    "engine": {
        "mode": "sum_threshold",
        "dt_ms": 1.0,
        "duration_ms": 60.0,
        "min_delay_steps": 1,
        "delay_rounding": "ceil",
        "weight_application": "emit",
        "sum_threshold_interval_steps": 1,
        "reset_between_samples": {
            "voltages": True,
            "conductances": True,
            "event_queue": True,
            "traces": True,
            "thresholds": False,
            "weights": False,
            "input_log": False,
        },
        "sequence_mode": False,
    },
    "cell_types": {
        "retinal_ganglion": {
            "dale": "excitatory",
            "compartments": ["soma"],
            "C_pF": {"soma": 80.0, "basal": 100.0, "apical": 100.0},
            "gL_nS": {"soma": 6.0, "basal": 5.0, "apical": 5.0},
            "EL_mV": {"soma": -65.0, "basal": -70.0, "apical": -70.0},
            "V_th_mV": -50.0,
            "V_reset_mV": -62.0,
            "t_ref_ms": 1.5,
            "target_rate_hz": 10.0,
            "sum_threshold_theta": 0.35,
        },
        "lgn_relay": {
            "dale": "excitatory",
            "compartments": ["soma"],
            "C_pF": {"soma": 100.0, "basal": 100.0, "apical": 100.0},
            "gL_nS": {"soma": 7.0, "basal": 5.0, "apical": 5.0},
            "EL_mV": {"soma": -68.0, "basal": -70.0, "apical": -70.0},
            "V_th_mV": -50.0,
            "V_reset_mV": -63.0,
            "t_ref_ms": 2.0,
            "target_rate_hz": 8.0,
            "sum_threshold_theta": 0.45,
        },
        "spiny_stellate": {
            "dale": "excitatory",
            "compartments": ["soma", "basal"],
            "C_pF": {"soma": 150.0, "basal": 90.0, "apical": 90.0},
            "gL_nS": {"soma": 8.0, "basal": 5.0, "apical": 5.0},
            "EL_mV": {"soma": -70.0, "basal": -70.0, "apical": -70.0},
            "g_couple_nS": {"soma_basal": 10.0, "soma_apical": 0.0},
            "V_th_mV": -50.0,
            "V_reset_mV": -65.0,
            "t_ref_ms": 2.0,
            "target_rate_hz": 5.0,
            "sum_threshold_theta": 0.8,
        },
        "pyramidal": {
            "dale": "excitatory",
            "compartments": ["soma", "basal", "apical"],
            "C_pF": {"soma": 200.0, "basal": 120.0, "apical": 120.0},
            "gL_nS": {"soma": 10.0, "basal": 6.0, "apical": 6.0},
            "EL_mV": {"soma": -70.0, "basal": -70.0, "apical": -70.0},
            "g_couple_nS": {"soma_basal": 9.0, "soma_apical": 4.0},
            "V_th_mV": -50.0,
            "V_reset_mV": -65.0,
            "t_ref_ms": 2.5,
            "target_rate_hz": 4.0,
            "sum_threshold_theta": 0.9,
        },
        "pv_basket": {
            "dale": "inhibitory",
            "compartments": ["soma"],
            "C_pF": {"soma": 100.0, "basal": 80.0, "apical": 80.0},
            "gL_nS": {"soma": 10.0, "basal": 5.0, "apical": 5.0},
            "EL_mV": {"soma": -65.0, "basal": -70.0, "apical": -70.0},
            "V_th_mV": -48.0,
            "V_reset_mV": -60.0,
            "t_ref_ms": 1.0,
            "target_rate_hz": 12.0,
            "sum_threshold_theta": 0.6,
        },
        "l1_inhibitory": {
            "dale": "inhibitory",
            "compartments": ["soma"],
            "C_pF": {"soma": 90.0, "basal": 80.0, "apical": 80.0},
            "gL_nS": {"soma": 8.0, "basal": 5.0, "apical": 5.0},
            "EL_mV": {"soma": -66.0, "basal": -70.0, "apical": -70.0},
            "V_th_mV": -50.0,
            "V_reset_mV": -62.0,
            "t_ref_ms": 2.0,
            "target_rate_hz": 5.0,
            "sum_threshold_theta": 0.8,
        },
    },
    "retina": {
        "image": {"max_side_px": 64, "fov_deg": 12.0, "input_colorspace": "srgb"},
        "channels": {"on_off_split": True, "keep_lowpass_lms": False, "lowpass_sigma_px": 6.0},
        "dog": {
            "center_sigma_px": 1.0,
            "surround_sigma_px": 3.0,
            "truncate": 4.0,
            "boundary_mode": "reflect",
            "normalize_each_kernel_to_unit_sum": True,
        },
        "drive": {
            "mode": "rate",
            "max_rate_hz": 50.0,
            "gain": 1.0,
            "baseline_rate_hz": 0.0,
            "current_per_hz_pA": 2.0,
            "sum_mode_scale": 1.0,
        },
    },
    "retinotopy": {
        "mapping": "log_polar",
        "e0_deg": 0.5,
        "n_radial": 2,
        "n_angular": 4,
        "fovea_patch": {"enabled": True, "radius_deg": 0.5, "grid": 2},
        "sampling": {
            "lowpass_before_sampling": True,
            "sigma_scale": 0.5,
            "min_sigma_px": 0.5,
            "interpolation_order": 1,
        },
    },
    "anatomy": {
        "area_gap_mm": 1.0,
        "jitter_mm": 0.01,
        "areas": {
            "Retina": {
                "kind": "retina",
                "surface_extent_mm": [1.0, 1.0],
                "neurons_per_layer": {"L_input": 0},
                "cell_type_fractions": {"L_input": {"retinal_ganglion": 1.0}},
                "visual_field": {"max_ecc_deg": 5.0, "rf_sigma_deg": 0.25, "rf_sigma_growth_per_deg": 0.06},
                "notes": "뉴런 수는 (채널 수 x 시야 샘플 수)로 유도된다.",
            },
            "LGN": {
                "kind": "thalamus",
                "surface_extent_mm": [1.0, 1.0],
                "neurons_per_layer": {"L_relay": 24},
                "cell_type_fractions": {"L_relay": {"lgn_relay": 1.0}},
                "visual_field": {"max_ecc_deg": 5.0, "rf_sigma_deg": 0.3, "rf_sigma_growth_per_deg": 0.07},
            },
            "V1": {
                "kind": "cortex",
                "hierarchy_level": 1.0,
                "surface_extent_mm": [1.5, 1.5],
                "layer_thickness_mm": {"L1": 0.1, "L2": 0.2, "L3": 0.3, "L4": 0.3, "L5": 0.35, "L6": 0.35},
                "neurons_per_layer": {"L1": 4, "L2": 12, "L3": 12, "L4": 24, "L5": 8, "L6": 8},
                "cell_type_fractions": {
                    "L1": {"l1_inhibitory": 1.0},
                    "L2": {"pyramidal": 0.75, "pv_basket": 0.25},
                    "L3": {"pyramidal": 0.75, "pv_basket": 0.25},
                    "L4": {"spiny_stellate": 0.75, "pv_basket": 0.25},
                    "L5": {"pyramidal": 1.0},
                    "L6": {"pyramidal": 1.0},
                },
                "visual_field": {"max_ecc_deg": 5.0, "rf_sigma_deg": 0.35, "rf_sigma_growth_per_deg": 0.08},
            },
        },
    },
    "wiring": {
        "max_total_synapses": 200000,
        "rules": [
            {
                "name": "Retina->LGN",
                "src": {"area": "Retina", "layer": ["L_input"]},
                "dst": {"area": "LGN", "layer": ["L_relay"]},
                "target_compartment": "soma",
                "receptor": "AMPA",
                "rule": "rf_knn",
                "k": 6,
                "rf_match_sigma_deg": 0.8,
                "probability": 0.8,
                "weight": {"dist": "lognormal", "median": 0.45, "sigma": 0.25, "min": 0.0, "max": 3.0},
                "conduction_velocity_mm_per_ms": 1.0,
                "synaptic_delay_ms": 1.0,
                "note": "망막 -> LGN 중계. 시야 위치 대응으로 연결한다.",
            },
            {
                "name": "LGN->V1_L4",
                "src": {"area": "LGN", "layer": ["L_relay"]},
                "dst": {"area": "V1", "layer": ["L4"], "cell_type": ["spiny_stellate"]},
                "target_compartment": "basal",
                "receptor": "AMPA",
                "rule": "rf_knn",
                "k": 8,
                "rf_match_sigma_deg": 1.0,
                "probability": 0.7,
                "weight": {"dist": "lognormal", "median": 0.5, "sigma": 0.3, "min": 0.0, "max": 3.0},
                "conduction_velocity_mm_per_ms": 0.5,
                "synaptic_delay_ms": 1.0,
                "plasticity_rule": "stdp",
                "note": "주 상향 입력은 L4 로 보낸다. 다른 층의 감각 입력을 금지하는 절대 규칙은 아니다.",
            },
            {
                "name": "LGN->V1_L6_weak",
                "src": {"area": "LGN", "layer": ["L_relay"]},
                "dst": {"area": "V1", "layer": ["L6"]},
                "target_compartment": "basal",
                "receptor": "AMPA",
                "rule": "rf_knn",
                "k": 3,
                "rf_match_sigma_deg": 1.2,
                "probability": 0.3,
                "weight": {"dist": "lognormal", "median": 0.2, "sigma": 0.3, "min": 0.0, "max": 2.0},
                "conduction_velocity_mm_per_ms": 0.5,
                "synaptic_delay_ms": 1.2,
                "note": "L4 가 아닌 층으로 가는 약한 상향 입력 (혼합 허용).",
            },
            {
                "name": "V1_L4->L4_PV",
                "src": {"area": "V1", "layer": ["L4"], "cell_type": ["spiny_stellate"]},
                "dst": {"area": "V1", "layer": ["L4"], "cell_type": ["pv_basket"]},
                "target_compartment": "soma",
                "receptor": "AMPA",
                "rule": "local_radius",
                "radius_mm": 0.4,
                "k": 6,
                "probability": 0.6,
                "weight": {"dist": "lognormal", "median": 0.4, "sigma": 0.2, "min": 0.0, "max": 3.0},
                "conduction_velocity_mm_per_ms": 0.2,
                "synaptic_delay_ms": 0.6,
            },
            {
                "name": "V1_L4_PV->L4",
                "src": {"area": "V1", "layer": ["L4"], "cell_type": ["pv_basket"]},
                "dst": {"area": "V1", "layer": ["L4"], "cell_type": ["spiny_stellate"]},
                "target_compartment": "soma",
                "receptor": "GABA_A",
                "rule": "local_radius",
                "radius_mm": 0.4,
                "k": 6,
                "probability": 0.7,
                "weight": {"dist": "lognormal", "median": 0.5, "sigma": 0.2, "min": 0.0, "max": 3.0},
                "conduction_velocity_mm_per_ms": 0.2,
                "synaptic_delay_ms": 0.5,
                "note": "억제 연결. 가중치는 비음수 크기이고 부호는 세포 유형이 만든다.",
            },
            {
                "name": "V1_L4->L2L3",
                "src": {"area": "V1", "layer": ["L4"], "cell_type": ["spiny_stellate"]},
                "dst": {"area": "V1", "layer": ["L2", "L3"], "cell_type": ["pyramidal"]},
                "target_compartment": "basal",
                "receptor": "AMPA",
                "rule": "local_radius",
                "radius_mm": 0.5,
                "k": 8,
                "probability": 0.6,
                "weight": {"dist": "lognormal", "median": 0.45, "sigma": 0.3, "min": 0.0, "max": 3.0},
                "conduction_velocity_mm_per_ms": 0.2,
                "synaptic_delay_ms": 0.8,
                "plasticity_rule": "stdp",
                "note": "분기 확인: 한 L4 뉴런이 여러 L2/3 뉴런으로 갈라진다.",
            },
            {
                "name": "V1_L2L3_lateral",
                "src": {"area": "V1", "layer": ["L2", "L3"], "cell_type": ["pyramidal"]},
                "dst": {"area": "V1", "layer": ["L2", "L3"], "cell_type": ["pyramidal"]},
                "target_compartment": "basal",
                "receptor": "AMPA",
                "rule": "local_radius",
                "radius_mm": 0.6,
                "k": 5,
                "probability": 0.35,
                "weight": {"dist": "lognormal", "median": 0.25, "sigma": 0.3, "min": 0.0, "max": 2.0},
                "conduction_velocity_mm_per_ms": 0.15,
                "synaptic_delay_ms": 1.2,
                "note": "수평 재귀 연결. 시간 진화에 실제 영향을 준다.",
            },
            {
                "name": "V1_L2L3->L5",
                "src": {"area": "V1", "layer": ["L2", "L3"], "cell_type": ["pyramidal"]},
                "dst": {"area": "V1", "layer": ["L5"]},
                "target_compartment": "basal",
                "receptor": "AMPA",
                "rule": "local_radius",
                "radius_mm": 0.6,
                "k": 4,
                "probability": 0.5,
                "weight": {"dist": "lognormal", "median": 0.35, "sigma": 0.3, "min": 0.0, "max": 3.0},
                "conduction_velocity_mm_per_ms": 0.2,
                "synaptic_delay_ms": 0.9,
            },
            {
                "name": "V1_L5->L6",
                "src": {"area": "V1", "layer": ["L5"]},
                "dst": {"area": "V1", "layer": ["L6"]},
                "target_compartment": "basal",
                "receptor": "AMPA",
                "rule": "local_radius",
                "radius_mm": 0.6,
                "k": 4,
                "probability": 0.5,
                "weight": {"dist": "lognormal", "median": 0.3, "sigma": 0.3, "min": 0.0, "max": 3.0},
                "conduction_velocity_mm_per_ms": 0.2,
                "synaptic_delay_ms": 0.9,
            },
            {
                "name": "V1_L6->LGN_feedback",
                "src": {"area": "V1", "layer": ["L6"]},
                "dst": {"area": "LGN", "layer": ["L_relay"]},
                "target_compartment": "soma",
                "receptor": "AMPA",
                "rule": "rf_knn",
                "k": 4,
                "rf_match_sigma_deg": 1.5,
                "probability": 0.4,
                "weight": {"dist": "lognormal", "median": 0.15, "sigma": 0.3, "min": 0.0, "max": 1.5},
                "conduction_velocity_mm_per_ms": 0.5,
                "synaptic_delay_ms": 2.0,
                "note": "L6 -> LGN 피드백 회로. 재귀 경로의 지연이 길다.",
            },
            {
                "name": "V1_L5->L1_inh",
                "src": {"area": "V1", "layer": ["L5"]},
                "dst": {"area": "V1", "layer": ["L1"]},
                "target_compartment": "soma",
                "receptor": "AMPA",
                "rule": "local_radius",
                "radius_mm": 0.8,
                "k": 3,
                "probability": 0.5,
                "weight": {"dist": "lognormal", "median": 0.25, "sigma": 0.3, "min": 0.0, "max": 2.0},
                "conduction_velocity_mm_per_ms": 0.2,
                "synaptic_delay_ms": 1.0,
                "note": "L1 은 세포체 밀도가 낮지만 비어 있지 않다.",
            },
            {
                "name": "V1_L1_inh->L2L3_apical",
                "src": {"area": "V1", "layer": ["L1"]},
                "dst": {"area": "V1", "layer": ["L2", "L3"], "cell_type": ["pyramidal"]},
                "target_compartment": "apical",
                "receptor": "GABA_A",
                "rule": "local_radius",
                "radius_mm": 0.8,
                "k": 5,
                "probability": 0.5,
                "weight": {"dist": "lognormal", "median": 0.3, "sigma": 0.3, "min": 0.0, "max": 2.0},
                "conduction_velocity_mm_per_ms": 0.15,
                "synaptic_delay_ms": 0.8,
                "note": "L1 억제 뉴런이 피라미드의 apical 구획에 접점을 만든다.",
            },
        ],
    },
    "learning": {"mode": "none"},
    "recording": {
        "mode": "full",
        "backend": "hdf5",
        "state_sample_every_steps": 1,
        "max_events": 2000000,
        "max_state_samples": 200000,
        "flush_every_steps": 50,
    },
    "checkpoint": {"enabled": True, "every_samples": 2, "keep_last": 3},
    "experiment": {
        "protocol": "single_pass",
        "n_samples": 4,
        "stimuli": [
            {"kind": "uniform", "n": 1, "params": {"size_px": 64, "value": 0.5}},
            {"kind": "dot", "n": 2, "params": {"size_px": 64, "radius_px": 3.0}},
            {
                "kind": "bar",
                "n": 2,
                "params": {"size_px": 64, "length_px": 32.0, "width_px": 3.0},
            },
            {
                "kind": "orientation_sweep",
                "n": 4,
                "params": {
                    "size_px": 64,
                    "n_orientations": 4,
                    "radius_px": 18.0,
                    "cycles_per_px": 0.08,
                },
            },
        ],
        "limits": {"max_disk_mb": 512, "max_ram_mb": 2048},
    },
    "validation": {
        "dt_convergence_factors": [1.0, 0.5, 0.25],
        "tolerances": {
            "lif_dt_convergence_mV": 0.6,
            "logpolar_roundtrip_deg": 0.05,
            "dog_uniform_response": 1e-06,
            "rao_gradient_rel": 0.0001,
        },
    },
}



def _engine_lif(duration_ms: float = 200.0) -> dict[str, Any]:
    return {
        "mode": "conductance_lif", "dt_ms": 0.5, "duration_ms": duration_ms,
        "min_delay_steps": 1, "delay_rounding": "ceil",
        "weight_application": "emit",
        "reset_between_samples": {
            "voltages": True, "conductances": True, "event_queue": True,
            "traces": True, "thresholds": False, "weights": False,
            "input_log": False,
        },
        "sequence_mode": False,
    }


def _v1_block() -> dict[str, Any]:
    return {
        "n_orientations": 12, "orientation_step_deg": 15.0,
        "phases_rad": [0.0, -1.5707963267948966],
        "pinwheel": {"enabled": True, "hypercolumn_mm": 0.8,
                     "n_pinwheels_per_mm2": 3.0},
        "ocular_dominance": {"enabled": True, "column_width_mm": 0.4,
                             "strength": 0.6},
        "gabor_init": {"enabled": True, "sigma_deg": 0.35, "aspect": 1.6,
                       "cycles_per_deg": 1.5},
        "fixed_gabor_reference": {"enabled": True, "sigma_deg": 0.35,
                                  "aspect": 1.6, "cycles_per_deg": 1.5,
                                  "energy_eps": 1e-06},
    }


def _validation_block() -> dict[str, Any]:
    return {
        "dt_convergence_factors": [1.0, 0.5, 0.25],
        "tolerances": {"lif_dt_convergence_mV": 0.6,
                       "logpolar_roundtrip_deg": 0.05,
                       "dog_uniform_response": 1e-06,
                       "rao_gradient_rel": 0.0001},
    }


def _config_v1_small() -> dict[str, Any]:
    """configs/v1_small.json 과 동일한 내용 (작은 영상 + V1 미세회로)."""
    return {
        "meta": {
            "name": "v1_small",
            "description": "작은 영상 + V1 미세회로. 방향·위상·주변 문맥 검사를 위한 설정.",
            "species_assumption": "primate_visual_cortex_model",
            "notes": [
                "12개 방향 x 2 위상은 계산을 위한 이산화다. 피질이 정확히 24개 균일 채널이라는 주장이 아니다.",
                "fixed_gabor_reference 는 학습이 아니라 고정 특징 추출 대조 경로다.",
            ],
        },
        "seeds": {"master": 20240202},
        "engine": _engine_lif(200.0),
        "cell_types": _CELL_TYPES,
        "retina": {
            "image": {"max_side_px": 96, "fov_deg": 12.0,
                      "input_colorspace": "srgb"},
            "channels": {"on_off_split": True, "keep_lowpass_lms": True,
                         "lowpass_sigma_px": 8.0},
            "drive": {"mode": "rate", "max_rate_hz": 60.0, "gain": 1.0,
                      "baseline_rate_hz": 0.5, "current_per_hz_pA": 2.5,
                      "sum_mode_scale": 1.0},
        },
        "retinotopy": {
            "mapping": "log_polar", "e0_deg": 0.5, "n_radial": 5,
            "n_angular": 10,
            "fovea_patch": {"enabled": True, "radius_deg": 0.5, "grid": 3},
            "sampling": {"lowpass_before_sampling": True, "sigma_scale": 0.5,
                         "min_sigma_px": 0.5, "interpolation_order": 1},
            "binocular": {"enabled": False, "interocular_shift_deg": 0.0},
        },
        "v1": _v1_block(),
        "anatomy": {
            "area_gap_mm": 1.0, "jitter_mm": 0.02,
            "areas": {
                "Retina": {
                    "kind": "retina", "surface_extent_mm": [1.5, 1.5],
                    "neurons_per_layer": {"L_input": 0},
                    "cell_type_fractions": {"L_input": {"retinal_ganglion": 1.0}},
                    "visual_field": {"max_ecc_deg": 5.0, "rf_sigma_deg": 0.25,
                                     "rf_sigma_growth_per_deg": 0.06},
                    "notes": "뉴런 수는 (채널 수 x 시야 샘플 수)로 유도된다.",
                },
                "LGN": {
                    "kind": "thalamus", "surface_extent_mm": [1.5, 1.5],
                    "neurons_per_layer": {"L_relay": 96},
                    "cell_type_fractions": {"L_relay": {"lgn_relay": 1.0}},
                    "visual_field": {"max_ecc_deg": 5.0, "rf_sigma_deg": 0.3,
                                     "rf_sigma_growth_per_deg": 0.07},
                },
                "V1": _cortical_area(
                    1.0, [2.4, 2.4],
                    {"L1": 12, "L2": 96, "L3": 96, "L4": 144, "L5": 64, "L6": 64},
                    5.0, 0.35, 0.08,
                    "hypercolumn 미세 좌표와 방향 지도가 배치·배선에 반영된다."),
            },
        },
        "wiring": {
            "max_total_synapses": 800000,
            "rules": [
                _rule("Retina->LGN", "Retina", ["L_input"], "LGN", ["L_relay"],
                      comp="soma", k=6, sigma=0.6, prob=0.8, median=1.4, vel=1.0,
                      delay=1.0, note="망막 -> LGN 중계 (시야 위치 대응)"),
                _rule("LGN->V1_L4", "LGN", ["L_relay"], "V1", ["L4"], k=12,
                      sigma=0.7, prob=0.6, median=1.2, vel=0.5, delay=1.2,
                      dst_types=["spiny_stellate"], plastic="stdp", gabor=True,
                      note="Gabor 모양으로 초기 배선을 정했다. 이 선택성은 학습된 것이 아니다."),
                _rule("LGN->V1_L4_PV", "LGN", ["L_relay"], "V1", ["L4"],
                      comp="soma", k=8, sigma=0.8, prob=0.4, median=1.0, vel=0.5,
                      delay=1.1, dst_types=["pv_basket"],
                      note="전방향 억제 (feedforward inhibition)"),
                _rule("LGN->V1_L6", "LGN", ["L_relay"], "V1", ["L6"], k=4,
                      sigma=1.0, prob=0.25, median=0.5, vel=0.5, delay=1.4,
                      dst_types=["pyramidal"],
                      note="L4 외 층으로 가는 약한 상향 입력 (혼합 허용)"),
                *_local_microcircuit("V1"),
                _rule("V1_L6->LGN_feedback", "V1", ["L6"], "LGN", ["L_relay"],
                      comp="soma", k=5, sigma=1.2, prob=0.35, median=0.4,
                      vel=0.5, delay=2.5, src_types=["pyramidal"],
                      note="L6 -> LGN 피드백. 재귀 경로가 시간 진화에 영향을 준다."),
            ],
        },
        "learning": {
            "mode": "none",
            "stdp": {"A_plus": 0.01, "A_minus": 0.012, "tau_plus_ms": 20.0,
                     "tau_minus_ms": 20.0, "weight_min": 0.0, "weight_max": 6.0},
            "homeostasis": {"eta_theta": 0.002, "window_ms": 500.0,
                            "theta_min_mV": -60.0, "theta_max_mV": -35.0},
        },
        "readout": {"enabled": False, "source_area": "V1",
                    "source_layer": ["L2", "L3"], "n_classes": 4},
        "recording": {
            "mode": "selected", "backend": "hdf5",
            "selection_criterion": "영역마다 균등 간격으로 뽑은 표본 뉴런만 상태를 기록한다. "
                                   "이벤트는 선택 뉴런에 도착/발신한 것만 저장된다.",
            "selected_areas": ["V1", "LGN"],
            "state_sample_every_steps": 2,
            "max_events": 8000000, "max_state_samples": 800000,
            "flush_every_steps": 100,
        },
        "checkpoint": {"enabled": True, "every_samples": 4, "keep_last": 3},
        "experiment": {
            "protocol": "single_pass", "n_samples": 1,
            "stimuli": [
                {"kind": "uniform", "n": 1,
                 "params": {"size_px": 96, "value": 0.5}},
                {"kind": "orientation_sweep", "n": 12,
                 "params": {"size_px": 96, "n_orientations": 12,
                            "radius_px": 28.0, "cycles_per_px": 0.06,
                            "contrast": 0.4}},
                {"kind": "phase_sweep", "n": 8,
                 "params": {"size_px": 96, "n_phases": 8, "radius_px": 28.0,
                            "cycles_per_px": 0.06}},
                {"kind": "contrast_sweep", "n": 5,
                 "params": {"size_px": 96, "min": 0.05, "max": 0.45,
                            "radius_px": 28.0, "cycles_per_px": 0.06}},
                {"kind": "length_sweep", "n": 6,
                 "params": {"size_px": 96, "min_px": 4.0, "max_px": 70.0,
                            "width_px": 3.0}},
                {"kind": "center_surround", "n": 1,
                 "params": {"size_px": 96, "n_orientations": 4,
                            "radius_px": 14.0, "outer_px": 40.0,
                            "cycles_per_px": 0.06}},
            ],
            "limits": {"max_disk_mb": 2048, "max_ram_mb": 4096},
        },
        "validation": _validation_block(),
    }


def _config_hierarchy_small() -> dict[str, Any]:
    """configs/hierarchy_small.json 과 동일한 내용 (LGN + V1/V2/V3/V4/IT)."""
    counts_small = {"L1": 8, "L2": 48, "L3": 48, "L4": 64, "L5": 32, "L6": 32}
    v1 = _config_v1_small()
    rules: list[dict[str, Any]] = [
        _rule("Retina->LGN", "Retina", ["L_input"], "LGN", ["L_relay"],
              comp="soma", k=6, sigma=0.6, prob=0.8, median=1.4, vel=1.0,
              delay=1.0, enabled=True),
        _rule("LGN->V1_L4", "LGN", ["L_relay"], "V1", ["L4"], k=12, sigma=0.7,
              prob=0.6, median=1.2, vel=0.5, delay=1.2, enabled=True,
              dst_types=["spiny_stellate"], plastic="stdp", gabor=True,
              note="Gabor 모양 초기 배선 (학습된 선택성이 아니다)"),
        _rule("LGN->V1_L4_PV", "LGN", ["L_relay"], "V1", ["L4"], comp="soma",
              k=8, sigma=0.8, prob=0.4, median=1.0, vel=0.5, delay=1.1,
              enabled=True, dst_types=["pv_basket"]),
        _rule("LGN->V1_L6", "LGN", ["L_relay"], "V1", ["L6"], k=4, sigma=1.0,
              prob=0.25, median=0.5, vel=0.5, delay=1.4, enabled=True,
              dst_types=["pyramidal"]),
    ]
    # V1 의 미세회로는 v1_small 과 동일한 규칙(enabled 키 없음)을 그대로 쓰고,
    # 상위 영역은 같은 모양의 규칙을 enabled=True 로 복제한다.
    rules += [r for r in v1["wiring"]["rules"]
              if r["name"].startswith("V1_") and "LGN" not in r["name"]]
    for area in ("V2", "V3", "V4", "IT"):
        rules += _local_microcircuit(area)
    # 기본 물체 인식 경로: V1 <-> V2 <-> V4 <-> IT
    rules += _feedforward("V1", "V2", sigma=0.9, delay=2.0)
    rules += _feedforward("V2", "V4", sigma=1.4, delay=2.4)
    rules += _feedforward("V4", "IT", sigma=2.2, delay=2.8)
    rules += _feedback("V1", "V2", sigma=1.2, delay=3.0)
    rules += _feedback("V2", "V4", sigma=1.8, delay=3.4)
    rules += _feedback("V4", "IT", sigma=2.6, delay=3.8)
    # 선택 경로: V2 <-> V3 <-> V4 (기본 활성화, 설정으로 끌 수 있다)
    rules += _feedforward("V2", "V3", sigma=1.2, delay=2.2)
    rules += _feedforward("V3", "V4", sigma=1.6, delay=2.4)
    rules += _feedback("V2", "V3", sigma=1.6, delay=3.2)
    rules += _feedback("V3", "V4", sigma=2.0, delay=3.4)
    rules += [
        _rule("V1->V4_bypass", "V1", ["L2", "L3"], "V4", ["L4"], k=5, sigma=1.6,
              prob=0.12, median=0.6, vel=0.5, delay=2.6, enabled=True,
              src_types=["pyramidal"], dst_types=["spiny_stellate"],
              note="영역 간 우회 경로. 집단 수준 경향의 예외를 허용한다."),
        _rule("V1_L6->LGN_feedback", "V1", ["L6"], "LGN", ["L_relay"],
              comp="soma", k=5, sigma=1.2, prob=0.35, median=0.4, vel=0.5,
              delay=2.5, enabled=True, src_types=["pyramidal"]),
    ]
    return {
        "meta": {
            "name": "hierarchy_small",
            "description": "LGN 과 지정한 모든 피질 영역(V1,V2,V3,V4,IT)을 포함한 작은 연결망.",
            "species_assumption": "primate_visual_cortex_model",
            "notes": [
                "기본 물체 인식 경로는 V1<->V2<->V4<->IT 이고, V2<->V3<->V4 경로는 설정으로 켤 수 있다.",
                "모든 정보가 V1->V2->V3->V4->IT 한 사슬만 거친다고 가정하지 않는다.",
                "영역별 뉴런 수·층 두께·연결 확률은 출처가 명시되지 않은 모형 파라미터다.",
                "이 기능 배분은 연구용 작업 가설이다. 같은 영역이 여러 시각 특성에 관여할 수 있다.",
            ],
        },
        "seeds": {"master": 20240303},
        "engine": _engine_lif(250.0),
        "cell_types": _CELL_TYPES,
        "retina": {
            "image": {"max_side_px": 96, "fov_deg": 12.0,
                      "input_colorspace": "srgb"},
            "channels": {"on_off_split": True, "keep_lowpass_lms": True,
                         "lowpass_sigma_px": 8.0},
            "drive": {"mode": "rate", "max_rate_hz": 60.0, "gain": 1.0,
                      "baseline_rate_hz": 0.5, "current_per_hz_pA": 2.5,
                      "sum_mode_scale": 1.0},
        },
        "retinotopy": {
            "mapping": "log_polar", "e0_deg": 0.5, "n_radial": 4,
            "n_angular": 8,
            "fovea_patch": {"enabled": True, "radius_deg": 0.5, "grid": 3},
            "sampling": {"lowpass_before_sampling": True, "sigma_scale": 0.5,
                         "min_sigma_px": 0.5, "interpolation_order": 1},
        },
        "v1": _v1_block(),
        "anatomy": {
            "area_gap_mm": 1.0, "jitter_mm": 0.02,
            "areas": {
                "Retina": {
                    "kind": "retina", "surface_extent_mm": [1.5, 1.5],
                    "neurons_per_layer": {"L_input": 0},
                    "cell_type_fractions": {"L_input": {"retinal_ganglion": 1.0}},
                    "visual_field": {"max_ecc_deg": 5.0, "rf_sigma_deg": 0.25,
                                     "rf_sigma_growth_per_deg": 0.06},
                },
                "LGN": {
                    "kind": "thalamus", "surface_extent_mm": [1.5, 1.5],
                    "neurons_per_layer": {"L_relay": 72},
                    "cell_type_fractions": {"L_relay": {"lgn_relay": 1.0}},
                    "visual_field": {"max_ecc_deg": 5.0, "rf_sigma_deg": 0.3,
                                     "rf_sigma_growth_per_deg": 0.07},
                },
                "V1": _cortical_area(1.0, [2.0, 2.0],
                                     {"L1": 10, "L2": 72, "L3": 72, "L4": 96,
                                      "L5": 48, "L6": 48}, 5.0, 0.35, 0.08,
                                     "국소 수용장, 방향/위상 채널"),
                "V2": _cortical_area(2.0, [1.8, 1.8], counts_small, 5.0, 0.7,
                                     0.14,
                                     "더 넓은 범위의 방향/위상 반응 통합과 재귀 상호작용"),
                "V3": _cortical_area(3.0, [1.6, 1.6], counts_small, 5.0, 1.0,
                                     0.18,
                                     "공간 통합. 운동 기능 평가에는 프레임 시퀀스가 필요하다."),
                "V4": _cortical_area(4.0, [1.6, 1.6], counts_small, 5.0, 1.4,
                                     0.22,
                                     "저주파 색 정보와 형태 정보가 합류하는 지점, 넓어진 수용장"),
                "IT": _cortical_area(5.0, [1.4, 1.4], counts_small, 5.0, 2.2,
                                     0.3,
                                     "집단 활동을 특징 벡터로 제공한다. 한 뉴런이 곧 카테고리 의미가 아니다."),
            },
        },
        "wiring": {"max_total_synapses": 2000000, "rules": rules},
        "learning": {
            "mode": "none",
            "stdp": {"A_plus": 0.008, "A_minus": 0.01, "tau_plus_ms": 20.0,
                     "tau_minus_ms": 20.0, "weight_min": 0.0, "weight_max": 6.0},
            "homeostasis": {"eta_theta": 0.002, "window_ms": 500.0,
                            "theta_min_mV": -60.0, "theta_max_mV": -35.0},
            "rao": {"n_levels": 2, "level_sizes": [32, 16], "sigma": 1.0,
                    "sigma_td": 2.0, "alpha": 0.05, "lambda_u": 0.001,
                    "settle_steps": 40, "r_step": 0.05, "u_step": 0.002},
        },
        "readout": {
            "enabled": True, "source_area": "IT", "source_layer": ["L2", "L3"],
            "classifier": "ridge", "l2": 1.0, "n_classes": 7,
        },
        "recording": {
            "mode": "selected", "backend": "hdf5",
            "selection_criterion": "영역마다 균등 간격으로 뽑은 표본 뉴런의 상태와 그 뉴런 "
                                   "관련 이벤트만 저장한다. 전체 뉴런의 매 스텝 상태는 저장하지 않는다.",
            "selected_areas": ["V1", "V2", "V4", "IT"],
            "state_sample_every_steps": 4,
            "max_events": 12000000, "max_state_samples": 1000000,
            "flush_every_steps": 200,
        },
        "checkpoint": {"enabled": True, "every_samples": 8, "keep_last": 3},
        "experiment": {
            "protocol": "train_dev_test", "n_samples": 1,
            "splits": {"train": 0.6, "dev": 0.2, "test": 0.2,
                       "stratified": True},
            "stimuli": [
                {"kind": "shape_classes", "params": {
                    "size_px": 96,
                    "classes": ["circle", "triangle", "square",
                                "bar_horizontal", "bar_vertical", "cross",
                                "blank"],
                    "n_per_class": 6, "n_variants": 2, "radius_px": 18.0,
                    "contrast": 0.4}},
                {"kind": "color_illumination",
                 "params": {"size_px": 96, "radius_px": 18.0}},
                {"kind": "transform",
                 "params": {"size_px": 96, "label": "square",
                            "radius_px": 15.0}},
            ],
            "limits": {"max_disk_mb": 4096, "max_ram_mb": 8192},
        },
        "validation": _validation_block(),
    }


def _config_megapixel_input() -> dict[str, Any]:
    """configs/megapixel_input.json 과 동일한 내용 (1024x1024 입력 점검)."""
    mp = copy.deepcopy(_config_v1_small())
    mp["meta"] = {
        "name": "megapixel_input",
        "description": "약 100만 화소(1024x1024) 입력과 제한된 피질 표본 수. "
                       "영상 화소 수와 피질 뉴런 수를 분리하는 것을 보이는 설정.",
        "species_assumption": "primate_visual_cortex_model",
        "notes": [
            "100만 화소 입력을 처리한다고 해서 100만 개 뉴런 객체를 만들지 않는다.",
            "피질 표본 수는 retinotopy 의 격자 크기로 정해지며 영상 크기와 독립이다.",
            "주변부 화소 수를 생물학 상수로 고정하지 않는다. 셀 면적과 px_per_deg 에서 계산한 추정값을 기록할 뿐이다.",
        ],
    }
    mp["seeds"] = {"master": 20240404}
    mp["engine"] = _engine_lif(120.0)
    mp["retina"]["image"] = {"max_side_px": 1024, "fov_deg": 30.0,
                             "input_colorspace": "srgb"}
    mp["retina"]["channels"] = {"on_off_split": True, "keep_lowpass_lms": True,
                                "lowpass_sigma_px": 24.0}
    mp["retina"]["dog"] = {"center_sigma_px": 2.0, "surround_sigma_px": 6.0,
                           "truncate": 4.0, "boundary_mode": "reflect",
                           "normalize_each_kernel_to_unit_sum": True}
    mp["retinotopy"] = {
        "mapping": "log_polar", "e0_deg": 0.5, "n_radial": 6, "n_angular": 12,
        "fovea_patch": {"enabled": True, "radius_deg": 0.5, "grid": 3},
        "sampling": {"lowpass_before_sampling": True, "sigma_scale": 0.5,
                     "min_sigma_px": 0.5, "interpolation_order": 1},
    }
    areas = mp["anatomy"]["areas"]
    areas["Retina"]["visual_field"] = {"max_ecc_deg": 14.0, "rf_sigma_deg": 0.25,
                                       "rf_sigma_growth_per_deg": 0.06}
    areas["LGN"]["neurons_per_layer"] = {"L_relay": 96}
    areas["LGN"]["visual_field"] = {"max_ecc_deg": 14.0, "rf_sigma_deg": 0.35,
                                    "rf_sigma_growth_per_deg": 0.08}
    areas["V1"]["neurons_per_layer"] = {"L1": 8, "L2": 48, "L3": 48, "L4": 72,
                                        "L5": 32, "L6": 32}
    areas["V1"]["visual_field"] = {"max_ecc_deg": 14.0, "rf_sigma_deg": 0.45,
                                   "rf_sigma_growth_per_deg": 0.1}
    mp["wiring"]["max_total_synapses"] = 1500000
    mp["recording"] = {
        "mode": "summary", "backend": "hdf5",
        "selection_criterion": "대규모 입력 점검 설정이므로 이벤트는 집계만 저장하고 "
                               "개별 사건은 저장하지 않는다. 뉴런별 상태도 저장하지 않는다. "
                               "재현 범위: 설정·시드·코드 해시로 같은 결과를 다시 만들 수 있으나 "
                               "개별 사건 추적은 불가능하다.",
        "selected_areas": ["V1"], "state_sample_every_steps": 10,
        "max_events": 2000000, "max_state_samples": 100000,
        "flush_every_steps": 200,
    }
    mp["experiment"] = {
        "protocol": "single_pass", "n_samples": 1,
        "stimuli": [
            {"kind": "uniform", "n": 1,
             "params": {"size_px": 1024, "value": 0.5}},
            {"kind": "bar", "n": 2,
             "params": {"size_px": 1024, "length_px": 400.0, "width_px": 12.0}},
            {"kind": "shape_classes", "params": {
                "size_px": 1024, "classes": ["circle", "square"],
                "n_per_class": 1, "n_variants": 1, "radius_px": 160.0}},
        ],
        "limits": {"max_disk_mb": 2048, "max_ram_mb": 8192},
    }
    return mp


#: 내장 설정 이름 -> 원시(해석 전) 설정을 만드는 함수
BUILTIN_CONFIGS: dict[str, Any] = {
    "minimal": _config_minimal,
    "v1_small": _config_v1_small,
    "hierarchy_small": _config_hierarchy_small,
    "megapixel_input": _config_megapixel_input,
}


def load_config_by_name_or_path(name_or_path: str) -> dict[str, Any]:
    """내장 설정 이름 또는 JSON 파일 경로에서 설정을 읽어 해석한다.

    이름(`minimal`, `v1_small`, `hierarchy_small`, `megapixel_input`)을 먼저
    보고, 아니면 파일 경로로 취급한다. 둘 다 아니면 친절한 한국어 오류를 낸다.
    """
    key = str(name_or_path).strip()
    stem = Path(key).stem
    if key in BUILTIN_CONFIGS:
        cfg = resolve_config(BUILTIN_CONFIGS[key]())
        cfg["meta"]["source_path"] = f"<builtin:{key}>"
        return cfg
    if Path(key).is_file():
        return load_config(key)
    if stem in BUILTIN_CONFIGS:
        cfg = resolve_config(BUILTIN_CONFIGS[stem]())
        cfg["meta"]["source_path"] = f"<builtin:{stem}>"
        return cfg
    raise ConfigError(
        f"설정을 찾을 수 없다: {name_or_path!r}\n"
        f"  - 내장 설정 이름: {', '.join(sorted(BUILTIN_CONFIGS))}\n"
        f"  - 또는 JSON 파일의 전체 경로를 주어라 (공백/한글 경로 가능)."
    )



# ============================================================================
# 섹션: menu  —  한국어 터미널 메뉴 (원래 파일: run.py)
# ============================================================================
"""한국어 터미널 메뉴.

**사용자가 메뉴에서 고르기 전에는 시뮬레이션이나 학습을 시작하지 않는다.**
메뉴를 고르는 것은 명시적인 실행 요청이므로 그때 해당 작업을 수행하고
진행률·중단 방법·결과 위치를 보여준다. 종료할 때 다른 실험을 자동으로
이어서 돌리지 않는다.

기본 경로는 이 파일 위치를 기준으로 해석하므로 다른 작업 디렉터리에서
실행해도 동작한다. 경로에 공백이나 한글이 있어도 된다.
"""

MENU = """
==================================================================
  3x3 뉴런 기록 구조 시각피질 시뮬레이터 (단일 파일 판)
  (망막 - LGN - V1 - V2 - V3 - V4 - IT, 연구용)
==================================================================
  1) 최소 모델 검증 실행           (내장 설정 minimal)
  2) V1 시뮬레이션 실행             (내장 설정 v1_small)
  3) 전체 시각 경로 작은 모델 실험  (내장 설정 hierarchy_small)
  4) 기존 실행 기록에서 보고서·그래프 생성
  5) 특정 뉴런의 입력·발화·출력 로그 조회
  6) 중단된 실험 재개
  7) 설정만 확인 (규모 추정, 실행하지 않음)
  8) 실행 기록 목록 보기
  9) 내장 설정 자체 점검 (configs/*.json 이 있으면 비교)
  0) 종료
==================================================================
"""

PRESETS: dict[str, tuple[str, str, str]] = {
    "1": ("minimal", "validate", "최소 모델 검증"),
    "2": ("v1_small", "simulate", "V1 시뮬레이션"),
    "3": ("hierarchy_small", "experiment", "전체 시각 경로 작은 모델 실험"),
}


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return val or default


def _check_dependencies() -> bool:
    missing: list[str] = []
    for mod, pkg in (("numpy", "numpy"), ("scipy", "scipy"),
                     ("PIL", "Pillow"), ("matplotlib", "matplotlib")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("\n[필수 패키지가 없다]")
        print(f"  없는 패키지: {', '.join(missing)}")
        print("  아래 명령을 먼저 실행하라 (사용자 환경에서 직접 실행):")
        print("    python -m pip install numpy scipy Pillow matplotlib h5py")
        return False
    try:
        __import__("h5py")
    except ImportError:
        print("\n[알림] h5py 가 설치되어 있지 않다.")
        print("  대용량 기록(events.h5, states.h5)을 쓰려면 설치가 필요하다:")
        print("    python -m pip install h5py")
        print("  설치하지 않으려면 설정에서 recording.backend 를 \"npz\" 로 바꿔라.")
    return True


def _resolve_config_choice(name_or_path: str) -> str | None:
    """내장 설정 이름 또는 파일 경로를 확인해 그대로 돌려준다."""
    key = str(name_or_path).strip()
    if key in BUILTIN_CONFIGS or Path(key).stem in BUILTIN_CONFIGS:
        return key
    if Path(key).expanduser().is_file():
        return str(Path(key).expanduser())
    print(f"\n[설정을 찾을 수 없다] {name_or_path}")
    print(f"  내장 설정 이름: {', '.join(sorted(BUILTIN_CONFIGS))}")
    cfg_dir = PROJECT_ROOT / "configs"
    if cfg_dir.is_dir():
        print(f"  또는 {cfg_dir} 안의 JSON 파일:")
        for f in sorted(cfg_dir.glob("*.json")):
            print(f"    - {f.name}")
    print("  전체 경로를 직접 입력해도 된다 (공백/한글 경로 가능).")
    return None


def _run_preset(choice: str) -> None:
    name, action, title = PRESETS[choice]
    cfg = _resolve_config_choice(_ask("설정 (내장 이름 또는 파일 경로)", name))
    if cfg is None:
        return
    out = _ask("출력 폴더 (비우면 자동 생성)", "")
    print(f"\n[{title}] 을(를) 실행한다.")
    print("  중단하려면 Ctrl+C 를 누르면 된다. 중단해도 기록과 체크포인트는 남는다.")
    argv = [action, "--config", cfg, "--execute"]
    if out:
        argv += ["--run-dir", out]
    else:
        argv += ["--runs-root", str(DEFAULT_RUNS)]
    code = cli_main(argv)
    print(f"\n[완료] 종료 코드 {code}. 결과는 {out or DEFAULT_RUNS} 아래에 있다.")


def _menu_report() -> None:
    run_dir = _ask("실행 기록 폴더 (runs/<run_id>)", "")
    if not run_dir:
        print("  실행 기록 폴더가 필요하다. 메뉴 8) 로 목록을 먼저 확인하라.")
        return
    if not Path(run_dir).is_dir():
        print(f"\n[폴더가 없다] {run_dir}")
        print("  메뉴 8) 로 실행 기록 목록을 확인하고 정확한 경로를 입력하라.")
        return
    cli_main(["report", "--run-dir", run_dir])
    nid = _ask("그림에 포함할 뉴런 ID (쉼표 구분, 비우면 생략)", "")
    argv = ["figures", "--run-dir", run_dir]
    if nid:
        argv += ["--neuron-ids", nid]
    if _ask("배치/지도 그림도 만들까? 모델을 다시 조립한다 (y/N)", "N").lower() == "y":
        argv += ["--rebuild-model"]
    cli_main(argv)
    print(f"\n[완료] 보고서와 그림은 {run_dir} 아래에 있다.")


def _menu_explain_neuron() -> None:
    run_dir = _ask("실행 기록 폴더", "")
    if not run_dir or not Path(run_dir).is_dir():
        print("  올바른 실행 기록 폴더가 필요하다. 메뉴 8) 로 목록을 확인하라.")
        return
    nid = _ask("뉴런 ID", "0")
    t0 = _ask("시작 시각 [ms]", "0")
    t1 = _ask("종료 시각 [ms]", "100")
    try:
        cli_main(["explain-neuron", "--run-dir", run_dir,
                  "--neuron-id", str(int(nid)), "--from-ms", str(float(t0)),
                  "--to-ms", str(float(t1))])
    except ValueError:
        print("  숫자를 입력해야 한다 (뉴런 ID 는 정수, 시각은 실수).")


def _menu_resume() -> None:
    run_dir = _ask("재개할 실행 기록 폴더", "")
    if not run_dir or not Path(run_dir).is_dir():
        print("  올바른 실행 기록 폴더가 필요하다. 메뉴 8) 로 목록을 확인하라.")
        return
    print("  재개한다. 중단하려면 Ctrl+C.")
    cli_main(["resume", "--run-dir", run_dir, "--execute"])


def _menu_inspect() -> None:
    cfg = _resolve_config_choice(_ask("설정 (내장 이름 또는 파일 경로)", "v1_small"))
    if cfg is None:
        return
    cli_main(["inspect-config", "--config", cfg])
    print("\n[알림] 위 값은 실행 전 추정이다. 실제 런타임은 측정 전에 확정하지 않는다.")


def menu_main() -> int:
    print(MENU)
    if not _check_dependencies():
        return 2
    while True:
        try:
            choice = _ask("번호를 선택하라", "0")
        except KeyboardInterrupt:
            print("\n종료한다.")
            return 0
        try:
            if choice == "0":
                print("종료한다. (다른 실험을 자동으로 실행하지 않는다.)")
                return 0
            if choice in PRESETS:
                _run_preset(choice)
            elif choice == "4":
                _menu_report()
            elif choice == "5":
                _menu_explain_neuron()
            elif choice == "6":
                _menu_resume()
            elif choice == "7":
                _menu_inspect()
            elif choice == "8":
                cli_main(["list-runs", "--runs-root", str(DEFAULT_RUNS)])
            elif choice == "9":
                selftest()
            else:
                print(f"  '{choice}' 은(는) 없는 번호다. 0~9 중에서 고르라.")
        except KeyboardInterrupt:
            print("\n[중단] 작업을 중단했다. 기록은 runs/ 아래에 남아 있고 "
                  "메뉴 6) 으로 재개할 수 있다.")
        except Exception as exc:  # 메뉴가 예외로 죽지 않게 한다
            print(f"\n[오류] {type(exc).__name__}: {exc}")
            print("  설정 이름/경로와 형식을 확인하라. 자세한 내용은 README_KO.md 참조.")
        print(MENU)


# ============================================================================
# 섹션: entrypoint  —  자체 점검과 진입점
# ============================================================================
def selftest() -> int:
    """내장 설정이 원래 JSON 과 같은지 확인한다. **시뮬레이션을 실행하지 않는다.**

    `configs/*.json` 이 이 파일 옆에 있으면 해석된 설정의 sha256 을 비교하고,
    없으면 내장 설정이 스키마 검증을 통과하는지만 확인한다.
    """
    cfg_dir = PROJECT_ROOT / "configs"
    n_ok = n_diff = n_skip = 0
    print("내장 설정 자체 점검 (스키마 검증 + JSON 대조). 시뮬레이션은 실행하지 않는다.")
    for name in sorted(BUILTIN_CONFIGS):
        try:
            builtin = resolve_config(BUILTIN_CONFIGS[name]())
        except ConfigError as exc:
            print(f"  [실패] {name}: 스키마 검증 실패 — {exc}")
            n_diff += 1
            continue
        h_builtin = config_hash(builtin)
        path = cfg_dir / f"{name}.json"
        if not path.is_file():
            print(f"  [건너뜀] {name}: 스키마 OK, 비교할 {path.name} 이 없다 "
                  f"(sha256 {h_builtin[:16]}…)")
            n_skip += 1
            continue
        ref = load_config(path)
        ref["meta"].pop("source_path", None)
        builtin["meta"].pop("source_path", None)
        h_ref = config_hash(ref)
        if h_builtin == h_ref:
            print(f"  [일치] {name}: sha256 {h_builtin[:16]}…")
            n_ok += 1
        else:
            print(f"  [다름] {name}: 내장 {h_builtin[:16]}… vs JSON {h_ref[:16]}…")
            for key in sorted(set(builtin) | set(ref)):
                if builtin.get(key) != ref.get(key):
                    print(f"      다른 항목: {key}")
            n_diff += 1
    print(f"\n일치 {n_ok} / 다름 {n_diff} / 건너뜀 {n_skip}")
    print("이 점검은 설정 스키마만 본다. 수치 실험 상태는 여전히 not_run 이다.")
    return 0 if n_diff == 0 else 1


_CLI_COMMANDS = {"inspect-config", "validate", "simulate", "experiment",
                 "resume", "report", "explain-neuron", "figures", "list-runs"}


def entry(argv: list[str] | None = None) -> int:
    """인자가 없으면 한국어 메뉴, 있으면 CLI 로 넘긴다.

        python cortex_all_in_one.py                       -> 메뉴
        python cortex_all_in_one.py selftest              -> 내장 설정 점검
        python cortex_all_in_one.py validate --config minimal --execute
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return menu_main()
    if args[0] == "selftest":
        return selftest()
    if args[0] in ("-h", "--help") or args[0] in _CLI_COMMANDS:
        return cli_main(args)
    print(f"[오류] 알 수 없는 명령: {args[0]!r}")
    print(f"  사용 가능한 명령: selftest, {', '.join(sorted(_CLI_COMMANDS))}")
    print("  인자 없이 실행하면 한국어 메뉴가 뜬다.")
    return 2


if __name__ == "__main__":
    raise SystemExit(entry())
