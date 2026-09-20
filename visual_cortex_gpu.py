#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""visual_cortex_gpu.py -- 고정 임계값 15, 출력 이득 P 만 학습하는 GPU 시각피질 모형.

이 파일 **하나만** 있으면 동작한다. 설정, 모델 생성, 자극 생성, GPU 엔진, 학습,
기록, 재개, 검증, 보고서, 한국어 메뉴가 모두 이 안에 있다.

실행
----
    python visual_cortex_gpu.py                      # 한국어 메뉴
    python visual_cortex_gpu.py --mode validate --device cuda --output "D:\\cortex_runs"
    python visual_cortex_gpu.py --mode diagnose --preset v1_small --device cuda --output "D:\\cortex_runs"
    python visual_cortex_gpu.py --mode train    --preset hierarchy_small --device cuda --output "D:\\cortex_runs"
    python visual_cortex_gpu.py --mode resume   --run-dir "D:\\cortex_runs\\run_..."
    python visual_cortex_gpu.py --mode report   --run-dir "D:\\cortex_runs\\run_..."

필요한 패키지
-------------
    필수:   numpy, torch, scipy
    선택:   pillow(영상 입출력), h5py(청크 기록), matplotlib(그림)

GPU 용 PyTorch 는 공식 선택기에서 자기 환경에 맞는 명령을 받아 설치한다:
https://pytorch.org/get-started/locally/
이 프로그램은 **pip 를 자동 실행하지 않는다.** CPU 전용 torch 와 CUDA torch 의
구분은 메뉴 1번 또는 ``--mode env`` 가 보여 준다.

이 모형의 계산상 가정 (생물학적 측정값이 아니다)
------------------------------------------------
* 모든 모델 뉴런의 발화 판정 임계값은 **정확히 15.0** 이다. 학습·적응·이웃 평균·
  항상성 조정을 하지 않는다. 15 는 휴지막전위 대비 15 mV 탈분극에 해당하는
  정규화 판정값이다 (``u = (V_soma - E_L_soma) / V_unit``, ``V_unit = 1 mV``).
* **학습하는 값은 뉴런별 출력 이득 P 뿐이다.** 시냅스 크기 w0, 지도, 지연, 막
  파라미터, 고정 해독기는 과제 학습 중 바뀌지 않는다. 분류기 가중치·편향도
  학습하지 않는다.
* 학습 규칙은 최종 과제 오차를 쓰는 **두 방향 섭동(SPSA 계열)** 이다. 연구용
  공학적 기준이며 뇌가 이 알고리즘을 쓴다는 주장이 아니다. 전역 오차 신호를
  사용한다는 사실을 숨기지 않는다.
* 고정 IT 해독기의 클래스 그룹은 **인공적인 해독 규칙**이다. 생물학적 의미를
  추출한 결과가 아니다.
* 이 프로그램은 인간 뇌의 완전한 복제도, 검증된 뇌 학습 법칙도 아니다.

단위 규약
---------
    시간 ms | 전압 mV | 전도도 nS | 용량 pF | 전류 pA | 거리 mm | 시야 deg
    [nS]·[mV] = [pA],  [pF]·[mV]/[ms] = [pA]  이므로 이 조합에서 환산 계수가 없다.
    해독기의 초 단위 시간창으로 넘어갈 때만 명시적으로 1000 배 환산한다.

시점 규칙
---------
    구간 시작 t_n 의 도착을 반영 -> [t_n, t_n+1] 적분 -> 그 결과의 발화는
    t_n+1 에 배정. 지연·불응·로그·GPU/CPU 참조가 모두 이 규칙을 쓴다.

import 부작용
-------------
이 파일을 import 하는 것만으로는 메뉴, GPU 초기화, 자료 생성, 파일 쓰기가
일어나지 않는다. 모든 동작은 ``main()`` 또는 명시적 호출에서만 시작한다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

__version__ = "1.0.0"

# ----------------------------------------------------------------------
# 선택/필수 라이브러리: import 실패를 조용히 넘기지 않고 사유를 보관한다.
# ----------------------------------------------------------------------
try:                                        # 주 계산 (필수)
    import torch
    _TORCH_ERROR: str | None = None
except Exception as _exc:                   # pragma: no cover - 환경 의존
    torch = None                            # type: ignore[assignment]
    _TORCH_ERROR = f"{type(_exc).__name__}: {_exc}"

try:                                        # 초기 배선 후보 (필수)
    from scipy.spatial import cKDTree
    _SCIPY_ERROR: str | None = None
except Exception as _exc:                   # pragma: no cover
    cKDTree = None                          # type: ignore[assignment]
    _SCIPY_ERROR = f"{type(_exc).__name__}: {_exc}"

try:                                        # 영상 입출력 (선택)
    from PIL import Image
    _PIL_ERROR: str | None = None
except Exception as _exc:                   # pragma: no cover
    Image = None                            # type: ignore[assignment]
    _PIL_ERROR = f"{type(_exc).__name__}: {_exc}"

try:                                        # 청크 기록 (선택, NPZ 대체 있음)
    import h5py
    _H5PY_ERROR: str | None = None
except Exception as _exc:                   # pragma: no cover
    h5py = None                             # type: ignore[assignment]
    _H5PY_ERROR = f"{type(_exc).__name__}: {_exc}"


def require_torch() -> "Any":
    """torch 가 없으면 설치 방법과 함께 분명히 실패한다."""
    if torch is None:
        raise RuntimeError(
            "PyTorch 를 불러오지 못했다: " + str(_TORCH_ERROR) + "\n"
            "  공식 선택기에서 자기 환경(OS/CUDA)에 맞는 설치 명령을 받아라:\n"
            "    https://pytorch.org/get-started/locally/\n"
            "  이 프로그램은 pip 를 자동으로 실행하지 않는다.")
    return torch


def require_scipy() -> "Any":
    if cKDTree is None:
        raise RuntimeError(
            "SciPy 를 불러오지 못했다: " + str(_SCIPY_ERROR) + "\n"
            "  초기 배선 후보 탐색에 scipy.spatial.cKDTree 가 필요하다.\n"
            "    python -m pip install scipy")
    return cKDTree


def require_pillow() -> "Any":
    if Image is None:
        raise RuntimeError(
            "Pillow 를 불러오지 못했다: " + str(_PIL_ERROR) + "\n"
            "  영상 파일 입출력에 필요하다.  python -m pip install pillow")
    return Image


# ======================================================================
# 0. 고정 상수와 단위
# ======================================================================
#: 발화 판정 임계값. **모든 뉴런에서 15.0 이며 학습하지 않는다.**
THRESHOLD: float = 15.0

#: 판정값 u 의 전압 환산 단위 [mV]. 전체 실행에서 고정이며 학습하지 않는다.
V_UNIT_MV: float = 1.0

MS_PER_S: float = 1000.0

UNITS: dict[str, str] = {
    "time": "ms", "voltage": "mV", "conductance": "nS", "capacitance": "pF",
    "current": "pA", "distance": "mm", "visual_field": "deg",
    "rate": "Hz", "threshold_u": "dimensionless (V/V_unit)",
}

COMPARTMENTS: tuple[str, ...] = ("soma", "basal", "apical")
N_COMP: int = len(COMPARTMENTS)
COMP_INDEX: dict[str, int] = {n: i for i, n in enumerate(COMPARTMENTS)}

RECEPTORS: tuple[str, ...] = ("AMPA", "NMDA", "GABA_A")
N_RECEPTOR: int = len(RECEPTORS)
RECEPTOR_INDEX: dict[str, int] = {n: i for i, n in enumerate(RECEPTORS)}

#: 수용체 고정 파라미터. 모형 파라미터이며 특정 실험의 측정값이 아니다.
RECEPTOR_PARAMS: dict[str, dict[str, float | bool]] = {
    "AMPA":   {"tau_ms": 2.0,   "E_rev_mV": 0.0,   "mg_block": False},
    "NMDA":   {"tau_ms": 100.0, "E_rev_mV": 0.0,   "mg_block": True},
    "GABA_A": {"tau_ms": 6.0,   "E_rev_mV": -70.0, "mg_block": False},
}

#: NMDA Mg 차단 계수 (Jahr-Stevens 형태의 모형 파라미터).
MG_SLOPE: float = 0.062
MG_CONC_MM: float = 1.0
MG_SCALE: float = 3.57

#: 검사 결과 상태. 실행하지 않은 것을 통과로 적지 않는다.
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_NOT_RUN = "not_run"
STATUS_MEASURED = "measured"          # 기준 없이 측정만 하는 항목
STATUS_NOT_SUPPORTED = "not_supported"

#: 기본 7개 도형 클래스. train 에서만 정의하거나 이 목록을 고정으로 쓴다.
DEFAULT_CLASSES: tuple[str, ...] = (
    "circle", "triangle", "square", "horizontal_bar", "vertical_bar",
    "cross", "blank",
)


# ======================================================================
# 1. FIX_REGISTER -- 이전 소스에서 확인된 결함과 이번 처리
# ======================================================================
@dataclass(frozen=True)
class FixEntry:
    """이전 구현의 결함 하나와 이번 처리 방법.

    ``check`` 는 **사용자가 실행하는** 검사 이름이다. 실행하기 전에는
    통과로 적지 않는다 (:class:`ValidationSuite` 참조).
    """

    fix_id: str
    previous: str
    handling: str
    check: str


FIX_REGISTER: tuple[FixEntry, ...] = (
    FixEntry("F01",
             "자동 선택한 뉴런 ID 가 기록기에만 전달되어 이벤트 selected 마스크가 전부 False",
             "선택 ID/정책을 RecordingPolicy 한 곳에서 만들어 기록기와 이벤트 경로가 "
             "같은 객체를 공유한다. 선택 여부는 계산에 영향을 주지 않는다.",
             "check_09_recording_modes"),
    FixEntry("F02",
             "이벤트 발생 수와 저장 행 수를 같은 n_events 로 보고",
             "표본별 emitted/arrived/recorded/filtered 를 분리해 세고 누적도 따로 남긴다.",
             "check_09_recording_modes"),
    FixEntry("F03",
             "rate->전류 입력의 명목 Hz 를 실제 발화 상한으로 오인",
             "입력 코드값, 주입 전류[pA], 실제 출력 발화율[Hz] 을 각각 다른 이름으로 "
             "기록한다. 정상상태 근사로 '절대 불가능' 을 단정하지 않는다.",
             "check_14_transmission"),
    FixEntry("F04",
             "망막 ON/OFF·색 정보가 LGN 배선에서 보존되지 않음",
             "LGN 중계 뉴런에 (채널, 극성) 을 부여하고 배선에서 같은 채널·극성만 잇는다. "
             "공간 수렴 수와 채널 중복을 따로 통제한다.",
             "check_12_lgn_channel_preserved"),
    FixEntry("F05",
             "극성이 없는 LGN 에 abs(Gabor) 를 적용",
             "부호 있는 Gabor 계수를 ON 은 양수부, OFF 는 음수부 절댓값으로 나누어 "
             "비음수 전도도로 구현한다. 억제는 역전위와 세포 유형이 만든다.",
             "check_12_lgn_channel_preserved"),
    FixEntry("F06",
             "층·세포 집단마다 다른 pinwheel 난수 지도 생성",
             "SharedOrientationMap 을 V1 에서 한 번만 생성하고 모든 층이 자기 표면 "
             "좌표에서 같은 지도를 평가한다.",
             "check_13_shared_orientation_map"),
    FixEntry("F07",
             "Rao->apical 함수는 있으나 주 경로에서 호출되지 않음",
             "P-only 주 모드에서 제외를 명시한다. 켜려 하면 not_supported 오류를 낸다. "
             "상위->L1/apical 고정 배선은 실제로 만들고 따로 검사한다.",
             "check_15_feedback_apical"),
    FixEntry("F08",
             "학습 none 인데 학습 실험으로 해석될 여지",
             "manifest 에 learning_mode, trainable 목록, P 변화량을 남긴다. "
             "fixed_gain 조건은 P 가 바뀌지 않았음을 수치로 보고한다.",
             "check_03_only_p_changes"),
    FixEntry("F09",
             "IT 0 특징에서 상수항 분류기가 한 클래스만 골라 우연 수준 정확도",
             "학습형 분류기를 없앴다. 영특징 비율, 다수 클래스 기준선, 혼동행렬, "
             "출력 분산을 함께 보고한다.",
             "check_20_readout"),
    FixEntry("F10",
             "전체 분할 라벨로 클래스 정의, 설정 클래스 수와 실제 클래스 수 불일치",
             "클래스는 설정 고정 또는 train 에서만 정의한다. 분할별 라벨을 검증하고 "
             "진단 자극은 분류 데이터와 분리한다.",
             "check_21_splits"),
    FixEntry("F11",
             "readout 시간창을 무시하고 표본 전체 발화를 집계",
             "반개구간 [t_start, t_end) 에 배정된 발신 출력만 집계한다.",
             "check_20_readout"),
    FixEntry("F12",
             "시점이 다른 피질 특징을 한 분류기에 섞음",
             "학습형 분류기를 제거했다. 평가마다 현재 P 로 특징을 새로 계산한다.",
             "check_20_readout"),
    FixEntry("F13",
             "simulate/experiment 체크포인트 인덱스 의미가 다름",
             "next_sample_index 하나로 통일하고 epoch/batch/replica 진행 위치를 함께 "
             "저장한다.",
             "check_22_resume"),
    FixEntry("F14",
             "재개 때 자극 재생성 RNG·분할·정규화가 바뀜",
             "자극 정의 해시, 순서(순열), train 정규화 계수, RNG 상태, 실행 단계를 "
             "모두 체크포인트에 저장하고 복원한다.",
             "check_22_resume"),
    FixEntry("F15",
             "새 EventLog 로 기존 HDF5 사건을 교체할 수 있음",
             "기록은 append-only 이고 committed row 위치를 남긴다. 재개는 그 위치부터 "
             "이어 쓴다. 기존 파일을 통째로 다시 만들지 않는다.",
             "check_27_limits_and_backpressure"),
    FixEntry("F16",
             "뉴런 조회가 여러 영상의 같은 시간대를 합침",
             "조회는 (run, sample, episode, replica, step) 으로 한정한다.",
             "check_10_query_isolation"),
    FixEntry("F17",
             "모든 표적에 배치의 마지막 event_id 를 배정",
             "표적별 마지막 사건 ID 를 따로 계산하고 큐 소비 위치와 조회 메타데이터를 "
             "구분한다.",
             "check_10_query_isolation"),
    FixEntry("F18",
             "상태 기록 한도가 초반에 소진되고 묶음 쓰기로 한도 초과",
             "전체 자극 기준 예산을 표본별로 배분하고, 쓰기 직전에 남은 용량을 정확히 "
             "검사한다.",
             "check_27_limits_and_backpressure"),
    FixEntry("F19",
             "Poisson 프레임마다 0 부터 다시 세어 엔진 누적 스텝과 어긋남",
             "외생 사건을 절대 스텝으로 배치한다. 뒤 프레임에도 사건이 도착하는지 "
             "검사한다.",
             "check_23_poisson_frames"),
    FixEntry("F20",
             "STDP 감쇠가 발화/활성 조건에 종속되어 조건마다 다르게 적용됨",
             "주 모드에서 STDP 와 가중치 감쇠를 제거했다. P 정규화 항을 쓰려면 별도 "
             "항으로 기록하고 같은 항만 있는 대조군을 둔다 (기본값 0).",
             "check_03_only_p_changes"),
    FixEntry("F21",
             "STDP 후 전체 weight clip 으로 고정 연결까지 바뀔 수 있음",
             "w0 는 불변이다. clip 은 z(P 좌표)에만 적용하고 다른 배열에는 하지 않는다.",
             "check_03_only_p_changes"),
    FixEntry("F22",
             "불응기 soma 를 나중에 덮어써 구획 연립식과 어긋남",
             "불응기 soma 클램프를 구획 연립식의 경계조건으로 넣는다. 수상돌기는 "
             "클램프된 soma 전압으로 푼다.",
             "check_08_membrane"),
    FixEntry("F23",
             "전처리 변화·작은 엔진·별도 참조 검사를 전체 회로 검증으로 오해",
             "검사마다 범위/엔진/자료/경로를 적는다. 실제 Runner·GPU 전달·저장 재개를 "
             "거치는 검사를 따로 둔다.",
             "check_22_resume"),
    FixEntry("F24",
             "양안 미측정을 passed 로 기록, 지표 계산만으로 통과 처리",
             "passed/failed/skipped/not_applicable/not_run/measured 를 분리하고 각 "
             "기준을 함께 적는다.",
             "check_11_retina_geometry"),
    FixEntry("F25",
             "측정이 흔적/RNG/모델을 바꿈",
             "진단은 독립 복사본 또는 순수 조회로만 한다. 전후 상태·P·RNG 불변을 "
             "검사한다.",
             "check_24_diagnostics_pure"),
    FixEntry("F26",
             "비교 조건의 회로·pass 수가 달랐음",
             "SPSA ± 는 같은 회로·초기 상태·시간·외생 사건을 쓰고 replica 축으로만 "
             "나뉜다.",
             "check_16_pm_pairing"),
    FixEntry("F27",
             "코드 변경 뒤 예전 수치 재사용, 보고서와 JSON 불일치",
             "실행마다 코드/설정/데이터 해시를 남기고 보고서는 저장된 로그에서만 만든다.",
             "check_28_report_matches_logs"),
    FixEntry("F28",
             "시험 세트로 하이퍼파라미터 선택, 변형 영상이 분할을 넘나듦",
             "base_id 그룹 단위로 분할한다. 선택은 train/dev 로만 하고 test 접근을 "
             "횟수까지 기록한다.",
             "check_21_splits"),
    FixEntry("F29",
             "고정 회로에 이득을 붙여 놓고 임계값·편향·해독기까지 학습할 위험",
             "trainable allowlist 는 z(P) 뿐이다. 고정 텐서는 해시로 검사한다.",
             "check_03_only_p_changes"),
    FixEntry("F30",
             "순방향 실행 완료와 학습 성공을 혼동",
             "run_status(completed) 와 learning_outcome(improved/not_improved/…) 을 "
             "따로 기록한다.",
             "check_19_spiking_perturbation"),
)


def fix_register_rows() -> list[dict[str, str]]:
    return [asdict(f) for f in FIX_REGISTER]


# ======================================================================
# 2. 환경 정보 / 난수 스트림 / 경로·해시 유틸
# ======================================================================
@dataclass
class EnvironmentInfo:
    """실행 환경을 **탐지해서** 담는다. 용량·버전을 하드코딩하지 않는다."""

    python_version: str
    platform: str
    numpy_version: str
    torch_version: str | None
    torch_import_error: str | None
    torch_cuda_build: str | None          # torch 가 빌드된 CUDA 버전 (None 이면 CPU 전용)
    cuda_available: bool
    cuda_runtime: str | None
    device_count: int
    devices: list[dict[str, Any]]
    scipy_available: bool
    pillow_available: bool
    h5py_available: bool
    matplotlib_available: bool

    @staticmethod
    def detect() -> "EnvironmentInfo":
        import platform as _platform
        tv = cuda_build = cuda_rt = None
        avail = False
        count = 0
        devs: list[dict[str, Any]] = []
        if torch is not None:
            tv = str(torch.__version__)
            cuda_build = getattr(torch.version, "cuda", None)
            try:
                avail = bool(torch.cuda.is_available())
            except Exception:
                avail = False
            if avail:
                cuda_rt = cuda_build
                try:
                    count = int(torch.cuda.device_count())
                except Exception:
                    count = 0
                for i in range(count):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        free_b, total_b = torch.cuda.mem_get_info(i)
                        devs.append({
                            "index": i, "name": props.name,
                            "capability": f"{props.major}.{props.minor}",
                            "total_vram_mb": round(total_b / 1024 ** 2, 1),
                            "free_vram_mb": round(free_b / 1024 ** 2, 1),
                            "multi_processor_count": int(props.multi_processor_count),
                        })
                    except Exception as exc:
                        devs.append({"index": i, "error": f"{type(exc).__name__}: {exc}"})
        import importlib.util as _ilu
        try:
            mpl = _ilu.find_spec("matplotlib") is not None
        except Exception:
            mpl = False
        return EnvironmentInfo(
            python_version=sys.version.split()[0],
            platform=f"{_platform.system()} {_platform.release()} ({_platform.machine()})",
            numpy_version=np.__version__,
            torch_version=tv, torch_import_error=_TORCH_ERROR,
            torch_cuda_build=cuda_build, cuda_available=avail,
            cuda_runtime=cuda_rt, device_count=count, devices=devs,
            scipy_available=cKDTree is not None,
            pillow_available=Image is not None,
            h5py_available=h5py is not None,
            matplotlib_available=mpl,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe_ko(self) -> str:
        lines = ["[환경]",
                 f"  Python            : {self.python_version}",
                 f"  플랫폼            : {self.platform}",
                 f"  NumPy             : {self.numpy_version}"]
        if self.torch_version is None:
            lines.append(f"  PyTorch           : 없음 ({self.torch_import_error})")
            lines.append("    -> https://pytorch.org/get-started/locally/ 에서 설치 명령을 받아라.")
        else:
            build = self.torch_cuda_build or "없음 (CPU 전용 빌드)"
            lines.append(f"  PyTorch           : {self.torch_version} (CUDA 빌드: {build})")
            if self.torch_cuda_build is None:
                lines.append("    -> 지금 설치된 torch 는 **CPU 전용**이다. GPU 를 쓰려면 "
                             "CUDA 빌드를 설치해야 한다.")
            lines.append(f"  CUDA 사용 가능    : {self.cuda_available}")
            for d in self.devices:
                if "error" in d:
                    lines.append(f"    GPU {d['index']}: 조회 실패 {d['error']}")
                else:
                    lines.append(f"    GPU {d['index']}: {d['name']} "
                                 f"(compute {d['capability']}, "
                                 f"VRAM {d['free_vram_mb']:.0f} / {d['total_vram_mb']:.0f} MB 여유)")
            if self.cuda_available and not self.devices:
                lines.append("    GPU 목록을 읽지 못했다.")
        lines.append(f"  SciPy / Pillow / h5py / matplotlib : "
                     f"{self.scipy_available} / {self.pillow_available} / "
                     f"{self.h5py_available} / {self.matplotlib_available}")
        if not self.h5py_available:
            lines.append("    h5py 가 없으면 기록은 NPZ 청크 대체 경로로 저장된다 "
                         "(명시적으로 기록되며 조회 기능은 유지된다).")
        lines.append("  GPU 를 가지고 있다는 사실만으로 GPU 실행 성공을 기록하지 않는다. "
                     "실제 device 는 실행 기록에 남는다.")
        return "\n".join(lines)


def resolve_device(choice: str, env: EnvironmentInfo | None = None) -> "Any":
    """``auto|cuda|cpu`` -> torch.device. 무엇을 골랐는지 숨기지 않는다."""
    t = require_torch()
    env = env or EnvironmentInfo.detect()
    choice = (choice or "auto").lower()
    if choice == "cpu":
        return t.device("cpu")
    if choice == "cuda":
        if not env.cuda_available:
            raise RuntimeError(
                "--device cuda 를 요구했지만 CUDA 를 쓸 수 없다.\n"
                f"  torch {env.torch_version}, CUDA 빌드 {env.torch_cuda_build!r}, "
                f"cuda_available={env.cuda_available}\n"
                "  CPU 전용 torch 가 설치되어 있으면 CUDA 빌드로 바꿔야 한다:\n"
                "    https://pytorch.org/get-started/locally/\n"
                "  CPU 로 실행하려면 --device cpu 를 쓰라.")
        return t.device("cuda")
    if choice != "auto":
        raise ValueError(f"--device 는 auto|cuda|cpu 중 하나다: {choice!r}")
    if env.cuda_available:
        return t.device("cuda")
    print("[알림] CUDA 를 쓸 수 없어 **CPU** 로 실행한다. "
          "(GPU 를 쓰려면 CUDA 빌드 torch 설치 후 --device cuda)")
    return t.device("cpu")


# ----------------------------------------------------------------------
#: 이름 -> 고정 오프셋. 조건을 추가해도 기존 스트림의 난수열이 바뀌지 않는다.
STREAM_OFFSETS: dict[str, int] = {
    "anatomy": 11, "wiring": 22, "weights": 33, "orientation_map": 44,
    "ocular_dominance": 45, "stimulus": 55, "split": 66, "input_noise": 77,
    "decoder_groups": 88, "spsa": 99, "diagnostics": 111, "benchmark": 122,
}


class SeedStreams:
    """이름이 고정된 독립 난수 스트림 (NumPy + torch).

    ``SeedSequence([master, offset, sub])`` 로 만들므로 이름과 sub 인덱스가 같으면
    항상 같은 수열이 나온다. 스트림 사이에 상태가 새지 않는다.
    """

    def __init__(self, master_seed: int) -> None:
        self.master_seed = int(master_seed)
        self._np: dict[tuple[str, int], np.random.Generator] = {}

    def numpy(self, name: str, sub: int = 0) -> np.random.Generator:
        if name not in STREAM_OFFSETS:
            raise KeyError(f"알 수 없는 난수 스트림 이름: {name!r} "
                           f"(가능: {sorted(STREAM_OFFSETS)})")
        key = (name, int(sub))
        if key not in self._np:
            ss = np.random.SeedSequence([self.master_seed, STREAM_OFFSETS[name], int(sub)])
            self._np[key] = np.random.default_rng(ss)
        return self._np[key]

    def torch_generator(self, name: str, sub: int, device: "Any") -> "Any":
        """torch 생성기. 매번 같은 (name, sub) 에서 같은 상태로 시작한다."""
        t = require_torch()
        if name not in STREAM_OFFSETS:
            raise KeyError(f"알 수 없는 난수 스트림 이름: {name!r}")
        ss = np.random.SeedSequence([self.master_seed, STREAM_OFFSETS[name], int(sub)])
        seed = int(ss.generate_state(1, dtype=np.uint32)[0])
        g = t.Generator(device=device)
        g.manual_seed(seed)
        return g

    def state_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"master_seed": self.master_seed, "numpy": {}}
        for (name, sub), gen in self._np.items():
            st = gen.bit_generator.state
            out["numpy"][f"{name}:{sub}"] = json.loads(json.dumps(st, default=_json_default))
        return out

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.master_seed = int(state["master_seed"])
        self._np.clear()
        for key, st in (state.get("numpy") or {}).items():
            name, _, sub = key.partition(":")
            gen = self.numpy(name, int(sub))
            gen.bit_generator.state = _restore_bitgen_state(st)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def _restore_bitgen_state(st: dict[str, Any]) -> dict[str, Any]:
    """JSON 왕복으로 list 가 된 PCG64 상태를 다시 정수로 되돌린다."""
    out = dict(st)
    inner = dict(out.get("state") or {})
    for k in ("state", "inc"):
        if k in inner:
            inner[k] = int(inner[k])
    if inner:
        out["state"] = inner
    for k in ("has_uint32", "uinteger"):
        if k in out:
            out[k] = int(out[k])
    return out


# ----------------------------------------------------------------------
def clean_user_path(raw: str) -> Path:
    """사용자가 붙여넣은 경로 문자열 -> :class:`Path`.

    Windows 탐색기의 '경로 복사' 는 ``"C:\\경로"`` 처럼 바깥따옴표를 붙인다.
    공백·한글·OneDrive·네트워크(UNC) 경로도 평범한 경로로 취급한다.
    ``eval`` / 셸 문자열 / ``unicode_escape`` 해석을 쓰지 않는다.
    """
    s = str(raw).strip()
    for quote in ('"', "'"):
        if len(s) >= 2 and s.startswith(quote) and s.endswith(quote):
            s = s[1:-1].strip()
    if not s:
        raise ValueError("빈 경로다. 저장할 폴더 경로를 입력하라.")
    return Path(s).expanduser()


def ensure_writable_dir(path: Path) -> Path:
    """폴더를 만들고 실제로 쓸 수 있는지 확인한다. 조용히 다른 곳에 쓰지 않는다."""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"폴더를 만들 수 없다: {path}\n"
            f"  원인: {type(exc).__name__}: {exc}\n"
            f"  경로 철자, 드라이브 존재 여부, 쓰기 권한을 확인하라. "
            f"다른 폴더로 대신 저장하지 않는다.") from exc
    probe = path / f".write_test_{os.getpid()}.tmp"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise OSError(f"폴더에 쓸 수 없다: {path}\n  원인: {type(exc).__name__}: {exc}") from exc
    return path


_UNSAFE_NAME_CHARS = '<>:"/\\|?*'
_RESERVED_NAMES = ({"CON", "PRN", "AUX", "NUL"}
                   | {f"COM{i}" for i in range(1, 10)}
                   | {f"LPT{i}" for i in range(1, 10)})


def safe_name(name: str) -> str:
    """어느 OS 에서나 폴더/파일 이름으로 쓸 수 있는 문자열."""
    cleaned = "".join("_" if (c in _UNSAFE_NAME_CHARS or ord(c) < 32) else c
                      for c in str(name)).strip(" .")
    if not cleaned.strip("_"):
        return "unnamed"
    if cleaned.upper() in _RESERVED_NAMES:
        return cleaned + "_"
    return cleaned


def new_run_dir(root: Path, tag: str = "run") -> Path:
    """``<root>/<tag>_<UTC시각>_<짧은ID>`` 를 만든다. 기존 폴더를 덮어쓰지 않는다."""
    root = ensure_writable_dir(Path(root))
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for attempt in range(1000):
        uid = hashlib.sha256(
            f"{stamp}{os.getpid()}{time.time_ns()}{attempt}".encode()).hexdigest()[:8]
        cand = root / safe_name(f"{tag}_{stamp}_{uid}")
        if not cand.exists():
            cand.mkdir(parents=True)
            return cand
    raise RuntimeError(f"실행 폴더 이름을 만들지 못했다: {root}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hash() -> dict[str, str]:
    """이 소스 파일 자신의 해시. 실행 결과가 어느 코드에서 나왔는지 남긴다."""
    try:
        p = Path(__file__).resolve()
        return {"file": p.name, "sha256": sha256_file(p), "size": str(p.stat().st_size)}
    except Exception as exc:                      # 대화형/압축 실행 등
        return {"file": "<unknown>", "sha256": "", "error": f"{type(exc).__name__}: {exc}"}


def _json_safe(obj: Any) -> Any:
    """JSON 으로 쓸 수 있는 구조로 바꾼다.

    ``json`` 은 dict 키로 str/int/float/bool/None 만 받는다. 검사 결과처럼 튜플을
    키로 쓴 곳이 하나라도 있으면 저장 전체가 죽는다. 값은 ``_json_default`` 가
    처리하지만 **키는 처리할 방법이 없으므로** 여기서 미리 문자열로 만든다.
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, (str, int, float, bool)) or k is None:
                key = k
            elif isinstance(k, tuple):
                key = ",".join(str(x) for x in k)
            else:
                key = str(k)
            out[key] = _json_safe(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [_json_safe(v) for v in sorted(obj, key=str)]
    return obj


def dumps(obj: Any, indent: int | None = 1) -> str:
    return json.dumps(_json_safe(obj), ensure_ascii=False, indent=indent,
                      default=_json_default)


def write_json(path: Path, obj: Any) -> None:
    """원자적으로 JSON 을 쓴다.

    직렬화할 수 없는 값이 하나 섞였다고 해서 이미 끝난 실행 결과를 통째로 버리지
    않는다. 그 경우 문제 지점을 ``repr`` 로 바꿔 저장하고, 무엇을 바꿨는지
    ``_serialization_fallback`` 에 남긴다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        text = dumps(obj)
    except (TypeError, ValueError) as exc:
        safe = json.loads(json.dumps(_json_safe(obj), ensure_ascii=False,
                                     default=repr))
        if isinstance(safe, dict):
            safe["_serialization_fallback"] = (
                f"일부 값을 그대로 저장할 수 없어 repr 로 바꿔 저장했다: "
                f"{type(exc).__name__}: {exc}")
        text = json.dumps(safe, ensure_ascii=False, indent=1)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)                       # 원자적 교체


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tensor_hash(t: "Any") -> str:
    """텐서 내용의 sha256. 고정 파라미터 불변 검사에 쓴다."""
    arr = t.detach().to("cpu").contiguous().numpy()
    return hashlib.sha256(arr.tobytes() + str(arr.dtype).encode()
                          + str(arr.shape).encode()).hexdigest()


# ======================================================================
# 3. 설정과 근거 구분
# ======================================================================
#: 근거 등급. 수치는 거의 전부 '계산상 가정' 이다. 확인하지 못한 출처를 만들지 않는다.
EVIDENCE_OBSERVED = "관찰 자료에 근거"
EVIDENCE_BORROWED = "다른 종·영역에서 차용"
EVIDENCE_ASSUMED = "계산상 가정"


@dataclass(frozen=True)
class ProvenanceEntry:
    topic: str
    grade: str
    note: str


#: **정성적 구조**의 근거와, 그 수치가 어디서 왔는지의 구분.
#: 아래 어떤 항목도 "이 논문의 수치를 그대로 썼다" 는 뜻이 아니다.
PARAM_PROVENANCE: tuple[ProvenanceEntry, ...] = (
    ProvenanceEntry(
        "신피질의 6개 층 구조와 L4 의 시상 입력 표적",
        EVIDENCE_OBSERVED,
        "포유류 신피질 일반에 대한 고전적 해부 서술. 이 모형의 층 두께[mm]와 층별 "
        "뉴런 수는 그 서술에서 나온 값이 아니라 계산 규모에 맞춘 가정이다."),
    ProvenanceEntry(
        "V1 뉴런의 방향 선택성이 존재한다는 사실",
        EVIDENCE_OBSERVED,
        "Hubel & Wiesel 계열의 고양이·붉은털원숭이 V1 기록. 이 모형의 방향 12구간, "
        "위상 2개, Gabor 파라미터는 계산용 이산화이며 측정값이 아니다."),
    ProvenanceEntry(
        "V1 표면의 방향 지도(pinwheel) 구조",
        EVIDENCE_OBSERVED,
        "고양이·원숭이 V1 광학 영상에서 보고된 정성적 구조. 이 모형은 무작위 위상 "
        "평면파 중첩으로 비슷한 모양을 만들 뿐이고, 주기·개수는 가정이다."),
    ProvenanceEntry(
        "영역 간 상향은 상부층 기원·중간층 표적, 하향은 L4 밖 표적이라는 경향",
        EVIDENCE_BORROWED,
        "붉은털원숭이 피질 계층 해부 문헌의 집단 수준 경향을 차용했다. 개별 시냅스의 "
        "완전한 목록이 아니며 혼합 기원을 허용한다. 확률·수렴 수는 가정이다."),
    ProvenanceEntry(
        "망막의 중심-주변 대비와 ON/OFF 분리, 색 대립 경로",
        EVIDENCE_BORROWED,
        "영장류 망막의 정성적 구조를 차용했다. DoG 시그마[px], 채널 이득, 구동 계수는 "
        "이 모형의 가정이다."),
    ProvenanceEntry(
        "로그-극좌표 확대(cortical magnification)",
        EVIDENCE_BORROWED,
        "영장류 V1 의 편심도 의존 확대라는 정성적 성질을 차용했다. e0[deg] 와 격자 "
        "크기는 계산 가정이다."),
    ProvenanceEntry(
        "모든 뉴런의 임계값 15.0",
        EVIDENCE_ASSUMED,
        "이번 연구의 고정 가정이다. 휴지막전위 대비 15 mV 탈분극에 해당하도록 단위를 "
        "정의했다. 생물학적 측정값이 아니다."),
    ProvenanceEntry(
        "출력 이득 P 만 학습하는 규칙과 SPSA 갱신",
        EVIDENCE_ASSUMED,
        "연구용 공학적 기준이다. 뇌가 이 알고리즘을 쓴다는 주장이 아니다."),
    ProvenanceEntry(
        "고정 IT 해독기의 클래스 그룹",
        EVIDENCE_ASSUMED,
        "인공적인 해독 규칙이다. 생물학적 의미를 추출한 결과가 아니다."),
    ProvenanceEntry(
        "막 파라미터(C, gL, EL, t_ref, 구획 결합)와 수용체 시상수·역전위",
        EVIDENCE_ASSUMED,
        "자주 쓰이는 범위 안에서 고른 모형 파라미터다. 특정 실험의 측정값을 옮긴 것이 "
        "아니다."),
)


#: 세포 유형 표. 모든 수치는 모형 파라미터다 (PARAM_PROVENANCE 참조).
CELL_TYPES: dict[str, dict[str, Any]] = {
    "retinal_ganglion": {
        "dale": "excitatory", "compartments": ["soma"],
        "C_pF": {"soma": 80.0}, "gL_nS": {"soma": 6.0}, "EL_mV": {"soma": -65.0},
        "V_reset_mV": -68.0, "t_ref_ms": 2.0, "g_couple_nS": {},
    },
    "lgn_relay": {
        "dale": "excitatory", "compartments": ["soma"],
        "C_pF": {"soma": 100.0}, "gL_nS": {"soma": 7.0}, "EL_mV": {"soma": -68.0},
        "V_reset_mV": -70.0, "t_ref_ms": 2.0, "g_couple_nS": {},
    },
    "spiny_stellate": {
        "dale": "excitatory", "compartments": ["soma", "basal"],
        "C_pF": {"soma": 150.0, "basal": 90.0},
        "gL_nS": {"soma": 8.0, "basal": 5.0},
        "EL_mV": {"soma": -70.0, "basal": -70.0},
        "V_reset_mV": -72.0, "t_ref_ms": 2.0,
        "g_couple_nS": {"basal": 10.0},
    },
    "pyramidal": {
        "dale": "excitatory", "compartments": ["soma", "basal", "apical"],
        "C_pF": {"soma": 200.0, "basal": 120.0, "apical": 120.0},
        "gL_nS": {"soma": 10.0, "basal": 6.0, "apical": 6.0},
        "EL_mV": {"soma": -70.0, "basal": -70.0, "apical": -70.0},
        "V_reset_mV": -72.0, "t_ref_ms": 2.5,
        "g_couple_nS": {"basal": 9.0, "apical": 4.0},
    },
    "pv_basket": {
        "dale": "inhibitory", "compartments": ["soma"],
        "C_pF": {"soma": 100.0}, "gL_nS": {"soma": 10.0}, "EL_mV": {"soma": -65.0},
        "V_reset_mV": -67.0, "t_ref_ms": 1.0, "g_couple_nS": {},
    },
    "sst_martinotti": {
        "dale": "inhibitory", "compartments": ["soma"],
        "C_pF": {"soma": 110.0}, "gL_nS": {"soma": 8.0}, "EL_mV": {"soma": -66.0},
        "V_reset_mV": -68.0, "t_ref_ms": 2.0, "g_couple_nS": {},
    },
    "l1_inhibitory": {
        "dale": "inhibitory", "compartments": ["soma"],
        "C_pF": {"soma": 90.0}, "gL_nS": {"soma": 8.0}, "EL_mV": {"soma": -66.0},
        "V_reset_mV": -68.0, "t_ref_ms": 2.0, "g_couple_nS": {},
    },
}

#: 망막 채널. 앞 6개는 대비(ON/OFF) 경로, 뒤 3개는 균일 밝기·색을 보존하는
#: **고정 저주파 경로**다. 둘을 혼동하지 않는다.
RETINA_CHANNELS: tuple[tuple[str, str, int], ...] = (
    # (채널 이름, 대립 축, 극성 +1=ON / -1=OFF / 0=저주파)
    ("lum_ON", "luminance", +1),
    ("lum_OFF", "luminance", -1),
    ("rg_ON", "red_green", +1),
    ("rg_OFF", "red_green", -1),
    ("by_ON", "blue_yellow", +1),
    ("by_OFF", "blue_yellow", -1),
    ("lowpass_L", "lowpass_L", 0),
    ("lowpass_M", "lowpass_M", 0),
    ("lowpass_S", "lowpass_S", 0),
)
CHANNEL_NAMES: tuple[str, ...] = tuple(c[0] for c in RETINA_CHANNELS)
N_CHANNEL: int = len(RETINA_CHANNELS)


def _cortical_area(level: int, extent_mm: Sequence[float],
                   counts: dict[str, int], rf_sigma_deg: float,
                   layer_thickness_mm: dict[str, float] | None = None) -> dict[str, Any]:
    return {
        "kind": "cortex", "level": int(level),
        "surface_extent_mm": [float(extent_mm[0]), float(extent_mm[1])],
        "layer_thickness_mm": dict(layer_thickness_mm or {
            "L1": 0.10, "L2": 0.22, "L3": 0.33, "L4": 0.30, "L5": 0.38, "L6": 0.37}),
        "neurons_per_layer": dict(counts),
        "cell_type_fractions": {
            "L1": {"l1_inhibitory": 1.0},
            "L2": {"pyramidal": 0.72, "pv_basket": 0.18, "sst_martinotti": 0.10},
            "L3": {"pyramidal": 0.72, "pv_basket": 0.18, "sst_martinotti": 0.10},
            "L4": {"spiny_stellate": 0.80, "pv_basket": 0.20},
            "L5": {"pyramidal": 0.82, "pv_basket": 0.18},
            "L6": {"pyramidal": 0.85, "pv_basket": 0.15},
        },
        "rf_sigma_deg": float(rf_sigma_deg),
    }


def _v1_counts(scale: int) -> dict[str, int]:
    return {"L1": max(2, scale // 4), "L2": 2 * scale, "L3": 2 * scale,
            "L4": 3 * scale, "L5": scale, "L6": scale}


def _local_rules(area: str, scale: float = 1.0) -> list[dict[str, Any]]:
    """한 피질 영역 내부의 층간·억제·재귀 배선.

    층을 가로지르는 규칙은 **표면 접선 거리**(``radius_space='surface'``)로 후보를
    고른다. 깊이를 포함한 3D 거리로 재면 층 간격보다 작은 반경에서 후보가 하나도
    나오지 않아 이름만 있는 경로가 된다. 지연은 언제나 3D 직선 거리를 쓴다.
    """
    def rule(name, src_l, dst_l, *, comp="basal", receptor="AMPA", radius,
             space, k, prob, w_med, src_types=None, dst_types=None,
             vel=0.2, delay=0.8, note=""):
        return {"name": f"{area}_{name}", "enabled": True,
                "src": {"area": area, "layers": list(src_l),
                        "cell_types": list(src_types or [])},
                "dst": {"area": area, "layers": list(dst_l),
                        "cell_types": list(dst_types or [])},
                "target_compartment": comp, "receptor": receptor,
                "selection": "local_radius", "radius_mm": float(radius),
                "radius_space": space, "k": int(k), "probability": float(prob),
                "w0_median_nS": float(w_med * scale), "w0_sigma": 0.3,
                "w0_max_nS": float(8.0 * scale),
                "conduction_velocity_mm_per_ms": float(vel),
                "synaptic_delay_ms": float(delay),
                "gabor_initialized": False, "note": note}

    return [
        rule("L4->L2L3", ["L4"], ["L2", "L3"], radius=0.45, space="surface", k=8,
             prob=0.6, w_med=2.6, src_types=["spiny_stellate"],
             dst_types=["pyramidal"], note="L4 -> L2/3 상행"),
        rule("L4->L4_PV", ["L4"], ["L4"], comp="soma", radius=0.35, space="cortical_3d",
             k=6, prob=0.6, w_med=2.2, src_types=["spiny_stellate"],
             dst_types=["pv_basket"]),
        rule("L4_PV->L4", ["L4"], ["L4"], comp="soma", receptor="GABA_A",
             radius=0.35, space="cortical_3d", k=6, prob=0.65, w_med=2.0,
             src_types=["pv_basket"], dst_types=["spiny_stellate"],
             note="PV 는 soma 표적 억제"),
        rule("L2L3_lateral", ["L2", "L3"], ["L2", "L3"], radius=0.6,
             space="cortical_3d", k=6, prob=0.35, w_med=1.4,
             src_types=["pyramidal"], dst_types=["pyramidal"], vel=0.15, delay=1.4,
             note="수평 재귀 연결 (지연이 실제로 계산에 들어간다)"),
        rule("L2L3->PV", ["L2", "L3"], ["L2", "L3"], comp="soma", radius=0.5,
             space="cortical_3d", k=5, prob=0.5, w_med=2.0,
             src_types=["pyramidal"], dst_types=["pv_basket"]),
        rule("PV->L2L3", ["L2", "L3"], ["L2", "L3"], comp="soma", receptor="GABA_A",
             radius=0.5, space="cortical_3d", k=6, prob=0.6, w_med=2.2,
             src_types=["pv_basket"], dst_types=["pyramidal"]),
        rule("L2L3->SST", ["L2", "L3"], ["L2", "L3"], comp="soma", radius=0.5,
             space="cortical_3d", k=4, prob=0.4, w_med=1.6,
             src_types=["pyramidal"], dst_types=["sst_martinotti"]),
        rule("SST->apical", ["L2", "L3"], ["L2", "L3"], comp="apical",
             receptor="GABA_A", radius=0.7, space="cortical_3d", k=5, prob=0.5,
             w_med=1.8, src_types=["sst_martinotti"], dst_types=["pyramidal"],
             note="SST 는 첨단수상돌기 표적 억제"),
        rule("L2L3->L5", ["L2", "L3"], ["L5"], radius=0.6, space="surface", k=5,
             prob=0.5, w_med=2.4, src_types=["pyramidal"], dst_types=["pyramidal"],
             delay=0.9),
        rule("L5->L6", ["L5"], ["L6"], radius=0.6, space="surface", k=4, prob=0.5,
             w_med=2.2, src_types=["pyramidal"], dst_types=["pyramidal"], delay=0.9),
        rule("L6->L4", ["L6"], ["L4"], radius=0.6, space="surface", k=4, prob=0.4,
             w_med=1.2, src_types=["pyramidal"], dst_types=["spiny_stellate"],
             delay=1.1, note="층내 피드백"),
        rule("L5->L1_inh", ["L5"], ["L1"], comp="soma", radius=0.9, space="surface",
             k=3, prob=0.5, w_med=1.8, src_types=["pyramidal"],
             dst_types=["l1_inhibitory"], delay=1.0),
        rule("L1_inh->apical", ["L1"], ["L2", "L3", "L5"], comp="apical",
             receptor="GABA_A", radius=0.9, space="surface", k=6, prob=0.5,
             w_med=1.6, src_types=["l1_inhibitory"], dst_types=["pyramidal"],
             vel=0.15, note="L1 맥락 경로. 정답을 아는 교사층이 아니다"),
    ]


def _interareal_rules(src: str, dst: str, *, sigma_deg: float, delay: float,
                      w_med: float, feedback: bool = False) -> list[dict[str, Any]]:
    """영역 간 배선. 상향은 L2/3 -> L4, 하향은 L4 밖(apical/L1, L5) 을 표적으로 한다."""
    if not feedback:
        return [{
            "name": f"{src}->{dst}_FF", "enabled": True,
            "src": {"area": src, "layers": ["L2", "L3"], "cell_types": ["pyramidal"]},
            "dst": {"area": dst, "layers": ["L4"], "cell_types": ["spiny_stellate"]},
            "target_compartment": "basal", "receptor": "AMPA",
            "selection": "rf_knn", "k": 8, "rf_match_sigma_deg": float(sigma_deg),
            "probability": 0.6, "w0_median_nS": float(w_med), "w0_sigma": 0.3,
            "w0_max_nS": 8.0, "conduction_velocity_mm_per_ms": 0.5,
            "synaptic_delay_ms": float(delay), "gabor_initialized": False,
            "note": "상향: 상부층 기원·중간층 표적이라는 집단 수준 경향 (혼합 허용)",
        }]
    return [{
        "name": f"{dst}->{src}_FB_apical", "enabled": True,
        "src": {"area": dst, "layers": ["L5", "L6"], "cell_types": ["pyramidal"]},
        "dst": {"area": src, "layers": ["L2", "L3"], "cell_types": ["pyramidal"]},
        "target_compartment": "apical", "receptor": "AMPA",
        "selection": "rf_knn", "k": 6, "rf_match_sigma_deg": float(sigma_deg),
        "probability": 0.45, "w0_median_nS": float(w_med), "w0_sigma": 0.3,
        "w0_max_nS": 8.0, "conduction_velocity_mm_per_ms": 0.5,
        "synaptic_delay_ms": float(delay), "gabor_initialized": False,
        "note": "하향: L4 밖(apical) 표적. 고정 배선이며 오차를 전달하지 않는다",
    }, {
        "name": f"{dst}->{src}_FB_L1", "enabled": True,
        "src": {"area": dst, "layers": ["L5", "L6"], "cell_types": ["pyramidal"]},
        "dst": {"area": src, "layers": ["L1"], "cell_types": ["l1_inhibitory"]},
        "target_compartment": "soma", "receptor": "AMPA",
        "selection": "rf_knn", "k": 4, "rf_match_sigma_deg": float(sigma_deg * 1.3),
        "probability": 0.4, "w0_median_nS": float(w_med * 0.8), "w0_sigma": 0.3,
        "w0_max_nS": 8.0, "conduction_velocity_mm_per_ms": 0.5,
        "synaptic_delay_ms": float(delay + 0.4), "gabor_initialized": False,
        "note": "하향: L1 맥락 경로",
    }]


PRESETS: tuple[str, ...] = ("tiny", "v1_small", "hierarchy_small", "megapixel_input")


def build_config(preset: str = "v1_small", **overrides: Any) -> dict[str, Any]:
    """preset 이름 -> 해석된 설정 dict.

    모든 값이 여기에 모여 있고 그대로 ``resolved_config.json`` 에 저장된다.
    실행 기록만 보고도 무엇이 쓰였는지 알 수 있다.
    """
    preset = str(preset)
    if preset not in PRESETS:
        raise ValueError(f"알 수 없는 preset: {preset!r} (가능: {list(PRESETS)})")

    if preset == "tiny":
        image_px, n_radial, n_angular = 32, 2, 8
        v1_scale, higher, sample_ms = 4, [], 60.0
        it_counts = {"L1": 2, "L2": 6, "L3": 6, "L4": 6, "L5": 4, "L6": 4}
    elif preset == "v1_small":
        image_px, n_radial, n_angular = 64, 4, 12
        v1_scale, higher, sample_ms = 10, [], 120.0
        it_counts = {"L1": 4, "L2": 16, "L3": 16, "L4": 16, "L5": 10, "L6": 10}
    elif preset == "hierarchy_small":
        image_px, n_radial, n_angular = 64, 4, 12
        v1_scale, higher, sample_ms = 10, ["V2", "V3", "V4"], 160.0
        it_counts = {"L1": 4, "L2": 20, "L3": 20, "L4": 16, "L5": 12, "L6": 12}
    else:                                   # megapixel_input
        image_px, n_radial, n_angular = 1024, 12, 32
        v1_scale, higher, sample_ms = 14, [], 120.0
        it_counts = {"L1": 4, "L2": 20, "L3": 20, "L4": 16, "L5": 12, "L6": 12}

    areas: dict[str, dict[str, Any]] = {
        "Retina": {"kind": "retina", "level": 0, "surface_extent_mm": [2.0, 2.0],
                   "cell_type": "retinal_ganglion", "rf_sigma_deg": 0.25},
        "LGN": {"kind": "thalamus", "level": 1, "surface_extent_mm": [1.5, 1.5],
                "cell_type": "lgn_relay", "rf_sigma_deg": 0.35,
                "relay_per_channel": max(2, (n_radial * n_angular) // 3)},
        "V1": _cortical_area(2, [1.6, 1.6], _v1_counts(v1_scale), 0.45),
    }
    rf = 0.9
    for i, name in enumerate(higher):
        areas[name] = _cortical_area(3 + i, [1.4, 1.4], _v1_counts(max(4, v1_scale - 2)), rf)
        rf *= 1.6
    areas["IT"] = _cortical_area(3 + len(higher), [1.2, 1.2], it_counts, rf * 1.6)

    # --- 연결 그래프: 직렬만 가정하지 않고 우회 경로를 명시한다 ---------
    cortical = ["V1"] + list(higher) + ["IT"]
    graph_ff: list[tuple[str, str]] = [(cortical[i], cortical[i + 1])
                                       for i in range(len(cortical) - 1)]
    if "V2" in higher and "V4" in higher:
        graph_ff.append(("V2", "V4"))       # V3 를 반드시 거치지 않는 우회
        graph_ff.append(("V1", "V4"))       # V1 -> V4 bypass
    rules: list[dict[str, Any]] = []

    # 망막 -> LGN: 채널과 극성을 보존한다 (F04)
    rules.append({
        "name": "Retina->LGN", "enabled": True,
        "src": {"area": "Retina", "layers": [], "cell_types": []},
        "dst": {"area": "LGN", "layers": [], "cell_types": []},
        "target_compartment": "soma", "receptor": "AMPA",
        "selection": "channel_knn", "k": 4, "rf_match_sigma_deg": 0.6,
        "probability": 1.0, "w0_median_nS": 9.0, "w0_sigma": 0.25, "w0_max_nS": 24.0,
        "conduction_velocity_mm_per_ms": 1.0, "synaptic_delay_ms": 1.0,
        "gabor_initialized": False,
        "require_same_channel": True, "min_distinct_sources": 2,
        "note": "같은 (채널, 극성) 끼리만 잇는다. 채널 중복이 공간 수렴 수를 대신하지 않는다",
    })
    # LGN -> V1 L4: 부호 있는 Gabor 를 ON/OFF 와 흥분/억제로 나눈다 (F05)
    rules.append({
        "name": "LGN->V1_L4", "enabled": True,
        "src": {"area": "LGN", "layers": [], "cell_types": []},
        "dst": {"area": "V1", "layers": ["L4"], "cell_types": ["spiny_stellate"]},
        "target_compartment": "basal", "receptor": "AMPA",
        "selection": "rf_knn", "k": 12, "rf_match_sigma_deg": 0.7,
        "probability": 0.75, "w0_median_nS": 7.0, "w0_sigma": 0.3, "w0_max_nS": 24.0,
        "conduction_velocity_mm_per_ms": 0.5, "synaptic_delay_ms": 1.2,
        "gabor_initialized": True,
        "note": "초기 배선을 Gabor 모양으로 정했다. 이 선택성은 학습된 것이 아니다",
    })
    rules.append({
        "name": "LGN->V1_L4_PV", "enabled": True,
        "src": {"area": "LGN", "layers": [], "cell_types": []},
        "dst": {"area": "V1", "layers": ["L4"], "cell_types": ["pv_basket"]},
        "target_compartment": "soma", "receptor": "AMPA",
        "selection": "rf_knn", "k": 8, "rf_match_sigma_deg": 0.8,
        "probability": 0.5, "w0_median_nS": 6.0, "w0_sigma": 0.3, "w0_max_nS": 24.0,
        "conduction_velocity_mm_per_ms": 0.5, "synaptic_delay_ms": 1.1,
        "gabor_initialized": False, "note": "전방 억제",
    })
    for name in cortical:
        rules += _local_rules(name)
    for src, dst in graph_ff:
        lvl = abs(areas[dst]["level"] - areas[src]["level"])
        rules += _interareal_rules(src, dst, sigma_deg=0.9 * lvl + 0.3,
                                   delay=2.0 + 0.4 * lvl, w_med=3.0)
        rules += _interareal_rules(src, dst, sigma_deg=1.2 * lvl + 0.4,
                                   delay=3.0 + 0.4 * lvl, w_med=1.6, feedback=True)

    cfg: dict[str, Any] = {
        "meta": {
            "preset": preset, "version": __version__,
            "description": f"{preset} preset. 고정 임계값 {THRESHOLD}, P 만 학습.",
            "claims": [
                "모든 수치는 모형 파라미터다. 생물학적 측정값이 아니다.",
                "이 결과를 인간 뇌의 완전한 복제라고 부르지 않는다.",
                "SPSA 는 전역 과제 오차를 쓰는 공학적 기준이다.",
            ],
        },
        "seed": 20260101,
        "device": "auto", "dtype": "float32", "reference_dtype": "float64",
        # 기본은 fast 모드다. True 로 두면 도착 합산이 표적 정렬 + 누적합 경로로 바뀌고
        # torch.use_deterministic_algorithms(True) 가 켜진다 (메모리·시간이 더 든다).
        "deterministic": False,
        "threshold": {"value": THRESHOLD, "v_unit_mV": V_UNIT_MV,
                      "trainable": False,
                      "note": "휴지막전위 대비 15 mV 탈분극에 해당하는 정규화 판정값"},
        "engine": {
            "dt_ms": 1.0, "sample_ms": float(sample_ms), "min_delay_steps": 1,
            "delay_rounding": "ceil", "reset_between_samples": True,
            "nan_policy": "stop",
            "timing_note": "t_n 도착 반영 -> [t_n,t_n+1] 적분 -> 발화는 t_n+1 에 배정",
        },
        "image": {"max_side_px": int(image_px), "fov_deg": 12.0,
                  "input_colorspace": "srgb", "white_point": "D65"},
        "retina": {
            "dog": {"center_sigma_px": 1.0, "surround_sigma_px": 3.0,
                    "truncate": 4.0, "boundary": "reflect",
                    "normalize_kernel_sum": True},
            "lowpass_sigma_px": 6.0,
            "normalization": {"percentile": 99.0, "min_scale_ratio": 1e-3,
                              "fit_split": "train"},
            "drive": {"mode": "current", "input_gain": 1.0,
                      "current_per_unit_pA": 260.0, "baseline_pA": 0.0,
                      "poisson_rate_hz": 60.0, "poisson_pulse_nS": 6.0,
                      "note": "입력 코드값·주입 전류·실제 발화율은 서로 다른 값이다"},
        },
        "retinotopy": {"mapping": "log_polar", "e0_deg": 0.5,
                       "n_radial": int(n_radial), "n_angular": int(n_angular),
                       "fovea_patch": 2, "interpolation": "bilinear",
                       "prefilter_by_eccentricity": True},
        "areas": areas,
        "wiring": {"rules": rules, "max_total_synapses": 4_000_000,
                   "graph_feedforward": [list(e) for e in graph_ff]},
        "v1": {
            "n_orientations": 12, "orientation_step_deg": 15.0,
            "phases_rad": [0.0, -math.pi / 2],
            "pinwheel": {"n_waves": 12, "hypercolumn_mm": 0.8},
            "gabor": {"sigma_deg": 0.35, "aspect": 1.6, "cycles_per_deg": 1.5},
            "note": "방향 12구간·위상 2개는 계산용 이산화다",
        },
        "decoder": {
            "area": "IT", "layers": ["L2", "L3"], "cell_types": ["pyramidal"],
            "classes": list(DEFAULT_CLASSES),
            "readout_start_ms": 0.4 * sample_ms, "readout_end_ms": float(sample_ms),
            "readout_scale": 1.0, "rate_unit_hz": 10.0, "bias": 0.0,
            "trainable": False,
            "note": "고정 해독기. 클래스 그룹은 인공적인 해독 규칙이다",
        },
        "training": {
            "mode": "all_neuron_gain_spsa",
            "P_min": 0.0, "P_max": 4.0, "P_init": 1.0,
            "eta": 0.05, "c": 0.02, "K": 1,
            "eta_schedule": "constant", "c_schedule": "constant",
            "batch_size": 4, "epochs": 2, "max_samples_per_epoch": 0,
            "p_regularization": 0.0,
            "optimizer": "plain_sgd",
            "trainable_allowlist": ["z"],
            "train_mask": "all",
            "note": "P 외의 학습 파라미터는 없다. Adam/clipping 을 몰래 넣지 않는다",
        },
        "data": {
            "n_base_per_class": 6, "n_variants": 2, "split_fractions": [0.6, 0.2, 0.2],
            "split_by": "base_id", "diagnostic_separate": True,
        },
        "recording": {
            "mode": "selected", "backend": "auto",
            "n_selected_per_area": 4,
            "event_budget_rows": 200_000, "state_budget_rows": 200_000,
            "state_every_steps": 4, "queue_max_items": 64,
            "note": "선택 기록은 예산을 표본에 배분한다. 한도 초과는 명시적으로 중단",
        },
        "limits": {"max_vram_mb": 0, "max_ram_mb": 0, "edge_chunk": 2_000_000},
        "benchmark": {"warmup_steps": 5, "measure_steps": 30},
    }
    for key, val in overrides.items():
        if key not in cfg:
            raise KeyError(f"알 수 없는 설정 키: {key!r}")
        if isinstance(cfg[key], dict) and isinstance(val, dict):
            cfg[key].update(val)
        else:
            cfg[key] = val
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """설정 자체의 모순을 **실행 전에** 잡는다."""
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(f"설정 오류: {msg}")

    need(cfg["threshold"]["value"] == THRESHOLD,
         f"임계값은 {THRESHOLD} 로 고정이다 (설정으로 바꿀 수 없다)")
    need(cfg["threshold"]["trainable"] is False, "임계값을 학습 대상으로 둘 수 없다")
    need(cfg["decoder"]["trainable"] is False, "해독기를 학습 대상으로 둘 수 없다")
    need(list(cfg["training"]["trainable_allowlist"]) == ["z"],
         "학습 가능한 것은 P 의 좌표 z 뿐이다")
    eng = cfg["engine"]
    need(eng["dt_ms"] > 0, "engine.dt_ms 는 양수")
    need(eng["sample_ms"] > 0, "engine.sample_ms 는 양수")
    need(int(eng["min_delay_steps"]) >= 1, "min_delay_steps 는 1 이상 (0 지연 재귀 금지)")
    tr = cfg["training"]
    need(tr["P_min"] >= 0.0, "P_min 은 0 이상 (출력 이득은 비음수)")
    need(tr["P_max"] > tr["P_min"], "P_max > P_min")
    need(tr["P_min"] < tr["P_init"] < tr["P_max"] or tr["P_init"] == tr["P_min"],
         "P_init 은 [P_min, P_max) 안이어야 한다")
    need(int(tr["K"]) >= 1, "SPSA K 는 1 이상")
    need(tr["c"] > 0 and tr["eta"] > 0, "SPSA c, eta 는 양수")
    need(tr["optimizer"] == "plain_sgd",
         "기본 갱신은 단순 식이다. 다른 최적화기는 별도 옵션·기록이 필요하다")
    dec = cfg["decoder"]
    need(0.0 <= dec["readout_start_ms"] < dec["readout_end_ms"] <= eng["sample_ms"],
         "readout 시간창은 [start, end) 이고 sample_ms 안이어야 한다")
    need(len(dec["classes"]) >= 2, "클래스가 2개 이상이어야 한다")
    need(len(set(dec["classes"])) == len(dec["classes"]), "클래스 이름이 중복됐다")
    need(dec["area"] in cfg["areas"], f"decoder.area {dec['area']!r} 가 영역에 없다")
    rec = cfg["recording"]
    need(rec["mode"] in ("full", "selected", "summary"),
         "recording.mode 는 full|selected|summary")
    need(rec["backend"] in ("auto", "hdf5", "npz"), "recording.backend 는 auto|hdf5|npz")
    for r in cfg["wiring"]["rules"]:
        tag = f"wiring 규칙 {r['name']!r}"
        need(r["src"]["area"] in cfg["areas"], f"{tag}: 알 수 없는 src.area")
        need(r["dst"]["area"] in cfg["areas"], f"{tag}: 알 수 없는 dst.area")
        need(r["receptor"] in RECEPTORS, f"{tag}: 알 수 없는 receptor")
        need(r["target_compartment"] in COMPARTMENTS, f"{tag}: 알 수 없는 구획")
        need(r["selection"] in ("rf_knn", "local_radius", "channel_knn"),
             f"{tag}: 알 수 없는 selection {r['selection']!r}")
        need(r.get("radius_space", "cortical_3d") in ("cortical_3d", "surface"),
             f"{tag}: radius_space 는 cortical_3d|surface")
        need(r["w0_median_nS"] >= 0 and r["w0_max_nS"] > 0, f"{tag}: w0 는 비음수")
        need(r["conduction_velocity_mm_per_ms"] > 0, f"{tag}: 전도 속도는 양수")
        need(r["synaptic_delay_ms"] >= 0, f"{tag}: 시냅스 지연은 0 이상")
    rt = cfg["retinotopy"]
    need(rt["e0_deg"] > 0, "retinotopy.e0_deg 는 양수")
    need(rt["n_radial"] >= 1 and rt["n_angular"] >= 1, "격자 크기는 1 이상")
    need(cfg["image"]["max_side_px"] >= 8, "image.max_side_px 는 8 이상")
    need(cfg["image"]["fov_deg"] > 0, "image.fov_deg 는 양수")


def config_hash(cfg: dict[str, Any]) -> str:
    return sha256_text(json.dumps(cfg, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":"), default=_json_default))


# ======================================================================
# 4. 망막 부호화기와 불균일 샘플러
# ======================================================================
#: sRGB (D65) -> CIE XYZ. 이 행렬은 **XYZ** 변환이며 LMS 가 아니다.
RGB_TO_XYZ = np.array([
    [0.4123907992659595, 0.3575843393838780, 0.1804807884018343],
    [0.2126390058715104, 0.7151686787677559, 0.0721923153607337],
    [0.0193308187155918, 0.1191947797946259, 0.9505321522496608],
], dtype=np.float64)

#: CIE XYZ -> LMS (Hunt-Pointer-Estevez, D65 정규화). 위 RGB->XYZ 와 **다른** 단계다.
XYZ_TO_LMS_HPE_D65 = np.array([
    [0.4002, 0.7076, -0.0808],
    [-0.2263, 1.1653, 0.0457],
    [0.0000, 0.0000, 0.9182],
], dtype=np.float64)

#: D65 백색점 (2도 표준 관찰자).
WHITE_POINT_D65_XYZ = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


def _gauss_kernel1d(sigma: float, truncate: float, dtype: Any, device: Any) -> Any:
    """유한 창에서 **정규화된** 1D 가우시안. 창을 자른 뒤 합을 1 로 맞춘다."""
    t = require_torch()
    radius = max(1, int(round(truncate * float(sigma))))
    x = t.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = t.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum()


def gaussian_blur(x: Any, sigma: float, truncate: float = 4.0,
                  boundary: str = "reflect") -> Any:
    """분리형 가우시안 블러. ``x`` 는 ``[B, C, H, W]``."""
    t = require_torch()
    if sigma <= 0:
        return x
    k = _gauss_kernel1d(sigma, truncate, x.dtype, x.device)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    mode = {"reflect": "reflect", "replicate": "replicate", "zeros": "constant"}[boundary]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    y = t.nn.functional.pad(x, (r, r, 0, 0), mode=mode)
    y = t.nn.functional.conv2d(y, kx, groups=c)
    y = t.nn.functional.pad(y, (0, 0, r, r), mode=mode)
    y = t.nn.functional.conv2d(y, ky, groups=c)
    return y


class RetinaEncoder:
    """영상 -> 9개 망막 채널 ``[B, C, H, W]``.

    경로를 분리한다.

    * 대비 경로: 휘도/적-녹/청-황 대립 신호에 **정규화된** 중심-주변(DoG) 필터를
      적용하고 ON/OFF 로 나눈다 (비음수).
    * 고정 저주파 경로: 균일한 밝기·색을 잃지 않도록 L/M/S 를 저역통과한 별도
      채널을 둔다. DoG 대비 경로와 혼동하지 않는다.

    정규화 계수는 **train 에서만** 추정하고 dev/test/재개에서 재사용한다.
    """

    def __init__(self, cfg: dict[str, Any], device: Any, dtype: Any) -> None:
        t = require_torch()
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.dog = cfg["retina"]["dog"]
        self.lowpass_sigma_px = float(cfg["retina"]["lowpass_sigma_px"])
        self.colorspace = cfg["image"]["input_colorspace"]
        self.rgb_to_xyz = t.tensor(RGB_TO_XYZ, dtype=dtype, device=device)
        self.xyz_to_lms = t.tensor(XYZ_TO_LMS_HPE_D65, dtype=dtype, device=device)
        self.white_point = t.tensor(WHITE_POINT_D65_XYZ, dtype=dtype, device=device)
        self.scale: Any | None = None              # [C] 정규화 계수
        self.norm_info: dict[str, Any] = {"fitted": False}

    # -- 색 변환 -------------------------------------------------------
    def srgb_to_linear(self, x: Any) -> Any:
        t = require_torch()
        a = 0.055
        return t.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

    def to_lms(self, rgb: Any) -> Any:
        """``[B,3,H,W]`` sRGB (또는 선형) -> LMS. 두 변환 단계를 분리해 적용한다."""
        require_torch()          # torch 없으면 여기서 멈춘다
        lin = self.srgb_to_linear(rgb) if self.colorspace == "srgb" else rgb
        b, _, h, w = lin.shape
        flat = lin.permute(0, 2, 3, 1).reshape(-1, 3)
        xyz = flat @ self.rgb_to_xyz.T          # 1단계: RGB -> XYZ
        lms = xyz @ self.xyz_to_lms.T           # 2단계: XYZ -> LMS (백색점 D65)
        return lms.reshape(b, h, w, 3).permute(0, 3, 1, 2)

    # -- 채널 생성 -----------------------------------------------------
    def encode(self, rgb: Any) -> Any:
        """``[B,3,H,W]`` -> ``[B,9,H,W]`` 비음수 채널."""
        t = require_torch()
        lms = self.to_lms(rgb)
        L, M, S = lms[:, 0:1], lms[:, 1:2], lms[:, 2:3]
        opponent = t.cat([0.6 * L + 0.4 * M,        # 휘도
                          L - M,                     # 적-녹
                          S - 0.5 * (L + M)], dim=1)  # 청-황
        sc = float(self.dog["center_sigma_px"])
        ss = float(self.dog["surround_sigma_px"])
        tr = float(self.dog["truncate"])
        bnd = str(self.dog["boundary"])
        center = gaussian_blur(opponent, sc, tr, bnd)
        surround = gaussian_blur(opponent, ss, tr, bnd)
        d = center - surround                    # 균일 입력이면 이론상 0
        on = t.clamp(d, min=0.0)
        off = t.clamp(-d, min=0.0)
        contrast = t.stack([on[:, 0], off[:, 0], on[:, 1], off[:, 1],
                            on[:, 2], off[:, 2]], dim=1)
        lp = t.clamp(gaussian_blur(lms, self.lowpass_sigma_px, tr, bnd), min=0.0)
        return t.cat([contrast, lp], dim=1)

    # -- 정규화 --------------------------------------------------------
    def fit_normalization(self, sampled_values: Any, split: str) -> dict[str, Any]:
        """**격자 샘플값** ``[n, C, S]`` 에서 채널 스케일을 추정한다.

        적용할 표현과 같은 표현에서 추정해야 한다. 영상 해상도에서 추정한 계수를
        불균일 격자 샘플에 쓰면 저역통과 때문에 구동이 수십 배 약해진다.
        """
        if split != self.cfg["retina"]["normalization"]["fit_split"]:
            raise ValueError(
                f"정규화 계수는 {self.cfg['retina']['normalization']['fit_split']!r} "
                f"에서만 추정한다 (요청: {split!r})")
        arr = sampled_values.detach().to("cpu").to(torch.float64).numpy()
        flat = arr.transpose(1, 0, 2).reshape(arr.shape[1], -1)
        pos = np.where(flat > 0, flat, np.nan)
        all_nan = ~np.isfinite(pos).any(axis=1)
        pct = float(self.cfg["retina"]["normalization"]["percentile"])
        scale = np.full(flat.shape[0], np.nan, dtype=np.float64)
        if (~all_nan).any():
            scale[~all_nan] = np.nanpercentile(pos[~all_nan], pct, axis=1)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        ratio = float(self.cfg["retina"]["normalization"]["min_scale_ratio"])
        floor = ratio * float(scale.max())
        floored = [int(i) for i, v in enumerate(scale) if v < floor]
        scale = np.maximum(scale, floor)
        t = require_torch()
        self.scale = t.tensor(scale, dtype=self.dtype, device=self.device)
        self.norm_info = {
            "fitted": True, "split": split, "percentile": pct,
            "min_scale_ratio": ratio, "floored_channels": floored,
            "scale": [float(v) for v in scale],
            "fit_representation": "grid_samples", "n_images": int(arr.shape[0]),
            "note_ko": ("신호가 없는 채널이 수치 잔차만으로 최대 세기까지 증폭되지 "
                        "않도록 채널 스케일에 바닥을 두었다."),
        }
        return dict(self.norm_info)

    def normalize(self, values: Any) -> Any:
        """``[..., C, S]`` 를 채널 스케일로 나누고 [0,1] 로 자른다."""
        if self.scale is None:
            raise RuntimeError(
                "정규화 계수가 아직 없다. train 분할로 fit_normalization 을 먼저 "
                "호출하라 (조용히 1.0 을 쓰지 않는다).")
        t = require_torch()
        shape = [1] * values.dim()
        shape[-2] = self.scale.numel()
        return t.clamp(values / self.scale.view(shape), 0.0, 1.0)

    def state_dict(self) -> dict[str, Any]:
        return {"norm_info": self.norm_info,
                "scale": None if self.scale is None else
                [float(v) for v in self.scale.detach().to("cpu").numpy()]}

    def load_state_dict(self, st: dict[str, Any]) -> None:
        self.norm_info = dict(st.get("norm_info") or {"fitted": False})
        sc = st.get("scale")
        if sc is None:
            self.scale = None
        else:
            t = require_torch()
            self.scale = t.tensor(np.asarray(sc, dtype=np.float64),
                                  dtype=self.dtype, device=self.device)


class RetinotopicSampler:
    """편심도 의존 저역통과 + 로그-극좌표 샘플링.

    좌표 공간을 섞지 않는다.

    * ``image_px``      : 영상 화소 (row, col)
    * ``visual_field``  : 시야 각도 (x_deg, y_deg), 중심이 원점
    * ``sample index``  : 격자 표본 번호 0..S-1

    중심 특이점은 ``rho = log(1 + ecc/e0)`` 로 피하고, 중심부는 별도의 Cartesian
    패치로 덮는다. 각도는 주기적이며 영상 밖 표본은 valid 마스크로 표시한다.
    """

    def __init__(self, cfg: dict[str, Any], device: Any, dtype: Any) -> None:
        require_torch()
        rt = cfg["retinotopy"]
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.e0 = float(rt["e0_deg"])
        self.n_radial = int(rt["n_radial"])
        self.n_angular = int(rt["n_angular"])
        self.fovea_patch = int(rt["fovea_patch"])
        self.fov_deg = float(cfg["image"]["fov_deg"])
        self.ecc_max = 0.5 * self.fov_deg
        self.prefilter = bool(rt["prefilter_by_eccentricity"])
        x, y, sigma, kind = self._build_grid()
        self.x_deg = np.asarray(x, dtype=np.float64)
        self.y_deg = np.asarray(y, dtype=np.float64)
        self.sigma_deg = np.asarray(sigma, dtype=np.float64)
        self.kind = list(kind)                   # 'fovea' | 'logpolar'
        self.n_samples = int(self.x_deg.size)
        self.ecc_deg = np.hypot(self.x_deg, self.y_deg)
        self.theta_rad = np.mod(np.arctan2(self.y_deg, self.x_deg), 2 * math.pi)
        self.rho = self.ecc_to_rho(self.ecc_deg)

    # -- 좌표 변환 (왕복 가능) ----------------------------------------
    def ecc_to_rho(self, ecc: np.ndarray | float) -> np.ndarray:
        return np.log(1.0 + np.asarray(ecc, dtype=np.float64) / self.e0)

    def rho_to_ecc(self, rho: np.ndarray | float) -> np.ndarray:
        return self.e0 * (np.exp(np.asarray(rho, dtype=np.float64)) - 1.0)

    def _build_grid(self) -> tuple[list[float], list[float], list[float], list[str]]:
        xs: list[float] = []
        ys: list[float] = []
        sg: list[float] = []
        kd: list[str] = []
        # 중심 Cartesian 패치: log-polar 의 중심 특이점을 덮는다.
        ecc_inner = self.rho_to_ecc(self.ecc_to_rho(self.ecc_max) / (self.n_radial + 1))
        f = self.fovea_patch
        if f > 0:
            step = (2.0 * ecc_inner) / f
            for i in range(f):
                for j in range(f):
                    xs.append(-ecc_inner + step * (j + 0.5))
                    ys.append(-ecc_inner + step * (i + 0.5))
                    sg.append(max(step * 0.5, 1e-4))
                    kd.append("fovea")
        rho_max = self.ecc_to_rho(self.ecc_max)
        rho_min = self.ecc_to_rho(ecc_inner)
        for i in range(self.n_radial):
            frac = (i + 0.5) / self.n_radial
            rho = rho_min + (rho_max - rho_min) * frac
            ecc = float(self.rho_to_ecc(rho))
            drho = (rho_max - rho_min) / self.n_radial
            d_ecc = float(self.rho_to_ecc(rho + 0.5 * drho)
                          - self.rho_to_ecc(rho - 0.5 * drho))
            d_tan = ecc * (2 * math.pi / self.n_angular)
            sigma = max(1e-4, 0.5 * math.sqrt(max(d_ecc, 1e-6) * max(d_tan, 1e-6)))
            for j in range(self.n_angular):
                th = 2 * math.pi * j / self.n_angular
                xs.append(ecc * math.cos(th))
                ys.append(ecc * math.sin(th))
                sg.append(sigma)
                kd.append("logpolar")
        return xs, ys, sg, kd

    def px_per_deg(self, h: int, w: int) -> float:
        return max(h, w) / self.fov_deg

    def pixel_coords(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """표본 -> (col, row, valid). 영상 밖은 valid=False 로 표시한다."""
        ppd = self.px_per_deg(h, w)
        col = (w - 1) / 2.0 + self.x_deg * ppd
        row = (h - 1) / 2.0 + self.y_deg * ppd
        valid = (col >= 0) & (col <= w - 1) & (row >= 0) & (row <= h - 1)
        return col, row, valid

    def sample(self, channels: Any) -> tuple[Any, dict[str, Any]]:
        """``[B,C,H,W]`` -> ``[B,C,S]``.

        표본마다 자기 수용장 크기에 맞는 저역통과를 **먼저** 적용한 뒤 뽑는다
        (aliasing 을 줄이기 위한 것이며, 측정은 검사 11 에서 한다).
        """
        t = require_torch()
        b, c, h, w = channels.shape
        ppd = self.px_per_deg(h, w)
        col, row, valid = self.pixel_coords(h, w)
        gx = t.tensor(col / max(w - 1, 1) * 2 - 1, dtype=channels.dtype,
                      device=channels.device)
        gy = t.tensor(row / max(h - 1, 1) * 2 - 1, dtype=channels.dtype,
                      device=channels.device)
        grid = t.stack([gx, gy], dim=-1).view(1, 1, self.n_samples, 2).expand(b, 1, -1, 2)
        sigma_px = np.maximum(self.sigma_deg * ppd, 0.0)
        keys = np.round(sigma_px, 2)
        out = t.zeros((b, c, self.n_samples), dtype=channels.dtype,
                      device=channels.device)
        used: list[float] = []
        for sg in np.unique(keys):
            idx = np.nonzero(keys == sg)[0]
            src = channels if (not self.prefilter or sg < 0.5) else \
                gaussian_blur(channels, float(sg), 4.0, self.cfg["retina"]["dog"]["boundary"])
            used.append(float(sg))
            sub = grid[:, :, idx, :]
            got = t.nn.functional.grid_sample(
                src, sub, mode="bilinear", padding_mode="zeros", align_corners=True)
            out[:, :, t.tensor(idx, device=channels.device, dtype=t.long)] = got[:, :, 0, :]
        vmask = t.tensor(valid, device=channels.device)
        out = out * vmask.view(1, 1, -1).to(out.dtype)
        meta = {"n_samples": self.n_samples, "px_per_deg": float(ppd),
                "prefilter_sigmas_px": sorted(used),
                "n_invalid_samples": int((~valid).sum()),
                "fovea_samples": int(sum(1 for k in self.kind if k == "fovea"))}
        return out, meta


# ======================================================================
# 5. 자극 생성과 분할
# ======================================================================
@dataclass
class Stimulus:
    """자극 하나. ``frames`` 는 ``[H,W,3]`` float32 sRGB (0..1) 목록이다."""

    stimulus_id: str
    base_id: str
    label: str
    kind: str                       # "classification" | "diagnostic"
    frames: list[np.ndarray]
    meta: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.stimulus_id.encode())
        h.update(self.base_id.encode())
        h.update(self.label.encode())
        for fr in self.frames:
            h.update(np.ascontiguousarray(fr, dtype=np.float32).tobytes())
        return h.hexdigest()


def _blank(size: int, level: float = 0.5) -> np.ndarray:
    return np.full((size, size, 3), float(level), dtype=np.float32)


def _coords(size: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    return x - c, y - c


def _paint(img: np.ndarray, mask: np.ndarray, color: Sequence[float]) -> np.ndarray:
    out = img.copy()
    col = np.asarray(color, dtype=np.float32).reshape(1, 3)
    out[mask] = col
    return out


def draw_shape(size: int, label: str, *, cx: float = 0.0, cy: float = 0.0,
               scale: float = 0.3, angle_rad: float = 0.0,
               color: Sequence[float] = (0.92, 0.92, 0.92),
               bg: float = 0.25) -> np.ndarray:
    """도형 하나를 그린다. 좌표는 영상 중심 기준 화소 단위다."""
    img = _blank(size, bg)
    x, y = _coords(size)
    x = x - cx
    y = y - cy
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    xr = ca * x + sa * y
    yr = -sa * x + ca * y
    r = scale * size
    if label == "blank":
        return img
    if label == "circle":
        mask = (xr ** 2 + yr ** 2) <= r ** 2
    elif label == "square":
        mask = (np.abs(xr) <= r * 0.85) & (np.abs(yr) <= r * 0.85)
    elif label == "triangle":
        h = r * 1.5
        mask = (yr >= -h / 2) & (yr <= h / 2) & \
               (np.abs(xr) <= (h / 2 - yr) * (r / max(h, 1e-6)))
    elif label == "horizontal_bar":
        mask = (np.abs(yr) <= r * 0.22) & (np.abs(xr) <= r * 1.1)
    elif label == "vertical_bar":
        mask = (np.abs(xr) <= r * 0.22) & (np.abs(yr) <= r * 1.1)
    elif label == "cross":
        mask = ((np.abs(yr) <= r * 0.22) & (np.abs(xr) <= r * 1.1)) | \
               ((np.abs(xr) <= r * 0.22) & (np.abs(yr) <= r * 1.1))
    else:
        raise ValueError(f"알 수 없는 도형 라벨: {label!r}")
    return _paint(img, mask, color)


def grating_image(size: int, *, orientation_rad: float, phase_rad: float,
                  cycles_per_image: float = 4.0, contrast: float = 0.8,
                  bg: float = 0.5) -> np.ndarray:
    """진단용 사인 격자. 분류 데이터와 **분리해서** 쓴다."""
    x, y = _coords(size)
    k = 2 * math.pi * cycles_per_image / size
    proj = x * math.cos(orientation_rad) + y * math.sin(orientation_rad)
    v = bg + 0.5 * contrast * np.sin(k * proj + phase_rad)
    v = np.clip(v, 0.0, 1.0).astype(np.float32)
    return np.repeat(v[:, :, None], 3, axis=2)


def color_patch(size: int, rgb: Sequence[float], bg: float = 0.5) -> np.ndarray:
    img = _blank(size, bg)
    x, y = _coords(size)
    mask = (x ** 2 + y ** 2) <= (0.3 * size) ** 2
    return _paint(img, mask, rgb)


class StimulusGenerator:
    """분류용 도형 자극과 진단용 자극을 **따로** 만든다.

    ``base_id`` 는 변형(위치/크기/회전/색) 이전의 원본 식별자다. 분할은 base_id
    단위로 하므로 같은 원본의 변형이 train 과 test 에 나뉘어 들어가지 않는다.
    """

    def __init__(self, cfg: dict[str, Any], seeds: SeedStreams) -> None:
        self.cfg = cfg
        self.seeds = seeds
        self.size = int(cfg["image"]["max_side_px"])
        self.classes = list(cfg["decoder"]["classes"])

    def classification_set(self) -> list[Stimulus]:
        rng = self.seeds.numpy("stimulus", 0)
        n_base = int(self.cfg["data"]["n_base_per_class"])
        n_var = int(self.cfg["data"]["n_variants"])
        out: list[Stimulus] = []
        for label in self.classes:
            for b in range(n_base):
                base_id = f"{label}_b{b:03d}"
                base_cx = float(rng.uniform(-0.10, 0.10) * self.size)
                base_cy = float(rng.uniform(-0.10, 0.10) * self.size)
                base_scale = float(rng.uniform(0.20, 0.30))
                base_angle = float(rng.uniform(0.0, math.pi))
                for v in range(n_var):
                    jx = base_cx + float(rng.uniform(-0.04, 0.04) * self.size)
                    jy = base_cy + float(rng.uniform(-0.04, 0.04) * self.size)
                    sc = base_scale * float(rng.uniform(0.92, 1.08))
                    ang = base_angle + float(rng.uniform(-0.15, 0.15))
                    grey = float(rng.uniform(0.82, 0.98))
                    frame = draw_shape(self.size, label, cx=jx, cy=jy, scale=sc,
                                       angle_rad=ang, color=(grey, grey, grey))
                    out.append(Stimulus(
                        stimulus_id=f"{base_id}_v{v:02d}", base_id=base_id,
                        label=label, kind="classification", frames=[frame],
                        meta={"cx": jx, "cy": jy, "scale": sc, "angle_rad": ang}))
        return out

    def diagnostic_set(self) -> list[Stimulus]:
        """방향·위상·색·균일·다중 프레임 진단 자극. 분류 데이터와 섞지 않는다."""
        out: list[Stimulus] = []
        n_ori = int(self.cfg["v1"]["n_orientations"])
        step = float(self.cfg["v1"]["orientation_step_deg"])
        for i in range(n_ori):
            ori = math.radians(i * step)
            out.append(Stimulus(f"diag_ori_{i:02d}", "diag_ori", "grating",
                                "diagnostic",
                                [grating_image(self.size, orientation_rad=ori,
                                               phase_rad=0.0)],
                                {"orientation_rad": ori, "phase_rad": 0.0}))
        for j in range(8):
            ph = 2 * math.pi * j / 8
            out.append(Stimulus(f"diag_phase_{j:02d}", "diag_phase", "grating",
                                "diagnostic",
                                [grating_image(self.size, orientation_rad=0.0,
                                               phase_rad=ph)],
                                {"orientation_rad": 0.0, "phase_rad": ph}))
        out.append(Stimulus("diag_uniform", "diag_uniform", "uniform", "diagnostic",
                            [_blank(self.size, 0.5)], {"role": "균일 영상 (DoG 검사)"}))
        out.append(Stimulus("diag_dark", "diag_dark", "uniform", "diagnostic",
                            [_blank(self.size, 0.0)], {"role": "무입력 대조"}))
        for name, rgb in (("red", (0.85, 0.15, 0.15)), ("green", (0.15, 0.85, 0.15)),
                          ("blue", (0.15, 0.15, 0.85))):
            out.append(Stimulus(f"diag_color_{name}", "diag_color", "color",
                                "diagnostic", [color_patch(self.size, rgb)],
                                {"rgb": list(rgb)}))
        # 다중 프레임: 뒤 프레임에도 외생 사건이 도착하는지 검사한다 (F19)
        out.append(Stimulus(
            "diag_frames", "diag_frames", "sequence", "diagnostic",
            [grating_image(self.size, orientation_rad=0.0, phase_rad=0.0),
             grating_image(self.size, orientation_rad=math.pi / 2, phase_rad=0.0),
             grating_image(self.size, orientation_rad=math.pi / 4, phase_rad=0.0)],
            {"role": "여러 프레임 입력 도착 검사", "n_frames": 3}))
        return out

    def split(self, stims: Sequence[Stimulus]) -> dict[str, list[Stimulus]]:
        """base_id 그룹 단위 분할. 같은 원본의 변형이 분할을 넘나들지 않는다."""
        rng = self.seeds.numpy("split", 0)
        fr = [float(x) for x in self.cfg["data"]["split_fractions"]]
        if abs(sum(fr) - 1.0) > 1e-9:
            raise ValueError(f"split_fractions 합이 1 이 아니다: {fr}")
        by_label: dict[str, list[str]] = {}
        for s in stims:
            by_label.setdefault(s.label, [])
            if s.base_id not in by_label[s.label]:
                by_label[s.label].append(s.base_id)
        assign: dict[str, str] = {}
        for label in sorted(by_label):
            bases = sorted(by_label[label])
            order = rng.permutation(len(bases))
            n = len(bases)
            n_tr = max(1, int(round(fr[0] * n)))
            n_dev = max(1, int(round(fr[1] * n))) if n - n_tr >= 2 else 0
            for rank, idx in enumerate(order):
                if rank < n_tr:
                    assign[bases[idx]] = "train"
                elif rank < n_tr + n_dev:
                    assign[bases[idx]] = "dev"
                else:
                    assign[bases[idx]] = "test"
        splits: dict[str, list[Stimulus]] = {"train": [], "dev": [], "test": []}
        for s in stims:
            splits[assign[s.base_id]].append(s)
        return splits

    @staticmethod
    def split_report(splits: dict[str, list[Stimulus]]) -> dict[str, Any]:
        base_sets = {k: {s.base_id for s in v} for k, v in splits.items()}
        overlaps: dict[str, list[str]] = {}
        names = list(base_sets)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                inter = sorted(base_sets[names[i]] & base_sets[names[j]])
                if inter:
                    overlaps[f"{names[i]}&{names[j]}"] = inter
        return {
            "counts": {k: len(v) for k, v in splits.items()},
            "n_base": {k: len(v) for k, v in base_sets.items()},
            "labels": {k: sorted({s.label for s in v}) for k, v in splits.items()},
            "base_overlaps": overlaps, "no_overlap": not overlaps,
            "digest": {k: sha256_text("|".join(sorted(s.stimulus_id for s in v)))
                       for k, v in splits.items()},
        }


def load_image_file(path: Path, max_side_px: int) -> np.ndarray:
    """사용자 영상 파일 -> ``[H,W,3]`` float32 (0..1). 최대 변 길이로만 줄인다."""
    pil = require_pillow()
    img = pil.open(str(path)).convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m > max_side_px:
        scale = max_side_px / m
        img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                         pil.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr)


# ======================================================================
# 6. 뉴런/시냅스 텐서 배열과 3x3 조회 뷰
# ======================================================================
class NeuronArrays:
    """모든 뉴런의 **고정 파라미터**를 담는 구조화된 텐서 배열.

    Python 객체를 뉴런마다 만들지 않는다. 3x3 기록 인터페이스는
    :class:`NeuronView3x3` 가 이 배열들을 조회해서 만든다 (사본이 아니다).
    """

    def __init__(self, n: int, device: Any, dtype: Any) -> None:
        t = require_torch()
        self.n = int(n)
        self.device = device
        self.dtype = dtype
        z = lambda *s, dt=dtype: t.zeros(s, dtype=dt, device=device)  # noqa: E731
        self.position_mm = z(n, 3)
        self.surface_uv_mm = z(n, 2)
        self.visual_field_deg = t.full((n, 2), float("nan"), dtype=dtype, device=device)
        self.rf_sigma_deg = t.full((n,), float("nan"), dtype=dtype, device=device)
        self.pref_orientation_rad = t.full((n,), float("nan"), dtype=dtype, device=device)
        self.pref_phase_rad = t.full((n,), float("nan"), dtype=dtype, device=device)
        self.area_id = t.full((n,), -1, dtype=t.long, device=device)
        self.layer_id = t.full((n,), -1, dtype=t.long, device=device)
        self.cell_type_id = t.full((n,), -1, dtype=t.long, device=device)
        self.channel_id = t.full((n,), -1, dtype=t.long, device=device)
        self.polarity = t.zeros(n, dtype=t.long, device=device)
        self.sample_id = t.full((n,), -1, dtype=t.long, device=device)
        self.dale = t.zeros(n, dtype=t.long, device=device)          # +1 / -1
        self.has_comp = t.zeros((n, N_COMP), dtype=t.bool, device=device)
        self.has_comp[:, 0] = True
        self.C_pF = t.ones((n, N_COMP), dtype=dtype, device=device)
        self.gL_nS = t.ones((n, N_COMP), dtype=dtype, device=device)
        self.EL_mV = t.full((n, N_COMP), -70.0, dtype=dtype, device=device)
        self.g_couple_nS = z(n, N_COMP)        # [:,1]=soma-basal, [:,2]=soma-apical
        self.V_reset_mV = t.full((n,), -72.0, dtype=dtype, device=device)
        self.t_ref_ms = t.full((n,), 2.0, dtype=dtype, device=device)
        #: **읽기 전용** 임계값. 모든 뉴런에서 정확히 THRESHOLD 이며 학습하지 않는다.
        self.threshold_u = t.full((n,), THRESHOLD, dtype=dtype, device=device)
        self.area_names: list[str] = []
        self.layer_names: list[str] = []
        self.cell_type_names: list[str] = []

    # -- 고정 파라미터 해시 (학습 전후 불변 검사에 쓴다) ---------------
    def fixed_hashes(self) -> dict[str, str]:
        names = ("position_mm", "surface_uv_mm", "visual_field_deg", "rf_sigma_deg",
                 "pref_orientation_rad", "pref_phase_rad", "area_id", "layer_id",
                 "cell_type_id", "channel_id", "polarity", "dale", "has_comp",
                 "C_pF", "gL_nS", "EL_mV", "g_couple_nS", "V_reset_mV", "t_ref_ms",
                 "threshold_u")
        return {k: tensor_hash(getattr(self, k)) for k in names}

    def area_index(self, name: str) -> int:
        return self.area_names.index(name)

    def indices_of(self, area: str | None = None, layers: Sequence[str] | None = None,
                   cell_types: Sequence[str] | None = None) -> Any:
        t = require_torch()
        mask = t.ones(self.n, dtype=t.bool, device=self.device)
        if area is not None:
            mask &= (self.area_id == self.area_names.index(area))
        if layers:
            lm = t.zeros_like(mask)
            for l in layers:
                if l in self.layer_names:
                    lm |= (self.layer_id == self.layer_names.index(l))
            mask &= lm
        if cell_types:
            cm = t.zeros_like(mask)
            for c in cell_types:
                if c in self.cell_type_names:
                    cm |= (self.cell_type_id == self.cell_type_names.index(c))
            mask &= cm
        return t.nonzero(mask, as_tuple=False).flatten()


class SynapseArrays:
    """희소 edge list (+ 지연 그룹 인덱스).

    전체 N x N 거리/가중치 행렬을 만들지 않는다. 지연별로 묶어 둔 정적 인덱스를
    미리 GPU 에 올려두고, 매 스텝 그 인덱스로만 gather/scatter 한다.
    """

    def __init__(self, src: Any, dst: Any, comp: Any, receptor: Any, w0_nS: Any,
                 delay_steps: Any, rule_index: Any, n_neurons: int) -> None:
        t = require_torch()
        self.src = src.to(t.long)
        self.dst = dst.to(t.long)
        self.comp = comp.to(t.long)
        self.receptor = receptor.to(t.long)
        self.w0_nS = w0_nS
        self.delay_steps = delay_steps.to(t.long)
        self.rule_index = rule_index.to(t.long)
        self.n_neurons = int(n_neurons)
        self.n_edges = int(self.src.numel())
        #: 표적 (뉴런, 구획, 수용체) 를 1차원으로 편 인덱스. scatter_add 에 쓴다.
        self.flat_target = (self.dst * (N_COMP * N_RECEPTOR)
                            + self.comp * N_RECEPTOR + self.receptor)
        self.delay_groups: list[tuple[int, Any]] = []
        #: 결정론 경로용. 지연 그룹마다 (정렬된 edge 인덱스, 고유 표적, 세그먼트 끝).
        #: 표적 기준으로 정렬해 두면 누적 순서가 실행마다 같아 원자적 연산이 필요 없다.
        self.sorted_groups: list[tuple[int, Any, Any, Any]] = []
        if self.n_edges:
            for d in sorted(set(int(x) for x in self.delay_steps.tolist())):
                idx = t.nonzero(self.delay_steps == d, as_tuple=False).flatten()
                self.delay_groups.append((int(d), idx))
                tgt = self.flat_target[idx]
                order = t.argsort(tgt, stable=True)
                idx_sorted = idx[order]
                tgt_sorted = tgt[order]
                uniq, counts = t.unique_consecutive(tgt_sorted, return_counts=True)
                seg_end = t.cumsum(counts, dim=0)
                self.sorted_groups.append((int(d), idx_sorted, uniq, seg_end))
        self.max_delay = max((d for d, _ in self.delay_groups), default=1)

    def fixed_hashes(self) -> dict[str, str]:
        return {k: tensor_hash(getattr(self, k))
                for k in ("src", "dst", "comp", "receptor", "w0_nS",
                          "delay_steps", "rule_index")}

    def in_degree(self) -> Any:
        t = require_torch()
        out = t.zeros(self.n_neurons, dtype=t.long, device=self.dst.device)
        out.scatter_add_(0, self.dst, t.ones_like(self.dst))
        return out

    def out_degree(self) -> Any:
        t = require_torch()
        out = t.zeros(self.n_neurons, dtype=t.long, device=self.src.device)
        out.scatter_add_(0, self.src, t.ones_like(self.src))
        return out

    def outgoing_ids(self, neuron_id: int) -> Any:
        t = require_torch()
        return t.nonzero(self.src == int(neuron_id), as_tuple=False).flatten()

    def incoming_ids(self, neuron_id: int) -> Any:
        t = require_torch()
        return t.nonzero(self.dst == int(neuron_id), as_tuple=False).flatten()


class NeuronView3x3:
    """명세 2절의 3x3 조회 뷰.

    ==== =========================================================
    칸   내용
    ==== =========================================================
    [0][0..2]  x, y, z 위치 [mm]
    [1][0]     **출력 연결 목록**의 조회 핸들 (입력 총합이 아니다)
    [1][1]     읽기 전용 임계값 15.0
    [1][2]     현재 출력 이득 P_i
    [2][0]     들어온 사건의 보존 로그 조회 핸들
    [2][1]     막전위·전도도·불응기·현재 판정값 핸들
    [2][2]     영역·층·세포유형·수용장·채널·방향·위상 메타데이터
    ==== =========================================================

    숫자만 있는 선형대수 행렬이 **아니다**. 3행 1열의 과거 로그를 매 스텝 다시
    더하지 않는다. 현재 도착한 미처리 사건만 한 번 반영하고 그 효과는 동적
    상태에 남는다.
    """

    def __init__(self, model: "CorticalModel", neuron_id: int, *, run_id: str = "",
                 sample_id: int = -1, episode_id: int = -1, replica_id: int = 0,
                 event_source: Callable[[], list[dict[str, Any]]] | None = None) -> None:
        self.model = model
        self.neuron_id = int(neuron_id)
        if not (0 <= self.neuron_id < model.neurons.n):
            raise IndexError(f"뉴런 ID 범위를 벗어났다: {neuron_id} "
                             f"(0..{model.neurons.n - 1})")
        self.scope = {"run_id": run_id, "sample_id": int(sample_id),
                      "episode_id": int(episode_id), "replica_id": int(replica_id)}
        self._event_source = event_source

    # -- 각 칸 ---------------------------------------------------------
    @property
    def position_mm(self) -> tuple[float, float, float]:
        p = self.model.neurons.position_mm[self.neuron_id].tolist()
        return (float(p[0]), float(p[1]), float(p[2]))

    @property
    def outgoing(self) -> dict[str, Any]:
        syn = self.model.synapses
        ids = syn.outgoing_ids(self.neuron_id)
        return {
            "handle": "outgoing_connection_list",
            "synapse_ids": [int(v) for v in ids.tolist()],
            "n": int(ids.numel()),
            "dst_ids": [int(v) for v in syn.dst[ids].tolist()],
            "w0_nS": [float(v) for v in syn.w0_nS[ids].tolist()],
            "delay_steps": [int(v) for v in syn.delay_steps[ids].tolist()],
            "note": "출력 연결 목록이다. 입력 총합을 여기에 저장하지 않는다.",
        }

    @property
    def threshold(self) -> float:
        return float(self.model.neurons.threshold_u[self.neuron_id])

    @property
    def output_gain_P(self) -> float:
        return float(self.model.gains.P()[self.scope["replica_id"], self.neuron_id])

    @property
    def input_log(self) -> dict[str, Any]:
        """보존된 입력 사건 조회 핸들. **조회 범위를 반드시 한정한다** (F16)."""
        if self._event_source is None:
            return {"handle": "input_event_log", "available": False,
                    "scope": dict(self.scope),
                    "note": "이 조회에는 사건 로그가 연결되지 않았다. "
                            "'로그에 없으니 입력이 없었다' 고 해석하지 말 것."}
        rows = self._event_source()
        return {"handle": "input_event_log", "available": True,
                "scope": dict(self.scope), "n_events": len(rows), "events": rows}

    @property
    def dynamic_state(self) -> dict[str, Any]:
        st = self.model.state
        r, i = self.scope["replica_id"], self.neuron_id
        if st is None:
            return {"handle": "dynamic_state", "initialized": False}
        b = 0
        V = st.V[r, b, i].tolist()
        EL = self.model.neurons.EL_mV[i].tolist()
        u = (V[0] - EL[0]) / V_UNIT_MV
        return {
            "handle": "dynamic_state", "initialized": True,
            "V_mV": {c: float(V[k]) for k, c in enumerate(COMPARTMENTS)},
            "E_L_mV": {c: float(EL[k]) for k, c in enumerate(COMPARTMENTS)},
            "u_decision_value": float(u), "threshold_u": THRESHOLD,
            "threshold_margin_u": float(u - THRESHOLD),
            "g_nS": {c: {r_: float(st.g[r, b, i, k, m])
                         for m, r_ in enumerate(RECEPTORS)}
                     for k, c in enumerate(COMPARTMENTS)},
            "refractory_until_step": int(st.refrac_until[r, b, i]),
            "last_spike_step": int(st.last_spike[r, b, i]),
            "spike_count": int(st.spike_count[r, b, i]),
            "note": "u 는 무차원 판정값, V 는 mV 다. 둘을 같은 값으로 쓰지 않는다.",
        }

    @property
    def metadata(self) -> dict[str, Any]:
        a = self.model.neurons
        i = self.neuron_id
        ai, li, ci = int(a.area_id[i]), int(a.layer_id[i]), int(a.cell_type_id[i])
        return {
            "handle": "metadata", "neuron_id": i,
            "area": a.area_names[ai] if ai >= 0 else None,
            "layer": a.layer_names[li] if li >= 0 else None,
            "cell_type": a.cell_type_names[ci] if ci >= 0 else None,
            "dale": "excitatory" if int(a.dale[i]) > 0 else "inhibitory",
            "surface_uv_mm": [float(v) for v in a.surface_uv_mm[i].tolist()],
            "visual_field_deg": [float(v) for v in a.visual_field_deg[i].tolist()],
            "rf_sigma_deg": float(a.rf_sigma_deg[i]),
            "channel": (CHANNEL_NAMES[int(a.channel_id[i])]
                        if 0 <= int(a.channel_id[i]) < N_CHANNEL else None),
            "polarity": int(a.polarity[i]),
            "sample_id": int(a.sample_id[i]),
            "pref_orientation_rad": float(a.pref_orientation_rad[i]),
            "pref_phase_rad": float(a.pref_phase_rad[i]),
            "compartments": [c for k, c in enumerate(COMPARTMENTS)
                             if bool(a.has_comp[i, k])],
        }

    def as_matrix(self) -> np.ndarray:
        """``(3,3) dtype=object`` 배열. 수치 행렬로 바꾸려 하지 말 것."""
        m = np.empty((3, 3), dtype=object)
        x, y, z = self.position_mm
        m[0, 0], m[0, 1], m[0, 2] = x, y, z
        m[1, 0], m[1, 1], m[1, 2] = self.outgoing, self.threshold, self.output_gain_P
        m[2, 0], m[2, 1], m[2, 2] = self.input_log, self.dynamic_state, self.metadata
        return m

    def to_dict(self) -> dict[str, Any]:
        x, y, z = self.position_mm
        return {
            "scope": dict(self.scope),
            "row1": {"x_mm": x, "y_mm": y, "z_mm": z},
            "row2": {"outgoing": self.outgoing, "threshold": self.threshold,
                     "output_gain_P": self.output_gain_P},
            "row3": {"input_log": self.input_log, "dynamic_state": self.dynamic_state,
                     "metadata": self.metadata},
        }

    @staticmethod
    def layout_description() -> list[list[str]]:
        return [["x_mm", "y_mm", "z_mm"],
                ["outgoing_connection_list", "threshold_15_readonly", "output_gain_P"],
                ["input_event_log", "dynamic_state", "metadata"]]


# ======================================================================
# 7. 공통 방향 지도와 피질 모델 생성
# ======================================================================
class SharedOrientationMap:
    """한 영역에서 **한 번만** 생성하는 표면 방향 지도 (F06).

    무작위 위상 평면파를 중첩해 pinwheel 모양을 만든다. 모든 층이 자기 표면
    좌표 (u, v) 에서 **같은 지도**를 평가하므로, 같은 (u,v) 이면 층이 달라도
    선호 방향이 같다. 층마다 난수 지도를 다시 만들지 않는다.
    """

    def __init__(self, n_waves: int, hypercolumn_mm: float, rng: np.random.Generator,
                 area: str) -> None:
        self.area = area
        self.n_waves = int(n_waves)
        self.hypercolumn_mm = float(hypercolumn_mm)
        k = 2.0 * math.pi / max(self.hypercolumn_mm, 1e-6)
        ang = rng.uniform(0.0, 2 * math.pi, size=self.n_waves)
        self.kx = k * np.cos(ang)
        self.ky = k * np.sin(ang)
        self.phase = rng.uniform(0.0, 2 * math.pi, size=self.n_waves)
        self.signature = sha256_text(
            json.dumps({"area": area, "n": self.n_waves, "h": self.hypercolumn_mm,
                        "kx": self.kx.tolist(), "ky": self.ky.tolist(),
                        "phase": self.phase.tolist()}, sort_keys=True))

    def evaluate(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """표면 좌표 -> 선호 방향 [0, pi)."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        z = np.zeros(u.shape, dtype=np.complex128)
        for i in range(self.n_waves):
            z += np.exp(1j * (self.kx[i] * u + self.ky[i] * v + self.phase[i]))
        return 0.5 * np.mod(np.angle(z), 2 * math.pi)

    def to_dict(self) -> dict[str, Any]:
        return {"area": self.area, "n_waves": self.n_waves,
                "hypercolumn_mm": self.hypercolumn_mm, "signature": self.signature}


def gabor_coefficient(dx_deg: np.ndarray, dy_deg: np.ndarray, orientation: np.ndarray,
                      phase: np.ndarray, sigma_deg: float, aspect: float,
                      cycles_per_deg: float) -> np.ndarray:
    """**부호 있는** Gabor 계수. 원본 시야 좌표(deg)에서 정의한다.

    절댓값을 취해 극성 없는 입력에 붙이지 않는다. 부호는 호출부에서 ON/OFF 와
    흥분/억제 경로로 나눈다.
    """
    xr = dx_deg * np.cos(orientation) + dy_deg * np.sin(orientation)
    yr = -dx_deg * np.sin(orientation) + dy_deg * np.cos(orientation)
    env = np.exp(-0.5 * ((xr / sigma_deg) ** 2 + (yr / (sigma_deg * aspect)) ** 2))
    return env * np.cos(2 * math.pi * cycles_per_deg * xr + phase)


@dataclass
class BuildReport:
    n_neurons: int
    per_area: dict[str, int]
    per_layer: dict[str, int]
    n_synapses: int
    per_rule: list[dict[str, Any]]
    w0_calibration: dict[str, Any]
    orientation_maps: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CorticalBuilder:
    """설정 -> 뉴런 배열 + 시냅스 배열.

    초기 시냅스 크기 ``w0`` 는 **수렴 수·수용체 시상수·고정 임계값 15** 를 함께
    고려해 생성하고 그 규칙을 기록한다. 생성이 끝나면 과제 학습 중 w0 는 절대
    바뀌지 않는다.
    """

    #: w0 보정 기준. 기록에 남으며 학습 중에는 쓰이지 않는다.
    REFERENCE_PRESYN_RATE_HZ = 40.0
    TARGET_RATIO = 1.2
    FACTOR_BOUNDS = (0.05, 80.0)

    def __init__(self, cfg: dict[str, Any], seeds: SeedStreams, sampler: RetinotopicSampler,
                 device: Any, dtype: Any) -> None:
        require_torch()
        require_scipy()
        self.cfg = cfg
        self.seeds = seeds
        self.sampler = sampler
        self.device = device
        self.dtype = dtype
        self.notes: list[str] = []

    # ------------------------------------------------------------------
    def build(self) -> tuple[NeuronArrays, SynapseArrays, BuildReport,
                             dict[str, SharedOrientationMap]]:
        meta = self._plan_neurons()
        arrays = self._make_arrays(meta)
        maps = self._orientation_maps(arrays, meta)
        syn, per_rule, calib = self._wire(arrays, meta)
        per_area: dict[str, int] = {}
        per_layer: dict[str, int] = {}
        for rec in meta["rows"]:
            per_area[rec["area"]] = per_area.get(rec["area"], 0) + 1
            key = f"{rec['area']}/{rec['layer']}/{rec['cell_type']}"
            per_layer[key] = per_layer.get(key, 0) + 1
        report = BuildReport(
            n_neurons=arrays.n, per_area=per_area, per_layer=per_layer,
            n_synapses=syn.n_edges, per_rule=per_rule, w0_calibration=calib,
            orientation_maps={k: v.to_dict() for k, v in maps.items()},
            notes=list(self.notes))
        return arrays, syn, report, maps

    # ------------------------------------------------------------------
    def _plan_neurons(self) -> dict[str, Any]:
        """뉴런 한 개마다 (영역, 층, 세포유형, 표면좌표, 시야좌표, 채널) 을 정한다."""
        cfg = self.cfg
        rng = self.seeds.numpy("anatomy", 0)
        rows: list[dict[str, Any]] = []
        s = self.sampler

        # --- 망막: (채널 x 격자 표본). 화소 수와 뉴런 수를 구분한다. ----
        ret = cfg["areas"]["Retina"]
        ext = ret["surface_extent_mm"]
        scale_mm = float(ext[0]) / max(2.0 * s.ecc_max, 1e-6)
        for ch_idx, (ch_name, _axis, pol) in enumerate(RETINA_CHANNELS):
            for sid in range(s.n_samples):
                rows.append({
                    "area": "Retina", "layer": "L_input",
                    "cell_type": ret["cell_type"],
                    "uv": (float(s.x_deg[sid] * scale_mm + ext[0] / 2),
                           float(s.y_deg[sid] * scale_mm + ext[1] / 2)),
                    "depth": 0.0,
                    "vf": (float(s.x_deg[sid]), float(s.y_deg[sid])),
                    "rf_sigma": float(max(s.sigma_deg[sid], ret["rf_sigma_deg"])),
                    "channel": ch_idx, "polarity": int(pol), "sample": sid,
                })

        # --- LGN: 채널마다 서로 다른 공간 표본을 고른다 (F04) -----------
        lgn = cfg["areas"]["LGN"]
        ext = lgn["surface_extent_mm"]
        per_ch = int(lgn["relay_per_channel"])
        for ch_idx, (ch_name, _axis, pol) in enumerate(RETINA_CHANNELS):
            pick = np.linspace(0, s.n_samples - 1, num=min(per_ch, s.n_samples))
            chosen = sorted({int(round(v)) for v in pick})
            for sid in chosen:
                rows.append({
                    "area": "LGN", "layer": "L_relay", "cell_type": lgn["cell_type"],
                    "uv": (float(s.x_deg[sid] * (ext[0] / max(2 * s.ecc_max, 1e-6))
                                 + ext[0] / 2),
                           float(s.y_deg[sid] * (ext[1] / max(2 * s.ecc_max, 1e-6))
                                 + ext[1] / 2)),
                    "depth": 0.15,
                    "vf": (float(s.x_deg[sid]), float(s.y_deg[sid])),
                    "rf_sigma": float(lgn["rf_sigma_deg"]),
                    "channel": ch_idx, "polarity": int(pol), "sample": sid,
                })

        # --- 피질 영역: 층 깊이와 표면 좌표, 역 로그-극좌표 시야 대응 ----
        rho_max = float(s.ecc_to_rho(s.ecc_max))
        for aname, spec in cfg["areas"].items():
            if spec["kind"] != "cortex":
                continue
            ext = spec["surface_extent_mm"]
            thick = spec["layer_thickness_mm"]
            depth_top = {}
            acc = 0.0
            for lname in ("L1", "L2", "L3", "L4", "L5", "L6"):
                depth_top[lname] = acc
                acc += float(thick.get(lname, 0.0))
            for lname, count in spec["neurons_per_layer"].items():
                count = int(count)
                if count <= 0:
                    continue
                fracs = spec["cell_type_fractions"][lname]
                types: list[str] = []
                for ct, fr in sorted(fracs.items()):
                    types += [ct] * int(round(float(fr) * count))
                while len(types) < count:
                    types.append(sorted(fracs)[0])
                types = types[:count]
                rng.shuffle(types)
                side = max(1, int(math.ceil(math.sqrt(count))))
                for i in range(count):
                    gy, gx = divmod(i, side)
                    u = (gx + 0.5 + rng.uniform(-0.25, 0.25)) / side * ext[0]
                    v = (gy + 0.5 + rng.uniform(-0.25, 0.25)) / side * ext[1]
                    u = float(min(max(u, 0.0), ext[0]))
                    v = float(min(max(v, 0.0), ext[1]))
                    rho = rho_max * (u / max(ext[0], 1e-9))
                    theta = 2 * math.pi * (v / max(ext[1], 1e-9)) - math.pi
                    ecc = float(s.rho_to_ecc(rho))
                    depth = depth_top[lname] + float(thick.get(lname, 0.1)) * \
                        float(rng.uniform(0.2, 0.8))
                    rows.append({
                        "area": aname, "layer": lname, "cell_type": types[i],
                        "uv": (u, v), "depth": depth,
                        "vf": (ecc * math.cos(theta), ecc * math.sin(theta)),
                        "rf_sigma": float(spec["rf_sigma_deg"]),
                        "channel": -1, "polarity": 0, "sample": -1,
                    })

        area_names = sorted({r["area"] for r in rows})
        layer_names = sorted({r["layer"] for r in rows})
        ct_names = sorted({r["cell_type"] for r in rows})
        return {"rows": rows, "area_names": area_names, "layer_names": layer_names,
                "cell_type_names": ct_names}

    # ------------------------------------------------------------------
    def _make_arrays(self, meta: dict[str, Any]) -> NeuronArrays:
        t = require_torch()
        rows = meta["rows"]
        n = len(rows)
        a = NeuronArrays(n, self.device, self.dtype)
        a.area_names = list(meta["area_names"])
        a.layer_names = list(meta["layer_names"])
        a.cell_type_names = list(meta["cell_type_names"])
        pos = np.zeros((n, 3), dtype=np.float64)
        uv = np.zeros((n, 2), dtype=np.float64)
        vf = np.zeros((n, 2), dtype=np.float64)
        rf = np.zeros(n, dtype=np.float64)
        aid = np.zeros(n, dtype=np.int64)
        lid = np.zeros(n, dtype=np.int64)
        cid = np.zeros(n, dtype=np.int64)
        chan = np.zeros(n, dtype=np.int64)
        pol = np.zeros(n, dtype=np.int64)
        smp = np.zeros(n, dtype=np.int64)
        dale = np.zeros(n, dtype=np.int64)
        has = np.zeros((n, N_COMP), dtype=bool)
        C = np.ones((n, N_COMP)); gL = np.ones((n, N_COMP))
        EL = np.full((n, N_COMP), -70.0); gc = np.zeros((n, N_COMP))
        vr = np.zeros(n); tr = np.zeros(n)
        for i, r in enumerate(rows):
            uvv = r["uv"]
            pos[i] = (uvv[0], uvv[1], r["depth"])
            uv[i] = uvv
            vf[i] = r["vf"]
            rf[i] = r["rf_sigma"]
            aid[i] = a.area_names.index(r["area"])
            lid[i] = a.layer_names.index(r["layer"])
            cid[i] = a.cell_type_names.index(r["cell_type"])
            chan[i] = r["channel"]; pol[i] = r["polarity"]; smp[i] = r["sample"]
            spec = CELL_TYPES[r["cell_type"]]
            dale[i] = 1 if spec["dale"] == "excitatory" else -1
            for k, cname in enumerate(COMPARTMENTS):
                if cname in spec["compartments"]:
                    has[i, k] = True
                    C[i, k] = float(spec["C_pF"][cname])
                    gL[i, k] = float(spec["gL_nS"][cname])
                    EL[i, k] = float(spec["EL_mV"][cname])
            for k, cname in enumerate(COMPARTMENTS):
                if k == 0:
                    continue
                if cname in spec["compartments"]:
                    gc[i, k] = float(spec["g_couple_nS"].get(cname, 0.0))
            vr[i] = float(spec["V_reset_mV"]); tr[i] = float(spec["t_ref_ms"])
        dev, dt = self.device, self.dtype
        a.position_mm = t.tensor(pos, dtype=dt, device=dev)
        a.surface_uv_mm = t.tensor(uv, dtype=dt, device=dev)
        a.visual_field_deg = t.tensor(vf, dtype=dt, device=dev)
        a.rf_sigma_deg = t.tensor(rf, dtype=dt, device=dev)
        a.area_id = t.tensor(aid, dtype=t.long, device=dev)
        a.layer_id = t.tensor(lid, dtype=t.long, device=dev)
        a.cell_type_id = t.tensor(cid, dtype=t.long, device=dev)
        a.channel_id = t.tensor(chan, dtype=t.long, device=dev)
        a.polarity = t.tensor(pol, dtype=t.long, device=dev)
        a.sample_id = t.tensor(smp, dtype=t.long, device=dev)
        a.dale = t.tensor(dale, dtype=t.long, device=dev)
        a.has_comp = t.tensor(has, dtype=t.bool, device=dev)
        a.C_pF = t.tensor(C, dtype=dt, device=dev)
        a.gL_nS = t.tensor(gL, dtype=dt, device=dev)
        a.EL_mV = t.tensor(EL, dtype=dt, device=dev)
        a.g_couple_nS = t.tensor(gc, dtype=dt, device=dev)
        a.V_reset_mV = t.tensor(vr, dtype=dt, device=dev)
        a.t_ref_ms = t.tensor(tr, dtype=dt, device=dev)
        a.threshold_u = t.full((n,), THRESHOLD, dtype=dt, device=dev)
        self._meta_np = {"pos": pos, "uv": uv, "vf": vf, "aid": aid, "lid": lid,
                         "cid": cid, "chan": chan, "pol": pol, "smp": smp,
                         "dale": dale, "gL": gL, "EL": EL}
        return a

    # ------------------------------------------------------------------
    def _orientation_maps(self, arrays: NeuronArrays,
                          meta: dict[str, Any]) -> dict[str, SharedOrientationMap]:
        """영역마다 **한 번만** 지도를 만들고 모든 층이 그것을 평가한다."""
        maps: dict[str, SharedOrientationMap] = {}
        pin = self.cfg["v1"]["pinwheel"]
        phases = [float(p) for p in self.cfg["v1"]["phases_rad"]]
        for k, aname in enumerate(arrays.area_names):
            if self.cfg["areas"][aname]["kind"] != "cortex":
                continue
            rng = self.seeds.numpy("orientation_map", k)
            maps[aname] = SharedOrientationMap(int(pin["n_waves"]),
                                               float(pin["hypercolumn_mm"]), rng, aname)
        uv = self._meta_np["uv"]
        aid = self._meta_np["aid"]
        ori = np.full(arrays.n, np.nan)
        pha = np.full(arrays.n, np.nan)
        rng_ph = self.seeds.numpy("orientation_map", 999)
        for aname, omap in maps.items():
            idx = np.nonzero(aid == arrays.area_names.index(aname))[0]
            if idx.size == 0:
                continue
            ori[idx] = omap.evaluate(uv[idx, 0], uv[idx, 1])
            pha[idx] = np.asarray(phases)[rng_ph.integers(0, len(phases), size=idx.size)]
        t = require_torch()
        arrays.pref_orientation_rad = t.tensor(ori, dtype=self.dtype, device=self.device)
        arrays.pref_phase_rad = t.tensor(pha, dtype=self.dtype, device=self.device)
        self._meta_np["ori"] = ori
        self._meta_np["pha"] = pha
        self.notes.append(
            "방향 지도는 영역마다 한 번 생성하고 모든 층이 같은 지도를 평가한다. "
            "같은 표면 좌표면 층이 달라도 선호 방향이 같다.")
        return maps

    # ------------------------------------------------------------------
    def _select(self, arrays: NeuronArrays, spec: dict[str, Any]) -> np.ndarray:
        m = self._meta_np
        mask = (m["aid"] == arrays.area_names.index(spec["area"]))
        if spec.get("layers"):
            lm = np.zeros_like(mask)
            for l in spec["layers"]:
                if l in arrays.layer_names:
                    lm |= (m["lid"] == arrays.layer_names.index(l))
            mask &= lm
        if spec.get("cell_types"):
            cm = np.zeros_like(mask)
            for c in spec["cell_types"]:
                if c in arrays.cell_type_names:
                    cm |= (m["cid"] == arrays.cell_type_names.index(c))
            mask &= cm
        return np.nonzero(mask)[0]

    def _pairs(self, rule: dict[str, Any], arrays: NeuronArrays, src: np.ndarray,
               dst: np.ndarray, rng: np.random.Generator
               ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """규칙별 (src, dst) 후보 쌍. 전체 거리 행렬을 만들지 않는다."""
        tree_cls = require_scipy()
        m = self._meta_np
        kind = rule["selection"]
        info: dict[str, Any] = {}
        if kind == "local_radius":
            space = rule.get("radius_space", "cortical_3d")
            pts_src = m["uv"][src] if space == "surface" else m["pos"][src]
            pts_dst = m["uv"][dst] if space == "surface" else m["pos"][dst]
            tree = tree_cls(pts_src)
            neigh = tree.query_ball_point(pts_dst, r=float(rule["radius_mm"]))
            s_list, d_list = [], []
            for j, cand in enumerate(neigh):
                if not cand:
                    continue
                c = np.asarray(cand, dtype=np.int64)
                if c.size > int(rule["k"]) > 0:
                    c = rng.choice(c, size=int(rule["k"]), replace=False)
                keep = rng.random(c.size) < float(rule["probability"])
                c = c[keep]
                if c.size:
                    s_list.append(src[c])
                    d_list.append(np.full(c.size, dst[j], dtype=np.int64))
            info["space"] = space
            if not s_list:
                return np.zeros(0, np.int64), np.zeros(0, np.int64), info
            return np.concatenate(s_list), np.concatenate(d_list), info

        # rf_knn / channel_knn : 시야 좌표(deg) 에서 최근접
        s_all, d_all = [], []
        groups: list[tuple[np.ndarray, np.ndarray]] = []
        if kind == "channel_knn":
            for ch in np.unique(m["chan"][dst]):
                if ch < 0:
                    continue
                sg = src[m["chan"][src] == ch]
                dg = dst[m["chan"][dst] == ch]
                if sg.size and dg.size:
                    groups.append((sg, dg))
            info["channel_matched"] = True
        else:
            groups.append((src, dst))
        sigma = float(rule["rf_match_sigma_deg"])
        min_distinct = int(rule.get("min_distinct_sources", 0))
        distinct_counts: list[int] = []
        for sg, dg in groups:
            vf_s = m["vf"][sg]
            vf_d = m["vf"][dg]
            tree = tree_cls(vf_s)
            k = int(min(int(rule["k"]), sg.size))
            dist, idx = tree.query(vf_d, k=k)
            if k == 1:
                dist = dist[:, None]
                idx = idx[:, None]
            prob = float(rule["probability"]) * np.exp(-0.5 * (dist / sigma) ** 2)
            keep = rng.random(prob.shape) < prob
            for j in range(dg.size):
                sel = idx[j][keep[j]]
                if min_distinct > 0:
                    n_distinct = int(np.unique(m["smp"][sg[sel]]).size) if sel.size else 0
                    if n_distinct < min_distinct:
                        sel = idx[j][:max(min_distinct, sel.size)]
                        n_distinct = int(np.unique(m["smp"][sg[sel]]).size)
                    distinct_counts.append(n_distinct)
                if sel.size:
                    s_all.append(sg[sel])
                    d_all.append(np.full(sel.size, dg[j], dtype=np.int64))
        if distinct_counts:
            info["mean_distinct_spatial_sources"] = float(np.mean(distinct_counts))
            info["min_distinct_spatial_sources"] = int(np.min(distinct_counts))
        if not s_all:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), info
        return np.concatenate(s_all), np.concatenate(d_all), info

    # ------------------------------------------------------------------
    def _wire(self, arrays: NeuronArrays, meta: dict[str, Any]
              ) -> tuple[SynapseArrays, list[dict[str, Any]], dict[str, Any]]:
        t = require_torch()
        m = self._meta_np
        cfg = self.cfg
        dt_ms = float(cfg["engine"]["dt_ms"])
        min_steps = int(cfg["engine"]["min_delay_steps"])
        gab = cfg["v1"]["gabor"]
        all_src: list[np.ndarray] = []
        all_dst: list[np.ndarray] = []
        all_comp: list[np.ndarray] = []
        all_rec: list[np.ndarray] = []
        all_w: list[np.ndarray] = []
        all_delay: list[np.ndarray] = []
        all_rule: list[np.ndarray] = []
        per_rule: list[dict[str, Any]] = []
        rule_slices: list[tuple[int, int, int]] = []     # (rule_index, start, stop)
        total = 0
        for r_idx, rule in enumerate(cfg["wiring"]["rules"]):
            if not rule.get("enabled", True):
                per_rule.append({"name": rule["name"], "enabled": False, "n": 0})
                continue
            rng = self.seeds.numpy("wiring", r_idx)
            wrng = self.seeds.numpy("weights", r_idx)
            src = self._select(arrays, rule["src"])
            dst = self._select(arrays, rule["dst"])
            if src.size == 0 or dst.size == 0:
                per_rule.append({"name": rule["name"], "enabled": True, "n": 0,
                                 "n_src": int(src.size), "n_dst": int(dst.size),
                                 "reason": "src 또는 dst 후보가 없다"})
                continue
            s, d, info = self._pairs(rule, arrays, src, dst, rng)
            if s.size == 0:
                per_rule.append({"name": rule["name"], "enabled": True, "n": 0,
                                 "n_src": int(src.size), "n_dst": int(dst.size),
                                 "reason": "후보 쌍이 만들어지지 않았다", **info})
                continue
            # 자기 자신으로 가는 연결은 만들지 않는다.
            keep = s != d
            s, d = s[keep], d[keep]
            if s.size == 0:
                per_rule.append({"name": rule["name"], "enabled": True, "n": 0,
                                 "reason": "자기 연결만 남아 제거됐다", **info})
                continue
            # --- 초기 크기 w0 ---------------------------------------
            base = float(rule["w0_median_nS"]) * np.exp(
                float(rule["w0_sigma"]) * wrng.standard_normal(s.size))
            if rule.get("gabor_initialized", False):
                dx = m["vf"][s, 0] - m["vf"][d, 0]
                dy = m["vf"][s, 1] - m["vf"][d, 1]
                coef = gabor_coefficient(dx, dy, m["ori"][d], m["pha"][d],
                                         float(gab["sigma_deg"]), float(gab["aspect"]),
                                         float(gab["cycles_per_deg"]))
                pol = m["pol"][s]
                # 부호 있는 계수 -> 비음수 ON/OFF 대응식 (abs 를 쓰지 않는다)
                shape = np.where(pol > 0, np.maximum(coef, 0.0),
                                 np.where(pol < 0, np.maximum(-coef, 0.0),
                                          0.25 * np.abs(coef)))
                base = base * shape
            w0 = np.clip(base, 0.0, float(rule["w0_max_nS"]))
            alive = w0 > 1e-9
            s, d, w0 = s[alive], d[alive], w0[alive]
            if s.size == 0:
                per_rule.append({"name": rule["name"], "enabled": True, "n": 0,
                                 "reason": "Gabor 계수가 0 이라 남은 연결이 없다", **info})
                continue
            # --- 지연: 3D 직선 거리를 축삭 길이로 근사 ----------------
            dist = np.linalg.norm(m["pos"][s] - m["pos"][d], axis=1)
            delay_ms = float(rule["synaptic_delay_ms"]) + \
                dist / float(rule["conduction_velocity_mm_per_ms"])
            steps = np.maximum(min_steps, np.ceil(delay_ms / dt_ms)).astype(np.int64)
            comp = np.full(s.size, COMP_INDEX[rule["target_compartment"]], dtype=np.int64)
            rec = np.full(s.size, RECEPTOR_INDEX[rule["receptor"]], dtype=np.int64)
            # 억제는 음의 전도도가 아니라 발신 세포 유형과 역전위가 만든다.
            if (m["dale"][s] < 0).any() and rule["receptor"] != "GABA_A":
                self.notes.append(
                    f"규칙 {rule['name']!r}: 억제성 발신 뉴런이 흥분성 수용체를 표적으로 "
                    f"한다. Dale 규칙에 맞게 GABA_A 로 바꾸었다.")
                rec = np.where(m["dale"][s] < 0, RECEPTOR_INDEX["GABA_A"], rec)
            rule_slices.append((r_idx, total, total + int(s.size)))
            total += int(s.size)
            all_src.append(s); all_dst.append(d); all_comp.append(comp)
            all_rec.append(rec); all_w.append(w0); all_delay.append(steps)
            all_rule.append(np.full(s.size, r_idx, dtype=np.int64))
            per_rule.append({
                "name": rule["name"], "enabled": True, "n": int(s.size),
                "n_src": int(src.size), "n_dst": int(dst.size),
                "selection": rule["selection"], "receptor": rule["receptor"],
                "target_compartment": rule["target_compartment"],
                "gabor_initialized": bool(rule.get("gabor_initialized", False)),
                "mean_in_degree": float(s.size / max(1, np.unique(d).size)),
                "delay_steps_min": int(steps.min()), "delay_steps_max": int(steps.max()),
                "w0_mean_nS_raw": float(w0.mean()), **info,
            })
            if total > int(cfg["wiring"]["max_total_synapses"]):
                raise RuntimeError(
                    f"시냅스 수 {total} 가 한도 {cfg['wiring']['max_total_synapses']} 를 "
                    f"넘었다. 규모를 줄이거나 한도를 명시적으로 올려라. "
                    f"뉴런·연결을 몰래 삭제하지 않는다.")
        if not all_src:
            raise RuntimeError("시냅스가 하나도 만들어지지 않았다. 배선 규칙을 확인하라.")
        src_np = np.concatenate(all_src); dst_np = np.concatenate(all_dst)
        comp_np = np.concatenate(all_comp); rec_np = np.concatenate(all_rec)
        w_np = np.concatenate(all_w); del_np = np.concatenate(all_delay)
        rule_np = np.concatenate(all_rule)
        calib = self._calibrate_w0(arrays, cfg, per_rule, rule_slices,
                                   src_np, dst_np, rec_np, w_np)
        dev = self.device
        syn = SynapseArrays(
            t.tensor(src_np, dtype=t.long, device=dev),
            t.tensor(dst_np, dtype=t.long, device=dev),
            t.tensor(comp_np, dtype=t.long, device=dev),
            t.tensor(rec_np, dtype=t.long, device=dev),
            t.tensor(w_np, dtype=self.dtype, device=dev),
            t.tensor(del_np, dtype=t.long, device=dev),
            t.tensor(rule_np, dtype=t.long, device=dev),
            arrays.n)
        for entry in per_rule:
            f = calib["per_rule_factor"].get(entry["name"])
            if f is not None:
                entry["w0_calibration_factor"] = f
                entry["w0_mean_nS"] = entry["w0_mean_nS_raw"] * f
        return syn, per_rule, calib

    # ------------------------------------------------------------------
    def _calibrate_w0(self, arrays: NeuronArrays, cfg: dict[str, Any],
                      per_rule: list[dict[str, Any]],
                      rule_slices: list[tuple[int, int, int]], src: np.ndarray,
                      dst: np.ndarray, rec: np.ndarray, w0: np.ndarray) -> dict[str, Any]:
        """고정 임계값 15 에 닿을 수 있도록 w0 를 한 번만 보정한다.

        정상상태 근사::

            V_th          = E_L + 15 * V_unit
            g_need [nS]   = gL * (V_th - E_L) / (E_rev - V_th)
            g(R)   [nS]   = deg * w * (tau/1000) * R
            factor        = TARGET_RATIO * g_need / g(R_ref)

        흥분성 규칙에만 적용하고, 억제성 규칙은 같은 표적 영역 흥분성 규칙의 평균
        배율을 따라가 E/I 균형을 유지한다. **보정은 생성 시점에 한 번만** 하며
        이후 과제 학습 중 w0 는 바뀌지 않는다.
        """
        m = self._meta_np
        factors: dict[str, float] = {}
        rows: list[dict[str, Any]] = []
        exc_by_area: dict[str, list[float]] = {}
        rules = cfg["wiring"]["rules"]
        lo, hi = self.FACTOR_BOUNDS
        for r_idx, start, stop in rule_slices:
            rule = rules[r_idx]
            name = rule["name"]
            sl = slice(start, stop)
            rname = RECEPTORS[int(rec[start])]
            params = RECEPTOR_PARAMS[rname]
            targets = np.unique(dst[sl])
            deg = (stop - start) / max(1, targets.size)
            w_mean = float(w0[sl].mean())
            gL = float(m["gL"][targets, 0].mean())
            EL = float(m["EL"][targets, 0].mean())
            v_th = EL + THRESHOLD * V_UNIT_MV
            driving = float(params["E_rev_mV"]) - v_th
            row = {"rule": name, "receptor": rname, "mean_in_degree": float(deg),
                   "w0_mean_raw_nS": w_mean, "gL_nS": gL, "EL_mV": EL,
                   "V_threshold_mV": v_th}
            if driving <= 0.0:                    # 억제성: 임계를 넘길 일이 없다
                row["kind"] = "inhibitory"
                rows.append(row)
                continue
            g_need = gL * (v_th - EL) / driving
            achievable = deg * w_mean * (float(params["tau_ms"]) / MS_PER_S) * \
                self.REFERENCE_PRESYN_RATE_HZ
            factor = float(np.clip(self.TARGET_RATIO * g_need / max(achievable, 1e-12),
                                   lo, hi))
            factors[name] = factor
            w0[sl] *= factor
            area = rule["dst"]["area"]
            exc_by_area.setdefault(area, []).append(factor)
            row.update({"kind": "excitatory", "g_need_nS": g_need,
                        "g_at_reference_nS": achievable, "factor": factor,
                        "w0_mean_calibrated_nS": float(w0[sl].mean())})
            rows.append(row)
        for r_idx, start, stop in rule_slices:
            rule = rules[r_idx]
            name = rule["name"]
            if name in factors:
                continue
            area = rule["dst"]["area"]
            f = float(np.mean(exc_by_area.get(area, [1.0])))
            f = float(np.clip(f, lo, hi))
            factors[name] = f
            w0[slice(start, stop)] *= f
            for row in rows:
                if row["rule"] == name:
                    row["factor"] = f
                    row["w0_mean_calibrated_nS"] = float(w0[slice(start, stop)].mean())
        self.notes.append(
            "w0 는 수렴 수·수용체 시상수·고정 임계값 15 를 함께 고려해 생성 시점에 "
            f"한 번 보정했다 (기준 발화율 {self.REFERENCE_PRESYN_RATE_HZ} Hz, "
            f"목표 배율 {self.TARGET_RATIO}). 이후 과제 학습 중 w0 는 고정이다.")
        return {
            "reference_presyn_rate_hz": self.REFERENCE_PRESYN_RATE_HZ,
            "target_ratio": self.TARGET_RATIO,
            "factor_bounds": list(self.FACTOR_BOUNDS),
            "per_rule_factor": factors, "rows": rows,
            "note_ko": ("정상상태 단일 구획 근사다. 정확한 예측이 아니라 자릿수 맞춤이며, "
                        "실제 전달 여부는 진단(메뉴 3)에서 측정한다."),
        }


# ======================================================================
# 8. 출력 이득 / 지연 ring / 동적 상태 / GPU 엔진
# ======================================================================
class GainParameters:
    """학습하는 **유일한** 파라미터: 뉴런별 출력 이득 P.

    ``P(z) = P_min + (P_max - P_min) * sigmoid(z)`` 로 재매개변수화한다.
    z 는 P 의 좌표 표현일 뿐 별도의 임계값이나 편향이 아니다.
    학습 마스크가 False 인 뉴런은 섭동 단계부터 제외되고 z 가 보존된다.
    """

    def __init__(self, n: int, cfg: dict[str, Any], device: Any, dtype: Any,
                 mask: Any | None = None) -> None:
        t = require_torch()
        tr = cfg["training"]
        self.n = int(n)
        self.P_min = float(tr["P_min"])
        self.P_max = float(tr["P_max"])
        p0 = float(tr["P_init"])
        frac = (p0 - self.P_min) / max(self.P_max - self.P_min, 1e-12)
        frac = float(min(max(frac, 1e-6), 1 - 1e-6))
        z0 = math.log(frac / (1.0 - frac))
        self.z = t.full((self.n,), z0, dtype=dtype, device=device)
        self.mask = (t.ones(self.n, dtype=t.bool, device=device)
                     if mask is None else mask.to(device).bool())
        self.device = device
        self.dtype = dtype
        self._current_P: Any = self.P_from_z(self.z).unsqueeze(0)

    def P_from_z(self, z: Any) -> Any:
        t = require_torch()
        return self.P_min + (self.P_max - self.P_min) * t.sigmoid(z)

    def P(self) -> Any:
        """가장 최근에 엔진에 넘긴 ``[R, N]`` 이득. 3x3 조회가 이것을 읽는다."""
        return self._current_P

    def set_current(self, P: Any) -> None:
        self._current_P = P

    def replicas(self, deltas: Any | None) -> Any:
        """``deltas`` ``[R, N]`` -> ``P`` ``[R, N]``. None 이면 기본 1 복제본."""
        require_torch()          # torch 없으면 여기서 멈춘다
        if deltas is None:
            P = self.P_from_z(self.z).unsqueeze(0)
        else:
            P = self.P_from_z(self.z.unsqueeze(0) + deltas)
        self._current_P = P
        return P

    def stats(self) -> dict[str, float]:
        P = self.P_from_z(self.z)
        sat_lo = float((P <= self.P_min + 1e-4).float().mean())
        sat_hi = float((P >= self.P_max - 1e-4).float().mean())
        return {"P_mean": float(P.mean()), "P_std": float(P.std(unbiased=False)),
                "P_min_seen": float(P.min()), "P_max_seen": float(P.max()),
                "saturated_low_fraction": sat_lo, "saturated_high_fraction": sat_hi,
                "n_trainable": int(self.mask.sum())}

    def state_dict(self) -> dict[str, Any]:
        return {"z": self.z.detach().to("cpu").numpy().tolist(),
                "mask": self.mask.detach().to("cpu").numpy().tolist(),
                "P_min": self.P_min, "P_max": self.P_max}

    def load_state_dict(self, st: dict[str, Any]) -> None:
        t = require_torch()
        self.z = t.tensor(np.asarray(st["z"], dtype=np.float64),
                          dtype=self.dtype, device=self.device)
        self.mask = t.tensor(np.asarray(st["mask"], dtype=bool), device=self.device)
        self.P_min = float(st["P_min"])
        self.P_max = float(st["P_max"])
        self._current_P = self.P_from_z(self.z).unsqueeze(0)


class DelayRing:
    """발신 시점의 P 를 **이미 곱한** 출력 ``q`` 의 원형 버퍼.

    ``q_history[ring, R, B, N]``. 과거 신호를 나중의 P 로 다시 계산하지 않는다.
    슬롯 길이는 ``max_delay + 2`` 이고 첫 스텝은 0 으로 초기화된다.
    """

    def __init__(self, max_delay: int, n_replicas: int, batch: int, n: int,
                 device: Any, dtype: Any) -> None:
        t = require_torch()
        self.length = int(max_delay) + 2
        self.buf = t.zeros((self.length, n_replicas, batch, n), dtype=dtype, device=device)
        self.max_delay = int(max_delay)

    def reset(self) -> None:
        self.buf.zero_()

    def write(self, step: int, q: Any) -> None:
        self.buf[step % self.length] = q

    def read(self, step: int, delay: int) -> Any:
        """시각 ``step`` 에 도착하는, ``delay`` 스텝 전에 발신된 출력."""
        src = step - int(delay)
        if src < 0:
            return None                      # 아직 발신된 적이 없다 (0 으로 취급)
        return self.buf[src % self.length]

    def state_dict(self) -> dict[str, Any]:
        return {"length": self.length, "max_delay": self.max_delay,
                "buf": self.buf.detach().to("cpu").numpy()}


class DynamicState:
    """시간에 따라 변하는 상태. **학습 파라미터가 아니다.**"""

    def __init__(self, n_replicas: int, batch: int, n: int, arrays: NeuronArrays,
                 device: Any, dtype: Any) -> None:
        t = require_torch()
        self.R, self.B, self.N = int(n_replicas), int(batch), int(n)
        self.device, self.dtype = device, dtype
        self.arrays = arrays
        self.V = arrays.EL_mV.view(1, 1, n, N_COMP).expand(self.R, self.B, n, N_COMP).clone()
        self.g = t.zeros((self.R, self.B, n, N_COMP, N_RECEPTOR), dtype=dtype, device=device)
        self.refrac_until = t.full((self.R, self.B, n), -(10 ** 9), dtype=t.long,
                                  device=device)
        self.last_spike = t.full((self.R, self.B, n), -(10 ** 9), dtype=t.long,
                                 device=device)
        self.spike_count = t.zeros((self.R, self.B, n), dtype=t.long, device=device)
        self.last_u = t.zeros((self.R, self.B, n), dtype=dtype, device=device)

    def reset(self) -> None:
        self.V.copy_(self.arrays.EL_mV.view(1, 1, self.N, N_COMP)
                     .expand(self.R, self.B, self.N, N_COMP))
        self.g.zero_()
        self.refrac_until.fill_(-(10 ** 9))
        self.last_spike.fill_(-(10 ** 9))
        self.spike_count.zero_()
        self.last_u.zero_()

    def state_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k).detach().to("cpu").numpy()
                for k in ("V", "g", "refrac_until", "last_spike", "spike_count", "last_u")}

    def load_state_dict(self, st: dict[str, Any]) -> None:
        t = require_torch()
        for k in ("V", "g", "last_u"):
            getattr(self, k).copy_(t.tensor(np.asarray(st[k]), dtype=self.dtype,
                                            device=self.device))
        for k in ("refrac_until", "last_spike", "spike_count"):
            getattr(self, k).copy_(t.tensor(np.asarray(st[k]), dtype=t.long,
                                            device=self.device))


@dataclass
class ExternalDrive:
    """망막에 주는 외생 입력.

    ``mode='current'`` 면 ``values`` 는 pA, ``mode='poisson'`` 이면 nS 전도도
    펄스다. 어느 쪽이든 모든 뉴런은 **같은 임계값 15 판정을 거친다**. Poisson
    사건을 강제 발화로 우회시키지 않는다.
    """

    mode: str
    neuron_ids: Any                 # [n_drive] long
    values: Any                     # [T, B, n_drive]
    receptor: int = RECEPTOR_INDEX["AMPA"]
    compartment: int = COMP_INDEX["soma"]
    meta: dict[str, Any] = field(default_factory=dict)


class EngineDivergence(RuntimeError):
    """수치 발산. 클리핑으로 숨기지 않고 상태와 함께 중단한다."""


class GpuConductanceEngine:
    """공통 시간 루프. CPU/GPU 가 같은 수식·배선·시점 규칙을 쓴다.

    한 스텝 순서 (명세 6.3)::

        1. 지연 ring 에서 이번 도착 분을 gather
        2. 고정 w0 와 곱해 (표적, 구획, 수용체) 별 합산
        3. 외생 입력·잔류 전도도·구획 결합을 반영해 상태 갱신
        4. 고정 임계값 15 와 불응기 판정
        5. q = P * s 계산 후 미래 전달 버퍼에 저장
        6. 기록용 스냅샷 전달

    입력 합산이 끝나기 전에 같은 뉴런의 이번 발화를 계산하지 않는다.
    """

    def __init__(self, cfg: dict[str, Any], arrays: NeuronArrays, syn: SynapseArrays,
                 device: Any, dtype: Any) -> None:
        t = require_torch()
        self.cfg = cfg
        self.a = arrays
        self.syn = syn
        self.device = device
        self.dtype = dtype
        self.dt = float(cfg["engine"]["dt_ms"])
        self.edge_chunk = int(cfg["limits"]["edge_chunk"])
        #: True 면 표적 정렬 + 누적합 세그먼트 경로를 쓴다 (원자적 연산 없음).
        #: False 면 index_add_ (CUDA 에서 원자적 연산, 합산 순서가 실행마다 다를 수 있다).
        self.deterministic = bool(cfg.get("deterministic", False))
        self.arrival_path = ("deterministic_segment_sum" if self.deterministic
                             else "index_add")
        self.tau = t.tensor([float(RECEPTOR_PARAMS[r]["tau_ms"]) for r in RECEPTORS],
                            dtype=dtype, device=device)
        self.E_rev = t.tensor([float(RECEPTOR_PARAMS[r]["E_rev_mV"]) for r in RECEPTORS],
                              dtype=dtype, device=device)
        self.mg_mask = t.tensor([1.0 if RECEPTOR_PARAMS[r]["mg_block"] else 0.0
                                 for r in RECEPTORS], dtype=dtype, device=device)
        self.decay = t.exp(-self.dt / self.tau)
        self.avg_factor = (self.tau / self.dt) * (1.0 - self.decay)
        self.t_ref_steps = t.clamp(
            t.ceil(arrays.t_ref_ms / self.dt).to(t.long), min=1)
        self.counters: dict[str, int] = {}
        self.reset_counters()

    def reset_counters(self) -> None:
        self.counters = {"steps": 0, "spikes_emitted": 0, "arrivals_applied": 0,
                         "exogenous_events": 0}

    # ------------------------------------------------------------------
    def _mg_block(self, V: Any) -> Any:
        """NMDA Mg 차단. **스텝 시작 전압에서 평가한 선형화**이며 완전 암시적 해가 아니다."""
        t = require_torch()
        b = 1.0 / (1.0 + t.exp(-MG_SLOPE * V) * (MG_CONC_MM / MG_SCALE))
        return b.unsqueeze(-1) * self.mg_mask + (1.0 - self.mg_mask)

    def _gather_arrivals(self, ring: DelayRing, step: int, R: int, B: int) -> Any:
        """지연 ring -> ``[R, B, N, C, Rc]`` 도착 전도도 증가분 (모두 비음수).

        두 경로 중 하나를 **명시적으로** 고른다.

        * ``deterministic=False``: ``index_add_`` (중복 표적을 누적한다). CUDA 에서는
          원자적 연산이라 부동소수점 합산 순서가 실행마다 다를 수 있다.
        * ``deterministic=True``: 표적 기준으로 미리 정렬해 둔 순서로 누적합을 내고
          세그먼트 차이를 취한다. 원자적 연산이 없어 같은 하드웨어·버전에서 순서가
          고정된다. 누적합은 float64 로 계산해 정밀도 손실을 줄인다 (메모리를 더 쓴다).

        어느 경우에도 중복 표적의 값을 단순 대입으로 덮어쓰지 않는다.
        """
        t = require_torch()
        n = self.a.n
        M = n * N_COMP * N_RECEPTOR
        flat = t.zeros((R, B, M), dtype=self.dtype, device=self.device)
        applied = 0
        if self.deterministic:
            for delay, idx_sorted, seg_target, seg_end in self.syn.sorted_groups:
                past = ring.read(step, delay)
                if past is None or idx_sorted.numel() == 0:
                    continue
                src = self.syn.src[idx_sorted]
                amount = (past[:, :, src]
                          * self.syn.w0_nS[idx_sorted].view(1, 1, -1)).to(t.float64)
                csum = t.cumsum(amount, dim=-1)
                end_vals = csum[..., seg_end - 1]
                start_vals = t.cat([t.zeros_like(end_vals[..., :1]),
                                    end_vals[..., :-1]], dim=-1)
                totals = (end_vals - start_vals).to(self.dtype)
                flat[:, :, seg_target] = flat[:, :, seg_target] + totals
                applied += int(idx_sorted.numel())
        else:
            flat2 = flat.view(R * B, M)
            for delay, idx in self.syn.delay_groups:
                past = ring.read(step, delay)
                if past is None:
                    continue
                e = int(idx.numel())
                for beg in range(0, e, self.edge_chunk):
                    sub = idx[beg:beg + self.edge_chunk]
                    src = self.syn.src[sub]
                    amount = past[:, :, src] * self.syn.w0_nS[sub].view(1, 1, -1)
                    # 1차원 index + [R*B, E] source -> 중복 표적이 누적된다
                    flat2.index_add_(1, self.syn.flat_target[sub],
                                     amount.reshape(R * B, -1))
                    applied += int(sub.numel())
        self.counters["arrivals_applied"] += applied
        return flat.view(R, B, n, N_COMP, N_RECEPTOR)

    # ------------------------------------------------------------------
    def step(self, state: DynamicState, ring: DelayRing, P: Any, step_index: int,
             drive: ExternalDrive | None = None, drive_row: int | None = None
             ) -> dict[str, Any]:
        """한 스텝. 발화는 ``step_index + 1`` 에 배정된다."""
        t = require_torch()
        a = self.a
        R, B, n = state.R, state.B, state.N

        # 1~2. 도착 합산 -----------------------------------------------
        delta_g = self._gather_arrivals(ring, step_index, R, B)

        # 외생 입력: 전도도 펄스 또는 전류 (같은 임계 판정을 거친다)
        I_ext = t.zeros((R, B, n, N_COMP), dtype=self.dtype, device=self.device)
        n_exo = 0
        if drive is not None and drive_row is not None:
            vals = drive.values[drive_row]                     # [B, n_drive]
            if vals.dim() == 2:
                vals = vals.unsqueeze(0).expand(R, -1, -1)     # 복제본에 동일하게
            ids = drive.neuron_ids
            if drive.mode == "current":
                I_ext[:, :, ids, drive.compartment] += vals
            elif drive.mode == "poisson":
                delta_g[:, :, ids, drive.compartment, drive.receptor] += vals
                n_exo = int((vals > 0).sum())
            else:
                raise ValueError(f"알 수 없는 외생 입력 모드: {drive.mode!r}")
        self.counters["exogenous_events"] += n_exo

        # 3. 상태 갱신 (후향 오일러 + 구획 결합 닫힌 해) ----------------
        g_plus = state.g + delta_g                            # 모두 비음수
        g_bar = g_plus * self.avg_factor.view(1, 1, 1, 1, -1)
        g_next = g_plus * self.decay.view(1, 1, 1, 1, -1)
        g_eff = g_bar * self._mg_block(state.V)
        G = g_eff.sum(dim=-1)                                  # [R,B,N,C]
        E = (g_eff * self.E_rev.view(1, 1, 1, 1, -1)).sum(dim=-1)

        C_dt = a.C_pF / self.dt                                # [N,C]
        gL = a.gL_nS
        EL = a.EL_mV
        g_sb = a.g_couple_nS[:, 1].view(1, 1, n)
        g_sa = a.g_couple_nS[:, 2].view(1, 1, n)
        a_s = C_dt[:, 0].view(1, 1, n) + gL[:, 0].view(1, 1, n) + G[..., 0] + g_sb + g_sa
        a_b = C_dt[:, 1].view(1, 1, n) + gL[:, 1].view(1, 1, n) + G[..., 1] + g_sb
        a_a = C_dt[:, 2].view(1, 1, n) + gL[:, 2].view(1, 1, n) + G[..., 2] + g_sa
        rhs = C_dt.view(1, 1, n, N_COMP) * state.V + \
            (gL * EL).view(1, 1, n, N_COMP) + E + I_ext
        denom = a_s - g_sb ** 2 / a_b - g_sa ** 2 / a_a        # 항상 양수 (문서 참조)
        V_s_free = (rhs[..., 0] + g_sb * rhs[..., 1] / a_b
                    + g_sa * rhs[..., 2] / a_a) / denom

        # 4. 불응기: soma 클램프를 **연립식의 경계조건으로** 적용한다 (F22)
        next_step = step_index + 1
        refractory = state.refrac_until > next_step
        V_s = t.where(refractory, a.V_reset_mV.view(1, 1, n).expand_as(V_s_free), V_s_free)
        V_b = (rhs[..., 1] + g_sb * V_s) / a_b
        V_a = (rhs[..., 2] + g_sa * V_s) / a_a
        V_new = t.stack([V_s, V_b, V_a], dim=-1)
        missing = ~a.has_comp.view(1, 1, n, N_COMP)
        V_new = t.where(missing, EL.view(1, 1, n, N_COMP).expand_as(V_new), V_new)

        if not bool(t.isfinite(V_new).all()):
            bad = t.nonzero(~t.isfinite(V_new), as_tuple=False)[:8].tolist()
            raise EngineDivergence(
                f"막전위에 NaN/Inf 가 생겼다 (step {step_index}). "
                f"클리핑으로 숨기지 않고 중단한다. 예시 인덱스(R,B,N,C): {bad}")

        state.V = V_new
        state.g = g_next

        # 5. 고정 임계값 15 판정 ----------------------------------------
        u = (V_new[..., 0] - EL[:, 0].view(1, 1, n)) / V_UNIT_MV
        state.last_u = u
        spikes = (u >= a.threshold_u.view(1, 1, n)) & (~refractory)
        s_f = spikes.to(self.dtype)
        if bool(spikes.any()):
            state.V[..., 0] = t.where(spikes, a.V_reset_mV.view(1, 1, n), state.V[..., 0])
            state.refrac_until = t.where(
                spikes, next_step + self.t_ref_steps.view(1, 1, n), state.refrac_until)
            state.last_spike = t.where(spikes, t.full_like(state.last_spike, next_step),
                                       state.last_spike)
            state.spike_count += spikes.to(t.long)
        n_spk = int(spikes.sum())
        self.counters["spikes_emitted"] += n_spk

        # 6. q = P * s 를 발신 시점의 P 와 곱해 ring 에 저장 -------------
        q = P.unsqueeze(1) * s_f                               # [R,B,N]
        ring.write(next_step, q)
        self.counters["steps"] += 1
        return {"step": step_index, "spike_step": next_step, "spikes": spikes,
                "q": q, "u": u, "n_spikes": n_spk,
                "delta_g": delta_g, "refractory": refractory}


# ======================================================================
# 9. 고정 해독기
# ======================================================================
class FixedITDecoder:
    """학습하지 않는 IT 해독기.

    지정 영역의 흥분성 집단을 **고정 시드**로 균형 있게 클래스 그룹에 나눈다.
    그룹·스케일·편향은 고정이며 어떤 것도 학습하지 않는다. 이 분할은 생물학적
    의미를 추출한 결과가 아니라 인공적인 해독 규칙이다.
    """

    def __init__(self, cfg: dict[str, Any], arrays: NeuronArrays, seeds: SeedStreams,
                 device: Any, dtype: Any) -> None:
        t = require_torch()
        dec = cfg["decoder"]
        self.cfg = cfg
        self.classes = list(dec["classes"])
        self.scale = float(dec["readout_scale"])
        self.rate_unit = float(dec["rate_unit_hz"])
        self.bias = float(dec["bias"])
        self.device, self.dtype = device, dtype
        idx = arrays.indices_of(dec["area"], dec["layers"], dec["cell_types"])
        exc = arrays.dale[idx] > 0
        idx = idx[exc]
        if int(idx.numel()) < len(self.classes):
            raise RuntimeError(
                f"해독 대상 뉴런이 클래스 수보다 적다: {int(idx.numel())} < "
                f"{len(self.classes)}. preset 규모를 키우거나 decoder.area 를 바꾸라.")
        self.neuron_ids = idx
        order = seeds.numpy("decoder_groups", 0).permutation(int(idx.numel()))
        group = np.zeros(int(idx.numel()), dtype=np.int64)
        for rank, pos in enumerate(order):
            group[pos] = rank % len(self.classes)      # 균형 분할
        self.group = t.tensor(group, dtype=t.long, device=device)
        onehot = t.zeros((len(self.classes), int(idx.numel())), dtype=dtype, device=device)
        onehot[self.group, t.arange(int(idx.numel()), device=device)] = 1.0
        counts = onehot.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.group_mean = onehot / counts              # [C, n_dec] 고정 평균 연산자
        start = float(dec["readout_start_ms"])
        end = float(dec["readout_end_ms"])
        dt = float(cfg["engine"]["dt_ms"])
        self.start_step = int(math.ceil(start / dt))
        self.end_step = int(math.ceil(end / dt))       # 반개구간 [start, end)
        self.window_seconds = max((self.end_step - self.start_step) * dt / MS_PER_S, 1e-9)
        self.signature = sha256_text(json.dumps(
            {"ids": [int(v) for v in idx.tolist()], "group": group.tolist(),
             "scale": self.scale, "rate_unit": self.rate_unit, "bias": self.bias,
             "window": [self.start_step, self.end_step]}, sort_keys=True))

    def in_window(self, spike_step: int) -> bool:
        """반개구간 ``[start_step, end_step)`` 판정 (F11)."""
        return self.start_step <= spike_step < self.end_step

    def rates(self, accumulated_q: Any) -> Any:
        """창 안에서 모은 ``q`` 합 ``[R,B,n_dec]`` -> ``r_i`` [Hz]."""
        return accumulated_q / self.window_seconds

    def logits(self, rates: Any) -> Any:
        t = require_torch()
        return self.scale * t.einsum("cn,rbn->rbc", self.group_mean, rates) \
            / self.rate_unit + self.bias

    def loss(self, logits: Any, labels: Any) -> tuple[Any, Any]:
        """평균 교차엔트로피와 표본별 값. ``labels`` 는 ``[B]`` long."""
        t = require_torch()
        R, B, C = logits.shape
        flat = logits.reshape(R * B, C)
        lab = labels.view(1, B).expand(R, B).reshape(R * B)
        per = t.nn.functional.cross_entropy(flat, lab, reduction="none").view(R, B)
        return per.mean(dim=1), per

    def diagnostics(self, rates: Any, logits: Any, labels: Any) -> dict[str, Any]:
        """영특징 비율·분산·다수 기준선·클래스별 정확도. 동률 처리도 밝힌다."""
        t = require_torch()
        R, B, C = logits.shape
        pred = logits.argmax(dim=-1)
        lab = labels.view(1, B).expand(R, B)
        correct = (pred == lab).to(self.dtype)
        zero_feature = (rates.abs().sum(dim=-1) <= 0).to(self.dtype)
        spread = logits.max(dim=-1).values - logits.min(dim=-1).values
        tie = (spread <= 1e-12).to(self.dtype)
        per_class: dict[str, float] = {}
        for c, name in enumerate(self.classes):
            sel = (lab == c)
            per_class[name] = float((correct * sel.to(self.dtype)).sum()
                                    / sel.to(self.dtype).sum().clamp(min=1.0))
        counts = t.bincount(labels, minlength=C).to(self.dtype)
        return {
            "accuracy": float(correct.mean()),
            "per_class_accuracy": per_class,
            "zero_feature_fraction": float(zero_feature.mean()),
            "tie_fraction": float(tie.mean()),
            "logit_spread_mean": float(spread.mean()),
            "rate_variance": float(rates.var(unbiased=False)),
            "majority_baseline": float(counts.max() / counts.sum().clamp(min=1.0)),
            "note_ko": ("특징이 전부 0 이면 softmax 는 균등 분포다. 동률 argmax 로 나온 "
                        "정확도를 인지 성공으로 읽지 말 것."),
        }

    def confusion(self, logits: Any, labels: Any) -> list[list[int]]:
        pred = logits.argmax(dim=-1).reshape(-1)
        lab = labels.view(1, -1).expand(logits.shape[0], -1).reshape(-1)
        C = len(self.classes)
        mat = np.zeros((C, C), dtype=np.int64)
        for p, l in zip(pred.tolist(), lab.tolist()):
            mat[int(l), int(p)] += 1
        return mat.tolist()

    def to_dict(self) -> dict[str, Any]:
        return {"area": self.cfg["decoder"]["area"], "classes": self.classes,
                "n_neurons": int(self.neuron_ids.numel()),
                "group_sizes": [int((self.group == c).sum()) for c in range(len(self.classes))],
                "readout_steps": [self.start_step, self.end_step],
                "window_seconds": self.window_seconds, "scale": self.scale,
                "rate_unit_hz": self.rate_unit, "bias": self.bias,
                "trainable": False, "signature": self.signature,
                "note_ko": "인공적인 해독 규칙이다. 생물학적 의미 추출이 아니다."}


# ======================================================================
# 10. 모델 조립과 순방향 실행
# ======================================================================
class CorticalModel:
    """설정 -> 완성된 모델. 순방향 실행과 조회를 제공한다."""

    def __init__(self, cfg: dict[str, Any], device: Any, dtype: Any,
                 seeds: SeedStreams) -> None:
        require_torch()          # torch 없으면 여기서 멈춘다
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.seeds = seeds
        self.sampler = RetinotopicSampler(cfg, device, dtype)
        self.encoder = RetinaEncoder(cfg, device, dtype)
        builder = CorticalBuilder(cfg, seeds, self.sampler, device, dtype)
        self.neurons, self.synapses, self.build_report, self.maps = builder.build()
        self.gains = GainParameters(self.neurons.n, cfg, device, dtype)
        self.engine = GpuConductanceEngine(cfg, self.neurons, self.synapses, device, dtype)
        self.decoder = FixedITDecoder(cfg, self.neurons, seeds, device, dtype)
        self.state: DynamicState | None = None
        self.ring: DelayRing | None = None
        self._retina_index = self._make_retina_index()
        self.fixed_hashes = self._compute_fixed_hashes()

    # -- 고정 텐서 해시 ------------------------------------------------
    def _compute_fixed_hashes(self) -> dict[str, str]:
        h = {f"neurons.{k}": v for k, v in self.neurons.fixed_hashes().items()}
        h.update({f"synapses.{k}": v for k, v in self.synapses.fixed_hashes().items()})
        h["decoder.signature"] = self.decoder.signature
        for name, omap in self.maps.items():
            h[f"orientation_map.{name}"] = omap.signature
        return h

    def verify_fixed_unchanged(self) -> dict[str, Any]:
        """고정 파라미터가 그대로인지 해시로 확인한다 (F29)."""
        now = self._compute_fixed_hashes()
        changed = sorted(k for k in self.fixed_hashes
                         if now.get(k) != self.fixed_hashes.get(k))
        return {"unchanged": not changed, "changed_keys": changed,
                "n_checked": len(self.fixed_hashes)}

    # -- 망막 채널/표본 -> 뉴런 인덱스 ---------------------------------
    def _make_retina_index(self) -> Any:
        t = require_torch()
        a = self.neurons
        ret = a.indices_of("Retina")
        ch = a.channel_id[ret]
        sm = a.sample_id[ret]
        idx = t.full((N_CHANNEL, self.sampler.n_samples), -1, dtype=t.long,
                     device=self.device)
        idx[ch, sm] = ret
        if int((idx < 0).sum()) != 0:
            raise RuntimeError("망막 뉴런과 (채널, 표본) 대응이 완전하지 않다.")
        return idx

    @property
    def retina_neuron_ids(self) -> Any:
        return self._retina_index.reshape(-1)

    # -- 입력 부호화 ---------------------------------------------------
    def encode_batch(self, images: Any, *, fit_normalization: str | None = None
                     ) -> tuple[Any, dict[str, Any]]:
        """``[B,3,H,W]`` -> 정규화된 격자 샘플 ``[B,C,S]``."""
        channels = self.encoder.encode(images)
        values, meta = self.sampler.sample(channels)
        if fit_normalization is not None:
            meta["normalization"] = self.encoder.fit_normalization(values,
                                                                  fit_normalization)
        return self.encoder.normalize(values), meta

    # -- 외생 입력 -----------------------------------------------------
    def make_drive(self, normalized: Any, n_steps: int, *, frame_boundaries: Sequence[int]
                   | None = None, rng_key: tuple[str, int] = ("input_noise", 0)
                   ) -> ExternalDrive:
        """정규화 값 -> 외생 입력.

        여러 프레임은 **절대 스텝**으로 배치한다 (F19). 프레임마다 0 부터 다시
        세지 않는다. 복제본(±)에는 **같은** 외생 입력을 준다 (F26).
        """
        t = require_torch()
        d = self.cfg["retina"]["drive"]
        if normalized.dim() == 4:                        # [B, F, C, S] 여러 프레임
            flat = normalized
        else:
            flat = normalized.unsqueeze(1)
        nB, nF, nC, nS = flat.shape
        per_frame = flat.reshape(nB, nF, nC * nS)[:, :, self._flat_order()]
        bounds = list(frame_boundaries) if frame_boundaries is not None else \
            [int(round(i * n_steps / nF)) for i in range(nF)] + [n_steps]
        ids = self.retina_neuron_ids
        vals = t.zeros((n_steps, nB, int(ids.numel())), dtype=self.dtype,
                       device=self.device)
        mode = str(d["mode"])
        gain = float(d["input_gain"])
        if mode == "current":
            amp = float(d["current_per_unit_pA"])
            base = float(d["baseline_pA"])
            for f in range(nF):
                lo, hi = bounds[f], bounds[f + 1]
                vals[lo:hi] = (base + amp * gain * per_frame[:, f]).unsqueeze(0)
        elif mode == "poisson":
            rate = float(d["poisson_rate_hz"])
            pulse = float(d["poisson_pulse_nS"])
            dt = float(self.cfg["engine"]["dt_ms"])
            gen = self.seeds.torch_generator(rng_key[0], rng_key[1], self.device)
            for f in range(nF):
                lo, hi = bounds[f], bounds[f + 1]
                lam = (rate * gain * per_frame[:, f] * dt / MS_PER_S)
                lam = lam.unsqueeze(0).expand(hi - lo, -1, -1)
                draw = t.poisson(lam.contiguous(), generator=gen)
                vals[lo:hi] = draw * pulse
        else:
            raise ValueError(f"retina.drive.mode 는 current|poisson 이다: {mode!r}")
        return ExternalDrive(
            mode=mode, neuron_ids=ids, values=vals,
            meta={"n_frames": nF, "frame_boundaries": bounds,
                  "input_code_range": [float(normalized.min()), float(normalized.max())],
                  "drive_unit": "pA" if mode == "current" else "nS",
                  "note_ko": ("입력 코드값·주입량·실제 발화율은 서로 다른 값이다. "
                              "여기의 숫자를 발화 빈도로 읽지 말 것.")})

    def _flat_order(self) -> Any:
        """``[C*S]`` 순서를 ``retina_neuron_ids`` 순서에 맞춘다."""
        t = require_torch()
        return t.arange(N_CHANNEL * self.sampler.n_samples, device=self.device)

    # -- 상태 준비 -----------------------------------------------------
    def prepare(self, n_replicas: int, batch: int) -> None:
        need = (self.state is None or self.state.R != n_replicas
                or self.state.B != batch)
        if need:
            self.state = DynamicState(n_replicas, batch, self.neurons.n, self.neurons,
                                      self.device, self.dtype)
            self.ring = DelayRing(self.synapses.max_delay, n_replicas, batch,
                                  self.neurons.n, self.device, self.dtype)
        self.state.reset()
        self.ring.reset()

    # -- 순방향 실행 ---------------------------------------------------
    def run(self, drive: ExternalDrive, P: Any, n_steps: int, *,
            labels: Any | None = None, step_hook: Callable[[dict[str, Any]], None] | None = None,
            reset: bool = True) -> dict[str, Any]:
        """자극 한 배치를 끝까지 돌리고 해독 특징까지 만든다.

        ``P`` 는 ``[R, N]``. ``reset=True`` 면 막전위·전도도·불응기·지연 ring 을
        같은 초기 상태로 되돌린다 (기본 정지영상 과제).
        """
        t = require_torch()
        R = int(P.shape[0])
        B = int(drive.values.shape[1])
        if self.state is None or self.state.R != R or self.state.B != B:
            self.prepare(R, B)
        elif reset:
            self.state.reset()
            self.ring.reset()
        self.gains.set_current(P)
        self.engine.reset_counters()
        dec = self.decoder
        acc = t.zeros((R, B, int(dec.neuron_ids.numel())), dtype=self.dtype,
                      device=self.device)
        spikes_by_step = t.zeros(n_steps, dtype=t.long, device=self.device)
        with t.no_grad():                     # 주 학습에 backward 를 쓰지 않는다
            for step in range(n_steps):
                info = self.engine.step(self.state, self.ring, P, step, drive, step)
                sp_step = info["spike_step"]
                if dec.in_window(sp_step):
                    acc += info["q"][:, :, dec.neuron_ids]
                spikes_by_step[step] = info["n_spikes"]
                if step_hook is not None:
                    step_hook(info)
        rates = dec.rates(acc)
        logits = dec.logits(rates)
        out: dict[str, Any] = {
            "rates": rates, "logits": logits, "n_steps": n_steps,
            "spike_count": self.state.spike_count.clone(),
            "spikes_by_step": spikes_by_step,
            "engine_counters": dict(self.engine.counters),
        }
        if labels is not None:
            loss_mean, loss_per = dec.loss(logits, labels)
            out["loss"] = loss_mean
            out["loss_per_sample"] = loss_per
            out["diagnostics"] = dec.diagnostics(rates, logits, labels)
        return out

    # -- 규모 추정 -----------------------------------------------------
    def memory_estimate(self, n_replicas: int, batch: int) -> dict[str, Any]:
        el = 4 if self.dtype == torch.float32 else 8
        n, e = self.neurons.n, self.synapses.n_edges
        ring = (self.synapses.max_delay + 2) * n_replicas * batch * n * el
        state = n_replicas * batch * n * (N_COMP + N_COMP * N_RECEPTOR) * el
        idx = e * 8 * 6
        w = e * el
        px = int(self.cfg["image"]["max_side_px"])
        images = batch * 3 * px * px * el
        channels = batch * N_CHANNEL * px * px * el
        arrivals = n_replicas * batch * n * N_COMP * N_RECEPTOR * el
        total = ring + state + idx + w + images + channels + arrivals
        return {
            "n_neurons": n, "n_synapses": e, "n_replicas": n_replicas, "batch": batch,
            "delay_ring_mb": round(ring / 1024 ** 2, 2),
            "dynamic_state_mb": round(state / 1024 ** 2, 2),
            "edge_index_mb": round((idx + w) / 1024 ** 2, 2),
            "image_and_channels_mb": round((images + channels) / 1024 ** 2, 2),
            "arrival_buffer_mb": round(arrivals / 1024 ** 2, 2),
            "total_estimate_mb": round(total / 1024 ** 2, 2),
            "note_ko": ("실행 전 추정이다. 실제 사용량은 실행 중 측정한다. "
                        "측정하지 않은 속도를 여기 쓰지 않는다."),
        }

    # -- 조회 ----------------------------------------------------------
    def neuron_view(self, neuron_id: int, *, run_id: str = "", sample_id: int = -1,
                    episode_id: int = -1, replica_id: int = 0,
                    event_source: Callable[[], list[dict[str, Any]]] | None = None
                    ) -> NeuronView3x3:
        return NeuronView3x3(self, neuron_id, run_id=run_id, sample_id=sample_id,
                             episode_id=episode_id, replica_id=replica_id,
                             event_source=event_source)


# ======================================================================
# 11. 기록 정책 / 비동기 기록기 / 체크포인트
# ======================================================================
#: 사건 행의 필수 키 (명세 9절). 순서가 곧 저장 열 순서다.
EVENT_FIELDS: tuple[str, ...] = (
    "run_seq", "sample_id", "base_seq", "episode_id", "replica_id",
    "perturbation_id", "phase_code", "global_step", "sample_step", "event_id",
    "parent_spike_id", "src_id", "dst_id", "synapse_id", "emit_step", "arrival_step",
    "compartment", "receptor", "w0_snapshot", "P_emit_snapshot", "amount",
    "amount_unit_code",
)
PHASE_CODES: dict[str, int] = {"base": 0, "plus": 1, "minus": 2, "eval": 3}
AMOUNT_UNIT_CODES: dict[str, int] = {"nS": 0, "pA": 1, "dimensionless": 2}

STATE_FIELDS: tuple[str, ...] = (
    "sample_id", "episode_id", "replica_id", "global_step", "sample_step",
    "neuron_id", "V_soma_mV", "V_basal_mV", "V_apical_mV", "u_value",
    "threshold_u", "g_AMPA_nS", "g_NMDA_nS", "g_GABA_A_nS", "refractory_until_step",
    "spiked",
)
GAIN_FIELDS: tuple[str, ...] = (
    "iteration", "epoch", "batch_index", "neuron_id", "z_before", "z_after",
    "P_before", "P_after", "delta_P", "trainable", "eta", "c", "K",
    "loss_plus", "loss_minus", "perturbation_seed",
)


class RecordingPolicy:
    """무엇을 기록할지 **한 곳에서** 정한다 (F01).

    선택 뉴런 ID 는 여기서 한 번 만들어 기록기와 이벤트 경로가 **같은 목록**을
    쓴다. 선택 여부는 신경 계산에 영향을 주지 않는다.
    """

    def __init__(self, cfg: dict[str, Any], arrays: NeuronArrays,
                 seeds: SeedStreams) -> None:
        rec = cfg["recording"]
        self.mode = str(rec["mode"])
        self.state_every = max(1, int(rec["state_every_steps"]))
        self.event_budget = int(rec["event_budget_rows"])
        self.state_budget = int(rec["state_budget_rows"])
        self.queue_max = int(rec["queue_max_items"])
        n_per_area = int(rec["n_selected_per_area"])
        ids: list[int] = []
        for ai, aname in enumerate(arrays.area_names):
            idx = arrays.indices_of(aname)
            if int(idx.numel()) == 0:
                continue
            take = min(n_per_area, int(idx.numel()))
            step = max(1, int(idx.numel()) // take)
            ids += [int(v) for v in idx[::step][:take].tolist()]
        self.selected_ids: list[int] = sorted(set(ids))
        self.selection_criterion = (
            f"영역마다 균등 간격으로 {n_per_area}개까지 선택. 선택 목록은 기록기와 "
            f"이벤트 경로가 공유한다 (계산에는 영향이 없다).")
        self._allocated = False
        self.per_sample_event_budget = self.event_budget
        self.per_sample_state_budget = self.state_budget

    def allocate(self, n_samples: int) -> dict[str, Any]:
        """전체 자극 수 기준으로 표본별 예산을 배분한다 (F18)."""
        n = max(1, int(n_samples))
        self.per_sample_event_budget = max(1, self.event_budget // n)
        self.per_sample_state_budget = max(1, self.state_budget // n)
        self._allocated = True
        return {"n_samples": n, "event_rows_total": self.event_budget,
                "state_rows_total": self.state_budget,
                "event_rows_per_sample": self.per_sample_event_budget,
                "state_rows_per_sample": self.per_sample_state_budget}

    def wants_events(self) -> bool:
        return self.mode in ("full", "selected")

    def wants_states(self) -> bool:
        return self.mode in ("full", "selected")

    def event_targets(self, n_neurons: int) -> list[int] | None:
        """None 이면 전체(full), 목록이면 그 표적만 기록한다."""
        return None if self.mode == "full" else list(self.selected_ids)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "selected_ids": self.selected_ids,
                "n_selected": len(self.selected_ids),
                "selection_criterion": self.selection_criterion,
                "state_every_steps": self.state_every,
                "event_budget_rows": self.event_budget,
                "state_budget_rows": self.state_budget,
                "budget_allocated": self._allocated,
                "per_sample_event_budget": self.per_sample_event_budget,
                "per_sample_state_budget": self.per_sample_state_budget,
                "retention_note_ko": (
                    "selected 는 선택 ID 의 사건과 전체 통계를 남긴다. 기록하지 않은 "
                    "내용을 '입력이 없었다' 로 해석하면 안 된다.")}


class TableStore:
    """행 단위 append 저장소. **기존 행을 지우거나 덮어쓰지 않는다** (F15)."""

    def __init__(self, run_dir: Path, backend: str) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        if backend == "auto":
            backend = "hdf5" if h5py is not None else "npz"
        if backend == "hdf5" and h5py is None:
            raise RuntimeError(
                "recording.backend='hdf5' 를 요구했지만 h5py 가 없다.\n"
                "  python -m pip install h5py  또는 backend 를 'npz' 로 명시하라. "
                "필수 기능을 조용히 생략하지 않는다.")
        self.backend = backend
        self.committed: dict[str, int] = {}
        self._h5: dict[str, Any] = {}
        self._npz_chunks: dict[str, int] = {}
        self.fallback_note = ("" if backend == "hdf5" else
                              "h5py 가 없어 NPZ 청크 경로로 저장했다. 조회 기능은 같다.")

    def _h5_path(self, table: str) -> Path:
        return self.dir / f"{table}.h5"

    def append(self, table: str, fields: Sequence[str], rows: np.ndarray) -> int:
        """``rows`` 는 ``[n, len(fields)]`` float64. 실제로 기록한 행 수를 돌려준다."""
        if rows.size == 0:
            return 0
        rows = np.ascontiguousarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != len(fields):
            raise ValueError(f"{table}: 행 모양이 필드 수와 다르다 "
                             f"{rows.shape} vs {len(fields)}")
        if self.backend == "hdf5":
            f = self._h5.get(table)
            if f is None:
                f = h5py.File(self._h5_path(table), "a")
                self._h5[table] = f
                if "rows" not in f:
                    f.create_dataset("rows", shape=(0, len(fields)),
                                     maxshape=(None, len(fields)),
                                     dtype="f8", chunks=(min(4096, max(1, rows.shape[0])),
                                                         len(fields)),
                                     compression="gzip", compression_opts=4)
                    f.attrs["fields"] = json.dumps(list(fields), ensure_ascii=False)
            ds = f["rows"]
            start = ds.shape[0]
            ds.resize(start + rows.shape[0], axis=0)
            ds[start:start + rows.shape[0]] = rows
            f.flush()
        else:
            k = self._npz_chunks.get(table, 0)
            np.savez_compressed(self.dir / f"{table}_chunk{k:05d}.npz",
                                rows=rows, fields=np.array(list(fields), dtype=object))
            self._npz_chunks[table] = k + 1
        self.committed[table] = self.committed.get(table, 0) + int(rows.shape[0])
        return int(rows.shape[0])

    def read_all(self, table: str) -> tuple[list[str], np.ndarray]:
        if self.backend == "hdf5":
            p = self._h5_path(table)
            if not p.is_file():
                return [], np.zeros((0, 0))
            with h5py.File(p, "r") as f:
                fields = json.loads(f.attrs["fields"])
                return list(fields), np.asarray(f["rows"][...])
        chunks = sorted(self.dir.glob(f"{table}_chunk*.npz"))
        if not chunks:
            return [], np.zeros((0, 0))
        fields: list[str] = []
        parts = []
        for c in chunks:
            with np.load(c, allow_pickle=True) as z:
                fields = [str(x) for x in z["fields"].tolist()]
                parts.append(z["rows"])
        return fields, np.concatenate(parts, axis=0)

    def close(self) -> None:
        for f in self._h5.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5.clear()


class RecordingBudgetExceeded(RuntimeError):
    """``full`` 기록에서 용량 한도에 도달했다. 이유를 남기고 안전하게 중단한다.

    선택/요약 기록으로 조용히 바꾸지 않는다.
    """


class AsyncRecorder:
    """단일 writer thread 로 파일을 쓴다. 사건을 몰래 버리지 않는다.

    큐가 꽉 차면 **손실 없이 대기**한다. 예산 초과는 이유를 기록하고 그 표에
    대해 명시적으로 중단한다 (조용한 모드 전환을 하지 않는다).
    """

    def __init__(self, run_dir: Path, policy: RecordingPolicy, cfg: dict[str, Any]) -> None:
        self.dir = ensure_writable_dir(Path(run_dir))
        self.policy = policy
        self.cfg = cfg
        prev = self.dir / "manifest.json"
        if prev.is_file():
            # 재개 등으로 같은 폴더를 다시 열 때 이전 manifest 를 남겨 둔다.
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            shutil.copy2(prev, self.dir / f"manifest_prev_{stamp}.json")
        self.store = TableStore(self.dir, str(cfg["recording"]["backend"]))
        self.queue: "queue.Queue[tuple[str, Any] | None]" = queue.Queue(
            maxsize=max(4, policy.queue_max))
        self.manifest: dict[str, Any] = {}
        self.counters: dict[str, int] = {
            "events_emitted": 0, "events_arrived": 0, "events_recorded": 0,
            "events_filtered": 0, "state_rows_recorded": 0,
            "state_rows_skipped_budget": 0, "gain_rows_recorded": 0,
        }
        self.stop_reasons: dict[str, str] = {}
        self._sample_used = {"events": 0, "states": 0}
        self._log_path = self.dir / "run.log"
        self._metrics_path = self.dir / "metrics.jsonl"
        self._errors_path = self.dir / "errors.jsonl"
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._worker, name="recorder",
                                        daemon=True)
        self._thread.start()
        self.manifest["recording"] = {"backend": self.store.backend,
                                      "fallback_note": self.store.fallback_note,
                                      "policy": policy.to_dict()}
        write_json(self.dir / "manifest.json", self.manifest)
        self.log(f"기록 시작: backend={self.store.backend} mode={policy.mode}")
        if self.store.fallback_note:
            self.log(self.store.fallback_note)

    # -- writer thread -------------------------------------------------
    def _worker(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                kind, payload = item
                if kind == "table":
                    table, fields, rows = payload
                    self.store.append(table, fields, rows)
                elif kind == "text":
                    path, line = payload
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
            except BaseException as exc:            # writer 오류를 삼키지 않는다
                self._error = exc
            finally:
                self.queue.task_done()

    def _check_worker(self) -> None:
        if self._error is not None:
            err = self._error
            self._error = None
            raise RuntimeError(f"기록 writer 에서 오류가 났다: "
                               f"{type(err).__name__}: {err}") from err

    # -- 공개 API ------------------------------------------------------
    def log(self, message: str) -> None:
        self._check_worker()
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.queue.put(("text", (self._log_path, f"[{stamp}] {message}")))

    def metric(self, **fields: Any) -> None:
        self._check_worker()
        fields.setdefault("utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.queue.put(("text", (self._metrics_path, dumps(fields, indent=None))))

    def error(self, message: str, exc: BaseException | None = None, **extra: Any) -> None:
        row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "message": message, **extra}
        if exc is not None:
            row["exception"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
        self.queue.put(("text", (self._errors_path, dumps(row, indent=None))))

    def begin_sample(self) -> None:
        self._sample_used = {"events": 0, "states": 0}

    def write_events(self, rows: np.ndarray, n_emitted: int, n_arrived: int,
                     n_filtered: int) -> None:
        """사건 행을 기록한다. 발생/도착/기록/제외 수를 **따로** 센다 (F02)."""
        self._check_worker()
        self.counters["events_emitted"] += int(n_emitted)
        self.counters["events_arrived"] += int(n_arrived)
        self.counters["events_filtered"] += int(n_filtered)
        if rows.size == 0:
            return
        allow = self.policy.per_sample_event_budget - self._sample_used["events"]
        total_left = self.policy.event_budget - self.counters["events_recorded"]
        allow = min(allow, total_left)
        if allow <= 0:
            self._note_stop("events", "예산 소진")
            return
        if rows.shape[0] > allow:
            if self.policy.mode == "full":
                self._note_stop("events",
                                "full 기록 예산 초과: 안전하게 중단한다. 남은 사건을 "
                                "선택/요약으로 몰래 바꾸지 않는다.")
                self.queue.put(("table", ("events", EVENT_FIELDS, rows[:allow])))
                self.counters["events_recorded"] += int(allow)
                self.flush()
                raise RecordingBudgetExceeded(
                    f"full 기록 예산({self.policy.event_budget} 행)을 넘었다. "
                    f"기록한 행 {self.counters['events_recorded']}개까지 저장하고 "
                    f"중단한다. 예산을 늘리거나 recording.mode 를 명시적으로 바꾸라.")
            rows = rows[:allow]
            self._note_stop("events", "표본 예산 초과분을 기록하지 않았다")
        self.queue.put(("table", ("events", EVENT_FIELDS, rows)))
        self.counters["events_recorded"] += int(rows.shape[0])
        self._sample_used["events"] += int(rows.shape[0])

    def write_states(self, rows: np.ndarray) -> None:
        self._check_worker()
        if rows.size == 0:
            return
        allow = self.policy.per_sample_state_budget - self._sample_used["states"]
        total_left = self.policy.state_budget - self.counters["state_rows_recorded"]
        allow = min(allow, total_left)
        if allow <= 0:
            self.counters["state_rows_skipped_budget"] += int(rows.shape[0])
            self._note_stop("states", "예산 소진")
            return
        if rows.shape[0] > allow:
            self.counters["state_rows_skipped_budget"] += int(rows.shape[0] - allow)
            rows = rows[:allow]
            self._note_stop("states", "표본 예산 초과분을 기록하지 않았다")
        self.queue.put(("table", ("states", STATE_FIELDS, rows)))
        self.counters["state_rows_recorded"] += int(rows.shape[0])
        self._sample_used["states"] += int(rows.shape[0])

    def write_gain_updates(self, rows: np.ndarray) -> None:
        self._check_worker()
        if rows.size == 0:
            return
        self.queue.put(("table", ("gain_updates", GAIN_FIELDS, rows)))
        self.counters["gain_rows_recorded"] += int(rows.shape[0])

    def _note_stop(self, table: str, reason: str) -> None:
        if table not in self.stop_reasons:
            self.stop_reasons[table] = reason
            self.log(f"[기록 한도] {table}: {reason}")
            self.error(f"기록 한도: {table}", None, reason=reason,
                       counters=dict(self.counters))

    def committed_rows(self) -> dict[str, int]:
        self.flush()
        return dict(self.store.committed)

    def flush(self) -> None:
        self.queue.join()
        self._check_worker()

    def write_summary_csv(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(self.dir / "summary.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (dumps(v, indent=None) if isinstance(v, (dict, list))
                                else v) for k, v in r.items()})

    def close(self, status: str, reason: str = "") -> None:
        self.flush()
        self.manifest.setdefault("recording", {})
        self.manifest["recording"].update({
            "policy": self.policy.to_dict(), "counters": dict(self.counters),
            "stop_reasons": dict(self.stop_reasons),
            "committed_rows": dict(self.store.committed),
            "backend": self.store.backend, "fallback_note": self.store.fallback_note,
        })
        self.manifest["status"] = status
        if reason:
            self.manifest["status_reason"] = reason
        self.manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(self.dir / "manifest.json", self.manifest)
        self.queue.put(None)
        self._thread.join(timeout=30)
        self.store.close()


class CheckpointManager:
    """임시 파일 -> 원자적 교체. 재개 단위는 **완료된 미니배치**다.

    미완료 ± 쌍/배치의 학습 갱신은 적용하지 않고, 마지막 완료 지점의 입력 정의와
    상태로 다시 실행한다. 지원하지 않는 재개 지점을 지원한다고 표시하지 않는다.
    """

    VERSION = 2

    def __init__(self, run_dir: Path) -> None:
        self.dir = ensure_writable_dir(Path(run_dir) / "checkpoints")

    def save(self, name: str, *, meta: dict[str, Any], tensors: dict[str, np.ndarray]
             ) -> Path:
        base = self.dir / safe_name(name)
        # numpy.savez 는 '.npz' 로 끝나지 않으면 확장자를 덧붙인다. 임시 파일 이름이
        # '.npz' 로 끝나게 해서 os.replace 대상이 어긋나지 않게 한다.
        npz_tmp = Path(str(base) + ".tmp.npz")
        json_tmp = Path(str(base) + ".json.tmp")
        np.savez_compressed(npz_tmp, **{k: np.asarray(v) for k, v in tensors.items()})
        payload = dict(meta)
        payload["checkpoint_version"] = self.VERSION
        payload["saved_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload["tensor_keys"] = sorted(tensors)
        payload["committed"] = True
        json_tmp.write_text(dumps(payload), encoding="utf-8")
        os.replace(npz_tmp, base.with_suffix(".npz"))
        os.replace(json_tmp, base.with_suffix(".json"))
        return base.with_suffix(".json")

    def list(self) -> list[Path]:
        return sorted(self.dir.glob("*.json"))

    def latest(self) -> Path | None:
        items = self.list()
        if not items:
            return None
        return max(items, key=lambda p: p.stat().st_mtime)

    def load(self, path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        meta = read_json(path)
        if int(meta.get("checkpoint_version", -1)) != self.VERSION:
            raise RuntimeError(
                f"체크포인트 버전이 다르다: {meta.get('checkpoint_version')} != "
                f"{self.VERSION}. 새 코드로 옛 결과를 강제 복원하지 않는다. "
                f"별도 실행 폴더에서 새로 시작하라.")
        if not meta.get("committed", False):
            raise RuntimeError("커밋되지 않은 체크포인트다. 사용하지 않는다.")
        with np.load(Path(path).with_suffix(".npz"), allow_pickle=False) as z:
            tensors = {k: np.asarray(z[k]) for k in z.files}
        return meta, tensors

    @staticmethod
    def check_compatible(meta: dict[str, Any], *, code_sha: str, config_sha: str,
                         data_digest: str) -> dict[str, Any]:
        """**쓰기 전에** 호환성을 먼저 확인한다."""
        issues: list[str] = []
        if meta.get("code_sha256") != code_sha:
            issues.append("코드 해시가 다르다 (소스가 바뀌었다)")
        if meta.get("config_sha256") != config_sha:
            issues.append("설정 해시가 다르다")
        if meta.get("data_digest") != data_digest:
            issues.append("자극 정의 해시가 다르다")
        return {"compatible": not issues, "issues": issues}


# ======================================================================
# 12. 출력 이득 SPSA 학습기
# ======================================================================
TRAIN_MODES: tuple[str, ...] = (
    "fixed_gain", "all_neuron_gain_spsa", "it_output_gain_only",
    "global_gain_only", "random_direction_matched",
)

#: 이번 주 모드에서 **지원하지 않는** 학습 규칙. 켜면 조용히 넘어가지 않는다 (F07).
UNSUPPORTED_LEARNING: dict[str, str] = {
    "stdp": "P-only 주 모드에서 제외했다. 가중치를 학습하지 않는다.",
    "weight_mirror": "P-only 주 모드에서 제외했다.",
    "kolen_pollack": "P-only 주 모드에서 제외했다.",
    "rao_apical_correction": ("P-only 주 모드에서 제외했다. 상위->apical 고정 배선은 "
                              "실제로 존재하며 별도 검사(check_15)로 확인한다."),
    "threshold_adaptation": "임계값 15 는 고정이며 학습·적응하지 않는다.",
    "homeostasis": "임계값 항상성 조정을 쓰지 않는다.",
    "learned_bias": "해독기 편향을 학습하지 않는다.",
}


def assert_supported_learning(name: str) -> None:
    if name in UNSUPPORTED_LEARNING:
        raise NotImplementedError(
            f"not_supported: {name} -- {UNSUPPORTED_LEARNING[name]}\n"
            f"  이 이름을 켜도 조용히 아무 일도 하지 않는 대신 여기서 분명히 멈춘다.")


class GainSPSATrainer:
    """최종 과제 오차를 쓰는 두 방향 출력 이득 섭동.

    이것은 신경망의 정확한 미분이 **아니다**. 불연속 발화 때문에 작은 섭동에서
    오차 차이가 0 일 수 있고 큰 섭동에서는 잡음이 크다. 단일 단계의 손실 감소나
    수렴을 보장하지 않는다. 뇌가 이 알고리즘을 쓴다는 주장도 아니다.
    """

    def __init__(self, cfg: dict[str, Any], model: CorticalModel, seeds: SeedStreams,
                 mode: str | None = None) -> None:
        require_torch()          # torch 없으면 여기서 멈춘다
        self.cfg = cfg
        self.model = model
        self.seeds = seeds
        self.mode = str(mode or cfg["training"]["mode"])
        if self.mode not in TRAIN_MODES:
            raise ValueError(f"알 수 없는 학습 조건: {self.mode!r} (가능: {list(TRAIN_MODES)})")
        tr = cfg["training"]
        self.eta = float(tr["eta"])
        self.c = float(tr["c"])
        self.K = int(tr["K"])
        self.p_reg = float(tr["p_regularization"])
        self.iteration = 0
        self.forward_calls = 0
        self.matched_step_norms: list[float] = []
        self.history: list[dict[str, Any]] = []
        self.model.gains.mask = self._build_mask()
        self.global_scalar = 0.0

    # ------------------------------------------------------------------
    def _build_mask(self) -> Any:
        t = require_torch()
        n = self.model.neurons.n
        dev = self.model.device
        if self.mode == "it_output_gain_only":
            m = t.zeros(n, dtype=t.bool, device=dev)
            m[self.model.decoder.neuron_ids] = True
            return m
        return t.ones(n, dtype=t.bool, device=dev)

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "eta": self.eta, "c": self.c, "K": self.K,
            "p_regularization": self.p_reg,
            "optimizer": self.cfg["training"]["optimizer"],
            "trainable": ["z (P 의 좌표)"],
            "n_trainable_neurons": int(self.model.gains.mask.sum()),
            "forward_calls_per_iteration": 0 if self.mode == "fixed_gain" else 2 * self.K,
            "uses_backward": False,
            "note_ko": ("전역 과제 오차를 쓰는 SPSA 계열이다. 정확한 기울기가 아니며 "
                        "손실 감소를 보장하지 않는다."),
        }

    # ------------------------------------------------------------------
    def _directions(self, sub: int) -> Any:
        """Rademacher 방향 ``[K, N]``. 학습 마스크 밖은 **섭동 단계부터** 0 이다."""
        t = require_torch()
        n = self.model.neurons.n
        gen = self.seeds.torch_generator("spsa", sub, self.model.device)
        raw = t.randint(0, 2, (self.K, n), generator=gen, device=self.model.device,
                        dtype=t.long)
        delta = (raw.to(self.model.dtype) * 2.0 - 1.0)
        if self.mode == "global_gain_only":
            # 공통 스칼라 하나만 움직이는 제약 대조: 모든 성분이 같은 값
            sign = delta[:, :1]
            delta = sign.expand(-1, n).clone()
        delta = delta * self.model.gains.mask.view(1, n).to(self.model.dtype)
        return delta

    def step(self, images: Any, labels: Any, n_steps: int, *,
             drive_key: tuple[str, int], epoch: int, batch_index: int,
             recorder: AsyncRecorder | None = None,
             step_hook: Callable[[dict[str, Any]], None] | None = None
             ) -> dict[str, Any]:
        """미니배치 한 번. **완료된 배치에서만** z 를 갱신한다.

        ``step_hook`` 은 스텝마다 호출되며 사건·상태 기록에 쓴다. ± 복제본은 하나의
        실행에서 replica 축으로 나뉘므로 hook 이 두 조건을 같은 스텝에서 본다.
        """
        t = require_torch()
        gains = self.model.gains
        z_before = gains.z.clone()
        normalized, _meta = self.model.encode_batch(images)
        # ± 쌍은 같은 이미지·라벨·초기 상태·외생 사건을 쓴다 (F26)
        drive = self.model.make_drive(normalized, n_steps, rng_key=drive_key)

        if self.mode == "fixed_gain":
            P = gains.replicas(None)
            out = self.model.run(drive, P, n_steps, labels=labels,
                                 step_hook=step_hook)
            self.forward_calls += 1
            rec = {"iteration": self.iteration, "epoch": epoch,
                   "batch_index": batch_index, "mode": self.mode,
                   "loss_base": float(out["loss"][0]), "delta_z_norm": 0.0,
                   "delta_P_max": 0.0, "zero_difference_pairs": 0,
                   "forward_calls": 1, "updated": False,
                   "note_ko": "초기 P 고정 조건이다. z 를 갱신하지 않는다."}
            rec.update({f"diag_{k}": v for k, v in out["diagnostics"].items()
                        if not isinstance(v, (dict, list))})
            self.history.append(rec)
            self.iteration += 1
            return rec

        deltas = self._directions(self.iteration)
        R = 2 * self.K
        rows = t.zeros((R, self.model.neurons.n), dtype=self.model.dtype,
                       device=self.model.device)
        for k in range(self.K):
            rows[2 * k] = self.c * deltas[k]
            rows[2 * k + 1] = -self.c * deltas[k]
        if self.mode == "random_direction_matched":
            # 방향은 무작위, 크기는 주 조건에서 가져온다 (기록에 남긴다)
            rows = rows * 0.0
        P = gains.replicas(rows)
        out = self.model.run(drive, P, n_steps, labels=labels, step_hook=step_hook)
        self.forward_calls += R
        losses = out["loss"]                                  # [R]
        g_hat = t.zeros_like(gains.z)
        zero_pairs = 0
        lp_list, lm_list = [], []
        for k in range(self.K):
            lp = float(losses[2 * k])
            lm = float(losses[2 * k + 1])
            lp_list.append(lp)
            lm_list.append(lm)
            if lp == lm:
                zero_pairs += 1
            g_hat += ((lp - lm) / (2.0 * self.c)) * deltas[k]
        g_hat /= float(self.K)
        g_hat = g_hat * gains.mask.to(g_hat.dtype)

        if self.mode == "random_direction_matched":
            gen = self.seeds.torch_generator("spsa", 10_000 + self.iteration,
                                             self.model.device)
            direction = t.randn(gains.z.shape, generator=gen, device=self.model.device,
                                dtype=self.model.dtype)
            direction = direction * gains.mask.to(direction.dtype)
            norm = float(direction.norm()) or 1.0
            target = (self.matched_step_norms[self.iteration]
                      if self.iteration < len(self.matched_step_norms) else 0.0)
            update = direction / norm * target
            matched_note = ("주 조건의 z 갱신 크기를 그대로 가져와 무작위 방향에 "
                            "적용했다. 방향의 정보와 단순 변화량을 구분하기 위한 대조다.")
        else:
            update = self.eta * g_hat
            matched_note = ""
        if self.p_reg > 0.0:                       # 기본 0. 쓰면 따로 기록한다.
            update = update + self.p_reg * gains.z * gains.mask.to(gains.z.dtype)

        gains.z = gains.z - update
        delta_z = gains.z - z_before
        P_before = gains.P_from_z(z_before)
        P_after = gains.P_from_z(gains.z)
        delta_P = P_after - P_before

        if recorder is not None:
            sel = self.model_selected_ids(recorder)
            if sel:
                idx = t.tensor(sel, dtype=t.long, device=self.model.device)
                arr = np.zeros((len(sel), len(GAIN_FIELDS)), dtype=np.float64)
                arr[:, 0] = self.iteration
                arr[:, 1] = epoch
                arr[:, 2] = batch_index
                arr[:, 3] = np.asarray(sel, dtype=np.float64)
                arr[:, 4] = z_before[idx].detach().to("cpu").numpy()
                arr[:, 5] = gains.z[idx].detach().to("cpu").numpy()
                arr[:, 6] = P_before[idx].detach().to("cpu").numpy()
                arr[:, 7] = P_after[idx].detach().to("cpu").numpy()
                arr[:, 8] = delta_P[idx].detach().to("cpu").numpy()
                arr[:, 9] = gains.mask[idx].detach().to("cpu").numpy().astype(np.float64)
                arr[:, 10] = self.eta
                arr[:, 11] = self.c
                arr[:, 12] = self.K
                arr[:, 13] = float(np.mean(lp_list))
                arr[:, 14] = float(np.mean(lm_list))
                arr[:, 15] = float(self.iteration)
                recorder.write_gain_updates(arr)

        rec = {
            "iteration": self.iteration, "epoch": epoch, "batch_index": batch_index,
            "mode": self.mode, "loss_plus_mean": float(np.mean(lp_list)),
            "loss_minus_mean": float(np.mean(lm_list)),
            "loss_difference_mean": float(np.mean(np.array(lp_list) - np.array(lm_list))),
            "zero_difference_pairs": zero_pairs, "K": self.K,
            "zero_difference_rate": zero_pairs / max(1, self.K),
            "delta_z_norm": float(delta_z.norm()),
            "delta_P_max": float(delta_P.abs().max()),
            "delta_P_mean_abs": float(delta_P.abs().mean()),
            "grad_estimate_norm": float(g_hat.norm()),
            "forward_calls": R, "updated": True,
            "eta": self.eta, "c": self.c,
            "p_regularization_applied": self.p_reg,
        }
        if matched_note:
            rec["matched_note_ko"] = matched_note
        rec.update({f"diag_{k}": v for k, v in out["diagnostics"].items()
                    if not isinstance(v, (dict, list))})
        rec.update(self.model.gains.stats())
        self.history.append(rec)
        self.iteration += 1
        return rec

    @staticmethod
    def model_selected_ids(recorder: AsyncRecorder) -> list[int]:
        return list(recorder.policy.selected_ids)

    def state_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "iteration": self.iteration,
                "forward_calls": self.forward_calls, "eta": self.eta, "c": self.c,
                "K": self.K, "p_regularization": self.p_reg,
                "matched_step_norms": list(self.matched_step_norms),
                "global_scalar": self.global_scalar}

    def load_state_dict(self, st: dict[str, Any]) -> None:
        self.mode = str(st["mode"])
        self.iteration = int(st["iteration"])
        self.forward_calls = int(st["forward_calls"])
        self.eta = float(st["eta"])
        self.c = float(st["c"])
        self.K = int(st["K"])
        self.p_reg = float(st.get("p_regularization", 0.0))
        self.matched_step_norms = [float(v) for v in st.get("matched_step_norms", [])]
        self.global_scalar = float(st.get("global_scalar", 0.0))


# ======================================================================
# 13. 사건 포착과 전달 진단
# ======================================================================
class EventCapture:
    """선택된 표적으로 **실제 도착한** 사건을 행으로 만든다.

    선택 목록은 :class:`RecordingPolicy` 가 만든 것을 그대로 쓴다 (F01).
    발생(emitted)·도착(arrived)·기록(recorded)·제외(filtered) 수를 따로 센다 (F02).
    표적별 마지막 사건 ID 를 각각 계산한다 (F17).
    """

    def __init__(self, model: CorticalModel, policy: RecordingPolicy,
                 max_rows_per_step: int = 4096) -> None:
        t = require_torch()
        self.model = model
        self.policy = policy
        self.max_rows = int(max_rows_per_step)
        syn = model.synapses
        targets = policy.event_targets(model.neurons.n)
        if targets is None:
            self.edge_idx = t.arange(syn.n_edges, device=model.device)
        else:
            sel = t.zeros(model.neurons.n, dtype=t.bool, device=model.device)
            if targets:
                sel[t.tensor(targets, dtype=t.long, device=model.device)] = True
            self.edge_idx = t.nonzero(sel[syn.dst], as_tuple=False).flatten()
        # 지연 그룹마다 선택된 edge 를 미리 골라 둔다. 매 스텝·매 복제본마다
        # 같은 교집합을 다시 계산하지 않는다.
        keep = t.zeros(syn.n_edges, dtype=t.bool, device=model.device)
        if self.edge_idx.numel():
            keep[self.edge_idx] = True
        self.groups: list[tuple[int, Any]] = []
        for delay, idx in syn.delay_groups:
            sub = idx[keep[idx]]
            if sub.numel():
                self.groups.append((int(delay), sub))
        self.event_counter = 0
        self.spike_counter = 0
        self.rows: list[np.ndarray] = []
        self.n_emitted = 0
        self.n_arrived = 0
        self.n_filtered = 0
        self.last_event_id_per_target: dict[int, int] = {}

    def reset(self) -> None:
        self.rows.clear()
        self.n_emitted = 0
        self.n_arrived = 0
        self.n_filtered = 0

    def capture(self, info: dict[str, Any], *, sample_id: int, base_seq: int,
                episode_id: int, phase: str, perturbation_id: int, global_step: int,
                run_seq: int, replica: int = 0, batch_row: int = 0,
                P: Any | None = None) -> None:
        t = require_torch()
        syn = self.model.synapses
        ring = self.model.ring
        step = int(info["step"])
        self.n_emitted += int(info["n_spikes"])
        self.spike_counter += int(info["n_spikes"])
        if self.edge_idx.numel() == 0:
            return
        parts_src, parts_edge, parts_amt, parts_delay = [], [], [], []
        for delay, sub in self.groups:
            past = ring.read(step, delay)
            if past is None:
                continue
            q = past[replica, batch_row][syn.src[sub]]
            amount = q * syn.w0_nS[sub]
            nz = t.nonzero(amount > 0, as_tuple=False).flatten()
            if nz.numel() == 0:
                continue
            parts_edge.append(sub[nz])
            parts_src.append(syn.src[sub[nz]])
            parts_amt.append(amount[nz])
            parts_delay.append(t.full((int(nz.numel()),), delay, dtype=t.long,
                                      device=self.model.device))
        if not parts_edge:
            return
        edge = t.cat(parts_edge)
        amt = t.cat(parts_amt)
        delay = t.cat(parts_delay)
        n = int(edge.numel())
        self.n_arrived += n
        if n > self.max_rows:
            self.n_filtered += n - self.max_rows
            edge, amt, delay = edge[:self.max_rows], amt[:self.max_rows], delay[:self.max_rows]
            n = self.max_rows
        dst = syn.dst[edge]
        src = syn.src[edge]
        ids = np.arange(self.event_counter, self.event_counter + n, dtype=np.float64)
        self.event_counter += n
        arr = np.zeros((n, len(EVENT_FIELDS)), dtype=np.float64)
        col = {k: i for i, k in enumerate(EVENT_FIELDS)}
        arr[:, col["run_seq"]] = run_seq
        arr[:, col["sample_id"]] = sample_id
        arr[:, col["base_seq"]] = base_seq
        arr[:, col["episode_id"]] = episode_id
        arr[:, col["replica_id"]] = replica
        arr[:, col["perturbation_id"]] = perturbation_id
        arr[:, col["phase_code"]] = PHASE_CODES.get(phase, 0)
        arr[:, col["global_step"]] = global_step
        arr[:, col["sample_step"]] = step
        arr[:, col["event_id"]] = ids
        arr[:, col["parent_spike_id"]] = -1.0
        arr[:, col["src_id"]] = src.detach().to("cpu").numpy()
        arr[:, col["dst_id"]] = dst.detach().to("cpu").numpy()
        arr[:, col["synapse_id"]] = edge.detach().to("cpu").numpy()
        d_np = delay.detach().to("cpu").numpy()
        arr[:, col["arrival_step"]] = step
        arr[:, col["emit_step"]] = step - d_np
        arr[:, col["compartment"]] = syn.comp[edge].detach().to("cpu").numpy()
        arr[:, col["receptor"]] = syn.receptor[edge].detach().to("cpu").numpy()
        arr[:, col["w0_snapshot"]] = syn.w0_nS[edge].detach().to("cpu").numpy()
        if P is not None:
            arr[:, col["P_emit_snapshot"]] = P[replica][src].detach().to("cpu").numpy()
        arr[:, col["amount"]] = amt.detach().to("cpu").numpy()
        arr[:, col["amount_unit_code"]] = AMOUNT_UNIT_CODES["nS"]
        for tgt, eid in zip(arr[:, col["dst_id"]].astype(np.int64), ids):
            self.last_event_id_per_target[int(tgt)] = int(eid)   # 표적별 마지막 ID
        self.rows.append(arr)

    def drain(self) -> np.ndarray:
        if not self.rows:
            return np.zeros((0, len(EVENT_FIELDS)), dtype=np.float64)
        out = np.concatenate(self.rows, axis=0)
        self.rows.clear()
        return out


def state_rows(model: CorticalModel, info: dict[str, Any], selected: Sequence[int], *,
               sample_id: int, episode_id: int, replica: int, global_step: int,
               batch_row: int = 0) -> np.ndarray:
    """선택 뉴런의 상태 스냅샷 행."""
    t = require_torch()
    if not selected:
        return np.zeros((0, len(STATE_FIELDS)), dtype=np.float64)
    idx = t.tensor(list(selected), dtype=t.long, device=model.device)
    st = model.state
    V = st.V[replica, batch_row][idx].detach().to("cpu").numpy()
    g = st.g[replica, batch_row][idx, COMP_INDEX["soma"]].detach().to("cpu").numpy()
    u = info["u"][replica, batch_row][idx].detach().to("cpu").numpy()
    sp = info["spikes"][replica, batch_row][idx].detach().to("cpu").numpy()
    ref = st.refrac_until[replica, batch_row][idx].detach().to("cpu").numpy()
    n = len(selected)
    arr = np.zeros((n, len(STATE_FIELDS)), dtype=np.float64)
    col = {k: i for i, k in enumerate(STATE_FIELDS)}
    arr[:, col["sample_id"]] = sample_id
    arr[:, col["episode_id"]] = episode_id
    arr[:, col["replica_id"]] = replica
    arr[:, col["global_step"]] = global_step
    arr[:, col["sample_step"]] = int(info["step"])
    arr[:, col["neuron_id"]] = np.asarray(selected, dtype=np.float64)
    arr[:, col["V_soma_mV"]] = V[:, 0]
    arr[:, col["V_basal_mV"]] = V[:, 1]
    arr[:, col["V_apical_mV"]] = V[:, 2]
    arr[:, col["u_value"]] = u
    arr[:, col["threshold_u"]] = THRESHOLD
    arr[:, col["g_AMPA_nS"]] = g[:, RECEPTOR_INDEX["AMPA"]]
    arr[:, col["g_NMDA_nS"]] = g[:, RECEPTOR_INDEX["NMDA"]]
    arr[:, col["g_GABA_A_nS"]] = g[:, RECEPTOR_INDEX["GABA_A"]]
    arr[:, col["refractory_until_step"]] = ref
    arr[:, col["spiked"]] = sp.astype(np.float64)
    return arr


def transmission_diagnostic(model: CorticalModel, stim: Stimulus, n_steps: int
                            ) -> dict[str, Any]:
    """망막 -> LGN -> V1 전달 진단 (학습 전에 본다).

    입력 코드값, 주입량, **실제 발화율**, 도착 사건 수, 전도도, 임계 여유를
    각각 보고한다. 전달이 약하다고 임계값을 낮추거나 라벨을 입력에 넣지 않는다.
    """
    t = require_torch()
    img = t.tensor(np.stack([stim.frames[0]]).transpose(0, 3, 1, 2),
                   dtype=model.dtype, device=model.device)
    normalized, meta = model.encode_batch(img)
    drive = model.make_drive(normalized, n_steps)
    P = model.gains.replicas(None)
    peak_u: dict[str, float] = {}
    out = model.run(drive, P, n_steps)
    a = model.neurons
    by_area: dict[str, Any] = {}
    duration_s = n_steps * float(model.cfg["engine"]["dt_ms"]) / MS_PER_S
    for aname in a.area_names:
        idx = a.indices_of(aname)
        if int(idx.numel()) == 0:
            continue
        spikes = out["spike_count"][0, 0][idx]
        u = model.state.last_u[0, 0][idx]
        g_soma = model.state.g[0, 0][idx, COMP_INDEX["soma"]].sum(dim=-1)
        by_area[aname] = {
            "n_neurons": int(idx.numel()),
            "total_spikes": int(spikes.sum()),
            "mean_rate_hz": float(spikes.to(model.dtype).mean() / duration_s),
            "max_rate_hz": float(spikes.max().to(model.dtype) / duration_s),
            "silent_fraction": float((spikes == 0).to(model.dtype).mean()),
            "final_u_max": float(u.max()),
            "final_u_mean": float(u.mean()),
            "threshold_margin_u_max": float(u.max() - THRESHOLD),
            "final_g_soma_nS_max": float(g_soma.max()),
        }
        peak_u[aname] = float(u.max())
    chain = ["Retina", "LGN", "V1"]
    broken: list[str] = []
    for i, aname in enumerate(chain):
        if aname in by_area and by_area[aname]["total_spikes"] == 0:
            broken.append(aname)
    return {
        "stimulus_id": stim.stimulus_id, "n_steps": n_steps,
        "input_code": {"min": float(normalized.min()), "max": float(normalized.max()),
                       "mean": float(normalized.mean())},
        "drive": {"mode": drive.mode, "unit": drive.meta["drive_unit"],
                  "max_value": float(drive.values.max()),
                  "mean_value": float(drive.values.mean())},
        "sampling": {k: v for k, v in meta.items() if k != "normalization"},
        "by_area": by_area, "silent_chain_areas": broken,
        "chain_ok": not broken,
        "engine_counters": out["engine_counters"],
        "note_ko": ("입력 코드값·주입량·실제 발화율은 서로 다른 값이다. 평균 정상상태 "
                    "근사만으로 '절대 불가능' 을 단정하지 않는다. 전달이 약하면 "
                    "임계값을 낮추지 말고 w0·입력 단위 설정을 새 실행으로 바꿔라."),
    }


# ======================================================================
# 14. 실험 실행기
# ======================================================================
class ExperimentRunner:
    """실행 폴더 하나를 만들고 그 안에서만 결과를 만든다.

    ``import`` 만으로는 아무 것도 실행되지 않는다. 메뉴나 CLI 가 명시적으로
    호출할 때만 동작한다. 기존 실행 폴더를 덮어쓰지 않는다.
    """

    def __init__(self, cfg: dict[str, Any], *, output_root: Path | None = None,
                 run_dir: Path | None = None, device_choice: str = "auto",
                 env: EnvironmentInfo | None = None) -> None:
        self.cfg = cfg
        self.env = env or EnvironmentInfo.detect()
        self.device_choice = device_choice
        self.device = resolve_device(device_choice, self.env)
        t = require_torch()
        self.dtype = t.float32 if cfg["dtype"] == "float32" else t.float64
        self.seeds = SeedStreams(int(cfg["seed"]))
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.output_root = Path(output_root) if output_root is not None else None
        self.recorder: AsyncRecorder | None = None
        self.model: CorticalModel | None = None
        self.policy: RecordingPolicy | None = None
        self.checkpoints: CheckpointManager | None = None
        self.stims: list[Stimulus] = []
        self.diagnostics: list[Stimulus] = []
        self.splits: dict[str, list[Stimulus]] = {}
        self.data_digest = ""
        self.test_access = {"n_test_evaluations": 0, "events": []}
        self.run_seq = 0
        self.global_step = 0
        if cfg["deterministic"]:
            self._set_deterministic()

    # ------------------------------------------------------------------
    def _set_deterministic(self) -> None:
        """deterministic 옵션을 조용히 무시하지 않는다.

        도착 합산은 원자적 연산을 쓰지 않는 **미리 구현한 결정론 경로**로 바뀐다
        (:meth:`GpuConductanceEngine._gather_arrivals`). 그래도 지원하지 않는 연산이
        나오면 오류로 멈춘다.
        """
        t = require_torch()
        try:
            t.use_deterministic_algorithms(True, warn_only=False)
            self.deterministic_status = "enabled"
            self.deterministic_note = ("결정론적 알고리즘을 켰다. 도착 합산은 표적 정렬 "
                                       "+ 누적합 경로를 쓴다. 지원하지 않는 연산이 "
                                       "나오면 조용히 넘어가지 않고 오류가 난다.")
        except Exception as exc:
            self.deterministic_status = "unavailable"
            self.deterministic_note = (
                f"이 환경에서 결정론적 모드를 켜지 못했다: {type(exc).__name__}: {exc}. "
                f"fast 모드로 실행되며 CUDA 합산 순서 차이가 생길 수 있다.")
        if hasattr(t, "backends") and hasattr(t.backends, "cudnn"):
            try:
                t.backends.cudnn.benchmark = False
            except Exception:
                pass

    # ------------------------------------------------------------------
    def setup(self, tag: str) -> Path:
        if self.run_dir is None:
            if self.output_root is None:
                raise ValueError("결과를 쓸 폴더(--output)가 필요하다.")
            self.run_dir = new_run_dir(self.output_root, tag)
        ensure_writable_dir(self.run_dir)
        self.run_seq = int(hashlib.sha256(self.run_dir.name.encode()).hexdigest()[:8], 16)
        write_json(self.run_dir / "resolved_config.json", self.cfg)
        return self.run_dir

    def _start_recorder(self, arrays: NeuronArrays) -> AsyncRecorder:
        assert self.run_dir is not None
        self.policy = RecordingPolicy(self.cfg, arrays, self.seeds)
        rec = AsyncRecorder(self.run_dir, self.policy, self.cfg)
        rec.manifest = {
            "run_id": self.run_dir.name, "run_seq": self.run_seq,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": __version__, "command": " ".join(sys.argv),
            "code": source_hash(), "config_sha256": config_hash(self.cfg),
            "config": self.cfg, "environment": self.env.to_dict(),
            "device": str(self.device), "dtype": str(self.dtype),
            "deterministic": {"requested": bool(self.cfg["deterministic"]),
                              "status": getattr(self, "deterministic_status", "fast_mode"),
                              "arrival_path": ("deterministic_segment_sum"
                                               if self.cfg["deterministic"]
                                               else "index_add"),
                              "note_ko": getattr(
                                  self, "deterministic_note",
                                  "fast 모드다. CUDA 부동소수점 합산 순서가 실행마다 "
                                  "다를 수 있고 CPU/GPU 비트 일치를 약속하지 않는다.")},
            "units": UNITS,
            "threshold": {"value": THRESHOLD, "v_unit_mV": V_UNIT_MV,
                          "trainable": False},
            "learning": {"trainable": ["z (P 좌표)"], "mode": self.cfg["training"]["mode"],
                         "unsupported_if_enabled": UNSUPPORTED_LEARNING},
            "fix_register": fix_register_rows(),
            "provenance": [asdict(p) for p in PARAM_PROVENANCE],
            "claims_note_ko": ("이 실행의 수치는 모형 파라미터로 만든 모형의 출력이다. "
                               "생물학적 측정값이 아니며 뇌의 복제도 아니다."),
        }
        self.checkpoints = CheckpointManager(self.run_dir)
        return rec

    # ------------------------------------------------------------------
    def build(self, *, with_recorder: bool = True) -> CorticalModel:
        t0 = time.time()
        model = CorticalModel(self.cfg, self.device, self.dtype, self.seeds)
        self.model = model
        if with_recorder:
            self.recorder = self._start_recorder(model.neurons)
            self.recorder.manifest["model"] = model.build_report.to_dict()
            self.recorder.manifest["decoder"] = model.decoder.to_dict()
            self.recorder.manifest["memory_estimate"] = model.memory_estimate(
                1, int(self.cfg["training"]["batch_size"]))
            self.recorder.manifest["fixed_hashes"] = model.fixed_hashes
            self.recorder.log(f"모델 조립 완료: 뉴런 {model.neurons.n}, "
                              f"시냅스 {model.synapses.n_edges} ({time.time() - t0:.1f}s)")
        return model

    def build_data(self) -> dict[str, Any]:
        gen = StimulusGenerator(self.cfg, self.seeds)
        self.stims = gen.classification_set()
        self.diagnostics = gen.diagnostic_set()
        self.splits = gen.split(self.stims)
        report = gen.split_report(self.splits)
        if not report["no_overlap"]:
            raise RuntimeError(f"분할이 겹친다 (base_id 누출): {report['base_overlaps']}")
        digests = {s.stimulus_id: s.digest() for s in self.stims}
        self.data_digest = sha256_text(json.dumps(
            {"stims": digests,
             "split": {k: sorted(s.stimulus_id for s in v)
                       for k, v in self.splits.items()}}, sort_keys=True))
        info = {"split_report": report, "data_digest": self.data_digest,
                "n_classification": len(self.stims),
                "n_diagnostic": len(self.diagnostics),
                "classes": list(self.cfg["decoder"]["classes"]),
                "labels_present": sorted({s.label for s in self.stims}),
                "diagnostic_separate": True}
        missing = set(self.cfg["decoder"]["classes"]) - set(info["labels_present"])
        extra = set(info["labels_present"]) - set(self.cfg["decoder"]["classes"])
        info["class_mismatch"] = {"missing": sorted(missing), "unexpected": sorted(extra)}
        if missing or extra:
            raise RuntimeError(
                f"설정 클래스와 실제 라벨이 다르다. 없는 클래스 {sorted(missing)}, "
                f"예상 밖 라벨 {sorted(extra)} (F10).")
        if self.recorder is not None:
            self.recorder.manifest["data"] = info
        return info

    # ------------------------------------------------------------------
    def _batch_tensor(self, stims: Sequence[Stimulus]) -> tuple[Any, Any]:
        t = require_torch()
        frames = np.stack([s.frames[0] for s in stims]).transpose(0, 3, 1, 2)
        images = t.tensor(frames, dtype=self.dtype, device=self.device)
        classes = list(self.cfg["decoder"]["classes"])
        labels = t.tensor([classes.index(s.label) for s in stims], dtype=t.long,
                          device=self.device)
        return images, labels

    def fit_normalization(self) -> dict[str, Any]:
        """**train 분할로만** 정규화 계수를 추정한다."""
        assert self.model is not None
        train = self.splits["train"]
        if not train:
            raise RuntimeError("train 분할이 비어 있다.")
        take = train[:min(len(train), 16)]
        images, _ = self._batch_tensor(take)
        _, meta = self.model.encode_batch(images, fit_normalization="train")
        info = meta["normalization"]
        if self.recorder is not None:
            self.recorder.manifest["input_normalization"] = info
            self.recorder.log(f"정규화 계수를 train {len(take)}개로 추정했다 "
                              f"(격자 샘플 표현).")
        return info

    @property
    def n_steps(self) -> int:
        return int(round(float(self.cfg["engine"]["sample_ms"])
                         / float(self.cfg["engine"]["dt_ms"])))

    # ------------------------------------------------------------------
    def evaluate(self, split: str, *, max_samples: int = 0,
                 note: str = "") -> dict[str, Any]:
        """한 분할을 평가한다. 평가마다 **현재 P 로 특징을 새로 계산**한다 (F12)."""
        require_torch()          # torch 없으면 여기서 멈춘다
        assert self.model is not None
        stims = list(self.splits[split])
        if max_samples > 0:
            stims = stims[:max_samples]
        if not stims:
            return {"split": split, "n": 0, "status": STATUS_SKIPPED,
                    "reason": "분할이 비어 있다"}
        if split == "test":
            self.test_access["n_test_evaluations"] += 1
            self.test_access["events"].append(
                {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "n": len(stims), "note": note})
        bs = int(self.cfg["training"]["batch_size"])
        losses: list[float] = []
        accs: list[float] = []
        zero_feats: list[float] = []
        conf = np.zeros((len(self.cfg["decoder"]["classes"]),) * 2, dtype=np.int64)
        per_class_hits: dict[str, list[float]] = {}
        for beg in range(0, len(stims), bs):
            sub = stims[beg:beg + bs]
            images, labels = self._batch_tensor(sub)
            normalized, _ = self.model.encode_batch(images)
            drive = self.model.make_drive(normalized, self.n_steps,
                                          rng_key=("input_noise", 1000 + beg))
            P = self.model.gains.replicas(None)
            out = self.model.run(drive, P, self.n_steps, labels=labels)
            losses.append(float(out["loss"][0]))
            d = out["diagnostics"]
            accs.append(float(d["accuracy"]))
            zero_feats.append(float(d["zero_feature_fraction"]))
            conf += np.asarray(self.model.decoder.confusion(out["logits"], labels))
            for k, v in d["per_class_accuracy"].items():
                per_class_hits.setdefault(k, []).append(float(v))
        counts = {c: sum(1 for s in stims if s.label == c)
                  for c in self.cfg["decoder"]["classes"]}
        majority = max(counts.values()) / max(1, sum(counts.values()))
        bal = float(np.mean([float(np.mean(v)) for v in per_class_hits.values()])) \
            if per_class_hits else 0.0
        res = {
            "split": split, "n": len(stims), "status": "completed",
            "cross_entropy": float(np.mean(losses)),
            "accuracy": float(np.mean(accs)),
            "balanced_accuracy": bal,
            "per_class_accuracy": {k: float(np.mean(v)) for k, v in per_class_hits.items()},
            "confusion_matrix": conf.tolist(),
            "zero_feature_fraction": float(np.mean(zero_feats)),
            "majority_baseline": float(majority),
            "class_counts": counts, "note": note,
            "note_ko": ("영특징 비율이 높으면 정확도는 동률 처리 결과일 수 있다. "
                        "다수 기준선과 함께 읽어야 한다."),
        }
        if self.recorder is not None:
            self.recorder.metric(kind="evaluation", **res)
        return res

    # ------------------------------------------------------------------
    def _run_with_recording(self, stims: Sequence[Stimulus], *, sample_offset: int,
                            episode_id: int, phase: str, P: Any,
                            perturbation_id: int = -1) -> dict[str, Any]:
        """자극 배치를 돌리면서 선택 사건·상태를 기록한다."""
        t = require_torch()
        assert self.model is not None and self.recorder is not None
        model, rec = self.model, self.recorder
        images, labels = self._batch_tensor(stims)
        frames = max(len(s.frames) for s in stims)
        if frames > 1:
            seq = np.stack([np.stack(s.frames + [s.frames[-1]] * (frames - len(s.frames)))
                            for s in stims]).transpose(0, 1, 4, 2, 3)
            imgs = t.tensor(seq, dtype=self.dtype, device=self.device)
            chan = [model.encode_batch(imgs[:, f])[0] for f in range(frames)]
            normalized = t.stack(chan, dim=1)
        else:
            normalized, _ = model.encode_batch(images)
        drive = model.make_drive(normalized, self.n_steps,
                                 rng_key=("input_noise", sample_offset))
        capture = EventCapture(model, self.policy) if self.policy.wants_events() else None
        selected = list(self.policy.selected_ids) if self.policy.wants_states() else []
        rec.begin_sample()
        base_seq = int(hashlib.sha256(stims[0].base_id.encode()).hexdigest()[:8], 16)

        def hook(info: dict[str, Any]) -> None:
            if capture is not None:
                capture.capture(info, sample_id=sample_offset, base_seq=base_seq,
                                episode_id=episode_id, phase=phase,
                                perturbation_id=perturbation_id,
                                global_step=self.global_step + int(info["step"]),
                                run_seq=self.run_seq, replica=0, batch_row=0, P=P)
            if selected and int(info["step"]) % self.policy.state_every == 0:
                rec.write_states(state_rows(
                    model, info, selected, sample_id=sample_offset,
                    episode_id=episode_id, replica=0,
                    global_step=self.global_step + int(info["step"])))

        out = model.run(drive, P, self.n_steps, labels=labels, step_hook=hook)
        if capture is not None:
            rows = capture.drain()
            rec.write_events(rows, capture.n_emitted, capture.n_arrived,
                             capture.n_filtered)
            out["event_counts"] = {"emitted": capture.n_emitted,
                                   "arrived": capture.n_arrived,
                                   "filtered": capture.n_filtered,
                                   "rows_built": int(rows.shape[0])}
            out["last_event_id_per_target"] = dict(capture.last_event_id_per_target)
        self.global_step += self.n_steps
        return out

    # ------------------------------------------------------------------
    def run_inspect(self) -> dict[str, Any]:
        """환경·설정·메모리 예상만 계산한다. 시뮬레이션을 실행하지 않는다."""
        model = self.build(with_recorder=False)
        bs = int(self.cfg["training"]["batch_size"])
        info = {
            "environment": self.env.to_dict(),
            "device": str(self.device), "dtype": str(self.dtype),
            "config_sha256": config_hash(self.cfg),
            "preset": self.cfg["meta"]["preset"],
            "model": model.build_report.to_dict(),
            "decoder": model.decoder.to_dict(),
            "memory_estimate_base": model.memory_estimate(1, bs),
            "memory_estimate_spsa": model.memory_estimate(
                2 * int(self.cfg["training"]["K"]), bs),
            "n_steps_per_sample": self.n_steps,
            "recording_mode": self.cfg["recording"]["mode"],
            "experiment_status": STATUS_NOT_RUN,
            "note_ko": ("실행 전 추정이다. 측정하지 않은 속도·정확도를 여기 쓰지 않는다."),
        }
        return info

    def run_diagnose(self) -> dict[str, Any]:
        """망막->LGN->V1 전달 진단 (양성 대조 + 무입력 대조)."""
        self.setup("diagnose")
        model = self.build()
        self.build_data()
        self.fit_normalization()
        rec = self.recorder
        assert rec is not None
        pos = next(s for s in self.diagnostics if s.stimulus_id == "diag_ori_00")
        neg = next(s for s in self.diagnostics if s.stimulus_id == "diag_dark")
        result = {
            "positive_control": transmission_diagnostic(model, pos, self.n_steps),
            "no_input_control": transmission_diagnostic(model, neg, self.n_steps),
            "w0_calibration": model.build_report.w0_calibration,
        }
        p, q = result["positive_control"], result["no_input_control"]
        result["verdict"] = {
            "chain_transmits": bool(p["chain_ok"]),
            "silent_areas_with_input": p["silent_chain_areas"],
            "no_input_is_quiet": bool(q["by_area"].get("V1", {}).get("total_spikes", 0) == 0),
            "note_ko": ("작은 회로 통과를 전체 계층 통과로 대신하지 않는다. 여기 결과는 "
                        "이 preset 의 실제 전체 배선에서 측정한 것이다."),
        }
        rec.metric(kind="diagnose", **{k: v for k, v in result["verdict"].items()})
        write_json(self.run_dir / "diagnose.json", result)
        rec.close("completed")
        return result

    def run_simulate(self, *, stimulus_id: str | None = None,
                     image_path: Path | None = None) -> dict[str, Any]:
        """지정 영상/도형 하나를 시뮬레이션하고 기록한다."""
        self.setup("simulate")
        model = self.build()
        self.build_data()
        self.fit_normalization()
        rec = self.recorder
        assert rec is not None
        if image_path is not None:
            arr = load_image_file(Path(image_path), int(self.cfg["image"]["max_side_px"]))
            stim = Stimulus("user_image", "user_image", "unknown", "classification",
                            [arr], {"source": str(image_path)})
        elif stimulus_id is not None:
            pool = {s.stimulus_id: s for s in self.stims + self.diagnostics}
            if stimulus_id not in pool:
                raise KeyError(f"자극 ID 를 찾을 수 없다: {stimulus_id!r}. "
                               f"예: {sorted(pool)[:5]} ...")
            stim = pool[stimulus_id]
        else:
            stim = self.splits["train"][0]
        self.policy.allocate(1)
        P = model.gains.replicas(None)
        out = self._run_with_recording([stim], sample_offset=0, episode_id=0,
                                       phase="eval", P=P)
        a = model.neurons
        duration_s = self.n_steps * float(self.cfg["engine"]["dt_ms"]) / MS_PER_S
        by_area = {}
        for aname in a.area_names:
            idx = a.indices_of(aname)
            spikes = out["spike_count"][0, 0][idx]
            by_area[aname] = {
                "n": int(idx.numel()), "spikes": int(spikes.sum()),
                "mean_rate_hz": float(spikes.to(self.dtype).mean() / duration_s),
                "silent_fraction": float((spikes == 0).to(self.dtype).mean())}
        res = {"stimulus_id": stim.stimulus_id, "label": stim.label,
               "n_steps": self.n_steps, "by_area": by_area,
               "engine_counters": out["engine_counters"],
               "event_counts": out.get("event_counts", {}),
               "logits": out["logits"][0, 0].tolist(),
               "classes": list(self.cfg["decoder"]["classes"]),
               "committed_rows": rec.committed_rows()}
        rec.metric(kind="simulate", **{k: v for k, v in res.items()
                                       if not isinstance(v, (dict, list))})
        write_json(self.run_dir / "simulate.json", res)
        rec.manifest["last_event_id_per_target"] = out.get("last_event_id_per_target", {})
        rec.close("completed")
        return res

    # ------------------------------------------------------------------
    def _training_record_hook(self, trainer: "GainSPSATrainer", rec: AsyncRecorder, *,
                              sample_id: int, episode_id: int, base_id: str
                              ) -> tuple[Callable[[dict[str, Any]], None],
                                         Callable[[], dict[str, Any]]]:
        """학습 중에도 선택 뉴런의 사건·상태를 기록한다.

        **보존 정책**: ± 두 조건을 모두 남기되 배치의 첫 표본(batch_row 0)만 기록한다.
        예산은 :meth:`RecordingPolicy.allocate` 가 표본 수로 미리 나눠 둔 값을 쓴다.
        기록하지 않은 나머지를 '입력이 없었다' 로 읽으면 안 된다.
        """
        assert self.model is not None and self.policy is not None
        model = self.model
        if not (self.policy.wants_events() or self.policy.wants_states()):
            def skip() -> dict[str, Any]:
                self.global_step += self.n_steps   # 기록을 안 해도 시각은 흐른다
                return {"recorded": False, "reason": "recording.mode=summary"}
            return (lambda info: None), skip
        capture = EventCapture(model, self.policy) if self.policy.wants_events() else None
        selected = list(self.policy.selected_ids) if self.policy.wants_states() else []
        base_seq = int(hashlib.sha256(base_id.encode()).hexdigest()[:8], 16)
        rec.begin_sample()
        phases = (["base"] if trainer.mode == "fixed_gain"
                  else ["plus" if r % 2 == 0 else "minus"
                        for r in range(2 * trainer.K)])

        def hook(info: dict[str, Any]) -> None:
            n_rep = int(info["q"].shape[0])
            if capture is not None:
                # 이 스텝에 실제로 쓰인 P 를 읽는다 (model.run 이 먼저 set_current 한다).
                P_now = model.gains.P()
                for r in range(min(n_rep, len(phases))):
                    capture.capture(
                        info, sample_id=sample_id, base_seq=base_seq,
                        episode_id=episode_id, phase=phases[r],
                        perturbation_id=trainer.iteration,
                        global_step=self.global_step + int(info["step"]),
                        run_seq=self.run_seq, replica=r, batch_row=0, P=P_now)
            if selected and int(info["step"]) % self.policy.state_every == 0:
                rec.write_states(state_rows(
                    model, info, selected, sample_id=sample_id,
                    episode_id=episode_id, replica=0,
                    global_step=self.global_step + int(info["step"])))

        def finish() -> dict[str, Any]:
            self.global_step += self.n_steps
            if capture is None:
                return {"recorded": False, "reason": "recording.mode 가 사건을 남기지 않는다"}
            rows = capture.drain()
            rec.write_events(rows, capture.n_emitted, capture.n_arrived,
                             capture.n_filtered)
            return {"recorded": True, "emitted": capture.n_emitted,
                    "arrived": capture.n_arrived, "filtered": capture.n_filtered,
                    "rows_built": int(rows.shape[0]),
                    "replicas_recorded": len(phases), "batch_rows_recorded": 1}

        return hook, finish

    # ------------------------------------------------------------------
    def _checkpoint_payload(self, trainer: GainSPSATrainer, *, epoch: int,
                            next_batch_index: int, next_sample_index: int,
                            order: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        assert self.model is not None and self.recorder is not None
        model = self.model
        meta = {
            "run_id": self.run_dir.name, "code_sha256": source_hash().get("sha256", ""),
            "config_sha256": config_hash(self.cfg), "data_digest": self.data_digest,
            "epoch": int(epoch), "next_batch_index": int(next_batch_index),
            "next_sample_index": int(next_sample_index),
            "global_step": int(self.global_step),
            "trainer": trainer.state_dict(),
            "encoder": model.encoder.state_dict(),
            "rng": self.seeds.state_dict(),
            "split_ids": {k: [s.stimulus_id for s in v] for k, v in self.splits.items()},
            "split_base_ids": {k: sorted({s.base_id for s in v})
                               for k, v in self.splits.items()},
            "test_access": self.test_access,
            "committed_rows": self.recorder.committed_rows(),
            "fixed_hashes": model.fixed_hashes,
            "readout_window_steps": [model.decoder.start_step, model.decoder.end_step],
            "train_mask_sum": int(model.gains.mask.sum()),
            "resume_unit": "completed_minibatch",
            "resume_note_ko": ("재개 단위는 완료된 미니배치다. 중단된 ± 쌍의 갱신은 "
                               "적용하지 않는다. 시퀀스 중간 재개는 지원하지 않는다."),
        }
        tensors: dict[str, np.ndarray] = {
            "z": model.gains.z.detach().to("cpu").numpy(),
            "mask": model.gains.mask.detach().to("cpu").numpy(),
            "order": np.asarray(order, dtype=np.int64),
        }
        if model.state is not None:
            for k, v in model.state.state_dict().items():
                tensors[f"state_{k}"] = v
        if model.ring is not None:
            tensors["ring_buf"] = model.ring.buf.detach().to("cpu").numpy()
        if model.encoder.scale is not None:
            tensors["norm_scale"] = model.encoder.scale.detach().to("cpu").numpy()
        return meta, tensors

    def run_train(self, *, mode: str | None = None, resume_from: Path | None = None,
                  epochs: int | None = None, max_samples: int = 0,
                  matched_step_norms: Sequence[float] | None = None,
                  stop_after_batches: int = 0) -> dict[str, Any]:
        """출력 이득 P 만 학습한다. 다른 파라미터는 건드리지 않는다.

        ``matched_step_norms`` 는 ``random_direction_matched`` 대조군에서만 쓴다.
        주 조건의 z 갱신 크기를 그대로 가져온다는 사실이 기록에 남는다.
        """
        require_torch()          # torch 없으면 여기서 멈춘다
        if resume_from is None:
            self.setup("train")
        model = self.build()
        self.build_data()
        self.fit_normalization()
        rec = self.recorder
        assert rec is not None and self.policy is not None
        trainer = GainSPSATrainer(self.cfg, model, self.seeds, mode=mode)
        if matched_step_norms is not None:
            trainer.matched_step_norms = [float(v) for v in matched_step_norms]
            rec.manifest["matched_step_norms"] = trainer.matched_step_norms
            rec.manifest["matched_note_ko"] = (
                "이 조건의 갱신 크기는 주 조건(all_neuron_gain_spsa)에서 가져왔다.")
        n_epochs = int(epochs if epochs is not None else self.cfg["training"]["epochs"])
        bs = int(self.cfg["training"]["batch_size"])
        train = list(self.splits["train"])
        if max_samples > 0:
            train = train[:max_samples]
        n_batches = max(1, math.ceil(len(train) / bs))
        self.policy.allocate(max(1, len(train) * max(1, n_epochs)))
        rec.manifest["recording_budget"] = self.policy.to_dict()
        rec.manifest["training_retention_ko"] = (
            "학습 중 사건 기록은 배치의 첫 표본(batch_row 0)만 남기고 ± 두 조건을 "
            "모두 남긴다. 상태 기록은 replica 0 만 남긴다. 예산은 표본 수로 미리 "
            "나눈다. 남기지 않은 부분을 '입력이 없었다' 로 해석하면 안 된다.")
        rec.manifest["trainer"] = trainer.describe()
        rec.manifest["experiment_status"] = "running"
        before = model.gains.P_from_z(model.gains.z).clone()
        fixed_before = model.verify_fixed_unchanged()

        start_epoch, start_batch = 0, 0
        order = self.seeds.numpy("split", 1).permutation(len(train))
        if resume_from is not None:
            meta, tensors = self.checkpoints.load(Path(resume_from))
            compat = CheckpointManager.check_compatible(
                meta, code_sha=source_hash().get("sha256", ""),
                config_sha=config_hash(self.cfg), data_digest=self.data_digest)
            if not compat["compatible"]:
                raise RuntimeError(
                    "체크포인트가 지금 코드/설정/자료와 호환되지 않는다: "
                    f"{compat['issues']}. 새 실행 폴더에서 시작하라.")
            model.gains.load_state_dict({"z": tensors["z"].tolist(),
                                         "mask": tensors["mask"].tolist(),
                                         "P_min": model.gains.P_min,
                                         "P_max": model.gains.P_max})
            model.encoder.load_state_dict(meta["encoder"])
            self.seeds.load_state_dict(meta["rng"])
            trainer.load_state_dict(meta["trainer"])
            self.test_access = meta.get("test_access", self.test_access)
            self.global_step = int(meta.get("global_step", 0))
            order = np.asarray(tensors["order"], dtype=np.int64)
            start_epoch = int(meta["epoch"])
            start_batch = int(meta["next_batch_index"])
            rec.log(f"체크포인트에서 재개: epoch {start_epoch}, batch {start_batch}, "
                    f"committed_rows={meta.get('committed_rows')}")
            rec.manifest["resumed_from"] = str(resume_from)
            rec.manifest["resume_compatibility"] = compat

        history: list[dict[str, Any]] = []
        dev_history: list[dict[str, Any]] = []
        status, reason = "completed", ""
        try:
            for epoch in range(start_epoch, n_epochs):
                if epoch > start_epoch:
                    order = self.seeds.numpy("split", 1 + epoch).permutation(len(train))
                    start_batch = 0
                for bi in range(start_batch, n_batches):
                    sel = order[bi * bs:(bi + 1) * bs]
                    if sel.size == 0:
                        continue
                    batch = [train[int(i)] for i in sel]
                    images, labels = self._batch_tensor(batch)
                    sample_id = int(bi * bs)
                    hook, finish = self._training_record_hook(
                        trainer, rec, sample_id=sample_id, episode_id=epoch,
                        base_id=batch[0].base_id)
                    r = trainer.step(images, labels, self.n_steps,
                                     drive_key=("input_noise", epoch * 10_000 + bi),
                                     epoch=epoch, batch_index=bi, recorder=rec,
                                     step_hook=hook)
                    r["event_counts"] = finish()
                    r["n_in_batch"] = len(batch)
                    history.append(r)
                    rec.metric(kind="train_batch", **r)
                    meta, tensors = self._checkpoint_payload(
                        trainer, epoch=epoch, next_batch_index=bi + 1,
                        next_sample_index=int(min((bi + 1) * bs, len(train))), order=order)
                    self.checkpoints.save(f"e{epoch:03d}_b{bi:05d}", meta=meta,
                                          tensors=tensors)
                    if stop_after_batches and len(history) >= stop_after_batches:
                        # 중단을 재현하기 위한 명시적 정지. 완료된 배치까지만 반영된다.
                        raise KeyboardInterrupt(
                            f"stop_after_batches={stop_after_batches} 로 중단했다")
                dev = self.evaluate("dev", note=f"epoch {epoch}")
                dev_history.append(dev)
                rec.log(f"epoch {epoch}: dev CE={dev.get('cross_entropy')} "
                        f"acc={dev.get('accuracy')}")
        except KeyboardInterrupt:
            status, reason = "interrupted", "사용자가 Ctrl+C 로 중단했다"
            rec.error(reason)
            rec.log("중단: 마지막 완료 배치의 체크포인트가 남아 있다. "
                    "부분 배치의 갱신은 저장하지 않았다.")
        except EngineDivergence as exc:
            status, reason = "failed", str(exc)
            rec.error("수치 발산으로 중단", exc,
                      last_checkpoint=str(self.checkpoints.latest()))
        except RecordingBudgetExceeded as exc:
            status, reason = "interrupted", str(exc)
            rec.error("기록 예산 초과로 안전하게 중단", exc,
                      last_checkpoint=str(self.checkpoints.latest()))
        after = model.gains.P_from_z(model.gains.z).clone()
        dP = (after - before)
        fixed_after = model.verify_fixed_unchanged()
        test = {"status": STATUS_NOT_RUN,
                "reason": "학습이 정상 종료되지 않아 test 를 평가하지 않았다."}
        if status == "completed":
            test = self.evaluate("test", note="최종 1회 평가")
        result = {
            "mode": trainer.mode, "status": status, "reason": reason,
            "epochs_planned": n_epochs, "n_batches_per_epoch": n_batches,
            "n_train": len(train), "history": history, "dev_history": dev_history,
            "test": test,
            "forward_calls": trainer.forward_calls,
            "P_change": {"mean_abs": float(dP.abs().mean()),
                         "max_abs": float(dP.abs().max()),
                         "changed_fraction": float((dP.abs() > 1e-9).to(self.dtype).mean())},
            "P_stats_after": model.gains.stats(),
            "fixed_parameters_unchanged": fixed_after["unchanged"],
            "fixed_changed_keys": fixed_after["changed_keys"],
            "fixed_before_check": fixed_before,
            "zero_difference_rate_mean": float(np.mean(
                [h.get("zero_difference_rate", 0.0) for h in history])) if history else None,
            "test_access": self.test_access,
            "learning_outcome": self._learning_outcome(history, dev_history),
            "note_ko": ("'실행 완료' 와 '학습 개선' 은 다른 상태다. 아래 "
                        "learning_outcome 을 따로 읽어라."),
        }
        rec.manifest["experiment_status"] = status
        rec.manifest["learning"] = {**rec.manifest.get("learning", {}),
                                    "outcome": result["learning_outcome"],
                                    "trainable_changed": result["P_change"],
                                    "fixed_unchanged": fixed_after}
        rec.manifest["test_access"] = self.test_access
        write_json(self.run_dir / "train.json", result)
        rec.write_summary_csv([{k: v for k, v in h.items()
                                if not isinstance(v, (dict, list))} for h in history])
        rec.close(status, reason)
        return result

    @staticmethod
    def _learning_outcome(history: list[dict[str, Any]],
                          dev: list[dict[str, Any]]) -> dict[str, Any]:
        """순방향 실행 성공과 학습 개선을 **분리해서** 판정한다 (F30)."""
        if not history:
            return {"verdict": STATUS_NOT_RUN, "reason": "학습 반복이 없었다"}
        if not any(h.get("updated") for h in history):
            return {"verdict": "no_update",
                    "reason": "이 조건은 P 를 갱신하지 않는다 (fixed_gain)"}
        if len(dev) < 2:
            return {"verdict": "insufficient_evidence",
                    "reason": "dev 평가가 2회 미만이라 개선 여부를 말할 수 없다",
                    "dev_ce": [d.get("cross_entropy") for d in dev]}
        first, last = dev[0].get("cross_entropy"), dev[-1].get("cross_entropy")
        if first is None or last is None:
            return {"verdict": "insufficient_evidence", "reason": "dev CE 가 없다"}
        improved = last < first
        return {"verdict": "improved" if improved else "not_improved",
                "dev_ce_first": float(first), "dev_ce_last": float(last),
                "delta": float(last - first),
                "note_ko": ("단일 시드·소수 에폭의 결과다. 개선이 없으면 없는 대로 "
                            "보고한다.")}

    def run_resume(self) -> dict[str, Any]:
        """마지막 **완료된 미니배치** 체크포인트에서 이어서 학습한다."""
        if self.run_dir is None:
            raise ValueError("--run-dir 가 필요하다")
        self.checkpoints = CheckpointManager(self.run_dir)
        latest = self.checkpoints.latest()
        if latest is None:
            raise RuntimeError(f"체크포인트가 없다: {self.run_dir / 'checkpoints'}")
        return self.run_train(resume_from=latest)

    # ------------------------------------------------------------------
    def run_benchmark(self) -> dict[str, Any]:
        """CPU/GPU 시간 비교. **측정한 것만** 적는다.

        warm-up, 전처리, 배선 생성, kernel 계산, 기록 I/O, 전체 wall time 을
        나누어 잰다. GPU 는 CUDA Event 와 필요한 동기화로 측정한다.
        """
        t = require_torch()
        self.setup("benchmark")
        results: dict[str, Any] = {"device_requested": self.device_choice,
                                   "environment": self.env.to_dict(), "runs": []}
        bench = self.cfg["benchmark"]
        devices = ["cpu"]
        if self.env.cuda_available:
            devices.insert(0, "cuda")
        else:
            results["cuda_note_ko"] = ("CUDA 를 쓸 수 없어 GPU 측정을 하지 않았다. "
                                       "GPU 항목은 not_run 이다.")
        for dev_name in devices:
            dev = t.device(dev_name)
            seeds = SeedStreams(int(self.cfg["seed"]))
            t0 = time.perf_counter()
            model = CorticalModel(self.cfg, dev, self.dtype, seeds)
            build_s = time.perf_counter() - t0
            gen = StimulusGenerator(self.cfg, seeds)
            stims = gen.classification_set()[:int(self.cfg["training"]["batch_size"])]
            frames = np.stack([s.frames[0] for s in stims]).transpose(0, 3, 1, 2)
            images = t.tensor(frames, dtype=self.dtype, device=dev)
            t1 = time.perf_counter()
            values, _ = model.sampler.sample(model.encoder.encode(images))
            model.encoder.fit_normalization(values, "train")
            normalized, _ = model.encode_batch(images)
            pre_s = time.perf_counter() - t1
            drive = model.make_drive(normalized, self.n_steps)
            P = model.gains.replicas(None)
            warm = int(bench["warmup_steps"])
            meas = int(bench["measure_steps"])
            model.prepare(1, int(images.shape[0]))
            for s in range(warm):
                model.engine.step(model.state, model.ring, P, s, drive, s)
            if dev_name == "cuda":
                t.cuda.synchronize()
                ev0, ev1 = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
                ev0.record()
            t2 = time.perf_counter()
            for s in range(warm, warm + meas):
                model.engine.step(model.state, model.ring, P, s, drive,
                                  min(s, self.n_steps - 1))
            if dev_name == "cuda":
                ev1.record()
                t.cuda.synchronize()
                kernel_ms = float(ev0.elapsed_time(ev1))
            else:
                kernel_ms = (time.perf_counter() - t2) * 1000.0
            wall_ms = (time.perf_counter() - t2) * 1000.0
            core_device = str(model.synapses.w0_nS.device)
            results["runs"].append({
                "device": dev_name, "core_tensor_device": core_device,
                "n_neurons": model.neurons.n, "n_synapses": model.synapses.n_edges,
                "build_seconds": round(build_s, 3),
                "preprocess_seconds": round(pre_s, 3),
                "warmup_steps": warm, "measured_steps": meas,
                "kernel_ms_total": round(kernel_ms, 3),
                "kernel_ms_per_step": round(kernel_ms / max(1, meas), 4),
                "wall_ms_total": round(wall_ms, 3),
                "measurement_method": ("CUDA Event + synchronize" if dev_name == "cuda"
                                       else "perf_counter"),
                "note_ko": "기록 I/O 는 이 측정에 포함하지 않았다 (별도 항목).",
            })
            del model
            if dev_name == "cuda":
                t.cuda.empty_cache()
        results["note_ko"] = ("측정하지 않은 속도를 '최적'·'빠르다' 라고 쓰지 않는다. "
                              "여기 숫자는 이 환경·이 preset 의 실측이다.")
        write_json(self.run_dir / "benchmark.json", results)
        return results


def run_comparison(cfg: dict[str, Any], output_root: Path, device_choice: str,
                   conditions: Sequence[str] | None = None, *, epochs: int | None = None,
                   max_samples: int = 0, progress: Callable[[str], None] | None = None
                   ) -> dict[str, Any]:
    """고정·학습·대조 조건을 **짝지어** 비교한다 (명세 12절).

    모든 조건의 초기 배선·P·입력·분할·해독기·시간을 같게 둔다. 계산 예산과
    순방향 호출 수를 함께 보고한다. 이 대조군으로 생물학적 인과가 입증된다고
    주장하지 않는다.
    """
    say = progress or (lambda m: None)
    conds = list(conditions or TRAIN_MODES)
    root = new_run_dir(Path(output_root), "compare")
    write_json(root / "resolved_config.json", cfg)
    summary: dict[str, Any] = {
        "root": str(root), "conditions": conds, "config_sha256": config_hash(cfg),
        "paired_note_ko": ("모든 조건이 같은 시드로 같은 배선·초기 P·분할·해독기·시간을 "
                           "쓴다. 조건 사이에 다른 것은 학습 규칙뿐이다."),
        "results": {},
    }
    matched_norms: list[float] = []
    ordered = [c for c in conds if c != "random_direction_matched"] + \
              [c for c in conds if c == "random_direction_matched"]
    for cond in ordered:
        say(f"조건 {cond} 실행")
        runner = ExperimentRunner(cfg, run_dir=root / safe_name(f"cond_{cond}"),
                                  device_choice=device_choice)
        runner.setup(cond)
        try:
            res = runner.run_train(
                mode=cond, epochs=epochs, max_samples=max_samples,
                matched_step_norms=(matched_norms
                                    if cond == "random_direction_matched" else None))
            if cond == "all_neuron_gain_spsa":
                matched_norms = [float(h.get("delta_z_norm", 0.0))
                                 for h in res["history"]]
        except Exception as exc:
            res = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()}
        summary["results"][cond] = {
            k: v for k, v in res.items() if k not in ("history", "dev_history")}
        summary["results"][cond]["n_iterations"] = len(res.get("history", []))
    summary["compute_budget"] = {
        c: {"forward_calls": summary["results"][c].get("forward_calls"),
            "iterations": summary["results"][c].get("n_iterations")}
        for c in summary["results"]}
    summary["note_ko"] = ("성능 개선이 없거나 출력 조정만 있는 대조군과 같으면 그대로 "
                          "보고한다. 이전 버전의 수치와 직접 순위를 매기지 않는다.")
    write_json(root / "comparison.json", summary)
    return summary


# ======================================================================
# 15. 조회 / 보고서 / 그림
# ======================================================================
class NeuronInspector:
    """저장된 기록만 읽어 뉴런 하나를 설명한다.

    조회는 반드시 ``(run, sample, episode, replica, step)`` 으로 한정한다 (F16).
    **관측된 입력 기여**와 **인과적 원인**을 구분한다. 인과 주장을 하려면 연결
    소거 비교(:func:`ablation_compare`)를 별도로 실행해야 한다.
    """

    def __init__(self, run_dir: Path) -> None:
        self.dir = Path(run_dir)
        self.manifest = read_json(self.dir / "manifest.json") \
            if (self.dir / "manifest.json").is_file() else {}
        if not (self.dir / "manifest.json").is_file():
            raise FileNotFoundError(
                f"실행 폴더가 아니다 (manifest.json 이 없다): {self.dir}\n"
                f"  메뉴 4(시뮬레이션) 또는 5(학습)가 만든 폴더를 넣어라. "
                f"그 폴더 이름은 simulate_... 또는 train_... 으로 시작한다.")
        backend = str(self.manifest.get("recording", {}).get("backend", "auto"))
        self.store = TableStore(self.dir, backend if backend in ("hdf5", "npz") else "auto")

    def available_tables(self) -> dict[str, Any]:
        """이 폴더에 어떤 기록 표가 있는지. '로그에 없다 = 입력이 없었다' 가 아니다."""
        out: dict[str, Any] = {"backend": self.store.backend,
                               "recording_mode": self.manifest.get(
                                   "recording", {}).get("policy", {}).get("mode")}
        for table in ("events", "states", "gain_updates"):
            fields, rows = self.store.read_all(table)
            out[table] = {"rows": int(rows.shape[0]) if rows.size else 0,
                          "fields": len(fields)}
        out["note_ko"] = ("기록하지 않은 내용을 '입력이 없었다' 로 해석하면 안 된다. "
                          "recording.mode 와 선택 뉴런 목록을 함께 보라.")
        return out

    def events_for(self, neuron_id: int, *, sample_id: int | None = None,
                   episode_id: int | None = None, replica_id: int | None = None,
                   step_from: int | None = None, step_to: int | None = None
                   ) -> list[dict[str, Any]]:
        fields, rows = self.store.read_all("events")
        if rows.size == 0:
            return []
        col = {k: i for i, k in enumerate(fields)}
        m = rows[:, col["dst_id"]] == float(neuron_id)
        if sample_id is not None:
            m &= rows[:, col["sample_id"]] == float(sample_id)
        if episode_id is not None:
            m &= rows[:, col["episode_id"]] == float(episode_id)
        if replica_id is not None:
            m &= rows[:, col["replica_id"]] == float(replica_id)
        if step_from is not None:
            m &= rows[:, col["sample_step"]] >= float(step_from)
        if step_to is not None:
            m &= rows[:, col["sample_step"]] < float(step_to)
        sel = rows[m]
        return [{k: (int(v) if k not in ("amount", "w0_snapshot", "P_emit_snapshot")
                     else float(v)) for k, v in zip(fields, r)} for r in sel]

    def explain(self, neuron_id: int, **scope: Any) -> dict[str, Any]:
        rows = self.events_for(neuron_id, **scope)
        by_src: dict[int, float] = {}
        for r in rows:
            by_src[int(r["src_id"])] = by_src.get(int(r["src_id"]), 0.0) + float(r["amount"])
        top = sorted(by_src.items(), key=lambda kv: -kv[1])[:10]
        return {
            "neuron_id": int(neuron_id), "scope": scope, "n_events": len(rows),
            "total_observed_amount_nS": float(sum(by_src.values())),
            "top_sources_by_observed_amount": [{"src_id": k, "amount_nS": v}
                                               for k, v in top],
            "events": rows[:200],
            "interpretation_note_ko": (
                "여기 값은 **관측된 입력 기여**다. 인과적 원인이 아니다. 인과를 보려면 "
                "같은 초기 상태·외생 사건에서 연결을 소거한 독립 복사본과 비교해야 한다."),
        }


def ablation_compare(cfg: dict[str, Any], device: Any, dtype: Any, seeds_seed: int,
                     stim: Stimulus, n_steps: int, *, ablate_rule: str | None = None,
                     ablate_synapse_ids: Sequence[int] | None = None) -> dict[str, Any]:
    """연결 소거 비교. **독립 복사본 두 개**를 같은 초기 상태·외생 사건으로 돌린다 (F25)."""
    t = require_torch()

    def make() -> CorticalModel:
        return CorticalModel(cfg, device, dtype, SeedStreams(seeds_seed))

    base = make()
    img = t.tensor(np.stack([stim.frames[0]]).transpose(0, 3, 1, 2), dtype=dtype,
                   device=device)
    values, _ = base.sampler.sample(base.encoder.encode(img))
    base.encoder.fit_normalization(values, "train")
    normalized, _ = base.encode_batch(img)
    drive = base.make_drive(normalized, n_steps, rng_key=("diagnostics", 0))
    out_base = base.run(drive, base.gains.replicas(None), n_steps)

    abl = make()
    abl.encoder.load_state_dict(base.encoder.state_dict())
    if ablate_rule is not None:
        names = [r["name"] for r in cfg["wiring"]["rules"]]
        if ablate_rule not in names:
            raise KeyError(f"배선 규칙을 찾을 수 없다: {ablate_rule!r}")
        mask = abl.synapses.rule_index == names.index(ablate_rule)
    elif ablate_synapse_ids is not None:
        mask = t.zeros(abl.synapses.n_edges, dtype=t.bool, device=device)
        mask[t.tensor(list(ablate_synapse_ids), dtype=t.long, device=device)] = True
    else:
        raise ValueError("ablate_rule 또는 ablate_synapse_ids 중 하나가 필요하다")
    n_removed = int(mask.sum())
    abl.synapses.w0_nS = abl.synapses.w0_nS.clone()
    abl.synapses.w0_nS[mask] = 0.0
    drive2 = abl.make_drive(normalized, n_steps, rng_key=("diagnostics", 0))
    out_abl = abl.run(drive2, abl.gains.replicas(None), n_steps)

    sb = out_base["spike_count"][0, 0]
    sa = out_abl["spike_count"][0, 0]
    changed = int((sb != sa).sum())
    return {
        "ablated_rule": ablate_rule, "n_synapses_removed": n_removed,
        "total_spikes_base": int(sb.sum()), "total_spikes_ablated": int(sa.sum()),
        "neurons_with_changed_spike_count": changed,
        "exogenous_identical": True,
        "base_model_untouched": base.verify_fixed_unchanged(),
        "note_ko": ("두 모델은 같은 설정·시드로 따로 만든 독립 복사본이다. 원본 모델의 "
                    "가중치·상태·RNG 는 이 비교로 바뀌지 않는다."),
    }


class ReportBuilder:
    """**저장된 JSON/CSV 에서만** 한국어 보고서를 만든다 (F27).

    보고서에 새 수치를 계산해 넣지 않는다. 실행하지 않은 항목은 그대로
    ``not_run`` 으로 적는다.
    """

    def __init__(self, run_dir: Path) -> None:
        self.dir = Path(run_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"실행 폴더가 없다: {self.dir}")
        self.manifest = read_json(self.dir / "manifest.json") \
            if (self.dir / "manifest.json").is_file() else {}

    def _load(self, name: str) -> Any | None:
        p = self.dir / name
        return read_json(p) if p.is_file() else None

    def build(self) -> Path:
        m = self.manifest
        L: list[str] = []
        A = L.append
        A(f"# 실행 보고서 — {m.get('run_id', self.dir.name)}")
        A("")
        A("> 이 문서는 이 폴더에 저장된 JSON/CSV 에서만 만들었다. 새로 계산한 수치는 "
          "없다. 실행하지 않은 항목은 `not_run` 으로 적는다.")
        A("")
        A("## 1. 실행 정보")
        A("")
        A(f"- 상태: **{m.get('status', '알 수 없음')}** "
          f"{m.get('status_reason', '')}")
        A(f"- 시작/종료(UTC): {m.get('started_utc')} / {m.get('finished_utc')}")
        A(f"- 장치: `{m.get('device')}` / dtype `{m.get('dtype')}`")
        A(f"- 결정론 모드: {m.get('deterministic', {}).get('status')} "
          f"— {m.get('deterministic', {}).get('note_ko', '')}")
        A(f"- 코드 sha256: `{m.get('code', {}).get('sha256', '')[:16]}…`")
        A(f"- 설정 sha256: `{str(m.get('config_sha256', ''))[:16]}…`")
        A(f"- 명령: `{m.get('command', '')}`")
        env = m.get("environment", {})
        A(f"- PyTorch {env.get('torch_version')} (CUDA 빌드 {env.get('torch_cuda_build')}), "
          f"cuda_available={env.get('cuda_available')}")
        for d in env.get("devices", []):
            if "name" in d:
                A(f"    - GPU {d['index']}: {d['name']}, VRAM "
                  f"{d.get('free_vram_mb')} / {d.get('total_vram_mb')} MB")
        A("")
        A("## 2. 고정 조건")
        A("")
        A(f"- 모든 뉴런의 발화 판정 임계값: **{THRESHOLD}** (학습·적응하지 않음)")
        A(f"- 학습 대상: {m.get('learning', {}).get('trainable')}")
        A(f"- 학습 조건: `{m.get('learning', {}).get('mode')}`")
        A("- 해독기: 고정 (클래스 그룹·스케일·편향 모두 학습하지 않음)")
        fx = m.get("learning", {}).get("fixed_unchanged")
        if fx:
            A(f"- 고정 파라미터 불변 검사: {fx.get('unchanged')} "
              f"(검사한 항목 {fx.get('n_checked')}개, 바뀐 항목 {fx.get('changed_keys')})")
        A("")
        model = m.get("model", {})
        if model:
            A("## 3. 모델 규모")
            A("")
            A(f"- 뉴런 {model.get('n_neurons')} / 시냅스 {model.get('n_synapses')}")
            A(f"- 영역별 뉴런 수: {model.get('per_area')}")
            cal = model.get("w0_calibration", {})
            if cal:
                A(f"- w0 보정: 기준 발화율 {cal.get('reference_presyn_rate_hz')} Hz, "
                  f"목표 배율 {cal.get('target_ratio')} "
                  f"({cal.get('note_ko','')})")
            A("")
        rec = m.get("recording", {})
        if rec:
            A("## 4. 기록")
            A("")
            A(f"- 모드: `{rec.get('policy', {}).get('mode')}` / backend "
              f"`{rec.get('backend')}`")
            if rec.get("fallback_note"):
                A(f"- {rec['fallback_note']}")
            c = rec.get("counters", {})
            A(f"- 사건: 발생 {c.get('events_emitted')} / 도착 {c.get('events_arrived')} / "
              f"기록 {c.get('events_recorded')} / 제외 {c.get('events_filtered')}")
            A(f"- 상태 행: 기록 {c.get('state_rows_recorded')} / "
              f"예산으로 생략 {c.get('state_rows_skipped_budget')}")
            A(f"- 저장된 행 수(committed): {rec.get('committed_rows')}")
            if rec.get("stop_reasons"):
                A(f"- 기록 중단 사유: {rec['stop_reasons']}")
            A(f"- 보존 정책: {rec.get('policy', {}).get('retention_note_ko','')}")
            A("")
        for name, title in (("train.json", "5. 학습"), ("diagnose.json", "5. 전달 진단"),
                            ("simulate.json", "5. 시뮬레이션"),
                            ("benchmark.json", "5. 성능 측정"),
                            ("validation.json", "5. 검증")):
            data = self._load(name)
            if data is None:
                continue
            A(f"## {title} (`{name}`)")
            A("")
            if name == "train.json":
                A(f"- 상태: **{data.get('status')}** {data.get('reason','')}")
                A(f"- 조건: `{data.get('mode')}` / 순방향 호출 {data.get('forward_calls')}")
                A(f"- P 변화: 평균 |ΔP| {data.get('P_change', {}).get('mean_abs')}, "
                  f"최대 {data.get('P_change', {}).get('max_abs')}")
                A(f"- 고정 파라미터 불변: {data.get('fixed_parameters_unchanged')}")
                lo = data.get("learning_outcome", {})
                A(f"- **학습 결과 판정**: `{lo.get('verdict')}` — {lo.get('reason', '')} "
                  f"{lo.get('note_ko','')}")
                A(f"- ± 오차 차이가 0 이었던 비율(평균): "
                  f"{data.get('zero_difference_rate_mean')}")
                test = data.get("test", {})
                A(f"- test 평가: 상태 `{test.get('status')}`, CE "
                  f"{test.get('cross_entropy')}, 정확도 {test.get('accuracy')}, "
                  f"균형 정확도 {test.get('balanced_accuracy')}, 다수 기준선 "
                  f"{test.get('majority_baseline')}, 영특징 비율 "
                  f"{test.get('zero_feature_fraction')}")
                A(f"- test 접근 횟수: "
                  f"{data.get('test_access', {}).get('n_test_evaluations')}")
                A("- '실행 완료' 와 '학습 개선' 은 다른 상태다. 위 판정을 따로 읽어라.")
            elif name == "validation.json":
                A(f"- 통과 {data.get('n_passed')} / 실패 {data.get('n_failed')} / "
                  f"건너뜀 {data.get('n_skipped')} / 해당없음 "
                  f"{data.get('n_not_applicable')} / 미실행 {data.get('n_not_run')} / "
                  f"측정만 {data.get('n_measured')}")
                A("")
                A("| 검사 | 상태 | 기준 | 관측 |")
                A("|---|---|---|---|")
                for c in data.get("checks", []):
                    A(f"| {c.get('name')} | **{c.get('status')}** | "
                      f"{str(c.get('criterion',''))[:80]} | "
                      f"{str(c.get('observed',''))[:80]} |")
            else:
                A("```json")
                A(dumps({k: v for k, v in data.items()
                         if not isinstance(v, list) or len(v) < 20})[:4000])
                A("```")
            A("")
        A("## 6. 해석 시 주의")
        A("")
        A("- 이 수치는 모형 파라미터로 만든 모형의 출력이다. 생물학적 측정값이 아니다.")
        A("- 고정 Gabor 초기 배선의 방향 선택성은 학습된 것이 아니다.")
        A("- 고정 IT 해독기의 클래스 그룹은 인공적인 해독 규칙이다.")
        A("- SPSA 는 전역 과제 오차를 쓰는 공학적 기준이며 뇌의 학습 법칙이 아니다.")
        A("- 정지 영상만 실행했다면 운동 관련 항목은 `not_applicable` 이다.")
        A("- 이전 버전 모형과 성능 순위를 직접 매기지 않는다.")
        out = self.dir / "report_ko.md"
        out.write_text("\n".join(L) + "\n", encoding="utf-8")
        return out


def make_figures(run_dir: Path) -> list[str]:
    """**저장된 기록만** 읽어 그림을 만든다. 계산을 다시 돌리지 않는다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            f"matplotlib 를 쓸 수 없다: {type(exc).__name__}: {exc}\n"
            f"  python -m pip install matplotlib") from exc
    run_dir = Path(run_dir)
    fig_dir = ensure_writable_dir(run_dir / "figures")
    made: list[str] = []
    metrics = run_dir / "metrics.jsonl"
    if metrics.is_file():
        rows = [json.loads(l) for l in metrics.read_text(encoding="utf-8").splitlines() if l]
        tb = [r for r in rows if r.get("kind") == "train_batch"]
        if tb:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            it = [r["iteration"] for r in tb]
            lp = [r.get("loss_plus_mean", r.get("loss_base")) for r in tb]
            lm = [r.get("loss_minus_mean", r.get("loss_base")) for r in tb]
            ax[0].plot(it, lp, label="L+")
            ax[0].plot(it, lm, label="L-")
            ax[0].set_xlabel("iteration"); ax[0].set_ylabel("cross entropy")
            ax[0].legend(); ax[0].set_title("SPSA losses (saved log)")
            ax[1].plot(it, [r.get("delta_P_max", 0.0) for r in tb])
            ax[1].set_xlabel("iteration"); ax[1].set_ylabel("max |dP|")
            ax[1].set_title("gain change")
            fig.tight_layout()
            p = fig_dir / "training.png"
            fig.savefig(p, dpi=120); plt.close(fig)
            made.append(str(p))
        ev = [r for r in rows if r.get("kind") == "evaluation"]
        if ev:
            fig, ax = plt.subplots(figsize=(6, 4))
            for split in ("dev", "test"):
                xs = [i for i, r in enumerate(ev) if r.get("split") == split]
                ys = [r["cross_entropy"] for r in ev if r.get("split") == split]
                if xs:
                    ax.plot(xs, ys, marker="o", label=split)
            ax.set_xlabel("evaluation index"); ax.set_ylabel("cross entropy")
            ax.legend(); ax.set_title("evaluations (saved log)")
            fig.tight_layout()
            p = fig_dir / "evaluation.png"
            fig.savefig(p, dpi=120); plt.close(fig)
            made.append(str(p))
    return made


# ======================================================================
# 16. 사용자 실행 검증 (메뉴 2 / --mode validate 에서만 실행)
# ======================================================================
@dataclass
class CheckResult:
    name: str
    title: str
    status: str
    criterion: str
    observed: Any
    scope: str
    details: dict[str, Any] = field(default_factory=dict)
    device: str = ""
    dtype: str = ""
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ValidationSuite:
    """명세 11절의 28개 검사.

    통과 기준을 관측값에 맞춰 사후에 바꾸지 않는다. 측정만 하는 항목은
    ``measured`` 로 두고 강제로 ``passed`` 로 적지 않는다. 실행하지 않은 항목은
    ``not_run`` 이다.
    """

    def __init__(self, cfg: dict[str, Any], device: Any, dtype: Any,
                 progress: Callable[[str], None] | None = None) -> None:
        require_torch()
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.say = progress or (lambda m: None)
        self.tiny_cfg = build_config("tiny")
        self.tiny_cfg["deterministic"] = False
        self._model: CorticalModel | None = None

    # -- 공용 작은 모델 -------------------------------------------------
    def model(self) -> CorticalModel:
        if self._model is None:
            self._model = CorticalModel(self.tiny_cfg, self.device, self.dtype,
                                        SeedStreams(int(self.tiny_cfg["seed"])))
        return self._model

    def _fresh_model(self, cfg: dict[str, Any] | None = None,
                     seed: int | None = None) -> CorticalModel:
        c = cfg or self.tiny_cfg
        return CorticalModel(c, self.device, self.dtype,
                             SeedStreams(int(seed if seed is not None else c["seed"])))

    def _tiny_stim(self, kind: str = "grating") -> Stimulus:
        gen = StimulusGenerator(self.tiny_cfg, SeedStreams(int(self.tiny_cfg["seed"])))
        if kind == "grating":
            return next(s for s in gen.diagnostic_set() if s.stimulus_id == "diag_ori_00")
        if kind == "dark":
            return next(s for s in gen.diagnostic_set() if s.stimulus_id == "diag_dark")
        if kind == "frames":
            return next(s for s in gen.diagnostic_set() if s.stimulus_id == "diag_frames")
        return gen.classification_set()[0]

    def _prepared(self, model: CorticalModel, stim: Stimulus, n_steps: int
                  ) -> tuple[ExternalDrive, Any]:
        t = require_torch()
        img = t.tensor(np.stack([stim.frames[0]]).transpose(0, 3, 1, 2),
                       dtype=self.dtype, device=self.device)
        values, _ = model.sampler.sample(model.encoder.encode(img))
        if not model.encoder.norm_info.get("fitted"):
            model.encoder.fit_normalization(values, "train")
        normalized, _ = model.encode_batch(img)
        drive = model.make_drive(normalized, n_steps, rng_key=("diagnostics", 1))
        return drive, normalized

    # ==================================================================
    def run_all(self, only: Sequence[str] | None = None) -> dict[str, Any]:
        checks = [
            self.check_01_view_ids, self.check_02_threshold_is_15,
            self.check_03_only_p_changes, self.check_04_single_application,
            self.check_05_gpu_sum_vs_cpu_reference, self.check_06_delay_ring,
            self.check_07_zero_gain_blocks_output, self.check_08_membrane,
            self.check_09_recording_modes, self.check_10_query_isolation,
            self.check_11_retina_geometry, self.check_12_lgn_channel_preserved,
            self.check_13_shared_orientation_map, self.check_14_transmission,
            self.check_15_feedback_apical, self.check_16_pm_pairing,
            self.check_17_zero_error_no_update, self.check_18_spsa_on_quadratic,
            self.check_19_spiking_perturbation, self.check_20_readout,
            self.check_21_splits, self.check_22_resume,
            self.check_23_poisson_frames, self.check_24_diagnostics_pure,
            self.check_25_gpu_cpu_tolerance, self.check_26_import_and_paths,
            self.check_27_limits_and_backpressure, self.check_28_report_matches_logs,
        ]
        results: list[CheckResult] = []
        for fn in checks:
            name = fn.__name__
            if only and name not in only:
                results.append(CheckResult(name, name, STATUS_NOT_RUN,
                                           "사용자가 고르지 않았다", None, "-"))
                continue
            self.say(f"검사 실행: {name}")
            t0 = time.time()
            try:
                res = fn()
            except NotImplementedError as exc:
                res = CheckResult(name, name, STATUS_NOT_SUPPORTED, "지원 여부",
                                  str(exc), "-")
            except Exception as exc:
                res = CheckResult(name, name, STATUS_FAILED, "예외 없이 완료",
                                  f"{type(exc).__name__}: {exc}", "-",
                                  {"traceback": traceback.format_exc()})
            res.device = str(self.device)
            res.dtype = str(self.dtype)
            res.elapsed_sec = round(time.time() - t0, 3)
            results.append(res)
        counts = {s: sum(1 for r in results if r.status == s)
                  for s in (STATUS_PASSED, STATUS_FAILED, STATUS_SKIPPED,
                            STATUS_NOT_APPLICABLE, STATUS_NOT_RUN, STATUS_MEASURED,
                            STATUS_NOT_SUPPORTED)}
        return {
            "n_checks": len(results),
            "n_passed": counts[STATUS_PASSED], "n_failed": counts[STATUS_FAILED],
            "n_skipped": counts[STATUS_SKIPPED],
            "n_not_applicable": counts[STATUS_NOT_APPLICABLE],
            "n_not_run": counts[STATUS_NOT_RUN], "n_measured": counts[STATUS_MEASURED],
            "n_not_supported": counts[STATUS_NOT_SUPPORTED],
            "device": str(self.device), "dtype": str(self.dtype),
            "checks": [r.to_dict() for r in results],
            "note_ko": ("measured 는 기준 없이 측정만 한 항목이며 passed 가 아니다. "
                        "not_applicable/not_run 을 통과로 세지 않는다."),
        }

    # ==================================================================
    def check_01_view_ids(self) -> CheckResult:
        m = self.model()
        nid = int(m.decoder.neuron_ids[0])
        view = m.neuron_view(nid, run_id="validate", sample_id=0, episode_id=0)
        mat = view.as_matrix()
        ok_shape = mat.shape == (3, 3) and mat.dtype == object
        out = view.outgoing
        syn = m.synapses
        ids = out["synapse_ids"]
        ok_ids = all(0 <= i < syn.n_edges for i in ids)
        ok_src = all(int(syn.src[i]) == nid for i in ids)
        ok_dst = out["dst_ids"] == [int(syn.dst[i]) for i in ids]
        meta = view.metadata
        ok_meta = meta["neuron_id"] == nid and meta["area"] is not None
        ok_thr = view.threshold == THRESHOLD
        ok_cell10 = isinstance(mat[1, 0], dict) and mat[1, 0]["handle"] == \
            "outgoing_connection_list"
        passed = all([ok_shape, ok_ids, ok_src, ok_dst, ok_meta, ok_thr, ok_cell10])
        return CheckResult(
            "check_01_view_ids", "3x3 뷰와 실제 텐서/연결/로그의 ID 일치",
            STATUS_PASSED if passed else STATUS_FAILED,
            "as_matrix 가 (3,3) object 이고 [1][0] 이 출력 연결 목록이며 "
            "synapse_id/src/dst 가 edge 배열과 일치",
            {"shape_ok": ok_shape, "ids_valid": ok_ids, "src_match": ok_src,
             "dst_match": ok_dst, "meta_ok": ok_meta, "threshold_ok": ok_thr,
             "cell_1_0_is_outgoing": ok_cell10},
            "tiny preset 실제 모델, 조회 경로",
            {"neuron_id": nid, "n_outgoing": out["n"],
             "layout": NeuronView3x3.layout_description()})

    def check_02_threshold_is_15(self) -> CheckResult:
        require_torch()          # torch 없으면 여기서 멈춘다
        m = self._fresh_model()
        after_build = bool((m.neurons.threshold_u == THRESHOLD).all())
        gains = m.gains
        gains.z = gains.z + 0.5                        # 학습을 흉내 낸 변화
        after_train = bool((m.neurons.threshold_u == THRESHOLD).all())
        st = {"z": gains.z.detach().to("cpu").numpy().tolist(),
              "mask": gains.mask.detach().to("cpu").numpy().tolist(),
              "P_min": gains.P_min, "P_max": gains.P_max}
        m2 = self._fresh_model()
        m2.gains.load_state_dict(st)                   # 재개를 흉내 낸 복원
        after_resume = bool((m2.neurons.threshold_u == THRESHOLD).all())
        exact = float(m.neurons.threshold_u.min()), float(m.neurons.threshold_u.max())
        passed = after_build and after_train and after_resume and exact == (THRESHOLD, THRESHOLD)
        return CheckResult(
            "check_02_threshold_is_15", "모든 뉴런 임계값이 생성·학습·재개 뒤에도 15.0",
            STATUS_PASSED if passed else STATUS_FAILED,
            "threshold_u 의 min == max == 15.0 이고 세 시점 모두 동일",
            {"after_build": after_build, "after_train_like": after_train,
             "after_resume": after_resume, "min_max": exact},
            "tiny preset 실제 모델")

    def check_03_only_p_changes(self) -> CheckResult:
        t = require_torch()
        m = self._fresh_model()
        before = dict(m.fixed_hashes)
        w0_before = tensor_hash(m.synapses.w0_nS)
        z_before = m.gains.z.clone()
        seeds = SeedStreams(int(self.tiny_cfg["seed"]))
        trainer = GainSPSATrainer(self.tiny_cfg, m, seeds)
        stim = self._tiny_stim("shape")
        img = t.tensor(np.stack([stim.frames[0]]).transpose(0, 3, 1, 2),
                       dtype=self.dtype, device=self.device)
        values, _ = m.sampler.sample(m.encoder.encode(img))
        m.encoder.fit_normalization(values, "train")
        labels = t.tensor([0], dtype=t.long, device=self.device)
        n_steps = int(round(self.tiny_cfg["engine"]["sample_ms"]
                            / self.tiny_cfg["engine"]["dt_ms"]))
        trainer.step(img, labels, n_steps, drive_key=("input_noise", 0), epoch=0,
                     batch_index=0)
        after = m._compute_fixed_hashes()
        changed = sorted(k for k in before if before[k] != after.get(k))
        z_changed = bool((m.gains.z != z_before).any())
        w0_same = tensor_hash(m.synapses.w0_nS) == w0_before
        passed = (not changed) and w0_same
        return CheckResult(
            "check_03_only_p_changes", "학습 전후 고정 파라미터 불변, P/z 만 변경",
            STATUS_PASSED if passed else STATUS_FAILED,
            "고정 텐서 해시가 하나도 바뀌지 않고 w0 해시가 같다",
            {"changed_fixed_keys": changed, "w0_unchanged": w0_same,
             "z_changed": z_changed},
            "tiny preset, SPSA 1 스텝 실제 실행",
            {"n_fixed_checked": len(before),
             "trainable_allowlist": self.tiny_cfg["training"]["trainable_allowlist"]})

    def check_04_single_application(self) -> CheckResult:
        """한 사건이 두 번 반영되지 않고, 잔류 전도도와 새 입력이 구분되는지."""
        t = require_torch()
        m = self._fresh_model()
        n = m.neurons.n
        m.prepare(1, 1)
        P = m.gains.replicas(None)
        src = int(m.synapses.src[0])
        # 한 번만 발신시킨 뒤 모든 스텝의 도착량을 모은다. 같은 발신 뉴런이 서로 다른
        # 지연의 출력 연결을 가지면 도착 스텝은 여러 개가 된다. 불변식은 "각 연결이
        # 정확히 한 번만 반영된다" 이므로 총합과 지연 종류 수로 검사한다.
        m.ring.reset()
        q = t.zeros((1, 1, n), dtype=self.dtype, device=self.device)
        q[0, 0, src] = 1.0
        m.ring.write(0, q)
        out_edges = m.synapses.outgoing_ids(src)
        expected_total = float(m.synapses.w0_nS[out_edges].sum())
        distinct_delays = int(t.unique(m.synapses.delay_steps[out_edges]).numel())
        arrivals: list[float] = []
        for step in range(0, m.ring.max_delay + 2):
            delta = m.engine._gather_arrivals(m.ring, step, 1, 1)
            arrivals.append(float(delta.sum()))
        total = float(sum(arrivals))
        n_steps_with_input = sum(1 for a in arrivals if a > 0)
        tol = 1e-3 if self.dtype == t.float32 else 1e-9
        applied_once = (abs(total - expected_total) <= tol * max(1.0, expected_total)
                        and n_steps_with_input == distinct_delays)
        # 잔류 전도도: 입력이 없어도 지수 감쇠로 남는다 (새 입력과 구분된다)
        st = DynamicState(1, 1, n, m.neurons, self.device, self.dtype)
        st.g[0, 0, 0, 0, RECEPTOR_INDEX["AMPA"]] = 5.0
        g0 = float(st.g.sum())
        info = m.engine.step(st, m.ring, P, 50)
        g1 = float(st.g.sum())
        new_input = float(info["delta_g"].sum())
        residual_decayed = g1 < g0 and g1 > 0.0
        passed = applied_once and residual_decayed and new_input == 0.0
        return CheckResult(
            "check_04_single_application", "사건 중복 적용 방지, 잔류 전도도와 새 입력 구분",
            STATUS_PASSED if passed else STATUS_FAILED,
            "한 번의 발신이 만든 도착량 총합이 그 뉴런의 w0 합과 같고(각 연결이 정확히 "
            "한 번만 반영), 도착 스텝 수가 그 뉴런의 지연 종류 수와 같으며, 새 입력 0 인 "
            "스텝에서도 잔류 전도도는 감쇠만 한다",
            {"arrival_amount_per_step": arrivals, "total_arrived": total,
             "expected_total_from_w0": expected_total,
             "n_steps_with_input": n_steps_with_input,
             "n_distinct_delays_of_source": distinct_delays,
             "applied_exactly_once": applied_once,
             "g_before": g0, "g_after": g1, "new_input_sum": new_input},
            "tiny preset 엔진 직접 호출")

    def check_05_gpu_sum_vs_cpu_reference(self) -> CheckResult:
        """다중 입력 합산을 작은 CPU float64 참조와 비교한다 (중복 표적·chunk 경계 포함)."""
        t = require_torch()
        m = self._fresh_model()
        n = m.neurons.n
        syn = m.synapses
        gen = np.random.default_rng(12345)
        q_np = gen.random(n) * (gen.random(n) < 0.3)          # 동시에 여러 뉴런 발화
        q = t.tensor(q_np, dtype=self.dtype, device=self.device).view(1, 1, n)
        m.prepare(1, 1)
        m.ring.reset()
        m.ring.write(0, q)
        small_chunk = max(1, syn.n_edges // 7)                # chunk 경계를 강제로 만든다
        original = m.engine.edge_chunk
        results = {}
        for chunk in (original, small_chunk, 1 if syn.n_edges < 4000 else small_chunk // 2 + 1):
            m.engine.edge_chunk = max(1, int(chunk))
            got = m.engine._gather_arrivals(m.ring, min(d for d, _ in syn.delay_groups),
                                            1, 1)
            results[int(chunk)] = got.detach().to("cpu").to(t.float64).numpy()
        m.engine.edge_chunk = original
        # 결정론 경로(표적 정렬 + 누적합)도 같은 값을 주는지 함께 본다
        was = m.engine.deterministic
        m.engine.deterministic = True
        det = m.engine._gather_arrivals(m.ring, min(d for d, _ in syn.delay_groups), 1, 1)
        m.engine.deterministic = was
        det_np = det.detach().to("cpu").to(t.float64).numpy()
        # CPU float64 참조: 같은 지연의 edge 만 더한다
        d0 = min(d for d, _ in syn.delay_groups)
        idx = (syn.delay_steps == d0).detach().to("cpu").numpy()
        src = syn.src.detach().to("cpu").numpy()[idx]
        w0 = syn.w0_nS.detach().to("cpu").to(t.float64).numpy()[idx]
        flat_t = syn.flat_target.detach().to("cpu").numpy()[idx]
        ref = np.zeros(n * N_COMP * N_RECEPTOR, dtype=np.float64)
        np.add.at(ref, flat_t, q_np[src] * w0)
        ref = ref.reshape(n, N_COMP, N_RECEPTOR)
        errs = {k: float(np.max(np.abs(v[0, 0] - ref))) for k, v in results.items()}
        det_err = float(np.max(np.abs(det_np[0, 0] - ref)))
        dup_targets = int(len(flat_t) - len(np.unique(flat_t)))
        tol = 1e-4 if self.dtype == t.float32 else 1e-10
        passed = (all(e <= tol for e in errs.values()) and det_err <= tol
                  and dup_targets > 0)
        return CheckResult(
            "check_05_gpu_sum_vs_cpu_reference",
            "GPU 다중 입력 합산 vs CPU float64 참조 (중복 표적·chunk 경계)",
            STATUS_PASSED if passed else STATUS_FAILED,
            f"두 도착 경로(index_add / 결정론 세그먼트) 모두 최대 절대 오차 <= {tol} "
            f"이고 중복 표적이 실제로 존재",
            {"max_abs_error_by_chunk": errs,
             "max_abs_error_deterministic_path": det_err,
             "duplicate_target_pairs": dup_targets, "tolerance": tol},
            "tiny preset 실제 배선, 엔진 합산 경로",
            {"n_edges_same_delay": int(idx.sum()), "delay": d0})

    def check_06_delay_ring(self) -> CheckResult:
        t = require_torch()
        m = self._fresh_model()
        n = m.neurons.n
        m.prepare(1, 1)
        ring = m.ring
        ring.reset()
        first = m.engine._gather_arrivals(ring, 0, 1, 1)
        first_zero = float(first.sum()) == 0.0
        min_delay = int(m.synapses.delay_steps.min())
        ok_min = min_delay >= int(self.tiny_cfg["engine"]["min_delay_steps"]) >= 1
        # wrap-around: ring 길이를 넘겨도 같은 값이 읽혀야 한다
        q = t.zeros((1, 1, n), dtype=self.dtype, device=self.device)
        src = int(m.synapses.src[0])
        q[0, 0, src] = 3.0
        big_step = ring.length * 3 + 2
        ring.write(big_step, q)
        back = ring.read(big_step + min_delay, min_delay)
        ok_wrap = bool(t.equal(back, q))
        # 발신 시점 P 보존: P 를 바꿔도 이미 저장된 q 는 그대로다
        m.gains.z = m.gains.z + 2.0
        again = ring.read(big_step + min_delay, min_delay)
        ok_emit_P = bool(t.equal(again, q))
        passed = first_zero and ok_min and ok_wrap and ok_emit_P
        return CheckResult(
            "check_06_delay_ring", "지연 ring 첫 스텝·최소 지연·wrap-around·발신 시점 P 보존",
            STATUS_PASSED if passed else STATUS_FAILED,
            "첫 스텝 도착 0, 최소 지연 >= 1, ring 길이를 넘겨도 같은 값, P 변경 후에도 "
            "과거 q 불변",
            {"first_step_zero": first_zero, "min_delay": min_delay,
             "wrap_ok": ok_wrap, "emit_time_P_preserved": ok_emit_P,
             "ring_length": ring.length},
            "tiny preset, DelayRing 직접 검사")

    def check_07_zero_gain_blocks_output(self) -> CheckResult:
        """읽기 전용 검사용 모델에서 P=0 이면 출력 전달이 0 인지.

        이것은 'P 가 학습 범위 끝값에 도달했다' 는 주장이 아니다.
        """
        t = require_torch()
        m = self._fresh_model()
        stim = self._tiny_stim("grating")
        n_steps = 20
        drive, _ = self._prepared(m, stim, n_steps)
        P_zero = t.zeros((1, m.neurons.n), dtype=self.dtype, device=self.device)
        out_zero = m.run(drive, P_zero, n_steps)
        transferred_zero = float(m.ring.buf.abs().sum())
        m2 = self._fresh_model()
        m2.encoder.load_state_dict(m.encoder.state_dict())
        drive2, _ = self._prepared(m2, stim, n_steps)
        out_base = m2.run(drive2, m2.gains.replicas(None), n_steps)
        transferred_base = float(m2.ring.buf.abs().sum())
        passed = transferred_zero == 0.0 and transferred_base >= 0.0
        return CheckResult(
            "check_07_zero_gain_blocks_output", "P=0 이면 출력 전달이 0",
            STATUS_PASSED if passed else STATUS_FAILED,
            "P=0 실행에서 지연 ring 에 쌓인 q 의 절댓값 합이 정확히 0",
            {"q_sum_with_P0": transferred_zero, "q_sum_with_P_init": transferred_base,
             "spikes_with_P0": int(out_zero["spike_count"].sum()),
             "spikes_with_P_init": int(out_base["spike_count"].sum())},
            "tiny preset, 읽기 전용 검사 모델",
            {"note_ko": "P=0 은 출력 전달만 막는다. 뉴런 자체의 발화 판정은 계속된다."})

    def check_08_membrane(self) -> CheckResult:
        """누설·불응기·구획 결합·부호·비음수 전도도·dt 감소 검사."""
        require_torch()          # torch 없으면 여기서 멈춘다
        m = self._fresh_model()
        n = m.neurons.n
        st = DynamicState(1, 1, n, m.neurons, self.device, self.dtype)
        P = m.gains.replicas(None)
        m.prepare(1, 1)
        # (a) 무입력 누설: EL 아닌 값에서 출발하면 EL 로 돌아간다
        i0 = int(m.decoder.neuron_ids[0])
        EL = float(m.neurons.EL_mV[i0, 0])
        st.V[0, 0, i0, 0] = EL + 10.0
        dt = float(self.tiny_cfg["engine"]["dt_ms"])
        tau = float(m.neurons.C_pF[i0, 0]) / float(m.neurons.gL_nS[i0, 0])
        n_leak = int(math.ceil(math.log(10.0 / 0.05) / math.log(1 + dt / tau)))
        for s in range(n_leak):
            m.engine.step(st, m.ring, P, 1000 + s)
        v_leak = float(st.V[0, 0, i0, 0])
        leak_ok = abs(v_leak - EL) < 0.5
        # (b) 불응기: soma 클램프가 연립식의 경계조건으로 들어갔는지 (F22)
        st2 = DynamicState(1, 1, n, m.neurons, self.device, self.dtype)
        st2.refrac_until[0, 0, i0] = 3000
        st2.g[0, 0, i0, COMP_INDEX["basal"], RECEPTOR_INDEX["AMPA"]] = 30.0
        info = m.engine.step(st2, m.ring, P, 2000)
        v_soma = float(st2.V[0, 0, i0, 0])
        v_basal = float(st2.V[0, 0, i0, 1])
        reset_v = float(m.neurons.V_reset_mV[i0])
        clamp_ok = abs(v_soma - reset_v) < 1e-6
        # 경계조건이 맞으면 basal 은 (EL 이 아니라) 클램프된 soma 로 풀린다
        basal_uses_clamped = v_basal != float(m.neurons.EL_mV[i0, 1])
        # (c) 전도도 비음수, 억제는 역전위로 구현
        nonneg = bool((st2.g >= 0).all()) and bool((info["delta_g"] >= 0).all())
        gaba_rev = float(RECEPTOR_PARAMS["GABA_A"]["E_rev_mV"])
        inhibitory_by_reversal = gaba_rev < THRESHOLD * V_UNIT_MV + EL
        # (d) dt 감소 수렴
        finals = []
        for factor in (1.0, 0.5, 0.25):
            c2 = json.loads(json.dumps(self.tiny_cfg))
            c2["engine"]["dt_ms"] = dt * factor
            m2 = CorticalModel(c2, self.device, self.dtype,
                               SeedStreams(int(c2["seed"])))
            s2 = DynamicState(1, 1, m2.neurons.n, m2.neurons, self.device, self.dtype)
            m2.prepare(1, 1)
            steps = int(round(20.0 / (dt * factor)))
            P2 = m2.gains.replicas(None)
            for s in range(steps):
                s2.g[0, 0, i0, 0, RECEPTOR_INDEX["AMPA"]] += 0.6 * factor
                m2.engine.step(s2, m2.ring, P2, s)
            finals.append(float(s2.V[0, 0, i0, 0]))
        spread = max(finals) - min(finals)
        dt_ok = spread <= 2.0
        passed = all([leak_ok, clamp_ok, nonneg, inhibitory_by_reversal, dt_ok])
        return CheckResult(
            "check_08_membrane", "누설·불응기 클램프·구획 결합·부호·비음수·dt 수렴",
            STATUS_PASSED if passed else STATUS_FAILED,
            "무입력 누설이 EL 로 0.5 mV 안에 수렴, 불응기 soma 가 V_reset 으로 클램프, "
            "전도도가 모두 비음수, dt 를 4배 줄여도 20 ms 후 차이 <= 2 mV",
            {"leak_final_mV": v_leak, "E_L_mV": EL, "leak_ok": leak_ok,
             "refractory_soma_mV": v_soma, "V_reset_mV": reset_v,
             "clamp_ok": clamp_ok, "basal_solved_with_clamped_soma": basal_uses_clamped,
             "conductance_nonnegative": nonneg,
             "inhibition_by_reversal_not_negative_g": inhibitory_by_reversal,
             "dt_finals_mV": finals, "dt_spread_mV": spread, "dt_ok": dt_ok},
            "tiny preset 엔진 직접 호출")

    def check_09_recording_modes(self) -> CheckResult:
        """full/selected/summary 에서 **신경 계산이 같은지**, selected 사건이 실제로 남는지."""
        require_torch()          # torch 없으면 여기서 멈춘다
        stim = self._tiny_stim("grating")
        n_steps = 20
        spikes: dict[str, int] = {}
        recorded: dict[str, int] = {}
        counters: dict[str, dict[str, int]] = {}
        tmp = Path(tempfile.mkdtemp(prefix="vcg_rec_"))
        try:
            for mode in ("full", "selected", "summary"):
                cfg = json.loads(json.dumps(self.tiny_cfg))
                cfg["recording"]["mode"] = mode
                m = CorticalModel(cfg, self.device, self.dtype,
                                  SeedStreams(int(cfg["seed"])))
                policy = RecordingPolicy(cfg, m.neurons, SeedStreams(int(cfg["seed"])))
                policy.allocate(1)
                rec = AsyncRecorder(tmp / mode, policy, cfg)
                drive, _ = self._prepared(m, stim, n_steps)
                P = m.gains.replicas(None)
                cap = EventCapture(m, policy) if policy.wants_events() else None
                sel = list(policy.selected_ids) if policy.wants_states() else []

                def hook(info: dict[str, Any], cap=cap, sel=sel, rec=rec, m=m) -> None:
                    if cap is not None:
                        cap.capture(info, sample_id=0, base_seq=0, episode_id=0,
                                    phase="eval", perturbation_id=-1,
                                    global_step=int(info["step"]), run_seq=0, P=P)
                    if sel and int(info["step"]) % policy.state_every == 0:
                        rec.write_states(state_rows(m, info, sel, sample_id=0,
                                                    episode_id=0, replica=0,
                                                    global_step=int(info["step"])))

                out = m.run(drive, P, n_steps, step_hook=hook)
                if cap is not None:
                    rows = cap.drain()
                    rec.write_events(rows, cap.n_emitted, cap.n_arrived, cap.n_filtered)
                rec.flush()
                spikes[mode] = int(out["spike_count"].sum())
                counters[mode] = dict(rec.counters)
                recorded[mode] = rec.counters["events_recorded"]
                rec.close("completed")
            same_compute = len(set(spikes.values())) == 1
            selected_kept = recorded["selected"] > 0
            summary_no_events = recorded["summary"] == 0
            separate_counts = all(
                counters[mo]["events_emitted"] != counters[mo]["events_recorded"]
                or counters[mo]["events_recorded"] == 0
                for mo in ("full", "selected"))
            passed = same_compute and selected_kept and summary_no_events
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return CheckResult(
            "check_09_recording_modes",
            "full/selected/summary 의 신경 계산 동일, selected 사건 실제 보존",
            STATUS_PASSED if passed else STATUS_FAILED,
            "세 모드의 총 스파이크 수가 같고, selected 에서 기록된 사건 행 > 0, "
            "summary 는 사건 행 0",
            {"spikes_by_mode": spikes, "events_recorded": recorded,
             "same_computation": same_compute, "selected_events_kept": selected_kept,
             "summary_has_no_events": summary_no_events,
             "counters_separate_emitted_vs_recorded": separate_counts},
            "tiny preset, 실제 기록기·임시 폴더",
            {"counters": counters})

    def check_10_query_isolation(self) -> CheckResult:
        """sample/episode/replica 가 다른 로그가 조회·집계에서 섞이지 않는지 (F16,F17)."""
        m = self._fresh_model()
        cfg = self.tiny_cfg
        policy = RecordingPolicy(cfg, m.neurons, SeedStreams(int(cfg["seed"])))
        cap = EventCapture(m, policy)
        n = m.neurons.n
        t = require_torch()
        m.prepare(1, 1)
        P = m.gains.replicas(None)
        q = t.ones((1, 1, n), dtype=self.dtype, device=self.device)
        for sample in (0, 1):
            for episode in (0, 1):
                m.ring.reset()
                m.ring.write(0, q)
                d = min(d for d, _ in m.synapses.delay_groups)
                info = {"step": d, "n_spikes": 0, "spikes": None, "u": None}
                cap.capture(info, sample_id=sample, base_seq=sample, episode_id=episode,
                            phase="eval", perturbation_id=-1, global_step=d,
                            run_seq=0, P=P)
        rows = cap.drain()
        col = {k: i for i, k in enumerate(EVENT_FIELDS)}
        combos = {(int(r[col["sample_id"]]), int(r[col["episode_id"]])) for r in rows}
        # 키는 JSON 으로 저장되므로 튜플이 아니라 사람이 읽는 문자열로 만든다.
        per_combo = {f"sample={c[0]},episode={c[1]}":
                     int(sum(1 for r in rows
                             if (int(r[col["sample_id"]]),
                                 int(r[col["episode_id"]])) == c))
                     for c in sorted(combos)}
        ids_unique = len({int(r[col["event_id"]]) for r in rows}) == rows.shape[0]
        last_ids = cap.last_event_id_per_target
        distinct_last = len(set(last_ids.values())) > 1 if len(last_ids) > 1 else True
        passed = (len(combos) == 4 and len(set(per_combo.values())) == 1
                  and ids_unique and distinct_last)
        return CheckResult(
            "check_10_query_isolation", "sample/episode/replica 로그 분리와 표적별 사건 ID",
            STATUS_PASSED if passed else STATUS_FAILED,
            "4개 (sample, episode) 조합이 각각 같은 수의 행을 갖고, event_id 가 모두 "
            "유일하며, 표적별 마지막 event_id 가 하나의 값으로 뭉개지지 않는다",
            {"combinations": [list(c) for c in sorted(combos)],
             "rows_per_combination": per_combo,
             "event_ids_unique": ids_unique,
             "distinct_last_event_ids": distinct_last,
             "n_targets_with_last_id": len(last_ids)},
            "tiny preset, EventCapture 직접 검사")

    def check_11_retina_geometry(self) -> CheckResult:
        """색 변환·균일 영상 DoG·로그-극좌표 좌표 대응·anti-aliasing 측정.

        좌표 변환의 **역함수 검사**와 다운샘플된 영상의 복원을 구분한다.
        손실 압축한 영상의 완전한 왕복 복원을 요구하지 않는다.
        """
        t = require_torch()
        m = self._fresh_model()
        size = int(self.tiny_cfg["image"]["max_side_px"])
        # (a) 색 변환: 두 단계가 분리되어 있는지 + D65 백색이 LMS 에서 양수인지
        white = t.ones((1, 3, 4, 4), dtype=self.dtype, device=self.device)
        lms = m.encoder.to_lms(white)
        lms_positive = bool((lms > 0).all())
        xyz_is_not_lms = not np.allclose(RGB_TO_XYZ, XYZ_TO_LMS_HPE_D65 @ RGB_TO_XYZ)
        # (b) 균일 영상의 DoG: 이론상 0, 경계에서만 잔여
        uni = t.full((1, 3, size, size), 0.5, dtype=self.dtype, device=self.device)
        ch = m.encoder.encode(uni)
        inner = ch[:, :6, size // 4:-size // 4, size // 4:-size // 4]
        dog_inner_max = float(inner.abs().max())
        lowpass_mean = float(ch[:, 6:].mean())
        dog_ok = dog_inner_max < 1e-3 and lowpass_mean > 0.0
        # (c) 로그-극좌표 역함수 왕복 (좌표만 검사한다)
        s = m.sampler
        rho = s.ecc_to_rho(s.ecc_deg)
        back = s.rho_to_ecc(rho)
        round_trip_err = float(np.max(np.abs(back - s.ecc_deg)))
        theta_ok = bool(np.all((s.theta_rad >= 0) & (s.theta_rad < 2 * math.pi + 1e-9)))
        col, row, valid = s.pixel_coords(size, size)
        in_image = float(np.mean(valid))
        # (d) anti-aliasing: 저역통과를 켠 것과 끈 것의 고주파 응답 차이를 **측정**
        high = grating_image(size, orientation_rad=0.0, phase_rad=0.0,
                             cycles_per_image=size / 3.0)
        img = t.tensor(high.transpose(2, 0, 1)[None], dtype=self.dtype, device=self.device)
        chan = m.encoder.encode(img)
        s.prefilter = True
        with_pref, _ = s.sample(chan)
        s.prefilter = False
        without_pref, _ = s.sample(chan)
        s.prefilter = True
        alias_metric = float((without_pref - with_pref).abs().mean())
        passed = lms_positive and xyz_is_not_lms and dog_ok and round_trip_err < 1e-9 \
            and theta_ok
        return CheckResult(
            "check_11_retina_geometry",
            "색 변환·균일 DoG·로그-극좌표 왕복·경계·중심 처리, anti-aliasing 측정",
            STATUS_PASSED if passed else STATUS_FAILED,
            "RGB->XYZ 와 XYZ->LMS 가 다른 단계이고, 균일 영상의 내부 DoG < 1e-3, "
            "ecc<->rho 왕복 오차 < 1e-9, theta in [0,2pi). anti-aliasing 은 측정만 한다",
            {"lms_positive_for_white": lms_positive,
             "xyz_matrix_distinct_from_lms": xyz_is_not_lms,
             "uniform_dog_inner_max": dog_inner_max,
             "lowpass_channel_mean": lowpass_mean,
             "ecc_rho_round_trip_max_error": round_trip_err,
             "theta_in_range": theta_ok, "fraction_samples_inside_image": in_image,
             "aliasing_difference_measured": alias_metric},
            "tiny preset, 망막/샘플러 경로",
            {"measurement_only_ko": ("anti-aliasing 차이는 측정값이다. 통과 기준을 "
                                     "두지 않았다."),
             "binocular": STATUS_NOT_SUPPORTED,
             "binocular_note_ko": ("좌우 영상과 경로를 실제로 분리하지 않았으므로 "
                                   "양안 검사는 제공하지 않는다. 미측정을 통과로 "
                                   "적지 않는다.")})

    def check_12_lgn_channel_preserved(self) -> CheckResult:
        """LGN 까지 ON/OFF·색 경로가 보존되는지, V1 반응이 입력에 따라 변하는지."""
        t = require_torch()
        m = self._fresh_model()
        a = m.neurons
        syn = m.synapses
        names = [r["name"] for r in self.tiny_cfg["wiring"]["rules"]]
        r_idx = names.index("Retina->LGN")
        sel = syn.rule_index == r_idx
        src_ch = a.channel_id[syn.src[sel]]
        dst_ch = a.channel_id[syn.dst[sel]]
        channel_match = float((src_ch == dst_ch).to(self.dtype).mean())
        src_pol = a.polarity[syn.src[sel]]
        dst_pol = a.polarity[syn.dst[sel]]
        polarity_match = float((src_pol == dst_pol).to(self.dtype).mean())
        # 공간 수렴 수: 같은 좌표의 채널 중복이 아니라 서로 다른 표본이어야 한다
        lgn_ids = a.indices_of("LGN")
        distinct: list[int] = []
        for nid in lgn_ids[:16].tolist():
            inc = syn.incoming_ids(int(nid))
            inc = inc[syn.rule_index[inc] == r_idx]
            distinct.append(int(t.unique(a.sample_id[syn.src[inc]]).numel()))
        mean_distinct = float(np.mean(distinct)) if distinct else 0.0
        # abs(Gabor) 를 쓰지 않았는지: LGN->V1 의 ON/OFF 가중치 분포가 다르다
        g_idx = names.index("LGN->V1_L4")
        gsel = syn.rule_index == g_idx
        pol_src = a.polarity[syn.src[gsel]]
        w = syn.w0_nS[gsel]
        on_mean = float(w[pol_src > 0].mean()) if int((pol_src > 0).sum()) else 0.0
        off_mean = float(w[pol_src < 0].mean()) if int((pol_src < 0).sum()) else 0.0
        # V1 실제 반응이 방향/위상에 따라 바뀌는지
        resp: dict[str, float] = {}
        counts: list[float] = []
        for sid in ("diag_ori_00", "diag_ori_03", "diag_phase_04"):
            stim = next(s for s in StimulusGenerator(
                self.tiny_cfg, SeedStreams(int(self.tiny_cfg["seed"]))).diagnostic_set()
                if s.stimulus_id == sid)
            mm = self._fresh_model()
            drive, _ = self._prepared(mm, stim, 30)
            out = mm.run(drive, mm.gains.replicas(None), 30)
            v = float(out["spike_count"][0, 0][mm.neurons.indices_of("V1")].sum())
            resp[sid] = v
            counts.append(v)
        varies = len(set(counts)) > 1
        passed = (channel_match == 1.0 and polarity_match == 1.0
                  and mean_distinct >= 2.0)
        return CheckResult(
            "check_12_lgn_channel_preserved",
            "망막 채널·극성이 LGN 까지 보존, V1 반응이 입력에 따라 변함",
            STATUS_PASSED if passed else STATUS_FAILED,
            "Retina->LGN 의 채널·극성 일치율이 1.0 이고 LGN 한 개의 서로 다른 공간 "
            "표본 수가 평균 2 이상",
            {"channel_match_fraction": channel_match,
             "polarity_match_fraction": polarity_match,
             "mean_distinct_spatial_sources": mean_distinct,
             "gabor_on_mean_w0": on_mean, "gabor_off_mean_w0": off_mean,
             "v1_spikes_by_stimulus": resp, "v1_response_varies": varies},
            "tiny preset 실제 배선 + 실제 응답",
            {"note_ko": ("V1 반응 변화는 측정값이다. 여기서 방향 선택성이 학습되었다고 "
                         "말하지 않는다 (초기 배선이 Gabor 모양이다).")})

    def check_13_shared_orientation_map(self) -> CheckResult:
        """같은 표면 좌표면 층이 달라도 선호 방향이 같은지 (F06)."""
        m = self._fresh_model()
        omap = m.maps["V1"]
        u = np.linspace(0.1, 1.4, 25)
        v = np.linspace(0.1, 1.4, 25)
        a1 = omap.evaluate(u, v)
        a2 = omap.evaluate(u, v)
        deterministic = float(np.max(np.abs(a1 - a2))) == 0.0
        a = m.neurons
        uv = a.surface_uv_mm.detach().to("cpu").numpy()
        ori = a.pref_orientation_rad.detach().to("cpu").numpy()
        lid = a.layer_id.detach().to("cpu").numpy()
        aid = a.area_id.detach().to("cpu").numpy()
        v1 = aid == a.area_names.index("V1")
        expected = omap.evaluate(uv[v1, 0], uv[v1, 1])
        max_err = float(np.max(np.abs(ori[v1] - expected)))
        layers = sorted(set(lid[v1].tolist()))
        n_maps = len({m.maps[k].signature for k in m.maps if k == "V1"})
        passed = deterministic and max_err < 1e-9 and n_maps == 1 and len(layers) >= 2
        return CheckResult(
            "check_13_shared_orientation_map", "층 간 공통 방향 지도 일치",
            STATUS_PASSED if passed else STATUS_FAILED,
            "V1 의 모든 층 뉴런의 선호 방향이 하나의 지도를 표면 좌표에서 평가한 값과 "
            "1e-9 안에서 같다",
            {"map_deterministic": deterministic, "max_abs_error": max_err,
             "n_v1_maps": n_maps, "n_layers_checked": len(layers)},
            "tiny preset, 실제 배치된 뉴런의 메타데이터",
            {"note_ko": "지도 일치는 구조 검사다. 실제 발화 선택성은 check_12 에서 따로 본다."})

    def check_14_transmission(self) -> CheckResult:
        """망막->LGN->V1 전달 양성 대조와 무입력 대조 (작은 회로로 대신하지 않는다)."""
        m_pos = self._fresh_model()
        pos = transmission_diagnostic(m_pos, self._tiny_stim("grating"), 40)
        m_neg = self._fresh_model()
        m_neg.encoder.load_state_dict(m_pos.encoder.state_dict())
        neg = transmission_diagnostic(m_neg, self._tiny_stim("dark"), 40)
        chain_ok = pos["chain_ok"]
        quiet = neg["by_area"].get("V1", {}).get("total_spikes", 0) == 0
        passed = bool(chain_ok)
        return CheckResult(
            "check_14_transmission", "망막->LGN->V1 전달 양성/무입력 대조",
            STATUS_PASSED if passed else STATUS_FAILED,
            "자극이 있으면 Retina/LGN/V1 각각에서 스파이크 > 0",
            {"positive_by_area": {k: v["total_spikes"] for k, v in pos["by_area"].items()},
             "positive_rates_hz": {k: v["mean_rate_hz"] for k, v in pos["by_area"].items()},
             "threshold_margin_u": {k: v["threshold_margin_u_max"]
                                    for k, v in pos["by_area"].items()},
             "silent_chain_areas": pos["silent_chain_areas"],
             "no_input_v1_spikes": neg["by_area"].get("V1", {}).get("total_spikes"),
             "no_input_quiet": quiet},
            "tiny preset 전체 배선 (작은 대체 회로가 아니다)",
            {"input_code": pos["input_code"], "drive": pos["drive"],
             "note_ko": ("전달이 약해도 임계값을 낮추거나 라벨을 입력에 넣지 않는다. "
                         "입력 파형·전류·실제 발화율·임계 여유를 위에 그대로 적는다.")})

    def check_15_feedback_apical(self) -> CheckResult:
        """상위->L1/apical 고정 배선이 표적 막전위를 실제로 바꾸는지 (F07).

        이 검사로 '오차가 줄었다' 까지 입증했다고 말하지 않는다.
        """
        t = require_torch()
        m = self._fresh_model()
        a, syn = m.neurons, m.synapses
        names = [r["name"] for r in self.tiny_cfg["wiring"]["rules"]]
        fb = [i for i, nm in enumerate(names) if "_FB_" in nm]
        if not fb:
            return CheckResult(
                "check_15_feedback_apical", "상위->L1/apical 경로 효과",
                STATUS_NOT_APPLICABLE,
                "이 preset 에 영역 간 하향 배선이 없으면 해당 없음",
                {"n_feedback_rules": 0, "areas": a.area_names},
                "tiny preset (V1+IT 만 있는 구성)",
                {"note_ko": "hierarchy_small preset 에서 다시 검사하라."})
        mask = t.zeros(syn.n_edges, dtype=t.bool, device=self.device)
        for i in fb:
            mask |= (syn.rule_index == i)
        apical = mask & (syn.comp == COMP_INDEX["apical"])
        targets = t.unique(syn.dst[apical]) if int(apical.sum()) else t.tensor([], dtype=t.long)
        if int(targets.numel()) == 0:
            return CheckResult(
                "check_15_feedback_apical", "상위->L1/apical 경로 효과",
                STATUS_FAILED, "apical 을 표적으로 하는 하향 시냅스가 존재",
                {"n_apical_feedback_edges": 0}, "tiny preset")
        n = a.n
        st = DynamicState(1, 1, n, a, self.device, self.dtype)
        m.prepare(1, 1)
        P = m.gains.replicas(None)
        tgt = int(targets[0])
        base_V = []
        for s in range(5):
            m.engine.step(st, m.ring, P, s)
            base_V.append(float(st.V[0, 0, tgt, 0]))
        st2 = DynamicState(1, 1, n, a, self.device, self.dtype)
        with_V = []
        for s in range(5):
            st2.g[0, 0, tgt, COMP_INDEX["apical"], RECEPTOR_INDEX["AMPA"]] += 20.0
            m.engine.step(st2, m.ring, P, s)
            with_V.append(float(st2.V[0, 0, tgt, 0]))
        delta = with_V[-1] - base_V[-1]
        rao_blocked = False
        try:
            assert_supported_learning("rao_apical_correction")
        except NotImplementedError:
            rao_blocked = True
        passed = abs(delta) > 1e-6 and rao_blocked
        return CheckResult(
            "check_15_feedback_apical",
            "상위->L1/apical 고정 배선이 막전위에 영향, Rao 교정은 not_supported",
            STATUS_PASSED if passed else STATUS_FAILED,
            "apical 전도도 주입이 soma 막전위를 1e-6 mV 이상 바꾸고, "
            "rao_apical_correction 을 켜면 명확한 오류가 난다",
            {"n_feedback_rules": len(fb),
             "n_apical_feedback_edges": int(apical.sum()),
             "V_soma_without_apical_mV": base_V[-1],
             "V_soma_with_apical_mV": with_V[-1], "delta_mV": delta,
             "rao_correction_raises": rao_blocked},
            "tiny preset 실제 배선 + 엔진",
            {"note_ko": "막전위 변화만 확인했다. 오차 감소를 입증한 것이 아니다."})

    def check_16_pm_pairing(self) -> CheckResult:
        """± 조건의 초기 상태·외생 입력이 같고 한쪽이 다른 쪽을 오염시키지 않는지 (F26)."""
        t = require_torch()
        m = self._fresh_model()
        stim = self._tiny_stim("shape")
        n_steps = 20
        drive, _ = self._prepared(m, stim, n_steps)
        # 복제본 축으로만 나뉘고 외생 입력은 [T,B,·] 하나를 공유한다
        exo_shared = drive.values.dim() == 3
        delta = t.zeros((2, m.neurons.n), dtype=self.dtype, device=self.device)
        d0 = t.randint(0, 2, (m.neurons.n,), device=self.device).to(self.dtype) * 2 - 1
        delta[0] = 0.05 * d0
        delta[1] = -0.05 * d0
        P = m.gains.replicas(delta)
        m.prepare(2, 1)
        v0 = m.state.V.clone()
        same_init = bool(t.equal(v0[0], v0[1]))
        labels = t.tensor([0], dtype=t.long, device=self.device)
        out = m.run(drive, P, n_steps, labels=labels)
        # 한쪽만 P=0 으로 만들어 다른 쪽 결과가 바뀌지 않는지 본다
        P2 = P.clone()
        P2[1] = 0.0
        out2 = m.run(drive, P2, n_steps, labels=labels)
        plus_unaffected = float((out["loss_per_sample"][0] - out2["loss_per_sample"][0]).abs())
        no_leak = plus_unaffected < 1e-6
        passed = same_init and exo_shared and no_leak
        return CheckResult(
            "check_16_pm_pairing", "± 조건 초기 상태·외생 입력 일치, 오염 없음",
            STATUS_PASSED if passed else STATUS_FAILED,
            "두 복제본의 초기 막전위가 비트 단위로 같고, 외생 입력 텐서를 공유하며, "
            "한쪽 P 를 바꿔도 다른 쪽 손실이 1e-6 안에서 그대로",
            {"same_initial_state": same_init, "exogenous_shared": exo_shared,
             "plus_side_loss_change_when_minus_zeroed": plus_unaffected,
             "no_cross_contamination": no_leak},
            "tiny preset, 실제 ± 복제본 실행")

    def check_17_zero_error_no_update(self) -> CheckResult:
        """오차 비교 신호를 0 으로 강제하면 Δz = ΔP = 0 인지.

        이것은 '손실 자체가 0' 이라는 뜻이 아니다.
        """
        t = require_torch()
        m = self._fresh_model()
        seeds = SeedStreams(int(self.tiny_cfg["seed"]))
        trainer = GainSPSATrainer(self.tiny_cfg, m, seeds)
        stim = self._tiny_stim("shape")
        img = t.tensor(np.stack([stim.frames[0]]).transpose(0, 3, 1, 2),
                       dtype=self.dtype, device=self.device)
        values, _ = m.sampler.sample(m.encoder.encode(img))
        m.encoder.fit_normalization(values, "train")
        labels = t.tensor([0], dtype=t.long, device=self.device)
        z_before = m.gains.z.clone()
        original = m.decoder.loss

        def zero_loss(logits: Any, lab: Any) -> tuple[Any, Any]:
            R, B, _ = logits.shape
            per = t.full((R, B), 2.5, dtype=self.dtype, device=self.device)
            return per.mean(dim=1), per       # L+ 와 L- 가 정확히 같다 (손실은 0 이 아니다)

        m.decoder.loss = zero_loss            # type: ignore[assignment]
        try:
            rec = trainer.step(img, labels, 20, drive_key=("input_noise", 3), epoch=0,
                               batch_index=0)
        finally:
            m.decoder.loss = original         # type: ignore[assignment]
        dz = float((m.gains.z - z_before).abs().max())
        dP = float(rec["delta_P_max"])
        loss_not_zero = rec["loss_plus_mean"] != 0.0
        passed = dz == 0.0 and dP == 0.0 and loss_not_zero
        return CheckResult(
            "check_17_zero_error_no_update", "오차 차이 0 이면 Δz = ΔP = 0",
            STATUS_PASSED if passed else STATUS_FAILED,
            "L+ == L- 이면 |Δz|max == 0 이고 |ΔP|max == 0. 손실 값 자체는 0 이 아니어도 된다",
            {"max_abs_delta_z": dz, "max_abs_delta_P": dP,
             "loss_plus": rec["loss_plus_mean"], "loss_minus": rec["loss_minus_mean"],
             "loss_is_nonzero": loss_not_zero},
            "tiny preset, 손실만 상수로 바꾼 실제 SPSA 스텝")

    def check_18_spsa_on_quadratic(self) -> CheckResult:
        """매끄러운 이차함수에서 SPSA 수식의 부호·스케일 검사.

        작은 차원에서 **모든 ± 방향의 평균**과 정확한 미분을 비교한다.
        단일 무작위 방향이 정확한 기울기라고 주장하지 않는다.
        """
        rng = np.random.default_rng(7)
        d = 6
        A = rng.normal(size=(d, d))
        H = A.T @ A + np.eye(d)
        b = rng.normal(size=d)
        x = rng.normal(size=d)

        def f(v: np.ndarray) -> float:
            return float(0.5 * v @ H @ v + b @ v)

        exact = H @ x + b
        c = 1e-4
        # 모든 Rademacher 방향(2^d)의 평균은 정확한 기울기로 수렴한다
        total = np.zeros(d)
        n_dir = 0
        for bits in range(2 ** d):
            delta = np.array([1.0 if (bits >> i) & 1 else -1.0 for i in range(d)])
            g = (f(x + c * delta) - f(x - c * delta)) / (2 * c) * delta
            total += g
            n_dir += 1
        avg = total / n_dir
        rel = float(np.linalg.norm(avg - exact) / max(np.linalg.norm(exact), 1e-12))
        single = (f(x + c * np.ones(d)) - f(x - c * np.ones(d))) / (2 * c) * np.ones(d)
        single_rel = float(np.linalg.norm(single - exact) / max(np.linalg.norm(exact), 1e-12))
        sign_ok = float(np.dot(avg, exact)) > 0
        passed = rel < 1e-6 and sign_ok
        return CheckResult(
            "check_18_spsa_on_quadratic", "매끄러운 이차함수에서 SPSA 부호·스케일",
            STATUS_PASSED if passed else STATUS_FAILED,
            f"모든 {2 ** d}개 ± 방향 평균과 정확한 미분의 상대 오차 < 1e-6, 내적 > 0",
            {"dimension": d, "n_directions_averaged": n_dir,
             "relative_error_all_directions": rel,
             "relative_error_single_direction": single_rel, "sign_ok": sign_ok},
            "순수 수치 함수 (신경 모델이 아니다)",
            {"note_ko": ("단일 무작위 방향의 오차가 큰 것은 정상이다. 이것이 SPSA 가 "
                         "정확한 기울기가 아니라는 뜻이다.")})

    def check_19_spiking_perturbation(self) -> CheckResult:
        """스파이크 모델에서 섭동에 따른 오차 변화와 0 차이 비율을 **측정**한다.

        불연속 발화가 있으므로 '정확한 BP 기울기를 검증했다' 고 말하지 않는다.
        기준을 두지 않는 측정 항목이므로 상태는 ``measured`` 다.
        """
        t = require_torch()
        m = self._fresh_model()
        seeds = SeedStreams(int(self.tiny_cfg["seed"]))
        stim = self._tiny_stim("shape")
        img = t.tensor(np.stack([stim.frames[0]]).transpose(0, 3, 1, 2),
                       dtype=self.dtype, device=self.device)
        values, _ = m.sampler.sample(m.encoder.encode(img))
        m.encoder.fit_normalization(values, "train")
        labels = t.tensor([0], dtype=t.long, device=self.device)
        rows: list[dict[str, Any]] = []
        for c in (0.002, 0.02, 0.2):
            cfg = json.loads(json.dumps(self.tiny_cfg))
            cfg["training"]["c"] = c
            tr = GainSPSATrainer(cfg, m, seeds)
            zero = 0
            diffs: list[float] = []
            for k in range(4):
                z_keep = m.gains.z.clone()
                rec = tr.step(img, labels, 20, drive_key=("input_noise", 5 + k),
                              epoch=0, batch_index=k)
                m.gains.z = z_keep               # 측정이 상태를 바꾸지 않게 되돌린다
                diffs.append(float(rec["loss_difference_mean"]))
                zero += int(rec["zero_difference_pairs"])
            rows.append({"c": c, "zero_difference_pairs": zero, "n_trials": 4,
                         "loss_difference_mean": float(np.mean(diffs)),
                         "loss_difference_std": float(np.std(diffs))})
        return CheckResult(
            "check_19_spiking_perturbation",
            "스파이크 모델에서 섭동에 따른 오차 변화와 0 차이 비율 (측정)",
            STATUS_MEASURED,
            "기준 없음 — 측정만 한다. 작은 섭동에서 0 차이가 많고 큰 섭동에서 잡음이 "
            "커지는 것이 이 방법의 알려진 성질이다",
            rows,
            "tiny preset, 실제 스파이크 모델",
            {"note_ko": ("이 결과로 '정확한 기울기' 를 검증했다고 말하지 않는다. "
                         "단일 단계 손실 감소나 수렴을 보장하지도 않는다.")})

    def check_20_readout(self) -> CheckResult:
        """readout 시간창·클래스 일치·영특징 경고·고정 해독기 불변."""
        t = require_torch()
        m = self._fresh_model()
        dec = m.decoder
        sig_before = dec.signature
        in_window = [dec.in_window(s) for s in (dec.start_step - 1, dec.start_step,
                                                dec.end_step - 1, dec.end_step)]
        half_open_ok = in_window == [False, True, True, False]
        classes_ok = dec.classes == list(self.tiny_cfg["decoder"]["classes"])
        # 창 밖 발화는 특징에 들어가지 않는다
        n_dec = int(dec.neuron_ids.numel())
        acc = t.zeros((1, 1, n_dec), dtype=self.dtype, device=self.device)
        rates = dec.rates(acc)
        logits = dec.logits(rates)
        labels = t.tensor([0], dtype=t.long, device=self.device)
        diag = dec.diagnostics(rates, logits, labels)
        zero_flag = diag["zero_feature_fraction"] == 1.0
        uniform = float(logits.max() - logits.min()) == 0.0
        # 고정 해독기 불변
        sig_after = dec.signature
        groups_balanced = max([int((dec.group == c).sum()) for c in range(len(dec.classes))]) \
            - min([int((dec.group == c).sum()) for c in range(len(dec.classes))]) <= 1
        passed = half_open_ok and classes_ok and zero_flag and uniform \
            and sig_before == sig_after and groups_balanced
        return CheckResult(
            "check_20_readout", "readout 시간창·클래스·영특징 경고·해독기 불변",
            STATUS_PASSED if passed else STATUS_FAILED,
            "시간창이 반개구간 [start,end) 이고, 특징이 0 이면 영특징 비율 1.0 과 "
            "균등 logit 이 보고되며, 해독기 서명이 변하지 않는다",
            {"half_open_window_ok": half_open_ok,
             "window_steps": [dec.start_step, dec.end_step],
             "classes_match_config": classes_ok,
             "zero_feature_flagged": zero_flag, "logits_uniform_when_zero": uniform,
             "decoder_signature_stable": sig_before == sig_after,
             "groups_balanced": groups_balanced,
             "majority_baseline": diag["majority_baseline"]},
            "tiny preset, 고정 해독기 직접 검사",
            {"note_ko": diag["note_ko"]})

    def check_21_splits(self) -> CheckResult:
        """같은 base_id 변형이 분할을 넘나들지 않고, dev/test 가 P·정규화·클래스를 바꾸지 않는지."""
        t = require_torch()
        seeds = SeedStreams(int(self.tiny_cfg["seed"]))
        gen = StimulusGenerator(self.tiny_cfg, seeds)
        stims = gen.classification_set()
        splits = gen.split(stims)
        report = gen.split_report(splits)
        m = self._fresh_model()
        take = splits["train"][:4]
        img = t.tensor(np.stack([s.frames[0] for s in take]).transpose(0, 3, 1, 2),
                       dtype=self.dtype, device=self.device)
        values, _ = m.sampler.sample(m.encoder.encode(img))
        info = m.encoder.fit_normalization(values, "train")
        scale_before = m.encoder.scale.clone()
        z_before = m.gains.z.clone()
        classes_before = list(m.decoder.classes)
        # dev 로 다시 추정하려 하면 막혀야 한다
        refit_blocked = False
        try:
            m.encoder.fit_normalization(values, "dev")
        except ValueError:
            refit_blocked = True
        # dev/test 평가는 P 와 정규화를 바꾸지 않는다
        for split in ("dev", "test"):
            sub = splits[split][:2]
            if not sub:
                continue
            im = t.tensor(np.stack([s.frames[0] for s in sub]).transpose(0, 3, 1, 2),
                          dtype=self.dtype, device=self.device)
            normalized, _ = m.encode_batch(im)
            drive = m.make_drive(normalized, 20)
            m.run(drive, m.gains.replicas(None), 20)
        scale_same = bool(t.equal(scale_before, m.encoder.scale))
        z_same = bool(t.equal(z_before, m.gains.z))
        classes_same = classes_before == list(m.decoder.classes)
        diag_separate = all(s.kind == "diagnostic" for s in gen.diagnostic_set())
        passed = (report["no_overlap"] and refit_blocked and scale_same and z_same
                  and classes_same and diag_separate)
        return CheckResult(
            "check_21_splits", "base_id 분할 누출 없음, dev/test 가 상태를 바꾸지 않음",
            STATUS_PASSED if passed else STATUS_FAILED,
            "분할 간 base_id 교집합이 없고, dev 로 정규화 재추정이 막히며, dev/test "
            "평가 후 P·정규화 계수·클래스 목록이 그대로",
            {"base_overlaps": report["base_overlaps"], "counts": report["counts"],
             "refit_on_dev_blocked": refit_blocked,
             "normalization_unchanged": scale_same, "P_unchanged": z_same,
             "classes_unchanged": classes_same,
             "diagnostics_separate_from_classification": diag_separate,
             "normalization_fit_split": info["split"]},
            "tiny preset 자극 생성기 + 실제 평가 경로")

    def check_22_resume(self) -> CheckResult:
        """실제 파일 저장과 Runner 재개를 거쳐 연속 실행과 중단·재개를 비교한다."""
        t = require_torch()
        cfg = json.loads(json.dumps(self.tiny_cfg))
        cfg["training"].update({"batch_size": 2, "epochs": 1, "K": 1})
        cfg["recording"]["mode"] = "selected"
        tmp = Path(tempfile.mkdtemp(prefix="vcg_resume_"))
        try:
            cont = ExperimentRunner(cfg, output_root=tmp / "continuous",
                                    device_choice=("cuda" if self.device.type == "cuda"
                                                   else "cpu"))
            res_a = cont.run_train(epochs=1, max_samples=4)
            ck_a = CheckpointManager(cont.run_dir).latest()
            meta_a, tens_a = CheckpointManager(cont.run_dir).load(ck_a)

            split = ExperimentRunner(cfg, output_root=tmp / "split",
                                     device_choice=("cuda" if self.device.type == "cuda"
                                                    else "cpu"))
            split.run_train(epochs=1, max_samples=4, stop_after_batches=1)
            run_dir_b = split.run_dir
            resumed = ExperimentRunner(cfg, run_dir=run_dir_b,
                                       device_choice=("cuda" if self.device.type == "cuda"
                                                      else "cpu"))
            res_b = resumed.run_train(
                epochs=1, max_samples=4,
                resume_from=CheckpointManager(run_dir_b).latest())
            ck_b = CheckpointManager(run_dir_b).latest()
            meta_b, tens_b = CheckpointManager(run_dir_b).load(ck_b)

            z_diff = float(np.max(np.abs(tens_a["z"] - tens_b["z"])))
            order_same = bool(np.array_equal(tens_a["order"], tens_b["order"]))
            split_same = meta_a["split_ids"] == meta_b["split_ids"]
            digest_same = meta_a["data_digest"] == meta_b["data_digest"]
            norm_same = float(np.max(np.abs(
                tens_a.get("norm_scale", np.zeros(1))
                - tens_b.get("norm_scale", np.zeros(1))))) < 1e-9
            rows_a = meta_a.get("committed_rows", {})
            rows_b = meta_b.get("committed_rows", {})
            tol = 1e-5 if self.dtype == t.float32 else 1e-12
            passed = (z_diff <= tol and order_same and split_same and digest_same
                      and norm_same)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return CheckResult(
            "check_22_resume", "저장·재개가 연속 실행과 같은 결과를 주는지",
            STATUS_PASSED if passed else STATUS_FAILED,
            f"연속 실행과 중단·재개의 최종 z 최대 차이 <= {tol}, 자극 순서·분할·자료 "
            f"해시·정규화 계수가 모두 같다",
            {"max_abs_z_difference": z_diff, "order_identical": order_same,
             "splits_identical": split_same, "data_digest_identical": digest_same,
             "normalization_identical": norm_same,
             "committed_rows_continuous": rows_a, "committed_rows_resumed": rows_b,
             "resume_unit": meta_b.get("resume_unit")},
            "tiny preset, 실제 파일 저장 + Runner 재개 (임시 폴더)",
            {"continuous_status": res_a["status"], "resumed_status": res_b["status"]})

    def check_23_poisson_frames(self) -> CheckResult:
        """Poisson 여러 프레임에서 **뒤 프레임에도** 입력 사건이 도착하는지 (F19)."""
        t = require_torch()
        cfg = json.loads(json.dumps(self.tiny_cfg))
        cfg["retina"]["drive"]["mode"] = "poisson"
        m = CorticalModel(cfg, self.device, self.dtype, SeedStreams(int(cfg["seed"])))
        stim = self._tiny_stim("frames")
        n_steps = 30
        frames = np.stack(stim.frames).transpose(0, 3, 1, 2)
        imgs = t.tensor(frames[None], dtype=self.dtype, device=self.device)
        per = [m.encode_batch(imgs[:, f])[0] if f else
               m.encode_batch(imgs[:, f], fit_normalization="train")[0]
               for f in range(frames.shape[0])]
        normalized = t.stack(per, dim=1)
        drive = m.make_drive(normalized, n_steps, rng_key=("input_noise", 9))
        bounds = drive.meta["frame_boundaries"]
        per_frame_events = []
        for f in range(len(bounds) - 1):
            lo, hi = bounds[f], bounds[f + 1]
            per_frame_events.append(int((drive.values[lo:hi] > 0).sum()))
        later_ok = all(v > 0 for v in per_frame_events[1:]) if len(per_frame_events) > 1 \
            else False
        absolute_steps = bounds[0] == 0 and bounds[-1] == n_steps
        forced_spikes_used = False        # Poisson 을 강제 발화로 우회하지 않는다
        passed = later_ok and absolute_steps and not forced_spikes_used
        return CheckResult(
            "check_23_poisson_frames", "Poisson 다중 프레임에서 뒤 프레임에도 입력 도착",
            STATUS_PASSED if passed else STATUS_FAILED,
            "프레임 경계가 절대 스텝으로 배치되고 두 번째 이후 프레임의 사건 수 > 0",
            {"frame_boundaries": bounds, "events_per_frame": per_frame_events,
             "later_frames_have_events": later_ok,
             "absolute_step_placement": absolute_steps,
             "poisson_implemented_as_conductance_pulse": drive.mode == "poisson",
             "bypassed_threshold_with_forced_spikes": forced_spikes_used},
            "tiny preset, Poisson 모드 실제 구동 텐서",
            {"note_ko": ("Poisson 사건은 전도도 펄스로 들어가며 모든 뉴런이 같은 "
                         "임계값 15 판정을 거친다.")})

    def check_24_diagnostics_pure(self) -> CheckResult:
        """진단·시각화 전후 모델/P/RNG 상태가 바뀌지 않는지 (F25)."""
        t = require_torch()
        m = self._fresh_model()
        stim = self._tiny_stim("grating")
        drive, _ = self._prepared(m, stim, 20)
        m.run(drive, m.gains.replicas(None), 20)
        fixed_before = dict(m.fixed_hashes)
        z_before = m.gains.z.clone()
        rng_before = SeedStreams(int(self.tiny_cfg["seed"])).state_dict()
        _ = transmission_diagnostic(m, stim, 20)
        view = m.neuron_view(int(m.decoder.neuron_ids[0]), run_id="x", sample_id=0)
        _ = view.to_dict()
        abl = ablation_compare(self.tiny_cfg, self.device, self.dtype,
                               int(self.tiny_cfg["seed"]), stim, 20,
                               ablate_rule="Retina->LGN")
        fixed_after = m._compute_fixed_hashes()
        changed = sorted(k for k in fixed_before if fixed_before[k] != fixed_after.get(k))
        z_same = bool(t.equal(z_before, m.gains.z))
        rng_same = rng_before == SeedStreams(int(self.tiny_cfg["seed"])).state_dict()
        passed = (not changed) and z_same and rng_same and abl["n_synapses_removed"] > 0
        return CheckResult(
            "check_24_diagnostics_pure", "진단·조회·소거 비교가 모델을 바꾸지 않음",
            STATUS_PASSED if passed else STATUS_FAILED,
            "진단 전후 고정 텐서 해시·z·명명된 RNG 상태가 모두 같다",
            {"changed_fixed_keys": changed, "P_unchanged": z_same,
             "rng_unchanged": rng_same,
             "ablation_used_independent_copy": abl["note_ko"],
             "ablation_removed": abl["n_synapses_removed"],
             "ablation_changed_neurons": abl["neurons_with_changed_spike_count"]},
            "tiny preset, 실제 진단 경로")

    def check_25_gpu_cpu_tolerance(self) -> CheckResult:
        """GPU/CPU 비교는 허용오차와 임계 여유를 밝힌 작은 사례로 한다.

        임계 근처 분기와 장시간 재귀 회로에서 비트 일치를 요구하지 않는다.
        """
        t = require_torch()
        if not EnvironmentInfo.detect().cuda_available:
            return CheckResult(
                "check_25_gpu_cpu_tolerance", "GPU/CPU 작은 사례 비교",
                STATUS_NOT_RUN, "CUDA 가 있어야 비교할 수 있다",
                {"cuda_available": False},
                "-", {"note_ko": "GPU 가 없는 환경이므로 실행하지 않았다. 통과로 적지 않는다."})
        stim = self._tiny_stim("grating")
        n_steps = 12
        outs: dict[str, Any] = {}
        for dev_name in ("cpu", "cuda"):
            dev = t.device(dev_name)
            m = CorticalModel(self.tiny_cfg, dev, t.float64 if dev_name == "cpu"
                              else self.dtype, SeedStreams(int(self.tiny_cfg["seed"])))
            img = t.tensor(np.stack([stim.frames[0]]).transpose(0, 3, 1, 2),
                           dtype=m.dtype, device=dev)
            values, _ = m.sampler.sample(m.encoder.encode(img))
            m.encoder.fit_normalization(values, "train")
            normalized, _ = m.encode_batch(img)
            drive = m.make_drive(normalized, n_steps, rng_key=("diagnostics", 2))
            out = m.run(drive, m.gains.replicas(None), n_steps)
            outs[dev_name] = {
                "u": m.state.last_u[0, 0].detach().to("cpu").to(t.float64).numpy(),
                "spikes": int(out["spike_count"].sum()),
            }
        du = float(np.max(np.abs(outs["cpu"]["u"] - outs["cuda"]["u"])))
        margin = float(np.min(np.abs(outs["cpu"]["u"] - THRESHOLD)))
        tol = 1e-3
        passed = du <= tol
        return CheckResult(
            "check_25_gpu_cpu_tolerance", "GPU/CPU 작은 사례 비교 (허용오차 명시)",
            STATUS_PASSED if passed else STATUS_FAILED,
            f"판정값 u 의 최대 절대 차이 <= {tol} (비트 일치를 요구하지 않는다)",
            {"max_abs_u_difference": du, "tolerance": tol,
             "min_threshold_margin_u": margin,
             "spikes_cpu": outs["cpu"]["spikes"], "spikes_cuda": outs["cuda"]["spikes"]},
            "tiny preset, 12 스텝 짧은 실행",
            {"note_ko": ("임계 근처에서는 작은 차이가 발화 여부를 바꿀 수 있다. "
                         "위 임계 여유와 함께 읽어야 한다.")})

    def check_26_import_and_paths(self) -> CheckResult:
        """import 부작용 없음, 한글·공백·따옴표 경로 처리, 잘못된 경로의 명시적 실패."""
        src = Path(__file__).resolve()
        tmp = Path(tempfile.mkdtemp(prefix="vcg_import_"))
        import_ok = False
        created: list[str] = []
        detail: dict[str, Any] = {}
        try:
            code = (
                "import importlib.util, sys, os\n"
                f"spec = importlib.util.spec_from_file_location('vcg', r'{src}')\n"
                "m = importlib.util.module_from_spec(spec)\n"
                "sys.modules['vcg'] = m\n"
                "spec.loader.exec_module(m)\n"
                "print('IMPORT_OK', 'matplotlib' in sys.modules)\n"
            )
            proc = subprocess.run([sys.executable, "-c", code], cwd=str(tmp),
                                  capture_output=True, text=True, timeout=300)
            import_ok = proc.returncode == 0 and "IMPORT_OK" in proc.stdout
            detail["stdout"] = proc.stdout.strip()[-500:]
            detail["stderr"] = proc.stderr.strip()[-500:]
            detail["matplotlib_loaded_on_import"] = "True" in proc.stdout.split()[-1:]
            created = [p.name for p in tmp.iterdir()]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        no_side_effects = import_ok and not created
        cases = {
            '"C:\\사용자 폴더\\결과"': "C:\\사용자 폴더\\결과",
            "'D:/cortex runs/한글 폴더'": "D:/cortex runs/한글 폴더",
            "  \\\\server\\share\\결과  ": "\\\\server\\share\\결과",
        }
        path_ok = True
        parsed: dict[str, str] = {}
        for raw, expect in cases.items():
            got = clean_user_path(raw)
            parsed[raw] = str(got)
            if Path(expect).name != got.name:
                path_ok = False
        empty_raises = False
        try:
            clean_user_path('  ""  ')
        except ValueError:
            empty_raises = True
        bad_raises = False
        try:
            ensure_writable_dir(Path("\x00invalid"))
        except (OSError, ValueError):
            bad_raises = True
        passed = no_side_effects and path_ok and empty_raises and bad_raises
        return CheckResult(
            "check_26_import_and_paths", "import 부작용 없음 + 경로 처리",
            STATUS_PASSED if passed else STATUS_FAILED,
            "다른 폴더에서 import 만 했을 때 파일이 생기지 않고, 따옴표/공백/한글/UNC "
            "경로를 그대로 해석하며, 빈 경로와 잘못된 경로에서 명시적으로 실패",
            {"import_ok": import_ok, "files_created_by_import": created,
             "no_side_effects": no_side_effects, "parsed_paths": parsed,
             "path_parsing_ok": path_ok, "empty_path_raises": empty_raises,
             "invalid_path_raises": bad_raises},
            "별도 프로세스에서 이 파일을 import", detail)

    def check_27_limits_and_backpressure(self) -> CheckResult:
        """상태·이벤트 한도 초과와 writer backpressure 처리, 오류 후 일관된 재개."""
        cfg = json.loads(json.dumps(self.tiny_cfg))
        cfg["recording"].update({"mode": "selected", "event_budget_rows": 20,
                                 "state_budget_rows": 10, "queue_max_items": 4})
        m = self._fresh_model()
        policy = RecordingPolicy(cfg, m.neurons, SeedStreams(int(cfg["seed"])))
        alloc = policy.allocate(2)
        tmp = Path(tempfile.mkdtemp(prefix="vcg_limits_"))
        try:
            rec = AsyncRecorder(tmp, policy, cfg)
            rec.begin_sample()
            big = np.zeros((500, len(EVENT_FIELDS)), dtype=np.float64)
            big[:, 0] = np.arange(500)
            rec.write_events(big, n_emitted=500, n_arrived=500, n_filtered=0)
            rec.write_states(np.zeros((500, len(STATE_FIELDS)), dtype=np.float64))
            rec.flush()
            recorded_events = rec.counters["events_recorded"]
            recorded_states = rec.counters["state_rows_recorded"]
            stop_recorded = bool(rec.stop_reasons)
            # backpressure: 큐 크기보다 훨씬 많은 항목을 넣어도 손실 없이 처리된다
            for i in range(200):
                rec.log(f"backpressure {i}")
            rec.flush()
            committed = rec.committed_rows()
            rec.close("completed")
            # 저장된 행을 그대로 다시 읽을 수 있는지 (오류 후 일관된 재개)
            store = TableStore(tmp, cfg["recording"]["backend"])
            fields, rows = store.read_all("events")
            store.close()
            reread_ok = rows.shape[0] == committed.get("events", 0)
            log_lines = len((tmp / "run.log").read_text(encoding="utf-8").splitlines())
            errors_logged = (tmp / "errors.jsonl").is_file()
            budget_respected = (recorded_events <= policy.per_sample_event_budget
                                and recorded_states <= policy.per_sample_state_budget)
            passed = (budget_respected and stop_recorded and reread_ok
                      and log_lines >= 200 and errors_logged)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return CheckResult(
            "check_27_limits_and_backpressure", "기록 한도·backpressure·재읽기 일관성",
            STATUS_PASSED if passed else STATUS_FAILED,
            "표본 예산을 넘겨 쓰지 않고, 한도 도달 사유를 기록하며, 큐 크기보다 많은 "
            "로그도 손실 없이 저장되고, 저장된 행 수와 다시 읽은 행 수가 같다",
            {"allocation": alloc, "events_recorded": recorded_events,
             "states_recorded": recorded_states,
             "per_sample_event_budget": policy.per_sample_event_budget,
             "per_sample_state_budget": policy.per_sample_state_budget,
             "stop_reason_recorded": stop_recorded,
             "committed_rows": committed, "reread_matches_committed": reread_ok,
             "log_lines_written": log_lines, "errors_file_written": errors_logged},
            "실제 기록기·임시 폴더",
            {"note_ko": ("한도에 걸리면 이유를 적고 멈춘다. 선택/요약 기록으로 몰래 "
                         "바꾼 결과를 full 이라고 부르지 않는다.")})

    def check_28_report_matches_logs(self) -> CheckResult:
        """저장된 JSON/CSV 로 만든 보고서의 수치와 조건명이 일치하는지 (F27)."""
        tmp = Path(tempfile.mkdtemp(prefix="vcg_report_"))
        try:
            train = {
                "mode": "all_neuron_gain_spsa", "status": "completed", "reason": "",
                "forward_calls": 42,
                "P_change": {"mean_abs": 0.012345, "max_abs": 0.6789,
                             "changed_fraction": 1.0},
                "fixed_parameters_unchanged": True,
                "zero_difference_rate_mean": 0.25,
                "learning_outcome": {"verdict": "not_improved", "reason": "테스트용",
                                     "note_ko": ""},
                "test": {"status": "completed", "cross_entropy": 1.9459,
                         "accuracy": 0.1428, "balanced_accuracy": 0.1428,
                         "majority_baseline": 0.1428,
                         "zero_feature_fraction": 0.0},
                "test_access": {"n_test_evaluations": 1},
            }
            write_json(tmp / "train.json", train)
            write_json(tmp / "manifest.json", {
                "run_id": "report_test", "status": "completed",
                "learning": {"mode": "all_neuron_gain_spsa",
                             "trainable": ["z (P 좌표)"]},
                "code": {"sha256": "abc"}, "config_sha256": "def",
                "environment": EnvironmentInfo.detect().to_dict(),
                "device": str(self.device), "dtype": str(self.dtype),
                "deterministic": {"status": "n/a", "note_ko": ""},
            })
            path = ReportBuilder(tmp).build()
            text = path.read_text(encoding="utf-8")
            checks = {
                "forward_calls": "42" in text,
                "mode_name": "all_neuron_gain_spsa" in text,
                "P_change_mean": "0.012345" in text,
                "verdict": "not_improved" in text,
                "test_ce": "1.9459" in text,
                "majority_baseline": "0.1428" in text,
                "threshold_15": f"{THRESHOLD}" in text,
                "no_invented_accuracy": "0.99" not in text,
            }
            passed = all(checks.values())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return CheckResult(
            "check_28_report_matches_logs", "보고서 수치·조건명이 저장된 로그와 일치",
            STATUS_PASSED if passed else STATUS_FAILED,
            "train.json 의 조건명·순방향 호출 수·P 변화·판정·test 수치가 보고서 본문에 "
            "그대로 나타나고, 저장되지 않은 수치가 새로 생기지 않는다",
            checks, "임시 폴더에 만든 가짜 기록 -> ReportBuilder",
            {"note_ko": "보고서는 저장된 파일에서만 만든다. 새 계산을 넣지 않는다."})


def run_validation(cfg: dict[str, Any], output_root: Path, device_choice: str,
                   only: Sequence[str] | None = None,
                   progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """검사 모음을 실행하고 실행 폴더에 결과를 저장한다."""
    env = EnvironmentInfo.detect()
    device = resolve_device(device_choice, env)
    t = require_torch()
    dtype = t.float32 if cfg["dtype"] == "float32" else t.float64
    run_dir = new_run_dir(Path(output_root), "validate")
    write_json(run_dir / "resolved_config.json", cfg)
    suite = ValidationSuite(cfg, device, dtype, progress)
    result = suite.run_all(only)
    result["environment"] = env.to_dict()
    result["code"] = source_hash()
    result["config_sha256"] = config_hash(cfg)
    result["fix_register"] = fix_register_rows()
    result["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(run_dir / "validation.json", result)
    write_json(run_dir / "manifest.json", {
        "run_id": run_dir.name, "status": "completed",
        "started_utc": result["started_utc"],
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(sys.argv), "code": source_hash(),
        "config_sha256": config_hash(cfg), "environment": env.to_dict(),
        "device": str(device), "dtype": str(dtype),
        "deterministic": {"status": "suite", "note_ko": "검사마다 조건을 따로 적었다"},
        "learning": {"mode": "n/a (검증 실행)", "trainable": ["z (P 좌표)"]},
        "units": UNITS, "fix_register": fix_register_rows(),
    })
    result["run_dir"] = str(run_dir)
    try:
        ReportBuilder(run_dir).build()
    except Exception:
        pass
    return result


# ======================================================================
# 17. 한국어 메뉴와 CLI
# ======================================================================
MENU_TEXT = """
==================================================================
  시각피질 GPU 시뮬레이터 (고정 임계값 15, 출력 이득 P 만 학습)
==================================================================
  1. 환경·설정·메모리 예상 확인
  2. 작은 기능 검사 실행
  3. 망막->LGN->V1 신호 전달 진단
  4. 지정 영상/도형 시뮬레이션
  5. 출력 이득 P만 학습
  6. 고정·학습·대조 조건 비교
  7. 뉴런 3x3 상태와 입력 원인 조회
  8. 체크포인트에서 재개
  9. 저장 기록으로 보고서/그림 생성
 10. CPU/GPU 성능 비교
  0. 종료
==================================================================
  고른 작업만 실행한다. 아무 것도 자동으로 이어서 돌리지 않는다.
=================================================================="""


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return val or default


def _ask_preset() -> str:
    while True:
        p = _ask(f"preset ({'/'.join(PRESETS)})", "v1_small")
        if p in PRESETS:
            return p
        print(f"  '{p}' 은(는) 없는 preset 이다.")


def _ask_device(env: EnvironmentInfo) -> str:
    default = "auto"
    while True:
        d = _ask("장치 (auto/cuda/cpu)", default)
        if d in ("auto", "cuda", "cpu"):
            if d == "cuda" and not env.cuda_available:
                print("  CUDA 를 쓸 수 없다. auto 또는 cpu 를 고르거나 CUDA 빌드 "
                      "torch 를 설치하라: https://pytorch.org/get-started/locally/")
                continue
            return d
        print("  auto, cuda, cpu 중에서 고르라.")


def _ask_output_dir() -> Path:
    while True:
        raw = _ask("기록을 저장할 폴더 경로를 붙여넣으세요 (공백·한글·따옴표 가능)")
        if not raw:
            print("  폴더 경로가 필요하다.")
            continue
        try:
            path = clean_user_path(raw)
            return ensure_writable_dir(path)
        except (OSError, ValueError) as exc:
            print(f"  [경로 오류] {exc}")


class KoreanMenu:
    """사용자가 고른 작업만 실행하는 한국어 메뉴."""

    def __init__(self) -> None:
        self.env = EnvironmentInfo.detect()
        self.output: Path | None = None
        self.preset = "v1_small"
        self.device_choice = "auto"

    def show_setup(self) -> None:
        cfg = build_config(self.preset)
        print()
        print(f"[설정] preset={self.preset}, seed={cfg['seed']}, "
              f"dtype={cfg['dtype']}, device 요청={self.device_choice}")
        print(f"  임계값 {THRESHOLD} (모든 뉴런 고정, 학습하지 않음), "
              f"V_unit={V_UNIT_MV} mV")
        print(f"  한 표본 {cfg['engine']['sample_ms']} ms / dt {cfg['engine']['dt_ms']} ms "
              f"= {int(cfg['engine']['sample_ms'] / cfg['engine']['dt_ms'])} 스텝")
        print(f"  기록 모드: {cfg['recording']['mode']} "
              f"(backend={cfg['recording']['backend']})")
        print(f"  학습 대상: 출력 이득 P 뿐 (조건 {cfg['training']['mode']})")
        print("  규모·메모리 예상은 메뉴 1번에서 실제로 모델을 만들어 계산한다.")

    def loop(self) -> int:
        print(self.env.describe_ko())
        if torch is None:
            print("\nPyTorch 가 없어 계산을 실행할 수 없다. 위 안내를 보고 설치하라.")
            return 2
        self.output = _ask_output_dir()
        self.preset = _ask_preset()
        self.device_choice = _ask_device(self.env)
        self.show_setup()
        while True:
            print(MENU_TEXT)
            choice = _ask("번호를 선택하라", "0")
            try:
                if choice == "0":
                    print("종료한다. (다른 작업을 자동으로 실행하지 않는다.)")
                    return 0
                self.dispatch(choice)
            except KeyboardInterrupt:
                print("\n[중단] 작업을 중단했다. 마지막 완료 지점의 기록은 남아 있다.")
            except Exception as exc:
                print(f"\n[오류] {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=3)

    def dispatch(self, choice: str) -> None:
        cfg = build_config(self.preset)
        out = self.output
        assert out is not None
        if choice == "1":
            runner = ExperimentRunner(cfg, output_root=out,
                                      device_choice=self.device_choice, env=self.env)
            info = runner.run_inspect()
            print(dumps(info)[:4000])
            print("\n[알림] 위 값은 실행 전 추정이다. 실제 속도는 메뉴 10 에서 측정한다.")
        elif choice == "2":
            res = run_validation(cfg, out, self.device_choice,
                                 progress=lambda m: print(f"  … {m}"))
            print(f"\n검사 {res['n_checks']}건: 통과 {res['n_passed']} / "
                  f"실패 {res['n_failed']} / 건너뜀 {res['n_skipped']} / "
                  f"해당없음 {res['n_not_applicable']} / 미실행 {res['n_not_run']} / "
                  f"측정만 {res['n_measured']} / 미지원 {res['n_not_supported']}")
            for c in res["checks"]:
                print(f"  [{c['status']:>15s}] {c['name']}")
            print(f"\n결과: {res['run_dir']}")
        elif choice == "3":
            runner = ExperimentRunner(cfg, output_root=out,
                                      device_choice=self.device_choice, env=self.env)
            res = runner.run_diagnose()
            print(dumps(res["verdict"]))
            for area, v in res["positive_control"]["by_area"].items():
                print(f"  {area:8s} 스파이크 {v['total_spikes']:8d} "
                      f"평균 {v['mean_rate_hz']:7.2f} Hz  "
                      f"임계 여유 u {v['threshold_margin_u_max']:+.3f}")
            print(f"\n결과: {runner.run_dir}")
        elif choice == "4":
            path = _ask("영상 파일 경로 (비우면 내장 도형 사용)", "")
            runner = ExperimentRunner(cfg, output_root=out,
                                      device_choice=self.device_choice, env=self.env)
            if path:
                res = runner.run_simulate(image_path=clean_user_path(path))
            else:
                sid = _ask("자극 ID (비우면 train 첫 자극)", "")
                res = runner.run_simulate(stimulus_id=sid or None)
            print(dumps({k: v for k, v in res.items() if k != "by_area"}))
            for area, v in res["by_area"].items():
                print(f"  {area:8s} 스파이크 {v['spikes']:8d} "
                      f"평균 {v['mean_rate_hz']:7.2f} Hz")
            print(f"\n결과: {runner.run_dir}")
        elif choice == "5":
            mode = _ask(f"학습 조건 ({'/'.join(TRAIN_MODES)})", "all_neuron_gain_spsa")
            epochs = int(_ask("에폭 수", str(cfg["training"]["epochs"])))
            runner = ExperimentRunner(cfg, output_root=out,
                                      device_choice=self.device_choice, env=self.env)
            res = runner.run_train(mode=mode, epochs=epochs)
            print(f"\n상태: {res['status']} / 학습 결과 판정: "
                  f"{res['learning_outcome']['verdict']}")
            print(f"  P 변화 평균 |ΔP| {res['P_change']['mean_abs']:.6f}, "
                  f"고정 파라미터 불변 {res['fixed_parameters_unchanged']}")
            print(f"  test: {res['test'].get('status')} CE "
                  f"{res['test'].get('cross_entropy')} 정확도 "
                  f"{res['test'].get('accuracy')} (다수 기준선 "
                  f"{res['test'].get('majority_baseline')})")
            print(f"\n결과: {runner.run_dir}")
        elif choice == "6":
            epochs = int(_ask("조건마다 에폭 수", "1"))
            res = run_comparison(cfg, out, self.device_choice, epochs=epochs,
                                 progress=lambda m: print(f"  … {m}"))
            for cond, r in res["results"].items():
                print(f"  {cond:28s} 상태 {r.get('status'):10s} "
                      f"순방향 {r.get('forward_calls')} "
                      f"판정 {r.get('learning_outcome', {}).get('verdict')}")
            print(f"\n결과: {res['root']}")
        elif choice == "7":
            print("\n  넣을 것은 **파일이 아니라 실행 폴더**다. 메뉴 4(시뮬레이션)나 "
                  "5(학습)가 끝날 때 찍어 준 '결과: ...' 경로를 그대로 붙여넣어라.")
            print(f"  예) {out / 'simulate_20260920T012345Z_ab12cd34'}")
            print("  그 폴더 안에 manifest.json, resolved_config.json, "
                  "events.h5(또는 events_chunk*.npz) 가 있다.")
            recent = sorted((d for d in out.iterdir()
                             if d.is_dir() and (d / "manifest.json").is_file()),
                            key=lambda d: d.stat().st_mtime, reverse=True)[:5]
            if recent:
                print("  최근 실행 폴더:")
                for d in recent:
                    print(f"    - {d}")
            rd = clean_user_path(_ask("조회할 실행 폴더"))
            nid = int(_ask("뉴런 ID", "0"))
            sample = _ask("sample_id (비우면 전체)", "")
            episode = _ask("episode_id (비우면 전체)", "")
            replica = _ask("replica_id (비우면 전체)", "")
            insp = NeuronInspector(rd)
            scope: dict[str, Any] = {}
            if sample:
                scope["sample_id"] = int(sample)
            if episode:
                scope["episode_id"] = int(episode)
            if replica:
                scope["replica_id"] = int(replica)
            tables = insp.available_tables()
            print("\n[이 폴더의 기록]")
            print(dumps(tables))
            if tables["events"]["rows"] == 0:
                print("  [알림] 이 폴더에는 입력 사건 행이 없다. recording.mode 가 "
                      "summary 였거나, 조회하려는 뉴런이 선택 목록에 없었을 수 있다. "
                      "아래 3x3 조회의 구조·메타데이터는 그대로 볼 수 있다.")
            res = insp.explain(nid, **scope)
            print(dumps({k: v for k, v in res.items() if k != "events"})[:3000])
            cfg_run = read_json(rd / "resolved_config.json")
            env_dev = resolve_device(self.device_choice, self.env)
            t = require_torch()
            model = CorticalModel(cfg_run, env_dev,
                                  t.float32 if cfg_run["dtype"] == "float32" else t.float64,
                                  SeedStreams(int(cfg_run["seed"])))
            view = model.neuron_view(nid, run_id=rd.name, **scope)
            print("\n[3x3 기록 인터페이스]")
            print(dumps(view.to_dict())[:3000])
        elif choice == "8":
            rd = clean_user_path(_ask("재개할 실행 폴더"))
            cfg_run = read_json(rd / "resolved_config.json")
            runner = ExperimentRunner(cfg_run, run_dir=rd,
                                      device_choice=self.device_choice, env=self.env)
            res = runner.run_resume()
            print(f"상태: {res['status']} / 판정 "
                  f"{res['learning_outcome']['verdict']}")
        elif choice == "9":
            rd = clean_user_path(_ask("보고서를 만들 실행 폴더"))
            path = ReportBuilder(rd).build()
            print(f"보고서: {path}")
            try:
                figs = make_figures(rd)
                print(f"그림 {len(figs)}개: {figs}")
            except RuntimeError as exc:
                print(f"[그림 생략] {exc}")
        elif choice == "10":
            runner = ExperimentRunner(cfg, output_root=out,
                                      device_choice=self.device_choice, env=self.env)
            res = runner.run_benchmark()
            for r in res["runs"]:
                print(f"  {r['device']:5s} 조립 {r['build_seconds']:6.2f}s  "
                      f"전처리 {r['preprocess_seconds']:6.3f}s  "
                      f"스텝당 {r['kernel_ms_per_step']:8.3f} ms "
                      f"({r['measurement_method']})")
            print(f"\n결과: {runner.run_dir}")
        else:
            print(f"  '{choice}' 은(는) 없는 번호다. 0~10 중에서 고르라.")


CLI_MODES: tuple[str, ...] = (
    "env", "inspect", "validate", "diagnose", "simulate", "train", "compare",
    "resume", "report", "benchmark", "inspect-neuron",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"python {Path(__file__).name}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "고정 임계값 15, 출력 이득 P 만 학습하는 GPU 시각피질 모형.\n"
            "인자 없이 실행하면 한국어 메뉴가 열린다."),
        epilog=(
            "예시:\n"
            f"  python {Path(__file__).name}\n"
            f"  python {Path(__file__).name} --mode validate --device cuda "
            "--output \"D:\\cortex_runs\"\n"
            f"  python {Path(__file__).name} --mode diagnose --preset v1_small "
            "--device cuda --output \"D:\\cortex_runs\"\n"
            f"  python {Path(__file__).name} --mode train --preset hierarchy_small "
            "--device cuda --output \"D:\\cortex_runs\"\n"
            f"  python {Path(__file__).name} --mode resume "
            "--run-dir \"D:\\cortex_runs\\run_...\"\n"
            f"  python {Path(__file__).name} --mode report "
            "--run-dir \"D:\\cortex_runs\\run_...\"\n"))
    p.add_argument("--mode", choices=CLI_MODES,
                   help="실행할 작업. 생략하면 한국어 메뉴가 열린다")
    p.add_argument("--preset", choices=PRESETS, default="v1_small",
                   help="모델 규모 preset (기본: v1_small)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="계산 장치. auto 는 CUDA 가 없으면 CPU 를 쓰고 그 사실을 알린다")
    p.add_argument("--output", default=None,
                   help="결과를 저장할 폴더 (공백·한글·따옴표 경로 가능)")
    p.add_argument("--run-dir", default=None, help="기존 실행 폴더 (resume/report/조회)")
    p.add_argument("--seed", type=int, default=None, help="난수 시드 덮어쓰기")
    p.add_argument("--epochs", type=int, default=None, help="에폭 수")
    p.add_argument("--max-samples", type=int, default=0,
                   help="train 표본 수 제한 (0 이면 제한 없음)")
    p.add_argument("--batch-size", type=int, default=None, help="미니배치 크기")
    p.add_argument("--train-mode", choices=TRAIN_MODES, default=None,
                   help="학습 조건 (기본: 설정값)")
    p.add_argument("--conditions", default=None,
                   help="compare 에서 쓸 조건 목록 (쉼표 구분)")
    p.add_argument("--recording-mode", choices=["full", "selected", "summary"],
                   default=None, help="기록 모드")
    p.add_argument("--backend", choices=["auto", "hdf5", "npz"], default=None,
                   help="기록 backend. h5py 가 없으면 auto 가 npz 로 간다 (기록에 남는다)")
    p.add_argument("--stimulus", default=None, help="시뮬레이션할 자극 ID")
    p.add_argument("--image", default=None, help="시뮬레이션할 영상 파일 경로")
    p.add_argument("--checks", default=None,
                   help="실행할 검사 이름 목록 (쉼표 구분, 생략하면 전부)")
    p.add_argument("--neuron-id", type=int, default=None, help="조회할 뉴런 ID")
    p.add_argument("--sample-id", type=int, default=None)
    p.add_argument("--episode-id", type=int, default=None)
    p.add_argument("--replica-id", type=int, default=None)
    p.add_argument("--figures", action="store_true",
                   help="report 모드에서 그림도 만든다 (matplotlib 필요)")
    return p


def _cfg_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = build_config(args.preset)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.train_mode is not None:
        cfg["training"]["mode"] = args.train_mode
    if args.recording_mode is not None:
        cfg["recording"]["mode"] = args.recording_mode
    if args.backend is not None:
        cfg["recording"]["backend"] = args.backend
    validate_config(cfg)
    return cfg


def _need_output(args: argparse.Namespace) -> Path:
    if not args.output:
        raise SystemExit("이 작업에는 --output <폴더> 가 필요하다.")
    return ensure_writable_dir(clean_user_path(args.output))


def _need_run_dir(args: argparse.Namespace) -> Path:
    if not args.run_dir:
        raise SystemExit("이 작업에는 --run-dir <실행 폴더> 가 필요하다.")
    p = clean_user_path(args.run_dir)
    if not p.is_dir():
        raise SystemExit(f"실행 폴더가 없다: {p}")
    return p


def run_cli(args: argparse.Namespace) -> int:
    env = EnvironmentInfo.detect()
    say = lambda m: print(f"  … {m}", flush=True)      # noqa: E731
    if args.mode == "env":
        print(env.describe_ko())
        print("\n" + dumps(env.to_dict()))
        return 0
    if torch is None:
        print(env.describe_ko())
        print("\nPyTorch 가 없어 계산을 실행할 수 없다.")
        return 2
    if args.mode == "report":
        rd = _need_run_dir(args)
        path = ReportBuilder(rd).build()
        print(f"보고서: {path}")
        if args.figures:
            try:
                print("그림:", make_figures(rd))
            except RuntimeError as exc:
                print(f"[그림 생략] {exc}")
        return 0
    if args.mode == "inspect-neuron":
        rd = _need_run_dir(args)
        if args.neuron_id is None:
            raise SystemExit("--neuron-id 가 필요하다.")
        scope = {k: v for k, v in (("sample_id", args.sample_id),
                                   ("episode_id", args.episode_id),
                                   ("replica_id", args.replica_id)) if v is not None}
        res = NeuronInspector(rd).explain(int(args.neuron_id), **scope)
        print(dumps({k: v for k, v in res.items() if k != "events"}))
        return 0

    cfg = _cfg_from_args(args)
    if args.mode == "inspect":
        runner = ExperimentRunner(cfg, output_root=(clean_user_path(args.output)
                                                    if args.output else None),
                                  device_choice=args.device, env=env)
        print(dumps(runner.run_inspect()))
        return 0
    if args.mode == "validate":
        out = _need_output(args)
        only = [c.strip() for c in args.checks.split(",")] if args.checks else None
        res = run_validation(cfg, out, args.device, only, progress=say)
        print(f"\n검사 {res['n_checks']}건: 통과 {res['n_passed']} / "
              f"실패 {res['n_failed']} / 건너뜀 {res['n_skipped']} / "
              f"해당없음 {res['n_not_applicable']} / 미실행 {res['n_not_run']} / "
              f"측정만 {res['n_measured']} / 미지원 {res['n_not_supported']}")
        for c in res["checks"]:
            print(f"  [{c['status']:>15s}] {c['name']}  ({c['elapsed_sec']}s)")
        print(f"\n결과: {res['run_dir']}")
        return 0 if res["n_failed"] == 0 else 1
    if args.mode == "diagnose":
        out = _need_output(args)
        runner = ExperimentRunner(cfg, output_root=out, device_choice=args.device, env=env)
        res = runner.run_diagnose()
        print(dumps(res["verdict"]))
        print(f"결과: {runner.run_dir}")
        return 0 if res["verdict"]["chain_transmits"] else 1
    if args.mode == "simulate":
        out = _need_output(args)
        runner = ExperimentRunner(cfg, output_root=out, device_choice=args.device, env=env)
        res = runner.run_simulate(
            stimulus_id=args.stimulus,
            image_path=clean_user_path(args.image) if args.image else None)
        print(dumps({k: v for k, v in res.items() if k != "by_area"}))
        print(f"결과: {runner.run_dir}")
        return 0
    if args.mode == "train":
        out = _need_output(args)
        runner = ExperimentRunner(cfg, output_root=out, device_choice=args.device, env=env)
        res = runner.run_train(mode=args.train_mode, epochs=args.epochs,
                               max_samples=args.max_samples)
        print(dumps({k: v for k, v in res.items()
                     if k not in ("history", "dev_history")}))
        print(f"결과: {runner.run_dir}")
        return 0 if res["status"] == "completed" else 1
    if args.mode == "compare":
        out = _need_output(args)
        conds = [c.strip() for c in args.conditions.split(",")] if args.conditions else None
        res = run_comparison(cfg, out, args.device, conds, epochs=args.epochs,
                             max_samples=args.max_samples, progress=say)
        print(dumps(res["compute_budget"]))
        print(f"결과: {res['root']}")
        return 0
    if args.mode == "resume":
        rd = _need_run_dir(args)
        cfg_run = read_json(rd / "resolved_config.json")
        runner = ExperimentRunner(cfg_run, run_dir=rd, device_choice=args.device, env=env)
        res = runner.run_resume()
        print(dumps({k: v for k, v in res.items()
                     if k not in ("history", "dev_history")}))
        return 0 if res["status"] == "completed" else 1
    if args.mode == "benchmark":
        out = _need_output(args)
        runner = ExperimentRunner(cfg, output_root=out, device_choice=args.device, env=env)
        res = runner.run_benchmark()
        print(dumps(res))
        print(f"결과: {runner.run_dir}")
        return 0
    raise SystemExit(f"알 수 없는 mode: {args.mode}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.mode is None:
        return KoreanMenu().loop()
    try:
        return run_cli(args)
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 Ctrl+C 로 중단했다. 마지막 완료 지점의 체크포인트가 "
              "남아 있다. --mode resume --run-dir <폴더> 로 이어서 할 수 있다.")
        return 130
    except EngineDivergence as exc:
        print(f"\n[수치 발산] {exc}")
        return 3
    except (OSError, ValueError, KeyError, RuntimeError, NotImplementedError) as exc:
        print(f"\n[오류] {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
