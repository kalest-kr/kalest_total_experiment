# -*- coding: utf-8 -*-
"""cortex_local_learning_v2_v1_guided_routing.py — cortex_local_learning_v1 위에 V1 안내 라우팅을 더한 판 (v2.0.0).

[무엇이 바뀌었나 — 범위는 하나]
  V4 L2 곡선 뉴런이 받는 두 번째 입력의 '위치' 선택만 바꾼다. 두 번째 입력의 채널 (V4 L4 contour θB), 첫 번째 입력,
  V4 의 다른 역할 채널 (직선 연속·접합 중계·선 끝), V1·V2·IT, L6, L1 국소 q 규칙, prototype 갱신은 v1 과 같다.
  같은 순방향에서 이미 계산한 V1 L3 실제 전송 출력 (휘도 채널) 을 '안내 지도' 로만 읽고, V4 출력에 더하거나 이어 붙이지 않는다.
  미리 만든 고정 후보 사이에서 이미지마다 하나를 고르는 동적 라우팅이며, 후보 목록을 바꾸는 영구 재배선이 아니다.
  (U-Net 에서 빌린 것은 '앞단의 세밀한 공간 정보를 뒤쪽 처리에 다시 제공한다' 는 스킵 연결 착상뿐이다. U-Net 자체는 구조적 재배선을 하지 않는다.)
  V1 안내에 따른 입력별 연결 선택이며, 영구 재배선이나 인간 피질의 재현을 입증한 것은 아니다.

[라우팅 모드 — 설정 routing.mode, 실행 (run) 마다 고정]
  legacy        v1 재현. v1 규칙: P 에서 θA 방향으로 D 만큼 떨어진 점에 가장 가까운 격자 위치 (그 위치가 P 자신이면 같은 위치 P 를 그대로 쓴다).
                곡률 방향 (현 방향) 을 쓰지 않는다. v1 이 만든 checkpoint 는 이 모드로 읽는다.
  geometry_only 수정된 기하 후보 집합 (아래) 의 고정 첫 후보. V1 안내 영향 없음.
  v1_guided     geometry_only 와 정확히 같은 후보 집합에서, 이번 이미지의 V1 L3 경로 지지 점수 S 가 가장 큰 후보 (새 실행 기본값).
  효과는 나눠 본다: legacy→geometry_only = 기하 배선 수정 효과, geometry_only→v1_guided = 안내 지도의 추가 효과.
  프로그램은 사용자가 고른 모델 하나만 학습한다 (세 모드 동시 학습·시드 반복·격자 탐색을 자동으로 하지 않는다).

  후보: Q ≠ P, 거리 [0.5D, 1.5D], 현 방향이 φC = θA + s·κ/2 (연속 각도) 에서 45° 이내 → 거리·방향 정규화 오차 합과 입력 주소 순 정렬
        → cubic Hermite 경로 (vA = |Q−P|·tA, vB = |Q−P|·tB, M = 9 점) 검사 (곡률 부호·접힘·영 접선·수용장·진행 방향) 통과만 최대 K = 8.
        후보가 없으면 같은 위치로 대체하지 않고 u = 0 (missing_candidate). 다른 역할 뉴런은 끄지 않는다.
  점수: r_m = a_m/(b_m + ε) (b_m < guide_min_total 이면 0), S = 평균 r_m. S 는 방향 지지 정도이며 확률이 아니다.
  선택: hard top-1 (같은 점수는 기하 순서). 모든 S ≤ 0 이면 기하 첫 후보 + no_guidance. B = V4 L4[Q*, θB], u = min(A, B). S 를 u 에 곱하지 않는다.
  모든 값은 설정 파일 routing 에 노출된 공학적 초기값이다. 데이터를 보고 몰래 바꾸지 않는다 (바꾸면 9-8 설계 변경 기록).

[중요한 기하 결과 — 작성 환경에서 구조만 만들어 센 값 (학습·데이터와 무관)]
  곡률 부호 검사가 45° 현 방향 허용보다 강하게 작동한다. κ = 30° 에서 현이 φC 의 ±6.75° 안일 때만 Hermite 경로가 한 방향으로 휜다
  (κ 15/45/60° 에서는 ±3.5/±10.5/±15.25°). 그래서 기본 설정 (128 px, V4 4×8) 에서 곡선 뉴런 768 개 중 79.2 % 가 후보 없음 (u = 0) 이다.
  거리·방향 조건만으로는 35.3 %, 나머지는 곡률 부호 검사 탈락이다 (경로 검사 탈락 2,530 건 모두 곡률 부호). 유효 후보 평균 0.46.
  즉 새 두 모드에서는 legacy 와 달리 대부분의 곡선 뉴런이 꺼지고, v1_guided 와 geometry_only 가 다를 수 있는 뉴런은 일부다.
  이것은 V1 안내로 해결되지 않는 격자·기하 문제이며 학습 실패와 구분한다 (routing/missing_by_ring_orientation.csv).

[설치]
  Python 3.10 이상.  필수: numpy, Pillow.  GPU 계산: PyTorch.  선택: matplotlib (그림).
    pip install numpy pillow matplotlib
    PyTorch 는 https://pytorch.org/get-started/locally/ 에서 자기 PC 의 드라이버·CUDA 에 맞는 설치 명령을 골라 설치한다
    (이 파일은 CUDA 버전을 정하지 않는다).
  프로그램은 패키지를 자동 설치하지 않고 외부 서버에 접속하지 않는다. PyTorch 가 없으면 시작할 때 알리고,
  사용자가 명시적으로 고를 때만 같은 수식의 NumPy CPU 경로 (느림) 로 계속한다. GPU 메모리 부족이면 gather 조각을 줄여 같은 배치를 다시 계산하고
  (oom_events.jsonl), CPU 로 몰래 바꾸지 않는다.
  R01 (legacy 가 v1 을 재현하는지) 은 같은 폴더의 cortex_local_learning_v1.py 를 읽어 비교한다 (다른 위치면 routing.legacy_reference_path).

[실행 — Windows / PyCharm: 이 파일을 열고 Run → 한국어 번호 메뉴 (기존 번호 1~8 유지, 새 기능은 9)]
  권장 순서 (새 실행, 기본 v1_guided):
    1 (데이터·저장 폴더) → [필요하면 9 → 2 로 모드 선택] → 2 (데이터 검사·분할) → 3 (V1 생성) → 7 → 1 (기본 검사)
    → 4 (V1 학습) → 5 (V2) → 4 → 5 (V4 생성: 라우팅 후보 표·후보 없음 통계 출력, 학습 전 평가 기록) → 9 → 4 (라우팅 구현 검사)
    → 4 (V4 학습) → 9 → 5 (곡선 역할 진단) / 9 → 6 (안내 개입 진단 = 즉각적 민감도 검사) / 9 → 7 (겹쳐 그린 그림)
    → 5 (IT) → 4 → 7 → 5 (최종 test 평가, 고정 모델에서 한 번) → 6 (추론) → 8 (저장).
  모드 비교 (사용자가 고를 때만): 같은 V1·V2 run 에서 9 → 3 으로 geometry_only·legacy run 을 분기 → 각각 V4·IT 학습 → dev 기록 비교.
  9) V1 안내 라우팅:  9-1 상태·후보 통계  9-2 새 실행 모드 선택  9-3 다른 모드로 분기 (새 run)  9-4 구현 검사 R00~R12
                      9-5 곡선 역할 진단  9-6 안내 개입 진단  9-7 라우팅 그림  9-8 설계 변경 기록
  명령줄:
    python cortex_local_learning_v2_v1_guided_routing.py --self-test --output-dir "D:\\cortex_out"
    python cortex_local_learning_v2_v1_guided_routing.py --data-root "D:\\shapes" --output-dir "D:\\cortex_out" --train-stage V2
    python cortex_local_learning_v2_v1_guided_routing.py --data-root "D:\\shapes" --output-dir "D:\\cortex_out" --routing-mode geometry_only --train-stage V4
    python cortex_local_learning_v2_v1_guided_routing.py --resume "D:\\cortex_out\\run_..." --train-stage V4 --routing-check --routing-diag
    python cortex_local_learning_v2_v1_guided_routing.py --branch-from "D:\\cortex_out\\run_..." --routing-mode v1_guided --train-stage V4
    python cortex_local_learning_v2_v1_guided_routing.py --resume "D:\\cortex_out\\run_..." --infer "D:\\test_images"
  기타: --device auto|cuda|cpu, --backend auto|torch|numpy, --seed N, --config 설정.json, --epochs N, --diagnose, --role-test, --test-eval, --yes

[기존 checkpoint 호환]
  · v1 이 만든 run (config 에 routing 이 없음) 은 legacy 로 읽는다. legacy 의 구조 해시·config 구조 해시가 v1 과 같다 (R01, 자체 검사 ST19).
  · 재개는 모드를 바꾸지 않는다. --resume 에 다른 --routing-mode 를 주면 오류. 모드 변경 = 분기 (9-3, --branch-from):
    원본 run 은 읽기만 하고, V1·V2 의 q·prototype·동결 해시를 확인해 새 run 으로 옮긴 뒤 V4 를 새 모드로 만들고
    V4 prototype 을 train 출력으로 새로 초기화한다 (기존 V4 prototype 을 재사용하지 않는다). IT 는 그 뒤 다시 만들고 학습한다.
  · 새 모드 checkpoint 는 라우팅 모드·설정·후보 표 해시·좌표 규약을 담는다. 다시 불러올 때 표를 다시 만들어 해시를 비교한다 (R09).
  · v2 가 만든 run (legacy 포함) 은 v1 프로그램으로 읽을 수 없다: v1 은 config 의 routing 키를 몰라 ConfigError 로 멈춘다 (작성 환경에서 확인).
    v1 run 을 v2 로 읽거나 분기해도 원본 v1 run 폴더는 바뀌지 않는다 (ST19 에서 폴더 해시 비교).
  · 영역 출력 (V1~IT) 은 캐시하지 않는다. 캐시는 LGN 표본 (전처리 해시 키) 뿐이라 라우팅 변경 전 V4/IT 출력을 재사용할 수 없다.

[설계 가정 — 생물학에서 착안한 공학적 근사이며 인간 시각 피질의 재현이 아니다]
  · 결정론적 rate 모델. u = Σ A·transmitted + external, s = max(0, u − 임계값), CONF = q·s, 전송 출력 = L6 g·CONF.
  · 3×3 표기 [X Y Z / NEXT 임계값 INPUT / 전송 출력 CONF q] 는 논리 표기. 라우팅 곡선 뉴런의 INPUT 은 '정적 A + 동적 B (정적 후보 목록)' 로 따로 보인다.
  · 교정: δ = prototype_정답 − h, m_L3 = α3 δ, m_L2 = α2 R δ (R = 1/|P(j)|), q ← clip(q + η·mean(s/(s+e0)·m)) — 검증되지 않은 국소 휴리스틱.
    라우팅은 L2→L3 부모 목록을 바꾸지 않는다 (R02 로 확인). 역전파·autograd·optimizer·손실 유한차분으로 연결을 고르지 않는다.
  · 분류: argmax cos(h, prototype). 영역은 V1 → V2 → V4 → IT 순서로 추가·학습하고 앞 영역은 동결한다.

[미검증 가설 — 이 프로그램은 확인할 수단을 제공할 뿐 성공을 보장하지 않는다]
  · V1 안내 선택이 곡선 뉴런의 위치·방향 오차를 줄이고 곡선 역할 반응이나 분류를 개선한다 (geometry_only 분기와 비교해야 알 수 있다).
  · 국소 q 교정·L6 조절·순차 동결·prototype EMA 가 분류에 충분하다 (v1 과 같은 가설).

[이 파일을 만든 환경에서 실제 확인한 것 / 확인하지 못한 것]  (자세히: VERIFICATION_NOTES)
  · 확인 (NumPy 경로, 작성 환경에 torch 없음, 극소 합성 입력): py_compile·pyflakes, --self-test 22 항목 실패 0,
    라우팅 구현 검사 R00~R10·R12 PASSED (작은 설정, geometry_only run, 학습 전 기본 크기 구조), R11 NOT_RUN (GPU 없음),
    메뉴 9 입력 재생, CLI 모드 지정·불일치 오류·분기, 기본 크기 후보 통계.
  · 확인하지 못함: PyTorch (CPU·CUDA) 실행과 R11, 사용자 데이터 학습·분류 성능·모드 간 성능 차이, GPU 메모리·시간, Windows/PyCharm 동작.
    정확도·속도 예상치는 적지 않는다.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import dataclasses
import datetime as _dt
import hashlib
import importlib.util
import inspect
import io
import json
import logging
import math
import multiprocessing
import os
import platform
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

# cuBLAS 결정성 설정은 torch 를 불러오기 전에 둔다 (사용자가 이미 정했으면 그대로).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

try:
    from PIL import Image
    _PIL_ERROR: str | None = None
except Exception as _exc:                                   # pragma: no cover - 환경 의존
    Image = None                                            # type: ignore[assignment]
    _PIL_ERROR = f"{type(_exc).__name__}: {_exc}"

try:
    import torch
    _TORCH_ERROR: str | None = None
except Exception as _exc:                                   # pragma: no cover - 환경 의존
    torch = None                                            # type: ignore[assignment]
    _TORCH_ERROR = f"{type(_exc).__name__}: {_exc}"

VERSION = "2.0.0"
PROGRAM = "cortex_local_learning_v2_v1_guided_routing"
CKPT_FORMAT = "cortex_local_learning_v2/checkpoint/1"
CKPT_FORMATS_READABLE = ("cortex_local_learning_v1/checkpoint/1", "cortex_local_learning_v2/checkpoint/1")
AREA_ORDER: tuple[str, ...] = ("V1", "V2", "V4", "IT")
AREA_INDEX = {a: i + 1 for i, a in enumerate(AREA_ORDER)}
LAYER_ORDER: tuple[str, ...] = ("L4", "L2", "L3")
LAYER_NUMBER = {"L4": 4, "L2": 2, "L3": 3}

PASSED, FAILED, MEASURED, NOT_RUN, NOT_APPLICABLE = "PASSED", "FAILED", "MEASURED", "NOT_RUN", "NOT_APPLICABLE"
STAGE_TERMINAL = ("success_criterion_met", "plateau", "budget_exhausted")

VERIFICATION_NOTES: dict[str, list[str]] = {
    "verified_in_authoring_environment_numpy_backend": [
        "py_compile 문법 검사, pyflakes 정적 검사 (경고 0)",
        "--self-test 22 항목 실패 0 (20 PASSED + ST11·ST17 MEASURED): v1 의 15 항목 (데이터 검사·분할, 전처리 DC, 구조, 3×3 조회, "
        "기본 진단, V1~IT 학습, 중단·재개 동일성, 배치 중간 중단 되돌림, 추론, 교란, test 평가 기록, 다시 불러오기, 경로 처리) + "
        "ST16 라우팅 구현 검사, ST17 곡선 역할·안내 개입 진단, ST18 legacy run → v1_guided 분기 (V1·V2 q·prototype 동일, 원본 폴더 불변), "
        "ST19 v1 프로그램이 만든 run 을 legacy 로 읽기·분기 (q·구조 해시·V2 출력 동일), ST20 메모리 부족 주입 → 줄인 조각으로 재시도 "
        "(q 가 주입 없는 실행과 비트 단위 동일, 뒷단 변경 없음), ST21 라우팅 기록 파일, ST22 geometry_only run 의 구현 검사",
        "라우팅 구현 검사 R00~R10·R12 PASSED, R07 MEASURED, R11 NOT_RUN — 자체 검사 설정 (64 px), geometry_only run, "
        "기본 설정 (128 px, 학습 전 구조) 에서 각각 실행. R01: 원본 v1 파일과 legacy 모드의 V1~IT L3 출력 최대 차 0, 구조·설정 해시 동일",
        "기본 설정 구조 통계 (학습·데이터 무관): 곡선 뉴런 768, 후보 없음 79.2 % (거리·방향 조건만으로 35.3 %), 경로 검사 탈락 2,530 건 "
        "전부 곡률 부호, 유효 후보 평균 0.46. 곡률 부호 검사를 통과하는 현 방향 범위 κ=30° → ±6.75°",
        "한국어 메뉴 입력 재생 (1→9-2→2→3→4→5→4→5→9-1→9-4→9-5→9-7→9-8→4→9-6→9-3→9-1→0), "
        "CLI: --routing-mode 새 실행, --resume 모드 불일치 거부, --branch-from, --routing-check, --routing-diag",
        "v1 프로그램은 v2 run 을 읽지 못한다 (routing 키 ConfigError) — 작성 환경에서 확인",
    ],
    "not_verified": [
        "PyTorch 경로 (CPU·CUDA) 실행과 R11 CPU/GPU 비교 — 작성 환경에 torch 가 없다. 같은 수식의 gather·합·where·argmax·min 만 쓰도록 맞췄지만 실행하지 않았다",
        "사용자 데이터 학습, 분류 성능, 모드 간 (legacy / geometry_only / v1_guided) 성능 차이, 설정 탐색 (요청대로 실행하지 않음)",
        "RTX 4070 의 실제 메모리 사용량·시간, 실제 CUDA 메모리 부족 처리 (주입한 RuntimeError 로만 재시도 경로를 확인)",
        "Windows 콘솔 / PyCharm 에서의 Ctrl+C·한글 경로 (PyCharm 의 Stop 버튼은 프로세스를 바로 끝낼 수 있다 → 마지막 checkpoint 부터 재개)",
    ],
}


# ======================================================================
# 0. 공통 유틸리티
# ======================================================================
class ConfigError(ValueError):
    """설정 오류. 값을 몰래 고쳐 통과시키지 않는다."""


class DataError(RuntimeError):
    """데이터 폴더·이미지 오류."""


class SplitError(DataError):
    """층화 분할이 성립하지 않는다 (중복 배정하지 않는다)."""


class StructureError(RuntimeError):
    """연결 구조 오류 (부모 없는 L3, 범위 밖 주소, 중복 edge 등)."""


class L1AddressError(RuntimeError):
    """L1 배달 주소 (영역·층·neuron_id·모양) 불일치."""


class CompatibilityError(RuntimeError):
    """재개할 실행과 데이터·설정·구조가 맞지 않는다."""


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_array(*arrays: Any) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(f"{a.dtype.str}|{a.shape}|".encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def json_ready(o: Any) -> Any:
    """JSON 으로 쓸 수 있게 바꾼다. NaN/Inf 는 null (미정의) 로 쓴다."""
    if isinstance(o, dict):
        return {str(k): json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_ready(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return sorted(str(v) for v in o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.ndarray):
        return json_ready(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return json_ready(dataclasses.asdict(o))
    return o


def dumps(o: Any, indent: int | None = 2) -> str:
    return json.dumps(json_ready(o), ensure_ascii=False, indent=indent)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def write_json(path: Path, obj: Any) -> None:
    _atomic_write_bytes(Path(path), dumps(obj).encode("utf-8"))


def write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(Path(path), text.encode("utf-8"))


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_npz(path: Path, **arrays: Any) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: np.asarray(v) for k, v in arrays.items()})
    _atomic_write_bytes(Path(path), buf.getvalue())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def clean_user_path(raw: Any) -> Path:
    """탐색기에서 복사한 경로: 앞뒤 공백·따옴표, 보이지 않는 방향 표시 문자를 지우고 ~ 와 환경변수를 푼다."""
    s = str(raw).strip().strip("\u202a\u202b\u202c\u202d\u202e\ufeff").strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    if not s:
        raise ValueError("빈 경로다.")
    s = os.path.expandvars(os.path.expanduser(s))
    return Path(s)


def ensure_writable_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    if not p.is_dir():
        raise NotADirectoryError(f"폴더가 아니다: {p}")
    probe = p / f".write_probe_{os.getpid()}_{time.time_ns()}"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        raise PermissionError(f"쓰기 불가: {p} ({exc})") from exc
    return p.resolve()


def unique_run_dir(root: Path, prefix: str = "run") -> Path:
    """기존 결과를 덮어쓰지 않는 새 실행 폴더."""
    root = ensure_writable_dir(root)
    stamp = local_stamp()
    for k in range(10000):
        d = root / (f"{prefix}_{stamp}_{os.getpid()}" + (f"_{k}" if k else ""))
        try:
            d.mkdir()
            return d
        except FileExistsError:
            continue
    raise RuntimeError(f"새 실행 폴더를 만들 수 없다: {root}")


def deep_merge(base: dict[str, Any], over: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "NA"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "NA"
    if f != 0.0 and abs(f) < 0.5 * 10.0 ** (-nd):          # 표시 자릿수보다 작은 값은 0 으로 보이지 않게 지수 표기
        return f"{f:.2e}"
    return f"{f:.{nd}f}"


def pct_count(correct: int | None, total: int | None) -> str:
    if correct is None or not total:
        return "NA"
    return f"{100.0 * correct / total:.2f}% ({int(correct)}/{int(total)})"


# ======================================================================
# 1. 설정 (Config) — 모든 기본값을 한 곳에서 관리한다
# ======================================================================
DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 20251011,
    "data": {
        "image_size": 128,                       # 처리 크기 S (정사각). 긴 변을 S 에 맞추고 짧은 변은 가장자리 복제로 채운다
        "extensions": [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"],
        "split_ratios": {"train": 0.70, "dev": 0.15, "test": 0.15},
        "group_suffix_regex": r"(?:[_\-\s]?(?:aug|copy|dup|flip|rot)\d*|\s?\(\d+\))+$",
        "transparent_background": 1.0,           # 투명 픽셀을 합성할 배경 밝기 (0..1)
        "cache_preprocessed": True,              # LGN 표본 벡터를 RAM (CPU) 에 캐시 (GPU 에 데이터셋 전체를 올리지 않는다)
        "cache_max_mb": 1024,
    },
    "retina": {
        "sigma_center_px": 1.0,                  # DoG 중심 Gaussian σ (처리 크기 화소)
        "sigma_surround_px": 3.0,                # DoG 주변 Gaussian σ
        "truncate_sigma": 3.0,                   # 이산 커널 반경 = ceil(truncate·σ) (각 커널 합 1 로 정규화)
        "gain": 1.0,                             # DoG 출력 배율 (무차원)
    },
    "logpolar": {
        "n_r": 32, "n_phi": 64,                  # LGN 표본 격자 (반경 bin × 각도 bin)
        "r0_px": 2.0,                            # r(u) = r0·(exp(u·log(1+Rmax/r0)) − 1)
        "rmax_margin_px": 2.0,                   # Rmax = S/2 − margin (유효 영상 안)
        "phi_offset_bins": 0.5,                  # 각도 bin 중심 오프셋 (0 rad 이음매에 표본을 두지 않는다)
        "blur_beta": 0.5,                        # 주변부 저역통과 σ_req = β · 국소 표본 간격
        "blur_levels_px": [0.0, 1.0, 2.0, 4.0, 8.0],
    },
    "neuron": {"threshold": 0.0, "q_init": 1.0, "q_min": 0.0, "q_max": 4.0},
    "areas": {
        "V1": {"n_r": 16, "n_phi": 32,
               "orientations_deg": [15.0 * k for k in range(12)],     # 0..165 도, 원영상 좌표 기준 (반시계)
               "phases_rad": [0.0, -math.pi / 2.0],
               "polarities": [1, -1],                                 # 부호 있는 필터 응답의 양·음 극성을 별도 뉴런으로
               "color_channels": ["lum"],                             # lum (=R+G) / rg / by. 색 배수는 따로 보고한다
               "gabor_sigma_factor": 1.0, "gabor_sigma_min_px": 1.5,
               "gabor_wavelength_factor": 2.0, "rf_radius_sigma": 2.0},
        "V2": {"n_r": 8, "n_phi": 16, "offset_factor": 1.0, "endstop_inhibition": 1.0, "cross_diffs_deg": [90.0, 60.0]},
        "V4": {"n_r": 4, "n_phi": 8, "offset_factor": 1.0, "curvature_deg": [30.0], "l3_grid_radius": 1},
        "IT": {"region_r": 2, "region_phi": 4},
    },
    "l6": {"enabled": True, "layers": ["L2", "L3"], "epsilon": 1e-6, "kappa": 0.25, "g_min": 0.5, "g_max": 2.0,
           "neighbor_grid_radius": 1, "include_self": False},
    "learning": {"alpha2": 0.5, "alpha3": 0.5, "eta": 0.01, "e0": 1.0, "rho": 0.05, "batch_size": 16,
                 "max_epochs": 20, "patience": 5, "min_delta": 0.0, "success_dev_accuracy": None,
                 "change_tol": 1e-7, "eps_norm": 1e-12},
    "records": {"correction_samples_per_epoch": 64, "correction_samples_max_per_stage": 512,
                "checkpoint_every_batches": 50, "keep_periodic_checkpoints": 3, "store_train_h_max_mb": 1024,
                "diagnostic_samples": 4},
    "compute": {"dtype": "float32", "max_gather_elements": 1 << 25, "deterministic": True, "gpu_timing_sync": True},
    # V1 안내 라우팅 (V4 L2 곡선 뉴런의 두 번째 입력 위치 선택). 값은 공학적 초기값이며 데이터를 보고 몰래 바꾸지 않는다.
    "routing": {
        "mode": "v1_guided",                     # legacy | geometry_only | v1_guided (새 실행 기본 v1_guided; v1 기록은 legacy 로 읽는다)
        "target": "V4_L2_curve_second_input",
        "guide_source": "V1/L3",                 # 이번 이미지의 V1 L3 실제 전송 출력 (같은 순방향에서 재사용)
        "guide_color": "lum",
        "max_candidates": 8,                     # 후보 수 상한 K
        "dist_min_factor": 0.5, "dist_max_factor": 1.5,   # 거리 범위 [0.5D, 1.5D] (D = 기존 국소 간격)
        "max_chord_dev_deg": 45.0,               # 기대 현 방향에 대한 허용 편차
        "path_points": 9,                        # 경로 지점 수 M
        "curvature_turn_tol_deg": 0.5,           # 이웃 경로 지점 사이 반대 방향 회전 허용 (수치 여유)
        "guide_epsilon": 1e-6,                   # r_m = a_m / (b_m + ε)
        "guide_min_total": 1e-3,                 # b_m 유효 기준 (V1 L3 휘도 방향 채널 합, 무차원). 미만이면 r_m = 0
        "high_guide_score": 0.5,                 # 기록용: 안내 점수가 높은데 선택 입력 B = 0 인 사례
        "samples_per_epoch": 16, "samples_max_per_stage": 128,   # routing_samples.npz 상한
        "oom_retry_halvings": 3,                 # OOM 시 같은 배치를 gather 조각 크기만 반씩 줄여 다시 (배치 크기는 바꾸지 않음)
        "legacy_reference_path": "",             # 검사 R01: 원본 v1 파일 경로 (비우면 이 파일 옆 cortex_local_learning_v1.py)
    },
}
ROUTING_RECORD_ONLY_KEYS = ("high_guide_score", "samples_per_epoch", "samples_max_per_stage", "oom_retry_halvings", "legacy_reference_path")

# 재개 때 바꿔도 되는 학습 설정 (예산·종료 기준). 나머지는 같아야 같은 실행으로 이어 붙인다.
RESUME_MUTABLE_LEARNING = ("max_epochs", "patience", "min_delta", "success_dev_accuracy")


def validate_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    c = deep_merge(DEFAULT_CONFIG, cfg or {})
    unknown = sorted(set(c) - set(DEFAULT_CONFIG))
    if unknown:
        raise ConfigError(f"알 수 없는 설정 키: {unknown}")

    def pos_int(v: Any, what: str, lo: int = 1) -> int:
        if isinstance(v, bool) or int(v) != float(v) or int(v) < lo:
            raise ConfigError(f"{what} 는 {lo} 이상 정수다: {v!r}")
        return int(v)

    def num(v: Any, what: str, lo: float | None = None, hi: float | None = None, lo_open: bool = False) -> float:
        f = float(v)
        if not math.isfinite(f) or (lo is not None and (f <= lo if lo_open else f < lo)) or (hi is not None and f > hi):
            raise ConfigError(f"{what} 범위 오류: {v!r}")
        return f

    d = c["data"]
    pos_int(d["image_size"], "data.image_size", 16)
    r = d["split_ratios"]
    if set(r) != {"train", "dev", "test"} or any(float(v) <= 0 for v in r.values()) or abs(sum(map(float, r.values())) - 1.0) > 1e-9:
        raise ConfigError("data.split_ratios 는 train/dev/test 가 모두 양수이고 합이 1 이다.")
    re.compile(str(d["group_suffix_regex"]))
    num(d["transparent_background"], "data.transparent_background", 0.0, 1.0)
    rt = c["retina"]
    num(rt["sigma_center_px"], "retina.sigma_center_px", 0.0, lo_open=True)
    if float(rt["sigma_surround_px"]) <= float(rt["sigma_center_px"]):
        raise ConfigError("retina.sigma_surround_px 는 sigma_center_px 보다 커야 한다.")
    num(rt["truncate_sigma"], "retina.truncate_sigma", 1.0)
    num(rt["gain"], "retina.gain", 0.0, lo_open=True)
    lp = c["logpolar"]
    pos_int(lp["n_r"], "logpolar.n_r", 2)
    pos_int(lp["n_phi"], "logpolar.n_phi", 2)
    num(lp["r0_px"], "logpolar.r0_px", 0.0, lo_open=True)
    num(lp["rmax_margin_px"], "logpolar.rmax_margin_px", 0.0)
    num(lp["phi_offset_bins"], "logpolar.phi_offset_bins", 0.0, 1.0)
    num(lp["blur_beta"], "logpolar.blur_beta", 0.0)
    if not lp["blur_levels_px"] or any(float(x) < 0 for x in lp["blur_levels_px"]):
        raise ConfigError("logpolar.blur_levels_px 는 0 이상 값의 목록이다.")
    S = int(d["image_size"])
    if S / 2.0 - float(lp["rmax_margin_px"]) <= float(lp["r0_px"]):
        raise ConfigError("Rmax = S/2 − margin 이 r0 보다 커야 한다.")
    ne = c["neuron"]
    if not (0.0 <= float(ne["q_min"]) <= float(ne["q_init"]) <= float(ne["q_max"])):
        raise ConfigError("neuron: 0 <= q_min <= q_init <= q_max 이어야 한다.")
    num(ne["threshold"], "neuron.threshold")
    a = c["areas"]
    v1 = a["V1"]
    pos_int(v1["n_r"], "areas.V1.n_r")
    pos_int(v1["n_phi"], "areas.V1.n_phi")
    if not v1["orientations_deg"] or not v1["phases_rad"] or not v1["polarities"]:
        raise ConfigError("areas.V1 방향·위상·극성 목록이 비어 있다.")
    if any(int(p) not in (1, -1) for p in v1["polarities"]) or len(set(int(p) for p in v1["polarities"])) != len(v1["polarities"]):
        raise ConfigError("areas.V1.polarities 는 1 과 -1 중에서 중복 없이 고른다.")
    if not v1["color_channels"] or any(ch not in ("lum", "rg", "by") for ch in v1["color_channels"]):
        raise ConfigError("areas.V1.color_channels 는 lum/rg/by 중에서 고른다.")
    for k in ("gabor_sigma_factor", "gabor_sigma_min_px", "gabor_wavelength_factor", "rf_radius_sigma"):
        num(v1[k], f"areas.V1.{k}", 0.0, lo_open=True)
    for name, prev in (("V2", "V1"), ("V4", "V2")):
        g = a[name]
        pos_int(g["n_r"], f"areas.{name}.n_r")
        pos_int(g["n_phi"], f"areas.{name}.n_phi")
        if int(a[prev]["n_r"]) % int(g["n_r"]) or int(a[prev]["n_phi"]) % int(g["n_phi"]):
            raise ConfigError(f"areas.{prev} 격자 ({a[prev]['n_r']}×{a[prev]['n_phi']}) 가 {name} 격자 "
                              f"({g['n_r']}×{g['n_phi']}) 의 정수배여야 한다 (블록 풀링).")
        num(g["offset_factor"], f"areas.{name}.offset_factor", 0.0, lo_open=True)
    num(a["V2"]["endstop_inhibition"], "areas.V2.endstop_inhibition", 0.0)
    pos_int(a["V4"]["l3_grid_radius"], "areas.V4.l3_grid_radius", 0)
    it = a["IT"]
    pos_int(it["region_r"], "areas.IT.region_r")
    pos_int(it["region_phi"], "areas.IT.region_phi")
    if int(a["V4"]["n_r"]) % int(it["region_r"]) or int(a["V4"]["n_phi"]) % int(it["region_phi"]):
        raise ConfigError("areas.V4 격자가 IT 영역 격자 (region_r × region_phi) 의 정수배여야 한다.")
    l6 = c["l6"]
    if any(x not in ("L2", "L3") for x in l6["layers"]):
        raise ConfigError("l6.layers 는 L2/L3 중에서 고른다 (L6 조절은 L2/L3 에만 적용).")
    num(l6["epsilon"], "l6.epsilon", 0.0, lo_open=True)
    num(l6["kappa"], "l6.kappa", 0.0)
    if not (0.0 < float(l6["g_min"]) <= 1.0 <= float(l6["g_max"])):
        raise ConfigError("l6: 0 < g_min <= 1 <= g_max 이어야 한다.")
    pos_int(l6["neighbor_grid_radius"], "l6.neighbor_grid_radius", 0)
    le = c["learning"]
    for k in ("alpha2", "alpha3", "eta"):
        num(le[k], f"learning.{k}", 0.0)
    num(le["e0"], "learning.e0", 0.0, lo_open=True)
    num(le["rho"], "learning.rho", 0.0, 1.0)
    pos_int(le["batch_size"], "learning.batch_size")
    pos_int(le["max_epochs"], "learning.max_epochs", 0)
    pos_int(le["patience"], "learning.patience")
    num(le["min_delta"], "learning.min_delta", 0.0)
    if le["success_dev_accuracy"] is not None:
        num(le["success_dev_accuracy"], "learning.success_dev_accuracy", 0.0, 1.0)
    num(le["change_tol"], "learning.change_tol", 0.0)
    num(le["eps_norm"], "learning.eps_norm", 0.0, lo_open=True)
    rc = c["records"]
    for k in ("correction_samples_per_epoch", "correction_samples_max_per_stage", "checkpoint_every_batches",
              "keep_periodic_checkpoints", "store_train_h_max_mb"):
        pos_int(rc[k], f"records.{k}", 0)
    pos_int(rc["diagnostic_samples"], "records.diagnostic_samples", 2)
    cp = c["compute"]
    if cp["dtype"] != "float32":
        raise ConfigError("compute.dtype 는 float32 로 시작한다 (혼합 정밀도는 기본으로 쓰지 않는다).")
    pos_int(cp["max_gather_elements"], "compute.max_gather_elements", 1024)
    ro = c["routing"]
    if ro["mode"] not in ("legacy", "geometry_only", "v1_guided"):
        raise ConfigError("routing.mode 는 legacy / geometry_only / v1_guided 중 하나다.")
    if ro["target"] != "V4_L2_curve_second_input" or ro["guide_source"] != "V1/L3" or ro["guide_color"] != "lum":
        raise ConfigError("이번 버전의 라우팅 대상은 V4 L2 곡선 두 번째 입력, 안내는 V1 L3 휘도 채널만 지원한다.")
    pos_int(ro["max_candidates"], "routing.max_candidates")
    pos_int(ro["path_points"], "routing.path_points", 3)
    if not (0.0 < float(ro["dist_min_factor"]) < float(ro["dist_max_factor"])):
        raise ConfigError("routing: 0 < dist_min_factor < dist_max_factor 이어야 한다.")
    num(ro["max_chord_dev_deg"], "routing.max_chord_dev_deg", 0.0, 180.0, lo_open=True)
    num(ro["curvature_turn_tol_deg"], "routing.curvature_turn_tol_deg", 0.0)
    num(ro["guide_epsilon"], "routing.guide_epsilon", 0.0, lo_open=True)
    num(ro["guide_min_total"], "routing.guide_min_total", 0.0)
    num(ro["high_guide_score"], "routing.high_guide_score", 0.0, 1.0)
    for k in ("samples_per_epoch", "samples_max_per_stage", "oom_retry_halvings"):
        pos_int(ro[k], f"routing.{k}", 0)
    return c


def config_structure_sha(cfg: dict[str, Any]) -> str:
    part = {"image_size": cfg["data"]["image_size"], "transparent_background": cfg["data"]["transparent_background"],
            "retina": cfg["retina"], "logpolar": cfg["logpolar"], "neuron": cfg["neuron"], "areas": cfg["areas"], "l6": cfg["l6"],
            "dtype": cfg["compute"]["dtype"]}
    ro = cfg.get("routing") or {}
    if ro.get("mode", "legacy") != "legacy":            # legacy 는 v1 과 같은 해시 (v1 checkpoint 를 그대로 읽는다)
        part["routing"] = {k: v for k, v in ro.items() if k not in ROUTING_RECORD_ONLY_KEYS}
    return sha256_text(dumps(part, indent=None))


def saved_config_with_mode(raw: dict[str, Any]) -> dict[str, Any]:
    """저장된 실행의 설정. routing 기록이 없는 실행 (v1 프로그램이 만든 것) 은 legacy 로 본다 (새 기본값을 덮어쓰지 않는다)."""
    raw = copy.deepcopy(raw)
    if "routing" not in raw:
        raw["routing"] = {"mode": "legacy"}
    return validate_config(raw)


def config_learning_sha(cfg: dict[str, Any]) -> str:
    le = {k: v for k, v in cfg["learning"].items() if k not in RESUME_MUTABLE_LEARNING}
    return sha256_text(dumps({"learning": le, "seed": cfg["seed"], "split": cfg["data"]["split_ratios"],
                              "group_regex": cfg["data"]["group_suffix_regex"]}, indent=None))


class Config:
    """설정 관리 (한 곳): DEFAULT_CONFIG 위에 사용자 JSON 을 덮고 validate 로 검사한다. 실제 적용값과 해시는 config.json 에 기록된다."""

    DEFAULT = DEFAULT_CONFIG
    validate = staticmethod(validate_config)
    structure_sha = staticmethod(config_structure_sha)
    learning_sha = staticmethod(config_learning_sha)

    @staticmethod
    def load(path: str | None, seed: int | None) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        if path:
            cfg = read_json(clean_user_path(path))
            cfg = cfg.get("config", cfg)                     # 실행 폴더의 config.json 도 그대로 받는다
        if seed is not None:
            cfg = deep_merge(cfg, {"seed": int(seed)})
        return validate_config(cfg)


# 수식 기록 (config.json 과 architecture.json 에 그대로 남긴다)
FORMULAS: dict[str, str] = {
    "neuron": "u_j = Σ_i A_ji·transmitted_i + external_j ;  s_j = max(0, u_j − threshold_j) ;  CONF_j = q_j·s_j ;  "
              "transmitted_j = attention_gain_j·CONF_j  (attention_gain = L6 g, L4 는 1)",
    "role_op_min2": "역할 연산 min2 (교차·곡선·결합 뉴런): u_j = min(Σ_{i∈A_j} A_ji x_i, Σ_{i∈B_j} A_ji x_i)  (두 입력군이 모두 있어야 반응)",
    "l6": "같은 층 CONF 스냅샷에서 병렬:  r_j = CONF_j/(mean_{k∈N(j)} CONF_k + ε) ;  g_j = clip(1 + κ(r_j − 1), g_min, g_max) ;  "
          "transmitted_j = g_j·CONF_j ;  이웃 없음 → g=1",
    "prototype_init": "prototype_c = mean_{train, label=c} h  (모델 고정, 라벨은 출력 묶음에만)",
    "prototype_update": "에폭 끝, q 고정, train 재순방향:  prototype_c ← (1−ρ)·prototype_c + ρ·mean_{train,c} h",
    "classify": "pred = argmax_c cos(h, prototype_c) ; ‖h‖ ≤ eps_norm → unknown (실패로 집계)",
    "delta": "δ = prototype_true − h  (B × n_L3, L3 neuron_id 순서)",
    "router": "m_L3 = α3·δ ;  m_L2[i] = α2·Σ_{j: i∈P(j)} δ_j/|P(j)|  (R_ij = 1/|P(j)|, P(j) = 실제 L2→L3 부모 목록; 가중치·전치·미분 미사용)",
    "local_update": "e_j = s_j/(s_j + e0) ;  Δq_j = η·mean_batch(e_j·m_j) ;  q_j ← clip(q_j + Δq_j, q_min, q_max)  "
                    "(L2·L3 갱신량을 같은 스냅샷에서 모두 계산한 뒤 동시에 적용)",
    "cycle": "(1) 동적 상태 초기화 (2) 자유 순방향 → h_before, s, CONF, g (3) 동결 prototype 비교 (4) L1 배달 (5) 활성 영역 q 갱신 "
             "(6) 같은 입력 재순방향 → h_after (7) 같은 이전 prototype 기준 개선·악화·무변화 기록",
    "routing_directions": "곡선 뉴런 (첫 입력 위치 P, 첫 방향 θA, 곡률 부호 s, 꺾임 κ):  φA = θA,  φB = θA + s·κ (θB = φB mod 180),  "
                          "φC = φA + s·κ/2  (연속 각도로 계산, 0°/180° 를 단순 평균하지 않는다)",
    "routing_candidates": "Q ≠ P 인 V4 L4 위치 중  dist_min·D ≤ |Q−P| ≤ dist_max·D  그리고  ∠(Q−P, φC) ≤ max_chord_dev  "
                          "→ 기하 점수 |d−D|/(반폭·D) + 방향오차/max_dev 와 입력 주소 순서로 정렬 → 경로 검사 통과만 K 개.  "
                          "후보 없음 → u = 0 (missing_candidate), 같은 위치로 대체하지 않는다",
    "routing_path": "삼차 Hermite  p(t) = h00 P + h10 L·dir(φA) + h01 Q + h11 L·dir(φB),  L = |Q−P|,  t_m = m/(M−1).  "
                    "곡률 부호가 s 와 반대 (허용 curvature_turn_tol)·접힘·영 접선·수용장 밖·진행 방향 충돌이면 그 후보는 무효",
    "routing_score": "a_m = F1(p(t_m), 접선 선방향) (로그-극좌표 역보간 + 180 주기 방향 보간),  b_m = 같은 위치의 모든 방향 반응 합,  "
                     "r_m = a_m/(b_m + ε)  (b_m < b_min 이면 0),  S = (1/M) Σ r_m  — 방향 지지 정도, 확률이 아니다",
    "routing_select": "v1_guided: Q* = argmax_{유효 후보} S (같은 점수는 기하 순서), 모든 S ≤ 0 → 기하 첫 후보 + no_guidance.  "
                      "geometry_only: 기하 첫 후보.  legacy: v1 규칙.  B = transmitted_V4L4[Q*, θB],  u = min(A, B)  — S 를 곱하지 않는다",
}

ASSUMPTIONS: dict[str, list[str]] = {
    "생물학적 착안 (확정된 생물학이 아님)": [
        "망막/LGN 의 중심-주변 대립 반응과 ON/OFF 경로, 시선 중심에서 멀수록 거칠어지는 표본 (로그-극좌표).",
        "V1 의 방향·위상 선택 단순 세포 근사와 위상 풀링, V2 의 윤곽·끝점·교차, V4 의 곡선 조각, IT 의 넓은 결합.",
        "층 경로 L4 → L2 → L3 → 다음 영역 L4, L6 의 국소 이득 조절, L1 의 위에서 오는 신호 수신 구획.",
    ],
    "계산 가정": [
        "결정론적 rate 모델. 활동·출력은 무차원 수치이고 생물학적 시간·전하·전도도를 나타내지 않는다.",
        "CONF 는 L6 조절 전 기본 출력 (q·s), 실제 전송 출력은 g·CONF. 둘 다 표본마다 새로 계산하는 동적 상태다.",
        "q (3×3 의 마지막 칸) 는 다음 입력에도 남는 지속 출력 파라미터이며 학습은 q 에만 적용한다. 임계값은 고정이다.",
        "L4 는 감각 입력 수신·고정 중계 (q 학습 없음, L6 없음). L1 교정 대상은 활성 영역의 L2·L3 뿐이다.",
        "L6 는 같은 층 공간 이웃 상대 활동 비교만 한다 (방향 선택적 정규화 풀·적응 없음).",
        "L1 배달은 연결 목록 기반의 검증되지 않은 국소 교정 휴리스틱이다. 부호·비선형·상쇄를 정확히 역산하지 않는다.",
        "R 은 L2→L3 부모 목록 개수로만 만든다. L3 풀링 가중치가 평균 (1/|P|) 이라 수치가 같아 보이지만 가중치를 읽지 않는다.",
        "한 영역만 활성 (q 학습·prototype 갱신). 앞 영역은 동결하되 출력과 L6 g 는 매 입력 다시 계산한다.",
    ],
    "명세가 지정하지 않아 고른 구현 선택 (ASSUMPTIONS 로그)": [
        "이미지 크기 맞춤: 긴 변을 S 에 맞춰 Lanczos 축소/확대, 짧은 변은 가운데 정렬 후 가장자리 화소 복제 패딩. 투명 화소는 흰 배경 위 합성. 색 범위 [0,1].",
        "대립색: lum = R+G, rg = R−G, by = B−(R+G)/2 (디지털 RGB 계산 근사, LMS 변환이 아님).",
        "DoG: σc=1, σs=3 화소, 각 커널 합 1, reflect 경계. ON = max(0, DoG), OFF = max(0, −DoG).",
        "로그-극좌표: 원점에는 표본이 없고 각도 bin 중심을 반 칸 옮긴다. 각도 이음매는 이웃 계산에서 wrap 한다.",
        "주변부 저역통과: 국소 표본 간격 × β 에 가장 가까운 Gaussian 피라미드 단계에서 쌍선형 보간으로 표본을 뽑는다.",
        "방향 기준은 원영상 좌표 (x 오른쪽, y 위쪽, 반시계 각도). V1 Gabor 가중치는 LGN 표본의 원영상 좌표에서 계산한다 "
        "(로그-극좌표 격자 위에 Cartesian 필터를 씌우지 않는다).",
        "V1 수용장 σ = factor × 국소 V1 간격 (반경·각도 간격의 기하 평균, 하한 1.5 화소), 파장 = 2σ, 반경 2σ. "
        "표본 면적 가중 envelope 평균을 빼 DC 를 0 으로, Σ|g| = 1 로 정규화한다.",
        "V1 극성: 같은 필터의 +/− 를 별도 뉴런으로 둔다 (위상 2 × 극성 2). 색 배수 기본 1 (lum).",
        "V1 L3: 같은 위치·색·방향의 위상×극성 L2 4 개 평균 (부모 4). 완전한 위상 불변성을 주장하지 않는다.",
        "V2 L4: V1 L3 의 2×2 위치 블록과 색을 방향별로 평균. V2 L2: 윤곽 연속 (중심+앞뒤), 끝점 (한쪽 억제), 교차 (min2). V2 L3: 일대일.",
        "V4 L4: V2 L3 블록 평균. V4 L2: 곡선 조각·직선 연속 (min2), 접합 중계, 선 끝 합. V4 L3: 3×3 격자 이웃 평균 (각도 wrap).",
        "IT L4: V4 L3 를 영역 (region_r × region_phi) 으로 평균. IT L2: 전역 채널 평균, 영역별 특징군 평균, 특징군 쌍 결합 (min2). IT L3: 일대일.",
        "변위 이웃 (윤곽·끝점·곡선) 은 원영상 좌표에서 θ 방향으로 한 간격 떨어진 점에 가장 가까운 격자 위치. 자기 자신이면 그 edge 를 두지 않는다 (개수 기록).",
        "L6 이웃: 같은 층 (같은 Z) 에서 피질 격자 Chebyshev 거리 ≤ 1 인 위치의 모든 채널, 자기 자신 제외. IT 층은 수용장이 전 시야라 층 전체가 한 이웃.",
        "중복 edge 정책: 같은 (수신, 입력군, 송신) 은 가중치를 더해 하나로 합친다 (merge_sum). 합친 개수를 기록한다.",
        "무효 (영벡터) prototype 의 클래스 표본은 δ=0 으로 두어 q 교정을 하지 않는다 (개수 기록).",
        "단계 종료 시 마지막 상태를 그대로 쓴다 (best dev 상태로 되돌리지 않는다). best dev 에폭은 기록만 한다.",
        "데이터 로더는 프로세스 안 순차 로딩 (DataLoader num_workers=0 과 같음). 전처리 결과는 RAM 캐시.",
        "결정성: 축약 순서가 고정된 gather·합·clamp 만 쓰고 atomic 누적 (index_add) 을 쓰지 않는다. CUDA 는 TF32 를 끄고 cuBLAS workspace 를 고정한다.",
    ],
    "V1 안내 라우팅 (v2 에서 추가, V4 L2 곡선 뉴런의 둘째 입력만)": [
        "같은 순방향의 V1 L3 실제 전송 출력 (휘도 채널) 을 안내 지도로만 읽고 V4 출력에 더하거나 이어 붙이지 않는다.",
        "고정 후보 사이에서 이미지마다 하나를 고르는 동적 선택이다. 연결 목록 (후보 표) 은 구조 생성 때 정해지고 학습 중 바뀌지 않는다 — 영구 재배선이 아니다.",
        "후보 표는 라벨·prototype·오차를 쓰지 않는다. 선택은 hard top-1 이며 softmax 혼합·후보 합산·S 곱셈을 하지 않는다.",
        "q 교정 규칙은 v1 과 같다. 선택은 gather·마스크 argmax·min 으로만 계산하며 역전파·autograd·optimizer·손실 유한차분을 쓰지 않는다.",
        "V1 채널의 θ 는 선 (접선) 방향이다 (R00 이 통제 선 자극으로 확인). 경로 접선 방향으로 V1 을 읽는다.",
        "V4 정의가 바뀌므로 V4 prototype 은 train 출력으로 새로 초기화하고 IT 는 다시 만든다. 모드 변경은 새 run 으로 분기한다.",
        "곡률 부호 검사는 κ 에 비해 넓은 현 방향 허용 (45°) 보다 강하게 작동한다 (κ=30° 에서 φC±6.75° 안의 현만 한 방향으로 휘는 Hermite 경로가 된다). "
        "그래서 후보 없음 비율이 높다 — 데이터를 보고 완화하지 않았고, 바꾸려면 설계 변경으로 기록한다.",
    ],
    "미검증 가설": [
        "국소 q 교정이 prototype 오차를 줄이거나 분류를 개선한다.",
        "L6 상대 활동 조절이 약한 특징 보존·주의·분류 향상에 도움이 된다.",
        "V1 안내 라우팅이 곡선 뉴런의 경로 증거를 늘리거나 분류를 개선한다 (geometry_only 와의 비교·개입 진단으로만 확인).",
        "고정 역할 연산이 실제 이미지에서 그 역할을 수행한다 (통제 자극 측정으로만 확인).",
        "순차 동결이 상위 영역 학습에 충분한 정보를 남긴다.",
        "prototype EMA 가 collapse 없이 클래스를 구분한다.",
        "코사인 분류는 벡터 전체 크기 변화만으로는 개선되지 않는다. 크기 오차와 방향 오차를 따로 기록하지만 둘을 L2/L3 책임으로 나누지 않는다.",
    ],
    "미구현·채택하지 않은 것": [
        "방향 선택적 억제 풀, 헤브 필터 자율 형성, 발화 피로·적응, 잡음 정밀도 추정, 예측 부호화 전체 회로.",
        "L5/L6 오차 위치 추적·복원기·DTP·KP·weight mirror, 임계값 학습, prototype 각도 강제·벌리기.",
        "3구획 ODE, 확률적 스파이크, NMDA, GABA-B, 시간 지연 시뮬레이터, 혼합 정밀도, 다중 모델 자동 비교 학습.",
        "근접 중복 (다시 찍은 사진 등) 의 완전한 탐지 — 같은 화소 내용과 파일 이름 규칙만 그룹으로 묶는다.",
    ],
}


def assumptions_markdown() -> str:
    lines = [f"# {PROGRAM} 가정·한계 (v{VERSION})", "",
             "이 문서는 프로그램이 자동으로 쓴다. 뉴런 수·배치·층 역할·학습 규칙은 생물학에서 착안한 공학적 근사이며, "
             "인간 시각 피질의 완전한 재현이나 뇌가 아래 수식을 쓴다는 주장이 아니다. "
             "단방향 연결이라는 해부학적 비유와 알고리즘이 역전파를 쓰는지는 별개다.", ""]
    for sec, items in ASSUMPTIONS.items():
        lines += [f"## {sec}", ""] + [f"- {x}" for x in items] + [""]
    lines += ["## 수식", ""] + [f"- **{k}**: {v}" for k, v in FORMULAS.items()] + [""]
    return "\n".join(lines)


# ======================================================================
# 2. 계산 뒷단 (DeviceManager / Backend)
# ======================================================================
class Backend:
    """수치 연산 뒷단.

    kind="torch": 기본. PyTorch 텐서 (CUDA 또는 CPU). torch.set_grad_enabled(False) 로 미분 추적을 끄고 수동 갱신만 한다.
    kind="numpy": PyTorch 가 없을 때 사용자가 명시적으로 고르는 CPU 경로. 같은 수식·같은 연산 순서 (gather → 곱 → 마지막 축 합).
    쓰는 연산은 gather (고정 정수 인덱스), 원소별 곱·합·clamp·min·where 뿐이다. atomic 누적 (index_add) 을 쓰지 않으므로
    같은 입력·같은 파라미터의 순방향은 같은 결과를 낸다 (CUDA 축약도 모양이 같으면 결정적).
    """

    def __init__(self, kind: str, device: str = "cpu", *, deterministic: bool = True, max_gather_elements: int = 1 << 25):
        if kind not in ("torch", "numpy"):
            raise ValueError(f"알 수 없는 backend: {kind}")
        self.kind = kind
        self.max_gather = int(max_gather_elements)
        self.deterministic = bool(deterministic)
        self.notes: list[str] = []
        if kind == "torch":
            if torch is None:
                raise RuntimeError(f"PyTorch 를 불러올 수 없다: {_TORCH_ERROR}")
            torch.set_grad_enabled(False)
            self.dev = torch.device(device)
            if self.dev.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA 장치를 쓸 수 없다 (torch.cuda.is_available() = False).")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if self.deterministic:
                torch.use_deterministic_algorithms(True, warn_only=True)
                self.notes.append("torch.use_deterministic_algorithms(True, warn_only=True)")
            self.device = str(self.dev)
        else:
            self.dev = None
            self.device = "cpu"

    # ---------------------------------------------------------------- 생성·변환
    @property
    def is_cuda(self) -> bool:
        return self.kind == "torch" and self.dev.type == "cuda"

    def f32(self, a: Any) -> Any:
        a = np.array(a, dtype=np.float32, copy=True, order="C")
        return a if self.kind == "numpy" else torch.from_numpy(a).to(self.dev)

    def i64(self, a: Any) -> Any:
        a = np.array(a, dtype=np.int64, copy=True, order="C")
        return a if self.kind == "numpy" else torch.from_numpy(a).to(self.dev)

    def bool_(self, a: Any) -> Any:
        a = np.array(a, dtype=bool, copy=True, order="C")
        return a if self.kind == "numpy" else torch.from_numpy(a).to(self.dev)

    def to_np(self, x: Any) -> np.ndarray:
        if self.kind == "numpy":
            return np.asarray(x)
        return x.detach().to("cpu").numpy()

    def zeros(self, shape: Sequence[int]) -> Any:
        shape = tuple(int(s) for s in shape)
        return np.zeros(shape, np.float32) if self.kind == "numpy" else torch.zeros(shape, dtype=torch.float32, device=self.dev)

    def zeros_i64(self, shape: Sequence[int]) -> Any:
        shape = tuple(int(s) for s in shape)
        return np.zeros(shape, np.int64) if self.kind == "numpy" else torch.zeros(shape, dtype=torch.int64, device=self.dev)

    def max_last(self, x: Any) -> Any:
        return x.max(-1) if self.kind == "numpy" else x.max(dim=-1).values

    def argmax_last(self, x: Any) -> Any:
        return x.argmax(-1) if self.kind == "numpy" else x.argmax(dim=-1)

    def take_last(self, table: Any, k: Any) -> Any:
        """table (n, K), k (B, n) → table[j, k[b, j]] (B, n)."""
        if self.kind == "numpy":
            return np.take_along_axis(np.broadcast_to(table, (k.shape[0],) + tuple(table.shape)), k[..., None], -1)[..., 0]
        return table.unsqueeze(0).expand(int(k.shape[0]), -1, -1).gather(-1, k.unsqueeze(-1)).squeeze(-1)

    def gather_cols(self, x: Any, idx: Any) -> Any:
        """x (B, N), idx (B, m) → x[b, idx[b, i]] (B, m)  (표본마다 다른 주소)."""
        return np.take_along_axis(x, idx, 1) if self.kind == "numpy" else torch.gather(x, 1, idx)

    def ones_like(self, x: Any) -> Any:
        return np.ones_like(x) if self.kind == "numpy" else torch.ones_like(x)

    def clone(self, x: Any) -> Any:
        return x.copy() if self.kind == "numpy" else x.clone()

    def cat1(self, xs: Sequence[Any]) -> Any:
        return np.concatenate(list(xs), axis=1) if self.kind == "numpy" else torch.cat(list(xs), dim=1)

    def cat0(self, xs: Sequence[Any]) -> Any:
        return np.concatenate(list(xs), axis=0) if self.kind == "numpy" else torch.cat(list(xs), dim=0)

    # ---------------------------------------------------------------- 연산
    def ell(self, xe: Any, idx: Any, w: Any) -> Any:
        """xe: (B, n_src+1) (마지막 열 = 0, 채움 주소), idx/w: (r, K) → (B, r) = Σ_k xe[:, idx[r,k]]·w[r,k]."""
        B = int(xe.shape[0])
        r, K = int(idx.shape[0]), int(idx.shape[1])
        rows = max(1, self.max_gather // max(1, B * K))
        if rows >= r:
            return (xe[:, idx] * w).sum(-1)
        parts = [(xe[:, idx[a:a + rows]] * w[a:a + rows]).sum(-1) for a in range(0, r, rows)]
        return self.cat1(parts)

    def pad_zero_col(self, x: Any) -> Any:
        return self.cat1([x, self.zeros((int(x.shape[0]), 1))])

    def take_cols(self, x: Any, idx: Any) -> Any:
        return x[:, idx]

    def relu(self, x: Any) -> Any:
        return np.maximum(x, np.float32(0.0)) if self.kind == "numpy" else torch.clamp(x, min=0.0)

    def clip(self, x: Any, lo: float, hi: float) -> Any:
        if self.kind == "numpy":
            return np.clip(x, np.float32(lo), np.float32(hi))
        return torch.clamp(x, min=float(lo), max=float(hi))

    def minimum(self, a: Any, b: Any) -> Any:
        return np.minimum(a, b) if self.kind == "numpy" else torch.minimum(a, b)

    def where(self, c: Any, a: Any, b: Any) -> Any:
        return np.where(c, a, b) if self.kind == "numpy" else torch.where(c, a, b)

    def repeat_cols(self, x: Any, k: int) -> Any:
        return np.repeat(x, int(k), axis=1) if self.kind == "numpy" else torch.repeat_interleave(x, int(k), dim=1)

    def mean0(self, x: Any) -> Any:
        return x.mean(axis=0) if self.kind == "numpy" else x.mean(dim=0)

    def sum0(self, x: Any) -> Any:
        return x.sum(axis=0) if self.kind == "numpy" else x.sum(dim=0)

    def norm_last(self, x: Any) -> Any:
        return np.sqrt((x * x).sum(-1)) if self.kind == "numpy" else torch.sqrt((x * x).sum(-1))

    def matmul_t(self, a: Any, b: Any) -> Any:
        return a @ b.T

    def rows(self, P: Any, idx: Any) -> Any:
        return P[idx]

    def all_finite(self, x: Any) -> bool:
        return bool(np.isfinite(x).all()) if self.kind == "numpy" else bool(torch.isfinite(x).all().item())

    def assign_(self, dst: Any, src: Any) -> None:
        if self.kind == "numpy":
            dst[...] = src
        else:
            dst.copy_(src)

    def add_row_(self, M: Any, i: int, v: Any) -> None:
        M[i] += v

    def sync(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize(self.dev)

    def requires_grad_any(self, xs: Sequence[Any]) -> bool:
        if self.kind == "numpy":
            return False
        return any(bool(getattr(x, "requires_grad", False)) for x in xs)

    def grad_enabled(self) -> bool:
        return False if self.kind == "numpy" else bool(torch.is_grad_enabled())

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"backend": self.kind, "device": self.device, "dtype": "float32",
                             "max_gather_elements": self.max_gather, "deterministic_requested": self.deterministic,
                             "notes": list(self.notes)}
        if self.kind == "torch":
            d.update(torch_version=torch.__version__, grad_enabled=bool(torch.is_grad_enabled()),
                     matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                     cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                     deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
                     cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
        else:
            d.update(note_ko="NumPy CPU 경로: 사용자가 명시적으로 고른 경우에만 쓰는 같은 수식의 대체 경로 (GPU 미사용).")
        d["reduction_determinism_ko"] = ("순방향·교정·갱신은 고정 인덱스 gather 와 마지막 축 합만 쓴다 (atomic index_add 없음). "
                                         "분류 코사인의 행렬곱은 cuBLAS (TF32 끔, workspace 고정).")
        return d


def is_oom_error(exc: BaseException) -> bool:
    if torch is not None and hasattr(torch.cuda, "OutOfMemoryError") and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, (RuntimeError, MemoryError)) and ("out of memory" in str(exc).lower() or isinstance(exc, MemoryError))


class DeviceManager:
    """장치·라이브러리 확인과 Backend 생성. 설치 실패를 숨기지 않고, 자동 설치하지 않는다."""

    @staticmethod
    def torch_status() -> dict[str, Any]:
        st: dict[str, Any] = {"torch_available": torch is not None, "torch_error": _TORCH_ERROR}
        if torch is not None:
            st["torch_version"] = torch.__version__
            st["cuda_available"] = bool(torch.cuda.is_available())
            st["cuda_version"] = getattr(torch.version, "cuda", None)
            gpus = []
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    g = {"index": i, "name": torch.cuda.get_device_name(i)}
                    try:
                        free, total = torch.cuda.mem_get_info(i)
                        g.update(vram_total_mb=round(total / 2 ** 20, 1), vram_free_mb=round(free / 2 ** 20, 1))
                    except Exception as exc:                 # noqa: BLE001
                        g["vram_error"] = f"{type(exc).__name__}: {exc}"
                    gpus.append(g)
            st["gpus"] = gpus
        return st

    @staticmethod
    def create(backend: str, device: str, cfg: dict[str, Any]) -> Backend:
        """backend: auto|torch|numpy, device: auto|cuda|cpu. auto 에서 torch 가 없으면 예외 (호출자가 사용자에게 묻는다)."""
        cp = cfg["compute"]
        if backend in ("auto", "torch"):
            if torch is None:
                raise RuntimeError(f"PyTorch 를 불러올 수 없다 ({_TORCH_ERROR}). 설치하거나 --backend numpy 를 명시하라.")
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("--device cuda 를 골랐지만 CUDA 를 쓸 수 없다.")
            dev = "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
            return Backend("torch", dev, deterministic=bool(cp["deterministic"]), max_gather_elements=int(cp["max_gather_elements"]))
        if backend == "numpy":
            if device == "cuda":
                raise RuntimeError("NumPy 경로는 CPU 만 쓴다.")
            return Backend("numpy", "cpu", deterministic=True, max_gather_elements=int(cp["max_gather_elements"]))
        raise ValueError(f"알 수 없는 backend: {backend}")

    @staticmethod
    def environment(be: Backend | None) -> dict[str, Any]:
        mpl_ver, mpl_err = None, None
        try:
            import matplotlib
            mpl_ver = matplotlib.__version__
        except Exception as exc:                             # noqa: BLE001
            mpl_err = f"{type(exc).__name__}: {exc}"
        pil_ver = None
        if Image is not None:
            import PIL
            pil_ver = PIL.__version__
        return {"utc": utc_now(), "program": PROGRAM, "program_version": VERSION, "os": platform.platform(),
                "python": sys.version, "executable": sys.executable, "numpy": np.__version__, "pillow": pil_ver,
                "pillow_error": _PIL_ERROR, "matplotlib": mpl_ver, "matplotlib_error": mpl_err,
                **DeviceManager.torch_status(), "backend": (be.describe() if be is not None else None),
                "cpu_count": os.cpu_count(), "dataloader_ko": "프로세스 안 순차 로딩 (num_workers=0 과 같음)"}

    @staticmethod
    def print_environment(env: dict[str, Any], say: Callable[[str], None] = print) -> None:
        say(f"  Python {env['python'].split()[0]} | numpy {env['numpy']} | Pillow {env['pillow']} | matplotlib {env['matplotlib']}")
        if env.get("torch_available"):
            say(f"  PyTorch {env.get('torch_version')} | CUDA 사용 가능 {env.get('cuda_available')} (CUDA {env.get('cuda_version')})")
            for g in env.get("gpus") or []:
                say(f"    GPU {g['index']}: {g['name']} | VRAM 전체 {g.get('vram_total_mb')} MB, 여유 {g.get('vram_free_mb')} MB")
        else:
            say(f"  PyTorch 없음: {env.get('torch_error')}")
        b = env.get("backend") or {}
        if b:
            say(f"  계산 뒷단 {b.get('backend')} | 장치 {b.get('device')} | dtype {b.get('dtype')} | 결정성 요청 {b.get('deterministic_requested')}")


# ======================================================================
# 3. 실행 기록 (RunRecorder), 시간 측정, 안전한 중단
# ======================================================================
def _csv_cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return repr(f) if math.isfinite(f) else ""
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (list, dict, tuple)):
        return dumps(v, indent=None)
    return v


class RunRecorder:
    """한 실행 폴더의 모든 기록. 파일은 임시 파일에 쓴 뒤 원자적으로 바꾼다."""

    SUBDIRS = ("checkpoints", "figures", "prototype_snapshots", "correction_samples", "diagnostics", "inference", "structure", "routing")
    _count = 0

    def __init__(self, run_dir: Path, *, say: Callable[[str], None] = print):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for s in self.SUBDIRS:
            (self.dir / s).mkdir(exist_ok=True)
        RunRecorder._count += 1
        self.log = logging.getLogger(f"{PROGRAM}.run{RunRecorder._count}.{id(self)}")
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        fmt_ = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        self._handlers = []
        for name, level in (("execution.log", logging.INFO), ("errors.log", logging.WARNING)):
            h = logging.FileHandler(self.dir / name, encoding="utf-8")
            h.setLevel(level)
            h.setFormatter(fmt_)
            self.log.addHandler(h)
            self._handlers.append(h)
        self.say = say

    def p(self, rel: str | Path) -> Path:
        return self.dir / rel

    def info(self, msg: str, echo: bool = False) -> None:
        self.log.info(msg)
        if echo:
            self.say(msg)

    def warn(self, msg: str, echo: bool = True) -> None:
        self.log.warning(msg)
        if echo:
            self.say("[경고] " + msg)

    def error(self, msg: str, exc: BaseException | None = None, echo: bool = True) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc is not None else ""
        self.log.error(msg + ("\n" + tb if tb else ""))
        if echo:
            self.say("[오류] " + msg)

    def json(self, rel: str | Path, obj: Any) -> Path:
        path = self.p(rel)
        write_json(path, obj)
        return path

    def text(self, rel: str | Path, s: str) -> Path:
        path = self.p(rel)
        write_text(path, s)
        return path

    def npz(self, rel: str | Path, **arrays: Any) -> Path:
        path = self.p(rel)
        save_npz(path, **arrays)
        return path

    def append_jsonl(self, rel: str | Path, obj: Any) -> None:
        path = self.p(rel)
        with open(path, "a", encoding="utf-8") as f:
            f.write(dumps(obj, indent=None) + "\n")

    def read_csv(self, rel: str | Path) -> list[dict[str, str]]:
        path = self.p(rel)
        if not path.is_file():
            return []
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def append_csv(self, rel: str | Path, row: dict[str, Any]) -> None:
        path = self.p(rel)
        row = {k: _csv_cell(v) for k, v in row.items()}
        if path.is_file():
            with open(path, newline="", encoding="utf-8") as f:
                rd = csv.reader(f)
                header = next(rd, None) or []
            new = [k for k in row if k not in header]
            if new:                                  # 새 열이 생기면 원자적으로 다시 쓴다
                old = self.read_csv(rel)
                cols = header + new
                buf = io.StringIO()
                w = csv.DictWriter(buf, fieldnames=cols)
                w.writeheader()
                for r in old:
                    w.writerow(r)
                w.writerow(row)
                write_text(path, buf.getvalue())
                return
            with open(path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=header).writerow(row)
            return
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
        write_text(path, buf.getvalue())

    def close(self) -> None:
        for h in self._handlers:
            self.log.removeHandler(h)
            h.close()
        self._handlers = []


class PhaseTimer:
    """실행 시간 (벽시계). gpu=True 이면 CUDA 동기화 뒤에 잰다. 시뮬레이션 시간이 아니다."""

    def __init__(self, be: Backend):
        self.be = be
        self.t: dict[str, float] = {}

    @contextlib.contextmanager
    def phase(self, name: str, gpu: bool = False) -> Iterator[None]:
        if gpu:
            self.be.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if gpu:
                self.be.sync()
            self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0

    def take(self) -> dict[str, float]:
        out = {f"time_{k}_s": round(v, 4) for k, v in self.t.items()}
        self.t = {}
        return out


class StopFlag:
    """학습 중 Ctrl+C. 첫 번째: 현재 배치를 마친 안전한 경계에서 멈춤. 두 번째: 즉시 중단 (진행 중 배치의 q 는 되돌린다).
    critical() 안에서는 즉시 중단을 미뤘다가 끝에서 일으킨다 (q 적용·기록·checkpoint 를 반쯤 끝내지 않기 위해)."""

    def __init__(self, say: Callable[[str], None] = print):
        self.say = say
        self.requested = False
        self._old: Any = None
        self._in_critical = 0
        self._pending = False
        self._installed = False

    def __enter__(self) -> "StopFlag":
        if threading.current_thread() is threading.main_thread():
            try:
                self._old = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, self._handler)
                self._installed = True
            except (ValueError, OSError):
                self._installed = False
        return self

    def _handler(self, signum: int, frame: Any) -> None:
        if self.requested:
            if self._in_critical:
                self._pending = True
                return
            raise KeyboardInterrupt
        self.requested = True
        self.say("\n[중단 요청] 현재 배치를 마친 뒤 checkpoint 를 저장하고 멈춘다. 한 번 더 누르면 즉시 중단 (진행 중 배치 취소).")

    @contextlib.contextmanager
    def critical(self) -> Iterator[None]:
        self._in_critical += 1
        try:
            yield
        finally:
            self._in_critical -= 1
            if not self._in_critical and self._pending:
                self._pending = False
                raise KeyboardInterrupt

    def __exit__(self, *exc: Any) -> None:
        if self._installed:
            signal.signal(signal.SIGINT, self._old)


# ======================================================================
# 4. 이미지 입력, 데이터 검사·분할 (DatasetManager)
# ======================================================================
def _require_pil() -> None:
    if Image is None:
        raise RuntimeError(f"Pillow 를 불러올 수 없다: {_PIL_ERROR}")


def load_pil_rgb(path: Path, background: float = 1.0) -> tuple[Any, dict[str, Any]]:
    """이미지를 RGB 로 연다. 투명 화소는 background 밝기 위에 합성한다. 여러 프레임이면 첫 프레임."""
    _require_pil()
    with Image.open(path) as im:
        meta = {"width": int(im.width), "height": int(im.height), "mode": im.mode, "format": im.format,
                "n_frames": int(getattr(im, "n_frames", 1))}
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            v = int(round(float(background) * 255))
            base = Image.new("RGBA", rgba.size, (v, v, v, 255))
            out = Image.alpha_composite(base, rgba).convert("RGB")
            meta["alpha_composited"] = True
        else:
            out = im.convert("RGB")
            meta["alpha_composited"] = False
        out.load()
    return out, meta


class DatasetManager:
    """데이터 폴더: 하위 폴더 이름 = 클래스. 손상·미지원·빈 클래스·중복 이미지를 기록하고, 그룹을 지켜 층화 분할한다."""

    def __init__(self, cfg: dict[str, Any], root: Path):
        self.cfg = cfg
        self.root = Path(root)
        self.report: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []
        self.manifest: dict[str, Any] | None = None
        self.classes: list[str] = []
        self.class_to_index: dict[str, int] = {}
        self._items_by_split: dict[str, list[dict[str, Any]]] = {}

    # ---------------------------------------------------------------- 검사
    def scan(self, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
        _require_pil()
        if not self.root.is_dir():
            raise DataError(f"데이터 폴더가 없다: {self.root}")
        exts = {e.lower() for e in self.cfg["data"]["extensions"]}
        bg = float(self.cfg["data"]["transparent_background"])
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")], key=lambda d: d.name)
        root_files = sorted(p.name for p in self.root.iterdir() if p.is_file())
        recs: list[dict[str, Any]] = []
        corrupt, unsupported, hidden, empty = [], [], [], []
        per_class_files: dict[str, int] = {}
        for cdir in class_dirs:
            files = sorted([p for p in cdir.rglob("*") if p.is_file()], key=lambda p: p.relative_to(self.root).as_posix())
            n_ok = 0
            for p in files:
                rel = p.relative_to(self.root).as_posix()
                if any(part.startswith(".") for part in p.relative_to(cdir).parts):
                    hidden.append(rel)
                    continue
                if p.suffix.lower() not in exts:
                    unsupported.append(rel)
                    continue
                try:
                    with Image.open(p) as im0:
                        im0.verify()
                    im, meta = load_pil_rgb(p, bg)
                    arr = np.asarray(im, dtype=np.uint8)
                    pix = sha256_bytes(f"{arr.shape[1]}x{arr.shape[0]}|".encode() + arr.tobytes())
                except Exception as exc:                     # noqa: BLE001
                    corrupt.append({"relpath": rel, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                recs.append({"relpath": rel, "class": cdir.name, "sha256": sha256_file(p), "pixel_sha256": pix,
                             "width": meta["width"], "height": meta["height"], "mode": meta["mode"], "format": meta["format"],
                             "alpha_composited": meta["alpha_composited"]})
                n_ok += 1
            per_class_files[cdir.name] = n_ok
            if n_ok == 0:
                empty.append(cdir.name)
            if progress:
                progress(f"  {cdir.name}: 유효 이미지 {n_ok}")
        by_pix: dict[str, list[dict[str, Any]]] = {}
        for r in recs:
            by_pix.setdefault(r["pixel_sha256"], []).append(r)
        conflicts = {k: v for k, v in by_pix.items() if len({r["class"] for r in v}) > 1}
        excluded = {r["relpath"] for v in conflicts.values() for r in v}
        kept = [r for r in recs if r["relpath"] not in excluded]
        dup_groups = [[r["relpath"] for r in v] for k, v in by_pix.items() if len(v) > 1 and k not in conflicts]
        self.records = kept
        self.classes = sorted({r["class"] for r in kept})
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}
        ws = np.array([r["width"] for r in kept]) if kept else np.zeros(0)
        hs = np.array([r["height"] for r in kept]) if kept else np.zeros(0)
        self.report = {
            "data_root": str(self.root), "utc": utc_now(), "n_class_folders": len(class_dirs),
            "classes": self.classes, "n_classes": len(self.classes),
            "valid_images_per_class": {c: sum(1 for r in kept if r["class"] == c) for c in self.classes},
            "n_valid_images": len(kept), "corrupt": corrupt, "unsupported": unsupported, "hidden_ignored": hidden,
            "empty_classes_excluded": empty, "root_files_ignored": root_files,
            "exact_duplicate_groups": dup_groups,
            "cross_class_duplicates_excluded": [[r["relpath"] for r in v] for v in conflicts.values()],
            "original_size": {"width_min": int(ws.min()) if ws.size else None, "width_max": int(ws.max()) if ws.size else None,
                              "height_min": int(hs.min()) if hs.size else None, "height_max": int(hs.max()) if hs.size else None,
                              "width_median": float(np.median(ws)) if ws.size else None,
                              "height_median": float(np.median(hs)) if hs.size else None},
            "duplicate_rule_ko": ("같은 화소 내용 (디코드한 RGB 와 크기의 SHA256) 과 파일 이름 규칙 (group_suffix_regex 로 접미사 제거) 으로만 "
                                  "그룹을 만든다. 다시 찍은 사진 같은 근접 중복은 탐지하지 못한다. 서로 다른 클래스에 같은 화소 내용이 있으면 "
                                  "라벨 충돌로 모두 제외한다."),
        }
        if len(self.classes) < 2:
            raise DataError(f"유효 이미지가 있는 클래스가 2 개 미만이다: {self.classes}")
        return self.report

    # ---------------------------------------------------------------- 분할
    def _groups(self) -> list[int]:
        rx = re.compile(str(self.cfg["data"]["group_suffix_regex"]), re.IGNORECASE)
        n = len(self.records)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            a, b = find(i), find(j)
            if a != b:
                parent[max(a, b)] = min(a, b)

        first: dict[tuple[str, ...], int] = {}
        for i, r in enumerate(self.records):
            p = Path(r["relpath"])
            stem = rx.sub("", p.stem.lower()) or p.stem.lower()
            for key in (("pix", r["pixel_sha256"]), ("stem", r["class"], p.parent.as_posix(), stem)):
                if key in first:
                    union(i, first[key])
                else:
                    first[key] = i
        return [find(i) for i in range(n)]

    def split(self, seed: int) -> dict[str, Any]:
        if not self.records:
            raise SplitError("먼저 scan() 으로 데이터를 검사하라.")
        ratios = self.cfg["data"]["split_ratios"]
        gid = self._groups()
        rng = np.random.default_rng([int(seed), 7])
        assign: dict[int, str] = {}
        per_class: dict[str, Any] = {}
        for c in self.classes:
            groups = sorted({gid[i] for i, r in enumerate(self.records) if r["class"] == c},
                            key=lambda g: self.records[g]["relpath"])
            ng = len(groups)
            if ng < 3:
                raise SplitError(f"클래스 '{c}' 의 독립 그룹이 {ng} 개라 train/dev/test 로 나눌 수 없다 (이미지를 중복 배정하지 않는다).")
            n_test = max(1, int(round(ng * float(ratios["test"]))))
            n_dev = max(1, int(round(ng * float(ratios["dev"]))))
            while n_test + n_dev > ng - 1:
                if n_test >= n_dev and n_test > 1:
                    n_test -= 1
                elif n_dev > 1:
                    n_dev -= 1
                else:
                    raise SplitError(f"클래스 '{c}': 그룹 {ng} 개로 분할 불가.")
            order = [groups[k] for k in rng.permutation(ng)]
            for k, g in enumerate(order):
                assign[g] = "test" if k < n_test else ("dev" if k < n_test + n_dev else "train")
            per_class[c] = {"n_groups": ng, "n_test_groups": n_test, "n_dev_groups": n_dev, "n_train_groups": ng - n_test - n_dev}
        items = []
        for i, r in enumerate(self.records):
            items.append({"index": i, "relpath": r["relpath"], "class": r["class"], "label": self.class_to_index[r["class"]],
                          "group": f"g{gid[i]:06d}", "split": assign[gid[i]], "sha256": r["sha256"],
                          "pixel_sha256": r["pixel_sha256"], "width": r["width"], "height": r["height"]})
        for c in self.classes:
            for sp in ("train", "dev", "test"):
                per_class[c][f"n_{sp}_images"] = sum(1 for it in items if it["class"] == c and it["split"] == sp)
        self.manifest = self._finish_manifest(items, seed, per_class)
        return self.manifest

    def _finish_manifest(self, items: list[dict[str, Any]], seed: int, per_class: dict[str, Any]) -> dict[str, Any]:
        fp = sha256_text(dumps(sorted((it["relpath"], it["sha256"], it["class"]) for it in items), indent=None))
        ssha = sha256_text(dumps(sorted((it["relpath"], it["split"]) for it in items), indent=None))
        man = {"format": f"{PROGRAM}/split/1", "utc": utc_now(), "data_root": str(self.root), "seed": int(seed),
               "split_ratios": self.cfg["data"]["split_ratios"], "classes": self.classes, "class_to_index": self.class_to_index,
               "n_items": len(items), "per_class": per_class, "data_fingerprint_sha256": fp, "split_sha256": ssha,
               "group_rule_ko": "같은 화소 내용 또는 같은 정규화 파일 이름 (같은 하위 폴더) 은 같은 그룹 → 같은 분할", "items": items}
        self._items_by_split = {sp: [it for it in items if it["split"] == sp] for sp in ("train", "dev", "test")}
        return man

    @classmethod
    def from_manifest(cls, cfg: dict[str, Any], root: Path, man: dict[str, Any]) -> "DatasetManager":
        dm = cls(cfg, root)
        dm.classes = list(man["classes"])
        dm.class_to_index = {str(k): int(v) for k, v in man["class_to_index"].items()}
        dm.manifest = man
        dm._items_by_split = {sp: [it for it in man["items"] if it["split"] == sp] for sp in ("train", "dev", "test")}
        return dm

    def verify_files(self) -> list[str]:
        """재개 전에 분할 목록의 파일이 그대로인지 확인 (SHA256)."""
        probs = []
        for it in (self.manifest or {}).get("items", []):
            p = self.root / it["relpath"]
            if not p.is_file():
                probs.append(f"없음: {it['relpath']}")
            elif sha256_file(p) != it["sha256"]:
                probs.append(f"내용 바뀜: {it['relpath']}")
            if len(probs) > 20:
                probs.append("... (이하 생략)")
                break
        return probs

    def items(self, split: str) -> list[dict[str, Any]]:
        return list(self._items_by_split.get(split, []))

    def ids_sha(self, split: str) -> str:
        return sha256_text("\n".join(sorted(it["relpath"] for it in self.items(split))))


# ======================================================================
# 5. 망막/LGN 근사와 로그-극좌표 표본 (RetinaLGNPreprocessor)
# ======================================================================
def logpolar_positions(n_r: int, n_phi: int, r0: float, rmax: float, cx: float, cy: float, phi_offset: float) -> dict[str, Any]:
    """r(u) = r0·(exp(u·log(1+Rmax/r0)) − 1), u_k = (k+0.5)/n_r ;  φ_m = 2π(m+offset)/n_phi ;
    x = cx + r cos φ,  y = cy − r sin φ  (화소 좌표: x 오른쪽, y 아래쪽. φ 는 위쪽 기준 반시계).  위치 순서 pos = k·n_phi + m."""
    L = math.log1p(rmax / r0)
    u = (np.arange(n_r) + 0.5) / n_r
    r = r0 * (np.exp(u * L) - 1.0)
    phi = 2.0 * math.pi * (np.arange(n_phi) + phi_offset) / n_phi
    R, PHI = np.meshgrid(r, phi, indexing="ij")
    x = cx + R * np.cos(PHI)
    y = cy - R * np.sin(PHI)
    sr = (R + r0) * L / n_r
    sp = 2.0 * math.pi * R / n_phi
    return {"n_r": n_r, "n_phi": n_phi, "x": x.ravel(), "y": y.ravel(), "r": R.ravel(), "phi": PHI.ravel(),
            "ring": np.repeat(np.arange(n_r), n_phi), "ang": np.tile(np.arange(n_phi), n_r),
            "spacing_r": sr.ravel(), "spacing_phi": sp.ravel(), "L": L, "r0": r0, "rmax": rmax}


def gauss1d(sigma: float, truncate: float, n_max: int) -> np.ndarray:
    if sigma <= 0:
        return np.ones(1, np.float64)
    r = int(min(math.ceil(truncate * sigma), max(n_max - 1, 0)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-x * x / (2.0 * sigma * sigma))
    return k / k.sum()


def _conv1d_reflect(m: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    r = (len(k) - 1) // 2
    if r == 0:
        return m * k[0]
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(m, pad, mode="reflect")
    out = np.zeros_like(m)
    n = m.shape[axis]
    for i, kv in enumerate(k):
        sl = [slice(None), slice(None)]
        sl[axis] = slice(i, i + n)
        out += kv * p[tuple(sl)]
    return out


def blur2d(m: np.ndarray, k: np.ndarray) -> np.ndarray:
    return _conv1d_reflect(_conv1d_reflect(m, k, 0), k, 1)


class RetinaLGNPreprocessor:
    """망막/LGN 근사 (디지털 RGB 계산 근사이며 생리학적 LMS 변환이 아니다).

    입력 RGB [0,1] (H,W,3) → (S,S,3) 크기 맞춤 → 대립색 lum=R+G, rg=R−G, by=B−(R+G)/2
    → DoG = G_σc * x − G_σs * x (각 커널 합 1, reflect 경계) → ON = max(0, DoG), OFF = max(0, −DoG)
    → 이심률별 저역통과 (표본 간격 × β 에 가장 가까운 Gaussian 단계) → 로그-극좌표 표본 (쌍선형 보간).
    출력: (n_pos·n_maps,) float32, index = pos·n_maps + map (위치 우선), pos = ring·n_phi + angle.
    맵은 V1 이 실제로 쓰는 색 채널의 ON/OFF 만 만든다 (기본 lum_on, lum_off).
    """

    def __init__(self, cfg: dict[str, Any]):
        d, rt, lp = cfg["data"], cfg["retina"], cfg["logpolar"]
        self.S = S = int(d["image_size"])
        self.bg = float(d["transparent_background"])
        self.cx = self.cy = (S - 1) / 2.0
        self.colors = list(cfg["areas"]["V1"]["color_channels"])
        self.maps = [f"{c}_{pol}" for c in self.colors for pol in ("on", "off")]
        self.n_maps = len(self.maps)
        self.sig_c, self.sig_s = float(rt["sigma_center_px"]), float(rt["sigma_surround_px"])
        self.trunc = float(rt["truncate_sigma"])
        self.gain = float(rt["gain"])
        self.kc = gauss1d(self.sig_c, self.trunc, S)
        self.ks = gauss1d(self.sig_s, self.trunc, S)
        self.r0 = float(lp["r0_px"])
        self.rmax = S / 2.0 - float(lp["rmax_margin_px"])
        self.phi_offset = float(lp["phi_offset_bins"])
        g = logpolar_positions(int(lp["n_r"]), int(lp["n_phi"]), self.r0, self.rmax, self.cx, self.cy, self.phi_offset)
        self.grid = g
        self.n_pos = int(lp["n_r"]) * int(lp["n_phi"])
        self.n_out = self.n_pos * self.n_maps
        x, y = g["x"], g["y"]
        if (x < 0).any() or (x > S - 1).any() or (y < 0).any() or (y > S - 1).any():
            raise ConfigError("로그-극좌표 표본이 영상 밖에 있다. rmax_margin_px 를 키워라.")
        self.levels = sorted({0.0} | {float(v) for v in lp["blur_levels_px"]})
        spacing = np.maximum(g["spacing_r"], g["spacing_phi"])
        req = float(lp["blur_beta"]) * spacing
        lv = np.asarray(self.levels)
        self.level_of_pos = np.argmin(np.abs(req[:, None] - lv[None, :]), axis=1)
        self.level_kernels = [gauss1d(s, self.trunc, S) for s in self.levels]
        x0 = np.clip(np.floor(x).astype(np.int64), 0, S - 2)
        y0 = np.clip(np.floor(y).astype(np.int64), 0, S - 2)
        self._bil = (y0, x0, (y - y0).astype(np.float64), (x - x0).astype(np.float64))
        self.sample_area_px2 = (g["spacing_r"] * g["spacing_phi"]).astype(np.float64)
        self._nearest_map: tuple[np.ndarray, np.ndarray] | None = None
        # 원점 근처 근접 중복 표본 (0.5 화소 안)
        close = 0
        for a in range(0, self.n_pos, 512):
            d2 = (x[a:a + 512, None] - x[None, :]) ** 2 + (y[a:a + 512, None] - y[None, :]) ** 2
            close += int(((d2 < 0.25).sum() - min(512, self.n_pos - a)))
        self.near_duplicate_pairs = close // 2

    # ---------------------------------------------------------------- 단계
    def resize_pad(self, im: Any) -> tuple[np.ndarray, dict[str, Any]]:
        S = self.S
        w, h = im.size
        s = S / float(max(w, h))
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        im2 = im.resize((nw, nh), resample) if (nw, nh) != (w, h) else im
        arr = np.asarray(im2, dtype=np.float32) / 255.0
        top, left = (S - nh) // 2, (S - nw) // 2
        arr = np.pad(arr, ((top, S - nh - top), (left, S - nw - left), (0, 0)), mode="edge")
        return arr, {"orig_w": w, "orig_h": h, "scale": s, "resized_w": nw, "resized_h": nh, "pad_top": top, "pad_left": left}

    @staticmethod
    def opponent(arr: np.ndarray) -> dict[str, np.ndarray]:
        R, G, B = (arr[..., i].astype(np.float64) for i in range(3))
        return {"lum": R + G, "rg": R - G, "by": B - 0.5 * (R + G)}

    def dog(self, m: np.ndarray) -> np.ndarray:
        return self.gain * (blur2d(m, self.kc) - blur2d(m, self.ks))

    def lgn_maps(self, arr: np.ndarray) -> dict[str, np.ndarray]:
        opp = self.opponent(arr)
        out: dict[str, np.ndarray] = {}
        for c in self.colors:
            dg = self.dog(opp[c])
            out[f"{c}_on"] = np.maximum(dg, 0.0)
            out[f"{c}_off"] = np.maximum(-dg, 0.0)
        return out

    def sample(self, maps: dict[str, np.ndarray]) -> np.ndarray:
        y0, x0, fy, fx = self._bil
        vec = np.zeros((self.n_pos, self.n_maps), np.float64)
        for li, k in enumerate(self.level_kernels):
            sel = np.nonzero(self.level_of_pos == li)[0]
            if sel.size == 0:
                continue
            for mi, name in enumerate(self.maps):
                m = maps[name] if len(k) == 1 else blur2d(maps[name], k)
                yy, xx, wy, wx = y0[sel], x0[sel], fy[sel], fx[sel]
                vec[sel, mi] = ((1 - wx) * (1 - wy) * m[yy, xx] + wx * (1 - wy) * m[yy, xx + 1]
                                + (1 - wx) * wy * m[yy + 1, xx] + wx * wy * m[yy + 1, xx + 1])
        return vec.reshape(-1).astype(np.float32)

    def from_array(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape != (self.S, self.S, 3):
            raise ValueError(f"배열 모양 {arr.shape} != ({self.S},{self.S},3)")
        return self.sample(self.lgn_maps(np.asarray(arr, np.float32)))

    def from_path(self, path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        im, meta = load_pil_rgb(path, self.bg)
        arr, m2 = self.resize_pad(im)
        return self.from_array(arr), {**meta, **m2}

    # ---------------------------------------------------------------- 진단
    def info(self) -> dict[str, Any]:
        g = self.grid
        counts = np.bincount(self.level_of_pos, minlength=len(self.levels))
        return {"image_size_px": self.S, "center_px": [self.cx, self.cy], "colors": self.colors, "maps": self.maps,
                "dog_sigma_center_px": self.sig_c, "dog_sigma_surround_px": self.sig_s,
                "dog_kernel_radius_px": [int((len(self.kc) - 1) // 2), int((len(self.ks) - 1) // 2)],
                "border_mode": "reflect", "resize_ko": "긴 변을 S 에 맞춤 (Lanczos), 짧은 변은 가장자리 복제 패딩, 색 범위 [0,1]",
                "logpolar": {"n_r": g["n_r"], "n_phi": g["n_phi"], "r0_px": self.r0, "rmax_px": self.rmax,
                             "phi_offset_bins": self.phi_offset, "rule": "r(u)=r0*(exp(u*log(1+Rmax/r0))-1), u=(k+0.5)/n_r",
                             "ring_radius_px": [float(v) for v in g["r"][::g["n_phi"]]],
                             "origin_sample": False, "angle_seam": "각도 이음매는 이웃 계산에서 wrap",
                             "near_duplicate_pairs_lt_0p5px": self.near_duplicate_pairs,
                             "coverage_fraction_of_image": float(math.pi * self.rmax ** 2 / self.S ** 2)},
                "lowpass_levels_px": self.levels, "positions_per_level": [int(v) for v in counts],
                "interpolation": "bilinear", "n_positions": self.n_pos, "n_maps": self.n_maps, "n_outputs": self.n_out,
                "orientation_frame_ko": "원영상 좌표 (x 오른쪽, y 위쪽, 반시계 각도). V1 필터는 표본의 원영상 좌표에서 계산한다.",
                "dog_border_affected_samples": int((g["r"] > (self.S / 2.0 - (len(self.ks) - 1) / 2.0)).sum())}

    def dc_check(self) -> dict[str, Any]:
        """균일 영상: DoG 커널이 각각 합 1 이라 내부 잔여가 0 이어야 한다. reflect 경계라 균일 영상은 경계에서도 0."""
        S, rad = self.S, int((len(self.ks) - 1) // 2)
        rows = []
        worst_int, worst_border, worst_lgn = 0.0, 0.0, 0.0
        for rgb in ((0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.2, 0.6, 0.9)):
            arr = np.ones((S, S, 3), np.float32) * np.asarray(rgb, np.float32)
            opp = self.opponent(arr)
            for c in ("lum", "rg", "by"):
                dg = self.dog(opp[c])
                inner = dg[rad:S - rad, rad:S - rad] if S > 2 * rad else dg[:0, :0]
                bmask = np.ones_like(dg, bool)
                if S > 2 * rad:
                    bmask[rad:S - rad, rad:S - rad] = False
                mi = float(np.abs(inner).max()) if inner.size else 0.0
                mb = float(np.abs(dg[bmask]).max()) if bmask.any() else 0.0
                worst_int, worst_border = max(worst_int, mi), max(worst_border, mb)
                rows.append({"rgb": list(rgb), "channel": c, "max_abs_interior": mi, "max_abs_border": mb})
            worst_lgn = max(worst_lgn, float(np.abs(self.from_array(arr)).max()))
        tol = 1e-6
        return {"status": PASSED if max(worst_int, worst_border, worst_lgn) <= tol else FAILED, "tolerance": tol,
                "max_abs_interior": worst_int, "max_abs_border": worst_border, "max_abs_lgn_sample": worst_lgn,
                "valid_interior_margin_px": rad, "rows": rows,
                "note_ko": "실제 이미지는 가장자리 복제 패딩과 reflect 경계라 테두리 근처 DoG 가 내부와 다르게 계산될 수 있다 "
                           "(dog_border_affected_samples 참고)."}

    def _nearest(self) -> tuple[np.ndarray, np.ndarray]:
        if self._nearest_map is None:
            S = self.S
            yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)
            vx, vy = xx - self.cx, self.cy - yy
            r = np.hypot(vx, vy)
            phi = np.mod(np.arctan2(vy, vx), 2 * math.pi)
            u = np.log1p(r / self.r0) / self.grid["L"]
            k = np.clip(np.floor(u * self.grid["n_r"]).astype(np.int64), 0, self.grid["n_r"] - 1)
            m = np.mod(np.round(phi * self.grid["n_phi"] / (2 * math.pi) - self.phi_offset).astype(np.int64), self.grid["n_phi"])
            self._nearest_map = (k * self.grid["n_phi"] + m, r)
        return self._nearest_map

    def info_loss(self, arr: np.ndarray) -> dict[str, Any]:
        """표본 → 가장 가까운 표본으로 화소 복원 → 원래 ON/OFF 맵과의 상대 오차 (이심률 구간별). 저역통과와 성긴 표본의 손실 지표."""
        maps = self.lgn_maps(arr)
        vec = self.sample(maps).reshape(self.n_pos, self.n_maps).astype(np.float64)
        near, r = self._nearest()
        inside = r <= self.rmax
        bands = np.linspace(0.0, self.rmax, 5)
        out = {"outside_rmax_pixel_fraction": float(1.0 - inside.mean()), "bands_px": bands.tolist(), "maps": {}}
        for mi, name in enumerate(self.maps):
            orig = maps[name]
            rec = vec[near, mi]
            rows = []
            for a, b in zip(bands[:-1], bands[1:]):
                sel = inside & (r >= a) & (r < b if b < self.rmax else r <= b)
                num_ = float(np.sqrt(((rec - orig)[sel] ** 2).sum()))
                den = float(np.sqrt((orig[sel] ** 2).sum()))
                rows.append({"r_from": float(a), "r_to": float(b), "rel_error": (num_ / den) if den > 0 else None,
                             "orig_energy": den})
            out["maps"][name] = rows
        return out


class BatchLoader:
    """분할 목록 → LGN 벡터 배치. 프로세스 안 순차 로딩 (num_workers=0). 전처리 결과는 RAM (CPU) 에 캐시한다.
    분할별 접근 횟수를 센다 (dev/test 사용 기록)."""

    def __init__(self, data: DatasetManager, pre: RetinaLGNPreprocessor, be: Backend, cfg: dict[str, Any], timer: PhaseTimer):
        self.data, self.pre, self.be, self.timer = data, pre, be, timer
        self.cache: dict[str, np.ndarray] = {}
        self.cache_on = bool(cfg["data"]["cache_preprocessed"])
        self.cache_cap = int(cfg["data"]["cache_max_mb"]) * 2 ** 20
        self.cache_bytes = 0
        self.access: dict[str, int] = {"train": 0, "dev": 0, "test": 0, "external": 0}

    def vector(self, item: dict[str, Any]) -> np.ndarray:
        key = item["relpath"]
        v = self.cache.get(key)
        if v is not None:
            return v
        with self.timer.phase("file_read"):
            im, _meta = load_pil_rgb(self.data.root / key, self.pre.bg)
        with self.timer.phase("preprocess"):
            arr, _m2 = self.pre.resize_pad(im)
            v = self.pre.from_array(arr)
        if self.cache_on and self.cache_bytes + v.nbytes <= self.cache_cap:
            self.cache[key] = v
            self.cache_bytes += v.nbytes
        return v

    def batch(self, items: Sequence[dict[str, Any]], split: str) -> tuple[Any, np.ndarray]:
        self.access[split] = self.access.get(split, 0) + len(items)
        X = np.stack([self.vector(it) for it in items], axis=0)
        y = np.asarray([int(it["label"]) for it in items], np.int64)
        return self.be.f32(X), y


# ======================================================================
# 6. 통제 자극 (역할 반응 진단·자체 검사용)
# ======================================================================
class StimulusFactory:
    """처리 크기 S 의 회색조 자극 (RGB 3 채널 동일). 좌표: 화면 중심 기준, x 오른쪽, y 위쪽, 각도 반시계 (원영상 좌표)."""

    def __init__(self, S: int, supersample: int = 3):
        self.S = int(S)
        self.ss = int(supersample)
        n = self.S * self.ss
        c = (n - 1) / 2.0
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
        self.vx = (xx - c) / self.ss
        self.vy = (c - yy) / self.ss

    def _down(self, m: np.ndarray) -> np.ndarray:
        s = self.ss
        return m.reshape(self.S, s, self.S, s).mean(axis=(1, 3))

    @staticmethod
    def rgb(gray: np.ndarray) -> np.ndarray:
        return np.repeat(np.clip(gray, 0, 1)[..., None], 3, axis=2).astype(np.float32)

    def constant(self, v: float) -> np.ndarray:
        return self.rgb(np.full((self.S, self.S), float(v)))

    def _coords(self, theta_deg: float, center: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        t = math.radians(theta_deg)
        dx, dy = self.vx - center[0], self.vy - center[1]
        along = dx * math.cos(t) + dy * math.sin(t)
        perp = -dx * math.sin(t) + dy * math.cos(t)
        return along, perp

    def line_mask(self, theta_deg: float, *, width: float = 2.0, length: float | None = None,
                  center: tuple[float, float] = (0.0, 0.0), half: bool = False) -> np.ndarray:
        along, perp = self._coords(theta_deg, center)
        m = np.abs(perp) <= width / 2.0
        if length is not None:
            m &= np.abs(along) <= length / 2.0
        if half:
            m &= along >= 0
        return self._down(m.astype(np.float64))

    def line(self, theta_deg: float, *, width: float = 2.0, length: float | None = None, fg: float = 1.0, bg: float = 0.0,
             center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        m = self.line_mask(theta_deg, width=width, length=length, center=center)
        return self.rgb(bg + (fg - bg) * m)

    def grating(self, theta_deg: float, wavelength: float, phase: float, contrast: float = 1.0, mean: float = 0.5) -> np.ndarray:
        _along, perp = self._coords(theta_deg, (0.0, 0.0))
        g = mean + 0.5 * contrast * np.cos(2 * math.pi * perp / wavelength + phase)
        return self.rgb(self._down(g))

    def cross(self, t1: float, t2: float, *, width: float = 2.0, fg: float = 1.0, bg: float = 0.0) -> np.ndarray:
        m = np.maximum(self.line_mask(t1, width=width), self.line_mask(t2, width=width))
        return self.rgb(bg + (fg - bg) * m)

    def corner(self, t1: float, t2: float, *, width: float = 2.0, fg: float = 1.0, bg: float = 0.0) -> np.ndarray:
        m = np.maximum(self.line_mask(t1, width=width, half=True), self.line_mask(t2, width=width, half=True))
        return self.rgb(bg + (fg - bg) * m)

    def arc(self, radius: float, *, width: float = 2.0, start_deg: float = 0.0, span_deg: float = 120.0,
            center: tuple[float, float] = (0.0, 0.0), fg: float = 1.0, bg: float = 0.0) -> np.ndarray:
        dx, dy = self.vx - center[0], self.vy - center[1]
        rr = np.hypot(dx, dy)
        ang = np.mod(np.degrees(np.arctan2(dy, dx)) - start_deg, 360.0)
        m = (np.abs(rr - radius) <= width / 2.0) & (ang <= span_deg)
        return self.rgb(bg + (fg - bg) * self._down(m.astype(np.float64)))

    @staticmethod
    def _polygon(px: np.ndarray, py: np.ndarray, verts: Sequence[tuple[float, float]]) -> np.ndarray:
        inside = np.zeros(px.shape, bool)
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            cond = (y1 > py) != (y2 > py)
            xint = (x2 - x1) * (py - y1) / ((y2 - y1) if y2 != y1 else 1e-12) + x1
            inside ^= cond & (px < xint)
        return inside

    def shape(self, kind: str, size: float, *, center: tuple[float, float] = (0.0, 0.0), rotation_deg: float = 0.0,
              fg: float = 0.0, bg: float = 1.0) -> np.ndarray:
        t = math.radians(rotation_deg)
        dx, dy = self.vx - center[0], self.vy - center[1]
        ax = dx * math.cos(t) + dy * math.sin(t)
        ay = -dx * math.sin(t) + dy * math.cos(t)
        R = size / 2.0
        if kind == "circle":
            m = np.hypot(ax, ay) <= R
        elif kind == "square":
            m = np.maximum(np.abs(ax), np.abs(ay)) <= R * 0.85
        elif kind == "triangle":
            verts = [(R * math.cos(math.radians(90 + 120 * k)), R * math.sin(math.radians(90 + 120 * k))) for k in range(3)]
            m = self._polygon(ax, ay, verts)
        elif kind == "star":
            verts = []
            for k in range(10):
                rr = R if k % 2 == 0 else R * 0.45
                a = math.radians(90 + 36 * k)
                verts.append((rr * math.cos(a), rr * math.sin(a)))
            m = self._polygon(ax, ay, verts)
        else:
            raise ValueError(f"알 수 없는 도형: {kind}")
        return self.rgb(bg + (fg - bg) * self._down(m.astype(np.float64)))


# ======================================================================
# 7. 연결 구조 (LayerSpec / Connectivity) 와 영역 생성기 (V1 / V2 / V4 / IT)
# ======================================================================
@dataclass
class LayerSpec:
    """한 영역·한 층의 고정 구조 (CPU numpy).

    뉴런 순서는 위치 우선: local = pos·n_ch + ch.  입력 연결은 CSR 두 입력군 (A, B):
      A: a_ptr (n+1,), a_src (E_a,) int64 (원천 층 local 주소), a_w (E_a,) float32
      B: b_ptr, b_src, b_w — 역할 연산 min2 뉴런만 쓴다 (u = min(Σ_A, Σ_B)). 선형 뉴런은 u = Σ_A (B 비어 있음).
    좌표: x, y = 수용장 중심 (처리 영상 화소, 원점 왼쪽 위 화소 중심, x 오른쪽, y 아래쪽),
          Z = 영역 번호×10 + 피질층 번호 (무차원 계층 좌표; L6 이웃은 같은 Z 안에서만 고른다).
    """

    area: str
    name: str
    src_key: str
    n_src: int
    n_pos: int
    n_ch: int
    channel_names: list[str]
    channel_roles: list[str]
    channel_meta: dict[str, list[Any]]
    op_min: np.ndarray
    a_ptr: np.ndarray
    a_src: np.ndarray
    a_w: np.ndarray
    b_ptr: np.ndarray
    b_src: np.ndarray
    b_w: np.ndarray
    x: np.ndarray
    y: np.ndarray
    rf: np.ndarray
    ring: np.ndarray
    ang: np.ndarray
    pos_x: np.ndarray
    pos_y: np.ndarray
    pos_rf: np.ndarray
    grid_shape: tuple[int, int] | None
    pos_nb: np.ndarray
    pos_nb_count: np.ndarray
    learnable: bool
    l6_capable: bool
    description: dict[str, Any]
    build_stats: dict[str, Any] = field(default_factory=dict)
    start: int = -1
    end: int = -1
    routed: np.ndarray | None = None          # (n,) bool — 두 번째 입력을 이미지마다 라우팅으로 고르는 행 (새 모드의 V4 L2 곡선)
    missing: np.ndarray | None = None         # (n,) bool — 유효 후보 없음 (missing_candidate): u = 0
    routing: Any = None                       # RoutingTable (새 모드) 또는 None (legacy)

    @property
    def key(self) -> str:
        return f"{self.area}/{self.name}"

    @property
    def n(self) -> int:
        return int(self.n_pos * self.n_ch)

    @property
    def z(self) -> float:
        return float(AREA_INDEX[self.area] * 10 + LAYER_NUMBER[self.name])

    def fan_in(self) -> np.ndarray:
        return np.diff(self.a_ptr) + np.diff(self.b_ptr)

    def row(self, j: int) -> dict[str, np.ndarray]:
        return {"a_src": self.a_src[self.a_ptr[j]:self.a_ptr[j + 1]], "a_w": self.a_w[self.a_ptr[j]:self.a_ptr[j + 1]],
                "b_src": self.b_src[self.b_ptr[j]:self.b_ptr[j + 1]], "b_w": self.b_w[self.b_ptr[j]:self.b_ptr[j + 1]]}

    def hash(self) -> str:
        names = sha256_text(dumps([self.key, self.src_key, self.n_src, self.n_pos, self.n_ch, self.channel_names,
                                   self.channel_roles, list(self.grid_shape) if self.grid_shape else None], indent=None))
        h = sha256_text(names + sha256_array(self.op_min, self.a_ptr, self.a_src, self.a_w, self.b_ptr, self.b_src, self.b_w,
                                             self.x, self.y, self.pos_nb, self.pos_nb_count))
        if self.routing is None:                  # legacy: v1 과 같은 해시
            return h
        return sha256_text(h + sha256_array(self.routed, self.missing) + self.routing.hash())


class CSRBuilder:
    """행별 (송신 주소 → 가중치) 사전. 같은 (수신, 입력군, 송신) 이 다시 오면 가중치를 더한다 (merge_sum 정책, 개수 기록)."""

    def __init__(self, n_rows: int, n_src: int):
        self.n, self.n_src = int(n_rows), int(n_src)
        self.a: list[dict[int, float]] = [{} for _ in range(self.n)]
        self.b: list[dict[int, float]] = [{} for _ in range(self.n)]
        self.merged = 0

    def add(self, row: int, src: int, w: float, group: str = "A") -> None:
        if not (0 <= src < self.n_src):
            raise StructureError(f"송신 주소 {src} 가 원천 층 범위 [0,{self.n_src}) 밖이다.")
        d = (self.a if group == "A" else self.b)[row]
        if src in d:
            d[src] += float(w)
            self.merged += 1
        else:
            d[src] = float(w)

    @staticmethod
    def _csr(rows: list[dict[int, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lens = np.asarray([len(r) for r in rows], np.int64)
        ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        src = np.empty(int(ptr[-1]), np.int64)
        w = np.empty(int(ptr[-1]), np.float32)
        for i, r in enumerate(rows):
            if r:
                ks = sorted(r)
                src[ptr[i]:ptr[i + 1]] = ks
                w[ptr[i]:ptr[i + 1]] = [r[k] for k in ks]
        return ptr, src, w

    def finalize(self) -> tuple[np.ndarray, ...]:
        return (*self._csr(self.a), *self._csr(self.b))


def identity_csr(n: int) -> tuple[np.ndarray, ...]:
    return (np.arange(n + 1, dtype=np.int64), np.arange(n, dtype=np.int64), np.ones(n, np.float32),
            np.zeros(n + 1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32))


def grid_neighbors(n_r: int, n_phi: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """피질 격자 Chebyshev 거리 ≤ radius 인 위치 (자기 위치 포함). 각도는 wrap, 반경은 경계에서 자른다. 채움 = n_pos."""
    n = n_r * n_phi
    rows = []
    for k in range(n_r):
        for m in range(n_phi):
            s = set()
            for dk in range(-radius, radius + 1):
                kk = k + dk
                if 0 <= kk < n_r:
                    for dm in range(-radius, radius + 1):
                        s.add(kk * n_phi + (m + dm) % n_phi)
            rows.append(sorted(s))
    K = max(len(r) for r in rows)
    idx = np.full((n, K), n, np.int64)
    for i, r in enumerate(rows):
        idx[i, :len(r)] = r
    return idx, np.asarray([len(r) for r in rows], np.int64)


def single_position_neighbors() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((1, 1), np.int64), np.ones(1, np.int64)


def validate_layer(sp: LayerSpec) -> dict[str, Any]:
    n = sp.n
    for nm in ("op_min", "x", "y", "rf", "ring", "ang"):
        if len(getattr(sp, nm)) != n:
            raise StructureError(f"{sp.key}.{nm} 길이 {len(getattr(sp, nm))} != {n}")
    if len(sp.channel_names) != sp.n_ch or len(sp.channel_roles) != sp.n_ch:
        raise StructureError(f"{sp.key}: 채널 이름·역할 수가 n_ch 와 다르다.")
    out: dict[str, Any] = {}
    for g, ptr, src, w in (("A", sp.a_ptr, sp.a_src, sp.a_w), ("B", sp.b_ptr, sp.b_src, sp.b_w)):
        if len(ptr) != n + 1 or ptr[0] != 0 or (np.diff(ptr) < 0).any() or ptr[-1] != len(src) or len(src) != len(w):
            raise StructureError(f"{sp.key}: 입력군 {g} CSR 형식 오류")
        if len(src) and (src.min() < 0 or src.max() >= sp.n_src):
            raise StructureError(f"{sp.key}: 입력군 {g} 주소가 원천 층 범위 밖")
        if not np.isfinite(w).all():
            raise StructureError(f"{sp.key}: 입력군 {g} 가중치에 NaN/Inf")
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
        keys = np.sort(rows * np.int64(sp.n_src) + src)
        dup = int((keys[1:] == keys[:-1]).sum()) if len(keys) > 1 else 0
        if dup:
            raise StructureError(f"{sp.key}: 입력군 {g} 에 중복 edge {dup} 개 (merge_sum 뒤에도 남음)")
        out[f"edges_{g}"] = int(len(src))
    fa, fb = np.diff(sp.a_ptr), np.diff(sp.b_ptr)
    routed = sp.routed if sp.routed is not None else np.zeros(n, bool)
    if (fb[~sp.op_min] > 0).any():
        raise StructureError(f"{sp.key}: 선형 뉴런에 입력군 B 가 있다.")
    if (routed & ~sp.op_min).any() or (fb[routed] > 0).any():
        raise StructureError(f"{sp.key}: 라우팅 행은 min2 이고 정적 입력군 B 가 비어 있어야 한다 (B 는 이미지마다 후보에서 고른다).")
    if ((fa[sp.op_min] == 0) | ((fb[sp.op_min] == 0) & ~routed[sp.op_min])).any():
        raise StructureError(f"{sp.key}: min2 뉴런은 입력군 A·B 가 모두 있어야 한다 (라우팅 행은 B 후보 표로 대신한다).")
    fan = fa + fb
    out.update(n_neurons=n, fan_in_min=int(fan.min()) if n else 0, fan_in_mean=float(fan.mean()) if n else 0.0,
               fan_in_max=int(fan.max()) if n else 0, n_without_input=int((fan == 0).sum()),
               n_min2=int(sp.op_min.sum()), n_linear=int(n - sp.op_min.sum()))
    if sp.name == "L3" and out["n_without_input"]:
        raise StructureError(f"{sp.key}: 부모 (L2 입력) 없는 L3 출력 {out['n_without_input']} 개")
    return out


def _channel_meta(n_ch: int, **cols: list[Any]) -> dict[str, list[Any]]:
    for k, v in cols.items():
        if len(v) != n_ch:
            raise StructureError(f"채널 메타 {k} 길이 {len(v)} != {n_ch}")
    return {k: list(v) for k, v in cols.items()}


def _block_children(nr_c: int, nphi_c: int, nr_p: int, nphi_p: int) -> list[list[int]]:
    fr, fa = nr_c // nr_p, nphi_c // nphi_p
    out = []
    for K in range(nr_p):
        for M in range(nphi_p):
            out.append([k * nphi_c + m for k in range(K * fr, (K + 1) * fr) for m in range(M * fa, (M + 1) * fa)])
    return out


def _pool_positions(child: LayerSpec, children: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    px = np.asarray([child.pos_x[c].mean() for c in children], np.float64)
    py = np.asarray([child.pos_y[c].mean() for c in children], np.float64)
    prf = np.asarray([float(np.max(np.hypot(child.pos_x[c] - px[i], child.pos_y[c] - py[i]) + child.pos_rf[c]))
                      for i, c in enumerate(children)], np.float64)
    return px, py, prf


def _local_spacing(px: np.ndarray, py: np.ndarray, pre: "RetinaLGNPreprocessor", n_r: int, n_phi: int) -> np.ndarray:
    r = np.hypot(px - pre.cx, pre.cy - py)
    L = pre.grid["L"]
    return np.sqrt(((r + pre.r0) * L / n_r) * (2 * math.pi * r / n_phi))


def _displaced(px: np.ndarray, py: np.ndarray, P: int, theta_deg: float, D: float, sign: int) -> int | None:
    """원영상 좌표에서 θ 방향 (sign=+1 앞, −1 뒤) 으로 D 만큼 떨어진 점에 가장 가까운 격자 위치. 자기 자신이면 None."""
    t = math.radians(theta_deg)
    tx, ty = px[P] + sign * D * math.cos(t), py[P] - sign * D * math.sin(t)
    j = int(np.argmin((px - tx) ** 2 + (py - ty) ** 2))
    return None if j == P else j


def _ori_index(oris: list[float], theta: float) -> int | None:
    th = theta % 180.0
    for i, o in enumerate(oris):
        if abs(((o - th + 90.0) % 180.0) - 90.0) < 1e-6:
            return i
    return None


def _spec(area: str, name: str, src_key: str, n_src: int, n_pos: int, n_ch: int, names: list[str], roles: list[str],
          meta: dict[str, list[Any]], op_min: np.ndarray, csr: tuple[np.ndarray, ...], x: np.ndarray, y: np.ndarray,
          rf: np.ndarray, ring: np.ndarray, ang: np.ndarray, pos: tuple[np.ndarray, np.ndarray, np.ndarray],
          grid_shape: tuple[int, int] | None, nb: tuple[np.ndarray, np.ndarray], learnable: bool, desc: dict[str, Any],
          stats: dict[str, Any] | None = None, routed: np.ndarray | None = None, missing: np.ndarray | None = None,
          routing: Any = None) -> LayerSpec:
    a_ptr, a_src, a_w, b_ptr, b_src, b_w = csr
    sp = LayerSpec(area=area, name=name, src_key=src_key, n_src=int(n_src), n_pos=int(n_pos), n_ch=int(n_ch),
                   channel_names=list(names), channel_roles=list(roles), channel_meta=meta, op_min=np.asarray(op_min, bool),
                   a_ptr=a_ptr, a_src=a_src, a_w=a_w.astype(np.float32), b_ptr=b_ptr, b_src=b_src, b_w=b_w.astype(np.float32),
                   x=np.asarray(x, np.float32), y=np.asarray(y, np.float32), rf=np.asarray(rf, np.float32),
                   ring=np.asarray(ring, np.int32), ang=np.asarray(ang, np.int32),
                   pos_x=np.asarray(pos[0], np.float32), pos_y=np.asarray(pos[1], np.float32), pos_rf=np.asarray(pos[2], np.float32),
                   grid_shape=grid_shape, pos_nb=nb[0], pos_nb_count=nb[1], learnable=learnable, l6_capable=(name in ("L2", "L3")),
                   description=desc, build_stats=dict(stats or {}), routed=routed, missing=missing, routing=routing)
    sp.build_stats.update(validate_layer(sp))
    return sp


def _per_neuron(arr: np.ndarray, n_ch: int) -> np.ndarray:
    return np.repeat(np.asarray(arr), n_ch)


# ---------------------------------------------------------------------- V1
def build_v1(cfg: dict[str, Any], pre: "RetinaLGNPreprocessor") -> list[LayerSpec]:
    a = cfg["areas"]["V1"]
    rad6 = int(cfg["l6"]["neighbor_grid_radius"])
    lg = pre.grid
    n_maps, n4 = pre.n_maps, pre.n_out
    nb4 = grid_neighbors(lg["n_r"], lg["n_phi"], rad6)
    L4 = _spec("V1", "L4", "LGN", n4, pre.n_pos, n_maps, list(pre.maps), ["V1_L4_lgn_relay"] * n_maps,
               _channel_meta(n_maps, map=list(pre.maps), color=[m.split("_")[0] for m in pre.maps],
                             onoff=[m.split("_")[1] for m in pre.maps]),
               np.zeros(n4, bool), identity_csr(n4), _per_neuron(lg["x"], n_maps), _per_neuron(lg["y"], n_maps),
               _per_neuron(np.maximum(lg["spacing_r"], lg["spacing_phi"]), n_maps), _per_neuron(lg["ring"], n_maps),
               _per_neuron(lg["ang"], n_maps), (lg["x"], lg["y"], np.maximum(lg["spacing_r"], lg["spacing_phi"])),
               (lg["n_r"], lg["n_phi"]), nb4, False,
               {"role_ko": "감각 입력 수신·고정 일대일 중계 (LGN 표본 → V1 L4). q 학습 없음, L6 없음.",
                "operation": "u_j = LGN_j (가중치 1, 일대일)", "input_shape": [pre.n_pos, n_maps], "output_shape": [pre.n_pos, n_maps],
                "connection_rule_ko": "LGN 표본 위치·맵 하나당 L4 뉴런 하나 (위치 우선 순서 동일)"})
    # V1 단순 세포 위치 격자 (로그-극좌표, LGN 과 같은 r0·Rmax)
    g1 = logpolar_positions(int(a["n_r"]), int(a["n_phi"]), pre.r0, pre.rmax, pre.cx, pre.cy, pre.phi_offset)
    n_pos1 = int(a["n_r"]) * int(a["n_phi"])
    sp1 = np.sqrt(g1["spacing_r"] * g1["spacing_phi"])
    sigma = np.maximum(float(a["gabor_sigma_factor"]) * sp1, float(a["gabor_sigma_min_px"]))
    lam = float(a["gabor_wavelength_factor"]) * sigma
    rad = float(a["rf_radius_sigma"]) * sigma
    oris = [float(t) for t in a["orientations_deg"]]
    phases = [float(p) for p in a["phases_rad"]]
    pols = [int(p) for p in a["polarities"]]
    colors = list(pre.colors)
    n_o, n_f, n_s = len(oris), len(phases), len(pols)
    chans = [(c, o, f, s) for c in range(len(colors)) for o in range(n_o) for f in range(n_f) for s in range(n_s)]
    n_ch2 = len(chans)
    lx, ly, area_k = lg["x"], lg["y"], pre.sample_area_px2
    sels, total, n_nearest_fallback = [], 0, 0
    for p in range(n_pos1):
        d2 = (lx - g1["x"][p]) ** 2 + (ly - g1["y"][p]) ** 2
        sel = np.nonzero(d2 <= rad[p] ** 2)[0]
        if sel.size == 0:
            sel = np.asarray([int(np.argmin(d2))])
            n_nearest_fallback += 1
        sels.append(sel)
        total += n_ch2 * 2 * sel.size
    a_ptr = np.zeros(n_pos1 * n_ch2 + 1, np.int64)
    a_src = np.empty(total, np.int64)
    a_w = np.empty(total, np.float32)
    cos_t = np.cos(np.radians(oris))[:, None]
    sin_t = np.sin(np.radians(oris))[:, None]
    on_idx = [pre.maps.index(f"{c}_on") for c in colors]
    off_idx = [pre.maps.index(f"{c}_off") for c in colors]
    pos, row, degenerate = 0, 0, 0
    for p in range(n_pos1):
        sel = sels[p]
        ns = sel.size
        vx = (lx[sel] - g1["x"][p])[None, :]
        vy = (-(ly[sel] - g1["y"][p]))[None, :]
        along = vx * cos_t + vy * sin_t
        perp = -vx * sin_t + vy * cos_t
        env = area_k[sel][None, :] * np.exp(-(along ** 2 + perp ** 2) / (2.0 * sigma[p] ** 2))
        den = env.sum(1, keepdims=True)
        G = []
        for ph in phases:
            car = np.cos(2.0 * math.pi * perp / lam[p] + ph)
            cbar = (env * car).sum(1, keepdims=True) / np.where(den > 0, den, 1.0)
            gg = env * (car - cbar)
            nrm = np.abs(gg).sum(1, keepdims=True)
            degenerate += int((nrm[:, 0] <= 1e-12).sum())
            G.append(np.where(nrm > 1e-12, gg / np.where(nrm > 1e-12, nrm, 1.0), 0.0))
        for (ci, oi, fi, si) in chans:
            gw = (pols[si] * G[fi][oi]).astype(np.float32)
            a_src[pos:pos + ns] = sel * n_maps + on_idx[ci]
            a_w[pos:pos + ns] = gw
            a_src[pos + ns:pos + 2 * ns] = sel * n_maps + off_idx[ci]
            a_w[pos + ns:pos + 2 * ns] = -gw
            pos += 2 * ns
            row += 1
            a_ptr[row] = pos
    n2 = n_pos1 * n_ch2
    names2 = [f"{colors[c]}_o{oris[o]:g}_p{f}_{'+' if pols[s] > 0 else '-'}" for (c, o, f, s) in chans]
    nb1 = grid_neighbors(int(a["n_r"]), int(a["n_phi"]), rad6)
    L2 = _spec("V1", "L2", "V1/L4", n4, n_pos1, n_ch2, names2, ["V1_simple_oriented_gabor"] * n_ch2,
               _channel_meta(n_ch2, orientation_deg=[oris[o] for (_c, o, _f, _s) in chans],
                             phase_rad=[phases[f] for (_c, _o, f, _s) in chans], polarity=[pols[s] for (_c, _o, _f, s) in chans],
                             color=[colors[c] for (c, _o, _f, _s) in chans], family=["simple"] * n_ch2),
               np.zeros(n2, bool), (a_ptr, a_src, a_w, np.zeros(n2 + 1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)),
               _per_neuron(g1["x"], n_ch2), _per_neuron(g1["y"], n_ch2), _per_neuron(sigma, n_ch2),
               _per_neuron(g1["ring"], n_ch2), _per_neuron(g1["ang"], n_ch2), (g1["x"], g1["y"], sigma),
               (int(a["n_r"]), int(a["n_phi"])), nb1, True,
               {"role_ko": "위치별 방향·위상 고정 필터 (Gabor) 반응. 부호 있는 필터 내부 응답 u 와 비음수 출력 max(0,u−θ) 를 구분하고 "
                           "양·음 극성을 별도 뉴런으로 둔다.",
                "operation": "u_j = Σ_k g_jk·(ON_k − OFF_k),  g = 면적 가중 Gaussian envelope × (cos(2π·perp/λ + φ) − 가중 평균), Σ|g| = 1, "
                             "극성 −1 은 −g",
                "orientation_frame_ko": "원영상 좌표 기준 (x 오른쪽, y 위쪽, 반시계). perp = −vx·sinθ + vy·cosθ",
                "input_shape": [pre.n_pos, n_maps], "output_shape": [n_pos1, n_ch2],
                "channels": {"orientations_deg": oris, "phases_rad": phases, "polarities": pols, "colors": colors,
                             "n_orientation_x_phase": n_o * n_f, "polarity_multiplier": n_s, "color_multiplier": len(colors)},
                "connection_rule_ko": "V1 위치에서 반경 rf_radius_sigma·σ 안의 LGN 표본 (원영상 거리), 같은 색의 ON·OFF 맵",
                "sigma_px": {"min": float(sigma.min()), "max": float(sigma.max())},
                "wavelength_px": {"min": float(lam.min()), "max": float(lam.max())}},
               {"degenerate_filters": degenerate, "nearest_sample_fallback_positions": n_nearest_fallback,
                "inner_positions_sigma_floor": int((float(a["gabor_sigma_factor"]) * sp1 < float(a["gabor_sigma_min_px"])).sum())})
    # V1 L3: 같은 위치·색·방향의 위상×극성 풀링 (부모 n_f·n_s)
    n_ch3 = len(colors) * n_o
    kpar = n_f * n_s
    n3 = n_pos1 * n_ch3
    a3_ptr = np.arange(n3 + 1, dtype=np.int64) * kpar
    a3_src = np.empty(n3 * kpar, np.int64)
    r_ = 0
    for p in range(n_pos1):
        for c in range(len(colors)):
            for o in range(n_o):
                base = p * n_ch2 + ((c * n_o + o) * n_f) * n_s
                a3_src[r_ * kpar:(r_ + 1) * kpar] = base + np.arange(kpar)
                r_ += 1
    a3_w = np.full(n3 * kpar, 1.0 / kpar, np.float32)
    names3 = [f"{colors[c]}_o{oris[o]:g}" for c in range(len(colors)) for o in range(n_o)]
    L3 = _spec("V1", "L3", "V1/L2", n2, n_pos1, n_ch3, names3, ["V1_phase_pooled_orientation"] * n_ch3,
               _channel_meta(n_ch3, orientation_deg=[oris[o] for c in range(len(colors)) for o in range(n_o)],
                             color=[colors[c] for c in range(len(colors)) for o in range(n_o)], family=["complex_like"] * n_ch3),
               np.zeros(n3, bool), (a3_ptr, a3_src, a3_w, np.zeros(n3 + 1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)),
               _per_neuron(g1["x"], n_ch3), _per_neuron(g1["y"], n_ch3), _per_neuron(sigma, n_ch3),
               _per_neuron(g1["ring"], n_ch3), _per_neuron(g1["ang"], n_ch3), (g1["x"], g1["y"], sigma),
               (int(a["n_r"]), int(a["n_phi"])), nb1, True,
               {"role_ko": "같은 위치·색·방향의 위상×극성 정류 반응 평균 (국소 결합). 완전한 위상 불변성을 주장하지 않는다.",
                "operation": f"u_j = (1/{kpar})·Σ_{{phase,polarity}} transmitted_L2", "input_shape": [n_pos1, n_ch2],
                "output_shape": [n_pos1, n_ch3], "connection_rule_ko": f"수렴: L3 하나당 L2 부모 {kpar} 개 (같은 위치·색·방향)"})
    return [L4, L2, L3]


# ---------------------------------------------------------------------- V2
def build_v2(cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", v1: LayerSpec) -> list[LayerSpec]:
    a = cfg["areas"]["V2"]
    rad6 = int(cfg["l6"]["neighbor_grid_radius"])
    oris = [float(t) for t in cfg["areas"]["V1"]["orientations_deg"]]
    colors = list(pre.colors)
    n_o = len(oris)
    nr1, nphi1 = v1.grid_shape
    nr2, nphi2 = int(a["n_r"]), int(a["n_phi"])
    children = _block_children(nr1, nphi1, nr2, nphi2)
    px, py, prf = _pool_positions(v1, children)
    n_pos = nr2 * nphi2
    ring = np.repeat(np.arange(nr2), nphi2)
    ang = np.tile(np.arange(nphi2), nr2)
    nb = grid_neighbors(nr2, nphi2, rad6)
    # L4: 방향별로 V1 L3 의 블록 위치·색 평균
    b4 = CSRBuilder(n_pos * n_o, v1.n)
    for P, ch in enumerate(children):
        k = 1.0 / (len(ch) * len(colors))
        for o in range(n_o):
            for c1 in ch:
                for ci in range(len(colors)):
                    b4.add(P * n_o + o, c1 * v1.n_ch + ci * n_o + o, k)
    names4 = [f"o{t:g}" for t in oris]
    L4 = _spec("V2", "L4", "V1/L3", v1.n, n_pos, n_o, names4, ["V2_L4_pooled_orientation"] * n_o,
               _channel_meta(n_o, orientation_deg=oris, family=["orientation"] * n_o), np.zeros(n_pos * n_o, bool),
               b4.finalize(), _per_neuron(px, n_o), _per_neuron(py, n_o), _per_neuron(prf, n_o), _per_neuron(ring, n_o),
               _per_neuron(ang, n_o), (px, py, prf), (nr2, nphi2), nb, False,
               {"role_ko": "고정 수렴 중계: V1 L3 의 블록 위치와 색을 방향별로 평균",
                "operation": "u = mean_{block, color} transmitted_V1L3(같은 방향)", "input_shape": [v1.n_pos, v1.n_ch],
                "output_shape": [n_pos, n_o], "connection_rule_ko": f"블록 {nr1 // nr2}×{nphi1 // nphi2} 위치 × 색 {len(colors)} → 1"})
    # L2: 윤곽 연속 / 끝점 / 교차
    sp = _local_spacing(px, py, pre, nr2, nphi2)
    D = np.maximum(float(a["offset_factor"]) * sp, 1.0)
    pairs: list[tuple[int, int]] = []
    for diff in a["cross_diffs_deg"]:
        for i in range(n_o):
            j = _ori_index(oris, oris[i] + float(diff))
            if j is not None and j != i:
                pr = (min(i, j), max(i, j))
                if pr not in pairs:
                    pairs.append(pr)
    chans: list[tuple[str, int, int]] = [("contour", o, 0) for o in range(n_o)] + \
        [("endstop", o, 1) for o in range(n_o)] + [("endstop", o, -1) for o in range(n_o)] + \
        [("cross", i, j) for (i, j) in pairs]
    n_ch = len(chans)
    b2 = CSRBuilder(n_pos * n_ch, n_pos * n_o)
    op = np.zeros(n_pos * n_ch, bool)
    st = {"contour_forward_missing": 0, "contour_backward_missing": 0, "endstop_inhibition_missing": 0}
    inh = float(a["endstop_inhibition"])
    for P in range(n_pos):
        for ci, (fam, i, j) in enumerate(chans):
            r = P * n_ch + ci
            if fam == "contour":
                f = _displaced(px, py, P, oris[i], D[P], +1)
                bk = _displaced(px, py, P, oris[i], D[P], -1)
                parts = [(P, 0.5)] + ([(f, 0.25)] if f is not None else []) + ([(bk, 0.25)] if bk is not None else [])
                st["contour_forward_missing"] += f is None
                st["contour_backward_missing"] += bk is None
                tot = sum(w for _q, w in parts)
                for q_, w in parts:
                    b2.add(r, q_ * n_o + i, w / tot)
            elif fam == "endstop":
                q_ = _displaced(px, py, P, oris[i], D[P], j)
                b2.add(r, P * n_o + i, 1.0)
                if q_ is not None and inh > 0:
                    b2.add(r, q_ * n_o + i, -inh)
                else:
                    st["endstop_inhibition_missing"] += 1
            else:
                b2.add(r, P * n_o + i, 1.0, "A")
                b2.add(r, P * n_o + j, 1.0, "B")
                op[r] = True
    st["merged_duplicate_edges"] = b2.merged
    names2 = [f"contour_o{oris[i]:g}" if f == "contour" else (f"endstop_o{oris[i]:g}_{'fwd' if j > 0 else 'bwd'}" if f == "endstop"
              else f"cross_o{oris[i]:g}_o{oris[j]:g}") for (f, i, j) in chans]
    roles2 = ["V2_contour_continuation" if f == "contour" else ("V2_end_stopped" if f == "endstop" else "V2_crossing_junction")
              for (f, _i, _j) in chans]
    L2 = _spec("V2", "L2", "V2/L4", n_pos * n_o, n_pos, n_ch, names2, roles2,
               _channel_meta(n_ch, family=[f for (f, _i, _j) in chans], orientation_deg=[oris[i] for (_f, i, _j) in chans],
                             orientation2_deg=[(oris[j] if f == "cross" else None) for (f, _i, j) in chans],
                             direction=[(j if f == "endstop" else 0) for (f, _i, j) in chans]),
               op, b2.finalize(), _per_neuron(px, n_ch), _per_neuron(py, n_ch), _per_neuron(prf + D, n_ch),
               _per_neuron(ring, n_ch), _per_neuron(ang, n_ch), (px, py, prf + D), (nr2, nphi2), nb, True,
               {"role_ko": "근처 선 반응의 조합: 윤곽 연속 (같은 방향 앞뒤 위치), 끝점 (한쪽 끝 억제), 교차·접합 (두 방향 동시, min2)",
                "operation": {"contour": "u = 0.5·o(P,θ) + 0.25·o(P+D·dir θ, θ) + 0.25·o(P−D·dir θ, θ) (없는 쪽은 빼고 재정규화)",
                              "endstop": "u = o(P,θ) − w_inh·o(P ± D·dir θ, θ)",
                              "cross": "u = min(o(P,θ1), o(P,θ2))  (역할 연산 min2)"},
                "input_shape": [n_pos, n_o], "output_shape": [n_pos, n_ch],
                "connection_rule_ko": "D = offset_factor × 국소 V2 간격 (원영상 화소). 변위 이웃은 가장 가까운 격자 위치 (자기 자신이면 생략)",
                "cross_pairs": [[oris[i], oris[j]] for (i, j) in pairs]}, st)
    L3 = _spec("V2", "L3", "V2/L2", n_pos * n_ch, n_pos, n_ch, names2, [r_ + "_L3" for r_ in roles2], dict(L2.channel_meta),
               np.zeros(n_pos * n_ch, bool), identity_csr(n_pos * n_ch), L2.x, L2.y, L2.rf, L2.ring, L2.ang,
               (px, py, prf + D), (nr2, nphi2), nb, True,
               {"role_ko": "정리 (일대일 중계) 후 다음 영역으로 출력", "operation": "u_j = transmitted_L2_j", "input_shape": [n_pos, n_ch],
                "output_shape": [n_pos, n_ch], "connection_rule_ko": "일대일 (L3 j ← L2 j)"})
    return [L4, L2, L3]


# ---------------------------------------------------------------------- V4
def build_v4(cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", v2: LayerSpec, v1_l3: LayerSpec | None = None) -> list[LayerSpec]:
    a = cfg["areas"]["V4"]
    rmode = str(cfg["routing"]["mode"])
    routing_on = rmode != "legacy"
    if routing_on and v1_l3 is None:
        raise StructureError("라우팅 모드는 V1 L3 구조가 있어야 한다.")
    rad6 = int(cfg["l6"]["neighbor_grid_radius"])
    oris = [float(t) for t in cfg["areas"]["V1"]["orientations_deg"]]
    n_o = len(oris)
    nr2, nphi2 = v2.grid_shape
    nr4, nphi4 = int(a["n_r"]), int(a["n_phi"])
    children = _block_children(nr2, nphi2, nr4, nphi4)
    px, py, prf = _pool_positions(v2, children)
    n_pos = nr4 * nphi4
    ring = np.repeat(np.arange(nr4), nphi4)
    ang = np.tile(np.arange(nphi4), nr4)
    nb = grid_neighbors(nr4, nphi4, rad6)
    n_in = v2.n_ch
    b4 = CSRBuilder(n_pos * n_in, v2.n)
    for P, ch in enumerate(children):
        for cch in range(n_in):
            for c in ch:
                b4.add(P * n_in + cch, c * n_in + cch, 1.0 / len(ch))
    L4 = _spec("V4", "L4", "V2/L3", v2.n, n_pos, n_in, list(v2.channel_names), ["V4_L4_pooled_" + r for r in v2.channel_roles],
               dict(v2.channel_meta), np.zeros(n_pos * n_in, bool), b4.finalize(), _per_neuron(px, n_in), _per_neuron(py, n_in),
               _per_neuron(prf, n_in), _per_neuron(ring, n_in), _per_neuron(ang, n_in), (px, py, prf), (nr4, nphi4), nb, False,
               {"role_ko": "고정 수렴 중계: V2 L3 의 블록 위치 평균 (채널별)", "operation": "u = mean_block transmitted_V2L3",
                "input_shape": [v2.n_pos, n_in], "output_shape": [n_pos, n_in],
                "connection_rule_ko": f"블록 {nr2 // nr4}×{nphi2 // nphi4} 위치 → 1"})
    fam = v2.channel_meta["family"]
    ori_in = v2.channel_meta["orientation_deg"]
    dirs = v2.channel_meta["direction"]
    contour = {i: next(k for k in range(n_in) if fam[k] == "contour" and abs(ori_in[k] - oris[i]) < 1e-6) for i in range(n_o)}
    endf = {i: next(k for k in range(n_in) if fam[k] == "endstop" and dirs[k] > 0 and abs(ori_in[k] - oris[i]) < 1e-6) for i in range(n_o)}
    endb = {i: next(k for k in range(n_in) if fam[k] == "endstop" and dirs[k] < 0 and abs(ori_in[k] - oris[i]) < 1e-6) for i in range(n_o)}
    crosses = [k for k in range(n_in) if fam[k] == "cross"]
    chans: list[tuple[str, int, int]] = []
    curve_info: list[tuple[int, int, int, int, float]] = []          # (채널, θA, θB, 곡률 부호, κ) — 라우팅 표용 (legacy 배열에는 영향 없음)
    for kappa in a["curvature_deg"]:
        for sgn in (1, -1):
            for i in range(n_o):
                j = _ori_index(oris, oris[i] + sgn * float(kappa))
                if j is not None and j != i:
                    curve_info.append((len(chans), i, j, sgn, float(kappa)))
                    chans.append(("curve", i, j))
    chans += [("straight", i, i) for i in range(n_o)] + [("junction", k, k) for k in crosses] + [("lineend", i, i) for i in range(n_o)]
    n_ch = len(chans)
    sp = _local_spacing(px, py, pre, nr4, nphi4)
    D = np.maximum(float(a["offset_factor"]) * sp, 1.0)
    b2 = CSRBuilder(n_pos * n_ch, n_pos * n_in)
    op = np.zeros(n_pos * n_ch, bool)
    routed = np.zeros(n_pos * n_ch, bool)
    st = {"curve_same_position_fallback": 0, "straight_backward_used": 0, "straight_same_position_fallback": 0}
    for P in range(n_pos):
        for ci, (f, i, j) in enumerate(chans):
            r = P * n_ch + ci
            if f == "curve" and routing_on:
                b2.add(r, P * n_in + contour[i], 1.0, "A")             # 첫 입력 (P, θA) 은 그대로
                op[r] = True
                routed[r] = True                                      # 두 번째 입력은 라우팅 후보 표에서 이미지마다 고른다
            elif f == "curve":
                q_ = _displaced(px, py, P, oris[i], D[P], +1)
                if q_ is None:
                    q_ = P
                    st["curve_same_position_fallback"] += 1
                b2.add(r, P * n_in + contour[i], 1.0, "A")
                b2.add(r, q_ * n_in + contour[j], 1.0, "B")
                op[r] = True
            elif f == "straight":
                q_ = _displaced(px, py, P, oris[i], D[P], +1)
                if q_ is None:
                    q_ = _displaced(px, py, P, oris[i], D[P], -1)
                    st["straight_backward_used"] += q_ is not None
                if q_ is None:
                    q_ = P
                    st["straight_same_position_fallback"] += 1
                b2.add(r, P * n_in + contour[i], 1.0, "A")
                b2.add(r, q_ * n_in + contour[i], 1.0, "B")
                op[r] = True
            elif f == "junction":
                b2.add(r, P * n_in + i, 1.0)
            else:
                b2.add(r, P * n_in + endf[i], 0.5)
                b2.add(r, P * n_in + endb[i], 0.5)
    st["merged_duplicate_edges"] = b2.merged
    table = None
    missing_rows = None
    if routing_on:
        st.pop("curve_same_position_fallback")
        table = build_routing_table(cfg, pre, L4, n_ch, curve_info, contour, D, v1_l3, rmode)
        missing_rows = np.zeros(n_pos * n_ch, bool)
        missing_rows[table.curve_rows[table.missing]] = True
        st.update(routing_mode=rmode, routing_curve_neurons=int(len(table.curve_rows)), routing_missing_candidate=int(table.missing.sum()),
                  routing_valid_candidates_mean=float(table.cand_valid.sum(1).mean()) if len(table.curve_rows) else 0.0)
    names2, roles2, fam2, ori2 = [], [], [], []
    for (f, i, j) in chans:
        if f == "curve":
            names2.append(f"curve_o{oris[i]:g}_to_o{oris[j]:g}")
            roles2.append("V4_curve_fragment")
            ori2.append(oris[i])
        elif f == "straight":
            names2.append(f"straight_o{oris[i]:g}")
            roles2.append("V4_straight_continuation")
            ori2.append(oris[i])
        elif f == "junction":
            names2.append("junction_" + v2.channel_names[i])
            roles2.append("V4_junction_relay")
            ori2.append(ori_in[i])
        else:
            names2.append(f"lineend_o{oris[i]:g}")
            roles2.append("V4_line_end")
            ori2.append(oris[i])
        fam2.append(f)
    csign = [None] * n_ch
    ckap = [None] * n_ch
    for (ch_, _i, _j, s_, k_) in curve_info:
        csign[ch_], ckap[ch_] = s_, k_
    L2 = _spec("V4", "L2", "V4/L4", n_pos * n_in, n_pos, n_ch, names2, roles2,
               _channel_meta(n_ch, family=fam2, orientation_deg=ori2, curve_sign=csign, curvature_deg=ckap),
               op, b2.finalize(), _per_neuron(px, n_ch), _per_neuron(py, n_ch), _per_neuron(prf + D, n_ch), _per_neuron(ring, n_ch),
               _per_neuron(ang, n_ch), (px, py, prf + D), (nr4, nphi4), nb, True,
               {"role_ko": "더 넓은 수용장에서 선 조합: 곡선 조각 (θ 다음 위치에 θ±κ, min2), 직선 연속 (min2), 접합 중계, 선 끝 합",
                "operation": {"curve": "u = min(contour(P,θ), contour(P+D·dir θ, θ±κ))", "straight": "u = min(contour(P,θ), contour(P+D·dir θ, θ))",
                              "junction": "u = cross(P)", "lineend": "u = 0.5·endstop_fwd(P,θ) + 0.5·endstop_bwd(P,θ)"},
                "input_shape": [n_pos, n_in], "output_shape": [n_pos, n_ch],
                "connection_rule_ko": "D = offset_factor × 국소 V4 간격. 앞쪽 이웃이 없으면 곡선은 같은 위치를 쓴다 (개수 기록)",
                "curvature_deg": [float(k) for k in a["curvature_deg"]],
                "routing_ko": ("legacy: 곡선 두 번째 입력 = θA 방향 D 떨어진 점의 최근접 위치, 없으면 같은 위치 (v1 그대로)" if not routing_on else
                               f"{rmode}: 곡선 두 번째 입력 위치를 고정 후보 표 (K={cfg['routing']['max_candidates']}) 에서 이미지마다 고른다 "
                               "(geometry_only = 기하 첫 후보, v1_guided = V1 L3 경로 지지 점수 최대). 후보 없음 → u = 0 (missing_candidate)")},
               st, routed=(routed if routing_on else None), missing=missing_rows, routing=table)
    r3 = int(a["l3_grid_radius"])
    nb3, nb3c = grid_neighbors(nr4, nphi4, r3)
    b3 = CSRBuilder(n_pos * n_ch, n_pos * n_ch)
    for P in range(n_pos):
        qs = [int(q) for q in nb3[P, :nb3c[P]]]
        for cch in range(n_ch):
            for q_ in qs:
                b3.add(P * n_ch + cch, q_ * n_ch + cch, 1.0 / len(qs))
    L3 = _spec("V4", "L3", "V4/L2", n_pos * n_ch, n_pos, n_ch, list(names2), [r_ + "_L3" for r_ in roles2], dict(L2.channel_meta),
               np.zeros(n_pos * n_ch, bool), b3.finalize(), L2.x, L2.y, L2.rf, L2.ring, L2.ang, (px, py, prf + D), (nr4, nphi4), nb, True,
               {"role_ko": "같은 채널의 격자 이웃 평균 (국소 결합·정리)", "operation": "u_j = mean_{Q∈N(P)} transmitted_L2(Q, 채널)",
                "input_shape": [n_pos, n_ch], "output_shape": [n_pos, n_ch],
                "connection_rule_ko": f"수렴: Chebyshev 반경 {r3} (각도 wrap) 안 위치의 같은 채널 L2 → L3"})
    return [L4, L2, L3]


# ---------------------------------------------------------------------- IT
def build_it(cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", v4: LayerSpec) -> list[LayerSpec]:
    a = cfg["areas"]["IT"]
    rad6 = int(cfg["l6"]["neighbor_grid_radius"])
    nr4, nphi4 = v4.grid_shape
    rr, rp = int(a["region_r"]), int(a["region_phi"])
    children = _block_children(nr4, nphi4, rr, rp)
    px, py, prf = _pool_positions(v4, children)
    n_reg = rr * rp
    n_in = v4.n_ch
    nb4 = grid_neighbors(rr, rp, rad6)
    b4 = CSRBuilder(n_reg * n_in, v4.n)
    for R, ch in enumerate(children):
        for cch in range(n_in):
            for c in ch:
                b4.add(R * n_in + cch, c * n_in + cch, 1.0 / len(ch))
    L4 = _spec("IT", "L4", "V4/L3", v4.n, n_reg, n_in, list(v4.channel_names), ["IT_L4_region_pooled_" + r for r in v4.channel_roles],
               dict(v4.channel_meta), np.zeros(n_reg * n_in, bool), b4.finalize(), _per_neuron(px, n_in), _per_neuron(py, n_in),
               _per_neuron(prf, n_in), _per_neuron(np.repeat(np.arange(rr), rp), n_in), _per_neuron(np.tile(np.arange(rp), rr), n_in),
               (px, py, prf), (rr, rp), nb4, False,
               {"role_ko": "넓은 영역 풀링 중계: V4 L3 를 영역 (반경 블록 × 각도 블록) 별 평균", "operation": "u = mean_region transmitted_V4L3",
                "input_shape": [v4.n_pos, n_in], "output_shape": [n_reg, n_in],
                "connection_rule_ko": f"블록 {nr4 // rr}×{nphi4 // rp} 위치 → 1 영역"})
    fam = v4.channel_meta["family"]
    groups = {g: [k for k in range(n_in) if fam[k] == g] for g in ("curve", "straight", "junction", "lineend")}
    groups = {g: v for g, v in groups.items() if v}
    gnames = list(groups)
    gpairs = [(gnames[i], gnames[j]) for i in range(len(gnames)) for j in range(i + 1, len(gnames))]
    neurons: list[tuple[str, Any]] = [("global", c) for c in range(n_in)] + [("rgroup", (R, g)) for R in range(n_reg) for g in gnames] + \
        [("conj", (R, g1, g2)) for R in range(n_reg) for (g1, g2) in gpairs]
    n = len(neurons)
    b2 = CSRBuilder(n, n_reg * n_in)
    op = np.zeros(n, bool)
    xs, ys, rfs, names, roles, fams = [], [], [], [], [], []
    for j, (f, info) in enumerate(neurons):
        if f == "global":
            for R in range(n_reg):
                b2.add(j, R * n_in + info, 1.0 / n_reg)
            xs.append(pre.cx), ys.append(pre.cy), rfs.append(pre.rmax)
            names.append("global_" + v4.channel_names[info])
            roles.append("IT_global_feature")
        elif f == "rgroup":
            R, g = info
            for k in groups[g]:
                b2.add(j, R * n_in + k, 1.0 / len(groups[g]))
            xs.append(px[R]), ys.append(py[R]), rfs.append(prf[R])
            names.append(f"region{R}_{g}")
            roles.append("IT_region_group")
        else:
            R, g1, g2 = info
            for k in groups[g1]:
                b2.add(j, R * n_in + k, 1.0 / len(groups[g1]), "A")
            for k in groups[g2]:
                b2.add(j, R * n_in + k, 1.0 / len(groups[g2]), "B")
            op[j] = True
            xs.append(px[R]), ys.append(py[R]), rfs.append(prf[R])
            names.append(f"region{R}_{g1}_and_{g2}")
            roles.append("IT_conjunction")
        fams.append(f)
    nb1 = single_position_neighbors()
    zero_i = np.zeros(n, np.int32) - 1
    L2 = _spec("IT", "L2", "IT/L4", n_reg * n_in, 1, n, names, roles, _channel_meta(n, family=fams), op, b2.finalize(),
               np.asarray(xs), np.asarray(ys), np.asarray(rfs), zero_i, zero_i, (np.asarray([pre.cx]), np.asarray([pre.cy]),
                                                                                 np.asarray([pre.rmax])), None, nb1, True,
               {"role_ko": "넓은 범위 특징 결합: 전역 채널 평균 (위치 허용), 영역별 특징군 평균, 영역별 특징군 쌍 결합 (min2)",
                "operation": {"global": "u = mean_regions L4(R, c)", "rgroup": "u = mean_{c∈G} L4(R, c)",
                              "conj": "u = min(mean_{c∈G1} L4(R,c), mean_{c∈G2} L4(R,c))"},
                "groups": {g: len(v) for g, v in groups.items()}, "input_shape": [n_reg, n_in], "output_shape": [1, n],
                "connection_rule_ko": "IT 는 수용장이 전 시야라 L6 이웃은 층 전체 (위치 하나로 다룬다)"},
               {"merged_duplicate_edges": b2.merged})
    L3 = _spec("IT", "L3", "IT/L2", n, 1, n, list(names), [r_ + "_L3" for r_ in roles], dict(L2.channel_meta), np.zeros(n, bool),
               identity_csr(n), L2.x, L2.y, L2.rf, zero_i, zero_i, (np.asarray([pre.cx]), np.asarray([pre.cy]), np.asarray([pre.rmax])),
               None, nb1, True,
               {"role_ko": "최종 클래스 비교용 벡터 (일대일 정리)", "operation": "u_j = transmitted_L2_j", "input_shape": [1, n],
                "output_shape": [1, n], "connection_rule_ko": "일대일"})
    return [L4, L2, L3]


class CortexArea:
    """영역 하나 = 고정 구조 L4 → L2 → L3 (LayerSpec 3 개). 하위 클래스 V1/V2/V4/IT 가 역할 연산 (생성기) 을 정한다.
    영역 수준 교정 배분기 (LocalErrorRouter) 와 장치 쪽 배열 (LayerRuntime) 은 CortexModel.add_area 에서 붙인다."""

    name = ""
    role_ko = ""

    def __init__(self, specs: Sequence[LayerSpec]):
        self.specs = list(specs)
        self.l4, self.l2, self.l3 = self.specs

    @classmethod
    def builder(cls, cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", prev_l3: LayerSpec | None,
                context: dict[str, LayerSpec]) -> list[LayerSpec]:
        raise StructureError(f"{cls.__name__} 에 생성기가 없다.")

    @classmethod
    def build(cls, cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", prev_l3: LayerSpec | None,
              context: dict[str, LayerSpec] | None = None) -> "CortexArea":
        if cls.name != "V1" and prev_l3 is None:
            raise StructureError(f"{cls.name} 는 앞 영역 L3 가 있어야 만든다.")
        return cls(cls.builder(cfg, pre, prev_l3, context or {}))

    def describe(self) -> dict[str, Any]:
        return {"area": self.name, "role_ko": self.role_ko, "layers": {sp.name: sp.description for sp in self.specs},
                "h_dim": self.l3.n, "structure_sha256": area_structure_hash(self.specs)}


class V1(CortexArea):
    name = "V1"
    role_ko = "위치별 방향·위상 고정 필터 반응 (L2) 과 위상×극성 국소 결합 (L3)"

    @classmethod
    def builder(cls, cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", prev_l3: LayerSpec | None,
                context: dict[str, LayerSpec]) -> list[LayerSpec]:
        return build_v1(cfg, pre)


class V2(CortexArea):
    name = "V2"
    role_ko = "근처 선 반응의 조합: 윤곽 연속·끝점·교차 (L2), 일대일 정리 (L3)"

    @classmethod
    def builder(cls, cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", prev_l3: LayerSpec | None,
                context: dict[str, LayerSpec]) -> list[LayerSpec]:
        return build_v2(cfg, pre, prev_l3)


class V4(CortexArea):
    name = "V4"
    role_ko = "더 넓은 수용장의 선 조합: 곡선 조각·직선 연속·접합·선 끝 (L2), 격자 이웃 결합 (L3)"

    @classmethod
    def builder(cls, cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", prev_l3: LayerSpec | None,
                context: dict[str, LayerSpec]) -> list[LayerSpec]:
        return build_v4(cfg, pre, prev_l3, context.get("V1/L3"))


class IT(CortexArea):
    name = "IT"
    role_ko = "넓은 범위 특징 결합 (L2) 과 최종 클래스 비교용 벡터 (L3)"

    @classmethod
    def builder(cls, cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", prev_l3: LayerSpec | None,
                context: dict[str, LayerSpec]) -> list[LayerSpec]:
        return build_it(cfg, pre, prev_l3)


AREA_CLASSES: dict[str, type[CortexArea]] = {"V1": V1, "V2": V2, "V4": V4, "IT": IT}


def area_structure_hash(specs: Sequence[LayerSpec]) -> str:
    return sha256_text("|".join(sp.hash() for sp in specs))


# ======================================================================
# 8. 뉴런 저장소 (NeuronStore) 와 조회 (NeuronInspector)
# ======================================================================
class Connectivity:
    """연결 목록. INPUT = 수신 층 LayerSpec 의 CSR (입력군 A/B), NEXT = 원천 층별로 만든 나가는 연결 CSR (송신 local → 수신 local).
    실제 계산에 쓰이는 배열과 같은 것에서 만들므로 NEXT 는 계산 연결과 일치한다. 전체 N×N 행렬을 만들지 않는다."""

    def __init__(self) -> None:
        self.next_csr: dict[str, dict[str, Any]] = {}

    def add(self, sp: LayerSpec, src_sp: LayerSpec) -> None:
        rows_a = np.repeat(np.arange(sp.n, dtype=np.int64), np.diff(sp.a_ptr))
        rows_b = np.repeat(np.arange(sp.n, dtype=np.int64), np.diff(sp.b_ptr))
        src = np.concatenate([sp.a_src, sp.b_src])
        dst = np.concatenate([rows_a, rows_b])
        w = np.concatenate([sp.a_w, sp.b_w])
        grp = np.concatenate([np.zeros(len(rows_a), np.int8), np.ones(len(rows_b), np.int8)])
        order = np.argsort(src, kind="stable")
        cnt = np.bincount(src, minlength=src_sp.n)
        self.next_csr[sp.src_key] = {"dst_key": sp.key, "ptr": np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64),
                                     "dst": dst[order], "w": w[order], "grp": grp[order]}

    def fan_out(self, key: str, n: int) -> np.ndarray:
        nx = self.next_csr.get(key)
        return np.zeros(n, np.int64) if nx is None else np.diff(nx["ptr"])

    def report(self, layers: dict[str, LayerSpec], order: Sequence[str]) -> list[dict[str, Any]]:
        """팬인·팬아웃, 입력 없는 뉴런, 출력 없는 뉴런, 일대일 여부, 중복 (구조 검증에서 0 확인)."""
        rows = []
        for key in order:
            sp = layers[key]
            fi, fo = sp.fan_in(), self.fan_out(key, sp.n)
            rows.append({"layer": key, "fan_in_min": int(fi.min()), "fan_in_max": int(fi.max()), "fan_in_mean": float(fi.mean()),
                         "n_without_input": int((fi == 0).sum()), "fan_out_min": int(fo.min()), "fan_out_max": int(fo.max()),
                         "n_without_output": int((fo == 0).sum()),
                         "without_output_note_ko": ("마지막 영역 L3 는 prototype 비교기로 출력 (연결 아님)" if key not in self.next_csr else None),
                         "one_to_one_input": bool((fi == 1).all() and sp.n == sp.n_src),
                         "duplicates_remaining": 0, "merged_duplicates_at_build": int(sp.build_stats.get("merged_duplicate_edges", 0))})
        return rows


class NeuronStore:
    """모든 뉴런의 연속 저장소.
    CPU (numpy): 좌표 (N,3) float32 [X, Y, Z], 영역·층·역할·채널·위치 코드 (정수), 수용장 크기.
    장치 텐서: 임계값 (N,), q (N,) — 층마다 neuron_id 범위 [start, end) 의 연속 슬라이스 (view).
    CONF·전송 출력은 배치마다 계산하는 동적 텐서 (B, n_layer) 이다 (표본 사이에 남기지 않는다).
    NEXT (나가는 연결) 는 원천 층별 CSR, INPUT (들어오는 연결) 은 수신 층 CSR 이다. 주소는 int64 (float 에 넣지 않는다)."""

    def __init__(self, be: Backend, cfg: dict[str, Any]):
        self.be, self.cfg = be, cfg
        self.layers: dict[str, LayerSpec] = {}
        self.order: list[str] = []
        self.N = 0
        self.coords = np.zeros((0, 3), np.float32)
        self.area_code = np.zeros(0, np.int8)
        self.layer_code = np.zeros(0, np.int8)
        self.role_code = np.zeros(0, np.int32)
        self.channel = np.zeros(0, np.int32)
        self.pos_id = np.zeros(0, np.int32)
        self.roles: list[str] = []
        self.q = be.f32(np.zeros(0))
        self.thr = be.f32(np.zeros(0))
        self.conn = Connectivity()
        self.next_csr = self.conn.next_csr

    def add_layers(self, specs: Sequence[LayerSpec]) -> None:
        ne = self.cfg["neuron"]
        coords, ac, lc, rc, chs, ps = [], [], [], [], [], []
        for sp in specs:
            if sp.key in self.layers:
                raise StructureError(f"{sp.key} 가 이미 있다.")
            sp.start, sp.end = self.N, self.N + sp.n
            self.N = sp.end
            self.layers[sp.key] = sp
            self.order.append(sp.key)
            coords.append(np.stack([sp.x, sp.y, np.full(sp.n, sp.z, np.float32)], axis=1))
            ac.append(np.full(sp.n, AREA_INDEX[sp.area], np.int8))
            lc.append(np.full(sp.n, LAYER_NUMBER[sp.name], np.int8))
            codes = []
            for r in sp.channel_roles:
                if r not in self.roles:
                    self.roles.append(r)
                codes.append(self.roles.index(r))
            rc.append(np.tile(np.asarray(codes, np.int32), sp.n_pos))
            chs.append(np.tile(np.arange(sp.n_ch, dtype=np.int32), sp.n_pos))
            ps.append(np.repeat(np.arange(sp.n_pos, dtype=np.int32), sp.n_ch))
            if sp.src_key != "LGN":
                src_sp = self.layers.get(sp.src_key)
                if src_sp is None:
                    raise StructureError(f"{sp.key} 의 원천 층 {sp.src_key} 가 아직 없다.")
                self.conn.add(sp, src_sp)
        n_new = sum(sp.n for sp in specs)
        self.coords = np.concatenate([self.coords] + coords).astype(np.float32)
        self.area_code = np.concatenate([self.area_code] + ac)
        self.layer_code = np.concatenate([self.layer_code] + lc)
        self.role_code = np.concatenate([self.role_code] + rc)
        self.channel = np.concatenate([self.channel] + chs)
        self.pos_id = np.concatenate([self.pos_id] + ps)
        self.q = self.be.cat0([self.q, self.be.f32(np.full(n_new, float(ne["q_init"])))])
        self.thr = self.be.cat0([self.thr, self.be.f32(np.full(n_new, float(ne["threshold"])))])

    def q_view(self, key: str) -> Any:
        sp = self.layers[key]
        return self.q[sp.start:sp.end]

    def thr_view(self, key: str) -> Any:
        sp = self.layers[key]
        return self.thr[sp.start:sp.end]

    def q_numpy(self) -> np.ndarray:
        return self.be.to_np(self.q).astype(np.float32).copy()

    def load_q(self, arr: np.ndarray) -> None:
        arr = np.asarray(arr, np.float32)
        ne = self.cfg["neuron"]
        if arr.shape != (self.N,):
            raise CompatibilityError(f"저장된 q 길이 {arr.shape} != 현재 뉴런 수 {self.N}")
        if not np.isfinite(arr).all() or arr.min() < float(ne["q_min"]) - 1e-7 or arr.max() > float(ne["q_max"]) + 1e-7:
            raise CompatibilityError("저장된 q 에 NaN/Inf 또는 범위 밖 값이 있다.")
        self.be.assign_(self.q, self.be.f32(arr))

    def locate(self, nid: int) -> tuple[str, int]:
        if not (0 <= int(nid) < self.N):
            raise IndexError(f"neuron_id {nid} 가 범위 [0,{self.N}) 밖이다.")
        for key in self.order:
            sp = self.layers[key]
            if sp.start <= nid < sp.end:
                return key, int(nid - sp.start)
        raise IndexError(nid)

    def ids_sha(self, key: str) -> str:
        sp = self.layers[key]
        return sha256_array(np.arange(sp.start, sp.end, dtype=np.int64))

    def fan_out(self, key: str) -> np.ndarray:
        return self.conn.fan_out(key, self.layers[key].n)


class NeuronInspector:
    """inspect_neuron(neuron_id): 3×3 표기와 INPUT/NEXT 연결 목록을 저장소에서 복원한다.
    [ X 좌표,         Y 좌표,        Z 좌표          ]
    [ NEXT 참조,      고정 임계값,    INPUT 참조      ]
    [ 실제 전송 출력,  CONF,          지속 출력 q     ]   (전송 출력·CONF 는 주어진 표본의 동적 값, 없으면 None)"""

    def __init__(self, store: NeuronStore):
        self.store = store

    def inspect_neuron(self, nid: int, dyn: dict[str, dict[str, np.ndarray]] | None = None, max_edges: int = 40) -> dict[str, Any]:
        st = self.store
        key, j = st.locate(int(nid))
        sp = st.layers[key]
        row = sp.row(j)
        src_sp = st.layers.get(sp.src_key)
        base = src_sp.start if src_sp is not None else None

        def addr(local: int) -> Any:
            return int(base + local) if base is not None else f"LGN[{int(local)}]"

        inputs = [{"src": addr(s), "w": float(w), "group": "A"} for s, w in zip(row["a_src"], row["a_w"])] + \
                 [{"src": addr(s), "w": float(w), "group": "B"} for s, w in zip(row["b_src"], row["b_w"])]
        nx = st.next_csr.get(key)
        nexts = []
        if nx is not None:
            dsp = st.layers[nx["dst_key"]]
            a_, b_ = int(nx["ptr"][j]), int(nx["ptr"][j + 1])
            nexts = [{"dst": int(dsp.start + d), "w": float(w), "group": "AB"[int(g)]}
                     for d, w, g in zip(nx["dst"][a_:b_], nx["w"][a_:b_], nx["grp"][a_:b_])]
        routing = None
        if sp.routing is not None and sp.routed is not None and bool(sp.routed[j]):
            t = sp.routing
            ci = int(np.nonzero(t.curve_rows == j)[0][0])
            l4 = st.layers[sp.src_key]
            routing = {"kind": "routed_curve_neuron", "mode": t.mode, "P_position": int(t.P[ci]), "theta_A_deg": float(t.theta_a[ci]),
                       "theta_B_deg": float(t.theta_b[ci]), "curve_sign": int(t.sign[ci]), "expected_chord_deg": float(t.phi_c[ci]),
                       "D_px": float(t.D[ci]), "missing_candidate": bool(t.missing[ci]), "legacy_Q_position": int(t.legacy_q[ci]),
                       "static_candidates": [{"slot": s_, "Q_position": int(t.cand_q[ci, s_]), "src_neuron_id": int(l4.start + t.cand_src[ci, s_]),
                                              "dist_px": float(t.cand_dist[ci, s_]), "dir_err_deg": float(t.cand_dir_err[ci, s_]),
                                              "geo_score": float(t.cand_geo[ci, s_]), "missing_guide": bool(t.missing_guide[ci, s_]),
                                              "valid_path_points": int(t.pt_valid[ci, s_].sum())} for s_ in range(t.K) if t.cand_valid[ci, s_]],
                       "note_ko": "정적 후보 목록이다. 실제 B 입력은 이미지마다 이 중 하나 (동적 선택)."}
            if dyn and "__routing__" in dyn:
                rr = dyn["__routing__"]
                k_ = int(rr["k"][ci])
                routing["selected_in_this_sample"] = {
                    "slot": None if t.missing[ci] else k_, "Q_position": None if t.missing[ci] else int(t.cand_q[ci, k_]),
                    "guide_scores": (rr["S"][ci].tolist() if rr.get("S") is not None else None), "no_guidance": bool(rr["no_guidance"][ci]),
                    "A": float(rr["A"][ci]), "B": float(rr["B"][ci]), "u": 0.0 if t.missing[ci] else float(min(rr["A"][ci], rr["B"][ci])),
                    "guide_score_used_for_selection": bool(rr.get("S_used_for_selection", False)),
                    "selected_path_point_r": (rr["R_sel"][ci].tolist() if rr.get("R_sel") is not None else None),
                    "selected_path_point_valid": (rr["OK_sel"][ci].tolist() if rr.get("OK_sel") is not None else None)}
        nxt_l2 = st.layers.get(f"{sp.area}/L2") if sp.name == "L4" else None
        if nxt_l2 is not None and nxt_l2.routing is not None:
            t = nxt_l2.routing
            hit = np.argwhere((t.cand_src == j) & t.cand_valid)
            routing = {"kind": "routing_candidate_source", "mode": t.mode,
                       "candidate_for": [{"curve_neuron_id": int(nxt_l2.start + t.curve_rows[c_]), "slot": int(s_)} for c_, s_ in hit[:max_edges]],
                       "n_candidate_refs": int(len(hit)),
                       "note_ko": "정적 NEXT 와 별도로, 이 뉴런은 위 곡선 뉴런들의 B 입력 후보다 (이미지마다 선택될 때만 실제로 전달)."}
        q = float(st.be.to_np(st.q[int(nid):int(nid) + 1])[0])
        thr = float(st.be.to_np(st.thr[int(nid):int(nid) + 1])[0])
        conf = trans = None
        if dyn and key in dyn:
            conf = float(dyn[key]["conf"][j]) if "conf" in dyn[key] else None
            trans = float(dyn[key]["out"][j]) if "out" in dyn[key] else None
        ch = j % sp.n_ch
        meta = {k: v[ch] for k, v in sp.channel_meta.items()}
        X, Y, Z = (float(v) for v in st.coords[int(nid)])
        nxt_ref = (f"NEXT→{len(nexts)} edges ({nx['dst_key']})" if nx is not None else
                   ("NEXT→0 (활성 영역 L3: prototype 비교기로 출력, 연결 아님)" if sp.name == "L3" else "NEXT→0"))
        inp_ref = f"INPUT←{len(inputs)} edges ({sp.src_key})"
        if routing is not None and routing["kind"] == "routed_curve_neuron":
            inp_ref = f"INPUT←{len(inputs)} 정적 edge + 동적 B {0 if routing['missing_candidate'] else 1} (정적 후보 {len(routing['static_candidates'])}) ({sp.src_key})"
        elif routing is not None and routing["kind"] == "routing_candidate_source":
            nxt_ref += f" + 곡선 B 후보 {routing['n_candidate_refs']} (동적)"
        matrix = [[X, Y, Z], [nxt_ref, thr, inp_ref], [trans, conf, q]]
        return {"neuron_id": int(nid), "layer": key, "local_index": j, "position": j // sp.n_ch, "channel": ch,
                "channel_name": sp.channel_names[ch], "role_tag": sp.channel_roles[ch], "channel_meta": meta,
                "grid_position": [int(sp.ring[j]), int(sp.ang[j])], "receptive_field_px": float(sp.rf[j]),
                "area": sp.area, "layer_name": sp.name, "preferred_orientation_deg": meta.get("orientation_deg"),
                "phase_rad": meta.get("phase_rad"), "polarity": meta.get("polarity"),
                "h_index_if_L3": (j if sp.name == "L3" else None),
                "op": "min2" if bool(sp.op_min[j]) else "linear", "matrix_3x3": matrix,
                "n_input": len(inputs), "n_next": len(nexts), "inputs": inputs[:max_edges], "nexts": nexts[:max_edges], "routing": routing,
                "coordinate_units_ko": "X,Y = 처리 영상 화소 (원점 왼쪽 위, x 오른쪽, y 아래); Z = 영역번호×10+층번호 (무차원)"}

    @staticmethod
    def format(info: dict[str, Any]) -> str:
        m = info["matrix_3x3"]
        lines = [f"neuron {info['neuron_id']}  {info['layer']} local {info['local_index']}  역할 {info['role_tag']}  "
                 f"채널 {info['channel_name']}  연산 {info['op']}",
                 f"[ X={fmt(m[0][0], 2)}, Y={fmt(m[0][1], 2)}, Z={fmt(m[0][2], 1)} ]",
                 f"[ {m[1][0]}, 임계값={fmt(m[1][1])}, {m[1][2]} ]",
                 f"[ 전송 출력={fmt(m[2][0])}, CONF={fmt(m[2][1])}, q={fmt(m[2][2])} ]",
                 f"  격자 위치 {info['grid_position']}, 수용장 {fmt(info['receptive_field_px'], 2)} px, 채널 메타 {info['channel_meta']}",
                 f"  INPUT (처음 {len(info['inputs'])}/{info['n_input']}): " +
                 ", ".join(f"{e['src']}:{e['w']:+.4f}{e['group']}" for e in info["inputs"][:12]) + (" ..." if info["n_input"] > 12 else ""),
                 f"  NEXT (처음 {len(info['nexts'])}/{info['n_next']}): " +
                 ", ".join(f"{e['dst']}:{e['w']:+.4f}{e['group']}" for e in info["nexts"][:12]) + (" ..." if info["n_next"] > 12 else "")]
        ro = info.get("routing")
        if ro and ro["kind"] == "routed_curve_neuron":
            lines.append(f"  라우팅 ({ro['mode']}): P={ro['P_position']} θA={ro['theta_A_deg']:g} θB={ro['theta_B_deg']:g} 부호 {ro['curve_sign']} "
                         f"현 방향 {ro['expected_chord_deg']:g}° D={ro['D_px']:.2f}px, 후보 없음={ro['missing_candidate']}, legacy Q={ro['legacy_Q_position']}")
            lines.append("  정적 B 후보: " + (", ".join(f"[{c['slot']}] Q{c['Q_position']}→{c['src_neuron_id']} d={c['dist_px']:.1f} "
                                                    f"Δ={c['dir_err_deg']:.1f}°" for c in ro["static_candidates"]) or "없음 (u = 0)"))
            if "selected_in_this_sample" in ro:
                s_ = ro["selected_in_this_sample"]
                lines.append(f"  이번 표본의 실제 선택: slot {s_['slot']} (Q{s_['Q_position']}), no_guidance={s_['no_guidance']}, "
                             f"A={s_['A']:.4g}, B={s_['B']:.4g}, u=min={s_['u']:.4g}, 점수 {s_['guide_scores']}")
        elif ro:
            lines.append(f"  라우팅 후보 원천 ({ro['mode']}): {ro['n_candidate_refs']} 개 곡선 뉴런의 B 후보 (동적) — "
                         + ", ".join(f"{c['curve_neuron_id']}[{c['slot']}]" for c in ro["candidate_for"][:10]))
        return "\n".join(lines)


# ======================================================================
# 7b. V1 안내 라우팅 — V4 L2 곡선 뉴런의 두 번째 입력 위치를 이미지마다 제한된 후보 안에서 고른다 (동적 라우팅)
#     후보 목록 자체는 구조 생성 때 한 번 만들고 바꾸지 않는다 (영구 재배선이 아니다).
# ======================================================================
ROUTING_MODES: tuple[str, ...] = ("legacy", "geometry_only", "v1_guided")
PATH_REASONS = {0: "ok", 1: "curvature_sign", 2: "folded", 3: "zero_tangent", 4: "outside_rf", 5: "direction_conflict"}
ROUTING_CONVENTION_KO = {
    "frame": "기하 계산은 시각 좌표 (vx = x − cx 오른쪽, vy = cy − y 위쪽). 화소 배열의 y 는 아래로 증가하므로 부호를 바꾼다.",
    "angle": "각도는 +x 에서 반시계 (위쪽 y 기준) 양의 방향, 단위 도.",
    "channel_theta": "V1 방향 채널 θ 는 선의 접선 방향 (θ 방향으로 놓인 선에 가장 크게 반응) 이며 180 도 주기다 (자체 검사 R00 에서 합성 직선으로 확인).",
    "direction_vs_orientation": "공간 이동 방향 φ (360 도, 앞/뒤 구분) 와 선 방향 θ (180 도 주기) 를 별도 변수로 다룬다.",
    "curve_forward": "곡선 채널 (θA → θB, 부호 s, 곡률 κ) 은 P 에서 +θA 방향으로 나아가며 접선이 s·κ 만큼 돈다: φA = θA, φB = θA + s·κ (연속 각도), "
                     "기대 현 방향 φC = θA + s·κ/2. 0/180 도 이음매를 산술 평균하지 않는다.",
    "interpolation": "원영상 점 → 로그-극좌표 역사상 (u = log(1+r/r0)/log(1+Rmax/r0)) → (반경 bin, 각도 bin) 쌍선형 보간. 반경은 안쪽·바깥 고리에서 "
                     "자르고 (r > Rmax 는 무효), 각도는 wrap. 방향 채널은 180 도 주기 선형 보간.",
}


def wrap180(x: Any) -> Any:
    return (np.asarray(x, np.float64) + 180.0) % 360.0 - 180.0


def d_ori(a: Any, b: Any) -> Any:
    """180 도 주기 선 방향 차: min(|a−b| mod 180, 180 − (|a−b| mod 180))."""
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)) % 180.0
    return np.minimum(d, 180.0 - d)


def d_dir(a: Any, b: Any) -> Any:
    """360 도 이동 방향 차 (앞/뒤 구분)."""
    return np.abs(wrap180(np.asarray(a, np.float64) - np.asarray(b, np.float64)))


def curve_directions(theta_a: float, sign: int, kappa: float) -> tuple[float, float, float]:
    """연속 각도 표현: φA = θA, φB = θA + s·κ, φC = θA + s·κ/2 (도)."""
    return float(theta_a), float(theta_a + sign * kappa), float(theta_a + sign * kappa / 2.0)


def hermite_path(P: np.ndarray, Q: np.ndarray, phi_a: float, phi_b: float, M: int) -> tuple[np.ndarray, np.ndarray]:
    """cubic Hermite: p(0)=P, p(1)=Q, vA = ‖Q−P‖·tA, vB = ‖Q−P‖·tB.
    p(t) = (2t³−3t²+1)P + (t³−2t²+t)vA + (−2t³+3t²)Q + (t³−t²)vB ;  p'(t) 는 이 기하 경로의 접선 (학습 미분이 아니다)."""
    t = np.linspace(0.0, 1.0, int(M))[:, None]
    L = float(np.hypot(*(Q - P)))
    tA = np.asarray([math.cos(math.radians(phi_a)), math.sin(math.radians(phi_a))])
    tB = np.asarray([math.cos(math.radians(phi_b)), math.sin(math.radians(phi_b))])
    vA, vB = L * tA, L * tB
    p = (2 * t ** 3 - 3 * t ** 2 + 1) * P + (t ** 3 - 2 * t ** 2 + t) * vA + (-2 * t ** 3 + 3 * t ** 2) * Q + (t ** 3 - t ** 2) * vB
    dp = (6 * t ** 2 - 6 * t) * P + (3 * t ** 2 - 4 * t + 1) * vA + (-6 * t ** 2 + 6 * t) * Q + (3 * t ** 2 - 2 * t) * vB
    return p, dp


def path_check(p: np.ndarray, dp: np.ndarray, P: np.ndarray, Q: np.ndarray, phi_a: float, phi_b: float, sign: int,
               tol_deg: float, rf_p: float, rf_q: float) -> tuple[int, float]:
    """경로 유효성 코드와 총 회전각. 1 곡률 부호 불일치·변곡, 2 접힘 (총 |회전| > 180), 3 영 접선, 4 수용장 밖, 5 진행 방향 충돌."""
    L = float(np.hypot(*(Q - P)))
    ch = Q - P
    if L <= 0:
        return 5, 0.0
    for phi in (phi_a, phi_b):
        if ch[0] * math.cos(math.radians(phi)) + ch[1] * math.sin(math.radians(phi)) <= 0:
            return 5, 0.0                                           # 접선이 P→Q 진행과 반대
    nrm = np.hypot(dp[:, 0], dp[:, 1])
    if (nrm < 1e-9 * L).any():
        return 3, 0.0
    psi = np.degrees(np.arctan2(dp[:, 1], dp[:, 0]))
    dpsi = wrap180(np.diff(psi))
    total = float(dpsi.sum())
    if (sign * dpsi < -tol_deg).any() or sign * total <= 0:
        return 1, total
    if float(np.abs(dpsi).sum()) > 180.0:
        return 2, total
    in_rf = (np.hypot(*(p - P).T) <= rf_p) | (np.hypot(*(p - Q).T) <= rf_q)
    if not in_rf.all():
        return 4, total
    return 0, total


class GuideSampler:
    """V1 L3 실제 전송 출력 (휘도 채널) 을 원영상 점·선 방향에서 읽는 보간 표.
    a(점, θ) = Σ_{4 격자 위치} Σ_{2 방향 채널} w·F1 ,  b(점) = Σ_{4 격자 위치} w·Σ_θ F1  (같은 공간 가중치)."""

    def __init__(self, v1_l3: LayerSpec, pre: "RetinaLGNPreprocessor", cfg: dict[str, Any]):
        if v1_l3.grid_shape is None:
            raise StructureError("V1 L3 격자 정보가 없다.")
        self.n_r, self.n_phi = v1_l3.grid_shape
        self.n_pos, self.n_ch, self.n = v1_l3.n_pos, v1_l3.n_ch, v1_l3.n
        oris = np.asarray([float(o) for o in cfg["areas"]["V1"]["orientations_deg"]], np.float64)
        self.n_o = len(oris)
        self.d_o = 180.0 / self.n_o
        if not np.allclose(oris, oris[0] + self.d_o * np.arange(self.n_o), atol=1e-6):
            raise StructureError("V1 방향 채널이 오름차순 180/n 등간격이 아니어서 방향 보간을 지원하지 않는다 (v1_guided 미지원).")
        self.ori0 = float(oris[0])
        colors = []
        for c in v1_l3.channel_meta["color"]:
            if c not in colors:
                colors.append(c)
        if "lum" not in colors:
            raise StructureError("V1 L3 에 휘도 (lum) 채널이 없어 v1_guided 안내를 지원하지 않는다 (임의 채널을 고르지 않는다).")
        self.lum_base = colors.index("lum") * self.n_o
        self.r0, self.L, self.rmax, self.off = pre.r0, pre.grid["L"], pre.rmax, pre.phi_offset
        self.cx, self.cy = pre.cx, pre.cy

    def table(self, vx: np.ndarray, vy: np.ndarray, ori_deg: np.ndarray) -> tuple[np.ndarray, ...]:
        vx, vy, ori_deg = (np.asarray(a, np.float64) for a in (vx, vy, ori_deg))
        r = np.hypot(vx, vy)
        valid = r <= self.rmax
        kt = np.log1p(r / self.r0) / self.L * self.n_r - 0.5
        k0 = np.floor(kt)
        wk = kt - k0
        inner = k0 < 0
        k0 = np.where(inner, 0, k0)
        wk = np.where(inner, 0.0, wk)
        outer = k0 >= self.n_r - 1
        k0 = np.where(outer, self.n_r - 1, k0).astype(np.int64)
        wk = np.where(outer, 0.0, wk)
        k1 = np.minimum(k0 + 1, self.n_r - 1)
        phi = np.mod(np.arctan2(vy, vx), 2 * math.pi)
        mt = phi * self.n_phi / (2 * math.pi) - self.off
        m0f = np.floor(mt)
        wm = mt - m0f
        m0 = np.mod(m0f, self.n_phi).astype(np.int64)
        m1 = np.mod(m0 + 1, self.n_phi)
        pos = np.stack([k0 * self.n_phi + m0, k0 * self.n_phi + m1, k1 * self.n_phi + m0, k1 * self.n_phi + m1], axis=-1)
        wp = np.stack([(1 - wk) * (1 - wm), (1 - wk) * wm, wk * (1 - wm), wk * wm], axis=-1)
        o = np.mod(ori_deg - self.ori0, 180.0) / self.d_o
        c0f = np.floor(o)
        wc = o - c0f
        c0 = np.mod(c0f, self.n_o).astype(np.int64)
        c1 = np.mod(c0 + 1, self.n_o)
        cc = np.stack([c0, c1], axis=-1)
        wcc = np.stack([1 - wc, wc], axis=-1)
        idx_a = (pos[..., :, None] * self.n_ch + self.lum_base + cc[..., None, :]).reshape(pos.shape[:-1] + (8,))
        w_a = (wp[..., :, None] * wcc[..., None, :]).reshape(pos.shape[:-1] + (8,))
        idx_a = np.where(valid[..., None], idx_a, self.n)
        w_a = np.where(valid[..., None], w_a, 0.0)
        idx_b = np.where(valid[..., None], pos, self.n_pos)
        w_b = np.where(valid[..., None], wp, 0.0)
        return idx_a.astype(np.int64), w_a.astype(np.float32), idx_b.astype(np.int64), w_b.astype(np.float32), valid


@dataclass
class RoutingTable:
    """곡선 뉴런별 고정 후보 표 (구조 생성 때 한 번). slot 0..K−1 은 기하 점수 순서 (같은 점수는 원래 입력 주소 순서) 의 유효 후보,
    빈 slot 은 마스크로 표시하고 주소는 0 패딩 열 (n_src) 을 가리킨다 (−1 을 넣지 않는다)."""

    mode: str
    K: int
    M: int
    params: dict[str, Any]
    curve_rows: np.ndarray          # (nc,) V4 L2 local 행
    P: np.ndarray                   # (nc,) V4 L4 위치
    theta_a: np.ndarray
    theta_b: np.ndarray
    sign: np.ndarray
    kappa: np.ndarray
    phi_c: np.ndarray
    D: np.ndarray
    srcA: np.ndarray                # (nc,) 첫 입력 주소 (V4 L4 local)
    cand_q: np.ndarray              # (nc,K) 후보 위치 (빈 slot −1, 기록용)
    cand_src: np.ndarray            # (nc,K) 후보 입력 주소 (빈 slot n_src)
    cand_valid: np.ndarray
    cand_dist: np.ndarray
    cand_dir_err: np.ndarray
    cand_geo: np.ndarray
    path_xy: np.ndarray             # (nc,K,M,2) 화소 좌표
    path_ori: np.ndarray            # (nc,K,M) 경로 접선의 선 방향 (180 주기)
    gA_idx: np.ndarray
    gA_w: np.ndarray
    gB_idx: np.ndarray
    gB_w: np.ndarray
    pt_valid: np.ndarray            # (nc,K,M)
    missing: np.ndarray             # (nc,) 유효 후보 없음
    missing_guide: np.ndarray       # (nc,K) 기하적으로 유효하지만 경로 위에 유효 V1 지도 지점이 하나도 없는 후보 (S 는 늘 0)
    n_geometric: np.ndarray         # (nc,) 거리·방향 조건 통과 수 (경로 검사·K 상한 전)
    n_path_rejected: np.ndarray     # (nc,) 경로 검사 탈락 수
    rejected_q: np.ndarray          # (nc,R) 경로 탈락 후보 위치 (그림·기록용, −1 빈칸)
    rejected_reason: np.ndarray     # (nc,R)
    n_rejected_by_reason: np.ndarray  # (nc,6) 사유별 전체 탈락 수 (열 = 사유 코드, 0 열은 쓰지 않음)
    legacy_q: np.ndarray            # (nc,) 원본 규칙 (최근접, 없으면 같은 위치) 의 위치 — 비교 기록용
    n_src: int
    n_v1: int
    n_pos1: int

    def hash(self) -> str:
        return sha256_text(dumps({"mode": self.mode, "K": self.K, "M": self.M, "params": self.params}, indent=None) + sha256_array(
            self.curve_rows, self.P, self.cand_src, self.cand_valid, self.gA_idx, self.gA_w, self.gB_idx, self.gB_w, self.pt_valid,
            self.missing))

    def arrays(self) -> dict[str, np.ndarray]:
        out = {}
        for f_ in dataclasses.fields(self):
            v = getattr(self, f_.name)
            if isinstance(v, np.ndarray):
                out[f_.name] = v
        return out

    def static_summary(self, l4: LayerSpec, pre: "RetinaLGNPreprocessor") -> dict[str, Any]:
        nc = len(self.curve_rows)
        ring = l4.ring[self.P * l4.n_ch]
        ecc = np.hypot(l4.pos_x[self.P] - pre.cx, pre.cy - l4.pos_y[self.P])
        by_ring = {int(k): {"n": int((ring == k).sum()), "missing_frac": float(self.missing[ring == k].mean())} for k in np.unique(ring)}
        by_ori = {f"{float(t):g}": {"n": int((self.theta_a == t).sum()), "missing_frac": float(self.missing[self.theta_a == t].mean())}
                  for t in np.unique(self.theta_a)}
        by_sign = {int(s): float(self.missing[self.sign == s].mean()) for s in np.unique(self.sign)}
        nv = self.cand_valid.sum(1)
        return {"mode": self.mode, "n_curve_neurons": nc, "K": self.K, "M": self.M, "params": self.params,
                "missing_candidate_frac": float(self.missing.mean()) if nc else None, "n_missing": int(self.missing.sum()),
                "valid_candidates_mean": float(nv.mean()) if nc else None, "valid_candidates_max": int(nv.max()) if nc else 0,
                "geometric_before_cap_mean": float(self.n_geometric.mean()) if nc else None,
                "capped_by_K_frac": float((self.n_geometric - self.n_path_rejected > self.K).mean()) if nc else None,
                "path_rejected_total": int(self.n_path_rejected.sum()),
                "path_reject_reason_rule_ko": "검사 순서 (진행 방향 → 영 접선 → 곡률 부호 → 접힘 → 수용장) 에서 처음 걸린 사유 하나로 센다",
                "missing_guide_candidates": int(self.missing_guide.sum()),
                "missing_guide_candidate_frac": float(self.missing_guide.sum() / max(1, self.cand_valid.sum())),
                "neurons_all_candidates_missing_guide": int(((self.missing_guide.sum(1) == self.cand_valid.sum(1)) & ~self.missing).sum()),
                "valid_path_point_frac_static": float(self.pt_valid[self.cand_valid].mean()) if self.cand_valid.any() else None,
                "path_reject_reasons": {PATH_REASONS[int(k)]: int(self.n_rejected_by_reason[:, k].sum()) for k in range(1, 6)},
                "missing_by_ring": by_ring, "missing_by_theta_a": by_ori, "missing_by_curve_sign": by_sign,
                "eccentricity_px": {"min": float(ecc.min()), "max": float(ecc.max())} if nc else None,
                "legacy_same_position_fallback_frac": float((self.legacy_q == self.P).mean()) if nc else None,
                "geometry_first_differs_from_legacy_frac": float(((self.cand_q[:, 0] != self.legacy_q) & ~self.missing).sum() / max(1, (~self.missing).sum())),
                "extra_gather_elements_per_sample": int(nc * self.K * self.M * 12), "table_sha256": self.hash(),
                "note_ko": "후보 없음은 V4 격자가 성기거나 거리·방향 조건을 만족하는 위치가 없어서다. V1 안내로 해결되지 않으며 학습 실패와 구분한다."}


def build_routing_table(cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", l4: LayerSpec, n_ch2: int,
                        curve_info: list[tuple[int, int, int, int, float]], contour: dict[int, int], D: np.ndarray,
                        v1_l3: LayerSpec, mode: str) -> RoutingTable:
    """curve_info: (V4 L2 채널, θA 인덱스, θB 인덱스, 곡률 부호, κ). 후보 = 같지 않은 V4 L4 위치 중 거리 [dmin·D, dmax·D],
    이동 방향이 기대 현 방향 φC 에서 max_dev 이내 → 기하 점수 (|d−D|/(반폭·D) + 방향오차/max_dev) 와 주소 순 정렬 → 경로 검사 통과만 K 개."""
    rc = cfg["routing"]
    oris = [float(t) for t in cfg["areas"]["V1"]["orientations_deg"]]
    K, M = int(rc["max_candidates"]), int(rc["path_points"])
    dmin_f, dmax_f, dev = float(rc["dist_min_factor"]), float(rc["dist_max_factor"]), float(rc["max_chord_dev_deg"])
    tol = float(rc["curvature_turn_tol_deg"])
    R = 8
    gs = GuideSampler(v1_l3, pre, cfg)
    px, py = l4.pos_x.astype(np.float64), l4.pos_y.astype(np.float64)
    vx, vy = px - pre.cx, pre.cy - py
    rf = l4.pos_rf.astype(np.float64)
    n_pos = l4.n_pos
    n_in = l4.n_ch
    rows, Ps, ta, tb, jb, sg, kp, pc, Dl, sA = [], [], [], [], [], [], [], [], [], []
    for P in range(n_pos):
        for (ch, i, j, s, kap) in curve_info:
            rows.append(P * n_ch2 + ch)
            Ps.append(P)
            ta.append(oris[i])
            tb.append(oris[j])
            jb.append(j)
            sg.append(s)
            kp.append(kap)
            pc.append(curve_directions(oris[i], s, kap)[2])
            Dl.append(D[P])
            sA.append(P * n_in + contour[i])
    nc = len(rows)
    cand_q = np.full((nc, K), -1, np.int64)
    cand_src = np.full((nc, K), l4.n, np.int64)
    cand_valid = np.zeros((nc, K), bool)
    cand_dist = np.zeros((nc, K), np.float32)
    cand_dir = np.zeros((nc, K), np.float32)
    cand_geo = np.full((nc, K), np.inf, np.float32)
    path_xy = np.zeros((nc, K, M, 2), np.float32)
    path_ori = np.zeros((nc, K, M), np.float32)
    gA_idx = np.full((nc, K, M, 8), gs.n, np.int64)
    gA_w = np.zeros((nc, K, M, 8), np.float32)
    gB_idx = np.full((nc, K, M, 4), gs.n_pos, np.int64)
    gB_w = np.zeros((nc, K, M, 4), np.float32)
    pt_valid = np.zeros((nc, K, M), bool)
    n_geo = np.zeros(nc, np.int64)
    n_rej = np.zeros(nc, np.int64)
    rej_q = np.full((nc, R), -1, np.int64)
    rej_r = np.zeros((nc, R), np.int8)
    rej_n = np.zeros((nc, 6), np.int64)
    legacy_q = np.zeros(nc, np.int64)
    idxQ = np.arange(n_pos)
    for c in range(nc):
        P, s, Dp = Ps[c], sg[c], Dl[c]
        phi_a, phi_b, phi_c = curve_directions(ta[c], s, kp[c])
        lq = _displaced(px, py, P, ta[c], Dp, +1)
        legacy_q[c] = P if lq is None else lq
        dx, dy = vx - vx[P], vy - vy[P]
        d = np.hypot(dx, dy)
        psi = np.degrees(np.arctan2(dy, dx))
        derr = d_dir(psi, phi_c)
        ok = (idxQ != P) & (d >= dmin_f * Dp) & (d <= dmax_f * Dp) & (derr <= dev)
        qs = idxQ[ok]
        n_geo[c] = qs.size
        if qs.size == 0:
            continue
        half = 0.5 * (dmax_f - dmin_f) * Dp
        geo = np.abs(d[qs] - Dp) / max(half, 1e-12) + derr[qs] / dev
        src = qs * n_in + contour[jb[c]]
        order = np.lexsort((src, geo))
        slot, nr = 0, 0
        Pv = np.asarray([vx[P], vy[P]])
        for o_ in order:
            Q = int(qs[o_])
            Qv = np.asarray([vx[Q], vy[Q]])
            p, dp = hermite_path(Pv, Qv, phi_a, phi_b, M)
            code, _tot = path_check(p, dp, Pv, Qv, phi_a, phi_b, s, tol, rf[P], rf[Q])
            if code != 0:
                n_rej[c] += 1
                rej_n[c, code] += 1
                if nr < R:
                    rej_q[c, nr], rej_r[c, nr] = Q, code
                    nr += 1
                continue
            if slot >= K:
                continue
            ori_m = np.mod(np.degrees(np.arctan2(dp[:, 1], dp[:, 0])), 180.0)
            ia, wa, ib, wb, val = gs.table(p[:, 0], p[:, 1], ori_m)
            cand_q[c, slot], cand_src[c, slot], cand_valid[c, slot] = Q, int(src[o_]), True
            cand_dist[c, slot], cand_dir[c, slot], cand_geo[c, slot] = d[Q], derr[Q], geo[o_]
            path_xy[c, slot, :, 0] = p[:, 0] + pre.cx
            path_xy[c, slot, :, 1] = pre.cy - p[:, 1]
            path_ori[c, slot] = ori_m
            gA_idx[c, slot], gA_w[c, slot], gB_idx[c, slot], gB_w[c, slot], pt_valid[c, slot] = ia, wa, ib, wb, val
            slot += 1
    missing = ~cand_valid.any(axis=1)
    missing_guide = cand_valid & ~pt_valid.any(axis=-1)
    params = {k: rc[k] for k in ("max_candidates", "dist_min_factor", "dist_max_factor", "max_chord_dev_deg", "path_points",
                                 "curvature_turn_tol_deg", "guide_epsilon", "guide_min_total", "guide_source", "guide_color")}
    return RoutingTable(mode=mode, K=K, M=M, params=params, curve_rows=np.asarray(rows, np.int64), P=np.asarray(Ps, np.int64),
                        theta_a=np.asarray(ta, np.float32), theta_b=np.asarray(tb, np.float32), sign=np.asarray(sg, np.int8),
                        kappa=np.asarray(kp, np.float32), phi_c=np.asarray(pc, np.float32), D=np.asarray(Dl, np.float32),
                        srcA=np.asarray(sA, np.int64), cand_q=cand_q, cand_src=cand_src, cand_valid=cand_valid, cand_dist=cand_dist,
                        cand_dir_err=cand_dir, cand_geo=cand_geo, path_xy=path_xy, path_ori=path_ori, gA_idx=gA_idx, gA_w=gA_w,
                        gB_idx=gB_idx, gB_w=gB_w, pt_valid=pt_valid, missing=missing, missing_guide=missing_guide, n_geometric=n_geo, n_path_rejected=n_rej,
                        rejected_q=rej_q, rejected_reason=rej_r, n_rejected_by_reason=rej_n, legacy_q=legacy_q, n_src=l4.n, n_v1=gs.n, n_pos1=gs.n_pos)


class RoutingRuntime:
    """장치 쪽 라우팅 표와 이미지별 선택 (매 배치: gather, 곱·합, 마스크 argmax, min — 최근접 탐색·후보별 전체 재실행 없음).

        a_m = F1(p(t_m), 경로 접선 방향),  b_m = 같은 위치의 모든 방향 반응 합,  r_m = a_m/(b_m + ε)  (b_m < b_min 또는 무효 점은 0)
        S(P,Q) = (1/M) Σ_m r_m   ∈ [0,1)   — 방향 지지 정도 (곡선일 확률·정답 확률이 아니다)
        Q* = argmax_{유효 후보} S  (같은 점수는 기하 순서). 모든 S ≤ 0 이면 기하 첫 후보 + no_guidance 기록.
        B = transmitted_V4L4[Q*, θB],  u = min(A, B)  — S 를 u 에 곱하지 않고 V1 활동을 더하지 않는다.
    입력에 라벨·prototype·오차가 없다 (함수 인자에 없음). 표본마다 독립 (배치 순서와 무관)."""

    def __init__(self, table: RoutingTable, be: Backend, cfg: dict[str, Any], n_layer: int, v1_n_ch: int, lum_base: int, n_o: int):
        rc = cfg["routing"]
        self.table, self.be, self.mode = table, be, table.mode
        self.K, self.M = table.K, table.M
        self.nc = len(table.curve_rows)
        self.eps, self.b_min = float(rc["guide_epsilon"]), float(rc["guide_min_total"])
        self.cand_src = be.i64(table.cand_src)
        self.cand_valid = be.bool_(table.cand_valid)
        self.gA_idx, self.gA_w = be.i64(table.gA_idx), be.f32(table.gA_w)
        self.gB_idx, self.gB_w = be.i64(table.gB_idx), be.f32(table.gB_w)
        self.pt_valid = be.bool_(table.pt_valid)
        cmap = np.full(n_layer, self.nc, np.int64)
        cmap[table.curve_rows] = np.arange(self.nc)
        self.curve_map = be.i64(cmap)
        self.curve_rows = be.i64(table.curve_rows)
        miss = np.zeros(n_layer, bool)
        miss[table.curve_rows[table.missing]] = True
        self.missing_rows = be.bool_(miss)
        self.n_pos1, self.v1_n_ch, self.lum_base, self.n_o = table.n_pos1, v1_n_ch, lum_base, n_o
        self.slot_weight = be.f32(np.arange(self.K, 0, -1, dtype=np.float32))

    def scores(self, guide: Any, keep_points: bool = False) -> Any:
        """guide: (B, n_V1L3) 이번 이미지들의 V1 L3 실제 전송 출력 → S (B, nc, K) (무효 후보 −1).
        keep_points=True 면 (S, r (B,nc,K,M), 유효 지점 마스크) 를 돌려준다 (기록용)."""
        be = self.be
        B = int(guide.shape[0])
        F1e = be.pad_zero_col(guide)
        Fsum = guide.reshape(B, self.n_pos1, self.v1_n_ch)[:, :, self.lum_base:self.lum_base + self.n_o].sum(-1)
        Fsume = be.pad_zero_col(Fsum)
        per = max(1, self.K * self.M * 12)
        rows = max(1, be.max_gather // max(1, B * per))
        parts, rparts, oparts = [], [], []
        for a in range(0, self.nc, rows):
            ia, wa = self.gA_idx[a:a + rows], self.gA_w[a:a + rows]
            ib, wb = self.gB_idx[a:a + rows], self.gB_w[a:a + rows]
            av = (be.take_cols(F1e, ia) * wa).sum(-1)                 # (B, r, K, M)
            bv = (be.take_cols(Fsume, ib) * wb).sum(-1)
            ok = self.pt_valid[a:a + rows] & (bv >= self.b_min)
            r_ = be.where(ok, av / (bv + self.eps), be.zeros(av.shape))
            parts.append(r_.sum(-1) / float(self.M))
            if keep_points:
                rparts.append(r_)
                oparts.append(ok)
        S = parts[0] if len(parts) == 1 else be.cat1(parts)
        S = be.where(self.cand_valid, S, be.zeros(S.shape) - 1.0)
        if not keep_points:
            return S
        return S, (rparts[0] if len(rparts) == 1 else be.cat1(rparts)), (oparts[0] if len(oparts) == 1 else be.cat1(oparts))

    def select(self, xe_l4: Any, guide: Any | None, *, mode: str | None = None, record: bool = False) -> dict[str, Any]:
        """record=True: 기록용으로 경로 지점 점수 r 와 유효 지점 마스크를 함께 돌려준다. geometry_only 에서도 안내 지도가 있으면
        같은 S 를 기록용으로만 계산한다 (선택에는 쓰지 않는다 — S_used_for_selection=False)."""
        be = self.be
        mode = mode or self.mode
        B = int(xe_l4.shape[0])
        R = OK = None
        if mode == "geometry_only" or guide is None:
            if mode not in ("geometry_only", "v1_guided"):
                raise ValueError(f"알 수 없는 라우팅 모드: {mode}")
            k = be.zeros_i64((B, self.nc))
            S = None
            no_g = be.bool_(np.zeros((B, self.nc), bool)) if mode == "geometry_only" else be.bool_(np.ones((B, self.nc), bool))
            if record and guide is not None:
                S, R, OK = self.scores(guide, keep_points=True)
        elif mode == "v1_guided":
            if record:
                S, R, OK = self.scores(guide, keep_points=True)
            else:
                S = self.scores(guide)
            mx = be.max_last(S)
            no_g = mx <= 0.0
            ismax = (S == mx.reshape(B, self.nc, 1)) & self.cand_valid
            k = be.argmax_last(be.where(ismax, self.slot_weight + be.zeros(S.shape), be.zeros(S.shape)))
            k = be.where(no_g, be.zeros_i64((B, self.nc)), k)
        else:
            raise ValueError(f"알 수 없는 라우팅 모드: {mode}")
        sel_src = be.take_last(self.cand_src, k)
        Bv = be.gather_cols(xe_l4, sel_src)
        return {"k": k, "sel_src": sel_src, "B": Bv, "S": S, "no_guidance": no_g, "mode": mode, "R": R, "OK": OK,
                "S_used_for_selection": mode == "v1_guided" and guide is not None}

    def nbytes(self) -> int:
        t = self.table
        return int(t.gA_idx.size * 12 + t.gB_idx.size * 12 + t.cand_src.size * 9 + t.pt_valid.size)


def routing_rec_np(be: Backend, sel: dict[str, Any]) -> dict[str, np.ndarray | None]:
    k = be.to_np(sel["k"]).astype(np.int64)
    out = {"k": k, "B": be.to_np(sel["B"]).astype(np.float64), "A": be.to_np(sel["A"]).astype(np.float64),
           "S": (be.to_np(sel["S"]).astype(np.float64) if sel.get("S") is not None else None),
           "no_guidance": be.to_np(sel["no_guidance"]).astype(bool), "R_sel": None, "OK_sel": None,
           "S_used_for_selection": bool(sel.get("S_used_for_selection", False))}
    if sel.get("R") is not None:
        out["R_sel"] = np.take_along_axis(be.to_np(sel["R"]), k[:, :, None, None], 2)[:, :, 0, :].astype(np.float64)   # 선택 후보의 경로 지점 r
        out["OK_sel"] = np.take_along_axis(be.to_np(sel["OK"]), k[:, :, None, None], 2)[:, :, 0, :].astype(bool)
    return out


def point_summary(R_sel: np.ndarray, OK_sel: np.ndarray) -> dict[str, np.ndarray]:
    """선택 후보 경로의 지점 점수 요약 (표본×뉴런). 무효 지점의 r 는 0 으로 둔 값이다."""
    M = R_sel.shape[-1]
    inner = R_sel[..., 1:M - 1] if M > 2 else R_sel
    return {"valid_point_frac": OK_sel.mean(-1), "r_min": R_sel.min(-1), "r_median": np.median(R_sel, -1),
            "r_endpoints": 0.5 * (R_sel[..., 0] + R_sel[..., M - 1]), "r_interior": inner.mean(-1)}


def acc_add(acc: dict[str, float], part: dict[str, float]) -> None:
    for k, v in part.items():
        acc[k] = acc.get(k, 0.0) + float(v)


def routing_batch_stats(table: RoutingTable, rec: dict[str, np.ndarray], high: float) -> dict[str, float]:
    """한 배치의 라우팅 기록 (numpy) → 합계 (에폭 끝에 비율로). 후보 없음 뉴런은 선택 통계에서 뺀다."""
    k, Bv, A = rec["k"], rec["B"], rec["A"]
    ok = ~table.missing
    nb = k.shape[0]
    kk = k[:, ok]
    jj = np.nonzero(ok)[0]
    q_sel = table.cand_q[jj[None, :], kk]
    out = {"n_samples": float(nb), "n_pairs": float(kk.size), "n_missing_pairs": float(nb * int((~ok).sum())),
           "no_guidance": float(rec["no_guidance"][:, ok].sum()), "differs_from_geometry_first": float((kk != 0).sum()),
           "differs_from_legacy_position": float((q_sel != table.legacy_q[jj][None, :]).sum()),
           "sum_dist_err": float((np.abs(table.cand_dist[jj[None, :], kk] - table.D[jj][None, :]) / np.maximum(table.D[jj][None, :], 1e-12)).sum()),
           "sum_dir_err": float(table.cand_dir_err[jj[None, :], kk].sum()),
           "zero_B": float((Bv[:, ok] <= 0).sum()), "u_positive": float((np.minimum(A[:, ok], Bv[:, ok]) > 0).sum())}
    for s in range(table.K):
        out[f"k{s}"] = float((kk == s).sum())
    if rec.get("S") is not None:
        S = rec["S"]
        ssel = np.take_along_axis(S[:, ok, :], kk[..., None], -1)[..., 0]
        out["sum_S_selected"] = float(ssel.sum())
        out["high_S_zero_B"] = float(((ssel >= high) & (Bv[:, ok] <= 0)).sum())
    out["missing_guide_selected"] = float(table.missing_guide[jj[None, :], kk].sum())
    if rec.get("R_sel") is not None:
        ps = point_summary(rec["R_sel"][:, ok], rec["OK_sel"][:, ok])
        for key_, v_ in ps.items():
            out[f"sum_{key_}"] = float(v_.sum())
    return out


def routing_finalize(acc: dict[str, float], K: int) -> dict[str, Any]:
    n = max(1.0, acc.get("n_pairs", 0.0))
    freqs = [acc.get(f"k{s}", 0.0) / n for s in range(K)]
    nz = [f for f in freqs if f > 0]
    ent = -sum(f * math.log2(f) for f in nz) / math.log2(K) if K > 1 and nz else 0.0
    tot = acc.get("n_pairs", 0.0) + acc.get("n_missing_pairs", 0.0)
    return {"n_samples": int(acc.get("n_samples", 0)), "missing_candidate_frac": (acc.get("n_missing_pairs", 0.0) / tot) if tot else None,
            "no_guidance_frac": acc.get("no_guidance", 0.0) / n, "selection_freq_by_slot": freqs,
            "selection_top_slot_share": max(freqs) if freqs else None, "selection_entropy_norm": ent,
            "differs_from_geometry_first_frac": acc.get("differs_from_geometry_first", 0.0) / n,
            "differs_from_legacy_position_frac": acc.get("differs_from_legacy_position", 0.0) / n,
            "mean_dist_err_rel": acc.get("sum_dist_err", 0.0) / n, "mean_dir_err_deg": acc.get("sum_dir_err", 0.0) / n,
            "zero_B_frac": acc.get("zero_B", 0.0) / n, "u_positive_frac": acc.get("u_positive", 0.0) / n,
            "mean_S_selected": (acc["sum_S_selected"] / n) if "sum_S_selected" in acc else None,
            "high_S_but_zero_B": int(acc.get("high_S_zero_B", 0.0)) if "high_S_zero_B" in acc else None,
            "missing_guide_selected_frac": acc.get("missing_guide_selected", 0.0) / n,
            **{f"mean_{key_}_selected": (acc[f"sum_{key_}"] / n if f"sum_{key_}" in acc else None)
               for key_ in ("valid_point_frac", "r_min", "r_median", "r_endpoints", "r_interior")}}


def routing_reference_scores(table: RoutingTable, F1: np.ndarray, cfg: dict[str, Any], v1_n_ch: int, lum_base: int, n_o: int) -> np.ndarray:
    """검사용 NumPy float64 기준 구현 (같은 수식, 장치 경로와 독립)."""
    rc = cfg["routing"]
    eps, bmin = float(rc["guide_epsilon"]), float(rc["guide_min_total"])
    F1 = np.asarray(F1, np.float64)
    B = F1.shape[0]
    F1e = np.concatenate([F1, np.zeros((B, 1))], 1)
    Fs = F1.reshape(B, table.n_pos1, v1_n_ch)[:, :, lum_base:lum_base + n_o].sum(-1)
    Fse = np.concatenate([Fs, np.zeros((B, 1))], 1)
    a = (F1e[:, table.gA_idx] * table.gA_w.astype(np.float64)).sum(-1)
    b = (Fse[:, table.gB_idx] * table.gB_w.astype(np.float64)).sum(-1)
    ok = table.pt_valid[None] & (b >= bmin)
    r = np.where(ok, a / (b + eps), 0.0)
    S = r.mean(-1)
    return np.where(table.cand_valid[None], S, -1.0)


# ======================================================================
# 9. 순방향 계산 (CortexArea 런타임, L6 RelativeActivityController, CortexModel)
# ======================================================================
class BucketELL:
    """CSR 입력 목록 → 팬인 크기별 ELL 묶음 (팬인을 2 의 거듭제곱으로 올려 묶는다). 각 행은 정확히 한 묶음에 속하고,
    묶음 결과를 고정 역순열 gather 로 원래 순서에 놓는다 (atomic 누적 없음 → 결정적)."""

    def __init__(self, be: Backend, ptr: np.ndarray, src: np.ndarray, w: np.ndarray, n_src: int):
        n = len(ptr) - 1
        self.n, self.n_src, self.edges = n, int(n_src), int(len(src))
        self.parts: list[tuple[Any, Any]] = []
        self.inv: Any = None
        self.slots = 0
        if self.edges == 0:
            return
        fan = np.diff(ptr)
        K = np.ones(n, np.int64)
        nz = fan > 0
        K[nz] = np.left_shift(1, np.ceil(np.log2(fan[nz])).astype(np.int64))
        row_of_edge = np.repeat(np.arange(n, dtype=np.int64), fan)
        col = np.arange(len(src), dtype=np.int64) - ptr[row_of_edge]
        perm = []
        for k in np.unique(K):
            rows = np.nonzero(K == k)[0]
            local = np.full(n, -1, np.int64)
            local[rows] = np.arange(len(rows))
            idx = np.full((len(rows), int(k)), self.n_src, np.int64)
            ww = np.zeros((len(rows), int(k)), np.float32)
            em = K[row_of_edge] == k
            idx[local[row_of_edge[em]], col[em]] = src[em]
            ww[local[row_of_edge[em]], col[em]] = w[em]
            self.parts.append((be.i64(idx), be.f32(ww)))
            perm.append(rows)
            self.slots += len(rows) * int(k)
        perm_ = np.concatenate(perm)
        if not np.array_equal(perm_, np.arange(n)):
            self.inv = be.i64(np.argsort(perm_))

    def apply(self, be: Backend, xe: Any) -> Any:
        if not self.parts:
            return be.zeros((int(xe.shape[0]), self.n))
        outs = [be.ell(xe, idx, w) for idx, w in self.parts]
        out = outs[0] if len(outs) == 1 else be.cat1(outs)
        return out if self.inv is None else be.take_cols(out, self.inv)

    def nbytes(self) -> int:
        return int(self.slots * 12 + (self.n * 8 if self.inv is not None else 0))


class LayerRuntime:
    """한 층의 장치 쪽 고정 배열 (입력군 A·B 의 ELL 묶음, min2 표시, L6 이웃 위치)."""

    def __init__(self, spec: LayerSpec, be: Backend, cfg: dict[str, Any]):
        self.spec, self.key, self.src_key = spec, spec.key, spec.src_key
        self.n, self.n_pos, self.n_ch = spec.n, spec.n_pos, spec.n_ch
        self.A = BucketELL(be, spec.a_ptr, spec.a_src, spec.a_w, spec.n_src)
        self.has_b = spec.b_src.size > 0
        self.Bm = BucketELL(be, spec.b_ptr, spec.b_src, spec.b_w, spec.n_src) if self.has_b else None
        self.op_min = be.bool_(spec.op_min) if self.has_b else None
        l6 = cfg["l6"]
        self.l6 = bool(l6["enabled"]) and spec.l6_capable and spec.name in l6["layers"]
        self.include_self = bool(l6["include_self"])
        cnt = np.repeat(spec.pos_nb_count * spec.n_ch, spec.n_ch) - (0 if self.include_self else 1)
        self.nb_count = cnt
        self.n_no_neighbor = int((cnt <= 0).sum())
        self.pos_nb = be.i64(spec.pos_nb)
        self.has_nb = be.bool_(cnt > 0)
        self.nb_div = be.f32(np.maximum(cnt, 1))
        self.missing = be.bool_(spec.missing) if (spec.missing is not None and spec.missing.any()) else None
        self.routing: RoutingRuntime | None = None            # 새 모드의 V4/L2 에만 붙는다

    def nbytes(self) -> int:
        b = self.A.nbytes() + (self.Bm.nbytes() if self.Bm is not None else 0)
        return int(b + self.spec.pos_nb.size * 8 + self.n * 5 + (self.routing.nbytes() if self.routing is not None else 0))


class RelativeActivityController:
    """L6: 기존 상대 활동 조절만 한다 (방향 선택적 정규화 풀·적응 없음).

        같은 층의 L6 조절 전 CONF 를 모두 계산한 뒤, 그 같은 스냅샷에서 병렬로
        r_j = CONF_j / (mean_{k∈N(j)} CONF_k + ε)
        g_j = clip(1 + κ·(r_j − 1), g_min, g_max)
        transmitted_j = g_j · CONF_j                    (이웃이 없으면 g_j = 1)
    N(j): 같은 층 (같은 Z) 에서 피질 격자 Chebyshev 거리 ≤ R 인 위치의 모든 채널, 기본 자기 자신 제외 (방향 가중치 없음).
    이웃 합 = Σ_{위치∈N} (위치별 채널 합) − CONF_j. 조절된 값을 분모에 다시 넣지 않고 반복하지 않는다.
    라벨·prototype·미래 표본 정보를 쓰지 않으며 q 를 바꾸지 않는다 (g 는 이번 입력의 일시적 값)."""

    def __init__(self, cfg: dict[str, Any], be: Backend):
        l6 = cfg["l6"]
        self.be = be
        self.eps, self.kappa = float(l6["epsilon"]), float(l6["kappa"])
        self.g_min, self.g_max = float(l6["g_min"]), float(l6["g_max"])

    def gain(self, conf: Any, rt: LayerRuntime) -> Any:
        be = self.be
        B = int(conf.shape[0])
        pos_sum = conf.reshape(B, rt.n_pos, rt.n_ch).sum(-1)                  # (B, n_pos)
        nb_sum = be.take_cols(be.pad_zero_col(pos_sum), rt.pos_nb).sum(-1)    # (B, n_pos)
        nb_n = be.repeat_cols(nb_sum, rt.n_ch)                                # (B, n)
        if not rt.include_self:
            nb_n = be.relu(nb_n - conf)                                       # 부동소수 반올림으로 생긴 음수 제거
        mean = nb_n / rt.nb_div
        r = conf / (mean + self.eps)
        g = be.clip(1.0 + self.kappa * (r - 1.0), self.g_min, self.g_max)
        return be.where(rt.has_nb, g, be.ones_like(g))

    def describe(self) -> dict[str, Any]:
        return {"epsilon": self.eps, "kappa": self.kappa, "g_min": self.g_min, "g_max": self.g_max,
                "formula": FORMULAS["l6"], "applies_to": "각 영역 L2·L3 의 실제 전송 출력 (L4 없음)",
                "uses_labels_or_prototypes": False, "modifies_q": False}


class CortexModel:
    """영역 (CortexArea: V1/V2/V4/IT) 의 층을 순서대로 이어 계산한다. 감각 회로에 재귀 흥분 연결은 없다.
    forward 는 표본마다 동적 상태 (INPUT 합 u, s, CONF, g, 전송 출력) 를 새로 계산하고 남기지 않는다 (q·prototype 만 유지)."""

    def __init__(self, cfg: dict[str, Any], be: Backend, pre: RetinaLGNPreprocessor):
        self.cfg, self.be, self.pre = cfg, be, pre
        self.store = NeuronStore(be, cfg)
        self.inspector = NeuronInspector(self.store)
        self.specs: dict[str, LayerSpec] = {}
        self.rt: dict[str, LayerRuntime] = {}
        self.areas: list[str] = []
        self.active: str | None = None
        self.struct_hash: dict[str, str] = {}
        self.routers: dict[str, "LocalErrorRouter"] = {}
        self.area_objs: dict[str, CortexArea] = {}
        self.l6 = RelativeActivityController(cfg, be)
        self.timer: PhaseTimer | None = None

    @property
    def routing_mode(self) -> str:
        return str(self.cfg["routing"]["mode"])

    def routing_runtime(self) -> RoutingRuntime | None:
        rt = self.rt.get("V4/L2")
        return rt.routing if rt is not None else None

    def plan(self, area: str) -> list[LayerSpec]:
        want = AREA_ORDER[len(self.areas)] if len(self.areas) < len(AREA_ORDER) else None
        if area != want:
            raise StructureError(f"다음에 만들 수 있는 영역은 {want} 이다 (요청 {area}).")
        prev = self.specs.get(f"{self.areas[-1]}/L3") if self.areas else None
        return AREA_CLASSES[area].build(self.cfg, self.pre, prev, dict(self.specs)).specs

    def add_area(self, area: str, specs: Sequence[LayerSpec]) -> None:
        if [sp.area for sp in specs] != [area] * 3 or [sp.name for sp in specs] != list(LAYER_ORDER):
            raise StructureError("영역 층 순서는 L4, L2, L3 이다.")
        self.store.add_layers(specs)
        self.area_objs[area] = AREA_CLASSES[area](specs)
        for sp in specs:
            self.specs[sp.key] = sp
            self.rt[sp.key] = LayerRuntime(sp, self.be, self.cfg)
            if sp.routing is not None:
                gs = GuideSampler(self.specs["V1/L3"], self.pre, self.cfg)
                self.rt[sp.key].routing = RoutingRuntime(sp.routing, self.be, self.cfg, sp.n, gs.n_ch, gs.lum_base, gs.n_o)
        self.areas.append(area)
        self.struct_hash[area] = area_structure_hash(specs)
        le = self.cfg["learning"]
        self.routers[area] = LocalErrorRouter(specs[1], specs[2], float(le["alpha2"]), float(le["alpha3"]), self.be)

    def layer_keys(self, upto: str | None = None) -> list[str]:
        out = []
        for a in self.areas:
            out += [f"{a}/{n}" for n in LAYER_ORDER]
            if a == upto:
                break
        return out

    def forward(self, lgn: Any, *, upto: str | None = None, keep: Sequence[str] = (),
                override: dict[str, LayerRuntime] | None = None, routing_mode: str | None = None,
                guide_transform: Callable[[Any], Any] | None = None, routing_record: bool = False) -> dict[str, dict[str, Any]]:
        """lgn: (B, n_lgn) LGN 표본 (위치 우선). upto: 이 영역까지 계산 (기본 마지막 영역).
        반환: 각 영역 L3 의 {"out"} 과 keep 층의 {"u","s","conf","g","out"} (모두 (B, n_layer)). "LGN" 을 keep 하면 입력도 돌려준다.
        라우팅 (새 모드의 V4/L2): 같은 순방향에서 이미 계산한 V1 L3 실제 전송 출력을 안내로 쓴다 (V1 을 다시 실행하지 않음, 라벨 입력 없음).
        routing_mode 로 이 호출만 선택 규칙을 바꿀 수 있고 (진단), guide_transform 은 안내 사본만 바꾼다 (감각 경로는 그대로).
        routing_record=True 또는 V4/L2 를 keep 하면 res["V4/L2"]["routing"] 에 이번 표본들의 선택 (k, 주소, S, A, B) 을 남긴다."""
        be, st = self.be, self.store
        keep_s = set(keep)
        res: dict[str, dict[str, Any]] = {}
        if "LGN" in keep_s:
            res["LGN"] = {"out": lgn}
        prev_key, prev = "LGN", lgn
        for key in self.layer_keys(upto):
            rt = (override or {}).get(key) or self.rt[key]
            if rt.src_key != prev_key:
                raise StructureError(f"{key} 의 원천 {rt.src_key} 가 직전 층 {prev_key} 와 다르다.")
            xe = be.pad_zero_col(prev)
            sa = rt.A.apply(be, xe)
            sel = None
            if rt.has_b or rt.routing is not None:
                sb = rt.Bm.apply(be, xe) if rt.Bm is not None else be.zeros((int(xe.shape[0]), rt.n))
                if rt.routing is not None:
                    guide = res["V1/L3"]["out"]
                    if guide_transform is not None:
                        guide = guide_transform(guide)
                    mode = routing_mode or rt.routing.mode
                    tm = self.timer.phase("routing", gpu=bool(self.cfg["compute"]["gpu_timing_sync"]) and be.is_cuda) \
                        if self.timer is not None else contextlib.nullcontext()
                    with tm:
                        sel = rt.routing.select(xe, guide, mode=mode, record=bool(routing_record or key in keep_s))
                        sb = sb + be.take_cols(be.pad_zero_col(sel["B"]), rt.routing.curve_map)   # 라우팅 행의 정적 B 는 비어 있다
                u = be.where(rt.op_min, be.minimum(sa, sb), sa + sb)
                if rt.missing is not None:
                    u = be.where(rt.missing, be.zeros(u.shape), u)                               # missing_candidate: u = 0
            else:
                u = sa
            s = be.relu(u - st.thr_view(key))
            conf = st.q_view(key) * s
            g = self.l6.gain(conf, rt) if rt.l6 else None
            out = conf if g is None else g * conf
            if key in keep_s:
                res[key] = {"u": u, "s": s, "conf": conf, "g": g, "out": out}
            elif key.endswith("/L3"):
                res[key] = {"out": out}
            if sel is not None and (routing_record or key in keep_s):
                sel["A"] = be.take_cols(sa, rt.routing.curve_rows)
                res.setdefault(key, {})["routing"] = sel
            prev_key, prev = key, out
        return res

    def device_bytes(self) -> int:
        return int(sum(rt.nbytes() for rt in self.rt.values()) + self.store.N * 8)


def structure_summary(specs: Sequence[LayerSpec], be: Backend, cfg: dict[str, Any], n_classes: int,
                      existing_bytes: int = 0) -> dict[str, Any]:
    """모델 생성 전에 출력하는 뉴런 수·연결 수·예상 텐서 메모리."""
    B = int(cfg["learning"]["batch_size"])
    rows, total_edges, total_slots, total_n = [], 0, 0, 0
    for sp in specs:
        e = len(sp.a_src) + len(sp.b_src)
        slots = 0
        for ptr in (sp.a_ptr, sp.b_ptr):
            fan = np.diff(ptr)
            nz = fan[fan > 0]
            slots += int(np.left_shift(1, np.ceil(np.log2(nz)).astype(np.int64)).sum()) if nz.size else 0
        total_edges += e
        total_slots += slots
        total_n += sp.n
        rows.append({"layer": sp.key, "n_neurons": sp.n, "positions": sp.n_pos, "channels": sp.n_ch, "edges": e,
                     "fan_in_mean": sp.build_stats.get("fan_in_mean"), "fan_in_max": sp.build_stats.get("fan_in_max"),
                     "n_min2": sp.build_stats.get("n_min2"), "ell_slots": slots})
    graph_mb = (total_slots * 12 + total_n * 8) / 2 ** 20
    act_mb = B * total_n * 4 * 6 / 2 ** 20
    gather_mb = min(int(cfg["compute"]["max_gather_elements"]), max(1, B * total_slots)) * 4 * 2 / 2 ** 20
    proto_mb = n_classes * specs[-1].n * 4 * 2 / 2 ** 20
    rc = cfg["records"]
    diag_mb = (int(rc["diagnostic_samples"]) * total_n * 4 * 6 + int(rc["correction_samples_per_epoch"]) * specs[-1].n * 4 * 3) / 2 ** 20
    return {"layers": rows, "total_neurons": total_n, "total_edges": total_edges, "ell_slots": total_slots,
            "memory_mb": {"graph": round(graph_mb, 2), "batch_activations_upper": round(act_mb, 2),
                          "gather_chunk_upper": round(gather_mb, 2), "prototypes": round(proto_mb, 3),
                          "diagnostic_and_record_buffers": round(diag_mb, 3), "existing_model": round(existing_bytes / 2 ** 20, 2),
                          "total_estimate": round(graph_mb + act_mb + gather_mb + proto_mb + diag_mb + existing_bytes / 2 ** 20, 2)},
            "batch_size": B, "backend": be.kind, "device": be.device,
            "note_ko": "추정치다 (활성 영역은 u·s·CONF·g·out 을 보관, 나머지 층은 출력만 잠시 둔다). 실제 사용량은 장치에서 다르다."}


# ======================================================================
# 10. 클래스별 prototype (PrototypeBank) 과 코사인 분류
# ======================================================================
class PrototypeBank:
    """영역마다 독립 클래스별 prototype (C, n_L3). h = 그 영역 L3 실제 전송 출력 (neuron_id 순서).
    초기화·갱신은 train 출력의 클래스 평균만 쓴다 (dev/test·추론 중 갱신 없음). 영벡터 prototype 은 무효로 표시한다."""

    def __init__(self, area: str, classes: Sequence[str], dim: int, be: Backend, eps: float):
        self.area, self.classes, self.dim, self.be, self.eps = area, list(classes), int(dim), be, float(eps)
        C = len(self.classes)
        self.P = be.zeros((C, self.dim))
        self.valid = np.zeros(C, bool)
        self.counts = np.zeros(C, np.int64)
        self.version = 0
        self.history: list[dict[str, Any]] = []

    def set(self, P_new: Any, counts: np.ndarray, *, source: str, split: str, sample_ids_sha256: str, epoch: int | None,
            rho: float | None) -> dict[str, Any]:
        if split != "train":
            raise RuntimeError("prototype 은 train 출력으로만 만든다.")
        norms = self.be.to_np(self.be.norm_last(P_new)).astype(np.float64)
        old = self.be.to_np(self.P).astype(np.float64)
        new = self.be.to_np(P_new).astype(np.float64)
        self.P = P_new
        self.valid = norms > self.eps
        self.counts = np.asarray(counts, np.int64)
        self.version += 1
        rec = {"version": self.version, "source": source, "split": split, "sample_ids_sha256": sample_ids_sha256, "epoch": epoch,
               "rho": rho, "counts": self.counts.tolist(), "norms": norms.tolist(), "valid": self.valid.tolist(),
               "update_l2": float(np.linalg.norm(new - old)), "utc": utc_now(),
               "invalid_classes": [c for c, v in zip(self.classes, self.valid) if not v]}
        self.history.append(rec)
        return rec

    def targets(self, y: np.ndarray) -> Any:
        return self.be.rows(self.P, self.be.i64(y))

    def sha(self) -> str:
        return sha256_array(self.be.to_np(self.P).astype(np.float32), self.valid)

    def pairwise_cos(self) -> np.ndarray:
        P = self.be.to_np(self.P).astype(np.float64)
        n = np.linalg.norm(P, axis=1)
        C = len(self.classes)
        out = np.full((C, C), np.nan)
        ok = n > self.eps
        if ok.any():
            Pn = P[ok] / n[ok, None]
            out[np.ix_(ok, ok)] = Pn @ Pn.T
        return out


def cosine_classify(be: Backend, h: Any, P: Any, valid_P: np.ndarray, eps: float) -> dict[str, np.ndarray]:
    """pred = argmax_c cos(h, P_c). ‖h‖ ≤ eps → unknown (−1). 무효 prototype 클래스는 후보에서 뺀다. 코사인은 확률이 아니다."""
    hn = be.to_np(be.norm_last(h)).astype(np.float64)
    pn = be.to_np(be.norm_last(P)).astype(np.float64)
    dots = be.to_np(be.matmul_t(h, P)).astype(np.float64)
    B, C = dots.shape
    valid_h = hn > eps
    cv = np.asarray(valid_P, bool) & (pn > eps)
    cos = np.full((B, C), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        if cv.any():
            cos[:, cv] = dots[:, cv] / (hn[:, None] * pn[None, cv])
    cos[~valid_h] = np.nan
    pred = np.full(B, -1, np.int64)
    ties = 0
    for i in range(B):
        if valid_h[i] and cv.any():
            row = cos[i]
            m = np.nanmax(row)
            pred[i] = int(np.nanargmax(row))
            ties += int((np.abs(row[cv] - m) <= 1e-12).sum() > 1)
    return {"cos": cos, "pred": pred, "h_norm": hn, "valid_h": valid_h, "n_ties": np.asarray(ties)}


# ======================================================================
# 11. 교정: LocalErrorRouter → L1Relay → LocalOutputLearner
# ======================================================================
class LocalErrorRouter:
    """영역 수준 교정 배분기. 결과를 L1 패킷으로 넘긴다 (L5/L6 는 오차를 계산·배달하지 않는다).

        δ = prototype_true − h                                   (B, n3)  — StageTrainer 가 계산, L3 neuron_id 순서
        m_L3[:, j] = α3 · δ[:, j]                                (B, n3)
        m_L2[:, i] = α2 · Σ_{j : i ∈ P(j)} δ[:, j] / |P(j)|      (B, n2)   즉 m_L2 = α2 · R δ,  R_ij = 1/|P(j)| (i ∈ P(j))
    P(j) = L3 뉴런 j 의 실제 L2 부모 집합 (L3 의 CSR 입력 목록에서 주소만 읽는다; 가중치·전치·미분·복원기·무작위 피드백 미사용).
    일대일 구간이면 R 은 대응 항등 배달이다. 이 배분은 부호·비선형·상쇄를 역산하지 않는다 (위치 배달 ≠ 유익한 교정)."""

    def __init__(self, spec2: LayerSpec, spec3: LayerSpec, alpha2: float, alpha3: float, be: Backend):
        if spec3.src_key != spec2.key:
            raise StructureError(f"{spec3.key} 의 원천이 {spec2.key} 가 아니다.")
        self.be, self.alpha2, self.alpha3 = be, float(alpha2), float(alpha3)
        self.area = spec3.area
        n2, n3 = spec2.n, spec3.n
        rows = np.concatenate([np.repeat(np.arange(n3, dtype=np.int64), np.diff(spec3.a_ptr)),
                               np.repeat(np.arange(n3, dtype=np.int64), np.diff(spec3.b_ptr))])
        src = np.concatenate([spec3.a_src, spec3.b_src])
        pairs = np.unique(rows * np.int64(n2) + src)
        prow, psrc = pairs // n2, pairs % n2
        cnt = np.bincount(prow, minlength=n3)
        if (cnt == 0).any():
            raise StructureError(f"{spec3.key}: 부모 없는 L3 출력 {int((cnt == 0).sum())} 개 (오류)")
        coef = 1.0 / cnt[prow]
        order = np.argsort(psrc, kind="stable")
        ch_cnt = np.bincount(psrc, minlength=n2)
        Kc = max(int(ch_cnt.max()), 1)
        ptr = np.concatenate([[0], np.cumsum(ch_cnt)])
        col = np.arange(len(order)) - ptr[psrc[order]]
        ch_idx = np.full((n2, Kc), n3, np.int64)
        ch_coef = np.zeros((n2, Kc), np.float32)
        ch_idx[psrc[order], col] = prow[order]
        ch_coef[psrc[order], col] = coef[order]
        self.parent_rows, self.parent_src = prow, psrc
        self.parent_count, self.child_count = cnt, ch_cnt
        self.R_col_sum = np.bincount(prow, weights=coef, minlength=n3)
        self.one_to_one = bool((cnt == 1).all() and (ch_cnt <= 1).all())
        self.n2_without_child = int((ch_cnt == 0).sum())
        self.ch_idx_np, self.ch_coef_np = ch_idx, ch_coef
        self.ch_idx, self.ch_coef = be.i64(ch_idx), be.f32(ch_coef)

    def route(self, delta: Any) -> tuple[Any, Any]:
        be = self.be
        m3 = self.alpha3 * delta
        m2 = self.alpha2 * (be.take_cols(be.pad_zero_col(delta), self.ch_idx) * self.ch_coef).sum(-1)
        return m2, m3

    def describe(self) -> dict[str, Any]:
        pc = self.parent_count
        return {"formula": FORMULAS["router"], "alpha2": self.alpha2, "alpha3": self.alpha3, "one_to_one": self.one_to_one,
                "parents_per_L3": {"min": int(pc.min()), "max": int(pc.max()), "mean": float(pc.mean())},
                "children_per_L2": {"min": int(self.child_count.min()), "max": int(self.child_count.max())},
                "L2_without_L3_child": self.n2_without_child,
                "R_column_sum_max_abs_error": float(np.abs(self.R_col_sum - 1.0).max())}


@dataclass
class L1Packet:
    area: str
    layer: str
    start: int
    end: int
    ids_sha256: str
    m: Any
    batch_size: int
    sample_ids_sha256: str
    model_version: str
    kind: str = "teacher_correction"          # 감각 입력 경로와 구분되는 교사 (prototype 비교) 경로


class L1Relay:
    """L1: 교정 배분기가 계산한 m_L2·m_L3 를 layer_id·neuron_id 범위·모양을 확인해 해당 층 q 학습기로 배달하는 논리 구획.
    거대한 세포체 집단이 아니라 수신 뉴런별 교정 버퍼 (패킷의 m 행) 와 명시적 주소로 나타낸 근사다. 오차를 계산하지 않는다.
    패킷은 배치마다 새로 만들고 남기지 않는다."""

    def __init__(self, area: str, store: NeuronStore):
        self.area = area
        self.addr = {}
        for lay in ("L2", "L3"):
            sp = store.layers[f"{area}/{lay}"]
            self.addr[lay] = (sp.start, sp.end, store.ids_sha(sp.key), sp.n)
        self.delivered = 0
        self.rejected = 0

    def packet(self, layer: str, m: Any, sample_sha: str, version: str) -> L1Packet:
        start, end, sha, _n = self.addr[layer]
        return L1Packet(self.area, layer, start, end, sha, m, int(m.shape[0]), sample_sha, version)

    def deliver(self, pkt: L1Packet, batch_size: int, be: Backend) -> Any:
        def reject(why: str) -> None:
            self.rejected += 1
            raise L1AddressError(f"L1 배달 거부 ({self.area}): {why}")
        if pkt.area != self.area:
            reject(f"영역 {pkt.area} != {self.area}")
        if pkt.layer not in self.addr:
            reject(f"층 {pkt.layer} 은 교정 대상이 아니다 (L2/L3 만)")
        start, end, sha, n = self.addr[pkt.layer]
        if (pkt.start, pkt.end, pkt.ids_sha256) != (start, end, sha):
            reject("neuron_id 범위·해시 불일치")
        if tuple(int(s) for s in pkt.m.shape) != (int(batch_size), int(n)):
            reject(f"모양 {tuple(pkt.m.shape)} != ({batch_size}, {n})")
        if not be.all_finite(pkt.m):
            reject("NaN/Inf")
        self.delivered += 1
        return pkt.m


def _scalar(be: Backend, x: Any) -> float:
    return float(np.asarray(be.to_np(x)).reshape(-1)[0]) if not isinstance(x, (int, float)) else float(x)


class LocalOutputLearner:
    """국소 지속 파라미터 갱신 (local_update).

        e_j  = s_j / (s_j + e0)                         s_j: 같은 순방향 스냅샷의 L6 조절 전 국소 활동 (B, n), e0 > 0
        Δq_j = η · mean_batch(e_j · m_j)
        q_j  ← clip(q_j + Δq_j, q_min, q_max)
    L2·L3 의 Δq 를 같은 스냅샷에서 모두 계산한 뒤 동시에 적용한다. m = 0 이면 Δq = 0 (정확히).
    별도 감쇠·항상성 갱신·eta_commit·임계값 변경·숨은 배율이 없다. e 는 표본의 활동 표시이며 STDP·칼슘 흔적이 아니다."""

    def __init__(self, cfg: dict[str, Any], be: Backend, store: NeuronStore):
        le, ne = cfg["learning"], cfg["neuron"]
        self.be, self.store = be, store
        self.eta, self.e0 = float(le["eta"]), float(le["e0"])
        self.q_min, self.q_max = float(ne["q_min"]), float(ne["q_max"])

    def proposal(self, s: Any, m: Any) -> Any:
        e = s / (s + self.e0)
        return self.eta * self.be.mean0(e * m)

    def apply(self, proposals: dict[str, Any]) -> dict[str, dict[str, float]]:
        be = self.be
        olds = {k: be.clone(self.store.q_view(k)) for k in proposals}
        raws = {k: olds[k] + dq for k, dq in proposals.items()}
        news = {k: be.clip(raws[k], self.q_min, self.q_max) for k in proposals}
        stats: dict[str, dict[str, float]] = {}
        for k in proposals:
            if not be.all_finite(news[k]):
                raise FloatingPointError(f"{k}: q 갱신값에 NaN/Inf (적용하지 않는다)")
            d = news[k] - olds[k]
            stats[k] = {"dq_raw_l2": _scalar(be, be.norm_last(proposals[k])), "dq_applied_l2": _scalar(be, be.norm_last(d)),
                        "dq_raw_abs_max": _scalar(be, abs(proposals[k]).max()),
                        "n_clip_low": _scalar(be, (raws[k] < self.q_min).sum()), "n_clip_high": _scalar(be, (raws[k] > self.q_max).sum()),
                        "n_at_min": _scalar(be, (news[k] <= self.q_min).sum()), "n_at_max": _scalar(be, (news[k] >= self.q_max).sum()),
                        "n_changed": _scalar(be, (d != 0).sum()), "q_mean": _scalar(be, news[k].mean()),
                        "q_min": _scalar(be, news[k].min()), "q_max": _scalar(be, news[k].max())}
        for k in proposals:
            be.assign_(self.store.q_view(k), news[k])
        return stats


# ======================================================================
# 12. 세션 상태·기록·checkpoint (StageTrainer 가 쓰는 공통 상태)
# ======================================================================
def new_stage_state(area: str, cfg: dict[str, Any]) -> dict[str, Any]:
    le = cfg["learning"]
    return {"area": area, "status": "not_started", "epochs_completed": 0, "phase": "training", "next_batch": 0,
            "max_epochs": int(le["max_epochs"]), "patience": int(le["patience"]), "min_delta": float(le["min_delta"]),
            "success_dev_accuracy": le["success_dev_accuracy"], "best_dev_accuracy": None, "best_epoch": None, "bad_epochs": 0,
            "acc": empty_acc(), "dist_start_vs_P": None, "history": [], "corr_saved": 0, "corr_epoch_saved": 0,
            "transitions": [], "started_utc": None, "ended_utc": None, "budget_changes": [], "start_eval": None,
            "routing_acc": {}, "route_epoch_saved": 0, "route_saved": 0, "structure_sha_at_creation": None}


def empty_acc() -> dict[str, float]:
    keys = ["n_batches", "n_samples", "n_valid", "n_skipped_invalid_proto", "sum_d0", "sum_d1", "sum_c0", "sum_c1",
            "n_improved", "n_worsened", "n_unchanged", "sum_cos_dh_delta", "n_cos_dh_delta", "n_dh_zero",
            "sum_mag_err", "sum_abs_mag_err", "sum_ang_err", "n_ang", "sum_delta_norm"]
    for lay in ("L2", "L3"):
        keys += [f"{lay}_{k}" for k in ("sum_dq_raw_l2", "sum_dq_applied_l2", "sum_clip_low", "sum_clip_high", "sum_n_changed",
                                        "last_at_min", "last_at_max", "sum_active_frac", "sum_s_mean", "sum_g_min_frac",
                                        "sum_g_max_frac", "sum_conf_mean")]
    return {k: 0.0 for k in keys}


def _numel(x: Any) -> int:
    return int(np.prod([int(s) for s in x.shape]))


class Session:
    """한 실행 (run 폴더) 의 상태: 설정·데이터·모델·prototype·단계 상태·동결 해시·평가 횟수·RNG."""

    def __init__(self, cfg: dict[str, Any], be: Backend, *, say: Callable[[str], None] = print,
                 confirm: Callable[[str], bool] | None = None):
        self.cfg = validate_config(cfg)
        self.be = be
        self.say = say
        self.confirm = confirm or (lambda _m: True)
        self.data_root: Path | None = None
        self.output_root: Path | None = None
        self.rec: RunRecorder | None = None
        self.data: DatasetManager | None = None
        self.pre = RetinaLGNPreprocessor(self.cfg)
        self.timer = PhaseTimer(be)
        self.loader: BatchLoader | None = None
        self.model: CortexModel | None = None
        self.protos: dict[str, PrototypeBank] = {}
        self.stage: dict[str, dict[str, Any]] = {}
        self.frozen: dict[str, dict[str, Any]] = {}
        self.counters = {"dev_evaluations": 0, "test_evaluations": 0, "train_passes": 0, "inference_images": 0}
        self.rng = np.random.default_rng([int(self.cfg["seed"]), 99])
        self.ckpt_seq = 0
        self.corr_buf: list[dict[str, Any]] = []
        self.fault: str | None = None              # 자체 검사용 결함 주입 (평소 None)
        self.last_checkpoint: str | None = None
        self.route_buf: list[dict[str, Any]] = []         # routing_samples 버퍼 (checkpoint 경계에서 기록)

    # ---------------------------------------------------------------- 경로·기록
    def set_paths(self, data_root: Path | None, output_root: Path | None) -> None:
        if data_root is not None:
            d = Path(data_root)
            if not d.is_dir():
                raise DataError(f"데이터 폴더가 없다: {d}")
            self.data_root = d.resolve()
        if output_root is not None:
            self.output_root = ensure_writable_dir(Path(output_root))

    def start_run(self) -> RunRecorder:
        if self.output_root is None:
            raise RuntimeError("저장 폴더를 먼저 정한다 (메뉴 1).")
        run = unique_run_dir(self.output_root)
        self.rec = RunRecorder(run, say=self.say)
        self.write_static()
        self.rec.info(f"새 실행 폴더 {run}", echo=True)
        return self.rec

    def program_sha(self) -> str:
        try:
            return sha256_file(Path(__file__))
        except OSError:
            return "unknown"

    def write_static(self) -> None:
        r = self.rec
        r.json("config.json", {"program": PROGRAM, "program_version": VERSION, "program_sha256": self.program_sha(),
                               "config": self.cfg, "config_structure_sha256": config_structure_sha(self.cfg),
                               "config_learning_sha256": config_learning_sha(self.cfg), "formulas": FORMULAS,
                               "backend": self.be.describe(), "utc": utc_now()})
        r.json("environment.json", DeviceManager.environment(self.be))
        r.text("assumptions.md", assumptions_markdown())
        r.json("effective_config.json", self.effective_config())
        r.json("run_info.json", {"data_root": str(self.data_root) if self.data_root else None,
                                 "output_root": str(self.output_root) if self.output_root else None,
                                 "run_dir": str(r.dir), "created_utc": utc_now(), "program_sha256": self.program_sha()})

    def effective_config(self) -> dict[str, Any]:
        ro = self.cfg["routing"]
        return {"program": PROGRAM, "program_version": VERSION, "utc": utc_now(), "config": self.cfg, "routing_mode": ro["mode"],
                "routing_modes_ko": {"legacy": "v1 의 연결·계산 재현 (θA 방향 최근접 위치, 없으면 같은 위치)",
                                     "geometry_only": "수정된 기하 후보 집합의 고정 첫 후보 (V1 안내 영향 없음)",
                                     "v1_guided": "같은 후보 집합에서 이번 이미지의 V1 L3 경로 지지 점수가 가장 큰 후보"},
                "effect_separation_ko": "legacy→geometry_only = 기하 배선 수정 효과, geometry_only→v1_guided = 안내 지도의 추가 효과 (합쳐 보고하지 않는다)",
                "routing_conventions": ROUTING_CONVENTION_KO, "formulas": FORMULAS,
                "config_structure_sha256": config_structure_sha(self.cfg), "config_learning_sha256": config_learning_sha(self.cfg),
                "cache_rule_ko": "캐시는 LGN 표본 벡터 (RAM) 만 둔다. 키 = 상대 경로 + 전처리 설정 해시. 영역 출력 (V1~IT) 은 캐시하지 않고 매 입력 다시 계산한다 "
                                 "(라우팅 변경 전의 V4/IT 출력을 재사용할 수 없다).",
                "backend": self.be.describe()}

    def log_design_change(self, note: str) -> dict[str, Any]:
        """dev 반복 확인에 따른 설계 변경 기록. test 를 본 뒤의 변경이면 이후 test 는 독립 평가가 아니라고 표시한다."""
        rec = {"utc": utc_now(), "note": note, "model_version": self.model_version() if self.model else None,
               "dev_evaluations_so_far": self.counters["dev_evaluations"], "test_evaluations_so_far": self.counters["test_evaluations"],
               "after_test_seen": self.counters["test_evaluations"] > 0}
        self.need_rec().append_jsonl("design_changes.jsonl", rec)
        return rec

    def need_rec(self) -> RunRecorder:
        if self.rec is None:
            raise RuntimeError("실행 폴더가 없다 (메뉴 2 에서 데이터 분할을 먼저 한다).")
        return self.rec

    # ---------------------------------------------------------------- 데이터
    def scan_and_split(self, progress: Callable[[str], None] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.model is not None:
            raise RuntimeError("이미 모델이 있는 세션이다. 다른 분할로 이어 붙이지 않는다 (새 세션에서 다시 시작).")
        if self.data_root is None:
            raise RuntimeError("데이터 폴더를 먼저 정한다 (메뉴 1).")
        dm = DatasetManager(self.cfg, self.data_root)
        rep = dm.scan(progress)
        man = dm.split(int(self.cfg["seed"]))
        if self.rec is None:
            self.start_run()
        r = self.need_rec()
        r.json("data_report.json", rep)
        r.json("split_manifest.json", man)
        r.json("class_to_index.json", {"class_to_index": man["class_to_index"], "classes": man["classes"],
                                       "data_fingerprint_sha256": man["data_fingerprint_sha256"], "seed": man["seed"]})
        r.json("run_info.json", {**read_json(r.p("run_info.json")), "data_root": str(self.data_root)})
        r.json("preprocessing.json", {"info": self.pre.info(), "dc_check": self.pre.dc_check()})
        self.data = dm
        self.loader = BatchLoader(dm, self.pre, self.be, self.cfg, self.timer)
        r.info(f"데이터 분할: 클래스 {man['classes']}, 항목 {man['n_items']}, 지문 {man['data_fingerprint_sha256'][:12]}")
        return rep, man

    @property
    def classes(self) -> list[str]:
        if self.data is None:
            raise RuntimeError("데이터 분할이 없다.")
        return list(self.data.classes)

    # ---------------------------------------------------------------- 모델
    def plan_next(self) -> tuple[str, list[LayerSpec], dict[str, Any]]:
        if self.data is None:
            raise RuntimeError("데이터 분할을 먼저 한다 (메뉴 2).")
        if self.model is None:
            self.model = CortexModel(self.cfg, self.be, self.pre)
            self.model.timer = self.timer
        if len(self.model.areas) >= len(AREA_ORDER):
            raise RuntimeError("IT 까지 모두 만들었다.")
        area = AREA_ORDER[len(self.model.areas)]
        specs = self.model.plan(area)
        summ = structure_summary(specs, self.be, self.cfg, len(self.classes), self.model.device_bytes())
        return area, specs, summ

    def build_area(self, area: str, specs: list[LayerSpec], summary: dict[str, Any]) -> dict[str, Any]:
        r = self.need_rec()
        self.model.add_area(area, specs)
        self.model.active = area
        self.stage[area] = new_stage_state(area, self.cfg)
        r.json(f"structure/{area}_plan_summary.json", summary)
        arrs: dict[str, np.ndarray] = {}
        for sp in specs:
            for nm in ("a_ptr", "a_src", "a_w", "b_ptr", "b_src", "b_w", "op_min", "x", "y", "rf", "pos_nb", "pos_nb_count"):
                arrs[f"{sp.name}__{nm}"] = getattr(sp, nm)
        r.npz(f"structure/{area}_structure.npz", **arrs)
        self.write_architecture()
        l2 = self.model.specs.get(f"{area}/L2")
        if l2 is not None and l2.routing is not None:
            self.write_routing_static(area)
        self.stage[area]["structure_sha_at_creation"] = self.model.struct_hash[area]
        rec = self.init_prototypes(area)
        self.stage_start_eval(area)
        self.save_checkpoint(f"{area}_created", keep=True)
        r.info(f"{area} 생성: 구조 해시 {self.model.struct_hash[area][:12]}", echo=False)
        return rec

    def write_routing_static(self, area: str) -> None:
        r = self.need_rec()
        t = self.model.specs[f"{area}/L2"].routing
        l4 = self.model.specs[f"{area}/L4"]
        qq = np.maximum(t.cand_q, 0)
        r.npz("routing/routing_candidates.npz", **t.arrays(), conventions=np.asarray(dumps(ROUTING_CONVENTION_KO, indent=None)),
              params=np.asarray(dumps(t.params, indent=None)), P_x_px=l4.pos_x[t.P], P_y_px=l4.pos_y[t.P],
              candidate_x_px=np.where(t.cand_valid, l4.pos_x[qq], np.nan), candidate_y_px=np.where(t.cand_valid, l4.pos_y[qq], np.nan),
              curve_neuron_id=self.model.specs[f"{area}/L2"].start + t.curve_rows, candidate_src_neuron_id=np.where(t.cand_valid, l4.start + t.cand_src, -1),
              table_sha256=np.asarray(t.hash()), interpolation_rule=np.asarray(
                  "V1 L3 휘도 채널: 원영상 점 → 로그-극좌표 역사상 (반경 log1p, 각도 wrap) 쌍선형 4 위치 × 180 주기 방향 선형 2 채널"))
        summ = t.static_summary(l4, self.pre)
        r.json("routing/routing_static_summary.json", summ)
        ring = l4.ring[t.P * l4.n_ch]
        for k in np.unique(ring):
            for th in np.unique(t.theta_a):
                for s in np.unique(t.sign):
                    sel = (ring == k) & (t.theta_a == th) & (t.sign == s)
                    if sel.any():
                        r.append_csv("routing/missing_by_ring_orientation.csv", {
                            "ring": int(k), "eccentricity_px_mean": float(np.hypot(l4.pos_x[t.P[sel]] - self.pre.cx, self.pre.cy - l4.pos_y[t.P[sel]]).mean()),
                            "theta_a_deg": float(th), "curve_sign": int(s), "n": int(sel.sum()), "missing_candidate_frac": float(t.missing[sel].mean()),
                            "valid_candidates_mean": float(t.cand_valid[sel].sum(1).mean())})
        r.info(f"{area} 라우팅 후보 표: 곡선 {summ['n_curve_neurons']}, 후보 없음 {fmt(summ['missing_candidate_frac'])}, "
               f"유효 후보 평균 {fmt(summ['valid_candidates_mean'], 2)}", echo=True)

    def stage_start_eval(self, area: str) -> dict[str, Any]:
        """단계 시작 전 (학습 전) 평가: 초기 구조 성능과 학습 후 성능을 분리한다."""
        bank = self.protos[area]
        dv = self.evaluate(area, "dev", bank, purpose=f"{area}_before_training")
        tr = self.pass_split(area, "train", bank=bank)
        row = {"utc": utc_now(), "area": area, "epoch": -1, "epoch_number": 0, "row_kind": "before_training",
               "model_version": self.model_version(area), "prototype_version_after": bank.version,
               "train_acc_vs_epoch_prototype": (tr["correct"] / tr["n"]) if tr["n"] else None, "train_correct_vs_epoch_prototype": tr["correct"],
               "train_n": tr["n"], "dev_n": dv["n"], "dev_correct": dv["correct"], "dev_accuracy": dv["accuracy_overall"],
               "dev_accuracy_valid_only": dv["accuracy_valid_only"], "dev_n_unknown_zero_vector": dv["n_unknown_zero_vector"],
               "dev_per_class_accuracy": dv["per_class_accuracy"], "dev_confusion": dv["confusion"],
               "dev_output_norm_mean": dv["geometry"].get("output_norm_mean"), "stage_status": "before_training"}
        self.stage[area]["start_eval"] = row
        if dv.get("routing"):
            self.write_routing_row(area, -1, "dev_before_training", dv["routing"])
        return row

    def write_routing_row(self, area: str, epoch: int, context: str, summ: dict[str, Any]) -> None:
        row = {"utc": utc_now(), "active_area": area, "epoch": epoch, "epoch_number": epoch + 1, "context": context,
               "routing_mode": self.cfg["routing"]["mode"], "model_version": self.model_version(area)}
        for k, v in summ.items():
            row[k] = v
        self.need_rec().append_csv("routing/routing_summary.csv", row)

    def write_architecture(self) -> None:
        m = self.model
        r = self.need_rec()
        layers = []
        for key in m.store.order:
            sp = m.specs[key]
            fo = m.store.fan_out(key)
            rt = m.rt[key]
            layers.append({"layer": key, "area": sp.area, "name": sp.name, "neuron_id_range": [sp.start, sp.end],
                           "n_neurons": sp.n, "n_positions": sp.n_pos, "n_channels": sp.n_ch, "Z": sp.z,
                           "grid_shape": list(sp.grid_shape) if sp.grid_shape else None, "source": sp.src_key,
                           "channels": sp.channel_names, "roles": sorted(set(sp.channel_roles)), "description": sp.description,
                           "build_stats": sp.build_stats, "fan_out": {"min": int(fo.min()), "max": int(fo.max()),
                                                                      "n_without_output": int((fo == 0).sum())},
                           "learnable_q": sp.learnable, "l6_applied": rt.l6, "l6_neurons_without_neighbor": rt.n_no_neighbor,
                           "structure_sha256": sp.hash()})
        r.json("architecture.json", {
            "program_version": VERSION, "areas": m.areas, "active_area": m.active, "total_neurons": m.store.N,
            "coordinate_systems_ko": {"X_Y": "수용장 중심, 처리 영상 화소 (원점 왼쪽 위 화소 중심, x 오른쪽, y 아래쪽)",
                                      "Z": "영역번호×10 + 피질층 번호 (V1=1, V2=2, V4=3, IT=4; L4=4, L2=2, L3=3), 무차원",
                                      "grid": "피질 격자 (ring, angle) = 로그-극좌표 (반경 bin, 각도 bin)",
                                      "orientation": "원영상 좌표 기준 반시계 각도 (y 위쪽)"},
            "paths_ko": {"sensory": "LGN → V1 L4 → V1 L2 → V1 L3 → V2 L4 → ... → IT L3 (감각 경로)",
                         "teacher": "활성 영역 L3 → prototype 비교기 → LocalErrorRouter → L1Relay → L2/L3 q (교사 교정 경로, 감각 입력을 바꾸지 않는다)",
                         "l6": "각 영역 L2·L3 의 CONF 스냅샷 → g → 실제 전송 출력 (일시적)"},
            "l6_table": [{"layer": k, "applied": m.rt[k].l6} for k in m.store.order],
            "l1_table": [{"area": a, "targets": [f"{a}/L2", f"{a}/L3"], "router": m.routers[a].describe()} for a in m.areas],
            "l6_controller": m.l6.describe(), "layers": layers, "structure_sha256": m.struct_hash, "formulas": FORMULAS,
            "areas_detail": {a: m.area_objs[a].describe() for a in m.areas},
            "connectivity_report": m.store.conn.report(m.specs, m.store.order),
            "role_mapping_note_ko": "역할 배치는 생물학적 층별 기능이 확정되었다는 주장이 아니다."})

    def can_store_h(self, area: str) -> bool:
        n_train = len(self.data.items("train"))
        dim = self.model.specs[f"{area}/L3"].n
        return n_train * dim * 4 <= int(self.cfg["records"]["store_train_h_max_mb"]) * 2 ** 20

    def pass_split(self, area: str, split: str, *, bank: PrototypeBank | None = None, override: dict[str, LayerRuntime] | None = None,
                   collect_h: bool = False, routing_mode: str | None = None, guide_transform: Callable[[Any], Any] | None = None,
                   routing_acc: dict[str, float] | None = None, collect_routing: bool = False) -> dict[str, Any]:
        """split 전체 순방향 (갱신 없음). 클래스 합·개수, (bank 가 있으면) 코사인 분류·정답 prototype 거리·혼동행렬.
        routing_acc 를 주면 V4 라우팅 선택 통계를 모은다. routing_mode·guide_transform 은 안내만 바꾸는 진단용 (감각 입력·q·prototype 고정)."""
        be, items = self.be, self.data.items(split)
        B = int(self.cfg["learning"]["batch_size"])
        key = f"{area}/L3"
        dim = self.model.specs[key].n
        C = len(self.classes)
        eps = float(self.cfg["learning"]["eps_norm"])
        sums = be.zeros((C, dim))
        counts = np.zeros(C, np.int64)
        out = {"n": 0, "correct": 0, "n_valid_h": 0, "n_ties": 0, "dist_sum": 0.0, "n_dist": 0, "cos_true_sum": 0.0, "n_cos": 0,
               "margin_sum": 0.0, "n_margin": 0, "per_class_n": np.zeros(C, np.int64), "per_class_correct": np.zeros(C, np.int64),
               "n_unknown": 0, "confusion": np.zeros((C, C + 1), np.int64)}
        H, Y = [], []
        rr = self.model.routing_runtime()
        want_route = rr is not None and AREA_INDEX[area] >= AREA_INDEX["V4"] and (routing_acc is not None or collect_routing)
        high = float(self.cfg["routing"]["high_guide_score"])
        routes: list[dict[str, np.ndarray]] = []
        for a in range(0, len(items), B):
            chunk = items[a:a + B]
            lgn, y = self.loader.batch(chunk, split)
            with self.timer.phase("forward", gpu=be.is_cuda and bool(self.cfg["compute"]["gpu_timing_sync"])):
                res_ = self.model.forward(lgn, upto=area, override=override, routing_mode=routing_mode, guide_transform=guide_transform,
                                          routing_record=want_route)
                h = res_[key]["out"]
            if want_route:
                rn = routing_rec_np(be, res_["V4/L2"]["routing"])
                if routing_acc is not None:
                    acc_add(routing_acc, routing_batch_stats(rr.table, rn, high))
                if collect_routing:
                    routes.append(rn)
            for c in np.unique(y):
                be.add_row_(sums, int(c), be.sum0(be.rows(h, be.i64(np.nonzero(y == c)[0]))))
            counts += np.bincount(y, minlength=C)
            out["n"] += len(chunk)
            np.add.at(out["per_class_n"], y, 1)
            if bank is not None:
                cl = cosine_classify(be, h, bank.P, bank.valid, eps)
                np.add.at(out["confusion"], (y, np.where(cl["pred"] >= 0, cl["pred"], C)), 1)
                ok = cl["pred"] == y
                out["correct"] += int(ok.sum())
                np.add.at(out["per_class_correct"], y[ok], 1)
                out["n_valid_h"] += int(cl["valid_h"].sum())
                out["n_unknown"] += int((cl["pred"] < 0).sum())
                out["n_ties"] += int(cl["n_ties"])
                vp = bank.valid[y]
                if vp.any():
                    d = be.to_np(be.norm_last(bank.targets(y) - h)).astype(np.float64)
                    out["dist_sum"] += float(d[vp].sum())
                    out["n_dist"] += int(vp.sum())
                ct = cl["cos"][np.arange(len(y)), y]
                fin = np.isfinite(ct)
                out["cos_true_sum"] += float(ct[fin].sum())
                out["n_cos"] += int(fin.sum())
                for i in np.nonzero(fin)[0]:
                    other = np.delete(cl["cos"][i], y[i])
                    other = other[np.isfinite(other)]
                    if other.size:
                        out["margin_sum"] += float(ct[i] - other.max())
                        out["n_margin"] += 1
            else:
                out["n_valid_h"] += int((be.to_np(be.norm_last(h)) > eps).sum())
            if collect_h:
                H.append(be.to_np(h).astype(np.float32))
                Y.append(y)
        if split == "train":
            self.counters["train_passes"] += 1
        out.update(sums=sums, counts=counts, ids_sha=self.data.ids_sha(split),
                   H=np.concatenate(H) if H else None, Y=np.concatenate(Y) if Y else None,
                   routes={k: (np.concatenate([r_[k] for r_ in routes]) if routes and routes[0].get(k) is not None else None)
                           for k in ("k", "B", "A", "S", "no_guidance")} if routes else None)
        return out

    @staticmethod
    def mean_dist(P: np.ndarray, valid: np.ndarray, H: np.ndarray, Y: np.ndarray) -> float | None:
        vp = valid[Y]
        if not vp.any():
            return None
        return float(np.linalg.norm(P[Y[vp]].astype(np.float64) - H[vp].astype(np.float64), axis=1).mean())

    def init_prototypes(self, area: str) -> dict[str, Any]:
        be = self.be
        dim = self.model.specs[f"{area}/L3"].n
        store_h = self.can_store_h(area)
        tp = self.pass_split(area, "train", collect_h=store_h)
        counts = tp["counts"]
        P = tp["sums"] * be.f32((1.0 / np.maximum(counts, 1))[:, None])
        bank = PrototypeBank(area, self.classes, dim, be, float(self.cfg["learning"]["eps_norm"]))
        rec = bank.set(P, counts, source="init_class_mean", split="train", sample_ids_sha256=tp["ids_sha"], epoch=None, rho=None)
        self.protos[area] = bank
        rec["train_zero_vector_fraction"] = 1.0 - tp["n_valid_h"] / max(1, tp["n"])
        if store_h:
            dist = self.mean_dist(be.to_np(bank.P), bank.valid, tp["H"], tp["Y"])
        else:
            dist = self.pass_split(area, "train", bank=bank)
            dist = dist["dist_sum"] / dist["n_dist"] if dist["n_dist"] else None
        self.stage[area]["dist_start_vs_P"] = dist
        self.record_prototypes(area, rec)
        if rec["invalid_classes"]:
            self.rec.warn(f"{area} prototype 무효 (영벡터) 클래스: {rec['invalid_classes']} — 해당 클래스 표본은 교정하지 않는다.")
        return rec

    def record_prototypes(self, area: str, rec: dict[str, Any]) -> None:
        bank = self.protos[area]
        r = self.need_rec()
        Pn = self.be.to_np(bank.P).astype(np.float32)
        r.npz(f"prototype_snapshots/{area}_v{bank.version:04d}.npz", P=Pn, valid=bank.valid, counts=bank.counts,
              classes=np.asarray(self.classes))
        pc = bank.pairwise_cos()
        C = len(self.classes)
        off = pc[~np.eye(C, dtype=bool)] if C > 1 else np.zeros(0)
        min_angle = float(np.degrees(np.arccos(np.clip(np.nanmax(off), -1, 1)))) if off.size and np.isfinite(off).any() else None
        for c, name in enumerate(self.classes):
            others = np.delete(pc[c], c)
            others = others[np.isfinite(others)]
            r.append_csv("prototype_metrics.csv", {
                "utc": rec["utc"], "area": area, "version": bank.version, "source": rec["source"], "epoch": rec["epoch"],
                "split": rec["split"], "class": name, "n_samples": int(bank.counts[c]), "norm": rec["norms"][c],
                "valid": bool(bank.valid[c]), "max_cos_to_other": float(others.max()) if others.size else None,
                "min_pair_angle_deg_all": min_angle, "update_l2_all": rec["update_l2"], "rho": rec["rho"],
                "collapse_warning": bool(min_angle is not None and min_angle < 5.0),
                "train_zero_vector_fraction": rec.get("train_zero_vector_fraction")})

    def model_version(self, area: str | None = None) -> str:
        area = area or (self.model.active if self.model else None)
        if area is None:
            return "no_model"
        st = self.stage.get(area, {})
        bank = self.protos.get(area)
        qsha = sha256_array(self.model.store.q_numpy())[:10]
        tag = ""
        if "V4" in self.model.areas and self.model.routing_mode != "legacy":
            tag = "-Rgeo" if self.model.routing_mode == "geometry_only" else "-Rv1g"
        return f"{area}-e{st.get('epochs_completed', 0)}-b{st.get('next_batch', 0)}-P{bank.version if bank else 0}-q{qsha}{tag}"

    def evaluate(self, area: str, split: str, bank: PrototypeBank, *, purpose: str,
                 override: dict[str, LayerRuntime] | None = None, log: bool = True, routing_mode: str | None = None,
                 guide_transform: Callable[[Any], Any] | None = None, collect_routing: bool = False) -> dict[str, Any]:
        if split == "dev":
            self.counters["dev_evaluations"] += 1
        elif split == "test":
            self.counters["test_evaluations"] += 1
        dim = self.model.specs[f"{area}/L3"].n
        n_items = len(self.data.items(split))
        collect = n_items * dim * 4 <= int(self.cfg["records"]["store_train_h_max_mb"]) * 2 ** 20
        racc: dict[str, float] = {}
        r = self.pass_split(area, split, bank=bank, override=override, collect_h=collect, routing_mode=routing_mode,
                            guide_transform=guide_transform, routing_acc=racc, collect_routing=collect_routing)
        n = r["n"]
        m: dict[str, Any] = {"split": split, "area": area, "purpose": purpose, "n": n, "correct": r["correct"],
                             "accuracy_overall": (r["correct"] / n) if n else None,
                             "n_valid_h": r["n_valid_h"], "n_unknown_zero_vector": r["n_unknown"],
                             "accuracy_valid_only": (r["correct"] / r["n_valid_h"]) if r["n_valid_h"] else None,
                             "n_ties": r["n_ties"], "mean_dist_to_true_prototype": (r["dist_sum"] / r["n_dist"]) if r["n_dist"] else None,
                             "mean_cos_to_true_prototype": (r["cos_true_sum"] / r["n_cos"]) if r["n_cos"] else None,
                             "mean_margin": (r["margin_sum"] / r["n_margin"]) if r["n_margin"] else None,
                             "per_class": {c: {"n": int(r["per_class_n"][i]), "correct": int(r["per_class_correct"][i])}
                                           for i, c in enumerate(self.classes)},
                             "model_version": self.model_version(area), "prototype_version": bank.version,
                             "per_class_accuracy": {c: (int(r["per_class_correct"][i]) / int(r["per_class_n"][i])) if r["per_class_n"][i] else None
                                                    for i, c in enumerate(self.classes)},
                             "confusion": r["confusion"].tolist(), "confusion_columns": self.classes + ["unknown"],
                             "routing_mode_used": routing_mode or self.cfg["routing"]["mode"], "guide_transformed": guide_transform is not None}
        rr = self.model.routing_runtime()
        if racc and rr is not None:
            m["routing"] = routing_finalize(racc, rr.K)
        if collect_routing:
            m["routing_records"] = r["routes"]
        m["geometry"] = self.geometry(r["H"], r["Y"]) if r["H"] is not None else {"status": NOT_RUN, "reason_ko": "메모리 상한으로 h 미보관"}
        if log and self.rec is not None:
            self.rec.append_jsonl("evaluations.jsonl", {"utc": utc_now(), **{k: v for k, v in m.items() if k not in ("per_class", "routing_records")},
                                                        "dev_evaluations": self.counters["dev_evaluations"],
                                                        "test_evaluations": self.counters["test_evaluations"]})
        return m

    def geometry(self, H: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
        eps = float(self.cfg["learning"]["eps_norm"])
        C = len(self.classes)
        Hd = H.astype(np.float64)
        n = np.linalg.norm(Hd, axis=1)
        ok = n > eps
        g: dict[str, Any] = {"zero_vector_fraction": float(1.0 - ok.mean()) if len(ok) else None,
                             "output_variance_mean": float(Hd.var(axis=0).mean()) if len(Hd) else None,
                             "output_norm_mean": float(n.mean()) if len(n) else None}
        means = np.zeros((C, H.shape[1]))
        have = np.zeros(C, bool)
        within = []
        for c in range(C):
            sel = (Y == c) & ok
            if sel.any():
                means[c] = Hd[sel].mean(axis=0)
                have[c] = np.linalg.norm(means[c]) > eps
                if have[c]:
                    mu = means[c] / np.linalg.norm(means[c])
                    within += list((Hd[sel] / n[sel, None]) @ mu)
        g["within_class_mean_cos_to_class_mean"] = float(np.mean(within)) if within else None
        if have.sum() >= 2:
            Mn = means[have] / np.linalg.norm(means[have], axis=1, keepdims=True)
            pc = Mn @ Mn.T
            off = pc[~np.eye(len(Mn), dtype=bool)]
            g["between_class_mean_cos"] = float(off.mean())
            g["between_class_max_cos"] = float(off.max())
            d = np.linalg.norm(means[have][:, None, :] - means[have][None, :, :], axis=2)
            g["between_class_mean_distance"] = float(d[~np.eye(len(Mn), dtype=bool)].mean())
        return g

    # ---------------------------------------------------------------- 단계 전환
    def add_next_area_prepare(self) -> tuple[str, list[LayerSpec], dict[str, Any], dict[str, Any]]:
        cur = self.model.active
        st = self.stage[cur]
        info = {"current_area": cur, "current_status": st["status"], "epochs_completed": st["epochs_completed"],
                "best_dev_accuracy": st["best_dev_accuracy"],
                "warning_ko": (None if st["status"] == "success_criterion_met" else
                               f"{cur} 단계 상태가 '{st['status']}' 이다 (설정한 성공 기준 충족이 아니다). 그래도 추가할 수 있으나 기록한다.")}
        area, specs, summ = self.plan_next()
        return area, specs, summ, info

    def freeze_area(self, area: str) -> dict[str, Any]:
        st = self.model.store
        qs = {lay: sha256_array(self.be.to_np(st.q_view(f"{area}/{lay}"))) for lay in LAYER_ORDER}
        h = {"q_sha256": qs, "prototype_sha256": self.protos[area].sha(), "prototype_version": self.protos[area].version,
             "structure_sha256": self.model.struct_hash[area], "preprocessing_sha256": sha256_text(dumps(self.pre.info(), indent=None)),
             "frozen_utc": utc_now(), "model_version_at_freeze": self.model_version(area)}
        self.frozen[area] = h
        return h

    def frozen_check(self) -> dict[str, Any]:
        out = {}
        for area, h in self.frozen.items():
            st = self.model.store
            qs = {lay: sha256_array(self.be.to_np(st.q_view(f"{area}/{lay}"))) for lay in LAYER_ORDER}
            out[area] = {"q_unchanged": qs == h["q_sha256"], "prototype_unchanged": self.protos[area].sha() == h["prototype_sha256"],
                         "structure_unchanged": self.model.struct_hash[area] == h["structure_sha256"],
                         "preprocessing_unchanged": sha256_text(dumps(self.pre.info(), indent=None)) == h["preprocessing_sha256"]}
        return out

    def add_next_area_commit(self, area: str, specs: list[LayerSpec], summ: dict[str, Any], info: dict[str, Any],
                             *, user_choice: str) -> dict[str, Any]:
        cur = self.model.active
        hashes = self.freeze_area(cur)
        tr = {"utc": utc_now(), "from": cur, "to": area, "from_status": info["current_status"], "warning_ko": info["warning_ko"],
              "user_choice": user_choice, "frozen_hashes": hashes}
        self.stage[cur]["transitions"].append(tr)
        self.need_rec().append_jsonl("stage_transitions.jsonl", tr)
        rec = self.build_area(area, specs, summ)
        return {"transition": tr, "prototype_init": rec}

    # ---------------------------------------------------------------- checkpoint
    def flush_routing_samples(self) -> None:
        if not self.route_buf:
            return
        r = self.need_rec()
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for c in self.route_buf:
            groups.setdefault((c["area"], c["epoch"]), []).append(c)
        for (area, ep), rows in groups.items():
            k = len(list(r.p("routing").glob(f"routing_samples_{area}_e{ep:03d}_part*.npz")))
            r.npz(f"routing/routing_samples_{area}_e{ep:03d}_part{k:03d}.npz",
                  sample_id=np.asarray([x["sample_id"] for x in rows]), batch_index=np.asarray([x["batch"] for x in rows], np.int64),
                  label_recorded_not_used_for_selection=np.asarray([x["label"] for x in rows], np.int64),
                  P=np.asarray(rows[0]["P"]), candidate_Q=np.asarray(rows[0]["cand_q"]), selected_slot=np.stack([x["k"] for x in rows]),
                  selected_Q=np.stack([x["q_sel"] for x in rows]), guide_score=np.stack([x["S"] for x in rows]) if rows[0]["S"] is not None else np.zeros(0),
                  no_guidance=np.stack([x["no_guidance"] for x in rows]), A=np.stack([x["A"] for x in rows]), B=np.stack([x["B"] for x in rows]),
                  u=np.stack([np.minimum(x["A"], x["B"]) * (~x["missing"]) for x in rows]), missing_candidate=np.asarray(rows[0]["missing"]),
                  routing_mode=np.asarray(rows[0]["mode"]), guide_score_used_for_selection=np.asarray(bool(rows[0]["S_used_for_selection"])),
                  **({f"selected_path_{k_}": np.stack([x["points"][k_] for x in rows]) for k_ in rows[0]["points"]}
                     if rows[0].get("points") is not None else {}))
        self.route_buf = []

    def flush_corrections(self) -> None:
        if not self.corr_buf:
            return
        r = self.need_rec()
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for c in self.corr_buf:
            groups.setdefault((c["area"], c["epoch"]), []).append(c)
        for (area, ep), rows in groups.items():
            k = len(list(r.p("correction_samples").glob(f"{area}_e{ep:03d}_part*.npz")))
            r.npz(f"correction_samples/{area}_e{ep:03d}_part{k:03d}.npz",
                  sample_id=np.asarray([x["sample_id"] for x in rows]), label=np.asarray([x["label"] for x in rows], np.int64),
                  batch_index=np.asarray([x["batch"] for x in rows], np.int64),
                  sensory_h_before=np.stack([x["h_before"] for x in rows]), teacher_old_target=np.stack([x["target"] for x in rows]),
                  sensory_h_after=np.stack([x["h_after"] for x in rows]),
                  delta_norm=np.asarray([x["delta_norm"] for x in rows]), m_L2_norm=np.asarray([x["m2_norm"] for x in rows]),
                  m_L3_norm=np.asarray([x["m3_norm"] for x in rows]), batch_dq_L2_applied_l2=np.asarray([x["dq2"] for x in rows]),
                  batch_dq_L3_applied_l2=np.asarray([x["dq3"] for x in rows]),
                  model_version_before=np.asarray([x["version"] for x in rows]))
        self.corr_buf = []

    def save_checkpoint(self, tag: str, keep: bool = False) -> Path:
        with self.timer.phase("save"):
            return self._save_checkpoint(tag, keep)

    def _save_checkpoint(self, tag: str, keep: bool) -> Path:
        r = self.need_rec()
        self.flush_corrections()
        self.flush_routing_samples()
        self.ckpt_seq += 1
        name = f"ckpt_{self.ckpt_seq:06d}_{re.sub(r'[^A-Za-z0-9_]+', '_', tag)}"
        arrays: dict[str, np.ndarray] = {"q": self.model.store.q_numpy()}
        for area, bank in self.protos.items():
            arrays[f"P__{area}"] = self.be.to_np(bank.P).astype(np.float32)
            arrays[f"Pvalid__{area}"] = bank.valid
            arrays[f"Pcounts__{area}"] = bank.counts
        npz_path = r.p(f"checkpoints/{name}.npz")
        save_npz(npz_path, **arrays)
        meta = {"format": CKPT_FORMAT, "seq": self.ckpt_seq, "tag": tag, "keep": bool(keep), "utc": utc_now(),
                "program_version": VERSION, "program_sha256": self.program_sha(), "npz": npz_path.name, "npz_sha256": sha256_file(npz_path),
                "areas": self.model.areas, "active_area": self.model.active,
                "layer_ranges": {k: [self.model.specs[k].start, self.model.specs[k].end] for k in self.model.store.order},
                "structure_sha256": self.model.struct_hash, "config_structure_sha256": config_structure_sha(self.cfg),
                "config_learning_sha256": config_learning_sha(self.cfg),
                "data_fingerprint_sha256": self.data.manifest["data_fingerprint_sha256"], "split_sha256": self.data.manifest["split_sha256"],
                "stage": self.stage, "frozen": self.frozen, "counters": self.counters,
                "prototypes": {a: {"version": b.version, "history": b.history} for a, b in self.protos.items()},
                "rng_state": self.rng.bit_generator.state, "model_version": self.model_version(), "q_sha256": sha256_array(arrays["q"]),
                "sampler_ko": "에폭 순서 = default_rng([seed, 영역번호, epoch, 11]).permutation(n_train); 진행 위치 = stage[영역].next_batch",
                "routing": {"mode": self.cfg["routing"]["mode"], "params": self.cfg["routing"],
                            "table_sha256": (self.model.specs["V4/L2"].routing.hash() if ("V4/L2" in self.model.specs and
                                                                                          self.model.specs["V4/L2"].routing is not None) else None),
                            "conventions": ROUTING_CONVENTION_KO, "program_sha256": self.program_sha()}}
        r.json(f"checkpoints/{name}.json", meta)
        r.json("checkpoints/latest.json", {"name": name, "utc": meta["utc"], "model_version": meta["model_version"]})
        self.last_checkpoint = name
        self._prune_checkpoints()
        return npz_path

    def _prune_checkpoints(self) -> None:
        r = self.need_rec()
        keep_n = int(self.cfg["records"]["keep_periodic_checkpoints"])
        metas = sorted(r.p("checkpoints").glob("ckpt_*.json"))
        periodic = []
        for p in metas:
            try:
                m = read_json(p)
            except (OSError, json.JSONDecodeError):
                continue
            if not m.get("keep") and p.stem != self.last_checkpoint:
                periodic.append(p)
        for p in periodic[:max(0, len(periodic) - keep_n)]:
            with contextlib.suppress(OSError):
                (p.with_suffix(".npz")).unlink()
                p.unlink()

    @classmethod
    def load(cls, run_dir: Path, be: Backend, *, say: Callable[[str], None] = print, confirm: Callable[[str], bool] | None = None,
             data_root: Path | None = None, checkpoint: str | None = None, read_only: bool = False) -> "Session":
        """read_only=True: 분기 원본으로만 읽는다 (원본 폴더의 교정 조각 정리·csv 동기화·로그를 하지 않는다)."""
        run_dir = Path(run_dir)
        cf = read_json(run_dir / "config.json")
        sess = cls(saved_config_with_mode(cf["config"]), be, say=say, confirm=confirm)
        sess.rec = RunRecorder(run_dir, say=say)
        info = read_json(run_dir / "run_info.json")
        sess.output_root = Path(info["output_root"]) if info.get("output_root") else run_dir.parent
        droot = Path(data_root) if data_root else (Path(info["data_root"]) if info.get("data_root") else None)
        if droot is None or not droot.is_dir():
            raise CompatibilityError(f"데이터 폴더를 찾을 수 없다 ({droot}). --data-root 또는 메뉴에서 새 위치를 준다.")
        sess.data_root = droot
        if cf.get("program_sha256") and cf["program_sha256"] != sess.program_sha():
            msg = "이 실행을 만든 프로그램 파일과 현재 파일의 SHA256 이 다르다 (코드 변경). 결과 해석에 주의한다."
            if read_only:
                say("[경고] " + msg)
            else:
                sess.rec.warn(msg)
        man = read_json(run_dir / "split_manifest.json")
        sess.data = DatasetManager.from_manifest(sess.cfg, droot, man)
        probs = sess.data.verify_files()
        if probs:
            raise CompatibilityError("데이터가 분할 목록과 다르다: " + "; ".join(probs[:5]))
        sess.loader = BatchLoader(sess.data, sess.pre, be, sess.cfg, sess.timer)
        name = checkpoint or read_json(run_dir / "checkpoints" / "latest.json")["name"]
        meta = read_json(run_dir / "checkpoints" / f"{name}.json")
        if meta.get("format") not in CKPT_FORMATS_READABLE:
            raise CompatibilityError(f"알 수 없는 checkpoint 형식: {meta.get('format')}")
        npz = run_dir / "checkpoints" / meta["npz"]
        if sha256_file(npz) != meta["npz_sha256"]:
            raise CompatibilityError(f"checkpoint 배열 파일 해시가 다르다: {npz}")
        if meta["config_structure_sha256"] != config_structure_sha(sess.cfg) or meta["config_learning_sha256"] != config_learning_sha(sess.cfg):
            raise CompatibilityError("checkpoint 와 config.json 의 설정 해시가 다르다.")
        if meta["data_fingerprint_sha256"] != man["data_fingerprint_sha256"] or meta["split_sha256"] != man["split_sha256"]:
            raise CompatibilityError("checkpoint 와 분할 목록이 다르다.")
        arrays = load_npz(npz)
        sess.model = CortexModel(sess.cfg, be, sess.pre)
        sess.model.timer = sess.timer
        for area in meta["areas"]:
            specs = sess.model.plan(area)
            sess.model.add_area(area, specs)
            if sess.model.struct_hash[area] != meta["structure_sha256"][area]:
                raise CompatibilityError(f"{area} 구조를 다시 만들었더니 해시가 다르다 (설정·코드 불일치).")
        sess.model.store.load_q(arrays["q"])
        sess.model.active = meta["active_area"]
        for area, pinfo in meta["prototypes"].items():
            dim = sess.model.specs[f"{area}/L3"].n
            bank = PrototypeBank(area, sess.classes, dim, be, float(sess.cfg["learning"]["eps_norm"]))
            bank.P = be.f32(arrays[f"P__{area}"])
            bank.valid = np.asarray(arrays[f"Pvalid__{area}"], bool)
            bank.counts = np.asarray(arrays[f"Pcounts__{area}"], np.int64)
            bank.version = int(pinfo["version"])
            bank.history = list(pinfo["history"])
            sess.protos[area] = bank
        sess.stage = meta["stage"]
        sess.frozen = meta["frozen"]
        sess.counters = meta["counters"]
        sess.rng.bit_generator.state = meta["rng_state"]
        for st_ in sess.stage.values():                                   # v1 기록에는 새 키가 없다
            for k_, v_ in new_stage_state(st_["area"], sess.cfg).items():
                st_.setdefault(k_, v_)
        sess.ckpt_seq = int(meta["seq"])
        sess.last_checkpoint = name
        if not read_only:
            sess._discard_uncommitted_corrections()
            sess.sync_training_csv()
            sess.rec.info(f"재개: {name} ({meta['model_version']}, 라우팅 {sess.cfg['routing']['mode']})", echo=True)
        return sess

    def _discard_uncommitted_corrections(self) -> None:
        """checkpoint 위치보다 뒤의 교정 표본 조각 (중단 전에 쓰였지만 되돌린 배치) 을 지운다 → 재개 뒤 중복 기록 방지."""
        r = self.need_rec()
        for area, st in self.stage.items():
            for p in r.p("correction_samples").glob(f"{area}_e*_part*.npz"):
                ep = int(re.search(r"_e(\d+)_part", p.name).group(1))
                if ep < st["epochs_completed"]:
                    continue
                if ep > st["epochs_completed"]:
                    p.unlink()
                    continue
                z = load_npz(p)
                if z["batch_index"].size and int(z["batch_index"].max()) >= int(st["next_batch"]) and st["phase"] == "training":
                    p.unlink()
            for p in r.p("routing").glob(f"routing_samples_{area}_e*_part*.npz"):
                ep = int(re.search(r"_e(\d+)_part", p.name).group(1))
                if ep > st["epochs_completed"]:
                    p.unlink()
                elif ep == st["epochs_completed"] and st["phase"] == "training":
                    z = load_npz(p)
                    if z["batch_index"].size and int(z["batch_index"].max()) >= int(st["next_batch"]):
                        p.unlink()

    def sync_training_csv(self) -> None:
        r = self.need_rec()
        have = {(row.get("area"), row.get("epoch")) for row in r.read_csv("training.csv")}
        for area in AREA_ORDER:
            st = self.stage.get(area) or {}
            rows = ([st["start_eval"]] if st.get("start_eval") else []) + list(st.get("history", []))
            for row in rows:
                if (row["area"], str(row["epoch"])) not in have:
                    r.append_csv("training.csv", row)

    @classmethod
    def branch(cls, src_run: Path, be: Backend, *, mode: str, say: Callable[[str], None] = print,
               confirm: Callable[[str], bool] | None = None, data_root: Path | None = None, output_root: Path | None = None,
               keep_upto: str = "V2") -> "Session":
        """기존 실행의 V1·V2 (q·prototype·동결 상태) 를 재사용하고, 다른 라우팅 모드로 V4 를 새로 만드는 새 실행 (원본은 바꾸지 않는다).
        V4 출력 정의가 바뀌므로 V4 prototype 은 train 으로 새로 초기화하고 IT 는 이후 다시 만든다."""
        src = cls.load(Path(src_run), be, say=say, confirm=confirm, data_root=data_root, read_only=True)
        keep = list(AREA_ORDER[:AREA_INDEX[keep_upto]])
        lack = [a for a in keep if a not in src.model.areas]
        if lack:
            raise CompatibilityError(f"분기 원본에 {lack} 가 없다 (V1·V2 까지 만든 실행이 필요하다).")
        cfg = copy.deepcopy(src.cfg)
        cfg["routing"]["mode"] = mode
        new = cls(validate_config(cfg), be, say=say, confirm=confirm)
        new.set_paths(src.data_root, output_root or src.output_root)
        new.start_run()
        r = new.rec
        for fname in ("data_report.json", "split_manifest.json", "class_to_index.json", "preprocessing.json"):
            p = src.rec.p(fname)
            if p.is_file():
                r.json(fname, read_json(p))
        new.data = DatasetManager.from_manifest(new.cfg, src.data_root, src.data.manifest)
        new.loader = BatchLoader(new.data, new.pre, be, new.cfg, new.timer)
        new.model = CortexModel(new.cfg, be, new.pre)
        new.model.timer = new.timer
        for area in keep:
            new.model.add_area(area, new.model.plan(area))
            if new.model.struct_hash[area] != src.model.struct_hash[area]:
                raise CompatibilityError(f"{area} 구조 해시가 원본과 다르다 (설정·코드 불일치) — 분기하지 않는다.")
        qn, qs = new.model.store.q_numpy(), src.model.store.q_numpy()
        for area in keep:
            for lay in LAYER_ORDER:
                a_, b_ = new.model.specs[f"{area}/{lay}"], src.model.specs[f"{area}/{lay}"]
                qn[a_.start:a_.end] = qs[b_.start:b_.end]
        new.model.store.load_q(qn)
        for area in keep:
            sb = src.protos[area]
            bank = PrototypeBank(area, new.classes, sb.dim, be, sb.eps)
            bank.P, bank.valid, bank.counts = be.f32(be.to_np(sb.P)), sb.valid.copy(), sb.counts.copy()
            bank.version, bank.history = sb.version, copy.deepcopy(sb.history)
            new.protos[area] = bank
            new.stage[area] = copy.deepcopy(src.stage[area])
        new.frozen = {a: copy.deepcopy(v) for a, v in src.frozen.items() if a in keep}
        bad = {a: v for a, v in new.frozen_check().items() if not all(v.values())}
        if bad:
            raise CompatibilityError(f"분기 원본의 동결 해시와 다시 만든 상태가 다르다: {bad}")
        new.model.active = keep[-1]
        info = {"utc": utc_now(), "kind": "branch", "source_run": str(src.rec.dir), "source_checkpoint": src.last_checkpoint,
                "source_model_version": src.model_version(), "source_routing_mode": src.cfg["routing"]["mode"], "new_routing_mode": mode,
                "reused_areas": keep, "source_counters": src.counters,
                "note_ko": "V1·V2 의 q·prototype·동결 해시를 그대로 옮겼다. 원본 실행 폴더는 바꾸지 않았다."}
        r.json("run_info.json", {**read_json(r.p("run_info.json")), "branched_from": info})
        r.append_jsonl("stage_transitions.jsonl", info)
        new.write_architecture()
        new.save_checkpoint("branch_base", keep=True)
        nxt, specs, summ, pinfo = new.add_next_area_prepare()
        new.add_next_area_commit(nxt, specs, summ, pinfo, user_choice=f"branch_from:{Path(src_run).name}")
        src.rec.close()
        return new

    def inspect_neuron(self, nid: int, item: dict[str, Any] | None = None) -> dict[str, Any]:
        dyn = None
        if item is not None:
            key, _j = self.model.store.locate(int(nid))
            lgn, _y = self.loader.batch([item], item.get("split", "external"))
            res = self.model.forward(lgn, upto=key.split("/")[0], keep=(key,), routing_record=key.startswith("V4"))
            dyn = {key: {"conf": self.be.to_np(res[key]["conf"])[0], "out": self.be.to_np(res[key]["out"])[0]}}
            if "V4/L2" in res and "routing" in res["V4/L2"]:
                rn = routing_rec_np(self.be, res["V4/L2"]["routing"])
                dyn["__routing__"] = {k: (v[0] if isinstance(v, np.ndarray) else v) for k, v in rn.items()}
        return self.model.inspector.inspect_neuron(int(nid), dyn)


# ======================================================================
# 13. 단계 학습 (StageTrainer)
# ======================================================================
def epoch_permutation(seed: int, area: str, epoch: int, n: int) -> np.ndarray:
    return np.random.default_rng([int(seed), AREA_INDEX[area], int(epoch), 11]).permutation(n)


OOM_GUIDE_KO = ("GPU 메모리 부족. 현재 배치는 되돌렸고 gather 조각 크기를 반씩 줄여 같은 배치를 다시 시도했지만 실패했다. "
                "마지막 경계에서 checkpoint 를 저장했다. CPU 로 몰래 바꾸지 않는다.\n"
                "  줄이는 방법 (새 실행에서 설정 변경): learning.batch_size ↓, compute.max_gather_elements ↓, "
                "areas.V1.n_r/n_phi ↓ 또는 logpolar.n_r/n_phi ↓ (V1 팬인), areas.V1.rf_radius_sigma ↓, data.image_size ↓.\n"
                "  같은 설정으로 이어 가려면 다른 GPU 프로그램을 닫고 메뉴 4 (재개) 로 다시 시작한다.")


class StageTrainer:
    """현재 활성 영역 하나만 q 학습과 prototype 갱신을 한다. 앞 영역은 동결 (출력·L6 g 는 매 입력 다시 계산).

    배치 순환 (같은 입력):
      (1) 동적 상태 초기화 (forward 는 표본마다 새로 계산)  (2) 자유 순방향 → h_before, s, CONF, g
      (3) 동결 prototype 과 비교: δ = P_true − h_before      (4) LocalErrorRouter → L1Relay (주소 확인)
      (5) LocalOutputLearner: L2·L3 Δq 를 같은 스냅샷에서 계산해 동시에 적용
      (6) 같은 입력 재순방향 → h_after (실제 네트워크 출력)   (7) 같은 이전 prototype 기준 개선·악화·무변화
    에폭 끝: q 고정, train 재순방향 → 클래스 평균 → P ← (1−ρ)P + ρ·h̄ (한 번만) → dev 평가 (갱신 없음) → 단계 상태 판단.
    """

    def __init__(self, sess: Session):
        self.s = sess

    # ---------------------------------------------------------------- 진입점
    def run(self, *, extra_epochs: int | None = None, stop_after_batches: int | None = None) -> dict[str, Any]:
        s = self.s
        area = s.model.active
        st = s.stage[area]
        if extra_epochs is not None:
            new_max = int(st["epochs_completed"]) + int(extra_epochs)
            st["budget_changes"].append({"utc": utc_now(), "from_max_epochs": st["max_epochs"], "to_max_epochs": new_max,
                                         "from_status": st["status"]})
            st["max_epochs"] = new_max
            if st["status"] in STAGE_TERMINAL:
                st["bad_epochs"] = 0
        elif st["status"] in STAGE_TERMINAL:
            return {"status": st["status"], "area": area,
                    "message_ko": f"{area} 단계는 이미 '{st['status']}' 로 끝났다. 더 학습하려면 추가 에폭 수를 준다."}
        if st["epochs_completed"] >= st["max_epochs"] and st["phase"] == "training":
            st["status"] = "budget_exhausted"
            return {"status": st["status"], "area": area, "message_ko": "에폭 예산이 0 이다 (추가 에폭을 준다)."}
        st["status"] = "running"
        st["started_utc"] = st["started_utc"] or utc_now()
        with StopFlag(s.say) as stop:
            try:
                return self._loop(area, st, stop, stop_after_batches)
            except KeyboardInterrupt:
                return self._stop(area, st, "interrupted_manual", "Ctrl+C 즉시 중단: 진행 중 배치는 되돌렸다.")

    def _stop(self, area: str, st: dict[str, Any], status: str, why: str) -> dict[str, Any]:
        st["status"] = status
        self.s.save_checkpoint(f"{area}_{status}", keep=True)
        self.s.rec.info(f"{area} 학습 멈춤 ({status}): {why}", echo=True)
        return {"status": status, "area": area, "message_ko": why, "checkpoint": self.s.last_checkpoint,
                "epochs_completed": st["epochs_completed"], "next_batch": st["next_batch"],
                "best_dev_accuracy": st["best_dev_accuracy"], "best_epoch": st["best_epoch"]}

    # ---------------------------------------------------------------- 에폭 반복
    def _loop(self, area: str, st: dict[str, Any], stop: StopFlag, stop_after: int | None) -> dict[str, Any]:
        s, be = self.s, self.s.be
        items = s.data.items("train")
        n, B = len(items), int(s.cfg["learning"]["batch_size"])
        n_batches = int(math.ceil(n / B))
        k2, k3 = f"{area}/L2", f"{area}/L3"
        router = s.model.routers[area]
        relay = L1Relay(area, s.model.store)
        learner = LocalOutputLearner(s.cfg, be, s.model.store)
        every = int(s.cfg["records"]["checkpoint_every_batches"])
        done_now = 0
        while st["epochs_completed"] < st["max_epochs"]:
            e = int(st["epochs_completed"])
            perm = epoch_permutation(int(s.cfg["seed"]), area, e, n)
            if st["phase"] == "training":
                while st["next_batch"] < n_batches:
                    if stop.requested:
                        return self._stop(area, st, "interrupted_manual", "Ctrl+C: 배치 경계에서 멈췄다.")
                    bi = int(st["next_batch"])
                    chunk = [items[i] for i in perm[bi * B:(bi + 1) * B]]
                    snap = {k: be.clone(s.model.store.q_view(k)) for k in (k2, k3)}
                    retries = 0
                    failure: BaseException | None = None
                    while True:
                        try:
                            out = self._batch(area, chunk, e, bi, router, relay, learner, stop)
                            with stop.critical():
                                self._commit(st, out)
                            failure = None
                            break
                        except KeyboardInterrupt:
                            self._restore(snap)
                            raise
                        except BaseException as exc:        # noqa: BLE001
                            self._restore(snap)
                            if is_oom_error(exc) and retries < int(s.cfg["routing"]["oom_retry_halvings"]) and be.max_gather > 4096:
                                retries += 1
                                old = be.max_gather
                                be.max_gather = max(4096, old // 2)
                                if torch is not None and be.is_cuda:
                                    torch.cuda.empty_cache()
                                s.rec.append_jsonl("oom_events.jsonl", {
                                    "utc": utc_now(), "area": area, "epoch": e, "batch": bi, "retry": retries, "max_gather_from": old,
                                    "max_gather_to": be.max_gather, "batch_size_unchanged": len(chunk), "error": f"{type(exc).__name__}: {exc}"[:300],
                                    "note_ko": "q 를 배치 전으로 되돌리고 같은 배치를 gather 조각만 줄여 다시 계산 (수식·배치 평균은 같다)"})
                                s.rec.warn(f"메모리 부족 → gather 조각 {old} → {be.max_gather} 로 같은 배치 재시도 ({retries})")
                                continue
                            failure = exc
                            break
                    if failure is not None:
                        if is_oom_error(failure):
                            s.rec.error("메모리 부족", failure)
                            r_ = self._stop(area, st, "oom", OOM_GUIDE_KO)
                            s.say(OOM_GUIDE_KO)
                            return r_
                        s.rec.error(f"{area} epoch {e} batch {bi} 오류 — q 를 배치 전으로 되돌리고 checkpoint 저장", failure)
                        st["status"] = "failed"
                        s.save_checkpoint(f"{area}_failed", keep=True)
                        raise failure
                    done_now += 1
                    step = max(1, n_batches // 10)
                    if n_batches >= 20 and st["next_batch"] % step == 0 and st["next_batch"] < n_batches:
                        a_ = st["acc"]
                        s.say(f"    [{area} epoch {e + 1}] 배치 {st['next_batch']}/{n_batches}, 거리 개선 비율 "
                              f"{fmt(a_['n_improved'] / max(1.0, a_['n_valid']))}")
                    if every and st["next_batch"] % every == 0 and st["next_batch"] < n_batches:
                        with stop.critical():
                            s.save_checkpoint(f"{area}_e{e}_b{st['next_batch']}")
                    if stop_after and done_now >= stop_after:
                        return self._stop(area, st, "interrupted_test", f"검사용: {done_now} 배치 뒤 멈춤")
                with stop.critical():
                    st["phase"] = "train_done"
                    s.save_checkpoint(f"{area}_e{e}_train_done")
            if stop.requested:
                return self._stop(area, st, "interrupted_manual", "Ctrl+C: 에폭 학습 끝, prototype 갱신 전에서 멈췄다 (재개하면 갱신부터).")
            row = self._epoch_end(area, st, stop)
            s.say(f"  [{area} epoch {row['epoch'] + 1}] dev 정확도 {pct_count(row['dev_correct'], row['dev_n'])} "
                  f"(유효 출력만 {fmt(row['dev_accuracy_valid_only'])}), 거리 개선 비율 {fmt(row['frac_improved'])}, "
                  f"q 효과 {fmt(row['q_effect_dist'])}, prototype 효과 {fmt(row['prototype_effect_dist'])} → {row['stage_status']}")
            if st["status"] in STAGE_TERMINAL:
                break
        if st["status"] not in STAGE_TERMINAL:
            st["status"] = "budget_exhausted"
        st["ended_utc"] = utc_now()
        st["structure_unchanged_at_end"] = (s.model.struct_hash[area] == st.get("structure_sha_at_creation")) \
            if st.get("structure_sha_at_creation") else None
        st["frozen_check_at_end"] = s.frozen_check()
        s.save_checkpoint(f"{area}_stage_{st['status']}", keep=True)
        return {"status": st["status"], "area": area, "epochs_completed": st["epochs_completed"],
                "best_dev_accuracy": st["best_dev_accuracy"], "best_epoch": st["best_epoch"], "checkpoint": s.last_checkpoint}

    def _restore(self, snap: dict[str, Any]) -> None:
        for k, v in snap.items():
            self.s.be.assign_(self.s.model.store.q_view(k), v)

    # ---------------------------------------------------------------- 한 배치
    def _batch(self, area: str, chunk: list[dict[str, Any]], e: int, bi: int, router: LocalErrorRouter, relay: L1Relay,
               learner: LocalOutputLearner, stop: StopFlag) -> dict[str, Any]:
        s, be = self.s, self.s.be
        k2, k3 = f"{area}/L2", f"{area}/L3"
        bank = s.protos[area]
        tm = s.timer
        lgn, y = s.loader.batch(chunk, "train")
        B = len(chunk)
        sha = sha256_text("\n".join(it["relpath"] for it in chunk))
        version = s.model_version(area)
        gsync = be.is_cuda and bool(s.cfg["compute"]["gpu_timing_sync"])
        rr = s.model.routing_runtime()
        want_route = rr is not None and AREA_INDEX[area] >= AREA_INDEX["V4"]
        if s.fault == "oom_once":
            s.fault = None
            raise RuntimeError("CUDA out of memory (자체 검사용 주입)")
        with tm.phase("forward", gpu=gsync):
            f0 = s.model.forward(lgn, upto=area, keep=(k2, k3), routing_record=want_route)
        h0 = f0[k3]["out"]
        route_stats, route_samples = None, []
        if want_route:
            rn = routing_rec_np(be, f0["V4/L2"]["routing"])
            route_stats = routing_batch_stats(rr.table, rn, float(s.cfg["routing"]["high_guide_score"]))
            st_ = s.stage[area]
            ro = s.cfg["routing"]
            quota = min(int(ro["samples_per_epoch"]) - int(st_["route_epoch_saved"]), int(ro["samples_max_per_stage"]) - int(st_["route_saved"]))
            t = rr.table
            ps = point_summary(rn["R_sel"], rn["OK_sel"]) if rn.get("R_sel") is not None else None
            for i in range(min(max(0, quota), B)):
                k_ = rn["k"][i]
                route_samples.append({"area": area, "epoch": e, "batch": bi, "sample_id": chunk[i]["relpath"], "label": int(y[i]),
                                      "P": t.P, "cand_q": t.cand_q, "k": k_, "q_sel": np.where(t.missing, -1, t.cand_q[np.arange(len(k_)), k_]),
                                      "S": (rn["S"][i].astype(np.float32) if rn["S"] is not None else None), "no_guidance": rn["no_guidance"][i],
                                      "A": rn["A"][i].astype(np.float32), "B": rn["B"][i].astype(np.float32), "missing": t.missing, "mode": t.mode,
                                      "points": ({k_p: v_p[i].astype(np.float32) for k_p, v_p in ps.items()} if ps is not None else None),
                                      "S_used_for_selection": rn["S_used_for_selection"]})
        with tm.phase("correction", gpu=be.is_cuda):
            vp = bank.valid[y]
            T = bank.targets(y)
            delta = (T - h0) * be.f32(vp.astype(np.float32)[:, None])          # 무효 prototype 표본은 δ=0
            m2_raw, m3_raw = router.route(delta)
            m2 = relay.deliver(relay.packet("L2", m2_raw, sha, version), B, be)
            m3 = relay.deliver(relay.packet("L3", m3_raw, sha, version), B, be)
            dq2 = learner.proposal(f0[k2]["s"], m2)
            dq3 = learner.proposal(f0[k3]["s"], m3)
        with tm.phase("update", gpu=be.is_cuda):
            with stop.critical():
                ust = learner.apply({k2: dq2, k3: dq3})
        if s.fault == "interrupt_after_apply":
            s.fault = None
            raise KeyboardInterrupt
        with tm.phase("forward", gpu=be.is_cuda):
            h1 = s.model.forward(lgn, upto=area)[k3]["out"]
        nrm = lambda x: be.to_np(be.norm_last(x)).astype(np.float64)          # noqa: E731
        dot = lambda a, b: be.to_np((a * b).sum(-1)).astype(np.float64)       # noqa: E731
        Tn, hn0, hn1 = nrm(T), nrm(h0), nrm(h1)
        d0, d1 = nrm(T - h0), nrm(T - h1)
        dh = h1 - h0
        dhn, dln = nrm(dh), nrm(delta)
        with np.errstate(divide="ignore", invalid="ignore"):
            c0 = dot(h0, T) / (hn0 * Tn)
            c1 = dot(h1, T) / (hn1 * Tn)
            cdd = dot(dh, delta) / (dhn * dln)
        act = {}
        for lay, key in (("L2", k2), ("L3", k3)):
            ss, gg, cf = f0[key]["s"], f0[key]["g"], f0[key]["conf"]
            ne = _numel(ss)
            act[lay] = {"active_frac": _scalar(be, (ss > 0).sum()) / ne, "s_mean": _scalar(be, ss.mean()),
                        "conf_mean": _scalar(be, cf.mean()),
                        "g_min_frac": (_scalar(be, (gg <= s.model.l6.g_min + 1e-7).sum()) / ne) if gg is not None else 0.0,
                        "g_max_frac": (_scalar(be, (gg >= s.model.l6.g_max - 1e-7).sum()) / ne) if gg is not None else 0.0}
        samples = []
        st = s.stage[area]
        rc = s.cfg["records"]
        quota = min(int(rc["correction_samples_per_epoch"]) - int(st["corr_epoch_saved"]),
                    int(rc["correction_samples_max_per_stage"]) - int(st["corr_saved"]))
        if quota > 0:
            sel = [i for i in range(B) if vp[i]][:quota]
            if sel:
                idx = be.i64(np.asarray(sel))
                H0 = be.to_np(be.rows(h0, idx))
                H1 = be.to_np(be.rows(h1, idx))
                TT = be.to_np(be.rows(T, idx))
                m2n, m3n = nrm(m2), nrm(m3)
                for k_, i in enumerate(sel):
                    samples.append({"area": area, "epoch": e, "batch": bi, "sample_id": chunk[i]["relpath"], "label": int(y[i]),
                                    "h_before": H0[k_].astype(np.float32), "target": TT[k_].astype(np.float32),
                                    "h_after": H1[k_].astype(np.float32), "delta_norm": float(dln[i]), "m2_norm": float(m2n[i]),
                                    "m3_norm": float(m3n[i]), "dq2": ust[k2]["dq_applied_l2"], "dq3": ust[k3]["dq_applied_l2"],
                                    "version": version})
        return {"B": B, "vp": vp, "d0": d0, "d1": d1, "c0": c0, "c1": c1, "cdd": cdd, "dhn": dhn, "dln": dln, "Tn": Tn, "hn0": hn0,
                "ust": ust, "act": act, "samples": samples, "k2": k2, "k3": k3, "route_stats": route_stats, "route_samples": route_samples}

    def _commit(self, st: dict[str, Any], o: dict[str, Any]) -> None:
        a = st["acc"]
        tol = float(self.s.cfg["learning"]["change_tol"])
        vp = o["vp"]
        a["n_batches"] += 1
        a["n_samples"] += o["B"]
        a["n_valid"] += int(vp.sum())
        a["n_skipped_invalid_proto"] += int((~vp).sum())
        d0, d1 = o["d0"][vp], o["d1"][vp]
        a["sum_d0"] += float(d0.sum())
        a["sum_d1"] += float(d1.sum())
        c0, c1 = o["c0"][vp], o["c1"][vp]
        a["sum_c0"] += float(np.nansum(c0))
        a["sum_c1"] += float(np.nansum(c1))
        a["n_improved"] += int((d1 < d0 - tol).sum())
        a["n_worsened"] += int((d1 > d0 + tol).sum())
        a["n_unchanged"] += int((np.abs(d1 - d0) <= tol).sum())
        cdd = o["cdd"][vp]
        fin = np.isfinite(cdd)
        a["sum_cos_dh_delta"] += float(cdd[fin].sum())
        a["n_cos_dh_delta"] += int(fin.sum())
        a["n_dh_zero"] += int((o["dhn"][vp] == 0).sum())
        mag = o["Tn"][vp] - o["hn0"][vp]
        a["sum_mag_err"] += float(mag.sum())
        a["sum_abs_mag_err"] += float(np.abs(mag).sum())
        ang = np.degrees(np.arccos(np.clip(c0, -1, 1)))
        a["sum_ang_err"] += float(np.nansum(ang))
        a["n_ang"] += int(np.isfinite(ang).sum())
        a["sum_delta_norm"] += float(o["dln"][vp].sum())
        for lay, key in (("L2", o["k2"]), ("L3", o["k3"])):
            u = o["ust"][key]
            a[f"{lay}_sum_dq_raw_l2"] += u["dq_raw_l2"]
            a[f"{lay}_sum_dq_applied_l2"] += u["dq_applied_l2"]
            a[f"{lay}_sum_clip_low"] += u["n_clip_low"]
            a[f"{lay}_sum_clip_high"] += u["n_clip_high"]
            a[f"{lay}_sum_n_changed"] += u["n_changed"]
            a[f"{lay}_last_at_min"] = u["n_at_min"]
            a[f"{lay}_last_at_max"] = u["n_at_max"]
            ac = o["act"][lay]
            a[f"{lay}_sum_active_frac"] += ac["active_frac"]
            a[f"{lay}_sum_s_mean"] += ac["s_mean"]
            a[f"{lay}_sum_conf_mean"] += ac["conf_mean"]
            a[f"{lay}_sum_g_min_frac"] += ac["g_min_frac"]
            a[f"{lay}_sum_g_max_frac"] += ac["g_max_frac"]
        if o["samples"]:
            self.s.corr_buf.extend(o["samples"])
            st["corr_saved"] += len(o["samples"])
            st["corr_epoch_saved"] += len(o["samples"])
        if o.get("route_stats"):
            acc_add(st["routing_acc"], o["route_stats"])
        if o.get("route_samples"):
            self.s.route_buf.extend(o["route_samples"])
            st["route_saved"] += len(o["route_samples"])
            st["route_epoch_saved"] += len(o["route_samples"])
        st["next_batch"] += 1

    # ---------------------------------------------------------------- 에폭 끝
    def _epoch_end(self, area: str, st: dict[str, Any], stop: StopFlag) -> dict[str, Any]:
        s, be = self.s, self.s.be
        bank = s.protos[area]
        rho = float(s.cfg["learning"]["rho"])
        e = int(st["epochs_completed"])
        store_h = s.can_store_h(area)
        P_old = be.clone(bank.P)
        ver_old = bank.version
        tp = s.pass_split(area, "train", bank=bank, collect_h=store_h)        # q 고정, train 만
        counts = tp["counts"]
        hbar = tp["sums"] * be.f32((1.0 / np.maximum(counts, 1))[:, None])
        has = be.f32((counts > 0).astype(np.float32)[:, None])
        P_new = P_old + has * (rho * (hbar - P_old))                         # = (1−ρ)P + ρ·h̄ (표본 있는 클래스만)
        dist_end_old = (tp["dist_sum"] / tp["n_dist"]) if tp["n_dist"] else None
        train_acc_old = (tp["correct"] / tp["n"]) if tp["n"] else None
        with stop.critical():
            rec = bank.set(P_new, counts, source="epoch_ema_update", split="train", sample_ids_sha256=tp["ids_sha"], epoch=e, rho=rho)
            rec["train_zero_vector_fraction"] = 1.0 - tp["n_valid_h"] / max(1, tp["n"])
            s.record_prototypes(area, rec)
        if store_h:
            dist_end_new = s.mean_dist(be.to_np(bank.P), bank.valid, tp["H"], tp["Y"])
        else:
            t2 = s.pass_split(area, "train", bank=bank)
            dist_end_new = (t2["dist_sum"] / t2["n_dist"]) if t2["n_dist"] else None
        dv = s.evaluate(area, "dev", bank, purpose=f"{area}_epoch{e}_dev")
        a = st["acc"]
        nv = max(1.0, a["n_valid"])
        nb = max(1.0, a["n_batches"])
        q_effect = (dist_end_old - st["dist_start_vs_P"]) if (dist_end_old is not None and st["dist_start_vs_P"] is not None) else None
        proto_effect = (dist_end_new - dist_end_old) if (dist_end_new is not None and dist_end_old is not None) else None
        qn = s.model.store.q_numpy()
        row: dict[str, Any] = {
            "utc": utc_now(), "area": area, "epoch": e, "epoch_number": e + 1, "model_version": s.model_version(area),
            "prototype_version_used_in_epoch": ver_old, "prototype_version_after": bank.version,
            "n_train_batches": int(a["n_batches"]), "n_train_samples": int(a["n_samples"]), "n_skipped_invalid_prototype": int(a["n_skipped_invalid_proto"]),
            "mean_dist_before": a["sum_d0"] / nv, "mean_dist_after": a["sum_d1"] / nv,
            "mean_cos_before": a["sum_c0"] / nv, "mean_cos_after": a["sum_c1"] / nv,
            "frac_improved": a["n_improved"] / nv, "frac_worsened": a["n_worsened"] / nv, "frac_unchanged": a["n_unchanged"] / nv,
            "mean_cos_dh_vs_delta": (a["sum_cos_dh_delta"] / a["n_cos_dh_delta"]) if a["n_cos_dh_delta"] else None,
            "n_dh_zero": int(a["n_dh_zero"]), "mean_magnitude_error": a["sum_mag_err"] / nv, "mean_abs_magnitude_error": a["sum_abs_mag_err"] / nv,
            "mean_direction_error_deg": (a["sum_ang_err"] / a["n_ang"]) if a["n_ang"] else None, "mean_delta_norm": a["sum_delta_norm"] / nv,
            "train_acc_vs_epoch_prototype": train_acc_old, "train_correct_vs_epoch_prototype": tp["correct"], "train_n": tp["n"],
            "train_dist_start_vs_P_old": st["dist_start_vs_P"], "train_dist_end_vs_P_old": dist_end_old, "train_dist_end_vs_P_new": dist_end_new,
            "q_effect_dist": q_effect, "prototype_effect_dist": proto_effect, "prototype_update_l2": rec["update_l2"],
            "dev_n": dv["n"], "dev_correct": dv["correct"], "dev_accuracy": dv["accuracy_overall"], "dev_accuracy_valid_only": dv["accuracy_valid_only"],
            "dev_n_unknown_zero_vector": dv["n_unknown_zero_vector"], "dev_n_ties": dv["n_ties"], "dev_mean_margin": dv["mean_margin"],
            "dev_within_class_cos": dv["geometry"].get("within_class_mean_cos_to_class_mean"),
            "dev_between_class_cos": dv["geometry"].get("between_class_mean_cos"),
            "dev_between_class_distance": dv["geometry"].get("between_class_mean_distance"),
            "dev_zero_vector_fraction": dv["geometry"].get("zero_vector_fraction"), "dev_output_variance": dv["geometry"].get("output_variance_mean"),
            "prototype_min_pair_angle_deg": None}
        pc = bank.pairwise_cos()
        C = pc.shape[0]
        off = pc[~np.eye(C, dtype=bool)] if C > 1 else np.zeros(0)
        if off.size and np.isfinite(off).any():
            row["prototype_min_pair_angle_deg"] = float(np.degrees(np.arccos(np.clip(np.nanmax(off), -1, 1))))
        for lay in ("L2", "L3"):
            sp = s.model.specs[f"{area}/{lay}"]
            q = qn[sp.start:sp.end]
            row.update({f"{lay}_dq_raw_l2_mean": a[f"{lay}_sum_dq_raw_l2"] / nb, f"{lay}_dq_applied_l2_mean": a[f"{lay}_sum_dq_applied_l2"] / nb,
                        f"{lay}_n_clip_low_total": int(a[f"{lay}_sum_clip_low"]), f"{lay}_n_clip_high_total": int(a[f"{lay}_sum_clip_high"]),
                        f"{lay}_q_mean": float(q.mean()), f"{lay}_q_min": float(q.min()), f"{lay}_q_max": float(q.max()),
                        f"{lay}_n_q_at_min": int((q <= float(s.cfg['neuron']['q_min'])).sum()),
                        f"{lay}_n_q_at_max": int((q >= float(s.cfg['neuron']['q_max'])).sum()),
                        f"{lay}_active_frac_mean": a[f"{lay}_sum_active_frac"] / nb, f"{lay}_s_mean": a[f"{lay}_sum_s_mean"] / nb,
                        f"{lay}_conf_mean": a[f"{lay}_sum_conf_mean"] / nb, f"{lay}_g_at_min_frac": a[f"{lay}_sum_g_min_frac"] / nb,
                        f"{lay}_g_at_max_frac": a[f"{lay}_sum_g_max_frac"] / nb})
        row.update(s.timer.take())
        row.update(dev_per_class_accuracy=dv["per_class_accuracy"], dev_confusion=dv["confusion"],
                   dev_output_norm_mean=dv["geometry"].get("output_norm_mean"), routing_mode=s.cfg["routing"]["mode"],
                   cache_mb=round(s.loader.cache_bytes / 2 ** 20, 3), max_gather_elements_now=be.max_gather)
        if be.is_cuda:
            row["gpu_max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated(be.dev) / 2 ** 20, 1)
            torch.cuda.reset_peak_memory_stats(be.dev)
        rr = s.model.routing_runtime()
        if rr is not None and AREA_INDEX[area] >= AREA_INDEX["V4"]:
            tr_r = routing_finalize(st["routing_acc"], rr.K)
            s.write_routing_row(area, e, "train_epoch", tr_r)
            if dv.get("routing"):
                s.write_routing_row(area, e, "dev_eval", dv["routing"])
            row.update({f"routing_train_{k}": v for k, v in tr_r.items() if not isinstance(v, list)})
            if dv.get("routing"):
                row.update({f"routing_dev_{k}": v for k, v in dv["routing"].items() if not isinstance(v, list)})
        st["routing_acc"], st["route_epoch_saved"] = {}, 0
        dev_acc = dv["accuracy_overall"]
        improved = st["best_dev_accuracy"] is None or (dev_acc is not None and dev_acc > st["best_dev_accuracy"] + st["min_delta"])
        if improved:
            st["best_dev_accuracy"], st["best_epoch"], st["bad_epochs"] = dev_acc, e, 0
        else:
            st["bad_epochs"] += 1
        st["epochs_completed"] = e + 1
        succ = st["success_dev_accuracy"]
        if succ is not None and dev_acc is not None and dev_acc >= float(succ):
            st["status"] = "success_criterion_met"
        elif st["bad_epochs"] >= st["patience"]:
            st["status"] = "plateau"
        elif st["epochs_completed"] >= st["max_epochs"]:
            st["status"] = "budget_exhausted"
        else:
            st["status"] = "running"
        row["stage_status"] = st["status"]
        row["best_dev_accuracy_so_far"] = st["best_dev_accuracy"]
        row["patience_counter"] = st["bad_epochs"]
        st["history"].append(row)
        st["phase"], st["next_batch"], st["acc"] = "training", 0, empty_acc()
        st["dist_start_vs_P"] = dist_end_new
        st["corr_epoch_saved"] = 0
        with stop.critical():
            s.save_checkpoint(f"{area}_e{e}_done", keep=False)
            s.sync_training_csv()
        return row


# ======================================================================
# 14. 진단 (Diagnostics) — 원본 상태를 복원하거나 복제 구조 (override) 에서 수행한다
# ======================================================================
FORBIDDEN_GRAD_PATTERNS = (r"\.backward\s*\(", r"\bautograd\b", r"torch\.func\b", r"\bjacrev\b", r"\bjacfwd\b", r"\bvjp\b", r"\bjvp\b",
                           r"requires_grad\s*=\s*True", r"requires_grad_\(\s*True", r"enable_grad", r"retain_graph", r"\bfunctorch\b")


class Diagnostics:
    def __init__(self, sess: Session):
        self.s = sess

    # ---------------------------------------------------------------- 공통
    def _probe(self, n: int | None = None, split: str = "train") -> tuple[Any, np.ndarray, list[dict[str, Any]]]:
        n = n or int(self.s.cfg["records"]["diagnostic_samples"])
        items = self.s.data.items(split)[:n]
        lgn, y = self.s.loader.batch(items, split)
        return lgn, y, items

    def _snapshot(self) -> dict[str, Any]:
        be = self.s.be
        return {"q": be.clone(self.s.model.store.q), "P": {a: be.clone(b.P) for a, b in self.s.protos.items()},
                "valid": {a: b.valid.copy() for a, b in self.s.protos.items()}, "rng": copy.deepcopy(self.s.rng.bit_generator.state),
                "q_sha": sha256_array(self.s.model.store.q_numpy()), "P_sha": {a: b.sha() for a, b in self.s.protos.items()}}

    def _restore(self, snap: dict[str, Any]) -> bool:
        be = self.s.be
        be.assign_(self.s.model.store.q, snap["q"])
        for a, b in self.s.protos.items():
            b.P = be.clone(snap["P"][a])
            b.valid = snap["valid"][a].copy()
        self.s.rng.bit_generator.state = snap["rng"]
        return sha256_array(self.s.model.store.q_numpy()) == snap["q_sha"] and all(
            b.sha() == snap["P_sha"][a] for a, b in self.s.protos.items())

    @staticmethod
    def _run(name: str, fn: Callable[[], dict[str, Any]], out: dict[str, Any]) -> None:
        try:
            out[name] = fn()
        except Exception as exc:                             # noqa: BLE001  — 검사 실패는 FAILED 로 남긴다 (성공으로 표시하지 않음)
            out[name] = {"status": FAILED, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=6)}

    # ---------------------------------------------------------------- 기본 검사 1~14
    def run_basic(self) -> dict[str, Any]:
        s = self.s
        if s.model is None:
            raise RuntimeError("모델이 없다 (메뉴 3).")
        snap = self._snapshot()
        out: dict[str, Any] = {}
        try:
            for name, fn in (("C01_repeat_forward_and_eval_invariance", self.c01), ("C02_zero_m_no_change_and_q_bounds", self.c02),
                             ("C03_l1_address", self.c03), ("C04_routing_one_to_one_and_convergence", self.c04),
                             ("C05_input_contribution_sum", self.c05), ("C06_frozen_areas_unchanged", self.c06),
                             ("C07_prototype_train_only", self.c07), ("C08_l6_does_not_change_q", self.c08),
                             ("C09_q_change_persists_in_next_forward", self.c09), ("C10_unresponsive_neurons", self.c10),
                             ("C11_h_before_after_one_step", self.c11), ("C12_prototype_and_output_geometry", self.c12),
                             ("C13_role_responses_quick", lambda: self.role_responses(quick=True)),
                             ("C14_no_global_gradient", self.c14)):
                self._run(name, fn, out)
                if not self._restore(snap):
                    out[name + "_RESTORE"] = {"status": FAILED, "error": "진단 뒤 q/prototype 복원 실패"}
        finally:
            restored = self._restore(snap)
        summary = {"utc": utc_now(), "model_version": s.model_version(), "probe_scope_ko": "train 앞쪽 표본 몇 개 (갱신은 복원)",
                   "restored": restored, "n_failed": sum(1 for v in out.values() if isinstance(v, dict) and v.get("status") == FAILED),
                   "checks": out}
        summary["status"] = FAILED if (summary["n_failed"] or not restored) else PASSED
        if s.rec is not None:
            s.rec.json(f"diagnostics/basic_{local_stamp()}_{time.time_ns() % 100000}.json", summary)
            self._append_index("basic", summary)
        return summary

    def _append_index(self, kind: str, summary: dict[str, Any]) -> None:
        r = self.s.rec
        p = r.p("diagnostics.json")
        cur = read_json(p) if p.is_file() else {"runs": []}
        cur["runs"].append({"kind": kind, "utc": summary.get("utc"), "model_version": summary.get("model_version"),
                            "status": summary.get("status"),
                            "checks": {k: (v.get("status") if isinstance(v, dict) else v) for k, v in (summary.get("checks") or {}).items()}})
        r.json("diagnostics.json", cur)

    def c01(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        lgn, _y, _ = self._probe()
        keys = s.model.layer_keys()
        a = s.model.forward(lgn, keep=keys)
        b = s.model.forward(lgn, keep=keys)
        diff = {k: float(np.abs(be.to_np(a[k]["out"]) - be.to_np(b[k]["out"])).max()) for k in keys}
        q0, p0 = sha256_array(s.model.store.q_numpy()), {x: bk.sha() for x, bk in s.protos.items()}
        area = s.model.active
        bank = s.protos[area]
        cosine_classify(be, a[f"{area}/L3"]["out"], bank.P, bank.valid, float(s.cfg["learning"]["eps_norm"]))
        q1, p1 = sha256_array(s.model.store.q_numpy()), {x: bk.sha() for x, bk in s.protos.items()}
        ok = all(v == 0.0 for v in diff.values()) and q0 == q1 and p0 == p1
        return {"status": PASSED if ok else FAILED, "max_abs_diff_by_layer": diff, "eval_q_unchanged": q0 == q1,
                "eval_prototypes_unchanged": p0 == p1, "rule_ko": "같은 상태·입력의 반복 순방향 차이 0, 평가 전후 q·prototype 해시 동일"}

    def c02(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        area = s.model.active
        lgn, _y, _ = self._probe()
        k2, k3 = f"{area}/L2", f"{area}/L3"
        f = s.model.forward(lgn, keep=(k2, k3))
        L = LocalOutputLearner(s.cfg, be, s.model.store)
        dq2 = L.proposal(f[k2]["s"], be.zeros(f[k2]["s"].shape))
        dq3 = L.proposal(f[k3]["s"], be.zeros(f[k3]["s"].shape))
        mx = max(float(np.abs(be.to_np(dq2)).max()), float(np.abs(be.to_np(dq3)).max()))
        before = s.model.store.q_numpy()
        L.apply({k2: dq2, k3: dq3})
        after = s.model.store.q_numpy()
        ne = s.cfg["neuron"]
        l4 = np.concatenate([after[s.model.specs[f"{a}/L4"].start:s.model.specs[f"{a}/L4"].end] for a in s.model.areas])
        bounds_ok = bool(np.isfinite(after).all() and after.min() >= float(ne["q_min"]) and after.max() <= float(ne["q_max"]))
        ok = mx == 0.0 and np.array_equal(before, after) and bounds_ok and bool((l4 == float(ne["q_init"])).all())
        return {"status": PASSED if ok else FAILED, "max_abs_dq_for_m0": mx, "q_identical_after_apply": bool(np.array_equal(before, after)),
                "q_finite_and_in_bounds": bounds_ok, "L4_q_fixed_at_init": bool((l4 == float(ne["q_init"])).all()),
                "q_range": [float(after.min()), float(after.max())]}

    def c03(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        area = s.model.active
        relay = L1Relay(area, s.model.store)
        n3 = s.model.specs[f"{area}/L3"].n
        B = 3
        delta = be.f32(self.s.rng.normal(0, 0.1, size=(B, n3)))
        m2, m3 = s.model.routers[area].route(delta)
        ok_deliver = True
        try:
            relay.deliver(relay.packet("L2", m2, "x", "v"), B, be)
            relay.deliver(relay.packet("L3", m3, "x", "v"), B, be)
        except L1AddressError:
            ok_deliver = False
        bad = {}
        tamper = {"wrong_layer": lambda p: dataclasses.replace(p, layer="L4"),
                  "wrong_area": lambda p: dataclasses.replace(p, area="ZZ"),
                  "wrong_ids": lambda p: dataclasses.replace(p, ids_sha256="0" * 64),
                  "wrong_range": lambda p: dataclasses.replace(p, start=p.start + 1),
                  "wrong_shape": lambda p: dataclasses.replace(p, m=p.m[:, :-1])}
        for nm, fn in tamper.items():
            try:
                relay.deliver(fn(relay.packet("L3", m3, "x", "v")), B, be)
                bad[nm] = "accepted (FAIL)"
            except L1AddressError:
                bad[nm] = "rejected"
        before = s.model.store.q_numpy()
        lgn, _y, _ = self._probe(B)
        k2, k3 = f"{area}/L2", f"{area}/L3"
        f = s.model.forward(lgn, keep=(k2, k3))
        L = LocalOutputLearner(s.cfg, be, s.model.store)
        L.apply({k2: L.proposal(f[k2]["s"], m2), k3: L.proposal(f[k3]["s"], m3)})
        after = s.model.store.q_numpy()
        changed = np.nonzero(after != before)[0]
        lo, hi = s.model.specs[k2].start, s.model.specs[k3].end
        inside = bool(((changed >= lo) & (changed < hi)).all())
        ok = ok_deliver and all(v == "rejected" for v in bad.values()) and inside
        return {"status": PASSED if ok else FAILED, "valid_packets_delivered": ok_deliver, "tampered_packets": bad,
                "n_q_changed": int(changed.size), "changes_only_inside_active_L2_L3": inside, "active_id_range": [int(lo), int(hi)]}

    def c04(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        rows = {}
        ok = True
        for area in s.model.areas:
            rt = s.model.routers[area]
            n3 = s.model.specs[f"{area}/L3"].n
            colsum_err = float(np.abs(rt.R_col_sum - 1.0).max())
            d = self.s.rng.normal(0, 1, size=(2, n3)).astype(np.float32)
            m2, _m3 = rt.route(be.f32(d))
            m2 = be.to_np(m2).astype(np.float64)
            ref = np.zeros_like(m2)
            for j_row, i_src in zip(rt.parent_rows, rt.parent_src):
                ref[:, i_src] += rt.alpha2 * d[:, j_row] / rt.parent_count[j_row]
            ref_err = float(np.abs(ref - m2).max())
            tot_err = float(np.abs(m2.sum(1) - rt.alpha2 * d.astype(np.float64).sum(1)).max())
            r = {"one_to_one": rt.one_to_one, "R_column_sum_max_abs_error": colsum_err, "max_abs_vs_explicit_parent_loop": ref_err,
                 "total_mass_error": tot_err, "parents_per_L3": rt.describe()["parents_per_L3"],
                 "L2_without_child": rt.n2_without_child}
            if rt.one_to_one:
                exact = np.zeros_like(m2)
                exact[:, rt.parent_src] = rt.alpha2 * d[:, rt.parent_rows]
                r["one_to_one_exact"] = bool(np.array_equal(exact.astype(np.float32), m2.astype(np.float32)))
                ok &= r["one_to_one_exact"]
            ok &= colsum_err < 1e-6 and ref_err < 1e-5 and (tot_err < 1e-3 * (1 + n3) or rt.n2_without_child > 0)
            rows[area] = r
        return {"status": PASSED if ok else FAILED, "areas": rows,
                "rule_ko": "R_ij = 1/|P(j)| 열 합 1, 명시적 부모 반복과 일치, 일대일 구간은 m_L2 = α2·δ (대응 위치) 정확히"}

    def c05(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        lgn, _y, _ = self._probe(2)
        keys = s.model.layer_keys()
        res = s.model.forward(lgn, keep=["LGN"] + keys)
        worst, rows = 0.0, []
        for key in keys:
            sp = s.model.specs[key]
            src = be.to_np(res[sp.src_key]["out"]).astype(np.float64)
            u = be.to_np(res[key]["u"]).astype(np.float64)
            pick = self.s.rng.choice(sp.n, size=min(6, sp.n), replace=False)
            routed = sp.routed if sp.routed is not None else np.zeros(sp.n, bool)
            if routed.any():                                       # 라우팅 행도 표본에 섞어 넣는다
                pick = np.unique(np.concatenate([pick, self.s.rng.choice(np.nonzero(routed)[0], size=min(6, int(routed.sum())), replace=False)]))
            for j in pick:
                r = sp.row(int(j))
                ua = (src[:, r["a_src"]] * r["a_w"]).sum(1)
                ub = (src[:, r["b_src"]] * r["b_w"]).sum(1) if r["b_src"].size else np.zeros(src.shape[0])
                if routed[j]:                                      # 이번 표본에서 실제로 고른 B 주소 (정적 후보가 아니라 선택 기록)
                    t = sp.routing
                    ci = int(np.nonzero(t.curve_rows == j)[0][0])
                    sel = be.to_np(res[key]["routing"]["sel_src"])[:, ci]
                    srcp = np.concatenate([src, np.zeros((src.shape[0], 1))], 1)
                    ub = srcp[np.arange(src.shape[0]), sel]
                ue = np.minimum(ua, ub) if sp.op_min[j] else ua + ub
                if sp.missing is not None and sp.missing[j]:
                    ue = np.zeros_like(ue)
                err = float(np.abs(ue - u[:, j]).max() / (1.0 + np.abs(ue).max()))
                worst = max(worst, err)
                rows.append({"layer": key, "local": int(j), "op": "min2" if sp.op_min[j] else "linear", "rel_err": err,
                             "n_input": int(r["a_src"].size + r["b_src"].size), "routed": bool(routed[j]),
                             "missing_candidate": bool(sp.missing is not None and sp.missing[j])})
        dups = {k: s.model.specs[k].build_stats.get("merged_duplicate_edges", 0) for k in keys}
        return {"status": PASSED if worst < 1e-5 else FAILED, "worst_rel_error": worst, "samples": rows,
                "duplicate_policy": "merge_sum (같은 수신·입력군·송신은 가중치 합으로 하나), 구조 검증에서 남은 중복 0 확인",
                "merged_duplicates_at_build": dups}

    def c06(self) -> dict[str, Any]:
        if not self.s.frozen:
            return {"status": NOT_APPLICABLE, "reason_ko": "동결된 앞 영역이 아직 없다 (V1 단계)."}
        chk = self.s.frozen_check()
        ok = all(all(v.values()) for v in chk.values())
        return {"status": PASSED if ok else FAILED, "areas": chk}

    def c07(self) -> dict[str, Any]:
        s = self.s
        train_sha = s.data.ids_sha("train")
        n_train = len(s.data.items("train"))
        rows = {}
        ok = True
        for area, bank in s.protos.items():
            hs = bank.history
            splits = sorted({h["split"] for h in hs})
            same_ids = all(h["sample_ids_sha256"] == train_sha for h in hs)
            counts_ok = all(sum(h["counts"]) == n_train for h in hs)
            rows[area] = {"n_updates": len(hs), "splits_used": splits, "sample_ids_equal_train": same_ids, "counts_sum_equal_n_train": counts_ok}
            ok &= splits == ["train"] and same_ids and counts_ok
        return {"status": PASSED if ok else FAILED, "areas": rows, "loader_access": dict(s.loader.access)}

    def c08(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        lgn, _y, _ = self._probe()
        keys = [k for k in s.model.layer_keys() if s.model.rt[k].l6]
        q0 = s.model.store.q_numpy()
        res = s.model.forward(lgn, keep=keys)
        q1 = s.model.store.q_numpy()
        gstats, ok_g = {}, True
        for k in keys:
            g = be.to_np(res[k]["g"])
            conf, out = be.to_np(res[k]["conf"]), be.to_np(res[k]["out"])
            ok_g &= bool(np.allclose(out, g * conf, atol=1e-7)) and g.min() >= s.model.l6.g_min - 1e-7 and g.max() <= s.model.l6.g_max + 1e-7
            gstats[k] = {"g_min": float(g.min()), "g_max": float(g.max()), "g_mean": float(g.mean()),
                         "frac_at_g_min": float((g <= s.model.l6.g_min + 1e-7).mean()), "frac_at_g_max": float((g >= s.model.l6.g_max - 1e-7).mean()),
                         "neurons_without_neighbor": s.model.rt[k].n_no_neighbor}
        ok = np.array_equal(q0, q1) and ok_g
        return {"status": PASSED if ok else FAILED, "q_unchanged_by_l6": bool(np.array_equal(q0, q1)), "out_equals_g_times_conf": ok_g,
                "layers": gstats}

    def c09(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        area = s.model.active
        k3 = f"{area}/L3"
        lgn, _y, _ = self._probe(1)
        f0 = s.model.forward(lgn, keep=(k3,))
        sv = be.to_np(f0[k3]["s"])[0]
        q = be.to_np(s.model.store.q_view(k3)).copy()
        qmax = float(s.cfg["neuron"]["q_max"])
        cand = np.nonzero((sv > 0) & (q < qmax - 0.1))[0]
        if cand.size == 0:
            return {"status": MEASURED, "reason_ko": "탐침 표본에서 s>0 이고 q 를 올릴 수 있는 L3 뉴런이 없다 (s=0 또는 q 포화).",
                    "n_active_L3": int((sv > 0).sum())}
        j = int(cand[0])
        q2 = q.copy()
        q2[j] = min(qmax, q[j] + 0.5)
        be.assign_(s.model.store.q_view(k3), be.f32(q2))
        o1 = be.to_np(s.model.forward(lgn)[k3]["out"])[0]
        o2 = be.to_np(s.model.forward(lgn)[k3]["out"])[0]
        o0 = be.to_np(f0[k3]["out"])[0]
        changed = bool(o1[j] != o0[j])
        persist = bool(np.array_equal(o1, o2))
        return {"status": PASSED if (changed and persist) else FAILED, "neuron_local": j, "q_before": float(q[j]), "q_after": float(q2[j]),
                "out_before": float(o0[j]), "out_after": float(o1[j]), "effect_persists_next_forward": persist,
                "n_other_neurons_changed_via_l6": int((o1 != o0).sum() - (1 if changed else 0))}

    def c10(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        area = s.model.active
        lgn, _y, _ = self._probe()
        keys = (f"{area}/L2", f"{area}/L3")
        res = s.model.forward(lgn, keep=keys)
        ne = s.cfg["neuron"]
        rows = {}
        for k in keys:
            sv = be.to_np(res[k]["s"])
            q = be.to_np(s.model.store.q_view(k))
            g = be.to_np(res[k]["g"]) if res[k]["g"] is not None else np.ones_like(sv)
            never = (sv <= 0).all(axis=0)
            qz = q <= float(ne["q_min"])
            rows[k] = {"n": int(sv.shape[1]), "s_zero_all_probe": int(never.sum()), "q_at_min": int(qz.sum()),
                       "q_at_min_but_s_positive_recoverable_if_delta_positive": int((qz & ~never).sum()),
                       "q_at_max": int((q >= float(ne["q_max"])).sum()),
                       "g_at_bound_all_probe": int(((g <= s.model.l6.g_min + 1e-7) | (g >= s.model.l6.g_max - 1e-7)).all(axis=0).sum()),
                       "cause_ko": "s=0 이면 e=0 이라 이 입력들로는 q 가 바뀌지 않는다; q=q_min 이면 출력 0 (s>0 이면 δ>0 일 때 회복 가능, 보장 아님)"}
        return {"status": MEASURED, "probe_samples": int(lgn.shape[0]), "layers": rows}

    def c11(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        area = s.model.active
        bank = s.protos[area]
        lgn, y, _ = self._probe()
        k2, k3 = f"{area}/L2", f"{area}/L3"
        f0 = s.model.forward(lgn, keep=(k2, k3))
        h0 = f0[k3]["out"]
        T = bank.targets(y)
        vp = bank.valid[y]
        delta = (T - h0) * be.f32(vp.astype(np.float32)[:, None])
        m2, m3 = s.model.routers[area].route(delta)
        L = LocalOutputLearner(s.cfg, be, s.model.store)
        ust = L.apply({k2: L.proposal(f0[k2]["s"], m2), k3: L.proposal(f0[k3]["s"], m3)})
        h1 = s.model.forward(lgn)[k3]["out"]
        H0, H1, TT, D = (be.to_np(x).astype(np.float64) for x in (h0, h1, T, delta))
        d0, d1 = np.linalg.norm(TT - H0, axis=1), np.linalg.norm(TT - H1, axis=1)
        dh = H1 - H0
        with np.errstate(divide="ignore", invalid="ignore"):
            cdd = (dh * D).sum(1) / (np.linalg.norm(dh, axis=1) * np.linalg.norm(D, axis=1))
        tol = float(s.cfg["learning"]["change_tol"])
        return {"status": MEASURED, "dist_before": d0.tolist(), "dist_after": d1.tolist(),
                "improved": int((d1 < d0 - tol).sum()), "worsened": int((d1 > d0 + tol).sum()), "unchanged": int((np.abs(d1 - d0) <= tol).sum()),
                "cos_dh_vs_delta": [None if not np.isfinite(v) else float(v) for v in cdd], "update_stats": ust,
                "note_ko": "한 번의 국소 갱신 뒤 실제 재순방향 결과 (같은 이전 prototype 기준). 개선을 통과 조건으로 쓰지 않는다. 진단 뒤 q 를 되돌린다."}

    def c12(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        area = s.model.active
        bank = s.protos[area]
        items = s.data.items("train")[:64]
        lgn, y = s.loader.batch(items, "train")
        H = be.to_np(s.model.forward(lgn, upto=area)[f"{area}/L3"]["out"])
        pc = bank.pairwise_cos()
        C = pc.shape[0]
        off = pc[~np.eye(C, dtype=bool)] if C > 1 else np.zeros(0)
        g = s.geometry(H, y)
        return {"status": MEASURED, "prototype_pairwise_cos": pc.tolist(),
                "prototype_max_offdiag_cos": float(np.nanmax(off)) if off.size and np.isfinite(off).any() else None,
                "collapse_warning": bool(off.size and np.isfinite(off).any() and np.nanmax(off) > 0.996),
                "prototype_norms": be.to_np(be.norm_last(bank.P)).tolist(), "train_subset_geometry": g, "n_subset": len(items)}

    def c14(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        hits = {}
        for obj in (CortexModel, RelativeActivityController, BucketELL, LayerRuntime, LocalErrorRouter, L1Relay, LocalOutputLearner,
                    PrototypeBank, StageTrainer, Session, Backend, cosine_classify):
            src = inspect.getsource(obj)
            found = [p for p in FORBIDDEN_GRAD_PATTERNS if re.search(p, src)]
            if found:
                hits[getattr(obj, "__name__", str(obj))] = found
        tensors = [s.model.store.q, s.model.store.thr] + [b.P for b in s.protos.values()]
        for rt in s.model.rt.values():
            tensors += [w for _i, w in rt.A.parts] + ([w for _i, w in rt.Bm.parts] if rt.Bm else [])
        rg = be.requires_grad_any(tensors)
        ge = be.grad_enabled()
        ok = not hits and not rg and not ge
        return {"status": PASSED if ok else FAILED, "static_forbidden_patterns": hits, "any_tensor_requires_grad": rg,
                "grad_mode_enabled": ge, "backend": be.kind,
                "note_ko": "정적 검사 (학습 경로 소스) + 실행 검사 (requires_grad, grad mode). backward 가 없다는 이유만으로 역전파가 없다고 판단하지 않도록 "
                           "LocalErrorRouter·LocalOutputLearner 의 수식을 주석과 config.json 에 그대로 남겼다."}

    # ---------------------------------------------------------------- 역할 반응 (통제 자극)
    def role_responses(self, quick: bool = False) -> dict[str, Any]:
        s, be = self.s, self.s.be
        S = s.pre.S
        fac = StimulusFactory(S, supersample=2 if quick else 3)
        model = s.model
        areas = list(model.areas)
        lw = max(1.5, S / 48.0)

        def run(imgs: list[np.ndarray], keep: Sequence[str]) -> dict[str, np.ndarray]:
            X = be.f32(np.stack([s.pre.from_array(im) for im in imgs]))
            r = model.forward(X, keep=keep)
            return {k: be.to_np(v["out"]).astype(np.float64) for k, v in r.items()}

        out: dict[str, Any] = {"utc": utc_now(), "model_version": s.model_version(), "image_size": S, "quick": quick}
        const = run([fac.constant(v) for v in (0.0, 0.25, 0.5, 0.75, 1.0)], [f"{a}/L2" for a in areas])
        out["constant_images"] = {k: {"max_abs_out": float(np.abs(v).max())} for k, v in const.items()}
        oris = [float(t) for t in s.cfg["areas"]["V1"]["orientations_deg"]]
        sp2, sp3 = model.specs["V1/L2"], model.specs["V1/L3"]
        o2 = np.asarray(sp2.channel_meta["orientation_deg"])
        o3 = np.asarray(sp3.channel_meta["orientation_deg"])
        lines = [fac.line(t, width=lw) for t in oris] + [fac.line(t, width=lw, fg=0.0, bg=1.0) for t in oris]
        r = run(lines, ["V1/L2", "V1/L3"])

        def tuning(resp: np.ndarray, sp: LayerSpec, chan_ori: np.ndarray) -> np.ndarray:
            m = resp.reshape(resp.shape[0], sp.n_pos, sp.n_ch).mean(axis=1)
            return np.stack([m[:, np.isclose(chan_ori, o)].mean(axis=1) for o in oris], axis=1)

        res_ori = {}
        for key, sp, co in (("V1/L2", sp2, o2), ("V1/L3", sp3, o3)):
            tb = tuning(r[key], sp, co)
            pref = np.asarray(oris)[np.argmax(tb, axis=1)]
            stim = np.asarray(oris * 2)
            err = np.abs(((pref - stim) + 90.0) % 180.0 - 90.0)
            res_ori[key] = {"preferred_matches_stimulus_fraction": float((err <= 7.5).mean()), "mean_abs_error_deg": float(err.mean()),
                            "selectivity_mean": float(((tb.max(1) - tb.mean(1)) / (tb.max(1) + 1e-12)).mean()),
                            "tuning_matrix_bright_lines": tb[:len(oris)].tolist()}
        out["V1_orientation_lines"] = res_ori
        lam = float(np.median(np.asarray(model.specs["V1/L2"].rf))) * float(s.cfg["areas"]["V1"]["gabor_wavelength_factor"])
        lam = max(lam, 4.0)
        phases = [2 * math.pi * k / 8 for k in range(8)]
        rg = run([fac.grating(0.0, lam, ph) for ph in phases], ["V1/L2", "V1/L3"])

        def mod_depth(resp: np.ndarray, sp: LayerSpec, mask: np.ndarray) -> float:
            m = resp.reshape(resp.shape[0], sp.n_pos, sp.n_ch)[:, :, mask].mean(axis=(1, 2))
            return float((m.max() - m.min()) / (m.max() + m.min() + 1e-12))

        out["V1_phase_modulation"] = {
            "grating_wavelength_px": lam, "orientation_deg": 0.0,
            "L2_modulation_depth_by_phase_channel": {str(p): mod_depth(rg["V1/L2"], sp2, np.isclose(o2, 0.0) &
                                                                         np.isclose(np.asarray(sp2.channel_meta["phase_rad"]), p))
                                                     for p in sorted(set(sp2.channel_meta["phase_rad"]))},
            "L3_modulation_depth": mod_depth(rg["V1/L3"], sp3, np.isclose(o3, 0.0)),
            "note_ko": "L3 는 위상×극성 정류 반응의 평균이라 L2 보다 위상 변조가 작을 수 있으나 완전한 위상 불변성은 아니다."}
        if "V2" in areas:
            sp = model.specs["V2/L2"]
            fam = np.asarray(sp.channel_meta["family"])
            ori = np.asarray(sp.channel_meta["orientation_deg"], dtype=object)
            o_arr = np.asarray([float(v) for v in ori])
            seg_len = 2.0 * float(np.median(model.specs["V2/L4"].pos_rf))
            rv = run([fac.line(0.0, width=lw, length=seg_len), fac.line(0.0, width=lw), fac.cross(0.0, 90.0, width=lw)], ["V2/L2"])
            m = rv["V2/L2"].reshape(3, sp.n_pos, sp.n_ch).mean(axis=1)
            es = (fam == "endstop") & np.isclose(o_arr, 0.0)
            cr = (fam == "cross") & np.asarray([sp.channel_meta["orientation2_deg"][k] is not None and
                                                 abs(float(sp.channel_meta["orientation2_deg"][k]) - 90.0) < 1e-6 and abs(o_arr[k]) < 1e-6
                                                 for k in range(sp.n_ch)])
            ct = (fam == "contour") & np.isclose(o_arr, 0.0)
            out["V2_roles"] = {"segment_length_px": seg_len,
                               "endstop_short_over_long": float(m[0, es].mean() / (m[1, es].mean() + 1e-12)) if es.any() else None,
                               "cross_0_90_cross_over_line": float(m[2, cr].mean() / (m[1, cr].mean() + 1e-12)) if cr.any() else None,
                               "contour_0_long_over_short": float(m[1, ct].mean() / (m[0, ct].mean() + 1e-12)) if ct.any() else None}
        if "V4" in areas:
            sp = model.specs["V4/L2"]
            fam = np.asarray(sp.channel_meta["family"])
            rv = run([fac.arc(S / 4.0, width=lw, start_deg=0.0, span_deg=360.0), fac.line(0.0, width=lw), fac.line(90.0, width=lw)], ["V4/L2"])
            m = rv["V4/L2"].reshape(3, sp.n_pos, sp.n_ch).mean(axis=1)
            cv, st_ = fam == "curve", fam == "straight"
            out["V4_roles"] = {"curve_over_straight_for_circle": float(m[0, cv].mean() / (m[0, st_].mean() + 1e-12)) if cv.any() and st_.any() else None,
                               "curve_over_straight_for_lines": float(m[1:, cv].mean() / (m[1:, st_].mean() + 1e-12)) if cv.any() and st_.any() else None}
        shapes = ["circle", "square", "triangle"]
        sz = S * 0.45
        shift = S / 10.0
        imgs = [fac.shape(k, sz) for k in shapes] + [fac.shape(k, sz, center=(shift, 0.0)) for k in shapes]
        rt_ = run(imgs, [])
        trans = {}
        for a in areas:
            H = rt_[f"{a}/L3"]
            n = np.linalg.norm(H, axis=1)
            Hn = H / np.where(n > 0, n, 1.0)[:, None]
            same = [float(Hn[i] @ Hn[i + 3]) for i in range(3)]
            diff = [float(Hn[i] @ Hn[j]) for i in range(3) for j in range(i + 1, 3)]
            trans[a] = {"cos_same_shape_shifted_mean": float(np.mean(same)), "cos_different_shapes_center_mean": float(np.mean(diff)),
                        "zero_vectors": int((n == 0).sum())}
        out["translation"] = {"shift_px": shift, "areas": trans}
        out["status"] = MEASURED
        out["note_ko"] = "역할 반응 측정값이다. 통과 기준이 아니며 역할 설계의 일반적 우월성을 증명하지 않는다."
        if s.rec is not None and not quick:
            s.rec.json(f"diagnostics/role_responses_{local_stamp()}.json", out)
            self._append_index("role_responses", {"utc": out["utc"], "model_version": out["model_version"], "status": MEASURED})
            save_role_figure(s, out)
        return out

    # ---------------------------------------------------------------- 역할 교란 비교 (같은 저장 모델, 비교 모델 학습 없음)
    def perturbation(self, kind: str = "l2_rows_within_position", fraction: float = 0.5, area: str | None = None) -> dict[str, Any]:
        s, be = self.s, self.s.be
        area = area or s.model.active
        snap = self._snapshot()
        rng = np.random.default_rng([int(s.cfg["seed"]), 31, AREA_INDEX[area], int(1000 * fraction)])
        if kind == "l2_rows_within_position":
            key = f"{area}/L2"
            sp = s.model.specs[key]
            sel = rng.choice(sp.n_pos, size=max(1, int(round(fraction * sp.n_pos))), replace=False)
            order = np.arange(sp.n)
            for p in sel:
                order[p * sp.n_ch:(p + 1) * sp.n_ch] = p * sp.n_ch + rng.permutation(sp.n_ch)
            new = self._reorder_rows(sp, order)
            what = f"{key}: 선택 위치 {len(sel)}/{sp.n_pos} 에서 채널 행 (입력 목록·연산) 을 섞음 — q 는 neuron_id 에 남는다"
        elif kind == "l4_spatial":
            key = f"{area}/L4"
            sp = s.model.specs[key]
            src_sp = s.model.specs.get(sp.src_key)
            nps, nch = (src_sp.n_pos, src_sp.n_ch) if src_sp is not None else (s.pre.n_pos, s.pre.n_maps)
            pick = rng.choice(nps, size=max(2, int(round(fraction * nps))), replace=False)
            pm = np.arange(nps)
            pm[pick] = rng.permutation(pick)
            remap = (pm[np.arange(sp.n_src) // nch] * nch + np.arange(sp.n_src) % nch).astype(np.int64)
            new = dataclasses.replace(sp, a_src=remap[sp.a_src], b_src=remap[sp.b_src])
            what = f"{key}: 원천 위치 {len(pick)}/{nps} 를 서로 바꿈 (공간 배치 교란)"
        else:
            raise ValueError(f"알 수 없는 교란 종류: {kind}")
        h_before = s.model.struct_hash[area]
        override = {key: LayerRuntime(new, be, s.cfg)}
        bank = s.protos[area]

        def remean(ov: dict[str, LayerRuntime] | None) -> PrototypeBank:
            tp = s.pass_split(area, "train", override=ov)
            b = PrototypeBank(area, s.classes, bank.dim, be, bank.eps)
            b.P = tp["sums"] * be.f32((1.0 / np.maximum(tp["counts"], 1))[:, None])
            b.valid = be.to_np(be.norm_last(b.P)) > bank.eps
            return b

        try:
            res = {"original_with_current_prototypes": s.evaluate(area, "dev", bank, purpose="perturbation_original"),
                   "original_with_remeaned_prototypes": s.evaluate(area, "dev", remean(None), purpose="perturbation_original_remean"),
                   "perturbed_with_current_prototypes": s.evaluate(area, "dev", bank, purpose="perturbation_perturbed", override=override),
                   "perturbed_with_remeaned_prototypes": s.evaluate(area, "dev", remean(override), purpose="perturbation_perturbed_remean",
                                                                    override=override)}
        finally:
            restored = self._restore(snap)
        out = {"utc": utc_now(), "kind": kind, "fraction": fraction, "area": area, "what_ko": what,
               "model_version": s.model_version(area), "structure_unchanged_after": s.model.struct_hash[area] == h_before and restored,
               "dev_accuracy": {k: {"overall": v["accuracy_overall"], "correct": v["correct"], "n": v["n"],
                                    "valid_only": v["accuracy_valid_only"]} for k, v in res.items()},
               "status": MEASURED,
               "note_ko": "같은 저장 모델의 고정 평가 비교다 (비교 모델을 학습하지 않음). 이 모델의 역할 의존성을 보여 줄 뿐 역할 설계의 일반적 우월성을 증명하지 않는다. "
                          "remeaned = q 를 바꾸지 않고 train 출력 평균으로 다시 만든 임시 prototype (저장하지 않음)."}
        if s.rec is not None:
            s.rec.json(f"diagnostics/perturbation_{kind}_{local_stamp()}.json", out)
            self._append_index("perturbation", {"utc": out["utc"], "model_version": out["model_version"], "status": MEASURED})
        return out

    @staticmethod
    def _reorder_rows(sp: LayerSpec, order: np.ndarray) -> LayerSpec:
        def csr(ptr: np.ndarray, src: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            lens = np.diff(ptr)[order]
            nptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
            ns = np.concatenate([src[ptr[i]:ptr[i + 1]] for i in order]) if len(src) else src.copy()
            nw = np.concatenate([w[ptr[i]:ptr[i + 1]] for i in order]) if len(w) else w.copy()
            return nptr, ns, nw
        a = csr(sp.a_ptr, sp.a_src, sp.a_w)
        b = csr(sp.b_ptr, sp.b_src, sp.b_w)
        return dataclasses.replace(sp, a_ptr=a[0], a_src=a[1], a_w=a[2], b_ptr=b[0], b_src=b[1], b_w=b[2], op_min=sp.op_min[order])

    # ---------------------------------------------------------------- 최종 test 평가
    def test_evaluation(self) -> dict[str, Any]:
        s = self.s
        area = s.model.active
        prior = int(s.counters["test_evaluations"])
        dc_path = s.rec.p("design_changes.jsonl")
        changes = [json.loads(x) for x in dc_path.read_text(encoding="utf-8").splitlines() if x.strip()] if dc_path.is_file() else []
        info = read_json(s.rec.p("run_info.json")).get("branched_from") if s.rec.p("run_info.json").is_file() else None
        src_tests = int(((info or {}).get("source_counters") or {}).get("test_evaluations", 0))
        m = s.evaluate(area, "test", s.protos[area], purpose="final_test_fixed_model")
        indep = {"prior_test_evaluations_this_run": prior, "design_changes_logged": len(changes),
                 "design_changes_after_test_seen": sum(1 for c in changes if c.get("after_test_seen")),
                 "branched_from_run": (info or {}).get("source_run"), "source_run_test_evaluations_before_branch": src_tests,
                 "routing_mode": s.cfg["routing"]["mode"]}
        indep["independent_first_look"] = prior == 0 and src_tests == 0 and indep["design_changes_after_test_seen"] == 0
        out = {"utc": utc_now(), "area": area, "model_version": m["model_version"], "checkpoint": s.last_checkpoint,
               "test_evaluation_number": s.counters["test_evaluations"], "metrics": m, "independence": indep,
               "note_ko": "test 는 고정 모델 평가용이다. 이 결과로 설정·에폭·모델·라우팅 모드를 고르지 않는다. 반복 평가, test 를 본 뒤의 설계 변경, "
                          "test 를 본 실행에서의 분기는 독립 시험이 아니며 independence 에 기록된다."}
        s.rec.json(f"diagnostics/test_eval_{s.counters['test_evaluations']:03d}.json", out)
        self._append_index("test_eval", {"utc": out["utc"], "model_version": out["model_version"], "status": MEASURED})
        return out

    def state_summary(self) -> dict[str, Any]:
        s = self.s
        q = s.model.store.q_numpy()
        ne = s.cfg["neuron"]
        out = {"model_version": s.model_version(), "areas": {}}
        for a in s.model.areas:
            row = {"stage_status": s.stage[a]["status"], "epochs_completed": s.stage[a]["epochs_completed"],
                   "best_dev_accuracy": s.stage[a]["best_dev_accuracy"], "frozen": a in s.frozen,
                   "prototype_version": s.protos[a].version, "prototype_valid": s.protos[a].valid.tolist()}
            for lay in LAYER_ORDER:
                sp = s.model.specs[f"{a}/{lay}"]
                qq = q[sp.start:sp.end]
                row[lay] = {"q_mean": float(qq.mean()), "q_min": float(qq.min()), "q_max": float(qq.max()),
                            "n_at_q_min": int((qq <= float(ne["q_min"])).sum()), "n_at_q_max": int((qq >= float(ne["q_max"])).sum())}
            out["areas"][a] = row
        return out


# ======================================================================
# 15. 추론 (저장 모델, 갱신 없음)
# ======================================================================
def list_images(path: Path, exts: Sequence[str]) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        e = {x.lower() for x in exts}
        return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in e])
    raise FileNotFoundError(f"이미지 파일·폴더가 없다: {path}")


def run_inference(sess: Session, path: Path, *, area: str | None = None, figure: bool = True) -> dict[str, Any]:
    s, be = sess, sess.be
    area = area or s.model.active
    bank = s.protos[area]
    files = list_images(path, s.cfg["data"]["extensions"])
    if not files:
        raise FileNotFoundError(f"지원 이미지가 없다: {path}")
    eps = float(s.cfg["learning"]["eps_norm"])
    rows = []
    out_dir = s.rec.p(f"inference/{local_stamp()}_{time.time_ns() % 100000}")
    out_dir.mkdir(parents=True, exist_ok=True)
    first = None
    for p in files:
        try:
            v, meta = s.pre.from_path(p)
        except Exception as exc:                             # noqa: BLE001
            rows.append({"image": str(p), "status": "load_failed", "error": f"{type(exc).__name__}: {exc}"})
            continue
        X = be.f32(v[None, :])
        rr = s.model.routing_runtime()
        use_route = rr is not None and AREA_INDEX[area] >= AREA_INDEX["V4"]
        res = s.model.forward(X, upto=area, keep=(f"{area}/L2", f"{area}/L3"), routing_record=use_route)    # 라벨 없이 같은 V1 안내 경로
        h = res[f"{area}/L3"]["out"]
        cl = cosine_classify(be, h, bank.P, bank.valid, eps)
        pred = int(cl["pred"][0])
        row = {"image": str(p), "status": "ok" if pred >= 0 else "unknown_zero_vector",
               "prediction": s.classes[pred] if pred >= 0 else "unknown", "output_norm": float(cl["h_norm"][0]),
               "valid_output": bool(cl["valid_h"][0]), "area": area, "model_version": s.model_version(area),
               "checkpoint": s.last_checkpoint, "orig_size": [meta["orig_w"], meta["orig_h"]]}
        for c, name in enumerate(s.classes):
            cv = cl["cos"][0, c]
            row[f"cos_{name}"] = float(cv) if np.isfinite(cv) else None
        row["routing_mode"] = s.cfg["routing"]["mode"]
        if use_route and "routing" in res.get("V4/L2", {}):
            rs = routing_finalize(routing_batch_stats(rr.table, routing_rec_np(be, res["V4/L2"]["routing"]),
                                                      float(s.cfg["routing"]["high_guide_score"])), rr.K)
            for k_ in ("no_guidance_frac", "differs_from_geometry_first_frac", "mean_S_selected", "zero_B_frac", "missing_candidate_frac"):
                row[f"routing_{k_}"] = rs.get(k_)
        rows.append(row)
        if first is None:
            first = (p, {k: be.to_np(v_["out"])[0] for k, v_ in res.items() if k.startswith(area)})
    s.counters["inference_images"] += len(files)
    for r_ in rows:
        s.rec.append_csv(f"inference/{out_dir.name}/inference_results.csv", r_)
    write_json(out_dir / "inference_results.json", {"utc": utc_now(), "source": str(path), "area": area, "rows": rows,
                                                    "note_ko": "코사인 유사도는 확률이나 보정된 신뢰도가 아니다. 추론 중 q·prototype 갱신 없음."})
    figs = []
    if figure and first is not None:
        f = save_feature_figure(s, area, first[1], out_dir / "features_first_image.png", title=first[0].name)
        if f:
            figs.append(str(f))
    return {"rows": rows, "out_dir": str(out_dir), "figures": figs}


# ======================================================================
# 16. 그림 (matplotlib 선택) 과 요약 보고서
# ======================================================================
_MPL_STATE: dict[str, Any] = {"plt": None, "error": None, "tried": False}


def _plt() -> Any:
    if not _MPL_STATE["tried"]:
        _MPL_STATE["tried"] = True
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            _MPL_STATE["plt"] = plt
        except Exception as exc:                             # noqa: BLE001
            _MPL_STATE["error"] = f"{type(exc).__name__}: {exc}"
    return _MPL_STATE["plt"]


def _fig_save(sess: Session, fig: Any, path: Path) -> Path | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=110, bbox_inches="tight", format="png")
        _atomic_write_bytes(path, buf.getvalue())
        return path
    except Exception as exc:                                 # noqa: BLE001
        sess.rec.error(f"그림 저장 실패 {path.name}", exc, echo=False)
        return None
    finally:
        _plt().close(fig)


def save_structure_figure(sess: Session, area: str) -> Path | None:
    plt = _plt()
    if plt is None:
        sess.rec.warn(f"matplotlib 없음 — 구조 그림 생략 ({_MPL_STATE['error']})", echo=False)
        return None
    m = sess.model
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, lay in zip(axes, LAYER_ORDER):
        sp = m.specs[f"{area}/{lay}"]
        ax.scatter(sp.pos_x, sp.pos_y, s=6, c=np.hypot(sp.pos_x - sess.pre.cx, sp.pos_y - sess.pre.cy), cmap="viridis")
        ax.set_xlim(0, sess.pre.S - 1)
        ax.set_ylim(sess.pre.S - 1, 0)
        ax.set_aspect("equal")
        ax.set_title(f"{sp.key}: {sp.n_pos} pos × {sp.n_ch} ch")
    l2 = m.specs.get(f"{area}/L2")
    note = "" if l2 is None or l2.routing is None else \
        f" — curve 2nd input is chosen per image among static candidates (mode {l2.routing.mode}); see figures/routing_*.png"
    fig.suptitle(f"{area} RF centers (image px){note}", fontsize=9 if note else None)
    return _fig_save(sess, fig, sess.rec.p(f"figures/structure_{area}.png"))


def save_prototype_figure(sess: Session, area: str) -> Path | None:
    plt = _plt()
    if plt is None:
        return None
    pc = sess.protos[area].pairwise_cos()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(pc, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(sess.classes)), sess.classes, rotation=45, ha="right")
    ax.set_yticks(range(len(sess.classes)), sess.classes)
    for i in range(pc.shape[0]):
        for j in range(pc.shape[1]):
            if np.isfinite(pc[i, j]):
                ax.text(j, i, f"{pc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{area} prototype cosine (v{sess.protos[area].version})")
    return _fig_save(sess, fig, sess.rec.p(f"figures/prototype_similarity_{area}.png"))


def save_training_figure(sess: Session, area: str) -> Path | None:
    plt = _plt()
    hist = sess.stage.get(area, {}).get("history", [])
    if plt is None or not hist:
        return None
    ep = [h["epoch"] + 1 for h in hist]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(ep, [h["dev_accuracy"] for h in hist], "o-", label="dev acc (overall)")
    axes[0].plot(ep, [h["train_acc_vs_epoch_prototype"] for h in hist], "s--", label="train acc (epoch P)")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[1].plot(ep, [h["mean_dist_before"] for h in hist], "o-", label="dist before")
    axes[1].plot(ep, [h["mean_dist_after"] for h in hist], "s--", label="dist after (same P)")
    axes[1].legend()
    axes[2].plot(ep, [h["frac_improved"] for h in hist], "o-", label="improved")
    axes[2].plot(ep, [h["frac_worsened"] for h in hist], "s--", label="worsened")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("epoch")
    fig.suptitle(f"{area} training (status {sess.stage[area]['status']})")
    return _fig_save(sess, fig, sess.rec.p(f"figures/training_{area}.png"))


def save_role_figure(sess: Session, rr: dict[str, Any]) -> Path | None:
    plt = _plt()
    if plt is None:
        return None
    tb = np.asarray(rr["V1_orientation_lines"]["V1/L2"]["tuning_matrix_bright_lines"])
    oris = sess.cfg["areas"]["V1"]["orientations_deg"]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(tb, cmap="magma")
    ax.set_xticks(range(len(oris)), [f"{o:g}" for o in oris], rotation=90)
    ax.set_yticks(range(len(oris)), [f"{o:g}" for o in oris])
    ax.set_xlabel("V1 L2 channel orientation (deg)")
    ax.set_ylabel("line stimulus orientation (deg)")
    fig.colorbar(im, ax=ax)
    ax.set_title("V1 L2 mean response")
    return _fig_save(sess, fig, sess.rec.p("figures/role_v1_orientation.png"))


def save_feature_figure(sess: Session, area: str, outs: dict[str, np.ndarray], path: Path, title: str = "") -> Path | None:
    plt = _plt()
    if plt is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, lay in zip(axes, ("L2", "L3")):
        key = f"{area}/{lay}"
        sp = sess.model.specs[key]
        v = outs.get(key)
        if v is None:
            continue
        pm = v.reshape(sp.n_pos, sp.n_ch).mean(axis=1)
        sc = ax.scatter(sp.pos_x, sp.pos_y, c=pm, s=30 if sp.n_pos < 200 else 8, cmap="inferno")
        ax.set_xlim(0, sess.pre.S - 1)
        ax.set_ylim(sess.pre.S - 1, 0)
        ax.set_aspect("equal")
        ax.set_title(f"{key} mean transmitted per position")
        fig.colorbar(sc, ax=ax)
    fig.suptitle(title)
    return _fig_save(sess, fig, path)


def save_all_figures(sess: Session) -> list[str]:
    out = []
    for a in sess.model.areas:
        for f in (save_structure_figure(sess, a), save_prototype_figure(sess, a), save_training_figure(sess, a)):
            if f is not None:
                out.append(str(f))
    if sess.model.routing_runtime() is not None and "V4" in sess.model.areas:
        S_ = sess.pre.S
        img = StimulusFactory(S_, supersample=2).arc(S_ / 4.0, width=max(1.5, S_ / 48.0), start_deg=0, span_deg=150)
        f = save_routing_figure(sess, img, "synthetic_arc", true_path=_arc_points((0, 0), S_ / 4.0, 0, 150))
        if f is not None:
            out.append(str(f))
    return out


def build_result_summary(sess: Session) -> Path:
    """저장된 수치로만 result_summary.md 를 만든다 (임의의 수치를 넣지 않는다; 미실행은 '미실행')."""
    r = sess.need_rec()
    L: list[str] = [f"# 결과 요약 — {PROGRAM} v{VERSION}", "", f"- 생성 UTC: {utc_now()}", f"- 실행 폴더: `{r.dir}`",
                    f"- 현재 모델 상태: `{sess.model_version() if sess.model else '모델 없음'}` (마지막 checkpoint `{sess.last_checkpoint}`)",
                    f"- 계산 뒷단: {sess.be.kind} / {sess.be.device}", ""]
    if sess.data is not None:
        man = sess.data.manifest
        L += ["## 데이터", "", f"- 클래스 {man['classes']} (자동 결정), 항목 {man['n_items']}, 지문 `{man['data_fingerprint_sha256'][:16]}`", ""]
        L += ["| 클래스 | train | dev | test |", "|---|---:|---:|---:|"]
        for c, v in man["per_class"].items():
            L.append(f"| {c} | {v['n_train_images']} | {v['n_dev_images']} | {v['n_test_images']} |")
        rp = r.p("data_report.json")
        if rp.is_file():
            d = read_json(rp)
            L += ["", f"- 손상 {len(d['corrupt'])}, 미지원 {len(d['unsupported'])}, 빈 클래스 {d['empty_classes_excluded']}, "
                      f"같은 화소 중복 그룹 {len(d['exact_duplicate_groups'])}, 클래스 간 충돌 제외 {len(d['cross_class_duplicates_excluded'])}", ""]
    if sess.model is not None:
        L += ["## 구조", "", "| 층 | 뉴런 | 위치×채널 | 연결 | 팬인 평균/최대 | min2 | L6 |", "|---|---:|---|---:|---|---:|---|"]
        for k in sess.model.store.order:
            sp = sess.model.specs[k]
            b = sp.build_stats
            L.append(f"| {k} | {sp.n} | {sp.n_pos}×{sp.n_ch} | {b.get('edges_A', 0) + b.get('edges_B', 0)} | "
                     f"{fmt(b.get('fan_in_mean'), 1)}/{b.get('fan_in_max')} | {b.get('n_min2')} | {sess.model.rt[k].l6} |")
        L.append("")
        L += ["## 단계 학습", ""]
        for a in sess.model.areas:
            st = sess.stage[a]
            L.append(f"### {a} — 상태 `{st['status']}`, 에폭 {st['epochs_completed']}/{st['max_epochs']}, "
                     f"best dev {fmt(st['best_dev_accuracy'])} (epoch {st['best_epoch'] + 1 if st['best_epoch'] is not None else 'NA'})")
            if st["history"]:
                L += ["", "| epoch | dev 정확도 | 유효 출력만 | unknown | 거리 전→후 (같은 P) | 개선/악화 | q 효과 | P 효과 | P 최소각 |",
                      "|---:|---|---:|---:|---|---|---:|---:|---:|"]
                for h in st["history"]:
                    L.append(f"| {h['epoch'] + 1} | {pct_count(h['dev_correct'], h['dev_n'])} | {fmt(h['dev_accuracy_valid_only'])} | "
                             f"{h['dev_n_unknown_zero_vector']} | {fmt(h['mean_dist_before'])}→{fmt(h['mean_dist_after'])} | "
                             f"{fmt(h['frac_improved'], 3)}/{fmt(h['frac_worsened'], 3)} | {fmt(h['q_effect_dist'])} | "
                             f"{fmt(h['prototype_effect_dist'])} | {fmt(h['prototype_min_pair_angle_deg'], 2)} |")
            else:
                L.append("- 학습 에폭: 미실행")
            L.append("")
    dj = r.p("diagnostics.json")
    L += ["## 진단", ""]
    if dj.is_file():
        for run in read_json(dj)["runs"][-6:]:
            L.append(f"- {run['utc']} {run['kind']} ({run['model_version']}): {run['status']}")
            for k, v in (run.get("checks") or {}).items():
                L.append(f"  - {k}: {v}")
    else:
        L.append("- 진단: 미실행")
    tests = sorted(r.p("diagnostics").glob("test_eval_*.json"))
    L += ["", "## 최종 test 평가", ""]
    if tests:
        for p in tests:
            t = read_json(p)
            m = t["metrics"]
            L.append(f"- #{t['test_evaluation_number']} {t['utc']} {t['model_version']}: {pct_count(m['correct'], m['n'])} "
                     f"(유효 출력만 {fmt(m['accuracy_valid_only'])}, unknown {m['n_unknown_zero_vector']})")
    else:
        L.append("- 미실행")
    L += routing_summary_lines(sess)
    L += ["", "## 한계 (자동 기록)", "",
          "- 코사인 분류는 출력 벡터 전체 크기 변화만으로는 바뀌지 않는다. 크기 오차와 방향 오차를 따로 기록하지만 두 오차를 L2/L3 책임으로 나누지 않는다.",
          "- prototype 은 벌리기 규칙 없이 EMA 로만 움직인다. collapse 가능성은 prototype_metrics.csv 의 최소각·collapse_warning 으로 본다.",
          "- 순차 동결이라 앞 영역의 정보 손실·오분류 구조는 뒤 영역에서 고칠 수 없다 (전체 미세조정·역전파를 넣지 않았다).",
          "- L1 배달·q 갱신은 검증되지 않은 국소 휴리스틱이다. 거리 개선 비율·q 효과로 실제 효과를 확인한다.",
          "- 가정 전체: assumptions.md, 수식: config.json 의 formulas.",
          "- V4 곡선 뉴런의 둘째 입력은 V1 안내에 따른 입력별 연결 선택이며, 영구 재배선이나 인간 피질의 재현을 입증한 것은 아니다.", ""]
    return r.text("result_summary.md", "\n".join(L))


def routing_summary_lines(sess: Session) -> list[str]:
    """result_summary.md 의 라우팅 절. 저장된 기록만 쓴다 (없으면 '미실행' / '해당 없음')."""
    r = sess.need_rec()
    mode = sess.cfg["routing"]["mode"]
    L = ["", "## V4 곡선 뉴런 둘째 입력 라우팅", "", f"- 이 실행의 라우팅 모드: `{mode}` (모드는 실행마다 고정, 바꾸려면 새 run 으로 분기)"]
    rp = r.p("run_info.json")
    info = read_json(rp).get("branched_from") if rp.is_file() else None
    if info:
        L.append(f"- 분기 원본: `{info['source_run']}` (모드 `{info['source_routing_mode']}`, 재사용 {info['reused_areas']}, "
                 f"원본 checkpoint `{info['source_checkpoint']}`)")
    if mode == "legacy":
        L.append("- legacy 모드: v1 의 곡선 둘째 입력 규칙 (θA 방향 최근접 위치, 없으면 같은 위치) 을 그대로 쓴다. 라우팅 기록은 해당 없음.")
        return L
    sp = r.p("routing/routing_static_summary.json")
    if not sp.is_file():
        L.append("- V4 가 아직 없다 — 후보 표·선택 기록 미실행.")
        return L
    st = read_json(sp)
    L += [f"- 곡선 뉴런 {st['n_curve_neurons']}, 후보 상한 K={st['K']}, 경로 점 M={st['M']}",
          f"- 후보 없음 (missing_candidate, u=0) 비율 {fmt(st['missing_candidate_frac'])} ({st['n_missing']} 개), 유효 후보 평균 {fmt(st['valid_candidates_mean'], 2)}",
          f"- 경로 검사로 제외된 후보 {st['path_rejected_total']} — 사유 {st['path_reject_reasons']}",
          f"- 기하 첫 후보가 legacy 위치와 다른 비율 {fmt(st['geometry_first_differs_from_legacy_frac'])}, legacy 의 같은 위치 대체 비율 "
          f"{fmt(st['legacy_same_position_fallback_frac'])}",
          "- 고리별·방향별 후보 없음: `routing/missing_by_ring_orientation.csv`, 정적 표: `routing/routing_candidates.npz`"]
    rows = r.read_csv("routing/routing_summary.csv") if r.p("routing/routing_summary.csv").is_file() else []
    if rows:
        L += ["", "| 맥락 | 활성 | epoch | 표본 | 안내 없음 | 기하 첫 후보와 다름 | legacy 위치와 다름 | 선택 S 평균 | 첫 칸 몫 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for x in rows[-12:]:
            L.append(f"| {x.get('context')} | {x.get('active_area')} | {x.get('epoch_number')} | {x.get('n_samples')} | {fmt(x.get('no_guidance_frac'))} | "
                     f"{fmt(x.get('differs_from_geometry_first_frac'))} | {fmt(x.get('differs_from_legacy_position_frac'))} | "
                     f"{fmt(x.get('mean_S_selected'))} | {fmt(x.get('selection_top_slot_share'))} |")
    else:
        L.append("- 이미지별 선택 기록: 미실행")
    for a in sess.model.areas if sess.model else []:
        se = sess.stage[a].get("start_eval")
        if se:
            L.append(f"- {a} 학습 전 dev: {pct_count(se['dev_correct'], se['dev_n'])}")
    for kind, label in (("routing_checks", "라우팅 구현 검사"), ("routing_role", "곡선 역할 진단 (측정값)"),
                        ("routing_intervention", "안내 개입 진단 — 즉각적 민감도 검사 (측정값)")):
        fs = sorted(r.p("diagnostics").glob(f"{kind}_*.json"))
        L.append(f"- {label}: " + (f"`{fs[-1].relative_to(r.dir).as_posix()}`" if fs else "미실행"))
    L += ["- 모드 효과는 나눠 본다: legacy→geometry_only = 기하 배선 수정 효과, geometry_only→v1_guided = 안내 지도의 추가 효과. "
          "이 실행은 한 모드만 학습했으므로 다른 모드 결과는 분기한 다른 run 의 기록으로 비교한다.",
          "- S 는 방향 지지 정도이며 확률이 아니다. S 를 u 에 곱하지 않는다."]
    return L


# ======================================================================
# 14b. 라우팅 검사·진단 (구현 검사 = PASSED/FAILED/NOT_RUN, 역할·성능 = MEASURED/NOT_RUN)
# ======================================================================
R01_CONFIG: dict[str, Any] = {
    "data": {"image_size": 48}, "logpolar": {"n_r": 8, "n_phi": 16, "r0_px": 1.5},
    "areas": {"V1": {"n_r": 4, "n_phi": 16}, "V2": {"n_r": 2, "n_phi": 8}, "V4": {"n_r": 2, "n_phi": 8}, "IT": {"region_r": 1, "region_phi": 2}},
}


def _arc_points(center: tuple[float, float], radius: float, start: float, span: float, n: int = 200) -> np.ndarray:
    a = np.radians(start + np.linspace(0.0, span, n))
    return np.stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)], axis=1)   # 시각 좌표


class RoutingDiagnostics:
    """V1 안내 라우팅 검사. 모두 복원 (q·prototype·RNG) 하거나 구조 사본에서 하며, 사용자가 고를 때만 실행한다."""

    def __init__(self, sess: Session):
        self.s = sess
        self.base = Diagnostics(sess)

    # ---------------------------------------------------------------- 공통
    def _table_and_runtime(self) -> tuple[RoutingTable | None, RoutingRuntime | None]:
        rr = self.s.model.routing_runtime() if self.s.model else None
        return (rr.table if rr is not None else None), rr

    def _rebuild_table(self, mode: str) -> RoutingTable:
        m = self.s.model
        if "V2/L3" not in m.specs:
            raise RuntimeError("V2 가 아직 없다 (후보 표는 V2 L3 위에서 만든다).")
        cfg2 = copy.deepcopy(self.s.cfg)
        cfg2["routing"]["mode"] = mode
        return build_v4(cfg2, self.s.pre, m.specs["V2/L3"], m.specs["V1/L3"])[1].routing

    def _fac(self) -> StimulusFactory:
        return StimulusFactory(self.s.pre.S, supersample=2)

    def _forward_imgs(self, imgs: list[np.ndarray], *, routing_mode: str | None = None,
                      guide_transform: Callable[[Any], Any] | None = None) -> dict[str, Any]:
        be = self.s.be
        X = be.f32(np.stack([self.s.pre.from_array(im) for im in imgs]))
        keep = [k for k in ("V1/L3", "V4/L4", "V4/L2") if k in self.s.model.specs]
        return self.s.model.forward(X, keep=keep, routing_mode=routing_mode, guide_transform=guide_transform,
                                    routing_record=("V4" in self.s.model.areas))

    def _snapshot(self) -> dict[str, Any]:
        snap = self.base._snapshot()
        snap["struct"] = dict(self.s.model.struct_hash)
        return snap

    def _restore(self, snap: dict[str, Any]) -> bool:
        ok = self.base._restore(snap)
        return ok and dict(self.s.model.struct_hash) == snap["struct"]

    # ---------------------------------------------------------------- 구현 검사
    def implementation_checks(self) -> dict[str, Any]:
        snap = self._snapshot()
        out: dict[str, Any] = {}
        for name, fn in (("R00_v1_channel_theta_is_tangent", self.r00), ("R01_legacy_reproduces_v1", self.r01),
                         ("R02_same_candidates_geometry_vs_guided", self.r02), ("R03_candidate_conditions_no_self", self.r03),
                         ("R04_edge_cases", self.r04), ("R05_no_label_path", self.r05), ("R06_synthetic_guide_selects_known", self.r06),
                         ("R07_real_curve_stimuli_selection", self.r07), ("R08_batch_single_order_consistency", self.r08),
                         ("R09_save_load_reproduces", self.r09), ("R11_backend_consistency", self.r11),
                         ("R12_no_grad_and_forbidden_calls", self.r12)):
            Diagnostics._run(name, fn, out)
            if not self._restore(snap):
                out[name + "_RESTORE"] = {"status": FAILED, "error": "검사 뒤 q/prototype/RNG/구조 복원 실패"}
        restored = self._restore(snap)
        out["R10_persistent_state_unchanged"] = {"status": PASSED if restored else FAILED,
                                                 "checked_ko": "q, prototype, RNG, 구조 해시 (라우팅 표 포함) 를 검사 전후 비교"}
        summ = {"utc": utc_now(), "model_version": self.s.model_version(), "routing_mode": self.s.cfg["routing"]["mode"], "checks": out,
                "n_failed": sum(1 for v in out.values() if isinstance(v, dict) and v.get("status") == FAILED)}
        summ["status"] = FAILED if summ["n_failed"] else PASSED
        if self.s.rec is not None:
            self.s.rec.json(f"diagnostics/routing_checks_{local_stamp()}_{time.time_ns() % 100000}.json", summ)
            self.base._append_index("routing_checks", summ)
        return summ

    def r00(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        fac = self._fac()
        oris = [float(t) for t in s.cfg["areas"]["V1"]["orientations_deg"]]
        X = be.f32(np.stack([s.pre.from_array(fac.line(t, width=max(1.5, s.pre.S / 48.0))) for t in oris]))
        r = be.to_np(s.model.forward(X, upto="V1")["V1/L3"]["out"]).astype(np.float64)
        sp = s.model.specs["V1/L3"]
        ch_o = np.asarray(sp.channel_meta["orientation_deg"], np.float64)
        lum = np.asarray(sp.channel_meta["color"]) == "lum"
        m = r.reshape(len(oris), sp.n_pos, sp.n_ch).mean(axis=1)
        pref = np.asarray([ch_o[lum][int(np.argmax(m[i, lum]))] for i in range(len(oris))])
        err_t = d_ori(pref, np.asarray(oris))
        err_n = d_ori(pref, np.asarray(oris) + 90.0)
        ok = bool((err_t <= 1e-6).all())
        return {"status": PASSED if ok else FAILED, "stimulus_theta": oris, "preferred_channel": pref.tolist(),
                "matches_tangent": int((err_t <= 1e-6).sum()), "matches_normal": int((err_n <= 1e-6).sum()),
                "convention_ko": ROUTING_CONVENTION_KO["channel_theta"] + " " + ROUTING_CONVENTION_KO["frame"]}

    def r01(self) -> dict[str, Any]:
        rc = self.s.cfg["routing"]
        p = Path(rc["legacy_reference_path"]) if rc["legacy_reference_path"] else Path(__file__).with_name("cortex_local_learning_v1.py")
        if not p.is_file():
            return {"status": NOT_RUN, "reason_ko": f"원본 v1 파일이 없다: {p} (routing.legacy_reference_path 로 지정)"}
        name = f"cortex_v1_reference_{os.getpid()}_{time.time_ns() % 100000}"
        spec = importlib.util.spec_from_file_location(name, p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
            c1 = mod.validate_config(R01_CONFIG)
            c2 = validate_config(deep_merge(R01_CONFIG, {"routing": {"mode": "legacy"}}))
            m1 = mod.CortexModel(c1, mod.Backend("numpy"), mod.RetinaLGNPreprocessor(c1))
            m2 = CortexModel(c2, Backend("numpy"), RetinaLGNPreprocessor(c2))
            for a in AREA_ORDER:
                m1.add_area(a, m1.plan(a))
                m2.add_area(a, m2.plan(a))
            fac = StimulusFactory(int(c2["data"]["image_size"]), supersample=2)
            S = int(c2["data"]["image_size"])
            imgs = [fac.arc(S / 4.0, width=2, start_deg=a, span_deg=150) for a in (0, 120, 240)] + [fac.line(30, width=2), fac.shape("star", S * 0.5)]
            X1 = m1.be.f32(np.stack([m1.pre.from_array(im) for im in imgs]))
            X2 = m2.be.f32(np.stack([m2.pre.from_array(im) for im in imgs]))
            r1, r2 = m1.forward(X1), m2.forward(X2)
            diff = {k: float(np.abs(np.asarray(r1[k]["out"]) - np.asarray(r2[k]["out"])).max()) for k in r1}
            hashes = {a: m1.struct_hash[a] == m2.struct_hash[a] for a in AREA_ORDER}
            cfg_eq = mod.config_structure_sha(c1) == config_structure_sha(c2)
        finally:
            sys.modules.pop(name, None)
        ok = all(hashes.values()) and cfg_eq and max(diff.values()) <= 1e-6
        return {"status": PASSED if ok else FAILED, "reference_file": str(p), "reference_sha256": sha256_file(p),
                "structure_hash_equal": hashes, "config_structure_sha_equal": cfg_eq, "max_abs_diff_by_L3": diff, "tolerance": 1e-6,
                "note_ko": "작은 결정론적 합성 입력 (NumPy) 에서 원본 v1 과 legacy 모드의 V1~IT 출력·구조 해시 비교"}

    def r02(self) -> dict[str, Any]:
        tg, tv = self._rebuild_table("geometry_only"), self._rebuild_table("v1_guided")
        fields = ("curve_rows", "P", "cand_q", "cand_src", "cand_valid", "cand_dist", "cand_dir_err", "cand_geo", "gA_idx", "gA_w",
                  "gB_idx", "gB_w", "pt_valid", "missing")
        eq = {f_: bool(np.array_equal(getattr(tg, f_), getattr(tv, f_))) for f_ in fields}
        m = self.s.model
        cfg_l = copy.deepcopy(self.s.cfg)
        cfg_l["routing"]["mode"] = "legacy"
        leg = build_v4(cfg_l, self.s.pre, m.specs["V2/L3"], m.specs["V1/L3"])
        cfg_v = copy.deepcopy(self.s.cfg)
        cfg_v["routing"]["mode"] = "v1_guided"
        new = build_v4(cfg_v, self.s.pre, m.specs["V2/L3"], m.specs["V1/L3"])
        l2l, l2n = leg[1], new[1]

        def row(sp: LayerSpec, j: int) -> tuple:
            a0, a1, b0, b1 = int(sp.a_ptr[j]), int(sp.a_ptr[j + 1]), int(sp.b_ptr[j]), int(sp.b_ptr[j + 1])
            return (sp.a_src[a0:a1].tolist(), sp.a_w[a0:a1].tolist(), sp.b_src[b0:b1].tolist(), sp.b_w[b0:b1].tolist(), bool(sp.op_min[j]))

        curve = set(int(c) for c in tv.curve_rows)
        other = [j for j in range(l2l.n) if j not in curve]
        first_same = all(row(l2l, j)[:2] == row(l2n, j)[:2] for j in curve)
        structure = {"V4_L4_identical_to_legacy": leg[0].hash() == new[0].hash(),
                     "V4_L3_identical_to_legacy_parent_map_for_L1": leg[2].hash() == new[2].hash(),
                     "V4_L2_non_curve_rows_identical_to_legacy": all(row(l2l, j) == row(l2n, j) for j in other),
                     "curve_first_input_A_identical_to_legacy": first_same,
                     "curve_static_B_empty_in_new_mode": all(not row(l2n, j)[2] for j in curve)}
        ok = all(eq.values()) and all(structure.values())
        return {"status": PASSED if ok else FAILED, "equal_fields": eq, "structure_vs_legacy": structure, "n_curve": int(len(tg.curve_rows)),
                "n_non_curve_rows_compared": len(other),
                "valid_candidates_total": [int(tg.cand_valid.sum()), int(tv.cand_valid.sum())],
                "missing_total": [int(tg.missing.sum()), int(tv.missing.sum())]}

    def r03(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        t, rr = self._table_and_runtime()
        if t is None:
            t = self._rebuild_table("v1_guided")
        rc = s.cfg["routing"]
        l4n = (s.model.specs["V4/L4"].n_ch if "V4/L4" in s.model.specs else
               s.model.specs["V2/L3"].n_ch)
        fam = (s.model.specs["V4/L4"].channel_meta if "V4/L4" in s.model.specs else s.model.specs["V2/L3"].channel_meta)
        v = t.cand_valid
        jj, kk = np.nonzero(v)
        Pj = t.P[jj]
        Q = t.cand_q[jj, kk]
        src = t.cand_src[jj, kk]
        ch = src % l4n
        chan_ok = np.asarray([fam["family"][c] == "contour" and abs(float(fam["orientation_deg"][c]) - float(t.theta_b[j])) < 1e-6
                              for c, j in zip(ch, jj)], bool)
        D = t.D[jj]
        dist_ok = (t.cand_dist[jj, kk] >= float(rc["dist_min_factor"]) * D - 1e-4) & (t.cand_dist[jj, kk] <= float(rc["dist_max_factor"]) * D + 1e-4)
        dir_ok = t.cand_dir_err[jj, kk] <= float(rc["max_chord_dev_deg"]) + 1e-4
        addr_ok = (src >= 0) & (src < t.n_src)
        pad_ok = bool((t.cand_src[~v] == t.n_src).all())
        res = {"n_valid_candidates": int(v.sum()), "no_self": bool((Q != Pj).all()), "address_in_range": bool(addr_ok.all()),
               "padding_uses_zero_column_not_minus1": pad_ok, "channel_is_contour_theta_b": bool(chan_ok.all()),
               "distance_in_range": bool(dist_ok.all()), "direction_within_tolerance": bool(dir_ok.all())}
        if rr is not None and "V4" in s.model.areas:
            lgn, _y, _ = self.base._probe()
            okn = ~t.missing
            for mode in ("geometry_only", "v1_guided"):
                k = be.to_np(s.model.forward(lgn, routing_record=True, routing_mode=mode)["V4/L2"]["routing"]["k"])
                rows_ = np.arange(len(t.P))[None, :].repeat(k.shape[0], 0)
                res[f"selected_valid_{mode}"] = bool(t.cand_valid[rows_, k][:, okn].all())
                res[f"selected_not_P_{mode}"] = bool((t.cand_q[rows_, k][:, okn] != t.P[okn][None, :]).all())
        ok = all(v_ for v_ in res.values() if isinstance(v_, bool))
        return {"status": PASSED if ok else FAILED, **res}

    def r04(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        res: dict[str, Any] = {}
        phi_a, phi_b, phi_c = curve_directions(165.0, +1, 30.0)
        res["seam_chord_deg"] = phi_c
        res["seam_ok"] = abs(((phi_c - 180.0) + 180.0) % 360.0 - 180.0) < 1e-9 and abs(float(d_ori(165.0, 15.0)) - 30.0) < 1e-9
        res["naive_mean_would_be"] = (165.0 + 15.0) / 2.0
        gs = GuideSampler(s.model.specs["V1/L3"], s.pre, s.cfg)
        rpt = 0.5 * s.pre.rmax
        ang = math.radians(-0.4)
        ia, wa, ib, wb, val = gs.table(np.asarray([rpt * math.cos(ang)]), np.asarray([rpt * math.sin(ang)]), np.asarray([0.0]))
        angs = sorted({int(p % gs.n_phi) for p in ib[0]})
        res["polar_seam_angle_bins"] = angs
        res["polar_seam_ok"] = bool(val[0]) and (0 in angs) and (gs.n_phi - 1 in angs) and abs(float(wb[0].sum()) - 1.0) < 1e-5
        ia2, wa2, ib2, wb2, val2 = gs.table(np.asarray([s.pre.rmax + 1.0]), np.asarray([0.0]), np.asarray([0.0]))
        res["outside_image_invalid"] = (not bool(val2[0])) and float(wa2.sum()) == 0.0 and bool((ia2 == gs.n).all())
        _ia3, wa3, _ib3, _wb3, _v3 = gs.table(np.asarray([rpt]), np.asarray([0.0]), np.asarray([gs.ori0 + 180.0 - gs.d_o / 2.0]))
        chans = sorted({int((i_ % gs.n_ch) - gs.lum_base) for i_, w_ in zip(_ia3[0], wa3[0]) if w_ > 0})
        res["orientation_wrap_channels"] = chans
        res["orientation_wrap_ok"] = chans == [0, gs.n_o - 1]
        t, rr = self._table_and_runtime()
        if rr is not None and "V4" in s.model.areas:
            lgn, _y, _ = self.base._probe()
            r = s.model.forward(lgn, keep=("V4/L2",), routing_record=True)
            u = be.to_np(r["V4/L2"]["u"])
            miss_rows = t.curve_rows[t.missing]
            res["missing_rows"] = int(miss_rows.size)
            res["missing_u_zero"] = bool((u[:, miss_rows] == 0).all()) if miss_rows.size else None
            non_curve = np.setdiff1d(np.arange(u.shape[1]), t.curve_rows)
            r_geo = s.model.forward(lgn, keep=("V4/L2",), routing_mode="geometry_only")
            res["other_roles_unaffected_by_mode"] = bool(np.array_equal(u[:, non_curve], be.to_np(r_geo["V4/L2"]["u"])[:, non_curve]))
            z = s.model.forward(lgn, keep=("V4/L2",), routing_record=True, routing_mode="v1_guided", guide_transform=lambda g: g * 0.0)
            ng = be.to_np(z["V4/L2"]["routing"]["no_guidance"])
            kz = be.to_np(z["V4/L2"]["routing"]["k"])
            res["zero_guide_all_no_guidance"] = bool(ng[:, ~t.missing].all())
            res["zero_guide_falls_back_to_geometry_first"] = bool((kz == 0).all()) and bool(
                np.array_equal(be.to_np(z["V4/L2"]["routing"]["B"]), be.to_np(s.model.forward(lgn, routing_mode="geometry_only",
                                                                                                routing_record=True)["V4/L2"]["routing"]["B"])))
            zero = s.model.forward(be.zeros(tuple(lgn.shape)), keep=("V4/L2",))
            res["zero_input_finite_and_zero"] = all(be.all_finite(v_["out"]) and float(np.abs(be.to_np(v_["out"])).max()) == 0.0
                                                    for v_ in zero.values() if "out" in v_)
        else:
            res["model_level_cases"] = NOT_RUN
        ok = all(v_ for v_ in res.values() if isinstance(v_, bool)) and all(v_ is not False for v_ in res.values())
        return {"status": PASSED if ok else FAILED, **res}

    def r05(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        bad = {}
        for fn in (RoutingRuntime.select, RoutingRuntime.scores, GuideSampler.table, CortexModel.forward, build_routing_table):
            names = [p_.lower() for p_ in inspect.signature(fn).parameters]
            hit = [n for n in names if any(w in n for w in ("label", "target", "proto", "class", "truth"))]
            if hit:
                bad[fn.__qualname__] = hit
        src = inspect.getsource(StageTrainer._batch)
        fwd_calls = re.findall(r"forward\(([^)]*)\)", src)
        label_in_forward = [c for c in fwd_calls if re.search(r"\by\b|label", c)]
        res: dict[str, Any] = {"forbidden_parameters": bad, "label_like_args_in_training_forward_calls": label_in_forward}
        if "V4" in s.model.areas and s.model.routing_runtime() is not None:
            items = s.data.items("train")[:4]
            relab = [{**it, "label": (int(it["label"]) + 1) % len(s.classes)} for it in items]      # 같은 이미지, 라벨만 바꾼 사본
            lgn, y1 = s.loader.batch(items, "train")
            lgn2, y2 = s.loader.batch(relab, "train")
            k1 = be.to_np(s.model.forward(lgn, routing_record=True, routing_mode="v1_guided")["V4/L2"]["routing"]["k"])
            k2 = be.to_np(s.model.forward(lgn2, routing_record=True, routing_mode="v1_guided")["V4/L2"]["routing"]["k"])
            res["labels_changed"] = bool((y1 != y2).all())
            res["selection_same_with_changed_labels"] = bool(np.array_equal(k1, k2))
        ok = not bad and not label_in_forward and res.get("selection_same_with_changed_labels", True) and res.get("labels_changed", True)
        return {"status": PASSED if ok else FAILED, **res}

    def r06(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        t, rr = self._table_and_runtime()
        if rr is None:
            return {"status": NOT_RUN, "reason_ko": "현재 모델에 라우팅 (V4, 새 모드) 이 없다."}
        gs = GuideSampler(s.model.specs["V1/L3"], s.pre, s.cfg)
        cand = np.nonzero(t.cand_valid.sum(1) >= 2)[0][:24]
        if cand.size == 0:
            return {"status": NOT_RUN, "reason_ko": "유효 후보가 2 개 이상인 곡선 뉴런이 없다 (격자가 성김)."}
        max_diff, sel_mismatch, hits, strict = 0.0, 0, 0, 0
        for j in cand:
            kt = int(t.cand_valid[j].sum()) - 1
            F1 = np.zeros((1, t.n_v1), np.float64)
            for m_ in range(t.M):
                if t.pt_valid[j, kt, m_]:
                    np.add.at(F1[0], t.gA_idx[j, kt, m_], t.gA_w[j, kt, m_])
            F1 = np.minimum(F1, 1.0).astype(np.float32)
            S_dev = be.to_np(rr.scores(be.f32(F1))).astype(np.float64)[0]
            S_ref = routing_reference_scores(t, F1, s.cfg, gs.n_ch, gs.lum_base, gs.n_o)[0]
            max_diff = max(max_diff, float(np.abs(S_dev - S_ref)[t.cand_valid].max()))
            Sd = np.where(t.cand_valid[j], S_dev[j], -1.0)
            mx = Sd.max()
            k_rule = 0 if mx <= 0 else int(np.nonzero((Sd == mx) & t.cand_valid[j])[0][0])   # 최댓값 중 기하 순서 첫 칸
            xe = be.pad_zero_col(be.zeros((1, t.n_src)))
            k_dev = int(be.to_np(rr.select(xe, be.f32(F1), mode="v1_guided")["k"])[0, j])                 # 실행 모드와 무관하게 안내 선택 규칙 검사
            sel_mismatch += int(k_dev != k_rule)
            Sr = np.where(t.cand_valid[j], S_ref[j], -1.0)
            if Sr[kt] > np.max(np.delete(Sr, kt)):
                strict += 1
                hits += int(k_dev == kt)
        ok = max_diff <= 1e-5 and sel_mismatch == 0 and hits == strict
        return {"status": PASSED if ok else FAILED, "n_neurons_tested": int(cand.size), "max_abs_S_diff_vs_float64_reference": max_diff,
                "selection_mismatch_vs_tie_rule": sel_mismatch, "target_strictly_best": strict, "target_selected_when_strictly_best": hits,
                "selection_rule_tested": "v1_guided", "run_mode": s.cfg["routing"]["mode"],
                "note_ko": "합성 안내 지도 (목표 후보 경로 위에만 반응) 로 점수 함수와 선택 규칙을 검사하는 구현 검사다. 실제 곡선 판별 성능이 아니다."}

    def r07(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        t, rr = self._table_and_runtime()
        if rr is None or "V4" not in s.model.areas:
            return {"status": NOT_RUN, "reason_ko": "라우팅이 있는 V4 가 없다."}
        S = s.pre.S
        fac = self._fac()
        l4 = s.model.specs["V4/L4"]
        pv = np.stack([l4.pos_x - s.pre.cx, s.pre.cy - l4.pos_y], 1).astype(np.float64)
        rows = []
        for R in (S / 6.0, S / 4.0):
            for start in range(0, 360, 60):
                img = fac.arc(R, width=max(1.5, S / 48.0), start_deg=start, span_deg=150)
                arc = _arc_points((0.0, 0.0), R, start, 150)
                rec = {}
                for mode in ("v1_guided", "geometry_only"):
                    rr_ = self._forward_imgs([img], routing_mode=mode)["V4/L2"]["routing"]
                    rec[mode] = be.to_np(rr_["k"])[0]
                dP = np.min(np.hypot(*(pv[t.P][:, None, :] - arc[None, :, :]).transpose(2, 0, 1)), axis=1)
                ang_c = np.degrees(np.arctan2(pv[t.P][:, 1], pv[t.P][:, 0]))
                tang_ccw = ang_c + 90.0
                fwd_ccw = np.cos(np.radians(t.theta_a - tang_ccw)) > 0
                sign_ok = np.where(fwd_ccw, t.sign > 0, t.sign < 0)
                match = (~t.missing) & (dP <= 0.5 * t.D + 1.0) & (d_ori(t.theta_a, tang_ccw) <= 15.0) & sign_ok
                for mode, k in rec.items():
                    q = t.cand_q[np.arange(len(t.P)), k]
                    dq = np.min(np.hypot(*(pv[np.maximum(q, 0)][:, None, :] - arc[None, :, :]).transpose(2, 0, 1)), axis=1)
                    rows.append({"R": R, "start": start, "mode": mode, "n_matched": int(match.sum()),
                                 "mean_Q_dist_to_arc_px": float(dq[match].mean()) if match.any() else None})
                dl = np.min(np.hypot(*(pv[t.legacy_q][:, None, :] - arc[None, :, :]).transpose(2, 0, 1)), axis=1)
                rows.append({"R": R, "start": start, "mode": "legacy_position_reference", "n_matched": int(match.sum()),
                             "mean_Q_dist_to_arc_px": float(dl[match].mean()) if match.any() else None})
        agg = {}
        for mode in ("v1_guided", "geometry_only", "legacy_position_reference"):
            v = [r["mean_Q_dist_to_arc_px"] for r in rows if r["mode"] == mode and r["mean_Q_dist_to_arc_px"] is not None]
            agg[mode] = float(np.mean(v)) if v else None
        return {"status": MEASURED, "per_stimulus": rows, "mean_selected_Q_distance_to_arc_px": agg,
                "note_ko": "전체 영상부터 처리한 원호 자극에서 기하적으로 맞는 곡선 뉴런 (P 가 원호 위, θA 가 접선, 곡률 부호 일치) 의 선택 Q 와 원호 거리. "
                           "성공 기준이 아니라 측정값이다."}

    def r08(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        if s.model.routing_runtime() is None or "V4" not in s.model.areas:
            return {"status": NOT_RUN, "reason_ko": "라우팅이 있는 V4 가 없다."}
        items = s.data.items("train")[:4]
        out: dict[str, Any] = {"tolerance": 1e-6}
        ok = True
        for mode in ("v1_guided", "geometry_only"):
            def run(its: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
                x_, _ = s.loader.batch(its, "train")
                r_ = s.model.forward(x_, keep=("V4/L2",), routing_record=True, routing_mode=mode)
                return be.to_np(r_["V4/L2"]["routing"]["k"]), be.to_np(r_["V4/L2"]["out"])
            kb, ob = run(items)
            singles = [run([it]) for it in items]
            kr, orr = run(items[::-1])
            same_k = bool(np.array_equal(kb, np.concatenate([k_ for k_, _ in singles])) and np.array_equal(kb, kr[::-1]))
            dmax = float(max(np.abs(ob - np.concatenate([o_ for _, o_ in singles])).max(), np.abs(ob - orr[::-1]).max()))
            out[mode] = {"selection_identical": same_k, "max_abs_output_diff": dmax}
            ok &= same_k and dmax <= 1e-6
        return {"status": PASSED if ok else FAILED, **out}

    def r09(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        t, rr = self._table_and_runtime()
        if rr is None:
            return {"status": NOT_RUN, "reason_ko": "라우팅이 있는 V4 가 없다."}
        meta = read_json(s.rec.p(f"checkpoints/{s.last_checkpoint}.json")) if s.last_checkpoint else {}
        saved = (meta.get("routing") or {}).get("table_sha256")
        t2 = self._rebuild_table(s.cfg["routing"]["mode"])
        gs = GuideSampler(s.model.specs["V1/L3"], s.pre, s.cfg)
        rr2 = RoutingRuntime(t2, be, s.cfg, s.model.specs["V4/L2"].n, gs.n_ch, gs.lum_base, gs.n_o)
        lgn, _y, _ = self.base._probe()
        r = s.model.forward(lgn, keep=("V4/L4",), routing_record=True)
        g = r["V1/L3"]["out"]
        xe = be.pad_zero_col(r["V4/L4"]["out"])
        k1 = np.stack([be.to_np(rr.select(xe, g, mode=m_)["k"]) for m_ in ("geometry_only", "v1_guided")])
        k2 = np.stack([be.to_np(rr2.select(xe, g, mode=m_)["k"]) for m_ in ("geometry_only", "v1_guided")])
        ok = saved == t.hash() == t2.hash() and bool(np.array_equal(k1, k2))
        return {"status": PASSED if ok else FAILED, "checkpoint_table_sha256": saved, "current": t.hash(), "rebuilt": t2.hash(),
                "selection_identical_after_rebuild": bool(np.array_equal(k1, k2)), "checkpoint": s.last_checkpoint}

    def r11(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        t, rr = self._table_and_runtime()
        if rr is None:
            return {"status": NOT_RUN, "reason_ko": "라우팅이 있는 V4 가 없다."}
        if be.kind == "numpy":
            return {"status": NOT_RUN, "reason_ko": "현재 NumPy 경로라 CPU/GPU (torch) 비교를 할 수 없다. GPU 가 있는 PC 에서 실행한다."}
        gs = GuideSampler(s.model.specs["V1/L3"], s.pre, s.cfg)
        lgn, _y, _ = self.base._probe()
        r = s.model.forward(lgn, routing_record=True, routing_mode="v1_guided")
        g = r["V1/L3"]["out"]
        S_dev = be.to_np(r["V4/L2"]["routing"]["S"]).astype(np.float64)
        S_ref = routing_reference_scores(t, be.to_np(g), s.cfg, gs.n_ch, gs.lum_base, gs.n_o)
        k_dev = be.to_np(r["V4/L2"]["routing"]["k"])
        Sm = np.where(t.cand_valid[None], S_ref, -1.0)
        mx = Sm.max(-1, keepdims=True)
        k_ref = np.argmax((Sm == mx) * np.arange(t.K, 0, -1), axis=-1)
        k_ref = np.where(mx[..., 0] <= 0, 0, k_ref)
        mism = (k_dev != k_ref) & ~t.missing[None]
        top2 = np.sort(Sm, -1)[..., -2:] if t.K >= 2 else np.zeros(Sm.shape[:-1] + (2,))
        near = (top2[..., 1] - top2[..., 0]) < 1e-5
        ok = bool((mism <= near).all()) and float(np.abs(S_dev - S_ref)[np.broadcast_to(t.cand_valid[None], S_ref.shape)].max()) <= 1e-4
        return {"status": PASSED if ok else FAILED, "device": be.device, "max_abs_S_diff_vs_cpu_float64": float(np.abs(S_dev - S_ref).max()),
                "selection_mismatches": int(mism.sum()), "mismatches_at_near_ties": int((mism & near).sum()), "near_ties": int(near.sum())}

    def r12(self) -> dict[str, Any]:
        r = self.base.c14()
        hits = {}
        for obj in (RoutingRuntime, GuideSampler, build_routing_table, hermite_path, path_check, routing_batch_stats):
            src = inspect.getsource(obj)
            found = [p for p in FORBIDDEN_GRAD_PATTERNS if re.search(p, src)]
            if found:
                hits[obj.__name__] = found
        t, rr = self._table_and_runtime()
        rg = self.s.be.requires_grad_any([rr.gA_w, rr.gB_w, rr.cand_src]) if rr is not None else False
        ok = r["status"] == PASSED and not hits and not rg
        return {"status": PASSED if ok else FAILED, "base_c14": r["status"], "routing_forbidden_patterns": hits, "routing_tensors_requires_grad": rg,
                "note_ko": "p'(t) 는 기하 경로의 접선이며 손실 미분·역전파가 아니다."}

    # ---------------------------------------------------------------- 곡선 역할 진단 (MEASURED)
    def _matched(self, t: RoutingTable, pv: np.ndarray, arc: dict[str, float]) -> np.ndarray:
        """원호 위에 P 가 있고 θA 가 접선 (±15°) 이며 곡률 부호가 진행 방향과 맞는 곡선 뉴런 (기하 기준, 선택과 무관)."""
        c = np.asarray(arc["center"], np.float64)
        pts = _arc_points(arc["center"], arc["R"], arc["start"], arc["span"])
        dP = np.min(np.hypot(*(pv[t.P][:, None, :] - pts[None, :, :]).transpose(2, 0, 1)), axis=1)
        rel = pv[t.P] - c[None, :]
        ang = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
        on_span = d_dir(ang, arc["start"] + arc["span"] / 2.0) <= arc["span"] / 2.0 + 10.0
        tang = ang + 90.0                                                    # 반시계 진행 접선
        fwd = np.cos(np.radians(t.theta_a - tang)) > 0
        sign_ok = np.where(fwd, t.sign > 0, t.sign < 0)
        return (dP <= 0.5 * t.D + 1.0) & on_span & (d_ori(t.theta_a, tang) <= 15.0) & sign_ok

    def role_diagnostics(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        if "V4" not in s.model.areas:
            return {"status": NOT_RUN, "reason_ko": "V4 가 없다."}
        t_live, rr = self._table_and_runtime()
        t = t_live if t_live is not None else self._rebuild_table("v1_guided")   # legacy: 곡선 뉴런 기하 (P, θA, 부호) 만 참고
        S = s.pre.S
        fac = self._fac()
        w = max(1.5, S / 48.0)
        R = S / 4.0
        rng = np.random.default_rng([int(s.cfg["seed"]), 777])
        l4 = s.model.specs["V4/L4"]
        pv = np.stack([l4.pos_x - s.pre.cx, s.pre.cy - l4.pos_y], 1).astype(np.float64)

        def dots(pts: list[tuple[float, float]]) -> np.ndarray:
            img = np.zeros((S, S, 3), np.float32)
            for (x, y) in pts:
                img = np.maximum(img, fac.shape("circle", 2.5 * w, center=(x, y), fg=1.0, bg=0.0))
            return img

        g_arc = {"center": (0.0, 0.0), "R": R, "start": 0.0, "span": 150.0}
        g_opp = {"center": (0.0, 2.0 * R), "R": R, "start": 200.0, "span": 140.0}       # 같은 꼭대기 (0, R) 를 지나는 반대 곡률
        ends = _arc_points((0, 0), R, 0, 150, 2)
        stim: dict[str, tuple[np.ndarray, dict[str, Any] | None]] = {
            "arc_ccw": (fac.arc(R, width=w, start_deg=0, span_deg=150), g_arc),
            "arc_opposite_curvature": (fac.arc(R, width=w, start_deg=200, span_deg=140, center=(0.0, 2.0 * R)), g_opp),
            "arc_gap_middle": (np.maximum(fac.arc(R, width=w, start_deg=0, span_deg=50), fac.arc(R, width=w, start_deg=100, span_deg=50)), g_arc),
            "two_dots_at_arc_ends": (dots([tuple(ends[0]), tuple(ends[1])]), None),
            "straight_line_tangent_at_arc_top": (fac.line(0.0, width=w, center=(0.0, R)), None),
            "corner": (fac.corner(0.0, 60.0, width=w), None), "cross": (fac.cross(0.0, 90.0, width=w), None),
            "arc_low_contrast": (fac.arc(R, width=w, start_deg=0, span_deg=150, fg=0.5), g_arc),
            "arc_shifted": (fac.arc(R, width=w, start_deg=0, span_deg=150, center=(S / 10.0, 0.0)), {**g_arc, "center": (S / 10.0, 0.0)}),
            "arc_rotated_90": (fac.arc(R, width=w, start_deg=90, span_deg=150), {**g_arc, "start": 90.0}),
            "noise": (StimulusFactory.rgb(rng.uniform(0, 1, (S, S))), None)}
        T_arc = self._matched(t, pv, g_arc)
        T_opp = self._matched(t, pv, g_opp)
        sp = s.model.specs["V4/L2"]
        fam = np.asarray(sp.channel_meta["family"])
        ch_of = np.arange(sp.n) % sp.n_ch
        curve_rows = np.nonzero(fam[ch_of] == "curve")[0]
        straight_rows = np.nonzero(fam[ch_of] == "straight")[0]
        run_mode = s.cfg["routing"]["mode"]
        modes = [run_mode] + (["geometry_only"] if rr is not None and run_mode != "geometry_only" else [])
        rows: dict[str, Any] = {}
        for name, (img, geo) in stim.items():
            row = {}
            own = self._matched(t, pv, geo) if geo is not None else None
            for mode in modes:
                r = self._forward_imgs([img], routing_mode=(None if mode == run_mode else mode))
                out = be.to_np(r["V4/L2"]["out"])[0].astype(np.float64)
                v = {"arc_target_neurons_mean_out": float(out[t.curve_rows[T_arc]].mean()) if T_arc.any() else None,
                     "opposite_target_neurons_mean_out": float(out[t.curve_rows[T_opp]].mean()) if T_opp.any() else None,
                     "all_curve_mean_out": float(out[curve_rows].mean()), "all_straight_mean_out": float(out[straight_rows].mean()),
                     "curve_max_out": float(out[curve_rows].max())}
                if own is not None:
                    v["own_matched_neurons"] = int(own.sum())
                    v["own_matched_missing_candidate_frac"] = float(t.missing[own].mean()) if own.any() else None
                    v["own_matched_mean_out"] = float(out[t.curve_rows[own]].mean()) if own.any() else None
                if rr is not None and "routing" in r["V4/L2"]:
                    rn = routing_rec_np(be, r["V4/L2"]["routing"])
                    k = rn["k"][0]
                    v["selection_differs_from_geometry_first_frac"] = float((k[~t.missing] != 0).mean()) if (~t.missing).any() else None
                    if own is not None and (own & ~t.missing).any():
                        sel = own & ~t.missing
                        pts = _arc_points(geo["center"], geo["R"], geo["start"], geo["span"])
                        q = t.cand_q[np.arange(len(t.P)), k]
                        dq = np.min(np.hypot(*(pv[np.maximum(q, 0)][:, None, :] - pts[None, :, :]).transpose(2, 0, 1)), axis=1)
                        v["own_selected_Q_dist_to_generating_path_px"] = float(dq[sel].mean())
                        if rn["S"] is not None:
                            v["own_selected_S_mean"] = float(np.take_along_axis(rn["S"][0], k[:, None], 1)[:, 0][sel].mean())
                        if rn.get("R_sel") is not None:
                            ps = point_summary(rn["R_sel"][0][sel], rn["OK_sel"][0][sel])
                            v.update({f"own_selected_{k_}_mean": float(x_.mean()) for k_, x_ in ps.items()})
                row[mode] = v
            rows[name] = row
        ref = rows["arc_ccw"][run_mode]
        ratios = {}
        for nm in stim:
            if nm == "arc_ccw":
                continue
            v = rows[nm][run_mode]
            if ref["arc_target_neurons_mean_out"]:
                ratios[f"arc_targets_{nm}_over_arc_ccw"] = (v["arc_target_neurons_mean_out"] or 0.0) / ref["arc_target_neurons_mean_out"]
        if ref.get("opposite_target_neurons_mean_out"):
            ratios["opposite_targets_opposite_arc_over_arc_ccw"] = (rows["arc_opposite_curvature"][run_mode]["opposite_target_neurons_mean_out"] or 0.0) / \
                ref["opposite_target_neurons_mean_out"]
        out = {"status": MEASURED, "utc": utc_now(), "model_version": s.model_version(), "routing_mode": run_mode, "modes_measured": modes,
               "n_arc_target_neurons": int(T_arc.sum()), "n_opposite_target_neurons": int(T_opp.sum()),
               "arc_target_missing_candidate_frac": float(t.missing[T_arc].mean()) if T_arc.any() else None,
               "stimuli": rows, "ratios_run_mode": ratios,
               "target_rule_ko": "대상 뉴런 = 생성 원호 위 (0.5D+1 화소) 에 P, θA 가 접선 ±15°, 곡률 부호가 진행 방향과 일치 (기하 기준이며 학습·선택과 무관)",
               "note_ko": "곡선 역할 반응 측정값이다. 같은 끝점 (두 점)·중간 끊김·접선 직선·반대 곡률과 비교한다. 실패를 없애려고 사후에 기준을 바꾸지 않는다. "
                          "후보 없음 뉴런은 새 모드에서 u=0 이라 반응이 0 이며 이는 학습 실패가 아니다."}
        if s.rec is not None:
            s.rec.json(f"diagnostics/routing_role_{local_stamp()}.json", out)
            self.base._append_index("routing_role", {"utc": out["utc"], "model_version": out["model_version"], "status": MEASURED})
            f = save_routing_figure(s, stim["arc_ccw"][0], "arc_ccw", true_path=_arc_points((0, 0), R, 0, 150))
            if f:
                out["figure"] = str(f)
            f2 = save_routing_figure(s, stim["arc_opposite_curvature"][0], "arc_opposite_curvature", true_path=_arc_points((0.0, 2.0 * R), R, 200, 140))
            if f2:
                out["figure_opposite"] = str(f2)
        return out

    # ---------------------------------------------------------------- 안내 개입 진단 (즉각적 민감도 검사, MEASURED)
    def intervention_diagnostics(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        t, rr = self._table_and_runtime()
        area = s.model.active
        if rr is None or AREA_INDEX[area] < AREA_INDEX["V4"]:
            return {"status": NOT_APPLICABLE if rr is None else NOT_RUN,
                    "reason_ko": "라우팅 (새 모드) V4 가 활성이거나 그 뒤 영역일 때만 한다."}
        snap = self._snapshot()
        rng = np.random.default_rng([int(s.cfg["seed"]), 4242])            # 별도 무작위 스트림
        sp = s.model.specs["V1/L3"]
        perm_pos = rng.permutation(sp.n_pos)
        idx_pos = (perm_pos[:, None] * sp.n_ch + np.arange(sp.n_ch)[None, :]).reshape(-1)
        gs = GuideSampler(sp, s.pre, s.cfg)
        perm_o = rng.permutation(gs.n_o)
        ch = np.arange(sp.n_ch)
        ch[gs.lum_base:gs.lum_base + gs.n_o] = gs.lum_base + perm_o
        idx_ori = (np.arange(sp.n_pos)[:, None] * sp.n_ch + ch[None, :]).reshape(-1)
        ip, io = be.i64(idx_pos), be.i64(idx_ori)
        bank = s.protos[area]
        try:
            conds = {"normal_guide": dict(routing_mode="v1_guided"),
                     "spatially_shuffled_guide": dict(routing_mode="v1_guided", guide_transform=lambda g: be.take_cols(g, ip)),
                     "orientation_shuffled_guide": dict(routing_mode="v1_guided", guide_transform=lambda g: be.take_cols(g, io)),
                     "geometry_only": dict(routing_mode="geometry_only")}
            res = {nm: s.evaluate(area, "dev", bank, purpose=f"routing_intervention_{nm}", collect_routing=True, **kw) for nm, kw in conds.items()}
        finally:
            restored = self._restore(snap)
        k0 = res["normal_guide"]["routing_records"]["k"]
        rows = {}
        for nm, m in res.items():
            kk = m["routing_records"]["k"]
            rows[nm] = {"dev_accuracy": m["accuracy_overall"], "dev_correct": m["correct"], "dev_n": m["n"],
                        "selection_changed_vs_normal_frac": float((kk[:, ~t.missing] != k0[:, ~t.missing]).mean()) if (~t.missing).any() else None,
                        "routing": m.get("routing")}
        out = {"status": MEASURED, "utc": utc_now(), "model_version": s.model_version(area), "active_area": area, "conditions": rows,
               "restored": restored, "trained_routing_mode": s.cfg["routing"]["mode"],
               "condition_rules_ko": "정상·섞은 안내 조건은 v1_guided 선택 규칙, geometry_only 조건은 기하 첫 후보 (학습한 모드와 무관하게 같은 고정 모델)",
               "note_ko": "같은 고정 모델에서 안내만 바꾼 즉각적 민감도 검사다 (V4 L4 감각 입력·q·prototype 고정, 별도 무작위 스트림). "
                          "각 모드를 제대로 학습한 최종 성능 비교가 아니다."}
        s.rec.json(f"diagnostics/routing_intervention_{local_stamp()}.json", out)
        self.base._append_index("routing_intervention", {"utc": out["utc"], "model_version": out["model_version"], "status": MEASURED})
        return out


def save_routing_figure(sess: Session, img: np.ndarray, title: str, *, true_path: np.ndarray | None = None, n_neurons: int = 10,
                        routing_mode: str | None = None) -> Path | None:
    """원영상 + V1 방향 반응 + V4 첫 입력 P·후보 Q·선택 Q*·가정한 경로. 가정한 경로는 정답 윤곽이 아니다."""
    plt = _plt()
    m = sess.model
    rr = m.routing_runtime() if m else None
    if plt is None or rr is None or "V4" not in m.areas:
        return None
    be, t, pre = sess.be, rr.table, sess.pre
    X = be.f32(sess.pre.from_array(img)[None, :])
    r = m.forward(X, keep=("V1/L3", "V4/L2"), routing_mode=routing_mode, routing_record=True)
    rec = routing_rec_np(be, r["V4/L2"]["routing"])
    v1 = be.to_np(r["V1/L3"]["out"])[0]
    sp1 = m.specs["V1/L3"]
    gs = GuideSampler(sp1, pre, sess.cfg)
    F = v1.reshape(sp1.n_pos, sp1.n_ch)[:, gs.lum_base:gs.lum_base + gs.n_o]
    u = np.minimum(rec["A"][0], rec["B"][0])
    order = [int(j) for j in np.argsort(-u) if not t.missing[j]][:n_neurons]
    l4 = m.specs["V4/L4"]
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.imshow(img.mean(axis=2), cmap="gray", vmin=0, vmax=1, alpha=0.55)
    mx = float(F.max()) if F.size else 0.0
    if mx > 0:
        for p in range(sp1.n_pos):
            c = int(np.argmax(F[p]))
            a = F[p, c] / mx
            if a < 0.15:
                continue
            th = math.radians(gs.ori0 + gs.d_o * c)
            L = 2.5 * a
            x0, y0 = sp1.pos_x[p], sp1.pos_y[p]
            ax.plot([x0 - L * math.cos(th), x0 + L * math.cos(th)], [y0 + L * math.sin(th), y0 - L * math.sin(th)], color="tab:cyan", lw=1, alpha=0.8)
    if true_path is not None:
        ax.plot(true_path[:, 0] + pre.cx, pre.cy - true_path[:, 1], "--", color="tab:green", lw=2, label="synthetic generating path")
    lab = {"P": False, "cand": False, "rej": False, "sel": False, "path": False}
    for j in order:
        P = t.P[j]
        ax.plot(l4.pos_x[P], l4.pos_y[P], "o", color="tab:blue", ms=7, label=None if lab["P"] else "V4 first input P")
        lab["P"] = True
        k = int(rec["k"][0, j])
        for s_ in range(t.K):
            if not t.cand_valid[j, s_]:
                continue
            q = t.cand_q[j, s_]
            if s_ == k:
                ax.plot(l4.pos_x[q], l4.pos_y[q], "*", color="tab:red", ms=13, label=None if lab["sel"] else "selected Q*")
                ax.plot(t.path_xy[j, s_, :, 0], t.path_xy[j, s_, :, 1], "-", color="tab:red", lw=1.6,
                        label=None if lab["path"] else "assumed path (not ground truth)")
                lab["sel"] = lab["path"] = True
            else:
                ax.plot(l4.pos_x[q], l4.pos_y[q], "o", mfc="none", color="tab:orange", ms=8, label=None if lab["cand"] else "unselected candidate")
                ax.plot(t.path_xy[j, s_, :, 0], t.path_xy[j, s_, :, 1], ":", color="tab:orange", lw=0.8)
                lab["cand"] = True
        for s_ in range(t.rejected_q.shape[1]):
            q = t.rejected_q[j, s_]
            if q >= 0:
                ax.plot(l4.pos_x[q], l4.pos_y[q], "x", color="0.4", ms=6, label=None if lab["rej"] else "invalid candidate (path check)")
                lab["rej"] = True
    ax.set_xlim(0, pre.S - 1)
    ax.set_ylim(pre.S - 1, 0)
    ax.set_aspect("equal")
    ax.set_title(f"V1-guided routing: {title} (mode {routing_mode or t.mode}, top {len(order)} curve neurons by u)")
    ax.legend(loc="lower left", fontsize=7)
    return _fig_save(sess, fig, sess.rec.p(f"figures/routing_{re.sub(r'[^A-Za-z0-9_]+', '_', title)}.png"))


# ======================================================================
# 17. 한국어 메뉴와 CLI
# ======================================================================
def ask(prompt: str, default: str | None = None) -> str:
    sfx = f" [{default}]" if default not in (None, "") else ""
    try:
        v = input(f"{prompt}{sfx}: ")
    except EOFError:
        v = ""
    v = v.strip()
    return v if v else (default or "")


def ask_yes(prompt: str, default: bool = False) -> bool:
    v = ask(prompt + (" (Y/n)" if default else " (y/N)"), "").lower()
    return default if not v else v in ("y", "yes", "예", "ㅇ", "1")


def print_summary_table(summ: dict[str, Any], say: Callable[[str], None] = print) -> None:
    say("  층            뉴런     위치×채널     연결(edge)   팬인 평균/최대   min2")
    for r in summ["layers"]:
        say(f"  {r['layer']:<10} {r['n_neurons']:>8}   {r['positions']:>5}×{r['channels']:<5} {r['edges']:>11}   "
            f"{fmt(r['fan_in_mean'], 1):>8}/{r['fan_in_max']:<6} {r['n_min2']:>6}")
    m = summ["memory_mb"]
    say(f"  합계 뉴런 {summ['total_neurons']}, 연결 {summ['total_edges']}, ELL 칸 {summ['ell_slots']}")
    say(f"  예상 메모리 (MB): 구조 {m['graph']}, 배치 활동 상한 {m['batch_activations_upper']}, gather 조각 상한 {m['gather_chunk_upper']}, "
        f"prototype {m['prototypes']}, 진단·기록 버퍼 {m['diagnostic_and_record_buffers']}, 기존 모델 {m['existing_model']}, "
        f"합계 약 {m['total_estimate']} (배치 {summ['batch_size']}, {summ['device']})")
    say("  " + summ["note_ko"])


def print_train_result(res: dict[str, Any], say: Callable[[str], None] = print) -> None:
    be_ = res.get("best_epoch")
    say(f"  결과: {res.get('area')} 상태 {res.get('status')}, 완료 에폭 {res.get('epochs_completed')}, best dev {fmt(res.get('best_dev_accuracy'))}"
        f" (epoch {be_ + 1 if be_ is not None else 'NA'}), checkpoint {res.get('checkpoint')}")
    if res.get("message_ko"):
        say("  " + res["message_ko"])


class MenuApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = Config.load(args.config, args.seed)
        print(f"{PROGRAM} v{VERSION} — 3×3 뉴런 기반 시각 피질 모사 국소 학습 (공학적 근사, 학습 성공을 보장하지 않음)")
        self.be = self._backend(args.backend, args.device)
        self.sess = Session(self.cfg, self.be, say=print, confirm=ask_yes)
        DeviceManager.print_environment(DeviceManager.environment(self.be))
        try:
            self.sess.set_paths(clean_user_path(args.data_root) if args.data_root else None,
                                clean_user_path(args.output_dir) if args.output_dir else None)
        except Exception as exc:                             # noqa: BLE001
            print(f"  [경로 오류] {exc}")

    def _backend(self, backend: str, device: str) -> Backend:
        try:
            return DeviceManager.create(backend, device, self.cfg)
        except RuntimeError as exc:
            print(f"  [계산 뒷단] {exc}")
            if torch is None and backend == "auto":
                print("  1) 종료 (PyTorch 설치 후 다시 실행)   2) NumPy CPU 경로로 계속 (같은 수식, GPU 미사용, 느림)")
                if ask("선택", "1") == "2":
                    return DeviceManager.create("numpy", "cpu", self.cfg)
            raise SystemExit(1)

    def status(self) -> str:
        s = self.sess
        run = s.rec.dir.name if s.rec else "없음"
        model = "없음"
        if s.model and s.model.areas:
            a = s.model.active
            st = s.stage[a]
            model = f"{'/'.join(s.model.areas)} (활성 {a}: {st['status']}, 에폭 {st['epochs_completed']}/{st['max_epochs']})"
        return (f"데이터 {s.data_root or '미설정'} | 저장 {s.output_root or '미설정'} | run {run} | 모델 {model} | 라우팅 {s.cfg['routing']['mode']} | "
                f"{self.be.kind}/{self.be.device}")

    def loop(self) -> None:
        menu = ["1) 데이터 폴더와 저장 폴더 설정 / 환경 확인", "2) 데이터 검사 및 학습·검증·시험 분할", "3) 새 V1 모델 생성 / 구조·연결 요약",
                "4) 현재 활성 영역 학습 또는 중단 지점 재개", "5) 다음 영역 추가 및 학습 단계 전환", "6) 저장 모델로 이미지 한 장 또는 폴더 추론",
                "7) 전달·교정·prototype·역할 진단", "8) 모델, 로그, 시각화 저장", "9) V1 안내 라우팅 (V4 곡선 뉴런 둘째 입력)", "0) 종료"]
        while True:
            print("\n" + "=" * 78)
            print(" " + self.status())
            print("-" * 78)
            for m in menu:
                print("  " + m)
            c = ask("선택")
            try:
                if c == "0":
                    if self.sess.model is not None and self.sess.rec is not None and ask_yes("종료 전에 checkpoint 를 저장할까?", True):
                        self.sess.save_checkpoint("menu_exit", keep=True)
                    return
                fn = {"1": self.m1, "2": self.m2, "3": self.m3, "4": self.m4, "5": self.m5, "6": self.m6, "7": self.m7, "8": self.m8, "9": self.m9}.get(c)
                if fn is None:
                    print("  없는 선택이다.")
                    continue
                fn()
            except KeyboardInterrupt:
                print("\n  [취소] 메뉴로 돌아간다.")
            except Exception as exc:                         # noqa: BLE001
                print(f"  [오류] {type(exc).__name__}: {exc}")
                traceback.print_exc()
                if self.sess.rec is not None:
                    self.sess.rec.error(f"메뉴 {c} 오류", exc, echo=False)

    # ---------------------------------------------------------------- 메뉴
    def m1(self) -> None:
        s = self.sess
        raw = ask("데이터 폴더 (하위 폴더 = 클래스, Enter = 유지)", str(s.data_root) if s.data_root else "")
        out = ask("저장 폴더 (Enter = 유지)", str(s.output_root) if s.output_root else "")
        s.set_paths(clean_user_path(raw) if raw else None, clean_user_path(out) if out else None)
        print(f"  데이터: {s.data_root}\n  저장: {s.output_root} (쓰기 확인)")
        env = DeviceManager.environment(self.be)
        DeviceManager.print_environment(env)
        if s.model is None and s.data is None:
            dv = ask("장치 변경 (auto/cuda/cpu, Enter = 유지)", "")
            if dv in ("auto", "cuda", "cpu"):
                self.be = self._backend("torch" if torch is not None else "numpy", dv if torch is not None else "cpu")
                paths = (s.data_root, s.output_root)
                self.sess = Session(self.cfg, self.be, say=print, confirm=ask_yes)
                self.sess.set_paths(*paths)
                DeviceManager.print_environment(DeviceManager.environment(self.be))
        if s.rec is not None:
            s.rec.json("environment.json", env)

    def _new_session(self) -> None:
        paths = (self.sess.data_root, self.sess.output_root)
        self.sess = Session(self.cfg, self.be, say=print, confirm=ask_yes)
        self.sess.set_paths(*paths)

    def m2(self) -> None:
        if self.sess.model is not None:
            if not ask_yes("현재 세션에 모델이 있다. 새 실행 (새 run 폴더) 으로 다시 시작할까?"):
                return
            self._new_session()
        rep, man = self.sess.scan_and_split(progress=print)
        print(f"  클래스 {rep['classes']} ({rep['n_classes']} 개, 자동 결정), 유효 이미지 {rep['n_valid_images']}")
        print(f"  손상 {len(rep['corrupt'])}, 미지원 {len(rep['unsupported'])}, 빈 클래스 {rep['empty_classes_excluded']}, "
              f"같은 화소 중복 그룹 {len(rep['exact_duplicate_groups'])}, 클래스 간 충돌 제외 {len(rep['cross_class_duplicates_excluded'])}")
        print(f"  원본 크기 {rep['original_size']}")
        for c, v in man["per_class"].items():
            print(f"    {c}: 그룹 {v['n_groups']} → train {v['n_train_images']}, dev {v['n_dev_images']}, test {v['n_test_images']}")
        dc = read_json(self.sess.rec.p("preprocessing.json"))["dc_check"]
        print(f"  전처리 DC 검사 (균일 영상): {dc['status']} (최대 잔여 {dc['max_abs_interior']:.2e})")
        print(f"  실행 폴더: {self.sess.rec.dir}")

    def m3(self) -> None:
        s = self.sess
        if s.model is not None and s.model.areas:
            print(f"  이미 모델이 있다 ({s.model.areas}). 구조 요약: {s.rec.p('architecture.json')}")
            for k in s.model.store.order:
                sp = s.model.specs[k]
                print(f"    {k}: {sp.n} 뉴런 ({sp.n_pos}×{sp.n_ch}), 팬인 평균 {fmt(sp.build_stats['fan_in_mean'], 1)}, "
                      f"출력 없음 {int((s.model.store.fan_out(k) == 0).sum())}")
            return
        area, specs, summ = s.plan_next()
        print(f"  {area} 구조 계획 (장치 텐서를 만들기 전):")
        print_summary_table(summ)
        if not ask_yes(f"{area} 를 생성하고 train 출력으로 prototype 을 초기화할까?", True):
            return
        rec = s.build_area(area, specs, summ)
        print(f"  prototype 초기화: 클래스별 표본 {rec['counts']}, norm {[round(v, 4) for v in rec['norms']]}, 무효 {rec['invalid_classes']}")
        save_structure_figure(s, area)
        print(f"  구조 기록: {s.rec.p('architecture.json')}")

    def m4(self) -> None:
        s = self.sess
        if s.model is None:
            raw = ask("재개할 run 폴더 (Enter = 취소)")
            if not raw:
                return
            dr = ask("데이터 폴더 (Enter = run 기록의 경로)", "")
            self.sess = s = Session.load(clean_user_path(raw), self.be, say=print, confirm=ask_yes,
                                         data_root=clean_user_path(dr) if dr else None)
        area = s.model.active
        st = s.stage[area]
        extra = None
        print(f"  활성 영역 {area}: 상태 {st['status']}, 에폭 {st['epochs_completed']}/{st['max_epochs']}, 다음 배치 {st['next_batch']}")
        if st["status"] in STAGE_TERMINAL:
            v = ask("이미 끝난 단계다. 추가 에폭 수 (Enter = 취소)", "")
            if not v:
                return
            extra = int(v)
        print("  Ctrl+C 한 번: 배치 경계에서 멈춤 / 두 번: 즉시 중단 (진행 중 배치 취소)")
        res = StageTrainer(s).run(extra_epochs=extra)
        print_train_result(res)
        save_training_figure(s, area)
        save_prototype_figure(s, area)
        build_result_summary(s)

    def m5(self) -> None:
        s = self.sess
        if s.model is None:
            print("  모델이 없다.")
            return
        area, specs, summ, info = s.add_next_area_prepare()
        print(f"  다음 영역 {area}: 현재 {info['current_area']} 상태 {info['current_status']}, best dev {fmt(info['best_dev_accuracy'])}")
        if info["warning_ko"]:
            print("  [경고] " + info["warning_ko"])
        print_summary_table(summ)
        if not ask_yes(f"{info['current_area']} 를 동결하고 {area} 를 추가할까?", False):
            s.rec.append_jsonl("stage_transitions.jsonl", {"utc": utc_now(), "from": info["current_area"], "to": area,
                                                          "user_choice": "declined", "warning_ko": info["warning_ko"]})
            return
        out = s.add_next_area_commit(area, specs, summ, info, user_choice="menu_confirmed")
        print(f"  {info['current_area']} 동결 해시 기록, {area} prototype 초기화 (무효 {out['prototype_init']['invalid_classes']})")
        save_structure_figure(s, area)

    def m6(self) -> None:
        s = self.sess
        if s.model is None:
            raw = ask("저장 run 폴더 (Enter = 취소)")
            if not raw:
                return
            self.sess = s = Session.load(clean_user_path(raw), self.be, say=print, confirm=ask_yes)
        p = ask("이미지 파일 또는 폴더 경로")
        if not p:
            return
        area = ask(f"비교 영역 ({'/'.join(s.model.areas)})", s.model.active)
        res = run_inference(s, clean_user_path(p), area=area)
        for r in res["rows"][:30]:
            if r["status"] == "load_failed":
                print(f"    {Path(r['image']).name}: 읽기 실패 {r['error']}")
                continue
            cs = ", ".join(f"{c}={fmt(r.get('cos_' + c), 3)}" for c in s.classes)
            print(f"    {Path(r['image']).name}: {r['prediction']} [{cs}] norm {fmt(r['output_norm'])} {r['status']}")
        print(f"  전체 결과: {res['out_dir']} (코사인은 확률이 아니다)")

    def m7(self) -> None:
        s = self.sess
        if s.model is None:
            print("  모델이 없다.")
            return
        d = Diagnostics(s)
        while True:
            print("  7-1) 기본 검사 14 항목  7-2) 역할 반응 (통제 자극)  7-3) 역할 교란 비교  7-4) 뉴런 조회 (3×3)")
            print("  7-5) 최종 test 평가 (고정 모델)  7-6) q·prototype 상태 요약  7-7) 전처리 DC·정보 손실  0) 돌아가기")
            c = ask("진단 선택")
            if c in ("0", ""):
                return
            if c == "1":
                r = d.run_basic()
                for k, v in r["checks"].items():
                    print(f"    {k}: {v.get('status')}" + (f" — {v.get('error')}" if v.get("error") else ""))
                print(f"  종합: {r['status']} (복원 {r['restored']})")
            elif c == "2":
                r = d.role_responses()
                ov = r["V1_orientation_lines"]
                print(f"    V1 L2 선호 방향 일치 {fmt(ov['V1/L2']['preferred_matches_stimulus_fraction'])}, L3 {fmt(ov['V1/L3']['preferred_matches_stimulus_fraction'])}")
                print(f"    균일 영상 최대 출력 {r['constant_images']}")
                for k in ("V1_phase_modulation", "V2_roles", "V4_roles", "translation"):
                    if k in r:
                        print(f"    {k}: {r[k]}")
            elif c == "3":
                kind = ask("교란 종류 (l2_rows_within_position / l4_spatial)", "l2_rows_within_position")
                frac = float(ask("교란 비율 (0~1)", "0.5"))
                r = d.perturbation(kind, frac)
                for k, v in r["dev_accuracy"].items():
                    print(f"    {k}: {pct_count(v['correct'], v['n'])}")
                print("    " + r["note_ko"])
            elif c == "4":
                raw = ask("neuron_id 또는 '영역/층:local' (예: V1/L3:12 = 그 영역 h 의 12 번 성분)")
                if ":" in raw:
                    key, loc = raw.split(":", 1)
                    nid = s.model.specs[key.strip()].start + int(loc)
                else:
                    nid = int(raw)
                use = ask_yes("train 첫 표본으로 동적 값 (CONF·전송 출력) 도 계산할까?", True)
                info = s.inspect_neuron(nid, s.data.items("train")[0] if use else None)
                print(NeuronInspector.format(info))
            elif c == "5":
                print(f"  지금까지 test 평가 {s.counters['test_evaluations']} 회. test 는 고정 모델 최종 평가용이며 선택에 쓰지 않는다.")
                if ask_yes("현재 고정 모델로 test 평가를 할까?"):
                    r = d.test_evaluation()
                    m = r["metrics"]
                    print(f"    test {pct_count(m['correct'], m['n'])}, 유효 출력만 {fmt(m['accuracy_valid_only'])}, unknown {m['n_unknown_zero_vector']}")
            elif c == "6":
                print(dumps(d.state_summary()))
            elif c == "7":
                dc = s.pre.dc_check()
                print(f"    DC 검사 {dc['status']} (내부 {dc['max_abs_interior']:.2e}, 경계 {dc['max_abs_border']:.2e}, LGN {dc['max_abs_lgn_sample']:.2e})")
                it = s.data.items("train")[0]
                im, _m = load_pil_rgb(s.data.root / it["relpath"], s.pre.bg)
                arr, _m2 = s.pre.resize_pad(im)
                il = s.pre.info_loss(arr)
                s.rec.json(f"diagnostics/preprocessing_{local_stamp()}.json", {"dc_check": dc, "info_loss_first_train_image": il,
                                                                               "info": s.pre.info()})
                for name, rows in il["maps"].items():
                    print(f"    {name}: " + ", ".join(f"r {r['r_from']:.0f}~{r['r_to']:.0f}: {fmt(r['rel_error'], 3)}" for r in rows))
                print(f"    Rmax 밖 화소 비율 {fmt(il['outside_rmax_pixel_fraction'], 3)}")

    def m8(self) -> None:
        s = self.sess
        if s.model is None:
            print("  저장할 모델이 없다.")
            return
        s.save_checkpoint("menu_save", keep=True)
        figs = save_all_figures(s)
        p = build_result_summary(s)
        print(f"  checkpoint {s.last_checkpoint}, 그림 {len(figs)} 개, 요약 {p}")


    def m9(self) -> None:
        while True:
            s = self.sess
            print(f"  현재 라우팅 모드: {s.cfg['routing']['mode']} — " + ROUTING_MODE_KO[s.cfg["routing"]["mode"]])
            print("  9-1) 라우팅 상태·후보 통계 보기  9-2) 새 실행의 라우팅 모드 선택  9-3) 다른 모드로 분기 (V1·V2 재사용, 새 run)")
            print("  9-4) 라우팅 구현 검사 R00~R12  9-5) 곡선 역할 진단 (통제 자극, 측정값)  9-6) 안내 개입 진단 (즉각적 민감도 검사, dev)")
            print("  9-7) 라우팅 그림 (이미지 파일 또는 원호 자극)  9-8) 설계 변경 기록  0) 돌아가기")
            c = ask("라우팅 선택")
            if c in ("0", ""):
                return
            if c == "1":
                for line in routing_status_lines(s):
                    print("    " + line)
            elif c == "2":
                self.m9_mode()
            elif c == "3":
                self.m9_branch()
            elif c in ("4", "5", "6", "7"):
                if s.model is None:
                    print("  모델이 없다 (메뉴 3, 또는 메뉴 4 로 run 을 불러온다).")
                    continue
                rd = RoutingDiagnostics(s)
                if c == "4":
                    r = rd.implementation_checks()
                    for k, v in r["checks"].items():
                        print(f"    {k}: {v.get('status')}" + (f" — {v.get('error') or v.get('reason_ko')}" if v.get("error") or v.get("reason_ko") else ""))
                    print(f"  종합: {r['status']} (실패 {r['n_failed']})")
                elif c == "5":
                    r = rd.role_diagnostics()
                    if r["status"] != MEASURED:
                        print(f"  {r['status']}: {r.get('reason_ko')}")
                        continue
                    print(f"    원호 대상 곡선 뉴런 {r['n_arc_target_neurons']} (후보 없음 {fmt(r['arc_target_missing_candidate_frac'])}), "
                          f"반대 곡률 대상 {r['n_opposite_target_neurons']}")
                    for nm, row in r["stimuli"].items():
                        print(f"    {nm}: " + "; ".join(f"{m_} 원호 대상 평균 {fmt(v['arc_target_neurons_mean_out'])}, 전체 곡선 평균 {fmt(v['all_curve_mean_out'])}"
                                                      for m_, v in row.items()))
                    for k_, v_ in r["ratios_run_mode"].items():
                        print(f"    {k_}: {fmt(v_, 3)}")
                    print("    " + r["note_ko"] + (f"  그림 {r['figure']}" if r.get("figure") else ""))
                elif c == "6":
                    r = rd.intervention_diagnostics()
                    if r["status"] != MEASURED:
                        print(f"  {r['status']}: {r.get('reason_ko')}")
                        continue
                    for nm, row in r["conditions"].items():
                        print(f"    {nm}: dev {pct_count(row['dev_correct'], row['dev_n'])}, 정상 안내 대비 선택 변경 {fmt(row['selection_changed_vs_normal_frac'])}")
                    print("    " + r["note_ko"])
                else:
                    raw = ask("이미지 파일 경로 (Enter = 합성 원호 자극)", "")
                    if raw:
                        im, _m = load_pil_rgb(clean_user_path(raw), s.pre.bg)
                        img, _m2 = s.pre.resize_pad(im)
                        title, path = Path(raw).stem, None
                    else:
                        S_ = s.pre.S
                        img = StimulusFactory(S_, supersample=2).arc(S_ / 4.0, width=max(1.5, S_ / 48.0), start_deg=0, span_deg=150)
                        title, path = "arc_stimulus", _arc_points((0, 0), S_ / 4.0, 0, 150)
                    mode = ask("모드 (Enter = 실행 모드, geometry_only = 비교용)", "")
                    f = save_routing_figure(s, img, title, true_path=path, routing_mode=mode or None)
                    print(f"  그림: {f}" if f else "  그림을 만들 수 없다 (matplotlib 없음 또는 라우팅 V4 없음).")
            elif c == "8":
                if s.rec is None:
                    print("  실행 폴더가 없다.")
                    continue
                p_ = s.rec.p("design_changes.jsonl")
                if p_.is_file():
                    for line in p_.read_text(encoding="utf-8").splitlines()[-10:]:
                        print("    " + line)
                note = ask("새 설계 변경 내용 (Enter = 기록 안 함)", "")
                if note:
                    rec = s.log_design_change(note)
                    print("  기록했다." + (" [주의] test 를 이미 본 뒤의 변경이다 — 이후 test 는 독립 평가가 아니다." if rec["after_test_seen"] else ""))

    def m9_mode(self) -> None:
        s = self.sess
        if s.model is not None:
            print("  이 실행은 이미 모델이 있다. 모드를 바꾸면 V4 정의가 달라지므로 9-3 (분기, 새 run) 을 쓴다.")
            return
        print("  " + "\n  ".join(f"{m_}: {ROUTING_MODE_KO[m_]}" for m_ in ROUTING_MODES))
        mode = ask("새 실행의 라우팅 모드", s.cfg["routing"]["mode"])
        if mode not in ROUTING_MODES:
            print("  없는 모드다.")
            return
        if s.rec is not None and not ask_yes("이미 run 폴더가 있다 (설정 기록됨). 새 모드는 새 run 으로 시작한다 (메뉴 2 를 다시 한다). 진행할까?"):
            return
        cfg = copy.deepcopy(self.cfg)
        cfg["routing"]["mode"] = mode
        self.cfg = validate_config(cfg)
        self._new_session()
        print(f"  새 실행 모드 {mode}. 메뉴 2 (데이터 분할) 부터 진행한다.")

    def m9_branch(self) -> None:
        s = self.sess
        raw = ask("분기 원본 run 폴더 (V1·V2 를 만든 실행; Enter = 현재 run)", str(s.rec.dir) if s.rec else "")
        if not raw:
            return
        mode = ask(f"새 run 의 라우팅 모드 ({'/'.join(ROUTING_MODES)})", "v1_guided")
        if mode not in ROUTING_MODES:
            print("  없는 모드다.")
            return
        if s.rec is not None and s.model is not None:
            s.save_checkpoint("before_branch", keep=True)
        new = Session.branch(clean_user_path(raw), self.be, mode=mode, say=print, confirm=ask_yes, output_root=s.output_root)
        self.sess = new
        print(f"  새 run {new.rec.dir}: V1·V2 재사용, V4 를 {mode} 로 새로 만들고 train 으로 prototype 초기화. 메뉴 4 로 V4 를 학습한다.")


ROUTING_MODE_KO = {"legacy": "v1 재현 (θA 방향 최근접 위치, 없으면 같은 위치)", "geometry_only": "수정된 기하 후보의 고정 첫 후보 (V1 안내 없음)",
                   "v1_guided": "같은 후보 중 이번 이미지의 V1 L3 경로 지지 점수 최대 후보 (새 실행 기본값)"}


def routing_status_lines(s: Session) -> list[str]:
    out = [f"라우팅 모드 {s.cfg['routing']['mode']} — {ROUTING_MODE_KO[s.cfg['routing']['mode']]}"]
    if s.rec is None:
        return out + ["실행 폴더 없음"]
    info = read_json(s.rec.p("run_info.json")).get("branched_from") if s.rec.p("run_info.json").is_file() else None
    if info:
        out.append(f"분기 원본 {info['source_run']} (모드 {info['source_routing_mode']}, 재사용 {info['reused_areas']})")
    sp = s.rec.p("routing/routing_static_summary.json")
    if sp.is_file():
        st = read_json(sp)
        out += [f"곡선 뉴런 {st['n_curve_neurons']}, 후보 없음 {fmt(st['missing_candidate_frac'])} ({st['n_missing']}), 유효 후보 평균 {fmt(st['valid_candidates_mean'], 2)}, "
                f"경로 검사 제외 {st['path_rejected_total']} {st['path_reject_reasons']}",
                f"고리별 후보 없음 {st['missing_by_ring']}", f"표 해시 {st['table_sha256'][:16]}"]
    else:
        out.append("V4 라우팅 표 없음 (legacy 이거나 V4 미생성)")
    rows = s.rec.read_csv("routing/routing_summary.csv")
    for x in rows[-6:]:
        out.append(f"{x['context']} {x['active_area']} epoch {x['epoch_number']}: 안내 없음 {fmt(x.get('no_guidance_frac'))}, 기하 첫 후보와 다름 "
                   f"{fmt(x.get('differs_from_geometry_first_frac'))}, 선택 S 평균 {fmt(x.get('mean_S_selected'))}")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{PROGRAM}: 3×3 뉴런 기반 시각 피질 모사 국소 학습 (인자 없이 실행하면 한국어 메뉴)")
    p.add_argument("--data-root", default=None, help="데이터 폴더 (하위 폴더 = 클래스)")
    p.add_argument("--output-dir", default=None, help="저장 폴더 (새 run 폴더를 그 안에 만든다)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"],
                   help="auto/torch = PyTorch (없으면 오류), numpy = 같은 수식의 CPU 대체 경로 (명시할 때만)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default=None, help="설정 JSON (DEFAULT_CONFIG 위에 덮어쓴다)")
    p.add_argument("--self-test", action="store_true", help="극소 인공 이미지로 구현 자체 검사 (사용자 데이터 미사용)")
    p.add_argument("--train-stage", default=None, choices=list(AREA_ORDER), help="이 영역까지 순서대로 학습 (앞 단계는 동결)")
    p.add_argument("--epochs", type=int, default=None,
                   help="이번 실행에서 학습하는 단계의 에폭 예산 (완료 에폭 + N). 이미 끝난 단계는 --train-stage 의 목표일 때만 늘린다")
    p.add_argument("--infer", default=None, help="추론할 이미지 파일 또는 폴더")
    p.add_argument("--resume", default=None, help="이어 갈 run 폴더")
    p.add_argument("--diagnose", action="store_true", help="기본 검사 14 항목")
    p.add_argument("--role-test", action="store_true", help="역할 반응 (통제 자극)")
    p.add_argument("--test-eval", action="store_true", help="최종 test 평가 (고정 모델)")
    p.add_argument("--routing-mode", default=None, choices=list(ROUTING_MODES),
                   help="새 실행의 V4 곡선 둘째 입력 라우팅 (기본 v1_guided). --resume 과 다르면 오류 (모드 변경은 --branch-from)")
    p.add_argument("--branch-from", default=None, help="이 run 의 V1·V2 를 재사용해 --routing-mode 로 V4 를 새로 만드는 새 run")
    p.add_argument("--routing-check", action="store_true", help="라우팅 구현 검사 R00~R12")
    p.add_argument("--routing-diag", action="store_true", help="곡선 역할 진단 + 안내 개입 진단 (즉각적 민감도 검사, 측정값)")
    p.add_argument("--yes", action="store_true", help="확인 질문에 모두 예")
    p.add_argument("--menu", action="store_true", help="다른 인자가 있어도 메뉴로 시작")
    return p


def run_cli(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config, args.seed)
    be = DeviceManager.create(args.backend, args.device, cfg)
    conf: Callable[[str], bool] = (lambda _m: True) if args.yes else ask_yes
    DeviceManager.print_environment(DeviceManager.environment(be))
    if args.resume and args.branch_from:
        raise SystemExit("--resume 과 --branch-from 은 함께 쓰지 않는다.")
    if args.branch_from:
        if not args.routing_mode:
            raise SystemExit("--branch-from 에는 --routing-mode 가 필요하다.")
        sess = Session.branch(clean_user_path(args.branch_from), be, mode=args.routing_mode, say=print, confirm=conf,
                              data_root=clean_user_path(args.data_root) if args.data_root else None,
                              output_root=clean_user_path(args.output_dir) if args.output_dir else None)
        print(f"  분기한 새 run {sess.rec.dir} (모드 {args.routing_mode})")
    elif args.resume:
        sess = Session.load(clean_user_path(args.resume), be, say=print, confirm=conf,
                            data_root=clean_user_path(args.data_root) if args.data_root else None)
        if args.seed is not None and int(args.seed) != int(sess.cfg["seed"]):
            raise CompatibilityError(f"--seed {args.seed} 가 실행의 seed {sess.cfg['seed']} 와 다르다.")
        if args.routing_mode and args.routing_mode != sess.cfg["routing"]["mode"]:
            raise CompatibilityError(f"--routing-mode {args.routing_mode} 가 실행의 모드 {sess.cfg['routing']['mode']} 와 다르다 "
                                     "(모드 변경은 --branch-from 으로 새 run 을 만든다).")
    else:
        if args.routing_mode:
            cfg["routing"]["mode"] = args.routing_mode
            cfg = validate_config(cfg)
        if not (args.data_root and args.output_dir):
            raise SystemExit("새 실행에는 --data-root 와 --output-dir 가 필요하다 (또는 --resume).")
        sess = Session(cfg, be, say=print, confirm=conf)
        sess.set_paths(clean_user_path(args.data_root), clean_user_path(args.output_dir))
        rep, man = sess.scan_and_split(progress=print)
        print(f"  클래스 {rep['classes']}, 항목 {man['n_items']}, 실행 폴더 {sess.rec.dir}")
        area, specs, summ = sess.plan_next()
        print_summary_table(summ)
        if not conf(f"{area} 를 생성할까?"):
            return 1
        sess.build_area(area, specs, summ)
    code = 0
    if args.train_stage:
        target = args.train_stage
        if AREA_INDEX[target] < AREA_INDEX[sess.model.active]:
            raise SystemExit(f"현재 활성 영역 {sess.model.active} 보다 앞 단계 {target} 로 돌아가지 않는다.")
        while True:
            area = sess.model.active
            st = sess.stage[area]
            ended = st["status"] in STAGE_TERMINAL
            if not ended or (area == target and args.epochs):          # 끝난 앞 단계는 목표 단계일 때만 더 학습한다
                res = StageTrainer(sess).run(extra_epochs=args.epochs)
                print_train_result(res)
                if res["status"] not in STAGE_TERMINAL:
                    code = 2
                    break
            else:
                print(f"  {area} 단계는 이미 '{st['status']}' 로 끝났다 (목표 단계가 아니라 더 학습하지 않는다).")
            if area == target:
                break
            nxt, specs, summ, info = sess.add_next_area_prepare()
            if info["warning_ko"]:
                print("  [경고] " + info["warning_ko"])
            print_summary_table(summ)
            if not conf(f"{area} 를 동결하고 {nxt} 를 추가할까?"):
                break
            sess.add_next_area_commit(nxt, specs, summ, info, user_choice="cli_train_stage_auto_progress")
    d = Diagnostics(sess)
    if args.diagnose:
        r = d.run_basic()
        for k, v in r["checks"].items():
            print(f"    {k}: {v.get('status')}" + (f" — {v.get('error')}" if v.get("error") else ""))
        code = code or (3 if r["status"] == FAILED else 0)
    if args.role_test:
        r = d.role_responses()
        print(f"  V1 방향 일치 {fmt(r['V1_orientation_lines']['V1/L2']['preferred_matches_stimulus_fraction'])}")
    if args.routing_check:
        r = RoutingDiagnostics(sess).implementation_checks()
        for k, v in r["checks"].items():
            print(f"    {k}: {v.get('status')}" + (f" — {v.get('error') or v.get('reason_ko')}" if v.get("error") or v.get("reason_ko") else ""))
        code = code or (3 if r["status"] == FAILED else 0)
    if args.routing_diag:
        rd = RoutingDiagnostics(sess)
        r1 = rd.role_diagnostics()
        print(f"  곡선 역할 진단: {r1['status']} {r1.get('ratios_run_mode', r1.get('reason_ko'))}")
        r2 = rd.intervention_diagnostics()
        print(f"  안내 개입 진단 (즉각적 민감도 검사): {r2['status']}")
        for nm, row in (r2.get("conditions") or {}).items():
            print(f"    {nm}: dev {pct_count(row['dev_correct'], row['dev_n'])}, 선택 변경 {fmt(row['selection_changed_vs_normal_frac'])}")
    for line in routing_status_lines(sess):
        print("  [라우팅] " + line)
    if args.infer:
        res = run_inference(sess, clean_user_path(args.infer))
        for r in res["rows"][:30]:
            print(f"    {Path(r['image']).name}: {r.get('prediction')} {r['status']}")
        print(f"  결과 {res['out_dir']}")
    if args.test_eval and conf("최종 test 평가를 할까? (고정 모델, 횟수 기록)"):
        r = d.test_evaluation()
        print(f"  test {pct_count(r['metrics']['correct'], r['metrics']['n'])}")
    sess.save_checkpoint("cli_end", keep=True)
    save_all_figures(sess)
    print(f"  요약: {build_result_summary(sess)}")
    return code


# ======================================================================
# 18. 자체 검사 (--self-test): 극소 인공 입력에 대한 기능 검사 (성능 검사가 아니다)
# ======================================================================
SELFTEST_CONFIG: dict[str, Any] = {
    # 자체 검사용 작은 구성 (라우팅 후보가 생기도록 V4 격자를 3×12 로 둔다 — 사용자 기본 설정과 별개이며 성능 검사가 아니다)
    "data": {"image_size": 64},
    "logpolar": {"n_r": 12, "n_phi": 24, "r0_px": 1.5},
    "areas": {"V1": {"n_r": 6, "n_phi": 24}, "V2": {"n_r": 3, "n_phi": 12}, "V4": {"n_r": 3, "n_phi": 12}, "IT": {"region_r": 1, "region_phi": 4}},
    "learning": {"batch_size": 4, "max_epochs": 2, "patience": 3, "eta": 0.05},
    "records": {"checkpoint_every_batches": 2, "correction_samples_per_epoch": 6, "correction_samples_max_per_stage": 20,
                "diagnostic_samples": 3, "keep_periodic_checkpoints": 2},
}


def make_selftest_dataset(root: Path, n_per_class: int = 8, size: int = 64, seed: int = 5) -> dict[str, Any]:
    _require_pil()
    rng = np.random.default_rng(seed)
    fac = StimulusFactory(size, supersample=2)
    made: dict[str, Any] = {"classes": ["circle", "square", "triangle"], "files": []}
    for cls in made["classes"]:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = fac.shape(cls, size * rng.uniform(0.45, 0.7), center=(rng.uniform(-4, 4), rng.uniform(-4, 4)),
                            rotation_deg=(0.0 if cls == "circle" else rng.uniform(0, 90)), fg=0.1, bg=0.95)
            tint = np.asarray([1.0, rng.uniform(0.9, 1.0), rng.uniform(0.85, 1.0)], np.float32)
            arr = (np.clip(img * tint, 0, 1) * 255).astype(np.uint8)
            p = d / f"{cls}_{i:02d}.png"
            Image.fromarray(arr).save(p)
            made["files"].append(p.relative_to(root).as_posix())
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., :3] = (StimulusFactory(size, 2).shape("circle", size * 0.5, fg=0.1, bg=0.95) * 255).astype(np.uint8)
    rgba[..., 3] = 255
    rgba[:8, :8, 3] = 0
    Image.fromarray(rgba, "RGBA").save(root / "circle" / "circle_rgba.png")
    shutil.copyfile(root / "square" / "square_00.png", root / "square" / "square_00_copy1.png")
    (root / "triangle" / "broken.png").write_bytes(bytes(rng.integers(0, 256, 200, dtype=np.uint8)))
    (root / "triangle" / "notes.txt").write_text("not an image", encoding="utf-8")
    (root / "zz_empty").mkdir(exist_ok=True)
    return made


def run_self_test(output_dir: Path | None, be: Backend, say: Callable[[str], None] = print) -> int:
    root = ensure_writable_dir(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="cortex_selftest_"))
    if output_dir is None:
        say(f"  --output-dir 가 없어 임시 폴더를 쓴다: {root}")
    st_dir = unique_run_dir(root, prefix="selftest")
    data_dir = st_dir / "synthetic_data"
    runs = st_dir / "runs"
    make_selftest_dataset(data_dir)
    cfg = validate_config(SELFTEST_CONFIG)
    results: dict[str, Any] = {}
    t_start = time.perf_counter()
    quiet = lambda _m: None                                   # noqa: E731

    def check(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            r = fn()
        except Exception as exc:                             # noqa: BLE001
            r = {"status": FAILED, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8)}
        r["seconds"] = round(time.perf_counter() - t0, 3)
        results[name] = r
        say(f"  {name:<44} {r['status']}" + (f"  {r.get('error')}" if r.get("error") else ""))
        return r

    def new_session() -> Session:
        s = Session(cfg, be, say=quiet)
        s.set_paths(data_dir, runs)
        s.scan_and_split()
        area, specs, summ = s.plan_next()
        s.build_area(area, specs, summ)
        return s

    main: dict[str, Session] = {}

    def st01() -> dict[str, Any]:
        s = Session(cfg, be, say=quiet)
        s.set_paths(data_dir, runs)
        rep, man = s.scan_and_split()
        main["s"] = s
        dup = [g for g in rep["exact_duplicate_groups"] if any("square_00" in x for x in g)]
        split_of = {it["relpath"]: it["split"] for it in man["items"]}
        dup_same = bool(dup) and len({split_of[x] for x in dup[0]}) == 1
        per_ok = all(v["n_train_images"] > 0 and v["n_dev_images"] > 0 and v["n_test_images"] > 0 for v in man["per_class"].values())
        ok = (rep["classes"] == ["circle", "square", "triangle"] and len(rep["corrupt"]) == 1 and
              any(u.endswith("notes.txt") for u in rep["unsupported"]) and rep["empty_classes_excluded"] == ["zz_empty"] and dup_same and per_ok)
        return {"status": PASSED if ok else FAILED, "classes": rep["classes"], "corrupt": len(rep["corrupt"]),
                "unsupported": rep["unsupported"], "empty": rep["empty_classes_excluded"], "duplicate_group_same_split": dup_same,
                "per_class": man["per_class"]}

    def st02() -> dict[str, Any]:
        s = main["s"]
        dc = s.pre.dc_check()
        it = s.data.items("train")[0]
        im, _m = load_pil_rgb(s.data.root / it["relpath"])
        arr, _m2 = s.pre.resize_pad(im)
        il = s.pre.info_loss(arr)
        return {"status": dc["status"], "dc_max_abs": dc["max_abs_interior"], "info_loss": il["maps"]}

    def st03() -> dict[str, Any]:
        s = main["s"]
        area, specs, summ = s.plan_next()
        s.build_area(area, specs, summ)
        a = cfg["areas"]["V1"]
        npos = a["n_r"] * a["n_phi"]
        L4, L2, L3 = (s.model.specs[f"V1/{k}"] for k in LAYER_ORDER)
        rt = s.model.routers["V1"]
        ok = (L4.n == s.pre.n_pos * 2 and L2.n == npos * 12 * 2 * 2 and L3.n == npos * 12 and bool((rt.parent_count == 4).all())
              and bool((s.model.store.fan_out("V1/L2") == 1).all()))
        return {"status": PASSED if ok else FAILED, "n": {k: s.model.specs[k].n for k in s.model.store.order},
                "summary_memory_mb": summ["memory_mb"], "L4_without_output": int((s.model.store.fan_out("V1/L4") == 0).sum())}

    def st04() -> dict[str, Any]:
        s = main["s"]
        ok = True
        rows = []
        for key in s.model.store.order:
            sp = s.model.specs[key]
            for j in (0, sp.n // 2, sp.n - 1):
                info = s.inspect_neuron(sp.start + j, s.data.items("train")[0])
                good = (info["n_input"] == int(sp.fan_in()[j]) and info["n_next"] == int(s.model.store.fan_out(key)[j])
                        and info["matrix_3x3"][2][2] == float(cfg["neuron"]["q_init"]) and info["matrix_3x3"][2][1] is not None)
                ok &= good
                rows.append({"neuron_id": sp.start + j, "ok": good})
        txt = NeuronInspector.format(s.inspect_neuron(s.model.specs["V1/L2"].start + 3))
        return {"status": PASSED if ok else FAILED, "rows": rows, "example": txt}

    def st05() -> dict[str, Any]:
        r = Diagnostics(main["s"]).run_basic()
        return {"status": r["status"], "checks": {k: v.get("status") for k, v in r["checks"].items()},
                "errors": {k: v.get("error") for k, v in r["checks"].items() if v.get("error")}}

    def st06() -> dict[str, Any]:
        s = main["s"]
        q0 = s.model.store.q_numpy()
        res = StageTrainer(s).run()
        q1 = s.model.store.q_numpy()
        rows = [r for r in s.rec.read_csv("training.csv") if r["area"] == "V1" and r.get("row_kind") != "before_training"]
        ok = (s.stage["V1"]["epochs_completed"] == 2 and len(rows) == 2 and s.protos["V1"].version == 3 and np.isfinite(q1).all()
              and not np.array_equal(q0, q1) and res["status"] in STAGE_TERMINAL)
        return {"status": PASSED if ok else FAILED, "result": res, "q_changed": int((q0 != q1).sum()),
                "last_row": {k: rows[-1].get(k) for k in ("dev_accuracy", "frac_improved", "q_effect_dist", "prototype_effect_dist")} if rows else None,
                "note_ko": "q 가 바뀌었는지 (학습 경로 실행) 만 본다. 분류 개선은 통과 조건이 아니다."}

    def st07() -> dict[str, Any]:
        A = new_session()
        A.stage["V1"]["max_epochs"] = 1
        StageTrainer(A).run()
        B = new_session()
        B.stage["V1"]["max_epochs"] = 1
        r1 = StageTrainer(B).run(stop_after_batches=2)
        B2 = Session.load(B.rec.dir, be, say=quiet)
        r2 = StageTrainer(B2).run()
        same_q = bool(np.array_equal(A.model.store.q_numpy(), B2.model.store.q_numpy()))
        same_P = A.protos["V1"].sha() == B2.protos["V1"].sha()
        rows = [r for r in B2.rec.read_csv("training.csv") if r["area"] == "V1" and r.get("row_kind") != "before_training"]
        ok = same_q and same_P and r1["status"] == "interrupted_test" and len(rows) == 1 and r2["status"] in STAGE_TERMINAL
        return {"status": PASSED if ok else FAILED, "q_identical": same_q, "prototypes_identical": same_P, "first": r1["status"],
                "after_resume": r2["status"], "training_rows": len(rows)}

    def st08() -> dict[str, Any]:
        C = new_session()
        C.stage["V1"]["max_epochs"] = 1
        q0 = C.model.store.q_numpy()
        C.fault = "interrupt_after_apply"
        r1 = StageTrainer(C).run()
        q1 = C.model.store.q_numpy()
        rolled = bool(np.array_equal(q0, q1)) and C.stage["V1"]["next_batch"] == 0
        C2 = Session.load(C.rec.dir, be, say=quiet)
        r2 = StageTrainer(C2).run()
        parts = sorted(C2.rec.p("correction_samples").glob("V1_e000_part*.npz"))
        ids = [x for p in parts for x in load_npz(p)["sample_id"].tolist()]
        no_dup = len(ids) == len(set(ids))
        ok = r1["status"] == "interrupted_manual" and rolled and r2["status"] in STAGE_TERMINAL and no_dup
        return {"status": PASSED if ok else FAILED, "first": r1["status"], "q_rolled_back_and_batch_not_committed": rolled,
                "after_resume": r2["status"], "correction_samples": len(ids), "no_duplicate_correction_samples": no_dup}

    def st09() -> dict[str, Any]:
        s = main["s"]
        out = {}
        ok = True
        for _ in range(3):
            nxt, specs, summ, info = s.add_next_area_prepare()
            s.add_next_area_commit(nxt, specs, summ, info, user_choice="self_test")
            s.stage[nxt]["max_epochs"] = 1
            res = StageTrainer(s).run()
            fc = s.frozen_check()
            ok &= all(all(v.values()) for v in fc.values()) and res["status"] in STAGE_TERMINAL
            out[nxt] = {"status": res["status"], "frozen_check": fc, "h_dim": s.model.specs[f"{nxt}/L3"].n}
        r = Diagnostics(s).run_basic()
        ok &= r["status"] == PASSED
        out["basic_checks_at_IT"] = {k: v.get("status") for k, v in r["checks"].items()}
        out["errors"] = {k: v.get("error") for k, v in r["checks"].items() if v.get("error")}
        return {"status": PASSED if ok else FAILED, **out}

    def st10() -> dict[str, Any]:
        s = main["s"]
        blank = st_dir / "blank.png"
        Image.fromarray(np.full((40, 40, 3), 128, np.uint8)).save(blank)
        r1 = run_inference(s, data_dir / "circle")
        r2 = run_inference(s, blank)
        ok = len(r1["rows"]) == len(list_images(data_dir / "circle", cfg["data"]["extensions"])) and r2["rows"][0]["status"] == "unknown_zero_vector"
        return {"status": PASSED if ok else FAILED, "n_rows": len(r1["rows"]), "blank": r2["rows"][0]["status"],
                "predictions": [x.get("prediction") for x in r1["rows"]]}

    def st11() -> dict[str, Any]:
        r = Diagnostics(main["s"]).role_responses()
        return {"status": MEASURED, "V1_L2_orientation_match": r["V1_orientation_lines"]["V1/L2"]["preferred_matches_stimulus_fraction"],
                "V1_L3_orientation_match": r["V1_orientation_lines"]["V1/L3"]["preferred_matches_stimulus_fraction"],
                "constant": r["constant_images"], "V2": r.get("V2_roles"), "V4": r.get("V4_roles"), "translation": r["translation"]}

    def st12() -> dict[str, Any]:
        d = Diagnostics(main["s"])
        a = d.perturbation("l2_rows_within_position", 0.5)
        b = d.perturbation("l4_spatial", 0.5, area="V1")
        ok = a["structure_unchanged_after"] and b["structure_unchanged_after"]
        return {"status": PASSED if ok else FAILED, "l2_rows": a["dev_accuracy"], "l4_spatial_V1": b["dev_accuracy"]}

    def st13() -> dict[str, Any]:
        s = main["s"]
        n0 = s.counters["test_evaluations"]
        Diagnostics(s).test_evaluation()
        s.save_checkpoint("selftest_final", keep=True)
        figs = save_all_figures(s)
        p = build_result_summary(s)
        ok = s.counters["test_evaluations"] == n0 + 1 and p.is_file()
        for f in ("config.json", "environment.json", "assumptions.md", "split_manifest.json", "class_to_index.json", "architecture.json",
                  "training.csv", "prototype_metrics.csv", "diagnostics.json", "execution.log", "errors.log", "result_summary.md"):
            ok &= s.rec.p(f).is_file()
        return {"status": PASSED if ok else FAILED, "figures": len(figs), "matplotlib": _MPL_STATE["plt"] is not None,
                "summary": str(p), "corr_files": len(list(s.rec.p("correction_samples").glob("*.npz")))}

    def st14() -> dict[str, Any]:
        s = main["s"]
        L = Session.load(s.rec.dir, be, say=quiet)
        ok = (np.array_equal(L.model.store.q_numpy(), s.model.store.q_numpy()) and L.model.active == "IT"
              and all(L.protos[a].sha() == s.protos[a].sha() for a in s.protos) and L.frozen.keys() == s.frozen.keys())
        return {"status": PASSED if ok else FAILED, "areas": L.model.areas, "active": L.model.active}

    def st15() -> dict[str, Any]:
        cases = {'  "C:\\Users\\홍 길동\\out"  ': "C:\\Users\\홍 길동\\out", "'D:\\a b'": "D:\\a b", "\u202aE:\\x\\y": "E:\\x\\y"}
        ok = all(str(clean_user_path(k)) == str(Path(v)) for k, v in cases.items())
        a = unique_run_dir(st_dir / "u")
        b = unique_run_dir(st_dir / "u")
        ok &= a != b and a.is_dir() and b.is_dir()
        return {"status": PASSED if ok else FAILED, "cases": {k: str(clean_user_path(k)) for k in cases}}

    def train_upto(sess: Session, last: str, epochs: int = 1) -> None:
        if sess.model is None or not sess.model.areas:
            area, specs, summ = sess.plan_next()
            sess.build_area(area, specs, summ)
        while True:
            sess.stage[sess.model.active]["max_epochs"] = epochs
            StageTrainer(sess).run()
            if sess.model.active == last:
                return
            nxt, specs, summ, info = sess.add_next_area_prepare()
            sess.add_next_area_commit(nxt, specs, summ, info, user_choice="self_test")

    def tree_sha(d: Path) -> str:
        h = hashlib.sha256()
        for f in sorted(x for x in Path(d).rglob("*") if x.is_file()):
            h.update(f.relative_to(d).as_posix().encode())
            h.update(sha256_file(f).encode())
        return h.hexdigest()

    def v1_module() -> Any | None:
        p = Path(cfg["routing"]["legacy_reference_path"]) if cfg["routing"]["legacy_reference_path"] else Path(__file__).with_name("cortex_local_learning_v1.py")
        if not p.is_file():
            return None
        name = f"cortex_v1_selftest_{os.getpid()}"
        spec = importlib.util.spec_from_file_location(name, p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    def st16() -> dict[str, Any]:
        r = RoutingDiagnostics(main["s"]).implementation_checks()
        st = {k: v.get("status") for k, v in r["checks"].items()}
        return {"status": r["status"], "checks": st, "errors": {k: v.get("error") for k, v in r["checks"].items() if v.get("error")},
                "not_run": {k: v.get("reason_ko") for k, v in r["checks"].items() if v.get("status") == NOT_RUN}}

    def st17() -> dict[str, Any]:
        rd = RoutingDiagnostics(main["s"])
        a, b = rd.role_diagnostics(), rd.intervention_diagnostics()
        ok = a["status"] == MEASURED and b["status"] == MEASURED and b["restored"]
        return {"status": MEASURED if ok else FAILED, "role_ratios": a.get("ratios_run_mode"), "figure": a.get("figure"),
                "intervention": {k: {"dev_correct": v["dev_correct"], "selection_changed_vs_normal_frac": v["selection_changed_vs_normal_frac"]}
                                 for k, v in (b.get("conditions") or {}).items()},
                "note_ko": "극소 합성 자료의 측정값이다 (성능 검사가 아니다)."}

    def st18() -> dict[str, Any]:
        c_leg = validate_config(deep_merge(SELFTEST_CONFIG, {"routing": {"mode": "legacy"}}))
        L = Session(c_leg, be, say=quiet)
        L.set_paths(data_dir, runs)
        L.scan_and_split()
        train_upto(L, "V2")
        L.save_checkpoint("selftest_legacy_v2", keep=True)
        src_sha = tree_sha(L.rec.dir)
        N = Session.branch(L.rec.dir, be, mode="v1_guided", say=quiet)
        same_q = all(np.array_equal(N.model.store.q_numpy()[N.model.specs[f"{a}/{l}"].start:N.model.specs[f"{a}/{l}"].end],
                                    L.model.store.q_numpy()[L.model.specs[f"{a}/{l}"].start:L.model.specs[f"{a}/{l}"].end])
                     for a in ("V1", "V2") for l in LAYER_ORDER)
        same_p = all(N.protos[a].sha() == L.protos[a].sha() for a in ("V1", "V2"))
        Lp = L.plan_next()[1]                                        # 같은 원본에서 legacy V4 를 만들면 나올 구조 (비교용, 추가하지 않음)
        leg_v4 = Lp[1].hash()
        info = read_json(N.rec.p("run_info.json")).get("branched_from") or {}
        fc = N.frozen_check()
        ok = (same_q and same_p and N.model.active == "V4" and N.cfg["routing"]["mode"] == "v1_guided" and N.model.routing_runtime() is not None
              and N.model.specs["V4/L2"].hash() != leg_v4 and info.get("new_routing_mode") == "v1_guided" and all(all(v.values()) for v in fc.values())
              and tree_sha(L.rec.dir) == src_sha and N.protos["V4"].version >= 1 and N.rec.p("routing/routing_candidates.npz").is_file()
              and N.stage["V4"].get("start_eval") is not None)
        N.stage["V4"]["max_epochs"] = 1
        r = StageTrainer(N).run()
        ok &= r["status"] in STAGE_TERMINAL
        main["branch"] = N
        main["legacy_run"] = L
        return {"status": PASSED if ok else FAILED, "q_V1V2_identical": same_q, "prototypes_V1V2_identical": same_p,
                "V4_L2_hash_differs_from_legacy": N.model.specs["V4/L2"].hash() != leg_v4, "source_run_unchanged": tree_sha(L.rec.dir) == src_sha,
                "frozen_check": fc, "branched_from": info.get("source_run"), "V4_train": r["status"]}

    def st22() -> dict[str, Any]:
        L = main.get("legacy_run")
        if L is None:
            return {"status": NOT_RUN, "reason_ko": "ST18 의 legacy 실행이 없다."}
        G = Session.branch(L.rec.dir, be, mode="geometry_only", say=quiet)
        G.stage["V4"]["max_epochs"] = 1
        StageTrainer(G).run()
        rd = RoutingDiagnostics(G)
        r = rd.implementation_checks()
        iv = rd.intervention_diagnostics()
        st = {k: v.get("status") for k, v in r["checks"].items()}
        k_geo = G.model.forward(G.loader.batch(G.data.items("dev")[:3], "dev")[0], routing_record=True)["V4/L2"]["routing"]["k"]
        ok = r["status"] == PASSED and int(np.abs(be.to_np(k_geo)).sum()) == 0 and iv["status"] == MEASURED
        return {"status": PASSED if ok else FAILED, "checks": st, "errors": {k: v.get("error") for k, v in r["checks"].items() if v.get("error")},
                "geometry_only_run_selects_first_candidate": int(np.abs(be.to_np(k_geo)).sum()) == 0,
                "intervention_selection_changed": {k: v["selection_changed_vs_normal_frac"] for k, v in iv.get("conditions", {}).items()}}

    def st19() -> dict[str, Any]:
        mod = v1_module()
        if mod is None:
            return {"status": NOT_RUN, "reason_ko": "원본 v1 파일 (cortex_local_learning_v1.py) 이 같은 폴더에 없다 (routing.legacy_reference_path 로 지정)."}
        c1 = mod.validate_config(SELFTEST_CONFIG)
        b1 = mod.DeviceManager.create("numpy", "cpu", c1)
        S1 = mod.Session(c1, b1, say=quiet)
        S1.set_paths(data_dir, runs)
        S1.scan_and_split()
        a_, sp_, su_ = S1.plan_next()
        S1.build_area(a_, sp_, su_)
        S1.stage["V1"]["max_epochs"] = 1
        mod.StageTrainer(S1).run()
        nx, sp_, su_, inf_ = S1.add_next_area_prepare()
        S1.add_next_area_commit(nx, sp_, su_, inf_, user_choice="self_test")
        S1.save_checkpoint("v1_program_run", keep=True)
        src_sha = tree_sha(S1.rec.dir)
        Lv = Session.load(S1.rec.dir, be, say=quiet, read_only=True)
        same_q = bool(np.array_equal(Lv.model.store.q_numpy(), S1.model.store.q_numpy()))
        same_hash = dict(Lv.model.struct_hash) == dict(S1.model.struct_hash)
        it = Lv.data.items("dev")[:3]
        x1, _ = S1.loader.batch(it, "dev")
        x2, _ = Lv.loader.batch(it, "dev")
        o1 = np.asarray(S1.model.forward(x1)["V2/L3"]["out"])
        o2 = be.to_np(Lv.model.forward(x2)["V2/L3"]["out"])
        diff = float(np.abs(o1 - o2).max())
        N = Session.branch(S1.rec.dir, be, mode="v1_guided", say=quiet)
        ok = (Lv.cfg["routing"]["mode"] == "legacy" and same_q and same_hash and diff <= (1e-6 if be.kind == "numpy" else 1e-4)
              and "-R" not in Lv.model_version() and N.model.routing_runtime() is not None and tree_sha(S1.rec.dir) == src_sha)
        return {"status": PASSED if ok else FAILED, "loaded_mode": Lv.cfg["routing"]["mode"], "q_identical": same_q, "structure_hash_identical": same_hash,
                "max_abs_V2L3_diff": diff, "model_version": Lv.model_version(), "branch_to_v1_guided_V4": N.model.routing_runtime() is not None,
                "v1_run_unchanged": tree_sha(S1.rec.dir) == src_sha, "v1_file": str(Path(mod.__file__))}

    def st20() -> dict[str, Any]:
        A = new_session()
        A.stage["V1"]["max_epochs"] = 1
        StageTrainer(A).run()
        B = new_session()
        B.stage["V1"]["max_epochs"] = 1
        g0 = be.max_gather
        B.fault = "oom_once"
        try:
            r = StageTrainer(B).run()
        finally:
            g1 = be.max_gather
            be.max_gather = g0
        ev = [json.loads(x) for x in B.rec.p("oom_events.jsonl").read_text(encoding="utf-8").splitlines()] if B.rec.p("oom_events.jsonl").is_file() else []
        same = bool(np.array_equal(A.model.store.q_numpy(), B.model.store.q_numpy()))
        ok = r["status"] in STAGE_TERMINAL and len(ev) == 1 and g1 < g0 and same and be.kind == B.be.kind
        return {"status": PASSED if ok else FAILED, "oom_events": len(ev), "max_gather_before": g0, "max_gather_after_retry": g1,
                "q_identical_to_run_without_fault": same, "backend_unchanged": B.be.kind, "result": r["status"],
                "note_ko": "주입한 메모리 부족 1 회 → 같은 배치를 줄인 gather 조각으로 다시 계산 (CPU 로 바꾸지 않음)."}

    def st21() -> dict[str, Any]:
        s = main["s"]
        need = ["effective_config.json", "routing/routing_candidates.npz", "routing/routing_static_summary.json", "routing/routing_summary.csv",
                "routing/missing_by_ring_orientation.csv"]
        have = {f: s.rec.p(f).is_file() for f in need}
        samples = sorted(s.rec.p("routing").glob("routing_samples_V4_*.npz"))
        rows = s.rec.read_csv("routing/routing_summary.csv")
        ctx = sorted({r_["context"] for r_ in rows})
        before = [r_ for r_ in s.rec.read_csv("training.csv") if r_.get("row_kind") == "before_training"]
        z = load_npz(samples[0]) if samples else {}
        keys_ok = all(k in z for k in ("sample_id", "selected_slot", "selected_Q", "guide_score", "no_guidance", "A", "B", "u")) if samples else False
        summ = s.rec.p("result_summary.md").read_text(encoding="utf-8") if s.rec.p("result_summary.md").is_file() else ""
        ok = (all(have.values()) and bool(samples) and keys_ok and {"dev_before_training", "train_epoch", "dev_eval"} <= set(ctx)
              and len(before) == len(s.model.areas) and "영구 재배선이나 인간 피질의 재현을 입증한 것은 아니다" in summ)
        return {"status": PASSED if ok else FAILED, "files": have, "routing_sample_parts": len(samples), "sample_keys": sorted(z) if samples else [],
                "summary_contexts": ctx, "before_training_rows": len(before)}

    say(f"[자체 검사] {PROGRAM} v{VERSION}, 뒷단 {be.kind}/{be.device}, 폴더 {st_dir}")
    for name, fn in (("ST01_data_scan_split", st01), ("ST02_preprocessing_dc_infoloss", st02), ("ST03_build_V1_structure", st03),
                     ("ST04_inspect_neuron_3x3", st04), ("ST05_basic_checks_V1", st05), ("ST06_train_V1", st06),
                     ("ST07_resume_equivalence", st07), ("ST08_interrupt_rollback", st08), ("ST09_stages_V2_V4_IT", st09),
                     ("ST10_inference", st10), ("ST11_role_responses", st11), ("ST12_perturbation", st12),
                     ("ST13_test_eval_records_figures", st13), ("ST14_reload_final", st14), ("ST15_path_handling", st15),
                     ("ST16_routing_implementation_checks", st16), ("ST17_routing_role_and_intervention", st17),
                     ("ST18_branch_legacy_run_to_v1_guided", st18), ("ST19_v1_program_checkpoint_as_legacy", st19),
                     ("ST20_oom_retry_same_result", st20), ("ST21_routing_records", st21), ("ST22_geometry_only_run_checks", st22)):
        if name.startswith(("ST02", "ST03", "ST04", "ST05", "ST06", "ST09", "ST10", "ST11", "ST12", "ST13", "ST14", "ST16", "ST17", "ST21")) and "s" not in main:
            results[name] = {"status": NOT_RUN, "reason_ko": "ST01 실패로 세션 없음"}
            continue
        check(name, fn)
    n_fail = sum(1 for v in results.values() if v.get("status") == FAILED)
    summary = {"utc": utc_now(), "program_version": VERSION, "backend": be.describe(), "seconds": round(time.perf_counter() - t_start, 2),
               "n_failed": n_fail, "results": results, "config": cfg,
               "note_ko": "극소 인공 입력에 대한 구현 기능 검사다. 학습 성능·분류 개선은 통과 조건이 아니며 측정하지 않는다."}
    write_json(st_dir / "selftest_results.json", summary)
    say(f"[자체 검사] 실패 {n_fail} / {len(results)} — 결과 {st_dir / 'selftest_results.json'}")
    for s_ in [main.get("s")]:
        if s_ is not None and s_.rec is not None:
            s_.rec.close()
    return 0 if n_fail == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    multiprocessing.freeze_support()
    for stream in (sys.stdout, sys.stderr):              # Windows 콘솔 코드 페이지에 없는 기호가 있어도 멈추지 않게
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    if args.self_test:
        cfg = validate_config(SELFTEST_CONFIG)
        if args.backend == "numpy" or (args.backend == "auto" and torch is None):
            if torch is None and args.backend == "auto":
                print(f"  PyTorch 없음 ({_TORCH_ERROR}) — 자체 검사는 NumPy 경로로 실행한다 (GPU 경로는 검사하지 않음).")
            be = DeviceManager.create("numpy", "cpu", cfg)
        else:
            try:
                be = DeviceManager.create("torch", args.device, cfg)
            except RuntimeError as exc:
                print(f"[오류] {exc}")
                return 1
        return run_self_test(clean_user_path(args.output_dir) if args.output_dir else None, be)
    actions = any([args.train_stage, args.infer, args.diagnose, args.test_eval, args.role_test, args.branch_from, args.routing_check,
                   args.routing_diag])
    if args.menu or not actions:
        MenuApp(args).loop()
        return 0
    try:
        return run_cli(args)
    except (RuntimeError, CompatibilityError, DataError, ConfigError, FileNotFoundError, StructureError) as exc:
        print(f"[오류] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
