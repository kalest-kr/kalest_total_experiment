# -*- coding: utf-8 -*-
"""cortex_local_learning_v5_all_region_ratio_routing.py — 모든 영역 (V1·V2·V4·IT) 의 입력 비율 기반 목적지 선택과 국소 대표 벡터 학습 (v5.0.0).

[기준 파일 — 작업 전에 확인한 것]
  cortex_local_learning_v4_v1_destination_routing.py 내부 버전 4.0.0, SHA256 70d744bdacd2c0180bb74884442fd692348decdf619bca681b259ff82d21792c (명세와 일치).
  이 파일은 4.0.0 의 데이터·전처리·연결 구조·L6·기록 기반을 가져와 고쳐 만들었다. 실행에 다른 프로젝트 소스 파일은 필요 없다.
  기존 라우팅 모드 (legacy·geometry_only·v1_guided·expanded_pool·band_gated·v1_destination_guided) 의 코드는 이 파일에 넣지 않았다
  (새 실험은 all_region_ratio_codebook 하나만 학습한다). 구버전 run 의 checkpoint 는 이어 학습으로 읽지 않는다 (분할 목록만 재사용 가능).

[이번 실험 — 무엇을 하는가]  (경로 선택 기준의 국소 경쟁학습이다. 기존 시냅스 가중치 자체의 학습이 아니다)
  1. 각 영역 (V1 → V2 → V4 → IT) 의 L4 는 하위의 실제 출력을 기존 고정 수렴 그대로 받는다 (V1 L4 = LGN 일대일 중계).
  2. 선택 집단 g = 영역 + 수신 공간 위치 (IT 는 공간 범위) + 역할 계열. 집단마다 실제 L4 원천 목록 source_ids[g] 을 위치·채널 순서로 고정한다
     (값 크기로 정렬하지 않는다). V1 은 수용장 안 LGN ON/OFF 신호 여러 개를 한 입력 패킷으로 묶는다 (단일 값 [1] 퇴화 방지).
  3. x_g = 그 순서로 모은 이번 이미지의 L4 실제 출력. S_g = Σ x_g ≤ routing_min_total 이면 선택 없음·출력 0·갱신 0.
     그 밖에는 p = x/S, x̂ = p/‖p‖ (방향). max/sum 은 우세도 진단값으로만 기록한다.
  4. 후보 = 합법적인 완전한 입력 배정 묶음 (목적지 L2 하나 + A/B 슬롯 원천과 기저 가중치). 곡선 B 원천 후보가 여러 개면 목적지가 같아도
     후보 (route_id) 가 다르다. 대표 벡터 c[g,k] 는 같은 x_g 좌표의 비음수 단위 벡터이며 후보의 구조적 입력 지원 밖은 0 이다.
  5. score = x̂·c, 무효 = −∞, k* = argmax (동점 = 작은 route_id), 동점률·1·2위 차를 기록한다. 점수는 확률이 아니다.
  6. 실제 전달 = 기저 가중치 × 실제 L4 출력 × 선택 마스크. 점수·대표 벡터를 곱하지 않고 S 를 슬롯에 복제하지 않는다.
     선택된 L2 는 기존 역할 연산 (V1 부호 있는 Gabor 합, V2 윤곽·끝점 (억제 −w), 교차·곡선·직선·결합 min(A,B), 선 끝 0.5·fwd+0.5·bwd) 을 한다.
     선택되지 않은 같은 집단 L2 는 u = s = 0. 선택됐지만 역할 출력이 0 이면 selected_but_zero_output 으로 기록 (몰래 재선택하지 않음).
  7. L2 → L3 (V1 위상·극성 평균, V2 일대일, V4 3×3 이웃 평균, IT 일대일) 와 L3 → 다음 L4 고정 수렴은 그대로다 (원핫 병목으로 바꾸지 않음).
     s = max(0, u − θ), CONF = q·s, L6 g = clip(1 + κ(r − 1), g_min, g_max), 출력 = g·CONF. q = 1, θ = 0 고정 (학습하지 않음).
  8. 학습: 활성 영역의 c[g,k] 만. 배치 전체를 학습 전 코드북으로 순방향 → 집단 g 에서 k 가 이긴 유효 입력의 x̂ 평균 v → 지원 마스크 →
     ‖v‖ > ε 이면 c ← normalize((1 − η)c + η·v/‖v‖) (η = 0.05, 모든 영역 같음). 미선택 후보는 그대로. 라벨·dev·test·상위 손실 미분 미사용.
  9. 초기화: train 에서만 min(1024, n_train) 장 (고정 seed, 라벨 미사용). 후보마다 고정 역할 연산이 양으로 반응하는 비영 패킷 중
     역할 선택도 u/S 가 가장 큰 (같은 집단에서 아직 안 쓴, 방향이 겹치지 않는) 패킷의 x̂ 를 지원 마스크로 잘라 정규화. 없으면 지원 질량 최대 패킷
     (따로 기록), 그것도 없으면 초기화 실패 (경쟁에서 제외, 0 벡터를 코사인 비교하지 않음). 이 초기화도 데이터로부터의 학습 과정으로 기록한다.
  10. 분류 판독기: 평가 때마다 모델을 고정하고 train 출력의 클래스 평균 μ_y → P_y = μ_y/‖μ_y‖, 예측 = argmax cos(h, P_y).
      출력 0 또는 클래스 평균 0 은 분류 불능 (정답으로 세지 않음, 비율 따로). 코드북과 P_y 는 다른 이름·저장 공간·차원이다.
  11. 단계 학습: V1 → V2 → V4 → IT. 에폭 0 = 초기화 코드북 + train 판독기 (갱신 전 기준, '무학습 모델' 이 아니다). 매 에폭 끝에 같은
      코드북 스냅샷으로 train 판독기 재계산 → train·dev 평가 (dev 는 갱신 없음). 선택 = dev 정확도 최고 (동률이면 이른 에폭), 최대 20 에폭,
      patience 5. 끝나면 선택 에폭의 코드북·판독기로 되돌리고 다음 영역은 그 상태에서 시작한다. test 는 IT 선택 뒤 사용자가 최종 평가할 때만.

[설치]
  Python 3.10 이상.  필수: numpy, Pillow.  GPU 계산: PyTorch (사용자 PC 에 설치된 것).  선택: matplotlib (그림).
    pip install numpy pillow matplotlib
    PyTorch 는 https://pytorch.org/get-started/locally/ 에서 자기 PC 의 드라이버·CUDA 에 맞는 명령으로 설치한다 (이 파일은 CUDA 버전을 정하지 않는다).
  프로그램은 패키지를 자동 설치하지 않는다. MNIST 공식 파일 내려받기 (메뉴 2, 허용한 경우) 외에는 외부 서버에 접속하지 않는다.
  PyTorch 가 없으면 시작할 때 알리고, 사용자가 고를 때만 같은 수식의 NumPy CPU 경로 (느림) 로 계속한다 (--backend numpy).

[실행 — Windows / PyCharm: 이 파일을 열고 Run → 한국어 번호 메뉴]
  python cortex_local_learning_v5_all_region_ratio_routing.py
    1 (데이터 종류·MNIST 캐시 폴더·저장 폴더) → 2 (내려받기·검사·분할, 또는 기존 run 분할 재사용) → 3 (V1 생성: 구조·후보·코드북 메모리 표시 →
    train 으로 코드북 초기화 → 에폭 0 기준 평가) → 7 → 1 (사전 검사 16) → 4 (V1 학습) → 5 (V2 추가) → 4 → 5 (V4) → 4 → 5 (IT) → 4
    → 7 → 4 (최종 test, 선택된 IT 에서 한 번) → 6 (추론) → 8 (저장·REPORT.md).
  명령줄:
    python cortex_local_learning_v5_all_region_ratio_routing.py --self-test --output-dir "D:\\cortex_out"
    python cortex_local_learning_v5_all_region_ratio_routing.py --prepare-data --data-root "D:\\mnist" --output-dir "D:\\cortex_out"
    python cortex_local_learning_v5_all_region_ratio_routing.py --data-root "D:\\mnist" --output-dir "D:\\cortex_out" --train-stage V1 --precheck
    python cortex_local_learning_v5_all_region_ratio_routing.py --resume "D:\\cortex_out\\run_..." --train-stage IT
    python cortex_local_learning_v5_all_region_ratio_routing.py --resume "D:\\cortex_out\\run_..." --test-eval --infer "D:\\test_images"
    python cortex_local_learning_v5_all_region_ratio_routing.py --reuse-split "D:\\cortex_out\\run_v4..." --data-root "D:\\mnist" --output-dir "D:\\cortex_out" --train-stage V1
    python cortex_local_learning_v5_all_region_ratio_routing.py --resume "D:\\cortex_out\\run_..."      (인자 없이: 그 run 을 불러온 메뉴)
  기타: --device auto|cuda|cpu, --backend auto|torch|numpy, --seed N, --config 설정.json, --epochs N, --role-probe, --yes

[자동 기록 (run 폴더) — REPORT.md 의 숫자는 저장된 JSON/CSV 에서 만든다]
  config.json · environment.json · source.sha256 · split_manifest.json · structure.json · training_metrics.csv · routing_metrics.csv ·
  codebook_metrics.csv · role_probe_metrics.csv · sample_routes.npz · checkpoints/ · REPORT.md
  그 밖: structure/candidates_{영역}.npz (후보 레코드 전체: 무효 후보와 탈락 사유 포함), codebook/init_{영역}.npz·json (초기화 표본 ID·방식),
  codebook/usage_{영역}_eNNN.npz (후보별 선택 횟수), readout/ (에폭별 판독기), routes/ (상세 표본 조각), diagnostics/ (사전 검사·역할 검사·test).

[이 파일을 만든 환경에서 실제 확인한 것 / 확인하지 못한 것]  (자세히: VERIFICATION_NOTES)
  · 확인 (Linux, Python 3.13, NumPy 와 PyTorch CPU; 3.11 은 문법 검사만): 문법·정적 검사, --self-test 17 항목 (실패 0), 작은 합성 이미지에서 사전 검사 16
    (C14 는 GPU 가 없어 NOT_RUN), 합성 이미지로 V1→V2→V4→IT 각 1 에폭 흐름·재개·test·추론·보고서, 한국어 메뉴 재생, 명령줄 오류 안내,
    기본 설정의 구조 통계 (합성 이미지로 구조만 만든 값). 이 수치는 구현 동작 확인이며 성능 근거가 아니다.
  · 확인하지 못함: CUDA GPU (RTX 4070) 실행, MNIST 학습·분류 성능, 실제 메모리·시간, Windows/PyCharm 동작. 정확도·속도 예상치는 적지 않는다.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import dataclasses
import datetime as _dt
import gzip
import hashlib
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
import struct
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
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

VERSION = "5.0.0"
PROGRAM = "cortex_local_learning_v5_all_region_ratio_routing"
CKPT_FORMAT = "cortex_local_learning_v5/checkpoint/1"
CKPT_FORMATS_READABLE = ("cortex_local_learning_v5/checkpoint/1",)
OLD_CKPT_FORMATS = ("cortex_local_learning_v1/checkpoint/1", "cortex_local_learning_v2/checkpoint/1",
                    "cortex_local_learning_v3/checkpoint/1", "cortex_local_learning_v4/checkpoint/1")
MODE = "all_region_ratio_codebook"
BASE_FILE_INFO: dict[str, str] = {
    "file": "cortex_local_learning_v4_v1_destination_routing.py", "version": "4.0.0",
    "sha256": "70d744bdacd2c0180bb74884442fd692348decdf619bca681b259ff82d21792c",
    "verified_ko": "작성 시 첨부 기준 파일의 내부 버전 4.0.0 과 SHA256 이 명세와 일치함을 확인했다 (작업 저장소의 v4 커밋 파일과도 같은 해시).",
    "reused_ko": "데이터 (폴더·MNIST)·전처리 (Retina/LGN·로그 극좌표)·V1/V2/V4/IT 역할 연결 생성기·L6·CSR/ELL·기록·메뉴 골격을 가져왔다. "
                 "V4 곡선 B 후보의 기하 조건 (수용장 안, 진행 방향, 굽힘 쪽, Hermite 경로 유효) 은 4.0.0 의 후보 그래프 규칙을 그대로 쓴다.",
    "removed_ko": "V1→V4 안내 우회 (GuideSampler·RoutingTable·DestTable), 강도 구간 게이트 (BandTable), q 학습 (LocalErrorRouter·L1Relay·"
                  "LocalOutputLearner·클래스 prototype 오차), prototype EMA 는 이 파일에 없다 (새 모드에서 호출할 수 없도록 코드 자체를 넣지 않았다).",
}
AREA_ORDER: tuple[str, ...] = ("V1", "V2", "V4", "IT")
AREA_INDEX = {a: i + 1 for i, a in enumerate(AREA_ORDER)}
LAYER_ORDER: tuple[str, ...] = ("L4", "L2", "L3")
LAYER_NUMBER = {"L4": 4, "L2": 2, "L3": 3}

PASSED, FAILED, MEASURED, NOT_RUN, NOT_APPLICABLE = "PASSED", "FAILED", "MEASURED", "NOT_RUN", "NOT_APPLICABLE"
STAGE_TERMINAL = ("success_criterion_met", "plateau", "budget_exhausted")

VERIFICATION_NOTES: dict[str, list[str]] = {
    "base_file_check": [
        "첨부 기준 파일 cortex_local_learning_v4_v1_destination_routing.py: 내부 버전 4.0.0, SHA256 70d744bd…792c 가 명세와 일치함을 확인 (작업 전).",
    ],
    "verified_in_authoring_environment": [
        "py_compile (Python 3.11, 3.13) 와 pyflakes: 오류 없음",
        "--self-test ST01–ST17 (극소 합성 이미지): NumPy·PyTorch CPU 경로 각각 실패 0 — 16 항목 PASSED, ST11 역할 검사는 판정 없는 MEASURED",
        "사전 검사 16 (작은 합성 구조): C01–C13·C15·C16 PASSED, C14 NOT_RUN (GPU 없음). C14 안의 float64 기준식 대비 비교는 실행",
        "C14 의 비교 코드 경로만: PyTorch CPU 를 GPU 로 간주하게 바꿔 NumPy CPU 와 비교 → 점수 차 ≤ 4.2e-7, 선택 불일치 0. 실제 GPU 수치 검사가 아니다",
        "합성 이미지 (원·사각형·삼각형 26 장) 로 V1→V2→V4→IT 각 1 에폭: 에폭 0 기준 평가, dev 기준 선택 에폭 복원, IT 선택 후 test 1 회, 추론, REPORT.md 생성",
        "명령줄: 새 run --train-stage V1 --precheck (사전 검사가 학습 전에 실행), --resume … --train-stage IT --test-eval --infer, "
        "--reuse-split (모의 v4 run) --prepare-data, --resume 만 주면 그 run 을 불러온 메뉴",
        "오류 안내: 구버전 run --resume → CompatibilityError, 옵션 충돌, seed·데이터 종류 불일치, 앞 단계로 돌아가기, 알 수 없는 설정 키 (q 학습 등)",
        "한국어 메뉴 재생 (입력 파일): 1→2→3→7-1→7-3→4→5→4→5→4→5→4→7-4→7-5→6→8→9→0 오류 없음. 입력이 닫히면 (EOF) 메뉴가 끝남",
        "기본 설정 구조 (합성 이미지로 구조만): V1 집단 512·합법 후보 24,576·코드북 35.95 MB, V2 집단 384·후보 6,912, "
        "V4 집단 128·합법 5,253 (기하 탈락 12,411, 후보 없는 목적지 78), IT 집단 20·후보 146, 전체 뉴런 56,948",
        "기본 V1 코드북 초기화 1,024 패킷 (합성 이미지, NumPy float64): 작성 환경 CPU 에서 약 12 s, 최대 RSS 약 1 GB, 초기화 실패 0",
        "라우터 결과와 같은 파일 안의 float64 기준식 비교: 선택 불일치는 동점 (gap < 1e-5) 에서만",
    ],
    "not_verified": [
        "CUDA GPU (RTX 4070) 실행과 사전 검사 14 의 실제 CPU/GPU 비교, GPU 메모리·시간 — 작성 환경에 GPU 가 없다",
        "MNIST 내려받기·학습·평가, 분류 성능, 이전 버전과의 성능 차이 (요청대로 실행하지 않음, 비교 모델 학습 없음)",
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


class NumericalError(RuntimeError):
    """비유한 값 (NaN/Inf) 또는 음수 L4 입력이 라우팅 패킷에 들어왔다. 조용히 전달하지 않고 해당 계산을 실패 처리한다."""


# ======================================================================
# 1. 설정 (Config) — 모든 기본값을 한 곳에서 관리한다
# ======================================================================
DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 20251011,
    "data": {
        "source": "mnist",                      # 새 실행: mnist / folder
        "mnist_dev_size": 10000,                 # 공식 train 안에서만 추출 (공식 test 는 항상 따로 보존)
        "mnist_download": True,                  # 누락된 공식 gzip IDX 만 받음. 기존 파일은 MD5 검증
        "image_size": 128,                       # 처리 크기 S (정사각). 긴 변을 S 에 맞추고 짧은 변은 가장자리 복제로 채운다
        "extensions": [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"],
        "split_ratios": {"train": 0.70, "dev": 0.15, "test": 0.15},
        "group_suffix_regex": r"(?:[_\-\s]?(?:aug|copy|dup|flip|rot)\d*|\s?\(\d+\))+$",
        "transparent_background": 1.0,           # 투명 픽셀을 합성할 배경 밝기 (0..1)
        "cache_preprocessed": True,              # LGN 표본 벡터를 RAM (CPU) 에 캐시 (GPU 에 데이터셋 전체를 올리지 않는다)
        "cache_max_mb": 1024,
    },
    "retina": {
        "sigma_center_px": 1.0, "sigma_surround_px": 3.0, "truncate_sigma": 3.0, "gain": 1.0,
    },
    "logpolar": {
        "n_r": 32, "n_phi": 64, "r0_px": 2.0, "rmax_margin_px": 2.0, "phi_offset_bins": 0.5, "blur_beta": 0.5,
        "blur_levels_px": [0.0, 1.0, 2.0, 4.0, 8.0],
    },
    "neuron": {"threshold": 0.0, "q": 1.0},     # v5: q·임계값 고정 (학습하지 않는다). 기준 파일 기본값 그대로
    "areas": {
        "V1": {"n_r": 16, "n_phi": 32,
               "orientations_deg": [15.0 * k for k in range(12)],     # 0..165 도, 원영상 좌표 기준 (반시계)
               "phases_rad": [0.0, -math.pi / 2.0],
               "polarities": [1, -1],
               "color_channels": ["lum"],
               "gabor_sigma_factor": 1.0, "gabor_sigma_min_px": 1.5,
               "gabor_wavelength_factor": 2.0, "rf_radius_sigma": 2.0},
        "V2": {"n_r": 8, "n_phi": 16, "offset_factor": 1.0, "endstop_inhibition": 1.0, "cross_diffs_deg": [90.0, 60.0]},
        "V4": {"n_r": 4, "n_phi": 8, "offset_factor": 1.0, "curvature_deg": [30.0], "l3_grid_radius": 1},
        "IT": {"region_r": 2, "region_phi": 4},
    },
    "l6": {"enabled": True, "layers": ["L2", "L3"], "epsilon": 1e-6, "kappa": 0.25, "g_min": 0.5, "g_max": 2.0,
           "neighbor_grid_radius": 1, "include_self": False},
    # 모든 영역의 입력 비율 라우팅 (v5 의 유일한 모드). 값은 출발 설정·수치 보호값이며 생물학적 값이나 최적값이라고 주장하지 않는다.
    "routing": {
        "mode": MODE,
        "routing_min_total": 1e-8,               # S_g ≤ 이 값이면 선택 없음·출력 0·갱신 0 (수치 보호값)
        "norm_epsilon": 1e-8,                    # 정규화 분모 하한 (수치 보호값)
        "eta_codebook": 0.05,                    # 모든 영역 같은 국소 학습률 (출발 설정)
        "init_samples": 1024,                    # 코드북 초기화 표본 수 = min(이 값, train 크기) (train 만, 라벨 미사용)
        "init_seed_offset": 5050,                # 초기화 표본 선택 RNG = default_rng([seed, offset, 영역번호]) (학습 순서 RNG 와 별도)
        "init_duplicate_cos_tol": 1e-6,          # cos ≥ 1 − tol 이면 같은 방향으로 본다 (중복 초기화 기록)
        "init_alternatives": 8,                  # 방향이 겹칠 때 다음 순위 패킷을 몇 개까지 볼지
        "near_tie_gap": 1e-6,                    # 기록용: 1·2 위 차가 이보다 작으면 '거의 동점'
        "curve_path_points": 9,                  # V4 곡선 B 후보의 Hermite 경로 검사 점 수 (기준 파일 값)
        "bucket_padding_max": 1.25,              # 집단 묶음의 패딩 상한 (계산 분할용, 후보 삭제 없음)
        "bucket_max_elements": 1 << 24,          # 묶음 하나의 G×K×D 상한 (넘으면 묶음을 나눈다)
    },
    "learning": {"batch_size": 16, "max_epochs": 20, "patience": 5, "min_delta": 0.0, "success_dev_accuracy": None, "eps_norm": 1e-12},
    "records": {"checkpoint_every_batches": 500, "keep_periodic_checkpoints": 3, "store_train_h_max_mb": 1024,
                "train_accuracy_each_epoch": True, "detail_samples_per_epoch": 4, "detail_samples_max_per_stage": 32,
                "diagnostic_samples": 4, "role_probe_after_init": True, "role_probe_after_stage": True},
    "compute": {"dtype": "float32", "max_gather_elements": 1 << 25, "deterministic": True, "gpu_timing_sync": True,
                "oom_retry_halvings": 3},
}
ROUTING_RECORD_ONLY_KEYS = ("near_tie_gap",)
RESUME_MUTABLE_LEARNING = ("max_epochs", "patience", "min_delta", "success_dev_accuracy")


def validate_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    c = deep_merge(DEFAULT_CONFIG, cfg or {})
    unknown = sorted(set(c) - set(DEFAULT_CONFIG))
    if unknown:
        raise ConfigError(f"알 수 없는 설정 키: {unknown} (v5 에는 band·destination 설정이 없다)")
    for sec in ("data", "retina", "logpolar", "neuron", "l6", "routing", "learning", "records", "compute"):
        extra = sorted(set(c[sec]) - set(DEFAULT_CONFIG[sec]))
        if extra:
            raise ConfigError(f"{sec} 에 알 수 없는 키: {extra} (v5 는 q 학습·안내 라우팅·구간 게이트 설정을 받지 않는다)")

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
    if d["source"] not in ("mnist", "folder"):
        raise ConfigError("data.source 는 mnist 또는 folder 다.")
    pos_int(d["mnist_dev_size"], "data.mnist_dev_size", 10)
    if int(d["mnist_dev_size"]) > 59990:
        raise ConfigError("data.mnist_dev_size 는 59990 이하다 (클래스별 train/dev 확보).")
    if not isinstance(d["mnist_download"], bool):
        raise ConfigError("data.mnist_download 는 true 또는 false 다.")
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
    num(ne["threshold"], "neuron.threshold")
    num(ne["q"], "neuron.q", 0.0, lo_open=True)
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
    ro = c["routing"]
    if ro["mode"] != MODE:
        raise ConfigError(f"routing.mode 는 이 버전에서 {MODE} 만 지원한다 (기존 모드 비교 학습을 하지 않는다).")
    num(ro["routing_min_total"], "routing.routing_min_total", 0.0)
    num(ro["norm_epsilon"], "routing.norm_epsilon", 0.0, lo_open=True)
    num(ro["eta_codebook"], "routing.eta_codebook", 0.0, 1.0, lo_open=True)
    pos_int(ro["init_samples"], "routing.init_samples")
    pos_int(ro["init_seed_offset"], "routing.init_seed_offset", 0)
    num(ro["init_duplicate_cos_tol"], "routing.init_duplicate_cos_tol", 0.0, 1.0)
    pos_int(ro["init_alternatives"], "routing.init_alternatives")
    num(ro["near_tie_gap"], "routing.near_tie_gap", 0.0)
    pos_int(ro["curve_path_points"], "routing.curve_path_points", 3)
    num(ro["bucket_padding_max"], "routing.bucket_padding_max", 1.0)
    pos_int(ro["bucket_max_elements"], "routing.bucket_max_elements", 1024)
    le = c["learning"]
    pos_int(le["batch_size"], "learning.batch_size")
    pos_int(le["max_epochs"], "learning.max_epochs", 0)
    pos_int(le["patience"], "learning.patience")
    num(le["min_delta"], "learning.min_delta", 0.0)
    if le["success_dev_accuracy"] is not None:
        num(le["success_dev_accuracy"], "learning.success_dev_accuracy", 0.0, 1.0)
    num(le["eps_norm"], "learning.eps_norm", 0.0, lo_open=True)
    rc = c["records"]
    for k in ("checkpoint_every_batches", "keep_periodic_checkpoints", "store_train_h_max_mb", "detail_samples_per_epoch",
              "detail_samples_max_per_stage"):
        pos_int(rc[k], f"records.{k}", 0)
    pos_int(rc["diagnostic_samples"], "records.diagnostic_samples", 2)
    for k in ("train_accuracy_each_epoch", "role_probe_after_init", "role_probe_after_stage"):
        if not isinstance(rc[k], bool):
            raise ConfigError(f"records.{k} 는 true 또는 false 다.")
    cp = c["compute"]
    if cp["dtype"] != "float32":
        raise ConfigError("compute.dtype 는 float32 로 시작한다 (혼합 정밀도는 기본으로 쓰지 않는다).")
    pos_int(cp["max_gather_elements"], "compute.max_gather_elements", 1024)
    pos_int(cp["oom_retry_halvings"], "compute.oom_retry_halvings", 0)
    return c


def config_structure_sha(cfg: dict[str, Any]) -> str:
    """구조·순방향에 영향을 주는 설정의 해시 (재개 때 같아야 한다)."""
    part = {"image_size": cfg["data"]["image_size"], "transparent_background": cfg["data"]["transparent_background"],
            "retina": cfg["retina"], "logpolar": cfg["logpolar"], "neuron": cfg["neuron"], "areas": cfg["areas"], "l6": cfg["l6"],
            "dtype": cfg["compute"]["dtype"],
            "routing": {k: v for k, v in cfg["routing"].items() if k not in ROUTING_RECORD_ONLY_KEYS}}
    return sha256_text(dumps(part, indent=None))


def config_learning_sha(cfg: dict[str, Any]) -> str:
    le = {k: v for k, v in cfg["learning"].items() if k not in RESUME_MUTABLE_LEARNING}
    part = {"learning": le, "seed": cfg["seed"], "split": cfg["data"]["split_ratios"], "group_regex": cfg["data"]["group_suffix_regex"],
            "mode": cfg["routing"]["mode"]}
    if cfg["data"]["source"] == "mnist":
        part["dataset"] = {"source": "mnist", "dev_size": cfg["data"]["mnist_dev_size"], "split_policy": "official-test-stratified-train-v1"}
    return sha256_text(dumps(part, indent=None))


def config_split_sha(cfg: dict[str, Any]) -> str:
    """분할 목록 재사용 판단용 (데이터 종류·seed·분할 규칙)."""
    part = {"source": cfg["data"]["source"], "seed": cfg["seed"], "split": cfg["data"]["split_ratios"],
            "group_regex": cfg["data"]["group_suffix_regex"], "dev_size": cfg["data"]["mnist_dev_size"]}
    return sha256_text(dumps(part, indent=None))


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


# 수식 기록 (config.json · structure.json · REPORT.md 에 그대로 남긴다)
FORMULAS: dict[str, str] = {
    "neuron": "u_j = (역할 연산의 입력 합) ;  s_j = max(0, u_j − θ_j) ;  CONF_j = q_j·s_j ;  out_j = g_j·CONF_j  (g = L6, L4 는 1). "
              "v5: θ = 0, q = 1 고정 (학습하지 않음)",
    "role_ops": "V1 Gabor u = Σ_k g_k·(ON_k − OFF_k) (부호 있는 가중치, 극성 −1 은 −g) ;  V2 윤곽 u = 0.5·o(P) + 0.25·o(앞) + 0.25·o(뒤), "
                "끝점 u = o(P) − w_inh·o(P ± D), 교차 u = min(o(P,θ1), o(P,θ2)) ;  V4 곡선 u = min(A, B), 직선 u = min(A, B), 접합 u = cross(P), "
                "선 끝 u = 0.5·fwd + 0.5·bwd ;  IT 전역 u = mean_R, 영역 특징군 u = mean_c, 결합 u = min(mean_G1, mean_G2)",
    "l6": "같은 층 CONF 스냅샷에서 병렬:  r_j = CONF_j/(mean_{k∈N(j)} CONF_k + ε) ;  g_j = clip(1 + κ(r_j − 1), g_min, g_max) ;  "
          "out_j = g_j·CONF_j ;  N(j) = 같은 층 Chebyshev ≤ 1 위치의 모든 채널, 자기 제외 ;  이웃 없음 → g = 1",
    "packet": "집단 g = 영역 + 수신 공간 위치 (IT 는 공간 범위) + 역할 계열.  x_g = L4 실제 출력 [source_ids[g]] (L4 local 주소 = 위치·채널 순서, 값 정렬 없음, "
              "패딩 = 0 열·마스크).  같은 원천이 A·B 슬롯 모두에 쓰여도 x_g 에는 한 번만 넣는다",
    "ratio": "S_g = Σ x_g ;  S_g ≤ routing_min_total → 선택 없음 (출력 0, 갱신 0) ;  그 밖 p = x/S_g ,  x̂ = p / max(‖p‖, norm_epsilon) ;  "
             "우세도 = max(x)/S_g (진단만) ;  NaN/Inf·음수 입력 → 그 집단 계산 실패 (선택 없음, 학습 배치는 오류로 멈춤)",
    "score": "score[g,k] = x̂_g · c[g,k] ;  무효 후보 (불법·패딩·초기화 실패) = −∞ ;  k* = argmax (같은 점수 → 작은 route_id) ;  "
             "기록: 1·2 위 차, 정확한 동점 (차 = 0), 거의 동점 (차 < near_tie_gap). 점수는 확률이 아니다",
    "deliver": "delivered_edge = base_weight × L4 실제 출력 × [k = k*] ;  선택된 L2: u = 역할 연산 (A 합, min2 는 min(A 합, B 합)) ;  "
               "같은 집단의 다른 L2: u = s = 0 ;  점수·대표 벡터·S 를 전달값에 곱하거나 복제하지 않는다",
    "codebook_update": "배치 전체를 같은 학습 전 코드북으로 순방향 → n[g,k] = 유효 승리 수, v = mean(x̂ | k 승리) ⊙ 지원 마스크 ;  "
                       "n > 0 이고 ‖v‖ > ε → c ← normalize((1 − η)·c + η·v/‖v‖) ;  그 밖 c 그대로. 배치마다 한 번 (합이 아니라 평균)",
    "codebook_init": "train 에서 min(init_samples, n_train) 장 (default_rng([seed, offset, 영역번호]), 라벨 미사용) → 후보마다 "
                     "역할 연산 u_k > θ 인 비영 패킷 중 u_k/S 최대 (같은 집단에서 아직 쓰지 않고 방향이 겹치지 않는 것 우선) → c = normalize(x̂ ⊙ 지원) ;  "
                     "없으면 지원 질량 ‖x̂ ⊙ 지원‖² 최대 패킷 (fallback 기록) ;  단위 벡터를 만들 수 없으면 초기화 실패 (경쟁 제외)",
    "readout": "평가 때마다 고정 모델로 μ_y = mean(h | train, label y), P_y = μ_y/‖μ_y‖ ;  예측 = argmax_y cos(h, P_y) ;  "
               "‖h‖ ≤ eps 또는 유효 P 없음 → 분류 불능 (오답으로 셈, 비율 따로)",
    "selection": "에폭 0 = 초기화 코드북 + train 판독기 (갱신 전 기준). 매 에폭 끝 dev 정확도 > best + min_delta 이면 갱신 (동률 → 이른 에폭). "
                 "patience 에폭 연속 개선 없음 → plateau, 최대 에폭 → budget_exhausted, 성공 기준 도달 → success_criterion_met. 끝나면 선택 에폭 상태로 복원",
    "cost": "점수 계산 O(batch × Σ_g K_g D_g), 역할 연산 같은 차수, 코드북 메모리 O(Σ_g K_g D_g). 원핫 전달이어도 후보 비교 비용은 없어지지 않는다",
}

ASSUMPTIONS: dict[str, list[str]] = {
    "설계 범위 (주장하지 않는 것)": [
        "결정론적 rate 모델이다. 시간축·막전위 적분·스파이크·전역 역전파를 넣지 않았다. 층 이름 (L1·L2·L3·L4·L6) 은 생물학적으로 확립된 회로라는 뜻이 아니다.",
        "이번 학습은 입력 패턴을 조직화하는 국소 경쟁학습 (경로 선택 기준의 학습) 이다. 기존 시냅스 가중치 (Gabor·윤곽·결합 가중치) 는 학습하지 않는다.",
        "고정 역할 이름 (곡선·접합 등) 이 실제 생물학적 뉴런 기능을 입증하지 않는다. 역할 수행 여부는 통제 자극 검사로만 측정한다.",
        "상위 손실 미분 없이 계산한다는 사실이 뇌의 학습 기제를 재현했다는 뜻이 아니다.",
    ],
    "집단·패킷 정의 (명세가 정하지 않아 고른 구현)": [
        "V1: 집단 = V1 위치 × Gabor 계열 (색 채널별). 후보 = 그 위치의 방향 12 × 위상 2 × 극성 2 = 48 L2 뉴런. x_g = 수용장 (2σ) 안 LGN 위치의 ON·OFF "
        "L4 값 (LGN 위치 순서 × ON/OFF). 모든 후보의 지원 = 같은 수용장 (Gabor 가중치가 정의된 원천).",
        "V2: 집단 = V2 위치 × {윤곽, 끝점 (앞·뒤 방향 24 후보), 교차}. x_g = 그 계열 후보들이 쓰는 V2 L4 원천의 합집합 (위치·방향 순서).",
        "V4: 집단 = V4 위치 × {곡선, 직선, 접합, 선 끝}. 곡선 후보 = (곡선 뉴런, 유효 B 위치 Q) 완전 배정 묶음 (기준 파일 4.0.0 의 기하 조건). "
        "직선·접합·선 끝은 기존 원천 연결 하나가 후보다.",
        "IT: 집단 = 전체 시야 × 전역 특징 (V4 계열별 4 집단: 곡선·직선·접합·선 끝 채널), 영역 R × 영역 특징군, 영역 R × 특징군 결합. IT 전체 top-1 이 아니다.",
        "후보의 구조적 지원 마스크 = 그 후보의 A·B 슬롯 원천. 지원이 1 차원인 후보 (V4 접합 중계 등) 는 대표 벡터가 단위축으로 고정되어 학습으로 바뀌지 않는다 (자유도 0, 기록).",
        "코드북은 집단끼리 공유하지 않는다 (codebook_id = 집단 안 후보 칸, descriptor_schema_id = 집단 번호).",
    ],
    "계산 가정": [
        "L4 출력은 비음수다 (s = max(0, u − θ), q > 0). 음수가 들어오면 계산 실패로 처리한다.",
        "argmax 는 가장 앞 칸 (가장 작은 route_id) 을 고른다 (NumPy·PyTorch 모두 첫 최댓값). 점수 비교는 float32, 기준 검사는 float64.",
        "결정성: 고정 인덱스 gather·배치 행렬곱·마스크 argmax·비교 gather 만 쓴다. atomic 누적 (index_add·scatter_add) 을 쓰지 않는다. "
        "CUDA 는 TF32 를 끄고 cuBLAS workspace 를 고정한다. 그래도 GPU 축약 순서가 CPU 와 다를 수 있어 거의 동점의 선택은 달라질 수 있다 (검사 14).",
        "판독기의 h = 현재 단계 영역 L3 실제 출력. 동결된 아래 영역도 매 입력 L6·목적지 선택을 다시 계산한다 (동결 = 코드북·판독기 갱신 중단).",
        "데이터 로더는 프로세스 안 순차 로딩 (num_workers=0 과 같음). 전처리 결과 (LGN 표본) 만 RAM 캐시하고 영역 출력은 캐시하지 않는다.",
    ],
    "미구현·채택하지 않은 것 (이번 기본 실험)": [
        "강제 균등 배분, 대표 간 반발력, top-k·soft 선택, dead-code 재초기화, 무작위 목적지 전송 — 경로 독점·죽은 후보는 실패 가능성으로 측정만 한다.",
        "q 학습, 임계값 학습, 클래스 prototype 오차의 L1 배달, V1→V4 안내 우회, 강도 구간 게이트, V4 전용 목적지 게이트, 전역 역전파·STE·유한차분.",
        "여러 학습법의 자동 비교, 학습률 격자 탐색, 비교 모델 학습.",
    ],
    "미검증 가설": [
        "코사인 일치 (점수) 가 높아지면 분류가 좋아진다 — 보장되지 않는다.",
        "코드북 갱신이 일어나면 각 뉴런이 의도한 역할을 학습한다 — 보장되지 않는다 (selected_but_zero_output·역할 검사로 본다).",
        "hard top-1 선택이 분류에 필요한 정보를 보존한다 — 원핫 선택은 정보를 버린다 (L2→L3 병합·영벡터 비율로 본다).",
        "모든 후보를 저장했으므로 모든 후보가 학습된다 — 미사용 후보 비율로 본다.",
    ],
}


def assumptions_markdown() -> str:
    lines = [f"# {PROGRAM} 가정·한계 (v{VERSION})", "",
             "이 문서는 프로그램이 자동으로 쓴다. 뉴런 수·배치·층 역할·학습 규칙은 생물학에서 착안한 공학적 근사이며, "
             "인간 시각 피질의 재현이나 뇌가 아래 수식을 쓴다는 주장이 아니다.", ""]
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

    def exp(self, x: Any) -> Any:
        return np.exp(x) if self.kind == "numpy" else torch.exp(x)

    def isnan(self, x: Any) -> Any:
        return np.isnan(x) if self.kind == "numpy" else torch.isnan(x)

    def isfinite(self, x: Any) -> Any:
        return np.isfinite(x) if self.kind == "numpy" else torch.isfinite(x)

    def as_f32(self, x: Any) -> Any:
        return np.asarray(x, np.float32) if self.kind == "numpy" else x.to(torch.float32)

    def sqrt(self, x: Any) -> Any:
        return np.sqrt(x) if self.kind == "numpy" else torch.sqrt(x)

    def sum_last(self, x: Any) -> Any:
        return x.sum(-1)

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

    # ---------------------------------------------------------------- v5 라우팅 연산 (gather·배치 행렬곱·마스크 argmax, atomic 누적 없음)
    def bmm(self, a: Any, b: Any) -> Any:
        """(G, M, D) @ (G, D, N) → (G, M, N) (집단별 행렬곱)."""
        return np.matmul(a, b) if self.kind == "numpy" else torch.bmm(a, b)

    def permute(self, x: Any, axes: Sequence[int]) -> Any:
        return np.transpose(x, tuple(axes)) if self.kind == "numpy" else x.permute(*axes)

    def gather_last(self, x: Any, idx: Any) -> Any:
        """x (..., K), idx (...) → x[..., idx] (마지막 축에서 칸 하나)."""
        if self.kind == "numpy":
            return np.take_along_axis(x, idx[..., None], -1)[..., 0]
        return torch.gather(x, -1, idx.unsqueeze(-1)).squeeze(-1)

    def onehot_last(self, idx: Any, K: int) -> Any:
        if self.kind == "numpy":
            return (idx[..., None] == np.arange(int(K))).astype(np.float32)
        return (idx.unsqueeze(-1) == torch.arange(int(K), device=self.dev)).to(torch.float32)

    def full(self, shape: Sequence[int], value: float) -> Any:
        shape = tuple(int(s) for s in shape)
        return np.full(shape, value, np.float32) if self.kind == "numpy" else torch.full(shape, float(value), dtype=torch.float32, device=self.dev)

    def full_i64(self, shape: Sequence[int], value: int) -> Any:
        shape = tuple(int(s) for s in shape)
        return np.full(shape, value, np.int64) if self.kind == "numpy" else torch.full(shape, int(value), dtype=torch.int64, device=self.dev)

    def zeros_like(self, x: Any) -> Any:
        return np.zeros_like(x) if self.kind == "numpy" else torch.zeros_like(x)

    def all_last(self, x: Any) -> Any:
        return x.all(-1) if self.kind == "numpy" else x.all(dim=-1)

    def any_last(self, x: Any) -> Any:
        return x.any(-1) if self.kind == "numpy" else x.any(dim=-1)

    def clamp_min(self, x: Any, lo: float) -> Any:
        return np.maximum(x, np.float32(lo)) if self.kind == "numpy" else torch.clamp(x, min=float(lo))

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
        d["reduction_determinism_ko"] = ("순방향·선택·코드북 갱신은 고정 인덱스 gather·배치 행렬곱·마스크 argmax·비교 gather 만 쓴다 "
                                         "(atomic index_add·scatter_add 없음). 행렬곱은 cuBLAS (TF32 끔, workspace 고정).")
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

    SUBDIRS = ("checkpoints", "figures", "readout", "codebook", "routes", "diagnostics", "inference", "structure")
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
    """학습 중 Ctrl+C. 첫 번째: 현재 배치를 마친 안전한 경계에서 멈춤. 두 번째: 즉시 중단 (진행 중 배치의 코드북은 되돌린다).
    critical() 안에서는 즉시 중단을 미뤘다가 끝에서 일으킨다 (코드북 적용·기록·checkpoint 를 반쯤 끝내지 않기 위해)."""

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
        man = {"format": f"{PROGRAM}/split/1", "source": "folder", "utc": utc_now(), "data_root": str(self.root), "seed": int(seed),
               "split_ratios": self.cfg["data"]["split_ratios"], "classes": self.classes, "class_to_index": self.class_to_index,
               "n_items": len(items), "per_class": per_class, "data_fingerprint_sha256": fp, "split_sha256": ssha,
               "group_rule_ko": "같은 화소 내용 또는 같은 정규화 파일 이름 (같은 하위 폴더) 은 같은 그룹 → 같은 분할", "items": items}
        self._items_by_split = {sp: [it for it in items if it["split"] == sp] for sp in ("train", "dev", "test")}
        return man

    @classmethod
    def from_manifest(cls, cfg: dict[str, Any], root: Path, man: dict[str, Any]) -> "DatasetManager":
        source = man.get("source", "folder")
        if source != cfg["data"]["source"]:
            raise CompatibilityError("설정과 저장 분할의 데이터 종류가 다르다. 데이터 변경은 새 실행에서 한다.")
        dm = make_dataset_manager(cfg, root)
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

    def load_image(self, item: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        return load_pil_rgb(self.root / item["relpath"], float(self.cfg["data"]["transparent_background"]))

    def ids_sha(self, split: str) -> str:
        return sha256_text("\n".join(sorted(it["relpath"] for it in self.items(split))))


# MNIST 공식 gzip IDX. 주소/체크섬: torchvision의 공개 MNIST 데이터 명세.
# https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html
# torchvision 실행/설치 없이 표준 라이브러리와 NumPy만 사용한다.
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_RESOURCES = (
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873", "train_images", (60000, 28, 28)),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432", "train_labels", (60000,)),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3", "test_images", (10000, 28, 28)),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c", "test_labels", (10000,)),
)


def _md5_file(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)  # 공개 데이터 무결성 체크섬; 인증/암호화 용도가 아님
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_mnist_idx(path: Path, shape: tuple[int, ...]) -> tuple[np.ndarray, str]:
    """magic/type/shape/정확한 payload 길이 검사. 잘못된 IDX를 유효 이미지로 묵인하지 않는다."""
    header_size = 4 + 4 * len(shape)
    size = header_size + math.prod(shape)
    with gzip.open(path, "rb") as f:
        payload = f.read(size + 1)
    if len(payload) != size:
        raise DataError(f"MNIST IDX 길이 오류: {path.name} ({len(payload)} != {size})")
    magic = struct.unpack_from(">I", payload, 0)[0]
    actual = struct.unpack_from(">" + "I" * len(shape), payload, 4)
    if magic != 2048 + len(shape) or actual != shape:
        raise DataError(f"MNIST IDX 헤더 오류: {path.name}, magic={magic}, shape={actual}, 기대={shape}")
    arr = np.frombuffer(payload, dtype=np.uint8, offset=header_size).reshape(shape)
    if len(shape) == 1 and np.any(arr > 9):
        raise DataError(f"MNIST 라벨 범위 오류: {path.name} (0~9 필요)")
    return arr, sha256_bytes(payload)


class MNISTDatasetManager(DatasetManager):
    """공식 test를 보존하는 가상 이미지 목록. 모델/학습 경로는 폴더형과 동일하다.
    train/dev는 공식 train의 인덱스를 seed 고정 층화 분할한다. 근접 중복/필체(writer) 분할은 아니다.
    정확한 화소 중복은 보고하되 표준 test 표본을 삭제하거나 재배정하지 않는다.
    """

    def __init__(self, cfg: dict[str, Any], root: Path):
        super().__init__(cfg, root)
        self._arrays: dict[str, np.ndarray] = {}
        self._archives: dict[str, dict[str, Any]] = {}

    def _archive(self, name: str, checksum: str, download: bool,
                 progress: Callable[[str], None] | None) -> Path:
        # 이 프로그램의 캐시, torchvision 캐시, 사용자가 복사한 gzip 파일을 받는다.
        choices = [self.root / "MNIST" / "raw" / name, self.root / "raw" / name, self.root / name]
        for p in choices:
            if p.is_file():
                if _md5_file(p) != checksum:
                    raise DataError(f"MNIST 체크섬 불일치: {p}. 잘못된 파일을 별도로 옮긴 뒤 다시 준비하라.")
                return p
        if not download:
            raise DataError(f"MNIST 파일 없음: {name}. 캐시 폴더에 공식 .gz 4개를 복사하거나 내려받기를 허용하라.")
        dest = choices[0]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(f"  MNIST 내려받는 중: {name}")
        fd, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".part", dir=dest.parent)
        part = Path(temporary)
        try:
            with os.fdopen(fd, "wb") as out:
                request = urllib.request.Request(MNIST_MIRROR + name, headers={"User-Agent": PROGRAM + "/" + VERSION})
                with urllib.request.urlopen(request, timeout=30) as response:
                    total = 0
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        total += len(block)
                        if total > 64 * 1024 * 1024:
                            raise DataError("MNIST 압축 파일 크기 상한(64 MiB)을 넘었다.")
                        out.write(block)
            if _md5_file(part) != checksum:
                raise DataError(f"내려받은 MNIST 체크섬 불일치: {name}")
            os.replace(part, dest)
        except (OSError, ValueError, DataError) as exc:
            raise DataError(f"MNIST 준비 실패 ({name}): {exc}. 인터넷 연결을 확인하거나 공식 .gz 4개를 캐시에 복사하라.") from exc
        finally:
            with contextlib.suppress(FileNotFoundError):
                part.unlink()
        return dest

    def _load_archives(self, download: bool, progress: Callable[[str], None] | None = None) -> None:
        if self._arrays:
            return
        arrays, archives = {}, {}
        for name, checksum, key, shape in MNIST_RESOURCES:
            p = self._archive(name, checksum, download, progress)
            arrays[key], raw_sha = read_mnist_idx(p, shape)
            archives[name] = {"relpath": p.relative_to(self.root).as_posix(), "md5": checksum,
                              "sha256": sha256_file(p), "idx_sha256": raw_sha, "shape": list(shape)}
        for source in ("train", "test"):
            if len(arrays[source + "_images"]) != len(arrays[source + "_labels"]):
                raise DataError(f"MNIST {source} 영상/라벨 개수 불일치")
        self._arrays, self._archives = arrays, archives

    def scan(self, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
        _require_pil()
        self._load_archives(bool(self.cfg["data"]["mnist_download"]), progress)
        self.classes = [str(i) for i in range(10)]
        self.class_to_index = {c: int(c) for c in self.classes}
        recs, by_pix = [], {}
        for source in ("train", "test"):
            images, labels = self._arrays[source + "_images"], self._arrays[source + "_labels"]
            for idx, (im, label) in enumerate(zip(images, labels)):
                pix = sha256_bytes(b"28x28|" + im.tobytes())
                rec = {"relpath": f"mnist/{source}/{idx:05d}.png", "class": str(int(label)),
                       "official_split": source, "official_index": idx, "virtual_image": True,
                       "sha256": sha256_bytes(im.tobytes() + bytes([int(label)])), "pixel_sha256": pix,
                       "width": 28, "height": 28}
                recs.append(rec)
                by_pix.setdefault(pix, []).append(rec)
            if progress:
                progress(f"  MNIST 공식 {source}: {len(images):,} 장 (IDX 무결성 확인)")
        self.records = recs
        dup = [v for v in by_pix.values() if len(v) > 1]
        conflicts = [v for v in dup if len({r["class"] for r in v}) > 1]
        self.report = {
            "source": "mnist", "data_root": str(self.root), "utc": utc_now(), "archives": self._archives,
            "n_class_folders": 0, "classes": self.classes, "n_classes": 10, "n_valid_images": len(recs),
            "valid_images_per_class": {c: sum(r["class"] == c for r in recs) for c in self.classes},
            "corrupt": [], "unsupported": [], "hidden_ignored": [], "empty_classes_excluded": [], "root_files_ignored": [],
            "exact_duplicate_groups": [[r["relpath"] for r in v] for v in dup],
            "cross_class_duplicates_excluded": [],
            "cross_class_duplicates_observed_not_excluded": [[r["relpath"] for r in v] for v in conflicts],
            "original_size": {"width_min": 28, "width_max": 28, "width_median": 28.0,
                              "height_min": 28, "height_max": 28, "height_median": 28.0},
            "duplicate_rule_ko": "원본 28×28 화소 완전 일치만 기록. 공식 test 구성 보존을 위해 중복을 삭제하지 않는다. "
                                 "train/dev는 인덱스 층화 분할; 중복이 경계를 넘으면 split_manifest에 수를 보고한다. writer/근접 중복 분할 아님.",
            "preprocessing_ko": f"28×28 회색조 → R=G=B 복제 → 기존 {self.cfg['data']['image_size']}×"
                                f"{self.cfg['data']['image_size']} 전처리. 반전/추가 블러/전 데이터 표준화 없음. 확대는 새 정보를 만들지 않음.",
        }
        return self.report

    def split(self, seed: int) -> dict[str, Any]:
        if not self.records:
            raise SplitError("먼저 scan()으로 MNIST를 검사하라.")
        labels = self._arrays["train_labels"]
        n_dev = int(self.cfg["data"]["mnist_dev_size"])
        counts = np.bincount(labels, minlength=10)
        if np.any(counts < 2) or not 10 <= n_dev <= len(labels) - 10:
            raise SplitError("MNIST train/dev에 각 숫자 최소 1장을 배정할 수 없다.")
        quotas = counts.astype(np.float64) * n_dev / len(labels)
        allocation = np.minimum(np.maximum(np.floor(quotas).astype(int), 1), counts - 1)
        # 최대 잔여 방식 + 클래스 순서 동률 처리. 정확히 요청한 dev 개수를 보존한다.
        while int(allocation.sum()) < n_dev:
            scores = np.where(allocation < counts - 1, quotas - allocation, -np.inf)
            allocation[int(np.argmax(scores))] += 1
        while int(allocation.sum()) > n_dev:
            scores = np.where(allocation > 1, allocation - quotas, -np.inf)
            allocation[int(np.argmax(scores))] -= 1
        rng = np.random.default_rng([int(seed), 107])  # 모델 초기화/학습 순서와 독립
        dev = set()
        for label in range(10):
            inds = np.flatnonzero(labels == label)
            dev.update(map(int, rng.permutation(inds)[:int(allocation[label])]))
        items = []
        for index, rec in enumerate(self.records):
            source, idx = rec["official_split"], rec["official_index"]
            sp = "test" if source == "test" else ("dev" if idx in dev else "train")
            items.append({**rec, "index": index, "label": int(rec["class"]),
                          "group": f"mnist-{source}-{idx:05d}", "split": sp})
        per_class = {}
        for c in self.classes:
            per_class[c] = {"n_groups": sum(it["class"] == c for it in items)}
            for sp in ("train", "dev", "test"):
                n = sum(it["class"] == c and it["split"] == sp for it in items)
                per_class[c]["n_" + sp + "_images"] = n
                per_class[c][sp + "_groups"] = n
        man = self._finish_manifest(items, seed, per_class)
        hash_splits: dict[str, set[str]] = {}
        for it in items:
            hash_splits.setdefault(it["pixel_sha256"], set()).add(it["split"])
        overlap = {f"{a}__{b}": sum(a in splits and b in splits for splits in hash_splits.values())
                   for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))}
        man.update({"source": "mnist", "archives": self._archives, "split_ratios": None,
                    "split_policy": "official-test-stratified-train-v1", "mnist_dev_size": n_dev,
                    "official_counts": {sp: len(self._arrays[sp + "_labels"]) for sp in ("train", "test")},
                    "exact_pixel_overlap_groups": overlap,
                    "group_rule_ko": "공식 test 전체 유지. 공식 train만 seed 고정 라벨 층화 분할; 인덱스별 독립 배정. "
                                     "동일 화소의 경계 중복 수는 별도 보고; 클래스별 폴더 비율은 적용하지 않음."})
        self.manifest = man
        return man

    def load_image(self, item: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        self._load_archives(download=False)        # 학습/추론 중 숨은 내려받기 없음
        source, idx = item["official_split"], int(item["official_index"])
        image = Image.fromarray(self._arrays[source + "_images"][idx]).convert("RGB")
        return image, {"width": 28, "height": 28, "mode": "L", "format": "MNIST_IDX", "alpha_composited": False}

    def verify_files(self) -> list[str]:
        """재개: 공식 checksum, IDX, 라벨/분할/샘플 해시를 재검증. 네트워크 접근 없음."""
        try:
            cfg = copy.deepcopy(self.cfg)
            cfg["data"]["mnist_download"] = False
            check = MNISTDatasetManager(cfg, self.root)
            check.scan()
            expected = check.split(int(cfg["seed"]))
            saved = self.manifest or {}
            for key in ("source", "classes", "class_to_index", "n_items", "data_fingerprint_sha256", "split_sha256",
                        "split_policy", "mnist_dev_size", "items"):
                if saved.get(key) != expected[key]:
                    return [f"MNIST 저장 목록/분할 불일치: {key}"]
            self._arrays, self._archives = check._arrays, check._archives
            return []
        except (OSError, ValueError, DataError, SplitError) as exc:
            return [f"MNIST 검증 실패: {exc}"]


def make_dataset_manager(cfg: dict[str, Any], root: Path) -> DatasetManager:
    return (MNISTDatasetManager if cfg["data"]["source"] == "mnist" else DatasetManager)(cfg, root)


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
            im, _meta = self.data.load_image(item)
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
# 7. 연결 구조 (LayerSpec) 와 영역 생성기 (V1 / V2 / V4 / IT)
# ======================================================================
@dataclass
class LayerSpec:
    """한 영역·한 층의 고정 구조 (CPU numpy).

    뉴런 순서는 위치 우선: local = pos·n_ch + ch.  입력 연결은 CSR 두 입력군 (A, B):
      A: a_ptr (n+1,), a_src (E_a,) int64 (원천 층 local 주소), a_w (E_a,) float32
      B: b_ptr, b_src, b_w — 역할 연산 min2 뉴런만 쓴다 (u = min(Σ_A, Σ_B)). 선형 뉴런은 u = Σ_A (B 비어 있음).
    L2 층은 이 CSR 이 '역할별 기저 연결' 이다. 실제 전달은 입력 비율 라우팅으로 고른 후보 하나 (집단마다) 만 한다 (RouteTable).
    V4 곡선 행은 A (P, θA) 만 정적이고 B 는 후보 원천 목록 ropt 에서 후보 묶음으로 정한다 (routed).
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
    routed: np.ndarray | None = None              # (n,) bool — B 를 후보 원천 목록에서 고르는 행 (V4 L2 곡선)
    route_space: np.ndarray | None = None         # (n,) int64 — 선택 집단의 공간 키 (위치, IT 는 공간 범위)
    route_family: np.ndarray | None = None        # (n,) int64 — 선택 집단의 역할 계열 코드
    route_family_names: list[str] | None = None
    route_space_kind: str | None = None
    ropt: dict[str, np.ndarray] | None = None     # routed 행의 B 후보 원천 (탈락 후보·사유 포함)
    ropt_removed: dict[str, Any] | None = None    # 수용장 단계에서 후보로 넣지 않은 수 (같은 위치·수용장 밖) 와 조건별 탈락 수

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
        if self.route_space is None:
            return h
        extra = sha256_array(self.route_space, self.route_family, self.routed if self.routed is not None else np.zeros(self.n, bool))
        extra += dumps([self.route_family_names, self.route_space_kind], indent=None)
        if self.ropt:
            extra += sha256_array(*[self.ropt[k] for k in sorted(self.ropt)])
        return sha256_text(h + extra)


def set_route_keys(sp: LayerSpec, space: np.ndarray, family: Sequence[str], kind: str) -> None:
    """L2 뉴런마다 선택 집단 키 (공간 키, 역할 계열) 를 단다. 집단 = (공간 키, 계열)."""
    names: list[str] = []
    for f_ in family:
        if f_ not in names:
            names.append(f_)
    sp.route_space = np.asarray(space, np.int64)
    sp.route_family = np.asarray([names.index(f_) for f_ in family], np.int64)
    sp.route_family_names = names
    sp.route_space_kind = kind
    if len(sp.route_space) != sp.n or len(sp.route_family) != sp.n:
        raise StructureError(f"{sp.key}: 선택 집단 키 길이가 뉴런 수와 다르다.")


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
          stats: dict[str, Any] | None = None, routed: np.ndarray | None = None) -> LayerSpec:
    a_ptr, a_src, a_w, b_ptr, b_src, b_w = csr
    sp = LayerSpec(area=area, name=name, src_key=src_key, n_src=int(n_src), n_pos=int(n_pos), n_ch=int(n_ch),
                   channel_names=list(names), channel_roles=list(roles), channel_meta=meta, op_min=np.asarray(op_min, bool),
                   a_ptr=a_ptr, a_src=a_src, a_w=a_w.astype(np.float32), b_ptr=b_ptr, b_src=b_src, b_w=b_w.astype(np.float32),
                   x=np.asarray(x, np.float32), y=np.asarray(y, np.float32), rf=np.asarray(rf, np.float32),
                   ring=np.asarray(ring, np.int32), ang=np.asarray(ang, np.int32),
                   pos_x=np.asarray(pos[0], np.float32), pos_y=np.asarray(pos[1], np.float32), pos_rf=np.asarray(pos[2], np.float32),
                   grid_shape=grid_shape, pos_nb=nb[0], pos_nb_count=nb[1], learnable=learnable, l6_capable=(name in ("L2", "L3")),
                   description=desc, build_stats=dict(stats or {}), routed=routed)
    sp.build_stats.update(validate_layer(sp))
    return sp


def _per_neuron(arr: np.ndarray, n_ch: int) -> np.ndarray:
    return np.repeat(np.asarray(arr), n_ch)


# ---------------------------------------------------------------------- 곡선 기하 (기준 파일 4.0.0 과 같은 식)
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
    set_route_keys(L2, np.repeat(np.arange(n_pos1), n_ch2), [f"gabor_{colors[c]}" for _p in range(n_pos1) for (c, _o, _f, _s) in chans],
                   "position")
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
    set_route_keys(L2, np.repeat(np.arange(n_pos), n_ch), [f for _p in range(n_pos) for (f, _i, _j) in chans], "position")
    return [L4, L2, L3]


# ---------------------------------------------------------------------- V4
CURVE_REASONS = {0: "valid", 1: "behind_P", 2: "bend_side", 3: "path_outside_field", 4: "zero_tangent", 5: "folded_path"}
CURVE_REASONS_KO = {
    "same_position": "Q = P (같은 위치는 두 번째 위치가 아니다) — 후보 레코드에 넣지 않고 수만 센다",
    "outside_rf": "‖Q − P‖ > R_j (V4 L2 수용장 밖, 원영상 화소 거리) — 후보 레코드에 넣지 않고 수만 센다",
    "behind_P": "진행 방향 조건: (Q − P)·dir(φA) ≤ 0 (곡선 역할은 P 에서 +θA 방향으로 나아간다 — 기존 역할 정의)",
    "bend_side": "뒤집힌 굽힘: s·cross(dir(φA), Q − P) ≤ 0 (Q 가 P 접선의 굽힘 쪽 반평면 밖)",
    "path_outside_field": "Hermite 경로 점이 로그-극좌표 표본 원 (r > Rmax) 밖 (시야 밖)",
    "zero_tangent": "경로 접선 크기 0 (방향을 정의할 수 없다)",
    "folded_path": "경로가 P→Q 현 방향으로 되돌아가거나 (p'·(Q−P) ≤ 0) 총 회전 > 180° (접힘·고리)",
}


def curve_route_options(cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", l4: LayerSpec, l2_rf: np.ndarray, n_ch2: int,
                        chans: list[tuple[str, int, int]], curve_info: list[tuple[int, int, int, int, float]],
                        contour: dict[int, int]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """V4 L2 곡선 뉴런 (P, θA → θB, 부호 s, κ) 의 B 원천 후보 = 수용장 안 θB contour 위치 Q (기준 파일 4.0.0 의 기하 조건).
    데이터·라벨을 쓰지 않는 구조 규칙이다. 탈락 후보도 사유와 함께 레코드로 남긴다 (valid = False)."""
    M = int(cfg["routing"]["curve_path_points"])
    oris = [float(t) for t in cfg["areas"]["V1"]["orientations_deg"]]
    px, py = l4.pos_x.astype(np.float64), l4.pos_y.astype(np.float64)
    vx, vy = px - pre.cx, pre.cy - py
    n_pos, n_in = l4.n_pos, l4.n_ch
    ring4 = l4.ring[::n_in].astype(np.int64)
    cinfo = {ch: (i, j, s, kap) for (ch, i, j, s, kap) in curve_info}
    keys = ("same_position", "outside_rf", *[CURVE_REASONS[c] for c in range(1, 6)])
    rem_total = {k: 0 for k in keys}
    rem_ring: dict[int, dict[str, int]] = {int(k): {kk: 0 for kk in keys} for k in np.unique(ring4)}
    rows, srcs, qs, reasons, dists, cdevs = [], [], [], [], [], []
    for P in range(n_pos):
        Rj = float(l2_rf[P])
        for ch in range(n_ch2):
            if chans[ch][0] != "curve":
                continue
            i, j, s, kap = cinfo[ch]
            phi_a, phi_b, phi_c = curve_directions(oris[i], s, kap)
            ta = np.asarray([math.cos(math.radians(phi_a)), math.sin(math.radians(phi_a))])
            Pv = np.asarray([vx[P], vy[P]])
            items = []
            for Q in range(n_pos):
                if Q == P:
                    rem_total["same_position"] += 1
                    rem_ring[int(ring4[P])]["same_position"] += 1
                    continue
                d = float(math.hypot(px[Q] - px[P], py[Q] - py[P]))
                if d > Rj:
                    rem_total["outside_rf"] += 1
                    rem_ring[int(ring4[P])]["outside_rf"] += 1
                    continue
                Qv = np.asarray([vx[Q], vy[Q]])
                chord = Qv - Pv
                L = float(np.hypot(*chord))
                if L <= 1e-12:
                    raise StructureError(f"V4 위치 {P} 와 {Q} 가 같은 좌표다 (곡선 후보 현 길이 0).")
                cdev = float(d_dir(math.degrees(math.atan2(chord[1], chord[0])), phi_c))
                p_, dp_ = hermite_path(Pv, Qv, phi_a, phi_b, M)
                psi = np.degrees(np.arctan2(dp_[:, 1], dp_[:, 0]))
                tot_abs = float(np.abs(wrap180(np.diff(psi))).sum())
                if float(chord @ ta) <= 0.0:
                    reason = 1
                elif s * float(ta[0] * chord[1] - ta[1] * chord[0]) <= 0.0:
                    reason = 2
                elif (np.hypot(p_[:, 0], p_[:, 1]) > pre.rmax).any():
                    reason = 3
                elif (np.hypot(dp_[:, 0], dp_[:, 1]) < 1e-9 * L).any():
                    reason = 4
                elif ((dp_ @ chord) <= 0.0).any() or tot_abs > 180.0:
                    reason = 5
                else:
                    reason = 0
                if reason:
                    rem_total[CURVE_REASONS[reason]] += 1
                    rem_ring[int(ring4[P])][CURVE_REASONS[reason]] += 1
                items.append((Q * n_in + contour[j], Q, reason, d, cdev))
            items.sort(key=lambda it: it[0])                                # 원천 L4 주소 오름차순 (route_id 순서 = 동점 해소 순서)
            for (src, Q, reason, d, cdev) in items:
                rows.append(P * n_ch2 + ch)
                srcs.append(src)
                qs.append(Q)
                reasons.append(reason)
                dists.append(d)
                cdevs.append(cdev)
    ropt = {"row": np.asarray(rows, np.int64), "src": np.asarray(srcs, np.int64), "w": np.ones(len(rows), np.float32),
            "q": np.asarray(qs, np.int64), "valid": np.asarray(reasons, np.int8) == 0, "reason": np.asarray(reasons, np.int8),
            "dist_px": np.asarray(dists, np.float32), "chord_dev_deg": np.asarray(cdevs, np.float32)}
    return ropt, {"curve_total": rem_total, "curve_by_ring": rem_ring, "path_points": M, "reasons_ko": CURVE_REASONS_KO}


def build_v4(cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", v2: LayerSpec) -> list[LayerSpec]:
    a = cfg["areas"]["V4"]
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
    curve_info: list[tuple[int, int, int, int, float]] = []          # (채널, θA, θB, 곡률 부호, κ)
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
    st: dict[str, Any] = {"straight_backward_used": 0, "straight_same_position_fallback": 0}
    for P in range(n_pos):
        for ci, (f, i, j) in enumerate(chans):
            r = P * n_ch + ci
            if f == "curve":
                b2.add(r, P * n_in + contour[i], 1.0, "A")             # 첫 입력 (P, θA) 은 고정, B 는 후보 원천 목록 (ropt)
                op[r] = True
                routed[r] = True
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
    ropt, removed = curve_route_options(cfg, pre, L4, prf + D, n_ch, chans, curve_info, contour)
    has_valid = np.zeros(n_pos * n_ch, bool)
    has_valid[ropt["row"][ropt["valid"]]] = True
    st.update(curve_route_options=int(len(ropt["row"])), curve_route_options_valid=int(ropt["valid"].sum()),
              curve_neurons_without_valid_option=int((routed & ~has_valid).sum()))
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
    ori_b = [None] * n_ch
    for (ch_, _i, j_, s_, k_) in curve_info:
        csign[ch_], ckap[ch_], ori_b[ch_] = s_, k_, oris[j_]
    L2 = _spec("V4", "L2", "V4/L4", n_pos * n_in, n_pos, n_ch, names2, roles2,
               _channel_meta(n_ch, family=fam2, orientation_deg=ori2, orientation_b_deg=ori_b, curve_sign=csign, curvature_deg=ckap),
               op, b2.finalize(), _per_neuron(px, n_ch), _per_neuron(py, n_ch), _per_neuron(prf + D, n_ch), _per_neuron(ring, n_ch),
               _per_neuron(ang, n_ch), (px, py, prf + D), (nr4, nphi4), nb, True,
               {"role_ko": "더 넓은 수용장에서 선 조합: 곡선 조각 (θA 다음 위치에 θA±κ, min2), 직선 연속 (min2), 접합 중계, 선 끝 합",
                "operation": {"curve": "u = min(contour(P,θA), contour(Q,θB))  (Q = 선택된 후보 묶음의 B 원천)",
                              "straight": "u = min(contour(P,θ), contour(P+D·dir θ, θ))", "junction": "u = cross(P)",
                              "lineend": "u = 0.5·endstop_fwd(P,θ) + 0.5·endstop_bwd(P,θ)"},
                "input_shape": [n_pos, n_in], "output_shape": [n_pos, n_ch],
                "connection_rule_ko": "D = offset_factor × 국소 V4 간격. 직선 B 는 앞쪽 변위 (없으면 뒤쪽, 그것도 없으면 같은 위치, 개수 기록). "
                                      "곡선 B 는 수용장 안 θB contour 위치 후보 (기준 파일 4.0.0 의 기하 조건, 탈락 사유 기록)",
                "curvature_deg": [float(k) for k in a["curvature_deg"]],
                "routing_ko": f"{MODE}: 위치 × 역할 계열 집단마다 입력 비율과 대표 벡터의 코사인으로 후보 묶음 하나를 고른다 (곡선은 (뉴런, Q) 묶음)."},
               st, routed=routed)
    L2.ropt, L2.ropt_removed = ropt, removed
    set_route_keys(L2, np.repeat(np.arange(n_pos), n_ch), fam2 * n_pos, "position")
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
    set_route_keys(L2, np.asarray([n_reg if f_ == "global" else info_[0] for (f_, info_) in neurons], np.int64),
                   [("global_" + str(fam[info_])) if f_ == "global" else f_ for (f_, info_) in neurons],
                   "it_range (공간 키 = 영역 R, 전역 특징 = n_reg '전체 시야')")
    return [L4, L2, L3]


class CortexArea:
    """영역 하나 = 고정 구조 L4 → L2 → L3 (LayerSpec 3 개). 하위 클래스 V1/V2/V4/IT 가 역할 연산 (생성기) 을 정한다.
    L4 → L2 목적지 선택 (RatioRouter) 은 모든 영역이 같은 클래스를 쓴다 (CortexModel.add_area)."""

    name = ""
    role_ko = ""

    def __init__(self, specs: Sequence[LayerSpec]):
        self.specs = list(specs)
        self.l4, self.l2, self.l3 = self.specs

    @classmethod
    def builder(cls, cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", prev_l3: LayerSpec | None) -> list[LayerSpec]:
        raise StructureError(f"{cls.__name__} 에 생성기가 없다.")

    @classmethod
    def build(cls, cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", prev_l3: LayerSpec | None) -> "CortexArea":
        if cls.name != "V1" and prev_l3 is None:
            raise StructureError(f"{cls.name} 는 앞 영역 L3 가 있어야 만든다.")
        specs = cls.builder(cfg, pre, prev_l3)
        if specs[1].route_space is None:
            raise StructureError(f"{cls.name} L2 에 선택 집단 키가 없다 (모든 영역이 입력 비율 라우팅을 쓴다).")
        return cls(specs)

    def describe(self) -> dict[str, Any]:
        return {"area": self.name, "role_ko": self.role_ko, "layers": {sp.name: sp.description for sp in self.specs},
                "h_dim": self.l3.n, "structure_sha256": area_structure_hash(self.specs)}


class V1(CortexArea):
    name = "V1"
    role_ko = "위치별 방향·위상·극성 고정 필터 (L2, 수용장 패킷으로 후보 하나 선택) 와 위상×극성 국소 결합 (L3)"

    @classmethod
    def builder(cls, cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", prev_l3: LayerSpec | None) -> list[LayerSpec]:
        return build_v1(cfg, pre)


class V2(CortexArea):
    name = "V2"
    role_ko = "근처 선 반응의 조합: 윤곽 연속·끝점·교차 (L2), 일대일 정리 (L3)"

    @classmethod
    def builder(cls, cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", prev_l3: LayerSpec | None) -> list[LayerSpec]:
        return build_v2(cfg, pre, prev_l3)


class V4(CortexArea):
    name = "V4"
    role_ko = "더 넓은 수용장의 선 조합: 곡선 조각·직선 연속·접합·선 끝 (L2), 격자 이웃 결합 (L3)"

    @classmethod
    def builder(cls, cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", prev_l3: LayerSpec | None) -> list[LayerSpec]:
        return build_v4(cfg, pre, prev_l3)


class IT(CortexArea):
    name = "IT"
    role_ko = "넓은 범위 특징 결합 (L2: 전역·영역·결합 집단) 과 최종 판독 벡터 (L3)"

    @classmethod
    def builder(cls, cfg: dict[str, Any], pre: "RetinaLGNPreprocessor", prev_l3: LayerSpec | None) -> list[LayerSpec]:
        return build_it(cfg, pre, prev_l3)


AREA_CLASSES: dict[str, type[CortexArea]] = {"V1": V1, "V2": V2, "V4": V4, "IT": IT}


def area_structure_hash(specs: Sequence[LayerSpec]) -> str:
    return sha256_text("|".join(sp.hash() for sp in specs))


# ======================================================================
# 8. 입력 비율 라우팅 후보 그래프 (RouteTable) — 구조 생성 때 한 번, 데이터·라벨 미사용
#    집단 g = 영역 + 수신 공간 키 + 역할 계열. 후보 = 완전한 입력 배정 묶음 (목적지 L2 하나 + A/B 슬롯 원천·기저 가중치).
#    합법적인 후보는 모두 저장한다 (첫 K 개만 남기지 않는다). 계산은 묶음 (bucket) 과 집단 조각으로 나눈다.
# ======================================================================
ROUTE_REASONS = {0: "legal", 1: "behind_P", 2: "bend_side", 3: "path_outside_field", 4: "zero_tangent", 5: "folded_path",
                 6: "no_A_input", 7: "min2_missing_B"}
ROUTE_REASONS_KO = {**{k: v for k, v in CURVE_REASONS_KO.items() if k not in ("same_position", "outside_rf")},
                    "legal": "합법 후보 (경쟁에 참여)", "no_A_input": "A 슬롯 원천이 없다", "min2_missing_B": "min2 역할인데 B 슬롯 원천이 없다"}
SLOT_NAMES = ("A", "B")
ROUTE_CONVENTION_KO = {
    "group": "집단 = (영역, 공간 키, 역할 계열). V1·V2·V4 공간 키 = 수신 L2 위치, IT 공간 키 = 영역 R (전역 특징은 '전체 시야' 하나). 영역 전체·IT 전체 top-1 이 아니다.",
    "candidate": "후보 = 목적지 L2 하나와 그 A/B 슬롯 원천 묶음. 곡선은 (곡선 뉴런, Q) 묶음마다 다른 route_id. 목록 순번을 각도·의미로 쓰지 않는다.",
    "source_order": "source_ids[g] = 집단 합법 후보들의 원천 L4 local 주소 합집합의 오름차순 (= 위치 우선·채널 순서, ON/OFF 는 채널). 값 크기 정렬 없음.",
    "no_duplicate": "같은 원천이 A·B 슬롯 모두에 필요하면 슬롯 가중치는 따로 두되 x_g 에는 한 번만 넣는다.",
    "support": "후보의 구조적 지원 = 그 후보의 A·B 슬롯 원천. 대표 벡터는 지원 밖에서 0, 지원 위에서 비음수 단위 노름.",
    "no_candidate": "기하 후보가 없는 목적지·집단은 '후보 없음' 으로 기록한다 (활동 0 과 구분). 0 번 뉴런으로 보내지 않는다.",
    "codebook_id": "codebook_id = (묶음, 묶음 안 집단 칸, 후보 칸). 집단끼리 대표 벡터를 공유하지 않는다. descriptor_schema_id = 집단 번호.",
    "fanout": "같은 L4 가 여러 집단·후보에 들어갈 수 있다 (top-1 은 집단당 하나이며 L4 뉴런당 축삭 하나가 아니다). fan-out 을 기록한다.",
}


@dataclass
class RouteBucket:
    """같은 역할 계열·비슷한 크기의 집단 묶음 (연속 배열, 패딩 + 마스크)."""

    family: int
    groups: np.ndarray        # (G,) 영역 안 집단 번호
    src: np.ndarray           # (G, D) L4 local 주소 (패딩 = n_l4 → 0 열)
    dvalid: np.ndarray        # (G, D)
    route: np.ndarray         # (G, K) route_id (패딩 −1)
    legal: np.ndarray         # (G, K)
    dst: np.ndarray           # (G, K) L2 local (패딩 n_l2)
    support: np.ndarray       # (G, K, D) bool
    WA: np.ndarray            # (G, K, D) float32 A 슬롯 기저 가중치 (부호 유지)
    WB: np.ndarray | None     # (G, K, D) float32 B 슬롯 (min2 후보가 없는 묶음은 None)
    op_min: np.ndarray        # (G, K) bool

    @property
    def G(self) -> int:
        return int(self.src.shape[0])

    @property
    def D(self) -> int:
        return int(self.src.shape[1])

    @property
    def K(self) -> int:
        return int(self.route.shape[1])

    def nbytes(self) -> int:
        n = self.G * self.K * self.D
        return int(n * (4 + 1 + 4 + (4 if self.WB is not None else 0)) + self.src.size * 8 + self.route.size * 18)


@dataclass
class RouteTable:
    area: str
    n_l4: int
    n_l2: int
    family_names: list[str]
    space_kind: str
    params: dict[str, Any]
    # 후보 레코드 (탈락 후보 포함)
    r_dst: np.ndarray
    r_group: np.ndarray             # 집단 번호 (탈락 −1)
    r_legal: np.ndarray
    r_reason: np.ndarray            # int8 (ROUTE_REASONS)
    r_op_min: np.ndarray
    r_ptr: np.ndarray               # (R+1,) → e_*
    e_src: np.ndarray
    e_slot: np.ndarray              # 0 = A, 1 = B
    e_w: np.ndarray
    r_q: np.ndarray                 # 곡선 B 위치 (그 밖 −1)
    r_dist_px: np.ndarray
    r_chord_dev_deg: np.ndarray
    r_bucket: np.ndarray            # codebook_id (묶음, 집단 칸, 후보 칸) — 탈락 −1
    r_bk_g: np.ndarray
    r_bk_k: np.ndarray
    # 집단
    g_space: np.ndarray
    g_family: np.ndarray
    g_n_routes: np.ndarray
    g_D: np.ndarray
    g_bucket: np.ndarray            # 후보 없는 집단 −1
    g_local: np.ndarray
    g_src_ptr: np.ndarray
    g_src: np.ndarray
    g_mem_ptr: np.ndarray
    g_mem: np.ndarray
    l2_group: np.ndarray
    l2_n_legal: np.ndarray
    buckets: list[RouteBucket]
    removed: dict[str, Any]

    @property
    def n_routes(self) -> int:
        return int(len(self.r_dst))

    @property
    def n_groups(self) -> int:
        return int(len(self.g_space))

    def group_sources(self, g: int) -> np.ndarray:
        return self.g_src[self.g_src_ptr[g]:self.g_src_ptr[g + 1]]

    def group_members(self, g: int) -> np.ndarray:
        return self.g_mem[self.g_mem_ptr[g]:self.g_mem_ptr[g + 1]]

    def group_routes(self, g: int) -> np.ndarray:
        return np.nonzero(self.r_group == g)[0]

    def route_edges(self, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a, b = int(self.r_ptr[r]), int(self.r_ptr[r + 1])
        return self.e_src[a:b], self.e_slot[a:b], self.e_w[a:b]

    def hash(self) -> str:
        parts = [self.r_dst, self.r_group, self.r_legal, self.r_reason, self.r_op_min, self.r_ptr, self.e_src, self.e_slot, self.e_w,
                 self.g_space, self.g_family, self.g_src_ptr, self.g_src, self.g_bucket, self.g_local]
        for bk in self.buckets:
            parts += [bk.src, bk.route, bk.legal, bk.dst, bk.support, bk.WA, bk.op_min] + ([bk.WB] if bk.WB is not None else [])
        return sha256_text(dumps({"area": self.area, "families": self.family_names, "space_kind": self.space_kind, "params": self.params},
                                 indent=None) + sha256_array(*parts))

    def nbytes(self) -> int:
        return int(sum(bk.nbytes() for bk in self.buckets))


def ragged(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """길이 counts 인 구간들의 (구간 번호, 구간 안 위치) — 가변 길이 목록을 반복문 없이 펼친다."""
    counts = np.asarray(counts, np.int64)
    rep = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    off = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    return rep, off


def build_route_table(area: str, l4: LayerSpec, l2: LayerSpec, cfg: dict[str, Any]) -> RouteTable:
    """L2 의 역할별 기저 연결 (CSR) 과 곡선 B 후보 원천 (ropt) 에서 후보 그래프·집단·묶음을 만든다."""
    if l2.route_space is None or l2.route_family is None:
        raise StructureError(f"{l2.key}: 선택 집단 키가 없다.")
    ro = cfg["routing"]
    n2, n4 = l2.n, l4.n
    if l2.n_src != n4:
        raise StructureError(f"{l2.key} 의 원천 크기 {l2.n_src} != {l4.key} 뉴런 수 {n4}")
    routed = l2.routed if l2.routed is not None else np.zeros(n2, bool)
    ropt = l2.ropt or {}
    o_row = np.asarray(ropt.get("row", np.zeros(0, np.int64)), np.int64)
    if len(o_row) and ((np.diff(o_row) < 0).any() or not routed[o_row].all()):
        raise StructureError(f"{l2.key}: 곡선 B 후보 원천 목록이 행 순서가 아니거나 routed 가 아닌 행을 가리킨다.")
    fanA = np.diff(l2.a_ptr).astype(np.int64)
    fanB = np.diff(l2.b_ptr).astype(np.int64)
    nr = np.where(routed, 0, 1).astype(np.int64)
    if len(o_row):
        nr += np.bincount(o_row, minlength=n2)
    R = int(nr.sum())
    r_dst_a = np.repeat(np.arange(n2, dtype=np.int64), nr)                    # 후보 순서 = L2 행 순서, 곡선은 원천 주소 순서
    r_routed = routed[r_dst_a] if R else np.zeros(0, bool)
    opt_of = np.full(R, -1, np.int64)
    opt_of[r_routed] = np.arange(int(r_routed.sum()))
    fa = fanA[r_dst_a]
    fb = np.where(r_routed, 1, fanB[r_dst_a])
    r_ptr_a = np.concatenate([[0], np.cumsum(fa + fb)]).astype(np.int64)
    E = int(r_ptr_a[-1])
    e_src_a = np.full(E, -1, np.int64)
    e_slot_a = np.zeros(E, np.int8)
    e_w_a = np.zeros(E, np.float32)

    rep, off = ragged(fa)                                                   # A 슬롯 (모든 후보)
    pos = r_ptr_a[rep] + off
    e_src_a[pos] = l2.a_src[l2.a_ptr[r_dst_a[rep]] + off]
    e_w_a[pos] = l2.a_w[l2.a_ptr[r_dst_a[rep]] + off]
    nb_ = np.where(r_routed, 0, fb)
    rep, off = ragged(nb_)                                                  # B 슬롯 (기존 CSR B)
    pos = r_ptr_a[rep] + fa[rep] + off
    e_src_a[pos] = l2.b_src[l2.b_ptr[r_dst_a[rep]] + off]
    e_w_a[pos] = l2.b_w[l2.b_ptr[r_dst_a[rep]] + off]
    e_slot_a[pos] = 1
    rr_ = np.nonzero(r_routed)[0]                                           # B 슬롯 (곡선: 후보 묶음의 Q 원천 하나)
    pos = r_ptr_a[rr_] + fa[rr_]
    if len(rr_):
        e_src_a[pos] = ropt["src"][opt_of[rr_]]
        e_w_a[pos] = ropt["w"][opt_of[rr_]]
        e_slot_a[pos] = 1
    if (e_src_a < 0).any() or (e_src_a >= n4).any():
        raise StructureError(f"{l2.key} 후보 원천 주소가 범위 밖이다.")
    r_op = np.where(r_routed, True, l2.op_min[r_dst_a] if R else np.zeros(0, bool)).astype(bool)
    r_reason = np.zeros(R, np.int8)
    r_reason[r_routed] = ropt["reason"][opt_of[r_routed]] if len(rr_) else 0
    r_reason[(~r_routed) & r_op & (fb == 0)] = 7
    r_reason[fa == 0] = 6
    r_q = np.full(R, -1, np.int64)
    r_dist = np.full(R, np.nan, np.float32)
    r_cdev = np.full(R, np.nan, np.float32)
    if len(rr_):
        r_q[rr_] = ropt["q"][opt_of[rr_]]
        r_dist[rr_] = ropt["dist_px"][opt_of[rr_]]
        r_cdev[rr_] = ropt["chord_dev_deg"][opt_of[rr_]]
    r_legal_a = r_reason == 0
    # ---------------------------------------------------------------- 집단 (L2 뉴런 순서로 처음 나타난 순서)
    key_of: dict[tuple[int, int], int] = {}
    l2_group = np.zeros(n2, np.int64)
    gkeys: list[tuple[int, int]] = []
    for j in range(n2):
        k_ = (int(l2.route_space[j]), int(l2.route_family[j]))
        if k_ not in key_of:
            key_of[k_] = len(gkeys)
            gkeys.append(k_)
        l2_group[j] = key_of[k_]
    Gt = len(gkeys)
    r_group = np.where(r_legal_a, l2_group[r_dst_a] if R else np.zeros(0, np.int64), -1).astype(np.int64)
    g_routes: list[list[int]] = [[] for _ in range(Gt)]
    for r in range(R):
        if r_legal_a[r]:
            g_routes[int(r_group[r])].append(r)
    g_src_list: list[np.ndarray] = []
    for g in range(Gt):
        ss = [e_src_a[r_ptr_a[r]:r_ptr_a[r + 1]] for r in g_routes[g]]
        g_src_list.append(np.unique(np.concatenate(ss)) if ss else np.zeros(0, np.int64))
    mem = [[] for _ in range(Gt)]
    for j in range(n2):
        mem[int(l2_group[j])].append(j)
    l2_n_legal = np.bincount(r_dst_a[r_legal_a], minlength=n2).astype(np.int64) if R else np.zeros(n2, np.int64)
    # ---------------------------------------------------------------- 묶음 (같은 계열, 크기순, 패딩 상한)
    pad_max = float(ro["bucket_padding_max"])
    max_el = int(ro["bucket_max_elements"])
    g_bucket = np.full(Gt, -1, np.int64)
    g_local = np.full(Gt, -1, np.int64)
    plans: list[tuple[int, list[int]]] = []
    for fam in range(len(l2.route_family_names or [])):
        gs = [g for g in range(Gt) if gkeys[g][1] == fam and g_routes[g]]
        gs.sort(key=lambda g: (len(g_src_list[g]), len(g_routes[g]), g))
        cur: list[int] = []
        for g in gs:
            trial = cur + [g]
            Dm = max(len(g_src_list[x]) for x in trial)
            Km = max(len(g_routes[x]) for x in trial)
            actual = sum(len(g_src_list[x]) * len(g_routes[x]) for x in trial)
            if cur and (len(trial) * Dm * Km > pad_max * actual or len(trial) * Dm * Km > max_el):
                plans.append((fam, cur))
                cur = [g]
            else:
                cur = trial
        if cur:
            plans.append((fam, cur))
    r_bucket = np.full(R, -1, np.int64)
    r_bk_g = np.full(R, -1, np.int64)
    r_bk_k = np.full(R, -1, np.int64)
    buckets: list[RouteBucket] = []
    for b, (fam, gs) in enumerate(plans):
        gs = sorted(gs)
        G = len(gs)
        Dm = max(len(g_src_list[g]) for g in gs)
        Km = max(len(g_routes[g]) for g in gs)
        src = np.full((G, Dm), n4, np.int64)
        dvalid = np.zeros((G, Dm), bool)
        route = np.full((G, Km), -1, np.int64)
        legal = np.zeros((G, Km), bool)
        dst = np.full((G, Km), n2, np.int64)
        support = np.zeros((G, Km, Dm), bool)
        WA = np.zeros((G, Km, Dm), np.float32)
        WB = np.zeros((G, Km, Dm), np.float32)
        op_min = np.zeros((G, Km), bool)
        for gl, g in enumerate(gs):
            ss = g_src_list[g]
            src[gl, :len(ss)] = ss
            dvalid[gl, :len(ss)] = True
            g_bucket[g], g_local[g] = b, gl
            rs = np.asarray(g_routes[g], np.int64)
            nk = len(rs)
            route[gl, :nk], legal[gl, :nk], dst[gl, :nk], op_min[gl, :nk] = rs, True, r_dst_a[rs], r_op[rs]
            r_bucket[rs], r_bk_g[rs], r_bk_k[rs] = b, gl, np.arange(nk)
            rep_, off_ = ragged(r_ptr_a[rs + 1] - r_ptr_a[rs])
            ei = r_ptr_a[rs][rep_] + off_
            d_ = np.searchsorted(ss, e_src_a[ei])
            sl_ = e_slot_a[ei]
            support[gl, rep_, d_] = True
            np.add.at(WA[gl], (rep_[sl_ == 0], d_[sl_ == 0]), e_w_a[ei][sl_ == 0])
            np.add.at(WB[gl], (rep_[sl_ == 1], d_[sl_ == 1]), e_w_a[ei][sl_ == 1])
        buckets.append(RouteBucket(family=fam, groups=np.asarray(gs, np.int64), src=src, dvalid=dvalid, route=route, legal=legal, dst=dst,
                                   support=support, WA=WA, WB=WB if op_min.any() else None, op_min=op_min))
    g_src_ptr = np.concatenate([[0], np.cumsum([len(s_) for s_ in g_src_list])]).astype(np.int64)
    g_mem_ptr = np.concatenate([[0], np.cumsum([len(m_) for m_ in mem])]).astype(np.int64)
    params = {"mode": MODE, "bucket_padding_max": pad_max, "bucket_max_elements": max_el, "space_kind": l2.route_space_kind,
              "curve_path_points": int(ro["curve_path_points"])}
    return RouteTable(
        area=area, n_l4=n4, n_l2=n2, family_names=list(l2.route_family_names or []), space_kind=str(l2.route_space_kind), params=params,
        r_dst=r_dst_a, r_group=r_group, r_legal=r_legal_a, r_reason=r_reason, r_op_min=r_op,
        r_ptr=r_ptr_a, e_src=e_src_a, e_slot=e_slot_a, e_w=e_w_a, r_q=r_q, r_dist_px=r_dist,
        r_chord_dev_deg=r_cdev, r_bucket=r_bucket, r_bk_g=r_bk_g, r_bk_k=r_bk_k,
        g_space=np.asarray([k_[0] for k_ in gkeys], np.int64), g_family=np.asarray([k_[1] for k_ in gkeys], np.int64),
        g_n_routes=np.asarray([len(x) for x in g_routes], np.int64), g_D=np.asarray([len(s_) for s_ in g_src_list], np.int64),
        g_bucket=g_bucket, g_local=g_local, g_src_ptr=g_src_ptr,
        g_src=np.concatenate(g_src_list).astype(np.int64) if g_src_list else np.zeros(0, np.int64),
        g_mem_ptr=g_mem_ptr, g_mem=np.concatenate([np.asarray(m_, np.int64) for m_ in mem]) if mem else np.zeros(0, np.int64),
        l2_group=l2_group, l2_n_legal=l2_n_legal, buckets=buckets, removed=dict(l2.ropt_removed or {}))


def route_table_summary(t: RouteTable, l4: LayerSpec, l2: LayerSpec, l3: LayerSpec, batch_size: int) -> dict[str, Any]:
    """structure.json 의 영역 절: 집단·후보·차원·지원·메모리·계산량·fan-out·L2→L3 병합 구조 (모두 구조에서 센 값)."""
    fam_rows: dict[str, Any] = {}
    # L2 → L3 부모 집합 (각 L2 뉴런을 읽는 L3 행들)
    rows3 = np.concatenate([np.repeat(np.arange(l3.n, dtype=np.int64), np.diff(l3.a_ptr)),
                            np.repeat(np.arange(l3.n, dtype=np.int64), np.diff(l3.b_ptr))])
    src3 = np.concatenate([l3.a_src, l3.b_src]).astype(np.int64)
    l3_of: dict[int, tuple[int, ...]] = {}
    order = np.lexsort((rows3, src3))
    for s_, r_ in zip(src3[order], rows3[order]):
        l3_of.setdefault(int(s_), tuple())
        l3_of[int(s_)] = l3_of[int(s_)] + (int(r_),)
    sup_size = np.diff(t.r_ptr)
    for fi, fname in enumerate(t.family_names):
        gsel = np.nonzero(t.g_family == fi)[0]
        rsel = np.nonzero(t.r_legal & (t.r_group >= 0) & np.isin(t.r_group, gsel))[0]
        rall = np.nonzero(np.isin(t.l2_group[t.r_dst], gsel))[0] if t.n_routes else np.zeros(0, np.int64)
        K = t.g_n_routes[gsel]
        D = t.g_D[gsel]
        dsts = np.concatenate([t.group_members(int(g)) for g in gsel]) if len(gsel) else np.zeros(0, np.int64)
        supp = np.concatenate([bk.support.sum(-1)[bk.legal] for bk in t.buckets if bk.family == fi] or [np.zeros(0, np.int64)])
        merge = []
        for g in gsel:
            rr = t.group_routes(int(g))
            if len(rr) == 0:
                continue
            d_ = {int(t.r_dst[r]) for r in rr}
            l3s = {l3_of.get(int(x), ()) for x in d_}
            merge.append((len(rr), len(d_), len(l3s)))
        merge_a = np.asarray(merge, np.int64).reshape(-1, 3)
        fam_rows[fname] = {
            "n_groups": int(len(gsel)), "groups_without_legal_candidate": int((K == 0).sum()),
            "candidates_per_group_K": {"min": int(K.min()) if K.size else 0, "mean": float(K.mean()) if K.size else 0.0, "max": int(K.max()) if K.size else 0},
            "descriptor_dim_D": {"min": int(D.min()) if D.size else 0, "mean": float(D.mean()) if D.size else 0.0, "max": int(D.max()) if D.size else 0},
            "legal_candidates": int(len(rsel)), "rejected_candidates_by_reason": {ROUTE_REASONS[int(c)]: int((t.r_reason[rall] == c).sum())
                                                                                  for c in np.unique(t.r_reason[rall]) if c != 0},
            "destinations": int(len(dsts)), "destinations_without_legal_candidate": int((t.l2_n_legal[dsts] == 0).sum()),
            "support_dim": {"min": int(supp.min()) if supp.size else 0, "mean": float(supp.mean()) if supp.size else 0.0,
                            "max": int(supp.max()) if supp.size else 0},
            "candidates_with_support_1_fixed_code": int((supp == 1).sum()),
            "codebook_floats_sum_KD": int((K * D).sum()),
            "L2_L3_merge_mean_per_group": {"candidates": float(merge_a[:, 0].mean()) if len(merge_a) else 0.0,
                                           "distinct_L2_destinations": float(merge_a[:, 1].mean()) if len(merge_a) else 0.0,
                                           "distinct_L3_parent_sets": float(merge_a[:, 2].mean()) if len(merge_a) else 0.0}}
    KD = int((t.g_n_routes * t.g_D).sum())
    padded = int(sum(bk.G * bk.K * bk.D for bk in t.buckets))
    used = t.e_src[np.repeat(t.r_legal, sup_size)]
    route_fo = np.bincount(used, minlength=t.n_l4)
    desc_fo = np.bincount(t.g_src, minlength=t.n_l4)
    l4_names = l4.channel_names
    return {
        "area": t.area, "mode": MODE, "space_kind": t.space_kind, "families": t.family_names, "n_groups": t.n_groups,
        "n_buckets": len(t.buckets), "n_candidate_records": t.n_routes, "n_legal_candidates": int(t.r_legal.sum()),
        "n_rejected_candidates": int((~t.r_legal).sum()), "n_destinations": t.n_l2,
        "destinations_without_legal_candidate": int((t.l2_n_legal == 0).sum()),
        "groups_without_legal_candidate": int((t.g_n_routes == 0).sum()), "families_detail": fam_rows,
        "sum_K_times_D": KD, "padded_K_times_D": padded, "padding_overhead": (padded / KD) if KD else None,
        "codebook_memory_mb_float32": round(padded * 4 / 2 ** 20, 3), "route_arrays_memory_mb": round(t.nbytes() / 2 ** 20, 3),
        "score_macs_per_sample": KD, "score_macs_per_batch": KD * int(batch_size), "role_op_macs_per_sample_upper": 2 * padded,
        "cost_note_ko": "점수 O(batch × Σ_g K_g D_g), 코드북 메모리 O(Σ_g K_g D_g) (묶음 패딩 포함 값 따로). 원핫 전달이어도 후보 비교 비용은 그대로다.",
        "L4_fanout": {"descriptor_groups_per_L4": {"min": int(desc_fo.min()), "mean": float(desc_fo.mean()), "max": int(desc_fo.max()),
                                                   "n_L4_in_no_group": int((desc_fo == 0).sum())},
                      "legal_candidates_per_L4": {"min": int(route_fo.min()), "mean": float(route_fo.mean()), "max": int(route_fo.max())},
                      "note_ko": "같은 L4 출력값이 여러 집단의 패킷과 여러 후보 슬롯에 그대로 쓰인다 (값을 나누지 않는다, 전역 신호량 보존을 주장하지 않는다)."},
        "L4_channels": l4_names[:64], "removed_curve_options": t.removed.get("curve_total") if t.removed else None,
        "route_table_sha256": t.hash(), "conventions": ROUTE_CONVENTION_KO}


def candidate_records(t: RouteTable, l4: LayerSpec, l2: LayerSpec) -> dict[str, np.ndarray]:
    """후보 레코드 전체 (structure/candidates_{영역}.npz). 명세의 필드를 모두 담는다 (탈락 후보 포함)."""
    e_pos = (t.e_src // l4.n_ch).astype(np.int32)
    l2_pos = t.r_dst // l2.n_ch
    meta_cols = {}
    for k_, v_ in l2.channel_meta.items():
        vals = [v_[int(c)] for c in (t.r_dst % l2.n_ch)]
        meta_cols[f"role_param_{k_}"] = np.asarray([("" if x is None else str(x)) for x in vals])
    return {
        "area": np.asarray(t.area), "route_id": np.arange(t.n_routes, dtype=np.int64), "group_id": t.r_group,
        "destination_L2_id": l2.start + t.r_dst, "destination_L2_local": t.r_dst, "destination_position": l2_pos,
        "destination_xy_px": np.stack([l2.x[t.r_dst], l2.y[t.r_dst]], 1) if t.n_routes else np.zeros((0, 2), np.float32),
        "role_family": np.asarray([t.family_names[int(l2.route_family[int(j)])] for j in t.r_dst]),
        "role_channel": np.asarray([l2.channel_names[int(j) % l2.n_ch] for j in t.r_dst]),
        "op_min2": t.r_op_min, "valid_mask": t.r_legal, "rejection_reason": t.r_reason,
        "rejection_reason_names": np.asarray(dumps(ROUTE_REASONS, indent=None)), "curve_B_position_Q": t.r_q,
        "curve_B_dist_px": t.r_dist_px, "curve_chord_dev_deg_diag": t.r_chord_dev_deg,
        "codebook_bucket": t.r_bucket, "codebook_group_slot": t.r_bk_g, "codebook_candidate_slot": t.r_bk_k,
        "descriptor_schema_id": t.r_group, "edge_ptr": t.r_ptr, "source_L4_id": (l4.start + t.e_src).astype(np.int64),
        "input_slot": t.e_slot, "input_slot_names": np.asarray(SLOT_NAMES), "signed_base_weight": t.e_w,
        "source_position": e_pos, "L4_position_xy_px": np.stack([l4.pos_x, l4.pos_y], 1),
        "source_channel_metadata_code": (t.e_src % l4.n_ch).astype(np.int16), "L4_channel_names": np.asarray(l4.channel_names),
        "L4_channel_meta": np.asarray(dumps(l4.channel_meta, indent=None)),
        "group_space": t.g_space, "group_family": t.g_family, "group_family_names": np.asarray(t.family_names),
        "group_n_candidates": t.g_n_routes, "group_descriptor_dim": t.g_D, "group_source_ptr": t.g_src_ptr,
        "group_source_L4_id": l4.start + t.g_src, "group_member_ptr": t.g_mem_ptr, "group_member_L2_id": l2.start + t.g_mem,
        "route_table_sha256": np.asarray(t.hash()), "conventions": np.asarray(dumps(ROUTE_CONVENTION_KO, indent=None)), **meta_cols}


def route_reference(t: RouteTable, codes: Sequence[np.ndarray], ok: Sequence[np.ndarray], x4: np.ndarray, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """float64 독립 기준식 (묶음 dense 가중치·행렬곱을 쓰지 않고 후보 레코드의 edge 목록에서 직접 계산).
    반환: 집단별 has·route (−1)·top1·gap·S, L2 u (선택된 목적지만), 선택 마스크."""
    ro = cfg["routing"]
    mt, eps = float(ro["routing_min_total"]), float(ro["norm_epsilon"])
    B = int(x4.shape[0])
    xs = np.concatenate([np.asarray(x4, np.float64), np.zeros((B, 1))], 1)
    G = t.n_groups
    has = np.zeros((B, G), bool)
    route = np.full((B, G), -1, np.int64)
    top1 = np.full((B, G), np.nan)
    gap = np.full((B, G), np.nan)
    S_all = np.zeros((B, G))
    u2 = np.zeros((B, t.n_l2))
    mask = np.zeros((B, t.n_l2), bool)
    for g in range(G):
        rr = t.group_routes(g)
        if len(rr) == 0:
            continue
        src = t.group_sources(g)
        b_, gl = int(t.g_bucket[g]), int(t.g_local[g])
        Dg = len(src)
        x = xs[:, src]
        S = x.sum(1)
        S_all[:, g] = S
        fin = np.isfinite(x).all(1) & (x >= 0).all(1)
        act = fin & (S > mt)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = x / np.where(act, S, 1.0)[:, None]
            xh = p / np.maximum(np.linalg.norm(p, axis=1), eps)[:, None]
        for b in range(B):
            if not act[b]:
                continue
            best, bk, sc = -np.inf, -1, []
            for k, r in enumerate(rr):
                if not bool(ok[b_][gl, k]):
                    continue
                s_ = float(xh[b] @ np.asarray(codes[b_][gl, k, :Dg], np.float64))
                sc.append(s_)
                if s_ > best:
                    best, bk = s_, k
            if bk < 0:
                continue
            has[b, g] = True
            route[b, g] = int(rr[bk])
            top1[b, g] = best
            if len(sc) >= 2:
                srt = sorted(sc, reverse=True)
                gap[b, g] = srt[0] - srt[1]
            r = int(rr[bk])
            es, sl, ws = t.route_edges(r)
            ua = float((xs[b, es[sl == 0]] * ws[sl == 0]).sum())
            ub = float((xs[b, es[sl == 1]] * ws[sl == 1]).sum())
            u = min(ua, ub) if bool(t.r_op_min[r]) else ua
            j = int(t.r_dst[r])
            u2[b, j] = u
            mask[b, j] = True
    return {"has": has, "route": route, "top1": top1, "gap": gap, "S": S_all, "u": u2, "mask": mask}


# ======================================================================
# 9. 순방향 계산 (ELL, L6 RelativeActivityController, RatioRouter, CodebookLearner, CortexModel)
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


class RelativeActivityController:
    """L6: 기존 상대 활동 조절만 한다 (방향 선택적 정규화 풀·적응 없음).

        같은 층의 L6 조절 전 CONF 를 모두 계산한 뒤, 그 같은 스냅샷에서 병렬로
        r_j = CONF_j / (mean_{k∈N(j)} CONF_k + ε)
        g_j = clip(1 + κ·(r_j − 1), g_min, g_max)
        transmitted_j = g_j · CONF_j                    (이웃이 없으면 g_j = 1)
    N(j): 같은 층 (같은 Z) 에서 피질 격자 Chebyshev 거리 ≤ R 인 위치의 모든 채널, 기본 자기 자신 제외 (방향 가중치 없음).
    이웃 합 = Σ_{위치∈N} (위치별 채널 합) − CONF_j. 조절된 값을 분모에 다시 넣지 않고 반복하지 않는다.
    라벨·판독기·미래 표본 정보를 쓰지 않으며 q 를 바꾸지 않는다 (g 는 이번 입력의 일시적 값)."""

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
                "uses_labels_or_readout": False, "modifies_q": False, "runs_in_frozen_areas": True,
                "freeze_ko": "영역 동결은 코드북·판독기 갱신 중단이다. L6 g 와 목적지 선택·실제 출력은 입력마다 재계산한다."}


class LayerRuntime:
    """한 층의 장치 쪽 고정 배열. L4·L3 = CSR 입력군의 ELL 묶음, L2 = 입력 비율 라우터 (RatioRouter, 모든 영역 같은 클래스)."""

    def __init__(self, spec: LayerSpec, be: Backend, cfg: dict[str, Any], routed: bool = False):
        self.spec, self.key, self.src_key = spec, spec.key, spec.src_key
        self.n, self.n_pos, self.n_ch = spec.n, spec.n_pos, spec.n_ch
        self.routed = bool(routed)
        if self.routed:                                    # L2 는 CSR 을 직접 계산하지 않는다 (후보 그래프의 선택된 묶음만 전달)
            self.A, self.Bm, self.has_b, self.op_min = None, None, False, None
        else:
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
        self.router: RatioRouter | None = None

    def nbytes(self) -> int:
        b = 0 if self.A is None else self.A.nbytes() + (self.Bm.nbytes() if self.Bm is not None else 0)
        return int(b + self.spec.pos_nb.size * 8 + self.n * 5 + (self.router.nbytes() if self.router is not None else 0))


class RatioRouter:
    """한 영역의 L4 → L2 목적지 선택 (V1·V2·V4·IT 공통 클래스).

        x_g = L4 실제 출력[source_ids[g]] ;  S = Σ x ;  S ≤ min_total → 선택 없음 ;  p = x/S ;  x̂ = p/‖p‖
        score = x̂·c ;  무효 = −∞ ;  k* = argmax (첫 최댓값 = 작은 route_id) ;  u(목적지) = 역할 연산 (기저 가중치 × 실제 L4 값)
    장치 계산: gather (고정 인덱스) → 집단별 배치 행렬곱 (bmm) → 마스크 argmax → 선택 칸 gather → L2 위치로 비교 gather.
    atomic 누적 (index_add·scatter_add) 을 쓰지 않는다. 거대한 [batch, 모든 원천, 모든 목적지, 차원] 텐서를 만들지 않는다 (묶음·집단 조각 단위).
    라벨·클래스 판독기·오차는 인자로 받지 않는다 (함수 서명에 없음)."""

    def __init__(self, table: RouteTable, be: Backend, cfg: dict[str, Any]):
        self.t, self.be = table, be
        ro = cfg["routing"]
        self.min_total = float(ro["routing_min_total"])
        self.eps = float(ro["norm_epsilon"])
        self.near_tie = float(ro["near_tie_gap"])
        self.dev: list[dict[str, Any]] = []
        for bk in table.buckets:
            self.dev.append({"src": be.i64(bk.src), "legal": be.bool_(bk.legal), "dst": be.i64(bk.dst), "support": be.f32(bk.support),
                             "WA": be.f32(bk.WA), "WB": be.f32(bk.WB) if bk.WB is not None else None,
                             "op_min": be.bool_(bk.op_min) if bk.WB is not None else None,
                             "op_f": be.f32(bk.op_min.astype(np.float32)), "G": bk.G, "K": bk.K, "D": bk.D})
        self.C: list[Any] = [be.f32(np.zeros((bk.G, bk.K, bk.D), np.float32)) for bk in table.buckets]
        self.ok: list[Any] = [be.bool_(np.zeros((bk.G, bk.K), bool)) for bk in table.buckets]
        self.initialized = False
        self.version = 0                                                   # 코드북이 바뀔 때마다 1 증가 (모델 버전 표시용)
        offs = np.concatenate([[0], np.cumsum([bk.G for bk in table.buckets])]).astype(np.int64)
        self.offsets = offs
        self.G_total = int(offs[-1])
        gidx = np.full(table.n_l2, self.G_total, np.int64)                # 후보 없는 집단의 목적지 = 더미 열 (항상 선택 없음)
        for b, bk in enumerate(table.buckets):
            for gl, g in enumerate(bk.groups):
                gidx[table.group_members(int(g))] = offs[b] + gl
        self.l2_gidx_np = gidx
        self.l2_gidx = be.i64(gidx)
        self.l2_ids = be.i64(np.arange(table.n_l2, dtype=np.int64))

    # ---------------------------------------------------------------- 코드북
    def set_codebook(self, C: Sequence[np.ndarray], ok: Sequence[np.ndarray]) -> None:
        if len(C) != len(self.t.buckets) or len(ok) != len(self.t.buckets):
            raise CompatibilityError("코드북 묶음 수가 후보 그래프와 다르다.")
        for b, bk in enumerate(self.t.buckets):
            c_ = np.asarray(C[b], np.float32)
            o_ = np.asarray(ok[b], bool)
            if c_.shape != (bk.G, bk.K, bk.D) or o_.shape != (bk.G, bk.K):
                raise CompatibilityError(f"{self.t.area} 묶음 {b} 코드북 모양 {c_.shape} != {(bk.G, bk.K, bk.D)}")
            if not np.isfinite(c_).all() or (c_ < 0).any() or (o_ & ~bk.legal).any():
                raise CompatibilityError(f"{self.t.area} 묶음 {b} 코드북에 NaN·음수 또는 불법 후보의 유효 표시가 있다.")
            self.C[b] = self.be.f32(c_)
            self.ok[b] = self.be.bool_(o_)
        self.initialized = True
        self.version += 1

    def get_codebook(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return ([self.be.to_np(c).astype(np.float32).copy() for c in self.C], [self.be.to_np(o).astype(bool).copy() for o in self.ok])

    def codebook_sha(self) -> str:
        C, ok = self.get_codebook()
        return sha256_array(*(C + ok))

    def nbytes(self) -> int:
        return int(self.t.nbytes() + sum(bk.G * bk.K * bk.D * 4 for bk in self.t.buckets))

    # ---------------------------------------------------------------- 순방향
    def route(self, x4: Any, *, learn: bool = False, detail: bool = False) -> dict[str, Any]:
        """x4: (B, n_L4) 이번 입력의 L4 실제 출력 (비음수). 반환 u (B, n_L2) 와 선택 마스크, 묶음별 기록."""
        be = self.be
        B = int(x4.shape[0])
        xe = be.pad_zero_col(x4)
        recs: list[dict[str, Any]] = []
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for b, d in enumerate(self.dev):
                recs.append(self._bucket(b, d, xe, B, learn, detail))
        dst_parts = [r["dst_sel"] for r in recs] + [be.full_i64((B, 1), -1)]
        u_parts = [r["u_sel"] for r in recs] + [be.zeros((B, 1))]
        dst_all = be.cat1(dst_parts)
        u_allg = be.cat1(u_parts)
        dj = be.take_cols(dst_all, self.l2_gidx)                                  # (B, n_L2): 그 목적지 집단의 선택 목적지
        mask = dj == self.l2_ids
        u = be.where(mask, be.take_cols(u_allg, self.l2_gidx), be.zeros((B, self.t.n_l2)))
        return {"u": u, "mask": mask, "buckets": recs}

    def _bucket(self, b: int, d: dict[str, Any], xe: Any, B: int, learn: bool, detail: bool) -> dict[str, Any]:
        G, K, D = d["G"], d["K"], d["D"]
        per = max(1, B * (3 * D + 7 * K))
        gc = max(1, min(G, self.be.max_gather // per))
        parts = [self._chunk(b, d, xe, B, g0, min(G, g0 + gc), learn, detail) for g0 in range(0, G, gc)]
        if len(parts) == 1:
            return parts[0]
        out: dict[str, Any] = {}
        for k_ in parts[0]:
            out[k_] = self.be.cat0([p[k_] for p in parts]) if k_ in ("wins", "sums") else self.be.cat1([p[k_] for p in parts])
        return out

    def _chunk(self, b: int, d: dict[str, Any], xe: Any, B: int, g0: int, g1: int, learn: bool, detail: bool) -> dict[str, Any]:
        be = self.be
        K = d["K"]
        xd = xe[:, d["src"][g0:g1]]                                                # (B, Gc, D) 실제 L4 출력 (패딩 0)
        fin_e = be.isfinite(xd)
        neg_e = xd < 0
        bad = ~be.all_last(fin_e) | be.any_last(neg_e)
        xd = be.where(fin_e & ~neg_e, xd, be.zeros_like(xd))
        S = be.sum_last(xd)
        active = (~bad) & (S > self.min_total)
        Ss = be.where(active, S, be.ones_like(S))
        p = xd / Ss[..., None]
        pn = be.norm_last(p)
        clamp = active & (pn <= self.eps)
        xhat = p / be.clamp_min(pn, self.eps)[..., None] * be.as_f32(active)[..., None]
        xt = be.permute(xhat, (1, 0, 2))                                          # (Gc, B, D)
        sc = be.permute(be.bmm(xt, be.permute(self.C[b][g0:g1], (0, 2, 1))), (1, 0, 2))     # (B, Gc, K) = x̂·c
        valid = d["legal"][g0:g1] & self.ok[b][g0:g1]
        ninf = be.full(tuple(sc.shape), -math.inf)
        scm = be.where(valid[None], sc, ninf)
        k = be.argmax_last(scm)
        top1 = be.max_last(scm)
        nval = be.sum_last(be.as_f32(valid))
        has = active & (nval > 0)[None]
        oh = be.onehot_last(k, K)
        top2 = be.max_last(be.where(oh > 0, ninf, scm))
        two = has & (nval >= 2)[None]
        nan_ = be.full(tuple(top1.shape), math.nan)
        gap = be.where(two, top1 - top2, nan_)
        top1 = be.where(has, top1, nan_)
        xr = be.permute(xd, (1, 0, 2))
        uA = be.permute(be.bmm(xr, be.permute(d["WA"][g0:g1], (0, 2, 1))), (1, 0, 2))       # 역할 연산 A 합 (부호 있는 기저 가중치 × 실제 값)
        if d["WB"] is not None:
            uB = be.permute(be.bmm(xr, be.permute(d["WB"][g0:g1], (0, 2, 1))), (1, 0, 2))
            uall = be.where(d["op_min"][g0:g1][None], be.minimum(uA, uB), uA)
            uB_sel = be.gather_last(uB, k)
        else:
            uall = uA
            uB_sel = be.zeros(tuple(k.shape))
        zero = be.zeros(tuple(k.shape))
        u_sel = be.where(has, be.gather_last(uall, k), zero)
        dst_sel = be.where(has, be.take_last(d["dst"][g0:g1], k), be.full_i64(tuple(k.shape), -1))
        dom = be.where(active, be.max_last(xd) / Ss, nan_)
        rec = {"k": k, "has": has, "active": active, "bad": bad, "clamp": clamp, "top1": top1, "gap": gap, "S": S, "dom": dom,
               "u_sel": u_sel, "uA_sel": be.where(has, be.gather_last(uA, k), zero), "uB_sel": be.where(has, uB_sel, zero),
               "op_sel": be.where(has, be.take_last(d["op_f"][g0:g1], k), zero), "dst_sel": dst_sel}
        if learn:
            ohm = oh * be.as_f32(has)[..., None]
            rec["wins"] = be.sum0(ohm)                                            # (Gc, K) 유효 승리 수
            rec["sums"] = be.bmm(be.permute(ohm, (1, 2, 0)), xt)                  # (Gc, K, D) 승리한 x̂ 의 합 (평균은 갱신 때)
        if detail:
            rec.update(xhat=xhat, p=p * be.as_f32(active)[..., None], scores=scm, xd=xd, u_all=uall)
        return rec


class CodebookLearner:
    """모든 영역 공통 국소 대표 벡터 갱신 (배치마다 한 번, 같은 배치의 순방향은 갱신 전 코드북을 썼다).
        v = mean(x̂ | k 승리) ⊙ 지원 ;  n > 0, ‖v‖ > ε → c ← normalize((1 − η)c + η·v/‖v‖) ;  그 밖 c 그대로.
    라벨·판독기·오차·미분을 쓰지 않는다. 미선택 후보는 바뀌지 않는다."""

    def __init__(self, router: RatioRouter, cfg: dict[str, Any]):
        self.r, self.be = router, router.be
        self.eta = float(cfg["routing"]["eta_codebook"])
        self.eps = float(cfg["routing"]["norm_epsilon"])

    def update(self, rec_buckets: Sequence[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
        """이번 배치에서 이긴 후보 행만 계산하고, 모든 묶음의 새 코드북을 만든 뒤 한꺼번에 바꾼다 (도중 오류면 이전 코드북 그대로)."""
        be = self.be
        out: list[dict[str, np.ndarray]] = []
        new_C = []
        for b, rec in enumerate(rec_buckets):
            d = self.r.dev[b]
            C = self.r.C[b]
            G, K = d["G"], d["K"]
            n_np = be.to_np(rec["wins"]).astype(np.float64)
            gi, ki = np.nonzero(n_np > 0)
            upd = np.zeros((G, K), bool)
            dl = np.zeros((G, K))
            cs = np.ones((G, K))
            if len(gi) == 0:
                new_C.append(C)
                out.append({"upd": upd, "delta": dl, "cos": cs, "wins": n_np})
                continue
            gt, kt = be.i64(gi), be.i64(ki)
            v = rec["sums"][gt, kt] / be.f32(n_np[gi, ki])[:, None] * d["support"][gt, kt]     # 승리 x̂ 평균 ⊙ 지원
            nv = be.norm_last(v)
            u_ = (nv > self.eps) & d["legal"][gt, kt] & self.r.ok[b][gt, kt]
            vhat = v / be.clamp_min(nv, self.eps)[:, None]
            c0 = C[gt, kt]
            craw = (1.0 - self.eta) * c0 + self.eta * vhat
            cnew = craw / be.clamp_min(be.norm_last(craw), self.eps)[:, None]
            rows = be.where(u_[:, None], cnew, c0)
            if not be.all_finite(rows):
                raise NumericalError(f"{self.r.t.area} 묶음 {b}: 코드북 갱신값에 NaN/Inf (적용하지 않는다)")
            Cn = be.clone(C)
            Cn[gt, kt] = rows                                                      # 서로 다른 칸에만 쓴다 (누적 없음)
            upd[gi, ki] = be.to_np(u_).astype(bool)
            dl[gi, ki] = be.to_np(be.norm_last(rows - c0)).astype(np.float64)
            cs[gi, ki] = be.to_np(be.sum_last(rows * c0)).astype(np.float64)
            new_C.append(Cn)
            out.append({"upd": upd, "delta": dl, "cos": cs, "wins": n_np})
        for b, Cn in enumerate(new_C):
            self.r.C[b] = Cn
        self.r.version += 1
        return out


class CodebookInitializer:
    """학습 단계 시작 시 train 패킷으로 대표 벡터를 만든다 (데이터로부터의 학습 과정, 라벨 미사용, NumPy float64).
    후보마다: 고정 역할 연산 u_k > θ 인 비영 패킷 중 역할 선택도 u_k/S 가 큰 순서로, 같은 집단에서 아직 쓰지 않았고 기존 대표와 방향이
    겹치지 않는 (cos < 1 − tol) 첫 패킷 → c = normalize(x̂ ⊙ 지원). 양의 반응 패킷이 없으면 지원 질량 ‖x̂ ⊙ 지원‖² 최대 패킷 (fallback 기록).
    단위 벡터를 만들 수 없으면 초기화 실패 (경쟁 제외). 강제 균등·반발·무작위 초기화를 하지 않는다."""

    KINDS = {0: "failed", 1: "role_positive_packet", 2: "support_mass_fallback"}

    def __init__(self, router: RatioRouter, cfg: dict[str, Any]):
        ro = cfg["routing"]
        self.r = router
        self.min_total = float(ro["routing_min_total"])
        self.eps = float(ro["norm_epsilon"])
        self.thr = float(cfg["neuron"]["threshold"])
        self.tol = float(ro["init_duplicate_cos_tol"])
        self.alts = int(ro["init_alternatives"])
        self.max_el = int(cfg["compute"]["max_gather_elements"])

    def run(self, X4: np.ndarray) -> dict[str, Any]:
        t = self.r.t
        N = int(X4.shape[0])
        xs = np.concatenate([np.asarray(X4, np.float64), np.zeros((N, 1))], 1)
        if not np.isfinite(xs).all() or (xs < 0).any():
            raise NumericalError(f"{t.area} 초기화 패킷에 NaN/Inf 또는 음수 L4 값이 있다.")
        C_list, ok_list, info = [], [], []
        for bk in t.buckets:
            G, K, D = bk.G, bk.K, bk.D
            C = np.zeros((G, K, D), np.float32)
            ok = np.zeros((G, K), bool)
            rec = {"init_index": np.full((G, K), -1, np.int64), "init_kind": np.zeros((G, K), np.int8),
                   "init_role_selectivity": np.full((G, K), np.nan, np.float32), "dup_packet": np.zeros((G, K), bool),
                   "dup_direction": np.zeros((G, K), bool), "n_role_positive_packets": np.zeros((G, K), np.int64),
                   "group_active_packets": np.zeros(G, np.int64), "group_distinct_directions": np.zeros(G, np.int64)}
            per = max(1, N * (2 * D + 3 * K))
            gc = max(1, min(G, (self.max_el // 2) // per))
            for g0 in range(0, G, gc):
                g1 = min(G, g0 + gc)
                xd = xs[:, bk.src[g0:g1]]                                         # (N, Gc, D)
                S = xd.sum(-1)
                act = S > self.min_total
                p = xd / np.where(act, S, 1.0)[..., None]
                xh = p / np.maximum(np.linalg.norm(p, axis=-1), self.eps)[..., None] * act[..., None]
                xr = np.transpose(xd, (1, 0, 2))
                uA = np.matmul(xr, np.transpose(bk.WA[g0:g1].astype(np.float64), (0, 2, 1)))        # (Gc, N, K)
                if bk.WB is not None:
                    uB = np.matmul(xr, np.transpose(bk.WB[g0:g1].astype(np.float64), (0, 2, 1)))
                    u = np.where(bk.op_min[g0:g1][:, None, :], np.minimum(uA, uB), uA)
                else:
                    u = uA
                for gl in range(g1 - g0):
                    self._group(g0 + gl, xh[:, gl, :], u[gl], S[:, gl], act[:, gl], bk, C, ok, rec)
            C_list.append(C)
            ok_list.append(ok)
            info.append(rec)
        return {"C": C_list, "ok": ok_list, "info": info, "n_samples": N}

    def _group(self, g: int, xh: np.ndarray, u: np.ndarray, S: np.ndarray, act: np.ndarray, bk: RouteBucket,
               C: np.ndarray, ok: np.ndarray, rec: dict[str, np.ndarray]) -> None:
        N = len(S)
        rec["group_active_packets"][g] = int(act.sum())
        sup = bk.support[g].astype(np.float64)                                     # (K, D)
        mass = (xh * xh) @ sup.T                                                    # (N, K) 지원 위 x̂ 에너지
        sel = np.where(act[:, None], u / np.where(act, S, 1.0)[:, None], -np.inf)
        elig = act[:, None] & (u > self.thr) & (mass > 0)
        used = np.zeros(N, bool)
        codes: list[np.ndarray] = []
        distinct = 0

        def unit(n: int, k: int) -> np.ndarray | None:
            v = xh[n] * sup[k]
            nv = float(np.linalg.norm(v))
            return None if nv <= self.eps else v / nv

        for k in range(bk.K):
            if not bk.legal[g, k]:
                continue
            E = elig[:, k]
            nE = int(E.sum())
            rec["n_role_positive_packets"][g, k] = nE
            pick: tuple[int, np.ndarray] | None = None
            kind = 0
            if nE:
                order = np.argsort(-np.where(E, sel[:, k], -np.inf), kind="stable")[:nE]
                kind = 1
            else:
                F = act & (mass[:, k] > 0)
                nF = int(F.sum())
                order = np.argsort(-np.where(F, mass[:, k], -np.inf), kind="stable")[:nF]
                kind = 2 if nF else 0
            if kind:
                fresh = [int(n) for n in order if not used[n]]
                M = np.stack(codes) if codes else None
                for n in fresh[:self.alts]:
                    c = unit(n, k)
                    if c is not None and (M is None or float((M @ c).max()) < 1.0 - self.tol):
                        pick = (n, c)
                        break
                if pick is None:
                    for n in (fresh[:1] or [int(order[0])]):
                        c = unit(n, k)
                        if c is not None:
                            pick = (n, c)
                if pick is None:
                    kind = 0
            rec["init_kind"][g, k] = kind
            if pick is None:
                continue
            n, c = pick
            dupd = bool(codes) and float((np.stack(codes) @ c).max()) >= 1.0 - self.tol
            rec["dup_packet"][g, k] = bool(used[n])
            rec["dup_direction"][g, k] = dupd
            distinct += 0 if dupd else 1
            used[n] = True
            codes.append(c)
            C[g, k] = c.astype(np.float32)
            ok[g, k] = True
            rec["init_index"][g, k] = n
            rec["init_role_selectivity"][g, k] = float(sel[n, k]) if kind == 1 else np.nan
        rec["group_distinct_directions"][g] = distinct


def codebook_init_summary(t: RouteTable, info: Sequence[dict[str, np.ndarray]], ok: Sequence[np.ndarray]) -> dict[str, Any]:
    """계열별 초기화 결과 (후보 수·실패·fallback·중복 패킷·중복 방향·유효 방향 수)."""
    rows: dict[str, Any] = {}
    for fi, fname in enumerate(t.family_names):
        acc = {"n_legal_candidates": 0, "n_code_ok": 0, "n_failed": 0, "n_role_positive": 0, "n_support_mass_fallback": 0,
               "n_duplicate_packet": 0, "n_duplicate_direction": 0, "n_groups": 0, "n_groups_without_active_packet": 0,
               "sum_distinct_directions": 0, "sum_role_positive_packets": 0}
        for b, bk in enumerate(t.buckets):
            if bk.family != fi:
                continue
            r = info[b]
            lg = bk.legal
            acc["n_legal_candidates"] += int(lg.sum())
            acc["n_code_ok"] += int(ok[b].sum())
            acc["n_failed"] += int((lg & ~ok[b]).sum())
            acc["n_role_positive"] += int(((r["init_kind"] == 1) & lg).sum())
            acc["n_support_mass_fallback"] += int(((r["init_kind"] == 2) & lg).sum())
            acc["n_duplicate_packet"] += int((r["dup_packet"] & lg).sum())
            acc["n_duplicate_direction"] += int((r["dup_direction"] & lg).sum())
            acc["n_groups"] += bk.G
            acc["n_groups_without_active_packet"] += int((r["group_active_packets"] == 0).sum())
            acc["sum_distinct_directions"] += int(r["group_distinct_directions"].sum())
            acc["sum_role_positive_packets"] += int(r["n_role_positive_packets"][lg].sum())
        acc["mean_distinct_directions_per_group"] = acc["sum_distinct_directions"] / acc["n_groups"] if acc["n_groups"] else None
        acc["mean_candidates_per_group"] = acc["n_legal_candidates"] / acc["n_groups"] if acc["n_groups"] else None
        acc["mean_role_positive_packets_per_candidate"] = (acc["sum_role_positive_packets"] / acc["n_legal_candidates"]
                                                           if acc["n_legal_candidates"] else None)
        rows[fname] = acc
    return rows


class Connectivity:
    """연결 목록. INPUT = 수신 층 연결, NEXT = 원천 층별로 만든 나가는 연결 CSR (송신 local → 수신 local).
    L4·L3 는 실제 계산 CSR, L2 는 후보 그래프의 합법 후보 edge (실제 전달은 이미지마다 집단당 하나) 에서 만든다."""

    def __init__(self) -> None:
        self.next_csr: dict[str, dict[str, Any]] = {}

    def add_edges(self, src_key: str, dst_key: str, n_src: int, src: np.ndarray, dst: np.ndarray, w: np.ndarray, grp: np.ndarray,
                  kind: str) -> None:
        order = np.argsort(src, kind="stable")
        cnt = np.bincount(src, minlength=n_src)
        self.next_csr[src_key] = {"dst_key": dst_key, "ptr": np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64),
                                  "dst": dst[order], "w": w[order], "grp": grp[order], "kind": kind}

    def add(self, sp: LayerSpec, src_sp: LayerSpec) -> None:
        rows_a = np.repeat(np.arange(sp.n, dtype=np.int64), np.diff(sp.a_ptr))
        rows_b = np.repeat(np.arange(sp.n, dtype=np.int64), np.diff(sp.b_ptr))
        self.add_edges(sp.src_key, sp.key, src_sp.n, np.concatenate([sp.a_src, sp.b_src]), np.concatenate([rows_a, rows_b]),
                       np.concatenate([sp.a_w, sp.b_w]), np.concatenate([np.zeros(len(rows_a), np.int8), np.ones(len(rows_b), np.int8)]),
                       "fixed")

    def add_routes(self, t: RouteTable, src_sp: LayerSpec, dst_key: str) -> None:
        legal_e = np.repeat(t.r_legal, np.diff(t.r_ptr))
        dst = np.repeat(t.r_dst, np.diff(t.r_ptr))
        self.add_edges(src_sp.key, dst_key, src_sp.n, t.e_src[legal_e], dst[legal_e], t.e_w[legal_e], t.e_slot[legal_e], "candidate")

    def fan_out(self, key: str, n: int) -> np.ndarray:
        nx = self.next_csr.get(key)
        return np.zeros(n, np.int64) if nx is None else np.diff(nx["ptr"])


class NeuronStore:
    """모든 뉴런의 연속 저장소.
    CPU (numpy): 좌표 (N,3) float32 [X, Y, Z], 영역·층·역할·채널·위치 코드 (정수), 수용장 크기.
    장치 텐서: 임계값 (N,), q (N,) — 층마다 neuron_id 범위 [start, end) 의 연속 슬라이스 (view). v5 에서 둘 다 고정값이다.
    CONF·전송 출력은 배치마다 계산하는 동적 텐서 (B, n_layer) 이다 (표본 사이에 남기지 않는다).
    NEXT (나가는 연결) 는 원천 층별 CSR (L4 → L2 는 후보 edge), INPUT 은 수신 층 연결이다. 주소는 int64."""

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

    def add_layers(self, specs: Sequence[LayerSpec], table: RouteTable | None = None) -> None:
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
                if sp.name == "L2" and table is not None:
                    self.conn.add_routes(table, src_sp, sp.key)
                else:
                    self.conn.add(sp, src_sp)
        n_new = sum(sp.n for sp in specs)
        self.coords = np.concatenate([self.coords] + coords).astype(np.float32)
        self.area_code = np.concatenate([self.area_code] + ac)
        self.layer_code = np.concatenate([self.layer_code] + lc)
        self.role_code = np.concatenate([self.role_code] + rc)
        self.channel = np.concatenate([self.channel] + chs)
        self.pos_id = np.concatenate([self.pos_id] + ps)
        self.q = self.be.cat0([self.q, self.be.f32(np.full(n_new, float(ne["q"])))])
        self.thr = self.be.cat0([self.thr, self.be.f32(np.full(n_new, float(ne["threshold"])))])

    def q_view(self, key: str) -> Any:
        sp = self.layers[key]
        return self.q[sp.start:sp.end]

    def thr_view(self, key: str) -> Any:
        sp = self.layers[key]
        return self.thr[sp.start:sp.end]

    def q_numpy(self) -> np.ndarray:
        return self.be.to_np(self.q).astype(np.float32).copy()

    def thr_numpy(self) -> np.ndarray:
        return self.be.to_np(self.thr).astype(np.float32).copy()

    def locate(self, nid: int) -> tuple[str, int]:
        if not (0 <= int(nid) < self.N):
            raise IndexError(f"neuron_id {nid} 가 범위 [0,{self.N}) 밖이다.")
        for key in self.order:
            sp = self.layers[key]
            if sp.start <= nid < sp.end:
                return key, int(nid - sp.start)
        raise IndexError(nid)

    def fan_out(self, key: str) -> np.ndarray:
        return self.conn.fan_out(key, self.layers[key].n)


class CortexModel:
    """영역 (V1/V2/V4/IT) 의 층을 순서대로 이어 계산한다. 감각 회로에 재귀 흥분 연결은 없다.
    forward 는 표본마다 동적 상태 (u, s, CONF, g, 전송 출력, 목적지 선택) 를 새로 계산하고 남기지 않는다 (코드북만 지속 파라미터)."""

    def __init__(self, cfg: dict[str, Any], be: Backend, pre: RetinaLGNPreprocessor):
        self.cfg, self.be, self.pre = cfg, be, pre
        self.store = NeuronStore(be, cfg)
        self.specs: dict[str, LayerSpec] = {}
        self.rt: dict[str, LayerRuntime] = {}
        self.areas: list[str] = []
        self.active: str | None = None
        self.struct_hash: dict[str, str] = {}
        self.tables: dict[str, RouteTable] = {}
        self.routers: dict[str, RatioRouter] = {}
        self.area_objs: dict[str, CortexArea] = {}
        self.l6 = RelativeActivityController(cfg, be)
        self.timer: PhaseTimer | None = None

    def plan(self, area: str) -> list[LayerSpec]:
        want = AREA_ORDER[len(self.areas)] if len(self.areas) < len(AREA_ORDER) else None
        if area != want:
            raise StructureError(f"다음에 만들 수 있는 영역은 {want} 이다 (요청 {area}).")
        prev = self.specs.get(f"{self.areas[-1]}/L3") if self.areas else None
        return AREA_CLASSES[area].build(self.cfg, self.pre, prev).specs

    def add_area(self, area: str, specs: Sequence[LayerSpec], table: RouteTable | None = None) -> None:
        if [sp.area for sp in specs] != [area] * 3 or [sp.name for sp in specs] != list(LAYER_ORDER):
            raise StructureError("영역 층 순서는 L4, L2, L3 이다.")
        table = table or build_route_table(area, specs[0], specs[1], self.cfg)
        self.store.add_layers(specs, table)
        self.area_objs[area] = AREA_CLASSES[area](specs)
        for sp in specs:
            self.specs[sp.key] = sp
            self.rt[sp.key] = LayerRuntime(sp, self.be, self.cfg, routed=(sp.name == "L2"))
        router = RatioRouter(table, self.be, self.cfg)
        self.rt[f"{area}/L2"].router = router
        self.tables[area], self.routers[area] = table, router
        self.areas.append(area)
        self.struct_hash[area] = sha256_text(area_structure_hash(specs) + table.hash())

    def layer_keys(self, upto: str | None = None) -> list[str]:
        out = []
        for a in self.areas:
            out += [f"{a}/{n}" for n in LAYER_ORDER]
            if a == upto:
                break
        return out

    def forward(self, lgn: Any, *, upto: str | None = None, keep: Sequence[str] = (), record: Sequence[str] = (),
                learn: str | None = None, detail: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
        """lgn: (B, n_lgn). upto: 이 영역까지 (기본 마지막 영역). 반환: 각 영역 L3 {"out"}, keep 층의 {"u","s","conf","g","out"},
        record·learn·detail 영역의 res["영역/route"] (라우터 기록). learn = 승리 x̂ 합·수를 모을 영역 (갱신은 하지 않는다)."""
        keep_s, rec_s, det_s = set(keep), set(record), set(detail)
        res: dict[str, dict[str, Any]] = {}
        if "LGN" in keep_s:
            res["LGN"] = {"out": lgn}
        prev_key, prev = "LGN", lgn
        for key in self.layer_keys(upto):
            rt = self.rt[key]
            if rt.src_key != prev_key:
                raise StructureError(f"{key} 의 원천 {rt.src_key} 가 직전 층 {prev_key} 와 다르다.")
            area = key.split("/")[0]
            lay = self.layer_step(key, rt, prev, learn=(learn == area), detail=(area in det_s))
            if key in keep_s:
                res[key] = {k_: lay[k_] for k_ in ("u", "s", "conf", "g", "out")}
            elif key.endswith("/L3"):
                res[key] = {"out": lay["out"]}
            if lay["route"] is not None and (area in rec_s or learn == area or area in det_s):
                res[f"{area}/route"] = lay["route"]
            prev_key, prev = key, lay["out"]
        return res

    def layer_step(self, key: str, rt: LayerRuntime, prev: Any, *, learn: bool = False, detail: bool = False) -> dict[str, Any]:
        """한 층의 순방향 (forward 와 검사가 같은 코드를 쓴다). prev = 원천 층 전송 출력 (B, n_src)."""
        be, st = self.be, self.store
        rr = None
        if rt.router is not None:
            gsync = bool(self.cfg["compute"]["gpu_timing_sync"]) and be.is_cuda
            tm = self.timer.phase(f"route_{rt.spec.area}", gpu=gsync) if self.timer is not None else contextlib.nullcontext()
            with tm:
                rr = rt.router.route(prev, learn=learn, detail=detail)
            u = rr["u"]
            s = be.where(rr["mask"], be.relu(u - st.thr_view(key)), be.zeros(tuple(u.shape)))       # 선택되지 않은 목적지: u = s = 0
        else:
            xe = be.pad_zero_col(prev)
            sa = rt.A.apply(be, xe)
            if rt.has_b:
                sb = rt.Bm.apply(be, xe)
                u = be.where(rt.op_min, be.minimum(sa, sb), sa + sb)
            else:
                u = sa
            s = be.relu(u - st.thr_view(key))
        conf = st.q_view(key) * s
        g = self.l6.gain(conf, rt) if rt.l6 else None
        out = conf if g is None else g * conf
        return {"u": u, "s": s, "conf": conf, "g": g, "out": out, "route": rr}

    def device_bytes(self) -> int:
        return int(sum(rt.nbytes() for rt in self.rt.values()) + self.store.N * 8)


def structure_summary(specs: Sequence[LayerSpec], table: RouteTable, be: Backend, cfg: dict[str, Any], n_classes: int,
                      existing_bytes: int = 0) -> dict[str, Any]:
    """모델 생성 전에 보여 주는 뉴런 수·후보 수·코드북 메모리·학습 대상과 고정 항목."""
    B = int(cfg["learning"]["batch_size"])
    rows, total_edges, total_slots, total_n = [], 0, 0, 0
    for sp in specs:
        e = len(sp.a_src) + len(sp.b_src)
        slots = 0
        if sp.name != "L2":
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
    KD = int((table.g_n_routes * table.g_D).sum())
    padded = int(sum(bk.G * bk.K * bk.D for bk in table.buckets))
    cb_mb = padded * 4 / 2 ** 20
    route_mb = table.nbytes() / 2 ** 20
    graph_mb = (total_slots * 12 + total_n * 8) / 2 ** 20
    act_mb = B * total_n * 4 * 6 / 2 ** 20
    gather_mb = int(cfg["compute"]["max_gather_elements"]) * 4 * 3 / 2 ** 20
    learn_mb = padded * 4 / 2 ** 20                                    # 승리 x̂ 합 (활성 영역 학습 중)
    return {"layers": rows, "total_neurons": total_n, "total_edges": total_edges, "ell_slots": total_slots,
            "routing": {"n_groups": table.n_groups, "n_buckets": len(table.buckets), "n_candidate_records": table.n_routes,
                        "n_legal_candidates": int(table.r_legal.sum()), "n_rejected_candidates": int((~table.r_legal).sum()),
                        "groups_without_legal_candidate": int((table.g_n_routes == 0).sum()),
                        "destinations_without_legal_candidate": int((table.l2_n_legal == 0).sum()),
                        "sum_K_times_D": KD, "padded_K_times_D": padded, "score_macs_per_batch": KD * B},
            "memory_mb": {"graph": round(graph_mb, 2), "codebook": round(cb_mb, 2), "route_arrays": round(route_mb, 2),
                          "learning_sums_active_area": round(learn_mb, 2), "batch_activations_upper": round(act_mb, 2),
                          "gather_chunk_upper": round(gather_mb, 2), "existing_model": round(existing_bytes / 2 ** 20, 2),
                          "total_estimate": round(graph_mb + cb_mb + route_mb + learn_mb + act_mb + gather_mb + existing_bytes / 2 ** 20, 2)},
            "learned_items_ko": [f"{specs[0].area} 라우팅 대표 벡터 c[g,k] (합법 후보 {int(table.r_legal.sum())}, 실수 {KD})"],
            "fixed_items_ko": ["후보 그래프·선택 집단·원천 순서", "역할 기저 가중치 (Gabor·윤곽·끝점·결합)", "임계값 θ (= "
                               f"{cfg['neuron']['threshold']})", f"q (= {cfg['neuron']['q']})", "L2→L3·L3→다음 L4 고정 결합", "L6 식·이웃·κ·clip",
                               "아래 영역 (동결된 코드북)"],
            "batch_size": B, "backend": be.kind, "device": be.device,
            "note_ko": "추정치다. 실제 사용량은 장치·조각 크기에서 다르다 (기록의 gpu_max_memory_allocated_mb 를 본다)."}


# ======================================================================
# 10. 분류용 클래스 판독기 (ClassReadout) — 라우팅 코드북과 별개의 통계적 판독기 (train 만으로 계산)
# ======================================================================
class ClassReadout:
    """P_y = μ_y / ‖μ_y‖, μ_y = mean(h | train, label y) (평가 시점마다 고정 모델로 다시 계산). 코드북과 다른 이름·저장 공간·차원이다.
    클래스 평균이 0 이면 그 클래스는 무효 (분류 후보에서 뺀다). 라벨 y 는 이 판독기의 train 통계와 정확도 평가에만 쓴다."""

    def __init__(self, area: str, classes: Sequence[str], dim: int, be: Backend, eps: float):
        self.area, self.classes, self.dim, self.be, self.eps = area, list(classes), int(dim), be, float(eps)
        C = len(self.classes)
        self.P = be.zeros((C, self.dim))
        self.valid = np.zeros(C, bool)
        self.counts = np.zeros(C, np.int64)
        self.norms = np.zeros(C, np.float64)
        self.version = 0
        self.meta: dict[str, Any] = {}

    def fit(self, sums: Any, counts: np.ndarray, *, split: str, sample_ids_sha256: str, epoch_number: int | None,
            codebook_sha256: str) -> dict[str, Any]:
        if split != "train":
            raise RuntimeError("클래스 판독기는 train 출력으로만 만든다 (dev/test 미사용).")
        be = self.be
        counts = np.asarray(counts, np.int64)
        mu = sums * be.f32((1.0 / np.maximum(counts, 1))[:, None])
        norms = be.to_np(be.norm_last(mu)).astype(np.float64)
        valid = (counts > 0) & (norms > self.eps)
        scale = np.where(valid, 1.0 / np.where(norms > 0, norms, 1.0), 0.0)
        self.P = mu * be.f32(scale[:, None])
        self.valid, self.counts, self.norms = valid, counts, norms
        self.version += 1
        self.meta = {"version": self.version, "split": split, "sample_ids_sha256": sample_ids_sha256, "epoch_number": epoch_number,
                     "codebook_sha256": codebook_sha256, "counts": counts.tolist(), "mean_norms": norms.tolist(), "valid": valid.tolist(),
                     "invalid_classes": [c for c, v in zip(self.classes, valid) if not v], "utc": utc_now()}
        return dict(self.meta)

    def sha(self) -> str:
        return sha256_array(self.be.to_np(self.P).astype(np.float32), self.valid)

    def state(self, prefix: str) -> dict[str, np.ndarray]:
        return {f"{prefix}P": self.be.to_np(self.P).astype(np.float32), f"{prefix}valid": self.valid, f"{prefix}counts": self.counts,
                f"{prefix}norms": self.norms}

    def load_state(self, arrays: dict[str, np.ndarray], prefix: str, meta: dict[str, Any]) -> None:
        self.P = self.be.f32(arrays[f"{prefix}P"])
        self.valid = np.asarray(arrays[f"{prefix}valid"], bool)
        self.counts = np.asarray(arrays[f"{prefix}counts"], np.int64)
        self.norms = np.asarray(arrays[f"{prefix}norms"], np.float64)
        self.meta = dict(meta or {})
        self.version = int(self.meta.get("version", 0))

    def copy(self) -> "ClassReadout":
        r = ClassReadout(self.area, self.classes, self.dim, self.be, self.eps)
        r.P, r.valid, r.counts, r.norms = self.be.clone(self.P), self.valid.copy(), self.counts.copy(), self.norms.copy()
        r.version, r.meta = self.version, copy.deepcopy(self.meta)
        return r

    def pairwise_cos(self) -> np.ndarray:
        P = self.be.to_np(self.P).astype(np.float64)
        C = len(self.classes)
        out = np.full((C, C), np.nan)
        if self.valid.any():
            Pv = P[self.valid]
            out[np.ix_(self.valid, self.valid)] = Pv @ Pv.T
        return out


def cosine_classify(be: Backend, h: Any, P: Any, valid_P: np.ndarray, eps: float) -> dict[str, np.ndarray]:
    """pred = argmax_c cos(h, P_c). ‖h‖ ≤ eps 또는 유효 P 없음 → 분류 불능 (−1). 무효 클래스는 후보에서 뺀다. 코사인은 확률이 아니다."""
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
# 11. 라우팅 집계 (전체 데이터) — 표본 몇 개의 기록을 전체 통계처럼 쓰지 않는다
# ======================================================================
ROUTE_STAT_KEYS = ("n_eval", "n_active", "n_has", "n_active_no_sel", "n_bad", "n_clamp", "n_sel_zero_out", "sum_top1", "n_gap", "sum_gap",
                   "n_tie", "n_near_tie", "sum_dom", "n_min2_sel", "n_A_only", "n_B_only", "n_both", "n_neither")
ROUTE_REC_KEYS = ("k", "has", "active", "bad", "clamp", "top1", "gap", "S", "dom", "u_sel", "uA_sel", "uB_sel", "op_sel", "dst_sel")
LAYER_STAT_KEYS = ("n_samples", "n_zero_vec", "n_active", "n_elements", "n_g_max", "n_g_min", "n_g", "n_g_max_active", "n_g_min_active", "sum_norm")


def route_rec_np(be: Backend, recs: Sequence[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
    return [{k: be.to_np(r[k]) for k in ROUTE_REC_KEYS} for r in recs]


def layer_reduce(be: Backend, lay: dict[str, Any], g_min: float, g_max: float, vector: bool) -> dict[str, Any]:
    out, s, g = lay["out"], lay["s"], lay.get("g")
    B, n = int(out.shape[0]), int(out.shape[1])
    nrm = be.norm_last(out)
    r = {"n_samples": B, "n_zero_vec": float(be.to_np(be.as_f32(nrm <= 0).sum())), "n_active": float(be.to_np(be.as_f32(s > 0).sum())),
         "n_elements": B * n, "sum_norm": float(be.to_np(nrm.sum()))}
    if g is not None:
        act = s > 0
        hi, lo = g >= g_max - 1e-7, g <= g_min + 1e-7
        r.update(n_g=B * n, n_g_max=float(be.to_np(be.as_f32(hi).sum())), n_g_min=float(be.to_np(be.as_f32(lo).sum())),
                 n_g_max_active=float(be.to_np(be.as_f32(hi & act).sum())), n_g_min_active=float(be.to_np(be.as_f32(lo & act).sum())))
    if vector:
        r["vsum"] = be.to_np(be.sum0(out)).astype(np.float64)
        r["vsq"] = be.to_np(be.sum0(out * out)).astype(np.float64)
    return r


class RouteStats:
    """영역·묶음별 (G,) 합계와 후보별 승리 수 (G,K), 층별 영벡터·활성·L6 clip·출력 노름 (+ 판독 층 분산). 체크포인트에 저장·복원된다."""

    def __init__(self, model: "CortexModel", areas: Sequence[str]):
        self.model = model
        self.areas = list(areas)
        self.thr = float(model.cfg["neuron"]["threshold"])
        self.near = float(model.cfg["routing"]["near_tie_gap"])
        self.g: dict[str, list[dict[str, np.ndarray]]] = {}
        self.wins: dict[str, list[np.ndarray]] = {}
        self.n_samples: dict[str, int] = {}
        for a in self.areas:
            t = model.tables[a]
            self.g[a] = [{k: np.zeros(bk.G, np.float64) for k in ROUTE_STAT_KEYS} for bk in t.buckets]
            self.wins[a] = [np.zeros((bk.G, bk.K), np.int64) for bk in t.buckets]
            self.n_samples[a] = 0
        self.layers: dict[str, dict[str, Any]] = {}

    def add_route(self, area: str, recs: Sequence[dict[str, np.ndarray]]) -> None:
        B = 0
        for b, r in enumerate(recs):
            acc = self.g[area][b]
            has, act = r["has"].astype(bool), r["active"].astype(bool)
            B = has.shape[0]
            K = self.wins[area][b].shape[1]
            G = has.shape[1]
            acc["n_eval"] += B
            acc["n_active"] += act.sum(0)
            acc["n_has"] += has.sum(0)
            acc["n_active_no_sel"] += (act & ~has).sum(0)
            acc["n_bad"] += r["bad"].astype(bool).sum(0)
            acc["n_clamp"] += r["clamp"].astype(bool).sum(0)
            acc["n_sel_zero_out"] += (has & (r["u_sel"] <= self.thr)).sum(0)
            acc["sum_top1"] += np.where(has, r["top1"], 0.0).sum(0)
            gap = r["gap"].astype(np.float64)
            two = has & np.isfinite(gap)
            acc["n_gap"] += two.sum(0)
            acc["sum_gap"] += np.where(two, gap, 0.0).sum(0)
            acc["n_tie"] += (two & (gap == 0.0)).sum(0)
            acc["n_near_tie"] += (two & (gap < self.near)).sum(0)
            acc["sum_dom"] += np.where(act, r["dom"], 0.0).sum(0)
            m2 = has & (r["op_sel"] > 0.5)
            a_, b_ = r["uA_sel"] > 0, r["uB_sel"] > 0
            acc["n_min2_sel"] += m2.sum(0)
            acc["n_A_only"] += (m2 & a_ & ~b_).sum(0)
            acc["n_B_only"] += (m2 & ~a_ & b_).sum(0)
            acc["n_both"] += (m2 & a_ & b_).sum(0)
            acc["n_neither"] += (m2 & ~a_ & ~b_).sum(0)
            gi = np.broadcast_to(np.arange(G)[None, :], has.shape)
            flat = (gi[has] * K + r["k"][has].astype(np.int64))
            self.wins[area][b] += np.bincount(flat, minlength=G * K).reshape(G, K)
        self.n_samples[area] += B

    def add_layer(self, key: str, red: dict[str, Any]) -> None:
        cur = self.layers.setdefault(key, {k: 0.0 for k in LAYER_STAT_KEYS})
        for k in LAYER_STAT_KEYS:
            cur[k] = cur.get(k, 0.0) + float(red.get(k, 0.0))
        if "vsum" in red:
            cur["vsum"] = cur.get("vsum", 0.0) + red["vsum"]
            cur["vsq"] = cur.get("vsq", 0.0) + red["vsq"]

    # ---------------------------------------------------------------- 저장·복원 (에폭 중간 재개)
    def state(self, prefix: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for a in self.areas:
            out[f"{prefix}ns__{a}"] = np.asarray(self.n_samples[a], np.int64)
            for b, acc in enumerate(self.g[a]):
                for k, v in acc.items():
                    out[f"{prefix}g__{a}__{b}__{k}"] = v
                out[f"{prefix}w__{a}__{b}"] = self.wins[a][b]
        for key, cur in self.layers.items():
            kk = key.replace("/", "_")
            for k in LAYER_STAT_KEYS:
                out[f"{prefix}L__{kk}__{k}"] = np.asarray(cur.get(k, 0.0), np.float64)
            if "vsum" in cur:
                out[f"{prefix}L__{kk}__vsum"] = np.asarray(cur["vsum"], np.float64)
                out[f"{prefix}L__{kk}__vsq"] = np.asarray(cur["vsq"], np.float64)
        return out

    def load_state(self, arrays: dict[str, np.ndarray], prefix: str) -> None:
        for a in self.areas:
            if f"{prefix}ns__{a}" not in arrays:
                continue
            self.n_samples[a] = int(arrays[f"{prefix}ns__{a}"])
            for b, acc in enumerate(self.g[a]):
                for k in acc:
                    acc[k] = np.asarray(arrays[f"{prefix}g__{a}__{b}__{k}"], np.float64).copy()
                self.wins[a][b] = np.asarray(arrays[f"{prefix}w__{a}__{b}"], np.int64).copy()
        for name in arrays:
            if name.startswith(f"{prefix}L__"):
                kk, k = name[len(prefix) + 3:].rsplit("__", 1)
                key = kk.replace("_", "/", 1)
                cur = self.layers.setdefault(key, {})
                cur[k] = np.asarray(arrays[name], np.float64).copy() if k in ("vsum", "vsq") else float(arrays[name])

    # ---------------------------------------------------------------- 요약
    def family_rows(self, area: str) -> dict[str, dict[str, Any]]:
        t = self.model.tables[area]
        router = self.model.routers[area]
        _C, ok = router.get_codebook()
        rows: dict[str, dict[str, Any]] = {}
        for fi, fname in enumerate(t.family_names):
            tot = {k: 0.0 for k in ROUTE_STAT_KEYS}
            ent, eff, mx, unused, nvalid, never, mono, n_used_groups = [], [], [], 0, 0, 0, 0, 0
            for b, bk in enumerate(t.buckets):
                if bk.family != fi:
                    continue
                for k in ROUTE_STAT_KEYS:
                    tot[k] += float(self.g[area][b][k].sum())
                W = self.wins[area][b]
                V = bk.legal & ok[b]
                for gl in range(bk.G):
                    v = V[gl]
                    nv = int(v.sum())
                    if nv == 0:
                        continue
                    w = W[gl][v].astype(np.float64)
                    nvalid += nv
                    unused += int((w == 0).sum())
                    s = float(w.sum())
                    if s <= 0:
                        never += 1
                        continue
                    n_used_groups += 1
                    p = w / s
                    h = float(-(p[p > 0] * np.log(p[p > 0])).sum())
                    ent.append(h)
                    eff.append(math.exp(h))
                    mx.append(float(p.max()))
                    mono += int(p.max() >= 0.9 and s >= 10)
            ne, na, nh = tot["n_eval"], tot["n_active"], tot["n_has"]
            rows[fname] = {
                "group_evaluations": int(ne), "active_input_frac": (na / ne) if ne else None, "selected_frac": (nh / ne) if ne else None,
                "nonzero_input_but_no_selection_frac": (tot["n_active_no_sel"] / na) if na else None,
                "nonfinite_or_negative_frac": (tot["n_bad"] / ne) if ne else None, "norm_clamp_count": int(tot["n_clamp"]),
                "selected_but_zero_output_frac": (tot["n_sel_zero_out"] / nh) if nh else None,
                "mean_max_cosine": (tot["sum_top1"] / nh) if nh else None,
                "mean_1_minus_max_cosine": (1.0 - tot["sum_top1"] / nh) if nh else None,
                "mean_top1_minus_top2": (tot["sum_gap"] / tot["n_gap"]) if tot["n_gap"] else None,
                "exact_tie_frac": (tot["n_tie"] / tot["n_gap"]) if tot["n_gap"] else None,
                "near_tie_frac": (tot["n_near_tie"] / tot["n_gap"]) if tot["n_gap"] else None,
                "mean_dominance_max_over_sum": (tot["sum_dom"] / na) if na else None,
                "min2_selected": int(tot["n_min2_sel"]),
                "min2_A_only_frac": (tot["n_A_only"] / tot["n_min2_sel"]) if tot["n_min2_sel"] else None,
                "min2_B_only_frac": (tot["n_B_only"] / tot["n_min2_sel"]) if tot["n_min2_sel"] else None,
                "min2_both_frac": (tot["n_both"] / tot["n_min2_sel"]) if tot["n_min2_sel"] else None,
                "min2_neither_frac": (tot["n_neither"] / tot["n_min2_sel"]) if tot["n_min2_sel"] else None,
                "valid_candidates": nvalid, "unused_candidate_frac": (unused / nvalid) if nvalid else None,
                "groups_never_selected": never, "mean_usage_entropy_nats": float(np.mean(ent)) if ent else None,
                "mean_effective_used_candidates": float(np.mean(eff)) if eff else None, "mean_max_usage_share": float(np.mean(mx)) if mx else None,
                "monopoly_groups_frac": (mono / n_used_groups) if n_used_groups else None}
        return rows

    def layer_rows(self) -> dict[str, dict[str, Any]]:
        out = {}
        for key, c in self.layers.items():
            ns, ne = c.get("n_samples", 0.0), c.get("n_elements", 0.0)
            row = {"zero_vector_frac": (c["n_zero_vec"] / ns) if ns else None, "active_neuron_frac": (c["n_active"] / ne) if ne else None,
                   "mean_output_norm": (c["sum_norm"] / ns) if ns else None,
                   "L6_g_at_max_frac": (c["n_g_max"] / c["n_g"]) if c.get("n_g") else None,
                   "L6_g_at_min_frac": (c["n_g_min"] / c["n_g"]) if c.get("n_g") else None,
                   "L6_g_at_max_frac_of_active": (c.get("n_g_max_active", 0.0) / c["n_active"]) if c.get("n_g") and c.get("n_active") else None,
                   "L6_g_at_min_frac_of_active": (c.get("n_g_min_active", 0.0) / c["n_active"]) if c.get("n_g") and c.get("n_active") else None}
            if "vsum" in c and ns:
                m = np.asarray(c["vsum"]) / ns
                row["output_variance_mean"] = float(np.maximum(np.asarray(c["vsq"]) / ns - m * m, 0.0).mean())
            out[key] = row
        return out


# ======================================================================
# 12. 3×3 뉴런 조회 (NeuronInspector) — 가변 길이 연결·대표 벡터는 3×3 행렬 밖의 별도 배열 참조로 보인다
# ======================================================================
def _fnum(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def route_group_detail(model: "CortexModel", area: str, g: int, recs: Sequence[dict[str, Any]] | None, i: int = 0,
                       max_items: int = 64) -> dict[str, Any]:
    """집단 g 의 고정 후보 목록·코드북 참조와 (recs 가 있으면) 표본 i 의 입력 비율·후보별 점수·선택·실제 전달값."""
    be = model.be
    t = model.tables[area]
    l4, l2 = model.specs[f"{area}/L4"], model.specs[f"{area}/L2"]
    router = model.routers[area]
    src = t.group_sources(g)
    rr = t.group_routes(g)
    b, gl = int(t.g_bucket[g]), int(t.g_local[g])
    ok = be.to_np(router.ok[b])[gl] if b >= 0 else np.zeros(0, bool)
    C = be.to_np(router.C[b])[gl] if b >= 0 else np.zeros((0, 0))
    cands = []
    for k, r in enumerate(rr[:max_items]):
        es, sl, ws = t.route_edges(int(r))
        d_idx = np.searchsorted(src, es)
        cands.append({"route_id": int(r), "destination_L2_id": int(l2.start + t.r_dst[r]),
                      "destination_channel": l2.channel_names[int(t.r_dst[r]) % l2.n_ch], "op": "min2" if t.r_op_min[r] else "linear",
                      "codebook_ref": {"bucket": b, "group_slot": gl, "candidate_slot": k}, "code_ok": bool(ok[k]),
                      "curve_B_position": int(t.r_q[r]) if t.r_q[r] >= 0 else None,
                      "slots": [{"src_L4_id": int(l4.start + s_), "slot": SLOT_NAMES[int(q_)], "base_weight": float(w_),
                                 "code_value": float(C[k, d_]) if C.size else None} for s_, q_, w_, d_ in zip(es, sl, ws, d_idx)]})
    info: dict[str, Any] = {"area": area, "group_id": int(g), "family": t.family_names[int(t.g_family[g])], "space_key": int(t.g_space[g]),
                            "space_kind": t.space_kind, "descriptor_schema_id": int(g), "descriptor_dim": int(len(src)),
                            "source_L4_ids": [int(l4.start + s_) for s_ in src[:max_items]], "n_candidates": int(len(rr)),
                            "candidates": cands, "n_members": int(len(t.group_members(g)))}
    if recs is not None and b >= 0:
        rec = recs[b]
        D = len(src)
        K = len(rr)
        has = bool(be.to_np(rec["has"])[i, gl])
        k_sel = int(be.to_np(rec["k"])[i, gl])
        ts: dict[str, Any] = {"S": float(be.to_np(rec["S"])[i, gl]), "active": bool(be.to_np(rec["active"])[i, gl]),
                              "dominance_max_over_sum": _fnum(be.to_np(rec["dom"])[i, gl]), "has_selection": has,
                              "selected_route_id": int(rr[k_sel]) if has else None,
                              "selected_destination_L2_id": int(l2.start + t.r_dst[rr[k_sel]]) if has else None,
                              "top1_score": _fnum(be.to_np(rec["top1"])[i, gl]), "top1_minus_top2": _fnum(be.to_np(rec["gap"])[i, gl])}
        if "xhat" in rec:
            xd = be.to_np(rec["xd"])[i, gl, :D]
            ts["input_x"] = [float(v) for v in xd[:max_items]]
            ts["ratio_p"] = [float(v) for v in be.to_np(rec["p"])[i, gl, :D][:max_items]]
            ts["direction_xhat"] = [float(v) for v in be.to_np(rec["xhat"])[i, gl, :D][:max_items]]
            sc = be.to_np(rec["scores"])[i, gl, :K]
            ts["candidate_scores"] = [{"route_id": int(r), "score": _fnum(sc[k]), "selected": has and k == k_sel}
                                      for k, r in enumerate(rr[:max_items])]
            if has:
                r = int(rr[k_sel])
                es, sl, ws = t.route_edges(r)
                xs = dict(zip(src.tolist(), xd.tolist()))
                ts["delivered_edges"] = [{"src_L4_id": int(l4.start + s_), "slot": SLOT_NAMES[int(q_)], "base_weight": float(w_),
                                          "L4_output": float(xs[int(s_)]), "delivered_value": float(w_) * float(xs[int(s_)])}
                                         for s_, q_, w_ in zip(es, sl, ws)]
                ts["u_selected"] = float(be.to_np(rec["u_sel"])[i, gl])
        info["this_sample"] = ts
    return info


class NeuronInspector:
    """inspect_neuron(neuron_id): 3×3 표기와 연결 참조를 저장소에서 복원한다.
    [ X 좌표,         Y 좌표,        Z 좌표          ]
    [ NEXT 참조,      고정 임계값,    INPUT 참조      ]
    [ 실제 전송 출력,  CONF,          q (고정)        ]   (출력·CONF 는 주어진 표본의 동적 값, 없으면 None)
    NEXT/INPUT 은 별도 배열 (후보 그래프·CSR) 의 참조다. 상세 화면에 후보 목록·코드북 참조·입력 비율·후보별 점수·선택·실제 전달값을 붙인다."""

    def __init__(self, model: "CortexModel"):
        self.model = model

    def inspect_neuron(self, nid: int, dyn: dict[str, Any] | None = None, max_edges: int = 40) -> dict[str, Any]:
        m = self.model
        st = m.store
        key, j = st.locate(int(nid))
        sp = st.layers[key]
        area = sp.area
        src_sp = st.layers.get(sp.src_key)
        base = src_sp.start if src_sp is not None else None

        def addr(local: int) -> Any:
            return int(base + local) if base is not None else f"LGN[{int(local)}]"

        t = m.tables.get(area)
        routing: dict[str, Any] | None = None
        if sp.name == "L2":
            rr = np.nonzero(t.r_dst == j)[0]
            g = int(t.l2_group[j])
            inputs = []
            for r in rr:
                es, sl, ws = t.route_edges(int(r))
                inputs += [{"src": addr(s_), "w": float(w_), "group": SLOT_NAMES[int(q_)], "route_id": int(r), "legal": bool(t.r_legal[r])}
                           for s_, q_, w_ in zip(es, sl, ws)]
            routing = {"kind": "L2_destination", "group": route_group_detail(m, area, g, dyn.get("route") if dyn else None, 0),
                       "own_routes": [{"route_id": int(r), "legal": bool(t.r_legal[r]), "reason": ROUTE_REASONS[int(t.r_reason[r])],
                                       "curve_B_position": int(t.r_q[r]) if t.r_q[r] >= 0 else None} for r in rr],
                       "no_geometric_candidate": bool(t.l2_n_legal[j] == 0)}
            inp_ref = (f"INPUT←후보 묶음 {len(rr)} (합법 {int(t.r_legal[rr].sum())}, edge {len(inputs)}) · 집단 {g} "
                       f"({t.family_names[int(t.g_family[g])]}, 후보 {int(t.g_n_routes[g])}, 차원 {int(t.g_D[g])}) ({sp.src_key})")
        else:
            row = sp.row(j)
            inputs = [{"src": addr(s_), "w": float(w_), "group": "A"} for s_, w_ in zip(row["a_src"], row["a_w"])] + \
                     [{"src": addr(s_), "w": float(w_), "group": "B"} for s_, w_ in zip(row["b_src"], row["b_w"])]
            inp_ref = f"INPUT←{len(inputs)} edges ({sp.src_key})"
        nx = st.next_csr.get(key)
        nexts = []
        if nx is not None:
            dsp = st.layers[nx["dst_key"]]
            a_, b_ = int(nx["ptr"][j]), int(nx["ptr"][j + 1])
            nexts = [{"dst": int(dsp.start + d), "w": float(w), "group": SLOT_NAMES[int(g_)]}
                     for d, w, g_ in zip(nx["dst"][a_:b_], nx["w"][a_:b_], nx["grp"][a_:b_])]
        if sp.name == "L4":
            gid_of_src = np.repeat(np.arange(t.n_groups), np.diff(t.g_src_ptr))
            groups = np.unique(gid_of_src[t.g_src == j])
            used = []
            if dyn and dyn.get("route") is not None:
                recs = dyn["route"]
                for g in groups:
                    b, gl = int(t.g_bucket[g]), int(t.g_local[g])
                    if b < 0 or not bool(m.be.to_np(recs[b]["has"])[0, gl]):
                        continue
                    r = int(t.group_routes(int(g))[int(m.be.to_np(recs[b]["k"])[0, gl])])
                    es, sl, ws = t.route_edges(r)
                    for s_, q_, w_ in zip(es, sl, ws):
                        if int(s_) == j:
                            used.append({"group": int(g), "route_id": r, "dst_L2_id": int(m.specs[f"{area}/L2"].start + t.r_dst[r]),
                                         "slot": SLOT_NAMES[int(q_)], "base_weight": float(w_)})
            routing = {"kind": "L4_source", "descriptor_groups": [int(g) for g in groups[:max_edges]], "n_descriptor_groups": int(len(groups)),
                       "n_candidate_edges": len(nexts), "this_sample_used_by": used if dyn else None,
                       "note_ko": "NEXT 는 후보 그래프의 합법 후보 edge 다 (실제 전달은 이미지마다 집단당 선택된 묶음뿐). 같은 출력값이 각 사용처로 간다."}
        q = float(st.be.to_np(st.q[int(nid):int(nid) + 1])[0])
        thr = float(st.be.to_np(st.thr[int(nid):int(nid) + 1])[0])
        conf = trans = None
        if dyn and key in dyn:
            conf = float(dyn[key]["conf"][j]) if "conf" in dyn[key] else None
            trans = float(dyn[key]["out"][j]) if "out" in dyn[key] else None
            if routing is not None and sp.name == "L2":
                routing["this_sample_neuron"] = {"u": float(dyn[key]["u"][j]), "s": float(dyn[key]["s"][j]), "conf": conf, "out": trans,
                                                 "selected": bool(dyn[key].get("mask", np.zeros(sp.n, bool))[j])}
        ch = j % sp.n_ch
        meta = {k: v[ch] for k, v in sp.channel_meta.items()}
        X, Y, Z = (float(v) for v in st.coords[int(nid)])
        nxt_ref = (f"NEXT→{len(nexts)} edges ({nx['dst_key']}, {'후보 edge' if nx.get('kind') == 'candidate' else '고정'})" if nx is not None else
                   ("NEXT→0 (활성 영역 L3: 클래스 판독기로 출력, 연결 아님)" if sp.name == "L3" else "NEXT→0"))
        matrix = [[X, Y, Z], [nxt_ref, thr, inp_ref], [trans, conf, q]]
        return {"neuron_id": int(nid), "layer": key, "local_index": j, "position": j // sp.n_ch, "channel": ch,
                "channel_name": sp.channel_names[ch], "role_tag": sp.channel_roles[ch], "channel_meta": meta,
                "grid_position": [int(sp.ring[j]), int(sp.ang[j])], "receptive_field_px": float(sp.rf[j]), "area": area,
                "layer_name": sp.name, "op": "min2" if bool(sp.op_min[j]) else "linear", "matrix_3x3": matrix,
                "n_input": len(inputs), "n_next": len(nexts), "inputs": inputs[:max_edges], "nexts": nexts[:max_edges], "routing": routing,
                "coordinate_units_ko": "X,Y = 처리 영상 화소 (원점 왼쪽 위, x 오른쪽, y 아래); Z = 영역번호×10+층번호 (무차원)"}

    @staticmethod
    def format(info: dict[str, Any]) -> str:
        m = info["matrix_3x3"]
        lines = [f"neuron {info['neuron_id']}  {info['layer']} local {info['local_index']}  역할 {info['role_tag']}  "
                 f"채널 {info['channel_name']}  연산 {info['op']}",
                 f"[ X={fmt(m[0][0], 2)}, Y={fmt(m[0][1], 2)}, Z={fmt(m[0][2], 1)} ]",
                 f"[ {m[1][0]}, 임계값={fmt(m[1][1])}, {m[1][2]} ]",
                 f"[ 전송 출력={fmt(m[2][0])}, CONF={fmt(m[2][1])}, q={fmt(m[2][2])} (고정) ]",
                 f"  격자 위치 {info['grid_position']}, 수용장 {fmt(info['receptive_field_px'], 2)} px, 채널 메타 {info['channel_meta']}",
                 f"  INPUT (처음 {len(info['inputs'])}/{info['n_input']}): " +
                 ", ".join(f"{e['src']}:{e['w']:+.4f}{e['group']}" for e in info["inputs"][:12]) + (" ..." if info["n_input"] > 12 else ""),
                 f"  NEXT (처음 {len(info['nexts'])}/{info['n_next']}): " +
                 ", ".join(f"{e['dst']}:{e['w']:+.4f}{e['group']}" for e in info["nexts"][:12]) + (" ..." if info["n_next"] > 12 else "")]
        ro = info.get("routing")
        if ro and ro["kind"] == "L2_destination":
            g = ro["group"]
            lines.append(f"  선택 집단 {g['group_id']} ({g['family']}, 공간 키 {g['space_key']} [{g['space_kind']}]), 후보 {g['n_candidates']}, "
                         f"입력 차원 {g['descriptor_dim']}, 기하 후보 없음 {ro['no_geometric_candidate']}")
            lines.append("  이 뉴런의 후보 묶음: " + (", ".join(f"r{c['route_id']}{'' if c['legal'] else '(탈락:' + c['reason'] + ')'}"
                                                      for c in ro["own_routes"][:12]) or "없음"))
            for c in g["candidates"][:8]:
                lines.append(f"    후보 r{c['route_id']} → {c['destination_L2_id']} ({c['destination_channel']}, {c['op']}), 코드북 "
                             f"{c['codebook_ref']} 유효 {c['code_ok']}: " + ", ".join(
                                 f"{s['src_L4_id']}[{s['slot']} w={s['base_weight']:+.3g} c={fmt(s['code_value'], 3)}]" for s in c["slots"][:6])
                             + (" ..." if len(c["slots"]) > 6 else ""))
            ts = g.get("this_sample")
            if ts:
                lines.append(f"  이번 표본: S={fmt(ts['S'])}, 활성 {ts['active']}, 우세도 {fmt(ts['dominance_max_over_sum'])}, 선택 r{ts['selected_route_id']} "
                             f"→ {ts['selected_destination_L2_id']}, 최고 점수 {fmt(ts['top1_score'])}, 1·2 위 차 {fmt(ts['top1_minus_top2'])}")
                if ts.get("ratio_p") is not None:
                    lines.append("  입력 비율 p (앞쪽): " + ", ".join(fmt(v, 3) for v in ts["ratio_p"][:12]))
                    lines.append("  후보별 점수: " + ", ".join(f"r{c['route_id']}={fmt(c['score'], 3)}{'*' if c['selected'] else ''}"
                                                          for c in ts["candidate_scores"][:12]))
                if ts.get("delivered_edges") is not None:
                    lines.append("  실제 전달 (기저 가중치 × L4 출력): " + ", ".join(
                        f"{e['src_L4_id']}[{e['slot']}]={fmt(e['delivered_value'])}" for e in ts["delivered_edges"][:12]))
            ns = ro.get("this_sample_neuron")
            if ns:
                lines.append(f"  이 뉴런: 선택 {ns['selected']}, u={fmt(ns['u'])}, s={fmt(ns['s'])}, 출력={fmt(ns['out'])}")
        elif ro and ro["kind"] == "L4_source":
            lines.append(f"  입력 패킷에 들어간 집단 {ro['n_descriptor_groups']} 개, 후보 edge {ro['n_candidate_edges']} 개 (고정 후보 그래프)")
            if ro.get("this_sample_used_by") is not None:
                lines.append("  이번 표본 실제 사용: " + (", ".join(f"r{u['route_id']}→{u['dst_L2_id']}[{u['slot']}]"
                                                             for u in ro["this_sample_used_by"][:10]) or "없음"))
        return "\n".join(lines)


# ======================================================================
# 13. 세션 상태·기록·checkpoint
# ======================================================================
def new_stage_state(area: str, cfg: dict[str, Any]) -> dict[str, Any]:
    le = cfg["learning"]
    return {"area": area, "status": "not_started", "epochs_completed": 0, "phase": "training", "next_batch": 0,
            "max_epochs": int(le["max_epochs"]), "patience": int(le["patience"]), "min_delta": float(le["min_delta"]),
            "success_dev_accuracy": le["success_dev_accuracy"], "best_dev_accuracy": None, "best_epoch": None, "bad_epochs": 0,
            "history": [], "baseline": None, "transitions": [], "started_utc": None, "ended_utc": None, "budget_changes": [],
            "selected_epoch": None, "selected_restored": False, "detail_epoch_saved": 0, "detail_saved": 0,
            "structure_sha_at_creation": None, "codebook_init": None, "completion": None, "n_nonfinite_events": 0}


TRAINING_CSV = "training_metrics.csv"
ROUTING_CSV = "routing_metrics.csv"
CODEBOOK_CSV = "codebook_metrics.csv"
ROLE_CSV = "role_probe_metrics.csv"
UPD_KEYS = ("n_upd", "sum_delta", "max_delta", "sum_angle", "max_angle")


def empty_update_acc(t: RouteTable) -> list[dict[str, np.ndarray]]:
    return [{"n_upd": np.zeros((bk.G, bk.K), np.int64), "sum_delta": np.zeros((bk.G, bk.K)), "max_delta": np.zeros((bk.G, bk.K)),
             "sum_angle": np.zeros((bk.G, bk.K)), "max_angle": np.zeros((bk.G, bk.K))} for bk in t.buckets]


class Session:
    """한 실행 (run 폴더) 의 상태: 설정·데이터·모델 (코드북)·판독기·단계 상태·동결 해시·평가 횟수·RNG."""

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
        self.readouts: dict[str, ClassReadout] = {}
        self.stage: dict[str, dict[str, Any]] = {}
        self.frozen: dict[str, dict[str, Any]] = {}
        self.counters = {"dev_evaluations": 0, "test_evaluations": 0, "train_passes": 0, "inference_images": 0, "readout_fits": 0,
                         "codebook_updates": 0}
        self.rng = np.random.default_rng([int(self.cfg["seed"]), 99])
        self.ckpt_seq = 0
        self.last_checkpoint: str | None = None
        self.best: dict[str, Any] | None = None                 # 활성 영역의 선택 후보 상태 (에폭 번호·코드북·판독기)
        self.epoch_stats: RouteStats | None = None              # 활성 영역 이번 에폭 학습 배치의 라우팅 집계
        self.upd_acc: list[dict[str, np.ndarray]] | None = None  # 활성 영역 이번 에폭의 코드북 갱신 집계
        self.init_codes: dict[str, list[np.ndarray]] = {}
        self.detail_buf: list[dict[str, Any]] = []
        self.fault: str | None = None                           # 자체 검사용 결함 주입 (평소 None)
        self._last_dev_stats: RouteStats | None = None

    # ---------------------------------------------------------------- 경로·기록
    def set_paths(self, data_root: Path | None, output_root: Path | None) -> None:
        if data_root is not None:
            d = Path(data_root)
            if self.cfg["data"]["source"] == "mnist" and not d.exists():
                d.mkdir(parents=True, exist_ok=True)
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
        sha = self.program_sha()
        r.json("config.json", {"program": PROGRAM, "program_version": VERSION, "program_sha256": sha, "mode": MODE,
                               "config": self.cfg, "config_structure_sha256": config_structure_sha(self.cfg),
                               "config_learning_sha256": config_learning_sha(self.cfg), "formulas": FORMULAS,
                               "base_file": BASE_FILE_INFO, "backend": self.be.describe(), "utc": utc_now(),
                               "numeric_guards_ko": "routing_min_total·norm_epsilon 은 수치 보호값이다 (생물학적 임계값·성능 최적값이 아니다)."})
        r.json("environment.json", DeviceManager.environment(self.be))
        r.text("source.sha256", f"{sha}  {Path(__file__).name}\n")
        r.text("assumptions.md", assumptions_markdown())
        r.json("run_info.json", {"data_root": str(self.data_root) if self.data_root else None,
                                 "output_root": str(self.output_root) if self.output_root else None,
                                 "run_dir": str(r.dir), "created_utc": utc_now(), "program_sha256": sha, "mode": MODE})

    def need_rec(self) -> RunRecorder:
        if self.rec is None:
            raise RuntimeError("실행 폴더가 없다 (메뉴 2 에서 데이터 분할을 먼저 한다).")
        return self.rec

    def log_design_change(self, note: str) -> dict[str, Any]:
        rec = {"utc": utc_now(), "note": note, "model_version": self.model_version() if self.model else None,
               "dev_evaluations_so_far": self.counters["dev_evaluations"], "test_evaluations_so_far": self.counters["test_evaluations"],
               "after_test_seen": self.counters["test_evaluations"] > 0}
        self.need_rec().append_jsonl("design_changes.jsonl", rec)
        return rec

    # ---------------------------------------------------------------- 데이터
    def scan_and_split(self, progress: Callable[[str], None] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.model is not None:
            raise RuntimeError("이미 모델이 있는 세션이다. 다른 분할로 이어 붙이지 않는다 (새 세션에서 다시 시작).")
        if self.data_root is None:
            raise RuntimeError("데이터 폴더를 먼저 정한다 (메뉴 1).")
        dm = make_dataset_manager(self.cfg, self.data_root)
        rep = dm.scan(progress)
        man = dm.split(int(self.cfg["seed"]))
        if self.rec is None:
            self.start_run()
        self._write_data_files(dm, rep, man, None)
        return rep, man

    def reuse_split(self, src_run: Path, progress: Callable[[str], None] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """다른 run (구버전 포함) 의 분할 목록만 재사용한다 (모델 상태는 가져오지 않는다). 데이터 파일을 다시 검증한다."""
        if self.model is not None:
            raise RuntimeError("이미 모델이 있는 세션이다.")
        if self.data_root is None:
            raise RuntimeError("데이터 폴더를 먼저 정한다 (메뉴 1).")
        src = Path(src_run)
        man = read_json(src / "split_manifest.json")
        if man.get("source", "folder") != self.cfg["data"]["source"]:
            raise CompatibilityError(f"분할 목록의 데이터 종류 {man.get('source', 'folder')} 가 현재 설정 {self.cfg['data']['source']} 와 다르다.")
        if int(man.get("seed", -1)) != int(self.cfg["seed"]):
            raise CompatibilityError(f"분할 목록의 seed {man.get('seed')} 가 현재 설정 seed {self.cfg['seed']} 와 다르다 (--seed 로 맞춘다).")
        if man.get("source") == "mnist" and int(man.get("mnist_dev_size", -1)) != int(self.cfg["data"]["mnist_dev_size"]):
            raise CompatibilityError("분할 목록의 MNIST dev 크기가 현재 설정과 다르다.")
        dm = DatasetManager.from_manifest(self.cfg, self.data_root, man)
        if progress:
            progress("  분할 목록의 데이터 파일을 다시 검증한다 (SHA256 / MNIST IDX 체크섬·분할 재현)...")
        probs = dm.verify_files()
        if probs:
            raise CompatibilityError("데이터가 분할 목록과 다르다: " + "; ".join(probs[:5]))
        rep_p = src / "data_report.json"
        rep = read_json(rep_p) if rep_p.is_file() else {"note_ko": "원본 run 에 data_report.json 이 없다."}
        if self.rec is None:
            self.start_run()
        src_cfg = read_json(src / "config.json") if (src / "config.json").is_file() else {}
        prov = {"source_run": str(src.resolve()), "source_program": src_cfg.get("program"), "source_program_version": src_cfg.get("program_version"),
                "split_sha256": man.get("split_sha256"), "data_fingerprint_sha256": man.get("data_fingerprint_sha256"),
                "manifest_sha256": sha256_file(src / "split_manifest.json"), "utc": utc_now(),
                "note_ko": "분할 목록·데이터 경로만 재사용했다. 모델 상태 (코드북·판독기·q·prototype) 는 가져오지 않았다 (구버전 checkpoint 를 이어 학습으로 읽지 않음)."}
        self._write_data_files(dm, rep, man, prov)
        return rep, man

    def _write_data_files(self, dm: DatasetManager, rep: dict[str, Any], man: dict[str, Any], prov: dict[str, Any] | None) -> None:
        r = self.need_rec()
        r.json("data_report.json", rep)
        r.json("split_manifest.json", man)
        r.json("class_to_index.json", {"class_to_index": man["class_to_index"], "classes": man["classes"],
                                       "data_fingerprint_sha256": man["data_fingerprint_sha256"], "seed": man["seed"]})
        info = {**read_json(r.p("run_info.json")), "data_root": str(self.data_root)}
        if prov:
            info["reused_split_from"] = prov
        r.json("run_info.json", info)
        r.json("preprocessing.json", {"info": self.pre.info(), "dc_check": self.pre.dc_check()})
        self.data = dm
        self.loader = BatchLoader(dm, self.pre, self.be, self.cfg, self.timer)
        r.info(f"데이터 분할: 클래스 {man['classes']}, 항목 {man['n_items']}, 지문 {man['data_fingerprint_sha256'][:12]}"
               + (f" (재사용: {prov['source_run']})" if prov else ""))

    @property
    def classes(self) -> list[str]:
        if self.data is None:
            raise RuntimeError("데이터 분할이 없다.")
        return list(self.data.classes)

    # ---------------------------------------------------------------- 모델 생성
    def plan_next(self) -> tuple[str, list[LayerSpec], RouteTable, dict[str, Any]]:
        if self.data is None:
            raise RuntimeError("데이터 분할을 먼저 한다 (메뉴 2).")
        if self.model is None:
            self.model = CortexModel(self.cfg, self.be, self.pre)
            self.model.timer = self.timer
        if len(self.model.areas) >= len(AREA_ORDER):
            raise RuntimeError("IT 까지 모두 만들었다.")
        area = AREA_ORDER[len(self.model.areas)]
        specs = self.model.plan(area)
        table = build_route_table(area, specs[0], specs[1], self.cfg)
        summ = structure_summary(specs, table, self.be, self.cfg, len(self.classes), self.model.device_bytes())
        return area, specs, table, summ

    def build_area(self, area: str, specs: list[LayerSpec], table: RouteTable, summary: dict[str, Any]) -> dict[str, Any]:
        r = self.need_rec()
        acc0 = dict(self.loader.access)
        self.model.add_area(area, specs, table)
        self.model.active = area
        self.stage[area] = new_stage_state(area, self.cfg)
        self.stage[area]["structure_sha_at_creation"] = self.model.struct_hash[area]
        r.json(f"structure/{area}_plan_summary.json", summary)
        self.write_structure(area)
        init = self.init_codebook(area)
        acc1 = dict(self.loader.access)
        self.best = None
        self.epoch_stats, self.upd_acc = None, None
        row = self.epoch_evaluation(area, 0, "init_baseline")
        st = self.stage[area]
        row.update(stage_status="created", best_dev_accuracy_so_far=row["dev_accuracy"], patience_counter=0, selected_epoch_so_far=0)
        st.update(baseline=row, best_dev_accuracy=row["dev_accuracy"], best_epoch=0)
        self.update_best(area, 0, row)
        self.write_training_row(row)
        acc2 = dict(self.loader.access)
        r.json(f"structure/{area}_data_access.json", {
            "utc": utc_now(), "area": area, "before_creation": acc0, "after_codebook_init": acc1, "after_baseline_eval": acc2,
            "codebook_init_delta": {k: acc1.get(k, 0) - acc0.get(k, 0) for k in acc1},
            "baseline_eval_delta": {k: acc2.get(k, 0) - acc1.get(k, 0) for k in acc2},
            "rule_ko": "후보 그래프·집단은 구조만으로 만든다 (데이터 미사용). 코드북 초기화는 train 만, 판독기는 train 만, 기준 평가는 train·dev 만 읽는다 (test 0)."})
        if self.cfg["records"]["role_probe_after_init"]:
            with contextlib.suppress(Exception):
                RoleProbes(self).run(context=f"{area}_after_init")
        self.save_checkpoint(f"{area}_created", keep=True)
        r.info(f"{area} 생성: 구조 해시 {self.model.struct_hash[area][:12]}, 코드북 초기화 {init['n_samples']} 표본", echo=False)
        return {"init": init, "baseline": row}

    def write_structure(self, area: str) -> None:
        """structure.json (모든 영역 누적) 과 structure/candidates_{영역}.npz (후보 레코드 전체)."""
        r = self.need_rec()
        m = self.model
        t = m.tables[area]
        l4, l2, l3 = (m.specs[f"{area}/{x}"] for x in LAYER_ORDER)
        r.npz(f"structure/candidates_{area}.npz", **candidate_records(t, l4, l2))
        arrs: dict[str, np.ndarray] = {}
        for sp in (l4, l2, l3):
            for nm in ("a_ptr", "a_src", "a_w", "b_ptr", "b_src", "b_w", "op_min", "x", "y", "rf", "pos_nb", "pos_nb_count"):
                arrs[f"{sp.name}__{nm}"] = getattr(sp, nm)
        r.npz(f"structure/{area}_layers.npz", **arrs)
        p = r.p("structure.json")
        cur = read_json(p) if p.is_file() else {"program": PROGRAM, "program_version": VERSION, "mode": MODE, "areas": {}}
        layers = {}
        for sp in (l4, l2, l3):
            rt = m.rt[sp.key]
            layers[sp.name] = {"layer": sp.key, "neuron_id_range": [sp.start, sp.end], "n_neurons": sp.n, "n_positions": sp.n_pos,
                               "n_channels": sp.n_ch, "grid_shape": list(sp.grid_shape) if sp.grid_shape else None, "source": sp.src_key,
                               "channels": sp.channel_names, "roles": sorted(set(sp.channel_roles)), "description": sp.description,
                               "build_stats": sp.build_stats, "l6_applied": rt.l6, "l6_neurons_without_neighbor": rt.n_no_neighbor,
                               "structure_sha256": sp.hash()}
        summ = route_table_summary(t, l4, l2, l3, int(self.cfg["learning"]["batch_size"]))
        cur["areas"][area] = {"layers": layers, "routing": summ, "structure_sha256": m.struct_hash[area],
                              "dimension_table": AREA_DIMENSION_TABLE[area], "learned_items_ko": ["라우팅 대표 벡터 c[g,k] (이 영역이 활성일 때만)"],
                              "fixed_items_ko": ["후보 그래프·집단·원천 순서", "역할 기저 가중치", "임계값", "q", "L2→L3·L3→다음 L4 결합", "L6"]}
        cur.update(total_neurons=m.store.N, areas_built=m.areas, formulas=FORMULAS, conventions=ROUTE_CONVENTION_KO, utc=utc_now(),
                   l6=m.l6.describe(), not_in_this_program_ko=BASE_FILE_INFO["removed_ko"])
        r.json("structure.json", cur)

    # ---------------------------------------------------------------- 코드북 초기화 (train 만, 라벨 미사용)
    def init_codebook(self, area: str) -> dict[str, Any]:
        r = self.need_rec()
        ro = self.cfg["routing"]
        train = self.data.items("train")
        n = min(int(ro["init_samples"]), len(train))
        rng = np.random.default_rng([int(self.cfg["seed"]), int(ro["init_seed_offset"]), AREA_INDEX[area]])
        idx = np.sort(rng.choice(len(train), size=n, replace=False))
        items = [train[int(i)] for i in idx]
        B = int(self.cfg["learning"]["batch_size"])
        key4 = f"{area}/L4"
        X4 = []
        t0 = time.perf_counter()
        for a in range(0, n, B):
            lgn, _y = self.loader.batch(items[a:a + B], "train")
            res = self.model.forward(lgn, upto=area, keep=(key4,))
            X4.append(self.be.to_np(res[key4]["out"]).astype(np.float32))
        X4a = np.concatenate(X4) if X4 else np.zeros((0, self.model.specs[key4].n), np.float32)
        router = self.model.routers[area]
        out = CodebookInitializer(router, self.cfg).run(X4a)
        router.set_codebook(out["C"], out["ok"])
        self.init_codes[area] = [c.copy() for c in out["C"]]
        t = self.model.tables[area]
        summ = codebook_init_summary(t, out["info"], out["ok"])
        arrays: dict[str, np.ndarray] = {"sample_index_in_train": idx.astype(np.int64), "sample_id": np.asarray([it["relpath"] for it in items]),
                                         "kinds": np.asarray(dumps(CodebookInitializer.KINDS, indent=None))}
        for b in range(len(t.buckets)):
            arrays[f"b{b}__C"] = out["C"][b]
            arrays[f"b{b}__ok"] = out["ok"][b]
            arrays[f"b{b}__route_id"] = t.buckets[b].route
            for k_, v_ in out["info"][b].items():
                arrays[f"b{b}__{k_}"] = v_
        r.npz(f"codebook/init_{area}.npz", **arrays)
        meta = {"utc": utc_now(), "area": area, "split": "train", "labels_used": False, "dev_used": False, "test_used": False,
                "n_samples": n, "n_train": len(train), "seed": [int(self.cfg["seed"]), int(ro["init_seed_offset"]), AREA_INDEX[area]],
                "sample_ids_sha256": sha256_text("\n".join(it["relpath"] for it in items)),
                "method_ko": FORMULAS["codebook_init"], "kinds": CodebookInitializer.KINDS, "families": summ,
                "seconds": round(time.perf_counter() - t0, 3), "codebook_sha256": router.codebook_sha(),
                "note_ko": "이 초기화도 train 데이터로부터 대표 벡터를 정한 학습 과정이다 (라벨은 쓰지 않았다). 중복·실패·fallback 수를 그대로 기록한다."}
        r.json(f"codebook/init_{area}.json", meta)
        for fam, row in summ.items():
            r.append_csv(CODEBOOK_CSV, {"utc": meta["utc"], "area": area, "epoch_number": 0, "context": "init", "family": fam, **row})
        self.stage[area]["codebook_init"] = {"n_samples": n, "sample_ids_sha256": meta["sample_ids_sha256"], "families": summ,
                                             "codebook_sha256": meta["codebook_sha256"]}
        return {"n_samples": n, "families": summ, "seconds": meta["seconds"]}

    # ---------------------------------------------------------------- 순방향 패스 (갱신 없음)
    def can_store_h(self, area: str, split: str = "train") -> bool:
        n = len(self.data.items(split))
        dim = self.model.specs[f"{area}/L3"].n
        return n * dim * 4 <= int(self.cfg["records"]["store_train_h_max_mb"]) * 2 ** 20

    def pass_split(self, area: str, split: str, *, readout: ClassReadout | None = None, collect_h: bool = False,
                   stats_areas: Sequence[str] = (), items: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
        """split 전체 순방향 (코드북·판독기 갱신 없음). 클래스 합·개수, (readout 이 있으면) 분류·혼동행렬, (stats_areas) 라우팅·층 집계."""
        be = self.be
        items = list(items) if items is not None else self.data.items(split)
        B = int(self.cfg["learning"]["batch_size"])
        key = f"{area}/L3"
        dim = self.model.specs[key].n
        C = len(self.classes)
        eps = float(self.cfg["learning"]["eps_norm"])
        sums = be.zeros((C, dim))
        counts = np.zeros(C, np.int64)
        out: dict[str, Any] = {"n": 0, "correct": 0, "n_valid_h": 0, "n_unclassifiable": 0, "n_ties": 0, "cos_true_sum": 0.0, "n_cos": 0,
                               "margin_sum": 0.0, "n_margin": 0, "per_class_n": np.zeros(C, np.int64), "per_class_correct": np.zeros(C, np.int64),
                               "confusion": np.zeros((C, C + 1), np.int64), "n_nonfinite_h": 0}
        stats = RouteStats(self.model, stats_areas) if stats_areas else None
        keep = tuple(f"{a}/{l}" for a in stats_areas for l in ("L2", "L3")) if stats else ()
        l6 = self.model.l6
        H, Y, PR = [], [], []
        for a0 in range(0, len(items), B):
            chunk = items[a0:a0 + B]
            lgn, y = self.loader.batch(chunk, split)
            with self.timer.phase("forward", gpu=be.is_cuda and bool(self.cfg["compute"]["gpu_timing_sync"])):
                res = self.model.forward(lgn, upto=area, keep=keep, record=tuple(stats_areas))
            h = res[key]["out"] if key in res else None
            if stats is not None:
                for a in stats_areas:
                    stats.add_route(a, route_rec_np(be, res[f"{a}/route"]["buckets"]))
                for k_ in keep:
                    stats.add_layer(k_, layer_reduce(be, res[k_], l6.g_min, l6.g_max, vector=(k_ == key)))
            if not be.all_finite(h):
                out["n_nonfinite_h"] += 1
            for c in np.unique(y):
                be.add_row_(sums, int(c), be.sum0(be.rows(h, be.i64(np.nonzero(y == c)[0]))))
            counts += np.bincount(y, minlength=C)
            out["n"] += len(chunk)
            np.add.at(out["per_class_n"], y, 1)
            if readout is not None:
                self._classify_into(out, be, h, y, readout, eps)
                PR.append(out.pop("_last_pred"))
            else:
                out["n_valid_h"] += int((be.to_np(be.norm_last(h)) > eps).sum())
            if collect_h:
                H.append(be.to_np(h).astype(np.float32))
                Y.append(y)
        if split == "train":
            self.counters["train_passes"] += 1
        out.update(sums=sums, counts=counts, ids_sha=sha256_text("\n".join(sorted(it["relpath"] for it in items))),
                   H=np.concatenate(H) if H else None, Y=np.concatenate(Y) if Y else None, pred=np.concatenate(PR) if PR else None,
                   stats=stats)
        return out

    @staticmethod
    def _classify_into(out: dict[str, Any], be: Backend, h: Any, y: np.ndarray, readout: ClassReadout, eps: float) -> None:
        C = len(readout.classes)
        cl = cosine_classify(be, h, readout.P, readout.valid, eps)
        pred = cl["pred"]
        np.add.at(out["confusion"], (y, np.where(pred >= 0, pred, C)), 1)
        ok = pred == y
        out["correct"] += int(ok.sum())
        np.add.at(out["per_class_correct"], y[ok], 1)
        out["n_valid_h"] += int(cl["valid_h"].sum())
        out["n_unclassifiable"] += int((pred < 0).sum())
        out["n_ties"] += int(cl["n_ties"])
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
        out["_last_pred"] = pred

    def classify_stored(self, H: np.ndarray, Y: np.ndarray, readout: ClassReadout) -> dict[str, Any]:
        """저장한 train 출력 H 로 분류 (두 번째 train 순방향 대신, 같은 float32 값)."""
        be = self.be
        C = len(self.classes)
        out: dict[str, Any] = {"n": int(len(Y)), "correct": 0, "n_valid_h": 0, "n_unclassifiable": 0, "n_ties": 0, "cos_true_sum": 0.0,
                               "n_cos": 0, "margin_sum": 0.0, "n_margin": 0, "per_class_n": np.bincount(Y, minlength=C).astype(np.int64),
                               "per_class_correct": np.zeros(C, np.int64), "confusion": np.zeros((C, C + 1), np.int64)}
        eps = float(self.cfg["learning"]["eps_norm"])
        step = 4096
        for a in range(0, len(Y), step):
            self._classify_into(out, be, be.f32(H[a:a + step]), Y[a:a + step], readout, eps)
            out.pop("_last_pred")
        return out

    def metrics_from(self, r: dict[str, Any], split: str, purpose: str, readout: ClassReadout) -> dict[str, Any]:
        n = r["n"]
        return {"split": split, "purpose": purpose, "n": n, "correct": r["correct"], "accuracy_overall": (r["correct"] / n) if n else None,
                "n_unclassifiable": r["n_unclassifiable"], "unclassifiable_frac": (r["n_unclassifiable"] / n) if n else None,
                "n_classifiable": n - r["n_unclassifiable"],
                "accuracy_classifiable_only": (r["correct"] / (n - r["n_unclassifiable"])) if (n - r["n_unclassifiable"]) else None,
                "n_ties": r["n_ties"], "mean_cos_to_true_class": (r["cos_true_sum"] / r["n_cos"]) if r["n_cos"] else None,
                "mean_margin": (r["margin_sum"] / r["n_margin"]) if r["n_margin"] else None,
                "per_class": {c: {"n": int(r["per_class_n"][i]), "correct": int(r["per_class_correct"][i])} for i, c in enumerate(self.classes)},
                "per_class_accuracy": {c: (int(r["per_class_correct"][i]) / int(r["per_class_n"][i])) if r["per_class_n"][i] else None
                                       for i, c in enumerate(self.classes)},
                "confusion": r["confusion"].tolist(), "confusion_columns": self.classes + ["unclassifiable"],
                "readout_version": readout.version, "readout_invalid_classes": [c for c, v in zip(self.classes, readout.valid) if not v]}

    # ---------------------------------------------------------------- 판독기·평가
    def fit_readout(self, area: str, epoch_number: int | None, *, train_accuracy: bool = True) -> tuple[ClassReadout, dict[str, Any], dict[str, Any] | None]:
        """고정 모델로 train 클래스 평균을 다시 계산한다 (dev/test 미사용). train 정확도는 같은 판독기로 (저장한 H 또는 두 번째 순방향)."""
        store_h = train_accuracy and self.can_store_h(area)
        with self.timer.phase("readout_fit"):
            tp = self.pass_split(area, "train", collect_h=store_h)
        ro = ClassReadout(area, self.classes, self.model.specs[f"{area}/L3"].n, self.be, float(self.cfg["learning"]["eps_norm"]))
        meta = ro.fit(tp["sums"], tp["counts"], split="train", sample_ids_sha256=tp["ids_sha"], epoch_number=epoch_number,
                      codebook_sha256=self.model.routers[area].codebook_sha())
        prev = self.readouts.get(area)
        ro.version = (prev.version if prev is not None else 0) + 1
        meta["version"] = ro.meta["version"] = ro.version
        self.counters["readout_fits"] += 1
        tr = None
        if train_accuracy:
            with self.timer.phase("train_accuracy"):
                tr = self.classify_stored(tp["H"], tp["Y"], ro) if store_h else self.pass_split(area, "train", readout=ro)
            tr = self.metrics_from(tr, "train", f"{area}_e{epoch_number}_train_readout", ro)
            tr["train_h_stored_in_memory"] = bool(store_h)
        meta["train_zero_vector_frac"] = 1.0 - tp["n_valid_h"] / max(1, tp["n"])
        return ro, meta, tr

    def evaluate(self, area: str, split: str, readout: ClassReadout, *, purpose: str, stats: bool = True, log: bool = True) -> dict[str, Any]:
        if split == "dev":
            self.counters["dev_evaluations"] += 1
        elif split == "test":
            self.counters["test_evaluations"] += 1
        areas = tuple(self.model.areas[:self.model.areas.index(area) + 1]) if stats else ()
        r = self.pass_split(area, split, readout=readout, stats_areas=areas)
        self._last_dev_stats = r["stats"] if split == "dev" else getattr(self, "_last_dev_stats", None)
        m = self.metrics_from(r, split, purpose, readout)
        m["model_version"] = self.model_version(area)
        m["n_nonfinite_batches"] = r["n_nonfinite_h"]
        if r["stats"] is not None:
            m["routing"] = {a: r["stats"].family_rows(a) for a in areas}
            m["layers"] = r["stats"].layer_rows()
        if log and self.rec is not None:
            self.rec.append_jsonl("evaluations.jsonl", {"utc": utc_now(), **{k: v for k, v in m.items() if k not in ("per_class", "routing", "layers")},
                                                        "dev_evaluations": self.counters["dev_evaluations"],
                                                        "test_evaluations": self.counters["test_evaluations"]})
        return m

    def epoch_evaluation(self, area: str, epoch_number: int, row_kind: str) -> dict[str, Any]:
        """에폭 끝 (또는 에폭 0 기준): 같은 코드북 스냅샷으로 train 판독기 재계산 → train·dev 평가. dev 에는 갱신이 없다."""
        r = self.need_rec()
        ro, meta, tr = self.fit_readout(area, epoch_number, train_accuracy=bool(self.cfg["records"]["train_accuracy_each_epoch"]))
        self.readouts[area] = ro
        r.npz(f"readout/{area}_e{epoch_number:03d}.npz", **ro.state(""), classes=np.asarray(self.classes))
        r.json(f"readout/{area}_e{epoch_number:03d}.json", meta)
        with self.timer.phase("dev_eval"):
            dv = self.evaluate(area, "dev", ro, purpose=f"{area}_e{epoch_number}_dev")
        st = self.stage[area]
        dl = (dv.get("layers") or {}).get(f"{area}/L3", {})
        dl2 = (dv.get("layers") or {}).get(f"{area}/L2", {})
        row: dict[str, Any] = {
            "utc": utc_now(), "area": area, "epoch_number": epoch_number, "row_kind": row_kind,
            "row_kind_ko": ("초기화 코드북 + train 판독기 (코드북 갱신 전 기준, 라벨은 판독기 통계에만 사용)" if row_kind == "init_baseline"
                            else "에폭 학습 뒤 같은 코드북 스냅샷의 train 판독기 재계산 + 평가"),
            "model_version": self.model_version(area), "readout_version": ro.version, "codebook_version": self.model.routers[area].version,
            "train_n": tr["n"] if tr else None, "train_correct": tr["correct"] if tr else None,
            "train_accuracy": tr["accuracy_overall"] if tr else None, "train_unclassifiable": tr["n_unclassifiable"] if tr else None,
            "dev_n": dv["n"], "dev_correct": dv["correct"], "dev_accuracy": dv["accuracy_overall"],
            "dev_unclassifiable": dv["n_unclassifiable"], "dev_unclassifiable_frac": dv["unclassifiable_frac"],
            "dev_accuracy_classifiable_only": dv["accuracy_classifiable_only"], "dev_mean_margin": dv["mean_margin"],
            "dev_mean_cos_to_true_class": dv["mean_cos_to_true_class"], "dev_n_ties": dv["n_ties"],
            "readout_invalid_classes": meta["invalid_classes"], "train_zero_vector_frac": meta["train_zero_vector_frac"],
            "dev_output_norm_mean": dl.get("mean_output_norm"), "dev_output_variance_mean": dl.get("output_variance_mean"),
            "dev_L3_zero_vector_frac": dl.get("zero_vector_frac"), "dev_L3_active_frac": dl.get("active_neuron_frac"),
            "dev_L2_zero_vector_frac": dl2.get("zero_vector_frac"), "dev_L2_active_frac": dl2.get("active_neuron_frac"),
            "dev_L2_g_at_max_frac": dl2.get("L6_g_at_max_frac_of_active"), "dev_L3_g_at_max_frac": dl.get("L6_g_at_max_frac_of_active"),
            "dev_per_class_accuracy": dv["per_class_accuracy"], "dev_confusion": dv["confusion"],
            "train_per_class_accuracy": tr["per_class_accuracy"] if tr else None,
            "max_gather_elements_now": self.be.max_gather, "cache_mb": round(self.loader.cache_bytes / 2 ** 20, 3),
            "loader_access": dict(self.loader.access), "counters": dict(self.counters)}
        row.update(self.timer.take())
        if self.be.is_cuda:
            row["gpu_max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated(self.be.dev) / 2 ** 20, 1)
            torch.cuda.reset_peak_memory_stats(self.be.dev)
        ctx = "dev_init_baseline" if row_kind == "init_baseline" else "dev_eval"
        for a, fams in (dv.get("routing") or {}).items():
            for fam, frow in fams.items():
                r.append_csv(ROUTING_CSV, {"utc": row["utc"], "active_area": area, "epoch_number": epoch_number, "context": ctx, "area": a,
                                           "family": fam, "frozen": a != area, **frow})
        for key, lrow in (dv.get("layers") or {}).items():
            r.append_csv(ROUTING_CSV, {"utc": row["utc"], "active_area": area, "epoch_number": epoch_number, "context": ctx,
                                       "area": key.split("/")[0], "family": f"layer_{key.split('/')[1]}", "frozen": key.split("/")[0] != area, **lrow})
        if self.epoch_stats is not None and row_kind == "epoch":
            for fam, frow in self.epoch_stats.family_rows(area).items():
                r.append_csv(ROUTING_CSV, {"utc": row["utc"], "active_area": area, "epoch_number": epoch_number, "context": "train_epoch_learning",
                                           "area": area, "family": fam, "frozen": False, **frow})
            for key, lrow in self.epoch_stats.layer_rows().items():
                r.append_csv(ROUTING_CSV, {"utc": row["utc"], "active_area": area, "epoch_number": epoch_number, "context": "train_epoch_learning",
                                           "area": area, "family": f"layer_{key.split('/')[1]}", "frozen": False, **lrow})
            r.npz(f"codebook/usage_{area}_e{epoch_number:03d}.npz",
                  **{f"b{b}__wins_train": w for b, w in enumerate(self.epoch_stats.wins[area])},
                  **{f"b{b}__route_id": bk.route for b, bk in enumerate(self.model.tables[area].buckets)},
                  **{f"b{b}__wins_dev": w for b, w in enumerate(self._dev_wins(dv, area))},
                  note=np.asarray("wins_train = 이 에폭 학습 배치 (갱신 전 코드북으로 순방향) 의 후보별 선택 횟수, route_id = 후보 칸의 route_id (−1 패딩)"))
        if row_kind == "epoch":
            self.write_codebook_epoch_rows(area, epoch_number)
        st["last_dev_routing"] = {a: {f: {k: v for k, v in x.items() if k in ("selected_frac", "selected_but_zero_output_frac",
                                                                                  "unused_candidate_frac", "mean_max_cosine")}
                                          for f, x in fams.items()} for a, fams in (dv.get("routing") or {}).items()}
        return row

    def write_training_row(self, row: dict[str, Any]) -> None:
        self.need_rec().append_csv(TRAINING_CSV, {k: v for k, v in row.items() if k not in ("loader_access", "counters")}
                                   | {"loader_train_reads": row["loader_access"].get("train"), "loader_dev_reads": row["loader_access"].get("dev"),
                                      "loader_test_reads": row["loader_access"].get("test"), "dev_evaluations": row["counters"]["dev_evaluations"],
                                      "test_evaluations": row["counters"]["test_evaluations"], "readout_fits": row["counters"]["readout_fits"]})

    def _dev_wins(self, dv: dict[str, Any], area: str) -> list[np.ndarray]:
        return self._last_dev_stats.wins[area] if getattr(self, "_last_dev_stats", None) is not None else \
            [np.zeros((bk.G, bk.K), np.int64) for bk in self.model.tables[area].buckets]

    def write_codebook_epoch_rows(self, area: str, epoch_number: int) -> None:
        """codebook_metrics.csv: 이번 에폭 갱신 노름·각도, 갱신된 후보 수, 초기값 대비 각도 (계열별)."""
        r = self.need_rec()
        t = self.model.tables[area]
        C, ok = self.model.routers[area].get_codebook()
        init = self.init_codes.get(area)
        for fi, fam in enumerate(t.family_names):
            n_upd = s_d = s_a = 0.0
            m_d = m_a = 0.0
            n_cand_upd = n_valid = 0
            drift = []
            for b, bk in enumerate(t.buckets):
                if bk.family != fi:
                    continue
                ua = self.upd_acc[b] if self.upd_acc else None
                v = bk.legal & ok[b]
                n_valid += int(v.sum())
                if ua is not None:
                    n_upd += float(ua["n_upd"].sum())
                    s_d += float(ua["sum_delta"].sum())
                    s_a += float(ua["sum_angle"].sum())
                    m_d = max(m_d, float(ua["max_delta"].max()) if ua["max_delta"].size else 0.0)
                    m_a = max(m_a, float(ua["max_angle"].max()) if ua["max_angle"].size else 0.0)
                    n_cand_upd += int((ua["n_upd"] > 0).sum())
                if init is not None:
                    cs = np.clip((C[b].astype(np.float64) * init[b].astype(np.float64)).sum(-1), -1.0, 1.0)
                    drift += list(np.degrees(np.arccos(cs[v])))
            r.append_csv(CODEBOOK_CSV, {"utc": utc_now(), "area": area, "epoch_number": epoch_number, "context": "epoch_updates", "family": fam,
                                        "valid_candidates": n_valid, "candidate_updates": int(n_upd), "candidates_updated_at_least_once": n_cand_upd,
                                        "candidates_updated_frac": (n_cand_upd / n_valid) if n_valid else None,
                                        "mean_update_norm": (s_d / n_upd) if n_upd else None, "max_update_norm": m_d,
                                        "mean_update_angle_deg": (s_a / n_upd) if n_upd else None, "max_update_angle_deg": m_a,
                                        "mean_angle_from_init_deg": float(np.mean(drift)) if drift else None,
                                        "max_angle_from_init_deg": float(np.max(drift)) if drift else None,
                                        "codebook_version": self.model.routers[area].version})

    def update_best(self, area: str, epoch_number: int, row: dict[str, Any]) -> None:
        C, ok = self.model.routers[area].get_codebook()
        self.best = {"area": area, "epoch": int(epoch_number), "C": C, "ok": ok, "readout": self.readouts[area].copy(),
                     "dev_accuracy": row.get("dev_accuracy"), "codebook_sha256": sha256_array(*(C + ok))}

    def restore_selected(self, area: str, reason: str) -> dict[str, Any]:
        """선택 에폭 (dev 최고, 동률 → 이른 에폭) 의 코드북·판독기로 되돌린다. 다음 영역은 이 상태에서 시작한다."""
        st = self.stage[area]
        if self.best is None or self.best["area"] != area:
            raise RuntimeError(f"{area} 의 선택 후보 상태가 없다.")
        self.model.routers[area].set_codebook(self.best["C"], self.best["ok"])
        self.readouts[area] = self.best["readout"].copy()
        st["selected_epoch"] = self.best["epoch"]
        st["selected_restored"] = True
        info = {"utc": utc_now(), "area": area, "selected_epoch_number": self.best["epoch"], "selected_dev_accuracy": self.best["dev_accuracy"],
                "codebook_sha256": self.best["codebook_sha256"], "reason": reason, "stage_status": st["status"],
                "epochs_completed": st["epochs_completed"]}
        st["completion"] = info
        self.need_rec().append_jsonl("stage_transitions.jsonl", {"kind": "restore_selected", **info})
        return info

    def model_version(self, area: str | None = None) -> str:
        area = area or (self.model.active if self.model else None)
        if area is None or self.model is None or area not in self.model.routers:
            return "no_model"
        st = self.stage.get(area, {})
        ro = self.readouts.get(area)
        return (f"{area}-e{st.get('epochs_completed', 0)}-b{st.get('next_batch', 0)}-cbv{self.model.routers[area].version}"
                f"-ro{ro.version if ro else 0}-{MODE}")

    # ---------------------------------------------------------------- 단계 전환
    def add_next_area_prepare(self) -> tuple[str, list[LayerSpec], RouteTable, dict[str, Any], dict[str, Any]]:
        cur = self.model.active
        st = self.stage[cur]
        info = {"current_area": cur, "current_status": st["status"], "epochs_completed": st["epochs_completed"],
                "best_dev_accuracy": st["best_dev_accuracy"], "best_epoch": st["best_epoch"], "selected_restored": st["selected_restored"],
                "warning_ko": (None if st["status"] in STAGE_TERMINAL else
                               f"{cur} 단계가 끝나지 않았다 (상태 '{st['status']}'). 추가하면 지금까지의 선택 에폭 (epoch {st['best_epoch']}) 으로 되돌려 "
                               "동결하고 '사용자 수동 진행' 으로 기록한다.")}
        area, specs, table, summ = self.plan_next()
        return area, specs, table, summ, info

    def freeze_area(self, area: str) -> dict[str, Any]:
        st = self.model.store
        sp = [self.model.specs[f"{area}/{l}"] for l in LAYER_ORDER]
        h = {"codebook_sha256": self.model.routers[area].codebook_sha(), "readout_sha256": self.readouts[area].sha(),
             "readout_version": self.readouts[area].version, "structure_sha256": self.model.struct_hash[area],
             "q_sha256": sha256_array(*[self.be.to_np(st.q_view(s.key)) for s in sp]),
             "threshold_sha256": sha256_array(*[self.be.to_np(st.thr_view(s.key)) for s in sp]),
             "preprocessing_sha256": sha256_text(dumps(self.pre.info(), indent=None)), "frozen_utc": utc_now(),
             "model_version_at_freeze": self.model_version(area), "selected_epoch": self.stage[area].get("selected_epoch")}
        self.frozen[area] = h
        return h

    def frozen_check(self) -> dict[str, Any]:
        out = {}
        st = self.model.store
        for area, h in self.frozen.items():
            sp = [self.model.specs[f"{area}/{l}"] for l in LAYER_ORDER]
            out[area] = {"codebook_unchanged": self.model.routers[area].codebook_sha() == h["codebook_sha256"],
                         "readout_unchanged": self.readouts[area].sha() == h["readout_sha256"],
                         "structure_unchanged": self.model.struct_hash[area] == h["structure_sha256"],
                         "q_unchanged": sha256_array(*[self.be.to_np(st.q_view(s.key)) for s in sp]) == h["q_sha256"],
                         "threshold_unchanged": sha256_array(*[self.be.to_np(st.thr_view(s.key)) for s in sp]) == h["threshold_sha256"],
                         "preprocessing_unchanged": sha256_text(dumps(self.pre.info(), indent=None)) == h["preprocessing_sha256"]}
        return out

    def add_next_area_commit(self, area: str, specs: list[LayerSpec], table: RouteTable, summ: dict[str, Any], info: dict[str, Any],
                             *, user_choice: str) -> dict[str, Any]:
        cur = self.model.active
        st = self.stage[cur]
        manual = st["status"] not in STAGE_TERMINAL
        if not st["selected_restored"] or manual:
            if manual:
                st["status"] = "stopped_by_user_progression"
            self.restore_selected(cur, "add_next_area" + ("_manual" if manual else ""))
        hashes = self.freeze_area(cur)
        tr = {"utc": utc_now(), "from": cur, "to": area, "from_status": st["status"], "stage_completed": not manual,
              "user_manual_progression": manual, "warning_ko": info["warning_ko"], "user_choice": user_choice, "frozen_hashes": hashes,
              "selected_epoch": st["selected_epoch"]}
        st["transitions"].append(tr)
        self.need_rec().append_jsonl("stage_transitions.jsonl", tr)
        res = self.build_area(area, specs, table, summ)
        return {"transition": tr, **res}

    # ---------------------------------------------------------------- 상세 표본 기록 (routes/ 조각 → sample_routes.npz)
    def flush_details(self) -> None:
        if not self.detail_buf:
            return
        r = self.need_rec()
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for c in self.detail_buf:
            groups.setdefault((c["area"], c["epoch"]), []).append(c)
        for (area, ep), rows in groups.items():
            k = len(list(r.p("routes").glob(f"sample_routes_{area}_e{ep:03d}_part*.npz")))
            t = self.model.tables[area]
            arr: dict[str, np.ndarray] = {"sample_id": np.asarray([x["sample_id"] for x in rows]),
                                          "batch_index": np.asarray([x["batch"] for x in rows], np.int64),
                                          "label_recorded_not_used_for_routing": np.asarray([x["label"] for x in rows], np.int64),
                                          "context": np.asarray([x["context"] for x in rows]),
                                          "delivered_ptr": np.concatenate([[0], np.cumsum([len(x["e_src"]) for x in rows])]).astype(np.int64),
                                          "delivered_src_L4_id": np.concatenate([x["e_src"] for x in rows]).astype(np.int64),
                                          "delivered_dst_L2_id": np.concatenate([x["e_dst"] for x in rows]).astype(np.int64),
                                          "delivered_route_id": np.concatenate([x["e_route"] for x in rows]).astype(np.int64),
                                          "delivered_slot": np.concatenate([x["e_slot"] for x in rows]).astype(np.int8),
                                          "delivered_base_weight": np.concatenate([x["e_w"] for x in rows]).astype(np.float32),
                                          "delivered_L4_output": np.concatenate([x["e_x"] for x in rows]).astype(np.float32),
                                          "delivered_value": np.concatenate([x["e_val"] for x in rows]).astype(np.float32)}
            for b, bk in enumerate(t.buckets):
                arr[f"b{b}__route_id"] = bk.route
                arr[f"b{b}__source_L4_local"] = bk.src
                arr[f"b{b}__group_id"] = bk.groups
                for key_ in ("k", "has", "S", "top1", "gap", "dom"):
                    arr[f"b{b}__{key_}"] = np.stack([x["b"][b][key_] for x in rows])
                arr[f"b{b}__scores_f16"] = np.stack([x["b"][b]["scores"] for x in rows]).astype(np.float16)
                arr[f"b{b}__xhat_f16"] = np.stack([x["b"][b]["xhat"] for x in rows]).astype(np.float16)
            arr["note"] = np.asarray("표본별 상세 (지정한 소수 표본): 묶음별 집단 S·선택 칸 k·최고 점수·1·2위 차·우세도, 후보별 점수·입력 방향 x̂ (float16), "
                                     "실제 전달 edge (기저 가중치 × L4 출력). 라벨은 기록만 하며 라우팅에 쓰지 않았다.")
            r.npz(f"routes/sample_routes_{area}_e{ep:03d}_part{k:03d}.npz", **arr)
        self.detail_buf = []

    def detail_records(self, area: str, recs: Sequence[dict[str, Any]], x4: np.ndarray, items: Sequence[dict[str, Any]], y: np.ndarray,
                       n: int, epoch: int, batch: int, context: str) -> list[dict[str, Any]]:
        be = self.be
        t = self.model.tables[area]
        l4s, l2s = self.model.specs[f"{area}/L4"].start, self.model.specs[f"{area}/L2"].start
        npb = [{k: be.to_np(v) for k, v in rec.items() if k in ("k", "has", "S", "top1", "gap", "dom", "scores", "xhat")} for rec in recs]
        out = []
        for i in range(min(n, len(items))):
            es, ed, er, esl, ew, ex = [], [], [], [], [], []
            per_b = []
            for b, bk in enumerate(t.buckets):
                d = npb[b]
                per_b.append({"k": d["k"][i], "has": d["has"][i], "S": d["S"][i], "top1": d["top1"][i], "gap": d["gap"][i], "dom": d["dom"][i],
                              "scores": np.where(np.isfinite(d["scores"][i]), d["scores"][i], np.nan), "xhat": d["xhat"][i]})
                for gl in np.nonzero(d["has"][i])[0]:
                    rid = int(bk.route[gl, int(d["k"][i, gl])])
                    s_, sl_, w_ = t.route_edges(rid)
                    es.append(l4s + s_)
                    ed.append(np.full(len(s_), l2s + int(t.r_dst[rid])))
                    er.append(np.full(len(s_), rid))
                    esl.append(sl_)
                    ew.append(w_)
                    ex.append(x4[i, s_])
            cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)     # noqa: E731
            e_x = cat(ex, np.float32)
            e_w = cat(ew, np.float32)
            out.append({"area": area, "epoch": epoch, "batch": batch, "sample_id": items[i]["relpath"], "label": int(y[i]), "context": context,
                        "b": per_b, "e_src": cat(es, np.int64), "e_dst": cat(ed, np.int64), "e_route": cat(er, np.int64),
                        "e_slot": cat(esl, np.int8), "e_w": e_w, "e_x": e_x, "e_val": e_w * e_x})
        return out

    def consolidate_sample_routes(self) -> Path | None:
        """routes/ 조각을 sample_routes.npz 하나로 모은다 (키 앞에 조각 이름). 조각이 없으면 빈 색인만 쓴다."""
        r = self.need_rec()
        parts = sorted(r.p("routes").glob("sample_routes_*_part*.npz"))
        arr: dict[str, np.ndarray] = {"parts": np.asarray([p.stem for p in parts])}
        for p in parts:
            for k, v in load_npz(p).items():
                arr[f"{p.stem}__{k}"] = v
        arr["readme"] = np.asarray("키 = <조각 이름>__<배열>. 조각 이름 sample_routes_<영역>_e<에폭>_part<번호>. 학습 배치의 지정 표본 상세 기록.")
        return r.npz("sample_routes.npz", **arr)

    # ---------------------------------------------------------------- checkpoint
    def save_checkpoint(self, tag: str, keep: bool = False) -> Path:
        with self.timer.phase("save"):
            return self._save_checkpoint(tag, keep)

    def _save_checkpoint(self, tag: str, keep: bool) -> Path:
        r = self.need_rec()
        self.flush_details()
        self.ckpt_seq += 1
        name = f"ckpt_{self.ckpt_seq:06d}_{re.sub(r'[^A-Za-z0-9_]+', '_', tag)}"
        arrays: dict[str, np.ndarray] = {"q": self.model.store.q_numpy(), "thr": self.model.store.thr_numpy()}
        cb_sha = {}
        for area in self.model.areas:
            C, ok = self.model.routers[area].get_codebook()
            cb_sha[area] = sha256_array(*(C + ok))
            for b in range(len(C)):
                arrays[f"cb__{area}__{b}"] = C[b]
                arrays[f"ok__{area}__{b}"] = ok[b]
            if area in self.readouts:
                arrays.update(self.readouts[area].state(f"ro__{area}__"))
        best_meta = None
        if self.best is not None:
            for b in range(len(self.best["C"])):
                arrays[f"best__cb__{b}"] = self.best["C"][b]
                arrays[f"best__ok__{b}"] = self.best["ok"][b]
            arrays.update(self.best["readout"].state("best__ro__"))
            best_meta = {"area": self.best["area"], "epoch": self.best["epoch"], "dev_accuracy": self.best["dev_accuracy"],
                         "codebook_sha256": self.best["codebook_sha256"], "readout_meta": self.best["readout"].meta}
        if self.epoch_stats is not None:
            arrays.update(self.epoch_stats.state("es__"))
        if self.upd_acc is not None:
            for b, ua in enumerate(self.upd_acc):
                for k_, v_ in ua.items():
                    arrays[f"ua__{b}__{k_}"] = v_
        npz_path = r.p(f"checkpoints/{name}.npz")
        save_npz(npz_path, **arrays)
        meta = {"format": CKPT_FORMAT, "seq": self.ckpt_seq, "tag": tag, "keep": bool(keep), "utc": utc_now(), "mode": MODE,
                "program": PROGRAM, "program_version": VERSION, "program_sha256": self.program_sha(), "npz": npz_path.name,
                "npz_sha256": sha256_file(npz_path), "areas": self.model.areas, "active_area": self.model.active,
                "layer_ranges": {k: [self.model.specs[k].start, self.model.specs[k].end] for k in self.model.store.order},
                "structure_sha256": self.model.struct_hash, "route_table_sha256": {a: self.model.tables[a].hash() for a in self.model.areas},
                "codebook_sha256": cb_sha, "codebook_version": {a: self.model.routers[a].version for a in self.model.areas},
                "readouts": {a: ro.meta for a, ro in self.readouts.items()}, "best": best_meta,
                "config_structure_sha256": config_structure_sha(self.cfg), "config_learning_sha256": config_learning_sha(self.cfg),
                "data_fingerprint_sha256": self.data.manifest["data_fingerprint_sha256"], "split_sha256": self.data.manifest["split_sha256"],
                "stage": self.stage, "frozen": self.frozen, "counters": self.counters, "rng_state": self.rng.bit_generator.state,
                "model_version": self.model_version(), "has_epoch_stats": self.epoch_stats is not None, "has_update_acc": self.upd_acc is not None,
                "sampler_ko": "에폭 순서 = default_rng([seed, 영역번호, epoch, 11]).permutation(n_train); 진행 위치 = stage[영역].next_batch",
                "base_file": BASE_FILE_INFO}
        r.json(f"checkpoints/{name}.json", meta)
        r.json("checkpoints/latest.json", {"name": name, "utc": meta["utc"], "model_version": meta["model_version"]})
        self.last_checkpoint = name
        self._prune_checkpoints()
        return npz_path

    def _prune_checkpoints(self) -> None:
        r = self.need_rec()
        keep_n = int(self.cfg["records"]["keep_periodic_checkpoints"])
        periodic = []
        for p in sorted(r.p("checkpoints").glob("ckpt_*.json")):
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
        """v5 run 만 이어 간다. 구버전 run (v1~v4) 은 코드북·패킷 스키마가 없어 오류로 안내한다 (분할 재사용은 reuse_split)."""
        run_dir = Path(run_dir)
        cf = read_json(run_dir / "config.json")
        if cf.get("program") != PROGRAM or not str(cf.get("program_version", "")).startswith("5."):
            raise CompatibilityError(
                f"이 run 은 {cf.get('program')} v{cf.get('program_version')} 가 만들었다. v5 의 코드북·입력 패킷 스키마가 없어 이어 학습으로 읽지 않는다. "
                "같은 데이터 분할을 쓰려면 메뉴 2 → '기존 run 분할 재사용' 또는 --reuse-split 으로 새 run 을 시작한다.")
        sess = cls(validate_config(cf["config"]), be, say=say, confirm=confirm)
        sess.rec = RunRecorder(run_dir, say=say)
        info = read_json(run_dir / "run_info.json")
        sess.output_root = Path(info["output_root"]) if info.get("output_root") else run_dir.parent
        droot = Path(data_root) if data_root else (Path(info["data_root"]) if info.get("data_root") else None)
        if droot is None or not droot.is_dir():
            raise CompatibilityError(f"데이터 폴더를 찾을 수 없다 ({droot}). --data-root 또는 메뉴에서 새 위치를 준다.")
        sess.data_root = droot
        if cf.get("program_sha256") and cf["program_sha256"] != sess.program_sha():
            msg = "이 실행을 만든 프로그램 파일과 현재 파일의 SHA256 이 다르다 (코드 변경). 결과 해석에 주의한다."
            say("[경고] " + msg) if read_only else sess.rec.warn(msg)
        man = read_json(run_dir / "split_manifest.json")
        sess.data = DatasetManager.from_manifest(sess.cfg, droot, man)
        probs = sess.data.verify_files()
        if probs:
            raise CompatibilityError("데이터가 분할 목록과 다르다: " + "; ".join(probs[:5]))
        sess.loader = BatchLoader(sess.data, sess.pre, be, sess.cfg, sess.timer)
        name = checkpoint or read_json(run_dir / "checkpoints" / "latest.json")["name"]
        meta = read_json(run_dir / "checkpoints" / f"{name}.json")
        if meta.get("format") in OLD_CKPT_FORMATS:
            raise CompatibilityError(f"구버전 checkpoint 형식 ({meta.get('format')}) 은 이어 학습으로 읽지 않는다.")
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
                raise CompatibilityError(f"{area} 구조·후보 그래프를 다시 만들었더니 해시가 다르다 (설정·코드 불일치).")
            nb = len(sess.model.tables[area].buckets)
            sess.model.routers[area].set_codebook([arrays[f"cb__{area}__{b}"] for b in range(nb)], [arrays[f"ok__{area}__{b}"] for b in range(nb)])
            if sess.model.routers[area].codebook_sha() != meta["codebook_sha256"][area]:
                raise CompatibilityError(f"{area} 코드북 해시가 checkpoint 기록과 다르다.")
            sess.model.routers[area].version = int(meta["codebook_version"][area])
            ip = run_dir / "codebook" / f"init_{area}.npz"
            if ip.is_file():
                z = load_npz(ip)
                sess.init_codes[area] = [z[f"b{b}__C"] for b in range(nb)]
            if f"ro__{area}__P" in arrays:
                ro = ClassReadout(area, sess.classes, sess.model.specs[f"{area}/L3"].n, be, float(sess.cfg["learning"]["eps_norm"]))
                ro.load_state(arrays, f"ro__{area}__", meta["readouts"].get(area, {}))
                sess.readouts[area] = ro
        if not np.array_equal(arrays["q"], sess.model.store.q_numpy()) or not np.array_equal(arrays["thr"], sess.model.store.thr_numpy()):
            raise CompatibilityError("checkpoint 의 q·임계값이 고정값과 다르다 (q·임계값은 학습하지 않는다).")
        sess.model.active = meta["active_area"]
        if meta.get("best"):
            bm = meta["best"]
            nb = len(sess.model.tables[bm["area"]].buckets)
            ro = ClassReadout(bm["area"], sess.classes, sess.model.specs[f"{bm['area']}/L3"].n, be, float(sess.cfg["learning"]["eps_norm"]))
            ro.load_state(arrays, "best__ro__", bm["readout_meta"])
            sess.best = {"area": bm["area"], "epoch": int(bm["epoch"]), "C": [arrays[f"best__cb__{b}"] for b in range(nb)],
                         "ok": [np.asarray(arrays[f"best__ok__{b}"], bool) for b in range(nb)], "readout": ro, "dev_accuracy": bm["dev_accuracy"],
                         "codebook_sha256": bm["codebook_sha256"]}
        if meta.get("has_epoch_stats"):
            sess.epoch_stats = RouteStats(sess.model, [sess.model.active])
            sess.epoch_stats.load_state(arrays, "es__")
        if meta.get("has_update_acc"):
            t = sess.model.tables[sess.model.active]
            sess.upd_acc = empty_update_acc(t)
            for b, ua in enumerate(sess.upd_acc):
                for k_ in ua:
                    ua[k_] = np.asarray(arrays[f"ua__{b}__{k_}"]).copy()
        sess.stage = meta["stage"]
        sess.frozen = meta["frozen"]
        sess.counters = meta["counters"]
        sess.rng.bit_generator.state = meta["rng_state"]
        sess.ckpt_seq = int(meta["seq"])
        sess.last_checkpoint = name
        if not read_only:
            sess._discard_uncommitted_details()
            sess.rec.info(f"재개: {name} ({meta['model_version']})", echo=True)
        return sess

    def _discard_uncommitted_details(self) -> None:
        """checkpoint 위치보다 뒤의 상세 기록 조각 (중단 전에 쓰였지만 되돌린 배치) 을 지운다 → 재개 뒤 중복 기록 방지."""
        r = self.need_rec()
        for area, st in self.stage.items():
            for p in r.p("routes").glob(f"sample_routes_{area}_e*_part*.npz"):
                ep = int(re.search(r"_e(\d+)_part", p.name).group(1))
                cur = int(st["epochs_completed"]) + 1
                if ep > cur:
                    p.unlink()
                elif ep == cur and st["phase"] == "training":
                    z = load_npz(p)
                    if z["batch_index"].size and int(z["batch_index"].max()) >= int(st["next_batch"]):
                        p.unlink()

    def inspect_neuron(self, nid: int, item: dict[str, Any] | None = None) -> dict[str, Any]:
        dyn = None
        if item is not None:
            key, _j = self.model.store.locate(int(nid))
            area = key.split("/")[0]
            lgn, _y = self.loader.batch([item], item.get("split", "external"))
            res = self.model.forward(lgn, upto=area, keep=(key,) + ((f"{area}/L2",) if key != f"{area}/L2" else ()), detail=(area,))
            be = self.be
            dyn = {k_: {kk: be.to_np(vv)[0] for kk, vv in v_.items() if vv is not None and kk in ("u", "s", "conf", "out")}
                   for k_, v_ in res.items() if "/L" in k_ and k_ in (key,)}
            if key.endswith("/L2"):
                dyn[key]["mask"] = be.to_np(res[f"{area}/route"]["mask"])[0]
            dyn["route"] = res[f"{area}/route"]["buckets"]
        return NeuronInspector(self.model).inspect_neuron(int(nid), dyn)


AREA_DIMENSION_TABLE: dict[str, dict[str, str]] = {
    "V1": {"L4_input_meaning": "LGN 위치·대비·ON/OFF 채널 (일대일 중계)", "L2_role_family": "방향·위상·극성 Gabor (위치당 48 후보)",
           "L3_processing": "같은 위치·방향의 위상×극성 4 L2 평균", "must_check": "단일 L4 입력 [1] 퇴화 방지 (수용장 패킷), signed 필터 보존"},
    "V2": {"L4_input_meaning": "V1 L3 방향 특징의 2×2 블록 수렴", "L2_role_family": "윤곽 연속·끝점 (앞/뒤)·교차",
           "L3_processing": "일대일 (기존 대응 출력)", "must_check": "교차의 두 방향과 끝점 억제 입력의 출처 보존"},
    "V4": {"L4_input_meaning": "V2 L3 의 윤곽·끝점·교차 (블록 평균)", "L2_role_family": "곡선 (뉴런 × Q 묶음)·직선·접합·선 끝",
           "L3_processing": "같은 채널 3×3 격자 이웃 평균", "must_check": "A/B 공동 배달, 곡률·공간 조건, 과도한 선택 희소성"},
    "IT": {"L4_input_meaning": "V4 L3 특징의 영역 평균 (넓은 범위 수렴)", "L2_role_family": "전역 특징 (V4 계열별 4 집단)·영역 특징군·특징군 결합",
           "L3_processing": "일대일 (최종 판독 벡터)", "must_check": "IT 전체 하나만 고르는 전역 top-1 금지 (역할·공간별 집단)"},
}


# ======================================================================
# 14. 단계 학습 (StageTrainer) — 활성 영역 하나의 코드북만 국소 경쟁학습
# ======================================================================
def epoch_permutation(seed: int, area: str, epoch: int, n: int) -> np.ndarray:
    return np.random.default_rng([int(seed), AREA_INDEX[area], int(epoch), 11]).permutation(n)


OOM_GUIDE_KO = ("GPU 메모리 부족. 현재 배치는 적용하지 않았고 gather 조각 크기를 반씩 줄여 같은 배치를 다시 시도했지만 실패했다. "
                "마지막 경계에서 checkpoint 를 저장했다. CPU 로 몰래 바꾸지 않는다. 후보는 삭제하지 않는다.\n"
                "  줄이는 방법 (새 실행에서 설정 변경): learning.batch_size ↓, compute.max_gather_elements ↓, routing.bucket_max_elements ↓.\n"
                "  같은 설정으로 이어 가려면 다른 GPU 프로그램을 닫고 메뉴 4 (재개) 로 다시 시작한다.")


class StageTrainer:
    """현재 활성 영역 하나만 라우팅 코드북을 학습한다. 앞 영역은 동결 (L6·목적지 선택·출력은 매 입력 다시 계산).

    배치 순환: (1) 같은 학습 전 코드북으로 배치 전체 순방향 (승리 x̂ 합·수 수집, 라벨 미사용) (2) 비유한 값 검사
               (3) CodebookLearner: 배치마다 한 번 c ← normalize((1−η)c + η·v̂) (4) 집계·기록 커밋 (원자적)
    에폭 끝: 코드북 고정 → train 판독기 재계산 → train·dev 평가 (dev 는 갱신 없음) → 선택 (dev 최고, 동률 → 이른 에폭) → 종료 판단.
    종료 (성공 기준·plateau·예산) 시 선택 에폭의 코드북·판독기로 되돌린다."""

    def __init__(self, sess: Session):
        self.s = sess

    def run(self, *, extra_epochs: int | None = None, stop_after_batches: int | None = None) -> dict[str, Any]:
        s = self.s
        area = s.model.active
        st = s.stage[area]
        if extra_epochs is not None:
            new_max = int(st["epochs_completed"]) + int(extra_epochs)
            st["budget_changes"].append({"utc": utc_now(), "from_max_epochs": st["max_epochs"], "to_max_epochs": new_max, "from_status": st["status"],
                                         "continues_from_selected_epoch": st["selected_epoch"] if st["selected_restored"] else None})
            st["max_epochs"] = new_max
            if st["status"] in STAGE_TERMINAL:
                st["bad_epochs"] = 0
                st["selected_restored"] = False
        elif st["status"] in STAGE_TERMINAL:
            return {"status": st["status"], "area": area, "best_epoch": st["best_epoch"], "best_dev_accuracy": st["best_dev_accuracy"],
                    "message_ko": f"{area} 단계는 이미 '{st['status']}' 로 끝났다 (선택 에폭 {st['selected_epoch']}). 더 학습하려면 추가 에폭 수를 준다."}
        if st["epochs_completed"] >= st["max_epochs"] and st["phase"] == "training":
            st["status"] = "budget_exhausted"
            return self._finish(area, st, "에폭 예산이 0 이다 (추가 에폭을 준다).")
        st["status"] = "running"
        st["started_utc"] = st["started_utc"] or utc_now()
        with StopFlag(s.say) as stop:
            try:
                return self._loop(area, st, stop, stop_after_batches)
            except KeyboardInterrupt:
                return self._stop(area, st, "interrupted_manual", "Ctrl+C 즉시 중단: 진행 중 배치는 적용하지 않았다.")

    def _stop(self, area: str, st: dict[str, Any], status: str, why: str) -> dict[str, Any]:
        st["status"] = status
        self.s.save_checkpoint(f"{area}_{status}", keep=True)
        self.s.rec.info(f"{area} 학습 멈춤 ({status}): {why}", echo=True)
        return {"status": status, "area": area, "message_ko": why, "checkpoint": self.s.last_checkpoint,
                "epochs_completed": st["epochs_completed"], "next_batch": st["next_batch"],
                "best_dev_accuracy": st["best_dev_accuracy"], "best_epoch": st["best_epoch"]}

    def _finish(self, area: str, st: dict[str, Any], why: str = "") -> dict[str, Any]:
        s = self.s
        st["ended_utc"] = utc_now()
        info = s.restore_selected(area, st["status"])
        st["structure_unchanged_at_end"] = s.model.struct_hash[area] == st.get("structure_sha_at_creation")
        st["frozen_check_at_end"] = s.frozen_check()
        if s.cfg["records"]["role_probe_after_stage"]:
            try:
                RoleProbes(s).run(context=f"{area}_selected_e{st['selected_epoch']:03d}")
            except Exception as exc:                                       # noqa: BLE001  (측정 실패는 기록하고 학습 결과는 유지)
                s.rec.error("역할 검사 실패", exc, echo=False)
        s.save_checkpoint(f"{area}_stage_{st['status']}", keep=True)
        s.rec.info(f"{area} 단계 종료 {st['status']}: 선택 에폭 {st['selected_epoch']} (dev {fmt(st['best_dev_accuracy'])}), 코드북·판독기 복원", echo=True)
        return {"status": st["status"], "area": area, "epochs_completed": st["epochs_completed"], "best_dev_accuracy": st["best_dev_accuracy"],
                "best_epoch": st["best_epoch"], "selected_epoch": st["selected_epoch"], "checkpoint": s.last_checkpoint, "restore": info,
                "message_ko": why or None}

    def _loop(self, area: str, st: dict[str, Any], stop: StopFlag, stop_after: int | None) -> dict[str, Any]:
        s, be = self.s, self.s.be
        items = s.data.items("train")
        n, B = len(items), int(s.cfg["learning"]["batch_size"])
        n_batches = int(math.ceil(n / B))
        router = s.model.routers[area]
        learner = CodebookLearner(router, s.cfg)
        every = int(s.cfg["records"]["checkpoint_every_batches"])
        done_now = 0
        while st["epochs_completed"] < st["max_epochs"]:
            e = int(st["epochs_completed"])
            perm = epoch_permutation(int(s.cfg["seed"]), area, e, n)
            if st["phase"] == "training":
                if s.epoch_stats is None or st["next_batch"] == 0:
                    s.epoch_stats = RouteStats(s.model, [area])
                    s.upd_acc = empty_update_acc(s.model.tables[area])
                    st["detail_epoch_saved"] = 0
                while st["next_batch"] < n_batches:
                    if stop.requested:
                        return self._stop(area, st, "interrupted_manual", "Ctrl+C: 배치 경계에서 멈췄다.")
                    bi = int(st["next_batch"])
                    chunk = [items[i] for i in perm[bi * B:(bi + 1) * B]]
                    retries = 0
                    while True:
                        old_C, old_ver = list(router.C), router.version
                        committed = False
                        try:
                            out = self._batch(area, chunk, e, bi)
                            with stop.critical():
                                with s.timer.phase("update", gpu=be.is_cuda):
                                    ust = learner.update(out["recs"])
                                if s.fault == "interrupt_after_apply":
                                    s.fault = None
                                    raise KeyboardInterrupt
                                self._commit(area, st, out, ust)
                                committed = True
                            break
                        except KeyboardInterrupt:
                            if not committed:
                                router.C, router.version = old_C, old_ver
                            raise
                        except BaseException as exc:        # noqa: BLE001
                            if not committed:
                                router.C, router.version = old_C, old_ver
                            if is_oom_error(exc) and retries < int(s.cfg["compute"]["oom_retry_halvings"]) and be.max_gather > 4096:
                                retries += 1
                                old = be.max_gather
                                be.max_gather = max(4096, old // 2)
                                if torch is not None and be.is_cuda:
                                    torch.cuda.empty_cache()
                                s.rec.append_jsonl("oom_events.jsonl", {
                                    "utc": utc_now(), "area": area, "epoch": e, "batch": bi, "retry": retries, "max_gather_from": old,
                                    "max_gather_to": be.max_gather, "batch_size_unchanged": len(chunk), "error": f"{type(exc).__name__}: {exc}"[:300],
                                    "note_ko": "코드북을 배치 전 상태로 두고 같은 배치를 gather 조각만 줄여 다시 계산 (수식·배치 평균·후보는 같다)"})
                                s.rec.warn(f"메모리 부족 → gather 조각 {old} → {be.max_gather} 로 같은 배치 재시도 ({retries})")
                                continue
                            if is_oom_error(exc):
                                s.rec.error("메모리 부족", exc)
                                r_ = self._stop(area, st, "oom", OOM_GUIDE_KO)
                                s.say(OOM_GUIDE_KO)
                                return r_
                            status = "failed_nonfinite" if isinstance(exc, NumericalError) else "failed"
                            if isinstance(exc, NumericalError):
                                st["n_nonfinite_events"] = int(st.get("n_nonfinite_events", 0)) + 1
                            s.rec.error(f"{area} epoch {e + 1} batch {bi} 오류 — 이 배치의 코드북 갱신은 적용하지 않았다", exc)
                            st["status"] = status
                            s.save_checkpoint(f"{area}_{status}", keep=True)
                            raise
                    done_now += 1
                    step = max(1, n_batches // 10)
                    if n_batches >= 20 and st["next_batch"] % step == 0 and st["next_batch"] < n_batches:
                        fr = s.epoch_stats.family_rows(area)
                        sel = np.nanmean([x["selected_frac"] for x in fr.values() if x["selected_frac"] is not None] or [np.nan])
                        zo = np.nanmean([x["selected_but_zero_output_frac"] for x in fr.values() if x["selected_but_zero_output_frac"] is not None] or [np.nan])
                        s.say(f"    [{area} epoch {e + 1}] 배치 {st['next_batch']}/{n_batches}, 집단 선택 비율 {fmt(sel)}, "
                              f"선택됐지만 출력 0 비율 {fmt(zo)}")
                    if every and st["next_batch"] % every == 0 and st["next_batch"] < n_batches:
                        with stop.critical():
                            s.save_checkpoint(f"{area}_e{e + 1}_b{st['next_batch']}")
                    if stop_after and done_now >= stop_after:
                        return self._stop(area, st, "interrupted_test", f"검사용: {done_now} 배치 뒤 멈춤")
                with stop.critical():
                    st["phase"] = "train_done"
                    s.save_checkpoint(f"{area}_e{e + 1}_train_done")
            if stop.requested:
                return self._stop(area, st, "interrupted_manual", "Ctrl+C: 에폭 학습 끝, 판독기 재계산·dev 평가 전에서 멈췄다 (재개하면 평가부터).")
            row = self._epoch_end(area, st, stop)
            s.say(f"  [{area} epoch {row['epoch_number']}] train {pct_count(row['train_correct'], row['train_n'])}, dev "
                  f"{pct_count(row['dev_correct'], row['dev_n'])} (분류 불능 {row['dev_unclassifiable']}), 선택 에폭 {st['best_epoch']} → {row['stage_status']}")
            if st["status"] in STAGE_TERMINAL:
                break
        if st["status"] not in STAGE_TERMINAL:
            st["status"] = "budget_exhausted"
        return self._finish(area, st)

    def _batch(self, area: str, chunk: list[dict[str, Any]], e: int, bi: int) -> dict[str, Any]:
        s, be = self.s, self.s.be
        st = s.stage[area]
        lgn, y = s.loader.batch(chunk, "train")
        if s.fault == "oom_once":
            s.fault = None
            raise RuntimeError("CUDA out of memory (자체 검사용 주입)")
        if s.fault == "nonfinite_once":
            s.fault = None
            x_ = be.to_np(lgn).copy()
            x_[0, :] = np.nan
            lgn = be.f32(x_)
        rc = s.cfg["records"]
        quota = min(int(rc["detail_samples_per_epoch"]) - int(st["detail_epoch_saved"]), int(rc["detail_samples_max_per_stage"]) - int(st["detail_saved"]))
        keys = (f"{area}/L4", f"{area}/L2", f"{area}/L3")
        with s.timer.phase("forward", gpu=be.is_cuda and bool(s.cfg["compute"]["gpu_timing_sync"])):
            res = s.model.forward(lgn, upto=area, keep=keys, learn=area, detail=((area,) if quota > 0 else ()))
        recs = res[f"{area}/route"]["buckets"]
        rec_np = route_rec_np(be, recs)
        n_bad = int(sum(int(r["bad"].sum()) for r in rec_np))
        if n_bad:
            raise NumericalError(f"{area} 학습 배치 {bi}: 비유한 값 또는 음수 L4 입력이 있는 집단 {n_bad} 개 (조용히 전달하지 않는다)")
        l6 = s.model.l6
        layers = {k_: layer_reduce(be, res[k_], l6.g_min, l6.g_max, vector=k_.endswith("/L3")) for k_ in keys[1:]}
        details = []
        if quota > 0:
            details = s.detail_records(area, recs, be.to_np(res[f"{area}/L4"]["out"]).astype(np.float32), chunk, y, quota, e + 1, bi, "train_batch")
        return {"recs": recs, "rec_np": rec_np, "layers": layers, "details": details, "B": len(chunk)}

    def _commit(self, area: str, st: dict[str, Any], o: dict[str, Any], ust: list[dict[str, np.ndarray]]) -> None:
        s = self.s
        s.epoch_stats.add_route(area, o["rec_np"])
        for k_, red in o["layers"].items():
            s.epoch_stats.add_layer(k_, red)
        for b, u in enumerate(ust):
            ua = s.upd_acc[b]
            m = u["upd"]
            ang = np.degrees(np.arccos(np.clip(u["cos"], -1.0, 1.0)))
            ua["n_upd"] += m.astype(np.int64)
            ua["sum_delta"] += np.where(m, u["delta"], 0.0)
            ua["max_delta"] = np.maximum(ua["max_delta"], np.where(m, u["delta"], 0.0))
            ua["sum_angle"] += np.where(m, ang, 0.0)
            ua["max_angle"] = np.maximum(ua["max_angle"], np.where(m, ang, 0.0))
        if o["details"]:
            s.detail_buf.extend(o["details"])
            st["detail_saved"] = int(st["detail_saved"]) + len(o["details"])
            st["detail_epoch_saved"] = int(st["detail_epoch_saved"]) + len(o["details"])
        s.counters["codebook_updates"] = int(s.counters.get("codebook_updates", 0)) + 1
        st["next_batch"] += 1

    def _epoch_end(self, area: str, st: dict[str, Any], stop: StopFlag) -> dict[str, Any]:
        s = self.s
        e = int(st["epochs_completed"])
        ep = e + 1
        row = s.epoch_evaluation(area, ep, "epoch")
        dev_acc = row["dev_accuracy"]
        improved = st["best_dev_accuracy"] is None or (dev_acc is not None and dev_acc > st["best_dev_accuracy"] + st["min_delta"])
        if improved:
            st["best_dev_accuracy"], st["best_epoch"], st["bad_epochs"] = dev_acc, ep, 0
            s.update_best(area, ep, row)
        else:
            st["bad_epochs"] += 1
        st["epochs_completed"] = ep
        succ = st["success_dev_accuracy"]
        if succ is not None and dev_acc is not None and dev_acc >= float(succ):
            st["status"] = "success_criterion_met"
        elif st["bad_epochs"] >= st["patience"]:
            st["status"] = "plateau"
        elif st["epochs_completed"] >= st["max_epochs"]:
            st["status"] = "budget_exhausted"
        else:
            st["status"] = "running"
        row.update(stage_status=st["status"], best_dev_accuracy_so_far=st["best_dev_accuracy"], patience_counter=st["bad_epochs"],
                   selected_epoch_so_far=st["best_epoch"], improved_this_epoch=bool(improved),
                   n_train_batches=int(math.ceil(len(s.data.items("train")) / int(s.cfg["learning"]["batch_size"]))))
        with stop.critical():
            s.write_training_row(row)
            st["history"].append({k: v for k, v in row.items() if k not in ("dev_confusion",)})
            st["phase"], st["next_batch"] = "training", 0
            s.epoch_stats, s.upd_acc = None, None
            s.save_checkpoint(f"{area}_e{ep}_done", keep=False)
        return row


# ======================================================================
# 15. 사전 검사 (PreChecks 1~16) · 역할 검사 · 최종 test
# ======================================================================
class _SplitSpy:
    """BatchLoader.batch 호출의 분할 이름을 센다 (데이터 경계 검사용). 원래 함수로 되돌린다."""

    def __init__(self, loader: BatchLoader):
        self.loader = loader
        self.splits: list[str] = []
        self._orig: Any = None

    def __enter__(self) -> "_SplitSpy":
        self._orig = self.loader.batch

        def spy(items: Sequence[dict[str, Any]], split: str) -> tuple[Any, np.ndarray]:
            self.splits.append(split)
            return self._orig(items, split)

        self.loader.batch = spy                              # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: Any) -> None:
        self.loader.batch = self._orig                       # type: ignore[method-assign]


class _LimitItems:
    """검사용: 분할 목록을 앞쪽 n 개로 잠시 줄인다 (전체 MNIST train/dev 를 검사 때마다 다시 읽지 않게). 원래 함수로 되돌린다."""

    def __init__(self, data: DatasetManager, n: int):
        self.data, self.n = data, int(n)
        self._orig: Any = None

    def __enter__(self) -> "_LimitItems":
        self._orig = self.data.items
        self.data.items = lambda split: self._orig(split)[:self.n]     # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: Any) -> None:
        self.data.items = self._orig                                   # type: ignore[method-assign]


FORBIDDEN_GRAD_PATTERNS = (r"\.backward\s*\(", r"\bautograd\b", r"torch\.func\b", r"\bjacrev\b", r"\bjacfwd\b", r"\bvjp\b", r"\bjvp\b",
                           r"requires_grad\s*=\s*True", r"requires_grad_\(\s*True", r"enable_grad", r"retain_graph", r"\bfunctorch\b", r"torch\.optim\b")
FORBIDDEN_CALL_MODULES = ("torch.autograd", "torch.optim", "torch.func", "functorch", "torch._functorch")
FORBIDDEN_CALL_NAMES = ("backward", "grad", "vjp", "jvp", "jacrev", "jacfwd", "retain_grad", "requires_grad_", "enable_grad", "set_grad_enabled")
REMOVED_LEGACY_NAMES = ("LocalOutputLearner", "LocalErrorRouter", "L1Relay", "PrototypeBank", "GuideSampler", "RoutingRuntime", "BandRuntime",
                        "BandTable", "DestRuntime", "DestTable", "run_band_calibration")


class _NotRun(RuntimeError):
    pass


def tiny_route_setup(cfg: dict[str, Any], be: Backend, n_src: int, rows: Sequence[dict[str, Any]]) -> tuple[RouteTable, RatioRouter, LayerSpec, LayerSpec]:
    """검사용 작은 후보 그래프. rows[j] = {"A": [(src, w)], "B": [(src, w)], "op": bool, "family": str, "space": int,
    "routed": [(src, w, reason)] (곡선처럼 B 후보 목록이 있는 행)}. L2 뉴런 j = rows[j] (위치 j, 채널 1)."""
    n2 = len(rows)
    a = CSRBuilder(n2, n_src)
    op = np.zeros(n2, bool)
    routed = np.zeros(n2, bool)
    ro_rows, ro_src, ro_w, ro_reason = [], [], [], []
    for j, rw in enumerate(rows):
        for s_, w_ in rw.get("A", []):
            a.add(j, s_, w_, "A")
        for s_, w_ in rw.get("B", []):
            a.add(j, s_, w_, "B")
        op[j] = bool(rw.get("op", False)) or bool(rw.get("routed"))
        if rw.get("routed"):
            routed[j] = True
            for s_, w_, rs_ in sorted(rw["routed"]):
                ro_rows.append(j)
                ro_src.append(s_)
                ro_w.append(w_)
                ro_reason.append(rs_)
    nb = single_position_neighbors()
    l4 = _spec("V1", "L4", "LGN", n_src, n_src, 1, ["x"], ["test_source"], {"family": ["x"]}, np.zeros(n_src, bool), identity_csr(n_src),
               np.zeros(n_src), np.zeros(n_src), np.ones(n_src), np.zeros(n_src), np.zeros(n_src), (np.zeros(n_src), np.zeros(n_src), np.ones(n_src)),
               None, nb, False, {"role_ko": "검사용 원천"})
    l2 = _spec("V1", "L2", "V1/L4", n_src, n2, 1, ["c"], ["test_destination"], {"family": ["c"]}, op, a.finalize(), np.zeros(n2), np.zeros(n2),
               np.ones(n2), np.zeros(n2), np.zeros(n2), (np.zeros(n2), np.zeros(n2), np.ones(n2)), None, nb, True, {"role_ko": "검사용 목적지"},
               routed=routed)
    if routed.any():
        l2.ropt = {"row": np.asarray(ro_rows, np.int64), "src": np.asarray(ro_src, np.int64), "w": np.asarray(ro_w, np.float32),
                   "q": np.asarray(ro_src, np.int64), "valid": np.asarray(ro_reason) == 0, "reason": np.asarray(ro_reason, np.int8),
                   "dist_px": np.zeros(len(ro_rows), np.float32), "chord_dev_deg": np.zeros(len(ro_rows), np.float32)}
    set_route_keys(l2, np.asarray([rw.get("space", 0) for rw in rows]), [rw.get("family", "f") for rw in rows], "test")
    t = build_route_table("precheck", l4, l2, cfg)
    return t, RatioRouter(t, be, cfg), l4, l2


def _set_codes(router: RatioRouter, codes: dict[int, Sequence[float]]) -> None:
    """route_id → 대표 벡터 (집단 원천 순서 좌표). 지원 마스크 밖 성분은 검사 입력에서 0 으로 둔다."""
    t = router.t
    C = [np.zeros((bk.G, bk.K, bk.D), np.float32) for bk in t.buckets]
    ok = [np.zeros((bk.G, bk.K), bool) for bk in t.buckets]
    for r, v in codes.items():
        b, gl, k = int(t.r_bucket[r]), int(t.r_bk_g[r]), int(t.r_bk_k[r])
        v = np.asarray(v, np.float64)
        C[b][gl, k, :len(v)] = (v / np.linalg.norm(v)).astype(np.float32)
        ok[b][gl, k] = True
    router.set_codebook(C, ok)


def _group_sel(router: RatioRouter, out: dict[str, Any], g: int, i: int = 0) -> tuple[bool, int, float, float]:
    t, be = router.t, router.be
    b, gl = int(t.g_bucket[g]), int(t.g_local[g])
    rec = out["buckets"][b]
    has = bool(be.to_np(rec["has"])[i, gl])
    k = int(be.to_np(rec["k"])[i, gl])
    return has, (int(t.buckets[b].route[gl, k]) if has else -1), float(be.to_np(rec["u_sel"])[i, gl]), float(be.to_np(rec["top1"])[i, gl])


class PreChecks:
    """명세 15 장의 사전 검사 16 항목 (구현의 수학적·경로 정합성 검사이며 학습 성공을 증명하지 않는다).
    PASSED / FAILED / NOT_RUN. 1~9 는 작은 합성 후보 그래프, 10~16 은 현재 세션 모델에서 수행하고 상태를 복원한다 (test 미사용)."""

    def __init__(self, sess: Session):
        self.s = sess

    def _snapshot(self) -> dict[str, Any]:
        s = self.s
        snap = {"cb": {a: s.model.routers[a].get_codebook() for a in s.model.areas}, "ver": {a: s.model.routers[a].version for a in s.model.areas},
                "ro": {a: r.copy() for a, r in s.readouts.items()}, "stage": copy.deepcopy(s.stage), "best": s.best,
                "es": s.epoch_stats.state("x") if s.epoch_stats is not None else None, "ua": copy.deepcopy(s.upd_acc),
                "detail": list(s.detail_buf), "rng": copy.deepcopy(s.rng.bit_generator.state), "counters": copy.deepcopy(s.counters),
                "q": s.model.store.q_numpy(), "thr": s.model.store.thr_numpy(), "max_gather": s.be.max_gather}
        snap["sha"] = {a: sha256_array(*(c + o)) for a, (c, o) in snap["cb"].items()}
        return snap

    def _restore(self, snap: dict[str, Any]) -> bool:
        s = self.s
        for a, (c, o) in snap["cb"].items():
            s.model.routers[a].set_codebook(c, o)
            s.model.routers[a].version = snap["ver"][a]
        s.readouts = {a: r.copy() for a, r in snap["ro"].items()}
        s.stage = copy.deepcopy(snap["stage"])
        s.best = snap["best"]
        if snap["es"] is not None:
            s.epoch_stats = RouteStats(s.model, [s.model.active])
            s.epoch_stats.load_state(snap["es"], "x")
        else:
            s.epoch_stats = None
        s.upd_acc = copy.deepcopy(snap["ua"])
        s.detail_buf = list(snap["detail"])
        s.rng.bit_generator.state = snap["rng"]
        dev_n, test_n = s.counters.get("dev_evaluations", 0), s.counters.get("test_evaluations", 0)
        s.counters = copy.deepcopy(snap["counters"])
        s.counters["dev_evaluations"], s.counters["test_evaluations"] = dev_n, test_n      # 검사도 dev 를 본 것으로 센다 (되돌리지 않음)
        s.be.max_gather = snap["max_gather"]
        return all(s.model.routers[a].codebook_sha() == h for a, h in snap["sha"].items())

    def run(self) -> dict[str, Any]:
        s = self.s
        if s.model is None or not s.model.areas:
            raise RuntimeError("모델이 없다 (메뉴 3).")
        snap = self._snapshot()
        test0 = int(s.loader.access.get("test", 0))
        out: dict[str, Any] = {}
        checks = (("C01_ratio_descriptor_selects_matching_code", self.c01), ("C02_scale_invariant_route_and_doubled_delivery", self.c02),
                  ("C03_different_ratios_different_descriptors", self.c03), ("C04_zero_input_no_selection_output_update", self.c04),
                  ("C05_illegal_padding_no_candidate_not_delivered", self.c05), ("C06_AB_codelivery_no_synthesis", self.c06),
                  ("C07_signed_weights_and_role_ops_preserved", self.c07), ("C08_one_update_matches_formula", self.c08),
                  ("C09_support_mask_and_unit_norm", self.c09), ("C10_all_areas_common_router_learner", self.c10),
                  ("C11_freeze_eval_invariance_l6_and_selection_run", self.c11), ("C12_labels_do_not_change_routing_or_updates", self.c12),
                  ("C13_data_boundary_no_test_no_dev_in_fitting", self.c13), ("C14_cpu_vs_gpu_and_reference", self.c14),
                  ("C15_save_restore_reproduces_and_resumes", self.c15), ("C16_disabled_paths_not_called", self.c16))
        for name, fn in checks:
            t0 = time.perf_counter()
            try:
                out[name] = fn()
            except _NotRun as exc:
                out[name] = {"status": NOT_RUN, "reason_ko": str(exc)}
            except Exception as exc:                             # noqa: BLE001  — 검사 실패는 FAILED 로 남긴다
                out[name] = {"status": FAILED, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=6)}
            out[name]["seconds"] = round(time.perf_counter() - t0, 3)
            if not self._restore(snap):
                out[name + "_RESTORE"] = {"status": FAILED, "error": "검사 뒤 코드북 복원 실패"}
        restored = self._restore(snap)
        summ = {"utc": utc_now(), "model_version": s.model_version(), "areas": s.model.areas, "backend": s.be.describe(), "checks": out,
                "restored": restored, "test_access_delta": int(s.loader.access.get("test", 0)) - test0,
                "n_failed": sum(1 for v in out.values() if v.get("status") == FAILED),
                "n_not_run": sum(1 for v in out.values() if v.get("status") == NOT_RUN),
                "note_ko": "구현의 수학적·경로 정합성 검사다. 정확도·역할 학습 성공을 증명하지 않으며 통과 조건으로 정확도를 쓰지 않는다."}
        summ["status"] = FAILED if (summ["n_failed"] or not restored) else PASSED
        if s.rec is not None:
            s.rec.json(f"diagnostics/prechecks_{local_stamp()}_{time.time_ns() % 100000}.json", summ)
            self._append_index("prechecks", summ)
        return summ

    def _append_index(self, kind: str, summary: dict[str, Any]) -> None:
        r = self.s.rec
        p = r.p("diagnostics.json")
        cur = read_json(p) if p.is_file() else {"runs": []}
        cur["runs"].append({"kind": kind, "utc": summary.get("utc"), "model_version": summary.get("model_version"), "status": summary.get("status"),
                            "checks": {k: (v.get("status") if isinstance(v, dict) else v) for k, v in (summary.get("checks") or {}).items()}})
        r.json("diagnostics.json", cur)

    # ---------------------------------------------------------------- 1~9: 작은 합성 후보 그래프
    def _four(self) -> tuple[RouteTable, RatioRouter]:
        rows = [{"A": [(0, 1.0), (1, 1.0), (2, 1.0)], "family": "f", "space": 0} for _ in range(4)]
        t, r, _l4, _l2 = tiny_route_setup(self.s.cfg, self.s.be, 3, rows)
        _set_codes(r, {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1], 3: [1, 1, 2]})
        return t, r

    def c01(self) -> dict[str, Any]:
        t, r = self._four()
        out = r.route(self.s.be.f32([[1.0, 1.0, 2.0]]))
        has, route, u, top1 = _group_sel(r, out, 0)
        ok = has and route == 3 and abs(top1 - 1.0) < 1e-6
        return {"status": PASSED if ok else FAILED, "selected_route": route, "expected_route": 3, "top1_score": top1,
                "codes": "[1,0,0], [0,1,0], [0,0,1], [1,1,2]/√6", "input": [1, 1, 2]}

    def c02(self) -> dict[str, Any]:
        t, r = self._four()
        be = self.s.be
        out = r.route(be.f32([[1.0, 1.0, 2.0], [2.0, 2.0, 4.0]]))
        h1, r1, u1, s1 = _group_sel(r, out, 0, 0)
        h2, r2, u2, s2 = _group_sel(r, out, 0, 1)
        uL2 = be.to_np(out["u"])
        ok = h1 and h2 and r1 == r2 and u2 == 2.0 * u1 and abs(s1 - s2) < 1e-6 and float(uL2[1].max()) == 2.0 * float(uL2[0].max())
        return {"status": PASSED if ok else FAILED, "route_x": r1, "route_2x": r2, "delivered_u_x": u1, "delivered_u_2x": u2,
                "score_x": s1, "score_2x": s2, "note_ko": "전달 모듈 (L6·clip·임계 제외) 에서 실제 전달값이 정확히 두 배인지 본다."}

    def c03(self) -> dict[str, Any]:
        t, r = self._four()
        be = self.s.be
        out = r.route(be.f32([[1.0, 1.0, 2.0], [2.0, 1.0, 1.0]]), detail=True)
        xh = be.to_np(out["buckets"][0]["xhat"])[:, 0, :3].astype(np.float64)
        dom = be.to_np(out["buckets"][0]["dom"])[:, 0]
        cos = float(xh[0] @ xh[1])
        _h1, r1, _u1, _s1 = _group_sel(r, out, 0, 0)
        _h2, r2, _u2, _s2 = _group_sel(r, out, 0, 1)
        ok = cos < 1.0 - 1e-6 and abs(float(dom[0]) - 0.5) < 1e-6 and abs(float(dom[1]) - 0.5) < 1e-6
        return {"status": PASSED if ok else FAILED, "descriptor_cos": cos, "dominance": dom.tolist(), "routes": [r1, r2],
                "note_ko": "우세도 (max/sum) 는 둘 다 0.5 이지만 방향 x̂ 은 다르다. 같은 후보를 고를 수도 있으므로 다른 후보를 강제하지 않는다."}

    def c04(self) -> dict[str, Any]:
        t, r = self._four()
        be = self.s.be
        C0, _ok0 = r.get_codebook()
        out = r.route(be.f32([[0.0, 0.0, 0.0]]), learn=True)
        has, route, u, _ = _group_sel(r, out, 0)
        ust = CodebookLearner(r, self.s.cfg).update(out["buckets"])
        C1, _ok1 = r.get_codebook()
        same = all(np.array_equal(a, b) for a, b in zip(C0, C1))
        ok = (not has) and float(np.abs(be.to_np(out["u"])).max()) == 0.0 and not bool(be.to_np(out["mask"]).any()) and same \
            and not any(u_["upd"].any() for u_ in ust) and float(be.to_np(out["buckets"][0]["wins"]).sum()) == 0.0
        return {"status": PASSED if ok else FAILED, "has_selection": has, "max_abs_u": float(np.abs(be.to_np(out["u"])).max()),
                "codebook_unchanged": same, "n_updated": int(sum(u_["upd"].sum() for u_ in ust))}

    def c05(self) -> dict[str, Any]:
        be = self.s.be
        rows = [{"A": [(0, 1.0)], "routed": [(1, 1.0, 0), (2, 1.0, 1)], "family": "curve", "space": 0},      # 합법 후보 1 + 탈락 후보 1
                {"A": [(3, 1.0)], "family": "curve", "space": 1}, {"A": [(4, 1.0)], "family": "curve", "space": 1},
                {"A": [(5, 1.0)], "family": "curve", "space": 1},
                {"A": [(0, 1.0)], "routed": [(2, 1.0, 2)], "family": "curve", "space": 2}]                   # 후보 없는 집단 (모두 탈락)
        cfg2 = validate_config(deep_merge(self.s.cfg, {"routing": {"bucket_padding_max": 100.0}}))    # 두 집단을 한 묶음에 넣어 패딩 칸을 만든다
        t, r, _l4, _l2 = tiny_route_setup(cfg2, be, 6, rows)
        g0, g2 = int(t.l2_group[0]), int(t.l2_group[4])
        legal0_r = int(t.group_routes(g0)[0])
        _set_codes(r, {int(x): ([0.0, 1.0] if int(x) == legal0_r else np.ones(int(t.g_D[t.r_group[x]]))) for x in np.nonzero(t.r_legal)[0]})
        C, ok = r.get_codebook()
        b, gl = int(t.g_bucket[g0]), int(t.g_local[g0])
        if t.buckets[b].K > int(t.g_n_routes[g0]):                            # 패딩 칸에 완벽히 맞는 대표를 몰래 넣어도 선택되면 안 된다
            C[b][gl, int(t.g_n_routes[g0]):, :] = 1.0 / math.sqrt(t.buckets[b].D)
        r.set_codebook(C, ok)
        x = np.zeros((1, 6), np.float32)
        x[0, [0, 1, 2]] = [1.0, 0.5, 9.0]                                     # 탈락 후보 (원천 2) 쪽 입력이 크다
        out = r.route(be.f32(x))
        has0, route0, _u0, _ = _group_sel(r, out, g0)
        mask = be.to_np(out["mask"])[0]
        u = be.to_np(out["u"])[0]
        legal0 = [int(x_) for x_ in t.group_routes(g0)]
        ok_ = (t.n_routes == 3 + 1 + 1 + 1 and int(t.r_legal.sum()) == 4 and int(t.g_n_routes[g2]) == 0 and int(t.g_bucket[g2]) == -1
               and has0 and route0 in legal0 and not mask[4] and u[4] == 0.0 and t.l2_n_legal[4] == 0 and not mask[1:4].any()
               and t.buckets[b].K > int(t.g_n_routes[g0]))
        return {"status": PASSED if ok_ else FAILED, "selected_route_group0": route0, "legal_routes_group0": legal0,
                "rejected_reasons": [ROUTE_REASONS[int(v)] for v in t.r_reason[~t.r_legal]], "no_candidate_group_bucket": int(t.g_bucket[g2]),
                "no_candidate_destination_u": float(u[4]), "masks": mask.tolist(),
                "note_ko": "탈락 후보·패딩 칸·후보 없는 집단은 전달되지 않고 0 번 뉴런 등 다른 뉴런으로 보내지지 않는다 (후보 없음 ≠ 활동 0 으로 따로 기록)."}

    def c06(self) -> dict[str, Any]:
        be, s = self.s.be, self.s
        rows = [{"A": [(0, 1.0)], "B": [(1, 1.0)], "op": True, "family": "pair", "space": 0},
                {"A": [(1, 1.0)], "B": [(2, 1.0)], "op": True, "family": "pair", "space": 0}]
        t, r, _l4, _l2 = tiny_route_setup(s.cfg, be, 3, rows)
        _set_codes(r, {0: [1, 1, 0], 1: [0, 1, 1]})
        out = r.route(be.f32([[3.0, 5.0, 0.0], [3.0, 0.0, 0.0]]))
        u = be.to_np(out["u"])
        rec = route_rec_np(be, out["buckets"])[0]
        stats = RouteStats.__new__(RouteStats)
        stats.model, stats.areas, stats.thr, stats.near = None, ["T"], 0.0, 1e-6
        stats.g = {"T": [{k: np.zeros(bk.G) for k in ROUTE_STAT_KEYS} for bk in t.buckets]}
        stats.wins = {"T": [np.zeros((bk.G, bk.K), np.int64) for bk in t.buckets]}
        stats.n_samples = {"T": 0}
        stats.layers = {}
        stats.add_route("T", [rec])
        g = stats.g["T"][0]
        tiny_ok = (u[0, 0] == 3.0 and u[0, 1] == 0.0 and u[1].max() == 0.0 and float(rec["uA_sel"][1, 0]) == 3.0
                   and float(rec["uB_sel"][1, 0]) == 0.0 and int(g["n_A_only"].sum()) == 1 and int(g["n_both"].sum()) == 1)
        model_rows = {}
        model_ok = True
        items = s.data.items("train")[:min(4, len(s.data.items("train")))]
        lgn, _y = s.loader.batch(items, "train")
        for a in s.model.areas:
            res = s.model.forward(lgn, upto=a, keep=(f"{a}/L4", f"{a}/L2"), record=(a,))
            x4 = be.to_np(res[f"{a}/L4"]["out"])
            C, ok = s.model.routers[a].get_codebook()
            ref = route_reference(s.model.tables[a], C, ok, x4, s.cfg)
            mk = be.to_np(res[f"{a}/route"]["mask"])
            uu = be.to_np(res[f"{a}/L2"]["u"]).astype(np.float64)
            both = mk & ref["mask"]
            err = float(np.abs(uu - ref["u"])[both].max()) if both.any() else 0.0
            scale = float(np.abs(ref["u"]).max()) if ref["u"].size else 0.0
            gap = ref["gap"]
            dis = (mk != ref["mask"]).any(1)
            n_dis = int((mk != ref["mask"]).sum())
            near_only = True
            if n_dis:
                bi_, ji_ = np.nonzero(mk != ref["mask"])                     # 어긋난 (표본, 목적지) 의 집단에서만 기준식 1·2 위 차를 본다
                gg = gap[bi_, s.model.tables[a].l2_group[ji_]]
                near_only = bool(np.all(~np.isfinite(gg) | (gg < 1e-5)))
            model_rows[a] = {"max_abs_u_err_selected": err, "u_scale": scale, "mask_disagreements": n_dis, "only_at_near_ties": near_only,
                             "samples_with_disagreement": int(dis.sum())}
            model_ok &= err <= 1e-5 * (1.0 + scale) and (n_dis == 0 or near_only)
        ok = tiny_ok and model_ok
        return {"status": PASSED if ok else FAILED, "tiny_min2": {"u_both_present": float(u[0, 0]), "u_B_missing": float(u[1].max()),
                                                                     "A_only": int(g["n_A_only"].sum()), "both": int(g["n_both"].sum())},
                "model_vs_float64_reference": model_rows,
                "note_ko": "한 후보 묶음의 A·B 가 같은 목적지로 전달되고, B 가 없으면 A 를 복제하거나 양수 바닥값을 넣지 않는다 (u = min(A, 0) = 0, A-only 로 기록)."}

    def c07(self) -> dict[str, Any]:
        s = self.s
        rows = {}
        ok = True
        for a in s.model.areas:
            t = s.model.tables[a]
            l2 = s.model.specs[f"{a}/L2"]
            bad = 0
            neg = int((t.e_w < 0).sum())
            for j in range(0, l2.n, max(1, l2.n // 400)):
                row = l2.row(j)
                for r_ in np.nonzero(t.r_dst == j)[0]:
                    es, sl, ws = t.route_edges(int(r_))
                    if not (np.array_equal(es[sl == 0], row["a_src"]) and np.array_equal(ws[sl == 0], row["a_w"])):
                        bad += 1
                    if not (l2.routed is not None and l2.routed[j]) and not (np.array_equal(es[sl == 1], row["b_src"]) and np.array_equal(ws[sl == 1], row["b_w"])):
                        bad += 1
            dense_bad = 0
            for bk in t.buckets:
                for gl in range(0, bk.G, max(1, bk.G // 50)):
                    for k in range(bk.K):
                        if not bk.legal[gl, k]:
                            continue
                        es, sl, ws = t.route_edges(int(bk.route[gl, k]))
                        d_ = np.searchsorted(bk.src[gl, :int(bk.dvalid[gl].sum())], es)
                        wa = np.zeros(bk.D)
                        np.add.at(wa, d_[sl == 0], ws[sl == 0])
                        dense_bad += int(not np.allclose(wa, bk.WA[gl, k]))
            rows[a] = {"route_weight_mismatches_vs_csr": bad, "dense_weight_mismatches": dense_bad, "negative_base_weights": neg,
                       "min_base_weight": float(t.e_w.min()) if len(t.e_w) else None}
            ok &= bad == 0 and dense_bad == 0
        if "V1" in s.model.areas:
            ok &= rows["V1"]["negative_base_weights"] > 0
        if "V2" in s.model.areas:
            ok &= rows["V2"]["negative_base_weights"] > 0
        be = s.be
        neg_rows = [{"A": [(0, 1.0), (1, -1.0)], "family": "g", "space": 0}, {"A": [(0, -1.0), (1, 1.0)], "family": "g", "space": 0}]
        t, r, _a, _b = tiny_route_setup(s.cfg, be, 2, neg_rows)
        _set_codes(r, {0: [0, 1], 1: [1, 0]})
        out = r.route(be.f32([[1.0, 3.0]]))
        _h, route, u, _ = _group_sel(r, out, 0)
        s_out = float(np.maximum(be.to_np(out["u"])[0] - float(s.cfg["neuron"]["threshold"]), 0).max())
        tiny_ok = route == 0 and u == -2.0 and s_out == 0.0
        ok &= tiny_ok
        return {"status": PASSED if ok else FAILED, "areas": rows, "tiny_signed": {"selected_route": route, "u_selected": u, "s_after_threshold": s_out},
                "note_ko": "라우팅 설명자는 비음수 L4 값으로 만들고 부호 있는 기저 가중치 (V1 Gabor ±, V2 endstop −w_inh) 는 선택 뒤 역할 연산에만 쓴다. "
                           "선택된 후보의 역할 출력이 음수면 s = 0 (selected_but_zero_output)."}

    def c08(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        rows = [{"A": [(0, 1.0), (1, 1.0), (2, 1.0)], "family": "f", "space": 0} for _ in range(3)]
        t, r, _a, _b = tiny_route_setup(s.cfg, be, 3, rows)
        _set_codes(r, {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]})
        C0, _ = r.get_codebook()
        X = np.asarray([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [3.0, 1.0, 0.0]])
        out = r.route(be.f32(X), learn=True)
        ks = [_group_sel(r, out, 0, i)[1] for i in range(3)]
        CodebookLearner(r, s.cfg).update(out["buckets"])
        C1, _ = r.get_codebook()
        eta = float(s.cfg["routing"]["eta_codebook"])
        xh = X / X.sum(1, keepdims=True)
        xh = xh / np.linalg.norm(xh, axis=1, keepdims=True)
        exp = {}
        for k in range(3):
            c = C0[0][0, k].astype(np.float64)
            sel = [i for i in range(3) if ks[i] == k]
            if sel:
                v = xh[sel].mean(0)
                v = v / np.linalg.norm(v)
                c = (1 - eta) * c + eta * v
                c = c / np.linalg.norm(c)
            exp[k] = c
        err = max(float(np.abs(C1[0][0, k] - exp[k]).max()) for k in range(3))
        unsel = [k for k in range(3) if k not in ks]
        unchanged = all(np.array_equal(C1[0][0, k], C0[0][0, k]) for k in unsel)
        ok = err < 1e-6 and unchanged and ks == [0, 1, 0]
        return {"status": PASSED if ok else FAILED, "winners": ks, "max_abs_err_vs_formula": err, "unselected": unsel,
                "unselected_unchanged_exactly": unchanged, "eta": eta}

    def c09(self) -> dict[str, Any]:
        s = self.s
        rows = {}
        ok = True
        for a in s.model.areas:
            t = s.model.tables[a]
            C, okc = s.model.routers[a].get_codebook()
            out_sup = neg = bad_norm = bad_failed = 0
            for b, bk in enumerate(t.buckets):
                c = C[b].astype(np.float64)
                out_sup += int((np.abs(c[~bk.support]) > 0).sum())
                neg += int((c < 0).sum())
                n = np.linalg.norm(c, axis=-1)
                bad_norm += int((np.abs(n[okc[b]] - 1.0) > 1e-5).sum())
                bad_failed += int((n[~okc[b]] != 0).sum())
            rows[a] = {"values_outside_support": out_sup, "negative_values": neg, "valid_codes_not_unit": bad_norm,
                       "invalid_codes_nonzero": bad_failed, "valid_codes": int(sum(o.sum() for o in okc))}
            ok &= out_sup == 0 and neg == 0 and bad_norm == 0 and bad_failed == 0
        return {"status": PASSED if ok else FAILED, "areas": rows}

    # ---------------------------------------------------------------- 10~16: 현재 모델
    def _probe(self, n: int = 4, split: str = "train") -> tuple[Any, np.ndarray, list[dict[str, Any]]]:
        items = self.s.data.items(split)[:n]
        lgn, y = self.s.loader.batch(items, split)
        return lgn, y, items

    def c10(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        rows = {}
        ok = True
        lgn, _y, _ = self._probe(2)
        for a in s.model.areas:
            rt = s.model.rt[f"{a}/L2"]
            router = rt.router
            same_obj = router is s.model.routers[a]
            res = s.model.forward(lgn, upto=a, keep=(f"{a}/L4", f"{a}/L2"), record=(a,))
            direct = router.route(res[f"{a}/L4"]["out"])
            eq = bool(np.array_equal(be.to_np(direct["u"]), be.to_np(res[f"{a}/L2"]["u"])))
            rows[a] = {"router_class": type(router).__name__, "learner_class": CodebookLearner.__name__, "same_router_object": same_obj,
                       "L2_has_no_csr_runtime": rt.A is None and rt.Bm is None, "L2_u_equals_router_output": eq,
                       "L4_L3_fixed_csr": s.model.rt[f"{a}/L4"].A is not None and s.model.rt[f"{a}/L3"].A is not None}
            ok &= type(router) is RatioRouter and same_obj and rt.A is None and eq and rows[a]["L4_L3_fixed_csr"]
        return {"status": PASSED if ok else FAILED, "areas": rows, "note_ko": "모든 영역 L2 가 같은 RatioRouter·CodebookLearner 를 쓰고 구버전 경로 (CSR 직접 계산) 가 없다."}

    def c11(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        area = s.model.active
        h0 = {a: s.model.routers[a].codebook_sha() for a in s.model.areas}
        r0 = {a: ro.sha() for a, ro in s.readouts.items()}
        q0, t0 = s.model.store.q_numpy(), s.model.store.thr_numpy()
        with _LimitItems(s.data, 2 * int(s.cfg["learning"]["batch_size"])):
            s.evaluate(area, "dev", s.readouts[area], purpose="precheck_c11_eval", log=False)
        lgn, _y, _ = self._probe(3)
        keys = [f"{a}/{l}" for a in s.model.areas for l in ("L2", "L3")]
        res = s.model.forward(lgn, keep=keys, record=tuple(s.model.areas))
        g_rows = {}
        for k in keys:
            g = res[k]["g"]
            g_rows[k] = {"g_computed": g is not None, "g_not_all_one": bool(g is not None and np.abs(be.to_np(g) - 1.0).max() > 0)}
        sel_rows = {a: int(sum(int(be.to_np(rb["has"]).sum()) for rb in res[f"{a}/route"]["buckets"])) for a in s.model.areas}
        cosine_classify(be, res[f"{area}/L3"]["out"], s.readouts[area].P, s.readouts[area].valid, float(s.cfg["learning"]["eps_norm"]))
        h1 = {a: s.model.routers[a].codebook_sha() for a in s.model.areas}
        r1 = {a: ro.sha() for a, ro in s.readouts.items()}
        unchanged = h0 == h1 and r0 == r1 and np.array_equal(q0, s.model.store.q_numpy()) and np.array_equal(t0, s.model.store.thr_numpy())
        l6_ok = all(v["g_computed"] for v in g_rows.values())
        ok = unchanged and l6_ok and all(v > 0 for v in sel_rows.values())
        return {"status": PASSED if ok else FAILED, "codebook_readout_q_threshold_unchanged": unchanged, "l6_layers": g_rows,
                "selections_per_area_in_probe": sel_rows, "frozen_areas": list(s.frozen)}

    def c12(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        area = s.model.active
        bad = {}
        for fn in (RatioRouter.route, RatioRouter._chunk, CodebookLearner.update, CodebookInitializer.run, CortexModel.forward, CortexModel.layer_step,
                   StageTrainer._batch):
            names = [p_.lower() for p_ in inspect.signature(fn).parameters]
            hit = [n for n in names if any(w in n for w in ("label", "target", "proto", "class", "truth", "readout")) or n == "y"]
            if hit:
                bad[fn.__qualname__] = hit
        items = s.data.items("train")[:int(s.cfg["learning"]["batch_size"])]
        relab = [{**it, "label": (int(it["label"]) + 1) % len(s.classes)} for it in items]
        a1, y1 = s.loader.batch(items, "train")
        a2, y2 = s.loader.batch(relab, "train")
        o1 = s.model.forward(a1, upto=area, learn=area)[f"{area}/route"]["buckets"]
        o2 = s.model.forward(a2, upto=area, learn=area)[f"{area}/route"]["buckets"]
        same_route = all(np.array_equal(be.to_np(x["k"]), be.to_np(y_["k"])) and np.array_equal(be.to_np(x["has"]), be.to_np(y_["has"])) for x, y_ in zip(o1, o2))
        same_upd = all(np.array_equal(be.to_np(x["wins"]), be.to_np(y_["wins"])) and np.array_equal(be.to_np(x["sums"]), be.to_np(y_["sums"])) for x, y_ in zip(o1, o2))
        C = len(s.classes)
        readout_stats_differ = not np.array_equal(np.bincount(y1, minlength=C), np.bincount(y2, minlength=C)) or bool((y1 != y2).any())
        ok = not bad and same_route and same_upd and bool((y1 != y2).all())
        return {"status": PASSED if ok else FAILED, "label_like_parameters": bad, "labels_changed": bool((y1 != y2).all()),
                "routing_identical": same_route, "update_statistics_identical": same_upd, "readout_statistics_can_differ": readout_stats_differ}

    def c13(self) -> dict[str, Any]:
        s = self.s
        r = s.need_rec()
        ids = {sp: {it["relpath"] for it in s.data.items(sp)} for sp in ("train", "dev", "test")}
        rows = {}
        ok = True
        for a in s.model.areas:
            p = r.p(f"codebook/init_{a}.npz")
            jp = r.p(f"codebook/init_{a}.json")
            if not p.is_file() or not jp.is_file():
                rows[a] = {"init_record": "없음"}
                ok = False
                continue
            sid = [str(x) for x in load_npz(p)["sample_id"]]
            js = read_json(jp)
            rows[a] = {"init_samples": len(sid), "init_in_train": all(x in ids["train"] for x in sid),
                       "init_in_dev_or_test": sum(1 for x in sid if x in ids["dev"] or x in ids["test"]), "init_split": js.get("split"),
                       "labels_used": js.get("labels_used")}
            ok &= rows[a]["init_in_train"] and rows[a]["init_in_dev_or_test"] == 0 and js.get("split") == "train" and js.get("labels_used") is False
            acc = r.p(f"structure/{a}_data_access.json")
            if acc.is_file():
                aj = read_json(acc)
                rows[a]["creation_test_reads"] = {"codebook_init": aj["codebook_init_delta"].get("test"), "baseline": aj["baseline_eval_delta"].get("test"),
                                                  "codebook_init_dev": aj["codebook_init_delta"].get("dev")}
                ok &= aj["codebook_init_delta"].get("test", 0) == 0 and aj["baseline_eval_delta"].get("test", 0) == 0 and aj["codebook_init_delta"].get("dev", 0) == 0
        train_sha = sha256_text("\n".join(sorted(ids["train"])))
        ro_ok = all(ro.meta.get("split") == "train" and ro.meta.get("sample_ids_sha256") == train_sha for ro in s.readouts.values())
        test0 = int(s.loader.access.get("test", 0))
        with _SplitSpy(s.loader) as spy:
            with _LimitItems(s.data, 2 * int(s.cfg["learning"]["batch_size"])):
                s.fit_readout(s.model.active, None, train_accuracy=False)
            area = s.model.active
            StageTrainer(s)._batch(area, s.data.items("train")[:int(s.cfg["learning"]["batch_size"])], 0, 0)
        splits = sorted(set(spy.splits))
        ok &= ro_ok and splits == ["train"] and int(s.loader.access.get("test", 0)) == test0
        return {"status": PASSED if ok else FAILED, "areas": rows, "readouts_train_only": ro_ok, "splits_requested_by_fit_and_learning": splits,
                "test_access_delta": int(s.loader.access.get("test", 0)) - test0, "test_evaluations_so_far": s.counters["test_evaluations"]}

    def c14(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        lgn, _y, _ = self._probe(4)
        ref_rows = {}
        ref_ok = True
        for a in s.model.areas:
            res = s.model.forward(lgn, upto=a, keep=(f"{a}/L4",), record=(a,))
            x4 = be.to_np(res[f"{a}/L4"]["out"])
            C, ok = s.model.routers[a].get_codebook()
            ref = route_reference(s.model.tables[a], C, ok, x4, s.cfg)
            t = s.model.tables[a]
            top = np.full(ref["top1"].shape, np.nan)
            rid = np.full(ref["route"].shape, -1)
            for b, bk in enumerate(t.buckets):
                rb = res[f"{a}/route"]["buckets"][b]
                hb, kb, tb = be.to_np(rb["has"]), be.to_np(rb["k"]), be.to_np(rb["top1"])
                for gl, g in enumerate(bk.groups):
                    top[:, g] = tb[:, gl]
                    rid[:, g] = np.where(hb[:, gl], bk.route[gl][kb[:, gl]], -1)
            fin = np.isfinite(top) & np.isfinite(ref["top1"])
            err = float(np.abs(top - ref["top1"])[fin].max()) if fin.any() else 0.0
            mism = rid != ref["route"]
            near = ~np.isfinite(ref["gap"]) | (ref["gap"] < 1e-5)
            ref_rows[a] = {"max_abs_top_score_diff": err, "selection_mismatches": int(mism.sum()), "mismatches_at_near_ties": int((mism & near).sum())}
            ref_ok &= err <= 1e-5 and int((mism & ~near).sum()) == 0
        res_ = {"reference_float64_vs_backend": ref_rows, "reference_ok": ref_ok, "backend": f"{be.kind}/{be.device}"}
        if not be.is_cuda:
            res_.update(status=NOT_RUN if ref_ok else FAILED,
                        reason_ko=f"현재 장치가 {be.kind}/{be.device} 라 CPU/GPU 비교를 할 수 없다 (GPU 가 있는 PC 에서 메뉴 7-1). "
                                  "float64 기준식과 현재 경로의 비교는 실행했다 (reference_float64_vs_backend).")
            return res_
        cpu = Backend("numpy")
        gpu_rows = {}
        ok_ = ref_ok
        for a in s.model.areas:
            t = s.model.tables[a]
            rc = RatioRouter(t, cpu, s.cfg)
            C, okc = s.model.routers[a].get_codebook()
            rc.set_codebook(C, okc)
            x4 = s.model.forward(lgn, upto=a, keep=(f"{a}/L4",))[f"{a}/L4"]["out"]
            og = s.model.routers[a].route(x4, learn=True)
            oc = rc.route(cpu.f32(be.to_np(x4)), learn=True)
            sc_err, mism, near_m, upd_err = 0.0, 0, 0, 0.0
            for b in range(len(t.buckets)):
                tg, tc = be.to_np(og["buckets"][b]["top1"]).astype(np.float64), np.asarray(oc["buckets"][b]["top1"], np.float64)
                fin = np.isfinite(tg) & np.isfinite(tc)
                sc_err = max(sc_err, float(np.abs(tg - tc)[fin].max()) if fin.any() else 0.0)
                kg, kc = be.to_np(og["buckets"][b]["k"]), np.asarray(oc["buckets"][b]["k"])
                gap = np.asarray(oc["buckets"][b]["gap"], np.float64)
                m_ = (kg != kc) & np.asarray(oc["buckets"][b]["has"])
                mism += int(m_.sum())
                near_m += int((m_ & (~np.isfinite(gap) | (gap < 1e-5))).sum())
                if not m_.any():
                    upd_err = max(upd_err, float(np.abs(be.to_np(og["buckets"][b]["sums"]) - np.asarray(oc["buckets"][b]["sums"])).max()))
            gpu_rows[a] = {"max_abs_score_diff": sc_err, "selection_mismatches": mism, "mismatches_at_near_ties": near_m,
                           "max_abs_update_sum_diff": upd_err}
            ok_ &= sc_err <= 1e-4 and mism == near_m and upd_err <= 1e-4
        res_.update(status=PASSED if ok_ else FAILED, gpu_vs_cpu=gpu_rows,
                    note_ko="GPU 축약 순서가 CPU 와 달라 거의 동점 (1·2 위 차 < 1e-5) 의 선택은 달라질 수 있다. 비트 단위 동일성을 주장하지 않는다.")
        return res_

    def c15(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        if s.rec is None:
            raise _NotRun("실행 폴더가 없다.")
        s.save_checkpoint("precheck_C15")
        name = s.last_checkpoint
        L = Session.load(s.rec.dir, be, say=lambda _m: None, checkpoint=name, read_only=True)
        try:
            lgn, _y, items = self._probe(3)
            x2, _ = L.loader.batch(items, "train")
            areas = tuple(s.model.areas)
            r1 = s.model.forward(lgn, record=areas)
            r2 = L.model.forward(x2, record=areas)
            top = f"{s.model.areas[-1]}/L3"
            out_eq = bool(np.array_equal(be.to_np(r1[top]["out"]), be.to_np(r2[top]["out"])))
            route_eq = all(np.array_equal(be.to_np(r1[f"{a}/route"]["mask"]), be.to_np(r2[f"{a}/route"]["mask"])) for a in areas)
            res = {"checkpoint": name, "outputs_identical": out_eq, "routes_identical": route_eq,
                   "codebooks_identical": all(s.model.routers[a].codebook_sha() == L.model.routers[a].codebook_sha() for a in areas),
                   "readouts_identical": all(s.readouts[a].sha() == L.readouts[a].sha() for a in s.readouts),
                   "stage_position_identical": all(s.stage[a]["epochs_completed"] == L.stage[a]["epochs_completed"] and
                                                   s.stage[a]["next_batch"] == L.stage[a]["next_batch"] for a in areas),
                   "rng_identical": s.rng.bit_generator.state == L.rng.bit_generator.state,
                   "epoch_accumulators_identical": (s.epoch_stats is None and L.epoch_stats is None) or (
                       s.epoch_stats is not None and L.epoch_stats is not None and
                       all(np.array_equal(v, L.epoch_stats.state("x")[k]) for k, v in s.epoch_stats.state("x").items())),
                   "best_snapshot_identical": (s.best is None and L.best is None) or (s.best is not None and L.best is not None and
                                                                                      s.best["codebook_sha256"] == L.best["codebook_sha256"])}
        finally:
            L.rec.close()
        ok = all(v for k, v in res.items() if isinstance(v, bool))
        return {"status": PASSED if ok else FAILED, **res, "note_ko": "에폭 중간 재개의 동등성 (끊고 이어 간 학습 = 끊지 않은 학습) 은 자체 검사 ST07 이 확인한다."}

    def c16(self) -> dict[str, Any]:
        s, be = self.s, self.s.be
        mod = sys.modules[__name__]
        present = [n for n in REMOVED_LEGACY_NAMES if hasattr(mod, n)]
        static = {}
        for obj in (RatioRouter, CodebookLearner, CodebookInitializer, CortexModel, StageTrainer, Session.epoch_evaluation, Session.fit_readout,
                    ClassReadout, cosine_classify, RelativeActivityController, build_route_table):
            src = inspect.getsource(obj)
            found = [p for p in FORBIDDEN_GRAD_PATTERNS if re.search(p, src)]
            if found:
                static[getattr(obj, "__qualname__", str(obj))] = found
        area = s.model.active
        items = s.data.items("train")[:int(s.cfg["learning"]["batch_size"])]
        calls: dict[str, int] = {}
        bad: list[str] = []

        def prof(frame: Any, event: str, arg: Any) -> None:
            if event == "call":
                key = f"{frame.f_globals.get('__name__', '')}.{frame.f_code.co_name}"
            elif event == "c_call":
                key = f"{getattr(arg, '__module__', None) or ''}.{getattr(arg, '__qualname__', getattr(arg, '__name__', '?'))}"
            else:
                return
            calls[key] = calls.get(key, 0) + 1
            nm = key.rsplit(".", 1)[-1]
            if any(key.startswith(p) for p in FORBIDDEN_CALL_MODULES) or (nm in FORBIDDEN_CALL_NAMES and "torch" in key):
                bad.append(key)

        q0, t0 = s.model.store.q_numpy(), s.model.store.thr_numpy()
        grad0 = be.grad_enabled()
        tr = StageTrainer(s)
        sys.setprofile(prof)
        try:
            out = tr._batch(area, items, 0, 0)
            CodebookLearner(s.model.routers[area], s.cfg).update(out["recs"])
        finally:
            sys.setprofile(None)
        grad1 = be.grad_enabled()
        tensors = [s.model.store.q, s.model.store.thr] + list(s.model.routers[area].C) + [ro.P for ro in s.readouts.values()]
        rg = be.requires_grad_any(tensors)
        q_ok = np.array_equal(q0, s.model.store.q_numpy()) and np.array_equal(t0, s.model.store.thr_numpy())
        learn_calls = sorted(k for k in calls if k.startswith(__name__ + ".") and any(x in k for x in ("route", "update", "_chunk", "_batch")))
        ok = not present and not static and not bad and not grad0 and not grad1 and not rg and q_ok and any(".update" in c for c in learn_calls)
        return {"status": PASSED if ok else FAILED, "legacy_learning_or_guide_objects_present": present, "static_forbidden_patterns": static,
                "forbidden_calls_traced": sorted(set(bad)), "grad_enabled_before_after": [grad0, grad1], "any_tensor_requires_grad": rg,
                "q_and_threshold_unchanged_by_training_batch": q_ok, "local_learning_calls_seen": learn_calls,
                "config_rejects_q_learning_keys": self._config_rejects(),
                "note_ko": "q 학습 (L1 오차 배달)·V1 안내 우회·강도 구간 게이트 코드가 이 파일에 없고, 실제 학습 배치 호출 경로에 autograd·optimizer 호출이 없다."}

    @staticmethod
    def _config_rejects() -> dict[str, bool]:
        out = {}
        for k, v in (("learning", {"eta": 0.01}), ("learning", {"alpha2": 0.5}), ("band", {"w_floor": 0.05}), ("destination", {"top_k": 1})):
            try:
                validate_config({k: v})
                out[f"{k}.{list(v)[0]}"] = False
            except ConfigError:
                out[f"{k}.{list(v)[0]}"] = True
        return out


# ======================================================================
# 15. 역할 검사 (통제 자극) — 측정값 (MEASURED). 통과 기준이 아니며 역할 이름이 생물학적 기능을 입증하지 않는다
# ======================================================================
class RoleProbes:
    """통제 자극에서: 층을 지나며 방향·공간 구별이 사라지는지 (최근접 복호·서로 다른 패턴 수·평균 코사인), L2 선택 코드가 L3 에서 합쳐지는지,
    V1 선택 후보의 고정 방향이 자극 방향과 맞는지, 선택됐지만 역할 출력이 0 인 비율. role_probe_metrics.csv 에 기록한다."""

    def __init__(self, sess: Session):
        self.s = sess

    def stimuli(self) -> dict[str, list[tuple[str, np.ndarray, dict[str, Any]]]]:
        S = self.s.pre.S
        fac = StimulusFactory(S, supersample=2)
        lw = max(1.5, S / 48.0)
        oris = [float(t) for t in self.s.cfg["areas"]["V1"]["orientations_deg"]]
        out: dict[str, list[tuple[str, np.ndarray, dict[str, Any]]]] = {"orientation": [], "position": [], "shape": []}
        for off in (0, 1):
            for o in oris:
                t_ = math.radians(o + 90.0)
                c = (off * S / 10.0 * math.cos(t_), off * S / 10.0 * math.sin(t_))
                out["orientation"].append((f"line_{o:g}_off{off}", fac.line(o, width=lw, center=c), {"orientation": o, "offset": off}))
        for k in range(8):
            a_ = 2 * math.pi * k / 8
            c = (S / 4.0 * math.cos(a_), S / 4.0 * math.sin(a_))
            out["position"].append((f"disc_{k}", fac.shape("circle", S / 8.0, center=c, fg=1.0, bg=0.0), {"position": k}))
        out["shape"] = [("arc", fac.arc(S / 4.0, width=lw, start_deg=0.0, span_deg=150.0), {"kind": "curve"}),
                        ("line", fac.line(0.0, width=lw), {"kind": "straight"}),
                        ("corner", fac.corner(0.0, 90.0, width=lw), {"kind": "corner"}),
                        ("cross", fac.cross(0.0, 90.0, width=lw), {"kind": "cross"}),
                        ("line_end", fac.rgb(fac.line_mask(0.0, width=lw, half=True)), {"kind": "line_end"})]
        return out

    def run(self, context: str = "manual") -> dict[str, Any]:
        s, be = self.s, self.s.be
        m = s.model
        areas = list(m.areas)
        keys = [f"{a}/{l}" for a in areas for l in LAYER_ORDER]
        rows: list[dict[str, Any]] = []
        res_all: dict[str, Any] = {"utc": utc_now(), "context": context, "model_version": s.model_version(), "areas": areas, "probes": {}}
        oris = np.asarray([float(t) for t in s.cfg["areas"]["V1"]["orientations_deg"]])

        def add(probe: str, key: str, metric: str, value: Any, n: int) -> None:
            v = _fnum(value) if not isinstance(value, (int, np.integer)) else int(value)
            rows.append({"utc": res_all["utc"], "context": context, "model_version": res_all["model_version"], "probe": probe,
                         "area": key.split("/")[0], "layer": key.split("/")[1] if "/" in key else key, "metric": metric, "value": v, "n_stimuli": n})

        for probe, stims in self.stimuli().items():
            X = be.f32(np.stack([s.pre.from_array(im) for _n, im, _m in stims]))
            res = m.forward(X, keep=keys, record=tuple(areas))
            n = len(stims)
            pr: dict[str, Any] = {}
            for key in keys:
                H = be.to_np(res[key]["out"]).astype(np.float64)
                nr = np.linalg.norm(H, axis=1)
                ok = nr > 0
                Hn = H[ok] / nr[ok, None]
                pc = Hn @ Hn.T if ok.sum() >= 2 else np.zeros((0, 0))
                off = pc[~np.eye(len(Hn), dtype=bool)] if len(Hn) >= 2 else np.zeros(0)
                scale = np.abs(H).max() if H.size else 0.0
                distinct = len({tuple(np.round(h / scale, 6)) for h in H}) if scale > 0 else (1 if n else 0)
                d = {"zero_vector_frac": float(1.0 - ok.mean()), "mean_pairwise_cos": float(off.mean()) if off.size else None, "distinct_patterns": distinct}
                if probe == "orientation":
                    k_ = len(oris)
                    tr, te = H[:k_], H[k_:]
                    trn, ten = np.linalg.norm(tr, axis=1), np.linalg.norm(te, axis=1)
                    hit = 0
                    for i in range(k_):
                        if ten[i] > 0 and (trn > 0).any():
                            cs = np.where(trn > 0, (tr @ te[i]) / np.where(trn > 0, trn, 1.0) / ten[i], -np.inf)
                            hit += int(int(np.argmax(cs)) == i)
                    d["nn_orientation_decoding_across_offset"] = hit / k_
                for mk, mv in d.items():
                    add(probe, key, mk, mv, n)
                pr[key] = d
            for a in areas:
                t = m.tables[a]
                recs = route_rec_np(be, res[f"{a}/route"]["buckets"])
                codes = []
                for i in range(n):
                    codes.append(tuple(np.concatenate([np.where(r_["has"][i], r_["k"][i], -1) for r_ in recs]).tolist()) if recs else ())
                H3 = be.to_np(res[f"{a}/L3"]["out"]).astype(np.float64)
                diff_code = merged = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if codes[i] != codes[j]:
                            diff_code += 1
                            merged += int(np.allclose(H3[i], H3[j], rtol=1e-7, atol=1e-12))
                add(probe, f"{a}/L2", "distinct_selection_codes", len(set(codes)), n)
                add(probe, f"{a}/L2", "pairs_with_different_L2_codes", diff_code, n)
                add(probe, f"{a}/L3", "pairs_different_L2_code_but_same_L3", merged, n)
                sel = sum(int(r_["has"].sum()) for r_ in recs)
                zero = sum(int((r_["has"] & (r_["u_sel"] <= float(s.cfg["neuron"]["threshold"]))).sum()) for r_ in recs)
                add(probe, f"{a}/L2", "selected_but_zero_output_frac", (zero / sel) if sel else None, n)
                pr[f"{a}/codes"] = {"distinct": len(set(codes)), "pairs_diff": diff_code, "pairs_merged_at_L3": merged}
                if a == "V1" and probe == "orientation":
                    l2 = m.specs["V1/L2"]
                    ch_ori = np.asarray(l2.channel_meta["orientation_deg"], np.float64)
                    match = tot = 0
                    for i, (_nm, _im, meta) in enumerate(stims):
                        for b, bk in enumerate(t.buckets):
                            h_ = recs[b]["has"][i]
                            if not h_.any():
                                continue
                            dst = bk.dst[np.nonzero(h_)[0], recs[b]["k"][i][h_]]
                            o_sel = ch_ori[dst % l2.n_ch]
                            match += int((d_ori(o_sel, meta["orientation"]) <= 15.0 + 1e-6).sum())
                            tot += int(h_.sum())
                    add(probe, "V1/L2", "selected_candidate_orientation_within_15deg_frac", (match / tot) if tot else None, n)
                    pr["V1/role_consistency"] = {"within_15deg": match, "selected": tot}
                if a == "V4" and probe == "shape":
                    for i, (nm, _im, _meta) in enumerate(stims):
                        for fname in ("curve", "straight", "junction", "lineend"):
                            pos = tot = 0
                            for b, bk in enumerate(t.buckets):
                                if t.family_names[bk.family] != fname:
                                    continue
                                h_ = recs[b]["has"][i]
                                tot += int(h_.sum())
                                pos += int((h_ & (recs[b]["u_sel"][i] > float(s.cfg["neuron"]["threshold"]))).sum())
                            add(f"shape_{nm}", "V4/L2", f"{fname}_selected_positive_output_frac", (pos / tot) if tot else None, 1)
            res_all["probes"][probe] = pr
        for row in rows:
            s.need_rec().append_csv(ROLE_CSV, row)
        res_all["status"] = MEASURED
        res_all["note_ko"] = "통제 자극 측정값이다. 통과 기준이 아니며 역할 설계의 우월성·생물학적 기능을 입증하지 않는다."
        s.need_rec().json(f"diagnostics/role_probes_{re.sub(r'[^A-Za-z0-9_]+', '_', context)}_{local_stamp()}.json", res_all)
        return res_all


# ======================================================================
# 16. 최종 test 평가 (선택된 IT 고정 모델, 사용자가 실행할 때만) 와 상태 요약
# ======================================================================
def test_evaluation(sess: Session) -> dict[str, Any]:
    s = sess
    area = s.model.active
    st = s.stage[area]
    if area != "IT" or st["status"] not in STAGE_TERMINAL or not st.get("selected_restored"):
        raise RuntimeError(f"test 는 IT 단계가 끝나 선택 체크포인트로 되돌린 뒤에만 쓴다 (현재 {area}, 상태 {st['status']}). 선택·비교에 test 를 쓰지 않는다.")
    prior = int(s.counters["test_evaluations"])
    dc_path = s.rec.p("design_changes.jsonl")
    changes = [json.loads(x) for x in dc_path.read_text(encoding="utf-8").splitlines() if x.strip()] if dc_path.is_file() else []
    test_sha = s.data.ids_sha("test")
    proj = project_test_history(s, test_sha)
    m = s.evaluate(area, "test", s.readouts[area], purpose="final_test_selected_IT")
    indep = {"prior_test_evaluations_this_run": prior, "design_changes_logged": len(changes),
             "design_changes_after_test_seen": sum(1 for c in changes if c.get("after_test_seen")), "test_ids_sha256": test_sha,
             "project_prior_test_evaluations_same_test_set": proj["n_prior_evaluations"], "project_runs_that_saw_this_test_set": proj["runs"],
             "project_history_sources": proj["sources"], "selected_epoch_IT": st["selected_epoch"]}
    indep["independent_first_look"] = prior == 0 and indep["design_changes_after_test_seen"] == 0 and proj["n_prior_evaluations"] == 0
    out = {"utc": utc_now(), "area": area, "model_version": m["model_version"], "checkpoint": s.last_checkpoint,
           "test_evaluation_number": s.counters["test_evaluations"], "metrics": m, "independence": indep,
           "note_ko": "test 는 선택된 고정 IT 모델의 최종 평가다. 이 결과로 설정·에폭·모델을 고르지 않는다. 반복 평가·test 를 본 뒤의 변경은 독립 시험이 아니다."}
    s.rec.json(f"diagnostics/test_eval_{s.counters['test_evaluations']:03d}.json", out)
    PreChecks(s)._append_index("test_eval", {"utc": out["utc"], "model_version": out["model_version"], "status": MEASURED})
    led = project_ledger_path(s)
    if led is not None:
        try:
            with led.open("a", encoding="utf-8") as fh:
                fh.write(dumps({"utc": out["utc"], "run_dir": str(s.rec.dir), "run_test_evaluation_number": out["test_evaluation_number"],
                                "test_ids_sha256": test_sha, "model_version": out["model_version"], "program_version": VERSION, "mode": MODE},
                               indent=None) + "\n")
        except OSError as exc:
            s.rec.warn(f"프로젝트 test 장부 쓰기 실패 ({led}): {exc}")
    try:
        build_report(s)
    except Exception as exc:                                       # noqa: BLE001  (test 기록은 이미 저장됨)
        s.rec.warn(f"test 평가 뒤 보고서 갱신 실패: {type(exc).__name__}: {exc}")
    return out


def state_summary(sess: Session) -> dict[str, Any]:
    s = sess
    out: dict[str, Any] = {"model_version": s.model_version(), "mode": MODE, "q_fixed": float(s.cfg["neuron"]["q"]),
                           "threshold_fixed": float(s.cfg["neuron"]["threshold"]), "areas": {}}
    for a in s.model.areas:
        st = s.stage[a]
        t = s.model.tables[a]
        _C, ok = s.model.routers[a].get_codebook()
        out["areas"][a] = {"stage_status": st["status"], "epochs_completed": st["epochs_completed"], "max_epochs": st["max_epochs"],
                           "next_batch": st["next_batch"], "best_dev_accuracy": st["best_dev_accuracy"], "best_epoch": st["best_epoch"],
                           "selected_epoch": st["selected_epoch"], "selected_restored": st["selected_restored"], "frozen": a in s.frozen,
                           "codebook_version": s.model.routers[a].version, "valid_codes": int(sum(o.sum() for o in ok)),
                           "legal_candidates": int(t.r_legal.sum()), "readout_version": s.readouts[a].version if a in s.readouts else None,
                           "readout_invalid_classes": [c for c, v in zip(s.classes, s.readouts[a].valid) if not v] if a in s.readouts else None}
    return out


def project_ledger_path(sess: Session) -> Path | None:
    root = sess.output_root or (sess.rec.dir.parent if sess.rec else None)
    return Path(root) / "test_set_history.jsonl" if root else None


def project_test_history(sess: Session, test_sha: str) -> dict[str, Any]:
    """같은 저장 폴더에서 같은 test 표본 집합을 이미 평가한 이력 (이 실행 제외).
    원천: 프로젝트 장부 test_set_history.jsonl + 형제 실행 폴더의 diagnostics/test_eval_*.json (장부 이전 v1·v2 실행 포함)."""
    me = str(sess.rec.dir.resolve()) if sess.rec else ""
    seen: dict[tuple[str, int], str] = {}
    led = project_ledger_path(sess)
    sources = []
    if led is not None and led.is_file():
        sources.append(str(led))
        for line in led.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            rd = str(Path(e.get("run_dir", "")).resolve()) if e.get("run_dir") else ""
            if e.get("test_ids_sha256") == test_sha and rd and rd != me:
                seen[(rd, int(e.get("run_test_evaluation_number", 0)))] = "ledger"
    root = led.parent if led is not None else None
    if root is not None and root.is_dir():
        sources.append(f"{root}/*/diagnostics/test_eval_*.json")
        for d in sorted(x for x in root.iterdir() if x.is_dir()):
            rd = str(d.resolve())
            if rd == me:
                continue
            man = d / "split_manifest.json"
            tests = sorted((d / "diagnostics").glob("test_eval_*.json")) if (d / "diagnostics").is_dir() else []
            if not tests or not man.is_file():
                continue
            try:
                items = read_json(man).get("items", [])
                sha = sha256_text("\n".join(sorted(it["relpath"] for it in items if it.get("split") == "test")))
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if sha != test_sha:
                continue
            for t in tests:
                try:
                    n = int(read_json(t).get("test_evaluation_number", 0))
                except (OSError, ValueError, TypeError):
                    continue
                seen.setdefault((rd, n), "run_folder_scan")
    return {"n_prior_evaluations": len(seen), "runs": sorted({k[0] for k in seen}), "sources": sources}


# ======================================================================
# 17. 추론 (저장 모델, 갱신 없음)
# ======================================================================
def list_images(path: Path, exts: Sequence[str]) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        e = {x.lower() for x in exts}
        return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in e])
    raise FileNotFoundError(f"이미지 파일·폴더가 없다: {path}")


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


def run_inference(sess: Session, path: Path, *, area: str | None = None, figure: bool = True) -> dict[str, Any]:
    """저장 모델로 추론 (코드북·판독기 갱신 없음, 라벨 없음). 판독기가 현재 코드북과 다른 스냅샷이면 표시한다."""
    s, be = sess, sess.be
    area = area or s.model.active
    ro = s.readouts.get(area)
    if ro is None:
        raise RuntimeError(f"{area} 판독기가 없다.")
    files = list_images(path, s.cfg["data"]["extensions"])
    if not files:
        raise FileNotFoundError(f"지원 이미지가 없다: {path}")
    eps = float(s.cfg["learning"]["eps_norm"])
    same_snapshot = ro.meta.get("codebook_sha256") == s.model.routers[area].codebook_sha()
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
        res = s.model.forward(X, upto=area, keep=(f"{area}/L2", f"{area}/L3"), record=(area,))
        h = res[f"{area}/L3"]["out"]
        cl = cosine_classify(be, h, ro.P, ro.valid, eps)
        pred = int(cl["pred"][0])
        st_ = RouteStats(s.model, [area])
        st_.add_route(area, route_rec_np(be, res[f"{area}/route"]["buckets"]))
        fam = st_.family_rows(area)
        row = {"image": str(p), "status": "ok" if pred >= 0 else "unclassifiable", "prediction": s.classes[pred] if pred >= 0 else "unclassifiable",
               "output_norm": float(cl["h_norm"][0]), "area": area, "model_version": s.model_version(area), "checkpoint": s.last_checkpoint,
               "readout_matches_current_codebook": same_snapshot, "orig_size": [meta["orig_w"], meta["orig_h"]]}
        for c, name in enumerate(s.classes):
            cv = cl["cos"][0, c]
            row[f"cos_{name}"] = float(cv) if np.isfinite(cv) else None
        for f_, x_ in fam.items():
            row[f"route_{f_}_selected_frac"] = x_["selected_frac"]
            row[f"route_{f_}_selected_but_zero_output_frac"] = x_["selected_but_zero_output_frac"]
        rows.append(row)
        if first is None:
            first = (p, {k: be.to_np(v_["out"])[0] for k, v_ in res.items() if k.startswith(area + "/L")})
    s.counters["inference_images"] += len(files)
    for r_ in rows:
        s.rec.append_csv(f"inference/{out_dir.name}/inference_results.csv", r_)
    write_json(out_dir / "inference_results.json", {"utc": utc_now(), "source": str(path), "area": area, "rows": rows,
                                                    "readout_matches_current_codebook": same_snapshot,
                                                    "note_ko": "코사인 유사도는 확률이나 보정된 신뢰도가 아니다. 추론 중 코드북·판독기 갱신 없음."})
    figs = []
    if figure and first is not None:
        f = save_feature_figure(s, area, first[1], out_dir / "features_first_image.png", title=first[0].name)
        if f:
            figs.append(str(f))
    return {"rows": rows, "out_dir": str(out_dir), "figures": figs, "readout_matches_current_codebook": same_snapshot}


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
        ax.set_title(f"{sp.key}: {sp.n_pos} pos x {sp.n_ch} ch")
    t = m.tables[area]
    fig.suptitle(f"{area} RF centers (image px); routing groups {t.n_groups}, legal candidates {int(t.r_legal.sum())}", fontsize=9)
    return _fig_save(sess, fig, sess.rec.p(f"figures/structure_{area}.png"))


def save_training_figure(sess: Session, area: str) -> Path | None:
    plt = _plt()
    rows = [r for r in sess.rec.read_csv(TRAINING_CSV) if r.get("area") == area]
    if plt is None or not rows:
        return None
    ep = [int(r["epoch_number"]) for r in rows]
    fl = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else np.nan        # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(ep, [fl(r, "dev_accuracy") for r in rows], "o-", label="dev acc (train-fitted readout)")
    axes[0].plot(ep, [fl(r, "train_accuracy") for r in rows], "s--", label="train acc")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].set_xlabel("epoch (0 = init codebook baseline)")
    axes[1].plot(ep, [fl(r, "dev_unclassifiable_frac") for r in rows], "o-", label="dev unclassifiable frac")
    axes[1].plot(ep, [fl(r, "dev_L2_zero_vector_frac") for r in rows], "s--", label="dev L2 zero-vector frac")
    axes[1].legend()
    axes[1].set_xlabel("epoch")
    fig.suptitle(f"{area} (status {sess.stage[area]['status']}, selected epoch {sess.stage[area].get('selected_epoch')})")
    return _fig_save(sess, fig, sess.rec.p(f"figures/training_{area}.png"))


def save_usage_figure(sess: Session, area: str) -> Path | None:
    """마지막 에폭 학습 배치의 후보별 선택 횟수 분포 (계열별)."""
    plt = _plt()
    files = sorted(sess.rec.p("codebook").glob(f"usage_{area}_e*.npz"))
    if plt is None or not files:
        return None
    z = load_npz(files[-1])
    t = sess.model.tables[area]
    fig, ax = plt.subplots(figsize=(7, 4))
    for fi, fam in enumerate(t.family_names):
        w = np.concatenate([z[f"b{b}__wins_train"][bk.legal] for b, bk in enumerate(t.buckets) if bk.family == fi] or [np.zeros(0)])
        if w.size:
            ax.hist(np.log10(1 + w), bins=30, alpha=0.5, label=f"{fam} (unused {float((w == 0).mean()):.2f})")
    ax.set_xlabel("log10(1 + selections per candidate) — last epoch training batches")
    ax.set_ylabel("candidates")
    ax.legend(fontsize=8)
    ax.set_title(f"{area} candidate usage ({files[-1].stem})")
    return _fig_save(sess, fig, sess.rec.p(f"figures/usage_{area}.png"))


def save_readout_figure(sess: Session, area: str) -> Path | None:
    plt = _plt()
    if plt is None or area not in sess.readouts:
        return None
    pc = sess.readouts[area].pairwise_cos()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(pc, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(sess.classes)), sess.classes, rotation=45, ha="right")
    ax.set_yticks(range(len(sess.classes)), sess.classes)
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{area} class readout P_y cosine (v{sess.readouts[area].version})")
    return _fig_save(sess, fig, sess.rec.p(f"figures/readout_similarity_{area}.png"))


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
        for f in (save_structure_figure(sess, a), save_training_figure(sess, a), save_usage_figure(sess, a), save_readout_figure(sess, a)):
            if f is not None:
                out.append(str(f))
    return out


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    L = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    return L


def build_report(sess: Session) -> Path:
    """REPORT.md — 저장된 JSON/CSV 에서만 숫자를 읽는다 (손으로 옮긴 숫자 없음). 미실행 항목은 '미실행' 으로 쓴다."""
    r = sess.need_rec()
    L: list[str] = [f"# REPORT — {PROGRAM} v{VERSION}", "", f"- 생성 UTC: {utc_now()} · 실행 폴더 `{r.dir}`",
                    f"- 모드 `{MODE}` · 모델 상태 `{sess.model_version() if sess.model else '모델 없음'}` · 마지막 checkpoint `{sess.last_checkpoint}`",
                    f"- 계산 뒷단 {sess.be.kind} / {sess.be.device}",
                    f"- 기준 파일: `{BASE_FILE_INFO['file']}` v{BASE_FILE_INFO['version']} (SHA256 `{BASE_FILE_INFO['sha256'][:16]}…`) — {BASE_FILE_INFO['verified_ko']}", "",
                    "## 이번 실험이 하는 것 (해석의 범위)", "",
                    "- 각 영역 L4 → L2 에서 입력 비율 방향 x̂ 과 후보별 대표 벡터 c 의 코사인으로 집단마다 후보 묶음 하나를 고르고 (hard top-1), "
                    "선택된 묶음의 실제 L4 값만 기존 역할 연산으로 전달한다. 학습은 선택된 c 를 그 입력 방향 쪽으로 조금 옮기는 **국소 경쟁학습** 이다.",
                    "- **라우팅 대표 벡터 c[g,k]** = 학습 대상 (활성 영역만). **q** = 1 고정 (학습 안 함, CONF = q·s 표기 유지). "
                    "**클래스 판독기 P_y** = 평가 때마다 train 클래스 평균으로 다시 만드는 통계적 분류기 (코드북과 별개, 오차를 되돌려 보내지 않음).",
                    "- 기존 시냅스 가중치 (Gabor·윤곽·결합 가중치) 는 학습하지 않는다. L2→L3·L3→다음 L4 는 고정 결합이다.",
                    "- 다음 주장은 하지 않는다: 코사인이 높아져 분류가 좋아진다, 갱신이 일어났으니 역할을 학습했다, 원핫이라 정보 손실이 없다, "
                    "모든 후보가 학습된다, 출력이 0 이 아니니 특징이 충분하다, 역할 이름이 생물학적 기능을 입증한다, 미분이 없으니 뇌의 학습을 재현했다.", ""]
    if sess.data is not None:
        man = sess.data.manifest
        L += ["## 데이터", "", f"- 클래스 {man['classes']}, 항목 {man['n_items']}, 지문 `{man['data_fingerprint_sha256'][:16]}`, 분할 `{man['split_sha256'][:16]}`"]
        info = read_json(r.p("run_info.json")) if r.p("run_info.json").is_file() else {}
        if info.get("reused_split_from"):
            ru = info["reused_split_from"]
            L.append(f"- 분할 재사용: `{ru['source_run']}` ({ru.get('source_program')} v{ru.get('source_program_version')}) — 모델 상태는 가져오지 않았다")
        L += [""] + _md_table(["클래스", "train", "dev", "test"], [[c, v["n_train_images"], v["n_dev_images"], v["n_test_images"]]
                                                                 for c, v in man["per_class"].items()]) + [""]
    sj = read_json(r.p("structure.json")) if r.p("structure.json").is_file() else None
    if sj:
        L += ["## 구조와 후보 그래프 (structure.json)", ""]
        rows = []
        for a, v in sj["areas"].items():
            ro = v["routing"]
            n = {k: v["layers"][k]["n_neurons"] for k in LAYER_ORDER}
            rows.append([a, n["L4"], n["L2"], n["L3"], ro["n_groups"], ro["n_legal_candidates"], ro["n_rejected_candidates"],
                         ro["destinations_without_legal_candidate"], ro["sum_K_times_D"], ro["padded_K_times_D"], ro["codebook_memory_mb_float32"],
                         ro["score_macs_per_sample"]])
        L += _md_table(["영역", "L4", "L2", "L3", "집단", "합법 후보", "탈락 후보", "후보 없는 목적지", "Σ K·D", "패딩 포함", "코드북 MB", "점수 MAC/표본"], rows)
        L += ["", "계산량: 점수 O(batch × Σ_g K_g D_g), 코드북 메모리 O(Σ_g K_g D_g). 원핫 전달이어도 후보 비교 비용은 그대로다.", "",
              "### 영역별 차원 (실제 코드에서 센 값)", ""]
        rows = []
        for a, v in sj["areas"].items():
            dt = v["dimension_table"]
            for fam, fr in v["routing"]["families_detail"].items():
                rows.append([a, dt["L4_input_meaning"], f"{fam}: {dt['L2_role_family']}", dt["L3_processing"], dt["must_check"], fr["n_groups"],
                             f"{fr['candidates_per_group_K']['min']}–{fr['candidates_per_group_K']['max']} (평균 {fmt(fr['candidates_per_group_K']['mean'], 1)})",
                             f"{fr['descriptor_dim_D']['min']}–{fr['descriptor_dim_D']['max']} (평균 {fmt(fr['descriptor_dim_D']['mean'], 1)})",
                             f"{fr['support_dim']['min']}–{fr['support_dim']['max']}", fr["candidates_with_support_1_fixed_code"],
                             f"{fmt(fr['L2_L3_merge_mean_per_group']['candidates'], 1)} → L2 {fmt(fr['L2_L3_merge_mean_per_group']['distinct_L2_destinations'], 1)}"
                             f" → L3 {fmt(fr['L2_L3_merge_mean_per_group']['distinct_L3_parent_sets'], 1)}"])
        L += _md_table(["영역", "L4 입력 의미", "L2 역할 계열", "L3 처리", "반드시 확인할 문제", "집단", "후보 K", "입력 차원 D", "지원 차원",
                        "지원 1 (고정 코드)", "후보 → 목적지 → L3 부모 집합 (집단 평균)"], rows)
        L += ["", "- '지원 1' 후보는 대표 벡터가 단위축으로 고정되어 학습해도 바뀌지 않는다 (자유도 0). 후보 → L3 부모 집합 수가 줄면 서로 다른 선택이 L3 에서 합쳐진다.", ""]
    L += ["## 코드북 초기화 (train 만, 라벨 미사용 — 데이터로부터의 학습 과정)", ""]
    rows = []
    for p in sorted(r.p("codebook").glob("init_*.json")):
        j = read_json(p)
        for fam, v in j["families"].items():
            rows.append([j["area"], fam, j["n_samples"], v["n_legal_candidates"], v["n_code_ok"], v["n_failed"], v["n_role_positive"],
                         v["n_support_mass_fallback"], v["n_duplicate_packet"], v["n_duplicate_direction"], fmt(v["mean_distinct_directions_per_group"], 1),
                         v["n_groups_without_active_packet"]])
    L += (_md_table(["영역", "계열", "표본", "합법 후보", "대표 생성", "초기화 실패", "양의 역할 반응 패킷", "지원 질량 fallback", "패킷 중복", "방향 중복",
                     "집단당 서로 다른 방향", "활성 패킷 없는 집단"], rows) if rows else ["- 미실행"]) + [""]
    trs = r.read_csv(TRAINING_CSV)
    L += ["## 단계 학습 (training_metrics.csv)", ""]
    if sess.model is not None:
        for a in sess.model.areas:
            st = sess.stage[a]
            rr = [x for x in trs if x.get("area") == a]
            term = {"success_criterion_met": "성공 기준 도달", "plateau": f"정체 (patience {st['patience']} 에폭 동안 dev 개선 없음 — 성공이 아니다)",
                    "budget_exhausted": "에폭 예산 소진", "stopped_by_user_progression": "사용자가 다음 영역으로 진행 (단계 미완료)"}.get(st["status"], st["status"])
            L.append(f"### {a} — 상태 `{st['status']}` ({term}), 에폭 {st['epochs_completed']}/{st['max_epochs']}, 선택 에폭 {st.get('selected_epoch')} "
                     f"(dev {fmt(st['best_dev_accuracy'])}), 선택 상태로 복원 {st.get('selected_restored')}")
            if rr:
                L += [""] + _md_table(["epoch", "종류", "train", "dev", "dev 분류 불능", "dev L2 영벡터", "dev L3 활성", "L6 g=max (L2 활성 중)", "선택(지금까지)", "상태", "시간 s"],
                                      [[x["epoch_number"], "초기 기준" if x.get("row_kind") == "init_baseline" else "학습",
                                        pct_count(int(float(x["train_correct"])) if x.get("train_correct") else None, int(float(x["train_n"])) if x.get("train_n") else None),
                                        pct_count(int(float(x["dev_correct"])), int(float(x["dev_n"]))), x.get("dev_unclassifiable"),
                                        fmt(x.get("dev_L2_zero_vector_frac")), fmt(x.get("dev_L3_active_frac")), fmt(x.get("dev_L2_g_at_max_frac")),
                                        x.get("selected_epoch_so_far"), x.get("stage_status"),
                                        fmt(sum(float(v) for k, v in x.items() if k.startswith("time_") and v not in ("", None)), 1)] for x in rr])
                L.append("")
                L.append("- 에폭 0 은 '초기화 코드북 + train 판독기 (코드북 갱신 전)' 기준이다. 라벨을 전혀 쓰지 않은 무학습 모델이 아니다 (판독기와 초기화는 train 을 썼다).")
            L.append("")
    rms = r.read_csv(ROUTING_CSV)
    L += ["## 라우팅 집계 (routing_metrics.csv, 마지막 dev 평가 — 전체 dev 집계)", ""]
    if rms:
        last_ep = {}
        for x in rms:
            if x.get("context") in ("dev_eval", "dev_init_baseline"):
                last_ep[x["active_area"]] = x["epoch_number"]
        rows, lrows = [], []
        for x in rms:
            if x.get("context") not in ("dev_eval", "dev_init_baseline") or x["epoch_number"] != last_ep.get(x["active_area"]):
                continue
            if x["active_area"] != (sess.model.active if sess.model else x["active_area"]):
                continue
            if x["family"].startswith("layer_"):
                lrows.append([x["area"], x["family"][6:], fmt(x.get("zero_vector_frac")), fmt(x.get("active_neuron_frac")), fmt(x.get("mean_output_norm")),
                              fmt(x.get("output_variance_mean")), fmt(x.get("L6_g_at_max_frac_of_active")), fmt(x.get("L6_g_at_min_frac_of_active")),
                              fmt(x.get("L6_g_at_max_frac"))])
            else:
                rows.append([x["area"], x["family"], fmt(x.get("selected_frac")), fmt(x.get("nonzero_input_but_no_selection_frac")),
                             fmt(x.get("selected_but_zero_output_frac")), fmt(x.get("mean_max_cosine")), fmt(x.get("mean_top1_minus_top2")),
                             fmt(x.get("exact_tie_frac")), fmt(x.get("near_tie_frac")), fmt(x.get("unused_candidate_frac")),
                             fmt(x.get("mean_effective_used_candidates"), 2), fmt(x.get("monopoly_groups_frac")),
                             f"{fmt(x.get('min2_A_only_frac'), 3)}/{fmt(x.get('min2_B_only_frac'), 3)}/{fmt(x.get('min2_both_frac'), 3)}"])
        L += _md_table(["영역", "계열", "선택 비율", "비영 입력·선택 없음", "선택·출력 0", "평균 최대 cos", "1·2위 차", "정확 동점", "거의 동점", "미사용 후보",
                        "유효 사용 후보", "독점 집단", "min2 A만/B만/둘다"], rows) + [""]
        L += _md_table(["영역", "층", "영벡터 비율", "활성 뉴런 비율", "평균 출력 노름", "출력 분산 평균", "L6 g=max (활성 중)", "L6 g=min (활성 중)",
                        "L6 g=max (전체)"], lrows) + [""]
        L.append("- 동결 영역도 입력마다 L6·목적지 선택을 다시 계산한 값이다. 학습 배치 집계 (context = train_epoch_learning) 는 같은 CSV 에 있다.")
    else:
        L.append("- 미실행")
    L.append("")
    cms = r.read_csv(CODEBOOK_CSV)
    L += ["## 코드북 갱신 (codebook_metrics.csv)", ""]
    ep_rows = [x for x in cms if x.get("context") == "epoch_updates"]
    if ep_rows:
        L += _md_table(["영역", "epoch", "계열", "갱신 횟수", "한 번 이상 갱신된 후보 비율", "평균 갱신 노름", "평균 갱신 각 (°)", "초기값 대비 평균 각 (°)", "최대 각 (°)"],
                       [[x["area"], x["epoch_number"], x["family"], x.get("candidate_updates"), fmt(x.get("candidates_updated_frac")),
                         fmt(x.get("mean_update_norm")), fmt(x.get("mean_update_angle_deg"), 3), fmt(x.get("mean_angle_from_init_deg"), 3),
                         fmt(x.get("max_angle_from_init_deg"), 3)] for x in ep_rows[-24:]])
    else:
        L.append("- 학습 에폭 미실행")
    L.append("")
    rps = r.read_csv(ROLE_CSV)
    L += ["## 역할 검사 (role_probe_metrics.csv, 통제 자극 측정값)", ""]
    if rps:
        ctx = rps[-1]["context"]
        sel = [x for x in rps if x["context"] == ctx and x["metric"] in ("nn_orientation_decoding_across_offset", "distinct_patterns", "zero_vector_frac",
                                                                          "selected_candidate_orientation_within_15deg_frac",
                                                                          "pairs_different_L2_code_but_same_L3", "distinct_selection_codes")]
        L += [f"- 마지막 측정 맥락 `{ctx}` ({len(sel)} 행 발췌)", ""] + _md_table(["자극", "영역", "층", "지표", "값", "자극 수"],
                                                                            [[x["probe"], x["area"], x["layer"], x["metric"], fmt(x["value"]), x["n_stimuli"]] for x in sel[:80]])
    else:
        L.append("- 미실행")
    L.append("")
    if sess.model is not None and trs:
        top = sess.model.active
        rr = [x for x in trs if x.get("area") == top]
        sel_ep = sess.stage[top].get("selected_epoch")
        pick = [x for x in rr if str(x.get("epoch_number")) == str(sel_ep)] or rr[-1:]
        if pick:
            cm = json.loads(pick[-1]["dev_confusion"]) if pick[-1].get("dev_confusion") else None
            pca = json.loads(pick[-1]["dev_per_class_accuracy"]) if pick[-1].get("dev_per_class_accuracy") else {}
            L += [f"## {top} 선택 에폭 ({pick[-1]['epoch_number']}) dev confusion matrix (행 = 정답, 열 = 예측, 마지막 열 = 분류 불능)", ""]
            if cm:
                L += _md_table(["정답\\예측"] + sess.classes + ["불능"], [[c] + row for c, row in zip(sess.classes, cm)])
            L += ["", "- 클래스별 dev 정확도: " + ", ".join(f"{k} {fmt(v)}" for k, v in pca.items()), ""]
    tests = sorted(r.p("diagnostics").glob("test_eval_*.json"))
    L += ["## 최종 test 평가", ""]
    if tests:
        for p in tests:
            t = read_json(p)
            m = t["metrics"]
            L.append(f"- #{t['test_evaluation_number']} {t['utc']} {t['model_version']}: {pct_count(m['correct'], m['n'])} "
                     f"(분류 불능 {m['n_unclassifiable']}, 독립 첫 평가 {t['independence']['independent_first_look']})")
    else:
        L.append("- 미실행 (test 는 IT 선택 뒤 사용자가 메뉴 7-4 / --test-eval 로 실행할 때만 쓴다)")
    dj = r.p("diagnostics.json")
    L += ["", "## 사전 검사·진단", ""]
    if dj.is_file():
        for run in read_json(dj)["runs"][-4:]:
            L.append(f"- {run['utc']} {run['kind']} ({run['model_version']}): {run['status']}")
            for k, v in (run.get("checks") or {}).items():
                L.append(f"  - {k}: {v}")
    else:
        L.append("- 미실행")
    L += ["", "## 읽기·평가 횟수·자원", ""]
    L.append(f"- 계수기: {sess.counters}")
    if sess.loader is not None:
        L.append(f"- 분할별 읽은 표본 수 (이 세션): {sess.loader.access}")
    if trs:
        last = trs[-1]
        tm = {k: v for k, v in last.items() if k.startswith("time_")}
        L.append(f"- 마지막 행 시간 (초): {tm}, GPU 최대 메모리 MB {last.get('gpu_max_memory_allocated_mb') or '미측정 (GPU 아님)'}, "
                 f"gather 조각 상한 {last.get('max_gather_elements_now')}, LGN 캐시 MB {last.get('cache_mb')}")
    L += ["", "## 실행/미실행 항목", ""]
    for k, v in VERIFICATION_NOTES.items():
        L.append(f"- {k}:")
        L += [f"  - {x}" for x in v]
    L += ["", "## 한계 (자동 기록)", "",
          "- hard top-1 은 집단마다 후보 하나만 남겨 나머지 후보의 특징을 버린다. L2→L3 풀링에서 서로 다른 선택이 같은 출력으로 합쳐질 수 있다 (구조·역할 검사 표).",
          "- 한 후보가 집단을 독점하거나 한 번도 선택되지 않는 후보가 생길 수 있다 (미사용 비율·유효 사용 후보 수·독점 집단 비율). 이번 실험은 이를 고치는 장치를 넣지 않았다.",
          "- 코드북이 입력 방향을 따라가도 고정 역할 필터와 맞지 않으면 선택됐지만 출력 0 이 된다 (selected_but_zero_output). 역할 학습 성공을 뜻하지 않는다.",
          "- 지원 차원 1 인 후보 (V4 접합 중계 등) 와 초기화 실패 후보는 학습으로 바뀌지 않는다.",
          "- 순차 동결이라 앞 영역의 정보 손실은 뒤 영역에서 고칠 수 없다. 판독기는 train 클래스 평균이며 dev 정확도로 에폭만 고른다.",
          "- 가정 전체: assumptions.md, 수식: config.json 의 formulas.", ""]
    return r.text("REPORT.md", "\n".join(L))


# ======================================================================
# 17. 한국어 메뉴와 CLI
# ======================================================================
_INPUT_STATE = {"closed": False}                    # 표준 입력이 닫히면 (EOF) 메뉴가 무한 반복하지 않게


def ask(prompt: str, default: str | None = None) -> str:
    sfx = f" [{default}]" if default not in (None, "") else ""
    try:
        v = input(f"{prompt}{sfx}: ")
    except EOFError:
        _INPUT_STATE["closed"] = True
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
    ro = summ["routing"]
    say(f"  라우팅: 집단 {ro['n_groups']} (후보 없는 집단 {ro['groups_without_legal_candidate']}), 후보 레코드 {ro['n_candidate_records']} "
        f"(합법 {ro['n_legal_candidates']}, 탈락 {ro['n_rejected_candidates']}), 후보 없는 목적지 {ro['destinations_without_legal_candidate']}, "
        f"Σ K·D {ro['sum_K_times_D']} (패딩 포함 {ro['padded_K_times_D']}), 점수 MAC/배치 {ro['score_macs_per_batch']}")
    m = summ["memory_mb"]
    say(f"  예상 메모리 (MB): 코드북 {m['codebook']}, 후보 배열 {m['route_arrays']}, 학습 합 {m['learning_sums_active_area']}, 고정 CSR {m['graph']}, "
        f"배치 활동 상한 {m['batch_activations_upper']}, gather 조각 상한 {m['gather_chunk_upper']}, 기존 모델 {m['existing_model']}, "
        f"합계 약 {m['total_estimate']} (배치 {summ['batch_size']}, {summ['device']})")
    say("  학습 대상: " + "; ".join(summ["learned_items_ko"]))
    say("  고정 항목: " + "; ".join(summ["fixed_items_ko"]))
    say("  " + summ["note_ko"])


def print_train_result(res: dict[str, Any], say: Callable[[str], None] = print) -> None:
    say(f"  결과: {res.get('area')} 상태 {res.get('status')}, 완료 에폭 {res.get('epochs_completed')}, best dev {fmt(res.get('best_dev_accuracy'))}"
        f" (선택 에폭 {res.get('best_epoch')}), checkpoint {res.get('checkpoint')}")
    if res.get("message_ko"):
        say("  " + res["message_ko"])


def pretraining_lines(s: Session) -> list[str]:
    """장기 학습 전에 보여 줄 실제 영역별 뉴런 수·후보 수·코드북 메모리·학습 대상과 고정 항목 (structure.json 에서 읽는다)."""
    out = []
    sj = read_json(s.rec.p("structure.json")) if s.rec and s.rec.p("structure.json").is_file() else {"areas": {}}
    for a, v in sj["areas"].items():
        ro = v["routing"]
        n = sum(v["layers"][k]["n_neurons"] for k in LAYER_ORDER)
        out.append(f"{a}: 뉴런 {n} (L4 {v['layers']['L4']['n_neurons']}, L2 {v['layers']['L2']['n_neurons']}, L3 {v['layers']['L3']['n_neurons']}), "
                   f"집단 {ro['n_groups']}, 합법 후보 {ro['n_legal_candidates']}, 코드북 {ro['codebook_memory_mb_float32']} MB"
                   + (" ← 학습 대상" if a == s.model.active else " (동결: 입력마다 계산만)"))
    out.append(f"학습 대상: {s.model.active} 라우팅 대표 벡터 c[g,k] (η = {s.cfg['routing']['eta_codebook']}, 배치 {s.cfg['learning']['batch_size']}). "
               f"고정: 후보 그래프·역할 가중치·θ = {s.cfg['neuron']['threshold']}·q = {s.cfg['neuron']['q']}·L2→L3·L3→L4·L6·아래 영역 코드북.")
    st = s.stage[s.model.active]
    n_train = len(s.data.items("train"))
    out.append(f"에폭 예산 {st['max_epochs']} (patience {st['patience']}), 에폭마다 train {n_train} 장 학습 + 판독기용 train 순방향 "
               f"{1 if s.can_store_h(s.model.active) or not s.cfg['records']['train_accuracy_each_epoch'] else 2} 회 + dev {len(s.data.items('dev'))} 장 평가.")
    return out


class MenuApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = config_from_cli(args)
        print(f"{PROGRAM} v{VERSION} — 모든 영역 입력 비율 라우팅·국소 대표 벡터 학습 (공학적 근사, 학습 성공을 보장하지 않음)")
        self.be = self._backend(args.backend, args.device)
        self.sess = Session(self.cfg, self.be, say=print, confirm=ask_yes)
        DeviceManager.print_environment(DeviceManager.environment(self.be))
        try:
            self.sess.set_paths(clean_user_path(args.data_root) if args.data_root else None,
                                clean_user_path(args.output_dir) if args.output_dir else None)
        except Exception as exc:                             # noqa: BLE001
            print(f"  [경로 오류] {exc}")
        if args.resume:                                      # --resume 만 주면 그 v5 run 을 불러온 상태로 메뉴를 시작한다
            try:
                self.sess = Session.load(clean_user_path(args.resume), self.be, say=print, confirm=ask_yes,
                                         data_root=clean_user_path(args.data_root) if args.data_root else None)
            except (CompatibilityError, FileNotFoundError, DataError, StructureError, ConfigError) as exc:
                print(f"  [재개 오류] {type(exc).__name__}: {exc}")
                print("  빈 메뉴로 시작한다. 구버전 run 은 2) 에서 분할만 재사용할 수 있다.")

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
            model = f"{'/'.join(s.model.areas)} (활성 {a}: {st['status']}, 에폭 {st['epochs_completed']}/{st['max_epochs']}, 선택 {st.get('selected_epoch')})"
        return (f"데이터({s.cfg['data']['source']}) {s.data_root or '미설정'} | 저장 {s.output_root or '미설정'} | run {run} | 모델 {model} | "
                f"{MODE} | {self.be.kind}/{self.be.device}")

    def loop(self) -> None:
        menu = ["1) 데이터 폴더와 저장 폴더 설정 / 환경 확인", "2) 데이터 검사 및 학습·검증·시험 분할 (또는 기존 run 의 분할 재사용)",
                "3) 새 V1 모델 생성 (구조·후보·코드북 메모리 → train 코드북 초기화 → 에폭 0 기준 평가)",
                "4) 현재 활성 영역 학습 또는 중단 지점 재개", "5) 다음 영역 추가 (현재 영역 선택 상태로 동결)", "6) 저장 모델로 이미지 한 장 또는 폴더 추론",
                "7) 진단 (사전 검사 16·역할 검사·3×3 조회·최종 test·상태)", "8) 모델·로그·그림·REPORT.md 저장", "9) 라우팅·코드북 통계 보기 / 설계 변경 기록", "0) 종료"]
        while True:
            print("\n" + "=" * 78)
            print(" " + self.status())
            print("-" * 78)
            for m in menu:
                print("  " + m)
            c = ask("선택")
            if c == "" and _INPUT_STATE["closed"]:
                print("  입력이 닫혔다 (EOF). 메뉴를 끝낸다. 추가 저장은 하지 않는다 (마지막 checkpoint 는 그대로 남는다).")
                return
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

    def m1(self) -> None:
        s = self.sess
        if s.rec is None and s.data is None and s.model is None:
            source = ask("데이터 종류 (mnist/folder, Enter = 유지)", s.cfg["data"]["source"]).lower()
            if source not in ("mnist", "folder"):
                raise ConfigError("데이터 종류는 mnist 또는 folder 다.")
            s.cfg["data"]["source"] = source
            self.cfg = copy.deepcopy(s.cfg)
        desc = ("MNIST 캐시 폴더 (없으면 생성, 메뉴 2에서 공식 .gz 내려받기/검사)"
                if s.cfg["data"]["source"] == "mnist" else "데이터 폴더 (하위 폴더 = 클래스, Enter = 유지)")
        raw = ask(desc, str(s.data_root) if s.data_root else "")
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
        if self.sess.rec is not None:
            self.sess.rec.json("environment.json", env)

    def _new_session(self) -> None:
        paths = (self.sess.data_root, self.sess.output_root)
        self.sess = Session(self.cfg, self.be, say=print, confirm=ask_yes)
        self.sess.set_paths(*paths)

    def m2(self) -> None:
        if self.sess.model is not None:
            if not ask_yes("현재 세션에 모델이 있다. 새 실행 (새 run 폴더) 으로 다시 시작할까?"):
                return
            self._new_session()
        src = ask("기존 run 의 분할 목록을 재사용할까? 재사용할 run 폴더 (Enter = 새로 검사·분할)", "")
        if src:
            rep, man = self.sess.reuse_split(clean_user_path(src), progress=print)
            print(f"  분할 재사용: 항목 {man['n_items']}, 분할 해시 {man['split_sha256'][:12]} (모델 상태는 가져오지 않았다)")
        else:
            rep, man = self.sess.scan_and_split(progress=print)
            print(f"  클래스 {rep['classes']} ({rep['n_classes']} 개, 자동 결정), 유효 이미지 {rep['n_valid_images']}")
            print(f"  손상 {len(rep['corrupt'])}, 미지원 {len(rep['unsupported'])}, 빈 클래스 {rep['empty_classes_excluded']}, "
                  f"같은 화소 중복 그룹 {len(rep['exact_duplicate_groups'])}, 클래스 간 충돌 제외 {len(rep['cross_class_duplicates_excluded'])}")
        for c, v in man["per_class"].items():
            print(f"    {c}: train {v['n_train_images']}, dev {v['n_dev_images']}, test {v['n_test_images']}")
        if man.get("source") == "mnist":
            print("  MNIST 공식 test는 재분할하지 않는다. train/dev는 공식 train에서만 나눈다.")
        dc = read_json(self.sess.rec.p("preprocessing.json"))["dc_check"]
        print(f"  전처리 DC 검사 (균일 영상): {dc['status']} (최대 잔여 {dc['max_abs_interior']:.2e})")
        print(f"  실행 폴더: {self.sess.rec.dir}")

    def m3(self) -> None:
        s = self.sess
        if s.model is not None and s.model.areas:
            print(f"  이미 모델이 있다 ({s.model.areas}). 구조 요약: {s.rec.p('structure.json')}")
            for line in pretraining_lines(s):
                print("    " + line)
            return
        area, specs, table, summ = s.plan_next()
        print(f"  {area} 구조 계획 (장치 텐서를 만들기 전):")
        print_summary_table(summ)
        if not ask_yes(f"{area} 를 생성하고 train 으로 코드북을 초기화한 뒤 에폭 0 기준 평가를 할까?", True):
            return
        out = s.build_area(area, specs, table, summ)
        b = out["baseline"]
        print(f"  코드북 초기화: train {out['init']['n_samples']} 장, {out['init']['seconds']} 초")
        for fam, v in out["init"]["families"].items():
            print(f"    {fam}: 후보 {v['n_legal_candidates']}, 대표 생성 {v['n_code_ok']}, 실패 {v['n_failed']}, fallback {v['n_support_mass_fallback']}, "
                  f"방향 중복 {v['n_duplicate_direction']}")
        print(f"  에폭 0 기준 (초기화 코드북 + train 판독기): train {pct_count(b['train_correct'], b['train_n'])}, dev {pct_count(b['dev_correct'], b['dev_n'])}")
        save_structure_figure(s, area)

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
        print(f"  활성 영역 {area}: 상태 {st['status']}, 에폭 {st['epochs_completed']}/{st['max_epochs']}, 다음 배치 {st['next_batch']}, "
              f"선택 에폭 {st.get('selected_epoch')}")
        if st["status"] in STAGE_TERMINAL:
            v = ask("이미 끝난 단계다 (선택 에폭으로 복원됨). 추가 에폭 수 (Enter = 취소, 선택 상태에서 이어 학습)", "")
            if not v:
                return
            extra = int(v)
        for line in pretraining_lines(s):
            print("    " + line)
        if st["epochs_completed"] == 0 and st["next_batch"] == 0 and not ask_yes("이 설정으로 학습을 시작할까?", True):
            return
        print("  Ctrl+C 한 번: 배치 경계에서 멈춤 / 두 번: 즉시 중단 (진행 중 배치는 적용하지 않음)")
        res = StageTrainer(s).run(extra_epochs=extra)
        print_train_result(res)
        save_training_figure(s, area)
        save_usage_figure(s, area)
        build_report(s)

    def m5(self) -> None:
        s = self.sess
        if s.model is None:
            print("  모델이 없다.")
            return
        area, specs, table, summ, info = s.add_next_area_prepare()
        print(f"  다음 영역 {area}: 현재 {info['current_area']} 상태 {info['current_status']}, best dev {fmt(info['best_dev_accuracy'])} "
              f"(선택 에폭 {info['best_epoch']})")
        if info["warning_ko"]:
            print("  [경고] " + info["warning_ko"])
        print_summary_table(summ)
        if not ask_yes(f"{info['current_area']} 를 선택 상태로 동결하고 {area} 를 추가할까?", False):
            s.rec.append_jsonl("stage_transitions.jsonl", {"utc": utc_now(), "from": info["current_area"], "to": area,
                                                          "user_choice": "declined", "warning_ko": info["warning_ko"]})
            return
        out = s.add_next_area_commit(area, specs, table, summ, info, user_choice="menu_confirmed")
        b = out["baseline"]
        print(f"  {info['current_area']} 동결 (선택 에폭 {out['transition']['selected_epoch']}), {area} 코드북 초기화 {out['init']['n_samples']} 장, "
              f"에폭 0 dev {pct_count(b['dev_correct'], b['dev_n'])}")
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
        area = ask(f"판독 영역 ({'/'.join(s.model.areas)})", s.model.active)
        res = run_inference(s, clean_user_path(p), area=area)
        if not res["readout_matches_current_codebook"]:
            print("  [주의] 판독기가 현재 코드북과 다른 스냅샷 (에폭 중간 상태) 이다. 결과 해석에 주의한다.")
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
        while True:
            print("  7-1) 사전 검사 16 항목  7-2) 역할 검사 (통제 자극)  7-3) 뉴런 조회 (3×3 + 후보·점수·전달값)  7-4) 최종 test 평가 (선택된 IT)")
            print("  7-5) 상태 요약  7-6) 전처리 DC·정보 손실  0) 돌아가기")
            c = ask("진단 선택")
            if c in ("0", ""):
                return
            if c == "1":
                r = PreChecks(s).run()
                for k, v in r["checks"].items():
                    print(f"    {k}: {v.get('status')}" + (f" — {v.get('error') or v.get('reason_ko')}" if v.get("error") or v.get("reason_ko") else ""))
                print(f"  종합: {r['status']} (실패 {r['n_failed']}, 미실행 {r['n_not_run']}, 복원 {r['restored']}) — 구현 검사이며 성능 검사가 아니다.")
            elif c == "2":
                r = RoleProbes(s).run(context="menu")
                for probe, pr in r["probes"].items():
                    for key, d in pr.items():
                        if key.endswith("/L3") or key.endswith("/codes") or key.startswith("V1/role"):
                            print(f"    {probe} {key}: {d}")
                print("    " + r["note_ko"])
            elif c == "3":
                raw = ask("neuron_id 또는 '영역/층:local' (예: V1/L2:100)")
                if ":" in raw:
                    key, loc = raw.split(":", 1)
                    nid = s.model.specs[key.strip()].start + int(loc)
                else:
                    nid = int(raw)
                use = ask_yes("train 첫 표본으로 동적 값 (입력 비율·점수·선택·전달값) 도 계산할까?", True)
                info = s.inspect_neuron(nid, s.data.items("train")[0] if use else None)
                print(NeuronInspector.format(info))
            elif c == "4":
                ph = project_test_history(s, s.data.ids_sha("test"))
                print(f"  이 실행의 test 평가 {s.counters['test_evaluations']} 회, 같은 저장 폴더의 다른 실행에서 같은 test 집합을 본 횟수 "
                      f"{ph['n_prior_evaluations']}. test 는 선택된 IT 고정 모델의 최종 평가용이며 선택에 쓰지 않는다.")
                if ask_yes("선택된 IT 로 test 평가를 할까?"):
                    r = test_evaluation(s)
                    m = r["metrics"]
                    print(f"    test {pct_count(m['correct'], m['n'])}, 분류 불능 {m['n_unclassifiable']}, 독립 첫 평가 {r['independence']['independent_first_look']}")
            elif c == "5":
                print(dumps(state_summary(s)))
            elif c == "6":
                dc = s.pre.dc_check()
                print(f"    DC 검사 {dc['status']} (내부 {dc['max_abs_interior']:.2e}, 경계 {dc['max_abs_border']:.2e}, LGN {dc['max_abs_lgn_sample']:.2e})")
                it = s.data.items("train")[0]
                im, _m = s.data.load_image(it)
                arr, _m2 = s.pre.resize_pad(im)
                il = s.pre.info_loss(arr)
                s.rec.json(f"diagnostics/preprocessing_{local_stamp()}.json", {"dc_check": dc, "info_loss_first_train_image": il, "info": s.pre.info()})
                for name, rows in il["maps"].items():
                    print(f"    {name}: " + ", ".join(f"r {r['r_from']:.0f}~{r['r_to']:.0f}: {fmt(r['rel_error'], 3)}" for r in rows))

    def m8(self) -> None:
        s = self.sess
        if s.model is None:
            print("  저장할 모델이 없다.")
            return
        s.save_checkpoint("menu_save", keep=True)
        figs = save_all_figures(s)
        s.consolidate_sample_routes()
        p = build_report(s)
        print(f"  checkpoint {s.last_checkpoint}, 그림 {len(figs)} 개, sample_routes.npz, 보고서 {p}")

    def m9(self) -> None:
        s = self.sess
        if s.rec is None:
            print("  실행 폴더가 없다.")
            return
        while True:
            print("  9-1) 구조·후보 요약  9-2) 최근 라우팅 집계 (dev)  9-3) 코드북 갱신 집계  9-4) 설계 변경 기록  0) 돌아가기")
            c = ask("선택")
            if c in ("0", ""):
                return
            if c == "1" and s.model is not None:
                for line in pretraining_lines(s):
                    print("    " + line)
            elif c == "2":
                for x in [x for x in s.rec.read_csv(ROUTING_CSV) if not x["family"].startswith("layer_")][-12:]:
                    print(f"    {x['context']} {x['area']} {x['family']} e{x['epoch_number']}: 선택 {fmt(x.get('selected_frac'))}, 선택·출력0 "
                          f"{fmt(x.get('selected_but_zero_output_frac'))}, 최대 cos {fmt(x.get('mean_max_cosine'))}, 미사용 후보 {fmt(x.get('unused_candidate_frac'))}, "
                          f"독점 {fmt(x.get('monopoly_groups_frac'))}")
            elif c == "3":
                for x in s.rec.read_csv(CODEBOOK_CSV)[-12:]:
                    print(f"    {x['context']} {x['area']} {x['family']} e{x['epoch_number']}: " + ", ".join(
                        f"{k}={fmt(v)}" for k, v in x.items() if k not in ("utc", "area", "family", "context", "epoch_number") and v not in ("", None))[:300])
            elif c == "4":
                p_ = s.rec.p("design_changes.jsonl")
                if p_.is_file():
                    for line in p_.read_text(encoding="utf-8").splitlines()[-10:]:
                        print("    " + line)
                note = ask("새 설계 변경 내용 (Enter = 기록 안 함)", "")
                if note:
                    rec = s.log_design_change(note)
                    print("  기록했다." + (" [주의] test 를 이미 본 뒤의 변경이다 — 이후 test 는 독립 평가가 아니다." if rec["after_test_seen"] else ""))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{PROGRAM}: 모든 영역 입력 비율 라우팅·국소 대표 벡터 학습 (인자 없이 실행하면 한국어 메뉴)")
    p.add_argument("--data-root", default=None, help="MNIST 캐시 폴더 또는 --dataset folder 의 클래스별 이미지 폴더")
    p.add_argument("--dataset", choices=["mnist", "folder"], default=None, help="새 실행 기본 mnist. 재개에서는 저장 데이터 종류와 일치해야 함")
    p.add_argument("--mnist-dev-size", type=int, default=None, help="공식 MNIST train 중 dev 개수 (기본 10000)")
    p.add_argument("--no-download", action="store_true", help="MNIST 캐시에 없는 파일은 자동으로 받지 않고 오류 표시")
    p.add_argument("--prepare-data", action="store_true", help="데이터 검사·분할·기록만 수행 (모델 생성/학습 없음)")
    p.add_argument("--reuse-split", default=None, help="이 run (구버전 포함) 의 split_manifest.json 만 재사용해 새 run 을 시작 (모델 상태는 가져오지 않음)")
    p.add_argument("--output-dir", default=None, help="저장 폴더 (새 run 폴더를 그 안에 만든다)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--backend", default="auto", choices=["auto", "torch", "numpy"],
                   help="auto/torch = PyTorch (없으면 오류), numpy = 같은 수식의 CPU 대체 경로 (명시할 때만)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default=None, help="설정 JSON (DEFAULT_CONFIG 위에 덮어쓴다)")
    p.add_argument("--self-test", action="store_true", help="극소 인공 이미지로 구현 자체 검사 (사용자 데이터 미사용)")
    p.add_argument("--train-stage", default=None, choices=list(AREA_ORDER), help="이 영역까지 순서대로 학습 (앞 단계는 선택 상태로 동결)")
    p.add_argument("--epochs", type=int, default=None, help="이번 실행에서 학습하는 단계의 에폭 예산 (완료 에폭 + N)")
    p.add_argument("--infer", default=None, help="추론할 이미지 파일 또는 폴더")
    p.add_argument("--resume", default=None, help="이어 갈 v5 run 폴더")
    p.add_argument("--precheck", action="store_true", help="사전 검사 16 항목")
    p.add_argument("--role-probe", action="store_true", help="역할 검사 (통제 자극)")
    p.add_argument("--test-eval", action="store_true", help="최종 test 평가 (선택된 IT 고정 모델, 한 번)")
    p.add_argument("--yes", action="store_true", help="확인 질문에 모두 예")
    p.add_argument("--menu", action="store_true", help="다른 인자가 있어도 메뉴로 시작")
    return p


def config_from_cli(args: argparse.Namespace) -> dict[str, Any]:
    cfg = Config.load(args.config, args.seed)
    if args.dataset:
        cfg["data"]["source"] = args.dataset
    if args.mnist_dev_size is not None:
        cfg["data"]["mnist_dev_size"] = args.mnist_dev_size
    if args.no_download:
        cfg["data"]["mnist_download"] = False
    return validate_config(cfg)


def run_cli(args: argparse.Namespace) -> int:
    cfg = config_from_cli(args)
    if args.prepare_data and any((args.resume, args.train_stage, args.infer, args.test_eval, args.precheck, args.role_probe)):
        raise SystemExit("--prepare-data 는 새 실행의 데이터 준비 전용이며 학습/재개/진단 명령과 함께 쓰지 않는다.")
    if args.resume and args.reuse_split:
        raise SystemExit("--resume 과 --reuse-split 은 함께 쓰지 않는다.")
    be = DeviceManager.create(args.backend, args.device, cfg)
    conf: Callable[[str], bool] = (lambda _m: True) if args.yes else ask_yes
    DeviceManager.print_environment(DeviceManager.environment(be))
    if args.resume:
        sess = Session.load(clean_user_path(args.resume), be, say=print, confirm=conf,
                            data_root=clean_user_path(args.data_root) if args.data_root else None)
        if args.seed is not None and int(args.seed) != int(sess.cfg["seed"]):
            raise CompatibilityError(f"--seed {args.seed} 가 실행의 seed {sess.cfg['seed']} 와 다르다.")
        if args.dataset and args.dataset != sess.cfg["data"]["source"]:
            raise CompatibilityError("기존 모델의 데이터 종류를 바꿔 이어 학습하지 않는다.")
    else:
        if not (args.data_root and args.output_dir):
            raise SystemExit("새 실행에는 --data-root 와 --output-dir 가 필요하다 (또는 --resume).")
        sess = Session(cfg, be, say=print, confirm=conf)
        sess.set_paths(clean_user_path(args.data_root), clean_user_path(args.output_dir))
        if args.reuse_split:
            rep, man = sess.reuse_split(clean_user_path(args.reuse_split), progress=print)
        else:
            rep, man = sess.scan_and_split(progress=print)
        print(f"  클래스 {man['classes']}, 항목 {man['n_items']}, 실행 폴더 {sess.rec.dir}")
        if args.prepare_data:
            print("  데이터 준비 완료. 모델/학습은 실행하지 않았다. 메뉴에서는 1 → 2 → 3 순서로 시작한다.")
            sess.rec.close()
            return 0
        area, specs, table, summ = sess.plan_next()
        print_summary_table(summ)
        if not conf(f"{area} 를 생성할까?"):
            return 1
        sess.build_area(area, specs, table, summ)
    code = 0
    if args.precheck:
        # 사전 검사는 학습 전에 실행한다. 실패하면 이번 실행의 학습을 시작하지 않는다.
        r = PreChecks(sess).run()
        for k, v in r["checks"].items():
            print(f"    {k}: {v.get('status')}" + (f" — {v.get('error') or v.get('reason_ko')}" if v.get("error") or v.get("reason_ko") else ""))
        if r["status"] == FAILED:
            code = 3
            if args.train_stage:
                print("  사전 검사에 FAILED 가 있어 --train-stage 학습을 시작하지 않는다 (diagnostics/prechecks_*.json 확인).")
    if args.train_stage and code == 0:
        target = args.train_stage
        if AREA_INDEX[target] < AREA_INDEX[sess.model.active]:
            raise SystemExit(f"현재 활성 영역 {sess.model.active} 보다 앞 단계 {target} 로 돌아가지 않는다.")
        while True:
            area = sess.model.active
            st = sess.stage[area]
            ended = st["status"] in STAGE_TERMINAL
            if not ended or (area == target and args.epochs):
                for line in pretraining_lines(sess):
                    print("    " + line)
                res = StageTrainer(sess).run(extra_epochs=args.epochs)
                print_train_result(res)
                if res["status"] not in STAGE_TERMINAL:
                    code = 2
                    break
            else:
                print(f"  {area} 단계는 이미 '{st['status']}' 로 끝났다 (선택 에폭 {st.get('selected_epoch')}).")
            if area == target:
                break
            nxt, specs, table, summ, info = sess.add_next_area_prepare()
            if info["warning_ko"]:
                print("  [경고] " + info["warning_ko"])
            print_summary_table(summ)
            if not conf(f"{area} 를 선택 상태로 동결하고 {nxt} 를 추가할까?"):
                break
            sess.add_next_area_commit(nxt, specs, table, summ, info, user_choice="cli_train_stage_auto_progress")
    if args.role_probe:
        r = RoleProbes(sess).run(context="cli")
        print(f"  역할 검사: {r['status']} (role_probe_metrics.csv)")
    if args.infer:
        res = run_inference(sess, clean_user_path(args.infer))
        for r in res["rows"][:30]:
            print(f"    {Path(r['image']).name}: {r.get('prediction')} {r['status']}")
        print(f"  결과 {res['out_dir']}")
    if args.test_eval and conf("최종 test 평가를 할까? (선택된 IT 고정 모델, 횟수 기록)"):
        r = test_evaluation(sess)
        print(f"  test {pct_count(r['metrics']['correct'], r['metrics']['n'])}")
    sess.save_checkpoint("cli_end", keep=True)
    save_all_figures(sess)
    sess.consolidate_sample_routes()
    print(f"  보고서: {build_report(sess)}")
    return code


# ======================================================================
# 19. 자체 검사 (--self-test): 극소 인공 입력에 대한 기능 검사 (성능 검사가 아니다)
# ======================================================================
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


SELFTEST_CONFIG: dict[str, Any] = {
    # 자체 검사용 작은 구성 (사용자 기본 설정과 별개이며 성능 검사가 아니다)
    "data": {"source": "folder", "image_size": 64},
    "logpolar": {"n_r": 12, "n_phi": 24, "r0_px": 1.5},
    "areas": {"V1": {"n_r": 6, "n_phi": 24}, "V2": {"n_r": 3, "n_phi": 12}, "V4": {"n_r": 3, "n_phi": 12}, "IT": {"region_r": 1, "region_phi": 4}},
    "learning": {"batch_size": 4, "max_epochs": 2, "patience": 3},
    "records": {"checkpoint_every_batches": 2, "keep_periodic_checkpoints": 2, "detail_samples_per_epoch": 3, "detail_samples_max_per_stage": 9,
                "diagnostic_samples": 3},
}


def run_self_test(output_dir: Path | None, be: Backend, say: Callable[[str], None] = print) -> int:
    root = ensure_writable_dir(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="cortex_v5_selftest_"))
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
    main: dict[str, Session] = {}

    def check(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            r = fn()
        except Exception as exc:                             # noqa: BLE001
            r = {"status": FAILED, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=8)}
        r["seconds"] = round(time.perf_counter() - t0, 3)
        results[name] = r
        say(f"  {name:<52} {r['status']}" + (f"  {r.get('error')}" if r.get("error") else ""))
        return r

    def new_session(cfg_: dict[str, Any] | None = None) -> Session:
        s = Session(cfg_ or cfg, be, say=quiet)
        s.set_paths(data_dir, runs)
        s.scan_and_split()
        area, specs, table, summ = s.plan_next()
        s.build_area(area, specs, table, summ)
        return s

    def codebook_shas(s: Session) -> dict[str, str]:
        return {a: s.model.routers[a].codebook_sha() for a in s.model.areas}

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
              any(u.endswith("notes.txt") for u in rep["unsupported"]) and rep["empty_classes_excluded"] == ["zz_empty"] and dup_same and per_ok
              and s.rec.p("source.sha256").is_file() and s.rec.p("environment.json").is_file())
        return {"status": PASSED if ok else FAILED, "classes": rep["classes"], "corrupt": len(rep["corrupt"]), "duplicate_group_same_split": dup_same}

    def st02() -> dict[str, Any]:
        s = main["s"]
        dc = s.pre.dc_check()
        it = s.data.items("train")[0]
        im, _m = load_pil_rgb(s.data.root / it["relpath"])
        arr, _m2 = s.pre.resize_pad(im)
        il = s.pre.info_loss(arr)
        return {"status": dc["status"], "dc_max_abs": dc["max_abs_interior"], "info_loss_maps": list(il["maps"])}

    def st03() -> dict[str, Any]:
        """V1 단독 tiny 모델의 구조 검사: 집단·후보·패킷 차원·후보 레코드·코드북 초기화·에폭 0 기준 행."""
        s = main["s"]
        area, specs, table, summ = s.plan_next()
        s.build_area(area, specs, table, summ)
        a = cfg["areas"]["V1"]
        npos = a["n_r"] * a["n_phi"]
        t = s.model.tables["V1"]
        L2 = s.model.specs["V1/L2"]
        cand = load_npz(s.rec.p("structure/candidates_V1.npz"))
        need = ("area", "group_id", "route_id", "destination_L2_id", "source_L4_id", "input_slot", "signed_base_weight", "source_position",
                "destination_position", "source_channel_metadata_code", "role_family", "valid_mask", "rejection_reason", "codebook_bucket",
                "codebook_group_slot", "codebook_candidate_slot", "descriptor_schema_id", "edge_ptr")
        fields_ok = all(k in cand for k in need)
        init = read_json(s.rec.p("codebook/init_V1.json"))
        rows = s.rec.read_csv(TRAINING_CSV)
        sj = read_json(s.rec.p("structure.json"))
        C, ok = s.model.routers["V1"].get_codebook()
        ok_ = (t.n_groups == npos and int(t.r_legal.sum()) == L2.n and int(t.g_D.min()) > 1 and fields_ok and init["split"] == "train"
               and init["labels_used"] is False and len(rows) == 1 and rows[0]["row_kind"] == "init_baseline" and rows[0]["epoch_number"] == "0"
               and "V1" in sj["areas"] and sum(int(o.sum()) for o in ok) > 0 and (t.e_w < 0).any())
        return {"status": PASSED if ok_ else FAILED, "groups": t.n_groups, "legal_candidates": int(t.r_legal.sum()), "descriptor_dim_min_max":
                [int(t.g_D.min()), int(t.g_D.max())], "candidate_record_fields_ok": fields_ok, "init_families": init["families"],
                "baseline_dev": rows[0].get("dev_accuracy") if rows else None, "buckets": len(t.buckets)}

    def st04() -> dict[str, Any]:
        s = main["s"]
        item = s.data.items("train")[0]
        rows = []
        ok = True
        for key in s.model.store.order:
            sp = s.model.specs[key]
            for j in (0, sp.n // 2, sp.n - 1):
                info = s.inspect_neuron(sp.start + j, item)
                m3 = info["matrix_3x3"]
                good = m3[2][2] == float(cfg["neuron"]["q"]) and m3[1][1] == float(cfg["neuron"]["threshold"]) and m3[2][1] is not None
                if key.endswith("/L2"):
                    g = info["routing"]["group"]
                    good &= "this_sample" in g and "candidate_scores" in g["this_sample"] and len(g["candidates"]) > 0
                ok &= bool(good)
                rows.append({"neuron_id": sp.start + j, "ok": bool(good)})
        L2 = s.model.specs["V1/L2"]
        res = s.model.forward(s.loader.batch([item], "train")[0], upto="V1", record=("V1",))
        sel = np.nonzero(be.to_np(res["V1/route"]["mask"])[0])[0]
        txt = ""
        if sel.size:
            info = s.inspect_neuron(L2.start + int(sel[0]), item)
            ts = info["routing"]["group"]["this_sample"]
            ok &= ts["has_selection"] and len(ts.get("delivered_edges", [])) > 0 and info["routing"]["this_sample_neuron"]["selected"]
            txt = NeuronInspector.format(info)
        ok &= bool(sel.size) and "실제 전달" in txt
        return {"status": PASSED if ok else FAILED, "rows": rows, "example": txt[:1500]}

    def st05() -> dict[str, Any]:
        r = PreChecks(main["s"]).run()
        st = {k: v.get("status") for k, v in r["checks"].items()}
        bad = [k for k, v in st.items() if v == FAILED or (v == NOT_RUN and not k.startswith("C14"))]
        ok = not bad and r["restored"] and r["test_access_delta"] == 0
        return {"status": PASSED if ok else FAILED, "checks": st, "errors": {k: v.get("error") for k, v in r["checks"].items() if v.get("error")},
                "c14": r["checks"].get("C14_cpu_vs_gpu_and_reference", {}).get("reference_float64_vs_backend")}

    def st06() -> dict[str, Any]:
        s = main["s"]
        h0 = codebook_shas(s)["V1"]
        q0 = s.model.store.q_numpy()
        res = StageTrainer(s).run()
        st = s.stage["V1"]
        rows = [r for r in s.rec.read_csv(TRAINING_CSV) if r["area"] == "V1"]
        rctx = {r["context"] for r in s.rec.read_csv(ROUTING_CSV)}
        cm = [r for r in s.rec.read_csv(CODEBOOK_CSV) if r["context"] == "epoch_updates"]
        changed = any(float(r["candidate_updates"] or 0) > 0 for r in cm)
        best_sha = s.best["codebook_sha256"] if s.best else None
        ok = (res["status"] in STAGE_TERMINAL and st["epochs_completed"] == 2 and len(rows) == 3 and np.array_equal(q0, s.model.store.q_numpy())
              and {"dev_eval", "dev_init_baseline", "train_epoch_learning"} <= rctx and changed and st["selected_restored"]
              and s.model.routers["V1"].codebook_sha() == best_sha and (st["selected_epoch"] == 0 or s.model.routers["V1"].codebook_sha() != h0)
              and s.rec.p("codebook/usage_V1_e001.npz").is_file() and s.rec.p("readout/V1_e002.npz").is_file())
        return {"status": PASSED if ok else FAILED, "result": {k: res.get(k) for k in ("status", "best_epoch", "selected_epoch", "best_dev_accuracy")},
                "training_rows": len(rows), "routing_contexts": sorted(rctx), "codebook_updated": changed,
                "note_ko": "코드북이 갱신되고 선택 에폭으로 복원되는지 (학습 경로 실행) 만 본다. 분류 개선은 통과 조건이 아니다."}

    def st07() -> dict[str, Any]:
        A = new_session()
        A.stage["V1"]["max_epochs"] = 1
        StageTrainer(A).run()
        B = new_session()
        B.stage["V1"]["max_epochs"] = 1
        r1 = StageTrainer(B).run(stop_after_batches=2)
        B2 = Session.load(B.rec.dir, be, say=quiet)
        r2 = StageTrainer(B2).run()
        same_cb = codebook_shas(A) == codebook_shas(B2)
        same_ro = A.readouts["V1"].sha() == B2.readouts["V1"].sha()
        drop = ("utc", "model_version", "readout_version")
        ra = [r for r in A.rec.read_csv(ROUTING_CSV) if r["context"] == "train_epoch_learning"]
        rb = [r for r in B2.rec.read_csv(ROUTING_CSV) if r["context"] == "train_epoch_learning"]
        same_rows = len(ra) == len(rb) > 0 and all({k: v for k, v in x.items() if k not in drop} == {k: v for k, v in y.items() if k not in drop}
                                                   for x, y in zip(ra, rb))
        ca = [r for r in A.rec.read_csv(CODEBOOK_CSV) if r["context"] == "epoch_updates"]
        cb = [r for r in B2.rec.read_csv(CODEBOOK_CSV) if r["context"] == "epoch_updates"]
        same_cbm = len(ca) == len(cb) > 0 and all({k: v for k, v in x.items() if k not in ("utc", "codebook_version")} ==
                                                  {k: v for k, v in y.items() if k not in ("utc", "codebook_version")} for x, y in zip(ca, cb))
        ok = same_cb and same_ro and r1["status"] == "interrupted_test" and r2["status"] in STAGE_TERMINAL and same_rows and same_cbm
        return {"status": PASSED if ok else FAILED, "codebooks_identical": same_cb, "readouts_identical": same_ro, "first": r1["status"],
                "after_resume": r2["status"], "routing_rows_identical": same_rows, "codebook_rows_identical": same_cbm}

    def st08() -> dict[str, Any]:
        C_ = new_session()
        C_.stage["V1"]["max_epochs"] = 1
        h0 = codebook_shas(C_)
        C_.fault = "interrupt_after_apply"
        r1 = StageTrainer(C_).run()
        rolled = codebook_shas(C_) == h0 and C_.stage["V1"]["next_batch"] == 0
        C2 = Session.load(C_.rec.dir, be, say=quiet)
        r2 = StageTrainer(C2).run()
        C2.flush_details()
        parts = sorted(C2.rec.p("routes").glob("sample_routes_V1_e001_part*.npz"))
        ids = [x for p in parts for x in load_npz(p)["sample_id"].tolist()]
        no_dup = len(ids) == len(set(ids)) and len(ids) > 0
        ok = r1["status"] == "interrupted_manual" and rolled and r2["status"] in STAGE_TERMINAL and no_dup
        return {"status": PASSED if ok else FAILED, "first": r1["status"], "codebook_rolled_back_and_batch_not_committed": rolled,
                "after_resume": r2["status"], "detail_samples": len(ids), "no_duplicate_detail_samples": no_dup}

    def st09() -> dict[str, Any]:
        """V1 → V2 → V4 → IT tiny 모델: 단계 전환·동결·구조 검사·사전 검사 (IT 활성)."""
        s = main["s"]
        out = {}
        ok = True
        for _ in range(3):
            nxt, specs, table, summ, info = s.add_next_area_prepare()
            s.add_next_area_commit(nxt, specs, table, summ, info, user_choice="self_test")
            s.stage[nxt]["max_epochs"] = 1
            res = StageTrainer(s).run()
            fc = s.frozen_check()
            ok &= all(all(v.values()) for v in fc.values()) and res["status"] in STAGE_TERMINAL
            t = s.model.tables[nxt]
            out[nxt] = {"status": res["status"], "groups": t.n_groups, "legal": int(t.r_legal.sum()), "rejected": int((~t.r_legal).sum()),
                        "families": t.family_names, "frozen_ok": all(all(v.values()) for v in fc.values())}
        t_it = s.model.tables["IT"]
        it_ok = t_it.n_groups > 1 and len({int(f) for f in t_it.g_family}) >= 3
        r = PreChecks(s).run()
        st = {k: v.get("status") for k, v in r["checks"].items()}
        bad = [k for k, v in st.items() if v == FAILED or (v == NOT_RUN and not k.startswith("C14"))]
        sj = read_json(s.rec.p("structure.json"))
        ok &= not bad and it_ok and set(sj["areas"]) == set(AREA_ORDER)
        out["prechecks_at_IT"] = st
        out["errors"] = {k: v.get("error") for k, v in r["checks"].items() if v.get("error")}
        out["IT_not_global_top1"] = it_ok
        return {"status": PASSED if ok else FAILED, **out}

    def st10() -> dict[str, Any]:
        s = main["s"]
        blank = st_dir / "blank.png"
        Image.fromarray(np.full((40, 40, 3), 128, np.uint8)).save(blank)
        r1 = run_inference(s, data_dir / "circle")
        r2 = run_inference(s, blank)
        ok = len(r1["rows"]) == len(list_images(data_dir / "circle", cfg["data"]["extensions"])) and r2["rows"][0]["status"] == "unclassifiable"
        return {"status": PASSED if ok else FAILED, "n_rows": len(r1["rows"]), "blank": r2["rows"][0]["status"],
                "predictions": [x.get("prediction") for x in r1["rows"]]}

    def st11() -> dict[str, Any]:
        s = main["s"]
        r = RoleProbes(s).run(context="selftest")
        rows = [x for x in s.rec.read_csv(ROLE_CSV) if x["context"] == "selftest"]
        metrics = {x["metric"] for x in rows}
        need = {"nn_orientation_decoding_across_offset", "distinct_patterns", "pairs_different_L2_code_but_same_L3",
                "selected_candidate_orientation_within_15deg_frac"}
        ok = r["status"] == MEASURED and need <= metrics
        return {"status": MEASURED if ok else FAILED, "n_rows": len(rows), "metrics": sorted(metrics),
                "note_ko": "극소 합성 자료의 측정값이다 (성능 검사가 아니다)."}

    def st12() -> dict[str, Any]:
        s = main["s"]
        early = new_session()
        refused = False
        try:
            test_evaluation(early)
        except RuntimeError:
            refused = True
        n0 = s.counters["test_evaluations"]
        test_evaluation(s)
        s.save_checkpoint("selftest_final", keep=True)
        figs = save_all_figures(s)
        s.consolidate_sample_routes()
        p = build_report(s)
        txt = p.read_text(encoding="utf-8")
        sect = txt.split("## 최종 test 평가", 1)[-1][:400]
        need = ("config.json", "environment.json", "source.sha256", "split_manifest.json", "structure.json", TRAINING_CSV, ROUTING_CSV, CODEBOOK_CSV,
                ROLE_CSV, "sample_routes.npz", "REPORT.md", "assumptions.md", "diagnostics.json", "execution.log")
        have = {f: s.rec.p(f).is_file() for f in need}
        sr = load_npz(s.rec.p("sample_routes.npz"))
        ok = (refused and s.counters["test_evaluations"] == n0 + 1 and all(have.values()) and s.rec.p("checkpoints").is_dir() and "#1" in sect
              and len(sr["parts"]) > 0 and "## 구조와 후보 그래프" in txt and "국소 경쟁학습" in txt)
        return {"status": PASSED if ok else FAILED, "test_before_IT_refused": refused, "files": have, "figures": len(figs),
                "matplotlib": _MPL_STATE["plt"] is not None, "sample_route_parts": int(len(sr["parts"]))}

    def st13() -> dict[str, Any]:
        s = main["s"]
        L = Session.load(s.rec.dir, be, say=quiet)
        ok = (codebook_shas(L) == codebook_shas(s) and L.model.active == "IT" and all(L.readouts[a].sha() == s.readouts[a].sha() for a in s.readouts)
              and L.frozen.keys() == s.frozen.keys())
        return {"status": PASSED if ok else FAILED, "areas": L.model.areas, "active": L.model.active}

    def st14() -> dict[str, Any]:
        cases = {'  "C:\\Users\\홍 길동\\out"  ': "C:\\Users\\홍 길동\\out", "'D:\\a b'": "D:\\a b", "\u202aE:\\x\\y": "E:\\x\\y"}
        ok = all(str(clean_user_path(k)) == str(Path(v)) for k, v in cases.items())
        a = unique_run_dir(st_dir / "u")
        b = unique_run_dir(st_dir / "u")
        ok &= a != b and a.is_dir() and b.is_dir()
        return {"status": PASSED if ok else FAILED, "cases": {k: str(clean_user_path(k)) for k in cases}}

    def st15() -> dict[str, Any]:
        """구버전 run 거부 (이어 학습 아님) 와 분할 목록 재사용."""
        s = main["s"]
        old = st_dir / "fake_v4_run"
        (old / "checkpoints").mkdir(parents=True, exist_ok=True)
        write_json(old / "config.json", {"program": "cortex_local_learning_v4_v1_destination_routing", "program_version": "4.0.0",
                                         "config": {"routing": {"mode": "v1_destination_guided"}}})
        shutil.copyfile(s.rec.p("split_manifest.json"), old / "split_manifest.json")
        write_json(old / "run_info.json", {"data_root": str(data_dir), "output_root": str(runs)})
        write_json(old / "checkpoints" / "latest.json", {"name": "ckpt_000001_x"})
        rejected = False
        msg = ""
        try:
            Session.load(old, be, say=quiet)
        except CompatibilityError as exc:
            rejected, msg = True, str(exc)
        N = Session(cfg, be, say=quiet)
        N.set_paths(data_dir, runs)
        rep, man = N.reuse_split(old)
        same = man["split_sha256"] == s.data.manifest["split_sha256"]
        area, specs, table, summ = N.plan_next()
        N.build_area(area, specs, table, summ)
        info = read_json(N.rec.p("run_info.json"))
        ok = rejected and same and "reused_split_from" in info and N.model.active == "V1"
        return {"status": PASSED if ok else FAILED, "old_run_rejected": rejected, "message": msg[:200], "split_identical": same}

    def st16() -> dict[str, Any]:
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
        same = codebook_shas(A) == codebook_shas(B)
        ok = r["status"] in STAGE_TERMINAL and len(ev) == 1 and g1 < g0 and same
        return {"status": PASSED if ok else FAILED, "oom_events": len(ev), "max_gather_before": g0, "max_gather_after_retry": g1,
                "codebooks_identical_to_run_without_fault": same, "note_ko": "주입한 메모리 부족 1 회 → 같은 배치를 줄인 조각으로 다시 계산 (후보 삭제 없음)."}

    def st17() -> dict[str, Any]:
        C_ = new_session()
        h0 = codebook_shas(C_)
        C_.fault = "nonfinite_once"
        raised = False
        try:
            StageTrainer(C_).run()
        except NumericalError:
            raised = True
        ok = raised and C_.stage["V1"]["status"] == "failed_nonfinite" and codebook_shas(C_) == h0 and C_.stage["V1"]["next_batch"] == 0
        return {"status": PASSED if ok else FAILED, "numerical_error_raised": raised, "status_after": C_.stage["V1"]["status"],
                "codebook_unchanged": codebook_shas(C_) == h0}

    say(f"[자체 검사] {PROGRAM} v{VERSION}, 뒷단 {be.kind}/{be.device}, 폴더 {st_dir}")
    order = (("ST01_data_scan_split", st01), ("ST02_preprocessing_dc_infoloss", st02), ("ST03_V1_only_structure_init_baseline", st03),
             ("ST04_inspect_neuron_3x3_routes", st04), ("ST05_prechecks_16_V1_only", st05), ("ST06_train_V1_select_restore", st06),
             ("ST07_resume_equivalence_mid_epoch", st07), ("ST08_interrupt_rollback", st08), ("ST09_stages_V2_V4_IT_prechecks", st09),
             ("ST10_inference", st10), ("ST11_role_probes", st11), ("ST12_test_eval_records_report", st12), ("ST13_reload_final", st13),
             ("ST14_path_handling", st14), ("ST15_old_run_rejected_split_reuse", st15), ("ST16_oom_retry_same_result", st16),
             ("ST17_nonfinite_input_fails_batch", st17))
    for name, fn in order:
        if name[:4] in ("ST02", "ST03", "ST04", "ST05", "ST06", "ST09", "ST10", "ST11", "ST12", "ST13", "ST15") and "s" not in main:
            results[name] = {"status": NOT_RUN, "reason_ko": "ST01 실패로 세션 없음"}
            continue
        check(name, fn)
    n_fail = sum(1 for v in results.values() if v.get("status") == FAILED)
    summary = {"utc": utc_now(), "program_version": VERSION, "backend": be.describe(), "seconds": round(time.perf_counter() - t_start, 2),
               "n_failed": n_fail, "results": results, "config": cfg,
               "note_ko": "극소 인공 입력에 대한 구현 기능 검사다. 학습 성능·분류 개선은 통과 조건이 아니며 측정하지 않는다."}
    write_json(st_dir / "selftest_results.json", summary)
    say(f"[자체 검사] 실패 {n_fail} / {len(results)} — 결과 {st_dir / 'selftest_results.json'}")
    if main.get("s") is not None and main["s"].rec is not None:
        main["s"].rec.close()
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
    actions = any([args.prepare_data, args.train_stage, args.infer, args.precheck, args.test_eval, args.role_probe, args.reuse_split])
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
