#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고정 V1·V2 + V4·IT 흔적(TRACE) 학습 — cortex_v12_frozen_v1v2_trace_v4it.py (버전 12.0.0)

[이 프로그램이 하는 한 가지 학습]
  MNIST 분할 → V1·V2 준비 학습 (또는 v11 공통 초기 snapshot) → V1·V2 동결 (q·임계값·연결·전처리·L6 설정 모두)
  → 이미지마다 V2 L3 최종 출력 (L6 포함) 을 디스크 캐시 (float32 .npy, 현재 배치만 메모리로 읽음)
  → 캐시에서 V4 → IT → 10 클래스 출력층 (ClassHead) 순방향
  → 출력층 → IT → V4 까지만 TRACE + CE 교정 (V4 L4 에서 V2→V4 gain 갱신안을 만든 뒤 역방향을 끝낸다)
  → V4·IT 연결 gain 과 출력층 W·b 를 같은 순방향 상태에서 계산해 배치 끝에 한 번에 적용
  → epoch 마다 dev 로 최적 상태 선택 (정확도 → CE → 이른 epoch, epoch 0 포함) → 학습이 끝난 뒤 선택 상태로 test 한 번.
  수식은 참고 v11 (cortex_local_learning_v11_f4_long_scope_compare.py 11.0.0) 의 F4_TRACE_CE · v4_it 와 같다.
  v11 파일은 실행에 필요하지 않다 (import 하지 않는다).

[설치] — 자동 설치하지 않는다
  필수: Python 3.10 이상, NumPy, Pillow           pip install numpy pillow
  선택 (CUDA GPU): PyTorch. https://pytorch.org 의 설치 안내에서 자기 CUDA 버전에 맞는 명령을 고른다.
        예) pip install torch --index-url https://download.pytorch.org/whl/cu121
  PyTorch 가 없거나 '장치: CPU' 를 고르면 NumPy CPU 로 계산한다. 출력층 (head) 은 항상 NumPy float64 이다.

[PyCharm 에서 실행]
  1) 이 파일을 열고 Run (Shift+F10). 실행 창에 화면 하나가 나온다 (하위 메뉴 없음).
  2) MNIST 경로: 공식 gz 4 개 (train/t10k images·labels) 가 있는 폴더 (폴더 바로 아래, raw/ 또는 MNIST/raw/).
     없으면 내려받을지 묻는다 (공식 MD5 확인). 합성 데이터로 몰래 바꾸지 않는다.
  3) 결과 폴더 또는 재개할 폴더: 새 실행이면 그 안에 run_<시각> 폴더를 만든다.
     이 프로그램이 만든 run 폴더를 주면 저장된 위치를 보여 주고 이어 간다.
  4) 공통 초기 snapshot 폴더: 비우면 V1·V2 준비 학습을 이 프로그램이 한다 (V1 20 · V2 20 epoch 한도, patience 5,
     train 으로 학습·dev 로 종료 판단, v11 과 같은 종료 상태 정책). v11 장기 비교 run 폴더 (manifest.json 이 있는
     v11long_… 폴더) 를 주면 그 run 의 공통 초기 상태 (init_snapshot/seed_<seed>_CORTEX_FIXED_init.npz: q·threshold·W0·b0)
     만 manifest·config·분할·구조·파일 해시로 검증해서 쓴다. 학습이 끝난 F4·all_regions 결과는 읽지 않는다.
  5) 장치 (자동/CPU/CUDA) · seed · 주 학습 epoch · 배치 크기를 확인하고 Enter.
  명령 줄도 같다: python cortex_v12_frozen_v1v2_trace_v4it.py --mnist <폴더> --out <폴더> [--snapshot <v11 run 폴더>]
                  [--device auto|cpu|cuda] [--seed 20251011] [--epochs 100] [--batch 16] [--yes]

[중단과 재개]
  Ctrl+C 한 번: 진행 중 배치 (또는 캐시 조각) 를 마친 경계에서 저장하고 멈춘다. 두 번: 즉시 멈춘다
  (마지막 저장 지점은 그대로 남는다 — 임시 파일에 쓴 뒤 원자적으로 바꾼다).
  다시 Run 하고 그 run 폴더를 '결과 폴더 또는 재개할 폴더' 에 주면 phase (prepare_v1 / prepare_v2 / cache / train /
  evaluate) 와 epoch·배치 위치를 보여 주고 이어 간다. 같은 소스·설정이면 재개 결과는 끊기지 않은 실행과 같다.
  seed·설정·데이터·소스·장치가 저장 기록과 다르면 이어 붙이지 않는다 (새 run 으로 시작한다).
  PyCharm 의 Stop 버튼은 프로그램을 바로 끝낼 수 있다. 그래도 마지막 저장 지점에서 이어 갈 수 있다.

[저장 파일 (run_<시각>/)]
  config.json · frozen_stem.npz (동결 V1·V2 q·임계값, V4·IT 초기 q, head 초기값) · split_ids.npz (분할 표본 ID·라벨 순서)
  cache_manifest.json · cache_train.npy · cache_dev.npy · cache_test.npy (test 는 학습·선택이 끝난 뒤 만든다)
  last_state.json · last_params.npz (이어하기 기준) · best_state.json · best_params.npz (dev 최적) · epochs.csv · final_metrics.json
  기본 크기 float32 캐시: train 약 1.3824 GB, dev·test 각 약 276.48 MB (train+dev 약 1.66 GB, test 포함 약 1.94 GB).
  새 준비 경로는 준비 중에만 임시 파일 (prep_lgn_*.npy 약 0.98 GB, prep_v1l3_*.npy 약 1.47 GB) 을 쓰고 캐시를 만든 뒤 지운다.

[작성 환경에서 실제로 확인한 것] Linux · Python 3.13 (3.11 은 문법 검사만) · NumPy 2.5 · Pillow 12 · CPU 4 개 · PyTorch 없음
  - 공식 MNIST 가 없어 MNIST 형식 (gzip IDX, 60,000/10,000 장) 의 합성 숫자로 돌렸다 (검사 스크립트에서 MD5 표만 바꿈).
  - v11 11.0.0 과 NumPy CPU 비트 단위 일치: 구조 해시 4 영역 · 뉴런 56,948 · IT L3 146 · V2 L3 6,912 · 학습 연결 31,142, LGN 전처리,
    한 배치 갱신안 (V4/IT/head 출력·CE·ΔW·Δb·여섯 층 gain, 6 배치 연속), 짧은 전체 실행 (train 96·dev 48·test 48, 준비 V1·V2 각 3 epoch
    한도·patience 1, 주 학습 3 epoch → test) 의 준비 상태·epoch 별 dev·최종 W·b·gain·best·test, v11 공통 초기 snapshot 가져오기,
    캐시 = 영상에서 직접 계산한 V2 L3 (섞어 읽어도 ID·라벨 일치).
  - 중단·재개 (준비·캐시·학습·dev 평가·test 캐시 중 주입 7 지점, 실제 Ctrl+C 한 번·두 번) 결과 = 끊김 없는 실행, 손상·변조 파일 거부,
    구버전 없는 한글·공백·따옴표 폴더에서 단독 실행, 실행 화면 (표준 입력 모의).
  - 같은 배치 반복 시간 (기본 구조, 배치 16): v11 F4 v4_it 학습 8.71 초 → 이 프로그램 0.0132 초 (캐시 행에서 시작),
    평가 순방향 1.107 초 → 0.0034 초. 캐시 생성 비용: V1·V2 순방향 약 1.1 초/16 장, LGN 전처리 약 4.9 ms/장.
[확인하지 못한 것] 실제 MNIST 전체 실행 (주 학습 100 epoch 포함) 과 그 정확도·시간, CUDA/PyTorch 경로 (CPU 와 비트 단위로 같다고 보장하지 않는다),
  Windows·PyCharm 의 실제 화면·Stop 버튼, 공식 MNIST 내려받기, Python 3.10 실행.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import gc
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import signal
import struct
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

try:
    import numpy as np
except ImportError as _exc:                     # main() 에서 설치 방법을 안내한다 (import 만으로는 아무것도 하지 않는다)
    np = None  # type: ignore[assignment]
    _NUMPY_ERROR = str(_exc)
else:
    _NUMPY_ERROR = ""
try:
    from PIL import Image
except ImportError as _exc:
    Image = None  # type: ignore[assignment]
    _PIL_ERROR = str(_exc)
else:
    _PIL_ERROR = ""

torch: Any = None                                # PyTorch 는 CUDA 를 고를 때만 불러온다

VERSION = "12.0.0"
PROGRAM = "cortex_v12_frozen_v1v2_trace_v4it"

# ======================================================================
# 설정 — 바꿀 값은 여기서만 바꾼다 (실행 화면의 입력값이 seed·epochs·batch_size 를 덮는다)
# ======================================================================
SETTINGS: dict[str, Any] = {
    "seed": 20251011,
    "epochs": 100,                    # 주 학습 epoch (고정 예산. 조기 종료·학습률 스케줄 없음)
    "batch_size": 16,                 # 주 학습과 준비 학습의 배치 크기 (v11 과 같다)
    "eta_head": 0.01,                 # ΔW = η_head·mean(d_z ⊗ h), Δb = η_head·mean(d_z)
    "eta_edge": 0.001,                # Δγ = η_edge·mean(ℓ·λ·w0·x)
    "budget": 1.0,                    # 고정 총량 B (a = B·p)
    "gain_min": 0.05, "gain_max": 4.0,
    "prep": {                         # V1·V2 준비 학습 (snapshot 을 주지 않은 경우만). v11 기본값과 같다
        "epochs": {"V1": 20, "V2": 20}, "patience": 5, "min_delta": 0.0,
        "eta": 0.01, "rho": 0.05, "alpha2": 0.5, "alpha3": 0.5, "e0": 1.0, "eps_norm": 1e-12,
    },
    "data": {
        "mnist_dev_size": 10000,      # 공식 train 60,000 → train 50,000 / dev 10,000 (seed 별 층화 규칙, v11 과 같음)
        "limits": {"train": 0, "dev": 0, "test": 0},   # 0 = 전체. 작은 동작 확인용 한도 (v11 과 같은 층화 추출)
        "download": True,             # 공식 파일이 없을 때 내려받기를 물을지
    },
    "compute": {
        "memory_mb": 1024,            # 역방향 조각 크기 상한 (v11 기본 1024 와 같아야 합산 순서가 같다)
        "max_gather_elements": 1 << 25,
        "cache_batch_size": 64,       # 캐시·준비 입력 파일을 만들 때의 묶음 (표본별 값은 묶음과 무관하다; 1 표본 묶음은 만들지 않는다)
    },
    "save": {"every_batches": 200, "min_seconds": 60.0, "progress_every_batches": 500},
}

# 모델 구조·전처리 (v11 CORTEX_FIXED 기본값). 바꾸면 v11 과 다른 모델이 되고, 공통 초기 snapshot 은 같은 값일 때만 받는다.
MODEL: dict[str, Any] = {
    "image_size": 128, "transparent_background": 1.0,
    "retina": {"sigma_center_px": 1.0, "sigma_surround_px": 3.0, "truncate_sigma": 3.0, "gain": 1.0},
    "logpolar": {"n_r": 32, "n_phi": 64, "r0_px": 2.0, "rmax_margin_px": 2.0, "phi_offset_bins": 0.5, "blur_beta": 0.5,
                 "blur_levels_px": [0.0, 1.0, 2.0, 4.0, 8.0]},
    "neuron": {"threshold": 0.0, "q_init": 1.0, "q_min": 0.0, "q_max": 4.0},
    "areas": {
        "V1": {"n_r": 16, "n_phi": 32, "orientations_deg": [15.0 * k for k in range(12)], "phases_rad": [0.0, -math.pi / 2.0],
               "polarities": [1, -1], "color_channels": ["lum"], "gabor_sigma_factor": 1.0, "gabor_sigma_min_px": 1.5,
               "gabor_wavelength_factor": 2.0, "rf_radius_sigma": 2.0},
        "V2": {"n_r": 8, "n_phi": 16, "offset_factor": 1.0, "endstop_inhibition": 1.0, "cross_diffs_deg": [90.0, 60.0]},
        "V4": {"n_r": 4, "n_phi": 8, "offset_factor": 1.0, "curvature_deg": [30.0], "l3_grid_radius": 1},
        "IT": {"region_r": 2, "region_phi": 4},
    },
    "l6": {"enabled": True, "layers": ["L2", "L3"], "epsilon": 1e-6, "kappa": 0.25, "g_min": 0.5, "g_max": 2.0,
           "neighbor_grid_radius": 1, "include_self": False},
    "v4_fixed": {"score_epsilon": 1e-9},
    "curve_path_points": 9,
}

AREAS = ("V1", "V2", "V4", "IT")
AREA_INDEX = {a: i + 1 for i, a in enumerate(AREAS)}
LAYERS = ("L4", "L2", "L3")
STEM_AREAS = ("V1", "V2")
LEARN_KEYS = ("V4/L4", "V4/L2", "V4/L3", "IT/L4", "IT/L2", "IT/L3")
N_CLASSES = 10
PHASES = ("prepare_v1", "prepare_v2", "cache", "train", "evaluate", "done")
STAGE_TERMINAL = ("success_criterion_met", "plateau", "budget_exhausted")
FIXED_MODE = "v4_fixed_geometry"
STATE_FORMAT = "cortex_v12/last_state/1"
RUN_FORMAT = "cortex_v12/run_config/1"
STEM_FORMAT = "cortex_v12/frozen_stem/1"
CACHE_FORMAT = "cortex_v12/feature_cache/1"
# 공통 초기 snapshot 을 받을 수 있는 원본: v11 장기 비교 run (선언 소스 SHA256 까지 같아야 한다)
V11_PROGRAM, V11_VERSION = "cortex_local_learning_v11_f4_long_scope_compare", "11.0.0"
V11_SOURCE_SHA256 = "d314e5958f65cd29ddf11a6402eedf571fc96b92403163a2520f22ff361c1307"
V11_MANIFEST_FORMAT = "cortex_local_learning_v11/long_manifest/1"
V11_NETWORK = "CORTEX_FIXED"
DEFAULT_NUMERIC_NOTE = "NumPy CPU 경로가 기준이다 (v11 NumPy 경로와 비트 단위 비교). CUDA 는 부동소수 축약 순서가 달라 같은 값을 보장하지 않는다."


# ======================================================================
# 공통 도구 (해시는 v11 과 같은 방식이라 기록을 서로 대조할 수 있다)
# ======================================================================
class V12Error(RuntimeError):
    """사용자에게 그대로 보여 주는 오류 (원인과 할 일을 담는다)."""


class DataError(V12Error):
    pass


class IntegrityError(V12Error):
    pass


class NumericalError(V12Error):
    pass


class Interrupted(Exception):
    """Ctrl+C: 안전한 경계에서 저장한 뒤 멈춤."""


def say(msg: str = "") -> None:
    print(msg, flush=True)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
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
    if isinstance(o, dict):
        return {str(k): json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_ready(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if np is not None and isinstance(o, np.ndarray):
        return json_ready(o.tolist())
    if isinstance(o, bool) or (np is not None and isinstance(o, np.bool_)):
        return bool(o)
    if np is not None and isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float) or (np is not None and isinstance(o, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def dumps(o: Any, indent: int | None = 2) -> str:
    return json.dumps(json_ready(o), ensure_ascii=False, indent=indent)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """임시 파일에 쓰고 fsync 한 뒤 os.replace (Windows 의 일시적 파일 잠금은 몇 번 다시 시도한다)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        with contextlib.suppress(OSError):
            os.chmod(tmp, 0o644)                                  # mkstemp 의 0600 대신 보통 파일 권한
        for k in range(8):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if k == 7:
                    raise
                time.sleep(0.25 * (k + 1))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(Path(path), dumps(obj).encode("utf-8"))


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_npz(path: Path, arrays: dict[str, Any], meta: dict[str, Any] | None = None) -> None:
    """배열 + (선택) 상태 JSON 을 한 파일에 원자적으로 저장한다 (배열과 상태가 서로 다른 시점의 것이 될 수 없다)."""
    buf = io.BytesIO()
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    if meta is not None:
        payload["__meta__"] = np.frombuffer(dumps(meta, indent=None).encode("utf-8"), np.uint8)
    np.savez_compressed(buf, **payload)
    atomic_write_bytes(Path(path), buf.getvalue())


def load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    try:
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except Exception as exc:                                    # noqa: BLE001 — 손상된 파일을 정상 상태로 읽지 않는다
        raise IntegrityError(f"{path} 를 읽을 수 없다 (손상되었거나 다른 형식): {type(exc).__name__}: {exc}") from exc
    meta = None
    if "__meta__" in arrays:
        meta = json.loads(bytes(arrays.pop("__meta__")).decode("utf-8"))
    return arrays, meta


def clean_user_path(raw: Any) -> Path:
    """탐색기·PyCharm 에서 복사한 경로: 앞뒤 공백·따옴표·보이지 않는 방향 문자를 지우고 ~·환경변수를 푼다 (한글·공백 경로 그대로)."""
    s = str(raw).strip().strip("‪‫‬‭‮﻿").strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    if not s:
        raise V12Error("빈 경로다.")
    return Path(os.path.expandvars(os.path.expanduser(s)))


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    probe = p / f".write_probe_{os.getpid()}_{time.time_ns()}"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        raise V12Error(f"폴더에 쓸 수 없다: {p} ({exc})") from exc
    return p.resolve()


def program_sha() -> str:
    try:
        return sha256_file(Path(__file__))
    except OSError:
        return "unknown"


def fmt_pct(x: Any) -> str:
    return "-" if x is None else f"{100.0 * float(x):.2f}%"


def fmt_s(sec: float) -> str:
    sec = float(sec)
    if sec < 120:
        return f"{sec:.1f}초"
    h, rem = divmod(int(round(sec)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}시간 {m}분 {s}초" if h else f"{m}분 {s}초"


def fmt_bytes(n: float) -> str:
    n = float(n)
    return f"{n / 1e9:.4f} GB" if n >= 1e9 else f"{n / 1e6:.2f} MB"


class StopFlag:
    """Ctrl+C 첫 번째: 진행 중 배치를 마친 경계에서 저장하고 멈추도록 표시만 한다. 두 번째: 즉시 KeyboardInterrupt.
    critical() 안 (갱신 적용·저장 도중) 에서는 즉시 중단을 미뤘다가 끝에서 일으킨다 (반쯤 적용된 상태를 만들지 않는다)."""

    def __init__(self) -> None:
        self.requested = False
        self._old: Any = None
        self._in_critical = 0
        self._pending = False
        self._installed = False

    def __enter__(self) -> "StopFlag":
        if threading.current_thread() is threading.main_thread():
            with contextlib.suppress(ValueError, OSError):
                self._old = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, self._handler)
                self._installed = True
        return self

    def _handler(self, signum: int, frame: Any) -> None:
        if self.requested:
            if self._in_critical:
                self._pending = True
                return
            raise KeyboardInterrupt
        self.requested = True
        say("\n[중단 요청] 진행 중 배치를 마친 경계에서 저장하고 멈춘다. 한 번 더 누르면 즉시 멈춘다 (마지막 저장 지점은 그대로).")

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
# 계산 뒷단: NumPy (CPU, 기준) 또는 PyTorch (CUDA). gather·원소별 곱·마지막 축 합·clip·where 만 쓴다 (atomic 누적 없음)
# ======================================================================
def load_torch() -> tuple[Any, str]:
    global torch
    if torch is None:
        try:
            import torch as _torch                         # noqa: PLC0415 — CUDA 를 고를 때만 불러온다
        except Exception as exc:                           # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
        torch = _torch
    return torch, ""


class Backend:
    def __init__(self, kind: str, device: str = "cpu", *, max_gather: int = 1 << 25):
        if kind not in ("numpy", "torch"):
            raise V12Error(f"알 수 없는 계산 뒷단: {kind}")
        self.kind, self.max_gather = kind, int(max_gather)
        if kind == "torch":
            t, err = load_torch()
            if t is None:
                raise V12Error(f"PyTorch 를 불러올 수 없다 ({err}). CUDA 를 쓰려면 PyTorch 를 설치한다 (파일 머리말 참고).")
            t.set_grad_enabled(False)                      # 표준 autograd 를 쓰지 않는다 (수동 교정만)
            self.dev = t.device(device)
            if self.dev.type == "cuda" and not t.cuda.is_available():
                raise V12Error("CUDA 장치를 쓸 수 없다 (torch.cuda.is_available() = False). 장치를 CPU 로 고르거나 CUDA 용 PyTorch 를 설치한다.")
            t.backends.cuda.matmul.allow_tf32 = False
            t.backends.cudnn.allow_tf32 = False
            t.use_deterministic_algorithms(True, warn_only=True)
            self.device = str(self.dev)
        else:
            self.dev, self.device = None, "cpu"

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
        return np.asarray(x) if self.kind == "numpy" else x.detach().to("cpu").numpy()

    def zeros(self, shape: Sequence[int]) -> Any:
        shape = tuple(int(s) for s in shape)
        return np.zeros(shape, np.float32) if self.kind == "numpy" else torch.zeros(shape, dtype=torch.float32, device=self.dev)

    def ones_like(self, x: Any) -> Any:
        return np.ones_like(x) if self.kind == "numpy" else torch.ones_like(x)

    def zeros_like(self, x: Any) -> Any:
        return np.zeros_like(x) if self.kind == "numpy" else torch.zeros_like(x)

    def cat1(self, xs: Sequence[Any]) -> Any:
        return np.concatenate(list(xs), axis=1) if self.kind == "numpy" else torch.cat(list(xs), dim=1)

    def cat0(self, xs: Sequence[Any]) -> Any:
        return np.concatenate(list(xs), axis=0) if self.kind == "numpy" else torch.cat(list(xs), dim=0)

    def pad_zero_col(self, x: Any) -> Any:
        return self.cat1([x, self.zeros((int(x.shape[0]), 1))])

    def take_cols(self, x: Any, idx: Any) -> Any:
        return x[:, idx]

    def ell(self, xe: Any, idx: Any, w: Any) -> Any:
        """xe (B, n_src+1) (마지막 열 = 0 채움), idx·w (r, K) → (B, r) = Σ_k xe[:, idx[r,k]]·w[r,k]. 행 조각으로만 나눈다 (값 같음)."""
        B, r, K = int(xe.shape[0]), int(idx.shape[0]), int(idx.shape[1])
        rows = max(1, self.max_gather // max(1, B * K))
        if rows >= r:
            return (xe[:, idx] * w).sum(-1)
        return self.cat1([(xe[:, idx[a:a + rows]] * w[a:a + rows]).sum(-1) for a in range(0, r, rows)])

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

    def as_f32(self, x: Any) -> Any:
        return np.asarray(x, np.float32) if self.kind == "numpy" else x.to(torch.float32)

    def norm_last(self, x: Any) -> Any:
        return np.sqrt((x * x).sum(-1)) if self.kind == "numpy" else torch.sqrt((x * x).sum(-1))

    def all_finite(self, x: Any) -> bool:
        return bool(np.isfinite(x).all()) if self.kind == "numpy" else bool(torch.isfinite(x).all().item())

    def all_finite_many(self, xs: Sequence[Any]) -> bool:
        """여러 배열의 유한성을 장치 왕복 한 번으로 확인한다."""
        if not xs:
            return True
        if self.kind == "numpy":
            return all(bool(np.isfinite(x).all()) for x in xs)
        return bool(torch.stack([torch.isfinite(x).all() for x in xs]).all().item())

    def assign_(self, dst: Any, src: Any) -> None:
        if self.kind == "numpy":
            dst[...] = src
        else:
            dst.copy_(src)

    def sync(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize(self.dev)

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "device": self.device, "dtype_cortex": "float32", "dtype_head": "float64 (NumPy, CPU)",
                             "numpy_version": np.__version__, "max_gather_elements": self.max_gather}
        if self.kind == "torch":
            d.update(torch_version=torch.__version__, cuda_device=(torch.cuda.get_device_name(self.dev) if self.is_cuda else None),
                     tf32=False, deterministic_algorithms="warn_only")
        return d

    def numeric_id(self) -> dict[str, Any]:
        """캐시 키·이어하기 동일성에 쓰는 계산 구현 식별값."""
        d = {"kind": self.kind, "device_type": ("cuda" if self.is_cuda else "cpu"), "numpy": np.__version__}
        if self.kind == "torch":
            d.update(torch=torch.__version__, cuda_device=(torch.cuda.get_device_name(self.dev) if self.is_cuda else None))
        return d


def make_backend(choice: str, max_gather: int) -> Backend:
    """자동: CUDA 가 되면 PyTorch CUDA, 아니면 NumPy CPU. CPU: NumPy. CUDA: PyTorch CUDA (없으면 오류, 몰래 CPU 로 바꾸지 않는다)."""
    c = (choice or "auto").strip().lower()
    if c in ("auto", "자동", ""):
        t, _err = load_torch()
        if t is not None and t.cuda.is_available():
            return Backend("torch", "cuda", max_gather=max_gather)
        return Backend("numpy", max_gather=max_gather)
    if c in ("cpu", "numpy"):
        return Backend("numpy", max_gather=max_gather)
    if c in ("cuda", "gpu"):
        return Backend("torch", "cuda", max_gather=max_gather)
    raise V12Error(f"장치는 자동 / CPU / CUDA 중 하나다 (입력: {choice!r}).")


# ======================================================================
# MNIST (공식 gzip IDX) 와 분할 — v11 과 같은 규칙
#   공식 train 60,000 → seed 별 층화로 dev 10,000 을 뽑고 나머지는 train. 공식 test 10,000 은 그대로 (마지막 평가에만).
# ======================================================================
MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = (
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873", "train_images", (60000, 28, 28)),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432", "train_labels", (60000,)),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3", "test_images", (10000, 28, 28)),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c", "test_labels", (10000,)),
)
SPLITS = ("train", "dev", "test")


def _md5_file(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)                      # 공개 데이터 무결성 확인용 (보안 용도 아님)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mnist_candidates(root: Path, name: str) -> list[Path]:
    return [root / "MNIST" / "raw" / name, root / "raw" / name, root / name]


def mnist_missing(root: Path) -> list[str]:
    return [nm for nm, _md5, _k, _s in MNIST_FILES if not any(p.is_file() for p in mnist_candidates(root, nm))]


def mnist_download(root: Path, names: Sequence[str]) -> None:
    for name in names:
        md5 = next(m for n, m, _k, _s in MNIST_FILES if n == name)
        dest = mnist_candidates(root, name)[0]
        dest.parent.mkdir(parents=True, exist_ok=True)
        say(f"  MNIST 내려받는 중: {name}")
        fd, tmp = tempfile.mkstemp(prefix=name + ".", suffix=".part", dir=str(dest.parent))
        try:
            with os.fdopen(fd, "wb") as out:
                req = urllib.request.Request(MNIST_MIRROR + name, headers={"User-Agent": f"{PROGRAM}/{VERSION}"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    total = 0
                    for block in iter(lambda: resp.read(1 << 20), b""):
                        total += len(block)
                        if total > 64 * 2 ** 20:
                            raise DataError("MNIST 압축 파일 크기 상한 (64 MiB) 을 넘었다.")
                        out.write(block)
            if _md5_file(Path(tmp)) != md5:
                raise DataError(f"내려받은 {name} 의 MD5 가 공식 값과 다르다.")
            os.replace(tmp, dest)
        except (OSError, ValueError) as exc:
            raise DataError(f"MNIST 내려받기 실패 ({name}): {exc}. 인터넷 연결을 확인하거나 공식 gz 4 개를 {root} 에 복사한다.") from exc
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)


def read_idx(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    header = 4 + 4 * len(shape)
    size = header + math.prod(shape)
    with gzip.open(path, "rb") as f:
        payload = f.read(size + 1)
    if len(payload) != size:
        raise DataError(f"MNIST IDX 길이 오류: {path.name} ({len(payload)} != {size})")
    magic = struct.unpack_from(">I", payload, 0)[0]
    dims = struct.unpack_from(">" + "I" * len(shape), payload, 4)
    if magic != 2048 + len(shape) or dims != shape:
        raise DataError(f"MNIST IDX 헤더 오류: {path.name} (magic {magic}, 모양 {dims}, 기대 {shape})")
    arr = np.frombuffer(payload, np.uint8, offset=header).reshape(shape)
    if len(shape) == 1 and (arr > 9).any():
        raise DataError(f"MNIST 라벨 범위 오류: {path.name}")
    return arr


class Mnist:
    """공식 파일 4 개 (MD5 확인) 와 분할. 분할 표본 = (출처 0 train / 1 test, 공식 번호, 라벨) 배열, 목록 순서는 v11 과 같다."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.files: dict[str, dict[str, Any]] = {}
        arrays: dict[str, np.ndarray] = {}
        miss = mnist_missing(self.root)
        if miss:
            raise DataError(f"MNIST 파일이 없다: {', '.join(miss)} (폴더 {self.root}). 공식 gz 4 개를 이 폴더 (또는 그 안의 raw/ · MNIST/raw/) 에 "
                            f"두거나 내려받기를 허용한다. 주소: {MNIST_MIRROR}")
        for name, md5, key, shape in MNIST_FILES:
            p = next(c for c in mnist_candidates(self.root, name) if c.is_file())
            if _md5_file(p) != md5:
                raise DataError(f"MNIST 체크섬 불일치: {p}. 공식 파일이 아니다 (다른 파일을 옮기고 다시 준비한다).")
            arrays[key] = read_idx(p, shape)
            self.files[name] = {"path": str(p), "md5": md5, "shape": list(shape)}
        for src in ("train", "test"):
            if len(arrays[src + "_images"]) != len(arrays[src + "_labels"]):
                raise DataError(f"MNIST {src} 영상·라벨 개수가 다르다.")
        self.images = {0: arrays["train_images"], 1: arrays["test_images"]}
        self.labels = {0: arrays["train_labels"].astype(np.int64), 1: arrays["test_labels"].astype(np.int64)}
        self.split: dict[str, dict[str, np.ndarray]] = {}
        self.limits_info: dict[str, Any] = {}

    def make_split(self, seed: int, dev_size: int) -> None:
        """v11 MNISTDatasetManager.split 와 같은 규칙: 클래스별 할당 (최대 잔여 방식) → default_rng([seed, 107]) 로 클래스마다 dev 선택."""
        labels = self.labels[0]
        counts = np.bincount(labels, minlength=10)
        n_dev = int(dev_size)
        if np.any(counts < 2) or not 10 <= n_dev <= len(labels) - 10:
            raise DataError("MNIST train/dev 에 숫자마다 최소 1 장을 배정할 수 없다 (mnist_dev_size 확인).")
        quotas = counts.astype(np.float64) * n_dev / len(labels)
        alloc = np.minimum(np.maximum(np.floor(quotas).astype(int), 1), counts - 1)
        while int(alloc.sum()) < n_dev:
            alloc[int(np.argmax(np.where(alloc < counts - 1, quotas - alloc, -np.inf)))] += 1
        while int(alloc.sum()) > n_dev:
            alloc[int(np.argmax(np.where(alloc > 1, alloc - quotas, -np.inf)))] -= 1
        rng = np.random.default_rng([int(seed), 107])
        dev = np.zeros(len(labels), bool)
        for c in range(10):
            inds = np.flatnonzero(labels == c)
            dev[rng.permutation(inds)[:int(alloc[c])]] = True
        tr, dv = np.flatnonzero(~dev), np.flatnonzero(dev)
        te = np.arange(len(self.labels[1]))
        self.split = {"train": self._mk(0, tr), "dev": self._mk(0, dv), "test": self._mk(1, te)}
        self._split_manifest_sha = sha256_text(dumps(sorted(
            [(self.relpath(0, int(i)), "dev" if dev[i] else "train") for i in range(len(labels))]
            + [(self.relpath(1, int(i)), "test") for i in te]), indent=None))

    def _mk(self, src: int, idx: np.ndarray) -> dict[str, np.ndarray]:
        idx = np.asarray(idx, np.int64)
        return {"src": np.full(len(idx), src, np.int8), "idx": idx, "y": self.labels[src][idx].astype(np.int64)}

    def apply_limits(self, limits: dict[str, int], seed: int) -> None:
        """분할마다 표본 한도 (0 = 전체). v11 apply_split_limits 와 같다: 클래스별 default_rng([seed, 7070, 분할번호]) 순서를 번갈아 뽑고 목록 순서 유지."""
        info: dict[str, Any] = {}
        for k, sp in enumerate(SPLITS):
            d = self.split[sp]
            n_av, lim = len(d["idx"]), int(limits.get(sp, 0) or 0)
            if lim and n_av > lim:
                rng = np.random.default_rng([int(seed), 7070, k])
                by_c: dict[int, list[int]] = {}
                for i, c in enumerate(d["y"].tolist()):
                    by_c.setdefault(int(c), []).append(i)
                queues = {c: [idx[j] for j in rng.permutation(len(idx))] for c, idx in sorted(by_c.items())}
                pick: list[int] = []
                while len(pick) < lim:
                    for c in sorted(queues):
                        if queues[c] and len(pick) < lim:
                            pick.append(queues[c].pop(0))
                sel = np.asarray(sorted(pick), np.int64)
                self.split[sp] = {kk: v[sel] for kk, v in d.items()}
            info[sp] = {"n_available": n_av, "n_used": len(self.split[sp]["idx"]), "limit": lim}
        self.limits_info = info

    @staticmethod
    def relpath(src: int, idx: int) -> str:
        return f"mnist/{'train' if int(src) == 0 else 'test'}/{int(idx):05d}.png"

    def ids_sha(self, sp: str) -> str:
        d = self.split[sp]
        return sha256_text("\n".join(sorted(self.relpath(s, i) for s, i in zip(d["src"].tolist(), d["idx"].tolist()))))

    def split_sha256(self) -> str:
        """v11 공통 초기 기록의 split_sha256 와 같은 식 (한도 적용 뒤 분할별 ID 해시)."""
        return sha256_text(dumps({sp: self.ids_sha(sp) for sp in SPLITS}, indent=None))

    def order_sha(self, sp: str) -> str:
        """분할 목록 순서 그대로의 (출처, 번호, 라벨) 해시 — 캐시 행 순서 확인용."""
        d = self.split[sp]
        return sha256_array(d["src"], d["idx"], d["y"])

    def image(self, src: int, idx: int) -> Any:
        return Image.fromarray(self.images[int(src)][int(idx)]).convert("RGB")


# ======================================================================
# 망막/LGN 근사와 로그-극좌표 표본 (v11 RetinaLGNPreprocessor 와 같은 계산)
#   RGB [0,1] → S×S (Lanczos, 가장자리 복제) → lum = R+G → DoG (각 커널 합 1, reflect) → ON/OFF
#   → 이심률별 저역통과 (표본 간격 × β 에 가장 가까운 Gaussian 단계) → 로그-극좌표 쌍선형 표본 (위치 우선, float32)
# ======================================================================
def logpolar_positions(n_r: int, n_phi: int, r0: float, rmax: float, cx: float, cy: float, phi_offset: float) -> dict[str, Any]:
    L = math.log1p(rmax / r0)
    u = (np.arange(n_r) + 0.5) / n_r
    r = r0 * (np.exp(u * L) - 1.0)
    phi = 2.0 * math.pi * (np.arange(n_phi) + phi_offset) / n_phi
    R, PHI = np.meshgrid(r, phi, indexing="ij")
    x = cx + R * np.cos(PHI)
    y = cy - R * np.sin(PHI)
    return {"n_r": n_r, "n_phi": n_phi, "x": x.ravel(), "y": y.ravel(), "r": R.ravel(), "ring": np.repeat(np.arange(n_r), n_phi),
            "ang": np.tile(np.arange(n_phi), n_r), "spacing_r": ((R + r0) * L / n_r).ravel(), "spacing_phi": (2.0 * math.pi * R / n_phi).ravel(),
            "L": L}


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


class RetinaLGN:
    def __init__(self, model: dict[str, Any]):
        rt, lp = model["retina"], model["logpolar"]
        self.S = S = int(model["image_size"])
        self.cx = self.cy = (S - 1) / 2.0
        self.colors = list(model["areas"]["V1"]["color_channels"])
        self.maps = [f"{c}_{pol}" for c in self.colors for pol in ("on", "off")]
        self.n_maps = len(self.maps)
        self.gain, self.trunc = float(rt["gain"]), float(rt["truncate_sigma"])
        self.kc = gauss1d(float(rt["sigma_center_px"]), self.trunc, S)
        self.ks = gauss1d(float(rt["sigma_surround_px"]), self.trunc, S)
        self.r0 = float(lp["r0_px"])
        self.rmax = S / 2.0 - float(lp["rmax_margin_px"])
        self.phi_offset = float(lp["phi_offset_bins"])
        g = logpolar_positions(int(lp["n_r"]), int(lp["n_phi"]), self.r0, self.rmax, self.cx, self.cy, self.phi_offset)
        self.grid = g
        self.n_pos = int(lp["n_r"]) * int(lp["n_phi"])
        self.n_out = self.n_pos * self.n_maps
        x, y = g["x"], g["y"]
        if (x < 0).any() or (x > S - 1).any() or (y < 0).any() or (y > S - 1).any():
            raise V12Error("로그-극좌표 표본이 영상 밖에 있다 (MODEL logpolar.rmax_margin_px 확인).")
        self.levels = sorted({0.0} | {float(v) for v in lp["blur_levels_px"]})
        req = float(lp["blur_beta"]) * np.maximum(g["spacing_r"], g["spacing_phi"])
        self.level_of_pos = np.argmin(np.abs(req[:, None] - np.asarray(self.levels)[None, :]), axis=1)
        self.level_kernels = [gauss1d(s, self.trunc, S) for s in self.levels]
        x0 = np.clip(np.floor(x).astype(np.int64), 0, S - 2)
        y0 = np.clip(np.floor(y).astype(np.int64), 0, S - 2)
        self._bil = (y0, x0, (y - y0).astype(np.float64), (x - x0).astype(np.float64))
        self.sample_area_px2 = (g["spacing_r"] * g["spacing_phi"]).astype(np.float64)
        self._lanczos = getattr(getattr(Image, "Resampling", Image), "LANCZOS") if Image is not None else None

    def resize_pad(self, im: Any) -> np.ndarray:
        S = self.S
        w, h = im.size
        s = S / float(max(w, h))
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        im2 = im.resize((nw, nh), self._lanczos) if (nw, nh) != (w, h) else im
        arr = np.asarray(im2, dtype=np.float32) / 255.0
        top, left = (S - nh) // 2, (S - nw) // 2
        return np.pad(arr, ((top, S - nh - top), (left, S - nw - left), (0, 0)), mode="edge")

    def lgn_maps(self, arr: np.ndarray) -> dict[str, np.ndarray]:
        R, G, B = (arr[..., i].astype(np.float64) for i in range(3))
        opp = {"lum": R + G, "rg": R - G, "by": B - 0.5 * (R + G)}
        out: dict[str, np.ndarray] = {}
        for c in self.colors:
            dg = self.gain * (blur2d(opp[c], self.kc) - blur2d(opp[c], self.ks))
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

    def vector(self, im: Any) -> np.ndarray:
        return self.sample(self.lgn_maps(np.asarray(self.resize_pad(im), np.float32)))

    def describe(self) -> dict[str, Any]:
        return {"image_size": self.S, "maps": self.maps, "n_positions": self.n_pos, "n_outputs": self.n_out, "rmax_px": self.rmax,
                "levels_px": self.levels}


# ======================================================================
# 고정 연결 구조 (CORTEX_FIXED: V1 → V2 → V4 고정 기하 → IT). 데이터·라벨을 쓰지 않는다.
#   뉴런 순서 = 위치 우선 (local = pos·n_ch + ch). 입력 연결은 CSR 입력군 A·B (B 는 min2 뉴런만: u = min(ΣA, ΣB)).
#   구조 해시는 v11 LayerSpec.hash · area_structure_hash 와 같은 식이다 (공통 초기 snapshot 검증에 쓴다).
# ======================================================================
class LayerSpec:
    def __init__(self, area: str, name: str, src_key: str, n_src: int, n_pos: int, n_ch: int, names: Sequence[str], roles: Sequence[str],
                 meta: dict[str, list[Any]], op_min: Any, csr: Sequence[np.ndarray], x: Any, y: Any,
                 pos: tuple[Any, Any, Any], grid_shape: tuple[int, int] | None, nb: tuple[np.ndarray, np.ndarray],
                 missing: np.ndarray | None = None, fixed_params: dict[str, Any] | None = None):
        a_ptr, a_src, a_w, b_ptr, b_src, b_w = csr
        self.area, self.name, self.src_key = area, name, src_key
        self.n_src, self.n_pos, self.n_ch = int(n_src), int(n_pos), int(n_ch)
        self.channel_names, self.channel_roles, self.channel_meta = list(names), list(roles), meta
        self.op_min = np.asarray(op_min, bool)
        self.a_ptr, self.a_src, self.a_w = a_ptr, a_src, np.asarray(a_w).astype(np.float32)
        self.b_ptr, self.b_src, self.b_w = b_ptr, b_src, np.asarray(b_w).astype(np.float32)
        self.x, self.y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        self.pos_x, self.pos_y, self.pos_rf = (np.asarray(v, np.float32) for v in pos)
        self.grid_shape = grid_shape
        self.pos_nb, self.pos_nb_count = nb
        self.missing = missing                      # V4 L2 고정 기하: 합법 B 원천이 없는 구조상 비활성 검출기 (u = s = 0)
        self.fixed_params = fixed_params
        self.start = self.end = -1
        self._validate()

    @property
    def key(self) -> str:
        return f"{self.area}/{self.name}"

    @property
    def n(self) -> int:
        return self.n_pos * self.n_ch

    def _validate(self) -> None:
        n = self.n
        if len(self.op_min) != n or len(self.x) != n or len(self.channel_names) != self.n_ch:
            raise V12Error(f"{self.key}: 구조 배열 길이 오류")
        for g, ptr, src, w in (("A", self.a_ptr, self.a_src, self.a_w), ("B", self.b_ptr, self.b_src, self.b_w)):
            if len(ptr) != n + 1 or ptr[0] != 0 or (np.diff(ptr) < 0).any() or ptr[-1] != len(src) or len(src) != len(w):
                raise V12Error(f"{self.key}: 입력군 {g} CSR 형식 오류")
            if len(src) and (src.min() < 0 or src.max() >= self.n_src):
                raise V12Error(f"{self.key}: 입력군 {g} 주소가 원천 층 범위 밖")
            if not np.isfinite(w).all():
                raise V12Error(f"{self.key}: 입력군 {g} 가중치에 NaN/Inf")
        fa, fb = np.diff(self.a_ptr), np.diff(self.b_ptr)
        inact = self.missing if self.missing is not None else np.zeros(n, bool)
        if (fb[~self.op_min] > 0).any() or ((fa[self.op_min] == 0) | ((fb[self.op_min] == 0) & ~inact[self.op_min])).any():
            raise V12Error(f"{self.key}: min2 뉴런은 입력군 A·B 가 모두 있어야 하고 선형 뉴런에는 B 가 없어야 한다.")
        if self.name == "L3" and ((fa + fb) == 0).any():
            raise V12Error(f"{self.key}: 입력 없는 L3 출력이 있다.")

    def hash(self) -> str:
        names = sha256_text(dumps([self.key, self.src_key, self.n_src, self.n_pos, self.n_ch, self.channel_names, self.channel_roles,
                                   list(self.grid_shape) if self.grid_shape else None], indent=None))
        h = sha256_text(names + sha256_array(self.op_min, self.a_ptr, self.a_src, self.a_w, self.b_ptr, self.b_src, self.b_w,
                                             self.x, self.y, self.pos_nb, self.pos_nb_count))
        if self.fixed_params is not None:
            return sha256_text(h + sha256_array(self.missing) + sha256_text(dumps(self.fixed_params, indent=None)))
        return h


def area_hash(specs: Sequence[LayerSpec]) -> str:
    return sha256_text("|".join(sp.hash() for sp in specs))


class CSR:
    """행별 (원천 → 가중치). 같은 (행, 입력군, 원천) 이 다시 오면 더한다 (v11 merge_sum). 끝에 원천 번호 순으로 정렬."""

    def __init__(self, n_rows: int, n_src: int):
        self.n, self.n_src = int(n_rows), int(n_src)
        self.a: list[dict[int, float]] = [{} for _ in range(self.n)]
        self.b: list[dict[int, float]] = [{} for _ in range(self.n)]

    def add(self, row: int, src: int, w: float, group: str = "A") -> None:
        if not 0 <= src < self.n_src:
            raise V12Error(f"원천 주소 {src} 가 범위 [0,{self.n_src}) 밖이다.")
        d = (self.a if group == "A" else self.b)[row]
        d[src] = d[src] + float(w) if src in d else float(w)

    @staticmethod
    def _csr(rows: list[dict[int, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ptr = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
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
    """피질 격자 Chebyshev 거리 ≤ radius 인 위치 (자기 포함, 각도 wrap, 반경 경계에서 자름). 채움 = n_pos."""
    n = n_r * n_phi
    rows = []
    for k in range(n_r):
        for m in range(n_phi):
            s = set()
            for dk in range(-radius, radius + 1):
                if 0 <= k + dk < n_r:
                    for dm in range(-radius, radius + 1):
                        s.add((k + dk) * n_phi + (m + dm) % n_phi)
            rows.append(sorted(s))
    K = max(len(r) for r in rows)
    idx = np.full((n, K), n, np.int64)
    for i, r in enumerate(rows):
        idx[i, :len(r)] = r
    return idx, np.asarray([len(r) for r in rows], np.int64)


def single_position_neighbors() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((1, 1), np.int64), np.ones(1, np.int64)


def _per(arr: Any, n_ch: int) -> np.ndarray:
    return np.repeat(np.asarray(arr), n_ch)


def _block_children(nr_c: int, nphi_c: int, nr_p: int, nphi_p: int) -> list[list[int]]:
    fr, fa = nr_c // nr_p, nphi_c // nphi_p
    return [[k * nphi_c + m for k in range(K * fr, (K + 1) * fr) for m in range(M * fa, (M + 1) * fa)]
            for K in range(nr_p) for M in range(nphi_p)]


def _pool_positions(child: LayerSpec, children: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    px = np.asarray([child.pos_x[c].mean() for c in children], np.float64)
    py = np.asarray([child.pos_y[c].mean() for c in children], np.float64)
    prf = np.asarray([float(np.max(np.hypot(child.pos_x[c] - px[i], child.pos_y[c] - py[i]) + child.pos_rf[c]))
                      for i, c in enumerate(children)], np.float64)
    return px, py, prf


def _local_spacing(px: np.ndarray, py: np.ndarray, pre: RetinaLGN, n_r: int, n_phi: int) -> np.ndarray:
    r = np.hypot(px - pre.cx, pre.cy - py)
    L = pre.grid["L"]
    return np.sqrt(((r + pre.r0) * L / n_r) * (2 * math.pi * r / n_phi))


def _displaced(px: np.ndarray, py: np.ndarray, P: int, theta_deg: float, D: float, sign: int) -> int | None:
    """원영상 좌표에서 θ 방향 (sign +1 앞, −1 뒤) 으로 D 떨어진 점에 가장 가까운 격자 위치 (자기 자신이면 None)."""
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


def wrap180(x: Any) -> Any:
    return (np.asarray(x, np.float64) + 180.0) % 360.0 - 180.0


def curve_directions(theta_a: float, sign: int, kappa: float) -> tuple[float, float, float]:
    return float(theta_a), float(theta_a + sign * kappa), float(theta_a + sign * kappa / 2.0)


def hermite_path(P: np.ndarray, Q: np.ndarray, phi_a: float, phi_b: float, M: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, int(M))[:, None]
    L = float(np.hypot(*(Q - P)))
    vA = L * np.asarray([math.cos(math.radians(phi_a)), math.sin(math.radians(phi_a))])
    vB = L * np.asarray([math.cos(math.radians(phi_b)), math.sin(math.radians(phi_b))])
    p = (2 * t ** 3 - 3 * t ** 2 + 1) * P + (t ** 3 - 2 * t ** 2 + t) * vA + (-2 * t ** 3 + 3 * t ** 2) * Q + (t ** 3 - t ** 2) * vB
    dp = (6 * t ** 2 - 6 * t) * P + (3 * t ** 2 - 4 * t + 1) * vA + (-6 * t ** 2 + 6 * t) * Q + (3 * t ** 2 - 2 * t) * vB
    return p, dp


# ---------------------------------------------------------------------- V1: LGN 중계 (L4) → 방향·위상 Gabor 단순 세포 (L2) → 위상×극성 결합 (L3)
def build_v1(m: dict[str, Any], pre: RetinaLGN) -> list[LayerSpec]:
    a = m["areas"]["V1"]
    rad6 = int(m["l6"]["neighbor_grid_radius"])
    lg = pre.grid
    n_maps, n4 = pre.n_maps, pre.n_out
    sp_lg = np.maximum(lg["spacing_r"], lg["spacing_phi"])
    L4 = LayerSpec("V1", "L4", "LGN", n4, pre.n_pos, n_maps, list(pre.maps), ["V1_L4_lgn_relay"] * n_maps, {}, np.zeros(n4, bool),
                   identity_csr(n4), _per(lg["x"], n_maps), _per(lg["y"], n_maps), (lg["x"], lg["y"], sp_lg), (lg["n_r"], lg["n_phi"]),
                   grid_neighbors(lg["n_r"], lg["n_phi"], rad6))
    g1 = logpolar_positions(int(a["n_r"]), int(a["n_phi"]), pre.r0, pre.rmax, pre.cx, pre.cy, pre.phi_offset)
    n_pos1 = int(a["n_r"]) * int(a["n_phi"])
    sigma = np.maximum(float(a["gabor_sigma_factor"]) * np.sqrt(g1["spacing_r"] * g1["spacing_phi"]), float(a["gabor_sigma_min_px"]))
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
    sels, total = [], 0
    for p in range(n_pos1):
        d2 = (lx - g1["x"][p]) ** 2 + (ly - g1["y"][p]) ** 2
        sel = np.nonzero(d2 <= rad[p] ** 2)[0]
        if sel.size == 0:
            sel = np.asarray([int(np.argmin(d2))])
        sels.append(sel)
        total += n_ch2 * 2 * sel.size
    a_ptr = np.zeros(n_pos1 * n_ch2 + 1, np.int64)
    a_src = np.empty(total, np.int64)
    a_w = np.empty(total, np.float32)
    cos_t, sin_t = np.cos(np.radians(oris))[:, None], np.sin(np.radians(oris))[:, None]
    on_idx = [pre.maps.index(f"{c}_on") for c in colors]
    off_idx = [pre.maps.index(f"{c}_off") for c in colors]
    pos = row = 0
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
    nb1 = grid_neighbors(int(a["n_r"]), int(a["n_phi"]), rad6)
    grid1 = (int(a["n_r"]), int(a["n_phi"]))
    names2 = [f"{colors[c]}_o{oris[o]:g}_p{f}_{'+' if pols[s] > 0 else '-'}" for (c, o, f, s) in chans]
    L2 = LayerSpec("V1", "L2", "V1/L4", n4, n_pos1, n_ch2, names2, ["V1_simple_oriented_gabor"] * n_ch2, {}, np.zeros(n2, bool),
                   (a_ptr, a_src, a_w, np.zeros(n2 + 1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)),
                   _per(g1["x"], n_ch2), _per(g1["y"], n_ch2), (g1["x"], g1["y"], sigma), grid1, nb1)
    n_ch3, kpar = len(colors) * n_o, n_f * n_s
    n3 = n_pos1 * n_ch3
    a3_src = np.empty(n3 * kpar, np.int64)
    r_ = 0
    for p in range(n_pos1):
        for c in range(len(colors)):
            for o in range(n_o):
                base = p * n_ch2 + ((c * n_o + o) * n_f) * n_s
                a3_src[r_ * kpar:(r_ + 1) * kpar] = base + np.arange(kpar)
                r_ += 1
    names3 = [f"{colors[c]}_o{oris[o]:g}" for c in range(len(colors)) for o in range(n_o)]
    L3 = LayerSpec("V1", "L3", "V1/L2", n2, n_pos1, n_ch3, names3, ["V1_phase_pooled_orientation"] * n_ch3, {}, np.zeros(n3, bool),
                   (np.arange(n3 + 1, dtype=np.int64) * kpar, a3_src, np.full(n3 * kpar, 1.0 / kpar, np.float32),
                    np.zeros(n3 + 1, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)),
                   _per(g1["x"], n_ch3), _per(g1["y"], n_ch3), (g1["x"], g1["y"], sigma), grid1, nb1)
    return [L4, L2, L3]


# ---------------------------------------------------------------------- V2: 방향 블록 평균 (L4) → 윤곽 연속·끝점·교차 min2 (L2) → 일대일 (L3)
def build_v2(m: dict[str, Any], pre: RetinaLGN, v1: LayerSpec) -> list[LayerSpec]:
    a = m["areas"]["V2"]
    rad6 = int(m["l6"]["neighbor_grid_radius"])
    oris = [float(t) for t in m["areas"]["V1"]["orientations_deg"]]
    colors = list(pre.colors)
    n_o = len(oris)
    nr1, nphi1 = v1.grid_shape
    nr2, nphi2 = int(a["n_r"]), int(a["n_phi"])
    children = _block_children(nr1, nphi1, nr2, nphi2)
    px, py, prf = _pool_positions(v1, children)
    n_pos = nr2 * nphi2
    nb = grid_neighbors(nr2, nphi2, rad6)
    b4 = CSR(n_pos * n_o, v1.n)
    for P, ch in enumerate(children):
        k = 1.0 / (len(ch) * len(colors))
        for o in range(n_o):
            for c1 in ch:
                for ci in range(len(colors)):
                    b4.add(P * n_o + o, c1 * v1.n_ch + ci * n_o + o, k)
    L4 = LayerSpec("V2", "L4", "V1/L3", v1.n, n_pos, n_o, [f"o{t:g}" for t in oris], ["V2_L4_pooled_orientation"] * n_o, {},
                   np.zeros(n_pos * n_o, bool), b4.finalize(), _per(px, n_o), _per(py, n_o), (px, py, prf), (nr2, nphi2), nb)
    D = np.maximum(float(a["offset_factor"]) * _local_spacing(px, py, pre, nr2, nphi2), 1.0)
    pairs: list[tuple[int, int]] = []
    for diff in a["cross_diffs_deg"]:
        for i in range(n_o):
            j = _ori_index(oris, oris[i] + float(diff))
            if j is not None and j != i and (min(i, j), max(i, j)) not in pairs:
                pairs.append((min(i, j), max(i, j)))
    chans: list[tuple[str, int, int]] = ([("contour", o, 0) for o in range(n_o)] + [("endstop", o, 1) for o in range(n_o)]
                                         + [("endstop", o, -1) for o in range(n_o)] + [("cross", i, j) for (i, j) in pairs])
    n_ch = len(chans)
    b2 = CSR(n_pos * n_ch, n_pos * n_o)
    op = np.zeros(n_pos * n_ch, bool)
    inh = float(a["endstop_inhibition"])
    for P in range(n_pos):
        for ci, (fam, i, j) in enumerate(chans):
            r = P * n_ch + ci
            if fam == "contour":
                f = _displaced(px, py, P, oris[i], D[P], +1)
                bk = _displaced(px, py, P, oris[i], D[P], -1)
                parts = [(P, 0.5)] + ([(f, 0.25)] if f is not None else []) + ([(bk, 0.25)] if bk is not None else [])
                tot = sum(w for _q, w in parts)
                for q_, w in parts:
                    b2.add(r, q_ * n_o + i, w / tot)
            elif fam == "endstop":
                q_ = _displaced(px, py, P, oris[i], D[P], j)
                b2.add(r, P * n_o + i, 1.0)
                if q_ is not None and inh > 0:
                    b2.add(r, q_ * n_o + i, -inh)
            else:
                b2.add(r, P * n_o + i, 1.0, "A")
                b2.add(r, P * n_o + j, 1.0, "B")
                op[r] = True
    names2 = [f"contour_o{oris[i]:g}" if f == "contour" else (f"endstop_o{oris[i]:g}_{'fwd' if j > 0 else 'bwd'}" if f == "endstop"
              else f"cross_o{oris[i]:g}_o{oris[j]:g}") for (f, i, j) in chans]
    roles2 = ["V2_contour_continuation" if f == "contour" else ("V2_end_stopped" if f == "endstop" else "V2_crossing_junction")
              for (f, _i, _j) in chans]
    meta = {"family": [f for (f, _i, _j) in chans], "orientation_deg": [oris[i] for (_f, i, _j) in chans],
            "direction": [(j if f == "endstop" else 0) for (f, _i, j) in chans]}
    L2 = LayerSpec("V2", "L2", "V2/L4", n_pos * n_o, n_pos, n_ch, names2, roles2, meta, op, b2.finalize(), _per(px, n_ch), _per(py, n_ch),
                   (px, py, prf + D), (nr2, nphi2), nb)
    L3 = LayerSpec("V2", "L3", "V2/L2", n_pos * n_ch, n_pos, n_ch, names2, [r_ + "_L3" for r_ in roles2], dict(meta),
                   np.zeros(n_pos * n_ch, bool), identity_csr(n_pos * n_ch), L2.x, L2.y, (px, py, prf + D), (nr2, nphi2), nb)
    return [L4, L2, L3]


# ---------------------------------------------------------------------- V4 고정 기하 (v11 v4_fixed_geometry): 블록 평균 (L4) → 곡선·직선 min2 / 접합 / 선 끝 (L2) → 격자 이웃 평균 (L3)
def curve_candidates(pre: RetinaLGN, l4: LayerSpec, l2_rf: np.ndarray, n_ch: int, chans: list[tuple[str, int, int]],
                     curve_info: list[tuple[int, int, int, int, float]], contour: dict[int, int], oris: list[float], M: int
                     ) -> dict[int, list[tuple[int, int]]]:
    """곡선 L2 행 (P, θA → θB, 부호 s, κ) 의 합법 B 원천 (θB contour 위치 Q): 같은 위치·수용장 밖 제외, 진행 방향·굽힘 쪽·Hermite 경로가
    시야 안·접선 0 아님·접힘 없음 (v11 curve_route_options 의 valid 조건). 원천 주소 오름차순."""
    px, py = l4.pos_x.astype(np.float64), l4.pos_y.astype(np.float64)
    vx, vy = px - pre.cx, pre.cy - py
    n_pos, n_in = l4.n_pos, l4.n_ch
    cinfo = {ch: (i, j, s, kap) for (ch, i, j, s, kap) in curve_info}
    out: dict[int, list[tuple[int, int]]] = {}
    for P in range(n_pos):
        Rj = float(l2_rf[P])
        Pv = np.asarray([vx[P], vy[P]])
        for ch in range(n_ch):
            if chans[ch][0] != "curve":
                continue
            i, j, s, kap = cinfo[ch]
            phi_a, phi_b, _phi_c = curve_directions(oris[i], s, kap)
            ta = np.asarray([math.cos(math.radians(phi_a)), math.sin(math.radians(phi_a))])
            items = []
            for Q in range(n_pos):
                if Q == P or float(math.hypot(px[Q] - px[P], py[Q] - py[P])) > Rj:
                    continue
                Qv = np.asarray([vx[Q], vy[Q]])
                chord = Qv - Pv
                L = float(np.hypot(*chord))
                if L <= 1e-12:
                    raise V12Error(f"V4 위치 {P} 와 {Q} 가 같은 좌표다.")
                p_, dp_ = hermite_path(Pv, Qv, phi_a, phi_b, M)
                psi = np.degrees(np.arctan2(dp_[:, 1], dp_[:, 0]))
                tot_abs = float(np.abs(wrap180(np.diff(psi))).sum())
                if (float(chord @ ta) <= 0.0 or s * float(ta[0] * chord[1] - ta[1] * chord[0]) <= 0.0
                        or (np.hypot(p_[:, 0], p_[:, 1]) > pre.rmax).any() or (np.hypot(dp_[:, 0], dp_[:, 1]) < 1e-9 * L).any()
                        or ((dp_ @ chord) <= 0.0).any() or tot_abs > 180.0):
                    continue
                items.append((Q * n_in + contour[j], Q))
            items.sort(key=lambda it: it[0])
            out[P * n_ch + ch] = items
    return out


def build_v4(m: dict[str, Any], pre: RetinaLGN, v2: LayerSpec) -> list[LayerSpec]:
    a = m["areas"]["V4"]
    rad6 = int(m["l6"]["neighbor_grid_radius"])
    oris = [float(t) for t in m["areas"]["V1"]["orientations_deg"]]
    n_o = len(oris)
    nr2, nphi2 = v2.grid_shape
    nr4, nphi4 = int(a["n_r"]), int(a["n_phi"])
    children = _block_children(nr2, nphi2, nr4, nphi4)
    px, py, prf = _pool_positions(v2, children)
    n_pos = nr4 * nphi4
    nb = grid_neighbors(nr4, nphi4, rad6)
    n_in = v2.n_ch
    b4 = CSR(n_pos * n_in, v2.n)
    for P, ch in enumerate(children):
        for cch in range(n_in):
            for c in ch:
                b4.add(P * n_in + cch, c * n_in + cch, 1.0 / len(ch))
    L4 = LayerSpec("V4", "L4", "V2/L3", v2.n, n_pos, n_in, list(v2.channel_names), ["V4_L4_pooled_" + r for r in v2.channel_roles],
                   dict(v2.channel_meta), np.zeros(n_pos * n_in, bool), b4.finalize(), _per(px, n_in), _per(py, n_in), (px, py, prf),
                   (nr4, nphi4), nb)
    fam, ori_in, dirs = v2.channel_meta["family"], v2.channel_meta["orientation_deg"], v2.channel_meta["direction"]
    contour = {i: next(k for k in range(n_in) if fam[k] == "contour" and abs(ori_in[k] - oris[i]) < 1e-6) for i in range(n_o)}
    endf = {i: next(k for k in range(n_in) if fam[k] == "endstop" and dirs[k] > 0 and abs(ori_in[k] - oris[i]) < 1e-6) for i in range(n_o)}
    endb = {i: next(k for k in range(n_in) if fam[k] == "endstop" and dirs[k] < 0 and abs(ori_in[k] - oris[i]) < 1e-6) for i in range(n_o)}
    crosses = [k for k in range(n_in) if fam[k] == "cross"]
    chans: list[tuple[str, int, int]] = []
    curve_info: list[tuple[int, int, int, int, float]] = []
    for kappa in a["curvature_deg"]:
        for sgn in (1, -1):
            for i in range(n_o):
                j = _ori_index(oris, oris[i] + sgn * float(kappa))
                if j is not None and j != i:
                    curve_info.append((len(chans), i, j, sgn, float(kappa)))
                    chans.append(("curve", i, j))
    chans += [("straight", i, i) for i in range(n_o)] + [("junction", k, k) for k in crosses] + [("lineend", i, i) for i in range(n_o)]
    n_ch = len(chans)
    D = np.maximum(float(a["offset_factor"]) * _local_spacing(px, py, pre, nr4, nphi4), 1.0)
    # 곡선 B = 합법 후보 중 ‖Q − Q_target‖² / max(D², ε) 최소 (동률 → 위치 ID → 원천 ID), Q_target = P + D·(cos φc, sin φc). 모델 구성 때 한 번.
    eps = float(m["v4_fixed"]["score_epsilon"])
    cand = curve_candidates(pre, L4, prf + D, n_ch, chans, curve_info, contour, oris, int(m["curve_path_points"]))
    cinfo = {ch: (i, j, s, kap) for (ch, i, j, s, kap) in curve_info}
    inactive = np.zeros(n_pos * n_ch, bool)
    b2 = CSR(n_pos * n_ch, n_pos * n_in)
    op = np.zeros(n_pos * n_ch, bool)
    for P in range(n_pos):
        for ci, (f, i, j) in enumerate(chans):
            r = P * n_ch + ci
            if f in ("curve", "straight"):
                b2.add(r, P * n_in + contour[i], 1.0, "A")
                op[r] = True
                if f == "curve":
                    _i, _j, s, kap = cinfo[ci]
                    phi_c = curve_directions(oris[i], s, kap)[2]
                    tx = float(px[P]) + float(D[P]) * math.cos(math.radians(phi_c))
                    ty = float(py[P]) - float(D[P]) * math.sin(math.radians(phi_c))
                    items = cand[r]
                    bsrc = -1
                    if items:
                        src_ = np.asarray([it[0] for it in items], np.int64)
                        q_ = np.asarray([it[1] for it in items], np.int64)
                        sc = ((px[q_] - tx) ** 2 + (py[q_] - ty) ** 2) / max(float(D[P]) ** 2, eps)
                        bsrc = int(src_[np.lexsort((src_, q_, sc))[0]])
                else:
                    q_f = _displaced(px, py, P, oris[i], D[P], +1)
                    q_b = _displaced(px, py, P, oris[i], D[P], -1) if q_f is None else None
                    qq = q_f if q_f is not None else (q_b if q_b is not None else -1)
                    bsrc = qq * n_in + contour[i] if qq >= 0 else -1
                if bsrc >= 0:
                    b2.add(r, int(bsrc), 1.0, "B")
                else:
                    inactive[r] = True                                   # 합법 B 원천 없음 → 구조상 비활성 (같은 위치로 대체하지 않는다)
            elif f == "junction":
                b2.add(r, P * n_in + i, 1.0)
            else:
                b2.add(r, P * n_in + endf[i], 0.5)
                b2.add(r, P * n_in + endb[i], 0.5)
    names2, roles2, fam2 = [], [], []
    for (f, i, j) in chans:
        if f == "curve":
            names2.append(f"curve_o{oris[i]:g}_to_o{oris[j]:g}")
            roles2.append("V4_curve_fragment")
        elif f == "straight":
            names2.append(f"straight_o{oris[i]:g}")
            roles2.append("V4_straight_continuation")
        elif f == "junction":
            names2.append("junction_" + v2.channel_names[i])
            roles2.append("V4_junction_relay")
        else:
            names2.append(f"lineend_o{oris[i]:g}")
            roles2.append("V4_line_end")
        fam2.append(f)
    fixed_params = {"mode": FIXED_MODE, "score_epsilon": eps, "curve_path_points": int(m["curve_path_points"])}
    L2 = LayerSpec("V4", "L2", "V4/L4", n_pos * n_in, n_pos, n_ch, names2, roles2, {"family": fam2}, op, b2.finalize(), _per(px, n_ch),
                   _per(py, n_ch), (px, py, prf + D), (nr4, nphi4), nb, missing=inactive, fixed_params=fixed_params)
    nb3, nb3c = grid_neighbors(nr4, nphi4, int(a["l3_grid_radius"]))
    b3 = CSR(n_pos * n_ch, n_pos * n_ch)
    for P in range(n_pos):
        qs = [int(q) for q in nb3[P, :nb3c[P]]]
        for cch in range(n_ch):
            for q_ in qs:
                b3.add(P * n_ch + cch, q_ * n_ch + cch, 1.0 / len(qs))
    L3 = LayerSpec("V4", "L3", "V4/L2", n_pos * n_ch, n_pos, n_ch, list(names2), [r_ + "_L3" for r_ in roles2], {"family": fam2},
                   np.zeros(n_pos * n_ch, bool), b3.finalize(), L2.x, L2.y, (px, py, prf + D), (nr4, nphi4), nb)
    return [L4, L2, L3]


# ---------------------------------------------------------------------- IT: 영역 풀링 (L4) → 전역·영역 특징군·특징군 쌍 min2 (L2) → 일대일 (L3)
def build_it(m: dict[str, Any], pre: RetinaLGN, v4: LayerSpec) -> list[LayerSpec]:
    a = m["areas"]["IT"]
    rad6 = int(m["l6"]["neighbor_grid_radius"])
    nr4, nphi4 = v4.grid_shape
    rr, rp = int(a["region_r"]), int(a["region_phi"])
    children = _block_children(nr4, nphi4, rr, rp)
    px, py, prf = _pool_positions(v4, children)
    n_reg, n_in = rr * rp, v4.n_ch
    b4 = CSR(n_reg * n_in, v4.n)
    for R, ch in enumerate(children):
        for cch in range(n_in):
            for c in ch:
                b4.add(R * n_in + cch, c * n_in + cch, 1.0 / len(ch))
    L4 = LayerSpec("IT", "L4", "V4/L3", v4.n, n_reg, n_in, list(v4.channel_names), ["IT_L4_region_pooled_" + r for r in v4.channel_roles],
                   dict(v4.channel_meta), np.zeros(n_reg * n_in, bool), b4.finalize(), _per(px, n_in), _per(py, n_in), (px, py, prf),
                   (rr, rp), grid_neighbors(rr, rp, rad6))
    fam = v4.channel_meta["family"]
    groups = {g: [k for k in range(n_in) if fam[k] == g] for g in ("curve", "straight", "junction", "lineend")}
    groups = {g: v for g, v in groups.items() if v}
    gnames = list(groups)
    gpairs = [(gnames[i], gnames[j]) for i in range(len(gnames)) for j in range(i + 1, len(gnames))]
    neurons: list[tuple[str, Any]] = ([("global", c) for c in range(n_in)] + [("rgroup", (R, g)) for R in range(n_reg) for g in gnames]
                                      + [("conj", (R, g1, g2)) for R in range(n_reg) for (g1, g2) in gpairs])
    n = len(neurons)
    b2 = CSR(n, n_reg * n_in)
    op = np.zeros(n, bool)
    xs, ys, names, roles, fams = [], [], [], [], []
    for j, (f, info) in enumerate(neurons):
        if f == "global":
            for R in range(n_reg):
                b2.add(j, R * n_in + info, 1.0 / n_reg)
            xs.append(pre.cx), ys.append(pre.cy)
            names.append("global_" + v4.channel_names[info])
            roles.append("IT_global_feature")
        elif f == "rgroup":
            R, g = info
            for k in groups[g]:
                b2.add(j, R * n_in + k, 1.0 / len(groups[g]))
            xs.append(px[R]), ys.append(py[R])
            names.append(f"region{R}_{g}")
            roles.append("IT_region_group")
        else:
            R, g1, g2 = info
            for k in groups[g1]:
                b2.add(j, R * n_in + k, 1.0 / len(groups[g1]), "A")
            for k in groups[g2]:
                b2.add(j, R * n_in + k, 1.0 / len(groups[g2]), "B")
            op[j] = True
            xs.append(px[R]), ys.append(py[R])
            names.append(f"region{R}_{g1}_and_{g2}")
            roles.append("IT_conjunction")
        fams.append(f)
    one = (np.asarray([pre.cx]), np.asarray([pre.cy]), np.asarray([pre.rmax]))
    L2 = LayerSpec("IT", "L2", "IT/L4", n_reg * n_in, 1, n, names, roles, {"family": fams}, op, b2.finalize(), np.asarray(xs), np.asarray(ys),
                   one, None, single_position_neighbors())
    L3 = LayerSpec("IT", "L3", "IT/L2", n, 1, n, list(names), [r_ + "_L3" for r_ in roles], {"family": fams}, np.zeros(n, bool),
                   identity_csr(n), L2.x, L2.y, one, None, single_position_neighbors())
    return [L4, L2, L3]


def build_structure(m: dict[str, Any], pre: RetinaLGN) -> dict[str, LayerSpec]:
    """12 층 전체 구조 (V4·IT 구조를 만들려면 V1·V2 의 위치 정보가 필요하다). start·end = v11 NeuronStore 와 같은 연속 번호."""
    specs: dict[str, LayerSpec] = {}
    for sp in build_v1(m, pre):
        specs[sp.key] = sp
    for sp in build_v2(m, pre, specs["V1/L3"]):
        specs[sp.key] = sp
    for sp in build_v4(m, pre, specs["V2/L3"]):
        specs[sp.key] = sp
    for sp in build_it(m, pre, specs["V4/L3"]):
        specs[sp.key] = sp
    N = 0
    for sp in specs.values():
        sp.start, sp.end = N, N + sp.n
        N = sp.end
    return specs


def struct_hashes(specs: dict[str, LayerSpec]) -> dict[str, str]:
    return {a: area_hash([specs[f"{a}/{l}"] for l in LAYERS]) for a in AREAS}


def count_learnable_edges(specs: dict[str, LayerSpec]) -> dict[str, int]:
    """학습 연결 = V4·IT 수신 층의 유효 연결 (w0 ≠ 0, 수신 뉴런이 구조상 유효). V2→V4 블록 평균 연결 포함."""
    out = {}
    for key in LEARN_KEYS:
        sp = specs[key]
        inv = sp.missing if sp.missing is not None else np.zeros(sp.n, bool)
        k = 0
        for ptr, w in ((sp.a_ptr, sp.a_w), (sp.b_ptr, sp.b_w)):
            row = np.repeat(np.arange(sp.n, dtype=np.int64), np.diff(ptr))
            k += int(((w != 0) & ~inv[row]).sum())
        out[key] = k
    return out


# ======================================================================
# 층 계산 공통 부품 (v11 BucketELL · RelativeActivityController 와 같은 묶음·같은 합산 순서)
# ======================================================================
class BucketELL:
    """CSR 입력 목록 → 팬인 크기별 ELL 묶음 (팬인을 2 의 거듭제곱으로 올림). 묶음 결과는 고정 역순열 gather 로 원래 순서에 놓는다."""

    def __init__(self, be: Backend, ptr: np.ndarray, src: np.ndarray, w: np.ndarray, n_src: int):
        n = len(ptr) - 1
        self.n, self.n_src, self.edges = n, int(n_src), int(len(src))
        self.parts: list[tuple[Any, Any]] = []
        self.inv: Any = None
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
        perm_ = np.concatenate(perm)
        if not np.array_equal(perm_, np.arange(n)):
            self.inv = be.i64(np.argsort(perm_))

    def apply(self, be: Backend, xe: Any) -> Any:
        if not self.parts:
            return be.zeros((int(xe.shape[0]), self.n))
        outs = [be.ell(xe, idx, w) for idx, w in self.parts]
        out = outs[0] if len(outs) == 1 else be.cat1(outs)
        return out if self.inv is None else be.take_cols(out, self.inv)


class L6:
    """L6 상대 활동 조절 (같은 층 CONF 스냅샷에서 병렬, 반복 없음):
       r_j = CONF_j / (mean_{k∈N(j)} CONF_k + ε),  g_j = clip(1 + κ(r_j − 1), g_min, g_max),  이웃 없음 → g = 1.
       N(j) = 같은 층에서 피질 격자 Chebyshev 거리 ≤ R 인 위치의 모든 채널 (기본 자기 자신 제외)."""

    def __init__(self, be: Backend, sp: LayerSpec, l6cfg: dict[str, Any]):
        self.be = be
        self.n_pos, self.n_ch = sp.n_pos, sp.n_ch
        self.include_self = bool(l6cfg["include_self"])
        self.eps, self.kappa = float(l6cfg["epsilon"]), float(l6cfg["kappa"])
        self.g_min, self.g_max = float(l6cfg["g_min"]), float(l6cfg["g_max"])
        cnt = np.repeat(sp.pos_nb_count * sp.n_ch, sp.n_ch) - (0 if self.include_self else 1)
        self.pos_nb = be.i64(sp.pos_nb)
        self.has_nb = be.bool_(cnt > 0)
        self.nb_div = be.f32(np.maximum(cnt, 1))

    @staticmethod
    def applies(sp: LayerSpec, l6cfg: dict[str, Any]) -> bool:
        return bool(l6cfg["enabled"]) and sp.name in ("L2", "L3") and sp.name in l6cfg["layers"]

    def gain(self, conf: Any) -> Any:
        be = self.be
        B = int(conf.shape[0])
        pos_sum = conf.reshape(B, self.n_pos, self.n_ch).sum(-1)
        nb_sum = be.take_cols(be.pad_zero_col(pos_sum), self.pos_nb).sum(-1)
        nb_n = be.repeat_cols(nb_sum, self.n_ch)
        if not self.include_self:
            nb_n = be.relu(nb_n - conf)
        r = conf / (nb_n / self.nb_div + self.eps)
        g = be.clip(1.0 + self.kappa * (r - 1.0), self.g_min, self.g_max)
        return be.where(self.has_nb, g, be.ones_like(g))


# ======================================================================
# 동결 줄기 (V1·V2) 계산: 준비 학습과 캐시 생성에만 쓴다. 주 학습 엔진은 이 객체를 만들지 않는다 (역방향 구조 없음)
# ======================================================================
class StemLayer:
    def __init__(self, be: Backend, sp: LayerSpec, model: dict[str, Any], q: np.ndarray, thr: np.ndarray):
        self.key, self.src_key, self.n = sp.key, sp.src_key, sp.n
        self.A = BucketELL(be, sp.a_ptr, sp.a_src, sp.a_w, sp.n_src)
        self.has_b = sp.b_src.size > 0
        self.Bm = BucketELL(be, sp.b_ptr, sp.b_src, sp.b_w, sp.n_src) if self.has_b else None
        self.op_min = be.bool_(sp.op_min) if self.has_b else None
        if sp.missing is not None and sp.missing.any():
            raise V12Error(f"{sp.key}: V1·V2 에는 구조상 비활성 뉴런이 없어야 한다.")
        self.l6 = L6(be, sp, model["l6"]) if L6.applies(sp, model["l6"]) else None
        self.q = be.f32(q)
        self.thr = be.f32(thr)


class Stem:
    """V1·V2 여섯 층 (L4 → L2 → L3). 층 계산: u = ΣA (min2 뉴런은 min(ΣA, ΣB)), s = max(0, u − θ), CONF = q·s, 출력 = L6 g·CONF."""

    def __init__(self, be: Backend, specs: dict[str, LayerSpec], model: dict[str, Any], q_full: np.ndarray, thr_full: np.ndarray):
        self.be = be
        self.layers: dict[str, StemLayer] = {}
        for a in STEM_AREAS:
            for l in LAYERS:
                sp = specs[f"{a}/{l}"]
                self.layers[sp.key] = StemLayer(be, sp, model, q_full[sp.start:sp.end], thr_full[sp.start:sp.end])
        self.order = list(self.layers)

    def step(self, L: StemLayer, prev: Any) -> dict[str, Any]:
        be = self.be
        xe = be.pad_zero_col(prev)
        sa = L.A.apply(be, xe)
        if L.has_b:
            sb = L.Bm.apply(be, xe)
            u = be.where(L.op_min, be.minimum(sa, sb), sa + sb)
        else:
            u = sa
        s = be.relu(u - L.thr)
        conf = L.q * s
        out = conf if L.l6 is None else L.l6.gain(conf) * conf
        return {"s": s, "out": out}

    def forward(self, x: Any, start: str, stop: str, keep: Sequence[str] = ()) -> dict[str, Any]:
        """start = 입력의 층 이름 ('LGN' 또는 'V1/L3'), stop = 마지막으로 계산할 층. keep 층은 s·out 을 돌려준다."""
        keys = self.order
        i0 = 0 if start == "LGN" else keys.index(start) + 1
        i1 = keys.index(stop) + 1
        res: dict[str, Any] = {}
        prev = x
        for key in keys[i0:i1]:
            L = self.layers[key]
            r = self.step(L, prev)
            if key in keep:
                res[key] = r
            prev = r["out"]
        res["out"] = prev
        return res

    def q_numpy(self, key: str) -> np.ndarray:
        return self.be.to_np(self.layers[key].q).astype(np.float32).copy()


# ======================================================================
# V1·V2 준비 학습 (새 준비 경로만). v11 StageTrainer 의 q 학습·prototype·종료 상태 정책과 같다
#   배치: 순방향 → δ = P_y − h (무효 prototype 표본은 0) → m_L3 = α3·δ, m_L2 = α2·Σ_{j∋i} δ_j/|P(j)| (L3 의 L2 부모로 배달)
#         → Δq = η·mean(e·m), e = s/(s + e0) (L2·L3 같은 스냅샷) → q ← clip(q + Δq, q_min, q_max) 동시 적용
#   epoch 끝: q 고정 train 통과 → 클래스 평균 h̄ → P ← P + 1[n_c>0]·ρ(h̄ − P) → dev 코사인 판독 (갱신 없음)
#         → 개선 (> best + min_delta) 이면 best, 아니면 patience 카운트 → plateau / budget_exhausted. 끝 상태 (마지막 epoch q) 를 쓴다.
#   v11 이 진단용으로 하던 두 번째 순방향·거리 기록은 상태에 영향이 없어 하지 않는다.
# ======================================================================
class ParentRouter:
    """L3 출력 j 의 L2 부모 집합 P(j) 로 교정을 배달 (v11 LocalErrorRouter: 주소만 읽는다, 가중치·전치·미분 없음)."""

    def __init__(self, be: Backend, sp2: LayerSpec, sp3: LayerSpec, alpha2: float, alpha3: float):
        self.be, self.alpha2, self.alpha3 = be, float(alpha2), float(alpha3)
        n2, n3 = sp2.n, sp3.n
        rows = np.concatenate([np.repeat(np.arange(n3, dtype=np.int64), np.diff(sp3.a_ptr)),
                               np.repeat(np.arange(n3, dtype=np.int64), np.diff(sp3.b_ptr))])
        src = np.concatenate([sp3.a_src, sp3.b_src])
        pairs = np.unique(rows * np.int64(n2) + src)
        prow, psrc = pairs // n2, pairs % n2
        cnt = np.bincount(prow, minlength=n3)
        if (cnt == 0).any():
            raise V12Error(f"{sp3.key}: 부모 없는 L3 출력이 있다.")
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
        self.ch_idx, self.ch_coef = be.i64(ch_idx), be.f32(ch_coef)

    def route(self, delta: Any) -> tuple[Any, Any]:
        be = self.be
        return self.alpha2 * (be.take_cols(be.pad_zero_col(delta), self.ch_idx) * self.ch_coef).sum(-1), self.alpha3 * delta


def epoch_permutation_prep(seed: int, area: str, epoch: int, n: int) -> np.ndarray:
    """준비 학습 train 순서 (v11 epoch_permutation): default_rng([seed, 영역번호, epoch(0부터), 11])."""
    return np.random.default_rng([int(seed), AREA_INDEX[area], int(epoch), 11]).permutation(int(n))


def cosine_predict(be: Backend, h: Any, P: Any, valid_P: np.ndarray, eps: float) -> np.ndarray:
    """pred = argmax_c cos(h, P_c) (v11 cosine_classify). ‖h‖ ≤ eps → −1 (unknown, 오답). 무효 prototype 은 후보에서 뺀다."""
    hn = be.to_np(be.norm_last(h)).astype(np.float64)
    pn = be.to_np(be.norm_last(P)).astype(np.float64)
    dots = be.to_np(h @ P.T).astype(np.float64)
    B, C = dots.shape
    valid_h = hn > eps
    cv = np.asarray(valid_P, bool) & (pn > eps)
    cos = np.full((B, C), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        if cv.any():
            cos[:, cv] = dots[:, cv] / (hn[:, None] * pn[None, cv])
    cos[~valid_h] = np.nan
    pred = np.full(B, -1, np.int64)
    for i in range(B):
        if valid_h[i] and cv.any():
            pred[i] = int(np.nanargmax(cos[i]))
    return pred


# ======================================================================
# V4·IT 학습 엔진 (v11 FbELL · FbLayer · FbNet 의 F4 TRACE 경로와 같은 묶음·같은 연산 순서)
#   층 6 개: V4 L4 (V2→V4 블록 평균, 입력 = 캐시된 V2 L3) → V4 L2 → V4 L3 → IT L4 → IT L2 → IT L3 (146 차원)
#   실제 가중치 w = w0·γ (주소·w0 부호·연결 유무 고정, γ 만 학습). γ 는 잔차 res = γ − 1 (float32) 로 누적한다
#   (1 근처 float32 간격보다 작은 수정도 사라지지 않게). 순방향은 gam = float32(1 + res).
#   역방향 (TRACE): v_e = w0·γ·x, score = |v|, P_e = score/Σ_수신 score (0 이면 0), κ = q·g·1[s>0]·유효, ℓ = κ·d,
#   d_src += sign(w0)·P_e·ℓ. gain 갱신량 Σ_batch ℓ·λ·x 를 연결 칸마다 모은다 (λ: 합 1, min 은 작은 슬롯 1·다른 슬롯 0, 동점 0.5).
#   V4 L4 에서는 갱신량만 계산하고 끝낸다 (V2 로 보낼 d_src·그 정규화·전치 gather 를 만들지 않는다).
# ======================================================================
class GainELL:
    """입력군 하나 (A 또는 B) 의 연결 묶음과 연결별 gain (BucketELL 과 같은 묶음·열 순서)."""

    def __init__(self, be: Backend, ptr: np.ndarray, src: np.ndarray, w: np.ndarray, n_src: int):
        n = len(ptr) - 1
        self.n, self.n_src, self.edges = n, int(n_src), int(len(src))
        self.parts: list[dict[str, Any]] = []
        self.inv: Any = None
        self.part_of_row = np.full(n, -1, np.int64)
        self.local_of_row = np.full(n, -1, np.int64)
        if self.edges == 0:
            return
        fan = np.diff(ptr)
        K = np.ones(n, np.int64)
        nz = fan > 0
        K[nz] = np.left_shift(1, np.ceil(np.log2(fan[nz])).astype(np.int64))
        row_of_edge = np.repeat(np.arange(n, dtype=np.int64), fan)
        col = np.arange(len(src), dtype=np.int64) - ptr[row_of_edge]
        perm = []
        for pi, k in enumerate(np.unique(K)):
            rows = np.nonzero(K == k)[0]
            self.part_of_row[rows] = pi
            self.local_of_row[rows] = np.arange(len(rows))
            idx = np.full((len(rows), int(k)), self.n_src, np.int64)
            ww = np.zeros((len(rows), int(k)), np.float32)
            em = K[row_of_edge] == k
            idx[self.local_of_row[row_of_edge[em]], col[em]] = src[em]
            ww[self.local_of_row[row_of_edge[em]], col[em]] = w[em]
            w0 = be.f32(ww)
            self.parts.append({"rows": rows, "rows_be": be.i64(rows), "r": int(len(rows)), "K": int(k), "idx": be.i64(idx), "w0": w0,
                               "res": be.f32(np.zeros((len(rows), int(k)), np.float32)),
                               "gam": be.f32(np.ones((len(rows), int(k)), np.float32)), "weff": w0, "dg": None, "off": 0})
            perm.append(rows)
        perm_ = np.concatenate(perm)
        if not np.array_equal(perm_, np.arange(n)):
            self.inv = be.i64(np.argsort(perm_))

    def assemble(self, be: Backend, outs: Sequence[Any], B: int) -> Any:
        if not self.parts:
            return be.zeros((int(B), self.n))
        out = outs[0] if len(outs) == 1 else be.cat1(list(outs))
        return out if self.inv is None else be.take_cols(out, self.inv)

    def apply(self, be: Backend, xe: Any) -> Any:
        if not self.parts:
            return be.zeros((int(xe.shape[0]), self.n))
        return self.assemble(be, [be.ell(xe, p["idx"], p["weff"]) for p in self.parts], int(xe.shape[0]))


class GainLayer:
    def __init__(self, be: Backend, sp: LayerSpec, model: dict[str, Any], q: np.ndarray, thr: np.ndarray, *, first: bool):
        self.key, self.src_key, self.n, self.n_src = sp.key, sp.src_key, sp.n, sp.n_src
        self.first = first
        self.A = GainELL(be, sp.a_ptr, sp.a_src, sp.a_w, sp.n_src)
        self.B = GainELL(be, sp.b_ptr, sp.b_src, sp.b_w, sp.n_src) if len(sp.b_src) else None
        self.op_min = be.bool_(sp.op_min) if self.B is not None else None
        inv = np.zeros(self.n, bool) if sp.missing is None else np.asarray(sp.missing, bool).copy()
        self.invalid_np = inv
        self.invalid = be.bool_(inv) if inv.any() else None
        self.valid_f = be.f32((~inv).astype(np.float32)) if inv.any() else None
        self.q, self.thr = be.f32(q), be.f32(thr)
        self.l6 = L6(be, sp, model["l6"]) if L6.applies(sp, model["l6"]) else None
        self.F = 0
        nvalid = np.zeros(self.n, np.int64)
        pos_l, src_l, sgn_l = [], [], []
        for ELL, ptr, src, w in ((self.A, sp.a_ptr, sp.a_src, sp.a_w), (self.B, sp.b_ptr, sp.b_src, sp.b_w)):
            if ELL is None or not ELL.parts:
                continue
            for p in ELL.parts:
                p["off"] = self.F
                self.F += p["r"] * p["K"]
            row = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(ptr))
            col = np.arange(len(src), dtype=np.int64) - ptr[row]
            pofs = np.asarray([p["off"] for p in ELL.parts], np.int64)
            pK = np.asarray([p["K"] for p in ELL.parts], np.int64)
            pr = ELL.part_of_row[row]
            flat = pofs[pr] + ELL.local_of_row[row] * pK[pr] + col
            ok = (w != 0) & ~inv[row]
            nvalid += np.bincount(row[ok], minlength=self.n)
            pos_l.append(flat[ok])
            src_l.append(np.asarray(src, np.int64)[ok])
            sgn_l.append(np.sign(w[ok]).astype(np.float32))
        self.n_valid_edges = int(nvalid.sum())
        self.n_edges = int(len(sp.a_src) + len(sp.b_src))
        self.T: BucketELL | None = None
        if not first:                       # 전치 묶음 (원천 뉴런 → 자기가 보낸 연결 칸, 가중치 sign(w0)). 첫 층 (V4 L4) 은 만들지 않는다
            pos = np.concatenate(pos_l) if pos_l else np.zeros(0, np.int64)
            s_ = np.concatenate(src_l) if src_l else np.zeros(0, np.int64)
            g_ = np.concatenate(sgn_l) if sgn_l else np.zeros(0, np.float32)
            order = np.lexsort((pos, s_))
            t_ptr = np.concatenate([[0], np.cumsum(np.bincount(s_, minlength=self.n_src))]).astype(np.int64)
            self.T = BucketELL(be, t_ptr, pos[order], g_[order], self.F)
        self.t_calls = 0

    def groups(self) -> list[tuple[str, GainELL]]:
        return [("A", self.A)] + ([("B", self.B)] if self.B is not None else [])

    def parts(self) -> Iterator[tuple[str, int, dict[str, Any]]]:
        for nm, ELL in self.groups():
            for i, p in enumerate(ELL.parts):
                yield nm, i, p


class Engine:
    def __init__(self, be: Backend, specs: dict[str, LayerSpec], model: dict[str, Any], q_full: np.ndarray, thr_full: np.ndarray,
                 *, gain_min: float, gain_max: float, memory_mb: int):
        self.be = be
        self.layers = [GainLayer(be, specs[k], model, q_full[specs[k].start:specs[k].end], thr_full[specs[k].start:specs[k].end],
                                 first=(i == 0)) for i, k in enumerate(LEARN_KEYS)]
        self.by_key = {L.key: L for L in self.layers}
        self.in_dim = self.layers[0].n_src
        self.dim = self.layers[-1].n
        self.lo, self.hi = float(gain_min) - 1.0, float(gain_max) - 1.0
        self.elem_budget = max(1 << 16, int(int(memory_mb) * 2 ** 20 / 16))   # v11 FbNet 과 같은 표본 조각 상한

    # ---------------------------------------------------------------- 순방향
    def forward(self, x0: Any, *, record: bool) -> dict[str, Any]:
        be = self.be
        lay: dict[str, dict[str, Any]] = {}
        prev = x0
        for L in self.layers:
            xe = be.pad_zero_col(prev)
            sa = L.A.apply(be, xe)
            sb = None
            if L.B is not None:
                sb = L.B.apply(be, xe)
                u = be.where(L.op_min, be.minimum(sa, sb), sa + sb)
                if L.invalid is not None:
                    u = be.where(L.invalid, be.zeros(u.shape), u)
            else:
                u = sa
            s = be.relu(u - L.thr)
            if L.invalid is not None:
                s = be.where(L.invalid, be.zeros(s.shape), s)
            conf = L.q * s
            g = L.l6.gain(conf) if L.l6 is not None else None
            out = conf if g is None else g * conf
            if record:
                lay[L.key] = {"sa": sa, "sb": sb, "s": s, "g": g, "out": out}
            prev = out
        return {"x0": x0, "layers": lay, "h": prev}

    # ---------------------------------------------------------------- 역방향 (TRACE 배분 + gain 갱신량 누적)
    def _lambdas(self, L: GainLayer, r: dict[str, Any]) -> tuple[Any, Any]:
        if L.B is None:
            return None, None
        be = self.be
        sa, sb = r["sa"], r["sb"]
        one, zero = be.ones_like(sa), be.zeros_like(sa)
        half = one * 0.5
        la = be.where(sa < sb, one, be.where(sb < sa, zero, half))
        lb = be.where(sb < sa, one, be.where(sa < sb, zero, half))
        return be.where(L.op_min, la, one), be.where(L.op_min, lb, one)

    def _kappa(self, L: GainLayer, r: dict[str, Any]) -> Any:
        act = self.be.as_f32(r["s"] > 0)
        k = L.q * act if r["g"] is None else L.q * r["g"] * act
        return k if L.valid_f is None else k * L.valid_f

    def route_back(self, rec: dict[str, Any], d_top: Any) -> None:
        """d_top: IT L3 출력에 도착한 교정 (B, 146). IT L3 → … → V4 L4. 각 층 ℓ = κ·d 로 gain 갱신량을 모으고, 앞 층으로 d_src 를 보낸다.
        V4 L4 에서는 갱신량만 모으고 끝낸다 (V2 로 가는 교정은 만들지 않는다)."""
        d = d_top
        for li in range(len(self.layers) - 1, -1, -1):
            L = self.layers[li]
            r = rec["layers"][L.key]
            x = rec["x0"] if li == 0 else rec["layers"][self.layers[li - 1].key]["out"]
            ell = self._kappa(L, r) * d
            lamA, lamB = self._lambdas(L, r)
            d = self._route_layer(L, x, ell, lamA, lamB, need_src=(li > 0))

    def _route_layer(self, L: GainLayer, x: Any, ell: Any, lamA: Any, lamB: Any, need_src: bool) -> Any:
        be = self.be
        B = int(x.shape[0])
        xe = be.pad_zero_col(x)
        Bc = max(1, min(B, int(self.elem_budget // max(1, L.F))))
        groups = [(L.A, lamA)] + ([(L.B, lamB)] if L.B is not None else [])
        chunks = []
        for b0 in range(0, B, Bc):
            b1 = min(B, b0 + Bc)
            nb = b1 - b0
            xc, ec = xe[b0:b1], ell[b0:b1]
            scores: list[Any] = []
            S = None
            for ELL, lam in groups:
                lc = lam[b0:b1] if lam is not None else None
                outs = []
                for p in ELL.parts:
                    xg = xc[:, p["idx"]]                                       # (nb, r, K) 원천 출력 x (채움 칸 = 0)
                    er = be.take_cols(ec, p["rows_be"])
                    if lc is not None:
                        er = er * be.take_cols(lc, p["rows_be"])
                    acc = (er[:, :, None] * xg).sum(0)                         # Σ_batch ℓ_dst·λ_e·x_src
                    p["dg"] = acc if p["dg"] is None else p["dg"] + acc
                    if need_src:
                        sc = abs(xg * p["weff"])                               # score_e = |w0·γ·x| (raw TRACE)
                        outs.append(sc.sum(-1))
                        scores.append(sc)
                if need_src:
                    Sg = ELL.assemble(be, outs, nb)
                    S = Sg if S is None else S + Sg
            if not need_src:
                continue
            if L.valid_f is not None:
                S = S * L.valid_f
            pos = S > 0
            s_safe = be.where(pos, S, be.ones_like(S))                         # Σ score = 0 → 배분 0 (ℓ 도 0 으로 둔다)
            fac = be.where(pos, ec, be.zeros_like(S))
            cflat = be.zeros((nb, L.F + 1))
            k = 0
            for ELL, _lam in groups:
                for p in ELL.parts:
                    fr = be.take_cols(fac, p["rows_be"])[:, :, None]
                    c = (scores[k] / be.take_cols(s_safe, p["rows_be"])[:, :, None]) * fr      # P_e = score_e/S_j 를 먼저, 그다음 ℓ_j
                    cflat[:, p["off"]:p["off"] + p["r"] * p["K"]] = c.reshape(nb, -1)
                    scores[k] = None
                    k += 1
            L.t_calls += 1
            chunks.append(L.T.apply(be, cflat))                                # d_src = Σ sign(w0)·P_e·ℓ (원천별 고정 주소 합)
        if not need_src:
            return None
        return chunks[0] if len(chunks) == 1 else be.cat0(chunks)

    # ---------------------------------------------------------------- gain 갱신 (계산 → 배치 끝에 함께 적용)
    def clear_grads(self) -> None:
        for L in self.layers:
            for _nm, _i, p in L.parts():
                p["dg"] = None

    def proposals(self, eta_edge: float, n_batch: int) -> list[tuple[dict[str, Any], Any]]:
        """Δγ = η_edge·w0·(Σ_batch ℓ·λ·x)/배치 수, res_new = clip(res + Δγ, γ_min − 1, γ_max − 1). NaN/Inf 가 있으면 적용하지 않는다."""
        be = self.be
        todo: list[tuple[dict[str, Any], Any]] = []
        raws: list[Any] = []
        try:
            for L in self.layers:
                for _nm, _i, p in L.parts():
                    if p["dg"] is None:
                        continue
                    delta = p["w0"] * p["dg"] * (float(eta_edge) / float(n_batch))
                    raw = p["res"] + delta
                    raws.append(raw)
                    todo.append((p, be.clip(raw, self.lo, self.hi)))
            if not be.all_finite_many(raws):
                raise NumericalError("연결 gain 갱신값에 NaN/Inf 가 있다 (적용하지 않고 멈춘다).")
        finally:
            self.clear_grads()
        return todo

    @staticmethod
    def _set(p: dict[str, Any], res: Any) -> None:
        p["res"] = res
        p["gam"] = res + 1.0
        p["weff"] = p["w0"] * p["gam"]

    def commit(self, todo: Sequence[tuple[dict[str, Any], Any]]) -> None:
        for p, new in todo:
            self._set(p, new)

    # ---------------------------------------------------------------- 상태 (gain 잔차를 묶음 배치 그대로)
    def gain_arrays(self, prefix: str = "g") -> dict[str, np.ndarray]:
        return {f"{prefix}.{L.key.replace('/', '.')}.{nm}{i}": self.be.to_np(p["res"]).astype(np.float32).copy()
                for L in self.layers for nm, i, p in L.parts()}

    def load_gain_arrays(self, arrays: dict[str, np.ndarray], prefix: str = "g") -> None:
        for L in self.layers:
            for nm, i, p in L.parts():
                k = f"{prefix}.{L.key.replace('/', '.')}.{nm}{i}"
                if k not in arrays:
                    raise IntegrityError(f"저장 상태에 gain 배열 {k} 가 없다 (다른 구조의 상태다).")
                a = np.asarray(arrays[k], np.float32)
                if a.shape != (p["r"], p["K"]) or not np.isfinite(a).all() or a.min() < self.lo - 1e-6 or a.max() > self.hi + 1e-6:
                    raise IntegrityError(f"gain 배열 {k} 의 모양·값이 올바르지 않다.")
                self._set(p, self.be.f32(a))

    def gains_sha(self) -> str:
        h = hashlib.sha256()
        for L in self.layers:
            for _nm, _i, p in L.parts():
                h.update(np.ascontiguousarray(self.be.to_np(p["res"]).astype(np.float32)).tobytes())
        return h.hexdigest()

    def gain_summary(self) -> dict[str, Any]:
        """층별 유효 연결 gain 분포 요약 (epoch 끝에만 계산)."""
        out: dict[str, Any] = {}
        for L in self.layers:
            gs = []
            for _nm, _i, p in L.parts():
                w0 = self.be.to_np(p["w0"])
                msk = (w0 != 0) & ~L.invalid_np[p["rows"]][:, None]
                gs.append(1.0 + self.be.to_np(p["res"])[msk].astype(np.float64))
            g = np.concatenate(gs) if gs else np.zeros(0)
            out[L.key] = {"n": int(len(g)), "mean": float(g.mean()) if len(g) else None, "min": float(g.min()) if len(g) else None,
                          "max": float(g.max()) if len(g) else None, "n_changed": int((g != 1.0).sum()),
                          "n_at_min": int((g <= self.lo + 1.0 + 1e-7).sum()), "n_at_max": int((g >= self.hi + 1.0 - 1e-7).sum())}
        return out

    def structure(self) -> list[dict[str, Any]]:
        return [{"layer": L.key, "n": L.n, "n_src": L.n_src, "edges": L.n_edges, "learnable_edges": L.n_valid_edges, "ell_slots": L.F,
                 "inactive_neurons": int(L.invalid_np.sum()), "l6": L.l6 is not None, "transposed_backward": L.T is not None}
                for L in self.layers]


# ======================================================================
# 10 클래스 출력층 (float64, NumPy CPU): z = W h + b, p = softmax(z), a = B·p, L = −log p_y, d_z = t − p
# ======================================================================
def head_init(seed: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """v11 fb_head_init: W ~ N(0, (0.01/√dim)²) (default_rng([seed, 9090, dim])), b = 0."""
    rng = np.random.default_rng([int(seed), 9090, int(dim)])
    W = rng.standard_normal((N_CLASSES, int(dim))) * (0.01 / math.sqrt(int(dim)))
    return W.astype(np.float64), np.zeros(N_CLASSES, np.float64)


def head_forward(W: np.ndarray, b: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore", over="ignore"):
        z = h @ W.T + b
    if not np.isfinite(z).all():
        raise NumericalError("출력층 logit 에 NaN/Inf 가 있다 (피질 특징 또는 W·b 가 비유한).")
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    se = e.sum(axis=1, keepdims=True)
    return z, e / se, (z - m) - np.log(se)


def head_route_trace(W: np.ndarray, h: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """출력층 → IT 교정: 클래스 k 로 들어온 연결 i 에서 score = |W[k,i]·h_i|, P = score/Σ_i score (0 이면 0), d_h_i = Σ_k sign(W[k,i])·P_ki·d_z_k.
    bias 교정은 피질로 보내지 않는다. 갱신 전 W 를 쓴다."""
    v = W[None, :, :] * h[:, None, :]
    score = np.abs(v)
    S = score.sum(axis=2, keepdims=True)
    P = np.divide(score, S, out=np.zeros_like(score), where=S > 0)
    R = np.sign(W)[None, :, :] * P
    return (R * dz[:, :, None]).sum(axis=1)


class EvalAcc:
    """분할 평가 누적 (v11 FbEvalAcc 와 같은 합산 순서: 배치 합을 float64 로 더한다)."""

    def __init__(self) -> None:
        self.n = self.correct = 0
        self.ce = 0.0
        self.conf = np.zeros((N_CLASSES, N_CLASSES), np.int64)

    def add(self, p: np.ndarray, logp: np.ndarray, y: np.ndarray) -> None:
        i = np.arange(len(y))
        pred = p.argmax(axis=1)
        self.n += len(y)
        self.correct += int((pred == y).sum())
        self.ce += float((-logp[i, y]).sum())
        np.add.at(self.conf, (y, pred), 1)

    def result(self) -> dict[str, Any]:
        n = max(1, self.n)
        return {"n": self.n, "accuracy": self.correct / n if self.n else None, "correct": self.correct, "ce": self.ce / n,
                "confusion": self.conf.tolist()}


def epoch_permutation_main(seed: int, epoch: int, n: int) -> np.ndarray:
    """주 학습 train 순서 (v11 fb_epoch_permutation): default_rng([seed, 1010, epoch(1부터), 11])."""
    return np.random.default_rng([int(seed), 1010, int(epoch), 11]).permutation(int(n))


# ======================================================================
# v11 공통 초기 snapshot 검증 (읽기만 한다. 원본 run 의 어떤 파일도 고치지 않는다)
#   받는 것: v11 장기 비교 run (manifest.json) 의 seeds[seed].init_file = init_snapshot/seed_<seed>_CORTEX_FIXED_init.npz
#   (q · threshold · W0 · b0). 학습이 끝난 F4·all_regions 단위 상태 (units/…/checkpoints) 는 읽지 않는다.
# ======================================================================
STEM_CONFIG_KEYS = ("image_size", "transparent_background", "retina", "logpolar", "neuron", "areas", "l6")


def stem_config_view(model: dict[str, Any]) -> dict[str, Any]:
    """전처리·구조·L6 설정 (v11 config.json 의 같은 항목과 비교하는 모양)."""
    return json.loads(dumps({"image_size": model["image_size"], "transparent_background": model["transparent_background"],
                             "retina": model["retina"], "logpolar": model["logpolar"], "neuron": model["neuron"], "areas": model["areas"],
                             "l6": model["l6"], "v4_fixed": model["v4_fixed"], "curve_path_points": model["curve_path_points"]}, indent=None))


def resolve_v11_run(raw: Any) -> Path:
    p = clean_user_path(raw)
    if p.is_file() and p.name == "manifest.json":
        p = p.parent
    elif p.is_file() and p.suffix == ".npz" and p.parent.name == "init_snapshot":
        p = p.parent.parent
    elif p.is_dir() and p.name == "init_snapshot":
        p = p.parent
    if not (p / "manifest.json").is_file():
        raise V12Error(f"{p} 에 manifest.json 이 없다. v11 장기 비교 run 폴더 (예: v11long_<시각>_<번호>) 를 준다 "
                       "(그 안의 init_snapshot/ 에 seed_<seed>_CORTEX_FIXED_init.npz 가 있다).")
    return p


def load_v11_snapshot(raw: Any, seed: int, settings: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """manifest·config·분할 설정·파일 해시를 확인하고 q·threshold·W0·b0 를 읽는다. 구조·분할·q 기록 대조는 verify_v11_snapshot 에서 한다."""
    root = resolve_v11_run(raw)
    mp = root / "manifest.json"
    man = read_json(mp)
    probs = []
    if man.get("format") != V11_MANIFEST_FORMAT:
        raise V12Error(f"{root}: 지원하지 않는 run 형식 {man.get('format')!r}. v11 장기 비교 run ({V11_MANIFEST_FORMAT}) 의 공통 초기 snapshot 만 받는다.")
    if (man.get("program"), man.get("program_version")) != (V11_PROGRAM, V11_VERSION):
        raise V12Error(f"{root}: 원본 프로그램 {man.get('program')} v{man.get('program_version')} 은 지원하지 않는다 ({V11_PROGRAM} v{V11_VERSION} 만).")
    if man.get("source_sha256") != V11_SOURCE_SHA256:
        raise V12Error(f"{root}: 원본 소스 SHA256 {str(man.get('source_sha256'))[:16]}… 이 지원하는 v11 ({V11_SOURCE_SHA256[:16]}…) 과 다르다 (수정된 v11).")
    plan = man.get("plan") or {}
    if plan.get("network") != V11_NETWORK:
        probs.append(f"망 {plan.get('network')} ≠ {V11_NETWORK}")
    if plan.get("data_source") != "mnist":
        probs.append(f"데이터 {plan.get('data_source')} ≠ mnist")
    lim_src = {k: int((plan.get("limits") or {}).get(k, 0) or 0) for k in SPLITS}
    lim_dst = {k: int(settings["data"]["limits"].get(k, 0) or 0) for k in SPLITS}
    if lim_src != lim_dst:
        probs.append(f"표본 한도 {lim_src} ≠ 이 실행 {lim_dst} (분할이 달라진다)")
    if plan.get("cfg_override") or plan.get("config_path"):
        probs.append("원본 run 이 설정 덮어쓰기 (cfg_override / config_path) 를 썼다 — 이 프로그램은 MODEL 설정이 같을 때만 받는다")
    if probs:
        raise V12Error(f"{root}: 공통 초기 snapshot 을 쓸 수 없다 — " + "; ".join(probs))
    seeds = man.get("seeds") or {}
    sr = seeds.get(str(int(seed)))
    if not sr or not sr.get("init_file"):
        have = sorted(k for k, v in seeds.items() if (v or {}).get("init_file"))
        raise V12Error(f"{root} 에 seed {seed} 의 공통 초기 snapshot 이 없다 (있는 seed: {have or '없음'}). seed 를 바꾸거나 비워서 새로 준비한다.")
    ip = root / sr["init_file"]
    if ip.name != f"seed_{int(seed)}_{V11_NETWORK}_init.npz" or ip.parent.name != "init_snapshot":
        raise V12Error(f"seed {seed} 의 초기 파일 {sr['init_file']} 이 공통 초기 snapshot 위치·이름이 아니다 (학습 결과 상태는 받지 않는다).")
    if not ip.is_file():
        raise V12Error(f"공통 초기 snapshot 파일이 없다: {ip}")
    fsha = sha256_file(ip)
    if fsha != sr.get("init_file_sha256"):
        raise IntegrityError(f"{ip.name} 의 SHA256 {fsha[:16]}… 이 manifest 기록 {str(sr.get('init_file_sha256'))[:16]}… 과 다르다 (바뀐 파일).")
    arrays, _meta = load_npz(ip)
    if set(arrays) != {"q", "threshold", "W0", "b0"}:
        raise V12Error(f"{ip.name}: 배열 {sorted(arrays)} — 공통 초기 snapshot 형식 (q, threshold, W0, b0) 이 아니다.")
    # 원본 V1·V2 준비 run 의 설정 (base snapshot 의 config.json · split_manifest.json, 해시 확인)
    b = sr.get("base_snapshot") or {}
    pin = b.get("pin") or {}
    files = pin.get("files") or {}
    bdir = root / str(b.get("dir") or "_missing_")
    for rel in ("config.json", "split_manifest.json"):
        if rel not in files:
            raise V12Error(f"seed {seed}: 원본 run 에 V1·V2 준비 설정 기록 ({rel}) 이 없다 (구조·전처리를 확인할 수 없다).")
        fp = bdir / rel
        if not fp.is_file() or sha256_file(fp) != files[rel]:
            raise IntegrityError(f"seed {seed}: 원본 run 의 {fp} 가 없거나 기록된 해시와 다르다.")
    cfg_src = (read_json(bdir / "config.json") or {}).get("config") or {}
    smf = read_json(bdir / "split_manifest.json")
    want = stem_config_view(model)
    have = {"image_size": (cfg_src.get("data") or {}).get("image_size"),
            "transparent_background": (cfg_src.get("data") or {}).get("transparent_background"),
            **{k: cfg_src.get(k) for k in ("retina", "logpolar", "neuron", "areas", "l6", "v4_fixed")},
            "curve_path_points": ((cfg_src.get("routing") or {}).get("ratio_codebook") or {}).get("curve_path_points")}
    diff = [k for k in want if json.loads(dumps(have.get(k), indent=None)) != want[k]]
    if int(cfg_src.get("seed", -1)) != int(seed):
        diff.append(f"seed {cfg_src.get('seed')}")
    if (cfg_src.get("data") or {}).get("source") != "mnist" or int((cfg_src.get("data") or {}).get("mnist_dev_size", -1)) != int(settings["data"]["mnist_dev_size"]):
        diff.append("data.source / mnist_dev_size")
    if (cfg_src.get("compute") or {}).get("dtype") != "float32":
        diff.append("compute.dtype")
    if diff:
        raise V12Error(f"seed {seed}: 원본 V1·V2 준비 설정이 이 프로그램의 MODEL·분할 설정과 다르다 ({', '.join(map(str, diff))}). "
                       "다른 구조·전처리의 상태를 대신 쓰지 않는다.")
    return {"root": root, "record": sr, "arrays": arrays, "split_manifest_sha256": smf.get("split_sha256"),
            "source": {"kind": "v11_common_init_snapshot", "run_dir": str(root), "run_name": root.name, "seed": int(seed),
                       "program": man.get("program"), "program_version": man.get("program_version"), "source_sha256": man.get("source_sha256"),
                       "manifest_sha256": sha256_file(mp), "init_file": sr["init_file"], "init_file_sha256": fsha,
                       "base_config_sha256": files["config.json"], "base_split_manifest_sha256": files["split_manifest.json"],
                       "source_prep": plan.get("prep"), "source_batch_size": plan.get("batch_size"), "source_backend": man.get("backend"),
                       "imported_from": (sr.get("import") or {}).get("run_name")}}


def verify_v11_snapshot(snap: dict[str, Any], seed: int, data: Mnist, specs: dict[str, LayerSpec], model: dict[str, Any]) -> dict[str, Any]:
    """다시 만든 분할·구조로 원본 기록 (split_sha256 · struct_hash · feature_dim · base_q_sha256 · head_init_sha256 · init_state_sha256) 을 대조하고
    V4·IT q = 1, 임계값 = 설정값을 확인한다. 하나라도 다르면 멈춘다 (조용히 대신 쓰지 않는다)."""
    sr, a = snap["record"], snap["arrays"]
    N = max(sp.end for sp in specs.values())
    dim = specs["IT/L3"].n
    q, thr = np.asarray(a["q"]), np.asarray(a["threshold"])
    W0, b0 = np.asarray(a["W0"]), np.asarray(a["b0"])
    probs = []
    if q.dtype != np.float32 or thr.dtype != np.float32 or q.shape != (N,) or thr.shape != (N,):
        probs.append(f"q·threshold 모양·형 {q.shape}/{q.dtype}, {thr.shape}/{thr.dtype} ≠ ({N},) float32")
    if W0.dtype != np.float64 or b0.dtype != np.float64 or W0.shape != (N_CLASSES, dim) or b0.shape != (N_CLASSES,):
        probs.append(f"W0·b0 모양·형 {W0.shape}/{W0.dtype}, {b0.shape}/{b0.dtype} ≠ ({N_CLASSES},{dim}) float64")
    if probs:
        raise V12Error("공통 초기 snapshot 이 이 구조와 맞지 않는다: " + "; ".join(probs))
    ne = model["neuron"]
    if not (np.isfinite(q).all() and np.isfinite(thr).all() and np.isfinite(W0).all() and np.isfinite(b0).all()):
        probs.append("비유한 값")
    if q.min() < float(ne["q_min"]) - 1e-7 or q.max() > float(ne["q_max"]) + 1e-7:
        probs.append("q 범위 밖")
    for key in LEARN_KEYS:
        sp = specs[key]
        if not np.array_equal(q[sp.start:sp.end], np.ones(sp.n, np.float32)):
            probs.append(f"{key} 초기 q ≠ 1")
    if not np.array_equal(thr, np.full(N, float(ne["threshold"]), np.float32)):
        probs.append(f"임계값이 설정값 {ne['threshold']} 과 다르다")
    sh = struct_hashes(specs)
    base_q = {k: q[specs[k].start:specs[k].end] for k in (f"{a_}/{l}" for a_ in STEM_AREAS for l in LAYERS)}
    rec = {"split_sha256": data.split_sha256(), "struct_hash": sh, "feature_dim": dim,
           "base_q_sha256": sha256_array(*[base_q[k] for k in sorted(base_q)]), "head_init_sha256": sha256_array(W0, b0),
           "init_state_sha256": sha256_text(sha256_array(q, thr, W0, b0) + dumps(sh, indent=None) + "gain=1")}
    for k, v in rec.items():
        if json.loads(dumps(sr.get(k), indent=None)) != json.loads(dumps(v, indent=None)):
            probs.append(f"{k}: 원본 {str(sr.get(k))[:40]} ≠ 다시 만든 값 {str(v)[:40]}")
    if snap.get("split_manifest_sha256") != data._split_manifest_sha:
        probs.append("원본 split_manifest 의 분할 해시가 이 MNIST·seed 분할과 다르다")
    if probs:
        raise V12Error("공통 초기 snapshot 검증 실패 — " + "; ".join(probs) + ". 구조·전처리·분할·q·head 가 같은 상태만 쓴다.")
    W_rule, b_rule = head_init(seed, dim)
    return {"verified_utc": utc_now(), "fields_identical": sorted(rec), "v4_it_q_is_one": True, "threshold_is_config": True,
            "head_equals_seed_rule": bool(np.array_equal(W_rule, W0) and np.array_equal(b_rule, b0))}


# ======================================================================
# 실행 (run 폴더 하나 = seed 하나 · 학습 하나). phase: prepare_v1 → prepare_v2 → cache → train → evaluate → done
#   last_params.npz 에 배열과 상태 JSON 을 함께 원자적으로 저장한다 (이어하기의 유일한 기준). last_state.json 은 사람이 읽는 사본.
# ======================================================================
def chunk_bounds(n: int, size: int) -> list[tuple[int, int]]:
    """목록 순서의 연속 조각. 마지막 조각이 1 표본이면 앞 조각과 합친다 (1 표본 묶음의 float32 합산 차이를 피한다)."""
    size = max(2, int(size))
    b = list(range(0, int(n), size)) + [int(n)]
    pairs = list(zip(b[:-1], b[1:]))
    if len(pairs) >= 2 and pairs[-1][1] - pairs[-1][0] == 1:
        pairs = pairs[:-2] + [(pairs[-2][0], int(n))]
    return pairs


def settings_view(settings: dict[str, Any]) -> dict[str, Any]:
    return json.loads(dumps(settings, indent=None))


def run_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"settings_sha256": sha256_text(dumps(cfg["settings"], indent=None)), "model_sha256": sha256_text(dumps(cfg["model"], indent=None)),
            "stem_source_kind": cfg["stem_source"]["kind"], "seed": int(cfg["settings"]["seed"])}


class Run:
    def __init__(self, run_dir: Path, cfg: dict[str, Any], be: Backend, mnist_root: Path):
        self.dir, self.cfg, self.be = Path(run_dir), cfg, be
        self.S, self.M = cfg["settings"], cfg["model"]
        self.seed, self.B = int(self.S["seed"]), int(self.S["batch_size"])
        self.E = int(self.S["epochs"])
        self.mnist_root = Path(mnist_root)
        self.data: Mnist | None = None
        self.pre: RetinaLGN | None = None
        self.specs: dict[str, LayerSpec] | None = None
        self.state: dict[str, Any] = {}
        self.arrays: dict[str, np.ndarray] = {}
        self.book: dict[str, Any] = {}
        self.stop: StopFlag | None = None
        self._mark = time.perf_counter()
        self._sha_ok: set[str] = set()
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.best_arrays: dict[str, np.ndarray] = {}
        self.engine: Engine | None = None

    # ---------------------------------------------------------------- 데이터·구조
    def load_world(self) -> None:
        if self.data is not None:
            return
        d = Mnist(self.mnist_root)
        d.make_split(self.seed, int(self.S["data"]["mnist_dev_size"]))
        d.apply_limits(self.S["data"]["limits"], self.seed)
        self.data = d
        self.pre = RetinaLGN(self.M)
        self.specs = build_structure(self.M, self.pre)
        if self.specs["V2/L3"].n != self.specs["V4/L4"].n_src:
            raise V12Error("V2 L3 차원과 V4 L4 입력 차원이 다르다 (구조 오류).")

    def world_record(self) -> dict[str, Any]:
        d, sp = self.data, self.specs
        return {"split_sha256": d.split_sha256(), "split_manifest_sha256": d._split_manifest_sha,
                "order_sha256": {s: d.order_sha(s) for s in SPLITS}, "n": {s: int(len(d.split[s]["idx"])) for s in SPLITS},
                "struct_hash": struct_hashes(sp), "N": int(max(x.end for x in sp.values())), "cache_dim": int(sp["V2/L3"].n),
                "feature_dim": int(sp["IT/L3"].n), "learnable_edges": count_learnable_edges(sp)}

    def write_split_ids(self) -> None:
        d = self.data
        arrs = {}
        for s in SPLITS:
            for k in ("src", "idx", "y"):
                arrs[f"{s}.{k}"] = d.split[s][k]
        save_npz(self.dir / "split_ids.npz", arrs, {"seed": self.seed, "mnist_dev_size": self.S["data"]["mnist_dev_size"],
                                                     "limits": self.S["data"]["limits"], "limits_info": d.limits_info,
                                                     "split_sha256": d.split_sha256(), "order_sha256": {s: d.order_sha(s) for s in SPLITS},
                                                     "src_code": {"0": "MNIST 공식 train", "1": "MNIST 공식 test"},
                                                     "rule_ko": "train/dev = 공식 train 을 seed 별 층화 (v11 규칙), test = 공식 test 전체. 행 순서 = 캐시 행 순서."})

    # ---------------------------------------------------------------- 시간·저장
    def _tick(self, bucket: str) -> None:
        now = time.perf_counter()
        t = self.state.setdefault("time", {})
        t[bucket] = float(t.get(bucket, 0.0)) + (now - self._mark)
        self._mark = now

    def _skip_time(self) -> None:
        self._mark = time.perf_counter()

    def save(self, bucket: str | None = None) -> None:
        """현재 상태와 배열을 함께 저장한다 (Ctrl+C 를 미루는 구간)."""
        with self.stop.critical() if self.stop is not None else contextlib.nullcontext():
            if bucket:
                self.be.sync()
                self._tick(bucket)
            st = self.state
            st["seq"] = int(st.get("seq", 0)) + 1
            st["utc"] = utc_now()
            arrays = dict(self.arrays)
            if self.W is not None:
                arrays.update({"W": self.W, "b": self.b})
                arrays.update(self.engine.gain_arrays("g"))
                arrays.update({f"best.{k}": v for k, v in self.best_arrays.items()})
                st["params_sha256"] = self.params_sha()
            save_npz(self.dir / "last_params.npz", arrays, st)
            write_json(self.dir / "last_state.json", st)
            if self.best_arrays and st.get("train", {}).get("best") and st["train"]["best"].get("mirrored_seq") is None:
                b = st["train"]["best"]
                save_npz(self.dir / "best_params.npz", self.best_arrays, {"format": "cortex_v12/best_params/1", "best": b, "run": self.dir.name})
                write_json(self.dir / "best_state.json", {"format": "cortex_v12/best_state/1", "best": b, "run": self.dir.name,
                                                         "note_ko": "dev 정확도 최대 → 동률이면 CE 최소 → 이른 epoch. 이어 학습은 last 에서 한다."})
                b["mirrored_seq"] = st["seq"]
            self._tick("save_s")

    def params_sha(self, W: np.ndarray | None = None, b: np.ndarray | None = None, gains_sha: str | None = None) -> str:
        W = self.W if W is None else W
        b = self.b if b is None else b
        return sha256_text(sha256_array(W, b) + (self.engine.gains_sha() if gains_sha is None else gains_sha))

    # ---------------------------------------------------------------- 진입
    def execute(self) -> None:
        with StopFlag() as stop:
            self.stop = stop
            self._skip_time()
            self.state.setdefault("events", []).append({"utc": utc_now(), "start_phase": self.state.get("phase"),
                                                        "position": self.position_text(), "backend": self.be.describe()})
            while True:
                ph = self.state["phase"]
                if ph == "prepare_v1":
                    self.prepare("V1")
                elif ph == "prepare_v2":
                    self.prepare("V2")
                elif ph == "cache":
                    self.cache_phase()
                elif ph == "train":
                    self.train_phase()
                elif ph == "evaluate":
                    self.evaluate_phase()
                elif ph == "done":
                    break
                else:
                    raise IntegrityError(f"알 수 없는 phase: {ph}")

    def position_text(self) -> str:
        st = self.state
        ph = st.get("phase")
        if ph in ("prepare_v1", "prepare_v2"):
            pr = (st.get("prep") or {}).get("stage") or {}
            if not pr:
                return f"{ph}: 준비 입력 파일·prototype 초기화 단계"
            return (f"{ph}: {pr.get('area')} epoch {int(pr.get('epochs_completed', 0)) + 1}/{pr.get('max_epochs')} 배치 {pr.get('next_batch')} "
                    f"(dev best {fmt_pct(pr.get('best_dev_accuracy'))})")
        if ph == "cache":
            return "cache: V2 L3 캐시 생성 (train·dev) — 완료한 행 다음부터"
        if ph == "train":
            tr = st.get("train") or {}
            b = tr.get("best") or {}
            e = int(tr.get("epoch", -1))
            if e < 0:
                return "train: epoch 0 (학습 전) dev 평가부터"
            nb = int(math.ceil(len(self.data.split["train"]["idx"]) / self.B)) if self.data is not None else None
            return (f"train: epoch {e + 1}/{self.E} 배치 {tr.get('next_batch')}/{nb} (완료 epoch {e}, best epoch {b.get('epoch')} "
                    f"dev {fmt_pct(b.get('dev_accuracy'))})")
        if ph == "evaluate":
            return "evaluate: 선택 상태로 test 캐시·평가"
        return str(ph)

    # ---------------------------------------------------------------- 캐시 파일 (memmap .npy, 이어 만들기)
    def _book_load(self) -> None:
        p = self.dir / "cache_manifest.json"
        self.book = read_json(p) if p.is_file() else {"format": CACHE_FORMAT, "files": {}}

    def _book_save(self) -> None:
        write_json(self.dir / "cache_manifest.json", self.book)

    def _feature_key(self, name: str, sp: str, boundary: str, dim: int, stem_sha: dict[str, Any]) -> dict[str, Any]:
        d = self.data
        return {"format": CACHE_FORMAT, "program_version": VERSION, "name": name, "split": sp, "n": int(len(d.split[sp]["idx"])),
                "order_sha256": d.order_sha(sp), "ids_sha256": d.ids_sha(sp), "boundary": boundary, "dim": int(dim), "dtype": "float32",
                "preprocessing": json.loads(dumps({k: self.M[k] for k in ("image_size", "transparent_background", "retina", "logpolar")}
                                                  | {"color_channels": self.M["areas"]["V1"]["color_channels"]}, indent=None)),
                "l6": json.loads(dumps(self.M["l6"], indent=None)), "stem": stem_sha, "backend": self.be.numeric_id()}

    def disk_plan(self, names_dims: Sequence[tuple[str, int, int]]) -> tuple[int, str]:
        need, parts = 0, []
        for name, n, dim in names_dims:
            ent = (self.book.get("files") or {}).get(name) or {}
            if ent.get("complete"):
                continue
            b = int(n) * int(dim) * 4
            need += b if not (self.dir / f"{name}.npy").is_file() else 0
            parts.append(f"{name} {fmt_bytes(b)} ({n:,} × {dim} float32)")
        return need, ", ".join(parts)

    def check_disk(self, names_dims: Sequence[tuple[str, int, int]]) -> None:
        need, txt = self.disk_plan(names_dims)
        if not txt:
            return
        free = shutil.disk_usage(self.dir).free
        say(f"  디스크: {txt} — 새로 필요 {fmt_bytes(need)}, 여유 {fmt_bytes(free)}")
        if need > free * 0.98:
            raise V12Error(f"디스크 여유 ({fmt_bytes(free)}) 가 캐시에 필요한 {fmt_bytes(need)} 보다 작다. 공간을 비우거나 다른 결과 폴더를 쓴다.")

    def build_feature(self, name: str, n: int, dim: int, key: dict[str, Any], rows_fn: Callable[[int, int], np.ndarray], label: str,
                      bucket: str) -> None:
        """rows_fn(a0, a1) → (a1 − a0, dim) float32. 조각마다 쓰고, 진행 행 수를 cache_manifest.json 에 기록한다 (중단 뒤 다음 행부터)."""
        files = self.book.setdefault("files", {})
        ksha = sha256_text(dumps(key, indent=None))
        path = self.dir / f"{name}.npy"
        ent = files.get(name)
        if ent and ent.get("key_sha256") == ksha and ent.get("complete") and path.is_file():
            return
        if not ent or ent.get("key_sha256") != ksha or not path.is_file():
            if ent and ent.get("key_sha256") != ksha:
                say(f"  {name}: 캐시 키가 달라 처음부터 다시 만든다 (데이터·전처리·동결 파라미터·계산 뒷단 중 하나가 다르다).")
            ent = {"file": path.name, "key": key, "key_sha256": ksha, "rows": int(n), "dim": int(dim), "dtype": "float32", "rows_done": 0,
                   "complete": False, "created_utc": utc_now()}
            files[name] = ent
            self._book_save()
            mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(int(n), int(dim)))
        else:
            mm = np.lib.format.open_memmap(path, mode="r+")
            if mm.shape != (int(n), int(dim)) or mm.dtype != np.float32:
                raise IntegrityError(f"{path.name}: 모양·형 {mm.shape}/{mm.dtype} 이 기록과 다르다.")
        done = int(ent["rows_done"])
        if done:
            say(f"  {label}: {done:,}/{n:,} 행까지 만들어 둔 것을 이어 만든다.")
        else:
            say(f"  {label}: {n:,} 행 × {dim} 차원 float32 ({fmt_bytes(n * dim * 4)}) 만들기 시작")
        t_last, t0, done0 = time.perf_counter(), time.perf_counter(), done
        try:
            for a0, a1 in chunk_bounds(n, int(self.S["compute"]["cache_batch_size"])):
                if a1 <= done:
                    continue
                if self.stop.requested:
                    raise Interrupted(f"{label} {done:,}/{n:,} 행에서 멈춤")
                rows = rows_fn(a0, a1)
                if rows.shape != (a1 - a0, int(dim)) or not np.isfinite(rows).all():
                    raise NumericalError(f"{label}: 행 {a0}~{a1} 의 모양 또는 값 (NaN/Inf) 이 잘못되었다.")
                mm[a0:a1] = rows
                done = a1
                now = time.perf_counter()
                if now - t_last >= 20.0 or a1 == n:
                    mm.flush()
                    with self.stop.critical():
                        ent["rows_done"] = done
                        self._book_save()
                    rate = (now - t0) / max(1, done - done0)
                    say(f"    {label}: {done:,}/{n:,} 행 ({fmt_s(now - t0)} 경과, 남은 예상 {fmt_s(rate * (n - done))})")
                    t_last = now
        finally:
            mm.flush()
            with self.stop.critical():
                ent["rows_done"] = done
                self._book_save()
            del mm
            self.be.sync()
            self._tick(bucket)
        sha = sha256_file(path)
        with self.stop.critical():
            ent.update(rows_done=int(n), complete=True, sha256=sha, bytes=int(path.stat().st_size), completed_utc=utc_now())
            self._book_save()
        self._sha_ok.add(name)
        self._tick(bucket)

    def open_feature(self, name: str, key: dict[str, Any] | None = None) -> np.ndarray:
        ent = (self.book.get("files") or {}).get(name)
        path = self.dir / f"{name}.npy"
        if not ent or not ent.get("complete") or not path.is_file():
            raise IntegrityError(f"{name}: 완성된 캐시가 아니다 (미완료 파일을 완성 캐시로 읽지 않는다).")
        if key is not None and ent.get("key_sha256") != sha256_text(dumps(key, indent=None)):
            raise IntegrityError(f"{name}: 캐시 키가 지금 데이터·동결 파라미터·계산 뒷단과 다르다.")
        if path.stat().st_size != ent.get("bytes"):
            raise IntegrityError(f"{name}: 파일 크기가 기록과 다르다.")
        srec = (self.state.get("caches") or {}).get(name)
        if srec and srec.get("sha256") != ent.get("sha256"):
            raise IntegrityError(f"{name}: cache_manifest.json 의 해시가 학습 상태에 기록된 캐시 해시와 다르다.")
        if name not in self._sha_ok:
            if sha256_file(path) != ent.get("sha256"):
                raise IntegrityError(f"{name}: SHA256 이 기록과 다르다 (바뀐 캐시).")
            self._sha_ok.add(name)
        mm = np.load(path, mmap_mode="r")
        if mm.shape != (int(ent["rows"]), int(ent["dim"])) or mm.dtype != np.float32:
            raise IntegrityError(f"{name}: 모양·형이 기록과 다르다.")
        return mm

    def delete_feature(self, name: str) -> None:
        p = self.dir / f"{name}.npy"
        for k in range(3):
            try:
                p.unlink()
                break
            except FileNotFoundError:
                break
            except OSError as exc:                                   # Windows: 아직 열린 memmap 이 있으면 지우지 못한다 → 정리 후 다시
                gc.collect()
                time.sleep(0.2 * (k + 1))
                if k == 2:
                    say(f"  [알림] 임시 파일 {p.name} 을 지우지 못했다 ({exc}). 결과와 무관하다 — 나중에 직접 지워도 된다.")
        with self.stop.critical():
            (self.book.get("files") or {}).pop(name, None)
            self._book_save()

    def lgn_rows(self, sp: str, a0: int, a1: int) -> np.ndarray:
        d = self.data.split[sp]
        if sp == "test":
            t = self.state.setdefault("test", {})
            t["images_read"] = int(t.get("images_read", 0)) + (a1 - a0)
            t.setdefault("first_read_phase", self.state["phase"])
        return np.stack([self.pre.vector(self.data.image(s, i)) for s, i in zip(d["src"][a0:a1].tolist(), d["idx"][a0:a1].tolist())])

    # ---------------------------------------------------------------- 동결 줄기
    def stem_sha(self, q_full: np.ndarray, thr_full: np.ndarray, areas: Sequence[str] = STEM_AREAS) -> dict[str, Any]:
        keys = [f"{a}/{l}" for a in areas for l in LAYERS]
        sp = self.specs
        return {"areas": list(areas), "q_sha256": sha256_array(*[q_full[sp[k].start:sp[k].end] for k in keys]),
                "thr_sha256": sha256_array(*[thr_full[sp[k].start:sp[k].end] for k in keys]),
                "struct_hash": {a: struct_hashes(sp)[a] for a in areas}}

    def read_stem(self) -> dict[str, Any]:
        p = self.dir / "frozen_stem.npz"
        rec = self.state.get("stem") or {}
        if not p.is_file() or sha256_file(p) != rec.get("file_sha256"):
            raise IntegrityError("frozen_stem.npz 가 없거나 해시가 기록과 다르다 (동결 V1·V2 가 바뀌었다).")
        arrays, meta = load_npz(p)
        if (meta or {}).get("format") != STEM_FORMAT:
            raise IntegrityError("frozen_stem.npz 형식이 다르다.")
        if (meta or {}).get("struct_hash") != self.state["world"]["struct_hash"]:      # world 는 열 때 지금 구조로 다시 계산해 확인했다
            raise IntegrityError("frozen_stem.npz 의 구조 해시가 이 run 의 구조와 다르다.")
        return {"q": arrays["q"], "threshold": arrays["threshold"], "W0": arrays["W0"], "b0": arrays["b0"], "meta": meta}

    def write_stem(self, q_full: np.ndarray, thr_full: np.ndarray, W0: np.ndarray, b0: np.ndarray, source: dict[str, Any],
                   prep_summary: dict[str, Any] | None) -> None:
        sp = self.specs
        sh = struct_hashes(sp)
        base_q = {k: q_full[sp[k].start:sp[k].end] for k in (f"{a}/{l}" for a in STEM_AREAS for l in LAYERS)}
        v11_rec = {"split_sha256": self.data.split_sha256(), "struct_hash": sh, "feature_dim": int(sp["IT/L3"].n),
                   "base_q_sha256": sha256_array(*[base_q[k] for k in sorted(base_q)]), "head_init_sha256": sha256_array(W0, b0),
                   "init_state_sha256": sha256_text(sha256_array(q_full, thr_full, W0, b0) + dumps(sh, indent=None) + "gain=1")}
        meta = {"format": STEM_FORMAT, "program": PROGRAM, "version": VERSION, "created_utc": utc_now(), "seed": self.seed, "source": source,
                "struct_hash": sh, "N": int(len(q_full)), "layer_ranges": {k: [v.start, v.end] for k, v in sp.items()},
                "frozen_areas": list(STEM_AREAS), "stem": self.stem_sha(q_full, thr_full), "prep": prep_summary,
                "v11_comparable_record": v11_rec,
                "note_ko": "V1·V2 의 q·임계값·연결·전처리·L6 설정은 이후 바뀌지 않는다. V4·IT q = 1·임계값은 고정, 학습은 V4·IT 연결 gain 과 출력층만."}
        p = self.dir / "frozen_stem.npz"
        with self.stop.critical() if self.stop is not None else contextlib.nullcontext():
            save_npz(p, {"q": q_full.astype(np.float32), "threshold": thr_full.astype(np.float32), "W0": W0.astype(np.float64),
                         "b0": b0.astype(np.float64)}, meta)
            self.state["stem"] = {"file_sha256": sha256_file(p), "source_kind": source.get("kind"), "v11_comparable_record": v11_rec,
                                  "stem": meta["stem"]}

    def make_stem(self, q_full: np.ndarray, thr_full: np.ndarray) -> Stem:
        return Stem(self.be, self.specs, self.M, q_full, thr_full)

    # ---------------------------------------------------------------- V1·V2 준비 학습 (새 준비 경로)
    def _prep_inputs(self, area: str, stem: Stem, q_full: np.ndarray, thr_full: np.ndarray) -> tuple[str, dict[str, np.ndarray]]:
        """준비 입력 파일: V1 = LGN 표본 (prep_lgn_*), V2 = 동결 V1 L3 출력 (prep_v1l3_*). train·dev 만 (test 는 읽지 않는다)."""
        sp = self.specs
        if area == "V1":
            dims = [(f"prep_lgn_{s}", len(self.data.split[s]["idx"]), sp["V1/L4"].n_src) for s in ("train", "dev")]
            self.check_disk(dims)
            for s in ("train", "dev"):
                key = self._feature_key(f"prep_lgn_{s}", s, "LGN", sp["V1/L4"].n_src, {"preprocessing_only": True})
                self.build_feature(f"prep_lgn_{s}", len(self.data.split[s]["idx"]), sp["V1/L4"].n_src, key,
                                   lambda a0, a1, s=s: self.lgn_rows(s, a0, a1), f"준비 입력 LGN {s}", "prep_s")
            return "LGN", {s: self.open_feature(f"prep_lgn_{s}") for s in ("train", "dev")}
        ssha = self.stem_sha(q_full, thr_full, ("V1",))
        dims = [(f"prep_v1l3_{s}", len(self.data.split[s]["idx"]), sp["V1/L3"].n) for s in ("train", "dev")]
        self.check_disk(dims)
        for s in ("train", "dev"):
            key = self._feature_key(f"prep_v1l3_{s}", s, "V1/L3", sp["V1/L3"].n, ssha)
            ent = (self.book.get("files") or {}).get(f"prep_v1l3_{s}") or {}
            if ent.get("complete") and ent.get("key_sha256") == sha256_text(dumps(key, indent=None)):
                continue
            lgn = self.open_feature(f"prep_lgn_{s}")
            self.build_feature(f"prep_v1l3_{s}", len(self.data.split[s]["idx"]), sp["V1/L3"].n, key,
                               lambda a0, a1, lgn=lgn: self.be.to_np(stem.forward(self.be.f32(lgn[a0:a1]), "LGN", "V1/L3")["out"]).astype(np.float32),
                               f"준비 입력 V1 L3 {s} (동결 V1)", "prep_s")
            del lgn
        for s in ("train", "dev"):
            if (self.dir / f"prep_lgn_{s}.npy").is_file():
                self.delete_feature(f"prep_lgn_{s}")
        return "V1/L3", {s: self.open_feature(f"prep_v1l3_{s}") for s in ("train", "dev")}

    def _class_sums(self, stem: Stem, X: np.ndarray, y: np.ndarray, start: str, stop_key: str, dim: int) -> tuple[Any, np.ndarray]:
        be = self.be
        sums = be.zeros((N_CLASSES, dim))
        counts = np.zeros(N_CLASSES, np.int64)
        for a0 in range(0, len(y), self.B):
            if self.stop.requested:
                raise Interrupted("준비 학습 train 통과 도중 멈춤 (이어 갈 때 이 통과를 처음부터 다시 한다)")
            yy = y[a0:a0 + self.B]
            h = stem.forward(be.f32(X[a0:a0 + self.B]), start, stop_key)["out"]
            for c in np.unique(yy):
                rows = be.i64(np.nonzero(yy == c)[0])
                sums[int(c)] += h[rows].sum(0)
            counts += np.bincount(yy, minlength=N_CLASSES)
        return sums, counts

    def prepare(self, area: str) -> None:
        be, S, P_ = self.be, self.S, self.S["prep"]
        self.load_world()
        k2, k3 = f"{area}/L2", f"{area}/L3"
        sp = self.specs
        N = max(x.end for x in sp.values())
        q_full = np.full(N, float(self.M["neuron"]["q_init"]), np.float32)
        thr_full = np.full(N, float(self.M["neuron"]["threshold"]), np.float32)
        for key in (f"{a}/{l}" for a in STEM_AREAS for l in LAYERS):
            if f"q.{key}" in self.arrays:
                q_full[sp[key].start:sp[key].end] = self.arrays[f"q.{key}"]
        stem = self.make_stem(q_full, thr_full)
        say(f"\n[{'prepare_v1' if area == 'V1' else 'prepare_v2'}] {area} 국소 q 준비 학습 (최대 {P_['epochs'][area]} epoch, patience {P_['patience']}, "
            f"train {len(self.data.split['train']['idx']):,} · dev {len(self.data.split['dev']['idx']):,})")
        start, X = self._prep_inputs(area, stem, q_full, thr_full)
        y_tr, y_dv = self.data.split["train"]["y"], self.data.split["dev"]["y"]
        dim = sp[k3].n
        eps = float(P_["eps_norm"])
        pr = self.state.setdefault("prep", {})
        st = pr.get("stage")
        if not st or st.get("area") != area:
            self._skip_time()
            sums, counts = self._class_sums(stem, X["train"], y_tr, start, k3, dim)
            Pm = sums * be.f32((1.0 / np.maximum(counts, 1))[:, None])
            st = {"area": area, "status": "running", "epochs_completed": 0, "next_batch": 0, "max_epochs": int(P_["epochs"][area]),
                  "patience": int(P_["patience"]), "min_delta": float(P_["min_delta"]), "best_dev_accuracy": None, "best_epoch": None,
                  "bad_epochs": 0, "history": []}
            pr["stage"] = st
            self.arrays.update({"proto.P": be.to_np(Pm).astype(np.float32), "proto.valid": be.to_np(be.norm_last(Pm)).astype(np.float64) > eps,
                                "proto.counts": counts})
            self.save("prep_s")
            say(f"  {area} prototype 초기화 (train 클래스 평균, 무효 클래스 {int((~self.arrays['proto.valid']).sum())} 개)")
        Pm = be.f32(self.arrays["proto.P"])
        valid = np.asarray(self.arrays["proto.valid"], bool)
        router = ParentRouter(be, sp[k2], sp[k3], float(P_["alpha2"]), float(P_["alpha3"]))
        L2, L3 = stem.layers[k2], stem.layers[k3]
        ne = self.M["neuron"]
        n = len(y_tr)
        nb = int(math.ceil(n / self.B))
        eta, e0, rho = float(P_["eta"]), float(P_["e0"]), float(P_["rho"])
        every, min_s = int(S["save"]["every_batches"]), float(S["save"]["min_seconds"])
        t_save = time.perf_counter()
        self._skip_time()
        while st["status"] not in STAGE_TERMINAL and int(st["epochs_completed"]) < int(st["max_epochs"]):
            e = int(st["epochs_completed"])
            perm = epoch_permutation_prep(self.seed, area, e, n)
            t_ep = time.perf_counter()
            for bi in range(int(st["next_batch"]), nb):
                idx = perm[bi * self.B:(bi + 1) * self.B]
                y = y_tr[idx]
                r = stem.forward(be.f32(X["train"][idx]), start, k3, keep=(k2, k3))
                vp = valid[y]
                delta = (Pm[be.i64(y)] - r[k3]["out"]) * be.f32(vp.astype(np.float32)[:, None])
                m2, m3 = router.route(delta)
                s2, s3 = r[k2]["s"], r[k3]["s"]
                dq2 = eta * (s2 / (s2 + e0) * m2).mean(0) if be.kind == "numpy" else eta * (s2 / (s2 + e0) * m2).mean(dim=0)
                dq3 = eta * (s3 / (s3 + e0) * m3).mean(0) if be.kind == "numpy" else eta * (s3 / (s3 + e0) * m3).mean(dim=0)
                n2 = be.clip(L2.q + dq2, float(ne["q_min"]), float(ne["q_max"]))
                n3 = be.clip(L3.q + dq3, float(ne["q_min"]), float(ne["q_max"]))
                if not be.all_finite_many([m2, m3, n2, n3]):
                    raise NumericalError(f"{area} 준비 학습 epoch {e + 1} 배치 {bi}: 교정·q 갱신값에 NaN/Inf (적용하지 않는다).")
                with self.stop.critical():
                    L2.q, L3.q = n2, n3
                    st["next_batch"] = bi + 1
                if (bi + 1 < nb and (bi + 1) % every == 0 and time.perf_counter() - t_save >= min_s) or self.stop.requested:
                    self._prep_store(stem)
                    self.save("prep_s")
                    t_save = time.perf_counter()
                if self.stop.requested:
                    raise Interrupted(f"{area} 준비 학습 epoch {e + 1} 배치 {bi + 1}/{nb} 에서 멈춤 (저장됨)")
                if (bi + 1) % int(S["save"]["progress_every_batches"]) == 0 and bi + 1 < nb:
                    say(f"    {area} epoch {e + 1} 배치 {bi + 1}/{nb} ({fmt_s(time.perf_counter() - t_ep)} 경과)")
            self._prep_store(stem)
            self.save("prep_s")
            # epoch 끝: q 고정 train 통과 → prototype EMA → dev 판독 → 단계 상태
            sums, counts = self._class_sums(stem, X["train"], y_tr, start, k3, dim)
            hbar = sums * be.f32((1.0 / np.maximum(counts, 1))[:, None])
            has = be.f32((counts > 0).astype(np.float32)[:, None])
            P_new = Pm + has * (rho * (hbar - Pm))
            valid_new = be.to_np(be.norm_last(P_new)).astype(np.float64) > eps
            correct = 0
            for a0 in range(0, len(y_dv), self.B):
                if self.stop.requested:
                    raise Interrupted(f"{area} 준비 학습 epoch {e + 1} dev 판독 도중 멈춤 (이어 갈 때 epoch 끝 처리부터 다시)")
                h = stem.forward(be.f32(X["dev"][a0:a0 + self.B]), start, k3)["out"]
                correct += int((cosine_predict(be, h, P_new, valid_new, eps) == y_dv[a0:a0 + self.B]).sum())
            acc = correct / len(y_dv) if len(y_dv) else None
            improved = st["best_dev_accuracy"] is None or (acc is not None and acc > st["best_dev_accuracy"] + st["min_delta"])
            if improved:
                st["best_dev_accuracy"], st["best_epoch"], st["bad_epochs"] = acc, e, 0
            else:
                st["bad_epochs"] = int(st["bad_epochs"]) + 1
            st["epochs_completed"] = e + 1
            if int(st["bad_epochs"]) >= int(st["patience"]):
                st["status"] = "plateau"
            elif int(st["epochs_completed"]) >= int(st["max_epochs"]):
                st["status"] = "budget_exhausted"
            st["next_batch"] = 0
            st["history"].append({"epoch_number": e + 1, "dev_accuracy": acc, "dev_correct": correct, "best_dev_accuracy": st["best_dev_accuracy"],
                                  "bad_epochs": st["bad_epochs"], "status": st["status"], "invalid_prototypes": int((~valid_new).sum()),
                                  "epoch_seconds": round(time.perf_counter() - t_ep, 2), "utc": utc_now()})
            Pm, valid = P_new, valid_new
            self.arrays.update({"proto.P": be.to_np(Pm).astype(np.float32), "proto.valid": valid, "proto.counts": counts})
            self._prep_store(stem)
            self.save("prep_s")
            say(f"  [{area} epoch {e + 1}/{st['max_epochs']}] dev 코사인 판독 정확도 {fmt_pct(acc)} (best {fmt_pct(st['best_dev_accuracy'])}, "
                f"epoch {int(st['best_epoch']) + 1}) → {st['status']} — 준비 누적 {fmt_s(self.state['time'].get('prep_s', 0.0))}")
            if self.stop.requested:
                raise Interrupted(f"{area} 준비 학습 epoch {e + 1} 끝에서 멈춤 (저장됨)")
        if st["status"] not in STAGE_TERMINAL:
            st["status"] = "budget_exhausted"
        pr[area] = {k: st[k] for k in ("status", "epochs_completed", "best_dev_accuracy", "best_epoch", "max_epochs", "patience")}
        pr[area]["history"] = st["history"]
        pr[area]["policy_ko"] = "끝 상태 정책: 마지막 epoch 의 q 를 쓴다 (dev 최선 epoch 로 되돌리지 않는다, v11 과 같다)."
        self._prep_store(stem)
        if area == "V1":
            pr["stage"] = None
            for k in ("proto.P", "proto.valid", "proto.counts"):
                self.arrays.pop(k, None)
            self.state["phase"] = "prepare_v2"
            self.save("prep_s")
            return
        for k in (f"{a}/{l}" for a in STEM_AREAS for l in LAYERS):
            q_full[sp[k].start:sp[k].end] = stem.q_numpy(k)
        for k in LEARN_KEYS:
            q_full[sp[k].start:sp[k].end] = 1.0
        W0, b0 = head_init(self.seed, sp["IT/L3"].n)
        summ = {a: {k: v for k, v in pr[a].items() if k != "history"} for a in STEM_AREAS}
        self.write_stem(q_full, thr_full, W0, b0, {"kind": "fresh_prep_in_this_run", "settings": P_, "batch_size": self.B}, summ)
        pr["stage"] = None
        for k in list(self.arrays):
            if k.startswith(("q.", "proto.")):
                self.arrays.pop(k)
        self.state["phase"] = "cache"
        self.save("prep_s")
        say(f"  V1·V2 준비 완료 → 동결 (V1 {pr['V1']['status']} {pr['V1']['epochs_completed']} epoch, V2 {pr['V2']['status']} "
            f"{pr['V2']['epochs_completed']} epoch). frozen_stem.npz 저장")

    def _prep_store(self, stem: Stem) -> None:
        for key in stem.order:
            self.arrays[f"q.{key}"] = stem.q_numpy(key)

    # ---------------------------------------------------------------- 캐시 (V2 L3 최종 출력, L6 포함)
    def _main_cache_key(self, s: str) -> dict[str, Any]:
        """캐시 키: 분할 ID·순서, 전처리, 동결 V1·V2 q·임계값·구조 해시 (frozen_stem 기록), L6 설정, 차원·dtype, 계산 뒷단·버전, 캐시 형식."""
        return self._feature_key(f"cache_{s}", s, "V2/L3 (L6 포함 최종 전송 출력)", int(self.state["world"]["cache_dim"]), self.state["stem"]["stem"])

    def _build_main_cache(self, splits: Sequence[str], bucket: str) -> None:
        self.load_world()
        self.ensure_full_specs()
        stem_d = self.read_stem()
        dim = int(self.state["world"]["cache_dim"])
        self.check_disk([(f"cache_{s}", len(self.data.split[s]["idx"]), dim) for s in splits])
        stem = None
        for s in splits:
            key = self._main_cache_key(s)
            ent = (self.book.get("files") or {}).get(f"cache_{s}") or {}
            if ent.get("complete") and ent.get("key_sha256") == sha256_text(dumps(key, indent=None)):
                continue
            if stem is None:
                stem = self.make_stem(stem_d["q"], stem_d["threshold"])
            n = len(self.data.split[s]["idx"])
            pre_name = f"prep_v1l3_{s}"
            pent = (self.book.get("files") or {}).get(pre_name) or {}
            v1_key = self._feature_key(pre_name, s, "V1/L3", self.specs["V1/L3"].n, self.stem_sha(stem_d["q"], stem_d["threshold"], ("V1",)))
            if pent.get("complete") and pent.get("key_sha256") == sha256_text(dumps(v1_key, indent=None)):
                v1 = self.open_feature(pre_name)        # 새 준비 경로: 동결 V1 L3 (같은 V1 q) 에서 V2 만 계산 (같은 값)
                fn = (lambda a0, a1, v1=v1, st_=stem: self.be.to_np(st_.forward(self.be.f32(v1[a0:a1]), "V1/L3", "V2/L3")["out"]).astype(np.float32))
                label = f"V2 L3 캐시 {s} (동결 V1 L3 → V2)"
            else:
                fn = (lambda a0, a1, s=s, st_=stem: self.be.to_np(st_.forward(self.be.f32(self.lgn_rows(s, a0, a1)), "LGN", "V2/L3")["out"]).astype(np.float32))
                label = f"V2 L3 캐시 {s} (영상 → LGN → V1 → V2)"
            self.build_feature(f"cache_{s}", n, dim, key, fn, label, bucket)
        stem = None                                                    # 동결 줄기 계산 구조를 놓아 준다
        self._skip_time()

    def cache_phase(self) -> None:
        say("\n[cache] 동결 V1·V2 의 V2 L3 최종 출력 (L6 포함) 캐시 — train·dev (test 는 학습·선택이 끝난 뒤)")
        self._build_main_cache(("train", "dev"), "cache_s")
        for s in ("train", "dev"):
            if (self.dir / f"prep_v1l3_{s}.npy").is_file() or f"prep_v1l3_{s}" in (self.book.get("files") or {}):
                self.delete_feature(f"prep_v1l3_{s}")
        self.state["caches"] = {n: {k: self.book["files"][n][k] for k in ("sha256", "bytes", "rows", "dim", "key_sha256")}
                                for n in ("cache_train", "cache_dev")}
        self.state["phase"] = "train"
        self.state.setdefault("train", {"epoch": -1, "next_batch": 0, "acc": {}, "best": None, "history": []})
        self.save("cache_s")
        say(f"  캐시 완료 (train {fmt_bytes((self.dir / 'cache_train.npy').stat().st_size)}, dev {fmt_bytes((self.dir / 'cache_dev.npy').stat().st_size)}). "
            "준비용 임시 파일은 지웠다.")

    # ---------------------------------------------------------------- 주 학습 (V4·IT gain + 출력층)
    def ensure_full_specs(self) -> None:
        """준비·캐시 생성에는 V1·V2 구조가 필요하다 (주 학습 중에는 내려놓는다)."""
        self.load_world()
        if any(f"{a}/{l}" not in self.specs for a in STEM_AREAS for l in LAYERS):
            self.specs = build_structure(self.M, self.pre)

    def drop_stem_specs(self) -> None:
        for k in [k for k in self.specs if k.split("/")[0] in STEM_AREAS]:
            del self.specs[k]

    def ensure_engine(self) -> None:
        """V4·IT 엔진과 last 학습 상태 (W·b·gain·best) 를 한 번만 만든다 (같은 프로세스에서 train → evaluate 로 넘어가도 그대로 쓴다)."""
        if getattr(self, "engine", None) is not None:
            return
        self.build_engine()
        self._restore_train_arrays()

    def build_engine(self) -> None:
        stem_d = self.read_stem()
        self.engine = Engine(self.be, self.specs, self.M, stem_d["q"], stem_d["threshold"], gain_min=float(self.S["gain_min"]),
                             gain_max=float(self.S["gain_max"]), memory_mb=int(self.S["compute"]["memory_mb"]))
        self.stem_W0, self.stem_b0 = stem_d["W0"], stem_d["b0"]

    def _restore_train_arrays(self) -> None:
        a = self.arrays
        if "W" in a:
            self.W, self.b = np.asarray(a["W"], np.float64).copy(), np.asarray(a["b"], np.float64).copy()
            self.engine.load_gain_arrays(a, "g")
            if self.params_sha() != self.state.get("params_sha256"):
                raise IntegrityError("last_params.npz 의 W·b·gain 해시가 상태 기록과 다르다.")
            self.best_arrays = {k[5:]: v for k, v in a.items() if k.startswith("best.")}
        else:
            self.W, self.b = np.asarray(self.stem_W0, np.float64).copy(), np.asarray(self.stem_b0, np.float64).copy()
        for k in [k for k in a if k in ("W", "b") or k.startswith(("g.", "best."))]:
            a.pop(k)

    def eval_cache(self, mm: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray, label: str) -> dict[str, Any]:
        be, eng = self.be, self.engine
        acc = EvalAcc()
        for a0 in range(0, len(y), self.B):
            if self.stop.requested:
                raise Interrupted(f"{label} 평가 도중 멈춤 (이 평가는 버리고 이어 갈 때 다시 한다)")
            h = be.to_np(eng.forward(be.f32(mm[a0:a0 + self.B]), record=False)["h"]).astype(np.float64)
            _z, p, logp = head_forward(W, b, h)
            acc.add(p, logp, y[a0:a0 + self.B])
        return acc.result()

    def train_step(self, x0: Any, y: np.ndarray) -> tuple[list[Any], np.ndarray, np.ndarray, float, int]:
        """같은 순방향 상태에서 ΔW·Δb·Δγ 를 모두 계산한다 (적용은 호출한 쪽에서 한 번에). 라벨은 목표 t 에만 쓴다."""
        be, eng = self.be, self.engine
        rec = eng.forward(x0, record=True)
        h = be.to_np(rec["h"]).astype(np.float64)
        _z, p, logp = head_forward(self.W, self.b, h)
        n = len(y)
        t = np.zeros((n, N_CLASSES), np.float64)
        t[np.arange(n), y] = 1.0
        dz = t - p
        dW = float(self.S["eta_head"]) * (dz.T @ h) / n
        db = float(self.S["eta_head"]) * dz.mean(axis=0)
        d_h = head_route_trace(self.W, h, dz)                          # 갱신 전 W 로 배분
        try:
            eng.route_back(rec, be.f32(d_h))
        except BaseException:
            eng.clear_grads()
            raise
        todo = eng.proposals(float(self.S["eta_edge"]), n)
        Wn, bn = self.W + dW, self.b + db
        if not (np.isfinite(Wn).all() and np.isfinite(bn).all()):
            raise NumericalError("출력층 W·b 갱신값에 NaN/Inf 가 있다 (적용하지 않는다).")
        ce = -logp[np.arange(n), y]
        return todo, Wn, bn, float(ce.sum()), int((p.argmax(axis=1) == y).sum())

    @staticmethod
    def better(acc: float | None, ce: float, best: dict[str, Any] | None) -> bool:
        """dev 정확도 최대 → 동률이면 dev CE 최소 → 다시 동률이면 이른 epoch (나중 epoch 는 엄격히 좋아야 한다)."""
        if best is None:
            return True
        if acc is None:
            return False
        return acc > best["dev_accuracy"] or (acc == best["dev_accuracy"] and ce < best["dev_ce"])

    def _epoch_end(self, e: int, dev: dict[str, Any], dev_s: float, train_s: float) -> dict[str, Any]:
        tr = self.state["train"]
        a = tr.get("acc") or {}
        new_best = self.better(dev["accuracy"], dev["ce"], tr.get("best"))
        gsum = self.engine.gain_summary()
        if new_best:
            self.best_arrays = {"W": self.W.copy(), "b": self.b.copy(), **self.engine.gain_arrays("g")}
            tr["best"] = {"epoch": e, "dev_accuracy": dev["accuracy"], "dev_ce": dev["ce"], "dev_confusion": dev["confusion"],
                          "params_sha256": self.params_sha(), "mirrored_seq": None, "utc": utc_now()}
        n_ = max(1, int(a.get("n", 0)))
        row = {"epoch": e, "train_ce_running": (float(a.get("ce_sum", 0.0)) / n_) if a.get("n") else None,
               "train_accuracy_running": (int(a.get("correct", 0)) / n_) if a.get("n") else None, "train_n": int(a.get("n", 0)),
               "dev_accuracy": dev["accuracy"], "dev_ce": dev["ce"], "new_best": bool(new_best), "best_epoch": tr["best"]["epoch"],
               "epoch_train_s": round(train_s, 3), "epoch_dev_eval_s": round(dev_s, 3), "utc": utc_now(),
               "gain_changed": int(sum(v["n_changed"] for v in gsum.values())), "gain_at_min": int(sum(v["n_at_min"] for v in gsum.values())),
               "gain_at_max": int(sum(v["n_at_max"] for v in gsum.values())), "gain_by_layer": gsum}
        tr["history"].append(row)
        tr["epoch"], tr["next_batch"], tr["acc"] = e, 0, {}
        return row

    def write_epochs_csv(self) -> None:
        cols = ("epoch", "train_ce_running", "train_accuracy_running", "dev_accuracy", "dev_ce", "best_epoch", "new_best", "gain_changed",
                "gain_at_min", "gain_at_max", "epoch_train_s", "epoch_dev_eval_s", "utc")
        lines = [",".join(cols)]
        for r in self.state["train"]["history"]:
            lines.append(",".join("" if r.get(c) is None else (f"{r[c]:.10g}" if isinstance(r.get(c), float) else str(r.get(c))) for c in cols))
        atomic_write_bytes(self.dir / "epochs.csv", ("\n".join(lines) + "\n").encode("utf-8"))

    def train_phase(self) -> None:
        self.load_world()
        be, S = self.be, self.S
        self.ensure_engine()
        self.drop_stem_specs()                                             # 주 학습에는 V1·V2 구조가 필요 없다 (메모리에서 내린다)
        mm_tr = self.open_feature("cache_train", self._main_cache_key("train"))
        mm_dv = self.open_feature("cache_dev", self._main_cache_key("dev"))
        y_tr, y_dv = self.data.split["train"]["y"], self.data.split["dev"]["y"]
        if mm_tr.shape[1] != self.engine.in_dim:
            raise IntegrityError("캐시 차원과 V4 L4 입력 차원이 다르다.")
        tr = self.state["train"]
        n = len(y_tr)
        nb = int(math.ceil(n / self.B))
        es = self.engine.structure()
        say(f"\n[train] V4·IT 흔적 학습: {self.E} epoch × {nb:,} 배치 (배치 {self.B}), 학습 연결 {sum(x['learnable_edges'] for x in es):,} 개 "
            f"(V2→V4 {es[0]['learnable_edges']:,}), IT L3 {self.engine.dim} 차원, 장치 {be.kind}/{be.device}")
        self._skip_time()
        if int(tr["epoch"]) < 0:
            t0 = time.perf_counter()
            dev = self.eval_cache(mm_dv, y_dv, self.W, self.b, "epoch 0 dev")
            be.sync()
            self._tick("dev_eval_s")
            row = self._epoch_end(0, dev, time.perf_counter() - t0, 0.0)
            self.save()
            self.write_epochs_csv()
            say(f"  [epoch 0/{self.E}] (학습 전) dev 정확도 {fmt_pct(row['dev_accuracy'])} · CE {row['dev_ce']:.4f}")
        every, min_s = int(S["save"]["every_batches"]), float(S["save"]["min_seconds"])
        prog = int(S["save"]["progress_every_batches"])
        t_save = time.perf_counter()
        while int(tr["epoch"]) < self.E:
            e = int(tr["epoch"]) + 1
            perm = epoch_permutation_main(self.seed, e, n)
            tr["perm_sha256"] = sha256_array(perm)
            t_ep = time.perf_counter()
            b0 = int(tr["next_batch"])
            for bi in range(b0, nb):
                idx = perm[bi * self.B:(bi + 1) * self.B]
                y = y_tr[idx]
                todo, Wn, bn, ce_sum, correct = self.train_step(be.f32(mm_tr[idx]), y)
                with self.stop.critical():                                  # W·b·γ 를 함께 적용 (반쯤 적용된 상태를 남기지 않는다)
                    self.engine.commit(todo)
                    self.W, self.b = Wn, bn
                    a = tr.setdefault("acc", {})
                    a["n"] = int(a.get("n", 0)) + len(y)
                    a["ce_sum"] = float(a.get("ce_sum", 0.0)) + ce_sum
                    a["correct"] = int(a.get("correct", 0)) + correct
                    tr["next_batch"] = bi + 1
                if (bi + 1 < nb and (bi + 1) % every == 0 and time.perf_counter() - t_save >= min_s) or self.stop.requested:
                    self.save("train_s")
                    t_save = time.perf_counter()
                if self.stop.requested:
                    raise Interrupted(f"epoch {e} 배치 {bi + 1}/{nb} 에서 멈춤 (배치 전체 갱신 뒤 저장됨)")
                if prog and (bi + 1) % prog == 0 and bi + 1 < nb:
                    a = tr["acc"]
                    el = time.perf_counter() - t_ep
                    say(f"    epoch {e} 배치 {bi + 1:,}/{nb:,} · train CE (진행 평균) {a['ce_sum'] / max(1, a['n']):.4f} · "
                        f"정확도 {fmt_pct(a['correct'] / max(1, a['n']))} · {el / max(1, bi + 1 - b0):.4f} 초/배치")
            self.save("train_s")
            train_s = time.perf_counter() - t_ep
            t_dev = time.perf_counter()
            dev = self.eval_cache(mm_dv, y_dv, self.W, self.b, f"epoch {e} dev")
            be.sync()
            self._tick("dev_eval_s")
            row = self._epoch_end(e, dev, time.perf_counter() - t_dev, train_s)
            if e >= self.E:
                self.state["phase"] = "evaluate"
            self.save()
            self.write_epochs_csv()
            t = self.state["time"]
            say(f"  [epoch {e}/{self.E}] train CE {row['train_ce_running']:.4f} · 정확도 {fmt_pct(row['train_accuracy_running'])} (진행 평균) | "
                f"dev 정확도 {fmt_pct(row['dev_accuracy'])} · CE {row['dev_ce']:.4f} | best epoch {row['best_epoch']}{' (새 최선)' if row['new_best'] else ''} | "
                f"학습 {fmt_s(row['epoch_train_s'])} · dev {fmt_s(row['epoch_dev_eval_s'])} | 누적 학습 {fmt_s(t.get('train_s', 0))}")
            if self.stop.requested and e < self.E:
                raise Interrupted(f"epoch {e} 평가·저장 뒤 멈춤")
        self.state["phase"] = "evaluate"
        self.save()

    # ---------------------------------------------------------------- 평가 (학습·선택이 끝난 뒤 test 한 번)
    def evaluate_phase(self) -> None:
        tr = self.state["train"]
        if int(tr["epoch"]) < self.E or not tr.get("best"):
            raise IntegrityError("학습·선택이 끝나지 않았다 (test 를 먼저 읽지 않는다).")
        self.load_world()
        say("\n[evaluate] 학습과 dev 선택이 끝났다 → test 캐시 생성 → 선택 상태 (best) 로 test 한 번 평가")
        self.state.setdefault("test", {})["selection_locked_epoch"] = tr["best"]["epoch"]
        self._skip_time()
        self._build_main_cache(("test",), "test_cache_s")
        self.ensure_engine()                                           # 엔진 (frozen_stem 구조 해시 확인에 V1·V2 구조가 필요) → 그 뒤 V1·V2 구조를 내린다
        self.drop_stem_specs()
        best = tr["best"]
        if not self.best_arrays:
            raise IntegrityError("선택 상태 (best) 배열이 없다.")
        last_gain_arrays = self.engine.gain_arrays("g")
        last_W, last_b = self.W, self.b
        self.engine.load_gain_arrays(self.best_arrays, "g")           # 평가용 복원 (학습용 last 는 그대로 남긴다)
        Wb, bb = np.asarray(self.best_arrays["W"], np.float64), np.asarray(self.best_arrays["b"], np.float64)
        if self.params_sha(Wb, bb) != best["params_sha256"]:
            raise IntegrityError("선택 상태 복원 뒤 파라미터 해시가 선택 때와 다르다.")
        mm_te = self.open_feature("cache_test", self._main_cache_key("test"))
        self._skip_time()
        test = self.eval_cache(mm_te, self.data.split["test"]["y"], Wb, bb, "test")
        self.be.sync()
        self._tick("test_eval_s")
        self.engine.load_gain_arrays(last_gain_arrays, "g")
        self.W, self.b = last_W, last_b
        stem_ok = sha256_file(self.dir / "frozen_stem.npz") == self.state["stem"]["file_sha256"]
        t = self.state["time"]
        prep_cache = sum(float(t.get(k, 0.0)) for k in ("prep_s", "cache_s", "test_cache_s"))
        core = sum(float(t.get(k, 0.0)) for k in ("train_s", "dev_eval_s", "save_s", "test_eval_s"))
        fm = {"format": "cortex_v12/final_metrics/1", "program": PROGRAM, "version": VERSION, "run": self.dir.name, "utc": utc_now(),
              "seed": self.seed, "selected_epoch": best["epoch"], "completed_epochs": int(tr["epoch"]),
              "selection_rule_ko": "dev 정확도 최대 → 동률이면 dev CE 최소 → 다시 동률이면 이른 epoch (epoch 0 포함). test 는 선택 뒤 한 번만.",
              "dev_at_selected": {"accuracy": best["dev_accuracy"], "ce": best["dev_ce"], "confusion": best["dev_confusion"]},
              "test": test, "test_n": test["n"], "test_accuracy": test["accuracy"], "test_ce": test["ce"],
              "test_images_first_read_phase": self.state.get("test", {}).get("first_read_phase"),
              "time_s": {k: round(float(v), 2) for k, v in t.items()},
              "time_total_s": round(prep_cache + core, 2), "time_without_prep_and_cache_s": round(core, 2),
              "stem_source": self.cfg["stem_source"], "stem_unchanged": bool(stem_ok), "stem_record": self.state["stem"],
              "world": self.state.get("world"), "backend": self.be.describe(), "program_sha256": program_sha(),
              "settings": self.S, "engine": self.engine.structure(),
              "notes_ko": ["CPU NumPy 경로는 v11 NumPy 경로와 같은 연산 순서다. CUDA 결과는 부동소수 축약 순서 때문에 CPU 와 비트 단위로 같지 않을 수 있다.",
                           "시간은 이 PC 에서 잰 값이다 (준비·캐시 포함 / 제외 따로)."]}
        write_json(self.dir / "final_metrics.json", fm)
        self.state["phase"] = "done"
        self.state["test"]["result"] = {k: test[k] for k in ("n", "accuracy", "ce")}
        self.save()
        say(f"  선택 epoch {best['epoch']} (dev {fmt_pct(best['dev_accuracy'])} · CE {best['dev_ce']:.4f}) → test 정확도 {fmt_pct(test['accuracy'])} · "
            f"CE {test['ce']:.4f} (n = {test['n']:,})")
        say(f"  시간: 전체 {fmt_s(prep_cache + core)} (준비·캐시 {fmt_s(prep_cache)} 포함), 준비·캐시 제외 {fmt_s(core)}. 결과: {self.dir / 'final_metrics.json'}")


# ======================================================================
# run 폴더 만들기·열기 (재개 전 동일성 확인)
# ======================================================================
def build_settings(seed: int, epochs: int, batch: int) -> dict[str, Any]:
    s = json.loads(dumps(SETTINGS, indent=None))
    s.update(seed=int(seed), epochs=int(epochs), batch_size=int(batch))
    if s["seed"] < 0 or s["epochs"] < 1 or s["batch_size"] < 1:
        raise V12Error("seed ≥ 0, 주 학습 epoch ≥ 1, 배치 크기 ≥ 1 이어야 한다.")
    return s


def unique_run_dir(out_root: Path) -> Path:
    base = out_root / f"run_{local_stamp()}"
    p, k = base, 2
    while p.exists():
        p, k = Path(f"{base}_{k}"), k + 1
    return p


def create_run(out_root: Path, mnist_root: Path, snapshot: str | None, device: str, seed: int, epochs: int, batch: int) -> Run:
    settings = build_settings(seed, epochs, batch)
    model = json.loads(dumps(MODEL, indent=None))
    be = make_backend(device, int(settings["compute"]["max_gather_elements"]))
    say(f"  계산 장치: {be.kind}/{be.device}" + ("" if be.kind == "numpy" else f" ({be.describe().get('cuda_device')})"))
    cfg = {"format": RUN_FORMAT, "program": PROGRAM, "version": VERSION, "program_sha256": program_sha(), "created_utc": utc_now(),
           "settings": settings, "model": model, "mnist_root": str(mnist_root), "device": "cuda" if be.is_cuda else "cpu",
           "stem_source": {"kind": "fresh_prep"} if not snapshot else {"kind": "v11_common_init_snapshot", "path": str(snapshot)}}
    run = Run(unique_run_dir(out_root), cfg, be, mnist_root)
    say("  MNIST 확인 (MD5) · 분할 · 구조 생성 …")
    run.load_world()
    world = run.world_record()
    le = world["learnable_edges"]
    say(f"  분할 train {world['n']['train']:,} / dev {world['n']['dev']:,} / test {world['n']['test']:,} · 뉴런 {world['N']:,} · "
        f"캐시 차원 (V2 L3) {world['cache_dim']:,} · IT L3 {world['feature_dim']} · V4·IT 학습 연결 {sum(le.values()):,} (V2→V4 {le['V4/L4']:,})")
    snap = None
    if snapshot:
        say(f"  공통 초기 snapshot 확인: {snapshot}")
        snap = load_v11_snapshot(snapshot, run.seed, settings, model)
        snap["verified"] = verify_v11_snapshot(snap, run.seed, run.data, run.specs, model)
        cfg["stem_source"] = {**snap["source"], "verified": snap["verified"], "target_program": PROGRAM, "target_version": VERSION,
                              "target_source_sha256": cfg["program_sha256"],
                              "note_ko": "원본 v11 과 이 프로그램의 소스 SHA256 은 다른 것이 정상이다 (따로 기록). 원본 run 은 읽기만 했다."}
        say(f"  snapshot 검증 통과: {snap['source']['run_name']} seed {run.seed} (분할·구조·q·임계값·head·초기 상태 해시 일치, V4·IT q = 1)")
    ensure_dir(run.dir)
    run._book_load()
    write_json(run.dir / "config.json", cfg)
    run.write_split_ids()
    run.state = {"format": STATE_FORMAT, "program": PROGRAM, "version": VERSION, "run": run.dir.name, "created_utc": utc_now(),
                 "phase": "cache" if snapshot else "prepare_v1", "seq": 0, "world": world, "time": {}, "events": [], "test": {},
                 "identity": {"program_sha256": cfg["program_sha256"], "backend": be.numeric_id(), "run": run_identity(cfg)}}
    run.stop = None
    if snap is not None:
        a = snap["arrays"]
        run.write_stem(np.asarray(a["q"], np.float32), np.asarray(a["threshold"], np.float32), np.asarray(a["W0"], np.float64),
                       np.asarray(a["b0"], np.float64), cfg["stem_source"], None)
    run.save()
    say(f"  새 run 폴더: {run.dir}")
    return run


def read_run(p: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    cfg = read_json(p / "config.json")
    if cfg.get("format") != RUN_FORMAT:
        raise V12Error(f"{p} 는 이 프로그램의 run 폴더가 아니다 (config.json 형식 {cfg.get('format')!r}).")
    lp = p / "last_params.npz"
    if not lp.is_file():
        raise V12Error(f"{p} 에 last_params.npz 가 없다 (만들다가 멈춘 run). 새 run 으로 시작한다.")
    arrays, state = load_npz(lp)
    if not state or state.get("format") != STATE_FORMAT:
        raise IntegrityError(f"{lp} 의 상태 형식이 다르다 (다른 프로그램의 상태를 이어하기 상태로 읽지 않는다).")
    return cfg, state, arrays


def open_run(p: Path, mnist_root: Path | None = None) -> Run:
    cfg, state, arrays = read_run(p)
    probs = []
    if state["identity"]["program_sha256"] != program_sha():
        probs.append("소스 파일이 run 을 만든 때와 다르다 (SHA256). 같은 소스로 이어 가야 연속 실행과 같은 결과가 보장된다")
    if run_identity(cfg) != state["identity"]["run"]:
        probs.append("config.json 의 설정이 저장 상태와 다르다")
    if probs:
        raise V12Error(f"{p} 를 이어 갈 수 없다 — " + "; ".join(probs) + ". 새 run 으로 시작한다.")
    be = make_backend(cfg["device"], int(cfg["settings"]["compute"]["max_gather_elements"]))
    if be.numeric_id() != state["identity"]["backend"]:
        raise V12Error(f"계산 뒷단·버전 {be.numeric_id()} 이 run 의 {state['identity']['backend']} 과 다르다. 같은 장치·라이브러리 버전에서 이어 간다 "
                       "(바꾸면 연속 실행과 같은 수치를 보장할 수 없다).")
    run = Run(p, cfg, be, Path(mnist_root) if mnist_root else Path(cfg["mnist_root"]))
    run.state, run.arrays = state, arrays
    run._book_load()
    run.load_world()
    w_now, w_old = json.loads(dumps(run.world_record(), indent=None)), json.loads(dumps(state["world"], indent=None))
    if w_now != w_old:
        diff = [k for k in w_old if w_now.get(k) != w_old.get(k)]
        raise V12Error(f"데이터 분할·구조가 저장 기록과 다르다 ({', '.join(diff)}). 이어 붙이지 않는다.")
    if (p / "last_state.json").is_file():
        with contextlib.suppress(Exception):
            if read_json(p / "last_state.json").get("seq") != state.get("seq"):
                write_json(p / "last_state.json", state)
    if mnist_root and str(mnist_root) != cfg["mnist_root"]:
        state.setdefault("events", []).append({"utc": utc_now(), "mnist_root_changed_to": str(mnist_root), "verified": "MD5·분할 해시 같음"})
    return run


def classify_folder(p: Path) -> tuple[str, Any]:
    """'run' (이 프로그램의 run 폴더), 'foreign' (다른 프로그램·버전의 실험 폴더), 'parent' (그 안에 새 run 을 만든다)."""
    if p.exists() and not p.is_dir():
        return "file", None
    if (p / "config.json").is_file() or (p / "last_params.npz").is_file():
        try:
            return "run", read_run(p)
        except (V12Error, OSError, ValueError) as exc:
            own = False
            with contextlib.suppress(Exception):
                own = read_json(p / "config.json").get("format") == RUN_FORMAT
            if own:
                return "foreign", f"이 프로그램의 run 폴더지만 저장 상태를 이어 갈 수 없다 — {exc} 새 run 은 상위 폴더를 결과 폴더로 준다."
            return "foreign", f"이 프로그램의 run 폴더가 아니다 ({exc})"
    if (p / "manifest.json").is_file():
        return "foreign", "v10·v11 실험 run 폴더다 (manifest.json). 결과 폴더로 쓰지 않는다 — 공통 초기 snapshot 이면 그 칸에 입력한다."
    runs = []
    if p.is_dir():
        for c in sorted(p.iterdir()):
            if c.is_dir() and c.name.startswith("run_") and (c / "last_params.npz").is_file():
                with contextlib.suppress(Exception):
                    cfg, st, _a = read_run(c)
                    if st.get("phase") != "done":
                        runs.append((c, st))
    return "parent", runs


def describe_state(st: dict[str, Any], cfg: dict[str, Any]) -> str:
    ph = st.get("phase")
    s = cfg["settings"]
    if ph in ("prepare_v1", "prepare_v2"):
        g = (st.get("prep") or {}).get("stage") or {}
        pos = (f"{g.get('area')} epoch {int(g.get('epochs_completed', 0)) + 1}/{g.get('max_epochs')} · 배치 {g.get('next_batch')}" if g
               else "준비 입력 파일·prototype 초기화")
        return f"{ph} — {pos}"
    if ph == "cache":
        return "cache — V2 L3 캐시를 만든 행 다음부터"
    if ph == "train":
        tr = st.get("train") or {}
        b = tr.get("best") or {}
        e = int(tr.get("epoch", -1))
        if e < 0:
            return "train — epoch 0 (학습 전 dev 평가) 부터"
        return (f"train — epoch {e + 1}/{s['epochs']} 배치 {tr.get('next_batch')} 부터 (완료 {e} epoch, best epoch {b.get('epoch')} · "
                f"dev {fmt_pct(b.get('dev_accuracy'))})")
    if ph == "evaluate":
        return "evaluate — 선택 상태로 test 평가"
    if ph == "done":
        r = (st.get("test") or {}).get("result") or {}
        return f"done — 끝난 run (test 정확도 {fmt_pct(r.get('accuracy'))})"
    return str(ph)


def cache_estimate(settings: dict[str, Any], model: dict[str, Any]) -> str:
    oris = [float(t) for t in model["areas"]["V1"]["orientations_deg"]]
    pairs = set()
    for diff in model["areas"]["V2"]["cross_diffs_deg"]:
        for i in range(len(oris)):
            j = _ori_index(oris, oris[i] + float(diff))
            if j is not None and j != i:
                pairs.add((min(i, j), max(i, j)))
    dim = int(model["areas"]["V2"]["n_r"]) * int(model["areas"]["V2"]["n_phi"]) * (3 * len(oris) + len(pairs))
    lim = settings["data"]["limits"]
    n_dev = int(settings["data"]["mnist_dev_size"])
    ns = {"train": 60000 - n_dev, "dev": n_dev, "test": 10000}
    ns = {k: (min(v, int(lim.get(k) or 0)) if int(lim.get(k) or 0) else v) for k, v in ns.items()}
    return (f"train {ns['train']:,}×{dim:,} = {fmt_bytes(ns['train'] * dim * 4)}, dev {ns['dev']:,}×{dim:,} = {fmt_bytes(ns['dev'] * dim * 4)} "
            f"(test {fmt_bytes(ns['test'] * dim * 4)} 는 학습·선택 뒤)")


# ======================================================================
# 단일 실행 화면 (하위 메뉴 없음) 과 명령 줄
# ======================================================================
class QuitRequested(Exception):
    pass


def ask(label: str, default: str = "") -> str:
    shown = f" [{default}]" if default else ""
    try:
        s = input(f"{label}{shown}: ").strip()
    except EOFError as exc:
        raise V12Error("입력이 닫혔다 (명령 줄 인자와 --yes 로도 실행할 수 있다).") from exc
    if s.lower() in ("q", "quit", "exit", "종료"):
        raise QuitRequested
    return s or default


def ask_int(label: str, default: int, lo: int) -> int:
    while True:
        s = ask(label, str(default))
        try:
            v = int(s)
            if v >= lo:
                return v
        except ValueError:
            pass
        say(f"  {lo} 이상의 정수를 입력한다.")


def ensure_mnist(root: Path, interactive: bool) -> None:
    miss = mnist_missing(root)
    if not miss:
        return
    say(f"  MNIST 공식 파일이 없다: {', '.join(miss)}\n  찾은 위치: {root}, {root / 'raw'}, {root / 'MNIST' / 'raw'}")
    if not SETTINGS["data"]["download"]:
        raise DataError(f"공식 gz 4 개를 {root} 에 복사한다 (주소 {MNIST_MIRROR}). 합성 데이터로 대신하지 않는다.")
    if interactive:
        a = ask(f"  공식 주소 ({MNIST_MIRROR}) 에서 내려받을까? (Y/n)", "Y")
        if a.lower() not in ("y", "yes", "예", "ㅛ"):
            raise DataError(f"MNIST 가 없어 시작하지 않는다. 공식 gz 4 개를 {root} 에 복사한 뒤 다시 실행한다.")
    root.mkdir(parents=True, exist_ok=True)
    mnist_download(root, miss)


def screen(args: argparse.Namespace) -> Callable[[], Run] | None:
    here = Path(__file__).resolve().parent
    bar = "─" * 64
    say(bar)
    say(f" 고정 V1·V2 + V4·IT 흔적 학습   ({PROGRAM} v{VERSION})")
    say(bar)
    say(" 값을 입력하고 Enter (비우면 [ ] 안의 값). 언제든 Q: 종료")
    try:
        while True:
            mnist = clean_user_path(ask("MNIST 경로", args.mnist or str(here / "mnist_data")))
            try:
                ensure_mnist(mnist, True)
                break
            except DataError as exc:
                say(f"  [MNIST] {exc}")
        while True:
            out = clean_user_path(ask("결과 폴더 또는 재개할 폴더", args.out or str(here / "runs_v12")))
            kind, info = classify_folder(out)
            if kind == "run":
                cfg, st, _a = info
                s = cfg["settings"]
                say(f"  이 프로그램의 run 폴더다: {out.name}")
                say(f"    저장 설정: seed {s['seed']} · 주 학습 {s['epochs']} epoch · 배치 {s['batch_size']} · 장치 {cfg['device']} · "
                    f"V1·V2 {'v11 snapshot ' + str(cfg['stem_source'].get('run_name')) if cfg['stem_source']['kind'] != 'fresh_prep' else '새 준비 학습'}")
                say(f"    재개 위치: {describe_state(st, cfg)}")
                if st.get("phase") == "done":
                    say(f"  이미 끝난 run 이다 (결과: {out / 'final_metrics.json'}). 새 run 은 상위 폴더를 결과 폴더로 준다.")
                    continue
                say("  (재개할 때는 저장된 seed·epoch·배치·장치·V1·V2 출처를 그대로 쓴다)")
                ask("Enter: 실행 (이어하기)    Q: 종료", "")
                return lambda out=out, mnist=mnist: open_run(out, mnist)
            if kind in ("foreign", "file"):
                say(f"  [결과 폴더] {info or '파일이다 (폴더를 준다)'}")
                continue
            if info:
                say("  이 폴더 안에 끝나지 않은 run 이 있다:")
                for k, (c, st) in enumerate(info, 1):
                    say(f"    {k}) {c.name}: {st.get('phase')}")
                pick = ask("  이어 갈 번호 (Enter = 새 run)", "")
                if pick.isdigit() and 1 <= int(pick) <= len(info):
                    rd = info[int(pick) - 1][0]
                    cfg, st, _a = read_run(rd)
                    say(f"    재개 위치: {describe_state(st, cfg)}")
                    ask("Enter: 실행 (이어하기)    Q: 종료", "")
                    return lambda rd=rd, mnist=mnist: open_run(rd, mnist)
            break
        snap = ask("공통 초기 snapshot 폴더 [비우면 V1·V2 준비 학습]", args.snapshot or "")
        device = ask("장치 [자동 / CPU / CUDA]", args.device or "자동")
        seed = ask_int("seed", args.seed if args.seed is not None else int(SETTINGS["seed"]), 0)
        epochs = ask_int("주 학습 epoch", args.epochs if args.epochs is not None else int(SETTINGS["epochs"]), 1)
        batch = ask_int("배치 크기", args.batch if args.batch is not None else int(SETTINGS["batch_size"]), 1)
        st_ = build_settings(seed, epochs, batch)
        say(bar)
        say(f"  새 run → {out}{os.sep}run_<시각>")
        say(f"  V1·V2: {'v11 공통 초기 snapshot ' + str(clean_user_path(snap)) if snap else '새 준비 학습 (V1 ' + str(SETTINGS['prep']['epochs']['V1']) + ' · V2 ' + str(SETTINGS['prep']['epochs']['V2']) + ' epoch 한도, patience ' + str(SETTINGS['prep']['patience']) + ')'}")
        say(f"  장치 {device} · seed {seed} · 주 학습 {epochs} epoch · 배치 {batch} · η_head {SETTINGS['eta_head']} · η_edge {SETTINGS['eta_edge']} · B {SETTINGS['budget']}")
        say(f"  캐시 (float32): {cache_estimate(st_, MODEL)}")
        say(bar)
        ask("Enter: 실행    Q: 종료", "")
        return lambda: create_run(out, mnist, str(clean_user_path(snap)) if snap else None, device, seed, epochs, batch)
    except QuitRequested:
        return None


def plan_from_args(args: argparse.Namespace) -> Callable[[], Run]:
    if not args.mnist or not args.out:
        raise V12Error("--yes 로 실행하려면 --mnist 와 --out 이 필요하다.")
    mnist, out = clean_user_path(args.mnist), clean_user_path(args.out)
    ensure_mnist(mnist, False)
    kind, info = classify_folder(out)
    if kind == "run":
        if info[1].get("phase") == "done":
            raise V12Error(f"{out} 는 이미 끝난 run 이다.")
        say(f"  재개: {out} — {describe_state(info[1], info[0])}")
        return lambda: open_run(out, mnist)
    if kind != "parent":
        raise V12Error(f"결과 폴더로 쓸 수 없다: {info}")
    seed = args.seed if args.seed is not None else int(SETTINGS["seed"])
    epochs = args.epochs if args.epochs is not None else int(SETTINGS["epochs"])
    batch = args.batch if args.batch is not None else int(SETTINGS["batch_size"])
    return lambda: create_run(out, mnist, args.snapshot or None, args.device or "auto", seed, epochs, batch)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROGRAM, description="고정 V1·V2 + V4·IT 흔적 학습 (인자 없이 실행하면 실행 화면을 연다)")
    p.add_argument("--mnist", help="MNIST 공식 gz 폴더")
    p.add_argument("--out", help="결과 폴더 (새 run 을 만들 곳) 또는 이어 갈 run 폴더")
    p.add_argument("--snapshot", help="v11 장기 비교 run 폴더 (공통 초기 snapshot). 비우면 V1·V2 준비 학습")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), help="계산 장치 (기본 auto)")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int, help="주 학습 epoch (기본 100)")
    p.add_argument("--batch", type=int, help="배치 크기 (기본 16)")
    p.add_argument("--yes", action="store_true", help="화면 없이 바로 실행 (재개 폴더면 이어 간다)")
    return p


_CURRENT: dict[str, Any] = {"run": None}


def main(argv: Sequence[str] | None = None) -> int:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(errors="replace")                 # 한글 외 기호를 못 쓰는 콘솔에서도 멈추지 않게
    if np is None or Image is None:
        print("필수 패키지가 없다: " + ", ".join(n for n, m in (("numpy", np), ("pillow", Image)) if m is None)
              + "\n  설치: pip install numpy pillow   (CUDA 를 쓰려면 PyTorch 도 — 파일 머리말 참고)")
        return 3
    args = build_parser().parse_args(argv)
    try:
        job = plan_from_args(args) if args.yes else screen(args)
        if job is None:
            say("종료.")
            return 0
        run = job()
        _CURRENT["run"] = run
        say(f"\n시작 위치: {run.position_text()}")
        run.execute()
        say("\n완료.")
        return 0
    except Interrupted as exc:
        rd = getattr(_CURRENT.get("run"), "dir", None)
        say(f"\n[멈춤] {exc}.\n  다시 실행해서 '결과 폴더 또는 재개할 폴더' 에 {rd} 를 주면 이 위치에서 이어 간다.")
        return 2
    except KeyboardInterrupt:
        rd = getattr(_CURRENT.get("run"), "dir", None)
        say(f"\n[즉시 중단] 마지막 저장 지점은 그대로다. 이어 가려면 {rd or '그 run 폴더'} 를 다시 준다.")
        return 2
    except (V12Error, OSError) as exc:
        say(f"\n[오류] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
