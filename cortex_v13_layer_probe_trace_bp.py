#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
층별 판독과 TRACE / 정확 BP 비교 — cortex_v13_layer_probe_trace_bp.py (버전 13.0.0)

[목적] 검증된 v12 결과 (동결 V1·V2 의 V2/L3 캐시) 를 입력으로 받아 정확도 제한 원인 후보를 분리하는 진단 전용 프로그램이다.
  ① 같은 시작 상태 (v12 frozen_stem 의 q·threshold·W0·b0, gain = 1) 에서 같은 미니배치·학습률로
     TRACE (v12 규칙) 와 BP_FULL (같은 CE 를 gain 잔차에 대해 연쇄법칙으로 미분한 SGD) 를 각각 학습한다.
  ② INITIAL · TRACE_BEST · BP_FULL_BEST 세 상태의 7 개 출력 (V2/L3, V4/L4, V4/L2, V4/L3, IT/L4, IT/L2, IT/L3) 에
     같은 계열의 선형 판독기 (표준화 + L2 penalty + float64 L-BFGS) 를 학습한다 (V2/L3 공유 → 19 개).
  ③ 두 best 의 원래 IT head 만 raw IT/L3 특징으로 다시 최적화한다 (RAW_IT_HEAD_REFIT, 2 개).
  ④ 같은 상태·같은 배치에서 TRACE 후보와 정확 gradient 의 방향·크기를 비교한다 (gain-only / joint 분리).
  ⑤ 모든 dev 선택을 잠근 뒤 (selection_lock.json) test 를 평가하고 한국어 요약을 쓴다.
  판독 정확도는 그 표현의 '선형 판독 가능성' 측정값이며 정보량·인과적 손실률이 아니다. 한 실험으로 원인을 확정하지 않는다.

[필수 패키지] — 자동 설치하지 않는다
  Python 3.10 이상, NumPy, PyTorch (정확한 BP 와 L-BFGS 에 필수, CPU 또는 CUDA)
      pip install numpy
      PyTorch: https://pytorch.org 의 설치 안내에서 자기 환경 (CUDA 버전) 명령을 고른다. 예) pip install torch
  선택: Matplotlib (그림). 없어도 CSV·JSON·한국어 TXT 요약은 만든다.   pip install matplotlib

[입력: v12 결과] cortex_v12_frozen_v1v2_trace_v4it.py 12.0.0 (SHA256 8e250aff…) 이 만든 run 폴더 또는 그 run 이 든 ZIP
  필요한 파일: config.json · frozen_stem.npz · split_ids.npz · cache_manifest.json · cache_train.npy · cache_dev.npy ·
  cache_test.npy (최종 평가 때만) · best_params.npz · best_state.json (메뉴 2) · last_state.json · final_metrics.json (출처·참고 지표)
  예) C:\\Users\\<사용자>\\PyCharmMiscProject\\runs_v12.zip  또는  압축을 푼 runs_v12\\run_<시각> 폴더
  폴더 입력은 읽기 전용으로 memmap 참조한다. ZIP 은 필요한 파일만 새 결과 폴더의 _input_cache 에 한 번 푼다 (test 는 마지막에).
  원본 ZIP·폴더·v12 상태는 고치거나 지우지 않는다. v12 .py 는 실행에 필요 없다 (import 하지 않는다).

[PyCharm 에서 실행]
  이 파일을 열고 Run → 화면에서 v12 결과 경로와 저장 폴더를 확인하고 메뉴 1 (Enter) 을 고르면
  입력 검증 → 공통 초기값 → TRACE / BP_FULL 학습 → 층별 판독 → raw head 재학습 → 같은 상태 교정 진단
  → 선택 잠금 → test → 요약 (SUMMARY_KO.txt) 까지 진행한다.
  메뉴 2: 기존 v12 best 한 상태의 층별 판독 7 개 + raw IT head 재학습만 (새 TRACE/BP 비교가 아니다)
  메뉴 3: TRACE/BP 학습과 교정 진단만 (판독기는 실행하지 않는다)      메뉴 4: 미완료 v13 실행 이어하기      메뉴 5: 저장 결과 요약 보기
  명령 줄: python cortex_v13_layer_probe_trace_bp.py --input <v12 run 또는 ZIP> [--out <폴더>] [--menu 1] [--device auto|cpu|cuda] [--yes]
           python cortex_v13_layer_probe_trace_bp.py --resume <v13 run 폴더>         python cortex_v13_layer_probe_trace_bp.py --self-test

[중단과 재개]
  Ctrl+C 한 번: 진행 중 배치 (판독기는 진행 중 L-BFGS step) 를 마친 안전한 경계에서 저장하고 멈춘다. L-BFGS line search 중이면 직전 수락
  상태로 되돌린 뒤 저장한다 (미완료 trial 을 저장하지 않는다). 두 번: 즉시 멈춘다 (마지막 저장 지점은 그대로).
  다시 Run → 메뉴 4 → v13 run 폴더를 주면 완료 작업 목록 다음부터 이어 간다. 소스·입력·설정·장치·라이브러리 버전이 다르면 이어 붙이지 않는다.

[작업량] 기본 피질 학습 2 회 (TRACE·BP_FULL 각 100 epoch × 3,125 배치) + 판독기 19 개 + raw head 재학습 2 개 + 진단 3 상태.
  짧은 실행 시간을 보장하지 않는다. 실행 전 화면에 작업 목록과 필요한 디스크를 표시한다.

[주의]
  · 구조 확인: V4 곡선 연결의 기하 경계 비교 (정확히 0 근처) 는 NumPy·수학 라이브러리 판에 따라 달라질 수 있다. 다시 만든 V4 구조 해시가
    v12 기록과 다르면 조용히 맞추지 않고 멈춘다 → v12 를 실행한 PC 의 같은 NumPy 판 (cache_manifest.json 의 backend.numpy) 에서 실행한다.
  · 결정성: TF32·AMP 를 끄고 결정적 알고리즘을 요청한다 (warn_only). 결정적 구현이 없는 연산의 경고는 state.json 과 요약에 기록한다.
    같은 seed 라도 CPU/CUDA·PyTorch 판이 다르면 비트 단위로 같지 않을 수 있다. 이어하기는 같은 장치·판·소스에서만 한다.
  · 그림: Matplotlib 이 있고 한글 글꼴 (Malgun Gothic 등) 이 있으면 한글, 없으면 영문 표기로 그린다.
  · --self-test: 합성 입력으로 미분 관례·gradcheck·VJP·TRACE 배분 예 (A = 1, B = 10)·판독기 수학·예산 복원·이어하기를 검사한다 (임시 폴더).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as _dt
import functools
import gc
import hashlib
import io
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

try:
    import numpy as np
except ImportError as _exc:                       # main() 에서 설치 방법을 안내한다 (import 만으로는 아무것도 하지 않는다)
    np = None  # type: ignore[assignment]
    _NUMPY_ERROR = str(_exc)
else:
    _NUMPY_ERROR = ""
try:
    import torch
    import torch.nn.functional as F
except ImportError as _exc:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_ERROR = str(_exc)
else:
    _TORCH_ERROR = ""

VERSION = "13.0.0"
PROGRAM = "cortex_v13_layer_probe_trace_bp"
NUMERIC_IMPL = "v13-torch-engine-1"              # 수치 구현 판 (캐시 키에 들어간다)

# ======================================================================
# 설정 — 바꿀 값은 여기서만 (실행 화면 입력이 epochs·batch_size·probe 예산을 덮는다)
# ======================================================================
SETTINGS: dict[str, Any] = {
    "epochs": 100, "batch_size": 16,             # 두 조건 공통 (같은 미니배치)
    "eta_head": 0.01, "eta_edge": 0.001,         # 같은 학습률·가산 갱신 (momentum·Adam·weight decay·clipping·AMP 없음)
    "gain_min": 0.05, "gain_max": 4.0,           # γ ∈ [0.05, 4] ↔ 잔차 res ∈ [−0.95, 3] (적용 뒤 투영)
    "probe": {
        "max_iter": 100,                         # 수락된 L-BFGS iteration 최대
        "max_evals": 300,                        # train 전체 closure 평가 최대 (line search·확인 평가 포함)
        "history_size": 10, "lr": 1.0, "line_search_evals": 25,   # strong-Wolfe line search 한 번의 평가 상한 (PyTorch 기본값과 같다)
        "penalty": 1e-4,                         # 목적함수 = mean CE + (penalty/2)·ΣW² (bias 제외). raw head 재학습은 0
        "sigma_floor": 1e-6, "microbatch": 512, "grad_tol": 1e-5, "tolerance_change": 1e-9,
    },
    "diagnostics": {"per_class": 10, "batch_size": 16, "rng_tag": 4242, "fd_eps": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                    "fd_rtol": 1e-3, "fd_atol": 1e-5, "fd_batches": 2},  # 방향 유한차분 (float64 그림자) 은 앞 2 배치에서
    "compute": {"memory_mb": 1024, "max_gather_elements": 1 << 25,   # v12 와 같은 조각 크기 (같은 연산 순서)
                "feature_batch": 256},                                # 특징 캐시·test forward 묶음 (표본별 값은 묶음과 무관)
    "save": {"every_batches": 200, "min_seconds": 60.0, "progress_every_batches": 500},
}

V12_PROGRAM, V12_VERSION = "cortex_v12_frozen_v1v2_trace_v4it", "12.0.0"
V12_SOURCE_SHA256 = "8e250aff9734586bee8ee58e27fa2c3d5677c44cc5f75eed1112f0d99910bde4"
V12_RUN_FORMAT, V12_STATE_FORMAT = "cortex_v12/run_config/1", "cortex_v12/last_state/1"
V12_STEM_FORMAT, V12_CACHE_FORMAT = "cortex_v12/frozen_stem/1", "cortex_v12/feature_cache/1"
V12_REQUIRED = ("config.json", "frozen_stem.npz", "split_ids.npz", "cache_manifest.json", "cache_train.npy", "cache_dev.npy")
V12_OPTIONAL = ("cache_test.npy", "best_params.npz", "best_state.json", "last_state.json", "final_metrics.json")
EXAMPLE_INPUT = r"C:\Users\ginok\PyCharmMiscProject\runs_v12.zip"

AREAS = ("V1", "V2", "V4", "IT")
LAYERS = ("L4", "L2", "L3")
LEARN_KEYS = ("V4/L4", "V4/L2", "V4/L3", "IT/L4", "IT/L2", "IT/L3")
PROBE_KEYS = ("V2/L3",) + LEARN_KEYS
N_CLASSES = 10
SPLITS = ("train", "dev", "test")
FIXED_MODE = "v4_fixed_geometry"
CONDS = ("TRACE", "BP_FULL")
COND_DIR = {"TRACE": "trace", "BP_FULL": "bp_full"}
STATES = ("INITIAL", "TRACE_BEST", "BP_FULL_BEST")
V12_STATE = "V12_IMPORTED_BEST"
STATES_ALL = STATES + (V12_STATE,)
RUN_FORMAT = "cortex_v13/run_config/1"
STATE_FORMAT = "cortex_v13/state/1"
MENU_SCOPE = {"1": "full", "2": "v12_best_probes", "3": "train_and_diagnostics"}


# ======================================================================
# 공통 도구 (해시는 v12 와 같은 방식 → v12 기록을 그대로 대조한다)
# ======================================================================
class V13Error(RuntimeError):
    """사용자에게 그대로 보여 주는 오류 (원인과 할 일을 담는다)."""


class InputError(V13Error):
    pass


class IntegrityError(V13Error):
    pass


class NumericalError(V13Error):
    pass


class Interrupted(Exception):
    """Ctrl+C: 안전한 경계에서 저장한 뒤 멈춤."""


class BudgetExhausted(Exception):
    """L-BFGS closure 평가 예산을 모두 썼다 (직전 수락 상태로 되돌린다)."""


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
    if torch is not None and isinstance(o, torch.Tensor):
        return json_ready(o.detach().cpu().numpy())
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
            os.chmod(tmp, 0o644)
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


def write_csv(path: Path, rows: Sequence[dict[str, Any]], cols: Sequence[str] | None = None) -> None:
    cols = list(cols) if cols else (list(rows[0]) if rows else [])
    out = io.StringIO()
    out.write(",".join(cols) + "\n")
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if v is None:
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.10g}")
            else:
                s = str(v)
                cells.append('"' + s.replace('"', '""') + '"' if ("," in s or '"' in s) else s)
        out.write(",".join(cells) + "\n")
    atomic_write_bytes(Path(path), out.getvalue().encode("utf-8"))


def clean_user_path(raw: Any) -> Path:
    """탐색기·PyCharm 에서 복사한 경로: 앞뒤 공백·따옴표·보이지 않는 방향 문자를 지우고 ~·환경변수를 푼다 (한글·공백 경로 그대로)."""
    s = str(raw).strip().strip("‪‫‬‭‮﻿").strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    if not s:
        raise V13Error("빈 경로다.")
    return Path(os.path.expandvars(os.path.expanduser(s)))


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    probe = p / f".write_probe_{os.getpid()}_{time.time_ns()}"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        raise V13Error(f"폴더에 쓸 수 없다: {p} ({exc})") from exc
    return p.resolve()


def program_sha() -> str:
    try:
        return sha256_file(Path(__file__))
    except OSError:
        return "unknown"


def fmt_pct(x: Any) -> str:
    return "-" if x is None else f"{100.0 * float(x):.2f}%"


def fmt_f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "-"
    x = float(x)
    if x != 0.0 and (abs(x) < 10 ** (-nd) or abs(x) >= 1e6):
        return f"{x:.{max(1, nd - 1)}e}"
    return f"{x:.{nd}f}"


def fmt_s(sec: float) -> str:
    sec = float(sec)
    if sec < 120:
        return f"{sec:.1f}초"
    h, rem = divmod(int(round(sec)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}시간 {m}분 {s}초" if h else f"{m}분 {s}초"


def fmt_bytes(n: float) -> str:
    n = float(n)
    return f"{n / 1e9:.3f} GB" if n >= 1e9 else f"{n / 1e6:.2f} MB"


class StopFlag:
    """Ctrl+C 첫 번째: 진행 중 배치·step 을 마친 경계에서 저장하고 멈추도록 표시만 한다. 두 번째: 즉시 KeyboardInterrupt.
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
        say("\n[중단 요청] 진행 중 배치·step 을 마친 경계에서 저장하고 멈춘다. 한 번 더 누르면 즉시 멈춘다 (마지막 저장 지점은 그대로).")

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


_NULL_STOP = StopFlag()


# ======================================================================
# PyTorch 계산 정책 (모든 조건이 같은 정책): float32 피질, float64 head·판독기, TF32·AMP 끔, 결정적 알고리즘 요청
# ======================================================================
_DET_WARNINGS: list[str] = []


def torch_setup(choice: str) -> "torch.device":
    c = (choice or "auto").strip().lower()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")       # cuBLAS 결정적 작업 공간 (CUDA 초기화 전에)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with contextlib.suppress(Exception):
        torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)          # 결정적 구현이 없는 연산은 경고 → 기록 (조용히 바꾸지 않는다)
    warnings.filterwarnings("always", message=".*deterministic.*")
    if not getattr(warnings.showwarning, "_v13", False):
        _orig = warnings.showwarning

        def _show(message: Any, category: Any, filename: Any, lineno: Any, file: Any = None, line: Any = None) -> None:
            txt = str(message)
            if "determinis" in txt and len(_DET_WARNINGS) < 50 and txt not in _DET_WARNINGS:
                _DET_WARNINGS.append(txt)
            _orig(message, category, filename, lineno, file, line)

        _show._v13 = True  # type: ignore[attr-defined]
        warnings.showwarning = _show
    if c in ("auto", "자동", ""):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if c == "cpu":
        return torch.device("cpu")
    if c in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise V13Error("CUDA 장치를 쓸 수 없다 (torch.cuda.is_available() = False). 장치를 CPU 로 고르거나 CUDA 용 PyTorch 를 설치한다.")
        return torch.device("cuda")
    raise V13Error(f"장치는 자동 / CPU / CUDA 중 하나다 (입력: {choice!r}).")


def backend_id(device: "torch.device") -> dict[str, Any]:
    """이어하기 동일성·캐시 키에 쓰는 계산 구현 식별값."""
    d = {"device_type": device.type, "torch": torch.__version__, "numpy": np.__version__, "numeric_impl": NUMERIC_IMPL}
    if device.type == "cuda":
        d["cuda_device"] = torch.cuda.get_device_name(device)
        d["cuda_runtime"] = torch.version.cuda
    return d


def backend_describe(device: "torch.device") -> dict[str, Any]:
    d = backend_id(device)
    d.update(tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32), tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
             deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
             deterministic_warn_only=bool(torch.is_deterministic_algorithms_warn_only_enabled()),
             cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"), threads=torch.get_num_threads(),
             grad_enabled_globally=bool(torch.is_grad_enabled()), amp=False, python=sys.version.split()[0])
    return d


def sync(device: "torch.device") -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rng_record(seed: int) -> dict[str, Any]:
    return {"numpy_bit_generator": "PCG64 (default_rng, 목적별 seed 목록)", "torch_initial_seed": int(torch.initial_seed()),
            "train_order": f"default_rng([{seed}, 1010, epoch, 11]).permutation(n) (v12 와 같음)",
            "diagnostic_sample": f"default_rng([{seed}, {SETTINGS['diagnostics']['rng_tag']}, 13]) (학습 순열과 별도)",
            "fd_directions": f"default_rng([{seed}, {SETTINGS['diagnostics']['rng_tag']}, 77, 상태번호])"}


# ======================================================================
# 구조 재구성 — v12 build_v2 · build_v4 · build_it 와 같은 계산 (소스에서 그대로 옮겼다; 구조 해시로 확인한다)
#   V1 은 연결을 만들지 않는다: V2 블록 평균에 필요한 V1/L3 좌표·채널 수와 뉴런 번호 범위만 계산한다.
#   V2 구조는 V4/L4 의 원천 주소 확인과 구조 해시에만 쓰고, 학습·순방향에는 V2 캐시 (V2/L3 최종 출력) 만 쓴다.
# ======================================================================
# v12 기본 모델 (자체 검사와 '기본 구조인지' 표시에만 쓴다. 실제 실험은 입력 run 의 config.json 모델로 구조를 만든다)
V12_MODEL_DEFAULT: dict[str, Any] = {
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



class RetinaGeometry:
    """v12 RetinaLGN 의 좌표 부분만 (영상 크기·중심·로그-극좌표 격자). 영상·DoG·blur 계산은 없다."""

    def __init__(self, model: dict[str, Any]):
        lp = model["logpolar"]
        self.S = S = int(model["image_size"])
        self.cx = self.cy = (S - 1) / 2.0
        self.colors = list(model["areas"]["V1"]["color_channels"])
        self.maps = [f"{c}_{pol}" for c in self.colors for pol in ("on", "off")]
        self.n_maps = len(self.maps)
        self.r0 = float(lp["r0_px"])
        self.rmax = S / 2.0 - float(lp["rmax_margin_px"])
        self.phi_offset = float(lp["phi_offset_bins"])
        self.grid = logpolar_positions(int(lp["n_r"]), int(lp["n_phi"]), self.r0, self.rmax, self.cx, self.cy, self.phi_offset)
        self.n_pos = int(lp["n_r"]) * int(lp["n_phi"])
        self.n_out = self.n_pos * self.n_maps


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
            raise V13Error(f"{self.key}: 구조 배열 길이 오류")
        for g, ptr, src, w in (("A", self.a_ptr, self.a_src, self.a_w), ("B", self.b_ptr, self.b_src, self.b_w)):
            if len(ptr) != n + 1 or ptr[0] != 0 or (np.diff(ptr) < 0).any() or ptr[-1] != len(src) or len(src) != len(w):
                raise V13Error(f"{self.key}: 입력군 {g} CSR 형식 오류")
            if len(src) and (src.min() < 0 or src.max() >= self.n_src):
                raise V13Error(f"{self.key}: 입력군 {g} 주소가 원천 층 범위 밖")
            if not np.isfinite(w).all():
                raise V13Error(f"{self.key}: 입력군 {g} 가중치에 NaN/Inf")
        fa, fb = np.diff(self.a_ptr), np.diff(self.b_ptr)
        inact = self.missing if self.missing is not None else np.zeros(n, bool)
        if (fb[~self.op_min] > 0).any() or ((fa[self.op_min] == 0) | ((fb[self.op_min] == 0) & ~inact[self.op_min])).any():
            raise V13Error(f"{self.key}: min2 뉴런은 입력군 A·B 가 모두 있어야 하고 선형 뉴런에는 B 가 없어야 한다.")
        if self.name == "L3" and ((fa + fb) == 0).any():
            raise V13Error(f"{self.key}: 입력 없는 L3 출력이 있다.")

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
            raise V13Error(f"원천 주소 {src} 가 범위 [0,{self.n_src}) 밖이다.")
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


def _local_spacing(px: np.ndarray, py: np.ndarray, pre: RetinaGeometry, n_r: int, n_phi: int) -> np.ndarray:
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


class PositionMeta:
    """V1/L3 좌표 메타데이터 (v12 LayerSpec 가운데 build_v2 가 읽는 속성만: 격자 모양·채널 수·위치·수용장 반경, float32 로 저장)."""

    def __init__(self, key: str, n_pos: int, n_ch: int, grid_shape: tuple[int, int], pos: tuple[Any, Any, Any]):
        self.key, self.n_pos, self.n_ch, self.grid_shape = key, int(n_pos), int(n_ch), grid_shape
        self.pos_x, self.pos_y, self.pos_rf = (np.asarray(v, np.float32) for v in pos)

    @property
    def n(self) -> int:
        return self.n_pos * self.n_ch


def v1_meta_and_sizes(m: dict[str, Any], pre: RetinaGeometry) -> tuple[PositionMeta, dict[str, int]]:
    """v12 build_v1 의 위치·σ 계산과 층 크기만 (Gabor 연결은 만들지 않는다)."""
    a = m["areas"]["V1"]
    g1 = logpolar_positions(int(a["n_r"]), int(a["n_phi"]), pre.r0, pre.rmax, pre.cx, pre.cy, pre.phi_offset)
    n_pos1 = int(a["n_r"]) * int(a["n_phi"])
    sigma = np.maximum(float(a["gabor_sigma_factor"]) * np.sqrt(g1["spacing_r"] * g1["spacing_phi"]), float(a["gabor_sigma_min_px"]))
    n_c, n_o, n_f, n_s = len(pre.colors), len(a["orientations_deg"]), len(a["phases_rad"]), len(a["polarities"])
    meta = PositionMeta("V1/L3", n_pos1, n_c * n_o, (int(a["n_r"]), int(a["n_phi"])), (g1["x"], g1["y"], sigma))
    return meta, {"V1/L4": pre.n_out, "V1/L2": n_pos1 * n_c * n_o * n_f * n_s, "V1/L3": n_pos1 * n_c * n_o}


# ---------------------------------------------------------------------- V2: 방향 블록 평균 (L4) → 윤곽 연속·끝점·교차 min2 (L2) → 일대일 (L3)
def build_v2(m: dict[str, Any], pre: RetinaGeometry, v1: "LayerSpec | PositionMeta") -> list[LayerSpec]:
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
def curve_candidates(pre: RetinaGeometry, l4: LayerSpec, l2_rf: np.ndarray, n_ch: int, chans: list[tuple[str, int, int]],
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
                    raise V13Error(f"V4 위치 {P} 와 {Q} 가 같은 좌표다.")
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


def build_v4(m: dict[str, Any], pre: RetinaGeometry, v2: LayerSpec) -> list[LayerSpec]:
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
def build_it(m: dict[str, Any], pre: RetinaGeometry, v4: LayerSpec) -> list[LayerSpec]:
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


def build_structure(m: dict[str, Any]) -> tuple[dict[str, LayerSpec], dict[str, list[int]], dict[str, str]]:
    """V2·V4·IT 9 층 구조 + 12 층 뉴런 번호 범위 (v12 NeuronStore 와 같은 연속 번호) + V2·V4·IT 구조 해시 (v12 와 같은 식)."""
    pre = RetinaGeometry(m)
    v1, sizes = v1_meta_and_sizes(m, pre)
    specs: dict[str, LayerSpec] = {}
    for sp in build_v2(m, pre, v1):
        specs[sp.key] = sp
    for sp in build_v4(m, pre, specs["V2/L3"]):
        specs[sp.key] = sp
    for sp in build_it(m, pre, specs["V4/L3"]):
        specs[sp.key] = sp
    sizes.update({k: sp.n for k, sp in specs.items()})
    ranges: dict[str, list[int]] = {}
    N = 0
    for a in AREAS:
        for l in LAYERS:
            k = f"{a}/{l}"
            ranges[k] = [N, N + int(sizes[k])]
            N += int(sizes[k])
    for k, sp in specs.items():
        sp.start, sp.end = ranges[k]
    hashes = {a: area_hash([specs[f"{a}/{l}"] for l in LAYERS]) for a in ("V2", "V4", "IT")}
    return specs, ranges, hashes


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
# 비매끄러운 점의 미분 관례 (BP 그래프에서만 쓰는 autograd 함수). 순방향 값은 v12 PyTorch 경로와 같은 연산이다.
#   정확 BP = 매끄러운 구간의 정확한 연쇄법칙 + 아래에 명시한 경계 subgradient (float32 반올림의 이산 함수 미분이 아니다).
# ======================================================================
CONVENTIONS: dict[str, str] = {
    "relu": "뉴런 s = max(0, u − θ): ∂s/∂u = 1[u − θ > 0] (정확히 0 이면 0 — TRACE 의 1[s > 0] 와 같은 관례). "
            "torch.clamp 의 기본 미분 관례에 기대지 않고 직접 정의했다.",
    "min": "u = min(A, B): 작은 가지 1 · 큰 가지 0 · 정확한 동률이면 0.5 / 0.5",
    "l6_clip": "L6 g = clip(1 + κ(r − 1), 0.5, 2): 범위 밖 0 · 안 1 · 정확한 경계 (0.5 또는 2) 1",
    "l6_rect": "L6 이웃 합에서 자기 값을 뺀 값 a 의 음수 반올림 잔여 정류 max(0, a): a ≥ 0 이면 1 (0 포함: 자기 제외 이웃 합은 수학적으로 "
               "이웃 값의 선형 합이므로), a < 0 (반올림 잔여를 0 으로 바꾼 곳) 이면 0",
    "l6_full": "BP_FULL 은 L6 g 의 자기 항·이웃 항 미분을 모두 포함한다 (g 를 detach 하지 않는다). BP_STOP_L6 (진단 전용) 만 g 를 고정한다.",
    "gamma": "γ = float32(1 + res), w = w0·γ: ∂γ/∂res = 1, ∂w/∂res = w0 (float32 반올림 자체는 미분하지 않는다)",
    "inactive": "구조상 비활성 뉴런 (V4/L2 의 합법 B 원천 없음, u = s = 0 으로 고정) 과 ELL 채움 칸: 미분 0",
    "v2_input": "V2/L3 캐시는 상수 입력 (detach). V1·V2 는 학습하지 않는다.",
}

if torch is not None:
    class _Relu0(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any) -> Any:
            ctx.save_for_backward(x)
            return torch.clamp(x, min=0.0)

        @staticmethod
        def backward(ctx: Any, g: Any) -> Any:
            (x,) = ctx.saved_tensors
            return g * (x > 0).to(g.dtype)

    class _RectNb(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any) -> Any:
            ctx.save_for_backward(x)
            return torch.clamp(x, min=0.0)

        @staticmethod
        def backward(ctx: Any, g: Any) -> Any:
            (x,) = ctx.saved_tensors
            return g * (x >= 0).to(g.dtype)

    class _Min2(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, a: Any, b: Any) -> Any:
            ctx.save_for_backward(a, b)
            return torch.minimum(a, b)

        @staticmethod
        def backward(ctx: Any, g: Any) -> Any:
            a, b = ctx.saved_tensors
            eq = (a == b).to(g.dtype) * 0.5
            return g * ((a < b).to(g.dtype) + eq), g * ((b < a).to(g.dtype) + eq)

    class _ClipIn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any, lo: float, hi: float) -> Any:
            ctx.save_for_backward(x)
            ctx.lo, ctx.hi = lo, hi
            return torch.clamp(x, min=lo, max=hi)

        @staticmethod
        def backward(ctx: Any, g: Any) -> Any:
            (x,) = ctx.saved_tensors
            return g * ((x >= ctx.lo) & (x <= ctx.hi)).to(g.dtype), None, None


def _nograd(fn: Callable[..., Any]) -> Callable[..., Any]:
    """torch.no_grad 를 호출 때만 건다 (PyTorch 가 없어도 import 는 된다; 전역 grad 설정은 바꾸지 않는다)."""
    @functools.wraps(fn)
    def wrapper(*a: Any, **k: Any) -> Any:
        with torch.no_grad():
            return fn(*a, **k)
    return wrapper


def f_relu0(x: Any) -> Any:
    return _Relu0.apply(x) if x.requires_grad else torch.clamp(x, min=0.0)


def f_rect(x: Any) -> Any:
    return _RectNb.apply(x) if x.requires_grad else torch.clamp(x, min=0.0)


def f_min2(a: Any, b: Any) -> Any:
    return _Min2.apply(a, b) if (a.requires_grad or b.requires_grad) else torch.minimum(a, b)


def f_clip_in(x: Any, lo: float, hi: float) -> Any:
    return _ClipIn.apply(x, float(lo), float(hi)) if x.requires_grad else torch.clamp(x, min=float(lo), max=float(hi))


class TT:
    """장치·자료형 묶음. v12 Backend 의 PyTorch 경로와 같은 연산 (gather·원소별 곱·마지막 축 합·cat·where·clamp) 만 쓴다."""

    def __init__(self, device: Any, dtype: Any, max_gather: int):
        self.dev, self.dt, self.max_gather = device, dtype, int(max_gather)
        self.np_dt = np.float32 if dtype == torch.float32 else np.float64

    def f(self, a: Any) -> Any:
        return torch.from_numpy(np.array(a, dtype=self.np_dt, copy=True, order="C")).to(self.dev)

    def i64(self, a: Any) -> Any:
        return torch.from_numpy(np.array(a, dtype=np.int64, copy=True, order="C")).to(self.dev)

    def bool_(self, a: Any) -> Any:
        return torch.from_numpy(np.array(a, dtype=bool, copy=True, order="C")).to(self.dev)

    def zeros(self, shape: Sequence[int]) -> Any:
        return torch.zeros(tuple(int(s) for s in shape), dtype=self.dt, device=self.dev)

    def pad_zero_col(self, x: Any) -> Any:
        return torch.cat([x, self.zeros((int(x.shape[0]), 1))], dim=1)

    def ell(self, xe: Any, idx: Any, w: Any) -> Any:
        """xe (B, n_src+1) (마지막 열 = 0 채움), idx·w (r, K) → (B, r) = Σ_k xe[:, idx[r,k]]·w[r,k] (v12 와 같은 행 조각)."""
        B, r, K = int(xe.shape[0]), int(idx.shape[0]), int(idx.shape[1])
        rows = max(1, self.max_gather // max(1, B * K))
        if rows >= r:
            return (xe[:, idx] * w).sum(-1)
        return torch.cat([(xe[:, idx[a:a + rows]] * w[a:a + rows]).sum(-1) for a in range(0, r, rows)], dim=1)

    def to_np(self, x: Any) -> np.ndarray:
        return x.detach().to("cpu").numpy()


class BucketELL:
    """CSR → 팬인 크기별 ELL 묶음 (v12 BucketELL 과 같은 묶음·열 순서). TRACE 의 전치 배분 (연결 칸 → 원천 뉴런) 에만 쓴다."""

    def __init__(self, tt: TT, ptr: np.ndarray, src: np.ndarray, w: np.ndarray, n_src: int):
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
            self.parts.append((tt.i64(idx), tt.f(ww)))
            perm.append(rows)
        perm_ = np.concatenate(perm)
        if not np.array_equal(perm_, np.arange(n)):
            self.inv = tt.i64(np.argsort(perm_))

    def apply(self, tt: TT, xe: Any) -> Any:
        if not self.parts:
            return tt.zeros((int(xe.shape[0]), self.n))
        outs = [tt.ell(xe, idx, w) for idx, w in self.parts]
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        return out if self.inv is None else out[:, self.inv]


class L6:
    """L6 상대 활동 조절 (v12 와 같다): r_j = CONF_j / (mean_{k∈N(j), k≠j} CONF_k + ε), g_j = clip(1 + κ(r_j − 1), g_min, g_max), 이웃 없음 → 1."""

    def __init__(self, tt: TT, sp: LayerSpec, l6cfg: dict[str, Any]):
        self.n_pos, self.n_ch = sp.n_pos, sp.n_ch
        self.include_self = bool(l6cfg["include_self"])
        self.eps, self.kappa = float(l6cfg["epsilon"]), float(l6cfg["kappa"])
        self.g_min, self.g_max = float(l6cfg["g_min"]), float(l6cfg["g_max"])
        cnt = np.repeat(sp.pos_nb_count * sp.n_ch, sp.n_ch) - (0 if self.include_self else 1)
        self.pos_nb = tt.i64(sp.pos_nb)
        self.has_nb = tt.bool_(cnt > 0)
        self.has_nb_np = cnt > 0
        self.nb_div = tt.f(np.maximum(cnt, 1))

    @staticmethod
    def applies(sp: LayerSpec, l6cfg: dict[str, Any]) -> bool:
        return bool(l6cfg["enabled"]) and sp.name in ("L2", "L3") and sp.name in l6cfg["layers"]

    def gain(self, tt: TT, conf: Any, probe: dict[str, Any] | None = None) -> Any:
        B = int(conf.shape[0])
        pos_sum = conf.reshape(B, self.n_pos, self.n_ch).sum(-1)
        nb_sum = tt.pad_zero_col(pos_sum)[:, self.pos_nb].sum(-1)
        nb_n = torch.repeat_interleave(nb_sum, self.n_ch, dim=1)
        arg = None
        if not self.include_self:
            arg = nb_n - conf
            nb_n = f_rect(arg)
        r = conf / (nb_n / self.nb_div + self.eps)
        x = 1.0 + self.kappa * (r - 1.0)
        g = torch.where(self.has_nb, f_clip_in(x, self.g_min, self.g_max), torch.ones_like(x))
        if probe is not None:
            probe["l6_x"], probe["l6_arg"] = x, arg
        return g


class Part:
    """연결 묶음 하나 (입력군 A/B 의 팬인 크기별 ELL): 주소·w0·잔차 res (γ = 1 + res)."""

    def __init__(self, layer: str, grp: str, i: int, rows: np.ndarray, idx: np.ndarray, w0: np.ndarray, tt: TT):
        self.layer, self.grp, self.i = layer, grp, i
        self.name = f"{layer.replace('/', '.')}.{grp}{i}"
        self.rows = rows
        self.r, self.K = int(idx.shape[0]), int(idx.shape[1])
        self.idx_np, self.w0_np = idx, w0
        self.rows_t, self.idx = tt.i64(rows), tt.i64(idx)
        self.w0 = tt.f(w0)
        self.res = tt.zeros((self.r, self.K))
        self.gam = self.res + 1.0
        self.weff = self.w0 * self.gam
        self.dg: Any = None
        self.off = 0
        self.valid_np = np.zeros((self.r, self.K), bool)       # 유효 연결 칸 (w0 ≠ 0, 수신 뉴런 유효) — GainLayer 가 채운다

    def set_res(self, res: Any) -> None:
        self.res = res
        self.gam = res + 1.0
        self.weff = self.w0 * self.gam


class GainELL:
    """입력군 하나 (A 또는 B) 의 연결 묶음 (v12 GainELL 과 같은 묶음·열 순서)."""

    def __init__(self, tt: TT, layer: str, grp: str, ptr: np.ndarray, src: np.ndarray, w: np.ndarray, n_src: int):
        n = len(ptr) - 1
        self.n, self.n_src, self.edges = n, int(n_src), int(len(src))
        self.parts: list[Part] = []
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
            self.parts.append(Part(layer, grp, pi, rows, idx, ww, tt))
            perm.append(rows)
        perm_ = np.concatenate(perm)
        if not np.array_equal(perm_, np.arange(n)):
            self.inv = tt.i64(np.argsort(perm_))

    def assemble(self, tt: TT, outs: Sequence[Any], B: int) -> Any:
        if not self.parts:
            return tt.zeros((int(B), self.n))
        out = outs[0] if len(outs) == 1 else torch.cat(list(outs), dim=1)
        return out if self.inv is None else out[:, self.inv]

    def apply(self, tt: TT, xe: Any, weffs: Sequence[Any] | None = None) -> Any:
        if not self.parts:
            return tt.zeros((int(xe.shape[0]), self.n))
        ws = [p.weff for p in self.parts] if weffs is None else weffs
        return self.assemble(tt, [tt.ell(xe, p.idx, w) for p, w in zip(self.parts, ws)], int(xe.shape[0]))


class GainLayer:
    def __init__(self, tt: TT, sp: LayerSpec, model: dict[str, Any], q: np.ndarray, thr: np.ndarray, *, first: bool, with_trace: bool):
        self.key, self.src_key, self.n, self.n_src = sp.key, sp.src_key, sp.n, sp.n_src
        self.first = first
        self.A = GainELL(tt, sp.key, "A", sp.a_ptr, sp.a_src, sp.a_w, sp.n_src)
        self.B = GainELL(tt, sp.key, "B", sp.b_ptr, sp.b_src, sp.b_w, sp.n_src) if len(sp.b_src) else None
        self.op_min_np = np.asarray(sp.op_min, bool)
        self.op_min = tt.bool_(sp.op_min) if self.B is not None else None
        inv = np.zeros(self.n, bool) if sp.missing is None else np.asarray(sp.missing, bool).copy()
        self.invalid_np = inv
        self.invalid = tt.bool_(inv) if inv.any() else None
        self.valid_f = tt.f((~inv).astype(np.float32)) if inv.any() else None
        self.q, self.thr = tt.f(q), tt.f(thr)
        self.l6 = L6(tt, sp, model["l6"]) if L6.applies(sp, model["l6"]) else None
        self.F = 0
        nvalid = np.zeros(self.n, np.int64)
        pos_l, src_l, sgn_l = [], [], []
        for ELL, ptr, src, w in ((self.A, sp.a_ptr, sp.a_src, sp.a_w), (self.B, sp.b_ptr, sp.b_src, sp.b_w)):
            if ELL is None or not ELL.parts:
                continue
            for p in ELL.parts:
                p.off = self.F
                self.F += p.r * p.K
                p.valid_np = (p.w0_np != 0) & ~inv[p.rows][:, None]
            row = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(ptr))
            col = np.arange(len(src), dtype=np.int64) - ptr[row]
            pofs = np.asarray([p.off for p in ELL.parts], np.int64)
            pK = np.asarray([p.K for p in ELL.parts], np.int64)
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
        if not first and with_trace:        # 전치 묶음 (원천 뉴런 ← 자기가 보낸 연결 칸, 가중치 sign(w0)). 첫 층 (V4/L4) 은 만들지 않는다
            pos = np.concatenate(pos_l) if pos_l else np.zeros(0, np.int64)
            s_ = np.concatenate(src_l) if src_l else np.zeros(0, np.int64)
            g_ = np.concatenate(sgn_l) if sgn_l else np.zeros(0, np.float32)
            order = np.lexsort((pos, s_))
            t_ptr = np.concatenate([[0], np.cumsum(np.bincount(s_, minlength=self.n_src))]).astype(np.int64)
            self.T = BucketELL(tt, t_ptr, pos[order], g_[order], self.F)

    def groups(self) -> list[tuple[str, GainELL]]:
        return [("A", self.A)] + ([("B", self.B)] if self.B is not None else [])

    def parts(self) -> list[Part]:
        return [p for _nm, ELL in self.groups() for p in ELL.parts]


class Engine:
    """V4·IT 6 층 (입력 = V2/L3 캐시). 순방향 하나 + TRACE 배분 + 정확 BP (VJP). float32 (주 엔진) 또는 float64 (진단용 그림자)."""

    def __init__(self, tt: TT, specs: dict[str, LayerSpec], model: dict[str, Any], q_full: np.ndarray, thr_full: np.ndarray, *,
                 gain_min: float, gain_max: float, memory_mb: int, with_trace: bool = True):
        self.tt = tt
        self.layers = [GainLayer(tt, specs[k], model, q_full[specs[k].start:specs[k].end], thr_full[specs[k].start:specs[k].end],
                                 first=(i == 0), with_trace=with_trace) for i, k in enumerate(LEARN_KEYS)]
        self.by_key = {L.key: L for L in self.layers}
        self.in_dim = self.layers[0].n_src
        self.dim = self.layers[-1].n
        self.lo, self.hi = float(gain_min) - 1.0, float(gain_max) - 1.0
        self.elem_budget = max(1 << 16, int(int(memory_mb) * 2 ** 20 / 16))      # v12 와 같은 표본 조각 상한 (같은 합산 순서)
        self.parts: list[Part] = [p for L in self.layers for p in L.parts()]
        self.layer_parts = {L.key: L.parts() for L in self.layers}
        self.n_learnable = sum(L.n_valid_edges for L in self.layers)

    # ---------------------------------------------------------------- 순방향 (record: none / outs / trace / full)
    def forward(self, x0: Any, *, record: str = "none", leaves: Sequence[Any] | None = None, stop_l6: bool = False) -> dict[str, Any]:
        tt = self.tt
        lay: dict[str, dict[str, Any]] = {}
        prev = x0
        leaf_of = {id(p): k for k, p in enumerate(self.parts)} if leaves is not None else {}
        for L in self.layers:
            xe = tt.pad_zero_col(prev)
            if leaves is None:
                sa = L.A.apply(tt, xe)
                sb = L.B.apply(tt, xe) if L.B is not None else None
            else:                                                                # BP: w = w0·(res + 1) 를 그래프 안에서 (같은 연산 순서)
                sa = L.A.apply(tt, xe, [p.w0 * (leaves[leaf_of[id(p)]] + 1.0) for p in L.A.parts])
                sb = L.B.apply(tt, xe, [p.w0 * (leaves[leaf_of[id(p)]] + 1.0) for p in L.B.parts]) if L.B is not None else None
            if sb is not None:
                u = torch.where(L.op_min, f_min2(sa, sb), sa + sb)
                if L.invalid is not None:
                    u = torch.where(L.invalid, tt.zeros(u.shape), u)
            else:
                u = sa
            pre = u - L.thr
            s = f_relu0(pre)
            if L.invalid is not None:
                s = torch.where(L.invalid, tt.zeros(s.shape), s)
            conf = L.q * s
            probe: dict[str, Any] | None = {} if record == "full" else None
            g = None
            if L.l6 is not None:
                if stop_l6 and conf.requires_grad:
                    with torch.no_grad():
                        g = L.l6.gain(tt, conf.detach(), probe)
                else:
                    g = L.l6.gain(tt, conf, probe)
            out = conf if g is None else g * conf
            if record == "outs":
                lay[L.key] = {"out": out}
            elif record in ("trace", "full"):
                lay[L.key] = {"sa": sa, "sb": sb, "s": s, "g": g, "out": out}
                if record == "full":
                    lay[L.key].update(pre=pre, **(probe or {}))
            prev = out
        return {"x0": x0, "layers": lay, "h": prev}

    # ---------------------------------------------------------------- TRACE (v12 raw TRACE 와 같은 연산; min_aware 는 진단 전용 변형)
    def _lambdas(self, L: GainLayer, r: dict[str, Any]) -> tuple[Any, Any]:
        if L.B is None:
            return None, None
        sa, sb = r["sa"], r["sb"]
        one, zero = torch.ones_like(sa), torch.zeros_like(sa)
        half = one * 0.5
        la = torch.where(sa < sb, one, torch.where(sb < sa, zero, half))
        lb = torch.where(sb < sa, one, torch.where(sa < sb, zero, half))
        return torch.where(L.op_min, la, one), torch.where(L.op_min, lb, one)

    def _kappa(self, L: GainLayer, r: dict[str, Any]) -> Any:
        act = (r["s"] > 0).to(self.tt.dt)
        k = L.q * act if r["g"] is None else L.q * r["g"] * act
        return k if L.valid_f is None else k * L.valid_f

    @_nograd
    def route_back(self, rec: dict[str, Any], d_top: Any, *, min_aware: bool = False) -> None:
        """d_top: IT/L3 출력에 도착한 교정 (B, 146). 층마다 ℓ = κ·d 로 Σ_batch ℓ·λ·x 를 연결 칸마다 모으고 앞 층으로 d_src 를 보낸다.
        V4/L4 에서는 갱신량만 모으고 끝낸다 (V2 로 가는 교정은 만들지 않는다)."""
        d = d_top
        for li in range(len(self.layers) - 1, -1, -1):
            L = self.layers[li]
            r = rec["layers"][L.key]
            x = rec["x0"] if li == 0 else rec["layers"][self.layers[li - 1].key]["out"]
            ell = self._kappa(L, r) * d
            lamA, lamB = self._lambdas(L, r)
            d = self._route_layer(L, x, ell, lamA, lamB, need_src=(li > 0), min_aware=min_aware)

    def _route_layer(self, L: GainLayer, x: Any, ell: Any, lamA: Any, lamB: Any, need_src: bool, min_aware: bool) -> Any:
        tt = self.tt
        B = int(x.shape[0])
        xe = tt.pad_zero_col(x)
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
                    xg = xc[:, p.idx]                                          # (nb, r, K) 원천 출력 x (채움 칸 = 0)
                    er = ec[:, p.rows_t]
                    if lc is not None:
                        er = er * lc[:, p.rows_t]
                    acc = (er[:, :, None] * xg).sum(0)                          # Σ_batch ℓ_dst·λ_e·x_src
                    p.dg = acc if p.dg is None else p.dg + acc
                    if need_src:
                        sc = abs(xg * p.weff)                                  # score_e = |w0·γ·x| (raw TRACE: λ 를 곱하지 않는다)
                        if min_aware and lc is not None:
                            sc = sc * lc[:, p.rows_t][:, :, None]              # TRACE_MIN_AWARE (진단 전용): score·λ 를 수신별로 다시 정규화
                        outs.append(sc.sum(-1))
                        scores.append(sc)
                if need_src:
                    Sg = ELL.assemble(tt, outs, nb)
                    S = Sg if S is None else S + Sg
            if not need_src:
                continue
            if L.valid_f is not None:
                S = S * L.valid_f
            pos = S > 0
            s_safe = torch.where(pos, S, torch.ones_like(S))                     # Σ score = 0 → 배분 0
            fac = torch.where(pos, ec, torch.zeros_like(S))
            cflat = tt.zeros((nb, L.F + 1))
            k = 0
            for ELL, _lam in groups:
                for p in ELL.parts:
                    fr = fac[:, p.rows_t][:, :, None]
                    c = (scores[k] / s_safe[:, p.rows_t][:, :, None]) * fr      # P_e = score_e / S_j 를 먼저, 그다음 ℓ_j
                    cflat[:, p.off:p.off + p.r * p.K] = c.reshape(nb, -1)
                    scores[k] = None
                    k += 1
            chunks.append(L.T.apply(tt, cflat))                                 # d_src = Σ sign(w0)·P_e·ℓ (원천별 고정 주소 합)
        if not need_src:
            return None
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)

    def clear_grads(self) -> None:
        for p in self.parts:
            p.dg = None

    @_nograd
    def trace_deltas(self, eta_edge: float, n_batch: int) -> list[Any]:
        """Δres = η_edge·w0·(Σ_batch ℓ·λ·x)/배치 수 (v12 와 같은 곱 순서). 모은 값은 비운다."""
        out = []
        for p in self.parts:
            out.append(None if p.dg is None else p.w0 * p.dg * (float(eta_edge) / float(n_batch)))
        self.clear_grads()
        return out

    @_nograd
    def propose(self, deltas: Sequence[Any]) -> list[tuple[Part, Any]]:
        """res_new = clip(res + Δ, γ_min − 1, γ_max − 1) (적용 뒤 투영). NaN/Inf 가 있으면 적용하지 않고 멈춘다."""
        todo, raws = [], []
        for p, d in zip(self.parts, deltas):
            if d is None:
                continue
            raw = p.res + d
            raws.append(raw)
            todo.append((p, torch.clamp(raw, min=self.lo, max=self.hi)))
        if raws and not bool(torch.stack([torch.isfinite(x).all() for x in raws]).all().item()):
            raise NumericalError("연결 gain 갱신값에 NaN/Inf 가 있다 (적용하지 않고 멈춘다).")
        return todo

    @staticmethod
    def commit(todo: Sequence[tuple[Part, Any]]) -> None:
        for p, new in todo:
            p.set_res(new)

    # ---------------------------------------------------------------- 정확 BP (VJP: NumPy float64 head 의 해석 미분 → autograd)
    def make_leaves(self) -> list[Any]:
        return [p.res.detach().clone().requires_grad_(True) for p in self.parts]

    def bp_grads(self, x0: Any, W: np.ndarray, b: np.ndarray, y: np.ndarray, *, stop_l6: bool = False) -> dict[str, Any]:
        """같은 CE 를 같은 gain 잔차로 미분한다. 반환: grads (부품 순서, ∂CE/∂res), h·p·logp (NumPy float64)."""
        leaves = self.make_leaves()
        with torch.enable_grad():
            rec = self.forward(x0, leaves=leaves, stop_l6=stop_l6)
            h_t = rec["h"]
            if not h_t.requires_grad or h_t.grad_fn is None:
                raise NumericalError("BP 그래프가 끊겼다 (h 에 grad_fn 이 없다).")
            h = h_t.detach().to("cpu").numpy().astype(np.float64)
            _z, p, logp = head_forward(W, b, h)
            n = len(y)
            t = onehot(y, n)
            dL_dh = ((p - t) @ W) / n                                            # 평균은 여기서 한 번만
            go = torch.from_numpy(np.ascontiguousarray(dL_dh.astype(self.tt.np_dt))).to(self.tt.dev)
            grads = torch.autograd.grad(h_t, leaves, grad_outputs=go, create_graph=False, retain_graph=False, allow_unused=False)
        return {"grads": [g.detach() for g in grads], "h": h, "p": p, "logp": logp}

    # ---------------------------------------------------------------- 상태 (gain 잔차를 묶음 배치 그대로 — v12 와 같은 이름·순서)
    def gain_arrays(self, prefix: str = "g") -> dict[str, np.ndarray]:
        return {f"{prefix}.{p.name}": self.tt.to_np(p.res).astype(np.float32).copy() for p in self.parts}

    def res_list(self) -> list[np.ndarray]:
        return [self.tt.to_np(p.res).astype(np.float32).copy() for p in self.parts]

    def load_gain_arrays(self, arrays: dict[str, np.ndarray], prefix: str = "g") -> None:
        new = []
        for p in self.parts:
            k = f"{prefix}.{p.name}"
            if k not in arrays:
                raise IntegrityError(f"저장 상태에 gain 배열 {k} 가 없다 (다른 구조의 상태다).")
            a = np.asarray(arrays[k], np.float32)
            if a.shape != (p.r, p.K) or not np.isfinite(a).all() or a.min() < self.lo - 1e-6 or a.max() > self.hi + 1e-6:
                raise IntegrityError(f"gain 배열 {k} 의 모양·값이 올바르지 않다.")
            if np.any(a[~p.valid_np] != 0):
                raise IntegrityError(f"gain 배열 {k} 의 채움·비활성 칸이 0 이 아니다.")
            new.append(a)
        for p, a in zip(self.parts, new):
            p.set_res(self.tt.f(a))

    def set_res_list(self, arrs: Sequence[np.ndarray]) -> None:
        for p, a in zip(self.parts, arrs):
            p.set_res(self.tt.f(a))

    def reset_gains(self) -> None:
        for p in self.parts:
            p.set_res(self.tt.zeros((p.r, p.K)))

    def gains_sha(self) -> str:
        h = hashlib.sha256()
        for p in self.parts:
            h.update(np.ascontiguousarray(self.tt.to_np(p.res).astype(np.float32)).tobytes())
        return h.hexdigest()

    def gain_values(self, layer: str | None = None) -> np.ndarray:
        """유효 연결의 γ = float32(1 + res) (float64 로)."""
        vals = []
        for p in (self.parts if layer is None else self.layer_parts[layer]):
            vals.append(self.tt.to_np(p.gam).astype(np.float64)[p.valid_np])
        return np.concatenate(vals) if vals else np.zeros(0)

    def gain_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for L in self.layers:
            g = self.gain_values(L.key)
            d = np.abs(g - 1.0)
            out[L.key] = {"n": int(len(g)), "mean": float(g.mean()) if len(g) else None, "min": float(g.min()) if len(g) else None,
                          "max": float(g.max()) if len(g) else None, "n_changed": int((g != 1.0).sum()),
                          "n_ge_1pct": int((d >= 0.01).sum()), "n_ge_10pct": int((d >= 0.10).sum()),
                          "n_at_min": int((g <= self.lo + 1.0 + 1e-7).sum()), "n_at_max": int((g >= self.hi + 1.0 - 1e-7).sum())}
        return out

    def structure(self) -> list[dict[str, Any]]:
        return [{"layer": L.key, "n": L.n, "n_src": L.n_src, "edges": L.n_edges, "learnable_edges": L.n_valid_edges, "ell_slots": L.F,
                 "inactive_neurons": int(L.invalid_np.sum()), "min_neurons": int((L.op_min_np & ~L.invalid_np).sum()) if L.B is not None else 0,
                 "l6": L.l6 is not None, "transposed_backward": L.T is not None} for L in self.layers]


# ======================================================================
# 10 클래스 출력층 (float64, NumPy CPU — v12 와 같다): z = W h + b, p = softmax(z), L = −log p_y, d_z = t − p
# ======================================================================
def onehot(y: np.ndarray, n: int | None = None) -> np.ndarray:
    n = len(y) if n is None else int(n)
    t = np.zeros((n, N_CLASSES), np.float64)
    t[np.arange(n), np.asarray(y, np.int64)] = 1.0
    return t


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
    """출력층 → IT 교정 (v12): score = |W[k,i]·h_i|, P = score/Σ_i score (0 이면 0), d_h_i = Σ_k sign(W[k,i])·P_ki·d_z_k. 갱신 전 W."""
    v = W[None, :, :] * h[:, None, :]
    score = np.abs(v)
    S = score.sum(axis=2, keepdims=True)
    P = np.divide(score, S, out=np.zeros_like(score), where=S > 0)
    R = np.sign(W)[None, :, :] * P
    return (R * dz[:, :, None]).sum(axis=1)


class EvalAcc:
    """분할 평가 누적 (v12 와 같은 합산 순서: 배치 합을 float64 로 더한다)."""

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
    """주 학습 train 순서 (v12 와 같다): default_rng([seed, 1010, epoch(1부터), 11]). 두 조건이 같은 순열을 쓴다."""
    return np.random.default_rng([int(seed), 1010, int(epoch), 11]).permutation(int(n))


def better(acc: float | None, ce: float, best: dict[str, Any] | None, acc_key: str = "dev_accuracy", ce_key: str = "dev_ce") -> bool:
    """dev 정확도 최대 → 동률이면 dev CE 최소 → 다시 동률이면 이른 쪽 (나중 것은 엄격히 좋아야 한다)."""
    if best is None:
        return True
    if acc is None:
        return False
    return acc > best[acc_key] or (acc == best[acc_key] and ce < best[ce_key])


# ======================================================================
# v12 입력 어댑터 (run 폴더 또는 그 run 이 든 ZIP). 원본은 읽기만 한다.
#   폴더: 그 자리에서 읽기 전용 memmap.   ZIP: 필요한 파일만 결과 폴더의 _input_cache 에 한 번 푼다 (test 캐시는 마지막 평가 때).
# ======================================================================
def _safe_member(name: str) -> str | None:
    """ZIP 항목 이름 검사: 절대 경로·드라이브 문자·'..'·NUL 이 있으면 None (경로 이탈 방지)."""
    n = name.replace("\\", "/")
    if not n or "\x00" in n or n.startswith("/") or (len(n) >= 2 and n[1] == ":"):
        return None
    if any(part == ".." for part in n.split("/")):
        return None
    return n


def _safe_dirname(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)[:80] or "run"


def npy_header(f: Any) -> tuple[tuple[int, ...], str, bool, int]:
    """.npy 머리 (모양, dtype 문자열, fortran_order, 머리 길이). 본문은 읽지 않는다."""
    major, _minor = np.lib.format.read_magic(f)
    if major == 1:
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    elif major in (2, 3):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    else:
        raise InputError(f".npy 형식 판 {major} 은 지원하지 않는다.")
    return tuple(int(s) for s in shape), np.dtype(dtype).str, bool(fortran), int(f.tell())


class V12Input:
    """v12 run 하나. kind = folder | zip. 파일 접근은 local(name) 으로만 (ZIP 은 처음 요청할 때 한 번 푼다)."""

    def __init__(self, raw: Any, out_root: Path, *, run_hint: str | None = None,
                 chooser: Callable[[list[str]], int] | None = None):
        self.raw = str(raw)
        self.path = clean_user_path(raw)
        if not self.path.exists():
            raise InputError(f"입력 경로가 없다: {self.path}\n  v12 run 폴더 (config.json·frozen_stem.npz·cache_train.npy 가 든 폴더) 또는 그 run 이 든 ZIP 을 준다.")
        self.path = self.path.resolve()
        self.out_root = Path(out_root)
        self.zf: zipfile.ZipFile | None = None
        self.members: dict[str, zipfile.ZipInfo] = {}
        self.unsafe_members: list[str] = []
        if self.path.is_dir():
            self.kind = "folder"
            cands = self._folder_candidates()
            names = [str(c.relative_to(self.path)) if c != self.path else "." for c in cands]
        elif self.path.is_file() and zipfile.is_zipfile(self.path):
            self.kind = "zip"
            cands = self._zip_candidates()
            names = [c.rstrip("/") or "." for c in cands]
        else:
            raise InputError(f"{self.path} 는 폴더도 ZIP 도 아니다.")
        if not cands:
            extra = f" (경로 이탈 위험으로 무시한 ZIP 항목 {len(self.unsafe_members)} 개)" if self.unsafe_members else ""
            raise InputError(f"{self.path} 에서 v12 run ({', '.join(V12_REQUIRED)} 이 모두 있고 config.json 형식이 {V12_RUN_FORMAT}) 을 찾지 못했다{extra}.")
        k = 0
        if len(cands) > 1:
            k = -1
            if run_hint:
                hits = [i for i, nm in enumerate(names) if nm == run_hint or nm.split("/")[-1] == run_hint or Path(nm).name == run_hint]
                if len(hits) == 1:
                    k = hits[0]
            if k < 0 and chooser is not None:
                k = int(chooser(names))
            if k < 0:
                raise InputError("입력 안에 v12 run 이 여러 개다: " + ", ".join(names) + " — --run <이름> 으로 하나를 고른다.")
        self.candidates = names
        self.run_name = names[k].split("/")[-1] if names[k] != "." else self.path.stem
        if self.kind == "folder":
            self.run_dir: Path | None = cands[k]
            self.prefix = ""
            self.extract_dir: Path | None = None
        else:
            self.run_dir = None
            self.prefix = cands[k]
            need = sorted(n for n in self.members if n.startswith(self.prefix) and n[len(self.prefix):] in V12_REQUIRED + V12_OPTIONAL)
            ident = sha256_text(dumps([[n, self.members[n].CRC, self.members[n].file_size] for n in need], indent=None))[:16]
            self.extract_dir = self.out_root / "_input_cache" / f"{_safe_dirname(self.run_name)}_{ident}"
        self.chosen = names[k]

    # ---------------------------------------------------------------- 후보 찾기
    def _folder_candidates(self) -> list[Path]:
        out: list[Path] = []

        def consider(d: Path) -> None:
            c = d / "config.json"
            if c.is_file() and all((d / f).is_file() for f in V12_REQUIRED):
                with contextlib.suppress(Exception):
                    if read_json(c).get("format") == V12_RUN_FORMAT:
                        out.append(d)

        consider(self.path)
        if out:
            return out
        with contextlib.suppress(OSError):
            for c1 in sorted(self.path.iterdir()):
                if c1.is_dir():
                    consider(c1)
                    with contextlib.suppress(OSError):
                        for c2 in sorted(c1.iterdir()):
                            if c2.is_dir():
                                consider(c2)
        return out

    def _zip_candidates(self) -> list[str]:
        self.zf = zipfile.ZipFile(self.path)
        for info in self.zf.infolist():
            if info.is_dir():
                continue
            nm = _safe_member(info.filename)
            if nm is None:
                self.unsafe_members.append(info.filename)
                continue
            self.members[nm] = info
        out = []
        for nm in sorted(self.members):
            if nm == "config.json" or nm.endswith("/config.json"):
                pre = nm[:-len("config.json")]
                if all((pre + f) in self.members for f in V12_REQUIRED):
                    with contextlib.suppress(Exception):
                        if json.loads(self.zf.read(self.members[nm]).decode("utf-8")).get("format") == V12_RUN_FORMAT:
                            out.append(pre)
        return out

    # ---------------------------------------------------------------- 파일 접근
    def has(self, name: str) -> bool:
        if self.kind == "folder":
            return (self.run_dir / name).is_file()
        return (self.prefix + name) in self.members

    def size(self, name: str) -> int | None:
        if not self.has(name):
            return None
        if self.kind == "folder":
            return int((self.run_dir / name).stat().st_size)
        return int(self.members[self.prefix + name].file_size)

    def read_bytes(self, name: str) -> bytes:
        if self.kind == "folder":
            return (self.run_dir / name).read_bytes()
        return self.zf.read(self.members[self.prefix + name])

    def header(self, name: str) -> tuple[tuple[int, ...], str, bool, int]:
        if self.kind == "folder":
            with open(self.run_dir / name, "rb") as f:
                return npy_header(f)
        with self.zf.open(self.members[self.prefix + name]) as f:
            return npy_header(f)

    def _record_path(self) -> Path:
        return self.extract_dir / "_extract_record.json"

    def local(self, name: str) -> Path:
        """읽을 수 있는 로컬 경로. 폴더 입력은 원본 그대로 (읽기 전용으로만 연다), ZIP 은 처음 한 번 풀고 다음부터 재사용한다."""
        if self.kind == "folder":
            p = self.run_dir / name
            if not p.is_file():
                raise InputError(f"입력 run 에 {name} 이 없다: {self.run_dir}")
            return p
        info = self.members.get(self.prefix + name)
        if info is None:
            raise InputError(f"ZIP 의 run {self.chosen} 에 {name} 이 없다.")
        dst = self.extract_dir / name
        rec = read_json(self._record_path()) if self._record_path().is_file() else {}
        ent = (rec.get("members") or {}).get(name) or {}
        if dst.is_file() and ent.get("crc") == info.CRC and ent.get("size") == info.file_size and dst.stat().st_size == info.file_size:
            return dst
        ensure_dir(self.extract_dir)
        free = shutil.disk_usage(self.extract_dir).free
        if info.file_size > free * 0.98:
            raise InputError(f"ZIP 에서 {name} ({fmt_bytes(info.file_size)}) 을 풀 공간이 없다 (여유 {fmt_bytes(free)}).")
        big = info.file_size >= 64 << 20
        if big:
            say(f"  ZIP 에서 {name} ({fmt_bytes(info.file_size)}) 을 {self.extract_dir} 에 한 번 푼다 …")
        t0 = time.perf_counter()
        tmp = dst.with_name(dst.name + ".partial")
        try:
            with self.zf.open(info) as fsrc, open(tmp, "wb") as fdst:          # zipfile 이 CRC 를 끝에서 확인한다 (다르면 BadZipFile)
                shutil.copyfileobj(fsrc, fdst, 1 << 22)
            if tmp.stat().st_size != info.file_size:
                raise InputError(f"ZIP 항목 {name} 의 풀린 크기가 다르다.")
            os.replace(tmp, dst)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        rec.setdefault("zip", str(self.path))
        rec.setdefault("run", self.chosen)
        rec.setdefault("members", {})[name] = {"crc": info.CRC, "size": info.file_size, "utc": utc_now()}
        write_json(self._record_path(), rec)
        if big:
            say(f"    완료 ({fmt_s(time.perf_counter() - t0)})")
        return dst

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": str(self.path), "chosen_run": self.chosen, "run_name": self.run_name, "candidates": self.candidates,
                "extract_dir": str(self.extract_dir) if self.extract_dir else None, "unsafe_zip_members_ignored": self.unsafe_members[:20],
                "n_unsafe_zip_members": len(self.unsafe_members),
                "note_ko": "원본 ZIP·폴더는 읽기만 했다. ZIP 은 필요한 파일만 _input_cache 에 한 번 풀었다 (test 캐시는 최종 평가 때)."}


def part_layout(specs: dict[str, LayerSpec]) -> list[tuple[str, int, int]]:
    """엔진 gain 묶음의 (이름, 행 수, K) 순서 — PyTorch 없이 v12 best_params.npz 의 배열 배치를 확인한다."""
    out = []
    for key in LEARN_KEYS:
        sp = specs[key]
        for grp, ptr, src in (("A", sp.a_ptr, sp.a_src), ("B", sp.b_ptr, sp.b_src)):
            if len(src) == 0:
                continue
            fan = np.diff(ptr)
            K = np.ones(len(fan), np.int64)
            nz = fan > 0
            K[nz] = np.left_shift(1, np.ceil(np.log2(fan[nz])).astype(np.int64))
            for pi, k in enumerate(np.unique(K)):
                out.append((f"{key.replace('/', '.')}.{grp}{pi}", int((K == k).sum()), int(k)))
    return out


def v12_params_sha(W: np.ndarray, b: np.ndarray, arrays: dict[str, np.ndarray], layout: list[tuple[str, int, int]], prefix: str = "g") -> str:
    """v12 params_sha = sha256_text(sha256_array(W, b) + gains_sha) (gains_sha = 묶음 순서대로 res float32 바이트의 SHA256)."""
    h = hashlib.sha256()
    for name, r, K in layout:
        a = np.asarray(arrays[f"{prefix}.{name}"], np.float32)
        if a.shape != (r, K):
            raise IntegrityError(f"gain 배열 {prefix}.{name} 의 모양 {a.shape} 이 구조 ({r}, {K}) 와 다르다.")
        h.update(np.ascontiguousarray(a).tobytes())
    return sha256_text(sha256_array(np.asarray(W, np.float64), np.asarray(b, np.float64)) + h.hexdigest())


def _relpath(src: int, idx: int) -> str:
    return f"mnist/{'train' if int(src) == 0 else 'test'}/{int(idx):05d}.png"


def _jn(o: Any) -> Any:
    return json.loads(dumps(o, indent=None))


def validate_input(src: V12Input, scope: str, *, verify_bodies: bool = True) -> dict[str, Any]:
    """v12 run 검증. 반환 ctx: 모델·구조·동결 줄기·분할·캐시 정보·검사 목록. 필수 검사가 하나라도 틀리면 InputError (이유 목록)."""
    checks: list[dict[str, Any]] = []
    fails: list[str] = []

    def chk(cid: str, ok: Any, detail: str, required: bool = True) -> bool:
        checks.append({"id": cid, "ok": bool(ok), "required": required, "detail": detail})
        if not ok and required:
            fails.append(f"[{cid}] {detail}")
        return bool(ok)

    def stop_if_failed() -> None:
        if fails:
            raise InputError("입력 v12 run 검증 실패 — 조용히 맞춰 읽지 않고 멈춘다:\n  " + "\n  ".join(fails))

    t0 = time.perf_counter()
    files: dict[str, Any] = {}
    # ---- config.json
    cfg_b = src.read_bytes("config.json")
    files["config.json"] = {"bytes": len(cfg_b), "sha256": sha256_bytes(cfg_b)}
    try:
        cfg = json.loads(cfg_b.decode("utf-8"))
    except ValueError as exc:
        raise InputError(f"config.json 을 읽을 수 없다: {exc}") from exc
    chk("config.format", cfg.get("format") == V12_RUN_FORMAT, f"형식 {cfg.get('format')!r} (기대 {V12_RUN_FORMAT})")
    chk("config.program", cfg.get("program") == V12_PROGRAM and cfg.get("version") == V12_VERSION,
        f"프로그램 {cfg.get('program')} {cfg.get('version')} (기대 {V12_PROGRAM} {V12_VERSION})")
    chk("config.source_sha256", cfg.get("program_sha256") == V12_SOURCE_SHA256,
        f"출처 소스 SHA256 {str(cfg.get('program_sha256'))[:16]}… (기대 {V12_SOURCE_SHA256[:16]}…)")
    settings, model = cfg.get("settings") or {}, cfg.get("model") or {}
    seed = settings.get("seed")
    chk("config.seed", isinstance(seed, int), f"seed {seed!r}")
    stop_if_failed()
    seed = int(seed)
    chk("config.model_is_v12_default", _jn(model) == _jn(V12_MODEL_DEFAULT), "모델 설정이 v12 기본값과 같다 (다르면 그 설정으로 구조를 만든다)", required=False)
    # ---- 구조 재구성 (V2 메타 + V4·IT)
    say("  구조 재구성 (V2 채널·좌표 메타 + V4·IT 6 층; V1 Gabor 연결은 만들지 않는다) …")
    specs, ranges, hashes = build_structure(model)
    edges = count_learnable_edges(specs)
    N = ranges["IT/L3"][1]
    cache_dim, feat_dim = specs["V2/L3"].n, specs["IT/L3"].n
    chk("structure.v2_to_v4", specs["V4/L4"].n_src == cache_dim, f"V4/L4 입력 {specs['V4/L4'].n_src} = V2/L3 {cache_dim}")
    # ---- frozen_stem.npz
    stem_p = src.local("frozen_stem.npz")
    files["frozen_stem.npz"] = {"bytes": stem_p.stat().st_size, "sha256": sha256_file(stem_p)}
    st_arr, st_meta = load_npz(stem_p)
    st_meta = st_meta or {}
    chk("stem.format", st_meta.get("format") == V12_STEM_FORMAT and st_meta.get("version") == V12_VERSION,
        f"형식 {st_meta.get('format')!r} 판 {st_meta.get('version')!r}")
    chk("stem.seed", st_meta.get("seed") == seed, f"동결 줄기 seed {st_meta.get('seed')} / config seed {seed}")
    need = ("q", "threshold", "W0", "b0")
    chk("stem.arrays", all(k in st_arr for k in need), f"배열 {sorted(st_arr)}")
    stop_if_failed()
    q, thr, W0, b0 = (st_arr[k] for k in need)
    chk("stem.N", st_meta.get("N") == N and q.shape == (N,) and thr.shape == (N,), f"N 기록 {st_meta.get('N')} · q {q.shape} · 재구성 {N}")
    chk("stem.dtype", q.dtype == np.float32 and thr.dtype == np.float32 and W0.dtype == np.float64 and b0.dtype == np.float64,
        f"q {q.dtype} · threshold {thr.dtype} · W0 {W0.dtype} · b0 {b0.dtype}")
    chk("stem.layer_ranges", _jn(st_meta.get("layer_ranges")) == _jn(ranges), "12 층 뉴런 번호 범위 (기록 = 재구성)")
    sh = st_meta.get("struct_hash") or {}
    np_v12 = None
    with contextlib.suppress(Exception):
        np_v12 = json.loads(src.read_bytes("cache_manifest.json").decode("utf-8"))["files"]["cache_train"]["key"]["backend"]["numpy"]
    for a in ("V2", "V4", "IT"):
        chk(f"structure.hash.{a}", sh.get(a) == hashes[a], f"{a} 구조 해시 기록 {str(sh.get(a))[:16]}… / 재구성 {hashes[a][:16]}…"
            + (f" — V4 곡선 연결의 기하 경계 비교 (정확히 0 근처) 는 NumPy·수학 라이브러리 판에 따라 달라질 수 있다 (v12 캐시 기록의 NumPy {np_v12}, "
               f"지금 NumPy {np.__version__}). v12 를 실행한 PC 와 같은 NumPy 판에서 실행한다 (구조를 조용히 맞추지 않는다)." if a == "V4" else ""))
    chk("stem.head_shape", W0.shape == (N_CLASSES, feat_dim) and b0.shape == (N_CLASSES,), f"W0 {W0.shape} · b0 {b0.shape} (기대 [10,{feat_dim}] · [10])")
    chk("stem.finite", all(np.isfinite(a).all() for a in (q, thr, W0, b0)), "q·threshold·W0·b0 유한")
    stop_if_failed()
    v4it_q1 = all(bool((q[ranges[k][0]:ranges[k][1]] == 1.0).all()) for k in LEARN_KEYS)
    v4it_t0 = all(bool((thr[ranges[k][0]:ranges[k][1]] == 0.0).all()) for k in LEARN_KEYS)
    chk("stem.v4it_q1_thr0", v4it_q1 and v4it_t0, f"V4·IT q = 1 ({v4it_q1}) · threshold = 0 ({v4it_t0})")
    stem_rec = st_meta.get("stem") or {}
    keys12 = [f"{a}/{l}" for a in ("V1", "V2") for l in LAYERS]
    q_sha = sha256_array(*[q[ranges[k][0]:ranges[k][1]] for k in keys12])
    t_sha = sha256_array(*[thr[ranges[k][0]:ranges[k][1]] for k in keys12])
    chk("stem.v1v2_q_thr_sha", stem_rec.get("q_sha256") == q_sha and stem_rec.get("thr_sha256") == t_sha, "동결 V1·V2 q·threshold 해시 (기록 = 배열)")
    chk("stem.record_struct", (stem_rec.get("struct_hash") or {}).get("V2") == hashes["V2"]
        and (stem_rec.get("struct_hash") or {}).get("V1") == sh.get("V1"), "동결 줄기 기록의 V1·V2 구조 해시 (V1 은 기록 간 대조만)")
    # ---- split_ids.npz
    sp_p = src.local("split_ids.npz")
    files["split_ids.npz"] = {"bytes": sp_p.stat().st_size, "sha256": sha256_file(sp_p)}
    sp_arr, sp_meta = load_npz(sp_p)
    sp_meta = sp_meta or {}
    chk("split.seed", sp_meta.get("seed") == seed, f"분할 seed {sp_meta.get('seed')} / {seed}")
    split: dict[str, dict[str, np.ndarray]] = {}
    ids_sha, order_sha, pairs = {}, {}, {}
    for s in SPLITS:
        ok = all(f"{s}.{k}" in sp_arr for k in ("src", "idx", "y"))
        chk(f"split.{s}.arrays", ok, f"{s} src·idx·y 배열")
        if not ok:
            continue
        a_src, a_idx, a_y = sp_arr[f"{s}.src"], sp_arr[f"{s}.idx"], sp_arr[f"{s}.y"]
        chk(f"split.{s}.dtype", a_src.dtype == np.int8 and a_idx.dtype == np.int64 and a_y.dtype == np.int64 and len(a_src) == len(a_idx) == len(a_y),
            f"{s} dtype {a_src.dtype}/{a_idx.dtype}/{a_y.dtype} · 길이 {len(a_src)}/{len(a_idx)}/{len(a_y)}")
        chk(f"split.{s}.labels", len(a_y) > 0 and int(a_y.min()) >= 0 and int(a_y.max()) < N_CLASSES, f"{s} 라벨 범위 0~9")
        chk(f"split.{s}.source", bool((a_src == (1 if s == "test" else 0)).all()), f"{s} 출처 (train·dev = 공식 train, test = 공식 test)")
        key = a_src.astype(np.int64) * 1_000_000 + a_idx
        chk(f"split.{s}.unique", len(np.unique(key)) == len(key), f"{s} 표본 ID 중복 없음 ({len(key):,} 개)")
        pairs[s] = key
        order_sha[s] = sha256_array(a_src, a_idx, a_y)
        ids_sha[s] = sha256_text("\n".join(sorted(_relpath(a, i) for a, i in zip(a_src.tolist(), a_idx.tolist()))))
        chk(f"split.{s}.order_sha", (sp_meta.get("order_sha256") or {}).get(s) == order_sha[s], f"{s} 순서 해시 (기록 = 배열)")
        split[s] = {"src": a_src, "idx": a_idx, "y": a_y}
    for s1, s2 in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if s1 in pairs and s2 in pairs:
            inter = int(np.intersect1d(pairs[s1], pairs[s2]).size)
            chk(f"split.disjoint.{s1}_{s2}", inter == 0, f"{s1} ∩ {s2} = {inter}")
    stop_if_failed()
    split_sha = sha256_text(dumps({sp: ids_sha[sp] for sp in SPLITS}, indent=None))
    chk("split.split_sha", sp_meta.get("split_sha256") == split_sha, "분할 해시 (기록 = 재계산)")
    # ---- cache_manifest.json + 캐시 파일
    bk_b = src.read_bytes("cache_manifest.json")
    files["cache_manifest.json"] = {"bytes": len(bk_b), "sha256": sha256_bytes(bk_b)}
    book = json.loads(bk_b.decode("utf-8"))
    chk("cache.format", book.get("format") == V12_CACHE_FORMAT, f"캐시 기록 형식 {book.get('format')!r}")
    caches: dict[str, Any] = {}
    test_available = True
    for s in SPLITS:
        name = f"cache_{s}"
        ent = (book.get("files") or {}).get(name)
        if s == "test" and (ent is None or not src.has(f"{name}.npy")):
            test_available = False
            chk("cache.test.present", False, "cache_test.npy 가 없다 → 최종 test 평가를 할 수 없다 (나머지는 진행)", required=False)
            continue
        if not chk(f"cache.{s}.entry", ent is not None and src.has(f"{name}.npy"), f"{name} 기록·파일"):
            continue
        key = ent.get("key") or {}
        chk(f"cache.{s}.key_sha", ent.get("key_sha256") == sha256_text(dumps(key, indent=None)), f"{name} 캐시 키 해시")
        chk(f"cache.{s}.complete", ent.get("complete") is True and ent.get("rows_done") == ent.get("rows"), f"{name} 완성 (rows_done {ent.get('rows_done')})")
        n_s = len(split[s]["y"])
        chk(f"cache.{s}.shape_record", ent.get("rows") == n_s and ent.get("dim") == cache_dim and ent.get("dtype") == "float32",
            f"{name} 기록 행 {ent.get('rows')} (분할 {n_s}) · 차원 {ent.get('dim')} (V2/L3 {cache_dim}) · {ent.get('dtype')}")
        chk(f"cache.{s}.key_fields", key.get("format") == V12_CACHE_FORMAT and key.get("split") == s and key.get("n") == n_s
            and key.get("dim") == cache_dim and key.get("dtype") == "float32" and "V2/L3" in str(key.get("boundary")),
            f"{name} 키 (형식·분할·행·차원·경계 {key.get('boundary')!r})")
        chk(f"cache.{s}.order", key.get("order_sha256") == order_sha[s] and key.get("ids_sha256") == ids_sha[s],
            f"{name} 행 순서·ID 해시 = split_ids.npz (행 ↔ 라벨 대응)")
        chk(f"cache.{s}.stem", _jn(key.get("stem")) == _jn(stem_rec), f"{name} 동결 V1·V2 기록 = frozen_stem")
        chk(f"cache.{s}.l6", _jn(key.get("l6")) == _jn(model.get("l6")), f"{name} L6 설정 = config")
        size = src.size(f"{name}.npy")
        chk(f"cache.{s}.bytes", size == ent.get("bytes"), f"{name}.npy 크기 {size} / 기록 {ent.get('bytes')}")
        caches[s] = {"file": f"{name}.npy", "rows": n_s, "dim": cache_dim, "bytes": size, "sha256": ent.get("sha256"),
                     "key_sha256": ent.get("key_sha256"), "backend_v12": key.get("backend"), "verified_sha": False}
    stop_if_failed()
    for s in ("train", "dev"):
        name = caches[s]["file"]
        shape, dts, fortran, _hl = src.header(name)
        chk(f"cache.{s}.header", shape == (caches[s]["rows"], cache_dim) and dts == "<f4" and not fortran, f"{name} 머리 {shape} {dts} fortran={fortran}")
    if test_available and src.kind == "folder":
        shape, dts, fortran, _hl = src.header(caches["test"]["file"])
        chk("cache.test.header", shape == (caches["test"]["rows"], cache_dim) and dts == "<f4" and not fortran, f"cache_test.npy 머리 {shape} {dts}")
    stop_if_failed()
    if verify_bodies:
        for s in ("train", "dev"):
            p = src.local(caches[s]["file"])
            t1 = time.perf_counter()
            sha = sha256_file(p)
            chk(f"cache.{s}.body_sha", sha == caches[s]["sha256"], f"{caches[s]['file']} 본문 SHA256 ({fmt_s(time.perf_counter() - t1)})")
            caches[s]["verified_sha"] = sha == caches[s]["sha256"]
            caches[s]["path"] = str(p)
        stop_if_failed()
    # ---- 출처·참고 기록 (last_state · best_state · final_metrics)
    ref: dict[str, Any] = {"available": {n: src.has(n) for n in V12_OPTIONAL}}
    if src.has("last_state.json"):
        ls = json.loads(src.read_bytes("last_state.json").decode("utf-8"))
        w = ls.get("world") or {}
        chk("v12.last_state.format", ls.get("format") == V12_STATE_FORMAT, f"last_state 형식 {ls.get('format')!r}")
        wsh = w.get("struct_hash") or {}
        chk("v12.world", w.get("split_sha256") == split_sha and w.get("N") == N and w.get("cache_dim") == cache_dim
            and w.get("feature_dim") == feat_dim and _jn(w.get("learnable_edges")) == _jn(edges)
            and all(wsh.get(a) == hashes[a] for a in ("V2", "V4", "IT")) and wsh.get("V1") == sh.get("V1"),
            "v12 world 기록 (분할·구조·N·차원·학습 연결 수) = 재구성")
        for s in ("train", "dev"):
            cr = (ls.get("caches") or {}).get(f"cache_{s}") or {}
            chk(f"v12.state_cache.{s}", cr.get("sha256") == caches[s]["sha256"], f"v12 상태의 cache_{s} 해시 = 캐시 기록")
        tr = ls.get("train") or {}
        hist = tr.get("history") or []
        last_row = hist[-1] if hist else {}
        prep_v2 = (ls.get("prep") or {}).get("V2") or {}
        ref["v12_last_state"] = {"phase": ls.get("phase"), "completed_epochs": tr.get("epoch"), "best": tr.get("best"),
                                 "last_epoch_running_train_accuracy": last_row.get("train_accuracy_running"),
                                 "last_epoch_running_train_ce": last_row.get("train_ce_running"),
                                 "last_epoch": last_row.get("epoch"),
                                 "dev_accuracy_by_epoch": [[r.get("epoch"), r.get("dev_accuracy")] for r in hist],
                                 "v2_prep_cosine": {"best_dev_accuracy": prep_v2.get("best_dev_accuracy"),
                                                    "last_epoch_dev_accuracy": ((prep_v2.get("history") or [{}])[-1] or {}).get("dev_accuracy"),
                                                    "epochs_completed": prep_v2.get("epochs_completed"), "status": prep_v2.get("status")}}
    else:
        chk("v12.last_state.present", False, "last_state.json 이 없다 (출처 기록 대조를 일부 건너뜀)", required=False)
    if src.has("final_metrics.json"):
        fm = json.loads(src.read_bytes("final_metrics.json").decode("utf-8"))
        ref["v12_final_metrics"] = {"selected_epoch": fm.get("selected_epoch"), "dev_at_selected": (fm.get("dev_at_selected") or {}).get("accuracy"),
                                    "test_accuracy": fm.get("test_accuracy"), "test_ce": fm.get("test_ce"), "backend": fm.get("backend"),
                                    "note_ko": "v12 가 이미 조회한 test 결과 (참고 기록). v13 의 새 결과가 아니다."}
    best_ok = None
    if src.has("best_params.npz") and src.has("best_state.json"):
        bs = json.loads(src.read_bytes("best_state.json").decode("utf-8"))
        bp_p = src.local("best_params.npz")
        files["best_params.npz"] = {"bytes": bp_p.stat().st_size, "sha256": sha256_file(bp_p)}
        bp_arr, bp_meta = load_npz(bp_p)
        try:
            psha = v12_params_sha(bp_arr["W"], bp_arr["b"], bp_arr, part_layout(specs))
            best_ok = psha == (bs.get("best") or {}).get("params_sha256") == ((bp_meta or {}).get("best") or {}).get("params_sha256")
        except (KeyError, IntegrityError) as exc:
            psha, best_ok = f"오류: {exc}", False
        chk("v12.best_params", best_ok, f"v12 best W·b·gain 해시 = best_state 기록 (epoch {(bs.get('best') or {}).get('epoch')})",
            required=(scope == "v12_best_probes"))
        ref["v12_best"] = {"epoch": (bs.get("best") or {}).get("epoch"), "dev_accuracy": (bs.get("best") or {}).get("dev_accuracy"),
                           "dev_ce": (bs.get("best") or {}).get("dev_ce"), "params_sha256": (bs.get("best") or {}).get("params_sha256"),
                           "verified": best_ok}
    elif scope == "v12_best_probes":
        chk("v12.best_params", False, "메뉴 2 는 best_params.npz 와 best_state.json 이 필요하다")
    stop_if_failed()
    input_id = sha256_text(dumps({"config": files["config.json"]["sha256"], "stem": files["frozen_stem.npz"]["sha256"],
                                  "split": files["split_ids.npz"]["sha256"], "manifest": files["cache_manifest.json"]["sha256"],
                                  "cache_train": caches["train"]["sha256"], "cache_dev": caches["dev"]["sha256"],
                                  "cache_test": (caches.get("test") or {}).get("sha256")}, indent=None))
    n_req = sum(1 for c in checks if c["required"])
    say(f"  입력 검증 통과: 필수 검사 {n_req} 개 · 참고 검사 {len(checks) - n_req} 개 ({fmt_s(time.perf_counter() - t0)})")
    return {"cfg": cfg, "model": model, "seed": seed, "settings_v12": settings, "specs": specs, "ranges": ranges, "hashes": hashes,
            "struct_hash_v1_record": sh.get("V1"), "edges": edges, "N": N, "cache_dim": cache_dim, "feat_dim": feat_dim,
            "q": q, "thr": thr, "W0": W0, "b0": b0, "stem_meta": st_meta, "split": split, "split_sha256": split_sha,
            "order_sha": order_sha, "caches": caches, "test_available": test_available, "files": files, "checks": checks,
            "reference": ref, "input_id": input_id, "best_verified": best_ok}


# ======================================================================
# 주 학습 (TRACE / BP_FULL). 같은 초기 snapshot · 같은 epoch 순열 · 같은 미니배치 · 같은 학습률 · 가산 갱신 (momentum·Adam·decay·clipping 없음).
#   last.npz = 학습용 마지막 상태 (W·b·gain + best 사본 + 상태 JSON 을 한 파일에 원자적으로). best.npz = 평가용 선택 상태 (따로 보관).
# ======================================================================
def chunk_rows(n: int, size: int) -> list[tuple[int, int]]:
    size = max(1, int(size))
    return [(a, min(int(n), a + size)) for a in range(0, int(n), size)]


def eval_split(eng: Engine, mm: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray, B: int, stop: StopFlag | None = None,
               label: str = "") -> dict[str, Any]:
    """고정 상태의 평가 (v12 eval_cache 와 같은 묶음 크기·합산 순서)."""
    acc = EvalAcc()
    with torch.no_grad():
        for a0 in range(0, len(y), B):
            if stop is not None and stop.requested:
                raise Interrupted(f"{label} 평가 도중 멈춤 (이 평가는 버리고 이어 갈 때 다시 한다)")
            h = eng.tt.to_np(eng.forward(eng.tt.f(mm[a0:a0 + B]))["h"]).astype(np.float64)
            _z, p, logp = head_forward(W, b, h)
            acc.add(p, logp, y[a0:a0 + B])
    return acc.result()


def params_sha(W: np.ndarray, b: np.ndarray, gains_sha: str) -> str:
    """v12 와 같은 식: sha256_text(sha256_array(W, b) + gains_sha)."""
    return sha256_text(sha256_array(np.asarray(W, np.float64), np.asarray(b, np.float64)) + gains_sha)


class MainTrainer:
    """한 조건의 주 학습. 조건 차이는 gain 갱신 함수 하나뿐 (TRACE: v12 raw TRACE, BP_FULL: 같은 CE 의 정확한 gradient 로 SGD)."""

    FORMAT = "cortex_v13/main_last/1"

    def __init__(self, run: Any, cond: str):
        if cond not in CONDS:
            raise V13Error(f"알 수 없는 조건 {cond}")
        self.run, self.cond = run, cond
        self.dir = run.dir / COND_DIR[cond]
        S = run.S
        self.B, self.E = int(S["batch_size"]), int(S["epochs"])
        self.eta_h, self.eta_e = float(S["eta_head"]), float(S["eta_edge"])
        self.eng: Engine = run.engine()
        self.st: dict[str, Any] = {}
        self.W = self.b = None
        self.best_arrays: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- 상태
    def _init_state(self) -> None:
        init = self.run.initial_params()
        self.eng.reset_gains()
        self.W, self.b = init["W"].copy(), init["b"].copy()
        sha = self.params_sha()
        if sha != init["params_sha256"]:
            raise IntegrityError("공통 초기 snapshot 의 해시가 다시 만든 시작 상태와 다르다.")
        self.st = {"format": self.FORMAT, "cond": self.cond, "epoch": -1, "next_batch": 0, "acc": {}, "upd": {}, "best": None,
                   "history": [], "seq": 0, "start_params_sha256": sha, "time": {}, "perm_sha256_by_epoch": {}}

    def _load(self) -> None:
        p = self.dir / "last.npz"
        if not p.is_file():
            self._init_state()
            return
        arrays, meta = load_npz(p)
        if not meta or meta.get("format") != self.FORMAT or meta.get("cond") != self.cond:
            raise IntegrityError(f"{p} 는 이 조건의 학습 상태가 아니다.")
        self.eng.load_gain_arrays(arrays, "g")
        self.W, self.b = np.asarray(arrays["W"], np.float64).copy(), np.asarray(arrays["b"], np.float64).copy()
        if self.params_sha() != meta.get("params_sha256"):
            raise IntegrityError(f"{p} 의 W·b·gain 해시가 상태 기록과 다르다 (손상된 저장).")
        self.best_arrays = {k[5:]: v for k, v in arrays.items() if k.startswith("best.")}
        self.st = meta

    def params_sha(self, W: np.ndarray | None = None, b: np.ndarray | None = None) -> str:
        return params_sha(self.W if W is None else W, self.b if b is None else b, self.eng.gains_sha())

    def save(self) -> None:
        """best 사본 (새 best 가 생겼을 때) → last 순서로 원자적 저장. last 에는 best 배열 사본도 들어 있어 last 하나로 이어 갈 수 있다."""
        run = self.run
        with run.stop.critical():
            t0 = time.perf_counter()
            st = self.st
            st["seq"] = int(st.get("seq", 0)) + 1
            st["utc"] = utc_now()
            st["params_sha256"] = self.params_sha()
            b = st.get("best")
            if self.best_arrays and b and b.get("mirrored_seq") is None:
                b["mirrored_seq"] = st["seq"]
                save_npz(self.dir / "best.npz", self.best_arrays, {"format": "cortex_v13/main_best/1", "cond": self.cond, "best": b})
                write_json(self.dir / "best_state.json", {"format": "cortex_v13/main_best/1", "cond": self.cond, "best": b,
                                                         "note_ko": "dev 정확도 최대 → 동률이면 CE 최소 → 이른 epoch (epoch 0 포함). 평가용 사본이다."})
            arrays = {"W": self.W, "b": self.b, **self.eng.gain_arrays("g")}
            arrays.update({f"best.{k}": v for k, v in self.best_arrays.items()})
            save_npz(self.dir / "last.npz", arrays, st)
            write_json(self.dir / "last_state.json", st)
            run.add_time("save_s", time.perf_counter() - t0)

    # ---------------------------------------------------------------- 한 배치 (갱신안 계산 → 호출한 쪽에서 함께 적용)
    def step(self, x0: Any, y: np.ndarray) -> tuple[list[tuple[Part, Any]], np.ndarray, np.ndarray, float, int, np.ndarray]:
        eng, W, b = self.eng, self.W, self.b
        n = len(y)
        t = onehot(y, n)
        if self.cond == "TRACE":
            with torch.no_grad():
                rec = eng.forward(x0, record="trace")
            h = eng.tt.to_np(rec["h"]).astype(np.float64)
            _z, p, logp = head_forward(W, b, h)
            dz = t - p
            dW = self.eta_h * (dz.T @ h) / n
            db = self.eta_h * dz.mean(axis=0)
            d_h = head_route_trace(W, h, dz)                                   # 갱신 전 W 로 배분
            try:
                eng.route_back(rec, eng.tt.f(d_h))
            except BaseException:
                eng.clear_grads()
                raise
            deltas = eng.trace_deltas(self.eta_e, n)
        else:
            out = eng.bp_grads(x0, W, b, y)                                    # 갱신 전 W·b·res 로 계산 (같은 snapshot)
            h, p, logp = out["h"], out["p"], out["logp"]
            dz = t - p
            dW = self.eta_h * (dz.T @ h) / n
            db = self.eta_h * dz.mean(axis=0)
            with torch.no_grad():
                deltas = [g * (-self.eta_e) for g in out["grads"]]
        todo = eng.propose(deltas)
        Wn, bn = W + dW, b + db
        if not (np.isfinite(Wn).all() and np.isfinite(bn).all()):
            raise NumericalError("출력층 W·b 갱신값에 NaN/Inf 가 있다 (적용하지 않는다).")
        ce = -logp[np.arange(n), y]
        return todo, Wn, bn, float(ce.sum()), int((p.argmax(axis=1) == y).sum()), np.asarray([np.linalg.norm(dW), np.linalg.norm(db)])

    def _upd_record(self, todo: Sequence[tuple[Part, Any]], dwb: np.ndarray) -> None:
        """실제 적용된 갱신 크기 (투영·반올림 뒤 res_new − res) 를 층별로 모은다 (장치 왕복 1 번)."""
        with torch.no_grad():
            sq_np = torch.stack([((new - p.res).double() ** 2).sum() for p, new in todo]).cpu().numpy() if todo else np.zeros(0)
        u = self.st.setdefault("upd", {})
        u["n"] = int(u.get("n", 0)) + 1
        by = u.setdefault("res_sq_sum_by_layer", {})
        layer_sq: dict[str, float] = {}
        for (p, _new), v in zip(todo, sq_np.tolist()):
            layer_sq[p.layer] = layer_sq.get(p.layer, 0.0) + float(v)
        for k, v in layer_sq.items():
            by[k] = float(by.get(k, 0.0)) + math.sqrt(v)                       # 배치별 층 L2 의 합 (epoch 끝에 배치 수로 나눈다)
        u["res_l2_sum"] = float(u.get("res_l2_sum", 0.0)) + math.sqrt(sum(layer_sq.values()))
        u["W_l2_sum"] = float(u.get("W_l2_sum", 0.0)) + float(dwb[0])
        u["b_l2_sum"] = float(u.get("b_l2_sum", 0.0)) + float(dwb[1])

    # ---------------------------------------------------------------- epoch 끝
    def _epoch_end(self, e: int, dev: dict[str, Any], dev_s: float, train_s: float) -> dict[str, Any]:
        st = self.st
        a, u = st.get("acc") or {}, st.get("upd") or {}
        new_best = better(dev["accuracy"], dev["ce"], st.get("best"))
        gsum = self.eng.gain_summary()
        if new_best:
            self.best_arrays = {"W": self.W.copy(), "b": self.b.copy(), **self.eng.gain_arrays("g")}
            st["best"] = {"epoch": e, "dev_accuracy": dev["accuracy"], "dev_ce": dev["ce"], "dev_confusion": dev["confusion"],
                          "params_sha256": self.params_sha(), "gains_sha256": self.eng.gains_sha(), "mirrored_seq": None, "utc": utc_now()}
        n_ = max(1, int(a.get("n", 0)))
        nb = max(1, int(u.get("n", 0)))
        dist = {L.key: float(np.sqrt(sum(float((self.eng.tt.to_np(p.res).astype(np.float64) ** 2).sum()) for p in L.parts())))
                for L in self.eng.layers}
        row = {"cond": self.cond, "epoch": e, "train_ce_running": (float(a.get("ce_sum", 0.0)) / n_) if a.get("n") else None,
               "train_accuracy_running": (int(a.get("correct", 0)) / n_) if a.get("n") else None, "train_n": int(a.get("n", 0)),
               "dev_accuracy": dev["accuracy"], "dev_ce": dev["ce"], "new_best": bool(new_best), "best_epoch": st["best"]["epoch"],
               "perm_sha256": st.get("perm_sha256_by_epoch", {}).get(str(e)),
               "upd_res_l2_mean": (float(u.get("res_l2_sum", 0.0)) / nb) if u.get("n") else None,
               "upd_W_l2_mean": (float(u.get("W_l2_sum", 0.0)) / nb) if u.get("n") else None,
               "upd_b_l2_mean": (float(u.get("b_l2_sum", 0.0)) / nb) if u.get("n") else None,
               "upd_res_l2_mean_by_layer": {k: float(v) / nb for k, v in (u.get("res_sq_sum_by_layer") or {}).items()} if u.get("n") else {},
               "res_dist_from_init_l2_by_layer": dist, "res_dist_from_init_l2": float(np.sqrt(sum(v * v for v in dist.values()))),
               "gain_changed": int(sum(v["n_changed"] for v in gsum.values())), "gain_ge_1pct": int(sum(v["n_ge_1pct"] for v in gsum.values())),
               "gain_ge_10pct": int(sum(v["n_ge_10pct"] for v in gsum.values())), "gain_at_min": int(sum(v["n_at_min"] for v in gsum.values())),
               "gain_at_max": int(sum(v["n_at_max"] for v in gsum.values())), "gain_by_layer": gsum,
               "epoch_train_s": round(train_s, 3), "epoch_dev_eval_s": round(dev_s, 3), "utc": utc_now()}
        st["history"].append(row)
        st["epoch"], st["next_batch"], st["acc"], st["upd"] = e, 0, {}, {}
        return row

    # ---------------------------------------------------------------- 실행
    def execute(self) -> dict[str, Any]:
        run, eng = self.run, self.eng
        ensure_dir(self.dir)
        self._load()
        mm_tr, y_tr = run.cache("train")
        mm_dv, y_dv = run.cache("dev")
        n = len(y_tr)
        nb = int(math.ceil(n / self.B))
        st = self.st
        es = eng.structure()
        say(f"\n[{'train_trace' if self.cond == 'TRACE' else 'train_bp'}] {self.cond}: {self.E} epoch × {nb:,} 배치 (배치 {self.B}), "
            f"학습 연결 {sum(x['learnable_edges'] for x in es):,} 개 (V2→V4 {es[0]['learnable_edges']:,}), η_head {self.eta_h} · η_edge {self.eta_e}, "
            f"장치 {eng.tt.dev}" + ("" if int(st["epoch"]) < 0 else f" — epoch {int(st['epoch']) + 1} 배치 {int(st['next_batch']) + 1} 부터 이어 간다"))
        if self.cond == "BP_FULL" and int(st["epoch"]) < 0:
            say("  BP_FULL: V2 캐시는 상수, V4·IT 의 min·ReLU·L6 (자기·이웃 항) 을 모두 미분한다 (관례는 config.json 의 conventions).")
        if int(st["epoch"]) < 0:
            t0 = time.perf_counter()
            dev = eval_split(eng, mm_dv, y_dv, self.W, self.b, self.B, run.stop, "epoch 0 dev")
            sync(eng.tt.dev)
            dev_s = time.perf_counter() - t0
            run.add_time(f"{COND_DIR[self.cond]}_dev_eval_s", dev_s)
            row = self._epoch_end(0, dev, dev_s, 0.0)
            self.save()
            write_epochs_csv(run)
            say(f"  [epoch 0/{self.E}] (학습 전) dev 정확도 {fmt_pct(row['dev_accuracy'])} · CE {row['dev_ce']:.4f}")
        S = run.S["save"]
        every, min_s, prog = int(S["every_batches"]), float(S["min_seconds"]), int(S["progress_every_batches"])
        t_save = time.perf_counter()
        while int(st["epoch"]) < self.E:
            e = int(st["epoch"]) + 1
            perm = epoch_permutation_main(run.seed, e, n)
            st.setdefault("perm_sha256_by_epoch", {})[str(e)] = sha256_array(perm)
            t_ep = t_mark = time.perf_counter()
            b0 = int(st["next_batch"])
            for bi in range(b0, nb):
                idx = perm[bi * self.B:(bi + 1) * self.B]
                y = y_tr[idx]
                todo, Wn, bn, ce_sum, correct, dwb = self.step(eng.tt.f(mm_tr[idx]), y)
                with run.stop.critical():                                      # W·b·res 를 함께 적용 (반쯤 적용된 상태를 남기지 않는다)
                    self._upd_record(todo, dwb)
                    eng.commit(todo)
                    self.W, self.b = Wn, bn
                    a = st.setdefault("acc", {})
                    a["n"] = int(a.get("n", 0)) + len(y)
                    a["ce_sum"] = float(a.get("ce_sum", 0.0)) + ce_sum
                    a["correct"] = int(a.get("correct", 0)) + correct
                    st["next_batch"] = bi + 1
                if (bi + 1 < nb and (bi + 1) % every == 0 and time.perf_counter() - t_save >= min_s) or run.stop.requested:
                    sync(eng.tt.dev)
                    run.add_time(f"{COND_DIR[self.cond]}_train_s", time.perf_counter() - t_mark)
                    self.save()
                    t_mark = time.perf_counter()
                    t_save = time.perf_counter()
                if run.stop.requested:
                    raise Interrupted(f"{self.cond} epoch {e} 배치 {bi + 1}/{nb} 에서 멈춤 (배치 전체 갱신 뒤 저장됨)")
                if prog and (bi + 1) % prog == 0 and bi + 1 < nb:
                    a = st["acc"]
                    say(f"    {self.cond} epoch {e} 배치 {bi + 1:,}/{nb:,} · train CE (진행 평균) {a['ce_sum'] / max(1, a['n']):.4f} · "
                        f"정확도 {fmt_pct(a['correct'] / max(1, a['n']))}")
            sync(eng.tt.dev)
            train_s = time.perf_counter() - t_ep
            run.add_time(f"{COND_DIR[self.cond]}_train_s", time.perf_counter() - t_mark)
            self.save()
            t_dev = time.perf_counter()
            dev = eval_split(eng, mm_dv, y_dv, self.W, self.b, self.B, run.stop, f"{self.cond} epoch {e} dev")
            sync(eng.tt.dev)
            dev_s = time.perf_counter() - t_dev
            run.add_time(f"{COND_DIR[self.cond]}_dev_eval_s", dev_s)
            row = self._epoch_end(e, dev, dev_s, train_s)
            self.save()
            write_epochs_csv(run)
            say(f"  [{self.cond} epoch {e}/{self.E}] train CE {row['train_ce_running']:.4f} · 정확도 {fmt_pct(row['train_accuracy_running'])} "
                f"(진행 평균) | dev {fmt_pct(row['dev_accuracy'])} · CE {row['dev_ce']:.4f} | best epoch {row['best_epoch']}"
                f"{' (새 최선)' if row['new_best'] else ''} | 갱신 ‖Δres‖ {fmt_f(row['upd_res_l2_mean'], 3)} | 학습 {fmt_s(train_s)} · dev {fmt_s(dev_s)}")
            if run.stop.requested and e < self.E:
                raise Interrupted(f"{self.cond} epoch {e} 평가·저장 뒤 멈춤")
        b = st["best"]
        return {"cond": self.cond, "completed_epochs": int(st["epoch"]), "best_epoch": b["epoch"], "best_dev_accuracy": b["dev_accuracy"],
                "best_dev_ce": b["dev_ce"], "best_params_sha256": b["params_sha256"], "start_params_sha256": st["start_params_sha256"],
                "last_params_sha256": st.get("params_sha256"), "perm_sha256_by_epoch": st.get("perm_sha256_by_epoch")}


# ======================================================================
# 상태별 특징 캐시: V4·IT 6 층의 실제 다음 층 전송값 (q·L6 적용, 정규화 전 float32). 한 상태씩 만들고, 그 상태의 판독기·raw refit 이
#   끝나면 지운다 (정규화한 사본은 만들지 않는다 — 읽을 때 변환). 같은 forward 에서 그 상태 원래 head 의 고정 train/dev 재평가도 한다.
# ======================================================================
class FeatureStore:
    FORMAT = "cortex_v13/features/1"

    def __init__(self, run: Any, state: str):
        self.run, self.state = run, state
        self.dir = run.dir / "work" / "features" / state
        self.man_path = self.dir / "features.json"
        self._verified: set[str] = set()

    def key(self) -> dict[str, Any]:
        run = self.run
        eng = run.engine()
        return {"format": self.FORMAT, "input_id": run.input_id, "state": self.state, "params_sha256": run.state_params(self.state)["params_sha256"],
                "layers": list(LEARN_KEYS), "dims": {L.key: L.n for L in eng.layers}, "boundary": "실제 다음 층 전송값 (q·L6 적용, 정규화 전)",
                "dtype": "float32", "backend": run.backend, "numeric_impl": NUMERIC_IMPL, "feature_batch": int(run.S["compute"]["feature_batch"]),
                "rows": {s: int(len(run.labels(s))) for s in ("train", "dev")}}

    def file(self, layer: str, split: str) -> Path:
        return self.dir / f"{layer.replace('/', '_')}_{split}.npy"

    def _man(self) -> dict[str, Any]:
        return read_json(self.man_path) if self.man_path.is_file() else {}

    def bytes_needed(self) -> int:
        eng = self.run.engine()
        return sum(L.n for L in eng.layers) * sum(len(self.run.labels(s)) for s in ("train", "dev")) * 4

    def build(self) -> dict[str, Any]:
        run = self.run
        eng = run.engine()
        key = self.key()
        ksha = sha256_text(dumps(key, indent=None))
        man = self._man()
        if man.get("key_sha256") != ksha:
            if man:
                say(f"  {self.state} 특징 캐시 키가 달라 처음부터 다시 만든다.")
            self.release(quiet=True)
            ensure_dir(self.dir)
            free = shutil.disk_usage(self.dir).free
            need = self.bytes_needed()
            say(f"  {self.state} 특징 캐시 (6 층 × train·dev, float32): {fmt_bytes(need)} 필요 · 여유 {fmt_bytes(free)}")
            if need > free * 0.98:
                raise V13Error(f"디스크 여유 ({fmt_bytes(free)}) 가 특징 캐시 {fmt_bytes(need)} 보다 작다. 공간을 비우거나 다른 결과 폴더를 쓴다.")
            man = {"format": self.FORMAT, "key": key, "key_sha256": ksha, "created_utc": utc_now(),
                   "splits": {s: {"rows_done": 0, "complete": False} for s in ("train", "dev")}, "eval_partial": {}, "files": {}}
            write_json(self.man_path, man)
        prm = run.state_params(self.state)
        run.load_state_into_engine(self.state)
        W, b = prm["W"], prm["b"]
        FB = int(run.S["compute"]["feature_batch"])
        t0 = time.perf_counter()
        for split in ("train", "dev"):
            ent = man["splits"][split]
            if ent.get("complete"):
                continue
            mm_in, y = run.cache(split)
            n = len(y)
            done = int(ent.get("rows_done", 0))
            mode = "r+" if done > 0 else "w+"
            mms = {L.key: np.lib.format.open_memmap(self.file(L.key, split), mode=mode, dtype=np.float32, shape=(n, L.n)) for L in eng.layers}
            acc = EvalAcc()
            part = (man.get("eval_partial") or {}).get(split)
            if done and part:
                acc.n, acc.correct, acc.ce = int(part["n"]), int(part["correct"]), float(part["ce"])
                acc.conf = np.asarray(part["conf"], np.int64)
            say(f"  {self.state} 특징 {split}: {n:,} 행 " + (f"({done:,} 행 다음부터 이어 만든다)" if done else "만들기 시작"))
            t_last = time.perf_counter()
            try:
                with torch.no_grad():
                    for a0, a1 in chunk_rows(n, FB):
                        if a1 <= done:
                            continue
                        if run.stop.requested:
                            raise Interrupted(f"{self.state} 특징 {split} {done:,}/{n:,} 행에서 멈춤")
                        rec = eng.forward(eng.tt.f(mm_in[a0:a1]), record="outs")
                        outs = {k: eng.tt.to_np(v["out"]) for k, v in rec["layers"].items()}
                        if not all(np.isfinite(o).all() for o in outs.values()):
                            raise NumericalError(f"{self.state} 특징 {split} 행 {a0}~{a1} 에 NaN/Inf")
                        _z, p, logp = head_forward(W, b, outs["IT/L3"].astype(np.float64))
                        with run.stop.critical():                              # 행 쓰기·평가 누적·완료 행 수를 함께 (반쯤 반영된 조각을 남기지 않는다)
                            for k, o in outs.items():
                                mms[k][a0:a1] = o
                            acc.add(p, logp, y[a0:a1])
                            done = a1
                        if time.perf_counter() - t_last >= 20.0 or a1 == n:
                            self._checkpoint(mms, man, ent, split, done, acc)
                            t_last = time.perf_counter()
            finally:
                self._checkpoint(mms, man, ent, split, done, acc)
                del mms
                gc.collect()
            res = acc.result()
            with run.stop.critical():
                for L in eng.layers:
                    p_ = self.file(L.key, split)
                    man["files"][p_.name] = {"sha256": sha256_file(p_), "bytes": int(p_.stat().st_size)}
                    self._verified.add(p_.name)
                ent.update(rows_done=n, complete=True, completed_utc=utc_now())
                man.setdefault("fixed_eval", {})[split] = res
                write_json(self.man_path, man)
        run.add_time("features_s", time.perf_counter() - t0)
        fe = man.get("fixed_eval") or {}
        return {"state": self.state, "fixed_train": fe.get("train"), "fixed_dev": fe.get("dev"), "key_sha256": ksha,
                "bytes": sum(int(v["bytes"]) for v in man["files"].values())}

    def _checkpoint(self, mms: dict[str, np.ndarray], man: dict[str, Any], ent: dict[str, Any], split: str, done: int, acc: EvalAcc) -> None:
        """memmap 을 먼저 flush 한 뒤 완료 행 수·평가 누적을 기록한다 (기록된 행은 항상 디스크에 있다)."""
        for m in mms.values():
            m.flush()
        with self.run.stop.critical():
            ent["rows_done"] = int(done)
            man.setdefault("eval_partial", {})[split] = {"n": acc.n, "correct": acc.correct, "ce": acc.ce, "conf": acc.conf.tolist()}
            write_json(self.man_path, man)

    def open(self, layer: str, split: str) -> np.ndarray:
        man = self._man()
        key_sha = sha256_text(dumps(self.key(), indent=None))
        p = self.file(layer, split)
        if man.get("key_sha256") != key_sha or not (man.get("splits") or {}).get(split, {}).get("complete") or not p.is_file():
            raise IntegrityError(f"{self.state} 의 {layer} {split} 특징 캐시가 완성되지 않았거나 키가 다르다.")
        ent = (man.get("files") or {}).get(p.name) or {}
        if p.stat().st_size != ent.get("bytes"):
            raise IntegrityError(f"{p.name}: 파일 크기가 기록과 다르다.")
        if p.name not in self._verified:
            if sha256_file(p) != ent.get("sha256"):
                raise IntegrityError(f"{p.name}: SHA256 이 기록과 다르다 (바뀐 특징 캐시).")
            self._verified.add(p.name)
        return np.load(p, mmap_mode="r")

    def key_sha(self) -> str:
        return sha256_text(dumps(self.key(), indent=None))

    def release(self, quiet: bool = False) -> int:
        """특징 파일을 지운다 (판독기 결과·정규화 통계·지표는 각 probe 폴더에 남는다)."""
        freed = 0
        gc.collect()
        if self.dir.is_dir():
            for p in sorted(self.dir.glob("*.npy")):
                for k in range(4):
                    try:
                        freed += p.stat().st_size
                        p.unlink()
                        break
                    except FileNotFoundError:
                        break
                    except OSError as exc:                                     # Windows: 아직 열린 memmap → 정리 후 다시
                        gc.collect()
                        time.sleep(0.25 * (k + 1))
                        if k == 3 and not quiet:
                            say(f"  [알림] {p.name} 을 지우지 못했다 ({exc}). 결과와 무관하다 — 나중에 직접 지워도 된다.")
            with contextlib.suppress(OSError):
                self.man_path.unlink()
            with contextlib.suppress(OSError):
                self.dir.rmdir()                                               # 비어 있으면 폴더도 지운다
        return freed


# ======================================================================
# 선형 판독기 (d → 10 softmax + bias) — float64, train 전체의 결정적 objective, L-BFGS (history 10, lr 1, strong-Wolfe)
#   objective = mean CE + (λ/2)·ΣW² (bias 제외). closure 하나 = 같은 train 전체를 같은 순서로 microbatch 512 누적 (CE 합 / N).
#   LBFGS(max_iter=1) 의 step 을 바깥 루프에서 부르고, 수락된 상태마다 dev 로 best 를 고른다 (초기 상태 포함, dev 는 closure 에 쓰지 않는다).
#   같은 파라미터 (해시) 의 평가는 다시 계산하지 않고 재사용한다 (예산은 실제 full-train 계산 횟수). line search 도중 예산 소진·중단이면
#   직전 수락 파라미터와 optimizer 상태로 되돌린다 (쓴 closure 횟수·시간은 되돌리지 않는다).
# ======================================================================
class StepAbort(Exception):
    """Ctrl+C 로 line search 를 멈춤 (직전 수락 상태로 되돌린다)."""


def column_stats(mm: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """train 만으로 float64 평균·표준편차 (두 번 읽기: 평균 → 편차 제곱 평균). 상수열은 σ = 0 (정확히)."""
    n, d = mm.shape
    s = np.zeros(d, np.float64)
    for a0, a1 in chunk_rows(n, chunk):
        s += np.asarray(mm[a0:a1], np.float64).sum(axis=0)
    mu = s / n
    ss = np.zeros(d, np.float64)
    for a0, a1 in chunk_rows(n, chunk):
        dl = np.asarray(mm[a0:a1], np.float64) - mu
        ss += (dl * dl).sum(axis=0)
    return mu, np.sqrt(ss / n)


class LBFGSProbe:
    FORMAT = "cortex_v13/probe_last/1"

    def __init__(self, run: Any, *, pid: str, kind: str, state: str, layer: str, out_dir: Path, X_tr: np.ndarray, y_tr: np.ndarray,
                 X_dv: np.ndarray, y_dv: np.ndarray, standardize: bool, penalty: float, W0: np.ndarray | None, b0: np.ndarray | None,
                 feature_key: dict[str, Any], shared_states: Sequence[str] = ()):
        self.run, self.pid, self.kind, self.state, self.layer = run, pid, kind, state, layer
        self.out_dir = Path(out_dir)
        self.X_tr, self.X_dv = X_tr, X_dv
        self.N, self.d = int(X_tr.shape[0]), int(X_tr.shape[1])
        P = run.S["probe"]
        self.max_iter, self.max_evals = int(P["max_iter"]), int(P["max_evals"])
        self.mb, self.grad_tol, self.tol_change = int(P["microbatch"]), float(P["grad_tol"]), float(P["tolerance_change"])
        self.floor = float(P["sigma_floor"])
        self.standardize, self.penalty = bool(standardize), float(penalty)
        self.feature_key = feature_key
        self.fkey_sha = sha256_text(dumps(feature_key, indent=None))
        self.shared_states = list(shared_states)
        self.dev = run.device
        self.W_init = np.zeros((N_CLASSES, self.d), np.float64) if W0 is None else np.asarray(W0, np.float64).copy()
        self.b_init = np.zeros(N_CLASSES, np.float64) if b0 is None else np.asarray(b0, np.float64).copy()
        self.y_tr_np, self.y_dv_np = np.asarray(y_tr, np.int64), np.asarray(y_dv, np.int64)
        self.mu_t = self.sig_t = None
        self.norm_info: dict[str, Any] = {}
        self.cache: dict[str, dict[str, Any]] = {}
        self.computed = self.calls = 0
        self.it = 0
        self.history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.best_W = self.best_b = None
        self.cur_key = ""
        self.aborts = 0
        self.time_s = 0.0
        self.termination: dict[str, Any] | None = None

    # ---------------------------------------------------------------- 자료
    def _normalization(self) -> None:
        p = self.out_dir / "normalization.npz"
        if not self.standardize:
            self.norm_info = {"standardize": False, "note_ko": "정규화 없음 (raw 특징 그대로)"}
            return
        if p.is_file():
            arrs, meta = load_npz(p)
            if (meta or {}).get("feature_key_sha256") != self.fkey_sha:
                raise IntegrityError(f"{p} 의 특징 키가 지금 특징과 다르다.")
            mu, sigma = arrs["mu"], arrs["sigma"]
            self.norm_info = meta
        else:
            t0 = time.perf_counter()
            mu, sigma = column_stats(self.X_tr)
            self.norm_info = {"format": "cortex_v13/normalization/1", "standardize": True, "feature_key_sha256": self.fkey_sha,
                              "train_rows": self.N, "dim": self.d, "sigma_floor": self.floor, "n_constant": int((sigma == 0).sum()),
                              "n_small_sigma": int(((sigma > 0) & (sigma < self.floor)).sum()),
                              "n_all_zero": int(((sigma == 0) & (mu == 0)).sum()), "mu_sha256": sha256_array(mu), "sigma_sha256": sha256_array(sigma),
                              "rule_ko": "x' = (x − μ) / max(σ, 1e-6). μ·σ 는 이 상태·층의 train 특징만으로 float64 두 번 읽기 (dev/test 로 추정하지 않음).",
                              "time_s": round(time.perf_counter() - t0, 3)}
            save_npz(p, {"mu": mu, "sigma": sigma}, self.norm_info)
        self.mu_np, self.sigma_np = np.asarray(mu, np.float64), np.asarray(sigma, np.float64)
        self.mu_t = torch.from_numpy(np.ascontiguousarray(mu, np.float64)).to(self.dev)
        self.sig_t = torch.from_numpy(np.ascontiguousarray(np.maximum(sigma, self.floor), np.float64)).to(self.dev)

    def _x(self, X: np.ndarray, a0: int, a1: int) -> Any:
        x = torch.from_numpy(np.array(X[a0:a1], dtype=np.float32, order="C")).to(self.dev).to(torch.float64)    # 읽기 전용 memmap → 복사본
        return (x - self.mu_t) / self.sig_t if self.standardize else x

    def _phash(self) -> str:
        return sha256_array(self.W.detach().cpu().numpy(), self.b.detach().cpu().numpy())

    # ---------------------------------------------------------------- closure (train 전체)
    def _evaluate(self) -> dict[str, Any]:
        W, b = self.W, self.b
        W.grad = None
        b.grad = None
        ce_total, correct = 0.0, 0
        for a0, a1 in chunk_rows(self.N, self.mb):
            if self.run.stop.requested:
                raise StepAbort()
            x = self._x(self.X_tr, a0, a1)
            z = x @ W.T + b
            ce = -(torch.log_softmax(z, dim=1) * self.oh_tr[a0:a1]).sum()
            (ce / self.N).backward()                                           # microbatch 그래프는 여기서 바로 해제된다
            ce_total += float(ce.item())
            correct += int((z.detach().argmax(dim=1) == self.yt_tr[a0:a1]).sum().item())
        pen = 0.0
        if self.penalty > 0:
            pt = 0.5 * self.penalty * (W * W).sum()                            # penalty 는 closure 전체에 한 번 (bias 제외)
            pt.backward()
            pen = float(pt.item())
        return {"obj": ce_total / self.N + pen, "ce": ce_total / self.N, "correct": correct, "pen": pen,
                "gW": W.grad.detach().clone(), "gb": b.grad.detach().clone()}

    def _set_grad(self, ent: dict[str, Any]) -> None:
        self.W.grad = ent["gW"].clone()
        self.b.grad = ent["gb"].clone()

    def closure(self) -> Any:
        self.calls += 1
        key = self._phash()
        ent = self.cache.get(key)
        if ent is None:
            if self.computed >= self.max_evals:
                raise BudgetExhausted()
            if self.run.stop.requested:
                raise StepAbort()
            ent = self._evaluate()
            self.computed += 1
            self.cache[key] = ent
        self._set_grad(ent)
        return torch.tensor(ent["obj"], dtype=torch.float64)

    def _grad_norms(self, ent: dict[str, Any]) -> tuple[float, float]:
        g = torch.cat([ent["gW"].reshape(-1), ent["gb"].reshape(-1)])
        return float(g.abs().max().item()), float(torch.linalg.vector_norm(g).item())

    @_nograd
    def _dev_eval(self, W: Any, b: Any) -> dict[str, Any]:
        correct, ce_sum = 0, 0.0
        n = int(self.X_dv.shape[0])
        for a0, a1 in chunk_rows(n, self.mb):
            z = self._x(self.X_dv, a0, a1) @ W.T + b
            ce_sum += float(-(torch.log_softmax(z, dim=1) * self.oh_dv[a0:a1]).sum().item())
            correct += int((z.argmax(dim=1) == self.yt_dv[a0:a1]).sum().item())
        return {"accuracy": correct / n, "ce": ce_sum / n, "n": n}

    # ---------------------------------------------------------------- 수락 상태 기록
    def _record(self, step_t: Any) -> None:
        ent = self.cache[self.cur_key]
        g_inf, g_l2 = self._grad_norms(ent)
        dv = self._dev_eval(self.W, self.b)
        row = {"pid": self.pid, "kind": self.kind, "state": self.state, "layer": self.layer, "iteration": self.it,
               "closures_computed": self.computed, "closure_calls": self.calls, "objective": ent["obj"], "train_ce": ent["ce"],
               "train_accuracy": ent["correct"] / self.N, "penalty": ent["pen"], "grad_inf": g_inf, "grad_l2": g_l2,
               "step_t": None if step_t is None else float(step_t), "dev_accuracy": dv["accuracy"], "dev_ce": dv["ce"], "new_best": False}
        if better(dv["accuracy"], dv["ce"], self.best):
            row["new_best"] = True
            self.best = {k: row[k] for k in ("iteration", "objective", "train_ce", "train_accuracy", "grad_inf", "grad_l2", "dev_accuracy",
                                            "dev_ce", "closures_computed")}
            self.best_W, self.best_b = self.W.detach().clone(), self.b.detach().clone()
        self.history.append(row)

    # ---------------------------------------------------------------- 저장·복원
    def _meta(self) -> dict[str, Any]:
        cur = self.cache.get(self.cur_key) or {}
        return {"format": self.FORMAT, "pid": self.pid, "kind": self.kind, "state": self.state, "layer": self.layer,
                "feature_key_sha256": self.fkey_sha, "dim": self.d, "n_train": self.N, "it": self.it, "computed": self.computed,
                "calls": self.calls, "aborts": self.aborts, "history": self.history, "best": self.best, "cur_key": self.cur_key,
                "cur": {k: cur.get(k) for k in ("obj", "ce", "correct", "pen")}, "termination": self.termination,
                "time_s": self.time_s, "budget": {"max_iter": self.max_iter, "max_evals": self.max_evals}}

    def _save(self) -> None:
        with self.run.stop.critical():
            cur = self.cache[self.cur_key]
            obj = {"meta_json": dumps(self._meta(), indent=None), "W": self.W.detach().cpu(), "b": self.b.detach().cpu(),
                   "opt": self.opt.state_dict(), "cur_gW": cur["gW"].cpu(), "cur_gb": cur["gb"].cpu(),
                   "best_W": self.best_W.cpu(), "best_b": self.best_b.cpu()}
            buf = io.BytesIO()
            torch.save(obj, buf)
            atomic_write_bytes(self.out_dir / "last.pt", buf.getvalue())

    def _load(self) -> bool:
        p = self.out_dir / "last.pt"
        if not p.is_file():
            return False
        try:
            obj = torch.load(io.BytesIO(p.read_bytes()), map_location=self.dev, weights_only=True)
            meta = json.loads(obj["meta_json"])
        except Exception as exc:                                               # noqa: BLE001
            raise IntegrityError(f"{p} 를 읽을 수 없다 (손상): {type(exc).__name__}: {exc}") from exc
        if meta.get("format") != self.FORMAT or meta.get("pid") != self.pid or meta.get("feature_key_sha256") != self.fkey_sha:
            raise IntegrityError(f"{p} 는 이 판독기·특징의 상태가 아니다.")
        if meta.get("budget") != {"max_iter": self.max_iter, "max_evals": self.max_evals}:
            raise IntegrityError(f"{p} 의 예산 {meta.get('budget')} 이 이 실행의 설정과 다르다 (예산을 몰래 바꾸지 않는다).")
        with torch.no_grad():
            self.W.copy_(obj["W"].to(self.dev))
            self.b.copy_(obj["b"].to(self.dev))
        self.opt.load_state_dict(obj["opt"])
        self.it, self.computed, self.calls, self.aborts = int(meta["it"]), int(meta["computed"]), int(meta["calls"]), int(meta["aborts"])
        self.history, self.best, self.time_s, self.termination = meta["history"], meta["best"], float(meta["time_s"]), meta["termination"]
        self.cur_key = meta["cur_key"]
        if self._phash() != self.cur_key:
            raise IntegrityError(f"{p}: 저장된 파라미터 해시가 기록과 다르다.")
        c = meta["cur"]
        self.cache = {self.cur_key: {"obj": c["obj"], "ce": c["ce"], "correct": c["correct"], "pen": c["pen"],
                                     "gW": obj["cur_gW"].to(self.dev), "gb": obj["cur_gb"].to(self.dev)}}
        self.best_W, self.best_b = obj["best_W"].to(self.dev), obj["best_b"].to(self.dev)
        return True

    # ---------------------------------------------------------------- 실행
    def execute(self) -> dict[str, Any]:
        run = self.run
        ensure_dir(self.out_dir)
        P = run.S["probe"]
        self._normalization()
        self.oh_tr = torch.from_numpy(onehot(self.y_tr_np)).to(self.dev)
        self.oh_dv = torch.from_numpy(onehot(self.y_dv_np)).to(self.dev)
        self.yt_tr = torch.from_numpy(self.y_tr_np).to(self.dev)
        self.yt_dv = torch.from_numpy(self.y_dv_np).to(self.dev)
        self.W = torch.from_numpy(self.W_init.copy()).to(self.dev).requires_grad_(True)
        self.b = torch.from_numpy(self.b_init.copy()).to(self.dev).requires_grad_(True)
        self.opt = torch.optim.LBFGS([self.W, self.b], lr=float(P["lr"]), max_iter=1, max_eval=1 + int(P["line_search_evals"]),
                                     tolerance_grad=self.grad_tol, tolerance_change=self.tol_change, history_size=int(P["history_size"]),
                                     line_search_fn="strong_wolfe")
        resumed = self._load()
        label = f"{self.kind} {self.state} {self.layer}"
        if resumed:
            say(f"  {label}: 수락 iteration {self.it} · closure {self.computed}/{self.max_evals} 에서 이어 간다")
        t0 = time.perf_counter()
        try:
            if self.termination is None:
                if not resumed:
                    self.cur_key = self._phash()
                    try:
                        with torch.enable_grad():
                            self.closure()                                      # 초기 상태 (iteration 0) 의 objective·gradient
                    except StepAbort:
                        raise Interrupted(f"{label}: 초기 평가 도중 멈춤 (저장할 수락 상태가 아직 없다 — 이어 갈 때 처음부터)") from None
                    self._record(None)
                    self._save()
                self._loop(label)
        finally:
            self.time_s += time.perf_counter() - t0
            run.add_time("probe_s" if self.kind == "probe" else "raw_refit_s", time.perf_counter() - t0)
        return self._finish()

    def _loop(self, label: str) -> None:
        t_save = time.perf_counter()
        while True:
            ent = self.cache[self.cur_key]
            g_inf, _g_l2 = self._grad_norms(ent)
            if g_inf <= self.grad_tol:
                self.termination = {"status": "converged", "detail_ko": f"gradient 무한 노름 {g_inf:.3e} ≤ {self.grad_tol:g}"}
                break
            if self.it >= self.max_iter:
                self.termination = {"status": "budget_exhausted", "detail_ko": f"수락 iteration 상한 {self.max_iter} 에 도달 (gradient 무한 노름 {g_inf:.3e})"}
                break
            if self.computed >= self.max_evals:
                self.termination = {"status": "budget_exhausted", "detail_ko": f"closure 예산 {self.max_evals} 을 모두 씀"}
                break
            if self.run.stop.requested:
                self._save()
                raise Interrupted(f"{label}: 수락 iteration {self.it} 뒤 멈춤")
            saved_W, saved_b = self.W.detach().clone(), self.b.detach().clone()
            saved_opt = copy.deepcopy(self.opt.state_dict())
            prev_obj = ent["obj"]
            try:
                self.opt.step(self.closure)
            except (BudgetExhausted, StepAbort) as exc:
                with torch.no_grad():                                           # 미완료 trial 을 버리고 직전 수락 상태로
                    self.W.copy_(saved_W)
                    self.b.copy_(saved_b)
                self.opt.load_state_dict(saved_opt)
                self.cache = {self.cur_key: self.cache[self.cur_key]}
                if isinstance(exc, StepAbort):
                    self.aborts += 1
                    self._save()
                    raise Interrupted(f"{label}: line search 도중 멈춤 → 직전 수락 상태 (iteration {self.it}) 로 되돌려 저장") from None
                self.termination = {"status": "budget_exhausted", "detail_ko": f"line search 도중 closure 예산 {self.max_evals} 소진 → 직전 수락 상태로 복원"}
                break
            new_key = self._phash()
            if new_key == self.cur_key:
                self.termination = {"status": "stalled", "detail_ko": "line search 가 움직이지 못했다 (step 0 또는 하강 방향 아님)"}
                break
            if new_key not in self.cache:                                       # 보통은 line search 가 이미 계산한 점 (해시로 재사용)
                try:
                    with torch.enable_grad():
                        self.closure()
                except (BudgetExhausted, StepAbort) as exc:
                    with torch.no_grad():
                        self.W.copy_(saved_W)
                        self.b.copy_(saved_b)
                    self.opt.load_state_dict(saved_opt)
                    self.cache = {self.cur_key: self.cache[self.cur_key]}
                    if isinstance(exc, StepAbort):
                        self.aborts += 1
                        self._save()
                        raise Interrupted(f"{label}: 수락 점 확인 평가 도중 멈춤 → 직전 수락 상태로 되돌려 저장") from None
                    self.termination = {"status": "budget_exhausted", "detail_ko": "수락 점 확인 평가에 쓸 closure 예산이 없다 → 직전 수락 상태로 복원"}
                    break
            with torch.no_grad():
                dmax = max(float((self.W.detach() - saved_W).abs().max().item()), float((self.b.detach() - saved_b).abs().max().item()))
            self.cur_key = new_key
            self.cache = {new_key: self.cache[new_key]}
            self.it += 1
            st = self.opt.state[self.opt._params[0]]
            self._record(st.get("t"))
            obj = self.cache[new_key]["obj"]
            if abs(obj - prev_obj) < self.tol_change or dmax <= self.tol_change:     # PyTorch LBFGS 의 진행 부족 조건과 같은 기준
                g_now, _ = self._grad_norms(self.cache[new_key])
                if g_now <= self.grad_tol:
                    self.termination = {"status": "converged", "detail_ko": f"gradient 무한 노름 {g_now:.3e} ≤ {self.grad_tol:g}"}
                else:
                    self.termination = {"status": "stalled", "detail_ko": f"진행 부족: objective 변화 {abs(obj - prev_obj):.3e}, 최대 파라미터 변화 {dmax:.3e} "
                                                                          f"(기준 {self.tol_change:g}); gradient 무한 노름 {g_now:.3e} > {self.grad_tol:g}"}
                break
            if time.perf_counter() - t_save >= 30.0:
                self._save()
                t_save = time.perf_counter()
        self._save()

    def _finish(self) -> dict[str, Any]:
        fin = self.history[-1]
        best = self.best
        res = {"pid": self.pid, "kind": self.kind, "state": self.state, "layer": self.layer, "dim": self.d,
               "n_params": self.d * N_CLASSES + N_CLASSES, "n_train": self.N, "n_dev": int(self.X_dv.shape[0]),
               "standardize": self.standardize, "penalty": self.penalty, "normalization": {k: self.norm_info.get(k) for k in
                                                                                          ("n_constant", "n_small_sigma", "n_all_zero", "sigma_floor")},
               "termination": self.termination["status"], "termination_detail_ko": self.termination["detail_ko"],
               "accepted_iterations": self.it, "closures_computed": self.computed, "closure_calls": self.calls,
               "line_search_aborts_restored": self.aborts, "budget": {"max_iter": self.max_iter, "max_evals": self.max_evals},
               "final": {k: fin[k] for k in ("iteration", "objective", "train_ce", "train_accuracy", "grad_inf", "grad_l2", "dev_accuracy", "dev_ce")},
               "best": best, "best_is_final": best["iteration"] == fin["iteration"], "feature_key_sha256": self.fkey_sha,
               "shared_states": self.shared_states, "time_s": round(self.time_s, 3),
               "note_ko": "best = dev 정확도 최대 → CE 최소 → 이른 iteration (초기 상태 포함). 보고 CE 에는 penalty 를 넣지 않는다. "
                          "수렴 여부는 최종 상태 기준이며 best 의 gradient 노름은 best iteration 의 값이다."}
        Wb, bb = self.best_W.detach().cpu().numpy(), self.best_b.detach().cpu().numpy()
        res["best_head_sha256"] = sha256_array(Wb, bb)
        res["normalization_sha256"] = (sha256_text(sha256_array(self.mu_np) + sha256_array(self.sigma_np)) if self.standardize else None)
        save_npz(self.out_dir / "best.npz", {"W": Wb, "b": bb}, {"format": "cortex_v13/probe_best/1", "pid": self.pid, "best": best,
                                                                 "best_head_sha256": res["best_head_sha256"], "standardize": self.standardize,
                                                                 "feature_key_sha256": self.fkey_sha})
        write_json(self.out_dir / "result.json", res)
        write_json(self.out_dir / "history.json", self.history)
        return res


def probe_head_eval(X: Any, y: np.ndarray, W: np.ndarray, b: np.ndarray, mu: np.ndarray | None, sig: np.ndarray | None, acc: EvalAcc,
                    device: Any) -> None:
    """고정 판독기 평가 (float64 torch, 판독기 학습과 같은 변환). X: (n, d) float32 NumPy 또는 장치 텐서."""
    with torch.no_grad():
        x = (X if torch.is_tensor(X) else torch.from_numpy(np.array(X, dtype=np.float32, order="C"))).to(device).to(torch.float64)
        if mu is not None:
            x = (x - torch.from_numpy(mu).to(device)) / torch.from_numpy(sig).to(device)
        z = x @ torch.from_numpy(W).to(device).T + torch.from_numpy(b).to(device)
        logp = torch.log_softmax(z, dim=1).cpu().numpy()
    acc.add(np.exp(logp), logp, np.asarray(y, np.int64))


# ======================================================================
# 같은 상태 교정 진단: 같은 snapshot · 같은 forward · 같은 배치에서 TRACE 후보와 정확 gradient 를 둘 다 계산하고 적용하지 않는다.
#   표본: train 클래스별 10 개 = 100 개 (학습 순열과 별도의 고정 RNG), batch 16. gain-only 와 joint (W·b 포함) 를 분리한다.
#   진단 전용 변형: BP_STOP_L6 (L6 g 만 고정한 미분 — 정확한 전체 BP 아님), TRACE_MIN_AWARE (min 의 λ 를 score 에 곱해 수신별 재정규화).
# ======================================================================
PAIRS = (("TRACE", "BP_FULL"), ("TRACE_MIN_AWARE", "BP_FULL"), ("BP_STOP_L6", "BP_FULL"), ("TRACE", "BP_STOP_L6"))


def vec_compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """a = 비교 방향 (예: TRACE), b = 기준 방향 (예: −∂CE/∂res). 부호 일치는 양쪽 모두 충분히 0 이 아닌 성분만 (eps = 1e-12·max(1, max|v|))."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    n = int(len(a))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    ea, eb = 1e-12 * max(1.0, float(np.abs(a).max()) if n else 1.0), 1e-12 * max(1.0, float(np.abs(b).max()) if n else 1.0)
    za, zb = np.abs(a) <= ea, np.abs(b) <= eb
    both = ~za & ~zb
    nboth = int(both.sum())
    agree = int(((np.sign(a) == np.sign(b)) & both).sum())
    return {"n": n, "cos": float(a @ b / (na * nb)) if na > 0 and nb > 0 else None, "norm_a": na, "norm_b": nb,
            "rms_a": na / math.sqrt(n) if n else None, "rms_b": nb / math.sqrt(n) if n else None, "norm_ratio": na / nb if nb > 0 else None,
            "sign_agree": agree / nboth if nboth else None, "n_both_nonzero": nboth, "n_a_zero_only": int((za & ~zb).sum()),
            "n_b_zero_only": int((~za & zb).sum()), "n_both_zero": int((za & zb).sum())}


def dot_sign(grad: np.ndarray, delta: np.ndarray) -> tuple[str, float]:
    """dot(∂CE/∂θ, Δθ): 음수 = 국소 하강, 양수 = 상승, |dot| ≤ 1e-9·‖g‖·‖Δ‖ 이거나 영벡터면 판정 유보."""
    d = float(np.dot(grad, delta))
    ng, nd = float(np.linalg.norm(grad)), float(np.linalg.norm(delta))
    if ng == 0.0 or nd == 0.0:
        return "undetermined_zero_vector", d
    if abs(d) <= 1e-9 * ng * nd:
        return "undetermined", d
    return ("descent" if d < 0 else "ascent"), d


class Diagnostics:
    def __init__(self, run: Any, state: str):
        self.run, self.state = run, state
        self.D = run.S["diagnostics"]
        self.eng: Engine = run.engine()
        self.tt = self.eng.tt
        self.vidx = [np.flatnonzero(p.valid_np.ravel()) for p in self.eng.parts]
        self.layer_of = np.concatenate([np.full(len(v), LEARN_KEYS.index(p.layer), np.int64) for p, v in zip(self.eng.parts, self.vidx)])

    # ---------------------------------------------------------------- 표본
    def samples(self) -> np.ndarray:
        y = self.run.labels("train")
        rng = np.random.default_rng([int(self.run.seed), int(self.D["rng_tag"]), 13])
        pick = []
        for c in range(N_CLASSES):
            idx = np.flatnonzero(y == c)
            k = min(int(self.D["per_class"]), len(idx))
            if k:
                pick.append(np.sort(rng.choice(idx, size=k, replace=False)))
        sel = np.concatenate(pick) if pick else np.zeros(0, np.int64)
        return sel[rng.permutation(len(sel))]

    def flat(self, tensors: Sequence[Any]) -> np.ndarray:
        return np.concatenate([self.tt.to_np(t).astype(np.float64).ravel()[v] for t, v in zip(tensors, self.vidx)])

    def off_valid_max(self, tensors: Sequence[Any]) -> float:
        m = 0.0
        for t, p in zip(tensors, self.eng.parts):
            a = np.abs(self.tt.to_np(t).astype(np.float64))[~p.valid_np]
            if a.size:
                m = max(m, float(a.max()))
        return m

    # ---------------------------------------------------------------- 갱신 반올림·투영 분석
    def rounding(self, deltas: Sequence[Any]) -> dict[str, dict[str, Any]]:
        eng = self.eng
        out: dict[str, dict[str, Any]] = {}
        with torch.no_grad():
            for p, d in zip(eng.parts, deltas):
                raw = p.res + d
                new = torch.clamp(raw, min=eng.lo, max=eng.hi)
                gam_change = (new + 1.0) - (p.res + 1.0)
                v = p.valid_np
                dn, rawn = self.tt.to_np(d)[v], self.tt.to_np(raw)[v]
                rc, gc_ = self.tt.to_np(new - p.res)[v], self.tt.to_np(gam_change)[v].astype(np.float64)
                w0 = p.w0_np[v].astype(np.float64)
                o = out.setdefault(p.layer, {"n_valid": 0, "n_candidate_nonzero": 0, "n_clipped": 0, "n_vanished_in_res": 0,
                                             "n_vanished_in_gamma": 0, "_dw": []})
                nzc = dn != 0
                o["n_valid"] += int(v.sum())
                o["n_candidate_nonzero"] += int(nzc.sum())
                o["n_clipped"] += int(((rawn < eng.lo) | (rawn > eng.hi)).sum())
                o["n_vanished_in_res"] += int((nzc & (rc == 0)).sum())
                o["n_vanished_in_gamma"] += int((nzc & (gc_ == 0)).sum())
                o["_dw"].append(np.abs(w0 * gc_))
        return out

    # ---------------------------------------------------------------- 활동·min·L6 집계
    def activity(self, rec: dict[str, Any], acc: dict[str, dict[str, Any]]) -> None:
        eng, tt = self.eng, self.tt
        for li, L in enumerate(eng.layers):
            r = rec["layers"][L.key]
            o = acc.setdefault(L.key, {"samples": 0, "valid_neurons": int((~L.invalid_np).sum()), "inactive_structural": int(L.invalid_np.sum()),
                                       "zero_activity": 0, "min_neurons": 0, "min_tie": 0, "min_tie_zero": 0, "min_share_n": 0,
                                       "min_share_small_sum": 0.0, "l6_n": 0, "l6_at_min": 0, "l6_at_max": 0, "l6_exact_boundary": 0,
                                       "l6_rect_negative": 0, "l6_rect_zero": 0, "l6_no_neighbor": 0})
            s = tt.to_np(r["s"])
            B = s.shape[0]
            valid = ~L.invalid_np
            o["samples"] += B
            o["zero_activity"] += int((s[:, valid] == 0).sum())
            if L.B is not None:
                mn = L.op_min_np & valid
                sa, sb = tt.to_np(r["sa"]), tt.to_np(r["sb"])
                o["min_neurons"] = int(mn.sum())
                tie = (sa == sb)[:, mn]
                o["min_tie"] += int(tie.sum())
                o["min_tie_zero"] += int((tie & (sa[:, mn] == 0)).sum())
                x = rec["x0"] if li == 0 else rec["layers"][eng.layers[li - 1].key]["out"]
                xe = tt.pad_zero_col(x)
                with torch.no_grad():
                    SA = tt.to_np(L.A.apply(tt, xe, [p.weff.abs() for p in L.A.parts]))
                    SB = tt.to_np(L.B.apply(tt, xe, [p.weff.abs() for p in L.B.parts]))
                tot = SA + SB
                act = (s > 0) & (tot > 0) & (sa != sb) & mn[None, :]
                small = np.where(sa < sb, SA, SB)
                o["min_share_n"] += int(act.sum())
                o["min_share_small_sum"] += float((small[act] / tot[act]).sum())
            if L.l6 is not None and r.get("l6_x") is not None:
                gx = tt.to_np(r["l6_x"])
                hn = L.l6.has_nb_np
                o["l6_n"] += int(hn.sum()) * B
                o["l6_at_min"] += int((gx[:, hn] < L.l6.g_min).sum())
                o["l6_at_max"] += int((gx[:, hn] > L.l6.g_max).sum())
                o["l6_exact_boundary"] += int(((gx[:, hn] == L.l6.g_min) | (gx[:, hn] == L.l6.g_max)).sum())
                o["l6_no_neighbor"] += int((~hn).sum()) * B
                if r.get("l6_arg") is not None:
                    ag = tt.to_np(r["l6_arg"])
                    o["l6_rect_negative"] += int((ag < 0).sum())
                    o["l6_rect_zero"] += int((ag == 0).sum())

    @staticmethod
    def patterns(rec: dict[str, Any], eng: Engine) -> dict[str, np.ndarray]:
        """경계 판정용 이산 패턴 (ReLU 활성, min 가지, L6 clip 구역, 정류 부호)."""
        out = {}
        for L in eng.layers:
            r = rec["layers"][L.key]
            out[f"{L.key}.relu"] = eng.tt.to_np(r["pre"] > 0)
            if L.B is not None:
                sa, sb = eng.tt.to_np(r["sa"]), eng.tt.to_np(r["sb"])
                out[f"{L.key}.min"] = np.sign(sa - sb)[:, L.op_min_np]
            if r.get("l6_x") is not None:
                gx = eng.tt.to_np(r["l6_x"])
                out[f"{L.key}.clip"] = (gx < L.l6.g_min).astype(np.int8) - (gx > L.l6.g_max).astype(np.int8) * 2
                if r.get("l6_arg") is not None:
                    out[f"{L.key}.rect"] = eng.tt.to_np(r["l6_arg"]) < 0
        return out

    # ---------------------------------------------------------------- float64 그림자: 해석 gradient · VJP 대조 · 방향 유한차분 · 국소 CE 변화
    def shadow_checks(self, bi: int, idx: np.ndarray, W: np.ndarray, b: np.ndarray, vec: dict[str, Any], g32: np.ndarray) -> dict[str, Any]:
        run = self.run
        sh: Engine = run.shadow_engine()
        sh.set_res_list(self.eng.res_list())
        mm, ytr = run.cache("train")
        x64 = sh.tt.f(mm[idx])
        y = ytr[idx]
        n = len(y)
        oh = torch.from_numpy(onehot(y, n)).to(sh.tt.dev)
        sv = [np.flatnonzero(p.valid_np.ravel()) for p in sh.parts]

        def flat64(ts: Sequence[Any]) -> np.ndarray:
            return np.concatenate([t.detach().cpu().numpy().astype(np.float64).ravel()[v] for t, v in zip(ts, sv)])

        def unflat(v: np.ndarray) -> list[Any]:
            out, k = [], 0
            for p, ix in zip(sh.parts, sv):
                a = np.zeros(p.r * p.K, np.float64)
                a[ix] = v[k:k + len(ix)]
                k += len(ix)
                out.append(torch.from_numpy(a.reshape(p.r, p.K)).to(sh.tt.dev))
            return out

        base = [p.res.detach().clone() for p in sh.parts]
        Wt0, bt0 = torch.from_numpy(W.copy()).to(sh.tt.dev), torch.from_numpy(b.copy()).to(sh.tt.dev)

        def ce_at(dres: Sequence[Any] | None, dW: Any = None, db: Any = None, record: str = "none") -> tuple[float, dict[str, Any]]:
            with torch.no_grad():
                leaves = [r0 + d for r0, d in zip(base, dres)] if dres is not None else base
                rec = sh.forward(x64, leaves=leaves, record=record)
                Wt = Wt0 if dW is None else Wt0 + dW
                bt = bt0 if db is None else bt0 + db
                z = rec["h"] @ Wt.T + bt
                return float(-(torch.log_softmax(z, dim=1) * oh).sum().item() / n), rec

        # 전체 Torch 그래프 (head 포함) 의 gradient vs NumPy head + VJP
        leaves = [r0.clone().requires_grad_(True) for r0 in base]
        Wl, bl = Wt0.clone().requires_grad_(True), bt0.clone().requires_grad_(True)
        with torch.enable_grad():
            rec = sh.forward(x64, leaves=leaves)
            z = rec["h"] @ Wl.T + bl
            ce = -(torch.log_softmax(z, dim=1) * oh).sum() / n
            gfull = torch.autograd.grad(ce, leaves + [Wl, bl], allow_unused=False)
        g_full = flat64(gfull[:-2])
        gW, gb = gfull[-2].detach().cpu().numpy(), gfull[-1].detach().cpu().numpy()
        vjp = sh.bp_grads(x64, W, b, y)
        g_vjp = flat64(vjp["grads"])
        den = max(float(np.abs(g_full).max()), 1e-300)
        res: dict[str, Any] = {"batch": bi, "n": n,
                               "vjp_vs_full_graph": {"max_abs_diff": float(np.abs(g_full - g_vjp).max()), "max_rel_to_maxabs": float(np.abs(g_full - g_vjp).max() / den),
                                                     "cos": vec_compare(g_vjp, g_full)["cos"]},
                               "float32_vs_float64": {"cos": vec_compare(g32, g_full)["cos"], "max_abs_diff": float(np.abs(g32 - g_full).max()),
                                                      "rel_l2": float(np.linalg.norm(g32 - g_full) / max(np.linalg.norm(g_full), 1e-300))}}
        _ce0, rec0 = ce_at(None, record="full")
        pat0 = self.patterns(rec0, sh)
        D = self.D
        rng = np.random.default_rng([int(self.run.seed), int(D["rng_tag"]), 77, STATES_ALL.index(self.state), bi])
        dirs_gain = {"BP_FULL(-grad)": -g_full, "TRACE": vec["TRACE_dir"], "TRACE_MIN_AWARE": vec["TRACE_MIN_AWARE_dir"],
                     "random": rng.standard_normal(len(g_full))}
        head_t = -np.concatenate([gW.ravel(), gb.ravel()])                     # TRACE·BP 의 head 갱신 방향은 둘 다 −∂CE/∂(W,b)
        g_joint = np.concatenate([g_full, gW.ravel(), gb.ravel()])
        dirs_joint = {"BP_FULL(-grad)": -g_joint, "TRACE": np.concatenate([vec["TRACE_dir"], head_t]),
                      "random": rng.standard_normal(len(g_joint))}
        fd_rows = []
        for scope, dirs, gvec in (("gain", dirs_gain, g_full), ("joint", dirs_joint, g_joint)):
            for dname, v in dirs.items():
                nv = float(np.linalg.norm(v))
                if nv == 0.0:
                    fd_rows.append({"scope": scope, "direction": dname, "eps": None, "verdict": "skipped_zero_vector"})
                    continue
                u = v / nv
                an = float(gvec @ u)
                for eps in D["fd_eps"]:
                    du = unflat(u[:len(g_full)] * eps)
                    dW = db = None
                    if scope == "joint":
                        hw = u[len(g_full):] * eps
                        dW = torch.from_numpy(hw[:N_CLASSES * W.shape[1]].reshape(W.shape)).to(sh.tt.dev)
                        db = torch.from_numpy(hw[N_CLASSES * W.shape[1]:]).to(sh.tt.dev)
                    fp, recp = ce_at(du, dW, db, record="full")
                    fm, recm = ce_at([-t for t in du], None if dW is None else -dW, None if db is None else -db, record="full")
                    fd = (fp - fm) / (2.0 * eps)
                    pp, pm = self.patterns(recp, sh), self.patterns(recm, sh)
                    cross = int(sum(int((pp[k] != pat0[k]).sum() + (pm[k] != pat0[k]).sum()) for k in pat0))
                    err = abs(fd - an)
                    ok = err <= float(D["fd_atol"]) + float(D["fd_rtol"]) * abs(an)
                    fd_rows.append({"scope": scope, "direction": dname, "eps": eps, "fd": fd, "analytic": an, "abs_err": err,
                                    "rel_err": err / abs(an) if an != 0 else None, "boundary_crossings": cross,
                                    "verdict": ("pass" if ok else "fail") if cross == 0 else ("pass_with_crossing" if ok else "boundary_crossing")})
        res["fd"] = fd_rows
        # 실제 크기의 후보를 적용했을 때 이 배치의 국소 CE 변화 (head 고정, gain-only; 투영 없이 방향 효과만)
        ce0, _ = ce_at(None)
        eta = float(self.run.S["eta_edge"])
        bp_eta = -eta * g32
        nb = float(np.linalg.norm(bp_eta))
        dce: dict[str, Any] = {"ce0": ce0}
        cand = {"BP_FULL(η·−grad32)": bp_eta, "TRACE(실제 후보)": vec["TRACE_eta"], "BP_STOP_L6(η·−grad)": -eta * vec["BP_STOP_L6_grad"]}
        for nm, key in (("TRACE(‖BP‖ 맞춤)", "TRACE_dir"), ("TRACE_MIN_AWARE(‖BP‖ 맞춤)", "TRACE_MIN_AWARE_dir")):
            v = vec[key]
            nv = float(np.linalg.norm(v))
            cand[nm] = (v / nv * nb) if (nv > 0 and nb > 0) else None
        for nm, v in cand.items():
            if v is None:
                dce[nm] = {"skipped": "영벡터"}
                continue
            c1, _ = ce_at(unflat(v))
            dce[nm] = {"delta_ce": c1 - ce0, "norm": float(np.linalg.norm(v)), "linear_pred": float(g_full @ v)}
        res["delta_ce"] = dce
        return res

    # ---------------------------------------------------------------- 실행
    def execute(self) -> dict[str, Any]:
        run, eng, tt = self.run, self.eng, self.tt
        t0 = time.perf_counter()
        run.load_state_into_engine(self.state)
        prm = run.state_params(self.state)
        W, b = prm["W"], prm["b"]
        sha_before = params_sha(W, b, eng.gains_sha())
        rng_np = np.random.get_state()[1][:4].tolist()
        torch_rng = sha256_bytes(torch.get_rng_state().numpy().tobytes())
        sel = self.samples()
        mm, ytr = run.cache("train")
        bs = int(self.D["batch_size"])
        eta_e = float(run.S["eta_edge"])
        say(f"\n[diagnostics] {self.state}: train 고정 표본 {len(sel)} 개 (클래스별 {self.D['per_class']}), batch {bs} → 같은 forward 에서 TRACE·BP 비교 (적용하지 않음)")
        rows: list[dict[str, Any]] = []
        sums: dict[str, np.ndarray] = {}
        rounding_acc: dict[str, dict[str, dict[str, Any]]] = {}
        act_acc: dict[str, dict[str, Any]] = {}
        shadow: list[dict[str, Any]] = []
        graph: dict[str, Any] = {"padding_or_inactive_grad_max": 0.0, "valid_grad_zero": 0, "valid_total": 0, "h_bp_vs_forward_max_abs": 0.0}
        dots: dict[str, list[str]] = {}
        n_fd = int(self.D.get("fd_batches", 2))
        for bi, (a0, a1) in enumerate(chunk_rows(len(sel), bs)):
            if run.stop.requested:
                raise Interrupted(f"diagnostics {self.state} 배치 {bi} 에서 멈춤 (진단은 처음부터 다시 한다)")
            idx = sel[a0:a1]
            y = ytr[idx]
            n = len(y)
            x0 = tt.f(mm[idx])
            with torch.no_grad():
                rec = eng.forward(x0, record="full")
            h = tt.to_np(rec["h"]).astype(np.float64)
            _z, p, _logp = head_forward(W, b, h)
            dz = onehot(y, n) - p
            d_h = head_route_trace(W, h, dz)
            eng.route_back(rec, tt.f(d_h))
            dg = [q.dg for q in eng.parts]
            t_eta = [q.w0 * g_ * (eta_e / float(n)) for q, g_ in zip(eng.parts, dg)]       # 실제 후보 (학습과 같은 곱 순서)
            t_dir = self.flat([q.w0.double() * g_.double() / float(n) for q, g_ in zip(eng.parts, dg)])
            eng.clear_grads()
            eng.route_back(rec, tt.f(d_h), min_aware=True)
            t_ma_dir = self.flat([q.w0.double() * q.dg.double() / float(n) for q in eng.parts])
            eng.clear_grads()
            bpf = eng.bp_grads(x0, W, b, y)
            bps = eng.bp_grads(x0, W, b, y, stop_l6=True)
            graph["h_bp_vs_forward_max_abs"] = max(graph["h_bp_vs_forward_max_abs"], float(np.abs(bpf["h"] - h).max()))
            g_full, g_stop = self.flat(bpf["grads"]), self.flat(bps["grads"])
            graph["padding_or_inactive_grad_max"] = max(graph["padding_or_inactive_grad_max"], self.off_valid_max(bpf["grads"]),
                                                        self.off_valid_max(bps["grads"]))
            graph["valid_grad_zero"] += int((g_full == 0).sum())
            graph["valid_total"] += len(g_full)
            gW, gb = -(dz.T @ h) / n, -dz.mean(axis=0)
            head = np.concatenate([gW.ravel(), gb.ravel()])
            vec = {"TRACE_dir": t_dir, "TRACE_MIN_AWARE_dir": t_ma_dir, "TRACE_eta": self.flat(t_eta), "BP_STOP_L6_grad": g_stop}
            dirs = {"TRACE": t_dir, "TRACE_MIN_AWARE": t_ma_dir, "BP_FULL": -g_full, "BP_STOP_L6": -g_stop}
            for k_, v_ in dirs.items():
                sums[k_] = v_ if k_ not in sums else sums[k_] + v_
            sums["_g_full"] = g_full if "_g_full" not in sums else sums["_g_full"] + g_full
            sums["_head"] = -head if "_head" not in sums else sums["_head"] - head
            rows += self._compare_rows(bi, dirs, -head, g_full, eta_e, float(run.S["eta_head"]))
            for nm, dl in (("TRACE", t_eta), ("BP_FULL", [g_ * (-eta_e) for g_ in bpf["grads"]])):
                for lk, o in self.rounding(dl).items():
                    acc_ = rounding_acc.setdefault(nm, {}).setdefault(lk, {"_dw": []})
                    for kk, vv in o.items():
                        if kk == "_dw":
                            acc_["_dw"] += vv
                        else:
                            acc_[kk] = acc_.get(kk, 0) + vv
            self.activity(rec, act_acc)
            for pr in ("TRACE", "TRACE_MIN_AWARE"):
                ds, _dv = dot_sign(g_full, dirs[pr])
                dots.setdefault(pr, []).append(ds)
            if bi < n_fd:
                shadow.append(self.shadow_checks(bi, idx, W, b, vec, g_full))
        rows += self._compare_rows("sum", {k: v for k, v in sums.items() if not k.startswith("_")}, sums["_head"], sums["_g_full"], eta_e,
                                   float(run.S["eta_head"]))
        sha_after = params_sha(W, b, eng.gains_sha())
        if sha_after != sha_before or np.random.get_state()[1][:4].tolist() != rng_np or sha256_bytes(torch.get_rng_state().numpy().tobytes()) != torch_rng:
            raise IntegrityError("진단 전후 파라미터 해시 또는 전역 RNG 상태가 달라졌다 (진단은 상태를 바꾸면 안 된다).")
        rnd_rows = []
        for nm, per in rounding_acc.items():
            for lk, o in per.items():
                dw = np.concatenate(o.pop("_dw")) if o.get("_dw") else np.zeros(0)
                o.pop("_dw", None)
                q = np.quantile(dw, [0.5, 0.9, 0.99]) if dw.size else [None] * 3
                rnd_rows.append({"state": self.state, "candidate": nm, "layer": lk, **o, "dw_rms": float(np.sqrt((dw ** 2).mean())) if dw.size else None,
                                 "dw_abs_q50": None if q[0] is None else float(q[0]), "dw_abs_q90": None if q[1] is None else float(q[1]),
                                 "dw_abs_q99": None if q[2] is None else float(q[2]), "dw_abs_max": float(dw.max()) if dw.size else None})
        act_rows = []
        for lk, o in act_acc.items():
            vn = max(1, o["valid_neurons"] * o["samples"])
            act_rows.append({"state": self.state, "layer": lk, "samples": o["samples"], "valid_neurons": o["valid_neurons"],
                             "inactive_structural": o["inactive_structural"], "zero_activity_frac": o["zero_activity"] / vn,
                             "min_neurons": o["min_neurons"], "min_tie_frac": (o["min_tie"] / max(1, o["min_neurons"] * o["samples"])) if o["min_neurons"] else None,
                             "min_tie_zero_frac": (o["min_tie_zero"] / max(1, o["min_neurons"] * o["samples"])) if o["min_neurons"] else None,
                             "min_trace_share_to_smaller_mean": (o["min_share_small_sum"] / o["min_share_n"]) if o["min_share_n"] else None,
                             "min_share_n": o["min_share_n"],
                             "l6_at_min_frac": (o["l6_at_min"] / o["l6_n"]) if o["l6_n"] else None,
                             "l6_at_max_frac": (o["l6_at_max"] / o["l6_n"]) if o["l6_n"] else None,
                             "l6_exact_boundary": o["l6_exact_boundary"] if o["l6_n"] else None,
                             "l6_rect_negative": o["l6_rect_negative"] if o["l6_n"] else None, "l6_rect_zero": o["l6_rect_zero"] if o["l6_n"] else None})
        gsum = eng.gain_summary()
        gain_rows = [{"state": self.state, "layer": k, **{kk: v[kk] for kk in ("n", "mean", "min", "max", "n_changed", "n_ge_1pct", "n_ge_10pct",
                                                                             "n_at_min", "n_at_max")}} for k, v in gsum.items()]
        graph["valid_grad_zero_frac"] = graph["valid_grad_zero"] / max(1, graph["valid_total"])
        graph["ok"] = graph["padding_or_inactive_grad_max"] == 0.0 and graph["h_bp_vs_forward_max_abs"] == 0.0
        out = {"state": self.state, "n_samples": int(len(sel)), "sample_ids_sha256": sha256_array(sel), "batch_size": bs,
               "scope_ko": f"train 고정 표본 {len(sel)} 개 (클래스별 {self.D['per_class']}, 별도 RNG) 의 통계다. 장기 정확도를 예측하지 않는다.",
               "params_sha_before": sha_before, "params_sha_after": sha_after, "params_unchanged": sha_before == sha_after,
               "graph": graph, "dot_sign_counts": {k: {s: v.count(s) for s in sorted(set(v))} for k, v in dots.items()},
               "comparison_rows": rows, "rounding_rows": rnd_rows, "activity_rows": act_rows, "gain_rows": gain_rows, "shadow": shadow,
               "conventions": CONVENTIONS, "time_s": round(time.perf_counter() - t0, 3)}
        run.add_time("diagnostics_s", time.perf_counter() - t0)
        return out

    def _compare_rows(self, bi: Any, dirs: dict[str, np.ndarray], head_dir: np.ndarray, g_full: np.ndarray, eta_e: float, eta_h: float
                      ) -> list[dict[str, Any]]:
        rows = []
        for a, b_ in PAIRS:
            va, vb = dirs[a], dirs[b_]
            for li, lk in enumerate(list(LEARN_KEYS) + ["total"]):
                m = slice(None) if lk == "total" else (self.layer_of == li)
                r = {"state": self.state, "batch": bi, "scope": "gain", "layer": lk, "pair": f"{a}_vs_{b_}", **vec_compare(va[m], vb[m])}
                if b_ == "BP_FULL" and a.startswith("TRACE"):
                    ds, dv = dot_sign(g_full[m], va[m] * eta_e)
                    r["dot_grad_delta_sign"], r["dot_grad_delta"] = ds, dv
                rows.append(r)
            ja = np.concatenate([va * eta_e, head_dir * eta_h])                 # joint: 실제 갱신 벡터 (gain η_edge, head η_head)
            jb = np.concatenate([vb * eta_e, head_dir * eta_h])
            r = {"state": self.state, "batch": bi, "scope": "joint", "layer": "total(W·b 포함)", "pair": f"{a}_vs_{b_}", **vec_compare(ja, jb)}
            if b_ == "BP_FULL" and a.startswith("TRACE"):
                ds, dv = dot_sign(np.concatenate([g_full, -head_dir]), ja)
                r["dot_grad_delta_sign"], r["dot_grad_delta"] = ds, dv
            rows.append(r)
        return rows


# ======================================================================
# 실행 하나 (run_<시각>/) 와 작업 스케줄러. 작업 ID 순서대로 진행하고, 완료한 작업은 state.json 에 기록한다 (이어하기의 기준).
#   phase: validate_input / train_trace / train_bp / probes / raw_refit / diagnostics / select_lock / test / done
#   상태별 특징 캐시를 한 번 만들어 그 상태의 판독기 6 개와 raw refit 에 함께 쓰도록 순서를 묶었다.
# ======================================================================
def plan_jobs(scope: str) -> list[dict[str, Any]]:
    J: list[dict[str, Any]] = []

    def add(jid: str, phase: str) -> None:
        J.append({"id": jid, "phase": phase, "status": "pending"})

    add("validate_input", "validate_input")
    if scope == "v12_best_probes":
        add(f"import:{V12_STATE}", "validate_input")
        states: tuple[str, ...] = (V12_STATE,)
    else:
        add("init_snapshot", "validate_input")
        add("train:TRACE", "train_trace")
        add("train:BP_FULL", "train_bp")
        states = STATES
    if scope in ("full", "v12_best_probes"):
        add("probe:SHARED:V2/L3", "probes")
        for s in states:
            add(f"features:{s}", "probes")
            for k in LEARN_KEYS:
                add(f"probe:{s}:{k}", "probes")
            if s != "INITIAL":
                add(f"refit:{s}", "raw_refit")
            add(f"release:{s}", "probes")
    else:
        for s in states:
            add(f"fixed_eval:{s}", "diagnostics")
    if scope in ("full", "train_and_diagnostics"):
        for s in states:
            add(f"diagnostics:{s}", "diagnostics")
    add("select_lock", "select_lock")
    add("test", "test")
    add("summary", "done")
    return J


def scope_states(scope: str) -> tuple[str, ...]:
    return (V12_STATE,) if scope == "v12_best_probes" else STATES


def layer_dir(layer: str) -> str:
    return layer.replace("/", "_")


class Run:
    def __init__(self, run_dir: Path, cfg: dict[str, Any], state: dict[str, Any]):
        self.dir, self.cfg, self.state = Path(run_dir), cfg, state
        self.S = cfg["settings"]
        self.scope = cfg["scope"]
        self.seed = int(cfg["seed"])
        self.input_id = cfg["input"]["input_id"]
        self.device: Any = None
        self.backend: dict[str, Any] = {}
        self.inp: V12Input | None = None
        self.ctx: dict[str, Any] = {}
        self._engine: Engine | None = None
        self._shadow: Engine | None = None
        self._caches: dict[str, np.ndarray] = {}
        self._params_cache: dict[str, dict[str, Any]] = {}
        self.layout: list[tuple[str, int, int]] = []
        self.stop: StopFlag = _NULL_STOP

    # ---------------------------------------------------------------- 연결
    def attach(self, inp: V12Input, ctx: dict[str, Any], device: Any) -> None:
        self.inp, self.ctx, self.device = inp, ctx, device
        self.backend = backend_id(device)
        self.layout = part_layout(ctx["specs"])

    def engine(self) -> Engine:
        if self._engine is None:
            c, S = self.ctx, self.S
            tt = TT(self.device, torch.float32, int(S["compute"]["max_gather_elements"]))
            self._engine = Engine(tt, c["specs"], c["model"], c["q"], c["thr"], gain_min=float(S["gain_min"]), gain_max=float(S["gain_max"]),
                                  memory_mb=int(S["compute"]["memory_mb"]))
            if self._engine.n_learnable != sum(c["edges"].values()):
                raise IntegrityError("엔진의 유효 학습 연결 수가 구조 계산과 다르다.")
        return self._engine

    def shadow_engine(self) -> Engine:
        """진단용 float64 그림자 (같은 주소·w0·q·threshold; TRACE 전치 구조 없음)."""
        if self._shadow is None:
            c, S = self.ctx, self.S
            tt = TT(self.device, torch.float64, int(S["compute"]["max_gather_elements"]))
            self._shadow = Engine(tt, c["specs"], c["model"], c["q"], c["thr"], gain_min=float(S["gain_min"]), gain_max=float(S["gain_max"]),
                                  memory_mb=int(S["compute"]["memory_mb"]), with_trace=False)
        return self._shadow

    def labels(self, split: str) -> np.ndarray:
        return self.ctx["split"][split]["y"]

    def cache(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """V2/L3 입력 캐시 (읽기 전용 memmap). test 는 선택 잠금 뒤에만 연다."""
        if split not in self._caches:
            c = self.ctx["caches"].get(split)
            if c is None:
                raise IntegrityError(f"{split} 캐시가 입력에 없다.")
            if split == "test":
                lock = self.dir / "selection_lock.json"
                if not lock.is_file() or sha256_file(lock) != self.state.get("lock_sha256"):
                    raise IntegrityError("test 캐시는 selection_lock.json 을 저장한 뒤에만 읽는다.")
            p = self.inp.local(c["file"])
            if not c.get("verified_sha"):
                t0 = time.perf_counter()
                sha = sha256_file(p)
                if sha != c["sha256"]:
                    raise IntegrityError(f"{c['file']} 본문 SHA256 이 v12 기록과 다르다 (바뀐 캐시).")
                c["verified_sha"] = True
                self.state.setdefault("input_checks", {})[f"{split}_body_sha_verified_utc"] = utc_now()
                self.add_time("input_s", time.perf_counter() - t0)
            mm = np.load(p, mmap_mode="r")
            if mm.shape != (c["rows"], c["dim"]) or mm.dtype != np.float32:
                raise IntegrityError(f"{c['file']} 모양·형 {mm.shape}/{mm.dtype} 이 기록과 다르다.")
            self._caches[split] = mm
        return self._caches[split], self.labels(split)

    # ---------------------------------------------------------------- 시간·저장
    def add_time(self, bucket: str, sec: float) -> None:
        t = self.state.setdefault("time", {})
        t[bucket] = float(t.get(bucket, 0.0)) + float(sec)

    def save_state(self) -> None:
        with self.stop.critical():
            self.state["seq"] = int(self.state.get("seq", 0)) + 1
            self.state["utc"] = utc_now()
            write_json(self.dir / "state.json", self.state)

    def results(self, key: str) -> dict[str, Any]:
        return self.state.setdefault("results", {}).setdefault(key, {})

    # ---------------------------------------------------------------- 상태 파라미터 (INITIAL · TRACE_BEST · BP_FULL_BEST · V12_IMPORTED_BEST)
    def _params_sha_np(self, W: np.ndarray, b: np.ndarray, gains: dict[str, np.ndarray] | None) -> str:
        if gains is None:
            gains = {f"g.{n}": np.zeros((r, K), np.float32) for n, r, K in self.layout}
        return v12_params_sha(W, b, gains, self.layout)

    def initial_params(self) -> dict[str, Any]:
        return self.state_params("INITIAL")

    def state_params(self, state: str) -> dict[str, Any]:
        if state in self._params_cache:
            return self._params_cache[state]
        if state == "INITIAL":
            p = self.dir / "initial_snapshot.npz"
        elif state in ("TRACE_BEST", "BP_FULL_BEST"):
            p = self.dir / COND_DIR[state[:-5]] / "best.npz"
        elif state == V12_STATE:
            p = self.dir / "v12_imported" / "best.npz"
        else:
            raise V13Error(f"알 수 없는 상태 {state}")
        if not p.is_file():
            raise IntegrityError(f"{state} 상태 파일이 없다: {p}")
        arrays, meta = load_npz(p)
        W, b = np.asarray(arrays["W"], np.float64), np.asarray(arrays["b"], np.float64)
        gains = {k: v for k, v in arrays.items() if k.startswith("g.")}
        sha = self._params_sha_np(W, b, gains if gains else None)
        if state == "INITIAL":
            expect = (meta or {}).get("params_sha256")
        elif state == V12_STATE:
            expect = ((meta or {}).get("v12_best") or {}).get("params_sha256")
        else:
            expect = ((self.results("main").get(state[:-5]) or {}).get("best_params_sha256"))
        if sha != expect:
            raise IntegrityError(f"{state} 상태 파일의 W·b·gain 해시가 기록 ({str(expect)[:16]}…) 과 다르다.")
        out = {"W": W, "b": b, "gains": gains if gains else None, "params_sha256": sha, "file": str(p.relative_to(self.dir)), "meta": meta}
        self._params_cache[state] = out
        return out

    def load_state_into_engine(self, state: str) -> None:
        eng = self.engine()
        prm = self.state_params(state)
        if prm["gains"] is None:
            eng.reset_gains()
        else:
            eng.load_gain_arrays(prm["gains"], "g")
        if params_sha(prm["W"], prm["b"], eng.gains_sha()) != prm["params_sha256"]:
            raise IntegrityError(f"{state} 를 엔진에 올린 뒤 파라미터 해시가 다르다.")

    # ---------------------------------------------------------------- 진입
    def execute(self) -> None:
        with StopFlag() as stop:
            self.stop = stop
            self.state.setdefault("events", []).append({"utc": utc_now(), "start_job": next((j["id"] for j in self.state["jobs"]
                                                                                              if j["status"] == "pending"), None),
                                                        "backend": backend_describe(self.device), "warnings": list(_DET_WARNINGS)})
            self.save_state()
            jobs = self.state["jobs"]
            for i, job in enumerate(jobs):
                if job["status"] != "pending":
                    continue
                self.state["phase"], self.state["current_job"] = job["phase"], job["id"]
                self.save_state()
                t0 = time.perf_counter()
                brief = self.run_job(job["id"])
                with stop.critical():
                    job["status"] = "done"
                    job["utc_done"] = utc_now()
                    if _DET_WARNINGS:
                        self.state["determinism_warnings"] = list(_DET_WARNINGS)
                    job["seconds_last_session"] = round(time.perf_counter() - t0, 2)
                    if brief:
                        job["brief"] = brief
                    self.save_state()
                if stop.requested and any(j["status"] == "pending" for j in jobs[i + 1:]):
                    raise Interrupted(f"작업 {job['id']} 를 마친 뒤 멈춤")
            self.state["phase"] = "done"
            self.state["current_job"] = None
            self.state["determinism_warnings"] = list(_DET_WARNINGS)
            self.save_state()

    def run_job(self, jid: str) -> dict[str, Any] | None:
        kind, _, rest = jid.partition(":")
        if kind == "validate_input":
            return {"note": "실행을 만들 때 검증했다 (input_manifest.json)"}
        if kind == "init_snapshot":
            return self.job_init_snapshot()
        if kind == "import":
            return self.job_import()
        if kind == "train":
            res = MainTrainer(self, rest).execute()
            self.results("main")[rest] = res
            self._params_cache.pop(f"{rest}_BEST", None)
            return {k: res[k] for k in ("best_epoch", "best_dev_accuracy", "completed_epochs")}
        if kind == "features":
            res = FeatureStore(self, rest).build()
            self.results("fixed_eval")[rest] = {"train": res["fixed_train"], "dev": res["fixed_dev"], "via": "features"}
            return {"bytes": res["bytes"]}
        if kind == "probe":
            st, _, layer = rest.partition(":")
            return self.job_probe(jid, st, layer)
        if kind == "refit":
            return self.job_refit(jid, rest)
        if kind == "release":
            freed = FeatureStore(self, rest).release()
            return {"freed_bytes": freed}
        if kind == "fixed_eval":
            return self.job_fixed_eval(rest)
        if kind == "diagnostics":
            return self.job_diagnostics(rest)
        if kind == "select_lock":
            return self.job_select_lock()
        if kind == "test":
            return self.job_test()
        if kind == "summary":
            return write_summary(self)
        raise V13Error(f"알 수 없는 작업 {jid}")

    # ---------------------------------------------------------------- 작업들
    def job_init_snapshot(self) -> dict[str, Any]:
        c = self.ctx
        W0, b0 = np.asarray(c["W0"], np.float64), np.asarray(c["b0"], np.float64)
        gains = {f"g.{n}": np.zeros((r, K), np.float32) for n, r, K in self.layout}
        sha = self._params_sha_np(W0, b0, None)
        qs = {f"q.{k.replace('/', '.')}": c["q"][c["ranges"][k][0]:c["ranges"][k][1]] for k in LEARN_KEYS}
        ts = {f"threshold.{k.replace('/', '.')}": c["thr"][c["ranges"][k][0]:c["ranges"][k][1]] for k in LEARN_KEYS}
        meta = {"format": "cortex_v13/initial_snapshot/1", "params_sha256": sha, "source": "frozen_stem.npz (v12): q·threshold·W0·b0, gain = 1",
                "frozen_stem_sha256": c["files"]["frozen_stem.npz"]["sha256"], "struct_hash": c["hashes"], "layer_ranges": c["ranges"],
                "seed": self.seed, "note_ko": "TRACE·BP_FULL 은 이 snapshot 의 독립 복사본에서 출발한다. v12 best 를 초기값으로 쓰지 않는다."}
        save_npz(self.dir / "initial_snapshot.npz", {"W": W0, "b": b0, **gains, **qs, **ts}, meta)
        self._params_cache.pop("INITIAL", None)
        prm = self.state_params("INITIAL")
        self.load_state_into_engine("INITIAL")
        say(f"  공통 초기 snapshot 저장 (W0 [10,{W0.shape[1]}] · b0 [10] · gain = 1, 해시 {prm['params_sha256'][:16]}…)")
        return {"params_sha256": sha}

    def job_import(self) -> dict[str, Any]:
        p = self.inp.local("best_params.npz")
        arrays, meta = load_npz(p)
        bs = json.loads(self.inp.read_bytes("best_state.json").decode("utf-8"))
        best = bs.get("best") or {}
        W, b = np.asarray(arrays["W"], np.float64), np.asarray(arrays["b"], np.float64)
        gains = {k: np.asarray(v, np.float32) for k, v in arrays.items() if k.startswith("g.")}
        sha = self._params_sha_np(W, b, gains)
        if sha != best.get("params_sha256"):
            raise IntegrityError("v12 best_params.npz 의 해시가 best_state.json 기록과 다르다.")
        ensure_dir(self.dir / "v12_imported")
        save_npz(self.dir / "v12_imported" / "best.npz", {"W": W, "b": b, **gains},
                 {"format": "cortex_v13/v12_imported_best/1", "v12_best": best, "source_file_sha256": sha256_file(p),
                  "note_ko": "v12 가 학습한 best (V12_IMPORTED_BEST). 새 TRACE/BP 비교 결과가 아니다."})
        self._params_cache.pop(V12_STATE, None)
        self.load_state_into_engine(V12_STATE)
        say(f"  v12 best 불러옴: epoch {best.get('epoch')} · dev {fmt_pct(best.get('dev_accuracy'))} (해시 확인)")
        return {"epoch": best.get("epoch"), "params_sha256": sha}

    def _probe_dirs(self, state: str, layer: str) -> Path:
        return self.dir / "probes" / state / layer_dir(layer)

    def job_probe(self, pid: str, state: str, layer: str) -> dict[str, Any]:
        P = self.S["probe"]
        if state == "SHARED":
            mm_tr, y_tr = self.cache("train")
            mm_dv, y_dv = self.cache("dev")
            fkey = {"source": "v12 V2/L3 캐시 (모든 상태에서 같은 입력)", "input_id": self.input_id, "layer": "V2/L3",
                    "cache_sha256": {s: self.ctx["caches"][s]["sha256"] for s in ("train", "dev")}, "pre_normalization": True, "dtype": "float32"}
            shared = list(scope_states(self.scope))
        else:
            fs = FeatureStore(self, state)
            mm_tr, mm_dv = fs.open(layer, "train"), fs.open(layer, "dev")
            y_tr, y_dv = self.labels("train"), self.labels("dev")
            fkey = {"features_key_sha256": fs.key_sha(), "layer": layer, "pre_normalization": True}
            shared = []
        say(f"\n[probes] {pid}: d = {mm_tr.shape[1]:,} → 10 (파라미터 {mm_tr.shape[1] * 10 + 10:,}), train {mm_tr.shape[0]:,} · dev {mm_dv.shape[0]:,}, "
            f"L-BFGS 최대 iteration {P['max_iter']} · closure 예산 {P['max_evals']}")
        pr = LBFGSProbe(self, pid=pid, kind="probe", state=state, layer=layer, out_dir=self._probe_dirs(state, layer), X_tr=mm_tr, y_tr=y_tr,
                        X_dv=mm_dv, y_dv=y_dv, standardize=True, penalty=float(P["penalty"]), W0=None, b0=None, feature_key=fkey,
                        shared_states=shared)
        res = pr.execute()
        self.results("probes")[pid] = res
        say(f"  → {res['termination']} · 수락 {res['accepted_iterations']} · closure {res['closures_computed']} | best it {res['best']['iteration']} "
            f"dev {fmt_pct(res['best']['dev_accuracy'])} · CE {res['best']['dev_ce']:.4f} | train {fmt_pct(res['best']['train_accuracy'])}")
        return {"termination": res["termination"], "best_dev_accuracy": res["best"]["dev_accuracy"]}

    def job_refit(self, pid: str, state: str) -> dict[str, Any]:
        fs = FeatureStore(self, state)
        mm_tr, mm_dv = fs.open("IT/L3", "train"), fs.open("IT/L3", "dev")
        prm = self.state_params(state)
        fkey = {"features_key_sha256": fs.key_sha(), "layer": "IT/L3", "raw": True, "init_head_params_sha256": prm["params_sha256"]}
        say(f"\n[raw_refit] {state}: raw IT/L3 ({mm_tr.shape[1]}) + 원래 head 에서 시작 · 정규화 없음 · penalty 0 · 같은 L-BFGS 예산")
        pr = LBFGSProbe(self, pid=pid, kind="refit", state=state, layer="IT/L3", out_dir=self.dir / "raw_refit" / state, X_tr=mm_tr,
                        y_tr=self.labels("train"), X_dv=mm_dv, y_dv=self.labels("dev"), standardize=False, penalty=0.0,
                        W0=prm["W"], b0=prm["b"], feature_key=fkey)
        res = pr.execute()
        res["original_head"] = (pr.history[0] if pr.history else {})
        write_json(self.dir / "raw_refit" / state / "result.json", res)
        self.results("refit")[state] = res
        say(f"  → {res['termination']} · best it {res['best']['iteration']} dev {fmt_pct(res['best']['dev_accuracy'])} "
            f"(원래 head dev {fmt_pct(res['original_head'].get('dev_accuracy'))})")
        return {"termination": res["termination"], "best_dev_accuracy": res["best"]["dev_accuracy"]}

    def job_fixed_eval(self, state: str) -> dict[str, Any]:
        """선택 모델의 고정 train·dev 재평가 (running train 과 별도). 특징 캐시를 만들지 않는 범위 (메뉴 3) 에서 쓴다."""
        self.load_state_into_engine(state)
        prm = self.state_params(state)
        eng = self.engine()
        out = {}
        t0 = time.perf_counter()
        for s in ("train", "dev"):
            mm, y = self.cache(s)
            out[s] = eval_split(eng, mm, y, prm["W"], prm["b"], int(self.S["compute"]["feature_batch"]), self.stop, f"{state} 고정 {s}")
        self.add_time("fixed_eval_s", time.perf_counter() - t0)
        out["via"] = "fixed_eval"
        self.results("fixed_eval")[state] = out
        say(f"  {state} 고정 재평가: train {fmt_pct(out['train']['accuracy'])} · dev {fmt_pct(out['dev']['accuracy'])}")
        return {"train_accuracy": out["train"]["accuracy"], "dev_accuracy": out["dev"]["accuracy"]}

    def job_diagnostics(self, state: str) -> dict[str, Any]:
        out = Diagnostics(self, state).execute()
        ensure_dir(self.dir / "diagnostics")
        write_json(self.dir / "diagnostics" / f"{state}.json", out)
        tot = [r for r in out["comparison_rows"] if r["batch"] == "sum" and r["scope"] == "gain" and r["layer"] == "total"]
        brief = {f"{r['pair']}.cos": r["cos"] for r in tot}
        self.results("diagnostics")[state] = {"file": f"diagnostics/{state}.json", "params_unchanged": out["params_unchanged"],
                                              "graph_ok": out["graph"]["ok"], "sum_total_gain": brief, "dot_sign_counts": out["dot_sign_counts"]}
        say("  " + " · ".join(f"{k[:-4]} cos {fmt_f(v, 3)}" for k, v in brief.items()))
        return brief

    def job_select_lock(self) -> dict[str, Any]:
        lock: dict[str, Any] = {"format": "cortex_v13/selection_lock/1", "run": self.dir.name, "utc": utc_now(), "scope": self.scope,
                                "input_id": self.input_id, "program_sha256": self.cfg["program_sha256"], "states": {}, "main": {}, "probes": {},
                                "refit": {},
                                "rule_ko": "모든 선택은 dev 로만 했다 (주 학습: dev 정확도 → CE → 이른 epoch, epoch 0 포함; 판독기·raw refit: dev 정확도 → CE → "
                                           "이른 iteration, 초기 상태 포함). test 는 이 파일을 저장한 뒤에만 읽고, test 로 다시 고르지 않는다.",
                                "note_ko": "기존 v12 test 결과가 이미 조회된 자료이므로 이 평가는 blind test 가 아니다. 중단 뒤 test 를 다시 계산하면 같은 snapshot 의 재평가다."}
        for cond, r in self.results("main").items():
            lock["main"][cond] = {k: r[k] for k in ("best_epoch", "best_dev_accuracy", "best_dev_ce", "best_params_sha256")}
        for s in scope_states(self.scope):
            prm = self.state_params(s)
            lock["states"][s] = {"params_sha256": prm["params_sha256"], "file": prm["file"]}
        for pid, r in self.results("probes").items():
            lock["probes"][pid] = {k: r.get(k) for k in ("state", "layer", "best_head_sha256", "normalization_sha256", "feature_key_sha256", "termination")}
            lock["probes"][pid].update(best_iteration=r["best"]["iteration"], dev_accuracy=r["best"]["dev_accuracy"], dev_ce=r["best"]["dev_ce"])
        for s, r in self.results("refit").items():
            lock["refit"][s] = {"best_head_sha256": r["best_head_sha256"], "best_iteration": r["best"]["iteration"], "dev_accuracy": r["best"]["dev_accuracy"],
                                "dev_ce": r["best"]["dev_ce"], "feature_key_sha256": r["feature_key_sha256"], "termination": r["termination"]}
        p = self.dir / "selection_lock.json"
        with self.stop.critical():
            write_json(p, lock)
            self.state["lock_sha256"] = sha256_file(p)
        say(f"\n[select_lock] dev 선택 잠금: 주 학습 {len(lock['main'])} · 상태 {len(lock['states'])} · 판독기 {len(lock['probes'])} · raw refit {len(lock['refit'])} "
            "→ selection_lock.json (test 는 이 뒤에만)")
        return {"lock_sha256": self.state["lock_sha256"]}

    def _load_head(self, d: Path, with_norm: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, str, str | None]:
        arrs, _meta = load_npz(d / "best.npz")
        W, b = np.asarray(arrs["W"], np.float64), np.asarray(arrs["b"], np.float64)
        mu = sig = nsha = None
        if with_norm:
            na, _nmeta = load_npz(d / "normalization.npz")
            mu, sigma = np.asarray(na["mu"], np.float64), np.asarray(na["sigma"], np.float64)
            nsha = sha256_text(sha256_array(mu) + sha256_array(sigma))
            sig = np.maximum(sigma, float(self.S["probe"]["sigma_floor"]))
        return W, b, mu, sig, sha256_array(W, b), nsha

    def job_test(self) -> dict[str, Any]:
        lock_p = self.dir / "selection_lock.json"
        if not lock_p.is_file() or sha256_file(lock_p) != self.state.get("lock_sha256"):
            raise IntegrityError("selection_lock.json 이 없거나 바뀌었다 (test 를 읽지 않는다).")
        lock = read_json(lock_p)
        tst = self.state.setdefault("test", {})
        tst["attempts"] = int(tst.get("attempts", 0)) + 1
        tst.setdefault("attempt_utc", []).append(utc_now())
        self.save_state()                                                       # test 를 읽기 전에 시도 횟수를 남긴다 (중단 뒤 재계산 = 재평가로 기록)
        if not self.ctx.get("test_available"):
            tst["skipped_ko"] = "입력 v12 run 에 cache_test.npy 가 없어 test 를 평가하지 않았다."
            say("\n[test] cache_test.npy 가 없다 → test 평가를 건너뛴다 (선택 잠금까지는 완료).")
            return {"skipped": True}
        say("\n[test] 선택 잠금 확인 → test 캐시 확인 (SHA256) → 상태마다 test forward 한 번으로 모든 판독기·head 지표를 모은다"
            + (" (같은 snapshot 재평가)" if tst["attempts"] > 1 else ""))
        t0 = time.perf_counter()
        c = self.ctx["caches"]["test"]
        shape, dts, fortran, _ = self.inp.header(c["file"])
        if shape != (c["rows"], c["dim"]) or dts != "<f4" or fortran:
            raise IntegrityError(f"cache_test.npy 머리 {shape} {dts} 가 기록과 다르다.")
        mm, y = self.cache("test")
        FB = int(self.S["compute"]["feature_batch"])
        eng = self.engine()
        out: dict[str, Any] = {"states": {}, "probes": {}, "refit": {}, "lock_sha256": self.state["lock_sha256"], "n": int(len(y))}
        shared_pid = "probe:SHARED:V2/L3"
        shared = lock["probes"].get(shared_pid)
        for si, s in enumerate(scope_states(self.scope)):
            self.load_state_into_engine(s)
            prm = self.state_params(s)
            if prm["params_sha256"] != lock["states"][s]["params_sha256"]:
                raise IntegrityError(f"{s} 의 파라미터 해시가 선택 잠금과 다르다.")
            heads: dict[str, Any] = {}
            for k in LEARN_KEYS:
                pid = f"probe:{s}:{k}"
                if pid in lock["probes"]:
                    W, b, mu, sig, hsha, nsha = self._load_head(self._probe_dirs(s, k), True)
                    if hsha != lock["probes"][pid]["best_head_sha256"] or nsha != lock["probes"][pid]["normalization_sha256"]:
                        raise IntegrityError(f"{pid} 의 head·정규화 해시가 선택 잠금과 다르다.")
                    heads[pid] = (k, W, b, mu, sig, EvalAcc())
            if shared and si == 0:
                W, b, mu, sig, hsha, nsha = self._load_head(self._probe_dirs("SHARED", "V2/L3"), True)
                if hsha != shared["best_head_sha256"] or nsha != shared["normalization_sha256"]:
                    raise IntegrityError("공유 V2/L3 판독기의 해시가 선택 잠금과 다르다.")
                heads[shared_pid] = ("V2/L3", W, b, mu, sig, EvalAcc())
            refit = None
            if s in lock["refit"]:
                W, b, _mu, _sig, hsha, _ = self._load_head(self.dir / "raw_refit" / s, False)
                if hsha != lock["refit"][s]["best_head_sha256"]:
                    raise IntegrityError(f"{s} raw refit head 해시가 선택 잠금과 다르다.")
                refit = (W, b, EvalAcc())
            main_acc = EvalAcc()
            with torch.no_grad():
                for a0, a1 in chunk_rows(len(y), FB):
                    if self.stop.requested:
                        raise Interrupted("test 평가 도중 멈춤 (이어 갈 때 같은 snapshot 을 다시 평가한다)")
                    x0 = eng.tt.f(mm[a0:a1])
                    rec = eng.forward(x0, record="outs")
                    yb = y[a0:a1]
                    for pid, (k, W, b, mu, sig, acc) in heads.items():
                        X = x0 if k == "V2/L3" else rec["layers"][k]["out"]
                        probe_head_eval(X, yb, W, b, mu, sig, acc, self.device)
                    h = eng.tt.to_np(rec["h"]).astype(np.float64)
                    _z, p, logp = head_forward(prm["W"], prm["b"], h)
                    main_acc.add(p, logp, yb)
                    if refit is not None:
                        _z, p2, logp2 = head_forward(refit[0], refit[1], h)
                        refit[2].add(p2, logp2, yb)
            out["states"][s] = {"main_head": main_acc.result(), "params_sha256": prm["params_sha256"]}
            for pid, (_k, *_r, acc) in heads.items():
                out["probes"][pid] = acc.result()
            if refit is not None:
                out["refit"][s] = refit[2].result()
            say(f"  {s}: 원래 head test {fmt_pct(out['states'][s]['main_head']['accuracy'])}"
                + (f" · raw refit {fmt_pct(out['refit'][s]['accuracy'])}" if refit is not None else "")
                + (" · 판독기 " + " ".join(f"{pid.split(':')[-1]} {fmt_pct(out['probes'][pid]['accuracy'])}" for pid in heads) if heads else ""))
        self.add_time("test_s", time.perf_counter() - t0)
        out["re_evaluation"] = tst["attempts"] > 1
        out["utc"] = utc_now()
        tst["result"] = out
        write_json(self.dir / "test_results.json", out)
        return {"n": out["n"], "states": len(out["states"]), "probes": len(out["probes"])}


# ======================================================================
# 결과 정리: CSV · final_metrics.json · SUMMARY_KO.txt · figures/ (Matplotlib 이 있을 때만)
#   실제로 계산한 값만 쓴다. 기존 v12 기록 (90.00%·74.38% 등) 은 '참고 기록' 으로 따로 표시하고 목표값으로 쓰지 않는다.
# ======================================================================
EPOCH_COLS = ("cond", "epoch", "train_ce_running", "train_accuracy_running", "train_n", "dev_accuracy", "dev_ce", "new_best", "best_epoch",
              "upd_res_l2_mean", "upd_W_l2_mean", "upd_b_l2_mean", "res_dist_from_init_l2", "gain_changed", "gain_ge_1pct", "gain_ge_10pct",
              "gain_at_min", "gain_at_max", "epoch_train_s", "epoch_dev_eval_s", "perm_sha256", "utc")


def write_epochs_csv(run: Any) -> None:
    rows = []
    for cond in CONDS:
        p = run.dir / COND_DIR[cond] / "last_state.json"
        if p.is_file():
            with contextlib.suppress(Exception):
                for r in read_json(p).get("history") or []:
                    rr = {k: r.get(k) for k in EPOCH_COLS}
                    for lk, v in (r.get("upd_res_l2_mean_by_layer") or {}).items():
                        rr[f"upd_res_l2_mean.{lk}"] = v
                    for lk, v in (r.get("res_dist_from_init_l2_by_layer") or {}).items():
                        rr[f"res_dist_l2.{lk}"] = v
                    rows.append(rr)
    cols = list(EPOCH_COLS) + [f"upd_res_l2_mean.{k}" for k in LEARN_KEYS] + [f"res_dist_l2.{k}" for k in LEARN_KEYS]
    write_csv(run.dir / "main_epochs.csv", rows, cols)


def _pct(x: Any) -> str:
    return fmt_pct(x)


def _f4(x: Any) -> str:
    return "-" if x is None else f"{float(x):.4f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    cells = [[str(h) for h in headers]] + [["-" if c is None else str(c) for c in r] for r in rows]

    def w(s: str) -> int:
        return sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)

    widths = [max(w(r[i]) for r in cells) for i in range(len(headers))]
    out = []
    for k, r in enumerate(cells):
        out.append("  " + " | ".join(c + " " * (widths[i] - w(c)) for i, c in enumerate(r)))
        if k == 0:
            out.append("  " + "-+-".join("-" * x for x in widths))
    return out


def _korean_font() -> str | None:
    try:
        from matplotlib import font_manager                                  # noqa: PLC0415
    except Exception:                                                        # noqa: BLE001
        return None
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum"):
        if cand in names:
            return cand
    return None


ENTITY_COLOR = {"TRACE": "#2a78d6", "TRACE_BEST": "#2a78d6", "BP_FULL": "#eb6834", "BP_FULL_BEST": "#eb6834", "INITIAL": "#1baf7a",
                V12_STATE: "#2a78d6"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def make_figures(run: Any, data: dict[str, Any]) -> list[str]:
    """그림 3 장 (선택). 색 = 개체 고정 (TRACE 파랑 · BP_FULL 주황 · INITIAL 청록), 범례 + 끝 직접 표시, 축 하나, 옅은 실선 격자."""
    try:
        import matplotlib                                                    # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt                                      # noqa: PLC0415
    except Exception:                                                        # noqa: BLE001
        return []
    fdir = ensure_dir(run.dir / "figures")
    kf = _korean_font()
    T = (lambda ko, en: ko) if kf else (lambda ko, en: en)
    rc = {"font.family": kf} if kf else {}
    made: list[str] = []

    def style(ax: Any) -> None:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.yaxis.label.set_color(INK2)
        ax.xaxis.label.set_color(INK2)

    def end_labels(ax: Any, items: list[tuple[float, float, str]]) -> None:
        """선 끝 직접 표시. 값이 가까우면 세로로 밀어 겹치지 않게 한다 (글자는 본문 색, 개체 색은 선이 가진다)."""
        items = [(x, y, t) for x, y, t in items if y is not None and not (isinstance(y, float) and math.isnan(y))]
        if not items:
            return
        lo, hi = ax.get_ylim()
        gap = 0.06 * (hi - lo)
        items.sort(key=lambda it: it[1])
        pos: list[float] = []
        for _x, y, _t in items:
            pos.append(max(y, pos[-1] + gap) if pos else y)
        from matplotlib.transforms import offset_copy                         # noqa: PLC0415
        tr = offset_copy(ax.transData, fig=ax.figure, x=8, units="points")
        for (x, _y, t), py in zip(items, pos):
            ax.text(x, py, t, transform=tr, va="center", fontsize=9, color=INK, clip_on=False)

    with plt.rc_context(rc):
        hist = data.get("epoch_history") or {}
        if hist:
            fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
            style(ax)
            ends: list[tuple[float, float, str]] = []
            for cond in CONDS:
                h = hist.get(cond) or []
                if not h:
                    continue
                xs = [r["epoch"] for r in h]
                ys = [100.0 * r["dev_accuracy"] for r in h]
                ax.plot(xs, ys, color=ENTITY_COLOR[cond], linewidth=1.6, label=cond)
                be = (data["main"].get(cond) or {}).get("best_epoch")
                if be is not None:
                    yb = next(100.0 * r["dev_accuracy"] for r in h if r["epoch"] == be)
                    ax.plot([be], [yb], marker="o", markersize=6, color=ENTITY_COLOR[cond], markeredgecolor=SURFACE, markeredgewidth=1.5)
                ends.append((xs[-1], ys[-1], cond))
            ax.set_xlim(right=ax.get_xlim()[1] + 0.12 * (ax.get_xlim()[1] - ax.get_xlim()[0]))
            end_labels(ax, ends)
            ax.set_xlabel("epoch")
            ax.set_ylabel(T("dev 정확도 (%)", "dev accuracy (%)"))
            ax.set_title(T("주 학습 dev 정확도 (점 = 선택 epoch)", "Main training: dev accuracy (dot = selected epoch)"), color=INK, fontsize=11, loc="left")
            ax.legend(frameon=False, fontsize=9, labelcolor=INK)
            fig.tight_layout()
            p = fdir / "main_dev_accuracy.png"
            fig.savefig(p, facecolor=SURFACE)
            plt.close(fig)
            made.append(p.name)
        pr = data.get("probe_table") or []
        states = [s for s in STATES_ALL if any(r["state"] == s for r in pr)]
        if pr and states:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150, sharey=True, facecolor=SURFACE)
            xs = list(range(len(PROBE_KEYS)))
            for ax, split in zip(axes, ("dev", "test")):
                style(ax)
                ends = []
                for s in states:
                    ys = []
                    for k in PROBE_KEYS:
                        r = next((r for r in pr if r["state"] == s and r["layer"] == k), None)
                        v = None if r is None else r.get(f"{split}_accuracy")
                        ys.append(np.nan if v is None else 100.0 * v)
                    if all(np.isnan(ys)):
                        continue
                    ax.plot(xs, ys, color=ENTITY_COLOR.get(s, "#2a78d6"), linewidth=1.6, marker="o", markersize=5, markeredgecolor=SURFACE,
                            markeredgewidth=1.2, label=s)
                    last = max(i for i, v in enumerate(ys) if not np.isnan(v))
                    ends.append((xs[last], ys[last], s))
                ax.set_xticks(xs)
                ax.set_xticklabels(PROBE_KEYS, rotation=30, ha="right")
                ax.set_title(T(f"층별 선형 판독 ({split})", f"Linear probe by layer ({split})"), color=INK, fontsize=11, loc="left")
                ax.set_xlim(-0.3, len(xs) - 1 + 2.2)
                end_labels(ax, ends)
            axes[0].set_ylabel(T("정확도 (%)", "accuracy (%)"))
            axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
            fig.tight_layout()
            p = fdir / "probe_accuracy_by_layer.png"
            fig.savefig(p, facecolor=SURFACE)
            plt.close(fig)
            made.append(p.name)
        cosd = data.get("cos_by_layer") or {}
        if cosd:
            fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
            style(ax)
            keys = list(LEARN_KEYS) + ["total"]
            xs = list(range(len(keys)))
            ax.axhline(0.0, color=INK2, linewidth=0.8)
            ends = []
            for s in STATES:
                vals = cosd.get(s)
                if not vals:
                    continue
                ys = [np.nan if vals.get(k) is None else vals[k] for k in keys]
                ax.plot(xs, ys, color=ENTITY_COLOR[s], linewidth=1.6, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=s)
                ends.append((xs[-1], ys[-1], s))
            ax.set_xticks(xs)
            ax.set_xticklabels(keys, rotation=30, ha="right")
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlim(-0.3, len(xs) - 1 + 2.2)
            end_labels(ax, ends)
            ax.set_ylabel("cos(TRACE, −∂CE/∂res)")
            ax.set_title(T("같은 상태 gain-only 방향 정렬 (100 표본 배치 합 벡터)", "Same-state gain-only alignment (sum over 100 samples)"),
                         color=INK, fontsize=11, loc="left")
            ax.legend(frameon=False, fontsize=9, labelcolor=INK)
            fig.tight_layout()
            p = fdir / "trace_bp_cosine_by_layer.png"
            fig.savefig(p, facecolor=SURFACE)
            plt.close(fig)
            made.append(p.name)
    return made


def _collect(run: Any) -> dict[str, Any]:
    st = run.state
    R = st.get("results") or {}
    test = (st.get("test") or {}).get("result") or {}
    main = R.get("main") or {}
    hist = {}
    for cond in CONDS:
        p = run.dir / COND_DIR[cond] / "last_state.json"
        if p.is_file():
            hist[cond] = read_json(p).get("history") or []
    fixed = R.get("fixed_eval") or {}
    probes = R.get("probes") or {}
    refit = R.get("refit") or {}
    ptab = []
    for pid, r in probes.items():
        states = r["shared_states"] if r["state"] == "SHARED" else [r["state"]]
        tr = (test.get("probes") or {}).get(pid) or {}
        for s in states:
            ptab.append({"state": s, "layer": r["layer"], "pid": pid, "shared": r["state"] == "SHARED", "dim": r["dim"], "n_params": r["n_params"],
                         "train_accuracy": r["best"]["train_accuracy"], "train_ce": r["best"]["train_ce"], "train_objective": r["best"]["objective"],
                         "dev_accuracy": r["best"]["dev_accuracy"], "dev_ce": r["best"]["dev_ce"], "test_accuracy": tr.get("accuracy"),
                         "test_ce": tr.get("ce"), "termination": r["termination"], "accepted_iterations": r["accepted_iterations"],
                         "closures_computed": r["closures_computed"], "best_iteration": r["best"]["iteration"], "best_grad_inf": r["best"]["grad_inf"],
                         "final_grad_inf": r["final"]["grad_inf"], "final_objective": r["final"]["objective"],
                         "n_constant": (r.get("normalization") or {}).get("n_constant"), "n_small_sigma": (r.get("normalization") or {}).get("n_small_sigma"),
                         "budget_max_iter": r["budget"]["max_iter"], "budget_max_evals": r["budget"]["max_evals"], "time_s": r["time_s"]})
    cos_by_layer: dict[str, dict[str, Any]] = {}
    diag_full: dict[str, Any] = {}
    for s, d in (R.get("diagnostics") or {}).items():
        p = run.dir / d["file"]
        if p.is_file():
            full = read_json(p)
            diag_full[s] = full
            cos_by_layer[s] = {r["layer"]: r["cos"] for r in full["comparison_rows"]
                               if r["batch"] == "sum" and r["scope"] == "gain" and r["pair"] == "TRACE_vs_BP_FULL"}
    return {"main": main, "epoch_history": hist, "fixed": fixed, "probes": probes, "refit": refit, "test": test, "probe_table": ptab,
            "cos_by_layer": cos_by_layer, "diag": diag_full}


def fairness_record(run: Any) -> dict[str, Any] | None:
    m = run.results("main")
    if "TRACE" not in m or "BP_FULL" not in m:
        return None
    a, b = m["TRACE"], m["BP_FULL"]
    pa, pb = a.get("perm_sha256_by_epoch") or {}, b.get("perm_sha256_by_epoch") or {}
    common = sorted(set(pa) & set(pb), key=int)
    init = run.state_params("INITIAL")["params_sha256"]
    return {"start_equal": a["start_params_sha256"] == b["start_params_sha256"] == init, "perm_equal": all(pa[e] == pb[e] for e in common),
            "epochs_compared": len(common), "initial_params_sha256": init,
            "note_ko": "두 조건이 같은 공통 초기 snapshot 해시에서 출발하고, epoch 마다 같은 순열 (같은 미니배치 ID) 을 썼는지 대조했다."}


def v12_best_gain_change(run: Any) -> dict[str, Any] | None:
    """참고 기록: 입력 v12 best 의 유효 gain 중 초기값에서 1%·10% 이상 변한 수 (이번 실행의 결과가 아니다)."""
    inp = run.inp
    if inp is None or not inp.has("best_params.npz") or not run.ctx.get("best_verified"):
        return None
    arrays, _m = load_npz(inp.local("best_params.npz"))
    eng = run.engine()
    eng.load_gain_arrays(arrays, "g")
    g = eng.gain_values()
    d = np.abs(g - 1.0)
    return {"n": int(len(g)), "n_ge_1pct": int((d >= 0.01).sum()), "n_ge_10pct": int((d >= 0.10).sum())}


def write_summary(run: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    write_epochs_csv(run)
    fr = fairness_record(run)
    if fr is not None:
        run.state["fairness"] = fr
    with contextlib.suppress(Exception):
        gch = v12_best_gain_change(run)
        if gch is not None:
            run.state.setdefault("reference_extra", {})["v12_best_gain_change"] = gch
    d = _collect(run)
    st, cfg = run.state, run.cfg
    test = d["test"]
    lines: list[str] = []
    L = lines.append
    L("층별 판독과 TRACE / 정확 BP 비교 — 결과 요약 (cortex_v13_layer_probe_trace_bp " + VERSION + ")")
    L(f"실행 {run.dir.name} · 범위 {run.scope} · seed {run.seed} (입력 v12 run 의 seed) · 장치 {run.backend.get('device_type')}"
      + (f" ({run.backend.get('cuda_device')})" if run.backend.get("cuda_device") else "") + f" · PyTorch {run.backend.get('torch')} · NumPy {run.backend.get('numpy')}")
    L(f"입력: {cfg['input']['source']['kind']} {cfg['input']['source']['path']} → run {cfg['input']['source']['chosen_run']}")
    L(f"범위: {cfg.get('scope_ko', run.scope)}")
    if run.scope == "v12_best_probes":
        L("  → 기존 v12 best 한 상태 (V12_IMPORTED_BEST) 의 판독만 했다. 새 TRACE/BP 비교 결과가 아니다.")
    elif run.scope == "train_and_diagnostics":
        L("  → 층별 판독기와 raw refit 은 실행하지 않았다 (메뉴 3).")
    L("주의: 판독 정확도는 그 표현의 '선형 판독 가능성' 측정값이다. 정보량이나 인과적 정보 손실률이 아니다. 한 실험으로 원인을 확정하지 않는다.")
    L("")
    # ---- 표 1
    main = d["main"]
    if main:
        L("[표 1] TRACE vs BP_FULL (같은 초기 snapshot · 같은 미니배치 · 같은 η; 고정 재평가 = 선택 상태로 train/dev 를 다시 계산)")
        rows = []
        for cond in CONDS:
            m = main.get(cond)
            if not m:
                continue
            s = f"{cond}_BEST"
            fx = d["fixed"].get(s) or {}
            ts = ((test.get("states") or {}).get(s) or {}).get("main_head") or {}
            h = d["epoch_history"].get(cond) or []
            upd = [r["upd_res_l2_mean"] for r in h if r.get("upd_res_l2_mean") is not None]
            dist = next((r.get("res_dist_from_init_l2") for r in h if r["epoch"] == m["best_epoch"]), None)
            tsec = float((st.get("time") or {}).get(f"{COND_DIR[cond]}_train_s", 0.0))
            rows.append([cond, m["best_epoch"], m["completed_epochs"], _pct((fx.get("train") or {}).get("accuracy")), _f4((fx.get("train") or {}).get("ce")),
                         _pct((fx.get("dev") or {}).get("accuracy")), _f4((fx.get("dev") or {}).get("ce")), _pct(ts.get("accuracy")), _f4(ts.get("ce")),
                         fmt_s(tsec), fmt_f(float(np.mean(upd)) if upd else None, 3), fmt_f(dist, 3)])
        lines += _table(["조건", "선택 epoch", "완료 epoch", "train 정확도", "train CE", "dev 정확도", "dev CE", "test 정확도", "test CE", "주 학습 시간",
                         "배치당 ‖Δres‖ 평균", "선택 상태 ‖res‖"], rows)
        L("  ‖Δres‖ = 투영·반올림 뒤 실제 적용된 gain 잔차 변화의 L2 (배치 평균). 같은 η 라도 두 규칙의 갱신 크기는 다를 수 있다.")
        L("")
    # ---- 표 2
    pt = d["probe_table"]
    if pt:
        L("[표 2] 상태 × 층 선형 판독기 (표준화 + penalty 1e-4/2·ΣW² + float64 L-BFGS; best = dev 선택; CE 보고값은 penalty 제외)")
        rows = []
        for s in STATES_ALL:
            for k in PROBE_KEYS:
                r = next((r for r in pt if r["state"] == s and r["layer"] == k), None)
                if r is None:
                    continue
                rows.append([s, k + (" (공유)" if r["shared"] else ""), r["dim"], f"{r['n_params']:,}", _pct(r["train_accuracy"]), _pct(r["dev_accuracy"]),
                             _pct(r["test_accuracy"]), _f4(r["train_objective"]), _f4(r["train_ce"]), _f4(r["dev_ce"]), r["termination"],
                             f"{r['best_iteration']}/{r['accepted_iterations']}", f"{r['closures_computed']}/{r['budget_max_evals']}",
                             fmt_f(r["best_grad_inf"], 2), fmt_f(r["final_grad_inf"], 2)])
        lines += _table(["상태", "층", "차원", "파라미터", "train", "dev", "test", "train 목적함수", "train CE", "dev CE", "종료", "best/수락 it",
                         "closure/예산", "best ‖g‖∞", "최종 ‖g‖∞"], rows)
        L("  파라미터 수가 층마다 다르다 (같은 함수 계열·손실·선택 절차일 뿐 같은 용량이 아니다). budget_exhausted·stalled 는 그 층의 판독 상한 판단에 불충분하다.")
        L("")
    # ---- 표 3
    if d["refit"]:
        L("[표 3] 원래 IT head vs raw IT refit (정규화 없음·penalty 0) vs 표준화 IT/L3 판독기 — 서로 섞지 않는다")
        rows = []
        for s, r in d["refit"].items():
            fx = d["fixed"].get(s) or {}
            ts = ((test.get("states") or {}).get(s) or {}).get("main_head") or {}
            tr = (test.get("refit") or {}).get(s) or {}
            pr = next((x for x in pt if x["state"] == s and x["layer"] == "IT/L3"), None) or {}
            rows.append([s, "원래 head", _pct((fx.get("train") or {}).get("accuracy")), _pct((fx.get("dev") or {}).get("accuracy")), _pct(ts.get("accuracy")), "-"])
            rows.append([s, "raw refit", _pct(r["best"]["train_accuracy"]), _pct(r["best"]["dev_accuracy"]), _pct(tr.get("accuracy")),
                         f"{r['termination']} (best it {r['best']['iteration']})"])
            rows.append([s, "표준화 판독기", _pct(pr.get("train_accuracy")), _pct(pr.get("dev_accuracy")), _pct(pr.get("test_accuracy")), pr.get("termination")])
        lines += _table(["상태", "판독", "train", "dev", "test", "종료"], rows)
        L("  raw refit 은 원래 head (iteration 0) 를 후보에 포함한 dev 선택이다. 개선이 있어도 '원래 optimizer 만의 문제' 로 단정하지 않는다 (특징 동결 뒤 추가 최적화의 효과).")
        L("")
    # ---- 표 4
    if d["diag"]:
        L("[표 4] 같은 snapshot 교정 방향 (train 고정 100 표본, batch 16; '합' = 배치별 방향 벡터의 합; 방향은 학습률 제외)")
        rows = []
        for s, full in d["diag"].items():
            cr = full["comparison_rows"]
            for k in list(LEARN_KEYS) + ["total", "total(W·b 포함)"]:
                for pair in ("TRACE_vs_BP_FULL", "TRACE_MIN_AWARE_vs_BP_FULL", "BP_STOP_L6_vs_BP_FULL"):
                    rs = [r for r in cr if r["layer"] == k and r["pair"] == pair]
                    sm = next((r for r in rs if r["batch"] == "sum"), None)
                    pb = [r["cos"] for r in rs if r["batch"] != "sum" and r["cos"] is not None]
                    if sm is None:
                        continue
                    desc = sum(1 for r in rs if r["batch"] != "sum" and r.get("dot_grad_delta_sign") == "descent")
                    rows.append([s, k, pair.replace("_vs_", " vs "), sm["scope"], fmt_f(sm["cos"], 3), fmt_f(float(np.mean(pb)) if pb else None, 3),
                                 fmt_pct(sm["sign_agree"]), fmt_f(sm["norm_ratio"], 3),
                                 f"{desc}/{len([r for r in rs if r['batch'] != 'sum'])}" if pair.startswith("TRACE") else "-"])
        lines += _table(["상태", "층", "비교", "범위", "cos (합)", "cos (배치 평균)", "부호 일치", "norm 비 (a/b)", "하강 배치"], rows)
        L("  부호 일치 = 양쪽 모두 |v| > 1e-12·max(1, max|v|) 인 성분만. 하강 배치 = dot(∂CE/∂res, TRACE 후보) < 0 인 배치 수 (미소값은 유보).")
        L("  BP_STOP_L6 (L6 g 고정) 과 TRACE_MIN_AWARE 는 진단 전용 규칙이다. 진단 cosine·norm 만으로 장기 정확도를 예측하지 않는다.")
        drows = []
        for s, full in d["diag"].items():
            for sh in full.get("shadow") or []:
                for nm, v in (sh.get("delta_ce") or {}).items():
                    if isinstance(v, dict) and "delta_ce" in v:
                        drows.append([s, sh["batch"], nm, f"{v['delta_ce']:.3e}", f"{v['linear_pred']:.3e}", f"{v['norm']:.3e}"])
        if drows:
            L("  국소 CE 변화 (float64 그림자, head 고정, 그 배치의 CE; 투영 없이 방향·크기 효과만):")
            lines += _table(["상태", "배치", "후보", "ΔCE", "1차 예측 g·Δ", "‖Δ‖"], drows)
        L("")
    # ---- 표 5
    if d["diag"]:
        L("[표 5] 초기 대비 gain 변화 · 실제 유효 갱신 · 활동·min·L6 집계 (진단 100 표본)")
        rows = []
        for s, full in d["diag"].items():
            for g in full["gain_rows"]:
                rows.append([s, g["layer"], g["n"], g["n_ge_1pct"], g["n_ge_10pct"], g["n_at_min"], g["n_at_max"], fmt_f(g["mean"], 5)])
        lines += _table(["상태", "층", "유효 연결", "|γ−1| ≥ 1%", "≥ 10%", "하한", "상한", "평균 γ"], rows)
        rows = []
        for s, full in d["diag"].items():
            for r in full["rounding_rows"]:
                rows.append([s, r["candidate"], r["layer"], r["n_candidate_nonzero"], r["n_clipped"], r["n_vanished_in_res"], r["n_vanished_in_gamma"],
                             fmt_f(r["dw_rms"], 3), fmt_f(r["dw_abs_q99"], 3)])
        lines += _table(["상태", "후보", "층", "0 아닌 후보", "투영", "res 반올림 소멸", "γ 반올림 소멸", "Δw RMS", "|Δw| q99"], rows)
        L("  γ 반올림 소멸 = float32(1+res_new) = float32(1+res) 인데 후보는 0 이 아닌 칸 (잔차 res 에는 누적되어 나중에 γ 에 나타날 수 있다).")
        rows = []
        for s, full in d["diag"].items():
            for a in full["activity_rows"]:
                rows.append([s, a["layer"], a["inactive_structural"], fmt_pct(a["zero_activity_frac"]), a["min_neurons"], fmt_pct(a["min_tie_frac"]),
                             fmt_f(a["min_trace_share_to_smaller_mean"], 3), fmt_pct(a["l6_at_min_frac"]), fmt_pct(a["l6_at_max_frac"])])
        lines += _table(["상태", "층", "구조상 비활성", "영활동", "min 뉴런", "min 동률", "작은 가지 배분 (raw TRACE)", "L6 하한", "L6 상한"], rows)
        L("  예) A = 1, B = 10 인 min 뉴런: 병목은 A 지만 raw TRACE 배분은 1/11 · 10/11 (자체 검사 예제). 이 예제가 성능 하락 크기를 입증하지는 않는다.")
        L("")
    # ---- 표 6
    L("[표 6] 검사 결과와 실행하지 않은 검사")
    ic = (run.cfg.get("input") or {}).get("checks_summary") or {}
    L(f"  입력 검증: 필수 {ic.get('required_passed')}/{ic.get('required_total')} 통과, 참고 항목 {ic.get('optional_failed_ids') or '모두 통과'}")
    for s, full in d["diag"].items():
        g = full["graph"]
        sh = full.get("shadow") or []
        vj = max((x["vjp_vs_full_graph"]["max_rel_to_maxabs"] for x in sh), default=None)
        c32 = min((x["float32_vs_float64"]["cos"] for x in sh if x["float32_vs_float64"]["cos"] is not None), default=None)
        verdicts: dict[str, int] = {}
        for x in sh:
            for r in x["fd"]:
                verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        L(f"  {s}: 그래프 {'정상' if g['ok'] else '문제'} (채움·비활성 gradient 최대 {g['padding_or_inactive_grad_max']:.1e}, BP forward = 진단 forward 차 "
          f"{g['h_bp_vs_forward_max_abs']:.1e}, 유효 연결 중 이 표본에서 gradient 0 비율 {fmt_pct(g['valid_grad_zero_frac'])}) · VJP vs 전체 그래프 상대차 "
          f"{fmt_f(vj, 2)} · float32/float64 gradient cos ≥ {fmt_f(c32, 6)} · 방향 유한차분 {verdicts} · 진단 전후 파라미터 {'동일' if full['params_unchanged'] else '변함'}")
    fair = st.get("fairness") or {}
    if fair:
        L(f"  공정성: 시작 해시 같음 {fair.get('start_equal')} · epoch 순열 해시 같음 {fair.get('perm_equal')} ({fair.get('epochs_compared')} epoch 비교)")
    L(f"  선택 잠금: {'있음' if st.get('lock_sha256') else '없음'} (sha {str(st.get('lock_sha256'))[:16]}…) · test 평가 횟수 {(st.get('test') or {}).get('attempts', 0)}"
      + (" (같은 snapshot 재평가 포함)" if (st.get('test') or {}).get('attempts', 0) > 1 else ""))
    ev = st.get("events") or []
    L(f"  실행 세션 {len(ev)} 회 (이어하기 {max(0, len(ev) - 1)} 회).")
    dw = sorted(set(st.get("determinism_warnings") or []) | set(_DET_WARNINGS))
    if dw:
        L(f"  결정적 구현 경고 {len(dw)} 건 (warn_only 로 계속 실행; 그 연산은 실행마다 마지막 비트가 다를 수 있다):")
        for w in dw[:8]:
            L("    · " + w.replace("\n", " ")[:220])
    else:
        L(f"  결정적 구현 경고 없음 (이 장치·판: {run.backend.get('device_type')} · PyTorch {run.backend.get('torch')}).")
    notrun = ["원본 MNIST 영상 재검증 (V2 캐시와 분할 해시만 확인)", "v12 코드와의 직접 대조 (프로그램은 v12 를 import 하지 않는다; 개발 중 검사는 별도)",
              "CPU↔CUDA·라이브러리 판 사이의 비트 동일성"]
    if run.scope != "full":
        notrun.append("층별 판독기 (메뉴 3)" if run.scope == "train_and_diagnostics" else "새 TRACE/BP 학습·교정 진단 (메뉴 2)")
    if not run.ctx.get("test_available", True):
        notrun.append("test 평가 (cache_test.npy 없음)")
    L("  실행하지 않은 검사: " + "; ".join(notrun))
    L("")
    # ---- 조건부 해석
    L("[해석] (아래 기준으로 계산한 조건부 문장이다. 원인을 확정하지 않는다)")
    interp = interpretations(run, d)
    for s in interp:
        L("  - " + s)
    L("")
    # ---- 참고 기록
    ref = (run.cfg.get("input") or {}).get("reference") or {}
    L("[참고: 기존 v12 기록 — 이번 실행이 계산한 값이 아니다]")
    ls = ref.get("v12_last_state") or {}
    pv = ls.get("v2_prep_cosine") or {}
    if pv:
        L(f"  V2 준비 단계 코사인 판독 (EMA prototype, 별도 객체): best dev {fmt_pct(pv.get('best_dev_accuracy'))} · 마지막 epoch dev {fmt_pct(pv.get('last_epoch_dev_accuracy'))}"
          " — 새 V2/L3 선형 판독기와 같은 측정이 아니다.")
    b = ref.get("v12_best") or {}
    fm = ref.get("v12_final_metrics") or {}
    if b or fm:
        L(f"  v12 IT head: best epoch {b.get('epoch')} dev {fmt_pct(b.get('dev_accuracy'))} · test {fmt_pct(fm.get('test_accuracy'))} (이미 조회된 test)"
          f" · 마지막 epoch running train {fmt_pct(ls.get('last_epoch_running_train_accuracy'))}")
    if (st.get("reference_extra") or {}).get("v12_best_gain_change"):
        g = st["reference_extra"]["v12_best_gain_change"]
        L(f"  v12 best 의 gain 중 초기값에서 1% 이상 변한 것: {g['n_ge_1pct']:,} / {g['n']:,} (10% 이상 {g['n_ge_10pct']:,})")
    L("")
    t = st.get("time") or {}
    L("[시간] " + " · ".join(f"{k} {fmt_s(v)}" for k, v in sorted(t.items())) + " (이 PC 에서 잰 값; 실행 시간 보장 아님)")
    figs = make_figures(run, d)
    if figs:
        L("[그림] figures/" + ", figures/".join(figs))
    else:
        L("[그림] Matplotlib 이 없어 그림을 만들지 않았다 (CSV·JSON·이 요약은 완성).")
    txt = "\n".join(lines) + "\n"
    atomic_write_bytes(run.dir / "SUMMARY_KO.txt", txt.encode("utf-8"))
    write_csvs(run, d)
    fm_out = {"format": "cortex_v13/final_metrics/1", "program": PROGRAM, "version": VERSION, "program_sha256": run.cfg["program_sha256"],
              "run": run.dir.name, "utc": utc_now(), "scope": run.scope, "seed": run.seed, "input": run.cfg["input"], "backend": backend_describe(run.device),
              "settings": run.S, "conventions": CONVENTIONS, "main": d["main"], "fixed_eval": d["fixed"], "probes": d["probes"], "raw_refit": d["refit"],
              "diagnostics": (st.get("results") or {}).get("diagnostics"), "test": test, "selection_lock_sha256": st.get("lock_sha256"),
              "fairness": st.get("fairness"), "time_s": {k: round(float(v), 2) for k, v in t.items()}, "interpretations_ko": interp,
              "counts": {"cortex_trainings": len(d["main"]), "probes_trained": len(d["probes"]),
                         "probe_results_reported": len(d["probe_table"]), "raw_refits": len(d["refit"])},
              "notes_ko": ["판독 정확도는 선형 판독 가능성이다 (정보량·인과적 손실률이 아니다).", "test 는 선택 잠금 뒤 한 번 계산했다 (blind test 아님).",
                           "기존 v12 기록은 참고 기록으로만 실었다."]}
    write_json(run.dir / "final_metrics.json", fm_out)
    run.add_time("summary_s", time.perf_counter() - t0)
    say("\n" + txt)
    say(f"결과 폴더: {run.dir}")
    return {"figures": figs}


def interpretations(run: Any, d: dict[str, Any]) -> list[str]:
    out: list[str] = []
    pt = d["probe_table"]
    main = d["main"]
    test = d["test"]
    out.append("판정 기준: '높다/낮다' 는 dev 정확도 차이 5%p 이상" + (", '정렬이 낮다' 는 진단한 상태들의 gain 전체 cos (배치 합) 평균 < 0.5"
                                                                 if d["cos_by_layer"] else "") + ". 기준은 해석을 위한 표시일 뿐 통계 검정이 아니다.")
    for s in STATES_ALL:
        rows = [next((r for r in pt if r["state"] == s and r["layer"] == k), None) for k in PROBE_KEYS]
        if not all(rows):
            continue
        accs = [r["dev_accuracy"] for r in rows]
        drops = [(accs[i] - accs[i + 1], PROBE_KEYS[i], PROBE_KEYS[i + 1]) for i in range(len(accs) - 1)]
        dmax = max(drops)
        unconv = [r["layer"] for r in rows if r["termination"] != "converged"]
        lo_i = int(np.argmin(accs[1:])) + 1
        if accs[0] - accs[lo_i] >= 0.05:
            out.append(f"{s}: V2/L3 판독 dev {fmt_pct(accs[0])} 에 비해 {PROBE_KEYS[lo_i]} 가 {fmt_pct(accs[lo_i])} 로 낮다. 가장 큰 단계 하락은 "
                       f"{dmax[1]} → {dmax[2]} ({100 * dmax[0]:.2f}%p) — 그 변환이 판독 가능성을 제한할 후보다. 차원·정규화·regularization·수렴 차이가 "
                       "남으므로 정보 소실률이라고 쓰지 않는다.")
        else:
            out.append(f"{s}: V2/L3 판독 dev ({fmt_pct(accs[0])}) 보다 5%p 이상 낮은 뒤 층 판독이 없다 (뒤 층 최저 {PROBE_KEYS[lo_i]} {fmt_pct(accs[lo_i])}; "
                       f"가장 큰 단계 하락 {dmax[1]} → {dmax[2]} {100 * dmax[0]:+.2f}%p).")
        if unconv:
            out.append(f"{s}: 수렴 표시 (‖g‖∞ ≤ 1e-5) 가 없는 판독기 {', '.join(unconv)} — 그 층의 성능 상한을 판단하기에 불충분하다 (예산을 늘려 재검사할 수 있다).")
    for s, r in d["refit"].items():
        orig = (r.get("original_head") or {}).get("dev_accuracy")
        if orig is not None and r["best"]["dev_accuracy"] > orig:
            out.append(f"{s}: raw IT refit dev {fmt_pct(r['best']['dev_accuracy'])} > 원래 head {fmt_pct(orig)} — 고정된 최종 특징에서 추가 판독 최적화의 여지가 있다 "
                       "(원래 optimizer 만의 문제라고 단정하지 않는다).")
        elif orig is not None:
            out.append(f"{s}: raw IT refit 이 원래 head 의 dev ({fmt_pct(orig)}) 를 넘지 못했다 (dev 선택이 iteration {r['best']['iteration']}).")
    tr, bp = main.get("TRACE"), main.get("BP_FULL")
    cos = [v.get("total") for v in d["cos_by_layer"].values() if v.get("total") is not None]
    if tr and bp:
        diff = bp["best_dev_accuracy"] - tr["best_dev_accuracy"]
        low_align = bool(cos) and float(np.mean(cos)) < 0.5
        out.append(f"선택 dev: BP_FULL {fmt_pct(bp['best_dev_accuracy'])} vs TRACE {fmt_pct(tr['best_dev_accuracy'])} (차 {100 * diff:+.2f}%p); "
                   f"TRACE 방향과 정확 gradient 의 정렬 (gain 전체 cos 평균) {fmt_f(float(np.mean(cos)) if cos else None, 3)}.")
        if diff >= 0.05 and low_align:
            out.append("BP_FULL 이 TRACE 보다 높고 정렬이 낮다 → 현재 교정 규칙이 제한일 가능성을 뒷받침한다. 같은 η 비교이므로 갱신 크기 차이의 효과도 포함된다.")
        v2 = next((r["dev_accuracy"] for r in pt if r["layer"] == "V2/L3"), None)
        if v2 is not None and max(bp["best_dev_accuracy"], tr["best_dev_accuracy"]) <= v2 - 0.05:
            out.append(f"BP_FULL 과 TRACE 가 모두 V2/L3 판독 ({fmt_pct(v2)}) 보다 5%p 이상 낮다 → 고정된 뒤쪽 표현·제약 또는 미완료 최적화를 계속 점검한다 "
                       "(곧바로 구조가 원인이라고 확정하지 않는다).")
        if diff < 0:
            out.append("BP_FULL 이 낮거나 불안정해도 표준 역전파 일반의 열등성으로 해석하지 않는다 (구현 검증·학습률·조건을 구분한다; 이 실험은 η 를 고정했다).")
        ts = test.get("states") or {}
        if ts.get("TRACE_BEST") and ts.get("BP_FULL_BEST"):
            out.append(f"test (선택 잠금 뒤, 보고용): TRACE_BEST {fmt_pct(ts['TRACE_BEST']['main_head']['accuracy'])} · BP_FULL_BEST "
                       f"{fmt_pct(ts['BP_FULL_BEST']['main_head']['accuracy'])}. test 로 모델·층을 다시 고르지 않는다.")
    return out


def write_csvs(run: Any, d: dict[str, Any]) -> None:
    write_csv(run.dir / "probe_results.csv", d["probe_table"],
              ["state", "layer", "pid", "shared", "dim", "n_params", "train_accuracy", "train_ce", "train_objective", "dev_accuracy", "dev_ce",
               "test_accuracy", "test_ce", "termination", "accepted_iterations", "closures_computed", "best_iteration", "best_grad_inf", "final_grad_inf",
               "final_objective", "n_constant", "n_small_sigma", "budget_max_iter", "budget_max_evals", "time_s"])
    hist = []
    for sub in ("probes", "raw_refit"):
        base = run.dir / sub
        if base.is_dir():
            for p in sorted(base.rglob("history.json")):
                with contextlib.suppress(Exception):
                    hist += read_json(p)
    write_csv(run.dir / "probe_history.csv", hist, ["pid", "kind", "state", "layer", "iteration", "closures_computed", "closure_calls", "objective",
                                                    "train_ce", "train_accuracy", "penalty", "grad_inf", "grad_l2", "step_t", "dev_accuracy", "dev_ce", "new_best"])
    rr = []
    test = d["test"]
    for s, r in d["refit"].items():
        fx = d["fixed"].get(s) or {}
        o = r.get("original_head") or {}
        rr.append({"state": s, "original_train_accuracy": (fx.get("train") or {}).get("accuracy"), "original_dev_accuracy": o.get("dev_accuracy"),
                   "original_dev_ce": o.get("dev_ce"), "original_test_accuracy": (((test.get("states") or {}).get(s) or {}).get("main_head") or {}).get("accuracy"),
                   "refit_best_iteration": r["best"]["iteration"], "refit_train_accuracy": r["best"]["train_accuracy"], "refit_train_ce": r["best"]["train_ce"],
                   "refit_dev_accuracy": r["best"]["dev_accuracy"], "refit_dev_ce": r["best"]["dev_ce"],
                   "refit_test_accuracy": ((test.get("refit") or {}).get(s) or {}).get("accuracy"), "refit_test_ce": ((test.get("refit") or {}).get(s) or {}).get("ce"),
                   "termination": r["termination"], "accepted_iterations": r["accepted_iterations"], "closures_computed": r["closures_computed"],
                   "final_grad_inf": r["final"]["grad_inf"]})
    write_csv(run.dir / "raw_refit_results.csv", rr)
    gc_rows, fd_rows, act_rows = [], [], []
    for s, full in d["diag"].items():
        gc_rows += full["comparison_rows"]
        for sh in full.get("shadow") or []:
            for r in sh["fd"]:
                fd_rows.append({"state": s, "batch": sh["batch"], **r})
        for r in full["activity_rows"]:
            act_rows.append({"kind": "activity", **r})
        for r in full["gain_rows"]:
            act_rows.append({"kind": "gain", **r})
        for r in full["rounding_rows"]:
            act_rows.append({"kind": f"update_{r['candidate']}", **{k: v for k, v in r.items() if k != "candidate"}})
    write_csv(run.dir / "gradient_comparison.csv", gc_rows,
              ["state", "batch", "scope", "layer", "pair", "n", "cos", "norm_a", "norm_b", "rms_a", "rms_b", "norm_ratio", "sign_agree", "n_both_nonzero",
               "n_a_zero_only", "n_b_zero_only", "n_both_zero", "dot_grad_delta_sign", "dot_grad_delta"])
    write_csv(run.dir / "directional_checks.csv", fd_rows, ["state", "batch", "scope", "direction", "eps", "fd", "analytic", "abs_err", "rel_err",
                                                           "boundary_crossings", "verdict"])
    cols: list[str] = []
    for r in act_rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    write_csv(run.dir / "activation_and_gain_stats.csv", act_rows, cols)


# ======================================================================
# 실행 만들기 · 이어하기 · 화면 · 명령 줄
# ======================================================================
SCOPE_KO = {"full": "전체 실험 (TRACE/BP 학습 → 층별 판독 19 → raw refit 2 → 교정 진단 → 잠금 → test → 요약)",
            "v12_best_probes": "기존 v12 best 의 층별 판독만 (판독 7 + raw refit 1; 새 TRACE/BP 비교 아님)",
            "train_and_diagnostics": "TRACE/BP 학습과 교정 진단만 (판독기는 실행하지 않는다)"}


def build_settings(epochs: int, batch: int, probe_iter: int, probe_evals: int) -> dict[str, Any]:
    s = json.loads(dumps(SETTINGS, indent=None))
    s["epochs"], s["batch_size"] = int(epochs), int(batch)
    s["probe"]["max_iter"], s["probe"]["max_evals"] = int(probe_iter), int(probe_evals)
    return s


def unique_run_dir(out_root: Path) -> Path:
    base = Path(out_root) / f"run_{local_stamp()}"
    p, k = base, 2
    while p.exists():
        p, k = Path(f"{base}_{k}"), k + 1
    return p


def default_out_root() -> Path:
    try:
        return Path(__file__).resolve().parent / "runs_v13"
    except NameError:
        return Path.cwd() / "runs_v13"


def plan_text(scope: str, ctx: dict[str, Any], S: dict[str, Any], inp: V12Input) -> list[str]:
    n = {s: len(ctx["split"][s]["y"]) for s in SPLITS}
    nb = int(math.ceil(n["train"] / int(S["batch_size"])))
    fdim = sum(ctx["specs"][k].n for k in LEARN_KEYS)
    out = [f"  분할: train {n['train']:,} · dev {n['dev']:,} · test {n['test']:,} (V2/L3 캐시 {ctx['cache_dim']:,} 차원) · 학습 연결 {sum(ctx['edges'].values()):,}"]
    if scope != "v12_best_probes":
        out.append(f"  피질 학습 2 회: TRACE · BP_FULL 각 {S['epochs']} epoch × {nb:,} 배치 (배치 {S['batch_size']}, η_head {S['eta_head']}, η_edge {S['eta_edge']}) + epoch 마다 dev 평가")
    if scope == "full":
        out.append(f"  층별 판독기 19 개 = V2/L3 공유 1 + 3 상태 (INITIAL·TRACE_BEST·BP_FULL_BEST) × 6 층 · raw IT head 재학습 2 개 "
                   f"(L-BFGS 수락 iteration ≤ {S['probe']['max_iter']}, train 전체 closure ≤ {S['probe']['max_evals']} 회씩)")
    elif scope == "v12_best_probes":
        out.append(f"  층별 판독기 7 개 (V12_IMPORTED_BEST) + raw IT head 재학습 1 개 (L-BFGS ≤ {S['probe']['max_iter']} iteration · ≤ {S['probe']['max_evals']} closure)")
    else:
        out.append("  층별 판독기: 실행하지 않는다 (메뉴 3). 선택 모델의 고정 train/dev 재평가만 한다.")
    if scope != "v12_best_probes":
        out.append(f"  같은 상태 교정 진단 3 상태 (train 고정 {N_CLASSES * int(S['diagnostics']['per_class'])} 표본, batch {S['diagnostics']['batch_size']}, "
                   "float64 방향 유한차분 포함)")
    out.append("  선택 잠금 (selection_lock.json) 뒤 test: 상태마다 test forward 한 번")
    if scope != "train_and_diagnostics":
        out.append(f"  디스크: 상태별 특징 캐시 (6 층 × train·dev, float32) {fmt_bytes(fdim * (n['train'] + n['dev']) * 4)} — 한 상태씩 만들고 지운다")
    if inp.kind == "zip":
        sz = sum(int(inp.size(f) or 0) for f in ("cache_train.npy", "cache_dev.npy", "cache_test.npy"))
        out.append(f"  ZIP 입력: 필요한 파일을 {inp.extract_dir} 에 한 번 푼다 (캐시 합계 약 {fmt_bytes(sz)}, test 캐시는 마지막에)")
    out.append("  실행 시간은 PC·장치마다 다르며 짧은 시간을 보장하지 않는다.")
    return out


def create_run(input_path: Any, out_root: Path, scope: str, device_choice: str, epochs: int, batch: int, probe_iter: int, probe_evals: int, *,
               run_hint: str | None = None, chooser: Callable[[list[str]], int] | None = None,
               confirm: Callable[[], bool] | None = None) -> Run | None:
    if scope not in SCOPE_KO:
        raise V13Error(f"알 수 없는 범위 {scope}")
    if epochs < 1 or batch < 1 or probe_iter < 1 or probe_evals < 1:
        raise V13Error("epoch·배치·probe iteration·closure 예산은 1 이상이다.")
    device = torch_setup(device_choice)
    say(f"  계산 장치: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "") + f" · PyTorch {torch.__version__}")
    out_root = ensure_dir(out_root)
    t0 = time.perf_counter()
    inp = V12Input(input_path, out_root, run_hint=run_hint, chooser=chooser)
    say(f"  입력: {'ZIP' if inp.kind == 'zip' else '폴더'} {inp.path} → v12 run {inp.chosen}")
    ctx = validate_input(inp, scope)
    S = build_settings(epochs, batch, probe_iter, probe_evals)
    say("\n[작업량]")
    for ln in plan_text(scope, ctx, S, inp):
        say(ln)
    if confirm is not None and not confirm():
        say("  시작하지 않았다.")
        return None
    run_dir = unique_run_dir(out_root)
    ensure_dir(run_dir)
    jobs = plan_jobs(scope)
    jobs[0]["status"] = "done"
    jobs[0]["utc_done"] = utc_now()
    req = [c for c in ctx["checks"] if c["required"]]
    opt_fail = [c["id"] for c in ctx["checks"] if not c["required"] and not c["ok"]]
    caches = {s: {k: v for k, v in c.items() if k != "path"} for s, c in ctx["caches"].items()}
    cfg = {"format": RUN_FORMAT, "program": PROGRAM, "version": VERSION, "program_sha256": program_sha(), "created_utc": utc_now(), "scope": scope,
           "scope_ko": SCOPE_KO[scope], "seed": ctx["seed"], "settings": S, "device_choice": device_choice, "device": device.type,
           "backend": backend_id(device), "out_root": str(out_root),
           "input": {"source": inp.describe(), "input_id": ctx["input_id"], "files": ctx["files"], "caches": caches, "reference": ctx["reference"],
                     "struct_hash": ctx["hashes"], "struct_hash_v1_record": ctx["struct_hash_v1_record"], "layer_ranges": ctx["ranges"],
                     "learnable_edges": ctx["edges"], "N": ctx["N"], "test_available": ctx["test_available"],
                     "checks_summary": {"required_total": len(req), "required_passed": sum(1 for c in req if c["ok"]), "optional_failed_ids": opt_fail}},
           "conventions": CONVENTIONS, "rng": rng_record(ctx["seed"]),
           "determinism": {"tf32": False, "amp": False, "deterministic_algorithms": "warn_only (경고는 state.json events 에 기록)",
                           "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                           "note_ko": "같은 seed 라도 다른 장치·라이브러리 판에서 비트 단위 재현을 보장하지 않는다."},
           "jobs_planned": [j["id"] for j in jobs]}
    manifest = {"format": "cortex_v13/input_manifest/1", "utc": utc_now(), "source": inp.describe(), "input_id": ctx["input_id"],
                "checks": ctx["checks"], "files": ctx["files"], "caches": caches,
                "split": {"n": {s: int(len(ctx["split"][s]["y"])) for s in SPLITS}, "order_sha256": ctx["order_sha"], "split_sha256": ctx["split_sha256"],
                          "class_counts": {s: np.bincount(ctx["split"][s]["y"], minlength=N_CLASSES).tolist() for s in SPLITS}},
                "stem": {k: ctx["stem_meta"].get(k) for k in ("format", "version", "seed", "N", "struct_hash", "layer_ranges", "stem", "source")},
                "structure": {"hash_recomputed": ctx["hashes"], "v1_hash_record_only": ctx["struct_hash_v1_record"], "learnable_edges": ctx["edges"]},
                "reference": ctx["reference"],
                "note_ko": "이 검사는 v12 기록·캐시·분할의 일치를 확인한다. 원본 MNIST 영상 전체를 다시 검증한 것은 아니다."}
    write_json(run_dir / "config.json", cfg)
    write_json(run_dir / "input_manifest.json", manifest)
    state = {"format": STATE_FORMAT, "program": PROGRAM, "version": VERSION, "run": run_dir.name, "created_utc": utc_now(), "scope": scope,
             "phase": "validate_input", "current_job": None, "jobs": jobs, "seq": 0, "time": {"input_s": time.perf_counter() - t0}, "events": [],
             "identity": {"program_sha256": cfg["program_sha256"], "settings_sha256": sha256_text(dumps(S, indent=None)), "input_id": ctx["input_id"],
                          "backend": cfg["backend"], "scope": scope}, "results": {}, "test": {}}
    run = Run(run_dir, cfg, state)
    run.attach(inp, ctx, device)
    run.save_state()
    say(f"  새 v13 실행 폴더: {run_dir}")
    return run


def open_run(run_dir: Path, input_override: Any = None) -> Run:
    run_dir = Path(run_dir)
    if not (run_dir / "config.json").is_file() or not (run_dir / "state.json").is_file():
        raise V13Error(f"{run_dir} 는 v13 실행 폴더가 아니다 (config.json·state.json 없음).")
    cfg, state = read_json(run_dir / "config.json"), read_json(run_dir / "state.json")
    if cfg.get("format") != RUN_FORMAT or state.get("format") != STATE_FORMAT:
        raise V13Error(f"{run_dir} 는 이 프로그램 ({RUN_FORMAT}) 의 실행 폴더가 아니다.")
    probs = []
    if state["identity"]["program_sha256"] != program_sha():
        probs.append("소스 파일이 실행을 만든 때와 다르다 (SHA256)")
    if sha256_text(dumps(cfg["settings"], indent=None)) != state["identity"]["settings_sha256"] or cfg["scope"] != state["identity"]["scope"]:
        probs.append("config.json 의 설정·범위가 저장 상태와 다르다")
    if probs:
        raise V13Error(f"{run_dir} 를 이어 갈 수 없다 — " + "; ".join(probs) + ". 같은 소스·설정으로 이어 가야 연속 실행과 같은 결과를 기대할 수 있다.")
    device = torch_setup(cfg["device"])
    if backend_id(device) != state["identity"]["backend"]:
        raise V13Error(f"계산 장치·라이브러리 {backend_id(device)} 가 실행의 {state['identity']['backend']} 와 다르다. 같은 장치·판에서 이어 간다.")
    src = cfg["input"]["source"]
    raw = input_override if input_override else src["path"]
    inp = V12Input(raw, Path(cfg["out_root"]), run_hint=src["chosen_run"])
    say(f"  입력 다시 확인: {inp.path} → v12 run {inp.chosen}")
    ctx = validate_input(inp, cfg["scope"])
    if ctx["input_id"] != cfg["input"]["input_id"]:
        raise V13Error("입력 v12 자료의 내용 해시가 실행을 만든 때와 다르다 (다른 입력). 이어 붙이지 않는다.")
    run = Run(run_dir, cfg, state)
    run.attach(inp, ctx, device)
    return run


def progress_text(state: dict[str, Any]) -> str:
    jobs = state.get("jobs") or []
    done = sum(1 for j in jobs if j["status"] == "done")
    nxt = next((j["id"] for j in jobs if j["status"] == "pending"), None)
    return f"작업 {done}/{len(jobs)} 완료 · 다음: {nxt or '없음 (완료)'} · phase {state.get('phase')}"


def find_runs(root: Path) -> list[Path]:
    out = []
    if (root / "state.json").is_file():
        return [root]
    with contextlib.suppress(OSError):
        for c in sorted(root.iterdir()):
            if c.is_dir() and c.name.startswith("run_") and (c / "state.json").is_file():
                out.append(c)
    return out


# ---------------------------------------------------------------------- 화면 입력
class QuitRequested(Exception):
    pass


def ask(label: str, default: str = "") -> str:
    try:
        s = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    except EOFError:
        raise QuitRequested from None
    return s or default


def ask_int(label: str, default: int, lo: int = 1) -> int:
    while True:
        s = ask(label, str(default))
        try:
            v = int(s)
            if v >= lo:
                return v
        except ValueError:
            pass
        say(f"  {lo} 이상의 정수를 입력한다.")


def ask_path(label: str, default: str = "", example: str = "") -> Path:
    while True:
        s = ask(label + (f" (예: {example})" if example else ""), default)
        if not s:
            say("  경로를 입력한다 (0 = 종료).")
            continue
        if s.strip() == "0":
            raise QuitRequested
        try:
            return clean_user_path(s)
        except V13Error as exc:
            say(f"  {exc}")


def choose_from(names: list[str]) -> int:
    say("  입력 안에 v12 run 이 여러 개다:")
    for i, nm in enumerate(names, 1):
        say(f"    {i}. {nm}")
    while True:
        k = ask_int("  사용할 run 번호", 1)
        if 1 <= k <= len(names):
            return k - 1


def run_and_report(run: Run) -> int:
    try:
        run.execute()
    except Interrupted as exc:
        say(f"\n[멈춤] {exc}\n  이어 가려면: 다시 실행 → 메뉴 4 → {run.dir}   (명령 줄: --resume \"{run.dir}\")")
        return 130
    except KeyboardInterrupt:
        say(f"\n[즉시 멈춤] 마지막 저장 지점부터 이어 갈 수 있다 (메뉴 4 → {run.dir}).")
        return 130
    except V13Error as exc:
        with contextlib.suppress(Exception):
            run.state.setdefault("events", []).append({"utc": utc_now(), "error": str(exc)[:2000]})
            write_json(run.dir / "state.json", run.state)
        say(f"\n[오류] {exc}\n  실행 폴더: {run.dir}")
        return 1
    say(f"\n완료: {run.dir / 'SUMMARY_KO.txt'}")
    return 0


def show_summary(p: Path) -> int:
    runs = find_runs(p)
    if not runs:
        say(f"  {p} 에 v13 실행 폴더가 없다.")
        return 1
    for r in runs:
        st = read_json(r / "state.json")
        say(f"\n=== {r.name} — {progress_text(st)}")
        if (r / "SUMMARY_KO.txt").is_file():
            say((r / "SUMMARY_KO.txt").read_text(encoding="utf-8"))
    return 0


def interactive() -> int:
    say(f"\n층별 판독과 TRACE / 정확 BP 비교   ({PROGRAM} {VERSION})\n")
    out_default = str(default_out_root())
    say("  v12 결과 폴더 또는 ZIP: [직접 입력]")
    say(f"  새 결과 저장 폴더: [{out_default}]")
    say("  장치: [자동 / CPU / CUDA]")
    say(f"  주 학습 epoch: [{SETTINGS['epochs']}]       주 학습 배치: [{SETTINGS['batch_size']}]")
    say(f"  probe 최대 iteration: [{SETTINGS['probe']['max_iter']}]   closure 예산: [{SETTINGS['probe']['max_evals']}]\n")
    say("  1. 전체 실험 실행 [기본]")
    say("     공통 초기값 → TRACE/BP → 층별 probe → raw head 재학습")
    say("     → 동일상태 교정 진단 → 선택 잠금 → test → 결과 요약")
    say("  2. 기존 v12 best의 층별 판독만 실행")
    say("  3. TRACE/BP 학습과 교정 진단만 실행")
    say("  4. 미완료 v13 실행 이어하기")
    say("  5. 저장 결과 요약 보기")
    say("  0. 종료\n")
    try:
        c = ask("선택", "1")
        if c == "0":
            return 0
        if c in MENU_SCOPE:
            scope = MENU_SCOPE[c]
            inp = ask_path("v12 결과 폴더 또는 ZIP", "", EXAMPLE_INPUT)
            out = ask_path("새 결과 저장 폴더", out_default)
            dev = ask("장치 (자동 / CPU / CUDA)", "자동")
            epochs, batch = SETTINGS["epochs"], SETTINGS["batch_size"]
            if scope != "v12_best_probes":
                epochs = ask_int("주 학습 epoch", epochs)
                batch = ask_int("주 학습 배치", batch)
            pit, pev = SETTINGS["probe"]["max_iter"], SETTINGS["probe"]["max_evals"]
            if scope != "train_and_diagnostics":
                pit = ask_int("probe 최대 iteration", pit)
                pev = ask_int("closure 예산", pev)
            say(f"\n  범위: {SCOPE_KO[scope]}")
            run = create_run(inp, out, scope, dev, epochs, batch, pit, pev, chooser=choose_from,
                             confirm=lambda: ask("\n시작한다 (Enter = 예, n = 아니오)", "y").lower() not in ("n", "no", "아니오"))
            return 0 if run is None else run_and_report(run)
        if c == "4":
            p = ask_path("이어 갈 v13 실행 폴더 (또는 그 상위 폴더)", out_default)
            runs = [r for r in find_runs(p) if read_json(r / "state.json").get("phase") != "done"]
            if not runs:
                say("  이어 갈 미완료 v13 실행이 없다.")
                return 1
            if len(runs) > 1:
                for i, r in enumerate(runs, 1):
                    say(f"    {i}. {r.name} — {progress_text(read_json(r / 'state.json'))}")
                k = ask_int("  번호", 1)
                runs = [runs[min(max(k, 1), len(runs)) - 1]]
            r = runs[0]
            say(f"  {r.name}: {progress_text(read_json(r / 'state.json'))}")
            cfg = read_json(r / "config.json")
            override = None
            if not Path(cfg["input"]["source"]["path"]).exists():
                override = ask_path("기록된 v12 입력 경로가 없다. 같은 v12 자료의 새 경로", "")
            return run_and_report(open_run(r, override))
        if c == "5":
            return show_summary(ask_path("v13 실행 폴더 (또는 그 상위 폴더)", out_default))
        say("  0~5 중에서 고른다.")
        return 1
    except QuitRequested:
        return 0
    except V13Error as exc:
        say(f"\n[중단] {exc}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="층별 판독과 TRACE / 정확 BP 비교 (v12 결과 입력). 인자 없이 실행하면 화면 메뉴.")
    ap.add_argument("--input", help="v12 run 폴더 또는 그 run 이 든 ZIP")
    ap.add_argument("--out", default=None, help="새 결과 저장 폴더 (기본: 프로그램 폴더/runs_v13)")
    ap.add_argument("--menu", choices=("1", "2", "3"), default="1", help="1 전체 · 2 v12 best 판독만 · 3 학습·진단만")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    ap.add_argument("--batch", type=int, default=SETTINGS["batch_size"])
    ap.add_argument("--probe-iter", type=int, default=SETTINGS["probe"]["max_iter"])
    ap.add_argument("--probe-evals", type=int, default=SETTINGS["probe"]["max_evals"])
    ap.add_argument("--run", default=None, help="입력에 v12 run 이 여러 개일 때 고를 run 이름")
    ap.add_argument("--yes", action="store_true", help="작업량 확인 없이 시작")
    ap.add_argument("--resume", default=None, help="이어 갈 v13 실행 폴더")
    ap.add_argument("--resume-input", default=None, help="(이어하기) 옮긴 v12 입력의 새 경로 (내용 해시가 같아야 한다)")
    ap.add_argument("--summary", default=None, help="v13 실행 폴더의 요약 보기")
    ap.add_argument("--self-test", action="store_true", help="작은 문법·수학 검사 (임시 폴더, 정규 실험과 분리)")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    if np is None or torch is None:
        say("필수 패키지가 없다 — 자동으로 설치하지 않는다.")
        if np is None:
            say(f"  NumPy: {_NUMPY_ERROR}\n    설치: pip install numpy")
        if torch is None:
            say(f"  PyTorch: {_TORCH_ERROR}\n    설치: https://pytorch.org 의 안내에서 자기 환경 (CPU / CUDA 판) 명령을 고른다 (예: pip install torch)")
        return 2
    if sys.version_info < (3, 10):
        say("Python 3.10 이상이 필요하다.")
        return 2
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            return self_test()
        if args.summary:
            return show_summary(clean_user_path(args.summary))
        if args.resume:
            return run_and_report(open_run(clean_user_path(args.resume), args.resume_input))
        if args.input:
            out = clean_user_path(args.out) if args.out else default_out_root()
            run = create_run(args.input, out, MENU_SCOPE[args.menu], args.device, args.epochs, args.batch, args.probe_iter, args.probe_evals,
                             run_hint=args.run, confirm=None if args.yes else (lambda: ask("\n시작한다 (Enter = 예, n = 아니오)", "y").lower()
                                                                                   not in ("n", "no", "아니오")))
            return 0 if run is None else run_and_report(run)
    except QuitRequested:
        return 0
    except V13Error as exc:
        say(f"\n[중단] {exc}")
        return 1
    except KeyboardInterrupt:
        say("\n[중단] Ctrl+C — 실행을 만들기 전이라 저장한 것이 없다 (원본 입력은 그대로).")
        return 130
    try:
        return interactive()
    except KeyboardInterrupt:
        say("\n[중단] Ctrl+C")
        return 130


# ======================================================================
# --self-test: 작은 문법·수학 검사 (합성 입력, 임시 폴더). 정규 실험 결과와 섞지 않는다.
# ======================================================================
TINY_L6 = {"enabled": True, "layers": ["L2", "L3"], "epsilon": 1e-6, "kappa": 0.25, "g_min": 0.5, "g_max": 2.0, "neighbor_grid_radius": 1,
           "include_self": False}


def _tiny_spec(area: str, name: str, src_key: str, n_src: int, rows_a: list[list[tuple[int, float]]],
               rows_b: list[list[tuple[int, float]]] | None = None, op_min: Sequence[int] = (), missing: Sequence[int] = ()) -> LayerSpec:
    n_pos, n_ch = 2, 2
    n = n_pos * n_ch
    c = CSR(n, n_src)
    for r, lst in enumerate(rows_a):
        for s, w in lst:
            c.add(r, s, w, "A")
    for r, lst in enumerate(rows_b or []):
        for s, w in lst:
            c.add(r, s, w, "B")
    om = np.zeros(n, bool)
    om[list(op_min)] = True
    ms = None
    if missing:
        ms = np.zeros(n, bool)
        ms[list(missing)] = True
    xs = np.arange(n, dtype=np.float64)
    return LayerSpec(area, name, src_key, n_src, n_pos, n_ch, ["c0", "c1"], ["tiny", "tiny"], {}, om, c.finalize(), xs, xs,
                     (np.arange(n_pos, dtype=np.float64), np.zeros(n_pos), np.ones(n_pos)), (1, n_pos), grid_neighbors(1, n_pos, 1), missing=ms)


def tiny_world() -> tuple[dict[str, LayerSpec], np.ndarray, np.ndarray]:
    """6 층 합성 구조 (V4/L2·IT/L2 에 min 뉴런, V4/L2 에 구조상 비활성 1 개, L2·L3 에 L6)."""
    sp = {
        "V4/L4": _tiny_spec("V4", "L4", "V2/L3", 6, [[(0, 1.0), (1, 0.5)], [(2, 1.0)], [(3, 0.7), (4, 0.3)], [(5, 1.0), (0, 0.2)]]),
        "V4/L2": _tiny_spec("V4", "L2", "V4/L4", 4, [[(0, 1.0)], [(0, 0.5), (1, 0.5), (2, 0.25)], [(2, 1.0)], [(3, 1.0)]],
                            [[(1, 1.0)], [], [(3, 1.0)], []], op_min=(0, 2, 3), missing=(3,)),
        "V4/L3": _tiny_spec("V4", "L3", "V4/L2", 4, [[(r, 0.6), ((r + 2) % 4, 0.4)] for r in range(4)]),
        "IT/L4": _tiny_spec("IT", "L4", "V4/L3", 4, [[(r, 1.0), ((r + 1) % 4, 0.5)] for r in range(4)]),
        "IT/L2": _tiny_spec("IT", "L2", "IT/L4", 4, [[(0, 1.0)], [(1, 1.0), (2, 0.5)], [(2, 1.0)], [(3, 1.0)]],
                            [[(1, 1.0)], [], [], [(0, 1.0)]], op_min=(0, 3)),
        "IT/L3": _tiny_spec("IT", "L3", "IT/L2", 4, [[(r, 1.0)] for r in range(4)]),
    }
    N = 0
    for k in LEARN_KEYS:
        sp[k].start, sp[k].end = N, N + sp[k].n
        N += sp[k].n
    q = np.ones(N, np.float32)
    thr = np.zeros(N, np.float32)
    thr[sp["V4/L3"].start] = 0.05
    return sp, q, thr


class _FakeRun:
    def __init__(self, device: Any, probe: dict[str, Any] | None = None):
        self.S = json.loads(dumps(SETTINGS, indent=None))
        if probe:
            self.S["probe"].update(probe)
        self.device = device
        self.stop = StopFlag()
        self.time: dict[str, float] = {}

    def add_time(self, k: str, v: float) -> None:
        self.time[k] = self.time.get(k, 0.0) + v


def self_test() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cortex_v13_selftest_"))
    say(f"\n[self-test] 임시 폴더 {tmp} (정규 실험과 분리, 합성 입력만 사용)")
    device = torch_setup("cpu")
    results: list[dict[str, Any]] = []

    def rec(name: str, ok: Any, detail: str = "") -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})
        say(f"  [{'통과' if ok else '실패'}] {name}" + (f" — {detail}" if detail else ""))

    def guard(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:                                               # noqa: BLE001 — 검사 하나의 실패가 나머지를 막지 않게
            rec(name, False, f"{type(exc).__name__}: {exc}")

    # ---- 1. 비매끄러운 점의 미분 관례
    def t_conventions() -> None:
        x = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64, requires_grad=True)
        (g,) = torch.autograd.grad(f_relu0(x).sum(), x)
        rec("ReLU 0 에서 미분 0", g.tolist() == [0.0, 0.0, 1.0], str(g.tolist()))
        x = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64, requires_grad=True)
        (g,) = torch.autograd.grad(f_rect(x).sum(), x)
        rec("L6 정류: a ≥ 0 이면 1, a < 0 이면 0", g.tolist() == [0.0, 1.0, 1.0], str(g.tolist()))
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([2.0, 2.0, 1.0], dtype=torch.float64, requires_grad=True)
        ga, gb = torch.autograd.grad(f_min2(a, b).sum(), (a, b))
        rec("min: 작은 가지 1 · 큰 가지 0 · 동률 0.5/0.5", ga.tolist() == [1.0, 0.5, 0.0] and gb.tolist() == [0.0, 0.5, 1.0], f"{ga.tolist()} / {gb.tolist()}")
        x = torch.tensor([0.4, 0.5, 1.0, 2.0, 2.1], dtype=torch.float64, requires_grad=True)
        (g,) = torch.autograd.grad(f_clip_in(x, 0.5, 2.0).sum(), x)
        rec("L6 clip: 밖 0 · 안 1 · 경계 1", g.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0], str(g.tolist()))
    guard("미분 관례", t_conventions)

    sp, q, thr = tiny_world()
    model = {"l6": TINY_L6}

    def eng_of(dt: Any, with_trace: bool = True) -> Engine:
        return Engine(TT(device, dt, 1 << 20), sp, model, q, thr, gain_min=0.05, gain_max=4.0, memory_mb=1, with_trace=with_trace)

    # ---- 2. TRACE 배분 예: A = 1, B = 10
    def t_trace_example() -> None:
        eng = eng_of(torch.float32)
        L = eng.by_key["V4/L2"]
        x = torch.tensor([[1.0, 10.0, 0.0, 0.0]])
        xe = eng.tt.pad_zero_col(x)
        sa, sb = L.A.apply(eng.tt, xe), L.B.apply(eng.tt, xe)
        lamA, lamB = eng._lambdas(L, {"sa": sa, "sb": sb})
        ell = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        raw = eng._route_layer(L, x, ell, lamA, lamB, need_src=True, min_aware=False)
        eng.clear_grads()
        ma = eng._route_layer(L, x, ell, lamA, lamB, need_src=True, min_aware=True)
        eng.clear_grads()
        r0, m0 = raw[0, :2].tolist(), ma[0, :2].tolist()
        rec("raw TRACE 배분 A=1·B=10 → 1/11·10/11", abs(r0[0] - 1 / 11) < 1e-7 and abs(r0[1] - 10 / 11) < 1e-6, f"{r0}")
        rec("TRACE_MIN_AWARE 배분 → 1·0", abs(m0[0] - 1.0) < 1e-7 and m0[1] == 0.0, f"{m0}")
    guard("TRACE 배분 예", t_trace_example)

    # ---- 3. 정확 BP: VJP = 전체 그래프, gradcheck (매끄러운 점), 그래프 연결
    def t_bp() -> None:
        eng = eng_of(torch.float64, with_trace=False)
        found = None
        for sd in range(50):
            rng = np.random.default_rng([13, sd])
            eng.set_res_list([rng.uniform(-0.1, 0.1, (p.r, p.K)) * p.valid_np for p in eng.parts])
            x = eng.tt.f(rng.uniform(0.2, 1.0, (5, 6)))
            with torch.no_grad():
                r = eng.forward(x, record="full")
            margins = []
            for Lk, L in eng.by_key.items():
                rr = r["layers"][Lk]
                margins.append(float(rr["pre"][:, torch.from_numpy(~L.invalid_np)].abs().min()))     # 구조상 비활성 (u = 0 고정) 은 제외
                if L.B is not None:
                    margins.append(float((rr["sa"] - rr["sb"]).abs()[:, torch.from_numpy(L.op_min_np & ~L.invalid_np)].min()))
                if rr.get("l6_x") is not None:
                    margins.append(float(torch.minimum((rr["l6_x"] - 0.5).abs(), (rr["l6_x"] - 2.0).abs()).min()))
            if min(margins) > 1e-3:
                found = (sd, rng, x)
                break
        if found is None:
            rec("gradcheck 용 매끄러운 점", False, "50 개 seed 에서 경계에서 떨어진 점을 찾지 못함")
            return
        _sd, rng, x = found
        W = rng.normal(0, 0.5, (N_CLASSES, 4))
        b = rng.normal(0, 0.1, N_CLASSES)
        y = np.asarray([0, 3, 7, 3, 9])
        out = eng.bp_grads(x, W, b, y)
        base = [p.res.detach().clone() for p in eng.parts]
        leaves = [t.clone().requires_grad_(True) for t in base]
        Wt, bt = torch.from_numpy(W).requires_grad_(True), torch.from_numpy(b).requires_grad_(True)
        oh = torch.from_numpy(onehot(y))
        with torch.enable_grad():
            h = eng.forward(x, leaves=leaves)["h"]
            ce = -(torch.log_softmax(h @ Wt.T + bt, dim=1) * oh).sum() / len(y)
            gfull = torch.autograd.grad(ce, leaves)
        diff = max(float((a - b_).abs().max()) for a, b_ in zip(out["grads"], gfull))
        rec("VJP (NumPy head 해석 미분 + autograd) = 전체 Torch 그래프", diff <= 1e-12, f"최대 차 {diff:.2e}")
        sizes = [p.r * p.K for p in eng.parts]

        def fn(flat: Any) -> Any:
            parts, k = [], 0
            for p, s in zip(eng.parts, sizes):
                parts.append(flat[k:k + s].reshape(p.r, p.K))
                k += s
            hh = eng.forward(x, leaves=parts)["h"]
            return -(torch.log_softmax(hh @ torch.from_numpy(W).T + torch.from_numpy(b), dim=1) * oh).sum() / len(y)

        flat0 = torch.cat([t.reshape(-1) for t in base]).clone().requires_grad_(True)
        ok = torch.autograd.gradcheck(fn, (flat0,), eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=False)
        rec("gradcheck (float64, 매끄러운 점, eps 1e-6 · atol 1e-5 · rtol 1e-3)", ok, "경계 여유 > 1e-3")
        pad = max(float(g.abs()[torch.from_numpy(~p.valid_np)].max()) if (~p.valid_np).any() else 0.0 for g, p in zip(out["grads"], eng.parts))
        rec("채움·구조상 비활성 칸의 gradient = 0", pad == 0.0, f"최대 {pad:.1e}")
        with torch.no_grad():
            h0 = eng.forward(x)["h"].numpy()
        rec("BP forward 값 = no_grad forward 값", np.array_equal(h0, out["h"]), "")
        rec("전역 grad 모드가 꺼져 있지 않다", torch.is_grad_enabled(), "")
        st = eng.bp_grads(x, W, b, y, stop_l6=True)
        dd = max(float((a - b_).abs().max()) for a, b_ in zip(st["grads"], out["grads"]))
        rec("BP_STOP_L6 는 BP_FULL 과 다른 미분 (L6 미분 기여 확인)", dd > 0, f"최대 차 {dd:.2e}")
    guard("정확 BP", t_bp)

    # ---- 4. TRACE 보존: 양수 가중치 층에서 Σ d_src = Σ_{S>0} ℓ
    def t_trace_conservation() -> None:
        eng = eng_of(torch.float64)
        L = eng.by_key["V4/L3"]
        rng = np.random.default_rng(5)
        x = eng.tt.f(rng.uniform(0.1, 1.0, (3, 4)))
        ell = eng.tt.f(rng.normal(0, 1, (3, 4)))
        d = eng._route_layer(L, x, ell, None, None, need_src=True, min_aware=False)
        eng.clear_grads()
        err = float((d.sum(1) - ell.sum(1)).abs().max())
        rec("TRACE 배분 보존 (Σ_src d = Σ_dst ℓ, 양수 가중치)", err < 1e-12, f"최대 차 {err:.1e}")
    guard("TRACE 보존", t_trace_conservation)

    # ---- 5. 선형 판독기 수학 · L-BFGS 예산·복원·이어하기
    def t_probe() -> None:
        rng = np.random.default_rng(11)
        N, d = 600, 6
        y = rng.integers(0, 3, N)
        X = rng.normal(0, 1, (N, d)).astype(np.float32)
        X[:, 0] = 3.0
        X[:, 1] = 0.0
        X[:, 2:] += y[:, None].astype(np.float32) * 0.8
        Xd = rng.normal(0, 1, (200, d)).astype(np.float32)
        yd = rng.integers(0, 3, 200)
        Xd[:, 2:] += yd[:, None].astype(np.float32) * 0.8

        def mk(sub: str, probe: dict[str, Any], **kw: Any) -> LBFGSProbe:
            fr = _FakeRun(device, probe)
            return LBFGSProbe(fr, pid=f"selftest:{sub}", kind="probe", state="SYN", layer="syn", out_dir=tmp / sub, X_tr=X, y_tr=y, X_dv=Xd,
                              y_dv=yd, standardize=True, penalty=1e-4, W0=None, b0=None, feature_key={"syn": sub}, **kw)

        a = mk("mb", {"microbatch": 64})
        a._normalization()
        rec("상수열·영벡터열 σ = 0 표시", a.norm_info["n_constant"] == 2 and a.norm_info["n_all_zero"] == 1, str({k: a.norm_info[k] for k in ("n_constant", "n_all_zero")}))
        Xs = (torch.from_numpy(X).double() - a.mu_t) / a.sig_t
        rec("train 정규화 뒤 상수열 = 0 (정확히)", float(Xs[:, :2].abs().max()) == 0.0, "")
        a.oh_tr, a.yt_tr = torch.from_numpy(onehot(y)), torch.from_numpy(y)
        Wr = torch.from_numpy(rng.normal(0, 0.3, (N_CLASSES, d))).requires_grad_(True)
        br = torch.from_numpy(rng.normal(0, 0.1, N_CLASSES)).requires_grad_(True)
        a.W, a.b = Wr, br
        e1 = a._evaluate()
        a.mb = N
        e2 = a._evaluate()
        dg = max(float((e1["gW"] - e2["gW"]).abs().max()), float((e1["gb"] - e2["gb"]).abs().max()))
        rec("microbatch 512 식 누적 = 전체 배치 계산 (objective·gradient)", abs(e1["obj"] - e2["obj"]) < 1e-13 and dg < 1e-13,
            f"objective 차 {abs(e1['obj'] - e2['obj']):.1e}, gradient 차 {dg:.1e}")
        pen = 0.5 * 1e-4 * float((Wr.detach() ** 2).sum())
        rec("penalty 는 closure 에 한 번 (bias 제외)", abs((e1["obj"] - e1["ce"]) - pen) < 1e-15 and abs(e1["pen"] - pen) < 1e-15, "")
        full = mk("full", {"max_iter": 12, "max_evals": 200})
        rf = full.execute()
        rec("L-BFGS 종료 사유 기록 (converged·stalled·budget_exhausted 중 하나)", rf["termination"] in ("converged", "stalled", "budget_exhausted"),
            f"{rf['termination']} · 수락 {rf['accepted_iterations']} · closure {rf['closures_computed']}")
        small = mk("budget", {"max_iter": 50, "max_evals": 4})
        rs = small.execute()
        last_hash = sha256_array(small.W.detach().numpy(), small.b.detach().numpy())
        rec("closure 예산 소진 → budget_exhausted · 직전 수락 상태로 복원", rs["termination"] == "budget_exhausted" and last_hash == small.cur_key
            and rs["closures_computed"] <= 4, f"{rs['termination_detail_ko']}")
        # 이어하기: 수락 iteration 5 뒤에 멈춤 → 새 객체로 이어 가기 → 연속 실행과 같은 결과
        part = mk("resume", {"max_iter": 12, "max_evals": 200})
        orig_record = part._record

        def stop_after_5(step_t: Any) -> None:
            orig_record(step_t)
            if part.it == 5:
                part.run.stop.requested = True
        part._record = stop_after_5                                            # type: ignore[method-assign]
        stopped = False
        try:
            part.execute()
        except Interrupted:
            stopped = True
        cont = mk("resume", {"max_iter": 12, "max_evals": 200})
        rc = cont.execute()
        same = (np.array_equal(cont.W.detach().numpy(), full.W.detach().numpy()) and
                [h["objective"] for h in cont.history] == [h["objective"] for h in full.history] and rc["termination"] == rf["termination"])
        rec("판독기 이어하기 (수락 step 경계에서 멈춤) = 연속 실행 (비트 단위)", stopped and same, f"멈춤 {stopped}, 같음 {same}")
        # line search 도중 멈춤 → trial 을 저장하지 않고 직전 수락 상태로
        ls = mk("ls_abort", {"max_iter": 12, "max_evals": 200})
        orig_eval = ls._evaluate
        cnt = {"n": 0}

        def eval_hook() -> dict[str, Any]:
            cnt["n"] += 1
            if cnt["n"] == 5:
                ls.run.stop.requested = True
            return orig_eval()
        ls._evaluate = eval_hook                                               # type: ignore[method-assign]
        aborted = False
        try:
            ls.execute()
        except Interrupted:
            aborted = True
        obj = torch.load(io.BytesIO((tmp / "ls_abort" / "last.pt").read_bytes()), weights_only=True)
        meta = json.loads(obj["meta_json"])
        saved_hash = sha256_array(obj["W"].numpy(), obj["b"].numpy())
        rec("line search 도중 멈춤 → 미완료 trial 미저장 · 직전 수락 상태 저장", aborted and saved_hash == meta["cur_key"] and meta["aborts"] == 1,
            f"수락 iteration {meta['it']}, closure {meta['computed']}")
        ls2 = mk("ls_abort", {"max_iter": 12, "max_evals": 200})
        r2 = ls2.execute()
        rec("line search 멈춤 뒤 이어하기 완료", r2["termination"] in ("converged", "stalled", "budget_exhausted"), f"{r2['termination']}")
    guard("판독기", t_probe)

    # ---- 6. 입력 어댑터 안전 · 원자적 저장
    def t_io() -> None:
        bad = ["../a.npy", "/abs/x", "C:/x", "a/../../b", "a\\..\\b", ""]
        good = ["runs_v12/run_1/config.json", "config.json"]
        rec("ZIP 경로 이탈 항목 거부", all(_safe_member(x) is None for x in bad) and all(_safe_member(x) == x for x in good), "")
        p = tmp / "io" / "t.npz"
        save_npz(p, {"a": np.arange(5)}, {"k": "한글 값"})
        arr, meta = load_npz(p)
        rec("npz + 상태 JSON 원자적 저장·읽기", np.array_equal(arr["a"], np.arange(5)) and meta == {"k": "한글 값"}, "")
        zp = tmp / "io" / "evil.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("../escape.txt", "x")
        try:
            V12Input(str(zp), tmp / "io_out")
            rec("경로 이탈만 있는 ZIP 은 run 으로 받지 않는다", False, "")
        except InputError as exc:
            rec("경로 이탈만 있는 ZIP 은 run 으로 받지 않는다", "무시한 ZIP 항목 1" in str(exc), "")
        rec("ZIP 밖으로 파일을 만들지 않았다", not (tmp / "escape.txt").exists() and not (tmp / "io_out" / ".." / "escape.txt").exists(), "")
    guard("입출력", t_io)

    # ---- 7. 실제 기본 구조 (v12 기본 모델) 와 합성 V2 입력의 엔진 동작
    def t_real() -> None:
        t0 = time.perf_counter()
        specs, ranges, hashes = build_structure(V12_MODEL_DEFAULT)
        edges = count_learnable_edges(specs)
        exp_edges = {"V4/L4": 6912, "V4/L2": 3492, "V4/L3": 15840, "IT/L4": 2112, "IT/L2": 2640, "IT/L3": 146}
        rec("기본 구조: N 56,948 · V2/L3 6,912 · IT/L3 146 · 학습 연결 31,142", ranges["IT/L3"][1] == 56948 and specs["V2/L3"].n == 6912
            and specs["IT/L3"].n == 146 and edges == exp_edges, f"{edges} ({fmt_s(time.perf_counter() - t0)})")
        rec("V4/L2 구조상 비활성 78 개", int(specs["V4/L2"].missing.sum()) == 78, "")
        rec("V2·IT 구조 해시 = v12 기록값", hashes["V2"].startswith("012ed5f450b4") and hashes["IT"].startswith("51c93d3a06df"),
            f"V2 {hashes['V2'][:12]} · IT {hashes['IT'][:12]} · V4 {hashes['V4'][:12]} (V4 는 NumPy·수학 라이브러리에 따라 다를 수 있어 입력 기록과 대조한다)")
        N = ranges["IT/L3"][1]
        qf, tf = np.ones(N, np.float32), np.zeros(N, np.float32)
        eng = Engine(TT(device, torch.float32, 1 << 25), specs, V12_MODEL_DEFAULT, qf, tf, gain_min=0.05, gain_max=4.0, memory_mb=1024)
        rng = np.random.default_rng(3)
        x = eng.tt.f(np.maximum(rng.normal(0.05, 0.1, (4, 6912)), 0.0))
        W, b = head_init(1, 146)
        y = np.asarray([1, 2, 3, 4])
        with torch.no_grad():
            r = eng.forward(x, record="trace")
        h = eng.tt.to_np(r["h"]).astype(np.float64)
        _z, p, _lp = head_forward(W, b, h)
        dz = onehot(y) - p
        eng.route_back(r, eng.tt.f(head_route_trace(W, h, dz)))
        td = eng.trace_deltas(0.001, 4)
        out = eng.bp_grads(x, W, b, y)
        pad = 0.0
        for g, pt in zip(out["grads"] + td, eng.parts + eng.parts):
            m = ~pt.valid_np
            if m.any():
                pad = max(pad, float(g.abs()[torch.from_numpy(m)].max()))
        rec("실제 구조: TRACE·BP 계산, 채움·비활성 칸 0, BP forward = forward", pad == 0.0 and np.array_equal(out["h"], h) and eng.n_learnable == 31142,
            f"유효 연결 {eng.n_learnable:,}")
    guard("실제 구조", t_real)

    n_ok = sum(1 for r in results if r["ok"])
    write_json(tmp / "selftest_report.json", {"program": PROGRAM, "version": VERSION, "utc": utc_now(), "results": results,
                                              "note_ko": "합성 입력만 쓴 문법·수학 검사다. 정규 실험 결과가 아니다."})
    say(f"\n[self-test] {n_ok}/{len(results)} 통과 · 보고서 {tmp / 'selftest_report.json'}")
    return 0 if n_ok == len(results) else 1


def head_init(seed: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """자체 검사용 작은 head (v12 head_init 과 같은 식)."""
    rng = np.random.default_rng([int(seed), 9090, int(dim)])
    return (rng.standard_normal((N_CLASSES, int(dim))) * (0.01 / math.sqrt(int(dim)))).astype(np.float64), np.zeros(N_CLASSES, np.float64)


if __name__ == "__main__":
    sys.exit(main())
