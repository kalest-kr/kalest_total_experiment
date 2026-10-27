#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparse_backprop_experiment.py
=============================

입력 크기 기반 희소(근사) 역전파 비교 실험 - MNIST, MLP 784 -> 256 -> 128 -> 10.

모든 조건에서 순전파는 동일한 가중합 + ReLU 이다. 역전파 때만 각 수신 노드(출력 노드)마다
순전파에서 기록한 입력 크기 기준 상위 k개 입력 연결에만 기울기를 전달한다. 가중합 기울기의
일부를 의도적으로 생략하므로 이 방식은 '근사 역전파'이며, 성능 개선을 가정하지 않는다.

검증 상태: 코드 작성 및 정적 검토 완료, 실행 검증 미실시.
(작성자는 데이터 다운로드, 학습, 평가, 벤치마크, 자체 검증을 실행하지 않았다.)

사용 예:
    python sparse_backprop_experiment.py --run-all
    python sparse_backprop_experiment.py --resume runs_sparse_backprop/run_YYYYMMDD_HHMMSS
    python sparse_backprop_experiment.py --self-test-only

근사 역전파 수식 (배치 입력 X[b,i], 가중치 W[j,i], 역전파로 도착한 값 Delta[b,j]):
    순전파      Z[b,j]  = sum_i W[j,i] X[b,i] + bias[j]
    선택 점수   input        : s[b,j,i] = |X[b,i]|
                contribution : s[b,j,i] = |W[j,i] X[b,i]|
    선택 마스크 M[b,j,i] = 1[i 가 (b,j) 의 상위 k 입력]   (동률은 작은 인덱스 우선)
    근사 dW[j,i] = sum_b M[b,j,i] Delta[b,j] X[b,i]
    근사 dX[b,i] = sum_j M[b,j,i] Delta[b,j] W[j,i]
    dbias[j]     = sum_b Delta[b,j]
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gc
import hashlib
import io
import json
import math
import os
import platform
import random
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # CPU 메모리(프로세스 RSS) 측정용. 없으면 해당 항목만 비워 둔다.
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


# ---------------------------------------------------------------------------
# 상수와 기본값
# ---------------------------------------------------------------------------

SCRIPT_VERSION = "1.0.0"
CHECKPOINT_FORMAT = 1
DIMS: Tuple[int, ...] = (784, 256, 128, 10)
MB = 1024.0 * 1024.0

# 조건 이름 -> (선택 방식, k). 선택 방식 None 은 표준 nn.Linear 역전파(dense 기준군).
CONDITIONS: Dict[str, Tuple[Optional[str], int]] = {
    "dense": (None, 0),
    "input_top1": ("input", 1),
    "input_top2": ("input", 2),
    "contribution_top1": ("contribution", 1),
    "contribution_top2": ("contribution", 2),
}
CONDITION_ORDER: List[str] = list(CONDITIONS)
CONDITION_DESC_KO = {
    "dense": "일반 역전파 (표준 nn.Linear)",
    "input_top1": "원입력 |x_i| 상위 1개 연결만 역전파 (근사 역전파)",
    "input_top2": "원입력 |x_i| 상위 2개 연결만 역전파 (근사 역전파)",
    "contribution_top1": "가중 기여량 |W_ji x_i| 상위 1개 연결만 역전파 (근사 역전파)",
    "contribution_top2": "가중 기여량 |W_ji x_i| 상위 2개 연결만 역전파 (근사 역전파)",
}

# 에폭별 배치 순서 난수의 고정 상수. (시드, 에폭)만으로 순서가 정해지므로
# 같은 시드의 모든 조건이 같은 배치 순서를 쓰고, 재개 시에도 순서가 재현된다.
EPOCH_ORDER_SALT = 20_240_917
RSS_SAMPLE_EVERY = 25          # CPU 메모리(RSS) 표본 간격(배치)
SMALL_K_ITERATIVE = 8          # 이 이하의 k 는 argmax 반복, 초과하면 안정 정렬로 선택
PHASES = ("data", "forward_incl_selection", "backward", "optimizer", "selection_stats")

DEFAULTS: Dict[str, Any] = {
    "epochs": 10,
    "batch_size": 128,
    "lr": 0.05,
    "momentum": 0.0,
    "weight_decay": 0.0,
    "seeds": [0, 1, 2],
    "split_seed": 2024,
    "device": "auto",
    "data_dir": "data",
    "output_dir": "runs_sparse_backprop",
    "sparse_layers": "all",
    "selection_chunk_size": 64,
    "target_accuracy": [],
    "conditions": list(CONDITION_ORDER),
    "eval_batch_size": 1000,
    "limit_train_batches": 0,
    "profile_warmup": 20,
    "profile_batches": 50,
    "selection_stats": "on",
    "deterministic": False,
    "cpu_threads": 0,
}

# 재개 시 달라지면 오류로 처리하는 설정(실험 결과나 시간 측정에 영향을 주는 항목).
DEFINING_KEYS: Tuple[str, ...] = (
    "epochs", "batch_size", "lr", "momentum", "weight_decay", "seeds", "split_seed",
    "sparse_layers", "selection_chunk_size", "conditions", "eval_batch_size",
    "limit_train_batches", "profile_warmup", "profile_batches", "selection_stats",
    "deterministic", "cpu_threads",
)

# 그래프 색: 범주형 고정 순서(조건 이름에 고정, 순위나 부분집합에 따라 바뀌지 않음).
PLOT_COLORS = {
    "dense": "#2a78d6",
    "input_top1": "#eb6834",
    "input_top2": "#1baf7a",
    "contribution_top1": "#eda100",
    "contribution_top2": "#e87ba4",
}
PLOT_MARKERS = {"dense": "o", "input_top1": "s", "input_top2": "^",
                "contribution_top1": "D", "contribution_top2": "v"}
INK = {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
       "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
PHASE_COLORS = {"data": "#b7d3f6", "forward_incl_selection": "#5598e7", "backward": "#1c5cab",
                "optimizer": "#0d366b", "selection_stats": "#c3c2b7"}


class SettingsMismatchError(RuntimeError):
    """재개 시 저장된 설정과 현재 설정이 다를 때."""


class SelfTestFailure(AssertionError):
    """자체 검증 실패."""


# ---------------------------------------------------------------------------
# 공용 도구
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


class _LogState:
    file: Optional[Path] = None


def log(msg: str = "") -> None:
    print(msg, flush=True)
    if _LogState.file is not None:
        try:
            with open(_LogState.file, "a", encoding="utf-8") as f:
                f.write(f"[{now_iso()}] {msg}\n")
        except OSError:
            pass


def configure_console() -> None:
    # Windows 콘솔 인코딩(cp949 등)에서 출력할 수 없는 문자가 있어도 중단되지 않게 한다.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path, torch.device, torch.dtype)):
        return str(o)
    raise TypeError(f"JSON 으로 저장할 수 없는 값: {type(o)}")


def _replace_with_retry(tmp: Path, path: Path) -> None:
    # Windows 에서는 백신/색인 프로그램이 파일을 잠깐 잡고 있으면 os.replace 가 실패할 수 있다.
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.25 * (attempt + 1))


def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp{os.getpid()}")


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with open(tmp, "w", encoding=encoding, newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    _replace_with_retry(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default))


def atomic_torch_save(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    _replace_with_retry(tmp, path)


def _csv_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "" if not math.isfinite(v) else format(v, ".8g")
    return str(v)


def atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: _csv_cell(r.get(k)) for k in fieldnames})
    # utf-8-sig: Windows Excel 에서 한글이 깨지지 않게 BOM 을 붙인다.
    atomic_write_text(path, buf.getvalue(), encoding="utf-8-sig")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def torch_load(path: Path) -> Any:
    # 이 프로그램이 직접 만든 파일만 읽는다(난수 상태 등 텐서 이외 객체 포함).
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # weights_only 인자가 없는 오래된 PyTorch
        return torch.load(path, map_location="cpu")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_dict_sha256(sd: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(sd):
        t = sd[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def capture_rng_state(device: torch.device) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda")
    if device.type == "cuda" and cuda_states is not None and len(cuda_states) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda_states)


_PROC = None


def _rss_bytes() -> Optional[int]:
    """프로세스 상주 메모리(RSS, Windows 는 Working Set). Python 객체 크기가 아니다."""
    global _PROC
    if psutil is None:
        return None
    try:
        if _PROC is None:
            _PROC = psutil.Process(os.getpid())
        return int(_PROC.memory_info().rss)
    except Exception:
        return None


def _stats_ms(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def run_key(seed: int, condition: str) -> str:
    return f"seed_{seed}/{condition}"


def cond_dir(run_dir: Path, seed: int, condition: str) -> Path:
    return run_dir / f"seed_{seed}" / condition


# ---------------------------------------------------------------------------
# 연결 선택 (상위 k, 동률은 작은 인덱스 우선)
# ---------------------------------------------------------------------------

def stable_topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """마지막 차원에서 점수가 큰 서로 다른 인덱스 k개를 고른다 (int64).

    - 동률이면 작은 인덱스가 먼저 선택된다. 0 점수도 같은 규칙을 따른다.
    - k 가 차원보다 크면 차원 크기까지만 선택한다.
    - 반환 순서: 점수 내림차순, 동률은 인덱스 오름차순.
    - k <= SMALL_K_ITERATIVE 이면 argmax 를 반복하며 scores 를 제자리에서 수정한다.
      (호출자는 새로 만든 점수 텐서를 넘긴다.) torch.argmax 는 최대값이 여러 개이면
      첫 번째 인덱스를 돌려준다(공식 문서). 이 동작은 자체 검증에서 장치별로 확인한다.
    """
    n = scores.shape[-1]
    k_eff = min(int(k), n)
    lead = tuple(scores.shape[:-1])
    if k_eff <= 0:
        raise ValueError("k 는 1 이상이어야 합니다.")
    if k_eff == n:
        return torch.arange(n, device=scores.device).expand(*lead, n).contiguous()
    if k_eff == 1:
        return scores.argmax(dim=-1, keepdim=True)
    if k_eff <= SMALL_K_ITERATIVE:
        out = torch.empty(*lead, k_eff, dtype=torch.long, device=scores.device)
        for t in range(k_eff):
            best = scores.argmax(dim=-1, keepdim=True)
            out[..., t] = best.squeeze(-1)
            if t + 1 < k_eff:
                # 점수는 절댓값(>= 0)이므로 -inf 로 지우면 다음 선택에서 제외된다(0 점수는 남는다).
                scores.scatter_(-1, best, float("-inf"))
        return out
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices
    return order[..., :k_eff].contiguous()


def compute_selection(x: torch.Tensor, weight: torch.Tensor, mode: str, k: int,
                      chunk_size: int) -> torch.Tensor:
    """현재 순전파의 입력 x[B,in] 과 가중치 weight[out,in] 으로 선택 인덱스를 구한다.

    반환 모양
      input        : [B, k_eff]        같은 표본의 모든 수신 노드가 같은 인덱스를 공유한다(복제하지 않음).
      contribution : [out, B, k_eff]   수신 노드마다 다른 인덱스.
    contribution 은 출력 노드를 chunk_size 개씩 나눠 [청크, B, in] 점수만 잠시 만든다.
    [B, out, in] 전체 점수 텐서는 보관하지 않는다.
    """
    if x.dim() != 2:
        raise ValueError(f"입력은 [배치, 입력] 2차원이어야 합니다: {tuple(x.shape)}")
    batch, in_f = x.shape
    out_f = weight.shape[0]
    k_eff = min(int(k), in_f)
    if k_eff < 1:
        raise ValueError("k 는 1 이상이어야 합니다.")
    if mode == "input":
        if k_eff == in_f:
            return torch.arange(in_f, device=x.device).expand(batch, in_f).contiguous()
        return stable_topk_indices(x.abs(), k_eff)
    if mode == "contribution":
        idx = torch.empty((out_f, batch, k_eff), dtype=torch.long, device=x.device)
        if k_eff == in_f:
            idx.copy_(torch.arange(in_f, device=x.device).expand(out_f, batch, in_f))
            return idx
        step = max(1, int(chunk_size))
        x_row = x.unsqueeze(0)                                   # [1, B, in]
        for j0 in range(0, out_f, step):
            j1 = min(out_f, j0 + step)
            scores = weight[j0:j1].unsqueeze(1) * x_row          # [c, B, in] 부호 있는 기여량
            scores.abs_()                                        # 선택 점수만 절댓값
            idx[j0:j1] = stable_topk_indices(scores, k_eff)
            del scores
        return idx
    raise ValueError(f"알 수 없는 선택 방식: {mode}")


# ---------------------------------------------------------------------------
# 근사 역전파 선형층
# ---------------------------------------------------------------------------

class ApproxBackpropLinearFunction(torch.autograd.Function):
    """순전파는 F.linear 그대로, 역전파는 선택된 항만 계산해 scatter_add_ 로 누적한다.

    전체 밀집 dW/dX 를 계산한 뒤 마스크를 곱하지 않는다. 저장하는 값은
    가중치(이미 존재하는 파라미터), 선택 인덱스, 선택된 입력값뿐이다.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, mode, k, chunk_size, sink):  # type: ignore[override]
        out = F.linear(x, weight, bias)
        # 자동미분이 꺼진 상태(autograd.Function.forward 내부)에서 선택 인덱스를 구한다.
        idx = compute_selection(x, weight, mode, k, chunk_size)
        if mode == "input":
            x_sel = x.gather(1, idx)                                         # [B, k]
        else:
            out_f = weight.shape[0]
            x_sel = x.unsqueeze(0).expand(out_f, -1, -1).gather(2, idx)      # [out, B, k]
        ctx.mode = mode
        ctx.in_features = x.shape[1]
        ctx.save_for_backward(weight, idx, x_sel)
        if sink is not None:
            sink.last_selection = idx   # 선택 빈도 통계용 참조(복사 없음)
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        weight, idx, x_sel = ctx.saved_tensors
        need_x, need_w, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        grad_x = grad_w = grad_b = None
        if grad_out is None:
            return None, None, None, None, None, None, None
        batch = grad_out.shape[0]
        out_f = weight.shape[0]
        in_f = ctx.in_features
        delta_t = grad_out.t().unsqueeze(2)                      # [out, B, 1]  Delta[b,j]
        if ctx.mode == "input":
            k = idx.shape[1]
            if need_w:
                # dW[j, idx[b,t]] += Delta[b,j] * X[b, idx[b,t]]
                src = delta_t * x_sel.unsqueeze(0)               # [out, B, k]
                index = idx.reshape(1, batch * k).expand(out_f, batch * k)   # 복제 없는 view
                grad_w = torch.zeros_like(weight).scatter_add_(1, index, src.reshape(out_f, batch * k))
            if need_x:
                # dX[b, idx[b,t]] += sum_j Delta[b,j] * W[j, idx[b,t]]
                w_sel = weight.index_select(1, idx.reshape(-1)).reshape(out_f, batch, k)
                vals = (w_sel * delta_t).sum(dim=0)              # [B, k]
                grad_x = torch.zeros((batch, in_f), dtype=x_sel.dtype, device=x_sel.device)
                grad_x.scatter_add_(1, idx, vals)
        else:
            k = idx.shape[2]
            flat_idx = idx.reshape(out_f, batch * k)
            if need_w:
                # dW[j, idx[j,b,t]] += Delta[b,j] * X[b, idx[j,b,t]]
                src = delta_t * x_sel                            # [out, B, k]
                grad_w = torch.zeros_like(weight).scatter_add_(1, flat_idx, src.reshape(out_f, batch * k))
            if need_x:
                # dX[b, idx[j,b,t]] += Delta[b,j] * W[j, idx[j,b,t]]  (여러 j 가 같은 i 로 오면 누적)
                w_sel = weight.gather(1, flat_idx).reshape(out_f, batch, k)
                src = delta_t * w_sel                            # [out, B, k]
                grad_x = torch.zeros((batch, in_f), dtype=x_sel.dtype, device=x_sel.device)
                grad_x.scatter_add_(1, idx.permute(1, 0, 2).reshape(batch, out_f * k),
                                    src.permute(1, 0, 2).reshape(batch, out_f * k))
        if need_b:
            grad_b = grad_out.sum(dim=0)                         # 편향은 일반 방식
        return grad_x, grad_w, grad_b, None, None, None, None


class ApproxBackpropLinear(nn.Linear):
    """nn.Linear 와 같은 파라미터/초기화/순전파. 학습 중 역전파만 근사한다.

    평가(eval) 또는 no_grad 에서는 선택 인덱스 계산을 생략하고 일반 F.linear 만 수행한다.
    """

    def __init__(self, in_features: int, out_features: int, mode: str, k: int,
                 chunk_size: int = 64, track_selection: bool = True, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        if mode not in ("input", "contribution"):
            raise ValueError(f"알 수 없는 선택 방식: {mode}")
        if int(k) < 1:
            raise ValueError("k 는 1 이상이어야 합니다.")
        self.selection_mode = mode
        self.k = int(k)
        self.chunk_size = int(chunk_size)
        self.track_selection = bool(track_selection)
        self.last_selection: Optional[torch.Tensor] = None
        self.selection_calls = 0
        # 선택 빈도 누적: input -> [in] (모든 수신 노드 공유), contribution -> [out*in]
        self.selection_counts: Optional[torch.Tensor] = None
        self._row_offsets: Optional[torch.Tensor] = None

    @property
    def k_eff(self) -> int:
        return min(self.k, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and torch.is_grad_enabled():
            self.selection_calls += 1
            sink = self if self.track_selection else None
            return ApproxBackpropLinearFunction.apply(
                x, self.weight, self.bias, self.selection_mode, self.k, self.chunk_size, sink)
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def accumulate_selection_stats(self) -> None:
        idx = self.last_selection
        self.last_selection = None
        if idx is None:
            return
        if self.selection_mode == "input":
            n_bins = self.in_features
            flat = idx.reshape(-1)
        else:
            n_bins = self.out_features * self.in_features
            if self._row_offsets is None or self._row_offsets.device != idx.device:
                self._row_offsets = (torch.arange(self.out_features, device=idx.device)
                                     * self.in_features).view(-1, 1, 1)
            flat = (idx + self._row_offsets).reshape(-1)
        counts = torch.bincount(flat, minlength=n_bins)
        if self.selection_counts is None:
            self.selection_counts = torch.zeros(n_bins, dtype=torch.long, device=counts.device)
        elif self.selection_counts.device != counts.device:
            self.selection_counts = self.selection_counts.to(counts.device)
        self.selection_counts += counts

    def connection_counts(self) -> torch.Tensor:
        """연결 (j, i) 가 선택된 횟수 [out, in] (CPU int64)."""
        shape = (self.out_features, self.in_features)
        if self.selection_counts is None:
            return torch.zeros(shape, dtype=torch.long)
        c = self.selection_counts.detach().cpu()
        if self.selection_mode == "input":
            return c.view(1, self.in_features).expand(*shape).clone()
        return c.view(*shape).clone()

    def extra_repr(self) -> str:
        return (f"{super().extra_repr()}, selection={self.selection_mode}, k={self.k}, "
                f"chunk_size={self.chunk_size}")


class ApproxBackpropMLP(nn.Module):
    """784 -> 256 -> 128 -> 10 MLP (은닉층 ReLU, 모든 선형층 편향 사용).

    mode=None 이면 모든 층이 표준 nn.Linear(dense 기준군).
    sparse_layers='all' 이면 최종 분류층 포함 모든 선형층, 'hidden' 이면 은닉 선형층만 근사.
    파라미터 이름은 조건과 무관하게 같아서 동일한 초기 가중치를 그대로 불러올 수 있다.
    """

    def __init__(self, dims: Sequence[int], mode: Optional[str], k: int, sparse_layers: str = "all",
                 chunk_size: int = 64, track_selection: bool = True):
        super().__init__()
        if sparse_layers not in ("all", "hidden"):
            raise ValueError(f"sparse_layers 는 all 또는 hidden: {sparse_layers}")
        self.dims = tuple(int(d) for d in dims)
        self.mode = mode
        self.k = int(k)
        self.sparse_layers = sparse_layers
        n_layers = len(self.dims) - 1
        layers: List[nn.Module] = []
        for li in range(n_layers):
            approx = mode is not None and (sparse_layers == "all" or li < n_layers - 1)
            if approx:
                layers.append(ApproxBackpropLinear(self.dims[li], self.dims[li + 1], mode, k,
                                                   chunk_size, track_selection))
            else:
                layers.append(nn.Linear(self.dims[li], self.dims[li + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = len(self.layers) - 1
        for li, layer in enumerate(self.layers):
            x = layer(x)
            if li < last:
                x = F.relu(x)
        return x

    def approx_layers(self) -> List[Tuple[int, ApproxBackpropLinear]]:
        return [(li, layer) for li, layer in enumerate(self.layers) if isinstance(layer, ApproxBackpropLinear)]

    @torch.no_grad()
    def layer_inputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """각 선형층에 들어가는 입력(순전파와 같은 연산)."""
        inputs = []
        h = x
        last = len(self.layers) - 1
        for li, layer in enumerate(self.layers):
            inputs.append(h)
            h = F.linear(h, layer.weight, layer.bias)
            if li < last:
                h = F.relu(h)
        return inputs

    @torch.no_grad()
    def run_selection_only(self, inputs: Sequence[torch.Tensor]) -> None:
        for li, layer in self.approx_layers():
            compute_selection(inputs[li], layer.weight, layer.selection_mode, layer.k, layer.chunk_size)

    def accumulate_selection_stats(self) -> None:
        for _, layer in self.approx_layers():
            layer.accumulate_selection_stats()

    def selection_counts_state(self) -> Dict[str, torch.Tensor]:
        return {str(li): layer.selection_counts.detach().cpu().clone()
                for li, layer in self.approx_layers() if layer.selection_counts is not None}

    def load_selection_counts(self, state: Dict[str, torch.Tensor], device: torch.device) -> None:
        for li, layer in self.approx_layers():
            t = state.get(str(li))
            layer.selection_counts = None if t is None else t.to(device=device, dtype=torch.long).clone()

    def selection_memory_estimate(self, batch_size: int) -> Dict[str, Any]:
        """선택 관련 추가 메모리의 해석적 추정(실측 아님)."""
        per_layer = []
        saved_total = 0
        temp_max = 0
        for li, layer in self.approx_layers():
            el = layer.weight.element_size()
            k = layer.k_eff
            if layer.selection_mode == "input":
                saved = batch_size * k * (8 + el)                     # 인덱스(int64) + 선택 입력값
                temp = batch_size * layer.in_features * el            # |x| 점수
            else:
                c = min(layer.chunk_size, layer.out_features)
                saved = layer.out_features * batch_size * k * (8 + el)
                temp = c * batch_size * layer.in_features * el        # 청크 점수 [c, B, in]
            per_layer.append({"layer": li, "saved_bytes": saved, "selection_temp_bytes": temp})
            saved_total += saved
            temp_max = max(temp_max, temp)
        return {"saved_index_and_value_bytes": saved_total, "max_selection_temp_bytes": temp_max,
                "per_layer": per_layer, "note": "해석적 추정. 실측 메모리는 cuda_* / saved_tensor_bytes 항목을 볼 것."}


def build_model(condition: str, settings: Dict[str, Any]) -> ApproxBackpropMLP:
    mode, k = CONDITIONS[condition]
    return ApproxBackpropMLP(DIMS, mode, k, settings["sparse_layers"], settings["selection_chunk_size"],
                             track_selection=(settings["selection_stats"] == "on"))


def make_optimizer(model: nn.Module, settings: Dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=settings["lr"], momentum=settings["momentum"],
                           weight_decay=settings["weight_decay"])


def backward_term_counts(model: ApproxBackpropMLP) -> Dict[str, Any]:
    """표본 1개당 역전파 곱셈 항 수(이론). 실측 속도와 구분해서 보고한다."""
    per_layer = []
    dense_total = approx_total = score_total = 0
    for li, layer in enumerate(model.layers):
        in_f, out_f = layer.in_features, layer.out_features
        n_mats = 1 if li == 0 else 2      # 첫 층은 입력 기울기(dX)가 필요 없다.
        dense = out_f * in_f * n_mats
        if isinstance(layer, ApproxBackpropLinear):
            approx = out_f * layer.k_eff * n_mats
            score = in_f if layer.selection_mode == "input" else out_f * in_f
        else:
            approx = dense
            score = 0
        per_layer.append({"layer": li, "in": in_f, "out": out_f, "approx": isinstance(layer, ApproxBackpropLinear),
                          "dense_terms": dense, "approx_terms": approx, "selection_score_elements": score})
        dense_total += dense
        approx_total += approx
        score_total += score
    return {
        "unit": "표본 1개당 곱셈 항 수 (dW + 필요한 dX, 편향 제외)",
        "dense_terms_per_sample": dense_total,
        "approx_terms_per_sample": approx_total,
        "terms_ratio": approx_total / dense_total,
        "selection_score_elements_per_sample": score_total,
        "per_layer": per_layer,
        "note": "선택한 항의 수 감소(이론)이며 실측 속도 향상이 아니다. 선택 점수 계산 비용은 별도 항목.",
    }


# ---------------------------------------------------------------------------
# 자체 검증 (사용자 컴퓨터에서 실행. 작성자는 실행하지 않았다.)
# ---------------------------------------------------------------------------

class _Sink:
    last_selection: Optional[torch.Tensor] = None


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise SelfTestFailure(msg)


def _st_tol(dtype: torch.dtype) -> Tuple[float, float]:
    return (1e-10, 1e-12) if dtype == torch.float64 else (1e-4, 1e-5)


def _assert_close(name: str, actual: Optional[torch.Tensor], expected: Optional[torch.Tensor],
                  dtype: torch.dtype) -> None:
    if actual is None or expected is None:
        _check(actual is None and expected is None, f"{name}: 한쪽만 None (구현={actual is None}, 기준={expected is None})")
        return
    a = actual.detach().to(torch.float64).cpu()
    e = expected.detach().to(torch.float64).cpu()
    _check(tuple(a.shape) == tuple(e.shape), f"{name}: 모양 불일치 구현 {tuple(a.shape)} / 기준 {tuple(e.shape)}")
    rtol, atol = _st_tol(dtype)
    if not torch.allclose(a, e, rtol=rtol, atol=atol):
        diff = (a - e).abs()
        pos = tuple(int(p) for p in np.unravel_index(int(diff.argmax()), tuple(diff.shape)))
        raise SelfTestFailure(f"{name}: 최대 오차 {diff.max().item():.3e}, 위치 {pos}, "
                              f"구현 {a[pos].item():.10g} / 기준 {e[pos].item():.10g}")


def _same_forward(a: torch.Tensor, b: torch.Tensor, device: torch.device, dtype: torch.dtype) -> Tuple[bool, bool]:
    """(통과 여부, 비트 단위 일치 여부). CPU 는 비트 단위 일치를 요구한다."""
    bitwise = torch.equal(a, b)
    if device.type == "cpu":
        return bitwise, bitwise
    tol = 1e-12 if dtype == torch.float64 else 1e-6
    return bool(torch.allclose(a, b, rtol=tol, atol=tol)), bitwise


def _q(shape: Sequence[int], gen: torch.Generator, device: torch.device, dtype: torch.dtype,
       lo: int = -4, hi: int = 4, scale: float = 0.5) -> torch.Tensor:
    """양자화된 값(동률이 자주 생기고 float32 에서도 곱이 정확함)."""
    t = torch.randint(lo, hi + 1, tuple(shape), generator=gen).to(torch.float64) * scale
    return t.to(device=device, dtype=dtype)


def _ref_topk(scores: Sequence[float], k: int) -> List[int]:
    """파이썬 기준 구현: 점수 내림차순, 동률은 작은 인덱스 우선."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:min(k, len(scores))]


def _reference_mask(X: torch.Tensor, W: torch.Tensor, mode: str, k: int) -> torch.Tensor:
    """전체 마스크 M[b,j,i] (검증용 작은 입력에서만 사용)."""
    xl = X.detach().to(torch.float64).cpu().tolist()
    wl = W.detach().to(torch.float64).cpu().tolist()
    batch, out_f, in_f = len(xl), len(wl), len(xl[0])
    m = torch.zeros(batch, out_f, in_f, dtype=torch.float64)
    for b in range(batch):
        for j in range(out_f):
            if mode == "input":
                s = [abs(v) for v in xl[b]]
            else:
                s = [abs(wl[j][i] * xl[b][i]) for i in range(in_f)]
            for i in _ref_topk(s, k):
                m[b, j, i] = 1.0
    return m


def _mask_from_indices(idx: torch.Tensor, mode: str, batch: int, out_f: int, in_f: int) -> torch.Tensor:
    il = idx.detach().cpu().tolist()
    m = torch.zeros(batch, out_f, in_f, dtype=torch.float64)
    for b in range(batch):
        for j in range(out_f):
            row = il[b] if mode == "input" else il[j][b]
            for i in row:
                m[b, j, i] = 1.0
    return m


def _reference_grads(M: torch.Tensor, X: torch.Tensor, W: torch.Tensor, G: torch.Tensor):
    x = X.detach().to(torch.float64).cpu()
    w = W.detach().to(torch.float64).cpu()
    g = G.detach().to(torch.float64).cpu()
    dW = torch.einsum("bji,bj,bi->ji", M, g, x)
    dX = torch.einsum("bji,bj,ji->bi", M, g, w)
    db = g.sum(dim=0)
    return dW, dX, db


def _run_function(X, W, b, G, mode, k, chunk, x_requires_grad):
    xv = X.detach().clone().requires_grad_(x_requires_grad)
    wv = W.detach().clone().requires_grad_(True)
    bv = b.detach().clone().requires_grad_(True)
    sink = _Sink()
    out = ApproxBackpropLinearFunction.apply(xv, wv, bv, mode, k, chunk, sink)
    (out * G).sum().backward()
    return out.detach(), (xv.grad if x_requires_grad else None), wv.grad, bv.grad, sink.last_selection


def _clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _quantized_state(dims: Sequence[int], gen: torch.Generator) -> Dict[str, torch.Tensor]:
    base = ApproxBackpropMLP(dims, None, 0)
    with torch.no_grad():
        for p in base.parameters():
            p.copy_(_q(p.shape, gen, torch.device("cpu"), torch.float64, -4, 4, 0.25).to(p.dtype))
    return _clone_state(base)


def st_forward_equivalence(device: torch.device, dtype: torch.dtype) -> str:
    gen = torch.Generator().manual_seed(11)
    nonbitwise = 0
    for dims in ((9, 7, 5, 3), DIMS):
        torch.manual_seed(5)
        base = ApproxBackpropMLP(dims, None, 0)
        state = _clone_state(base)
        x = torch.rand(6, dims[0], generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
        x[0].zero_()                                               # 0 입력 표본 포함
        ref_model = base.to(device=device, dtype=dtype).train()
        ref = ref_model(x)
        ref_model.eval()
        with torch.no_grad():
            ref_eval = ref_model(x)
        for cond, (mode, k) in CONDITIONS.items():
            for scope in ("all", "hidden"):
                m = ApproxBackpropMLP(dims, mode, k, scope, chunk_size=3).to(device=device, dtype=dtype)
                m.load_state_dict(state)
                m.train()
                out = m(x)
                ok, bit = _same_forward(out, ref, device, dtype)
                _check(ok, f"{cond}/{scope} dims={dims}: 학습 모드 순전파 출력이 dense 와 다름 "
                           f"(최대 차 {(out - ref).abs().max().item():.3e})")
                nonbitwise += 0 if bit else 1
                if mode is not None:
                    calls = sum(layer.selection_calls for _, layer in m.approx_layers())
                    _check(calls == len(m.approx_layers()), f"{cond}/{scope}: 학습 순전파에서 근사 경로가 쓰이지 않음")
                m.eval()
                with torch.no_grad():
                    out_eval = m(x)
                ok, bit = _same_forward(out_eval, ref_eval, device, dtype)
                _check(ok, f"{cond}/{scope} dims={dims}: 평가 모드 순전파 출력이 dense 와 다름")
                nonbitwise += 0 if bit else 1
    return "비트 단위 일치" if nonbitwise == 0 else f"허용오차 내 일치 (비트 단위 불일치 {nonbitwise}건, CUDA)"


def st_selection_fixed_cases(device: torch.device, dtype: torch.dtype) -> str:
    def sel(x_rows, w_rows, mode, k, chunk=2):
        X = torch.tensor(x_rows, dtype=dtype, device=device)
        W = torch.tensor(w_rows, dtype=dtype, device=device)
        idx = compute_selection(X, W, mode, k, chunk)
        _check(idx.dtype == torch.long, f"선택 인덱스 dtype 이 int64 가 아님: {idx.dtype}")
        return idx.cpu().tolist()

    # (1) input: 동률, 0 입력, 음수 입력, top-1/2/3, k > 입력 차원
    X1 = [[0.5, -0.5, 0.0, 0.25, -0.75, 0.75],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [-1.0, 0.0, 1.0, 0.5, 0.0, -1.0]]
    W1 = [[1.0] * 6, [-2.0] * 6]
    expect_in = {1: [[4], [0], [0]], 2: [[4, 5], [0, 1], [0, 2]], 3: [[0, 4, 5], [0, 1, 2], [0, 2, 5]],
                 8: [[0, 1, 2, 3, 4, 5]] * 3}
    for k, exp in expect_in.items():
        got = sel(X1, W1, "input", k)
        _check(len(got) == 3 and all(len(r) == min(k, 6) for r in got),
               f"input top-{k}: 인덱스 모양이 [B, k_eff] 가 아님: {got}")
        _check(all(len(set(r)) == len(r) for r in got), f"input top-{k}: 중복 인덱스: {got}")
        _check([sorted(r) for r in got] == exp, f"input top-{k}: 기대 {exp}, 구현 {[sorted(r) for r in got]}")

    # (2) contribution: 음수 가중치, 동률, 0 가중치 / 같은 X 에서 input 과 다른 선택
    X2 = [[1.0, 2.0, 3.0, 0.0]]
    W2 = [[-5.0, 1.0, 0.1, 9.0], [0.5, -1.0, 1.0, 2.0], [0.0, 0.0, 0.0, 0.0], [3.0, 1.5, 1.0, -7.0]]
    expect_c = {1: [[0], [2], [0], [0]], 2: [[0, 1], [1, 2], [0, 1], [0, 1]], 5: [[0, 1, 2, 3]] * 4}
    for k, exp in expect_c.items():
        got = sel(X2, W2, "contribution", k)
        _check(len(got) == 4 and all(len(node) == 1 and len(node[0]) == min(k, 4) for node in got),
               f"contribution top-{k}: 인덱스 모양이 [out, B, k_eff] 가 아님: {got}")
        got_sets = [sorted(node[0]) for node in got]
        _check(got_sets == exp, f"contribution top-{k}: 기대 {exp}, 구현 {got_sets}")
    got_in = sel(X2, W2, "input", 1)
    _check(got_in == [[2]], f"input top-1 (X={X2}): 기대 [[2]], 구현 {got_in}")
    got_in2 = sel(X2, W2, "input", 2)
    _check(sorted(got_in2[0]) == [1, 2], f"input top-2 (X={X2}): 기대 [1, 2], 구현 {got_in2}")
    _check(got_in[0][0] != sel(X2, W2, "contribution", 1)[0][0][0],
           "같은 X 에서 input(|x|) 과 contribution(|Wx|) 이 각자의 기준으로 다른 연결을 골라야 함")

    # (3) 음수 입력의 contribution
    X3 = [[-2.0, 1.0]]
    W3 = [[1.0, 1.0], [-1.0, 3.0]]
    got = sel(X3, W3, "contribution", 1)
    _check([node[0] for node in got] == [[0], [1]], f"음수 입력 contribution top-1: 기대 [[0],[1]], 구현 {got}")
    _check(sel(X3, W3, "input", 1) == [[0]], "음수 입력 input top-1: 기대 [[0]]")

    # (4) 실제 계산은 원래 부호 유지 (input top-1)
    X4 = torch.tensor([[-3.0, 1.0, 0.0]], dtype=dtype, device=device)
    W4 = torch.tensor([[2.0, -1.0, 4.0]], dtype=dtype, device=device)
    b4 = torch.tensor([0.5], dtype=dtype, device=device)
    G4 = torch.tensor([[1.0]], dtype=dtype, device=device)
    _, gx, gw, gb, _ = _run_function(X4, W4, b4, G4, "input", 1, 2, True)
    _assert_close("부호 유지 dW", gw, torch.tensor([[-3.0, 0.0, 0.0]]), dtype)
    _assert_close("부호 유지 dX", gx, torch.tensor([[2.0, 0.0, 0.0]]), dtype)
    _assert_close("부호 유지 dbias", gb, torch.tensor([1.0]), dtype)
    return "고정 사례 통과"


def st_selection_random_cases(device: torch.device, dtype: torch.dtype) -> str:
    gen = torch.Generator().manual_seed(1234)
    in_sizes = (3, 5, 7, 9, 12, 14)
    checked = 0
    for trial in range(18):
        in_f = in_sizes[trial % len(in_sizes)]
        batch, out_f = 5, 4
        X = _q((batch, in_f), gen, device, dtype, -2, 2, 0.5)      # {-1,-0.5,0,0.5,1} : 동률/0 다수
        W = _q((out_f, in_f), gen, device, dtype, -4, 4, 0.5)
        ks = {1, 2, 3, in_f, in_f + 1}
        if in_f > SMALL_K_ITERATIVE + 2:
            ks.add(SMALL_K_ITERATIVE + 2)                          # 안정 정렬 경로도 검사
        for mode in ("input", "contribution"):
            for k in sorted(ks):
                idx = compute_selection(X, W, mode, k, chunk_size=3)
                got = _mask_from_indices(idx, mode, batch, out_f, in_f)
                ref = _reference_mask(X, W, mode, k)
                if not torch.equal(got, ref):
                    bad = (got != ref).nonzero()[0].tolist()
                    raise SelfTestFailure(f"무작위 사례 trial={trial} {mode} k={k} in={in_f}: "
                                          f"선택 마스크 불일치 (첫 위치 b,j,i={bad})")
                counts = got.sum(dim=-1)
                _check(bool((counts == min(k, in_f)).all()),
                       f"trial={trial} {mode} k={k}: 행마다 서로 다른 인덱스 {min(k, in_f)}개가 아님")
                checked += 1
    return f"{checked}개 사례 일치"


def st_selection_chunk_invariance(device: torch.device, dtype: torch.dtype) -> str:
    gen = torch.Generator().manual_seed(99)
    X = _q((7, 13), gen, device, dtype, -2, 2, 0.5)
    W = _q((11, 13), gen, device, dtype, -4, 4, 0.5)
    for k in (1, 2, 3, 10):
        ref = compute_selection(X, W, "contribution", k, chunk_size=11)
        for c in (1, 2, 3, 4, 100):
            got = compute_selection(X, W, "contribution", k, chunk_size=c)
            _check(torch.equal(got, ref), f"contribution k={k}: 청크 크기 {c} 와 11 의 선택 결과가 다름")
    return "청크 크기와 무관"


def st_gradient_formula(device: torch.device, dtype: torch.dtype) -> str:
    gen = torch.Generator().manual_seed(77)
    n = 0
    for mode in ("input", "contribution"):
        for (batch, out_f, in_f) in ((4, 3, 6), (5, 4, 7)):
            for k in (1, 2, 3, in_f, in_f + 2):
                for x_grad in (True, False):
                    X = _q((batch, in_f), gen, device, dtype, -2, 2, 0.5)
                    X[0] = 0.0                                     # 0 입력 표본
                    W = _q((out_f, in_f), gen, device, dtype, -4, 4, 0.5)
                    b = _q((out_f,), gen, device, dtype, -4, 4, 0.5)
                    G = torch.randn(batch, out_f, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
                    out, gx, gw, gb, idx = _run_function(X, W, b, G, mode, k, 2, x_grad)
                    tag = f"{mode} k={k} B={batch} out={out_f} in={in_f} x_grad={x_grad}"
                    ok, _ = _same_forward(out, F.linear(X, W, b), device, dtype)
                    _check(ok, f"{tag}: 순전파가 F.linear 와 다름")
                    k_eff = min(k, in_f)
                    exp_shape = (batch, k_eff) if mode == "input" else (out_f, batch, k_eff)
                    _check(idx is not None and tuple(idx.shape) == exp_shape,
                           f"{tag}: 저장된 선택 인덱스 모양 {None if idx is None else tuple(idx.shape)} != {exp_shape}")
                    M = _reference_mask(X, W, mode, k)
                    dW_ref, dX_ref, db_ref = _reference_grads(M, X, W, G)
                    _assert_close(f"{tag} dW", gw, dW_ref, dtype)
                    _assert_close(f"{tag} dbias", gb, db_ref, dtype)
                    if x_grad:
                        _assert_close(f"{tag} dX", gx, dX_ref, dtype)
                    else:
                        _check(gx is None, f"{tag}: 입력이 기울기를 요구하지 않는데 dX 가 계산됨")
                    n += 1
    return f"{n}개 사례에서 근사 수식과 일치"


def st_gradient_accumulation(device: torch.device, dtype: torch.dtype) -> str:
    def t(rows):
        return torch.tensor(rows, dtype=dtype, device=device)

    # (a) 여러 표본이 같은 dW 위치로 도착 (input top-1: 두 표본 모두 i=0 선택)
    _, gx, gw, gb, _ = _run_function(t([[3.0, 1.0, 0.0], [2.0, 0.0, 1.0]]), t([[1.0, 1.0, 1.0]]), t([0.0]),
                                     t([[1.0], [1.0]]), "input", 1, 2, True)
    _assert_close("표본 간 dW 누적", gw, torch.tensor([[5.0, 0.0, 0.0]]), dtype)
    _assert_close("표본 간 dX", gx, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), dtype)
    _assert_close("표본 간 dbias", gb, torch.tensor([2.0]), dtype)

    # (b) 여러 수신 노드가 같은 dX 위치로 도착 (contribution top-1: 두 노드 모두 i=0 선택)
    Xb, Wb, Gb = t([[1.0, 1.0, 1.0]]), t([[2.0, -1.0, 0.0], [-3.0, 1.0, 0.0]]), t([[1.0, 2.0]])
    dX_exp = torch.tensor([[1.0 * 2.0 + 2.0 * (-3.0), 0.0, 0.0]])
    dW_exp = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    _, gx, gw, gb, _ = _run_function(Xb, Wb, t([0.0, 0.0]), Gb, "contribution", 1, 1, True)
    _assert_close("수신 노드 간 dX 누적", gx, dX_exp, dtype)
    _assert_close("수신 노드 간 dW", gw, dW_exp, dtype)
    _assert_close("수신 노드 간 dbias", gb, torch.tensor([1.0, 2.0]), dtype)

    # (c) 같은 파라미터로 두 번 역전파하면 .grad 에 누적
    layer = ApproxBackpropLinear(3, 2, "contribution", 1, 1).to(device=device, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(Wb)
        layer.bias.zero_()
    x = Xb.clone().requires_grad_(True)
    for _ in range(2):
        (layer(x) * Gb).sum().backward()
    _assert_close(".grad 2회 누적 dW", layer.weight.grad, 2 * dW_exp, dtype)
    _assert_close(".grad 2회 누적 dX", x.grad, 2 * dX_exp, dtype)
    _assert_close(".grad 2회 누적 dbias", layer.bias.grad, torch.tensor([2.0, 4.0]), dtype)
    return "누적 확인"


def st_batch_mean_once(device: torch.device, dtype: torch.dtype) -> str:
    gen = torch.Generator().manual_seed(21)
    dims = (6, 5, 4, 3)
    batch = 5
    state = _quantized_state(dims, gen)
    X = _q((batch, dims[0]), gen, device, dtype, 0, 4, 0.25)
    y = torch.randint(0, dims[-1], (batch,), generator=gen).to(device)
    # (a) 배치 평균 기울기 == 표본별(배치 1) 기울기의 평균
    for cond, (mode, k) in CONDITIONS.items():
        if mode is None:
            continue
        m = ApproxBackpropMLP(dims, mode, k, "all", 2).to(device=device, dtype=dtype)
        m.load_state_dict(state)
        m.train()
        F.cross_entropy(m(X), y).backward()
        g_batch = {name: p.grad.detach().clone() for name, p in m.named_parameters()}
        acc = {name: torch.zeros_like(p) for name, p in m.named_parameters()}
        for i in range(batch):
            m.zero_grad(set_to_none=True)
            F.cross_entropy(m(X[i:i + 1]), y[i:i + 1]).backward()
            for name, p in m.named_parameters():
                acc[name] += p.grad
        for name in g_batch:
            _assert_close(f"{cond} {name}: 배치 평균 vs 표본별 평균", g_batch[name], acc[name] / batch, dtype)
    # (b) 단일 근사층 + CrossEntropy(mean): Delta = (softmax - onehot) / B 로 직접 계산한 값과 비교
    for mode in ("input", "contribution"):
        for k in (1, 2):
            layer = ApproxBackpropLinear(dims[0], 3, mode, k, 2).to(device=device, dtype=dtype)
            with torch.no_grad():
                layer.weight.copy_(_q(layer.weight.shape, gen, device, dtype, -4, 4, 0.5))
                layer.bias.copy_(_q(layer.bias.shape, gen, device, dtype, -4, 4, 0.5))
            yy = torch.randint(0, 3, (batch,), generator=gen).to(device)
            z = layer(X)
            F.cross_entropy(z, yy).backward()
            zd = z.detach().to(torch.float64).cpu()
            delta = (torch.softmax(zd, dim=1) - F.one_hot(yy.cpu(), 3).to(torch.float64)) / batch
            M = _reference_mask(X, layer.weight, mode, k)
            dW_ref, _, db_ref = _reference_grads(M, X, layer.weight, delta)
            _assert_close(f"{mode} k={k}: CE 평균 1회 적용 dW", layer.weight.grad, dW_ref, dtype)
            _assert_close(f"{mode} k={k}: CE 평균 1회 적용 dbias", layer.bias.grad, db_ref, dtype)
    return "배치 평균 1회"


def st_full_k_matches_dense(device: torch.device, dtype: torch.dtype) -> str:
    gen = torch.Generator().manual_seed(31)
    dims = (6, 5, 4, 3)
    batch = 5
    torch.manual_seed(4)
    state = _clone_state(ApproxBackpropMLP(dims, None, 0))
    X = torch.randn(batch, dims[0], generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)
    X[0] = 0.0
    y = torch.randint(0, dims[-1], (batch,), generator=gen).to(device)
    dense = ApproxBackpropMLP(dims, None, 0).to(device=device, dtype=dtype)
    dense.load_state_dict(state)
    F.cross_entropy(dense(X), y).backward()
    ref = {name: p.grad.detach().clone() for name, p in dense.named_parameters()}
    max_in = max(dims[:-1])
    for mode in ("input", "contribution"):
        for scope in ("all", "hidden"):
            for k in (max_in, max_in + 3):            # k == 입력 차원, k > 입력 차원
                m = ApproxBackpropMLP(dims, mode, k, scope, 2).to(device=device, dtype=dtype)
                m.load_state_dict(state)
                F.cross_entropy(m(X), y).backward()
                for name, p in m.named_parameters():
                    _assert_close(f"{mode}/{scope}/k={k} {name}: 일반 역전파와 비교", p.grad, ref[name], dtype)
    return "k >= 입력 차원에서 일반 역전파와 일치"


def st_hidden_scope(device: torch.device, dtype: torch.dtype) -> str:
    gen = torch.Generator().manual_seed(41)
    dims = (6, 5, 4, 3)
    batch = 5
    state = _quantized_state(dims, gen)
    X = _q((batch, dims[0]), gen, device, dtype, 0, 4, 0.25)
    y = torch.randint(0, dims[-1], (batch,), generator=gen).to(device)
    dense = ApproxBackpropMLP(dims, None, 0).to(device=device, dtype=dtype)
    dense.load_state_dict(state)
    F.cross_entropy(dense(X), y).backward()
    for mode in ("input", "contribution"):
        m = ApproxBackpropMLP(dims, mode, 1, "hidden", 2).to(device=device, dtype=dtype)
        m.load_state_dict(state)
        _check(type(m.layers[-1]) is nn.Linear, f"{mode}/hidden: 최종층이 표준 nn.Linear 가 아님")
        _check(all(isinstance(layer, ApproxBackpropLinear) for layer in m.layers[:-1]),
               f"{mode}/hidden: 은닉 선형층이 근사층이 아님")
        m_all = ApproxBackpropMLP(dims, mode, 1, "all", 2)
        _check(all(isinstance(layer, ApproxBackpropLinear) for layer in m_all.layers),
               f"{mode}/all: 모든 선형층이 근사층이어야 함")
        F.cross_entropy(m(X), y).backward()
        last = len(dims) - 2
        for pname in ("weight", "bias"):
            _assert_close(f"{mode}/hidden 최종층 {pname}: 일반 역전파와 같아야 함",
                          getattr(m.layers[last], pname).grad, getattr(dense.layers[last], pname).grad, dtype)
    return "hidden 범위 확인"


def st_eval_skips_selection(device: torch.device, dtype: torch.dtype) -> str:
    gen = torch.Generator().manual_seed(51)
    m = ApproxBackpropMLP((6, 5, 4, 3), "contribution", 2, "all", 2).to(device=device, dtype=dtype)
    X = _q((4, 6), gen, device, dtype, -2, 2, 0.5)

    def calls():
        return sum(layer.selection_calls for _, layer in m.approx_layers())

    c0 = calls()
    m.eval()
    with torch.no_grad():
        m(X)
    m(X)                               # eval 모드 + 기울기 활성
    m.train()
    with torch.no_grad():
        m(X)                           # train 모드 + no_grad
    _check(calls() == c0, "평가/no_grad 순전파에서 선택 인덱스를 계산함")
    m(X)
    _check(calls() == c0 + len(m.approx_layers()), "학습 순전파에서 선택 인덱스를 계산하지 않음")
    return "평가 시 선택 생략"


def st_selection_stats(device: torch.device, dtype: torch.dtype) -> str:
    gen = torch.Generator().manual_seed(61)
    dims = (6, 5, 4, 3)
    batch = 5
    state = _quantized_state(dims, gen)
    X = _q((batch, dims[0]), gen, device, dtype, 0, 4, 0.25)
    y = torch.randint(0, dims[-1], (batch,), generator=gen).to(device)
    for mode in ("input", "contribution"):
        for k in (1, 2):
            m = ApproxBackpropMLP(dims, mode, k, "all", 2, track_selection=True).to(device=device, dtype=dtype)
            m.load_state_dict(state)
            m.train()
            inputs = m.layer_inputs(X)
            refs = {li: _reference_mask(inputs[li], layer.weight, mode, k).sum(dim=0)
                    for li, layer in m.approx_layers()}
            for _ in range(2):                         # 같은 가중치로 2회 -> 기대 횟수 2배
                F.cross_entropy(m(X), y).backward()
                m.accumulate_selection_stats()
                m.zero_grad(set_to_none=True)
            for li, layer in m.approx_layers():
                _assert_close(f"{mode} k={k} 층{li}: 선택 빈도", layer.connection_counts(), 2 * refs[li],
                              torch.float64)
    return "선택 빈도 집계 일치"


SELF_TESTS: List[Tuple[str, Callable[[torch.device, torch.dtype], str], str]] = [
    ("순전파 출력 일치 (모든 조건 vs dense)", st_forward_equivalence, "all"),
    ("선택 규칙 고정 사례 (top-1/2, 동률, 0 입력, 음수 가중치, k>입력)", st_selection_fixed_cases, "all"),
    ("선택 규칙 무작위 동률 사례 vs 파이썬 기준", st_selection_random_cases, "all"),
    ("contribution 청크 크기 불변성", st_selection_chunk_invariance, "all"),
    ("근사 dW/dX/dbias vs 마스크 수식 직접 계산", st_gradient_formula, "all"),
    ("같은 위치로 도착한 기울기 누적", st_gradient_accumulation, "all"),
    ("배치 평균 1회 적용", st_batch_mean_once, "f64"),
    ("k = 입력 차원이면 일반 역전파와 일치", st_full_k_matches_dense, "f64"),
    ("hidden 범위: 최종층은 일반 역전파", st_hidden_scope, "f64"),
    ("평가 시 선택 인덱스 계산 생략", st_eval_skips_selection, "all"),
    ("연결 선택 빈도 통계", st_selection_stats, "all"),
]


def _failure_location(exc: BaseException) -> Dict[str, str]:
    frames = traceback.extract_tb(exc.__traceback__)
    this = os.path.abspath(__file__)
    mine = [f for f in frames if os.path.abspath(f.filename) == this]

    def fmt(f) -> str:
        return f"{os.path.basename(f.filename)}:{f.lineno} ({f.name})"

    check_at = next((f for f in reversed(mine) if f.name.startswith("st_")), None)
    deepest = mine[-1] if mine else (frames[-1] if frames else None)
    return {"check_at": fmt(check_at) if check_at else "", "raised_at": fmt(deepest) if deepest else ""}


def run_self_tests(device: torch.device) -> Dict[str, Any]:
    configs: List[Tuple[torch.device, torch.dtype]] = [(torch.device("cpu"), torch.float64),
                                                        (torch.device("cpu"), torch.float32)]
    if device.type == "cuda":
        configs += [(device, torch.float64), (device, torch.float32)]
    results = []
    t0 = time.perf_counter()
    for name, fn, scope in SELF_TESTS:
        for dev, dt in configs:
            if scope == "f64" and dt != torch.float64:
                continue
            label = f"{name} [{dev.type}, {str(dt).replace('torch.', '')}]"
            try:
                detail = fn(dev, dt)
                results.append({"name": name, "device": dev.type, "dtype": str(dt), "passed": True, "detail": detail})
                log(f"  [통과] {label}: {detail}")
            except Exception as exc:  # 구현 오류도 실패로 기록하고 계속 검사
                loc = _failure_location(exc)
                results.append({"name": name, "device": dev.type, "dtype": str(dt), "passed": False,
                                "error_type": type(exc).__name__, "detail": str(exc), **loc,
                                "traceback": traceback.format_exc()})
                log(f"  [실패] {label}")
                log(f"         원인: {type(exc).__name__}: {exc}")
                log(f"         검사 위치: {loc['check_at']}  /  발생 위치: {loc['raised_at']}")
    passed = all(r["passed"] for r in results)
    return {"passed": passed, "n_checks": len(results), "n_failed": sum(not r["passed"] for r in results),
            "elapsed_s": time.perf_counter() - t0, "results": results, "created_at": now_iso(),
            "note": "근사 수식과의 일치를 검사한다. 순전파 함수의 실제 미분을 의도적으로 바꾸므로 "
                    "유한차분 gradcheck 는 요구하지 않는다."}


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def prepare_data(data_dir: Path, split_seed: int, device: torch.device, split_file: Path) -> Dict[str, Any]:
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise RuntimeError("torchvision 을 불러올 수 없습니다. README_ko.md 의 설치 방법을 따르세요.") from exc
    data_dir.mkdir(parents=True, exist_ok=True)
    log(f"  데이터 폴더: {data_dir}")
    log("  MNIST 파일이 없으면 torchvision 이 내려받습니다.")
    train_set = datasets.MNIST(root=str(data_dir), train=True, download=True)
    test_set = datasets.MNIST(root=str(data_dir), train=False, download=True)
    # 픽셀(uint8)을 그대로 보관하고 배치마다 float32 / 255 로 변환한다(0~1, 펼침).
    full_x = train_set.data.reshape(-1, 28 * 28).to(torch.uint8).contiguous()
    full_y = train_set.targets.to(torch.long)
    test_x = test_set.data.reshape(-1, 28 * 28).to(torch.uint8).contiguous()
    test_y = test_set.targets.to(torch.long)
    if full_x.shape[0] != 60000 or test_x.shape[0] != 10000:
        raise RuntimeError(f"MNIST 크기가 예상과 다릅니다: 학습 {full_x.shape[0]}, 시험 {test_x.shape[0]}")

    # 고정 분할 시드: numpy RandomState(버전 간 난수열 호환이 보장되는 구형 생성기)
    perm = np.random.RandomState(split_seed).permutation(60000).astype(np.int64)
    split_hash = hashlib.sha256(perm.tobytes()).hexdigest()
    if split_file.exists():
        saved = torch_load(split_file)
        if int(saved["split_seed"]) != int(split_seed):
            raise SettingsMismatchError(f"저장된 분할 시드 {saved['split_seed']} 와 현재 {split_seed} 가 다릅니다.")
        saved_perm = saved["perm"].numpy().astype(np.int64)
        if not np.array_equal(saved_perm, perm):
            log("  [경고] 같은 분할 시드로 다시 만든 분할이 저장본과 다릅니다(라이브러리 버전 차이 가능). "
                "저장된 분할을 그대로 사용합니다.")
            perm = saved_perm
            split_hash = hashlib.sha256(perm.tobytes()).hexdigest()
    else:
        atomic_torch_save({"split_seed": int(split_seed), "perm": torch.from_numpy(perm.copy()),
                           "sha256": split_hash, "n_train": 50000, "n_val": 10000}, split_file)
    train_idx = torch.from_numpy(perm[:50000].copy())
    val_idx = torch.from_numpy(perm[50000:].copy())
    data = {
        "train_x": full_x.index_select(0, train_idx).to(device),
        "train_y": full_y.index_select(0, train_idx).to(device),
        "val_x": full_x.index_select(0, val_idx).to(device),
        "val_y": full_y.index_select(0, val_idx).to(device),
        "test_x": test_x.to(device),
        "test_y": test_y.to(device),
        "split_sha256": split_hash,
    }
    log(f"  학습 {data['train_x'].shape[0]} / 검증 {data['val_x'].shape[0]} / 시험 {data['test_x'].shape[0]} "
        f"(분할 시드 {split_seed}, 분할 해시 {split_hash[:12]}...)")
    log("  시험 데이터는 조건별 학습이 모두 끝난 뒤 검증 기준으로 고른 체크포인트 평가에만 사용합니다.")
    return data


def epoch_permutation(seed: int, epoch: int, n: int) -> torch.Tensor:
    """(실험 시드, 에폭)만으로 정해지는 배치 순서. 조건과 무관하고 재개 시 재현된다."""
    rs = np.random.RandomState([EPOCH_ORDER_SALT, int(seed), int(epoch)])
    return torch.from_numpy(rs.permutation(n).astype(np.int64))


def num_train_batches(n: int, batch_size: int, limit: int) -> int:
    nb = math.ceil(n / batch_size)
    return min(nb, limit) if limit and limit > 0 else nb


# ---------------------------------------------------------------------------
# 환경 정보
# ---------------------------------------------------------------------------

def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        return None


def resolve_device(request: str) -> torch.device:
    if request == "cpu":
        return torch.device("cpu")
    if request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda 를 지정했지만 CUDA 를 사용할 수 없습니다. "
                               "CUDA 용 PyTorch 설치 여부를 확인하거나 --device cpu 로 실행하세요.")
        return torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def collect_environment(device: torch.device, settings: Dict[str, Any], script_path: Path) -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "collected_at": now_iso(),
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "script_version": SCRIPT_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False) if psutil else None,
        "ram_total_gb": (psutil.virtual_memory().total / 1e9) if psutil else None,
        "packages": {name: _pkg_version(name) for name in ("torch", "torchvision", "numpy", "matplotlib", "psutil")},
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "settings_device_request": settings.get("device"),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            env["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
            env["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        except Exception:
            pass
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        env["gpu"] = {
            "name": props.name,
            "index": idx,
            "count": torch.cuda.device_count(),
            "total_memory_gb": props.total_memory / 1e9,
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessors": getattr(props, "multi_processor_count", None),
        }
        try:
            env["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            env["cudnn_version"] = None
    notes = [
        "CUDA 에서 scatter_add_, bincount 등의 원자적(atomic) 누적은 부동소수점 합산 순서가 실행마다 달라질 수 있어 "
        "같은 시드라도 결과가 비트 단위로 재현되지 않을 수 있다.",
        "cuBLAS 행렬곱도 설정에 따라 비결정적일 수 있다(CUBLAS_WORKSPACE_CONFIG, --deterministic 참고).",
        "CPU 실행은 대체로 결정적이지만 스레드 수가 바뀌면 합산 순서가 달라질 수 있다.",
    ]
    if settings.get("deterministic"):
        notes.append("--deterministic: torch.use_deterministic_algorithms(True, warn_only=True). 결정적 구현이 없는 "
                     "연산은 경고만 내고 비결정적으로 실행될 수 있다.")
    env["nondeterminism_notes"] = notes
    return env


def print_environment(env: Dict[str, Any]) -> None:
    log(f"  Python {env['python'].split()[0]} / PyTorch {env['torch_version']} "
        f"(CUDA 빌드 {env['torch_cuda_build']}) / torchvision {env['packages'].get('torchvision')}")
    log(f"  운영체제 {env['platform']} / 논리 CPU {env['cpu_count_logical']} / torch 스레드 {env['torch_num_threads']}")
    if "gpu" in env:
        g = env["gpu"]
        log(f"  장치: CUDA - {g['name']} ({g['total_memory_gb']:.1f} GB, CC {g['compute_capability']})")
    else:
        log(f"  장치: CPU (CUDA 사용 가능: {env['cuda_available']})")
    for name in ("matplotlib", "psutil", "torchvision"):
        if env["packages"].get(name) is None:
            log(f"  [경고] {name} 를 찾지 못했습니다. README_ko.md 의 설치 방법을 확인하세요.")


# ---------------------------------------------------------------------------
# 평가 / 학습 / 대표 배치 세부 측정
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    run_dir: Path
    settings: Dict[str, Any]
    device: torch.device
    data: Dict[str, Any]
    progress: "Progress"
    fingerprint: str


class Progress:
    """progress.json: 조건·시드별 상태(pending/training/trained/completed/failed/interrupted)."""

    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            self.data = load_json(path)
        else:
            self.data = {"created_at": now_iso(), "stage": None, "runs": {}, "sessions": [], "init_hashes": {}}

    def save(self) -> None:
        self.data["updated_at"] = now_iso()
        atomic_write_json(self.path, self.data)

    def stage(self, name: str) -> None:
        self.data["stage"] = name
        self.save()

    def run(self, key: str, **fields: Any) -> None:
        entry = self.data["runs"].setdefault(key, {"status": "pending"})
        entry.update(fields)
        entry["updated_at"] = now_iso()
        self.save()

    def get(self, key: str) -> Dict[str, Any]:
        return self.data["runs"].get(key, {})


@torch.no_grad()
def evaluate(model: nn.Module, x_u8: torch.Tensor, y: torch.Tensor, batch_size: int,
             device: torch.device) -> Tuple[float, float]:
    """밀집 MLP 순전파로 평균 손실과 정확도를 계산한다(선택 인덱스 계산 없음)."""
    was_training = model.training
    model.eval()
    n = x_u8.shape[0]
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    for s in range(0, n, batch_size):
        xb = x_u8[s:s + batch_size].to(torch.float32).div_(255.0)
        yb = y[s:s + batch_size]
        out = model(xb)
        loss_sum += F.cross_entropy(out, yb, reduction="sum").double()
        correct += (out.argmax(dim=1) == yb).sum()
    model.train(was_training)
    return loss_sum.item() / n, correct.item() / n


def train_one_epoch(model: ApproxBackpropMLP, opt: torch.optim.Optimizer, ctx: RunContext,
                    seed: int, epoch: int) -> Dict[str, Any]:
    s = ctx.settings
    device = ctx.device
    x_all, y_all = ctx.data["train_x"], ctx.data["train_y"]
    n = x_all.shape[0]
    batch = s["batch_size"]
    n_batches = num_train_batches(n, batch, s["limit_train_batches"])
    perm = epoch_permutation(seed, epoch, n).to(device)
    approx = [layer for _, layer in model.approx_layers()] if s["selection_stats"] == "on" else []
    model.train()
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    seen = 0
    # 메모리 기준점(시간 측정 구간 밖). 학습 루프의 최대값만 보기 위해 에폭마다 초기화한다.
    cuda_base = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        cuda_base = torch.cuda.memory_allocated(device)
    rss_base = _rss_bytes()
    rss_peak = rss_base

    sync(device)
    t0 = time.perf_counter()
    for bi in range(n_batches):
        idx = perm[bi * batch:(bi + 1) * batch]
        xb = x_all.index_select(0, idx).to(torch.float32).div_(255.0)
        yb = y_all.index_select(0, idx)
        out = model(xb)
        loss = F.cross_entropy(out, yb)          # 배치 평균은 여기서 한 번만 적용
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        for layer in approx:
            layer.accumulate_selection_stats()
        with torch.no_grad():
            loss_sum += loss.detach().double() * yb.shape[0]
            correct += (out.argmax(dim=1) == yb).sum()
        seen += int(yb.shape[0])
        if rss_base is not None and bi % RSS_SAMPLE_EVERY == 0:
            r = _rss_bytes()
            if r is not None and r > rss_peak:
                rss_peak = r
    sync(device)
    train_time = time.perf_counter() - t0

    result: Dict[str, Any] = {
        "train_loss": loss_sum.item() / seen,
        "train_acc": correct.item() / seen,
        "train_time_s": train_time,
        "train_samples": seen,
        "train_batches": n_batches,
        "cuda_train_peak_allocated_mb": None,
        "cuda_train_peak_minus_baseline_mb": None,
        "cpu_rss_peak_sampled_mb": None,
        "cpu_rss_peak_minus_baseline_mb": None,
    }
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        result["cuda_train_peak_allocated_mb"] = peak / MB
        result["cuda_train_peak_minus_baseline_mb"] = (peak - cuda_base) / MB
    r = _rss_bytes()
    if rss_base is not None and r is not None:
        rss_peak = max(rss_peak, r)
        result["cpu_rss_peak_sampled_mb"] = rss_peak / MB
        result["cpu_rss_peak_minus_baseline_mb"] = (rss_peak - rss_base) / MB
    return result


def measure_saved_tensor_bytes(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor) -> Optional[int]:
    """역전파를 위해 그래프가 붙잡는 텐서 바이트(파라미터와 입력 배치 제외, 고유 텐서 기준)."""
    exclude = {p.data_ptr() for p in model.parameters()}
    exclude.add(xb.data_ptr())
    seen = set()
    total = [0]

    def pack(t: torch.Tensor) -> torch.Tensor:
        try:
            key = (t.data_ptr(), t.numel(), t.dtype)
            if t.data_ptr() not in exclude and key not in seen:
                seen.add(key)
                total[0] += t.numel() * t.element_size()
        except Exception:
            pass
        return t

    def unpack(t: torch.Tensor) -> torch.Tensor:
        return t

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            out = model(xb)
            loss = F.cross_entropy(out, yb)
        loss.backward()
    except Exception as exc:
        log(f"  [경고] 저장 텐서 크기 측정 실패: {exc}")
        return None
    finally:
        model.zero_grad(set_to_none=True)
        model.accumulate_selection_stats()   # 남은 선택 참조 정리(복사본 모델이라 통계에 영향 없음)
    return total[0]


def profile_condition(ctx: RunContext, seed: int, condition: str,
                      init_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """학습과 분리된 초기 가중치 사본으로 대표 배치의 단계별 시간을 측정한다.

    단계마다 장치 동기화(torch.cuda.synchronize) 후 perf_counter 로 벽시계 시간을 잰다.
    같은 배치를 동기화 없이 한 번 더 돌려, 세부 측정 때문에 생긴 추가 비용을 따로 기록한다.
    """
    s = ctx.settings
    device = ctx.device
    warm, n_meas = s["profile_warmup"], s["profile_batches"]
    model = build_model(condition, s).to(device)
    model.load_state_dict(init_state)
    model.train()
    opt = make_optimizer(model, s)
    track = s["selection_stats"] == "on" and bool(model.approx_layers())
    batch = s["batch_size"]
    x_all, y_all = ctx.data["train_x"], ctx.data["train_y"]
    n = x_all.shape[0]
    n_avail = num_train_batches(n, batch, s["limit_train_batches"])
    perm = epoch_permutation(seed, 0, n).to(device)

    def get_batch(i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        j = i % n_avail
        idx = perm[j * batch:(j + 1) * batch]
        return x_all.index_select(0, idx).to(torch.float32).div_(255.0), y_all.index_select(0, idx)

    def full_step(xb: torch.Tensor, yb: torch.Tensor) -> None:
        out = model(xb)
        loss = F.cross_entropy(out, yb)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if track:
            model.accumulate_selection_stats()

    t_wall0 = time.perf_counter()
    for i in range(warm):
        full_step(*get_batch(i))
    sync(device)

    # (1) 단계별 동기화 측정
    phases: Dict[str, List[float]] = {p: [] for p in PHASES}
    for i in range(warm, warm + n_meas):
        sync(device)
        t0 = time.perf_counter()
        xb, yb = get_batch(i)
        sync(device)
        t1 = time.perf_counter()
        out = model(xb)
        loss = F.cross_entropy(out, yb)
        sync(device)
        t2 = time.perf_counter()
        loss.backward()
        sync(device)
        t3 = time.perf_counter()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync(device)
        t4 = time.perf_counter()
        if track:
            model.accumulate_selection_stats()
        sync(device)
        t5 = time.perf_counter()
        phases["data"].append((t1 - t0) * 1e3)
        phases["forward_incl_selection"].append((t2 - t1) * 1e3)
        phases["backward"].append((t3 - t2) * 1e3)
        phases["optimizer"].append((t4 - t3) * 1e3)
        phases["selection_stats"].append((t5 - t4) * 1e3)
        del out, loss

    # (2) 선택 인덱스 계산만 (층 입력은 측정 밖에서 미리 계산)
    selection_only = None
    if model.approx_layers():
        sel_ms = []
        for i in range(warm, warm + n_meas):
            xb, _ = get_batch(i)
            inputs = model.layer_inputs(xb)
            sync(device)
            t0 = time.perf_counter()
            model.run_selection_only(inputs)
            sync(device)
            sel_ms.append((time.perf_counter() - t0) * 1e3)
        selection_only = _stats_ms(sel_ms)

    # (3) 같은 배치를 동기화 없이 (마지막에 한 번만 동기화)
    sync(device)
    t0 = time.perf_counter()
    for i in range(warm, warm + n_meas):
        full_step(*get_batch(i))
    sync(device)
    unsynced_ms = (time.perf_counter() - t0) * 1e3 / max(n_meas, 1)

    # (4) 한 학습 스텝의 CUDA 최대 할당 증가분
    step_peak_mb = None
    if device.type == "cuda":
        xb, yb = get_batch(warm)
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        full_step(xb, yb)
        torch.cuda.synchronize(device)
        step_peak_mb = (torch.cuda.max_memory_allocated(device) - base) / MB

    # (5) 역전파용 저장 텐서 크기
    saved_bytes = measure_saved_tensor_bytes(model, *get_batch(warm))
    estimate = model.selection_memory_estimate(batch)
    wall = time.perf_counter() - t_wall0

    phase_stats = {p: _stats_ms(v) for p, v in phases.items()}
    synced_ms = sum(phase_stats[p]["mean"] for p in PHASES if phase_stats[p]["mean"] is not None)
    del model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "enabled": True,
        "condition": condition,
        "seed": seed,
        "warmup_batches": warm,
        "measured_batches": n_meas,
        "batch_size": batch,
        "timing_method": ("단계마다 장치 동기화 후 time.perf_counter 벽시계 측정 (CUDA 는 torch.cuda.synchronize). "
                          "학습과 분리된 초기 가중치 사본, 시드별 첫 에폭 배치 순서의 앞부분 사용."),
        "phase_ms": phase_stats,
        "selection_only_ms": selection_only,
        "synced_step_ms_mean": synced_ms,
        "unsynced_step_ms_mean": unsynced_ms,
        "sync_overhead_ms_per_batch": synced_ms - unsynced_ms,
        "saved_tensor_bytes": saved_bytes,
        "cuda_step_peak_extra_mb": step_peak_mb,
        "selection_memory_estimate": estimate,
        "profile_wall_time_s": wall,
        "note": "세부 단계 시간은 대표 배치 측정값이며 전체 학습 시간(total_train_time_s)과 별개다. "
                "profile_wall_time_s 는 학습 시간에 포함되지 않는다.",
    }


def get_or_create_init_state(ctx: RunContext, seed: int) -> Dict[str, torch.Tensor]:
    """시드별 초기 가중치(모든 조건 공통). 처음 한 번 CPU 에서 만들고 파일로 고정한다."""
    path = ctx.run_dir / "init" / f"seed_{seed}.pt"
    if path.exists():
        obj = torch_load(path)
        sd = obj["state_dict"]
        if state_dict_sha256(sd) != obj.get("sha256"):
            raise RuntimeError(f"초기 가중치 파일이 손상되었습니다: {path}")
        return sd
    torch.manual_seed(seed)
    model = ApproxBackpropMLP(DIMS, None, 0)        # nn.Linear 기본 초기화, CPU 에서 생성
    sd = _clone_state(model)
    h = state_dict_sha256(sd)
    atomic_torch_save({"seed": seed, "dims": list(DIMS), "state_dict": sd, "sha256": h, "created_at": now_iso(),
                       "init": "nn.Linear 기본 초기화 (CPU 에서 torch.manual_seed(seed) 후 생성)"}, path)
    ctx.progress.data.setdefault("init_hashes", {})[str(seed)] = h
    ctx.progress.save()
    return sd


def load_checkpoint(path: Path, ctx: RunContext, seed: int, condition: str) -> Dict[str, Any]:
    ckpt = torch_load(path)
    problems = []
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        problems.append(f"체크포인트 형식 {ckpt.get('format')} != {CHECKPOINT_FORMAT}")
    if ckpt.get("condition") != condition or ckpt.get("seed") != seed:
        problems.append(f"조건/시드 불일치 ({ckpt.get('condition')}, {ckpt.get('seed')})")
    if ckpt.get("settings_fingerprint") != ctx.fingerprint:
        problems.append("실험 설정 지문(fingerprint) 불일치")
    if problems:
        raise SettingsMismatchError(f"{path} 를 이어서 쓸 수 없습니다: " + "; ".join(problems))
    return ckpt


def _new_train_state() -> Dict[str, Any]:
    return {"epochs_completed": 0, "history": [], "cum_train_time_s": 0.0, "best_epoch": None,
            "best_val_acc": None, "best_val_loss": None, "segments": 0}


def _make_checkpoint(ctx: RunContext, seed: int, condition: str, model: ApproxBackpropMLP,
                     opt: torch.optim.Optimizer, state: Dict[str, Any],
                     best_model_state: Optional[Dict[str, torch.Tensor]], training_complete: bool) -> Dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "script_version": SCRIPT_VERSION,
        "seed": seed,
        "condition": condition,
        "settings_fingerprint": ctx.fingerprint,
        "training_complete": training_complete,
        "epochs_completed": state["epochs_completed"],
        "model_state": _clone_state(model),
        "optimizer_state": opt.state_dict(),
        "best_model_state": best_model_state,
        "train_state": state,
        "selection_counts": model.selection_counts_state(),
        "rng_state": capture_rng_state(ctx.device),
        "saved_at": now_iso(),
    }


EPOCH_CSV_FIELDS = [
    "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "train_time_s", "val_time_s",
    "cum_train_time_s", "train_samples", "train_batches", "is_best_so_far", "nonfinite_train_loss",
    "cuda_train_peak_allocated_mb", "cuda_train_peak_minus_baseline_mb",
    "cpu_rss_peak_sampled_mb", "cpu_rss_peak_minus_baseline_mb",
]


def train_condition(ctx: RunContext, seed: int, condition: str, init_state: Dict[str, torch.Tensor],
                    seed_pos: Tuple[int, int], cond_pos: Tuple[int, int]) -> None:
    s = ctx.settings
    device = ctx.device
    key = run_key(seed, condition)
    cdir = cond_dir(ctx.run_dir, seed, condition)
    if (cdir / "summary.json").exists():
        log(f"  [건너뜀] {key}: 완료된 결과가 이미 있습니다.")
        if ctx.progress.get(key).get("status") != "completed":
            ctx.progress.run(key, status="completed")
        return
    cdir.mkdir(parents=True, exist_ok=True)
    last_path = cdir / "last.pt"
    ckpt = load_checkpoint(last_path, ctx, seed, condition) if last_path.exists() else None
    if ckpt is not None and ckpt.get("training_complete"):
        log(f"  [건너뜀] {key}: 학습 완료, 시험 평가 단계에서 처리합니다.")
        ctx.progress.run(key, status="trained", epochs_completed=ckpt["epochs_completed"])
        return

    mode, k = CONDITIONS[condition]
    scope_ko = "모든 선형층(최종 분류층 포함)" if s["sparse_layers"] == "all" else "은닉 선형층만"
    log("")
    log(f"=== 시드 {seed} ({seed_pos[0] + 1}/{seed_pos[1]}) | 조건 {condition} ({cond_pos[0] + 1}/{cond_pos[1]}) ===")
    log(f"    {CONDITION_DESC_KO[condition]}" + ("" if mode is None else f" | 적용 범위: {scope_ko}"))
    attempts = int(ctx.progress.get(key).get("attempts", 0)) + 1
    ctx.progress.run(key, status="training", attempts=attempts, error=None)
    set_all_seeds(seed)

    # 대표 배치 세부 측정 (학습 전, 초기 가중치 사본, 학습 시간과 분리)
    prof_path = cdir / "profile.json"
    if s["profile_batches"] > 0 and not prof_path.exists():
        log(f"    대표 배치 세부 시간 측정: 워밍업 {s['profile_warmup']} + 측정 {s['profile_batches']} 배치 "
            f"(학습과 분리된 초기 가중치 사본)")
        prof = profile_condition(ctx, seed, condition, init_state)
        atomic_write_json(prof_path, prof)
        ph = prof["phase_ms"]
        sel_txt = "" if prof["selection_only_ms"] is None else f", 선택만 {prof['selection_only_ms']['mean']:.3f}"
        log(f"    배치당 평균(ms): 순전파(선택 포함) {ph['forward_incl_selection']['mean']:.3f}{sel_txt}, "
            f"역전파 {ph['backward']['mean']:.3f}, 옵티마이저 {ph['optimizer']['mean']:.3f}, "
            f"동기화 측정 합 {prof['synced_step_ms_mean']:.3f} / 동기화 없이 {prof['unsynced_step_ms_mean']:.3f}")

    model = build_model(condition, s).to(device)
    model.load_state_dict(init_state)
    opt = make_optimizer(model, s)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        model.load_selection_counts(ckpt.get("selection_counts", {}), device)
        state = ckpt["train_state"]
        best_model_state = ckpt.get("best_model_state")
        restore_rng_state(ckpt["rng_state"], device)
        log(f"    재개: 완료된 에폭 {state['epochs_completed']} -> 에폭 {state['epochs_completed'] + 1} 부터 이어서 학습")
    else:
        state = _new_train_state()
        best_model_state = None
    state["segments"] = int(state.get("segments", 0)) + 1
    terms = backward_term_counts(model)
    if mode is not None:
        log(f"    이론적 역전파 곱셈 항 수 비율(dense 대비): {terms['terms_ratio']:.4f} "
            f"(선택 항 수 감소이며 실측 속도가 아님)")

    epochs = s["epochs"]
    ran_epochs = 0
    for epoch in range(state["epochs_completed"], epochs):
        ran_epochs += 1
        tag = (f"[시드 {seed} ({seed_pos[0] + 1}/{seed_pos[1]}) | 조건 {condition} "
               f"({cond_pos[0] + 1}/{cond_pos[1]}) | 에폭 {epoch + 1}/{epochs}]")
        log(f"{tag} 학습 중...")
        tr = train_one_epoch(model, opt, ctx, seed, epoch)
        sync(device)
        t0 = time.perf_counter()
        val_loss, val_acc = evaluate(model, ctx.data["val_x"], ctx.data["val_y"], s["eval_batch_size"], device)
        sync(device)
        val_time = time.perf_counter() - t0
        state["cum_train_time_s"] = float(state["cum_train_time_s"]) + tr["train_time_s"]
        improved = state["best_val_acc"] is None or val_acc > state["best_val_acc"]   # 동률이면 이른 에폭 유지
        if improved:
            state["best_epoch"] = epoch + 1
            state["best_val_acc"] = val_acc
            state["best_val_loss"] = val_loss
            best_model_state = _clone_state(model)
        row = {
            "epoch": epoch + 1,
            "train_loss": tr["train_loss"],
            "train_acc": tr["train_acc"],
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_time_s": tr["train_time_s"],
            "val_time_s": val_time,
            "cum_train_time_s": state["cum_train_time_s"],
            "train_samples": tr["train_samples"],
            "train_batches": tr["train_batches"],
            "is_best_so_far": improved,
            "nonfinite_train_loss": not math.isfinite(tr["train_loss"]),
            "cuda_train_peak_allocated_mb": tr["cuda_train_peak_allocated_mb"],
            "cuda_train_peak_minus_baseline_mb": tr["cuda_train_peak_minus_baseline_mb"],
            "cpu_rss_peak_sampled_mb": tr["cpu_rss_peak_sampled_mb"],
            "cpu_rss_peak_minus_baseline_mb": tr["cpu_rss_peak_minus_baseline_mb"],
        }
        state["history"].append(row)
        state["epochs_completed"] = epoch + 1
        done = state["epochs_completed"] >= epochs
        atomic_torch_save(_make_checkpoint(ctx, seed, condition, model, opt, state, best_model_state, done), last_path)
        atomic_write_csv(cdir / "epochs.csv", EPOCH_CSV_FIELDS, state["history"])
        ctx.progress.run(key, status="trained" if done else "training", epochs_completed=epoch + 1)
        log(f"{tag} 학습 손실 {tr['train_loss']:.4f} / 정확도 {tr['train_acc'] * 100:.2f}% | "
            f"검증 손실 {val_loss:.4f} / 정확도 {val_acc * 100:.2f}% | "
            f"학습 {tr['train_time_s']:.2f}초, 검증 {val_time:.2f}초, 누적 학습 {state['cum_train_time_s']:.1f}초"
            + ("  <- 최고 검증" if improved else ""))
        if row["nonfinite_train_loss"]:
            log(f"{tag} [주의] 학습 손실이 유한하지 않습니다(NaN/inf). 설정을 바꾸지 않고 기록만 합니다.")

    if ran_epochs == 0 and state["epochs_completed"] >= epochs:
        # 마지막 에폭까지 저장됐지만 완료 표시가 없던 체크포인트(정상 경로에서는 생기지 않음)
        atomic_torch_save(_make_checkpoint(ctx, seed, condition, model, opt, state, best_model_state, True), last_path)
        ctx.progress.run(key, status="trained", epochs_completed=state["epochs_completed"])
    log(f"    학습 완료: 최고 검증 정확도 {state['best_val_acc'] * 100:.2f}% (에폭 {state['best_epoch']}), "
        f"총 학습 시간 {state['cum_train_time_s']:.1f}초")
    del model, opt


def evaluate_condition(ctx: RunContext, seed: int, condition: str) -> Optional[Dict[str, Any]]:
    """검증 기준으로 고른 체크포인트를 시험 데이터에 한 번 평가하고 요약을 저장한다."""
    key = run_key(seed, condition)
    cdir = cond_dir(ctx.run_dir, seed, condition)
    summary_path = cdir / "summary.json"
    if summary_path.exists():
        return load_json(summary_path)
    last_path = cdir / "last.pt"
    if not last_path.exists():
        return None
    ckpt = load_checkpoint(last_path, ctx, seed, condition)
    if not ckpt.get("training_complete") or ckpt.get("best_model_state") is None:
        return None
    st = ckpt["train_state"]
    model = build_model(condition, ctx.settings).to(ctx.device)
    model.load_state_dict(ckpt["best_model_state"])
    sync(ctx.device)
    t0 = time.perf_counter()
    test_loss, test_acc = evaluate(model, ctx.data["test_x"], ctx.data["test_y"], ctx.settings["eval_batch_size"],
                                   ctx.device)
    sync(ctx.device)
    test_time = time.perf_counter() - t0
    atomic_torch_save({"condition": condition, "seed": seed, "epoch": st["best_epoch"],
                       "val_acc": st["best_val_acc"], "model_state": ckpt["best_model_state"]}, cdir / "best.pt")
    # 선택 빈도(누적 통계)
    model.load_selection_counts(ckpt.get("selection_counts", {}), torch.device("cpu"))
    if model.approx_layers() and ctx.settings["selection_stats"] == "on":
        arrays: Dict[str, np.ndarray] = {}
        for li, layer in model.approx_layers():
            arrays[f"layer{li}_connection_counts"] = layer.connection_counts().numpy()
            if layer.selection_mode == "input" and layer.selection_counts is not None:
                arrays[f"layer{li}_input_counts"] = layer.selection_counts.cpu().numpy()
        tmp = _tmp_path(cdir / "selection_counts.npz")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        _replace_with_retry(tmp, cdir / "selection_counts.npz")
    summary = build_run_summary(ctx, seed, condition, ckpt, model, test_loss, test_acc, test_time)
    atomic_write_json(summary_path, summary)
    ctx.progress.run(key, status="completed", test_acc=test_acc)
    log(f"  {key}: 최고 검증 에폭 {st['best_epoch']} -> 시험 정확도 {test_acc * 100:.2f}%, 시험 손실 {test_loss:.4f}")
    return summary


def selection_summary(model: ApproxBackpropMLP, samples_seen: int) -> Dict[str, Any]:
    layers = []
    tot_conn = tot_never = 0
    for li, layer in model.approx_layers():
        cc = layer.connection_counts().to(torch.float64)
        conn = cc.numel()
        never = int((cc == 0).sum().item())
        freq = cc / max(samples_seen, 1)
        layers.append({
            "layer": li, "in_features": layer.in_features, "out_features": layer.out_features,
            "selection_mode": layer.selection_mode, "k_eff": layer.k_eff,
            "connections": conn, "never_selected": never, "never_selected_fraction": never / conn,
            "selection_events": float(cc.sum().item()),
            "expected_selection_events": float(samples_seen * layer.out_features * layer.k_eff),
            "mean_selection_frequency": float(freq.mean().item()),
            "median_selection_frequency": float(freq.median().item()),
            "max_selection_frequency": float(freq.max().item()),
        })
        tot_conn += conn
        tot_never += never
    return {
        "enabled": True,
        "samples_seen": samples_seen,
        "never_selected_fraction_overall": (tot_never / tot_conn) if tot_conn else None,
        "per_layer": layers,
        "note": "selection_frequency = 연결이 선택된 표본 수 / 학습한 전체 표본 수. "
                "한 번도 선택되지 않은 연결의 가중치는 초기값에서 갱신되지 않는다(weight_decay=0 일 때).",
    }


def build_run_summary(ctx: RunContext, seed: int, condition: str, ckpt: Dict[str, Any], model: ApproxBackpropMLP,
                      test_loss: float, test_acc: float, test_time: float) -> Dict[str, Any]:
    s = ctx.settings
    st = ckpt["train_state"]
    hist = st["history"]
    mode, k = CONDITIONS[condition]
    cdir = cond_dir(ctx.run_dir, seed, condition)
    samples = int(sum(r["train_samples"] for r in hist))
    total_train = float(st["cum_train_time_s"])
    prof_path = cdir / "profile.json"
    prof = load_json(prof_path) if prof_path.exists() else {"enabled": False}

    def _max(field: str) -> Optional[float]:
        vals = [r.get(field) for r in hist if r.get(field) is not None]
        return max(vals) if vals else None

    memory = {
        "cuda_train_peak_allocated_mb": _max("cuda_train_peak_allocated_mb"),
        "cuda_train_peak_minus_baseline_mb": _max("cuda_train_peak_minus_baseline_mb"),
        "cpu_rss_peak_sampled_mb": _max("cpu_rss_peak_sampled_mb"),
        "cpu_rss_peak_minus_baseline_mb": _max("cpu_rss_peak_minus_baseline_mb"),
        "saved_tensor_bytes_per_batch": prof.get("saved_tensor_bytes"),
        "cuda_step_peak_extra_mb": prof.get("cuda_step_peak_extra_mb"),
        "selection_memory_estimate": prof.get("selection_memory_estimate"),
        "method": ("CUDA: 에폭마다 학습 루프 직전 reset_peak_memory_stats, 루프 후 max_memory_allocated "
                   "(데이터셋·모델이 포함된 기준값과 그 증가분을 함께 기록). "
                   f"CPU: psutil 프로세스 RSS 를 {RSS_SAMPLE_EVERY} 배치마다 표본 추출(짧은 순간 최대값은 놓칠 수 있고, "
                   "Python 인터프리터·라이브러리·할당자 캐시가 포함된 프로세스 전체 값이며 텐서 메모리만의 값이 아님). "
                   "saved_tensor_bytes 는 역전파용으로 그래프가 붙잡는 텐서의 정확한 바이트 수."),
    }
    if mode is not None and s["selection_stats"] == "on":
        sel = selection_summary(model, samples)
    else:
        sel = {"enabled": False, "reason": "dense 조건" if mode is None else "--selection-stats off"}
    return {
        "condition": condition,
        "description_ko": CONDITION_DESC_KO[condition],
        "seed": seed,
        "mode": mode,
        "k": k,
        "sparse_layers": s["sparse_layers"],
        "approximate_backprop": mode is not None,
        "best_epoch": st["best_epoch"],
        "best_val_acc": st["best_val_acc"],
        "best_val_loss": st["best_val_loss"],
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_eval_time_s": test_time,
        "final_val_acc": hist[-1]["val_acc"] if hist else None,
        "epochs": len(hist),
        "total_train_time_s": total_train,
        "train_samples_total": samples,
        "throughput_samples_per_s": samples / total_train if total_train > 0 else None,
        "mean_epoch_train_time_s": total_train / len(hist) if hist else None,
        "total_val_time_s": float(sum(r["val_time_s"] for r in hist)),
        "profile": prof,
        "memory": memory,
        "selection_stats": sel,
        "theoretical_backward_terms": backward_term_counts(model),
        "nonfinite_loss_epochs": [r["epoch"] for r in hist if r.get("nonfinite_train_loss")],
        "resumed_segments": st.get("segments", 1),
        "history": hist,
        "settings_fingerprint": ctx.fingerprint,
        "created_at": now_iso(),
        "notes": [
            "시험 평가는 검증 정확도 최고 에폭(동률이면 이른 에폭)의 체크포인트로 한 번만 수행했다.",
            "total_train_time_s 는 학습 루프 벽시계 시간의 합이며 검증·체크포인트 저장·세부 측정 시간은 제외된다.",
            "학습 시간에는 --selection-stats on 일 때 선택 빈도 집계 비용이 포함된다(세부 측정의 selection_stats 항목 참고).",
        ],
    }


# ---------------------------------------------------------------------------
# 요약표 / 그래프
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = [
    "condition", "seed", "mode", "k", "sparse_layers",
    "best_epoch", "best_val_acc", "best_val_loss", "test_acc", "test_loss", "final_val_acc",
    "total_train_time_s", "throughput_samples_per_s", "mean_epoch_train_time_s", "total_val_time_s",
    "train_time_ratio_vs_dense",
    "prof_data_ms", "prof_forward_incl_selection_ms", "prof_selection_only_ms", "prof_backward_ms",
    "prof_optimizer_ms", "prof_selection_stats_ms", "prof_synced_step_ms", "prof_unsynced_step_ms",
    "prof_sync_overhead_ms", "forward_ms_ratio_vs_dense", "backward_ms_ratio_vs_dense",
    "saved_tensor_mb", "cuda_step_peak_extra_mb",
    "cuda_train_peak_allocated_mb", "cuda_train_peak_minus_baseline_mb",
    "cpu_rss_peak_sampled_mb", "cpu_rss_peak_minus_baseline_mb",
    "never_selected_fraction_overall", "never_selected_fraction_L0", "never_selected_fraction_L1",
    "never_selected_fraction_L2", "backward_terms_ratio_theory", "selection_score_elements_per_sample",
]


def flatten_summary(sm: Dict[str, Any]) -> Dict[str, Any]:
    prof = sm.get("profile") or {}
    ph = prof.get("phase_ms") or {}

    def pm(name: str) -> Optional[float]:
        return (ph.get(name) or {}).get("mean")

    mem = sm.get("memory") or {}
    sel = sm.get("selection_stats") or {}
    terms = sm.get("theoretical_backward_terms") or {}
    saved = mem.get("saved_tensor_bytes_per_batch")
    row: Dict[str, Any] = {
        "condition": sm["condition"], "seed": sm["seed"], "mode": sm.get("mode") or "dense", "k": sm.get("k"),
        "sparse_layers": sm.get("sparse_layers"),
        "best_epoch": sm.get("best_epoch"), "best_val_acc": sm.get("best_val_acc"),
        "best_val_loss": sm.get("best_val_loss"), "test_acc": sm.get("test_acc"), "test_loss": sm.get("test_loss"),
        "final_val_acc": sm.get("final_val_acc"), "total_train_time_s": sm.get("total_train_time_s"),
        "throughput_samples_per_s": sm.get("throughput_samples_per_s"),
        "mean_epoch_train_time_s": sm.get("mean_epoch_train_time_s"), "total_val_time_s": sm.get("total_val_time_s"),
        "prof_data_ms": pm("data"), "prof_forward_incl_selection_ms": pm("forward_incl_selection"),
        "prof_selection_only_ms": (prof.get("selection_only_ms") or {}).get("mean"),
        "prof_backward_ms": pm("backward"), "prof_optimizer_ms": pm("optimizer"),
        "prof_selection_stats_ms": pm("selection_stats"), "prof_synced_step_ms": prof.get("synced_step_ms_mean"),
        "prof_unsynced_step_ms": prof.get("unsynced_step_ms_mean"),
        "prof_sync_overhead_ms": prof.get("sync_overhead_ms_per_batch"),
        "saved_tensor_mb": (saved / MB) if saved is not None else None,
        "cuda_step_peak_extra_mb": mem.get("cuda_step_peak_extra_mb"),
        "cuda_train_peak_allocated_mb": mem.get("cuda_train_peak_allocated_mb"),
        "cuda_train_peak_minus_baseline_mb": mem.get("cuda_train_peak_minus_baseline_mb"),
        "cpu_rss_peak_sampled_mb": mem.get("cpu_rss_peak_sampled_mb"),
        "cpu_rss_peak_minus_baseline_mb": mem.get("cpu_rss_peak_minus_baseline_mb"),
        "never_selected_fraction_overall": sel.get("never_selected_fraction_overall") if sel.get("enabled") else None,
        "backward_terms_ratio_theory": terms.get("terms_ratio"),
        "selection_score_elements_per_sample": terms.get("selection_score_elements_per_sample"),
    }
    for layer in (sel.get("per_layer") or []) if sel.get("enabled") else []:
        row[f"never_selected_fraction_L{layer['layer']}"] = layer["never_selected_fraction"]
    return row


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _ratio(a: Any, b: Any) -> Optional[float]:
    return float(a) / float(b) if _is_num(a) and _is_num(b) and float(b) != 0 else None


def _target_hit(history: List[Dict[str, Any]], target: float) -> Optional[Dict[str, Any]]:
    for r in history:
        if r["val_acc"] >= target:
            return {"epoch": r["epoch"], "cum_train_time_s": r["cum_train_time_s"], "val_acc": r["val_acc"]}
    return None


def build_all_summaries(ctx: RunContext, environments: List[Dict[str, Any]]) -> Optional[Path]:
    s = ctx.settings
    sdir = ctx.run_dir / "summary"
    sdir.mkdir(parents=True, exist_ok=True)
    per_run: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for seed in s["seeds"]:
        for cond in s["conditions"]:
            p = cond_dir(ctx.run_dir, seed, cond) / "summary.json"
            if p.exists():
                per_run[(cond, seed)] = load_json(p)
    if not per_run:
        log("  요약할 완료 결과가 없습니다.")
        return None

    rows = [flatten_summary(sm) for (_, _), sm in sorted(per_run.items(), key=lambda kv: (kv[0][1], CONDITION_ORDER.index(kv[0][0])))]
    dense_by_seed = {r["seed"]: r for r in rows if r["condition"] == "dense"}
    for r in rows:
        d = dense_by_seed.get(r["seed"])
        if d is not None:
            r["train_time_ratio_vs_dense"] = _ratio(r["total_train_time_s"], d["total_train_time_s"])
            r["forward_ms_ratio_vs_dense"] = _ratio(r["prof_forward_incl_selection_ms"], d["prof_forward_incl_selection_ms"])
            r["backward_ms_ratio_vs_dense"] = _ratio(r["prof_backward_ms"], d["prof_backward_ms"])

    targets = list(s.get("target_accuracy") or [])
    target_rows = []
    for (cond, seed), sm in per_run.items():
        for t in targets:
            hit = _target_hit(sm.get("history", []), t)
            target_rows.append({
                "condition": cond, "seed": seed, "target_val_acc": t,
                "reached": "도달" if hit else "미도달",
                "first_epoch": hit["epoch"] if hit else "미도달",
                "cum_train_time_s": hit["cum_train_time_s"] if hit else "미도달",
            })
    target_rows.sort(key=lambda r: (r["target_val_acc"], CONDITION_ORDER.index(r["condition"]), r["seed"]))

    atomic_write_csv(sdir / "runs_summary.csv", SUMMARY_COLUMNS, rows)

    # 조건별 평균과 표준편차(표본 표준편차, ddof=1)
    metric_cols = [c for c in SUMMARY_COLUMNS if c not in ("condition", "seed", "mode", "sparse_layers")]
    agg_rows = []
    for cond in s["conditions"]:
        rs = [r for r in rows if r["condition"] == cond]
        if not rs:
            continue
        agg: Dict[str, Any] = {"condition": cond, "n_seeds": len(rs), "seeds": " ".join(str(r["seed"]) for r in rs)}
        for col in metric_cols:
            vals = [float(r[col]) for r in rs if _is_num(r.get(col))]
            agg[f"{col}_mean"] = float(np.mean(vals)) if vals else None
            agg[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        for t in targets:
            hits = [tr for tr in target_rows if tr["condition"] == cond and tr["target_val_acc"] == t]
            times = [tr["cum_train_time_s"] for tr in hits if tr["reached"] == "도달"]
            agg[f"target_{t:g}_reached"] = f"{len(times)}/{len(hits)}"
            agg[f"target_{t:g}_time_s_mean"] = float(np.mean(times)) if times else "미도달"
        agg_rows.append(agg)
    agg_fields = ["condition", "n_seeds", "seeds"] + [f"{c}_{x}" for c in metric_cols for x in ("mean", "std")]
    for t in targets:
        agg_fields += [f"target_{t:g}_reached", f"target_{t:g}_time_s_mean"]
    atomic_write_csv(sdir / "condition_mean_std.csv", agg_fields, agg_rows)
    if targets:
        atomic_write_csv(sdir / "target_accuracy.csv",
                         ["condition", "seed", "target_val_acc", "reached", "first_epoch", "cum_train_time_s"],
                         target_rows)

    config = load_json(ctx.run_dir / "config.json")
    atomic_write_json(sdir / "run_config_and_environment.json", {
        "config": config, "settings_used_for_summary": s, "environments": environments,
        "progress": ctx.progress.data, "created_at": now_iso(),
    })
    figs = make_plots(sdir, per_run, s["conditions"])
    print_final_table(agg_rows, targets)
    log(f"  요약 폴더: {sdir}")
    for f in ["runs_summary.csv", "condition_mean_std.csv"] + (["target_accuracy.csv"] if targets else []) \
            + ["run_config_and_environment.json"] + [p.name for p in figs]:
        log(f"    - {f}")
    return sdir


def print_final_table(agg_rows: List[Dict[str, Any]], targets: Sequence[float]) -> None:
    def ms(row, col, scale=1.0, fmt="{:.2f}"):
        m, sd = row.get(f"{col}_mean"), row.get(f"{col}_std")
        if m is None:
            return "-"
        txt = fmt.format(m * scale)
        return txt + (" ± " + fmt.format(sd * scale) if sd is not None else "")

    log("")
    log("  조건별 결과 (평균 ± 표본표준편차)")
    log("  조건 | 시드수 | 시험 정확도(%) | 총 학습 시간(초) | 학습 시간/dense(실측) | 역전파 항 비율(이론) | 미선택 연결 비율")
    for r in agg_rows:
        log(f"  {r['condition']} | {r['n_seeds']} | {ms(r, 'test_acc', 100)} | {ms(r, 'total_train_time_s', 1, '{:.1f}')} | "
            f"{ms(r, 'train_time_ratio_vs_dense', 1, '{:.3f}')} | {ms(r, 'backward_terms_ratio_theory', 1, '{:.4f}')} | "
            f"{ms(r, 'never_selected_fraction_overall', 1, '{:.3f}')}")
    for t in targets:
        parts = []
        for r in agg_rows:
            reached = r.get(f"target_{t:g}_reached")
            tm = r.get(f"target_{t:g}_time_s_mean")
            tm_txt = tm if isinstance(tm, str) else f"{tm:.1f}초"
            parts.append(f"{r['condition']} {reached} ({tm_txt})")
        log(f"  검증 정확도 {t * 100:.2f}% 첫 도달 (도달 시드/전체, 도달 시드 평균 누적 학습 시간): " + ", ".join(parts))
    log("  주의: '역전파 항 비율(이론)'은 선택한 항의 수 감소이고, '학습 시간/dense'가 실측 속도 비율이다.")


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    korean = next((c for c in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR")
                   if c in names), None)
    rc = {
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"], "savefig.facecolor": INK["surface"],
        "axes.edgecolor": INK["axis"], "axes.labelcolor": INK["secondary"], "text.color": INK["primary"],
        "xtick.color": INK["muted"], "ytick.color": INK["muted"], "axes.grid": True, "grid.color": INK["grid"],
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12,
        "axes.labelsize": 10, "legend.frameon": False, "lines.linewidth": 2.0, "axes.unicode_minus": False,
    }
    if korean:
        rc["font.family"] = korean
    plt.rcParams.update(rc)
    return plt, korean is not None


def _runs_for(per_run: Dict[Tuple[str, int], Dict[str, Any]], cond: str) -> List[Dict[str, Any]]:
    return [sm for (c, _), sm in sorted(per_run.items(), key=lambda kv: kv[0][1]) if c == cond]


def _bar_panel(ax, conds, values_by_cond, ylabel, title, fmt="{:.1f}"):
    xs, means, stds, colors, labels = [], [], [], [], []
    for i, cond in enumerate(conds):
        vals = [v for v in values_by_cond.get(cond, []) if _is_num(v)]
        if not vals:
            continue
        xs.append(i)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        colors.append(PLOT_COLORS[cond])
        labels.append(cond)
    if not xs:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color=INK["muted"])
        ax.set_title(title)
        return
    ax.bar(xs, means, yerr=stds, color=colors, width=0.62, edgecolor=INK["surface"], linewidth=2,
           error_kw={"ecolor": INK["secondary"], "elinewidth": 1, "capsize": 3})
    for x, m in zip(xs, means):
        ax.text(x, m, fmt.format(m), ha="center", va="bottom", fontsize=8, color=INK["primary"])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)


def make_plots(sdir: Path, per_run: Dict[Tuple[str, int], Dict[str, Any]], conds: Sequence[str]) -> List[Path]:
    try:
        plt, ko = _setup_matplotlib()
    except Exception as exc:
        log(f"  [경고] matplotlib 를 불러오지 못해 그래프를 건너뜁니다: {exc}")
        return []

    def T(ko_text: str, en_text: str) -> str:
        return ko_text if ko else en_text

    files: List[Path] = []

    def save(fig, name: str) -> None:
        path = sdir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        files.append(path)

    # 1) 검증 정확도 대 에폭
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        for cond in conds:
            runs = _runs_for(per_run, cond)
            curves = [[r["val_acc"] for r in sm["history"]] for sm in runs if sm.get("history")]
            if not curves:
                continue
            n_ep = min(len(c) for c in curves)
            arr = np.array([c[:n_ep] for c in curves], dtype=np.float64) * 100
            ep = np.arange(1, n_ep + 1)
            mean = arr.mean(axis=0)
            ax.plot(ep, mean, color=PLOT_COLORS[cond], marker=PLOT_MARKERS[cond], markersize=6,
                    label=f"{cond} (n={arr.shape[0]})")
            if arr.shape[0] > 1:
                sd = arr.std(axis=0, ddof=1)
                ax.fill_between(ep, mean - sd, mean + sd, color=PLOT_COLORS[cond], alpha=0.15, linewidth=0)
        ax.set_xlabel(T("에폭", "Epoch"))
        ax.set_ylabel(T("검증 정확도 (%)", "Validation accuracy (%)"))
        ax.set_title(T("검증 정확도 대 에폭 (평균 ± 표준편차)", "Validation accuracy vs epoch (mean ± std)"))
        ax.legend(fontsize=8)
        save(fig, "fig_val_acc_vs_epoch.png")
    except Exception as exc:
        log(f"  [경고] 그래프 생성 실패 (검증 정확도 대 에폭): {exc}")

    # 2) 검증 정확도 대 누적 학습 시간
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        for cond in conds:
            runs = [sm for sm in _runs_for(per_run, cond) if sm.get("history")]
            if not runs:
                continue
            for sm in runs:
                h = sm["history"]
                ax.plot([r["cum_train_time_s"] for r in h], [r["val_acc"] * 100 for r in h],
                        color=PLOT_COLORS[cond], alpha=0.3, linewidth=1)
            n_ep = min(len(sm["history"]) for sm in runs)
            t_mean = np.mean([[r["cum_train_time_s"] for r in sm["history"][:n_ep]] for sm in runs], axis=0)
            a_mean = np.mean([[r["val_acc"] * 100 for r in sm["history"][:n_ep]] for sm in runs], axis=0)
            ax.plot(t_mean, a_mean, color=PLOT_COLORS[cond], marker=PLOT_MARKERS[cond], markersize=6,
                    label=f"{cond} (n={len(runs)})")
        ax.set_xlabel(T("누적 학습 시간 (초, 검증·세부측정 제외)", "Cumulative training time (s, excl. eval/profiling)"))
        ax.set_ylabel(T("검증 정확도 (%)", "Validation accuracy (%)"))
        ax.set_title(T("검증 정확도 대 누적 학습 시간 (굵은 선: 시드 평균)",
                       "Validation accuracy vs cumulative training time (bold: seed mean)"))
        ax.legend(fontsize=8)
        save(fig, "fig_val_acc_vs_time.png")
    except Exception as exc:
        log(f"  [경고] 그래프 생성 실패 (검증 정확도 대 시간): {exc}")

    # 3) 시험 정확도 비교 (점 그림: 시드별 점 + 평균 ± 표준편차)
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ticks, labels = [], []
        for i, cond in enumerate(conds):
            vals = [sm["test_acc"] * 100 for sm in _runs_for(per_run, cond) if _is_num(sm.get("test_acc"))]
            if not vals:
                continue
            offs = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.zeros(1)
            ax.scatter(i + offs, vals, color=PLOT_COLORS[cond], marker=PLOT_MARKERS[cond], s=36, zorder=3,
                       edgecolors=INK["surface"], linewidths=1)
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ax.errorbar(i + 0.3, m, yerr=sd, fmt="_", color=INK["primary"], markersize=14, capsize=4, elinewidth=1)
            ax.text(i + 0.38, m, f"{m:.2f}", va="center", fontsize=8, color=INK["primary"])
            ticks.append(i)
            labels.append(cond)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(T("시험 정확도 (%)", "Test accuracy (%)"))
        ax.set_title(T("시험 정확도 (검증 최고 체크포인트, 점: 시드, 막대: 평균 ± 표준편차)",
                       "Test accuracy (best-val checkpoint; dots: seeds, bar: mean ± std)"))
        ax.grid(axis="x", visible=False)
        save(fig, "fig_test_acc.png")
    except Exception as exc:
        log(f"  [경고] 그래프 생성 실패 (시험 정확도): {exc}")

    # 4) 학습 시간과 메모리
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        _bar_panel(axes[0][0], conds,
                   {c: [sm.get("total_train_time_s") for sm in _runs_for(per_run, c)] for c in conds},
                   T("초", "seconds"), T("총 학습 시간 (학습 루프 합)", "Total training time (training loops)"))
        # 대표 배치 단계별 시간(누적 막대)
        ax = axes[0][1]
        bottoms = np.zeros(len(conds))
        phase_labels = {"data": T("데이터", "data"), "forward_incl_selection": T("순전파(선택 포함)", "forward (+selection)"),
                        "backward": T("역전파", "backward"), "optimizer": T("옵티마이저", "optimizer"),
                        "selection_stats": T("선택 통계", "selection stats")}
        any_phase = False
        for ph in PHASES:
            vals = []
            for cond in conds:
                v = [((sm.get("profile") or {}).get("phase_ms") or {}).get(ph, {}).get("mean")
                     for sm in _runs_for(per_run, cond)]
                v = [x for x in v if _is_num(x)]
                vals.append(float(np.mean(v)) if v else 0.0)
            vals_arr = np.array(vals)
            if vals_arr.sum() > 0:
                any_phase = True
            ax.bar(np.arange(len(conds)), vals_arr, bottom=bottoms, color=PHASE_COLORS[ph], width=0.62,
                   edgecolor=INK["surface"], linewidth=1, label=phase_labels[ph])
            bottoms = bottoms + vals_arr
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels(list(conds), rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(T("배치당 ms (동기화 측정)", "ms per batch (synchronized)"))
        ax.set_title(T("대표 배치 단계별 시간", "Per-phase time on representative batches"))
        ax.grid(axis="x", visible=False)
        if any_phase:
            ax.legend(fontsize=8)
        # 메모리
        uses_cuda = any(_is_num((sm.get("memory") or {}).get("cuda_train_peak_minus_baseline_mb"))
                        for sm in per_run.values())
        mem_key = "cuda_train_peak_minus_baseline_mb" if uses_cuda else "cpu_rss_peak_minus_baseline_mb"
        mem_title = (T("학습 중 CUDA 최대 할당 증가분", "CUDA peak allocated during training (above baseline)")
                     if uses_cuda else
                     T("학습 중 프로세스 RSS 표본 최대 증가분", "Process RSS sampled peak increase during training"))
        _bar_panel(axes[1][0], conds,
                   {c: [(sm.get("memory") or {}).get(mem_key) for sm in _runs_for(per_run, c)] for c in conds},
                   "MB", mem_title, fmt="{:.2f}")
        _bar_panel(axes[1][1], conds,
                   {c: [((sm.get("memory") or {}).get("saved_tensor_bytes_per_batch") or 0) / MB
                        if (sm.get("memory") or {}).get("saved_tensor_bytes_per_batch") is not None else None
                        for sm in _runs_for(per_run, c)] for c in conds},
                   "MB", T("역전파용 저장 텐서 (배치 1개)", "Tensors saved for backward (one batch)"), fmt="{:.3f}")
        save(fig, "fig_time_memory.png")
    except Exception as exc:
        log(f"  [경고] 그래프 생성 실패 (시간·메모리): {exc}")

    if not ko:
        log("  (한글 글꼴을 찾지 못해 그래프 문구는 영어로 표시했습니다.)")
    return files


# ---------------------------------------------------------------------------
# 명령줄 / 설정
# ---------------------------------------------------------------------------

def _positive_int(text: str) -> int:
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수가 필요합니다.")
    return v


def _nonneg_int(text: str) -> int:
    v = int(text)
    if v < 0:
        raise argparse.ArgumentTypeError("0 이상의 정수가 필요합니다.")
    return v


def build_parser() -> argparse.ArgumentParser:
    d = DEFAULTS
    p = argparse.ArgumentParser(
        description="입력 크기 기반 희소(근사) 역전파 비교 실험 (MNIST MLP 784-256-128-10)",
        argument_default=argparse.SUPPRESS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("예:\n"
                "  python sparse_backprop_experiment.py --run-all\n"
                "  python sparse_backprop_experiment.py --run-all --epochs 1 --seeds 0 --limit-train-batches 50\n"
                "  python sparse_backprop_experiment.py --resume runs_sparse_backprop/run_YYYYMMDD_HHMMSS\n"
                "  python sparse_backprop_experiment.py --self-test-only"),
    )
    m = p.add_argument_group("실행 모드 (하나 선택)")
    m.add_argument("--run-all", action="store_true",
                   help="환경 확인 -> 데이터 준비 -> 자체 검증 -> 학습 -> 최종 평가 -> 요약·그래프 (새 실행 폴더)")
    m.add_argument("--resume", metavar="실행폴더", help="중단된 실행 폴더를 이어서 진행 (완료 조건은 건너뜀)")
    m.add_argument("--self-test-only", action="store_true", help="환경 확인과 자체 검증만 실행 (데이터 불필요)")

    g = p.add_argument_group("실험 설정")
    g.add_argument("--device", choices=["auto", "cpu", "cuda"], help=f"계산 장치 (기본값: {d['device']})")
    g.add_argument("--epochs", type=_positive_int, help=f"에폭 수 (기본값: {d['epochs']})")
    g.add_argument("--batch-size", type=_positive_int, help=f"학습 배치 크기 (기본값: {d['batch_size']})")
    g.add_argument("--lr", type=float, help=f"SGD 학습률 (기본값: {d['lr']})")
    g.add_argument("--momentum", type=float, help=f"SGD momentum (기본값: {d['momentum']})")
    g.add_argument("--weight-decay", type=float, help=f"SGD weight_decay (기본값: {d['weight_decay']})")
    g.add_argument("--seeds", type=_nonneg_int, nargs="+", help=f"실험 시드 목록 (기본값: {' '.join(map(str, d['seeds']))})")
    g.add_argument("--split-seed", type=_nonneg_int, help=f"학습/검증 분할 시드, 모든 조건·시드 공통 (기본값: {d['split_seed']})")
    g.add_argument("--data-dir", help=f"MNIST 저장 폴더 (기본값: ./{d['data_dir']})")
    g.add_argument("--output-dir", help=f"실행 폴더를 만들 상위 폴더 (기본값: ./{d['output_dir']})")
    g.add_argument("--sparse-layers", choices=["all", "hidden"],
                   help=f"근사 역전파 적용 범위: all=최종 분류층 포함 모든 선형층, hidden=은닉 선형층만 (기본값: {d['sparse_layers']})")
    g.add_argument("--selection-chunk-size", type=_positive_int,
                   help=f"contribution 선택 시 한 번에 처리할 출력 노드 수 (기본값: {d['selection_chunk_size']})")
    g.add_argument("--target-accuracy", type=float, nargs="+",
                   help="검증 정확도 목표값(여러 개 가능). 0.98 또는 98 형식. 처음 도달한 누적 학습 시간을 기록")
    g.add_argument("--conditions", nargs="+", choices=CONDITION_ORDER,
                   help="실행할 조건 (기본값: 다섯 조건 모두)")
    g.add_argument("--eval-batch-size", type=_positive_int, help=f"평가 배치 크기 (기본값: {d['eval_batch_size']})")
    g.add_argument("--limit-train-batches", type=_nonneg_int,
                   help="에폭당 학습 배치 수 상한, 짧은 확인용. 0=전체 (기본값: 0)")
    g.add_argument("--profile-warmup", type=_nonneg_int, help=f"세부 측정 워밍업 배치 수 (기본값: {d['profile_warmup']})")
    g.add_argument("--profile-batches", type=_nonneg_int,
                   help=f"세부 측정 배치 수, 0이면 세부 측정 생략 (기본값: {d['profile_batches']})")
    g.add_argument("--selection-stats", choices=["on", "off"],
                   help=f"연결 선택 빈도 집계 (기본값: {d['selection_stats']}). off 는 순수 시간 비교용")
    g.add_argument("--deterministic", action="store_true",
                   help="torch.use_deterministic_algorithms(True, warn_only=True) 사용 (기본값: 사용 안 함)")
    g.add_argument("--cpu-threads", type=_nonneg_int, help="torch CPU 스레드 수, 0=PyTorch 기본값 (기본값: 0)")
    return p


def _dedupe(seq: Sequence[Any]) -> List[Any]:
    out: List[Any] = []
    for v in seq:
        if v not in out:
            out.append(v)
    return out


def normalize_options(opts: Dict[str, Any]) -> Dict[str, Any]:
    opts = dict(opts)
    if "target_accuracy" in opts:
        targets = []
        for v in opts["target_accuracy"]:
            t = float(v) / 100.0 if float(v) > 1.0 else float(v)
            if not (0.0 < t <= 1.0):
                raise ValueError(f"--target-accuracy 값이 범위를 벗어났습니다: {v}")
            targets.append(round(t, 6))
        opts["target_accuracy"] = _dedupe(targets)
    if "seeds" in opts:
        opts["seeds"] = _dedupe(opts["seeds"])
    if "conditions" in opts:
        chosen = set(opts["conditions"])
        opts["conditions"] = [c for c in CONDITION_ORDER if c in chosen]
    for key in ("data_dir", "output_dir"):
        if key in opts:
            opts[key] = str(Path(opts[key]).expanduser().resolve())
    return opts


def validate_settings(s: Dict[str, Any]) -> None:
    if not (s["lr"] > 0):
        raise ValueError("--lr 은 0 보다 커야 합니다.")
    if s["momentum"] < 0 or s["weight_decay"] < 0:
        raise ValueError("--momentum, --weight-decay 는 0 이상이어야 합니다.")
    if not s["seeds"]:
        raise ValueError("--seeds 가 비어 있습니다.")
    if any(int(x) >= 2 ** 32 for x in s["seeds"]) or int(s["split_seed"]) >= 2 ** 32:
        raise ValueError("시드는 0 이상 2^32 미만이어야 합니다.")
    if not s["conditions"]:
        raise ValueError("--conditions 가 비어 있습니다.")


def _norm(v: Any) -> Any:
    return json.loads(json.dumps(v, default=_json_default))


def merge_resume_settings(saved: Dict[str, Any], opts: Dict[str, Any]) -> Dict[str, Any]:
    diffs = []
    for key, value in opts.items():
        if key in DEFINING_KEYS and _norm(value) != _norm(saved.get(key)):
            diffs.append(f"  --{key.replace('_', '-')}: 저장된 값 {saved.get(key)!r} / 지금 지정한 값 {value!r}")
    if diffs:
        raise SettingsMismatchError(
            "[재개 오류] 저장된 실험 설정과 다른 값이 지정되었습니다. 결과를 섞지 않기 위해 중단합니다.\n"
            + "\n".join(diffs)
            + "\n  -> 같은 값으로 다시 실행하거나, 해당 옵션을 빼고 --resume 만 지정하세요(저장된 값 사용). "
              "다른 설정으로 실험하려면 --run-all 로 새 실행 폴더를 만드세요.")
    merged = dict(saved)
    merged.update({k: v for k, v in opts.items() if k not in DEFINING_KEYS})
    return merged


def settings_fingerprint(s: Dict[str, Any]) -> str:
    payload = {k: s[k] for k in DEFINING_KEYS}
    payload["device_type"] = s["device_type"]
    payload["dims"] = list(DIMS)
    return hashlib.sha256(json.dumps(_norm(payload), sort_keys=True).encode("utf-8")).hexdigest()


def apply_runtime_flags(s: Dict[str, Any]) -> None:
    if s.get("cpu_threads"):
        torch.set_num_threads(int(s["cpu_threads"]))
    if s.get("deterministic"):
        # cuBLAS 결정성 설정은 CUDA 초기화 전에 환경 변수로 지정해야 한다.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def _new_run_dir(output_dir: Path) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"run_{stamp}"
    cand = base
    i = 1
    while cand.exists():
        cand = output_dir / f"run_{stamp}_{i}"
        i += 1
    cand.mkdir(parents=True, exist_ok=False)
    return cand


def _record_failure(ctx: RunContext, key: str, exc: BaseException) -> None:
    tb = traceback.format_exc()
    ctx.progress.run(key, status="failed", error=f"{type(exc).__name__}: {exc}")
    try:
        with open(ctx.run_dir / "errors.log", "a", encoding="utf-8") as f:
            f.write(f"[{now_iso()}] {key}: {type(exc).__name__}: {exc}\n{tb}\n")
    except OSError:
        pass
    log(f"  [오류] {key}: {type(exc).__name__}: {exc}")
    log(f"         자세한 내용: {ctx.run_dir / 'errors.log'} (완료된 다른 결과는 그대로 두고 다음 조건을 진행합니다)")
    if "out of memory" in str(exc).lower() and ctx.device.type == "cuda":
        torch.cuda.empty_cache()


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console()
    parser = build_parser()
    provided = vars(parser.parse_args(argv))
    run_all = bool(provided.pop("run_all", False))
    resume = provided.pop("resume", None)
    self_test_only = bool(provided.pop("self_test_only", False))
    n_modes = int(run_all) + int(resume is not None) + int(self_test_only)
    if n_modes != 1:
        parser.print_help()
        log("")
        log("실행 모드를 하나만 지정하세요: --run-all, --resume 실행폴더, --self-test-only")
        return 3
    try:
        opts = normalize_options(provided)
    except ValueError as exc:
        log(f"[설정 오류] {exc}")
        return 3

    script_path = Path(__file__).resolve()
    config: Optional[Dict[str, Any]] = None
    if resume is not None:
        run_dir = Path(resume).expanduser().resolve()
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            log(f"[재개 오류] {cfg_path} 가 없습니다. --run-all 로 만든 실행 폴더를 지정하세요.")
            return 3
        config = load_json(cfg_path)
        try:
            settings = merge_resume_settings(config["settings"], opts)
        except SettingsMismatchError as exc:
            log(str(exc))
            return 3
    else:
        settings = _norm(DEFAULTS)
        settings.update(opts)
        for key in ("data_dir", "output_dir"):
            settings[key] = str(Path(settings[key]).expanduser().resolve())
    try:
        validate_settings(settings)
    except ValueError as exc:
        log(f"[설정 오류] {exc}")
        return 3
    apply_runtime_flags(settings)

    # [1] 환경 확인 ------------------------------------------------------------
    log("[1/6] 환경 확인")
    try:
        device = resolve_device(settings["device"])
    except RuntimeError as exc:
        log(f"  [오류] {exc}")
        return 3
    if config is not None and config["settings"].get("device_type") != device.type:
        log(f"[재개 오류] 저장된 실행은 {config['settings'].get('device_type')} 장치였지만 지금은 {device.type} 입니다. "
            "시간·메모리 비교가 섞이지 않도록 중단합니다. --device 를 원래 장치로 지정하세요.")
        return 3
    settings["device_type"] = device.type
    env = collect_environment(device, settings, script_path)
    print_environment(env)

    if self_test_only:
        log("[자체 검증] 작은 합성 입력으로 근사 역전파 구현을 검사합니다.")
        report = run_self_tests(device)
        out_dir = Path(settings["output_dir"]) / "self_tests"
        report_path = out_dir / f"self_test_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        atomic_write_json(report_path, {"environment": env, "report": report})
        log(f"  결과: {'모두 통과' if report['passed'] else str(report['n_failed']) + '건 실패'} -> {report_path}")
        return 0 if report["passed"] else 2

    if config is None:
        run_dir = _new_run_dir(Path(settings["output_dir"]))
        config = {"script_version": SCRIPT_VERSION, "script_sha256": env["script_sha256"], "created_at": now_iso(),
                  "settings": settings, "dims": list(DIMS),
                  "conditions_definition": {c: {"mode": m, "k": k, "description_ko": CONDITION_DESC_KO[c]}
                                            for c, (m, k) in CONDITIONS.items()},
                  "verification_status_of_code": "코드 작성 및 정적 검토 완료, 실행 검증 미실시 (작성자 기준)"}
        atomic_write_json(run_dir / "config.json", config)
        env_name = "environment.json"
    else:
        env_name = f"environment_resume_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _LogState.file = run_dir / "run_log.txt"
    atomic_write_json(run_dir / env_name, env)
    progress = Progress(run_dir / "progress.json")
    progress.data.setdefault("sessions", []).append(
        {"started_at": now_iso(), "mode": "resume" if resume else "run_all", "environment_file": env_name})
    progress.save()
    if resume is not None:
        log(f"  재개할 실행 폴더: {run_dir}")
        if config.get("script_sha256") != env["script_sha256"]:
            log("  [경고] 실행 폴더를 만든 뒤 스크립트 파일 내용이 바뀌었습니다. 결과 해석에 주의하세요 "
                "(environment*.json 에 해시가 기록됩니다).")
        saved_env_path = run_dir / "environment.json"
        if saved_env_path.exists():
            saved_env = load_json(saved_env_path)
            if saved_env.get("torch_version") != env["torch_version"]:
                log(f"  [경고] PyTorch 버전이 다릅니다: 처음 {saved_env.get('torch_version')} / 지금 {env['torch_version']}")
            if (saved_env.get("gpu") or {}).get("name") != (env.get("gpu") or {}).get("name"):
                log("  [경고] GPU 모델이 처음 실행과 다릅니다. 시간 비교에 주의하세요.")
    else:
        log(f"  실행 폴더: {run_dir}")
    fingerprint = settings_fingerprint(settings)

    # [2] 데이터 준비 ----------------------------------------------------------
    log("[2/6] 데이터 준비")
    progress.stage("data")
    try:
        data = prepare_data(Path(settings["data_dir"]), int(settings["split_seed"]), device,
                            run_dir / "split_indices.pt")
    except Exception as exc:
        log(f"  [오류] 데이터 준비 실패: {type(exc).__name__}: {exc}")
        log("  인터넷 연결, --data-dir 경로, torchvision 설치를 확인하세요.")
        with open(run_dir / "errors.log", "a", encoding="utf-8") as f:
            f.write(f"[{now_iso()}] data: {traceback.format_exc()}\n")
        return 4
    ctx = RunContext(run_dir=run_dir, settings=settings, device=device, data=data, progress=progress,
                     fingerprint=fingerprint)

    # [3] 자체 검증 ------------------------------------------------------------
    log("[3/6] 자체 검증 (작은 합성 입력, 실패하면 본학습을 시작하지 않음)")
    progress.stage("self_test")
    report = run_self_tests(device)
    report_name = "self_test_report.json" if resume is None else \
        f"self_test_report_resume_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    atomic_write_json(run_dir / report_name, report)
    progress.data["self_test"] = {"passed": report["passed"], "report": report_name, "at": now_iso()}
    progress.save()
    if not report["passed"]:
        log(f"  자체 검증 {report['n_failed']}건 실패 -> 본학습을 시작하지 않습니다. 상세: {run_dir / report_name}")
        return 2
    log(f"  자체 검증 {report['n_checks']}건 모두 통과 ({report['elapsed_s']:.1f}초)")

    # [4] 학습 ---------------------------------------------------------------
    seeds, conds = list(settings["seeds"]), list(settings["conditions"])
    log(f"[4/6] 학습: 조건 {len(conds)}개 x 시드 {len(seeds)}개 "
        f"(에폭 {settings['epochs']}, 배치 {settings['batch_size']}, 학습률 {settings['lr']}, SGD momentum "
        f"{settings['momentum']}, weight_decay {settings['weight_decay']}, 적용 범위 {settings['sparse_layers']})")
    progress.stage("train")
    for key in (run_key(sd, c) for sd in seeds for c in conds):
        if key not in progress.data["runs"]:
            progress.data["runs"][key] = {"status": "pending"}
    progress.save()
    for si, seed in enumerate(seeds):
        try:
            init_state = get_or_create_init_state(ctx, seed)
        except Exception as exc:
            for cond in conds:
                _record_failure(ctx, run_key(seed, cond), exc)
            continue
        for ci, cond in enumerate(conds):
            key = run_key(seed, cond)
            try:
                train_condition(ctx, seed, cond, init_state, (si, len(seeds)), (ci, len(conds)))
            except KeyboardInterrupt:
                progress.run(key, status="interrupted")
                log("")
                log("[중단] 사용자가 중단했습니다. 마지막으로 완료된 에폭까지 저장되어 있습니다. 이어서 하려면:")
                log(f'  python "{script_path.name}" --resume "{run_dir}"')
                return 130
            except Exception as exc:
                _record_failure(ctx, key, exc)
            finally:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # [5] 최종 평가 ------------------------------------------------------------
    log("[5/6] 최종 평가: 조건별로 검증 정확도 최고 체크포인트를 시험 데이터에 한 번 평가")
    progress.stage("test")
    for seed in seeds:
        for cond in conds:
            key = run_key(seed, cond)
            try:
                result = evaluate_condition(ctx, seed, cond)
                if result is None:
                    log(f"  {key}: 학습이 완료되지 않아 시험 평가를 건너뜁니다.")
            except KeyboardInterrupt:
                log(f'[중단] 이어서 하려면: python "{script_path.name}" --resume "{run_dir}"')
                return 130
            except Exception as exc:
                _record_failure(ctx, key, exc)

    # [6] 요약 ---------------------------------------------------------------
    log("[6/6] 요약표·그래프 생성")
    progress.stage("summary")
    environments = []
    for p in sorted(run_dir.glob("environment*.json")):
        try:
            environments.append({"file": p.name, **load_json(p)})
        except Exception:
            pass
    try:
        build_all_summaries(ctx, environments)
    except Exception as exc:
        log(f"  [오류] 요약 생성 실패: {type(exc).__name__}: {exc}")
        with open(run_dir / "errors.log", "a", encoding="utf-8") as f:
            f.write(f"[{now_iso()}] summary: {traceback.format_exc()}\n")

    statuses = {k: v.get("status") for k, v in progress.data["runs"].items()
                if k in {run_key(sd, c) for sd in seeds for c in conds}}
    not_done = sorted(k for k, v in statuses.items() if v != "completed")
    progress.stage("done" if not not_done else "done_with_failures")
    log("")
    if not_done:
        log(f"완료되지 않은 조건 {len(not_done)}개: " + ", ".join(f"{k}({statuses[k]})" for k in not_done))
        log(f'원인은 {run_dir / "errors.log"} 를 확인하고, 다음 명령으로 이어서 진행할 수 있습니다:')
        log(f'  python "{script_path.name}" --resume "{run_dir}"')
        return 1
    log(f"모든 조건 완료. 결과: {run_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 중단했습니다. --resume 실행폴더 로 이어서 진행할 수 있습니다.", flush=True)
        sys.exit(130)
