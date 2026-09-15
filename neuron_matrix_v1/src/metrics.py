"""metrics.py -- 평가 지표. 학습 함수와 완전히 분리되어 있다.

주 지표는 L2 활동 지도에 대한 **balanced accuracy** 와 **F1** 이다.
불활성 뉴런이 많으므로 전체 비트 정확도만으로 성공을 주장하지 않는다.

양성 또는 음성이 한 종류도 없는 뉴런에서는 지표가 정의되지 않는다.
그런 경우는 ``None`` 으로 두고 개수를 따로 보고한다 (0 으로 채우지 않는다).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """TP/FP/TN/FN. 입력은 0/1 배열 (shape 자유, 평탄화)."""
    t = np.asarray(y_true).astype(bool).ravel()
    p = np.asarray(y_pred).astype(bool).ravel()
    return {
        "TP": int(np.count_nonzero(t & p)),
        "FP": int(np.count_nonzero(~t & p)),
        "TN": int(np.count_nonzero(~t & ~p)),
        "FN": int(np.count_nonzero(t & ~p)),
    }


def rates_from_confusion(c: dict[str, int]) -> dict[str, float | None]:
    TP, FP, TN, FN = c["TP"], c["FP"], c["TN"], c["FN"]
    pos = TP + FN
    neg = TN + FP
    tpr = TP / pos if pos else None          # recall / sensitivity
    tnr = TN / neg if neg else None          # specificity
    fpr = FP / neg if neg else None
    fnr = FN / pos if pos else None
    prec = TP / (TP + FP) if (TP + FP) else None
    if tpr is None or tnr is None:
        bacc = None
    else:
        bacc = 0.5 * (tpr + tnr)
    if prec is None or tpr is None or (prec + tpr) == 0:
        f1 = 0.0 if (TP == 0 and (FP + FN) > 0) else None
    else:
        f1 = 2 * prec * tpr / (prec + tpr)
    return {
        "accuracy": (TP + TN) / max(1, TP + TN + FP + FN),
        "balanced_accuracy": bacc,
        "f1": f1,
        "precision": prec,
        "recall_tpr": tpr,
        "specificity_tnr": tnr,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "n_positive": pos,
        "n_negative": neg,
    }


def evaluate_map(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """(n_scenes, M) 목표/예측 -> 전체 집계 지표."""
    c = confusion(y_true, y_pred)
    out = {"confusion": c}
    out.update(rates_from_confusion(c))
    return out


def per_neuron_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """뉴런별 지표 + 정의되지 않는 경우의 처리.

    Parameters
    ----------
    y_true, y_pred : (n_scenes, M)

    Returns
    -------
    dict
        ``balanced_accuracy`` : (M,) float, 정의 불가면 np.nan
        ``n_positive`` : (M,) int  (뉴런별 양성 표본 수)
        ``activity_rate`` : (M,) float (예측 발화율)
        ``n_undefined`` : int
        ``n_no_positive_target`` : int  (학습되지 않은/미관측 채널 후보)
        ``n_never_active`` : int
    """
    t = np.asarray(y_true).astype(bool)
    p = np.asarray(y_pred).astype(bool)
    n_scenes, M = t.shape
    TP = np.count_nonzero(t & p, axis=0)
    FP = np.count_nonzero(~t & p, axis=0)
    TN = np.count_nonzero(~t & ~p, axis=0)
    FN = np.count_nonzero(t & ~p, axis=0)
    pos = TP + FN
    neg = TN + FP
    with np.errstate(invalid="ignore", divide="ignore"):
        tpr = np.where(pos > 0, TP / np.maximum(pos, 1), np.nan)
        tnr = np.where(neg > 0, TN / np.maximum(neg, 1), np.nan)
    bacc = 0.5 * (tpr + tnr)
    undefined = ~np.isfinite(bacc)
    return {
        "balanced_accuracy": bacc,
        "n_positive": pos.astype(int),
        "n_negative": neg.astype(int),
        "activity_rate": p.mean(axis=0),
        "n_undefined": int(np.count_nonzero(undefined)),
        "n_no_positive_target": int(np.count_nonzero(pos == 0)),
        "n_never_active": int(np.count_nonzero(p.sum(axis=0) == 0)),
        "n_always_active": int(np.count_nonzero(p.sum(axis=0) == n_scenes)),
        "mean_defined_balanced_accuracy": (
            float(np.nanmean(bacc)) if np.isfinite(bacc).any() else None
        ),
        "note_ko": (
            "양성 또는 음성이 없는 뉴런은 balanced accuracy 가 정의되지 않아 NaN 으로 두었다. "
            "0 으로 채우지 않는다."
        ),
    }


def baseline_metrics(y_true: np.ndarray) -> dict[str, Any]:
    """항상 비활성 / 항상 발화 예측의 기준값."""
    t = np.asarray(y_true)
    return {
        "always_inactive": evaluate_map(t, np.zeros_like(t)),
        "always_active": evaluate_map(t, np.ones_like(t)),
        "positive_rate": float(np.asarray(t).astype(bool).mean()),
    }


def blank_false_firing(
    y_pred: np.ndarray, is_blank: Sequence[bool]
) -> dict[str, Any]:
    """빈 화면에서의 불필요한 발화율."""
    p = np.asarray(y_pred).astype(bool)
    mask = np.asarray(list(is_blank), dtype=bool)
    if not mask.any():
        return {"n_blank_scenes": 0, "blank_false_firing_rate": None}
    sub = p[mask]
    return {
        "n_blank_scenes": int(mask.sum()),
        "blank_false_firing_rate": float(sub.mean()),
        "blank_scenes_with_any_firing": int(np.count_nonzero(sub.any(axis=1))),
        "mean_firing_neurons_per_blank": float(sub.sum(axis=1).mean()),
    }


def orientation_position_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    orientation_deg: np.ndarray,
    radial_bin: np.ndarray,
) -> dict[str, Any]:
    """위치(반경 bin)·방향별 반응과 오답 지도."""
    out: dict[str, Any] = {}
    ori_vals = sorted(set(float(o) for o in orientation_deg.tolist()))
    for o in ori_vals:
        m = orientation_deg == o
        out[f"orientation_{int(o)}"] = evaluate_map(y_true[:, m], y_pred[:, m])
    bins = sorted(set(int(k) for k in radial_bin.tolist()))
    by_bin = {}
    for k in bins:
        m = radial_bin == k
        by_bin[f"radial_bin_{k}"] = evaluate_map(y_true[:, m], y_pred[:, m])
    out["by_radial_bin"] = by_bin
    by_cell = {}
    for o in ori_vals:
        for k in bins:
            m = (orientation_deg == o) & (radial_bin == k)
            if not m.any():
                continue
            by_cell[f"ori{int(o)}_k{k}"] = evaluate_map(y_true[:, m], y_pred[:, m])
    out["by_orientation_and_radial_bin"] = by_cell
    return out


def paired_difference(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """시드별 짝지은 차이 (a - b) 의 평균/표준편차."""
    x = np.asarray([v for v in a], dtype=np.float64)
    y = np.asarray([v for v in b], dtype=np.float64)
    d = x - y
    return {
        "n": int(d.size),
        "mean_difference": float(d.mean()) if d.size else None,
        "std_difference": float(d.std(ddof=1)) if d.size > 1 else None,
        "per_seed_difference": d.tolist(),
    }


def summarize_seeds(values: Sequence[float | None]) -> dict[str, Any]:
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    return {
        "n": int(v.size),
        "mean": float(v.mean()) if v.size else None,
        "std": float(v.std(ddof=1)) if v.size > 1 else None,
        "min": float(v.min()) if v.size else None,
        "max": float(v.max()) if v.size else None,
        "values": [None if x is None else float(x) for x in values],
    }
