"""visualization.py -- 그림 생성 (명세 [15]).

그림 제목에는 **학습 전/후, 자유/유도, 이미지 ID, 시드**를 명시한다.
교사 목표를 표시하는 분석 모드에서도 목표가 학생의 자유 추론에 들어가지 않는다
(목표는 그림에만 쓰이고 run_trial 인자로 전달되지 않는다).

서로 다른 시점의 행렬과 발화 결과를 한 상태처럼 보여주지 않기 위해
3x3 상태 그림에는 반드시 해당 관찰 tick 을 함께 적는다.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

plt.rcParams["font.family"] = ["DejaVu Sans", "Unifont"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_input_and_sampling(circuit, image: np.ndarray, path: str,
                            title_extra: str = "") -> str:
    """입력 이미지 + 유효 시야 원 + 로그-극좌표 표본과 수용장 크기."""
    g = circuit.grid
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    ax = axes[0]
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.add_patch(Circle((g.cx, g.cy), g.R, fill=False, color="tab:red", lw=1.4))
    ax.set_title(f"입력 이미지 + 유효 시야(내접 원)\n{title_extra}", fontsize=9)
    ax.set_xlabel("바깥 모서리는 시야에 포함되지 않음", fontsize=8)

    ax = axes[1]
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for i in range(g.n_points):
        ax.add_patch(Circle((g.x[i], g.y[i]), g.rf_radius[i], fill=False,
                            color="tab:cyan", lw=0.4, alpha=0.7))
    ax.scatter(g.x, g.y, s=5, c="tab:orange")
    ax.add_patch(Circle((g.cx, g.cy), g.R, fill=False, color="tab:red", lw=1.0))
    ax.set_title("원본 영상 위의 로그-극좌표 표본과 모형 수용장 반경\n"
                 "(수용장 반경 = max(2, 2*sigma_pool), 설계값)", fontsize=9)

    ch = circuit.retina.encode(image, input_colorspace=circuit.config["image"]["colorspace"])
    ax = axes[2]
    dog = ch["on"] - ch["off"]
    lim = float(np.abs(dog).max()) or 1.0
    im = ax.imshow(dog, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.add_patch(Circle((g.cx, g.cy), g.R, fill=False, color="k", lw=1.0))
    ax.set_title("DoG (ON - OFF) 중심-주변 반응", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    return _save(fig, path)


def plot_activity_maps(circuit, activity: np.ndarray, path: str,
                       targets: np.ndarray | None = None,
                       title: str = "", analysis_mode: bool = False) -> str:
    """L2 방향별 활동 지도 (+ 분석 모드에서 교사 목표와 불일치 지도).

    Parameters
    ----------
    activity : (M,) 0/1  -- 자유 추론 결과
    targets : (M,) 0/1 | None -- 분석 모드에서만 사용. 학생 추론에는 쓰이지 않는다.
    """
    g = circuit.grid
    oris = sorted(set(float(o) for o in circuit.l2_orientation.tolist()))
    ncol = len(oris) * (3 if (analysis_mode and targets is not None) else 1)
    fig, axes = plt.subplots(1, ncol, figsize=(4.0 * ncol, 4.3), squeeze=False)
    axes = axes[0]
    col = 0
    for o in oris:
        m = circuit.l2_orientation == o
        pts = circuit.l2_point[m]
        act = np.asarray(activity)[m]
        ax = axes[col]; col += 1
        ax.scatter(g.x[pts], g.y[pts], c=act, cmap="viridis", vmin=0, vmax=1, s=38)
        ax.add_patch(Circle((g.cx, g.cy), g.R, fill=False, color="tab:red", lw=1.0))
        ax.set_title(f"L2 활동 (선호 {int(o)}°) 발화수={int(act.sum())}", fontsize=9)
        ax.invert_yaxis(); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        if analysis_mode and targets is not None:
            tg = np.asarray(targets)[m]
            ax = axes[col]; col += 1
            ax.scatter(g.x[pts], g.y[pts], c=tg, cmap="viridis", vmin=0, vmax=1, s=38)
            ax.add_patch(Circle((g.cx, g.cy), g.R, fill=False, color="tab:red", lw=1.0))
            ax.set_title(f"교사 목표 (선호 {int(o)}°) 양성={int(tg.sum())}\n"
                         "[분석 모드 표시용, 학생 추론에 미사용]", fontsize=8)
            ax.invert_yaxis(); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax = axes[col]; col += 1
            mism = act.astype(int) - tg.astype(int)   # +1 과다발화, -1 미발화
            ax.scatter(g.x[pts], g.y[pts], c=mism, cmap="bwr", vmin=-1, vmax=1, s=38)
            ax.add_patch(Circle((g.cx, g.cy), g.R, fill=False, color="k", lw=1.0))
            ax.set_title(f"불일치 (빨강 +1 과다발화 / 파랑 -1 미발화)\nFP={int((mism>0).sum())} FN={int((mism<0).sum())}",
                         fontsize=8)
            ax.invert_yaxis(); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


def plot_neuron_state_and_contributions(circuit, inspect_info: dict[str, Any],
                                        path: str, title: str = "") -> str:
    """선택 뉴런의 3x3 상태 (해당 관찰 tick 명시) + 흥분/억제 기여 막대그래프."""
    obs = inspect_info["observed"]
    nid = obs["neuron_id"]
    tick = obs["tick"]
    M = circuit.population.neurons[nid].M
    # 기록된 그 시점 값으로 3x3 을 재구성해 표시한다 (현재 행렬 상태가 아님)
    shown = np.array([
        [M[0, 0], M[0, 1], M[0, 2]],
        [obs["u"], obs["threshold"], M[1, 2]],
        [obs["E"], obs["I"], obs["b_base"] + obs["correction_input"]],
    ])
    labels = [["x", "y", "z"], ["u", "threshold", "P"], ["E", "I", "b"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.imshow(np.zeros((3, 3)), cmap="Greys", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{labels[i][j]}\n{shown[i, j]:.4g}", ha="center", va="center",
                    fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{obs['neuron_name']} 의 3x3 상태 @ tick {tick} (phase={obs['phase']})\n"
                 f"q={obs['q']}  weight_version={obs['weight_version']}", fontsize=9)

    ax = axes[1]
    top = obs["top_contributions"]
    if top:
        names = [f"{c['pre_name']}\n(c{c['connection_id']})" for c in top]
        vals = [c["signed_contribution"] for c in top]
        colors = ["tab:red" if c["kind"] == "excitatory" else "tab:blue" for c in top]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, fontsize=6, rotation=60, ha="right")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("부호 있는 기여 (weight * received_value)")
    ax.set_title(f"도착 입력 기여 상위 {len(top)}개 @ tick {tick}\n"
                 f"E={obs['E']:.4g}  I={obs['I']:.4g}  b={obs['b_base']+obs['correction_input']:.4g}  "
                 f"u={obs['u']:.4g}  thr={obs['threshold']:.4g}", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


def plot_learning_curve(histories: dict[str, list[dict[str, Any]]], path: str,
                        title: str = "") -> str:
    """조건별 dev balanced accuracy 추이."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for cond, hist in histories.items():
        xs = [h["epoch"] for h in hist if h.get("dev_balanced_accuracy") is not None]
        ys = [h["dev_balanced_accuracy"] for h in hist if h.get("dev_balanced_accuracy") is not None]
        if xs:
            ax.plot(xs, ys, marker="o", label=cond)
    ax.axhline(0.5, color="gray", ls="--", lw=0.9, label="우연 수준 0.5")
    ax.set_xlabel("epoch"); ax.set_ylabel("dev balanced accuracy")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_condition_summary(summary: dict[str, dict[str, Any]], path: str,
                           title: str = "") -> str:
    """조건별 test balanced accuracy / F1 막대그래프 (시드 평균 +- 표준편차)."""
    conds = list(summary.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, lab in ((axes[0], "balanced_accuracy", "test balanced accuracy"),
                         (axes[1], "f1", "test F1")):
        means = [summary[c][key]["mean"] for c in conds]
        stds = [summary[c][key]["std"] or 0.0 for c in conds]
        ax.bar(range(len(conds)), means, yerr=stds, capsize=4,
               color=["tab:gray", "tab:orange", "tab:green", "tab:purple"][: len(conds)])
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(lab)
        if key == "balanced_accuracy":
            ax.axhline(0.5, color="k", ls="--", lw=0.9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, path)


def plot_orientation_preference(circuit, measured: dict[str, Any], path: str,
                                title: str = "") -> str:
    """측정된 방향 선호 vs 메타데이터 preferred_orientation (별도로 보고)."""
    g = circuit.grid
    meas = np.asarray(measured["measured_preferred_orientation"], dtype=np.float64)
    meta = circuit.l2_orientation
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for ax, vals, lab in ((axes[0], meta, "메타데이터 preferred_orientation"),
                          (axes[1], meas, "측정된 방향 선호 (발화율 argmax)")):
        sc = ax.scatter(g.x[circuit.l2_point], g.y[circuit.l2_point], c=vals,
                        cmap="twilight", vmin=0, vmax=90, s=26)
        ax.add_patch(Circle((g.cx, g.cy), g.R, fill=False, color="tab:red", lw=1.0))
        ax.set_title(lab, fontsize=9); ax.invert_yaxis(); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046)
    fig.suptitle(title + f"  (측정 불가 채널 {measured['n_ambiguous']}개)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, path)
