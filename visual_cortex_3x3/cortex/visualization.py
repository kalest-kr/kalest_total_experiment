"""visualization.py -- 저장된 기록을 읽어 그림을 만든다.

명세 13절. **시각화는 별도 명령**이며 새 실험을 실행하지 않는다.
모든 함수는 ``run_dir`` 의 파일 또는 이미 조립된 모델 객체만 읽는다.

한글 글꼴이 없는 환경에서는 제목이 네모로 보일 수 있다. 그림 제목은 한국어를
쓰되, 축 라벨은 ASCII 로 두어 최소한의 정보는 항상 읽히게 했다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.unicode_minus"] = False
# 한국어 글꼴이 있으면 쓰고, 없으면 기본 글꼴로 떨어진다 (오류를 내지 않는다).
plt.rcParams["font.family"] = ["DejaVu Sans", "NanumGothic", "Malgun Gothic",
                               "AppleGothic", "Unifont"]

MAX_CONNECTION_LINES = 400


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_positions_3d(model: Any, path: Path, n_connections: int = 200,
                      rng: np.random.Generator | None = None) -> Path:
    """영역·층·세포 유형별 3D 위치와 **표본** 연결.

    전 뉴런의 연결선을 그리지 않는다 (최대 ``MAX_CONNECTION_LINES`` 개 표본).
    """
    rng = rng or np.random.default_rng(0)
    a = model.anat.population.arrays
    ids = model.anat.ids
    fig = plt.figure(figsize=(12, 5))
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
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

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
    from .analysis import read_events, read_states

    ev = read_events(run_dir)
    st = read_states(run_dir)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    if ev:
        from .events import EVENT_TYPE_INDEX
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
    from .recording import read_metrics

    rows = [m for m in read_metrics(run_dir) if m.get("kind") == "sample"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
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
    from .analysis import read_states
    from .recording import read_metrics

    rows = [m for m in read_metrics(run_dir) if m.get("kind") == "sample"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
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
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
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


__all__ = [
    "plot_positions_3d", "plot_maps", "plot_neuron_timeline",
    "plot_area_arrival_times", "plot_tuning", "plot_activity_and_weights",
    "plot_rao", "make_all",
]
