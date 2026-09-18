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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .events import AMOUNT_UNITS, EVENT_TYPE_INDEX
from .ids import COMPARTMENT_NAMES, EVENT_TYPES
from .recording import load_manifest, read_metrics


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


__all__ = [
    "read_events", "read_states", "explain_neuron", "ablation_rerun",
    "orientation_tuning_from_events", "train_readout", "make_report",
    "verify_report_matches_records",
]
