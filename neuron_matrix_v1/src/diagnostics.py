"""diagnostics.py -- 진단 기록, 뉴런 점검, 반사실 개입.

명세 [3]-C, [11].

원칙
----
* 진단은 **측정 부작용이 없어야 한다**. 기록을 켜고 끈 조건에서 학습 결과가
  달라지면 안 된다 (T7 에서 검사).
* 관측 사실과 **원인 가설**을 출력에서 분리한다. 기여가 크다는 이유만으로
  그 연결이 오류의 유일한 원인이라고 출력하지 않는다.
* 진단용 개입 결과를 학습 함수로 전달하지 않는다.
* 과거 시행을 진단할 때는 그 시행의 가중치 버전을 복원해 **무개입 재실행이
  저장된 반응과 일치하는지 먼저 확인**한다. 복원할 수 없으면 과거 원인에 대한
  개입 결과로 보고하지 않는다.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .connections import KIND_NAMES, ConnectionTable

STATE_COLUMNS = [
    "trial_id", "tick", "phase", "neuron_id", "neuron_name",
    "external_sensory_drive", "E", "I", "b_base", "correction_input",
    "u", "threshold", "q", "weight_version",
]
CONTRIBUTION_COLUMNS = [
    "trial_id", "tick", "phase", "neuron_id", "neuron_name",
    "connection_id", "pre_id", "pre_name", "kind",
    "received_value", "weight_used", "signed_contribution", "weight_version",
]


class TraceRecorder:
    """선택 뉴런·선택 시행·최대 행 수로 제한되는 진단 기록기.

    이벤트마다 모든 뉴런의 상태를 영구 복사하지 않는다.
    국소 학습 함수는 이 기록을 읽지 않는다.
    """

    def __init__(
        self,
        population,
        table: ConnectionTable,
        select_neurons: Sequence[int] | None = None,
        select_trials: Sequence[Any] | None = None,
        max_rows: int = 20000,
        record_contributions: bool = True,
    ) -> None:
        self.pop = population
        self.table = table
        self.select_neurons = (
            np.array(sorted(int(x) for x in select_neurons), dtype=np.int64)
            if select_neurons is not None else None
        )
        self.select_trials = set(select_trials) if select_trials is not None else None
        self.max_rows = int(max_rows)
        self.record_contributions = bool(record_contributions)
        self.state_rows: list[dict[str, Any]] = []
        self.contribution_rows: list[dict[str, Any]] = []
        self._trial_id: Any = None
        self._phase: str = "free"
        self._active = False
        self.n_dropped = 0

    # --- 기록 -------------------------------------------------------------
    def begin_trial(self, trial_id: Any, phase: str = "free") -> None:
        self._trial_id = trial_id
        self._phase = phase
        self._active = self.select_trials is None or trial_id in self.select_trials

    def record_tick(self, engine, tick, s, ext, E, I, b, u, q, corr) -> None:
        """엔진이 각 tick 끝에 호출한다. 계산 결과를 바꾸지 않는다."""
        if not self._active:
            return
        ids = self.select_neurons
        if ids is None:
            return
        obs = engine._obs_tick[ids]
        sel = ids[obs == tick]
        if sel.size == 0:
            return
        wv = self.table.weight_version
        thresholds = engine.pop.store.states[:, 1, 1]
        b_base = engine.pop.store.b_base
        for nid in sel.tolist():
            if len(self.state_rows) >= self.max_rows:
                self.n_dropped += 1
                break
            self.state_rows.append({
                "trial_id": self._trial_id, "tick": int(tick), "phase": self._phase,
                "neuron_id": int(nid), "neuron_name": self.pop.neurons[nid].name,
                "external_sensory_drive": float(ext[nid]),
                "E": float(E[nid]), "I": float(I[nid]),
                "b_base": float(b_base[nid]), "correction_input": float(corr[nid]),
                "u": float(u[nid]), "threshold": float(thresholds[nid]),
                "q": int(q[nid]), "weight_version": wv,
            })
            if not self.record_contributions:
                continue
            inc = self.table.incoming(nid)
            act = inc[s[inc] != 0.0]
            for cid in act.tolist():
                if len(self.contribution_rows) >= self.max_rows:
                    self.n_dropped += 1
                    break
                kind = int(self.table.kind[cid])
                rv = float(s[cid])
                w = float(self.table.weight[cid])
                self.contribution_rows.append({
                    "trial_id": self._trial_id, "tick": int(tick), "phase": self._phase,
                    "neuron_id": int(nid), "neuron_name": self.pop.neurons[nid].name,
                    "connection_id": int(cid), "pre_id": int(self.table.pre_id[cid]),
                    "pre_name": self.pop.neurons[int(self.table.pre_id[cid])].name,
                    "kind": KIND_NAMES[kind],
                    "received_value": rv, "weight_used": w,
                    "signed_contribution": (w * rv) * (1.0 if kind == 0 else -1.0),
                    "weight_version": wv,
                })

    # --- 조회 -------------------------------------------------------------
    def inspect_neuron(self, trial_id: Any, neuron_id: int, tick: int) -> dict[str, Any]:
        """기록에서 해당 시점 상태를 재구성한다 (부작용 없음).

        Returns
        -------
        dict
            ``observed``  : 관측 사실 (도착 입력, E/I/b/u/threshold/q, 큰 기여 목록)
            ``reconstruction_check`` : 저장된 기여 합이 저장된 E/I/u 와 일치하는지
            ``hypotheses`` : 원인 **가설** 목록 (관측 사실과 분리)
        """
        st = [r for r in self.state_rows
              if r["trial_id"] == trial_id and r["neuron_id"] == int(neuron_id)
              and r["tick"] == int(tick)]
        if not st:
            return {"found": False, "reason": "해당 시행/뉴런/tick 의 기록이 없다"}
        row = st[0]
        contribs = [r for r in self.contribution_rows
                    if r["trial_id"] == trial_id and r["neuron_id"] == int(neuron_id)
                    and r["tick"] == int(tick)]
        e_sum = sum(c["weight_used"] * c["received_value"] for c in contribs if c["kind"] == "excitatory")
        i_sum = sum(c["weight_used"] * c["received_value"] for c in contribs if c["kind"] == "inhibitory")
        e_total = e_sum + row["external_sensory_drive"]
        b = row["b_base"] + row["correction_input"]
        u_rec = e_total - i_sum + b
        top = sorted(contribs, key=lambda c: abs(c["signed_contribution"]), reverse=True)[:8]

        hypotheses: list[str] = []
        if not contribs and row["external_sensory_drive"] == 0.0:
            hypotheses.append("입력 부재: 이 시점에 도착한 연결 입력과 감각 주입이 모두 없다")
        if i_sum > e_total and row["q"] == 0:
            hypotheses.append("강한 억제: 억제 합이 흥분 합보다 크다 (억제가 유일 원인이라는 증명은 아님)")
        if row["q"] == 0 and e_total < row["threshold"] and i_sum <= 0.0:
            hypotheses.append("부족한 흥분: 억제가 없는데도 흥분 합이 임계값에 못 미친다")
        if row["q"] == 0 and row["u"] > 0 and row["threshold"] > row["u"]:
            hypotheses.append("높은 임계값: u>0 이지만 threshold 를 넘지 못했다")
        if row["q"] == 1 and i_sum == 0.0 and e_total >= row["threshold"]:
            hypotheses.append("억제 없는 흥분 통과: 억제 입력이 0 인 상태로 임계값을 넘었다")

        return {
            "found": True,
            "observed": {
                "trial_id": trial_id, "neuron_id": int(neuron_id),
                "neuron_name": row["neuron_name"], "tick": int(tick),
                "phase": row["phase"], "weight_version": row["weight_version"],
                "n_arriving_connections": len(contribs),
                "external_sensory_drive": row["external_sensory_drive"],
                "E": row["E"], "I": row["I"],
                "b_base": row["b_base"], "correction_input": row["correction_input"],
                "u": row["u"], "threshold": row["threshold"], "q": row["q"],
                "top_contributions": top,
            },
            "reconstruction_check": {
                "E_from_contributions": e_total,
                "I_from_contributions": i_sum,
                "u_from_contributions": u_rec,
                "matches_recorded_E": abs(e_total - row["E"]) < 1e-9,
                "matches_recorded_I": abs(i_sum - row["I"]) < 1e-9,
                "matches_recorded_u": abs(u_rec - row["u"]) < 1e-9,
            },
            "hypotheses": hypotheses,
            "note_ko": (
                "위 'observed' 는 관측 사실이고 'hypotheses' 는 확인 가능한 상태에서 "
                "세운 원인 가설이다. 기여가 크다는 사실만으로 그 연결이 오류의 "
                "유일한 원인이라고 결론짓지 않는다."
            ),
        }

    def target_comparison(self, trial_id: Any, neuron_id: int, tick: int, target: int) -> dict[str, Any]:
        """교사 목표가 있을 때 활동 부족/과다 여부 (관측 사실만)."""
        info = self.inspect_neuron(trial_id, neuron_id, tick)
        if not info.get("found"):
            return info
        q = info["observed"]["q"]
        if target == q:
            verdict = "일치"
        elif target == 1 and q == 0:
            verdict = "활동 부족 (필요한데 발화하지 않음, d=+1)"
        else:
            verdict = "활동 과다 (불필요한데 발화함, d=-1)"
        info["target"] = int(target)
        info["activity_verdict"] = verdict
        return info

    # --- 저장 -------------------------------------------------------------
    def write_csv(self, state_path: str, contribution_path: str) -> dict[str, int]:
        with open(state_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=STATE_COLUMNS)
            w.writeheader()
            for r in self.state_rows:
                w.writerow(r)
        with open(contribution_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CONTRIBUTION_COLUMNS)
            w.writeheader()
            for r in self.contribution_rows:
                w.writerow(r)
        return {
            "n_state_rows": len(self.state_rows),
            "n_contribution_rows": len(self.contribution_rows),
            "n_dropped": self.n_dropped,
        }


# ----------------------------------------------------------------------
def counterfactual_check(
    engine,
    drive: tuple[np.ndarray, np.ndarray],
    interventions: dict[int, float],
    observe_ids: np.ndarray,
    baseline_response: np.ndarray | None = None,
    weights_of_trial: np.ndarray | None = None,
    weight_version_of_trial: int | None = None,
    trial_id: Any = "counterfactual",
) -> dict[str, Any]:
    """선택 연결의 가중치를 조금 바꿔 **모델 복사본 상태**에서 재실행한다.

    Parameters
    ----------
    engine : SimulationEngine
    drive : (ids, values)
        원래 시행과 **동일한** 감각 입력.
    interventions : {connection_id: new_weight}
    observe_ids : (M,) int64
        비교할 뉴런 ID.
    baseline_response : (M,) | None
        과거 시행에 저장된 반응. 주어지면 무개입 재실행과 일치하는지 먼저 검사한다.
    weights_of_trial : (K,) | None
        해당 시행의 가중치 버전. 주어지면 복원해서 진단한다.

    Returns
    -------
    dict
        ``replay_matches_stored`` 가 False 면 개입 결과를 과거 원인에 대한
        결론으로 보고하지 않는다.

    부작용: 없음. 호출 전 상태(가중치, 상태 행렬, 이벤트 큐, tick)를 복원한다.
    """
    snap = engine.state_snapshot()
    try:
        if weights_of_trial is not None:
            engine.table.restore_weights(weights_of_trial, weight_version_of_trial)
        base = engine.run_trial(drive, trial_id=f"{trial_id}/baseline")
        base_q = base.observed_q[observe_ids].copy()
        base_u = base.observed_u[observe_ids].copy()

        replay_ok: bool | None = None
        if baseline_response is not None:
            replay_ok = bool(np.array_equal(base_q, np.asarray(baseline_response)))

        w = engine.table.snapshot_weights()
        for cid, nw in interventions.items():
            w[int(cid)] = float(nw)
        engine.table.set_weights(w)
        inter = engine.run_trial(drive, trial_id=f"{trial_id}/intervened")
        int_q = inter.observed_q[observe_ids].copy()
        int_u = inter.observed_u[observe_ids].copy()
    finally:
        engine.restore_snapshot(snap)

    changed = np.nonzero(base_q != int_q)[0]
    return {
        "interventions": {int(k): float(v) for k, v in interventions.items()},
        "n_observed": int(observe_ids.size),
        "n_changed_firing": int(changed.size),
        "changed_indices": changed.tolist()[:50],
        "baseline_q": base_q.tolist(),
        "intervened_q": int_q.tolist(),
        "baseline_u": base_u.tolist(),
        "intervened_u": int_u.tolist(),
        "delta_u": (int_u - base_u).tolist(),
        "replay_matches_stored": replay_ok,
        "note_ko": (
            "이것은 해당 조작의 효과를 보여준다. 그 연결이 생물학적으로 잘못된 "
            "연결이라는 증명이 아니다. replay_matches_stored 가 False 이거나 "
            "None 이면 과거 시행 원인에 대한 개입 결과로 해석하지 않는다."
        ),
    }


def connection_role_table(
    engine,
    drives: Sequence[tuple[np.ndarray, np.ndarray]],
    scene_labels: Sequence[dict[str, Any]],
    connection_id: int,
    observe_id: int,
    delta_weight: float = -0.4,
) -> list[dict[str, Any]]:
    """여러 선 위치·방향에서 한 연결의 기여와 개입 효과를 비교하는 표.

    단일 가중치에 고정된 의미가 들어 있다고 가정하지 않기 위한 진단이다.
    """
    rows: list[dict[str, Any]] = []
    obs = np.array([int(observe_id)], dtype=np.int64)
    snap = engine.state_snapshot()
    try:
        w0 = float(engine.table.weight[int(connection_id)])
        obs_tick = int(engine.pop.neurons[int(observe_id)].observation_tick)
        for drive, label in zip(drives, scene_labels):
            base = engine.run_trial(drive, trial_id="role/base")
            s = base.arrivals_at(obs_tick)
            rv = float(s[int(connection_id)])
            kind = int(engine.table.kind[int(connection_id)])
            contrib = w0 * rv * (1.0 if kind == 0 else -1.0)
            cf = counterfactual_check(
                engine, drive,
                {int(connection_id): max(0.0, w0 + delta_weight)},
                obs, trial_id="role",
            )
            rows.append({
                **label,
                "connection_id": int(connection_id),
                "kind": KIND_NAMES[kind],
                "weight": w0,
                "received_value": rv,
                "signed_contribution": contrib,
                "q_baseline": int(cf["baseline_q"][0]),
                "q_intervened": int(cf["intervened_q"][0]),
                "u_baseline": float(cf["baseline_u"][0]),
                "delta_u": float(cf["delta_u"][0]),
                "firing_changed": cf["baseline_q"][0] != cf["intervened_q"][0],
            })
    finally:
        engine.restore_snapshot(snap)
    return rows


def measured_orientation_preference(
    predictions: np.ndarray,
    scene_orientations: Sequence[float | None],
) -> dict[str, Any]:
    """실제 **측정된** 방향 선호 (메타데이터 preferred_orientation 과 별도로 보고).

    Parameters
    ----------
    predictions : (n_scenes, M) 0/1
    scene_orientations : 길이 n_scenes, blank 는 None

    Returns
    -------
    dict : 방향별 발화율 (M,) 과 argmax 로 정한 측정 선호 방향 (M,)
    """
    p = np.asarray(predictions).astype(np.float64)
    oris = [o for o in scene_orientations]
    uniq = sorted({float(o) for o in oris if o is not None})
    rates = {}
    for o in uniq:
        m = np.array([x is not None and float(x) == o for x in oris], dtype=bool)
        rates[o] = p[m].mean(axis=0) if m.any() else np.full(p.shape[1], np.nan)
    stack = np.stack([rates[o] for o in uniq], axis=0)   # (n_ori, M)
    best = np.argmax(stack, axis=0)
    ambiguous = (stack.max(axis=0) - stack.min(axis=0)) <= 1e-12
    measured = np.array([uniq[b] for b in best], dtype=np.float64)
    measured[ambiguous] = np.nan
    return {
        "orientations": uniq,
        "firing_rate_by_orientation": {str(int(o)): rates[o].tolist() for o in uniq},
        "measured_preferred_orientation": measured.tolist(),
        "n_ambiguous": int(np.count_nonzero(ambiguous)),
    }
